#!/usr/bin/env python3
"""Small source-specific downloads for the two public target datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

JSON_LIMIT = 4 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
DATA_ROOT = Path("/projects/EEG-foundation-model")


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("dataset config is malformed")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,application/vnd.api+json,application/vnd.mendeley-public-dataset.1+json",
            "User-Agent": "denoiseNet-private-research/1",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(JSON_LIMIT + 1)
    if len(body) > JSON_LIMIT:
        raise ValueError(f"metadata response exceeds {JSON_LIMIT} bytes")
    return json.loads(body)


def require_https(url: str, suffix: str) -> None:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (hostname == suffix or hostname.endswith("." + suffix)):
        raise ValueError(f"unexpected download URL host: {hostname}")


def safe_component(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe path component: {name!r}")
    return name


def configured_target(item: dict[str, Any]) -> Path:
    target = Path(os.path.abspath(str(item.get("target", ""))))
    if target == DATA_ROOT or os.path.commonpath((str(DATA_ROOT), str(target))) != str(DATA_ROOT):
        raise ValueError(f"dataset target is outside the data root: {target}")
    return target


def stream_download(url: str, destination: Path, expected_bytes: int, host_suffix: str) -> None:
    require_https(url, host_suffix)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/octet-stream", "User-Agent": "denoiseNet-private-research/1"},
    )
    received = 0
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as output:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            if received > expected_bytes:
                raise ValueError(f"download exceeded declared size for {destination.name}")
    if received != expected_bytes:
        raise ValueError(
            f"download size mismatch for {destination.name}: expected {expected_bytes}, got {received}"
        )


def partial_directory(target: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    partial = target.parent / f".{target.name}.partial-{job_id}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        raise FileExistsError(f"final target already exists: {target}")
    if os.path.lexists(partial):
        raise FileExistsError(f"partial target already exists: {partial}")
    partial.mkdir()
    return partial


def download_klados(item: dict[str, Any]) -> dict[str, Any]:
    if not item.get("download_authorized"):
        raise PermissionError("Klados download is not authorized in config")
    rows = fetch_json(str(item["files_api"]))
    if not isinstance(rows, list):
        raise ValueError("unexpected Mendeley public files response")
    expected_name = str(item["expected_filename"])
    matches = [row for row in rows if isinstance(row, dict) and row.get("filename") == expected_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Mendeley file named {expected_name}")
    row = matches[0]
    details = row.get("content_details")
    if not isinstance(details, dict):
        raise ValueError("Mendeley content_details missing")
    expected_bytes = int(item["expected_bytes"])
    if int(row.get("size", -1)) != expected_bytes or int(details.get("size", -1)) != expected_bytes:
        raise ValueError("Mendeley file size changed")
    download_url = str(details["download_url"])
    target = configured_target(item)
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise FileExistsError(f"unexpected existing target: {target}")
        archive = target / expected_name
        if archive.is_file() and archive.stat().st_size == expected_bytes:
            return {
                "mode": "download-klados",
                "state": "already_present",
                "target": str(target),
                "filename": expected_name,
                "bytes": expected_bytes,
            }
        raise FileExistsError(f"unexpected existing target: {target}")
    partial = partial_directory(target)
    stream_download(download_url, partial / expected_name, expected_bytes, "mendeley.com")
    if os.path.lexists(target):
        raise FileExistsError(f"final target appeared during download: {target}")
    os.replace(partial, target)
    return {
        "mode": "download-klados",
        "state": "downloaded_archive",
        "target": str(target),
        "filename": expected_name,
        "bytes": expected_bytes,
        "file_id": str(row.get("id")),
    }


def osf_listing(item: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    node = fetch_json(str(item["node_api"]))
    license_data = node["data"]["relationships"]["license"]["data"]
    license_id = str(license_data["id"])
    if license_id != str(item["expected_license_id"]):
        raise ValueError(f"OSF license changed: {license_id}")
    license_href = node["data"]["relationships"]["license"]["links"]["related"]["href"]
    license_payload = fetch_json(str(license_href))
    license_name = str(license_payload["data"]["attributes"]["name"])
    if "Attribution 4.0" not in license_name:
        raise PermissionError(f"unexpected OSF license: {license_name}")

    providers = fetch_json(str(item["files_api"]))
    provider_rows = providers.get("data", [])
    roots = [
        row["relationships"]["files"]["links"]["related"]["href"]
        for row in provider_rows
        if row.get("attributes", {}).get("provider") == "osfstorage"
    ]
    if len(roots) != 1:
        raise ValueError("expected one OSF storage provider")

    files: list[dict[str, Any]] = []
    pending: deque[tuple[str, PurePosixPath]] = deque([(str(roots[0]), PurePosixPath())])
    api_calls = 0
    while pending:
        next_url, prefix = pending.popleft()
        while next_url:
            api_calls += 1
            if api_calls > 100:
                raise ValueError("OSF metadata traversal exceeded 100 requests")
            page = fetch_json(next_url)
            for row in page.get("data", []):
                attributes = row.get("attributes", {})
                name = safe_component(str(attributes.get("name", "")))
                relative = prefix / name
                if attributes.get("kind") == "folder":
                    related = row["relationships"]["files"]["links"]["related"]["href"]
                    pending.append((str(related), relative))
                elif attributes.get("kind") == "file":
                    files.append(
                        {
                            "relative": relative.as_posix(),
                            "bytes": int(attributes["size"]),
                            "url": str(row["links"]["download"]),
                        }
                    )
                else:
                    raise ValueError(f"unexpected OSF item kind for {relative}")
                if len(files) > 10000:
                    raise ValueError("OSF listing exceeded 10,000 files")
            next_link = page.get("links", {}).get("next")
            next_url = str(next_link) if next_link else ""
    return sorted(files, key=lambda row: row["relative"]), license_name


def download_sgeyesub(item: dict[str, Any]) -> dict[str, Any]:
    if not item.get("download_authorized"):
        raise PermissionError("SGEYESUB download is not authorized in config")
    files, license_name = osf_listing(item)
    expected_count = int(item["expected_file_count"])
    expected_bytes = int(item["expected_bytes"])
    total_bytes = sum(int(row["bytes"]) for row in files)
    if len(files) != expected_count or total_bytes != expected_bytes:
        raise ValueError(
            f"OSF listing changed: files={len(files)} bytes={total_bytes}"
        )
    if total_bytes > int(item["max_download_bytes"]):
        raise ValueError("OSF dataset exceeds configured download limit")

    target = configured_target(item)
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_dir():
            raise FileExistsError(f"unexpected existing target: {target}")
        present = all(
            (target / row["relative"]).is_file()
            and (target / row["relative"]).stat().st_size == int(row["bytes"])
            for row in files
        )
        if present:
            return {
                "mode": "download-sgeyesub",
                "state": "already_present",
                "target": str(target),
                "file_count": len(files),
                "bytes": total_bytes,
                "license": license_name,
            }
        raise FileExistsError(f"unexpected existing target: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target.parent).free
    if free_bytes < total_bytes + 5 * 1024**3:
        raise OSError("insufficient free space for OSF download safety margin")
    partial = partial_directory(target)
    for row in files:
        destination = partial / row["relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        stream_download(str(row["url"]), destination, int(row["bytes"]), "osf.io")
    if os.path.lexists(target):
        raise FileExistsError(f"final target appeared during download: {target}")
    os.replace(partial, target)
    return {
        "mode": "download-sgeyesub",
        "state": "downloaded",
        "target": str(target),
        "file_count": len(files),
        "bytes": total_bytes,
        "license": license_name,
    }


def self_test() -> dict[str, Any]:
    assert safe_component("study01_p01_prep.set") == "study01_p01_prep.set"
    for invalid in ("", ".", "..", "a/b", "a\\b", "a\x00b"):
        try:
            safe_component(invalid)
        except ValueError:
            continue
        raise AssertionError(f"unsafe path was accepted: {invalid!r}")
    assert configured_target({"target": "/projects/EEG-foundation-model/example/v1"}).is_absolute()
    try:
        configured_target({"target": "/home/infres/yinwang/denoiseNet/not-data"})
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-root target was accepted")
    return {"mode": "self-test", "state": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("self-test", "download-klados", "download-sgeyesub"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "self-test":
        result = self_test()
    elif args.mode == "download-klados":
        result = download_klados(config["datasets"]["klados_bamidis_v1"])
    else:
        result = download_sgeyesub(config["datasets"]["sgeyesub"])
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
