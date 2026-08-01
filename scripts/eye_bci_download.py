#!/usr/bin/env python3
"""Targeted Synapse downloader for the Eye-BCI Neuroscan CSV selection."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from dataset_harness import _load_synapse_token, write_json

DATA_ROOT = Path("/projects/EEG-foundation-model")
PARTIAL_ROOT = DATA_ROOT / "eye_bci" / ".syn64005218-neuroscan.partial"
FINAL_ROOT = DATA_ROOT / "eye_bci" / "syn64005218-neuroscan"
PUBLISH_LOCK = DATA_ROOT / "eye_bci" / ".syn64005218-neuroscan.publish.lock"
PATH_PATTERN = re.compile(
    r"^(S(0[1-9]|[12][0-9]|3[01]))/Sess0([1-3])/Neuroscan/"
    r"(ME|MI|P3004L|P3005L|SSVEP)\2\3\.csv$"
)
CONTENT_RANGE_PATTERN = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request: Any, *args: Any, **kwargs: Any) -> None:
        return None


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("mode") != "inventory-eye-bci-neuroscan"
        or payload.get("state") != "inventoried"
        or not isinstance(payload.get("files"), list)
        or payload.get("file_count") != len(payload["files"])
    ):
        raise ValueError("invalid Eye-BCI Neuroscan manifest")
    observed_total = 0
    seen_paths: set[str] = set()
    for item in payload["files"]:
        if not isinstance(item, dict) or not PATH_PATTERN.fullmatch(str(item.get("path", ""))):
            raise ValueError("manifest contains a path outside the Neuroscan selection")
        if str(item["path"]) in seen_paths:
            raise ValueError("manifest contains duplicate paths")
        seen_paths.add(str(item["path"]))
        if not str(item.get("entity_id", "")).startswith("syn"):
            raise ValueError("manifest contains an invalid entity ID")
        if not str(item.get("file_handle_id", "")).isdigit():
            raise ValueError("manifest contains an invalid file handle ID")
        size = int(item.get("bytes", 0))
        if size <= 0:
            raise ValueError("manifest contains an invalid file size")
        observed_total += size
    if observed_total != int(payload.get("total_bytes", -1)):
        raise ValueError("manifest byte total is inconsistent")
    subjects = {path.split("/", 1)[0] for path in seen_paths}
    expected_subjects = {f"S{index:02d}" for index in range(1, 32)}
    if subjects != expected_subjects:
        raise ValueError("manifest subject set is not exactly S01-S31")
    return payload


def ensure_real_directory(path: Path) -> None:
    if path == DATA_ROOT:
        if path.resolve(strict=True) != DATA_ROOT:
            raise ValueError("data root does not resolve to the registered path")
        return
    relative = path.relative_to(DATA_ROOT)
    current = DATA_ROOT
    ensure_real_directory(DATA_ROOT)
    for component in relative.parts:
        current = current / component
        try:
            current.mkdir()
        except FileExistsError:
            pass
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"unsafe data directory component: {current}")


def signed_download_url(token: str, item: dict[str, Any]) -> str:
    query = urlencode(
        {
            "redirect": "false",
            "fileAssociateType": "FileEntity",
            "fileAssociateId": str(item["entity_id"]),
        }
    )
    request = urllib.request.Request(
        "https://repo-prod.prod.sagebase.org/file/v1/file/"
        f"{item['file_handle_id']}?{query}",
        headers={
            "Accept": "text/plain",
            "Authorization": f"Bearer {token}",
            "User-Agent": "denoiseNet-private-research/1",
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirectHandler()
    )
    try:
        with opener.open(request, timeout=30) as response:
            if int(response.status) != 200:
                raise RuntimeError("Synapse did not return a signed URL")
            value = response.read(8193).decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Synapse signed-URL request failed with HTTP {exc.code}"
        ) from None
    parsed = urlsplit(value)
    if len(value) > 8192 or parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Synapse returned an invalid signed URL")
    return value


def download_once(token: str, item: dict[str, Any], target: Path) -> tuple[int, bool]:
    expected = int(item["bytes"])
    temporary = target.with_name(target.name + ".partial")
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file() or target.stat().st_size != expected:
            raise ValueError(f"unexpected existing target: {target}")
        return 0, True
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError(f"unsafe partial file: {temporary}")
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected:
        raise ValueError(f"partial file exceeds expected size: {temporary}")
    if offset == expected:
        os.replace(temporary, target)
        return 0, False

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "denoiseNet-private-research/1",
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(signed_download_url(token, item), headers=headers)
    object_opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), NoRedirectHandler()
    )
    with object_opener.open(request, timeout=60) as response:
        status_code = int(getattr(response, "status", response.getcode()))
        resumed = offset > 0 and status_code == 206
        if offset and status_code not in (200, 206):
            raise IOError(f"unexpected resume status {status_code} for {item['entity_id']}")
        if not offset and status_code != 200:
            raise IOError(f"unexpected initial status {status_code} for {item['entity_id']}")
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            match = CONTENT_RANGE_PATTERN.fullmatch(content_range)
            if (
                match is None
                or int(match.group(1)) != offset
                or int(match.group(3)) != expected
            ):
                raise IOError(f"invalid Content-Range for {item['entity_id']}")
        elif offset:
            offset = 0
        content_length = response.headers.get("Content-Length")
        expected_response_bytes = expected - offset
        if content_length is not None and int(content_length) != expected_response_bytes:
            raise IOError(f"unexpected Content-Length for {item['entity_id']}")
        mode = "ab" if resumed else "wb"
        written = 0
        with temporary.open(mode) as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                written += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if temporary.stat().st_size != expected:
        raise IOError(
            f"incomplete download for {item['entity_id']}: "
            f"{temporary.stat().st_size} of {expected} bytes"
        )
    os.replace(temporary, target)
    return written, False


def download_with_retries(token: str, item: dict[str, Any], target: Path) -> tuple[int, bool]:
    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            return download_once(token, item, target)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(5 * attempt)
    raise RuntimeError(
        f"download failed after three attempts for {item['entity_id']}: "
        f"{type(last_error).__name__}"
    ) from None


def download_subject(manifest: dict[str, Any], array_index: int) -> dict[str, Any]:
    subjects = sorted({str(item["path"]).split("/", 1)[0] for item in manifest["files"]})
    if array_index < 0 or array_index >= len(subjects):
        raise ValueError("array index is outside the subject manifest")
    subject = subjects[array_index]
    selected = [item for item in manifest["files"] if str(item["path"]).startswith(subject + "/")]
    if FINAL_ROOT.exists() or FINAL_ROOT.is_symlink():
        if FINAL_ROOT.is_symlink() or not FINAL_ROOT.is_dir():
            raise ValueError("unsafe existing Eye-BCI final root")
        for item in selected:
            target = FINAL_ROOT / str(item["path"])
            if target.is_symlink() or not target.is_file() or target.stat().st_size != int(item["bytes"]):
                raise ValueError("published Eye-BCI selection is incomplete")
        return {
            "mode": "download-subject",
            "state": "already_published",
            "subject": subject,
            "array_index": array_index,
            "file_count": len(selected),
            "expected_bytes": sum(int(item["bytes"]) for item in selected),
            "network_bytes_written": 0,
            "already_present_files": len(selected),
            "final_root": str(FINAL_ROOT),
            "hashes_computed": False,
            "secret_values_logged": False,
            "signed_urls_logged": False,
        }
    ensure_real_directory(PARTIAL_ROOT)
    active_root = PARTIAL_ROOT
    token = _load_synapse_token()
    downloaded = 0
    skipped = 0
    started = time.monotonic()
    for item in selected:
        target = active_root / str(item["path"])
        ensure_real_directory(target.parent)
        written, already_present = download_with_retries(token, item, target)
        downloaded += written
        skipped += int(already_present)
    return {
        "mode": "download-subject",
        "state": "completed",
        "subject": subject,
        "array_index": array_index,
        "file_count": len(selected),
        "expected_bytes": sum(int(item["bytes"]) for item in selected),
        "network_bytes_written": downloaded,
        "already_present_files": skipped,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "partial_root": str(PARTIAL_ROOT),
        "final_root": str(FINAL_ROOT),
        "hashes_computed": False,
        "secret_values_logged": False,
        "signed_urls_logged": False,
    }


def download_shard(
    manifest: dict[str, Any], shard_index: int, shard_count: int
) -> dict[str, Any]:
    subjects = sorted({str(item["path"]).split("/", 1)[0] for item in manifest["files"]})
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid shard index/count")
    remaining_indices = list(range(1, len(subjects)))
    selected_indices = [
        subject_index
        for position, subject_index in enumerate(remaining_indices)
        if position % shard_count == shard_index
    ]
    started = time.monotonic()
    results = [download_subject(manifest, subject_index) for subject_index in selected_indices]
    return {
        "mode": "download-shard",
        "state": "completed",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "subjects": [result["subject"] for result in results],
        "file_count": sum(int(result["file_count"]) for result in results),
        "expected_bytes": sum(int(result["expected_bytes"]) for result in results),
        "network_bytes_written": sum(
            int(result["network_bytes_written"]) for result in results
        ),
        "already_present_files": sum(
            int(result["already_present_files"]) for result in results
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "hashes_computed": False,
        "secret_values_logged": False,
        "signed_urls_logged": False,
    }


def verify_selection(root: Path, manifest: dict[str, Any]) -> None:
    expected_paths = {str(item["path"]) for item in manifest["files"]}
    for item in manifest["files"]:
        path = root / str(item["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"missing or size-mismatched selected file: {item['path']}")
    observed_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise ValueError("selection contains unexpected, missing, or partial files")


def finalize(manifest: dict[str, Any]) -> dict[str, Any]:
    lock_descriptor = os.open(PUBLISH_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(lock_descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(lock_descriptor)
        if FINAL_ROOT.exists() or FINAL_ROOT.is_symlink():
            if FINAL_ROOT.is_symlink() or not FINAL_ROOT.is_dir():
                raise ValueError("unsafe existing Eye-BCI final root")
            verify_selection(FINAL_ROOT, manifest)
            state = "already_published"
        else:
            if PARTIAL_ROOT.is_symlink() or not PARTIAL_ROOT.is_dir():
                raise ValueError("Eye-BCI partial root is absent or unsafe")
            verify_selection(PARTIAL_ROOT, manifest)
            os.rename(PARTIAL_ROOT, FINAL_ROOT)
            state = "published"
    finally:
        os.close(lock_descriptor)
        PUBLISH_LOCK.unlink(missing_ok=True)
    return {
        "mode": "finalize",
        "state": state,
        "final_root": str(FINAL_ROOT),
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "selection": "Neuroscan CSV only; no Phantom video",
        "hashes_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("subject", "shard", "finalize"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--shard-count", type=int)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.mode == "subject":
        if args.array_index is None:
            parser.error("subject mode requires --array-index")
        result = download_subject(manifest, args.array_index)
    elif args.mode == "shard":
        if args.array_index is None or args.shard_count is None:
            parser.error("shard mode requires --array-index and --shard-count")
        result = download_shard(manifest, args.array_index, args.shard_count)
    else:
        if args.array_index is not None:
            parser.error("finalize mode does not accept --array-index")
        result = finalize(manifest)
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
