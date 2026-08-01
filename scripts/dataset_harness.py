#!/usr/bin/env python3
"""Stage-one private-project dataset locator and source probe.

The locator compares basenames only, does not request symlink traversal, never
reads file contents, and writes only matching paths plus a short summary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

EEGDENOISENET_FILES = (
    "EEG_all_epochs.mat",
    "EEG_all_epochs.npy",
    "EOG_all_epochs.mat",
    "EOG_all_epochs.npy",
    "EMG_all_epochs.mat",
    "EMG_all_epochs.npy",
)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("search"), dict)
        or not isinstance(payload.get("datasets"), dict)
    ):
        raise ValueError("dataset config is malformed")
    search = payload["search"]
    root = Path(str(search.get("root", "")))
    for name in ("max_depth", "max_entries", "max_seconds", "max_matches_per_dataset"):
        if not isinstance(search.get(name), int) or search[name] <= 0:
            raise ValueError(f"search.{name} must be a positive integer")
    if not root.is_absolute():
        raise ValueError("search root must be absolute")
    for dataset_id, item in payload["datasets"].items():
        if not isinstance(dataset_id, str) or not dataset_id or not isinstance(item, dict):
            raise ValueError("dataset entry is malformed")
        tokens = item.get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(token, str) or not token.strip() for token in tokens)
        ):
            raise ValueError(f"{dataset_id} tokens must be non-empty strings")
        target = Path(str(item.get("target", "")))
        if not target.is_absolute() or os.path.commonpath((root, target)) != str(root):
            raise ValueError(f"{dataset_id} target must be below the search root")
        urls = [item.get("source"), *item.get("probe_urls", [])]
        if any(urlsplit(str(url)).scheme != "https" for url in urls):
            raise ValueError(f"{dataset_id} URLs must use HTTPS")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def locate(
    root: Path,
    datasets: dict[str, Any],
    *,
    max_depth: int,
    max_entries: int,
    max_seconds: int,
    max_matches: int,
) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("search root must be a real directory")

    tokens = {
        dataset_id: tuple(str(token).lower() for token in item.get("tokens", []))
        for dataset_id, item in datasets.items()
    }
    matches: dict[str, list[str]] = {dataset_id: [] for dataset_id in datasets}
    match_counts = {dataset_id: 0 for dataset_id in datasets}
    entries_seen = 0
    directories_opened = 0
    directory_errors = 0
    stop_reason: str | None = None
    started = time.monotonic()
    pending: deque[tuple[Path, int]] = deque([(root, 0)])

    while pending:
        if time.monotonic() - started >= max_seconds:
            stop_reason = "time_limit"
            break
        directory, depth = pending.popleft()
        try:
            current = os.lstat(directory)
            if stat.S_ISLNK(current.st_mode) or not stat.S_ISDIR(current.st_mode):
                directory_errors += 1
                continue
            with os.scandir(directory) as iterator:
                directories_opened += 1
                for entry in iterator:
                    if entries_seen >= max_entries:
                        stop_reason = "entry_limit"
                        break
                    if entries_seen % 256 == 0 and time.monotonic() - started >= max_seconds:
                        stop_reason = "time_limit"
                        break
                    entries_seen += 1
                    lowered = entry.name.lower()
                    for dataset_id, dataset_tokens in tokens.items():
                        if any(token in lowered for token in dataset_tokens):
                            match_counts[dataset_id] += 1
                            if len(matches[dataset_id]) < max_matches:
                                matches[dataset_id].append(entry.path)
                    entry_depth = depth + 1
                    if entry_depth < max_depth and entry.is_dir(follow_symlinks=False):
                        pending.append((Path(entry.path), entry_depth))
                if stop_reason:
                    break
        except OSError:
            directory_errors += 1

    elapsed = time.monotonic() - started
    incomplete = stop_reason is not None or directory_errors > 0
    return {
        "mode": "locate",
        "root": str(root),
        "max_depth": max_depth,
        "max_entries": max_entries,
        "max_seconds": max_seconds,
        "entries_seen": entries_seen,
        "directories_opened": directories_opened,
        "directory_errors": directory_errors,
        "elapsed_seconds": round(elapsed, 3),
        "coverage": "incomplete" if incomplete else "complete_within_declared_bounds",
        "truncated": stop_reason is not None,
        "stop_reason": stop_reason,
        "datasets": {
            dataset_id: {
                "match_count": match_counts[dataset_id],
                "matches": sorted(matches[dataset_id]),
                "result": (
                    "found"
                    if match_counts[dataset_id]
                    else "not_found_incomplete"
                    if incomplete
                    else "not_found_within_bounds"
                ),
            }
            for dataset_id in datasets
        },
    }


def sanitize_url(url: str) -> str:
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    netloc = hostname if parsed.port is None else f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def probe_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json,text/html;q=0.8,*/*;q=0.1",
            "Range": "bytes=0-8191",
            "User-Agent": "denoiseNet-private-research/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(8192)
            return {
                "url": url,
                "status": response.status,
                "final_url": sanitize_url(response.geturl()),
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "bytes_read": len(body),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "url": url,
            "status": exc.code,
            "final_url": sanitize_url(exc.geturl()),
            "content_type": exc.headers.get("Content-Type") if exc.headers else None,
            "content_length": exc.headers.get("Content-Length") if exc.headers else None,
            "bytes_read": 0,
            "error": "HTTPError",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "url": url,
            "status": None,
            "final_url": None,
            "content_type": None,
            "content_length": None,
            "bytes_read": 0,
            "error": type(exc).__name__,
        }


def probe(datasets: dict[str, Any]) -> dict[str, Any]:
    rows = {
        dataset_id: {
            "metadata_public": bool(item.get("metadata_public", False)),
            "download_authorized": bool(item.get("download_authorized", False)),
            "source": item.get("source"),
            "responses": [probe_url(str(url)) for url in item.get("probe_urls", [])],
        }
        for dataset_id, item in datasets.items()
    }
    successful = sum(
        1
        for item in rows.values()
        for response in item["responses"]
        if response["status"] is not None and 200 <= response["status"] < 400
    )
    return {
        "mode": "probe",
        "state": "completed" if successful else "completed_with_errors",
        "successful_endpoint_count": successful,
        "datasets": rows,
    }


def inspect_eegdenoisenet_data(root: Path) -> dict[str, list[Any]]:
    import numpy as np

    missing = [name for name in EEGDENOISENET_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing EEGdenoiseNet files: {missing}")
    arrays: dict[str, list[Any]] = {}
    for name in EEGDENOISENET_FILES:
        if name.endswith(".npy"):
            array = np.load(root / name, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or min(array.shape) <= 0:
                raise ValueError(f"unexpected EEGdenoiseNet array shape: {name} {array.shape}")
            arrays[name] = [int(value) for value in array.shape]
    return arrays


def adopt_eegdenoisenet(item: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(item["local_source"]))
    target = Path(str(item["target"]))
    if target.exists():
        return {
            "mode": "adopt-eegdenoisenet",
            "state": "already_present",
            "target": str(target),
            "array_shapes": inspect_eegdenoisenet_data(target),
        }

    shapes = inspect_eegdenoisenet_data(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    partial = target.parent / f".{target.name}.partial-{job_id}"
    if partial.exists():
        raise FileExistsError(f"partial target already exists: {partial}")
    partial.mkdir()
    copied: list[str] = []
    for path in sorted(source.iterdir()):
        if path.name in shapes or path.suffix == ".mat":
            shutil.copy2(path, partial / path.name)
            copied.append(path.name)
    published_shapes = inspect_eegdenoisenet_data(partial)
    os.replace(partial, target)
    return {
        "mode": "adopt-eegdenoisenet",
        "state": "published",
        "source": str(source),
        "target": str(target),
        "copied_files": copied,
        "array_shapes": published_shapes,
    }


def link_eegdenoisenet(item: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(item["local_source"]))
    target = Path(str(item["target"]))
    shapes = inspect_eegdenoisenet_data(target)
    job_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    linked: list[str] = []
    already_linked: list[str] = []
    for name in EEGDENOISENET_FILES:
        local_path = source / name
        data_path = target / name
        if local_path.is_symlink():
            if local_path.resolve(strict=True) != data_path.resolve(strict=True):
                raise ValueError(f"unexpected existing symlink: {local_path}")
            already_linked.append(name)
            continue
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        temporary = source / f".{name}.link-{job_id}"
        if os.path.lexists(temporary):
            raise FileExistsError(temporary)
        temporary.symlink_to(data_path)
        os.replace(temporary, local_path)
        linked.append(name)
    inspect_eegdenoisenet_data(source)
    return {
        "mode": "link-eegdenoisenet",
        "state": "linked",
        "data_root": str(target),
        "linked_files": linked,
        "already_linked": already_linked,
        "array_shapes": shapes,
    }


def reader_tools() -> dict[str, Any]:
    return {
        "mode": "reader-tools",
        "state": "completed",
        "commands": {
            name: shutil.which(name) for name in ("7z", "7za", "unrar", "bsdtar")
        },
        "python_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("mne", "numpy", "scipy", "h5py")
        },
    }


def self_test() -> dict[str, Any]:
    datasets = {
        "klados_bamidis_v1": {"tokens": ["wb6yvr725d", "klados"]},
        "sgeyesub": {"tokens": ["sgeyesub", "2qgrd"]},
    }
    with tempfile.TemporaryDirectory(prefix="denoisenet-locator-") as temporary:
        root = Path(temporary) / "root"
        outside = Path(temporary) / "outside"
        (root / "a" / "wb6yvr725d-v1").mkdir(parents=True)
        (outside / "sgeyesub").mkdir(parents=True)
        (root / "outside-link").symlink_to(outside, target_is_directory=True)
        result = locate(
            root,
            datasets,
            max_depth=3,
            max_entries=100,
            max_seconds=10,
            max_matches=10,
        )
        assert result["datasets"]["klados_bamidis_v1"]["match_count"] == 1
        assert result["datasets"]["sgeyesub"]["match_count"] == 0
        bounded = locate(
            root,
            datasets,
            max_depth=3,
            max_entries=1,
            max_seconds=10,
            max_matches=10,
        )
        assert bounded["truncated"] and bounded["stop_reason"] == "entry_limit"
    return {"mode": "self-test", "state": "passed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "self-test",
            "locate",
            "probe",
            "adopt-eegdenoisenet",
            "link-eegdenoisenet",
            "reader-tools",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode == "self-test":
        result = self_test()
    elif args.mode == "adopt-eegdenoisenet":
        result = adopt_eegdenoisenet(config["datasets"]["eegdenoisenet"])
    elif args.mode == "link-eegdenoisenet":
        result = link_eegdenoisenet(config["datasets"]["eegdenoisenet"])
    elif args.mode == "reader-tools":
        result = reader_tools()
    elif args.mode == "probe":
        result = probe(config["datasets"])
    else:
        search = config["search"]
        result = locate(
            Path(search["root"]),
            config["datasets"],
            max_depth=int(search["max_depth"]),
            max_entries=int(search["max_entries"]),
            max_seconds=int(search["max_seconds"]),
            max_matches=int(search["max_matches_per_dataset"]),
        )
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
