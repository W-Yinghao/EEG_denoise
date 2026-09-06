#!/usr/bin/env python3
"""Stage-one private-project dataset locator and source probe.

The locator compares basenames only, does not request symlink traversal, never
reads file contents, and writes only matching paths plus a short summary.
"""

from __future__ import annotations

import argparse
import configparser
import importlib.util
import json
import os
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import yaml

EEGDENOISENET_FILES = (
    "EEG_all_epochs.mat",
    "EEG_all_epochs.npy",
    "EOG_all_epochs.mat",
    "EOG_all_epochs.npy",
    "EMG_all_epochs.mat",
    "EMG_all_epochs.npy",
)

SYNAPSE_CONFIG = Path("/home/infres/yinwang/.synapseConfig")


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
            name: shutil.which(name)
            for name in ("7z", "7za", "7zz", "unrar", "unar", "bsdtar")
        },
        "python_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("mne", "numpy", "scipy", "h5py", "rarfile", "libarchive")
        },
    }


def probe_eye_bci_auth(item: dict[str, Any]) -> dict[str, Any]:
    """Report whether a safe Synapse login route exists, without reading secrets."""
    config_exists = False
    config_safe = False
    config_issue: str | None = None
    try:
        config_stat = os.lstat(SYNAPSE_CONFIG)
        config_exists = True
        if stat.S_ISLNK(config_stat.st_mode):
            config_issue = "symlink_not_allowed"
        elif not stat.S_ISREG(config_stat.st_mode):
            config_issue = "not_a_regular_file"
        elif config_stat.st_uid != os.getuid():
            config_issue = "wrong_owner"
        elif stat.S_IMODE(config_stat.st_mode) & 0o077:
            config_issue = "group_or_world_accessible"
        else:
            config_safe = True
    except FileNotFoundError:
        config_issue = "missing"
    except OSError:
        config_issue = "stat_failed"

    token_present = bool(os.environ.get("SYNAPSE_AUTH_TOKEN"))
    client_module = importlib.util.find_spec("synapseclient") is not None
    cli_path = shutil.which("synapse")
    public_metadata = probe_url(str(item["probe_urls"][-1]))
    return {
        "mode": "probe-eye-bci-auth",
        "state": (
            "credential_route_present"
            if token_present or config_safe
            else "credentials_missing"
        ),
        "credential_checks": {
            "synapse_auth_token_present": token_present,
            "synapse_config_path": str(SYNAPSE_CONFIG),
            "synapse_config_exists": config_exists,
            "synapse_config_safe_permissions": config_safe,
            "synapse_config_issue": config_issue,
        },
        "client_checks": {
            "synapseclient_module_available": client_module,
            "synapse_cli_available": cli_path is not None,
        },
        "public_project_metadata": public_metadata,
        "secret_values_read_or_logged": False,
    }


def _load_synapse_token() -> str:
    config_stat = os.lstat(SYNAPSE_CONFIG)
    if (
        stat.S_ISLNK(config_stat.st_mode)
        or not stat.S_ISREG(config_stat.st_mode)
        or config_stat.st_uid != os.getuid()
        or stat.S_IMODE(config_stat.st_mode) & 0o077
    ):
        raise PermissionError("unsafe ~/.synapseConfig ownership or permissions")
    if config_stat.st_size > 65536:
        raise ValueError("~/.synapseConfig is unexpectedly large")

    parser = configparser.ConfigParser(interpolation=None)
    try:
        with SYNAPSE_CONFIG.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except configparser.Error:
        raise ValueError("~/.synapseConfig could not be parsed") from None
    for section in ("default", "authentication"):
        if parser.has_option(section, "authtoken"):
            token = parser.get(section, "authtoken").strip()
            if token and "\n" not in token and "\r" not in token:
                return token
    raise ValueError("no authtoken in ~/.synapseConfig default profile")


def _synapse_json(
    url: str, token: str, *, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    request_body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "denoiseNet-private-research/1",
        },
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read(65537)
            if len(body) > 65536:
                raise ValueError("Synapse response exceeds login-probe limit")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("unexpected Synapse response type")
            return payload
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Synapse request failed with HTTP {exc.code}") from None


def verify_eye_bci_login(item: dict[str, Any]) -> dict[str, Any]:
    """Authenticate by PAT and verify project visibility without logging identity."""
    token = _load_synapse_token()
    profile = _synapse_json(
        "https://repo-prod.prod.sagebase.org/repo/v1/userProfile", token
    )
    project = _synapse_json(str(item["probe_urls"][-1]), token)
    if not profile.get("ownerId") or project.get("id") != "syn64005218":
        raise ValueError("authenticated Synapse response lacks expected identifiers")
    return {
        "mode": "verify-eye-bci-login",
        "state": "authenticated",
        "authenticated_profile_verified": True,
        "account_identity_logged": False,
        "project": {
            "id": project.get("id"),
            "name": project.get("name"),
            "entity_type": project.get("concreteType"),
        },
        "download_attempted": False,
        "download_scope_verified": False,
        "secret_values_logged": False,
    }


def _synapse_children(token: str, parent_id: str) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    next_page_token: str | None = None
    pages = 0
    while True:
        request_payload: dict[str, Any] = {
            "parentId": parent_id,
            "includeTypes": ["folder", "file"],
            "sortBy": "NAME",
            "sortDirection": "ASC",
        }
        if next_page_token:
            request_payload["nextPageToken"] = next_page_token
        response = _synapse_json(
            "https://repo-prod.prod.sagebase.org/repo/v1/entity/children",
            token,
            payload=request_payload,
        )
        pages += 1
        page = response.get("page", [])
        if not isinstance(page, list):
            raise ValueError("unexpected Synapse children response")
        if any(not isinstance(child, dict) for child in page):
            raise ValueError("unexpected Synapse child entry")
        children.extend(page)
        if len(children) > 1000 or pages > 20:
            raise ValueError("Eye-BCI top-level listing exceeds safety bound")
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return children


def list_eye_bci_top() -> dict[str, Any]:
    """List only direct project children; do not recurse or download files."""
    token = _load_synapse_token()
    raw_children = _synapse_children(token, "syn64005218")
    children = [
        {"id": child.get("id"), "name": child.get("name"), "type": child.get("type")}
        for child in raw_children
    ]
    return {
        "mode": "list-eye-bci-top",
        "state": "listed",
        "project_id": "syn64005218",
        "direct_child_count": len(children),
        "children": children,
        "recursive": False,
        "download_attempted": False,
        "secret_values_logged": False,
    }


def inspect_eye_bci_sample() -> dict[str, Any]:
    """Inspect bounded Info/S01 structure to choose modality filters."""
    token = _load_synapse_token()
    roots = (("Info", "syn64087108"), ("S01", "syn64071512"))
    pending: deque[tuple[str, str, int]] = deque(
        (name, entity_id, 0) for name, entity_id in roots
    )
    entries: list[dict[str, Any]] = []
    while pending:
        parent_path, parent_id, depth = pending.popleft()
        for child in _synapse_children(token, parent_id):
            child_name = str(child.get("name", ""))
            child_type = str(child.get("type", ""))
            child_id = str(child.get("id", ""))
            child_path = f"{parent_path}/{child_name}"
            entries.append(
                {"id": child_id, "path": child_path, "type": child_type}
            )
            if child_type.endswith(".Folder") and depth < 3:
                pending.append((child_path, child_id, depth + 1))
            if len(entries) > 2000:
                raise ValueError("Eye-BCI sample listing exceeds safety bound")
    return {
        "mode": "inspect-eye-bci-sample",
        "state": "listed",
        "roots": [name for name, _ in roots],
        "max_depth": 4,
        "entry_count": len(entries),
        "entries": entries,
        "download_attempted": False,
        "secret_values_logged": False,
    }


def _synapse_file_metadata(
    token: str, entity_id: str, expected_name: str
) -> dict[str, Any]:
    response = _synapse_json(
        f"https://repo-prod.prod.sagebase.org/repo/v1/entity/{entity_id}/filehandles",
        token,
    )
    handles = response.get("list", [])
    if not isinstance(handles, list):
        raise ValueError("unexpected Synapse file-handle response")
    matches = [
        handle
        for handle in handles
        if isinstance(handle, dict)
        and handle.get("fileName") == expected_name
        and handle.get("status") == "AVAILABLE"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one available file handle for {entity_id}")
    handle = matches[0]
    return {
        "file_handle_id": str(handle["id"]),
        "bytes": int(handle["contentSize"]),
        "content_type": handle.get("contentType"),
    }


def inventory_eye_bci_neuroscan() -> dict[str, Any]:
    """Inventory only EEG-bearing Neuroscan CSV files and their server sizes."""
    token = _load_synapse_token()
    subject_pattern = re.compile(r"^S(0[1-9]|[12][0-9]|3[01])$")
    session_pattern = re.compile(r"^Sess0([1-3])$")
    recording_pattern = re.compile(
        r"^(ME|MI|P3004L|P3005L|SSVEP)(0[1-9]|[12][0-9]|3[01])([1-3])\.csv$"
    )
    subjects = [
        child
        for child in _synapse_children(token, "syn64005218")
        if str(child.get("type", "")).endswith(".Folder")
        and subject_pattern.fullmatch(str(child.get("name", "")))
    ]
    files: list[dict[str, Any]] = []
    sessions_seen = 0
    for subject in sorted(subjects, key=lambda item: str(item["name"])):
        subject_name = str(subject["name"])
        sessions = [
            child
            for child in _synapse_children(token, str(subject["id"]))
            if str(child.get("type", "")).endswith(".Folder")
            and session_pattern.fullmatch(str(child.get("name", "")))
        ]
        for session in sorted(sessions, key=lambda item: str(item["name"])):
            session_name = str(session["name"])
            sessions_seen += 1
            modality_folders = [
                child
                for child in _synapse_children(token, str(session["id"]))
                if child.get("name") == "Neuroscan"
                and str(child.get("type", "")).endswith(".Folder")
            ]
            if len(modality_folders) != 1:
                raise ValueError(f"expected one Neuroscan folder in {subject_name}/{session_name}")
            eeg_files = [
                child
                for child in _synapse_children(token, str(modality_folders[0]["id"]))
                if str(child.get("type", "")).endswith(".FileEntity")
            ]
            for child in sorted(eeg_files, key=lambda item: str(item["name"])):
                file_name = str(child["name"])
                match = recording_pattern.fullmatch(file_name)
                expected_subject = subject_name[1:]
                expected_session = session_name[-1]
                if (
                    match is None
                    or match.group(2) != expected_subject
                    or match.group(3) != expected_session
                ):
                    raise ValueError(f"unexpected Neuroscan filename: {file_name}")
                entity_id = str(child["id"])
                files.append(
                    {
                        "entity_id": entity_id,
                        "path": f"{subject_name}/{session_name}/Neuroscan/{file_name}",
                        **_synapse_file_metadata(token, entity_id, file_name),
                    }
                )
    return {
        "mode": "inventory-eye-bci-neuroscan",
        "state": "inventoried",
        "subject_count": len(subjects),
        "session_count": sessions_seen,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "download_attempted": False,
        "hashes_computed": False,
        "secret_values_logged": False,
    }


def probe_eye_bci_download_scope() -> dict[str, Any]:
    """Request a signed URL for one EEG file without fetching file content."""
    token = _load_synapse_token()
    query = urlencode(
        {
            "redirect": "false",
            "fileAssociateType": "FileEntity",
            "fileAssociateId": "syn64072638",
        }
    )
    request = urllib.request.Request(
        f"https://repo-prod.prod.sagebase.org/file/v1/file/149808833?{query}",
        headers={
            "Accept": "text/plain",
            "Authorization": f"Bearer {token}",
            "User-Agent": "denoiseNet-private-research/1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            signed_url = response.read(8193).decode("utf-8").strip()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Synapse download-scope probe failed with HTTP {exc.code}") from None
    parsed = urlsplit(signed_url)
    if len(signed_url) > 8192 or parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Synapse returned an invalid signed download URL")
    return {
        "mode": "probe-eye-bci-download-scope",
        "state": "download_authorized",
        "entity_id": "syn64072638",
        "signed_url_received": True,
        "file_content_requested": False,
        "bytes_downloaded": 0,
        "signed_url_logged": False,
        "secret_values_logged": False,
    }


def audit_eye_bci_pilot() -> dict[str, Any]:
    """Read only S01 CSV headers and one row from the completed download pilot."""
    import csv

    manifest_path = Path(
        "/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919275/attempt-0/result.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_root = Path(
        "/projects/EEG-foundation-model/eye_bci/syn64005218-neuroscan"
    )
    partial_root = Path(
        "/projects/EEG-foundation-model/eye_bci/.syn64005218-neuroscan.partial"
    )
    root = final_root if final_root.is_dir() and not final_root.is_symlink() else partial_root
    selected = [
        item for item in manifest["files"] if str(item["path"]).startswith("S01/")
    ]
    if len(selected) != 5:
        raise ValueError("pilot manifest does not contain five S01 files")
    files: list[dict[str, Any]] = []
    for item in selected:
        path = root / str(item["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"pilot file is missing or size-mismatched: {item['path']}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            first_row = next(reader)
        if (
            len(header) < 10
            or len(first_row) != len(header)
            or any(not name.strip() for name in header)
            or len(set(header)) != len(header)
        ):
            raise ValueError(f"invalid CSV header or first row: {item['path']}")
        files.append(
            {
                "path": item["path"],
                "bytes": int(item["bytes"]),
                "column_count": len(header),
                "columns": header,
                "sample_rows_read": 1,
            }
        )
    common_columns = sorted(set.intersection(*(set(item["columns"]) for item in files)))
    return {
        "mode": "audit-eye-bci-pilot",
        "state": "verified_readable",
        "file_count": len(files),
        "files": files,
        "common_columns": common_columns,
        "declared_merged_fields_present": {
            name: name in common_columns
            for name in (
                "Trig",
                "Cues",
                "PhanFrame",
                "PhanTime",
                "RelTime",
                "RecordingTimestamp",
                "LocalTimeStamp",
                "Blinks",
            )
        },
        "raw_signal_values_logged": False,
        "hashes_computed": False,
    }


def audit_eye_bci_download() -> dict[str, Any]:
    """Verify every selected CSV by registered size, header, and one data row."""
    import csv

    manifest_path = Path(
        "/home/infres/yinwang/denoiseNet/reports/dataset_harness/jobs/919275/attempt-0/result.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(
        "/projects/EEG-foundation-model/eye_bci/syn64005218-neuroscan"
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("published Eye-BCI Neuroscan root is absent or unsafe")
    profile_paths: dict[tuple[str, ...], list[str]] = {}
    subject_sessions: set[tuple[str, str]] = set()
    observed_bytes = 0
    for item in manifest["files"]:
        relative = Path(str(item["path"]))
        path = root / relative
        expected = int(item["bytes"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected:
            raise ValueError(f"published file is missing or size-mismatched: {relative}")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            first_row = next(reader)
        if (
            len(header) < 10
            or len(first_row) != len(header)
            or any(not name.strip() for name in header)
            or len(set(header)) != len(header)
        ):
            raise ValueError(f"invalid CSV header or first row: {relative}")
        profile_paths.setdefault(tuple(header), []).append(str(relative))
        parts = relative.parts
        subject_sessions.add((parts[0], parts[1]))
        observed_bytes += expected
    if observed_bytes != int(manifest["total_bytes"]):
        raise ValueError("published byte total differs from manifest")
    profile_rows = [
        {
            "file_count": len(paths),
            "column_count": len(columns),
            "columns": list(columns),
            "paths": paths,
        }
        for columns, paths in profile_paths.items()
    ]
    return {
        "mode": "audit-eye-bci-download",
        "state": "verified_readable",
        "subject_count": len({subject for subject, _ in subject_sessions}),
        "session_count": len(subject_sessions),
        "file_count": int(manifest["file_count"]),
        "total_bytes": observed_bytes,
        "schema_profile_count": len(profile_rows),
        "schema_profiles": profile_rows,
        "sample_rows_read_per_file": 1,
        "raw_signal_values_logged": False,
        "hashes_computed": False,
    }


def audit_sgeyesub(item: dict[str, Any]) -> dict[str, Any]:
    import mne
    import numpy as np

    root = Path(str(item["target"]))
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"SGEYESUB target is not a real directory: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    extension_counts: dict[str, int] = {}
    for path in files:
        extension = path.suffix.lower() or "[none]"
        extension_counts[extension] = extension_counts.get(extension, 0) + 1

    samples: list[dict[str, Any]] = []
    for study_name in ("study01", "study02", "study03", "study04", "study05"):
        study_root = root / study_name
        candidates = sorted(study_root.glob("*.set"))
        if not candidates:
            raise FileNotFoundError(f"no EEGLAB SET file in {study_root}")
        sample_path = candidates[0]
        epochs = mne.io.read_epochs_eeglab(sample_path, verbose="ERROR")
        first_epoch = epochs[0].get_data()
        samples.append(
            {
                "study": study_name,
                "file": sample_path.name,
                "channels": len(epochs.ch_names),
                "sampling_hz": float(epochs.info["sfreq"]),
                "epochs": len(epochs),
                "samples_per_epoch": len(epochs.times),
                "first_epoch_finite": bool(np.isfinite(first_epoch).all()),
            }
        )
    if not all(sample["first_epoch_finite"] for sample in samples):
        raise ValueError("non-finite values in an SGEYESUB sample window")
    return {
        "mode": "audit-sgeyesub",
        "state": "verified_readable",
        "target": str(root),
        "file_count": len(files),
        "extension_counts": extension_counts,
        "study_samples": samples,
    }


def _mat_values(value: Any) -> list[Any]:
    import numpy as np

    if value is None:
        return []
    return np.asarray(value, dtype=object).reshape(-1).tolist()


def _value_counts(value: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in _mat_values(value):
        if hasattr(item, "item"):
            item = item.item()
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        counts[str(item)] += 1
    return dict(sorted(counts.items()))


def _h5_deref(h5_file: Any, node: Any, *, max_depth: int = 8) -> Any:
    import h5py
    import numpy as np

    depth = 0
    while (
        isinstance(node, h5py.Dataset)
        and node.size == 1
        and h5py.check_dtype(ref=node.dtype) is not None
    ):
        if depth >= max_depth:
            raise ValueError("HDF5 reference depth exceeds safety limit")
        flat = np.asarray(node[()]).reshape(-1)
        reference = flat[0]
        if not isinstance(reference, h5py.Reference) or not reference:
            raise ValueError(f"invalid HDF5 reference in {node.name}")
        node = h5_file[reference]
        depth += 1
    return node


def _h5_array(h5_file: Any, node: Any, *, max_elements: int) -> Any:
    import h5py

    node = _h5_deref(h5_file, node)
    if not isinstance(node, h5py.Dataset):
        return None
    if h5py.check_dtype(ref=node.dtype) is not None:
        raise ValueError(f"unsupported reference array in {node.name}")
    if node.size > max_elements:
        raise ValueError(
            f"refusing to read {node.size} elements from metadata field {node.name}"
        )
    value = node[()]
    if getattr(value, "dtype", node.dtype).kind == "O":
        raise ValueError(f"unsupported object array in {node.name}")
    return value


def _h5_text(h5_file: Any, node: Any) -> str:
    import h5py
    import numpy as np

    node = _h5_deref(h5_file, node)
    if not isinstance(node, h5py.Dataset):
        return ""
    if node.size > 4096:
        raise ValueError(f"refusing to read oversized text field {node.name}")
    if h5py.check_dtype(vlen=node.dtype) in (str, bytes):
        raw = node.asstr()[()]
        return "".join(np.asarray(raw, dtype=str).reshape(-1, order="F"))
    value = _h5_array(h5_file, node, max_elements=4096)
    chars = np.asarray(value).reshape(-1, order="F")
    if chars.dtype.kind in "ui":
        return "".join(chr(int(code)) for code in chars if int(code) != 0)
    if chars.dtype.kind == "S":
        return b"".join(bytes(part) for part in chars).decode("utf-8")
    if chars.dtype.kind == "U":
        return "".join(chars.tolist())
    return ""


def _h5_text_list(
    h5_file: Any, node: Any, *, allow_empty: bool = False
) -> list[str]:
    import h5py
    import numpy as np

    if not isinstance(node, h5py.Dataset):
        return []
    if h5py.check_dtype(ref=node.dtype) is None:
        text = _h5_text(h5_file, node)
        return [text] if text else []
    if node.size > 512:
        raise ValueError(f"refusing oversized text reference array {node.name}")
    raw = node[()]
    values: list[str] = []
    for reference in np.asarray(raw).reshape(-1, order="F"):
        if not isinstance(reference, h5py.Reference) or not reference:
            raise ValueError(f"invalid text reference in {node.name}")
        text = _h5_text(h5_file, h5_file[reference])
        if not text and not allow_empty:
            raise ValueError(f"empty text value referenced by {node.name}")
        values.append(text)
    return values


def _h5_field(group: Any, name: str) -> Any:
    import h5py

    if isinstance(group, h5py.Group) and name in group:
        return group[name]
    return None


def _read_sgeyesub_hdf5_header(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as h5_file:
        if "EEG" not in h5_file:
            raise ValueError(f"missing EEG variable in {path}")
        eeg = _h5_deref(h5_file, h5_file["EEG"])
        etc = _h5_deref(h5_file, _h5_field(eeg, "etc"))
        chanlocs = _h5_deref(h5_file, _h5_field(eeg, "chanlocs"))

        def scalar(name: str) -> float:
            value = _h5_array(
                h5_file, _h5_field(eeg, name), max_elements=1
            )
            if value is None or np.size(value) != 1:
                raise ValueError(f"missing scalar EEG.{name} in {path}")
            return float(np.asarray(value).reshape(-1)[0])

        channels = int(scalar("nbchan"))
        trials = int(scalar("trials"))
        blocks = _h5_array(
            h5_file, _h5_field(etc, "trial_blocks"), max_elements=trials
        )
        labels = _h5_array(
            h5_file, _h5_field(etc, "trial_labels"), max_elements=trials
        )
        trial_ids = _h5_array(
            h5_file, _h5_field(etc, "trial_ids"), max_elements=trials
        )
        channel_labels = _h5_text_list(
            h5_file, _h5_field(chanlocs, "labels")
        )
        channel_types = _h5_text_list(
            h5_file, _h5_field(chanlocs, "type"), allow_empty=True
        )
        block_counts = _value_counts(blocks)
        label_counts = _value_counts(labels)
        trial_id_count = int(np.size(trial_ids)) if trial_ids is not None else 0
        if len(channel_labels) != channels or len(channel_types) != channels:
            raise ValueError(f"channel metadata count mismatch in {path}")
        if len(set(channel_labels)) != channels:
            raise ValueError(f"empty or duplicate channel label in {path}")
        if sum(block_counts.values()) != trials or sum(label_counts.values()) != trials:
            raise ValueError(f"trial block/label count mismatch in {path}")
        if trial_id_count not in (0, trials):
            raise ValueError(f"trial ID count mismatch in {path}")
        companion_fdt = path.with_suffix(".fdt")
        if not companion_fdt.is_file():
            raise FileNotFoundError(f"missing companion FDT: {companion_fdt}")
        return {
            "subject_field": _h5_text(h5_file, _h5_field(eeg, "subject")),
            "sampling_hz": scalar("srate"),
            "channels": channels,
            "trials": trials,
            "samples_per_trial": int(scalar("pnts")),
            "companion_fdt_candidate": companion_fdt.name,
            "channel_labels": channel_labels,
            "channel_types": channel_types,
            "trial_block_counts": block_counts,
            "trial_label_counts": label_counts,
            "trial_id_count": trial_id_count,
            "container_format": "matlab_v7.3_hdf5",
        }


def audit_sgeyesub_structure(item: dict[str, Any]) -> dict[str, Any]:
    """Read SET metadata and bounded block metadata; never open external FDT."""
    import h5py
    import numpy as np
    from scipy.io import loadmat

    root = Path(str(item["target"]))
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"SGEYESUB target is not a real directory: {root}")

    recordings: list[dict[str, Any]] = []
    block_metadata: list[dict[str, Any]] = []
    for study_root in sorted(path for path in root.iterdir() if path.is_dir()):
        for set_path in sorted(study_root.glob("*_prep.set")):
            if not h5py.is_hdf5(set_path):
                raise ValueError(
                    f"classic MAT SET needs a dedicated header reader: {set_path}"
                )
            header = _read_sgeyesub_hdf5_header(set_path)
            recordings.append(
                {
                    "study": study_root.name,
                    "participant_stem": set_path.name.removesuffix("_prep.set"),
                    **header,
                }
            )
        for mat_path in sorted(study_root.glob("*_block_dt.mat")):
            variables = []
            if h5py.is_hdf5(mat_path):
                with h5py.File(mat_path, "r") as h5_file:
                    for name, value in sorted(h5_file.items()):
                        if name == "#refs#":
                            continue
                        variables.append(
                            {
                                "name": name,
                                "shape": list(getattr(value, "shape", ())),
                                "dtype": str(getattr(value, "dtype", type(value).__name__)),
                            }
                        )
            else:
                if mat_path.stat().st_size > 1_000_000:
                    raise ValueError(f"block metadata exceeds 1 MiB: {mat_path}")
                payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
                for name, value in sorted(payload.items()):
                    if name.startswith("__"):
                        continue
                    variables.append(
                        {
                            "name": name,
                            "shape": list(np.shape(value)),
                            "dtype": str(getattr(value, "dtype", type(value).__name__)),
                        }
                    )
            block_metadata.append(
                {
                    "study": study_root.name,
                    "participant_stem": mat_path.name.removesuffix("_block_dt.mat"),
                    "variables": variables,
                }
            )
    if not recordings:
        raise FileNotFoundError(f"no *_prep.set files under {root}")
    recording_keys = [
        (recording["study"], recording["participant_stem"])
        for recording in recordings
    ]
    block_keys = [
        (record["study"], record["participant_stem"])
        for record in block_metadata
    ]
    if len(set(recording_keys)) != len(recording_keys):
        raise ValueError("duplicate SGEYESUB recording stem")
    if len(set(block_keys)) != len(block_keys):
        raise ValueError("duplicate SGEYESUB block metadata stem")
    if set(recording_keys) != set(block_keys):
        raise ValueError("SET and block metadata participant stems do not match")

    channel_layouts: list[dict[str, Any]] = []
    channel_layout_ids: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
    for recording in recordings:
        layout_key = (
            tuple(recording.pop("channel_labels")),
            tuple(recording.pop("channel_types")),
        )
        layout_id = channel_layout_ids.get(layout_key)
        if layout_id is None:
            layout_id = f"layout_{len(channel_layouts) + 1:02d}"
            channel_layout_ids[layout_key] = layout_id
            channel_layouts.append(
                {
                    "layout_id": layout_id,
                    "channel_labels": list(layout_key[0]),
                    "channel_types": list(layout_key[1]),
                }
            )
        recording["channel_layout_id"] = layout_id

    block_by_participant = {
        (record["study"], record["participant_stem"]): record["variables"]
        for record in block_metadata
    }
    block_profiles: list[dict[str, Any]] = []
    block_profile_ids: dict[str, str] = {}
    for recording in recordings:
        participant_key = (recording["study"], recording["participant_stem"])
        variables = block_by_participant[participant_key]
        profile_key = json.dumps(variables, sort_keys=True)
        profile_id = block_profile_ids.get(profile_key)
        if profile_id is None:
            profile_id = f"block_profile_{len(block_profiles) + 1:02d}"
            block_profile_ids[profile_key] = profile_id
            block_profiles.append(
                {"profile_id": profile_id, "variables": variables}
            )
        recording["block_metadata_profile_id"] = profile_id

    study_profiles: dict[str, list[dict[str, Any]]] = {}
    study_profile_keys: dict[str, set[str]] = {}
    for recording in recordings:
        profile = {
            key: recording[key]
            for key in (
                "sampling_hz",
                "channels",
                "trials",
                "samples_per_trial",
                "channel_layout_id",
                "trial_block_counts",
                "trial_label_counts",
                "trial_id_count",
            )
        }
        study = recording["study"]
        profile_key = json.dumps(profile, sort_keys=True)
        if profile_key not in study_profile_keys.setdefault(study, set()):
            study_profile_keys[study].add(profile_key)
            study_profiles.setdefault(study, []).append(profile)

    study_counts = Counter(recording["study"] for recording in recordings)
    return {
        "mode": "audit-sgeyesub-structure",
        "state": "structure_read",
        "target": str(root),
        "recording_count": len(recordings),
        "block_metadata_count": len(block_metadata),
        "study_recording_counts": dict(sorted(study_counts.items())),
        "study_profiles": study_profiles,
        "channel_layouts": channel_layouts,
        "recordings": recordings,
        "block_metadata_profiles": block_profiles,
        "fdt_access": "companion_exists_but_not_opened_by_code",
    }


def audit_klados_archive(item: dict[str, Any]) -> dict[str, Any]:
    """Check only the official size and archive signature.

    Native RAR parsing belongs to an existing archive tool, not this small
    project harness.  Member names recovered by an earlier diagnostic remain
    historical evidence and are deliberately not reproduced here.
    """
    archive = Path(str(item["target"])) / str(item["expected_filename"])
    expected_bytes = int(item["expected_bytes"])
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(archive)
    observed_bytes = archive.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"Klados archive size mismatch: expected {expected_bytes}, got {observed_bytes}"
        )
    with archive.open("rb") as stream:
        signature = stream.read(8)
    if signature.startswith(b"Rar!\x1a\x07\x00"):
        rar_version = "RAR4"
    elif signature == b"Rar!\x1a\x07\x01\x00":
        rar_version = "RAR5"
    else:
        raise ValueError(f"unexpected Klados archive signature: {signature.hex()}")
    return {
        "mode": "audit-klados-archive",
        "state": "archive_verified",
        "path": str(archive),
        "bytes": observed_bytes,
        "format": rar_version,
        "native_sample_read": False,
        "member_listing": "not_attempted",
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

    code_root = Path(__file__).resolve().parents[1]
    for gate_path in sorted((code_root / "reports/gates").glob("g*/gate_status.json")):
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        assert gate["threshold_status"] in ("TBD-PREREG", "frozen")
        assert gate["threshold_config"].startswith("configs/gates/")
        assert "threshold_hash" not in gate
    for prior_path in sorted((code_root / "configs/priors").glob("*.yaml")):
        prior = yaml.safe_load(prior_path.read_text(encoding="utf-8"))
        assert all(not str(key).endswith("_hash") for key in prior)
        assert prior["code_version"] == "git_worktree"
    native = yaml.safe_load(
        (code_root / "configs/baselines/native_sgeyesub.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert native["method_id"] == "native_sgeyesub_commit_2c95b4f"
    assert native["oracle_separation"]["native_is_oracle"] is False
    return {
        "mode": "self-test",
        "state": "passed",
        "gate_status_files": 5,
        "prior_files": 4,
    }


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
            "audit-sgeyesub",
            "audit-sgeyesub-structure",
            "audit-klados-archive",
            "probe-eye-bci-auth",
            "verify-eye-bci-login",
            "list-eye-bci-top",
            "inspect-eye-bci-sample",
            "inventory-eye-bci-neuroscan",
            "probe-eye-bci-download-scope",
            "audit-eye-bci-pilot",
            "audit-eye-bci-download",
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
    elif args.mode == "audit-sgeyesub":
        result = audit_sgeyesub(config["datasets"]["sgeyesub"])
    elif args.mode == "audit-sgeyesub-structure":
        result = audit_sgeyesub_structure(config["datasets"]["sgeyesub"])
    elif args.mode == "audit-klados-archive":
        result = audit_klados_archive(config["datasets"]["klados_bamidis_v1"])
    elif args.mode == "probe-eye-bci-auth":
        result = probe_eye_bci_auth(config["datasets"]["eye_bci"])
    elif args.mode == "verify-eye-bci-login":
        result = verify_eye_bci_login(config["datasets"]["eye_bci"])
    elif args.mode == "list-eye-bci-top":
        result = list_eye_bci_top()
    elif args.mode == "inspect-eye-bci-sample":
        result = inspect_eye_bci_sample()
    elif args.mode == "inventory-eye-bci-neuroscan":
        result = inventory_eye_bci_neuroscan()
    elif args.mode == "probe-eye-bci-download-scope":
        result = probe_eye_bci_download_scope()
    elif args.mode == "audit-eye-bci-pilot":
        result = audit_eye_bci_pilot()
    elif args.mode == "audit-eye-bci-download":
        result = audit_eye_bci_download()
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
