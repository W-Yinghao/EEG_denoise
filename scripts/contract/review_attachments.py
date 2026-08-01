#!/usr/bin/env python3
"""Read-only attachment inventory and safe extraction for scheduled review."""

from __future__ import annotations

import argparse
import ctypes
import errno
import csv
import hashlib
import io
import json
import math
import mimetypes
import os
import re
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".cls",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sty",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}
MAX_ARCHIVE_MEMBERS = 20_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_ARCHIVE_RATIO = 1_000.0
MAX_CONCATENATED_TEXT_BYTES = 16 * 1024**2
MAX_TEXT_MEMBER_BYTES = 64 * 1024**2
PDF_RENDER_DPI = 110
MAX_COMMAND_CAPTURE_BYTES = 1024**2
MAX_PDF_METADATA_BYTES = 1024**2
MAX_PDF_ANNOTATIONS = 20_000
MAX_PDF_LINKS = 20_000
MAX_PDF_IMAGES = 100_000
MAX_PDF_OBJECT_RECORD_BYTES = 32 * 1024**2
MAX_PDF_TEXT_BYTES = 256 * 1024**2
MAX_PDF_XREF_OBJECTS = 250_000
MIN_DISK_MARGIN_BYTES = 512 * 1024**2
REGISTERED_CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
RENAME_NOREPLACE = 1
AT_FDCWD = -100
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
CONTROL_MARKERS = {
    "artifacts_manifest.json",
    "READY.json",
    "EXTRACTION_COMPLETE.json",
    "EXTRACTION_FAILED.json",
}
CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("private_key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("authorization_bearer", re.compile(rb"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._~+/-]{20,}={0,2}")),
    ("jwt", re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("aws_access_key", re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}")),
    ("github_token", re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")),
    ("slack_token", re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("url_userinfo", re.compile(rb"(?i)https?://[^\s/@:]{1,128}:[^\s/@]{8,128}@")),
    ("signed_url", re.compile(rb"(?i)https?://[^\s\"'<>]{0,4096}(?:X-Amz-Signature|X-Goog-Signature|Signature|sig)=[A-Za-z0-9%_~+/-]{24,}")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    code_root = validate_code_root(REGISTERED_CODE_ROOT)
    descriptor, _ = safe_open_source(code_root, path)
    try:
        before = os.fstat(descriptor)
        digest = sha256_fd(descriptor)
        after = os.fstat(descriptor)
        if source_identity(before) != source_identity(after):
            raise OSError(f"file changed while hashed: {path}")
        return digest
    finally:
        os.close(descriptor)


def sha256_fd(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(file_descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def require_safe_filesystem_primitives() -> None:
    missing: list[str] = []
    for name in ("O_DIRECTORY", "O_NOFOLLOW"):
        if not hasattr(os, name):
            missing.append(name)
    for function, label in (
        (os.open, "open(dir_fd=...)"),
        (os.mkdir, "mkdir(dir_fd=...)"),
        (os.stat, "stat(dir_fd=...)"),
        (os.unlink, "unlink(dir_fd=...)"),
    ):
        if function not in os.supports_dir_fd:
            missing.append(label)
    if os.stat not in os.supports_follow_symlinks:
        missing.append("stat(follow_symlinks=False)")
    if missing:
        raise OSError("safe filesystem primitives unavailable: " + ", ".join(missing))


def validate_code_root(code_root: Path) -> Path:
    lexical = Path(os.path.abspath(code_root))
    if lexical != REGISTERED_CODE_ROOT:
        raise OSError(f"code root must be exactly {REGISTERED_CODE_ROOT}")
    metadata = lexical.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("code root must be a real directory, not a symbolic link")
    resolved = lexical.resolve(strict=True)
    if resolved != lexical:
        raise OSError("code root is not its own canonical path")
    return resolved


def relative_parts(code_root: Path, path: Path) -> tuple[str, ...]:
    lexical = path if path.is_absolute() else code_root / path
    lexical = Path(os.path.abspath(lexical))
    try:
        relative = lexical.relative_to(code_root)
    except ValueError as exc:
        raise OSError(f"path escapes code root: {lexical}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise OSError(f"unsafe relative path: {relative}")
    return relative.parts


def open_directory_by_parts(
    code_root: Path,
    parts: tuple[str, ...],
    *,
    create: bool,
    exclusive_last: bool = False,
) -> int:
    require_safe_filesystem_primitives()
    descriptor = os.open(
        code_root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        for index, component in enumerate(parts):
            last = index == len(parts) - 1
            created = False
            if create:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    if exclusive_last and last:
                        raise
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=descriptor,
            )
            metadata = os.fstat(next_descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_descriptor)
                raise OSError(f"non-directory path component: {component}")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def secure_create_directory(
    code_root: Path, path: Path, *, exclusive_last: bool = True
) -> Path:
    parts = relative_parts(code_root, path)
    descriptor = open_directory_by_parts(
        code_root, parts, create=True, exclusive_last=exclusive_last
    )
    os.close(descriptor)
    return code_root.joinpath(*parts)


def safe_existing_directory(code_root: Path, path: Path) -> Path:
    parts = relative_parts(code_root, path)
    descriptor = open_directory_by_parts(code_root, parts, create=False)
    os.close(descriptor)
    return code_root.joinpath(*parts)


def safe_open_source(code_root: Path, raw_path: Path) -> tuple[int, Path]:
    parts = relative_parts(code_root, raw_path)
    parent_parts = parts[:-1]
    parent_descriptor = (
        open_directory_by_parts(code_root, parent_parts, create=False)
        if parent_parts
        else os.open(
            code_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    )
    try:
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise OSError("attachment source must be a regular non-symbolic file")
    return descriptor, code_root.joinpath(*parts)


def safe_exclusive_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    code_root = validate_code_root(REGISTERED_CODE_ROOT)
    parts = relative_parts(code_root, path)
    descriptor = open_directory_by_parts(
        code_root, parts[:-1], create=False
    )
    file_descriptor = -1
    try:
        file_descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode,
            dir_fd=descriptor,
        )
        view = memoryview(content)
        while view:
            written = os.write(file_descriptor, view)
            view = view[written:]
        os.fsync(file_descriptor)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        os.close(descriptor)


def safe_exclusive_copy(source_descriptor: int, path: Path, maximum_bytes: int) -> tuple[int, str]:
    code_root = validate_code_root(REGISTERED_CODE_ROOT)
    parts = relative_parts(code_root, path)
    parent_descriptor = open_directory_by_parts(
        code_root, parts[:-1], create=False
    )
    output_descriptor = -1
    total = 0
    digest = hashlib.sha256()
    try:
        output_descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.lseek(source_descriptor, 0, os.SEEK_SET)
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise OSError(f"copy exceeds byte budget {maximum_bytes}")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(output_descriptor, view)
                view = view[written:]
        os.fsync(output_descriptor)
        return total, digest.hexdigest()
    finally:
        if output_descriptor >= 0:
            os.close(output_descriptor)
        os.close(parent_descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    code_root = validate_code_root(REGISTERED_CODE_ROOT)
    source_parts = relative_parts(code_root, source)
    destination_parts = relative_parts(code_root, destination)
    source_parent = open_directory_by_parts(code_root, source_parts[:-1], create=False)
    destination_parent = open_directory_by_parts(
        code_root, destination_parts[:-1], create=False
    )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    try:
        source_metadata = os.stat(
            source_parts[-1], dir_fd=source_parent, follow_symlinks=False
        )
        if not stat.S_ISDIR(source_metadata.st_mode):
            raise OSError(f"publish source is not a real directory: {source}")
        try:
            os.stat(
                destination_parts[-1],
                dir_fd=destination_parent,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"publish destination already exists: {destination}")
        result = renameat2(
            source_parent,
            os.fsencode(source_parts[-1]),
            destination_parent,
            os.fsencode(destination_parts[-1]),
            RENAME_NOREPLACE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), str(destination))
    finally:
        os.close(source_parent)
        os.close(destination_parent)


def unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON evidence is absent, non-regular, or symbolic: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_json_pairs)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence is not an object: {path}")
    return payload


def read_regular_beneath(code_root: Path, path: Path, maximum_bytes: int = 16 * 1024**2) -> bytes:
    descriptor, _ = safe_open_source(code_root, path)
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum_bytes:
            raise ValueError(f"evidence file exceeds {maximum_bytes} bytes: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise ValueError(f"evidence file exceeds {maximum_bytes} bytes: {path}")
            chunks.append(block)
        after = os.fstat(descriptor)
        if source_identity(before) != source_identity(after):
            raise ValueError(f"evidence file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_json_beneath(code_root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(
        read_regular_beneath(code_root, path).decode("utf-8"),
        object_pairs_hook=unique_json_pairs,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence is not an object: {path}")
    return payload


def load_unique_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ValueError("PyYAML is required for strict environment registration") from exc

    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    payload = yaml.load(
        read_regular_beneath(validate_code_root(REGISTERED_CODE_ROOT), path).decode(
            "utf-8"
        ),
        Loader=UniqueLoader,
    )
    if not isinstance(payload, dict):
        raise ValueError("environment registry is not a mapping")
    return payload


def scan_bytes_for_credentials(content: bytes) -> str | None:
    for identifier, pattern in CREDENTIAL_PATTERNS:
        if pattern.search(content):
            return identifier
    return None


def scan_file_for_credentials(path: Path) -> str | None:
    descriptor, _ = safe_open_source(
        validate_code_root(REGISTERED_CODE_ROOT), path
    )
    try:
        return scan_fd_for_credentials(descriptor)
    finally:
        os.close(descriptor)


def scan_fd_for_credentials(descriptor: int) -> str | None:
    carry = b""
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        candidate = carry + block
        detected = scan_bytes_for_credentials(candidate)
        if detected:
            return detected
        carry = candidate[-8192:]
    return scan_bytes_for_credentials(carry)


def scan_tree_for_credentials(root: Path) -> None:
    for directory_path, directory_names, file_names, directory_descriptor in os.fwalk(
        root, topdown=True, follow_symlinks=False
    ):
        for name in list(directory_names):
            metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError(f"unsafe output directory entry below {directory_path}")
        for name in file_names:
            before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise OSError(f"unsafe output file below {directory_path}")
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            try:
                detected = scan_fd_for_credentials(descriptor)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if source_identity(before) != source_identity(after):
                raise OSError(f"output changed during credential scan below {directory_path}")
            if detected:
                raise OSError(
                    f"high-confidence credential pattern {detected} detected in output"
                )


def require_disk_capacity(root: Path, required_bytes: int) -> None:
    free_bytes = shutil.disk_usage(root).free
    threshold = required_bytes + MIN_DISK_MARGIN_BYTES
    if free_bytes < threshold:
        raise OSError(
            f"insufficient free space: {free_bytes} bytes available, {threshold} required"
        )


def atomic_text(path: Path, content: str) -> None:
    encoded = content.encode("utf-8")
    detected = scan_bytes_for_credentials(encoded)
    if detected:
        raise OSError(f"high-confidence credential pattern {detected} detected")
    safe_exclusive_write(path, encoded)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def directory_bytes(root: Path) -> int:
    total = 0
    for directory_path, directory_names, file_names, directory_descriptor in os.fwalk(
        root, topdown=True, follow_symlinks=False
    ):
        for name in list(directory_names):
            metadata = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError(f"unsafe directory entry below {directory_path}")
        for name in file_names:
            metadata = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError(f"unsafe artifact inode below {directory_path}")
            total += metadata.st_size
    return total


def ensure_output_budget(root: Path, maximum: int) -> int:
    observed = directory_bytes(root)
    if observed > maximum:
        raise OSError(f"extraction bytes {observed} exceed budget {maximum}")
    return observed


def walk_regular_artifacts(root: Path) -> list[tuple[str, int, str]]:
    records: list[tuple[str, int, str]] = []
    for directory_path, directory_names, file_names, directory_descriptor in os.fwalk(
        root, topdown=True, follow_symlinks=False
    ):
        for name in list(directory_names):
            metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError(f"unsafe directory entry below {directory_path}")
        for name in sorted(file_names):
            metadata_before = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if not stat.S_ISREG(metadata_before.st_mode) or metadata_before.st_nlink != 1:
                raise OSError(f"unsafe artifact inode below {directory_path}")
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory_descriptor,
            )
            try:
                digest = sha256_fd(descriptor)
                metadata_after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if source_identity(metadata_before) != source_identity(metadata_after):
                raise OSError(f"artifact changed while hashed below {directory_path}")
            relative_directory = Path(directory_path).relative_to(root)
            relative = (relative_directory / name).as_posix()
            records.append((relative, metadata_before.st_size, digest))
    records.sort(key=lambda item: item[0])
    return records


def artifact_records(root: Path, excluded_paths: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return records
    for relative, size, digest in walk_regular_artifacts(root):
        if relative in excluded_paths:
            continue
        if len(relative.encode("utf-8")) > 4096:
            raise OSError("artifact relative path exceeds 4096 bytes")
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": digest,
            }
        )
        if len(records) > 250_000:
            raise OSError("artifact count exceeds 250000")
    return records


def finalize_extraction(
    staging: Path,
    *,
    source: Path,
    job_id: str,
    kind: str,
    maximum_bytes: int,
    extra: dict[str, Any] | None = None,
) -> None:
    records = artifact_records(
        staging,
        {"artifacts_manifest.json", "READY.json", "EXTRACTION_FAILED.json"},
    )
    manifest_path = staging / "artifacts_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "source_sha256": sha256_file(source),
            "slurm_job_id": job_id,
            "kind": kind,
            "generated_at_utc": utc_now(),
            "artifacts": records,
            "artifact_bytes": sum(record["bytes"] for record in records),
        },
    )
    ensure_output_budget(staging, maximum_bytes)
    completion = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "generated_at_utc": utc_now(),
        "job_id": job_id,
        "kind": kind,
        "review_complete": False,
    }
    if extra:
        completion.update(extra)
    completion["state"] = "READY"
    completion["extraction_only"] = True
    atomic_json(staging / "READY.json", completion)
    ensure_output_budget(staging, maximum_bytes)


def command_result(
    command: list[str],
    *,
    timeout: int = 900,
    max_file_bytes: int | None = None,
    max_capture_bytes: int = MAX_COMMAND_CAPTURE_BYTES,
) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command_sha256": hashlib.sha256(
                b"\0".join(os.fsencode(value) for value in command)
            ).hexdigest(),
            "status": "unavailable",
            "exit_code": None,
            "stdout": "",
            "stderr": f"{command[0]} is not installed",
        }
    try:
        limit_output = None
        if max_file_bytes is not None:
            if max_file_bytes <= 0:
                return {
                    "command_sha256": hashlib.sha256(
                        b"\0".join(os.fsencode(value) for value in command)
                    ).hexdigest(),
                    "status": "refused_output_budget",
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "no output byte budget remains",
                }

            def apply_file_limit() -> None:
                resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))

            limit_output = apply_file_limit
        with tempfile.TemporaryFile() as stdout_stream, tempfile.TemporaryFile() as stderr_stream:
            completed = subprocess.run(
                [executable, *command[1:]],
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
                timeout=timeout,
                preexec_fn=limit_output,
            )
            stdout_stream.seek(0, os.SEEK_END)
            stderr_stream.seek(0, os.SEEK_END)
            stdout_bytes = stdout_stream.tell()
            stderr_bytes = stderr_stream.tell()
            if stdout_bytes > max_capture_bytes or stderr_bytes > max_capture_bytes:
                return {
                    "command_sha256": hashlib.sha256(
                        b"\0".join(os.fsencode(value) for value in command)
                    ).hexdigest(),
                    "status": "refused_capture_budget",
                    "exit_code": completed.returncode,
                    "stdout": "",
                    "stderr": "",
                    "stdout_bytes": stdout_bytes,
                    "stderr_bytes": stderr_bytes,
                }
            stdout_stream.seek(0)
            stderr_stream.seek(0)
            stdout_raw = stdout_stream.read()
            stderr_raw = stderr_stream.read()
        detected = scan_bytes_for_credentials(stdout_raw + b"\n" + stderr_raw)
        if detected:
            return {
                "command_sha256": hashlib.sha256(
                    b"\0".join(os.fsencode(value) for value in command)
                ).hexdigest(),
                "status": "refused_credential_output",
                "exit_code": completed.returncode,
                "stdout": "",
                "stderr": "",
                "credential_pattern": detected,
            }
        return {
            "command_sha256": hashlib.sha256(
                b"\0".join(os.fsencode(value) for value in command)
            ).hexdigest(),
            "status": "completed" if completed.returncode == 0 else "failed",
            "exit_code": completed.returncode,
            "stdout": stdout_raw.decode("utf-8", errors="replace"),
            "stderr": stderr_raw.decode("utf-8", errors="replace"),
            "stdout_bytes": len(stdout_raw),
            "stderr_bytes": len(stderr_raw),
        }
    except subprocess.TimeoutExpired:
        return {
            "command_sha256": hashlib.sha256(
                b"\0".join(os.fsencode(value) for value in command)
            ).hexdigest(),
            "status": "timeout",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "diagnostic_policy": "timeout output discarded without publication",
        }


def detect_media_type(path: Path) -> tuple[str, dict[str, Any]]:
    detected = command_result(["file", "--brief", "--mime-type", str(path)], timeout=60)
    if detected["status"] == "completed" and detected["stdout"].strip():
        return detected["stdout"].strip(), detected
    guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return guessed, detected


def safe_member_path(name: str) -> tuple[PurePosixPath | None, str | None]:
    normalized = unicodedata.normalize("NFC", name.replace("\\", "/"))
    if "\x00" in normalized:
        return None, "NUL byte"
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        return None, "control character"
    if len(normalized.encode("utf-8")) > 4096:
        return None, "path exceeds 4096 UTF-8 bytes"
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return None, "absolute path"
    member = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in member.parts):
        return None, "empty/dot/path-traversal component"
    return member, None


def zip_member_kind(info: zipfile.ZipInfo) -> str:
    unix_mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(unix_mode)
    if file_type == stat.S_IFLNK:
        return "symlink"
    if info.is_dir():
        return "directory"
    return "file"


def archive_member_record(info: zipfile.ZipInfo) -> dict[str, Any]:
    safe_path, unsafe_reason = safe_member_path(info.filename)
    kind = zip_member_kind(info)
    if kind == "symlink" and unsafe_reason is None:
        unsafe_reason = "symbolic link member"
    return {
        "name": info.filename,
        "normalized_path": str(safe_path) if safe_path is not None else None,
        "kind": kind,
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": info.file_size,
        "crc32": f"{info.CRC:08x}",
        "unsafe_reason": unsafe_reason,
    }


def decode_text(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replacement"


def extract_zip(
    path: Path,
    destination: Path,
    job_id: str,
    *,
    max_extraction_bytes: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "zip",
        "status": "pending",
        "destination": str(destination),
        "safety": {},
        "text_files": [],
        "image_files": [],
        "nested_archives": [],
    }
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            members = [archive_member_record(info) for info in infos]
            total_uncompressed = sum(info.file_size for info in infos)
            total_compressed = sum(info.compress_size for info in infos)
            overall_ratio = total_uncompressed / max(total_compressed, 1)
            unsafe = [record for record in members if record["unsafe_reason"]]
            bomb_reasons: list[str] = []
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                bomb_reasons.append(f"member count {len(infos)} exceeds {MAX_ARCHIVE_MEMBERS}")
            archive_byte_limit = min(MAX_ARCHIVE_UNCOMPRESSED_BYTES, max_extraction_bytes)
            if total_uncompressed > archive_byte_limit:
                bomb_reasons.append(
                    f"uncompressed bytes {total_uncompressed} exceed {archive_byte_limit}"
                )
            if overall_ratio > MAX_ARCHIVE_RATIO and total_uncompressed > 100 * 1024**2:
                bomb_reasons.append(f"compression ratio {overall_ratio:.2f} exceeds {MAX_ARCHIVE_RATIO}")
            oversized_text_members = [
                info.filename
                for info in infos
                if Path(info.filename.replace("\\", "/")).suffix.lower() in TEXT_SUFFIXES
                and info.file_size > MAX_TEXT_MEMBER_BYTES
            ]
            if oversized_text_members:
                bomb_reasons.append(
                    f"text members exceed {MAX_TEXT_MEMBER_BYTES} bytes: "
                    + ", ".join(oversized_text_members[:20])
                )
            result["safety"] = {
                "member_count": len(infos),
                "compressed_bytes": total_compressed,
                "uncompressed_bytes": total_uncompressed,
                "compression_ratio": overall_ratio,
                "unsafe_members": unsafe,
                "bomb_reasons": bomb_reasons,
                "accepted": not unsafe and not bomb_reasons,
            }
            result["members"] = members
            normalized_paths = [
                record["normalized_path"]
                for record in members
                if record["normalized_path"] is not None
            ]
            if len(set(normalized_paths)) != len(normalized_paths):
                result["status"] = "refused_duplicate_normalized_archive_path"
                return result
            if unsafe or bomb_reasons:
                result["status"] = "refused_unsafe_archive"
                return result

            nested_archives = [
                record["normalized_path"]
                for record in members
                if record["normalized_path"]
                and Path(record["normalized_path"]).suffix.lower() in ARCHIVE_SUFFIXES
            ]
            if nested_archives:
                result["nested_archives"] = nested_archives
                result["status"] = "refused_nested_archive_requires_separate_review"
                return result

            unsupported_members = [
                record["normalized_path"]
                for record in members
                if record["kind"] == "file"
                and record["normalized_path"]
                and Path(record["normalized_path"]).suffix.lower()
                not in (TEXT_SUFFIXES | IMAGE_SUFFIXES)
            ]
            if unsupported_members:
                result["unsupported_members"] = unsupported_members
                result["status"] = "refused_unsupported_archive_member"
                return result

            if destination.exists():
                result["status"] = "refused_existing_destination"
                return result

            staging = destination.with_name(destination.name + f".partial-{job_id}")
            if staging.exists():
                result["status"] = "blocked_existing_partial"
                result["error"] = str(staging)
                return result
            secure_create_directory(
                validate_code_root(REGISTERED_CODE_ROOT), staging, exclusive_last=True
            )
            payload_root = staging / "payload"
            metadata_root = staging / "metadata"
            secure_create_directory(
                validate_code_root(REGISTERED_CODE_ROOT),
                payload_root,
                exclusive_last=True,
            )
            secure_create_directory(
                validate_code_root(REGISTERED_CODE_ROOT),
                metadata_root,
                exclusive_last=True,
            )

            concatenated_parts: list[str] = []
            concatenated_bytes = 0
            for info, member_record in zip(infos, members):
                relative = PurePosixPath(member_record["normalized_path"])
                target = payload_root.joinpath(*relative.parts)
                if info.is_dir():
                    directory_descriptor = open_directory_by_parts(
                        payload_root,
                        tuple(relative.parts),
                        create=True,
                        exclusive_last=False,
                    )
                    os.close(directory_descriptor)
                    continue
                parent_descriptor = (
                    open_directory_by_parts(
                        payload_root,
                        tuple(relative.parts[:-1]),
                        create=True,
                        exclusive_last=False,
                    )
                    if relative.parts[:-1]
                    else os.open(
                        payload_root,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    )
                )
                digest = hashlib.sha256()
                output_descriptor = -1
                raw_parts: list[bytes] = []
                try:
                    output_descriptor = os.open(
                        relative.parts[-1],
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        0o600,
                        dir_fd=parent_descriptor,
                    )
                    source = archive.open(info, "r")
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                        view = memoryview(block)
                        while view:
                            written = os.write(output_descriptor, view)
                            view = view[written:]
                        if target.suffix.lower() in TEXT_SUFFIXES:
                            raw_parts.append(block)
                        ensure_output_budget(staging, max_extraction_bytes)
                    source.close()
                    os.fsync(output_descriptor)
                finally:
                    if output_descriptor >= 0:
                        os.close(output_descriptor)
                    os.close(parent_descriptor)
                member_record["sha256"] = digest.hexdigest()
                suffix = target.suffix.lower()
                relative_text = relative.as_posix()
                if suffix in TEXT_SUFFIXES:
                    raw = b"".join(raw_parts)
                    content, encoding = decode_text(raw)
                    text_record = {
                        "path": relative_text,
                        "bytes": len(raw),
                        "encoding": encoding,
                        "sha256": digest.hexdigest(),
                        "read_status": "extracted_for_full_review",
                    }
                    result["text_files"].append(text_record)
                    section = (
                        f"\n===== BEGIN {relative_text} =====\n"
                        + content
                        + f"\n===== END {relative_text} =====\n"
                    )
                    section_bytes = len(section.encode("utf-8"))
                    detected = scan_bytes_for_credentials(section.encode("utf-8"))
                    if detected:
                        raise OSError(
                            f"high-confidence credential pattern {detected} in archive text member"
                        )
                    if concatenated_bytes + section_bytes <= MAX_CONCATENATED_TEXT_BYTES:
                        concatenated_parts.extend(
                            [
                                f"\n===== BEGIN {relative_text} =====\n",
                                content,
                                f"\n===== END {relative_text} =====\n",
                            ]
                        )
                        concatenated_bytes += section_bytes
                    else:
                        text_record["read_status"] = "extracted_not_concatenated_size_limit"
                elif suffix in IMAGE_SUFFIXES:
                    result["image_files"].append(relative_text)
                elif suffix in ARCHIVE_SUFFIXES:
                    raise OSError(f"nested archive escaped preflight: {relative_text}")

            atomic_text(metadata_root / "all_text.txt", "".join(concatenated_parts))
            atomic_json(metadata_root / "archive_members.json", members)
            ensure_output_budget(staging, max_extraction_bytes)
            finalize_extraction(
                staging,
                source=path,
                job_id=job_id,
                kind="zip",
                maximum_bytes=max_extraction_bytes,
            )
            try:
                rename_noreplace(staging, destination)
            except (FileExistsError, OSError) as exc:
                result["status"] = "concurrent_destination_conflict"
                result["error_type"] = type(exc).__name__
                return result
            result["status"] = "safely_extracted"
            return result
    except (OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        result["status"] = "unreadable_or_corrupt"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        return result


def jsonable_pdf_value(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        raise ValueError("PDF metadata nesting exceeds budget")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        if len(value) > 100_000:
            raise ValueError("PDF metadata sequence exceeds budget")
        return [jsonable_pdf_value(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 100_000:
            raise ValueError("PDF metadata mapping exceeds budget")
        return {
            str(key): jsonable_pdf_value(item, depth + 1)
            for key, item in value.items()
        }
    return str(value)


def extract_pdf_annotations(path: Path) -> dict[str, Any]:
    warnings.simplefilter("error")
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        return {
            "status": "pypdf_unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted:
            return {"status": "encrypted", "encrypted": True, "annotations": []}
        trailer_root = reader.trailer.get("/Root")
        root_object = trailer_root.get_object() if trailer_root is not None else {}
        names = root_object.get("/Names") if hasattr(root_object, "get") else None
        if names is not None:
            names = names.get_object()
        embedded = names.get("/EmbeddedFiles") if hasattr(names, "get") else None
        associated_files = root_object.get("/AF") if hasattr(root_object, "get") else None
        if embedded is not None or associated_files is not None:
            raise ValueError("PDF embedded files are refused")
        xref_size = int(reader.trailer.get("/Size", 0) or 0)
        if xref_size < 1 or xref_size > MAX_PDF_XREF_OBJECTS:
            raise ValueError(f"PDF object count {xref_size} is outside the registered budget")

        annotations: list[dict[str, Any]] = []
        page_sizes_points: list[dict[str, float]] = []
        link_count = 0
        image_count = 0
        serialized_bytes = 0
        for page_index, page in enumerate(reader.pages, start=1):
            width_points = float(page.mediabox.width)
            height_points = float(page.mediabox.height)
            if (
                not math.isfinite(width_points)
                or not math.isfinite(height_points)
                or width_points <= 0
                or height_points <= 0
            ):
                raise ValueError(f"page {page_index} has non-finite dimensions")
            page_sizes_points.append({"width": width_points, "height": height_points})
            resources = page.get("/Resources") or {}
            resources = resources.get_object() if hasattr(resources, "get_object") else resources
            xobjects = resources.get("/XObject") if hasattr(resources, "get") else None
            if xobjects is not None:
                xobjects = xobjects.get_object()
                image_count += len(xobjects)
                if image_count > MAX_PDF_IMAGES:
                    raise ValueError("PDF image object count exceeds budget")
            page_annotations = page.get("/Annots") or []
            if page.get("/AF") is not None:
                raise ValueError("PDF page-associated files are refused")
            if len(annotations) + len(page_annotations) > MAX_PDF_ANNOTATIONS:
                raise ValueError("PDF annotation count exceeds budget")
            for reference in page_annotations:
                annotation = reference.get_object()
                subtype = jsonable_pdf_value(annotation.get("/Subtype"))
                if str(subtype) == "/FileAttachment" or annotation.get("/FS") is not None:
                    raise ValueError("PDF file attachment annotation is refused")
                action = annotation.get("/A")
                action = action.get_object() if hasattr(action, "get_object") else action
                uri = jsonable_pdf_value(action.get("/URI")) if hasattr(action, "get") else None
                if str(subtype) == "/Link" or uri is not None:
                    link_count += 1
                    if link_count > MAX_PDF_LINKS:
                        raise ValueError("PDF link count exceeds budget")
                record = {
                    "page": page_index,
                    "subtype": subtype,
                    "contents": jsonable_pdf_value(annotation.get("/Contents")),
                    "title": jsonable_pdf_value(annotation.get("/T")),
                    "subject": jsonable_pdf_value(annotation.get("/Subj")),
                    "rectangle": jsonable_pdf_value(annotation.get("/Rect")),
                    "uri": uri,
                }
                encoded = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
                detected = scan_bytes_for_credentials(encoded)
                if detected:
                    raise ValueError(
                        f"high-confidence credential pattern {detected} in PDF annotation/link"
                    )
                serialized_bytes += len(encoded)
                if serialized_bytes > MAX_PDF_OBJECT_RECORD_BYTES:
                    raise ValueError("PDF annotation/link metadata bytes exceed budget")
                annotations.append(record)
        metadata = jsonable_pdf_value(reader.metadata)
        metadata_bytes = json.dumps(
            metadata, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        if len(metadata_bytes) > MAX_PDF_METADATA_BYTES:
            raise ValueError("PDF metadata bytes exceed budget")
        detected = scan_bytes_for_credentials(metadata_bytes)
        if detected:
            raise ValueError(f"high-confidence credential pattern {detected} in PDF metadata")
        return {
            "status": "completed",
            "encrypted": False,
            "page_count": len(reader.pages),
            "xref_object_count": xref_size,
            "page_sizes_points": page_sizes_points,
            "metadata": metadata,
            "annotations": annotations,
            "annotation_count": len(annotations),
            "link_count": link_count,
            "image_object_count": image_count,
            "serialized_object_bytes": serialized_bytes,
            "embedded_files": False,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def png_dimensions(path: Path) -> tuple[int, int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("rendered PNG is not a single-link regular file")
        header = os.read(descriptor, 24)
    finally:
        os.close(descriptor)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise OSError("rendered page is not a canonical PNG with an IHDR header")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width < 1 or height < 1:
        raise OSError("rendered PNG has invalid dimensions")
    return width, height


def extract_pdf(
    path: Path,
    destination: Path,
    job_id: str,
    *,
    max_pages: int,
    max_page_pixels: int,
    max_total_pixels: int,
    max_extraction_bytes: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "pdf",
        "status": "pending",
        "destination": str(destination),
    }
    if destination.exists():
        result["status"] = "refused_existing_destination"
        return result
    staging = destination.with_name(destination.name + f".partial-{job_id}")
    if staging.exists():
        result["status"] = "blocked_existing_partial"
        result["error"] = str(staging)
        return result
    secure_create_directory(
        validate_code_root(REGISTERED_CODE_ROOT), staging, exclusive_last=True
    )

    pdfinfo = command_result(["pdfinfo", str(path)])
    pdfinfo_audit = {
        "status": pdfinfo["status"],
        "exit_code": pdfinfo["exit_code"],
        "command_sha256": pdfinfo["command_sha256"],
        "stdout_bytes": pdfinfo.get("stdout_bytes", 0),
        "stderr_bytes": pdfinfo.get("stderr_bytes", 0),
        "stdout_sha256": hashlib.sha256(pdfinfo["stdout"].encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(pdfinfo["stderr"].encode("utf-8")).hexdigest(),
    }
    atomic_json(staging / "pdfinfo_audit.json", pdfinfo_audit)
    annotations = extract_pdf_annotations(path)
    atomic_json(staging / "annotations.json", annotations)
    if (
        pdfinfo["status"] != "completed"
        or bool(pdfinfo["stderr"].strip())
        or annotations.get("status") != "completed"
    ):
        result.update(
            {
                "status": "encrypted_or_unreadable_pdf",
                "pdfinfo": pdfinfo_audit,
                "annotation_extraction": annotations,
                "destination": str(staging),
            }
        )
        atomic_json(
            staging / "EXTRACTION_FAILED.json",
            {
                "schema_version": 1,
                "source_sha256": sha256_file(path),
                "job_id": job_id,
                "generated_at_utc": utc_now(),
                "reason": "PDF metadata, encryption, or annotation preflight failed",
                "pdfinfo_status": pdfinfo["status"],
                "annotation_status": annotations.get("status"),
            },
        )
        return result

    page_count = int(annotations["page_count"])
    page_sizes = annotations.get("page_sizes_points", [])
    if page_count < 1 or page_count > max_pages or len(page_sizes) != page_count:
        result["status"] = "refused_pdf_page_budget_or_missing_dimensions"
        result["page_count"] = page_count
        result["destination"] = str(staging)
        atomic_json(
            staging / "EXTRACTION_FAILED.json",
            {
                "schema_version": 1,
                "source_sha256": sha256_file(path),
                "job_id": job_id,
                "generated_at_utc": utc_now(),
                "reason": "page count or dimension preflight failed",
                "page_count": page_count,
                "max_pages": max_pages,
            },
        )
        return result
    scale = PDF_RENDER_DPI / 72.0
    page_pixel_counts: list[int] = []
    total_pixels = 0
    for page_number, size in enumerate(page_sizes, start=1):
        width = max(1, math.ceil(float(size["width"]) * scale))
        height = max(1, math.ceil(float(size["height"]) * scale))
        pixels = width * height
        if pixels > max_page_pixels:
            result["status"] = "refused_pdf_pixel_budget"
            result["error"] = f"page {page_number} pixels {pixels} exceed {max_page_pixels}"
            result["destination"] = str(staging)
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result
        total_pixels += pixels
        if total_pixels > max_total_pixels:
            result["status"] = "refused_pdf_pixel_budget"
            result["error"] = f"total pixels {total_pixels} exceed {max_total_pixels}"
            result["destination"] = str(staging)
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result
        page_pixel_counts.append(pixels)

    text_path = staging / "document.txt"
    layout_path = staging / "document-layout.txt"
    raw_text = command_result(
        ["pdftotext", str(path), str(text_path)],
        max_file_bytes=max_extraction_bytes - directory_bytes(staging),
    )
    if raw_text["status"] == "completed" and not raw_text["stderr"].strip():
        try:
            ensure_output_budget(staging, max_extraction_bytes)
        except OSError:
            raw_text["status"] = "refused_output_budget"
    layout_text = command_result(
        ["pdftotext", "-layout", str(path), str(layout_path)],
        max_file_bytes=max_extraction_bytes - directory_bytes(staging),
    )
    if (
        raw_text["status"] != "completed"
        or bool(raw_text["stderr"].strip())
        or layout_text["status"] != "completed"
        or bool(layout_text["stderr"].strip())
        or not text_path.is_file()
        or text_path.is_symlink()
        or not stat.S_ISREG(text_path.lstat().st_mode)
        or not layout_path.is_file()
        or layout_path.is_symlink()
        or not stat.S_ISREG(layout_path.lstat().st_mode)
    ):
        result.update(
            {
                "status": "pdf_text_extraction_failed",
                "plain_text": raw_text,
                "layout_text": layout_text,
                "destination": str(staging),
            }
        )
        atomic_json(staging / "EXTRACTION_FAILED.json", result)
        return result
    text_bytes = text_path.stat().st_size + layout_path.stat().st_size
    if text_bytes > MAX_PDF_TEXT_BYTES:
        result.update(
            {
                "status": "refused_pdf_text_budget",
                "text_bytes": text_bytes,
                "max_text_bytes": MAX_PDF_TEXT_BYTES,
                "destination": str(staging),
            }
        )
        atomic_json(staging / "EXTRACTION_FAILED.json", result)
        return result
    for text_output in (text_path, layout_path):
        detected = scan_file_for_credentials(text_output)
        if detected:
            text_output.unlink(missing_ok=True)
            result.update(
                {
                    "status": "refused_credential_output",
                    "credential_pattern": detected,
                    "destination": str(staging),
                }
            )
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result
    try:
        ensure_output_budget(staging, max_extraction_bytes)
    except OSError as exc:
        result.update(
            {
                "status": "refused_extraction_byte_budget",
                "error": str(exc),
                "destination": str(staging),
            }
        )
        atomic_json(staging / "EXTRACTION_FAILED.json", result)
        return result

    pages_dir = staging / "pages"
    secure_create_directory(
        validate_code_root(REGISTERED_CODE_ROOT), pages_dir, exclusive_last=True
    )
    rendered_pages: list[dict[str, Any]] = []
    actual_total_pixels = 0
    for page_number in range(1, page_count + 1):
        prefix = pages_dir / f"page-{page_number:04d}"
        rendered = command_result(
            [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-singlefile",
                "-png",
                "-r",
                str(PDF_RENDER_DPI),
                str(path),
                str(prefix),
            ],
            timeout=300,
            max_file_bytes=max_extraction_bytes - directory_bytes(staging),
        )
        rendered["page"] = page_number
        rendered["expected_pixels"] = page_pixel_counts[page_number - 1]
        rendered_pages.append(rendered)
        expected_render = prefix.with_suffix(".png")
        if (
            rendered["status"] != "completed"
            or bool(rendered["stderr"].strip())
            or not expected_render.is_file()
            or expected_render.is_symlink()
            or not stat.S_ISREG(expected_render.lstat().st_mode)
        ):
            result.update(
                {
                    "status": "pdf_render_failed",
                    "rendered_pages": rendered_pages,
                    "destination": str(staging),
                }
            )
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result
        actual_width, actual_height = png_dimensions(expected_render)
        actual_pixels = actual_width * actual_height
        if actual_pixels > max_page_pixels:
            result.update(
                {
                    "status": "refused_actual_pdf_pixel_budget",
                    "page": page_number,
                    "actual_pixels": actual_pixels,
                    "destination": str(staging),
                }
            )
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result
        actual_total_pixels += actual_pixels
        if actual_total_pixels > max_total_pixels:
            result.update(
                {
                    "status": "refused_actual_pdf_pixel_budget",
                    "actual_total_pixels": actual_total_pixels,
                    "destination": str(staging),
                }
            )
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result
        rendered["actual_width"] = actual_width
        rendered["actual_height"] = actual_height
        rendered["actual_pixels"] = actual_pixels
        try:
            ensure_output_budget(staging, max_extraction_bytes)
        except OSError as exc:
            result.update(
                {
                    "status": "refused_extraction_byte_budget",
                    "error": str(exc),
                    "destination": str(staging),
                }
            )
            atomic_json(staging / "EXTRACTION_FAILED.json", result)
            return result

    result.update(
        {
            "pdfinfo": pdfinfo_audit,
            "plain_text": {
                "status": raw_text["status"],
                "exit_code": raw_text["exit_code"],
                "command_sha256": raw_text["command_sha256"],
            },
            "layout_text": {
                "status": layout_text["status"],
                "exit_code": layout_text["exit_code"],
                "command_sha256": layout_text["command_sha256"],
            },
            "rendered_pages": rendered_pages,
            "annotation_extraction": annotations,
            "rendered_page_files": sorted(item.name for item in pages_dir.glob("*.png")),
            "page_count": page_count,
            "total_render_pixels": total_pixels,
            "actual_total_render_pixels": actual_total_pixels,
        }
    )
    finalize_extraction(
        staging,
        source=path,
        job_id=job_id,
        kind="pdf",
        maximum_bytes=max_extraction_bytes,
        extra={
            "page_count": page_count,
            "total_render_pixels": total_pixels,
            "actual_total_render_pixels": actual_total_pixels,
            "render_dpi": PDF_RENDER_DPI,
        },
    )
    try:
        rename_noreplace(staging, destination)
    except (FileExistsError, OSError) as exc:
        result["status"] = "concurrent_destination_conflict"
        result["error_type"] = type(exc).__name__
        return result
    result["status"] = "extracted"
    return result


def extract_text(
    path: Path,
    destination: Path,
    job_id: str,
    *,
    max_extraction_bytes: int,
) -> dict[str, Any]:
    if destination.exists():
        return {
            "kind": "text",
            "status": "refused_existing_destination",
            "destination": str(destination),
        }
    staging = destination.with_name(destination.name + f".partial-{job_id}")
    if staging.exists():
        return {
            "kind": "text",
            "status": "blocked_existing_partial",
            "destination": str(destination),
            "error": str(staging),
        }
    secure_create_directory(
        validate_code_root(REGISTERED_CODE_ROOT), staging, exclusive_last=True
    )
    raw = read_regular_beneath(
        validate_code_root(REGISTERED_CODE_ROOT),
        path,
        min(max_extraction_bytes, MAX_TEXT_MEMBER_BYTES) + 1,
    )
    if len(raw) > min(max_extraction_bytes, MAX_TEXT_MEMBER_BYTES):
        return {
            "kind": "text",
            "status": "refused_extraction_byte_budget",
            "destination": str(staging),
            "bytes": len(raw),
            "budget": min(max_extraction_bytes, MAX_TEXT_MEMBER_BYTES),
        }
    detected = scan_bytes_for_credentials(raw)
    if detected:
        return {
            "kind": "text",
            "status": "refused_credential_output",
            "destination": str(staging),
            "credential_pattern": detected,
        }
    content, encoding = decode_text(raw)
    encoded_content = content.encode("utf-8")
    if len(encoded_content) > max_extraction_bytes:
        return {
            "kind": "text",
            "status": "refused_extraction_byte_budget",
            "destination": str(staging),
            "bytes": len(encoded_content),
            "budget": max_extraction_bytes,
        }
    safe_exclusive_write(staging / "content.txt", encoded_content)
    finalize_extraction(
        staging,
        source=path,
        job_id=job_id,
        kind="text",
        maximum_bytes=max_extraction_bytes,
        extra={"source_encoding": encoding},
    )
    rename_noreplace(staging, destination)
    return {
        "kind": "text",
        "status": "extracted_for_full_review",
        "destination": str(destination),
        "source_encoding": encoding,
    }


def extract_image(
    path: Path,
    destination: Path,
    job_id: str,
    *,
    max_extraction_bytes: int,
) -> dict[str, Any]:
    if destination.exists() or destination.is_symlink():
        return {
            "kind": "image",
            "status": "refused_existing_destination",
            "destination": str(destination),
        }
    staging = destination.with_name(destination.name + f".partial-{job_id}")
    if staging.exists() or staging.is_symlink():
        return {
            "kind": "image",
            "status": "blocked_existing_partial",
            "destination": str(destination),
        }
    secure_create_directory(
        validate_code_root(REGISTERED_CODE_ROOT), staging, exclusive_last=True
    )
    payload = staging / "payload"
    secure_create_directory(
        validate_code_root(REGISTERED_CODE_ROOT), payload, exclusive_last=True
    )
    source_descriptor, _ = safe_open_source(
        validate_code_root(REGISTERED_CODE_ROOT), path
    )
    source_metadata = os.fstat(source_descriptor)
    size = source_metadata.st_size
    if size > max_extraction_bytes:
        os.close(source_descriptor)
        return {
            "kind": "image",
            "status": "refused_extraction_byte_budget",
            "bytes": size,
            "budget": max_extraction_bytes,
            "destination": str(staging),
        }
    target = payload / ("source" + path.suffix.lower())
    try:
        safe_exclusive_copy(source_descriptor, target, max_extraction_bytes)
    finally:
        os.close(source_descriptor)
    finalize_extraction(
        staging,
        source=path,
        job_id=job_id,
        kind="image",
        maximum_bytes=max_extraction_bytes,
        extra={"visual_review_required": True},
    )
    rename_noreplace(staging, destination)
    return {
        "kind": "image",
        "status": "ready_for_visual_review",
        "destination": str(destination),
    }


def source_identity(metadata: os.stat_result) -> dict[str, int]:
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def snapshot_attachment(
    source_path: Path,
    code_root: Path,
    snapshot_root: Path,
    index: int,
    maximum_bytes: int,
) -> dict[str, Any]:
    descriptor, lexical = safe_open_source(code_root, source_path)
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum_bytes:
            raise OSError(
                f"attachment bytes {before.st_size} exceed per-file budget {maximum_bytes}"
            )
        basename = re.sub(r"[^A-Za-z0-9._-]+", "_", lexical.name)[:120] or "attachment"
        snapshot = snapshot_root / f"{index:04d}-{basename}"
        copied_bytes, digest = safe_exclusive_copy(descriptor, snapshot, maximum_bytes)
        after = os.fstat(descriptor)
        if source_identity(before) != source_identity(after):
            raise OSError("attachment changed while its job-private snapshot was created")
        if copied_bytes != before.st_size or sha256_fd(descriptor) != digest:
            raise OSError("attachment snapshot length or digest differs from the opened source")
        os.chmod(snapshot, 0o400, follow_symlinks=False)
        return {
            "original_filename": basename,
            "provided_path_sha256": hashlib.sha256(
                os.fsencode(str(source_path))
            ).hexdigest(),
            "source_relative_path": lexical.relative_to(code_root).as_posix(),
            "source_identity": source_identity(before),
            "source_sha256": digest,
            "snapshot_relative_path": snapshot.relative_to(code_root).as_posix(),
            "snapshot_sha256": sha256_file(snapshot),
            "snapshot_bytes": copied_bytes,
        }
    finally:
        os.close(descriptor)


def verify_original_source(code_root: Path, record: dict[str, Any]) -> None:
    relative = record.get("source_relative_path")
    if not isinstance(relative, str):
        raise OSError("source record lacks its code-root-relative path")
    descriptor, _ = safe_open_source(code_root, Path(relative))
    try:
        metadata = os.fstat(descriptor)
        if source_identity(metadata) != record.get("source_identity"):
            raise OSError("attachment path identity changed after snapshot")
        if sha256_fd(descriptor) != record.get("source_sha256"):
            raise OSError("attachment content changed after snapshot")
    finally:
        os.close(descriptor)


def inspect_attachment(
    path: Path,
    code_root: Path,
    extract_root: Path,
    job_id: str,
    *,
    max_input_bytes: int,
    max_pdf_pages: int,
    max_pdf_page_pixels: int,
    max_pdf_total_pixels: int,
    max_extraction_bytes: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "original_filename": path.name,
        "provided_path": str(path),
        "read_status": "pending",
    }
    try:
        lexical = path if path.is_absolute() else code_root / path
        lexical = Path(os.path.abspath(lexical))
        resolved = lexical.resolve(strict=True)
        code_resolved = code_root.resolve(strict=True)
        try:
            resolved.relative_to(code_resolved)
        except ValueError:
            record.update(
                {
                    "read_status": "refused_outside_code_root",
                    "resolved_path": str(resolved),
                }
            )
            return record
        metadata = lexical.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            record.update(
                {
                    "read_status": "refused_symbolic_link",
                    "resolved_path": str(resolved),
                }
            )
            return record
        if not resolved.is_file():
            record.update(
                {
                    "read_status": "not_a_regular_file",
                    "resolved_path": str(resolved),
                }
            )
            return record
        if metadata.st_size > max_input_bytes:
            record.update(
                {
                    "read_status": "refused_input_size_budget",
                    "resolved_path": str(resolved),
                    "size_bytes": metadata.st_size,
                    "max_input_bytes": max_input_bytes,
                }
            )
            return record
        digest = sha256_file(resolved)
        media_type, file_probe = detect_media_type(resolved)
        modified = datetime.fromtimestamp(metadata.st_mtime, timezone.utc).isoformat()
        record.update(
            {
                "resolved_path": str(resolved),
                "size_bytes": metadata.st_size,
                "modified_at_utc": modified,
                "sha256": digest,
                "media_type": media_type,
                "file_probe": file_probe,
                "readable": os.access(resolved, os.R_OK),
            }
        )
        if not record["readable"]:
            record["read_status"] = "unreadable"
            return record
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.stem)[:80]
        destination = extract_root / f"{safe_stem}-{digest}"
        remaining_extraction_bytes = max_extraction_bytes - directory_bytes(extract_root)
        if remaining_extraction_bytes <= 0:
            record["read_status"] = "refused_extraction_byte_budget"
            return record
        if media_type == "application/zip" or resolved.suffix.lower() == ".zip":
            extraction = extract_zip(
                resolved,
                destination,
                job_id,
                max_extraction_bytes=remaining_extraction_bytes,
            )
        elif media_type == "application/pdf" or resolved.suffix.lower() == ".pdf":
            extraction = extract_pdf(
                resolved,
                destination,
                job_id,
                max_pages=max_pdf_pages,
                max_page_pixels=max_pdf_page_pixels,
                max_total_pixels=max_pdf_total_pixels,
                max_extraction_bytes=remaining_extraction_bytes,
            )
        elif media_type.startswith("text/") or resolved.suffix.lower() in TEXT_SUFFIXES:
            extraction = extract_text(
                resolved,
                destination,
                job_id,
                max_extraction_bytes=remaining_extraction_bytes,
            )
        elif media_type.startswith("image/") and resolved.suffix.lower() in IMAGE_SUFFIXES:
            extraction = extract_image(
                resolved,
                destination,
                job_id,
                max_extraction_bytes=remaining_extraction_bytes,
            )
        elif resolved.suffix.lower() in ARCHIVE_SUFFIXES:
            extraction = {
                "kind": "unsupported_archive",
                "status": "unsupported_archive_format",
                "destination": None,
            }
        else:
            extraction = {
                "kind": "unsupported",
                "status": "unsupported_format",
                "destination": None,
            }
        if sha256_file(resolved) != digest:
            extraction = {
                "kind": extraction.get("kind", "unknown"),
                "status": "stale_source_changed_during_extraction",
                "destination": extraction.get("destination"),
            }
        record["extraction"] = extraction
        record["read_status"] = extraction["status"]
        return record
    except FileNotFoundError as exc:
        record.update({"read_status": "missing", "error": str(exc)})
        return record
    except (OSError, ValueError) as exc:
        record.update(
            {
                "read_status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return record


def review_markdown(records: Iterable[dict[str, Any]], job_id: str) -> str:
    lines = [
        "# Attachment review",
        "",
        f"Scheduled extraction job: `{job_id}`",
        "",
        "This file is the machine-generated extraction checkpoint. Metadata, archive safety, PDF text, page renderings, and annotations were collected without modifying the source attachments. Full semantic review by the primary agent is still required before this status can become `reviewed`; extraction alone is not treated as complete review.",
        "",
        "| Attachment | SHA-256 | Media type | Extraction/read status |",
        "|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` |".format(
                record.get("original_filename", "unknown"),
                record.get("sha256", "unavailable"),
                record.get("media_type", "unknown"),
                record.get("read_status", "unknown"),
            )
        )
    lines.extend(
        [
            "",
            "## Review state",
            "",
            "- Source files were opened read-only.",
            "- ZIP publication is refused if any absolute path, traversal component, symbolic link, or compression-bomb threshold is detected.",
            "- Extracted material is versioned by source hash under `reports/attachment_review_extract/` and cannot overwrite repository source files.",
            "- Requirement mapping and conflict resolution remain `pending_full_read` until every extracted text, PDF page/table/formula/comment, and relevant image has been inspected.",
            "",
        ]
    )
    return "\n".join(lines)


def write_traceability(path: Path, records: list[dict[str, Any]], job_id: str) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=(
            "requirement_id",
            "source",
            "requirement",
            "implementation_or_test",
            "status_and_evidence",
        ),
    )
    writer.writeheader()
    for index, record in enumerate(records, start=1):
        writer.writerow(
            {
                "requirement_id": f"ATT-{index:03d}",
                "source": record.get("provided_path_sha256", "unknown"),
                "requirement": "Complete read-only review and map all executable requirements without changing meaning",
                "implementation_or_test": "reports/attachment_review.md; reports/attachment_manifest.json",
                "status_and_evidence": f"pending_full_read; extraction_job={job_id}; status={record.get('read_status', 'unknown')}",
            }
        )
    atomic_text(path, stream.getvalue())


def nul_argv_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(os.fsencode(value))
        digest.update(b"\0")
    return digest.hexdigest()


def require_hex(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def validate_contract_evidence(args: argparse.Namespace) -> dict[str, Any]:
    code_root = validate_code_root(args.code_root)
    if args.current_job_id != os.environ.get("SLURM_JOB_ID") or not args.current_job_id.isdigit():
        raise ValueError("current Slurm job ID binding failed")
    if os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_ARRAY_TASK_ID"):
        raise ValueError("attachment jobs may not run as arrays")
    if args.profile != "cpu" or os.environ.get("DENOISENET_PROFILE") != args.profile:
        raise ValueError("attachment extraction must use the registered cpu profile")
    if os.environ.get("DENOISENET_JOB") != args.job:
        raise ValueError("job name differs from the submitter binding")
    request_id = os.environ.get("DENOISENET_REQUEST_ID", "")
    if not SAFE_REQUEST_ID.fullmatch(request_id):
        raise ValueError("submission request ID is absent or unsafe")
    if args.payload_sha256 != os.environ.get("DENOISENET_PAYLOAD_ARGS_SHA256"):
        raise ValueError("payload argv hash differs from the submitter binding")
    require_hex(args.payload_sha256, "payload argv hash")

    request_path = (
        code_root / "reports" / "slurm" / "submissions" / "requests" / f"{request_id}.json"
    )
    request = load_json_beneath(code_root, request_path)
    request_sha256 = hashlib.sha256(read_regular_beneath(code_root, request_path)).hexdigest()
    if request_sha256 != os.environ.get("DENOISENET_REQUEST_SHA256"):
        raise ValueError("submission request content differs from its exported hash")
    if not args.runtime_audit_job.isdigit() or not args.dependency_job.isdigit():
        raise ValueError("runtime-audit and dependency job IDs must be numeric")
    if args.job == "extract_pdf" and args.dependency_job == args.current_job_id:
        raise ValueError("PDF extraction cannot depend on itself")
    expected_dependency = f"afterok:{args.dependency_job}"
    expected_hash_fields = {
        "cluster_config_sha256": "DENOISENET_SUBMIT_CONFIG_SHA256",
        "environment_config_sha256": "DENOISENET_ENV_CONFIG_SHA256",
        "job_script_sha256": "DENOISENET_JOB_SCRIPT_SHA256",
        "submitter_sha256": "DENOISENET_SUBMITTER_SHA256",
        "contract_bundle_sha256": "DENOISENET_CONTRACT_BUNDLE_SHA256",
        "slurm_jobs_bundle_sha256": "DENOISENET_SLURM_JOBS_BUNDLE_SHA256",
    }
    if (
        request.get("schema_version") != 1
        or request.get("state") != "prepared_before_sbatch"
        or request.get("request_id") != request_id
        or request.get("job") != args.job
        or request.get("profile") != args.profile
        or request.get("partition") != "CPU"
        or request.get("account") != "c2s"
        or request.get("qos") != "normal"
        or request.get("dependency") != expected_dependency
        or request.get("array") != ""
        or request.get("payload_argument_count") != args.payload_count
        or request.get("payload_arguments_sha256") != args.payload_sha256
    ):
        raise ValueError("submission request does not bind the exact attachment invocation")
    for field, environment_name in expected_hash_fields.items():
        expected = require_hex(os.environ.get(environment_name), environment_name)
        if request.get(field) != expected:
            raise ValueError(f"submission request field {field} differs from exported provenance")

    submission_path = code_root / "reports" / "slurm" / "submissions" / f"{args.current_job_id}.json"
    submission = load_json_beneath(code_root, submission_path)
    if (
        submission.get("schema_version") != 1
        or str(submission.get("job_id")) != args.current_job_id
        or submission.get("request_id") != request_id
        or submission.get("job") != args.job
        or submission.get("profile") != args.profile
        or submission.get("dependency") != expected_dependency
        or submission.get("array") != ""
        or submission.get("request_sha256") != request_sha256
        or submission.get("payload_arguments_sha256") != args.payload_sha256
    ):
        raise ValueError("post-submit manifest does not bind the current attachment job")
    for field, environment_name in expected_hash_fields.items():
        if submission.get(field) != os.environ.get(environment_name):
            raise ValueError(f"post-submit field {field} differs from exported provenance")

    allocation = load_json_beneath(code_root, args.allocation_json)
    allocation_fields = allocation.get("fields")
    if not isinstance(allocation_fields, dict):
        raise ValueError("current allocation JSON lacks a fields object")
    if (
        allocation.get("schema_version") != 1
        or str(allocation.get("job_id")) != args.current_job_id
        or allocation_fields.get("JobId") != args.current_job_id
        or allocation_fields.get("Comment") != f"denoiseNet:{request_id}"
        or allocation_fields.get("Partition") != "CPU"
        or allocation_fields.get("Account") != "c2s"
        or allocation_fields.get("QOS") != "normal"
        or allocation_fields.get("NumCPUs") != "2"
    ):
        raise ValueError("current allocation differs from the registered cpu request")
    observed_dependency = os.environ.get("SLURM_JOB_DEPENDENCY")
    allocation_dependency = allocation_fields.get("Dependency")
    if expected_dependency not in {observed_dependency, allocation_dependency}:
        raise ValueError("live allocation does not expose the frozen runtime-audit dependency")

    environment_config = code_root / "configs" / "environments.yaml"
    config = load_unique_yaml(environment_config)
    environments = config.get("environments")
    if config.get("schema_version") != 1 or not isinstance(environments, dict):
        raise ValueError("environment registry schema is invalid")
    environment_entry = environments.get(args.environment_name)
    if not isinstance(environment_entry, dict):
        raise ValueError("environment is absent from the registry")
    registered_audit = environment_entry.get("strict_reaudit_job_id")
    if registered_audit is None:
        registered_audit = environment_entry.get("audit_job_id")
    if (
        str(registered_audit) != args.runtime_audit_job
        or environment_entry.get("path") != str(args.environment_path)
        or environment_entry.get("compatibility_status") != "compatible"
        or not str(environment_entry.get("responsibility_status", "")).startswith("verified")
    ):
        raise ValueError("runtime audit or environment responsibility is not verified and registered")
    registered_lock = require_hex(
        environment_entry.get("explicit_manifest_sha256"),
        "registered explicit environment manifest",
    )

    audit_dir = safe_existing_directory(
        code_root,
        code_root
        / "reports"
        / "environments"
        / args.environment_name
        / "jobs"
        / args.runtime_audit_job,
    )
    audit_paths = {
        "status": audit_dir / "status.json",
        "probe": audit_dir / "runtime_probe.json",
        "explicit": audit_dir / "conda-explicit.txt",
        "explicit_hash": audit_dir / "conda-explicit.sha256",
        "sanitization": audit_dir / "conda-explicit-sanitization.json",
        "allocation": audit_dir / "slurm_allocation.json",
    }
    audit_status = load_json_beneath(code_root, audit_paths["status"])
    expected_audit_profile = "cpu" if args.environment_name == "eeg2025" else "L40S"
    expected_audit_partition = "CPU" if args.environment_name == "eeg2025" else "L40S"
    if (
        audit_status.get("schema_version") != 1
        or audit_status.get("job") != "audit_runtime"
        or str(audit_status.get("job_id")) != args.runtime_audit_job
        or audit_status.get("environment_name") != args.environment_name
        or audit_status.get("environment_path") != str(args.environment_path)
        or audit_status.get("profile") != expected_audit_profile
        or audit_status.get("state") != "completed"
        or audit_status.get("exit_code") != 0
        or audit_status.get("provenance_complete") is not True
    ):
        raise ValueError("registered runtime audit lacks a provenance-complete success status")
    audit_request_id = audit_status.get("request_id")
    if not isinstance(audit_request_id, str) or not SAFE_REQUEST_ID.fullmatch(audit_request_id):
        raise ValueError("runtime audit status lacks a safe request ID")
    audit_request_path = (
        code_root
        / "reports"
        / "slurm"
        / "submissions"
        / "requests"
        / f"{audit_request_id}.json"
    )
    audit_request_bytes = read_regular_beneath(code_root, audit_request_path)
    audit_request_hash = hashlib.sha256(audit_request_bytes).hexdigest()
    audit_request = json.loads(audit_request_bytes.decode("utf-8"), object_pairs_hook=unique_json_pairs)
    expected_audit_payload = nul_argv_sha256(["--env", args.environment_name])
    if (
        not isinstance(audit_request, dict)
        or audit_request.get("schema_version") != 1
        or audit_request.get("request_id") != audit_request_id
        or audit_request.get("job") != "audit_runtime"
        or audit_request.get("profile") != expected_audit_profile
        or audit_request.get("partition") != expected_audit_partition
        or audit_request.get("payload_argument_count") != 2
        or audit_request.get("payload_arguments_sha256") != expected_audit_payload
        or audit_status.get("request_sha256") != audit_request_hash
        or audit_status.get("payload_arguments_sha256") != expected_audit_payload
    ):
        raise ValueError("runtime audit request chain is incomplete or mismatched")
    for field in expected_hash_fields:
        if audit_status.get(field) != audit_request.get(field):
            raise ValueError(f"runtime audit status/request provenance differs for {field}")
    audit_submission_path = (
        code_root / "reports" / "slurm" / "submissions" / f"{args.runtime_audit_job}.json"
    )
    audit_submission = load_json_beneath(code_root, audit_submission_path)
    if (
        str(audit_submission.get("job_id")) != args.runtime_audit_job
        or audit_submission.get("request_id") != audit_request_id
        or audit_submission.get("job") != "audit_runtime"
        or audit_submission.get("profile") != expected_audit_profile
        or audit_submission.get("request_sha256") != audit_request_hash
        or audit_submission.get("payload_arguments_sha256") != expected_audit_payload
    ):
        raise ValueError("runtime audit post-submit manifest is incomplete or mismatched")

    probe = load_json_beneath(code_root, audit_paths["probe"])
    probe_environment = probe.get("environment")
    if (
        probe.get("schema_version") != 1
        or probe.get("environment_name") != args.environment_name
        or probe.get("compatibility_status") != "compatible"
        or probe.get("critical_import_failures") != []
        or probe.get("compatibility_failures") != []
        or not isinstance(probe_environment, dict)
        or probe_environment.get("CONDA_PREFIX") != str(args.environment_path)
        or str(probe_environment.get("SLURM_JOB_ID")) != args.runtime_audit_job
        or probe_environment.get("SLURM_JOB_PARTITION") != expected_audit_partition
    ):
        raise ValueError("runtime probe does not close the registered environment audit")
    audit_allocation = load_json_beneath(code_root, audit_paths["allocation"])
    audit_allocation_fields = audit_allocation.get("fields")
    if (
        str(audit_allocation.get("job_id")) != args.runtime_audit_job
        or not isinstance(audit_allocation_fields, dict)
        or audit_allocation_fields.get("JobId") != args.runtime_audit_job
        or audit_allocation_fields.get("Comment") != f"denoiseNet:{audit_request_id}"
        or audit_allocation_fields.get("Partition") != expected_audit_partition
    ):
        raise ValueError("runtime audit allocation is incomplete or mismatched")
    sanitization = load_json_beneath(code_root, audit_paths["sanitization"])
    if (
        sanitization.get("schema_version") != 1
        or sanitization.get("return_code") != 0
        or sanitization.get("stdout_format") != "conda-explicit"
        or sanitization.get("stdout_format_valid") is not True
        or sanitization.get("stdout_structure_failures") != []
        or int(sanitization.get("stdout_requirement_count", 0)) < 1
    ):
        raise ValueError("runtime audit explicit manifest capture is not verified")
    explicit_bytes = read_regular_beneath(code_root, audit_paths["explicit"], 256 * 1024**2)
    explicit_hash = hashlib.sha256(explicit_bytes).hexdigest()
    hash_line = read_regular_beneath(code_root, audit_paths["explicit_hash"], 4096).decode(
        "utf-8"
    ).strip()
    expected_hash_line = f"{explicit_hash}  {audit_paths['explicit']}"
    if hash_line != expected_hash_line or explicit_hash != registered_lock:
        raise ValueError("runtime explicit environment lock chain is mismatched")

    evidence_hashes = {
        name: hashlib.sha256(read_regular_beneath(code_root, path, 256 * 1024**2)).hexdigest()
        for name, path in audit_paths.items()
    }
    return {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "provenance_complete": True,
        "job": args.job,
        "job_id": args.current_job_id,
        "profile": args.profile,
        "payload_argument_count": args.payload_count,
        "payload_arguments_sha256": args.payload_sha256,
        "request_id": request_id,
        "request_sha256": request_sha256,
        "request_dependency": expected_dependency,
        "dependency_job_id": args.dependency_job,
        "runtime_audit_job_id": args.runtime_audit_job,
        "runtime_audit_request_id": audit_request_id,
        "runtime_audit_evidence_sha256": evidence_hashes,
        "environment_name": args.environment_name,
        "environment_path": str(args.environment_path),
        "explicit_manifest_sha256": explicit_hash,
        "allocation_sha256": hashlib.sha256(
            read_regular_beneath(code_root, args.allocation_json)
        ).hexdigest(),
    }


def validate_artifact_record_set(
    root: Path,
    records: Any,
    allowed_controls: set[str],
    *,
    require_root_controls: bool = False,
) -> None:
    if not isinstance(records, list):
        raise ValueError("artifact records must be a list")
    expected: dict[str, tuple[int, str]] = {}
    previous = ""
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("artifact record is not an object")
        relative = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or PurePosixPath(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts)
            or relative in CONTROL_MARKERS
            or relative <= previous
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not HEX64.fullmatch(digest)
        ):
            raise ValueError(f"invalid or unsorted artifact record: {relative!r}")
        expected[relative] = (size, digest)
        previous = relative
    actual: dict[str, tuple[int, str]] = {}
    controls_seen: set[str] = set()
    for relative, size, digest in walk_regular_artifacts(root):
        if relative in allowed_controls:
            controls_seen.add(relative)
            continue
        if relative in CONTROL_MARKERS:
            raise ValueError(f"unexpected control marker: {relative}")
        actual[relative] = (size, digest)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))[:20]
        extra = sorted(set(actual) - set(expected))[:20]
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )[:20]
        raise ValueError(
            f"artifact closure mismatch missing={missing} extra={extra} changed={changed}"
        )
    if require_root_controls and not {"artifacts_manifest.json", "READY.json"}.issubset(
        controls_seen
    ):
        raise ValueError("artifact tree lacks its manifest or READY marker")


def verify_review_artifacts(
    *,
    code_root: Path,
    output_root: Path,
    extract_root: Path,
    snapshot_root: Path,
    expected_helper_sha256: str,
    expected_job_id: str,
    require_complete: bool,
) -> dict[str, Any]:
    for root in (output_root, extract_root, snapshot_root):
        safe_existing_directory(code_root, root)
    manifest_path = output_root / "artifacts_manifest.json"
    ready_path = output_root / "READY.json"
    manifest = load_json_beneath(code_root, manifest_path)
    ready = load_json_beneath(code_root, ready_path)
    if (
        manifest.get("schema_version") != 2
        or str(manifest.get("slurm_job_id")) != expected_job_id
        or manifest.get("helper_sha256") != expected_helper_sha256
    ):
        raise ValueError("attachment artifact manifest provenance is mismatched")
    trees = manifest.get("trees")
    if not isinstance(trees, dict) or set(trees) != {"outputs", "extraction", "snapshots"}:
        raise ValueError("attachment artifact manifest tree set is invalid")
    output_controls = {"artifacts_manifest.json", "READY.json"}
    if require_complete:
        output_controls.add("EXTRACTION_COMPLETE.json")
    validate_artifact_record_set(
        output_root, trees["outputs"], output_controls, require_root_controls=True
    )
    validate_artifact_record_set(extract_root, trees["extraction"], set())
    validate_artifact_record_set(snapshot_root, trees["snapshots"], set())
    manifest_sha256 = sha256_file(manifest_path)
    if (
        ready.get("schema_version") != 1
        or ready.get("state") != "READY"
        or ready.get("extraction_only") is not True
        or ready.get("review_complete") is not False
        or str(ready.get("slurm_job_id")) != expected_job_id
        or ready.get("helper_sha256") != expected_helper_sha256
        or ready.get("artifact_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("attachment READY marker does not bind the verified manifest")
    attachment_manifest = load_json_beneath(code_root, output_root / "attachment_manifest.json")
    attachments = attachment_manifest.get("attachments")
    if not isinstance(attachments, list) or not attachments:
        raise ValueError("attachment manifest is empty or malformed")
    for attachment in attachments:
        if not isinstance(attachment, dict):
            raise ValueError("attachment manifest record is malformed")
        source_record = attachment.get("source_snapshot")
        if not isinstance(source_record, dict):
            raise ValueError("attachment manifest lacks source snapshot provenance")
        verify_original_source(code_root, source_record)
        snapshot_relative = source_record.get("snapshot_relative_path")
        if not isinstance(snapshot_relative, str):
            raise ValueError("source snapshot path is missing")
        snapshot_path = code_root / snapshot_relative
        if (
            snapshot_path.is_symlink()
            or not snapshot_path.is_file()
            or sha256_file(snapshot_path) != source_record.get("snapshot_sha256")
            or snapshot_path.stat().st_size != source_record.get("snapshot_bytes")
        ):
            raise ValueError("job-private attachment snapshot changed")
    for root in (output_root, extract_root, snapshot_root):
        scan_tree_for_credentials(root)
    complete_sha256 = None
    if require_complete:
        complete_path = output_root / "EXTRACTION_COMPLETE.json"
        complete = load_json_beneath(code_root, complete_path)
        if (
            complete.get("schema_version") != 1
            or complete.get("state") != "COMPLETE"
            or complete.get("extraction_only") is not True
            or complete.get("review_complete") is not False
            or str(complete.get("slurm_job_id")) != expected_job_id
            or complete.get("helper_sha256") != expected_helper_sha256
            or complete.get("artifact_manifest_sha256") != manifest_sha256
            or complete.get("attachment_manifest_sha256")
            != sha256_file(output_root / "attachment_manifest.json")
            or complete.get("ready_sha256") != sha256_file(ready_path)
        ):
            raise ValueError("attachment COMPLETE marker is not bound to the verified READY tree")
        complete_sha256 = sha256_file(complete_path)
    return {
        "schema_version": 1,
        "verified_at_utc": utc_now(),
        "slurm_job_id": expected_job_id,
        "artifact_manifest_sha256": manifest_sha256,
        "ready_sha256": sha256_file(ready_path),
        "complete_sha256": complete_sha256,
        "credential_findings": 0,
        "review_complete": False,
    }


def finalize_review_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    code_root = validate_code_root(args.code_root)
    verification = verify_review_artifacts(
        code_root=code_root,
        output_root=args.output_root,
        extract_root=args.extract_root,
        snapshot_root=args.snapshot_root,
        expected_helper_sha256=args.expected_helper_sha256,
        expected_job_id=args.job_id,
        require_complete=False,
    )
    contract_sha256 = sha256_file(args.contract_validation)
    complete = {
        "schema_version": 1,
        "state": "COMPLETE",
        "extraction_only": True,
        "review_complete": False,
        "review_blocker": "primary-agent full semantic and visual read is pending",
        "slurm_job_id": args.job_id,
        "helper_sha256": args.expected_helper_sha256,
        "artifact_manifest_sha256": verification["artifact_manifest_sha256"],
        "attachment_manifest_sha256": sha256_file(
            args.output_root / "attachment_manifest.json"
        ),
        "ready_sha256": verification["ready_sha256"],
        "contract_validation_sha256": contract_sha256,
        "credential_findings": 0,
        "generated_at_utc": utc_now(),
    }
    atomic_json(args.output_root / "EXTRACTION_COMPLETE.json", complete)
    return verify_review_artifacts(
        code_root=code_root,
        output_root=args.output_root,
        extract_root=args.extract_root,
        snapshot_root=args.snapshot_root,
        expected_helper_sha256=args.expected_helper_sha256,
        expected_job_id=args.job_id,
        require_complete=True,
    )


def extraction_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--contract-validation", type=Path, required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--max-input-bytes", type=int, required=True)
    parser.add_argument("--max-pdf-pages", type=int, required=True)
    parser.add_argument("--max-pdf-page-pixels", type=int, required=True)
    parser.add_argument("--max-pdf-total-pixels", type=int, required=True)
    parser.add_argument("--max-extraction-bytes", type=int, required=True)
    parser.add_argument("--max-total-input-bytes", type=int, required=True)
    parser.add_argument("--max-attachments", type=int, required=True)
    parser.add_argument("--attachment", type=Path, action="append", required=True)
    args = parser.parse_args(argv)

    code_root = validate_code_root(args.code_root)
    helper = Path(__file__).resolve(strict=True)
    observed_helper_sha256 = sha256_file(helper)
    if observed_helper_sha256 != args.expected_helper_sha256.lower():
        raise SystemExit("attachment helper hash differs from the job-registered hash")
    if not all(
        value > 0
        for value in (
            args.max_input_bytes,
            args.max_pdf_pages,
            args.max_pdf_page_pixels,
            args.max_pdf_total_pixels,
            args.max_extraction_bytes,
            args.max_total_input_bytes,
            args.max_attachments,
        )
    ):
        raise SystemExit("all attachment extraction budgets must be positive")
    if not (
        len(args.expected_helper_sha256) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in args.expected_helper_sha256
        )
    ):
        raise SystemExit("expected helper SHA-256 must contain 64 hexadecimal characters")
    job_id = os.environ.get("SLURM_JOB_ID", "missing-slurm-job-id")
    if not job_id.isdigit():
        raise SystemExit("review_attachments must run inside a numeric Slurm job")
    if len(args.attachment) > args.max_attachments:
        raise SystemExit("attachment count exceeds the registered budget")

    expected_output_root = (
        code_root / "reports" / "attachment_jobs" / job_id / f"outputs.partial-{job_id}"
    )
    lexical_output_root = Path(os.path.abspath(args.output_root))
    if lexical_output_root != expected_output_root:
        raise SystemExit(f"output root must be exactly {expected_output_root}")
    output_root = secure_create_directory(code_root, expected_output_root, exclusive_last=True)
    expected_extract_root = (
        code_root
        / "reports"
        / "attachment_review_extract"
        / "jobs"
        / job_id
        / f"review.partial-{job_id}"
    )
    lexical_extract_root = Path(os.path.abspath(args.extract_root))
    if lexical_extract_root != expected_extract_root:
        raise SystemExit(f"extract root must be exactly {expected_extract_root}")
    extract_container = expected_extract_root.parent
    secure_create_directory(code_root, extract_container, exclusive_last=True)
    extract_root = secure_create_directory(code_root, expected_extract_root, exclusive_last=True)
    expected_snapshot_root = (
        code_root / "reports" / "attachment_jobs" / job_id / "snapshots"
    )
    if Path(os.path.abspath(args.snapshot_root)) != expected_snapshot_root:
        raise SystemExit(f"snapshot root must be exactly {expected_snapshot_root}")
    snapshot_root = secure_create_directory(code_root, expected_snapshot_root, exclusive_last=True)
    contract_validation = load_json_beneath(code_root, args.contract_validation)
    if (
        contract_validation.get("provenance_complete") is not True
        or str(contract_validation.get("job_id")) != job_id
    ):
        raise SystemExit("contract validation is absent or incomplete")
    contract_validation_sha256 = sha256_file(args.contract_validation)
    require_disk_capacity(
        code_root, args.max_extraction_bytes + args.max_total_input_bytes
    )

    snapshots: list[dict[str, Any]] = []
    total_input_bytes = 0
    for index, path in enumerate(args.attachment, start=1):
        snapshot = snapshot_attachment(
            path,
            code_root,
            snapshot_root,
            index,
            args.max_input_bytes,
        )
        total_input_bytes += int(snapshot["snapshot_bytes"])
        if total_input_bytes > args.max_total_input_bytes:
            raise OSError("total attachment input bytes exceed the registered budget")
        snapshots.append(snapshot)

    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        snapshot_path = code_root / str(snapshot["snapshot_relative_path"])
        record = inspect_attachment(
            snapshot_path,
            code_root,
            extract_root,
            job_id,
            max_input_bytes=args.max_input_bytes,
            max_pdf_pages=args.max_pdf_pages,
            max_pdf_page_pixels=args.max_pdf_page_pixels,
            max_pdf_total_pixels=args.max_pdf_total_pixels,
            max_extraction_bytes=args.max_extraction_bytes,
        )
        record.pop("provided_path", None)
        record.pop("resolved_path", None)
        record["original_filename"] = snapshot["original_filename"]
        record["provided_path_sha256"] = snapshot["provided_path_sha256"]
        record["sha256"] = snapshot["source_sha256"]
        record["source_snapshot"] = snapshot
        extraction = record.get("extraction")
        if isinstance(extraction, dict) and isinstance(extraction.get("destination"), str):
            destination_path = Path(extraction["destination"])
            try:
                extraction["destination"] = destination_path.relative_to(code_root).as_posix()
            except ValueError:
                extraction["destination"] = None
        records.append(record)

    for snapshot in snapshots:
        verify_original_source(code_root, snapshot)
    manifest = {
        "schema_version": 2,
        "generated_at_utc": utc_now(),
        "slurm_job_id": job_id,
        "helper_sha256": observed_helper_sha256,
        "code_root_sha256": hashlib.sha256(os.fsencode(str(code_root))).hexdigest(),
        "contract_validation_sha256": contract_validation_sha256,
        "source_policy": "explicit attachment paths only; read-only; no recursive workspace discovery",
        "budgets": {
            "max_input_bytes": args.max_input_bytes,
            "max_pdf_pages": args.max_pdf_pages,
            "max_pdf_page_pixels": args.max_pdf_page_pixels,
            "max_pdf_total_pixels": args.max_pdf_total_pixels,
            "max_extraction_bytes": args.max_extraction_bytes,
            "max_total_input_bytes": args.max_total_input_bytes,
            "max_attachments": args.max_attachments,
            "max_pdf_text_bytes": MAX_PDF_TEXT_BYTES,
            "max_pdf_annotations": MAX_PDF_ANNOTATIONS,
            "max_pdf_links": MAX_PDF_LINKS,
            "max_pdf_images": MAX_PDF_IMAGES,
            "max_pdf_xref_objects": MAX_PDF_XREF_OBJECTS,
        },
        "attachments": records,
        "review_complete": False,
        "review_blocker": "primary-agent full semantic and visual read is pending",
    }
    successful_statuses = {
        "safely_extracted",
        "extracted",
        "extracted_for_full_review",
        "ready_for_visual_review",
    }
    observed = {record.get("read_status") for record in records}
    extraction_succeeded = bool(records) and observed <= successful_statuses
    ensure_output_budget(extract_root, args.max_extraction_bytes)
    if not extraction_succeeded:
        atomic_json(
            output_root / "EXTRACTION_FAILED.json",
            {
                "schema_version": 1,
                "slurm_job_id": job_id,
                "helper_sha256": observed_helper_sha256,
                "generated_at_utc": utc_now(),
                "observed_statuses": sorted(str(status) for status in observed),
                "reason": "one or more attachments failed closed extraction",
                "review_complete": False,
            },
        )
        return 3
    atomic_json(output_root / "attachment_manifest.json", manifest)
    atomic_text(
        output_root / "attachment_extraction_checkpoint.md",
        review_markdown(records, job_id),
    )
    write_traceability(
        output_root / "requirement_traceability_pending.csv",
        records,
        job_id,
    )
    for root in (output_root, extract_root, snapshot_root):
        scan_tree_for_credentials(root)
    output_artifacts = artifact_records(
        output_root,
        {"artifacts_manifest.json", "READY.json", "EXTRACTION_FAILED.json"},
    )
    extraction_artifacts = artifact_records(extract_root, set())
    snapshot_artifacts = artifact_records(snapshot_root, set())
    artifacts_manifest = {
        "schema_version": 2,
        "slurm_job_id": job_id,
        "helper_sha256": observed_helper_sha256,
        "generated_at_utc": utc_now(),
        "review_complete": False,
        "credential_findings": 0,
        "artifact_count": len(output_artifacts)
        + len(extraction_artifacts)
        + len(snapshot_artifacts),
        "artifact_bytes": sum(
            int(record["bytes"])
            for records_for_tree in (
                output_artifacts,
                extraction_artifacts,
                snapshot_artifacts,
            )
            for record in records_for_tree
        ),
        "trees": {
            "outputs": output_artifacts,
            "extraction": extraction_artifacts,
            "snapshots": snapshot_artifacts,
        },
    }
    artifacts_manifest_path = output_root / "artifacts_manifest.json"
    atomic_json(artifacts_manifest_path, artifacts_manifest)
    ensure_output_budget(output_root, args.max_extraction_bytes)
    atomic_json(
        output_root / "READY.json",
        {
            "schema_version": 1,
            "state": "READY",
            "extraction_only": True,
            "slurm_job_id": job_id,
            "helper_sha256": observed_helper_sha256,
            "attachment_manifest_sha256": sha256_file(
                output_root / "attachment_manifest.json"
            ),
            "artifacts_manifest_sha256": sha256_file(artifacts_manifest_path),
            "contract_validation_sha256": contract_validation_sha256,
            "credential_findings": 0,
            "generated_at_utc": utc_now(),
            "review_complete": False,
            "review_blocker": "primary-agent full semantic and visual read is pending",
        },
    )
    verify_review_artifacts(
        code_root=code_root,
        output_root=output_root,
        extract_root=extract_root,
        snapshot_root=snapshot_root,
        expected_helper_sha256=observed_helper_sha256,
        expected_job_id=job_id,
        require_complete=False,
    )
    return 0


def dispatch() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("attachment helper action is required")
    action = sys.argv[1]
    argv = sys.argv[2:]
    if action == "extract":
        return extraction_main(argv)
    if action == "bootstrap":
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--directory", type=Path, required=True)
        args = parser.parse_args(argv)
        code_root = validate_code_root(args.code_root)
        secure_create_directory(code_root, args.directory, exclusive_last=True)
        return 0
    if action == "contract":
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--job", choices=("review_attachments",), required=True)
        parser.add_argument("--profile", choices=("cpu",), required=True)
        parser.add_argument("--payload-sha256", required=True)
        parser.add_argument("--payload-count", type=int, required=True)
        parser.add_argument("--current-job-id", required=True)
        parser.add_argument("--runtime-audit-job", required=True)
        parser.add_argument("--dependency-job", required=True)
        parser.add_argument("--environment-name", choices=("eeg2025",), required=True)
        parser.add_argument("--environment-path", type=Path, required=True)
        parser.add_argument("--allocation-json", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv)
        payload = validate_contract_evidence(args)
        atomic_json(args.output, payload)
        return 0
    if action in {"verify", "finalize"}:
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--output-root", type=Path, required=True)
        parser.add_argument("--extract-root", type=Path, required=True)
        parser.add_argument("--snapshot-root", type=Path, required=True)
        parser.add_argument("--expected-helper-sha256", required=True)
        parser.add_argument("--job-id", required=True)
        parser.add_argument("--contract-validation", type=Path, required=True)
        parser.add_argument("--require-complete", action="store_true")
        parser.add_argument("--output-record", type=Path)
        args = parser.parse_args(argv)
        if action == "finalize":
            payload = finalize_review_artifacts(args)
        else:
            payload = verify_review_artifacts(
                code_root=validate_code_root(args.code_root),
                output_root=args.output_root,
                extract_root=args.extract_root,
                snapshot_root=args.snapshot_root,
                expected_helper_sha256=args.expected_helper_sha256,
                expected_job_id=args.job_id,
                require_complete=args.require_complete,
            )
        if args.output_record is not None:
            atomic_json(args.output_record, payload)
        return 0
    if action == "publish":
        parser = argparse.ArgumentParser()
        parser.add_argument("--code-root", type=Path, required=True)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--destination", type=Path, required=True)
        args = parser.parse_args(argv)
        code_root = validate_code_root(args.code_root)
        safe_existing_directory(code_root, args.source)
        relative_parts(code_root, args.destination)
        rename_noreplace(args.source, args.destination)
        return 0
    raise SystemExit(f"unsupported attachment helper action: {action}")


if __name__ == "__main__":
    try:
        raise SystemExit(dispatch())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"attachment helper failed closed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(3) from exc
