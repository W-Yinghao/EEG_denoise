#!/usr/bin/env python3
"""Read-only, full-root Phase-I inventory for the fixed EEG data root.

This program intentionally uses only metadata syscalls and read-only file opens.
It never follows directory symlinks and never creates anything below DATA_ROOT.
Phase I discovers evidence; it does not validate licenses, dataset versions, or
EEG sample readability.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
DATA_ROOT = Path("/projects/EEG-foundation-model")
EEG_ENV = Path("/home/infres/yinwang/anaconda3/envs/eeg2025")
CONDA_BIN = Path("/home/infres/yinwang/anaconda3/bin/conda")
INVENTORY_PARENT = CODE_ROOT / "reports" / "data_inventory"

SCHEMA_VERSION = 1
SHARD_RECORD_LIMIT = 10_000
SHARD_SECONDS_LIMIT = 60.0
DIRECTORY_ENTRY_LSTAT_OPS_PER_SECOND = 200.0
HASH_BYTES_PER_SECOND = 32 * 1024 * 1024
HASH_MAX_FILE_BYTES = 64 * 1024 * 1024
HASH_TOTAL_BUDGET_BYTES = 20 * 1024 * 1024 * 1024
SHARD_OUTPUT_BUDGET_BYTES = 20 * 1024 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
CANDIDATE_EVIDENCE_LIMIT = 500
SCOPED_UNTRACKED_HASH_MAX_FILE_BYTES = 128 * 1024 * 1024
SCOPED_UNTRACKED_HASH_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
SCOPED_UNTRACKED_ROOTS = (
    "configs",
    "datasets",
    "saddpm",
    "scripts",
    "src",
    "tests",
)

DATA_SUFFIXES = {
    b".edf",
    b".bdf",
    b".set",
    b".fdt",
    b".mat",
    b".csv",
    b".tsv",
    b".npy",
    b".npz",
    b".h5",
    b".hdf5",
    b".zarr",
    b".vhdr",
    b".vmrk",
    b".eeg",
    b".fif",
    b".xdf",
}
ARCHIVE_SUFFIXES = {
    b".zip",
    b".tar",
    b".tgz",
    b".tar.gz",
    b".tar.bz2",
    b".tar.xz",
}
REPORTED_SUFFIXES = {
    item.decode("ascii") for item in DATA_SUFFIXES | ARCHIVE_SUFFIXES
} | {".tar*"}
HASHABLE_METADATA_SUFFIXES = {
    b".json",
    b".yaml",
    b".yml",
    b".txt",
    b".md",
    b".csv",
    b".tsv",
    b".sha256",
    b".sha512",
    b".sha1",
    b".md5",
    b".manifest",
    b".license",
}
HASHABLE_METADATA_PREFIXES = (
    b"readme",
    b"license",
    b"licence",
    b"copying",
    b"checksum",
    b"manifest",
    b"dataset_description",
    b"data_dictionary",
)

CANDIDATE_RULES: dict[str, tuple[bytes, ...]] = {
    "klados_bamidis_v1": (
        b"wb6yvr725d",
        b"klados",
        b"bamidis",
        b"semi-simulated eeg/eog",
        b"semi_simulated_eeg_eog",
    ),
    "sgeyesub": (
        b"sgeyesub",
        b"eyeartifactcorrection",
        b"2qgrd",
    ),
    "eye_bci": (
        b"syn64005218",
        b"eye-bci",
        b"eye_bci",
    ),
    "eegdenoisenet": (b"eegdenoisenet",),
}
CANDIDATE_SOURCE_ANCHORS = {
    "klados_bamidis_v1": "https://data.mendeley.com/datasets/wb6yvr725d/1",
    "sgeyesub": "https://osf.io/2qgrd/",
    "eye_bci": "https://doi.org/10.7303/syn64005218",
    "eegdenoisenet": "https://github.com/ncclabsustech/EEGdenoiseNet",
}

CHECKSUM_SUFFIXES = {b".md5", b".sha1", b".sha256", b".sha512"}
MANIFEST_SUFFIXES = {b".json", b".yaml", b".yml", b".manifest"}
CODE_SUFFIXES = {b".py", b".sh", b".bash", b".ipynb", b".r", b".jl"}

MOUNT_ESCAPE = re.compile(rb"\\([0-7]{3})")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_bundle_sha256(directory: Path, suffix: str) -> str:
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink() and path.name.endswith(suffix)
    )
    for path in paths:
        digest.update(f"{sha256_path(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def fixed_command_sha256(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        cwd=CODE_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fixed audit command failed with return code {completed.returncode}")
    return hashlib.sha256(completed.stdout).hexdigest()


def fixed_command_safe_value(argv: list[str], pattern: str) -> str:
    completed = subprocess.run(
        argv,
        cwd=CODE_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fixed provenance command failed with return code {completed.returncode}")
    value = completed.stdout.decode("ascii", errors="strict").strip()
    if not re.fullmatch(pattern, value):
        raise RuntimeError("fixed provenance command returned an unsafe value")
    return value


def scoped_untracked_content_sha256() -> str:
    command = [
        "/usr/bin/git",
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *SCOPED_UNTRACKED_ROOTS,
    ]
    completed = subprocess.run(
        command,
        cwd=CODE_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"scoped untracked-file enumeration failed with return code {completed.returncode}"
        )
    relative_paths = sorted(path for path in completed.stdout.split(b"\0") if path)
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_raw in relative_paths:
        components = relative_raw.split(b"/")
        if (
            relative_raw.startswith(b"/")
            or any(component in {b"", b".", b".."} for component in components)
        ):
            raise RuntimeError("unsafe path returned by scoped untracked-file enumeration")
        relative = os.fsdecode(relative_raw)
        digest.update(len(relative_raw).to_bytes(8, "big"))
        digest.update(relative_raw)
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        parent_fd = os.open(os.fsencode(CODE_ROOT), directory_flags)
        try:
            for component in components[:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = next_fd
            name = components[-1]
            metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(name, dir_fd=parent_fd)
                target_raw = os.fsencode(target) if isinstance(target, str) else target
                after_link = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (
                    after_link.st_dev != metadata.st_dev
                    or after_link.st_ino != metadata.st_ino
                    or after_link.st_mtime_ns != metadata.st_mtime_ns
                    or after_link.st_ctime_ns != metadata.st_ctime_ns
                ):
                    raise RuntimeError(f"scoped untracked symlink changed: {relative}")
                digest.update(b"L")
                digest.update(len(target_raw).to_bytes(8, "big"))
                digest.update(target_raw)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(f"unsupported untracked input type: {relative}")
            if metadata.st_size > SCOPED_UNTRACKED_HASH_MAX_FILE_BYTES:
                raise RuntimeError(
                    f"scoped untracked input exceeds per-file hash budget: {relative}"
                )
            if total_bytes + metadata.st_size > SCOPED_UNTRACKED_HASH_TOTAL_BYTES:
                raise RuntimeError("scoped untracked inputs exceed total hash budget")
            file_flags = os.O_RDONLY | os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                file_flags |= os.O_CLOEXEC
            descriptor = os.open(name, file_flags, dir_fd=parent_fd)
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                opened = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != metadata.st_size
                ):
                    raise RuntimeError(
                        f"scoped untracked input changed before hashing: {relative}"
                    )
                file_digest = hashlib.sha256()
                bytes_read = 0
                while chunk := stream.read(HASH_CHUNK_BYTES):
                    file_digest.update(chunk)
                    bytes_read += len(chunk)
                after = os.fstat(stream.fileno())
                if (
                    bytes_read != opened.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns
                    or after.st_ctime_ns != opened.st_ctime_ns
                ):
                    raise RuntimeError(
                        f"scoped untracked input changed while hashing: {relative}"
                    )
            total_bytes += bytes_read
            digest.update(b"F")
            digest.update(bytes_read.to_bytes(8, "big"))
            digest.update(file_digest.digest())
        finally:
            os.close(parent_fd)
    digest.update(len(relative_paths).to_bytes(8, "big"))
    digest.update(total_bytes.to_bytes(8, "big"))
    return digest.hexdigest()


def capture_dynamic_state() -> dict[str, str]:
    return {
        "contract_bundle_sha256": directory_bundle_sha256(
            CODE_ROOT / "scripts" / "contract", ".py"
        ),
        "slurm_jobs_bundle_sha256": directory_bundle_sha256(
            CODE_ROOT / "scripts" / "slurm" / "jobs", ".sbatch"
        ),
        "tracked_diff_sha256": fixed_command_sha256(
            ["/usr/bin/git", "diff", "--binary", "HEAD"]
        ),
        "git_head_sha256": fixed_command_sha256(
            ["/usr/bin/git", "rev-parse", "--verify", "HEAD"]
        ),
        "git_head": fixed_command_safe_value(
            ["/usr/bin/git", "rev-parse", "--verify", "HEAD"], r"[0-9a-f]{40,64}"
        ),
        "git_branch_sha256": fixed_command_sha256(
            ["/usr/bin/git", "rev-parse", "--abbrev-ref", "HEAD"]
        ),
        "git_branch_or_detached": fixed_command_safe_value(
            ["/usr/bin/git", "rev-parse", "--abbrev-ref", "HEAD"],
            r"[A-Za-z0-9._/-]+",
        ),
        "git_remotes_sha256": fixed_command_sha256(
            ["/usr/bin/git", "remote", "-v"]
        ),
        "git_origin": fixed_command_safe_value(
            ["/usr/bin/git", "remote", "get-url", "origin"],
            r"https://github\.com/W-Yinghao/EEG_denoise(?:\.git)?",
        ),
        "untracked_path_list_sha256": fixed_command_sha256(
            ["/usr/bin/git", "ls-files", "--others", "--exclude-standard", "-z"]
        ),
        "scoped_untracked_content_sha256": scoped_untracked_content_sha256(),
        "environment_explicit_raw_sha256": fixed_command_sha256(
            [str(CONDA_BIN), "list", "--explicit", "-p", str(EEG_ENV)]
        ),
        "environment_pip_freeze_raw_sha256": fixed_command_sha256(
            [
                str(CONDA_BIN),
                "run",
                "--no-capture-output",
                "-p",
                str(EEG_ENV),
                "python",
                "-m",
                "pip",
                "freeze",
            ]
        ),
    }


def require_safe_filesystem_primitives() -> None:
    """Fail closed when this host cannot provide the fd-relative no-follow walk."""

    missing: list[str] = []
    for name in ("O_DIRECTORY", "O_NOFOLLOW"):
        if not hasattr(os, name):
            missing.append(name)
    if os.open not in os.supports_dir_fd:
        missing.append("open(dir_fd=...)")
    if os.stat not in os.supports_dir_fd:
        missing.append("stat(dir_fd=...)")
    if os.readlink not in os.supports_dir_fd:
        missing.append("readlink(dir_fd=...)")
    if os.stat not in os.supports_follow_symlinks:
        missing.append("stat(follow_symlinks=False)")
    if missing:
        raise RuntimeError(
            "safe inventory filesystem primitives are unavailable: " + ", ".join(missing)
        )


def mount_signature(mount_info: dict[str, Any]) -> str:
    relevant = {
        "governing_mount": mount_info.get("governing_mount"),
        "declared_nested_mounts": mount_info.get("declared_nested_mounts", []),
        "mountinfo_error": mount_info.get("mountinfo_error"),
    }
    encoded = json.dumps(relevant, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "utf-8"
    )
    atomic_write_bytes(path, encoded)


def encoded_bytes(value: bytes) -> dict[str, Any]:
    try:
        utf8 = value.decode("utf-8", errors="strict")
        utf8_valid = True
    except UnicodeDecodeError:
        utf8 = value.decode("utf-8", errors="replace")
        utf8_valid = False
    surrogateescaped = value.decode(sys.getfilesystemencoding(), errors="surrogateescape")
    return {
        "display_utf8": utf8,
        "surrogateescaped": surrogateescaped,
        "utf8_valid": utf8_valid,
        "raw_base64": base64.b64encode(value).decode("ascii"),
    }


def encoded_path(path: bytes) -> dict[str, Any]:
    normalized = os.path.normpath(path)
    result = encoded_bytes(normalized)
    result["normalized_absolute"] = True
    return result


def ensure_safe_output_directory(path: Path) -> None:
    code_real = CODE_ROOT.resolve(strict=True)
    requested = Path(os.path.abspath(os.fspath(path)))
    try:
        common = Path(os.path.commonpath((os.fspath(code_real), os.fspath(requested))))
    except ValueError as exc:
        raise ValueError("output path is on an unrelated filesystem namespace") from exc
    if common != code_real:
        raise ValueError(f"output path is outside the code root: {requested}")
    try:
        requested.relative_to(INVENTORY_PARENT)
    except ValueError as exc:
        raise ValueError(f"output path is outside the inventory report root: {requested}") from exc

    current = code_real
    for component in requested.relative_to(code_real).parts:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            current.mkdir(mode=0o700)
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"output path component is a symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"output path component is not a directory: {current}")
    if requested.resolve(strict=True) != requested:
        raise ValueError("resolved output path differs from the fixed code-root path")


class RateLimiter:
    def __init__(self, rate: float) -> None:
        self.interval = 1.0 / rate
        self.next_allowed = time.monotonic()

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_allowed:
            time.sleep(self.next_allowed - now)
            now = time.monotonic()
        self.next_allowed = max(now, self.next_allowed) + self.interval


class OutputBudgetExceeded(RuntimeError):
    pass


class JsonlShardWriter:
    def __init__(self, directory: Path, prefix: str, shared_budget: dict[str, int]) -> None:
        self.directory = directory
        self.prefix = prefix
        self.shared_budget = shared_budget
        self.directory.mkdir(mode=0o700)
        self.index = 0
        self.record_count = 0
        self.shard_started = 0.0
        self.stream: BinaryIO | None = None
        self.partial: Path | None = None
        self.final: Path | None = None
        self.shards: list[dict[str, Any]] = []

    def _open(self) -> None:
        self.index += 1
        name = f"{self.prefix}-{self.index:06d}.jsonl"
        self.final = self.directory / name
        self.partial = self.directory / f"{name}.partial"
        self.stream = self.partial.open("xb")
        self.record_count = 0
        self.shard_started = time.monotonic()

    def write(self, record: dict[str, Any]) -> None:
        encoded = (
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        ).encode("utf-8")
        if self.shared_budget["bytes"] + len(encoded) > SHARD_OUTPUT_BUDGET_BYTES:
            raise OutputBudgetExceeded("inventory shard output budget exhausted")
        if self.stream is None:
            self._open()
        assert self.stream is not None
        self.stream.write(encoded)
        self.record_count += 1
        self.shared_budget["bytes"] += len(encoded)
        if (
            self.record_count >= SHARD_RECORD_LIMIT
            or time.monotonic() - self.shard_started >= SHARD_SECONDS_LIMIT
        ):
            self.close_shard()

    def close_shard(self) -> None:
        if self.stream is None:
            return
        assert self.partial is not None and self.final is not None
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        try:
            os.link(self.partial, self.final, follow_symlinks=False)
        except BaseException:
            self.partial.unlink(missing_ok=True)
            raise
        self.partial.unlink()
        self.shards.append(
            {
                "path": str(self.final),
                "records": self.record_count,
                "bytes": self.final.stat().st_size,
                "sha256": sha256_path(self.final),
            }
        )
        self.stream = None
        self.partial = None
        self.final = None

    def close(self) -> None:
        self.close_shard()


def mount_unescape(value: bytes) -> bytes:
    return MOUNT_ESCAPE.sub(lambda match: bytes((int(match.group(1), 8),)), value)


def path_is_within(path: bytes, root: bytes) -> bool:
    return path == root or path.startswith(root.rstrip(b"/") + b"/")


def parse_mountinfo(data_root: bytes) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with open("/proc/self/mountinfo", "rb") as stream:
        for raw_line in stream:
            fields = raw_line.rstrip(b"\n").split()
            if b"-" not in fields or len(fields) < 10:
                continue
            separator = fields.index(b"-")
            if separator + 2 >= len(fields):
                continue
            mount_point = os.path.normpath(mount_unescape(fields[4]))
            raw_source = mount_unescape(fields[separator + 2])
            record = {
                "mount_id": fields[0].decode("ascii", errors="replace"),
                "parent_mount_id": fields[1].decode("ascii", errors="replace"),
                "major_minor": fields[2].decode("ascii", errors="replace"),
                "root": encoded_path(os.path.normpath(mount_unescape(fields[3]))),
                "mount_point": encoded_path(mount_point),
                "mount_point_raw": mount_point,
                "filesystem_type": fields[separator + 1].decode("ascii", errors="replace"),
                "source": {
                    "sha256": hashlib.sha256(raw_source).hexdigest(),
                    "byte_count": len(raw_source),
                    "retention_policy": (
                        "raw mount source suppressed because it can contain userinfo, query "
                        "parameters, signed endpoints, or site-private topology"
                    ),
                },
                "declared_below_data_root": path_is_within(mount_point, data_root),
            }
            records.append(record)
    governing = [
        item
        for item in records
        if path_is_within(data_root, item["mount_point_raw"])
    ]
    governing.sort(key=lambda item: len(item["mount_point_raw"]), reverse=True)
    nested = [
        item
        for item in records
        if item["mount_point_raw"] != data_root
        and path_is_within(item["mount_point_raw"], data_root)
    ]
    for item in records:
        item.pop("mount_point_raw", None)
    return {
        "governing_mount": governing[0] if governing else None,
        "declared_nested_mounts": nested,
        "all_mount_record_count": len(records),
    }


def parse_environment_entry(path: Path, environment_name: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    in_environments = False
    in_target = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "environments:":
            in_environments = True
            continue
        if not in_environments:
            continue
        if re.match(r"^  [^ ]+:", line):
            key = line.strip()[:-1]
            in_target = key == environment_name
            continue
        if in_target and re.match(r"^    [A-Za-z0-9_]+:", line):
            key, value = line.strip().split(":", 1)
            fields[key] = value.strip().strip('"').strip("'")
        elif in_target and line and not line.startswith("    "):
            break
    return fields


def reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in {key}")
        value[key] = item
    return value


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_pairs
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def validate_runtime_audit(job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("runtime audit job ID must be numeric")
    audit_dir = CODE_ROOT / "reports" / "environments" / "eeg2025" / "jobs" / job_id
    if (
        audit_dir.is_symlink()
        or not audit_dir.is_dir()
        or audit_dir.resolve(strict=True) != audit_dir
    ):
        raise ValueError(f"runtime audit directory is missing or unsafe: {audit_dir}")
    required = {
        "status": audit_dir / "status.json",
        "probe": audit_dir / "runtime_probe.json",
        "explicit": audit_dir / "conda-explicit.txt",
        "explicit_hash": audit_dir / "conda-explicit.sha256",
        "sanitization": audit_dir / "conda-explicit-sanitization.json",
        "allocation": audit_dir / "slurm_allocation.txt",
        "allocation_json": audit_dir / "slurm_allocation.json",
        "request_validation": audit_dir / "submission-request-validation.json",
        "pip": audit_dir / "pip-freeze.txt",
        "pip_hash": audit_dir / "pip-freeze.sha256",
        "pip_sanitization": audit_dir / "pip-freeze-sanitization.json",
    }
    for label, path in required.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime audit evidence is missing or unsafe ({label}): {path}")

    status_payload = load_json_object(required["status"])
    if (
        status_payload.get("schema_version") != 1
        or status_payload.get("job") != "audit_runtime"
        or str(status_payload.get("job_id")) != job_id
        or status_payload.get("environment_name") != "eeg2025"
        or status_payload.get("environment_path") != str(EEG_ENV)
        or status_payload.get("profile") != "cpu"
        or status_payload.get("state") != "completed"
        or status_payload.get("provenance_complete") is not True
        or status_payload.get("exit_code") != 0
    ):
        raise ValueError("runtime audit status is not a completed eeg2025 audit")

    probe = load_json_object(required["probe"])
    probe_python = probe.get("python")
    probe_environment = probe.get("environment")
    if (
        probe.get("schema_version") != 1
        or probe.get("environment_name") != "eeg2025"
        or probe.get("compatibility_status") != "compatible"
        or probe.get("critical_import_failures") != []
        or probe.get("compatibility_failures") != []
        or not isinstance(probe_python, dict)
        or probe_python.get("prefix") != str(EEG_ENV)
        or probe_python.get("expected_prefix") != str(EEG_ENV)
        or not isinstance(probe_environment, dict)
        or str(probe_environment.get("SLURM_JOB_ID")) != job_id
        or probe_environment.get("SLURM_JOB_PARTITION") != "CPU"
    ):
        raise ValueError("runtime probe does not verify eeg2025 compatibility")

    allocation = load_json_object(required["allocation_json"])
    allocation_fields = allocation.get("fields")
    if (
        str(allocation.get("job_id")) != job_id
        or not isinstance(allocation_fields, dict)
        or allocation_fields.get("JobId") != job_id
        or allocation_fields.get("Partition") != "CPU"
        or allocation_fields.get("Account") != "c2s"
        or allocation_fields.get("QOS") != "normal"
        or allocation_fields.get("NumCPUs") != "2"
        or not allocation_fields.get("NodeList")
        or not allocation_fields.get("AllocTRES")
        or allocation_fields.get("Comment")
        != f"denoiseNet:{status_payload.get('request_id')}"
    ):
        raise ValueError("runtime audit allocation JSON is incomplete or mismatched")

    sanitization = load_json_object(required["sanitization"])
    if (
        sanitization.get("schema_version") != 1
        or sanitization.get("return_code") != 0
        or sanitization.get("stdout_format") != "conda-explicit"
        or sanitization.get("stdout_format_valid") is not True
        or sanitization.get("stdout_requirement_count", 0) < 1
        or sanitization.get("stdout_structure_failures") != []
        or sanitization.get("replayability")
        not in {"verbatim_replayable", "redacted_non_replayable"}
    ):
        raise ValueError("runtime audit explicit-lock capture is incomplete or invalid")
    raw_stream_sha = sanitization.get("raw_stdout_sha256")
    if not isinstance(raw_stream_sha, str) or not HEX64.fullmatch(raw_stream_sha):
        raise ValueError("runtime audit lacks a valid sanitized-lock raw stream hash")

    pip_sanitization = load_json_object(required["pip_sanitization"])
    if (
        pip_sanitization.get("schema_version") != 1
        or pip_sanitization.get("return_code") != 0
        or pip_sanitization.get("stdout_format") != "pip-freeze"
        or pip_sanitization.get("stdout_format_valid") is not True
        or pip_sanitization.get("stdout_requirement_count", 0) < 1
        or pip_sanitization.get("stdout_structure_failures") != []
        or pip_sanitization.get("replayability")
        not in {"verbatim_replayable", "redacted_non_replayable"}
    ):
        raise ValueError("runtime audit pip-freeze capture is incomplete or invalid")
    raw_pip_stream_sha = pip_sanitization.get("raw_stdout_sha256")
    if not isinstance(raw_pip_stream_sha, str) or not HEX64.fullmatch(raw_pip_stream_sha):
        raise ValueError("runtime audit lacks a valid pip-freeze raw stream hash")

    recorded_hash_fields = required["explicit_hash"].read_text(encoding="utf-8").split()
    if not recorded_hash_fields or not HEX64.fullmatch(recorded_hash_fields[0]):
        raise ValueError("runtime audit explicit manifest hash is malformed")
    actual_explicit_hash = sha256_path(required["explicit"])
    if actual_explicit_hash != recorded_hash_fields[0]:
        raise ValueError("runtime audit explicit manifest hash does not match its content")
    pip_hash_fields = required["pip_hash"].read_text(encoding="utf-8").split()
    if not pip_hash_fields or not HEX64.fullmatch(pip_hash_fields[0]):
        raise ValueError("runtime audit pip-freeze manifest hash is malformed")
    actual_pip_hash = sha256_path(required["pip"])
    if actual_pip_hash != pip_hash_fields[0]:
        raise ValueError("runtime audit pip-freeze manifest hash does not match its content")

    request_validation = load_json_object(required["request_validation"])
    expected_audit_payload_sha = hashlib.sha256(b"--env\0eeg2025\0").hexdigest()
    if (
        request_validation.get("schema_version") != 1
        or request_validation.get("status") != "matched"
        or str(request_validation.get("job_id")) != job_id
        or request_validation.get("request_id") != status_payload.get("request_id")
        or request_validation.get("request_sha256") != status_payload.get("request_sha256")
        or request_validation.get("payload_arguments_sha256") != expected_audit_payload_sha
        or status_payload.get("payload_arguments_sha256") != expected_audit_payload_sha
    ):
        raise ValueError("runtime audit submission request evidence is incomplete or mismatched")

    current_provenance = {
        "cluster_config_sha256": sha256_path(CODE_ROOT / "configs/cluster/slurm.yaml"),
        "job_script_sha256": sha256_path(
            CODE_ROOT / "scripts/slurm/jobs/audit_runtime.sbatch"
        ),
        "submitter_sha256": sha256_path(CODE_ROOT / "scripts/slurm/submit.sh"),
        "contract_bundle_sha256": directory_bundle_sha256(
            CODE_ROOT / "scripts/contract", ".py"
        ),
        "slurm_jobs_bundle_sha256": directory_bundle_sha256(
            CODE_ROOT / "scripts/slurm/jobs", ".sbatch"
        ),
    }
    for field, expected_value in current_provenance.items():
        if status_payload.get(field) != expected_value:
            raise ValueError(f"runtime audit provenance is stale for {field}")

    environment_fields = parse_environment_entry(
        CODE_ROOT / "configs" / "environments.yaml", "eeg2025"
    )
    if environment_fields.get("path") != str(EEG_ENV):
        raise ValueError("registered eeg2025 path does not match the fixed environment")
    if environment_fields.get("audit_job_id") != job_id:
        raise ValueError("explicit runtime audit job is not the currently registered eeg2025 audit")
    if environment_fields.get("compatibility_status") != "compatible":
        raise ValueError("registered eeg2025 compatibility is not verified")
    if not environment_fields.get("responsibility_status", "").startswith("verified"):
        raise ValueError("registered eeg2025 responsibility is not verified")
    if environment_fields.get("explicit_manifest_sha256") != actual_explicit_hash:
        raise ValueError("registered eeg2025 explicit manifest hash does not match audit evidence")
    if environment_fields.get("pip_manifest_sha256") != actual_pip_hash:
        raise ValueError("registered eeg2025 pip manifest hash does not match audit evidence")
    if required["allocation"].stat().st_size == 0:
        raise ValueError("runtime audit allocation evidence is empty")

    return {
        "job_id": job_id,
        "environment_name": "eeg2025",
        "environment_path": str(EEG_ENV),
        "explicit_manifest_sha256": actual_explicit_hash,
        "status_sha256": sha256_path(required["status"]),
        "runtime_probe_sha256": sha256_path(required["probe"]),
        "sanitization_audit_sha256": sha256_path(required["sanitization"]),
        "allocation_sha256": sha256_path(required["allocation"]),
        "allocation_json_sha256": sha256_path(required["allocation_json"]),
        "request_validation_sha256": sha256_path(required["request_validation"]),
        "raw_explicit_stream_sha256": raw_stream_sha,
        "pip_freeze_sha256": actual_pip_hash,
        "pip_sanitization_audit_sha256": sha256_path(required["pip_sanitization"]),
        "raw_pip_freeze_stream_sha256": raw_pip_stream_sha,
        "validated_at_utc": utc_now(),
    }


def validate_submission_request(
    runtime_audit_job: str, current_job_id: str, expected_payload_sha256: str
) -> dict[str, Any]:
    request_id = os.environ.get("DENOISENET_REQUEST_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
        raise ValueError("submission request ID is absent or unsafe")
    request_path = (
        CODE_ROOT
        / "reports"
        / "slurm"
        / "submissions"
        / "requests"
        / f"{request_id}.json"
    )
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("submission request record is absent or unsafe")
    payload = load_json_object(request_path)
    expected_dependency = f"afterok:{runtime_audit_job}"
    if (
        payload.get("schema_version") != 1
        or payload.get("request_id") != request_id
        or payload.get("job") != "inventory_data"
        or payload.get("profile") != "cpu-high"
        or payload.get("partition") != "cpu-high"
        or payload.get("dependency") != expected_dependency
        or payload.get("array") != ""
        or payload.get("payload_argument_count") != 7
        or payload.get("payload_arguments_sha256") != expected_payload_sha256
    ):
        raise ValueError("submission request does not bind the frozen inventory invocation")
    submission_path = CODE_ROOT / "reports" / "slurm" / "submissions" / f"{current_job_id}.json"
    if submission_path.is_file() and not submission_path.is_symlink():
        submission = load_json_object(submission_path)
        if (
            str(submission.get("job_id")) != current_job_id
            or submission.get("request_id") != request_id
            or submission.get("dependency") != expected_dependency
            or submission.get("payload_arguments_sha256") != expected_payload_sha256
        ):
            raise ValueError("post-submit manifest does not match the inventory allocation")
    return {
        "request_id": request_id,
        "request_sha256": sha256_path(request_path),
        "dependency": expected_dependency,
        "payload_arguments_sha256": expected_payload_sha256,
    }


def classify_suffix(name: bytes) -> str:
    lowered = name.lower()
    for compound in (b".tar.gz", b".tar.bz2", b".tar.xz"):
        if lowered.endswith(compound):
            return compound.decode("ascii")
    if b".tar." in lowered:
        return ".tar*"
    suffix = os.path.splitext(lowered)[1]
    return suffix.decode("ascii", errors="replace") if suffix else "<none>"


def is_hashable_metadata(name: bytes, size: int) -> bool:
    if size < 0 or size > HASH_MAX_FILE_BYTES:
        return False
    lowered = name.lower()
    suffix = os.path.splitext(lowered)[1]
    return suffix in HASHABLE_METADATA_SUFFIXES or lowered.startswith(HASHABLE_METADATA_PREFIXES)


def candidate_matches(path: bytes) -> list[tuple[str, str]]:
    lowered = path.lower()
    matches: list[tuple[str, str]] = []
    for candidate_id, patterns in CANDIDATE_RULES.items():
        for pattern in patterns:
            if pattern in lowered:
                matches.append((candidate_id, pattern.decode("ascii", errors="replace")))
                break
    return matches


def classification_flags(path: bytes, name: bytes, kind: str, suffix: str) -> list[str]:
    lowered_path = path.lower()
    lowered_name = name.lower()
    parts = [part for part in lowered_path.split(b"/") if part]
    flags: list[str] = []
    if any(part.startswith(b".") for part in parts):
        flags.append("hidden")
    if any(
        part in {b"cache", b".cache", b"downloads", b"download", b"download_cache"}
        or part.endswith(b"_cache")
        for part in parts
    ):
        flags.append("cache_or_download_cache")
    if any(b".partial" in part or part in {b"partial", b"incomplete"} for part in parts):
        flags.append("partial_or_incomplete")
    if any(part in {b"old", b"backup", b"deprecated", b"legacy"} for part in parts):
        flags.append("old_or_legacy_marker")
    if suffix in REPORTED_SUFFIXES and suffix in {
        ".zip",
        ".tar",
        ".tgz",
        ".tar.gz",
        ".tar.bz2",
        ".tar.xz",
        ".tar*",
    }:
        flags.append("archive")
    raw_suffix = os.path.splitext(lowered_name)[1]
    if raw_suffix in CHECKSUM_SUFFIXES or lowered_name.startswith(b"checksum"):
        flags.append("checksum")
    if raw_suffix in MANIFEST_SUFFIXES or lowered_name.startswith(
        (b"manifest", b"dataset_description", b"data_dictionary")
    ):
        flags.append("manifest_or_dictionary")
    if lowered_name.startswith((b"readme", b"license", b"licence", b"copying")):
        flags.append("readme_or_license")
    if raw_suffix in DATA_SUFFIXES or (kind == "directory" and raw_suffix == b".zarr"):
        flags.append("data_container_or_file")
    if raw_suffix in CODE_SUFFIXES:
        flags.append("unexpected_code_in_data_root")
    return flags


def stat_payload(metadata: os.stat_result) -> dict[str, Any]:
    return {
        "mode_octal": format(stat.S_IMODE(metadata.st_mode), "04o"),
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "nlink": metadata.st_nlink,
    }


def kind_from_mode(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular_file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "unknown"


class InventoryScanner:
    def __init__(
        self,
        data_root: Path,
        output_root: Path,
        runtime_audit: dict[str, Any],
        guards: list[tuple[Path, str]],
        job_id: str,
        attempt: int,
        dynamic_state_start: dict[str, str],
    ) -> None:
        self.data_root = os.fsencode(os.fspath(data_root))
        self.output_root = output_root
        self.runtime_audit = runtime_audit
        self.guards = guards
        self.job_id = job_id
        self.attempt = attempt
        self.dynamic_state_start = dynamic_state_start
        self.started_at_utc = utc_now()
        self.started_monotonic = time.monotonic()
        self.stop_requested: str | None = None
        self.directory_entry_lstat_limiter = RateLimiter(
            DIRECTORY_ENTRY_LSTAT_OPS_PER_SECOND
        )
        self.shared_output_budget = {"bytes": 0}
        self.entries = JsonlShardWriter(output_root / "entries", "entries", self.shared_output_budget)
        self.errors = JsonlShardWriter(output_root / "errors", "errors", self.shared_output_budget)
        self.counts: Counter[str] = Counter()
        self.error_counts: Counter[str] = Counter()
        self.suffix_counts: Counter[str] = Counter()
        self.device_counts: Counter[int] = Counter()
        self.device_first_path: dict[int, dict[str, Any]] = {}
        self.hash_stats: Counter[str] = Counter()
        self.hash_bytes = 0
        self.candidate_hits: Counter[str] = Counter()
        self.candidate_evidence: dict[str, list[dict[str, Any]]] = {
            name: [] for name in CANDIDATE_RULES
        }
        self.partial_reasons: set[str] = set()
        self.mount_info: dict[str, Any] = {}
        self.mount_signature_start: str | None = None
        self.nested_mount_paths: set[bytes] = set()
        self.visited_mount_paths: set[bytes] = set()
        self.mount_boundary_counts: Counter[str] = Counter()
        self.mount_boundary_evidence: list[dict[str, Any]] = []
        self.active_directories: set[tuple[int, int]] = set()
        self.data_root_filesystem: dict[str, int] | None = None

    def request_stop(self, signal_name: str) -> None:
        self.stop_requested = signal_name

    def record_error(
        self,
        code: str,
        path: bytes | None,
        exc: BaseException | None = None,
        detail: str | None = None,
    ) -> None:
        self.error_counts[code] += 1
        self.partial_reasons.add(code)
        record: dict[str, Any] = {
            "record_type": "inventory_error",
            "code": code,
            "path": encoded_path(path) if path is not None else None,
        }
        if exc is not None:
            record["error_type"] = type(exc).__name__
            record["errno"] = getattr(exc, "errno", None)
            record["error"] = str(exc)
        if detail is not None:
            record["detail"] = detail
        try:
            self.errors.write(record)
        except OutputBudgetExceeded:
            self.partial_reasons.add("output_budget_exhausted")
            self.stop_requested = "output_budget_exhausted"

    def record_candidates(self, path: bytes, kind: str) -> list[str]:
        identifiers: list[str] = []
        for candidate_id, pattern in candidate_matches(path):
            identifiers.append(candidate_id)
            self.candidate_hits[candidate_id] += 1
            evidence = self.candidate_evidence[candidate_id]
            if len(evidence) < CANDIDATE_EVIDENCE_LIMIT:
                evidence.append(
                    {
                        "path": encoded_path(path),
                        "entry_kind": kind,
                        "matched_pattern": pattern,
                    }
                )
        return identifiers

    def hash_metadata_file(
        self,
        path: bytes,
        name: bytes,
        metadata: os.stat_result,
        parent_fd: int | None,
    ) -> dict[str, Any] | None:
        if not is_hashable_metadata(name, metadata.st_size):
            self.hash_stats["ineligible"] += 1
            return None
        if self.hash_bytes + metadata.st_size > HASH_TOTAL_BUDGET_BYTES:
            self.hash_stats["budget_skipped"] += 1
            return {"status": "skipped_budget", "algorithm": "sha256"}

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        open_target = name if parent_fd is not None else path
        open_kwargs = {"dir_fd": parent_fd} if parent_fd is not None else {}
        try:
            noatime_used = False
            if hasattr(os, "O_NOATIME"):
                try:
                    descriptor = os.open(open_target, flags | os.O_NOATIME, **open_kwargs)
                    noatime_used = True
                except OSError as exc:
                    if exc.errno not in (1, 13, 22):
                        raise
                    descriptor = os.open(open_target, flags, **open_kwargs)
            else:
                descriptor = os.open(open_target, flags, **open_kwargs)
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                opened = os.fstat(stream.fileno())
                if (
                    opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != metadata.st_size
                    or not stat.S_ISREG(opened.st_mode)
                ):
                    raise RuntimeError("file identity changed before hashing")
                digest = hashlib.sha256()
                bytes_read = 0
                remaining = opened.st_size
                while remaining:
                    if self.stop_requested:
                        raise RuntimeError("stop requested while hashing metadata file")
                    chunk_started = time.monotonic()
                    chunk = stream.read(min(HASH_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise RuntimeError("file ended before its initial size was read")
                    digest.update(chunk)
                    bytes_read += len(chunk)
                    remaining -= len(chunk)
                    self.hash_bytes += len(chunk)
                    self.hash_stats["attempted_bytes"] += len(chunk)
                    minimum_elapsed = len(chunk) / HASH_BYTES_PER_SECOND
                    chunk_elapsed = time.monotonic() - chunk_started
                    if minimum_elapsed > chunk_elapsed:
                        time.sleep(minimum_elapsed - chunk_elapsed)
                after = os.fstat(stream.fileno())
                if (
                    after.st_size != opened.st_size
                    or after.st_mtime_ns != opened.st_mtime_ns
                    or after.st_ctime_ns != opened.st_ctime_ns
                ):
                    raise RuntimeError("file changed while hashing")
            self.hash_stats["hashed"] += 1
            self.hash_stats["hashed_bytes"] += bytes_read
            return {
                "status": "hashed",
                "algorithm": "sha256",
                "digest": digest.hexdigest(),
                "bytes_read": bytes_read,
                "o_noatime_used": noatime_used,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            self.hash_stats["errors"] += 1
            self.record_error("metadata_hash_error", path, exc)
            return {
                "status": "error",
                "algorithm": "sha256",
                "error_type": type(exc).__name__,
            }

    def record_entry(
        self,
        path: bytes,
        name: bytes,
        metadata: os.stat_result,
        parent_device: int | None,
        parent_fd: int | None = None,
    ) -> bool:
        kind = kind_from_mode(metadata.st_mode)
        self.counts["entries"] += 1
        self.counts[f"kind_{kind}"] += 1
        self.device_counts[metadata.st_dev] += 1
        self.device_first_path.setdefault(metadata.st_dev, encoded_path(path))
        suffix = classify_suffix(name)
        if suffix in REPORTED_SUFFIXES:
            self.suffix_counts[suffix] += 1
        else:
            self.suffix_counts["other_or_none"] += 1
        if stat.S_ISREG(metadata.st_mode):
            self.counts["regular_file_bytes"] += metadata.st_size

        candidates = self.record_candidates(path, kind)
        flags = classification_flags(path, name, kind, suffix)
        for flag in flags:
            self.counts[f"flag_{flag}"] += 1
        record: dict[str, Any] = {
            "record_type": "filesystem_entry",
            "path": encoded_path(path),
            "name": encoded_bytes(name),
            "kind": kind,
            "lstat": stat_payload(metadata),
            "candidate_ids": candidates,
            "classification_flags": flags,
            "symlink_followed": False,
        }
        if stat.S_ISREG(metadata.st_mode):
            try:
                if parent_fd is None:
                    readable = os.access(path, os.R_OK, follow_symlinks=False)
                else:
                    readable = os.access(
                        name,
                        os.R_OK,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
            except (NotImplementedError, OSError):
                readable = None
            record["readable"] = readable
            if readable is True:
                self.counts["regular_files_reported_readable"] += 1
            elif readable is False:
                self.counts["regular_files_reported_unreadable"] += 1
            else:
                self.counts["regular_files_readability_unknown"] += 1
            record["content_hash"] = self.hash_metadata_file(
                path, name, metadata, parent_fd
            )
        elif stat.S_ISLNK(metadata.st_mode):
            try:
                if parent_fd is None:
                    target = os.readlink(path)
                else:
                    target = os.readlink(name, dir_fd=parent_fd)
                if isinstance(target, str):
                    target = os.fsencode(target)
                if os.path.isabs(target):
                    lexical_target = os.path.normpath(target)
                else:
                    lexical_target = os.path.normpath(os.path.join(os.path.dirname(path), target))
                record["symlink"] = {
                    "target": encoded_bytes(target),
                    "lexically_resolved_target": encoded_path(lexical_target),
                    "lexical_target_inside_data_root": path_is_within(
                        lexical_target, self.data_root
                    ),
                    "target_stat_attempted": False,
                }
            except OSError as exc:
                record["symlink"] = {"target_read_error": type(exc).__name__}
                self.record_error("symlink_target_read_error", path, exc)

        if parent_device is not None and metadata.st_dev != parent_device:
            key = f"{parent_device}->{metadata.st_dev}"
            self.mount_boundary_counts[key] += 1
            if len(self.mount_boundary_evidence) < 500:
                self.mount_boundary_evidence.append(
                    {
                        "path": encoded_path(path),
                        "parent_device": parent_device,
                        "entry_device": metadata.st_dev,
                    }
                )
            if path not in self.nested_mount_paths:
                self.record_error(
                    "undeclared_device_boundary",
                    path,
                    detail="entry device differs from its parent but was not declared as a nested mount",
                )

        try:
            self.entries.write(record)
        except OutputBudgetExceeded as exc:
            self.partial_reasons.add("output_budget_exhausted")
            self.stop_requested = "output_budget_exhausted"
            self.record_error("output_budget_exhausted", path, exc)
            return False
        return True

    def load_mount_evidence(self) -> None:
        try:
            self.mount_info = parse_mountinfo(self.data_root)
            if self.mount_info.get("governing_mount") is None:
                self.record_error(
                    "governing_mount_unresolved",
                    self.data_root,
                    detail="no enclosing mount entry was resolved from /proc/self/mountinfo",
                )
            for item in self.mount_info["declared_nested_mounts"]:
                raw = base64.b64decode(item["mount_point"]["raw_base64"])
                self.nested_mount_paths.add(raw)
            self.mount_signature_start = mount_signature(self.mount_info)
        except (OSError, ValueError) as exc:
            self.mount_info = {
                "governing_mount": None,
                "declared_nested_mounts": [],
                "mountinfo_error": f"{type(exc).__name__}: {exc}",
            }
            self.record_error("mountinfo_unavailable", None, exc)
            self.mount_signature_start = mount_signature(self.mount_info)

    def scan(self) -> None:
        self.load_mount_evidence()
        self.directory_entry_lstat_limiter.wait()
        try:
            root_metadata = os.lstat(self.data_root)
        except OSError as exc:
            self.record_error("data_root_lstat_error", self.data_root, exc)
            return
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            self.record_error(
                "unsafe_data_root",
                self.data_root,
                detail="fixed data root must be a real directory, not a symlink",
            )
            return
        if os.path.realpath(self.data_root) != self.data_root:
            self.record_error(
                "unsafe_data_root_resolution",
                self.data_root,
                detail="fixed data root or one of its ancestors resolves through a symlink",
            )
            return
        try:
            filesystem = os.statvfs(self.data_root)
            self.data_root_filesystem = {
                "block_size": filesystem.f_frsize,
                "blocks_total": filesystem.f_blocks,
                "blocks_available_to_caller": filesystem.f_bavail,
                "bytes_total": filesystem.f_frsize * filesystem.f_blocks,
                "bytes_available_to_caller": filesystem.f_frsize * filesystem.f_bavail,
                "files_total": filesystem.f_files,
                "files_available_to_caller": filesystem.f_favail,
            }
        except OSError as exc:
            self.record_error("data_root_statvfs_error", self.data_root, exc)

        self.record_entry(
            self.data_root,
            os.path.basename(self.data_root),
            root_metadata,
            None,
        )
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC

        frames: list[dict[str, Any]] = []

        def close_frame(frame: dict[str, Any]) -> None:
            iterator = frame["iterator"]
            descriptor = frame["fd"]
            try:
                after_fd = os.fstat(descriptor)
                iterator.close()
                parent_fd = frame["parent_fd"]
                if parent_fd is None:
                    after_path = os.lstat(frame["path"])
                else:
                    after_path = os.stat(
                        frame["name"], dir_fd=parent_fd, follow_symlinks=False
                    )
                before = frame["metadata"]
                if (
                    after_fd.st_dev != before.st_dev
                    or after_fd.st_ino != before.st_ino
                    or after_path.st_dev != before.st_dev
                    or after_path.st_ino != before.st_ino
                    or after_fd.st_mtime_ns != before.st_mtime_ns
                    or after_fd.st_ctime_ns != before.st_ctime_ns
                ):
                    self.record_error(
                        "directory_changed_during_scan",
                        frame["path"],
                        detail="opened directory or its parent-relative name changed during scan",
                    )
            except OSError as exc:
                self.record_error("directory_postscan_lstat_error", frame["path"], exc)
            finally:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    self.record_error("directory_close_error", frame["path"], exc)
                self.active_directories.discard(frame["key"])

        def open_frame(
            path: bytes,
            name: bytes,
            expected: os.stat_result,
            parent_fd: int | None,
        ) -> dict[str, Any] | None:
            self.counts["directories_open_attempted"] += 1
            descriptor: int | None = None
            try:
                if parent_fd is None:
                    descriptor = os.open(path, directory_flags)
                else:
                    descriptor = os.open(name, directory_flags, dir_fd=parent_fd)
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or opened.st_dev != expected.st_dev
                    or opened.st_ino != expected.st_ino
                    or opened.st_mtime_ns != expected.st_mtime_ns
                    or opened.st_ctime_ns != expected.st_ctime_ns
                ):
                    os.close(descriptor)
                    self.record_error(
                        "directory_open_identity_changed",
                        path,
                        detail="O_NOFOLLOW directory descriptor differs from discovery metadata",
                    )
                    return None
                key = (opened.st_dev, opened.st_ino)
                if key in self.active_directories:
                    os.close(descriptor)
                    self.record_error(
                        "directory_cycle_skipped",
                        path,
                        detail="active ancestor device/inode encountered again",
                    )
                    return None
                iterator = os.scandir(descriptor)
            except OSError as exc:
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                self.record_error("directory_open_or_scandir_error", path, exc)
                return None
            self.active_directories.add(key)
            self.counts["directories_opened"] += 1
            if path in self.nested_mount_paths:
                self.visited_mount_paths.add(path)
            return {
                "path": path,
                "name": name,
                "metadata": opened,
                "parent_fd": parent_fd,
                "fd": descriptor,
                "iterator": iterator,
                "key": key,
            }

        root_frame = open_frame(
            self.data_root, os.path.basename(self.data_root), root_metadata, None
        )
        if root_frame is not None:
            frames.append(root_frame)

        while frames and not self.stop_requested:
            frame = frames[-1]
            try:
                entry = next(frame["iterator"])
            except StopIteration:
                frames.pop()
                close_frame(frame)
                continue
            except OSError as exc:
                self.record_error("directory_scandir_error", frame["path"], exc)
                frames.pop()
                close_frame(frame)
                continue

            entry_name = entry.name
            if isinstance(entry_name, str):
                entry_name = os.fsencode(entry_name)
            entry_path = os.path.normpath(os.path.join(frame["path"], entry_name))
            if not path_is_within(entry_path, self.data_root):
                self.record_error(
                    "entry_path_escape",
                    entry_path,
                    detail="directory entry name escaped the fixed data root lexically",
                )
                continue
            self.directory_entry_lstat_limiter.wait()
            try:
                metadata = os.stat(
                    entry_name, dir_fd=frame["fd"], follow_symlinks=False
                )
            except OSError as exc:
                self.record_candidates(entry_path, "unstatable")
                self.record_error("entry_lstat_error", entry_path, exc)
                continue
            if not self.record_entry(
                entry_path,
                entry_name,
                metadata,
                frame["metadata"].st_dev,
                frame["fd"],
            ):
                break
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                child = open_frame(entry_path, entry_name, metadata, frame["fd"])
                if child is not None:
                    frames.append(child)

        while frames:
            close_frame(frames.pop())

        if self.stop_requested:
            self.partial_reasons.add("scan_interrupted")
            self.record_error(
                "scan_interrupted",
                None,
                detail=f"stop requested: {self.stop_requested}",
            )
        missing_mounts = self.nested_mount_paths - self.visited_mount_paths
        for mount_path in sorted(missing_mounts):
            self.record_error(
                "declared_mount_not_visited",
                mount_path,
                detail="mountinfo declares a mount below data root that traversal did not open",
            )

    def finalize(
        self,
        guard_end: list[dict[str, Any]],
        stale_guards: list[str],
        *,
        forced_failure: bool,
    ) -> tuple[str, int]:
        stale_input_detected = bool(stale_guards)
        for stale in stale_guards:
            self.partial_reasons.add("stale_input")
            self.record_error("stale_input", None, detail=stale)
        try:
            dynamic_state_end = capture_dynamic_state()
            if dynamic_state_end != self.dynamic_state_start:
                stale_input_detected = True
                self.partial_reasons.add("stale_input")
                self.record_error(
                    "dynamic_input_changed",
                    None,
                    detail="code, Git worktree, or eeg2025 explicit state changed during inventory",
                )
        except (OSError, RuntimeError, ValueError) as exc:
            dynamic_state_end = {"capture_error": type(exc).__name__}
            stale_input_detected = True
            self.partial_reasons.add("stale_input")
            self.record_error("dynamic_state_capture_error", None, exc)
        try:
            mount_info_end = parse_mountinfo(self.data_root)
            mount_signature_end = mount_signature(mount_info_end)
            if mount_signature_end != self.mount_signature_start:
                self.record_error(
                    "mount_topology_changed",
                    None,
                    detail="governing or nested data-root mount topology changed during inventory",
                )
        except (OSError, ValueError) as exc:
            mount_signature_end = None
            self.record_error("mountinfo_end_unavailable", None, exc)
        self.entries.close()
        self.errors.close()

        state = "PARTIAL" if self.partial_reasons else "COMPLETE"
        candidate_rows: list[dict[str, Any]] = []
        for candidate_id in CANDIDATE_RULES:
            hits = self.candidate_hits[candidate_id]
            if hits:
                phase1_status = "present_unverified"
            elif state == "COMPLETE":
                phase1_status = "unknown"
            else:
                phase1_status = "unknown"
            candidate_rows.append(
                {
                    "dataset_id": candidate_id,
                    "source_anchor_for_phase2_verification": CANDIDATE_SOURCE_ANCHORS[
                        candidate_id
                    ],
                    "phase1_status": phase1_status,
                    "evidence_hit_count": hits,
                    "evidence_retained_count": len(self.candidate_evidence[candidate_id]),
                    "evidence_truncated": hits > len(self.candidate_evidence[candidate_id]),
                    "evidence": self.candidate_evidence[candidate_id],
                    "semantic_limit": (
                        "Phase I filename/path evidence only; version, license, integrity, fields, "
                        "and sample readability remain unverified"
                    ),
                }
            )

        nested_mount_rows = self.mount_info.get("declared_nested_mounts", [])
        visited_b64 = {
            base64.b64encode(path).decode("ascii") for path in self.visited_mount_paths
        }
        for row in nested_mount_rows:
            row["visited_as_directory"] = row["mount_point"]["raw_base64"] in visited_b64
        mounts_payload = {
            "schema_version": SCHEMA_VERSION,
            "data_root": encoded_path(self.data_root),
            "governing_mount": self.mount_info.get("governing_mount"),
            "declared_nested_mounts": nested_mount_rows,
            "mountinfo_error": self.mount_info.get("mountinfo_error"),
            "observed_devices": [
                {
                    "device": device,
                    "entry_count": count,
                    "first_path": self.device_first_path[device],
                }
                for device, count in sorted(self.device_counts.items())
            ],
            "device_boundary_counts": dict(sorted(self.mount_boundary_counts.items())),
            "device_boundary_evidence": self.mount_boundary_evidence,
        }
        coverage_payload = {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "partial_reasons": sorted(self.partial_reasons),
            "dynamic_state_at_start": self.dynamic_state_start,
            "dynamic_state_at_completion": dynamic_state_end,
            "mount_signature_at_start": self.mount_signature_start,
            "mount_signature_at_completion": mount_signature_end,
            "counts": dict(sorted(self.counts.items())),
            "error_counts": dict(sorted(self.error_counts.items())),
            "common_suffix_counts": dict(sorted(self.suffix_counts.items())),
            "hash_coverage": {
                "policy": "sha256 only for allowlisted metadata names/suffixes within size and total budgets",
                "max_file_bytes": HASH_MAX_FILE_BYTES,
                "total_budget_bytes": HASH_TOTAL_BUDGET_BYTES,
                "rate_limit_bytes_per_second": HASH_BYTES_PER_SECOND,
                "statistics": dict(sorted(self.hash_stats.items())),
            },
            "directory_entry_lstat_rate_limit_ops_per_second": (
                DIRECTORY_ENTRY_LSTAT_OPS_PER_SECOND
            ),
            "metadata_rate_limit_scope": (
                "paces root/entry lstat operations only; fstat, readlink, statvfs, mountinfo, "
                "and directory iteration are separately bounded by the single-process walk"
            ),
            "data_root_filesystem": self.data_root_filesystem,
            "shard_output_bytes": self.shared_output_budget["bytes"],
            "shard_output_budget_bytes": SHARD_OUTPUT_BUDGET_BYTES,
            "symlink_policy": "lstat and readlink only; symlink targets are never stat'ed or traversed",
            "write_policy": (
                "all program-created paths are below code_root/reports/data_inventory; "
                "no data-root write API is used"
            ),
            "atime_limit": (
                "metadata hashes request O_NOATIME when supported; permission/filesystem fallback "
                "uses a normal read-only open and can update access time under the mount policy"
            ),
            "mount_signature_limit": (
                "the start/end signature covers governing and nested mount identity, paths, "
                "filesystem type, and redacted-source hash; it does not prove mount-option or "
                "server-side snapshot stability"
            ),
        }
        candidates_payload = {
            "schema_version": SCHEMA_VERSION,
            "inventory_state": state,
            "phase1_status_rules": {
                "hit": "present_unverified",
                "no_hit_complete_scan": (
                    "unknown; absence of weak path/name evidence is insufficient for a "
                    "dataset-level missing determination"
                ),
                "no_hit_partial_scan": "unknown",
                "verified_available": "forbidden_in_phase1",
            },
            "candidates": candidate_rows,
        }
        atomic_write_json(self.output_root / "coverage.json", coverage_payload)
        atomic_write_json(self.output_root / "mounts.json", mounts_payload)
        atomic_write_json(self.output_root / "candidates.json", candidates_payload)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "inventory_phase": "phase1_full_root_read_only_discovery",
            "state": state,
            "job_id": self.job_id,
            "attempt": self.attempt,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - self.started_monotonic,
            "code_root": str(CODE_ROOT),
            "data_root": str(DATA_ROOT),
            "output_root": str(self.output_root),
            "runtime_audit": self.runtime_audit,
            "input_guards_at_completion": guard_end,
            "stale_guard_failures": stale_guards,
            "partial_reasons": sorted(self.partial_reasons),
            "entry_shards": self.entries.shards,
            "error_shards": self.errors.shards,
            "coverage_path": str(self.output_root / "coverage.json"),
            "mounts_path": str(self.output_root / "mounts.json"),
            "candidates_path": str(self.output_root / "candidates.json"),
            "scientific_use": "none; Phase I evidence cannot establish dataset usability",
            "snapshot_consistency": (
                "non-atomic namespace walk; COMPLETE means no observed coverage error, not a "
                "filesystem snapshot or content-integrity proof"
            ),
        }
        manifest_path = self.output_root / "inventory_manifest.json"
        atomic_write_json(manifest_path, manifest)
        manifest_sha = sha256_path(manifest_path)
        if forced_failure:
            process_exit_code = 1
        elif stale_input_detected:
            process_exit_code = 75
        else:
            process_exit_code = 0 if state == "COMPLETE" else 4
        status = {
            "schema_version": SCHEMA_VERSION,
            "phase": "scan",
            "state": state,
            "job_id": self.job_id,
            "attempt": self.attempt,
            "inventory_manifest": str(manifest_path),
            "inventory_manifest_sha256": manifest_sha,
            "partial_reasons": sorted(self.partial_reasons),
            "process_exit_code": process_exit_code,
            "generated_at_utc": utc_now(),
        }
        atomic_write_json(self.output_root / "status.json", status)
        atomic_write_json(self.output_root / state, status)
        atomic_write_bytes(
            self.output_root / "process_exit_code.txt",
            f"{process_exit_code}\n".encode("ascii"),
        )
        return state, process_exit_code


def validate_guards(guards: Iterable[tuple[Path, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    observed: list[dict[str, Any]] = []
    failures: list[str] = []
    code_real = CODE_ROOT.resolve(strict=True)
    for path, expected in guards:
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            common = Path(os.path.commonpath((os.fspath(code_real), os.fspath(absolute))))
        except ValueError:
            common = Path("/")
        try:
            resolved = absolute.resolve(strict=True)
        except OSError:
            resolved = Path("/")
        if (
            common != code_real
            or resolved != absolute
            or absolute.is_symlink()
            or not absolute.is_file()
        ):
            failures.append(f"unsafe or missing guard path: {absolute}")
            observed.append({"path": str(absolute), "expected_sha256": expected, "status": "unsafe"})
            continue
        actual = sha256_path(absolute)
        status_value = "matched" if actual == expected else "changed"
        observed.append(
            {
                "path": str(absolute),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status_value,
            }
        )
        if status_value != "matched":
            failures.append(f"guard changed: {absolute}")
    return observed, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-audit-job", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--read-only", action="store_true", required=True)
    parser.add_argument("--guard", action="append", nargs=2, metavar=("PATH", "SHA256"), default=[])
    args = parser.parse_args()

    if args.data_root != DATA_ROOT:
        parser.error(f"data root must be exactly {DATA_ROOT}")
    if not args.read_only:
        parser.error("--read-only is mandatory")
    if not re.fullmatch(r"[0-9]+", args.job_id):
        parser.error("job ID must be numeric")
    if os.environ.get("SLURM_JOB_ID") != args.job_id:
        parser.error("job ID does not match the Slurm allocation")
    if os.environ.get("DENOISENET_PROFILE") != "cpu-high":
        parser.error("inventory requires the registered cpu-high profile")
    if args.attempt < 0:
        parser.error("attempt must be nonnegative")
    try:
        require_safe_filesystem_primitives()
    except RuntimeError as exc:
        parser.error(str(exc))

    expected_output = (
        INVENTORY_PARENT / "jobs" / args.job_id / f"attempt-{args.attempt}" / "scan"
    )
    if Path(os.path.abspath(os.fspath(args.output_root))) != expected_output:
        parser.error(f"output root must be exactly {expected_output}")
    ensure_safe_output_directory(expected_output)
    if any(expected_output.iterdir()):
        parser.error("scan output directory must be empty")

    guard_pairs = [(Path(path), expected) for path, expected in args.guard]
    for _, expected in guard_pairs:
        if not HEX64.fullmatch(expected):
            parser.error("guard hashes must be lowercase SHA-256 values")
    guard_start, stale_start = validate_guards(guard_pairs)
    atomic_write_json(
        expected_output / "started.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": "RUNNING",
            "job_id": args.job_id,
            "attempt": args.attempt,
            "started_at_utc": utc_now(),
            "guards_at_start": guard_start,
        },
    )
    if stale_start:
        terminal = {
            "schema_version": SCHEMA_VERSION,
            "phase": "preflight",
            "state": "PARTIAL",
            "job_id": args.job_id,
            "attempt": args.attempt,
            "partial_reasons": ["stale_input_before_scan"],
            "details": stale_start,
            "process_exit_code": 75,
            "generated_at_utc": utc_now(),
        }
        atomic_write_json(expected_output / "status.json", terminal)
        atomic_write_json(expected_output / "PARTIAL", terminal)
        atomic_write_bytes(expected_output / "process_exit_code.txt", b"75\n")
        return 75

    try:
        runtime_audit = validate_runtime_audit(args.runtime_audit_job)
        runtime_audit["submission_request"] = validate_submission_request(
            args.runtime_audit_job,
            args.job_id,
            os.environ.get("DENOISENET_PAYLOAD_ARGS_SHA256", ""),
        )
        dynamic_state_start = capture_dynamic_state()
        if (
            dynamic_state_start["contract_bundle_sha256"]
            != os.environ.get("DENOISENET_CONTRACT_BUNDLE_SHA256")
            or dynamic_state_start["slurm_jobs_bundle_sha256"]
            != os.environ.get("DENOISENET_SLURM_JOBS_BUNDLE_SHA256")
            or dynamic_state_start["environment_explicit_raw_sha256"]
            != runtime_audit["raw_explicit_stream_sha256"]
            or dynamic_state_start["environment_pip_freeze_raw_sha256"]
            != runtime_audit["raw_pip_freeze_stream_sha256"]
        ):
            raise ValueError("dynamic code or eeg2025 state differs from frozen audit evidence")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "phase": "preflight",
            "state": "PARTIAL",
            "job_id": args.job_id,
            "attempt": args.attempt,
            "partial_reasons": ["runtime_audit_validation_failed"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "process_exit_code": 4,
            "generated_at_utc": utc_now(),
        }
        atomic_write_json(expected_output / "status.json", failure)
        atomic_write_json(expected_output / "PARTIAL", failure)
        atomic_write_bytes(expected_output / "process_exit_code.txt", b"4\n")
        return 4

    scanner = InventoryScanner(
        DATA_ROOT,
        expected_output,
        runtime_audit,
        guard_pairs,
        args.job_id,
        args.attempt,
        dynamic_state_start,
    )

    def stop_handler(signum: int, _frame: Any) -> None:
        scanner.request_stop(signal.Signals(signum).name)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, stop_handler)

    unexpected: BaseException | None = None
    try:
        scanner.scan()
    except BaseException as exc:  # preserve partial evidence before failing the job
        unexpected = exc
        scanner.record_error("unhandled_scanner_error", None, exc)
    guard_end, stale_end = validate_guards(guard_pairs)
    state, exit_code = scanner.finalize(
        guard_end, stale_end, forced_failure=unexpected is not None
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
