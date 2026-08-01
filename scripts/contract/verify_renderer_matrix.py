#!/usr/bin/env python3
"""Verify the bounded renderer factor screen and its positive control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from runtime_probe import IMPORTS as RUNTIME_IMPORTS


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
ICML_ENV = Path("/home/infres/yinwang/anaconda3/envs/icml")
MAX_RECORD_BYTES = 1024 * 1024
EXPECTED_PYMUPDF_VERSION = "1.26.5"
HEX64 = re.compile(r"[0-9a-f]{64}")
EXPECTED_MODULES = [
    module_name for module_name in RUNTIME_IMPORTS["icml"] if module_name != "fitz"
]


class MatrixValidationError(RuntimeError):
    """A fixed renderer-matrix validation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixValidationError(message)


def regular_file(path: Path, parent: Path, maximum_bytes: int) -> os.stat_result:
    require(path.parent == parent, "renderer evidence path leaves its job directory")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), "renderer evidence is not a regular file")
    require(metadata.st_size <= maximum_bytes, "renderer evidence exceeds its byte budget")
    return metadata


def load_json(path: Path, parent: Path) -> dict[str, object]:
    regular_file(path, parent, MAX_RECORD_BYTES)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "renderer JSON is not an object")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def publish_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("renderer verification write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def valid_torch_audit(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    operation = value.get("cuda_operation")
    names = value.get("cuda_device_names")
    operation_sum = operation.get("sum") if isinstance(operation, dict) else None
    return (
        value.get("cuda_available") is True
        and value.get("cuda_device_count") == 1
        and isinstance(value.get("compiled_cuda_version"), str)
        and bool(value.get("compiled_cuda_version"))
        and type(value.get("cudnn_version")) is int
        and value.get("cudnn_version", 0) > 0
        and isinstance(names, list)
        and len(names) == 1
        and "L40S" in str(names[0])
        and isinstance(operation, dict)
        and operation.get("status") == "ok"
        and type(operation_sum) in {int, float}
        and float(operation_sum) == 4.0
    )


def validate_positive_record(
    record: dict[str, object], *, job_id: str, replicate: int, status: str
) -> None:
    expected_common = {
        "schema_version": 2,
        "job_id": job_id,
        "mode": "conda_run",
        "preload_plan": "runtime_full_cuda",
        "warnings_policy": "default",
        "replicate": replicate,
        "cuda_warmup": True,
        "runtime_equivalent_preload": True,
        "status": status,
        "python_executable": str(ICML_ENV / "bin" / "python"),
        "python_prefix": str(ICML_ENV),
        "conda_prefix": str(ICML_ENV),
        "conda_default_env": "icml",
        "pythonwarnings_environment": None,
        "pythonhome_environment": None,
        "pythonpath_environment": None,
        "python_warnoptions": [],
    }
    for field, expected in expected_common.items():
        require(record.get(field) == expected, f"positive record field {field} differs")
    require(record.get("preloaded_modules") == EXPECTED_MODULES, "full preload order differs")
    versions = record.get("preload_versions")
    require(isinstance(versions, dict), "full preload versions are absent")
    require(set(versions) == set(EXPECTED_MODULES), "full preload version set differs")
    require(
        all(
            isinstance(versions.get(name), str) and versions.get(name)
            for name in EXPECTED_MODULES
        ),
        "full preload version value is absent",
    )
    require(valid_torch_audit(record.get("runtime_torch_audit")), "CUDA evidence differs")
    if status == "import_ok":
        require(
            record.get("pymupdf_version") == EXPECTED_PYMUPDF_VERSION,
            "registered PyMuPDF version differs",
        )


def validate_evidence_manifest(
    run_dir: Path, manifest: Path, expected_files: set[Path]
) -> str:
    regular_file(manifest, run_dir, MAX_RECORD_BYTES)
    actual_files = {
        path for path in run_dir.glob("renderer-import-*") if path.parent == run_dir
    }
    require(actual_files, "renderer evidence file set is empty")
    require(actual_files == expected_files, "renderer evidence filename set differs")
    require(
        all(path.is_file() and not path.is_symlink() for path in actual_files),
        "renderer evidence contains a non-regular path",
    )
    require(
        not any(path.name.endswith(".partial") for path in actual_files),
        "renderer evidence contains an incomplete atomic file",
    )
    recorded: dict[Path, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        require(
            len(fields) == 2 and HEX64.fullmatch(fields[0]) is not None,
            "bad evidence hash line",
        )
        path = Path(fields[1])
        require(path.parent == run_dir and path in actual_files, "evidence hash path differs")
        require(path not in recorded, "duplicate renderer evidence hash path")
        recorded[path] = fields[0]
    require(set(recorded) == actual_files, "renderer evidence hash set is incomplete")
    for path, expected in recorded.items():
        require(sha256_path(path) == expected, "renderer evidence hash mismatch")
    return sha256_path(manifest)


def validate_matrix(run_dir: Path, job_id: str) -> tuple[str, int]:
    comparison = load_json(run_dir / "renderer-import-comparison.json", run_dir)
    require(comparison.get("schema_version") == 2, "comparison schema differs")
    require(comparison.get("job_id") == job_id, "comparison job differs")
    require(comparison.get("core_dump_limit") == 0, "comparison core limit differs")
    require(
        comparison.get("capture_per_stream_limit_bytes") == MAX_RECORD_BYTES,
        "comparison capture budget differs",
    )
    require(
        comparison.get("capture_aggregate_upper_bound_bytes") == 32 * MAX_RECORD_BYTES,
        "comparison aggregate capture budget differs",
    )
    require(comparison.get("expected_cell_executions") == 16, "comparison cell count differs")
    require(comparison.get("replicates_per_plan") == 2, "comparison replicate count differs")
    require(comparison.get("matrix_setup_failed") is False, "matrix setup failed")
    require(
        comparison.get("positive_control")
        == {
            "launcher": "conda_run",
            "preload": "runtime_full_cuda",
            "warnings_policy": "default",
            "required_replicates": [1, 2],
        },
        "comparison positive control differs",
    )
    results = comparison.get("results")
    require(
        isinstance(results, list) and len(results) == 16,
        "comparison results are incomplete",
    )
    expected_cells = set(
        product(
            ("direct", "conda_run"),
            ("none", "runtime_full_cuda"),
            ("default", "error"),
            (1, 2),
        )
    )
    observed_cells: set[tuple[object, object, object, object]] = set()
    expected_evidence_files = {run_dir / "renderer-import-comparison.json"}
    for result in results:
        require(isinstance(result, dict), "comparison result is not an object")
        cell = (
            result.get("launcher"),
            result.get("preload"),
            result.get("warnings_policy"),
            result.get("replicate"),
        )
        require(cell in expected_cells and cell not in observed_cells, "comparison cell differs")
        observed_cells.add(cell)
        require(type(result.get("exit_code")) is int, "comparison exit code differs")
        require(type(result.get("preimport_ready")) is bool, "comparison marker flag differs")
        require(type(result.get("result_record")) is bool, "comparison result flag differs")
        for field in ("stdout_bytes", "stderr_bytes"):
            value = result.get(field)
            require(
                type(value) is int and 0 <= value <= MAX_RECORD_BYTES,
                "stream size differs",
            )
        launcher, preload, warnings_policy, replicate = cell
        stem = (
            f"renderer-import-{launcher}-{preload}-"
            f"warnings_{warnings_policy}-r{replicate}"
        )
        for suffix, size_field in (("stdout", "stdout_bytes"), ("stderr", "stderr_bytes")):
            stream_path = run_dir / f"{stem}.{suffix}"
            expected_evidence_files.add(stream_path)
            metadata = regular_file(stream_path, run_dir, MAX_RECORD_BYTES)
            require(metadata.st_size == result.get(size_field), "captured stream size mismatch")
        marker_path = run_dir / f"{stem}.preimport.json"
        result_path = run_dir / f"{stem}.json"
        if result.get("preimport_ready") is True:
            expected_evidence_files.add(marker_path)
        if result.get("result_record") is True:
            expected_evidence_files.add(result_path)
        require(marker_path.exists() is result.get("preimport_ready"), "marker flag mismatch")
        require(result_path.exists() is result.get("result_record"), "result flag mismatch")
        if (
            launcher == "conda_run"
            and preload == "runtime_full_cuda"
            and warnings_policy == "default"
        ):
            require(result.get("exit_code") == 0, "positive-control process failed")
            require(result.get("preimport_ready") is True, "positive-control marker is absent")
            require(result.get("result_record") is True, "positive-control result is absent")
            validate_positive_record(
                load_json(marker_path, run_dir),
                job_id=job_id,
                replicate=int(replicate),
                status="preimport_ready",
            )
            validate_positive_record(
                load_json(result_path, run_dir),
                job_id=job_id,
                replicate=int(replicate),
                status="import_ok",
            )
    require(observed_cells == expected_cells, "comparison cell set is incomplete")
    manifest_hash = validate_evidence_manifest(
        run_dir,
        run_dir / "renderer-evidence.sha256",
        expected_evidence_files,
    )
    return manifest_hash, len(observed_cells)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_run_dir = (
        CODE_ROOT / "reports" / "environments" / "icml" / "jobs" / args.job_id
    )
    expected_output = expected_run_dir / "renderer-positive-control-validation.json"
    require(args.job_id.isdigit(), "renderer verification job ID is not numeric")
    require(os.environ.get("SLURM_JOB_ID") == args.job_id, "renderer verification job differs")
    require(Path(os.path.abspath(args.run_dir)) == expected_run_dir, "renderer run path differs")
    require(Path(os.path.abspath(args.output)) == expected_output, "renderer output path differs")
    require(
        expected_run_dir.is_dir() and not expected_run_dir.is_symlink(),
        "renderer run is unsafe",
    )

    failures: list[str] = []
    evidence_manifest_sha256: str | None = None
    observed_cells = 0
    try:
        evidence_manifest_sha256, observed_cells = validate_matrix(
            expected_run_dir, args.job_id
        )
    except Exception as exc:
        failures.append(f"renderer_matrix_validation:{type(exc).__name__}")
    publish_json(
        expected_output,
        {
            "schema_version": 1,
            "job_id": args.job_id,
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "observed_cell_executions": observed_cells,
            "positive_control": "conda_run/runtime_full_cuda/default",
            "required_replicates": [1, 2],
            "pymupdf_version": EXPECTED_PYMUPDF_VERSION,
            "evidence_manifest_sha256": evidence_manifest_sha256,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
