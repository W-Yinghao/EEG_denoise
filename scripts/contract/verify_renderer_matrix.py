#!/usr/bin/env python3
"""Verify the two-process fixed renderer startup candidate and its evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
ICML_ENV = Path("/home/infres/yinwang/anaconda3/envs/icml")
STARTUP_MODULE = CODE_ROOT / "scripts/contract/renderer_startup.py"
MAX_RECORD_BYTES = 1024 * 1024
EXPECTED_PYMUPDF_VERSION = "1.26.5"
EXPECTED_CONTRACT_ID = "pymupdf_conda_standard_default_none_v1"
EXPECTED_CONTRACT_SHA256 = (
    "dfa6ace23bcb146e9bf23a50c078c5e3a391b3353e1fff83d337beaae7cb15ae"
)
EXPECTED_COMPONENTS = {
    "fitz/__init__.py",
    "pymupdf/__init__.py",
    "pymupdf/mupdf.py",
    "pymupdf/_mupdf.so",
    "pymupdf/_extra.so",
    "pymupdf/libmupdf.so.26.10",
    "pymupdf/libmupdfcpp.so.26.10",
}
HEX64 = re.compile(r"[0-9a-f]{64}")
COMMON_RECORD_KEYS = {
    "schema_version",
    "startup_contract_id",
    "startup_contract_sha256",
    "startup_module_sha256",
    "job_id",
    "role",
    "replicate",
    "invocation_id",
    "process_id",
    "launcher",
    "warnings_policy",
    "preload_plan",
    "stderr_policy",
    "stdout_policy",
    "deliberate_preloaded_modules",
    "forbidden_preloaded_modules_observed",
    "native_components_mapped_preimport",
    "python_executable",
    "python_prefix",
    "python_version",
    "conda_prefix",
    "conda_default_env",
    "conda_shlvl",
    "pythonwarnings_environment",
    "pythonhome_environment",
    "pythonpath_environment",
    "ld_preload_environment",
    "python_warnoptions",
    "core_dump_soft_limit",
    "faulthandler_enabled",
    "authority",
    "status",
    "generated_at_utc",
}


class CandidateValidationError(RuntimeError):
    """A fixed renderer-candidate validation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateValidationError(message)


def regular_file(path: Path, parent: Path, maximum_bytes: int) -> os.stat_result:
    require(path.parent == parent, "renderer evidence path leaves its job directory")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), "renderer evidence is not a regular file")
    require(not stat.S_ISLNK(metadata.st_mode), "renderer evidence is symbolic")
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


def parse_utc(value: object) -> datetime:
    require(isinstance(value, str), "renderer timestamp is absent")
    observed = datetime.fromisoformat(value)
    require(observed.tzinfo is not None, "renderer timestamp lacks timezone")
    require(observed.utcoffset() == timezone.utc.utcoffset(observed), "renderer timestamp is not UTC")
    return observed


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


def validate_record(
    record: dict[str, object],
    *,
    job_id: str,
    replicate: int,
    status: str,
    startup_module_sha256: str,
) -> tuple[datetime, dict[str, str] | None]:
    expected_keys = set(COMMON_RECORD_KEYS)
    if status == "import_ok":
        expected_keys.update({"pymupdf_version", "component_sha256"})
    require(set(record) == expected_keys, "renderer startup record key set differs")
    require(type(record.get("schema_version")) is int and record.get("schema_version") == 1, "record schema differs")
    require(record.get("startup_contract_id") == EXPECTED_CONTRACT_ID, "startup contract ID differs")
    require(record.get("startup_contract_sha256") == EXPECTED_CONTRACT_SHA256, "startup contract hash differs")
    require(record.get("startup_module_sha256") == startup_module_sha256, "startup module hash differs")
    require(record.get("job_id") == job_id, "record job differs")
    require(record.get("role") == f"probe_r{replicate}", "record role differs")
    require(type(record.get("replicate")) is int and record.get("replicate") == replicate, "record replicate differs")
    expected_invocation_id = hashlib.sha256(
        f"{EXPECTED_CONTRACT_SHA256}\0{job_id}\0probe_r{replicate}".encode("ascii")
    ).hexdigest()
    require(record.get("invocation_id") == expected_invocation_id, "invocation ID differs")
    require(type(record.get("process_id")) is int and int(record["process_id"]) > 1, "process ID differs")
    expected_values = {
        "launcher": "conda_run",
        "warnings_policy": "default",
        "preload_plan": "none",
        "stderr_policy": "empty",
        "stdout_policy": "empty",
        "deliberate_preloaded_modules": [],
        "forbidden_preloaded_modules_observed": [],
        "native_components_mapped_preimport": [],
        "python_executable": str(ICML_ENV / "bin" / "python"),
        "python_prefix": str(ICML_ENV),
        "conda_prefix": str(ICML_ENV),
        "conda_default_env": "icml",
        "pythonwarnings_environment": None,
        "pythonhome_environment": None,
        "pythonpath_environment": None,
        "ld_preload_environment": None,
        "python_warnoptions": [],
        "core_dump_soft_limit": 0,
        "faulthandler_enabled": True,
        "authority": {"kind": "self_audit_candidate", "audit_job_id": job_id},
        "status": status,
    }
    for field, expected in expected_values.items():
        require(record.get(field) == expected, f"renderer record field {field} differs")
    require(isinstance(record.get("python_version"), str) and bool(record["python_version"]), "Python version is absent")
    conda_shlvl = record.get("conda_shlvl")
    require(isinstance(conda_shlvl, str) and conda_shlvl.isdigit() and int(conda_shlvl) >= 1, "Conda activation depth differs")
    generated = parse_utc(record.get("generated_at_utc"))
    if status == "preimport_ready":
        return generated, None
    require(record.get("pymupdf_version") == EXPECTED_PYMUPDF_VERSION, "PyMuPDF version differs")
    components = record.get("component_sha256")
    require(isinstance(components, dict) and set(components) == EXPECTED_COMPONENTS, "renderer component set differs")
    require(
        all(isinstance(value, str) and HEX64.fullmatch(value) for value in components.values()),
        "renderer component hash differs",
    )
    return generated, {str(key): str(value) for key, value in components.items()}


def validate_evidence_manifest(
    run_dir: Path, manifest: Path, expected_files: set[Path]
) -> str:
    regular_file(manifest, run_dir, MAX_RECORD_BYTES)
    actual_files = {
        path for path in run_dir.glob("renderer-import-*") if path.parent == run_dir
    }
    require(actual_files == expected_files, "renderer evidence filename set differs")
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


def validate_candidate(run_dir: Path, job_id: str) -> dict[str, object]:
    module_metadata = STARTUP_MODULE.lstat()
    require(stat.S_ISREG(module_metadata.st_mode) and not stat.S_ISLNK(module_metadata.st_mode), "startup module is unsafe")
    startup_module_sha256 = sha256_path(STARTUP_MODULE)
    comparison = load_json(run_dir / "renderer-import-comparison.json", run_dir)
    expected_comparison_keys = {
        "schema_version",
        "job_id",
        "core_dump_limit",
        "capture_per_stream_limit_bytes",
        "capture_aggregate_upper_bound_bytes",
        "design",
        "candidate_id",
        "expected_cell_executions",
        "candidate_setup_failed",
        "positive_control",
        "replicates_per_plan",
        "results",
    }
    require(set(comparison) == expected_comparison_keys, "comparison key set differs")
    integer_expectations = {
        "schema_version": 3,
        "core_dump_limit": 0,
        "capture_per_stream_limit_bytes": MAX_RECORD_BYTES,
        "capture_aggregate_upper_bound_bytes": 4 * MAX_RECORD_BYTES,
        "expected_cell_executions": 2,
        "replicates_per_plan": 2,
    }
    for field, expected in integer_expectations.items():
        require(type(comparison.get(field)) is int and comparison.get(field) == expected, f"comparison field {field} differs")
    require(comparison.get("job_id") == job_id, "comparison job differs")
    require(comparison.get("design") == "fixed_single_candidate", "comparison design differs")
    require(comparison.get("candidate_id") == "conda_standard_default_none", "comparison candidate differs")
    require(comparison.get("candidate_setup_failed") is False, "candidate setup failed")
    require(
        comparison.get("positive_control")
        == {
            "launcher": "conda_run",
            "preload": "none",
            "warnings_policy": "default",
            "required_replicates": [1, 2],
        },
        "comparison positive control differs",
    )
    results = comparison.get("results")
    require(isinstance(results, list) and len(results) == 2, "comparison results are incomplete")
    expected_cells = {
        ("conda_run", "none", "default", 1),
        ("conda_run", "none", "default", 2),
    }
    observed_cells: set[tuple[object, object, object, object]] = set()
    expected_evidence_files = {run_dir / "renderer-import-comparison.json"}
    component_hashes: dict[str, str] | None = None
    startup_fingerprint: dict[str, object] | None = None
    process_ids: set[int] = set()
    invocation_ids: set[str] = set()
    for result in results:
        require(isinstance(result, dict), "comparison result is not an object")
        require(
            set(result)
            == {
                "launcher",
                "preload",
                "warnings_policy",
                "replicate",
                "exit_code",
                "preimport_ready",
                "result_record",
                "stdout_bytes",
                "stderr_bytes",
            },
            "comparison result key set differs",
        )
        replicate = result.get("replicate")
        require(type(replicate) is int, "comparison replicate type differs")
        cell = (
            result.get("launcher"),
            result.get("preload"),
            result.get("warnings_policy"),
            replicate,
        )
        require(cell in expected_cells and cell not in observed_cells, "comparison cell differs")
        observed_cells.add(cell)
        require(type(result.get("exit_code")) is int and result.get("exit_code") == 0, "candidate process failed")
        require(result.get("preimport_ready") is True, "candidate marker is absent")
        require(result.get("result_record") is True, "candidate result is absent")
        for field in ("stdout_bytes", "stderr_bytes"):
            require(type(result.get(field)) is int and result.get(field) == 0, "candidate stream is not empty")
        stem = f"renderer-import-conda_run-none-warnings_default-r{replicate}"
        for suffix in ("stdout", "stderr"):
            stream_path = run_dir / f"{stem}.{suffix}"
            expected_evidence_files.add(stream_path)
            require(regular_file(stream_path, run_dir, MAX_RECORD_BYTES).st_size == 0, "candidate stream size differs")
        marker_path = run_dir / f"{stem}.preimport.json"
        result_path = run_dir / f"{stem}.json"
        expected_evidence_files.update({marker_path, result_path})
        marker = load_json(marker_path, run_dir)
        imported = load_json(result_path, run_dir)
        marker_time, _ = validate_record(
            marker,
            job_id=job_id,
            replicate=replicate,
            status="preimport_ready",
            startup_module_sha256=startup_module_sha256,
        )
        result_time, observed_components = validate_record(
            imported,
            job_id=job_id,
            replicate=replicate,
            status="import_ok",
            startup_module_sha256=startup_module_sha256,
        )
        require(marker_time <= result_time, "renderer startup timestamp order differs")
        common_keys = COMMON_RECORD_KEYS - {"status", "generated_at_utc"}
        require(
            {key: marker[key] for key in common_keys}
            == {key: imported[key] for key in common_keys},
            "pre-import and import-ok common evidence differs",
        )
        process_ids.add(int(imported["process_id"]))
        invocation_ids.add(str(imported["invocation_id"]))
        dynamic_fields = {
            "generated_at_utc",
            "invocation_id",
            "process_id",
            "replicate",
            "role",
        }
        observed_fingerprint = {
            key: imported[key] for key in sorted(COMMON_RECORD_KEYS - dynamic_fields)
        }
        if startup_fingerprint is None:
            startup_fingerprint = observed_fingerprint
        else:
            require(
                startup_fingerprint == observed_fingerprint,
                "renderer startup fingerprint differs across replicates",
            )
        if component_hashes is None:
            component_hashes = observed_components
        else:
            require(component_hashes == observed_components, "component hashes differ across replicates")
    require(observed_cells == expected_cells, "comparison cell set is incomplete")
    require(len(process_ids) == 2, "renderer probes did not use separate processes")
    require(len(invocation_ids) == 2, "renderer probe invocation IDs are not distinct")
    require(component_hashes is not None, "renderer component hashes are absent")
    require(startup_fingerprint is not None, "renderer startup fingerprint is absent")
    manifest_hash = validate_evidence_manifest(
        run_dir,
        run_dir / "renderer-evidence.sha256",
        expected_evidence_files,
    )
    return {
        "evidence_manifest_sha256": manifest_hash,
        "observed_cell_executions": len(observed_cells),
        "startup_module_sha256": startup_module_sha256,
        "startup_fingerprint_sha256": hashlib.sha256(
            json.dumps(
                startup_fingerprint,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "component_sha256": component_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_run_dir = CODE_ROOT / "reports/environments/icml/jobs" / args.job_id
    expected_output = expected_run_dir / "renderer-positive-control-validation.json"
    require(args.job_id.isdigit(), "renderer verification job ID is not numeric")
    require(os.environ.get("SLURM_JOB_ID") == args.job_id, "renderer verification job differs")
    require(Path(os.path.abspath(args.run_dir)) == expected_run_dir, "renderer run path differs")
    require(Path(os.path.abspath(args.output)) == expected_output, "renderer output path differs")
    require(expected_run_dir.is_dir() and not expected_run_dir.is_symlink(), "renderer run is unsafe")

    failures: list[str] = []
    evidence: dict[str, object] = {}
    contract_bundle_sha256 = os.environ.get("DENOISENET_CONTRACT_BUNDLE_SHA256")
    slurm_jobs_bundle_sha256 = os.environ.get("DENOISENET_SLURM_JOBS_BUNDLE_SHA256")
    audit_request_id = os.environ.get("DENOISENET_REQUEST_ID")
    audit_request_sha256 = os.environ.get("DENOISENET_REQUEST_SHA256")
    if (
        not isinstance(contract_bundle_sha256, str)
        or HEX64.fullmatch(contract_bundle_sha256) is None
        or not isinstance(slurm_jobs_bundle_sha256, str)
        or HEX64.fullmatch(slurm_jobs_bundle_sha256) is None
        or not isinstance(audit_request_id, str)
        or re.fullmatch(r"[A-Za-z0-9_.-]+", audit_request_id) is None
        or not isinstance(audit_request_sha256, str)
        or HEX64.fullmatch(audit_request_sha256) is None
    ):
        failures.append("renderer_audit_provenance:invalid")
    try:
        evidence = validate_candidate(expected_run_dir, args.job_id)
    except Exception as exc:
        failures.append(f"renderer_candidate_validation:{type(exc).__name__}")
    publish_json(
        expected_output,
        {
            "schema_version": 2,
            "job_id": args.job_id,
            "candidate_id": "conda_standard_default_none",
            "status": "passed" if not failures else "failed",
            "failures": failures,
            "observed_cell_executions": evidence.get("observed_cell_executions", 0),
            "required_replicates": [1, 2],
            "startup_contract_id": EXPECTED_CONTRACT_ID,
            "startup_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "startup_module_sha256": evidence.get("startup_module_sha256"),
            "startup_fingerprint_sha256": evidence.get(
                "startup_fingerprint_sha256"
            ),
            "verifier_sha256": sha256_path(Path(__file__).resolve(strict=True)),
            "contract_bundle_sha256": contract_bundle_sha256,
            "slurm_jobs_bundle_sha256": slurm_jobs_bundle_sha256,
            "audit_request_id": audit_request_id,
            "audit_request_sha256": audit_request_sha256,
            "pymupdf_version": EXPECTED_PYMUPDF_VERSION,
            "component_sha256": evidence.get("component_sha256"),
            "evidence_manifest_sha256": evidence.get("evidence_manifest_sha256"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
