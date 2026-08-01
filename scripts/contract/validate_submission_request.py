#!/usr/bin/env python3
"""Strictly bind one running Slurm payload to its pre-sbatch request record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
REQUEST_PARENT = CODE_ROOT / "reports" / "slurm" / "submissions" / "requests"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PROFILE_RESOURCES: dict[str, dict[str, Any]] = {
    "cpu": {
        "cpus_per_task": 2,
        "memory": "8G",
        "walltime": "00:30:00",
        "gres": "null",
        "constraint": "null",
        "checkpoint_signal": "null",
    },
    "cpu-high": {
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "5-00:00:00",
        "gres": "null",
        "constraint": "null",
        "checkpoint_signal": "B:USR1@600",
    },
    "A100": {
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "12:00:00",
        "gres": "gpu:1",
        "constraint": "null",
        "checkpoint_signal": "B:USR1@300",
    },
    "H100": {
        "cpus_per_task": 15,
        "memory": "128G",
        "walltime": "12:00:00",
        "gres": "gpu:1",
        "constraint": "null",
        "checkpoint_signal": "B:USR1@300",
    },
    "L40S": {
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "02:00:00",
        "gres": "gpu:1",
        "constraint": "null",
        "checkpoint_signal": "B:USR1@300",
    },
}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_unique_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("submission request must be a JSON object")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-job", required=True)
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument("--expected-partition", required=True)
    parser.add_argument("--expected-dependency", default=None)
    parser.add_argument("--dependency-pattern", default=None)
    parser.add_argument("--expected-payload-count", type=int, required=True)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--expected-job-id", required=True)
    args = parser.parse_args()

    if bool(args.expected_dependency is not None) == bool(args.dependency_pattern is not None):
        parser.error("provide exactly one dependency policy")
    if not re.fullmatch(r"[0-9]+", args.expected_job_id):
        parser.error("expected job ID must be numeric")
    if not HEX64.fullmatch(args.expected_payload_sha256):
        parser.error("payload SHA-256 is malformed")
    resource_expectation = PROFILE_RESOURCES.get(args.expected_profile)
    if resource_expectation is None:
        parser.error("expected profile is not registered")

    request_id = os.environ.get("DENOISENET_REQUEST_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request_id):
        raise SystemExit("submission request ID is absent or unsafe")
    request_path = REQUEST_PARENT / f"{request_id}.json"
    if (
        request_path.is_symlink()
        or not request_path.is_file()
        or request_path.resolve(strict=True) != request_path
    ):
        raise SystemExit("submission request record is missing or unsafe")
    expected_request_sha = os.environ.get("DENOISENET_REQUEST_SHA256", "")
    observed_request_sha = sha256_path(request_path)
    if not HEX64.fullmatch(expected_request_sha) or observed_request_sha != expected_request_sha:
        raise SystemExit("submission request content differs from the exported request hash")

    request = load_unique_json(request_path)
    dependency = request.get("dependency")
    if args.expected_dependency is not None:
        dependency_valid = dependency == args.expected_dependency
    else:
        dependency_valid = isinstance(dependency, str) and bool(
            re.fullmatch(args.dependency_pattern or "", dependency)
        )

    expected_fields: dict[str, Any] = {
        "schema_version": 1,
        "state": "prepared_before_sbatch",
        "request_id": request_id,
        "job": args.expected_job,
        "profile": args.expected_profile,
        "partition": args.expected_partition,
        "account": "c2s",
        "qos": "normal",
        "array": "",
        "payload_argument_count": args.expected_payload_count,
        "payload_arguments_sha256": args.expected_payload_sha256,
        "cluster_config_sha256": os.environ.get("DENOISENET_SUBMIT_CONFIG_SHA256"),
        "environment_config_sha256": os.environ.get("DENOISENET_ENV_CONFIG_SHA256"),
        "job_script_sha256": os.environ.get("DENOISENET_JOB_SCRIPT_SHA256"),
        "submitter_sha256": os.environ.get("DENOISENET_SUBMITTER_SHA256"),
        "contract_bundle_sha256": os.environ.get("DENOISENET_CONTRACT_BUNDLE_SHA256"),
        "slurm_jobs_bundle_sha256": os.environ.get("DENOISENET_SLURM_JOBS_BUNDLE_SHA256"),
        **resource_expectation,
    }
    mismatches = [
        field for field, expected in expected_fields.items() if request.get(field) != expected
    ]
    if not dependency_valid:
        mismatches.append("dependency")
    if os.environ.get("DENOISENET_JOB") != args.expected_job:
        mismatches.append("exported_job")
    if os.environ.get("DENOISENET_PROFILE") != args.expected_profile:
        mismatches.append("exported_profile")
    if os.environ.get("DENOISENET_PAYLOAD_ARGS_SHA256") != args.expected_payload_sha256:
        mismatches.append("exported_payload_arguments_sha256")
    if os.environ.get("SLURM_JOB_ID") != args.expected_job_id:
        mismatches.append("slurm_job_id")
    if os.environ.get("SLURM_JOB_PARTITION") != args.expected_partition:
        mismatches.append("slurm_partition")
    if mismatches:
        raise SystemExit("submission request binding failed: " + ", ".join(sorted(set(mismatches))))

    post_path = CODE_ROOT / "reports" / "slurm" / "submissions" / f"{args.expected_job_id}.json"
    post_status = "not_yet_published"
    post_sha256: str | None = None
    if post_path.exists() or post_path.is_symlink():
        if post_path.is_symlink() or not post_path.is_file() or post_path.resolve() != post_path:
            raise SystemExit("post-submit record is unsafe")
        post = load_unique_json(post_path)
        post_expected = {
            "schema_version": 1,
            "request_id": request_id,
            "job_id": args.expected_job_id,
            "job": args.expected_job,
            "profile": args.expected_profile,
            "partition": args.expected_partition,
            "dependency": dependency,
            "array": "",
            "request_sha256": observed_request_sha,
            "payload_arguments_sha256": args.expected_payload_sha256,
            "cluster_config_sha256": os.environ.get("DENOISENET_SUBMIT_CONFIG_SHA256"),
            "environment_config_sha256": os.environ.get("DENOISENET_ENV_CONFIG_SHA256"),
            "job_script_sha256": os.environ.get("DENOISENET_JOB_SCRIPT_SHA256"),
            "submitter_sha256": os.environ.get("DENOISENET_SUBMITTER_SHA256"),
            "contract_bundle_sha256": os.environ.get("DENOISENET_CONTRACT_BUNDLE_SHA256"),
            "slurm_jobs_bundle_sha256": os.environ.get(
                "DENOISENET_SLURM_JOBS_BUNDLE_SHA256"
            ),
            **resource_expectation,
        }
        post_mismatches = [
            field for field, expected in post_expected.items() if post.get(field) != expected
        ]
        if post_mismatches:
            raise SystemExit(
                "post-submit record binding failed: " + ", ".join(sorted(post_mismatches))
            )
        post_status = "matched"
        post_sha256 = sha256_path(post_path)

    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "matched",
            "job_id": args.expected_job_id,
            "request_id": request_id,
            "request_path": str(request_path),
            "request_sha256": observed_request_sha,
            "dependency": dependency,
            "payload_arguments_sha256": args.expected_payload_sha256,
            "post_submit_status": post_status,
            "post_submit_sha256": post_sha256,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
