#!/usr/bin/env python3
"""Convert a live `scontrol show job -o` snapshot into a small safe JSON record."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


FIELD_PATTERN = re.compile(r"(?:^| )([A-Za-z][A-Za-z0-9_/:]*)=([^ ]*)")
ALLOWED_FIELDS = {
    "JobId",
    "JobName",
    "Account",
    "QOS",
    "JobState",
    "Reason",
    "Dependency",
    "KillOInInvalidDependent",
    "Restarts",
    "ExitCode",
    "RunTime",
    "TimeLimit",
    "SubmitTime",
    "EligibleTime",
    "StartTime",
    "EndTime",
    "Partition",
    "NodeList",
    "BatchHost",
    "NumNodes",
    "NumCPUs",
    "NumTasks",
    "CPUs/Task",
    "ReqTRES",
    "AllocTRES",
    "TresPerNode",
    "TresPerTask",
    "MinMemoryNode",
    "Features",
    "Comment",
}
PROFILE_EXPECTATIONS = {
    "cpu": {
        "partition": "CPU",
        "cpus": 2,
        "memory_mib": 8192,
        "gpus": 0,
        "time_limit": "00:30:00",
    },
    "cpu-high": {
        "partition": "cpu-high",
        "cpus": 8,
        "memory_mib": 65536,
        "gpus": 0,
        "time_limit": "5-00:00:00",
    },
    "A100": {
        "partition": "A100",
        "cpus": 8,
        "memory_mib": 65536,
        "gpus": 1,
        "time_limit": "12:00:00",
    },
    "H100": {
        "partition": "H100",
        "cpus": 15,
        "memory_mib": 131072,
        "gpus": 1,
        "time_limit": "12:00:00",
    },
    "L40S": {
        "partition": "L40S",
        "cpus": 8,
        "memory_mib": 65536,
        "gpus": 1,
        "time_limit": "02:00:00",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-job-id", required=True)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8", errors="replace").strip()
    observed = {key: value for key, value in FIELD_PATTERN.findall(text) if key in ALLOWED_FIELDS}
    if observed.get("JobId") != args.expected_job_id:
        raise SystemExit("allocation snapshot job ID mismatch")
    required_fields = {
        "Account",
        "QOS",
        "Partition",
        "NodeList",
        "NumCPUs",
        "ReqTRES",
        "AllocTRES",
        "Comment",
        "TimeLimit",
    }
    absent_fields = sorted(field for field in required_fields if not observed.get(field))
    if absent_fields:
        raise SystemExit(f"allocation snapshot lacks required fields: {absent_fields}")
    request_id = os.environ.get("DENOISENET_REQUEST_ID")
    if not request_id or observed.get("Comment") != f"denoiseNet:{request_id}":
        raise SystemExit("allocation comment does not bind the submission request ID")
    profile_name = os.environ.get("DENOISENET_PROFILE")
    expectation = PROFILE_EXPECTATIONS.get(profile_name or "")
    if expectation is None:
        raise SystemExit("allocation profile is not registered")
    if observed.get("Partition") != os.environ.get("SLURM_JOB_PARTITION"):
        raise SystemExit("allocation partition differs from the Slurm environment")
    if observed.get("Partition") != expectation["partition"]:
        raise SystemExit("allocation partition differs from the registered profile")
    if observed.get("Account") != "c2s" or observed.get("QOS") != "normal":
        raise SystemExit("allocation account or QOS differs from the registered mapping")
    if observed.get("TimeLimit") != expectation["time_limit"]:
        raise SystemExit("allocation walltime differs from the registered profile")
    try:
        if int(observed["NumCPUs"]) != expectation["cpus"]:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("allocation NumCPUs is invalid") from exc
    memory_value = os.environ.get("SLURM_MEM_PER_NODE")
    try:
        if int(memory_value or "") != expectation["memory_mib"]:
            raise ValueError
    except ValueError as exc:
        raise SystemExit("allocation memory differs from the registered profile") from exc
    gpu_pattern = re.compile(r"(?:^|,)gres/gpu(?:[:/][^=,]+)?=([0-9]+)(?:,|$)")
    allocated_gpu_counts = [int(value) for value in gpu_pattern.findall(observed["AllocTRES"])]
    allocated_gpus = sum(allocated_gpu_counts)
    requested_gpus = sum(int(value) for value in gpu_pattern.findall(observed["ReqTRES"]))
    if allocated_gpus != expectation["gpus"] or requested_gpus != expectation["gpus"]:
        raise SystemExit("requested or allocated GPU count differs from the registered profile")
    payload = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": args.expected_job_id,
        "fields": observed,
        "slurm_environment": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_GPUS_ON_NODE",
                "SLURM_JOB_GPUS",
                "SLURM_JOB_DEPENDENCY",
            )
        },
        "source": "live scontrol show job -o snapshot",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output, follow_symlinks=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
