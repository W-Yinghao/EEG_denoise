#!/usr/bin/env python3
"""Audit one registered runtime without modifying it."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMPORTS: dict[str, tuple[str, ...]] = {
    "eeg2025": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "h5py",
        "yaml",
        "torch",
    ),
    "icml": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "yaml",
        "torch",
        "einops",
    ),
}

EXPECTED_PREFIXES = {
    "eeg2025": Path("/home/infres/yinwang/anaconda3/envs/eeg2025"),
    "icml": Path("/home/infres/yinwang/anaconda3/envs/icml"),
}


def module_version(module_name: str, module: Any) -> str | None:
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    distribution_name = "scikit-learn" if module_name == "sklearn" else module_name
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_details(torch_module: Any) -> dict[str, Any]:
    details: dict[str, Any] = {
        "torch_version": str(torch_module.__version__),
        "compiled_cuda_version": getattr(torch_module.version, "cuda", None),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_device_count": int(torch_module.cuda.device_count()),
        "cudnn_version": torch_module.backends.cudnn.version(),
    }
    device_names: list[str] = []
    for index in range(details["cuda_device_count"]):
        try:
            device_names.append(str(torch_module.cuda.get_device_name(index)))
        except Exception as exc:  # audit must preserve partial evidence
            device_names.append(f"ERROR:{type(exc).__name__}:{exc}")
    details["cuda_device_names"] = device_names
    operation: dict[str, Any]
    if details["cuda_available"] and details["cuda_device_count"] > 0:
        try:
            value = torch_module.ones(4, device="cuda", dtype=torch_module.float32).sum().item()
            torch_module.cuda.synchronize()
            operation = {"status": "ok", "sum": float(value)}
        except Exception as exc:
            operation = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    else:
        operation = {"status": "not_attempted", "reason": "CUDA device unavailable"}
    details["cuda_operation"] = operation
    return details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", choices=tuple(IMPORTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    imports: dict[str, dict[str, Any]] = {}
    torch_audit: dict[str, Any] | None = None
    for name in IMPORTS[args.env_name]:
        try:
            module = importlib.import_module(name)
            imports[name] = {
                "status": "ok",
                "version": module_version(name, module),
            }
            if name == "torch":
                torch_audit = torch_details(module)
        except Exception as exc:
            imports[name] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    missing_critical = [
        name
        for name in IMPORTS[args.env_name]
        if imports.get(name, {}).get("status") != "ok"
    ]
    compatibility_failures = [f"missing critical import: {name}" for name in missing_critical]
    expected_prefix = EXPECTED_PREFIXES[args.env_name].resolve()
    observed_prefix = Path(sys.prefix).resolve()
    observed_executable = Path(sys.executable).resolve()
    if observed_prefix != expected_prefix:
        compatibility_failures.append(
            f"unexpected sys.prefix: expected {expected_prefix}, observed {observed_prefix}"
        )
    if expected_prefix not in observed_executable.parents:
        compatibility_failures.append(
            f"Python executable is outside registered prefix: {observed_executable}"
        )
    if args.env_name == "icml":
        if torch_audit is None:
            compatibility_failures.append("torch audit unavailable")
        else:
            if not torch_audit["cuda_available"]:
                compatibility_failures.append("CUDA is not available")
            if torch_audit["cuda_device_count"] != 1:
                compatibility_failures.append("CUDA device count is not exactly one")
            if torch_audit["compiled_cuda_version"] is None:
                compatibility_failures.append("PyTorch has no compiled CUDA version")
            if torch_audit["cudnn_version"] is None:
                compatibility_failures.append("cuDNN version is unavailable")
            if any(name.startswith("ERROR:") for name in torch_audit["cuda_device_names"]):
                compatibility_failures.append("CUDA device name query failed")
            if torch_audit["cuda_operation"].get("status") != "ok":
                compatibility_failures.append("CUDA tensor operation failed")
            if not any("L40S" in name for name in torch_audit["cuda_device_names"]):
                compatibility_failures.append("visible GPU is not the registered L40S audit device")
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_name": args.env_name,
        "python": {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "expected_prefix": str(expected_prefix),
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "platform": platform.platform(),
        },
        "environment": {
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
            "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
            "SLURMD_NODENAME": os.environ.get("SLURMD_NODENAME"),
        },
        "imports": imports,
        "torch": torch_audit,
        "critical_import_failures": missing_critical,
        "compatibility_failures": compatibility_failures,
        "compatibility_status": "compatible" if not compatibility_failures else "incompatible",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
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
    return 0 if not compatibility_failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
