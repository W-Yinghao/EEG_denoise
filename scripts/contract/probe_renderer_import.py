#!/usr/bin/env python3
"""Cold-start the registered PyMuPDF renderer and record bounded evidence."""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtime_probe import IMPORTS as RUNTIME_IMPORTS
from runtime_probe import module_version, torch_details


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
ICML_ENV = Path("/home/infres/yinwang/anaconda3/envs/icml")
EXPECTED_PYMUPDF_VERSION = "1.26.5"
RUNTIME_FULL_PRELOAD = tuple(
    module_name for module_name in RUNTIME_IMPORTS["icml"] if module_name != "fitz"
)
PRELOAD_PLANS: dict[str, tuple[str, ...]] = {
    "none": (),
    "numpy": ("numpy",),
    "prefix_2": ("numpy", "scipy"),
    "prefix_3": ("numpy", "scipy", "pandas"),
    "prefix_4": ("numpy", "scipy", "pandas", "sklearn"),
    "prefix_5": ("numpy", "scipy", "pandas", "sklearn", "mne"),
    "prefix_6": ("numpy", "scipy", "pandas", "sklearn", "mne", "yaml"),
    "cpu_stack": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "yaml",
        "einops",
    ),
    "torch": ("torch",),
    "torch_cuda": ("torch",),
    "numpy_torch": ("numpy", "torch"),
    "numpy_torch_cuda": ("numpy", "torch"),
    "pretorch_torch": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "yaml",
        "torch",
    ),
    "pretorch_torch_cuda": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "yaml",
        "torch",
    ),
    "full_imports": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "yaml",
        "torch",
        "einops",
    ),
    "full_cuda": (
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "mne",
        "yaml",
        "torch",
        "einops",
    ),
    "runtime_full_cuda": RUNTIME_FULL_PRELOAD,
}
CUDA_WARMUP_PLANS = {
    "torch_cuda",
    "numpy_torch_cuda",
    "pretorch_torch_cuda",
    "full_cuda",
    "runtime_full_cuda",
}


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
                raise OSError("renderer probe JSON write made no progress")
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


def main() -> int:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("direct", "conda_run"), required=True)
    parser.add_argument("--preload", choices=tuple(PRELOAD_PLANS), required=True)
    parser.add_argument(
        "--warnings-policy", choices=("default", "error"), required=True
    )
    parser.add_argument("--replicate", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("numeric SLURM_JOB_ID is required")
    expected_output = (
        CODE_ROOT
        / "reports"
        / "environments"
        / "icml"
        / "jobs"
        / job_id
        / (
            f"renderer-import-{args.mode}-{args.preload}-"
            f"warnings_{args.warnings_policy}-r{args.replicate}.json"
        )
    )
    output = Path(os.path.abspath(args.output))
    if output != expected_output:
        raise RuntimeError("renderer probe output is outside its registered job path")
    preimport_output = output.with_suffix(".preimport.json")
    parent = output.parent
    parent_metadata = parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeError("renderer probe output parent is unsafe")
    if Path(sys.prefix).resolve() != ICML_ENV or ICML_ENV not in Path(sys.executable).resolve().parents:
        raise RuntimeError("renderer probe is outside the registered icml environment")
    if args.mode == "conda_run":
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if (
            not conda_prefix
            or Path(conda_prefix).resolve() != ICML_ENV
            or os.environ.get("CONDA_DEFAULT_ENV") != "icml"
        ):
            raise RuntimeError("conda-run probe lacks the registered activation evidence")
    elif (
        os.environ.get("CONDA_PREFIX") is not None
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    ):
        raise RuntimeError("direct probe inherited Conda activation evidence")
    if os.environ.get("PYTHONHOME") is not None or os.environ.get("PYTHONPATH") is not None:
        raise RuntimeError("renderer probe inherited an unregistered Python path override")
    observed_pythonwarnings = os.environ.get("PYTHONWARNINGS")
    if args.warnings_policy == "default":
        if observed_pythonwarnings is not None or sys.warnoptions:
            raise RuntimeError("default-warnings probe has non-default warning options")
    elif observed_pythonwarnings != "error" or sys.warnoptions != ["error"]:
        raise RuntimeError("error-warnings probe lacks its exact warning policy")

    cuda_warmup = args.preload in CUDA_WARMUP_PLANS
    runtime_equivalent_preload = args.preload == "runtime_full_cuda"
    stage = "preload"
    preload_versions: dict[str, str | None] = {}
    runtime_torch_audit: dict[str, object] | None = None
    try:
        if "fitz" in sys.modules or "pymupdf" in sys.modules:
            raise RuntimeError("renderer was imported before the registered preload plan")
        imported: dict[str, object] = {}
        cuda_warmup_completed = False
        for module_name in PRELOAD_PLANS[args.preload]:
            imported[module_name] = importlib.import_module(module_name)
            if runtime_equivalent_preload:
                preload_versions[module_name] = module_version(
                    module_name, imported[module_name]
                )
            if module_name == "torch" and runtime_equivalent_preload:
                stage = "cuda_warmup"
                torch = imported["torch"]
                runtime_torch_audit = torch_details(torch)
                if (
                    runtime_torch_audit.get("cuda_available") is not True
                    or runtime_torch_audit.get("cuda_device_count") != 1
                    or runtime_torch_audit.get("compiled_cuda_version") is None
                    or runtime_torch_audit.get("cudnn_version") is None
                    or not any(
                        "L40S" in str(device_name)
                        for device_name in runtime_torch_audit.get(
                            "cuda_device_names", []
                        )
                    )
                    or not isinstance(runtime_torch_audit.get("cuda_operation"), dict)
                    or runtime_torch_audit["cuda_operation"].get("status") != "ok"
                    or float(runtime_torch_audit["cuda_operation"].get("sum", 0.0))
                    != 4.0
                ):
                    raise RuntimeError(
                        "runtime-equivalent preload lacks its registered CUDA evidence"
                    )
                cuda_warmup_completed = True
                stage = "preload"
            elif module_name == "torch" and cuda_warmup:
                stage = "cuda_warmup"
                torch = imported["torch"]
                if (
                    not torch.cuda.is_available()
                    or torch.cuda.device_count() != 1
                    or torch.version.cuda is None
                    or torch.backends.cudnn.version() is None
                    or "L40S" not in str(torch.cuda.get_device_name(0))
                ):
                    raise RuntimeError("renderer probe CUDA warmup lacks its registered device")
                value = torch.ones(4, device="cuda", dtype=torch.float32).sum().item()
                torch.cuda.synchronize()
                if float(value) != 4.0:
                    raise RuntimeError("renderer probe CUDA warmup produced an invalid value")
                cuda_warmup_completed = True
                stage = "preload"
        if cuda_warmup and not cuda_warmup_completed:
            raise RuntimeError("renderer probe did not execute its registered CUDA warmup")
        if "fitz" in sys.modules or "pymupdf" in sys.modules:
            raise RuntimeError("preload plan imported the renderer transitively")

        publish_json(
            preimport_output,
            {
                "schema_version": 2,
                "job_id": job_id,
                "mode": args.mode,
                "preload_plan": args.preload,
                "warnings_policy": args.warnings_policy,
                "replicate": args.replicate,
                "preloaded_modules": list(PRELOAD_PLANS[args.preload]),
                "preload_versions": preload_versions,
                "cuda_warmup": cuda_warmup,
                "runtime_equivalent_preload": runtime_equivalent_preload,
                "runtime_torch_audit": runtime_torch_audit,
                "status": "preimport_ready",
                "python_executable": sys.executable,
                "python_prefix": sys.prefix,
                "conda_prefix": os.environ.get("CONDA_PREFIX"),
                "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "pythonwarnings_environment": observed_pythonwarnings,
                "pythonhome_environment": os.environ.get("PYTHONHOME"),
                "pythonpath_environment": os.environ.get("PYTHONPATH"),
                "python_warnoptions": list(sys.warnoptions),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        stage = "renderer_import"
        fitz = importlib.import_module("fitz")
        version = module_version("fitz", fitz)
        if version != EXPECTED_PYMUPDF_VERSION:
            raise RuntimeError("PyMuPDF version differs from the registered renderer")
    except Exception:
        publish_json(
            output,
            {
                "schema_version": 2,
                "job_id": job_id,
                "mode": args.mode,
                "preload_plan": args.preload,
                "warnings_policy": args.warnings_policy,
                "replicate": args.replicate,
                "preloaded_modules": list(PRELOAD_PLANS[args.preload]),
                "preload_versions": preload_versions,
                "cuda_warmup": cuda_warmup,
                "runtime_equivalent_preload": runtime_equivalent_preload,
                "status": "failed",
                "failure_stage": stage,
                "python_executable": sys.executable,
                "python_prefix": sys.prefix,
                "conda_prefix": os.environ.get("CONDA_PREFIX"),
                "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "pythonwarnings_environment": observed_pythonwarnings,
                "pythonhome_environment": os.environ.get("PYTHONHOME"),
                "pythonpath_environment": os.environ.get("PYTHONPATH"),
                "python_warnoptions": list(sys.warnoptions),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 3
    publish_json(
        output,
        {
            "schema_version": 2,
            "job_id": job_id,
            "mode": args.mode,
            "preload_plan": args.preload,
            "warnings_policy": args.warnings_policy,
            "replicate": args.replicate,
            "preloaded_modules": list(PRELOAD_PLANS[args.preload]),
            "preload_versions": preload_versions,
            "cuda_warmup": cuda_warmup,
            "runtime_equivalent_preload": runtime_equivalent_preload,
            "runtime_torch_audit": runtime_torch_audit,
            "status": "import_ok",
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "pythonwarnings_environment": observed_pythonwarnings,
            "pythonhome_environment": os.environ.get("PYTHONHOME"),
            "pythonpath_environment": os.environ.get("PYTHONPATH"),
            "python_warnoptions": list(sys.warnoptions),
            "pymupdf_version": version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
