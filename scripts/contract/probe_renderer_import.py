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


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
ICML_ENV = Path("/home/infres/yinwang/anaconda3/envs/icml")
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
}
CUDA_WARMUP_PLANS = {
    "torch_cuda",
    "numpy_torch_cuda",
    "pretorch_torch_cuda",
    "full_cuda",
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
        / f"renderer-import-{args.mode}-{args.preload}-r{args.replicate}.json"
    )
    output = Path(os.path.abspath(args.output))
    if output != expected_output:
        raise RuntimeError("renderer probe output is outside its registered job path")
    parent = output.parent
    parent_metadata = parent.lstat()
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeError("renderer probe output parent is unsafe")
    if Path(sys.prefix).resolve() != ICML_ENV or ICML_ENV not in Path(sys.executable).resolve().parents:
        raise RuntimeError("renderer probe is outside the registered icml environment")

    cuda_warmup = args.preload in CUDA_WARMUP_PLANS
    stage = "preload"
    try:
        if "fitz" in sys.modules or "pymupdf" in sys.modules:
            raise RuntimeError("renderer was imported before the registered preload plan")
        imported: dict[str, object] = {}
        cuda_warmup_completed = False
        for module_name in PRELOAD_PLANS[args.preload]:
            imported[module_name] = importlib.import_module(module_name)
            if module_name == "torch" and cuda_warmup:
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

        stage = "renderer_import"
        fitz = importlib.import_module("fitz")
        version = getattr(fitz, "VersionBind", None)
        if not isinstance(version, str) or not version:
            raise RuntimeError("PyMuPDF version binding is unavailable")
    except Exception:
        publish_json(
            output,
            {
                "schema_version": 1,
                "job_id": job_id,
                "mode": args.mode,
                "preload_plan": args.preload,
                "replicate": args.replicate,
                "preloaded_modules": list(PRELOAD_PLANS[args.preload]),
                "cuda_warmup": cuda_warmup,
                "status": "failed",
                "failure_stage": stage,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        return 3
    publish_json(
        output,
        {
            "schema_version": 1,
            "job_id": job_id,
            "mode": args.mode,
            "preload_plan": args.preload,
            "replicate": args.replicate,
            "preloaded_modules": list(PRELOAD_PLANS[args.preload]),
            "cuda_warmup": cuda_warmup,
            "status": "import_ok",
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "pymupdf_version": version,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
