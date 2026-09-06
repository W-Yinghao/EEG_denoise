#!/usr/bin/env python
"""Environment & data sanity check (handoff §10).

Prints Python/platform info, versions of all required packages, GPU/CUDA status, and verifies
that the MOABB ``BNCI2014_001`` (BCI-IV-2a) dataset can be located and parsed.

Usage:
    python scripts/check_env.py                 # report only (login node: CUDA may be absent)
    python scripts/check_env.py --require-cuda  # exit non-zero if CUDA is unavailable (GPU jobs)
    python scripts/check_env.py --probe-subject 1   # also fully load+parse one subject via MOABB
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import platform
import socket
import sys
from pathlib import Path

REQUIRED = [
    "torch",
    "mne",
    "moabb",
    "numpy",
    "scipy",
    "sklearn",
    "einops",
    "yaml",
    "tqdm",
    "wandb",
    "matplotlib",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def _version(pkg: str) -> str:
    """Best-effort version string for an importable package."""
    try:
        module = importlib.import_module(pkg)
    except Exception as exc:  # noqa: BLE001 - report any import failure verbatim
        return f"MISSING ({type(exc).__name__}: {exc})"
    version = getattr(module, "__version__", None)
    if version is None:
        dist = {"yaml": "pyyaml", "sklearn": "scikit-learn"}.get(pkg, pkg)
        try:
            version = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            version = "?"
    return str(version)


def report_packages() -> list[str]:
    """Print versions for all required packages; return the list of missing ones."""
    print("== packages ==")
    missing: list[str] = []
    for pkg in REQUIRED:
        ver = _version(pkg)
        print(f"  {pkg:12s} {ver}")
        if ver.startswith("MISSING"):
            missing.append(pkg)
    return missing


def report_device(require_cuda: bool) -> bool:
    """Print device/CUDA info. Return False if CUDA is required but unavailable."""
    print("== device ==")
    try:
        import torch
    except ImportError:
        print("  torch not importable")
        return not require_cuda
    cuda = torch.cuda.is_available()
    print(f"  torch {torch.__version__} | CUDA build {torch.version.cuda} | cuda.is_available()={cuda}")
    if cuda:
        for i in range(torch.cuda.device_count()):
            print(f"    gpu[{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("  (no GPU here — expected on the login node; GPU jobs run on Slurm partition V100)")
    if require_cuda and not cuda:
        print("  ERROR: --require-cuda set but CUDA is unavailable.")
        return False
    return True


def report_dataset(probe_subject: int | None) -> bool:
    """Confirm the BNCI2014_001 cache exists and (optionally) parse one subject."""
    print("== dataset (BNCI2014_001 = BCI-IV-2a) ==")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from saddpm.data.config import DataConfig
    except Exception as exc:  # noqa: BLE001
        print(f"  could not import saddpm.data.config: {exc}")
        return False

    cfg = DataConfig.from_yaml(REPO_ROOT / "configs" / "data.yaml")
    try:
        from moabb.datasets import BNCI2014_001
    except ImportError as exc:
        print(f"  moabb not importable: {exc}")
        return False

    dataset = BNCI2014_001()
    print(f"  subjects available in metadata: {dataset.subject_list}")
    print(f"  event_id: {dataset.event_id}")

    cached = sorted(
        p.name
        for p in (Path.home() / "mne_data" / "MNE-bnci-data" / "database" / "data-sets" / "001-2014").glob("*.mat")
    ) if (Path.home() / "mne_data").exists() else []
    print(f"  cached .mat files at ~/mne_data: {cached if cached else 'none (will download on first use)'}")

    if probe_subject is not None:
        print(f"  parsing subject A{probe_subject:02d} via MOABB ...")
        from saddpm.data.bcic2a import load_subject

        per_session = load_subject(probe_subject, cfg)
        for sw in per_session.values():
            print(f"    {sw.summary()}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="fail if CUDA is unavailable")
    parser.add_argument("--probe-subject", type=int, default=None, help="fully load this subject via MOABB")
    args = parser.parse_args()

    print("== system ==")
    print(f"  python {platform.python_version()} ({sys.executable})")
    print(f"  platform {platform.platform()}")
    print(f"  host {socket.gethostname()}")

    missing = report_packages()
    device_ok = report_device(args.require_cuda)
    data_ok = report_dataset(args.probe_subject)

    print("== summary ==")
    ok = not missing and device_ok and data_ok
    if missing:
        print(f"  MISSING packages: {missing}")
    print(f"  RESULT: {'OK' if ok else 'PROBLEMS FOUND'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
