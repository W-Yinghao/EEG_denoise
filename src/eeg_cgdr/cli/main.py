"""Single Slurm-facing CGDR command."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from eeg_cgdr.data.klados import analyze_klados_metadata
from eeg_cgdr.validation import validate_real_cpu_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("metadata", "cpu-validate", "gpu-integrate", "train-fold", "eye-fold"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "metadata":
        result = analyze_klados_metadata(args.config, args.run_dir)
        return_code = 0 if result["status"] == "paired_provenance_supported" else 4
    elif args.mode == "cpu-validate":
        result = validate_real_cpu_path(config_path=args.config)
        (args.run_dir / "validation.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return_code = 0
    elif args.mode == "gpu-integrate":
        import torch

        from eeg_cgdr.experiments.integration import run_gpu_integration

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        result = run_gpu_integration(
            config, run_dir=args.run_dir, device=torch.device("cuda")
        )
        return_code = 0
    elif args.mode == "train-fold":
        import torch

        from eeg_cgdr.experiments.full_fold import run_full_klados_fold

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        split = Path(config["klados"]["split_manifest"])
        if not split.is_file():
            raise FileNotFoundError(f"frozen split manifest not found: {split}")
        shutil.copy2(split, args.run_dir / "split_manifest.csv")
        result = run_full_klados_fold(
            config, run_dir=args.run_dir, device=torch.device("cuda")
        )
        return_code = 75 if result["status"] == "checkpointed_for_resume" else 0
    elif args.mode == "eye-fold":
        import torch

        from eeg_cgdr.experiments.eye_fold import run_eye_bci_fold

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        result = run_eye_bci_fold(
            config, run_dir=args.run_dir, device=torch.device("cuda")
        )
        return_code = 0
    else:  # pragma: no cover
        raise AssertionError(args.mode)
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
