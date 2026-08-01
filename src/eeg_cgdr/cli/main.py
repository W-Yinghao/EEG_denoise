"""Single Slurm-facing CGDR command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

from eeg_cgdr.data.klados import analyze_klados_metadata
from eeg_cgdr.validation import validate_real_cpu_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "metadata",
            "cpu-validate",
            "gpu-integrate",
            "train-fold",
            "eye-fold",
            "mechanism-audit",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", default=None)
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
    elif args.mode == "mechanism-audit":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        protocol = config.get("audit_protocol")
        if protocol == "legacy_pre_repair_direction_check":
            if args.stage not in (None, "legacy-direction-check"):
                raise ValueError("legacy mechanism audit does not accept repaired stages")
            from eeg_cgdr.experiments.legacy_mechanism_audit import (
                run_legacy_mechanism_audit,
            )

            result = run_legacy_mechanism_audit(config, run_dir=args.run_dir)
            return_code = 0
        elif protocol == "repaired_source_record_mechanism_v1":
            if args.stage is None:
                raise ValueError("repaired mechanism audit requires --stage")
            from eeg_cgdr.experiments.mechanism_audit import (
                run_repaired_mechanism_stage,
            )

            gpu_stages = {
                "sampler-integration",
                "train-prior",
                "development-record",
                "untouched-record",
            }
            device = None
            if args.stage in gpu_stages:
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(f"{args.stage} requires a scheduled CUDA allocation")
                device = torch.device("cuda")
            task_text = os.environ.get("SLURM_ARRAY_TASK_ID")
            task_index = int(task_text) if task_text is not None else None
            result = run_repaired_mechanism_stage(
                config,
                stage=args.stage,
                run_dir=args.run_dir,
                device=device,
                task_index=task_index,
            )
            return_code = 75 if result["status"] == "checkpointed_for_resume" else 0
        else:
            raise ValueError(f"unknown mechanism audit protocol: {protocol!r}")
    else:  # pragma: no cover
        raise AssertionError(args.mode)
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
