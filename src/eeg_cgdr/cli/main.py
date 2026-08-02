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


def _array_task_index(stage: str) -> int:
    task_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_text is None:
        raise ValueError(f"{stage} requires a Slurm array task")
    return int(task_text)


def _development_only_mechanism_config(
    config: dict[str, object], config_path: Path
) -> bool:
    """Recognize the padding-repair config even during schema transition."""

    return (
        config.get("execution_scope") == "development_diagnostics_only"
        or config_path.name
        == "mechanism_audit_klados_padding_repair_development.yaml"
    )


def _write_run_result(run_dir: Path, result: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


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
            "development-diagnostics",
            "eegdfus-benchmark",
            "sgeyesub-protocol",
            "stage3-deterministic",
            "stage3-conditional-diffusion",
            "optimizer-step-audit",
            "diffusion-incremental-decision",
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
            if _development_only_mechanism_config(config, args.config) and args.stage in {
                "development-record",
                "aggregate-development",
                "untouched-record",
                "decision",
                "interpretation-audit",
            }:
                raise ValueError(
                    "the padding-repair development-only config cannot execute "
                    f"historical evaluation/decision stage {args.stage!r}"
                )
            if args.stage == "interpretation-audit":
                from eeg_cgdr.experiments.mechanism_audit import (
                    run_repaired_mechanism_stage,
                )

                # This is a naming-only alias for the repaired decision
                # aggregator.  That aggregator now preserves the historical
                # result_summary and writes a separate interpretation artifact.
                result = run_repaired_mechanism_stage(
                    config,
                    stage="decision",
                    run_dir=args.run_dir,
                )
                _write_run_result(args.run_dir, result)
                return_code = 0
            else:
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
                        raise RuntimeError(
                            f"{args.stage} requires a scheduled CUDA allocation"
                        )
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
                return_code = (
                    75 if result["status"] == "checkpointed_for_resume" else 0
                )
        else:
            raise ValueError(f"unknown mechanism audit protocol: {protocol!r}")
    elif args.mode == "development-diagnostics":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.stage == "calibration-duration":
            from eeg_cgdr.experiments.klados_development_diagnostics import (
                run_calibration_duration_axis,
            )

            result = run_calibration_duration_axis(config, run_dir=args.run_dir)
        elif args.stage == "b6-record":
            import torch

            from eeg_cgdr.experiments.klados_b6_gamma_diagnostic import (
                DEVELOPMENT_SOURCE_RECORDS,
                run_klados_b6_gamma_record,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("b6-record requires a scheduled CUDA allocation")
            task_index = _array_task_index("b6-record")
            if not 0 <= task_index < len(DEVELOPMENT_SOURCE_RECORDS):
                raise ValueError("b6-record task index must lie in [0, 7]")
            result = run_klados_b6_gamma_record(
                config,
                source_record=DEVELOPMENT_SOURCE_RECORDS[task_index],
                run_dir=args.run_dir,
                device=torch.device("cuda"),
            )
            result = {
                "status": "completed_exploratory_development_only",
                **result,
            }
            _write_run_result(args.run_dir, result)
        elif args.stage == "b6-aggregate":
            from eeg_cgdr.experiments.klados_b6_gamma_diagnostic import (
                DEVELOPMENT_SOURCE_RECORDS,
                aggregate_klados_b6_gamma_development,
            )

            output_root = Path(str(config["output_root"]))
            metric_paths = tuple(
                output_root / "records" / f"sim{record:02d}" / "metrics.csv"
                for record in DEVELOPMENT_SOURCE_RECORDS
            )
            result = aggregate_klados_b6_gamma_development(
                config,
                record_metric_paths=metric_paths,
                output_dir=output_root,
            )
            result = {
                "status": "completed_exploratory_development_only",
                **result,
            }
            _write_run_result(args.run_dir, result)
        else:
            raise ValueError(
                "development-diagnostics requires calibration-duration, "
                "b6-record, or b6-aggregate"
            )
        return_code = 0
    elif args.mode == "sgeyesub-protocol":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.stage == "metadata":
            from eeg_cgdr.experiments.sgeyesub_protocol import (
                run_sgeyesub_protocol_metadata,
            )

            result = run_sgeyesub_protocol_metadata(config, run_dir=args.run_dir)
        elif args.stage == "development-record":
            from eeg_cgdr.experiments.sgeyesub_operator_specificity import (
                run_sgeyesub_development_record,
            )

            result = run_sgeyesub_development_record(
                config,
                task_index=_array_task_index("development-record"),
                run_dir=args.run_dir,
            )
        elif args.stage == "aggregate-development":
            from eeg_cgdr.experiments.sgeyesub_operator_specificity import (
                run_sgeyesub_development_aggregate,
            )

            result = run_sgeyesub_development_aggregate(
                config,
                run_dir=args.run_dir,
            )
        elif args.stage in {"evaluation-record", "aggregate-evaluation"}:
            frozen_gamma = (
                Path(str(config["development_output_root"])) / "frozen_gamma.json"
            )
            if not frozen_gamma.is_file():
                raise RuntimeError(
                    "SGEYESUB evaluation is blocked until the development gamma "
                    f"is frozen at {frozen_gamma}"
                )
            from eeg_cgdr.experiments import sgeyesub_operator_specificity

            if args.stage == "evaluation-record":
                runner = getattr(
                    sgeyesub_operator_specificity,
                    "run_sgeyesub_evaluation_record",
                    None,
                )
                if runner is None:
                    raise RuntimeError(
                        "SGEYESUB evaluation runner is not implemented; refusing "
                        "to emit placeholder results"
                    )
                result = runner(
                    config,
                    task_index=_array_task_index("evaluation-record"),
                    run_dir=args.run_dir,
                )
            else:
                aggregator = getattr(
                    sgeyesub_operator_specificity,
                    "run_sgeyesub_evaluation_aggregate",
                    None,
                )
                if aggregator is None:
                    raise RuntimeError(
                        "SGEYESUB evaluation aggregator is not implemented; "
                        "refusing to emit placeholder results"
                    )
                result = aggregator(config, run_dir=args.run_dir)
        elif args.stage == "corrected-audit":
            from eeg_cgdr.experiments.sgeyesub_operator_specificity import (
                run_sgeyesub_corrected_audit,
            )

            result = run_sgeyesub_corrected_audit(
                config,
                run_dir=args.run_dir,
            )
        else:
            raise ValueError(
                "sgeyesub-protocol requires metadata, development-record, "
                "aggregate-development, evaluation-record, aggregate-evaluation, "
                "or corrected-audit"
            )
        return_code = 0
    elif args.mode == "stage3-deterministic":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        from eeg_cgdr.experiments.stage3_deterministic import (
            FROZEN_OPERATOR_SOURCES,
            aggregate_stage3,
            run_stage3_real_record_integration,
            run_stage3_record,
            train_task_matched_deterministic,
        )

        if args.stage in {"real-record-integration", "train-deterministic"}:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"{args.stage} requires a scheduled CUDA allocation"
                )
            if args.stage == "real-record-integration":
                result = run_stage3_real_record_integration(
                    config,
                    run_dir=args.run_dir,
                    device=torch.device("cuda"),
                )
                return_code = 0
            else:
                operator_scope = FROZEN_OPERATOR_SOURCES[
                    _array_task_index("train-deterministic")
                ]
                training_result = train_task_matched_deterministic(
                    config,
                    operator_source=operator_scope,
                    run_dir=args.run_dir,
                    device=torch.device("cuda"),
                )
                result = {
                    "status": training_result.status,
                    "operator_scope": training_result.operator_scope,
                    "operator_scope_deployable": training_result.deployable,
                }
                return_code = (
                    75
                    if training_result.status == "checkpointed_for_resume"
                    else 0
                )
        elif args.stage in {"development-record", "historical-record"}:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"{args.stage} requires a scheduled CUDA allocation"
                )
            partition = (
                "development"
                if args.stage == "development-record"
                else "historical_evaluation_already_used_in_diagnosis"
            )
            result = run_stage3_record(
                config,
                partition=partition,
                task_index=_array_task_index(args.stage),
                run_dir=args.run_dir,
                device=torch.device("cuda"),
            )
            return_code = 0
        elif args.stage in {"aggregate-development", "aggregate-historical"}:
            partition = (
                "development"
                if args.stage == "aggregate-development"
                else "historical_evaluation_already_used_in_diagnosis"
            )
            result = aggregate_stage3(
                config,
                partition=partition,
                run_dir=args.run_dir,
            )
            return_code = 0
        else:
            raise ValueError(
                "stage3-deterministic requires real-record-integration, "
                "train-deterministic, "
                "development-record, aggregate-development, historical-record, "
                "or aggregate-historical"
            )
    elif args.mode == "stage3-conditional-diffusion":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        from eeg_cgdr.experiments.stage3_conditional_diffusion import (
            FROZEN_OPERATOR_SOURCES,
            aggregate_conditional_development,
            run_conditional_development_record,
            train_operator_conditioned_diffusion,
        )

        if args.stage == "train-conditional":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "train-conditional requires a scheduled CUDA allocation"
                )
            operator_scope = FROZEN_OPERATOR_SOURCES[
                _array_task_index("train-conditional")
            ]
            training_result = train_operator_conditioned_diffusion(
                config,
                operator_scope=operator_scope,
                run_dir=args.run_dir,
                device=torch.device("cuda"),
            )
            result = {
                "status": training_result.status,
                "operator_scope": training_result.operator_scope,
                "target_updates": training_result.target_updates,
                "actual_updates": training_result.actual_updates,
            }
            return_code = (
                75
                if training_result.status == "checkpointed_for_resume"
                else 0
            )
        elif args.stage == "development-record":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "conditional development-record requires scheduled CUDA"
                )
            result = run_conditional_development_record(
                config,
                task_index=_array_task_index("conditional-development-record"),
                run_dir=args.run_dir,
                device=torch.device("cuda"),
            )
            return_code = 0
        elif args.stage == "aggregate-development":
            result = aggregate_conditional_development(
                config,
                run_dir=args.run_dir,
            )
            return_code = 0
        else:
            raise ValueError(
                "stage3-conditional-diffusion requires train-conditional, "
                "development-record, or aggregate-development"
            )
    elif args.mode == "eegdfus-benchmark":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.stage == "cpu-tests":
            from eeg_cgdr.experiments.eegdfus_benchmark import (
                run_eegdfus_cpu_validation,
            )

            result = run_eegdfus_cpu_validation(config, run_dir=args.run_dir)
            return_code = 0
        elif args.stage in {"smoke", "full"}:
            import torch

            from eeg_cgdr.experiments.eegdfus_benchmark import run_eegdfus_stage

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"EEGDfus {args.stage} requires a scheduled CUDA allocation"
                )
            result = run_eegdfus_stage(
                config,
                stage=args.stage,
                task_index=_array_task_index(f"eegdfus-{args.stage}"),
                run_dir=args.run_dir,
                device=torch.device("cuda"),
            )
            return_code = 75 if result["status"] == "checkpointed_for_resume" else 0
        elif args.stage == "aggregate-full":
            from eeg_cgdr.experiments.eegdfus_benchmark import (
                run_eegdfus_full_aggregate,
            )

            result = run_eegdfus_full_aggregate(config, run_dir=args.run_dir)
            return_code = 0
        else:
            raise ValueError(
                "eegdfus-benchmark requires cpu-tests, smoke, full, or aggregate-full"
            )
    elif args.mode == "optimizer-step-audit":
        if args.stage != "audit":
            raise ValueError("optimizer-step-audit requires stage audit")
        from eeg_cgdr.experiments.optimizer_step_audit import (
            run_optimizer_step_audit,
        )

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        result = run_optimizer_step_audit(config, run_dir=args.run_dir)
        _write_run_result(args.run_dir, result)
        return_code = (
            0
            if result["status"]
            == "passed_exact_six_checkpoints_at_expected_optimizer_step"
            else 6
        )
    elif args.mode == "diffusion-incremental-decision":
        if args.stage != "aggregate":
            raise ValueError(
                "diffusion-incremental-decision requires stage aggregate"
            )
        from eeg_cgdr.experiments.diffusion_incremental_decision import (
            run_diffusion_incremental_decision,
        )

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        result = run_diffusion_incremental_decision(config, run_dir=args.run_dir)
        return_code = 0
    else:  # pragma: no cover
        raise AssertionError(args.mode)
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
