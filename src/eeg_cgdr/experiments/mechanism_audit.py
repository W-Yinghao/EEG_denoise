"""Independent repaired CGDR mechanism-audit stage dispatcher."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    KLADOS_TRAIN_RECORDS,
    KLADOS_UNTOUCHED_RECORDS,
    fit_channel_normalizer,
    prepare_mechanism_record,
    write_mechanism_split_manifest,
)
from eeg_cgdr.operators import CalibrationBatch, P0Config, fit_p0
from saddpm.diffusion.schedule import DiffusionConfig, make_betas, validate_cgdr_schedule


AUDIT_PROTOCOL = "repaired_source_record_mechanism_v1"


def _historical_evaluation_records(config: dict[str, Any]) -> tuple[int, ...]:
    """Resolve the legacy evaluation partition without calling it untouched.

    Historical configs retain ``untouched_source_records`` so old results stay
    reproducible.  New development-only configs use the explicit
    ``historical_evaluation_source_records_already_used_in_diagnosis`` key.
    """

    klados = config["klados"]
    current = klados.get(
        "historical_evaluation_source_records_already_used_in_diagnosis"
    )
    legacy = klados.get("untouched_source_records")
    if current is not None and legacy is not None:
        raise ValueError("config cannot declare both current and legacy evaluation keys")
    values = current if current is not None else legacy
    if values is None:
        raise ValueError("historical evaluation source-record partition is missing")
    return tuple(int(value) for value in values)


def _loader_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config["klados"]
    return {
        "data_root": raw["data_root"],
        "files": {
            "contaminated": raw["contaminated"],
            "clean": raw["clean"],
            "heog": raw["heog"],
            "veog": raw["veog"],
        },
        "official_description": {"records": 54},
    }


def _p0_config(
    config: dict[str, Any], *, bootstrap_replicates: Optional[int] = None
) -> P0Config:
    raw = config["p0"]
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=(
            int(raw["bootstrap_replicates"])
            if bootstrap_replicates is None
            else int(bootstrap_replicates)
        ),
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=(
            float(raw["minimum_bootstrap_success"])
            if bootstrap_replicates is None
            else 0.0
        ),
        maximum_bootstrap_median_distance=(
            float(raw["maximum_bootstrap_median_distance"])
            if bootstrap_replicates is None
            else float("inf")
        ),
        maximum_bootstrap_q90_distance=(
            float(raw["maximum_bootstrap_q90_distance"])
            if bootstrap_replicates is None
            else float("inf")
        ),
        seed=int(config["seed"]),
    )


def _validate_protocol(config: dict[str, Any]) -> None:
    if config.get("audit_protocol") != AUDIT_PROTOCOL:
        raise ValueError(f"expected audit_protocol={AUDIT_PROTOCOL}")
    if tuple(config["klados"]["training_source_records"]) != KLADOS_TRAIN_RECORDS:
        raise ValueError("training source-record partition differs from the frozen protocol")
    if tuple(config["klados"]["development_source_records"]) != KLADOS_DEVELOPMENT_RECORDS:
        raise ValueError("development source-record partition differs from the frozen protocol")
    if _historical_evaluation_records(config) != KLADOS_UNTOUCHED_RECORDS:
        raise ValueError(
            "historical evaluation source-record partition differs from the frozen protocol"
        )
    if int(config["model"]["in_channels"]) != 19 or int(config["model"]["out_channels"]) != 19:
        raise ValueError("repaired Klados prior must be joint 19-channel")
    diffusion = DiffusionConfig(
        **{
            key: config["diffusion"][key]
            for key in ("num_timesteps", "beta_start", "beta_end", "schedule")
        }
    )
    terminal = validate_cgdr_schedule(diffusion)
    if terminal > float(config["diffusion"]["maximum_terminal_alpha_bar"]):
        raise ValueError("configured diffusion does not meet terminal alpha_bar contract")
    if config["p0"].get("fit_dtype") != "float64":
        raise ValueError("P0 mechanism fit must remain FP64")
    if config["p0"].get("svd_target") != "transfer_matrix_C_hat":
        raise ValueError("P0 SVD must act on C_hat")
    if config["p0"].get("reference_standardization") != "support_channel_zscore":
        raise ValueError("P0/EOG standardization must use support statistics only")
    sampling = config["sampling"]
    if float(sampling["eta"]) != 0.0:
        raise ValueError("registered repaired mechanism audit uses deterministic DDIM eta=0")
    if sampling.get("output_rule") != "posterior_mean_of_seed_samples":
        raise ValueError("mechanism output rule must average waveforms before metrics")
    seeds = [int(value) for value in sampling["seeds"]]
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError(
            "mechanism audit requires exactly five unique algorithmic seeds"
        )
    if not 0 < int(sampling["warm_start_steps"]) <= int(sampling["ddim_steps"]):
        raise ValueError("warm-start DDIM call count is invalid")
    if not 0 <= int(sampling["one_step_timestep"]) < int(config["diffusion"]["num_timesteps"]):
        raise ValueError("single-prior-evaluation timestep is invalid")
    from eeg_cgdr.inference.sampler_candidates import sampler_candidate

    candidates = [sampler_candidate(str(value)) for value in sampling["candidates"]]
    if {candidate.candidate_id for candidate in candidates} != {
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    } or len(candidates) != 6:
        raise ValueError("mechanism audit must register each M0--M5 exactly once")
    if float(config["klados"]["calibration_seconds"]) != 10.0:
        raise ValueError(
            "repaired source-record mechanism audit freezes a real 10-second support block"
        )
    if config["observation"].get("attenuation_source") != "framewise_external_eog":
        raise ValueError("mechanism attenuation must come from framewise external EOG")
    if not bool(config["observation"].get("residual_dimension_normalization")):
        raise ValueError("mechanism guidance must normalize by valid residual dimension")
    trusts = [float(value) for value in config["observation"]["trust_radius_candidates"]]
    if len(trusts) < 2 or len(set(trusts)) != len(trusts) or any(
        not np.isfinite(value) or value <= 0.0 for value in trusts
    ):
        raise ValueError("development trust-radius candidates must be unique and positive")
    if not 0.0 <= float(config["observation"]["rho"]) <= 1.0:
        raise ValueError("rho must lie in [0,1]")


def run_mechanism_cpu_validation(
    config: dict[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """J0: real-record semantic validation, complementary to scheduled pytest."""

    _validate_protocol(config)
    records = load_klados_records(_loader_config(config))
    if [record.record_id for record in records] != list(range(1, 55)):
        raise ValueError("Klados loader did not return exactly sim01-sim54 in native order")
    if any(record.clean.shape[0] != 19 for record in records):
        raise ValueError("Klados repaired prior requires the native 19-channel montage")
    normalizer = fit_channel_normalizer(records)
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    source_rate = int(config["klados"]["source_sampling_rate"])
    window_samples = int(config["preprocessing"]["window_samples"])
    validation_record_ids = (
        (1, 30, 31, 45)
        if config.get("execution_scope") == "development_diagnostics_only"
        else (31, 45, 37, 54)
    )
    prepared = [
        prepare_mechanism_record(
            records[record_id - 1],
            normalizer,
            source_rate=source_rate,
            target_rate=target_rate,
            window_samples=window_samples,
            calibration_seconds=float(config["klados"]["calibration_seconds"]),
            guard_seconds=float(config["klados"]["guard_seconds"]),
        )
        for record_id in validation_record_ids
    ]
    expected_query_start = float(config["klados"]["calibration_seconds"]) + float(
        config["klados"]["guard_seconds"]
    )
    for item in prepared:
        if (
            item.query_start_seconds != expected_query_start
            or item.query_end_seconds <= expected_query_start
        ):
            raise ValueError("frozen support/guard/query boundary is invalid")
        mask = item.valid_time_weight
        if mask.shape != (item.observed_windows.shape[0], window_samples):
            raise ValueError("valid-time mask shape mismatch")
        padded = mask == 0.0
        if np.any(item.observed_windows * padded[:, None, :] != 0.0):
            raise ValueError("Klados normalized padding is not zero")
        if np.any(item.clean_windows * padded[:, None, :] != 0.0):
            raise ValueError("Klados clean normalized padding is not zero")

    calibration = prepared[0].calibration
    outcome = fit_p0(
        calibration,
        _p0_config(config, bootstrap_replicates=0),
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    if outcome.transfer is None:
        raise ValueError(f"real sim31 P0 validation is ineligible: {outcome.reasons}")
    eeg = np.asarray(calibration.eeg, dtype=np.float64)
    eog = np.asarray(calibration.eog, dtype=np.float64)
    y = eeg - eeg.mean(axis=1, keepdims=True)
    e = eog - eog.mean(axis=1, keepdims=True)
    ridge = float(config["p0"]["ridge_lambda"])
    reference = np.linalg.solve(
        e @ e.T + ridge * np.eye(e.shape[0], dtype=np.float64),
        (y @ e.T).T,
    ).T
    np.testing.assert_allclose(
        outcome.transfer.transfer_matrix, reference, rtol=2.0e-12, atol=2.0e-12
    )
    projector = outcome.transfer.projector
    symmetry = float(np.linalg.norm(projector - projector.T, ord="fro"))
    idempotence = float(np.linalg.norm(projector @ projector - projector, ord="fro"))
    if symmetry > 1.0e-10 or idempotence > 1.0e-10:
        raise ValueError("real-record P0 projector failed orthoprojector checks")

    diffusion = DiffusionConfig(**{
        key: config["diffusion"][key]
        for key in ("num_timesteps", "beta_start", "beta_end", "schedule")
    })
    terminal_alpha_bar = validate_cgdr_schedule(diffusion)
    direct_terminal = float(np.prod((1.0 - make_betas(diffusion).numpy())))
    if not np.isclose(terminal_alpha_bar, direct_terminal, rtol=1.0e-12, atol=0.0):
        raise AssertionError("terminal alpha_bar implementations disagree")

    split_path = Path(config["klados"]["split_manifest"])
    write_mechanism_split_manifest(
        split_path,
        calibration_seconds=float(config["klados"]["calibration_seconds"]),
        guard_seconds=float(config["klados"]["guard_seconds"]),
        historical_evaluation_already_used=(
            config.get("execution_scope") == "development_diagnostics_only"
        ),
    )
    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    result = {
        "status": "passed",
        "audit_protocol": AUDIT_PROTOCOL,
        "real_source_records_loaded": len(records),
        "real_validation_records": list(validation_record_ids),
        "training_source_records": len(KLADOS_TRAIN_RECORDS),
        "development_source_records": len(KLADOS_DEVELOPMENT_RECORDS),
        "historical_evaluation_source_records_already_used_in_diagnosis": len(
            KLADOS_UNTOUCHED_RECORDS
        ),
        "participant_mapping_claimed": False,
        "calibration_query_disjoint": True,
        "normalization_sources": list(normalizer.source_records),
        "padding_zero_after_normalization": True,
        "p0_reference_match": True,
        "p0_rank": outcome.transfer.rank,
        "p0_singular_values": outcome.transfer.diagnostics["singular_values"],
        "p0_projector_symmetry_error": symmetry,
        "p0_projector_idempotence_error": idempotence,
        "terminal_alpha_bar": terminal_alpha_bar,
        "split_manifest": str(split_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mechanism_cpu_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_repaired_mechanism_stage(
    config: dict[str, Any],
    *,
    stage: str,
    run_dir: Path,
    device: Any = None,
    task_index: Optional[int] = None,
) -> dict[str, Any]:
    """Dispatch a repaired J0--J6 stage without reusing legacy train-fold."""

    _validate_protocol(config)
    if config.get("execution_scope") == "development_diagnostics_only" and stage not in {
        "cpu-tests",
        "train-prior",
        "sampler-integration",
    }:
        raise ValueError(
            f"stage {stage} is forbidden by development_diagnostics_only execution scope"
        )
    if stage == "cpu-tests":
        return run_mechanism_cpu_validation(config, run_dir=run_dir)
    if stage == "train-prior":
        if device is None:
            raise ValueError("train-prior requires a CUDA device")
        from .mechanism_training import train_mechanism_prior

        trained = train_mechanism_prior(config, run_dir=run_dir, device=device)
        return {
            "status": trained.status,
            "checkpoint": str(trained.checkpoint),
            "best_checkpoint": str(trained.best_checkpoint),
            "epochs_completed": trained.epochs_completed,
            "steps_completed": trained.steps_completed,
        }
    if stage == "aggregate-development":
        from .mechanism_aggregate import aggregate_development

        return aggregate_development(config)
    if stage == "decision":
        from .mechanism_aggregate import aggregate_untouched_and_decide

        return aggregate_untouched_and_decide(config)
    if stage in ("sampler-integration", "development-record", "untouched-record"):
        from .mechanism_runner import run_gpu_mechanism_stage

        if device is None:
            raise ValueError(f"{stage} requires a CUDA device")
        return run_gpu_mechanism_stage(
            config,
            stage=stage,
            run_dir=run_dir,
            device=device,
            task_index=task_index,
        )
    raise ValueError(f"unknown repaired mechanism stage: {stage}")
