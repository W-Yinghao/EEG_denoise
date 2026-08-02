"""Matched exploratory Klados operator-conditioned diffusion experiment.

The comparator is not the clean-only prior plus M1/M2/M4 guidance.  It trains a
conditional epsilon model on the same paired source-record windows and legal
conditioning fields as the task-matched deterministic U-Net.  Each operator
scope has an independent checkpoint, the common matching-P0 eligibility set is
shared by every scope, and both learned comparators use a fixed 6000-update
endpoint without development-outcome checkpoint selection.

Only sim31--sim36/sim44/sim45 development records may be evaluated here.  The
result is exploratory source-record evidence and cannot emit formal G1/G3 or a
diffusion-family decision.
"""

from __future__ import annotations

import csv
import json
import math
import random
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    KLADOS_TRAIN_RECORDS,
    ChannelNormalizer,
    fit_channel_normalizer,
    prepare_mechanism_record,
    select_records,
)
from eeg_cgdr.experiments.mechanism_training import (
    _loader_config,
    load_population_projector,
)
from eeg_cgdr.experiments.stage3_deterministic import (
    FROZEN_METHODS,
    FROZEN_OPERATOR_SOURCES,
    PROTOCOL_ID as DETERMINISTIC_PROTOCOL_ID,
    _attenuation_windows,
    _base_config,
    _bundle_record_summary,
    _common_matching_eligibility,
    _failed_metric_row,
    _mapping,
    _model_config,
    _operator_arms,
    _retainable_method_failure,
    _safe_metric_row,
    _same_normalizer,
    _scope_deployable,
    _scope_output_paths,
    _tensor_dataset,
    _validate_deterministic_checkpoint_payload,
    _window_bundle,
    validate_stage3_config,
)
from eeg_cgdr.models.conditional_diffusion import (
    OperatorConditionedEEGDiffusion,
)
from eeg_cgdr.models.deterministic_unet import TaskMatchedDeterministicUNet
from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
    scaler_optimizer_step_succeeded,
)
from saddpm.diffusion.schedule import DiffusionConfig, validate_cgdr_schedule


PROTOCOL_ID = "klados_operator_conditioned_diffusion_matched_v3"
METHOD_ID = "task_matched_multichannel_operator_conditioned_diffusion_DDIM100"
FIXED_OPTIMIZER_UPDATES = 6000
REQUIRED_CONDITIONAL_ROW_FIELDS = (
    "source_record",
    "method_id",
    "status",
    "operator_source",
    "effective_operator_source",
    "common_eligibility_status",
    "conditional_training_windows",
    "conditional_development_windows",
    "deterministic_training_windows",
    "deterministic_development_windows",
    "fixed_optimizer_updates_each",
    "conditional_actual_optimizer_updates",
    "deterministic_fixed_checkpoint_updates",
    "deterministic_actual_training_updates",
    "conditional_model_parameters",
    "deterministic_model_parameters",
    "conditional_training_walltime_seconds",
    "deterministic_training_walltime_seconds",
    "latency_seconds",
    "peak_memory_mb",
    "function_evaluations_per_seed_per_window",
    "total_function_evaluations_per_window",
    "algorithmic_seed_count",
    "same_paired_supervision_exposure",
    "conditional_training_objective",
    "deterministic_training_objective",
    "training_objectives_equal",
)
COMPARISON_METHODS = (METHOD_ID, *FROZEN_METHODS)


@dataclass(frozen=True)
class ConditionalTrainingResult:
    status: str
    operator_scope: str
    checkpoint: Path
    final_checkpoint: Path
    target_updates: int
    actual_updates: int
    resumed: bool


@dataclass(frozen=True)
class _MatchedData:
    deterministic_config: dict[str, Any]
    base: dict[str, Any]
    normalizer: ChannelNormalizer
    population: Any
    train_dataset: TensorDataset
    development_dataset: TensorDataset
    train_coverage: dict[str, Any]
    development_coverage: dict[str, Any]
    deterministic_payload: dict[str, Any]


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def _deterministic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(config.get("matched_deterministic_config", "")))
    if not path.is_file():
        raise FileNotFoundError(f"matched deterministic config is missing: {path}")
    value = _read_yaml(path)
    validate_stage3_config(value)
    if value.get("protocol_id") != DETERMINISTIC_PROTOCOL_ID:
        raise ValueError("conditional comparison requires deterministic v4 eligibility")
    return value


def _diffusion_config(config: Mapping[str, Any]) -> DiffusionConfig:
    raw = _mapping(config, "conditional_diffusion")
    value = DiffusionConfig(
        num_timesteps=int(raw["num_timesteps"]),
        beta_start=float(raw["beta_start"]),
        beta_end=float(raw["beta_end"]),
        schedule=str(raw["schedule"]),
    )
    terminal = validate_cgdr_schedule(value)
    if terminal > float(raw["terminal_alpha_bar_maximum"]):
        raise ValueError("conditional diffusion terminal alpha_bar is too large")
    return value


def validate_conditional_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID}")
    if int(config.get("harness_level", -1)) != 1:
        raise ValueError("conditional comparison requires HARNESS_LEVEL=1")
    deterministic = _deterministic_config(config)
    split = _mapping(config, "source_record_split")
    if tuple(int(value) for value in split.get("training", ())) != KLADOS_TRAIN_RECORDS:
        raise ValueError("conditional training records must be sim01-sim30")
    if tuple(int(value) for value in split.get("development", ())) != (
        KLADOS_DEVELOPMENT_RECORDS
    ):
        raise ValueError("conditional development records differ from the frozen split")
    if split.get("historical_evaluation_allowed") is not False:
        raise ValueError("already-used historical records are forbidden in this comparator")
    deterministic_split = _mapping(deterministic, "source_record_split")
    if list(split["training"]) != list(deterministic_split["training"]) or list(
        split["development"]
    ) != list(deterministic_split["development"]):
        raise ValueError("conditional and deterministic source-record splits differ")
    fairness = _mapping(config, "fair_comparison_contract")
    if tuple(fairness.get("operator_sources", ())) != FROZEN_OPERATOR_SOURCES:
        raise ValueError("conditional operator scopes differ from deterministic scopes")
    if fairness.get("common_eligibility_rule") != (
        "matching_p0_eligible_records_shared_by_all_operator_scopes"
    ):
        raise ValueError("conditional comparison must use common matching eligibility")
    required_true = (
        "operator_scope_isolated_checkpoints",
        "exact_window_bundle_builder_shared_with_deterministic",
        "development_records_only_for_diagnostics",
        "no_development_or_evaluation_outcome_checkpoint_selection",
        "same_paired_supervision_exposure",
    )
    if any(fairness.get(name) is not True for name in required_true):
        raise ValueError("conditional fairness contract has been weakened")
    if fairness.get("target_optimizer_updates") != (
        "fixed_6000_successful_optimizer_updates_for_both_models"
    ):
        raise ValueError("both comparators require the fixed 6000-update endpoint")
    if tuple(fairness.get("visible_inputs", ())) != (
        OperatorConditionedEEGDiffusion.visible_input_fields
    ):
        raise ValueError("conditional visible inputs differ from deterministic U-Net")
    if fairness.get("different_training_objective_disclosed") != (
        "epsilon_prediction_vs_deterministic_task_loss"
    ):
        raise ValueError("the two training objectives must be explicitly distinguished")
    if fairness.get("clean_target_visible_to_model_input") is not False:
        raise ValueError("paired clean targets cannot be conditional model inputs")
    deterministic_training = _mapping(deterministic, "deterministic_training")
    if (
        int(deterministic_training.get("minimum_updates", -1))
        != FIXED_OPTIMIZER_UPDATES
        or int(deterministic_training.get("maximum_updates", -1))
        != FIXED_OPTIMIZER_UPDATES
        or deterministic_training.get("checkpoint_selection")
        != "fixed_6000_update_endpoint_no_development_selection"
        or deterministic_training.get("development_loss_role")
        != "diagnostic_only_not_checkpoint_or_update_selection"
    ):
        raise ValueError("matched deterministic v4 is not a fixed 6000-step endpoint")
    if fairness.get("broad_diffusion_family_claim_allowed") is not False:
        raise ValueError("this exploratory arm cannot classify the diffusion family")
    inference = _mapping(config, "development_inference")
    if tuple(int(value) for value in inference.get("records", ())) != (
        KLADOS_DEVELOPMENT_RECORDS
    ):
        raise ValueError("conditional inference is development-only")
    if inference.get("common_eligible_records_only") is not True:
        raise ValueError("conditional metrics must use common-eligible records only")
    if int(inference.get("ddim_steps", -1)) != 100 or float(
        inference.get("eta", float("nan"))
    ) != 0.0:
        raise ValueError("conditional comparator is frozen to deterministic DDIM100")
    seeds = tuple(int(value) for value in inference.get("seeds", ()))
    deterministic_seeds = tuple(
        int(value)
        for value in _mapping(deterministic, "frozen_comparison")["seeds"]
    )
    if seeds != deterministic_seeds or len(set(seeds)) != 5:
        raise ValueError("conditional and deterministic algorithmic seeds differ")
    if inference.get("seeds_are_statistical_units") is not False:
        raise ValueError("algorithmic seeds cannot be statistical units")
    if int(inference.get("inference_batch_size", -1)) != int(
        _mapping(deterministic, "frozen_comparison")["inference_batch_size"]
    ):
        raise ValueError("conditional and deterministic inference batch sizes differ")
    conditional_training = _mapping(config, "training")
    if (
        conditional_training.get("final_checkpoint_rule")
        != "fixed_6000_update_endpoint_no_early_stop"
        or conditional_training.get("development_loss_role")
        != "diagnostic_only_not_checkpoint_selection"
        or conditional_training.get("optimizer_step_accounting")
        != "increment_only_when_grad_scaler_executes_optimizer_step"
        or float(conditional_training.get("amp_initial_scale", float("nan")))
        != 1024.0
        or int(conditional_training.get("maximum_skipped_optimizer_steps", -1))
        != 0
    ):
        raise ValueError("conditional training must use the fixed diagnostic-only endpoint")
    diffusion_raw = _mapping(config, "conditional_diffusion")
    if (
        int(diffusion_raw.get("num_timesteps", -1)) != 1000
        or float(diffusion_raw.get("beta_start", float("nan"))) != 1.0e-4
        or float(diffusion_raw.get("beta_end", float("nan"))) != 0.02
        or diffusion_raw.get("schedule") != "linear"
        or diffusion_raw.get("prediction_target") != "epsilon"
        or diffusion_raw.get("initial_distribution")
        != "standard_normal_at_timestep_999"
    ):
        raise ValueError("conditional diffusion schedule/target/initial state was changed")
    _diffusion_config(config)
    expected_root = Path(
        "/home/infres/yinwang/denoiseNet/results/cgdr/"
        "klados_stage3_conditional_diffusion_matched_v3"
    )
    outputs = _mapping(config, "outputs")
    if Path(str(outputs.get("root", ""))) != expected_root:
        raise ValueError("conditional output root differs from frozen protocol")
    if Path(str(outputs.get("development_root", ""))) != expected_root / "development":
        raise ValueError("conditional development output root is invalid")
    if tuple(config.get("required_comparison_fields", ())) != (
        REQUIRED_CONDITIONAL_ROW_FIELDS
    ):
        raise ValueError("conditional result schema differs from the frozen fields")


def _output_paths(config: Mapping[str, Any], operator_scope: str) -> dict[str, Path]:
    if operator_scope not in FROZEN_OPERATOR_SOURCES:
        raise ValueError(f"unknown operator scope: {operator_scope!r}")
    root = Path(str(_mapping(config, "outputs")["root"]))
    return {
        "checkpoint": root / "checkpoints" / operator_scope / "last.pt",
        "final_checkpoint": root / "checkpoints" / operator_scope / "final.pt",
        "history": root / "training" / operator_scope / "training_history.csv",
        "summary": root / "training" / operator_scope / "result_summary.json",
    }


def _normalizer_state(normalizer: ChannelNormalizer) -> dict[str, Any]:
    return {
        "mean": normalizer.mean.tolist(),
        "standard_deviation": normalizer.standard_deviation.tolist(),
        "source_records": list(normalizer.source_records),
        "sample_count": int(normalizer.sample_count),
    }


def _normalizer_from_state(payload: Mapping[str, Any]) -> ChannelNormalizer:
    return ChannelNormalizer(
        mean=np.asarray(payload["mean"], dtype=np.float64),
        standard_deviation=np.asarray(
            payload["standard_deviation"], dtype=np.float64
        ),
        source_records=tuple(int(value) for value in payload["source_records"]),
        sample_count=int(payload["sample_count"]),
    )


def _matched_data(
    config: Mapping[str, Any], *, operator_scope: str, device: torch.device
) -> _MatchedData:
    deterministic = _deterministic_config(config)
    base = _base_config(deterministic)
    records = load_klados_records(_loader_config(base))
    computed_normalizer = fit_channel_normalizer(records, KLADOS_TRAIN_RECORDS)
    population = load_population_projector(base)
    deterministic_path = _scope_output_paths(deterministic, operator_scope)[
        "best_checkpoint"
    ]
    deterministic_payload = load_training_checkpoint(
        deterministic_path, map_location=device
    )
    checkpoint_normalizer = _validate_deterministic_checkpoint_payload(
        deterministic,
        base,
        deterministic_payload,
        operator_source=operator_scope,
    )
    if not _same_normalizer(computed_normalizer, checkpoint_normalizer):
        raise ValueError("conditional data normalization differs from deterministic")
    train_eligibility = _common_matching_eligibility(
        base,
        records=records,
        normalizer=computed_normalizer,
        source_records=KLADOS_TRAIN_RECORDS,
    )
    development_eligibility = _common_matching_eligibility(
        base,
        records=records,
        normalizer=computed_normalizer,
        source_records=KLADOS_DEVELOPMENT_RECORDS,
    )
    population_projector = np.asarray(population.projector, dtype=np.float64)
    train_bundle = _window_bundle(
        deterministic,
        base,
        records=records,
        normalizer=computed_normalizer,
        population_projector=population_projector,
        source_records=KLADOS_TRAIN_RECORDS,
        operator_source=operator_scope,
        common_eligible_source_records=train_eligibility.included_record_ids,
    )
    development_bundle = _window_bundle(
        deterministic,
        base,
        records=records,
        normalizer=computed_normalizer,
        population_projector=population_projector,
        source_records=KLADOS_DEVELOPMENT_RECORDS,
        operator_source=operator_scope,
        common_eligible_source_records=development_eligibility.included_record_ids,
    )
    train_coverage = _bundle_record_summary(train_bundle)
    development_coverage = _bundle_record_summary(development_bundle)
    extra = deterministic_payload.get("extra", {})
    train_window_count = int(train_bundle.observed.shape[0])
    development_window_count = int(development_bundle.observed.shape[0])
    if int(extra.get("training_windows", -1)) != train_window_count or int(
        extra.get("validation_windows", -1)
    ) != development_window_count:
        raise ValueError("conditional and deterministic window counts differ")
    if extra.get("training_record_coverage") != train_coverage or extra.get(
        "validation_record_coverage"
    ) != development_coverage:
        raise ValueError("conditional and deterministic common record coverage differs")
    return _MatchedData(
        deterministic_config=deterministic,
        base=base,
        normalizer=computed_normalizer,
        population=population,
        train_dataset=_tensor_dataset(train_bundle),
        development_dataset=_tensor_dataset(development_bundle),
        train_coverage=train_coverage,
        development_coverage=development_coverage,
        deterministic_payload=deterministic_payload,
    )


def _model(
    config: Mapping[str, Any], deterministic: Mapping[str, Any]
) -> OperatorConditionedEEGDiffusion:
    return OperatorConditionedEEGDiffusion(
        _model_config(deterministic),
        _diffusion_config(config),
        enforce_scientific_schedule=True,
    )


def _checkpoint_contract(
    config: Mapping[str, Any], matched: _MatchedData, operator_scope: str
) -> dict[str, Any]:
    training = _mapping(matched.deterministic_config, "deterministic_training")
    return {
        "protocol_id": PROTOCOL_ID,
        "claim_scope": str(config["claim_scope"]),
        "operator_scope": operator_scope,
        "operator_scope_deployable": _scope_deployable(operator_scope),
        "matched_deterministic_protocol_id": matched.deterministic_config["protocol_id"],
        "training_source_records": list(KLADOS_TRAIN_RECORDS),
        "development_source_records": list(KLADOS_DEVELOPMENT_RECORDS),
        "training_record_coverage": matched.train_coverage,
        "development_record_coverage": matched.development_coverage,
        "target_optimizer_updates": FIXED_OPTIMIZER_UPDATES,
        "checkpoint_selection": "fixed_endpoint_no_development_outcome_selection",
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip": float(training["gradient_clip"]),
        "mixed_precision": bool(training["mixed_precision"]),
        "workers": int(training["workers"]),
        "backbone": asdict(_model_config(matched.deterministic_config)),
        "diffusion": asdict(_diffusion_config(config)),
        "objective": "valid_time_masked_epsilon_MSE",
        "visible_inputs": list(OperatorConditionedEEGDiffusion.visible_input_fields),
        "different_objective_from_deterministic": True,
    }


def _batch_loss(
    model: OperatorConditionedEEGDiffusion,
    batch: Sequence[torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    observed, clean, projector, attenuation, mask = (
        value.to(device, non_blocking=True) for value in batch
    )
    return model.training_loss(
        clean,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
    )


def _development_loss(
    model: OperatorConditionedEEGDiffusion,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
    seed: int,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            observed, clean, projector, attenuation, mask = (
                value.to(device, non_blocking=True) for value in batch
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                loss = model.training_loss(
                    clean,
                    observed=observed,
                    projector=projector,
                    attenuation=attenuation,
                    valid_time_mask=mask,
                    generator=generator,
                )
            total += float(loss) * int(observed.shape[0])
            count += int(observed.shape[0])
    if count < 1 or not math.isfinite(total):
        raise FloatingPointError("conditional development epsilon loss is invalid")
    return total / count


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty conditional diffusion table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def train_operator_conditioned_diffusion(
    config: dict[str, Any],
    *,
    operator_scope: str,
    run_dir: Path,
    device: torch.device,
) -> ConditionalTrainingResult:
    """Train to the exact same-scope deterministic checkpoint update count."""

    validate_conditional_config(config)
    if operator_scope not in FROZEN_OPERATOR_SOURCES:
        raise ValueError(f"unknown operator scope: {operator_scope!r}")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("conditional diffusion training requires scheduled CUDA")
    matched = _matched_data(config, operator_scope=operator_scope, device=device)
    deterministic_updates = int(matched.deterministic_payload["step"])
    target_updates = FIXED_OPTIMIZER_UPDATES
    if deterministic_updates != target_updates:
        raise ValueError("matched deterministic checkpoint is not the fixed 6000-step endpoint")
    if matched.deterministic_payload.get("extra", {}).get(
        "checkpoint_selection_used_development_loss"
    ) is not False:
        raise ValueError("matched deterministic endpoint was selected by development loss")
    _, deterministic_summary = _deterministic_endpoint_audit(
        matched.deterministic_config,
        operator_scope=operator_scope,
        device=device,
    )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = _model(config, matched.deterministic_config).to(device)
    deterministic_training = _mapping(
        matched.deterministic_config, "deterministic_training"
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(deterministic_training["learning_rate"]),
        weight_decay=float(deterministic_training["weight_decay"]),
    )
    amp = bool(deterministic_training["mixed_precision"])
    conditional_training = _mapping(config, "training")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp,
        init_scale=float(conditional_training["amp_initial_scale"]),
    )
    contract = _checkpoint_contract(config, matched, operator_scope)
    normalizer_state = _normalizer_state(matched.normalizer)
    paths = _output_paths(config, operator_scope)
    start_epoch = 0
    global_step = 0
    optimizer_step_attempts = 0
    skipped_optimizer_steps = 0
    resumed = False
    cumulative_prior_walltime = 0.0
    last_development_loss = float("nan")
    history: list[dict[str, Any]] = []
    if bool(_mapping(config, "training")["resume"]) and paths["checkpoint"].is_file():
        state = resume_training_checkpoint(
            paths["checkpoint"],
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=contract,
            map_location=device,
        )
        if state.normalizer_state != normalizer_state:
            raise ValueError("conditional checkpoint normalizer differs from deterministic")
        start_epoch = state.epoch + 1
        global_step = state.step
        optimizer_step_attempts = int(
            state.extra.get("optimizer_step_attempts", global_step)
        )
        skipped_optimizer_steps = int(
            state.extra.get("skipped_optimizer_steps_amp_overflow", 0)
        )
        cumulative_prior_walltime = float(
            state.extra.get("cumulative_training_walltime_seconds", 0.0)
        )
        last_development_loss = float(
            state.extra.get("last_development_epsilon_loss", float("nan"))
        )
        resumed = True
        if global_step > target_updates:
            raise ValueError("conditional checkpoint exceeds matched update budget")
        if (
            state.extra.get("fixed_endpoint_update") != FIXED_OPTIMIZER_UPDATES
            or state.extra.get("checkpoint_selection_used_development_loss") is not False
        ):
            raise ValueError("conditional resume checkpoint violates fixed-endpoint selection")
        if paths["history"].is_file():
            with paths["history"].open("r", encoding="utf-8", newline="") as stream:
                history = [dict(row) for row in csv.DictReader(stream)]

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGUSR1, request_stop)
    checkpoint_interval = int(
        _mapping(config, "training")["checkpoint_interval_updates"]
    )
    next_checkpoint = max(
        checkpoint_interval,
        ((global_step // checkpoint_interval) + 1) * checkpoint_interval,
    )
    epoch = start_epoch
    try:
        while global_step < target_updates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1000 + epoch)
            loader = DataLoader(
                matched.train_dataset,
                batch_size=int(deterministic_training["batch_size"]),
                shuffle=True,
                generator=generator,
                num_workers=int(deterministic_training["workers"]),
                pin_memory=True,
                drop_last=False,
                persistent_workers=False,
            )
            model.train()
            total_loss = 0.0
            batches = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp
                ):
                    loss = _batch_loss(model, batch, device=device)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite conditional loss epoch={epoch} step={global_step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(deterministic_training["gradient_clip"])
                )
                optimizer_step_attempts += 1
                optimizer_step_succeeded = scaler_optimizer_step_succeeded(
                    scaler, optimizer
                )
                total_loss += float(loss.detach())
                batches += 1
                if optimizer_step_succeeded:
                    global_step += 1
                else:
                    skipped_optimizer_steps += 1
                    if skipped_optimizer_steps > int(
                        conditional_training["maximum_skipped_optimizer_steps"]
                    ):
                        raise FloatingPointError(
                            "conditional AMP overflow exceeded the frozen zero-skip budget"
                        )
                if global_step >= target_updates:
                    break
            should_checkpoint = (
                global_step >= next_checkpoint
                or global_step >= target_updates
                or stop_requested
            )
            if should_checkpoint:
                last_development_loss = _development_loss(
                    model,
                    matched.development_dataset,
                    batch_size=int(deterministic_training["batch_size"]),
                    device=device,
                    amp=amp,
                    seed=seed + 900000,
                )
                cumulative = (
                    cumulative_prior_walltime + time.perf_counter() - started
                )
                extra = {
                    "operator_scope": operator_scope,
                    "operator_scope_deployable": _scope_deployable(operator_scope),
                    "training_windows": len(matched.train_dataset),
                    "validation_windows": len(matched.development_dataset),
                    "training_record_coverage": matched.train_coverage,
                    "validation_record_coverage": matched.development_coverage,
                    "target_optimizer_updates": target_updates,
                    "actual_optimizer_updates": global_step,
                    "optimizer_step_attempts": optimizer_step_attempts,
                    "skipped_optimizer_steps_amp_overflow": skipped_optimizer_steps,
                    "fixed_endpoint_update": FIXED_OPTIMIZER_UPDATES,
                    "checkpoint_selection_used_development_loss": False,
                    "last_development_epsilon_loss": last_development_loss,
                    "development_loss_used_for_selection": False,
                    "cumulative_training_walltime_seconds": cumulative,
                    "training_terminal": global_step >= target_updates,
                }
                save_training_checkpoint(
                    paths["checkpoint"],
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    config=contract,
                    normalizer=normalizer_state,
                    extra=extra,
                )
                history.append(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "train_epsilon_mse": total_loss / max(batches, 1),
                        "development_epsilon_mse_diagnostic_only": (
                            last_development_loss
                        ),
                    }
                )
                _write_csv(paths["history"], history)
                while next_checkpoint <= global_step:
                    next_checkpoint += checkpoint_interval
            epoch += 1
            if stop_requested:
                break
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)

    status = (
        "completed"
        if global_step >= target_updates
        else "checkpointed_for_resume"
    )
    cumulative_walltime = cumulative_prior_walltime + time.perf_counter() - started
    if status == "completed":
        if global_step != target_updates:
            raise RuntimeError("conditional training did not match deterministic updates")
        payload = load_training_checkpoint(paths["checkpoint"], map_location=device)
        if payload["config"] != contract or int(payload["step"]) != target_updates:
            raise ValueError("conditional terminal checkpoint contract is invalid")
        save_training_checkpoint(
            paths["final_checkpoint"],
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=int(payload["epoch"]),
            step=target_updates,
            config=contract,
            normalizer=normalizer_state,
            extra=payload["extra"],
        )
    deterministic_parameters = sum(
        parameter.numel()
        for parameter in TaskMatchedDeterministicUNet(
            _model_config(matched.deterministic_config)
        ).parameters()
    )
    summary = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "claim_scope": str(config["claim_scope"]),
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "operator_scope": operator_scope,
        "operator_scope_deployable": _scope_deployable(operator_scope),
        "training_source_records": list(KLADOS_TRAIN_RECORDS),
        "development_source_records": list(KLADOS_DEVELOPMENT_RECORDS),
        "historical_records_used": False,
        "training_record_coverage": matched.train_coverage,
        "development_record_coverage": matched.development_coverage,
        "training_windows": len(matched.train_dataset),
        "development_windows": len(matched.development_dataset),
        "deterministic_training_windows": int(
            matched.deterministic_payload["extra"]["training_windows"]
        ),
        "deterministic_development_windows": int(
            matched.deterministic_payload["extra"]["validation_windows"]
        ),
        "target_optimizer_updates": target_updates,
        "actual_optimizer_updates": global_step,
        "optimizer_step_attempts": optimizer_step_attempts,
        "skipped_optimizer_steps_amp_overflow": skipped_optimizer_steps,
        "exact_update_budget_matched": global_step == target_updates,
        "checkpoint_selection_used_development_loss": False,
        "fixed_endpoint_update": FIXED_OPTIMIZER_UPDATES,
        "deterministic_fixed_checkpoint_updates": int(
            matched.deterministic_payload["step"]
        ),
        "deterministic_actual_training_updates": int(
            deterministic_summary["steps_completed"]
        ),
        "deterministic_training_walltime_seconds": float(
            deterministic_summary["cumulative_training_walltime_seconds"]
        ),
        "development_loss_used_for_selection": False,
        "last_development_epsilon_loss": last_development_loss,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "deterministic_model_parameters": deterministic_parameters,
        "training_walltime_seconds": cumulative_walltime,
        "peak_memory_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
        ),
        "checkpoint": str(paths["checkpoint"].resolve()),
        "final_checkpoint": str(paths["final_checkpoint"].resolve()),
        "matched_deterministic_checkpoint": str(
            _scope_output_paths(matched.deterministic_config, operator_scope)[
                "best_checkpoint"
            ].resolve()
        ),
        "resumed": resumed,
        "training_objective": "valid_time_masked_epsilon_MSE",
        "deterministic_training_objective": "paired_task_loss",
        "visible_inputs_equal": True,
        "window_bundle_equal": True,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return ConditionalTrainingResult(
        status=status,
        operator_scope=operator_scope,
        checkpoint=paths["checkpoint"],
        final_checkpoint=paths["final_checkpoint"],
        target_updates=target_updates,
        actual_updates=global_step,
        resumed=resumed,
    )


def load_operator_conditioned_diffusion(
    config: Mapping[str, Any],
    *,
    operator_scope: str,
    device: torch.device,
) -> tuple[OperatorConditionedEEGDiffusion, ChannelNormalizer, dict[str, Any]]:
    validate_conditional_config(config)
    matched = _matched_data(config, operator_scope=operator_scope, device=device)
    path = _output_paths(config, operator_scope)["final_checkpoint"]
    payload = load_training_checkpoint(path, map_location=device)
    contract = _checkpoint_contract(config, matched, operator_scope)
    if payload["config"] != contract:
        raise ValueError("conditional checkpoint contract differs from frozen protocol")
    if int(payload["step"]) != FIXED_OPTIMIZER_UPDATES:
        raise ValueError("conditional checkpoint is not the fixed 6000-step endpoint")
    if int(matched.deterministic_payload["step"]) != FIXED_OPTIMIZER_UPDATES:
        raise ValueError("conditional and deterministic optimizer updates differ")
    model = _model(config, matched.deterministic_config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    normalizer = _normalizer_from_state(payload["normalizer_state"])
    if not _same_normalizer(normalizer, matched.normalizer):
        raise ValueError("conditional inference normalizer differs from training")
    return model, normalizer, payload


def _deterministic_endpoint_audit(
    deterministic: Mapping[str, Any],
    *,
    operator_scope: str,
    device: torch.device | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently load and validate the fixed deterministic endpoint and cost."""

    base = _base_config(deterministic)
    paths = _scope_output_paths(deterministic, operator_scope)
    payload = load_training_checkpoint(paths["best_checkpoint"], map_location=device)
    _validate_deterministic_checkpoint_payload(
        deterministic,
        base,
        payload,
        operator_source=operator_scope,
    )
    summary = json.loads(
        paths["result_summary"].read_text(encoding="utf-8")
    )
    if (
        summary.get("protocol_id") != DETERMINISTIC_PROTOCOL_ID
        or summary.get("operator_scope") != operator_scope
        or not str(summary.get("status", "")).startswith("completed")
        or int(summary.get("steps_completed", -1)) != FIXED_OPTIMIZER_UPDATES
        or int(payload["step"]) != FIXED_OPTIMIZER_UPDATES
        or summary.get("checkpoint_selection_used_development_loss") is not False
    ):
        raise ValueError("deterministic endpoint/cost audit differs from fixed v4")
    return payload, summary


def _frozen_common_eligible_records(
    deterministic: Mapping[str, Any],
) -> set[str]:
    """Read eligibility from the three fixed deterministic training summaries."""

    reference: Mapping[str, Any] | None = None
    for scope in FROZEN_OPERATOR_SOURCES:
        path = _scope_output_paths(deterministic, scope)["result_summary"]
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (
            summary.get("protocol_id") != DETERMINISTIC_PROTOCOL_ID
            or not str(summary.get("status", "")).startswith("completed")
        ):
            raise ValueError("deterministic v4 eligibility summary is not complete")
        coverage = summary.get("validation_record_coverage")
        if not isinstance(coverage, Mapping):
            raise ValueError("deterministic v4 summary lacks validation coverage")
        if reference is None:
            reference = coverage
        elif coverage != reference:
            raise ValueError("deterministic v4 scopes disagree on common eligibility")
    if reference is None:
        raise AssertionError("no deterministic v4 eligibility summary was loaded")
    requested = {int(value) for value in reference.get("requested_record_ids", ())}
    included = {int(value) for value in reference.get("included_record_ids", ())}
    skipped = {int(value) for value in reference.get("skipped_record_ids", ())}
    if (
        requested != set(KLADOS_DEVELOPMENT_RECORDS)
        or included & skipped
        or included | skipped != requested
    ):
        raise ValueError("deterministic v4 common eligibility coverage is invalid")
    return {f"sim{record:02d}" for record in included}


def _assert_required_conditional_row(row: Mapping[str, Any]) -> None:
    missing = sorted(set(REQUIRED_CONDITIONAL_ROW_FIELDS) - set(row))
    if missing:
        raise ValueError(f"conditional result row is missing required fields: {missing}")
    if row.get("method_id") != METHOD_ID:
        raise ValueError("conditional result row has the wrong method ID")
    if row.get("operator_source") not in FROZEN_OPERATOR_SOURCES:
        raise ValueError("conditional result row has an unknown operator scope")
    if row.get("common_eligibility_status") != "included":
        raise ValueError("conditional performance row is outside common eligibility")
    for field in (
        "fixed_optimizer_updates_each",
        "conditional_actual_optimizer_updates",
        "deterministic_fixed_checkpoint_updates",
        "deterministic_actual_training_updates",
    ):
        if int(row[field]) != FIXED_OPTIMIZER_UPDATES:
            raise ValueError(f"conditional result row violates fixed endpoint: {field}")
    paired_supervision = row.get("same_paired_supervision_exposure")
    objectives_equal = row.get("training_objectives_equal")
    if (
        str(paired_supervision).lower() != "true"
        or row.get("conditional_training_objective")
        != "valid_time_masked_epsilon_MSE"
        or row.get("deterministic_training_objective") != "paired_task_loss"
        or str(objectives_equal).lower() != "false"
    ):
        raise ValueError("conditional result row misstates supervision/objectives")
    if str(row.get("status", "")).startswith("success"):
        for field in (
            "latency_seconds",
            "peak_memory_mb",
            "function_evaluations_per_seed_per_window",
            "total_function_evaluations_per_window",
            "algorithmic_seed_count",
        ):
            if row[field] == "":
                raise ValueError(f"successful conditional row lacks {field}")


def _comparison_cell_status(
    row: Mapping[str, Any] | None,
    *,
    record_is_eligible: bool,
    family: str,
) -> str:
    """Retain ineligible, failed and missing cells in the aggregate matrix."""

    if row is not None:
        return str(row.get("status", "unknown"))
    if not record_is_eligible:
        return "ineligible_common_record"
    if family not in ("conditional", "reference"):
        raise ValueError(f"unknown comparison family: {family!r}")
    return f"unmatched_missing_{family}_cell"


def _paired_cell_status(conditional_status: str, deterministic_status: str) -> str:
    if conditional_status.startswith("success") and deterministic_status.startswith(
        "success"
    ):
        return "success_paired"
    return f"conditional={conditional_status};reference={deterministic_status}"


def _continuous(windows: np.ndarray, samples: int) -> np.ndarray:
    value = np.asarray(windows, dtype=np.float64)
    return value.transpose(1, 0, 2).reshape(value.shape[1], -1)[:, :samples]


def _conditional_restore(
    model: OperatorConditionedEEGDiffusion,
    prepared: Any,
    projector: np.ndarray,
    attenuation: np.ndarray,
    *,
    seeds: Sequence[int],
    ddim_steps: int,
    eta: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_outputs: list[np.ndarray] = []
    total_forward_invocations = 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        parts: list[np.ndarray] = []
        for start in range(0, prepared.observed_windows.shape[0], batch_size):
            stop = min(start + batch_size, prepared.observed_windows.shape[0])
            observed = torch.as_tensor(
                prepared.observed_windows[start:stop],
                dtype=torch.float32,
                device=device,
            )
            mask = torch.as_tensor(
                prepared.valid_time_weight[start:stop],
                dtype=torch.float32,
                device=device,
            )
            attenuation_batch = torch.as_tensor(
                attenuation[start:stop], dtype=torch.float32, device=device
            )
            sample = model.sample_ddim(
                observed=observed,
                projector=torch.as_tensor(
                    projector, dtype=torch.float32, device=device
                ),
                attenuation=attenuation_batch,
                valid_time_mask=mask,
                ddim_steps=ddim_steps,
                eta=eta,
                generator=generator,
            )
            total_forward_invocations += sample.network_calls
            parts.append(sample.restored.cpu().numpy())
        seed_outputs.append(np.concatenate(parts, axis=0))
    torch.cuda.synchronize(device)
    posterior_mean = np.mean(np.stack(seed_outputs, axis=0), axis=0)
    restored = _continuous(
        posterior_mean, prepared.observed_continuous.shape[1]
    )
    return restored, {
        "function_evaluations": ddim_steps,
        "function_evaluations_per_seed_per_window": ddim_steps,
        "total_function_evaluations_per_window": len(seeds) * ddim_steps,
        "network_forward_invocations": total_forward_invocations,
        "network_calls_total": total_forward_invocations,
        "algorithmic_seed_count": len(seeds),
        "seeds_are_statistical_units": False,
        "latency_seconds": time.perf_counter() - started,
        "peak_memory_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
        ),
    }


def run_conditional_development_record(
    config: dict[str, Any],
    *,
    task_index: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one complete common-eligible development source record."""

    validate_conditional_config(config)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("conditional development comparison requires scheduled CUDA")
    if not 0 <= int(task_index) < len(KLADOS_DEVELOPMENT_RECORDS):
        raise ValueError("conditional development task index must lie in [0,7]")
    source_record = int(KLADOS_DEVELOPMENT_RECORDS[int(task_index)])
    deterministic = _deterministic_config(config)
    base = _base_config(deterministic)
    records = load_klados_records(_loader_config(base))
    normalizer = fit_channel_normalizer(records, KLADOS_TRAIN_RECORDS)
    eligibility = _common_matching_eligibility(
        base,
        records=records,
        normalizer=normalizer,
        source_records=KLADOS_DEVELOPMENT_RECORDS,
    )
    output_dir = Path(str(_mapping(config, "outputs")["development_root"])) / (
        f"sim{source_record:02d}"
    )
    if source_record not in eligibility.included_record_ids:
        summary = {
            "status": "ineligible_common_record",
            "protocol_id": PROTOCOL_ID,
            "source_record": f"sim{source_record:02d}",
            "common_eligibility_status": "excluded_matching_p0_ineligible",
            "reasons": list(eligibility.skipped_reasons.get(source_record, ())),
            "confirmatory": False,
            "formal_G1_or_G3_evidence": False,
            "performance_metrics_emitted": False,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary
    models: dict[str, OperatorConditionedEEGDiffusion] = {}
    payloads: dict[str, dict[str, Any]] = {}
    deterministic_payloads: dict[str, dict[str, Any]] = {}
    deterministic_summaries: dict[str, dict[str, Any]] = {}
    loaded_normalizer: ChannelNormalizer | None = None
    for scope in FROZEN_OPERATOR_SOURCES:
        scope_model, scope_normalizer, payload = load_operator_conditioned_diffusion(
            config, operator_scope=scope, device=device
        )
        if loaded_normalizer is None:
            loaded_normalizer = scope_normalizer
        elif not _same_normalizer(loaded_normalizer, scope_normalizer):
            raise ValueError("conditional operator-scope normalizers differ")
        models[scope] = scope_model
        payloads[scope] = payload
        deterministic_payload, deterministic_summary = _deterministic_endpoint_audit(
            deterministic,
            operator_scope=scope,
            device=device,
        )
        deterministic_payloads[scope] = deterministic_payload
        deterministic_summaries[scope] = deterministic_summary
    if loaded_normalizer is None:
        raise AssertionError("no conditional checkpoint was loaded")
    population = load_population_projector(base)
    native = select_records(records, (source_record,))[0]
    prepared = prepare_mechanism_record(
        native,
        loaded_normalizer,
        source_rate=int(_mapping(base, "klados")["source_sampling_rate"]),
        target_rate=int(_mapping(base, "preprocessing")["target_sampling_rate"]),
        window_samples=int(_mapping(base, "preprocessing")["window_samples"]),
        calibration_seconds=float(_mapping(base, "klados")["calibration_seconds"]),
        guard_seconds=float(_mapping(base, "klados")["guard_seconds"]),
    )
    arms, oracle = _operator_arms(prepared, population, base)
    if not arms["matching_p0"].eligible:
        raise RuntimeError("common matching eligibility changed at development inference")
    attenuation = _attenuation_windows(prepared, base)
    eog_magnitude = np.sqrt(np.mean(np.square(prepared.eog_continuous), axis=0))
    artifact_mask = eog_magnitude >= float(
        _mapping(base, "observation")["artifact_eog_z_threshold"]
    )
    inference = _mapping(config, "development_inference")
    seeds = tuple(int(value) for value in inference["seeds"])
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    supervision = (
        "paired_supervised_epsilon_prediction_operator_conditioned_diffusion"
    )
    for scope in FROZEN_OPERATOR_SOURCES:
        arm = arms[scope]
        projector = np.asarray(arm.projector, dtype=np.float64)
        payload = payloads[scope]
        deterministic_payload = deterministic_payloads[scope]
        deterministic_summary = deterministic_summaries[scope]
        conditional_parameters = sum(
            parameter.numel() for parameter in models[scope].parameters()
        )
        common_runtime = {
            "training_updates": int(payload["step"]),
            "model_parameters": conditional_parameters,
            "training_walltime_seconds": float(
                payload["extra"].get("cumulative_training_walltime_seconds", float("nan"))
            ),
            "common_eligibility_status": "included",
            "conditional_checkpoint_operator_scope": scope,
            "matched_deterministic_updates": int(deterministic_payload["step"]),
            "conditional_training_windows": int(payload["extra"]["training_windows"]),
            "conditional_development_windows": int(
                payload["extra"]["validation_windows"]
            ),
            "deterministic_training_windows": int(
                deterministic_payload["extra"]["training_windows"]
            ),
            "deterministic_development_windows": int(
                deterministic_payload["extra"]["validation_windows"]
            ),
            "fixed_optimizer_updates_each": FIXED_OPTIMIZER_UPDATES,
            "conditional_actual_optimizer_updates": int(payload["step"]),
            "deterministic_fixed_checkpoint_updates": int(
                deterministic_payload["step"]
            ),
            "deterministic_actual_training_updates": int(
                deterministic_summary["steps_completed"]
            ),
            "conditional_model_parameters": conditional_parameters,
            "deterministic_model_parameters": int(
                deterministic_summary["model_parameters"]
            ),
            "conditional_training_walltime_seconds": float(
                payload["extra"].get(
                    "cumulative_training_walltime_seconds", float("nan")
                )
            ),
            "deterministic_training_walltime_seconds": float(
                deterministic_summary["cumulative_training_walltime_seconds"]
            ),
            "same_paired_supervision_exposure": True,
            "training_objectives_equal": False,
            "operator_specificity_interpretation_allowed": False,
        }
        try:
            restored, sampling_runtime = _conditional_restore(
                models[scope],
                prepared,
                projector,
                attenuation,
                seeds=seeds,
                ddim_steps=int(inference["ddim_steps"]),
                eta=float(inference["eta"]),
                batch_size=int(inference["inference_batch_size"]),
                device=device,
            )
            row = _safe_metric_row(
                failure_rows=failure_rows,
                partition="development",
                prepared=prepared,
                method_id=METHOD_ID,
                operator_source=scope,
                effective_operator_source=scope,
                restored=restored,
                projector=projector,
                oracle=oracle,
                artifact_mask=artifact_mask,
                runtime={**sampling_runtime, **common_runtime},
                fallback_used=False,
                query_clean_target_used_by_method=arm.query_clean_target_used,
                comparator_supervision=supervision,
            )
        except Exception as error:
            retained_error: Exception = error
            if isinstance(error, ValueError) and any(
                token in str(error).lower()
                for token in ("nan", "inf", "non-finite", "finite tensor")
            ):
                retained_error = FloatingPointError(str(error))
            if not _retainable_method_failure(retained_error):
                raise
            failure_rows.append(
                {
                    "partition": "development",
                    "source_record": f"sim{prepared.source_record:02d}",
                    "operator_source": scope,
                    "effective_operator_source": scope,
                    "method_id": METHOD_ID,
                    "seed": "",
                    "status": "failed_sampling_numerical",
                    "failure_type": type(retained_error).__name__,
                    "failure_message": str(retained_error),
                }
            )
            row = _failed_metric_row(
                partition="development",
                prepared=prepared,
                method_id=METHOD_ID,
                operator_source=scope,
                effective_operator_source=scope,
                fallback_used=False,
                query_clean_target_used_by_method=arm.query_clean_target_used,
                status="failed_sampling_numerical",
                failure_type=type(retained_error).__name__,
                failure_message=str(retained_error),
                runtime={
                    **common_runtime,
                    "latency_seconds": "",
                    "peak_memory_mb": "",
                    "function_evaluations_per_seed_per_window": "",
                    "total_function_evaluations_per_window": "",
                    "algorithmic_seed_count": len(seeds),
                },
            )
        row["comparator_supervision"] = supervision
        row["same_supervision_G3_comparison"] = False
        row["same_paired_supervision_exposure"] = True
        row["conditional_training_objective"] = "valid_time_masked_epsilon_MSE"
        row["deterministic_training_objective"] = "paired_task_loss"
        row["training_objectives_equal"] = False
        row["operator_specificity_interpretation_allowed"] = False
        _assert_required_conditional_row(row)
        rows.append(row)
    _write_csv(output_dir / "metrics.csv", rows)
    if failure_rows:
        _write_csv(output_dir / "failures.csv", failure_rows)
    summary = {
        "status": "completed_exploratory_conditional_diffusion_development",
        "protocol_id": PROTOCOL_ID,
        "claim_scope": str(config["claim_scope"]),
        "source_record": f"sim{source_record:02d}",
        "records_are_participants": False,
        "common_eligibility_status": "included",
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "diffusion_family_decision_emitted": False,
        "operator_sources": list(FROZEN_OPERATOR_SOURCES),
        "method_id": METHOD_ID,
        "successful_method_arms": sum(
            str(row["status"]).startswith("success") for row in rows
        ),
        "failed_method_arms": sum(
            not str(row["status"]).startswith("success") for row in rows
        ),
        "metrics": str(output_dir / "metrics.csv"),
        "failures": str(output_dir / "failures.csv") if failure_rows else "",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def aggregate_conditional_development(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Aggregate development diagnostics without a category-level verdict."""

    validate_conditional_config(config)
    outputs = _mapping(config, "outputs")
    output_root = Path(str(outputs["root"]))
    root = Path(str(outputs["development_root"]))
    training_endpoint_summaries: list[dict[str, Any]] = []
    for scope in FROZEN_OPERATOR_SOURCES:
        summary_path = output_root / "training" / scope / "result_summary.json"
        if not summary_path.is_file():
            raise ValueError(f"missing conditional training summary: {scope}")
        training_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            training_summary.get("status") != "completed"
            or training_summary.get("protocol_id") != PROTOCOL_ID
            or training_summary.get("operator_scope") != scope
            or int(training_summary.get("target_optimizer_updates", -1))
            != FIXED_OPTIMIZER_UPDATES
            or int(training_summary.get("actual_optimizer_updates", -1))
            != FIXED_OPTIMIZER_UPDATES
            or int(training_summary.get("optimizer_step_attempts", -1))
            != FIXED_OPTIMIZER_UPDATES
            or int(training_summary.get("skipped_optimizer_steps_amp_overflow", -1))
            != 0
            or training_summary.get("exact_update_budget_matched") is not True
            or training_summary.get("resumed") is not False
        ):
            raise ValueError(
                f"conditional training endpoint contract failed: {scope}"
            )
        training_endpoint_summaries.append(training_summary)
    deterministic = _deterministic_config(config)
    deterministic_root = Path(
        str(_mapping(deterministic, "outputs")["development_root"])
    )
    eligible_records = _frozen_common_eligible_records(deterministic)
    deterministic_record_summaries: dict[str, dict[str, Any]] = {}
    for record_id in KLADOS_DEVELOPMENT_RECORDS:
        record = f"sim{record_id:02d}"
        summary_path = deterministic_root / record / "result_summary.json"
        if not summary_path.is_file():
            continue
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("deterministic record summary must be a mapping")
        deterministic_record_summaries[record] = value
    prior_parameter_values = {
        int(value["diffusion_prior_parameters"])
        for value in deterministic_record_summaries.values()
        if value.get("diffusion_prior_parameters") not in (None, "")
    }
    if len(prior_parameter_values) > 1:
        raise ValueError("shared diffusion-prior parameter counts disagree")
    shared_prior_parameters: int | str = (
        next(iter(prior_parameter_values)) if prior_parameter_values else ""
    )
    shared_prior_history_steps: int | str = ""
    shared_prior_update_status = "training_history_not_available"
    try:
        deterministic_base = _base_config(deterministic)
        history_path = Path(
            str(_mapping(deterministic_base, "outputs")["training_history"])
        )
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8", newline="") as stream:
                history_rows = list(csv.DictReader(stream))
            history_steps = [
                int(row["step"])
                for row in history_rows
                if row.get("step") not in (None, "")
            ]
            if history_steps:
                shared_prior_history_steps = max(history_steps)
                shared_prior_update_status = (
                    "training_history_steps_captured_amp_skips_not_audited"
                )
    except FileNotFoundError:
        shared_prior_update_status = "training_history_file_missing"
    except (KeyError, TypeError, ValueError) as error:
        shared_prior_update_status = (
            "training_history_malformed_" + type(error).__name__
        )
    rows: list[dict[str, str]] = []
    coverage: list[dict[str, Any]] = []
    for record in KLADOS_DEVELOPMENT_RECORDS:
        directory = root / f"sim{record:02d}"
        summary_path = directory / "result_summary.json"
        metric_path = directory / "metrics.csv"
        if not summary_path.is_file():
            coverage.append(
                {
                    "source_record": f"sim{record:02d}",
                    "status": "unmatched_missing_conditional_result_summary",
                    "common_eligibility_status": "unknown_missing_result_summary",
                    "performance_metrics_emitted": False,
                    "metrics_file_missing": True,
                    "unexpected_metrics_file": metric_path.exists(),
                    "successful_method_arms": 0,
                    "failed_method_arms": 0,
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completed = str(summary["status"]).startswith("completed")
        coverage.append(
            {
                "source_record": f"sim{record:02d}",
                "status": summary["status"],
                "common_eligibility_status": summary["common_eligibility_status"],
                "performance_metrics_emitted": completed and metric_path.is_file(),
                "metrics_file_missing": completed and not metric_path.is_file(),
                "unexpected_metrics_file": not completed and metric_path.exists(),
                "successful_method_arms": int(
                    summary.get("successful_method_arms", 0)
                ),
                "failed_method_arms": int(summary.get("failed_method_arms", 0)),
            }
        )
        if completed and metric_path.is_file():
            with metric_path.open("r", encoding="utf-8", newline="") as stream:
                record_rows = list(csv.DictReader(stream))
            for row in record_rows:
                _assert_required_conditional_row(row)
                if row.get("source_record") != f"sim{record:02d}":
                    raise ValueError("conditional metric row source-record mismatch")
                if row.get("operator_source") not in FROZEN_OPERATOR_SOURCES:
                    raise ValueError("conditional metric row has unknown operator scope")
            rows.extend(record_rows)
    successful_records = {
        row["source_record"] for row in rows if row["status"].startswith("success")
    }
    fully_successful_records = {
        record
        for record in eligible_records
        if sum(
            row["source_record"] == record
            and row["status"].startswith("success")
            for row in rows
        )
        == len(FROZEN_OPERATOR_SOURCES)
    }
    expected_rows = len(eligible_records) * len(FROZEN_OPERATOR_SOURCES)
    conditional_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["source_record"], row["operator_source"])
        if key in conditional_by_key:
            raise ValueError("duplicate conditional comparator cell")
        conditional_by_key[key] = row
    expected_conditional_keys = {
        (record, scope)
        for record in eligible_records
        for scope in FROZEN_OPERATOR_SOURCES
    }
    observed_conditional_keys = set(conditional_by_key)
    missing_conditional_keys = expected_conditional_keys - observed_conditional_keys
    unexpected_conditional_keys = observed_conditional_keys - expected_conditional_keys
    deterministic_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for record_id in KLADOS_DEVELOPMENT_RECORDS:
        record = f"sim{record_id:02d}"
        metric_path = deterministic_root / record / "metrics.csv"
        if not metric_path.is_file():
            continue
        with metric_path.open("r", encoding="utf-8", newline="") as stream:
            for deterministic_row in csv.DictReader(stream):
                method = deterministic_row.get("method_id", "")
                scope = deterministic_row.get("operator_source", "")
                if method not in FROZEN_METHODS or scope not in FROZEN_OPERATOR_SOURCES:
                    continue
                if deterministic_row.get("source_record") != record:
                    raise ValueError("reference metric row source-record mismatch")
                key = (record, scope, method)
                if key in deterministic_by_key:
                    raise ValueError("duplicate deterministic comparator cell")
                deterministic_by_key[key] = deterministic_row

    comparison_matrix: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    paired_metrics = (
        "e_parallel",
        "e_perp",
        "rrmse",
        "correlation",
        "psd_distortion",
        "artifact_attenuation",
        "clean_interval_preservation",
    )
    for record_id in KLADOS_DEVELOPMENT_RECORDS:
        record = f"sim{record_id:02d}"
        record_is_eligible = record in eligible_records
        for scope in FROZEN_OPERATOR_SOURCES:
            conditional_row = conditional_by_key.get((record, scope))
            query_oracle = scope == "query_derived_oracle_projector"
            for method in COMPARISON_METHODS:
                source_row = (
                    conditional_row
                    if method == METHOD_ID
                    else deterministic_by_key.get((record, scope, method))
                )
                matrix_row: dict[str, Any] = dict(source_row or {})
                reported_method = (
                    f"hard_Q_{scope}_y"
                    if method == "deterministic_Qy" and not query_oracle
                    else method
                )
                matrix_row.update(
                    {
                        "source_record": record,
                        "operator_source": scope,
                        "method_id": reported_method,
                        "source_method_id": method,
                        "status": _comparison_cell_status(
                            source_row,
                            record_is_eligible=record_is_eligible,
                            family=(
                                "conditional" if method == METHOD_ID else "reference"
                            ),
                        ),
                        "comparison_family": (
                            "operator_conditioned_diffusion"
                            if method == METHOD_ID
                            else "frozen_deterministic_or_current_cgdr_arm"
                        ),
                        "common_eligibility_status": (
                            "included"
                            if record_is_eligible
                            else "excluded_matching_p0_ineligible"
                        ),
                        "query_oracle_geometry": query_oracle,
                        "deployable_operator_source": not query_oracle,
                        "oracle_or_qy_role": (
                            "nondeployable_query_clean_oracle_Qy"
                            if query_oracle and method == "deterministic_Qy"
                            else "nondeployable_query_clean_oracle_geometry"
                            if query_oracle
                            else "deployable_or_support_derived_geometry"
                        ),
                        "operator_specificity_interpretation_allowed": False,
                        "cross_scope_models_share_weights": False,
                        "shared_prior_model_parameters": (
                            shared_prior_parameters if method.startswith("M") else ""
                        ),
                        "shared_prior_training_history_steps": (
                            shared_prior_history_steps
                            if method.startswith("M")
                            else ""
                        ),
                        "shared_prior_update_status": (
                            shared_prior_update_status
                            if method.startswith("M")
                            else ""
                        ),
                    }
                )
                comparison_matrix.append(matrix_row)

            for method in FROZEN_METHODS:
                deterministic_row = deterministic_by_key.get((record, scope, method))
                conditional_status = _comparison_cell_status(
                    conditional_row,
                    record_is_eligible=record_is_eligible,
                    family="conditional",
                )
                deterministic_status = _comparison_cell_status(
                    deterministic_row,
                    record_is_eligible=record_is_eligible,
                    family="reference",
                )
                both_success = conditional_status.startswith(
                    "success"
                ) and deterministic_status.startswith("success")

                def optional_delta(metric: str) -> float | str:
                    if not both_success or conditional_row is None or deterministic_row is None:
                        return ""
                    conditional_value = conditional_row.get(metric, "")
                    deterministic_value = deterministic_row.get(metric, "")
                    if conditional_value == "" or deterministic_value == "":
                        return ""
                    return float(conditional_value) - float(deterministic_value)

                paired_rows.append(
                    {
                        "source_record": record,
                        "operator_source": scope,
                        "reference_method_id": (
                            f"hard_Q_{scope}_y"
                            if method == "deterministic_Qy" and not query_oracle
                            else method
                        ),
                        "source_reference_method_id": method,
                        "reference_method_family": (
                            "current_cgdr_diffusion"
                            if method.startswith("M")
                            else "task_matched_deterministic"
                            if method
                            == "task_matched_multichannel_deterministic_UNet"
                            else "nonlearned_algebraic"
                        ),
                        "pair_status": (
                            _paired_cell_status(
                                conditional_status, deterministic_status
                            )
                        ),
                        "conditional_status": conditional_status,
                        "reference_status": deterministic_status,
                        "confirmatory": False,
                        "formal_G3_evidence": False,
                        "comparison_role": (
                            "exploratory_exact_record_scope_window_pair"
                        ),
                        "same_paired_supervision_exposure": (
                            method == "task_matched_multichannel_deterministic_UNet"
                        ),
                        "training_objectives_equal": (
                            False
                            if method
                            == "task_matched_multichannel_deterministic_UNet"
                            else "N/A"
                        ),
                        "fixed_optimizer_updates_each": (
                            FIXED_OPTIMIZER_UPDATES
                            if method == "task_matched_multichannel_deterministic_UNet"
                            else "N/A"
                        ),
                        "conditional_actual_optimizer_updates": (
                            conditional_row.get(
                                "conditional_actual_optimizer_updates", ""
                            )
                            if conditional_row is not None
                            else ""
                        ),
                        "deterministic_fixed_checkpoint_updates": (
                            conditional_row.get(
                                "deterministic_fixed_checkpoint_updates", ""
                            )
                            if conditional_row is not None
                            and method
                            == "task_matched_multichannel_deterministic_UNet"
                            else ""
                        ),
                        "deterministic_actual_training_updates": (
                            conditional_row.get(
                                "deterministic_actual_training_updates", ""
                            )
                            if conditional_row is not None
                            and method
                            == "task_matched_multichannel_deterministic_UNet"
                            else ""
                        ),
                        "conditional_training_walltime_seconds": (
                            conditional_row.get(
                                "conditional_training_walltime_seconds", ""
                            )
                            if conditional_row is not None
                            else ""
                        ),
                        "deterministic_training_walltime_seconds": (
                            conditional_row.get(
                                "deterministic_training_walltime_seconds", ""
                            )
                            if conditional_row is not None
                            and method
                            == "task_matched_multichannel_deterministic_UNet"
                            else ""
                        ),
                        "query_oracle_geometry": query_oracle,
                        "deterministic_Qy_is_nondeployable_oracle": (
                            query_oracle and method == "deterministic_Qy"
                        ),
                        "operator_specificity_interpretation_allowed": False,
                        "conditional_parameters": (
                            conditional_row.get("conditional_model_parameters", "")
                            if conditional_row is not None
                            else ""
                        ),
                        "deterministic_parameters": (
                            deterministic_row.get("training_model_parameters", "")
                            if deterministic_row is not None
                            else ""
                        ),
                        "conditional_latency_seconds": (
                            conditional_row.get("latency_seconds", "")
                            if conditional_row is not None
                            else ""
                        ),
                        "deterministic_latency_seconds": (
                            deterministic_row.get("latency_seconds", "")
                            if deterministic_row is not None
                            else ""
                        ),
                        **{
                            f"conditional_minus_{method}_{metric}": optional_delta(metric)
                            for metric in paired_metrics
                        },
                    }
                )
    method_summaries: list[dict[str, Any]] = []
    for scope in FROZEN_OPERATOR_SOURCES:
        scope_rows = [row for row in rows if row["operator_source"] == scope]
        selected = [row for row in scope_rows if row["status"].startswith("success")]
        metric_summary = {
            f"median_{metric}": (
                float(np.median([float(row[metric]) for row in selected]))
                if selected
                else ""
            )
            for metric in (
                "e_parallel",
                "e_perp",
                "rrmse",
                "correlation",
                "psd_distortion",
            )
        }
        method_summaries.append(
            {
                "operator_source": scope,
                "method_id": METHOD_ID,
                "successful_source_records": len(selected),
                "failed_source_records": len(scope_rows) - len(selected),
                "unmatched_missing_source_records": len(eligible_records)
                - len(scope_rows),
                "observed_conditional_source_records": len(scope_rows),
                "common_eligible_source_records": len(eligible_records),
                "available_source_records_denominator": len(KLADOS_DEVELOPMENT_RECORDS),
                "fixed_optimizer_updates": FIXED_OPTIMIZER_UPDATES,
                "cross_scope_operator_specificity_interpretation_allowed": False,
                "cross_scope_models_share_weights": False,
                **metric_summary,
                "median_latency_seconds": (
                    float(np.median([float(row["latency_seconds"]) for row in selected]))
                    if selected
                    else ""
                ),
                "maximum_peak_memory_mb": (
                    max(float(row["peak_memory_mb"]) for row in selected)
                    if selected
                    else ""
                ),
            }
        )

    def _numeric_values_from_candidates(
        selected_rows: Sequence[Mapping[str, Any]],
        *keys: str,
        required: bool = False,
    ) -> list[float]:
        values: list[float] = []
        for selected_row in selected_rows:
            found = False
            for key in keys:
                value = selected_row.get(key, "")
                if value in ("", None):
                    continue
                numeric = float(value)
                if math.isfinite(numeric):
                    values.append(numeric)
                    found = True
                break
            if required and not found:
                raise ValueError(
                    "successful common-eligible row lacks finite required metric: "
                    + "/".join(keys)
                )
        return values

    def _median_from_candidates(
        selected_rows: Sequence[Mapping[str, Any]],
        *keys: str,
        required: bool = False,
    ) -> float | str:
        values = _numeric_values_from_candidates(
            selected_rows, *keys, required=required
        )
        return float(np.median(values)) if values else ""

    common_eligible_arm_summaries: list[dict[str, Any]] = []
    for scope in FROZEN_OPERATOR_SOURCES:
        for method in COMPARISON_METHODS:
            selected = [
                row
                for row in comparison_matrix
                if row["operator_source"] == scope
                and row["source_method_id"] == method
                and row["common_eligibility_status"] == "included"
                and str(row["status"]).startswith("success")
            ]
            metric_values = {
                metric: _numeric_values_from_candidates(
                    selected, metric, required=True
                )
                for metric in (
                    "e_parallel",
                    "e_perp",
                    "rrmse",
                    "correlation",
                    "psd_distortion",
                    "artifact_attenuation",
                    "clean_interval_preservation",
                )
            }
            latency_values = _numeric_values_from_candidates(
                selected, "latency_seconds", required=True
            )
            peak_memory_values = _numeric_values_from_candidates(
                selected, "peak_memory_mb", required=True
            )
            function_evaluation_values = _numeric_values_from_candidates(
                selected,
                "function_evaluations_per_seed_per_window",
                "function_evaluations",
                required=True,
            )
            total_function_evaluation_values = _numeric_values_from_candidates(
                selected,
                "total_function_evaluations_per_window",
                required=True,
            )
            if method == METHOD_ID:
                model_parameters = _median_from_candidates(
                    selected, "conditional_model_parameters", required=True
                )
                optimizer_updates = _median_from_candidates(
                    selected,
                    "conditional_actual_optimizer_updates",
                    required=True,
                )
                training_walltime = _median_from_candidates(
                    selected,
                    "conditional_training_walltime_seconds",
                    required=True,
                )
                training_history_steps: float | str = ""
                training_cost_scope = "operator_scope_conditional_training"
                training_walltime_status = "captured"
                optimizer_update_semantics = "successful_optimizer_updates_audited"
            elif method == "task_matched_multichannel_deterministic_UNet":
                model_parameters = _median_from_candidates(
                    selected, "training_model_parameters", required=True
                )
                optimizer_updates = _median_from_candidates(
                    selected,
                    "training_updates_completed",
                    "training_updates",
                    required=True,
                )
                training_walltime = _median_from_candidates(
                    selected, "training_walltime_seconds", required=True
                )
                training_history_steps = ""
                training_cost_scope = "operator_scope_deterministic_training"
                training_walltime_status = "captured"
                optimizer_update_semantics = "successful_optimizer_updates_audited"
            elif method.startswith("M"):
                model_parameters = _median_from_candidates(
                    selected, "shared_prior_model_parameters", required=True
                )
                optimizer_updates = ""
                training_history_steps = _median_from_candidates(
                    selected,
                    "shared_prior_training_history_steps",
                    required=True,
                )
                training_walltime = ""
                training_cost_scope = "shared_pretrained_clean_prior"
                training_walltime_status = "not_captured_in_prior_artifact"
                optimizer_update_semantics = (
                    "training_history_steps_amp_skips_not_audited"
                )
            else:
                model_parameters = ""
                optimizer_updates = ""
                training_history_steps = ""
                training_walltime = ""
                training_cost_scope = "no_learned_training"
                training_walltime_status = "not_applicable"
                optimizer_update_semantics = "not_applicable"
            common_eligible_arm_summaries.append(
                {
                    "operator_source": scope,
                    "source_method_id": method,
                    "reported_method_id": (
                        selected[0]["method_id"] if selected else method
                    ),
                    "comparison_role": (
                        "exploratory_exact_common_eligible_source_records"
                    ),
                    "available_source_records_denominator": len(
                        KLADOS_DEVELOPMENT_RECORDS
                    ),
                    "common_eligible_source_records": len(eligible_records),
                    "successful_source_records": len(selected),
                    "failed_within_common_eligible": len(eligible_records)
                    - len(selected),
                    "excluded_ineligible_source_records": len(
                        KLADOS_DEVELOPMENT_RECORDS
                    )
                    - len(eligible_records),
                    **{
                        f"median_{metric}": (
                            float(np.median(values)) if values else ""
                        )
                        for metric, values in metric_values.items()
                    },
                    **{
                        f"n_{metric}": len(values)
                        for metric, values in metric_values.items()
                    },
                    "median_latency_seconds": (
                        float(np.median(latency_values)) if latency_values else ""
                    ),
                    "n_latency_seconds": len(latency_values),
                    "median_peak_memory_mb": (
                        float(np.median(peak_memory_values))
                        if peak_memory_values
                        else ""
                    ),
                    "n_peak_memory_mb": len(peak_memory_values),
                    "median_function_evaluations_per_seed_per_window": (
                        float(np.median(function_evaluation_values))
                        if function_evaluation_values
                        else ""
                    ),
                    "n_function_evaluations_per_seed_per_window": len(
                        function_evaluation_values
                    ),
                    "median_total_function_evaluations_per_window": (
                        float(np.median(total_function_evaluation_values))
                        if total_function_evaluation_values
                        else ""
                    ),
                    "n_total_function_evaluations_per_window": len(
                        total_function_evaluation_values
                    ),
                    "model_parameters": model_parameters,
                    "optimizer_updates": optimizer_updates,
                    "training_history_steps": training_history_steps,
                    "optimizer_update_semantics": optimizer_update_semantics,
                    "training_walltime_seconds": training_walltime,
                    "training_cost_scope": training_cost_scope,
                    "training_walltime_status": training_walltime_status,
                    "algorithmic_seed_count": _median_from_candidates(
                        selected, "algorithmic_seed_count"
                    ),
                    "confirmatory": False,
                    "formal_G1_or_G3_evidence": False,
                }
            )
    if rows:
        _write_csv(root / "metrics.csv", rows)
    aggregate_failures = [
        row for row in rows if not row["status"].startswith("success")
    ]
    if aggregate_failures:
        _write_csv(root / "failures.csv", aggregate_failures)
    _write_csv(root / "coverage.csv", coverage)
    _write_csv(root / "method_summary.csv", method_summaries)
    _write_csv(root / "comparison_matrix.csv", comparison_matrix)
    _write_csv(root / "paired_vs_all_frozen_arms.csv", paired_rows)
    _write_csv(
        root / "common_eligible_arm_summary.csv",
        common_eligible_arm_summaries,
    )
    successful_pairs = [
        row for row in paired_rows if row["pair_status"] == "success_paired"
    ]
    unsuccessful_pairs = [
        row for row in paired_rows if row["pair_status"] != "success_paired"
    ]
    successful_unet_pairs = [
        row
        for row in successful_pairs
        if row["source_reference_method_id"]
        == "task_matched_multichannel_deterministic_UNet"
    ]
    missing_expected_cells = len(missing_conditional_keys)
    missing_result_summaries = sum(
        row["status"] == "unmatched_missing_conditional_result_summary"
        for row in coverage
    )
    completed_records_missing_metrics = sum(
        bool(row["metrics_file_missing"])
        and str(row["status"]).startswith("completed")
        for row in coverage
    )
    ineligible_records_with_unexpected_metrics = sum(
        bool(row["unexpected_metrics_file"]) for row in coverage
    )
    terminal_record_summaries = sum(
        str(row["status"]).startswith("completed")
        or row["status"] == "ineligible_common_record"
        for row in coverage
    )
    aggregate_complete = (
        observed_conditional_keys == expected_conditional_keys
        and missing_result_summaries == 0
        and completed_records_missing_metrics == 0
        and ineligible_records_with_unexpected_metrics == 0
        and terminal_record_summaries == len(KLADOS_DEVELOPMENT_RECORDS)
    )
    summary = {
        "status": (
            "completed_exploratory_development_no_family_decision"
            if aggregate_complete
            else "incomplete_exploratory_development_artifacts"
        ),
        "protocol_id": PROTOCOL_ID,
        "claim_scope": str(config["claim_scope"]),
        "available_source_records_denominator": len(KLADOS_DEVELOPMENT_RECORDS),
        "common_eligible_source_records": len(eligible_records),
        "common_eligible_source_record_ids": sorted(eligible_records),
        "records_with_any_successful_arm": len(successful_records),
        "fully_successful_common_eligible_source_records": len(
            fully_successful_records
        ),
        "ineligible_source_records": sum(
            row["status"] == "ineligible_common_record" for row in coverage
        ),
        "records_without_performance_metrics": sum(
            not bool(row["performance_metrics_emitted"]) for row in coverage
        ),
        "nonterminal_or_missing_result_summaries": (
            len(KLADOS_DEVELOPMENT_RECORDS) - terminal_record_summaries
        ),
        "failed_conditional_method_arms": sum(
            not row["status"].startswith("success") for row in rows
        ),
        "conditional_method_arm_failure_rate": (
            (len(aggregate_failures) + missing_expected_cells) / expected_rows
            if expected_rows
            else 0.0
        ),
        "observed_conditional_method_arm_failure_rate": (
            len(aggregate_failures) / len(rows) if rows else 0.0
        ),
        "expected_conditional_method_cells": expected_rows,
        "observed_conditional_method_cells": len(rows),
        "unmatched_missing_conditional_method_cells": len(
            missing_conditional_keys
        ),
        "unexpected_conditional_method_cells": len(
            unexpected_conditional_keys
        ),
        "conditional_record_scope_cartesian_product_exact": (
            observed_conditional_keys == expected_conditional_keys
        ),
        "conditional_training_endpoint_summaries_complete": (
            len(training_endpoint_summaries) == len(FROZEN_OPERATOR_SOURCES)
        ),
        "conditional_actual_updates_equal_attempts_6000_all_scopes": True,
        "conditional_zero_amp_skips_all_scopes": True,
        "conditional_training_resumed_false_all_scopes": True,
        "missing_conditional_result_summaries": missing_result_summaries,
        "completed_records_missing_metrics_files": completed_records_missing_metrics,
        "ineligible_records_with_unexpected_metrics_files": (
            ineligible_records_with_unexpected_metrics
        ),
        "records_are_participants": False,
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "diffusion_family_decision": "not_allowed_from_development_only_exploration",
        "training_objective_difference": (
            "conditional epsilon prediction versus deterministic paired task loss"
        ),
        "same_paired_supervision_exposure_for_conditional_and_UNet": True,
        "training_objectives_equal": False,
        "paired_comparison_cells_total": len(paired_rows),
        "paired_comparison_cells_successful": len(successful_pairs),
        "paired_comparison_cells_failed_ineligible_or_unmatched": len(
            unsuccessful_pairs
        ),
        "successful_conditional_vs_UNet_pairs": len(successful_unet_pairs),
        "fixed_optimizer_update_budget_equal_in_every_successful_UNet_pair": (
            bool(successful_unet_pairs)
            and all(
                int(row["conditional_actual_optimizer_updates"])
                == FIXED_OPTIMIZER_UPDATES
                and int(row["deterministic_fixed_checkpoint_updates"])
                == FIXED_OPTIMIZER_UPDATES
                and int(row["deterministic_actual_training_updates"])
                == FIXED_OPTIMIZER_UPDATES
                for row in successful_unet_pairs
            )
        ),
        "window_input_target_contract_equal": True,
        "cross_scope_operator_specificity_interpretation_allowed": False,
        "cross_scope_models_share_weights": False,
        "shared_prior_model_parameters": shared_prior_parameters,
        "shared_prior_training_history_steps": shared_prior_history_steps,
        "shared_prior_update_status": shared_prior_update_status,
        "metrics": str(root / "metrics.csv"),
        "failures": str(root / "failures.csv") if aggregate_failures else "",
        "coverage": str(root / "coverage.csv"),
        "method_summary": str(root / "method_summary.csv"),
        "comparison_matrix": str(root / "comparison_matrix.csv"),
        "paired_vs_all_frozen_arms": str(root / "paired_vs_all_frozen_arms.csv"),
        "common_eligible_arm_summary": str(
            root / "common_eligible_arm_summary.csv"
        ),
    }
    (root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "METHOD_ID",
    "PROTOCOL_ID",
    "ConditionalTrainingResult",
    "aggregate_conditional_development",
    "load_operator_conditioned_diffusion",
    "run_conditional_development_record",
    "train_operator_conditioned_diffusion",
    "validate_conditional_config",
]
