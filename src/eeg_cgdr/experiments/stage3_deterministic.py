"""Frozen Klados stage-3 deterministic-first comparison.

This is an exploratory source-record protocol.  It trains one independent
task-matched multichannel deterministic U-Net per operator scope on
sim01--sim30, selects/checks each checkpoint only on same-scope cells from
sim31--sim36/sim44/sim45, and may replay the already-used sixteen historical
records only with an explicit non-confirmatory label.  It does not implement a
broad A/B/C classifier and it cannot produce formal G1 or G3 evidence.
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
    KLADOS_NATIVE_CHANNEL_ORDER,
    KLADOS_TRAIN_RECORDS,
    KLADOS_UNTOUCHED_RECORDS,
    ChannelNormalizer,
    fit_channel_normalizer,
    prepare_mechanism_record,
    select_records,
)
from eeg_cgdr.experiments.mechanism_training import (
    _loader_config,
    _p0_config,
    load_population_projector,
    load_repaired_prior,
)
from eeg_cgdr.experiments.mechanism_runner import (
    _OperatorArm,
    _mechanism_metrics,
    _sample_one_seed,
)
from eeg_cgdr.inference import frame_attenuation_from_external_reference
from eeg_cgdr.models import (
    DeterministicUNetConfig,
    TaskMatchedDeterministicUNet,
)
from eeg_cgdr.operators import fit_p0
from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)


PROTOCOL_ID = "klados_stage3_deterministic_scope_isolated_v2"
FROZEN_METHODS = (
    "M1_observation_warm_start_sdedit",
    "M2_final_hard_q_consistency",
    "M4_per_step_quadratic_proximal_q_consistency",
    "deterministic_Qy",
    "deterministic_soft_proximal",
    "task_matched_multichannel_deterministic_UNet",
)
FROZEN_OPERATOR_SOURCES = (
    "population_projector",
    "matching_p0",
    "query_derived_oracle_projector",
)
FROZEN_STATUS = {
    "Klados": "current_M2_no_incremental_value",
    "diffusion_family": "not_tested",
    "SGE": "hard_Q_P0_tradeoff_inconclusive",
    "priority": "deterministic_first_diffusion_open",
}


@dataclass(frozen=True)
class DeterministicTrainingResult:
    status: str
    operator_scope: str
    deployable: bool
    checkpoint: Path
    best_checkpoint: Path
    steps_completed: int
    epochs_completed: int
    best_validation_loss: float
    resumed: bool


@dataclass(frozen=True)
class _WindowBundle:
    observed: np.ndarray
    clean: np.ndarray
    projector: np.ndarray
    attenuation: np.ndarray
    valid_time_weight: np.ndarray
    records: tuple[int, ...]
    operator_sources: tuple[str, ...]
    eligible_matching_records: int


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _operator_scope(value: str) -> str:
    scope = str(value)
    if scope not in FROZEN_OPERATOR_SOURCES:
        raise ValueError(f"unknown deterministic operator scope: {scope!r}")
    return scope


def _scope_deployable(operator_scope: str) -> bool:
    return _operator_scope(operator_scope) != "query_derived_oracle_projector"


def _scope_output_paths(
    config: Mapping[str, Any], operator_scope: str
) -> dict[str, Path]:
    scope = _operator_scope(operator_scope)
    root = Path(str(_mapping(config, "outputs")["root"]))
    scope_root = root / "training" / scope
    checkpoint_root = root / "checkpoints" / scope
    return {
        "scope_root": scope_root,
        "checkpoint": checkpoint_root / "last.pt",
        "best_checkpoint": checkpoint_root / "best.pt",
        "training_history": scope_root / "training_history.csv",
        "result_summary": scope_root / "result_summary.json",
    }


def _base_config(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(config.get("base_config", "")))
    if not path.is_file():
        raise FileNotFoundError(f"stage-3 base config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("stage-3 base config must be a mapping")
    return value


def validate_stage3_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID}")
    if int(config.get("harness_level", -1)) != 1:
        raise ValueError("stage-3 requires HARNESS_LEVEL=1")
    status = _mapping(config, "frozen_current_status")
    if dict(status) != FROZEN_STATUS:
        raise ValueError("stage-3 current-status boundary differs from the frozen audit")
    split = _mapping(config, "source_record_split")
    if tuple(int(value) for value in split.get("training", ())) != KLADOS_TRAIN_RECORDS:
        raise ValueError("stage-3 training records must be sim01-sim30")
    if tuple(int(value) for value in split.get("development", ())) != KLADOS_DEVELOPMENT_RECORDS:
        raise ValueError("stage-3 development records differ from frozen split")
    if tuple(
        int(value) for value in split.get("historical_evaluation_already_used", ())
    ) != KLADOS_UNTOUCHED_RECORDS:
        raise ValueError("stage-3 historical records differ from frozen split")
    if split.get("historical_records_are_fresh_evidence") is not False:
        raise ValueError("the old sixteen records cannot be fresh evidence")
    comparison = _mapping(config, "frozen_comparison")
    if tuple(comparison.get("methods", ())) != FROZEN_METHODS:
        raise ValueError("stage-3 method matrix is not frozen M1/M2/M4/Qy/soft/U-Net")
    if tuple(comparison.get("operator_sources", ())) != FROZEN_OPERATOR_SOURCES:
        raise ValueError("stage-3 operator-source matrix differs from protocol")
    if comparison.get("freeze_before_development_outcomes") is not True:
        raise ValueError("method and budget choices must be frozen before outcomes")
    if comparison.get("broad_classifier_enabled") is not False:
        raise ValueError("stage-3 must not add a broad classifier")
    seeds = tuple(int(value) for value in comparison.get("seeds", ()))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("stage-3 requires five distinct algorithmic seeds")
    if comparison.get("seeds_are_statistical_units") is not False:
        raise ValueError("stage-3 seeds cannot be treated as statistical units")
    visible = tuple(comparison.get("shared_visible_inputs", ()))
    if visible != (
        "observed_query_eeg",
        "operator_projector",
        "framewise_external_eog_attenuation",
        "valid_time_mask",
    ):
        raise ValueError("shared visible-input contract differs from the model API")
    training = _mapping(config, "deterministic_training")
    if int(training.get("minimum_updates", 0)) < 3000:
        raise ValueError("deterministic U-Net requires at least 3000 optimizer updates")
    if int(training.get("maximum_updates", 0)) < int(training["minimum_updates"]):
        raise ValueError("maximum_updates must cover minimum_updates")
    if tuple(training.get("operator_sources", ())) != FROZEN_OPERATOR_SOURCES:
        raise ValueError("training operator exposure must match the comparison sources")
    if training.get("operator_scope_isolated_checkpoints") is not True:
        raise ValueError("each deterministic operator scope requires an isolated checkpoint")
    if (
        training.get("checkpoint_selection_scope")
        != "same_operator_scope_development_cells_only"
    ):
        raise ValueError("checkpoint selection must remain inside one operator scope")
    if (
        training.get("oracle_conditioning_scope")
        != "training_and_development_source_records_only_nondeployable_upper_bound"
    ):
        raise ValueError("query-derived oracle conditioning must remain nondeployable")
    if training.get("historical_query_clean_target_used_for_training_or_selection") is not False:
        raise ValueError("historical evaluation clean targets cannot enter training or selection")
    integration = _mapping(config, "real_record_integration")
    if integration.get("source_record") != "sim31" or integration.get("complete_record") is not True:
        raise ValueError("real-record integration must use complete development record sim31")
    if tuple(integration.get("operator_sources", ())) != FROZEN_OPERATOR_SOURCES:
        raise ValueError("real-record integration must cover all frozen operator sources")
    if int(integration.get("minimum_minibatches", 0)) < 2:
        raise ValueError("real-record integration cannot be a single minibatch")
    budget = _mapping(comparison, "budget")
    iterative_budget_fields = (
        "M1_network_calls",
        "M2_network_calls",
        "M4_network_calls",
    )
    if any(int(budget.get(name, -1)) != 100 for name in iterative_budget_fields):
        raise ValueError("M1/M2/M4 must each use exactly 100 network calls")
    if int(budget.get("UNet_network_calls", -1)) != 1:
        raise ValueError("deterministic U-Net must use one network call")
    if budget.get("compute_is_equal") is not False or budget.get(
        "compute_difference_must_be_reported"
    ) is not True:
        raise ValueError("stage-3 must report its intentionally unequal compute budgets")
    base = _base_config(config)
    model_config = _model_config(config)
    channel_order = tuple(_mapping(base, "klados")["channel_order"])
    if model_config.eeg_channels != len(channel_order) or channel_order != KLADOS_NATIVE_CHANNEL_ORDER:
        raise ValueError("deterministic model must use the frozen Klados 19-channel montage")
    if model_config.signal_length != int(_mapping(base, "preprocessing")["window_samples"]):
        raise ValueError("deterministic model window must match repaired prior preprocessing")
    warm_start = int(_mapping(base, "sampling")["warm_start_timestep"])
    diffusion_steps = int(_mapping(base, "diffusion")["num_timesteps"])
    m1_calls = int(budget["M1_network_calls"])
    if not 0 < warm_start < diffusion_steps:
        raise ValueError("M1 warm-start timestep must lie inside the prior schedule")
    if m1_calls > warm_start + 1:
        raise ValueError(
            "M1 call budget exceeds the exact DDIM subsequence available from warm start"
        )
    if int(comparison.get("warm_start_timestep", -1)) != warm_start:
        raise ValueError("stage-3 and repaired-prior M1 warm-start timesteps differ")
    if float(comparison.get("quadratic_proximal_strength", float("nan"))) != float(
        _mapping(base, "sampling")["proximal_strength"]
    ):
        raise ValueError("stage-3 and repaired-prior M4 strengths differ")
    if (
        comparison.get("attenuation_source")
        != "same_support_standardized_framewise_external_EOG"
    ):
        raise ValueError("stage-3 attenuation source differs from the frozen shared input")
    if int(comparison.get("inference_batch_size", -1)) != int(
        _mapping(base, "sampling")["inference_batch_size"]
    ):
        raise ValueError("iterative and deterministic inference batch sizes must match")
    outputs = _mapping(config, "outputs")
    expected_root = Path(
        "/home/infres/yinwang/denoiseNet/results/cgdr/"
        "klados_stage3_deterministic_scope_isolated_v2"
    )
    if Path(str(outputs.get("root", ""))) != expected_root:
        raise ValueError("stage-3 v2 must use its new scope-isolated output root")
    if Path(str(outputs.get("development_root", ""))) != expected_root / "development":
        raise ValueError("stage-3 development output must stay under the v2 root")
    if Path(str(outputs.get("historical_root", ""))) != (
        expected_root / "historical_evaluation_already_used"
    ):
        raise ValueError("stage-3 historical output must stay under the v2 root")


def _model_config(config: Mapping[str, Any]) -> DeterministicUNetConfig:
    raw = dict(_mapping(config, "deterministic_model"))
    raw["channel_mults"] = tuple(int(value) for value in raw["channel_mults"])
    return DeterministicUNetConfig(**raw)


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


def _same_normalizer(left: ChannelNormalizer, right: ChannelNormalizer) -> bool:
    return (
        left.source_records == right.source_records
        and left.sample_count == right.sample_count
        and np.array_equal(left.mean, right.mean)
        and np.array_equal(left.standard_deviation, right.standard_deviation)
    )


def _checkpoint_contract(
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    operator_scope: str,
) -> dict[str, Any]:
    scope = _operator_scope(operator_scope)
    return {
        "protocol_id": PROTOCOL_ID,
        "operator_scope": scope,
        "operator_scope_deployable": _scope_deployable(scope),
        "training_bundle_operator_sources": [scope],
        "validation_bundle_operator_sources": [scope],
        "dataset": "klados_bamidis_v4",
        "training_source_records": list(KLADOS_TRAIN_RECORDS),
        "development_source_records": list(KLADOS_DEVELOPMENT_RECORDS),
        "historical_evaluation_already_used": list(KLADOS_UNTOUCHED_RECORDS),
        "channel_order": list(_mapping(base, "klados")["channel_order"]),
        "reference_id": str(_mapping(base, "klados")["reference_id"]),
        "preprocessing": dict(_mapping(base, "preprocessing")),
        "p0": dict(_mapping(base, "p0")),
        "observation": {
            key: _mapping(base, "observation")[key]
            for key in ("attenuation_source", "attenuation_floor", "attenuation_scale")
        },
        "model": asdict(_model_config(config)),
        "training": dict(_mapping(config, "deterministic_training")),
        "loss": dict(_mapping(config, "task_matched_loss")),
        "visible_inputs": list(
            _mapping(config, "frozen_comparison")["shared_visible_inputs"]
        ),
    }


def _oracle_projector(observed: np.ndarray, clean: np.ndarray, rank: int) -> np.ndarray:
    artifact = np.asarray(observed - clean, dtype=np.float64)
    basis, singular, _ = np.linalg.svd(artifact, full_matrices=False)
    if rank < 1 or rank > basis.shape[1] or singular[rank - 1] <= 0.0:
        raise ValueError("training clean target cannot define requested oracle projector")
    retained = basis[:, :rank]
    return retained @ retained.T


def _attenuation_windows(prepared: Any, base: Mapping[str, Any]) -> np.ndarray:
    eog = np.asarray(prepared.eog_windows, dtype=np.float64)
    magnitude = np.sqrt(np.mean(np.square(eog), axis=1))
    observation = _mapping(base, "observation")
    scale = float(observation["attenuation_scale"])
    floor = float(observation["attenuation_floor"])
    attenuation = np.sqrt(1.0 / (1.0 + np.square(magnitude / scale)))
    attenuation = np.clip(attenuation, floor, 1.0)
    return attenuation * np.asarray(prepared.valid_time_weight, dtype=np.float64)


def _window_bundle(
    config: Mapping[str, Any],
    base: dict[str, Any],
    *,
    records: Sequence[Any],
    normalizer: ChannelNormalizer,
    population_projector: np.ndarray,
    source_records: Sequence[int],
    operator_source: str,
    allow_matching_fallback: bool = False,
) -> _WindowBundle:
    operator_scope = _operator_scope(operator_source)
    values: dict[str, list[np.ndarray]] = {
        name: [] for name in ("observed", "clean", "projector", "attenuation", "mask")
    }
    record_labels: list[int] = []
    source_labels: list[str] = []
    matching_eligible = 0
    for native in select_records(records, source_records):
        prepared = prepare_mechanism_record(
            native,
            normalizer,
            source_rate=int(_mapping(base, "klados")["source_sampling_rate"]),
            target_rate=int(_mapping(base, "preprocessing")["target_sampling_rate"]),
            window_samples=int(_mapping(base, "preprocessing")["window_samples"]),
            calibration_seconds=float(_mapping(base, "klados")["calibration_seconds"]),
            guard_seconds=float(_mapping(base, "klados")["guard_seconds"]),
        )
        if operator_scope == "population_projector":
            projector = population_projector
        elif operator_scope == "matching_p0":
            outcome = fit_p0(
                prepared.calibration,
                _p0_config(base),
                movement_threshold=float(
                    _mapping(base, "p0")["movement_threshold"]
                ),
            )
            if outcome.transfer is None:
                # A rejected matching calibration is an effective population
                # cell, so it must not enter matching-scope fitting/selection.
                if not allow_matching_fallback:
                    continue
                projector = population_projector
            else:
                projector = np.asarray(outcome.transfer.projector, dtype=np.float64)
                matching_eligible += 1
        else:
            projector = _oracle_projector(
                prepared.observed_continuous,
                prepared.clean_continuous,
                int(_mapping(base, "p0")["target_rank"]),
            )
        attenuation = _attenuation_windows(prepared, base)
        windows = prepared.observed_windows.shape[0]
        values["observed"].append(prepared.observed_windows)
        values["clean"].append(prepared.clean_windows)
        values["attenuation"].append(attenuation)
        values["mask"].append(prepared.valid_time_weight)
        values["projector"].append(
            np.repeat(projector[None, :, :], windows, axis=0)
        )
        record_labels.extend([int(native.record_id)] * windows)
        source_labels.extend([operator_scope] * windows)
    if not source_labels:
        raise RuntimeError(
            f"operator scope {operator_scope!r} has no eligible training windows"
        )
    if set(source_labels) != {operator_scope}:
        raise AssertionError("deterministic window bundle crossed operator scopes")
    return _WindowBundle(
        observed=np.concatenate(values["observed"], axis=0),
        clean=np.concatenate(values["clean"], axis=0),
        projector=np.concatenate(values["projector"], axis=0),
        attenuation=np.concatenate(values["attenuation"], axis=0),
        valid_time_weight=np.concatenate(values["mask"], axis=0),
        records=tuple(record_labels),
        operator_sources=tuple(source_labels),
        eligible_matching_records=int(matching_eligible),
    )


def _merge_window_bundles(bundles: Sequence[_WindowBundle]) -> _WindowBundle:
    """Merge isolated bundles only for the ineligible engineering smoke."""

    values = tuple(bundles)
    if not values:
        raise ValueError("cannot merge an empty deterministic bundle sequence")
    if {source for bundle in values for source in bundle.operator_sources} != set(
        FROZEN_OPERATOR_SOURCES
    ):
        raise ValueError("engineering integration must cover all operator scopes")
    return _WindowBundle(
        observed=np.concatenate([bundle.observed for bundle in values], axis=0),
        clean=np.concatenate([bundle.clean for bundle in values], axis=0),
        projector=np.concatenate([bundle.projector for bundle in values], axis=0),
        attenuation=np.concatenate([bundle.attenuation for bundle in values], axis=0),
        valid_time_weight=np.concatenate(
            [bundle.valid_time_weight for bundle in values], axis=0
        ),
        records=tuple(record for bundle in values for record in bundle.records),
        operator_sources=tuple(
            source for bundle in values for source in bundle.operator_sources
        ),
        eligible_matching_records=sum(
            bundle.eligible_matching_records for bundle in values
        ),
    )


def _tensor_dataset(bundle: _WindowBundle) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(bundle.observed.astype(np.float32, copy=False)),
        torch.from_numpy(bundle.clean.astype(np.float32, copy=False)),
        torch.from_numpy(bundle.projector.astype(np.float32, copy=False)),
        torch.from_numpy(bundle.attenuation.astype(np.float32, copy=False)),
        torch.from_numpy(bundle.valid_time_weight.astype(np.float32, copy=False)),
    )


def _batch_loss(
    model: TaskMatchedDeterministicUNet,
    batch: Sequence[torch.Tensor],
    *,
    device: torch.device,
    loss_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    observed, clean, projector, attenuation, mask = (
        value.to(device, non_blocking=True) for value in batch
    )
    return model.task_loss(
        observed,
        clean,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
        parallel_weight=float(loss_config["parallel_weight"]),
        perpendicular_weight=float(loss_config["perpendicular_weight"]),
        derivative_weight=float(loss_config["derivative_weight"]),
    )


def _validation_loss(
    model: TaskMatchedDeterministicUNet,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
    loss_config: Mapping[str, Any],
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss, _ = _batch_loss(
                    model, batch, device=device, loss_config=loss_config
                )
            total += float(loss) * int(batch[0].shape[0])
            count += int(batch[0].shape[0])
    if count < 1 or not math.isfinite(total):
        raise FloatingPointError("deterministic validation loss is invalid")
    return total / count


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "step", "train_loss", "validation_loss", "best"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def train_task_matched_deterministic(
    config: dict[str, Any],
    *,
    operator_source: str,
    run_dir: Path,
    device: torch.device,
) -> DeterministicTrainingResult:
    """Train one operator-scope-isolated baseline for 3000--6000 updates."""

    validate_stage3_config(config)
    operator_scope = _operator_scope(operator_source)
    deployable = _scope_deployable(operator_scope)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("task-matched deterministic training requires scheduled CUDA")
    training_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    base = _base_config(config)
    if tuple(_mapping(base, "klados")["channel_order"]) != KLADOS_NATIVE_CHANNEL_ORDER:
        raise ValueError("Klados montage differs from frozen 19-channel order")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = load_klados_records(_loader_config(base))
    normalizer = fit_channel_normalizer(records, KLADOS_TRAIN_RECORDS)
    population = load_population_projector(base)
    population_value = np.asarray(population.projector, dtype=np.float64)
    train_bundle = _window_bundle(
        config,
        base,
        records=records,
        normalizer=normalizer,
        population_projector=population_value,
        source_records=KLADOS_TRAIN_RECORDS,
        operator_source=operator_scope,
    )
    validation_bundle = _window_bundle(
        config,
        base,
        records=records,
        normalizer=normalizer,
        population_projector=population_value,
        source_records=KLADOS_DEVELOPMENT_RECORDS,
        operator_source=operator_scope,
    )
    if set(train_bundle.operator_sources) != {operator_scope}:
        raise AssertionError("training bundle is not operator-scope isolated")
    if set(validation_bundle.operator_sources) != {operator_scope}:
        raise AssertionError("validation bundle is not operator-scope isolated")
    train_dataset = _tensor_dataset(train_bundle)
    validation_dataset = _tensor_dataset(validation_bundle)
    model = TaskMatchedDeterministicUNet(_model_config(config)).to(device)
    training = _mapping(config, "deterministic_training")
    loss_config = _mapping(config, "task_matched_loss")
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["mixed_precision"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    paths = _scope_output_paths(config, operator_scope)
    checkpoint = paths["checkpoint"]
    best_checkpoint = paths["best_checkpoint"]
    history_path = paths["training_history"]
    contract = _checkpoint_contract(config, base, operator_scope)
    normalizer_state = _normalizer_state(normalizer)
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    validations_without_improvement = 0
    resumed = False
    history: list[dict[str, Any]] = []
    if bool(training["resume"]) and checkpoint.is_file():
        state = resume_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=contract,
            map_location=device,
        )
        if state.normalizer_state != normalizer_state:
            raise ValueError("deterministic checkpoint normalizer differs from sim01-sim30")
        start_epoch = state.epoch + 1
        global_step = state.step
        best_loss = float(state.extra.get("best_validation_loss", float("inf")))
        validations_without_improvement = int(
            state.extra.get("validations_without_improvement", 0)
        )
        resumed = True
        if history_path.is_file():
            with history_path.open("r", encoding="utf-8", newline="") as stream:
                history = [
                    dict(row)
                    for row in csv.DictReader(stream)
                    if int(row["epoch"]) < start_epoch
                ]

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGUSR1, request_stop)
    minimum_updates = int(training["minimum_updates"])
    maximum_updates = int(training["maximum_updates"])
    validation_interval = int(training["validation_interval_updates"])
    next_validation = max(
        validation_interval,
        ((global_step // validation_interval) + 1) * validation_interval,
    )
    epoch = start_epoch
    last_validation = float("nan")
    try:
        while global_step < maximum_updates:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1000 + epoch)
            loader = DataLoader(
                train_dataset,
                batch_size=int(training["batch_size"]),
                shuffle=True,
                generator=generator,
                num_workers=int(training["workers"]),
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
                    loss, _ = _batch_loss(
                        model, batch, device=device, loss_config=loss_config
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite deterministic loss epoch={epoch} step={global_step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach())
                batches += 1
                global_step += 1
                if global_step >= maximum_updates:
                    break
            should_validate = (
                global_step >= next_validation
                or global_step >= maximum_updates
                or stop_requested
            )
            if should_validate:
                last_validation = _validation_loss(
                    model,
                    validation_dataset,
                    batch_size=int(training["batch_size"]),
                    device=device,
                    amp=amp,
                    loss_config=loss_config,
                )
                eligible_for_selection = global_step >= minimum_updates
                improved = eligible_for_selection and last_validation < best_loss - 1.0e-7
                if improved:
                    best_loss = last_validation
                    validations_without_improvement = 0
                elif eligible_for_selection:
                    validations_without_improvement += 1
                extra = {
                    "operator_scope": operator_scope,
                    "operator_scope_deployable": deployable,
                    "training_bundle_operator_sources": [operator_scope],
                    "validation_bundle_operator_sources": [operator_scope],
                    "best_validation_loss": best_loss,
                    "last_validation_loss": last_validation,
                    "validations_without_improvement": validations_without_improvement,
                    "minimum_updates_satisfied": global_step >= minimum_updates,
                    "training_windows": len(train_dataset),
                    "validation_windows": len(validation_dataset),
                    "training_matching_eligible_records": train_bundle.eligible_matching_records,
                    "validation_matching_eligible_records": validation_bundle.eligible_matching_records,
                }
                save_training_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    config=contract,
                    normalizer=normalizer_state,
                    extra=extra,
                )
                if improved:
                    save_training_checkpoint(
                        best_checkpoint,
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
                        "train_loss": total_loss / max(batches, 1),
                        "validation_loss": last_validation,
                        "best": improved,
                    }
                )
                _write_history(history_path, history)
                while next_validation <= global_step:
                    next_validation += validation_interval
            epoch += 1
            if stop_requested:
                break
            if (
                global_step >= minimum_updates
                and validations_without_improvement >= int(training["patience_validations"])
            ):
                break
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    status = "checkpointed_for_resume" if stop_requested else "completed"
    if status == "completed":
        if global_step < minimum_updates or not best_checkpoint.is_file():
            raise RuntimeError("deterministic training ended before the frozen minimum budget")
        if not math.isfinite(best_loss):
            raise RuntimeError("deterministic development selection produced no finite loss")
    summary = {
        "status": status,
        "protocol_id": PROTOCOL_ID,
        "operator_scope": operator_scope,
        "operator_scope_deployable": deployable,
        "training_bundle_operator_sources": [operator_scope],
        "validation_bundle_operator_sources": [operator_scope],
        "steps_completed": global_step,
        "minimum_updates": minimum_updates,
        "minimum_updates_satisfied": global_step >= minimum_updates,
        "epochs_completed": epoch,
        "best_validation_loss": best_loss,
        "last_validation_loss": last_validation,
        "resumed": resumed,
        "training_source_records": list(KLADOS_TRAIN_RECORDS),
        "development_source_records": list(KLADOS_DEVELOPMENT_RECORDS),
        "historical_records_used_for_training_or_selection": False,
        "operator_sources": [operator_scope],
        "visible_inputs": list(TaskMatchedDeterministicUNet.visible_input_fields),
        "checkpoint": str(checkpoint.resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()),
        "resume_command": str(
            _mapping(
                _mapping(config, "reproduction"),
                "resume_training_retry_commands",
            )[operator_scope]
        ),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "walltime_seconds": time.perf_counter() - training_started,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    paths["result_summary"].parent.mkdir(parents=True, exist_ok=True)
    paths["result_summary"].write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return DeterministicTrainingResult(
        status=status,
        operator_scope=operator_scope,
        deployable=deployable,
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
        steps_completed=global_step,
        epochs_completed=epoch,
        best_validation_loss=best_loss,
        resumed=resumed,
    )


def load_task_matched_deterministic(
    config: Mapping[str, Any],
    *,
    operator_source: str,
    device: torch.device,
) -> tuple[TaskMatchedDeterministicUNet, ChannelNormalizer, dict[str, Any]]:
    """Load one frozen >=3000-update operator-scope checkpoint."""

    validate_stage3_config(config)
    operator_scope = _operator_scope(operator_source)
    base = _base_config(config)
    path = _scope_output_paths(config, operator_scope)["best_checkpoint"]
    payload = load_training_checkpoint(path, map_location=device)
    normalizer = _validate_deterministic_checkpoint_payload(
        config,
        base,
        payload,
        operator_source=operator_scope,
    )
    model = TaskMatchedDeterministicUNet(_model_config(config)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, normalizer, payload


def _validate_deterministic_checkpoint_payload(
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    operator_source: str,
) -> ChannelNormalizer:
    """Reject under-budget or split-incompatible deterministic checkpoints."""

    operator_scope = _operator_scope(operator_source)
    if payload["config"] != _checkpoint_contract(config, base, operator_scope):
        raise ValueError("deterministic checkpoint contract differs from stage-3 config")
    extra = payload.get("extra", {})
    if extra.get("operator_scope") != operator_scope:
        raise ValueError("deterministic checkpoint extra state crossed operator scopes")
    if extra.get("operator_scope_deployable") is not _scope_deployable(
        operator_scope
    ):
        raise ValueError("deterministic checkpoint deployability label is inconsistent")
    if extra.get("training_bundle_operator_sources") != [operator_scope]:
        raise ValueError("deterministic training bundle crossed operator scopes")
    if extra.get("validation_bundle_operator_sources") != [operator_scope]:
        raise ValueError("deterministic validation selection crossed operator scopes")
    minimum_updates = int(_mapping(config, "deterministic_training")["minimum_updates"])
    if int(payload["step"]) < minimum_updates:
        raise ValueError("deterministic checkpoint predates the frozen minimum budget")
    normalizer = _normalizer_from_state(payload["normalizer_state"])
    if normalizer.source_records != KLADOS_TRAIN_RECORDS:
        raise ValueError("deterministic checkpoint normalization is not sim01-sim30")
    return normalizer


def run_stage3_real_record_integration(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Exercise the full baseline path on one complete development record.

    This is an engineering-only Slurm smoke.  It traverses every window and all
    three frozen operator sources for sim31, performs multiple optimizer
    updates, and proves a local checkpoint round trip.  Its checkpoint cannot
    be loaded as the >=3000-update comparison baseline.
    """

    validate_stage3_config(config)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stage-3 real-record integration requires scheduled CUDA")
    base = _base_config(config)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    records = load_klados_records(_loader_config(base))
    normalizer = fit_channel_normalizer(records, KLADOS_TRAIN_RECORDS)
    population = load_population_projector(base)
    bundle = _merge_window_bundles(
        tuple(
            _window_bundle(
                config,
                base,
                records=records,
                normalizer=normalizer,
                population_projector=np.asarray(
                    population.projector, dtype=np.float64
                ),
                source_records=(KLADOS_DEVELOPMENT_RECORDS[0],),
                operator_source=operator_scope,
                allow_matching_fallback=True,
            )
            for operator_scope in FROZEN_OPERATOR_SOURCES
        )
    )
    dataset = _tensor_dataset(bundle)
    integration = _mapping(config, "real_record_integration")
    batch_size = int(integration["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    if len(loader) < 2:
        raise ValueError("real-record integration must traverse multiple minibatches")

    model = TaskMatchedDeterministicUNet(_model_config(config)).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(_mapping(config, "deterministic_training")["learning_rate"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss_config = _mapping(config, "task_matched_loss")
    optimizer_updates = 0
    model.train()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _batch_loss(
            model,
            batch,
            device=device,
            loss_config=loss_config,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("real-record integration produced non-finite loss")
        loss.backward()
        optimizer.step()
        optimizer_updates += 1

    probe = tuple(value[:2].to(device) for value in dataset.tensors)
    model.eval()
    with torch.no_grad():
        before_reload = model(
            probe[0],
            projector=probe[2],
            attenuation=probe[3],
            valid_time_mask=probe[4],
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "engineering_only_real_record_checkpoint.pt"
    smoke_contract = {
        "protocol_id": PROTOCOL_ID,
        "stage": "real_record_integration_smoke_not_scientific_baseline",
        "source_record": "sim31",
        "operator_sources": list(FROZEN_OPERATOR_SOURCES),
        "scope_isolation_checkpoint_eligible": False,
    }
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=0,
        step=optimizer_updates,
        config=smoke_contract,
        normalizer=_normalizer_state(normalizer),
        extra={"scientific_result": False},
    )
    reloaded = TaskMatchedDeterministicUNet(_model_config(config)).to(device)
    reload_optimizer = AdamW(
        reloaded.parameters(),
        lr=float(_mapping(config, "deterministic_training")["learning_rate"]),
    )
    reload_scaler = torch.cuda.amp.GradScaler(enabled=False)
    resumed = resume_training_checkpoint(
        checkpoint,
        model=reloaded,
        optimizer=reload_optimizer,
        scaler=reload_scaler,
        expected_config=smoke_contract,
        map_location=device,
    )
    reloaded.eval()
    with torch.no_grad():
        after_reload = reloaded(
            probe[0],
            projector=probe[2],
            attenuation=probe[3],
            valid_time_mask=probe[4],
        )
    if not torch.equal(before_reload, after_reload):
        raise AssertionError("real-record integration checkpoint changed model output")
    torch.cuda.synchronize(device)
    summary = {
        "status": "passed_engineering_only_real_record_integration",
        "protocol_id": PROTOCOL_ID,
        "scientific_result": False,
        "formal_G1_or_G3_evidence": False,
        "source_record": "sim31",
        "complete_source_record_windows": len(dataset),
        "operator_sources": list(FROZEN_OPERATOR_SOURCES),
        "optimizer_updates": optimizer_updates,
        "multiple_minibatches": optimizer_updates >= 2,
        "checkpoint_roundtrip_equal": True,
        "checkpoint_step": resumed.step,
        "checkpoint": str(checkpoint.resolve()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "walltime_seconds": time.perf_counter() - started,
        "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
    }
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _matching_arm(prepared: Any, base: dict[str, Any]) -> Any:
    outcome = fit_p0(
        prepared.calibration,
        _p0_config(base),
        movement_threshold=float(_mapping(base, "p0")["movement_threshold"]),
    )
    return _OperatorArm(
        source="matching_p0",
        projector=(
            None
            if outcome.transfer is None
            else np.asarray(outcome.transfer.projector, dtype=np.float64)
        ),
        p0_outcome=outcome,
        calibration_id=f"sim{prepared.source_record:02d}_support",
    )


def _operator_arms(
    prepared: Any, population: Any, base: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    oracle = _oracle_projector(
        prepared.observed_continuous,
        prepared.clean_continuous,
        int(_mapping(base, "p0")["target_rank"]),
    )
    arms = {
        "population_projector": _OperatorArm(
            source="population_projector",
            projector=np.asarray(population.projector, dtype=np.float64),
            p0_outcome=None,
            calibration_id="sim01_sim30_population_projector",
        ),
        "matching_p0": _matching_arm(prepared, base),
        "query_derived_oracle_projector": _OperatorArm(
            source="query_derived_oracle_projector",
            projector=oracle,
            p0_outcome=None,
            calibration_id=f"sim{prepared.source_record:02d}_query_clean_upper_bound",
            query_clean_target_used=True,
        ),
    }
    return arms, oracle


def _continuous(windows: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(windows, dtype=np.float64)
    return value.transpose(1, 0, 2).reshape(value.shape[1], -1)[:, :length]


def _deterministic_restore(
    model: TaskMatchedDeterministicUNet,
    prepared: Any,
    projector: np.ndarray,
    base: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    eog_windows = np.asarray(prepared.eog_windows, dtype=np.float64)
    restored_parts: list[np.ndarray] = []
    calls = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, prepared.observed_windows.shape[0], batch_size):
            stop = min(start + batch_size, prepared.observed_windows.shape[0])
            observed = torch.as_tensor(
                prepared.observed_windows[start:stop],
                dtype=torch.float32,
                device=device,
            )
            eog = torch.as_tensor(
                eog_windows[start:stop], dtype=torch.float32, device=device
            )
            mask = torch.as_tensor(
                prepared.valid_time_weight[start:stop],
                dtype=torch.float32,
                device=device,
            )
            attenuation = frame_attenuation_from_external_reference(
                eog,
                scale=float(_mapping(base, "observation")["attenuation_scale"]),
                floor=float(_mapping(base, "observation")["attenuation_floor"]),
            ) * mask
            projection = torch.as_tensor(
                projector, dtype=torch.float32, device=device
            )
            restored_parts.append(
                model(
                    observed,
                    projector=projection,
                    attenuation=attenuation,
                    valid_time_mask=mask,
                ).cpu().numpy()
            )
            calls += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    else:
        peak_memory = 0.0
    restored = _continuous(
        np.concatenate(restored_parts, axis=0),
        prepared.observed_continuous.shape[1],
    )
    return restored, {
        "latency_seconds": time.perf_counter() - started,
        "peak_memory_mb": peak_memory,
        "function_evaluations": 1,
        "function_evaluations_per_seed_per_window": 1,
        "total_function_evaluations_per_window": 1,
        "network_forward_invocations": calls,
        "network_calls_total": calls,
    }


def _metric_row(
    *,
    partition: str,
    prepared: Any,
    method_id: str,
    operator_source: str,
    effective_operator_source: str,
    restored: np.ndarray,
    projector: np.ndarray,
    oracle: np.ndarray,
    artifact_mask: np.ndarray,
    runtime: Mapping[str, Any],
    fallback_used: bool,
    query_clean_target_used_by_method: bool,
    deterministic_checkpoint_operator_scope: str = "",
) -> dict[str, Any]:
    nondeployable_oracle = bool(query_clean_target_used_by_method)
    return {
        "partition": partition,
        "source_record": f"sim{prepared.source_record:02d}",
        "records_are_participants": False,
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "method_id": method_id,
        "operator_source": operator_source,
        "effective_operator_source": effective_operator_source,
        "fallback_used": fallback_used,
        "query_clean_target_used_by_method": nondeployable_oracle,
        "query_clean_target_used_for_scoring_only": not nondeployable_oracle,
        "deployable_operator_source": not nondeployable_oracle,
        "operator_role": (
            "nondeployable_query_clean_mechanism_upper_bound"
            if nondeployable_oracle
            else "deployable_external_eog_operator"
        ),
        "deterministic_checkpoint_operator_scope": (
            deterministic_checkpoint_operator_scope
        ),
        "status": "success",
        **_mechanism_metrics(
            restored,
            observed=prepared.observed_continuous,
            clean=prepared.clean_continuous,
            oracle_projector=oracle,
            estimated_projector=projector,
            artifact_mask=artifact_mask,
            sampling_rate=float(prepared.sampling_rate),
        ),
        **dict(runtime),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty stage-3 metrics")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_stage3_record(
    config: dict[str, Any],
    *,
    partition: str,
    task_index: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Run the frozen six-method matrix for one complete source record."""

    validate_stage3_config(config)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("stage-3 record comparison requires scheduled CUDA")
    record_started = time.perf_counter()
    if partition == "development":
        source_records = KLADOS_DEVELOPMENT_RECORDS
        stable_root = Path(str(_mapping(config, "outputs")["development_root"]))
    elif partition == "historical_evaluation_already_used_in_diagnosis":
        source_records = KLADOS_UNTOUCHED_RECORDS
        stable_root = Path(str(_mapping(config, "outputs")["historical_root"]))
    else:
        raise ValueError("stage-3 partition must be development or historical")
    if not 0 <= int(task_index) < len(source_records):
        raise ValueError("stage-3 array index lies outside the frozen partition")
    source_record = int(source_records[int(task_index)])
    base = _base_config(config)
    prior, _ = load_repaired_prior(base, device=device)
    deterministic_models: dict[str, TaskMatchedDeterministicUNet] = {}
    deterministic_checkpoints: dict[str, Mapping[str, Any]] = {}
    normalizer: ChannelNormalizer | None = None
    for operator_scope in FROZEN_OPERATOR_SOURCES:
        scope_model, scope_normalizer, scope_checkpoint = (
            load_task_matched_deterministic(
                config,
                operator_source=operator_scope,
                device=device,
            )
        )
        if normalizer is None:
            normalizer = scope_normalizer
        elif not _same_normalizer(normalizer, scope_normalizer):
            raise ValueError("operator-scope checkpoints use different normalizers")
        deterministic_models[operator_scope] = scope_model
        deterministic_checkpoints[operator_scope] = scope_checkpoint
    if normalizer is None:
        raise AssertionError("no deterministic operator-scope checkpoint was loaded")
    population = load_population_projector(base)
    records = load_klados_records(_loader_config(base))
    native = select_records(records, (source_record,))[0]
    prepared = prepare_mechanism_record(
        native,
        normalizer,
        source_rate=int(_mapping(base, "klados")["source_sampling_rate"]),
        target_rate=int(_mapping(base, "preprocessing")["target_sampling_rate"]),
        window_samples=int(_mapping(base, "preprocessing")["window_samples"]),
        calibration_seconds=float(_mapping(base, "klados")["calibration_seconds"]),
        guard_seconds=float(_mapping(base, "klados")["guard_seconds"]),
    )
    arms, oracle = _operator_arms(prepared, population, base)
    standardized_eog = np.asarray(prepared.eog_windows, dtype=np.float64)
    eog_magnitude = np.sqrt(np.mean(np.square(prepared.eog_continuous), axis=0))
    artifact_mask = eog_magnitude >= float(
        _mapping(base, "observation")["artifact_eog_z_threshold"]
    )
    comparison = _mapping(config, "frozen_comparison")
    budget = _mapping(comparison, "budget")
    seeds = tuple(int(value) for value in comparison["seeds"])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("stage-3 diffusion candidates require five fixed seeds")
    rows: list[dict[str, Any]] = []
    for requested_source in FROZEN_OPERATOR_SOURCES:
        requested_arm = arms[requested_source]
        fallback = not requested_arm.eligible
        effective_arm = arms["population_projector"] if fallback else requested_arm
        projector = np.asarray(effective_arm.projector, dtype=np.float64)
        effective_source = "population_projector" if fallback else requested_source
        deterministic = deterministic_models[effective_source]
        observed = np.asarray(prepared.observed_continuous, dtype=np.float64)
        algebra_started = time.perf_counter()
        projected = projector @ observed
        hard_qy = observed - projected
        hard_latency = time.perf_counter() - algebra_started
        soft_tau = float(comparison["soft_proximal_tau"])
        algebra_started = time.perf_counter()
        soft = hard_qy + soft_tau * projected
        soft_latency = time.perf_counter() - algebra_started
        deterministic_outputs = (
            (
                "deterministic_Qy",
                hard_qy,
                {
                    "function_evaluations": 0,
                    "function_evaluations_per_seed_per_window": 0,
                    "total_function_evaluations_per_window": 0,
                    "network_calls_total": 0,
                    "latency_seconds": hard_latency,
                    "peak_memory_mb": 0.0,
                },
            ),
            (
                "deterministic_soft_proximal",
                soft,
                {
                    "function_evaluations": 0,
                    "function_evaluations_per_seed_per_window": 0,
                    "total_function_evaluations_per_window": 0,
                    "network_calls_total": 0,
                    "latency_seconds": soft_latency,
                    "peak_memory_mb": 0.0,
                },
            ),
        )
        for method_id, restored, runtime in deterministic_outputs:
            rows.append(
                _metric_row(
                    partition=partition,
                    prepared=prepared,
                    method_id=method_id,
                    operator_source=requested_source,
                    effective_operator_source=effective_source,
                    restored=restored,
                    projector=projector,
                    oracle=oracle,
                    artifact_mask=artifact_mask,
                    runtime=runtime,
                    fallback_used=fallback,
                    query_clean_target_used_by_method=requested_arm.query_clean_target_used,
                )
            )
        unet_output, unet_runtime = _deterministic_restore(
            deterministic,
            prepared,
            projector,
            base,
            device=device,
            batch_size=int(comparison["inference_batch_size"]),
        )
        rows.append(
            _metric_row(
                partition=partition,
                prepared=prepared,
                method_id="task_matched_multichannel_deterministic_UNet",
                operator_source=requested_source,
                effective_operator_source=effective_source,
                restored=unet_output,
                projector=projector,
                oracle=oracle,
                artifact_mask=artifact_mask,
                runtime=unet_runtime,
                fallback_used=fallback,
                query_clean_target_used_by_method=requested_arm.query_clean_target_used,
                deterministic_checkpoint_operator_scope=effective_source,
            )
        )
        for candidate, method_id in (
            ("M1", "M1_observation_warm_start_sdedit"),
            ("M2", "M2_final_hard_q_consistency"),
            ("M4", "M4_per_step_quadratic_proximal_q_consistency"),
        ):
            restored_by_seed: list[np.ndarray] = []
            runtimes: list[Mapping[str, Any]] = []
            for seed in seeds:
                restored, runtime = _sample_one_seed(
                    prior=prior,
                    prepared=prepared,
                    standardized_eog_windows=standardized_eog,
                    population_projector=population,
                    arm=effective_arm,
                    candidate=candidate,
                    trust_radius=float(comparison["trust_radius"]),
                    seed=seed,
                    config=base,
                    device=device,
                    override_steps=int(budget[f"{candidate}_network_calls"]),
                )
                restored_by_seed.append(restored)
                runtimes.append(runtime)
            posterior_mean = np.mean(np.stack(restored_by_seed, axis=0), axis=0)
            runtime = {
                "function_evaluations": int(budget[f"{candidate}_network_calls"]),
                "function_evaluations_per_seed_per_window": int(
                    budget[f"{candidate}_network_calls"]
                ),
                "total_function_evaluations_per_window": (
                    len(seeds) * int(budget[f"{candidate}_network_calls"])
                ),
                "network_calls_total": sum(
                    int(value["network_calls_total"]) for value in runtimes
                ),
                "latency_seconds": sum(float(value["latency_seconds"]) for value in runtimes),
                "peak_memory_mb": max(float(value["peak_memory_mb"]) for value in runtimes),
                "algorithmic_seed_count": len(seeds),
                "seeds_are_statistical_units": False,
            }
            rows.append(
                _metric_row(
                    partition=partition,
                    prepared=prepared,
                    method_id=method_id,
                    operator_source=requested_source,
                    effective_operator_source=effective_source,
                    restored=posterior_mean,
                    projector=projector,
                    oracle=oracle,
                    artifact_mask=artifact_mask,
                    runtime=runtime,
                    fallback_used=fallback,
                    query_clean_target_used_by_method=requested_arm.query_clean_target_used,
                )
            )
    output_dir = stable_root / f"sim{source_record:02d}"
    _write_csv(output_dir / "metrics.csv", rows)
    summary = {
        "status": "completed_exploratory_source_record_comparison",
        "protocol_id": PROTOCOL_ID,
        "partition": partition,
        "source_record": f"sim{source_record:02d}",
        "records_are_participants": False,
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "historical_records_are_fresh_evidence": False,
        "query_derived_oracle_projector_role": (
            "nondeployable_query_clean_mechanism_upper_bound"
        ),
        "frozen_current_status": FROZEN_STATUS,
        "methods": list(FROZEN_METHODS),
        "operator_sources": list(FROZEN_OPERATOR_SOURCES),
        "deterministic_scope_checkpoints": {
            operator_scope: {
                "step": int(deterministic_checkpoints[operator_scope]["step"]),
                "deployable": _scope_deployable(operator_scope),
                "training_bundle_operator_sources": [operator_scope],
                "validation_bundle_operator_sources": [operator_scope],
                "best_checkpoint": str(
                    _scope_output_paths(config, operator_scope)["best_checkpoint"]
                ),
            }
            for operator_scope in FROZEN_OPERATOR_SOURCES
        },
        "deterministic_minimum_updates": int(
            _mapping(config, "deterministic_training")["minimum_updates"]
        ),
        "deterministic_model_parameters": sum(
            parameter.numel()
            for parameter in deterministic_models["population_projector"].parameters()
        ),
        "diffusion_prior_parameters": sum(parameter.numel() for parameter in prior.parameters()),
        "network_call_budget": dict(budget),
        "record_walltime_seconds": time.perf_counter() - record_started,
        "metrics": str(output_dir / "metrics.csv"),
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


def aggregate_stage3(
    config: Mapping[str, Any], *, partition: str, run_dir: Path
) -> dict[str, Any]:
    """Create descriptive source-record summaries without a broad classifier."""

    validate_stage3_config(config)
    if partition == "development":
        source_records = KLADOS_DEVELOPMENT_RECORDS
        root = Path(str(_mapping(config, "outputs")["development_root"]))
    elif partition == "historical_evaluation_already_used_in_diagnosis":
        source_records = KLADOS_UNTOUCHED_RECORDS
        root = Path(str(_mapping(config, "outputs")["historical_root"]))
    else:
        raise ValueError("unknown stage-3 aggregate partition")
    rows: list[dict[str, str]] = []
    for record in source_records:
        path = root / f"sim{record:02d}" / "metrics.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    expected = len(source_records) * len(FROZEN_OPERATOR_SOURCES) * len(FROZEN_METHODS)
    if len(rows) != expected:
        raise ValueError("stage-3 aggregate does not cover the frozen method matrix")
    expected_keys = {
        (f"sim{record:02d}", operator_source, method_id)
        for record in source_records
        for operator_source in FROZEN_OPERATOR_SOURCES
        for method_id in FROZEN_METHODS
    }
    actual_keys = {
        (row["source_record"], row["operator_source"], row["method_id"])
        for row in rows
    }
    if actual_keys != expected_keys or len(actual_keys) != len(rows):
        raise ValueError("stage-3 aggregate contains missing or duplicate matrix cells")
    for row in rows:
        oracle = row["operator_source"] == "query_derived_oracle_projector"
        if (row["query_clean_target_used_by_method"].lower() == "true") != oracle:
            raise ValueError("stage-3 query-oracle target-use label is inconsistent")
        if (row["deployable_operator_source"].lower() == "true") == oracle:
            raise ValueError("stage-3 query-oracle deployability label is inconsistent")
        checkpoint_scope = row.get("deterministic_checkpoint_operator_scope", "")
        if row["method_id"] == "task_matched_multichannel_deterministic_UNet":
            if checkpoint_scope != row["effective_operator_source"]:
                raise ValueError(
                    "deterministic U-Net checkpoint does not match effective operator scope"
                )
        elif checkpoint_scope:
            raise ValueError("non-U-Net method unexpectedly names a U-Net checkpoint scope")
    summaries: list[dict[str, Any]] = []
    for operator_source in FROZEN_OPERATOR_SOURCES:
        for method_id in FROZEN_METHODS:
            selected = [
                row
                for row in rows
                if row["operator_source"] == operator_source
                and row["method_id"] == method_id
            ]
            if len(selected) != len(source_records):
                raise AssertionError("stage-3 method summary lost source-record cells")
            summaries.append(
                {
                    "operator_source": operator_source,
                    "method_id": method_id,
                    "source_records": len(selected),
                    "deterministic_checkpoint_scopes": (
                        "|".join(
                            sorted(
                                {
                                    row["deterministic_checkpoint_operator_scope"]
                                    for row in selected
                                    if row["deterministic_checkpoint_operator_scope"]
                                }
                            )
                        )
                    ),
                    "fallback_rate": sum(
                        row["fallback_used"].lower() == "true" for row in selected
                    ) / len(selected),
                    **{
                        f"median_{metric}": float(
                            np.median([float(row[metric]) for row in selected])
                        )
                        for metric in (
                            "e_parallel",
                            "e_perp",
                            "rrmse",
                            "correlation",
                            "psd_distortion",
                        )
                    },
                }
            )
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "method_summary.csv", summaries)
    summary = {
        "status": "completed_descriptive_no_broad_classifier",
        "protocol_id": PROTOCOL_ID,
        "partition": partition,
        "source_records": [f"sim{value:02d}" for value in source_records],
        "records_are_participants": False,
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "historical_records_are_fresh_evidence": False,
        "frozen_current_status": FROZEN_STATUS,
        "broad_classifier_enabled": False,
        "operator_scope_isolation_verified": True,
        "metrics": str(root / "metrics.csv"),
        "method_summary": str(root / "method_summary.csv"),
    }
    (root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "FROZEN_METHODS",
    "FROZEN_OPERATOR_SOURCES",
    "FROZEN_STATUS",
    "PROTOCOL_ID",
    "aggregate_stage3",
    "load_task_matched_deterministic",
    "run_stage3_real_record_integration",
    "run_stage3_record",
    "train_task_matched_deterministic",
    "validate_stage3_config",
]
