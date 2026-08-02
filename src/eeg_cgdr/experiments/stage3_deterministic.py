"""Frozen Klados stage-3 deterministic-first comparison.

This is an exploratory source-record protocol.  It trains one independent
paired-supervised multichannel deterministic U-Net per operator scope on
sim01--sim30. Legacy v3 selected checkpoints on same-scope development cells;
active v4 uses one fixed 6000-update endpoint and treats those cells as
diagnostic only. The protocol may replay the already-used sixteen historical
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
from collections import Counter
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
    scaler_optimizer_step_succeeded,
)


LEGACY_PROTOCOL_ID = "klados_stage3_deterministic_scope_isolated_v2"
DEVELOPMENT_SELECTED_PROTOCOL_ID = "klados_stage3_deterministic_scope_isolated_v3"
PROTOCOL_ID = "klados_stage3_deterministic_scope_isolated_v4"
SUPPORTED_PROTOCOL_IDS = (
    LEGACY_PROTOCOL_ID,
    DEVELOPMENT_SELECTED_PROTOCOL_ID,
    PROTOCOL_ID,
)
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
    requested_record_ids: tuple[int, ...]
    included_record_ids: tuple[int, ...]
    skipped_record_ids: tuple[int, ...]


@dataclass(frozen=True)
class _CommonRecordEligibility:
    requested_record_ids: tuple[int, ...]
    included_record_ids: tuple[int, ...]
    skipped_record_ids: tuple[int, ...]
    skipped_reasons: Mapping[int, tuple[str, ...]]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _protocol_id(config: Mapping[str, Any]) -> str:
    protocol = str(config.get("protocol_id", ""))
    if protocol not in SUPPORTED_PROTOCOL_IDS:
        raise ValueError(
            "protocol_id must be one of " + ", ".join(SUPPORTED_PROTOCOL_IDS)
        )
    return protocol


def _uses_common_record_eligibility(config: Mapping[str, Any]) -> bool:
    return _protocol_id(config) in (DEVELOPMENT_SELECTED_PROTOCOL_ID, PROTOCOL_ID)


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
    protocol_id = _protocol_id(config)
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
    expected_selection_scope = (
        "fixed_same_operator_scope_6000_update_endpoint"
        if protocol_id == PROTOCOL_ID
        else "same_operator_scope_development_cells_only"
    )
    if training.get("checkpoint_selection_scope") != expected_selection_scope:
        raise ValueError("checkpoint selection scope differs from the protocol revision")
    if (
        training.get("oracle_conditioning_scope")
        != "training_and_development_source_records_only_nondeployable_upper_bound"
    ):
        raise ValueError("query-derived oracle conditioning must remain nondeployable")
    if training.get("historical_query_clean_target_used_for_training_or_selection") is not False:
        raise ValueError("historical evaluation clean targets cannot enter training or selection")
    if protocol_id == PROTOCOL_ID:
        loss_contract = _mapping(config, "task_matched_loss")
        if (
            int(training.get("minimum_updates", -1)) != 6000
            or int(training.get("maximum_updates", -1)) != 6000
            or training.get("checkpoint_selection")
            != "fixed_6000_update_endpoint_no_development_selection"
            or training.get("development_loss_role")
            != "diagnostic_only_not_checkpoint_or_update_selection"
        ):
            raise ValueError(
                "v4 requires one fixed 6000-update endpoint with diagnostic-only development loss"
            )
        if loss_contract.get("selection_target") != (
            "diagnostic_paired_clean_EEG_not_used_for_checkpoint_selection"
        ):
            raise ValueError("v4 development target must be diagnostic-only")
    if _uses_common_record_eligibility(config):
        eligibility = _mapping(config, "common_record_eligibility")
        if (
            eligibility.get("rule")
            != "matching_p0_eligible_records_shared_by_all_operator_scopes"
        ):
            raise ValueError("v3/v4 requires one common matching-P0 eligibility set")
        if eligibility.get("exclude_ineligible_from_all_scope_training") is not True:
            raise ValueError("v3/v4 must exclude matching-ineligible records from every scope")
        if eligibility.get("report_included_and_skipped_records") is not True:
            raise ValueError("v3/v4 must report common included/skipped records")
        comparator = _mapping(config, "deterministic_comparator_scope")
        if comparator.get("supervision") != "paired_supervised_clean_target":
            raise ValueError("v3/v4 U-Net must disclose paired clean-target supervision")
        if comparator.get("same_supervision_as_clean_prior") is not False:
            raise ValueError("v3/v4 U-Net must not be labelled same-supervision")
        if comparator.get("formal_G3_evidence") is not False:
            raise ValueError("v3/v4 exploratory U-Net cannot be formal G3 evidence")
    integration = _mapping(config, "real_record_integration")
    if (
        integration.get("source_record") != "sim31"
        or integration.get("complete_record") is not True
    ):
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
    if (
        model_config.eeg_channels != len(channel_order)
        or channel_order != KLADOS_NATIVE_CHANNEL_ORDER
    ):
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
    expected_root = Path("/home/infres/yinwang/denoiseNet/results/cgdr") / protocol_id
    if Path(str(outputs.get("root", ""))) != expected_root:
        raise ValueError("stage-3 must use the output root matching its protocol revision")
    if Path(str(outputs.get("development_root", ""))) != expected_root / "development":
        raise ValueError("stage-3 development output must stay under its revision root")
    if Path(str(outputs.get("historical_root", ""))) != (
        expected_root / "historical_evaluation_already_used"
    ):
        raise ValueError("stage-3 historical output must stay under its revision root")


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
    contract = {
        "protocol_id": _protocol_id(config),
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
    if _uses_common_record_eligibility(config):
        contract["common_record_eligibility"] = dict(
            _mapping(config, "common_record_eligibility")
        )
        contract["deterministic_comparator_scope"] = dict(
            _mapping(config, "deterministic_comparator_scope")
        )
    return contract


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


def _common_matching_eligibility(
    base: Mapping[str, Any],
    *,
    records: Sequence[Any],
    normalizer: ChannelNormalizer,
    source_records: Sequence[int],
) -> _CommonRecordEligibility:
    """Freeze one matching-P0 record set shared by all three U-Net scopes."""

    requested = tuple(int(value) for value in source_records)
    included: list[int] = []
    skipped: list[int] = []
    reasons: dict[int, tuple[str, ...]] = {}
    selected = select_records(records, requested)
    if tuple(int(native.record_id) for native in selected) != requested:
        raise ValueError("source-record loader order differs from the frozen split")
    for native in selected:
        prepared = prepare_mechanism_record(
            native,
            normalizer,
            source_rate=int(_mapping(base, "klados")["source_sampling_rate"]),
            target_rate=int(_mapping(base, "preprocessing")["target_sampling_rate"]),
            window_samples=int(_mapping(base, "preprocessing")["window_samples"]),
            calibration_seconds=float(_mapping(base, "klados")["calibration_seconds"]),
            guard_seconds=float(_mapping(base, "klados")["guard_seconds"]),
        )
        outcome = fit_p0(
            prepared.calibration,
            _p0_config(base),
            movement_threshold=float(_mapping(base, "p0")["movement_threshold"]),
        )
        record_id = int(native.record_id)
        if outcome.transfer is None:
            skipped.append(record_id)
            reasons[record_id] = tuple(str(value) for value in outcome.reasons)
        else:
            included.append(record_id)
    if not included:
        raise RuntimeError("common matching-P0 eligibility rejected every source record")
    return _CommonRecordEligibility(
        requested_record_ids=requested,
        included_record_ids=tuple(included),
        skipped_record_ids=tuple(skipped),
        skipped_reasons=reasons,
    )


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
    common_eligible_source_records: Sequence[int] | None = None,
) -> _WindowBundle:
    operator_scope = _operator_scope(operator_source)
    values: dict[str, list[np.ndarray]] = {
        name: [] for name in ("observed", "clean", "projector", "attenuation", "mask")
    }
    record_labels: list[int] = []
    source_labels: list[str] = []
    matching_eligible = 0
    requested_record_ids = tuple(int(value) for value in source_records)
    common_eligible = (
        None
        if common_eligible_source_records is None
        else frozenset(int(value) for value in common_eligible_source_records)
    )
    if _uses_common_record_eligibility(config) and common_eligible is None:
        raise ValueError("v3 window bundles require the shared eligibility set")
    included_record_ids: list[int] = []
    skipped_record_ids: list[int] = []
    for native in select_records(records, requested_record_ids):
        record_id = int(native.record_id)
        if common_eligible is not None and record_id not in common_eligible:
            skipped_record_ids.append(record_id)
            continue
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
                if common_eligible is not None:
                    raise RuntimeError(
                        "matching-P0 eligibility changed while constructing a shared bundle"
                    )
                # A rejected matching calibration is an effective population
                # cell, so it must not enter matching-scope fitting/selection.
                if not allow_matching_fallback:
                    skipped_record_ids.append(record_id)
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
        included_record_ids.append(record_id)
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
        requested_record_ids=requested_record_ids,
        included_record_ids=tuple(included_record_ids),
        skipped_record_ids=tuple(skipped_record_ids),
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
        requested_record_ids=tuple(
            record for bundle in values for record in bundle.requested_record_ids
        ),
        included_record_ids=tuple(
            record for bundle in values for record in bundle.included_record_ids
        ),
        skipped_record_ids=tuple(
            record for bundle in values for record in bundle.skipped_record_ids
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


def _training_terminal_reason(
    *,
    global_step: int,
    maximum_updates: int,
    minimum_updates: int,
    validations_without_improvement: int,
    patience_validations: int,
) -> str:
    """Return a persistent terminal reason, or an empty string while resumable."""

    if int(global_step) >= int(maximum_updates):
        return "maximum_updates_reached"
    if (
        int(global_step) >= int(minimum_updates)
        and int(validations_without_improvement) >= int(patience_validations)
    ):
        return "development_early_stop_patience_reached"
    return ""


def _resume_terminal_reason(
    extra: Mapping[str, Any],
    *,
    global_step: int,
    maximum_updates: int,
    minimum_updates: int,
    patience_validations: int,
) -> str:
    saved = str(extra.get("terminal_reason", ""))
    if bool(extra.get("training_terminal", False)) and not saved:
        raise ValueError("terminal checkpoint is missing its terminal reason")
    if saved:
        return saved
    return _training_terminal_reason(
        global_step=global_step,
        maximum_updates=maximum_updates,
        minimum_updates=minimum_updates,
        validations_without_improvement=int(
            extra.get("validations_without_improvement", 0)
        ),
        patience_validations=patience_validations,
    )


def _bundle_record_summary(bundle: _WindowBundle) -> dict[str, Any]:
    counts = Counter(bundle.records)
    return {
        "requested_record_count": len(bundle.requested_record_ids),
        "included_record_count": len(bundle.included_record_ids),
        "skipped_record_count": len(bundle.skipped_record_ids),
        "requested_record_ids": list(bundle.requested_record_ids),
        "included_record_ids": list(bundle.included_record_ids),
        "skipped_record_ids": list(bundle.skipped_record_ids),
        "windows_per_included_record": {
            f"sim{record_id:02d}": int(counts[record_id])
            for record_id in bundle.included_record_ids
        },
    }


def train_task_matched_deterministic(
    config: dict[str, Any],
    *,
    operator_source: str,
    run_dir: Path,
    device: torch.device,
) -> DeterministicTrainingResult:
    """Train one isolated baseline under the selected protocol revision."""

    validate_stage3_config(config)
    operator_scope = _operator_scope(operator_source)
    deployable = _scope_deployable(operator_scope)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("task-matched deterministic training requires scheduled CUDA")
    training_started = time.perf_counter()
    fixed_endpoint_protocol = _protocol_id(config) == PROTOCOL_ID
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
    train_eligibility = (
        _common_matching_eligibility(
            base,
            records=records,
            normalizer=normalizer,
            source_records=KLADOS_TRAIN_RECORDS,
        )
        if _uses_common_record_eligibility(config)
        else None
    )
    validation_eligibility = (
        _common_matching_eligibility(
            base,
            records=records,
            normalizer=normalizer,
            source_records=KLADOS_DEVELOPMENT_RECORDS,
        )
        if _uses_common_record_eligibility(config)
        else None
    )
    train_bundle = _window_bundle(
        config,
        base,
        records=records,
        normalizer=normalizer,
        population_projector=population_value,
        source_records=KLADOS_TRAIN_RECORDS,
        operator_source=operator_scope,
        common_eligible_source_records=(
            None
            if train_eligibility is None
            else train_eligibility.included_record_ids
        ),
    )
    validation_bundle = _window_bundle(
        config,
        base,
        records=records,
        normalizer=normalizer,
        population_projector=population_value,
        source_records=KLADOS_DEVELOPMENT_RECORDS,
        operator_source=operator_scope,
        common_eligible_source_records=(
            None
            if validation_eligibility is None
            else validation_eligibility.included_record_ids
        ),
    )
    if set(train_bundle.operator_sources) != {operator_scope}:
        raise AssertionError("training bundle is not operator-scope isolated")
    if set(validation_bundle.operator_sources) != {operator_scope}:
        raise AssertionError("validation bundle is not operator-scope isolated")
    if _uses_common_record_eligibility(config):
        if train_eligibility is None or validation_eligibility is None:
            raise AssertionError("v3 common record eligibility was not constructed")
        if train_bundle.included_record_ids != train_eligibility.included_record_ids:
            raise AssertionError("training bundle differs from common eligible records")
        if validation_bundle.included_record_ids != validation_eligibility.included_record_ids:
            raise AssertionError("validation bundle differs from common eligible records")
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
    optimizer_step_attempts = 0
    skipped_optimizer_steps = 0
    best_loss = float("inf")
    restored_last_validation = float("nan")
    validations_without_improvement = 0
    resumed = False
    resumed_terminal = False
    terminal_reason = ""
    prior_walltime_seconds = 0.0
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
        optimizer_step_attempts = int(
            state.extra.get("optimizer_step_attempts", global_step)
        )
        skipped_optimizer_steps = int(
            state.extra.get("skipped_optimizer_steps_amp_overflow", 0)
        )
        best_loss = float(state.extra.get("best_validation_loss", float("inf")))
        restored_last_validation = float(
            state.extra.get("last_validation_loss", float("nan"))
        )
        validations_without_improvement = int(
            state.extra.get("validations_without_improvement", 0)
        )
        prior_walltime_seconds = float(
            state.extra.get("cumulative_training_walltime_seconds", 0.0)
        )
        if fixed_endpoint_protocol:
            if global_step > int(training["maximum_updates"]):
                raise ValueError("fixed-endpoint checkpoint exceeds 6000 updates")
            terminal_reason = (
                "maximum_updates_reached"
                if global_step == int(training["maximum_updates"])
                else ""
            )
            if state.extra.get("checkpoint_selection_used_development_loss") is True:
                raise ValueError("v4 checkpoint used development loss for selection")
            if global_step == int(training["maximum_updates"]) and not (
                best_checkpoint.is_file()
            ):
                save_training_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=state.epoch,
                    step=global_step,
                    config=contract,
                    normalizer=normalizer_state,
                    extra=state.extra,
                )
        else:
            terminal_reason = _resume_terminal_reason(
                state.extra,
                global_step=global_step,
                maximum_updates=int(training["maximum_updates"]),
                minimum_updates=int(training["minimum_updates"]),
                patience_validations=int(training["patience_validations"]),
            )
        resumed_terminal = bool(state.extra.get("training_terminal", False)) or bool(
            terminal_reason
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
    last_validation = restored_last_validation
    try:
        while global_step < maximum_updates and not resumed_terminal:
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
                diagnostic_improved = last_validation < best_loss - 1.0e-7
                legacy_selection_improved = (
                    eligible_for_selection and diagnostic_improved
                )
                if (
                    fixed_endpoint_protocol and diagnostic_improved
                ) or legacy_selection_improved:
                    best_loss = last_validation
                    validations_without_improvement = 0
                elif eligible_for_selection:
                    validations_without_improvement += 1
                selected_checkpoint = (
                    global_step >= maximum_updates
                    if fixed_endpoint_protocol
                    else legacy_selection_improved
                )
                terminal_reason = (
                    "maximum_updates_reached"
                    if fixed_endpoint_protocol and global_step >= maximum_updates
                    else _training_terminal_reason(
                        global_step=global_step,
                        maximum_updates=maximum_updates,
                        minimum_updates=minimum_updates,
                        validations_without_improvement=validations_without_improvement,
                        patience_validations=int(training["patience_validations"]),
                    )
                )
                training_terminal = bool(terminal_reason) and not stop_requested
                elapsed_walltime = time.perf_counter() - training_started
                extra = {
                    "operator_scope": operator_scope,
                    "operator_scope_deployable": deployable,
                    "training_bundle_operator_sources": [operator_scope],
                    "validation_bundle_operator_sources": [operator_scope],
                    "best_validation_loss": best_loss,
                    "last_validation_loss": last_validation,
                    "validations_without_improvement": validations_without_improvement,
                    "minimum_updates_satisfied": global_step >= minimum_updates,
                    "fixed_endpoint_update": (
                        maximum_updates if fixed_endpoint_protocol else None
                    ),
                    "checkpoint_selection_used_development_loss": (
                        not fixed_endpoint_protocol
                    ),
                    "development_loss_role": (
                        "diagnostic_only_not_checkpoint_or_update_selection"
                        if fixed_endpoint_protocol
                        else "checkpoint_selection"
                    ),
                    "training_windows": len(train_dataset),
                    "validation_windows": len(validation_dataset),
                    "training_matching_eligible_records": train_bundle.eligible_matching_records,
                    "validation_matching_eligible_records": (
                        validation_bundle.eligible_matching_records
                    ),
                    "training_record_coverage": _bundle_record_summary(train_bundle),
                    "validation_record_coverage": _bundle_record_summary(validation_bundle),
                    "training_common_eligibility_skipped_reasons": (
                        {}
                        if train_eligibility is None
                        else {
                            f"sim{key:02d}": list(value)
                            for key, value in train_eligibility.skipped_reasons.items()
                        }
                    ),
                    "validation_common_eligibility_skipped_reasons": (
                        {}
                        if validation_eligibility is None
                        else {
                            f"sim{key:02d}": list(value)
                            for key, value in validation_eligibility.skipped_reasons.items()
                        }
                    ),
                    "training_terminal": training_terminal,
                    "optimizer_step_attempts": optimizer_step_attempts,
                    "successful_optimizer_updates": global_step,
                    "skipped_optimizer_steps_amp_overflow": skipped_optimizer_steps,
                    "terminal_reason": terminal_reason if training_terminal else "",
                    "cumulative_training_walltime_seconds": (
                        prior_walltime_seconds + elapsed_walltime
                    ),
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
                if selected_checkpoint:
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
                        "best": selected_checkpoint,
                    }
                )
                _write_history(history_path, history)
                while next_validation <= global_step:
                    next_validation += validation_interval
            epoch += 1
            if stop_requested:
                break
            if terminal_reason:
                break
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    status = (
        "checkpointed_for_resume"
        if stop_requested and not terminal_reason
        else "completed_terminal_resume"
        if resumed_terminal
        else "completed"
    )
    if status.startswith("completed"):
        if global_step < minimum_updates or not best_checkpoint.is_file():
            raise RuntimeError("deterministic training ended before the frozen minimum budget")
        if not math.isfinite(best_loss):
            raise RuntimeError("deterministic development diagnostic produced no finite loss")
    summary = {
        "status": status,
        "protocol_id": _protocol_id(config),
        "operator_scope": operator_scope,
        "operator_scope_deployable": deployable,
        "training_bundle_operator_sources": [operator_scope],
        "validation_bundle_operator_sources": [operator_scope],
        "steps_completed": global_step,
        "optimizer_step_attempts": optimizer_step_attempts,
        "successful_optimizer_updates": global_step,
        "skipped_optimizer_steps_amp_overflow": skipped_optimizer_steps,
        "minimum_updates": minimum_updates,
        "minimum_updates_satisfied": global_step >= minimum_updates,
        "epochs_completed": epoch,
        "best_validation_loss": best_loss,
        "last_validation_loss": last_validation,
        "fixed_endpoint_update": maximum_updates if fixed_endpoint_protocol else None,
        "checkpoint_selection_used_development_loss": not fixed_endpoint_protocol,
        "development_loss_role": (
            "diagnostic_only_not_checkpoint_or_update_selection"
            if fixed_endpoint_protocol
            else "checkpoint_selection"
        ),
        "resumed": resumed,
        "resumed_terminal_checkpoint_without_updates": resumed_terminal,
        "terminal_reason": terminal_reason,
        "training_source_records": list(KLADOS_TRAIN_RECORDS),
        "development_source_records": list(KLADOS_DEVELOPMENT_RECORDS),
        "historical_records_used_for_training_or_selection": False,
        "operator_sources": [operator_scope],
        "training_record_coverage": _bundle_record_summary(train_bundle),
        "validation_record_coverage": _bundle_record_summary(validation_bundle),
        "training_common_eligibility_skipped_reasons": (
            {}
            if train_eligibility is None
            else {
                f"sim{key:02d}": list(value)
                for key, value in train_eligibility.skipped_reasons.items()
            }
        ),
        "validation_common_eligibility_skipped_reasons": (
            {}
            if validation_eligibility is None
            else {
                f"sim{key:02d}": list(value)
                for key, value in validation_eligibility.skipped_reasons.items()
            }
        ),
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
        "cumulative_training_walltime_seconds": (
            prior_walltime_seconds + time.perf_counter() - training_started
        ),
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
    if _uses_common_record_eligibility(config):
        for name in ("training_record_coverage", "validation_record_coverage"):
            coverage = extra.get(name)
            if not isinstance(coverage, Mapping):
                raise ValueError(f"deterministic checkpoint lacks {name}")
            included = tuple(int(value) for value in coverage.get("included_record_ids", ()))
            skipped = tuple(int(value) for value in coverage.get("skipped_record_ids", ()))
            requested = tuple(int(value) for value in coverage.get("requested_record_ids", ()))
            if (
                not included
                or set(included) & set(skipped)
                or set(included) | set(skipped) != set(requested)
            ):
                raise ValueError(f"deterministic checkpoint has invalid {name}")
    minimum_updates = int(_mapping(config, "deterministic_training")["minimum_updates"])
    if int(payload["step"]) < minimum_updates:
        raise ValueError("deterministic checkpoint predates the frozen minimum budget")
    if _protocol_id(config) == PROTOCOL_ID:
        if int(payload["step"]) != 6000:
            raise ValueError("v4 deterministic checkpoint is not the fixed 6000-step endpoint")
        if extra.get("checkpoint_selection_used_development_loss") is not False:
            raise ValueError("v4 deterministic checkpoint used development loss for selection")
        if extra.get("fixed_endpoint_update") != 6000:
            raise ValueError("v4 deterministic checkpoint lacks the fixed endpoint contract")
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
    integration_source_records = (KLADOS_DEVELOPMENT_RECORDS[0],)
    integration_eligibility = (
        _common_matching_eligibility(
            base,
            records=records,
            normalizer=normalizer,
            source_records=integration_source_records,
        )
        if _uses_common_record_eligibility(config)
        else None
    )
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
                source_records=integration_source_records,
                operator_source=operator_scope,
                allow_matching_fallback=True,
                common_eligible_source_records=(
                    None
                    if integration_eligibility is None
                    else integration_eligibility.included_record_ids
                ),
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
        "protocol_id": _protocol_id(config),
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
        "protocol_id": _protocol_id(config),
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
    status: str = "success",
    comparator_supervision: str | None = None,
) -> dict[str, Any]:
    nondeployable_oracle = bool(query_clean_target_used_by_method)
    restored_value = np.asarray(restored, dtype=np.float64)
    if not np.isfinite(restored_value).all():
        raise FloatingPointError("restored method output contains non-finite values")
    metrics = _mechanism_metrics(
        restored_value,
        observed=prepared.observed_continuous,
        clean=prepared.clean_continuous,
        oracle_projector=oracle,
        estimated_projector=projector,
        artifact_mask=artifact_mask,
        sampling_rate=float(prepared.sampling_rate),
    )
    invalid_metrics = sorted(
        key
        for key, value in metrics.items()
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
    )
    if invalid_metrics:
        raise FloatingPointError(
            f"method metrics contain non-finite values: {invalid_metrics}"
        )
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
        "comparator_supervision": (
            comparator_supervision
            if comparator_supervision is not None
            else "paired_supervised_clean_target_stronger_differently_supervised_exploratory"
            if method_id == "task_matched_multichannel_deterministic_UNet"
            else "clean_prior_sampling"
            if method_id.startswith("M")
            else "no_learned_training"
        ),
        "same_supervision_G3_comparison": False,
        "status": status,
        **metrics,
        **dict(runtime),
    }


def _failed_metric_row(
    *,
    partition: str,
    prepared: Any,
    method_id: str,
    operator_source: str,
    effective_operator_source: str,
    fallback_used: bool,
    query_clean_target_used_by_method: bool,
    status: str,
    failure_type: str,
    failure_message: str,
    deterministic_checkpoint_operator_scope: str = "",
    runtime: Mapping[str, Any] | None = None,
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
        "comparator_supervision": (
            "paired_supervised_clean_target_stronger_differently_supervised_exploratory"
            if method_id == "task_matched_multichannel_deterministic_UNet"
            else "clean_prior_sampling"
            if method_id.startswith("M")
            else "no_learned_training"
        ),
        "same_supervision_G3_comparison": False,
        "status": status,
        "failure_type": failure_type,
        "failure_message": failure_message,
        **dict(runtime or {}),
    }


def _retainable_method_failure(error: Exception) -> bool:
    """Keep numerical method failures, but never hide systemic setup faults."""

    if isinstance(error, (FloatingPointError, np.linalg.LinAlgError)):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    systemic_tokens = (
        "cuda out of memory",
        "out of memory",
        "device-side",
        "no kernel image",
        "checkpoint",
        "size mismatch",
        "expected all tensors",
        "no such file",
    )
    if any(token in message for token in systemic_tokens):
        return False
    numerical_tokens = ("nan", "inf", "non-finite", "numerical", "singular")
    return any(token in message for token in numerical_tokens)


def _safe_metric_row(
    *, failure_rows: list[dict[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    try:
        return _metric_row(**kwargs)
    except Exception as error:
        if not _retainable_method_failure(error):
            raise
        prepared = kwargs["prepared"]
        failure_rows.append(
            {
                "partition": kwargs["partition"],
                "source_record": f"sim{prepared.source_record:02d}",
                "operator_source": kwargs["operator_source"],
                "effective_operator_source": kwargs["effective_operator_source"],
                "method_id": kwargs["method_id"],
                "seed": "",
                "status": "failed_metric_numerical",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        return _failed_metric_row(
            partition=kwargs["partition"],
            prepared=prepared,
            method_id=kwargs["method_id"],
            operator_source=kwargs["operator_source"],
            effective_operator_source=kwargs["effective_operator_source"],
            fallback_used=kwargs["fallback_used"],
            query_clean_target_used_by_method=kwargs[
                "query_clean_target_used_by_method"
            ],
            status="failed_metric_numerical",
            failure_type=type(error).__name__,
            failure_message=str(error),
            deterministic_checkpoint_operator_scope=kwargs.get(
                "deterministic_checkpoint_operator_scope", ""
            ),
            runtime=kwargs.get("runtime"),
        )


def _training_runtime_fields(
    config: Mapping[str, Any], operator_scope: str
) -> dict[str, Any]:
    path = _scope_output_paths(config, operator_scope)["result_summary"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != _protocol_id(config):
        raise ValueError("deterministic training summary protocol differs")
    if payload.get("operator_scope") != operator_scope:
        raise ValueError("deterministic training summary crossed operator scopes")
    if not str(payload.get("status", "")).startswith("completed"):
        raise ValueError("deterministic training summary is not terminal-complete")
    return {
        "training_updates_completed": int(payload["steps_completed"]),
        "training_selected_checkpoint_updates": "",
        "training_walltime_seconds": float(
            payload.get("cumulative_training_walltime_seconds", payload["walltime_seconds"])
        ),
        "training_model_parameters": int(payload["model_parameters"]),
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


def _write_optional_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, empty_fields: Sequence[str]
) -> None:
    if rows:
        _write_csv(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=tuple(empty_fields), lineterminator="\n"
        )
        writer.writeheader()


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
    if _uses_common_record_eligibility(config):
        for coverage_name in (
            "training_record_coverage",
            "validation_record_coverage",
        ):
            scope_coverages = [
                deterministic_checkpoints[scope]["extra"][coverage_name]
                for scope in FROZEN_OPERATOR_SOURCES
            ]
            reference = scope_coverages[0]
            for coverage in scope_coverages[1:]:
                if coverage != reference:
                    raise ValueError(
                        f"operator-scope checkpoints differ in {coverage_name}"
                    )
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
    seed_status_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
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
                _safe_metric_row(
                    failure_rows=failure_rows,
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
        unet_training_runtime = _training_runtime_fields(config, effective_source)
        unet_training_runtime["training_selected_checkpoint_updates"] = int(
            deterministic_checkpoints[effective_source]["step"]
        )
        try:
            unet_output, unet_runtime = _deterministic_restore(
                deterministic,
                prepared,
                projector,
                base,
                device=device,
                batch_size=int(comparison["inference_batch_size"]),
            )
        except Exception as error:
            if not _retainable_method_failure(error):
                raise
            failure = {
                "partition": partition,
                "source_record": f"sim{source_record:02d}",
                "operator_source": requested_source,
                "effective_operator_source": effective_source,
                "method_id": "task_matched_multichannel_deterministic_UNet",
                "seed": "",
                "status": "failed_method_numerical",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
            failure_rows.append(failure)
            rows.append(
                _failed_metric_row(
                    partition=partition,
                    prepared=prepared,
                    method_id="task_matched_multichannel_deterministic_UNet",
                    operator_source=requested_source,
                    effective_operator_source=effective_source,
                    fallback_used=fallback,
                    query_clean_target_used_by_method=(
                        requested_arm.query_clean_target_used
                    ),
                    status="failed_method_numerical",
                    failure_type=type(error).__name__,
                    failure_message=str(error),
                    deterministic_checkpoint_operator_scope=effective_source,
                    runtime=unet_training_runtime,
                )
            )
        else:
            rows.append(
                _safe_metric_row(
                    failure_rows=failure_rows,
                    partition=partition,
                    prepared=prepared,
                    method_id="task_matched_multichannel_deterministic_UNet",
                    operator_source=requested_source,
                    effective_operator_source=effective_source,
                    restored=unet_output,
                    projector=projector,
                    oracle=oracle,
                    artifact_mask=artifact_mask,
                    runtime={**unet_runtime, **unet_training_runtime},
                    fallback_used=fallback,
                    query_clean_target_used_by_method=(
                        requested_arm.query_clean_target_used
                    ),
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
                try:
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
                except Exception as error:
                    if not _retainable_method_failure(error):
                        raise
                    failure = {
                        "partition": partition,
                        "source_record": f"sim{source_record:02d}",
                        "operator_source": requested_source,
                        "effective_operator_source": effective_source,
                        "method_id": method_id,
                        "seed": seed,
                        "status": "failed_seed_numerical",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                    failure_rows.append(failure)
                    seed_status_rows.append(failure)
                    continue
                restored_by_seed.append(restored)
                runtimes.append(runtime)
                seed_status_rows.append(
                    {
                        "partition": partition,
                        "source_record": f"sim{source_record:02d}",
                        "operator_source": requested_source,
                        "effective_operator_source": effective_source,
                        "method_id": method_id,
                        "seed": seed,
                        "status": "success",
                        "network_calls_total": int(runtime["network_calls_total"]),
                        "latency_seconds": float(runtime["latency_seconds"]),
                        "peak_memory_mb": float(runtime["peak_memory_mb"]),
                    }
                )
            runtime = {
                "function_evaluations": int(budget[f"{candidate}_network_calls"]),
                "function_evaluations_per_seed_per_window": int(
                    budget[f"{candidate}_network_calls"]
                ),
                "total_function_evaluations_per_window": (
                    len(runtimes) * int(budget[f"{candidate}_network_calls"])
                ),
                "planned_total_function_evaluations_per_window": (
                    len(seeds) * int(budget[f"{candidate}_network_calls"])
                ),
                "network_calls_total": sum(
                    int(value["network_calls_total"]) for value in runtimes
                ),
                "latency_seconds": sum(float(value["latency_seconds"]) for value in runtimes),
                "peak_memory_mb": max(
                    (float(value["peak_memory_mb"]) for value in runtimes),
                    default=0.0,
                ),
                "algorithmic_seed_count": len(seeds),
                "successful_algorithmic_seed_count": len(runtimes),
                "failed_algorithmic_seed_count": len(seeds) - len(runtimes),
                "seeds_are_statistical_units": False,
            }
            if restored_by_seed:
                posterior_mean = np.mean(np.stack(restored_by_seed, axis=0), axis=0)
                row = _safe_metric_row(
                    failure_rows=failure_rows,
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
                    status=(
                        "success"
                        if len(restored_by_seed) == len(seeds)
                        else "failed_partial_seed_coverage"
                    ),
                )
                if len(restored_by_seed) != len(seeds):
                    row["failure_type"] = "partial_seed_failure"
                    row["failure_message"] = (
                        f"{len(seeds) - len(restored_by_seed)} of {len(seeds)} seeds failed"
                    )
                rows.append(row)
            else:
                rows.append(
                    _failed_metric_row(
                        partition=partition,
                        prepared=prepared,
                        method_id=method_id,
                        operator_source=requested_source,
                        effective_operator_source=effective_source,
                        fallback_used=fallback,
                        query_clean_target_used_by_method=(
                            requested_arm.query_clean_target_used
                        ),
                        status="failed_all_seeds",
                        failure_type="all_seed_numerical_failure",
                        failure_message=f"all {len(seeds)} seeds failed",
                        runtime=runtime,
                    )
                )
    output_dir = stable_root / f"sim{source_record:02d}"
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "seed_status.csv", seed_status_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failures.json").write_text(
        json.dumps(failure_rows, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed_exploratory_source_record_comparison",
        "protocol_id": _protocol_id(config),
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
        "method_rows": len(rows),
        "successful_method_rows": sum(row["status"] == "success" for row in rows),
        "failed_or_partial_method_rows": sum(
            row["status"] != "success" for row in rows
        ),
        "retained_failure_count": len(failure_rows),
        "per_seed_failure_count": sum(row.get("seed", "") != "" for row in failure_rows),
        "seed_status": str(output_dir / "seed_status.csv"),
        "failures": str(output_dir / "failures.json"),
        "deterministic_comparator_scope": (
            dict(_mapping(config, "deterministic_comparator_scope"))
            if _uses_common_record_eligibility(config)
            else {
                "supervision": "paired_supervised_clean_target",
                "same_supervision_as_clean_prior": False,
                "formal_G3_evidence": False,
            }
        ),
        "deterministic_scope_checkpoints": {
            operator_scope: {
                "step": int(deterministic_checkpoints[operator_scope]["step"]),
                "deployable": _scope_deployable(operator_scope),
                "training_bundle_operator_sources": [operator_scope],
                "validation_bundle_operator_sources": [operator_scope],
                "training_record_coverage": deterministic_checkpoints[
                    operator_scope
                ]["extra"].get("training_record_coverage", {}),
                "validation_record_coverage": deterministic_checkpoints[
                    operator_scope
                ]["extra"].get("validation_record_coverage", {}),
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


_STAGE3_PERFORMANCE_METRICS = (
    "e_parallel",
    "e_perp",
    "rrmse",
    "correlation",
    "psd_distortion",
)


def _csv_float(row: Mapping[str, str], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    if not value:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _descriptive_method_summary(
    *,
    selected: Sequence[Mapping[str, str]],
    operator_source: str,
    method_id: str,
    estimand: str,
    denominator_records: int,
) -> dict[str, Any]:
    successful = [row for row in selected if row.get("status") == "success"]
    result: dict[str, Any] = {
        "estimand": estimand,
        "operator_source": operator_source,
        "method_id": method_id,
        "availability_denominator_records": denominator_records,
        "requested_rows": len(selected),
        "success_rows": len(successful),
        "failed_or_partial_rows": len(selected) - len(successful),
        "fallback_rows": sum(
            str(row.get("fallback_used", "")).lower() == "true" for row in selected
        ),
        "deterministic_checkpoint_scopes": "|".join(
            sorted(
                {
                    row.get("deterministic_checkpoint_operator_scope", "")
                    for row in selected
                    if row.get("deterministic_checkpoint_operator_scope", "")
                }
            )
        ),
    }
    for metric in _STAGE3_PERFORMANCE_METRICS:
        values = [
            value
            for row in successful
            if (value := _csv_float(row, metric)) is not None
        ]
        result[f"median_{metric}"] = float(np.median(values)) if values else ""
    for metric, reducer in (
        ("latency_seconds", np.median),
        ("peak_memory_mb", np.max),
        ("function_evaluations", np.median),
        ("total_function_evaluations_per_window", np.median),
        ("network_calls_total", np.median),
        ("training_updates_completed", np.median),
        ("training_selected_checkpoint_updates", np.median),
        ("training_walltime_seconds", np.median),
        ("training_model_parameters", np.median),
    ):
        values = [
            value
            for row in successful
            if (value := _csv_float(row, metric)) is not None
        ]
        result[f"summary_{metric}"] = float(reducer(values)) if values else ""
    return result


def _paired_delta_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    successful = {
        (row["source_record"], row["operator_source"], row["method_id"]): row
        for row in rows
        if row.get("status") == "success"
    }
    output: list[dict[str, Any]] = []
    comparators = (
        "deterministic_Qy",
        "deterministic_soft_proximal",
        "task_matched_multichannel_deterministic_UNet",
    )
    for row in rows:
        if row.get("status") != "success":
            continue
        operator_source = row["operator_source"]
        fallback = str(row.get("fallback_used", "")).lower() == "true"
        if operator_source == "matching_p0" and fallback:
            estimands = ("matching_request_fallback_policy",)
        elif operator_source == "matching_p0":
            estimands = (
                "matching_p0_eligible_only",
                "matching_request_fallback_policy",
            )
        else:
            estimands = ("operator_effect",)
        for comparator in comparators:
            if row["method_id"] == comparator:
                continue
            reference = successful.get(
                (row["source_record"], operator_source, comparator)
            )
            if reference is None:
                continue
            for estimand in estimands:
                delta: dict[str, Any] = {
                    "estimand": estimand,
                    "source_record": row["source_record"],
                    "operator_source": operator_source,
                    "effective_operator_source": row["effective_operator_source"],
                    "fallback_used": fallback,
                    "method_id": row["method_id"],
                    "comparator_method_id": comparator,
                }
                complete = True
                for metric in _STAGE3_PERFORMANCE_METRICS:
                    left = _csv_float(row, metric)
                    right = _csv_float(reference, metric)
                    if left is None or right is None:
                        complete = False
                        break
                    delta[f"delta_{metric}"] = left - right
                if complete:
                    output.append(delta)
    return output


def _aggregate_stage3_v3(
    config: Mapping[str, Any],
    *,
    partition: str,
    source_records: Sequence[int],
    root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    seed_rows: list[dict[str, str]] = []
    failures: list[dict[str, Any]] = []
    for record in source_records:
        record_root = root / f"sim{record:02d}"
        with (record_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows.extend(csv.DictReader(stream))
        with (record_root / "seed_status.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            seed_rows.extend(csv.DictReader(stream))
        value = json.loads((record_root / "failures.json").read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("stage-3 record failure summary must be a list")
        failures.extend(value)

    expected = len(source_records) * len(FROZEN_OPERATOR_SOURCES) * len(FROZEN_METHODS)
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
    if len(rows) != expected or actual_keys != expected_keys or len(actual_keys) != len(rows):
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
                raise ValueError("deterministic U-Net checkpoint/effective scope mismatch")
        elif checkpoint_scope:
            raise ValueError("non-U-Net method unexpectedly names a U-Net checkpoint")

    operator_summaries: list[dict[str, Any]] = []
    fallback_summaries: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    denominator = len(source_records)
    for operator_source in FROZEN_OPERATOR_SOURCES:
        for method_id in FROZEN_METHODS:
            requested = [
                row
                for row in rows
                if row["operator_source"] == operator_source
                and row["method_id"] == method_id
            ]
            if len(requested) != denominator:
                raise AssertionError("stage-3 summary lost source-record cells")
            eligible = [
                row
                for row in requested
                if not (
                    operator_source == "matching_p0"
                    and row["fallback_used"].lower() == "true"
                )
            ]
            operator_summaries.append(
                _descriptive_method_summary(
                    selected=eligible,
                    operator_source=operator_source,
                    method_id=method_id,
                    estimand=(
                        "matching_p0_eligible_only"
                        if operator_source == "matching_p0"
                        else "operator_effect"
                    ),
                    denominator_records=denominator,
                )
            )
            coverage.append(
                {
                    "operator_source": operator_source,
                    "method_id": method_id,
                    "availability_denominator_records": denominator,
                    "eligible_operator_rows": len(eligible),
                    "fallback_rows": len(requested) - len(eligible),
                    "success_rows": sum(row["status"] == "success" for row in requested),
                    "failed_or_partial_rows": sum(
                        row["status"] != "success" for row in requested
                    ),
                }
            )
            if operator_source == "matching_p0":
                fallback_summaries.append(
                    _descriptive_method_summary(
                        selected=requested,
                        operator_source=operator_source,
                        method_id=method_id,
                        estimand="matching_request_fallback_policy",
                        denominator_records=denominator,
                    )
                )

    paired_deltas = _paired_delta_rows(rows)
    _write_csv(root / "metrics.csv", rows)
    _write_csv(root / "operator_effect_eligible_summary.csv", operator_summaries)
    _write_csv(root / "matching_fallback_policy_summary.csv", fallback_summaries)
    _write_csv(root / "coverage_and_feasibility.csv", coverage)
    _write_optional_csv(
        root / "within_record_paired_deltas.csv",
        paired_deltas,
        empty_fields=(
            "estimand",
            "source_record",
            "operator_source",
            "effective_operator_source",
            "fallback_used",
            "method_id",
            "comparator_method_id",
            *(f"delta_{metric}" for metric in _STAGE3_PERFORMANCE_METRICS),
        ),
    )
    _write_csv(root / "seed_status.csv", seed_rows)
    (root / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    failure_status_counts = Counter(row["status"] for row in rows)
    summary = {
        "status": "completed_descriptive_no_broad_classifier",
        "protocol_id": _protocol_id(config),
        "partition": partition,
        "source_records": [f"sim{value:02d}" for value in source_records],
        "records_are_participants": False,
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "historical_records_are_fresh_evidence": False,
        "frozen_current_status": FROZEN_STATUS,
        "broad_classifier_enabled": False,
        "operator_scope_isolation_verified": True,
        "common_record_eligibility_verified_from_checkpoints": True,
        "matching_operator_effect_estimand": "eligible_only_no_fallback_rows",
        "matching_deployment_policy_estimand": "all_requested_rows_with_POP_fallback",
        "paired_deltas_are_within_source_record": True,
        "method_status_counts": dict(sorted(failure_status_counts.items())),
        "retained_failure_count": len(failures),
        "per_seed_failure_count": sum(row.get("seed", "") != "" for row in failures),
        "deterministic_comparator_scope": dict(
            _mapping(config, "deterministic_comparator_scope")
        ),
        "metrics": str(root / "metrics.csv"),
        "operator_effect_eligible_summary": str(
            root / "operator_effect_eligible_summary.csv"
        ),
        "matching_fallback_policy_summary": str(
            root / "matching_fallback_policy_summary.csv"
        ),
        "coverage_and_feasibility": str(root / "coverage_and_feasibility.csv"),
        "within_record_paired_deltas": str(root / "within_record_paired_deltas.csv"),
        "seed_status": str(root / "seed_status.csv"),
        "failures": str(root / "failures.json"),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "result_summary.json").write_text(
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
    if _uses_common_record_eligibility(config):
        return _aggregate_stage3_v3(
            config,
            partition=partition,
            source_records=source_records,
            root=root,
            run_dir=run_dir,
        )
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
        "protocol_id": _protocol_id(config),
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
