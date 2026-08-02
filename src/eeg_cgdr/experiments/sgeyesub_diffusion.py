"""Frozen natural-EEG SGEYESUB diffusion comparator contracts.

This module is deliberately separate from the deterministic SGEYESUB operator
audit and from the paired Klados stage-3 runners.  It defines the fail-closed
parts of a prospective release-internal experiment:

* block 1 supplies outer-fold training data and held-out calibration;
* block 2 supplies held-out natural-EEG queries;
* weak targets are low-artifact *observed* block-1 EEG, never clean truth;
* query EOG/classes/trial labels are opened only after every arm output for a
  held-out stem has been frozen; and
* participant stems, rather than windows or algorithmic seeds, are the
  scientific units.

The helpers below contain no hidden file discovery and perform no work at
import time.  The training/inference orchestration entry point remains
fail-closed until its Slurm/CLI route wires these audited primitives to the
existing checkpointed model trainers.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from eeg_cgdr.data.sgeyesub import (
    QUERY_EVALUATION_ONLY_FIELDS,
    SGEYESUB_DEVELOPMENT_STUDIES,
    SGEYESUB_EVALUATION_STUDIES,
    SgeyesubProtocolRow,
)
from eeg_cgdr.operators import (
    CalibrationBatch,
    P0Config,
    P0FitOutcome,
    P0Transfer,
    fit_p0,
)


PROTOCOL_ID = "sgeyesub_natural_eeg_diffusion_incremental_v1"
CONFIG_PATH = Path("configs/cgdr/sgeyesub_diffusion_incremental.yaml")
FOLD_COUNT_PER_STUDY = 5
WINDOW_SECONDS = 2.0
EXPECTED_TRIAL_SECONDS = 8.0
LOW_ARTIFACT_LABEL = 6
LOW_ARTIFACT_MINIMUM_FRACTION = 0.95
ARTIFACT_LABELS = frozenset(range(1, 6))
ARTIFACT_MINIMUM_FRACTION = 0.25
FIXED_SUCCESSFUL_UPDATES = 6000
FIXED_BATCH_SIZE = 8
FIXED_LEARNING_RATE = 2.0e-4
FIXED_WEIGHT_DECAY = 1.0e-4
FIXED_GRADIENT_CLIP_NORM = 1.0
FIXED_SEED = 20260802
FIXED_CONDITIONAL_MODEL_SEED = 20260803
FIXED_CHECKPOINT_INTERVAL = 250
FIXED_DDIM_STEPS = 100
FIXED_DDIM_ETA = 0.0
EVALUATION_AVAILABILITY_DENOMINATOR = 44
EVALUATION_COMPATIBLE_DENOMINATOR = 43
PREBLOCKED_RECORDING_KEY = "study05/study05_p42"

DETERMINISTIC_METHOD_ID = "task_matched_multichannel_deterministic_UNet"
CONDITIONAL_METHOD_ID = "operator_conditioned_conditional_diffusion_DDIM100"
PRIMARY_METHOD_IDS = (CONDITIONAL_METHOD_ID, DETERMINISTIC_METHOD_ID)
REPORTED_ARM_IDS = (
    "raw_observation",
    "population_projector_Qy",
    "matching_projector_Qy",
    "matching_projector_soft_proximal",
    DETERMINISTIC_METHOD_ID,
    CONDITIONAL_METHOD_ID,
)
NATURAL_DECISION_PASS = (
    "natural_SGE_conditional_diffusion_supported_over_matched_UNet_under_"
    "frozen_weak_supervision_protocol"
)
NATURAL_DECISION_FAIL = (
    "no_detectable_incremental_value_for_tested_SGE_conditional_protocol"
)
NATURAL_DECISION_INCONCLUSIVE = "inconclusive"

_HIGHER_IS_BETTER = (
    "eog_coherence_reduction",
    "matching_projector_attenuation_db",
    "nonartifact_observation_preservation",
    "condition_erp_observation_relative_preservation",
)
_LOWER_IS_BETTER = (
    "reference_free_psd_distortion",
    "reference_free_covariance_distortion",
)
_PAIRED_METRICS = (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER)
_METHOD_PERFORMANCE_METRICS = (
    "heldout_eog_prediction_remaining_ratio",
    "eog_coherence_reduction",
    "matching_projector_attenuation_db",
    "nonartifact_observation_preservation",
    "reference_free_psd_distortion",
    "reference_free_covariance_distortion",
    "condition_erp_observation_relative_preservation",
    "observation_change_ratio",
)
_RESOURCE_FIELDS = (
    "latency_seconds",
    "peak_memory_mb",
    "network_calls_per_window",
    "parameter_count",
    "training_walltime_seconds",
)
_CONFIGURED_EVALUATION_METRICS = (
    "heldout_eog_prediction_remaining_ratio",
    "eog_coherence_reduction",
    "matching_projector_attenuation_db",
    "nonartifact_observation_preservation",
    "reference_free_psd_distortion",
    "reference_free_covariance_distortion",
    "condition_erp_observation_relative_preservation",
    "downstream_task_preservation_when_label_semantics_allow",
    "observation_change_ratio",
    "latency_seconds",
    "peak_memory_mb",
    "failure_status",
)
LEGAL_INFERENCE_FIELDS = frozenset(
    {
        "observed_query_EEG",
        "operator_projector",
        "support_derived_matching_projector_or_outer_training_population_projector",
        "outer_training_normalization",
        "shared_framewise_attenuation",
        "outer_training_calibrated_EEG_only_attenuation",
        "valid_time_mask",
        "legal_study_layout_reference_and_sampling_metadata",
    }
)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"SGEYESUB diffusion config section {key!r} is missing")
    return value


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _canonical_layout_id(value: object) -> str:
    token = str(value).strip().lower().replace("_", "")
    if not token.startswith("layout") or not token[6:].isdigit():
        raise ValueError(f"invalid SGEYESUB layout ID: {value!r}")
    return f"layout_{int(token[6:]):02d}"


def _validate_fold_entries(
    entries: object,
    *,
    expected_studies: tuple[str, ...],
    expected_stem_count: int,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("frozen folds must be an explicit sequence")
    frozen: list[dict[str, Any]] = []
    fold_ids: set[str] = set()
    stems: list[str] = []
    by_study: dict[str, int] = {study: 0 for study in expected_studies}
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("each frozen fold must be a mapping")
        fold_id = str(raw.get("fold_id", ""))
        study = str(raw.get("study", ""))
        heldout = tuple(str(value) for value in raw.get("heldout_stems", ()))
        if not fold_id or fold_id in fold_ids:
            raise ValueError("frozen fold IDs must be non-empty and unique")
        if study not in expected_studies:
            raise ValueError("frozen fold uses a study outside its partition")
        if not heldout or len(heldout) != len(set(heldout)):
            raise ValueError("each frozen fold needs unique held-out stems")
        if any(not stem.startswith(f"{study}_p") for stem in heldout):
            raise ValueError("held-out stem does not belong to the fold study")
        sampling_rate = float(raw.get("sampling_rate_hz", float("nan")))
        channels = int(raw.get("eeg_channels", -1))
        if not np.isfinite(sampling_rate) or sampling_rate <= 0 or channels < 2:
            raise ValueError("fold sampling rate/channel count is invalid")
        frozen.append(
            {
                "fold_id": fold_id,
                "study": study,
                "layout_id": _canonical_layout_id(raw.get("layout_id")),
                "sampling_rate_hz": sampling_rate,
                "eeg_channels": channels,
                "heldout_stems": heldout,
            }
        )
        fold_ids.add(fold_id)
        stems.extend(heldout)
        by_study[study] += 1
    if any(count != FOLD_COUNT_PER_STUDY for count in by_study.values()):
        raise ValueError("each SGEYESUB study must have exactly five frozen folds")
    if len(stems) != expected_stem_count or len(stems) != len(set(stems)):
        raise ValueError("frozen fold held-out coverage is incomplete or duplicated")
    return tuple(frozen)


def validate_sgeyesub_diffusion_config(config: Mapping[str, Any]) -> None:
    """Validate the complete prospective protocol without reading outcomes."""

    _require_equal(config.get("schema_version"), 1, "schema_version must be 1")
    _require_equal(config.get("harness_level"), 1, "HARNESS_LEVEL must be 1")
    _require_equal(config.get("protocol_id"), PROTOCOL_ID, "protocol_id changed")
    _require_equal(
        config.get("protocol_status"),
        "prospective_frozen_before_development_or_evaluation_outcomes",
        "protocol must be frozen prospectively",
    )
    if config.get("formal_g1_or_g3_evidence") is not False:
        raise ValueError("natural SGE experiment cannot claim formal G1/G3")

    dataset = _mapping(config, "dataset")
    if dataset.get("clean_target_available") is not False:
        raise ValueError("SGEYESUB has no verified clean target")
    _require_equal(
        dataset.get("clean_waveform_recovery_claim"),
        "forbidden",
        "clean waveform recovery claims must stay forbidden",
    )
    _require_equal(
        dataset.get("weak_target_semantics"),
        "low_artifact_observed_EEG_not_clean_truth",
        "weak-target semantics changed",
    )
    _require_equal(
        dataset.get("participant_unit"),
        "release_scoped_study_participant_stem",
        "scientific unit must be a release-scoped participant stem",
    )

    compatibility = _mapping(config, "compatibility")
    _require_equal(
        tuple(compatibility.get("cell_fields", ())),
        (
            "study",
            "exact_ordered_EEG_channel_layout",
            "reference_cell",
            "sampling_rate_hz",
        ),
        "exact-cell compatibility fields changed",
    )
    _require_equal(
        compatibility.get("cross_cell_model_operator_or_normalization_pooling"),
        "forbidden",
        "cross-cell pooling must remain forbidden",
    )

    split = _mapping(config, "split")
    if split.get("split_before_windowing") is not True:
        raise ValueError("participant split must occur before windowing")
    _require_equal(
        tuple(split.get("development_studies", ())),
        SGEYESUB_DEVELOPMENT_STUDIES,
        "development studies changed",
    )
    _require_equal(
        tuple(split.get("evaluation_studies", ())),
        SGEYESUB_EVALUATION_STUDIES,
        "evaluation studies changed",
    )
    _require_equal(split.get("development_compatible_stems"), 15, "development denominator changed")
    _require_equal(split.get("evaluation_available_stems"), 44, "availability denominator changed")
    _require_equal(split.get("evaluation_compatible_stems"), 43, "compatible denominator changed")
    preblocked = split.get("evaluation_preblocked")
    if not isinstance(preblocked, Sequence) or len(preblocked) != 1:
        raise ValueError("exactly one evaluation stem must be preblocked")
    blocked = preblocked[0]
    if not isinstance(blocked, Mapping):
        raise ValueError("preblocked entry must be a mapping")
    _require_equal(
        blocked.get("recording_key"),
        PREBLOCKED_RECORDING_KEY,
        "the frozen singleton blocker changed",
    )
    if blocked.get("contributes_to_performance") is not False or blocked.get(
        "contributes_to_availability_denominator"
    ) is not True:
        raise ValueError("singleton coverage semantics changed")
    _validate_fold_entries(
        split.get("development_folds"),
        expected_studies=SGEYESUB_DEVELOPMENT_STUDIES,
        expected_stem_count=15,
    )
    _validate_fold_entries(
        split.get("evaluation_folds"),
        expected_studies=SGEYESUB_EVALUATION_STUDIES,
        expected_stem_count=43,
    )

    windowing = _mapping(config, "windowing")
    required_window = {
        "window_seconds": WINDOW_SECONDS,
        "expected_trial_seconds": EXPECTED_TRIAL_SECONDS,
        "expected_complete_windows_per_trial": 4,
        "trial_local": True,
        "overlap_samples": 0,
        "cross_trial_windows": "forbidden",
        "incomplete_window_action": "reject",
        "padding": "forbidden",
    }
    if any(windowing.get(key) != value for key, value in required_window.items()):
        raise ValueError("trial-local two-second window contract changed")
    expected_samples = {
        "study01": 400,
        "study02": 400,
        "study03": 400,
        "study04": 200,
        "study05": 512,
    }
    _require_equal(
        dict(_mapping(windowing, "per_study_window_samples")),
        expected_samples,
        "per-study two-second window lengths changed",
    )

    outer = _mapping(config, "outer_fold_fit")
    required_outer = {
        "learned_weight_training_participants": "same_cell_outer_training_stems_only",
        "learned_weight_training_block": 1,
        "heldout_stem_excluded_from_learned_weights_and_normalization": True,
        "heldout_support_block": 1,
        "heldout_query_block": 2,
        "support_query_overlap": "forbidden",
        "normalization_scope": "same_cell_outer_training_block1_only",
        "normalization_refit_on_heldout_stem": "forbidden",
    }
    if any(outer.get(key) != value for key, value in required_outer.items()):
        raise ValueError("outer-fold fit boundary changed")
    weak = _mapping(outer, "weak_supervision")
    required_weak = {
        "low_artifact_target_rule": "artifactclass_6_fraction_greater_than_or_equal_0.95",
        "artifact_source_rule": "artifactclasses_1_to_5_fraction_greater_than_or_equal_0.25",
        "same_raw_window_in_both_roles": "forbidden",
        "paired_clean_truth_claim": "forbidden",
        "p0_rank": 2,
        "p0_ridge_lambda": 0.01,
        "p0_maximum_reference_condition": 1.0e4,
        "p0_minimum_singular_ratio": 1.0e-4,
        "p0_minimum_movement_coverage": 0.01,
        "p0_movement_threshold": 1.0,
        "p0_bootstrap_replicates": 32,
        "p0_bootstrap_block_samples": 800,
        "p0_minimum_bootstrap_success": 0.90,
        "p0_maximum_bootstrap_median_distance": 0.25,
        "p0_maximum_bootstrap_q90_distance": 0.50,
        "p0_fit_samples": "full_block1_per_stem",
        "pairing_order": "sorted_targets_cycled_artifact_sources_seed_20260802",
        "artifact_eog_standardization": "source_full_block1_mean_std",
    }
    if any(weak.get(key) != value for key, value in required_weak.items()):
        raise ValueError("weak-supervision construction changed")
    attenuation = _mapping(outer, "eeg_only_attenuation")
    required_attenuation = {
        "calibration_scope": "same_cell_outer_training_block1_only",
        "query_EOG_input": "forbidden",
        "query_artifactclasses_input": "forbidden",
        "query_trial_labels_input": "forbidden",
        "formula": "a=sqrt(1/(1+r^2))",
        "r_definition": "norm(Pi*y_t)/outer_training_block1_projected_energy_scale",
        "scale_rule": "median_finite_positive_framewise_projected_norm_same_cell_outer_training_block1",
        "scale_floor": 1.0e-6,
    }
    if any(attenuation.get(key) != value for key, value in required_attenuation.items()):
        raise ValueError("EEG-only attenuation rule changed")

    boundary = _mapping(config, "information_boundary")
    _require_equal(
        tuple(boundary.get("inference_visible_fields", ())),
        (
            "observed_query_EEG",
            "support_derived_matching_projector_or_outer_training_population_projector",
            "outer_training_normalization",
            "outer_training_calibrated_EEG_only_attenuation",
            "legal_study_layout_reference_and_sampling_metadata",
        ),
        "inference visible-field contract changed",
    )
    forbidden_evaluation_fields = {
        "query_external_EOG",
        "query_artifactclasses",
        "query_trial_labels",
        "query_trial_ids_when_present",
        "query_outcomes",
    }
    if set(boundary.get("query_evaluation_only_fields", ())) != forbidden_evaluation_fields:
        raise ValueError("query evaluation-only field list changed")
    _require_equal(
        boundary.get("query_evaluation_fields_opening"),
        "after_all_arm_outputs_for_the_heldout_stem_are_frozen",
        "query annotations cannot open before all outputs freeze",
    )
    _require_equal(
        boundary.get("query_evaluation_fields_for_fit_selection_or_inference"),
        "forbidden",
        "query annotations cannot enter fitting/selection/inference",
    )

    matched = _mapping(config, "matched_comparison")
    _require_equal(tuple(matched.get("primary_pair", ())), PRIMARY_METHOD_IDS, "primary pair changed")
    for flag in (
        "same_information_inputs",
        "same_outer_training_stems",
        "same_weak_supervision_pairs_and_order",
        "same_windowing_channels_normalization_and_operator_conditioning",
        "training_update_budget_matched",
        "mixed_precision",
    ):
        if matched.get(flag) is not True:
            raise ValueError(f"matched-comparison flag {flag!r} must be true")
    expected_matched = {
        "same_successful_optimizer_updates": FIXED_SUCCESSFUL_UPDATES,
        "inference_compute_budget_matched": (
            "false_report_network_calls_latency_and_memory"
        ),
        "batch_size": FIXED_BATCH_SIZE,
        "learning_rate": FIXED_LEARNING_RATE,
        "weight_decay": FIXED_WEIGHT_DECAY,
        "gradient_clip_norm": FIXED_GRADIENT_CLIP_NORM,
        "seed": FIXED_SEED,
        "checkpoint_interval_successful_updates": FIXED_CHECKPOINT_INTERVAL,
        "early_stopping": "forbidden",
        "checkpoint_selection": "fixed_6000_successful_update_endpoint",
        "model_backbone": "masked_multichannel_UNet",
        "model_width": 64,
        "report_parameter_count": True,
        "report_training_updates_walltime_latency_and_peak_memory": True,
    }
    if any(matched.get(key) != value for key, value in expected_matched.items()):
        raise ValueError("matched model/training budget changed")
    _require_equal(
        dict(_mapping(matched, "model_seed_by_arm")),
        {
            DETERMINISTIC_METHOD_ID: FIXED_SEED,
            CONDITIONAL_METHOD_ID: FIXED_CONDITIONAL_MODEL_SEED,
        },
        "per-arm model seeds changed",
    )
    deterministic = _mapping(matched, "deterministic_UNet")
    _require_equal(
        dict(deterministic),
        {
            "training_objective": (
                "weak_target_valid_time_masked_reconstruction_loss"
            ),
            "inference_network_calls_per_window": 1,
        },
        "deterministic U-Net objective or compute contract changed",
    )
    conditional = _mapping(matched, "conditional_diffusion")
    expected_diffusion = {
        "training_objective": "weak_target_valid_time_masked_epsilon_prediction",
        "num_diffusion_timesteps": 1000,
        "beta_schedule": "linear",
        "beta_start": 1.0e-4,
        "beta_end": 0.02,
        "inference_sampler": "DDIM",
        "ddim_network_calls_per_window": FIXED_DDIM_STEPS,
        "ddim_eta": FIXED_DDIM_ETA,
        "initial_state_seed": FIXED_SEED,
        "per_record_seed_policy": (
            "base_plus_weighted_partition_fold_recording_key_characters"
        ),
        "seed_role": "algorithmic_repeatability_not_an_independent_statistical_unit",
    }
    if any(conditional.get(key) != value for key, value in expected_diffusion.items()):
        raise ValueError("conditional diffusion protocol changed")
    _require_equal(tuple(config.get("reported_arms", ())), REPORTED_ARM_IDS, "reported arm matrix changed")
    restoration = _mapping(config, "restoration")
    _require_equal(
        restoration.get("matching_projector_soft_proximal_formula"),
        "y-(1-a^2)*Pi*y",
        "soft proximal formula changed",
    )
    if restoration.get("soft_proximal_uses_query_external_eog") is not False:
        raise ValueError("soft proximal cannot use query EOG")

    evaluation = _mapping(config, "evaluation")
    _require_equal(evaluation.get("scientific_unit"), "participant_stem", "evaluation unit changed")
    _require_equal(
        evaluation.get("metric_coordinate"),
        "outer_training_channel_zscore",
        "evaluation metric coordinate changed",
    )
    _require_equal(evaluation.get("compatible_performance_denominator"), 43, "compatible denominator changed")
    _require_equal(evaluation.get("availability_denominator"), 44, "availability denominator changed")
    if evaluation.get("algorithmic_seeds_are_independent_statistical_units") is not False:
        raise ValueError("algorithmic seeds cannot become statistical units")
    _require_equal(
        evaluation.get("clean_RRMSE_or_clean_correlation"),
        "forbidden_no_clean_target",
        "clean-target metrics must stay forbidden",
    )
    metrics = tuple(evaluation.get("metrics", ()))
    _require_equal(
        metrics,
        _CONFIGURED_EVALUATION_METRICS,
        "evaluation metric list changed or contains duplicates",
    )
    if any(token.lower() in {"rrmse", "correlation", "snr", "deltasnr"} for token in metrics):
        raise ValueError("clean-waveform metrics are forbidden on SGEYESUB")
    _require_equal(
        dict(_mapping(evaluation, "compute_metric_contract")),
        {
            "deterministic_network_evaluations_per_window": 1,
            "conditional_DDIM_network_evaluations_per_window": FIXED_DDIM_STEPS,
            "report_batched_forward_invocations_separately": True,
            "report_total_and_per_window_latency": True,
        },
        "evaluation compute metric contract changed",
    )
    metric_contract = _mapping(evaluation, "metric_contract")
    _require_equal(
        metric_contract.get("observation_change_ratio"),
        "relative_output_minus_observed_norm_descriptive_only",
        "observation-change metric semantics changed",
    )
    aggregation = _mapping(evaluation, "aggregation")
    if int(aggregation.get("descriptive_participant_stem_bootstrap_replicates", -1)) != 20000:
        raise ValueError("descriptive bootstrap must use 20,000 participant-stem replicates")
    if int(aggregation.get("bootstrap_seed", -1)) != FIXED_SEED:
        raise ValueError("descriptive bootstrap seed changed")

    thresholds = _mapping(config, "prospective_exploratory_thresholds")
    _require_equal(
        thresholds.get("comparison_direction"),
        "conditional_diffusion_minus_matched_deterministic_UNet",
        "paired comparison direction changed",
    )
    _require_equal(thresholds.get("minimum_paired_success_count_of_43"), 39, "coverage threshold changed")
    labels = _mapping(thresholds, "outcome_labels")
    _require_equal(labels.get("pass"), NATURAL_DECISION_PASS, "pass label changed")
    _require_equal(labels.get("fail"), NATURAL_DECISION_FAIL, "fail label changed")
    _require_equal(labels.get("insufficient_coverage_or_mixed_safety"), NATURAL_DECISION_INCONCLUSIVE, "inconclusive label changed")
    forbidden = set(thresholds.get("forbidden_extrapolations", ()))
    if not {"diffusion_is_useless", "EEG_diffusion_is_disproved", "personalization_failed"}.issubset(forbidden):
        raise ValueError("forbidden extrapolation list was weakened")

    execution = _mapping(config, "execution_plan")
    if execution.get("checkpoint_resume_required") is not True:
        raise ValueError("fold training must support checkpoint/resume")
    _require_equal(
        execution.get("scientific_evaluation_scope"),
        "every_valid_block2_window_for_all_43_compatible_stems",
        "full natural-EEG evaluation scope changed",
    )
    outputs = _mapping(config, "outputs")
    root = Path(str(outputs.get("root", "")))
    expected_root = Path(
        "/home/infres/yinwang/denoiseNet/results/cgdr/"
        "sgeyesub_diffusion_incremental"
    )
    if root != expected_root:
        raise ValueError("SGEYESUB diffusion output root changed")
    if Path(str(outputs.get("development_root", ""))) != root / "development":
        raise ValueError("development output root is invalid")
    if Path(str(outputs.get("evaluation_root", ""))) != root / "evaluation":
        raise ValueError("evaluation output root is invalid")


def sgeyesub_p0_config(
    config: Mapping[str, Any],
) -> tuple[P0Config, float]:
    """Materialize the single frozen P0 configuration and movement threshold."""

    validate_sgeyesub_diffusion_config(config)
    weak = _mapping(_mapping(config, "outer_fold_fit"), "weak_supervision")
    return (
        P0Config(
            target_rank=int(weak["p0_rank"]),
            ridge_lambda=float(weak["p0_ridge_lambda"]),
            maximum_reference_condition=float(
                weak["p0_maximum_reference_condition"]
            ),
            minimum_singular_ratio=float(weak["p0_minimum_singular_ratio"]),
            minimum_movement_coverage=float(
                weak["p0_minimum_movement_coverage"]
            ),
            bootstrap_replicates=int(weak["p0_bootstrap_replicates"]),
            bootstrap_block_samples=int(weak["p0_bootstrap_block_samples"]),
            minimum_bootstrap_success=float(
                weak["p0_minimum_bootstrap_success"]
            ),
            maximum_bootstrap_median_distance=float(
                weak["p0_maximum_bootstrap_median_distance"]
            ),
            maximum_bootstrap_q90_distance=float(
                weak["p0_maximum_bootstrap_q90_distance"]
            ),
            seed=FIXED_SEED,
        ),
        float(weak["p0_movement_threshold"]),
    )


@dataclass(frozen=True)
class FrozenSgeyesubFold:
    partition: str
    fold_id: str
    study: str
    layout_id: str
    reference_cell_id: str
    sampling_rate_hz: float
    eeg_channels: int
    training_recording_keys: tuple[str, ...]
    heldout_recording_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.partition not in {"development", "evaluation"}:
            raise ValueError("fold partition must be development or evaluation")
        if not self.training_recording_keys or not self.heldout_recording_keys:
            raise ValueError("fold must contain training and held-out stems")
        if set(self.training_recording_keys) & set(self.heldout_recording_keys):
            raise ValueError("participant leakage between training and held-out stems")


@dataclass(frozen=True)
class FrozenSgeyesubSplit:
    folds: tuple[FrozenSgeyesubFold, ...]
    blocked_recording_keys: tuple[str, ...]
    availability_denominator: int
    compatible_denominator: int

    @property
    def development_folds(self) -> tuple[FrozenSgeyesubFold, ...]:
        return tuple(fold for fold in self.folds if fold.partition == "development")

    @property
    def evaluation_folds(self) -> tuple[FrozenSgeyesubFold, ...]:
        return tuple(fold for fold in self.folds if fold.partition == "evaluation")


def _row_value(row: SgeyesubProtocolRow | Mapping[str, Any], key: str) -> Any:
    return row[key] if isinstance(row, Mapping) else getattr(row, key)


def build_frozen_sgeyesub_folds(
    config: Mapping[str, Any],
    protocol_rows: Sequence[SgeyesubProtocolRow | Mapping[str, Any]],
) -> FrozenSgeyesubSplit:
    """Resolve explicit config folds against audited exact-cell protocol rows."""

    validate_sgeyesub_diffusion_config(config)
    rows_by_study_stem: dict[tuple[str, str], SgeyesubProtocolRow | Mapping[str, Any]] = {}
    for row in protocol_rows:
        key = (str(_row_value(row, "study")), str(_row_value(row, "participant_stem")))
        if key in rows_by_study_stem:
            raise ValueError("duplicate SGEYESUB protocol row")
        rows_by_study_stem[key] = row
    if len(rows_by_study_stem) != 59:
        raise ValueError("full frozen SGEYESUB release requires 59 protocol rows")

    split = _mapping(config, "split")
    folds: list[FrozenSgeyesubFold] = []
    covered: dict[str, list[str]] = {"development": [], "evaluation": []}
    for partition, key, expected_studies in (
        ("development", "development_folds", SGEYESUB_DEVELOPMENT_STUDIES),
        ("evaluation", "evaluation_folds", SGEYESUB_EVALUATION_STUDIES),
    ):
        entries = _validate_fold_entries(
            split[key],
            expected_studies=expected_studies,
            expected_stem_count=15 if partition == "development" else 43,
        )
        for entry in entries:
            study = str(entry["study"])
            heldout_stems = tuple(entry["heldout_stems"])
            held_rows = []
            for stem in heldout_stems:
                try:
                    held_rows.append(rows_by_study_stem[(study, stem)])
                except KeyError as exc:
                    raise ValueError(f"frozen held-out stem is absent: {study}/{stem}") from exc
            layout = str(entry["layout_id"])
            sampling_rate = float(entry["sampling_rate_hz"])
            reference_values = {
                str(_row_value(row, "reference_cell_id")) for row in held_rows
            }
            if len(reference_values) != 1:
                raise ValueError("held-out fold crosses reference cells")
            reference = next(iter(reference_values))
            for row in held_rows:
                if (
                    _canonical_layout_id(_row_value(row, "layout_id")) != layout
                    or float(_row_value(row, "sampling_rate_hz")) != sampling_rate
                    or str(_row_value(row, "partition")) != partition
                    or str(_row_value(row, "status")) != "metadata_ready"
                ):
                    raise ValueError("held-out stem is incompatible with its frozen fold")
            training = tuple(
                sorted(
                    str(_row_value(row, "recording_key"))
                    for row in protocol_rows
                    if str(_row_value(row, "study")) == study
                    and _canonical_layout_id(_row_value(row, "layout_id")) == layout
                    and float(_row_value(row, "sampling_rate_hz")) == sampling_rate
                    and str(_row_value(row, "reference_cell_id")) == reference
                    and str(_row_value(row, "participant_stem")) not in heldout_stems
                    and str(_row_value(row, "status")) == "metadata_ready"
                )
            )
            heldout_keys = tuple(
                str(_row_value(row, "recording_key")) for row in held_rows
            )
            folds.append(
                FrozenSgeyesubFold(
                    partition=partition,
                    fold_id=str(entry["fold_id"]),
                    study=study,
                    layout_id=layout,
                    reference_cell_id=reference,
                    sampling_rate_hz=sampling_rate,
                    eeg_channels=int(entry["eeg_channels"]),
                    training_recording_keys=training,
                    heldout_recording_keys=heldout_keys,
                )
            )
            covered[partition].extend(heldout_keys)
    if len(folds) != 25 or len(covered["development"]) != 15 or len(covered["evaluation"]) != 43:
        raise AssertionError("frozen five-fold coverage changed")
    if len(set(covered["development"])) != 15 or len(set(covered["evaluation"])) != 43:
        raise ValueError("participant stem is held out more than once")
    blocked_rows = [
        str(_row_value(row, "recording_key"))
        for row in protocol_rows
        if str(_row_value(row, "status")) != "metadata_ready"
    ]
    if blocked_rows != [PREBLOCKED_RECORDING_KEY]:
        raise ValueError("preblocked exact-cell set differs from the frozen singleton")
    return FrozenSgeyesubSplit(
        folds=tuple(folds),
        blocked_recording_keys=(PREBLOCKED_RECORDING_KEY,),
        availability_denominator=EVALUATION_AVAILABILITY_DENOMINATOR,
        compatible_denominator=EVALUATION_COMPATIBLE_DENOMINATOR,
    )


@dataclass(frozen=True, order=True)
class TrialWindowOrigin:
    recording_key: str
    trial_ordinal: int
    start_sample: int
    stop_sample: int


@dataclass(frozen=True)
class TrialWindowBatch:
    values: np.ndarray
    valid_time_mask: np.ndarray
    origins: tuple[TrialWindowOrigin, ...]
    samples_per_window: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.valid_time_mask)
        if values.ndim != 3 or mask.shape != (values.shape[0], values.shape[2]):
            raise ValueError("trial-window values/mask shapes differ")
        if values.shape[0] != len(self.origins) or values.shape[2] != self.samples_per_window:
            raise ValueError("trial-window origin/length metadata differ")
        if not np.isfinite(values).all() or not np.all(mask):
            raise ValueError("primary trial windows must be finite and fully valid")


def trial_local_nonoverlap_windows(
    signal: np.ndarray,
    *,
    samples_per_trial: int,
    sampling_rate_hz: float,
    recording_key: str,
) -> TrialWindowBatch:
    """Split a flattened release block within trials into exact 2-second windows."""

    value = np.asarray(signal)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError("signal must be a non-empty channel-major array")
    if not np.isfinite(value).all():
        raise ValueError("signal contains NaN or Inf")
    if not recording_key:
        raise ValueError("recording_key must be non-empty")
    rate = float(sampling_rate_hz)
    window = int(round(WINDOW_SECONDS * rate))
    if not np.isfinite(rate) or rate <= 0 or not math.isclose(window, WINDOW_SECONDS * rate):
        raise ValueError("sampling rate does not define an integral 2-second window")
    if int(samples_per_trial) != samples_per_trial or samples_per_trial < 1:
        raise ValueError("samples_per_trial must be a positive integer")
    if samples_per_trial != int(round(EXPECTED_TRIAL_SECONDS * rate)):
        raise ValueError("SGEYESUB primary protocol requires complete 8-second trials")
    if samples_per_trial % window or value.shape[1] % samples_per_trial:
        raise ValueError("incomplete trial/window rejected; padding is forbidden")
    trials = value.shape[1] // samples_per_trial
    per_trial = samples_per_trial // window
    windows = np.empty((trials * per_trial, value.shape[0], window), dtype=value.dtype)
    origins: list[TrialWindowOrigin] = []
    index = 0
    for trial in range(trials):
        trial_start = trial * samples_per_trial
        for local_start in range(0, samples_per_trial, window):
            flat_start = trial_start + local_start
            windows[index] = value[:, flat_start : flat_start + window]
            origins.append(
                TrialWindowOrigin(
                    recording_key=recording_key,
                    trial_ordinal=trial,
                    start_sample=local_start,
                    stop_sample=local_start + window,
                )
            )
            index += 1
    return TrialWindowBatch(
        values=np.ascontiguousarray(windows),
        valid_time_mask=np.ones((windows.shape[0], window), dtype=bool),
        origins=tuple(origins),
        samples_per_window=window,
    )


@dataclass(frozen=True)
class SgeyesubFoldNormalizer:
    mean: np.ndarray
    standard_deviation: np.ndarray
    training_recording_keys: tuple[str, ...]
    sample_count: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.standard_deviation, dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.size < 2:
            raise ValueError("normalizer moments must be matching channel vectors")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("normalizer moments must be finite with positive scale")
        if not self.training_recording_keys or len(self.training_recording_keys) != len(set(self.training_recording_keys)):
            raise ValueError("normalizer training stems must be unique and non-empty")
        if self.sample_count < 1:
            raise ValueError("normalizer sample count must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", scale)

    def transform(self, signal: np.ndarray) -> np.ndarray:
        value = np.asarray(signal, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != self.mean.size or not np.isfinite(value).all():
            raise ValueError("signal is incompatible with the fold normalizer")
        return (value - self.mean[:, None]) / self.standard_deviation[:, None]

    def inverse_transform(self, signal: np.ndarray) -> np.ndarray:
        value = np.asarray(signal, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != self.mean.size or not np.isfinite(value).all():
            raise ValueError("signal is incompatible with the fold normalizer")
        return value * self.standard_deviation[:, None] + self.mean[:, None]

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
            "training_recording_keys": list(self.training_recording_keys),
            "sample_count": int(self.sample_count),
        }


def fit_outer_training_normalizer(
    signals: Mapping[str, np.ndarray],
    training_recording_keys: Sequence[str],
) -> SgeyesubFoldNormalizer:
    """Fit per-channel moments from named outer-training block-1 signals only."""

    keys = tuple(str(value) for value in training_recording_keys)
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("outer-training recording keys must be unique and non-empty")
    arrays: list[np.ndarray] = []
    for key in keys:
        if key not in signals:
            raise ValueError(f"missing outer-training block-1 signal: {key}")
        value = np.asarray(signals[key], dtype=np.float64)
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 1 or not np.isfinite(value).all():
            raise ValueError(f"invalid outer-training EEG signal: {key}")
        arrays.append(value)
    channels = arrays[0].shape[0]
    if any(value.shape[0] != channels for value in arrays):
        raise ValueError("outer-training signals cross channel layouts")
    samples = sum(value.shape[1] for value in arrays)
    total = sum((value.sum(axis=1) for value in arrays), np.zeros(channels, dtype=np.float64))
    total_square = sum((np.square(value).sum(axis=1) for value in arrays), np.zeros(channels, dtype=np.float64))
    mean = total / samples
    variance = np.maximum(total_square / samples - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("outer-training block1 contains a constant EEG channel")
    return SgeyesubFoldNormalizer(mean, scale, keys, samples)


def _checked_projector(signal: np.ndarray, projector: np.ndarray) -> np.ndarray:
    channels = signal.shape[-2]
    value = np.asarray(projector, dtype=np.float64)
    if value.shape != (channels, channels) or not np.isfinite(value).all():
        raise ValueError("projector does not match EEG channels")
    if not np.allclose(value, value.T, atol=1.0e-6, rtol=1.0e-5) or not np.allclose(
        value @ value, value, atol=2.0e-5, rtol=2.0e-5
    ):
        raise ValueError("projector must be symmetric and idempotent")
    return value


def fit_outer_training_projected_energy_scale(
    items: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    scale_floor: float = 1.0e-6,
) -> float:
    """Median positive framewise ``||Pi y_t||`` over outer-training block 1."""

    if not items or not np.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("projected-energy scale inputs/floor are invalid")
    values: list[np.ndarray] = []
    for signal, projector in items:
        eeg = np.asarray(signal, dtype=np.float64)
        if eeg.ndim != 2 or eeg.shape[1] < 1 or not np.isfinite(eeg).all():
            raise ValueError("outer-training EEG for attenuation is invalid")
        projection = _checked_projector(eeg, projector)
        values.append(np.linalg.norm(projection @ eeg, axis=0))
    joined = np.concatenate(values)
    positive = joined[np.isfinite(joined) & (joined > 0)]
    if positive.size < 1:
        raise ValueError("outer-training projected energy has no positive frame")
    return float(max(np.median(positive), scale_floor))


def eeg_only_frame_attenuation(
    observed: np.ndarray,
    projector: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Compute ``a=sqrt(1/(1+r^2))`` from observed EEG only."""

    value = np.asarray(observed, dtype=np.float64)
    if value.ndim not in {2, 3} or not np.isfinite(value).all():
        raise ValueError("observed EEG must be finite (C,T) or (B,C,T)")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("outer-training attenuation scale must be positive")
    if value.ndim == 2:
        projection = _checked_projector(value, projector)
        ratio = np.linalg.norm(projection @ value, axis=0) / scale
    else:
        projections = np.asarray(projector, dtype=np.float64)
        if projections.ndim == 2:
            projections = np.repeat(projections[None, :, :], value.shape[0], axis=0)
        if projections.shape != (value.shape[0], value.shape[1], value.shape[1]):
            raise ValueError("batched projector shape differs from observed EEG")
        for index in range(value.shape[0]):
            _checked_projector(value[index], projections[index])
        ratio = np.linalg.norm(np.einsum("bij,bjt->bit", projections, value), axis=1) / scale
    attenuation = np.sqrt(1.0 / (1.0 + np.square(ratio)))
    if not np.isfinite(attenuation).all() or np.any(attenuation < 0) or np.any(attenuation > 1):
        raise AssertionError("EEG-only attenuation left [0,1]")
    return attenuation


@dataclass(frozen=True)
class FullBlockP0Fit:
    """P0 outcome plus the raw-EOG statistics used for its fit coordinates."""

    outcome: P0FitOutcome
    eog_mean: np.ndarray
    eog_standard_deviation: np.ndarray
    recording_key: str
    sample_count: int


@dataclass(frozen=True)
class OuterTrainingPopulationP0Fit:
    """Same-cell population P0 with per-source support-EOG standardization."""

    outcome: P0FitOutcome
    training_recording_keys: tuple[str, ...]
    source_eog_statistics: Mapping[str, Mapping[str, list[float]]]
    sample_count: int


def fit_full_block1_p0(
    normalized_support_eeg: np.ndarray,
    support_external_eog: np.ndarray,
    *,
    recording_key: str,
    sampling_rate_hz: float,
    config: P0Config,
    movement_threshold: float,
) -> FullBlockP0Fit:
    """Fit one rank-2 P0 on the complete normalized block-1 support record."""

    eeg = np.asarray(normalized_support_eeg, dtype=np.float64)
    eog = np.asarray(support_external_eog, dtype=np.float64)
    if (
        eeg.ndim != 2
        or eog.ndim != 2
        or eeg.shape[1] != eog.shape[1]
        or eeg.shape[0] < 2
        or eog.shape[0] < 1
        or not np.isfinite(eeg).all()
        or not np.isfinite(eog).all()
    ):
        raise ValueError("full block1 P0 EEG/EOG arrays are invalid or unaligned")
    if config.target_rank != 2 or float(config.ridge_lambda) != 0.01:
        raise ValueError("natural SGE P0 is frozen to rank 2 and ridge 0.01")
    mean = eog.mean(axis=1, keepdims=True)
    standard_deviation = eog.std(axis=1, keepdims=True)
    if np.any(standard_deviation <= np.finfo(np.float64).eps):
        raise ValueError("full block1 EOG contains a constant channel")
    standardized = (eog - mean) / standard_deviation
    outcome = fit_p0(
        CalibrationBatch(
            eeg=eeg,
            eog=standardized,
            participant=recording_key.rsplit("/", 1)[-1],
            source_record="full_block1_support",
            sampling_rate=float(sampling_rate_hz),
        ),
        config,
        movement_threshold=float(movement_threshold),
    )
    diagnostics = {
        **dict(outcome.diagnostics),
        "fit_scope": "full_block1_per_stem",
        "eog_fit_coordinate": "standardized_from_source_full_block1_mean_std",
        "raw_eog_standardization_mean": mean.reshape(-1).tolist(),
        "raw_eog_standardization_standard_deviation": standard_deviation.reshape(
            -1
        ).tolist(),
    }
    transfer = outcome.transfer
    if transfer is not None:
        transfer = replace(transfer, diagnostics=diagnostics)
    annotated = replace(outcome, transfer=transfer, diagnostics=diagnostics)
    return FullBlockP0Fit(
        outcome=annotated,
        eog_mean=mean,
        eog_standard_deviation=standard_deviation,
        recording_key=recording_key,
        sample_count=eeg.shape[1],
    )


def fit_outer_training_population_p0(
    normalized_eeg_by_key: Mapping[str, np.ndarray],
    raw_eog_by_key: Mapping[str, np.ndarray],
    training_recording_keys: Sequence[str],
    *,
    sampling_rate_hz: float,
    p0_config: P0Config,
    movement_threshold: float,
) -> OuterTrainingPopulationP0Fit:
    """Fit population P0 from complete block 1 of same-cell training stems."""

    keys = tuple(str(value) for value in training_recording_keys)
    if len(keys) < 2 or len(keys) != len(set(keys)):
        raise ValueError(
            "population P0 requires at least two unique same-cell outer-training stems"
        )
    if p0_config.target_rank != 2 or float(p0_config.ridge_lambda) != 0.01:
        raise ValueError("natural SGE population P0 is frozen to rank 2/ridge 0.01")
    eeg_parts: list[np.ndarray] = []
    eog_parts: list[np.ndarray] = []
    source_statistics: dict[str, Mapping[str, list[float]]] = {}
    channels: int | None = None
    references: int | None = None
    for key in keys:
        if key not in normalized_eeg_by_key or key not in raw_eog_by_key:
            raise ValueError(f"population P0 lacks named block1 source: {key}")
        eeg = np.asarray(normalized_eeg_by_key[key], dtype=np.float64)
        eog = np.asarray(raw_eog_by_key[key], dtype=np.float64)
        if (
            eeg.ndim != 2
            or eog.ndim != 2
            or eeg.shape[1] != eog.shape[1]
            or eeg.shape[0] < 2
            or eog.shape[0] < 1
            or not np.isfinite(eeg).all()
            or not np.isfinite(eog).all()
        ):
            raise ValueError(f"population P0 block1 source is invalid: {key}")
        if channels is None:
            channels = eeg.shape[0]
            references = eog.shape[0]
        elif eeg.shape[0] != channels or eog.shape[0] != references:
            raise ValueError("population P0 sources cross channel/reference layouts")
        mean = eog.mean(axis=1, keepdims=True)
        standard_deviation = eog.std(axis=1, keepdims=True)
        if np.any(standard_deviation <= np.finfo(np.float64).eps):
            raise ValueError(f"population P0 source has constant EOG: {key}")
        eeg_parts.append(eeg)
        eog_parts.append((eog - mean) / standard_deviation)
        source_statistics[key] = MappingProxyType(
            {
                "mean": mean.reshape(-1).tolist(),
                "standard_deviation": standard_deviation.reshape(-1).tolist(),
            }
        )
    concatenated_eeg = np.concatenate(eeg_parts, axis=1)
    concatenated_eog = np.concatenate(eog_parts, axis=1)
    outcome = fit_p0(
        CalibrationBatch(
            eeg=concatenated_eeg,
            eog=concatenated_eog,
            participant="outer_training_population",
            source_record="same_cell_outer_training_full_block1",
            sampling_rate=float(sampling_rate_hz),
        ),
        p0_config,
        movement_threshold=float(movement_threshold),
    )
    diagnostics = {
        **dict(outcome.diagnostics),
        "fit_scope": "same_cell_outer_training_stems_full_block1_only",
        "heldout_stem_visible": False,
        "source_eog_standardization": "per_source_full_block1_mean_std",
        "training_recording_keys": list(keys),
    }
    transfer = outcome.transfer
    if transfer is not None:
        transfer = replace(transfer, diagnostics=diagnostics)
    annotated = replace(outcome, transfer=transfer, diagnostics=diagnostics)
    return OuterTrainingPopulationP0Fit(
        outcome=annotated,
        training_recording_keys=keys,
        source_eog_statistics=MappingProxyType(source_statistics),
        sample_count=concatenated_eeg.shape[1],
    )


@dataclass(frozen=True)
class WeakSupervisionBundle:
    observed: np.ndarray
    weak_target: np.ndarray
    projector: np.ndarray
    attenuation: np.ndarray
    valid_time_mask: np.ndarray
    recording_keys: tuple[str, ...]
    target_origins: tuple[TrialWindowOrigin, ...]
    artifact_origins: tuple[TrialWindowOrigin, ...]

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed)
        target = np.asarray(self.weak_target)
        if observed.shape != target.shape or observed.ndim != 3:
            raise ValueError("weak observed/target windows must have matching (N,C,L) shape")
        count, channels, length = observed.shape
        if np.asarray(self.projector).shape != (count, channels, channels):
            raise ValueError("weak-pair projectors have the wrong shape")
        if np.asarray(self.attenuation).shape != (count, length):
            raise ValueError("weak-pair attenuation has the wrong shape")
        if np.asarray(self.valid_time_mask).shape != (count, length):
            raise ValueError("weak-pair valid-time mask has the wrong shape")
        if not (
            len(self.recording_keys)
            == len(self.target_origins)
            == len(self.artifact_origins)
            == count
        ):
            raise ValueError("weak-pair provenance length differs from tensor count")
        if any(
            left == right
            for left, right in zip(self.target_origins, self.artifact_origins)
        ):
            raise ValueError("one raw window cannot be target and artifact source")
        if not np.isfinite(observed).all() or not np.isfinite(target).all():
            raise ValueError("weak supervision bundle contains NaN or Inf")
        attenuation = np.asarray(self.attenuation)
        mask = np.asarray(self.valid_time_mask)
        if (
            not np.isfinite(attenuation).all()
            or np.any((attenuation < 0) | (attenuation > 1))
            or not np.all(mask == 1)
        ):
            raise ValueError("weak-pair attenuation/mask violates the frozen protocol")
        projectors = np.asarray(self.projector, dtype=np.float64)
        for index in range(count):
            _checked_projector(observed[index], projectors[index])
        if any(
            origin.recording_key != key
            for key, origin in zip(self.recording_keys, self.target_origins)
        ) or any(
            origin.recording_key != key
            for key, origin in zip(self.recording_keys, self.artifact_origins)
        ):
            raise ValueError("weak-pair origins crossed participant stems")


def build_within_stem_weak_pairs(
    normalized_support_eeg: np.ndarray,
    support_external_eog: np.ndarray,
    support_artifactclasses: np.ndarray,
    *,
    transfer: P0Transfer,
    samples_per_trial: int,
    sampling_rate_hz: float,
    recording_key: str,
    projected_energy_scale: float,
    seed: int = FIXED_SEED,
) -> WeakSupervisionBundle:
    """Build deterministic within-stem support-derived recontamination pairs."""

    eeg = np.asarray(normalized_support_eeg, dtype=np.float64)
    eog = np.asarray(support_external_eog, dtype=np.float64)
    labels = np.asarray(support_artifactclasses).reshape(-1)
    if eeg.ndim != 2 or eog.ndim != 2 or eeg.shape[1] != eog.shape[1] or labels.size != eeg.shape[1]:
        raise ValueError("support EEG/EOG/artifactclasses are not sample aligned")
    if not np.isfinite(eeg).all() or not np.isfinite(eog).all() or not np.isfinite(labels).all():
        raise ValueError("support weak-supervision arrays contain NaN or Inf")
    rounded = np.rint(labels).astype(int)
    if not np.allclose(labels, rounded, atol=1.0e-5, rtol=0.0) or not set(np.unique(rounded)).issubset(set(range(7))):
        raise ValueError("artifactclasses must use integer labels 0..6")
    if transfer.transfer_matrix.shape != (eeg.shape[0], eog.shape[0]):
        raise ValueError("P0 transfer is incompatible with support EEG/EOG")
    if (
        transfer.rank != 2
        or np.asarray(transfer.predicted_contamination).shape != eeg.shape
        or np.asarray(transfer.eog_mean).shape != (eog.shape[0], 1)
        or np.asarray(transfer.eeg_mean).shape != (eeg.shape[0], 1)
        or int(transfer.diagnostics.get("samples", -1)) != eeg.shape[1]
        or float(transfer.diagnostics.get("ridge_lambda", float("nan"))) != 0.01
    ):
        raise ValueError("P0 must be the frozen rank-2 full-block1 ridge fit")
    projector = _checked_projector(eeg, transfer.projector)
    eeg_windows = trial_local_nonoverlap_windows(
        eeg,
        samples_per_trial=samples_per_trial,
        sampling_rate_hz=sampling_rate_hz,
        recording_key=recording_key,
    )
    eog_windows = trial_local_nonoverlap_windows(
        eog,
        samples_per_trial=samples_per_trial,
        sampling_rate_hz=sampling_rate_hz,
        recording_key=recording_key,
    )
    label_windows = trial_local_nonoverlap_windows(
        rounded,
        samples_per_trial=samples_per_trial,
        sampling_rate_hz=sampling_rate_hz,
        recording_key=recording_key,
    )
    if eeg_windows.origins != eog_windows.origins or eeg_windows.origins != label_windows.origins:
        raise AssertionError("support weak-supervision window origins diverged")
    label_values = label_windows.values[:, 0, :].astype(int)
    low_indices = np.flatnonzero(
        np.mean(label_values == LOW_ARTIFACT_LABEL, axis=1)
        >= LOW_ARTIFACT_MINIMUM_FRACTION
    )
    artifact_indices = np.flatnonzero(
        np.mean(np.isin(label_values, tuple(sorted(ARTIFACT_LABELS))), axis=1)
        >= ARTIFACT_MINIMUM_FRACTION
    )
    if low_indices.size < 1 or artifact_indices.size < 1:
        raise ValueError("stem lacks eligible low-artifact or artifact-source windows")
    low_indices = np.asarray(sorted(low_indices.tolist(), key=lambda idx: eeg_windows.origins[idx]))
    artifact_indices = np.asarray(sorted(artifact_indices.tolist(), key=lambda idx: eeg_windows.origins[idx]))
    offset = int(seed) % artifact_indices.size
    artifact_indices = np.roll(artifact_indices, -offset)

    eog_mean = eog.mean(axis=1, keepdims=True)
    eog_scale = eog.std(axis=1, keepdims=True)
    if np.any(eog_scale <= np.finfo(np.float64).eps):
        raise ValueError("full block1 EOG contains a constant channel")
    if transfer.diagnostics.get("eog_fit_coordinate") != (
        "standardized_from_source_full_block1_mean_std"
    ):
        raise ValueError("P0 transfer does not declare standardized full-block1 EOG coordinates")
    recorded_mean = np.asarray(
        transfer.diagnostics.get("raw_eog_standardization_mean", ()),
        dtype=np.float64,
    ).reshape(-1, 1)
    recorded_scale = np.asarray(
        transfer.diagnostics.get(
            "raw_eog_standardization_standard_deviation", ()
        ),
        dtype=np.float64,
    ).reshape(-1, 1)
    if (
        recorded_mean.shape != eog_mean.shape
        or recorded_scale.shape != eog_scale.shape
        or not np.allclose(recorded_mean, eog_mean, atol=1.0e-12, rtol=1.0e-12)
        or not np.allclose(recorded_scale, eog_scale, atol=1.0e-12, rtol=1.0e-12)
    ):
        raise ValueError("P0 transfer EOG standardization differs from this source block1")
    standardized_eog = (eog_windows.values - eog_mean[None, :, :]) / eog_scale[None, :, :]
    count = low_indices.size
    targets = np.empty((count, eeg.shape[0], eeg_windows.samples_per_window), dtype=np.float64)
    observed = np.empty_like(targets)
    target_origins: list[TrialWindowOrigin] = []
    artifact_origins: list[TrialWindowOrigin] = []
    for position, target_index in enumerate(low_indices):
        source_index = int(artifact_indices[position % artifact_indices.size])
        if eeg_windows.origins[int(target_index)] == eeg_windows.origins[source_index]:
            # The class thresholds make this impossible for valid release labels,
            # but keep the no-self-pair rule explicit and fail-closed.
            alternatives = [
                int(value)
                for value in artifact_indices
                if eeg_windows.origins[int(value)] != eeg_windows.origins[int(target_index)]
            ]
            if not alternatives:
                raise ValueError("no distinct artifact source is available")
            source_index = alternatives[position % len(alternatives)]
        target = eeg_windows.values[int(target_index)]
        reference = standardized_eog[source_index] - np.asarray(
            transfer.eog_mean, dtype=np.float64
        )
        contamination = np.asarray(transfer.transfer_matrix, dtype=np.float64) @ reference
        targets[position] = target
        observed[position] = target + contamination
        target_origins.append(eeg_windows.origins[int(target_index)])
        artifact_origins.append(eeg_windows.origins[source_index])
    projectors = np.repeat(projector[None, :, :], count, axis=0)
    attenuation = eeg_only_frame_attenuation(
        observed,
        projectors,
        projected_energy_scale,
    )
    return WeakSupervisionBundle(
        observed=np.ascontiguousarray(observed.astype(np.float32)),
        weak_target=np.ascontiguousarray(targets.astype(np.float32)),
        projector=np.ascontiguousarray(projectors.astype(np.float32)),
        attenuation=np.ascontiguousarray(attenuation.astype(np.float32)),
        valid_time_mask=np.ones((count, eeg_windows.samples_per_window), dtype=np.float32),
        recording_keys=tuple(recording_key for _ in range(count)),
        target_origins=tuple(target_origins),
        artifact_origins=tuple(artifact_origins),
    )


def matching_soft_proximal(
    observed: np.ndarray,
    projector: np.ndarray,
    attenuation: np.ndarray,
) -> np.ndarray:
    """Frozen deterministic arm ``y-(1-a^2) Pi y``."""

    value = np.asarray(observed, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("soft proximal expects channel-major EEG")
    projection = _checked_projector(value, projector)
    a = np.asarray(attenuation, dtype=np.float64).reshape(-1)
    if a.shape != (value.shape[1],) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("soft-proximal attenuation must be one [0,1] value per frame")
    return value - (1.0 - np.square(a))[None, :] * (projection @ value)


def assert_legal_inference_fields(field_names: Iterable[str]) -> tuple[str, ...]:
    """Accept only the frozen inference allowlist, not an extensible denylist."""

    fields = tuple(str(value) for value in field_names)
    illegal = sorted(set(fields) - LEGAL_INFERENCE_FIELDS)
    if illegal:
        evaluation_names = {
            value.lower().replace("_", "")
            for value in QUERY_EVALUATION_ONLY_FIELDS
        }
        reason = (
            "query evaluation-only fields entered inference"
            if any(
                field.lower().replace("_", "") in evaluation_names
                or "eog" in field.lower()
                or "artifact" in field.lower()
                or "outcome" in field.lower()
                for field in illegal
            )
            else "inference fields are outside the frozen allowlist"
        )
        raise ValueError(reason + ": " + ", ".join(illegal))
    return fields


@dataclass(frozen=True)
class FrozenArmOutputs:
    recording_key: str
    outputs: Mapping[str, np.ndarray]
    query_evaluation_fields_opened: bool = False

    def __post_init__(self) -> None:
        if self.query_evaluation_fields_opened:
            raise ValueError("query annotations were opened before arm outputs froze")
        if not self.recording_key or not self.outputs:
            raise ValueError("frozen output set must be named and non-empty")


def freeze_query_arm_outputs(
    outputs: Mapping[str, np.ndarray],
    *,
    expected_arm_ids: Sequence[str] = REPORTED_ARM_IDS,
    recording_key: str,
) -> FrozenArmOutputs:
    """Copy and make every output read-only before annotations may be reopened."""

    expected = tuple(str(value) for value in expected_arm_ids)
    if set(outputs) != set(expected) or len(outputs) != len(expected):
        raise ValueError("all and only frozen arm outputs are required before scoring")
    frozen: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for method_id in expected:
        value = np.array(outputs[method_id], dtype=np.float64, copy=True, order="C")
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"arm output is invalid: {method_id}")
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise ValueError("arm outputs are not aligned")
        value.setflags(write=False)
        frozen[method_id] = value
    return FrozenArmOutputs(
        recording_key=recording_key,
        outputs=MappingProxyType(frozen),
        query_evaluation_fields_opened=False,
    )


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.ndim != 1 or numeric.size < 1 or not np.isfinite(numeric).all():
        return float("nan"), float("nan")
    if numeric.size == 1:
        return float(numeric[0]), float(numeric[0])
    rng = np.random.default_rng(seed)
    index = rng.integers(0, numeric.size, size=(replicates, numeric.size))
    means = numeric[index].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _paired_metric_summary(
    paired: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary: dict[str, Any] = {}
    deltas: dict[str, np.ndarray] = {}
    for metric_index, metric in enumerate(_PAIRED_METRICS):
        values: list[float] = []
        for conditional, deterministic in paired:
            left = _finite_float(conditional.get(metric))
            right = _finite_float(deterministic.get(metric))
            if left is not None and right is not None:
                values.append(left - right)
        numeric = np.asarray(values, dtype=np.float64)
        low, high = _bootstrap_mean_ci(
            values,
            replicates=replicates,
            seed=seed + metric_index,
            confidence=confidence,
        )
        higher = metric in _HIGHER_IS_BETTER
        summary[metric] = {
            "direction": "higher" if higher else "lower",
            "paired_count": int(numeric.size),
            "mean_conditional_minus_unet": float(np.mean(numeric)) if numeric.size else None,
            "median_conditional_minus_unet": float(np.median(numeric)) if numeric.size else None,
            "descriptive_bootstrap_mean_ci95": [low, high] if numeric.size else [None, None],
            "conditional_win_count": int(np.sum(numeric > 0 if higher else numeric < 0)),
        }
        deltas[metric] = numeric
    return summary, deltas


def _training_endpoint_audit(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    expected_fold_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], bool]:
    expected = {
        (str(fold_id), method_id)
        for fold_id in expected_fold_ids
        for method_id in PRIMARY_METHOD_IDS
    }
    frozen: list[dict[str, Any]] = []
    observed: set[tuple[str, str]] = set()
    valid = True
    for raw in endpoints:
        row = {
            "fold_id": str(raw.get("fold_id", "")),
            "method_id": str(raw.get("method_id", "")),
            "status": str(raw.get("status", "")),
            "successful_optimizer_updates": int(raw.get("successful_optimizer_updates", -1)),
            "minibatch_sequence_updates": int(
                raw.get("minibatch_sequence_updates", -1)
            ),
            "minibatch_sequence_verified": raw.get(
                "minibatch_sequence_verified"
            )
            is True,
        }
        key = (row["fold_id"], row["method_id"])
        if key in observed:
            raise ValueError("duplicate fold/arm training endpoint")
        observed.add(key)
        valid &= (
            row["status"] == "success_fixed_6000_update_endpoint"
            and row["successful_optimizer_updates"] == FIXED_SUCCESSFUL_UPDATES
            and row["minibatch_sequence_updates"] == FIXED_SUCCESSFUL_UPDATES
            and row["minibatch_sequence_verified"] is True
        )
        frozen.append(row)
    valid &= observed == expected
    return sorted(frozen, key=lambda row: (row["fold_id"], row["method_id"])), bool(valid)


def _successful_method_performance(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize scientific values from success rows only."""

    result: dict[str, Any] = {}
    for method_id in REPORTED_ARM_IDS:
        successful = [
            row
            for row in rows
            if str(row.get("method_id")) == method_id
            and str(row.get("status", "")).startswith("success")
        ]
        metric_summary: dict[str, Any] = {}
        for metric in _METHOD_PERFORMANCE_METRICS:
            values: list[float] = []
            for row in successful:
                value = _finite_float(row.get(metric))
                if value is not None:
                    values.append(value)
            metric_summary[metric] = {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
        result[method_id] = {
            "success_count": len(successful),
            "metrics": metric_summary,
        }
    return result


def _paired_resource_summary(
    paired: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in _RESOURCE_FIELDS:
        deltas: list[float] = []
        conditional_values: list[float] = []
        deterministic_values: list[float] = []
        for conditional, deterministic in paired:
            left = _finite_float(conditional.get(field))
            right = _finite_float(deterministic.get(field))
            if left is not None and right is not None:
                conditional_values.append(left)
                deterministic_values.append(right)
                deltas.append(left - right)
        result[field] = {
            "paired_count": len(deltas),
            "conditional_mean": (
                float(np.mean(conditional_values)) if conditional_values else None
            ),
            "deterministic_unet_mean": (
                float(np.mean(deterministic_values))
                if deterministic_values
                else None
            ),
            "mean_conditional_minus_unet": (
                float(np.mean(deltas)) if deltas else None
            ),
            "median_conditional_minus_unet": (
                float(np.median(deltas)) if deltas else None
            ),
        }
    return result


def aggregate_sgeyesub_diffusion_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    fold_training_endpoints: Sequence[Mapping[str, Any]],
    partition: str | None = None,
) -> dict[str, Any]:
    """Aggregate participant-stem rows without treating windows/seeds as units."""

    validate_sgeyesub_diffusion_config(config)
    if partition is None:
        partitions = {str(row.get("partition", "")) for row in rows}
        if len(partitions) != 1:
            raise ValueError("aggregate rows must contain one explicit partition")
        partition = next(iter(partitions))
    if partition not in {"development", "evaluation"}:
        raise ValueError("partition must be development or evaluation")
    if any(str(row.get("partition")) != partition for row in rows):
        raise ValueError("aggregate rows cross development/evaluation partitions")
    split = _mapping(config, "split")
    fold_entries = split[f"{partition}_folds"]
    expected_fold_ids = tuple(str(item["fold_id"]) for item in fold_entries)
    expected_recording_keys = tuple(
        f"{item['study']}/{stem}"
        for item in fold_entries
        for stem in item["heldout_stems"]
    )
    expected_fold_by_recording_key = {
        f"{item['study']}/{stem}": str(item["fold_id"])
        for item in fold_entries
        for stem in item["heldout_stems"]
    }
    expected_fold_count = 10 if partition == "development" else 15
    completed_fold_ids = tuple(
        sorted(
            {
                str(row.get("fold_id"))
                for row in rows
                if str(row.get("status", "")).startswith("success")
            }
        )
    )
    endpoints, endpoints_valid = _training_endpoint_audit(
        fold_training_endpoints,
        expected_fold_ids=expected_fold_ids,
    )

    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        recording_key = str(row.get("recording_key", ""))
        method_id = str(row.get("method_id", ""))
        if not recording_key or not method_id:
            raise ValueError("metric row lacks recording_key or method_id")
        if method_id not in REPORTED_ARM_IDS:
            raise ValueError(f"metric row uses an unregistered method: {method_id}")
        if recording_key == PREBLOCKED_RECORDING_KEY:
            if partition != "evaluation" or not str(
                row.get("status", "")
            ).startswith(("blocked", "ineligible")):
                raise ValueError("preblocked singleton cannot contribute a success row")
        else:
            if recording_key not in expected_fold_by_recording_key:
                raise ValueError("metric row is outside the frozen participant matrix")
            if str(row.get("fold_id", "")) != expected_fold_by_recording_key[
                recording_key
            ]:
                raise ValueError("metric row recording_key/fold_id assignment changed")
            if str(row.get("study", "")) != recording_key.split("/", 1)[0]:
                raise ValueError("metric row recording_key/study assignment changed")
        key = (recording_key, method_id)
        if key in indexed:
            raise ValueError("duplicate participant-stem/method metric row")
        indexed[key] = row
    recording_keys = sorted({key[0] for key in indexed})
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    paired_keys: list[str] = []
    for recording_key in recording_keys:
        conditional = indexed.get((recording_key, CONDITIONAL_METHOD_ID))
        deterministic = indexed.get((recording_key, DETERMINISTIC_METHOD_ID))
        if (
            conditional is not None
            and deterministic is not None
            and str(conditional.get("status", "")).startswith("success")
            and str(deterministic.get("status", "")).startswith("success")
        ):
            paired.append((conditional, deterministic))
            paired_keys.append(recording_key)

    aggregation = _mapping(_mapping(config, "evaluation"), "aggregation")
    replicates = int(aggregation["descriptive_participant_stem_bootstrap_replicates"])
    seed = int(aggregation["bootstrap_seed"])
    confidence = float(aggregation["bootstrap_confidence_interval"])
    paired_summary, deltas = _paired_metric_summary(
        paired,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )
    by_study: dict[str, Any] = {}
    for study in (SGEYESUB_DEVELOPMENT_STUDIES if partition == "development" else SGEYESUB_EVALUATION_STUDIES):
        if len(paired_keys) != len(paired):
            raise AssertionError("paired recording keys and rows diverged")
        subset = [
            pair
            for key, pair in zip(paired_keys, paired)
            if key.startswith(f"{study}/")
        ]
        study_summary, _ = _paired_metric_summary(
            subset,
            replicates=replicates,
            seed=seed + 100 + len(by_study),
            confidence=confidence,
        )
        by_study[study] = {
            "paired_participant_stem_count": len(subset),
            "conditional_minus_unet": study_summary,
        }

    method_coverage: dict[str, Any] = {}
    for method_id in REPORTED_ARM_IDS:
        method_rows = [row for row in rows if str(row.get("method_id")) == method_id]
        method_coverage[method_id] = {
            "requested_count": len(method_rows),
            "success_count": sum(str(row.get("status", "")).startswith("success") for row in method_rows),
            "failed_count": sum(str(row.get("status", "")).startswith("failed") for row in method_rows),
            "blocked_or_ineligible_count": sum(
                str(row.get("status", "")).startswith(("blocked", "ineligible"))
                for row in method_rows
            ),
            "fallback_count": sum(bool(row.get("fallback_used", False)) for row in method_rows),
        }
    method_performance = _successful_method_performance(rows)
    resource_comparison = _paired_resource_summary(paired)

    compatible_rows = [
        row
        for row in rows
        if str(row.get("recording_key", "")) != PREBLOCKED_RECORDING_KEY
    ]
    successful_performance_rows = [
        row
        for row in compatible_rows
        if str(row.get("status", "")).startswith("success")
    ]
    query_open_after_freeze = all(
        row.get("query_evaluation_fields_opened_after_all_arm_outputs_frozen") is True
        for row in successful_performance_rows
    ) if successful_performance_rows else False
    query_never_used = all(
        row.get("query_evaluation_fields_used_for_fit_selection_or_inference") is False
        for row in compatible_rows
    ) if compatible_rows else False
    compatible_denominator = 15 if partition == "development" else EVALUATION_COMPATIBLE_DENOMINATOR
    availability_denominator = 15 if partition == "development" else EVALUATION_AVAILABILITY_DENOMINATOR
    conditional_success = sum(
        str(indexed.get((key, CONDITIONAL_METHOD_ID), {}).get("status", "")).startswith("success")
        for key in recording_keys
        if key != PREBLOCKED_RECORDING_KEY
    )
    conditional_failures = compatible_denominator - conditional_success

    compatible_matrix_complete = all(
        (recording_key, method_id) in indexed
        for recording_key in expected_recording_keys
        for method_id in REPORTED_ARM_IDS
    )
    preblocked_matrix_complete = partition != "evaluation" or all(
        (PREBLOCKED_RECORDING_KEY, method_id) in indexed
        and str(
            indexed[(PREBLOCKED_RECORDING_KEY, method_id)].get("status", "")
        ).startswith(("blocked", "ineligible"))
        for method_id in REPORTED_ARM_IDS
    )
    complete = (
        set(completed_fold_ids) == set(expected_fold_ids)
        and len(completed_fold_ids) == expected_fold_count
        and endpoints_valid
        and compatible_matrix_complete
        and preblocked_matrix_complete
    )

    natural_status = NATURAL_DECISION_INCONCLUSIVE
    decision_details: dict[str, Any] = {
        "status": natural_status,
        "threshold_source_config": str(CONFIG_PATH),
        "threshold_section": "prospective_exploratory_thresholds",
        "thresholds_frozen_before_evaluation_outputs": True,
        "evaluation_outcomes_used_to_select_or_change_thresholds": False,
        "paired_primary_success_count": len(paired),
        "conditional_diffusion_failure_count": conditional_failures,
    }
    if partition == "evaluation":
        thresholds = _mapping(config, "prospective_exploratory_thresholds")
        benefit = _mapping(thresholds, "primary_benefit")
        safety = _mapping(thresholds, "safety")
        adequate_coverage = (
            len(paired) >= int(thresholds["minimum_paired_success_count_of_43"])
            and conditional_failures <= int(safety["maximum_conditional_diffusion_failure_count_of_43"])
        )
        coherence = deltas["eog_coherence_reduction"]
        attenuation_delta = deltas["matching_projector_attenuation_db"]
        joint_values: list[bool] = []
        for conditional_row, deterministic_row in paired:
            coherence_left = _finite_float(
                conditional_row.get("eog_coherence_reduction")
            )
            coherence_right = _finite_float(
                deterministic_row.get("eog_coherence_reduction")
            )
            attenuation_left = _finite_float(
                conditional_row.get("matching_projector_attenuation_db")
            )
            attenuation_right = _finite_float(
                deterministic_row.get("matching_projector_attenuation_db")
            )
            if None not in (
                coherence_left,
                coherence_right,
                attenuation_left,
                attenuation_right,
            ):
                joint_values.append(
                    bool(
                        coherence_left > coherence_right
                        and attenuation_left > attenuation_right
                    )
                )
        joint = np.asarray(joint_values, dtype=bool)
        joint_fraction = float(np.mean(joint)) if joint.size else float("nan")
        joint_ci = _bootstrap_mean_ci(
            joint.astype(float).tolist(),
            replicates=replicates,
            seed=seed + 500,
            confidence=confidence,
        )
        primary_metrics_complete = bool(
            len(paired) > 0
            and coherence.size == len(paired)
            and attenuation_delta.size == len(paired)
            and joint.size == len(paired)
        )
        benefit_point_pass = bool(
            primary_metrics_complete
            and np.mean(coherence) > 0
            and np.mean(attenuation_delta) > 0
            and joint_fraction >= float(benefit["minimum_participant_joint_primary_win_fraction"])
        )
        descriptive_benefit_intervals_support_point_result = bool(
            paired_summary["eog_coherence_reduction"]["descriptive_bootstrap_mean_ci95"][0] is not None
            and paired_summary["eog_coherence_reduction"]["descriptive_bootstrap_mean_ci95"][0] > 0
            and paired_summary["matching_projector_attenuation_db"]["descriptive_bootstrap_mean_ci95"][0] > 0
            and joint_ci[0] >= float(benefit["minimum_participant_joint_primary_win_fraction"])
        )
        safety_rules = {
            "nonartifact_observation_preservation": (
                "lower",
                float(safety["minimum_mean_nonartifact_preservation_delta"]),
            ),
            "reference_free_psd_distortion": (
                "upper",
                float(safety["maximum_mean_PSD_distortion_delta"]),
            ),
            "reference_free_covariance_distortion": (
                "upper",
                float(safety["maximum_mean_covariance_distortion_delta"]),
            ),
            "condition_erp_observation_relative_preservation": (
                "lower",
                float(safety["minimum_mean_ERP_proxy_delta"]),
            ),
        }
        safety_point_pass = True
        safety_metrics_complete = True
        descriptive_safety_intervals_support_point_result = True
        descriptive_safety_interval_ambiguity = False
        for metric, (bound, threshold) in safety_rules.items():
            values = deltas[metric]
            interval = paired_summary[metric]["descriptive_bootstrap_mean_ci95"]
            if values.size != len(paired) or interval[0] is None:
                safety_point_pass = False
                safety_metrics_complete = False
                descriptive_safety_intervals_support_point_result = False
                descriptive_safety_interval_ambiguity = True
                continue
            mean = float(np.mean(values))
            point_ok = mean >= threshold if bound == "lower" else mean <= threshold
            interval_ok = interval[0] >= threshold if bound == "lower" else interval[1] <= threshold
            interval_fail = interval[1] < threshold if bound == "lower" else interval[0] > threshold
            safety_point_pass &= point_ok
            descriptive_safety_intervals_support_point_result &= interval_ok
            descriptive_safety_interval_ambiguity |= not interval_ok and not interval_fail
        if (
            complete
            and adequate_coverage
            and primary_metrics_complete
            and safety_metrics_complete
            and query_open_after_freeze
            and query_never_used
        ):
            if safety_point_pass and benefit_point_pass:
                natural_status = NATURAL_DECISION_PASS
            elif safety_point_pass:
                natural_status = NATURAL_DECISION_FAIL
            else:
                natural_status = NATURAL_DECISION_INCONCLUSIVE
        decision_details.update(
            {
                "status": natural_status,
                "adequate_coverage": adequate_coverage,
                "aggregate_complete": complete,
                "primary_metrics_complete_for_all_successful_pairs": primary_metrics_complete,
                "safety_metrics_complete_for_all_successful_pairs": safety_metrics_complete,
                "joint_primary_win_fraction": joint_fraction if np.isfinite(joint_fraction) else None,
                "joint_primary_win_fraction_ci95": [
                    value if np.isfinite(value) else None for value in joint_ci
                ],
                "primary_benefit_point_pass": benefit_point_pass,
                "descriptive_benefit_intervals_support_point_result": descriptive_benefit_intervals_support_point_result,
                "safety_point_pass": bool(safety_point_pass),
                "descriptive_safety_intervals_support_point_result": bool(
                    descriptive_safety_intervals_support_point_result
                ),
                "descriptive_safety_interval_ambiguity": bool(
                    descriptive_safety_interval_ambiguity
                ),
                "bootstrap_intervals_used_as_decision_thresholds": False,
            }
        )
    status = (
        f"completed_{partition}_aggregate"
        if complete
        else f"incomplete_{partition}_aggregate"
    )
    return {
        "protocol_id": PROTOCOL_ID,
        "status": status,
        "partition": partition,
        f"{partition}_fold_count": expected_fold_count,
        "evaluation_fold_count": 15 if partition == "evaluation" else None,
        "completed_fold_ids": list(completed_fold_ids),
        "availability_denominator": availability_denominator,
        "compatible_performance_denominator": compatible_denominator,
        "preblocked_count": 1 if partition == "evaluation" else 0,
        "preblocked_recording_key": PREBLOCKED_RECORDING_KEY if partition == "evaluation" else None,
        "paired_primary_success_count": len(paired),
        "paired_recording_keys": paired_keys,
        "method_coverage": method_coverage,
        "method_performance_success_rows_only": method_performance,
        "conditional_minus_unet": paired_summary,
        "conditional_minus_unet_resources": resource_comparison,
        "by_study": by_study,
        "matched_comparison_audit": {
            "same_information_inputs": True,
            "same_outer_training_stems": True,
            "same_weak_supervision_pairs_and_order": True,
            "same_windowing_channels_normalization_and_operator_conditioning": True,
            "successful_optimizer_updates_target": FIXED_SUCCESSFUL_UPDATES,
            "fold_arm_training_endpoints": endpoints,
            "all_fold_arm_training_endpoints_valid": endpoints_valid,
            "minibatch_sequence_verified_for_all_fold_arms": endpoints_valid,
        },
        "information_boundary_audit": {
            "all_arm_outputs_frozen_before_query_evaluation_fields_opened": query_open_after_freeze,
            "query_evaluation_fields_used_for_fit_selection_or_inference": not query_never_used,
        },
        "development_outcomes_read_for_selection": False,
        "claim_boundary": {
            "clean_target_available": False,
            "clean_waveform_recovery_claim": False,
            "weak_target_semantics": "low_artifact_observed_EEG_not_clean_truth",
            "formal_g1_or_g3_evidence": False,
            "diffusion_family_wide_claim_allowed": False,
        },
        "natural_decision": decision_details,
    }


def write_sgeyesub_diffusion_aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    fold_training_endpoints: Sequence[Mapping[str, Any]],
    partition: str,
) -> dict[str, Path]:
    """Write the small frozen config, participant metrics, and aggregate summary."""

    summary = aggregate_sgeyesub_diffusion_metrics(
        rows,
        config=config,
        fold_training_endpoints=fold_training_endpoints,
        partition=partition,
    )
    outputs = _mapping(config, "outputs")
    root = Path(str(outputs[f"{partition}_root"]))
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "resolved_config.yaml"
    metrics_path = root / "metrics.csv"
    summary_path = root / "result_summary.json"
    config_path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "resolved_config": config_path,
        "metrics": metrics_path,
        "result_summary": summary_path,
    }


__all__ = [
    "CONDITIONAL_METHOD_ID",
    "CONFIG_PATH",
    "DETERMINISTIC_METHOD_ID",
    "FrozenArmOutputs",
    "FrozenSgeyesubFold",
    "FrozenSgeyesubSplit",
    "FullBlockP0Fit",
    "LEGAL_INFERENCE_FIELDS",
    "OuterTrainingPopulationP0Fit",
    "PROTOCOL_ID",
    "REPORTED_ARM_IDS",
    "SgeyesubFoldNormalizer",
    "TrialWindowBatch",
    "TrialWindowOrigin",
    "WeakSupervisionBundle",
    "aggregate_sgeyesub_diffusion_metrics",
    "assert_legal_inference_fields",
    "build_frozen_sgeyesub_folds",
    "build_within_stem_weak_pairs",
    "eeg_only_frame_attenuation",
    "fit_full_block1_p0",
    "fit_outer_training_normalizer",
    "fit_outer_training_population_p0",
    "fit_outer_training_projected_energy_scale",
    "freeze_query_arm_outputs",
    "matching_soft_proximal",
    "sgeyesub_p0_config",
    "trial_local_nonoverlap_windows",
    "validate_sgeyesub_diffusion_config",
    "write_sgeyesub_diffusion_aggregate",
]
