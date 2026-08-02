"""Frozen v2 decision for protocol-scoped diffusion incremental value.

The evaluator consumes only two small terminal aggregate summaries: the v1
local-evidence decision and the prospective SGEYESUB natural-EEG comparison.
It does not read model predictions, EEG, or per-participant outcomes and does
not choose a threshold.  Invalid or incomplete evidence fails closed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_ID = "cgdr_diffusion_incremental_value_decision_v2"
V1_PROTOCOL_ID = "cgdr_diffusion_incremental_value_decision_v1"
NATURAL_SGE_PROTOCOL_ID = "sgeyesub_natural_eeg_diffusion_incremental_v1"
CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
SOURCE_CONFIG_ROOT = Path(__file__).resolve().parents[3]

CURRENT_M2_STATUS = "current_M2_no_incremental_value"
SUPPORTED = "conditional_diffusion_supported"
NO_DETECTABLE = "diffusion_no_detectable_incremental_value_under_tested_protocols"
INCONCLUSIVE = "inconclusive"
TOP_LEVEL_CONCLUSIONS = frozenset({SUPPORTED, NO_DETECTABLE, INCONCLUSIVE})
DIFFUSION_FAMILY_WIDE_STATUS = "not_tested"

EEGDFUS_MEETS = "meets_frozen_stability"
LOCAL_NO_DETECTABLE = "no_detectable_stability"
LOCAL_INCONCLUSIVE = "inconclusive"
KLADOS_CONFIGURATIONS = (
    "M1",
    "M4",
    "operator_conditioned_diffusion_DDIM100",
)
LEARNED_SGE_ARMS = (
    "operator_conditioned_conditional_diffusion_DDIM100",
    "task_matched_multichannel_deterministic_UNet",
)
EXPECTED_SGE_FOLDS = (
    "study02_fold01",
    "study02_fold02",
    "study02_fold03",
    "study02_fold04",
    "study02_fold05",
    "study04_fold01",
    "study04_fold02",
    "study04_fold03",
    "study04_fold04",
    "study04_fold05",
    "study05_fold01",
    "study05_fold02",
    "study05_fold03",
    "study05_fold04",
    "study05_fold05",
)
EXPECTED_SGE_STUDIES = ("study02", "study04", "study05")
EXPECTED_SGE_STUDY_COMPATIBLE_COUNTS = {
    "study02": 15,
    "study04": 15,
    "study05": 13,
}
NATURAL_PASS = (
    "natural_SGE_conditional_diffusion_supported_over_matched_UNet_under_"
    "frozen_weak_supervision_protocol"
)
NATURAL_FAIL = "no_detectable_incremental_value_for_tested_SGE_conditional_protocol"
NATURAL_INCONCLUSIVE = "inconclusive"

MINIMUM_NATURAL_PAIRED_SUCCESS = 39
MAXIMUM_NATURAL_DIFFUSION_FAILURES = 4
MINIMUM_JOINT_PRIMARY_WIN_FRACTION = 0.60
MINIMUM_NONARTIFACT_PRESERVATION_DELTA = -0.02
MAXIMUM_PSD_DISTORTION_DELTA = 0.05
MAXIMUM_COVARIANCE_DISTORTION_DELTA = 0.05
MINIMUM_ERP_PRESERVATION_DELTA = -0.02


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _sequence(parent: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = parent.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"missing sequence: {key}")
    return value


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an exact integer")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be an exact boolean")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _paired_delta_metric(
    paired_summary: Mapping[str, Any],
    metric: str,
    *,
    overall_paired_count: int,
) -> tuple[float | None, bool]:
    """Read one aggregate delta and independently determine completeness."""

    row = _mapping(paired_summary, metric)
    count = _exact_int(row.get("paired_count"), f"{metric} paired_count")
    if count < 0 or count > overall_paired_count:
        raise ValueError(f"{metric} paired_count is outside the paired matrix")
    raw_mean = row.get("mean_conditional_minus_unet")
    if count == 0:
        if raw_mean is not None:
            raise ValueError(f"{metric} has a mean without paired observations")
        return None, False
    mean = _finite_float(raw_mean, f"{metric} mean_conditional_minus_unet")
    return mean, count == overall_paired_count


def _expect_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if dict(actual) != dict(expected):
        raise ValueError(f"{label} changed")


def _source_sge_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read and verify the actual frozen SGE threshold/split source."""

    expected = _mapping(config, "expected_natural_sge")
    relative_path = str(expected.get("threshold_source_config", ""))
    if relative_path != "configs/cgdr/sgeyesub_diffusion_incremental.yaml":
        raise ValueError("natural SGE threshold source path changed")
    source_path = SOURCE_CONFIG_ROOT / relative_path
    try:
        loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read natural SGE threshold source: {source_path}") from error
    if not isinstance(loaded, Mapping):
        raise ValueError("natural SGE threshold source root is not a mapping")
    if loaded.get("protocol_id") != NATURAL_SGE_PROTOCOL_ID:
        raise ValueError("natural SGE threshold source protocol changed")

    thresholds = _mapping(loaded, str(expected.get("threshold_section", "")))
    if thresholds.get("comparison_direction") != (
        "conditional_diffusion_minus_matched_deterministic_UNet"
    ):
        raise ValueError("natural SGE comparison direction changed at source")
    if thresholds.get("threshold_scope") != (
        "pooled_43_compatible_evaluation_stems_equal_participant_weight"
    ):
        raise ValueError("natural SGE threshold scope changed at source")
    if thresholds.get(
        "by_study_results_required_but_not_additional_pass_fail_thresholds"
    ) is not True:
        raise ValueError("natural SGE by-study reporting contract changed at source")
    if _finite_float(
        thresholds.get("minimum_paired_success_fraction"),
        "source minimum paired success fraction",
    ) != 0.90:
        raise ValueError("natural SGE paired success fraction changed at source")
    if _exact_int(
        thresholds.get("minimum_paired_success_count_of_43"),
        "source minimum paired success count",
    ) != MINIMUM_NATURAL_PAIRED_SUCCESS:
        raise ValueError("natural SGE paired success count changed at source")

    benefit = _mapping(thresholds, "primary_benefit")
    _expect_mapping(
        benefit,
        {
            "mean_EOG_coherence_reduction_delta": "greater_than_0",
            "mean_matching_projector_attenuation_db_delta": "greater_than_0",
            "participant_joint_primary_win_definition": (
                "both_primary_deltas_greater_than_0_for_the_stem"
            ),
            "minimum_participant_joint_primary_win_fraction": (
                MINIMUM_JOINT_PRIMARY_WIN_FRACTION
            ),
        },
        "natural SGE source primary-benefit thresholds",
    )
    safety = _mapping(thresholds, "safety")
    _expect_mapping(
        safety,
        {
            "minimum_mean_nonartifact_preservation_delta": (
                MINIMUM_NONARTIFACT_PRESERVATION_DELTA
            ),
            "maximum_mean_PSD_distortion_delta": MAXIMUM_PSD_DISTORTION_DELTA,
            "maximum_mean_covariance_distortion_delta": (
                MAXIMUM_COVARIANCE_DISTORTION_DELTA
            ),
            "minimum_mean_ERP_proxy_delta": MINIMUM_ERP_PRESERVATION_DELTA,
            "maximum_conditional_diffusion_failure_fraction_of_43": 0.10,
            "maximum_conditional_diffusion_failure_count_of_43": (
                MAXIMUM_NATURAL_DIFFUSION_FAILURES
            ),
        },
        "natural SGE source safety thresholds",
    )
    if thresholds.get("all_conditions_required_for_protocol_scoped_positive") is not True:
        raise ValueError("natural SGE all-conditions rule changed at source")
    if thresholds.get("interval_crossing_zero_or_threshold_action") != (
        "report_descriptive_uncertainty_without_posthoc_rule_change"
    ):
        raise ValueError("natural SGE descriptive-interval role changed at source")
    _expect_mapping(
        _mapping(thresholds, "outcome_labels"),
        {
            "pass": NATURAL_PASS,
            "fail": NATURAL_FAIL,
            "insufficient_coverage_or_mixed_safety": NATURAL_INCONCLUSIVE,
        },
        "natural SGE source outcome labels",
    )

    split = _mapping(loaded, "split")
    folds = _sequence(split, "evaluation_folds")
    fold_ids: list[str] = []
    recording_keys: list[str] = []
    by_study: dict[str, list[str]] = {study: [] for study in EXPECTED_SGE_STUDIES}
    for raw_fold in folds:
        if not isinstance(raw_fold, Mapping):
            raise ValueError("natural SGE source evaluation fold is not a mapping")
        fold_id = str(raw_fold.get("fold_id", ""))
        study = str(raw_fold.get("study", ""))
        if study not in by_study:
            raise ValueError("natural SGE source evaluation study changed")
        fold_ids.append(fold_id)
        for raw_stem in _sequence(raw_fold, "heldout_stems"):
            stem = str(raw_stem)
            key = f"{study}/{stem}"
            recording_keys.append(key)
            by_study[study].append(key)
    if len(fold_ids) != len(set(fold_ids)) or set(fold_ids) != set(EXPECTED_SGE_FOLDS):
        raise ValueError("natural SGE source evaluation folds changed")
    if len(recording_keys) != 43 or len(set(recording_keys)) != 43:
        raise ValueError("natural SGE source must define exactly 43 unique compatible keys")
    for study, expected_count in EXPECTED_SGE_STUDY_COMPATIBLE_COUNTS.items():
        if len(by_study[study]) != expected_count:
            raise ValueError(f"natural SGE source compatible count changed: {study}")
    return {
        "source_path": str(source_path),
        "recording_keys": tuple(recording_keys),
        "recording_keys_by_study": {
            study: tuple(values) for study, values in by_study.items()
        },
    }


def _validated_method_coverage(
    method_coverage: Mapping[str, Any], method_id: str, label: str
) -> dict[str, int]:
    coverage = _mapping(method_coverage, method_id)
    counts = {
        name: _exact_int(coverage.get(name), f"{label} {name}")
        for name in (
            "requested_count",
            "success_count",
            "failed_count",
            "blocked_or_ineligible_count",
            "fallback_count",
        )
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError(f"natural SGE {label} coverage contains a negative count")
    if counts["requested_count"] != 44:
        raise ValueError(f"natural SGE {label} denominator changed")
    if sum(
        counts[name]
        for name in (
            "success_count",
            "failed_count",
            "blocked_or_ineligible_count",
            "fallback_count",
        )
    ) != counts["requested_count"]:
        raise ValueError(f"natural SGE {label} coverage categories do not sum to 44")
    if counts["blocked_or_ineligible_count"] != 1:
        raise ValueError(f"natural SGE {label} must retain exactly one preblocked stem")
    non_success = counts["failed_count"] + counts["fallback_count"]
    if counts["success_count"] + non_success != 43:
        raise ValueError(f"natural SGE {label} compatible coverage does not sum to 43")
    counts["compatible_non_success_count"] = non_success
    return counts


def validate_decision_v2_config(config: Mapping[str, Any]) -> None:
    """Reject any mutation of the prospective v2 decision contract."""

    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID}")
    if _exact_int(config.get("harness_level"), "harness_level") != 1:
        raise ValueError("v2 decision requires HARNESS_LEVEL=1")
    if config.get("frozen_before_natural_sge_evaluation_outcomes") is not True:
        raise ValueError("v2 must be frozen before natural SGE evaluation outcomes")
    if config.get("scientific_role") != (
        "protocol_scoped_top_level_decision_not_formal_G1_or_G3"
    ):
        raise ValueError("scientific role changed")

    _expect_mapping(
        _mapping(config, "artifacts"),
        {
            "v1_decision_summary": (
                "results/cgdr/diffusion_incremental_decision/result_summary.json"
            ),
            "natural_sge_evaluation_summary": (
                "results/cgdr/sgeyesub_diffusion_incremental/evaluation/"
                "result_summary.json"
            ),
            "output_root": "results/cgdr/diffusion_incremental_decision_v2",
        },
        "artifact paths",
    )
    _expect_mapping(
        _mapping(config, "retained_status"),
        {
            "current_M2": CURRENT_M2_STATUS,
            "formal_G1": "NOT_RUN_BLOCKED",
            "formal_G3": "NOT_RUN_BLOCKED",
        },
        "retained status",
    )

    expected_v1 = _mapping(config, "expected_v1")
    for key, expected in (
        ("protocol_id", V1_PROTOCOL_ID),
        ("terminal_status", "completed_fail_closed_decision"),
        ("conclusion", INCONCLUSIVE),
        ("natural_real_eeg_diffusion_comparator_status", "not_run"),
    ):
        if expected_v1.get(key) != expected:
            raise ValueError(f"expected v1 {key} changed")
    if expected_v1.get("top_level_diffusion_family_decision_eligible") is not False:
        raise ValueError("v1 top-level eligibility boundary changed")
    if tuple(_sequence(expected_v1, "eegdfus_allowed_local_outcomes")) != (
        EEGDFUS_MEETS,
        LOCAL_NO_DETECTABLE,
        LOCAL_INCONCLUSIVE,
    ):
        raise ValueError("allowed EEGDfus local outcomes changed")
    if tuple(_sequence(expected_v1, "required_klados_configuration_outcomes")) != (
        KLADOS_CONFIGURATIONS
    ):
        raise ValueError("required Klados configurations changed")
    if tuple(_sequence(expected_v1, "klados_allowed_local_outcomes")) != (
        EEGDFUS_MEETS,
        LOCAL_NO_DETECTABLE,
        LOCAL_INCONCLUSIVE,
    ):
        raise ValueError("allowed Klados local outcomes changed")

    natural = _mapping(config, "expected_natural_sge")
    for key, expected in (
        ("protocol_id", NATURAL_SGE_PROTOCOL_ID),
        ("terminal_status", "completed_evaluation_aggregate"),
        ("partition", "evaluation"),
        ("preblocked_recording_key", "study05/study05_p42"),
        (
            "threshold_source_config",
            "configs/cgdr/sgeyesub_diffusion_incremental.yaml",
        ),
        ("threshold_section", "prospective_exploratory_thresholds"),
        ("training_endpoint_status", "success_fixed_6000_update_endpoint"),
    ):
        if natural.get(key) != expected:
            raise ValueError(f"expected natural SGE {key} changed")
    for key, expected in (
        ("expected_fold_count", 15),
        ("availability_denominator", 44),
        ("compatible_performance_denominator", 43),
        ("preblocked_count", 1),
        ("successful_optimizer_updates_per_arm_per_fold", 6000),
    ):
        if _exact_int(natural.get(key), f"natural SGE {key}") != expected:
            raise ValueError(f"expected natural SGE {key} changed")
    if tuple(_sequence(natural, "expected_fold_ids")) != EXPECTED_SGE_FOLDS:
        raise ValueError("natural SGE fold identities changed")
    if tuple(_sequence(natural, "learned_arms")) != LEARNED_SGE_ARMS:
        raise ValueError("natural SGE learned arms changed")
    _expect_mapping(
        _mapping(natural, "natural_decision_labels"),
        {
            "pass": NATURAL_PASS,
            "fail": NATURAL_FAIL,
            "inconclusive": NATURAL_INCONCLUSIVE,
        },
        "natural SGE decision labels",
    )
    _expect_mapping(
        _mapping(natural, "information_boundary"),
        {
            "all_arm_outputs_frozen_before_query_evaluation_fields_opened": True,
            "query_evaluation_fields_used_for_fit_selection_or_inference": False,
        },
        "natural SGE information boundary",
    )
    _expect_mapping(
        _mapping(natural, "matched_comparison"),
        {
            "same_information_inputs": True,
            "same_outer_training_stems": True,
            "same_weak_supervision_pairs_and_order": True,
            "same_windowing_channels_normalization_and_operator_conditioning": True,
        },
        "natural SGE matched comparison",
    )
    _expect_mapping(
        _mapping(natural, "claim_boundary"),
        {
            "clean_target_available": False,
            "clean_waveform_recovery_claim": False,
            "weak_target_semantics": "low_artifact_observed_EEG_not_clean_truth",
        },
        "natural SGE claim boundary",
    )
    _source_sge_contract(config)

    rules = _mapping(config, "decision_rules")
    _expect_mapping(
        _mapping(rules, SUPPORTED),
        {
            "natural_sge_status": "pass",
            "eegdfus_local_outcome": EEGDFUS_MEETS,
            "klados_role": "exploratory_context_only_not_required",
            "claim_limit": "tested_SGE_natural_EEG_and_EEGDfus_protocols_only",
        },
        "positive decision rule",
    )
    _expect_mapping(
        _mapping(rules, NO_DETECTABLE),
        {
            "natural_sge_status": "fail",
            "eegdfus_local_outcome": LOCAL_NO_DETECTABLE,
            "required_klados_outcomes": {
                name: LOCAL_NO_DETECTABLE for name in KLADOS_CONFIGURATIONS
            },
            "claim_limit": (
                "tested_datasets_tasks_splits_objectives_and_configurations_only"
            ),
        },
        "negative decision rule",
    )
    if set(_sequence(_mapping(rules, INCONCLUSIVE), "applies_to")) != {
        "missing_or_invalid_input",
        "partial_natural_SGE_matrix",
        "natural_SGE_inconclusive",
        "EEGDfus_inconclusive",
        "Klados_inconclusive_for_negative_rule",
        "mixed_evidence_directions",
    }:
        raise ValueError("inconclusive cases changed")
    if set(_sequence(config, "forbidden_claims")) != {
        "diffusion_is_useless",
        "diffusion_sampler_has_no_value",
        "EEG_diffusion_is_disproved",
        "personalization_failed",
    }:
        raise ValueError("forbidden claims changed")
    _expect_mapping(
        _mapping(config, "reporting_semantics"),
        {
            "primary_status_field": "protocol_scoped_diffusion_status",
            "diffusion_family_wide_status": DIFFUSION_FAMILY_WIDE_STATUS,
            "legacy_status_field_retained_for_compatibility": (
                "diffusion_family_status"
            ),
            "natural_sge_rule_interpretation": (
                "frozen_point_direction_stability_rule_not_hypothesis_test"
            ),
            "bootstrap_role": "descriptive_only_not_decision_bearing",
        },
        "reporting semantics",
    )


def _validate_v1_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("protocol_id") != V1_PROTOCOL_ID:
        raise ValueError("v1 decision protocol mismatch")
    if summary.get("status") != "completed_fail_closed_decision":
        raise ValueError("v1 decision is not terminal-complete")
    if summary.get("conclusion") != INCONCLUSIVE:
        raise ValueError("v1 top-level conclusion was unexpectedly upgraded")
    if summary.get("natural_real_eeg_diffusion_comparator_status") != "not_run":
        raise ValueError("v1 natural real-EEG boundary changed")
    if summary.get("top_level_diffusion_family_decision_eligible") is not False:
        raise ValueError("v1 top-level eligibility changed")
    if summary.get("all_loaded_artifact_inputs_complete") is not True:
        raise ValueError("v1 underlying local evidence is incomplete")
    retained = _mapping(summary, "retained_status")
    if retained.get("current_M2") != CURRENT_M2_STATUS:
        raise ValueError("current M2 historical status changed")
    if summary.get("formal_G1_status") != "NOT_RUN_BLOCKED" or summary.get(
        "formal_G3_status"
    ) != "NOT_RUN_BLOCKED":
        raise ValueError("formal G1/G3 status changed")

    eegdfus = summary.get("eegdfus_local_outcome")
    if eegdfus not in {EEGDFUS_MEETS, LOCAL_NO_DETECTABLE, LOCAL_INCONCLUSIVE}:
        raise ValueError("invalid EEGDfus local outcome")
    tested = _mapping(summary, "tested_configuration_outcomes")
    klados: dict[str, str] = {}
    for name in KLADOS_CONFIGURATIONS:
        outcome = tested.get(name)
        if outcome not in {EEGDFUS_MEETS, LOCAL_NO_DETECTABLE, LOCAL_INCONCLUSIVE}:
            raise ValueError(f"invalid or missing Klados outcome: {name}")
        klados[name] = str(outcome)
    if tested.get("EEGDfus_conditional_diffusion") != eegdfus:
        raise ValueError("v1 EEGDfus local outcome is internally inconsistent")
    return {"eegdfus": str(eegdfus), "klados": klados}


def _validate_natural_sge_summary(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _mapping(config, "expected_natural_sge")
    source_contract = _source_sge_contract(config)
    if summary.get("protocol_id") != NATURAL_SGE_PROTOCOL_ID:
        raise ValueError("natural SGE protocol mismatch")
    if summary.get("status") != expected["terminal_status"]:
        raise ValueError("natural SGE aggregate is not terminal-complete")
    if summary.get("partition") != "evaluation":
        raise ValueError("natural SGE aggregate is not evaluation")
    if _exact_int(summary.get("evaluation_fold_count"), "evaluation_fold_count") != 15:
        raise ValueError("natural SGE must contain all 15 evaluation folds")
    completed_fold_ids = tuple(_sequence(summary, "completed_fold_ids"))
    if len(completed_fold_ids) != len(set(completed_fold_ids)) or set(
        completed_fold_ids
    ) != set(EXPECTED_SGE_FOLDS):
        raise ValueError("natural SGE completed fold identities are incomplete")
    if _exact_int(summary.get("availability_denominator"), "availability denominator") != 44:
        raise ValueError("natural SGE availability denominator must remain 44")
    if _exact_int(
        summary.get("compatible_performance_denominator"),
        "compatible performance denominator",
    ) != 43:
        raise ValueError("natural SGE compatible denominator must remain 43")
    if _exact_int(summary.get("preblocked_count"), "preblocked count") != 1:
        raise ValueError("natural SGE must retain the one preblocked stem")
    if summary.get("preblocked_recording_key") != "study05/study05_p42":
        raise ValueError("natural SGE preblocked stem changed")

    matched = _mapping(summary, "matched_comparison_audit")
    for field in (
        "same_information_inputs",
        "same_outer_training_stems",
        "same_weak_supervision_pairs_and_order",
        "same_windowing_channels_normalization_and_operator_conditioning",
    ):
        if matched.get(field) is not True:
            raise ValueError(f"natural SGE matched comparison failed: {field}")
    if _exact_int(
        matched.get("successful_optimizer_updates_target"),
        "successful optimizer update target",
    ) != 6000:
        raise ValueError("natural SGE learned-arm update target changed")
    endpoints = _sequence(matched, "fold_arm_training_endpoints")
    expected_pairs = {
        (fold_id, arm) for fold_id in EXPECTED_SGE_FOLDS for arm in LEARNED_SGE_ARMS
    }
    observed_pairs: set[tuple[str, str]] = set()
    for raw_row in endpoints:
        if not isinstance(raw_row, Mapping):
            raise ValueError("natural SGE training endpoint row is not a mapping")
        pair = (str(raw_row.get("fold_id", "")), str(raw_row.get("method_id", "")))
        if pair in observed_pairs:
            raise ValueError("duplicate natural SGE fold/arm training endpoint")
        observed_pairs.add(pair)
        if raw_row.get("status") != "success_fixed_6000_update_endpoint":
            raise ValueError(f"natural SGE training endpoint failed: {pair}")
        if _exact_int(
            raw_row.get("successful_optimizer_updates"),
            f"successful optimizer updates for {pair}",
        ) != 6000:
            raise ValueError(f"natural SGE learned arm did not reach 6000 updates: {pair}")
        if _exact_int(
            raw_row.get("minibatch_sequence_updates"),
            f"minibatch sequence updates for {pair}",
        ) != 6000:
            raise ValueError(f"natural SGE learned arms did not share 6000 minibatches: {pair}")
        if _exact_bool(
            raw_row.get("minibatch_sequence_verified"),
            f"minibatch sequence verified for {pair}",
        ) is not True:
            raise ValueError(f"natural SGE learned-arm minibatch sequence failed: {pair}")
    if observed_pairs != expected_pairs:
        raise ValueError("natural SGE fold/arm training endpoint matrix is incomplete")

    information = _mapping(summary, "information_boundary_audit")
    if information.get(
        "all_arm_outputs_frozen_before_query_evaluation_fields_opened"
    ) is not True:
        raise ValueError("natural SGE query annotations opened before outputs froze")
    if information.get(
        "query_evaluation_fields_used_for_fit_selection_or_inference"
    ) is not False:
        raise ValueError("natural SGE query annotations leaked into fitting or inference")
    claim = _mapping(summary, "claim_boundary")
    if claim.get("clean_target_available") is not False or claim.get(
        "clean_waveform_recovery_claim"
    ) is not False:
        raise ValueError("natural SGE aggregate makes an invalid clean-target claim")
    if claim.get("weak_target_semantics") != "low_artifact_observed_EEG_not_clean_truth":
        raise ValueError("natural SGE weak-target semantics changed")

    decision = _mapping(summary, "natural_decision")
    if decision.get("threshold_source_config") != expected["threshold_source_config"]:
        raise ValueError("natural SGE decision threshold source changed")
    if decision.get("threshold_section") != expected["threshold_section"]:
        raise ValueError("natural SGE decision threshold section changed")
    if decision.get("thresholds_frozen_before_evaluation_outputs") is not True:
        raise ValueError("natural SGE thresholds were not frozen prospectively")
    if decision.get("evaluation_outcomes_used_to_select_or_change_thresholds") is not False:
        raise ValueError("natural SGE evaluation outcomes changed thresholds")
    paired = _exact_int(
        decision.get("paired_primary_success_count"), "paired primary success count"
    )
    if paired < 0 or paired > 43:
        raise ValueError("natural SGE paired success count is outside [0, 43]")
    if _exact_int(
        summary.get("paired_primary_success_count"),
        "top-level paired primary success count",
    ) != paired:
        raise ValueError("natural SGE paired success counts are inconsistent")
    failures = _exact_int(
        decision.get("conditional_diffusion_failure_count"),
        "conditional diffusion failure count",
    )
    if failures < 0 or failures > 43:
        raise ValueError("natural SGE failure count is outside [0, 43]")

    method_coverage = _mapping(summary, "method_coverage")
    conditional_counts = _validated_method_coverage(
        method_coverage, LEARNED_SGE_ARMS[0], "conditional diffusion"
    )
    deterministic_counts = _validated_method_coverage(
        method_coverage, LEARNED_SGE_ARMS[1], "matched deterministic"
    )
    conditional_successes = conditional_counts["success_count"]
    deterministic_successes = deterministic_counts["success_count"]
    conditional_non_success = conditional_counts["compatible_non_success_count"]
    deterministic_non_success = deterministic_counts["compatible_non_success_count"]
    if failures != conditional_non_success or conditional_successes != 43 - failures:
        raise ValueError("natural SGE conditional failure count is inconsistent")
    if paired > min(conditional_successes, deterministic_successes):
        raise ValueError("natural SGE paired count exceeds learned-arm successes")

    paired_keys = tuple(
        str(value) for value in _sequence(summary, "paired_recording_keys")
    )
    if len(paired_keys) != paired or len(set(paired_keys)) != paired:
        raise ValueError("natural SGE paired recording keys are not unique and count-matched")
    expected_recording_keys = set(source_contract["recording_keys"])
    if len(expected_recording_keys) != 43 or not set(paired_keys).issubset(
        expected_recording_keys
    ):
        raise ValueError("natural SGE paired recording keys escaped the frozen 43-key matrix")
    if paired == 43 and set(paired_keys) != expected_recording_keys:
        raise ValueError("natural SGE complete paired matrix does not contain the exact 43 keys")

    paired_summary = _mapping(summary, "conditional_minus_unet")
    coherence, coherence_complete = _paired_delta_metric(
        paired_summary,
        "eog_coherence_reduction",
        overall_paired_count=paired,
    )
    attenuation, attenuation_complete = _paired_delta_metric(
        paired_summary,
        "matching_projector_attenuation_db",
        overall_paired_count=paired,
    )
    nonartifact, nonartifact_complete = _paired_delta_metric(
        paired_summary,
        "nonartifact_observation_preservation",
        overall_paired_count=paired,
    )
    psd, psd_complete = _paired_delta_metric(
        paired_summary,
        "reference_free_psd_distortion",
        overall_paired_count=paired,
    )
    covariance, covariance_complete = _paired_delta_metric(
        paired_summary,
        "reference_free_covariance_distortion",
        overall_paired_count=paired,
    )
    erp, erp_complete = _paired_delta_metric(
        paired_summary,
        "condition_erp_observation_relative_preservation",
        overall_paired_count=paired,
    )

    by_study = _mapping(summary, "by_study")
    if set(by_study) != set(EXPECTED_SGE_STUDIES):
        raise ValueError("natural SGE by-study matrix must contain study02/04/05")
    verified_by_study: dict[str, Any] = {}
    for study in EXPECTED_SGE_STUDIES:
        study_summary = _mapping(by_study, study)
        study_paired = sum(key.startswith(f"{study}/") for key in paired_keys)
        if _exact_int(
            study_summary.get("paired_participant_stem_count"),
            f"{study} paired participant-stem count",
        ) != study_paired:
            raise ValueError(f"natural SGE by-study paired count changed: {study}")
        study_metrics = _mapping(study_summary, "conditional_minus_unet")
        verified_metrics: dict[str, Any] = {}
        for metric in (
            "eog_coherence_reduction",
            "matching_projector_attenuation_db",
            "nonartifact_observation_preservation",
            "reference_free_psd_distortion",
            "reference_free_covariance_distortion",
            "condition_erp_observation_relative_preservation",
        ):
            metric_row = _mapping(study_metrics, metric)
            if _exact_int(
                metric_row.get("paired_count"), f"{study}/{metric} paired_count"
            ) != study_paired:
                raise ValueError(f"natural SGE by-study metric matrix is incomplete: {study}")
            raw_mean = metric_row.get("mean_conditional_minus_unet")
            if study_paired == 0:
                if raw_mean is not None:
                    raise ValueError(f"natural SGE {study}/{metric} has a mean without pairs")
            else:
                _finite_float(raw_mean, f"{study}/{metric} mean")
            verified_metrics[metric] = {
                "paired_count": study_paired,
                "mean_conditional_minus_unet": raw_mean,
            }
        verified_by_study[study] = {
            "paired_participant_stem_count": study_paired,
            "conditional_minus_unet": verified_metrics,
            "decision_role": "required_reporting_not_an_additional_gate",
        }

    raw_joint = decision.get("joint_primary_win_fraction")
    joint: float | None
    if raw_joint is None:
        joint = None
    else:
        joint = _finite_float(raw_joint, "joint primary win fraction")
        if joint < 0.0 or joint > 1.0:
            raise ValueError("joint primary win fraction is outside [0, 1]")

    aggregate_complete = _exact_bool(
        decision.get("aggregate_complete"), "aggregate_complete"
    )
    if not aggregate_complete:
        raise ValueError("natural SGE aggregate_complete must be true")
    if _exact_bool(
        decision.get("bootstrap_intervals_used_as_decision_thresholds"),
        "bootstrap_intervals_used_as_decision_thresholds",
    ):
        raise ValueError("descriptive bootstrap intervals cannot determine v2 status")

    adequate_coverage = (
        paired >= MINIMUM_NATURAL_PAIRED_SUCCESS
        and conditional_non_success <= MAXIMUM_NATURAL_DIFFUSION_FAILURES
        and deterministic_non_success <= MAXIMUM_NATURAL_DIFFUSION_FAILURES
    )
    primary_complete = bool(
        paired > 0
        and coherence_complete
        and attenuation_complete
        and joint is not None
    )
    safety_complete = bool(
        paired > 0
        and nonartifact_complete
        and psd_complete
        and covariance_complete
        and erp_complete
    )
    primary_pass = bool(
        primary_complete
        and coherence is not None
        and coherence > 0.0
        and attenuation is not None
        and attenuation > 0.0
        and joint is not None
        and joint >= MINIMUM_JOINT_PRIMARY_WIN_FRACTION
    )
    safety_pass = bool(
        safety_complete
        and nonartifact is not None
        and nonartifact >= MINIMUM_NONARTIFACT_PRESERVATION_DELTA
        and psd is not None
        and psd <= MAXIMUM_PSD_DISTORTION_DELTA
        and covariance is not None
        and covariance <= MAXIMUM_COVARIANCE_DISTORTION_DELTA
        and erp is not None
        and erp >= MINIMUM_ERP_PRESERVATION_DELTA
    )

    reported_flags = {
        "adequate_coverage": adequate_coverage,
        "primary_metrics_complete_for_all_successful_pairs": primary_complete,
        "safety_metrics_complete_for_all_successful_pairs": safety_complete,
        "primary_benefit_point_pass": primary_pass,
        "safety_point_pass": safety_pass,
    }
    for label, recomputed in reported_flags.items():
        if _exact_bool(decision.get(label), label) is not recomputed:
            raise ValueError(f"natural SGE {label} is inconsistent with aggregate values")

    expected_status = NATURAL_INCONCLUSIVE
    if adequate_coverage and primary_complete and safety_complete:
        if primary_pass and safety_pass:
            expected_status = NATURAL_PASS
        elif safety_pass:
            expected_status = NATURAL_FAIL
    reported_status = decision.get("status")
    if reported_status not in {NATURAL_PASS, NATURAL_FAIL, NATURAL_INCONCLUSIVE}:
        raise ValueError("natural SGE decision status is invalid")
    if reported_status != expected_status:
        raise ValueError("natural SGE decision status is inconsistent with aggregate values")

    return {
        "status": expected_status,
        "paired_primary_success_count": paired,
        "conditional_diffusion_failure_count": failures,
        "matched_deterministic_failure_count": deterministic_non_success,
        "paired_recording_keys": list(paired_keys),
        "frozen_compatible_recording_key_count": len(expected_recording_keys),
        "required_by_study_sections": list(EXPECTED_SGE_STUDIES),
        "by_study_descriptive_audit": verified_by_study,
        "threshold_source_read_and_verified": str(source_contract["source_path"]),
        "aggregate_complete": aggregate_complete,
        **reported_flags,
        "joint_primary_win_fraction": joint,
        "mean_conditional_minus_unet": {
            "eog_coherence_reduction": coherence,
            "matching_projector_attenuation_db": attenuation,
            "nonartifact_observation_preservation": nonartifact,
            "reference_free_psd_distortion": psd,
            "reference_free_covariance_distortion": covariance,
            "condition_erp_observation_relative_preservation": erp,
        },
        "frozen_point_thresholds": {
            "minimum_paired_success_count_of_43": MINIMUM_NATURAL_PAIRED_SUCCESS,
            "maximum_conditional_diffusion_failure_count_of_43": (
                MAXIMUM_NATURAL_DIFFUSION_FAILURES
            ),
            "maximum_matched_deterministic_failure_count_of_43": (
                MAXIMUM_NATURAL_DIFFUSION_FAILURES
            ),
            "minimum_mean_eog_coherence_reduction_delta": 0.0,
            "minimum_mean_matching_projector_attenuation_db_delta": 0.0,
            "minimum_joint_primary_win_fraction": (
                MINIMUM_JOINT_PRIMARY_WIN_FRACTION
            ),
            "minimum_mean_nonartifact_preservation_delta": (
                MINIMUM_NONARTIFACT_PRESERVATION_DELTA
            ),
            "maximum_mean_psd_distortion_delta": MAXIMUM_PSD_DISTORTION_DELTA,
            "maximum_mean_covariance_distortion_delta": (
                MAXIMUM_COVARIANCE_DISTORTION_DELTA
            ),
            "minimum_mean_erp_preservation_delta": MINIMUM_ERP_PRESERVATION_DELTA,
        },
        "rule_interpretation": (
            "frozen_point_direction_stability_rule_not_hypothesis_test"
        ),
        "bootstrap_role": "descriptive_only_not_decision_bearing",
        "bootstrap_intervals_used_as_decision_thresholds": False,
    }


def evaluate_diffusion_incremental_value_v2(
    config: Mapping[str, Any],
    *,
    v1_summary: Mapping[str, Any],
    natural_sge_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen v2 truth table from complete aggregate summaries."""

    validate_decision_v2_config(config)
    local = _validate_v1_summary(v1_summary)
    natural_audit = _validate_natural_sge_summary(natural_sge_summary, config)
    natural = str(natural_audit["status"])
    eegdfus = str(local["eegdfus"])
    klados = dict(local["klados"])

    if natural == NATURAL_PASS and eegdfus == EEGDFUS_MEETS:
        conclusion = SUPPORTED
        rationale = (
            "The prospective natural-SGE matched comparison passed its frozen "
            "point-direction stability rule and the independent EEGDfus local "
            "benchmark met its frozen stability rule. Bootstrap intervals are "
            "descriptive only, and support is limited to those tested protocols."
        )
        claim_limit = "tested_SGE_natural_EEG_and_EEGDfus_protocols_only"
    elif (
        natural == NATURAL_FAIL
        and eegdfus == LOCAL_NO_DETECTABLE
        and all(value == LOCAL_NO_DETECTABLE for value in klados.values())
    ):
        conclusion = NO_DETECTABLE
        rationale = (
            "The natural-SGE conditional protocol, EEGDfus local benchmark, and "
            "all three frozen exploratory Klados non-M2 configurations did not "
            "meet their frozen matched incremental-value rules. The natural-SGE "
            "rule uses point directions and stability, not a bootstrap hypothesis "
            "test. The protocol label is not a claim about untested methods or tasks."
        )
        claim_limit = (
            "tested_datasets_tasks_splits_objectives_and_configurations_only"
        )
    else:
        conclusion = INCONCLUSIVE
        rationale = (
            "The complete evidence directions do not satisfy either all-positive "
            "or all-no-detectable frozen rule. Mixed, local-inconclusive, or "
            "natural-SGE-inconclusive evidence remains fail-closed inconclusive."
        )
        claim_limit = "no_diffusion_family_decision_beyond_reported_local_outcomes"

    if conclusion not in TOP_LEVEL_CONCLUSIONS:
        raise AssertionError("v2 decision escaped its frozen conclusion set")
    return {
        "status": "completed_frozen_v2_decision",
        "protocol_id": PROTOCOL_ID,
        "conclusion": conclusion,
        "current_M2_status": CURRENT_M2_STATUS,
        "protocol_scoped_diffusion_status": conclusion,
        "diffusion_family_wide_status": DIFFUSION_FAMILY_WIDE_STATUS,
        # Retained only so existing small-result consumers do not break.
        "diffusion_family_status": conclusion,
        "natural_sge_local_outcome": natural,
        "natural_sge_recomputed_decision_audit": natural_audit,
        "eegdfus_local_outcome": eegdfus,
        "klados_exploratory_local_outcomes": klados,
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "formal_G3_status": "NOT_RUN_BLOCKED",
        "formal_G1_or_G3_evidence": False,
        "all_required_inputs_complete_and_valid": True,
        "claim_limit": claim_limit,
        "natural_sge_rule_interpretation": (
            "frozen_point_direction_stability_rule_not_hypothesis_test"
        ),
        "bootstrap_role": "descriptive_only_not_decision_bearing",
        "rationale": rationale,
        "evidence_roles": {
            "natural_SGE": (
                "prospective_frozen_release_internal_natural_EEG_matched_comparison"
            ),
            "EEGDfus": "frozen_source_epoch_benchmark_not_natural_multichannel_EEG",
            "Klados": "exploratory_development_source_records",
            "current_M2": "retained_post_hoc_absolute_baseline_audit",
            "confirmatory_formal_G1_G3": "not_run",
        },
        "forbidden_extrapolations": sorted(
            str(value) for value in _sequence(config, "forbidden_claims")
        ),
        "input_validation_blockers": [],
    }


def _fail_closed_result(config: Mapping[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "inconclusive_missing_or_invalid_v2_inputs",
        "protocol_id": str(config.get("protocol_id", PROTOCOL_ID)),
        "conclusion": INCONCLUSIVE,
        "current_M2_status": CURRENT_M2_STATUS,
        "protocol_scoped_diffusion_status": INCONCLUSIVE,
        "diffusion_family_wide_status": DIFFUSION_FAMILY_WIDE_STATUS,
        # Legacy compatibility field; it is not a family-wide scientific claim.
        "diffusion_family_status": INCONCLUSIVE,
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "formal_G3_status": "NOT_RUN_BLOCKED",
        "formal_G1_or_G3_evidence": False,
        "all_required_inputs_complete_and_valid": False,
        "claim_limit": "no_top_level_decision_missing_or_invalid_aggregate_inputs",
        "natural_sge_rule_interpretation": (
            "frozen_point_direction_stability_rule_not_hypothesis_test"
        ),
        "bootstrap_role": "descriptive_only_not_decision_bearing",
        "rationale": (
            "At least one frozen aggregate input is missing, partial, or invalid; "
            "v2 therefore fails closed without fabricating a scientific status."
        ),
        "input_validation_blockers": list(blockers),
    }


def _render_summary_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# CGDR diffusion incremental-value decision v2",
        "",
        f"- Status: `{result['status']}`",
        (
            "- Protocol-scoped diffusion status: "
            f"`{result['protocol_scoped_diffusion_status']}`"
        ),
        (
            "- Diffusion-family-wide status: "
            f"`{result['diffusion_family_wide_status']}`"
        ),
        f"- Retained current M2 status: `{result['current_M2_status']}`",
        f"- Formal G1: `{result['formal_G1_status']}`",
        f"- Formal G3: `{result['formal_G3_status']}`",
        f"- Claim limit: `{result['claim_limit']}`",
        (
            "- Natural-SGE rule: frozen point-direction stability rule, "
            "not a hypothesis test"
        ),
        "- Bootstrap intervals: descriptive only; not decision-bearing",
        "",
        str(result["rationale"]),
    ]
    blockers = result.get("input_validation_blockers", [])
    if isinstance(blockers, Sequence) and not isinstance(blockers, (str, bytes)) and blockers:
        lines.extend(["", "## Input blockers", ""])
        lines.extend(f"- {blocker}" for blocker in blockers)
    lines.extend(
        [
            "",
            "This aggregate does not constitute formal G1 or G3 and does not "
            "generalize beyond the explicitly tested datasets, splits, objectives, "
            "samplers, and matched comparator contracts.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diffusion_incremental_decision_v2(
    config: Mapping[str, Any], run_dir: Path | str
) -> dict[str, Any]:
    """Load the two frozen inputs, evaluate v2, and write small text artifacts."""

    validate_decision_v2_config(config)
    run_path = Path(run_dir)
    artifacts = _mapping(config, "artifacts")
    blockers: list[str] = []
    loaded: dict[str, Mapping[str, Any]] = {}
    for key in ("v1_decision_summary", "natural_sge_evaluation_summary"):
        path = CODE_ROOT / str(artifacts[key])
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("JSON root is not an object")
            loaded[key] = value
        except (OSError, json.JSONDecodeError, ValueError) as error:
            blockers.append(f"{key}: {path}: {error}")

    if blockers:
        result = _fail_closed_result(config, blockers)
    else:
        try:
            result = evaluate_diffusion_incremental_value_v2(
                config,
                v1_summary=loaded["v1_decision_summary"],
                natural_sge_summary=loaded["natural_sge_evaluation_summary"],
            )
        except (KeyError, TypeError, ValueError) as error:
            result = _fail_closed_result(config, [f"input_validation: {error}"])

    output_root = CODE_ROOT / str(artifacts["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)
    resolved = yaml.safe_dump(dict(config), sort_keys=False)
    rendered_json = json.dumps(result, indent=2, sort_keys=True) + "\n"
    rendered_markdown = _render_summary_markdown(result)
    for root in {output_root, run_path}:
        (root / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
        (root / "result_summary.json").write_text(rendered_json, encoding="utf-8")
        (root / "result_summary.md").write_text(rendered_markdown, encoding="utf-8")
    return result
