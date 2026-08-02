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

CURRENT_M2_STATUS = "current_M2_no_incremental_value"
SUPPORTED = "conditional_diffusion_supported"
NO_DETECTABLE = "diffusion_no_detectable_incremental_value_under_tested_protocols"
INCONCLUSIVE = "inconclusive"
TOP_LEVEL_CONCLUSIONS = frozenset({SUPPORTED, NO_DETECTABLE, INCONCLUSIVE})

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
    conditional_coverage = _mapping(method_coverage, LEARNED_SGE_ARMS[0])
    if _exact_int(
        conditional_coverage.get("requested_count"),
        "conditional diffusion requested count",
    ) != 44:
        raise ValueError("natural SGE conditional diffusion denominator changed")
    conditional_successes = _exact_int(
        conditional_coverage.get("success_count"),
        "conditional diffusion success count",
    )
    if conditional_successes != 43 - failures:
        raise ValueError("natural SGE conditional failure count is inconsistent")
    deterministic_coverage = _mapping(method_coverage, LEARNED_SGE_ARMS[1])
    if _exact_int(
        deterministic_coverage.get("requested_count"),
        "matched deterministic requested count",
    ) != 44:
        raise ValueError("natural SGE matched deterministic denominator changed")
    deterministic_successes = _exact_int(
        deterministic_coverage.get("success_count"),
        "matched deterministic success count",
    )
    if deterministic_successes < 0 or deterministic_successes > 43:
        raise ValueError("natural SGE matched deterministic success count is invalid")
    if paired > min(conditional_successes, deterministic_successes):
        raise ValueError("natural SGE paired count exceeds learned-arm successes")

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
        and failures <= MAXIMUM_NATURAL_DIFFUSION_FAILURES
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
            "thresholds and the independent EEGDfus local benchmark met its "
            "frozen stability rule. Support is limited to those tested protocols."
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
            "all three frozen exploratory Klados non-M2 configurations showed no "
            "detectable incremental value under their own matched rules. This is "
            "not a claim about untested diffusion methods or tasks."
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
        "diffusion_family_status": INCONCLUSIVE,
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "formal_G3_status": "NOT_RUN_BLOCKED",
        "formal_G1_or_G3_evidence": False,
        "all_required_inputs_complete_and_valid": False,
        "claim_limit": "no_top_level_decision_missing_or_invalid_aggregate_inputs",
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
        f"- Top-level conclusion: `{result['conclusion']}`",
        f"- Retained current M2 status: `{result['current_M2_status']}`",
        f"- Formal G1: `{result['formal_G1_status']}`",
        f"- Formal G3: `{result['formal_G3_status']}`",
        f"- Claim limit: `{result['claim_limit']}`",
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
