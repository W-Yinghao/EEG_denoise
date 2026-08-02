"""Fail-closed diffusion incremental-value decision aggregate.

This module reads only the small, completed aggregate artifacts named in the
frozen decision configuration.  It never reads EEG arrays, re-fits a model,
or chooses a threshold.  Missing, incomplete, inconsistent, or mixed evidence
can only produce ``inconclusive``.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import yaml


PROTOCOL_ID = "cgdr_diffusion_incremental_value_decision_v1"
CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
ALLOWED_CONCLUSIONS = frozenset(
    {
        "current_M2_no_incremental_value",
        "conditional_diffusion_supported",
        "diffusion_no_detectable_incremental_value_under_tested_protocols",
        "inconclusive",
    }
)
KLADOS_CANDIDATES = {
    "M1": "M1_observation_warm_start_sdedit",
    "M4": "M4_per_step_quadratic_proximal_q_consistency",
    "operator_conditioned_diffusion_DDIM100": (
        "task_matched_multichannel_operator_conditioned_diffusion_DDIM100"
    ),
}
OPERATOR_SCOPES = (
    "population_projector",
    "matching_p0",
    "query_derived_oracle_projector",
)
PRIMARY_EEGDFUS_METRICS = (
    "snr_improvement_db",
    "correlation",
    "rrmse_temporal",
)


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


def _finite(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is missing or non-numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is non-finite")
    return parsed


def _true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def validate_decision_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the frozen scientific decision contract."""

    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID}")
    if int(config.get("harness_level", -1)) != 1:
        raise ValueError("decision aggregate requires HARNESS_LEVEL=1")
    if config.get("frozen_before_evaluation_outputs") is not True:
        raise ValueError("decision rules must be frozen before evaluation outputs")
    if config.get("seeds_are_independent_units") is not False:
        raise ValueError("algorithmic seeds must not become independent units")

    retained = _mapping(config, "retained_status")
    expected_retained = {
        "current_M2": "current_M2_no_incremental_value",
        "diffusion_family_before_new_results": "not_tested",
        "formal_G1": "NOT_RUN_BLOCKED",
        "formal_G3": "NOT_RUN_BLOCKED",
    }
    if dict(retained) != expected_retained:
        raise ValueError("retained status boundary changed")

    required = set(_sequence(config, "required_complete_inputs"))
    if required != {
        "klados_operator_conditioned_diffusion_matched_v3",
        "klados_stage3_deterministic_scope_isolated_v4",
        "EEGDfus_official_native_seeded_wrapper",
        "EEGDfus_strict_source_epoch_before_mixing",
    }:
        raise ValueError("required complete-input matrix changed")

    strict = _mapping(config, "eegdfus_strict_stability")
    if tuple(strict.get("required_noise_types", ())) != ("EOG", "EMG"):
        raise ValueError("strict EEGDfus decision requires EOG and EMG")
    if tuple(_mapping(strict, "primary_metrics")) != PRIMARY_EEGDFUS_METRICS:
        raise ValueError("strict EEGDfus primary metric set changed")
    if strict.get("require_exact_source_manifest_and_update_budget_pairing") is not True:
        raise ValueError("EEGDfus pairing requirement must remain enabled")
    if dict(_mapping(strict, "safety_metric")) != {
        "rrmse_spectral_corrected_psd_denominator_shape": "lower_or_equal_mean"
    }:
        raise ValueError("EEGDfus corrected spectral safety rule changed")

    klados = _mapping(config, "klados_exploratory_stability")
    if klados.get("require_median_rrmse_delta_below_zero") is not True or klados.get(
        "require_median_correlation_delta_above_zero"
    ) is not True:
        raise ValueError("Klados paired direction rules must remain enabled")
    if klados.get("role") != "exploratory_support_only_not_fresh_confirmatory_evaluation":
        raise ValueError("Klados evidence role changed")

    rules = _mapping(config, "decision_rules")
    supported = _mapping(rules, "conditional_diffusion_supported")
    if supported.get("requires_all_strict_EEGDfus_noise_cells_to_meet_stability") is not True:
        raise ValueError("positive decision must require both strict EEGDfus cells")
    if supported.get("klados_role_if_available") != "supportive_exploratory_only":
        raise ValueError("positive decision overstates the Klados evidence role")
    no_value = _mapping(
        rules, "diffusion_no_detectable_incremental_value_under_tested_protocols"
    )
    if no_value.get("requires_all_complete_inputs") is not True:
        raise ValueError("negative decision must require all complete inputs")
    expected_tested = {
        "M1",
        "M4",
        "operator_conditioned_diffusion_DDIM100",
        "EEGDfus_conditional_diffusion",
    }
    if set(no_value.get("requires_non_M2_configurations_tested", ())) != expected_tested:
        raise ValueError("non-M2 configuration requirement changed")
    if no_value.get("requires_no_configuration_to_meet_its_frozen_matched_stability_rule") is not True:
        raise ValueError("negative decision must fail closed across all tested arms")
    inconclusive = _mapping(rules, "inconclusive")
    if set(inconclusive.get("applies_to", ())) != {
        "partial_matrix",
        "mixed_primary_directions",
        "safety_failure",
        "insufficient_pairing",
        "excess_failures",
        "only_one_EEGDfus_noise_type_stable",
    }:
        raise ValueError("inconclusive fail-closed cases changed")

    forbidden = set(_sequence(config, "forbidden_claims"))
    if forbidden != {
        "diffusion_is_useless",
        "diffusion_sampler_has_no_value",
        "EEG_diffusion_is_disproved",
        "personalization_failed",
    }:
        raise ValueError("forbidden scientific claims changed")

    artifacts = _mapping(config, "artifacts")
    required_paths = {
        "eegdfus_full_aggregate_summary",
        "klados_conditional_v3_summary",
        "klados_conditional_v3_paired_comparison",
        "klados_deterministic_v4_summary",
        "klados_deterministic_v4_paired_comparison",
        "output_root",
    }
    if set(artifacts) != required_paths:
        raise ValueError("decision artifact path set changed")


def _safe_code_path(raw: Any, *, label: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = CODE_ROOT / path
    path = path.resolve()
    if path != CODE_ROOT and CODE_ROOT not in path.parents:
        raise ValueError(f"{label} must remain under the code root")
    return path


def _validate_eegdfus_summary(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if summary.get("status") != "completed_full_aggregate":
        raise ValueError("EEGDfus full aggregate is not terminal-complete")
    if summary.get("benchmark_id") != "eegdfus_eegdenoisenet_official_and_strict_v1":
        raise ValueError("EEGDfus benchmark ID mismatch")
    if summary.get("scientific_result_eligible") is not True:
        raise ValueError("EEGDfus full aggregate is not scientific-result eligible")
    if int(summary.get("matrix_cells_expected", -1)) != 8 or int(
        summary.get("matrix_cells_completed", -2)
    ) != 8:
        raise ValueError("EEGDfus eight-cell matrix is incomplete")
    if set(summary.get("protocols_kept_separate", ())) != {
        "official_native",
        "strict_source_epoch",
    }:
        raise ValueError("EEGDfus protocols were not kept separate")
    spectral = _mapping(summary, "official_spectral_metric")
    if spectral.get("status") != "blocked_upstream_zero_denominator_shape_400_vs_512":
        raise ValueError("EEGDfus official spectral blockage changed")
    paired = _sequence(summary, "paired_summaries")
    if len(paired) != 4 or any(not isinstance(row, Mapping) for row in paired):
        raise ValueError("EEGDfus paired aggregate must contain four protocol/noise cells")
    keys = {(str(row.get("protocol")), str(row.get("noise_type"))) for row in paired}
    expected = {
        (protocol, noise)
        for protocol in ("official_native", "strict_source_epoch")
        for noise in ("EOG", "EMG")
    }
    if keys != expected:
        raise ValueError("EEGDfus paired protocol/noise matrix is incomplete")
    for row in paired:
        if int(row.get("snr_levels", -1)) != 11:
            raise ValueError("EEGDfus paired cell does not contain eleven SNR levels")
        if row.get("comparison") != "conditional_diffusion_minus_matched_deterministic":
            raise ValueError("EEGDfus comparator changed")
        if row.get("paired_source_manifest_equal") is not True or row.get(
            "paired_optimizer_updates_equal"
        ) is not True:
            raise ValueError("EEGDfus exact source/update pairing failed")
    return list(paired)


def _assess_eegdfus(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    paired = _validate_eegdfus_summary(summary)
    frozen = _mapping(config, "eegdfus_strict_stability")
    win_minimum = int(frozen["conditional_win_count_minimum_each_primary_metric"])
    required_snr = int(frozen["required_snr_levels_each"])
    if required_snr != 11:
        raise ValueError("frozen EEGDfus SNR count differs from completed protocol")

    rows: list[dict[str, Any]] = []
    outcomes: list[str] = []
    for noise_type in ("EOG", "EMG"):
        native = next(
            value
            for value in paired
            if value["protocol"] == "official_native"
            and value["noise_type"] == noise_type
        )
        rows.append(
            {
                "evidence_role": (
                    "confirmatory_frozen_native_protocol_reproduction_"
                    "with_disclosed_source_overlap_not_formal_G1_G3"
                ),
                "dataset": "EEGdenoiseNet",
                "configuration": "EEGDfus_conditional_diffusion",
                "protocol": "official_native",
                "noise_type": noise_type,
                "analysis_status": "completed_descriptive_not_decision_bearing",
                "paired_units": int(native["snr_levels"]),
                **{
                    f"conditional_win_count_{metric}": int(
                        native[f"conditional_win_count_{metric}"]
                    )
                    for metric in PRIMARY_EEGDFUS_METRICS
                },
                "mean_delta_corrected_spectral_rrmse": _finite(
                    native[
                        "mean_delta_rrmse_spectral_corrected_psd_denominator_shape"
                    ],
                    f"native {noise_type} corrected spectral delta",
                ),
                "formal_G1_or_G3_evidence": False,
            }
        )
        row = next(
            value
            for value in paired
            if value["protocol"] == "strict_source_epoch"
            and value["noise_type"] == noise_type
        )
        checks: dict[str, bool] = {}
        for metric in PRIMARY_EEGDFUS_METRICS:
            checks[f"win_count_{metric}"] = int(
                row[f"conditional_win_count_{metric}"]
            ) >= win_minimum
        spectral_delta = _finite(
            row[
                "mean_delta_rrmse_spectral_corrected_psd_denominator_shape"
            ],
            f"strict {noise_type} corrected spectral delta",
        )
        safety_pass = spectral_delta <= 0.0
        benefit_count = sum(checks.values())
        if not safety_pass:
            outcome = "inconclusive_safety_failure"
        elif benefit_count == len(PRIMARY_EEGDFUS_METRICS):
            outcome = "meets_frozen_stability"
        elif benefit_count == 0:
            outcome = "no_detectable_stability"
        else:
            outcome = "inconclusive_mixed_primary_directions"
        outcomes.append(outcome)
        rows.append(
            {
                "evidence_role": (
                    "confirmatory_frozen_source_epoch_benchmark_not_formal_G1_G3"
                ),
                "dataset": "EEGdenoiseNet",
                "configuration": "EEGDfus_conditional_diffusion",
                "protocol": "strict_source_epoch",
                "noise_type": noise_type,
                "analysis_status": outcome,
                "paired_units": required_snr,
                "required_win_count": win_minimum,
                **{
                    f"conditional_win_count_{metric}": int(
                        row[f"conditional_win_count_{metric}"]
                    )
                    for metric in PRIMARY_EEGDFUS_METRICS
                },
                "mean_delta_corrected_spectral_rrmse": spectral_delta,
                "safety_pass": safety_pass,
                "formal_G1_or_G3_evidence": False,
            }
        )
    if outcomes == ["meets_frozen_stability", "meets_frozen_stability"]:
        return "meets_frozen_stability", rows
    if outcomes == ["no_detectable_stability", "no_detectable_stability"]:
        return "no_detectable_stability", rows
    return "inconclusive", rows


def _validate_klados_summaries(
    conditional: Mapping[str, Any], deterministic: Mapping[str, Any]
) -> tuple[int, frozenset[str]]:
    if conditional.get("status") != (
        "completed_exploratory_development_no_family_decision"
    ):
        raise ValueError("Klados conditional-v2 aggregate is not terminal-complete")
    if conditional.get("protocol_id") != "klados_operator_conditioned_diffusion_matched_v3":
        raise ValueError("Klados conditional-v3 protocol mismatch")
    if conditional.get("confirmatory") is not False or conditional.get(
        "formal_G1_or_G3_evidence"
    ) is not False:
        raise ValueError("Klados development aggregate overstates its evidence role")
    if conditional.get("same_paired_supervision_exposure_for_conditional_and_UNet") is not True:
        raise ValueError("Klados paired supervision exposure is not matched")
    if conditional.get("fixed_optimizer_update_budget_equal_in_every_successful_UNet_pair") is not True:
        raise ValueError("Klados fixed-endpoint update budget is not matched")
    if conditional.get("window_input_target_contract_equal") is not True:
        raise ValueError("Klados conditional/U-Net input contract differs")
    available = int(conditional.get("available_source_records_denominator", -1))
    if available != 8:
        raise ValueError("Klados conditional aggregate must retain all eight development records")
    expected = int(conditional.get("expected_conditional_method_cells", -1))
    observed = int(conditional.get("observed_conditional_method_cells", -2))
    if expected < 1 or observed != expected or int(
        conditional.get("unmatched_missing_conditional_method_cells", -1)
    ) != 0:
        raise ValueError("Klados conditional method matrix is incomplete")
    if int(conditional.get("missing_conditional_result_summaries", -1)) != 0 or int(
        conditional.get("completed_records_missing_metrics_files", -1)
    ) != 0:
        raise ValueError("Klados conditional record aggregate has missing artifacts")

    if deterministic.get("status") != "completed_descriptive_no_broad_classifier":
        raise ValueError("Klados deterministic-v4 aggregate is not terminal-complete")
    if deterministic.get("protocol_id") != "klados_stage3_deterministic_scope_isolated_v4":
        raise ValueError("Klados deterministic-v4 protocol mismatch")
    if deterministic.get("partition") != "development":
        raise ValueError("Klados deterministic-v4 aggregate is not development-only")
    for field in (
        "operator_scope_isolation_verified",
        "common_record_eligibility_verified_from_checkpoints",
        "paired_deltas_are_within_source_record",
    ):
        if deterministic.get(field) is not True:
            raise ValueError(f"Klados deterministic-v4 failed {field}")
    if deterministic.get("broad_classifier_enabled") is not False:
        raise ValueError("Klados deterministic-v4 broad classifier must remain disabled")
    if deterministic.get("confirmatory") is not False or deterministic.get(
        "formal_G1_or_G3_evidence"
    ) is not False:
        raise ValueError("Klados deterministic-v4 overstates its evidence role")

    denominator = int(conditional.get("common_eligible_source_records", 0))
    if denominator < 1 or denominator > available or expected != denominator * 3:
        raise ValueError("Klados common eligible denominator is invalid")
    raw_record_ids = conditional.get("common_eligible_source_record_ids")
    if not isinstance(raw_record_ids, list):
        raise ValueError("Klados common eligible source-record IDs are missing")
    eligible_record_ids = frozenset(str(value) for value in raw_record_ids)
    if len(raw_record_ids) != denominator or len(eligible_record_ids) != denominator:
        raise ValueError("Klados common eligible source-record IDs are not unique")
    if any(not value.startswith("sim") for value in eligible_record_ids):
        raise ValueError("Klados common eligible source-record ID is malformed")
    return denominator, eligible_record_ids


def _klados_scope_assessment(
    *,
    candidate: str,
    scope: str,
    deltas: Sequence[Mapping[str, Any]],
    denominator: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = _mapping(config, "klados_exploratory_stability")
    paired = len(deltas)
    paired_record_ids = [str(row.get("source_record", "")) for row in deltas]
    if any(not value for value in paired_record_ids):
        raise ValueError(f"{candidate}/{scope} paired row lacks source_record")
    if len(set(paired_record_ids)) != paired:
        raise ValueError(f"{candidate}/{scope} has duplicate source-record pairs")
    if paired > denominator:
        raise ValueError(f"{candidate}/{scope} paired units exceed denominator")
    paired_fraction = paired / denominator
    failure_rate = 1.0 - paired_fraction
    complete_values = paired > 0
    values: dict[str, list[float]] = {}
    for metric in ("e_parallel", "e_perp", "rrmse", "correlation"):
        try:
            values[metric] = [
                _finite(row[f"delta_{metric}"], f"{candidate}/{scope}/{metric}")
                for row in deltas
            ]
        except (KeyError, ValueError):
            complete_values = False
            values[metric] = []

    enough_pairs = paired_fraction >= float(frozen["paired_fraction_minimum"])
    failures_safe = failure_rate <= float(frozen["maximum_failure_rate"])
    e_parallel_fraction = (
        sum(value < 0.0 for value in values["e_parallel"]) / paired
        if complete_values
        else 0.0
    )
    benefit_checks = {
        "e_parallel_improvement_fraction": (
            complete_values
            and e_parallel_fraction
            >= float(frozen["e_parallel_improvement_fraction_minimum"])
        ),
        "median_rrmse_below_zero": (
            complete_values and median(values["rrmse"]) < 0.0
        ),
        "median_correlation_above_zero": (
            complete_values and median(values["correlation"]) > 0.0
        ),
    }
    safety_pass = (
        complete_values
        and median(values["e_perp"])
        <= float(frozen["maximum_median_e_perp_delta"])
    )
    if not complete_values or not enough_pairs or not failures_safe:
        outcome = "inconclusive_insufficient_pairing_or_excess_failures"
    elif not safety_pass:
        outcome = "inconclusive_safety_failure"
    elif all(benefit_checks.values()):
        outcome = "meets_frozen_stability"
    elif not any(benefit_checks.values()):
        outcome = "no_detectable_stability"
    else:
        outcome = "inconclusive_mixed_primary_directions"
    return {
        "evidence_role": "exploratory_Klados_development_source_records",
        "dataset": "Klados_v4",
        "configuration": candidate,
        "operator_scope": scope,
        "analysis_status": outcome,
        "paired_units": paired,
        "available_units": denominator,
        "paired_source_records": ";".join(sorted(paired_record_ids)),
        "paired_fraction": paired_fraction,
        "failure_rate": failure_rate,
        "e_parallel_improvement_fraction": e_parallel_fraction,
        "median_delta_e_parallel": (
            median(values["e_parallel"]) if complete_values else ""
        ),
        "median_delta_e_perp": median(values["e_perp"]) if complete_values else "",
        "median_delta_rrmse": median(values["rrmse"]) if complete_values else "",
        "median_delta_correlation": (
            median(values["correlation"]) if complete_values else ""
        ),
        "safety_pass": safety_pass,
        "formal_G1_or_G3_evidence": False,
    }


def _assess_klados(
    *,
    conditional_summary: Mapping[str, Any],
    deterministic_summary: Mapping[str, Any],
    conditional_pairs: Sequence[Mapping[str, Any]],
    deterministic_pairs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    denominator, eligible_record_ids = _validate_klados_summaries(
        conditional_summary, deterministic_summary
    )
    output_rows: list[dict[str, Any]] = []
    outcomes: dict[str, str] = {}
    for short_name, method_id in KLADOS_CANDIDATES.items():
        scope_outcomes: list[str] = []
        for scope in OPERATOR_SCOPES:
            if short_name == "operator_conditioned_diffusion_DDIM100":
                prefix = (
                    "conditional_minus_"
                    "task_matched_multichannel_deterministic_UNet_"
                )
                selected = [
                    {
                        "source_record": row.get("source_record", ""),
                        **{
                            f"delta_{metric}": row.get(f"{prefix}{metric}", "")
                            for metric in (
                                "e_parallel",
                                "e_perp",
                                "rrmse",
                                "correlation",
                            )
                        },
                    }
                    for row in conditional_pairs
                    if row.get("operator_source") == scope
                    and row.get("source_record") in eligible_record_ids
                    and row.get("source_reference_method_id")
                    == "task_matched_multichannel_deterministic_UNet"
                    and row.get("pair_status") == "success_paired"
                ]
            else:
                estimand = (
                    "matching_p0_eligible_only"
                    if scope == "matching_p0"
                    else "operator_effect"
                )
                selected = [
                    row
                    for row in deterministic_pairs
                    if row.get("operator_source") == scope
                    and row.get("source_record") in eligible_record_ids
                    and row.get("method_id") == method_id
                    and row.get("comparator_method_id")
                    == "task_matched_multichannel_deterministic_UNet"
                    and row.get("estimand") == estimand
                ]
            assessment = _klados_scope_assessment(
                candidate=short_name,
                scope=scope,
                deltas=selected,
                denominator=denominator,
                config=config,
            )
            output_rows.append(assessment)
            scope_outcomes.append(str(assessment["analysis_status"]))
        if "meets_frozen_stability" in scope_outcomes:
            outcomes[short_name] = "meets_frozen_stability"
        elif any(value.startswith("inconclusive") for value in scope_outcomes):
            outcomes[short_name] = "inconclusive"
        else:
            outcomes[short_name] = "no_detectable_stability"
    return outcomes, output_rows


def evaluate_diffusion_incremental_value(
    config: Mapping[str, Any],
    *,
    eegdfus_summary: Mapping[str, Any],
    conditional_summary: Mapping[str, Any],
    deterministic_summary: Mapping[str, Any],
    conditional_pairs: Sequence[Mapping[str, Any]],
    deterministic_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen rules to already-completed aggregate artifacts."""

    validate_decision_config(config)
    eegdfus_outcome, eegdfus_rows = _assess_eegdfus(eegdfus_summary, config)
    klados_outcomes, klados_rows = _assess_klados(
        conditional_summary=conditional_summary,
        deterministic_summary=deterministic_summary,
        conditional_pairs=conditional_pairs,
        deterministic_pairs=deterministic_pairs,
        config=config,
    )
    tested = {
        **klados_outcomes,
        "EEGDfus_conditional_diffusion": eegdfus_outcome,
    }
    decision_rules = _mapping(config, "decision_rules")
    if eegdfus_outcome == "meets_frozen_stability":
        conclusion = "conditional_diffusion_supported"
        rationale = (
            "Both frozen strict-source EEGDfus noise cells met every primary "
            "direction and the corrected spectral safety rule."
        )
        claim_limit = str(
            _mapping(decision_rules, conclusion)["claim_limit"]
        )
    elif eegdfus_outcome == "no_detectable_stability" and all(
        value == "no_detectable_stability" for value in klados_outcomes.values()
    ):
        conclusion = "diffusion_no_detectable_incremental_value_under_tested_protocols"
        rationale = (
            "All required inputs completed and none of the four frozen non-M2 "
            "configurations met its matched stability rule."
        )
        claim_limit = str(
            _mapping(decision_rules, conclusion)["claim_limit"]
        )
    else:
        conclusion = "inconclusive"
        rationale = (
            "The completed matrix contains mixed, safety-limited, or otherwise "
            "non-decisive frozen-rule outcomes; fail-closed aggregation forbids "
            "an upgrade or category-wide negative decision."
        )
        claim_limit = "no_new_incremental_value_claim_from_mixed_frozen_evidence"
    if conclusion not in ALLOWED_CONCLUSIONS:
        raise AssertionError("decision escaped the user-approved conclusion set")

    decision_rows = [
        {
            "evidence_role": "post_hoc_absolute_baseline_audit",
            "dataset": "Klados_v4",
            "configuration": "current_M2",
            "analysis_status": "current_M2_no_incremental_value",
            "formal_G1_or_G3_evidence": False,
        },
        {
            "evidence_role": "post_hoc_corrected_operator_specificity_audit",
            "dataset": "SGEYESUB",
            "configuration": "hard_Q_P0",
            "analysis_status": "hard_Q_P0_tradeoff_inconclusive",
            "formal_G1_or_G3_evidence": False,
        },
        *klados_rows,
        *eegdfus_rows,
    ]
    return {
        "status": "completed_fail_closed_decision",
        "protocol_id": PROTOCOL_ID,
        "conclusion": conclusion,
        "allowed_conclusions": sorted(ALLOWED_CONCLUSIONS),
        "rationale": rationale,
        "retained_status": dict(_mapping(config, "retained_status")),
        "sge_operator_specificity_status": "hard_Q_P0_tradeoff_inconclusive",
        "engineering_priority": "deterministic_first_diffusion_open",
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "formal_G3_status": "NOT_RUN_BLOCKED",
        "all_required_inputs_complete": True,
        "tested_configuration_outcomes": tested,
        "evidence_roles": {
            "confirmatory": (
                "EEGDfus frozen source-epoch benchmark only; not formal G1/G3 "
                "and not participant-level real-EEG evidence"
            ),
            "exploratory": "Klados development source-record comparisons",
            "post_hoc": "retained current-M2 and corrected SGE audits",
        },
        "claim_limit": claim_limit,
        "decision_rows": decision_rows,
        "blockers": [],
    }


def _inconclusive_result(config: Mapping[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    retained = config.get("retained_status", {})
    return {
        "status": "inconclusive_incomplete_or_invalid_inputs",
        "protocol_id": str(config.get("protocol_id", PROTOCOL_ID)),
        "conclusion": "inconclusive",
        "allowed_conclusions": sorted(ALLOWED_CONCLUSIONS),
        "rationale": (
            "At least one frozen aggregate input is missing, incomplete, or "
            "inconsistent; fail-closed aggregation cannot upgrade the conclusion."
        ),
        "retained_status": dict(retained) if isinstance(retained, Mapping) else {},
        "sge_operator_specificity_status": "hard_Q_P0_tradeoff_inconclusive",
        "engineering_priority": "deterministic_first_diffusion_open",
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "formal_G3_status": "NOT_RUN_BLOCKED",
        "all_required_inputs_complete": False,
        "tested_configuration_outcomes": {},
        "evidence_roles": {
            "confirmatory": "not adjudicated because required inputs are incomplete",
            "exploratory": "Klados development source records if present",
            "post_hoc": "retained current-M2 and corrected SGE audits",
        },
        "claim_limit": "No new incremental-value conclusion from incomplete inputs.",
        "decision_rows": [
            {
                "evidence_role": "post_hoc_absolute_baseline_audit",
                "dataset": "Klados_v4",
                "configuration": "current_M2",
                "analysis_status": "current_M2_no_incremental_value",
                "formal_G1_or_G3_evidence": False,
            },
            *(
                {
                    "evidence_role": "input_feasibility",
                    "dataset": "",
                    "configuration": "required_aggregate_input",
                    "analysis_status": "blocked_or_invalid",
                    "blocker": blocker,
                    "formal_G1_or_G3_evidence": False,
                }
                for blocker in blockers
            ),
        ],
        "blockers": list(blockers),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(result: Mapping[str, Any]) -> str:
    tested = _mapping(result, "tested_configuration_outcomes")
    lines = [
        "# Diffusion incremental-value decision",
        "",
        f"Decision: `{result['conclusion']}`.",
        "",
        str(result["rationale"]),
        "",
        "The retained current-M2 status is `current_M2_no_incremental_value`; "
        "this does not become a diffusion-family conclusion. Formal G1 and G3 "
        "remain `NOT_RUN_BLOCKED`.",
        "",
        "## Frozen configuration outcomes",
        "",
        "| Configuration | Outcome | Evidence role |",
        "|---|---|---|",
    ]
    roles = {
        "M1": "exploratory Klados source records",
        "M4": "exploratory Klados source records",
        "operator_conditioned_diffusion_DDIM100": (
            "exploratory Klados source records"
        ),
        "EEGDfus_conditional_diffusion": (
            "confirmatory frozen source-epoch benchmark, not formal G1/G3"
        ),
    }
    for name in (
        "M1",
        "M4",
        "operator_conditioned_diffusion_DDIM100",
        "EEGDfus_conditional_diffusion",
    ):
        lines.append(f"| {name} | {tested.get(name, 'not_adjudicated')} | {roles[name]} |")
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            str(result["claim_limit"]),
            "The SGE operator-specificity status remains "
            "`hard_Q_P0_tradeoff_inconclusive` and is a corrected post-hoc audit, "
            "not diffusion evidence.",
            "",
        ]
    )
    blockers = result.get("blockers", ())
    if blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {value}" for value in blockers)
        lines.append("")
    return "\n".join(lines)


def run_diffusion_incremental_decision(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Load canonical small artifacts, decide fail-closed, and write JSON/CSV/MD."""

    validate_decision_config(config)
    artifacts = _mapping(config, "artifacts")
    paths = {
        key: _safe_code_path(value, label=key) for key, value in artifacts.items()
    }
    blockers: list[str] = []
    payloads: dict[str, Any] = {}
    readers = {
        "eegdfus_full_aggregate_summary": _read_json,
        "klados_conditional_v3_summary": _read_json,
        "klados_conditional_v3_paired_comparison": _read_csv,
        "klados_deterministic_v4_summary": _read_json,
        "klados_deterministic_v4_paired_comparison": _read_csv,
    }
    for key, reader in readers.items():
        path = paths[key]
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            payloads[key] = reader(path)
        except (OSError, ValueError, json.JSONDecodeError, csv.Error) as error:
            blockers.append(f"{key}: {type(error).__name__}: {error}")
    if blockers:
        result = _inconclusive_result(config, blockers)
    else:
        try:
            result = evaluate_diffusion_incremental_value(
                config,
                eegdfus_summary=payloads["eegdfus_full_aggregate_summary"],
                conditional_summary=payloads["klados_conditional_v3_summary"],
                deterministic_summary=payloads["klados_deterministic_v4_summary"],
                conditional_pairs=payloads[
                    "klados_conditional_v3_paired_comparison"
                ],
                deterministic_pairs=payloads[
                    "klados_deterministic_v4_paired_comparison"
                ],
            )
        except (KeyError, TypeError, ValueError) as error:
            result = _inconclusive_result(
                config,
                [f"scientific_input_validation: {type(error).__name__}: {error}"],
            )

    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    public = dict(result)
    decision_rows = list(public.pop("decision_rows"))
    public["outputs"] = {
        "decision_matrix": str(output_root / "decision_matrix.csv"),
        "result_summary": str(output_root / "result_summary.json"),
        "report": str(output_root / "result_summary.md"),
    }
    _write_csv(output_root / "decision_matrix.csv", decision_rows)
    (output_root / "result_summary.json").write_text(
        json.dumps(public, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "result_summary.md").write_text(
        _markdown(public), encoding="utf-8"
    )
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(public, indent=2) + "\n", encoding="utf-8"
    )
    return public


__all__ = [
    "ALLOWED_CONCLUSIONS",
    "PROTOCOL_ID",
    "evaluate_diffusion_incremental_value",
    "run_diffusion_incremental_decision",
    "validate_decision_config",
]
