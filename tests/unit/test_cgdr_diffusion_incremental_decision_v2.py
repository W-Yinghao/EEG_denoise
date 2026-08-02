"""Truth-table and structural tests for the frozen v2 decision."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import eeg_cgdr.experiments.diffusion_incremental_decision_v2 as module
from eeg_cgdr.experiments.diffusion_incremental_decision_v2 import (
    CURRENT_M2_STATUS,
    INCONCLUSIVE,
    NATURAL_FAIL,
    NATURAL_INCONCLUSIVE,
    NATURAL_PASS,
    NO_DETECTABLE,
    SUPPORTED,
    evaluate_diffusion_incremental_value_v2,
    run_diffusion_incremental_decision_v2,
    validate_decision_v2_config,
)


CONFIG_PATH = Path("configs/cgdr/diffusion_incremental_decision_v2.yaml")
SGE_CONFIG_PATH = Path("configs/cgdr/sgeyesub_diffusion_incremental.yaml")


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _expected_sge_keys() -> list[str]:
    source = yaml.safe_load(SGE_CONFIG_PATH.read_text(encoding="utf-8"))
    return [
        f"{fold['study']}/{stem}"
        for fold in source["split"]["evaluation_folds"]
        for stem in fold["heldout_stems"]
    ]


def _v1(
    *, eegdfus: str = "meets_frozen_stability", klados: str = "no_detectable_stability"
) -> dict[str, object]:
    return {
        "status": "completed_fail_closed_decision",
        "protocol_id": "cgdr_diffusion_incremental_value_decision_v1",
        "conclusion": "inconclusive",
        "retained_status": {"current_M2": CURRENT_M2_STATUS},
        "natural_real_eeg_diffusion_comparator_status": "not_run",
        "top_level_diffusion_family_decision_eligible": False,
        "all_loaded_artifact_inputs_complete": True,
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "formal_G3_status": "NOT_RUN_BLOCKED",
        "eegdfus_local_outcome": eegdfus,
        "tested_configuration_outcomes": {
            "M1": klados,
            "M4": klados,
            "operator_conditioned_diffusion_DDIM100": klados,
            "EEGDfus_conditional_diffusion": eegdfus,
        },
    }


def _natural(status: str = NATURAL_PASS) -> dict[str, object]:
    if status == NATURAL_PASS:
        coherence = 0.08
        attenuation = 0.20
        joint = 0.67
        nonartifact = 0.00
    elif status == NATURAL_FAIL:
        coherence = -0.01
        attenuation = 0.10
        joint = 0.55
        nonartifact = 0.00
    elif status == NATURAL_INCONCLUSIVE:
        coherence = 0.08
        attenuation = 0.20
        joint = 0.67
        nonartifact = -0.03
    else:
        raise AssertionError(f"unsupported fixture status: {status}")
    paired_count = 43
    failures = 0
    primary_pass = coherence > 0 and attenuation > 0 and joint >= 0.60
    safety_pass = nonartifact >= -0.02

    def paired_metric(
        mean: float, *, direction: str, count: int = paired_count
    ) -> dict[str, object]:
        return {
            "direction": direction,
            "paired_count": count,
            "mean_conditional_minus_unet": mean,
            "median_conditional_minus_unet": mean,
            "descriptive_bootstrap_mean_ci95": [mean - 0.01, mean + 0.01],
            "conditional_win_count": 30,
        }

    endpoints = [
        {
            "fold_id": fold_id,
            "method_id": method_id,
            "status": "success_fixed_6000_update_endpoint",
            "successful_optimizer_updates": 6000,
            "minibatch_sequence_updates": 6000,
            "minibatch_sequence_verified": True,
        }
        for fold_id in module.EXPECTED_SGE_FOLDS
        for method_id in module.LEARNED_SGE_ARMS
    ]
    paired_keys = _expected_sge_keys()
    by_study: dict[str, object] = {}
    metric_values = {
        "eog_coherence_reduction": (coherence, "higher"),
        "matching_projector_attenuation_db": (attenuation, "higher"),
        "nonartifact_observation_preservation": (nonartifact, "higher"),
        "reference_free_psd_distortion": (0.01, "lower"),
        "reference_free_covariance_distortion": (0.01, "lower"),
        "condition_erp_observation_relative_preservation": (0.00, "higher"),
    }
    for study in module.EXPECTED_SGE_STUDIES:
        count = sum(key.startswith(f"{study}/") for key in paired_keys)
        by_study[study] = {
            "paired_participant_stem_count": count,
            "conditional_minus_unet": {
                metric: paired_metric(mean, direction=direction, count=count)
                for metric, (mean, direction) in metric_values.items()
            },
        }
    return {
        "status": "completed_evaluation_aggregate",
        "protocol_id": "sgeyesub_natural_eeg_diffusion_incremental_v1",
        "partition": "evaluation",
        "evaluation_fold_count": 15,
        "completed_fold_ids": list(module.EXPECTED_SGE_FOLDS),
        "availability_denominator": 44,
        "compatible_performance_denominator": 43,
        "preblocked_count": 1,
        "preblocked_recording_key": "study05/study05_p42",
        "paired_primary_success_count": paired_count,
        "paired_recording_keys": paired_keys,
        "method_coverage": {
            module.LEARNED_SGE_ARMS[0]: {
                "requested_count": 44,
                "success_count": 43 - failures,
                "failed_count": failures,
                "blocked_or_ineligible_count": 1,
                "fallback_count": 0,
            },
            module.LEARNED_SGE_ARMS[1]: {
                "requested_count": 44,
                "success_count": paired_count,
                "failed_count": 43 - paired_count,
                "blocked_or_ineligible_count": 1,
                "fallback_count": 0,
            },
        },
        "conditional_minus_unet": {
            "eog_coherence_reduction": paired_metric(
                coherence, direction="higher"
            ),
            "matching_projector_attenuation_db": paired_metric(
                attenuation, direction="higher"
            ),
            "nonartifact_observation_preservation": paired_metric(
                nonartifact, direction="higher"
            ),
            "reference_free_psd_distortion": paired_metric(
                0.01, direction="lower"
            ),
            "reference_free_covariance_distortion": paired_metric(
                0.01, direction="lower"
            ),
            "condition_erp_observation_relative_preservation": paired_metric(
                0.00, direction="higher"
            ),
        },
        "by_study": by_study,
        "matched_comparison_audit": {
            "same_information_inputs": True,
            "same_outer_training_stems": True,
            "same_weak_supervision_pairs_and_order": True,
            "same_windowing_channels_normalization_and_operator_conditioning": True,
            "successful_optimizer_updates_target": 6000,
            "fold_arm_training_endpoints": endpoints,
        },
        "information_boundary_audit": {
            "all_arm_outputs_frozen_before_query_evaluation_fields_opened": True,
            "query_evaluation_fields_used_for_fit_selection_or_inference": False,
        },
        "claim_boundary": {
            "clean_target_available": False,
            "clean_waveform_recovery_claim": False,
            "weak_target_semantics": "low_artifact_observed_EEG_not_clean_truth",
        },
        "natural_decision": {
            "status": status,
            "threshold_source_config": (
                "configs/cgdr/sgeyesub_diffusion_incremental.yaml"
            ),
            "threshold_section": "prospective_exploratory_thresholds",
            "thresholds_frozen_before_evaluation_outputs": True,
            "evaluation_outcomes_used_to_select_or_change_thresholds": False,
            "paired_primary_success_count": paired_count,
            "conditional_diffusion_failure_count": failures,
            "adequate_coverage": True,
            "aggregate_complete": True,
            "primary_metrics_complete_for_all_successful_pairs": True,
            "safety_metrics_complete_for_all_successful_pairs": True,
            "joint_primary_win_fraction": joint,
            "primary_benefit_point_pass": primary_pass,
            "safety_point_pass": safety_pass,
            "bootstrap_intervals_used_as_decision_thresholds": False,
        },
    }


def _set_paired_keys(natural: dict[str, object], keys: list[str]) -> None:
    paired_count = len(keys)
    natural["paired_primary_success_count"] = paired_count
    natural["paired_recording_keys"] = keys
    decision = natural["natural_decision"]
    decision["paired_primary_success_count"] = paired_count
    decision["adequate_coverage"] = paired_count >= 39
    deterministic = natural["method_coverage"][module.LEARNED_SGE_ARMS[1]]
    deterministic["success_count"] = paired_count
    deterministic["failed_count"] = 43 - paired_count
    for metric in natural["conditional_minus_unet"].values():
        metric["paired_count"] = paired_count
    for study in module.EXPECTED_SGE_STUDIES:
        study_count = sum(key.startswith(f"{study}/") for key in keys)
        study_summary = natural["by_study"][study]
        study_summary["paired_participant_stem_count"] = study_count
        for metric in study_summary["conditional_minus_unet"].values():
            metric["paired_count"] = study_count


def _evaluate(
    *, natural: str, eegdfus: str, klados: str
) -> dict[str, object]:
    return evaluate_diffusion_incremental_value_v2(
        _config(),
        v1_summary=_v1(eegdfus=eegdfus, klados=klados),
        natural_sge_summary=_natural(natural),
    )


def test_frozen_config_is_valid() -> None:
    validate_decision_v2_config(_config())


def test_v2_reads_the_source_sge_threshold_values_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = yaml.safe_load(SGE_CONFIG_PATH.read_text(encoding="utf-8"))
    source["prospective_exploratory_thresholds"]["primary_benefit"][
        "minimum_participant_joint_primary_win_fraction"
    ] = 0.61
    target = tmp_path / SGE_CONFIG_PATH
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(module, "SOURCE_CONFIG_ROOT", tmp_path)
    with pytest.raises(ValueError, match="source primary-benefit thresholds changed"):
        validate_decision_v2_config(_config())


def test_positive_requires_natural_pass_and_eegdfus_meets() -> None:
    result = _evaluate(
        natural=NATURAL_PASS,
        eegdfus="meets_frozen_stability",
        klados="no_detectable_stability",
    )
    assert result["conclusion"] == SUPPORTED
    assert result["current_M2_status"] == CURRENT_M2_STATUS
    assert result["formal_G1_status"] == "NOT_RUN_BLOCKED"
    assert result["formal_G3_status"] == "NOT_RUN_BLOCKED"
    assert result["protocol_scoped_diffusion_status"] == SUPPORTED
    assert result["diffusion_family_wide_status"] == "not_tested"
    assert result["diffusion_family_status"] == SUPPORTED
    assert result["bootstrap_role"] == "descriptive_only_not_decision_bearing"
    assert "point-direction stability rule" in result["rationale"]
    audit = result["natural_sge_recomputed_decision_audit"]
    assert audit["status"] == NATURAL_PASS
    assert audit["primary_benefit_point_pass"] is True
    assert audit["safety_point_pass"] is True
    assert audit["frozen_compatible_recording_key_count"] == 43
    assert len(audit["paired_recording_keys"]) == 43
    assert set(audit["by_study_descriptive_audit"]) == {
        "study02",
        "study04",
        "study05",
    }


def test_negative_requires_all_three_domains_no_detectable() -> None:
    result = _evaluate(
        natural=NATURAL_FAIL,
        eegdfus="no_detectable_stability",
        klados="no_detectable_stability",
    )
    assert result["conclusion"] == NO_DETECTABLE
    assert "tested_datasets_tasks_splits" in result["claim_limit"]
    assert "did not meet their frozen matched incremental-value rules" in result[
        "rationale"
    ]
    assert "not a bootstrap hypothesis test" in result["rationale"]


@pytest.mark.parametrize(
    ("natural", "eegdfus", "klados"),
    [
        (NATURAL_PASS, "no_detectable_stability", "no_detectable_stability"),
        (NATURAL_FAIL, "meets_frozen_stability", "no_detectable_stability"),
        (NATURAL_PASS, "inconclusive", "no_detectable_stability"),
        (NATURAL_FAIL, "no_detectable_stability", "meets_frozen_stability"),
        (NATURAL_FAIL, "no_detectable_stability", "inconclusive"),
        (NATURAL_INCONCLUSIVE, "meets_frozen_stability", "meets_frozen_stability"),
    ],
)
def test_mixed_or_incomplete_directions_are_inconclusive(
    natural: str, eegdfus: str, klados: str
) -> None:
    result = _evaluate(natural=natural, eegdfus=eegdfus, klados=klados)
    assert result["conclusion"] == INCONCLUSIVE


@pytest.mark.parametrize("configuration", module.KLADOS_CONFIGURATIONS)
def test_each_klados_non_m2_configuration_is_required_for_negative(
    configuration: str,
) -> None:
    v1 = _v1(eegdfus="no_detectable_stability", klados="no_detectable_stability")
    v1["tested_configuration_outcomes"][configuration] = "inconclusive"
    result = evaluate_diffusion_incremental_value_v2(
        _config(), v1_summary=v1, natural_sge_summary=_natural(NATURAL_FAIL)
    )
    assert result["conclusion"] == INCONCLUSIVE


def test_partial_natural_fold_matrix_is_rejected() -> None:
    natural = _natural()
    natural["completed_fold_ids"] = natural["completed_fold_ids"][:-1]
    with pytest.raises(ValueError, match="fold identities"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_every_learned_arm_must_reach_6000_updates_in_every_fold() -> None:
    natural = _natural()
    natural["matched_comparison_audit"]["fold_arm_training_endpoints"][7][
        "successful_optimizer_updates"
    ] = 5999
    with pytest.raises(ValueError, match="did not reach 6000 updates"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minibatch_sequence_updates", 5999, "did not share 6000 minibatches"),
        ("minibatch_sequence_verified", False, "minibatch sequence failed"),
    ],
)
def test_every_fold_arm_endpoint_must_verify_the_shared_6000_minibatches(
    field: str, value: object, message: str
) -> None:
    natural = _natural()
    natural["matched_comparison_audit"]["fold_arm_training_endpoints"][7][
        field
    ] = value
    with pytest.raises(ValueError, match=message):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_query_annotation_leakage_is_rejected() -> None:
    natural = _natural()
    natural["information_boundary_audit"][
        "query_evaluation_fields_used_for_fit_selection_or_inference"
    ] = True
    with pytest.raises(ValueError, match="leaked"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_clean_target_claim_is_rejected() -> None:
    natural = _natural()
    natural["claim_boundary"]["clean_waveform_recovery_claim"] = True
    with pytest.raises(ValueError, match="clean-target claim"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_pass_or_fail_label_cannot_bypass_frozen_coverage() -> None:
    natural = _natural(NATURAL_PASS)
    _set_paired_keys(natural, _expected_sge_keys()[:38])
    with pytest.raises(ValueError, match="status is inconsistent"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_matched_unet_uses_the_same_four_failure_limit_as_diffusion() -> None:
    keys = _expected_sge_keys()
    allowed = _natural(NATURAL_PASS)
    _set_paired_keys(allowed, keys[:-4])
    result = evaluate_diffusion_incremental_value_v2(
        _config(), v1_summary=_v1(), natural_sge_summary=allowed
    )
    audit = result["natural_sge_recomputed_decision_audit"]
    assert audit["matched_deterministic_failure_count"] == 4
    assert audit["adequate_coverage"] is True

    excessive = _natural(NATURAL_PASS)
    _set_paired_keys(excessive, keys[:-5])
    with pytest.raises(ValueError, match="status is inconsistent"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=excessive
        )


def test_each_learned_arm_coverage_categories_must_sum_to_44() -> None:
    natural = _natural()
    natural["method_coverage"][module.LEARNED_SGE_ARMS[1]]["fallback_count"] = 1
    with pytest.raises(ValueError, match="coverage categories do not sum to 44"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_paired_recording_keys_must_be_unique_members_of_frozen_43() -> None:
    natural = _natural()
    keys = list(natural["paired_recording_keys"])
    keys[-1] = keys[0]
    natural["paired_recording_keys"] = keys
    with pytest.raises(ValueError, match="not unique and count-matched"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_all_three_by_study_sections_are_required_without_adding_a_study_gate() -> None:
    natural = _natural()
    natural["by_study"].pop("study05")
    with pytest.raises(ValueError, match="must contain study02/04/05"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )

    heterogeneous = _natural(NATURAL_PASS)
    study05 = heterogeneous["by_study"]["study05"]["conditional_minus_unet"]
    study05["eog_coherence_reduction"]["mean_conditional_minus_unet"] = -0.25
    study05["matching_projector_attenuation_db"][
        "mean_conditional_minus_unet"
    ] = -0.50
    result = evaluate_diffusion_incremental_value_v2(
        _config(), v1_summary=_v1(), natural_sge_summary=heterogeneous
    )
    assert result["conclusion"] == SUPPORTED


def _set_erp_metric_specific_counts(
    natural: dict[str, object], *, overall: int, by_study: dict[str, int]
) -> None:
    metric = "condition_erp_observation_relative_preservation"
    natural["conditional_minus_unet"][metric]["paired_count"] = overall
    for study, count in by_study.items():
        natural["by_study"][study]["conditional_minus_unet"][metric][
            "paired_count"
        ] = count


def test_by_study_allows_metric_specific_finite_counts_matching_overall() -> None:
    natural = _natural(NATURAL_INCONCLUSIVE)
    _set_erp_metric_specific_counts(
        natural,
        overall=41,
        by_study={"study02": 15, "study04": 15, "study05": 11},
    )
    natural["natural_decision"][
        "safety_metrics_complete_for_all_successful_pairs"
    ] = False

    result = evaluate_diffusion_incremental_value_v2(
        _config(), v1_summary=_v1(), natural_sge_summary=natural
    )

    audit = result["natural_sge_recomputed_decision_audit"]
    assert audit["status"] == NATURAL_INCONCLUSIVE
    assert audit["by_study_descriptive_audit"]["study02"][
        "conditional_minus_unet"
    ]["condition_erp_observation_relative_preservation"]["paired_count"] == 15
    assert audit["by_study_descriptive_audit"]["study04"][
        "conditional_minus_unet"
    ]["condition_erp_observation_relative_preservation"]["paired_count"] == 15
    assert audit["by_study_descriptive_audit"]["study05"][
        "conditional_minus_unet"
    ]["condition_erp_observation_relative_preservation"]["paired_count"] == 11


def test_by_study_metric_count_sum_must_match_overall_metric_count() -> None:
    natural = _natural(NATURAL_INCONCLUSIVE)
    _set_erp_metric_specific_counts(
        natural,
        overall=41,
        by_study={"study02": 15, "study04": 15, "study05": 10},
    )
    natural["natural_decision"][
        "safety_metrics_complete_for_all_successful_pairs"
    ] = False

    with pytest.raises(ValueError, match="sum does not match overall paired_count"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_by_study_metric_count_cannot_exceed_study_pair_count() -> None:
    natural = _natural(NATURAL_PASS)
    metric = "condition_erp_observation_relative_preservation"
    natural["by_study"]["study05"]["conditional_minus_unet"][metric][
        "paired_count"
    ] = 14

    with pytest.raises(ValueError, match="outside the study paired matrix"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


@pytest.mark.parametrize(
    ("status", "primary_pass", "safety_pass"),
    [
        (NATURAL_PASS, True, True),
        (NATURAL_FAIL, False, True),
        (NATURAL_INCONCLUSIVE, True, False),
    ],
)
def test_natural_decision_truth_table_is_recomputed_from_point_metrics(
    status: str, primary_pass: bool, safety_pass: bool
) -> None:
    result = evaluate_diffusion_incremental_value_v2(
        _config(),
        v1_summary=_v1(),
        natural_sge_summary=_natural(status),
    )
    audit = result["natural_sge_recomputed_decision_audit"]
    assert audit["status"] == status
    assert audit["primary_benefit_point_pass"] is primary_pass
    assert audit["safety_point_pass"] is safety_pass


def test_mutated_primary_point_flag_is_rejected() -> None:
    natural = _natural(NATURAL_PASS)
    natural["conditional_minus_unet"]["eog_coherence_reduction"][
        "mean_conditional_minus_unet"
    ] = -0.01
    with pytest.raises(ValueError, match="primary_benefit_point_pass is inconsistent"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_mutated_safety_point_flag_is_rejected() -> None:
    natural = _natural(NATURAL_PASS)
    natural["conditional_minus_unet"]["reference_free_psd_distortion"][
        "mean_conditional_minus_unet"
    ] = 0.051
    with pytest.raises(ValueError, match="safety_point_pass is inconsistent"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_mutated_completeness_flag_is_rejected() -> None:
    natural = _natural(NATURAL_PASS)
    natural["conditional_minus_unet"]["matching_projector_attenuation_db"][
        "paired_count"
    ] = 42
    natural["by_study"]["study05"]["conditional_minus_unet"][
        "matching_projector_attenuation_db"
    ]["paired_count"] = 12
    with pytest.raises(
        ValueError,
        match="primary_metrics_complete_for_all_successful_pairs is inconsistent",
    ):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_mutated_status_cannot_override_recomputed_truth_table() -> None:
    natural = _natural(NATURAL_PASS)
    natural["natural_decision"]["status"] = NATURAL_FAIL
    with pytest.raises(ValueError, match="status is inconsistent"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_aggregate_must_report_complete() -> None:
    natural = _natural(NATURAL_INCONCLUSIVE)
    natural["natural_decision"]["aggregate_complete"] = False
    with pytest.raises(ValueError, match="aggregate_complete must be true"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_descriptive_bootstrap_cannot_be_used_as_decision_threshold() -> None:
    natural = _natural(NATURAL_PASS)
    natural["natural_decision"][
        "bootstrap_intervals_used_as_decision_thresholds"
    ] = True
    with pytest.raises(ValueError, match="bootstrap intervals cannot determine"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_failure_count_must_match_conditional_method_coverage() -> None:
    natural = _natural(NATURAL_PASS)
    natural["natural_decision"]["conditional_diffusion_failure_count"] = 1
    with pytest.raises(ValueError, match="failure count is inconsistent"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_invalid_protocol_id_is_rejected() -> None:
    natural = _natural()
    natural["protocol_id"] = "made_up_protocol"
    with pytest.raises(ValueError, match="protocol mismatch"):
        evaluate_diffusion_incremental_value_v2(
            _config(), v1_summary=_v1(), natural_sge_summary=natural
        )


def test_runner_missing_inputs_writes_fail_closed_inconclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "CODE_ROOT", tmp_path)
    result = run_diffusion_incremental_decision_v2(_config(), tmp_path / "run")
    assert result["conclusion"] == INCONCLUSIVE
    assert result["protocol_scoped_diffusion_status"] == INCONCLUSIVE
    assert result["diffusion_family_wide_status"] == "not_tested"
    assert result["all_required_inputs_complete_and_valid"] is False
    assert len(result["input_validation_blockers"]) == 2
    written = json.loads(
        (
            tmp_path
            / "results/cgdr/diffusion_incremental_decision_v2/result_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert written["status"] == "inconclusive_missing_or_invalid_v2_inputs"
