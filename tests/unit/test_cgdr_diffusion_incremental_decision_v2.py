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


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


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
    endpoints = [
        {
            "fold_id": fold_id,
            "method_id": method_id,
            "status": "success_fixed_6000_update_endpoint",
            "successful_optimizer_updates": 6000,
        }
        for fold_id in module.EXPECTED_SGE_FOLDS
        for method_id in module.LEARNED_SGE_ARMS
    ]
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
            "paired_primary_success_count": 43,
            "conditional_diffusion_failure_count": 0,
        },
    }


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


def test_negative_requires_all_three_domains_no_detectable() -> None:
    result = _evaluate(
        natural=NATURAL_FAIL,
        eegdfus="no_detectable_stability",
        klados="no_detectable_stability",
    )
    assert result["conclusion"] == NO_DETECTABLE
    assert "tested_datasets_tasks_splits" in result["claim_limit"]


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


def test_each_klados_non_m2_configuration_is_required_for_negative() -> None:
    v1 = _v1(eegdfus="no_detectable_stability", klados="no_detectable_stability")
    v1["tested_configuration_outcomes"]["M4"] = "inconclusive"
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
    natural["natural_decision"]["paired_primary_success_count"] = 38
    with pytest.raises(ValueError, match="coverage thresholds"):
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
    assert result["all_required_inputs_complete_and_valid"] is False
    assert len(result["input_validation_blockers"]) == 2
    written = json.loads(
        (
            tmp_path
            / "results/cgdr/diffusion_incremental_decision_v2/result_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert written["status"] == "inconclusive_missing_or_invalid_v2_inputs"

