"""Frozen, fail-closed diffusion incremental-value decision tests.

These tests are executed by the aggregate CPU Slurm validation job; they do
not read raw EEG or run model inference.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

import eeg_cgdr.experiments.diffusion_incremental_decision as decision_module
from eeg_cgdr.experiments.diffusion_incremental_decision import (
    ALLOWED_CONCLUSIONS,
    evaluate_diffusion_incremental_value,
    run_diffusion_incremental_decision,
    validate_decision_config,
)


CONFIG_PATH = Path("configs/cgdr/diffusion_incremental_decision.yaml")


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _eegdfus_summary(
    *, eog_wins: int, emg_wins: int, eog_safety: float = -0.1, emg_safety: float = -0.1
) -> dict[str, object]:
    paired = []
    for protocol in ("official_native", "strict_source_epoch"):
        for noise, wins, safety in (
            ("EOG", eog_wins, eog_safety),
            ("EMG", emg_wins, emg_safety),
        ):
            paired.append(
                {
                    "protocol": protocol,
                    "noise_type": noise,
                    "snr_levels": 11,
                    "comparison": (
                        "conditional_diffusion_minus_matched_deterministic"
                    ),
                    "paired_source_manifest_equal": True,
                    "paired_optimizer_updates_equal": True,
                    "conditional_win_count_snr_improvement_db": wins,
                    "conditional_win_count_correlation": wins,
                    "conditional_win_count_rrmse_temporal": wins,
                    "mean_delta_rrmse_spectral_corrected_psd_denominator_shape": (
                        safety
                    ),
                }
            )
    return {
        "status": "completed_full_aggregate",
        "benchmark_id": "eegdfus_eegdenoisenet_official_and_strict_v1",
        "scientific_result_eligible": True,
        "matrix_cells_expected": 8,
        "matrix_cells_completed": 8,
        "protocols_kept_separate": ["official_native", "strict_source_epoch"],
        "official_spectral_metric": {
            "status": "blocked_upstream_zero_denominator_shape_400_vs_512"
        },
        "paired_summaries": paired,
    }


def _conditional_summary() -> dict[str, object]:
    return {
        "status": "completed_exploratory_development_no_family_decision",
        "protocol_id": "klados_operator_conditioned_diffusion_matched_v2",
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
        "same_paired_supervision_exposure_for_conditional_and_UNet": True,
        "fixed_optimizer_update_budget_equal_in_every_successful_UNet_pair": True,
        "window_input_target_contract_equal": True,
        "available_source_records_denominator": 8,
        "common_eligible_source_records": 8,
        "expected_conditional_method_cells": 24,
        "observed_conditional_method_cells": 24,
        "unmatched_missing_conditional_method_cells": 0,
        "missing_conditional_result_summaries": 0,
        "completed_records_missing_metrics_files": 0,
    }


def _deterministic_summary() -> dict[str, object]:
    return {
        "status": "completed_descriptive_no_broad_classifier",
        "protocol_id": "klados_stage3_deterministic_scope_isolated_v4",
        "partition": "development",
        "operator_scope_isolation_verified": True,
        "common_record_eligibility_verified_from_checkpoints": True,
        "paired_deltas_are_within_source_record": True,
        "broad_classifier_enabled": False,
        "confirmatory": False,
        "formal_G1_or_G3_evidence": False,
    }


def _delta_values(*, stable: bool) -> dict[str, float]:
    return {
        "e_parallel": -0.2 if stable else 0.2,
        "e_perp": 0.01,
        "rrmse": -0.1 if stable else 0.1,
        "correlation": 0.05 if stable else -0.05,
    }


def _klados_pairs(
    *, conditional_stable: bool, m1_stable: bool, m4_stable: bool
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conditional: list[dict[str, object]] = []
    deterministic: list[dict[str, object]] = []
    for scope in decision_module.OPERATOR_SCOPES:
        estimand = (
            "matching_p0_eligible_only" if scope == "matching_p0" else "operator_effect"
        )
        for record in range(8):
            values = _delta_values(stable=conditional_stable)
            conditional.append(
                {
                    "source_record": f"sim{record + 31:02d}",
                    "operator_source": scope,
                    "source_reference_method_id": (
                        "task_matched_multichannel_deterministic_UNet"
                    ),
                    "pair_status": "success_paired",
                    **{
                        "conditional_minus_"
                        "task_matched_multichannel_deterministic_UNet_"
                        f"{metric}": value
                        for metric, value in values.items()
                    },
                }
            )
            for short_name, stable in (("M1", m1_stable), ("M4", m4_stable)):
                deterministic.append(
                    {
                        "source_record": f"sim{record + 31:02d}",
                        "operator_source": scope,
                        "method_id": decision_module.KLADOS_CANDIDATES[short_name],
                        "comparator_method_id": (
                            "task_matched_multichannel_deterministic_UNet"
                        ),
                        "estimand": estimand,
                        **{
                            f"delta_{metric}": value
                            for metric, value in _delta_values(stable=stable).items()
                        },
                    }
                )
    return conditional, deterministic


def _evaluate(
    *, eeg_eog_wins: int, eeg_emg_wins: int, klados_stable: bool
) -> dict[str, object]:
    conditional, deterministic = _klados_pairs(
        conditional_stable=klados_stable,
        m1_stable=klados_stable,
        m4_stable=klados_stable,
    )
    return evaluate_diffusion_incremental_value(
        _config(),
        eegdfus_summary=_eegdfus_summary(
            eog_wins=eeg_eog_wins, emg_wins=eeg_emg_wins
        ),
        conditional_summary=_conditional_summary(),
        deterministic_summary=_deterministic_summary(),
        conditional_pairs=conditional,
        deterministic_pairs=deterministic,
    )


def test_frozen_config_and_allowed_conclusion_boundary() -> None:
    validate_decision_config(_config())
    assert ALLOWED_CONCLUSIONS == {
        "current_M2_no_incremental_value",
        "conditional_diffusion_supported",
        "diffusion_no_detectable_incremental_value_under_tested_protocols",
        "inconclusive",
    }


def test_both_strict_noise_cells_support_conditional_diffusion() -> None:
    result = _evaluate(eeg_eog_wins=9, eeg_emg_wins=9, klados_stable=False)
    assert result["conclusion"] == "conditional_diffusion_supported"
    assert result["retained_status"]["current_M2"] == (
        "current_M2_no_incremental_value"
    )
    assert result["formal_G1_status"] == "NOT_RUN_BLOCKED"
    assert result["formal_G3_status"] == "NOT_RUN_BLOCKED"


def test_all_four_non_m2_configurations_clear_fail_allow_limited_negative() -> None:
    result = _evaluate(eeg_eog_wins=0, eeg_emg_wins=0, klados_stable=False)
    assert result["conclusion"] == (
        "diffusion_no_detectable_incremental_value_under_tested_protocols"
    )
    assert set(result["tested_configuration_outcomes"]) == {
        "M1",
        "M4",
        "operator_conditioned_diffusion_DDIM100",
        "EEGDfus_conditional_diffusion",
    }


def test_only_one_strict_noise_cell_stable_is_inconclusive() -> None:
    result = _evaluate(eeg_eog_wins=9, eeg_emg_wins=0, klados_stable=False)
    assert result["conclusion"] == "inconclusive"


def test_mixed_klados_directions_never_auto_upgrade_negative() -> None:
    conditional, deterministic = _klados_pairs(
        conditional_stable=False, m1_stable=False, m4_stable=False
    )
    for row in deterministic:
        if row["method_id"] == decision_module.KLADOS_CANDIDATES["M1"]:
            row["delta_correlation"] = 0.05
    result = evaluate_diffusion_incremental_value(
        _config(),
        eegdfus_summary=_eegdfus_summary(eog_wins=0, emg_wins=0),
        conditional_summary=_conditional_summary(),
        deterministic_summary=_deterministic_summary(),
        conditional_pairs=conditional,
        deterministic_pairs=deterministic,
    )
    assert result["tested_configuration_outcomes"]["M1"] == "inconclusive"
    assert result["conclusion"] == "inconclusive"


def test_missing_inputs_write_inconclusive_without_reading_raw_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(decision_module, "CODE_ROOT", tmp_path)
    config = _config()
    artifacts = dict(config["artifacts"])
    artifacts["output_root"] = "results/decision"
    for key in tuple(artifacts):
        if key != "output_root":
            artifacts[key] = f"missing/{key}"
    config["artifacts"] = artifacts
    result = run_diffusion_incremental_decision(config, run_dir=tmp_path / "run")
    assert result["conclusion"] == "inconclusive"
    assert result["all_required_inputs_complete"] is False
    assert len(result["blockers"]) == 5
    written = json.loads(
        (tmp_path / "results/decision/result_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["conclusion"] == "inconclusive"
    with (tmp_path / "results/decision/decision_matrix.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["analysis_status"] == "current_M2_no_incremental_value"

