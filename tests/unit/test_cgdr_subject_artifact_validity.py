"""Numeric acceptance checks for subject-calibrated artifact experiments."""

from __future__ import annotations

import json

import pytest

from eeg_cgdr.experiments.subject_artifact_validity import (
    evaluate_v0,
    evaluate_v1,
    evaluate_v2,
    evaluate_v3,
    evaluate_validity,
)


def _config() -> dict[str, object]:
    return {
        "validity": {
            "V0": {
                "all_finite": True,
                "maximum_per_window_output_input_RMS_ratio": 10.0,
                "full_median_output_input_RMS_ratio": [0.5, 2.0],
                "low_artifact_median_output_input_RMS_ratio": [0.8, 1.2],
                "low_artifact_maximum_median_relative_observation_change": 0.2,
                "pure_operator_maximum_complement_consistency_relative_error": (
                    1.0e-5
                ),
                "pure_operator_maximum_union_span_consistency_relative_error": (
                    2.0e-5
                ),
            },
            "V1": {
                "minimum_relative_loss_reduction": 0.95,
                "maximum_standardized_latent_RMSE": 0.05,
                "maximum_zero_artifact_relative_observation_change": 0.02,
                "timesteps": [25, 500, 950],
                "both_models_must_fit_same_target": True,
            },
            "V2": {
                "maximum_unexplained_adjacent_RMS_ratio": 10.0,
                "exponential_growth_forbidden": True,
                "final_output_must_pass_V0": True,
            },
            "V3": {
                "maximum_repeat_relative_difference": 1.0e-6,
                "minimum_context_swap_artifact_relative_change": 0.01,
                "matching_population_wrong_shuffled_required": True,
                "wrong_and_shuffled_scale_safety_required": True,
                "original_subject_rho_held_fixed": True,
            },
        }
    }


def _v0_payload(kind: str = "complement") -> dict[str, object]:
    return {
        "output_input_rms_ratio": [0.9, 1.0, 1.1, 1.2],
        "low_artifact_output_input_rms_ratio": [0.95, 1.0, 1.05],
        "low_artifact_relative_observation_change": [0.01, 0.03, 0.04],
        "input_waveform_values": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "output_waveform_values": [-0.9, -0.45, 0.0, 0.45, 0.9],
        "channelwise_variance_ratio": [0.81, 1.0, 1.21],
        "span_consistency_kind": kind,
        "span_consistency_relative_error": 5.0e-6,
    }


def _v1_rows() -> list[dict[str, object]]:
    return [
        {
            "model_id": model,
            "target_id": "same_artifact_latent",
            "timestep": timestep,
            "initial_loss": 1.0,
            "final_loss": 0.03,
            "standardized_latent_rmse": 0.04,
            "zero_artifact_relative_observation_change": 0.01,
        }
        for model in ("deterministic", "diffusion")
        for timestep in (25, 500, 950)
    ]


def _v3_payload() -> dict[str, object]:
    return {
        "repeat_relative_difference_by_context": {
            "population": 1.0e-8,
            "matching": 2.0e-8,
            "wrong": 3.0e-8,
            "shuffled": 4.0e-8,
        },
        "context_swap_artifact_relative_change": {
            "population": 0.02,
            "wrong": 0.03,
            "shuffled": 0.04,
        },
        "scale_safety_by_context": {
            "population": True,
            "matching": True,
            "wrong": True,
            "shuffled": True,
        },
        "rho_by_context": {
            "population": 0.7,
            "matching": 0.7,
            "wrong": 0.7,
            "shuffled": 0.7,
        },
    }


def test_v0_reports_scale_quantiles_variance_and_complement() -> None:
    result = evaluate_v0(_config(), _v0_payload())

    assert result["status"] == "passed"
    assert result["checks"]["complement_span_consistency"]["passed"]
    assert result["checks"]["full_quantile_q50_bounds"]["passed"]
    assert result["checks"]["full_variance_ratio_q50_bounds"]["passed"]
    assert result["metrics"]["output_input_rms_ratio_quantiles"]["q95"] > 1.0
    waveform_quantiles = result["metrics"]["output_waveform_quantiles"]
    assert waveform_quantiles["q0.01"] == pytest.approx(-0.882)
    assert waveform_quantiles["q0.50"] == pytest.approx(0.0)
    assert waveform_quantiles["q0.99"] == pytest.approx(0.882)
    assert result["metrics"]["output_waveform_max_abs"] == pytest.approx(0.9)
    assert result["metrics"]["waveform_correlation"] == pytest.approx(1.0)
    variance_distribution = result["metrics"][
        "channelwise_variance_ratio_distribution"
    ]
    assert variance_distribution["count"] == 3
    assert variance_distribution["values"] == [0.81, 1.0, 1.21]
    json.dumps(result, allow_nan=False)


def test_v0_union_requires_its_configured_threshold_and_scale_failure_is_visible() -> None:
    config = _config()
    union = evaluate_v0(config, _v0_payload("union"))
    assert union["status"] == "passed"
    assert union["checks"]["union_span_consistency"]["passed"]

    unsafe = _v0_payload()
    unsafe["output_input_rms_ratio"] = [1.0, 1.0, 12.0]
    failed = evaluate_v0(config, unsafe)
    assert failed["status"] == "failed"
    assert not failed["checks"]["per_window_scale_safety"]["passed"]

    del config["validity"]["V0"][
        "pure_operator_maximum_union_span_consistency_relative_error"
    ]
    blocked = evaluate_v0(config, _v0_payload("union"))
    assert blocked["status"] == "blocked_invalid_config_or_input"


def test_v1_requires_both_models_same_target_and_every_frozen_timestep() -> None:
    result = evaluate_v1(_config(), {"timestep_results": _v1_rows()})
    assert result["status"] == "passed"
    assert len(result["timestep_results"]) == 6

    wrong_target = _v1_rows()
    wrong_target[-1]["target_id"] = "different_target"
    failed = evaluate_v1(_config(), {"timestep_results": wrong_target})
    assert failed["status"] == "failed"
    assert not failed["checks"]["both_models_same_target"]["passed"]

    missing_timestep = _v1_rows()[:-1]
    failed = evaluate_v1(_config(), {"timestep_results": missing_timestep})
    assert not failed["checks"]["configured_timestep_coverage"]["passed"]


def test_v1_loss_rmse_and_zero_artifact_identity_fail_independently() -> None:
    rows = _v1_rows()
    rows[0]["final_loss"] = 0.2
    rows[1]["standardized_latent_rmse"] = 0.08
    rows[2]["zero_artifact_relative_observation_change"] = 0.04
    result = evaluate_v1(_config(), {"timestep_results": rows})

    assert result["status"] == "failed"
    assert not result["checks"]["minimum_relative_loss_reduction"]["passed"]
    assert not result["checks"]["maximum_standardized_latent_rmse"]["passed"]
    assert not result["checks"]["zero_artifact_identity"]["passed"]


def test_v2_returns_plot_rows_and_detects_jump_or_cumulative_expansion() -> None:
    stable = evaluate_v2(
        _config(),
        {
            "trajectories": [
                {"trajectory_id": "diffusion", "steps": [9, 5, 0], "rms": [1.0, 1.2, 0.9]}
            ],
            "final_v0_status": "passed",
        },
    )
    assert stable["status"] == "passed"
    assert len(stable["chart_data"]) == 3
    assert stable["chart_data"][0]["step"] == 9

    expanded = evaluate_v2(
        _config(),
        {
            "trajectories": [
                {
                    "trajectory_id": "diffusion",
                    "rms": [1.0, 2.0, 4.0, 8.0, 12.0],
                }
            ],
            "final_v0_status": "passed",
        },
    )
    assert expanded["status"] == "failed"
    assert expanded["checks"]["no_unexplained_adjacent_jump"]["passed"]
    assert not expanded["checks"]["no_exponential_expansion"]["passed"]


def test_v2_final_v0_is_fail_closed() -> None:
    result = evaluate_v2(
        _config(),
        {
            "trajectories": [{"trajectory_id": "model", "rms": [1.0, 1.0]}],
            "final_v0_status": "failed",
        },
    )
    assert result["status"] == "failed"
    assert not result["checks"]["final_output_v0"]["passed"]


def test_v2_zero_to_nonzero_growth_fails_without_non_json_infinity() -> None:
    result = evaluate_v2(
        _config(),
        {
            "trajectories": [{"trajectory_id": "model", "rms": [0.0, 0.5]}],
            "final_v0_status": "passed",
        },
    )
    assert result["status"] == "failed"
    assert result["checks"]["no_unexplained_adjacent_jump"]["observed"] is None
    assert result["trajectory_summaries"][0][
        "maximum_adjacent_growth_ratio"
    ] is None
    json.dumps(result, allow_nan=False)


def test_v3_checks_repeat_context_change_scale_safety_and_fixed_rho() -> None:
    result = evaluate_v3(_config(), _v3_payload())
    assert result["status"] == "passed"
    assert result["checks"]["original_subject_rho_fixed"]["passed"]

    unsafe = _v3_payload()
    unsafe["scale_safety_by_context"]["wrong"] = False
    unsafe["context_swap_artifact_relative_change"]["shuffled"] = 0.001
    unsafe["rho_by_context"]["shuffled"] = 0.6
    failed = evaluate_v3(_config(), unsafe)
    assert failed["status"] == "failed"
    assert not failed["checks"]["wrong_and_shuffled_scale_safety"]["passed"]
    assert not failed["checks"]["context_swap_changes_artifact"]["passed"]
    assert not failed["checks"]["original_subject_rho_fixed"]["passed"]


def test_dispatcher_returns_machine_readable_blocked_status_without_defaults() -> None:
    blocked = evaluate_validity("V9", _config(), {})
    assert blocked == {
        "validity_level": "V9",
        "status": "blocked_invalid_config_or_input",
        "passed": False,
        "reason": "unknown validity level V9",
        "checks": {},
    }
    json.dumps(blocked, allow_nan=False)
