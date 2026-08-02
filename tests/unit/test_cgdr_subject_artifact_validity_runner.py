from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from eeg_cgdr.experiments.subject_artifact_validity_runner import (
    _activation_allowed,
    _combine_v0_results,
    _scale_payload,
    run_subject_artifact_validity,
)


def _v0(passed: bool) -> dict[str, object]:
    return {"status": "passed" if passed else "failed", "passed": passed}


def test_v0_combiner_cannot_hide_wrong_or_shuffled_scale_failure() -> None:
    results = {
        f"{model}:{context}": _v0(True)
        for model in ("deterministic", "diffusion")
        for context in (
            "full_training",
            "population",
            "matching",
            "wrong",
            "shuffled",
        )
    }
    results["diffusion:wrong"] = _v0(False)
    combined = _combine_v0_results(results)
    assert combined["passed"] is False
    assert combined["failed_result_ids"] == ["diffusion:wrong"]


def test_v0_payload_is_derived_from_valid_samples_not_caller_flags() -> None:
    observed = torch.tensor([[[1.0, 2.0, 999.0], [2.0, 4.0, 999.0]]])
    restored = torch.tensor([[[1.0, 1.0, -999.0], [2.0, 2.0, -999.0]]])
    valid = torch.tensor([[True, True, False]])
    payload = _scale_payload(
        observed,
        restored,
        valid,
        observed,
        observed,
        valid,
        span_kind="complement",
        span_error=0.0,
    )
    assert np.allclose(payload["output_input_rms_ratio"], [np.sqrt(10.0 / 25.0)])
    assert payload["input_waveform_values"].tolist() == [1.0, 2.0, 2.0, 4.0]
    assert payload["output_waveform_values"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert payload["low_artifact_relative_observation_change"].tolist() == [0.0]


def test_repair_activation_is_trajectory_or_identity_diagnostic_only(tmp_path) -> None:
    attempt0 = tmp_path / "primary_attempt_0"
    attempt0.mkdir()
    # A geometry-only failure must not activate the identity-data repair.
    attempt0.joinpath("result_summary.json").write_text(
        json.dumps(
            {
                "validity": {
                    "V0": {
                        "passed": False,
                        "results": {
                            "diffusion:matching": {
                                "checks": {
                                    "union_span_consistency": {"passed": False}
                                }
                            }
                        },
                    },
                    "V1": {"passed": True},
                    "V2": {"passed": True},
                    "V3": {"passed": True},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not supported"):
        _activation_allowed(tmp_path, "primary_attempt_1")

    # A V2 instability can activate clip repair directly, without a fake attempt 1.
    attempt0.joinpath("result_summary.json").write_text(
        json.dumps(
            {
                "validity": {
                    "V0": {"passed": True, "results": {}},
                    "V1": {"passed": True},
                    "V2": {"passed": False},
                    "V3": {"passed": False},
                }
            }
        ),
        encoding="utf-8",
    )
    _activation_allowed(tmp_path, "primary_attempt_2")


def test_validity_public_api_cannot_receive_query_scoring_fields() -> None:
    parameters = set(inspect.signature(run_subject_artifact_validity).parameters)
    assert parameters == {"config", "run_dir", "implementation"}
    assert not parameters & {
        "query_eog",
        "eye_tracking",
        "artifact_labels",
        "outcomes",
        "clean_target",
        "best_sample",
    }
