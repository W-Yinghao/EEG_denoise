from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eeg_cgdr.experiments import subject_artifact_validity_runner as validity_runner
from eeg_cgdr.experiments.subject_artifact_data import OuterTrainingLatentNormalizer
from eeg_cgdr.experiments.subject_artifact_training import SubjectArtifactTensorBatch
from eeg_cgdr.experiments.subject_artifact_validity_runner import (
    _activation_allowed,
    _combine_v0_results,
    _identity_repair_active,
    _scale_payload,
    _v1_shared_fit_batch,
    _validity_output_paths,
    run_subject_artifact_validity,
)


def _v0(passed: bool) -> dict[str, object]:
    return {"status": "passed" if passed else "failed", "passed": passed}


def _v1_source() -> tuple[SubjectArtifactTensorBatch, OuterTrainingLatentNormalizer]:
    batch, channels, latent, length = 4, 3, 2, 8
    observed = torch.arange(
        1, batch * channels * length + 1, dtype=torch.float32
    ).reshape(batch, channels, length)
    target = torch.linspace(
        -1.0, 1.0, batch * latent * length, dtype=torch.float32
    ).reshape(batch, latent, length)
    normalized = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.25]], dtype=torch.float32
    ).expand(batch, -1, -1).clone()
    source = SubjectArtifactTensorBatch(
        observed=observed,
        target_standardized_latent=target,
        full_transfer=normalized.clone(),
        normalized_transfer=normalized,
        transfer_scale=torch.ones(batch, latent),
        singular_values=torch.tensor([2.0, 1.0]).expand(batch, -1).clone(),
        rank=torch.full((batch,), 2, dtype=torch.long),
        rho=torch.full((batch,), 0.75),
        calibration_duration_seconds=torch.full((batch,), 30.0),
        channel_mask=torch.ones(batch, channels, dtype=torch.bool),
        valid_time_mask=torch.ones(batch, length, dtype=torch.bool),
    )
    normalizer = OuterTrainingLatentNormalizer(
        mean=np.asarray([1.0, -2.0]),
        standard_deviation=np.asarray([2.0, 4.0]),
        training_recording_keys=("training_a",),
    )
    return source, normalizer


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


def test_v1_identity_repair_routes_attempt0_and_attempt1() -> None:
    source, normalizer = _v1_source()
    attempt0, count0 = _v1_shared_fit_batch(
        source,
        normalizer,
        count=3,
        identity_repair_active=False,
    )
    assert count0 == 0
    assert attempt0.batch_size == 3
    assert torch.equal(
        attempt0.target_standardized_latent,
        source.target_standardized_latent[:3],
    )

    attempt1, count1 = _v1_shared_fit_batch(
        source,
        normalizer,
        count=3,
        identity_repair_active=True,
    )
    assert count1 == 1
    assert attempt1.batch_size == 4
    assert torch.equal(
        attempt1.target_standardized_latent[:3],
        source.target_standardized_latent[:3],
    )
    expected_physical_zero = torch.tensor([-0.5, 0.5])[:, None].expand(-1, 8)
    assert torch.equal(
        attempt1.target_standardized_latent[-1], expected_physical_zero
    )
    assert not torch.equal(attempt1.observed[-1], source.observed[0])


def test_overfit_v1_gives_both_models_the_same_repaired_target(monkeypatch) -> None:
    source, normalizer = _v1_source()
    prepared = SimpleNamespace(
        latent_normalizer=normalizer,
        fold=SimpleNamespace(fold_id="fold_shared_target"),
    )
    config = {
        "validity": {
            "single_batch_overfit_batch_size": 3,
            "single_batch_overfit_maximum_updates": 1,
            "check_interval_updates": 1,
            "V1": {
                "minimum_relative_loss_reduction": 0.95,
                "maximum_standardized_latent_RMSE": 0.05,
                "maximum_zero_artifact_relative_observation_change": 0.02,
                "timesteps": [0],
                "both_models_must_fit_same_target": True,
            },
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "seeds": [11, 12, 13],
            "gradient_clip_norm": 1.0,
        },
        "primary_diffusion": {"ema_decay": 0.9},
    }
    models = (torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    monkeypatch.setattr(validity_runner, "_models", lambda *args, **kwargs: models)
    monkeypatch.setattr(validity_runner, "_scaler", lambda enabled: object())
    monkeypatch.setattr(
        validity_runner,
        "_physical_identity_change_by_timestep",
        lambda *args, timesteps, **kwargs: {int(value): 0.0 for value in timesteps},
    )
    calls: list[tuple[SubjectArtifactTensorBatch, str, str]] = []

    class _FakeResult:
        loss_curve = ({"update": 1, "loss": 0.01},)

        def __init__(self, model_id: str, target_id: str) -> None:
            self.model_id = model_id
            self.target_id = target_id

        def validity_rows(self) -> list[dict[str, object]]:
            return [
                {
                    "model_id": self.model_id,
                    "target_id": self.target_id,
                    "timestep": 0,
                    "initial_loss": 1.0,
                    "final_loss": 0.01,
                    "standardized_latent_rmse": 0.01,
                    "zero_artifact_relative_observation_change": 99.0,
                }
            ]

    def _fake_overfit(model, fit_batch, **kwargs):
        del model
        calls.append((fit_batch, kwargs["target_id"], kwargs["model_id"]))
        return _FakeResult(kwargs["model_id"], kwargs["target_id"])

    monkeypatch.setattr(
        validity_runner, "run_v1_fixed_batch_overfit", _fake_overfit
    )
    result, _ = validity_runner._overfit_v1(
        config,
        prepared,
        source,
        None,
        None,
        device=torch.device("cpu"),
        identity_repair_active=True,
    )
    assert result["passed"] is True
    assert len(calls) == 2
    assert calls[0][0] is calls[1][0]
    assert calls[0][0].target_standardized_latent.data_ptr() == calls[1][
        0
    ].target_standardized_latent.data_ptr()
    assert calls[0][1] == calls[1][1]
    assert calls[0][2] != calls[1][2]
    assert calls[0][0].batch_size == 4


def test_identity_repair_inheritance_is_revision_local(tmp_path) -> None:
    revision_a = tmp_path / "revision_a"
    revision_b = tmp_path / "revision_b"
    revision_a.joinpath("primary_attempt_1").mkdir(parents=True)
    revision_a.joinpath("primary_attempt_1", "result_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )

    assert _identity_repair_active(revision_a, "primary_attempt_1") is True
    assert _identity_repair_active(revision_a, "primary_attempt_2") is True
    assert _identity_repair_active(revision_a, "residual_sdedit_backup") is True
    assert _identity_repair_active(revision_a, "primary_attempt_0") is False
    # A completed attempt 1 in another revision cannot activate repair here.
    assert _identity_repair_active(revision_b, "primary_attempt_2") is False
    assert _identity_repair_active(revision_b, "residual_sdedit_backup") is False


def test_validity_attempt_paths_are_nested_under_execution_revision(tmp_path) -> None:
    config = {
        "outputs": {"validity_root": str(tmp_path / "validity")},
        "validity": {"execution_revision": "identity_repair_revision"},
    }
    output_root, attempt_root, output, revision = _validity_output_paths(
        config, "primary_attempt_1"
    )
    assert revision == "identity_repair_revision"
    assert output_root == tmp_path / "validity"
    assert attempt_root == output_root / revision
    assert output == attempt_root / "primary_attempt_1"
    assert output != output_root / "primary_attempt_1"

    config["validity"]["execution_revision"] = "../escape"
    with pytest.raises(ValueError, match="safe path component"):
        _validity_output_paths(config, "primary_attempt_1")


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
