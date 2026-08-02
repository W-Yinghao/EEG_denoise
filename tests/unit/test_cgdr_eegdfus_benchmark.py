"""Semantic tests for the frozen external EEGDfus adapter.

These tests use deterministic algebraic arrays only.  A scheduled GPU smoke
stage must separately cover the real EEGdenoiseNet files and external model.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from eeg_cgdr.experiments.eegdfus_benchmark import (
    BENCHMARK_ID,
    OFFICIAL_BATCH_SIZE,
    OFFICIAL_COMBINATIONS,
    OFFICIAL_DIFFUSION_STEPS,
    OFFICIAL_EPOCHS,
    TASK_MATRIX,
    MatchedConditionOnly,
    audit_ssed_source_text,
    eegdfus_rrmse_s_corrected_denominator_shape,
    prepare_official_native,
    prepare_strict_source_epoch,
    source_split_manifest_rows,
    validate_eegdfus_config,
)


CONFIG_PATH = Path("configs/baselines/eegdfus_native_strict.yaml")


def _components(clean_count: int = 40, artifact_count: int = 32) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    clean = np.stack(
        [
            np.sin((1.0 + index / 20.0) * time) + 0.01 * index * np.cos(3.0 * time)
            for index in range(clean_count)
        ]
    )
    artifact = np.stack(
        [
            np.cos((0.5 + index / 30.0) * time)
            + 0.02 * index * np.sin(5.0 * time)
            for index in range(artifact_count)
        ]
    )
    return clean.astype(np.float64), artifact.astype(np.float64)


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_config_preserves_official_budget_and_matrix() -> None:
    config = _config()
    validate_eegdfus_config(config)
    assert config["benchmark_id"] == BENCHMARK_ID
    for protocol in ("official_native", "strict_source_epoch"):
        training = config["protocols"][protocol]["training"]
        diffusion = config["protocols"][protocol]["diffusion"]
        assert training["epochs"] == OFFICIAL_EPOCHS
        assert training["batch_size"] == OFFICIAL_BATCH_SIZE
        assert training["combinations"] == OFFICIAL_COMBINATIONS
        assert diffusion["num_steps"] == OFFICIAL_DIFFUSION_STEPS
    assert tuple(tuple(row) for row in config["task_matrix"]) == TASK_MATRIX
    assert (
        config["matched_deterministic"]["optimizer_updates"]
        == "exactly_equal_to_paired_diffusion_arm"
    )


def test_full_config_rejects_smaller_native_budget() -> None:
    config = deepcopy(_config())
    config["protocols"]["official_native"]["training"]["epochs"] = 3999
    with pytest.raises(ValueError, match="4000 epochs"):
        validate_eegdfus_config(config)


def test_official_native_preserves_and_reports_post_mixing_overlap() -> None:
    clean, artifact = _components()
    first = prepare_official_native(
        clean,
        artifact,
        noise_type="EOG",
        seed=17,
        combinations=3,
        test_snr_db=(-5.0, 0.0, 5.0),
    )
    second = prepare_official_native(
        clean,
        artifact,
        noise_type="EOG",
        seed=17,
        combinations=3,
        test_snr_db=(-5.0, 0.0, 5.0),
    )
    np.testing.assert_array_equal(first.train.noisy, second.train.noisy)
    assert first.source_audit["upstream_seed_defined"] is False
    assert first.source_audit["train_validation_leakage_preserved_not_repaired"] is True
    assert first.source_audit["train_validation_clean_overlap"] > 0
    assert first.source_audit["train_validation_artifact_overlap"] > 0
    assert first.source_audit["train_evaluation_clean_overlap"] == 0
    assert first.source_audit["train_evaluation_artifact_overlap"] == 0
    assert first.source_audit["unused_clean_source_epochs"] == 8
    assert first.source_audit["unused_artifact_source_epochs"] == 0


def test_official_native_emg_preserves_upstream_clean_reuse_within_split() -> None:
    clean, artifact = _components(clean_count=40, artifact_count=48)
    prepared = prepare_official_native(
        clean,
        artifact,
        noise_type="EMG",
        seed=19,
        combinations=2,
        test_snr_db=(0.0,),
    )
    assert prepared.noise_type == "EMG"
    assert prepared.source_audit["train_evaluation_clean_overlap"] == 0
    assert prepared.source_audit["train_evaluation_artifact_overlap"] == 0
    assert prepared.source_audit["train_validation_clean_overlap"] > 0
    assert prepared.source_audit["unused_clean_source_epochs"] == 0


@pytest.mark.parametrize("noise_type", ["EOG", "EMG"])
def test_strict_protocol_splits_source_epochs_before_mixing(noise_type: str) -> None:
    clean, artifact = _components(clean_count=40, artifact_count=48 if noise_type == "EMG" else 32)
    prepared = prepare_strict_source_epoch(
        clean,
        artifact,
        noise_type=noise_type,
        seed=23,
        combinations=2,
        test_snr_db=(-5.0, 5.0),
    )
    for key in (
        "train_validation_clean_overlap",
        "train_validation_artifact_overlap",
        "train_evaluation_clean_overlap",
        "train_evaluation_artifact_overlap",
        "validation_evaluation_clean_overlap",
        "validation_evaluation_artifact_overlap",
    ):
        assert prepared.source_audit[key] == 0
    assert prepared.source_audit["identity_unit"] == "source_epoch_not_participant"


def test_strict_evaluation_reuses_exact_pairs_across_snr_levels() -> None:
    clean, artifact = _components()
    prepared = prepare_strict_source_epoch(
        clean,
        artifact,
        noise_type="EOG",
        seed=29,
        combinations=2,
        test_snr_db=(-5.0, 0.0, 5.0),
    )
    reference = prepared.evaluation[0].pairs
    for level in prepared.evaluation[1:]:
        np.testing.assert_array_equal(
            reference.clean_source_epoch, level.pairs.clean_source_epoch
        )
        np.testing.assert_array_equal(
            reference.artifact_source_epoch, level.pairs.artifact_source_epoch
        )
        np.testing.assert_allclose(reference.clean, level.pairs.clean)
    assert not np.allclose(
        prepared.evaluation[0].pairs.noisy,
        prepared.evaluation[-1].pairs.noisy,
    )


def test_split_manifest_never_invents_participant_identity() -> None:
    clean, artifact = _components()
    strict = prepare_strict_source_epoch(
        clean,
        artifact,
        noise_type="EOG",
        seed=31,
        combinations=1,
        test_snr_db=(0.0,),
    )
    rows = source_split_manifest_rows(strict)
    assert rows
    assert len(rows) == clean.shape[0] + artifact.shape[0]
    assert all(row["identity_unit"] == "source_epoch_not_participant" for row in rows)
    assert all("participant" not in row for row in rows)
    assert all("+" not in row["split_membership"] for row in rows)


class _FakeOfficialBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def forward(
        self, latent: torch.Tensor, condition: torch.Tensor, level: torch.Tensor
    ) -> torch.Tensor:
        self.last_inputs = (latent, condition, level)
        return condition * 0.5


def test_matched_deterministic_has_only_the_same_noisy_condition() -> None:
    backbone = _FakeOfficialBackbone()
    model = MatchedConditionOnly(backbone)
    noisy = torch.randn(3, 1, 512)
    output = model(noisy)
    assert torch.equal(output, noisy * 0.5)
    assert backbone.last_inputs is not None
    latent, condition, level = backbone.last_inputs
    assert torch.equal(latent, noisy)
    assert torch.equal(condition, noisy)
    assert torch.equal(level, torch.ones(3, 1))
    assert model.visible_inputs == ("noisy_single_channel_epoch",)


def test_ssed_source_audit_reports_without_repairing_native() -> None:
    train_text = """
    test_idx, val_idx = train_test_split(list(range(len(val_test_idx))), test_size=0.5)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)
    test_set = Subset(train_val_set, test_idx)
    train_val_set = TensorDataset(y_train, X_train)
    test_loader = DataLoader(test_set)
    train(model, config)
    """
    preparation_text = "dataset = [eeg_train, noise_train]"
    findings = audit_ssed_source_text(train_text, preparation_text)
    assert "ssed_holdout_indices_are_rebased_then_applied_to_full_dataset" in findings
    assert "ssed_tensor_dataset_orders_noise_before_clean" in findings
    assert "ssed_test_loader_is_constructed_but_not_evaluated" in findings


def test_spectral_rrmse_compatibility_uses_psd_shaped_denominator() -> None:
    clean = np.arange(2 * 512, dtype=np.float64).reshape(2, 512) / 100.0
    denoised = clean + 0.25

    def get_psd(values: np.ndarray) -> np.ndarray:
        transformed = np.fft.fft(values, n=400, axis=-1)
        return np.square(np.abs(transformed)) / 400.0

    clean_psd = get_psd(clean)
    denoised_psd = get_psd(denoised)
    expected = np.sqrt(np.mean(np.square(denoised_psd - clean_psd))) / np.sqrt(
        np.mean(np.square(clean_psd))
    )
    actual = eegdfus_rrmse_s_corrected_denominator_shape(
        denoised,
        clean,
        get_psd=get_psd,
    )
    assert actual == pytest.approx(expected)
    assert clean_psd.shape == (2, 400)


def test_frozen_external_ssed_source_matches_recorded_audit() -> None:
    source = Path(".external/EEGDfus")
    assert source.is_dir(), "run the frozen EEGDfus checkout Slurm job first"
    train_text = (source / "train_ssed.py").read_text(encoding="utf-8")
    preparation_text = (
        source / "Data_Preparation" / "data_prepare_ssed.py"
    ).read_text(encoding="utf-8")
    findings = audit_ssed_source_text(train_text, preparation_text)
    assert findings == (
        "ssed_holdout_indices_are_rebased_then_applied_to_full_dataset",
        "ssed_tensor_dataset_orders_noise_before_clean",
        "ssed_test_loader_is_constructed_but_not_evaluated",
    )
    assert "(clean_batch, noisy_batch)" in (source / "utils.py").read_text(
        encoding="utf-8"
    )
    assert not any(
        candidate.is_file()
        for candidate in (
            source / "LICENSE",
            source / "LICENSE.txt",
            source / "COPYING",
            source / "NOTICE",
        )
    )
