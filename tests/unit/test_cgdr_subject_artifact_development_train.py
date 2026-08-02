"""Contracts for the fixed J3 subject-artifact training array."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_cgdr.experiments.sgeyesub_diffusion import TrialWindowOrigin
from eeg_cgdr.experiments.subject_artifact_data import (
    ArtifactLatentTrainingArrays,
    OuterTrainingLatentNormalizer,
    PreparedSubjectArtifactFold,
    RuntimeArtifactContext,
    SubjectArtifactModelDimensions,
    UnifiedDevelopmentFold,
)
from eeg_cgdr.experiments.subject_artifact_development_train import (
    ArtifactLatentNormalization,
    build_augmented_artifact_training,
    build_training_contract,
    deterministic_inner_recording_split,
    masked_validation_components,
    recording_indices,
    subject_artifact_training_task,
    subject_artifact_training_task_table,
)
from eeg_cgdr.experiments.subject_artifact_training import SubjectArtifactTensorBatch


def _config() -> dict[str, object]:
    return {
        "protocol_id": "unit_subject_artifact",
        "training": {
            "seeds": [20260811, 20260812, 20260813],
            "inner_validation": {
                "split_unit": "recording_key",
                "rule": "lexicographically_last_ceil_20_percent_minimum_one",
                "fraction": 0.20,
                "minimum_stems": 1,
                "heldout_block2_or_query_access": "forbidden",
            },
            "context_augmentation": {
                "entries": [
                    "matching_transfer_with_support_rho",
                    "population_transfer_with_same_support_rho",
                    "population_transfer_with_rho_zero_POP_endpoint",
                ],
                "reconstruction_source": "outer_training_weak_target_plus_real_support_EOG_latent",
                "query_information_used": False,
            },
            "equal_compute_updates": 8000,
            "maximum_updates": 12000,
            "batch_size": 8,
            "checkpoint_interval_updates": 250,
            "validation_interval_updates": 250,
            "learning_rate": 2.0e-4,
            "weight_decay": 1.0e-4,
            "gradient_clip_norm": 1.0,
            "mixed_precision": True,
            "convergence_patience_updates": 1500,
            "convergence_minimum_relative_improvement": 0.001,
            "best_checkpoint_rule": "latent_then_mapped_x0",
        },
        "model": {"base_channels": 8},
        "primary_diffusion": {"timesteps": 1000, "ema_decay": 0.999},
        "artifact_latent": {"posterior_samples": 8},
        "validity": {"V1": {"timesteps": [25, 500, 950]}},
    }


def test_fixed_150_task_map_pairs_models_within_fold_and_seed() -> None:
    table = subject_artifact_training_task_table(_config())
    assert len(table) == 150
    assert table[0].unified_fold_index == 0
    assert table[0].seed == table[1].seed == 20260811
    assert (table[0].model_kind, table[1].model_kind) == (
        "deterministic",
        "diffusion",
    )
    assert table[5].unified_fold_index == 0
    assert table[6].unified_fold_index == 1
    assert table[-1].unified_fold_index == 24
    assert table[-1].seed == 20260813
    assert table[-1].model_kind == "diffusion"
    with pytest.raises(ValueError):
        subject_artifact_training_task(_config(), 150)


def test_inner_split_is_recording_disjoint_and_about_twenty_percent() -> None:
    keys = tuple(f"study02/p{value:02d}" for value in range(10, 20))
    split = deterministic_inner_recording_split(tuple(reversed(keys)))
    assert len(split.training_recording_keys) == 8
    assert len(split.validation_recording_keys) == 2
    assert split.validation_fraction == pytest.approx(0.2)
    assert not set(split.training_recording_keys) & set(
        split.validation_recording_keys
    )
    repeated_windows = (keys[0], keys[0], keys[1], keys[1], keys[2])
    selected = recording_indices(repeated_windows, (keys[0], keys[2]))
    assert selected.tolist() == [0, 1, 4]
    with pytest.raises(ValueError):
        deterministic_inner_recording_split(("only",))
    seven = deterministic_inner_recording_split(tuple(f"p{value}" for value in range(7)))
    assert len(seven.validation_recording_keys) == 2


def _origin(key: str, index: int) -> TrialWindowOrigin:
    return TrialWindowOrigin(key, index, 0, 8)


def _prepared() -> PreparedSubjectArtifactFold:
    count, channels, eog, length = 4, 4, 2, 8
    keys = ("p01", "p01", "p02", "p02")
    normalized = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]],
        dtype=np.float32,
    )
    source_scale = np.asarray(
        [[2.0, 4.0], [2.0, 4.0], [3.0, 6.0], [3.0, 6.0]],
        dtype=np.float32,
    )
    standardized = np.linspace(-1.0, 1.0, count * eog * length).reshape(
        count, eog, length
    ).astype(np.float32)
    normalizer = OuterTrainingLatentNormalizer(
        mean=np.asarray([0.5, -0.25]),
        standard_deviation=np.asarray([2.0, 3.0]),
        training_recording_keys=("p01", "p02"),
    )
    z_subject = normalizer.inverse_transform(standardized)
    contamination = np.einsum(
        "nce,net->nct", normalized[None].repeat(count, axis=0), z_subject
    )
    weak_target = np.full((count, channels, length), 0.25, dtype=np.float32)
    arrays = ArtifactLatentTrainingArrays(
        observed=(weak_target + contamination).astype(np.float32),
        standardized_artifact_latent=standardized,
        valid_time_mask=np.ones((count, length), dtype=bool),
        full_transfer=normalized[None].repeat(count, axis=0)
        * source_scale[:, None, :],
        normalized_transfer=normalized[None].repeat(count, axis=0),
        transfer_scale=source_scale,
        singular_values=np.tile(np.asarray([4.0, 1.0]), (count, 1)),
        rank=np.full(count, 2),
        rho=np.asarray([0.4, 0.4, 0.8, 0.8]),
        calibration_duration_seconds=np.full(count, 30.0),
        channel_mask=np.ones((count, channels), dtype=bool),
        recording_keys=keys,
        target_origins=tuple(_origin(key, value) for value, key in enumerate(keys)),
        artifact_origins=tuple(
            _origin(key, value + 10) for value, key in enumerate(keys)
        ),
    )
    population_normalized = np.asarray(
        [[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    population_scale = np.asarray([5.0, 10.0], dtype=np.float64)
    population = RuntimeArtifactContext(
        role="population",
        context_id="cell:population",
        raw_transfer=population_normalized * population_scale[None, :],
        full_transfer=population_normalized * population_scale[None, :],
        normalized_transfer=population_normalized,
        transfer_scale=population_scale,
        singular_values=np.asarray([5.0, 2.0]),
        rank=2,
        projector=np.eye(channels),
        rho=0.0,
        calibration_duration_seconds=0.0,
        fit_recording_keys=("p01", "p02"),
    )
    return PreparedSubjectArtifactFold(
        fold=UnifiedDevelopmentFold(
            fold_id="unit_fold",
            unified_fold_index=0,
            original_partition="development",
            original_partition_index=0,
            study="study02",
            layout_id="unit_layout",
            sampling_rate_hz=250.0,
            training_recording_keys=("p01", "p02"),
            heldout_recording_keys=("p03",),
        ),
        model_dimensions=SubjectArtifactModelDimensions(channels, eog, length),
        training=arrays,
        latent_normalizer=normalizer,
        population_context=population,
        heldout={},
    )


def test_source_batch_has_equal_matching_population_and_rho0_pop_entries() -> None:
    prepared = _prepared()
    augmented = build_augmented_artifact_training(prepared, _config())
    assert augmented.arrays.observed.shape[0] == 12
    assert augmented.role_counts() == {
        "matching_subject_rho": 4,
        "population_subject_rho": 4,
        "population_rho_zero_endpoint": 4,
    }
    np.testing.assert_allclose(augmented.arrays.rho[:4], prepared.training.rho)
    np.testing.assert_allclose(augmented.arrays.rho[4:8], prepared.training.rho)
    np.testing.assert_allclose(augmented.arrays.rho[8:], 0.0)
    np.testing.assert_allclose(
        augmented.arrays.calibration_duration_seconds[4:8], 30.0
    )
    np.testing.assert_allclose(
        augmented.arrays.calibration_duration_seconds[8:], 0.0
    )
    # POP rho=0 remains a learned population correction endpoint, not raw EEG.
    np.testing.assert_allclose(
        augmented.arrays.observed[4:8], augmented.arrays.observed[8:]
    )
    np.testing.assert_allclose(
        augmented.arrays.standardized_artifact_latent[4:8],
        augmented.arrays.standardized_artifact_latent[8:],
    )
    assert not np.allclose(
        augmented.arrays.observed[8:], np.zeros_like(augmented.arrays.observed[8:])
    )
    bad = _config()
    bad["training"]["context_augmentation"]["entries"] = [  # type: ignore[index]
        "matching_transfer_with_support_rho"
    ]
    with pytest.raises(ValueError):
        build_augmented_artifact_training(prepared, bad)


def _validation_batch() -> SubjectArtifactTensorBatch:
    observed = torch.ones(2, 3, 4)
    target = torch.zeros(2, 2, 4)
    transfer = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [0.5, 0.5]]]
    ).repeat(2, 1, 1)
    return SubjectArtifactTensorBatch(
        observed=observed,
        target_standardized_latent=target,
        full_transfer=transfer,
        normalized_transfer=transfer,
        transfer_scale=torch.ones(2, 2),
        singular_values=torch.ones(2, 2),
        rank=torch.full((2,), 2),
        rho=torch.full((2,), 0.5),
        calibration_duration_seconds=torch.full((2,), 30.0),
        channel_mask=torch.ones(2, 3, dtype=torch.bool),
        valid_time_mask=torch.ones(2, 4, dtype=torch.bool),
    )


def test_validation_helper_reports_latent_and_observation_anchored_x0_mse() -> None:
    batch = _validation_batch()
    predicted = torch.ones_like(batch.target_standardized_latent)
    values = masked_validation_components(
        predicted,
        batch.target_standardized_latent,
        batch,
        latent_mean=torch.zeros(2),
        latent_standard_deviation=torch.ones(2),
    )
    latent_sum, mapped_sum, x0_sum, latent_count, mapped_count = values
    assert latent_sum / latent_count == pytest.approx(1.0)
    assert mapped_sum > 0.0
    assert x0_sum == pytest.approx(mapped_sum)
    assert mapped_count == 24


def test_resume_contract_records_split_budgets_and_no_query_selection() -> None:
    split = deterministic_inner_recording_split(("p01", "p02", "p03", "p04", "p05"))
    task = subject_artifact_training_task(_config(), 0)
    contract = build_training_contract(
        _config(),
        task=task,
        fold_id="study02_layout_fold0",
        dimensions=SubjectArtifactModelDimensions(62, 2, 1000),
        split=split,
        latent_normalization=ArtifactLatentNormalization(
            mean=np.zeros(2), standard_deviation=np.ones(2)
        ),
        implementation="deadbeef",
    )
    assert contract["equal_compute_updates"] == 8000
    assert contract["maximum_updates"] == 12000
    assert contract["heldout_block2_used_for_checkpoint_selection"] is False
    assert contract["query_eog_or_labels_used"] is False
    assert contract["latent_normalization_fit_scope"] == "outer_training_block1_only"
    assert contract["inner_validation_recording_keys"] == ["p05"]
    assert contract["unified_fold_index"] == 0
    assert contract["training_seed"] == 20260811
    assert contract["endpoint"] == "training_run"
    assert contract["eeg_channels"] == 62
