"""Registered P0 ridge and dataset-specific precision semantics.

These tests are submitted through the aggregate CPU Slurm job; they are not
intended to be executed on the login node.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_cgdr.inference import (
    CalibrationContextProjector,
    DatasetPopulationProjector,
    dataset_population_and_context_states,
    dataset_population_state,
    matched_population_and_context_states,
    population_state_only,
    rho_interpolated_precision_state,
)
from eeg_cgdr.operators import CalibrationBatch, P0Config, fit_p0


def _eligible_config(ridge_lambda: float) -> P0Config:
    return P0Config(
        target_rank=2,
        ridge_lambda=ridge_lambda,
        maximum_reference_condition=float("inf"),
        minimum_singular_ratio=0.0,
        minimum_movement_coverage=0.0,
        bootstrap_replicates=0,
        bootstrap_block_samples=64,
        minimum_bootstrap_success=0.0,
        maximum_bootstrap_median_distance=float("inf"),
        maximum_bootstrap_q90_distance=float("inf"),
    )


def test_p0_is_exact_fp64_ridge_and_uses_svd_of_transfer() -> None:
    rng = np.random.default_rng(20260801)
    samples = 257
    eog = rng.normal(size=(2, samples))
    true_transfer = np.asarray(
        [[1.2, -0.4], [0.1, 0.8], [-0.7, 0.2], [0.5, 1.1]],
        dtype=np.float64,
    )
    eeg = true_transfer @ eog + 0.03 * rng.normal(size=(4, samples)) + 3.5
    ridge = 7.25
    outcome = fit_p0(
        CalibrationBatch(eeg, eog, "fixture", "ridge", 256.0),
        _eligible_config(ridge),
        movement_threshold=0.0,
    )
    assert outcome.status == "eligible"
    assert outcome.transfer is not None

    y = eeg.astype(np.float64) - eeg.mean(axis=1, keepdims=True)
    e = eog.astype(np.float64) - eog.mean(axis=1, keepdims=True)
    regularized_gram = e @ e.T + ridge * np.eye(e.shape[0], dtype=np.float64)
    expected_transfer = np.linalg.solve(regularized_gram, (y @ e.T).T).T
    expected_basis, expected_singular_values, _ = np.linalg.svd(
        expected_transfer, full_matrices=False
    )
    expected_projector = expected_basis[:, :2] @ expected_basis[:, :2].T

    assert outcome.transfer.transfer_matrix.dtype == np.float64
    assert outcome.transfer.predicted_contamination.dtype == np.float64
    np.testing.assert_allclose(
        outcome.transfer.transfer_matrix, expected_transfer, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(
        outcome.transfer.predicted_contamination,
        expected_transfer @ e,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        outcome.transfer.projector, expected_projector, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(
        outcome.transfer.diagnostics["singular_values"],
        expected_singular_values,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert outcome.transfer.diagnostics["numeric_precision"] == "float64"
    assert outcome.transfer.diagnostics["svd_object"] == "transfer_matrix_C"


def test_p0_rejects_negative_ridge() -> None:
    eog = np.arange(16, dtype=np.float64).reshape(2, 8)
    eeg = np.vstack([eog, eog + 1.0])
    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_p0(
            CalibrationBatch(eeg, eog, "fixture", "bad-ridge", 256.0),
            _eligible_config(-0.1),
            movement_threshold=0.0,
        )


def test_p0_prediction_and_projector_are_invariant_to_orthogonal_eog_coordinates() -> None:
    rng = np.random.default_rng(14)
    samples = 401
    eog = rng.normal(size=(2, samples))
    eeg = rng.normal(size=(5, 2)) @ eog + 0.02 * rng.normal(size=(5, samples))
    angle = 0.73
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    first = fit_p0(
        CalibrationBatch(eeg, eog, "fixture", "native", 256.0),
        _eligible_config(0.4),
        movement_threshold=0.0,
    )
    transformed = fit_p0(
        CalibrationBatch(eeg, rotation @ eog, "fixture", "rotated", 256.0),
        _eligible_config(0.4),
        movement_threshold=0.0,
    )
    assert first.transfer is not None and transformed.transfer is not None
    np.testing.assert_allclose(
        transformed.transfer.transfer_matrix,
        first.transfer.transfer_matrix @ rotation.T,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        transformed.transfer.predicted_contamination,
        first.transfer.predicted_contamination,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        transformed.transfer.projector,
        first.transfer.projector,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_unregularized_identifiable_prediction_is_invariant_to_general_reparameterization() -> None:
    """The paper's C'E'=CE contract holds for a general invertible R at lambda=0.

    Isotropic ridge is coordinate dependent for non-orthogonal R, so the exact
    identifiable-object test deliberately uses the registered OLS endpoint.
    """

    rng = np.random.default_rng(20260802)
    eog = rng.normal(size=(2, 503))
    eeg = rng.normal(size=(7, 2)) @ eog + 0.01 * rng.normal(size=(7, 503))
    transform = np.asarray([[1.7, -0.6], [0.25, 0.9]], dtype=np.float64)
    first = fit_p0(
        CalibrationBatch(eeg, eog, "fixture", "native", 256.0),
        _eligible_config(0.0),
        movement_threshold=0.0,
    )
    transformed = fit_p0(
        CalibrationBatch(eeg, transform @ eog, "fixture", "general_R", 256.0),
        _eligible_config(0.0),
        movement_threshold=0.0,
    )
    assert first.transfer is not None and transformed.transfer is not None
    np.testing.assert_allclose(
        transformed.transfer.predicted_contamination,
        first.transfer.predicted_contamination,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        transformed.transfer.projector,
        first.transfer.projector,
        rtol=3.0e-12,
        atol=3.0e-12,
    )


def _projector(index: int, channels: int = 3) -> torch.Tensor:
    value = torch.zeros(channels, channels, dtype=torch.float32)
    value[index, index] = 1.0
    return value


def _formal_states():
    observation = torch.zeros(2, 3, 4, dtype=torch.float32)
    attenuation = torch.tensor(
        [[0.0, 0.5, 1.0, 0.25], [1.0, 0.75, 0.5, 0.0]],
        dtype=torch.float32,
    )
    valid_weight = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 0.5, 1.0, 1.0]],
        dtype=torch.float32,
    )
    pi0 = DatasetPopulationProjector(
        dataset_id="dataset-v1",
        montage_id="three-channel-reference-v1",
        projector=_projector(0),
        source="outer-training-only",
    )
    pic = CalibrationContextProjector(
        dataset_id="dataset-v1",
        montage_id="three-channel-reference-v1",
        projector=_projector(1),
        calibration_id="heldout-support-block",
    )
    population, context = dataset_population_and_context_states(
        observation,
        attenuation=attenuation,
        valid_weight=valid_weight,
        population_projector=pi0,
        context_projector=pic,
        base_precision=2.0,
        energy_scale=0.15,
    )
    return observation, attenuation, valid_weight, pi0, pic, population, context


def test_dataset_precision_uses_pi0_pic_per_frame_and_valid_weight() -> None:
    (
        _,
        attenuation,
        valid_weight,
        _,
        _,
        population,
        context,
    ) = _formal_states()
    identity = torch.eye(3, dtype=torch.float32)
    pi0 = _projector(0)
    pic = _projector(1)
    expected_population = torch.empty_like(population.precision)
    expected_context = torch.empty_like(context.precision)
    for batch in range(2):
        for frame in range(4):
            scale = 2.0 * valid_weight[batch, frame]
            a_squared = attenuation[batch, frame].square()
            expected_population[batch, frame] = scale * (
                identity - (1.0 - a_squared) * pi0
            )
            expected_context[batch, frame] = scale * (
                identity - (1.0 - a_squared) * pic
            )
    torch.testing.assert_close(population.precision, expected_population)
    torch.testing.assert_close(context.precision, expected_context)
    assert population.precision.shape == (2, 4, 3, 3)
    assert population.dataset_id == context.dataset_id == "dataset-v1"
    assert population.montage_id == context.montage_id
    assert population.energy_scale == context.energy_scale == 0.15
    assert population.precision_semantics == "dataset_population_and_context_precision"
    assert not bool(population.valid_time_mask[0, 3])
    assert bool(population.valid_time_mask[1, 1])
    torch.testing.assert_close(population.precision[0, 3], torch.zeros(3, 3))


def test_wrho_is_convex_precision_interpolation() -> None:
    *_, population, context = _formal_states()
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return context

    interpolated = rho_interpolated_precision_state(
        population,
        rho=0.25,
        calibration_accepted=True,
        context_state_factory=factory,
    )
    assert calls == 1
    torch.testing.assert_close(
        interpolated.precision,
        0.75 * population.precision + 0.25 * context.precision,
    )
    clean_estimate = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    torch.testing.assert_close(
        interpolated.energy_per_sample(clean_estimate),
        0.75 * population.energy_per_sample(clean_estimate)
        + 0.25 * context.energy_per_sample(clean_estimate),
    )


def test_rho_zero_returns_pop_without_context_construction() -> None:
    *_, population, _ = _formal_states()
    calls = 0

    def forbidden_factory():
        nonlocal calls
        calls += 1
        raise AssertionError("rho=0 constructed context precision")

    result = rho_interpolated_precision_state(
        population,
        rho=0.0,
        calibration_accepted=True,
        context_state_factory=forbidden_factory,
    )
    assert result is population
    assert calls == 0


def test_formal_population_constructor_requires_frame_fields_and_pi0() -> None:
    observation = torch.zeros(2, 3, 4, dtype=torch.float32)
    pi0 = DatasetPopulationProjector(
        "dataset-v1", "montage-v1", _projector(0), "outer-training-only"
    )
    with pytest.raises(ValueError, match="attenuation must have shape"):
        dataset_population_state(
            observation,
            attenuation=torch.ones(2),
            valid_weight=torch.ones(2, 4),
            population_projector=pi0,
            base_precision=1.0,
        )
    with pytest.raises(TypeError, match="DatasetPopulationProjector"):
        dataset_population_state(
            observation,
            attenuation=torch.ones(2, 4),
            valid_weight=torch.ones(2, 4),
            population_projector=None,  # type: ignore[arg-type]
            base_precision=1.0,
        )


def test_pi0_pic_dataset_and_montage_must_match() -> None:
    observation = torch.zeros(1, 3, 4, dtype=torch.float32)
    pi0 = DatasetPopulationProjector(
        "dataset-v1", "montage-v1", _projector(0), "outer-training-only"
    )
    wrong = CalibrationContextProjector(
        "dataset-v2", "montage-v1", _projector(1), "support"
    )
    with pytest.raises(ValueError, match="dataset IDs"):
        dataset_population_and_context_states(
            observation,
            attenuation=torch.ones(1, 4),
            valid_weight=torch.ones(1, 4),
            population_projector=pi0,
            context_projector=wrong,
            base_precision=1.0,
        )


def test_legacy_api_is_explicitly_labeled_isotropic_ablation() -> None:
    observation = torch.zeros(1, 3, 4, dtype=torch.float32)
    attenuation = torch.tensor([0.2], dtype=torch.float32)
    valid_time_mask = torch.tensor([[True, True, True, False]])
    population = population_state_only(
        observation,
        attenuation=attenuation,
        base_precision=1.0,
        valid_time_mask=valid_time_mask,
    )
    matched_population, _ = matched_population_and_context_states(
        observation,
        attenuation=attenuation,
        projector=_projector(1),
        base_precision=1.0,
        valid_time_mask=valid_time_mask,
    )
    assert population.precision_semantics == "legacy_isotropic_ablation"
    assert "legacy_isotropic_ablation" in population.name
    assert torch.equal(population.valid_time_mask, valid_time_mask)
    torch.testing.assert_close(population.precision, matched_population.precision)
