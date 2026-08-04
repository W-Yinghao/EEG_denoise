from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_cgdr.experiments.subject_artifact_subspace_diffusion import _mean_rows
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig,
    ArtifactSubspaceDiffusion,
    DeterministicSubspaceEstimator,
    aligned_artifact_basis,
    artifact_coordinates,
    batched_aligned_bases,
    bounded_subspace_target,
    complement_consistency_error,
    participant_sample_seeds,
    population_fallback_correction,
    reconstruct_from_subspace,
    union_span_consistency_error,
)


def test_basis_alignment_and_bounded_target() -> None:
    rng = np.random.default_rng(17)
    population = rng.normal(size=(19, 2))
    transfer = population @ np.array([[0.0, 2.0], [-3.0, 0.0]])
    population_basis, _, _ = aligned_artifact_basis(population)
    basis, singular, rank = aligned_artifact_basis(transfer, population_basis)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-6)
    assert np.all(np.sum(basis * population_basis, axis=0) > 0)
    assert np.all(singular > 0) and rank.tolist() == [True, True]
    artifact = rng.normal(size=(1, 19, 64))
    target = bounded_subspace_target(artifact, basis[None], np.array([2.0, 3.0]))
    assert target.shape == (1, 2, 64)
    assert np.max(np.abs(target)) <= 1.0


def test_batch_alignment_uses_population_cell() -> None:
    rng = np.random.default_rng(18)
    population = rng.normal(size=(8, 2))
    transfers = np.stack([population, population @ np.array([[0, 1], [-1, 0]])])
    reference, bases, _, masks = batched_aligned_bases(transfers, population)
    assert bases.shape == (2, 8, 2)
    assert np.allclose(bases[0], reference, atol=1e-5)
    assert masks.all()


def test_reconstruction_is_exactly_complement_consistent() -> None:
    torch.manual_seed(3)
    observed = torch.randn(2, 8, 64)
    raw = torch.randn(2, 8, 2)
    basis = torch.linalg.qr(raw).Q
    rank = torch.ones(2, 2, dtype=torch.bool)
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[:, -7:] = False
    u = torch.tanh(torch.randn(2, 2, 64))
    restored, correction = reconstruct_from_subspace(
        observed, basis, u, torch.tensor([1.2, 0.7]), rank, valid
    )
    assert torch.isfinite(restored).all() and torch.isfinite(correction).all()
    assert complement_consistency_error(observed, restored, basis, valid) <= 1e-6
    coordinates = artifact_coordinates(observed, basis, rank)
    assert coordinates.shape == (2, 2, 64)


def test_population_fallback_is_lazy_and_union_consistent() -> None:
    population = torch.randn(2, 8, 64)
    called = 0

    def subject() -> torch.Tensor:
        nonlocal called
        called += 1
        return torch.randn_like(population)

    direct, used = population_fallback_correction(population, 0.0, subject)
    assert direct is population and not used and called == 0
    torch.manual_seed(4)
    observed = torch.randn(2, 8, 64)
    a0 = torch.linalg.qr(torch.randn(2, 8, 2)).Q
    a1 = torch.linalg.qr(torch.randn(2, 8, 2)).Q
    delta0 = torch.einsum("bcr,brt->bct", a0, torch.randn(2, 2, 64))
    delta1 = torch.einsum("bcr,brt->bct", a1, torch.randn(2, 2, 64))
    blended, used = population_fallback_correction(delta0, 0.4, lambda: delta1)
    assert used
    restored = observed - blended
    assert union_span_consistency_error(
        observed, restored, a0, a1, torch.ones(2, 64, dtype=torch.bool)
    ) <= 1e-6


def test_models_share_target_but_not_constant_deterministic_state() -> None:
    config = ArtifactSubspaceConfig(eeg_channels=8, signal_length=64)
    diffusion = ArtifactSubspaceDiffusion(config)
    deterministic = DeterministicSubspaceEstimator(config)
    assert diffusion.backbone.unet.cfg.in_channels == 7
    assert deterministic.backbone.unet.cfg.in_channels == 5
    assert not any("query_EOG" == field for field in diffusion.visible_input_fields)


def test_participant_random_streams_are_shared_only_within_unit() -> None:
    first = participant_sample_seeds("study02/p02", 20260811)
    repeat = participant_sample_seeds("study02/p02", 20260811)
    other = participant_sample_seeds("study02/p03", 20260811)
    assert first == repeat
    assert first != other
    assert len(first) == len(set(first)) == 8


def test_diffusion_k1_k8_and_explicit_generator() -> None:
    config = ArtifactSubspaceConfig(eeg_channels=8, signal_length=64)
    model = ArtifactSubspaceDiffusion(config)
    batch = 2
    observed = torch.randn(batch, 8, 64)
    basis = torch.linalg.qr(torch.randn(batch, 8, 2)).Q
    rank = torch.ones(batch, 2, dtype=torch.bool)
    condition = {
        "observed": observed,
        "basis": basis,
        "reliability": torch.ones(batch),
        "rank_mask": rank,
        "valid_time_mask": torch.ones(batch, 64, dtype=torch.bool),
    }
    generator = torch.Generator(device="cpu").manual_seed(91)
    loss, diagnostics = model.training_loss(
        torch.zeros(batch, 2, 64), generator=generator, **condition
    )
    assert torch.isfinite(loss) and torch.isfinite(diagnostics["u_mse"])
    one, _, calls, _ = model.sample(sample_seeds=(1,), **condition)
    eight, _, calls8, _ = model.sample(sample_seeds=tuple(range(8)), **condition)
    assert one.shape == eight.shape == (batch, 2, 64)
    assert calls == 25 and calls8 == 200
    with pytest.raises(ValueError):
        model.sample(sample_seeds=(1, 2), **condition)


def test_second_level_summary_preserves_three_seed_coverage() -> None:
    raw = [
        {"dataset": "demo", "unit_id": "p01", "method": "M", "training_seed": seed, "metric": float(index)}
        for index, seed in enumerate((20260811, 20260812, 20260813))
    ]
    units = _mean_rows(raw, ("dataset", "unit_id", "method"))
    methods = _mean_rows(units, ("dataset", "method"))
    assert units[0]["seed_count"] == 3
    assert methods[0]["seed_count"] == 3
    assert "training_seed" not in methods[0]
