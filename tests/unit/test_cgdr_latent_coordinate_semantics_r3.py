"""Nontrivial coordinate checks for the J2 r3 semantic repair."""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from eeg_cgdr.experiments.subject_artifact_development_eval import (
    _canonical_map_window,
)
from eeg_cgdr.experiments.subject_artifact_next_round import _identity_repair_loss
from eeg_cgdr.models.artifact_latent_inference import (
    ArtifactInferenceContext,
    canonical_artifact_delta,
    canonical_physical_artifact_latent,
    deterministic_population_subject_restore,
)


MU = torch.tensor([0.75, -0.5], dtype=torch.float64)
SIGMA = torch.tensor([1.5, 0.25], dtype=torch.float64)
C_NORMALIZED = torch.tensor(
    [[1.0, 0.25], [0.0, 1.0], [0.5, -0.5]], dtype=torch.float64
)
TRANSFER_SCALE = torch.tensor([2.0, 4.0], dtype=torch.float64)


def _context(role: str) -> ArtifactInferenceContext:
    basis = torch.linalg.qr(C_NORMALIZED).Q[:, :2]
    return ArtifactInferenceContext(
        context_id=role,
        role=role,  # type: ignore[arg-type]
        full_transfer=C_NORMALIZED * TRANSFER_SCALE[None, :],
        normalized_transfer=C_NORMALIZED,
        transfer_scale=TRANSFER_SCALE,
        singular_values=torch.tensor([5.0, 2.0], dtype=torch.float64),
        rank=2,
        calibration_duration_seconds=30.0,
        latent_mean=MU,
        latent_standard_deviation=SIGMA,
        subspace_basis=basis,
    )


class _FixedModel:
    def __init__(self, standardized: torch.Tensor) -> None:
        self.standardized = standardized
        self.calls = 0

    def __call__(self, observed: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        self.calls += 1
        return self.standardized.to(observed)


def test_physical_zero_is_minus_mean_over_sigma_not_standardized_zero() -> None:
    z0 = (-MU / SIGMA)[None, :, None].expand(2, -1, 7).clone()
    physical = canonical_physical_artifact_latent(
        z0, latent_mean=MU, latent_standard_deviation=SIGMA
    )
    torch.testing.assert_close(physical, torch.zeros_like(physical), atol=0, rtol=0)

    standardized_zero = torch.zeros_like(z0)
    not_physical_zero = canonical_physical_artifact_latent(
        standardized_zero, latent_mean=MU, latent_standard_deviation=SIGMA
    )
    assert not torch.equal(not_physical_zero, torch.zeros_like(not_physical_zero))


def test_normalized_and_full_transfer_coordinate_forms_are_identical_with_mask() -> None:
    z = torch.linspace(-1.0, 1.0, 2 * 2 * 8, dtype=torch.float64).reshape(2, 2, 8)
    mask = torch.ones(2, 3, 8, dtype=torch.float64)
    mask[:, :, -2:] = 0.0
    physical = canonical_physical_artifact_latent(
        z, latent_mean=MU, latent_standard_deviation=SIGMA
    )
    normalized = canonical_artifact_delta(
        z,
        normalized_transfer=C_NORMALIZED,
        latent_mean=MU,
        latent_standard_deviation=SIGMA,
        output_mask=mask,
    )
    full_inverse_scale = torch.einsum(
        "ce,bet->bct",
        C_NORMALIZED * TRANSFER_SCALE[None, :],
        physical / TRANSFER_SCALE[None, :, None],
    ) * mask
    torch.testing.assert_close(normalized, full_inverse_scale)
    assert normalized[..., -2:].count_nonzero() == 0


def test_production_deterministic_and_development_evaluator_share_decoder() -> None:
    observed = torch.ones(1, 3, 8, dtype=torch.float64)
    z = torch.linspace(-0.5, 0.5, 16, dtype=torch.float64).reshape(1, 2, 8)
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    channels = torch.ones(3, dtype=torch.bool)
    model = _FixedModel(z)
    restored = deterministic_population_subject_restore(
        model,
        observed,
        population_context=_context("population"),
        rho=0.0,
        subject_context_factory=lambda: (_ for _ in ()).throw(
            AssertionError("rho=0 constructed subject context")
        ),
        channel_mask=channels,
        valid_time_mask=valid,
    )
    direct = canonical_artifact_delta(
        z,
        normalized_transfer=C_NORMALIZED,
        latent_mean=MU,
        latent_standard_deviation=SIGMA,
        output_mask=valid[:, None, :].double(),
    )
    torch.testing.assert_close(restored.mixed_delta, direct)
    assert model.calls == 1
    assert not restored.subject_context_constructed

    evaluator = _canonical_map_window(
        z.numpy(),
        C_NORMALIZED.numpy(),
        MU.numpy(),
        SIGMA.numpy(),
        valid[0].numpy(),
    )
    np.testing.assert_allclose(evaluator[0], direct[0, :, :6].numpy())


def test_identity_repair_loss_is_defined_in_physical_correction_coordinates() -> None:
    z0 = (-MU / SIGMA)[None, :, None].expand(1, -1, 8).clone()
    base = SimpleNamespace(
        observed=torch.ones(1, 3, 8, dtype=torch.float64),
        target_standardized_latent=z0,
        valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
        channel_mask=torch.ones(1, 3, dtype=torch.bool),
        normalized_transfer=C_NORMALIZED[None],
        model_kwargs=lambda: {},
    )
    model = _FixedModel(z0)
    total, base_loss, identity_loss = _identity_repair_loss(
        model,  # type: ignore[arg-type]
        base,
        base,
        latent_mean=MU,
        latent_standard_deviation=SIGMA,
        identity_scale_squared=1.0,
    )
    torch.testing.assert_close(total, torch.zeros_like(total), atol=0, rtol=0)
    torch.testing.assert_close(base_loss, torch.zeros_like(base_loss), atol=0, rtol=0)
    torch.testing.assert_close(
        identity_loss, torch.zeros_like(identity_loss), atol=0, rtol=0
    )

    standardized_zero_model = _FixedModel(torch.zeros_like(z0))
    total, _base_loss, identity_loss = _identity_repair_loss(
        standardized_zero_model,  # type: ignore[arg-type]
        base,
        base,
        latent_mean=MU,
        latent_standard_deviation=SIGMA,
        identity_scale_squared=1.0,
    )
    assert float(total) > 0.0
    assert float(identity_loss) > 0.0
