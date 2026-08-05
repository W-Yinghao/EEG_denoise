from __future__ import annotations

import numpy as np
import torch

from eeg_cgdr.models.parallel_subject_routes import support_summary
from eeg_cgdr.models.subject_bridge import (
    blocked_split_half_reliability,
    coordinate_corrected_bridge,
    fit_signed_beta,
    physical_eeg_delta,
)


def test_support_summary_has_no_target_derived_spectrum() -> None:
    full = torch.randn(3, 5, 2)
    value = support_summary(full, full * 0.8, torch.ones(3, 2), torch.ones(3, 2), torch.full((3,), 100.0))
    assert value.shape == (3, 2 * 5 * 2 + 2 * 2 + 1)


def test_physical_context_round_trip_common_eeg_units_nontrivial_normalization() -> None:
    z = torch.tensor([[[0.0, 1.0], [-1.0, 0.5]]])
    mean = torch.tensor([2.0, -3.0]); std = torch.tensor([4.0, 0.5])
    c = torch.tensor([[1.0, 2.0], [-0.5, 1.5], [0.25, -0.75]])
    mask = torch.tensor([[True, False]])
    got = physical_eeg_delta(z, normalized_transfer=c, latent_mean=mean, latent_standard_deviation=std, valid_time_mask=mask)
    physical = z * std[None, :, None] + mean[None, :, None]
    expected = torch.einsum("ce,bet->bct", c, physical) * mask[:, None]
    torch.testing.assert_close(got, expected)


def test_beta_or_rho_zero_exactly_short_circuits_to_population() -> None:
    population = torch.randn(2, 3, 5); context = torch.randn_like(population); base = torch.randn_like(population)
    gate = torch.rand(2, 1, 5); mask = torch.ones(2, 5, dtype=torch.bool)
    beta_zero, correction_beta = coordinate_corrected_bridge(base, context_delta=context, population_delta=population, beta=0.0, rho=1.0, activity_gate=gate, valid_time_mask=mask)
    rho_zero, correction_rho = coordinate_corrected_bridge(base, context_delta=context, population_delta=population, beta=1.0, rho=0.0, activity_gate=gate, valid_time_mask=mask)
    assert torch.equal(beta_zero, base) and torch.equal(rho_zero, base)
    assert torch.count_nonzero(correction_beta) == 0 and torch.count_nonzero(correction_rho) == 0


def test_blocked_reliability_uses_heldout_half() -> None:
    rng = np.random.default_rng(4); latent = rng.normal(size=(4, 2, 20)); c = rng.normal(size=(3, 2))
    eeg = np.einsum("ce,net->nct", c, latent); valid = np.ones((4, 20), dtype=bool)
    result = blocked_split_half_reliability(eeg, latent, valid, latent_mean=np.zeros(2), latent_standard_deviation=np.ones(2), ridge=1e-8)
    assert result.samples == 80 and result.reliability > 0.99 and result.heldout_error < 1e-10


def test_signed_beta_contains_zero_and_recovers_sign() -> None:
    direction = np.arange(24, dtype=float).reshape(2, 3, 4) / 10
    valid = np.ones((2, 4), dtype=bool)
    beta, zero, selected = fit_signed_beta(direction, -0.5 * direction, valid)
    assert np.isclose(beta, -0.5) and selected < zero
