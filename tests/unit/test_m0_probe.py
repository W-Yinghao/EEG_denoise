"""Unit tests for the U1-c LoRA probe scaffold and U0-b coverage math."""
from __future__ import annotations

import numpy as np
import pytest


def test_lora_zero_init_noop():
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond
    from eeg_chart.lora_probe import inject_score_lora, lora_parameters

    torch.manual_seed(0)
    model = CalibSADDPMCond(channels=4).eval()
    x_t = torch.randn(2, 4, 512)
    y = torch.randn(2, 4, 512)
    transfer = torch.randn(2, 4, 53)
    timestep = torch.tensor([10, 500])
    with torch.no_grad():
        before = model(x_t, y, timestep, transfer)
    summary = inject_score_lora(model, rank=4)
    with torch.no_grad():
        after = model(x_t, y, timestep, transfer)
    torch.testing.assert_close(before, after, rtol=0, atol=0)  # zero-init adapters: exact no-op
    assert summary.adapted_convolutions > 0
    assert all(parameter.requires_grad for parameter in lora_parameters(model))
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert frozen > summary.trainable_parameters


def test_subspace_readout_complement_identity():
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_chart.prior_model import CanonicalPrior, ddim_denoise_subspace
    from eeg_chart.transport import ordered_frame

    torch.manual_seed(0)
    model = CanonicalPrior(base=32).eval()          # small twin, same structure
    schedule = LinearX0Schedule()
    y = torch.randn(2, 121, 512)
    noise = torch.randn(2, 121, 512)
    u = torch.from_numpy(ordered_frame(np.random.default_rng(1)
                                       .standard_normal((121, 2))).astype(np.float32))
    x0 = ddim_denoise_subspace(model, y, noise, schedule, u, inference_steps=5)
    correction = (y - x0).numpy()
    projector = (u @ u.T).numpy()
    complement = correction - np.einsum("kl,blt->bkt", projector, correction)
    # hard data consistency: the correction lies in span(U) — complement ~ 0
    assert np.max(np.abs(complement)) <= 1e-4
    # zero-init residual head => raw x0 = y => correction exactly 0 (q99 = 1 structurally)
    assert np.max(np.abs(correction)) <= 1e-4


def test_whitening_off_contracts():
    from eeg_chart.geodesic import transport_family
    from eeg_chart.transport import ordered_frame, minimal_rotation, sh_lift

    rng = np.random.default_rng(3)
    positions = rng.standard_normal((19, 3))
    positions[:, 2] = np.abs(positions[:, 2])
    lift = sh_lift(positions)
    lift_pinv = np.linalg.pinv(lift)
    K = lift.shape[0]
    u_canon = ordered_frame(rng.standard_normal((K, 2)))
    u_subject = ordered_frame(u_canon + 0.02 * rng.standard_normal((K, 2)))
    u_pop = ordered_frame(u_canon + 0.012 * rng.standard_normal((K, 2)))
    rotation = minimal_rotation(u_subject, u_canon)
    base = minimal_rotation(u_pop, u_canon)
    sigma = np.eye(K)
    arm = transport_family(lift, lift_pinv, sigma, sigma, rotation, base, 0.7,
                           whitening="off")
    assert np.allclose(arm.align, np.eye(K))
    assert np.max(np.abs(arm.pinv @ arm.transport - np.eye(19))) <= 1e-10
    zero = transport_family(lift, lift_pinv, sigma, sigma, rotation, base, 0.0,
                            whitening="off")
    pop = transport_family(lift, lift_pinv, sigma, None, base, base, 0.0)
    assert np.array_equal(zero.transport, pop.transport)


def test_block_coverage_calibrated():
    from eeg_chart.posterior import block_coverage

    rng = np.random.default_rng(0)
    shape = (46, 2)
    tau2 = np.full(shape, 0.25)
    coverages = []
    for _ in range(60):
        pop = np.zeros(shape)
        truth = pop + rng.normal(scale=np.sqrt(tau2))
        blocks = truth[None] + rng.normal(scale=0.3, size=(4,) + shape)
        coverages.append(block_coverage(pop, tau2, blocks))
    assert 0.70 <= float(np.mean(coverages)) <= 0.90


def test_block_coverage_misspecified_prior_deviates():
    from eeg_chart.posterior import block_coverage

    rng = np.random.default_rng(1)
    shape = (46, 2)
    coverages = []
    for _ in range(40):
        truth = rng.normal(scale=3.0, size=shape)      # much wider than the claimed prior
        blocks = truth[None] + rng.normal(scale=0.05, size=(4,) + shape)
        coverages.append(block_coverage(np.zeros(shape), np.full(shape, 1e-4), blocks))
    assert float(np.mean(coverages)) < 0.70            # shrinkage toward a wrong prior undercovers
