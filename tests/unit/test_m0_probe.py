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
