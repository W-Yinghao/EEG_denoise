from __future__ import annotations

import torch

from eeg_cgdr.experiments.sge_eb_bridge_v8_1 import oracle_ddim_roundtrip
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig,
    ArtifactSubspaceDiffusion,
    window_noise_bank,
)


def test_oracle_roundtrip_propagates_current_state() -> None:
    model = ArtifactSubspaceDiffusion(
        ArtifactSubspaceConfig(eeg_channels=4, signal_length=32, base_channels=8)
    )
    target = torch.randn(2, 2, 32)
    noise = torch.randn_like(target)
    assert oracle_ddim_roundtrip(model, target, noise) < 1.0e-4
    first = model._sequence()[0]
    alpha = model.alpha_bar[first]
    perturbed = alpha.sqrt() * target + (1 - alpha).sqrt() * noise + 0.1
    # A loop that reconstructs the analytic state at every step would erase
    # this perturbation before the second incoming reverse state.
    error, states = oracle_ddim_roundtrip(
        model, target, noise, initial_state=perturbed, return_state_trace=True
    )
    assert error < 1.0e-4
    second_alpha = model.alpha_bar[model._sequence()[1]]
    analytic_second = second_alpha.sqrt() * target + (1 - second_alpha).sqrt() * noise
    assert not torch.allclose(states[1], analytic_second, atol=1.0e-5, rtol=0.0)


def test_window_noise_bank_is_batch_size_invariant_and_window_unique() -> None:
    full = window_noise_bank(
        "study02/p11", 20260807, range(7), posterior_samples=8,
        signal_length=32, device="cpu",
    )
    first = window_noise_bank(
        "study02/p11", 20260807, range(3), posterior_samples=8,
        signal_length=32, device="cpu",
    )
    second = window_noise_bank(
        "study02/p11", 20260807, range(3, 7), posterior_samples=8,
        signal_length=32, device="cpu",
    )
    torch.testing.assert_close(full, torch.cat((first, second), dim=1))
    assert not torch.equal(full[:, 0], full[:, 1])


def test_sample_accepts_explicit_common_noise_bank() -> None:
    model = ArtifactSubspaceDiffusion(
        ArtifactSubspaceConfig(eeg_channels=4, signal_length=32, base_channels=8)
    ).eval()
    observed = torch.randn(3, 4, 32)
    basis, _ = torch.linalg.qr(torch.randn(3, 4, 2))
    condition = {
        "observed": observed,
        "basis": basis,
        "reliability": torch.ones(3),
        "rank_mask": torch.ones(3, 2, dtype=torch.bool),
        "valid_time_mask": torch.ones(3, 32, dtype=torch.bool),
    }
    bank = window_noise_bank(
        "study04/p28", 20260807, range(3), posterior_samples=1,
        signal_length=32, device="cpu",
    )
    left = model.sample(initial_noise_bank=bank, **condition)[0]
    right = model.sample(initial_noise_bank=bank.clone(), **condition)[0]
    torch.testing.assert_close(left, right)
