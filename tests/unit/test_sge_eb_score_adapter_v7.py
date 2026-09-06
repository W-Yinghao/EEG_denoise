from __future__ import annotations

import numpy as np
import torch

from eeg_cgdr.experiments.sge_eb_score_adapter_v7 import (
    _frequency_features,
    _principal_angle,
    _x0_from_v,
)
from eeg_cgdr.models.dynamic_transfer_diffusion import DynamicTransferDiffusion, DynamicTransferModelConfig
from eeg_cgdr.models.eb_score_adapter import DynamicTransferScoreAdapter, EBAdapterConfig, fir_adjoint, normalized_fir_response


def test_oracle_v_ddim_roundtrip() -> None:
    model = DynamicTransferDiffusion(DynamicTransferModelConfig(eeg_channels=3, width=8, blocks=1, timesteps=1000, ddim_steps=25))
    generator = torch.Generator().manual_seed(20260807)
    target = torch.randn((2, 3, 64), generator=generator)
    noise = torch.randn((2, 3, 64), generator=generator)
    restored = model.oracle_v_roundtrip(target, initial_noise=noise)
    error = torch.linalg.norm(restored - target) / torch.linalg.norm(target)
    assert float(error) < 1e-4


def test_v_parameterization_recovers_x0() -> None:
    generator = torch.Generator().manual_seed(7)
    target = torch.randn((2, 3, 32), generator=generator)
    noise = torch.randn((2, 3, 32), generator=generator)
    alpha = torch.full((2, 1, 1), 0.37)
    state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
    torch.testing.assert_close(_x0_from_v(state, velocity, alpha), target, rtol=1e-5, atol=1e-6)


def test_frequency_and_subspace_distances_are_zero_for_same_transfer() -> None:
    transfer = np.random.default_rng(3).normal(size=(5, 2, 9))
    magnitude, phase = _frequency_features(transfer)
    magnitude2, phase2 = _frequency_features(transfer.copy())
    np.testing.assert_allclose(magnitude, magnitude2)
    np.testing.assert_allclose(phase, phase2)
    assert _principal_angle(transfer, transfer.copy()) < 1e-5


def test_diagnostic_sampler_accepts_k1_without_relaxing_primary_k8() -> None:
    model = DynamicTransferDiffusion(DynamicTransferModelConfig(eeg_channels=2, width=8, blocks=1, timesteps=16, ddim_steps=4))
    observed = torch.zeros((1, 2, 16))
    transfer = torch.zeros((1, 2, 2, 3))
    reliability = torch.ones(1)
    mean, variance, calls = model.sample_k(observed=observed, transfer=transfer, reliability=reliability, sample_seeds=(11,))
    assert mean.shape == observed.shape
    assert variance.shape == observed.shape
    assert calls == 4
    try:
        model.sample(observed=observed, transfer=transfer, reliability=reliability, sample_seeds=(11,))
    except ValueError as error:
        assert "K=8" in str(error)
    else:
        raise AssertionError("primary sampler must remain fixed at K=8")


def test_zero_initialized_dynamic_score_adapter_is_identity_on_population_score() -> None:
    adapter = DynamicTransferScoreAdapter(EBAdapterConfig(eeg_channels=3, width=8, blocks=1))
    state = torch.randn(2, 3, 32)
    observed = torch.randn(2, 3, 32)
    transfer = torch.randn(2, 3, 2, 5)
    output = adapter(state, torch.tensor([100, 700]), observed=observed, delta_transfer=transfer)
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_dynamic_response_retains_lag_information() -> None:
    eeg = torch.zeros(1, 2, 32); eeg[..., 16] = 1
    center = torch.zeros(1, 2, 2, 5); center[..., 2] = torch.eye(2)
    delayed = torch.zeros_like(center); delayed[..., 4] = torch.eye(2)
    adapter = DynamicTransferScoreAdapter(EBAdapterConfig(eeg_channels=2, eog_channels=2, width=8, blocks=1))
    center_response, _ = adapter.dynamic_features(eeg, eeg, center)
    delayed_response, _ = adapter.dynamic_features(eeg, eeg, delayed)
    assert float((center_response - delayed_response).abs().max()) > 1e-3
    assert fir_adjoint(center, eeg).shape == (1, 2, 32)
