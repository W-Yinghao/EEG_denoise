"""Unit tests for EEGdenoiseNet (M8): metrics, pairing, baselines, conditional diffusion."""

from __future__ import annotations

import numpy as np
import torch

from saddpm.baselines.dl_denoisers import make_denoiser
from saddpm.diffusion.conditional import ConditionalDiffusionDenoiser
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.eval.denoise_metrics import (
    correlation_coefficient,
    rrmse_spectral,
    rrmse_temporal,
)
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D


def test_metrics_perfect_and_scaled() -> None:
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(20, 256)).astype(np.float64)
    assert rrmse_temporal(clean, clean).mean() < 1e-9
    assert rrmse_spectral(clean, clean, fs=256).mean() < 1e-9
    np.testing.assert_allclose(correlation_coefficient(clean, clean), 1.0, atol=1e-6)
    # a positively-scaled copy keeps CC=1 (scale-invariant) but RRMSE_t > 0.
    assert correlation_coefficient(2.0 * clean, clean).mean() > 0.999
    assert rrmse_temporal(2.0 * clean, clean).mean() > 0.5


def test_baseline_denoiser_shapes() -> None:
    for name in ("fcnn", "simple_cnn", "novel_cnn"):
        model = make_denoiser(name, 512).eval()
        with torch.no_grad():
            assert model(torch.randn(3, 1, 512)).shape == (3, 512)


def test_conditional_diffusion_shapes() -> None:
    cfg = ModelConfig(in_channels=2, out_channels=1, signal_length=64, base_channels=8,
                      time_sinusoidal_dim=16, time_embed_dim=32, attention_length=8,
                      attention_heads=4, subject_embed_dim=8)
    unet = UNet1D(cfg, subject_conditioned=False)
    cdd = ConditionalDiffusionDenoiser(unet, GaussianDiffusion(DiffusionConfig(num_timesteps=20)))
    clean, noisy = torch.randn(2, 1, 64), torch.randn(2, 1, 64)
    assert cdd.loss(clean, noisy).ndim == 0
    assert cdd.denoise(noisy, ddim_steps=3).shape == (2, 1, 64)


def test_unet_out_channels_override() -> None:
    cfg = ModelConfig(in_channels=2, out_channels=1, signal_length=64, base_channels=8,
                      time_sinusoidal_dim=16, time_embed_dim=32, attention_length=8,
                      attention_heads=4, subject_embed_dim=8)
    model = UNet1D(cfg, subject_conditioned=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 2, 64), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 1, 64)
