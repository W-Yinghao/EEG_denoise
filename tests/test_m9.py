"""Unit tests for M9 (Option C): spatial topographies + subject-conditional denoiser."""

from __future__ import annotations

import numpy as np
import torch

from saddpm.data.synthetic_artifacts import emg_topography, eog_topography
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.models.cond_denoiser import SubjectConditionalDenoiser
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

CH = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
      "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]


def test_eog_topography_frontal_dominant() -> None:
    w = eog_topography(CH)
    assert w.shape == (22,)
    assert w[CH.index("Fz")] > w[CH.index("POz")]  # frontal > occipital
    assert w[CH.index("FC1")] == 1.0


def test_emg_topography_lateral_dominant() -> None:
    w = emg_topography(CH)
    assert w.shape == (22,)
    assert w[CH.index("C5")] > w[CH.index("Cz")]  # lateral > midline


def _tiny_denoiser(c: int):
    cfg = ModelConfig(in_channels=2 * c, out_channels=c, signal_length=64, base_channels=8,
                      time_sinusoidal_dim=16, time_embed_dim=32, attention_length=8,
                      attention_heads=4, num_subjects=5, subject_embed_dim=8)
    unet = UNet1D(cfg, subject_conditioned=True)
    return SubjectConditionalDenoiser(unet, GaussianDiffusion(DiffusionConfig(num_timesteps=20)))


def test_conditional_denoiser_loss_and_denoise() -> None:
    c = 4
    model = _tiny_denoiser(c).eval()
    clean = torch.randn(3, c, 64)
    corrupted = clean + 0.5 * torch.randn(3, c, 64)
    sid = torch.tensor([0, 1, 2])
    assert model.loss(clean, corrupted, sid).ndim == 0
    out = model.denoise(corrupted, sid, ddim_steps=3)
    assert out.shape == (3, c, 64) and torch.all(torch.isfinite(out))


def test_subject_id_changes_conditioning_path() -> None:
    """After perturbing embeddings + FiLM, different subject ids must give different denoised output."""
    c = 4
    model = _tiny_denoiser(c).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        model.unet.subject_embed.embed.weight.normal_()
        from saddpm.models.film import FiLM
        for m in model.unet.modules():
            if isinstance(m, FiLM):
                m.proj.weight.normal_(std=0.3)
    corrupted = torch.randn(2, c, 64)
    a = model.denoise(corrupted, torch.tensor([0, 0]), ddim_steps=4)
    b = model.denoise(corrupted, torch.tensor([1, 1]), ddim_steps=4)
    assert not torch.allclose(a, b, atol=1e-4)
