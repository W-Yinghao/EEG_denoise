"""Unit tests for subject conditioning (FiLM, embeddings), DDIM, and subject correlation (M3)."""

from __future__ import annotations

import numpy as np
import torch

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.eval.subject_corr import (
    descriptors_from_groups,
    diagonal_dominance,
    pearson_matrix,
    spectral_descriptor,
)
from saddpm.models.config import ModelConfig
from saddpm.models.film import FiLM
from saddpm.models.subject_embed import SubjectEmbedding
from saddpm.models.unet1d import UNet1D


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        in_channels=4, signal_length=64, base_channels=8, channel_mults=[1, 2, 4],
        num_res_blocks=2, groupnorm_groups=8, dropout=0.0, time_sinusoidal_dim=16,
        time_embed_dim=32, attention_length=8, attention_heads=4, num_subjects=5, subject_embed_dim=16,
    )


def test_film_is_identity_at_init() -> None:
    film = FiLM(emb_dim=16, channels=8)
    h = torch.randn(3, 8, 20)
    emb = torch.randn(3, 16)
    assert torch.allclose(film(h, emb), h, atol=1e-6)


def test_subject_embedding_shapes_and_null() -> None:
    se = SubjectEmbedding(num_subjects=5, emb_dim=16)
    out = se(torch.tensor([0, 4]))
    assert out.shape == (2, 16)
    null = se.null_ids(3, torch.device("cpu"))
    assert null.tolist() == [5, 5, 5]


def test_unet_subject_forward_shape() -> None:
    cfg = _tiny_cfg()
    model = UNet1D(cfg, subject_conditioned=True).eval()
    x = torch.randn(2, cfg.in_channels, cfg.signal_length)
    t = torch.randint(0, 1000, (2,))
    sid = torch.tensor([0, 1])
    with torch.no_grad():
        assert model(x, t, sid).shape == x.shape
        # works without subject ids (null embedding) too.
        assert model(x, t).shape == x.shape


def test_conditioning_changes_output_when_embeddings_differ() -> None:
    cfg = _tiny_cfg()
    model = UNet1D(cfg, subject_conditioned=True).eval()
    # FiLM is identity at init; simulate a trained model by randomizing embeddings + FiLM proj.
    torch.manual_seed(0)
    with torch.no_grad():
        model.subject_embed.embed.weight.normal_()
        for m in model.modules():
            if isinstance(m, FiLM):
                m.proj.weight.normal_(std=0.1)
    x = torch.randn(2, cfg.in_channels, cfg.signal_length)
    t = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        out0 = model(x, t, torch.tensor([0, 0]))
        out1 = model(x, t, torch.tensor([1, 1]))
    assert not torch.allclose(out0, out1, atol=1e-4)


def test_ddim_sample_loop_shape() -> None:
    model = UNet1D(_tiny_cfg(), subject_conditioned=True).eval()
    diff = GaussianDiffusion(DiffusionConfig(num_timesteps=100))
    eps_fn = lambda x, t: model(x, t, torch.zeros(x.shape[0], dtype=torch.long))  # noqa: E731
    out = diff.ddim_sample_loop(eps_fn, (2, 4, 64), torch.device("cpu"), ddim_steps=5)
    assert out.shape == (2, 4, 64) and torch.all(torch.isfinite(out))


def test_pearson_and_diagonal_dominance() -> None:
    rng = np.random.default_rng(0)
    desc = rng.normal(size=(5, 200))
    corr = pearson_matrix(desc, desc)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)
    assert diagonal_dominance(corr) == 1.0


def test_spectral_descriptor_shape() -> None:
    windows = np.random.randn(10, 22, 512).astype(np.float32)
    d = spectral_descriptor(windows)
    assert d.shape == (22 * (512 // 2 + 1),)
    stacked = descriptors_from_groups([windows, windows])
    assert stacked.shape == (2, 22 * 257)
