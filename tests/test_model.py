"""Unit tests for the 1D U-Net and DDPM reverse sampling (handoff §4, §7)."""

from __future__ import annotations

from pathlib import Path

import torch

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.losses.recon import reconstruction_loss
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import AttentionBlock1D, UNet1D, timestep_embedding

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_model_cfg(**overrides) -> ModelConfig:
    base = dict(
        in_channels=4,
        signal_length=64,
        base_channels=8,
        channel_mults=[1, 2, 4],
        num_res_blocks=2,
        groupnorm_groups=8,
        dropout=0.0,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        attention_length=8,
        attention_heads=4,
    )
    base.update(overrides)
    return ModelConfig(**base)


def test_unet_forward_shape_full_config() -> None:
    cfg = ModelConfig.from_yaml(REPO_ROOT / "configs" / "model.yaml")
    model = UNet1D(cfg).eval()
    x = torch.randn(2, cfg.in_channels, cfg.signal_length)
    t = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == (2, cfg.in_channels, cfg.signal_length)


def test_timestep_embedding_shape() -> None:
    emb = timestep_embedding(torch.arange(5), dim=128)
    assert emb.shape == (5, 128)


def test_attention_block_preserves_shape() -> None:
    attn = AttentionBlock1D(channels=32, num_heads=4, groups=8)
    x = torch.randn(3, 32, 16)
    assert attn(x).shape == x.shape


def test_p_sample_loop_shape() -> None:
    model = UNet1D(_tiny_model_cfg()).eval()
    diff = GaussianDiffusion(DiffusionConfig(num_timesteps=5))
    x0 = diff.p_sample_loop(lambda x, t: model(x, t), shape=(2, 4, 64), device=torch.device("cpu"))
    assert x0.shape == (2, 4, 64)
    assert torch.all(torch.isfinite(x0))


def test_overfit_single_batch_decreases_loss() -> None:
    """Smoke test of the full training path: a tiny U-Net memorizes one fixed (x_t, t, ε) example.

    This verifies q_sample -> model -> MSE -> backward learns. The harder 'overfit one batch
    across all t -> ~0' is the M2 milestone gate run on GPU (scripts/m2_overfit_batch.py).
    """
    torch.manual_seed(0)
    model = UNet1D(_tiny_model_cfg())
    diff = GaussianDiffusion(DiffusionConfig(num_timesteps=50))
    x0 = torch.randn(4, 4, 64)
    t = torch.full((4,), 10, dtype=torch.long)  # fixed timestep
    noise = torch.randn_like(x0)  # fixed target noise
    xt = diff.q_sample(x0, t, noise)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(200):
        loss = reconstruction_loss(model(xt, t), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 0.5, f"unexpectedly low initial loss {losses[0]:.3f}"
    assert losses[-1] < 0.1, f"failed to memorize fixed example: {losses[0]:.3f} -> {losses[-1]:.3f}"
