"""Unit tests for the dual-decoder SADDPM, ArcFace head, and the 3 losses (M4)."""

from __future__ import annotations

import torch

from saddpm.losses.arcface_loss import arcface_loss
from saddpm.losses.orthogonality import orthogonality_loss
from saddpm.models.arcface import ArcFace
from saddpm.models.config import ModelConfig
from saddpm.models.dual_decoder import DualDecoderSADDPM


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
        in_channels=4, signal_length=64, base_channels=8, channel_mults=[1, 2, 4],
        num_res_blocks=2, groupnorm_groups=8, dropout=0.0, time_sinusoidal_dim=16,
        time_embed_dim=32, attention_length=8, attention_heads=4, num_subjects=5,
        subject_embed_dim=16, arcface_margin=0.5, arcface_scale=30.0,
    )


def test_dual_decoder_forward_shapes() -> None:
    cfg = _tiny_cfg()
    model = DualDecoderSADDPM(cfg).eval()
    x = torch.randn(3, cfg.in_channels, cfg.signal_length)
    t = torch.randint(0, 1000, (3,))
    sid = torch.tensor([0, 1, 2])
    with torch.no_grad():
        out = model(x, t, sid)
    assert out["eps_theta"].shape == x.shape
    assert out["eps_phi"].shape == x.shape
    assert out["z_c"].shape == (3, cfg.subject_embed_dim)
    assert out["z_s"].shape == (3, cfg.subject_embed_dim)
    assert model.predict_eps(x, t, sid).shape == x.shape


def test_individual_branch_is_subject_invariant() -> None:
    """z_s comes from the non-conditioned individual decoder, so subject_ids must not change it."""
    cfg = _tiny_cfg()
    model = DualDecoderSADDPM(cfg).eval()
    x = torch.randn(2, cfg.in_channels, cfg.signal_length)
    t = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        a = model(x, t, torch.tensor([0, 0]))
        b = model(x, t, torch.tensor([1, 2]))
    assert torch.allclose(a["z_s"], b["z_s"], atol=1e-6)
    assert torch.allclose(a["eps_phi"], b["eps_phi"], atol=1e-6)


def test_orthogonality_loss_zero_and_nonneg() -> None:
    z = torch.randn(8, 16)
    assert orthogonality_loss(torch.zeros(8, 16), z).item() == 0.0
    assert orthogonality_loss(z, torch.randn(8, 16)).item() >= 0.0


def test_arcface_margin_reduces_target_logit() -> None:
    torch.manual_seed(0)
    head = ArcFace(in_features=16, num_classes=5, margin=0.5, scale=30.0)
    feat = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    plain = head.logits(feat)
    margined = head(feat, labels)
    # the margin lowers the logit of the true class relative to the plain cosine logit.
    for i, y in enumerate(labels):
        assert margined[i, y] <= plain[i, y] + 1e-4


def test_arcface_loss_scalar() -> None:
    head = ArcFace(in_features=16, num_classes=5)
    loss = arcface_loss(head, torch.randn(6, 16), torch.randint(0, 5, (6,)))
    assert loss.ndim == 0 and torch.isfinite(loss)
