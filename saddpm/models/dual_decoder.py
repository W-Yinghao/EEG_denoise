"""Dual-decoder SADDPM model (handoff §1.2, §4).

A shared (subject-agnostic) encoder feeds two decoders:
  * content decoder ``D_c`` — subject-conditioned via FiLM(e(s)); predicts ``ε_θ``;
  * individual decoder ``D_s`` — not subject-conditioned; predicts ``ε_φ``.
Each decoder's penultimate 64-ch feature is global-average-pooled and projected to a d=128 vector
(``z^c`` / ``z^s``, [DD-8]). The reverse-step noise estimate is ``ε_θ + ε_φ`` ([DD-1]); an ArcFace
head classifies the subject from ``z^s``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .arcface import ArcFace
from .config import ModelConfig
from .subject_embed import SubjectEmbedding
from .unet1d import (
    AttentionBlock1D,
    Downsample1D,
    ResBlock1D,
    TimeEmbedding,
    Upsample1D,
)


def _block(in_ch: int, out_ch: int, n: int, tdim: int, groups: int, drop: float,
           sdim: Optional[int]) -> nn.ModuleList:
    blocks: List[nn.Module] = [ResBlock1D(in_ch, out_ch, tdim, groups, drop, sdim)]
    for _ in range(n - 1):
        blocks.append(ResBlock1D(out_ch, out_ch, tdim, groups, drop, sdim))
    return nn.ModuleList(blocks)


def _run(blocks: nn.ModuleList, h: Tensor, temb: Tensor, subj: Optional[Tensor]) -> Tensor:
    for blk in blocks:
        h = blk(h, temb, subj)
    return h


class Encoder(nn.Module):
    """Shared subject-agnostic encoder (down path + bottleneck with self-attention)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        g, d, tdim = cfg.groupnorm_groups, cfg.dropout, cfg.time_embed_dim
        w0, w1, w2 = (cfg.base_channels * m for m in cfg.channel_mults)
        self.stem = nn.Conv1d(cfg.in_channels, cfg.base_channels, kernel_size=5, padding=2)
        self.enc0 = _block(cfg.base_channels, w0, cfg.num_res_blocks, tdim, g, d, None)
        self.down0 = Downsample1D(w0)
        self.enc1 = _block(w0, w1, cfg.num_res_blocks, tdim, g, d, None)
        self.down1 = Downsample1D(w1)
        self.enc2 = _block(w1, w2, cfg.num_res_blocks, tdim, g, d, None)
        self.down2 = Downsample1D(w2)
        self.mid1 = ResBlock1D(w2, w2, tdim, g, d, None)
        self.mid_attn = AttentionBlock1D(w2, cfg.attention_heads, g)
        self.mid2 = ResBlock1D(w2, w2, tdim, g, d, None)

    def forward(self, x: Tensor, temb: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        s0 = _run(self.enc0, self.stem(x), temb, None)
        s1 = _run(self.enc1, self.down0(s0), temb, None)
        s2 = _run(self.enc2, self.down1(s1), temb, None)
        h = self.down2(s2)
        h = self.mid2(self.mid_attn(self.mid1(h, temb, None)), temb, None)
        return s0, s1, s2, h


class DecoderBranch(nn.Module):
    """Up path mirroring the encoder, optionally subject-conditioned, returning ``(ε, z)``."""

    def __init__(self, cfg: ModelConfig, subject_conditioned: bool) -> None:
        super().__init__()
        g, d, tdim = cfg.groupnorm_groups, cfg.dropout, cfg.time_embed_dim
        w0, w1, w2 = (cfg.base_channels * m for m in cfg.channel_mults)
        sdim = cfg.subject_embed_dim if subject_conditioned else None
        self.up2 = Upsample1D(w2)
        self.dec2 = _block(w2 + w2, w1, cfg.num_res_blocks, tdim, g, d, sdim)
        self.up1 = Upsample1D(w1)
        self.dec1 = _block(w1 + w1, w0, cfg.num_res_blocks, tdim, g, d, sdim)
        self.up0 = Upsample1D(w0)
        self.dec0 = _block(w0 + w0, w0, cfg.num_res_blocks, tdim, g, d, sdim)
        self.out_norm = nn.GroupNorm(g, w0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(w0, cfg.in_channels, kernel_size=3, padding=1)
        self.proj = nn.Linear(w0, cfg.subject_embed_dim)  # GAP(64) -> d=128 ([DD-8])

    def forward(
        self,
        skips: Tuple[Tensor, Tensor, Tensor],
        h: Tensor,
        temb: Tensor,
        subj: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        s0, s1, s2 = skips
        h = _run(self.dec2, torch.cat([self.up2(h), s2], dim=1), temb, subj)
        h = _run(self.dec1, torch.cat([self.up1(h), s1], dim=1), temb, subj)
        feat = _run(self.dec0, torch.cat([self.up0(h), s0], dim=1), temb, subj)
        eps = self.out_conv(self.out_act(self.out_norm(feat)))
        z = self.proj(feat.mean(dim=-1))  # global average pool over time
        return eps, z


class DualDecoderSADDPM(nn.Module):
    """Full SADDPM: shared encoder + content/individual decoders + ArcFace subject head."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_embed = TimeEmbedding(cfg.time_sinusoidal_dim, cfg.time_embed_dim)
        self.subject_embed = SubjectEmbedding(cfg.num_subjects, cfg.subject_embed_dim)
        self.encoder = Encoder(cfg)
        self.content = DecoderBranch(cfg, subject_conditioned=True)
        self.individual = DecoderBranch(cfg, subject_conditioned=False)
        self.arcface = ArcFace(
            cfg.subject_embed_dim, cfg.num_subjects, cfg.arcface_margin, cfg.arcface_scale
        )

    def forward(self, x: Tensor, t: Tensor, subject_ids: Tensor) -> Dict[str, Tensor]:
        """Return ``{eps_theta, eps_phi, z_c, z_s}`` for a batch.

        Args:
            x: noisy signal ``(B, 22, 512)``.
            t: ``(B,)`` timestep indices.
            subject_ids: ``(B,)`` 0-based subject ids (content branch + ArcFace).
        """
        temb = self.time_embed(t)
        subj = self.subject_embed(subject_ids)
        s0, s1, s2, h = self.encoder(x, temb)
        eps_theta, z_c = self.content((s0, s1, s2), h, temb, subj)
        eps_phi, z_s = self.individual((s0, s1, s2), h, temb, None)
        return {"eps_theta": eps_theta, "eps_phi": eps_phi, "z_c": z_c, "z_s": z_s}

    def predict_eps(self, x: Tensor, t: Tensor, subject_ids: Tensor) -> Tensor:
        """Combined reverse-step noise estimate ``ε_θ + ε_φ`` ([DD-1]) for sampling / SDEdit."""
        out = self.forward(x, t, subject_ids)
        return out["eps_theta"] + out["eps_phi"]
