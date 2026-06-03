"""1D U-Net backbone for EEG diffusion (handoff §4).

M2 scope: single decoder, time-conditioned only (no subject embedding / FiLM yet). Structure is
the explicit 3-level encoder/decoder mandated by the spec:

    stem(22->64) @512
    enc0 @512 (64)  -down->  enc1 @256 (128)  -down->  enc2 @128 (256)  -down-> bottleneck @64 (256)
    bottleneck: ResBlock -> self-attention (len 64, [DD-7]) -> ResBlock
    up-> dec2 @128 -> up-> dec1 @256 -> up-> dec0 @512  (skip-concat at each level)
    head(64->22) @512

ResBlock order (paper §4): GN -> SiLU -> Conv -> +time-emb -> GN -> SiLU -> Dropout -> Conv + residual.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .film import FiLM
from .subject_embed import SubjectEmbedding


def timestep_embedding(timesteps: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Sinusoidal timestep embedding (Vaswani et al. / Ho et al.).

    Args:
        timesteps: ``(B,)`` long/float tensor of diffusion timestep indices.
        dim: embedding dimension.
        max_period: controls the minimum frequency.

    Returns:
        ``(B, dim)`` float tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding followed by an MLP (128 -> 512 -> 512, SiLU)."""

    def __init__(self, sinusoidal_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(timestep_embedding(t, self.sinusoidal_dim))


class ResBlock1D(nn.Module):
    """Residual block with time-embedding injection and optional subject FiLM (handoff §4).

    Order: GN → SiLU → Conv → +time-emb → FiLM(subject) → GN → SiLU → Dropout → Conv + residual.
    FiLM is built only when ``subject_emb_dim`` is given; it is applied only when a subject
    embedding is passed to :meth:`forward` (so the block degrades to the plain M2 behaviour).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        groups: int,
        dropout: float,
        subject_emb_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.film = FiLM(subject_emb_dim, out_channels) if subject_emb_dim is not None else None
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, temb: Tensor, subj_emb: Optional[Tensor] = None) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(temb)[:, :, None]
        if self.film is not None and subj_emb is not None:
            h = self.film(h, subj_emb)
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock1D(nn.Module):
    """Multi-head self-attention over the time axis (applied at the bottleneck)."""

    def __init__(self, channels: int, num_heads: int, groups: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, length = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv(self.norm(x))  # (B, 3C, L)
        q, k, v = qkv.reshape(b, 3, self.num_heads, head_dim, length).unbind(dim=1)
        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * scale  # (B, heads, L, L)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)  # (B, heads, head_dim, L)
        out = out.reshape(b, c, length)
        return x + self.proj(out)


class Downsample1D(nn.Module):
    """Stride-2 conv halving the time length."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


class Upsample1D(nn.Module):
    """Nearest-neighbour ×2 upsample followed by a smoothing conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet1D(nn.Module):
    """1D U-Net predicting ``ε_θ(x_t, t, s)`` for a 22-channel EEG window.

    When ``subject_conditioned`` is True (default, M3+), a learnable subject embedding modulates
    every ResBlock via FiLM. Calling ``forward(x, t)`` without subject ids falls back to the null
    embedding (the ``--no_subject`` ablation, [DD-4]), which also keeps the plain M2 call working.
    """

    def __init__(self, cfg: ModelConfig, subject_conditioned: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.subject_conditioned = subject_conditioned
        groups, drop, tdim = cfg.groupnorm_groups, cfg.dropout, cfg.time_embed_dim
        sdim = cfg.subject_embed_dim if subject_conditioned else None
        widths = [cfg.base_channels * m for m in cfg.channel_mults]
        if len(widths) != 3:
            raise ValueError("this explicit U-Net expects exactly 3 channel mults")
        w0, w1, w2 = widths

        self.time_embed = TimeEmbedding(cfg.time_sinusoidal_dim, tdim)
        self.subject_embed = (
            SubjectEmbedding(cfg.num_subjects, cfg.subject_embed_dim) if subject_conditioned else None
        )
        self.stem = nn.Conv1d(cfg.in_channels, cfg.base_channels, kernel_size=5, padding=2)

        def block(in_ch: int, out_ch: int) -> nn.ModuleList:
            blocks: List[nn.Module] = [ResBlock1D(in_ch, out_ch, tdim, groups, drop, sdim)]
            for _ in range(cfg.num_res_blocks - 1):
                blocks.append(ResBlock1D(out_ch, out_ch, tdim, groups, drop, sdim))
            return nn.ModuleList(blocks)

        # Encoder.
        self.enc0 = block(cfg.base_channels, w0)
        self.down0 = Downsample1D(w0)
        self.enc1 = block(w0, w1)
        self.down1 = Downsample1D(w1)
        self.enc2 = block(w1, w2)
        self.down2 = Downsample1D(w2)

        # Bottleneck @ length 64 with self-attention ([DD-7]).
        self.mid1 = ResBlock1D(w2, w2, tdim, groups, drop)
        self.mid_attn = AttentionBlock1D(w2, cfg.attention_heads, groups)
        self.mid2 = ResBlock1D(w2, w2, tdim, groups, drop)

        # Decoder (skip-concat at the first block of each level).
        self.up2 = Upsample1D(w2)
        self.dec2 = block(w2 + w2, w1)
        self.up1 = Upsample1D(w1)
        self.dec1 = block(w1 + w1, w0)
        self.up0 = Upsample1D(w0)
        self.dec0 = block(w0 + w0, w0)

        out_ch = cfg.out_channels if cfg.out_channels is not None else cfg.in_channels
        self.out_norm = nn.GroupNorm(groups, w0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(w0, out_ch, kernel_size=3, padding=1)

    @staticmethod
    def _run(blocks: nn.ModuleList, h: Tensor, temb: Tensor, subj: Optional[Tensor]) -> Tensor:
        for blk in blocks:
            h = blk(h, temb, subj)
        return h

    def _subject_emb(self, x: Tensor, subject_ids: Optional[Tensor]) -> Optional[Tensor]:
        """Resolve the subject embedding, defaulting to the null slot when conditioned but unset."""
        if not self.subject_conditioned:
            return None
        if subject_ids is None:
            subject_ids = self.subject_embed.null_ids(x.shape[0], x.device)
        return self.subject_embed(subject_ids)

    def forward(self, x: Tensor, t: Tensor, subject_ids: Optional[Tensor] = None) -> Tensor:
        """Predict ``ε_θ(x_t, t, s)``.

        Args:
            x: noisy signal ``(B, 22, 512)``.
            t: ``(B,)`` timestep indices.
            subject_ids: optional ``(B,)`` 0-based subject ids; null embedding if omitted.

        Returns:
            Predicted noise ``(B, 22, 512)``.
        """
        temb = self.time_embed(t)
        subj = self._subject_emb(x, subject_ids)
        h = self.stem(x)
        s0 = self._run(self.enc0, h, temb, subj)
        s1 = self._run(self.enc1, self.down0(s0), temb, subj)
        s2 = self._run(self.enc2, self.down1(s1), temb, subj)

        h = self.down2(s2)
        h = self.mid2(self.mid_attn(self.mid1(h, temb, subj)), temb, subj)

        h = self._run(self.dec2, torch.cat([self.up2(h), s2], dim=1), temb, subj)
        h = self._run(self.dec1, torch.cat([self.up1(h), s1], dim=1), temb, subj)
        h = self._run(self.dec0, torch.cat([self.up0(h), s0], dim=1), temb, subj)
        return self.out_conv(self.out_act(self.out_norm(h)))
