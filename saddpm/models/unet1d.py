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
from torch.nn import functional as F

from .config import ModelConfig
from .film import FiLM
from .subject_embed import SubjectEmbedding


def _canonical_time_mask(x: Tensor, valid_time_mask: Optional[Tensor]) -> Optional[Tensor]:
    """Return a boolean ``(B,1,L)`` mask for an internal U-Net feature map.

    ``None`` deliberately remains ``None`` so legacy callers that do not use
    padding retain the ordinary PyTorch module path.  Scientific CGDR callers
    pass an explicit mask, including an all-valid mask, through every layer.
    """

    if valid_time_mask is None:
        return None
    mask = torch.as_tensor(valid_time_mask, device=x.device)
    if mask.ndim == 2:
        mask = mask[:, None, :]
    if mask.shape != (x.shape[0], 1, x.shape[-1]):
        raise ValueError(
            "valid_time_mask must have shape (B,L) or (B,1,L) matching the feature map"
        )
    if mask.dtype != torch.bool:
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("numeric valid_time_mask must contain only 0/1")
        mask = mask.bool()
    if not bool(mask.flatten(start_dim=1).any(dim=1).all()):
        raise ValueError("every U-Net sample must retain at least one valid time point")
    return mask.detach()


def _apply_time_mask(x: Tensor, valid_time_mask: Optional[Tensor]) -> Tensor:
    if valid_time_mask is None:
        return x
    return x * valid_time_mask.to(dtype=x.dtype)


def _downsample_time_mask(valid_time_mask: Optional[Tensor]) -> Optional[Tensor]:
    """Propagate validity through a kernel-3, stride-2, padding-1 convolution."""

    if valid_time_mask is None:
        return None
    return F.max_pool1d(
        valid_time_mask.to(dtype=torch.float32),
        kernel_size=3,
        stride=2,
        padding=1,
    ).bool()


class MaskedGroupNorm1D(nn.GroupNorm):
    """GroupNorm that excludes invalid time samples from its statistics.

    Parameter names and shapes are identical to :class:`torch.nn.GroupNorm`,
    preserving checkpoint compatibility.  Invalid positions are zeroed after
    affine transformation so the learned bias cannot leak into neighbouring
    valid positions through a later convolution.
    """

    def forward(
        self,
        input: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if valid_time_mask is None:
            return super().forward(input)
        if input.ndim != 3:
            raise ValueError("MaskedGroupNorm1D expects (B,C,L) input")
        mask = _canonical_time_mask(input, valid_time_mask)
        assert mask is not None
        batch, channels, length = input.shape
        channels_per_group = channels // self.num_groups
        grouped = input.reshape(
            batch, self.num_groups, channels_per_group, length
        )
        weight = mask[:, None, :, :].to(dtype=input.dtype)
        count = (
            weight.sum(dim=(2, 3)) * float(channels_per_group)
        ).clamp_min(1.0)
        mean = (grouped * weight).sum(dim=(2, 3)) / count
        centered = grouped - mean[:, :, None, None]
        variance = (centered.square() * weight).sum(dim=(2, 3)) / count
        output = centered * torch.rsqrt(variance[:, :, None, None] + self.eps)
        output = output.reshape(batch, channels, length)
        if self.affine:
            output = (
                output * self.weight.reshape(1, channels, 1)
                + self.bias.reshape(1, channels, 1)
            )
        return _apply_time_mask(output, mask)


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
        self.norm1 = MaskedGroupNorm1D(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.film = FiLM(subject_emb_dim, out_channels) if subject_emb_dim is not None else None
        self.norm2 = MaskedGroupNorm1D(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        temb: Tensor,
        subj_emb: Optional[Tensor] = None,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mask = _canonical_time_mask(x, valid_time_mask)
        masked_x = _apply_time_mask(x, mask)
        h = self.conv1(self.act(self.norm1(masked_x, mask)))
        h = _apply_time_mask(h, mask)
        h = _apply_time_mask(h + self.time_proj(temb)[:, :, None], mask)
        if self.film is not None and subj_emb is not None:
            h = _apply_time_mask(self.film(h, subj_emb), mask)
        h = self.conv2(self.dropout(self.act(self.norm2(h, mask))))
        h = _apply_time_mask(h, mask)
        skip = _apply_time_mask(self.skip(masked_x), mask)
        return _apply_time_mask(h + skip, mask)


class AttentionBlock1D(nn.Module):
    """Multi-head self-attention over the time axis (applied at the bottleneck)."""

    def __init__(self, channels: int, num_heads: int, groups: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} not divisible by heads {num_heads}")
        self.num_heads = num_heads
        self.norm = MaskedGroupNorm1D(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mask = _canonical_time_mask(x, valid_time_mask)
        masked_x = _apply_time_mask(x, mask)
        b, c, length = x.shape
        head_dim = c // self.num_heads
        qkv = _apply_time_mask(self.qkv(self.norm(masked_x, mask)), mask)
        q, k, v = qkv.reshape(b, 3, self.num_heads, head_dim, length).unbind(dim=1)
        scale = 1.0 / math.sqrt(head_dim)
        attn = torch.einsum("bhdi,bhdj->bhij", q, k) * scale  # (B, heads, L, L)
        if mask is not None:
            key_is_valid = mask[:, :, None, :]
            attn = attn.masked_fill(~key_is_valid, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bhdj->bhdi", attn, v)  # (B, heads, head_dim, L)
        out = out.reshape(b, c, length)
        out = _apply_time_mask(self.proj(_apply_time_mask(out, mask)), mask)
        return _apply_time_mask(masked_x + out, mask)


class Downsample1D(nn.Module):
    """Stride-2 conv halving the time length."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(
        self,
        x: Tensor,
        valid_time_mask: Optional[Tensor] = None,
        output_valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mask = _canonical_time_mask(x, valid_time_mask)
        output = self.op(_apply_time_mask(x, mask))
        propagated = _downsample_time_mask(mask)
        if output_valid_time_mask is not None:
            propagated = _canonical_time_mask(output, output_valid_time_mask)
        return _apply_time_mask(output, propagated)


class Upsample1D(nn.Module):
    """Nearest-neighbour ×2 upsample followed by a smoothing conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: Tensor,
        valid_time_mask: Optional[Tensor] = None,
        output_valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mask = _canonical_time_mask(x, valid_time_mask)
        output_length = (
            output_valid_time_mask.shape[-1]
            if output_valid_time_mask is not None
            else 2 * x.shape[-1]
        )
        output = F.interpolate(
            _apply_time_mask(x, mask), size=output_length, mode="nearest"
        )
        propagated = None
        if mask is not None:
            propagated = F.interpolate(
                mask.to(dtype=torch.float32), size=output_length, mode="nearest"
            ).bool()
        if output_valid_time_mask is not None:
            target = _canonical_time_mask(output, output_valid_time_mask)
            propagated = target if propagated is None else propagated & target
        output = self.conv(_apply_time_mask(output, propagated))
        return _apply_time_mask(output, propagated)


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
        self.mid1 = ResBlock1D(w2, w2, tdim, groups, drop, sdim)
        self.mid_attn = AttentionBlock1D(w2, cfg.attention_heads, groups)
        self.mid2 = ResBlock1D(w2, w2, tdim, groups, drop, sdim)

        # Decoder (skip-concat at the first block of each level).
        self.up2 = Upsample1D(w2)
        self.dec2 = block(w2 + w2, w1)
        self.up1 = Upsample1D(w1)
        self.dec1 = block(w1 + w1, w0)
        self.up0 = Upsample1D(w0)
        self.dec0 = block(w0 + w0, w0)

        out_ch = cfg.out_channels if cfg.out_channels is not None else cfg.in_channels
        self.out_norm = MaskedGroupNorm1D(groups, w0)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(w0, out_ch, kernel_size=3, padding=1)

    @staticmethod
    def _run(
        blocks: nn.ModuleList,
        h: Tensor,
        temb: Tensor,
        subj: Optional[Tensor],
        valid_time_mask: Optional[Tensor],
    ) -> Tensor:
        for blk in blocks:
            h = blk(h, temb, subj, valid_time_mask)
        return h

    def _subject_emb(self, x: Tensor, subject_ids: Optional[Tensor]) -> Optional[Tensor]:
        """Resolve the subject embedding, defaulting to the null slot when conditioned but unset."""
        if not self.subject_conditioned:
            return None
        if subject_ids is None:
            subject_ids = self.subject_embed.null_ids(x.shape[0], x.device)
        return self.subject_embed(subject_ids)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        subject_ids: Optional[Tensor] = None,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict ``ε_θ(x_t, t, s)``.

        Args:
            x: noisy signal ``(B, 22, 512)``.
            t: ``(B,)`` timestep indices.
            subject_ids: optional ``(B,)`` 0-based subject ids; null embedding if omitted.
            valid_time_mask: optional ``(B,L)``/``(B,1,L)`` padding mask,
                propagated through normalization, convolution, resampling and
                bottleneck attention.

        Returns:
            Predicted noise ``(B, 22, 512)``.
        """
        mask0 = _canonical_time_mask(x, valid_time_mask)
        mask1 = _downsample_time_mask(mask0)
        mask2 = _downsample_time_mask(mask1)
        mask3 = _downsample_time_mask(mask2)
        temb = self.time_embed(t)
        subj = self._subject_emb(x, subject_ids)
        h = _apply_time_mask(self.stem(_apply_time_mask(x, mask0)), mask0)
        s0 = self._run(self.enc0, h, temb, subj, mask0)
        s1 = self._run(
            self.enc1,
            self.down0(s0, mask0, mask1),
            temb,
            subj,
            mask1,
        )
        s2 = self._run(
            self.enc2,
            self.down1(s1, mask1, mask2),
            temb,
            subj,
            mask2,
        )

        h = self.down2(s2, mask2, mask3)
        h = self.mid1(h, temb, subj, mask3)
        h = self.mid_attn(h, mask3)
        h = self.mid2(h, temb, subj, mask3)

        h = self._run(
            self.dec2,
            torch.cat([self.up2(h, mask3, mask2), s2], dim=1),
            temb,
            subj,
            mask2,
        )
        h = self._run(
            self.dec1,
            torch.cat([self.up1(h, mask2, mask1), s1], dim=1),
            temb,
            subj,
            mask1,
        )
        h = self._run(
            self.dec0,
            torch.cat([self.up0(h, mask1, mask0), s0], dim=1),
            temb,
            subj,
            mask0,
        )
        h = self.out_act(self.out_norm(h, mask0))
        return _apply_time_mask(self.out_conv(_apply_time_mask(h, mask0)), mask0)

    def forward_with_subject_embedding(
        self,
        x: Tensor,
        t: Tensor,
        subject_embedding: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run all FiLM-enabled ResBlocks from a support-derived embedding."""

        if not self.subject_conditioned:
            raise ValueError("support FiLM requires a FiLM-enabled U-Net")
        if subject_embedding.shape != (x.shape[0], self.cfg.subject_embed_dim):
            raise ValueError("support embedding shape differs from U-Net FiLM width")
        if not bool(torch.isfinite(subject_embedding).all()):
            raise ValueError("support embedding contains NaN/Inf")
        mask0 = _canonical_time_mask(x, valid_time_mask)
        mask1 = _downsample_time_mask(mask0)
        mask2 = _downsample_time_mask(mask1)
        mask3 = _downsample_time_mask(mask2)
        temb = self.time_embed(t)
        subj = subject_embedding
        h = _apply_time_mask(self.stem(_apply_time_mask(x, mask0)), mask0)
        s0 = self._run(self.enc0, h, temb, subj, mask0)
        s1 = self._run(self.enc1, self.down0(s0, mask0, mask1), temb, subj, mask1)
        s2 = self._run(self.enc2, self.down1(s1, mask1, mask2), temb, subj, mask2)
        h = self.down2(s2, mask2, mask3)
        h = self.mid1(h, temb, subj, mask3)
        h = self.mid_attn(h, mask3)
        h = self.mid2(h, temb, subj, mask3)
        h = self._run(self.dec2, torch.cat([self.up2(h, mask3, mask2), s2], dim=1), temb, subj, mask2)
        h = self._run(self.dec1, torch.cat([self.up1(h, mask2, mask1), s1], dim=1), temb, subj, mask1)
        h = self._run(self.dec0, torch.cat([self.up0(h, mask1, mask0), s0], dim=1), temb, subj, mask0)
        h = self.out_act(self.out_norm(h, mask0))
        return _apply_time_mask(self.out_conv(_apply_time_mask(h, mask0)), mask0)
