"""Minimal multichannel port of the frozen official EEGDfus dual-branch model.

The port keeps epsilon prediction, the dual noisy/conditional streams, four timestep
bridges, and the official linear schedule. Only the channel interface and the fixed
temporal model dimension change (1->46 channels and 512->256 samples).
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class TemporalEncoder(nn.Module):
    def __init__(self, samples: int, heads: int = 1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(samples, heads, batch_first=True)
        self.n1 = nn.LayerNorm(samples)
        self.ff = nn.Sequential(nn.Linear(samples, samples), nn.ReLU(), nn.Linear(samples, samples))
        self.n2 = nn.LayerNorm(samples)

    def forward(self, value: Tensor) -> Tensor:
        a, _ = self.attn(value, value, value, need_weights=False)
        value = self.n1(value + a)
        return self.n2(value + self.ff(value))


class NoiseBridge(nn.Module):
    def __init__(self, samples: int, features: int) -> None:
        super().__init__()
        self.samples = samples
        self.features = features
        self.gamma = nn.Linear(features, samples)
        self.beta = nn.Linear(features, samples)

    def forward(self, value: Tensor, embedding: Tensor) -> Tensor:
        return self.gamma(embedding)[:, None] * value + self.beta(embedding)[:, None]


def noise_embedding(noise_level: Tensor, features: int) -> Tensor:
    value = noise_level.reshape(-1)
    half = features // 2
    step = torch.arange(half, device=value.device, dtype=value.dtype) / max(half, 1)
    angle = value[:, None] * torch.exp(-math.log(1e4) * step[None])
    return torch.cat((angle.sin(), angle.cos()), dim=1)


class ZeroContextFiLM(nn.Module):
    """Identity-at-initialization context adapter."""

    def __init__(self, context_dim: int, features: int) -> None:
        super().__init__()
        self.proj = nn.Linear(context_dim, 2 * features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, value: Tensor, context: Tensor) -> Tensor:
        scale, shift = self.proj(context).chunk(2, 1)
        return value * (1 + scale[:, :, None]) + shift[:, :, None]


class EEGDfusMC(nn.Module):
    """Official-shape dual branch with two optional support injection sites."""

    def __init__(self, channels: int = 46, samples: int = 256, features: int = 64, context_dim: int = 128) -> None:
        super().__init__()
        self.channels, self.samples, self.features = channels, samples, features
        conv = lambda: nn.Sequential(nn.Conv1d(channels, features, 3, padding=1), nn.Conv1d(features, features, 3, padding=1))
        self.stream_x = nn.ModuleList([conv(), TemporalEncoder(samples), TemporalEncoder(samples), TemporalEncoder(samples)])
        self.stream_cond = nn.ModuleList([conv(), TemporalEncoder(samples), TemporalEncoder(samples), TemporalEncoder(samples)])
        self.bridges = nn.ModuleList([NoiseBridge(samples, features) for _ in range(4)])
        self.support_mid = ZeroContextFiLM(context_dim, features)
        self.support_late = ZeroContextFiLM(context_dim, features)
        self.out = nn.Sequential(nn.Conv1d(features, features, 3, padding=1), nn.Conv1d(features, channels, 3, padding=1))

    def forward(self, noisy: Tensor, observed: Tensor, noise_level: Tensor, context: Tensor | None = None, bypass: bool = False) -> Tensor:
        emb = noise_embedding(noise_level, self.features)
        skips = []
        value = noisy
        for index, (layer, bridge) in enumerate(zip(self.stream_x, self.bridges)):
            value = layer(value)
            if index == 1 and context is not None and not bypass:
                value = self.support_mid(value, context)
            skips.append(bridge(value, emb))
        value = observed
        for index, (skip, layer) in enumerate(zip(skips, self.stream_cond)):
            value = layer(value) + skip
            if index == 2 and context is not None and not bypass:
                value = self.support_late(value, context)
        return self.out(value)

    def freeze_population(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.support_mid, self.support_late):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


class CompactSupportEncoder(nn.Module):
    """Per-window temporal CNN, global pool, set mean, and 128-d MLP."""

    def __init__(self, eeg_channels: int = 46, eog_channels: int = 4, context_dim: int = 128) -> None:
        super().__init__()
        inputs = eeg_channels + eog_channels
        self.window = nn.Sequential(
            nn.Conv1d(inputs, 64, 7, padding=3), nn.SiLU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv1d(96, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, context_dim))

    def forward(self, eeg: Tensor, eog: Tensor) -> Tensor:
        if eeg.ndim != 4 or eog.ndim != 4 or eeg.shape[:2] != eog.shape[:2]:
            raise ValueError("support must have shape B,W,C,T with matched B/W")
        batch, windows = eeg.shape[:2]
        token = self.window(torch.cat((eeg, eog), 2).reshape(batch * windows, -1, eeg.shape[-1])).squeeze(-1)
        return self.mlp(token.reshape(batch, windows, -1).mean(1))


class LinearSchedule(nn.Module):
    def __init__(self, steps: int = 500, beta_start: float = 1e-4, beta_end: float = .02) -> None:
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, steps)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))

    def q_sample(self, clean: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        alpha = self.alpha_bar[timestep][:, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise


@torch.no_grad()
def ddim_sample(model: EEGDfusMC, observed: Tensor, noise: Tensor, steps: int = 25, context: Tensor | None = None, bypass: bool = False, schedule: LinearSchedule | None = None) -> Tensor:
    schedule = schedule or LinearSchedule().to(observed.device)
    indices = torch.linspace(len(schedule.alpha_bar) - 1, 0, steps, device=observed.device).long()
    current = noise
    for index, timestep in enumerate(indices):
        t = torch.full((len(observed),), int(timestep), device=observed.device, dtype=torch.long)
        alpha = schedule.alpha_bar[timestep]
        level = alpha.sqrt().expand(len(observed), 1)
        eps = model(current, observed, level, context=context, bypass=bypass)
        x0 = (current - (1 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(1e-6)
        if index + 1 == len(indices):
            current = x0
        else:
            next_alpha = schedule.alpha_bar[indices[index + 1]]
            current = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * eps
    return current


__all__ = ["CompactSupportEncoder", "EEGDfusMC", "LinearSchedule", "ddim_sample", "noise_embedding"]
