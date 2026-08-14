"""Canonical population prior: K=121 conditional diffusion U-Net (~28M params).

x0-parameterized with the observation-centred zero-init residual (V42R's
decisive fix, reused verbatim): x0_hat = y + residual(x_t, y, t), residual head
zero-initialized so the model starts at the identity route.  Population prior
only — no subject conditioning of any kind (conditioning-channel is closed).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from eeg_scad.models.calib_saddpm_cond_v42r import (BottleneckAttention, LinearX0Schedule,
                                                    ResidualBlock, time_embedding)
from eeg_chart.transport import K_CANONICAL


SAMPLES = 512


class CanonicalPrior(nn.Module):
    def __init__(self, channels: int = K_CANONICAL, samples: int = SAMPLES,
                 base: int = 192) -> None:
        super().__init__()
        widths = (base, base * 2, base * 3, base * 4)
        self.channels, self.samples = channels, samples
        self.time_mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv1d(2 * channels, widths[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, width in enumerate(widths):
            self.down_blocks.append(ResidualBlock(width, width))
            if index < len(widths) - 1:
                self.downsamples.append(nn.Conv1d(width, widths[index + 1], 4, stride=2, padding=1))
        self.middle1 = ResidualBlock(widths[-1], widths[-1])
        self.attention = BottleneckAttention(widths[-1])
        self.middle2 = ResidualBlock(widths[-1], widths[-1])
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current = widths[-1]
        for skip_width in reversed(widths[:-1]):
            self.upsamples.append(nn.ConvTranspose1d(current, skip_width, 4, stride=2, padding=1))
            self.up_blocks.append(ResidualBlock(2 * skip_width, skip_width))
            current = skip_width
        self.final_norm = nn.GroupNorm(math.gcd(current, 8), current)
        self.residual_head = nn.Conv1d(current, channels, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, x_t: Tensor, observed: Tensor, timestep: Tensor) -> Tensor:
        if x_t.shape != observed.shape or x_t.shape[1:] != (self.channels, self.samples):
            raise ValueError("CanonicalPrior expects matching BxKx512 tensors")
        embedding = self.time_mlp(time_embedding(timestep, 128))
        value = self.input(torch.cat((x_t, observed), dim=1))
        skips = []
        for index, block in enumerate(self.down_blocks):
            value = block(value, embedding)
            skips.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        value = self.middle2(self.attention(self.middle1(value, embedding)), embedding)
        for upsample, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            value = upsample(value)
            value = block(torch.cat((value, skip), dim=1), embedding)
        features = F.silu(self.final_norm(value))
        return observed + self.residual_head(features)


@torch.no_grad()
def ddim_denoise(model: CanonicalPrior, observed: Tensor, noise: Tensor,
                 schedule: LinearX0Schedule, inference_steps: int = 50) -> Tensor:
    current = noise.clone()
    timesteps = torch.linspace(len(schedule.beta) - 1, 0, inference_steps,
                               device=observed.device).round().long()
    for index, timestep in enumerate(timesteps):
        batch_t = timestep.expand(len(observed))
        predicted_x0 = model(current, observed, batch_t)
        alpha = schedule.alpha_bar[timestep]
        epsilon = (current - alpha.sqrt() * predicted_x0) / (1 - alpha).sqrt().clamp_min(1e-8)
        if index + 1 == len(timesteps):
            current = predicted_x0
        else:
            next_alpha = schedule.alpha_bar[timesteps[index + 1]]
            current = next_alpha.sqrt() * predicted_x0 + (1 - next_alpha).sqrt() * epsilon
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"nonfinite prior DDIM state at index {index}")
    return current


__all__ = ["CanonicalPrior", "SAMPLES", "ddim_denoise"]
