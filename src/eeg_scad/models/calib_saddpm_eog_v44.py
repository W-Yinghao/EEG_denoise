"""V44 EOG-guided joint conditional diffusion (a0-anchored).

Minimal modification of ``calib_saddpm_cond_v42r`` in a new file (the V42R
model is untouched): the network additionally receives a 46-channel artifact
estimate ``a0`` as input channels and predicts

    x0_hat = (y - a0) + Delta_pop + Delta_transfer

so ``a0 = 0`` reduces exactly to the V42R/V43 conditioning-only class (both
residual heads are zero-initialized, hence the prediction at initialization is
the anchored observation ``y - a0``).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from eeg_scad.models.calib_saddpm_cond_v42r import (
    EEG_CHANNELS, SAMPLES, TRANSFER_DIM, BottleneckAttention, LinearX0Schedule,
    ResidualBlock, TransferResidualDecoder, TransferStateEncoder, time_embedding)


class CalibSADDPMEOG(nn.Module):
    """V42R backbone with an a0 anchor channel block."""

    def __init__(self, channels: int = EEG_CHANNELS, samples: int = SAMPLES, base: int = 32) -> None:
        super().__init__()
        if samples != SAMPLES or channels < 1:
            raise ValueError("V44 models require one or more channels and 512 samples")
        import math
        widths = (base, base * 2, base * 3, base * 4)
        self.channels, self.samples = channels, samples
        self.time_mlp = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv1d(3 * channels, widths[0], 3, padding=1)
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
        self.population_head = nn.Conv1d(current, channels, 1)
        nn.init.zeros_(self.population_head.weight)
        nn.init.zeros_(self.population_head.bias)
        self.transfer_encoder = TransferStateEncoder(channels)
        self.transfer_decoder = TransferResidualDecoder(current, channels=channels)

    def forward(self, x_t: Tensor, observed: Tensor, a0: Tensor, timestep: Tensor,
                transfer: Tensor, transfer_enabled: bool = True) -> Tensor:
        if x_t.shape != observed.shape or x_t.shape != a0.shape or \
                x_t.shape[1:] != (self.channels, self.samples):
            raise ValueError("V44 expects matching Bx46x512 diffusion/observation/a0 tensors")
        embedding = self.time_mlp(time_embedding(timestep, 128))
        value = self.input(torch.cat((x_t, observed, a0), dim=1))
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
        population = self.population_head(features)
        if transfer_enabled:
            context, sensor_gate = self.transfer_encoder(transfer)
            personalized = self.transfer_decoder(features, context, sensor_gate)
        else:
            personalized = torch.zeros_like(population)
        return (observed - a0) + population + personalized

    def transfer_parameters(self):
        yield from self.transfer_encoder.parameters()
        yield from self.transfer_decoder.parameters()


@torch.no_grad()
def ddim_sample_eog(model: CalibSADDPMEOG, observed: Tensor, a0: Tensor, transfer: Tensor,
                    noise: Tensor, schedule: LinearX0Schedule, inference_steps: int = 50,
                    transfer_enabled: bool = True) -> Tensor:
    """Deterministic eta=0 DDIM with observation and a0 available at every step."""
    current = noise.clone()
    timesteps = torch.linspace(len(schedule.beta) - 1, 0, inference_steps,
                               device=observed.device).round().long()
    for index, timestep in enumerate(timesteps):
        batch_t = timestep.expand(len(observed))
        predicted_x0 = model(current, observed, a0, batch_t, transfer, transfer_enabled=transfer_enabled)
        alpha = schedule.alpha_bar[timestep]
        epsilon = (current - alpha.sqrt() * predicted_x0) / (1 - alpha).sqrt().clamp_min(1e-8)
        if index + 1 == len(timesteps):
            current = predicted_x0
        else:
            next_alpha = schedule.alpha_bar[timesteps[index + 1]]
            current = next_alpha.sqrt() * predicted_x0 + (1 - next_alpha).sqrt() * epsilon
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"nonfinite V44 DDIM state at index {index}")
    return current


__all__ = ["CalibSADDPMEOG", "TRANSFER_DIM", "ddim_sample_eog"]
