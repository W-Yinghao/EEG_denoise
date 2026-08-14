"""Clean-room joint multichannel x0 conditional diffusion for V42R.

The implementation uses standard DDPM forward marginals and deterministic DDIM
updates.  It imports no historical SADDPM or third-party model source.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EEG_CHANNELS = 46
SAMPLES = 512
TRANSFER_DIM = 53


def time_embedding(timestep: Tensor, width: int = 128) -> Tensor:
    value = timestep.float().reshape(-1)
    half = width // 2
    frequency = torch.exp(-math.log(10_000) * torch.arange(half, device=value.device) / max(half - 1, 1))
    angle = value[:, None] * frequency[None]
    return torch.cat((angle.sin(), angle.cos()), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, time_width: int = 128) -> None:
        super().__init__()
        groups_in = math.gcd(incoming, 8)
        groups_out = math.gcd(outgoing, 8)
        self.norm1 = nn.GroupNorm(groups_in, incoming)
        self.conv1 = nn.Conv1d(incoming, outgoing, 3, padding=1)
        self.time = nn.Linear(time_width, outgoing)
        self.norm2 = nn.GroupNorm(groups_out, outgoing)
        self.conv2 = nn.Conv1d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, value: Tensor, embedding: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time(embedding)[:, :, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(value)


class BottleneckAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(width, 8), width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)

    def forward(self, value: Tensor) -> Tensor:
        sequence = self.norm(value).transpose(1, 2)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        return value + attended.transpose(1, 2)


class TransferStateEncoder(nn.Module):
    """Shared row encoder, one spatial mixing layer, and global pooling."""

    def __init__(self, channels: int = EEG_CHANNELS, row_dim: int = TRANSFER_DIM,
                 hidden: int = 32, context: int = 128) -> None:
        super().__init__()
        self.channels = channels
        self.row = nn.Sequential(nn.Linear(row_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.spatial = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.global_projection = nn.Sequential(nn.Linear(2 * hidden, context), nn.SiLU(), nn.Linear(context, context))
        self.sensor_projection = nn.Linear(hidden, 1)

    def forward(self, transfer: Tensor) -> tuple[Tensor, Tensor]:
        if transfer.ndim != 3 or transfer.shape[1:] != (self.channels, TRANSFER_DIM):
            raise ValueError(f"transfer state must have shape Bx{self.channels}x53")
        rows = self.row(transfer)
        mixed = F.silu(self.spatial(rows.transpose(1, 2))).transpose(1, 2)
        pooled = torch.cat((mixed.mean(dim=1), mixed.amax(dim=1)), dim=1)
        return self.global_projection(pooled), self.sensor_projection(mixed).squeeze(-1)


class TransferResidualDecoder(nn.Module):
    def __init__(self, width: int, context: int = 128, channels: int = EEG_CHANNELS) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(width, 8), width)
        self.conv = nn.Conv1d(width, width, 3, padding=1)
        self.film = nn.Linear(context, 2 * width)
        self.output = nn.Conv1d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor, context: Tensor, sensor_gate: Tensor) -> Tensor:
        scale, shift = self.film(context).chunk(2, dim=1)
        hidden = self.conv(F.silu(self.norm(features)))
        hidden = hidden * (1 + scale[:, :, None]) + shift[:, :, None]
        return self.output(F.silu(hidden)) * (1 + torch.tanh(sensor_gate)[:, :, None])


class CalibSADDPMCond(nn.Module):
    """Four-scale joint U-Net with population and transfer residual heads."""

    def __init__(self, channels: int = EEG_CHANNELS, samples: int = SAMPLES, base: int = 32) -> None:
        super().__init__()
        if samples != SAMPLES or channels < 1:
            raise ValueError("V42R models require one or more channels and 512 samples")
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
        self.population_head = nn.Conv1d(current, channels, 1)
        nn.init.zeros_(self.population_head.weight)
        nn.init.zeros_(self.population_head.bias)
        self.transfer_encoder = TransferStateEncoder(channels)
        self.transfer_decoder = TransferResidualDecoder(current, channels=channels)

    def forward(self, x_t: Tensor, observed: Tensor, timestep: Tensor, transfer: Tensor,
                transfer_enabled: bool = True) -> Tensor:
        if x_t.shape != observed.shape or x_t.shape[1:] != (self.channels, self.samples):
            raise ValueError("V42R expects matching Bx46x512 diffusion and observation tensors")
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
        population = self.population_head(features)
        if transfer_enabled:
            context, sensor_gate = self.transfer_encoder(transfer)
            personalized = self.transfer_decoder(features, context, sensor_gate)
        else:
            personalized = torch.zeros_like(population)
        return observed + population + personalized

    def transfer_parameters(self):
        yield from self.transfer_encoder.parameters()
        yield from self.transfer_decoder.parameters()


class LinearX0Schedule(nn.Module):
    def __init__(self, steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alpha_bar = torch.cumprod(1 - beta, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward_sample(self, clean: Tensor, generator: torch.Generator,
                       timestep: Tensor | None = None, noise: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        if timestep is None:
            timestep = torch.randint(0, len(self.beta), (len(clean),), device=clean.device, generator=generator)
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        alpha = self.alpha_bar[timestep][:, None, None]
        return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, timestep, noise


@torch.no_grad()
def ddim_sample(model: CalibSADDPMCond, observed: Tensor, transfer: Tensor, noise: Tensor,
                schedule: LinearX0Schedule, inference_steps: int = 50,
                transfer_enabled: bool = True) -> Tensor:
    """Deterministic eta=0 DDIM with the observation available at every step."""
    current = noise.clone()
    timesteps = torch.linspace(len(schedule.beta) - 1, 0, inference_steps, device=observed.device).round().long()
    for index, timestep in enumerate(timesteps):
        batch_t = timestep.expand(len(observed))
        predicted_x0 = model(current, observed, batch_t, transfer, transfer_enabled=transfer_enabled)
        alpha = schedule.alpha_bar[timestep]
        epsilon = (current - alpha.sqrt() * predicted_x0) / (1 - alpha).sqrt().clamp_min(1e-8)
        if index + 1 == len(timesteps):
            current = predicted_x0
        else:
            next_alpha = schedule.alpha_bar[timesteps[index + 1]]
            current = next_alpha.sqrt() * predicted_x0 + (1 - next_alpha).sqrt() * epsilon
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"nonfinite V42R DDIM state at index {index}")
    return current


__all__ = ["CalibSADDPMCond", "EEG_CHANNELS", "LinearX0Schedule", "SAMPLES", "TRANSFER_DIM",
           "TransferStateEncoder", "ddim_sample", "time_embedding"]
