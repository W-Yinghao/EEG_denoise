"""Official-semantics shared-channel EEGDfus with explicit transfer FiLM.

This is not the V40R 46-channel port.  Every EEG channel is a 1x512 instance of
one shared dual-branch epsilon model.  The architecture follows the audited
official ``denoising_model_eegdnet.py`` tensor contract; only two zero-initialized
transfer FiLM modules and their 53-value per-channel condition are added.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


SAMPLES = 512
FEATURES = 64
TRANSFER_DIM = 53


class KaimingConv1d(nn.Conv1d):
    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class OfficialAttention(nn.Module):
    """One-head 512->64 Q/K/V attention used by the official EEGDfus model."""

    def __init__(self, samples: int = SAMPLES, key_dim: int = 64) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.q = nn.Linear(samples, key_dim, bias=False)
        self.k = nn.Linear(samples, key_dim, bias=False)
        self.v = nn.Linear(samples, key_dim, bias=False)
        self.out = nn.Linear(key_dim, samples, bias=False)
        self.norm = nn.LayerNorm(samples)

    def forward(self, value: Tensor) -> Tensor:
        score = self.q(value) @ self.k(value).transpose(-1, -2) / math.sqrt(self.key_dim)
        attended = torch.softmax(score, dim=-1) @ self.v(value)
        return self.norm(value + self.out(attended))


class OfficialEncoderLayer(nn.Module):
    def __init__(self, samples: int = SAMPLES, hidden: int = 512) -> None:
        super().__init__()
        self.attention = OfficialAttention(samples)
        self.feed_forward = nn.Sequential(nn.Linear(samples, hidden, bias=False), nn.ReLU(),
                                          nn.Linear(hidden, samples, bias=False))
        self.norm = nn.LayerNorm(samples)

    def forward(self, value: Tensor) -> Tensor:
        value = self.attention(value)
        return self.norm(value + self.feed_forward(value))


def noise_embedding(noise_level: Tensor, features: int = FEATURES) -> Tensor:
    value = noise_level.reshape(-1)
    count = features // 2
    step = torch.arange(count, device=value.device, dtype=value.dtype) / count
    angle = value[:, None] * torch.exp(-math.log(1e4) * step[None])
    return torch.cat((angle.sin(), angle.cos()), dim=-1)[:, :, None]


class OfficialNoiseFiLM(nn.Module):
    def __init__(self, samples: int = SAMPLES) -> None:
        super().__init__()
        self.gamma = nn.Linear(1, samples)
        self.beta = nn.Linear(1, samples)

    def forward(self, value: Tensor, embedding: Tensor) -> Tensor:
        return self.gamma(embedding) * value + self.beta(embedding)


class ZeroTransferFiLM(nn.Module):
    """Per-feature transfer modulation, exactly identity at initialization."""

    def __init__(self, condition_dim: int = TRANSFER_DIM, features: int = FEATURES) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(condition_dim, 128), nn.SiLU())
        self.out = nn.Linear(128, 2 * features)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.out(self.hidden(condition)).chunk(2, dim=-1)
        return value * (1 + scale[:, :, None]) + shift[:, :, None]


class CalibEEGDfus(nn.Module):
    def __init__(self, samples: int = SAMPLES, features: int = FEATURES, condition_dim: int = TRANSFER_DIM) -> None:
        super().__init__()
        if samples != SAMPLES:
            raise ValueError("V41R official waveform contract is fixed at 512 samples")
        conv = lambda: nn.Sequential(KaimingConv1d(1, features, 3, padding=1),
                                     KaimingConv1d(features, features, 3, padding=1))
        self.stream_x = nn.ModuleList((conv(), OfficialEncoderLayer(samples),
                                       OfficialEncoderLayer(samples), OfficialEncoderLayer(samples)))
        self.stream_condition = nn.ModuleList((conv(), OfficialEncoderLayer(samples),
                                               OfficialEncoderLayer(samples), OfficialEncoderLayer(samples)))
        self.bridges = nn.ModuleList(OfficialNoiseFiLM(samples) for _ in range(4))
        self.transfer_mid = ZeroTransferFiLM(condition_dim, features)
        self.transfer_late = ZeroTransferFiLM(condition_dim, features)
        self.output = nn.Sequential(KaimingConv1d(features, features, 3, padding=1),
                                    KaimingConv1d(features, 1, 3, padding=1))

    def forward(self, noisy: Tensor, observed: Tensor, noise_level: Tensor, transfer: Tensor) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (1, SAMPLES) or observed.shape != noisy.shape:
            raise ValueError("CalibEEGDfus requires Bx1x512 noisy/observed tensors")
        if transfer.shape != (len(noisy), TRANSFER_DIM):
            raise ValueError("transfer condition must have shape Bx53")
        embedding = noise_embedding(noise_level)
        skips = []
        value = noisy
        for layer, bridge in zip(self.stream_x, self.bridges):
            value = layer(value)
            skips.append(bridge(value, embedding))
        value = observed
        for index, (skip, layer) in enumerate(zip(skips, self.stream_condition)):
            value = layer(value) + skip
            if index == 1:
                value = self.transfer_mid(value, transfer)
            if index == 3:
                value = self.transfer_late(value, transfer)
        return self.output(value)

    def transfer_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.transfer_mid.parameters()
        yield from self.transfer_late.parameters()


class OfficialLinearSchedule(nn.Module):
    def __init__(self, steps: int = 500, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar_prev = torch.cat((torch.ones(1), alpha_bar[:-1]))
        posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_mean_coef1", beta * alpha_bar_prev.sqrt() / (1 - alpha_bar))
        self.register_buffer("posterior_mean_coef2", (1 - alpha_bar_prev) * alpha.sqrt() / (1 - alpha_bar))

    def training_sample(self, clean: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor, Tensor]:
        batch = len(clean)
        timestep = torch.randint(0, len(self.beta), (batch,), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        alpha = self.alpha_bar[timestep][:, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        return noisy, noise, alpha.sqrt()[:, 0]


@torch.no_grad()
def ancestral_sample(model: CalibEEGDfus, observed: Tensor, transfer: Tensor, seed: int,
                     schedule: OfficialLinearSchedule | None = None) -> Tensor:
    """Full 500-step official-style ancestral sampling with replayable noise."""
    schedule = schedule or OfficialLinearSchedule().to(observed.device)
    generator = torch.Generator(device=observed.device).manual_seed(int(seed))
    current = torch.randn(observed.shape, device=observed.device, dtype=observed.dtype, generator=generator)
    for timestep in reversed(range(len(schedule.beta))):
        level = schedule.alpha_bar[timestep].sqrt().expand(len(observed), 1)
        predicted_noise = model(current, observed, level, transfer)
        x0 = (current - (1 - schedule.alpha_bar[timestep]).sqrt() * predicted_noise) / schedule.alpha_bar[timestep].sqrt().clamp_min(1e-8)
        mean = schedule.posterior_mean_coef1[timestep] * x0 + schedule.posterior_mean_coef2[timestep] * current
        if timestep:
            noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
            current = mean + schedule.posterior_variance[timestep].clamp_min(1e-20).sqrt() * noise
        else:
            current = mean
        if not torch.isfinite(current).all():
            raise FloatingPointError(f"nonfinite ancestral trajectory at step {timestep}")
    return current


__all__ = [
    "CalibEEGDfus", "FEATURES", "OfficialLinearSchedule", "SAMPLES", "TRANSFER_DIM",
    "ZeroTransferFiLM", "ancestral_sample", "noise_embedding",
]
