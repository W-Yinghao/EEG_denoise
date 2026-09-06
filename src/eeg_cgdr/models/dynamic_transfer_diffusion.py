"""Matched deterministic and diffusion models for dynamic-transfer artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


def fir_projector(transfer: Tensor, rank: int = 4) -> Tensor:
    if transfer.ndim != 4:
        raise ValueError("transfer must be B,C,E,L")
    u, _, _ = torch.linalg.svd(transfer.flatten(2).float(), full_matrices=False)
    basis = u[..., : min(rank, u.shape[-1])].to(transfer.dtype)
    return basis @ basis.transpose(-1, -2)


def apply_fir(transfer: Tensor, eog: Tensor) -> Tensor:
    """Apply centered MIMO FIR coefficients without crossing batch boundaries."""
    if transfer.ndim != 4 or eog.ndim != 3:
        raise ValueError("expected transfer B,C,E,L and EOG B,E,T")
    if transfer.shape[0] != eog.shape[0] or transfer.shape[2] != eog.shape[1]:
        raise ValueError("FIR/EOG dimensions differ")
    padding = transfer.shape[-1] // 2
    return torch.cat([
        torch.nn.functional.conv1d(eog[i:i+1], transfer[i], padding=padding)[..., :eog.shape[-1]]
        for i in range(transfer.shape[0])
    ])


def _time_embedding(timestep: Tensor, width: int) -> Tensor:
    half = width // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timestep.device, dtype=torch.float32))
    phase = timestep.float()[:, None] * frequencies[None]
    value = torch.cat((phase.sin(), phase.cos()), dim=1)
    return torch.nn.functional.pad(value, (0, width - value.shape[1]))


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.time = nn.Linear(width, 2 * width)

    def forward(self, value: Tensor, time: Tensor) -> Tensor:
        scale, shift = self.time(time).chunk(2, dim=1)
        hidden = self.norm1(value) * (1 + scale[..., None]) + shift[..., None]
        hidden = self.conv1(torch.nn.functional.silu(hidden))
        return value + self.conv2(torch.nn.functional.silu(self.norm2(hidden)))


@dataclass(frozen=True)
class DynamicTransferModelConfig:
    eeg_channels: int
    width: int = 48
    blocks: int = 6
    timesteps: int = 1000
    ddim_steps: int = 25


class DynamicTransferBackbone(nn.Module):
    forbidden_inputs = ("query_EOG", "query_artifactclasses", "query_transfer_generator", "participant_ID", "query_outcomes")

    def __init__(self, config: DynamicTransferModelConfig) -> None:
        super().__init__(); self.config = config
        channels = config.eeg_channels
        # state, y, P_H y, Q_H y, per-channel FIR scale, rho
        self.input = nn.Conv1d(5 * channels + 1, config.width, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(config.width, config.width * 4), nn.SiLU(), nn.Linear(config.width * 4, config.width))
        self.blocks = nn.ModuleList(_ResidualBlock(config.width, 2 ** (i % 5)) for i in range(config.blocks))
        self.output = nn.Sequential(nn.GroupNorm(8, config.width), nn.SiLU(), nn.Conv1d(config.width, channels, 3, padding=1))

    def condition(self, observed: Tensor, transfer: Tensor, reliability: Tensor) -> Tensor:
        if observed.ndim != 3 or observed.shape[1] != self.config.eeg_channels:
            raise ValueError("observed montage differs from model")
        if transfer.shape[:2] != observed.shape[:2] or reliability.shape != (observed.shape[0],):
            raise ValueError("condition dimensions differ")
        transfer = transfer.to(device=observed.device, dtype=observed.dtype)
        reliability = reliability.to(device=observed.device, dtype=observed.dtype)
        projector = fir_projector(transfer)
        projected = torch.einsum("bij,bjt->bit", projector, observed)
        fir_scale = transfer.square().mean(dim=(2, 3)).sqrt()[..., None].expand_as(observed)
        rho = reliability[:, None, None].expand(-1, 1, observed.shape[-1])
        return torch.cat((observed, projected, observed - projected, fir_scale, rho), dim=1)

    def forward(self, state: Tensor, timestep: Tensor, *, observed: Tensor, transfer: Tensor, reliability: Tensor) -> Tensor:
        hidden = self.input(torch.cat((state, self.condition(observed, transfer, reliability)), dim=1))
        embedded = self.time(_time_embedding(timestep, self.config.width))
        for block in self.blocks:
            hidden = block(hidden, embedded)
        return self.output(hidden)


def cosine_alpha_bar(timesteps: int, offset: float = 0.008) -> Tensor:
    points = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    values = torch.cos(((points / timesteps + offset) / (1 + offset)) * math.pi / 2).square()
    return (values[1:] / values[0]).clamp(1e-8, 1).float()


class DynamicTransferDeterministic(nn.Module):
    def __init__(self, config: DynamicTransferModelConfig) -> None:
        super().__init__(); self.config = config; self.backbone = DynamicTransferBackbone(config)

    def forward(self, observed: Tensor, *, transfer: Tensor, reliability: Tensor) -> Tensor:
        return self.backbone(torch.zeros_like(observed), torch.zeros(observed.shape[0], dtype=torch.long, device=observed.device), observed=observed, transfer=transfer, reliability=reliability)


class DynamicTransferDiffusion(nn.Module):
    def __init__(self, config: DynamicTransferModelConfig) -> None:
        super().__init__(); self.config = config; self.backbone = DynamicTransferBackbone(config)
        self.register_buffer("alpha_bar", cosine_alpha_bar(config.timesteps))

    def training_loss(self, target: Tensor, *, observed: Tensor, transfer: Tensor, reliability: Tensor, generator: torch.Generator) -> Tensor:
        timestep = torch.randint(0, self.config.timesteps, (target.shape[0],), device=target.device, generator=generator)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        alpha = self.alpha_bar[timestep][:, None, None]
        state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
        predicted = self.backbone(state, timestep, observed=observed, transfer=transfer, reliability=reliability)
        snr = alpha / (1 - alpha).clamp_min(1e-8)
        weight = torch.minimum(snr, torch.full_like(snr, 5.0)) / (snr + 1)
        return ((predicted - velocity).square() * weight).mean()

    @torch.no_grad()
    def sample(self, *, observed: Tensor, transfer: Tensor, reliability: Tensor, sample_seeds: Sequence[int]) -> tuple[Tensor, Tensor, int]:
        if len(sample_seeds) != 8:
            raise ValueError("primary inference requires exactly K=8")
        return self.sample_k(observed=observed, transfer=transfer, reliability=reliability, sample_seeds=sample_seeds)

    @torch.no_grad()
    def sample_k(self, *, observed: Tensor, transfer: Tensor, reliability: Tensor, sample_seeds: Sequence[int], trace: bool = False) -> tuple[Tensor, Tensor, int] | tuple[Tensor, Tensor, int, list[dict[str, Tensor | int]]]:
        """Diagnostic sampler accepting an explicit K without changing primary K=8 inference."""
        if not sample_seeds:
            raise ValueError("at least one posterior seed is required")
        sequence = torch.linspace(self.config.timesteps - 1, 0, self.config.ddim_steps).round().long().tolist()
        samples = []
        trajectory: list[dict[str, Tensor | int]] = []
        for seed in sample_seeds:
            generator = torch.Generator(device=observed.device).manual_seed(int(seed))
            state = torch.randn(observed.shape, device=observed.device, generator=generator)
            for index, step in enumerate(sequence):
                timestep = torch.full((observed.shape[0],), int(step), dtype=torch.long, device=observed.device)
                velocity = self.backbone(state, timestep, observed=observed, transfer=transfer, reliability=reliability)
                alpha = self.alpha_bar[int(step)]
                x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * velocity
                epsilon = (1 - alpha).sqrt() * state + alpha.sqrt() * velocity
                if trace and seed == sample_seeds[0]:
                    trajectory.append({"timestep": int(step), "state": state.detach().clone(), "x0": x0.detach().clone(), "epsilon": epsilon.detach().clone()})
                if index + 1 == len(sequence): state = x0
                else:
                    next_alpha = self.alpha_bar[int(sequence[index + 1])]
                    state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
            samples.append(state)
        stack = torch.stack(samples)
        result = (stack.mean(0), stack.std(0, unbiased=False), len(sequence) * len(samples))
        return (*result, trajectory) if trace else result

    @torch.no_grad()
    def oracle_v_roundtrip(self, target: Tensor, *, initial_noise: Tensor) -> Tensor:
        """Run deterministic DDIM while analytically supplying v for a known x0."""
        sequence = torch.linspace(self.config.timesteps - 1, 0, self.config.ddim_steps).round().long().tolist()
        alpha = self.alpha_bar[int(sequence[0])]
        state = alpha.sqrt() * target + (1 - alpha).sqrt() * initial_noise
        for index, step in enumerate(sequence):
            alpha = self.alpha_bar[int(step)]
            epsilon = (state - alpha.sqrt() * target) / (1 - alpha).sqrt().clamp_min(1e-12)
            velocity = alpha.sqrt() * epsilon - (1 - alpha).sqrt() * target
            x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * velocity
            epsilon_hat = (1 - alpha).sqrt() * state + alpha.sqrt() * velocity
            if index + 1 == len(sequence):
                state = x0
            else:
                next_alpha = self.alpha_bar[int(sequence[index + 1])]
                state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon_hat
        return state


__all__ = ["DynamicTransferModelConfig", "DynamicTransferDeterministic", "DynamicTransferDiffusion", "apply_fir", "fir_projector"]
