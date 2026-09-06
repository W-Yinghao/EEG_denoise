from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from eeg_scad.models.diffusion_schedule import cosine_alpha_bar, extract
from eeg_scad.models.eegdus_backbone import sinusoidal_embedding
from .population_anchor_v24 import _ResidualBlock
from .pa_el_det import decode_deviation


@dataclass(frozen=True)
class PAELSCADConfig:
    eeg_channels: int = 46
    eog_dimensions: int = 4
    base_channels: int = 64
    timesteps: int = 1000
    ddim_steps: int = 25


class PAELResidualDiffusion(nn.Module):
    """x0 diffusion over the low-dimensional EOG latent residual."""

    forbidden_fields = ("query_EOG", "query_operator", "event")

    def __init__(self, config: PAELSCADConfig = PAELSCADConfig()) -> None:
        super().__init__()
        self.config = config
        inputs = config.eog_dimensions + 3 * config.eeg_channels + 2 * config.eog_dimensions
        self.inp = nn.Conv1d(inputs, config.base_channels, 1)
        self.time = nn.Sequential(nn.Linear(128, 2 * config.base_channels), nn.SiLU(), nn.Linear(2 * config.base_channels, config.base_channels))
        self.blocks = nn.ModuleList([_ResidualBlock(config.base_channels, dilation) for dilation in (1, 2, 4, 8, 16, 32)])
        self.out = nn.Conv1d(config.base_channels, config.eog_dimensions, 1)
        self.register_buffer("alpha_bar", cosine_alpha_bar(config.timesteps))

    def predict(self, state: Tensor, y: Tensor, a_pop: Tensor, q0: Tensor, zdet: Tensor, timestep: Tensor) -> Tensor:
        features = torch.cat((state, y, a_pop, y - a_pop, q0, zdet), dim=1)
        hidden = self.inp(features) + self.time(sinusoidal_embedding(timestep, 128))[:, :, None]
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(hidden)

    def training_loss(self, target: Tensor, y: Tensor, a_pop: Tensor, q0: Tensor, zdet: Tensor, generator: torch.Generator, timestep: Tensor | None = None, noise: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if timestep is None:
            timestep = torch.randint(0, self.config.timesteps, (len(target),), device=target.device, generator=generator)
        if noise is None:
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        alpha = extract(self.alpha_bar, timestep, target.ndim)
        state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted = self.predict(state, y, a_pop, q0, zdet, timestep)
        return (predicted - target).square().mean(), {"state": state, "predicted_x0": predicted, "timestep": timestep, "noise": noise}

    @torch.no_grad()
    def sample(self, y: Tensor, a_pop: Tensor, q0: Tensor, zdet: Tensor, initial_noise: Tensor, steps: int | None = None, trajectory: bool = False) -> tuple[Tensor, list[dict[str, float]]]:
        state = initial_noise.clone()
        schedule = torch.linspace(self.config.timesteps - 1, 0, steps or self.config.ddim_steps, device=y.device).round().long()
        trace: list[dict[str, float]] = []
        for index, tvalue in enumerate(schedule):
            timestep = torch.full((len(y),), int(tvalue), device=y.device, dtype=torch.long)
            x0 = self.predict(state, y, a_pop, q0, zdet, timestep)
            alpha = extract(self.alpha_bar, timestep, state.ndim)
            epsilon = (state - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp_min(1e-8)
            if trajectory:
                trace.append({"step": index, "timestep": int(tvalue), "r_t_rms": float(state.square().mean().sqrt()), "r_hat_rms": float(x0.square().mean().sqrt()), "e_det_rms": float(zdet.square().mean().sqrt()), "e_final_rms": float((zdet + x0).square().mean().sqrt()), "max_abs": float(x0.abs().max())})
            if index + 1 == len(schedule):
                state = x0
            else:
                next_t = torch.full_like(timestep, int(schedule[index + 1]))
                next_alpha = extract(self.alpha_bar, next_t, state.ndim)
                state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
        return state, trace

    @torch.no_grad()
    def artifact(self, y: Tensor, a_pop: Tensor, q0: Tensor, zdet: Tensor, deviation: Tensor, noise: Tensor, steps: int | None = None) -> Tensor:
        residual, _ = self.sample(y, a_pop, q0, zdet, noise, steps)
        return decode_deviation(a_pop, deviation, zdet + residual)


__all__ = ["PAELResidualDiffusion", "PAELSCADConfig"]

