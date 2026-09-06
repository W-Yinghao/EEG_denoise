"""Deterministic-warm-start sensor-coordinate artifact SDEdit for V26."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from eeg_scad.models.calib_refine_det import ArtifactRefinerBackbone
from eeg_scad.models.diffusion_schedule import cosine_alpha_bar, extract
from eeg_scad.models.eegdus_backbone import sinusoidal_embedding


def sigma_to_timestep(alpha_bar: Tensor, sigma: float) -> int:
    if sigma <= 0:
        return 0
    target = 1.0 - float(sigma) ** 2
    return int(torch.argmin((alpha_bar - target).abs()).item())


class _BaseSDEdit(nn.Module):
    def __init__(self, conditional_channels: int, width: int = 64, timesteps: int = 1000) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.backbone = ArtifactRefinerBackbone(46 + conditional_channels, width)
        self.time = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.register_buffer("alpha_bar", cosine_alpha_bar(timesteps))

    def _predict(self, state: Tensor, condition: Tensor, context: Tensor, timestep: Tensor, anchor: Tensor) -> Tensor:
        combined_context = context + self.time(sinusoidal_embedding(timestep, 128))
        return anchor + .1 * self.backbone(torch.cat((state, condition), 1), combined_context)

    def training_loss(self, target: Tensor, anchor: Tensor, condition: Tensor, context: Tensor, maximum_timestep: int, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        timestep = torch.randint(0, maximum_timestep + 1, (len(target),), device=target.device, generator=generator)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        alpha = extract(self.alpha_bar, timestep, target.ndim)
        state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        prediction = self._predict(state, condition, context, timestep, anchor)
        return (prediction - target).abs().mean() + .5 * (prediction - target).square().mean(), prediction, timestep

    @torch.no_grad()
    def _sample(self, anchor: Tensor, condition: Tensor, context: Tensor, noise: Tensor, sigma_start: float, steps: int) -> tuple[Tensor, list[dict[str, float]]]:
        if sigma_start <= 0:
            return anchor.clone(), [{"step": 0, "state_rms": float(anchor.square().mean().sqrt()), "x0_rms": float(anchor.square().mean().sqrt()), "refinement_rms": 0.0}]
        t0 = sigma_to_timestep(self.alpha_bar, sigma_start)
        alpha0 = self.alpha_bar[t0]
        state = alpha0.sqrt() * anchor + (1 - alpha0).sqrt() * noise
        schedule = torch.linspace(t0, 0, steps, device=anchor.device).round().long().unique_consecutive()
        trajectory = []
        for index, tvalue in enumerate(schedule):
            timestep = torch.full((len(anchor),), int(tvalue), device=anchor.device, dtype=torch.long)
            x0 = self._predict(state, condition, context, timestep, anchor)
            alpha = extract(self.alpha_bar, timestep, state.ndim)
            epsilon = (state - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp_min(1e-8)
            trajectory.append({"step": int(tvalue), "state_rms": float(state.square().mean().sqrt()), "x0_rms": float(x0.square().mean().sqrt()), "refinement_rms": float((x0-anchor).square().mean().sqrt()), "max_abs": float(x0.abs().max())})
            if index + 1 == len(schedule):
                state = x0
            else:
                next_t = torch.full_like(timestep, int(schedule[index + 1]))
                next_alpha = extract(self.alpha_bar, next_t, state.ndim)
                state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
        return state, trajectory


class CalibSDEdit(_BaseSDEdit):
    def __init__(self, width: int = 64, timesteps: int = 1000) -> None:
        super().__init__(46 * 5, width, timesteps)

    def condition(self, y: Tensor, artifact_det: Tensor, artifact_pop: Tensor) -> Tensor:
        return torch.cat((y, artifact_det, y-artifact_det, artifact_pop, artifact_det-artifact_pop), 1)

    def training_loss(self, target: Tensor, y: Tensor, artifact_det: Tensor, artifact_pop: Tensor, context: Tensor, maximum_timestep: int, generator: torch.Generator):
        return super().training_loss(target, artifact_det, self.condition(y, artifact_det, artifact_pop), context, maximum_timestep, generator)

    def sample(self, y: Tensor, artifact_det: Tensor, artifact_pop: Tensor, context: Tensor, noise: Tensor, sigma_start: float = .2, steps: int = 10):
        return self._sample(artifact_det, self.condition(y, artifact_det, artifact_pop), context, noise, sigma_start, steps)


class PopSDEdit(_BaseSDEdit):
    uses_subject_support = False

    def __init__(self, width: int = 64, timesteps: int = 1000) -> None:
        super().__init__(46 * 3, width, timesteps)
        self.register_buffer("population_context", torch.zeros(1, 128))

    def condition(self, y: Tensor, artifact_pop: Tensor) -> Tensor:
        return torch.cat((y, artifact_pop, y-artifact_pop), 1)

    def training_loss(self, target: Tensor, y: Tensor, artifact_pop: Tensor, maximum_timestep: int, generator: torch.Generator):
        context = self.population_context.expand(len(y), -1)
        return super().training_loss(target, artifact_pop, self.condition(y, artifact_pop), context, maximum_timestep, generator)

    def sample(self, y: Tensor, artifact_pop: Tensor, noise: Tensor, sigma_start: float = .2, steps: int = 10):
        context = self.population_context.expand(len(y), -1)
        return self._sample(artifact_pop, self.condition(y, artifact_pop), context, noise, sigma_start, steps)


__all__ = ["CalibSDEdit", "PopSDEdit", "sigma_to_timestep"]
