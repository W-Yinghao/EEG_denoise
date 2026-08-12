"""Stepwise energy sampling for a frozen V26 SDEdit model."""
from __future__ import annotations

import torch
from torch import Tensor

from eeg_scad.energy.partial_observation import partial_observation_prox
from eeg_scad.models.calib_sdedit import sigma_to_timestep
from eeg_scad.models.diffusion_schedule import extract


@torch.no_grad()
def sample_stepwise_energy(model, y: Tensor, artifact_det: Tensor, artifact_pop: Tensor, context: Tensor, noise: Tensor, projector: Tensor, mask: Tensor, lambda_anchor: float, lambda_measurement: float, sigma_start: float, steps: int):
    if sigma_start <= 0:
        return partial_observation_prox(artifact_det, artifact_det, projector, mask, lambda_anchor, lambda_measurement), []
    t0 = sigma_to_timestep(model.alpha_bar, sigma_start)
    alpha0 = model.alpha_bar[t0]
    state = alpha0.sqrt()*artifact_det + (1-alpha0).sqrt()*noise
    schedule = torch.linspace(t0, 0, steps, device=y.device).round().long().unique_consecutive()
    condition = model.condition(y, artifact_det, artifact_pop)
    trajectory = []
    for index, tvalue in enumerate(schedule):
        timestep = torch.full((len(y),), int(tvalue), device=y.device, dtype=torch.long)
        raw_x0 = model._predict(state, condition, context, timestep, artifact_det)
        x0 = partial_observation_prox(raw_x0, artifact_det, projector, mask, lambda_anchor, lambda_measurement)
        alpha = extract(model.alpha_bar, timestep, state.ndim)
        epsilon = (state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8)
        trajectory.append({"step": int(tvalue), "state_rms": float(state.square().mean().sqrt()), "raw_x0_rms": float(raw_x0.square().mean().sqrt()), "energy_x0_rms": float(x0.square().mean().sqrt())})
        if index+1 == len(schedule): state=x0
        else:
            next_t=torch.full_like(timestep, int(schedule[index+1])); next_alpha=extract(model.alpha_bar,next_t,state.ndim); state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
    return state, trajectory


__all__ = ["sample_stepwise_energy"]
