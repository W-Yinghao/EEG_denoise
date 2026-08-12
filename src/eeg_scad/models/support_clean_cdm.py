"""V28 support-conditioned clean-signal conditional diffusion."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from eeg_scad.models.calib_refine_det import ArtifactRefinerBackbone
from eeg_scad.models.diffusion_schedule import cosine_alpha_bar, extract
from eeg_scad.models.eegdus_backbone import sinusoidal_embedding


class CleanConditionalBackbone(nn.Module):
    """Shared clean-x0 backbone used by population and support variants."""

    def __init__(self, diffusion: bool, width: int = 64) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.network = ArtifactRefinerBackbone(92 if diffusion else 46, width)
        self.time = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)) if diffusion else None

    def forward(self, y: Tensor, context: Tensor, state: Tensor | None = None, timestep: Tensor | None = None) -> Tensor:
        if self.diffusion:
            if state is None or timestep is None:
                raise ValueError("diffusion requires clean state and timestep")
            context = context + self.time(sinusoidal_embedding(timestep, 128))
            value = torch.cat((state, y), 1)
        else:
            value = y
        # Observation anchored parameterization while still predicting clean x0.
        return y + .1 * self.network(value, context)


class SupportCleanCDM(nn.Module):
    forbidden_fields = ("query_EOG", "query_operator", "query_event", "subject_ID")

    def __init__(self, width: int = 64, timesteps: int = 1000) -> None:
        super().__init__(); self.backbone = CleanConditionalBackbone(True, width); self.register_buffer("alpha_bar", cosine_alpha_bar(timesteps))

    def predict_x0(self, state: Tensor, y: Tensor, context: Tensor, timestep: Tensor) -> Tensor:
        return self.backbone(y, context, state, timestep)

    def training_prediction(self, clean: Tensor, y: Tensor, context: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor]:
        timestep = torch.randint(0, len(self.alpha_bar), (len(clean),), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        alpha = extract(self.alpha_bar, timestep, clean.ndim)
        state = alpha.sqrt() * clean + (1-alpha).sqrt() * noise
        return self.predict_x0(state, y, context, timestep), timestep, state

    @torch.no_grad()
    def sample(self, y: Tensor, context: Tensor, noise: Tensor, steps: int = 25) -> tuple[Tensor, list[dict[str, float]]]:
        schedule = torch.linspace(len(self.alpha_bar)-1, 0, steps, device=y.device).round().long().unique_consecutive(); state=noise; trajectory=[]
        for index,tvalue in enumerate(schedule):
            timestep=torch.full((len(y),),int(tvalue),device=y.device,dtype=torch.long);x0=self.predict_x0(state,y,context,timestep);alpha=extract(self.alpha_bar,timestep,state.ndim);eps=(state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8)
            trajectory.append({"step":int(tvalue),"state_rms":float(state.square().mean().sqrt()),"x0_rms":float(x0.square().mean().sqrt()),"max_abs":float(x0.abs().max())})
            if index+1==len(schedule):state=x0
            else:
                next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=extract(self.alpha_bar,next_t,state.ndim);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*eps
        return state,trajectory


__all__ = ["CleanConditionalBackbone", "SupportCleanCDM"]
