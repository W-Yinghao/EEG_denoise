"""V29 support residual adapter around a frozen V28 population CDM."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from eeg_scad.models.diffusion_schedule import extract
from eeg_scad.models.support_adapter_common import SupportResidualAdapter


class SupportAdapterCDM(nn.Module):
    forbidden_fields = ("query_EOG", "query_operator", "query_event", "subject_ID")
    def __init__(self, width: int = 32) -> None:
        super().__init__(); self.adapter = SupportResidualAdapter(46 * 4, width, True)

    def increment(self, state: Tensor, y: Tensor, population: Tensor, context: Tensor, timestep: Tensor) -> Tensor:
        return self.adapter(torch.cat((state, y, population, y-population), 1), context, timestep)

    def predict_x0(self, state: Tensor, y: Tensor, population: Tensor, context: Tensor, timestep: Tensor, bypass: bool = False) -> Tensor:
        return population if bypass else population + self.increment(state, y, population, context, timestep)

    @torch.no_grad()
    def sample(self, population_model: nn.Module, y: Tensor, context: Tensor, noise: Tensor, steps: int = 10, bypass: bool = False):
        schedule=torch.linspace(len(population_model.alpha_bar)-1,0,steps,device=y.device).round().long().unique_consecutive();state=noise;trajectory=[]
        for index,tvalue in enumerate(schedule):
            timestep=torch.full((len(y),),int(tvalue),device=y.device,dtype=torch.long)
            population=population_model.predict_x0(state,y,population_model.context(len(y)),timestep)
            x0=self.predict_x0(state,y,population,context,timestep,bypass)
            alpha=extract(population_model.alpha_bar,timestep,state.ndim);eps=(state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8)
            trajectory.append({"step":int(tvalue),"state_rms":float(state.square().mean().sqrt()),"population_rms":float(population.square().mean().sqrt()),"increment_rms":float((x0-population).square().mean().sqrt()),"x0_rms":float(x0.square().mean().sqrt())})
            if index+1==len(schedule):state=x0
            else:
                next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=extract(population_model.alpha_bar,next_t,state.ndim);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*eps
        return state,trajectory


__all__ = ["SupportAdapterCDM"]
