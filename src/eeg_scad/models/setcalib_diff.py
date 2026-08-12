from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from eeg_scad.models.diffusion_schedule import cosine_alpha_bar,extract
from eeg_scad.models.eegdus_backbone import sinusoidal_embedding
from eeg_scad.models.population_anchor_v24 import _ResidualBlock

@dataclass(frozen=True)
class SetCalibDiffConfig:
    rank:int=8;base_channels:int=64;timesteps:int=1000;ddim_steps:int=25

class SetCalibResidualDiffusion(nn.Module):
    def __init__(self,config:SetCalibDiffConfig=SetCalibDiffConfig())->None:
        super().__init__();self.config=config;inputs=config.rank+46*3+4+config.rank;self.inp=nn.Conv1d(inputs,config.base_channels,1);self.time=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,config.base_channels));self.context=nn.Linear(128,2*config.base_channels);self.blocks=nn.Sequential(*[_ResidualBlock(config.base_channels,d) for d in (1,2,4,8,16,32)]);self.out=nn.Conv1d(config.base_channels,config.rank,1);self.register_buffer("alpha_bar",cosine_alpha_bar(config.timesteps))
    def predict(self,state:Tensor,y:Tensor,a0:Tensor,q0:Tensor,hdet:Tensor,context:Tensor,timestep:Tensor)->Tensor:
        hidden=self.inp(torch.cat((state,y,a0,y-a0,q0,hdet),1))+self.time(sinusoidal_embedding(timestep,128))[...,None];scale,shift=self.context(context).chunk(2,1);return self.out(self.blocks(hidden*(1+0.1*torch.tanh(scale)[...,None])+shift[...,None]))
    def training_loss(self,target:Tensor,y:Tensor,a0:Tensor,q0:Tensor,hdet:Tensor,context:Tensor,generator:torch.Generator)->tuple[Tensor,Tensor]:
        timestep=torch.randint(0,self.config.timesteps,(len(target),),device=target.device,generator=generator);noise=torch.randn(target.shape,device=target.device,generator=generator);alpha=extract(self.alpha_bar,timestep,target.ndim);state=alpha.sqrt()*target+(1-alpha).sqrt()*noise;prediction=self.predict(state,y,a0,q0,hdet,context,timestep);return (prediction-target).square().mean(),prediction
    @torch.no_grad()
    def sample(self,y:Tensor,a0:Tensor,q0:Tensor,hdet:Tensor,context:Tensor,noise:Tensor,steps:int|None=None)->Tensor:
        state=noise.clone();schedule=torch.linspace(self.config.timesteps-1,0,steps or self.config.ddim_steps,device=y.device).round().long()
        for index,tvalue in enumerate(schedule):
            timestep=torch.full((len(y),),int(tvalue),device=y.device,dtype=torch.long);x0=self.predict(state,y,a0,q0,hdet,context,timestep);alpha=extract(self.alpha_bar,timestep,state.ndim);epsilon=(state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8)
            if index+1==len(schedule):state=x0
            else:
                next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=extract(self.alpha_bar,next_t,state.ndim);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
        return state

__all__=["SetCalibResidualDiffusion","SetCalibDiffConfig"]
