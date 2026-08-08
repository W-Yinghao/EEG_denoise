"""EOG-anchored full-waveform residual diffusion used by V11."""

from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
import torch
from torch import Tensor,nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .artifact_latent_diffusion import cosine_alpha_bar


@dataclass(frozen=True)
class EOGResidualConfig:
    channels:int=3
    eog_channels:int=3
    signal_length:int=512
    base_channels:int=32
    timesteps:int=1000
    ddim_steps:int=25
    posterior_samples:int=8

    def __post_init__(self)->None:
        if self.channels!=3 or self.eog_channels!=3:raise ValueError("V11 freezes three EEG and three EOG channels")
        if self.signal_length%8:raise ValueError("signal length must be divisible by eight")
        if self.ddim_steps!=25 or self.posterior_samples!=8:raise ValueError("primary inference freezes DDIM25/K8")


class _ResidualUNet(nn.Module):
    def __init__(self,config:EOGResidualConfig,*,diffusion:bool)->None:
        super().__init__();inputs=config.channels+config.eog_channels+config.channels+(config.channels if diffusion else 0)+ (config.channels if diffusion else 0)
        self.diffusion=diffusion
        self.unet=UNet1D(ModelConfig(in_channels=inputs,out_channels=config.channels,signal_length=config.signal_length,base_channels=config.base_channels,channel_mults=[1,2,4],num_res_blocks=2,groupnorm_groups=8,dropout=.05,time_sinusoidal_dim=64,time_embed_dim=256,attention_length=64,attention_heads=4),subject_conditioned=False)

    def forward(self,state:Tensor|None,timestep:Tensor,*,y:Tensor,eog:Tensor,a0:Tensor,r_det:Tensor|None=None)->Tensor:
        fields=[y,eog,a0]
        if self.diffusion:
            if state is None or r_det is None:raise ValueError("diffusion requires state and deterministic anchor")
            fields.extend((r_det,state))
        return self.unet(torch.cat(fields,dim=1),timestep)


class DeterministicEOGResidual(nn.Module):
    visible_fields=("observed_EEG","query_EOG","support_operator")
    forbidden_fields=("query_labels","query_outcomes","participant_ID","query_transfer")
    def __init__(self,config:EOGResidualConfig)->None:
        super().__init__();self.config=config;self.backbone=_ResidualUNet(config,diffusion=False)
    def forward(self,*,y:Tensor,eog:Tensor,a0:Tensor)->Tensor:
        timestep=torch.zeros(len(y),dtype=torch.long,device=y.device);return self.backbone(None,timestep,y=y,eog=eog,a0=a0)


def _extract(values:Tensor,timestep:Tensor,ndim:int)->Tensor:
    return values.gather(0,timestep).reshape(len(timestep),*((1,)*(ndim-1)))


class EOGResidualDiffusion(nn.Module):
    visible_fields=DeterministicEOGResidual.visible_fields+("deterministic_residual",)
    forbidden_fields=DeterministicEOGResidual.forbidden_fields
    def __init__(self,config:EOGResidualConfig)->None:
        super().__init__();self.config=config;self.backbone=_ResidualUNet(config,diffusion=True);_,alpha=cosine_alpha_bar(config.timesteps);self.register_buffer("alpha_bar",alpha.float())

    def training_loss(self,target:Tensor,*,y:Tensor,eog:Tensor,a0:Tensor,r_det:Tensor,generator:torch.Generator,timestep:Tensor|None=None,noise:Tensor|None=None)->tuple[Tensor,dict[str,Tensor]]:
        if timestep is None:timestep=torch.randint(0,self.config.timesteps,(len(target),),device=target.device,generator=generator)
        if noise is None:noise=torch.randn(target.shape,device=target.device,dtype=target.dtype,generator=generator)
        alpha=_extract(self.alpha_bar,timestep,target.ndim);state=alpha.sqrt()*target+(1-alpha).sqrt()*noise;truth=alpha.sqrt()*noise-(1-alpha).sqrt()*target;prediction=self.backbone(state,timestep,y=y,eog=eog,a0=a0,r_det=r_det);x0=alpha.sqrt()*state-(1-alpha).sqrt()*prediction;return (prediction-truth).square().mean(),{"predicted_x0":x0,"timestep":timestep,"noise":noise}

    @torch.no_grad()
    def sample(self,*,y:Tensor,eog:Tensor,a0:Tensor,r_det:Tensor,initial_noise:Tensor)->Tensor:
        if initial_noise.shape!=y.shape:raise ValueError("initial noise differs from residual waveform")
        state=initial_noise.clone();schedule=torch.linspace(self.config.timesteps-1,0,self.config.ddim_steps,device=y.device).round().long()
        for index,t_value in enumerate(schedule):
            timestep=torch.full((len(y),),int(t_value),device=y.device,dtype=torch.long);alpha=_extract(self.alpha_bar,timestep,state.ndim);v=self.backbone(state,timestep,y=y,eog=eog,a0=a0,r_det=r_det);x0=alpha.sqrt()*state-(1-alpha).sqrt()*v;epsilon=(1-alpha).sqrt()*state+alpha.sqrt()*v
            if index+1==len(schedule):state=x0
            else:
                next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=_extract(self.alpha_bar,next_t,state.ndim);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
        return state

    @torch.no_grad()
    def oracle_roundtrip(self,target:Tensor,initial_noise:Tensor)->float:
        schedule=torch.linspace(self.config.timesteps-1,0,self.config.ddim_steps,device=target.device).round().long();first=torch.full((len(target),),int(schedule[0]),device=target.device,dtype=torch.long);alpha=_extract(self.alpha_bar,first,target.ndim);state=alpha.sqrt()*target+(1-alpha).sqrt()*initial_noise
        for index,t_value in enumerate(schedule):
            timestep=torch.full((len(target),),int(t_value),device=target.device,dtype=torch.long);alpha=_extract(self.alpha_bar,timestep,target.ndim);epsilon=(state-alpha.sqrt()*target)/(1-alpha).sqrt().clamp_min(1e-12);v=alpha.sqrt()*epsilon-(1-alpha).sqrt()*target;x0=alpha.sqrt()*state-(1-alpha).sqrt()*v
            if index+1==len(schedule):state=x0
            else:
                next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=_extract(self.alpha_bar,next_t,state.ndim);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
        return float(torch.linalg.vector_norm(state-target)/torch.linalg.vector_norm(target).clamp_min(1e-12))


class EMA:
    def __init__(self,module:nn.Module,decay:float=.999)->None:
        self.decay=decay;self.shadow={name:value.detach().clone() for name,value in module.state_dict().items()}
    @torch.no_grad()
    def update(self,module:nn.Module)->None:
        for name,value in module.state_dict().items():
            if torch.is_floating_point(value):self.shadow[name].lerp_(value.detach(),1-self.decay)
            else:self.shadow[name].copy_(value)
    def copy_to(self,module:nn.Module)->None:module.load_state_dict(self.shadow)
    def state_dict(self)->dict[str,object]:return {"decay":self.decay,"shadow":self.shadow}
    def load_state_dict(self,state:dict[str,object])->None:self.decay=float(state["decay"]);self.shadow=state["shadow"]


def checkpoint_payload(config:EOGResidualConfig,det:nn.Module,diff:nn.Module,ema:EMA,**extra:object)->dict[str,object]:
    return {"config":asdict(config),"det":det.state_dict(),"diff":diff.state_dict(),"ema":ema.state_dict(),**extra}


__all__=["EOGResidualConfig","DeterministicEOGResidual","EOGResidualDiffusion","EMA","checkpoint_payload"]
