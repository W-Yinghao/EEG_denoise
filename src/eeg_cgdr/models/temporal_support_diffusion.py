"""EEG-space correction diffusion with raw temporal support cross-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .temporal_support_conditioner import CrossAttentionBlock, TemporalSupportEncoder


def cosine_alpha_bar(timesteps:int,offset:float=0.008)->Tensor:
    steps=torch.arange(timesteps+1,dtype=torch.float64);value=torch.cos(((steps/timesteps+offset)/(1+offset))*math.pi/2).square();value=value/value[0]
    return value[1:].clamp(1e-8,1.0).float()


class TemporalCorrectionDenoiser(nn.Module):
    forbidden_inputs=("query_EOG","query_IMU","query_event_label","participant_ID","query_outcome")
    def __init__(self,eeg_channels:int=46,width:int=96)->None:
        super().__init__();self.support_encoder=TemporalSupportEncoder(eeg_channels,27,4,width);self.input=nn.Conv1d(2*eeg_channels,width,7,padding=3)
        self.time=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,width));self.block1=CrossAttentionBlock(width);self.down=nn.Conv1d(width,width,4,2,1);self.block2=CrossAttentionBlock(width);self.up=nn.ConvTranspose1d(width,width,4,2,1);self.output=nn.Sequential(nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,eeg_channels,3,padding=1))
    @staticmethod
    def _time_embedding(timestep:Tensor,width:int)->Tensor:
        half=width;frequency=torch.exp(-math.log(10000)*torch.arange(half,device=timestep.device)/(max(half-1,1)));phase=timestep.float()[:,None]*frequency[None];return torch.cat((phase.sin(),phase.cos()),dim=1)
    def forward(self,noisy_correction:Tensor,timestep:Tensor,observed:Tensor,*,support_eeg:Tensor,support_imu:Tensor,support_eog:Tensor,modality_present:Tensor,context_present:Tensor)->Tensor:
        tokens=self.support_encoder(support_eeg,support_imu,support_eog,modality_present,context_present);hidden=self.input(torch.cat((noisy_correction,observed),dim=1));time=self.time(self._time_embedding(timestep,hidden.shape[1]))[:,:,None];first=self.block1(hidden+time,tokens);second=self.block2(self.down(first)+time,tokens);up=self.up(second)
        if up.shape[-1]!=first.shape[-1]:up=torch.nn.functional.interpolate(up,size=first.shape[-1],mode="linear",align_corners=False)
        return self.output(first+up+time)


@dataclass(frozen=True)
class TemporalDiffusionConfig:
    timesteps:int=1000
    ddim_steps:int=25
    posterior_samples:int=8
    min_snr_gamma:float=5.0
    correction_clip:float=5.0


class TemporalSupportCorrectionDiffusion(nn.Module):
    def __init__(self,config:TemporalDiffusionConfig=TemporalDiffusionConfig())->None:
        super().__init__();self.config=config;self.denoiser=TemporalCorrectionDenoiser();self.register_buffer("alpha_bar",cosine_alpha_bar(config.timesteps))
    def training_loss(self,target:Tensor,observed:Tensor,kwargs:dict[str,Tensor],generator:torch.Generator)->Tensor:
        target=target.clamp(-self.config.correction_clip,self.config.correction_clip);batch=target.shape[0];t=torch.randint(0,self.config.timesteps,(batch,),device=target.device,generator=generator);noise=torch.randn(target.shape,device=target.device,generator=generator);alpha=self.alpha_bar[t][:,None,None];noisy=alpha.sqrt()*target+(1-alpha).sqrt()*noise;truth=alpha.sqrt()*noise-(1-alpha).sqrt()*target;prediction=self.denoiser(noisy,t,observed,**kwargs);snr=alpha/(1-alpha).clamp_min(1e-8);weight=torch.minimum(snr,torch.full_like(snr,self.config.min_snr_gamma))/(snr+1).clamp_min(1e-8);return (weight*(prediction-truth).square()).mean()
    @torch.no_grad()
    def sample(self,observed:Tensor,kwargs:dict[str,Tensor],generator:torch.Generator,k:int|None=None)->Tensor:
        samples=self.config.posterior_samples if k is None else k;batch=observed.shape[0];states=torch.randn((samples*batch,*observed.shape[1:]),device=observed.device,generator=generator);condition=observed.repeat(samples,1,1);expanded={key:value.repeat(samples,*([1]*(value.ndim-1))) for key,value in kwargs.items()};indices=torch.linspace(self.config.timesteps-1,0,self.config.ddim_steps,device=observed.device).round().long().unique_consecutive()
        for position,t_value in enumerate(indices):
            timestep=torch.full((states.shape[0],),int(t_value),device=states.device,dtype=torch.long);alpha=self.alpha_bar[timestep][:,None,None];v=self.denoiser(states,timestep,condition,**expanded);x0=(alpha.sqrt()*states-(1-alpha).sqrt()*v).clamp(-self.config.correction_clip,self.config.correction_clip);epsilon=(1-alpha).sqrt()*states+alpha.sqrt()*v
            if position==len(indices)-1:states=x0
            else:
                previous=self.alpha_bar[indices[position+1]];states=previous.sqrt()*x0+(1-previous).sqrt()*epsilon
        return states.reshape(samples,batch,*states.shape[1:])

