"""Observation-anchored clean-EEG posterior diffusion for BCI2b V12."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import torch
from torch import Tensor,nn
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D
from eeg_cgdr.models.artifact_latent_diffusion import cosine_alpha_bar
from eeg_cgdr.models.eog_residual_diffusion import EMA

@dataclass(frozen=True)
class CleanPosteriorConfig:
    channels:int=3
    signal_length:int=512
    base_channels:int=32
    timesteps:int=1000
    ddim_steps:int=25
    posterior_samples:int=8
    def __post_init__(self)->None:
        if self.signal_length%8:raise ValueError("signal length must be divisible by eight")
        if self.ddim_steps!=25 or self.posterior_samples!=8:raise ValueError("V12 freezes DDIM25/K8")

def _extract(value:Tensor,timestep:Tensor,ndim:int)->Tensor:
    return value.gather(0,timestep).reshape(len(timestep),*((1,)*(ndim-1)))

class _CleanUNet(nn.Module):
    def __init__(self,config:CleanPosteriorConfig,*,diffusion:bool)->None:
        super().__init__();inputs=config.channels*3 if diffusion else config.channels*2
        self.diffusion=diffusion;self.unet=UNet1D(ModelConfig(in_channels=inputs,out_channels=config.channels,signal_length=config.signal_length,base_channels=config.base_channels,channel_mults=[1,2,4],num_res_blocks=2,groupnorm_groups=8,dropout=.05,time_sinusoidal_dim=64,time_embed_dim=256,attention_length=64,attention_heads=4),subject_conditioned=False)
    def forward(self,state:Tensor|None,timestep:Tensor,*,x_lin:Tensor,a0:Tensor)->Tensor:
        fields=[x_lin,a0]
        if self.diffusion:
            if state is None:raise ValueError("clean diffusion requires a clean-state tensor")
            fields.insert(0,state)
        return self.unet(torch.cat(fields,dim=1),timestep)

class DeterministicCleanEstimator(nn.Module):
    visible_fields=("linear_clean_EEG","EOG_anchored_artifact")
    forbidden_fields=("clean_target","query_labels","query_transfer","query_outcomes")
    def __init__(self,config:CleanPosteriorConfig)->None:super().__init__();self.config=config;self.backbone=_CleanUNet(config,diffusion=False)
    def forward(self,*,x_lin:Tensor,a0:Tensor)->Tensor:
        t=torch.zeros(len(x_lin),dtype=torch.long,device=x_lin.device);return self.backbone(None,t,x_lin=x_lin,a0=a0)

class CleanPosteriorDiffusion(nn.Module):
    visible_fields=DeterministicCleanEstimator.visible_fields+("noisy_clean_state",)
    forbidden_fields=DeterministicCleanEstimator.forbidden_fields
    def __init__(self,config:CleanPosteriorConfig)->None:
        super().__init__();self.config=config;self.backbone=_CleanUNet(config,diffusion=True);_,alpha=cosine_alpha_bar(config.timesteps);self.register_buffer("alpha_bar",alpha.float())
    def training_loss(self,target:Tensor,*,x_lin:Tensor,a0:Tensor,generator:torch.Generator,timestep:Tensor|None=None,noise:Tensor|None=None,observation_anchored:bool=True)->tuple[Tensor,dict[str,Tensor]]:
        if timestep is None:timestep=torch.randint(0,self.config.timesteps,(len(target),),device=target.device,generator=generator)
        if noise is None:noise=torch.randn(target.shape,device=target.device,dtype=target.dtype,generator=generator)
        alpha=_extract(self.alpha_bar,timestep,target.ndim)
        if observation_anchored:
            state=alpha.sqrt()*x_lin+(1-alpha).sqrt()*noise
            truth=(alpha.sqrt()*state-target)/(1-alpha).sqrt().clamp_min(1e-6)
        else:
            state=alpha.sqrt()*target+(1-alpha).sqrt()*noise;truth=alpha.sqrt()*noise-(1-alpha).sqrt()*target
        prediction=self.backbone(state,timestep,x_lin=x_lin,a0=a0);x0=alpha.sqrt()*state-(1-alpha).sqrt()*prediction;return (prediction-truth).square().mean(),{"predicted_x0":x0,"timestep":timestep,"noise":noise}
    @torch.no_grad()
    def sample(self,*,x_lin:Tensor,a0:Tensor,t_start:int,initial_noise:Tensor,observation_variance:Tensor)->Tensor:
        if t_start==0:return x_lin.clone()
        if initial_noise.shape!=x_lin.shape:raise ValueError("initial noise shape mismatch")
        first=torch.full((len(x_lin),),int(t_start),device=x_lin.device,dtype=torch.long);alpha=_extract(self.alpha_bar,first,x_lin.ndim);state=alpha.sqrt()*x_lin+(1-alpha).sqrt()*initial_noise
        schedule=torch.linspace(t_start,0,self.config.ddim_steps,device=x_lin.device).round().long().unique_consecutive();obs_var=observation_variance.reshape(1,-1,1).clamp_min(1e-6)
        for index,t_value in enumerate(schedule):
            timestep=torch.full((len(state),),int(t_value),device=state.device,dtype=torch.long);alpha=_extract(self.alpha_bar,timestep,state.ndim);v=self.backbone(state,timestep,x_lin=x_lin,a0=a0);x0=alpha.sqrt()*state-(1-alpha).sqrt()*v;epsilon=(1-alpha).sqrt()*state+alpha.sqrt()*v
            if index+1==len(schedule):step_var=(1-alpha).clamp_min(1e-8)
            else:
                next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=_extract(self.alpha_bar,next_t,state.ndim);step_var=(1-alpha/next_alpha).clamp_min(1e-8)
            # Closed Gaussian proximal using the covariance of this reverse
            # transition, not the cumulative forward variance.
            x0=(x0/step_var+x_lin/obs_var)/(1/step_var+1/obs_var)
            if index+1==len(schedule):state=x0
            else:state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
        return state

def checkpoint_payload(config:CleanPosteriorConfig,det:nn.Module,diff:nn.Module,ema:EMA,**extra:object)->dict[str,object]:
    return {"config":asdict(config),"det":det.state_dict(),"diff":diff.state_dict(),"ema":ema.state_dict(),**extra}

__all__=["CleanPosteriorConfig","DeterministicCleanEstimator","CleanPosteriorDiffusion","EMA","checkpoint_payload"]
