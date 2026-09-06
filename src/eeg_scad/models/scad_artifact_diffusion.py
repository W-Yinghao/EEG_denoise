from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from .diffusion_schedule import cosine_alpha_bar,extract
from .eegdus_backbone import ArtifactBackbone


@dataclass(frozen=True)
class SCADConfig:
    channels:int=46;base_channels:int=32;context_input_dim:int=189;context_hidden_dim:int=256;context_dim:int=128;timesteps:int=1000;ddim_steps:int=25;parameterization:str="x0"


class SCADArtifactDiffusion(nn.Module):
    visible_fields=("artifact_state","contaminated_EEG","timestep","support_operator_context")
    forbidden_fields=("query_EOG","query_operator","query_event","clean_target_at_inference")
    def __init__(self,config:SCADConfig)->None:
        super().__init__();self.config=config;self.backbone=ArtifactBackbone(config.channels*2,config.channels,config.base_channels,config.context_input_dim,config.context_hidden_dim,config.context_dim,True);self.register_buffer("alpha_bar",cosine_alpha_bar(config.timesteps))
    def predict(self,state:Tensor,y:Tensor,timestep:Tensor,context:Tensor)->tuple[Tensor,Tensor]:
        raw=self.backbone(torch.cat((state,y),dim=1),context,timestep);alpha=extract(self.alpha_bar,timestep,state.ndim)
        if self.config.parameterization=="x0":x0=raw
        elif self.config.parameterization=="v":x0=alpha.sqrt()*state-(1-alpha).sqrt()*raw
        else:raise ValueError(self.config.parameterization)
        return raw,x0
    def training_loss(self,artifact:Tensor,y:Tensor,context:Tensor,generator:torch.Generator,timestep:Tensor|None=None,noise:Tensor|None=None)->tuple[Tensor,dict[str,Tensor]]:
        if timestep is None:timestep=torch.randint(0,self.config.timesteps,(len(artifact),),device=artifact.device,generator=generator)
        if noise is None:noise=torch.randn(artifact.shape,device=artifact.device,dtype=artifact.dtype,generator=generator)
        alpha=extract(self.alpha_bar,timestep,artifact.ndim);state=alpha.sqrt()*artifact+(1-alpha).sqrt()*noise;raw,x0=self.predict(state,y,timestep,context)
        target=artifact if self.config.parameterization=="x0" else alpha.sqrt()*noise-(1-alpha).sqrt()*artifact
        return (raw-target).square().mean(),{"predicted_x0":x0,"state":state,"noise":noise,"timestep":timestep}
    @torch.no_grad()
    def sample(self,y:Tensor,context:Tensor,initial_noise:Tensor,steps:int|None=None,trajectory:bool=False)->tuple[Tensor,list[dict[str,float]]]:
        state=initial_noise.clone();schedule=torch.linspace(self.config.timesteps-1,0,steps or self.config.ddim_steps,device=y.device).round().long();trace=[]
        for index,tvalue in enumerate(schedule):
            timestep=torch.full((len(y),),int(tvalue),device=y.device,dtype=torch.long);_,x0=self.predict(state,y,timestep,context);alpha=extract(self.alpha_bar,timestep,state.ndim);eps=(state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8)
            if trajectory:trace.append({"step":index,"timestep":int(tvalue),"state_rms":float(state.square().mean().sqrt()),"x0_rms":float(x0.square().mean().sqrt()),"max_abs":float(x0.abs().max())})
            if index+1==len(schedule):state=x0
            else:next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=extract(self.alpha_bar,next_t,state.ndim);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*eps
        return state,trace
    @torch.no_grad()
    def sample_mean(self,y:Tensor,context:Tensor,noise_bank:Tensor,steps:int|None=None)->Tensor:
        values=[self.sample(y,context,noise_bank[k],steps)[0] for k in range(len(noise_bank))];return torch.stack(values).mean(0)


def identity_postprocessor(*,observation:Tensor,clean_estimate:Tensor,artifact_estimate:Tensor,context:Tensor)->Tensor:
    del observation,artifact_estimate,context;return clean_estimate

