from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor,nn
from eeg_scad.context.operator_factorization import decode_torch
from eeg_scad.models.diffusion_schedule import cosine_alpha_bar,extract
from .of_deterministic import CoefficientTemporalNet


@dataclass(frozen=True)
class OFSCADConfig:
    coefficient_channels:int=8;base:int=48;timesteps:int=1000;ddim_steps:int=25


class OFResidualDiffusion(nn.Module):
    forbidden_fields=("query_EOG","query_operator","clean_target_at_inference")
    def __init__(self,config:OFSCADConfig=OFSCADConfig())->None:
        super().__init__();self.config=config;self.net=CoefficientTemporalNet(124,config.coefficient_channels,config.base,24,True);self.register_buffer("alpha_bar",cosine_alpha_bar(config.timesteps))
    def features(self,state:Tensor,y:Tensor,q:Tensor,projected:Tensor,zdet:Tensor)->Tensor:return torch.cat((state,y,projected,q,zdet),1)
    def predict(self,state:Tensor,y:Tensor,q:Tensor,projected:Tensor,zdet:Tensor,summary:Tensor,timestep:Tensor)->Tensor:return self.net(self.features(state,y,q,projected,zdet),summary,timestep)
    def training_loss(self,target:Tensor,y:Tensor,q:Tensor,projected:Tensor,zdet:Tensor,summary:Tensor,generator:torch.Generator,timestep:Tensor|None=None,noise:Tensor|None=None)->tuple[Tensor,dict[str,Tensor]]:
        if timestep is None:timestep=torch.randint(0,self.config.timesteps,(len(target),),device=target.device,generator=generator)
        if noise is None:noise=torch.randn(target.shape,device=target.device,dtype=target.dtype,generator=generator)
        alpha=extract(self.alpha_bar,timestep,target.ndim);state=alpha.sqrt()*target+(1-alpha).sqrt()*noise;x0=self.predict(state,y,q,projected,zdet,summary,timestep);return (x0-target).square().mean(),{"state":state,"predicted_x0":x0,"noise":noise,"timestep":timestep}
    @torch.no_grad()
    def sample(self,y:Tensor,q:Tensor,projected:Tensor,zdet:Tensor,summary:Tensor,initial_noise:Tensor,steps:int|None=None,trajectory:bool=False)->tuple[Tensor,list[dict[str,float]]]:
        state=initial_noise.clone();schedule=torch.linspace(self.config.timesteps-1,0,steps or self.config.ddim_steps,device=y.device).round().long();trace=[]
        for index,tvalue in enumerate(schedule):
            t=torch.full((len(y),),int(tvalue),device=y.device,dtype=torch.long);x0=self.predict(state,y,q,projected,zdet,summary,t);alpha=extract(self.alpha_bar,t,state.ndim);eps=(state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8)
            if trajectory:trace.append({"step":index,"timestep":int(tvalue),"r_t_rms":float(state.square().mean().sqrt()),"r_hat_rms":float(x0.square().mean().sqrt()),"z_det_rms":float(zdet.square().mean().sqrt()),"z_final_rms":float((zdet+x0).square().mean().sqrt()),"max_abs":float(x0.abs().max())})
            if index+1==len(schedule):state=x0
            else:nt=torch.full_like(t,int(schedule[index+1]));na=extract(self.alpha_bar,nt,state.ndim);state=na.sqrt()*x0+(1-na).sqrt()*eps
        return state,trace
    @torch.no_grad()
    def artifact(self,y:Tensor,q:Tensor,projected:Tensor,zdet:Tensor,summary:Tensor,basis:Tensor,noise:Tensor,steps:int|None=None)->Tensor:
        residual,_=self.sample(y,q,projected,zdet,summary,noise,steps);return decode_torch(basis,zdet+residual)


class PopulationMarginalSCAD(OFResidualDiffusion):
    uses_subject_support=False

