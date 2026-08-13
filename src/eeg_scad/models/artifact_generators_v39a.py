"""Matched conditional artifact generators for V39A."""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SpatialArtifactCodec:
    basis: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, artifacts: np.ndarray, rank: int = 8) -> "SpatialArtifactCodec":
        samples=np.asarray(artifacts,dtype=np.float64).transpose(0,2,1).reshape(-1,artifacts.shape[1])
        _,_,vectors=np.linalg.svd(samples[::max(1,len(samples)//20000)],full_matrices=False)
        basis=vectors[:rank].T.astype(np.float32);latent=np.einsum("cr,nct->nrt",basis,artifacts)
        scale=latent.std(axis=(0,2));scale=np.where(scale>1e-6,scale,1.).astype(np.float32)
        return cls(basis,scale)
    def encode(self,a:np.ndarray)->np.ndarray:return (np.einsum("cr,nct->nrt",self.basis,a)/self.scale[None,:,None]).astype(np.float32)
    def decode(self,u:np.ndarray)->np.ndarray:return np.einsum("cr,nrt->nct",self.basis,u*self.scale[None,:,None]).astype(np.float32)


class ConditionalBlock(nn.Module):
    def __init__(self,width:int,condition:int,dilation:int):
        super().__init__();self.norm=nn.GroupNorm(8,width);self.conv=nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation);self.film=nn.Linear(condition,2*width)
    def forward(self,x:Tensor,c:Tensor)->Tensor:
        scale,shift=self.film(c).chunk(2,1);h=self.conv(torch.nn.functional.silu(self.norm(x)));return x+h*(1+.1*torch.tanh(scale[...,None]))+shift[...,None]


class ArtifactGenerator(nn.Module):
    def __init__(self,rank:int=8,condition:int=131,width:int=64,noise_dim:int=16):
        super().__init__();self.noise_dim=noise_dim;self.inp=nn.Conv1d(noise_dim,width,1);self.blocks=nn.ModuleList([ConditionalBlock(width,condition,d) for d in (1,2,4,8,16)]);self.out=nn.Conv1d(width,rank,1)
    def forward(self,noise:Tensor,condition:Tensor)->Tensor:
        h=self.inp(noise)
        for block in self.blocks:h=block(h,condition)
        return self.out(h)


class ArtifactCritic(nn.Module):
    def __init__(self,rank:int=8,condition:int=131,width:int=64):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(rank,width,7,padding=3),nn.LeakyReLU(.2),nn.Conv1d(width,width,7,stride=2,padding=3),nn.LeakyReLU(.2),nn.Conv1d(width,2*width,7,stride=2,padding=3),nn.LeakyReLU(.2));self.out=nn.Linear(2*width+condition,1)
    def forward(self,x:Tensor,c:Tensor)->Tensor:return self.out(torch.cat((self.net(x).mean(-1),c),1)).squeeze(1)


def cosine_alpha_bar(timesteps:int=1000)->Tensor:
    t=torch.arange(timesteps+1,dtype=torch.float64);v=torch.cos(((t/timesteps+.008)/1.008)*math.pi/2)**2;return (v/v[0])[:-1].float().clamp(1e-6,1)


class TimeEmbedding(nn.Module):
    def __init__(self,dim:int=32):super().__init__();self.dim=dim;self.net=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,dim))
    def forward(self,t:Tensor)->Tensor:
        half=self.dim//2;f=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1));a=t.float()[:,None]*f[None];return self.net(torch.cat((a.sin(),a.cos()),1))


class ConditionalArtifactDiffusion(nn.Module):
    """Stable x0-prediction diffusion in a fixed spatial artifact basis."""
    def __init__(self,rank:int=8,condition:int=131,width:int=64,timesteps:int=1000):
        super().__init__();self.register_buffer("alpha_bar",cosine_alpha_bar(timesteps));self.time=TimeEmbedding();self.inp=nn.Conv1d(rank,width,1);self.blocks=nn.ModuleList([ConditionalBlock(width,condition+32,d) for d in (1,2,4,8,16)]);self.out=nn.Conv1d(width,rank,1)
    def forward(self,state:Tensor,condition:Tensor,t:Tensor)->Tensor:
        c=torch.cat((condition,self.time(t)),1);h=self.inp(state)
        for block in self.blocks:h=block(h,c)
        return self.out(h)
    def q_sample(self,x0:Tensor,t:Tensor,noise:Tensor)->Tensor:
        a=self.alpha_bar[t].to(x0.dtype)[:,None,None];return a.sqrt()*x0+(1-a).sqrt()*noise
    @torch.no_grad()
    def sample(self,condition:Tensor,noise:Tensor,steps:int=10)->Tensor:
        state=noise.clone();schedule=torch.linspace(len(self.alpha_bar)-1,0,steps,device=state.device).round().long()
        for i,tv in enumerate(schedule):
            t=torch.full((len(state),),int(tv),device=state.device,dtype=torch.long);x0=self(state,condition,t)
            if i+1==len(schedule):state=x0;continue
            a=self.alpha_bar[tv].to(state.dtype);an=self.alpha_bar[schedule[i+1]].to(state.dtype);eps=(state-a.sqrt()*x0)/(1-a).sqrt().clamp_min(1e-8);state=an.sqrt()*x0+(1-an).sqrt()*eps
        return state


class ConditionalArtifactGaussian:
    def __init__(self,mean_coef:np.ndarray,mean_intercept:np.ndarray,cov:dict[int,np.ndarray]):self.mean_coef=mean_coef;self.mean_intercept=mean_intercept;self.cov=cov
    @classmethod
    def fit(cls,latent:np.ndarray,condition:np.ndarray,severity_bin:np.ndarray):
        from sklearn.linear_model import Ridge
        flat=latent.reshape(len(latent),-1);reg=Ridge(alpha=10.).fit(condition,flat);res=flat-reg.predict(condition);cov={}
        for level in range(3):
            chosen=res[severity_bin==level] if np.sum(severity_bin==level)>=8 else res
            variance=np.var(chosen,axis=0)+1e-4;cov[level]=np.sqrt(variance).astype(np.float32)
        return cls(reg.coef_.T.astype(np.float32),np.asarray(reg.intercept_,np.float32),cov)
    def sample(self,condition:np.ndarray,severity_bin:np.ndarray,seed:int)->np.ndarray:
        rng=np.random.default_rng(seed);mean=condition@self.mean_coef+self.mean_intercept;return np.stack([row+rng.normal(size=row.shape).astype(np.float32)*self.cov[int(level)] for row,level in zip(mean,severity_bin)]).reshape(len(mean),8,256).astype(np.float32)


class SupportDenoiserV39(nn.Module):
    """One fixed deterministic support-conditioned denoiser for every arm."""
    def __init__(self,channels:int=46,condition:int=128,width:int=64):
        super().__init__();self.inp=nn.Conv1d(channels,width,7,padding=3);self.blocks=nn.ModuleList([ConditionalBlock(width,condition,d) for d in (1,2,4,8,16,32)]);self.out=nn.Conv1d(width,channels,7,padding=3)
    def forward(self,y:Tensor,context:Tensor)->Tensor:
        h=self.inp(y)
        for block in self.blocks:h=block(h,context)
        return y-self.out(h)


__all__=["ArtifactCritic","ArtifactGenerator","ConditionalArtifactDiffusion","ConditionalArtifactGaussian","SpatialArtifactCodec","SupportDenoiserV39"]
