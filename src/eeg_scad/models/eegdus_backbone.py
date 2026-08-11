"""Clean-room EEGDfus-style multichannel CNN/attention backbone with FiLM."""
from __future__ import annotations

import math
import torch
from torch import Tensor,nn
from eeg_scad.context.context_encoder import OperatorContextEncoder


def sinusoidal_embedding(timestep:Tensor,dimension:int)->Tensor:
    half=dimension//2;freq=torch.exp(torch.arange(half,device=timestep.device,dtype=torch.float32)*(-math.log(10000.)/max(half-1,1)))
    angle=timestep.float()[:,None]*freq[None];return torch.cat((torch.sin(angle),torch.cos(angle)),dim=1)


class FiLMResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,condition_dim:int,groups:int=8)->None:
        super().__init__();self.norm1=nn.GroupNorm(min(groups,in_channels),in_channels);self.conv1=nn.Conv1d(in_channels,out_channels,3,padding=1)
        self.norm2=nn.GroupNorm(min(groups,out_channels),out_channels);self.conv2=nn.Conv1d(out_channels,out_channels,3,padding=1);self.film=nn.Linear(condition_dim,2*out_channels)
        self.skip=nn.Conv1d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity();self.act=nn.SiLU()
    def forward(self,x:Tensor,condition:Tensor)->Tensor:
        h=self.conv1(self.act(self.norm1(x)));scale,shift=self.film(condition).chunk(2,dim=1);h=self.norm2(h)*(1+scale[:,:,None])+shift[:,:,None]
        return self.skip(x)+self.conv2(self.act(h))


class TemporalAttention(nn.Module):
    def __init__(self,channels:int,heads:int=4)->None:
        super().__init__();self.norm=nn.LayerNorm(channels);self.attn=nn.MultiheadAttention(channels,heads,batch_first=True);self.proj=nn.Sequential(nn.LayerNorm(channels),nn.Linear(channels,4*channels),nn.SiLU(),nn.Linear(4*channels,channels))
    def forward(self,x:Tensor)->Tensor:
        value=x.transpose(1,2);norm=self.norm(value);attn,_=self.attn(norm,norm,norm,need_weights=False);value=value+attn;return (value+self.proj(value)).transpose(1,2)


class ArtifactBackbone(nn.Module):
    def __init__(self,in_channels:int,eeg_channels:int=46,base:int=32,context_input_dim:int=189,context_hidden:int=256,context_dim:int=128,time_conditioned:bool=True)->None:
        super().__init__();self.time_conditioned=time_conditioned;self.context_encoder=OperatorContextEncoder(context_input_dim,context_hidden,context_dim)
        if time_conditioned:self.time_mlp=nn.Sequential(nn.Linear(context_dim,context_dim*2),nn.SiLU(),nn.Linear(context_dim*2,context_dim))
        self.input=nn.Conv1d(in_channels,base,3,padding=1)
        self.b1a=FiLMResidualBlock(base,base,context_dim);self.b1b=FiLMResidualBlock(base,base,context_dim);self.down1=nn.Conv1d(base,base*2,4,stride=2,padding=1)
        self.b2a=FiLMResidualBlock(base*2,base*2,context_dim);self.b2b=FiLMResidualBlock(base*2,base*2,context_dim);self.down2=nn.Conv1d(base*2,base*4,4,stride=2,padding=1)
        self.mid1=FiLMResidualBlock(base*4,base*4,context_dim);self.attn=TemporalAttention(base*4,4);self.mid2=FiLMResidualBlock(base*4,base*4,context_dim)
        self.up2=nn.ConvTranspose1d(base*4,base*2,4,stride=2,padding=1);self.u2a=FiLMResidualBlock(base*4,base*2,context_dim);self.u2b=FiLMResidualBlock(base*2,base*2,context_dim)
        self.up1=nn.ConvTranspose1d(base*2,base,4,stride=2,padding=1);self.u1a=FiLMResidualBlock(base*2,base,context_dim);self.u1b=FiLMResidualBlock(base,base,context_dim)
        self.out=nn.Sequential(nn.GroupNorm(8,base),nn.SiLU(),nn.Conv1d(base,eeg_channels,3,padding=1))
    def condition(self,context:Tensor,timestep:Tensor|None)->Tensor:
        result=self.context_encoder(context)
        if self.time_conditioned:
            if timestep is None:raise ValueError("diffusion backbone requires timestep")
            result=result+self.time_mlp(sinusoidal_embedding(timestep,result.shape[1]))
        return result
    def forward(self,x:Tensor,context:Tensor,timestep:Tensor|None=None)->Tensor:
        cond=self.condition(context,timestep);h=self.input(x);s1=self.b1b(self.b1a(h,cond),cond);h=self.down1(s1);s2=self.b2b(self.b2a(h,cond),cond);h=self.down2(s2)
        h=self.mid2(self.attn(self.mid1(h,cond)),cond);h=self.up2(h);h=self.u2b(self.u2a(torch.cat((h,s2),dim=1),cond),cond);h=self.up1(h);h=self.u1b(self.u1a(torch.cat((h,s1),dim=1),cond),cond);return self.out(h)

