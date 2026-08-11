"""Deterministic coefficient estimator with explicit operator decoding."""
from __future__ import annotations
import torch
from torch import Tensor,nn
from eeg_scad.models.eegdus_backbone import FiLMResidualBlock,TemporalAttention
from eeg_scad.context.operator_factorization import decode_torch


class CoefficientTemporalNet(nn.Module):
    def __init__(self,input_channels:int=108,output_channels:int=8,base:int=48,summary_dim:int=24,time_conditioned:bool=False)->None:
        super().__init__();self.time_conditioned=time_conditioned;condition_dim=128
        self.summary=nn.Sequential(nn.Linear(summary_dim,128),nn.SiLU(),nn.LayerNorm(128),nn.Linear(128,128))
        if time_conditioned:
            from eeg_scad.models.eegdus_backbone import sinusoidal_embedding
            self._embed=sinusoidal_embedding;self.time=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,128))
        self.inp=nn.Conv1d(input_channels,base,3,padding=1);self.b1=FiLMResidualBlock(base,base,condition_dim);self.down=nn.Conv1d(base,2*base,4,2,1);self.b2=FiLMResidualBlock(2*base,2*base,condition_dim);self.attn=TemporalAttention(2*base,4);self.up=nn.ConvTranspose1d(2*base,base,4,2,1);self.outblock=FiLMResidualBlock(2*base,base,condition_dim);self.out=nn.Conv1d(base,output_channels,3,padding=1)
    def forward(self,features:Tensor,summary:Tensor,timestep:Tensor|None=None)->Tensor:
        cond=self.summary(summary)
        if self.time_conditioned:
            if timestep is None:raise ValueError("timestep required")
            cond=cond+self.time(self._embed(timestep,128))
        x=self.inp(features);skip=self.b1(x,cond);x=self.attn(self.b2(self.down(skip),cond));x=self.up(x);return self.out(self.outblock(torch.cat((x,skip),1),cond))


class OFDeterministic(nn.Module):
    forbidden_fields=("query_EOG","query_operator","clean_target_at_inference")
    def __init__(self,base:int=48)->None:super().__init__();self.net=CoefficientTemporalNet(108,8,base,24,False)
    def forward(self,y:Tensor,q:Tensor,projected:Tensor,summary:Tensor)->Tensor:return self.net(torch.cat((y,projected,q),1),summary)
    def artifact(self,y:Tensor,q:Tensor,projected:Tensor,summary:Tensor,basis:Tensor)->Tensor:return decode_torch(basis,self(y,q,projected,summary))


class PopulationMarginalDET(OFDeterministic):
    uses_subject_support=False

