from __future__ import annotations
from typing import Literal
import torch
from torch import Tensor,nn
from eeg_scad.context.deepsets_encoder import DeepSetsSupportEncoder
from eeg_scad.context.set_transformer_encoder import SetTransformerSupportEncoder
from eeg_scad.context.learned_spatial_decoder import decode_residual
from eeg_scad.models.population_anchor_v24 import _ResidualBlock

class SetCalibDET(nn.Module):
    forbidden_fields=("query_EOG","query_operator","query_event","subject_ID")
    def __init__(self,encoder:Literal["deepsets","set_transformer"]="deepsets",rank:int=8,width:int=96)->None:
        super().__init__();self.rank=rank;self.support=DeepSetsSupportEncoder(rank=rank) if encoder=="deepsets" else SetTransformerSupportEncoder(rank=rank);self.inp=nn.Conv1d(46*3+4,width,1);self.blocks=nn.Sequential(*[_ResidualBlock(width,d) for d in (1,2,4,8,16,32)]);layer=nn.TransformerEncoderLayer(width,4,2*width,batch_first=True,norm_first=True);self.attention=nn.TransformerEncoder(layer,1);self.film=nn.Linear(128,2*width);self.out=nn.Conv1d(width,rank,1);self.encoder_name=encoder
    def encode_support(self,eeg:Tensor,eog:Tensor)->dict[str,Tensor]:return self.support(eeg,eog)
    def coefficient(self,y:Tensor,a0:Tensor,q0:Tensor,context:Tensor)->Tensor:
        hidden=self.inp(torch.cat((y,a0,y-a0,q0),1));scale,shift=self.film(context).chunk(2,1);hidden=self.blocks(hidden*(1+0.1*torch.tanh(scale)[...,None])+shift[...,None]);hidden=self.attention(hidden.transpose(1,2)).transpose(1,2);return self.out(hidden)
    def forward(self,y:Tensor,a0:Tensor,q0:Tensor,support_eeg:Tensor,support_eog:Tensor)->dict[str,Tensor]:
        encoded=self.encode_support(support_eeg,support_eog);coefficient=self.coefficient(y,a0,q0,encoded["context"]);artifact=decode_residual(a0,encoded["basis"],coefficient);return {**encoded,"coefficient":coefficient,"artifact":artifact,"clean":y-artifact}
    @staticmethod
    def population(y:Tensor,a0:Tensor)->dict[str,Tensor]:return {"artifact":a0,"clean":y-a0}

__all__=["SetCalibDET"]
