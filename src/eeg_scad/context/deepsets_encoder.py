from __future__ import annotations
import torch
from torch import Tensor,nn
from .support_window_encoder import SupportWindowEncoder

class DeepSetsSupportEncoder(nn.Module):
    permutation_invariant=True
    def __init__(self,dimension:int=128,rank:int=8)->None:
        super().__init__();self.windows=SupportWindowEncoder(token_dimension=dimension);self.score=nn.Linear(dimension,1);self.aggregate=nn.Sequential(nn.Linear(3*dimension,256),nn.SiLU(),nn.Linear(256,dimension),nn.LayerNorm(dimension));self.basis=nn.Linear(dimension,46*rank);self.operator=nn.Linear(dimension,46*4);self.rank=rank
    def forward(self,eeg:Tensor,eog:Tensor)->dict[str,Tensor]:
        tokens=self.windows(eeg,eog);weights=torch.softmax(self.score(tokens).squeeze(-1),1);weighted=(tokens*weights[...,None]).sum(1);context=self.aggregate(torch.cat((weighted,tokens.mean(1),tokens.max(1).values),1));raw=self.basis(context).reshape(len(context),46,self.rank);basis=raw/torch.linalg.vector_norm(raw,dim=1,keepdim=True).clamp_min(1e-8);return {"context":context,"basis":basis,"operator":self.operator(context).reshape(len(context),46,4),"attention":weights}

__all__=["DeepSetsSupportEncoder"]
