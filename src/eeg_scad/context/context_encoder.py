from __future__ import annotations

from torch import Tensor,nn


class OperatorContextEncoder(nn.Module):
    def __init__(self,input_dim:int=189,hidden_dim:int=256,output_dim:int=128)->None:
        super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.SiLU(),nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,output_dim),nn.SiLU(),nn.LayerNorm(output_dim))
    def forward(self,value:Tensor)->Tensor:return self.net(value)

