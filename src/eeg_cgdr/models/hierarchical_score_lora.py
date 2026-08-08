"""Low-dimensional hierarchical score residuals for V10.

Four shared rank-one directions are learned on outer-training participants.
An unseen participant adapts only a bounded four-vector; alpha=0 is exactly
the frozen population score network.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class HierarchicalLoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, *, directions: int = 4) -> None:
        super().__init__()
        if base.groups != 1 or directions != 4:
            raise ValueError("V10 freezes four ungrouped score directions")
        self.base=base
        for parameter in base.parameters():parameter.requires_grad_(False)
        self.down=nn.ModuleList([nn.Conv1d(base.in_channels,1,1,bias=False) for _ in range(directions)])
        self.up=nn.ModuleList([nn.Conv1d(1,base.out_channels,base.kernel_size,stride=base.stride,padding=base.padding,dilation=base.dilation,bias=False) for _ in range(directions)])
        for down,up in zip(self.down,self.up):
            down.to(base.weight);up.to(base.weight);nn.init.normal_(down.weight,std=base.in_channels**-.5);nn.init.zeros_(up.weight)
        self._alpha:Tensor|None=None

    def set_alpha(self,alpha:Tensor|None)->None:self._alpha=alpha

    def forward(self,value:Tensor)->Tensor:
        result=self.base(value)
        if self._alpha is None:return result
        alpha=self._alpha
        if alpha.ndim==1:alpha=alpha[None].expand(value.shape[0],-1)
        if alpha.shape!=(value.shape[0],4):raise ValueError("alpha must be (B,4)")
        for index,(down,up) in enumerate(zip(self.down,self.up)):result=result+alpha[:,index,None,None]*up(down(value))
        return result


@dataclass(frozen=True)
class HierarchicalSummary:
    directions:int
    adapted_convolutions:int
    shared_parameters:int


def inject_hierarchical_score_lora(module:nn.Module)->HierarchicalSummary:
    targets=[]
    for parameter in module.parameters():parameter.requires_grad_(False)
    for child in module.modules():
        for name in ("conv1","conv2"):
            value=getattr(child,name,None)
            if isinstance(value,nn.Conv1d):targets.append((child,name,value))
    if not targets:raise ValueError("no score ResBlock convolutions")
    for parent,name,base in targets:setattr(parent,name,HierarchicalLoRAConv1d(base))
    count=sum(p.numel() for p in module.parameters() if p.requires_grad)
    return HierarchicalSummary(4,len(targets),count)


def set_hierarchical_alpha(module:nn.Module,alpha:Tensor|None)->None:
    found=0
    for child in module.modules():
        if isinstance(child,HierarchicalLoRAConv1d):child.set_alpha(alpha);found+=1
    if not found:raise ValueError("hierarchical score directions are absent")


def shared_direction_parameters(module:nn.Module)->list[nn.Parameter]:
    return [p for child in module.modules() if isinstance(child,HierarchicalLoRAConv1d) for p in child.parameters() if p.requires_grad]


__all__=["HierarchicalLoRAConv1d","HierarchicalSummary","inject_hierarchical_score_lora","set_hierarchical_alpha","shared_direction_parameters"]
