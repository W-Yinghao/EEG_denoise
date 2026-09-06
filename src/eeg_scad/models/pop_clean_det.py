"""Subject-agnostic matched one-step clean predictor."""
from __future__ import annotations

import torch
from torch import Tensor

from eeg_scad.models.support_clean_det import SupportCleanDET


class PopCleanDET(SupportCleanDET):
    uses_subject_support=False

    def __init__(self,width:int=64)->None:
        super().__init__(width);self.register_buffer("population_context",torch.zeros(1,128))

    def forward(self,y:Tensor)->Tensor:
        return super().forward(y,self.population_context.expand(len(y),-1))


__all__=["PopCleanDET"]
