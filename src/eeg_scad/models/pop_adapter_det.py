"""Capacity-matched population-only V29 one-step adapter."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from eeg_scad.models.support_adapter_det import SupportAdapterDET


class PopAdapterDET(SupportAdapterDET):
    uses_subject_support = False
    def __init__(self, width: int = 32) -> None:
        super().__init__(width); self.population_token = nn.Parameter(torch.zeros(1, 128))
    def forward(self, y: Tensor, population: Tensor) -> Tensor:
        return super().forward(y, population, self.population_token.expand(len(y), -1))


__all__ = ["PopAdapterDET"]
