"""Capacity-matched population-only V29 diffusion adapter."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from eeg_scad.models.support_adapter_cdm import SupportAdapterCDM


class PopAdapterCDM(SupportAdapterCDM):
    uses_subject_support = False
    def __init__(self, width: int = 32) -> None:
        super().__init__(width); self.population_token = nn.Parameter(torch.zeros(1, 128))
    def context(self, batch: int) -> Tensor:
        return self.population_token.expand(batch, -1)
    @torch.no_grad()
    def sample(self, population_model: nn.Module, y: Tensor, noise: Tensor, steps: int = 10):
        return super().sample(population_model,y,self.context(len(y)),noise,steps,False)


__all__ = ["PopAdapterCDM"]
