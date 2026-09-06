"""V29 population-anchored support adapter for one-step clean prediction."""
from __future__ import annotations
from torch import Tensor, nn
from eeg_scad.models.support_adapter_common import SupportResidualAdapter


class SupportAdapterDET(nn.Module):
    forbidden_fields = ("query_EOG", "query_operator", "query_event", "subject_ID")
    def __init__(self, width: int = 32) -> None:
        super().__init__(); self.adapter = SupportResidualAdapter(46 * 3, width, False)

    def increment(self, y: Tensor, population: Tensor, context: Tensor) -> Tensor:
        return self.adapter(__import__('torch').cat((y, population, y-population), 1), context)

    def forward(self, y: Tensor, population: Tensor, context: Tensor, bypass: bool = False) -> Tensor:
        return population if bypass else population + self.increment(y, population, context)


__all__ = ["SupportAdapterDET"]
