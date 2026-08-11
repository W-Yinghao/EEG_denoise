from __future__ import annotations

import torch
from torch import Tensor, nn

from .population_anchor_v24 import PopulationAnchorV24
from .temporal_eog_net import TemporalEOGNet


def decode_deviation(a_pop: Tensor, deviation: Tensor, latent: Tensor) -> Tensor:
    if deviation.ndim != 3 or latent.ndim != 3:
        raise ValueError("deviation and latent must be batched matrices")
    return a_pop + torch.einsum("bcd,bdt->bct", deviation, latent)


class PAELDet(nn.Module):
    """Frozen population anchor plus shared EOG latent and context decoder."""

    forbidden_fields = ("query_EOG", "query_operator", "event", "participant_ID")

    def __init__(self, anchor: PopulationAnchorV24, temporal: TemporalEOGNet) -> None:
        super().__init__()
        self.anchor = anchor
        self.temporal = temporal

    def forward(self, y: Tensor, q0: Tensor, c0: Tensor, deviation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projected0 = torch.einsum("bcd,bdt->bct", c0, q0)
        a_pop = self.anchor(y, q0, projected0)
        latent = self.temporal(y, a_pop, q0)
        artifact = decode_deviation(a_pop, deviation, latent)
        return artifact, latent, a_pop


__all__ = ["PAELDet", "decode_deviation"]

