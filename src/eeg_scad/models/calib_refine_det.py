"""Matched one-step sensor-coordinate artifact refiners for V26."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from eeg_scad.models.population_anchor_v24 import _ResidualBlock


class ArtifactRefinerBackbone(nn.Module):
    """Capacity-matched temporal backbone shared by one-step and diffusion models."""

    def __init__(self, input_channels: int, width: int = 64, context_dimension: int = 128) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_channels, width, 1)
        self.blocks = nn.ModuleList([_ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 32)])
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=.1, activation="gelu", batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, 1)
        self.film = nn.ModuleList([nn.Linear(context_dimension, 2 * width) for _ in self.blocks])
        self.output = nn.Conv1d(width, 46, 1)

    def forward(self, value: Tensor, context: Tensor) -> Tensor:
        hidden = self.input(value)
        for block, film in zip(self.blocks, self.film):
            scale, shift = film(context).chunk(2, 1)
            hidden = block(hidden * (1 + .1 * torch.tanh(scale)[..., None]) + shift[..., None])
        hidden = self.attention(hidden.transpose(1, 2)).transpose(1, 2)
        return self.output(hidden)


class CalibRefineDET(nn.Module):
    """One-step support-conditioned refiner anchored at the frozen V25 estimate."""

    forbidden_fields = ("query_EOG", "query_operator", "query_event", "subject_ID")

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.backbone = ArtifactRefinerBackbone(46 * 5, width)

    def forward(self, y: Tensor, artifact_det: Tensor, artifact_pop: Tensor, context: Tensor) -> Tensor:
        clean_det = y - artifact_det
        delta = artifact_det - artifact_pop
        residual = self.backbone(torch.cat((y, artifact_det, clean_det, artifact_pop, delta), 1), context)
        return artifact_det + .1 * residual


class PopRefineDET(nn.Module):
    """Independent subject-agnostic one-step population refiner."""

    uses_subject_support = False

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.backbone = ArtifactRefinerBackbone(46 * 3, width)
        self.register_buffer("population_context", torch.zeros(1, 128))

    def forward(self, y: Tensor, artifact_pop: Tensor) -> Tensor:
        context = self.population_context.expand(len(y), -1)
        residual = self.backbone(torch.cat((y, artifact_pop, y - artifact_pop), 1), context)
        return artifact_pop + .1 * residual


__all__ = ["ArtifactRefinerBackbone", "CalibRefineDET", "PopRefineDET"]
