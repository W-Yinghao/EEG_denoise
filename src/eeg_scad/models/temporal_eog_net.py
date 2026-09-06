from __future__ import annotations

import torch
from torch import Tensor, nn

from .population_anchor_v24 import _ResidualBlock


class TemporalEOGNet(nn.Module):
    """Predict a physically defined EOG latent from query EEG and POP anchor."""

    forbidden_fields = ("query_EOG", "query_operator", "event", "subject_operator")

    def __init__(self, eeg_channels: int = 46, eog_dimensions: int = 4, width: int = 96) -> None:
        super().__init__()
        self.eog_dimensions = eog_dimensions
        self.inp = nn.Conv1d(3 * eeg_channels + eog_dimensions, width, 1)
        self.encoder = nn.Sequential(*[_ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 32)])
        self.down = nn.Conv1d(width, 128, 4, stride=2, padding=1)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.bottleneck = nn.TransformerEncoder(layer, 2)
        self.up = nn.ConvTranspose1d(128, width, 4, stride=2, padding=1)
        self.out = nn.Sequential(_ResidualBlock(2 * width), nn.Conv1d(2 * width, eog_dimensions, 1))

    def forward(self, y: Tensor, a_pop: Tensor, q0: Tensor) -> Tensor:
        x_pop = y - a_pop
        skip = self.encoder(self.inp(torch.cat((y, a_pop, x_pop, q0), dim=1)))
        hidden = self.down(skip)
        hidden = self.bottleneck(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.up(hidden)
        if hidden.shape[-1] != skip.shape[-1]:
            hidden = torch.nn.functional.interpolate(hidden, size=skip.shape[-1], mode="linear", align_corners=False)
        return self.out(torch.cat((hidden, skip), dim=1))


__all__ = ["TemporalEOGNet"]

