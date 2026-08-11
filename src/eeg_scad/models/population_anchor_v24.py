from __future__ import annotations

import torch
from torch import Tensor, nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.block(value)


class PopulationAnchorV24(nn.Module):
    """Strong subject-blind full-field artifact anchor."""

    uses_subject_support = False
    forbidden_fields = ("query_EOG", "query_operator", "subject_ID", "clean_target_at_inference")

    def __init__(self, eeg_channels: int = 46, eog_dimensions: int = 4, width: int = 64) -> None:
        super().__init__()
        self.inp = nn.Conv1d(2 * eeg_channels + eog_dimensions, width, 1)
        self.blocks = nn.Sequential(*[_ResidualBlock(width, dilation) for dilation in (1, 2, 4, 8, 16, 32)])
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, 2)
        self.out = nn.Conv1d(width, eeg_channels, 1)

    def forward(self, y: Tensor, q0: Tensor, projected0: Tensor) -> Tensor:
        hidden = self.blocks(self.inp(torch.cat((y, projected0, q0), dim=1)))
        hidden = self.attention(hidden.transpose(1, 2)).transpose(1, 2)
        return self.out(hidden)


__all__ = ["PopulationAnchorV24"]

