"""Typed configuration for the 1D U-Net (handoff §4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """1D U-Net hyperparameters.

    Subject-conditioning fields (``num_subjects``, ``subject_embed_dim``) are declared here but
    only consumed from M3 onward; at M2 the network is a plain single-decoder DDPM backbone.
    """

    in_channels: int = 22
    signal_length: int = 512
    base_channels: int = 64
    channel_mults: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2
    groupnorm_groups: int = 8
    dropout: float = 0.1
    time_sinusoidal_dim: int = 128
    time_embed_dim: int = 512
    attention_length: int = 64
    attention_heads: int = 4
    num_subjects: int = 9
    subject_embed_dim: int = 128

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load a :class:`ModelConfig` from ``configs/model.yaml``."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(**raw)
