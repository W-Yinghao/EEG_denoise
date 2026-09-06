"""U1-c OSCAR Stage-A: weight-space ceiling probe on frozen V42R checkpoints.

Rank-4 zero-init LoRA on the ResidualBlock body convolutions (conv1/conv2, the
v8 scaffold pattern), fine-tuned per held-out subject on ORACLE-operator
synthesized pairs (generative-truth supervision — non-deployable by
construction, labeled as such).  Evaluated against the frozen unadapted POP
route on the same episodes and noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ScoreLoRASummary:
    rank: int
    adapted_convolutions: int
    trainable_parameters: int


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, rank: int = 4) -> None:
        super().__init__()
        if base.groups != 1 or rank < 1:
            raise ValueError("LoRAConv1d requires groups=1 and rank >= 1")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(rank, base.out_channels, kernel_size=base.kernel_size,
                            stride=base.stride, padding=base.padding,
                            dilation=base.dilation, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / np.sqrt(base.in_channels))
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.base(value) + self.up(self.down(value))


def inject_score_lora(module: nn.Module, rank: int = 4) -> ScoreLoRASummary:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    adapted = 0
    for child in module.modules():
        for attribute in ("conv1", "conv2"):
            candidate = getattr(child, attribute, None)
            if isinstance(candidate, nn.Conv1d) and not isinstance(candidate, LoRAConv1d):
                setattr(child, attribute, LoRAConv1d(candidate, rank))
                adapted += 1
    if adapted == 0:
        raise RuntimeError("no convolutions adapted")
    trainable = sum(parameter.numel() for parameter in module.parameters()
                    if parameter.requires_grad)
    return ScoreLoRASummary(rank, adapted, trainable)


def lora_parameters(module: nn.Module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


__all__ = ["LoRAConv1d", "ScoreLoRASummary", "inject_score_lora", "lora_parameters"]
