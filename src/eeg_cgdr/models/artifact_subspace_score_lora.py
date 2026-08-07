"""True internal rank-r score LoRA for the artifact-subspace U-Net."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class LoRAConv1d(nn.Module):
    """Frozen Conv1d plus a zero-initialized low-rank weight residual."""

    def __init__(self, base: nn.Conv1d, rank: int = 4) -> None:
        super().__init__()
        if rank < 1 or base.groups != 1:
            raise ValueError("score LoRA requires a positive rank and ungrouped convolution")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.down = nn.Conv1d(base.in_channels, rank, kernel_size=1, bias=False)
        self.up = nn.Conv1d(rank, base.out_channels, kernel_size=base.kernel_size,
                            stride=base.stride, padding=base.padding, dilation=base.dilation, bias=False)
        self.down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.up.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.normal_(self.down.weight, std=1.0 / max(base.in_channels, 1) ** .5)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        return self.base(value) + self.up(self.down(value))


@dataclass(frozen=True)
class ScoreLoRASummary:
    rank: int
    adapted_convolutions: int
    trainable_parameters: int


def inject_score_lora(module: nn.Module, *, rank: int = 4) -> ScoreLoRASummary:
    """Freeze a population score model and adapt every ResBlock score conv."""

    for parameter in module.parameters():
        parameter.requires_grad_(False)
    targets: list[tuple[nn.Module, str, nn.Conv1d]] = []
    for child in module.modules():
        for name in ("conv1", "conv2"):
            value = getattr(child, name, None)
            if isinstance(value, nn.Conv1d):
                targets.append((child, name, value))
    if not targets:
        raise ValueError("no internal score ResBlock convolutions found")
    for parent, name, base in targets:
        setattr(parent, name, LoRAConv1d(base, rank=rank))
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return ScoreLoRASummary(rank=rank, adapted_convolutions=len(targets), trainable_parameters=trainable)


def lora_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items() if ".down." in name or ".up." in name}


__all__ = ["LoRAConv1d", "ScoreLoRASummary", "inject_score_lora", "lora_state_dict"]
