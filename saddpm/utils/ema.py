"""Small exponential-moving-average helper required by the committed trainers."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must lie in (0,1)")
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key in self.shadow:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.shadow.items():
            state[key].copy_(value)
