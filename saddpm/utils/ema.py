"""Exponential moving average of model weights (Polyak averaging).

Diffusion U-Nets are routinely evaluated with an EMA of the training weights — it is the standard
substitute for the running-statistics averaging that BatchNorm gives the CNN baselines (our U-Net
uses GroupNorm, which has none). Track during training, then :meth:`copy_to` a model for sampling.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class EMA:
    """Maintain a shadow copy of a model's floating-point tensors, updated as a moving average."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model's current floating-point tensors into the shadow copy."""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite ``model``'s floating-point tensors with the EMA shadow (in place)."""
        msd = model.state_dict()
        for k, shadow in self.shadow.items():
            msd[k].copy_(shadow)
