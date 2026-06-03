"""Reconstruction / noise-prediction loss (handoff §1.3, §5).

At M2 this is the plain DDPM objective ``L_simple = E‖ε − ε_θ(x_t, t)‖²``. At M4 the same
function scores ``L_r = E‖ε − (ε_θ + ε_φ)‖²`` for the dual decoder.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def reconstruction_loss(eps_pred: Tensor, eps_target: Tensor) -> Tensor:
    """Mean-squared error between predicted and true noise.

    Args:
        eps_pred: predicted noise ``(B, C, L)`` (``ε_θ`` or ``ε_θ + ε_φ``).
        eps_target: ground-truth noise ``(B, C, L)``.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(eps_pred, eps_target)
