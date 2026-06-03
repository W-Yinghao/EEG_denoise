"""FiLM conditioning (handoff §1.2, §4).

A per-site MLP maps the subject embedding ``e`` to per-channel ``(γ, δ)`` and modulates a feature
map ``h`` as ``h' = γ(e) ⊙ h + δ(e)``. The projection is initialised so that ``γ=1, δ=0`` at the
start of training (identity), which keeps the conditioned network stable initially.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FiLM(nn.Module):
    """Feature-wise linear modulation from a subject embedding."""

    def __init__(self, emb_dim: int, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(emb_dim, 2 * channels)
        # Identity init: zero weights, gamma bias = 1, delta bias = 0  ->  h' = h.
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias[:channels].fill_(1.0)
            self.proj.bias[channels:].fill_(0.0)

    def forward(self, h: Tensor, emb: Tensor) -> Tensor:
        """Modulate ``h`` (B, C, L) with per-channel scale/shift from ``emb`` (B, emb_dim)."""
        gamma, delta = self.proj(emb).chunk(2, dim=-1)  # each (B, C)
        return gamma[:, :, None] * h + delta[:, :, None]
