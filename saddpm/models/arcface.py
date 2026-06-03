"""ArcFace head for subject identification (handoff §1.3, [DD-9]).

Additive angular margin softmax (Deng et al., 2019). Operates on the individual-branch pooled
feature ``z^s`` to classify the subject. Margin ``m=0.5``, scale ``κ=30``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ArcFace(nn.Module):
    """Additive angular margin classifier over ``num_classes`` subjects.

    The margin is applied via the numerically stable identity
    ``cos(θ+m) = cosθ·cos m − sinθ·sin m`` (with ``sinθ = √(1−cos²θ)``) rather than ``acos``,
    whose gradient diverges at ``cosθ = ±1`` and causes NaNs (especially under AMP).
    """

    def __init__(self, in_features: int, num_classes: int, margin: float = 0.5, scale: float = 30.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # threshold below which cos(θ+m) would fold back; use the easy-margin fallback there.
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def cosine(self, feat: Tensor) -> Tensor:
        """Cosine similarity between normalised features and class weights, ``(B, num_classes)``.

        Computed in the weight dtype (float32) so it is robust under AMP autocast, where ``feat``
        may arrive as float16.
        """
        feat = feat.to(self.weight.dtype)
        return F.linear(F.normalize(feat, dim=-1), F.normalize(self.weight, dim=-1))

    def logits(self, feat: Tensor) -> Tensor:
        """Scaled cosine logits (no margin) for evaluation / prediction."""
        return self.scale * self.cosine(feat)

    def forward(self, feat: Tensor, labels: Tensor) -> Tensor:
        """Margined logits for training.

        Args:
            feat: ``(B, in_features)`` pooled feature ``z^s``.
            labels: ``(B,)`` ground-truth subject ids.

        Returns:
            ``(B, num_classes)`` scaled logits with the additive angular margin applied to the
            target class.
        """
        cos = self.cosine(feat).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt((1.0 - cos * cos).clamp(min=1e-9))
        phi = cos * self.cos_m - sin * self.sin_m  # cos(theta + m)
        # easy-margin fallback: only apply the margin where it keeps the function monotone.
        phi = torch.where(cos > self.threshold, phi, cos - self.mm)
        onehot = F.one_hot(labels, self.num_classes).to(cos.dtype)
        return self.scale * (onehot * phi + (1.0 - onehot) * cos)
