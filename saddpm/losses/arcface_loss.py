"""ArcFace subject-identification loss (handoff §1.3)."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F

from ..models.arcface import ArcFace


def arcface_loss(head: ArcFace, z_subject: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy over the margined ArcFace logits.

    Args:
        head: the :class:`ArcFace` module.
        z_subject: ``(B, d)`` individual-branch pooled features ``Z^s``.
        labels: ``(B,)`` ground-truth subject ids.

    Returns:
        Scalar ArcFace loss ``L_a``.
    """
    return F.cross_entropy(head(z_subject, labels), labels)
