"""Orthogonality loss disentangling content vs subject features (handoff §1.3).

``L_o = ‖ cross-cov(Z^c, Z^s) ‖_F²`` with column-mean-centered features. Using the empirical
cross-covariance ``(1/B) Z^cᵀ Z^s`` (rather than the raw Gram matrix) makes the loss scale
batch-size-invariant; it is zero iff the empirical cross-covariance is zero.
"""

from __future__ import annotations

import torch
from torch import Tensor


def orthogonality_loss(z_content: Tensor, z_subject: Tensor) -> Tensor:
    """Squared Frobenius norm of the empirical cross-covariance of the two feature sets.

    Args:
        z_content: ``(B, d)`` content-decoder pooled features ``Z^c``.
        z_subject: ``(B, d)`` individual-decoder pooled features ``Z^s``.

    Returns:
        Scalar orthogonality loss.
    """
    zc = z_content - z_content.mean(dim=0, keepdim=True)
    zs = z_subject - z_subject.mean(dim=0, keepdim=True)
    cross_cov = (zc.t() @ zs) / zc.shape[0]  # (d, d)
    return (cross_cov ** 2).sum()
