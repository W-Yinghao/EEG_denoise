"""Rotation-invariant support projectors."""
from __future__ import annotations

import torch
from torch import Tensor


def orthonormal_basis(basis: Tensor, tolerance: float = 1e-7) -> Tensor:
    """Return a thin SVD basis, dropping numerically null columns."""
    u, singular, _ = torch.linalg.svd(basis, full_matrices=False)
    if basis.ndim == 2:
        rank = max(1, int((singular > tolerance * singular.max().clamp_min(tolerance)).sum()))
        return u[:, :rank]
    ranks = (singular > tolerance * singular.amax(-1, keepdim=True).clamp_min(tolerance)).sum(-1)
    rank = max(1, int(ranks.min()))
    return u[..., :rank]


def projector(basis: Tensor) -> Tensor:
    orth = orthonormal_basis(basis)
    return orth @ orth.transpose(-1, -2)


def population_projector(projectors: Tensor, rank: int) -> Tensor:
    """Top-r eigenspace of the training-participant mean projector."""
    mean = projectors.mean(0)
    _, vectors = torch.linalg.eigh(mean)
    orth = vectors[:, -rank:]
    return orth @ orth.T


def diagnostics(value: Tensor) -> dict[str, float]:
    identity_error = torch.linalg.matrix_norm(value @ value - value)
    symmetry_error = torch.linalg.matrix_norm(value - value.T)
    return {
        "rank": float(torch.linalg.matrix_rank(value)),
        "symmetry_error": float(symmetry_error),
        "idempotence_error": float(identity_error),
    }


__all__ = ["orthonormal_basis", "projector", "population_projector", "diagnostics"]
