"""A compact covariance-whitened linear concept eraser (LEACE)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _symmetric_power(matrix: np.ndarray, power: float, ridge: float) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    values = np.maximum(values, ridge)
    return (vectors * np.power(values, power)) @ vectors.T


@dataclass
class LEACE:
    mean: np.ndarray
    eraser: np.ndarray
    rank: int

    @classmethod
    def fit(cls, z: np.ndarray, subject: np.ndarray, *, ridge: float = 1e-5) -> "LEACE":
        z = np.asarray(z, dtype=np.float64)
        labels = np.asarray(subject)
        mean = z.mean(axis=0, keepdims=True)
        centered = z - mean
        unique = np.unique(labels)
        onehot = np.stack([labels == value for value in unique], axis=1).astype(np.float64)
        onehot -= onehot.mean(axis=0, keepdims=True)
        cxx = centered.T @ centered / max(len(centered) - 1, 1)
        cxs = centered.T @ onehot / max(len(centered) - 1, 1)
        inv_sqrt = _symmetric_power(cxx, -0.5, ridge)
        sqrt = _symmetric_power(cxx, 0.5, ridge)
        whitened_cross = inv_sqrt @ cxs
        u, s, _ = np.linalg.svd(whitened_cross, full_matrices=False)
        tol = max(whitened_cross.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
        rank = int(np.sum(s > tol))
        projector = u[:, :rank] @ u[:, :rank].T if rank else np.zeros_like(cxx)
        eraser = inv_sqrt @ (np.eye(cxx.shape[0]) - projector) @ sqrt
        return cls(mean=mean.squeeze(0), eraser=eraser, rank=rank)

    def transform(self, z: np.ndarray) -> np.ndarray:
        centered = np.asarray(z, dtype=np.float64) - self.mean
        return (centered @ self.eraser + self.mean).astype(np.float32)

    def private(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        return z - self.transform(z)


__all__ = ["LEACE"]
