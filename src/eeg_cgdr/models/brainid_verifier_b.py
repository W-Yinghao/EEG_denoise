"""Independent evaluator-only BrainID verifier B.

This module is intentionally separate from verifier-A training and all M0
parameter selection code.  Its objects are only imported by evaluator stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import welch


def morphology_features(x: np.ndarray, sampling_rate: int = 250) -> np.ndarray:
    bins = x.reshape(len(x), x.shape[1], 8, 25).mean(-1).reshape(len(x), -1)
    f, p = welch(x, fs=sampling_rate, nperseg=min(128, x.shape[-1]), axis=-1)
    keep = (f >= 1) & (f <= 45)
    spectral = np.log(np.maximum(p[..., keep].mean(-1), 1e-12))
    cov = []
    for row in x:
        raw_cov = np.cov(row)
        # Unit-invariant trace-scaled regularization: a fixed 1e-3 in physical
        # voltage units would erase spatial structure when data are in volts.
        average_variance = max(float(np.trace(raw_cov) / row.shape[0]), 1e-12)
        c0 = raw_cov + np.eye(row.shape[0]) * average_variance * 1e-3
        _, logdet = np.linalg.slogdet(c0)
        c0 = c0 / np.exp(logdet / row.shape[0])
        values, vectors = np.linalg.eigh(c0)
        logc = (vectors * np.log(np.maximum(values, 1e-8))) @ vectors.T
        cov.append(logc[np.triu_indices(row.shape[0])])
    return np.concatenate([bins, spectral, np.asarray(cov)], axis=1).astype(np.float32)


@dataclass
class VerifierB:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_scale: np.ndarray
    metric_weight: np.ndarray

    def embed(self, x: np.ndarray) -> np.ndarray:
        features = morphology_features(x)
        z = (features - self.scaler_mean) / self.scaler_scale
        z = (z - self.pca_mean) @ self.pca_components.T
        z = z / self.pca_scale
        z = z * self.metric_weight
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


def load_verifier_b(path: Path) -> VerifierB:
    with np.load(path) as data:
        return VerifierB(*(np.asarray(data[key]) for key in (
            "scaler_mean", "scaler_scale", "pca_mean", "pca_components", "pca_scale", "metric_weight"
        )))
