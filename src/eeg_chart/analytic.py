"""Canonical-space analytic cleaner (U1-a) and the GAUGE-NULL control.

The cleaner is closed-form: exact rank-r least squares on the fixed canonical
artifact system U° with the population covariance Σ̄ as prior (GLS/Wiener
weighting).  Only the transport varies between arms; U° is arm-independent.
The no-correction path returns y bit-exactly (Lemma 4: T(ρ)^+ T(ρ) = I)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from eeg_chart.geodesic import TransportArm, rotation_log
from eeg_chart.transport import orth


def canonical_clean(arm: TransportArm, u_canon: np.ndarray, sigma_bar_inv: np.ndarray,
                    y: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    """x_hat = y - T(ρ)^+ U° â with â the Σ̄-weighted least-squares coefficient."""
    z = arm.transport @ np.asarray(y, np.float64)
    weighted = u_canon.T @ sigma_bar_inv
    gram = weighted @ u_canon + delta * np.eye(u_canon.shape[1])
    coefficients = np.linalg.solve(gram, weighted @ z)
    return np.asarray(y, np.float64) - arm.pinv @ (u_canon @ coefficients)


def gauge_null_rotation(rotation: np.ndarray, base: np.ndarray, seed: int) -> np.ndarray:
    """Random rotation supported on W with the principal-angle spectrum of
    rotation @ base.T (Prop-5' §5.5): tests that gains need CORRECT alignment."""
    support, _, angles = rotation_log(rotation, base)
    m = support.shape[1]
    if m == 0 or len(angles) == 0:
        return base.copy()
    rng = np.random.default_rng(seed)
    random_basis = orth(rng.standard_normal((m, m)))
    skew = np.zeros((m, m))
    for index, angle in enumerate(angles):
        if 2 * index + 1 >= m:
            break
        u = random_basis[:, 2 * index]
        v = random_basis[:, 2 * index + 1]
        skew += angle * (np.outer(u, v) - np.outer(v, u))
    partial = expm(skew)
    full = np.eye(len(base)) + support @ (partial - np.eye(m)) @ support.T
    return full @ base


__all__ = ["canonical_clean", "gauge_null_rotation"]
