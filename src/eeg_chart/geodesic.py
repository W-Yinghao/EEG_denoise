"""Corrected Prop-5' transport family (docs/prop5_endpoint_correction.md).

    Σ(ρ) = Σ̄^{1/2} (Σ̄^{-1/2} Σ̂ Σ̄^{-1/2})^ρ Σ̄^{1/2}
    G(ρ) = Σ̄^{1/2} Σ(ρ)^{-1/2}
    Q(ρ) = exp(ρ log(Q Q̄ᵀ)) Q̄            (geodesic FROM the population base Q̄)
    T(ρ) = Q(ρ) G(ρ) L

ρ = 0 short-circuits to Q̄ L with NO arithmetic, so the POP arm is bit-identical
by construction (the V43 λ=0 clamp contract analogue).  The principal-angle cap
(> π/2 → ρ := 0, ABSTAIN) folds into the abstention rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, logm

from eeg_chart.transport import orth, spd_power, truncated_inv_root


ANGLE_CAP = np.pi / 2


def rotation_log(rotation: np.ndarray, base: np.ndarray):
    """Principal log of M = rotation @ base.T via its invariant subspace.

    Returns (support basis V, small log block, principal angles)."""
    m = rotation @ base.T
    deviation = m - np.eye(len(m))
    support = orth(deviation)
    if support.shape[1] == 0:
        return support, np.zeros((0, 0)), np.zeros(0)
    block = support.T @ m @ support
    log_block = np.real(logm(block))
    log_block = (log_block - log_block.T) / 2
    skew_eigen = np.linalg.eigvals(log_block)
    angles = np.sort(np.abs(np.imag(skew_eigen)))[::-1]
    principal = angles[::2] if len(angles) >= 2 else angles
    return support, log_block, principal


def max_principal_angle(rotation: np.ndarray, base: np.ndarray) -> float:
    _, _, angles = rotation_log(rotation, base)
    return float(angles[0]) if len(angles) else 0.0


def rotation_geodesic(rotation: np.ndarray, base: np.ndarray, rho: float) -> np.ndarray:
    """Q(ρ) = exp(ρ log(Q Q̄ᵀ)) Q̄; ρ=0 returns base exactly, ρ=1 returns rotation."""
    if rho == 0.0:
        return base
    if rho == 1.0:
        return rotation
    support, log_block, _ = rotation_log(rotation, base)
    if support.shape[1] == 0:
        return base
    partial = expm(rho * log_block)
    full = np.eye(len(base)) + support @ (partial - np.eye(len(partial))) @ support.T
    return full @ base


@dataclass(frozen=True)
class TransportArm:
    """Arm-consistent (T, T^+, components) per Prop-5' §5.1."""
    rho: float
    transport: np.ndarray       # T(ρ) = Q(ρ) G(ρ) L,  K x C
    pinv: np.ndarray            # L^+ G(ρ)^{-1} Q(ρ)^T, C x K
    rotation: np.ndarray        # Q(ρ)
    align: np.ndarray           # G(ρ)
    abstained: bool


def transport_family(lift: np.ndarray, lift_pinv: np.ndarray, sigma_bar: np.ndarray,
                     sigma_hat: np.ndarray | None, rotation: np.ndarray,
                     base: np.ndarray, rho: float, angle_cap: float = ANGLE_CAP,
                     whitening: str = "full") -> TransportArm:
    """Build the arm at ρ; ρ=0 (or an angle-cap abstention) is the bit-identical POP arm.

    whitening="truncated" applies the frozen M13-W1 rank-truncated rule to the
    covariance alignment; the ρ=0 path is untouched (bit-identity preserved)."""
    abstained = False
    if rho > 0.0 and max_principal_angle(rotation, base) > angle_cap:
        rho, abstained = 0.0, True
    if rho == 0.0:
        eye = np.eye(len(base))
        return TransportArm(0.0, base @ lift, lift_pinv @ base.T, base, eye, abstained)
    if sigma_hat is None:
        raise ValueError("sigma_hat required for rho > 0")
    bar_root = spd_power(sigma_bar, 0.5)
    bar_inv_root = spd_power(sigma_bar, -0.5)
    whitened = bar_inv_root @ sigma_hat @ bar_inv_root
    sigma_rho = bar_root @ spd_power(whitened, rho) @ bar_root
    if whitening == "truncated":
        white, white_inv = truncated_inv_root(sigma_rho)
        align = bar_root @ white
        align_inv = white_inv @ bar_inv_root
    elif whitening == "full":
        align = bar_root @ spd_power(sigma_rho, -0.5)
        align_inv = np.linalg.inv(align)
    else:
        raise ValueError(whitening)
    q_rho = rotation_geodesic(rotation, base, rho)
    transport = q_rho @ align @ lift
    pinv = lift_pinv @ align_inv @ q_rho.T
    return TransportArm(rho, transport, pinv, q_rho, align, abstained)


def rotation_distance(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Geodesic distance d(Qa, Qb) = ||log(Qa Qb^T)||_F / sqrt(2)."""
    _, log_block, _ = rotation_log(rotation_a, rotation_b)
    return float(np.linalg.norm(log_block) / np.sqrt(2.0))


def rho_eb(tau2: float, v_s: float, hard_abstain: bool = False) -> float:
    """Closed-form EB shrinkage for the rotation factor (V43 λ-rule analogue)."""
    if hard_abstain:
        return 0.0
    return float(np.clip(tau2 / max(tau2 + v_s / 4.0, 1e-12), 0.0, 1.0))


__all__ = ["ANGLE_CAP", "TransportArm", "max_principal_angle", "rho_eb",
           "rotation_distance", "rotation_geodesic", "rotation_log", "transport_family"]
