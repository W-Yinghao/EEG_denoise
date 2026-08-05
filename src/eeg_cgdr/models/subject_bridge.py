"""Coordinate-corrected subject bridge primitives.

The bridge never changes the canonical artifact target when an operator is
intervened upon.  Context-specific latents are decoded through their own
column-normalized transfer into the common EEG coordinate before subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .clean_prior import canonical_valid_time_mask


@dataclass(frozen=True)
class ReliabilityDiagnostics:
    reliability: float
    heldout_error: float
    operator_stability: float
    samples: int


def physical_eeg_delta(
    standardized_latent: Tensor,
    *,
    normalized_transfer: Tensor,
    latent_mean: Tensor,
    latent_standard_deviation: Tensor,
    valid_time_mask: Tensor,
) -> Tensor:
    """Decode a context latent into the common physical EEG coordinate."""

    if standardized_latent.ndim != 3:
        raise ValueError("standardized latent must have shape (B,E,T)")
    batch, latent_channels, _ = standardized_latent.shape
    transfer = torch.as_tensor(
        normalized_transfer,
        device=standardized_latent.device,
        dtype=standardized_latent.dtype,
    )
    if transfer.ndim == 2:
        transfer = transfer[None].expand(batch, -1, -1)
    if transfer.ndim != 3 or transfer.shape[0] != batch or transfer.shape[2] != latent_channels:
        raise ValueError("normalized transfer and latent coordinates differ")
    mean = torch.as_tensor(
        latent_mean,
        device=standardized_latent.device,
        dtype=standardized_latent.dtype,
    )
    scale = torch.as_tensor(
        latent_standard_deviation,
        device=standardized_latent.device,
        dtype=standardized_latent.dtype,
    )
    if mean.shape == (latent_channels,):
        mean = mean[None].expand(batch, -1)
    if scale.shape == (latent_channels,):
        scale = scale[None].expand(batch, -1)
    if mean.shape != (batch, latent_channels) or scale.shape != mean.shape:
        raise ValueError("latent normalization differs from context coordinates")
    if bool((scale <= 0).any()):
        raise ValueError("latent standard deviation must be positive")
    physical = standardized_latent * scale[:, :, None] + mean[:, :, None]
    delta = torch.einsum("bce,bet->bct", transfer, physical)
    mask = canonical_valid_time_mask(delta, valid_time_mask).to(delta.dtype)
    return delta * mask


def coordinate_corrected_bridge(
    population_restored: Tensor,
    *,
    context_delta: Tensor,
    population_delta: Tensor,
    beta: float | Tensor,
    rho: float | Tensor,
    activity_gate: Tensor,
    valid_time_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return ``P(y) + beta*rho*G(y)*(C_o a_o-C_0 a_0)``."""

    if context_delta.shape != population_delta.shape or context_delta.shape != population_restored.shape:
        raise ValueError("bridge EEG tensors must have identical shapes")
    batch = population_restored.shape[0]
    mask = canonical_valid_time_mask(population_restored, valid_time_mask).to(population_restored.dtype)
    gate = torch.as_tensor(activity_gate, device=population_restored.device, dtype=population_restored.dtype)
    if gate.shape == (batch, 1, population_restored.shape[-1]):
        gate = gate.expand(-1, population_restored.shape[1], -1)
    if gate.shape != population_restored.shape or bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("activity gate must be Bx1xT/BxCxT in [0,1]")
    reliability = torch.as_tensor(rho, device=population_restored.device, dtype=population_restored.dtype)
    if reliability.ndim == 0:
        reliability = reliability.expand(batch)
    if reliability.shape != (batch,) or bool(((reliability < 0) | (reliability > 1)).any()):
        raise ValueError("rho must be B values in [0,1]")
    coefficient = torch.as_tensor(beta, device=population_restored.device, dtype=population_restored.dtype)
    if coefficient.ndim == 0:
        coefficient = coefficient.expand(batch)
    if coefficient.shape != (batch,) or not bool(torch.isfinite(coefficient).all()):
        raise ValueError("beta must be finite scalar/B values")
    bridge = (
        coefficient[:, None, None]
        * reliability[:, None, None]
        * gate
        * (context_delta - population_delta)
        * mask
    )
    output = (population_restored + bridge) * mask
    return output, bridge


def _fit_transfer(y: np.ndarray, latent: np.ndarray, ridge: float) -> np.ndarray:
    gram = latent @ latent.T + float(ridge) * np.eye(latent.shape[0])
    return np.linalg.solve(gram, (y @ latent.T).T).T


def blocked_split_half_reliability(
    observed: np.ndarray,
    standardized_latent: np.ndarray,
    valid_time_mask: np.ndarray,
    *,
    latent_mean: np.ndarray,
    latent_standard_deviation: np.ndarray,
    ridge: float,
) -> ReliabilityDiagnostics:
    """Support-only A→B/B→A validation for one static transfer."""

    y = np.asarray(observed, dtype=np.float64)
    z = np.asarray(standardized_latent, dtype=np.float64)
    valid = np.asarray(valid_time_mask, dtype=bool)
    if y.ndim != 3 or z.ndim != 3 or valid.shape != (y.shape[0], y.shape[2]):
        raise ValueError("blocked reliability arrays have incompatible shapes")
    physical = (
        z * np.asarray(latent_standard_deviation, dtype=np.float64)[None, :, None]
        + np.asarray(latent_mean, dtype=np.float64)[None, :, None]
    )
    y_flat = np.concatenate([y[index, :, valid[index]] for index in range(y.shape[0])], axis=1)
    a_flat = np.concatenate([physical[index, :, valid[index]] for index in range(y.shape[0])], axis=1)
    samples = y_flat.shape[1]
    split = samples // 2
    if split < max(8, 2 * a_flat.shape[0]) or samples - split < max(8, 2 * a_flat.shape[0]):
        return ReliabilityDiagnostics(0.0, float("inf"), float("inf"), samples)
    ya, yb = y_flat[:, :split], y_flat[:, split:]
    aa, ab = a_flat[:, :split], a_flat[:, split:]
    ca = _fit_transfer(ya, aa, ridge)
    cb = _fit_transfer(yb, ab, ridge)
    eps = np.finfo(np.float64).eps
    error_ab = np.mean(np.square(yb - ca @ ab)) / max(np.mean(np.square(yb)), eps)
    error_ba = np.mean(np.square(ya - cb @ aa)) / max(np.mean(np.square(ya)), eps)
    error = float(0.5 * (error_ab + error_ba))
    stability = float(np.linalg.norm(ca - cb) / max(0.5 * (np.linalg.norm(ca) + np.linalg.norm(cb)), eps))
    reliability = float(np.clip(np.exp(-error - stability), 0.0, 1.0))
    return ReliabilityDiagnostics(reliability, error, stability, samples)


def fit_signed_beta(
    bridge_direction: np.ndarray,
    clean_minus_population: np.ndarray,
    valid_time_mask: np.ndarray,
    *,
    lower: float = -2.0,
    upper: float = 2.0,
) -> tuple[float, float, float]:
    """Fit one training-only signed beta; zero is an explicit candidate."""

    direction = np.asarray(bridge_direction, dtype=np.float64)
    target = np.asarray(clean_minus_population, dtype=np.float64)
    valid = np.asarray(valid_time_mask, dtype=bool)
    if direction.shape != target.shape or valid.shape != (direction.shape[0], direction.shape[2]):
        raise ValueError("beta fit arrays have incompatible shapes")
    mask = np.broadcast_to(valid[:, None, :], direction.shape)
    q = direction[mask]
    r = target[mask]
    denominator = float(q @ q)
    raw = 0.0 if denominator <= np.finfo(float).eps else float((q @ r) / denominator)
    candidate = float(np.clip(raw, lower, upper))
    loss_zero = float(np.mean(np.square(r)))
    loss_candidate = float(np.mean(np.square(r - candidate * q)))
    beta = candidate if loss_candidate < loss_zero else 0.0
    return beta, loss_zero, min(loss_zero, loss_candidate)


__all__ = [
    "ReliabilityDiagnostics",
    "blocked_split_half_reliability",
    "coordinate_corrected_bridge",
    "fit_signed_beta",
    "physical_eeg_delta",
]
