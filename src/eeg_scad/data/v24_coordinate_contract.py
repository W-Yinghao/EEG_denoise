"""Explicit physical/standardized coordinate contract for V24.

The functions deliberately keep centering separate from scaling.  This makes
the two common V23 failure modes (double EEG scaling and omitted inverse EOG
scaling) directly testable instead of being hidden in a generic normalizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def robust_center_scale(value: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(value, dtype=np.float64)
    center = np.median(x, axis=-1)
    scale = 1.4826 * np.median(np.abs(x - center[:, None]), axis=-1)
    return center, np.maximum(scale, epsilon)


def canonical_operator(raw_operator: np.ndarray, eeg_scale: np.ndarray, eog_scale: np.ndarray) -> np.ndarray:
    """Return ``D_y^-1 C_raw D_e``."""
    raw = np.asarray(raw_operator, dtype=np.float64)
    return raw * np.asarray(eog_scale, dtype=np.float64)[None, :] / np.asarray(eeg_scale, dtype=np.float64)[:, None]


def eog_latent(eog_physical: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return ``D_e^-1 (E - mu_e)`` in support EOG coordinates."""
    eog = np.asarray(eog_physical, dtype=np.float64)
    return (eog - np.asarray(center, dtype=np.float64)[:, None]) / np.asarray(scale, dtype=np.float64)[:, None]


def artifact_raw(raw_operator: np.ndarray, centered_eog: np.ndarray, eeg_scale: np.ndarray) -> np.ndarray:
    return (np.asarray(raw_operator, dtype=np.float64) @ np.asarray(centered_eog, dtype=np.float64)) / np.asarray(eeg_scale, dtype=np.float64)[:, None]


def artifact_canonical(operator: np.ndarray, latent: np.ndarray) -> np.ndarray:
    return np.asarray(operator, dtype=np.float64) @ np.asarray(latent, dtype=np.float64)


def artifact_v23_committed(operator: np.ndarray, centered_eog: np.ndarray, eeg_scale: np.ndarray) -> np.ndarray:
    """Exact V23 online-generator expression, retained only for the audit."""
    return (np.asarray(operator, dtype=np.float64) @ np.asarray(centered_eog, dtype=np.float64)) / np.asarray(eeg_scale, dtype=np.float64)[:, None]


def comparison_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    value = np.asarray(candidate, dtype=np.float64)
    difference = value - ref
    denom = max(float(np.linalg.norm(ref)), np.finfo(np.float64).eps)
    ref_rms = np.sqrt(np.mean(ref * ref, axis=1))
    value_rms = np.sqrt(np.mean(value * value, axis=1))
    correlation = []
    for a, b in zip(ref, value):
        if np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
            correlation.append(float("nan"))
        else:
            correlation.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "relative_frobenius_difference": float(np.linalg.norm(difference) / denom),
        "artifact_rms_ratio": float(np.sqrt(np.mean(value * value)) / max(np.sqrt(np.mean(ref * ref)), 1e-15)),
        "channelwise_scale_ratio_min": float(np.min(value_rms / np.maximum(ref_rms, 1e-15))),
        "channelwise_scale_ratio_median": float(np.median(value_rms / np.maximum(ref_rms, 1e-15))),
        "channelwise_scale_ratio_max": float(np.max(value_rms / np.maximum(ref_rms, 1e-15))),
        "rowwise_correlation_median": float(np.nanmedian(correlation)),
    }


@dataclass(frozen=True)
class CoordinateCell:
    raw_operator: np.ndarray
    eeg_scale: np.ndarray
    eog_center: np.ndarray
    eog_scale: np.ndarray

    @property
    def canonical(self) -> np.ndarray:
        return canonical_operator(self.raw_operator, self.eeg_scale, self.eog_scale)

    def three_routes(self, eog_physical: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centered = np.asarray(eog_physical, dtype=np.float64) - self.eog_center[:, None]
        raw = artifact_raw(self.raw_operator, centered, self.eeg_scale)
        canonical = artifact_canonical(self.canonical, centered / self.eog_scale[:, None])
        committed = artifact_v23_committed(self.canonical, centered, self.eeg_scale)
        return raw, canonical, committed


__all__ = [
    "CoordinateCell", "artifact_canonical", "artifact_raw",
    "artifact_v23_committed", "canonical_operator", "comparison_metrics",
    "eog_latent", "robust_center_scale",
]

