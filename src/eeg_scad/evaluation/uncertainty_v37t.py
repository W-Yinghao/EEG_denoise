"""Participant-first ensemble uncertainty diagnostics for frozen V37T outputs."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import norm, spearmanr


def interval_metrics(samples: np.ndarray, target: np.ndarray, level: float) -> dict[str, float]:
    """Empirical central interval metrics; samples have leading ensemble axis."""
    alpha = 1.0 - level
    low, high = np.quantile(samples, (alpha / 2.0, 1.0 - alpha / 2.0), axis=0)
    below = np.maximum(low - target, 0.0)
    above = np.maximum(target - high, 0.0)
    score = high - low + (2.0 / alpha) * (below + above)
    return {
        "coverage": float(np.mean((target >= low) & (target <= high))),
        "interval_width": float(np.mean(high - low)),
        "interval_score": float(np.mean(score)),
    }


def constant_width_interval(mean: np.ndarray, samples: np.ndarray, target: np.ndarray, level: float) -> dict[str, float]:
    """Matched-average-dispersion Gaussian reference; it never sees the target."""
    sigma = float(np.mean(np.std(samples, axis=0, ddof=1)))
    half = float(norm.ppf((1.0 + level) / 2.0)) * sigma
    low, high = mean - half, mean + half
    alpha = 1.0 - level
    score = high - low + (2.0 / alpha) * (np.maximum(low-target, 0.0) + np.maximum(target-high, 0.0))
    return {
        "coverage": float(np.mean((target >= low) & (target <= high))),
        "interval_width": float(2.0 * half),
        "interval_score": float(np.mean(score)),
    }


def ensemble_crps(samples: np.ndarray, target: np.ndarray) -> float:
    """Exact empirical CRPS with O(K log K), avoiding a K by K tensor."""
    ordered = np.sort(samples, axis=0)
    n = len(ordered)
    weights = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * target.ndim)
    pair_term = np.sum(weights * ordered, axis=0) / float(n * n)
    return float(np.mean(np.mean(np.abs(samples-target[None]), axis=0) - pair_term))


def error_dispersion(samples: np.ndarray, target: np.ndarray) -> float:
    """Spearman association across windows between absolute error and ensemble spread."""
    if samples.shape[1] < 3:
        return float("nan")
    prediction = np.mean(samples, axis=0)
    error = np.mean(np.abs(prediction-target), axis=tuple(range(1, target.ndim)))
    spread = np.mean(np.std(samples, axis=0, ddof=1), axis=tuple(range(1, target.ndim)))
    value = spearmanr(error, spread).statistic
    return float(value) if np.isfinite(value) else 0.0


def projected_variance(samples: np.ndarray, projectors: np.ndarray) -> tuple[float, float]:
    centered = samples - np.mean(samples, axis=0, keepdims=True)
    parallel = np.einsum("nij,knjt->knit", projectors, centered, optimize=True)
    complement = centered - parallel
    return float(np.mean(parallel**2)), float(np.mean(complement**2))


def participant_mean(rows: Iterable[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    output = []
    for group, values in grouped.items():
        result = dict(zip(keys, group))
        numeric = [key for key, value in values[0].items() if key not in keys and isinstance(value, (int, float, np.number))]
        result.update({key: float(np.nanmean([float(value[key]) for value in values])) for key in numeric})
        output.append(result)
    return output


__all__ = ["constant_width_interval", "ensemble_crps", "error_dispersion", "interval_metrics", "participant_mean", "projected_variance"]

