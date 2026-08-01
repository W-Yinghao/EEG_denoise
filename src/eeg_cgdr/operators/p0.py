"""P0 instantaneous prediction-space ridge EOG-to-EEG transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CalibrationBatch:
    eeg: np.ndarray
    eog: np.ndarray
    participant: str
    source_record: str
    sampling_rate: float


@dataclass(frozen=True)
class P0Config:
    target_rank: int = 2
    ridge_lambda: float = 0.01
    maximum_reference_condition: float = 1.0e4
    minimum_singular_ratio: float = 1.0e-4
    minimum_movement_coverage: float = 0.01
    bootstrap_replicates: int = 200
    bootstrap_block_samples: int = 400
    minimum_bootstrap_success: float = 0.90
    maximum_bootstrap_median_distance: float = 0.25
    maximum_bootstrap_q90_distance: float = 0.50
    seed: int = 20260801


@dataclass(frozen=True)
class P0Transfer:
    transfer_matrix: np.ndarray
    eeg_subspace_basis: np.ndarray
    projector: np.ndarray
    predicted_contamination: np.ndarray
    eog_mean: np.ndarray
    eeg_mean: np.ndarray
    rank: int
    diagnostics: dict[str, float | int | str | list[float]]


@dataclass(frozen=True)
class P0FitOutcome:
    status: Literal["eligible", "ineligible"]
    transfer: P0Transfer | None
    reasons: tuple[str, ...]
    fallback: Literal["POP"] = "POP"


def _validate(batch: CalibrationBatch) -> tuple[np.ndarray, np.ndarray]:
    eeg = np.asarray(batch.eeg, dtype=np.float64)
    eog = np.asarray(batch.eog, dtype=np.float64)
    if eeg.ndim != 2 or eog.ndim != 2 or eeg.shape[1] != eog.shape[1]:
        raise ValueError(f"unaligned calibration arrays: EEG={eeg.shape}, EOG={eog.shape}")
    if eeg.shape[0] < 2 or eog.shape[0] < 1 or eeg.shape[1] < 4:
        raise ValueError("calibration arrays are too small")
    if not np.isfinite(eeg).all() or not np.isfinite(eog).all():
        raise ValueError("calibration contains non-finite values")
    return eeg, eog


def _fit_core(
    eeg: np.ndarray, eog: np.ndarray, ridge_lambda: float, target_rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the registered FP64 instantaneous ridge transfer.

    With channel-major centered arrays ``Y`` and ``E``, the estimator is

    ``C = Y E.T (E E.T + lambda I)^-1``.

    The right solve below is the numerically stable implementation of that
    expression.  The identifiable EEG-space basis is obtained from ``SVD(C)``;
    it is not obtained from an SVD of the calibration prediction ``C E``.
    """
    ridge = float(ridge_lambda)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    if int(target_rank) != target_rank or target_rank < 1:
        raise ValueError("target_rank must be a positive integer")
    eeg_mean = eeg.mean(axis=1, keepdims=True)
    eog_mean = eog.mean(axis=1, keepdims=True)
    y = np.asarray(eeg - eeg_mean, dtype=np.float64)
    e = np.asarray(eog - eog_mean, dtype=np.float64)
    if target_rank > min(y.shape[0], e.shape[0]):
        raise ValueError("target_rank exceeds the transfer matrix dimensions")
    gram = e @ e.T
    regularized_gram = gram + ridge * np.eye(e.shape[0], dtype=np.float64)
    cross_covariance = y @ e.T
    transfer = np.linalg.solve(regularized_gram, cross_covariance.T).T
    predicted = transfer @ e
    basis_full, singular_values, _ = np.linalg.svd(transfer, full_matrices=False)
    basis = basis_full[:, :target_rank]
    projector = basis @ basis.T
    return transfer, predicted, basis, projector, singular_values, np.concatenate(
        [eeg_mean, eog_mean], axis=0
    )


def _projector_distance(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / np.sqrt(2.0 * rank))


def _bootstrap(
    eeg: np.ndarray,
    eog: np.ndarray,
    full_projector: np.ndarray,
    config: P0Config,
) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(config.seed)
    samples = eeg.shape[1]
    block = min(config.bootstrap_block_samples, samples)
    starts = np.arange(0, max(samples - block + 1, 1), block, dtype=int)
    if starts.size < 2:
        return 0.0, float("inf"), float("inf"), 0
    distances: list[float] = []
    successes = 0
    for _ in range(config.bootstrap_replicates):
        chosen: list[np.ndarray] = []
        while sum(index.size for index in chosen) < samples:
            start = int(rng.choice(starts))
            chosen.append(np.arange(start, min(start + block, samples), dtype=int))
        index = np.concatenate(chosen)[:samples]
        try:
            _, predicted, basis, projector, singular_values, _ = _fit_core(
                eeg[:, index], eog[:, index], config.ridge_lambda, config.target_rank
            )
            if (
                basis.shape[1] != config.target_rank
                or singular_values.size < config.target_rank
                or singular_values[config.target_rank - 1]
                <= singular_values[0] * config.minimum_singular_ratio
                or not np.isfinite(predicted).all()
            ):
                continue
            successes += 1
            distances.append(_projector_distance(projector, full_projector, config.target_rank))
        except np.linalg.LinAlgError:
            continue
    rate = successes / max(config.bootstrap_replicates, 1)
    if not distances:
        return rate, float("inf"), float("inf"), successes
    return rate, float(np.median(distances)), float(np.quantile(distances, 0.90)), successes


def fit_p0(
    batch: CalibrationBatch,
    config: P0Config,
    *,
    movement_threshold: float,
) -> P0FitOutcome:
    eeg, eog = _validate(batch)
    if not np.isfinite(config.ridge_lambda) or config.ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    if config.target_rank < 1 or config.target_rank > min(eeg.shape[0], eog.shape[0]):
        raise ValueError("target_rank is incompatible with calibration dimensions")
    reasons: list[str] = []
    centered = eog - eog.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    if np.any(scale <= 1e-12):
        return P0FitOutcome("ineligible", None, ("constant_reference",))
    standardized = centered / scale
    reference_rank = int(np.linalg.matrix_rank(standardized))
    reference_condition = float(np.linalg.cond(standardized @ standardized.T))
    movement = np.sqrt(np.mean(standardized**2, axis=0))
    coverage = float(np.mean(movement >= movement_threshold))
    if reference_rank < config.target_rank:
        reasons.append("reference_rank")
    if not np.isfinite(reference_condition) or reference_condition > config.maximum_reference_condition:
        reasons.append("reference_condition")
    if coverage < config.minimum_movement_coverage:
        reasons.append("movement_coverage")
    if reasons:
        return P0FitOutcome("ineligible", None, tuple(reasons))
    try:
        transfer, predicted, basis, projector, singular_values, means = _fit_core(
            eeg, eog, config.ridge_lambda, config.target_rank
        )
    except np.linalg.LinAlgError:
        return P0FitOutcome("ineligible", None, ("linear_algebra",))
    if singular_values.size < config.target_rank or singular_values[0] <= 0:
        return P0FitOutcome("ineligible", None, ("predicted_rank",))
    singular_ratio = float(singular_values[config.target_rank - 1] / singular_values[0])
    if singular_ratio < config.minimum_singular_ratio:
        return P0FitOutcome("ineligible", None, ("singular_value_gap",))
    symmetry_error = float(np.linalg.norm(projector - projector.T, ord="fro"))
    idempotence_error = float(np.linalg.norm(projector @ projector - projector, ord="fro"))
    success_rate, bootstrap_median, bootstrap_q90, bootstrap_successes = _bootstrap(
        eeg, eog, projector, config
    )
    if success_rate < config.minimum_bootstrap_success:
        reasons.append("bootstrap_success")
    if bootstrap_median > config.maximum_bootstrap_median_distance:
        reasons.append("bootstrap_median")
    if bootstrap_q90 > config.maximum_bootstrap_q90_distance:
        reasons.append("bootstrap_q90")
    diagnostics: dict[str, float | int | str | list[float]] = {
        "samples": int(eeg.shape[1]),
        "numeric_precision": "float64",
        "ridge_estimator": "Y E.T solve(E E.T + lambda I)",
        "ridge_lambda": float(config.ridge_lambda),
        "svd_object": "transfer_matrix_C",
        "reference_rank": reference_rank,
        "reference_condition": reference_condition,
        "movement_coverage": coverage,
        "singular_values": singular_values[: config.target_rank + 2].tolist(),
        "singular_ratio": singular_ratio,
        "projector_symmetry_error": symmetry_error,
        "projector_idempotence_error": idempotence_error,
        "bootstrap_success_rate": success_rate,
        "bootstrap_successes": bootstrap_successes,
        "bootstrap_median_projector_distance": bootstrap_median,
        "bootstrap_q90_projector_distance": bootstrap_q90,
    }
    if reasons:
        return P0FitOutcome("ineligible", None, tuple(reasons))
    eeg_mean = means[: eeg.shape[0]]
    eog_mean = means[eeg.shape[0] :]
    estimate = P0Transfer(
        transfer_matrix=transfer,
        eeg_subspace_basis=basis,
        projector=projector,
        predicted_contamination=predicted,
        eog_mean=eog_mean,
        eeg_mean=eeg_mean,
        rank=config.target_rank,
        diagnostics=diagnostics,
    )
    return P0FitOutcome("eligible", estimate, ())
