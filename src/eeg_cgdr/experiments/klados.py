"""Leakage-aware preparation and operator controls for the Klados v4 source fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import resample_poly

from eeg_cgdr.data.klados import KladosRecord
from eeg_cgdr.operators.p0 import (
    CalibrationBatch,
    P0Config,
    P0FitOutcome,
    P0Transfer,
    fit_p0,
)


@dataclass(frozen=True)
class WindowedQuery:
    contaminated: np.ndarray
    clean: np.ndarray
    eog: np.ndarray
    valid_samples: np.ndarray
    artifact_mask: np.ndarray
    sampling_rate: int


def _resample(array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(array, dtype=np.float64)
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        np.asarray(array, dtype=np.float64),
        target_rate // divisor,
        source_rate // divisor,
        axis=-1,
    )


def _pad_windows(array: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Use every query sample; zero padding is excluded from all metrics."""
    samples = int(array.shape[-1])
    count = int(np.ceil(samples / length))
    windows = np.zeros((count, *array.shape[:-1], length), dtype=array.dtype)
    valid = np.zeros(count, dtype=np.int64)
    for index in range(count):
        start = index * length
        stop = min(start + length, samples)
        valid[index] = stop - start
        windows[index, ..., : valid[index]] = array[..., start:stop]
    return windows, valid


def prepare_query(
    record: KladosRecord,
    *,
    source_rate: int,
    target_rate: int,
    query_start_seconds: float,
    query_end_seconds: float,
    window_samples: int,
    attenuation_scale: float,
) -> WindowedQuery:
    start_native = int(round(query_start_seconds * source_rate))
    stop_native = min(int(round(query_end_seconds * source_rate)), record.samples)
    if start_native >= stop_native:
        raise ValueError("empty held-out Klados query")
    contaminated = _resample(
        record.contaminated[:, start_native:stop_native], source_rate, target_rate
    )
    clean = _resample(record.clean[:, start_native:stop_native], source_rate, target_rate)
    eog = _resample(
        np.stack([record.veog, record.heog], axis=0)[:, start_native:stop_native],
        source_rate,
        target_rate,
    )
    y_windows, valid = _pad_windows(contaminated, window_samples)
    x_windows, valid_clean = _pad_windows(clean, window_samples)
    eog_windows, valid_eog = _pad_windows(eog, window_samples)
    if not np.array_equal(valid, valid_clean) or not np.array_equal(valid, valid_eog):
        raise AssertionError("aligned query streams yielded different windowing")
    centered = eog - eog.mean(axis=1, keepdims=True)
    scale = np.maximum(eog.std(axis=1, keepdims=True), 1e-8)
    magnitude = np.sqrt(np.mean((centered / scale) ** 2, axis=0))
    threshold = float(attenuation_scale)
    artifact = magnitude >= threshold
    artifact_windows, artifact_valid = _pad_windows(artifact[None, :], window_samples)
    if not np.array_equal(valid, artifact_valid):
        raise AssertionError("artifact mask misalignment")
    return WindowedQuery(
        contaminated=y_windows,
        clean=x_windows,
        eog=eog_windows,
        valid_samples=valid,
        artifact_mask=artifact_windows[:, 0].astype(bool),
        sampling_rate=target_rate,
    )


def calibration_batch(
    record: KladosRecord,
    *,
    duration_seconds: float,
    source_rate: int,
    target_rate: int,
    source_label: str,
    eog_override: np.ndarray | None = None,
) -> CalibrationBatch:
    native_samples = min(int(round(duration_seconds * source_rate)), record.samples)
    if native_samples <= 0:
        raise ValueError("calibration duration must be positive")
    eeg = _resample(record.contaminated[:, :native_samples], source_rate, target_rate)
    native_eog = np.stack([record.veog, record.heog], axis=0)[:, :native_samples]
    eog = _resample(native_eog, source_rate, target_rate)
    if eog_override is not None:
        override = np.asarray(eog_override, dtype=np.float64)
        if override.shape != eog.shape:
            raise ValueError("shuffled EOG shape changed")
        eog = override
    return CalibrationBatch(
        eeg=eeg,
        eog=eog,
        participant="unresolved",
        source_record=source_label,
        sampling_rate=float(target_rate),
    )


def block_shuffle_reference(eog: np.ndarray, *, block_samples: int, seed: int) -> np.ndarray:
    """Shuffle whole temporal blocks so rank, dimensions and local spectrum remain matched."""
    reference = np.asarray(eog)
    if reference.ndim != 2 or block_samples <= 0:
        raise ValueError("invalid EOG block shuffle input")
    blocks = [reference[:, start : start + block_samples] for start in range(0, reference.shape[1], block_samples)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    return np.concatenate([blocks[int(index)] for index in order], axis=1)


def oracle_transfer(
    record: KladosRecord,
    *,
    start_seconds: float,
    stop_seconds: float,
    sampling_rate: int,
    target_rank: int,
) -> P0Transfer:
    start = int(round(start_seconds * sampling_rate))
    stop = min(int(round(stop_seconds * sampling_rate)), record.samples)
    eog = np.stack([record.veog, record.heog], axis=0)[:, start:stop]
    artifact = record.contaminated[:, start:stop] - record.clean[:, start:stop]
    transfer = artifact @ eog.T @ np.linalg.pinv(eog @ eog.T)
    predicted = transfer @ eog
    basis_full, singular_values, _ = np.linalg.svd(transfer, full_matrices=False)
    rank = min(target_rank, basis_full.shape[1])
    basis = basis_full[:, :rank]
    projector = basis @ basis.T
    residual = float(np.linalg.norm(artifact - predicted) / max(np.linalg.norm(artifact), 1e-12))
    return P0Transfer(
        transfer_matrix=transfer,
        eeg_subspace_basis=basis,
        projector=projector,
        predicted_contamination=predicted,
        eog_mean=np.zeros((eog.shape[0], 1), dtype=np.float64),
        eeg_mean=np.zeros((artifact.shape[0], 1), dtype=np.float64),
        rank=rank,
        diagnostics={
            "oracle_derived_from_paired_query": 1,
            "mixture_relative_residual": residual,
            "singular_values": singular_values.tolist(),
        },
    )


def fit_source_controls(
    *,
    matching_record: KladosRecord,
    wrong_record: KladosRecord,
    duration_seconds: float,
    source_rate: int,
    target_rate: int,
    config: P0Config,
    movement_threshold: float,
    seed: int,
) -> dict[str, P0FitOutcome]:
    matching = calibration_batch(
        matching_record,
        duration_seconds=duration_seconds,
        source_rate=source_rate,
        target_rate=target_rate,
        source_label=f"sim{matching_record.record_id}",
    )
    wrong = calibration_batch(
        wrong_record,
        duration_seconds=duration_seconds,
        source_rate=source_rate,
        target_rate=target_rate,
        source_label=f"sim{wrong_record.record_id}",
    )
    shuffled = block_shuffle_reference(
        matching.eog,
        block_samples=max(1, int(round(target_rate * 2.0))),
        seed=seed,
    )
    shuffled_batch = CalibrationBatch(
        eeg=matching.eeg,
        eog=shuffled,
        participant=matching.participant,
        source_record=f"{matching.source_record}_block_shuffled",
        sampling_rate=matching.sampling_rate,
    )
    return {
        "matching_p0": fit_p0(matching, config, movement_threshold=movement_threshold),
        "wrong_source_p0": fit_p0(wrong, config, movement_threshold=movement_threshold),
        "shuffled_calibration_p0": fit_p0(
            shuffled_batch, config, movement_threshold=movement_threshold
        ),
    }


def population_source_transfer(
    outcomes: Iterable[P0FitOutcome], *, target_rank: int
) -> P0FitOutcome:
    """Build a fixed-rank population control from eligible development-source projectors."""
    eligible = [outcome.transfer for outcome in outcomes if outcome.transfer is not None]
    if not eligible:
        return P0FitOutcome("ineligible", None, ("no_population_sources",))
    mean_projector = np.mean([item.projector for item in eligible], axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (mean_projector + mean_projector.T))
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order[:target_rank]]
    projector = basis @ basis.T
    transfer = np.mean([item.transfer_matrix for item in eligible], axis=0)
    estimate = P0Transfer(
        transfer_matrix=transfer,
        eeg_subspace_basis=basis,
        projector=projector,
        predicted_contamination=np.empty((projector.shape[0], 0), dtype=np.float64),
        eog_mean=np.mean([item.eog_mean for item in eligible], axis=0),
        eeg_mean=np.mean([item.eeg_mean for item in eligible], axis=0),
        rank=target_rank,
        diagnostics={
            "population_source_count": len(eligible),
            "participant_independence_verified": 0,
            "top_mean_projector_eigenvalues": eigenvalues[order[: target_rank + 2]].tolist(),
        },
    )
    return P0FitOutcome("eligible", estimate, ())


def orthogonal_subtraction(observed: np.ndarray, projector: np.ndarray) -> np.ndarray:
    y = np.asarray(observed)
    return y - np.einsum("cd,...dt->...ct", projector, y)
