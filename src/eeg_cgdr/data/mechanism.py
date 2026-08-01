"""Frozen Klados source-record partitions for the repaired mechanism audit.

The v4 files expose 54 aligned paired source records but do not expose a
reliable participant mapping.  This module therefore never invents participant
identities.  It applies the source-record split before windowing and normalizes
each EEG channel from the clean training records only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import resample_poly

from .klados import KladosRecord
from ..operators import CalibrationBatch


KLADOS_TRAIN_RECORDS = tuple(range(1, 31))
KLADOS_DEVELOPMENT_RECORDS = (31, 32, 33, 34, 35, 36, 44, 45)
KLADOS_NATIVE_CHANNEL_ORDER = (
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
)
KLADOS_UNTOUCHED_RECORDS = (
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
)


def assert_frozen_source_partition() -> None:
    """Assert the preregistered source-record split covers every record once."""

    groups = (
        set(KLADOS_TRAIN_RECORDS),
        set(KLADOS_DEVELOPMENT_RECORDS),
        set(KLADOS_UNTOUCHED_RECORDS),
    )
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError("Klados mechanism source-record groups overlap")
    if set.union(*groups) != set(range(1, 55)):
        raise AssertionError("Klados mechanism source-record groups do not cover sim01-sim54")


def write_mechanism_split_manifest(
    path: Path, *, calibration_seconds: float = 10.0, guard_seconds: float = 1.0
) -> None:
    """Write the small, source-record-only preregistered audit split."""

    if calibration_seconds <= 0.0 or guard_seconds < 0.0:
        raise ValueError("split calibration/guard durations are invalid")
    query_start = calibration_seconds + guard_seconds

    fields = (
        "dataset_version",
        "outer_fold",
        "split",
        "participant",
        "session",
        "record",
        "calibration_start",
        "calibration_end",
        "query_start",
        "query_end",
        "sampling_rate",
        "status",
    )
    rows: list[dict[str, object]] = []
    for record_id in KLADOS_TRAIN_RECORDS:
        rows.append(
            {
                "dataset_version": "klados_bamidis_v4",
                "outer_fold": "klados_v4_source_record_mechanism_audit",
                "split": "train",
                "participant": "unresolved_not_claimed",
                "session": "source_record",
                "record": f"sim{record_id:02d}",
                "calibration_start": "N/A",
                "calibration_end": "N/A",
                "query_start": "N/A",
                "query_end": "N/A",
                "sampling_rate": 200,
                "status": "population_prior_and_projector",
            }
        )
    for split, source_records in (
        ("development", KLADOS_DEVELOPMENT_RECORDS),
        ("untouched", KLADOS_UNTOUCHED_RECORDS),
    ):
        for record_id in source_records:
            rows.append(
                {
                    "dataset_version": "klados_bamidis_v4",
                    "outer_fold": "klados_v4_source_record_mechanism_audit",
                    "split": split,
                    "participant": "unresolved_not_claimed",
                    "session": "source_record",
                    "record": f"sim{record_id:02d}",
                    "calibration_start": 0,
                    "calibration_end": calibration_seconds,
                    "query_start": query_start,
                    "query_end": "record_end",
                    "sampling_rate": 200,
                    "status": "source_record_only_participant_mapping_unavailable",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class ChannelNormalizer:
    """Per-channel statistics fitted on complete clean training records."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    source_records: tuple[int, ...]
    sample_count: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.standard_deviation, dtype=np.float64)
        records = tuple(int(value) for value in self.source_records)
        if mean.ndim != 1 or scale.shape != mean.shape or mean.size < 1:
            raise ValueError("normalizer moments must be matching channel vectors")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("normalizer moments must be finite")
        if np.any(scale <= np.finfo(np.float64).eps):
            raise ValueError("normalizer scale must be strictly positive")
        if not records or len(records) != len(set(records)):
            raise ValueError("normalizer source records must be unique and non-empty")
        if int(self.sample_count) != self.sample_count or self.sample_count < 1:
            raise ValueError("normalizer sample_count must be a positive integer")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", scale)
        object.__setattr__(self, "source_records", records)
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def transform(self, signal: np.ndarray) -> np.ndarray:
        value = np.asarray(signal, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != self.mean.shape[0]:
            raise ValueError("EEG does not match the channel normalizer")
        if not np.isfinite(value).all():
            raise ValueError("cannot normalize non-finite EEG")
        return (value - self.mean[:, None]) / self.standard_deviation[:, None]


@dataclass(frozen=True)
class WindowedSignal:
    values: np.ndarray
    valid_time_weight: np.ndarray


@dataclass(frozen=True)
class KladosMechanismRecord:
    source_record: int
    calibration: CalibrationBatch
    observed_windows: np.ndarray
    clean_windows: np.ndarray
    eog_windows: np.ndarray
    valid_time_weight: np.ndarray
    observed_continuous: np.ndarray
    clean_continuous: np.ndarray
    eog_continuous: np.ndarray
    eog_calibration_mean: np.ndarray
    eog_calibration_standard_deviation: np.ndarray
    sampling_rate: int
    query_start_seconds: float
    query_end_seconds: float


def standardize_reference_from_support(
    support_eog: np.ndarray, query_eog: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each EOG regressor from support statistics only."""

    support = np.asarray(support_eog, dtype=np.float64)
    query = np.asarray(query_eog, dtype=np.float64)
    if support.ndim != 2 or query.ndim != 2 or support.shape[0] != query.shape[0]:
        raise ValueError("support/query EOG must be aligned channel-major arrays")
    if not np.isfinite(support).all() or not np.isfinite(query).all():
        raise ValueError("support/query EOG contains non-finite values")
    mean = support.mean(axis=1, keepdims=True)
    standard_deviation = support.std(axis=1, keepdims=True)
    if np.any(standard_deviation <= np.finfo(np.float64).eps):
        raise ValueError("support EOG contains a constant reference channel")
    return (
        (support - mean) / standard_deviation,
        (query - mean) / standard_deviation,
        mean,
        standard_deviation,
    )


def _record_map(records: Sequence[KladosRecord]) -> dict[int, KladosRecord]:
    mapped = {record.record_id: record for record in records}
    if len(mapped) != len(records):
        raise ValueError("duplicate Klados source record")
    return mapped


def select_records(
    records: Sequence[KladosRecord], source_records: Iterable[int]
) -> tuple[KladosRecord, ...]:
    mapped = _record_map(records)
    selected: list[KladosRecord] = []
    for record_id in source_records:
        try:
            selected.append(mapped[int(record_id)])
        except KeyError as exc:
            raise ValueError(f"missing Klados source record sim{int(record_id):02d}") from exc
    return tuple(selected)


def fit_channel_normalizer(
    records: Sequence[KladosRecord],
    source_records: Iterable[int] = KLADOS_TRAIN_RECORDS,
) -> ChannelNormalizer:
    """Fit channel moments without allowing windows to reweight a record."""

    frozen = tuple(int(item) for item in source_records)
    if not frozen or len(set(frozen)) != len(frozen):
        raise ValueError("normalization source records must be unique and non-empty")
    selected = select_records(records, frozen)
    channels = selected[0].clean.shape[0]
    total = np.zeros(channels, dtype=np.float64)
    total_square = np.zeros(channels, dtype=np.float64)
    samples = 0
    for record in selected:
        clean = np.asarray(record.clean, dtype=np.float64)
        if clean.shape[0] != channels or not np.isfinite(clean).all():
            raise ValueError(f"invalid clean training record sim{record.record_id:02d}")
        total += clean.sum(axis=1)
        total_square += np.square(clean).sum(axis=1)
        samples += clean.shape[1]
    mean = total / samples
    variance = np.maximum(total_square / samples - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("clean training data contain a constant EEG channel")
    return ChannelNormalizer(mean, scale, frozen, int(samples))


def resample_signal(array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sampling rates must be positive")
    if source_rate == target_rate:
        return value.copy()
    divisor = int(np.gcd(source_rate, target_rate))
    return resample_poly(
        value,
        target_rate // divisor,
        source_rate // divisor,
        axis=-1,
    )


def window_after_normalization(signal: np.ndarray, window_samples: int) -> WindowedSignal:
    """Pad with true normalized zero and return one frame weight per sample."""

    value = np.asarray(signal, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 1 or window_samples < 1:
        raise ValueError("invalid signal/window shape")
    count = int(np.ceil(value.shape[1] / window_samples))
    windows = np.zeros((count, value.shape[0], window_samples), dtype=np.float64)
    weight = np.zeros((count, window_samples), dtype=np.float64)
    for index in range(count):
        start = index * window_samples
        stop = min(start + window_samples, value.shape[1])
        valid = stop - start
        windows[index, :, :valid] = value[:, start:stop]
        weight[index, :valid] = 1.0
    return WindowedSignal(windows, weight)


def prepare_clean_training_windows(
    records: Sequence[KladosRecord],
    normalizer: ChannelNormalizer,
    *,
    source_records: Iterable[int] = KLADOS_TRAIN_RECORDS,
    source_rate: int = 200,
    target_rate: int = 256,
    window_samples: int = 512,
) -> WindowedSignal:
    """Use every sample of every frozen clean training source record."""

    value_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for record in select_records(records, source_records):
        normalized = normalizer.transform(record.clean)
        resampled = resample_signal(normalized, source_rate, target_rate)
        windowed = window_after_normalization(resampled, window_samples)
        value_parts.append(windowed.values)
        weight_parts.append(windowed.valid_time_weight)
    return WindowedSignal(
        np.concatenate(value_parts, axis=0),
        np.concatenate(weight_parts, axis=0),
    )


def prepare_population_calibration(
    record: KladosRecord,
    normalizer: ChannelNormalizer,
    *,
    source_rate: int = 200,
    target_rate: int = 256,
) -> CalibrationBatch:
    """Use one complete training record to estimate a population operator."""

    eeg = resample_signal(
        normalizer.transform(record.contaminated), source_rate, target_rate
    )
    eog_raw = resample_signal(
        np.stack([record.veog, record.heog], axis=0), source_rate, target_rate
    )
    eog, _, _, _ = standardize_reference_from_support(eog_raw, eog_raw)
    return CalibrationBatch(
        eeg=eeg,
        eog=eog,
        participant="training_source_record",
        source_record=f"sim{record.record_id:02d}",
        sampling_rate=float(target_rate),
    )


def prepare_mechanism_record(
    record: KladosRecord,
    normalizer: ChannelNormalizer,
    *,
    source_rate: int = 200,
    target_rate: int = 256,
    window_samples: int = 512,
    calibration_seconds: float = 10.0,
    guard_seconds: float = 1.0,
) -> KladosMechanismRecord:
    """Create non-overlapping support and the complete remaining query."""

    calibration_stop = int(round(calibration_seconds * source_rate))
    query_start = int(round((calibration_seconds + guard_seconds) * source_rate))
    if calibration_stop <= 0 or query_start >= record.samples:
        raise ValueError(
            f"sim{record.record_id:02d} cannot support the frozen calibration/guard"
        )

    support_eeg = normalizer.transform(record.contaminated[:, :calibration_stop])
    support_eeg = resample_signal(support_eeg, source_rate, target_rate)
    support_eog_raw = resample_signal(
        np.stack([record.veog, record.heog], axis=0)[:, :calibration_stop],
        source_rate,
        target_rate,
    )

    observed = normalizer.transform(record.contaminated[:, query_start:])
    clean = normalizer.transform(record.clean[:, query_start:])
    eog_raw = np.stack([record.veog, record.heog], axis=0)[:, query_start:]
    observed = resample_signal(observed, source_rate, target_rate)
    clean = resample_signal(clean, source_rate, target_rate)
    eog_raw = resample_signal(eog_raw, source_rate, target_rate)
    support_eog, eog, eog_mean, eog_standard_deviation = (
        standardize_reference_from_support(support_eog_raw, eog_raw)
    )
    calibration = CalibrationBatch(
        eeg=support_eeg,
        eog=support_eog,
        participant="unresolved_source_record",
        source_record=f"sim{record.record_id:02d}",
        sampling_rate=float(target_rate),
    )
    observed_windows = window_after_normalization(observed, window_samples)
    clean_windows = window_after_normalization(clean, window_samples)
    eog_windows = window_after_normalization(eog, window_samples)
    if not np.array_equal(
        observed_windows.valid_time_weight, clean_windows.valid_time_weight
    ) or not np.array_equal(
        observed_windows.valid_time_weight, eog_windows.valid_time_weight
    ):
        raise AssertionError("aligned Klados streams lost their shared valid-time mask")
    return KladosMechanismRecord(
        source_record=record.record_id,
        calibration=calibration,
        observed_windows=observed_windows.values,
        clean_windows=clean_windows.values,
        eog_windows=eog_windows.values,
        valid_time_weight=observed_windows.valid_time_weight,
        observed_continuous=observed,
        clean_continuous=clean,
        eog_continuous=eog,
        eog_calibration_mean=eog_mean,
        eog_calibration_standard_deviation=eog_standard_deviation,
        sampling_rate=target_rate,
        query_start_seconds=calibration_seconds + guard_seconds,
        query_end_seconds=record.samples / source_rate,
    )


assert_frozen_source_partition()
