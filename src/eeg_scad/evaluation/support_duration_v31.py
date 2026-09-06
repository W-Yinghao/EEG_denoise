"""Exact chronological-prefix support-duration contract for V31.

This module intentionally does not reuse V30's duration helpers.  V30 used
linspace-spaced windows and full-120-second EOG normalization.  V31 uses every
non-overlapping two-second window wholly contained in the declared prefix and
estimates EOG coordinates from that prefix only.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from eeg_scad.data.counterfactual_pairs import _load_signal, fold_eeg_scale
from eeg_scad.data.v24_coordinate_contract import robust_center_scale
from eeg_scad.evaluation.common_panel_v30 import (
    SESSIONS,
    TASKS,
    content_digest,
    support_bank_index,
)


def exact_support_starts(
    duration_seconds: int,
    window_samples: int = 200,
    rate_hz: int = 100,
) -> list[int]:
    """Return non-overlapping chronological windows wholly inside a prefix."""
    if duration_seconds < 0:
        raise ValueError("duration must be non-negative")
    count = (duration_seconds * rate_hz) // window_samples
    return [index * window_samples for index in range(count)]


def duration_contract(
    duration_seconds: int,
    window_samples: int = 200,
    rate_hz: int = 100,
) -> dict[str, Any]:
    starts = exact_support_starts(duration_seconds, window_samples, rate_hz)
    return {
        "duration_seconds": duration_seconds,
        "acquisition_span_seconds": duration_seconds,
        "window_samples": window_samples,
        "sampling_rate_hz": rate_hz,
        "window_count": len(starts),
        "effective_samples": len(starts) * window_samples,
        "effective_seconds": len(starts) * window_samples / rate_hz,
        "starts": starts,
    }


def standardized_prefix_support(
    eeg: np.ndarray,
    eog: np.ndarray,
    eeg_scale: np.ndarray,
    duration_seconds: int,
    window_samples: int = 200,
    rate_hz: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Build support using only samples in the declared acquisition prefix."""
    stop = duration_seconds * rate_hz
    if duration_seconds <= 0:
        raise ValueError("zero duration is an architectural population bypass")
    if min(eeg.shape[-1], eog.shape[-1]) < stop:
        raise RuntimeError("registered support signal is shorter than the declared prefix")
    starts = exact_support_starts(duration_seconds, window_samples, rate_hz)
    if not starts:
        raise RuntimeError("prefix does not contain one complete support window")
    # The normalization reads exactly the acquisition prefix.  For 5 s this
    # includes the final 1 s for coordinate estimation while model exposure is
    # the two complete, non-overlapping 2 s windows (4 s).
    center, scale = robust_center_scale(eog[:, :stop])
    support_eeg = np.stack(
        [eeg[:, start:start + window_samples] / eeg_scale[:, None] for start in starts]
    )
    support_eog = np.stack(
        [(eog[:, start:start + window_samples] - center[:, None]) / scale[:, None] for start in starts]
    )
    return (
        support_eeg.astype(np.float32),
        support_eog.astype(np.float32),
        np.asarray(center, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
        starts,
    )


def exact_support_episode(
    root: Path,
    owner: str,
    session: str,
    task: str,
    eeg_scale: np.ndarray,
    duration_seconds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], str]:
    try:
        eeg, eog = _load_signal(root, owner, session, task)
        actual_task = task
    except FileNotFoundError:
        actual_task = next(value for value in TASKS if value != task)
        eeg, eog = _load_signal(root, owner, session, actual_task)
    values = standardized_prefix_support(eeg, eog, eeg_scale, duration_seconds)
    return (*values, actual_task)


def materialize_exact_support_bank(
    data: Mapping[str, Any],
    fold: Mapping[str, Any],
    owners: list[str],
    destination: Path,
    durations: Iterable[int] = (5, 10, 30, 120),
) -> list[dict[str, Any]]:
    """Materialize immutable V31 support arrays without touching V30 assets."""
    root = Path(data["v19_derived_root"])
    eeg_scale = fold_eeg_scale(data, list(fold["train"]))
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for duration in durations:
        if duration == 0:
            continue
        contract = duration_contract(int(duration))
        eeg_values: list[np.ndarray] = []
        eog_values: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for owner in owners:
            for session in SESSIONS:
                for task in TASKS:
                    eeg, eog, center, scale, starts, actual = exact_support_episode(
                        root, owner, session, task, eeg_scale, int(duration)
                    )
                    eeg_values.append(eeg)
                    eog_values.append(eog)
                    centers.append(center)
                    scales.append(scale)
                    rows.append({
                        "fold": fold["fold"],
                        "owner": owner,
                        "session": session,
                        "task": task,
                        "actual_task": actual,
                        **contract,
                        "starts": ";".join(map(str, starts)),
                        "sample_ranges": ";".join(f"{start}:{start + 200}" for start in starts),
                        "normalization_prefix_samples": int(duration) * 100,
                        "normalization_source": "duration_prefix_only",
                        "overlap_samples": 0,
                        "repeated_samples": 0,
                        "future_support_samples_read": 0,
                        "digest": content_digest((eeg, eog, center, scale, np.asarray(starts, dtype=np.int32))),
                    })
        arrays[f"eeg_{duration}"] = np.asarray(eeg_values, dtype=np.float32)
        arrays[f"eog_{duration}"] = np.asarray(eog_values, dtype=np.float32)
        arrays[f"center_{duration}"] = np.asarray(centers, dtype=np.float64)
        arrays[f"scale_{duration}"] = np.asarray(scales, dtype=np.float64)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return rows


def attach_exact_support(
    batch: Mapping[str, Any],
    bank: Mapping[str, np.ndarray],
    owners: list[str],
    duration_seconds: int,
) -> dict[str, Any]:
    if duration_seconds == 0:
        result = dict(batch)
        result["population_bypass"] = True
        return result
    eeg: list[np.ndarray] = []
    eog: list[np.ndarray] = []
    for meta in batch["meta"]:
        index = support_bank_index(meta["participant"], meta["session"], meta["task"], owners)
        eeg.append(bank[f"eeg_{duration_seconds}"][index])
        eog.append(bank[f"eog_{duration_seconds}"][index])
    result = dict(batch)
    result["support_eeg"] = np.asarray(eeg)
    result["support_eog"] = np.asarray(eog)
    result["population_bypass"] = False
    return result


def validate_exact_manifest(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    for row in rows:
        duration = int(row["duration_seconds"])
        starts = [int(value) for value in str(row["starts"]).split(";") if value]
        expected = exact_support_starts(duration)
        if starts != expected:
            raise RuntimeError(f"non-exact chronological prefix: {row}")
        occupied = [sample for start in starts for sample in range(start, start + 200)]
        if len(occupied) != len(set(occupied)):
            raise RuntimeError(f"overlapping/repeated support samples: {row}")
        if occupied and max(occupied) >= duration * 100:
            raise RuntimeError(f"future support read: {row}")
        if int(row["normalization_prefix_samples"]) != duration * 100:
            raise RuntimeError(f"future normalization: {row}")
    five = [row for row in rows if int(row["duration_seconds"]) == 5]
    if not five or any(int(row["window_count"]) != 2 or float(row["effective_seconds"]) != 4 for row in five):
        raise RuntimeError("5 s contract must expose exactly two windows / four effective seconds")
    return {
        "rows": len(rows),
        "durations": sorted({int(row["duration_seconds"]) for row in rows}),
        "no_overlap": True,
        "no_repeated_samples": True,
        "no_future_support_read": True,
        "prefix_only_normalization": True,
    }


def timed_support_encoding(encode: Any, eeg: np.ndarray, eog: np.ndarray) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    encoded = encode(eeg, eog)
    return encoded, 1000.0 * (time.perf_counter() - started) / max(len(eeg), 1)


__all__ = [
    "attach_exact_support",
    "duration_contract",
    "exact_support_episode",
    "exact_support_starts",
    "materialize_exact_support_bank",
    "standardized_prefix_support",
    "timed_support_encoding",
    "validate_exact_manifest",
]
