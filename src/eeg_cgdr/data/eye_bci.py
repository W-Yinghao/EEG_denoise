"""Bounded reader for two explicitly registered Eye-BCI Neuroscan records.

This module never enumerates the Eye-BCI directory.  A caller names an exact
participant, session, and relative CSV path, and the reader consumes at most a
small, fixed number of rows from that file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


EYE_BCI_ROOT = Path(
    "/projects/EEG-foundation-model/eye_bci/syn64005218-neuroscan"
)

# M1/M2 are intentionally excluded until their source semantics are confirmed.
EYE_BCI_SCALP_CHANNELS = (
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POZ",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "OZ",
    "O2",
    "CB2",
)


@dataclass(frozen=True)
class EyeBciTarget:
    participant_id: str
    session_id: str
    relative_path: Path


DEFAULT_EYE_BCI_TARGETS = (
    EyeBciTarget("S01", "Sess01", Path("S01/Sess01/Neuroscan/ME011.csv")),
    EyeBciTarget("S02", "Sess01", Path("S02/Sess01/Neuroscan/ME021.csv")),
)


@dataclass(frozen=True)
class EyeBciBoundedRecord:
    participant_id: str
    session_id: str
    relative_path: Path
    eeg: np.ndarray
    heo: np.ndarray
    time: np.ndarray
    sampling_rate_hz: float
    time_units: str
    signal_units: str
    rows_read: int
    truncated: bool
    time_step_relative_jitter: float

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.eeg.shape[0]), int(self.eeg.shape[1])

    @property
    def finite(self) -> bool:
        return bool(
            np.isfinite(self.eeg).all()
            and np.isfinite(self.heo).all()
            and np.isfinite(self.time).all()
        )


def _safe_target_path(root: Path, target: EyeBciTarget) -> Path:
    relative = target.relative_path
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Eye-BCI target must be a contained relative path")
    expected_parent = Path(target.participant_id) / target.session_id / "Neuroscan"
    if relative.parent != expected_parent or relative.suffix.lower() != ".csv":
        raise ValueError(
            "Eye-BCI target path does not match its participant/session declaration"
        )
    root_resolved = root.resolve(strict=True)
    path = root / relative
    if path.is_symlink():
        raise ValueError(f"Eye-BCI target cannot be a symlink: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ValueError(f"Eye-BCI target is outside the registered root: {relative}")
    return resolved


def _infer_sampling_rate(time: np.ndarray) -> tuple[float, str, float]:
    if time.ndim != 1 or time.size < 2 or not np.isfinite(time).all():
        raise ValueError("Eye-BCI Time column is too short or non-finite")
    increments = np.diff(time)
    if np.any(increments <= 0):
        raise ValueError("Eye-BCI Time column is not strictly increasing")
    step = float(np.median(increments))
    jitter = float(np.quantile(np.abs(increments - step), 0.95) / step)
    candidates: list[tuple[float, str]] = []
    hz_seconds = 1.0 / step
    hz_milliseconds = 1000.0 / step
    if 50.0 <= hz_seconds <= 5000.0:
        candidates.append((hz_seconds, "seconds"))
    if 50.0 <= hz_milliseconds <= 5000.0:
        candidates.append((hz_milliseconds, "milliseconds"))
    if len(candidates) != 1:
        raise ValueError(
            f"cannot infer an unambiguous EEG sampling rate from Time step {step}"
        )
    sampling_rate, units = candidates[0]
    if jitter > 0.05:
        raise ValueError(f"Eye-BCI Time step jitter is too large: {jitter}")
    return float(sampling_rate), units, jitter


def read_eye_bci_target(
    root: Path,
    target: EyeBciTarget,
    *,
    max_rows: int = 4096,
) -> EyeBciBoundedRecord:
    """Read a bounded prefix of one explicit CSV without listing its directory."""

    if max_rows < 2 or max_rows > 16384:
        raise ValueError("max_rows must remain bounded in [2, 16384]")
    path = _safe_target_path(root, target)
    selected = ("Time", *EYE_BCI_SCALP_CHANNELS, "HEO")
    rows: list[list[float]] = []
    truncated = False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty Eye-BCI CSV: {target.relative_path}") from exc
        if any(not name.strip() for name in header) or len(header) != len(set(header)):
            raise ValueError(f"invalid Eye-BCI CSV header: {target.relative_path}")
        missing = [name for name in selected if name not in header]
        if missing:
            raise ValueError(f"Eye-BCI CSV lacks required columns: {missing}")
        indices = [header.index(name) for name in selected]
        for row_number, row in enumerate(reader, start=1):
            if row_number > max_rows:
                truncated = True
                break
            if len(row) != len(header):
                raise ValueError(
                    f"Eye-BCI CSV row width differs from header at row {row_number}"
                )
            try:
                values = [float(row[index]) for index in indices]
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric selected Eye-BCI value at row {row_number}"
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite selected Eye-BCI value at row {row_number}")
            rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"too few readable Eye-BCI rows: {target.relative_path}")

    matrix = np.asarray(rows, dtype=np.float64).T
    time = matrix[0]
    channel_stop = 1 + len(EYE_BCI_SCALP_CHANNELS)
    eeg = np.ascontiguousarray(matrix[1:channel_stop])
    heo = np.ascontiguousarray(matrix[channel_stop])
    sampling_rate, time_units, jitter = _infer_sampling_rate(time)
    eeg.setflags(write=False)
    heo.setflags(write=False)
    time.setflags(write=False)
    return EyeBciBoundedRecord(
        participant_id=target.participant_id,
        session_id=target.session_id,
        relative_path=target.relative_path,
        eeg=eeg,
        heo=heo,
        time=time,
        sampling_rate_hz=sampling_rate,
        time_units=time_units,
        # The audited CSV headers contain no amplitude-unit metadata.  Do not
        # infer microvolts from magnitude alone.
        signal_units="unknown_not_encoded_in_csv",
        rows_read=len(rows),
        truncated=truncated,
        time_step_relative_jitter=jitter,
    )


def read_default_eye_bci_targets(
    *,
    root: Path = EYE_BCI_ROOT,
    max_rows: int = 4096,
    targets: Sequence[EyeBciTarget] = DEFAULT_EYE_BCI_TARGETS,
) -> tuple[EyeBciBoundedRecord, ...]:
    """Read exactly the two frozen validation targets, never a directory scan."""

    frozen = tuple(targets)
    if frozen != DEFAULT_EYE_BCI_TARGETS:
        raise ValueError("CPU validation accepts only the two frozen Eye-BCI targets")
    keys = {(target.participant_id, target.session_id) for target in frozen}
    paths = {target.relative_path for target in frozen}
    if len(keys) != 2 or len(paths) != 2:
        raise ValueError("Eye-BCI validation targets must be distinct participant/session records")
    return tuple(
        read_eye_bci_target(root, target, max_rows=max_rows) for target in frozen
    )
