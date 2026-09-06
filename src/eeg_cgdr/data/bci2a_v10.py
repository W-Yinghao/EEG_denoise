"""Strict BCI Competition IV-2a data boundary for V10.

The loader keeps the 22 EEG and three EOG channels separate.  Query-facing
callers receive EEG only; EOG and task annotations are exposed by an explicit
evaluator method after cleaner outputs have been frozen.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BCI2ASession:
    subject: int
    session: str
    path: Path


def discover_sessions(root: Path) -> list[BCI2ASession]:
    candidates: dict[tuple[int, str], Path] = {}
    for path in root.rglob("A??[TE].gdf"):
        name = path.stem
        if len(name) == 4 and name[0] == "A" and name[1:3].isdigit() and name[3] in "TE":
            key = (int(name[1:3]), name[3])
            candidates.setdefault(key, path)
    return [BCI2ASession(s, session, candidates[(s, session)]) for s in range(1, 10) for session in "TE" if (s, session) in candidates]


def inspect_gdf(session: BCI2ASession) -> dict[str, object]:
    import mne

    raw = mne.io.read_raw_gdf(session.path, preload=False, verbose="ERROR")
    descriptions = Counter(str(value) for value in raw.annotations.description)
    return {
        "subject": session.subject,
        "session": session.session,
        "path": str(session.path),
        "sampling_rate": float(raw.info["sfreq"]),
        "channels": len(raw.ch_names),
        "channel_names": list(raw.ch_names),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "annotation_counts": dict(sorted(descriptions.items())),
        "annotation_onsets_finite": bool(np.isfinite(raw.annotations.onset).all()),
    }


def load_physical(session: BCI2ASession) -> tuple[np.ndarray, np.ndarray, float, list[str], list[str]]:
    """Load finite volts, mapping GDF missing-value sentinels to NaN then interpolating.

    This function is for preprocessing/evaluation jobs.  Models must consume
    arrays produced by the query-only builder rather than call it directly.
    """

    import mne

    raw = mne.io.read_raw_gdf(session.path, preload=True, verbose="ERROR")
    data = raw.get_data().astype(np.float64, copy=False)
    if data.shape[0] != 25:
        raise ValueError(f"{session.path.name}: expected 25 channels, got {data.shape[0]}")
    data[~np.isfinite(data)] = np.nan
    # MNE normally maps the GDF INT24 minimum to NaN.  Reject any remaining
    # impossible volt-scale values before per-channel linear interpolation.
    data[np.abs(data) > 1.0] = np.nan
    for channel in range(data.shape[0]):
        valid = np.flatnonzero(np.isfinite(data[channel]))
        if valid.size < 2:
            raise ValueError(f"{session.path.name}: channel {channel} has insufficient finite data")
        missing = np.flatnonzero(~np.isfinite(data[channel]))
        if missing.size:
            data[channel, missing] = np.interp(missing, valid, data[channel, valid])
    return data[:22].astype(np.float32), data[22:].astype(np.float32), float(raw.info["sfreq"]), list(raw.ch_names[:22]), list(raw.ch_names[22:])


def load_with_events(session: BCI2ASession) -> tuple[np.ndarray, np.ndarray, float, list[tuple[float, str]]]:
    eeg, eog, sfreq, _, _ = load_physical(session)
    import mne

    raw = mne.io.read_raw_gdf(session.path, preload=False, verbose="ERROR")
    events = [(float(onset), str(description)) for onset, description in zip(raw.annotations.onset, raw.annotations.description)]
    return eeg, eog, sfreq, events


def load_gdf_channels(path: Path, *, eeg_channels: int) -> tuple[np.ndarray, np.ndarray, float, list[tuple[float, str]]]:
    """Generic finite GDF loader used only by the protocol-triggered 2b audit."""
    import mne

    raw=mne.io.read_raw_gdf(path,preload=True,verbose="ERROR");data=raw.get_data().astype(np.float64,copy=False);data[~np.isfinite(data)]=np.nan;data[np.abs(data)>1.0]=np.nan
    if data.shape[0] != eeg_channels+3: raise ValueError(f"{path.name}: expected {eeg_channels+3} channels, got {data.shape[0]}")
    for channel in range(data.shape[0]):
        valid=np.flatnonzero(np.isfinite(data[channel]));missing=np.flatnonzero(~np.isfinite(data[channel]))
        if valid.size<2:raise ValueError(f"{path.name}: channel {channel} insufficient finite samples")
        if missing.size:data[channel,missing]=np.interp(missing,valid,data[channel,valid])
    events=[(float(onset),str(description)) for onset,description in zip(raw.annotations.onset,raw.annotations.description)]
    return data[:eeg_channels].astype(np.float32),data[eeg_channels:].astype(np.float32),float(raw.info["sfreq"]),events


def loso_manifest() -> list[dict[str, object]]:
    return [
        {
            "fold": heldout - 1,
            "heldout_subject": heldout,
            "outer_training_subjects": ";".join(str(s) for s in range(1, 10) if s != heldout),
            "same_session_protocol": "support_calibration_prefix_to_later_MI",
            "cross_session_protocol": "T_calibration_to_E_MI",
            "outcomes_opened_for_split": 0,
        }
        for heldout in range(1, 10)
    ]


__all__ = ["BCI2ASession", "discover_sessions", "inspect_gdf", "load_physical", "load_with_events", "load_gdf_channels", "loso_manifest"]
