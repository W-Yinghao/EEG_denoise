"""Strict BCI Competition IV-2a trial loader for V32P.

The official MATLAB files contain the true labels for both sessions.  We keep
one non-overlapping EEGNet input per trial so repeated windows cannot inflate
either task or participant attack sample sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class BCI2ATrials:
    eeg: np.ndarray
    task: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray
    artifact_flag: np.ndarray


def outer_folds() -> list[dict[str, object]]:
    """Three participant-grouped outer folds; no participant crosses a fold boundary."""
    groups = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    return [
        {
            "fold": fold,
            "test_subjects": list(test),
            "validation_subjects": list(groups[(fold + 1) % 3]),
            "train_subjects": list(groups[(fold + 2) % 3]),
            "model_train_session": "T",
            "model_validation_session": "E (participant-disjoint)",
            "adaptive_attack_train_session": "T",
            "adaptive_attack_test_session": "E",
        }
        for fold, test in enumerate(groups)
    ]


def _pad_512(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] != 500:
        raise ValueError(f"expected 500 samples, got {x.shape[-1]}")
    return np.pad(x, ((0, 0), (0, 0), (6, 6)), mode="constant")


def load_bci2a_session(
    root: str | Path,
    subject: int,
    session: str,
    *,
    low_hz: float = 4.0,
    high_hz: float = 38.0,
    cue_offset_s: float = 2.5,
    duration_s: float = 2.0,
) -> BCI2ATrials:
    """Load 22-channel, 2-second post-cue trials from one official session.

    ``trial`` in the released MATLAB files marks trial onset.  The visual cue
    occurs at +2 s, hence +2.5 s selects 0.5--2.5 s after cue.  Filtering is
    performed within each continuous run before epoching.  EOG channels are
    never returned to the representation model.
    """
    if session not in {"T", "E"}:
        raise ValueError("session must be T or E")
    path = Path(root) / f"A{subject:02d}{session}.mat"
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    rows: list[np.ndarray] = []
    labels: list[int] = []
    trials: list[int] = []
    flags: list[int] = []
    global_trial = 0
    for run in np.atleast_1d(payload):
        positions = np.asarray(run.trial).reshape(-1)
        if positions.size == 0:
            continue
        fs = int(run.fs)
        if fs != 250:
            raise ValueError(f"{path.name}: expected 250 Hz, got {fs}")
        continuous = np.asarray(run.X, dtype=np.float64)[:, :22].T
        sos = butter(4, (low_hz, high_hz), btype="bandpass", fs=fs, output="sos")
        continuous = sosfiltfilt(sos, continuous, axis=-1)
        start_delta = int(round(cue_offset_s * fs))
        length = int(round(duration_s * fs))
        ys = np.asarray(run.y).reshape(-1).astype(int)
        artifact = np.asarray(run.artifacts).reshape(-1).astype(int)
        for position, label, flag in zip(positions, ys, artifact):
            # MATLAB indices are one-based.
            start = int(position) - 1 + start_delta
            stop = start + length
            if start < 0 or stop > continuous.shape[-1]:
                raise ValueError(f"{path.name}: trial {global_trial} is out of bounds")
            epoch = continuous[:, start:stop]
            mean = epoch.mean(axis=-1, keepdims=True)
            std = epoch.std(axis=-1, keepdims=True)
            epoch = (epoch - mean) / np.maximum(std, 1e-6)
            rows.append(epoch.astype(np.float32))
            labels.append(label - 1)
            trials.append(global_trial)
            flags.append(flag)
            global_trial += 1
    eeg = _pad_512(np.stack(rows))
    if eeg.shape != (288, 22, 512):
        raise ValueError(f"{path.name}: unexpected trial tensor {eeg.shape}")
    return BCI2ATrials(
        eeg=eeg.astype(np.float32),
        task=np.asarray(labels, dtype=np.int64),
        subject=np.full(288, subject - 1, dtype=np.int64),
        session=np.full(288, 0 if session == "T" else 1, dtype=np.int64),
        trial=np.asarray(trials, dtype=np.int64),
        artifact_flag=np.asarray(flags, dtype=np.int64),
    )


def concatenate(parts: list[BCI2ATrials]) -> BCI2ATrials:
    return BCI2ATrials(**{name: np.concatenate([getattr(p, name) for p in parts]) for name in BCI2ATrials.__dataclass_fields__})


__all__ = ["BCI2ATrials", "concatenate", "load_bci2a_session", "outer_folds"]
