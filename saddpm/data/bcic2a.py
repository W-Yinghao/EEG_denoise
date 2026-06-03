"""Load and preprocess BCI Competition IV-2a (MOABB ``BNCI2014_001``).

Pipeline (handoff §3), per subject & session:
    raw runs -> concatenate -> pick 22 EEG -> band-pass 1-50 Hz FIR -> notch 50 Hz ->
    resample 250 Hz -> epoch the MI interval per trial -> sliding 2 s / 0.5 s windows ->
    per-channel z-score -> pad 500 -> 512 ([DD-6]).

MNE/MOABB are imported lazily inside functions so that pure-array utilities and unit tests do
not require those heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DataConfig
from .preprocessing import pad_time, sliding_windows, zscore_per_channel


@dataclass
class SubjectWindows:
    """Preprocessed sliding windows for one subject/session.

    Attributes:
        subject: 1-based subject id (1..9).
        session: MOABB session key (e.g. ``"0train"`` / ``"1test"``).
        session_role: ``"T"`` (train) or ``"E"`` (evaluation), assigned by session order.
        windows: ``(n_windows, n_eeg_channels, pad_to)`` float32 array, z-scored + padded.
        mi_labels: ``(n_windows,)`` int array of 4-class MI labels (0..3).
        trial_index: ``(n_windows,)`` int array mapping each window to its source trial.
        subject_id: ``(n_windows,)`` int array, all equal to ``subject - 1`` (0-based for embeddings).
        sfreq: sampling rate after resampling.
        pad: ``(left, right)`` zero-padding applied to reach ``pad_to`` (to undo on model output).
        ch_names: the 22 EEG channel names, in order.
        class_names: MI class names indexed by label.
    """

    subject: int
    session: str
    session_role: str
    windows: np.ndarray
    mi_labels: np.ndarray
    trial_index: np.ndarray
    subject_id: np.ndarray
    sfreq: int
    pad: Tuple[int, int]
    ch_names: List[str]
    class_names: List[str]

    def summary(self) -> str:
        """One-line human-readable shape/label summary."""
        counts = np.bincount(self.mi_labels, minlength=len(self.class_names))
        per_class = ", ".join(f"{n}:{c}" for n, c in zip(self.class_names, counts))
        return (
            f"subject A{self.subject:02d} session={self.session} ({self.session_role}) "
            f"windows={self.windows.shape} labels[{per_class}] sfreq={self.sfreq} pad={self.pad}"
        )


def _set_mne_data_path(cfg: DataConfig) -> None:
    """Point MNE/MOABB at an explicit data root if configured."""
    if cfg.mne_data_path:
        import mne

        mne.set_config("MNE_DATA", cfg.mne_data_path, set_env=True)
        mne.set_config("MNE_DATASETS_BNCI_PATH", cfg.mne_data_path, set_env=True)


def _build_dataset():
    """Instantiate the MOABB BNCI2014_001 dataset object (this *is* BCI-IV-2a)."""
    from moabb.datasets import BNCI2014_001

    return BNCI2014_001()


def _events_from_raw(raw, event_id: Dict[str, int]):
    """Extract (events, event_id) for the 4 MI classes, robust to annotation naming."""
    import mne

    try:
        events, used = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        if len(events) > 0:
            return events, used
    except (ValueError, KeyError):
        pass
    # Fallback: auto-discover annotation descriptions, then keep those matching the class names.
    events, auto = mne.events_from_annotations(raw, verbose=False)
    keep = {name: code for name, code in auto.items() if name in event_id}
    if keep:
        wanted = set(keep.values())
        events = events[np.isin(events[:, 2], list(wanted))]
        return events, keep
    return events, auto


def preprocess_session_raws(raws: Dict[str, "object"], cfg: DataConfig):
    """Concatenate a session's runs and apply §3 filtering/resampling to the EEG channels.

    Args:
        raws: mapping ``run_key -> mne.io.Raw`` for a single session.
        cfg: data configuration.

    Returns:
        A single preprocessed ``mne.io.Raw`` restricted to the 22 EEG channels.
    """
    import mne

    ordered = [raws[k] for k in sorted(raws.keys())]
    raw = mne.concatenate_raws([r.copy() for r in ordered], verbose=False)
    raw.pick("eeg", verbose=False)  # 22 EEG channels; drops EOG/stim for SADDPM

    pp = cfg.preprocess
    if raw.info["sfreq"] != pp.resample_hz:
        raw.resample(pp.resample_hz, verbose=False)
    raw.filter(
        l_freq=pp.bandpass_low_hz,
        h_freq=pp.bandpass_high_hz,
        method=pp.filter_method,
        fir_design=pp.fir_design,
        verbose=False,
    )
    raw.notch_filter(freqs=pp.notch_hz, method=pp.filter_method, verbose=False)
    return raw


def _epoch_trials(raw, event_id: Dict[str, int], cfg: DataConfig):
    """Epoch the MI interval per trial; returns ``(data [n_trials, C, T], labels, class_names)``."""
    import mne

    events, used = _events_from_raw(raw, event_id)
    # Stable class ordering by event code -> contiguous 0-based labels.
    code_to_name = {code: name for name, code in used.items()}
    sorted_codes = sorted(code_to_name)
    class_names = [code_to_name[c] for c in sorted_codes]
    code_to_label = {c: i for i, c in enumerate(sorted_codes)}

    epochs = mne.Epochs(
        raw,
        events,
        event_id=used,
        tmin=cfg.epoch.tmin_s,
        tmax=cfg.epoch.tmax_s,
        baseline=None,
        preload=True,
        verbose=False,
    )
    data = epochs.get_data(copy=False)  # (n_trials, C, T)
    labels = np.array([code_to_label[c] for c in epochs.events[:, 2]], dtype=np.int64)
    return data, labels, class_names


def _windowize(
    data: np.ndarray, labels: np.ndarray, cfg: DataConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Slide windows over trial epochs, z-score per channel, pad to ``pad_to``."""
    sfreq = cfg.preprocess.resample_hz
    length = cfg.window.length_samples(sfreq)
    step = cfg.window.step_samples(sfreq)

    windows, trial_index = sliding_windows(data, length=length, step=step)
    win_labels = labels[trial_index]
    if cfg.window.zscore_per_channel:
        windows = zscore_per_channel(windows)
    windows, pad = pad_time(windows, cfg.window.pad_to)
    return windows.astype(np.float32), win_labels, trial_index, pad


def epoch_and_window(
    raw_eeg: "object",
    event_id: Dict[str, int],
    cfg: DataConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int], List[str]]:
    """Epoch the MI interval of a (preprocessed, EEG-only) raw and slide windows over it.

    Shared by :func:`load_subject` and the ICA baseline so both produce windows the identical way.

    Returns:
        ``(windows, mi_labels, trial_index, pad, class_names)``.
    """
    data, labels, class_names = _epoch_trials(raw_eeg, event_id, cfg)
    windows, win_labels, trial_index, pad = _windowize(data, labels, cfg)
    return windows, win_labels, trial_index, pad, class_names


def load_subject(
    subject: int,
    cfg: DataConfig,
    sessions: Optional[List[str]] = None,
) -> Dict[str, SubjectWindows]:
    """Load and fully preprocess one subject into sliding windows, per session.

    Args:
        subject: 1-based subject id (1..9).
        cfg: data configuration.
        sessions: optional subset of MOABB session keys to keep (default: all).

    Returns:
        Mapping ``session_key -> SubjectWindows``.
    """
    _set_mne_data_path(cfg)
    dataset = _build_dataset()
    event_id = dict(dataset.event_id)
    raw_data = dataset.get_data(subjects=[subject])[subject]

    session_keys = sorted(raw_data.keys())
    roles = {key: ("T" if i == 0 else "E") for i, key in enumerate(session_keys)}

    out: Dict[str, SubjectWindows] = {}
    for session_key in session_keys:
        if sessions is not None and session_key not in sessions:
            continue
        raw = preprocess_session_raws(raw_data[session_key], cfg)
        ch_names = list(raw.ch_names)
        windows, win_labels, trial_index, pad, class_names = epoch_and_window(raw, event_id, cfg)
        out[session_key] = SubjectWindows(
            subject=subject,
            session=session_key,
            session_role=roles[session_key],
            windows=windows,
            mi_labels=win_labels,
            trial_index=trial_index,
            subject_id=np.full(len(windows), subject - 1, dtype=np.int64),
            sfreq=cfg.preprocess.resample_hz,
            pad=pad,
            ch_names=ch_names,
            class_names=class_names,
        )
    return out
