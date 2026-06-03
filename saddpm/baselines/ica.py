"""ICA denoising baseline (handoff §8.1).

Fit ICA on the 22 EEG channels, identify EOG-correlated components by correlation with the 3 EOG
channels (``find_bads_eog``), zero them, reconstruct, then epoch + window exactly like the SADDPM
pipeline so the downstream EEGNet comparison is fair. Results are cached on disk for reuse by the
9×9 sweep (M7).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml

from ..data.bcic2a import _set_mne_data_path, epoch_and_window, preprocess_session_raws
from ..data.cache import DEFAULT_CACHE_DIR, config_hash
from ..data.config import DataConfig


@dataclass(frozen=True)
class ICAConfig:
    method: str = "infomax"
    n_components: int = 20
    random_state: int = 42
    max_iter: int = 500

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ICAConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**yaml.safe_load(fh))


@dataclass
class ICADenoised:
    """ICA-denoised windows for one subject/session (same fields as the SADDPM pipeline needs)."""

    windows: np.ndarray
    mi_labels: np.ndarray
    n_excluded: int


def _ica_hash(data_cfg: DataConfig, ica_cfg: ICAConfig) -> str:
    payload = json.dumps({"data": config_hash(data_cfg), "ica": ica_cfg.__dict__}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]


def _load_session_raw_with_eog(subject: int, session_role: str, data_cfg: DataConfig):
    """Load + filter a session's raw keeping BOTH EEG and EOG channels."""
    import mne
    from moabb.datasets import BNCI2014_001

    _set_mne_data_path(data_cfg)
    dataset = BNCI2014_001()
    raw_data = dataset.get_data(subjects=[subject])[subject]
    keys = sorted(raw_data.keys())
    role_to_key = {("T" if i == 0 else "E"): k for i, k in enumerate(keys)}
    runs = raw_data[role_to_key[session_role]]

    ordered = [runs[k].copy() for k in sorted(runs.keys())]
    raw = mne.concatenate_raws(ordered, verbose=False)
    raw.pick(["eeg", "eog"], verbose=False)
    pp = data_cfg.preprocess
    if raw.info["sfreq"] != pp.resample_hz:
        raw.resample(pp.resample_hz, verbose=False)
    raw.filter(l_freq=pp.bandpass_low_hz, h_freq=pp.bandpass_high_hz,
               method=pp.filter_method, fir_design=pp.fir_design, verbose=False)
    raw.notch_filter(freqs=pp.notch_hz, method=pp.filter_method, verbose=False)
    return raw, dict(dataset.event_id)


def ica_denoise_session(
    subject: int,
    session_role: str,
    data_cfg: DataConfig,
    ica_cfg: ICAConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> ICADenoised:
    """ICA-denoise one subject/session and return windows + labels (cached on disk)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"ica_A{subject:02d}_{session_role}_{_ica_hash(data_cfg, ica_cfg)}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)

    import mne

    raw, event_id = _load_session_raw_with_eog(subject, session_role, data_cfg)
    eog_names = raw.copy().pick("eog", verbose=False).ch_names
    raw_eeg = raw.copy().pick("eeg", verbose=False)

    fit_params = {"extended": True} if ica_cfg.method == "infomax" else None
    ica = mne.preprocessing.ICA(
        n_components=ica_cfg.n_components, method=ica_cfg.method,
        random_state=ica_cfg.random_state, max_iter=ica_cfg.max_iter, fit_params=fit_params,
    )
    ica.fit(raw_eeg, verbose=False)
    eog_idx, _ = ica.find_bads_eog(raw, ch_name=eog_names, verbose=False)
    ica.exclude = eog_idx
    clean = raw_eeg.copy()
    ica.apply(clean, verbose=False)

    windows, mi_labels, _trial_index, _pad, _classes = epoch_and_window(clean, event_id, data_cfg)
    result = ICADenoised(windows=windows, mi_labels=mi_labels, n_excluded=len(eog_idx))
    torch.save(result, path)
    return result
