"""Synthetic paired (corrupted, clean) multi-channel BCI data (Option C / M9).

Borrows EEGdenoiseNet's mixing recipe (``y = x + coe·n`` at a controlled SNR) but applies it to the
22-channel BCI-IV-2a windows: a single-channel EEGdenoiseNet EOG/EMG artifact is projected across
channels with a plausible spatial topography (ocular = frontal-dominant, myogenic = lateral-dominant).
This yields ground-truth pairs *with subject labels*, so a conditional denoiser can be trained
supervised and the subject embedding becomes load-bearing (the clean target is subject-specific).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import DataConfig
from .datasets import load_session_windows

# Relative artifact amplitude per channel (by name). Ocular artifacts dominate frontally;
# myogenic artifacts dominate at lateral/peripheral sites. Values are a modeling choice (logged).
_FRONTAL = {"Fz", "FC3", "FC1", "FCz", "FC2", "FC4"}
_CENTRAL = {"C5", "C3", "C1", "Cz", "C2", "C4", "C6"}
_CENTRO_PARIETAL = {"CP3", "CP1", "CPz", "CP2", "CP4"}
_LATERAL = {"C5", "C6", "FC3", "FC4", "CP3", "CP4", "P1", "P2"}


def eog_topography(ch_names: Sequence[str]) -> np.ndarray:
    """Frontal-dominant ocular topography (``(C,)``, ~unit max)."""
    w = []
    for ch in ch_names:
        if ch in _FRONTAL:
            w.append(1.0)
        elif ch in _CENTRAL:
            w.append(0.35)
        elif ch in _CENTRO_PARIETAL:
            w.append(0.15)
        else:  # parietal/occipital
            w.append(0.05)
    return np.asarray(w, dtype=np.float64)


def emg_topography(ch_names: Sequence[str]) -> np.ndarray:
    """Lateral/peripheral-dominant myogenic topography (``(C,)``)."""
    return np.asarray([1.0 if ch in _LATERAL else 0.45 for ch in ch_names], dtype=np.float64)


def _get_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


@dataclass
class SyntheticPairs:
    """Paired data: ``(N, C, L)`` corrupted/clean float32 + ``(N,)`` 0-based subject ids + SNR (dB)."""

    corrupted: np.ndarray
    clean: np.ndarray
    subject_ids: np.ndarray
    snr_db: np.ndarray
    ch_names: List[str]


def build_synthetic_pairs(
    subjects: Sequence[int],
    data_cfg: DataConfig,
    artifact_epochs: np.ndarray,
    noise_type: str,
    session_role: str,
    snr_db_range: Tuple[float, float],
    seed: int,
    fixed_snr_levels: Sequence[float] | None = None,
) -> SyntheticPairs:
    """Inject single-channel artifacts into multi-channel BCI windows at controlled SNR.

    Args:
        subjects: 1-based subject ids.
        data_cfg: BCI data config.
        artifact_epochs: ``(M, L)`` single-channel EEGdenoiseNet EOG/EMG epochs.
        noise_type: ``"EOG"`` or ``"EMG"`` (selects the topography).
        session_role: ``"T"`` or ``"E"``.
        snr_db_range: (low, high) dB for random training SNR.
        seed: RNG seed.
        fixed_snr_levels: if given (e.g. test), cycle these dB levels instead of random SNR.

    Returns:
        A :class:`SyntheticPairs`.
    """
    rng = np.random.default_rng(seed)
    corr_list, clean_list, sid_list, snr_list = [], [], [], []
    ch_names: List[str] = []
    art_len = artifact_epochs.shape[1]

    for s in subjects:
        sw = load_session_windows(s, data_cfg, session_role)
        ch_names = sw.ch_names
        topo = (eog_topography if noise_type == "EOG" else emg_topography)(ch_names)[:, None]
        clean = sw.windows.astype(np.float64)  # (N, C, L), z-scored
        n, c, length = clean.shape
        # match artifact length to the window length (crop/tile).
        art_idx = rng.integers(0, artifact_epochs.shape[0], size=n)
        art = artifact_epochs[art_idx]  # (N, art_len)
        if art_len >= length:
            art = art[:, :length]
        else:
            art = np.pad(art, ((0, 0), (0, length - art_len)), mode="wrap")

        if fixed_snr_levels is None:
            snr_db = rng.uniform(snr_db_range[0], snr_db_range[1], size=n)
        else:
            snr_db = np.array([fixed_snr_levels[i % len(fixed_snr_levels)] for i in range(n)], dtype=float)
        snr_lin = 10 ** (0.1 * snr_db)

        mc_art = topo[None] * art[:, None, :]  # (N, C, L) spatially-projected artifact
        # scale the whole multi-channel artifact so the per-window power SNR matches snr_lin.
        rms_clean = np.sqrt((clean ** 2).mean(axis=(1, 2)))  # (N,)
        rms_art = np.sqrt((mc_art ** 2).mean(axis=(1, 2))) + 1e-12
        coe = (rms_clean / (rms_art * snr_lin))[:, None, None]
        corrupted = clean + coe * mc_art
        # per-window, per-channel z-score (same normalized space as the clean windows / downstream).
        std = corrupted.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        corrupted = corrupted / std
        clean_n = clean / std

        corr_list.append(corrupted.astype(np.float32))
        clean_list.append(clean_n.astype(np.float32))
        sid_list.append(np.full(n, s - 1, dtype=np.int64))
        snr_list.append(snr_db)

    return SyntheticPairs(
        corrupted=np.concatenate(corr_list),
        clean=np.concatenate(clean_list),
        subject_ids=np.concatenate(sid_list),
        snr_db=np.concatenate(snr_list),
        ch_names=ch_names,
    )
