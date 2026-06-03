"""EEGdenoiseNet paired-denoising data (handoff Phase 2 / M8; Zhang et al. 2021).

Builds single-channel ``(noisy, clean)`` pairs from clean EEG epochs + EOG/EMG artifact epochs:
``y = x + coe·n`` with ``coe = rms(x)/(rms(n)·SNR_lin)`` (eq. for SNR), then each epoch is divided by
``std(y)`` (matching the official ``data_prepare.py``). Training SNR is sampled uniform in [-7, 2] dB;
val/test use 10 fixed SNR levels. This is a faithful port of the official pipeline; the only change
is a seeded RNG for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EEGDenoiseConfig:
    data_dir: str = "EEGdenoiseNet/data"
    fs: int = 256
    segment_len: int = 512
    combin_num: int = 10
    train_per: float = 0.8
    snr_train_db: Tuple[float, float] = (-7.0, 2.0)
    snr_test_levels: int = 10
    psd_max_hz: int = 120
    seed: int = 42
    sdedit_t_star: int = 200
    ddim_steps: int = 50
    epochs: int = 60
    batch_size: int = 128
    lr: float = 1e-4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EEGDenoiseConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw["snr_train_db"] = tuple(raw["snr_train_db"])
        return cls(**raw)


@dataclass
class DenoisePairs:
    """Standardized (noisy, clean) pairs. Arrays are ``(N, L)`` float32; test is grouped by SNR."""

    noisy_train: np.ndarray
    clean_train: np.ndarray
    noisy_val: np.ndarray
    clean_val: np.ndarray
    noisy_test: np.ndarray
    clean_test: np.ndarray
    test_snr_db: np.ndarray  # (N_test_total,) SNR level (dB) of each test epoch
    snr_levels_db: np.ndarray  # (10,) the distinct test SNR levels


def get_rms(x: np.ndarray) -> np.ndarray:
    """Per-row RMS of a ``(N, L)`` array → ``(N,)``."""
    return np.sqrt(np.mean(x ** 2, axis=-1))


def _shuffle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return x[rng.permutation(x.shape[0])]


def _combine(x: np.ndarray, combin_num: int, rng: np.random.Generator) -> np.ndarray:
    """Stack ``combin_num`` independent shuffles of ``x`` → ``(combin_num*N, L)`` (augmentation)."""
    return np.concatenate([_shuffle(x, rng) for _ in range(combin_num)], axis=0)


def _mix_and_standardize(
    eeg: np.ndarray, noise: np.ndarray, snr_lin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Mix ``y = x + coe·n`` at the given per-epoch linear SNR and divide both by ``std(y)``."""
    coe = (get_rms(eeg) / (get_rms(noise) * snr_lin))[:, None]
    noisy = eeg + noise * coe
    std = noisy.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (noisy / std).astype(np.float32), (eeg / std).astype(np.float32)


def load_components(cfg: EEGDenoiseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load clean EEG, EOG and EMG epochs as ``(N, L)`` float64 arrays."""
    base = REPO_ROOT / cfg.data_dir
    eeg = np.load(base / "EEG_all_epochs.npy")
    eog = np.load(base / "EOG_all_epochs.npy")
    emg = np.load(base / "EMG_all_epochs.npy")
    return eeg, eog, emg


def prepare_pairs(noise_type: str, cfg: EEGDenoiseConfig) -> DenoisePairs:
    """Build train/val/test (noisy, clean) pairs for ``noise_type`` in {'EOG', 'EMG'}."""
    rng = np.random.default_rng(cfg.seed)
    eeg_all, eog_all, emg_all = load_components(cfg)
    noise_all = {"EOG": eog_all, "EMG": emg_all}[noise_type]

    eeg = _shuffle(eeg_all, rng)
    noise = _shuffle(noise_all, rng)
    if noise_type == "EMG":  # reuse EEG to match the larger EMG count
        reuse = noise.shape[0] - eeg.shape[0]
        eeg = np.vstack([eeg[:reuse], eeg]) if reuse > 0 else eeg[: noise.shape[0]]
    else:  # EOG: drop EEG down to the noise count
        eeg = eeg[: noise.shape[0]]

    n = eeg.shape[0]
    n_train = round(cfg.train_per * n)
    n_val = round((n - n_train) / 2)
    sl = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, n),
    }

    # --- train (augmented, random SNR) ---
    eeg_tr = _combine(eeg[sl["train"]], cfg.combin_num, rng)
    noise_tr = _combine(noise[sl["train"]], cfg.combin_num, rng)
    snr_tr = 10 ** (0.1 * rng.uniform(cfg.snr_train_db[0], cfg.snr_train_db[1], eeg_tr.shape[0]))
    noisy_train, clean_train = _mix_and_standardize(eeg_tr, noise_tr, snr_tr)

    # --- val / test (10 fixed SNR levels, stacked) ---
    levels_db = np.linspace(cfg.snr_train_db[0], cfg.snr_train_db[1], cfg.snr_test_levels)

    def fixed_levels(eeg_split: np.ndarray, noise_split: np.ndarray):
        noisy_blocks, clean_blocks, snr_tags = [], [], []
        for db in levels_db:
            snr_lin = np.full(eeg_split.shape[0], 10 ** (0.1 * db))
            ny, cl = _mix_and_standardize(eeg_split, noise_split, snr_lin)
            noisy_blocks.append(ny)
            clean_blocks.append(cl)
            snr_tags.append(np.full(eeg_split.shape[0], db))
        return (np.concatenate(noisy_blocks), np.concatenate(clean_blocks), np.concatenate(snr_tags))

    noisy_val, clean_val, _ = fixed_levels(eeg[sl["val"]], noise[sl["val"]])
    noisy_test, clean_test, test_snr = fixed_levels(eeg[sl["test"]], noise[sl["test"]])

    return DenoisePairs(
        noisy_train=noisy_train, clean_train=clean_train,
        noisy_val=noisy_val, clean_val=clean_val,
        noisy_test=noisy_test, clean_test=clean_test,
        test_snr_db=test_snr, snr_levels_db=levels_db,
    )
