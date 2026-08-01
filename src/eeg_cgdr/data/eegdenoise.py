"""Clean single-channel EEG prior data from the registered EEGdenoiseNet release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CleanPriorSplit:
    train: np.ndarray
    validation: np.ndarray
    mean: float
    standard_deviation: float
    sampling_rate: int = 256


def load_clean_prior_split(
    path: str | Path,
    *,
    validation_fraction: float,
    seed: int,
) -> CleanPriorSplit:
    """Load all real clean epochs and make a deterministic development split.

    EEGdenoiseNet does not provide participant identifiers.  It is therefore
    used only as an external clean-prior source, never for participant-specific
    transfer claims.  The held-out scientific datasets contain different
    participants.
    """
    epochs = np.asarray(np.load(Path(path), mmap_mode="r", allow_pickle=False), dtype=np.float32)
    if epochs.ndim != 2 or epochs.shape[1] != 512 or epochs.shape[0] < 1000:
        raise ValueError(f"unexpected EEGdenoiseNet clean array shape: {epochs.shape}")
    if not np.isfinite(epochs).all():
        raise ValueError("EEGdenoiseNet clean array contains non-finite values")
    if not 0.01 <= validation_fraction <= 0.25:
        raise ValueError("validation_fraction must be in [0.01, 0.25]")
    order = np.random.default_rng(seed).permutation(epochs.shape[0])
    validation_count = max(1, int(round(validation_fraction * epochs.shape[0])))
    validation_index = order[:validation_count]
    train_index = order[validation_count:]
    train_raw = epochs[train_index]
    mean = float(train_raw.mean(dtype=np.float64))
    standard_deviation = float(train_raw.std(dtype=np.float64))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-8:
        raise ValueError("invalid population clean-EEG scale")
    train = ((train_raw - mean) / standard_deviation).astype(np.float32, copy=False)
    validation = ((epochs[validation_index] - mean) / standard_deviation).astype(
        np.float32, copy=False
    )
    return CleanPriorSplit(train, validation, mean, standard_deviation)


def normalize_with_clean_prior_statistics(
    signal: np.ndarray, *, mean: float, standard_deviation: float
) -> np.ndarray:
    if standard_deviation <= 1e-8:
        raise ValueError("invalid fixed standard deviation")
    return ((np.asarray(signal) - mean) / standard_deviation).astype(np.float32)


def denormalize_with_clean_prior_statistics(
    signal: np.ndarray, *, mean: float, standard_deviation: float
) -> np.ndarray:
    return (np.asarray(signal) * standard_deviation + mean).astype(np.float32)
