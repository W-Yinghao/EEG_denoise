"""Unit tests for array-level preprocessing and the data config (no MOABB needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from saddpm.data.config import DataConfig
from saddpm.data.preprocessing import (
    pad_time,
    sliding_windows,
    unpad_time,
    zscore_per_channel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sliding_windows_shapes_and_count() -> None:
    # 4 trials, 22 channels, 1001 samples (the MI epoch length @ 250 Hz, tmax inclusive).
    epochs = np.random.randn(4, 22, 1001).astype(np.float32)
    windows, trial_index = sliding_windows(epochs, length=500, step=125)
    # starts at 0,125,250,375,500 -> 5 windows per trial.
    assert windows.shape == (4 * 5, 22, 500)
    assert trial_index.shape == (20,)
    assert trial_index.tolist() == sum(([t] * 5 for t in range(4)), [])


def test_sliding_windows_content() -> None:
    epochs = np.arange(1 * 1 * 10, dtype=np.float32).reshape(1, 1, 10)
    windows, _ = sliding_windows(epochs, length=4, step=2)
    # starts 0,2,4,6 -> 4 windows.
    assert windows.shape == (4, 1, 4)
    assert windows[0, 0].tolist() == [0, 1, 2, 3]
    assert windows[1, 0].tolist() == [2, 3, 4, 5]


def test_sliding_windows_rejects_too_long() -> None:
    epochs = np.zeros((2, 3, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        sliding_windows(epochs, length=200, step=10)


def test_zscore_per_channel() -> None:
    x = np.random.randn(7, 22, 500).astype(np.float32) * 3.0 + 5.0
    z = zscore_per_channel(x)
    assert z.shape == x.shape
    assert z.dtype == np.float32
    np.testing.assert_allclose(z.mean(axis=-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=-1), 1.0, atol=1e-3)


def test_zscore_flat_channel_is_safe() -> None:
    x = np.ones((1, 1, 50), dtype=np.float32)
    z = zscore_per_channel(x)
    assert np.all(np.isfinite(z))


def test_pad_unpad_roundtrip() -> None:
    x = np.random.randn(3, 22, 500).astype(np.float32)
    padded, pad = pad_time(x, target=512)
    assert padded.shape == (3, 22, 512)
    assert pad == (6, 6)
    # padded edges are exactly zero.
    assert np.all(padded[..., :6] == 0.0)
    assert np.all(padded[..., -6:] == 0.0)
    np.testing.assert_array_equal(unpad_time(padded, pad), x)


def test_pad_refuses_to_crop() -> None:
    x = np.zeros((1, 1, 600), dtype=np.float32)
    with pytest.raises(ValueError):
        pad_time(x, target=512)


def test_config_loads_and_derives_window_samples() -> None:
    cfg = DataConfig.from_yaml(REPO_ROOT / "configs" / "data.yaml")
    assert cfg.dataset.name == "BNCI2014_001"
    assert cfg.dataset.n_subjects == 9
    sfreq = cfg.preprocess.resample_hz
    assert cfg.window.length_samples(sfreq) == 500
    assert cfg.window.step_samples(sfreq) == 125
    assert cfg.window.pad_to == 512
