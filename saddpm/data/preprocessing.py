"""Array-level preprocessing: sliding windows, per-channel z-score, pad/crop to U-Net length.

These functions are pure NumPy (no MNE/MOABB) so they are fast to unit-test. MNE-level
filtering lives in :mod:`saddpm.data.bcic2a`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sliding_windows(
    epochs: np.ndarray,
    length: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut fixed-length sliding windows out of each trial epoch.

    Args:
        epochs: array of shape ``(n_trials, n_channels, n_times)``.
        length: window length in samples (e.g. 500 for 2 s @ 250 Hz).
        step: hop between consecutive window starts in samples (e.g. 125 for 0.5 s).

    Returns:
        A tuple ``(windows, trial_index)`` where ``windows`` has shape
        ``(n_windows, n_channels, length)`` and ``trial_index`` has shape ``(n_windows,)``
        mapping each window back to its source trial (so MI labels can be propagated).

    Raises:
        ValueError: if ``epochs`` is not 3-D or a single window does not fit.
    """
    if epochs.ndim != 3:
        raise ValueError(f"epochs must be (n_trials, n_channels, n_times); got {epochs.shape}")
    n_trials, _, n_times = epochs.shape
    if length > n_times:
        raise ValueError(f"window length {length} exceeds trial length {n_times}")
    if step <= 0:
        raise ValueError(f"step must be positive; got {step}")

    starts = list(range(0, n_times - length + 1, step))
    windows = np.stack(
        [epochs[:, :, s : s + length] for s in starts], axis=1
    )  # (n_trials, n_starts, C, length)
    n_starts = len(starts)
    windows = windows.reshape(n_trials * n_starts, *windows.shape[2:])
    trial_index = np.repeat(np.arange(n_trials), n_starts)
    return windows, trial_index


def zscore_per_channel(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score (zero mean, unit variance) computed within each window.

    Args:
        windows: array of shape ``(..., n_channels, length)``.
        eps: small constant to avoid division by zero for flat channels.

    Returns:
        Z-scored array of the same shape and dtype ``float32``.
    """
    mean = windows.mean(axis=-1, keepdims=True)
    std = windows.std(axis=-1, keepdims=True)
    return ((windows - mean) / (std + eps)).astype(np.float32)


def pad_time(windows: np.ndarray, target: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Symmetrically zero-pad the time axis up to ``target`` samples ([DD-6]).

    Args:
        windows: array of shape ``(..., length)`` with ``length <= target``.
        target: desired length after padding (e.g. 512).

    Returns:
        ``(padded, (left, right))`` — the padded array and the pad amounts so the padding
        can be removed from the model output with :func:`unpad_time`.

    Raises:
        ValueError: if the current length exceeds ``target`` (cropping is intentionally not
            silent; callers must choose an epoch/window that fits).
    """
    length = windows.shape[-1]
    if length > target:
        raise ValueError(f"length {length} exceeds pad target {target}; refusing to crop silently")
    total = target - length
    left = total // 2
    right = total - left
    pad_width = [(0, 0)] * (windows.ndim - 1) + [(left, right)]
    padded = np.pad(windows, pad_width, mode="constant", constant_values=0.0)
    return padded, (left, right)


def unpad_time(windows: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`pad_time`: remove the recorded ``(left, right)`` padding."""
    left, right = pad
    end = windows.shape[-1] - right
    return windows[..., left:end]
