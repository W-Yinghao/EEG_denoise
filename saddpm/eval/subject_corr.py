"""Subject-correlation analysis (handoff §8.2, Table 3 analogue, [DD-11]).

A per-subject descriptor is the trial-mean window flattened over channels×time; subject similarity
is the Pearson correlation between descriptors. Panel (a) correlates real subjects with each other;
panel (b) correlates SADDPM-generated samples of subject ``i`` against real subject ``j`` — the
diagonal should dominate if subject identity is preserved.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def trial_mean_descriptor(windows: np.ndarray) -> np.ndarray:
    """Mean window over the sample axis, flattened to a ``(C*L,)`` descriptor ([DD-11] literal).

    NOTE: for per-window z-scored data the trial mean is near zero and uninformative; prefer
    :func:`spectral_descriptor`. Kept for the literal [DD-11] option / ablations.

    Args:
        windows: ``(N, C, L)`` array of windows for one subject.

    Returns:
        ``(C*L,)`` float descriptor.
    """
    return windows.mean(axis=0).reshape(-1)


def spectral_descriptor(windows: np.ndarray) -> np.ndarray:
    """Per-channel log power spectrum averaged over windows, flattened to ``(C*F,)``.

    Survives per-window z-scoring (which kills the trial mean) by capturing each subject's
    spectral *shape*. This is the default subject descriptor (justified deviation from the
    [DD-11] trial-mean default; logged in RESULTS.md).

    Args:
        windows: ``(N, C, L)`` array of windows for one subject.

    Returns:
        ``(C*F,)`` float descriptor where ``F = L//2 + 1``.
    """
    power = (np.abs(np.fft.rfft(windows, axis=-1)) ** 2).mean(axis=0)  # (C, F)
    return np.log1p(power).reshape(-1)


def pearson_matrix(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of ``desc_a`` and every row of ``desc_b``.

    Args:
        desc_a: ``(n, D)`` descriptors.
        desc_b: ``(m, D)`` descriptors.

    Returns:
        ``(n, m)`` correlation matrix.
    """
    a = desc_a - desc_a.mean(axis=1, keepdims=True)
    b = desc_b - desc_b.mean(axis=1, keepdims=True)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def descriptors_from_groups(
    groups: Sequence[np.ndarray],
    descriptor: Callable[[np.ndarray], np.ndarray] = spectral_descriptor,
) -> np.ndarray:
    """Stack per-subject descriptors for a list of window arrays into ``(S, D)``."""
    return np.stack([descriptor(g) for g in groups], axis=0)


def diagonal_dominance(corr: np.ndarray) -> float:
    """Fraction of rows whose maximum correlation is on the diagonal (subject identity preserved)."""
    return float(np.mean(np.argmax(corr, axis=1) == np.arange(corr.shape[0])))
