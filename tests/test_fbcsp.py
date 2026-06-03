"""Unit test for FBCSP + LDA (M6/[DD-5] alternative classifier)."""

from __future__ import annotations

import numpy as np

from saddpm.baselines.fbcsp import FBCSPConfig, FBCSPLDA


def test_fbcsp_separates_spatial_classes() -> None:
    """Two classes with band-power in different channel groups should be CSP-separable."""
    rng = np.random.default_rng(0)
    n, c, t, sfreq = 40, 8, 256, 250
    x = rng.normal(scale=0.5, size=(2 * n, c, t)).astype(np.float32)
    y = np.array([0] * n + [1] * n)
    osc = np.sin(2 * np.pi * 10.0 * np.arange(t) / sfreq).astype(np.float32)
    x[:n, 0:3] += 2.0 * osc   # class 0: 10 Hz power in channels 0-2
    x[n:, 5:8] += 2.0 * osc   # class 1: 10 Hz power in channels 5-7

    cfg = FBCSPConfig(sfreq=sfreq, bands=[(8, 12)], n_components=4)
    tr = np.r_[0:30, n:n + 30]
    te = np.r_[30:n, n + 30:2 * n]
    clf = FBCSPLDA(cfg).fit(x[tr], y[tr])
    acc = clf.score(x[te], y[te])
    assert acc > 0.7, f"FBCSP failed to separate CSP-distinct classes: acc={acc:.3f}"
