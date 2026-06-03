"""FBCSP + LDA downstream classifier (handoff [DD-5], classic alternative to EEGNet).

Filter-Bank Common Spatial Patterns: band-pass the windows into a filter bank, fit a (multiclass,
one-vs-rest) CSP per band via MNE, take log-variance features, concatenate, and classify with LDA.
Drop-in alternative to EEGNet — used on the SAME denoised windows for a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class FBCSPConfig:
    sfreq: int = 250
    bands: List[Tuple[int, int]] = field(
        default_factory=lambda: [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
                                 (24, 28), (28, 32), (32, 36), (36, 40)]
    )
    n_components: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FBCSPConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(
            sfreq=int(raw["sfreq"]),
            bands=[tuple(b) for b in raw["bands"]],
            n_components=int(raw["n_components"]),
        )


class FBCSPLDA:
    """Filter-Bank CSP feature extractor + LDA classifier."""

    def __init__(self, cfg: FBCSPConfig) -> None:
        self.cfg = cfg
        self._csps: list = []
        self._lda = None

    def _bandpass(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
        import mne

        return mne.filter.filter_data(
            x.astype(np.float64), self.cfg.sfreq, low, high, method="iir", verbose=False
        )

    def _features(self, x: np.ndarray, fit: bool, y: np.ndarray | None = None) -> np.ndarray:
        from mne.decoding import CSP

        feats = []
        for k, (low, high) in enumerate(self.cfg.bands):
            xb = self._bandpass(x, low, high)
            if fit:
                csp = CSP(n_components=self.cfg.n_components, reg="ledoit_wolf", log=True, norm_trace=False)
                csp.fit(xb, y)
                self._csps.append(csp)
            feats.append(self._csps[k].transform(xb))
        return np.concatenate(feats, axis=1)

    def fit(self, windows: np.ndarray, labels: np.ndarray) -> "FBCSPLDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self._csps = []
        feats = self._features(windows, fit=True, y=labels)
        self._lda = LinearDiscriminantAnalysis()
        self._lda.fit(feats, labels)
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return self._lda.predict(self._features(windows, fit=False))

    def score(self, windows: np.ndarray, labels: np.ndarray) -> float:
        return float((self.predict(windows) == labels).mean())


def downstream_accuracy_fbcsp(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    cfg: FBCSPConfig,
) -> float:
    """Fit FBCSP+LDA on the training windows and return accuracy on the test set."""
    clf = FBCSPLDA(cfg).fit(train_windows, train_labels)
    return clf.score(test_windows, test_labels)
