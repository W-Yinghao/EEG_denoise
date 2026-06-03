"""Unit tests for evaluation helpers: matrix summary + downstream EEGNet API (M6/M7)."""

from __future__ import annotations

import numpy as np
import torch

from saddpm.eval.downstream import downstream_accuracy, evaluate_eegnet, train_eegnet
from saddpm.eval.pairwise import matrix_summary
from saddpm.models.eegnet import EEGNetConfig


def test_matrix_summary() -> None:
    mat = np.array([[0.8, 0.4], [0.6, 0.9]])
    s = matrix_summary(mat)
    assert abs(s["grand_mean"] - 0.675) < 1e-6
    assert abs(s["diag_mean"] - 0.85) < 1e-6
    assert len(s["source_mean_row"]) == 2


def test_downstream_api_returns_valid_accuracy() -> None:
    cfg = EEGNetConfig(n_channels=22, n_times=512, epochs=2, batch_size=16)
    rng = np.random.default_rng(0)
    tr_x = rng.normal(size=(32, 22, 512)).astype(np.float32)
    tr_y = rng.integers(0, cfg.n_classes, size=32)
    te_x = rng.normal(size=(16, 22, 512)).astype(np.float32)
    te_y = rng.integers(0, cfg.n_classes, size=16)
    acc = downstream_accuracy(tr_x, tr_y, te_x, te_y, cfg, torch.device("cpu"), seed=0)
    assert 0.0 <= acc <= 1.0


def test_eegnet_can_separate_toy_classes() -> None:
    """A class-dependent additive pattern should be learnable above chance."""
    cfg = EEGNetConfig(n_channels=22, n_times=512, epochs=8, batch_size=16, n_classes=2)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.5, size=(80, 22, 512)).astype(np.float32)
    y = rng.integers(0, 2, size=80)
    x[y == 1, :, :] += 1.0  # class-1 windows have a DC offset
    model = train_eegnet(x, y, cfg, torch.device("cpu"), seed=0)
    acc = evaluate_eegnet(model, x, y, torch.device("cpu"))
    assert acc > 0.6  # well above chance (0.5)
