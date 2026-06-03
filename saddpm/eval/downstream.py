"""Downstream 4-class classification with EEGNet (handoff §8.1).

Trains EEGNet on (denoised) source Session-T windows and evaluates on (denoised) target
Session-E windows. The same routine + config is used for both the ICA and SADDPM pipelines so the
9×9 accuracy comparison is fair.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..models.eegnet import EEGNet, EEGNetConfig


def train_eegnet(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    cfg: EEGNetConfig,
    device: torch.device,
    seed: int = 42,
) -> EEGNet:
    """Train an EEGNet classifier on labelled windows.

    Args:
        train_windows: ``(N, C, T)`` float array.
        train_labels: ``(N,)`` 4-class labels.
        cfg: EEGNet config (architecture + optimisation).
        device: training device.
        seed: RNG seed for weight init / shuffling.

    Returns:
        The trained :class:`EEGNet`.
    """
    torch.manual_seed(seed)
    model = EEGNet(cfg).to(device)
    x = torch.from_numpy(np.ascontiguousarray(train_windows)).float()
    y = torch.from_numpy(np.ascontiguousarray(train_labels)).long()
    loader = DataLoader(
        TensorDataset(x, y), batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_eegnet(
    model: EEGNet,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Classification accuracy of ``model`` on labelled test windows."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(test_windows)).float()
    y = torch.from_numpy(np.ascontiguousarray(test_labels)).long()
    correct = 0
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device)
        pred = model(xb).argmax(dim=1).cpu()
        correct += int((pred == y[i : i + batch_size]).sum().item())
    return correct / max(len(x), 1)


def downstream_accuracy(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    test_windows: np.ndarray,
    test_labels: np.ndarray,
    cfg: EEGNetConfig,
    device: torch.device,
    seed: int = 42,
) -> float:
    """Train EEGNet on (train_windows, train_labels) and return accuracy on the test set."""
    model = train_eegnet(train_windows, train_labels, cfg, device, seed)
    return evaluate_eegnet(model, test_windows, test_labels, device)
