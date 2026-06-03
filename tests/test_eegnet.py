"""Unit tests for the EEGNet-8,2 downstream classifier (M6, [DD-5])."""

from __future__ import annotations

from pathlib import Path

import torch

from saddpm.models.eegnet import EEGNet, EEGNetConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_eegnet_forward_shapes() -> None:
    cfg = EEGNetConfig.from_yaml(REPO_ROOT / "configs" / "eegnet.yaml")
    model = EEGNet(cfg).eval()
    x3 = torch.randn(5, cfg.n_channels, cfg.n_times)
    x4 = x3.unsqueeze(1)
    with torch.no_grad():
        assert model(x3).shape == (5, cfg.n_classes)
        assert model(x4).shape == (5, cfg.n_classes)


def test_eegnet_trainable_step() -> None:
    cfg = EEGNetConfig(n_channels=22, n_times=512)
    model = EEGNet(cfg)
    x = torch.randn(8, 22, 512)
    y = torch.randint(0, cfg.n_classes, (8,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss0 = torch.nn.functional.cross_entropy(model(x), y)
    for _ in range(20):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < loss0.item()
