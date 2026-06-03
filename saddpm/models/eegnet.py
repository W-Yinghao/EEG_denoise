"""EEGNet-8,2 downstream classifier (handoff [DD-5]; Lawhern et al., 2018).

Standard compact CNN for EEG. Operates on ``(B, 1, C, T)`` inputs (a window reshaped with a
singleton "image" channel). The same architecture + training config is used for both the ICA and
SADDPM denoising pipelines so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn


@dataclass(frozen=True)
class EEGNetConfig:
    n_classes: int = 4
    n_channels: int = 22
    n_times: int = 512
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 100

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EEGNetConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**yaml.safe_load(handle))


class _Constrained(nn.Module):
    """Wraps a module and clips its weight norm (EEGNet's max-norm constraints)."""

    def __init__(self, module: nn.Module, max_norm: float) -> None:
        super().__init__()
        self.module = module
        self.max_norm = max_norm

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            w = self.module.weight
            flat = w.reshape(w.shape[0], -1)
            norm = flat.norm(dim=1).clamp(min=1e-8)  # per-output-unit norm
            scale = (norm.clamp(max=self.max_norm) / norm).reshape(-1, *([1] * (w.ndim - 1)))
            w.mul_(scale)
        return self.module(x)


class EEGNet(nn.Module):
    """EEGNet-8,2 for 4-class motor-imagery classification."""

    def __init__(self, cfg: EEGNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        f1, d, f2 = cfg.F1, cfg.D, cfg.F2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, cfg.kernel_length), padding=(0, cfg.kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        # Depthwise spatial conv across channels (max-norm constrained).
        self.depthwise = _Constrained(
            nn.Conv2d(f1, f1 * d, (cfg.n_channels, 1), groups=f1, bias=False), max_norm=1.0
        )
        self.block1_tail = nn.Sequential(
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, cfg.pool1)),
            nn.Dropout(cfg.dropout),
        )
        # Separable conv: depthwise temporal + pointwise.
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, cfg.pool2)),
            nn.Dropout(cfg.dropout),
        )
        feat_len = cfg.n_times // (cfg.pool1 * cfg.pool2)
        self.classifier = _Constrained(nn.Linear(f2 * feat_len, cfg.n_classes), max_norm=0.25)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a batch of windows.

        Args:
            x: ``(B, C, T)`` or ``(B, 1, C, T)`` window batch.

        Returns:
            ``(B, n_classes)`` logits.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        h = self.block1(x)
        h = self.block1_tail(self.depthwise(h))
        h = self.separable(h)
        return self.classifier(h.flatten(1))
