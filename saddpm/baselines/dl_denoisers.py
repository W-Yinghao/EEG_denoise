"""Supervised single-channel DL denoisers (EEGdenoiseNet baselines, ported Keras→PyTorch).

Maps a noisy ``(B, 1, L)`` segment to a denoised ``(B, L)`` signal, trained with MSE against the
clean target. Ports of the official ``simple_CNN``, ``Novel_CNN`` and ``fcNN``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor, nn


class FCNN(nn.Module):
    """Fully-connected denoiser: 4 dense layers (L→L) with ReLU + dropout."""

    def __init__(self, segment_len: int) -> None:
        super().__init__()
        l, p = segment_len, 0.3
        self.net = nn.Sequential(
            nn.Linear(l, l), nn.ReLU(), nn.Dropout(p),
            nn.Linear(l, l), nn.ReLU(), nn.Dropout(p),
            nn.Linear(l, l), nn.ReLU(), nn.Dropout(p),
            nn.Linear(l, l),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.flatten(1))


class SimpleCNN(nn.Module):
    """4× Conv1d(64,k3)+BN+ReLU+Dropout → flatten → Linear(L) (official simple_CNN)."""

    def __init__(self, segment_len: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = 1
        for _ in range(4):
            layers += [nn.Conv1d(in_ch, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3)]
            in_ch = 64
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(64 * segment_len, segment_len)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.conv(x).flatten(1))


class NovelCNN(nn.Module):
    """7-stage 1D conv net with average pooling (official Novel_CNN)."""

    def __init__(self, segment_len: int) -> None:
        super().__init__()
        widths = [32, 64, 128, 256, 512, 1024, 2048]
        drops = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
        blocks: List[nn.Module] = []
        in_ch = 1
        for w, dp in zip(widths, drops):
            stage = [nn.Conv1d(in_ch, w, 3, padding=1), nn.ReLU(),
                     nn.Conv1d(w, w, 3, padding=1), nn.ReLU()]
            if dp > 0:
                stage.append(nn.Dropout(dp))
            stage.append(nn.AvgPool1d(2))
            blocks.append(nn.Sequential(*stage))
            in_ch = w
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(widths[-1] * (segment_len // (2 ** len(widths))), segment_len)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x).flatten(1))


class _ResBasicBlock(nn.Module):
    """Conv(32,k)->Conv(16,k)->Conv(32,k) with a residual skip (official complex_CNN)."""

    def __init__(self, k: int) -> None:
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, k, padding=p), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 16, k, padding=p), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, k, padding=p), nn.BatchNorm1d(32), nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x


class ComplexCNN(nn.Module):
    """Multi-scale residual CNN (official complex_CNN): 3 parallel kernel-size branches."""

    def __init__(self, segment_len: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        self.branches = nn.ModuleList(
            nn.Sequential(_ResBasicBlock(k), _ResBasicBlock(k)) for k in (3, 5, 7)
        )
        self.post = nn.Sequential(nn.Conv1d(96, 32, 1), nn.BatchNorm1d(32), nn.ReLU())
        self.head = nn.Linear(32 * segment_len, segment_len)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        h = torch.cat([b(h) for b in self.branches], dim=1)  # (B, 96, L)
        return self.head(self.post(h).flatten(1))


class RNNLstm(nn.Module):
    """LSTM denoiser (official RNN_lstm): LSTM(1) over the time axis -> dense head."""

    def __init__(self, segment_len: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(segment_len, segment_len), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(segment_len, segment_len), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(segment_len, segment_len),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.lstm(x.transpose(1, 2))  # (B, L, 1)
        return self.head(h.flatten(1))


def make_denoiser(name: str, segment_len: int) -> nn.Module:
    """Factory: 'fcnn' | 'simple_cnn' | 'complex_cnn' | 'rnn_lstm' | 'novel_cnn'."""
    return {
        "fcnn": FCNN, "simple_cnn": SimpleCNN, "complex_cnn": ComplexCNN,
        "rnn_lstm": RNNLstm, "novel_cnn": NovelCNN,
    }[name](segment_len)


def train_denoiser(
    model: nn.Module,
    noisy: np.ndarray,
    clean: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 42,
) -> nn.Module:
    """Train a supervised denoiser (MSE) on ``(noisy, clean)`` ``(N, L)`` arrays."""
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = model.to(device)
    x = torch.from_numpy(np.ascontiguousarray(noisy)).float().unsqueeze(1)  # (N,1,L)
    y = torch.from_numpy(np.ascontiguousarray(clean)).float()  # (N,L)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def denoise_with(model: nn.Module, noisy: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    """Apply a trained denoiser to ``(N, L)`` noisy epochs → ``(N, L)`` denoised."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(noisy)).float().unsqueeze(1)
    out = [model(x[i:i + batch_size].to(device)).cpu().numpy() for i in range(0, len(x), batch_size)]
    return np.concatenate(out, axis=0)
