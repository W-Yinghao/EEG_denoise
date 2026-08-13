"""EEGNet representations and matched deterministic privacy baselines."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class EEGNetRepresentation(nn.Module):
    def __init__(self, representation_dim: int = 128, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (22, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(32 * 16, representation_dim), nn.LayerNorm(representation_dim), nn.ELU())
        self.task_head = nn.Linear(representation_dim, 4)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x[:, None]))

    def forward(self, x: torch.Tensor, *, return_representation: bool = False):
        z = self.encode(x)
        logits = self.task_head(z)
        return (logits, z) if return_representation else logits


class _ReverseGradient(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return -ctx.scale * grad, None


class LatentDANN(nn.Module):
    def __init__(self, dim: int, n_subjects: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.subject_head = nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Linear(128, n_subjects))

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.adapter(z)

    def forward(self, z: torch.Tensor, grl_scale: float = 1.0):
        transformed = self.transform(z)
        return transformed, self.subject_head(_ReverseGradient.apply(transformed, grl_scale))


class SubjectAdversary(nn.Module):
    def __init__(self, dim: int, n_subjects: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 128), nn.ELU(), nn.Dropout(0.1), nn.Linear(128, n_subjects))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class OneStepSanitizer(nn.Module):
    """Matched one-step replacement using only kept state and frozen task logits."""
    def __init__(self, dim: int = 128, n_tasks: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + n_tasks, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, dim),
        )

    def forward(self, z_keep: torch.Tensor, task_logits: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_keep, task_logits], dim=-1))


__all__ = ["EEGNetRepresentation", "LatentDANN", "OneStepSanitizer", "SubjectAdversary"]
