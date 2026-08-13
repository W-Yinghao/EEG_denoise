"""Exact linear task-head fibers and compact conditional replacement models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .sandiff import TimeEmbedding, cosine_alpha_bar


@dataclass(frozen=True)
class HeadFiber:
    """Orthonormal row/fiber decomposition for a frozen linear softmax head."""

    weight: np.ndarray
    bias: np.ndarray
    centered_map: np.ndarray
    row_basis: np.ndarray
    null_basis: np.ndarray
    rank: int
    tolerance: float

    @classmethod
    def from_linear(cls, head: nn.Linear, *, rtol: float = 1e-7) -> "HeadFiber":
        weight = head.weight.detach().cpu().double().numpy()
        bias = head.bias.detach().cpu().double().numpy()
        classes = weight.shape[0]
        centering = np.eye(classes) - np.ones((classes, classes)) / classes
        centered_map = centering @ weight
        _, singular, vh = np.linalg.svd(centered_map, full_matrices=True)
        tolerance = float(max(centered_map.shape) * singular.max(initial=0.0) * rtol)
        rank = int((singular > tolerance).sum())
        return cls(
            weight=weight,
            bias=bias,
            centered_map=centered_map,
            row_basis=vh[:rank].T.copy(),
            null_basis=vh[rank:].T.copy(),
            rank=rank,
            tolerance=tolerance,
        )

    @property
    def representation_dim(self) -> int:
        return int(self.weight.shape[1])

    @property
    def fiber_dim(self) -> int:
        return int(self.null_basis.shape[1])

    def centered_logits(self, z: np.ndarray) -> np.ndarray:
        logits = np.asarray(z, dtype=np.float64) @ self.weight.T + self.bias
        return logits - logits.mean(axis=1, keepdims=True)

    def decompose(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(z, dtype=np.float64)
        z_head = (values @ self.row_basis) @ self.row_basis.T
        u = values @ self.null_basis
        return z_head.astype(np.float32), u.astype(np.float32), self.centered_logits(values).astype(np.float32)

    def compose(self, z_head: np.ndarray, u: np.ndarray) -> np.ndarray:
        return (np.asarray(z_head, dtype=np.float64) + np.asarray(u, dtype=np.float64) @ self.null_basis.T).astype(np.float32)

    def diagnostics(self) -> dict[str, float | int]:
        n = self.null_basis
        r = self.row_basis
        return {
            "head_rank": self.rank,
            "fiber_dim": self.fiber_dim,
            "svd_tolerance": self.tolerance,
            "null_residual_max_abs": float(np.abs(self.centered_map @ n).max(initial=0.0)),
            "null_orthogonality_max_abs": float(np.abs(n.T @ n - np.eye(self.fiber_dim)).max(initial=0.0)),
            "row_orthogonality_max_abs": float(np.abs(r.T @ r - np.eye(self.rank)).max(initial=0.0)),
            "row_null_max_abs": float(np.abs(r.T @ n).max(initial=0.0)),
        }


class FiberOneStep(nn.Module):
    """Conditional-mean fiber replacement using centered logits only."""

    def __init__(self, fiber_dim: int, condition_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, fiber_dim),
        )

    def forward(self, centered_logits: torch.Tensor) -> torch.Tensor:
        return self.net(centered_logits)


class FiberSANDiff(nn.Module):
    """x0 diffusion of pooled fiber coordinates conditioned on centered logits."""

    def __init__(self, fiber_dim: int, condition_dim: int = 4, steps: int = 1000) -> None:
        super().__init__()
        self.register_buffer("alpha_bar", cosine_alpha_bar(steps))
        self.time = TimeEmbedding(32)
        self.input = nn.Linear(fiber_dim + condition_dim + 32, 256)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
            for _ in range(3)
        ])
        self.output = nn.Linear(256, fiber_dim)

    def forward(self, u_t: torch.Tensor, centered_logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hidden = self.input(torch.cat([u_t, centered_logits, self.time(t)], dim=-1))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden)

    def q_sample(self, u0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_bar[t].to(u0.dtype)[:, None]
        return alpha.sqrt() * u0 + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(self, centered_logits: torch.Tensor, *, reverse_steps: int = 10, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch = len(centered_logits)
        fiber_dim = self.output.out_features
        state = torch.randn(batch, fiber_dim, device=centered_logits.device) if noise is None else noise.clone()
        schedule = torch.linspace(len(self.alpha_bar) - 1, 0, reverse_steps, device=centered_logits.device).round().long()
        for index, timestep in enumerate(schedule):
            t = torch.full((batch,), int(timestep), device=centered_logits.device, dtype=torch.long)
            u0 = self(state, centered_logits, t)
            if index == len(schedule) - 1:
                state = u0
                continue
            next_t = schedule[index + 1]
            alpha_t = self.alpha_bar[timestep].to(state.dtype)
            alpha_next = self.alpha_bar[next_t].to(state.dtype)
            epsilon = (state - alpha_t.sqrt() * u0) / (1 - alpha_t).sqrt().clamp_min(1e-8)
            state = alpha_next.sqrt() * u0 + (1 - alpha_next).sqrt() * epsilon
        return state


__all__ = ["HeadFiber", "FiberOneStep", "FiberSANDiff"]
