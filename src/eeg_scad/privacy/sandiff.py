"""Selective latent diffusion for replacing a subject-linked nuisance component."""

from __future__ import annotations

import math
import torch
from torch import nn


def cosine_alpha_bar(steps: int = 1000, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1)
    values = torch.cos(((x / steps + s) / (1 + s)) * math.pi * 0.5) ** 2
    values = values / values[0]
    betas = 1 - values[1:] / values[:-1]
    betas = betas.clamp(1e-5, 0.999)
    return torch.cumprod(1 - betas, dim=0)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__(); self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, 64), nn.SiLU(), nn.Linear(64, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        angles = t.float()[:, None] * frequencies[None]
        return self.proj(torch.cat([angles.sin(), angles.cos()], dim=-1))


class SANDiff(nn.Module):
    def __init__(self, dim: int = 128, n_tasks: int = 4, steps: int = 1000) -> None:
        super().__init__()
        self.register_buffer("alpha_bar", cosine_alpha_bar(steps))
        self.time = TimeEmbedding(32)
        input_dim = dim + dim + n_tasks + 32
        self.input = nn.Linear(input_dim, 256)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
            for _ in range(3)
        ])
        self.output = nn.Linear(256, dim)

    def forward(self, x_t: torch.Tensor, z_keep: torch.Tensor, task_logits: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.input(torch.cat([x_t, z_keep, task_logits, self.time(t)], dim=-1))
        for block in self.blocks:
            h = h + block(h)
        return self.output(h)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha_bar[t].to(x0.dtype)[:, None]
        return alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(self, z_keep: torch.Tensor, task_logits: torch.Tensor, *, reverse_steps: int = 10, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch, dim = z_keep.shape
        x = torch.randn(batch, dim, device=z_keep.device) if noise is None else noise.clone()
        schedule = torch.linspace(len(self.alpha_bar) - 1, 0, reverse_steps, device=z_keep.device).round().long()
        for index, t_value in enumerate(schedule):
            t = torch.full((batch,), int(t_value), device=z_keep.device, dtype=torch.long)
            x0 = self(x, z_keep, task_logits, t)
            if index == len(schedule) - 1:
                x = x0
                continue
            next_t = schedule[index + 1]
            alpha_t = self.alpha_bar[t_value].to(x.dtype)
            alpha_next = self.alpha_bar[next_t].to(x.dtype)
            epsilon = (x - alpha_t.sqrt() * x0) / (1 - alpha_t).sqrt().clamp_min(1e-8)
            x = alpha_next.sqrt() * x0 + (1 - alpha_next).sqrt() * epsilon
        return x


__all__ = ["SANDiff", "cosine_alpha_bar"]
