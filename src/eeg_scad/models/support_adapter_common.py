"""Small zero-initialized residual adapter shared by V29 DET and CDM."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from eeg_scad.models.eegdus_backbone import sinusoidal_embedding


class AdapterBlock(nn.Module):
    def __init__(self, width: int, dilation: int, context_dimension: int = 128) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8 if width >= 8 else 1, width)
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.support_film = nn.Linear(context_dimension, 2 * width)
        self.time_film = nn.Linear(context_dimension, 2 * width)

    def forward(self, value: Tensor, support: Tensor, time: Tensor) -> Tensor:
        support_scale, support_shift = self.support_film(support).chunk(2, 1)
        time_scale, time_shift = self.time_film(time).chunk(2, 1)
        hidden = self.norm(value)
        hidden = hidden * (1 + .1 * torch.tanh(support_scale + time_scale)[..., None])
        hidden = hidden + (support_shift + time_shift)[..., None]
        hidden = torch.nn.functional.silu(self.conv1(hidden))
        return value + self.conv2(hidden)


class SupportResidualAdapter(nn.Module):
    """Full-amplitude clean correction with separately projected support/time."""
    def __init__(self, input_channels: int, width: int = 32, time_conditioned: bool = False) -> None:
        super().__init__()
        self.time_conditioned = time_conditioned
        self.input = nn.Conv1d(input_channels, width, 1)
        self.support_projection = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.time_projection = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128)) if time_conditioned else None
        self.blocks = nn.ModuleList([AdapterBlock(width, dilation) for dilation in (1, 4, 16)])
        self.output = nn.Conv1d(width, 46, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor, context: Tensor, timestep: Tensor | None = None) -> Tensor:
        support = self.support_projection(context)
        if self.time_conditioned:
            if timestep is None:
                raise ValueError("diffusion adapter requires timestep")
            time = self.time_projection(sinusoidal_embedding(timestep, 128))
        else:
            time = torch.zeros_like(support)
        hidden = self.input(value)
        for block in self.blocks:
            hidden = block(hidden, support, time)
        return self.output(hidden)

    def conditioning_norms(self, context: Tensor, timestep: Tensor | None = None) -> dict[str, float]:
        support = self.support_projection(context)
        time = torch.zeros_like(support) if not self.time_conditioned else self.time_projection(sinusoidal_embedding(timestep, 128))
        film_scale, film_shift = self.blocks[0].support_film(support).chunk(2, 1)
        return {"support_context_norm": float(context.norm(dim=1).mean()), "support_projection_norm": float(support.norm(dim=1).mean()), "time_projection_norm": float(time.norm(dim=1).mean()), "film_scale_norm": float(film_scale.norm(dim=1).mean()), "film_shift_norm": float(film_shift.norm(dim=1).mean())}


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


__all__ = ["SupportResidualAdapter", "freeze"]
