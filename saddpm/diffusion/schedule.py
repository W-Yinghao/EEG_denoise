"""Diffusion noise schedule (handoff §5).

Variance-preserving DDPM forward process with a linear ``β`` schedule, T=1000, β∈[1e-4, 0.02].
The schedule is computed in float64 for precision, then consumed as float32 buffers by
:class:`saddpm.diffusion.gaussian_diffusion.GaussianDiffusion`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class DiffusionConfig:
    """Hyperparameters of the forward diffusion schedule."""

    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DiffusionConfig":
        """Load a :class:`DiffusionConfig` from ``configs/diffusion.yaml``."""
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(
            num_timesteps=int(raw["num_timesteps"]),
            beta_start=float(raw["beta_start"]),
            beta_end=float(raw["beta_end"]),
            schedule=str(raw["schedule"]),
        )


def make_betas(cfg: DiffusionConfig) -> torch.Tensor:
    """Build the ``β_t`` vector of length ``T`` (float64).

    Args:
        cfg: diffusion configuration.

    Returns:
        ``(T,)`` float64 tensor of per-step variances ``β_t``.

    Raises:
        ValueError: for an unknown schedule kind.
    """
    if cfg.schedule == "linear":
        return torch.linspace(
            cfg.beta_start, cfg.beta_end, cfg.num_timesteps, dtype=torch.float64
        )
    raise ValueError(f"unknown beta schedule: {cfg.schedule!r}")
