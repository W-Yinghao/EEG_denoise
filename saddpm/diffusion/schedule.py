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


CGDR_NUM_TIMESTEPS = 1000
CGDR_SCHEDULE = "linear"
CGDR_MAX_TERMINAL_ALPHA_BAR = 1.0e-4


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
    if cfg.num_timesteps < 2:
        raise ValueError("diffusion requires at least two timesteps")
    if not 0.0 < cfg.beta_start <= cfg.beta_end < 1.0:
        raise ValueError("diffusion betas must satisfy 0 < beta_start <= beta_end < 1")
    if cfg.schedule == "linear":
        return torch.linspace(
            cfg.beta_start, cfg.beta_end, cfg.num_timesteps, dtype=torch.float64
        )
    raise ValueError(f"unknown beta schedule: {cfg.schedule!r}")


def validate_cgdr_schedule(cfg: DiffusionConfig) -> float:
    """Validate the frozen scientific CGDR schedule and return terminal ``alpha_bar``.

    Small schedules remain available to explicitly labelled unit tests and
    compatibility ablations. Scientific CGDR checkpoints must use this
    contract: exactly 1000 linear steps and a terminal marginal that is
    effectively standard normal.
    """

    if cfg.num_timesteps != CGDR_NUM_TIMESTEPS:
        raise ValueError(
            f"scientific CGDR requires T={CGDR_NUM_TIMESTEPS}; got {cfg.num_timesteps}"
        )
    if cfg.schedule != CGDR_SCHEDULE:
        raise ValueError(
            f"scientific CGDR requires a {CGDR_SCHEDULE!r} schedule; got {cfg.schedule!r}"
        )
    betas = make_betas(cfg)
    terminal_alpha_bar = float(torch.prod(1.0 - betas))
    if terminal_alpha_bar > CGDR_MAX_TERMINAL_ALPHA_BAR:
        raise ValueError(
            "scientific CGDR terminal alpha_bar must be <= "
            f"{CGDR_MAX_TERMINAL_ALPHA_BAR:g}; got {terminal_alpha_bar:.8g}"
        )
    return terminal_alpha_bar
