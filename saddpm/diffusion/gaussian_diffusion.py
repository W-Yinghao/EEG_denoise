"""Gaussian diffusion forward process (handoff §1.1, §5).

Implements the variance-preserving forward marginal used for training:

    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)     (reparameterization)

and the single-step forward q(x_t | x_{t-1}) = N(√(1-β_t) x_{t-1}, β_t I), used by the
numerical marginal sanity check (§1.3). The reverse/posterior step (p_sample) and SDEdit are
added at M5; only the forward process is needed here.

Timesteps are 0-indexed internally: buffer index ``i`` corresponds to ``ᾱ`` after ``i+1`` noising
steps, so ``alphas_cumprod[i] = ∏_{j=0..i} (1-β_j)``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .schedule import DiffusionConfig, make_betas


def _extract(values: Tensor, t: Tensor, ndim: int) -> Tensor:
    """Gather ``values[t]`` and reshape to ``(B, 1, 1, ...)`` for broadcasting over a signal.

    Args:
        values: ``(T,)`` schedule buffer.
        t: ``(B,)`` long tensor of timestep indices.
        ndim: number of dimensions of the target tensor (e.g. 3 for ``(B, C, L)``).

    Returns:
        ``(B, 1, ...)`` tensor broadcastable against the signal.
    """
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (ndim - 1)))


class GaussianDiffusion(nn.Module):
    """Forward diffusion process with precomputed schedule buffers."""

    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.num_timesteps = cfg.num_timesteps

        betas = make_betas(cfg)  # float64, (T,)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # All buffers stored float32 (move with .to(device) automatically).
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas).float())
        self.register_buffer("sqrt_betas", torch.sqrt(betas).float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float()
        )

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Sample ``x_t ~ q(x_t | x_0)`` via the closed-form reparameterization.

        Args:
            x0: clean signal ``(B, C, L)``.
            t: ``(B,)`` long tensor of per-sample timestep indices in ``[0, T-1]``.
            noise: optional ``ε`` of the same shape as ``x0``; sampled standard normal if None.

        Returns:
            ``x_t`` of shape ``(B, C, L)``.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_abar = _extract(self.sqrt_alphas_cumprod, t, x0.ndim)
        sqrt_one_minus_abar = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.ndim)
        return sqrt_abar * x0 + sqrt_one_minus_abar * noise

    def q_sample_stepwise(
        self,
        x0: Tensor,
        t_index: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample ``x_{t_index}`` by iterating single forward steps from ``x_0``.

        Reference path for the §1.3 sanity check: applies ``t_index + 1`` single steps
        ``x_j = √(1-β_j) x_{j-1} + √β_j z_j`` so the result matches ``q(x_{t_index} | x_0)``.

        Args:
            x0: clean signal of any shape.
            t_index: target timestep index in ``[0, T-1]`` (applied to the whole batch).
            generator: optional RNG for reproducible noise.

        Returns:
            ``x_{t_index}`` of the same shape as ``x0``.
        """
        if not 0 <= t_index < self.num_timesteps:
            raise ValueError(f"t_index {t_index} out of range [0, {self.num_timesteps - 1}]")
        x = x0
        for j in range(t_index + 1):
            z = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = self.sqrt_alphas[j] * x + self.sqrt_betas[j] * z
        return x
