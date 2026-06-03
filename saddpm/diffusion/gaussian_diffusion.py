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

from typing import Callable, Optional

import torch
from torch import Tensor, nn

from .schedule import DiffusionConfig, make_betas

EpsFn = Callable[[Tensor, Tensor], Tensor]
"""Signature of a noise predictor: ``eps_fn(x_t, t) -> eps_hat`` with ``t`` a ``(B,)`` long tensor."""


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

        # --- reverse / posterior process (DDPM, Ho et al.) ---
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        # clamp the t=0 variance (which is 0) before taking the log.
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            ((1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)).float(),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float())
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0).float()
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

    # ------------------------------------------------------------------ reverse process

    def predict_xstart_from_eps(self, xt: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Recover the predicted clean signal ``x̂_0`` from ``x_t`` and predicted noise ``ε``."""
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, xt.ndim) * xt
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, xt.ndim) * eps
        )

    def q_posterior_mean(self, x0: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Mean of the forward posterior ``q(x_{t-1} | x_t, x_0)``."""
        return (
            _extract(self.posterior_mean_coef1, t, xt.ndim) * x0
            + _extract(self.posterior_mean_coef2, t, xt.ndim) * xt
        )

    @torch.no_grad()
    def p_sample(
        self,
        eps_fn: EpsFn,
        xt: Tensor,
        t: Tensor,
        clip_denoised: bool = False,
    ) -> Tensor:
        """One reverse DDPM step ``x_t -> x_{t-1}`` (Ho et al. ancestral sampling).

        Args:
            eps_fn: noise predictor ``(x_t, t) -> ε̂``.
            xt: current ``x_t`` of shape ``(B, C, L)``.
            t: ``(B,)`` long tensor of the current timestep indices.
            clip_denoised: if True, clamp ``x̂_0`` to ``[-1, 1]`` (off for z-scored EEG).

        Returns:
            ``x_{t-1}`` of shape ``(B, C, L)``.
        """
        eps = eps_fn(xt, t)
        x0 = self.predict_xstart_from_eps(xt, t, eps)
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        mean = self.q_posterior_mean(x0, xt, t)
        log_var = _extract(self.posterior_log_variance_clipped, t, xt.ndim)
        noise = torch.randn_like(xt)
        nonzero = (t != 0).float().reshape(-1, *((1,) * (xt.ndim - 1)))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        eps_fn: EpsFn,
        shape: tuple[int, ...],
        device: torch.device,
        clip_denoised: bool = False,
    ) -> Tensor:
        """Full ancestral sampling from ``x_T ~ N(0, I)`` down to ``x_0``.

        Args:
            eps_fn: noise predictor ``(x_t, t) -> ε̂``.
            shape: output shape ``(B, C, L)``.
            device: device to sample on.
            clip_denoised: passed through to :meth:`p_sample`.

        Returns:
            ``x_0`` of shape ``shape``.
        """
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(eps_fn, x, t, clip_denoised=clip_denoised)
        return x
