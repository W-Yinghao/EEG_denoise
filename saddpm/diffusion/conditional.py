"""Conditional diffusion denoiser (EEGdenoiseNet / M8).

A DDPM whose U-Net is conditioned on the noisy input: the network sees ``[x_t ; noisy]`` (2 input
channels) and predicts either the noise ``ε`` (``parameterization='eps'``) or the clean signal
``x_0`` (``parameterization='x0'``) of the clean target. Sampling runs the reverse process
conditioned on the (fixed) noisy segment → a denoised estimate. Reuses the shared
:class:`GaussianDiffusion` and :class:`UNet1D` (built with ``in_channels=2, out_channels=1``).

Inference knobs (added at M10 to close the gap with supervised CNN regressors):
  * ``t_star`` — conditional-SDEdit start: reverse from the forward-diffused *noisy* at ``t*``
    (a strong, low-variance prior) instead of from pure noise. ``None`` = full generation from ``x_T``.
  * ``k`` — posterior-mean ensembling: average ``k`` independent reverse draws. RRMSE_t / CC reward
    the conditional mean E[clean|noisy]; averaging draws is a Monte-Carlo estimate of that mean.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..models.unet1d import UNet1D
from .gaussian_diffusion import GaussianDiffusion, _extract


class ConditionalDiffusionDenoiser(nn.Module):
    """Noisy-conditioned DDPM for single-channel denoising with paired ground truth."""

    def __init__(self, unet: UNet1D, diffusion: GaussianDiffusion, parameterization: str = "eps") -> None:
        super().__init__()
        if parameterization not in ("x0", "eps"):
            raise ValueError(f"parameterization must be 'x0' or 'eps'; got {parameterization!r}")
        self.unet = unet
        self.diffusion = diffusion
        self.parameterization = parameterization

    def loss(self, clean: Tensor, noisy: Tensor) -> Tensor:
        """Supervised diffusion loss for a batch of ``(B, 1, L)`` clean/noisy pairs."""
        t = torch.randint(0, self.diffusion.num_timesteps, (clean.shape[0],), device=clean.device)
        eps = torch.randn_like(clean)
        x_t = self.diffusion.q_sample(clean, t, eps)
        pred = self.unet(torch.cat([x_t, noisy], dim=1), t)
        target = clean if self.parameterization == "x0" else eps
        return torch.nn.functional.mse_loss(pred, target)

    def _eps_fn(self, noisy: Tensor):
        """Wrap the model as an ``eps_fn(x, t)`` for the shared samplers (x0->eps if needed)."""

        def eps_fn(x: Tensor, t: Tensor) -> Tensor:
            pred = self.unet(torch.cat([x, noisy], dim=1), t)
            if self.parameterization == "eps":
                return pred
            sqrt_abar = _extract(self.diffusion.sqrt_alphas_cumprod, t, x.ndim)
            sqrt_1m = _extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x.ndim)
            return (x - sqrt_abar * pred) / (sqrt_1m + 1e-8)

        return eps_fn

    @torch.no_grad()
    def _denoise_once(self, noisy: Tensor, ddim_steps: int, t_star: Optional[int]) -> Tensor:
        eps_fn = self._eps_fn(noisy)
        if t_star is None:
            return self.diffusion.ddim_sample_loop(eps_fn, noisy.shape, noisy.device, ddim_steps=ddim_steps)
        t_star = min(t_star, self.diffusion.num_timesteps - 1)
        t = torch.full((noisy.shape[0],), t_star, device=noisy.device, dtype=torch.long)
        x_tstar = self.diffusion.q_sample(noisy, t, torch.randn_like(noisy))
        return self.diffusion.ddim_sample_loop(
            eps_fn, noisy.shape, noisy.device, ddim_steps=ddim_steps, x_t=x_tstar, t_start=t_star
        )

    @torch.no_grad()
    def denoise(self, noisy: Tensor, ddim_steps: int = 50, t_star: Optional[int] = None, k: int = 1) -> Tensor:
        """Denoise ``(B, 1, L)`` noisy segments.

        Args:
            noisy: conditioning input ``(B, 1, L)``.
            ddim_steps: number of reverse DDIM steps.
            t_star: conditional-SDEdit start timestep (``None`` = full generation from pure noise).
            k: number of independent reverse draws to average (posterior-mean ensembling).
        """
        if k == 1:
            return self._denoise_once(noisy, ddim_steps, t_star)
        acc = torch.zeros_like(noisy)
        for _ in range(k):
            acc += self._denoise_once(noisy, ddim_steps, t_star)
        return acc / k
