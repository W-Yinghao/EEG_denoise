"""Conditional diffusion denoiser (EEGdenoiseNet / M8).

A DDPM whose U-Net is conditioned on the noisy input: the network sees ``[x_t ; noisy]`` (2 input
channels) and predicts the noise ``ε`` of the clean signal ``x_0``. Sampling runs the reverse
process from ``x_T`` conditioned on the (fixed) noisy segment → a denoised estimate. Reuses the
shared :class:`GaussianDiffusion` and :class:`UNet1D` (built with ``in_channels=2, out_channels=1``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..models.unet1d import UNet1D
from .gaussian_diffusion import GaussianDiffusion


class ConditionalDiffusionDenoiser(nn.Module):
    """Noisy-conditioned DDPM for single-channel denoising with paired ground truth."""

    def __init__(self, unet: UNet1D, diffusion: GaussianDiffusion) -> None:
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion

    def loss(self, clean: Tensor, noisy: Tensor) -> Tensor:
        """Noise-prediction loss for a batch of ``(B, 1, L)`` clean/noisy pairs."""
        t = torch.randint(0, self.diffusion.num_timesteps, (clean.shape[0],), device=clean.device)
        eps = torch.randn_like(clean)
        x_t = self.diffusion.q_sample(clean, t, eps)
        pred = self.unet(torch.cat([x_t, noisy], dim=1), t)
        return torch.nn.functional.mse_loss(pred, eps)

    def _eps_fn(self, noisy: Tensor):
        def eps_fn(x: Tensor, t: Tensor) -> Tensor:
            return self.unet(torch.cat([x, noisy], dim=1), t)

        return eps_fn

    @torch.no_grad()
    def denoise(self, noisy: Tensor, ddim_steps: int = 50) -> Tensor:
        """Denoise ``(B, 1, L)`` noisy segments via the noisy-conditioned reverse process."""
        return self.diffusion.ddim_sample_loop(
            self._eps_fn(noisy), noisy.shape, noisy.device, ddim_steps=ddim_steps
        )
