"""Subject-aware conditional diffusion denoiser (Option C / M9).

Predicts the noise ``ε`` of the *clean* signal ``x_0`` from ``[x_t ; corrupted]`` (2C input channels),
FiLM-conditioned on the subject embedding. Trained supervised on synthetic ``(corrupted, clean)``
pairs, so — unlike the Phase-1 SADDPM — the subject embedding is *load-bearing*: the clean target is
subject-specific, so conditioning on ``e(s)`` measurably helps reconstruction (audit finding #2).
Reuses the shared :class:`UNet1D` (``in_channels=2C, out_channels=C, subject_conditioned=True``).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..diffusion.gaussian_diffusion import GaussianDiffusion
from .unet1d import UNet1D


class SubjectConditionalDenoiser(nn.Module):
    """Conditional, subject-conditioned diffusion denoiser for multi-channel EEG."""

    def __init__(self, unet: UNet1D, diffusion: GaussianDiffusion) -> None:
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion

    def loss(self, clean: Tensor, corrupted: Tensor, subject_ids: Tensor) -> Tensor:
        """Noise-prediction loss for ``(B, C, L)`` clean/corrupted pairs and subject ids."""
        t = torch.randint(0, self.diffusion.num_timesteps, (clean.shape[0],), device=clean.device)
        eps = torch.randn_like(clean)
        x_t = self.diffusion.q_sample(clean, t, eps)
        pred = self.unet(torch.cat([x_t, corrupted], dim=1), t, subject_ids)
        return torch.nn.functional.mse_loss(pred, eps)

    def _eps_fn(self, corrupted: Tensor, subject_ids: Tensor):
        def eps_fn(x: Tensor, t: Tensor) -> Tensor:
            return self.unet(torch.cat([x, corrupted], dim=1), t, subject_ids)

        return eps_fn

    @torch.no_grad()
    def denoise(self, corrupted: Tensor, subject_ids: Tensor, ddim_steps: int = 50) -> Tensor:
        """Denoise ``(B, C, L)`` corrupted windows conditioned on the subject embedding."""
        return self.diffusion.ddim_sample_loop(
            self._eps_fn(corrupted, subject_ids), corrupted.shape, corrupted.device, ddim_steps=ddim_steps
        )
