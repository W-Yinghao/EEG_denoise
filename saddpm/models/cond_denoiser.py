"""Subject-aware conditional diffusion denoiser (Option C / M9, improved).

A U-Net sees ``[x_t ; corrupted]`` (2C input channels), FiLM-conditioned on the subject embedding,
and is trained supervised on synthetic ``(corrupted, clean)`` pairs — so the subject embedding is
*load-bearing* (the clean target is subject-specific). Two improvements over the first M9 attempt:

  * ``parameterization='x0'`` (default): predict the CLEAN signal directly instead of ε. This is
    bounded/stable and avoids the magnitude blow-up that ε-from-noise produced on 22 channels
    (M9 RRMSE_t 4.29 → unstable); it mirrors why the supervised EEGdenoiseNet CNNs reach CC ~0.9.
  * conditional-SDEdit sampling: the reverse starts from the *forward-diffused corrupted* at ``t*``
    (a strong prior), not pure noise, so it needs fewer steps and is far more stable.

Reuses the shared :class:`UNet1D` (``in_channels=2C, out_channels=C, subject_conditioned=True``).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from ..diffusion.gaussian_diffusion import GaussianDiffusion, _extract
from .unet1d import UNet1D


class SubjectConditionalDenoiser(nn.Module):
    """Conditional, subject-conditioned diffusion denoiser for multi-channel EEG."""

    def __init__(self, unet: UNet1D, diffusion: GaussianDiffusion, parameterization: str = "x0") -> None:
        super().__init__()
        if parameterization not in ("x0", "eps"):
            raise ValueError(f"parameterization must be 'x0' or 'eps'; got {parameterization!r}")
        self.unet = unet
        self.diffusion = diffusion
        self.parameterization = parameterization

    def _predict(self, x: Tensor, corrupted: Tensor, t: Tensor, subject_ids: Tensor) -> Tensor:
        return self.unet(torch.cat([x, corrupted], dim=1), t, subject_ids)

    def loss(self, clean: Tensor, corrupted: Tensor, subject_ids: Tensor) -> Tensor:
        """Supervised loss for ``(B, C, L)`` clean/corrupted pairs and subject ids."""
        t = torch.randint(0, self.diffusion.num_timesteps, (clean.shape[0],), device=clean.device)
        eps = torch.randn_like(clean)
        x_t = self.diffusion.q_sample(clean, t, eps)
        pred = self._predict(x_t, corrupted, t, subject_ids)
        target = clean if self.parameterization == "x0" else eps
        return torch.nn.functional.mse_loss(pred, target)

    def _eps_fn(self, corrupted: Tensor, subject_ids: Tensor):
        """Wrap the model as an ``eps_fn(x, t)`` for the shared DDIM sampler (x0->eps if needed)."""

        def eps_fn(x: Tensor, t: Tensor) -> Tensor:
            pred = self._predict(x, corrupted, t, subject_ids)
            if self.parameterization == "eps":
                return pred
            # x0-prediction -> implied eps for the DDIM step.
            sqrt_abar = _extract(self.diffusion.sqrt_alphas_cumprod, t, x.ndim)
            sqrt_1m = _extract(self.diffusion.sqrt_one_minus_alphas_cumprod, t, x.ndim)
            return (x - sqrt_abar * pred) / (sqrt_1m + 1e-8)

        return eps_fn

    @torch.no_grad()
    def denoise(
        self,
        corrupted: Tensor,
        subject_ids: Tensor,
        ddim_steps: int = 50,
        t_star: Optional[int] = 400,
    ) -> Tensor:
        """Denoise ``(B, C, L)`` corrupted windows conditioned on the subject embedding.

        Args:
            corrupted: noisy/corrupted input ``(B, C, L)``.
            subject_ids: ``(B,)`` subject ids for FiLM conditioning.
            ddim_steps: number of reverse DDIM steps.
            t_star: if set, conditional-SDEdit start — forward-diffuse the corrupted to ``t*`` and
                reverse from there (strong, stable prior). If None, reverse from pure noise.
        """
        eps_fn = self._eps_fn(corrupted, subject_ids)
        if t_star is None:
            return self.diffusion.ddim_sample_loop(
                eps_fn, corrupted.shape, corrupted.device, ddim_steps=ddim_steps
            )
        t_star = min(t_star, self.diffusion.num_timesteps - 1)
        t = torch.full((corrupted.shape[0],), t_star, device=corrupted.device, dtype=torch.long)
        x_tstar = self.diffusion.q_sample(corrupted, t, torch.randn_like(corrupted))
        return self.diffusion.ddim_sample_loop(
            eps_fn, corrupted.shape, corrupted.device, ddim_steps=ddim_steps, x_t=x_tstar, t_start=t_star
        )
