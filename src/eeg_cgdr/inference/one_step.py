"""Strong information-matched one-step restoration baseline."""

from __future__ import annotations

import torch
from torch import Tensor

from eeg_cgdr.models.clean_prior import CleanEEGDiffusionPrior


class InformationMatchedOneStep:
    """One prior evaluation followed by an exact quadratic observation update."""

    def __init__(self, prior: CleanEEGDiffusionPrior) -> None:
        self.prior = prior

    @torch.no_grad()
    def restore(
        self,
        *,
        observation: Tensor,
        channel_precision: Tensor,
        seed: int,
        timestep: int,
        proximal_strength: float = 1.0,
    ) -> Tensor:
        if observation.ndim != 3:
            raise ValueError("observation must be (B,C,L)")
        batch, channels, _ = observation.shape
        if channel_precision.shape != (batch, channels, channels):
            raise ValueError("channel_precision must be (B,C,C)")
        if not 0 < timestep < self.prior.diffusion.num_timesteps:
            raise ValueError("one-step timestep out of range")
        if proximal_strength < 0:
            raise ValueError("proximal_strength must be non-negative")
        generator = torch.Generator(device=observation.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            observation.shape,
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        )
        timesteps = torch.full(
            (batch,), int(timestep), device=observation.device, dtype=torch.long
        )
        noisy = self.prior.diffusion.q_sample(observation, timesteps, noise)
        predicted_noise = self.prior.predict_noise(noisy, timesteps)
        prior_clean = self.prior.predict_clean(noisy, timesteps, predicted_noise)
        identity = torch.eye(
            channels, device=observation.device, dtype=observation.dtype
        ).expand(batch, -1, -1)
        system = identity + float(proximal_strength) * channel_precision
        right = prior_clean + float(proximal_strength) * torch.einsum(
            "bcd,bdl->bcl", channel_precision, observation
        )
        return torch.linalg.solve(system, right)
