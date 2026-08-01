"""Strong information-matched one-step restoration baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from eeg_cgdr.models.clean_prior import (
    CleanEEGDiffusionPrior,
    canonical_valid_time_mask,
)


@dataclass(frozen=True)
class OneStepResult:
    restored: Tensor
    prior_epsilon: Tensor
    prior_clean: Tensor


class InformationMatchedOneStep:
    """One prior evaluation followed by an exact quadratic observation update."""

    def __init__(self, prior: CleanEEGDiffusionPrior) -> None:
        self.prior = prior

    @torch.no_grad()
    def restore_detailed(
        self,
        *,
        observation: Tensor,
        channel_precision: Tensor,
        seed: int,
        timestep: int,
        proximal_strength: float = 1.0,
        valid_time_mask: Optional[Tensor] = None,
    ) -> OneStepResult:
        if observation.ndim != 3:
            raise ValueError("observation must be (B,C,L)")
        batch, channels, _ = observation.shape
        matrix_precision = channel_precision.shape == (batch, channels, channels)
        frame_precision = channel_precision.shape == (
            batch,
            observation.shape[-1],
            channels,
            channels,
        )
        if not matrix_precision and not frame_precision:
            raise ValueError("channel_precision must be (B,C,C) or (B,L,C,C)")
        if not 0 < timestep < self.prior.diffusion.num_timesteps:
            raise ValueError("one-step timestep out of range")
        if proximal_strength < 0:
            raise ValueError("proximal_strength must be non-negative")
        mask = canonical_valid_time_mask(observation, valid_time_mask)
        mask_float = mask.to(dtype=observation.dtype)
        observation = observation * mask_float
        generator = torch.Generator(device=observation.device)
        generator.manual_seed(int(seed))
        noise = torch.randn(
            observation.shape,
            device=observation.device,
            dtype=observation.dtype,
            generator=generator,
        ) * mask_float
        timesteps = torch.full(
            (batch,), int(timestep), device=observation.device, dtype=torch.long
        )
        noisy = self.prior.diffusion.q_sample(observation, timesteps, noise)
        predicted_noise = self.prior.predict_noise(
            noisy, timesteps, valid_time_mask=mask
        )
        prior_clean = self.prior.predict_clean(
            noisy,
            timesteps,
            predicted_noise,
            valid_time_mask=mask,
        )
        identity = torch.eye(
            channels, device=observation.device, dtype=observation.dtype
        ).expand(batch, -1, -1)
        if matrix_precision:
            system = identity + float(proximal_strength) * channel_precision
            right = prior_clean + float(proximal_strength) * torch.einsum(
                "bcd,bdl->bcl", channel_precision, observation
            )
            restored = torch.linalg.solve(system, right)
        else:
            frame_identity = identity[:, None, :, :]
            system = frame_identity + float(proximal_strength) * channel_precision
            observed_frames = observation.transpose(1, 2).unsqueeze(-1)
            right = prior_clean.transpose(1, 2).unsqueeze(-1) + float(
                proximal_strength
            ) * (channel_precision @ observed_frames)
            restored = torch.linalg.solve(system, right).squeeze(-1).transpose(1, 2)
        return OneStepResult(
            restored=restored * mask_float,
            prior_epsilon=predicted_noise * mask_float,
            prior_clean=prior_clean * mask_float,
        )

    @torch.no_grad()
    def restore(
        self,
        *,
        observation: Tensor,
        channel_precision: Tensor,
        seed: int,
        timestep: int,
        proximal_strength: float = 1.0,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Return only the restored tensor for the stable public baseline API."""

        return self.restore_detailed(
            observation=observation,
            channel_precision=channel_precision,
            seed=seed,
            timestep=timestep,
            proximal_strength=proximal_strength,
            valid_time_mask=valid_time_mask,
        ).restored
