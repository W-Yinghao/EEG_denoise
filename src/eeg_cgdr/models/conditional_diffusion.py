"""Operator-conditioned multichannel EEG diffusion comparator.

This model is deliberately separate from the unconditional clean prior used by
the CGDR M1/M2/M4 arms.  It predicts diffusion noise conditional on exactly the
deployment-time information exposed to the task-matched deterministic U-Net:
the observed EEG, one frozen operator projector, framewise external-reference
attenuation, and the valid-time mask.  ``x_t`` and the diffusion timestep are
algorithm state, not additional deployment information.

The implementation reuses the same masked multichannel :class:`UNet1D` and
Gaussian diffusion utilities as the clean prior.  It is intended for an
explicitly exploratory Klados source-record comparison; it is not a renamed
CGDR/M2 sampler and does not by itself constitute formal G1 or G3 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig, validate_cgdr_schedule
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .clean_prior import canonical_valid_time_mask
from .deterministic_unet import (
    DeterministicUNetConfig,
    _frame_attenuation,
    _projector_batch,
)


@dataclass(frozen=True)
class ConditionalSample:
    """One conditional DDIM result and its exact network-call count."""

    restored: Tensor
    network_calls: int


class OperatorConditionedEEGDiffusion(nn.Module):
    """Joint multichannel epsilon model with matched operator conditioning."""

    visible_input_fields = (
        "observed_query_eeg",
        "operator_projector",
        "framewise_external_eog_attenuation",
        "valid_time_mask",
    )
    algorithm_state_fields = ("diffused_clean_state_x_t", "diffusion_timestep")

    def __init__(
        self,
        backbone_config: DeterministicUNetConfig,
        diffusion_config: DiffusionConfig,
        *,
        enforce_scientific_schedule: bool = True,
    ) -> None:
        super().__init__()
        if backbone_config.eeg_channels < 2:
            raise ValueError("conditional scientific comparator must be multichannel")
        if enforce_scientific_schedule:
            terminal_alpha_bar = validate_cgdr_schedule(diffusion_config)
        else:
            terminal_alpha_bar = None
        channels = backbone_config.eeg_channels
        # The deterministic conditioning stack is [y, Pi*y, a_t, vec(Pi)].
        # Conditional diffusion prepends only its algorithmic state x_t.
        conditioning_channels = 2 * channels + 1 + channels * channels
        model_config = ModelConfig(
            in_channels=channels + conditioning_channels,
            out_channels=channels,
            signal_length=backbone_config.signal_length,
            base_channels=backbone_config.base_channels,
            channel_mults=list(backbone_config.channel_mults),
            num_res_blocks=backbone_config.num_res_blocks,
            groupnorm_groups=backbone_config.groupnorm_groups,
            dropout=backbone_config.dropout,
            time_sinusoidal_dim=backbone_config.time_sinusoidal_dim,
            time_embed_dim=backbone_config.time_embed_dim,
            attention_length=backbone_config.attention_length,
            attention_heads=backbone_config.attention_heads,
        )
        self.backbone_config = backbone_config
        self.diffusion_config = diffusion_config
        self.enforce_scientific_schedule = bool(enforce_scientific_schedule)
        self.terminal_alpha_bar = terminal_alpha_bar
        self.conditioning_channels = conditioning_channels
        self.unet = UNet1D(model_config, subject_conditioned=False)
        self.diffusion = GaussianDiffusion(diffusion_config)

    def _validate_state(self, x_t: Tensor, timesteps: Tensor) -> None:
        config = self.backbone_config
        if x_t.ndim != 3 or x_t.shape[1] != config.eeg_channels:
            raise ValueError("x_t must have shape (B,C,L) matching the montage")
        if x_t.shape[-1] != config.signal_length:
            raise ValueError("x_t window length differs from the model config")
        if not x_t.dtype.is_floating_point or not bool(torch.isfinite(x_t).all()):
            raise ValueError("x_t must be finite floating point")
        if timesteps.shape != (x_t.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a (B,) long tensor")
        if timesteps.device != x_t.device:
            raise ValueError("x_t and timesteps must be on the same device")
        if bool((timesteps < 0).any()) or bool(
            (timesteps >= self.diffusion.num_timesteps).any()
        ):
            raise ValueError("diffusion timestep lies outside the frozen schedule")

    def conditioning_features(
        self,
        observed: Tensor,
        *,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the exact deterministic-U-Net conditioning stack and mask."""

        config = self.backbone_config
        if observed.ndim != 3 or observed.shape[1] != config.eeg_channels:
            raise ValueError("observed EEG must have shape (B,C,L) matching montage")
        if observed.shape[-1] != config.signal_length:
            raise ValueError("observed EEG window length differs from model config")
        if not observed.dtype.is_floating_point or not bool(
            torch.isfinite(observed).all()
        ):
            raise ValueError("observed EEG must be finite floating point")
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        mask_float = mask.to(dtype=observed.dtype)
        masked_observed = observed * mask_float
        projection = _projector_batch(masked_observed, projector)
        attenuation_value = _frame_attenuation(
            masked_observed, attenuation
        ) * mask[:, 0, :]
        projected = torch.einsum("bij,bjl->bil", projection, masked_observed)
        operator_features = projection.flatten(start_dim=1)[:, :, None].expand(
            -1, -1, observed.shape[-1]
        )
        features = torch.cat(
            (
                masked_observed,
                projected,
                attenuation_value[:, None, :],
                operator_features,
            ),
            dim=1,
        ) * mask_float
        if features.shape[1] != self.conditioning_channels:
            raise AssertionError("conditional feature construction changed channel count")
        return features, mask

    def predict_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        observed: Tensor,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict epsilon from the noisy clean state and matched legal inputs."""

        self._validate_state(x_t, timesteps)
        if observed.shape != x_t.shape or observed.device != x_t.device:
            raise ValueError("observed EEG must match x_t shape and device")
        if observed.dtype != x_t.dtype:
            raise ValueError("observed EEG and x_t must use the same dtype")
        conditioning, mask = self.conditioning_features(
            observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=valid_time_mask,
        )
        mask_float = mask.to(dtype=x_t.dtype)
        model_input = torch.cat((x_t * mask_float, conditioning), dim=1)
        return self.unet(
            model_input,
            timesteps,
            valid_time_mask=mask,
        ) * mask_float

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        observed: Tensor,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.predict_noise(
            x_t,
            timesteps,
            observed=observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=valid_time_mask,
        )

    def training_loss(
        self,
        clean_target: Tensor,
        *,
        observed: Tensor,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Uniform-timestep epsilon MSE on the paired clean target."""

        if clean_target.shape != observed.shape or clean_target.device != observed.device:
            raise ValueError("clean target and observation must match shape and device")
        if clean_target.dtype != observed.dtype:
            raise ValueError("clean target and observation must use the same dtype")
        if not clean_target.dtype.is_floating_point or not bool(
            torch.isfinite(clean_target).all()
        ):
            raise ValueError("clean target must be finite floating point")
        mask = canonical_valid_time_mask(clean_target, valid_time_mask)
        mask_float = mask.to(dtype=clean_target.dtype)
        clean = clean_target * mask_float
        batch = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (batch,),
                device=clean.device,
                dtype=torch.long,
                generator=generator,
            )
        else:
            self._validate_state(clean, timesteps)
        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        elif (
            noise.shape != clean.shape
            or noise.device != clean.device
            or noise.dtype != clean.dtype
        ):
            raise ValueError("training noise must match the clean target")
        noise = noise * mask_float
        x_t = self.diffusion.q_sample(clean, timesteps, noise) * mask_float
        predicted = self.predict_noise(
            x_t,
            timesteps,
            observed=observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
        squared_error = (predicted - noise).square() * mask_float
        denominator = (mask_float.sum() * clean.shape[1]).clamp_min(1.0)
        return squared_error.sum() / denominator

    @torch.no_grad()
    def sample_ddim(
        self,
        *,
        observed: Tensor,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
        ddim_steps: int,
        eta: float = 0.0,
        initial_noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> ConditionalSample:
        """Run conditional DDIM and report the exact U-Net invocation count."""

        mask = canonical_valid_time_mask(observed, valid_time_mask)
        mask_float = mask.to(dtype=observed.dtype)
        if initial_noise is None:
            initial_noise = torch.randn(
                observed.shape,
                device=observed.device,
                dtype=observed.dtype,
                generator=generator,
            )
        elif (
            initial_noise.shape != observed.shape
            or initial_noise.device != observed.device
            or initial_noise.dtype != observed.dtype
        ):
            raise ValueError("initial noise must match the observed EEG")
        initial_noise = initial_noise * mask_float
        calls = 0

        def eps_fn(x_t: Tensor, timesteps: Tensor) -> Tensor:
            nonlocal calls
            calls += 1
            return self.predict_noise(
                x_t,
                timesteps,
                observed=observed,
                projector=projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )

        restored = self.diffusion.ddim_sample_loop(
            eps_fn,
            shape=tuple(observed.shape),
            device=observed.device,
            ddim_steps=int(ddim_steps),
            eta=float(eta),
            x_t=initial_noise,
            valid_time_mask=mask,
        )
        if calls != int(ddim_steps):
            raise AssertionError(
                f"DDIM requested {ddim_steps} steps but used {calls} network calls"
            )
        return ConditionalSample(restored=restored * mask_float, network_calls=calls)


__all__ = ["ConditionalSample", "OperatorConditionedEEGDiffusion"]
