"""Subject-calibrated diffusion in a low-dimensional ocular-artifact space.

The model never generates a full EEG recording.  It predicts a standardized
low-dimensional artifact latent conditional on the observed EEG and the
complete support-derived EOG-to-EEG transfer (HEOG/VEOG and, where present,
REOG).  Reconstruction is observation anchored: the arithmetic posterior-mean
latent is mapped through the transfer and subtracted from the observed query.
Participant identity and query-time EOG are not inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    artifact_conditioning_channels,
    build_artifact_conditioning,
)
from .artifact_latent_inference import canonical_artifact_delta


def _extract(values: Tensor, timesteps: Tensor, ndim: int) -> Tensor:
    return values.gather(0, timesteps).reshape(
        timesteps.shape[0], *((1,) * (ndim - 1))
    )


def cosine_alpha_bar(
    num_timesteps: int,
    *,
    offset: float = 0.008,
) -> tuple[Tensor, Tensor]:
    """Return float64 cosine betas and cumulative alphas.

    The construction follows the improved-DDPM cosine schedule and clips only
    individual betas, not the cumulative trajectory.
    """

    if isinstance(num_timesteps, bool) or int(num_timesteps) != num_timesteps:
        raise ValueError("num_timesteps must be an integer")
    steps = int(num_timesteps)
    if steps < 2:
        raise ValueError("cosine diffusion requires at least two timesteps")
    if not math.isfinite(offset) or not 0.0 <= float(offset) < 1.0:
        raise ValueError("cosine offset must lie in [0,1)")
    grid = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64)
    curve = torch.cos(
        ((grid + float(offset)) / (1.0 + float(offset))) * math.pi / 2.0
    ).square()
    curve = curve / curve[0]
    betas = (1.0 - curve[1:] / curve[:-1]).clamp(1.0e-8, 0.999)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    if not bool(torch.isfinite(betas).all()) or not bool(
        torch.isfinite(alphas_cumprod).all()
    ):
        raise AssertionError("cosine schedule produced non-finite values")
    if not bool((alphas_cumprod[1:] < alphas_cumprod[:-1]).all()):
        raise AssertionError("cosine cumulative alpha must strictly decrease")
    return betas, alphas_cumprod


def artifact_posterior_point_estimate(samples: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """Return the frozen arithmetic mean and population SD of exactly K=8.

    This API intentionally accepts no target, label, score or metric.  It
    therefore cannot implement target-aware or best-of-K sample selection.
    """

    values = tuple(samples)
    if len(values) != 8:
        raise ValueError("artifact posterior point estimate requires exactly K=8")
    reference = values[0]
    if not isinstance(reference, Tensor) or not reference.dtype.is_floating_point:
        raise ValueError("posterior samples must be floating-point tensors")
    for value in values:
        if (
            not isinstance(value, Tensor)
            or value.shape != reference.shape
            or value.device != reference.device
            or value.dtype != reference.dtype
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("posterior samples must have matching finite tensor values")
    stacked = torch.stack(values, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)


@dataclass(frozen=True)
class ArtifactLatentDiffusionConfig:
    num_timesteps: int = 1000
    cosine_offset: float = 0.008
    prediction_target: str = "v"
    min_snr_gamma: float = 5.0
    dynamic_threshold_quantile: float = 0.995
    standardized_latent_absolute_clip: float = 5.0
    posterior_samples: int = 8

    def __post_init__(self) -> None:
        if self.prediction_target != "v":
            raise ValueError("primary artifact diffusion requires v-prediction")
        _, alphas_cumprod = cosine_alpha_bar(
            self.num_timesteps,
            offset=self.cosine_offset,
        )
        if float(alphas_cumprod[-1]) > 1.0e-4:
            raise ValueError(
                "artifact diffusion terminal alpha_bar must not exceed 1e-4"
            )
        if not math.isfinite(self.min_snr_gamma) or self.min_snr_gamma <= 0:
            raise ValueError("min_snr_gamma must be finite and positive")
        if not 0.5 < self.dynamic_threshold_quantile < 1.0:
            raise ValueError("dynamic threshold quantile must lie in (0.5,1)")
        if (
            not math.isfinite(self.standardized_latent_absolute_clip)
            or self.standardized_latent_absolute_clip <= 1.0
        ):
            raise ValueError("standardized latent clip must exceed one")
        if self.posterior_samples != 8:
            raise ValueError("the frozen posterior point rule requires K=8")


@dataclass(frozen=True)
class ArtifactTrajectoryStep:
    sample_index: int
    reverse_index: int
    timestep: int
    latent_rms: float
    predicted_v_rms: float
    predicted_x0_rms: float
    mapped_contamination_rms: float
    adjacent_latent_rms_ratio: float | None
    finite: bool
    clipped_fraction: float


@dataclass(frozen=True)
class ArtifactPosterior:
    standardized_latent_mean: Tensor
    standardized_latent_standard_deviation: Tensor
    correction: Tensor
    restored: Tensor
    sample_count: int
    network_calls: int
    trajectories: tuple[ArtifactTrajectoryStep, ...]
    # Additive J4 fields.  Older diagnostic samplers may omit them, but the
    # primary artifact-latent sampler retains an explicit leading K dimension.
    standardized_latent_samples: Tensor | None = None
    correction_samples: Tensor | None = None


def _posterior_transfer(
    value: Tensor,
    *,
    observed: Tensor,
    latent_channels: int,
) -> Tensor:
    transfer = torch.as_tensor(
        value,
        device=observed.device,
        dtype=observed.dtype,
    ).detach()
    expected_shared = (observed.shape[1], latent_channels)
    expected_batched = (observed.shape[0], *expected_shared)
    if transfer.shape == expected_shared:
        transfer = transfer.unsqueeze(0).expand(observed.shape[0], -1, -1)
    if transfer.shape != expected_batched:
        raise ValueError(
            "normalized_transfer must have shape (C,E) or (B,C,E)"
        )
    if not bool(torch.isfinite(transfer).all()):
        raise ValueError("normalized_transfer contains non-finite values")
    return transfer


def _posterior_latent_stat(
    value: Tensor,
    *,
    observed: Tensor,
    latent_channels: int,
    name: str,
    strictly_positive: bool,
) -> Tensor:
    statistic = torch.as_tensor(
        value,
        device=observed.device,
        dtype=observed.dtype,
    ).detach()
    if statistic.shape == (latent_channels,):
        statistic = statistic.unsqueeze(0).expand(observed.shape[0], -1)
    if statistic.shape != (observed.shape[0], latent_channels):
        raise ValueError(f"{name} must have shape (E,) or (B,E)")
    if not bool(torch.isfinite(statistic).all()):
        raise ValueError(f"{name} contains non-finite values")
    if strictly_positive and bool((statistic <= 0.0).any()):
        raise ValueError(f"{name} must be strictly positive")
    return statistic


def _posterior_channel_mask(observed: Tensor, channel_mask: Tensor) -> Tensor:
    value = torch.as_tensor(channel_mask, device=observed.device)
    if value.shape == (observed.shape[1],):
        value = value.unsqueeze(0).expand(observed.shape[0], -1)
    if value.shape != (observed.shape[0], observed.shape[1]):
        raise ValueError("channel_mask must have shape (C,) or (B,C)")
    if value.dtype != torch.bool:
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("numeric channel_mask must contain only 0/1")
        value = value.bool()
    if not bool(value.any(dim=1).all()):
        raise ValueError("each sample must retain at least one montage channel")
    return value.detach()


class ArtifactLatentDiffusion(nn.Module):
    """Masked operator-conditioned v-prediction diffusion model."""

    visible_input_fields = (
        "observed_query_EEG",
        "full_support_transfer_C",
        "normalized_support_transfer_C",
        "transfer_scale",
        "singular_values_and_rank",
        "support_only_reliability_rho",
        "calibration_duration",
        "layout_reference_channel_mask",
        "valid_time_mask",
    )
    forbidden_input_fields = (
        "participant_identity",
        "query_EOG",
        "query_eye_tracking",
        "query_artifact_label",
        "query_outcome",
        "query_clean_target",
    )

    def __init__(
        self,
        model_config: ArtifactLatentModelConfig,
        diffusion_config: ArtifactLatentDiffusionConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        conditioning = artifact_conditioning_channels(model_config)
        self.conditioning_channels = conditioning
        backbone = ModelConfig(
            in_channels=model_config.latent_channels + conditioning,
            out_channels=model_config.latent_channels,
            signal_length=model_config.signal_length,
            base_channels=model_config.base_channels,
            channel_mults=list(model_config.channel_mults),
            num_res_blocks=model_config.num_res_blocks,
            groupnorm_groups=model_config.groupnorm_groups,
            dropout=model_config.dropout,
            time_sinusoidal_dim=model_config.time_sinusoidal_dim,
            time_embed_dim=model_config.time_embed_dim,
            attention_length=model_config.attention_length,
            attention_heads=model_config.attention_heads,
        )
        self.unet = UNet1D(backbone, subject_conditioned=False)
        betas, alphas_cumprod = cosine_alpha_bar(
            diffusion_config.num_timesteps,
            offset=diffusion_config.cosine_offset,
        )
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )

    @property
    def num_timesteps(self) -> int:
        return int(self.diffusion_config.num_timesteps)

    def _validate_latent(self, value: Tensor, *, name: str) -> None:
        if value.ndim != 3:
            raise ValueError(
                f"{name} must have shape (B,{self.model_config.latent_channels},"
                f"{self.model_config.signal_length})"
            )
        expected = (
            value.shape[0],
            self.model_config.latent_channels,
            self.model_config.signal_length,
        )
        if tuple(value.shape) != expected:
            raise ValueError(
                f"{name} must have shape (B,{self.model_config.latent_channels},"
                f"{self.model_config.signal_length})"
            )
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite floating point")

    def _validate_observed(self, observed: Tensor) -> None:
        expected_tail = (
            self.model_config.eeg_channels,
            self.model_config.signal_length,
        )
        if observed.ndim != 3 or tuple(observed.shape[1:]) != expected_tail:
            raise ValueError("observed EEG shape does not match the frozen model montage")
        if not observed.dtype.is_floating_point or not bool(
            torch.isfinite(observed).all()
        ):
            raise ValueError("observed EEG must be finite floating point")

    def _validate_timestep(self, timestep: Tensor, batch: int, device: torch.device) -> None:
        if timestep.shape != (batch,) or timestep.dtype != torch.long:
            raise ValueError("timestep must be a (B,) long tensor")
        if timestep.device != device:
            raise ValueError("timestep and latent must share a device")
        if bool((timestep < 0).any()) or bool((timestep >= self.num_timesteps).any()):
            raise ValueError("diffusion timestep is outside the cosine schedule")

    @staticmethod
    def _mask_latent(latent: Tensor, mask: Tensor) -> Tensor:
        return latent * mask.to(dtype=latent.dtype)

    def q_sample(self, x0: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        self._validate_latent(x0, name="standardized artifact latent")
        self._validate_latent(noise, name="artifact diffusion noise")
        self._validate_timestep(timestep, x0.shape[0], x0.device)
        return (
            _extract(self.sqrt_alphas_cumprod, timestep, x0.ndim) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, timestep, x0.ndim)
            * noise
        )

    def v_target(self, x0: Tensor, noise: Tensor, timestep: Tensor) -> Tensor:
        self._validate_latent(x0, name="standardized artifact latent")
        self._validate_latent(noise, name="artifact diffusion noise")
        self._validate_timestep(timestep, x0.shape[0], x0.device)
        return (
            _extract(self.sqrt_alphas_cumprod, timestep, x0.ndim) * noise
            - _extract(self.sqrt_one_minus_alphas_cumprod, timestep, x0.ndim)
            * x0
        )

    def x0_and_epsilon_from_v(
        self, x_t: Tensor, predicted_v: Tensor, timestep: Tensor
    ) -> tuple[Tensor, Tensor]:
        self._validate_latent(x_t, name="noisy artifact latent")
        self._validate_latent(predicted_v, name="predicted artifact v")
        self._validate_timestep(timestep, x_t.shape[0], x_t.device)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timestep, x_t.ndim)
        sqrt_one_minus = _extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, x_t.ndim
        )
        predicted_x0 = sqrt_alpha * x_t - sqrt_one_minus * predicted_v
        predicted_epsilon = sqrt_one_minus * x_t + sqrt_alpha * predicted_v
        return predicted_x0, predicted_epsilon

    def predict_v(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        *,
        observed: Tensor,
        full_transfer: Tensor,
        normalized_transfer: Tensor,
        transfer_scale: Tensor,
        singular_values: Tensor,
        rank: int | Tensor,
        rho: float | Tensor,
        calibration_duration_seconds: float | Tensor,
        channel_mask: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self._validate_latent(noisy_latent, name="noisy artifact latent")
        self._validate_observed(observed)
        self._validate_timestep(timestep, noisy_latent.shape[0], noisy_latent.device)
        features, mask = build_artifact_conditioning(
            observed,
            full_transfer=full_transfer,
            normalized_transfer=normalized_transfer,
            transfer_scale=transfer_scale,
            singular_values=singular_values,
            rank=rank,
            rho=rho,
            calibration_duration_seconds=calibration_duration_seconds,
            channel_mask=channel_mask,
            valid_time_mask=valid_time_mask,
        )
        if features.device != noisy_latent.device or features.dtype != noisy_latent.dtype:
            raise ValueError("artifact conditioning and latent must share device/dtype")
        if features.shape[1] != self.conditioning_channels:
            raise ValueError("conditioning width differs from diffusion model config")
        latent_mask = mask.to(dtype=noisy_latent.dtype)
        value = torch.cat((noisy_latent * latent_mask, features), dim=1)
        return self.unet(value, timestep, valid_time_mask=mask) * latent_mask

    def training_loss(
        self,
        standardized_artifact_latent: Tensor,
        *,
        observed: Tensor,
        full_transfer: Tensor,
        normalized_transfer: Tensor,
        transfer_scale: Tensor,
        singular_values: Tensor,
        rank: int | Tensor,
        rho: float | Tensor,
        calibration_duration_seconds: float | Tensor,
        channel_mask: Tensor,
        valid_time_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        self._validate_latent(
            standardized_artifact_latent, name="standardized artifact latent"
        )
        _, mask = build_artifact_conditioning(
            observed,
            full_transfer=full_transfer,
            normalized_transfer=normalized_transfer,
            transfer_scale=transfer_scale,
            singular_values=singular_values,
            rank=rank,
            rho=rho,
            calibration_duration_seconds=calibration_duration_seconds,
            channel_mask=channel_mask,
            valid_time_mask=valid_time_mask,
        )
        mask_float = mask.to(dtype=standardized_artifact_latent.dtype)
        x0 = standardized_artifact_latent * mask_float
        batch = x0.shape[0]
        if timestep is None:
            timestep = torch.randint(
                0,
                self.num_timesteps,
                (batch,),
                device=x0.device,
                dtype=torch.long,
                generator=generator,
            )
        self._validate_timestep(timestep, batch, x0.device)
        if noise is None:
            noise = torch.randn(
                x0.shape,
                device=x0.device,
                dtype=x0.dtype,
                generator=generator,
            )
        self._validate_latent(noise, name="artifact diffusion noise")
        noise = noise * mask_float
        x_t = self.q_sample(x0, timestep, noise) * mask_float
        target = self.v_target(x0, noise, timestep) * mask_float
        prediction = self.predict_v(
            x_t,
            timestep,
            observed=observed,
            full_transfer=full_transfer,
            normalized_transfer=normalized_transfer,
            transfer_scale=transfer_scale,
            singular_values=singular_values,
            rank=rank,
            rho=rho,
            calibration_duration_seconds=calibration_duration_seconds,
            channel_mask=channel_mask,
            valid_time_mask=mask,
        )
        squared = (prediction - target).square() * mask_float
        alpha = _extract(self.alphas_cumprod, timestep, x0.ndim)
        snr = alpha / (1.0 - alpha).clamp_min(1.0e-8)
        weight = torch.minimum(
            snr,
            torch.full_like(snr, self.diffusion_config.min_snr_gamma),
        ) / (snr + 1.0)
        denominator = (
            mask_float.sum() * self.model_config.latent_channels
        ).clamp_min(1.0)
        loss = (squared * weight).sum() / denominator
        predicted_x0, _ = self.x0_and_epsilon_from_v(x_t, prediction, timestep)
        x0_mse = ((predicted_x0 - x0).square() * mask_float).sum() / denominator
        return loss, {
            "v_mse": (squared.sum() / denominator).detach(),
            "x0_mse": x0_mse.detach(),
            "mean_min_snr_weight": weight.mean().detach(),
            "mean_timestep": timestep.float().mean().detach(),
        }

    def _dynamic_threshold(self, predicted_x0: Tensor, mask: Tensor) -> tuple[Tensor, float]:
        valid = predicted_x0.masked_select(mask.expand_as(predicted_x0))
        if valid.numel() < 1:
            raise ValueError("dynamic threshold requires valid artifact samples")
        quantile = torch.quantile(
            valid.abs().float(), self.diffusion_config.dynamic_threshold_quantile
        ).to(dtype=predicted_x0.dtype)
        threshold = quantile.clamp(
            min=1.0,
            max=self.diffusion_config.standardized_latent_absolute_clip,
        )
        clipped = predicted_x0.clamp(-threshold, threshold)
        clipped_fraction = float(
            ((predicted_x0.abs() > threshold) & mask.expand_as(predicted_x0))
            .float()
            .sum()
            .div(mask.expand_as(predicted_x0).sum().clamp_min(1))
            .detach()
            .cpu()
        )
        return clipped * mask.to(dtype=clipped.dtype), clipped_fraction

    @staticmethod
    def _timestep_sequence(num_timesteps: int, ddim_steps: int) -> tuple[int, ...]:
        if isinstance(ddim_steps, bool) or int(ddim_steps) != ddim_steps:
            raise ValueError("DDIM steps must be an integer")
        count = int(ddim_steps)
        if not 2 <= count <= num_timesteps:
            raise ValueError("DDIM steps must lie in [2,T]")
        values = (
            torch.linspace(num_timesteps - 1, 0, count, dtype=torch.float64)
            .round()
            .long()
            .tolist()
        )
        if len(set(values)) != count or any(
            left <= right for left, right in zip(values, values[1:])
        ):
            raise AssertionError("DDIM timestep sequence is not strictly decreasing")
        return tuple(int(value) for value in values)

    @torch.no_grad()
    def posterior_mean(
        self,
        *,
        observed: Tensor,
        full_transfer: Tensor,
        normalized_transfer: Tensor,
        transfer_scale: Tensor,
        singular_values: Tensor,
        rank: int | Tensor,
        rho: float | Tensor,
        calibration_duration_seconds: float | Tensor,
        channel_mask: Tensor,
        latent_mean: Tensor,
        latent_standard_deviation: Tensor,
        valid_time_mask: Optional[Tensor],
        sample_seeds: Sequence[int],
        ddim_steps: int,
        record_trajectory: bool = False,
    ) -> ArtifactPosterior:
        """Return the arithmetic mean of exactly eight posterior samples.

        No target, EOG, label, metric, or sample-selection argument exists.
        """

        self._validate_observed(observed)
        seeds = tuple(int(value) for value in sample_seeds)
        if len(seeds) != self.diffusion_config.posterior_samples:
            raise ValueError("posterior point estimate requires exactly K=8 seeds")
        if len(set(seeds)) != len(seeds):
            raise ValueError("posterior sample seeds must be unique")
        features, mask = build_artifact_conditioning(
            observed,
            full_transfer=full_transfer,
            normalized_transfer=normalized_transfer,
            transfer_scale=transfer_scale,
            singular_values=singular_values,
            rank=rank,
            rho=rho,
            calibration_duration_seconds=calibration_duration_seconds,
            channel_mask=channel_mask,
            valid_time_mask=valid_time_mask,
        )
        if features.shape[1] != self.conditioning_channels:
            raise ValueError("conditioning width differs from diffusion model config")
        del features
        mask_float = mask.to(dtype=observed.dtype)
        batch, _, length = observed.shape
        latent_channels = self.model_config.latent_channels
        transfer = _posterior_transfer(
            normalized_transfer,
            observed=observed,
            latent_channels=latent_channels,
        )
        mean = _posterior_latent_stat(
            latent_mean,
            observed=observed,
            latent_channels=latent_channels,
            name="latent_mean",
            strictly_positive=False,
        )
        standard_deviation = _posterior_latent_stat(
            latent_standard_deviation,
            observed=observed,
            latent_channels=latent_channels,
            name="latent_standard_deviation",
            strictly_positive=True,
        )
        available_channels = _posterior_channel_mask(observed, channel_mask)
        output_mask = (
            available_channels[:, :, None].to(dtype=observed.dtype) * mask_float
        )
        latent_shape = (batch, self.model_config.latent_channels, length)
        ts = self._timestep_sequence(self.num_timesteps, int(ddim_steps))
        samples: list[Tensor] = []
        traces: list[ArtifactTrajectoryStep] = []
        network_calls = 0
        for sample_index, raw_seed in enumerate(seeds):
            generator = torch.Generator(device=observed.device)
            generator.manual_seed(int(raw_seed))
            latent = torch.randn(
                latent_shape,
                device=observed.device,
                dtype=observed.dtype,
                generator=generator,
            ) * mask_float
            previous_rms: float | None = None
            for reverse_index, timestep_value in enumerate(ts):
                timestep = torch.full(
                    (batch,),
                    timestep_value,
                    device=observed.device,
                    dtype=torch.long,
                )
                predicted_v = self.predict_v(
                    latent,
                    timestep,
                    observed=observed,
                    full_transfer=full_transfer,
                    normalized_transfer=normalized_transfer,
                    transfer_scale=transfer_scale,
                    singular_values=singular_values,
                    rank=rank,
                    rho=rho,
                    calibration_duration_seconds=calibration_duration_seconds,
                    channel_mask=channel_mask,
                    valid_time_mask=mask,
                )
                network_calls += 1
                predicted_x0, predicted_epsilon = self.x0_and_epsilon_from_v(
                    latent, predicted_v, timestep
                )
                predicted_x0, clipped_fraction = self._dynamic_threshold(
                    predicted_x0, mask
                )
                if reverse_index == len(ts) - 1:
                    next_latent = predicted_x0
                else:
                    next_alpha = self.alphas_cumprod[ts[reverse_index + 1]]
                    next_latent = (
                        torch.sqrt(next_alpha) * predicted_x0
                        + torch.sqrt(1.0 - next_alpha) * predicted_epsilon
                    ) * mask_float
                valid_count = (
                    mask_float.sum() * self.model_config.latent_channels
                ).clamp_min(1.0)
                latent_rms = float(
                    torch.sqrt((latent.square() * mask_float).sum() / valid_count)
                    .detach()
                    .cpu()
                )
                next_rms = float(
                    torch.sqrt((next_latent.square() * mask_float).sum() / valid_count)
                    .detach()
                    .cpu()
                )
                if record_trajectory:
                    mapped = canonical_artifact_delta(
                        predicted_x0,
                        normalized_transfer=transfer,
                        latent_mean=mean,
                        latent_standard_deviation=standard_deviation,
                        output_mask=output_mask,
                    )
                    traces.append(
                        ArtifactTrajectoryStep(
                            sample_index=sample_index,
                            reverse_index=reverse_index,
                            timestep=timestep_value,
                            latent_rms=latent_rms,
                            predicted_v_rms=float(
                                torch.sqrt(
                                    (predicted_v.square() * mask_float).sum()
                                    / valid_count
                                )
                                .detach()
                                .cpu()
                            ),
                            predicted_x0_rms=float(
                                torch.sqrt(
                                    (predicted_x0.square() * mask_float).sum()
                                    / valid_count
                                )
                                .detach()
                                .cpu()
                            ),
                            mapped_contamination_rms=float(
                                torch.sqrt(
                                    mapped.square().sum()
                                    / output_mask.sum().clamp_min(1.0)
                                )
                                .detach()
                                .cpu()
                            ),
                            adjacent_latent_rms_ratio=(
                                None
                                if previous_rms is None
                                else next_rms / max(previous_rms, 1.0e-12)
                            ),
                            finite=bool(
                                torch.isfinite(next_latent).all()
                                and torch.isfinite(predicted_v).all()
                                and torch.isfinite(predicted_x0).all()
                            ),
                            clipped_fraction=clipped_fraction,
                        )
                    )
                previous_rms = next_rms
                latent = next_latent
            samples.append(latent * mask_float)
        latent_average, latent_std = artifact_posterior_point_estimate(samples)
        latent_average = latent_average * mask_float
        latent_std = latent_std * mask_float
        latent_samples = torch.stack(
            tuple(value.detach() for value in samples), dim=0
        ) * mask_float[None, :, :, :]
        if latent_samples.shape != (len(seeds), *latent_average.shape):
            raise AssertionError("posterior samples lost their explicit K dimension")
        if not torch.allclose(
            latent_samples.mean(dim=0),
            latent_average,
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise AssertionError("posterior point estimate is not the K=8 arithmetic mean")
        correction_from_latent_mean = canonical_artifact_delta(
            latent_average,
            normalized_transfer=transfer,
            latent_mean=mean,
            latent_standard_deviation=standard_deviation,
            output_mask=output_mask,
        )
        correction_samples = torch.stack(
            tuple(
                canonical_artifact_delta(
                    sample,
                    normalized_transfer=transfer,
                    latent_mean=mean,
                    latent_standard_deviation=standard_deviation,
                    output_mask=output_mask,
                )
                for sample in latent_samples
            ),
            dim=0,
        )
        correction_samples = correction_samples.detach()
        # The published point output is the arithmetic mean in EEG space.  In
        # poorly scaled cells, applying the affine physical-coordinate decoder
        # after averaging can differ from averaging decoded samples by more
        # than a fixed near-zero ``allclose`` tolerance solely due to floating
        # point cancellation.  Materialize the contractual EEG-space mean and
        # retain a scale-aware check that the affine identity still holds.
        correction = correction_samples.mean(dim=0)
        decoder_scale = torch.maximum(
            correction.abs().amax(), correction_from_latent_mean.abs().amax()
        ).clamp_min(1.0)
        decoder_error = (correction - correction_from_latent_mean).abs().amax()
        if bool(decoder_error > 5.0e-5 * decoder_scale):
            raise AssertionError("correction is not the K=8 arithmetic mean")
        restored = (observed * output_mask - correction) * output_mask
        if not bool(torch.isfinite(restored).all()):
            raise FloatingPointError("artifact-latent posterior produced non-finite EEG")
        return ArtifactPosterior(
            standardized_latent_mean=latent_average,
            standardized_latent_standard_deviation=latent_std,
            correction=correction,
            restored=restored,
            sample_count=len(samples),
            network_calls=network_calls,
            trajectories=tuple(traces),
            standardized_latent_samples=latent_samples.detach(),
            correction_samples=correction_samples,
        )


__all__ = [
    "ArtifactLatentDiffusion",
    "ArtifactLatentDiffusionConfig",
    "ArtifactPosterior",
    "ArtifactTrajectoryStep",
    "artifact_posterior_point_estimate",
    "cosine_alpha_bar",
]
