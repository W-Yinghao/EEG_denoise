"""Information-matched deterministic artifact-latent estimator.

The estimator predicts a standardized, low-dimensional artifact trajectory
``z`` from the observed EEG and one frozen operator context.  The context
contains both the complete EOG-coordinate transfer matrix ``C`` and its
column-normalized form ``C_norm`` with ``C = C_norm diag(scale)``.  The
physical latent is ``A = sigma_Z z + mu_Z`` and restoration is exactly
``X_hat = Y - C_normalized A`` on valid channels and time points.  The
``ArtifactLatentContext``/``restore`` surface is retained only as a deprecated
compatibility adapter; production uses :mod:`artifact_latent_inference`.
retained rank is a conditioning diagnostic; it never truncates the external
EOG-coordinate transfer or the predicted latent.

Population, matching, wrong and shuffled operators are runtime contexts for
one shared network.  They never select parameters or trigger context-specific
training.  :meth:`restore` accepts a lazy context factory so ``rho=0`` can
return through the population path before constructing or validating any
calibration-derived context.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .clean_prior import canonical_valid_time_mask
from .artifact_latent_inference import (
    canonical_artifact_delta,
    canonical_physical_artifact_latent,
    mix_population_subject_delta,
)


@dataclass(frozen=True)
class ArtifactLatentModelConfig:
    """Frozen montage and backbone dimensions for the latent estimator."""

    eeg_channels: int
    signal_length: int
    latent_channels: int = 2
    base_channels: int = 64
    channel_mults: tuple[int, int, int] = (1, 2, 4)
    num_res_blocks: int = 2
    groupnorm_groups: int = 8
    dropout: float = 0.05
    time_sinusoidal_dim: int = 128
    time_embed_dim: int = 512
    attention_length: int = 64
    attention_heads: int = 4

    def __post_init__(self) -> None:
        if self.eeg_channels < 2:
            raise ValueError("artifact-latent estimation requires multichannel EEG")
        if (
            isinstance(self.latent_channels, bool)
            or int(self.latent_channels) != self.latent_channels
        ):
            raise ValueError("latent_channels must be an integer")
        if not 1 <= self.latent_channels <= self.eeg_channels:
            raise ValueError("latent_channels must lie in [1,eeg_channels]")
        if self.signal_length < 8 or self.signal_length % 8:
            raise ValueError("signal_length must be a positive multiple of eight")
        if tuple(self.channel_mults) != (1, 2, 4):
            raise ValueError("the shared U-Net requires channel_mults (1,2,4)")
        if self.base_channels < 1 or self.groupnorm_groups < 1:
            raise ValueError("base_channels and groupnorm_groups must be positive")
        if self.attention_length < 1 or self.attention_heads < 1:
            raise ValueError("attention dimensions must be positive")
        widths = tuple(self.base_channels * value for value in self.channel_mults)
        if any(value % self.groupnorm_groups for value in widths):
            raise ValueError("all U-Net widths must be divisible by groupnorm_groups")
        if widths[-1] % self.attention_heads:
            raise ValueError("bottleneck width must be divisible by attention_heads")


@dataclass(frozen=True)
class ArtifactLatentContext:
    """Deprecated compatibility context; use ``ArtifactInferenceContext``.

    Tensors may be shared across the batch with shapes ``(C,E)``/``(E,)`` or
    batched with shapes ``(B,C,E)``/``(B,E)``, where ``E`` is the number of
    available external EOG coordinates.  All ``E`` columns remain present
    even when the projector retains a lower-dimensional subspace.
    """

    full_transfer: Tensor
    normalized_transfer: Tensor
    transfer_scale: Tensor
    singular_values: Tensor
    rank: int | Tensor
    calibration_duration_seconds: float | Tensor
    latent_mean: Tensor
    latent_standard_deviation: Tensor


@dataclass(frozen=True)
class ArtifactLatentEstimate:
    """Deprecated compatibility result in the canonical physical coordinate."""

    standardized_latent: Tensor
    latent: Tensor
    predicted_contamination: Tensor
    restored: Tensor
    retained_rank_mask: Tensor
    rho: float
    population_short_circuit: bool
    context_branch_used: bool


@dataclass(frozen=True)
class ArtifactLatentConditioning:
    """Validated information stack shared by deterministic and diffusion arms."""

    features: Tensor
    valid_time_mask: Tensor
    full_transfer: Tensor
    normalized_transfer: Tensor
    transfer_scale: Tensor
    singular_values: Tensor
    ranks: Tensor
    retained_rank_mask: Tensor
    channel_mask: Tensor
    rho: Tensor
    calibration_duration_seconds: Tensor


def _batch_matrix(
    value: Tensor,
    *,
    observed: Tensor,
    channels: int,
    latent_channels: int,
    label: str,
) -> Tensor:
    matrix = torch.as_tensor(
        value,
        device=observed.device,
        dtype=observed.dtype,
    ).detach()
    if matrix.shape == (channels, latent_channels):
        matrix = matrix.unsqueeze(0).expand(observed.shape[0], -1, -1)
    if matrix.shape != (observed.shape[0], channels, latent_channels):
        raise ValueError(
            f"{label} must have shape (C,E) or (B,C,E); got {tuple(matrix.shape)}"
        )
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{label} contains non-finite values")
    return matrix


def _batch_vector(
    value: Tensor,
    *,
    observed: Tensor,
    latent_channels: int,
    label: str,
) -> Tensor:
    vector = torch.as_tensor(
        value,
        device=observed.device,
        dtype=observed.dtype,
    ).detach()
    if vector.shape == (latent_channels,):
        vector = vector.unsqueeze(0).expand(observed.shape[0], -1)
    if vector.shape != (observed.shape[0], latent_channels):
        raise ValueError(
            f"{label} must have shape (E,) or (B,E); got {tuple(vector.shape)}"
        )
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _batch_ranks(value: int | Tensor, *, observed: Tensor, max_rank: int) -> Tensor:
    ranks = torch.as_tensor(value, device=observed.device)
    if ranks.ndim == 0:
        ranks = ranks.expand(observed.shape[0])
    if ranks.shape != (observed.shape[0],):
        raise ValueError("rank must be a scalar or a (B,) tensor")
    if ranks.dtype == torch.bool:
        raise ValueError("rank cannot be boolean")
    if ranks.dtype.is_floating_point:
        if not bool(torch.isfinite(ranks).all()) or not bool(
            (ranks == ranks.round()).all()
        ):
            raise ValueError("rank must contain finite integers")
    ranks = ranks.to(dtype=torch.long).detach()
    if bool(((ranks < 1) | (ranks > max_rank)).any()):
        raise ValueError("every retained rank must lie in [1,max_rank]")
    return ranks


def _channel_mask(observed: Tensor, channel_mask: Tensor) -> Tensor:
    value = torch.as_tensor(channel_mask, device=observed.device)
    channels = observed.shape[1]
    if value.shape == (channels,):
        value = value.unsqueeze(0).expand(observed.shape[0], -1)
    if value.shape != (observed.shape[0], channels):
        raise ValueError("channel_mask must have shape (C,) or (B,C)")
    if value.dtype != torch.bool:
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError("numeric channel_mask must contain only 0/1")
        value = value.bool()
    if not bool(value.any(dim=1).all()):
        raise ValueError("each sample must retain at least one montage channel")
    return value.detach()


def _rho(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("rho must be finite and lie in [0,1]")
    return result


def _batch_rho(value: float | Tensor, *, observed: Tensor) -> Tensor:
    result = torch.as_tensor(
        value,
        device=observed.device,
        dtype=observed.dtype,
    )
    if result.ndim == 0:
        result = result.expand(observed.shape[0])
    if result.shape != (observed.shape[0],):
        raise ValueError("rho must be scalar or have shape (B,)")
    if not bool(torch.isfinite(result).all()) or bool(
        ((result < 0.0) | (result > 1.0)).any()
    ):
        raise ValueError("rho must be finite and lie in [0,1]")
    return result.detach()


def _latent_channels_from_transfer(value: Tensor, *, observed: Tensor) -> int:
    matrix = torch.as_tensor(value)
    channels = observed.shape[1]
    if matrix.ndim == 2 and matrix.shape[0] == channels:
        latent_channels = int(matrix.shape[1])
    elif (
        matrix.ndim == 3
        and matrix.shape[0] == observed.shape[0]
        and matrix.shape[1] == channels
    ):
        latent_channels = int(matrix.shape[2])
    else:
        raise ValueError(
            "full_transfer must have shape (C,E) or (B,C,E) matching observed"
        )
    if not 1 <= latent_channels <= channels:
        raise ValueError("external EOG coordinate count must lie in [1,C]")
    return latent_channels


def _calibration_duration(
    value: float | Tensor,
    *,
    observed: Tensor,
) -> Tensor:
    duration = torch.as_tensor(
        value,
        device=observed.device,
        dtype=observed.dtype,
    )
    if duration.ndim == 0:
        duration = duration.expand(observed.shape[0])
    if duration.shape != (observed.shape[0],):
        raise ValueError("calibration_duration_seconds must be scalar or (B,)")
    if not bool(torch.isfinite(duration).all()) or bool((duration < 0.0).any()):
        raise ValueError("calibration duration must be finite and non-negative")
    return duration.detach()


def artifact_conditioning_channels(config: ArtifactLatentModelConfig) -> int:
    """Return the exact shared conditioning width for deterministic/diffusion."""

    if not isinstance(config, ArtifactLatentModelConfig):
        raise TypeError("config must be ArtifactLatentModelConfig")
    channels = config.eeg_channels
    latent = config.latent_channels
    # y, C_norm^T y, vec(C), vec(C_norm), scale, singular values,
    # retained-rank diagnostic, channel mask, rho, calibration duration and
    # valid time.
    return channels + latent + 2 * channels * latent + 3 * latent + channels + 3


def _prepare_artifact_conditioning(
    observed: Tensor,
    *,
    full_transfer: Tensor,
    normalized_transfer: Tensor,
    transfer_scale: Tensor,
    singular_values: Tensor,
    rank: int | Tensor,
    rho: float | Tensor,
    calibration_duration_seconds: float | Tensor,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
) -> ArtifactLatentConditioning:
    if observed.ndim != 3 or observed.shape[1] < 2:
        raise ValueError("observed EEG must have multichannel shape (B,C,L)")
    if not observed.dtype.is_floating_point or not bool(
        torch.isfinite(observed).all()
    ):
        raise ValueError("observed EEG must be finite floating point")
    channels = observed.shape[1]
    latent_channels = _latent_channels_from_transfer(
        full_transfer,
        observed=observed,
    )
    full = _batch_matrix(
        full_transfer,
        observed=observed,
        channels=channels,
        latent_channels=latent_channels,
        label="full_transfer",
    )
    normalized = _batch_matrix(
        normalized_transfer,
        observed=observed,
        channels=channels,
        latent_channels=latent_channels,
        label="normalized_transfer",
    )
    scale = _batch_vector(
        transfer_scale,
        observed=observed,
        latent_channels=latent_channels,
        label="transfer_scale",
    )
    singular = _batch_vector(
        singular_values,
        observed=observed,
        latent_channels=latent_channels,
        label="singular_values",
    )
    ranks = _batch_ranks(rank, observed=observed, max_rank=latent_channels)
    channels_available = _channel_mask(observed, channel_mask)
    valid_time = canonical_valid_time_mask(observed, valid_time_mask)
    rho_value = _batch_rho(rho, observed=observed)
    duration = _calibration_duration(
        calibration_duration_seconds,
        observed=observed,
    )
    if bool((channels_available.sum(dim=1) < ranks).any()):
        raise ValueError("retained rank exceeds available montage channels")
    retained = (
        torch.arange(latent_channels, device=observed.device)[None, :]
        < ranks[:, None]
    )
    if bool((scale <= 0.0).any()):
        raise ValueError("transfer_scale must remain strictly positive")
    if bool((singular < 0.0).any()):
        raise ValueError("singular values must be non-negative")
    if bool((singular[retained] <= 0.0).any()):
        raise ValueError("retained singular values must be positive")
    for index in range(observed.shape[0]):
        ordered = singular[index]
        if ordered.numel() > 1 and bool(
            (ordered[1:] > ordered[:-1] + 1.0e-8).any()
        ):
            raise ValueError("singular values must be non-increasing")
    expected_full = normalized * scale[:, None, :]
    if not torch.allclose(
        full.double(),
        expected_full.double(),
        atol=1.0e-7,
        rtol=1.0e-5,
    ):
        raise ValueError("full/normalized transfer and scale are inconsistent")

    channel_float = channels_available.to(dtype=observed.dtype)[:, :, None]
    full = full * channel_float
    normalized = normalized * channel_float
    time_float = valid_time.to(dtype=observed.dtype)
    masked_observed = observed * time_float * channel_float
    projected = torch.einsum("bcr,bct->brt", normalized, masked_observed)
    length = observed.shape[-1]

    def broadcast(vector: Tensor) -> Tensor:
        return vector[:, :, None].expand(-1, -1, length)

    full_features = full.flatten(start_dim=1)[:, :, None].expand(-1, -1, length)
    normalized_features = normalized.flatten(start_dim=1)[:, :, None].expand(
        -1, -1, length
    )
    retained_float = retained.to(dtype=observed.dtype)
    scale_features = torch.log1p(scale)
    singular_features = (
        singular
        / singular[:, :1].clamp_min(torch.finfo(observed.dtype).eps)
    )
    channel_features = channel_float.expand(-1, -1, length)
    rho_feature = rho_value[:, None, None].expand(-1, 1, length)
    duration_feature = torch.log1p(duration)[:, None, None].expand(-1, 1, length)
    features = torch.cat(
        (
            masked_observed,
            projected,
            full_features,
            normalized_features,
            broadcast(scale_features),
            broadcast(singular_features),
            broadcast(retained_float),
            channel_features,
            rho_feature,
            duration_feature,
            time_float,
        ),
        dim=1,
    ) * time_float
    return ArtifactLatentConditioning(
        features=features,
        valid_time_mask=valid_time,
        full_transfer=full,
        normalized_transfer=normalized,
        transfer_scale=scale,
        singular_values=singular,
        ranks=ranks,
        retained_rank_mask=retained,
        channel_mask=channels_available,
        rho=rho_value,
        calibration_duration_seconds=duration,
    )


def build_artifact_conditioning(
    observed: Tensor,
    *,
    full_transfer: Tensor,
    normalized_transfer: Tensor,
    transfer_scale: Tensor,
    singular_values: Tensor,
    rank: int | Tensor,
    rho: float | Tensor,
    calibration_duration_seconds: float | Tensor,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Build the exact public information stack shared with future diffusion."""

    prepared = _prepare_artifact_conditioning(
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
    return prepared.features, prepared.valid_time_mask


class DeterministicArtifactEstimator(nn.Module):
    """One shared network predicting only standardized artifact latent ``A``."""

    visible_input_fields = (
        "observed",
        "full_transfer",
        "normalized_transfer",
        "transfer_scale",
        "singular_values",
        "rank",
        "rho",
        "calibration_duration_seconds",
        "channel_mask",
        "valid_time_mask",
    )
    predicted_quantity = "standardized_artifact_latent_A"
    context_specific_parameters = False

    def __init__(self, config: ArtifactLatentModelConfig) -> None:
        super().__init__()
        self.config = config
        width = artifact_conditioning_channels(config)
        model_config = ModelConfig(
            in_channels=width,
            out_channels=config.latent_channels,
            signal_length=config.signal_length,
            base_channels=config.base_channels,
            channel_mults=list(config.channel_mults),
            num_res_blocks=config.num_res_blocks,
            groupnorm_groups=config.groupnorm_groups,
            dropout=config.dropout,
            time_sinusoidal_dim=config.time_sinusoidal_dim,
            time_embed_dim=config.time_embed_dim,
            attention_length=config.attention_length,
            attention_heads=config.attention_heads,
        )
        self.conditioning_channels = width
        self.unet = UNet1D(model_config, subject_conditioned=False)

    def _check_model_shape(self, observed: Tensor) -> None:
        if observed.ndim != 3 or observed.shape[1] != self.config.eeg_channels:
            raise ValueError("observed EEG shape does not match model montage")
        if observed.shape[-1] != self.config.signal_length:
            raise ValueError("observed EEG length differs from model config")

    def _predict(self, conditioning: ArtifactLatentConditioning) -> Tensor:
        if conditioning.features.shape[1] != self.conditioning_channels:
            raise ValueError("conditioning width differs from model config")
        condition = torch.zeros(
            conditioning.features.shape[0],
            dtype=torch.long,
            device=conditioning.features.device,
        )
        standardized = self.unet(
            conditioning.features,
            condition,
            valid_time_mask=conditioning.valid_time_mask,
        )
        expected = (
            conditioning.features.shape[0],
            self.config.latent_channels,
            conditioning.features.shape[-1],
        )
        if standardized.shape != expected:
            raise ValueError("latent backbone returned an unexpected shape")
        if not bool(torch.isfinite(standardized).all()):
            raise FloatingPointError("latent backbone returned NaN/Inf")
        return standardized * conditioning.valid_time_mask.to(
            dtype=standardized.dtype
        )

    def forward(
        self,
        observed: Tensor,
        *,
        full_transfer: Tensor,
        normalized_transfer: Tensor,
        transfer_scale: Tensor,
        singular_values: Tensor,
        rank: int | Tensor,
        rho: float | Tensor,
        calibration_duration_seconds: float | Tensor,
        channel_mask: Tensor,
        valid_time_mask: Tensor | None,
    ) -> Tensor:
        """Return standardized ``(B,E,L)`` artifact latent, never EEG."""

        self._check_model_shape(observed)
        prepared = _prepare_artifact_conditioning(
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
        return self._predict(prepared)

    def _estimate_context(
        self,
        observed: Tensor,
        *,
        context: ArtifactLatentContext,
        rho: float,
        channel_mask: Tensor,
        valid_time_mask: Tensor | None,
        population_short_circuit: bool,
        context_branch_used: bool,
    ) -> ArtifactLatentEstimate:
        if not isinstance(context, ArtifactLatentContext):
            raise TypeError("context must be ArtifactLatentContext")
        self._check_model_shape(observed)
        prepared = _prepare_artifact_conditioning(
            observed,
            full_transfer=context.full_transfer,
            normalized_transfer=context.normalized_transfer,
            transfer_scale=context.transfer_scale,
            singular_values=context.singular_values,
            rank=context.rank,
            rho=rho,
            calibration_duration_seconds=context.calibration_duration_seconds,
            channel_mask=channel_mask,
            valid_time_mask=valid_time_mask,
        )
        standardized = self._predict(prepared)
        latent = canonical_physical_artifact_latent(
            standardized,
            latent_mean=context.latent_mean,
            latent_standard_deviation=context.latent_standard_deviation,
            valid_time_mask=prepared.valid_time_mask,
        )
        output_mask = (
            prepared.valid_time_mask.to(dtype=observed.dtype)
            * prepared.channel_mask.to(dtype=observed.dtype)[:, :, None]
        )
        contamination = canonical_artifact_delta(
            standardized,
            normalized_transfer=prepared.normalized_transfer,
            latent_mean=context.latent_mean,
            latent_standard_deviation=context.latent_standard_deviation,
            output_mask=output_mask,
        )
        restored = (observed * output_mask - contamination) * output_mask
        return ArtifactLatentEstimate(
            standardized_latent=standardized,
            latent=latent,
            predicted_contamination=contamination,
            restored=restored,
            retained_rank_mask=prepared.retained_rank_mask,
            rho=rho,
            population_short_circuit=population_short_circuit,
            context_branch_used=context_branch_used,
        )

    def restore(
        self,
        observed: Tensor,
        *,
        population_context: ArtifactLatentContext,
        rho: float,
        context_factory: Callable[[], ArtifactLatentContext] | None,
        channel_mask: Tensor,
        valid_time_mask: Tensor | None,
    ) -> ArtifactLatentEstimate:
        """Deprecated adapter with lazy POP and physical-delta mixing."""

        warnings.warn(
            "DeterministicArtifactEstimator.restore is deprecated; use "
            "deterministic_population_subject_restore",
            DeprecationWarning,
            stacklevel=2,
        )

        rho_value = _rho(rho)
        if rho_value == 0.0:
            return self._estimate_context(
                observed,
                context=population_context,
                rho=0.0,
                channel_mask=channel_mask,
                valid_time_mask=valid_time_mask,
                population_short_circuit=True,
                context_branch_used=False,
            )
        if context_factory is None or not callable(context_factory):
            raise ValueError("non-zero rho requires a lazy context_factory")
        population = self._estimate_context(
            observed,
            context=population_context,
            rho=rho_value,
            channel_mask=channel_mask,
            valid_time_mask=valid_time_mask,
            population_short_circuit=False,
            context_branch_used=False,
        )
        subject = self._estimate_context(
            observed,
            context=context_factory(),
            rho=rho_value,
            channel_mask=channel_mask,
            valid_time_mask=valid_time_mask,
            population_short_circuit=False,
            context_branch_used=True,
        )
        mixed = mix_population_subject_delta(
            population.predicted_contamination,
            subject.predicted_contamination,
            rho_value,
        )
        output_mask = (
            canonical_valid_time_mask(observed, valid_time_mask).to(observed.dtype)
            * torch.as_tensor(channel_mask, device=observed.device)
            .bool()
            .to(observed.dtype)[None, :, None]
        )
        return ArtifactLatentEstimate(
            standardized_latent=subject.standardized_latent,
            latent=subject.latent,
            predicted_contamination=mixed,
            restored=(observed * output_mask - mixed) * output_mask,
            retained_rank_mask=subject.retained_rank_mask,
            rho=rho_value,
            population_short_circuit=False,
            context_branch_used=True,
        )

    def latent_training_loss(
        self,
        observed: Tensor,
        target_standardized_latent: Tensor,
        *,
        full_transfer: Tensor,
        normalized_transfer: Tensor,
        transfer_scale: Tensor,
        singular_values: Tensor,
        rank: int | Tensor,
        rho: float | Tensor,
        calibration_duration_seconds: float | Tensor,
        channel_mask: Tensor,
        valid_time_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Masked MSE for a registered standardized artifact-latent target."""

        predicted = self(
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
        target = torch.as_tensor(
            target_standardized_latent,
            device=observed.device,
            dtype=observed.dtype,
        )
        if target.shape != predicted.shape or not bool(torch.isfinite(target).all()):
            raise ValueError("target standardized latent has invalid shape or values")
        time = canonical_valid_time_mask(observed, valid_time_mask)
        weight = time.expand_as(predicted).to(dtype=observed.dtype)
        loss = ((predicted - target).square() * weight).sum()
        return loss / weight.sum().clamp_min(1.0), predicted


__all__ = [
    "ArtifactLatentConditioning",
    "ArtifactLatentContext",
    "ArtifactLatentEstimate",
    "ArtifactLatentModelConfig",
    "DeterministicArtifactEstimator",
    "artifact_conditioning_channels",
    "build_artifact_conditioning",
]
