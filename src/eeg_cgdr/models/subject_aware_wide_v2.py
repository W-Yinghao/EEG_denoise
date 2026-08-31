"""Shared canonical-latent carriers for the wide v2 development screen."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D
from .artifact_latent_deterministic import build_artifact_conditioning
from .artifact_latent_diffusion import ArtifactLatentDiffusion


def canonical_eog_latent(
    standardized_artifact_latent: np.ndarray,
    latent_mean: np.ndarray,
    latent_standard_deviation: np.ndarray,
    transfer_scale: np.ndarray,
    valid_time_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one exact-cell EOG coordinate independent of decoder/operator.

    The existing latent is transfer-scaled ``Z``.  Dividing physical ``Z`` by
    the fitted transfer column scale recovers the standardized EOG coordinate
    used by all operator interventions.  A second training-only normalization
    makes it suitable as a shared model target.
    """

    z = np.asarray(standardized_artifact_latent, dtype=np.float64)
    mean = np.asarray(latent_mean, dtype=np.float64)
    std = np.asarray(latent_standard_deviation, dtype=np.float64)
    scale = np.asarray(transfer_scale, dtype=np.float64)
    valid = np.asarray(valid_time_mask, dtype=bool)
    if z.ndim != 3 or mean.shape != (z.shape[1],) or std.shape != mean.shape:
        raise ValueError("latent normalization shapes differ")
    if scale.shape != z.shape[:2] or valid.shape != (z.shape[0], z.shape[2]):
        raise ValueError("transfer scale/mask shapes differ from latent")
    if np.any(std <= 0) or np.any(scale <= 0):
        raise ValueError("latent and transfer scales must be positive")
    physical_z = z * std[None, :, None] + mean[None, :, None]
    eog = physical_z / scale[:, :, None]
    mask = valid[:, None]
    coordinate_mean = np.asarray([eog[:, channel][valid].mean() for channel in range(eog.shape[1])])
    coordinate_std = np.asarray([eog[:, channel][valid].std() for channel in range(eog.shape[1])])
    coordinate_std = np.maximum(coordinate_std, 1.0e-6)
    target = ((eog - coordinate_mean[None, :, None]) / coordinate_std[None, :, None]) * mask
    return target.astype(np.float32), coordinate_mean.astype(np.float32), coordinate_std.astype(np.float32)


def physical_eog_latent(target: Tensor, mean: Tensor, standard_deviation: Tensor) -> Tensor:
    if target.ndim != 3:
        raise ValueError("canonical target must have shape (B,E,T)")
    center = torch.as_tensor(mean, device=target.device, dtype=target.dtype)
    scale = torch.as_tensor(standard_deviation, device=target.device, dtype=target.dtype)
    if center.shape == (target.shape[1],):
        center = center[None].expand(target.shape[0], -1)
    if scale.shape == (target.shape[1],):
        scale = scale[None].expand(target.shape[0], -1)
    if center.shape != target.shape[:2] or scale.shape != target.shape[:2] or bool((scale <= 0).any()):
        raise ValueError("canonical latent normalization is invalid")
    return target * scale[:, :, None] + center[:, :, None]


def static_transfer_correction(transfer: Tensor, latent: Tensor, valid_time_mask: Tensor) -> Tensor:
    value = torch.as_tensor(transfer, device=latent.device, dtype=latent.dtype)
    if value.ndim == 2:
        value = value[None].expand(latent.shape[0], -1, -1)
    if value.shape[:1] != latent.shape[:1] or value.shape[2] != latent.shape[1]:
        raise ValueError("static transfer/latent shapes differ")
    mask = torch.as_tensor(valid_time_mask, device=latent.device).bool()
    if mask.shape != (latent.shape[0], latent.shape[2]):
        raise ValueError("valid-time mask differs from latent")
    return torch.einsum("bce,bet->bct", value, latent) * mask[:, None].to(latent.dtype)


def full_c_subject_residual(
    observed: Tensor,
    population_output: Tensor,
    canonical_latent: Tensor,
    population_transfer: Tensor,
    subject_transfer: Tensor,
    gate: Tensor | float,
    valid_time_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Add support-only full-C residual to a strong population correction."""

    if observed.shape != population_output.shape:
        raise ValueError("population output and observation shapes differ")
    population_delta = observed - population_output
    residual_transfer = torch.as_tensor(subject_transfer, device=observed.device, dtype=observed.dtype) - torch.as_tensor(population_transfer, device=observed.device, dtype=observed.dtype)
    residual = static_transfer_correction(residual_transfer, canonical_latent, valid_time_mask)
    reliability = torch.as_tensor(gate, device=observed.device, dtype=observed.dtype)
    if reliability.ndim == 0:
        reliability = reliability.expand(observed.shape[0])
    if reliability.ndim == 1:
        reliability = reliability[:, None, None]
    if reliability.shape not in ((observed.shape[0], 1, 1), (observed.shape[0], 1, observed.shape[2])):
        raise ValueError("support/activity gate has an invalid shape")
    if bool(((reliability < 0) | (reliability > 1)).any()):
        raise ValueError("support/activity gate must lie in [0,1]")
    correction = population_delta + reliability * residual
    return observed - correction, correction


def lazy_subject_residual(
    population_output: Tensor,
    gate: float,
    factory: Callable[[], tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor | None, bool]:
    """Guarantee g=0 short-circuits before subject construction or RNG use."""

    value = float(gate)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("support gate must lie in [0,1]")
    if value == 0.0:
        return population_output, None, False
    restored, correction = factory()
    return restored, correction, True


def fir_transfer_correction(
    fir_transfer: Tensor,
    canonical_latent: Tensor,
    lags: Sequence[int],
    valid_time_mask: Tensor,
) -> Tensor:
    transfer = torch.as_tensor(fir_transfer, device=canonical_latent.device, dtype=canonical_latent.dtype)
    if transfer.ndim == 3:
        transfer = transfer[None].expand(canonical_latent.shape[0], -1, -1, -1)
    if transfer.ndim != 4 or transfer.shape[2] != canonical_latent.shape[1] or transfer.shape[3] != len(lags):
        raise ValueError("FIR transfer must have shape (B,C,E,Lag)")
    shifted = []
    for lag in lags:
        if lag == 0:
            value = canonical_latent
        elif lag > 0:
            value = F.pad(canonical_latent[..., :-lag], (lag, 0))
        else:
            value = F.pad(canonical_latent[..., -lag:], (0, -lag))
        shifted.append(value)
    stack = torch.stack(shifted, dim=2)  # B,E,Lag,T
    correction = torch.einsum("bcel,belt->bct", transfer, stack)
    return correction * torch.as_tensor(valid_time_mask, device=correction.device).bool()[:, None].to(correction.dtype)


def activity_gate_from_eeg_latent(canonical_latent: Tensor, threshold: Tensor | float, temperature: float = 0.2) -> Tensor:
    if temperature <= 0:
        raise ValueError("activity-gate temperature must be positive")
    activity = torch.sqrt(canonical_latent.square().mean(dim=1, keepdim=True) + 1.0e-8)
    boundary = torch.as_tensor(threshold, device=activity.device, dtype=activity.dtype)
    return torch.sigmoid((activity - boundary) / float(temperature))


class SupportSummaryEncoder(nn.Module):
    """Small support summary encoder for FiLM/adapter conditioning."""

    def __init__(self, summary_features: int, embedding_features: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(summary_features, embedding_features), nn.SiLU(),
            nn.Linear(embedding_features, embedding_features),
        )

    def forward(self, summary: Tensor) -> Tensor:
        if summary.ndim != 2 or not bool(torch.isfinite(summary).all()):
            raise ValueError("support summary must have finite shape (B,F)")
        return self.network(summary)


class LowRankSupportAdapter(nn.Module):
    """Small support-only low-rank residual adapter; no participant IDs."""

    def __init__(self, channels: int, rank: int = 4) -> None:
        super().__init__()
        if rank < 1 or rank >= channels:
            raise ValueError("adapter rank must be in [1,channels)")
        self.down = nn.Conv1d(channels, rank, 1, bias=False)
        self.up = nn.Conv1d(rank, channels, 1, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor, scale: Tensor | float = 1.0) -> Tensor:
        factor = torch.as_tensor(scale, device=value.device, dtype=value.dtype)
        if factor.ndim == 1:
            factor = factor[:, None, None]
        return value + factor * self.up(self.down(value))


class SupportFiLMArtifactLatentDiffusion(ArtifactLatentDiffusion):
    """Artifact diffusion with support summaries injected in every ResBlock."""

    def __init__(self, model_config: object, diffusion_config: object, *, low_rank_adapter: bool = False) -> None:
        super().__init__(model_config, diffusion_config)  # type: ignore[arg-type]
        cfg = self.model_config
        backbone = ModelConfig(
            in_channels=cfg.latent_channels + self.conditioning_channels,
            out_channels=cfg.latent_channels,
            signal_length=cfg.signal_length,
            base_channels=cfg.base_channels,
            channel_mults=list(cfg.channel_mults),
            num_res_blocks=cfg.num_res_blocks,
            groupnorm_groups=cfg.groupnorm_groups,
            dropout=cfg.dropout,
            time_sinusoidal_dim=cfg.time_sinusoidal_dim,
            time_embed_dim=cfg.time_embed_dim,
            attention_length=cfg.attention_length,
            attention_heads=cfg.attention_heads,
            subject_embed_dim=128,
            num_subjects=1,
        )
        self.unet = UNet1D(backbone, subject_conditioned=True)
        self.unet.subject_embed = None
        # Per-coordinate full-C and residual-C moments, singular values,
        # transfer scales, reliability, and support duration.  These are all
        # support-only summaries; participant identity is never represented.
        self.support_encoder = SupportSummaryEncoder(cfg.latent_channels * 10 + 2, 128)
        self.output_adapter = LowRankSupportAdapter(cfg.latent_channels, rank=1) if low_rank_adapter else None
        self.register_buffer("population_transfer_context", torch.empty(0), persistent=False)

    def set_population_transfer(self, value: Tensor) -> None:
        transfer = torch.as_tensor(value, device=self.population_transfer_context.device, dtype=torch.float32)
        if transfer.ndim != 2 or transfer.shape[1] != self.model_config.latent_channels:
            raise ValueError("population transfer must have shape (C,E)")
        self.population_transfer_context = transfer.detach().clone()

    def _summary(self, condition: Mapping[str, Tensor], observed: Tensor) -> Tensor:
        value = torch.as_tensor(condition["full_transfer"], device=observed.device, dtype=observed.dtype)
        if value.ndim == 2:
            value = value[None].expand(observed.shape[0], -1, -1)
        if self.population_transfer_context.numel() == 0:
            raise RuntimeError("support FiLM population transfer was not initialized")
        population = self.population_transfer_context.to(device=observed.device, dtype=observed.dtype)[None].expand(observed.shape[0], -1, -1)
        if population.shape != value.shape:
            raise ValueError("population and support transfer shapes differ")
        residual = value - population
        fields = torch.cat((
            value.mean(1), value.std(1, unbiased=False),
            torch.linalg.vector_norm(value, dim=1), value.abs().amax(1),
            residual.mean(1), residual.std(1, unbiased=False),
            torch.linalg.vector_norm(residual, dim=1), residual.abs().amax(1),
            torch.as_tensor(condition["singular_values"], device=observed.device, dtype=observed.dtype),
            torch.as_tensor(condition["transfer_scale"], device=observed.device, dtype=observed.dtype),
        ), dim=1)
        reliability = torch.as_tensor(condition["rho"], device=observed.device, dtype=observed.dtype)
        if reliability.ndim == 0:
            reliability = reliability.expand(observed.shape[0])
        duration = torch.as_tensor(condition["calibration_duration_seconds"], device=observed.device, dtype=observed.dtype)
        if duration.ndim == 0:
            duration = duration.expand(observed.shape[0])
        return torch.cat((fields, reliability[:, None], duration[:, None] / 30.0), dim=1)

    def predict_v(self, noisy_latent: Tensor, timestep: Tensor, **condition: Tensor) -> Tensor:
        observed = condition["observed"]
        features, mask = build_artifact_conditioning(
            observed,
            full_transfer=condition["full_transfer"], normalized_transfer=condition["normalized_transfer"],
            transfer_scale=condition["transfer_scale"], singular_values=condition["singular_values"],
            rank=condition["rank"], rho=condition["rho"],
            calibration_duration_seconds=condition["calibration_duration_seconds"],
            channel_mask=condition["channel_mask"], valid_time_mask=condition.get("valid_time_mask"),
        )
        latent_mask = mask.to(noisy_latent.dtype)
        value = torch.cat((noisy_latent * latent_mask, features), dim=1)
        support = self.support_encoder(self._summary(condition, observed))
        output = self.unet.forward_with_subject_embedding(value, timestep, support, valid_time_mask=mask) * latent_mask
        if self.output_adapter is not None:
            output = self.output_adapter(output, torch.as_tensor(condition["rho"], device=output.device, dtype=output.dtype)) * latent_mask
        return output


class SupportLoRAArtifactLatentDiffusion(ArtifactLatentDiffusion):
    """Frozen population backbone with a calibration-fitted low-rank head.

    The adapter is deliberately absent from the conditioning features.  Its
    only subject-specific parameters are fitted on that context's support
    EEG/EOG while the population diffusion backbone stays frozen.
    """

    def __init__(self, model_config: object, diffusion_config: object) -> None:
        super().__init__(model_config, diffusion_config)  # type: ignore[arg-type]
        self.output_adapter = LowRankSupportAdapter(self.model_config.latent_channels, rank=1)

    def freeze_population_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.output_adapter.parameters():
            parameter.requires_grad_(True)

    def reset_support_adapter(self) -> None:
        nn.init.normal_(self.output_adapter.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_adapter.up.weight)

    def predict_v(self, noisy_latent: Tensor, timestep: Tensor, **condition: Tensor) -> Tensor:
        base = super().predict_v(noisy_latent, timestep, **condition)
        return self.output_adapter(base, torch.as_tensor(condition["rho"], device=base.device, dtype=base.dtype))


__all__ = [
    "LowRankSupportAdapter", "SupportSummaryEncoder", "activity_gate_from_eeg_latent",
    "canonical_eog_latent", "fir_transfer_correction", "full_c_subject_residual",
    "lazy_subject_residual", "physical_eog_latent", "static_transfer_correction",
    "SupportFiLMArtifactLatentDiffusion",
    "SupportLoRAArtifactLatentDiffusion",
]
