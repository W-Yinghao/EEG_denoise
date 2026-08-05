"""Minimal models for the parallel subject-aware route screen.

The module deliberately keeps one canonical ocular latent.  Runtime operator
interventions change conditioning and the EEG reconstruction map, never the
training target.  Query EOG, labels, outcomes and participant identifiers are
absent from every inference surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from saddpm.models.film import FiLM
from saddpm.models.unet1d import (
    UNet1D,
    _apply_time_mask,
    _canonical_time_mask,
    _downsample_time_mask,
)

from .artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    build_artifact_conditioning,
)
from .artifact_latent_diffusion import ArtifactLatentDiffusion, ArtifactLatentDiffusionConfig
from .clean_prior import canonical_valid_time_mask


FORBIDDEN_QUERY_FIELDS = (
    "query_EOG",
    "query_eye_tracking",
    "query_artifact_label",
    "query_outcome",
    "participant_ID",
)


def canonical_target(target: Tensor, operator: Tensor | None = None) -> Tensor:
    """Return the precomputed target without consulting an intervention operator."""

    del operator
    if target.ndim != 3 or not target.dtype.is_floating_point:
        raise ValueError("canonical latent target must have shape (B,E,T)")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("canonical latent target contains non-finite values")
    return target


def full_c_population_residual_reconstruction(
    observed: Tensor,
    population_restored: Tensor,
    standardized_latent: Tensor,
    *,
    population_normalized_transfer: Tensor,
    subject_normalized_transfer: Tensor,
    latent_mean: Tensor,
    latent_standard_deviation: Tensor,
    valid_time_mask: Tensor,
    gain: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply ``delta0 + g (Cs-C0) a`` in the canonical physical coordinate."""

    if observed.shape != population_restored.shape or observed.ndim != 3:
        raise ValueError("observed and population output shapes differ")
    batch, channels, length = observed.shape
    latent_channels = standardized_latent.shape[1]
    expected = (batch, channels, latent_channels)
    c0 = torch.as_tensor(population_normalized_transfer, device=observed.device, dtype=observed.dtype)
    cs = torch.as_tensor(subject_normalized_transfer, device=observed.device, dtype=observed.dtype)
    if c0.shape == expected[1:]:
        c0 = c0[None].expand(batch, -1, -1)
    if cs.shape == expected[1:]:
        cs = cs[None].expand(batch, -1, -1)
    if c0.shape != expected or cs.shape != expected:
        raise ValueError("full transfer shape differs from the canonical latent")
    mean = torch.as_tensor(latent_mean, device=observed.device, dtype=observed.dtype)
    scale = torch.as_tensor(latent_standard_deviation, device=observed.device, dtype=observed.dtype)
    if mean.shape == (latent_channels,):
        mean = mean[None].expand(batch, -1)
    if scale.shape == (latent_channels,):
        scale = scale[None].expand(batch, -1)
    physical = standardized_latent * scale[:, :, None] + mean[:, :, None]
    mask = canonical_valid_time_mask(observed, valid_time_mask).to(observed.dtype)
    reliability = torch.as_tensor(gain, device=observed.device, dtype=observed.dtype)
    if reliability.ndim == 0:
        reliability = reliability.expand(batch)
    if reliability.shape != (batch,) or bool(((reliability < 0) | (reliability > 1)).any()):
        raise ValueError("gain must have shape (B,) and lie in [0,1]")
    subject_residual = torch.einsum("bce,bet->bct", cs - c0, physical)
    delta0 = observed - population_restored
    correction = (delta0 + reliability[:, None, None] * subject_residual) * mask
    return (observed - correction) * mask, correction


def support_summary(
    full_transfer: Tensor,
    population_transfer: Tensor,
    singular_values: Tensor,
    transfer_scale: Tensor,
    support_sample_count: Tensor,
    artifact_spectrum: Tensor,
) -> Tensor:
    """Create the full-C summary used by FiLM at every major residual block."""

    if full_transfer.ndim != 3 or population_transfer.shape != full_transfer.shape:
        raise ValueError("subject and population full transfers must have identical BxCxE shape")
    batch = full_transfer.shape[0]
    vectors = (
        full_transfer.flatten(1),
        (full_transfer - population_transfer).flatten(1),
        singular_values.flatten(1),
        transfer_scale.flatten(1),
        support_sample_count.reshape(batch, -1),
        artifact_spectrum.flatten(1),
    )
    value = torch.cat(vectors, dim=1)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("support summary contains non-finite values")
    return value


class _SummaryFiLMUNet(nn.Module):
    """UNet1D whose support summary modulates every residual block."""

    def __init__(self, base: UNet1D, summary_width: int, embedding_width: int) -> None:
        super().__init__()
        self.net = base
        self.encoder = nn.Sequential(
            nn.Linear(summary_width, embedding_width),
            nn.SiLU(),
            nn.Linear(embedding_width, embedding_width),
        )
        self.net.subject_conditioned = False
        self.net.subject_embed = None
        blocks = [*self.net.enc0, *self.net.enc1, *self.net.enc2, self.net.mid1, self.net.mid2,
                  *self.net.dec2, *self.net.dec1, *self.net.dec0]
        for block in blocks:
            block.film = FiLM(embedding_width, block.conv1.out_channels)

    def forward(self, x: Tensor, timestep: Tensor, summary: Tensor, valid_time_mask: Tensor) -> Tensor:
        mask0 = _canonical_time_mask(x, valid_time_mask)
        mask1 = _downsample_time_mask(mask0)
        mask2 = _downsample_time_mask(mask1)
        mask3 = _downsample_time_mask(mask2)
        temb = self.net.time_embed(timestep)
        context = self.encoder(summary)
        h = _apply_time_mask(self.net.stem(_apply_time_mask(x, mask0)), mask0)
        s0 = self.net._run(self.net.enc0, h, temb, context, mask0)
        s1 = self.net._run(self.net.enc1, self.net.down0(s0, mask0, mask1), temb, context, mask1)
        s2 = self.net._run(self.net.enc2, self.net.down1(s1, mask1, mask2), temb, context, mask2)
        h = self.net.down2(s2, mask2, mask3)
        h = self.net.mid1(h, temb, context, mask3)
        h = self.net.mid_attn(h, mask3)
        h = self.net.mid2(h, temb, context, mask3)
        h = self.net.up2(h, mask3, mask2)
        h = self.net._run(self.net.dec2, torch.cat((h, s2), dim=1), temb, context, mask2)
        h = self.net.up1(h, mask2, mask1)
        h = self.net._run(self.net.dec1, torch.cat((h, s1), dim=1), temb, context, mask1)
        h = self.net.up0(h, mask1, mask0)
        h = self.net._run(self.net.dec0, torch.cat((h, s0), dim=1), temb, context, mask0)
        h = self.net.out_act(self.net.out_norm(h, mask0))
        return _apply_time_mask(self.net.out_conv(h), mask0)


class FullCFiLMDiffusion(ArtifactLatentDiffusion):
    """Canonical latent diffusion with full-C FiLM in all residual blocks."""

    def __init__(
        self,
        model_config: ArtifactLatentModelConfig,
        diffusion_config: ArtifactLatentDiffusionConfig,
        *,
        population_transfer: Tensor,
        summary_extra_width: int = 9,
    ) -> None:
        super().__init__(model_config, diffusion_config)
        c0 = torch.as_tensor(population_transfer, dtype=torch.float32)
        if c0.shape != (model_config.eeg_channels, model_config.latent_channels):
            raise ValueError("population transfer differs from model montage")
        self.register_buffer("population_transfer", c0)
        summary_width = 2 * c0.numel() + 2 * model_config.latent_channels + summary_extra_width
        self.film_unet = _SummaryFiLMUNet(self.unet, summary_width, model_config.time_embed_dim)
        del self.unet
        self._runtime_support_sample_count: Tensor | None = None
        self._runtime_support_artifact_spectrum: Tensor | None = None

    @property
    def film_block_count(self) -> int:
        blocks = [*self.film_unet.net.enc0, *self.film_unet.net.enc1, *self.film_unet.net.enc2,
                  self.film_unet.net.mid1, self.film_unet.net.mid2,
                  *self.film_unet.net.dec2, *self.film_unet.net.dec1, *self.film_unet.net.dec0]
        return sum(block.film is not None for block in blocks)

    def predict_v(self, noisy_latent: Tensor, timestep: Tensor, **condition: Tensor) -> Tensor:
        observed = condition["observed"]
        features, mask = build_artifact_conditioning(
            observed,
            full_transfer=condition["full_transfer"],
            normalized_transfer=condition["normalized_transfer"],
            transfer_scale=condition["transfer_scale"],
            singular_values=condition["singular_values"],
            rank=condition["rank"],
            rho=condition["rho"],
            calibration_duration_seconds=condition["calibration_duration_seconds"],
            channel_mask=condition["channel_mask"],
            valid_time_mask=condition.get("valid_time_mask"),
        )
        full = torch.as_tensor(condition["full_transfer"], device=observed.device, dtype=observed.dtype)
        if full.ndim == 2:
            full = full[None].expand(observed.shape[0], -1, -1)
        population = self.population_transfer.to(observed)[None].expand_as(full)
        singular = torch.as_tensor(condition["singular_values"], device=observed.device, dtype=observed.dtype)
        scale = torch.as_tensor(condition["transfer_scale"], device=observed.device, dtype=observed.dtype)
        if singular.ndim == 1:
            singular = singular[None].expand(observed.shape[0], -1)
        if scale.ndim == 1:
            scale = scale[None].expand(observed.shape[0], -1)
        default_count: Tensor | float = 1.0 if self._runtime_support_sample_count is None else self._runtime_support_sample_count
        count = torch.as_tensor(condition.get("support_sample_count", default_count), device=observed.device, dtype=observed.dtype)
        if count.ndim == 0:
            count = count.expand(observed.shape[0])
        default_spectrum = torch.zeros(observed.shape[0], 8, device=observed.device) if self._runtime_support_artifact_spectrum is None else self._runtime_support_artifact_spectrum
        spectrum = torch.as_tensor(condition.get("support_artifact_spectrum", default_spectrum), device=observed.device, dtype=observed.dtype)
        summary = support_summary(full, population, singular, scale, count, spectrum)
        value = torch.cat((noisy_latent * mask.to(noisy_latent.dtype), features), dim=1)
        return self.film_unet(value, timestep, summary, mask) * mask.to(noisy_latent.dtype)

    def training_loss(
        self,
        standardized_artifact_latent: Tensor,
        *,
        support_sample_count: Tensor | None = None,
        support_artifact_spectrum: Tensor | None = None,
        **condition: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self._runtime_support_sample_count = support_sample_count
        self._runtime_support_artifact_spectrum = support_artifact_spectrum
        try:
            loss, diagnostics = super().training_loss(standardized_artifact_latent, **condition)
            return loss, dict(diagnostics)
        finally:
            self._runtime_support_sample_count = None
            self._runtime_support_artifact_spectrum = None

    def posterior_mean(
        self,
        *,
        support_sample_count: Tensor | None = None,
        support_artifact_spectrum: Tensor | None = None,
        **condition: Tensor,
    ):
        self._runtime_support_sample_count = support_sample_count
        self._runtime_support_artifact_spectrum = support_artifact_spectrum
        try:
            return super().posterior_mean(**condition)
        finally:
            self._runtime_support_sample_count = None
            self._runtime_support_artifact_spectrum = None


class AdaptiveActivityGate(nn.Module):
    """Query-EEG-only activity gate; EOG labels are training targets only."""

    forbidden_input_fields = FORBIDDEN_QUERY_FIELDS

    def __init__(self, eeg_channels: int, hidden: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(eeg_channels, hidden, 9, padding=4),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(self, observed: Tensor, valid_time_mask: Tensor) -> Tensor:
        mask = canonical_valid_time_mask(observed, valid_time_mask).to(observed.dtype)
        return torch.sigmoid(self.network(observed * mask)) * mask


def guided_latent_step(
    latent: Tensor,
    observed_coordinates: Tensor,
    *,
    strength: float,
    bound: float = 5.0,
) -> Tensor:
    """Bounded support-operator posterior guidance without query EOG."""

    value = float(strength)
    if not 0.0 <= value <= 1.0:
        raise ValueError("guidance strength must lie in [0,1]")
    anchor = observed_coordinates.clamp(-bound, bound)
    return ((1.0 - value) * latent + value * anchor).clamp(-bound, bound)


def sdedit_initial_latent(
    anchor: Tensor,
    alpha_bar: Tensor,
    noise: Tensor,
) -> Tensor:
    """Observation-anchored SDEdit initialization, never pure-noise generation."""

    alpha = torch.as_tensor(alpha_bar, device=anchor.device, dtype=anchor.dtype)
    if alpha.numel() != 1 or not 0.0 < float(alpha) < 1.0:
        raise ValueError("SDEdit alpha_bar must be scalar and lie in (0,1)")
    if noise.shape != anchor.shape:
        raise ValueError("SDEdit anchor/noise shapes differ")
    return alpha.sqrt() * anchor + (1.0 - alpha).sqrt() * noise


@dataclass(frozen=True)
class RouteTechnicalStatus:
    route: str
    finite: bool
    target_invariant: bool
    context_sensitive: bool
    checkpoint_reload: bool
    status: str


__all__ = [
    "AdaptiveActivityGate",
    "FullCFiLMDiffusion",
    "RouteTechnicalStatus",
    "canonical_target",
    "full_c_population_residual_reconstruction",
    "guided_latent_step",
    "sdedit_initial_latent",
    "support_summary",
]
