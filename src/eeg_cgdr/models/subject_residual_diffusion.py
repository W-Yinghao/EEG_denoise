"""Information-matched subject-conditioned EEG residual models.

Both learned comparators predict the same EEG-space residual relative to one
frozen population anchor.  The calibrated transfer is used only as a soft
conditioning variable; neither model receives query EOG or performs an
operator subtraction at inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .artifact_latent_diffusion import cosine_alpha_bar
from .clean_prior import canonical_valid_time_mask


def _extract(values: Tensor, timesteps: Tensor, ndim: int) -> Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (ndim - 1)))


@dataclass(frozen=True)
class SubjectResidualConfig:
    eeg_channels: int
    signal_length: int
    transfer_coordinates: int = 3
    context_features_per_channel: int = 2
    base_channels: int = 32
    num_timesteps: int = 1000
    cosine_offset: float = 0.008
    min_snr_gamma: float = 5.0
    context_dropout_probability: float = 0.25
    posterior_samples: int = 8
    ddim_steps: int = 50

    def __post_init__(self) -> None:
        if self.eeg_channels < 2 or self.signal_length < 8 or self.signal_length % 8:
            raise ValueError("subject residual model requires multichannel, 8-aligned EEG")
        if self.transfer_coordinates != 3 or self.context_features_per_channel != 2:
            raise ValueError("the frozen context encoder uses padded 3-coordinate transfers")
        if self.posterior_samples != 8 or self.ddim_steps != 50:
            raise ValueError("the mainline protocol freezes K=8 and DDIM50")
        if not 0.0 <= self.context_dropout_probability < 1.0:
            raise ValueError("context dropout must lie in [0,1)")
        _, alpha_bar = cosine_alpha_bar(self.num_timesteps, offset=self.cosine_offset)
        if float(alpha_bar[-1]) > 1.0e-4:
            raise ValueError("terminal alpha_bar must be at most 1e-4")


def _padded_transfer(value: Tensor, observed: Tensor, coordinates: int) -> Tensor:
    tensor = torch.as_tensor(value, device=observed.device, dtype=observed.dtype)
    if tensor.ndim != 3 or tensor.shape[:2] != observed.shape[:2]:
        raise ValueError("transfer must have shape (B,C,E)")
    if not 1 <= tensor.shape[2] <= coordinates or not bool(torch.isfinite(tensor).all()):
        raise ValueError("transfer coordinate count/value is invalid")
    if tensor.shape[2] < coordinates:
        tensor = torch.nn.functional.pad(tensor, (0, coordinates - tensor.shape[2]))
    return tensor.detach()


class SoftTransferContext(nn.Module):
    """Small per-channel encoder for ``(C0, Cs-C0, rho, masks)``."""

    def __init__(self, config: SubjectResidualConfig) -> None:
        super().__init__()
        width = 2 * config.transfer_coordinates + 3
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, config.context_features_per_channel)
        )

    def forward(
        self,
        observed: Tensor,
        *,
        population_transfer: Tensor,
        subject_transfer: Tensor,
        reliability: Tensor,
        channel_mask: Tensor,
        context_present: Tensor,
    ) -> Tensor:
        c0 = _padded_transfer(population_transfer, observed, self.config.transfer_coordinates)
        cs = _padded_transfer(subject_transfer, observed, self.config.transfer_coordinates)
        batch, channels, length = observed.shape
        rho = torch.as_tensor(reliability, device=observed.device, dtype=observed.dtype)
        present = torch.as_tensor(context_present, device=observed.device, dtype=observed.dtype)
        mask = torch.as_tensor(channel_mask, device=observed.device, dtype=observed.dtype)
        if rho.shape != (batch,) or present.shape != (batch,) or mask.shape != (batch, channels):
            raise ValueError("context reliability/presence/channel mask shape differs")
        if bool(((rho < 0) | (rho > 1)).any()) or not bool(torch.isfinite(rho).all()):
            raise ValueError("reliability must be finite in [0,1]")
        # Null/population context is exactly delta-C=0, while retaining the
        # query participant's reliability as required by the protocol.
        delta = (cs - c0) * present[:, None, None]
        features = torch.cat(
            [c0, delta, rho[:, None, None].expand(-1, channels, 1),
             mask[:, :, None], present[:, None, None].expand(-1, channels, 1)], dim=2
        )
        encoded = self.encoder(features) * mask[:, :, None]
        return encoded.reshape(batch, channels * self.config.context_features_per_channel, 1).expand(-1, -1, length)


class _ResidualBackbone(nn.Module):
    def __init__(self, config: SubjectResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.context = SoftTransferContext(config)
        inputs = 3 * config.eeg_channels + config.eeg_channels * config.context_features_per_channel
        self.unet = UNet1D(
            ModelConfig(
                in_channels=inputs,
                out_channels=config.eeg_channels,
                signal_length=config.signal_length,
                base_channels=config.base_channels,
                channel_mults=[1, 2, 4],
                num_res_blocks=2,
                groupnorm_groups=8,
                dropout=0.05,
                time_sinusoidal_dim=64,
                time_embed_dim=256,
                attention_length=64,
                attention_heads=4,
            ),
            subject_conditioned=False,
        )

    def forward(
        self,
        state: Tensor,
        timesteps: Tensor,
        *,
        observed: Tensor,
        population_anchor: Tensor,
        population_transfer: Tensor,
        subject_transfer: Tensor,
        reliability: Tensor,
        channel_mask: Tensor,
        context_present: Tensor,
        valid_time_mask: Tensor,
    ) -> Tensor:
        if state.shape != observed.shape or population_anchor.shape != observed.shape:
            raise ValueError("state, observation and population anchor must align")
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        context = self.context(
            observed,
            population_transfer=population_transfer,
            subject_transfer=subject_transfer,
            reliability=reliability,
            channel_mask=channel_mask,
            context_present=context_present,
        )
        features = torch.cat([state, observed, population_anchor, context], dim=1)
        return self.unet(features * mask.to(features.dtype), timesteps, valid_time_mask=mask) * mask.to(state.dtype)


class PopulationAnchor(nn.Module):
    """Population-only deterministic cleaner fitted on outer-training data."""

    def __init__(self, config: SubjectResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.unet = UNet1D(
            ModelConfig(
                in_channels=config.eeg_channels,
                out_channels=config.eeg_channels,
                signal_length=config.signal_length,
                base_channels=config.base_channels,
                channel_mults=[1, 2, 4],
                num_res_blocks=2,
                groupnorm_groups=8,
                dropout=0.05,
                time_sinusoidal_dim=64,
                time_embed_dim=256,
                attention_length=64,
                attention_heads=4,
            ),
            subject_conditioned=False,
        )

    def forward(self, observed: Tensor, valid_time_mask: Tensor) -> Tensor:
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        t = torch.zeros(observed.shape[0], dtype=torch.long, device=observed.device)
        return (observed + self.unet(observed * mask.to(observed.dtype), t, valid_time_mask=mask)) * mask.to(observed.dtype)


class BoundedResidual(nn.Module):
    """Training-only channel envelope, applied identically to both arms."""

    def __init__(self, channel_threshold: Tensor) -> None:
        super().__init__()
        threshold = torch.as_tensor(channel_threshold, dtype=torch.float32)
        if threshold.ndim != 1 or not bool(torch.isfinite(threshold).all()) or bool((threshold <= 0).any()):
            raise ValueError("residual thresholds must be positive finite channel values")
        self.register_buffer("channel_threshold", threshold)

    def forward(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        tau = self.channel_threshold[None, :, None].to(residual)
        bounded = tau * torch.tanh(residual / tau)
        fraction = (residual.abs() > tau).to(residual.dtype).mean()
        return bounded, fraction


class OneStepResidualEstimator(nn.Module):
    def __init__(self, config: SubjectResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _ResidualBackbone(config)

    def forward(self, *, observed: Tensor, population_anchor: Tensor, **context: Tensor) -> Tensor:
        state = torch.zeros_like(observed)
        timestep = torch.zeros(observed.shape[0], dtype=torch.long, device=observed.device)
        return self.backbone(state, timestep, observed=observed, population_anchor=population_anchor, **context)


class SubjectResidualDiffusion(nn.Module):
    visible_input_fields = ("observed_EEG", "population_anchor", "support_transfer_context", "support_reliability", "layout_channel_mask")
    forbidden_input_fields = ("query_EOG", "query_artifact_label", "query_outcome", "participant_ID")

    def __init__(self, config: SubjectResidualConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _ResidualBackbone(config)
        _, alpha_bar = cosine_alpha_bar(config.num_timesteps, offset=config.cosine_offset)
        self.register_buffer("alpha_bar", alpha_bar.float())

    def training_loss(self, target: Tensor, *, generator: torch.Generator | None = None, **condition: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        batch = target.shape[0]
        t = torch.randint(0, self.config.num_timesteps, (batch,), device=target.device, generator=generator)
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        alpha = _extract(self.alpha_bar, t, target.ndim)
        noisy = alpha.sqrt() * target + (1.0 - alpha).sqrt() * noise
        v_target = alpha.sqrt() * noise - (1.0 - alpha).sqrt() * target
        predicted = self.backbone(noisy, t, **condition)
        mask = canonical_valid_time_mask(target, condition["valid_time_mask"]).to(target.dtype)
        per = ((predicted - v_target) * mask).square().flatten(1).mean(1)
        snr = alpha.flatten(1)[:, 0] / (1.0 - alpha.flatten(1)[:, 0]).clamp_min(1e-8)
        weight = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma)) / (snr + 1.0)
        return (per * weight).mean(), {"timestep_mean": t.float().mean(), "raw_mse": per.mean()}

    def _timesteps(self) -> Tensor:
        values = torch.linspace(self.config.num_timesteps - 1, 0, self.config.ddim_steps, dtype=torch.float64).round().long()
        values = torch.unique_consecutive(values)
        if values.numel() != self.config.ddim_steps:
            raise AssertionError("DDIM50 sequence lost network calls")
        return values

    @torch.no_grad()
    def sample(self, *, shape: tuple[int, int, int], sample_seeds: tuple[int, ...], **condition: Tensor) -> tuple[Tensor, int]:
        if len(sample_seeds) != 8 or len(set(sample_seeds)) != 8:
            raise ValueError("posterior mean requires eight unique common-random-number seeds")
        sequence = self._timesteps().to(condition["observed"].device)
        samples: list[Tensor] = []
        for seed in sample_seeds:
            generator = torch.Generator(device=condition["observed"].device).manual_seed(int(seed))
            state = torch.randn(shape, device=condition["observed"].device, dtype=condition["observed"].dtype, generator=generator)
            mask = canonical_valid_time_mask(state, condition["valid_time_mask"]).to(state.dtype)
            state = state * mask
            for index, t_scalar in enumerate(sequence):
                t = torch.full((shape[0],), int(t_scalar), dtype=torch.long, device=state.device)
                alpha = _extract(self.alpha_bar, t, state.ndim)
                v = self.backbone(state, t, **condition)
                x0 = alpha.sqrt() * state - (1.0 - alpha).sqrt() * v
                eps = (1.0 - alpha).sqrt() * state + alpha.sqrt() * v
                if index + 1 == sequence.numel():
                    state = x0 * mask
                else:
                    next_t = torch.full_like(t, int(sequence[index + 1]))
                    next_alpha = _extract(self.alpha_bar, next_t, state.ndim)
                    state = (next_alpha.sqrt() * x0 + (1.0 - next_alpha).sqrt() * eps) * mask
            samples.append(state)
        return torch.stack(samples).mean(0), len(samples) * int(sequence.numel())


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "BoundedResidual", "OneStepResidualEstimator", "PopulationAnchor",
    "SoftTransferContext", "SubjectResidualConfig", "SubjectResidualDiffusion",
    "parameter_count",
]
