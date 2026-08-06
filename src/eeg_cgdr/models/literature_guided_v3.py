"""Minimal literature-driven subject-conditioning mechanisms for v3.

Only query-disjoint support enters these modules.  Participant identifiers and
query EOG/labels/outcomes are absent from every forward surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
    _extract,
)
from .artifact_latent_deterministic import ArtifactLatentModelConfig
from .clean_prior import canonical_valid_time_mask
from .parallel_subject_routes import _SummaryFiLMUNet


FORBIDDEN_QUERY_FIELDS = (
    "query_EOG",
    "query_eye_tracking",
    "query_artifact_label",
    "query_outcome",
    "participant_ID",
)


@dataclass(frozen=True)
class RawSupportConfig:
    token_width: int = 128
    token_count: int = 4
    encoder_layers: int = 2
    attention_heads: int = 4
    context_dropout_probability: float = 0.25

    def __post_init__(self) -> None:
        if self.token_width < 16 or self.token_width % self.attention_heads:
            raise ValueError("support token width must divide attention heads")
        if self.token_count < 1 or self.encoder_layers < 1:
            raise ValueError("support token count/layers must be positive")
        if not 0.0 <= self.context_dropout_probability < 1.0:
            raise ValueError("context dropout must lie in [0,1)")


class RawSupportSetEncoder(nn.Module):
    """Permutation-invariant support-window encoder with learned query tokens."""

    def __init__(self, eeg_channels: int, latent_channels: int, config: RawSupportConfig) -> None:
        super().__init__()
        self.eeg_channels = eeg_channels
        self.latent_channels = latent_channels
        self.config = config
        # mean, standard deviation and absolute mean for EEG and ocular latent.
        width = 3 * (eeg_channels + latent_channels) + 1
        self.window_encoder = nn.Sequential(
            nn.Linear(width, config.token_width),
            nn.SiLU(),
            nn.Linear(config.token_width, config.token_width),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_width,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.token_width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=config.encoder_layers)
        self.query_tokens = nn.Parameter(torch.zeros(config.token_count, config.token_width))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.pool = nn.MultiheadAttention(
            config.token_width, config.attention_heads, dropout=0.0, batch_first=True
        )
        self.output_norm = nn.LayerNorm(config.token_width)

    @staticmethod
    def _moments(value: Tensor, mask: Tensor) -> Tensor:
        weight = mask[:, :, None, :].to(value.dtype)
        denominator = weight.sum(dim=-1).clamp_min(1.0)
        mean = (value * weight).sum(dim=-1) / denominator
        variance = ((value - mean[..., None]).square() * weight).sum(dim=-1) / denominator
        absolute = (value.abs() * weight).sum(dim=-1) / denominator
        return torch.cat((mean, variance.clamp_min(0).sqrt(), absolute), dim=-1)

    def forward(
        self,
        support_eeg: Tensor,
        support_artifact_latent: Tensor,
        support_valid_time_mask: Tensor,
        context_present: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if support_eeg.ndim != 4 or support_artifact_latent.ndim != 4:
            raise ValueError("support EEG/latent must have shape (B,S,C/E,T)")
        batch, windows, channels, length = support_eeg.shape
        if channels != self.eeg_channels:
            raise ValueError("support EEG channel count differs")
        if support_artifact_latent.shape != (batch, windows, self.latent_channels, length):
            raise ValueError("support latent shape differs")
        mask = torch.as_tensor(support_valid_time_mask, device=support_eeg.device)
        if mask.shape != (batch, windows, length):
            raise ValueError("support valid-time mask shape differs")
        present = torch.as_tensor(context_present, device=support_eeg.device, dtype=support_eeg.dtype)
        if present.shape != (batch,) or not bool(((present == 0) | (present == 1)).all()):
            raise ValueError("context_present must be a binary (B,) tensor")
        if not bool(torch.isfinite(support_eeg).all()) or not bool(torch.isfinite(support_artifact_latent).all()):
            raise ValueError("support contains non-finite values")
        eeg_moments = self._moments(support_eeg, mask)
        latent_moments = self._moments(support_artifact_latent, mask)
        valid_fraction = mask.float().mean(dim=-1, keepdim=True).to(support_eeg.dtype)
        tokens = self.window_encoder(torch.cat((eeg_moments, latent_moments, valid_fraction), dim=-1))
        tokens = self.set_encoder(tokens)
        queries = self.query_tokens[None].expand(batch, -1, -1)
        pooled, _ = self.pool(queries, tokens, tokens, need_weights=False)
        pooled = self.output_norm(pooled) * present[:, None, None]
        summary = pooled.mean(dim=1)
        return pooled, summary


def _support_unet(model: ArtifactLatentModelConfig, summary_width: int) -> _SummaryFiLMUNet:
    base = UNet1D(
        ModelConfig(
            in_channels=model.latent_channels + model.eeg_channels + 1,
            out_channels=model.latent_channels,
            signal_length=model.signal_length,
            base_channels=model.base_channels,
            channel_mults=list(model.channel_mults),
            num_res_blocks=model.num_res_blocks,
            groupnorm_groups=model.groupnorm_groups,
            dropout=model.dropout,
            time_sinusoidal_dim=model.time_sinusoidal_dim,
            time_embed_dim=model.time_embed_dim,
            attention_length=model.attention_length,
            attention_heads=model.attention_heads,
        ),
        subject_conditioned=False,
    )
    return _SummaryFiLMUNet(base, summary_width, model.time_embed_dim)


class RawSupportTokenDiffusion(ArtifactLatentDiffusion):
    """Canonical artifact-latent diffusion conditioned on raw support tokens."""

    visible_input_fields = (
        "observed_query_EEG",
        "query_disjoint_support_EEG",
        "query_disjoint_support_EOG_derived_latent",
        "support_valid_time_mask",
        "context_present",
        "query_valid_time_mask",
    )
    forbidden_input_fields = FORBIDDEN_QUERY_FIELDS

    def __init__(
        self,
        model_config: ArtifactLatentModelConfig,
        diffusion_config: ArtifactLatentDiffusionConfig,
        support_config: RawSupportConfig,
    ) -> None:
        super().__init__(model_config, diffusion_config)
        del self.unet
        self.support_config = support_config
        self.support_encoder = RawSupportSetEncoder(
            model_config.eeg_channels, model_config.latent_channels, support_config
        )
        self.support_unet = _support_unet(model_config, support_config.token_width)

    def _context(
        self,
        *,
        support_eeg: Tensor,
        support_artifact_latent: Tensor,
        support_valid_time_mask: Tensor,
        context_present: Tensor,
    ) -> Tensor:
        _, summary = self.support_encoder(
            support_eeg, support_artifact_latent, support_valid_time_mask, context_present
        )
        return summary

    def predict_v(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        *,
        observed: Tensor,
        support_eeg: Tensor,
        support_artifact_latent: Tensor,
        support_valid_time_mask: Tensor,
        context_present: Tensor,
        valid_time_mask: Tensor,
    ) -> Tensor:
        self._validate_latent(noisy_latent, name="noisy artifact latent")
        self._validate_observed(observed)
        self._validate_timestep(timestep, noisy_latent.shape[0], noisy_latent.device)
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        summary = self._context(
            support_eeg=support_eeg,
            support_artifact_latent=support_artifact_latent,
            support_valid_time_mask=support_valid_time_mask,
            context_present=context_present,
        )
        mask_float = mask.to(observed.dtype)
        features = torch.cat((noisy_latent * mask_float, observed * mask_float, mask_float), dim=1)
        return self.support_unet(features, timestep, summary, mask) * mask_float

    def training_loss(
        self,
        target: Tensor,
        *,
        observed: Tensor,
        support_eeg: Tensor,
        support_artifact_latent: Tensor,
        support_valid_time_mask: Tensor,
        context_present: Tensor,
        valid_time_mask: Tensor,
        generator: torch.Generator | None = None,
        timestep: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        self._validate_latent(target, name="canonical artifact target")
        mask = canonical_valid_time_mask(observed, valid_time_mask).to(target.dtype)
        x0 = target * mask
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (target.shape[0],), device=target.device, generator=generator)
        if noise is None:
            noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
        noise = noise * mask
        xt = self.q_sample(x0, timestep, noise) * mask
        truth = self.v_target(x0, noise, timestep) * mask
        predicted = self.predict_v(
            xt, timestep, observed=observed, support_eeg=support_eeg,
            support_artifact_latent=support_artifact_latent,
            support_valid_time_mask=support_valid_time_mask,
            context_present=context_present, valid_time_mask=valid_time_mask,
        )
        squared = (predicted - truth).square() * mask
        alpha = _extract(self.alphas_cumprod, timestep, x0.ndim)
        snr = alpha / (1.0 - alpha).clamp_min(1e-8)
        weight = torch.minimum(snr, torch.full_like(snr, self.diffusion_config.min_snr_gamma)) / (snr + 1.0)
        denominator = (mask.sum() * target.shape[1]).clamp_min(1)
        loss = (squared * weight).sum() / denominator
        predicted_x0, _ = self.x0_and_epsilon_from_v(xt, predicted, timestep)
        return loss, {
            "v_mse": (squared.sum() / denominator).detach(),
            "x0_mse": (((predicted_x0 - x0).square() * mask).sum() / denominator).detach(),
            "mean_timestep": timestep.float().mean().detach(),
        }

    @torch.no_grad()
    def latent_samples(
        self,
        *,
        observed: Tensor,
        support_eeg: Tensor,
        support_artifact_latent: Tensor,
        support_valid_time_mask: Tensor,
        context_present: Tensor,
        valid_time_mask: Tensor,
        sample_seeds: Sequence[int],
        ddim_steps: int,
        v_adapter: DirectSupportAdapter | None = None,
    ) -> Tensor:
        seeds = tuple(int(value) for value in sample_seeds)
        if len(seeds) != 8 or len(set(seeds)) != 8:
            raise ValueError("raw-support posterior requires eight unique common-random seeds")
        mask = canonical_valid_time_mask(observed, valid_time_mask).to(observed.dtype)
        timesteps = self._timestep_sequence(self.num_timesteps, ddim_steps)
        samples = []
        for seed in seeds:
            generator = torch.Generator(device=observed.device).manual_seed(seed)
            latent = torch.randn(
                (observed.shape[0], self.model_config.latent_channels, observed.shape[-1]),
                device=observed.device, dtype=observed.dtype, generator=generator,
            ) * mask
            for reverse_index, raw_t in enumerate(timesteps):
                timestep = torch.full((observed.shape[0],), raw_t, device=observed.device, dtype=torch.long)
                predicted = self.predict_v(
                    latent, timestep, observed=observed, support_eeg=support_eeg,
                    support_artifact_latent=support_artifact_latent,
                    support_valid_time_mask=support_valid_time_mask,
                    context_present=context_present, valid_time_mask=valid_time_mask,
                )
                if v_adapter is not None:
                    predicted = v_adapter(predicted, valid_time_mask)
                x0, epsilon = self.x0_and_epsilon_from_v(latent, predicted, timestep)
                x0, _ = self._dynamic_threshold(x0, mask.bool())
                if reverse_index == len(timesteps) - 1:
                    latent = x0
                else:
                    next_alpha = self.alphas_cumprod[timesteps[reverse_index + 1]]
                    latent = (next_alpha.sqrt() * x0 + (1.0 - next_alpha).sqrt() * epsilon) * mask
            samples.append(latent)
        return torch.stack(samples)


class RawSupportTokenDeterministic(nn.Module):
    """Information-matched one-step estimator for P-A."""

    forbidden_input_fields = FORBIDDEN_QUERY_FIELDS

    def __init__(self, model_config: ArtifactLatentModelConfig, support_config: RawSupportConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.support_encoder = RawSupportSetEncoder(
            model_config.eeg_channels, model_config.latent_channels, support_config
        )
        self.support_unet = _support_unet(model_config, support_config.token_width)

    def forward(
        self,
        *,
        observed: Tensor,
        support_eeg: Tensor,
        support_artifact_latent: Tensor,
        support_valid_time_mask: Tensor,
        context_present: Tensor,
        valid_time_mask: Tensor,
    ) -> Tensor:
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        _, summary = self.support_encoder(
            support_eeg, support_artifact_latent, support_valid_time_mask, context_present
        )
        state = torch.zeros(
            observed.shape[0], self.model_config.latent_channels, observed.shape[-1],
            device=observed.device, dtype=observed.dtype,
        )
        timestep = torch.zeros(observed.shape[0], device=observed.device, dtype=torch.long)
        mask_float = mask.to(observed.dtype)
        features = torch.cat((state, observed * mask_float, mask_float), dim=1)
        return self.support_unet(features, timestep, summary, mask) * mask_float


class DirectSupportAdapter(nn.Module):
    """Zero-initialized low-rank residual adapter for the direct P-B upper bound."""

    def __init__(self, latent_channels: int, rank: int = 4) -> None:
        super().__init__()
        if not 1 <= rank <= latent_channels * 4:
            raise ValueError("adapter rank is invalid")
        self.down = nn.Conv1d(latent_channels, rank, 1, bias=False)
        self.up = nn.Conv1d(rank, latent_channels, 1, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, base_latent: Tensor, valid_time_mask: Tensor) -> Tensor:
        mask = canonical_valid_time_mask(base_latent, valid_time_mask).to(base_latent.dtype)
        return (base_latent + self.up(torch.nn.functional.silu(self.down(base_latent)))) * mask


class SupportStatisticControl(nn.Module):
    """ReVIN-style control using support-only channel mean and scale."""

    @staticmethod
    def statistics(support: Tensor, support_valid_time_mask: Tensor) -> tuple[Tensor, Tensor]:
        if support.ndim != 4:
            raise ValueError("support must have shape (B,S,C,T)")
        mask = torch.as_tensor(support_valid_time_mask, device=support.device)
        if mask.shape != (support.shape[0], support.shape[1], support.shape[3]):
            raise ValueError("support valid-time mask shape differs")
        weight = mask[:, :, None, :].to(support.dtype)
        denominator = weight.sum(dim=(1, 3)).clamp_min(1.0)
        mean = (support * weight).sum(dim=(1, 3)) / denominator
        variance = ((support - mean[:, None, :, None]).square() * weight).sum(dim=(1, 3)) / denominator
        return mean, variance.clamp_min(1e-6).sqrt()

    def forward(self, query: Tensor, support: Tensor, support_valid_time_mask: Tensor) -> Tensor:
        if query.ndim != 3 or support.ndim != 4 or support.shape[0] != query.shape[0] or support.shape[2] != query.shape[1]:
            raise ValueError("query/support shapes differ for support-statistic control")
        mean, scale = self.statistics(support, support_valid_time_mask)
        return (query - mean[:, :, None]) / scale[:, :, None]

    def normalize_support(self, support: Tensor, support_valid_time_mask: Tensor) -> Tensor:
        mean, scale = self.statistics(support, support_valid_time_mask)
        normalized = (support - mean[:, None, :, None]) / scale[:, None, :, None]
        return normalized * torch.as_tensor(
            support_valid_time_mask, device=support.device, dtype=support.dtype
        )[:, :, None, :]


def discrete_selector_features(
    query_eeg: Tensor,
    population_output: Tensor,
    matching_output: Tensor,
    posterior_samples: Tensor,
    support_quantiles: Tensor,
) -> Tensor:
    """Deployable P-C inputs; no external query signal is accepted."""

    if query_eeg.shape != population_output.shape or query_eeg.shape != matching_output.shape:
        raise ValueError("selector EEG outputs must align")
    if posterior_samples.ndim != 4 or posterior_samples.shape[1:] != query_eeg.shape:
        raise ValueError("posterior samples must have shape (K,B,C,T)")
    if support_quantiles.ndim != 2 or support_quantiles.shape[0] != query_eeg.shape[0]:
        raise ValueError("support quantiles must be a (B,Q) tensor")
    activity = torch.stack((query_eeg.abs().mean((1, 2)), query_eeg.std((1, 2))), dim=1)
    disagreement = (population_output - matching_output).square().mean((1, 2)).sqrt()[:, None]
    dispersion = posterior_samples.std(dim=0, unbiased=False).square().mean((1, 2)).sqrt()[:, None]
    return torch.cat((activity, disagreement, dispersion, support_quantiles), dim=1)
