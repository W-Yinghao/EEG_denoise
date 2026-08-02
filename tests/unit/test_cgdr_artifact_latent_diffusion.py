"""Focused contracts for operator-conditioned artifact-latent diffusion."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from torch import nn

from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
    artifact_posterior_point_estimate,
    cosine_alpha_bar,
)


class _ZeroV(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.calls = 0

    def forward(
        self,
        value: torch.Tensor,
        _timestep: torch.Tensor,
        *,
        valid_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(
            value.shape[0],
            self.latent_channels,
            value.shape[-1],
            device=value.device,
            dtype=value.dtype,
        )


class _OnesV(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels

    def forward(
        self,
        value: torch.Tensor,
        _timestep: torch.Tensor,
        *,
        valid_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ones(
            value.shape[0],
            self.latent_channels,
            value.shape[-1],
            device=value.device,
            dtype=value.dtype,
        )


class _ContextSensitiveV(nn.Module):
    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.latent_channels = latent_channels

    def forward(
        self,
        value: torch.Tensor,
        _timestep: torch.Tensor,
        *,
        valid_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = value[:, self.latent_channels :, :]
        summary = context.mean(dim=1, keepdim=True)
        return summary.expand(-1, self.latent_channels, -1)


def _model() -> ArtifactLatentDiffusion:
    return ArtifactLatentDiffusion(
        ArtifactLatentModelConfig(
            eeg_channels=4,
            signal_length=8,
            latent_channels=3,
            base_channels=8,
            num_res_blocks=1,
            groupnorm_groups=4,
            dropout=0.0,
            time_sinusoidal_dim=16,
            time_embed_dim=32,
            attention_length=8,
            attention_heads=4,
        ),
        ArtifactLatentDiffusionConfig(num_timesteps=16),
    )


def _operator(*, changed: bool = False) -> dict[str, object]:
    normalized = torch.tensor(
        [
            [1.0, 0.0, 0.25],
            [0.0, 1.0, 0.50],
            [0.25, 0.0, 0.75],
            [0.0, 0.25, 0.25],
        ]
    )
    if changed:
        normalized = normalized.clone()
        normalized[:, 2] = torch.tensor([0.75, 0.25, 0.50, 0.50])
    scale = torch.tensor([2.0, 3.0, 4.0])
    return {
        "full_transfer": normalized * scale[None, :],
        "normalized_transfer": normalized,
        "transfer_scale": scale,
        # Projector rank two does not truncate the third EOG coordinate.
        "singular_values": torch.tensor([5.0, 2.0, 0.5]),
        "rank": 2,
        "rho": 0.75,
        "calibration_duration_seconds": 30.0,
        "channel_mask": torch.ones(4, dtype=torch.bool),
    }


def test_cosine_schedule_reaches_negligible_terminal_alpha_bar() -> None:
    betas, alphas_cumprod = cosine_alpha_bar(1000)

    assert betas.dtype == torch.float64
    assert alphas_cumprod.dtype == torch.float64
    assert betas.shape == alphas_cumprod.shape == (1000,)
    assert torch.all((betas > 0.0) & (betas < 1.0))
    assert torch.all(alphas_cumprod[1:] < alphas_cumprod[:-1])
    assert float(alphas_cumprod[-1]) <= 1.0e-4


def test_v_prediction_round_trip_recovers_x0_and_epsilon() -> None:
    model = _model()
    x0 = torch.linspace(-1.0, 1.0, 2 * 3 * 8).reshape(2, 3, 8)
    epsilon = torch.linspace(0.75, -0.5, 2 * 3 * 8).reshape(2, 3, 8)
    timestep = torch.tensor([0, 11], dtype=torch.long)

    x_t = model.q_sample(x0, timestep, epsilon)
    target_v = model.v_target(x0, epsilon, timestep)
    recovered_x0, recovered_epsilon = model.x0_and_epsilon_from_v(
        x_t,
        target_v,
        timestep,
    )

    torch.testing.assert_close(recovered_x0, x0, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(
        recovered_epsilon,
        epsilon,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_predict_v_masks_padding_for_all_three_latent_coordinates() -> None:
    model = _model()
    model.unet = _OnesV(latent_channels=3)
    observed = torch.zeros(1, 4, 8)
    noisy = torch.ones(1, 3, 8)
    valid = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)

    predicted = model.predict_v(
        noisy,
        torch.tensor([5], dtype=torch.long),
        observed=observed,
        valid_time_mask=valid,
        **_operator(),
    )

    torch.testing.assert_close(predicted[:, :, :5], torch.ones(1, 3, 5))
    assert torch.count_nonzero(predicted[:, :, 5:]).item() == 0
    assert torch.count_nonzero(predicted[:, 2, :5]).item() == 5


def test_posterior_point_estimate_is_k8_arithmetic_mean_without_selector() -> None:
    samples = tuple(torch.full((1, 3, 8), float(index)) for index in range(8))

    mean, standard_deviation = artifact_posterior_point_estimate(samples)

    torch.testing.assert_close(mean, torch.full_like(mean, 3.5))
    torch.testing.assert_close(
        standard_deviation,
        torch.full_like(standard_deviation, math.sqrt(5.25)),
    )
    assert tuple(inspect.signature(artifact_posterior_point_estimate).parameters) == (
        "samples",
    )
    with pytest.raises(ValueError, match="exactly K=8"):
        artifact_posterior_point_estimate(samples[:-1])


def test_ddim_steps_equal_actual_network_calls_and_trajectory_is_finite() -> None:
    model = _model()
    zero_v = _ZeroV(latent_channels=3)
    model.unet = zero_v
    observed = torch.linspace(-0.5, 0.5, 2 * 4 * 8).reshape(2, 4, 8)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    operator = _operator()
    operator["channel_mask"] = torch.tensor([1, 1, 1, 0], dtype=torch.bool)

    posterior = model.posterior_mean(
        observed=observed,
        latent_mean=torch.zeros(3),
        latent_standard_deviation=torch.ones(3),
        valid_time_mask=valid,
        sample_seeds=tuple(range(100, 108)),
        ddim_steps=4,
        record_trajectory=True,
        **operator,
    )

    assert posterior.sample_count == 8
    assert posterior.network_calls == zero_v.calls == 8 * 4
    assert len(posterior.trajectories) == 8 * 4
    assert posterior.standardized_latent_mean.shape == (2, 3, 8)
    assert posterior.standardized_latent_standard_deviation.shape == (2, 3, 8)
    assert posterior.correction.shape == posterior.restored.shape == (2, 4, 8)
    assert torch.isfinite(posterior.standardized_latent_mean).all()
    assert torch.isfinite(posterior.restored).all()
    assert all(step.finite for step in posterior.trajectories)
    assert all(math.isfinite(step.latent_rms) for step in posterior.trajectories)
    assert torch.count_nonzero(posterior.restored[:, 3]).item() == 0
    assert torch.count_nonzero(posterior.restored[0, :, 6:]).item() == 0
    assert torch.count_nonzero(posterior.restored[1, :, 5:]).item() == 0


def test_runtime_operator_context_changes_prediction_with_shared_backbone() -> None:
    model = _model()
    model.unet = _ContextSensitiveV(latent_channels=3)
    backbone_identity = id(model.unet)
    observed = torch.zeros(1, 4, 8)
    noisy = torch.zeros(1, 3, 8)
    timestep = torch.tensor([7], dtype=torch.long)

    first = model.predict_v(
        noisy,
        timestep,
        observed=observed,
        valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
        **_operator(changed=False),
    )
    second = model.predict_v(
        noisy,
        timestep,
        observed=observed,
        valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
        **_operator(changed=True),
    )

    assert id(model.unet) == backbone_identity
    assert first.shape == second.shape == (1, 3, 8)
    assert not torch.equal(first, second)
