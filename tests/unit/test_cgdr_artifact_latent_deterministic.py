"""Focused contracts for the information-matched artifact-latent estimator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentContext,
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
    artifact_conditioning_channels,
    build_artifact_conditioning,
)


class _FixedStandardizedLatent(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(
        self,
        features: torch.Tensor,
        _condition: torch.Tensor,
        *,
        valid_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.value.to(features).expand(features.shape[0], -1, -1)


def _model() -> DeterministicArtifactEstimator:
    model = DeterministicArtifactEstimator(
        ArtifactLatentModelConfig(
            eeg_channels=3,
            signal_length=8,
            base_channels=8,
            num_res_blocks=1,
            groupnorm_groups=4,
            dropout=0.0,
            time_sinusoidal_dim=16,
            time_embed_dim=32,
            attention_heads=4,
        )
    )
    standardized = torch.stack(
        (torch.ones(8), 2.0 * torch.ones(8)),
        dim=0,
    )[None, :, :]
    model.unet = _FixedStandardizedLatent(standardized)
    return model


def _context(*, swapped: bool = False) -> ArtifactLatentContext:
    normalized = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    if swapped:
        normalized = normalized[:, [1, 0]]
    scale = torch.tensor([2.0, 4.0])
    return ArtifactLatentContext(
        full_transfer=normalized * scale[None, :],
        normalized_transfer=normalized,
        transfer_scale=scale,
        singular_values=torch.tensor([5.0, 2.0]),
        rank=2,
        calibration_duration_seconds=30.0,
        latent_mean=torch.tensor([0.5, -0.25]),
        latent_standard_deviation=torch.tensor([1.5, 0.5]),
    )


def test_standardized_latent_reconstructs_exact_y_minus_c_a() -> None:
    model = _model()
    observed = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    duration = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    layout = torch.tensor([1, 1, 0], dtype=torch.bool)
    context = _context()

    result = model.restore(
        observed,
        population_context=context,
        rho=0.75,
        context_factory=lambda: context,
        channel_mask=layout,
        valid_time_mask=duration,
    )

    expected_standardized = torch.stack(
        (torch.ones(8), 2.0 * torch.ones(8)), dim=0
    )[None]
    expected_standardized[:, :, 6:] = 0.0
    expected_latent = (
        expected_standardized
        * context.latent_standard_deviation[None, :, None]
        + context.latent_mean[None, :, None]
    )
    transfer = context.normalized_transfer[None]
    expected_contamination = torch.einsum("bcr,brt->bct", transfer, expected_latent)
    output_mask = duration[:, None, :] * layout[None, :, None]
    expected_contamination = expected_contamination * output_mask
    expected_restored = (observed * output_mask - expected_contamination) * output_mask

    torch.testing.assert_close(result.standardized_latent, expected_standardized)
    torch.testing.assert_close(result.latent, expected_latent)
    torch.testing.assert_close(result.predicted_contamination, expected_contamination)
    torch.testing.assert_close(result.restored, expected_restored)
    torch.testing.assert_close(
        result.restored,
        observed * output_mask
        - torch.einsum("bcr,brt->bct", transfer, result.latent) * output_mask,
    )
    assert result.rho == 0.75


def test_rho_zero_short_circuits_before_context_factory() -> None:
    model = _model()
    observed = torch.zeros(1, 3, 8)
    calls = 0

    def forbidden_context() -> ArtifactLatentContext:
        nonlocal calls
        calls += 1
        raise AssertionError("rho=0 constructed a calibration context")

    result = model.restore(
        observed,
        population_context=_context(),
        rho=0.0,
        context_factory=forbidden_context,
        channel_mask=torch.ones(3, dtype=torch.bool),
        valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
    )

    assert calls == 0
    assert result.population_short_circuit is True
    assert result.context_branch_used is False
    assert result.rho == 0.0


def test_runtime_context_switch_does_not_select_or_retrain_parameters() -> None:
    model = _model()
    observed = torch.zeros(1, 3, 8)
    duration = torch.ones(1, 8, dtype=torch.bool)
    layout = torch.ones(3, dtype=torch.bool)
    backbone_identity = id(model.unet)
    calls = 0

    def wrong_or_shuffled_context() -> ArtifactLatentContext:
        nonlocal calls
        calls += 1
        return _context(swapped=True)

    population = model.restore(
        observed,
        population_context=_context(),
        rho=0.0,
        context_factory=wrong_or_shuffled_context,
        channel_mask=layout,
        valid_time_mask=duration,
    )
    selected = model.restore(
        observed,
        population_context=_context(),
        rho=1.0,
        context_factory=wrong_or_shuffled_context,
        channel_mask=layout,
        valid_time_mask=duration,
    )

    assert calls == 1
    assert id(model.unet) == backbone_identity
    assert selected.context_branch_used is True
    assert selected.population_short_circuit is False
    assert not torch.equal(
        population.predicted_contamination,
        selected.predicted_contamination,
    )


def test_rejects_inconsistent_full_and_normalized_transfer() -> None:
    model = _model()
    context = _context()
    inconsistent = ArtifactLatentContext(
        full_transfer=context.full_transfer + 0.25,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        calibration_duration_seconds=context.calibration_duration_seconds,
        latent_mean=context.latent_mean,
        latent_standard_deviation=context.latent_standard_deviation,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        model(
            torch.zeros(1, 3, 8),
            full_transfer=inconsistent.full_transfer,
            normalized_transfer=inconsistent.normalized_transfer,
            transfer_scale=inconsistent.transfer_scale,
            singular_values=inconsistent.singular_values,
            rank=inconsistent.rank,
            rho=1.0,
            calibration_duration_seconds=30.0,
            channel_mask=torch.ones(3, dtype=torch.bool),
            valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
        )


def test_public_conditioning_stack_is_the_model_input_contract() -> None:
    config = ArtifactLatentModelConfig(eeg_channels=3, signal_length=8)
    context = _context()
    observed = torch.zeros(1, 3, 8)
    features, mask = build_artifact_conditioning(
        observed,
        full_transfer=context.full_transfer,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        rho=0.5,
        calibration_duration_seconds=context.calibration_duration_seconds,
        channel_mask=torch.ones(3, dtype=torch.bool),
        valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    assert features.shape == (1, artifact_conditioning_channels(config), 8)
    assert mask.shape == (1, 1, 8)

    model = _model()
    standardized = model(
        observed,
        full_transfer=context.full_transfer,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        rho=0.5,
        calibration_duration_seconds=context.calibration_duration_seconds,
        channel_mask=torch.ones(3, dtype=torch.bool),
        valid_time_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    assert isinstance(standardized, torch.Tensor)
    assert standardized.shape == (1, 2, 8)


def test_three_eog_coordinates_are_not_truncated_by_projector_rank() -> None:
    config = ArtifactLatentModelConfig(
        eeg_channels=4,
        signal_length=8,
        latent_channels=3,
        base_channels=8,
        num_res_blocks=1,
        groupnorm_groups=4,
        dropout=0.0,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        attention_heads=4,
    )
    model = DeterministicArtifactEstimator(config)
    fixed = torch.stack(
        (torch.ones(8), 2.0 * torch.ones(8), 3.0 * torch.ones(8)),
        dim=0,
    )[None]
    model.unet = _FixedStandardizedLatent(fixed)
    normalized = torch.tensor(
        [
            [1.0, 0.0, 0.25],
            [0.0, 1.0, 0.50],
            [0.0, 0.0, 0.75],
            [0.0, 0.0, 0.25],
        ]
    )
    scale = torch.tensor([2.0, 3.0, 4.0])
    observed = torch.zeros(1, 4, 8)
    kwargs = {
        "full_transfer": normalized * scale[None, :],
        "normalized_transfer": normalized,
        "transfer_scale": scale,
        # Rank two describes the retained EEG projector only.  The third
        # external-EOG coordinate and its nonzero transfer remain visible.
        "singular_values": torch.tensor([5.0, 2.0, 0.5]),
        "rank": 2,
        "rho": 1.0,
        "calibration_duration_seconds": 30.0,
        "channel_mask": torch.ones(4, dtype=torch.bool),
        "valid_time_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    features, _ = build_artifact_conditioning(observed, **kwargs)
    predicted = model(observed, **kwargs)

    assert features.shape == (1, artifact_conditioning_channels(config), 8)
    assert predicted.shape == (1, 3, 8)
    torch.testing.assert_close(predicted, fixed)
    assert torch.count_nonzero(predicted[:, 2]).item() == 8
