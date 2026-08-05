from __future__ import annotations

import torch

from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusionConfig
from eeg_cgdr.models.parallel_subject_routes import (
    AdaptiveActivityGate,
    FullCFiLMDiffusion,
    canonical_target,
    full_c_population_residual_reconstruction,
    guided_latent_step,
    sdedit_initial_latent,
)


def test_target_is_operator_invariant() -> None:
    target = torch.randn(3, 2, 64)
    first = torch.randn(3, 8, 2)
    second = torch.randn(3, 8, 2)
    assert torch.equal(canonical_target(target, first), canonical_target(target, second))


def test_full_c_residual_has_exact_g0_population_fallback() -> None:
    observed = torch.randn(2, 8, 64)
    population = observed * 0.8
    latent = torch.randn(2, 2, 64)
    c0 = torch.randn(2, 8, 2)
    cs = c0 + 0.2
    valid = torch.ones(2, 64, dtype=torch.bool)
    output, correction = full_c_population_residual_reconstruction(
        observed,
        population,
        latent,
        population_normalized_transfer=c0,
        subject_normalized_transfer=cs,
        latent_mean=torch.zeros(2),
        latent_standard_deviation=torch.ones(2),
        valid_time_mask=valid,
        gain=0.0,
    )
    assert torch.allclose(output, population)
    assert torch.allclose(correction, observed - population)


def test_full_c_film_reaches_all_major_blocks() -> None:
    model_cfg = ArtifactLatentModelConfig(
        eeg_channels=8,
        latent_channels=2,
        signal_length=64,
        base_channels=8,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        groupnorm_groups=8,
        attention_heads=4,
    )
    diff_cfg = ArtifactLatentDiffusionConfig(num_timesteps=1000)
    model = FullCFiLMDiffusion(model_cfg, diff_cfg, population_transfer=torch.randn(8, 2))
    assert model.film_block_count == 14


def test_activity_gate_and_bounded_route_helpers_are_finite() -> None:
    observed = torch.randn(2, 8, 64)
    valid = torch.ones(2, 64, dtype=torch.bool)
    gate = AdaptiveActivityGate(8)(observed, valid)
    assert gate.shape == (2, 1, 64)
    assert torch.isfinite(gate).all() and bool(((gate >= 0) & (gate <= 1)).all())
    latent = torch.randn(2, 2, 64)
    guided = guided_latent_step(latent, latent * 2, strength=0.25)
    initial = sdedit_initial_latent(latent, torch.tensor(0.7), torch.randn_like(latent))
    assert torch.isfinite(guided).all() and torch.isfinite(initial).all()


def test_query_external_fields_are_forbidden() -> None:
    assert "query_EOG" in AdaptiveActivityGate.forbidden_input_fields
    assert "participant_ID" in AdaptiveActivityGate.forbidden_input_fields
