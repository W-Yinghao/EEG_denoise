from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eeg_cgdr.experiments.parallel_subject_aware_routes_v1 import _donor_contexts

from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)
from eeg_cgdr.models.parallel_subject_routes import (
    AdaptiveActivityGate,
    FullCFiLMDiffusion,
    SupportOnlyLatentAdapter,
    canonical_target,
    full_c_population_residual_reconstruction,
    guided_latent_step,
    sdedit_initial_latent,
    structured_latent_samples,
)


def test_target_is_operator_invariant() -> None:
    target = torch.randn(3, 2, 64)
    first = torch.randn(3, 8, 2)
    second = torch.randn(3, 8, 2)
    assert torch.equal(canonical_target(target, first), canonical_target(target, second))


def test_donor_context_normalization_is_derived_from_selected_full_transfer() -> None:
    rng = np.random.default_rng(17)
    full = rng.normal(size=(3, 8, 2))
    source = SimpleNamespace(
        recording_keys=("p1", "p2", "p3"),
        full_transfer=full,
        # Deliberately stale parallel arrays must not be mixed into a donor.
        normalized_transfer=np.zeros_like(full),
        transfer_scale=np.ones((3, 2)),
        singular_values=np.ones((3, 2)),
        rank=np.full(3, 2),
        rho=np.full(3, 0.75),
        calibration_duration_seconds=np.full(3, 30.0),
    )
    contexts = _donor_contexts(SimpleNamespace(training=source))
    assert len(contexts) == 3
    for index, context in enumerate(contexts):
        assert np.allclose(
            context.full_transfer,
            context.normalized_transfer * context.transfer_scale[None, :],
        )
        assert np.allclose(context.full_transfer, full[index])


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


def test_support_adapter_is_small_and_masked() -> None:
    adapter = SupportOnlyLatentAdapter(2, 9, rank=1)
    latent = torch.randn(3, 2, 64)
    valid = torch.ones(3, 64, dtype=torch.bool)
    valid[:, -5:] = False
    output = adapter(latent, torch.randn(3, 9), valid)
    assert output.shape == latent.shape
    assert torch.count_nonzero(output[:, :, -5:]) == 0
    assert adapter.trainable_parameter_count < 100


def test_guided_and_sdedit_samplers_use_k1_without_external_query_fields() -> None:
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
    model = ArtifactLatentDiffusion(model_cfg, ArtifactLatentDiffusionConfig(num_timesteps=1000))
    observed = torch.randn(1, 8, 64)
    full = torch.randn(1, 8, 2)
    scale = torch.linalg.vector_norm(full, dim=1).clamp_min(0.1)
    normalized = full / scale[:, None]
    singular = torch.sort(torch.rand(1, 2) + 0.2, descending=True).values
    condition = {
        "observed": observed,
        "full_transfer": full,
        "normalized_transfer": normalized,
        "transfer_scale": scale,
        "singular_values": singular,
        "rank": torch.full((1,), 2),
        "rho": torch.ones(1),
        "calibration_duration_seconds": torch.full((1,), 30.0),
        "channel_mask": torch.ones(1, 8, dtype=torch.bool),
        "valid_time_mask": torch.ones(1, 64, dtype=torch.bool),
    }
    anchor = torch.zeros(1, 2, 64)
    guided, _, guided_calls = structured_latent_samples(
        model, observation_anchor=anchor, sample_seeds=(7,), condition=condition,
        mode="posterior_guidance", ddim_steps=2,
    )
    sdedit, _, sdedit_calls = structured_latent_samples(
        model, observation_anchor=anchor, sample_seeds=(7,), condition=condition,
        mode="anchored_sdedit", ddim_steps=2,
    )
    assert guided.shape == sdedit.shape == anchor.shape
    assert guided_calls == sdedit_calls == 2
    assert torch.isfinite(guided).all() and torch.isfinite(sdedit).all()
