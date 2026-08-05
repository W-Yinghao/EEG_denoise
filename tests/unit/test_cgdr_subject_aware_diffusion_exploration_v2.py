from __future__ import annotations

import numpy as np

from eeg_cgdr.experiments.subject_aware_diffusion_exploration_v2 import (
    _correlation,
    _fir_design,
    _prediction_error,
    _projector,
    _ridge,
    _risk_auc,
)
from eeg_cgdr.models.subject_aware_wide_v2 import (
    SupportFiLMArtifactLatentDiffusion,
    SupportLoRAArtifactLatentDiffusion,
    canonical_eog_latent,
    full_c_subject_residual,
    lazy_subject_residual,
    physical_eog_latent,
)
from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusionConfig
import torch


def test_ridge_and_projector_are_finite_and_symmetric() -> None:
    rng = np.random.default_rng(4)
    eog = rng.normal(size=(2, 200))
    eog -= eog.mean(axis=1, keepdims=True)
    truth = rng.normal(size=(8, 2))
    eeg = truth @ eog
    fitted = _ridge(eeg, eog, 1.0e-8)
    assert np.allclose(fitted, truth, atol=1.0e-7)
    projector = _projector(fitted)
    assert np.allclose(projector, projector.T, atol=1.0e-7)
    assert np.allclose(projector @ projector, projector, atol=1.0e-7)
    assert _prediction_error(eeg, eog, fitted) < 1.0e-7


def test_fir_design_preserves_lag_alignment() -> None:
    eog = np.arange(40, dtype=np.float64)[None]
    design, index = _fir_design(eog, (-2, 0, 2))
    assert design.shape == (3, index.size)
    assert np.array_equal(design[1], eog[0, index])
    assert np.array_equal(design[0], eog[0, index + 2])
    assert np.array_equal(design[2], eog[0, index - 2])


def test_risk_and_centered_correlations_use_unit_arrays() -> None:
    error = np.asarray([4.0, 3.0, 2.0, 1.0])
    uncertainty = error.copy()
    summary = _risk_auc(error, uncertainty)
    assert summary["risk_coverage_auc"] < summary["random_ranking_auc"]
    assert np.isclose(_correlation(error, uncertainty), 1.0)
    assert np.isclose(_correlation(error, uncertainty, rank=True), 1.0)


def test_canonical_target_does_not_change_with_operator_swap() -> None:
    rng = np.random.default_rng(9)
    z = rng.normal(size=(4, 2, 32))
    mean = np.asarray([0.4, -0.7])
    std = np.asarray([1.4, 0.6])
    scale = rng.uniform(.5, 2.0, size=(4, 2))
    valid = np.ones((4, 32), dtype=bool)
    target, center, target_std = canonical_eog_latent(z, mean, std, scale, valid)
    recovered = physical_eog_latent(torch.from_numpy(target), torch.from_numpy(center), torch.from_numpy(target_std)).numpy()
    expected = (z * std[None, :, None] + mean[None, :, None]) / scale[:, :, None]
    assert np.allclose(recovered, expected, atol=1e-5)


def test_full_c_residual_and_zero_gate_short_circuit() -> None:
    observed = torch.randn(2, 6, 32)
    population = observed * .9
    latent = torch.randn(2, 2, 32)
    c0 = torch.randn(2, 6, 2)
    cs = torch.randn(2, 6, 2)
    valid = torch.ones(2, 32, dtype=torch.bool)
    output, correction = full_c_subject_residual(observed, population, latent, c0, cs, .7, valid)
    assert torch.allclose(output, observed - correction)
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        return output, correction
    fallback, value, used = lazy_subject_residual(population, 0.0, factory)
    assert torch.equal(fallback, population)
    assert value is None and not used and calls == 0


def test_support_film_and_low_rank_adapter_paths_are_finite() -> None:
    torch.manual_seed(14)
    batch, channels, latent, length = 2, 8, 2, 32
    model_config = ArtifactLatentModelConfig(
        eeg_channels=channels,
        signal_length=length,
        latent_channels=latent,
        base_channels=8,
        num_res_blocks=1,
        groupnorm_groups=8,
        dropout=0.0,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        attention_length=4,
        attention_heads=4,
    )
    diffusion_config = ArtifactLatentDiffusionConfig(num_timesteps=1000)
    observed = torch.randn(batch, channels, length)
    transfer = torch.randn(batch, channels, latent)
    scale = torch.linalg.vector_norm(transfer, dim=1).clamp_min(1.0e-6)
    condition = {
        "observed": observed,
        "full_transfer": transfer,
        "normalized_transfer": transfer / scale[:, None],
        "transfer_scale": scale,
        "singular_values": torch.linalg.svdvals(transfer),
        "rank": torch.full((batch,), latent, dtype=torch.long),
        "rho": torch.tensor([0.4, 0.8]),
        "calibration_duration_seconds": torch.full((batch,), 30.0),
        "channel_mask": torch.ones(batch, channels, dtype=torch.bool),
        "valid_time_mask": torch.ones(batch, length, dtype=torch.bool),
    }
    target = torch.randn(batch, latent, length)
    timestep = torch.tensor([100, 700])
    noise = torch.randn_like(target)
    for low_rank in (False, True):
        model = SupportFiLMArtifactLatentDiffusion(
            model_config, diffusion_config, low_rank_adapter=low_rank
        )
        model.set_population_transfer(transfer.mean(0))
        loss, details = model.training_loss(
            target, timestep=timestep, noise=noise, **condition
        )
        assert torch.isfinite(loss)
        assert torch.isfinite(details["x0_mse"])
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
    adapter_model = SupportLoRAArtifactLatentDiffusion(model_config, diffusion_config)
    adapter_model.freeze_population_backbone()
    adapter_model.reset_support_adapter()
    adapter_loss, _ = adapter_model.training_loss(
        target, timestep=timestep, noise=noise, **condition
    )
    adapter_loss.backward()
    assert torch.isfinite(adapter_loss)
    assert any(parameter.grad is not None for parameter in adapter_model.output_adapter.parameters())
    assert all(
        parameter.grad is None
        for name, parameter in adapter_model.named_parameters()
        if not name.startswith("output_adapter.")
    )
