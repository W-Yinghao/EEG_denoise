from __future__ import annotations

import numpy as np

from eeg_cgdr.experiments.subject_aware_diffusion_exploration_v2 import (
    _correlation,
    _blocked_fir_crossfit,
    _blocked_state_crossfit,
    _fir_design,
    _fir_lags,
    _prediction_error,
    _projector,
    _ridge,
    _risk_auc,
)
from eeg_cgdr.models.subject_aware_wide_v2 import (
    SupportFiLMArtifactLatentDiffusion,
    SupportLoRAArtifactLatentDiffusion,
    canonical_eog_latent,
    fir_coefficients_from_lag_major,
    fir_coefficients_to_lag_major,
    fir_full_replacement,
    fir_transfer_correction,
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


def test_two_eog_five_lag_impulse_fit_cache_runtime_round_trip(tmp_path) -> None:
    eog_channels, eeg_channels = 2, 3
    lags = (-2, -1, 0, 1, 2)
    eog = np.zeros((eog_channels, 96), dtype=np.float64)
    eog[0, (12, 37, 72)] = (1.0, -0.7, -0.3)
    eog[1, (22, 51, 83)] = (-0.8, 1.2, -0.4)
    truth = np.arange(1, eeg_channels * eog_channels * len(lags) + 1, dtype=np.float64)
    truth = truth.reshape(eeg_channels, eog_channels, len(lags)) / 17.0
    latent = torch.from_numpy(eog[None]).double()
    valid = torch.ones((1, eog.shape[1]), dtype=torch.bool)
    generated = fir_transfer_correction(torch.from_numpy(truth), latent, lags, valid)[0].numpy()
    design, index = _fir_design(eog, lags)
    fitted_flat = _ridge(generated[:, index], design, 1.0e-12)
    fitted = fir_coefficients_from_lag_major(
        fitted_flat,
        eeg_channels=eeg_channels,
        eog_channels=eog_channels,
        lag_count=len(lags),
    )
    assert np.allclose(fitted, truth, atol=1.0e-8)
    assert np.allclose(fir_coefficients_to_lag_major(fitted), fitted_flat)
    cache = tmp_path / "fir_cache.npz"
    np.savez(cache, FIR=fitted, FIR_lags=np.asarray(lags))
    with np.load(cache) as loaded:
        replay = fir_transfer_correction(
            torch.from_numpy(loaded["FIR"]), latent,
            tuple(int(value) for value in loaded["FIR_lags"]), valid,
        )[0].numpy()
    assert np.allclose(replay, generated, atol=1.0e-8)


def test_fir_reconstruction_rho_endpoints_equal_full_replacement() -> None:
    torch.manual_seed(19)
    observed = torch.randn(2, 4, 48, dtype=torch.float64)
    latent = torch.randn(2, 2, 48, dtype=torch.float64)
    population = torch.randn(4, 2, 5, dtype=torch.float64)
    subject = torch.randn(4, 2, 5, dtype=torch.float64)
    lags = (-2, -1, 0, 1, 2)
    valid = torch.ones(2, 48, dtype=torch.bool)
    output0, correction0, effective0 = fir_full_replacement(
        observed, latent, population, subject, lags, 0.0, valid,
    )
    output1, correction1, effective1 = fir_full_replacement(
        observed, latent, population, subject, lags, 1.0, valid,
    )
    assert torch.equal(effective0, population[None].expand_as(effective0))
    assert torch.allclose(effective1, subject[None].expand_as(effective1), atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(correction0, fir_transfer_correction(population, latent, lags, valid))
    assert torch.allclose(correction1, fir_transfer_correction(subject, latent, lags, valid))
    assert torch.allclose(output0, observed - correction0)
    assert torch.allclose(output1, observed - correction1)


def test_fir_lags_are_defined_in_milliseconds_per_cell() -> None:
    audit = {"fir_lags_milliseconds": [-40, -20, 0, 20, 40]}
    assert _fir_lags(audit, 250.0) == (-10, -5, 0, 5, 10)
    assert _fir_lags(audit, 500.0) == (-20, -10, 0, 10, 20)


def test_fir_and_state_reliability_use_blocked_crossfit() -> None:
    rng = np.random.default_rng(23)
    eog = rng.normal(size=(2, 600))
    lags = (-2, -1, 0, 1, 2)
    truth_fir = rng.normal(scale=0.2, size=(4, 2, len(lags)))
    valid = torch.ones((1, eog.shape[1]), dtype=torch.bool)
    eeg_fir = fir_transfer_correction(
        torch.from_numpy(truth_fir), torch.from_numpy(eog[None]), lags, valid,
    )[0].numpy()
    fitted, alpha, scores, stability = _blocked_fir_crossfit(
        eeg_fir,
        eog,
        np.zeros_like(truth_fir),
        lags,
        1.0e-8,
        (0.0, 0.5, 1.0),
    )
    assert fitted.shape == truth_fir.shape
    assert alpha == 1.0 and scores[1.0] < scores[0.0]
    assert np.isfinite(stability)

    activity = np.sqrt(np.mean(np.square(eog), axis=0))
    threshold = np.quantile(activity, 0.75)
    active = activity >= threshold
    active_truth = rng.normal(scale=0.25, size=(4, 2))
    quiet_truth = rng.normal(scale=0.05, size=(4, 2))
    eeg_state = quiet_truth @ eog
    eeg_state[:, active] = active_truth @ eog[:, active]
    fitted_active, fitted_quiet, state_alpha, state_scores, state_stability = _blocked_state_crossfit(
        eeg_state,
        eog,
        np.zeros_like(active_truth),
        np.zeros_like(quiet_truth),
        1.0e-8,
        0.75,
        (0.0, 0.5, 1.0),
    )
    assert fitted_active.shape == fitted_quiet.shape == active_truth.shape
    assert state_alpha == 1.0 and state_scores[1.0] < state_scores[0.0]
    assert np.isfinite(state_stability)


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
