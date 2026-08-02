"""Acceptance tests for the repaired CGDR diffusion/sampler semantics.

These tests are submitted through the aggregate CPU Slurm validation job; they
must not be treated as scientific EEG evidence.
"""

from __future__ import annotations

from typing import Optional

import pytest
import torch
from torch import Tensor, nn

from eeg_cgdr.inference import (
    SAMPLER_CANDIDATES,
    CalibrationContextProjector,
    DatasetPopulationProjector,
    GuidanceStabilityConfig,
    GuidanceStepTrace,
    InformationMatchedOneStep,
    PopulationObservationState,
    PopulationOnlyInference,
    RepairedSamplerRunner,
    SamplerMechanism,
    dataset_population_and_context_states,
    rho_interpolated_precision_state,
    sampler_candidate,
)
from eeg_cgdr.models import CleanEEGDiffusionPrior, canonical_valid_time_mask
from saddpm.diffusion import (
    CGDR_MAX_TERMINAL_ALPHA_BAR,
    DiffusionConfig,
    GaussianDiffusion,
    validate_cgdr_schedule,
)
from saddpm.models.config import ModelConfig


def _joint_model_config(channels: int = 3) -> ModelConfig:
    return ModelConfig(
        in_channels=channels,
        out_channels=channels,
        signal_length=64,
        base_channels=8,
        channel_mults=[1, 2, 4],
        num_res_blocks=1,
        groupnorm_groups=8,
        dropout=0.0,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        attention_length=8,
        attention_heads=4,
    )


def _scientific_diffusion_config() -> DiffusionConfig:
    return DiffusionConfig(
        num_timesteps=1000,
        beta_start=1.0e-4,
        beta_end=0.02,
        schedule="linear",
    )


class _AnalyticJointPrior(nn.Module):
    """Small differentiable multichannel prior for exact sampler algebra tests."""

    def __init__(self, channels: int = 3, timesteps: int = 20) -> None:
        super().__init__()
        self.diffusion = GaussianDiffusion(
            DiffusionConfig(
                num_timesteps=timesteps,
                beta_start=1.0e-4,
                beta_end=0.02,
                schedule="linear",
            )
        )
        mix = torch.full((channels, channels), 0.04, dtype=torch.float64)
        mix.fill_diagonal_(0.12)
        self.register_buffer("mix", mix)
        self.network_calls = 0

    def predict_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        del timesteps
        self.network_calls += 1
        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        masked = x_t * mask.to(dtype=x_t.dtype)
        return torch.einsum("cd,bdl->bcl", self.mix.to(x_t), masked) * mask.to(
            dtype=x_t.dtype
        )

    def predict_clean(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Optional[Tensor] = None,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        epsilon = (
            self.predict_noise(x_t, timesteps, valid_time_mask=mask)
            if predicted_noise is None
            else predicted_noise
        )
        clean = self.diffusion.predict_xstart_from_eps(x_t, timesteps, epsilon)
        return clean * mask.to(dtype=clean.dtype)

    def noise_standard_deviation(self, timesteps: Tensor, ndim: int) -> Tensor:
        values = self.diffusion.sqrt_one_minus_alphas_cumprod.gather(0, timesteps)
        return values.reshape(timesteps.shape[0], *((1,) * (ndim - 1)))

    def score_from_epsilon(self, epsilon: Tensor, timesteps: Tensor) -> Tensor:
        return -epsilon / self.noise_standard_deviation(timesteps, epsilon.ndim)


def _analytic_state(length: int = 16) -> PopulationObservationState:
    observation = torch.linspace(-0.5, 0.5, 3 * length, dtype=torch.float64).reshape(
        1, 3, length
    )
    valid = torch.ones((1, length), dtype=torch.bool)
    valid[:, -4:] = False
    return PopulationObservationState(
        observation=observation,
        precision=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        energy_scale=0.3,
        valid_time_mask=valid,
        dataset_id="unit_fixture",
        montage_id="three_channel",
        precision_semantics="unit_quadratic",
    )


def test_scientific_schedule_is_1000_linear_and_terminally_noised() -> None:
    config = _scientific_diffusion_config()
    terminal = validate_cgdr_schedule(config)
    assert config.num_timesteps == 1000
    assert config.schedule == "linear"
    assert terminal <= CGDR_MAX_TERMINAL_ALPHA_BAR


def test_scientific_prior_rejects_old_t200_or_one_channel_contract() -> None:
    with pytest.raises(ValueError, match="at least two EEG channels"):
        CleanEEGDiffusionPrior(
            _joint_model_config(channels=1), _scientific_diffusion_config()
        )
    with pytest.raises(ValueError, match="T=1000"):
        CleanEEGDiffusionPrior(
            _joint_model_config(channels=3),
            DiffusionConfig(num_timesteps=200),
        )
    compatibility = CleanEEGDiffusionPrior(
        _joint_model_config(channels=1),
        DiffusionConfig(num_timesteps=20),
        prior_mode="independent_channel_ablation",
        enforce_scientific_schedule=False,
    )
    assert compatibility.prior_mode == "independent_channel_ablation"
    assert not compatibility.enforce_scientific_schedule


def test_joint_prior_padding_is_zero_and_cannot_affect_valid_output() -> None:
    torch.manual_seed(7)
    prior = CleanEEGDiffusionPrior(
        _joint_model_config(), _scientific_diffusion_config()
    ).eval()
    first = torch.randn(2, 3, 64)
    second = first.clone()
    second[:, :, 48:] = torch.randn_like(second[:, :, 48:]) * 1000.0
    valid = torch.zeros((2, 64), dtype=torch.bool)
    valid[:, :48] = True
    timesteps = torch.tensor([100, 700], dtype=torch.long)
    first_output = prior.predict_noise(first, timesteps, valid_time_mask=valid)
    second_output = prior.predict_noise(second, timesteps, valid_time_mask=valid)
    assert torch.equal(first_output[:, :, :48], second_output[:, :, :48])
    assert torch.count_nonzero(first_output[:, :, 48:]) == 0
    assert torch.count_nonzero(second_output[:, :, 48:]) == 0


def test_joint_prior_audit_requires_cross_channel_dependency() -> None:
    torch.manual_seed(11)
    prior = CleanEEGDiffusionPrior(
        _joint_model_config(), _scientific_diffusion_config()
    ).eval()
    probe = torch.randn(2, 3, 64)
    valid = torch.ones((2, 64), dtype=torch.bool)
    influence = prior.assert_cross_channel_dependency(
        probe,
        torch.tensor([200, 600], dtype=torch.long),
        valid_time_mask=valid,
        perturbation=1.0e-2,
        minimum_influence=1.0e-10,
    )
    assert influence.shape == (3, 3)
    off_diagonal = influence.clone()
    off_diagonal.fill_diagonal_(0.0)
    assert torch.all(off_diagonal.max(dim=1).values > 0.0)


def test_score_epsilon_conversion_has_explicit_negative_sign_and_round_trip() -> None:
    prior = CleanEEGDiffusionPrior(
        _joint_model_config(), _scientific_diffusion_config()
    ).eval()
    epsilon = torch.randn(2, 3, 64)
    timesteps = torch.tensor([50, 800], dtype=torch.long)
    score = prior.score_from_epsilon(epsilon, timesteps)
    sigma = prior.noise_standard_deviation(timesteps, epsilon.ndim)
    assert torch.allclose(score, -epsilon / sigma)
    assert torch.allclose(prior.epsilon_from_score(score, timesteps), epsilon)


def test_masked_training_loss_ignores_padding_values() -> None:
    torch.manual_seed(17)
    prior = CleanEEGDiffusionPrior(
        _joint_model_config(), _scientific_diffusion_config()
    ).eval()
    clean_a = torch.randn(2, 3, 64)
    clean_b = clean_a.clone()
    clean_b[:, :, 52:] = 1.0e5
    valid = torch.zeros((2, 64), dtype=torch.bool)
    valid[:, :52] = True
    noise = torch.randn_like(clean_a)
    noise[:, :, 52:] = -1.0e5
    timesteps = torch.tensor([20, 900], dtype=torch.long)
    loss_a = prior.training_loss(
        clean_a,
        timesteps=timesteps,
        noise=noise,
        valid_time_mask=valid,
    )
    loss_b = prior.training_loss(
        clean_b,
        timesteps=timesteps,
        noise=noise,
        valid_time_mask=valid,
    )
    assert torch.equal(loss_a, loss_b)


def test_observation_energy_ignores_invalid_padding() -> None:
    state = _analytic_state()
    first = torch.randn_like(state.observation)
    second = first.clone()
    second[:, :, -4:] = 1.0e9
    assert torch.equal(state.energy_per_sample(first), state.energy_per_sample(second))


def test_full_vjp_matches_directional_finite_difference() -> None:
    prior = _AnalyticJointPrior().double().eval()
    inference = PopulationOnlyInference(prior)  # type: ignore[arg-type]
    state = _analytic_state()
    x_t = torch.linspace(-1.0, 1.0, 48, dtype=torch.float64).reshape(1, 3, 16)
    timestep = torch.tensor([12], dtype=torch.long)
    direction = torch.cos(torch.arange(48, dtype=torch.float64)).reshape(1, 3, 16)
    direction = direction * state.valid_time_mask[:, None, :]
    direction = direction / torch.linalg.vector_norm(direction)
    result = inference.full_energy_vjp(
        x_t,
        timestep,
        energy=state.energy_per_sample,
        valid_time_mask=state.valid_time_mask,
    )

    def objective(value: Tensor) -> Tensor:
        epsilon = prior.predict_noise(
            value, timestep, valid_time_mask=state.valid_time_mask
        )
        clean = prior.predict_clean(
            value,
            timestep,
            epsilon,
            valid_time_mask=state.valid_time_mask,
        )
        return state.energy_per_sample(clean).sum()

    step = 1.0e-5
    finite_difference = (
        objective(x_t + step * direction) - objective(x_t - step * direction)
    ) / (2.0 * step)
    autodiff = torch.sum(result.energy_vjp * direction)
    assert torch.allclose(autodiff, finite_difference, rtol=2.0e-4, atol=2.0e-5)
    assert torch.count_nonzero(result.energy_vjp[:, :, -4:]) == 0


def test_guided_epsilon_has_correct_score_sign() -> None:
    prior = _AnalyticJointPrior().double().eval()
    inference = PopulationOnlyInference(  # type: ignore[arg-type]
        prior,
        stability=GuidanceStabilityConfig(
            normalize_by_residual_dimension=False,
            trust_radius_ratio=1.0e12,
        ),
    )
    state = _analytic_state()
    x_t = torch.randn((1, 3, 16), dtype=torch.float64)
    timestep = torch.tensor([9], dtype=torch.long)
    result = inference.full_energy_vjp(
        x_t,
        timestep,
        energy=state.energy_per_sample,
        valid_time_mask=state.valid_time_mask,
    )
    guided = inference.guided_epsilon(
        x_t,
        timestep,
        energy=state.energy_per_sample,
        valid_time_mask=state.valid_time_mask,
    )
    sigma = prior.noise_standard_deviation(timestep, x_t.ndim)
    assert torch.allclose(
        guided,
        result.prior_epsilon + sigma * result.energy_vjp,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    guided_score = prior.score_from_epsilon(guided, timestep)
    prior_score = prior.score_from_epsilon(result.prior_epsilon, timestep)
    assert torch.allclose(
        guided_score,
        prior_score - result.energy_vjp,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_ddim_steps_equal_network_calls_and_trace_rows() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    inference = PopulationOnlyInference(prior)  # type: ignore[arg-type]
    state = _analytic_state()
    trace: list[GuidanceStepTrace] = []
    output = inference.sample(state, seed=123, ddim_steps=5, guidance_trace=trace)
    assert prior.network_calls == 5
    assert len(trace) == 5
    assert [item.timestep for item in trace] == prior.diffusion.ddim_timesteps(5)
    assert all(item.network_evaluations == 1 for item in trace)
    assert torch.count_nonzero(output[:, :, -4:]) == 0


def test_rho_zero_and_direct_pop_have_identical_trajectories_without_context() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    inference = PopulationOnlyInference(prior)  # type: ignore[arg-type]
    state = _analytic_state()
    initial = inference.make_initial_noise(state, seed=991)
    direct_trace: list[GuidanceStepTrace] = []
    direct = inference.sample(
        state,
        initial_noise=initial,
        ddim_steps=6,
        guidance_trace=direct_trace,
    )
    calls_after_direct = prior.network_calls
    context_calls = 0

    def forbidden_context():
        nonlocal context_calls
        context_calls += 1
        raise AssertionError("rho=0 constructed a context-specific state")

    rho_trace: list[GuidanceStepTrace] = []
    rho_zero = inference.sample_cgdr(
        state,
        rho=0.0,
        calibration_accepted=True,
        context_state_factory=forbidden_context,
        initial_noise=initial,
        ddim_steps=6,
        guidance_trace=rho_trace,
    )
    assert context_calls == 0
    assert prior.network_calls - calls_after_direct == 6
    assert torch.equal(direct, rho_zero)
    assert direct_trace == rho_trace


def test_full_generation_initial_state_is_exact_seeded_masked_standard_normal() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    inference = PopulationOnlyInference(prior)  # type: ignore[arg-type]
    state = _analytic_state(length=128)
    actual = inference.make_initial_noise(state, seed=31415)
    generator = torch.Generator(device=state.observation.device)
    generator.manual_seed(31415)
    expected = torch.randn(
        state.observation.shape,
        dtype=state.observation.dtype,
        device=state.observation.device,
        generator=generator,
    ) * state.valid_time_mask[:, None, :]
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[:, :, -4:]) == 0


def test_full_generation_valid_initial_marginal_is_empirical_standard_normal() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    inference = PopulationOnlyInference(prior)  # type: ignore[arg-type]
    state = _analytic_state(length=8192)
    initial = inference.make_initial_noise(state, seed=271828)
    valid_values = initial[state.valid_time_mask[:, None, :].expand_as(initial)]
    assert abs(float(valid_values.mean())) < 0.02
    assert abs(float(valid_values.std(unbiased=True)) - 1.0) < 0.02
    assert torch.count_nonzero(initial[:, :, -4:]) == 0


def test_generic_ddim_uses_exact_requested_calls() -> None:
    diffusion = GaussianDiffusion(DiffusionConfig(num_timesteps=20))
    calls: list[int] = []

    def epsilon(x_t: Tensor, timestep: Tensor) -> Tensor:
        calls.append(int(timestep[0]))
        return torch.zeros_like(x_t)

    valid = torch.ones((1, 16), dtype=torch.bool)
    valid[:, -3:] = False
    output = diffusion.ddim_sample_loop(
        epsilon,
        (1, 3, 16),
        torch.device("cpu"),
        ddim_steps=6,
        x_t=torch.randn(1, 3, 16),
        valid_time_mask=valid,
    )
    assert calls == diffusion.ddim_timesteps(6)
    assert len(calls) == 6
    assert torch.count_nonzero(output[:, :, -3:]) == 0


def test_m5_one_step_supports_frame_precision_and_valid_mask() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    one_step = InformationMatchedOneStep(prior)  # type: ignore[arg-type]
    state = _analytic_state()
    identity = torch.eye(3, dtype=torch.float64)
    precision = identity.reshape(1, 1, 3, 3).expand(1, 16, 3, 3).clone()
    output = one_step.restore(
        observation=state.observation,
        channel_precision=precision,
        seed=5,
        timestep=10,
        proximal_strength=0.2,
        valid_time_mask=state.valid_time_mask,
    )
    assert output.shape == state.observation.shape
    assert prior.network_calls == 1
    assert torch.count_nonzero(output[:, :, -4:]) == 0


def test_sampler_candidate_names_are_exact_and_operator_orthogonal() -> None:
    expected = {
        "M0": SamplerMechanism.M0,
        "M1": SamplerMechanism.M1,
        "M2": SamplerMechanism.M2,
        "M3": SamplerMechanism.M3,
        "M4": SamplerMechanism.M4,
        "M5": SamplerMechanism.M5,
        "WP": SamplerMechanism.WP,
    }
    assert {item.candidate_id: item.mechanism for item in SAMPLER_CANDIDATES} == expected
    assert all(item.implementation_status == "implemented" for item in SAMPLER_CANDIDATES)
    forbidden_operator_names = {"matching", "population", "wrong", "shuffled", "oracle"}
    assert all(
        not forbidden_operator_names.intersection(item.mechanism.value.split("_"))
        for item in SAMPLER_CANDIDATES
    )
    for candidate_id, mechanism in expected.items():
        registered = f"{candidate_id}_{mechanism.value}"
        assert sampler_candidate(candidate_id).mechanism == mechanism
        assert sampler_candidate(registered).mechanism == mechanism
    with pytest.raises(ValueError, match="unknown sampler candidate"):
        sampler_candidate("M0_matching_full_generation_guided_ddim")


@pytest.mark.parametrize(
    "mechanism",
    [
        SamplerMechanism.M0,
        SamplerMechanism.M1,
        SamplerMechanism.M2,
        SamplerMechanism.M3,
        SamplerMechanism.M4,
        SamplerMechanism.M5,
    ],
)
def test_endpoint_and_legacy_repaired_sampler_candidates_are_callable(
    mechanism: SamplerMechanism,
) -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    inference = PopulationOnlyInference(prior)  # type: ignore[arg-type]
    runner = RepairedSamplerRunner(inference)
    state = _analytic_state()
    projector = torch.diag(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    result = runner.run(
        mechanism,
        state,
        seed=19,
        ddim_steps=1 if mechanism == SamplerMechanism.M5 else 5,
        projector=(
            projector
            if mechanism in (
                SamplerMechanism.M2,
                SamplerMechanism.M3,
                SamplerMechanism.M4,
                SamplerMechanism.M5,
            )
            else None
        ),
        warm_start_timestep=12 if mechanism == SamplerMechanism.M1 else None,
        one_step_timestep=10 if mechanism == SamplerMechanism.M5 else None,
        proximal_strength=0.3,
    )
    expected_calls = 1 if mechanism == SamplerMechanism.M5 else 5
    assert result.restored.shape == state.observation.shape
    assert result.candidate_id == mechanism.name
    assert result.mechanism_name == mechanism.value
    assert result.network_evaluations == expected_calls
    assert len(result.trace) == expected_calls
    assert torch.count_nonzero(result.restored[:, :, -4:]) == 0
    assert all(item.finite_fraction == 1.0 for item in result.trace)
    assert all(0.0 <= item.clipping_fraction <= 1.0 for item in result.trace)
    if mechanism in (SamplerMechanism.M2, SamplerMechanism.M3):
        assert result.trace[-1].q_residual_after is not None
        assert result.trace[-1].q_residual_after < 1.0e-10
    if mechanism == SamplerMechanism.M3:
        assert all(
            item.q_residual_after is not None and item.q_residual_after < 1.0e-10
            for item in result.trace
        )


def test_explicit_psd_wrho_proximal_candidate_is_callable_at_intermediate_rho() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    runner = RepairedSamplerRunner(
        PopulationOnlyInference(prior)  # type: ignore[arg-type]
    )
    observation = torch.linspace(
        -0.5, 0.5, 48, dtype=torch.float64
    ).reshape(1, 3, 16)
    valid = torch.ones((1, 16), dtype=torch.float64)
    attenuation = torch.full((1, 16), 0.5, dtype=torch.float64)
    pi0 = DatasetPopulationProjector(
        "fixture",
        "three_channel",
        torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)),
        "training",
    )
    pic = CalibrationContextProjector(
        "fixture",
        "three_channel",
        torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)),
        "support",
    )
    population, context = dataset_population_and_context_states(
        observation,
        attenuation=attenuation,
        valid_weight=valid,
        population_projector=pi0,
        context_projector=pic,
        base_precision=1.0,
        energy_scale=0.3,
    )
    state = rho_interpolated_precision_state(
        population,
        rho=0.25,
        calibration_accepted=True,
        context_state_factory=lambda: context,
    )
    result = runner.run(
        SamplerMechanism.WP,
        state,
        seed=21,
        ddim_steps=5,
        proximal_strength=0.3,
    )
    expected_semantics = (
        "intermediate_rho_psd_Wrho_quadratic_proximal_not_hard_Q"
    )
    assert result.candidate_id == "WP"
    assert result.consistency_semantics == expected_semantics
    assert len(result.trace) == 5
    assert all(item.consistency_semantics == expected_semantics for item in result.trace)
    assert all(item.q_residual_after is None for item in result.trace)
    assert all(item.precision_residual_after is not None for item in result.trace)


def test_m2_final_hard_q_consistency_has_the_exact_projector_identity() -> None:
    observation = (
        torch.arange(36, dtype=torch.float64).reshape(1, 3, 12) / 8.0
    )
    state = PopulationObservationState(
        observation=observation,
        precision=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        energy_scale=0.3,
        valid_time_mask=torch.ones((1, 12), dtype=torch.bool),
        dataset_id="unit_fixture",
        montage_id="three_channel",
        precision_semantics="unit_quadratic",
    )
    x_ddim = (
        torch.arange(35, -1, -1, dtype=torch.float64).reshape_as(observation)
        / 16.0
    )
    projector = torch.diag(
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    )
    complement = torch.eye(3, dtype=torch.float64) - projector

    output = RepairedSamplerRunner._hard_q(x_ddim, state, projector)
    p_x_ddim = torch.einsum("cd,bdl->bcl", projector, x_ddim)
    q_y = torch.einsum("cd,bdl->bcl", complement, state.observation)

    assert torch.equal(output, p_x_ddim + q_y)
    assert torch.equal(
        torch.einsum("cd,bdl->bcl", complement, output - state.observation),
        torch.zeros_like(output),
    )
    assert torch.equal(
        torch.einsum("cd,bdl->bcl", projector, output),
        p_x_ddim,
    )


def test_trace_marks_first_middle_last_and_records_required_norms() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    runner = RepairedSamplerRunner(
        PopulationOnlyInference(prior)  # type: ignore[arg-type]
    )
    state = _analytic_state()
    result = runner.run(
        SamplerMechanism.M0,
        state,
        seed=23,
        ddim_steps=5,
    )
    assert [item.checkpoint_label for item in result.trace] == [
        "first",
        "intermediate",
        "middle",
        "intermediate",
        "last",
    ]
    for item in (result.trace[0], result.trace[2], result.trace[-1]):
        assert item.prior_score_l2 >= 0.0
        assert item.prior_epsilon_l2 >= 0.0
        assert item.clean_estimate_l2 >= 0.0
        assert item.raw_energy_vjp_l2 >= 0.0
        assert item.normalized_energy_vjp_l2 >= 0.0
        assert item.guided_epsilon_l2 >= 0.0
        assert item.guided_score_l2 >= 0.0
        assert item.sample_norm_before_consistency is not None
        assert item.sample_norm_after_consistency is not None
        assert item.state_norm_before_ddim is not None
        assert item.state_norm_after_ddim is not None
        assert item.consistency_update_l2 is not None
        assert item.consistency_update_l2 >= 0.0
    dimensions = float(state.residual_dimensions()[0])
    assert result.trace[0].normalized_energy_vjp_l2 == pytest.approx(
        result.trace[0].raw_energy_vjp_l2 / dimensions,
        rel=1.0e-10,
        abs=1.0e-12,
    )


def test_long_registered_sampler_name_is_callable() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    runner = RepairedSamplerRunner(
        PopulationOnlyInference(prior)  # type: ignore[arg-type]
    )
    result = runner.run(
        "M0_full_generation_guided_ddim",
        _analytic_state(),
        seed=29,
        ddim_steps=2,
    )
    assert result.candidate_id == "M0"
    assert result.mechanism_name == "full_generation_guided_ddim"
    assert result.network_evaluations == 2


def test_guidance_trust_radius_clips_oversized_update() -> None:
    prior = _AnalyticJointPrior(timesteps=20).double().eval()
    inference = PopulationOnlyInference(  # type: ignore[arg-type]
        prior,
        stability=GuidanceStabilityConfig(trust_radius_ratio=1.0e-12),
    )
    state = _analytic_state()
    trace: list[GuidanceStepTrace] = []
    inference.sample(state, seed=31, ddim_steps=2, guidance_trace=trace)
    assert len(trace) == 2
    assert all(item.clipping_fraction == 1.0 for item in trace)
