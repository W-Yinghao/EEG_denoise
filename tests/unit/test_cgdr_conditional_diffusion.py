"""Contracts for the exploratory task-matched conditional diffusion arm."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import yaml

from eeg_cgdr.models.conditional_diffusion import (
    OperatorConditionedEEGDiffusion,
)
from eeg_cgdr.models.deterministic_unet import (
    DeterministicUNetConfig,
    TaskMatchedDeterministicUNet,
)
from saddpm.diffusion.schedule import DiffusionConfig


CONFIG_PATH = Path("configs/cgdr/klados_stage3_conditional_diffusion.yaml")


def _backbone() -> DeterministicUNetConfig:
    return DeterministicUNetConfig(
        eeg_channels=3,
        signal_length=64,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        groupnorm_groups=8,
        dropout=0.0,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        attention_length=8,
        attention_heads=4,
    )


def _model() -> OperatorConditionedEEGDiffusion:
    torch.manual_seed(991)
    return OperatorConditionedEEGDiffusion(
        _backbone(),
        DiffusionConfig(
            num_timesteps=8,
            beta_start=1.0e-4,
            beta_end=0.02,
            schedule="linear",
        ),
        enforce_scientific_schedule=False,
    ).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    observed = torch.randn(2, 3, 64)
    x_t = torch.randn_like(observed)
    projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    attenuation = torch.linspace(0.05, 1.0, 64).repeat(2, 1)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :47] = True
    timesteps = torch.tensor([2, 6], dtype=torch.long)
    return x_t, timesteps, observed, projector, attenuation, mask


def test_conditional_protocol_freezes_fair_inputs_and_development_only_scope() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["protocol_id"] == "klados_operator_conditioned_diffusion_matched_v1"
    assert config["source_record_split"]["training"] == list(range(1, 31))
    assert config["source_record_split"]["development"] == [
        31,
        32,
        33,
        34,
        35,
        36,
        44,
        45,
    ]
    assert config["source_record_split"]["historical_evaluation_allowed"] is False
    fairness = config["fair_comparison_contract"]
    assert fairness["common_eligibility_rule"] == (
        "matching_p0_eligible_records_shared_by_all_operator_scopes"
    )
    assert fairness["target_optimizer_updates"] == (
        "exact_same_scope_deterministic_best_checkpoint_step"
    )
    assert fairness["different_training_objective_disclosed"] == (
        "epsilon_prediction_vs_deterministic_task_loss"
    )
    assert fairness["broad_diffusion_family_claim_allowed"] is False


def test_conditional_model_has_same_visible_information_as_deterministic() -> None:
    model = _model()
    forward = inspect.signature(model.forward).parameters
    assert tuple(forward) == (
        "x_t",
        "timesteps",
        "observed",
        "projector",
        "attenuation",
        "valid_time_mask",
    )
    assert model.visible_input_fields == TaskMatchedDeterministicUNet.visible_input_fields
    assert "clean_target" not in forward
    assert model.algorithm_state_fields == (
        "diffused_clean_state_x_t",
        "diffusion_timestep",
    )


def test_conditioning_stack_is_exactly_the_deterministic_stack() -> None:
    backbone = _backbone()
    deterministic = TaskMatchedDeterministicUNet(backbone).eval()
    conditional = _model()
    x_t, timesteps, observed, projector, attenuation, mask = _inputs()
    deterministic_inputs: list[torch.Tensor] = []
    conditional_inputs: list[torch.Tensor] = []
    first = deterministic.unet.stem.register_forward_pre_hook(
        lambda _module, args: deterministic_inputs.append(args[0].detach().clone())
    )
    second = conditional.unet.stem.register_forward_pre_hook(
        lambda _module, args: conditional_inputs.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            deterministic(
                observed,
                projector=projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )
            conditional(
                x_t,
                timesteps,
                observed=observed,
                projector=projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )
    finally:
        first.remove()
        second.remove()
    assert len(deterministic_inputs) == len(conditional_inputs) == 1
    channels = backbone.eeg_channels
    assert conditional.unet.stem.in_channels == (
        deterministic.unet.stem.in_channels + channels
    )
    torch.testing.assert_close(
        conditional_inputs[0][:, channels:],
        deterministic_inputs[0],
        rtol=0.0,
        atol=0.0,
    )


def test_conditional_diffusion_is_internally_padding_safe() -> None:
    model = _model()
    x_t, timesteps, observed, projector, attenuation, mask = _inputs()
    changed_x_t = x_t.clone()
    changed_observed = observed.clone()
    changed_x_t[:, :, 47:] = 1.0e6 * torch.randn_like(changed_x_t[:, :, 47:])
    changed_observed[:, :, 47:] = 1.0e6 * torch.randn_like(
        changed_observed[:, :, 47:]
    )
    with torch.no_grad():
        first = model(
            x_t,
            timesteps,
            observed=observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
        second = model(
            changed_x_t,
            timesteps,
            observed=changed_observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
    torch.testing.assert_close(first[:, :, :47], second[:, :, :47], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(first[:, :, 47:]) == 0
    assert torch.count_nonzero(second[:, :, 47:]) == 0


def test_conditional_training_loss_masks_target_and_has_finite_gradient() -> None:
    model = _model().train()
    x_t, timesteps, observed, projector, attenuation, mask = _inputs()
    clean = torch.randn_like(observed)
    noise = torch.randn_like(clean)
    loss = model.training_loss(
        clean,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
        timesteps=timesteps,
        noise=noise,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_conditional_ddim_steps_equal_actual_network_calls() -> None:
    model = _model()
    _x_t, _timesteps, observed, projector, attenuation, mask = _inputs()
    initial_noise = torch.randn_like(observed)
    result = model.sample_ddim(
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
        ddim_steps=4,
        eta=0.0,
        initial_noise=initial_noise,
    )
    assert result.network_calls == 4
    assert result.restored.shape == observed.shape
    assert torch.count_nonzero(result.restored[:, :, 47:]) == 0
