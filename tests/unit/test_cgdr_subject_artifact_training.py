"""Tensor-only training contracts for subject-calibrated artifact models."""

from __future__ import annotations

import inspect

import torch
from torch import nn

from eeg_cgdr.experiments.subject_artifact_training import (
    ArtifactTrainingBudget,
    CheckpointableEMA,
    DevelopmentScore,
    SubjectArtifactTensorBatch,
    artifact_train_step,
    build_shared_minibatch_schedule,
    development_candidate_is_better,
    development_converged,
    load_artifact_training_checkpoint,
    resume_artifact_training_checkpoint,
    run_v1_fixed_batch_overfit,
    save_artifact_training_checkpoint,
    stratified_timesteps,
)
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)


class _DisabledScaler:
    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0

    def state_dict(self) -> dict[str, object]:
        return {"enabled": False}

    def load_state_dict(self, state: dict[str, object]) -> None:
        assert state == {"enabled": False}


class _TinyBackbone(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(
        self,
        value: torch.Tensor,
        _timestep: torch.Tensor,
        *,
        valid_time_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.projection(value) * valid_time_mask.to(value.dtype)


def _batch(*, size: int = 5, zero_target: bool = False) -> SubjectArtifactTensorBatch:
    observed = torch.linspace(-1.0, 1.0, size * 4 * 8).reshape(size, 4, 8)
    target = torch.linspace(0.5, -0.5, size * 3 * 8).reshape(size, 3, 8)
    if zero_target:
        target = torch.zeros_like(target)
    normalized = torch.tensor(
        [
            [1.0, 0.0, 0.25],
            [0.0, 1.0, 0.50],
            [0.25, 0.0, 0.75],
            [0.0, 0.25, 0.25],
        ]
    ).expand(size, -1, -1).clone()
    scale = torch.tensor([2.0, 3.0, 4.0]).expand(size, -1).clone()
    valid = torch.ones(size, 8, dtype=torch.bool)
    valid[:, -2:] = False
    target[:, :, -2:] = 0.0
    return SubjectArtifactTensorBatch(
        observed=observed,
        target_standardized_latent=target,
        full_transfer=normalized * scale[:, None, :],
        normalized_transfer=normalized,
        transfer_scale=scale,
        singular_values=torch.tensor([5.0, 2.0, 0.5])
        .expand(size, -1)
        .clone(),
        rank=torch.full((size,), 2, dtype=torch.long),
        rho=torch.full((size,), 0.75),
        calibration_duration_seconds=torch.full((size,), 30.0),
        channel_mask=torch.ones(size, 4, dtype=torch.bool),
        valid_time_mask=valid,
    )


def _deterministic() -> DeterministicArtifactEstimator:
    model = DeterministicArtifactEstimator(
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
        )
    )
    model.unet = _TinyBackbone(model.conditioning_channels, 3)
    return model


def _diffusion() -> ArtifactLatentDiffusion:
    model = ArtifactLatentDiffusion(
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
    model.unet = _TinyBackbone(3 + model.conditioning_channels, 3)
    return model


def test_tensor_batch_selection_keeps_every_context_field_aligned() -> None:
    batch = _batch()
    selected = batch.select(torch.tensor([4, 1, 4]))

    assert selected.batch_size == 3
    torch.testing.assert_close(selected.observed[0], batch.observed[4])
    torch.testing.assert_close(selected.full_transfer[1], batch.full_transfer[1])
    torch.testing.assert_close(selected.rho, torch.tensor([0.75, 0.75, 0.75]))
    assert selected.valid_time_mask.shape == (3, 8)


def test_one_shared_minibatch_schedule_is_reusable_by_both_model_arms() -> None:
    first = build_shared_minibatch_schedule(
        sample_count=5,
        batch_size=3,
        updates=7,
        seed=20260811,
    )
    second = build_shared_minibatch_schedule(
        sample_count=5,
        batch_size=3,
        updates=7,
        seed=20260811,
    )

    torch.testing.assert_close(first.indices, second.indices)
    assert first.indices.shape == (7, 3)
    assert torch.equal(first.at(4), second.at(4))
    assert set(first.indices[:2].flatten().tolist()) == set(range(5))


def test_stratified_timesteps_cover_low_and_high_noise_at_each_update() -> None:
    values = stratified_timesteps(
        num_timesteps=1000,
        batch_size=8,
        seed=20260811,
        update_index=25,
        device=torch.device("cpu"),
    )
    assert values.shape == (8,)
    assert int(values.min()) < 125
    assert int(values.max()) >= 875
    assert torch.equal(
        values,
        stratified_timesteps(
            num_timesteps=1000,
            batch_size=8,
            seed=20260811,
            update_index=25,
            device=torch.device("cpu"),
        ),
    )


def test_three_seed_budget_freezes_equal_compute_and_development_best_rule() -> None:
    budget = ArtifactTrainingBudget(
        seeds=(20260811, 20260812, 20260813),
        equal_compute_updates=8,
        maximum_updates=12,
        batch_size=3,
        validation_interval_updates=2,
        convergence_patience_updates=3,
        convergence_minimum_relative_improvement=0.01,
    )
    schedules = budget.schedules(sample_count=5)
    assert set(schedules) == {20260811, 20260812, 20260813}
    assert all(item.updates == 12 for item in schedules.values())
    best = DevelopmentScore(6, artifact_latent_mse=1.0, x0_reconstruction_mse=0.5)
    primary_win = DevelopmentScore(
        8,
        artifact_latent_mse=0.98,
        x0_reconstruction_mse=0.9,
    )
    tie_break_win = DevelopmentScore(
        8,
        artifact_latent_mse=0.995,
        x0_reconstruction_mse=0.4,
    )
    assert development_candidate_is_better(
        primary_win,
        best,
        minimum_relative_improvement=0.01,
    )
    assert development_candidate_is_better(
        tie_break_win,
        best,
        minimum_relative_improvement=0.01,
    )
    assert not development_converged(current_step=7, best_step=3, budget=budget)
    assert development_converged(current_step=9, best_step=6, budget=budget)


def test_train_step_supports_deterministic_and_diffusion_on_the_same_batch() -> None:
    batch = _batch(size=4)
    scaler = _DisabledScaler()
    deterministic = _deterministic()
    deterministic_optimizer = torch.optim.AdamW(deterministic.parameters(), lr=1.0e-3)
    deterministic_result = artifact_train_step(
        deterministic,
        batch,
        model_kind="deterministic",
        optimizer=deterministic_optimizer,
        scaler=scaler,
        ema=CheckpointableEMA(deterministic, decay=0.9),
        update_index=0,
        training_seed=20260811,
        gradient_clip_norm=1.0,
        mixed_precision=False,
    )

    diffusion = _diffusion()
    diffusion_optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1.0e-3)
    diffusion_result = artifact_train_step(
        diffusion,
        batch,
        model_kind="diffusion",
        optimizer=diffusion_optimizer,
        scaler=scaler,
        ema=CheckpointableEMA(diffusion, decay=0.9),
        update_index=0,
        training_seed=20260811,
        gradient_clip_norm=1.0,
        mixed_precision=False,
    )

    assert deterministic_result.optimizer_step_succeeded
    assert diffusion_result.optimizer_step_succeeded
    assert deterministic_result.timestep_mean is None
    assert diffusion_result.timestep_minimum is not None
    assert diffusion_result.timestep_maximum is not None
    assert "x0_mse" in diffusion_result.metrics


def test_artifact_checkpoint_restores_model_optimizer_scaler_and_ema(tmp_path) -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scaler = _DisabledScaler()
    ema = CheckpointableEMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.add_(0.5)
    ema.update(model)
    saved_model = {name: value.clone() for name, value in model.state_dict().items()}
    saved_ema = {name: value.clone() for name, value in ema.shadow.items()}
    checkpoint = tmp_path / "subject_artifact.pt"
    save_artifact_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        epoch=2,
        step=17,
        contract={"protocol": "unit"},
        history=[{"step": 17, "loss": 0.25}],
        extra={"best_step": 15},
    )
    payload = load_artifact_training_checkpoint(checkpoint)
    assert "ema_state" in payload["extra"]
    assert payload["extra"]["loss_curve"] == [{"step": 17, "loss": 0.25}]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for value in ema.shadow.values():
            value.zero_()

    state = resume_artifact_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        expected_contract={"protocol": "unit"},
        map_location="cpu",
    )

    assert state.epoch == 2 and state.step == 17
    assert state.history == ({"step": 17, "loss": 0.25},)
    assert state.extra == {"best_step": 15}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_model[name])
    for name, value in ema.shadow.items():
        torch.testing.assert_close(value, saved_ema[name])


def test_v1_helper_requires_explicit_identity_batch_and_emits_validity_rows() -> None:
    signature = inspect.signature(run_v1_fixed_batch_overfit)
    assert signature.parameters["identity_batch"].default is inspect.Parameter.empty
    model = _deterministic()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    result = run_v1_fixed_batch_overfit(
        model,
        _batch(size=3),
        identity_batch=_batch(size=3, zero_target=True),
        model_kind="deterministic",
        model_id="deterministic",
        target_id="same_artifact_latent",
        optimizer=optimizer,
        scaler=_DisabledScaler(),
        ema=CheckpointableEMA(model, decay=0.9),
        updates=2,
        training_seed=20260811,
        validation_timesteps=(1, 7, 14),
        gradient_clip_norm=1.0,
        mixed_precision=False,
        check_interval_updates=1,
    )

    assert result.updates_completed == 2
    assert set(result.x0_rmse_by_timestep) == {1, 7, 14}
    assert set(result.zero_artifact_relative_change_by_timestep) == {1, 7, 14}
    assert len(result.validity_rows()) == 3
    assert all(row["initial_loss"] == result.initial_loss for row in result.validity_rows())
