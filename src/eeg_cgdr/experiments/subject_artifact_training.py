"""Small tensor-only training core for subject-calibrated artifact models.

Data loading, split selection and experiment routing deliberately live outside
this module.  A runner supplies already leakage-safe tensor batches; this file
provides one shared minibatch schedule, truthful optimizer steps, EMA-aware
checkpoints and the fixed-batch V1 diagnostic for both artifact estimators.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from eeg_cgdr.models.artifact_latent_deterministic import (
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusion
from eeg_cgdr.models.artifact_latent_inference import canonical_artifact_delta
from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)
from eeg_cgdr.training.optimizer import scaler_optimizer_step_succeeded
from saddpm.utils.ema import EMA


ModelKind = Literal["deterministic", "diffusion"]


def _finite_float_tensor(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or not value.dtype.is_floating_point:
        raise TypeError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN/Inf")


@dataclass(frozen=True)
class SubjectArtifactTensorBatch:
    """All tensors visible to one deterministic or diffusion update."""

    observed: Tensor
    target_standardized_latent: Tensor
    full_transfer: Tensor
    normalized_transfer: Tensor
    transfer_scale: Tensor
    singular_values: Tensor
    rank: Tensor
    rho: Tensor
    calibration_duration_seconds: Tensor
    channel_mask: Tensor
    valid_time_mask: Tensor

    def __post_init__(self) -> None:
        _finite_float_tensor(self.observed, name="observed")
        _finite_float_tensor(
            self.target_standardized_latent,
            name="target_standardized_latent",
        )
        if self.observed.ndim != 3 or self.target_standardized_latent.ndim != 3:
            raise ValueError("observed and target must have shape (B,C_or_E,L)")
        batch, channels, length = self.observed.shape
        if batch < 1 or channels < 2 or length < 1:
            raise ValueError("subject-artifact batch dimensions must be nonempty")
        if (
            self.target_standardized_latent.shape[0] != batch
            or self.target_standardized_latent.shape[-1] != length
        ):
            raise ValueError("observed and target batch/time dimensions differ")
        if self.target_standardized_latent.dtype != self.observed.dtype:
            raise ValueError("observed and target must use the same floating dtype")
        latent = self.target_standardized_latent.shape[1]
        if not 1 <= latent <= channels:
            raise ValueError("artifact latent dimension must lie in [1,EEG channels]")
        for name, value in (
            ("full_transfer", self.full_transfer),
            ("normalized_transfer", self.normalized_transfer),
            ("transfer_scale", self.transfer_scale),
            ("singular_values", self.singular_values),
            ("rho", self.rho),
            ("calibration_duration_seconds", self.calibration_duration_seconds),
        ):
            _finite_float_tensor(value, name=name)
        if any(
            value.device != self.observed.device
            for value in (
                self.target_standardized_latent,
                self.full_transfer,
                self.normalized_transfer,
                self.transfer_scale,
                self.singular_values,
                self.rank,
                self.rho,
                self.calibration_duration_seconds,
                self.channel_mask,
                self.valid_time_mask,
            )
        ):
            raise ValueError("every batch tensor must share one device")
        if self.full_transfer.shape != (batch, channels, latent):
            raise ValueError("full_transfer must have shape (B,C,E)")
        if self.normalized_transfer.shape != self.full_transfer.shape:
            raise ValueError("normalized_transfer shape differs from full_transfer")
        if self.transfer_scale.shape != (batch, latent):
            raise ValueError("transfer_scale must have shape (B,E)")
        if self.singular_values.shape != (batch, latent):
            raise ValueError("singular_values must have shape (B,E)")
        if self.rank.shape != (batch,):
            raise ValueError("rank must have shape (B,)")
        if self.rho.shape != (batch,):
            raise ValueError("rho must have shape (B,)")
        if self.calibration_duration_seconds.shape != (batch,):
            raise ValueError("calibration duration must have shape (B,)")
        if self.channel_mask.shape != (batch, channels):
            raise ValueError("channel_mask must have shape (B,C)")
        if self.valid_time_mask.shape not in ((batch, length), (batch, 1, length)):
            raise ValueError("valid_time_mask must have shape (B,L) or (B,1,L)")
        if bool((self.transfer_scale <= 0).any()):
            raise ValueError("transfer_scale must be positive")
        if bool(((self.rho < 0) | (self.rho > 1)).any()):
            raise ValueError("rho must lie in [0,1]")
        if bool((self.calibration_duration_seconds < 0).any()):
            raise ValueError("calibration duration cannot be negative")
        if self.rank.dtype == torch.bool or bool(
            ((self.rank < 1) | (self.rank > latent)).any()
        ):
            raise ValueError("rank must contain integers in [1,E]")
        if self.rank.dtype.is_floating_point and not bool(
            (self.rank == self.rank.round()).all()
        ):
            raise ValueError("rank must contain integer values")
        for name, value in (
            ("channel_mask", self.channel_mask),
            ("valid_time_mask", self.valid_time_mask),
        ):
            if value.dtype != torch.bool and not bool(
                ((value == 0) | (value == 1)).all()
            ):
                raise ValueError(f"{name} must contain only boolean/0/1 values")
        if not bool(self.channel_mask.bool().any(dim=1).all()):
            raise ValueError("every sample must retain an EEG channel")
        valid = self.valid_time_mask.reshape(batch, -1).bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample must retain a valid time point")

    @property
    def batch_size(self) -> int:
        return int(self.observed.shape[0])

    def select(self, indices: Tensor) -> "SubjectArtifactTensorBatch":
        index = torch.as_tensor(indices, device=self.observed.device, dtype=torch.long)
        if index.ndim != 1 or index.numel() < 1:
            raise ValueError("minibatch indices must be a nonempty vector")
        if bool(((index < 0) | (index >= self.batch_size)).any()):
            raise IndexError("minibatch index is outside the tensor batch")
        return SubjectArtifactTensorBatch(
            **{
                item.name: getattr(self, item.name).index_select(0, index)
                for item in fields(self)
            }
        )

    def to(self, device: torch.device | str) -> "SubjectArtifactTensorBatch":
        return SubjectArtifactTensorBatch(
            **{
                item.name: getattr(self, item.name).to(device, non_blocking=True)
                for item in fields(self)
            }
        )

    def model_kwargs(self) -> dict[str, Tensor]:
        return {
            "full_transfer": self.full_transfer,
            "normalized_transfer": self.normalized_transfer,
            "transfer_scale": self.transfer_scale,
            "singular_values": self.singular_values,
            "rank": self.rank,
            "rho": self.rho,
            "calibration_duration_seconds": self.calibration_duration_seconds,
            "channel_mask": self.channel_mask,
            "valid_time_mask": self.valid_time_mask,
        }


@dataclass(frozen=True)
class SharedMinibatchSchedule:
    """Frozen indices consumed identically by deterministic and diffusion arms."""

    sample_count: int
    batch_size: int
    updates: int
    seed: int
    indices: Tensor

    def __post_init__(self) -> None:
        if self.sample_count < 1 or self.batch_size < 1 or self.updates < 1:
            raise ValueError("schedule dimensions must be positive")
        if self.indices.device.type != "cpu" or self.indices.dtype != torch.long:
            raise ValueError("schedule indices must be a CPU long tensor")
        if self.indices.shape != (self.updates, self.batch_size):
            raise ValueError("schedule index shape differs from update budget")
        if bool(((self.indices < 0) | (self.indices >= self.sample_count)).any()):
            raise ValueError("schedule contains an out-of-range sample index")

    def at(self, update_index: int) -> Tensor:
        if not 0 <= int(update_index) < self.updates:
            raise IndexError("update index is outside the shared schedule")
        return self.indices[int(update_index)]


def build_shared_minibatch_schedule(
    *,
    sample_count: int,
    batch_size: int,
    updates: int,
    seed: int,
) -> SharedMinibatchSchedule:
    """Build repeated seeded permutations without model-specific resampling."""

    if min(int(sample_count), int(batch_size), int(updates)) < 1:
        raise ValueError("sample_count, batch_size and updates must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    required = int(batch_size) * int(updates)
    pieces: list[Tensor] = []
    available = 0
    while available < required:
        permutation = torch.randperm(int(sample_count), generator=generator)
        pieces.append(permutation)
        available += int(permutation.numel())
    indices = torch.cat(pieces)[:required].reshape(int(updates), int(batch_size))
    return SharedMinibatchSchedule(
        sample_count=int(sample_count),
        batch_size=int(batch_size),
        updates=int(updates),
        seed=int(seed),
        indices=indices,
    )


@dataclass(frozen=True)
class ArtifactTrainingBudget:
    """Frozen three-seed equal-compute and development-convergence policy."""

    seeds: tuple[int, int, int]
    equal_compute_updates: int
    maximum_updates: int
    batch_size: int
    validation_interval_updates: int
    convergence_patience_updates: int
    convergence_minimum_relative_improvement: float

    def __post_init__(self) -> None:
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("artifact training requires exactly three unique seeds")
        if min(
            self.equal_compute_updates,
            self.maximum_updates,
            self.batch_size,
            self.validation_interval_updates,
            self.convergence_patience_updates,
        ) < 1:
            raise ValueError("training budget fields must be positive")
        if self.maximum_updates < self.equal_compute_updates:
            raise ValueError("maximum updates cannot precede equal-compute updates")
        improvement = float(self.convergence_minimum_relative_improvement)
        if not math.isfinite(improvement) or not 0.0 <= improvement < 1.0:
            raise ValueError("minimum relative improvement must lie in [0,1)")

    def schedules(self, sample_count: int) -> dict[int, SharedMinibatchSchedule]:
        return {
            int(seed): build_shared_minibatch_schedule(
                sample_count=sample_count,
                batch_size=self.batch_size,
                updates=self.maximum_updates,
                seed=int(seed),
            )
            for seed in self.seeds
        }


@dataclass(frozen=True)
class DevelopmentScore:
    step: int
    artifact_latent_mse: float
    x0_reconstruction_mse: float

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("development score step must be positive")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.artifact_latent_mse, self.x0_reconstruction_mse)
        ):
            raise ValueError("development losses must be finite and nonnegative")


def development_candidate_is_better(
    candidate: DevelopmentScore,
    best: DevelopmentScore | None,
    *,
    minimum_relative_improvement: float,
) -> bool:
    """Apply frozen latent-MSE-first, x0-MSE-second checkpoint selection."""

    if best is None:
        return True
    tolerance = best.artifact_latent_mse * float(minimum_relative_improvement)
    primary_delta = best.artifact_latent_mse - candidate.artifact_latent_mse
    if primary_delta > tolerance:
        return True
    if abs(primary_delta) <= tolerance:
        return candidate.x0_reconstruction_mse < best.x0_reconstruction_mse
    return False


def development_converged(
    *,
    current_step: int,
    best_step: int,
    budget: ArtifactTrainingBudget,
) -> bool:
    """Never stop before equal compute; then apply the frozen patience budget."""

    if current_step < budget.equal_compute_updates:
        return False
    return current_step - best_step >= budget.convergence_patience_updates


class CheckpointableEMA(EMA):
    """Repository EMA with an explicit checkpoint and temporary-use contract."""

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": float(self.decay),
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        decay = float(state.get("decay", float("nan")))
        shadow = state.get("shadow")
        if not math.isfinite(decay) or decay != float(self.decay):
            raise ValueError("EMA checkpoint decay differs from the current protocol")
        if not isinstance(shadow, Mapping) or set(shadow) != set(self.shadow):
            raise ValueError("EMA checkpoint parameter names differ from the model")
        for name, current in self.shadow.items():
            saved = torch.as_tensor(shadow[name], device=current.device)
            if saved.shape != current.shape or saved.dtype != current.dtype:
                raise ValueError(f"EMA checkpoint tensor differs for {name}")
            if not bool(torch.isfinite(saved).all()):
                raise ValueError(f"EMA checkpoint contains NaN/Inf for {name}")
            current.copy_(saved)

    @contextmanager
    def average_parameters(self, model: nn.Module):
        floating = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name in self.shadow
        }
        self.copy_to(model)
        try:
            yield model
        finally:
            current = model.state_dict()
            for name, value in floating.items():
                current[name].copy_(value)


def stratified_timesteps(
    *,
    num_timesteps: int,
    batch_size: int,
    seed: int,
    update_index: int,
    device: torch.device,
) -> Tensor:
    """Sample one jittered timestep per equal-width stratum, then permute."""

    if num_timesteps < 2 or batch_size < 1 or update_index < 0:
        raise ValueError("invalid stratified timestep request")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 1_000_003 * int(update_index) + 17)
    jitter = torch.rand(batch_size, generator=generator, dtype=torch.float64)
    values = torch.floor(
        (torch.arange(batch_size, dtype=torch.float64) + jitter)
        * float(num_timesteps)
        / float(batch_size)
    ).long()
    order = torch.randperm(batch_size, generator=generator)
    return values[order].clamp_max(num_timesteps - 1).to(device)


@dataclass(frozen=True)
class ArtifactTrainStepResult:
    update_index: int
    loss: float
    gradient_norm: float
    optimizer_step_succeeded: bool
    timestep_minimum: int | None
    timestep_maximum: int | None
    timestep_mean: float | None
    metrics: dict[str, float]


def _noise_for_update(
    target: Tensor,
    *,
    seed: int,
    update_index: int,
) -> Tensor:
    generator = torch.Generator(device=target.device)
    generator.manual_seed(int(seed) + 2_000_003 * int(update_index) + 29)
    return torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    )


def artifact_train_step(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    batch: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    optimizer: Optimizer,
    scaler: Any,
    ema: CheckpointableEMA,
    update_index: int,
    training_seed: int,
    gradient_clip_norm: float,
    mixed_precision: bool,
) -> ArtifactTrainStepResult:
    """Execute one truthful optimizer update using the common tensor contract."""

    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    device = batch.observed.device
    if mixed_precision and device.type != "cuda":
        raise ValueError("mixed precision is enabled only for scheduled CUDA")
    expected_type = (
        DeterministicArtifactEstimator
        if model_kind == "deterministic"
        else ArtifactLatentDiffusion
        if model_kind == "diffusion"
        else None
    )
    if expected_type is None or not isinstance(model, expected_type):
        raise TypeError("model_kind does not match the supplied artifact model")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    timesteps: Tensor | None = None
    noise: Tensor | None = None
    if model_kind == "diffusion":
        timesteps = stratified_timesteps(
            num_timesteps=model.num_timesteps,
            batch_size=batch.batch_size,
            seed=training_seed,
            update_index=update_index,
            device=device,
        )
        noise = _noise_for_update(
            batch.target_standardized_latent,
            seed=training_seed,
            update_index=update_index,
        )
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=mixed_precision,
    ):
        if model_kind == "deterministic":
            loss, _ = model.latent_training_loss(
                batch.observed,
                batch.target_standardized_latent,
                **batch.model_kwargs(),
            )
            raw_metrics: Mapping[str, Tensor] = {}
        else:
            assert timesteps is not None and noise is not None
            loss, raw_metrics = model.training_loss(
                batch.target_standardized_latent,
                observed=batch.observed,
                timestep=timesteps,
                noise=noise,
                **batch.model_kwargs(),
            )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("artifact training loss is NaN/Inf")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(gradient_clip_norm),
        error_if_nonfinite=True,
    )
    succeeded = scaler_optimizer_step_succeeded(scaler, optimizer)
    if succeeded:
        ema.update(model)
    metrics = {
        str(name): float(value.detach().cpu()) for name, value in raw_metrics.items()
    }
    return ArtifactTrainStepResult(
        update_index=int(update_index),
        loss=float(loss.detach().cpu()),
        gradient_norm=float(gradient_norm.detach().cpu()),
        optimizer_step_succeeded=bool(succeeded),
        timestep_minimum=(None if timesteps is None else int(timesteps.min().cpu())),
        timestep_maximum=(None if timesteps is None else int(timesteps.max().cpu())),
        timestep_mean=(
            None if timesteps is None else float(timesteps.float().mean().cpu())
        ),
        metrics=metrics,
    )


@dataclass(frozen=True)
class ArtifactUpdateRun:
    start_step: int
    completed_step: int
    optimizer_step_attempts: int
    skipped_optimizer_steps: int
    loss_curve: tuple[dict[str, Any], ...]


def train_artifact_updates(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    source_batch: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    schedule: SharedMinibatchSchedule,
    optimizer: Optimizer,
    scaler: Any,
    ema: CheckpointableEMA,
    device: torch.device,
    start_step: int,
    stop_step: int,
    training_seed: int,
    gradient_clip_norm: float,
    mixed_precision: bool,
    maximum_skipped_optimizer_steps: int = 8,
) -> ArtifactUpdateRun:
    """Run a resumable half-open update interval on the shared schedule."""

    if schedule.sample_count != source_batch.batch_size:
        raise ValueError("shared schedule sample count differs from tensor batch")
    if not 0 <= int(start_step) <= int(stop_step) <= schedule.updates:
        raise ValueError("requested update interval is outside the shared schedule")
    if int(maximum_skipped_optimizer_steps) < 0:
        raise ValueError("maximum skipped optimizer steps cannot be negative")
    attempts = 0
    skipped = 0
    curve: list[dict[str, Any]] = []
    step = int(start_step)
    while step < int(stop_step):
        minibatch = source_batch.select(schedule.at(step)).to(device)
        result = artifact_train_step(
            model,
            minibatch,
            model_kind=model_kind,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            update_index=step,
            training_seed=training_seed,
            gradient_clip_norm=gradient_clip_norm,
            mixed_precision=mixed_precision,
        )
        attempts += 1
        if not result.optimizer_step_succeeded:
            skipped += 1
            if skipped > int(maximum_skipped_optimizer_steps):
                raise FloatingPointError("AMP overflow exceeded the frozen skip budget")
            continue
        step += 1
        curve.append(
            {
                "step": step,
                "loss": result.loss,
                "gradient_norm": result.gradient_norm,
                "timestep_minimum": result.timestep_minimum,
                "timestep_maximum": result.timestep_maximum,
                "timestep_mean": result.timestep_mean,
                **result.metrics,
            }
        )
    return ArtifactUpdateRun(
        start_step=int(start_step),
        completed_step=step,
        optimizer_step_attempts=attempts,
        skipped_optimizer_steps=skipped,
        loss_curve=tuple(curve),
    )


@dataclass(frozen=True)
class ArtifactResumeState:
    epoch: int
    step: int
    history: tuple[dict[str, Any], ...]
    extra: dict[str, Any]


def save_artifact_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Any,
    ema: CheckpointableEMA,
    epoch: int,
    step: int,
    contract: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model/optimizer/scaler/RNG and the EMA shadow."""

    checkpoint_extra = dict(extra or {})
    checkpoint_extra["ema_state"] = ema.state_dict()
    checkpoint_extra["loss_curve"] = [dict(row) for row in history]
    return save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=int(epoch),
        step=int(step),
        config=contract,
        normalizer=None,
        extra=checkpoint_extra,
    )


def resume_artifact_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Any,
    ema: CheckpointableEMA,
    expected_contract: Mapping[str, Any],
    map_location: str | torch.device,
) -> ArtifactResumeState:
    """Restore the exact next update, including the EMA shadow and loss curve."""

    state = resume_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        expected_config=expected_contract,
        map_location=map_location,
    )
    ema_state = state.extra.get("ema_state")
    if not isinstance(ema_state, Mapping):
        raise ValueError("artifact checkpoint is missing its EMA shadow")
    ema.load_state_dict(ema_state)
    history = state.extra.get("loss_curve", ())
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("artifact checkpoint loss curve is invalid")
    remainder = {
        key: value
        for key, value in state.extra.items()
        if key not in {"ema_state", "loss_curve"}
    }
    return ArtifactResumeState(
        epoch=state.epoch,
        step=state.step,
        history=tuple(dict(row) for row in history),
        extra=remainder,
    )


def load_artifact_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load without mutation while requiring the artifact EMA/history fields."""

    payload = load_training_checkpoint(path, map_location=map_location)
    extra = payload.get("extra")
    if not isinstance(extra, Mapping) or not isinstance(
        extra.get("ema_state"), Mapping
    ):
        raise ValueError("artifact checkpoint is missing its EMA shadow")
    history = extra.get("loss_curve")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ValueError("artifact checkpoint loss curve is invalid")
    return payload


def _time_mask(batch: SubjectArtifactTensorBatch) -> Tensor:
    mask = batch.valid_time_mask
    if mask.ndim == 2:
        mask = mask[:, None, :]
    return mask.bool().to(dtype=batch.observed.dtype)


@torch.no_grad()
def _predicted_x0(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    batch: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    timestep: int,
    seed: int,
) -> Tensor:
    mask = _time_mask(batch)
    if model_kind == "deterministic":
        return model(
            batch.observed,
            **batch.model_kwargs(),
        ) * mask
    if not isinstance(model, ArtifactLatentDiffusion):
        raise TypeError("diffusion x0 validation requires ArtifactLatentDiffusion")
    if not 0 <= int(timestep) < model.num_timesteps:
        raise ValueError("V1 timestep is outside the diffusion schedule")
    timesteps = torch.full(
        (batch.batch_size,),
        int(timestep),
        device=batch.observed.device,
        dtype=torch.long,
    )
    target = batch.target_standardized_latent * mask
    noise = _noise_for_update(target, seed=seed, update_index=int(timestep) + 1)
    noise = noise * mask
    x_t = model.q_sample(target, timesteps, noise) * mask
    predicted_v = model.predict_v(
        x_t,
        timesteps,
        observed=batch.observed,
        **batch.model_kwargs(),
    )
    predicted_x0, _ = model.x0_and_epsilon_from_v(
        x_t,
        predicted_v,
        timesteps,
    )
    return predicted_x0 * mask


@torch.no_grad()
def _fixed_objective(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    batch: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    timesteps: Sequence[int],
    seed: int,
) -> float:
    if model_kind == "deterministic":
        loss, _ = model.latent_training_loss(
            batch.observed,
            batch.target_standardized_latent,
            **batch.model_kwargs(),
        )
        return float(loss.detach().cpu())
    if not isinstance(model, ArtifactLatentDiffusion):
        raise TypeError("diffusion objective requires ArtifactLatentDiffusion")
    losses: list[Tensor] = []
    for timestep in timesteps:
        values = torch.full(
            (batch.batch_size,),
            int(timestep),
            device=batch.observed.device,
            dtype=torch.long,
        )
        noise = _noise_for_update(
            batch.target_standardized_latent,
            seed=seed,
            update_index=int(timestep) + 1,
        )
        loss, _ = model.training_loss(
            batch.target_standardized_latent,
            observed=batch.observed,
            timestep=values,
            noise=noise,
            **batch.model_kwargs(),
        )
        losses.append(loss)
    return float(torch.stack(losses).mean().detach().cpu())


def _masked_rmse(predicted: Tensor, target: Tensor, mask: Tensor) -> float:
    weight = mask.expand_as(predicted)
    denominator = weight.sum().clamp_min(1.0)
    value = torch.sqrt(((predicted - target).square() * weight).sum() / denominator)
    return float(value.detach().cpu())


def _identity_relative_change(
    predicted_standardized_latent: Tensor,
    identity_batch: SubjectArtifactTensorBatch,
    *,
    latent_mean: Tensor,
    latent_standard_deviation: Tensor,
) -> float:
    mask = _time_mask(identity_batch)
    output_mask = mask * identity_batch.channel_mask[:, :, None].to(mask.dtype)
    correction = canonical_artifact_delta(
        predicted_standardized_latent,
        normalized_transfer=identity_batch.normalized_transfer,
        latent_mean=latent_mean,
        latent_standard_deviation=latent_standard_deviation,
        output_mask=output_mask,
    )
    observed = identity_batch.observed * output_mask
    numerator = torch.linalg.vector_norm(correction.flatten(start_dim=1), dim=1)
    denominator = torch.linalg.vector_norm(observed.flatten(start_dim=1), dim=1)
    ratio = numerator / denominator.clamp_min(torch.finfo(observed.dtype).eps)
    return float(ratio.mean().detach().cpu())


@dataclass(frozen=True)
class V1FixedBatchResult:
    model_id: str
    target_id: str
    initial_loss: float
    final_loss: float
    relative_loss_reduction: float
    x0_rmse_by_timestep: dict[int, float]
    zero_artifact_relative_change_by_timestep: dict[int, float]
    updates_completed: int
    loss_curve: tuple[dict[str, float | int], ...]

    def validity_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": self.model_id,
                "target_id": self.target_id,
                "timestep": int(timestep),
                "initial_loss": self.initial_loss,
                "final_loss": self.final_loss,
                "standardized_latent_rmse": float(rmse),
                "zero_artifact_relative_observation_change": float(
                    self.zero_artifact_relative_change_by_timestep[timestep]
                ),
            }
            for timestep, rmse in sorted(self.x0_rmse_by_timestep.items())
        ]


def run_v1_fixed_batch_overfit(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    fit_batch: SubjectArtifactTensorBatch,
    *,
    identity_batch: SubjectArtifactTensorBatch,
    latent_mean: Tensor,
    latent_standard_deviation: Tensor,
    model_kind: ModelKind,
    model_id: str,
    target_id: str,
    optimizer: Optimizer,
    scaler: Any,
    ema: CheckpointableEMA,
    updates: int,
    training_seed: int,
    validation_timesteps: Sequence[int],
    gradient_clip_norm: float,
    mixed_precision: bool,
    check_interval_updates: int,
) -> V1FixedBatchResult:
    """Overfit one fixed real batch; identity data must be supplied explicitly."""

    if updates < 1 or check_interval_updates < 1:
        raise ValueError("V1 update and check intervals must be positive")
    timesteps = tuple(int(value) for value in validation_timesteps)
    if not timesteps or len(set(timesteps)) != len(timesteps):
        raise ValueError("V1 validation timesteps must be nonempty and unique")
    if not model_id or not target_id:
        raise ValueError("V1 model_id and target_id are required")
    identity_mask = _time_mask(identity_batch)
    identity_target_delta = canonical_artifact_delta(
        identity_batch.target_standardized_latent,
        normalized_transfer=identity_batch.normalized_transfer,
        latent_mean=latent_mean,
        latent_standard_deviation=latent_standard_deviation,
        output_mask=(
            identity_mask
            * identity_batch.channel_mask[:, :, None].to(identity_mask.dtype)
        ),
    )
    if bool(identity_target_delta.abs().max() > 1.0e-7):
        raise ValueError("identity_batch must map to physical-zero correction")
    model.eval()
    initial_loss = _fixed_objective(
        model,
        fit_batch,
        model_kind=model_kind,
        timesteps=timesteps,
        seed=training_seed,
    )
    curve: list[dict[str, float | int]] = [
        {"update": 0, "loss": initial_loss, "gradient_norm": 0.0}
    ]
    successful = 0
    attempts = 0
    while successful < int(updates):
        result = artifact_train_step(
            model,
            fit_batch,
            model_kind=model_kind,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            update_index=attempts,
            training_seed=training_seed,
            gradient_clip_norm=gradient_clip_norm,
            mixed_precision=mixed_precision,
        )
        attempts += 1
        if not result.optimizer_step_succeeded:
            if attempts > int(updates) + 8:
                raise FloatingPointError("V1 AMP repeatedly skipped optimizer updates")
            continue
        successful += 1
        if successful % int(check_interval_updates) == 0 or successful == int(updates):
            curve.append(
                {
                    "update": successful,
                    "loss": result.loss,
                    "gradient_norm": result.gradient_norm,
                }
            )
    with ema.average_parameters(model):
        model.eval()
        final_loss = _fixed_objective(
            model,
            fit_batch,
            model_kind=model_kind,
            timesteps=timesteps,
            seed=training_seed,
        )
        fit_mask = _time_mask(fit_batch)
        target = fit_batch.target_standardized_latent * fit_mask
        x0_rmse: dict[int, float] = {}
        identity_change: dict[int, float] = {}
        for timestep in timesteps:
            predicted = _predicted_x0(
                model,
                fit_batch,
                model_kind=model_kind,
                timestep=timestep,
                seed=training_seed,
            )
            x0_rmse[timestep] = _masked_rmse(predicted, target, fit_mask)
            identity_prediction = _predicted_x0(
                model,
                identity_batch,
                model_kind=model_kind,
                timestep=timestep,
                seed=training_seed + 991,
            )
            identity_change[timestep] = _identity_relative_change(
                identity_prediction,
                identity_batch,
                latent_mean=latent_mean,
                latent_standard_deviation=latent_standard_deviation,
            )
    if not math.isfinite(initial_loss) or initial_loss <= 0.0:
        raise FloatingPointError("V1 initial loss must be finite and positive")
    if not math.isfinite(final_loss) or final_loss < 0.0:
        raise FloatingPointError("V1 final loss must be finite and nonnegative")
    return V1FixedBatchResult(
        model_id=str(model_id),
        target_id=str(target_id),
        initial_loss=initial_loss,
        final_loss=final_loss,
        relative_loss_reduction=(initial_loss - final_loss) / initial_loss,
        x0_rmse_by_timestep=x0_rmse,
        zero_artifact_relative_change_by_timestep=identity_change,
        updates_completed=successful,
        loss_curve=tuple(curve),
    )


__all__ = [
    "ArtifactResumeState",
    "ArtifactTrainStepResult",
    "ArtifactTrainingBudget",
    "ArtifactUpdateRun",
    "CheckpointableEMA",
    "DevelopmentScore",
    "SharedMinibatchSchedule",
    "SubjectArtifactTensorBatch",
    "V1FixedBatchResult",
    "artifact_train_step",
    "build_shared_minibatch_schedule",
    "development_candidate_is_better",
    "development_converged",
    "load_artifact_training_checkpoint",
    "resume_artifact_training_checkpoint",
    "run_v1_fixed_batch_overfit",
    "save_artifact_training_checkpoint",
    "stratified_timesteps",
    "train_artifact_updates",
]
