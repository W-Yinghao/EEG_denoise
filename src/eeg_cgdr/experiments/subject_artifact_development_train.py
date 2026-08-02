"""J3 training for the subject-calibrated artifact-latent development round.

One Slurm array task trains one model for one frozen SGEYESUB exact-cell fold
and one frozen seed.  The 150-task map is deliberately small and explicit:
25 folds x 3 seeds x (deterministic, diffusion).  Both model arms consume the
same block-1 weak targets and the same seeded minibatch schedule.  The reused
fold loader seals block-2 query signals inside held-out records, but this
module never accesses those records or admits them to checkpoint selection.

The outer-training recordings are split once more, by ``recording_key`` before
window selection, into gradient-training and development-validation stems.
The validation split is about 20 percent (at least one stem).  Its loss is used
only for the frozen latent-MSE-first checkpoint rule.  The equal-update
checkpoint is always written at 8,000 updates; training can then continue to
12,000 updates, subject to the frozen 1,500-update patience rule.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from eeg_cgdr.experiments.subject_artifact_data import (
    ArtifactLatentTrainingArrays,
    PreparedSubjectArtifactFold,
    SubjectArtifactModelDimensions,
    prepare_subject_artifact_fold,
)
from eeg_cgdr.experiments.subject_artifact_training import (
    ArtifactTrainingBudget,
    CheckpointableEMA,
    DevelopmentScore,
    SubjectArtifactTensorBatch,
    build_shared_minibatch_schedule,
    development_candidate_is_better,
    development_converged,
    resume_artifact_training_checkpoint,
    save_artifact_training_checkpoint,
    train_artifact_updates,
)
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)


ModelKind = Literal["deterministic", "diffusion"]
_MODEL_KINDS: tuple[ModelKind, ModelKind] = ("deterministic", "diffusion")
_FOLD_COUNT = 25
_SEED_COUNT = 3
_TASK_COUNT = _FOLD_COUNT * _SEED_COUNT * len(_MODEL_KINDS)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing configuration mapping: {key}")
    return result


def _frozen_seeds(config: Mapping[str, Any]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in _mapping(config, "training").get("seeds", ()))
    if len(values) != 3 or len(set(values)) != 3:
        raise ValueError("J3 requires exactly three unique frozen training seeds")
    return values  # type: ignore[return-value]


def _validate_training_protocol(config: Mapping[str, Any]) -> None:
    training = _mapping(config, "training")
    inner = _mapping(training, "inner_validation")
    if (
        inner.get("split_unit") != "recording_key"
        or inner.get("rule")
        != "lexicographically_last_ceil_20_percent_minimum_one"
        or float(inner.get("fraction", float("nan"))) != 0.20
        or int(inner.get("minimum_stems", 0)) != 1
        or inner.get("heldout_block2_or_query_access") != "forbidden"
    ):
        raise ValueError("inner recording-validation policy differs from frozen J3")
    augmentation = _mapping(training, "context_augmentation")
    expected = (
        "matching_transfer_with_support_rho",
        "population_transfer_with_same_support_rho",
        "population_transfer_with_rho_zero_POP_endpoint",
    )
    if (
        tuple(augmentation.get("entries", ())) != expected
        or augmentation.get("reconstruction_source")
        != "outer_training_weak_target_plus_real_support_EOG_latent"
        or augmentation.get("query_information_used") is not False
    ):
        raise ValueError("context augmentation differs from frozen 1:1:1 J3 policy")
    if int(training["checkpoint_interval_updates"]) != int(
        training["validation_interval_updates"]
    ):
        raise ValueError(
            "the frozen J3 implementation requires checkpoint and validation intervals to match"
        )


@dataclass(frozen=True)
class SubjectArtifactTrainingTask:
    """Auditable row in the fixed 150-element J3 array."""

    task_index: int
    unified_fold_index: int
    seed_index: int
    seed: int
    model_kind: ModelKind

    @property
    def task_id(self) -> str:
        return (
            f"fold_{self.unified_fold_index:02d}__seed_{self.seed}__"
            f"{self.model_kind}"
        )


def subject_artifact_training_task(
    config: Mapping[str, Any], task_index: int
) -> SubjectArtifactTrainingTask:
    """Map ``0..149`` as fold-major, then seed, then paired model arm."""

    if isinstance(task_index, bool) or not 0 <= int(task_index) < _TASK_COUNT:
        raise ValueError(f"J3 task_index must lie in [0,{_TASK_COUNT - 1}]")
    index = int(task_index)
    per_fold = _SEED_COUNT * len(_MODEL_KINDS)
    fold = index // per_fold
    within_fold = index % per_fold
    seed_index = within_fold // len(_MODEL_KINDS)
    model_kind = _MODEL_KINDS[within_fold % len(_MODEL_KINDS)]
    return SubjectArtifactTrainingTask(
        task_index=index,
        unified_fold_index=fold,
        seed_index=seed_index,
        seed=_frozen_seeds(config)[seed_index],
        model_kind=model_kind,
    )


def subject_artifact_training_task_table(
    config: Mapping[str, Any],
) -> tuple[SubjectArtifactTrainingTask, ...]:
    """Return the complete map without opening any EEG data."""

    return tuple(subject_artifact_training_task(config, value) for value in range(_TASK_COUNT))


@dataclass(frozen=True)
class InnerRecordingSplit:
    """Recording-disjoint split made before selecting training windows."""

    training_recording_keys: tuple[str, ...]
    validation_recording_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        train = tuple(self.training_recording_keys)
        validation = tuple(self.validation_recording_keys)
        if not train or not validation:
            raise ValueError("inner training and validation must both contain a stem")
        if len(train) != len(set(train)) or len(validation) != len(set(validation)):
            raise ValueError("inner split recording keys must be unique")
        if set(train) & set(validation):
            raise ValueError("inner training and validation recording keys overlap")

    @property
    def total_recordings(self) -> int:
        return len(self.training_recording_keys) + len(self.validation_recording_keys)

    @property
    def validation_fraction(self) -> float:
        return len(self.validation_recording_keys) / self.total_recordings


def deterministic_inner_recording_split(
    recording_keys: Sequence[str], *, validation_fraction: float = 0.20
) -> InnerRecordingSplit:
    """Hold out the frozen ceil-20% suffix of lexicographically sorted stems."""

    keys = tuple(sorted(str(value) for value in recording_keys))
    if len(keys) != len(set(keys)):
        raise ValueError("recording_keys must be unique before the inner split")
    if len(keys) < 2:
        raise ValueError("at least two outer-training stems are required")
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0,1)")
    count = max(1, int(math.ceil(len(keys) * fraction)))
    count = min(count, len(keys) - 1)
    return InnerRecordingSplit(
        training_recording_keys=keys[:-count],
        validation_recording_keys=keys[-count:],
    )


def recording_indices(
    recording_keys: Sequence[str], selected_recording_keys: Sequence[str]
) -> np.ndarray:
    """Select every window belonging to a set of stems, never individual windows."""

    selected = frozenset(str(value) for value in selected_recording_keys)
    if not selected:
        raise ValueError("selected_recording_keys cannot be empty")
    indices = np.asarray(
        [index for index, key in enumerate(recording_keys) if str(key) in selected],
        dtype=np.int64,
    )
    represented = {str(recording_keys[index]) for index in indices.tolist()}
    if represented != set(selected):
        missing = sorted(set(selected) - represented)
        raise ValueError(f"inner split stems contain no weak-target windows: {missing}")
    if indices.size < 1:
        raise ValueError("inner split selected no windows")
    return indices


@dataclass(frozen=True)
class ArtifactLatentNormalization:
    """Artifact-latent statistics frozen from outer-training block 1."""

    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.standard_deviation, dtype=np.float64)
        if mean.ndim != 1 or mean.shape not in ((2,), (3,)) or scale.shape != mean.shape:
            raise ValueError("inner latent normalization must describe 2/3 coordinates")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("inner latent normalization is non-finite or degenerate")
        mean = np.array(mean, copy=True)
        scale = np.array(scale, copy=True)
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", scale)


@dataclass(frozen=True)
class AugmentedArtifactTraining:
    """The fixed 1:1:1 matching/population/population-rho0 source batch."""

    arrays: ArtifactLatentTrainingArrays
    context_roles: tuple[str, ...]
    latent_normalization: ArtifactLatentNormalization

    def __post_init__(self) -> None:
        count = self.arrays.observed.shape[0]
        if len(self.context_roles) != count:
            raise ValueError("augmented context-role count differs from arrays")
        allowed = {
            "matching_subject_rho",
            "population_subject_rho",
            "population_rho_zero_endpoint",
        }
        if set(self.context_roles) != allowed:
            raise ValueError("augmented source batch must contain all three contexts")

    def role_counts(self, indices: np.ndarray | None = None) -> dict[str, int]:
        values = (
            range(len(self.context_roles))
            if indices is None
            else np.asarray(indices, dtype=np.int64).tolist()
        )
        return {
            role: sum(self.context_roles[index] == role for index in values)
            for role in (
                "matching_subject_rho",
                "population_subject_rho",
                "population_rho_zero_endpoint",
            )
        }


def build_augmented_artifact_training(
    prepared: PreparedSubjectArtifactFold,
    config: Mapping[str, Any],
) -> AugmentedArtifactTraining:
    """Derive POP examples from support-only weak pairs, never block-2 query.

    For each matching example, ``Z_s`` is the transfer-scaled EOG latent.
    ``E_std = Z_s / scale_s`` is remapped through the fold population transfer
    to obtain ``Z_0``.  The weak target is recovered as ``Y_s-C_s_norm Z_s``
    and paired with ``Y_0=weak_target+C_0_norm Z_0``.  The population example
    is duplicated once at the original support rho and once at rho=0, so both
    formal POP endpoints are trained without equating POP with raw EEG.
    """

    _validate_training_protocol(config)
    source = prepared.training
    count, channels, length = source.observed.shape
    eog = source.standardized_artifact_latent.shape[1]
    normalizer = ArtifactLatentNormalization(
        mean=prepared.latent_normalizer.mean,
        standard_deviation=prepared.latent_normalizer.standard_deviation,
    )
    time_mask = np.asarray(source.valid_time_mask, dtype=np.float64)[:, None, :]
    z_subject = prepared.latent_normalizer.inverse_transform(
        np.asarray(source.standardized_artifact_latent, dtype=np.float64)
    ) * time_mask
    subject_scale = np.asarray(source.transfer_scale, dtype=np.float64)
    if subject_scale.shape != (count, eog) or np.any(subject_scale <= 0):
        raise ValueError("subject transfer scale is invalid")
    eog_standardized = z_subject / subject_scale[:, :, None]
    population = prepared.population_context
    if population.normalized_transfer.shape != (channels, eog):
        raise ValueError("population transfer differs from the exact compatibility cell")
    population_scale = np.asarray(population.transfer_scale, dtype=np.float64)
    z_population = eog_standardized * population_scale[None, :, None]
    subject_artifact = np.einsum(
        "nce,net->nct",
        np.asarray(source.normalized_transfer, dtype=np.float64),
        z_subject,
    )
    weak_target = np.asarray(source.observed, dtype=np.float64) - subject_artifact
    population_artifact = np.einsum(
        "ce,net->nct",
        np.asarray(population.normalized_transfer, dtype=np.float64),
        z_population,
    )
    population_observed = weak_target + population_artifact
    z_population_standardized = prepared.latent_normalizer.transform(
        z_population
    ) * time_mask
    z_subject_standardized = (
        np.asarray(source.standardized_artifact_latent, dtype=np.float64) * time_mask
    )

    def repeat_population(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value, (count, *value.shape)).copy()

    population_full = repeat_population(np.asarray(population.full_transfer))
    population_normalized = repeat_population(
        np.asarray(population.normalized_transfer)
    )
    population_scales = np.broadcast_to(population_scale, (count, eog)).copy()
    population_singular = np.broadcast_to(
        np.asarray(population.singular_values), (count, eog)
    ).copy()
    population_rank = np.full(count, int(population.rank), dtype=np.int64)
    population_duration = np.full(
        count, float(population.calibration_duration_seconds), dtype=np.float32
    )
    arrays = ArtifactLatentTrainingArrays(
        observed=np.concatenate(
            (source.observed, population_observed, population_observed), axis=0
        ).astype(np.float32),
        standardized_artifact_latent=np.concatenate(
            (
                z_subject_standardized,
                z_population_standardized,
                z_population_standardized,
            ),
            axis=0,
        ).astype(np.float32),
        valid_time_mask=np.concatenate(
            (source.valid_time_mask, source.valid_time_mask, source.valid_time_mask),
            axis=0,
        ),
        full_transfer=np.concatenate(
            (source.full_transfer, population_full, population_full), axis=0
        ).astype(np.float32),
        normalized_transfer=np.concatenate(
            (source.normalized_transfer, population_normalized, population_normalized),
            axis=0,
        ).astype(np.float32),
        transfer_scale=np.concatenate(
            (source.transfer_scale, population_scales, population_scales), axis=0
        ).astype(np.float32),
        singular_values=np.concatenate(
            (source.singular_values, population_singular, population_singular), axis=0
        ).astype(np.float32),
        rank=np.concatenate((source.rank, population_rank, population_rank), axis=0),
        rho=np.concatenate(
            (source.rho, source.rho, np.zeros(count, dtype=np.float32)), axis=0
        ).astype(np.float32),
        calibration_duration_seconds=np.concatenate(
            (
                source.calibration_duration_seconds,
                source.calibration_duration_seconds,
                population_duration,
            ),
            axis=0,
        ).astype(np.float32),
        channel_mask=np.concatenate(
            (source.channel_mask, source.channel_mask, source.channel_mask), axis=0
        ),
        recording_keys=(
            tuple(source.recording_keys)
            + tuple(source.recording_keys)
            + tuple(source.recording_keys)
        ),
        target_origins=(
            tuple(source.target_origins)
            + tuple(source.target_origins)
            + tuple(source.target_origins)
        ),
        artifact_origins=(
            tuple(source.artifact_origins)
            + tuple(source.artifact_origins)
            + tuple(source.artifact_origins)
        ),
    )
    roles = (
        ("matching_subject_rho",) * count
        + ("population_subject_rho",) * count
        + ("population_rho_zero_endpoint",) * count
    )
    augmented = AugmentedArtifactTraining(
        arrays=arrays,
        context_roles=roles,
        latent_normalization=normalizer,
    )
    counts = augmented.role_counts()
    if len(set(counts.values())) != 1:
        raise AssertionError("context augmentation must retain its fixed 1:1:1 ratio")
    return augmented


def tensor_batch_from_arrays(
    arrays: ArtifactLatentTrainingArrays,
    indices: np.ndarray,
    *,
    standardized_latent_override: np.ndarray | None = None,
) -> SubjectArtifactTensorBatch:
    """Convert one recording-selected subset to the shared CPU tensor contract."""

    index = np.asarray(indices, dtype=np.int64)
    if index.ndim != 1 or index.size < 1:
        raise ValueError("training tensor indices must be a nonempty vector")
    latent = (
        arrays.standardized_artifact_latent
        if standardized_latent_override is None
        else np.asarray(standardized_latent_override)
    )
    if latent.shape != arrays.standardized_artifact_latent.shape:
        raise ValueError("standardized latent override shape differs from training arrays")
    return SubjectArtifactTensorBatch(
        observed=torch.from_numpy(np.asarray(arrays.observed[index], dtype=np.float32)),
        target_standardized_latent=torch.from_numpy(
            np.asarray(latent[index], dtype=np.float32)
        ),
        full_transfer=torch.from_numpy(
            np.asarray(arrays.full_transfer[index], dtype=np.float32)
        ),
        normalized_transfer=torch.from_numpy(
            np.asarray(arrays.normalized_transfer[index], dtype=np.float32)
        ),
        transfer_scale=torch.from_numpy(
            np.asarray(arrays.transfer_scale[index], dtype=np.float32)
        ),
        singular_values=torch.from_numpy(
            np.asarray(arrays.singular_values[index], dtype=np.float32)
        ),
        rank=torch.from_numpy(np.asarray(arrays.rank[index], dtype=np.int64)),
        rho=torch.from_numpy(np.asarray(arrays.rho[index], dtype=np.float32)),
        calibration_duration_seconds=torch.from_numpy(
            np.asarray(arrays.calibration_duration_seconds[index], dtype=np.float32)
        ),
        channel_mask=torch.from_numpy(
            np.asarray(arrays.channel_mask[index], dtype=bool)
        ),
        valid_time_mask=torch.from_numpy(
            np.asarray(arrays.valid_time_mask[index], dtype=bool)
        ),
    )


def _model_configuration(
    config: Mapping[str, Any], dimensions: SubjectArtifactModelDimensions
) -> ArtifactLatentModelConfig:
    values = _mapping(config, "model")
    return ArtifactLatentModelConfig(
        eeg_channels=int(dimensions.eeg_channels),
        signal_length=int(dimensions.signal_length),
        latent_channels=int(dimensions.eog_coordinates),
        base_channels=int(values["base_channels"]),
        channel_mults=tuple(int(value) for value in values["channel_mults"]),  # type: ignore[arg-type]
        num_res_blocks=int(values["num_res_blocks"]),
        groupnorm_groups=int(values["groupnorm_groups"]),
        dropout=float(values["dropout"]),
        time_sinusoidal_dim=int(values["time_sinusoidal_dim"]),
        time_embed_dim=int(values["time_embed_dim"]),
        attention_length=int(values.get("attention_length", 64)),
        attention_heads=int(values["attention_heads"]),
    )


def _build_model(
    config: Mapping[str, Any],
    dimensions: SubjectArtifactModelDimensions,
    model_kind: ModelKind,
) -> DeterministicArtifactEstimator | ArtifactLatentDiffusion:
    model_config = _model_configuration(config, dimensions)
    if model_kind == "deterministic":
        return DeterministicArtifactEstimator(model_config)
    values = _mapping(config, "primary_diffusion")
    return ArtifactLatentDiffusion(
        model_config,
        ArtifactLatentDiffusionConfig(
            num_timesteps=int(values["timesteps"]),
            cosine_offset=float(values["cosine_offset"]),
            prediction_target=str(values["prediction_target"]),
            min_snr_gamma=float(values["min_snr_gamma"]),
            dynamic_threshold_quantile=float(values["dynamic_threshold_quantile"]),
            standardized_latent_absolute_clip=float(
                values["standardized_latent_absolute_clip"]
            ),
            posterior_samples=int(
                _mapping(config, "artifact_latent")["posterior_samples"]
            ),
        ),
    )


def training_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def build_training_contract(
    config: Mapping[str, Any],
    *,
    task: SubjectArtifactTrainingTask,
    fold_id: str,
    dimensions: SubjectArtifactModelDimensions,
    split: InnerRecordingSplit,
    latent_normalization: ArtifactLatentNormalization,
    implementation: str,
) -> dict[str, Any]:
    """Minimal exact-resume contract; intentionally no hashes or provenance DAG."""

    implementation_value = str(implementation).strip()
    if not implementation_value:
        raise ValueError("implementation git SHA/identifier cannot be empty")
    training = _mapping(config, "training")
    return {
        "protocol_id": str(config.get("protocol_id", "")),
        "implementation": implementation_value,
        "task_index": task.task_index,
        "unified_fold_index": task.unified_fold_index,
        "fold_id": str(fold_id),
        "training_seed": task.seed,
        "model_kind": task.model_kind,
        "endpoint": "training_run",
        "eeg_channels": dimensions.eeg_channels,
        "eog_coordinates": dimensions.eog_coordinates,
        "signal_length": dimensions.signal_length,
        "model_dimensions": {
            "eeg_channels": dimensions.eeg_channels,
            "eog_coordinates": dimensions.eog_coordinates,
            "signal_length": dimensions.signal_length,
        },
        "inner_training_recording_keys": list(split.training_recording_keys),
        "inner_validation_recording_keys": list(split.validation_recording_keys),
        "outer_training_latent_mean": latent_normalization.mean.tolist(),
        "outer_training_latent_standard_deviation": latent_normalization.standard_deviation.tolist(),
        "latent_normalization_fit_scope": "outer_training_block1_only",
        "context_augmentation_entries": list(
            _mapping(training, "context_augmentation")["entries"]
        ),
        "context_sampling_ratio": "1:1:1_fixed_over_each_source_record",
        "equal_compute_updates": int(training["equal_compute_updates"]),
        "maximum_updates": int(training["maximum_updates"]),
        "batch_size": int(training["batch_size"]),
        "validation_interval_updates": int(training["validation_interval_updates"]),
        "best_checkpoint_rule": str(training["best_checkpoint_rule"]),
        "checkpoint_interval_updates": int(training["checkpoint_interval_updates"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "mixed_precision": bool(training["mixed_precision"]),
        "convergence_patience_updates": int(
            training["convergence_patience_updates"]
        ),
        "convergence_minimum_relative_improvement": float(
            training["convergence_minimum_relative_improvement"]
        ),
        "model": dict(_mapping(config, "model")),
        "primary_diffusion": dict(_mapping(config, "primary_diffusion")),
        "artifact_latent": dict(_mapping(config, "artifact_latent")),
        "validation_timesteps": list(
            _mapping(_mapping(config, "validity"), "V1")["timesteps"]
        ),
        "shared_minibatch_schedule_seed": task.seed,
        "heldout_block2_used_for_checkpoint_selection": False,
        "query_eog_or_labels_used": False,
    }


@dataclass(frozen=True)
class ValidationMetrics:
    artifact_latent_mse: float
    mapped_artifact_mse: float
    x0_reconstruction_mse: float
    timestep_rows: tuple[dict[str, float | int | str], ...]

    def score(self, step: int) -> DevelopmentScore:
        return DevelopmentScore(
            step=int(step),
            artifact_latent_mse=float(self.artifact_latent_mse),
            x0_reconstruction_mse=float(self.x0_reconstruction_mse),
        )


def masked_validation_components(
    predicted_standardized_latent: Tensor,
    target_standardized_latent: Tensor,
    batch: SubjectArtifactTensorBatch,
    *,
    latent_mean: Tensor,
    latent_standard_deviation: Tensor,
) -> tuple[float, float, float, int, int]:
    """Return latent/mapped/x0 squared sums and their valid denominators."""

    predicted = torch.as_tensor(predicted_standardized_latent)
    target = torch.as_tensor(target_standardized_latent)
    if predicted.shape != target.shape or predicted.shape != batch.target_standardized_latent.shape:
        raise ValueError("validation prediction/target shape differs from its batch")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("validation prediction or target contains NaN/Inf")
    mask = batch.valid_time_mask
    if mask.ndim == 2:
        mask = mask[:, None, :]
    latent_mask = mask.bool().expand_as(predicted)
    latent_squared_sum = float(
        (predicted - target).square().masked_select(latent_mask).sum().detach().cpu()
    )
    latent_count = int(latent_mask.sum().detach().cpu())
    mean = torch.as_tensor(latent_mean, device=predicted.device, dtype=predicted.dtype)
    scale = torch.as_tensor(
        latent_standard_deviation, device=predicted.device, dtype=predicted.dtype
    )
    if mean.shape != (predicted.shape[1],) or scale.shape != mean.shape or bool((scale <= 0).any()):
        raise ValueError("validation latent normalization has an invalid shape/scale")
    time = mask.to(dtype=predicted.dtype)
    predicted_physical = (
        predicted * scale[None, :, None] + mean[None, :, None]
    ) * time
    target_physical = (
        target * scale[None, :, None] + mean[None, :, None]
    ) * time
    predicted_artifact = torch.einsum(
        "bce,bet->bct", batch.normalized_transfer, predicted_physical
    )
    target_artifact = torch.einsum(
        "bce,bet->bct", batch.normalized_transfer, target_physical
    )
    output_mask = (
        batch.channel_mask[:, :, None].bool() & mask.bool()
    ).expand_as(predicted_artifact)
    mapped_squared_sum = float(
        (predicted_artifact - target_artifact)
        .square()
        .masked_select(output_mask)
        .sum()
        .detach()
        .cpu()
    )
    mapped_count = int(output_mask.sum().detach().cpu())
    # Because X0 = Y - Delta, this is exactly the mapped artifact error while
    # retaining the observation-anchored definition in the reported contract.
    predicted_x0 = batch.observed - predicted_artifact
    target_x0 = batch.observed - target_artifact
    x0_squared_sum = float(
        (predicted_x0 - target_x0)
        .square()
        .masked_select(output_mask)
        .sum()
        .detach()
        .cpu()
    )
    return latent_squared_sum, mapped_squared_sum, x0_squared_sum, latent_count, mapped_count


def _fixed_validation_prediction(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    batch: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    timestep: int | None,
    seed: int,
    batch_ordinal: int,
) -> Tensor:
    if model_kind == "deterministic":
        return model(batch.observed, **batch.model_kwargs())
    if not isinstance(model, ArtifactLatentDiffusion) or timestep is None:
        raise TypeError("diffusion validation requires a timestep and diffusion model")
    values = torch.full(
        (batch.batch_size,), int(timestep), device=batch.observed.device, dtype=torch.long
    )
    generator = torch.Generator(device=batch.observed.device)
    generator.manual_seed(int(seed) + 7_000_001 * int(timestep) + int(batch_ordinal))
    mask = batch.valid_time_mask
    if mask.ndim == 2:
        mask = mask[:, None, :]
    mask = mask.to(dtype=batch.observed.dtype)
    target = batch.target_standardized_latent * mask
    noise = torch.randn(
        target.shape,
        device=target.device,
        dtype=target.dtype,
        generator=generator,
    ) * mask
    noisy = model.q_sample(target, values, noise) * mask
    predicted_v = model.predict_v(
        noisy,
        values,
        observed=batch.observed,
        **batch.model_kwargs(),
    )
    predicted_x0, _ = model.x0_and_epsilon_from_v(noisy, predicted_v, values)
    return predicted_x0 * mask


@torch.no_grad()
def evaluate_development_validation(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    validation_batch: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    device: torch.device,
    batch_size: int,
    seed: int,
    diffusion_timesteps: Sequence[int],
    latent_normalization: ArtifactLatentNormalization,
) -> ValidationMetrics:
    """Evaluate all held-out block-1 windows with fixed noise/timesteps."""

    if batch_size < 1:
        raise ValueError("validation batch_size must be positive")
    timesteps: tuple[int | None, ...]
    if model_kind == "deterministic":
        timesteps = (None,)
    else:
        values = tuple(int(value) for value in diffusion_timesteps)
        if not values or len(values) != len(set(values)):
            raise ValueError("diffusion validation timesteps must be unique and nonempty")
        timesteps = values
    model.eval()
    per_timestep: list[dict[str, float | int | str]] = []
    for timestep in timesteps:
        latent_sum = mapped_sum = x0_sum = 0.0
        latent_count = mapped_count = 0
        batch_ordinal = 0
        for start in range(0, validation_batch.batch_size, int(batch_size)):
            stop = min(start + int(batch_size), validation_batch.batch_size)
            indices = torch.arange(start, stop, dtype=torch.long)
            current = validation_batch.select(indices).to(device)
            prediction = _fixed_validation_prediction(
                model,
                current,
                model_kind=model_kind,
                timestep=timestep,
                seed=seed,
                batch_ordinal=batch_ordinal,
            )
            components = masked_validation_components(
                prediction,
                current.target_standardized_latent,
                current,
                latent_mean=torch.as_tensor(latent_normalization.mean),
                latent_standard_deviation=torch.as_tensor(
                    latent_normalization.standard_deviation
                ),
            )
            latent_sum += components[0]
            mapped_sum += components[1]
            x0_sum += components[2]
            latent_count += components[3]
            mapped_count += components[4]
            batch_ordinal += 1
        if min(latent_count, mapped_count) < 1:
            raise ValueError("development validation has no valid samples")
        per_timestep.append(
            {
                "timestep": "deterministic" if timestep is None else int(timestep),
                "artifact_latent_mse": latent_sum / latent_count,
                "mapped_artifact_mse": mapped_sum / mapped_count,
                "x0_reconstruction_mse": x0_sum / mapped_count,
            }
        )
    return ValidationMetrics(
        artifact_latent_mse=float(
            np.mean([float(value["artifact_latent_mse"]) for value in per_timestep])
        ),
        mapped_artifact_mse=float(
            np.mean([float(value["mapped_artifact_mse"]) for value in per_timestep])
        ),
        x0_reconstruction_mse=float(
            np.mean([float(value["x0_reconstruction_mse"]) for value in per_timestep])
        ),
        timestep_rows=tuple(per_timestep),
    )


def _budget(config: Mapping[str, Any]) -> ArtifactTrainingBudget:
    values = _mapping(config, "training")
    return ArtifactTrainingBudget(
        seeds=_frozen_seeds(config),
        equal_compute_updates=int(values["equal_compute_updates"]),
        maximum_updates=int(values["maximum_updates"]),
        batch_size=int(values["batch_size"]),
        validation_interval_updates=int(values["validation_interval_updates"]),
        convergence_patience_updates=int(values["convergence_patience_updates"]),
        convergence_minimum_relative_improvement=float(
            values["convergence_minimum_relative_improvement"]
        ),
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row}) or ["status"]
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _score_dict(value: DevelopmentScore | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "step": value.step,
        "artifact_latent_mse": value.artifact_latent_mse,
        "x0_reconstruction_mse": value.x0_reconstruction_mse,
    }


def _score_from_dict(value: object) -> DevelopmentScore | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("resumed best development score is invalid")
    return DevelopmentScore(
        step=int(value["step"]),
        artifact_latent_mse=float(value["artifact_latent_mse"]),
        x0_reconstruction_mse=float(value["x0_reconstruction_mse"]),
    )


def _checkpoint_extra(
    *,
    best: DevelopmentScore | None,
    validation_history: Sequence[Mapping[str, Any]],
    equal_checkpoint_saved: bool,
    training_complete: bool,
    stop_reason: str | None,
    wall_time_seconds: float,
    checkpoint_role: str,
) -> dict[str, Any]:
    return {
        "best_development_score": _score_dict(best),
        "validation_curve": [dict(row) for row in validation_history],
        "equal_checkpoint_saved": bool(equal_checkpoint_saved),
        "training_complete": bool(training_complete),
        "stop_reason": stop_reason,
        "accumulated_wall_time_seconds": float(wall_time_seconds),
        "checkpoint_role": str(checkpoint_role),
    }


def run_subject_artifact_training(
    config: Mapping[str, Any],
    run_dir: str | Path,
    task_index: int,
    implementation: str,
) -> Mapping[str, Any]:
    """Train or strictly resume one J3 array task on its stable task path."""

    started = time.monotonic()
    _validate_training_protocol(config)
    task = subject_artifact_training_task(config, task_index)
    prepared = prepare_subject_artifact_fold(config, task.unified_fold_index)
    augmented = build_augmented_artifact_training(prepared, config)
    represented = tuple(sorted(set(augmented.arrays.recording_keys)))
    if not set(represented).issubset(prepared.fold.training_recording_keys):
        raise AssertionError("weak-target training windows escaped outer-training stems")
    validation_fraction = float(
        _mapping(_mapping(config, "training"), "inner_validation")["fraction"]
    )
    split = deterministic_inner_recording_split(
        represented, validation_fraction=validation_fraction
    )
    train_indices = recording_indices(
        augmented.arrays.recording_keys, split.training_recording_keys
    )
    validation_indices = recording_indices(
        augmented.arrays.recording_keys, split.validation_recording_keys
    )
    if set(train_indices.tolist()) & set(validation_indices.tolist()):
        raise AssertionError("inner training and validation windows overlap")
    training_batch = tensor_batch_from_arrays(
        augmented.arrays,
        train_indices,
    )
    validation_batch = tensor_batch_from_arrays(
        augmented.arrays,
        validation_indices,
    )
    latent_normalization = augmented.latent_normalization
    contract = build_training_contract(
        config,
        task=task,
        fold_id=prepared.fold.fold_id,
        dimensions=prepared.model_dimensions,
        split=split,
        latent_normalization=latent_normalization,
        implementation=implementation,
    )
    best_contract = {**contract, "endpoint": "development_validation_best"}
    equal_contract = {**contract, "endpoint": "equal_update_8000"}
    last_contract = {**contract, "endpoint": "last_resume"}
    output = (
        Path(run_dir)
        / f"fold_{task.unified_fold_index:02d}"
        / f"seed_{task.seed}"
        / task.model_kind
    )
    output.mkdir(parents=True, exist_ok=True)
    task_map_path = output / "task_mapping.json"
    _atomic_json(task_map_path, contract)
    checkpoint_output = (
        Path(str(_mapping(config, "outputs")["checkpoint_root"]))
        / f"fold_{task.unified_fold_index:02d}"
        / f"seed_{task.seed}"
        / task.model_kind
    )
    checkpoint_output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "checkpoint_pointer.json",
        {
            "checkpoint_directory": str(checkpoint_output.resolve()),
            "equal": str((checkpoint_output / "equal.pt").resolve()),
            "best": str((checkpoint_output / "best.pt").resolve()),
            "last": str((checkpoint_output / "last.pt").resolve()),
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("J3 subject-artifact training requires a scheduled GPU")
    _seed_everything(task.seed)
    model = _build_model(config, prepared.model_dimensions, task.model_kind).to(device)
    parameters = training_parameter_counts(model)
    values = _mapping(config, "training")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
    )
    mixed_precision = bool(values["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision, init_scale=1024.0)
    ema = CheckpointableEMA(
        model, decay=float(_mapping(config, "primary_diffusion")["ema_decay"])
    )
    budget = _budget(config)
    schedule = build_shared_minibatch_schedule(
        sample_count=training_batch.batch_size,
        batch_size=budget.batch_size,
        updates=budget.maximum_updates,
        seed=task.seed,
    )

    equal_path = checkpoint_output / "equal.pt"
    best_path = checkpoint_output / "best.pt"
    last_path = checkpoint_output / "last.pt"
    result_path = output / "result_summary.json"
    stable_result_path = checkpoint_output / "result_summary.json"
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best: DevelopmentScore | None = None
    equal_saved = equal_path.exists()
    step = 0
    prior_wall = 0.0
    if last_path.exists():
        resumed = resume_artifact_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            expected_contract=last_contract,
            map_location=device,
        )
        step = resumed.step
        history = [dict(row) for row in resumed.history]
        validation_history = [
            dict(row) for row in resumed.extra.get("validation_curve", ())
        ]
        best = _score_from_dict(resumed.extra.get("best_development_score"))
        equal_saved = bool(resumed.extra.get("equal_checkpoint_saved", equal_saved))
        prior_wall = float(resumed.extra.get("accumulated_wall_time_seconds", 0.0))
        if resumed.extra.get("training_complete") is True:
            if not stable_result_path.is_file():
                raise RuntimeError(
                    "completed resume checkpoint is missing its stable result summary"
                )
            with stable_result_path.open("r", encoding="utf-8") as stream:
                completed = json.load(stream)
            if not isinstance(completed, Mapping):
                raise ValueError("completed J3 result summary is invalid")
            _atomic_json(result_path, completed)
            return dict(completed)

    validation_interval = budget.validation_interval_updates
    if budget.equal_compute_updates % validation_interval:
        raise ValueError("equal-update endpoint must align with validation interval")
    validation_timesteps = tuple(
        int(value)
        for value in _mapping(_mapping(config, "validity"), "V1")["timesteps"]
    )
    stop_reason = "maximum_updates"
    while step < budget.maximum_updates:
        next_validation = min(
            budget.maximum_updates,
            ((step // validation_interval) + 1) * validation_interval,
        )
        update = train_artifact_updates(
            model,
            training_batch,
            model_kind=task.model_kind,
            schedule=schedule,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
            start_step=step,
            stop_step=next_validation,
            training_seed=task.seed,
            gradient_clip_norm=float(values["gradient_clip_norm"]),
            mixed_precision=mixed_precision,
        )
        history.extend(dict(row) for row in update.loss_curve)
        step = update.completed_step
        with ema.average_parameters(model):
            metrics = evaluate_development_validation(
                model,
                validation_batch,
                model_kind=task.model_kind,
                device=device,
                batch_size=budget.batch_size,
                seed=task.seed,
                diffusion_timesteps=validation_timesteps,
                latent_normalization=latent_normalization,
            )
        score = metrics.score(step)
        for row in metrics.timestep_rows:
            validation_history.append({"step": step, **dict(row)})
        if development_candidate_is_better(
            score,
            best,
            minimum_relative_improvement=budget.convergence_minimum_relative_improvement,
        ):
            best = score
            save_artifact_training_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                epoch=0,
                step=step,
                contract=best_contract,
                history=history,
                extra=_checkpoint_extra(
                    best=best,
                    validation_history=validation_history,
                    equal_checkpoint_saved=equal_saved,
                    training_complete=False,
                    stop_reason=None,
                    wall_time_seconds=prior_wall + time.monotonic() - started,
                    checkpoint_role="development_validation_best",
                ),
            )
        if step == budget.equal_compute_updates:
            equal_saved = True
            save_artifact_training_checkpoint(
                equal_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                epoch=0,
                step=step,
                contract=equal_contract,
                history=history,
                extra=_checkpoint_extra(
                    best=best,
                    validation_history=validation_history,
                    equal_checkpoint_saved=True,
                    training_complete=False,
                    stop_reason=None,
                    wall_time_seconds=prior_wall + time.monotonic() - started,
                    checkpoint_role="equal_update",
                ),
            )
        if best is not None and development_converged(
            current_step=step, best_step=best.step, budget=budget
        ):
            stop_reason = "development_patience"
            break
        save_artifact_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            epoch=0,
            step=step,
            contract=last_contract,
            history=history,
            extra=_checkpoint_extra(
                best=best,
                validation_history=validation_history,
                equal_checkpoint_saved=equal_saved,
                training_complete=False,
                stop_reason=None,
                wall_time_seconds=prior_wall + time.monotonic() - started,
                checkpoint_role="last_resume",
            ),
        )

    if not equal_saved or not equal_path.exists() or best is None or not best_path.exists():
        raise AssertionError("J3 did not produce its frozen equal/best checkpoints")
    total_wall = prior_wall + time.monotonic() - started
    _atomic_csv(output / "train_loss_curve.csv", history)
    _atomic_csv(output / "timestep_stratified_validation.csv", validation_history)
    summary: dict[str, Any] = {
        "status": "success",
        "scientific_role": "development_training_not_confirmation",
        "task": {
            "task_index": task.task_index,
            "task_id": task.task_id,
            "unified_fold_index": task.unified_fold_index,
            "fold_id": prepared.fold.fold_id,
            "seed": task.seed,
            "model_kind": task.model_kind,
        },
        "split": {
            "gradient_training_recording_count": len(split.training_recording_keys),
            "development_validation_recording_count": len(
                split.validation_recording_keys
            ),
            "gradient_training_recording_keys": list(split.training_recording_keys),
            "development_validation_recording_keys": list(
                split.validation_recording_keys
            ),
            "gradient_training_window_count": training_batch.batch_size,
            "development_validation_window_count": validation_batch.batch_size,
            "heldout_block2_used_for_checkpoint_selection": False,
            "inner_validation_fit_scope": (
                "recording_disjoint_for_gradient_updates_but_outer_training_"
                "population_operator_and_normalization_are_shared"
            ),
            "gradient_training_context_entry_counts": augmented.role_counts(
                train_indices
            ),
            "development_validation_context_entry_counts": augmented.role_counts(
                validation_indices
            ),
            "context_sampling_ratio": "1:1:1_fixed",
        },
        "training": {
            "completed_updates": step,
            "equal_update_endpoint": budget.equal_compute_updates,
            "maximum_updates": budget.maximum_updates,
            "stop_reason": stop_reason,
            "best_step": best.step,
            "best_artifact_latent_mse": best.artifact_latent_mse,
            "best_mapped_x0_mse": best.x0_reconstruction_mse,
            "mixed_precision": mixed_precision,
            "parameters": parameters,
            "wall_time_seconds": total_wall,
            "checkpoint_selection_interpretation": (
                "objective_convergence_only_not_end_to_end_model_comparison"
            ),
        },
        "checkpoints": {
            "equal_update": str(equal_path.resolve()),
            "best_validation": str(best_path.resolve()),
            "last_resume": str(last_path.resolve()),
            "EMA_saved": True,
        },
        "curves": {
            "training": str((output / "train_loss_curve.csv").resolve()),
            "timestep_stratified_validation": str(
                (output / "timestep_stratified_validation.csv").resolve()
            ),
        },
        "implementation": str(implementation),
        "query_eog_labels_or_outcomes_used": False,
    }
    _atomic_json(result_path, summary)
    _atomic_json(stable_result_path, summary)
    save_artifact_training_checkpoint(
        last_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        epoch=0,
        step=step,
        contract=last_contract,
        history=history,
        extra=_checkpoint_extra(
            best=best,
            validation_history=validation_history,
            equal_checkpoint_saved=True,
            training_complete=True,
            stop_reason=stop_reason,
            wall_time_seconds=total_wall,
            checkpoint_role="last_resume",
        ),
    )
    return summary


__all__ = [
    "AugmentedArtifactTraining",
    "ArtifactLatentNormalization",
    "InnerRecordingSplit",
    "SubjectArtifactTrainingTask",
    "ValidationMetrics",
    "build_training_contract",
    "build_augmented_artifact_training",
    "deterministic_inner_recording_split",
    "evaluate_development_validation",
    "masked_validation_components",
    "recording_indices",
    "run_subject_artifact_training",
    "subject_artifact_training_task",
    "subject_artifact_training_task_table",
    "tensor_batch_from_arrays",
    "training_parameter_counts",
]
