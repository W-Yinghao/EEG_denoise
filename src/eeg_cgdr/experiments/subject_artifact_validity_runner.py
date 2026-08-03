"""GPU validity stage for subject-calibrated artifact-latent models.

The stage is deliberately development-only.  It loads one complete frozen
SGEYESUB development fold, keeps block-2 annotations sealed, overfits the same
three real weak-pair windows for V1, then trains both information-matched
estimators on the complete outer-training arrays.  All V0--V3 inputs are
derived here from returned tensors; callers cannot assert a validity pass.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW

from eeg_cgdr.experiments.subject_artifact_data import (
    ArtifactLatentTrainingArrays,
    OuterTrainingLatentNormalizer,
    PreparedSubjectArtifactFold,
    RuntimeArtifactContext,
    prepare_subject_artifact_fold,
)
from eeg_cgdr.experiments.subject_artifact_training import (
    CheckpointableEMA,
    SubjectArtifactTensorBatch,
    build_shared_minibatch_schedule,
    resume_artifact_training_checkpoint,
    run_v1_fixed_batch_overfit,
    save_artifact_training_checkpoint,
    train_artifact_updates,
)
from eeg_cgdr.experiments.subject_artifact_validity import (
    evaluate_v0,
    evaluate_v1,
    evaluate_v2,
    evaluate_v3,
)
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
    ArtifactPosterior,
    ArtifactTrajectoryStep,
    artifact_posterior_point_estimate,
)
from eeg_cgdr.models.artifact_latent_inference import (
    ArtifactInferenceContext,
    PopulationSubjectRestoration,
    canonical_artifact_delta,
    deterministic_population_subject_restore,
    diffusion_population_subject_restore,
)


Implementation = Literal[
    "primary_attempt_0",
    "primary_attempt_1",
    "primary_attempt_2",
    "residual_sdedit_backup",
]
ModelKind = Literal["deterministic", "diffusion"]
_CONTEXTS = ("population", "matching", "wrong", "shuffled")
_SDEDIT_START_TIMESTEP = 500
_SDEDIT_ANCHOR_RIDGE = 1.0e-3


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({str(key) for row in rows for key in row})
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for raw in rows:
                row = {
                    key: (
                        json.dumps(raw[key], sort_keys=True)
                        if isinstance(raw.get(key), (dict, list, tuple))
                        else raw.get(key, "")
                    )
                    for key in keys
                }
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("J2 validity requires a scheduled CUDA allocation")
    return torch.device("cuda", 0)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _tensor_batch(arrays: ArtifactLatentTrainingArrays) -> SubjectArtifactTensorBatch:
    return SubjectArtifactTensorBatch(
        observed=torch.from_numpy(np.asarray(arrays.observed)).float(),
        target_standardized_latent=torch.from_numpy(
            np.asarray(arrays.standardized_artifact_latent)
        ).float(),
        full_transfer=torch.from_numpy(np.asarray(arrays.full_transfer)).float(),
        normalized_transfer=torch.from_numpy(
            np.asarray(arrays.normalized_transfer)
        ).float(),
        transfer_scale=torch.from_numpy(np.asarray(arrays.transfer_scale)).float(),
        singular_values=torch.from_numpy(np.asarray(arrays.singular_values)).float(),
        rank=torch.from_numpy(np.asarray(arrays.rank)).long(),
        rho=torch.from_numpy(np.asarray(arrays.rho)).float(),
        calibration_duration_seconds=torch.from_numpy(
            np.asarray(arrays.calibration_duration_seconds)
        ).float(),
        channel_mask=torch.from_numpy(np.asarray(arrays.channel_mask)).bool(),
        valid_time_mask=torch.from_numpy(np.asarray(arrays.valid_time_mask)).bool(),
    )


def _select_batch(batch: SubjectArtifactTensorBatch, count: int) -> SubjectArtifactTensorBatch:
    if batch.batch_size < count:
        raise ValueError("real V1 fold has fewer windows than the frozen batch size")
    return batch.select(torch.arange(count, dtype=torch.long))


def _weak_target(
    batch: SubjectArtifactTensorBatch,
    normalizer: OuterTrainingLatentNormalizer,
) -> Tensor:
    mean = torch.as_tensor(normalizer.mean, dtype=batch.observed.dtype)
    scale = torch.as_tensor(normalizer.standard_deviation, dtype=batch.observed.dtype)
    latent = batch.target_standardized_latent * scale[None, :, None] + mean[
        None, :, None
    ]
    correction = torch.einsum("bce,bet->bct", batch.normalized_transfer, latent)
    time = batch.valid_time_mask
    if time.ndim == 2:
        time = time[:, None, :]
    mask = time.to(batch.observed.dtype) * batch.channel_mask[:, :, None].to(
        batch.observed.dtype
    )
    return (batch.observed - correction) * mask


def _identity_batch(
    batch: SubjectArtifactTensorBatch,
    normalizer: OuterTrainingLatentNormalizer,
    *,
    physically_zero_standardized_target: bool,
) -> SubjectArtifactTensorBatch:
    weak = _weak_target(batch, normalizer)
    if physically_zero_standardized_target:
        mean = torch.as_tensor(normalizer.mean, dtype=batch.observed.dtype)
        scale = torch.as_tensor(
            normalizer.standard_deviation, dtype=batch.observed.dtype
        )
        zero = (-mean / scale)[None, :, None].expand_as(
            batch.target_standardized_latent
        ).clone()
    else:
        # The existing V1 helper explicitly tests a zero standardized target.
        zero = torch.zeros_like(batch.target_standardized_latent)
    values = {
        item.name: getattr(batch, item.name)
        for item in fields(SubjectArtifactTensorBatch)
    }
    values["observed"] = weak
    values["target_standardized_latent"] = zero
    return SubjectArtifactTensorBatch(**values)


def _concatenate_batches(
    first: SubjectArtifactTensorBatch,
    second: SubjectArtifactTensorBatch,
) -> SubjectArtifactTensorBatch:
    return SubjectArtifactTensorBatch(
        **{
            item.name: torch.cat(
                (getattr(first, item.name), getattr(second, item.name)), dim=0
            )
            for item in fields(SubjectArtifactTensorBatch)
        }
    )


def _model_configs(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    *,
    implementation: Implementation,
) -> tuple[ArtifactLatentModelConfig, ArtifactLatentDiffusionConfig]:
    raw_model = _mapping(config, "model")
    dimensions = prepared.model_dimensions
    model = ArtifactLatentModelConfig(
        eeg_channels=dimensions.eeg_channels,
        signal_length=dimensions.signal_length,
        latent_channels=dimensions.eog_coordinates,
        base_channels=int(raw_model["base_channels"]),
        channel_mults=tuple(int(value) for value in raw_model["channel_mults"]),
        num_res_blocks=int(raw_model["num_res_blocks"]),
        groupnorm_groups=int(raw_model["groupnorm_groups"]),
        dropout=float(raw_model["dropout"]),
        time_sinusoidal_dim=int(raw_model["time_sinusoidal_dim"]),
        time_embed_dim=int(raw_model["time_embed_dim"]),
        attention_heads=int(raw_model["attention_heads"]),
    )
    raw_diffusion = _mapping(config, "primary_diffusion")
    clip = 3.0 if implementation in {"primary_attempt_2", "residual_sdedit_backup"} else float(
        raw_diffusion["standardized_latent_absolute_clip"]
    )
    diffusion = ArtifactLatentDiffusionConfig(
        num_timesteps=int(raw_diffusion["timesteps"]),
        cosine_offset=float(raw_diffusion["cosine_offset"]),
        prediction_target=str(raw_diffusion["prediction_target"]),
        min_snr_gamma=float(raw_diffusion["min_snr_gamma"]),
        dynamic_threshold_quantile=float(
            raw_diffusion["dynamic_threshold_quantile"]
        ),
        standardized_latent_absolute_clip=clip,
        posterior_samples=int(raw_diffusion.get("posterior_samples", 8)),
    )
    return model, diffusion


def _models(
    model_config: ArtifactLatentModelConfig,
    diffusion_config: ArtifactLatentDiffusionConfig,
    *,
    device: torch.device,
) -> tuple[DeterministicArtifactEstimator, ArtifactLatentDiffusion]:
    return (
        DeterministicArtifactEstimator(model_config).to(device),
        ArtifactLatentDiffusion(model_config, diffusion_config).to(device),
    )


def _scaler(enabled: bool) -> Any:
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=1024.0)


def _context(
    runtime: RuntimeArtifactContext,
    normalizer: OuterTrainingLatentNormalizer,
    *,
    role: Literal["population", "subject"],
) -> ArtifactInferenceContext:
    full = torch.as_tensor(runtime.full_transfer, dtype=torch.float64)
    normalized = torch.as_tensor(runtime.normalized_transfer, dtype=torch.float64)
    basis, _, _ = torch.linalg.svd(full, full_matrices=False)
    return ArtifactInferenceContext(
        context_id=runtime.context_id,
        role=role,
        full_transfer=full,
        normalized_transfer=normalized,
        transfer_scale=torch.as_tensor(runtime.transfer_scale, dtype=torch.float64),
        singular_values=torch.as_tensor(
            runtime.singular_values, dtype=torch.float64
        ),
        rank=int(runtime.rank),
        calibration_duration_seconds=float(runtime.calibration_duration_seconds),
        latent_mean=torch.as_tensor(normalizer.mean, dtype=torch.float64),
        latent_standard_deviation=torch.as_tensor(
            normalizer.standard_deviation, dtype=torch.float64
        ),
        subspace_basis=basis[:, : int(runtime.rank)],
    )


def _training_context(
    prepared: PreparedSubjectArtifactFold,
    index: int,
) -> ArtifactInferenceContext:
    arrays = prepared.training
    normalized = torch.as_tensor(
        arrays.normalized_transfer[index], dtype=torch.float64
    )
    transfer_scale = torch.as_tensor(
        arrays.transfer_scale[index], dtype=torch.float64
    )
    # The data carrier is float32; reconstruct the exact registered
    # factorization instead of letting independent casts create tolerance noise.
    full = normalized * transfer_scale[None, :]
    basis, _, _ = torch.linalg.svd(full, full_matrices=False)
    rank = int(arrays.rank[index])
    return ArtifactInferenceContext(
        context_id=f"{arrays.recording_keys[index]}:training_matching",
        role="subject",
        full_transfer=full,
        normalized_transfer=normalized,
        transfer_scale=transfer_scale,
        singular_values=torch.as_tensor(
            arrays.singular_values[index], dtype=torch.float64
        ),
        rank=rank,
        calibration_duration_seconds=float(
            arrays.calibration_duration_seconds[index]
        ),
        latent_mean=torch.as_tensor(
            prepared.latent_normalizer.mean, dtype=torch.float64
        ),
        latent_standard_deviation=torch.as_tensor(
            prepared.latent_normalizer.standard_deviation, dtype=torch.float64
        ),
        subspace_basis=basis[:, :rank],
    )


def _clone_subject(
    context: ArtifactInferenceContext,
    context_id: str,
    *,
    calibration_duration_seconds: float | None = None,
) -> ArtifactInferenceContext:
    return ArtifactInferenceContext(
        context_id=context_id,
        role="subject",
        full_transfer=context.full_transfer,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        calibration_duration_seconds=(
            context.calibration_duration_seconds
            if calibration_duration_seconds is None
            else float(calibration_duration_seconds)
        ),
        latent_mean=context.latent_mean,
        latent_standard_deviation=context.latent_standard_deviation,
        subspace_basis=context.subspace_basis,
    )


def _sample_seeds(seed: int) -> tuple[int, ...]:
    return tuple(int(seed) + 104729 * index for index in range(8))


class _ObservationAnchoredSDEdit:
    """The sole backup sampler, anchored by EEG projection onto the supplied C.

    It reuses the trained v-prediction network.  The query-derived anchor uses
    observed EEG and the support/population transfer only; it has no EOG,
    annotation, outcome, or clean-target interface.
    """

    def __init__(
        self,
        model: ArtifactLatentDiffusion,
        *,
        start_timestep: int = _SDEDIT_START_TIMESTEP,
        anchor_ridge: float = _SDEDIT_ANCHOR_RIDGE,
    ) -> None:
        if not 0 < int(start_timestep) < model.num_timesteps:
            raise ValueError("SDEdit start timestep must lie strictly inside schedule")
        if not math.isfinite(anchor_ridge) or anchor_ridge <= 0.0:
            raise ValueError("SDEdit anchor ridge must be finite and positive")
        self.model = model
        self.start_timestep = int(start_timestep)
        self.anchor_ridge = float(anchor_ridge)

    @torch.no_grad()
    def posterior_mean(self, **kwargs: Any) -> ArtifactPosterior:
        allowed = {
            "observed",
            "full_transfer",
            "normalized_transfer",
            "transfer_scale",
            "singular_values",
            "rank",
            "rho",
            "calibration_duration_seconds",
            "channel_mask",
            "latent_mean",
            "latent_standard_deviation",
            "valid_time_mask",
            "sample_seeds",
            "ddim_steps",
            "record_trajectory",
        }
        if set(kwargs) != allowed:
            raise ValueError(
                f"SDEdit inputs differ from the frozen legal fields: {sorted(set(kwargs) ^ allowed)}"
            )
        observed = kwargs["observed"]
        if not isinstance(observed, Tensor) or observed.ndim != 3:
            raise ValueError("SDEdit observed must have shape (B,C,T)")
        normalized = torch.as_tensor(
            kwargs["normalized_transfer"],
            device=observed.device,
            dtype=observed.dtype,
        )
        latent_channels = self.model.model_config.latent_channels
        if normalized.shape == (observed.shape[1], latent_channels):
            normalized = normalized.unsqueeze(0).expand(observed.shape[0], -1, -1)
        if normalized.shape != (
            observed.shape[0],
            observed.shape[1],
            latent_channels,
        ):
            raise ValueError("SDEdit normalized transfer has invalid shape")
        valid = torch.as_tensor(kwargs["valid_time_mask"], device=observed.device)
        if valid.ndim == 2:
            valid = valid[:, None, :]
        if valid.shape != (observed.shape[0], 1, observed.shape[-1]):
            raise ValueError("SDEdit valid-time mask has invalid shape")
        valid = valid.bool()
        channels = torch.as_tensor(kwargs["channel_mask"], device=observed.device)
        if channels.shape == (observed.shape[1],):
            channels = channels.unsqueeze(0).expand(observed.shape[0], -1)
        if channels.shape != (observed.shape[0], observed.shape[1]):
            raise ValueError("SDEdit channel mask has invalid shape")
        channels = channels.bool()
        output_mask = channels[:, :, None].to(observed.dtype) * valid.to(
            observed.dtype
        )
        latent_mask = valid.to(observed.dtype)
        mean = torch.as_tensor(
            kwargs["latent_mean"], device=observed.device, dtype=observed.dtype
        ).reshape(-1)
        scale = torch.as_tensor(
            kwargs["latent_standard_deviation"],
            device=observed.device,
            dtype=observed.dtype,
        ).reshape(-1)
        if mean.shape != (latent_channels,) or scale.shape != (latent_channels,):
            raise ValueError("SDEdit latent normalization has invalid shape")
        gram = torch.einsum("bce,bcf->bef", normalized, normalized)
        gram = gram + self.anchor_ridge * torch.eye(
            latent_channels, device=observed.device, dtype=observed.dtype
        )[None, :, :]
        right = torch.einsum(
            "bce,bct->bet", normalized, observed * output_mask
        )
        physical_anchor = torch.linalg.solve(gram, right) * latent_mask
        standardized_anchor = (
            (physical_anchor - mean[None, :, None]) / scale[None, :, None]
        ) * latent_mask
        clip = float(
            self.model.diffusion_config.standardized_latent_absolute_clip
        )
        standardized_anchor = standardized_anchor.clamp(-clip, clip) * latent_mask

        raw_seeds = tuple(int(value) for value in kwargs["sample_seeds"])
        if len(raw_seeds) != 8 or len(set(raw_seeds)) != 8:
            raise ValueError("SDEdit requires the frozen eight unique seeds")
        requested_steps = int(kwargs["ddim_steps"])
        proportional_steps = max(
            2,
            int(round(requested_steps * (self.start_timestep + 1) / self.model.num_timesteps)),
        )
        timesteps = tuple(
            int(value)
            for value in torch.linspace(
                self.start_timestep, 0, proportional_steps, dtype=torch.float64
            )
            .round()
            .long()
            .tolist()
        )
        if len(set(timesteps)) != len(timesteps):
            raise AssertionError("SDEdit reverse timestep sequence duplicated a step")
        samples: list[Tensor] = []
        traces: list[ArtifactTrajectoryStep] = []
        network_calls = 0
        batch = observed.shape[0]
        valid_count = (valid.to(observed.dtype).sum() * latent_channels).clamp_min(1.0)
        for sample_index, seed in enumerate(raw_seeds):
            generator = torch.Generator(device=observed.device)
            generator.manual_seed(seed)
            noise = torch.randn(
                standardized_anchor.shape,
                device=observed.device,
                dtype=observed.dtype,
                generator=generator,
            ) * latent_mask
            start = torch.full(
                (batch,),
                self.start_timestep,
                device=observed.device,
                dtype=torch.long,
            )
            latent = self.model.q_sample(standardized_anchor, start, noise) * latent_mask
            previous_rms: float | None = None
            for reverse_index, timestep_value in enumerate(timesteps):
                timestep = torch.full(
                    (batch,), timestep_value, device=observed.device, dtype=torch.long
                )
                predicted_v = self.model.predict_v(
                    latent,
                    timestep,
                    observed=observed,
                    full_transfer=kwargs["full_transfer"],
                    normalized_transfer=kwargs["normalized_transfer"],
                    transfer_scale=kwargs["transfer_scale"],
                    singular_values=kwargs["singular_values"],
                    rank=kwargs["rank"],
                    rho=kwargs["rho"],
                    calibration_duration_seconds=kwargs[
                        "calibration_duration_seconds"
                    ],
                    channel_mask=kwargs["channel_mask"],
                    valid_time_mask=valid,
                )
                network_calls += 1
                predicted_x0, predicted_epsilon = self.model.x0_and_epsilon_from_v(
                    latent, predicted_v, timestep
                )
                predicted_x0, clipped_fraction = self.model._dynamic_threshold(
                    predicted_x0, valid
                )
                if reverse_index == len(timesteps) - 1:
                    next_latent = predicted_x0
                else:
                    next_alpha = self.model.alphas_cumprod[
                        timesteps[reverse_index + 1]
                    ]
                    next_latent = (
                        torch.sqrt(next_alpha) * predicted_x0
                        + torch.sqrt(1.0 - next_alpha) * predicted_epsilon
                    ) * latent_mask
                latent_rms = float(
                    torch.sqrt((latent.square() * latent_mask).sum() / valid_count)
                    .detach()
                    .cpu()
                )
                next_rms = float(
                    torch.sqrt(
                        (next_latent.square() * latent_mask).sum() / valid_count
                    )
                    .detach()
                    .cpu()
                )
                if bool(kwargs.get("record_trajectory", False)):
                    physical = (
                        predicted_x0 * scale[None, :, None] + mean[None, :, None]
                    ) * latent_mask
                    mapped = torch.einsum("bce,bet->bct", normalized, physical)
                    traces.append(
                        ArtifactTrajectoryStep(
                            sample_index=sample_index,
                            reverse_index=reverse_index,
                            timestep=timestep_value,
                            latent_rms=latent_rms,
                            predicted_v_rms=float(
                                torch.sqrt(
                                    (predicted_v.square() * latent_mask).sum()
                                    / valid_count
                                )
                                .detach()
                                .cpu()
                            ),
                            predicted_x0_rms=float(
                                torch.sqrt(
                                    (predicted_x0.square() * latent_mask).sum()
                                    / valid_count
                                )
                                .detach()
                                .cpu()
                            ),
                            mapped_contamination_rms=float(
                                torch.sqrt(
                                    (mapped.square() * output_mask).sum()
                                    / output_mask.sum().clamp_min(1.0)
                                )
                                .detach()
                                .cpu()
                            ),
                            adjacent_latent_rms_ratio=(
                                None
                                if previous_rms is None
                                else next_rms / max(previous_rms, 1.0e-12)
                            ),
                            finite=bool(
                                torch.isfinite(next_latent).all()
                                and torch.isfinite(predicted_v).all()
                                and torch.isfinite(predicted_x0).all()
                            ),
                            clipped_fraction=clipped_fraction,
                        )
                    )
                previous_rms = next_rms
                latent = next_latent
            samples.append(latent * latent_mask)
        latent_mean, latent_std = artifact_posterior_point_estimate(samples)
        physical = (
            latent_mean * scale[None, :, None] + mean[None, :, None]
        ) * latent_mask
        correction = torch.einsum("bce,bet->bct", normalized, physical) * output_mask
        restored = (observed * output_mask - correction) * output_mask
        return ArtifactPosterior(
            standardized_latent_mean=latent_mean,
            standardized_latent_standard_deviation=latent_std,
            correction=correction,
            restored=restored,
            sample_count=8,
            network_calls=network_calls,
            trajectories=tuple(traces),
        )


def _restore(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion | _ObservationAnchoredSDEdit,
    model_kind: ModelKind,
    observed: Tensor,
    valid_time_mask: Tensor,
    population: ArtifactInferenceContext,
    subject: ArtifactInferenceContext | None,
    *,
    rho: float,
    seed: int,
    ddim_steps: int,
) -> PopulationSubjectRestoration:
    channels = torch.ones(
        observed.shape[0], observed.shape[1], dtype=torch.bool, device=observed.device
    )
    factory = None if subject is None else (lambda: subject)
    if model_kind == "deterministic":
        assert isinstance(model, DeterministicArtifactEstimator)
        return deterministic_population_subject_restore(
            model,
            observed,
            population_context=population,
            rho=float(rho),
            subject_context_factory=factory,
            channel_mask=channels,
            valid_time_mask=valid_time_mask,
        )
    return diffusion_population_subject_restore(
        model,
        observed,
        population_context=population,
        rho=float(rho),
        subject_context_factory=factory,
        channel_mask=channels,
        valid_time_mask=valid_time_mask,
        sample_seeds=_sample_seeds(seed),
        ddim_steps=int(ddim_steps),
        record_trajectory=False,
    )


def _masked_window_rms(value: Tensor, mask: Tensor) -> np.ndarray:
    time = mask if mask.ndim == 3 else mask[:, None, :]
    weight = time.to(value.dtype).expand_as(value)
    count = weight.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    rms = torch.sqrt((value.square() * weight).flatten(start_dim=1).sum(dim=1) / count)
    return rms.detach().cpu().double().numpy()


def _valid_values(value: Tensor, mask: Tensor) -> np.ndarray:
    time = mask if mask.ndim == 3 else mask[:, None, :]
    selected = value.masked_select(time.expand_as(value).bool())
    return selected.detach().cpu().double().numpy()


def _channel_variance_ratio(observed: Tensor, restored: Tensor, mask: Tensor) -> np.ndarray:
    time = mask if mask.ndim == 3 else mask[:, None, :]
    values: list[float] = []
    for channel in range(observed.shape[1]):
        keep = time[:, 0, :].bool()
        first = observed[:, channel, :].masked_select(keep)
        second = restored[:, channel, :].masked_select(keep)
        denominator = float(first.var(unbiased=False).detach().cpu())
        numerator = float(second.var(unbiased=False).detach().cpu())
        values.append(numerator / max(denominator, np.finfo(np.float64).eps))
    return np.asarray(values, dtype=np.float64)


def _scale_payload(
    observed: Tensor,
    restored: Tensor,
    mask: Tensor,
    low_observed: Tensor,
    low_restored: Tensor,
    low_mask: Tensor,
    *,
    span_kind: str,
    span_error: float,
) -> dict[str, Any]:
    input_rms = _masked_window_rms(observed, mask)
    output_rms = _masked_window_rms(restored, mask)
    low_input_rms = _masked_window_rms(low_observed, low_mask)
    low_output_rms = _masked_window_rms(low_restored, low_mask)
    epsilon = np.finfo(np.float64).eps
    ratios = output_rms / np.maximum(input_rms, epsilon)
    low_ratios = low_output_rms / np.maximum(low_input_rms, epsilon)
    low_change = _masked_window_rms(low_restored - low_observed, low_mask) / np.maximum(
        low_input_rms, epsilon
    )
    return {
        "output_input_rms_ratio": ratios,
        "low_artifact_output_input_rms_ratio": low_ratios,
        "low_artifact_relative_observation_change": low_change,
        "input_waveform_values": _valid_values(observed, mask),
        "output_waveform_values": _valid_values(restored, mask),
        "channelwise_variance_ratio": _channel_variance_ratio(
            observed, restored, mask
        ),
        "span_consistency_kind": span_kind,
        "span_consistency_relative_error": float(span_error),
    }


def _context_v0_pass(result: Mapping[str, Any]) -> bool:
    return result.get("status") == "passed" and result.get("passed") is True


def _combine_v0_results(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require every model/context V0 result, including wrong and shuffled."""

    required = {
        f"{model}:{context}"
        for model in ("deterministic", "diffusion")
        for context in ("full_training", *_CONTEXTS)
    }
    missing = sorted(required - set(results))
    failed = sorted(
        key for key in required & set(results) if not _context_v0_pass(results[key])
    )
    passed = not missing and not failed
    return {
        "validity_level": "V0",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "required_result_ids": sorted(required),
        "missing_result_ids": missing,
        "failed_result_ids": failed,
        "results": {key: dict(value) for key, value in results.items()},
    }


def _repeat_relative(first: Tensor, second: Tensor) -> float:
    numerator = torch.linalg.vector_norm(first - second)
    denominator = torch.linalg.vector_norm(first).clamp_min(
        torch.finfo(first.dtype).eps
    )
    return float((numerator / denominator).detach().cpu())


def _context_change(reference: Tensor, other: Tensor) -> float:
    return _repeat_relative(reference, other)


def _scale_safe(
    config: Mapping[str, Any], observed: Tensor, restored: Tensor, mask: Tensor
) -> bool:
    v0 = _mapping(_mapping(config, "validity"), "V0")
    lower, upper = (float(value) for value in v0["full_median_output_input_RMS_ratio"])
    maximum = float(v0["maximum_per_window_output_input_RMS_ratio"])
    ratios = _masked_window_rms(restored, mask) / np.maximum(
        _masked_window_rms(observed, mask), np.finfo(np.float64).eps
    )
    return bool(
        np.isfinite(ratios).all()
        and np.max(ratios) <= maximum
        and lower <= np.median(ratios) <= upper
    )


@torch.no_grad()
def _physical_identity_change_by_timestep(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    model_kind: ModelKind,
    identity: SubjectArtifactTensorBatch,
    normalizer: OuterTrainingLatentNormalizer,
    *,
    timesteps: Sequence[int],
    seed: int,
) -> dict[int, float]:
    """Measure identity after converting predicted z back to physical latent A."""

    mask = identity.valid_time_mask
    if mask.ndim == 2:
        mask = mask[:, None, :]
    mask_float = mask.to(identity.observed.dtype)
    physical_zero = identity.target_standardized_latent * mask_float
    mean = torch.as_tensor(
        normalizer.mean,
        device=identity.observed.device,
        dtype=identity.observed.dtype,
    )
    scale = torch.as_tensor(
        normalizer.standard_deviation,
        device=identity.observed.device,
        dtype=identity.observed.dtype,
    )
    result: dict[int, float] = {}
    for raw_timestep in timesteps:
        timestep = int(raw_timestep)
        if model_kind == "deterministic":
            assert isinstance(model, DeterministicArtifactEstimator)
            predicted = model(identity.observed, **identity.model_kwargs())
        else:
            assert isinstance(model, ArtifactLatentDiffusion)
            values = torch.full(
                (identity.batch_size,),
                timestep,
                device=identity.observed.device,
                dtype=torch.long,
            )
            generator = torch.Generator(device=identity.observed.device)
            generator.manual_seed(int(seed) + 2_000_003 * (timestep + 1) + 29)
            noise = torch.randn(
                physical_zero.shape,
                device=physical_zero.device,
                dtype=physical_zero.dtype,
                generator=generator,
            ) * mask_float
            noisy = model.q_sample(physical_zero, values, noise) * mask_float
            predicted_v = model.predict_v(
                noisy,
                values,
                observed=identity.observed,
                **identity.model_kwargs(),
            )
            predicted, _ = model.x0_and_epsilon_from_v(
                noisy, predicted_v, values
            )
            predicted = predicted * mask_float
        output_mask = mask_float * identity.channel_mask[:, :, None].to(
            mask_float.dtype
        )
        correction = canonical_artifact_delta(
            predicted,
            normalized_transfer=identity.normalized_transfer,
            latent_mean=mean,
            latent_standard_deviation=scale,
            output_mask=output_mask,
        )
        observed = identity.observed * output_mask
        numerator = torch.linalg.vector_norm(
            correction.flatten(start_dim=1), dim=1
        )
        denominator = torch.linalg.vector_norm(
            observed.flatten(start_dim=1), dim=1
        ).clamp_min(torch.finfo(observed.dtype).eps)
        result[timestep] = float((numerator / denominator).mean().cpu())
    return result


def _train_one(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    source: SubjectArtifactTensorBatch,
    *,
    model_kind: ModelKind,
    seed: int,
    config: Mapping[str, Any],
    implementation: Implementation,
    output: Path,
    device: torch.device,
) -> tuple[CheckpointableEMA, list[dict[str, Any]], float]:
    training = _mapping(config, "training")
    validity = _mapping(config, "validity")
    updates = int(validity["diagnostic_training_updates"])
    schedule = build_shared_minibatch_schedule(
        sample_count=source.batch_size,
        batch_size=int(training["batch_size"]),
        updates=updates,
        seed=int(seed),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    mixed = bool(training["mixed_precision"])
    scaler = _scaler(mixed)
    ema = CheckpointableEMA(
        model, decay=float(_mapping(config, "primary_diffusion")["ema_decay"])
    )
    contract = {
        "protocol_id": str(config["protocol_id"]),
        "execution_revision": str(
            _mapping(config, "validity")["execution_revision"]
        ),
        "stage": "V0_V3_diagnostic_training",
        "implementation": implementation,
        "model_kind": model_kind,
        "seed": int(seed),
        "updates": updates,
        "sample_count": source.batch_size,
    }
    checkpoint = output / "checkpoints" / f"{model_kind}.pt"
    step = 0
    history: list[dict[str, Any]] = []
    if checkpoint.is_file():
        resumed = resume_artifact_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            expected_contract=contract,
            map_location=device,
        )
        step = int(resumed.step)
        history.extend(dict(row) for row in resumed.history)
    checkpoint_interval = int(training["checkpoint_interval_updates"])
    started = time.perf_counter()
    while step < updates:
        stop = min(updates, step + checkpoint_interval)
        run = train_artifact_updates(
            model,
            source,
            model_kind=model_kind,
            schedule=schedule,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
            start_step=step,
            stop_step=stop,
            training_seed=int(seed),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            mixed_precision=mixed,
        )
        history.extend(dict(row) for row in run.loss_curve)
        step = int(run.completed_step)
        save_artifact_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            epoch=0,
            step=step,
            contract=contract,
            history=history,
            extra={"completed": step == updates},
        )
    return ema, history, time.perf_counter() - started


def _v1_shared_fit_batch(
    source: SubjectArtifactTensorBatch,
    normalizer: OuterTrainingLatentNormalizer,
    *,
    count: int,
    identity_repair_active: bool,
) -> tuple[SubjectArtifactTensorBatch, int]:
    """Route the identity repair into the one batch shared by both V1 models."""

    base = _select_batch(source, int(count))
    if not identity_repair_active:
        return base, 0
    # The frozen 25% is an addition to the original batch, not 25% of the
    # post-augmentation total: base=3 therefore receives one identity example.
    identity_count = max(1, int(math.ceil(base.batch_size * 0.25)))
    identity_source = base.select(torch.arange(identity_count, dtype=torch.long))
    physical_identity = _identity_batch(
        identity_source,
        normalizer,
        physically_zero_standardized_target=True,
    )
    return _concatenate_batches(base, physical_identity), identity_count


def _overfit_v1(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    source: SubjectArtifactTensorBatch,
    model_config: ArtifactLatentModelConfig,
    diffusion_config: ArtifactLatentDiffusionConfig,
    *,
    device: torch.device,
    identity_repair_active: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validity = _mapping(config, "validity")
    training = _mapping(config, "training")
    count = int(validity["single_batch_overfit_batch_size"])
    fit_cpu, identity_fit_count = _v1_shared_fit_batch(
        source,
        prepared.latent_normalizer,
        count=count,
        identity_repair_active=identity_repair_active,
    )
    fit = fit_cpu.to(device)
    # Both calls below receive this exact object and therefore the exact same target.
    shared_fit_object = fit
    shared_target_id = (
        f"{prepared.fold.fold_id}:same_real_batch_base_{count}:"
        f"physical_zero_identity_{identity_fit_count}:total_{fit.batch_size}"
    )
    physical_identity = _identity_batch(
        _select_batch(source, count),
        prepared.latent_normalizer,
        physically_zero_standardized_target=True,
    ).to(device)
    deterministic, diffusion = _models(
        model_config, diffusion_config, device=device
    )
    updates = int(validity["single_batch_overfit_maximum_updates"])
    timesteps = tuple(int(value) for value in _mapping(validity, "V1")["timesteps"])
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for model_kind, model, offset in (
        ("deterministic", deterministic, 0),
        ("diffusion", diffusion, 1),
    ):
        if shared_fit_object is not fit:
            raise AssertionError("V1 models no longer share the exact real fit batch")
        optimizer = AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        ema = CheckpointableEMA(
            model, decay=float(_mapping(config, "primary_diffusion")["ema_decay"])
        )
        result = run_v1_fixed_batch_overfit(
            model,
            shared_fit_object,
            identity_batch=physical_identity,
            latent_mean=torch.as_tensor(
                prepared.latent_normalizer.mean,
                device=device,
                dtype=shared_fit_object.observed.dtype,
            ),
            latent_standard_deviation=torch.as_tensor(
                prepared.latent_normalizer.standard_deviation,
                device=device,
                dtype=shared_fit_object.observed.dtype,
            ),
            model_kind=model_kind,  # type: ignore[arg-type]
            model_id=f"artifact_latent_{model_kind}",
            target_id=shared_target_id,
            optimizer=optimizer,
            scaler=_scaler(bool(training["mixed_precision"])),
            ema=ema,
            updates=updates,
            training_seed=int(training["seeds"][0]) + offset,
            validation_timesteps=timesteps,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            mixed_precision=bool(training["mixed_precision"]),
            check_interval_updates=int(validity["check_interval_updates"]),
        )
        with ema.average_parameters(model):
            model.eval()
            physical_identity_change = _physical_identity_change_by_timestep(
                model,
                model_kind,  # type: ignore[arg-type]
                physical_identity,
                prepared.latent_normalizer,
                timesteps=timesteps,
                seed=int(training["seeds"][0]) + offset,
            )
        for row in result.validity_rows():
            row["zero_artifact_relative_observation_change"] = float(
                physical_identity_change[int(row["timestep"])]
            )
            rows.append(row)
        curves.extend(
            {"model_kind": model_kind, **dict(row)} for row in result.loss_curve
        )
    return evaluate_v1(config, {"timestep_results": rows}), curves


def _group_training_indices(prepared: PreparedSubjectArtifactFold) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(prepared.training.recording_keys):
        grouped[str(key)].append(index)
    return [grouped[key] for key in sorted(grouped)]


def _full_training_outputs(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion | _ObservationAnchoredSDEdit,
    model_kind: ModelKind,
    prepared: PreparedSubjectArtifactFold,
    source: SubjectArtifactTensorBatch,
    *,
    device: torch.device,
    seed: int,
    ddim_steps: int,
    chunk_size: int,
) -> tuple[Tensor, Tensor, float]:
    population = _context(
        prepared.population_context, prepared.latent_normalizer, role="population"
    )
    outputs: list[Tensor] = []
    low_outputs: list[Tensor] = []
    maximum_error = 0.0
    low_source = _identity_batch(
        source,
        prepared.latent_normalizer,
        physically_zero_standardized_target=True,
    )
    for indices in _group_training_indices(prepared):
        context = _training_context(prepared, indices[0])
        rho = float(prepared.training.rho[indices[0]])
        for start in range(0, len(indices), chunk_size):
            selection = indices[start : start + chunk_size]
            index = torch.tensor(selection, dtype=torch.long)
            current = source.select(index)
            low = low_source.select(index)
            for item, destination in ((current, outputs), (low, low_outputs)):
                observed = item.observed.to(device)
                mask = item.valid_time_mask.to(device)
                result = _restore(
                    model,
                    model_kind,
                    observed,
                    mask,
                    population,
                    context,
                    rho=rho,
                    seed=seed,
                    ddim_steps=ddim_steps,
                )
                destination.append(result.restored.detach().cpu())
                maximum_error = max(maximum_error, result.complement_relative_error)
    # Group iteration changes row order; preserve it explicitly for aligned V0 values.
    all_outputs = torch.cat(outputs, dim=0)
    all_low_outputs = torch.cat(low_outputs, dim=0)
    keyed_outputs: dict[int, Tensor] = {}
    keyed_low: dict[int, Tensor] = {}
    cursor = 0
    for indices in _group_training_indices(prepared):
        for original in indices:
            keyed_outputs[original] = all_outputs[cursor]
            keyed_low[original] = all_low_outputs[cursor]
            cursor += 1
    ordered = torch.stack([keyed_outputs[index] for index in range(source.batch_size)])
    ordered_low = torch.stack([keyed_low[index] for index in range(source.batch_size)])
    return ordered, ordered_low, maximum_error


def _heldout_context_results(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion | _ObservationAnchoredSDEdit,
    model_kind: ModelKind,
    prepared: PreparedSubjectArtifactFold,
    low_source: SubjectArtifactTensorBatch,
    *,
    device: torch.device,
    seed: int,
    ddim_steps: int,
) -> tuple[
    dict[str, tuple[Tensor, Tensor, Tensor, float, Tensor, float]],
    dict[str, PopulationSubjectRestoration],
    dict[str, Tensor],
]:
    population = _context(
        prepared.population_context, prepared.latent_normalizer, role="population"
    )
    accumulated: dict[str, list[Tensor]] = {name: [] for name in _CONTEXTS}
    observations: dict[str, list[Tensor]] = {name: [] for name in _CONTEXTS}
    masks: dict[str, list[Tensor]] = {name: [] for name in _CONTEXTS}
    deltas: dict[str, list[Tensor]] = {name: [] for name in _CONTEXTS}
    repeat_errors: dict[str, float] = {name: 0.0 for name in _CONTEXTS}
    errors: dict[str, float] = {name: 0.0 for name in _CONTEXTS}
    first_results: dict[str, PopulationSubjectRestoration] = {}
    for heldout in prepared.heldout.values():
        observed = torch.from_numpy(np.asarray(heldout.query.observed)).float().to(device)
        mask = torch.from_numpy(np.asarray(heldout.query.valid_time_mask)).bool().to(device)
        rho = float(heldout.matching.rho)
        contexts = {
            "population": _clone_subject(
                population,
                f"{heldout.recording_key}:population_swap",
                calibration_duration_seconds=heldout.matching.calibration_duration_seconds,
            ),
            "matching": _context(heldout.matching, prepared.latent_normalizer, role="subject"),
            "wrong": _context(heldout.wrong_same_cell, prepared.latent_normalizer, role="subject"),
            "shuffled": _context(heldout.shuffled_same_cell, prepared.latent_normalizer, role="subject"),
        }
        for context_name, subject in contexts.items():
            first = _restore(
                model,
                model_kind,
                observed,
                mask,
                population,
                subject,
                rho=rho,
                seed=seed,
                ddim_steps=ddim_steps,
            )
            second = _restore(
                model,
                model_kind,
                observed,
                mask,
                population,
                subject,
                rho=rho,
                seed=seed,
                ddim_steps=ddim_steps,
            )
            repeat_error = _repeat_relative(first.mixed_delta, second.mixed_delta)
            if repeat_error > 1.0e-6:
                raise AssertionError("same-context V3 repeat changed above tolerance")
            first_results.setdefault(context_name, first)
            accumulated[context_name].append(first.restored.detach().cpu())
            observations[context_name].append(observed.detach().cpu())
            masks[context_name].append(mask.detach().cpu())
            deltas[context_name].append(first.mixed_delta.detach().cpu())
            repeat_errors[context_name] = max(
                repeat_errors[context_name], repeat_error
            )
            errors[context_name] = max(
                errors[context_name], first.complement_relative_error
            )
    representative = next(iter(prepared.heldout.values()))
    representative_rho = float(representative.matching.rho)
    low_contexts = {
        "population": _clone_subject(
            population,
            f"{representative.recording_key}:population_swap:low_artifact",
            calibration_duration_seconds=representative.matching.calibration_duration_seconds,
        ),
        "matching": _context(
            representative.matching, prepared.latent_normalizer, role="subject"
        ),
        "wrong": _context(
            representative.wrong_same_cell,
            prepared.latent_normalizer,
            role="subject",
        ),
        "shuffled": _context(
            representative.shuffled_same_cell,
            prepared.latent_normalizer,
            role="subject",
        ),
    }
    low_by_context: dict[str, list[Tensor]] = {name: [] for name in _CONTEXTS}
    for start in range(0, low_source.batch_size, 32):
        index = torch.arange(start, min(low_source.batch_size, start + 32))
        batch = low_source.select(index)
        observed = batch.observed.to(device)
        mask = batch.valid_time_mask.to(device)
        for context_name, subject in low_contexts.items():
            result = _restore(
                model,
                model_kind,
                observed,
                mask,
                population,
                subject,
                rho=representative_rho,
                seed=seed,
                ddim_steps=ddim_steps,
            )
            low_by_context[context_name].append(result.restored.detach().cpu())
    return (
        {
            name: (
                torch.cat(observations[name]),
                torch.cat(accumulated[name]),
                torch.cat(masks[name]),
                errors[name],
                torch.cat(deltas[name]),
                repeat_errors[name],
            )
            for name in _CONTEXTS
        },
        first_results,
        {name: torch.cat(values) for name, values in low_by_context.items()},
    )


def _rho_zero_short_circuit_audit(
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion | _ObservationAnchoredSDEdit,
    model_kind: ModelKind,
    prepared: PreparedSubjectArtifactFold,
    *,
    device: torch.device,
    seed: int,
    ddim_steps: int,
    complement_tolerance: float,
) -> dict[str, Any]:
    heldout = next(iter(prepared.heldout.values()))
    observed = torch.from_numpy(np.asarray(heldout.query.observed)).float().to(device)
    mask = torch.from_numpy(np.asarray(heldout.query.valid_time_mask)).bool().to(device)
    population = _context(
        prepared.population_context, prepared.latent_normalizer, role="population"
    )
    channel_mask = torch.ones(
        observed.shape[0], observed.shape[1], dtype=torch.bool, device=device
    )
    factory_calls = 0

    def forbidden_factory() -> ArtifactInferenceContext:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("rho=0 constructed the personalized context")

    if model_kind == "deterministic":
        assert isinstance(model, DeterministicArtifactEstimator)
        result = deterministic_population_subject_restore(
            model,
            observed,
            population_context=population,
            rho=0.0,
            subject_context_factory=forbidden_factory,
            channel_mask=channel_mask,
            valid_time_mask=mask,
        )
    else:
        result = diffusion_population_subject_restore(
            model,
            observed,
            population_context=population,
            rho=0.0,
            subject_context_factory=forbidden_factory,
            channel_mask=channel_mask,
            valid_time_mask=mask,
            sample_seeds=_sample_seeds(seed),
            ddim_steps=ddim_steps,
            record_trajectory=False,
        )
    matching = _context(
        heldout.matching, prepared.latent_normalizer, role="subject"
    )
    pure_matching = _restore(
        model,
        model_kind,
        observed,
        mask,
        population,
        matching,
        rho=1.0,
        seed=seed,
        ddim_steps=ddim_steps,
    )
    matching_basis = matching.subspace_basis.to(
        device=pure_matching.mixed_delta.device,
        dtype=pure_matching.mixed_delta.dtype,
    )
    matching_projector = matching_basis @ matching_basis.T
    matching_complement = torch.eye(
        matching_projector.shape[0],
        device=matching_projector.device,
        dtype=matching_projector.dtype,
    ) - matching_projector
    matching_residual = torch.einsum(
        "cd,bdt->bct", matching_complement, pure_matching.mixed_delta
    )
    matching_denominator = torch.linalg.vector_norm(pure_matching.mixed_delta)
    matching_numerator = torch.linalg.vector_norm(matching_residual)
    matching_error = (
        0.0
        if float(matching_denominator) == 0.0 and float(matching_numerator) == 0.0
        else float("inf")
        if float(matching_denominator) == 0.0
        else float((matching_numerator / matching_denominator).detach().cpu())
    )
    passed = bool(
        factory_calls == 0
        and result.branch == "population"
        and result.subject is None
        and not result.subject_context_constructed
        and torch.isfinite(result.restored).all()
        and result.complement_relative_error <= float(complement_tolerance)
        and matching_error <= float(complement_tolerance)
    )
    return {
        "passed": passed,
        "subject_context_factory_calls": factory_calls,
        "branch": result.branch,
        "population_sampler_calls": result.population.sampler_calls,
        "population_network_calls": result.population.network_calls,
        "subject_sampler_calls": 0,
        "population_is_raw": False,
        "population_complement_relative_error": result.complement_relative_error,
        "matching_complement_relative_error": matching_error,
        "configured_tolerance": float(complement_tolerance),
    }


def _trajectory(
    model: ArtifactLatentDiffusion | _ObservationAnchoredSDEdit,
    prepared: PreparedSubjectArtifactFold,
    *,
    device: torch.device,
    seed: int,
    ddim_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    heldout = next(iter(prepared.heldout.values()))
    observed = torch.from_numpy(np.asarray(heldout.query.observed)).float().to(device)
    mask = torch.from_numpy(np.asarray(heldout.query.valid_time_mask)).bool().to(device)
    context = _context(heldout.matching, prepared.latent_normalizer, role="subject")
    posterior = model.posterior_mean(
        observed=observed,
        full_transfer=context.full_transfer,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        rho=float(heldout.matching.rho),
        calibration_duration_seconds=context.calibration_duration_seconds,
        channel_mask=torch.ones(
            observed.shape[0], observed.shape[1], dtype=torch.bool, device=device
        ),
        latent_mean=context.latent_mean,
        latent_standard_deviation=context.latent_standard_deviation,
        valid_time_mask=mask,
        sample_seeds=_sample_seeds(seed),
        ddim_steps=ddim_steps,
        record_trajectory=True,
    )
    raw_rows = [asdict(value) for value in posterior.trajectories]
    trajectories: list[dict[str, Any]] = []
    metric_fields = {
        "latent_A_t": "latent_rms",
        "predicted_v": "predicted_v_rms",
        "predicted_A0": "predicted_x0_rms",
        "mapped_C_A0": "mapped_contamination_rms",
    }
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_sample[int(row["sample_index"])].append(row)
    for sample_index, rows in sorted(by_sample.items()):
        rows.sort(key=lambda item: int(item["reverse_index"]))
        for metric_id, field in metric_fields.items():
            trajectories.append(
                {
                    "trajectory_id": f"sample{sample_index}:{metric_id}",
                    "steps": [int(item["timestep"]) for item in rows],
                    "rms": [float(item[field]) for item in rows],
                }
            )
    return trajectories, raw_rows


def _plot_trajectory(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    fields = {
        "A_t": "latent_rms",
        "vhat": "predicted_v_rms",
        "Ahat0": "predicted_x0_rms",
        "C_Ahat0": "mapped_contamination_rms",
    }
    for row in rows:
        for label in fields:
            grouped[(int(row["sample_index"]), label)].append(row)
    for (sample, label), values in sorted(grouped.items()):
        values = sorted(values, key=lambda item: int(item["reverse_index"]))
        axis.plot(
            [int(item["timestep"]) for item in values],
            [float(item[fields[label]]) for item in values],
            alpha=0.32,
            linewidth=0.8,
            label=label if sample == 0 else None,
        )
    axis.set_yscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("diffusion timestep")
    axis.set_ylabel("RMS")
    axis.set_title("Artifact-latent reverse trajectory (K=8)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _first_trajectory_instability(
    trajectories: Sequence[Mapping[str, Any]], maximum_ratio: float
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for raw in trajectories:
        steps = [int(value) for value in raw["steps"]]
        rms = [float(value) for value in raw["rms"]]
        for index in range(1, len(rms)):
            denominator = rms[index - 1]
            ratio = (
                1.0
                if denominator == 0.0 and rms[index] == 0.0
                else float("inf")
                if denominator == 0.0
                else rms[index] / denominator
            )
            if ratio > float(maximum_ratio):
                failures.append(
                    {
                        "trajectory_id": str(raw["trajectory_id"]),
                        "first_unstable_timestep": steps[index],
                        "adjacent_RMS_ratio": ratio if math.isfinite(ratio) else None,
                    }
                )
                break
    return failures


def _activation_allowed(output_root: Path, implementation: Implementation) -> None:
    if implementation == "primary_attempt_0":
        return
    previous_name = {
        "primary_attempt_1": "primary_attempt_0",
        "primary_attempt_2": (
            "primary_attempt_1"
            if (output_root / "primary_attempt_1" / "result_summary.json").is_file()
            else "primary_attempt_0"
        ),
        "residual_sdedit_backup": (
            "primary_attempt_2"
            if (output_root / "primary_attempt_2" / "result_summary.json").is_file()
            else "primary_attempt_1"
            if (output_root / "primary_attempt_1" / "result_summary.json").is_file()
            else "primary_attempt_0"
        ),
    }[implementation]
    path = output_root / previous_name / "result_summary.json"
    if not path.is_file():
        raise RuntimeError(f"{implementation} requires completed {previous_name}")
    previous = json.loads(path.read_text(encoding="utf-8"))
    levels = previous.get("validity", {})
    failed = {name for name, value in levels.items() if not bool(value.get("passed"))}
    v0 = levels.get("V0", {})
    nested_v0 = v0.get("results", {}) if isinstance(v0, Mapping) else {}
    v0_failed_checks = {
        str(check_name)
        for result in nested_v0.values()
        if isinstance(result, Mapping)
        for check_name, check in result.get("checks", {}).items()
        if isinstance(check, Mapping) and check.get("passed") is not True
    }
    low_identity_failure = bool(
        v0_failed_checks
        & {
            "low_artifact_quantile_q50_bounds",
            "low_artifact_variance_ratio_q50_bounds",
            "low_artifact_observation_change",
        }
    )
    output_scale_failure = bool(
        v0_failed_checks
        & {
            "all_finite",
            "per_window_scale_safety",
            "full_quantile_q50_bounds",
            "low_artifact_quantile_q50_bounds",
            "full_variance_ratio_q50_bounds",
            "low_artifact_variance_ratio_q50_bounds",
            "low_artifact_observation_change",
        }
    )
    permitted = {
        "primary_attempt_1": "V1" in failed or low_identity_failure,
        "primary_attempt_2": "V2" in failed or output_scale_failure,
        "residual_sdedit_backup": bool(failed & {"V0", "V1", "V2"}),
    }[implementation]
    if not permitted:
        raise RuntimeError(
            f"{implementation} activation is not supported by {previous_name}: {sorted(failed)}"
        )


def _validity_output_paths(
    config: Mapping[str, Any], implementation: Implementation
) -> tuple[Path, Path, Path, str]:
    """Resolve a revision-isolated attempt directory and the latest-gate root."""

    output_root = Path(str(_mapping(config, "outputs")["validity_root"]))
    raw_revision = str(_mapping(config, "validity").get("execution_revision", ""))
    revision = raw_revision.strip()
    if (
        not revision
        or revision in {".", ".."}
        or Path(revision).is_absolute()
        or Path(revision).name != revision
        or "/" in revision
        or "\\" in revision
    ):
        raise ValueError("validity.execution_revision must be one safe path component")
    attempt_root = output_root / revision
    return output_root, attempt_root, attempt_root / implementation, revision


def _identity_repair_active(
    attempt_root: Path, implementation: Implementation
) -> bool:
    """Apply repair 1 only when selected or inherited within this revision."""

    return implementation == "primary_attempt_1" or (
        implementation in {"primary_attempt_2", "residual_sdedit_backup"}
        and (
            attempt_root / "primary_attempt_1" / "result_summary.json"
        ).is_file()
    )


def run_subject_artifact_validity(
    config: Mapping[str, Any],
    run_dir: str | Path,
    implementation: str,
) -> Mapping[str, Any]:
    """Run one frozen primary validity attempt or the sole SDEdit backup.

    This API intentionally has no query EOG, eye-tracking, artifact-label,
    outcome, clean-target, or best-sample argument.
    """

    if implementation not in {
        "primary_attempt_0",
        "primary_attempt_1",
        "primary_attempt_2",
        "residual_sdedit_backup",
    }:
        raise ValueError("unknown subject-artifact validity implementation")
    implementation = str(implementation)
    output_root, attempt_root, output, execution_revision = _validity_output_paths(
        config, implementation  # type: ignore[arg-type]
    )
    output.mkdir(parents=True, exist_ok=True)
    _activation_allowed(attempt_root, implementation)  # type: ignore[arg-type]

    device = _device()
    cuda_device_index = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(cuda_device_index)
    validity_config = _mapping(config, "validity")
    identity_repair_active = _identity_repair_active(
        attempt_root, implementation  # type: ignore[arg-type]
    )
    fold_index = int(validity_config["development_fold_index"])
    prepared = prepare_subject_artifact_fold(config, fold_index)
    if any(value.query.annotations_sealed is not True for value in prepared.heldout.values()):
        raise AssertionError("J2 received an unsealed query")
    source = _tensor_batch(prepared.training)
    model_config, diffusion_config = _model_configs(
        config, prepared, implementation=implementation  # type: ignore[arg-type]
    )
    seed = int(_mapping(config, "training")["seeds"][0])
    _seed_everything(seed)
    v1, v1_curves = _overfit_v1(
        config,
        prepared,
        source,
        model_config,
        diffusion_config,
        device=device,
        identity_repair_active=identity_repair_active,
    )

    training_source = source
    if identity_repair_active:
        identity_count = max(1, int(math.ceil(source.batch_size * 0.25)))
        identity = _identity_batch(
            source.select(torch.arange(identity_count)),
            prepared.latent_normalizer,
            physically_zero_standardized_target=True,
        )
        training_source = _concatenate_batches(source, identity)

    deterministic, diffusion = _models(
        model_config, diffusion_config, device=device
    )
    train_rows: list[dict[str, Any]] = []
    training_runtime: dict[str, float] = {}
    ema_by_kind: dict[str, CheckpointableEMA] = {}
    for model_kind, model in (
        ("deterministic", deterministic),
        ("diffusion", diffusion),
    ):
        _seed_everything(seed)
        training_output = (
            attempt_root / "primary_attempt_2"
            if implementation == "residual_sdedit_backup"
            else output
        )
        training_implementation: Implementation = (
            "primary_attempt_2"
            if implementation == "residual_sdedit_backup"
            else implementation  # type: ignore[assignment]
        )
        ema, curve, runtime = _train_one(
            model,
            training_source,
            model_kind=model_kind,  # type: ignore[arg-type]
            seed=seed,
            config=config,
            implementation=training_implementation,
            output=training_output,
            device=device,
        )
        ema_by_kind[model_kind] = ema
        train_rows.extend({"model_kind": model_kind, **row} for row in curve)
        training_runtime[model_kind] = runtime

    ddim_steps = int(_mapping(config, "primary_diffusion")["ddim_steps"])
    # Sampling is sequential in K and timestep, so a larger inference-only
    # batch reduces network-call overhead without changing samples or outputs.
    chunk_size = max(32, int(_mapping(config, "training")["batch_size"]))
    low_observed = _weak_target(source, prepared.latent_normalizer)
    low_mask = source.valid_time_mask
    low_source = _identity_batch(
        source,
        prepared.latent_normalizer,
        physically_zero_standardized_target=True,
    )
    v0_results: dict[str, Mapping[str, Any]] = {}
    heldout_results: dict[str, dict[str, PopulationSubjectRestoration]] = {}
    heldout_arrays: dict[
        str, dict[str, tuple[Tensor, Tensor, Tensor, float, Tensor, float]]
    ] = {}
    rho_zero_audit: dict[str, Mapping[str, Any]] = {}
    for model_kind, model in (
        ("deterministic", deterministic),
        ("diffusion", diffusion),
    ):
        with ema_by_kind[model_kind].average_parameters(model):
            model.eval()
            inference_model: Any = (
                _ObservationAnchoredSDEdit(diffusion)
                if implementation == "residual_sdedit_backup"
                and model_kind == "diffusion"
                else model
            )
            full_output, low_output, span_error = _full_training_outputs(
                inference_model,
                model_kind,  # type: ignore[arg-type]
                prepared,
                source,
                device=device,
                seed=seed,
                ddim_steps=ddim_steps,
                chunk_size=chunk_size,
            )
            full_payload = _scale_payload(
                source.observed,
                full_output,
                source.valid_time_mask,
                low_observed,
                low_output,
                low_mask,
                span_kind="union",
                span_error=span_error,
            )
            v0_results[f"{model_kind}:full_training"] = evaluate_v0(
                config, full_payload
            )
            arrays, first, low_by_context = _heldout_context_results(
                inference_model,
                model_kind,  # type: ignore[arg-type]
                prepared,
                low_source,
                device=device,
                seed=seed,
                ddim_steps=ddim_steps,
            )
            heldout_arrays[model_kind] = arrays
            heldout_results[model_kind] = first
            rho_zero_audit[model_kind] = _rho_zero_short_circuit_audit(
                inference_model,
                model_kind,  # type: ignore[arg-type]
                prepared,
                device=device,
                seed=seed,
                ddim_steps=ddim_steps,
                complement_tolerance=float(
                    _mapping(_mapping(config, "validity"), "V0")[
                        "pure_operator_maximum_complement_consistency_relative_error"
                    ]
                ),
            )
            for context_name, (observed, restored, mask, error, _, _) in arrays.items():
                # Every held-out output is a population/subject mixture at the
                # original rho, hence the applicable invariant is union-span.
                payload = _scale_payload(
                    observed,
                    restored,
                    mask,
                    low_observed,
                    low_by_context[context_name],
                    low_mask,
                    span_kind="union",
                    span_error=error,
                )
                v0_results[f"{model_kind}:{context_name}"] = evaluate_v0(
                    config, payload
                )
    v0 = _combine_v0_results(v0_results)
    if not all(bool(value.get("passed")) for value in rho_zero_audit.values()):
        v0["passed"] = False
        v0["status"] = "failed"
        v0["failed_result_ids"] = sorted(
            set(v0["failed_result_ids"])
            | {
                f"{model_kind}:pure_operator_consistency"
                for model_kind, value in rho_zero_audit.items()
                if not bool(value.get("passed"))
            }
        )

    with ema_by_kind["diffusion"].average_parameters(diffusion):
        diffusion.eval()
        trajectory_model: ArtifactLatentDiffusion | _ObservationAnchoredSDEdit = (
            _ObservationAnchoredSDEdit(diffusion)
            if implementation == "residual_sdedit_backup"
            else diffusion
        )
        raw_trajectories, trajectory_rows = _trajectory(
            trajectory_model,
            prepared,
            device=device,
            seed=seed,
            ddim_steps=ddim_steps,
        )
    v2 = evaluate_v2(
        config,
        {
            "trajectories": raw_trajectories,
            "final_v0_status": (
                "passed"
                if _context_v0_pass(v0_results["diffusion:matching"])
                else "failed"
            ),
        },
    )
    v2["first_instability_by_trajectory"] = _first_trajectory_instability(
        raw_trajectories,
        float(
            _mapping(_mapping(config, "validity"), "V2")[
                "maximum_unexplained_adjacent_RMS_ratio"
            ]
        ),
    )

    v3_by_model: dict[str, Mapping[str, Any]] = {}
    for model_kind in ("deterministic", "diffusion"):
        first = heldout_results[model_kind]
        matching = heldout_arrays[model_kind]["matching"][4]
        repeats: dict[str, float] = {}
        changes: dict[str, float] = {}
        safety: dict[str, bool] = {}
        rho_values: dict[str, float] = {}
        for name in _CONTEXTS:
            result = first[name]
            observed, restored, mask, _, delta, repeat_error = heldout_arrays[
                model_kind
            ][name]
            repeats[name] = repeat_error
            if name != "matching":
                changes[name] = _context_change(matching, delta)
            safety[name] = _scale_safe(config, observed, restored, mask)
            rho_values[name] = float(result.rho)
        v3_by_model[model_kind] = evaluate_v3(
            config,
            {
                "repeat_relative_difference_by_context": repeats,
                "context_swap_artifact_relative_change": changes,
                "scale_safety_by_context": safety,
                "rho_by_context": rho_values,
            },
        )
    v3_passed = all(bool(value.get("passed")) for value in v3_by_model.values())
    v3 = {
        "validity_level": "V3",
        "status": "passed" if v3_passed else "failed",
        "passed": v3_passed,
        "by_model": v3_by_model,
    }

    validity = {"V0": v0, "V1": v1, "V2": v2, "V3": v3}
    passed = all(bool(value.get("passed")) for value in validity.values())
    summary: dict[str, Any] = {
        "stage": "J2_validity",
        "implementation": implementation,
        "execution_revision": execution_revision,
        "supersedes_diagnostic": validity_config.get("supersedes_diagnostic"),
        "attempt_result_path": str(output / "result_summary.json"),
        "identity_repair_active": identity_repair_active,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "model_validity": "passed" if passed else "failed",
        "scientific_comparison_eligibility": (
            "eligible_for_full_development_factorial" if passed else "blocked"
        ),
        "confirmation_eligibility": False,
        "fold": asdict(prepared.fold),
        "run_dir": str(Path(run_dir)),
        "query_annotations_opened": False,
        "training_window_count": source.batch_size,
        "training_runtime_seconds": training_runtime,
        "peak_cuda_memory_mb": float(
            torch.cuda.max_memory_allocated(cuda_device_index)
            / (1024.0 * 1024.0)
        ),
        "rho_zero_population_short_circuit": rho_zero_audit,
        "validity": validity,
        "checkpoint_resume_supported": True,
        "backup_sampler": (
            {
                "method": "observation_anchored_residual_SDEdit_artifact_latent",
                "start_timestep": _SDEDIT_START_TIMESTEP,
                "anchor_ridge": _SDEDIT_ANCHOR_RIDGE,
                "query_EOG_or_labels_used": False,
            }
            if implementation == "residual_sdedit_backup"
            else None
        ),
    }
    _write_csv(output / "v1_loss_curves.csv", v1_curves)
    _write_csv(output / "diagnostic_training_curves.csv", train_rows)
    _write_csv(output / "trajectory.csv", trajectory_rows)
    _plot_trajectory(output / "diffusion_trajectory_rms.png", trajectory_rows)
    _write_csv(
        output / "validity_metrics.csv",
        [
            {
                "level": level,
                "passed": bool(value.get("passed")),
                "status": value.get("status"),
                "details": value,
            }
            for level, value in validity.items()
        ],
    )
    # Publish the detailed attempt first.  The root wrapper is deliberately
    # written last so it always denotes the latest fully materialized gate,
    # while historical direct-attempt directories remain untouched.
    _atomic_json(output / "result_summary.json", summary)
    _atomic_json(output_root / "result_summary.json", summary)
    return summary


__all__ = ["run_subject_artifact_validity"]
