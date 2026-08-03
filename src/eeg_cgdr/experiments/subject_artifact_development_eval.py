"""Development-only natural-SGE 2x4 artifact-latent evaluation.

One Slurm array task evaluates one frozen exact-cell fold and one training
seed.  Both learned models share the same query windows, normalization and
(for diffusion) K=8 random streams at both best and equal-update checkpoints.
Population, matching, wrong and shuffled are runtime context substitutions;
they are not separately trained models.

Block-2 EOG and annotations are deliberately absent from every inference
surface.  They are reopened only after all 16 learned endpoint/arm point outputs
and their explicit low-dimensional K=8 standardized latents have been copied
into an immutable object and an atomic freeze manifest has been published.
EEG-space posterior samples are reconstructed one arm/window at a time rather
than persisted.  SGEYESUB has no clean target, so this module never emits
clean-waveform RRMSE or a confirmation claim.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from eeg_cgdr.data.sgeyesub import (
    SgeyesubLoadedRecord,
    SgeyesubReleaseRecord,
    load_sgeyesub_signal_record,
    load_sgeyesub_structure_audit,
)
from eeg_cgdr.experiments.sgeyesub_operator_specificity import _evaluate_output
from eeg_cgdr.experiments.subject_artifact_data import (
    HeldoutSubjectArtifactRecord,
    PreparedSubjectArtifactFold,
    RuntimeArtifactContext,
    prepare_subject_artifact_fold,
)
from eeg_cgdr.experiments.subject_artifact_development_train import (
    SelectedValidityRepair,
    load_selected_validity_repair,
)
from eeg_cgdr.experiments.subject_artifact_training import (
    load_artifact_training_checkpoint,
)
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)
from eeg_cgdr.models.artifact_latent_inference import (
    ArtifactInferenceContext,
    PopulationSubjectRestoration,
    deterministic_population_subject_restore,
    diffusion_population_subject_restore,
)


PROTOCOL_ID = "subject_calibrated_artifact_latent_diffusion_development_v1"
MODEL_IDS = ("deterministic", "diffusion")
CONTEXT_IDS = ("population", "matching", "wrong", "shuffled")
CHECKPOINT_ENDPOINTS = ("best", "equal")
FACTORIAL_ARM_IDS = tuple(
    f"{model_id}__{context_id}"
    for model_id in MODEL_IDS
    for context_id in CONTEXT_IDS
)
EVALUATION_ARM_IDS = tuple(
    f"{endpoint}__{arm_id}"
    for endpoint in CHECKPOINT_ENDPOINTS
    for arm_id in FACTORIAL_ARM_IDS
)
TASK_COUNT = 75
BLOCKED_STEM = "study05/study05_p42"
_PERFORMANCE_FIELDS = (
    "matching_projector_attenuation_db",
    "population_projector_attenuation_db",
    "nonartifact_observation_preservation",
    "eog_coherence_raw",
    "eog_coherence_output",
    "eog_coherence_reduction",
    "reference_free_psd_distortion",
    "reference_free_covariance_distortion",
    "heldout_eog_prediction_remaining_ratio",
    "condition_erp_observation_relative_preservation",
    "observation_change_ratio",
    "low_artifact_output_input_rms_ratio",
    "low_artifact_relative_observation_change",
)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("natural-SGE evaluation produced no metric rows")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


@dataclass(frozen=True)
class EvaluationTask:
    task_index: int
    unified_fold_index: int
    training_seed: int
    seed_ordinal: int


def evaluation_task(
    config: Mapping[str, Any], task_index: int
) -> EvaluationTask:
    """Map the frozen 0..74 array to 25 folds x three seeds."""

    if isinstance(task_index, bool) or not 0 <= int(task_index) < TASK_COUNT:
        raise ValueError("evaluation task index must lie in [0,74]")
    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("natural-SGE factorial requires exactly three training seeds")
    index = int(task_index)
    return EvaluationTask(
        task_index=index,
        unified_fold_index=index // len(seeds),
        training_seed=seeds[index % len(seeds)],
        seed_ordinal=index % len(seeds),
    )


def subject_artifact_checkpoint_path(
    config: Mapping[str, Any],
    *,
    unified_fold_index: int,
    training_seed: int,
    model_kind: str,
    checkpoint_endpoint: str = "best",
) -> Path:
    """Stable J3/J4 handoff for best and equal-update EMA endpoints."""

    if model_kind not in MODEL_IDS:
        raise ValueError("unknown subject-artifact model kind")
    if not 0 <= int(unified_fold_index) < 25:
        raise ValueError("checkpoint fold index must lie in [0,24]")
    if checkpoint_endpoint not in CHECKPOINT_ENDPOINTS:
        raise ValueError("checkpoint endpoint must be best or equal")
    root = Path(str(_mapping(config, "outputs")["checkpoint_root"]))
    return (
        root
        / f"fold_{int(unified_fold_index):02d}"
        / f"seed_{int(training_seed)}"
        / model_kind
        / f"{checkpoint_endpoint}.pt"
    )


def subject_artifact_checkpoint_contract(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    *,
    training_seed: int,
    model_kind: str,
    implementation: str,
    checkpoint_endpoint: str = "best",
    selected_repair: SelectedValidityRepair | None = None,
) -> dict[str, Any]:
    """Minimal subset that each requested J3 endpoint must expose to J4."""

    if model_kind not in MODEL_IDS:
        raise ValueError("unknown subject-artifact model kind")
    endpoint_contract = {
        "best": "development_validation_best",
        "equal": "equal_update_8000",
    }
    if checkpoint_endpoint not in endpoint_contract:
        raise ValueError("checkpoint endpoint must be best or equal")
    dimensions = prepared.model_dimensions
    result = {
        "protocol_id": str(config["protocol_id"]),
        "unified_fold_index": int(prepared.fold.unified_fold_index),
        "fold_id": prepared.fold.fold_id,
        "training_seed": int(training_seed),
        "model_kind": model_kind,
        "implementation": str(implementation),
        "endpoint": endpoint_contract[checkpoint_endpoint],
        "eeg_channels": int(dimensions.eeg_channels),
        "eog_coordinates": int(dimensions.eog_coordinates),
        "signal_length": int(dimensions.signal_length),
    }
    if selected_repair is not None:
        result.update(
            {
                "validity_execution_revision": selected_repair.execution_revision,
                "validity_selected_implementation": selected_repair.implementation,
                "identity_repair_active": selected_repair.identity_repair_active,
                "effective_standardized_latent_absolute_clip": (
                    selected_repair.standardized_latent_absolute_clip
                ),
                "effective_inference_sampler": selected_repair.inference_sampler,
            }
        )
    return result


def _model_config(
    config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold
) -> ArtifactLatentModelConfig:
    raw = _mapping(config, "model")
    dimensions = prepared.model_dimensions
    return ArtifactLatentModelConfig(
        eeg_channels=dimensions.eeg_channels,
        signal_length=dimensions.signal_length,
        latent_channels=dimensions.eog_coordinates,
        base_channels=int(raw["base_channels"]),
        channel_mults=tuple(int(value) for value in raw["channel_mults"]),  # type: ignore[arg-type]
        num_res_blocks=int(raw["num_res_blocks"]),
        groupnorm_groups=int(raw["groupnorm_groups"]),
        dropout=float(raw["dropout"]),
        time_sinusoidal_dim=int(raw["time_sinusoidal_dim"]),
        time_embed_dim=int(raw["time_embed_dim"]),
        attention_length=int(raw.get("attention_length", 64)),
        attention_heads=int(raw["attention_heads"]),
    )


def _build_model(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    model_kind: str,
    selected_repair: SelectedValidityRepair,
) -> nn.Module:
    backbone = _model_config(config, prepared)
    if model_kind == "deterministic":
        return DeterministicArtifactEstimator(backbone)
    if model_kind != "diffusion":
        raise ValueError("unknown subject-artifact model kind")
    raw = _mapping(config, "primary_diffusion")
    return ArtifactLatentDiffusion(
        backbone,
        ArtifactLatentDiffusionConfig(
            num_timesteps=int(raw["timesteps"]),
            cosine_offset=float(raw["cosine_offset"]),
            prediction_target=str(raw["prediction_target"]),
            min_snr_gamma=float(raw["min_snr_gamma"]),
            dynamic_threshold_quantile=float(raw["dynamic_threshold_quantile"]),
            standardized_latent_absolute_clip=float(
                selected_repair.standardized_latent_absolute_clip
            ),
            posterior_samples=int(
                _mapping(config, "artifact_latent")["posterior_samples"]
            ),
        ),
    )


def _load_ema_model(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    *,
    training_seed: int,
    model_kind: str,
    device: torch.device,
    implementation: str,
    checkpoint_endpoint: str = "best",
    selected_repair: SelectedValidityRepair,
) -> tuple[nn.Module, Path, Mapping[str, Any]]:
    path = subject_artifact_checkpoint_path(
        config,
        unified_fold_index=prepared.fold.unified_fold_index,
        training_seed=training_seed,
        model_kind=model_kind,
        checkpoint_endpoint=checkpoint_endpoint,
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing requested J4 checkpoint endpoint: {path}")
    payload = load_artifact_training_checkpoint(path, map_location=device)
    expected = subject_artifact_checkpoint_contract(
        config,
        prepared,
        training_seed=training_seed,
        model_kind=model_kind,
        implementation=implementation,
        checkpoint_endpoint=checkpoint_endpoint,
        selected_repair=selected_repair,
    )
    saved_contract = payload.get("config")
    if not isinstance(saved_contract, Mapping) or any(
        saved_contract.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("checkpoint does not match the frozen fold/seed/model contract")
    model = _build_model(config, prepared, model_kind, selected_repair).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    extra = payload["extra"]
    expected_role = {
        "best": "development_validation_best",
        "equal": "equal_update",
    }[checkpoint_endpoint]
    if not isinstance(extra, Mapping) or extra.get("checkpoint_role") != expected_role:
        raise ValueError("J4 checkpoint role differs from its requested endpoint")
    if checkpoint_endpoint == "equal" and int(payload.get("step", -1)) != int(
        _mapping(config, "training")["equal_compute_updates"]
    ):
        raise ValueError("equal-update checkpoint is not at the frozen update count")
    ema_state = extra.get("ema_state") if isinstance(extra, Mapping) else None
    shadow = ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
    if not isinstance(shadow, Mapping):
        raise ValueError("requested checkpoint is missing EMA shadow weights")
    state = model.state_dict()
    expected_shadow = {
        name for name, value in state.items() if value.dtype.is_floating_point
    }
    if set(shadow) != expected_shadow:
        raise ValueError("EMA checkpoint parameter names differ from the model")
    for name, saved in shadow.items():
        current = state[name]
        value = torch.as_tensor(saved, device=current.device, dtype=current.dtype)
        if value.shape != current.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"EMA tensor is invalid: {name}")
        current.copy_(value)
    model.eval()
    return model, path, payload


def _context_basis(value: RuntimeArtifactContext) -> np.ndarray:
    left, _, _ = np.linalg.svd(
        np.asarray(value.full_transfer, dtype=np.float64),
        full_matrices=False,
    )
    basis = np.ascontiguousarray(left[:, : value.rank])
    if not np.allclose(
        basis @ basis.T,
        value.projector,
        rtol=0.0,
        atol=2.0e-9,
    ):
        raise ValueError("runtime transfer and projector do not share one subspace")
    return basis


def _inference_context(
    value: RuntimeArtifactContext,
    prepared: PreparedSubjectArtifactFold,
    *,
    role: Literal["population", "subject"],
    device: torch.device,
    calibration_duration_seconds: float | None = None,
) -> ArtifactInferenceContext:
    # ArtifactInferenceContext verifies projector geometry at a substantially
    # tighter tolerance than the neural float32 path.  Keep the immutable
    # context in FP64; each model call explicitly casts it to the observation.
    floating = torch.float64
    return ArtifactInferenceContext(
        context_id=value.context_id,
        role=role,
        full_transfer=torch.as_tensor(
            np.array(value.full_transfer, copy=True), dtype=floating, device=device
        ),
        normalized_transfer=torch.as_tensor(
            np.array(value.normalized_transfer, copy=True),
            dtype=floating,
            device=device,
        ),
        transfer_scale=torch.as_tensor(
            np.array(value.transfer_scale, copy=True), dtype=floating, device=device
        ),
        singular_values=torch.as_tensor(
            np.array(value.singular_values, copy=True), dtype=floating, device=device
        ),
        rank=int(value.rank),
        calibration_duration_seconds=(
            float(value.calibration_duration_seconds)
            if calibration_duration_seconds is None
            else float(calibration_duration_seconds)
        ),
        latent_mean=torch.as_tensor(
            np.array(prepared.latent_normalizer.mean, copy=True),
            dtype=floating,
            device=device,
        ),
        latent_standard_deviation=torch.as_tensor(
            np.array(prepared.latent_normalizer.standard_deviation, copy=True),
            dtype=floating,
            device=device,
        ),
        subspace_basis=torch.as_tensor(
            _context_basis(value), dtype=floating, device=device
        ),
    )


@dataclass(frozen=True)
class FactorialContext:
    context_id: str
    subject: RuntimeArtifactContext
    rho: float
    calibration_duration_seconds: float
    is_formal_population_arm: bool


def factorial_context_plan(
    population: RuntimeArtifactContext,
    heldout: HeldoutSubjectArtifactRecord,
) -> tuple[FactorialContext, ...]:
    """Only C changes; every accepted factorial arm retains the stem's rho."""

    rho = float(heldout.matching.rho)
    duration = (
        0.0 if rho == 0.0 else float(heldout.matching.calibration_duration_seconds)
    )
    candidates = (
        FactorialContext("population", population, rho, duration, True),
        FactorialContext("matching", heldout.matching, rho, duration, False),
        FactorialContext("wrong", heldout.wrong_same_cell, rho, duration, False),
        FactorialContext("shuffled", heldout.shuffled_same_cell, rho, duration, False),
    )
    shapes = {tuple(item.subject.full_transfer.shape) for item in candidates}
    if len(shapes) != 1:
        raise ValueError("factorial context crossed an incompatible channel/EOG cell")
    if any(
        not math.isclose(item.rho, rho, rel_tol=0.0, abs_tol=0.0)
        for item in candidates
    ):
        raise AssertionError("factorial context changed the original stem rho")
    if any(
        not math.isclose(
            item.calibration_duration_seconds,
            duration,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for item in candidates
    ):
        raise AssertionError("factorial context changed calibration duration")
    if heldout.wrong_source_recording_key not in population.fit_recording_keys:
        raise ValueError("wrong context is not from this exact-cell training fold")
    return candidates


def _sample_seed_tuple(
    *, training_seed: int, fold_index: int, recording_key: str, batch_index: int
) -> tuple[int, ...]:
    token = sum((index + 1) * ord(value) for index, value in enumerate(recording_key))
    base = (
        int(training_seed)
        + 100_003 * int(fold_index)
        + 1_009 * int(batch_index)
        + token
    )
    return tuple(base + 17 * index for index in range(8))


@dataclass(frozen=True)
class ArmInference:
    windowed_output: np.ndarray
    status: str
    latency_seconds: float
    peak_memory_mb: float
    posterior_standardized_latent_sd_rms: float
    network_calls: int
    complement_or_union_relative_error: float
    branch: str
    population_standardized_latent_samples: np.ndarray | None = None
    subject_standardized_latent_samples: np.ndarray | None = None
    sample_seeds_by_batch: tuple[tuple[int, ...], ...] = ()
    posterior_output_sample_sd_rms: float = float("nan")


def _failed_arm(observed: np.ndarray, error: BaseException) -> ArmInference:
    """Preserve a failed arm in coverage without treating identity as science."""

    message = str(error).lower()
    if "out of memory" in message:
        category = "failed_inference_cuda_oom"
    elif "nan" in message or "inf" in message or isinstance(error, FloatingPointError):
        category = "failed_inference_nonfinite"
    else:
        category = "failed_inference_runtime_or_contract"
    return ArmInference(
        windowed_output=np.asarray(observed, dtype=np.float32).copy(),
        status=category,
        latency_seconds=float("nan"),
        peak_memory_mb=float("nan"),
        posterior_standardized_latent_sd_rms=float("nan"),
        network_calls=0,
        complement_or_union_relative_error=float("nan"),
        branch="identity_placeholder_excluded_from_performance",
    )


def _posterior_uncertainty(result: PopulationSubjectRestoration) -> float:
    """Conservative branch-SD bound when cross-branch covariance is not retained."""

    values: list[Tensor] = []
    weights: list[float] = []
    if result.population.stochastic_standard_deviation is not None:
        values.append(result.population.stochastic_standard_deviation)
        weights.append(1.0 if result.subject is None else 1.0 - result.rho)
    if result.subject is not None and result.subject.stochastic_standard_deviation is not None:
        values.append(result.subject.stochastic_standard_deviation)
        weights.append(result.rho)
    if not values:
        return float("nan")
    weighted_rms = sum(
        float(weight) * torch.sqrt(value.square().mean())
        for weight, value in zip(weights, values)
    )
    return float(weighted_rms.detach().cpu())


@torch.no_grad()
def _infer_arm(
    model: nn.Module,
    *,
    model_kind: str,
    prepared: PreparedSubjectArtifactFold,
    heldout: HeldoutSubjectArtifactRecord,
    context: FactorialContext,
    training_seed: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> ArmInference:
    windows = heldout.query.observed
    valid = heldout.query.valid_time_mask
    batch_size = int(_mapping(config, "training")["batch_size"])
    shared_duration = float(context.calibration_duration_seconds)
    population = _inference_context(
        prepared.population_context,
        prepared,
        role="population",
        device=device,
        calibration_duration_seconds=shared_duration,
    )
    outputs: list[np.ndarray] = []
    uncertainty: list[float] = []
    population_latent_samples: list[np.ndarray] = []
    subject_latent_samples: list[np.ndarray] = []
    sample_seeds_by_batch: list[tuple[int, ...]] = []
    output_sample_variance_sum = 0.0
    output_sample_variance_count = 0
    errors: list[float] = []
    network_calls = 0
    branches: set[str] = set()
    if device.type == "cuda":
        cuda_device_index = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_device_index)
        torch.cuda.synchronize(cuda_device_index)
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, windows.shape[0], batch_size)):
        stop = min(start + batch_size, windows.shape[0])
        observed = torch.as_tensor(
            windows[start:stop], dtype=torch.float32, device=device
        )
        time_mask = torch.as_tensor(valid[start:stop], dtype=torch.bool, device=device)
        channel_mask = torch.ones(
            observed.shape[0], observed.shape[1], dtype=torch.bool, device=device
        )

        # A rejected calibration (rho=0) is the only true POP short-circuit.
        # The accepted population factorial arm holds rho_s fixed and supplies
        # C0 as the subject-role context, so Delta=(1-rho)Delta0+rhoDelta0.
        if context.rho == 0.0:
            factory: Callable[[], ArtifactInferenceContext] | None = None
        else:
            # Construction itself is lazy so rejected calibration cannot even
            # validate or materialize a personalized context before POP.
            factory = lambda: _inference_context(
                context.subject,
                prepared,
                role="subject",
                device=device,
                calibration_duration_seconds=shared_duration,
            )
        if model_kind == "deterministic":
            if not isinstance(model, DeterministicArtifactEstimator):
                raise TypeError("deterministic arm received the wrong model")
            result = deterministic_population_subject_restore(
                model,
                observed,
                population_context=population,
                rho=context.rho,
                subject_context_factory=factory,
                channel_mask=channel_mask,
                valid_time_mask=time_mask,
            )
        elif model_kind == "diffusion":
            if not isinstance(model, ArtifactLatentDiffusion):
                raise TypeError("diffusion arm received the wrong model")
            batch_sample_seeds = _sample_seed_tuple(
                training_seed=training_seed,
                fold_index=prepared.fold.unified_fold_index,
                recording_key=heldout.recording_key,
                batch_index=batch_index,
            )
            result = diffusion_population_subject_restore(
                model,
                observed,
                population_context=population,
                rho=context.rho,
                subject_context_factory=factory,
                channel_mask=channel_mask,
                valid_time_mask=time_mask,
                sample_seeds=batch_sample_seeds,
                ddim_steps=int(_mapping(config, "primary_diffusion")["ddim_steps"]),
                record_trajectory=False,
            )
            if (
                result.population.standardized_latent_samples is None
                or result.mixed_delta_samples is None
                or result.restored_samples is None
            ):
                raise ValueError("J4 diffusion requires retained explicit K=8 samples")
            population_latent_samples.append(
                result.population.standardized_latent_samples.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if result.subject is not None:
                if result.subject.standardized_latent_samples is None:
                    raise ValueError("J4 subject branch discarded its K=8 latent samples")
                subject_latent_samples.append(
                    result.subject.standardized_latent_samples.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            sample_variance = result.restored_samples.var(dim=0, unbiased=False)
            valid_output = time_mask[:, None, :].to(sample_variance.dtype)
            output_sample_variance_sum += float(
                (sample_variance * valid_output).sum().detach().cpu()
            )
            output_sample_variance_count += int(time_mask.sum().item()) * int(
                observed.shape[1]
            )
            sample_seeds_by_batch.append(batch_sample_seeds)
        else:
            raise ValueError("unknown subject-artifact model kind")
        outputs.append(result.restored.detach().cpu().numpy().astype(np.float32))
        uncertainty.append(_posterior_uncertainty(result))
        errors.append(float(result.complement_relative_error))
        network_calls += result.population.network_calls
        if result.subject is not None:
            network_calls += result.subject.network_calls
        branches.add(result.branch)
    if device.type == "cuda":
        cuda_device_index = torch.cuda.current_device()
        torch.cuda.synchronize(cuda_device_index)
        peak = float(
            torch.cuda.max_memory_allocated(cuda_device_index) / (1024.0**2)
        )
    else:
        peak = 0.0
    elapsed = time.perf_counter() - started
    if len(branches) != 1:
        raise AssertionError("one arm changed inference branch across query batches")
    result_output = np.concatenate(outputs, axis=0)
    if result_output.shape != windows.shape or not np.isfinite(result_output).all():
        raise FloatingPointError("learned artifact arm produced an invalid output")
    finite_uncertainty = [value for value in uncertainty if math.isfinite(value)]
    retained_population = (
        np.concatenate(population_latent_samples, axis=1)
        if population_latent_samples
        else None
    )
    retained_subject = (
        np.concatenate(subject_latent_samples, axis=1)
        if subject_latent_samples
        else None
    )
    if model_kind == "diffusion":
        expected_latent_shape = (
            8,
            windows.shape[0],
            prepared.model_dimensions.eog_coordinates,
            windows.shape[2],
        )
        if (
            retained_population is None
            or retained_population.shape != expected_latent_shape
        ):
            raise ValueError("J4 population latent samples lost K/window/EOG/time alignment")
        if (
            context.rho != 0.0
            and (
                retained_subject is None
                or retained_subject.shape != expected_latent_shape
            )
        ):
            raise ValueError("J4 subject latent samples lost K/window/EOG/time alignment")
        if context.rho == 0.0 and retained_subject is not None:
            raise ValueError("rho=0 retained forbidden personalized latent samples")
        if output_sample_variance_count < 1:
            raise ValueError("J4 diffusion found no valid posterior output samples")
        output_sample_sd_rms = float(
            math.sqrt(output_sample_variance_sum / output_sample_variance_count)
        )
    else:
        output_sample_sd_rms = float("nan")
    return ArmInference(
        windowed_output=result_output,
        status=(
            "success_population_fallback_rho_zero"
            if context.rho == 0.0
            else "success"
        ),
        latency_seconds=elapsed,
        peak_memory_mb=peak,
        posterior_standardized_latent_sd_rms=(
            float(np.mean(finite_uncertainty))
            if finite_uncertainty
            else float("nan")
        ),
        network_calls=network_calls,
        complement_or_union_relative_error=max(errors, default=float("nan")),
        branch=next(iter(branches)),
        population_standardized_latent_samples=retained_population,
        subject_standardized_latent_samples=retained_subject,
        sample_seeds_by_batch=tuple(sample_seeds_by_batch),
        posterior_output_sample_sd_rms=output_sample_sd_rms,
    )


def _continuous(windows: np.ndarray) -> np.ndarray:
    value = np.asarray(windows)
    if value.ndim != 3 or not np.issubdtype(value.dtype, np.floating):
        raise ValueError("windowed EEG must have shape (N,C,L)")
    return np.ascontiguousarray(value.transpose(1, 0, 2).reshape(value.shape[1], -1))


@dataclass(frozen=True)
class FrozenFactorialOutputs:
    recording_key: str
    outputs: Mapping[str, np.ndarray]
    freeze_manifest_path: Path
    output_archive_path: Path
    query_evaluation_fields_opened: Literal[False] = False
    arm_ids: tuple[str, ...] = FACTORIAL_ARM_IDS
    posterior_samples: Mapping[str, np.ndarray] | None = None
    posterior_metadata: Mapping[str, Mapping[str, Any]] | None = None
    point_output_bytes: int = 0
    posterior_latent_bytes: int = 0
    archive_disk_bytes: int = 0

    def __post_init__(self) -> None:
        if set(self.outputs) != set(self.arm_ids):
            raise ValueError("factorial outputs differ from the declared frozen arms")
        if self.query_evaluation_fields_opened:
            raise ValueError("query annotations opened before factorial output freeze")
        if not self.freeze_manifest_path.is_file():
            raise ValueError("atomic output-freeze manifest was not published")
        if not self.output_archive_path.is_file():
            raise ValueError("factorial waveform archive was not published")


def freeze_factorial_outputs(
    outputs: Mapping[str, np.ndarray],
    *,
    recording_key: str,
    manifest_path: Path,
    expected_arm_ids: Sequence[str] = FACTORIAL_ARM_IDS,
    posterior_samples: Mapping[str, np.ndarray] | None = None,
    required_posterior_sample_keys: Sequence[str] | None = None,
    posterior_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> FrozenFactorialOutputs:
    """Freeze float32 point outputs and low-dimensional K=8 latents before scoring."""

    arm_ids = tuple(str(value) for value in expected_arm_ids)
    if len(arm_ids) != len(set(arm_ids)) or set(outputs) != set(arm_ids):
        raise ValueError("factorial output freeze differs from declared arms")
    frozen: dict[str, np.ndarray] = {}
    shape: tuple[int, int] | None = None
    for arm_id in arm_ids:
        value = np.array(outputs[arm_id], dtype=np.float32, copy=True, order="C")
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"invalid factorial output: {arm_id}")
        if shape is None:
            shape = value.shape
        elif shape != value.shape:
            raise ValueError("factorial output waveforms are not aligned")
        value.setflags(write=False)
        frozen[arm_id] = value
    frozen_samples: dict[str, np.ndarray] = {}
    for sample_id, raw in (posterior_samples or {}).items():
        value = np.array(raw, dtype=np.float32, copy=True, order="C")
        if not sample_id.endswith("standardized_latent_samples"):
            raise ValueError("only low-dimensional standardized latents may freeze")
        if value.ndim != 4 or value.shape[0] != 8 or not np.isfinite(value).all():
            raise ValueError(f"invalid explicit-K posterior sample array: {sample_id}")
        if shape is None or value.shape[1] * value.shape[3] != shape[1]:
            raise ValueError(f"posterior sample time axis is misaligned: {sample_id}")
        value.setflags(write=False)
        frozen_samples[str(sample_id)] = value
    required = (
        None
        if required_posterior_sample_keys is None
        else {str(value) for value in required_posterior_sample_keys}
    )
    if required is not None and set(frozen_samples) != required:
        raise ValueError("frozen posterior latent keys differ from successful diffusion arms")
    frozen_metadata = {
        str(key): MappingProxyType(dict(value))
        for key, value in (posterior_metadata or {}).items()
    }
    if set(frozen_metadata) != {
        key.rsplit("__", 1)[0] for key in frozen_samples
    }:
        raise ValueError("posterior reconstruction metadata differs from latent arms")
    point_output_bytes = sum(value.nbytes for value in frozen.values())
    posterior_latent_bytes = sum(value.nbytes for value in frozen_samples.values())
    archive_path = manifest_path.with_suffix(".npz")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.tmp")
    with temporary_archive.open("wb") as stream:
        np.savez_compressed(
            stream,
            **frozen,
            **{f"posterior__{key}": value for key, value in frozen_samples.items()},
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary_archive.replace(archive_path)
    archive_disk_bytes = int(archive_path.stat().st_size)
    manifest = {
        "recording_key": recording_key,
        "status": (
            "all_eight_factorial_outputs_frozen_before_scoring"
            if arm_ids == FACTORIAL_ARM_IDS
            else "all_endpoint_factorial_outputs_frozen_before_scoring"
        ),
        "arm_ids": list(arm_ids),
        "output_shape": list(shape or ()),
        "point_output_dtype": "float32",
        "query_evaluation_fields_opened": False,
        "point_waveforms_persisted": True,
        "waveform_archive": str(archive_path),
        "posterior_sample_keys": sorted(frozen_samples),
        "posterior_sample_count_K": 8 if frozen_samples else 0,
        "posterior_storage": "standardized_latent_only_no_EEG_correction_or_restored_samples",
        "posterior_reconstruction_metadata": {
            key: dict(value) for key, value in frozen_metadata.items()
        },
        "posterior_point_rule": "strict_arithmetic_mean_no_best_of_K",
        "query_eog_or_labels_used_for_inference": False,
        "estimated_uncompressed_point_output_bytes": point_output_bytes,
        "estimated_uncompressed_posterior_latent_bytes": posterior_latent_bytes,
        "estimated_uncompressed_total_bytes": (
            point_output_bytes + posterior_latent_bytes
        ),
        "actual_compressed_archive_bytes": archive_disk_bytes,
    }
    _atomic_json(manifest_path, manifest)
    return FrozenFactorialOutputs(
        recording_key=recording_key,
        outputs=MappingProxyType(frozen),
        freeze_manifest_path=manifest_path,
        output_archive_path=archive_path,
        arm_ids=arm_ids,
        posterior_samples=MappingProxyType(frozen_samples),
        posterior_metadata=MappingProxyType(frozen_metadata),
        point_output_bytes=point_output_bytes,
        posterior_latent_bytes=posterior_latent_bytes,
        archive_disk_bytes=archive_disk_bytes,
    )


AnnotationOpener = Callable[[], SgeyesubLoadedRecord]


def open_annotations_after_freeze(
    frozen: FrozenFactorialOutputs,
    opener: AnnotationOpener,
) -> SgeyesubLoadedRecord:
    """The only evaluation-only query annotation opening boundary."""

    if (
        frozen.query_evaluation_fields_opened
        or not frozen.freeze_manifest_path.is_file()
        or not frozen.output_archive_path.is_file()
    ):
        raise AssertionError("factorial outputs were not frozen before annotation access")
    annotated = opener()
    if annotated.query is None or annotated.query_annotations is None:
        raise AssertionError("evaluation-only query fields were not opened")
    return annotated


def _structure_maps(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, SgeyesubReleaseRecord]]:
    sge = _mapping(_mapping(config, "data"), "sgeyesub")
    layouts, records = load_sgeyesub_structure_audit(Path(str(sge["structure_audit"])))
    return (
        {value.layout_id: value for value in layouts},
        {value.recording_key: value for value in records},
    )


def _prediction_from_query_eog(
    config: Mapping[str, Any],
    heldout: HeldoutSubjectArtifactRecord,
    annotated: SgeyesubLoadedRecord,
) -> np.ndarray:
    """Evaluation-only C E prediction using support-frozen EOG statistics."""

    annotations = annotated.query_annotations
    if annotations is None:
        raise AssertionError("query EOG was not opened for scoring")
    samples = int(
        round(
            float(_mapping(config, "calibration")["calibration_duration_seconds"])
            * float(annotated.sampling_rate_hz)
        )
    )
    support = np.asarray(annotated.support.external_eog[:, :samples], dtype=np.float64)
    mean = support.mean(axis=1, keepdims=True)
    scale = support.std(axis=1, keepdims=True)
    if support.shape[1] != samples or np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("support-frozen EOG scoring coordinate is unavailable")
    standardized = (
        np.asarray(annotations.external_eog, dtype=np.float64) - mean
    ) / scale
    return np.asarray(heldout.matching.full_transfer @ standardized, dtype=np.float64)


def _scale_metrics(
    output_windows: np.ndarray,
    observed_windows: np.ndarray,
    *,
    maximum_ratio: float,
) -> dict[str, Any]:
    output = np.asarray(output_windows, dtype=np.float64)
    observed = np.asarray(observed_windows, dtype=np.float64)
    if output.shape != observed.shape or output.ndim != 3:
        raise ValueError("scale audit requires aligned windowed EEG")
    axes = (1, 2)
    output_rms = np.sqrt(np.mean(output * output, axis=axes))
    input_rms = np.sqrt(np.mean(observed * observed, axis=axes))
    ratio = output_rms / np.maximum(input_rms, np.finfo(np.float64).eps)
    change = np.linalg.norm(output - observed) / max(
        float(np.linalg.norm(observed)), np.finfo(np.float64).eps
    )
    output_continuous = _continuous(output)
    input_continuous = _continuous(observed)
    input_variance = np.var(input_continuous, axis=1)
    variance_ratio = np.var(output_continuous, axis=1) / np.maximum(
        input_variance, np.finfo(np.float64).eps
    )
    return {
        "all_output_values_finite": bool(np.isfinite(output).all()),
        "output_input_rms_ratio_mean": float(np.mean(ratio)),
        "output_input_rms_ratio_median": float(np.median(ratio)),
        "output_input_rms_ratio_maximum": float(np.max(ratio)),
        "channelwise_variance_ratio_median": float(np.median(variance_ratio)),
        "relative_observation_change": float(change),
        "output_maximum_absolute_value": float(np.max(np.abs(output))),
        "absolute_scale_safety_passed": bool(
            np.isfinite(output).all() and np.all(ratio <= maximum_ratio)
        ),
    }


def _low_artifact_metrics(
    output: np.ndarray, observed: np.ndarray, artifactclasses: np.ndarray
) -> dict[str, float]:
    rest = np.asarray(artifactclasses == 6, dtype=bool)
    if not np.any(rest):
        return {
            "low_artifact_output_input_rms_ratio": float("nan"),
            "low_artifact_relative_observation_change": float("nan"),
        }
    denominator = max(
        float(np.linalg.norm(observed[:, rest])), np.finfo(np.float64).eps
    )
    return {
        "low_artifact_output_input_rms_ratio": float(
            np.linalg.norm(output[:, rest]) / denominator
        ),
        "low_artifact_relative_observation_change": float(
            np.linalg.norm(output[:, rest] - observed[:, rest]) / denominator
        ),
    }


def _full_v0_scale_validity(
    config: Mapping[str, Any],
    output_windows: np.ndarray,
    observed_windows: np.ndarray,
    valid_time_mask: np.ndarray,
    artifactclasses: np.ndarray,
    *,
    span_consistency_relative_error: float,
    retained_samples_finite: bool,
) -> dict[str, Any]:
    """Apply every frozen V0 scale/safety rule to one full J4 arm."""

    output = np.asarray(output_windows, dtype=np.float64)
    observed = np.asarray(observed_windows, dtype=np.float64)
    valid = np.asarray(valid_time_mask, dtype=bool)
    if output.shape != observed.shape or output.ndim != 3:
        raise ValueError("full V0 requires aligned (N,C,T) EEG windows")
    if valid.shape != (output.shape[0], output.shape[2]):
        raise ValueError("full V0 valid-time mask differs from EEG windows")
    labels = np.asarray(artifactclasses).reshape(-1)
    if labels.size != output.shape[0] * output.shape[2]:
        raise ValueError("full V0 artifact labels do not align with frozen windows")
    low = labels.reshape(output.shape[0], output.shape[2]) == 6
    low &= valid
    channel_count = output.shape[1]
    epsilon = np.finfo(np.float64).eps

    def _ratios(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        usable = mask.sum(axis=1) > 0
        if not np.any(usable):
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        weight = mask[:, None, :].astype(np.float64)
        count = np.maximum(mask.sum(axis=1) * channel_count, 1)
        input_rms = np.sqrt((np.square(observed) * weight).sum(axis=(1, 2)) / count)
        output_rms = np.sqrt((np.square(output) * weight).sum(axis=(1, 2)) / count)
        difference_rms = np.sqrt(
            (np.square(output - observed) * weight).sum(axis=(1, 2)) / count
        )
        return (
            output_rms[usable] / np.maximum(input_rms[usable], epsilon),
            difference_rms[usable] / np.maximum(input_rms[usable], epsilon),
        )

    full_ratio, _ = _ratios(valid)
    low_ratio, low_change = _ratios(low)
    v0 = _mapping(_mapping(config, "validity"), "V0")
    maximum = float(v0["maximum_per_window_output_input_RMS_ratio"])
    full_bounds = tuple(
        float(value) for value in v0["full_median_output_input_RMS_ratio"]
    )
    low_bounds = tuple(
        float(value) for value in v0["low_artifact_median_output_input_RMS_ratio"]
    )
    if len(full_bounds) != 2 or len(low_bounds) != 2:
        raise ValueError("V0 RMS bounds must contain exactly [minimum, maximum]")
    low_change_maximum = float(
        v0["low_artifact_maximum_median_relative_observation_change"]
    )
    span_tolerance = min(
        float(v0["pure_operator_maximum_complement_consistency_relative_error"]),
        float(v0["pure_operator_maximum_union_span_consistency_relative_error"]),
    )
    full_median = float(np.median(full_ratio)) if full_ratio.size else float("nan")
    low_median = float(np.median(low_ratio)) if low_ratio.size else float("nan")
    low_change_median = (
        float(np.median(low_change)) if low_change.size else float("nan")
    )
    checks = {
        "all_finite": bool(
            np.isfinite(output).all()
            and retained_samples_finite
            and math.isfinite(float(span_consistency_relative_error))
        ),
        "per_window_RMS_ratio_at_most_10": bool(
            full_ratio.size
            and np.isfinite(full_ratio).all()
            and np.all(full_ratio <= maximum)
        ),
        "full_median_RMS_ratio_in_bounds": bool(
            math.isfinite(full_median) and full_bounds[0] <= full_median <= full_bounds[1]
        ),
        "low_artifact_median_RMS_ratio_in_bounds": bool(
            math.isfinite(low_median) and low_bounds[0] <= low_median <= low_bounds[1]
        ),
        "low_artifact_median_relative_change_at_most_0_20": bool(
            math.isfinite(low_change_median)
            and low_change_median <= low_change_maximum
        ),
        "union_or_complement_span_consistency": bool(
            math.isfinite(float(span_consistency_relative_error))
            and float(span_consistency_relative_error) <= span_tolerance
        ),
    }
    return {
        "full_V0_scale_validity_passed": all(checks.values()),
        "full_V0_checks": checks,
        "retained_posterior_samples_finite": bool(retained_samples_finite),
        "full_valid_window_count": int(full_ratio.size),
        "full_output_input_RMS_ratio_median": full_median,
        "low_artifact_valid_window_count": int(low_ratio.size),
        "low_artifact_output_input_RMS_ratio_median": low_median,
        "low_artifact_relative_observation_change_median": low_change_median,
        "span_consistency_tolerance": span_tolerance,
    }


def _performance_values_eligible(
    status: str, scale_metrics: Mapping[str, Any]
) -> bool:
    """Only a true successful, scale-safe learned arm contributes performance."""

    return bool(
        status == "success"
        and scale_metrics.get("full_V0_scale_validity_passed") is True
    )


def _arm_operator_source(context_id: str) -> str:
    return {
        "population": "same_cell_outer_training_population_C0",
        "matching": "heldout_stem_block1_matching_Cs",
        "wrong": "same_cell_outer_training_wrong_Cw",
        "shuffled": "heldout_block1_severity_stratified_shuffled_Cpi",
    }[context_id]


def _context_provenance(
    context: FactorialContext,
    heldout: HeldoutSubjectArtifactRecord,
) -> dict[str, Any]:
    """Record the actual precomputed context draw, not only its arm label."""

    common = {
        "operator_context_id": context.subject.context_id,
        "operator_fit_recording_keys": "|".join(context.subject.fit_recording_keys),
        "operator_context_role": context.subject.role,
    }
    if context.context_id == "wrong":
        return {
            **common,
            "wrong_context_source_recording_key": heldout.wrong_source_recording_key,
            "wrong_context_draw_id": heldout.wrong_source_recording_key,
            "wrong_context_draw_rule": "lexicographically_first_same_cell_outer_training_stem",
            "shuffled_context_source_recording_key": None,
            "shuffled_context_draw_id": None,
            "shuffled_context_draw_rule": None,
        }
    if context.context_id == "shuffled":
        return {
            **common,
            "wrong_context_source_recording_key": None,
            "wrong_context_draw_id": None,
            "wrong_context_draw_rule": None,
            "shuffled_context_source_recording_key": heldout.recording_key,
            "shuffled_context_draw_id": (
                f"{heldout.recording_key}:support_block1:"
                "within_artifactclass_half_roll"
            ),
            "shuffled_context_draw_rule": (
                "support_block1_within_artifactclass_half_roll_"
                "with_one_third_roll_fallback"
            ),
        }
    return {
        **common,
        "wrong_context_source_recording_key": None,
        "wrong_context_draw_id": None,
        "wrong_context_draw_rule": None,
        "shuffled_context_source_recording_key": None,
        "shuffled_context_draw_id": None,
        "shuffled_context_draw_rule": None,
    }


def _runtime_context_metadata(value: RuntimeArtifactContext) -> dict[str, Any]:
    """Small non-signal context needed to reconstruct C A from frozen latents."""

    return {
        "context_id": value.context_id,
        "role": value.role,
        "full_transfer_C": np.asarray(value.full_transfer, dtype=np.float32).tolist(),
        "normalized_transfer_C": np.asarray(
            value.normalized_transfer, dtype=np.float32
        ).tolist(),
        "transfer_scale": np.asarray(value.transfer_scale, dtype=np.float32).tolist(),
        "rank": int(value.rank),
    }


def _posterior_reconstruction_metadata(
    prepared: PreparedSubjectArtifactFold,
    context: FactorialContext,
    inference: ArmInference,
) -> dict[str, Any]:
    subject = None if context.rho == 0.0 else context.subject
    return {
        "rho": float(context.rho),
        "population": _runtime_context_metadata(prepared.population_context),
        "subject": None if subject is None else _runtime_context_metadata(subject),
        "latent_mean": np.asarray(
            prepared.latent_normalizer.mean, dtype=np.float32
        ).tolist(),
        "latent_standard_deviation": np.asarray(
            prepared.latent_normalizer.standard_deviation, dtype=np.float32
        ).tolist(),
        "sample_seeds_by_batch": [
            list(value) for value in inference.sample_seeds_by_batch
        ],
        "same_index_population_subject_mixing": True,
    }


def _canonical_layout_token(value: object) -> str:
    token = str(value).strip().lower().replace("_", "")
    if not token.startswith("layout") or not token[6:].isdigit():
        raise ValueError("invalid SGEYESUB layout token")
    return f"layout_{int(token[6:]):02d}"


def _uncertainty_window_rows(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    heldout: HeldoutSubjectArtifactRecord,
    arm_inference: Mapping[str, ArmInference],
    frozen: FrozenFactorialOutputs,
    annotated: SgeyesubLoadedRecord,
    *,
    task: EvaluationTask,
) -> list[dict[str, Any]]:
    """Reconstruct one K8 latent arm/window at a time after immutable freeze."""

    if (
        not frozen.freeze_manifest_path.is_file()
        or not frozen.output_archive_path.is_file()
        or frozen.posterior_samples is None
    ):
        raise AssertionError("uncertainty scoring requires frozen latent posterior state")
    freeze_payload = json.loads(
        frozen.freeze_manifest_path.read_text(encoding="utf-8")
    )
    frozen_metadata = freeze_payload.get("posterior_reconstruction_metadata")
    if (
        freeze_payload.get("query_evaluation_fields_opened") is not False
        or freeze_payload.get("posterior_storage")
        != "standardized_latent_only_no_EEG_correction_or_restored_samples"
        or not isinstance(frozen_metadata, Mapping)
    ):
        raise AssertionError("posterior manifest does not prove latent-only freeze")
    annotations = annotated.query_annotations
    if annotations is None:
        raise AssertionError("uncertainty scoring requires post-freeze annotations")
    windows = heldout.query.observed
    count, channels, length = windows.shape
    labels = np.asarray(annotations.artifactclasses).reshape(count, length)
    predicted_continuous = _prediction_from_query_eog(config, heldout, annotated)
    if predicted_continuous.shape != (channels, count * length):
        raise ValueError("uncertainty EOG proxy does not align with frozen windows")
    predicted = predicted_continuous.reshape(channels, count, length).transpose(1, 0, 2)
    valid = np.asarray(heldout.query.valid_time_mask, dtype=bool)
    expected_latent_mean = np.asarray(
        prepared.latent_normalizer.mean, dtype=np.float64
    )
    expected_latent_scale = np.asarray(
        prepared.latent_normalizer.standard_deviation, dtype=np.float64
    )
    contexts = {
        value.context_id: value
        for value in factorial_context_plan(prepared.population_context, heldout)
    }

    def map_window(
        standardized: np.ndarray,
        normalized_transfer: np.ndarray,
        latent_mean: np.ndarray,
        latent_scale: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        selected = np.compress(mask, standardized, axis=-1)
        physical = (
            selected * latent_scale[None, :, None]
            + latent_mean[None, :, None]
        )
        return np.einsum(
            "ce,ket->kct",
            normalized_transfer,
            physical,
        )

    rows: list[dict[str, Any]] = []
    for evaluation_arm_id, inference in arm_inference.items():
        checkpoint_endpoint, model_id, context_id = evaluation_arm_id.split("__", 2)
        if model_id != "diffusion":
            continue
        population_key = (
            f"{evaluation_arm_id}__population_standardized_latent_samples"
        )
        subject_key = f"{evaluation_arm_id}__subject_standardized_latent_samples"
        if population_key not in frozen.posterior_samples:
            if inference.status.startswith("success"):
                raise ValueError("successful diffusion arm lacks frozen population latents")
            continue
        population_samples = np.asarray(
            frozen.posterior_samples[population_key], dtype=np.float64
        )
        context = contexts[context_id]
        if population_samples.shape[:2] != (8, count):
            raise ValueError("frozen population latent K/window axes are invalid")
        subject_samples = (
            None
            if subject_key not in frozen.posterior_samples
            else np.asarray(frozen.posterior_samples[subject_key], dtype=np.float64)
        )
        if context.rho == 0.0 and subject_samples is not None:
            raise ValueError("rho=0 uncertainty path found personalized latent samples")
        if context.rho != 0.0 and subject_samples is None:
            raise ValueError("nonzero-rho uncertainty path lacks subject latent samples")
        metadata = frozen_metadata.get(evaluation_arm_id)
        if (
            not isinstance(metadata, Mapping)
            or float(metadata.get("rho", float("nan"))) != float(context.rho)
            or metadata.get("same_index_population_subject_mixing") is not True
            or metadata.get("sample_seeds_by_batch")
            != [list(value) for value in inference.sample_seeds_by_batch]
        ):
            raise ValueError("frozen posterior reconstruction metadata is stale")
        population_metadata = metadata.get("population")
        subject_metadata = metadata.get("subject")
        if not isinstance(population_metadata, Mapping):
            raise ValueError("frozen population C metadata is unavailable")
        if population_metadata.get("context_id") != prepared.population_context.context_id:
            raise ValueError("frozen population C metadata identifies another context")
        if context.rho != 0.0 and (
            not isinstance(subject_metadata, Mapping)
            or subject_metadata.get("context_id") != context.subject.context_id
        ):
            raise ValueError("frozen subject C metadata identifies another context")
        frozen_latent_mean = np.asarray(metadata.get("latent_mean"), dtype=np.float64)
        frozen_latent_scale = np.asarray(
            metadata.get("latent_standard_deviation"), dtype=np.float64
        )
        frozen_population_C = np.asarray(
            population_metadata.get("normalized_transfer_C"), dtype=np.float64
        )
        frozen_subject_C = (
            None
            if subject_metadata is None
            else np.asarray(
                subject_metadata.get("normalized_transfer_C"), dtype=np.float64
            )
        )
        if (
            frozen_latent_mean.shape != expected_latent_mean.shape
            or frozen_latent_scale.shape != expected_latent_scale.shape
            or frozen_population_C.shape
            != np.asarray(prepared.population_context.normalized_transfer).shape
            or not np.isfinite(frozen_latent_mean).all()
            or not np.isfinite(frozen_latent_scale).all()
            or np.any(frozen_latent_scale <= 0.0)
            or not np.isfinite(frozen_population_C).all()
            or not np.allclose(
                frozen_latent_mean, expected_latent_mean, rtol=1.0e-6, atol=1.0e-7
            )
            or not np.allclose(
                frozen_latent_scale,
                expected_latent_scale,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
            or not np.allclose(
                frozen_population_C,
                np.asarray(prepared.population_context.normalized_transfer),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        ):
            raise ValueError("frozen latent/C reconstruction metadata is invalid")
        if subject_samples is not None and (
            frozen_subject_C is None
            or frozen_subject_C.shape != frozen_population_C.shape
            or not np.isfinite(frozen_subject_C).all()
            or not np.allclose(
                frozen_subject_C,
                np.asarray(context.subject.normalized_transfer),
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        ):
            raise ValueError("frozen subject C reconstruction metadata is invalid")
        frozen_point = np.asarray(
            frozen.outputs[evaluation_arm_id], dtype=np.float64
        ).reshape(channels, count, length).transpose(1, 0, 2)
        for window_index in range(count):
            mask = valid[window_index]
            if not np.any(mask):
                continue
            population_correction = map_window(
                population_samples[:, window_index],
                frozen_population_C,
                frozen_latent_mean,
                frozen_latent_scale,
                mask,
            )
            if subject_samples is None:
                sample_correction = population_correction
            else:
                if frozen_subject_C is None:
                    raise AssertionError("subject C vanished after metadata validation")
                subject_correction = map_window(
                    subject_samples[:, window_index],
                    frozen_subject_C,
                    frozen_latent_mean,
                    frozen_latent_scale,
                    mask,
                )
                sample_correction = (
                    (1.0 - context.rho) * population_correction
                    + context.rho * subject_correction
                )
            observed_window = np.compress(
                mask, windows[window_index].astype(np.float64), axis=-1
            )
            sample_restored = observed_window[None, :, :] - sample_correction
            target = np.compress(mask, predicted[window_index], axis=-1)
            point = sample_correction.mean(axis=0)
            frozen_point_window = np.compress(
                mask, frozen_point[window_index], axis=-1
            )
            if not np.allclose(
                sample_restored.mean(axis=0),
                frozen_point_window,
                rtol=1.0e-5,
                atol=1.0e-6,
            ):
                raise ValueError("frozen latent samples do not reproduce the point output")
            per_sample_proxy_rmse = np.sqrt(
                np.mean(np.square(sample_correction - target[None, :, :]), axis=(1, 2))
            )
            rows.append(
                {
                    "recording_key": heldout.recording_key,
                    "participant_stem": heldout.recording_key.split("/", 1)[-1],
                    "unified_fold_index": task.unified_fold_index,
                    "training_seed": task.training_seed,
                    "checkpoint_endpoint": checkpoint_endpoint,
                    "model_id": model_id,
                    "context_id": context_id,
                    "inference_status": inference.status,
                    "window_index_within_stem": window_index,
                    "posterior_sample_count_K": 8,
                    "posterior_output_SD_RMS": float(
                        np.sqrt(np.mean(np.var(sample_restored, axis=0)))
                    ),
                    "posterior_correction_SD_RMS": float(
                        np.sqrt(np.mean(np.var(sample_correction, axis=0)))
                    ),
                    "point_EOG_contamination_proxy_RMSE": float(
                        np.sqrt(np.mean(np.square(point - target)))
                    ),
                    "posterior_sample_EOG_proxy_RMSE_mean": float(
                        np.mean(per_sample_proxy_rmse)
                    ),
                    "posterior_sample_EOG_proxy_RMSE_SD": float(
                        np.std(per_sample_proxy_rmse)
                    ),
                    "artifact_labelled_fraction": float(
                        np.mean(labels[window_index, mask] != 6)
                    ),
                    "risk_coverage_unit": "window_nested_within_participant_stem",
                    "risk_coverage_input_only_not_success_claim": True,
                    "query_EOG_and_labels_opened_after_all_outputs_frozen": True,
                    "query_EOG_or_labels_used_for_inference_or_sample_selection": False,
                    "best_of_K_used": False,
                }
            )
    return rows


def _score_record(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    heldout: HeldoutSubjectArtifactRecord,
    frozen: FrozenFactorialOutputs,
    arm_inference: Mapping[str, ArmInference],
    annotated: SgeyesubLoadedRecord,
    *,
    task: EvaluationTask,
    implementation: Mapping[str, Any],
    checkpoint_endpoint: str,
    include_raw_reference: bool,
) -> list[dict[str, Any]]:
    if checkpoint_endpoint not in CHECKPOINT_ENDPOINTS:
        raise ValueError("scoring endpoint must be best or equal")
    annotation = annotated.query_annotations
    if annotation is None:
        raise AssertionError("query annotation missing after output freeze")
    observed_windows = heldout.query.observed.astype(np.float64)
    observed = _continuous(observed_windows)
    if annotation.external_eog.shape[1] != observed.shape[1]:
        raise ValueError("query scoring fields do not align with frozen EEG outputs")
    predicted = _prediction_from_query_eog(config, heldout, annotated)
    maximum_ratio = float(
        _mapping(_mapping(config, "validity"), "V0")[
            "maximum_per_window_output_input_RMS_ratio"
        ]
    )
    rows: list[dict[str, Any]] = []
    common = {
        "protocol_id": PROTOCOL_ID,
        "scientific_role": "development_exploratory_natural_EEG",
        "statistical_unit": "participant_stem",
        "window_level_inference": False,
        "windows_aggregated_within_stem": int(observed_windows.shape[0]),
        "unified_fold_index": task.unified_fold_index,
        "fold_id": prepared.fold.fold_id,
        "original_partition": prepared.fold.original_partition,
        "study": prepared.fold.study,
        "layout_id": prepared.fold.layout_id,
        "sampling_rate_hz": prepared.fold.sampling_rate_hz,
        "participant_stem": heldout.recording_key.split("/", 1)[-1],
        "recording_key": heldout.recording_key,
        "training_seed": task.training_seed,
        "original_stem_rho": heldout.matching.rho,
        "factorial_calibration_duration_seconds": (
            0.0
            if heldout.matching.rho == 0.0
            else heldout.matching.calibration_duration_seconds
        ),
        "query_evaluation_fields_opened_after_all_endpoint_outputs_frozen": True,
        "query_evaluation_fields_used_for_fit_selection_or_inference": False,
        "query_eog_used_for_inference": False,
        "query_labels_used_for_inference": False,
        "clean_waveform_metric": "N/A_no_clean_target",
        **dict(implementation),
    }
    plan = {
        value.context_id: value
        for value in factorial_context_plan(prepared.population_context, heldout)
    }
    for model_id in MODEL_IDS:
        for context_id in CONTEXT_IDS:
            arm_id = f"{model_id}__{context_id}"
            evaluation_arm_id = f"{checkpoint_endpoint}__{arm_id}"
            inference = arm_inference[evaluation_arm_id]
            # The immutable archive stays float32; only scoring math promotes.
            output = np.asarray(
                frozen.outputs[evaluation_arm_id], dtype=np.float64
            )
            row = _evaluate_output(
                method_id=evaluation_arm_id,
                output=output,
                observed=observed,
                matching_projector=heldout.matching.projector,
                population_projector=prepared.population_context.projector,
                query_eog=annotation.external_eog,
                artifactclasses=annotation.artifactclasses,
                predicted_contamination=predicted,
                trial_labels=annotation.trial_labels,
                samples_per_trial=int(
                    round(8.0 * float(prepared.fold.sampling_rate_hz))
                ),
                minimum_trials_per_condition=2,
                status=inference.status,
                operator_source=_arm_operator_source(context_id),
                gamma=None,
                fallback_used=inference.status.endswith("rho_zero"),
                uses_query_external_eog=False,
            )
            scale = _scale_metrics(
                inference.windowed_output,
                observed_windows,
                maximum_ratio=maximum_ratio,
            )
            retained_samples = (
                inference.population_standardized_latent_samples,
                inference.subject_standardized_latent_samples,
            )
            retained_samples_finite = all(
                value is None or np.isfinite(value).all()
                for value in retained_samples
            )
            full_v0 = _full_v0_scale_validity(
                config,
                inference.windowed_output,
                observed_windows,
                heldout.query.valid_time_mask,
                annotation.artifactclasses,
                span_consistency_relative_error=(
                    inference.complement_or_union_relative_error
                ),
                retained_samples_finite=retained_samples_finite,
            )
            scale.update(full_v0)
            performance_eligible = _performance_values_eligible(
                inference.status, scale
            )
            row.update(
                {
                    **common,
                    "checkpoint_endpoint": checkpoint_endpoint,
                    "model_id": model_id,
                    "context_id": context_id,
                    "latency_total_seconds": inference.latency_seconds,
                    "latency_seconds_per_window": inference.latency_seconds
                    / observed_windows.shape[0],
                    "peak_memory_mb": inference.peak_memory_mb,
                    "network_calls": inference.network_calls,
                    "posterior_standardized_latent_sd_rms": (
                        inference.posterior_standardized_latent_sd_rms
                    ),
                    "posterior_output_sample_sd_rms": (
                        inference.posterior_output_sample_sd_rms
                    ),
                    "posterior_sample_count_K": (
                        8
                        if inference.population_standardized_latent_samples is not None
                        else 0
                    ),
                    "posterior_point_rule": "strict_arithmetic_mean_no_best_of_K",
                    "posterior_sample_storage": "standardized_latent_only",
                    "posterior_latent_archive_arm_prefix": (
                        evaluation_arm_id
                        if inference.population_standardized_latent_samples is not None
                        else None
                    ),
                    "sample_seeds_by_batch": json.dumps(
                        inference.sample_seeds_by_batch
                    ),
                    "posterior_uncertainty_summary": (
                        "explicit_same_index_K8_output_samples_retained_"
                        "point_is_strict_arithmetic_mean"
                        if model_id == "diffusion"
                        else "N/A_deterministic_point_estimate"
                    ),
                    "complement_or_union_relative_error": (
                        inference.complement_or_union_relative_error
                    ),
                    "inference_branch": inference.branch,
                    "performance_values_eligible": performance_eligible,
                    "identity_placeholder_used_for_freeze_only": inference.status.startswith(
                        "failed"
                    ),
                    **_context_provenance(plan[context_id], heldout),
                    **scale,
                    **_low_artifact_metrics(
                        output, observed, annotation.artifactclasses
                    ),
                }
            )
            if not performance_eligible:
                for field in _PERFORMANCE_FIELDS:
                    row[field] = None
                row["condition_erp_proxy_status"] = "not_scored_ineligible_arm"
            rows.append(row)
    if not include_raw_reference:
        return rows
    raw = _evaluate_output(
        method_id="raw_observation",
        output=observed,
        observed=observed,
        matching_projector=heldout.matching.projector,
        population_projector=prepared.population_context.projector,
        query_eog=annotation.external_eog,
        artifactclasses=annotation.artifactclasses,
        predicted_contamination=predicted,
        trial_labels=annotation.trial_labels,
        samples_per_trial=int(round(8.0 * float(prepared.fold.sampling_rate_hz))),
        minimum_trials_per_condition=2,
        status="success_reference",
        operator_source="none_raw_observation",
        gamma=None,
        fallback_used=False,
        uses_query_external_eog=False,
    )
    raw.update(
        {
            **common,
            "checkpoint_endpoint": "shared_raw_reference",
            "model_id": "raw_reference",
            "context_id": "raw",
            "latency_total_seconds": 0.0,
            "latency_seconds_per_window": 0.0,
            "peak_memory_mb": 0.0,
            "network_calls": 0,
            "posterior_standardized_latent_sd_rms": float("nan"),
            "posterior_output_sample_sd_rms": float("nan"),
            "posterior_sample_count_K": 0,
            "posterior_point_rule": "N/A_raw_observation",
            "posterior_sample_storage": "N/A_raw_observation",
            "posterior_latent_archive_arm_prefix": None,
            "sample_seeds_by_batch": "[]",
            "posterior_uncertainty_summary": "N/A_raw_observation",
            "complement_or_union_relative_error": 0.0,
            "inference_branch": "raw_reference",
            "performance_values_eligible": True,
            "identity_placeholder_used_for_freeze_only": False,
            **_scale_metrics(
                observed_windows, observed_windows, maximum_ratio=maximum_ratio
            ),
            **_full_v0_scale_validity(
                config,
                observed_windows,
                observed_windows,
                heldout.query.valid_time_mask,
                annotation.artifactclasses,
                span_consistency_relative_error=0.0,
                retained_samples_finite=True,
            ),
            **_low_artifact_metrics(
                observed, observed, annotation.artifactclasses
            ),
        }
    )
    rows.append(raw)
    return rows


def _annotation_opener(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    recording_key: str,
) -> AnnotationOpener:
    layouts, records = _structure_maps(config)
    if recording_key not in records:
        raise ValueError("held-out recording is absent from the frozen structure audit")
    record = records[recording_key]
    if (
        record.study != prepared.fold.study
        or _canonical_layout_token(record.layout_id)
        != _canonical_layout_token(prepared.fold.layout_id)
        or float(record.sampling_rate_hz) != float(prepared.fold.sampling_rate_hz)
    ):
        raise ValueError("held-out scoring record crossed an exact compatibility cell")
    sge = _mapping(_mapping(config, "data"), "sgeyesub")
    root = Path(str(sge["data_root"]))
    return lambda: load_sgeyesub_signal_record(
        root,
        record,
        layouts[record.layout_id],
        include_query=True,
        include_query_annotations=True,
    )


def _validate_public_boundaries(config: Mapping[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("subject-artifact development protocol changed")
    if config.get("scientific_role") != "development_exploratory_not_confirmation":
        raise ValueError("natural-SGE factorial cannot be confirmation evidence")
    boundaries = _mapping(config, "boundaries")
    forbidden = (
        "participant_identity_input",
        "query_eog_or_eye_tracking_input",
        "query_artifact_label_input",
        "query_outcome_input",
        "best_of_k_selection",
        "confirmation_outcomes_this_round",
    )
    if any(boundaries.get(key) != "forbidden" for key in forbidden):
        raise ValueError("query-time information boundary was weakened")
    if tuple(_mapping(config, "factorial")["models"]) != (
        "deterministic_artifact_estimator",
        "artifact_latent_diffusion",
    ):
        raise ValueError("2x4 model matrix changed")
    if tuple(_mapping(config, "factorial")["contexts"]) != (
        "population",
        "matching",
        "wrong_same_cell",
        "shuffled_same_cell_severity_stratum",
    ):
        raise ValueError("2x4 context matrix changed")


def run_subject_artifact_evaluation(
    config: Mapping[str, Any],
    run_dir: str | Path,
    task_index: int,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one fold/seed after J3, with labels opened only for scoring."""

    _validate_public_boundaries(config)
    task = evaluation_task(config, task_index)
    prepared = prepare_subject_artifact_fold(config, task.unified_fold_index)
    if not prepared.heldout or set(prepared.fold.training_recording_keys) & set(
        prepared.fold.heldout_recording_keys
    ):
        raise AssertionError("natural-SGE fold is empty or identity-leaking")
    if not torch.cuda.is_available():
        raise RuntimeError("full natural-SGE factorial requires a scheduled GPU")
    device = torch.device("cuda")
    selected_repair = load_selected_validity_repair(config)
    models: dict[str, nn.Module] = {}
    checkpoint_paths: dict[str, str] = {}
    checkpoint_steps: dict[str, int] = {}
    implementation_commit = str(implementation.get("git_commit", "")).strip()
    if len(implementation_commit) != 40:
        raise ValueError("J4 requires the scheduled implementation Git commit")
    for checkpoint_endpoint in CHECKPOINT_ENDPOINTS:
        for model_id in MODEL_IDS:
            key = f"{checkpoint_endpoint}__{model_id}"
            model, path, payload = _load_ema_model(
                config,
                prepared,
                training_seed=task.training_seed,
                model_kind=model_id,
                device=device,
                implementation=implementation_commit,
                checkpoint_endpoint=checkpoint_endpoint,
                selected_repair=selected_repair,
            )
            models[key] = model
            checkpoint_paths[key] = str(path)
            checkpoint_steps[key] = int(payload["step"])
    output_root = Path(str(_mapping(config, "outputs")["development_root"]))
    destination = (
        output_root
        / "natural_sge_factorial"
        / f"fold_{task.unified_fold_index:02d}"
        / f"seed_{task.training_seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    freeze_manifests: list[str] = []
    estimated_point_output_bytes = 0
    estimated_posterior_latent_bytes = 0
    actual_compressed_archive_bytes = 0
    for recording_key in prepared.fold.heldout_recording_keys:
        heldout = prepared.heldout[recording_key]
        plan = factorial_context_plan(prepared.population_context, heldout)
        inference: dict[str, ArmInference] = {}
        continuous_outputs: dict[str, np.ndarray] = {}
        retained_samples: dict[str, np.ndarray] = {}
        required_retained_sample_keys: list[str] = []
        posterior_metadata: dict[str, Mapping[str, Any]] = {}
        for checkpoint_endpoint in CHECKPOINT_ENDPOINTS:
            for model_id in MODEL_IDS:
                for context in plan:
                    arm_id = f"{model_id}__{context.context_id}"
                    evaluation_arm_id = f"{checkpoint_endpoint}__{arm_id}"
                    try:
                        result = _infer_arm(
                            models[f"{checkpoint_endpoint}__{model_id}"],
                            model_kind=model_id,
                            prepared=prepared,
                            heldout=heldout,
                            context=context,
                            training_seed=task.training_seed,
                            config=config,
                            device=device,
                        )
                    except (RuntimeError, ValueError, FloatingPointError) as error:
                        result = _failed_arm(heldout.query.observed, error)
                        if device.type == "cuda" and "out of memory" in str(error).lower():
                            torch.cuda.empty_cache()
                    inference[evaluation_arm_id] = result
                    continuous_outputs[evaluation_arm_id] = _continuous(
                        result.windowed_output
                    )
                    if model_id == "diffusion" and result.status.startswith("success"):
                        required_values = [
                            (
                                "population_standardized_latent_samples",
                                result.population_standardized_latent_samples,
                            )
                        ]
                        if context.rho != 0.0:
                            required_values.append(
                                (
                                    "subject_standardized_latent_samples",
                                    result.subject_standardized_latent_samples,
                                )
                            )
                        for suffix, value in required_values:
                            if value is None:
                                raise AssertionError(
                                    "successful diffusion arm discarded required K8 latents"
                                )
                            key = f"{evaluation_arm_id}__{suffix}"
                            required_retained_sample_keys.append(key)
                            retained_samples[key] = value
                        posterior_metadata[evaluation_arm_id] = (
                            _posterior_reconstruction_metadata(
                                prepared, context, result
                            )
                        )
        manifest_path = destination / "output_freeze" / (
            recording_key.replace("/", "__") + ".json"
        )
        frozen = freeze_factorial_outputs(
            continuous_outputs,
            recording_key=recording_key,
            manifest_path=manifest_path,
            expected_arm_ids=EVALUATION_ARM_IDS,
            posterior_samples=retained_samples,
            required_posterior_sample_keys=required_retained_sample_keys,
            posterior_metadata=posterior_metadata,
        )
        estimated_point_output_bytes += frozen.point_output_bytes
        estimated_posterior_latent_bytes += frozen.posterior_latent_bytes
        actual_compressed_archive_bytes += frozen.archive_disk_bytes
        # Release pre-freeze copies before opening any scoring-only fields.
        continuous_outputs.clear()
        retained_samples.clear()
        posterior_metadata.clear()
        freeze_manifests.append(str(manifest_path))
        annotated = open_annotations_after_freeze(
            frozen,
            _annotation_opener(config, prepared, recording_key),
        )
        for checkpoint_endpoint in CHECKPOINT_ENDPOINTS:
            rows.extend(
                _score_record(
                    config,
                    prepared,
                    heldout,
                    frozen,
                    inference,
                    annotated,
                    task=task,
                    implementation=implementation,
                    checkpoint_endpoint=checkpoint_endpoint,
                    include_raw_reference=checkpoint_endpoint == "best",
                )
            )
        uncertainty_rows.extend(
            _uncertainty_window_rows(
                config,
                prepared,
                heldout,
                inference,
                frozen,
                annotated,
                task=task,
            )
        )
    expected_rows = len(prepared.heldout) * (len(EVALUATION_ARM_IDS) + 1)
    if len(rows) != expected_rows or any("window_index" in row for row in rows):
        raise AssertionError(
            "factorial metrics are not stem-level two-endpoint 2x4 plus raw"
        )
    metrics_path = destination / "metrics.csv"
    _atomic_csv(metrics_path, rows)
    uncertainty_path = destination / "uncertainty_windows.csv"
    if uncertainty_rows:
        _atomic_csv(uncertainty_path, uncertainty_rows)
    learned_rows = [row for row in rows if row["model_id"] in MODEL_IDS]
    summary = {
        "status": (
            "success_complete_fold_seed_development_factorial"
            if all(row["performance_values_eligible"] is True for row in learned_rows)
            else "completed_with_failed_or_ineligible_arms"
        ),
        "protocol_id": PROTOCOL_ID,
        "scientific_role": "development_exploratory_not_confirmation",
        "task_index": task.task_index,
        "unified_fold_index": task.unified_fold_index,
        "fold_id": prepared.fold.fold_id,
        "training_seed": task.training_seed,
        "training_stem_count": len(prepared.fold.training_recording_keys),
        "heldout_stem_count": len(prepared.fold.heldout_recording_keys),
        "learned_arm_rows": len(learned_rows),
        "successful_learned_arm_rows": sum(
            row["performance_values_eligible"] is True for row in learned_rows
        ),
        "failed_or_ineligible_learned_arm_rows": sum(
            row["performance_values_eligible"] is not True for row in learned_rows
        ),
        "preblocked_stem": BLOCKED_STEM,
        "preblocked_stem_count": 1,
        "preblocked_stem_performance_row_emitted": False,
        "statistical_unit": "participant_stem",
        "window_level_significance": False,
        "query_evaluation_fields_opened_after_output_freeze": True,
        "query_fields_used_for_inference_or_selection": False,
        "posterior_point_estimate": "arithmetic_K8_mean_no_best_of_K",
        "checkpoint_endpoints": list(CHECKPOINT_ENDPOINTS),
        "validity_execution_revision": selected_repair.execution_revision,
        "validity_selected_implementation": selected_repair.implementation,
        "identity_repair_active": selected_repair.identity_repair_active,
        "effective_standardized_latent_absolute_clip": (
            selected_repair.standardized_latent_absolute_clip
        ),
        "effective_inference_sampler": selected_repair.inference_sampler,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_steps": checkpoint_steps,
        "metrics_path": str(metrics_path),
        "uncertainty_window_metrics_path": (
            str(uncertainty_path) if uncertainty_rows else None
        ),
        "output_freeze_manifests": freeze_manifests,
        "large_waveforms_persisted": True,
        "large_waveforms_persisted_scope": "point_outputs_only_float32",
        "posterior_sample_storage": (
            "K8_standardized_latents_only_no_EEG_correction_or_restored_samples"
        ),
        "estimated_uncompressed_point_output_bytes": estimated_point_output_bytes,
        "estimated_uncompressed_posterior_latent_bytes": (
            estimated_posterior_latent_bytes
        ),
        "estimated_uncompressed_total_archive_bytes": (
            estimated_point_output_bytes + estimated_posterior_latent_bytes
        ),
        "actual_compressed_archive_bytes": actual_compressed_archive_bytes,
        "clean_waveform_metric": "N/A_no_clean_target",
        **dict(implementation),
    }
    summary_path = destination / "result_summary.json"
    _atomic_json(summary_path, summary)
    run_destination = Path(run_dir)
    run_destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_destination / "result_summary.json", summary)
    return summary


__all__ = [
    "CHECKPOINT_ENDPOINTS",
    "CONTEXT_IDS",
    "EVALUATION_ARM_IDS",
    "FACTORIAL_ARM_IDS",
    "MODEL_IDS",
    "TASK_COUNT",
    "EvaluationTask",
    "FactorialContext",
    "FrozenFactorialOutputs",
    "evaluation_task",
    "factorial_context_plan",
    "freeze_factorial_outputs",
    "open_annotations_after_freeze",
    "run_subject_artifact_evaluation",
    "subject_artifact_checkpoint_contract",
    "subject_artifact_checkpoint_path",
]
