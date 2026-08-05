"""Independent wide screen for full-C subject-aware diffusion routes.

This experiment lives in a dedicated worktree/result root.  It reuses the
frozen real-data folds, fixes one canonical artifact target per training
window, and keeps operator interventions out of target construction.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW

from eeg_cgdr.experiments.mainline_subject_residual_diffusion import (
    SEEDS,
    _annotation_opener,
    _continuous,
    _evaluate_output,
    _klados_eval_records,
    _load_models as _load_population_baseline,
    _paired_metrics,
    _prepared,
    _sge_samples_per_trial,
    _split_indices,
)
from eeg_cgdr.experiments.subject_artifact_data import PreparedSubjectArtifactFold
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)
from eeg_cgdr.models.artifact_subspace_diffusion import participant_sample_seeds
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


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_parallel_explore"))
PROTOCOL = "parallel_subject_aware_routes_v1"
SCREEN_SEED = 20260811
ROUTES = ("P1_FULL_C_RESIDUAL", "P2_FULL_C_FILM", "P3_ACTIVITY_GATE", "P4_SUPPORT_ADAPTER", "P5_POSTERIOR_GUIDANCE", "P6_ANCHORED_SDEDIT")


class RouteBlockedError(RuntimeError):
    """A scientifically required screen control cannot be constructed."""


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    if config.get("protocol_id") != PROTOCOL or int(config.get("harness_level", -1)) != 1:
        raise ValueError("parallel route protocol/harness changed")
    base_path = CODE_ROOT / str(config["base_subject_artifact_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base subject-artifact config is invalid")
    return base, CODE_ROOT / str(_mapping(config, "outputs")["root"])


def _implementation() -> dict[str, Any]:
    return {
        "git_sha": os.environ.get("DENOISENET_GIT_HEAD", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


def fold_rows(seeds: Sequence[int] = SEEDS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.append({"task_index": len(rows), "dataset": "klados", "fold_index": 0, "seed": int(seed)})
        for fold in range(25):
            rows.append({"task_index": len(rows), "dataset": "sgeyesub", "fold_index": fold, "seed": int(seed)})
    return rows


def screen_rows(routes: Sequence[str] = ROUTES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        for fold in fold_rows((SCREEN_SEED,)):
            rows.append({"task_index": len(rows), "route": route, **{key: value for key, value in fold.items() if key != "task_index"}})
    return rows


def base_screen_rows() -> list[dict[str, Any]]:
    return screen_rows(("P1_FULL_C_RESIDUAL", "P2_FULL_C_FILM"))


def dependent_screen_rows() -> list[dict[str, Any]]:
    return screen_rows(("P3_ACTIVITY_GATE", "P4_SUPPORT_ADAPTER", "P5_POSTERIOR_GUIDANCE", "P6_ANCHORED_SDEDIT"))


def _prepared_route(base: Mapping[str, Any], row: Mapping[str, Any]) -> PreparedSubjectArtifactFold:
    return _prepared(base, str(row["dataset"]), int(row["fold_index"]))


def _model_configs(prepared: PreparedSubjectArtifactFold, config: Mapping[str, Any]) -> tuple[ArtifactLatentModelConfig, ArtifactLatentDiffusionConfig]:
    model = _mapping(config, "model")
    diffusion = _mapping(config, "diffusion")
    return (
        ArtifactLatentModelConfig(
            eeg_channels=prepared.model_dimensions.eeg_channels,
            latent_channels=prepared.model_dimensions.eog_coordinates,
            signal_length=prepared.model_dimensions.signal_length,
            base_channels=int(model["base_channels"]),
            time_sinusoidal_dim=int(model["time_sinusoidal_dim"]),
            time_embed_dim=int(model["time_embed_dim"]),
        ),
        ArtifactLatentDiffusionConfig(
            num_timesteps=int(diffusion["timesteps"]),
            min_snr_gamma=float(diffusion["min_snr_gamma"]),
            posterior_samples=8,
        ),
    )


def _population_arrays(prepared: PreparedSubjectArtifactFold) -> dict[str, np.ndarray]:
    source = prepared.training
    count = source.observed.shape[0]
    population = prepared.population_context
    latent = np.asarray(source.standardized_artifact_latent, dtype=np.float32)
    # The target is copied once and is never projected/recomputed under a
    # matching/population/wrong intervention.
    return {
        "observed": np.asarray(source.observed, dtype=np.float32),
        "target": latent.copy(),
        "valid": np.asarray(source.valid_time_mask, dtype=bool),
        "full": np.repeat(population.full_transfer[None], count, axis=0).astype(np.float32),
        "normalized": np.repeat(population.normalized_transfer[None], count, axis=0).astype(np.float32),
        "scale": np.repeat(population.transfer_scale[None], count, axis=0).astype(np.float32),
        "singular": np.repeat(population.singular_values[None], count, axis=0).astype(np.float32),
        "rank": np.full(count, population.rank, dtype=np.int64),
        "rho": np.zeros(count, dtype=np.float32),
        "duration": np.zeros(count, dtype=np.float32),
        "channel_mask": np.asarray(source.channel_mask, dtype=bool),
        "recording_keys": np.asarray(source.recording_keys),
    }


def _subject_arrays(prepared: PreparedSubjectArtifactFold) -> dict[str, np.ndarray]:
    source = prepared.training
    target = np.asarray(source.standardized_artifact_latent, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(target.astype(np.float64), axis=2)).mean(axis=1)
    bins = np.array_split(np.arange(spectrum.shape[1]), 8)
    compact = np.stack([spectrum[:, index].mean(axis=1) for index in bins], axis=1)
    compact /= np.maximum(compact.mean(axis=1, keepdims=True), np.finfo(np.float64).eps)
    return {
        "observed": np.asarray(source.observed, dtype=np.float32),
        "target": target.copy(),
        "valid": np.asarray(source.valid_time_mask, dtype=bool),
        "full": np.asarray(source.full_transfer, dtype=np.float32),
        "normalized": np.asarray(source.normalized_transfer, dtype=np.float32),
        "scale": np.asarray(source.transfer_scale, dtype=np.float32),
        "singular": np.asarray(source.singular_values, dtype=np.float32),
        "rank": np.asarray(source.rank, dtype=np.int64),
        "rho": np.asarray(source.rho, dtype=np.float32),
        "duration": np.asarray(source.calibration_duration_seconds, dtype=np.float32),
        "channel_mask": np.asarray(source.channel_mask, dtype=bool),
        "support_sample_count": np.rint(source.calibration_duration_seconds * prepared.fold.sampling_rate_hz).astype(np.float32),
        "support_artifact_spectrum": compact.astype(np.float32),
        "recording_keys": np.asarray(source.recording_keys),
    }


def _batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[Tensor, dict[str, Tensor]]:
    def tensor(key: str, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(arrays[key][indices], device=device, dtype=dtype)
    target = canonical_target(tensor("target", torch.float32))
    condition = {
        "observed": tensor("observed", torch.float32),
        "full_transfer": tensor("full", torch.float32),
        "normalized_transfer": tensor("normalized", torch.float32),
        "transfer_scale": tensor("scale", torch.float32),
        "singular_values": tensor("singular", torch.float32),
        "rank": tensor("rank", torch.long),
        "rho": tensor("rho", torch.float32),
        "calibration_duration_seconds": tensor("duration", torch.float32),
        "channel_mask": tensor("channel_mask", torch.bool),
        "valid_time_mask": tensor("valid", torch.bool),
    }
    return target, condition


def _film_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[Tensor, dict[str, Tensor]]:
    target, condition = _batch(arrays, indices, device)
    condition["support_sample_count"] = torch.as_tensor(arrays["support_sample_count"][indices], device=device, dtype=torch.float32)
    condition["support_artifact_spectrum"] = torch.as_tensor(arrays["support_artifact_spectrum"][indices], device=device, dtype=torch.float32)
    return target, condition


def _checkpoint_path(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "checkpoints/P0" / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}" / f"seed_{int(row['seed'])}" / "model.pt"


def _route_checkpoint_path(root: Path, route: str, row: Mapping[str, Any]) -> Path:
    return root / "checkpoints" / route / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}" / f"seed_{int(row['seed'])}" / "model.pt"


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _ema_validation(
    model: ArtifactLatentDiffusion,
    ema: Mapping[str, Tensor],
    target: Tensor,
    condition: Mapping[str, Tensor],
    seed: int,
) -> float:
    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(ema)
    model.eval()
    generator = torch.Generator(device=target.device).manual_seed(seed)
    with torch.no_grad():
        score = float(model.training_loss(target, generator=generator, **condition)[1]["x0_mse"].cpu())
    model.load_state_dict(live)
    model.train()
    return score


def _train_p0(config: Mapping[str, Any], run_dir: Path, row: Mapping[str, Any]) -> Mapping[str, Any]:
    base, root = _load(config)
    checkpoint = _checkpoint_path(root, row)
    summary_path = checkpoint.parent / "result_summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed_P0_population_training":
            return {**prior, "resume_action": "skipped_completed"}
    prepared = _prepared_route(base, row)
    arrays = _population_arrays(prepared)
    model_config, diffusion_config = _model_configs(prepared, config)
    device = torch.device("cuda", 0)
    seed = int(row["seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    model = ArtifactLatentDiffusion(model_config, diffusion_config).to(device)
    deterministic = DeterministicArtifactEstimator(model_config).to(device)
    training = _mapping(config, "training")
    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    deterministic_optimizer = AdamW(deterministic.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train_indices, validation_indices = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    start = 0; curve: list[dict[str, Any]] = []
    resume = checkpoint.parent / "resume.pt"
    if resume.is_file():
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        deterministic.load_state_dict(payload["deterministic"]); deterministic_optimizer.load_state_dict(payload["deterministic_optimizer"])
        ema = {name: value.to(device) for name, value in payload["ema"].items()}
        start = int(payload["step"]); curve = list(payload.get("curve", ()))
        rng.bit_generator.state = payload["numpy_state"]; generator.set_state(payload["cuda_generator_state"])
    batch_size = int(training["batch_size"])
    maximum = int(training["maximum_updates"])
    best_score = float("inf"); best_state: dict[str, Tensor] | None = None; best_step = 0
    best_det_score = float("inf"); best_det_state: dict[str, Tensor] | None = None; best_det_step = 0
    started = time.perf_counter()
    for step in range(start + 1, maximum + 1):
        indices = rng.choice(train_indices, size=batch_size, replace=train_indices.size < batch_size)
        target, condition = _batch(arrays, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.training_loss(target, generator=generator, **condition)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("P0 population diffusion loss is NaN/Inf")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        deterministic_optimizer.zero_grad(set_to_none=True)
        det_prediction = deterministic(condition["observed"], **{key: value for key, value in condition.items() if key != "observed"})
        weight = condition["valid_time_mask"][:, None].to(det_prediction.dtype)
        det_loss = ((det_prediction - target).square() * weight).sum() / (weight.sum() * target.shape[1]).clamp_min(1)
        det_loss.backward()
        torch.nn.utils.clip_grad_norm_(deterministic.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True)
        deterministic_optimizer.step()
        decay = float(_mapping(config, "diffusion")["ema_decay"])
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "deterministic_loss": float(det_loss.detach()), "gradient_norm": float(gradient), **{key: float(value) for key, value in diagnostics.items()}})
        if step % int(training["validation_interval_updates"]) == 0 or step == maximum:
            validation_target, validation_condition = _batch(arrays, validation_indices, device)
            score = _ema_validation(model, ema, validation_target, validation_condition, seed + 99001)
            deterministic.eval()
            with torch.no_grad():
                det_value = deterministic(validation_condition["observed"], **{key: value for key, value in validation_condition.items() if key != "observed"})
                det_weight = validation_condition["valid_time_mask"][:, None].to(det_value.dtype)
                det_score = float((((det_value - validation_target).square() * det_weight).sum() / (det_weight.sum() * validation_target.shape[1]).clamp_min(1)).cpu())
            deterministic.train()
            curve.append({"step": step, "ema_validation_x0_mse": score, "deterministic_validation_MSE": det_score})
            if score < best_score:
                best_score, best_step = score, step
                best_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
            if det_score < best_det_score:
                best_det_score, best_det_step = det_score, step
                best_det_state = {name: value.detach().cpu().clone() for name, value in deterministic.state_dict().items()}
        if step % int(training["checkpoint_interval_updates"]) == 0:
            _save_checkpoint(resume, {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "deterministic": deterministic.state_dict(), "deterministic_optimizer": deterministic_optimizer.state_dict(), "ema": ema, "curve": curve, "numpy_state": rng.bit_generator.state, "cuda_generator_state": generator.get_state()})
    if best_state is None or best_det_state is None:
        raise AssertionError("validation did not select both P0 checkpoints")
    _save_checkpoint(checkpoint, {"protocol_id": PROTOCOL, "route": dict(row), "model_config": model_config.__dict__, "diffusion_config": diffusion_config.__dict__, "diffusion_ema": best_state, "deterministic": best_det_state, "best_step": best_step, "deterministic_best_step": best_det_step, "validation_x0_mse": best_score, "deterministic_validation_MSE": best_det_score, "latent_mean": prepared.latent_normalizer.mean, "latent_standard_deviation": prepared.latent_normalizer.standard_deviation, "population_context": prepared.population_context})
    _write_csv(checkpoint.parent / "training_curve.csv", curve)
    summary = {"status": "completed_P0_population_training", **_implementation(), **dict(row), "checkpoint": str(checkpoint), "best_step": best_step, "deterministic_best_step": best_det_step, "validation_x0_mse": best_score, "deterministic_validation_MSE": best_det_score, "runtime_seconds": time.perf_counter() - started, "target": "fixed_standardized_artifact_latent", "EMA_used_for_validation_and_checkpoint": True}
    _write_json(summary_path, summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def _train_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    tasks = fold_rows()
    indices = list(range(worker, len(tasks), 8))
    results = [_train_p0(config, run_dir / f"task_{index:03d}", tasks[index]) for index in indices]
    summary = {"status": "completed_P0_worker", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results)}
    _write_json(run_dir / "worker_summary.json", summary)
    return summary


def _train_chunk(config: Mapping[str, Any], run_dir: Path, worker: int, chunks: int = 16) -> Mapping[str, Any]:
    tasks = fold_rows()
    indices = list(range(worker, len(tasks), chunks))
    results = [_train_p0(config, run_dir / f"task_{index:03d}", tasks[index]) for index in indices]
    summary = {"status": "completed_P0_chunk", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results)}
    _write_json(run_dir / "worker_summary.json", summary)
    return summary


def _train_film(config: Mapping[str, Any], run_dir: Path, row: Mapping[str, Any]) -> Path:
    base, root = _load(config)
    checkpoint = _route_checkpoint_path(root, "P2_FULL_C_FILM", row)
    if checkpoint.is_file():
        return checkpoint
    prepared = _prepared_route(base, row)
    arrays = _subject_arrays(prepared)
    model_config, diffusion_config = _model_configs(prepared, config)
    device = torch.device("cuda", 0)
    seed = int(row["seed"])
    torch.manual_seed(seed + 200); torch.cuda.manual_seed_all(seed + 200)
    rng = np.random.default_rng(seed + 200)
    generator = torch.Generator(device=device).manual_seed(seed + 217)
    model = FullCFiLMDiffusion(model_config, diffusion_config, population_transfer=torch.as_tensor(prepared.population_context.full_transfer)).to(device)
    training = _mapping(config, "training")
    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train_indices, validation_indices = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    best_score = float("inf"); best_state: dict[str, Tensor] | None = None; best_step = 0
    curve: list[dict[str, Any]] = []
    maximum = int(_mapping(config, "screen").get("route_training_updates", 4000))
    batch_size = int(training["batch_size"])
    started = time.perf_counter()
    for step in range(1, maximum + 1):
        indices = rng.choice(train_indices, size=batch_size, replace=train_indices.size < batch_size)
        target, condition = _film_batch(arrays, indices, device)
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = model.training_loss(target, generator=generator, **condition)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer.step()
        decay = float(_mapping(config, "diffusion")["ema_decay"])
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient), **{key: float(value) for key, value in diagnostics.items()}})
        if step % int(training["validation_interval_updates"]) == 0 or step == maximum:
            target_v, condition_v = _film_batch(arrays, validation_indices, device)
            score = _ema_validation(model, ema, target_v, condition_v, seed + 99201)
            curve.append({"step": step, "ema_validation_x0_mse": score})
            if score < best_score:
                best_score, best_step = score, step
                best_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
    if best_state is None:
        raise AssertionError("FiLM route validation selected no checkpoint")
    _save_checkpoint(checkpoint, {"protocol_id": PROTOCOL, "route": dict(row), "model_config": model_config.__dict__, "diffusion_config": diffusion_config.__dict__, "diffusion_ema": best_state, "best_step": best_step, "validation_x0_mse": best_score, "latent_mean": prepared.latent_normalizer.mean, "latent_standard_deviation": prepared.latent_normalizer.standard_deviation, "population_context": prepared.population_context})
    _write_csv(checkpoint.parent / "training_curve.csv", curve)
    _write_json(checkpoint.parent / "result_summary.json", {"status": "completed_P2_FiLM_training", **_implementation(), **dict(row), "checkpoint": str(checkpoint), "best_step": best_step, "validation_x0_mse": best_score, "runtime_seconds": time.perf_counter() - started, "film_every_major_ResBlock": True, "canonical_target_invariant": True})
    return checkpoint


def _train_film_with_grad(
    config: Mapping[str, Any], run_dir: Path, row: Mapping[str, Any]
) -> Path:
    """Train/load FiLM even when called by the no-grad inference surface."""

    with torch.enable_grad():
        return _train_film(config, run_dir, row)


def _train_activity_gate(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, row: Mapping[str, Any], device: torch.device) -> Path:
    _, root = _load(config)
    checkpoint = _route_checkpoint_path(root, "P3_ACTIVITY_GATE", row)
    if checkpoint.is_file():
        return checkpoint
    arrays = _subject_arrays(prepared)
    seed = int(row["seed"])
    torch.manual_seed(seed + 300)
    rng = np.random.default_rng(seed + 300)
    gate = AdaptiveActivityGate(prepared.model_dimensions.eeg_channels).to(device)
    optimizer = AdamW(gate.parameters(), lr=float(_mapping(config, "training")["learning_rate"]))
    train_indices, validation_indices = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    magnitude = np.sqrt(np.mean(np.square(arrays["target"]), axis=1))
    scale = float(np.quantile(magnitude[train_indices], 0.75))
    activity = np.clip(magnitude / max(scale, np.finfo(float).eps), 0.0, 1.0).astype(np.float32)
    best_score = float("inf"); best_state: dict[str, Tensor] | None = None
    curve: list[dict[str, Any]] = []
    batch_size = int(_mapping(config, "training")["batch_size"])
    maximum = int(_mapping(config, "screen").get("gate_training_updates", 1500))
    for step in range(1, maximum + 1):
        indices = rng.choice(train_indices, size=batch_size, replace=train_indices.size < batch_size)
        y = torch.as_tensor(arrays["observed"][indices], device=device)
        mask = torch.as_tensor(arrays["valid"][indices], device=device)
        target = torch.as_tensor(activity[indices], device=device)[:, None]
        optimizer.zero_grad(set_to_none=True)
        predicted = gate(y, mask)
        weight = mask[:, None].to(predicted.dtype)
        loss = ((predicted - target).square() * weight).sum() / weight.sum().clamp_min(1)
        loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "activity_loss": float(loss.detach())})
        if step % 250 == 0 or step == maximum:
            yv = torch.as_tensor(arrays["observed"][validation_indices], device=device)
            mv = torch.as_tensor(arrays["valid"][validation_indices], device=device)
            tv = torch.as_tensor(activity[validation_indices], device=device)[:, None]
            with torch.no_grad():
                pv = gate(yv, mv); weight = mv[:, None].to(pv.dtype)
                score = float((((pv - tv).square() * weight).sum() / weight.sum().clamp_min(1)).cpu())
            curve.append({"step": step, "activity_validation_MSE": score})
            if score < best_score:
                best_score = score; best_state = {name: value.detach().cpu().clone() for name, value in gate.state_dict().items()}
    if best_state is None:
        raise AssertionError("activity gate validation selected no checkpoint")
    _save_checkpoint(checkpoint, {"gate": best_state, "activity_scale": scale, "validation_MSE": best_score, "query_EOG_used_for_inference": False, "training_target_source": "outer_training_canonical_latent_activity"})
    _write_csv(checkpoint.parent / "training_curve.csv", curve)
    return checkpoint


def _train_activity_gate_with_grad(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    row: Mapping[str, Any],
    device: torch.device,
) -> Path:
    """Train/load the activity gate from the no-grad inference surface."""

    with torch.enable_grad():
        return _train_activity_gate(config, prepared, row, device)


def _technical(config: Mapping[str, Any], run_dir: Path, route_index: int) -> Mapping[str, Any]:
    if not 0 <= route_index < 5:
        raise ValueError("technical route index must lie in [0,4]")
    base, _ = _load(config)
    prepared = _prepared(base, "klados", 0)
    arrays = _population_arrays(prepared)
    device = torch.device("cuda", 0)
    indices = np.arange(min(int(_mapping(config, "technical")["real_batch_size"]), arrays["target"].shape[0]))
    target, condition = _batch(arrays, indices, device)
    model_config, diffusion_config = _model_configs(prepared, config)
    base_model = ArtifactLatentDiffusion(model_config, diffusion_config).to(device)
    population = torch.tensor(np.array(prepared.population_context.full_transfer, copy=True), dtype=torch.float32)
    film = FullCFiLMDiffusion(model_config, diffusion_config, population_transfer=population).to(device)
    selected: torch.nn.Module = film if route_index in (2, 3) else base_model
    optimizer = AdamW(selected.parameters(), lr=2.0e-4)
    generator = torch.Generator(device=device).manual_seed(8800 + route_index)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = selected.training_loss(target, generator=generator, **condition)
    loss.backward(); optimizer.step()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("route technical loss is non-finite")
    invariant = torch.equal(canonical_target(target, condition["full_transfer"]), canonical_target(target, torch.flip(condition["full_transfer"], dims=(0,))))
    observed = condition["observed"]
    pop = observed * 0.95
    restored, correction = full_c_population_residual_reconstruction(observed, pop, target, population_normalized_transfer=condition["normalized_transfer"], subject_normalized_transfer=condition["normalized_transfer"] + 0.01, latent_mean=torch.as_tensor(prepared.latent_normalizer.mean, device=device), latent_standard_deviation=torch.as_tensor(prepared.latent_normalizer.standard_deviation, device=device), valid_time_mask=condition["valid_time_mask"], gain=1.0)
    gate = AdaptiveActivityGate(observed.shape[1]).to(device)(observed, condition["valid_time_mask"])
    guided = guided_latent_step(target, target * 0.5, strength=0.1)
    anchored = sdedit_initial_latent(target, torch.tensor(0.7, device=device), torch.randn_like(target))
    checkpoint = run_dir / "technical_checkpoint.pt"
    _save_checkpoint(checkpoint, {"model": selected.state_dict(), "route_index": route_index})
    reload_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    selected.load_state_dict(reload_payload["model"])
    context_difference = float(correction.abs().mean())
    status = {"status": "passed", "route": ("P0", "P1", "P2", "P3", "P6")[route_index], "finite": bool(torch.isfinite(restored).all() and torch.isfinite(gate).all() and torch.isfinite(guided).all() and torch.isfinite(anchored).all()), "target_invariant": invariant, "context_sensitive": context_difference > 1.0e-8, "checkpoint_reload": True, "film_block_count": film.film_block_count, "real_records_loaded": len(set(arrays["recording_keys"].tolist())), "loss": float(loss.detach()), **_implementation()}
    if not all(status[key] for key in ("finite", "target_invariant", "context_sensitive", "checkpoint_reload")):
        status["status"] = "failed"
    _write_json(run_dir / "technical_status.json", status)
    return status


def _load_screen_models(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, row: Mapping[str, Any], route: str, device: torch.device):
    base, root = _load(config)
    p0_path = _checkpoint_path(root, row)
    payload = torch.load(p0_path, map_location=device, weights_only=False)
    model_config = ArtifactLatentModelConfig(**payload["model_config"])
    diffusion_config = ArtifactLatentDiffusionConfig(**payload["diffusion_config"])
    if route == "P2_FULL_C_FILM" or route == "P3_ACTIVITY_GATE":
        route_path = _train_film_with_grad(
            config, Path(root) / "runs/internal_film_train", row
        )
        route_payload = torch.load(route_path, map_location=device, weights_only=False)
        model = FullCFiLMDiffusion(model_config, diffusion_config, population_transfer=torch.as_tensor(prepared.population_context.full_transfer)).to(device)
        model.load_state_dict(route_payload["diffusion_ema"])
        used_path = route_path
    else:
        model = ArtifactLatentDiffusion(model_config, diffusion_config).to(device)
        model.load_state_dict(payload["diffusion_ema"])
        used_path = p0_path
    deterministic = DeterministicArtifactEstimator(model_config).to(device)
    deterministic.load_state_dict(payload["deterministic"])
    old_config = yaml.safe_load((CODE_ROOT / "configs/cgdr/mainline_subject_residual_diffusion.yaml").read_text(encoding="utf-8"))
    anchor, _, _, _, old_checkpoint = _load_population_baseline(old_config, prepared, row, device)
    model.eval(); deterministic.eval(); anchor.eval()
    return model, deterministic, anchor, payload, used_path, old_checkpoint


def _runtime_condition(context: Any, count: int, prepared: PreparedSubjectArtifactFold, valid: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    def repeat(value: Any, dtype: torch.dtype) -> Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        return tensor[None].expand(count, *tensor.shape)
    singular = np.asarray(context.singular_values, dtype=np.float32)
    spectrum = np.resize(singular / max(float(np.mean(singular)), np.finfo(float).eps), 8).astype(np.float32)
    return {
        "full_transfer": repeat(context.full_transfer, torch.float32),
        "normalized_transfer": repeat(context.normalized_transfer, torch.float32),
        "transfer_scale": repeat(context.transfer_scale, torch.float32),
        "singular_values": repeat(context.singular_values, torch.float32),
        "rank": torch.full((count,), int(context.rank), device=device, dtype=torch.long),
        "rho": torch.full((count,), float(context.rho), device=device),
        "calibration_duration_seconds": torch.full((count,), float(context.calibration_duration_seconds), device=device),
        "channel_mask": torch.ones((count, prepared.model_dimensions.eeg_channels), device=device, dtype=torch.bool),
        "valid_time_mask": torch.as_tensor(valid, device=device, dtype=torch.bool),
        "support_sample_count": torch.full((count,), float(max(1, round(context.calibration_duration_seconds * prepared.fold.sampling_rate_hz))), device=device),
        "support_artifact_spectrum": torch.as_tensor(spectrum, device=device)[None].expand(count, -1),
    }


def _donor_contexts(prepared: PreparedSubjectArtifactFold, minimum: int = 3) -> list[Any]:
    source = prepared.training
    first: dict[str, int] = {}
    for index, key in enumerate(source.recording_keys):
        first.setdefault(str(key), index)
    contexts: list[Any] = []
    from eeg_cgdr.experiments.subject_artifact_data import RuntimeArtifactContext
    for donor, index in sorted(first.items())[:minimum]:
        full = np.asarray(source.full_transfer[index], dtype=np.float64)
        # Donor indices select one canonical full transfer first.  Derive its
        # normalized columns and scales together rather than combining cached
        # arrays whose leading window indices need not refer to the same raw
        # support fit.
        scale = np.maximum(np.linalg.norm(full, axis=0), np.finfo(float).eps)
        normalized = full / scale[None, :]
        singular = np.linalg.svd(full, compute_uv=False)
        left = np.linalg.svd(full.astype(np.float64), full_matrices=False)[0][:, : int(source.rank[index])]
        contexts.append(RuntimeArtifactContext(role="wrong_same_cell", context_id=f"training_donor:{donor}", raw_transfer=full, full_transfer=full, normalized_transfer=normalized, transfer_scale=scale, singular_values=singular, rank=int(source.rank[index]), projector=left @ left.T, rho=float(source.rho[index]), calibration_duration_seconds=float(source.calibration_duration_seconds[index]), fit_recording_keys=(donor,)))
    if len(contexts) < minimum:
        raise RouteBlockedError("exact cell supplies fewer than three outer-training wrong donors")
    return contexts


def _window_support(eeg: np.ndarray, latent: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(eeg.shape[1], latent.shape[1]) // length
    if count < 1:
        raise ValueError("calibration support is shorter than one model window")
    observed = np.stack([eeg[:, index * length:(index + 1) * length] for index in range(count)]).astype(np.float32)
    target = np.stack([latent[:, index * length:(index + 1) * length] for index in range(count)]).astype(np.float32)
    return observed, target, np.ones((count, length), dtype=bool)


def _sge_matching_support(base: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, fold_index: int, unit_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from eeg_cgdr.experiments.subject_artifact_data import (
        _calibration_samples,
        _eog_order,
        _fit_subject_transfer,
        _load_frozen_config,
        _unified_fold_route,
    )
    from eeg_cgdr.experiments.sgeyesub_diffusion_runner import _prepare_fold

    frozen = _load_frozen_config(base)
    partition, local = _unified_fold_route(frozen, fold_index)
    raw = _prepare_fold(frozen, partition, local)
    loaded = raw.heldout[unit_key]
    transfer = _fit_subject_transfer(base, raw, unit_key)
    count, _ = _calibration_samples(base, loaded.sampling_rate_hz, loaded.support.eeg.shape[1])
    eeg = raw.normalizer.transform(loaded.support.eeg)[:, :count]
    latent = transfer.standardized_artifact_latent(loaded.support.external_eog[:, :count], input_order=_eog_order(raw, unit_key))
    standardized = prepared.latent_normalizer.transform(latent[None])[0]
    return _window_support(eeg, standardized, prepared.model_dimensions.signal_length)


def _klados_matching_support(base: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, mechanism: Any, unit_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from eeg_cgdr.experiments.subject_artifact_klados_paired import EOG_ORDER, _fit_transfer, _raw_support_eog

    transfer = _fit_transfer(base, mechanism.calibration.eeg, _raw_support_eog(mechanism), fit_scope="support_only", fit_id=f"{unit_key}:parallel_adapter")
    latent = transfer.standardized_artifact_latent(_raw_support_eog(mechanism), input_order=EOG_ORDER)
    standardized = prepared.latent_normalizer.transform(latent[None])[0]
    return _window_support(mechanism.calibration.eeg, standardized, prepared.model_dimensions.signal_length)


def _adapter_summary(context: Any, population: Any, device: torch.device) -> Tensor:
    full = torch.as_tensor(np.array(context.full_transfer, copy=True), device=device, dtype=torch.float32)
    pop = torch.as_tensor(np.array(population.full_transfer, copy=True), device=device, dtype=torch.float32)
    singular = torch.as_tensor(np.array(context.singular_values, copy=True), device=device, dtype=torch.float32)
    scale = torch.as_tensor(np.array(context.transfer_scale, copy=True), device=device, dtype=torch.float32)
    return torch.cat((full.flatten(), (full - pop).flatten(), singular, scale, torch.tensor([float(context.calibration_duration_seconds)], device=device)))[None]


def _fit_support_adapter(
    deterministic: DeterministicArtifactEstimator,
    context: Any,
    population: Any,
    support: tuple[np.ndarray, np.ndarray, np.ndarray],
    prepared: PreparedSubjectArtifactFold,
    device: torch.device,
) -> tuple[SupportOnlyLatentAdapter, Tensor]:
    observed, target, valid = support
    count = observed.shape[0]
    condition = _runtime_condition(context, count, prepared, valid, device)
    standard = {key: value for key, value in condition.items() if key not in {"support_sample_count", "support_artifact_spectrum"}}
    y = torch.as_tensor(observed, device=device)
    truth = torch.as_tensor(target, device=device)
    with torch.no_grad():
        base_latent = deterministic(y, **standard).detach()
    summary = _adapter_summary(context, population, device).expand(count, -1)
    adapter = SupportOnlyLatentAdapter(truth.shape[1], summary.shape[1], rank=1).to(device)
    optimizer = AdamW(adapter.parameters(), lr=1.0e-3)
    mask = torch.as_tensor(valid, device=device)
    with torch.enable_grad():
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            prediction = adapter(base_latent, summary, mask)
            weight = mask[:, None].to(prediction.dtype)
            loss = ((prediction - truth).square() * weight).sum() / (weight.sum() * truth.shape[1]).clamp_min(1)
            loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
    adapter.eval()
    return adapter, _adapter_summary(context, population, device)


@torch.no_grad()
def _infer_route(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    row: Mapping[str, Any],
    route: str,
    *,
    unit_key: str,
    observed: np.ndarray,
    valid: np.ndarray,
    matching: Any,
    matching_support: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model, deterministic, anchor, payload, checkpoint, old_checkpoint = _load_screen_models(config, prepared, row, route, device)
    donors = _donor_contexts(prepared)
    batch_size = int(_mapping(config, "training")["batch_size"])
    names = ("RAW", "POP", "DET-MATCH", "DIFF-POP", "DIFF-MATCH", "DIFF-MATCH-K1", "DIFF-WRONG-0", "DIFF-WRONG-1", "DIFF-WRONG-2")
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    latent_mean = torch.as_tensor(payload["latent_mean"], device=device, dtype=torch.float32)
    latent_std = torch.as_tensor(payload["latent_standard_deviation"], device=device, dtype=torch.float32)
    seeds = participant_sample_seeds(unit_key, int(row["seed"]))
    adapters: dict[str, tuple[SupportOnlyLatentAdapter, Tensor]] = {}
    if route == "P4_SUPPORT_ADAPTER":
        if matching_support is None:
            raise ValueError("P4 requires query-disjoint calibration support")
        adapters["DIFF-MATCH"] = _fit_support_adapter(deterministic, matching, prepared.population_context, matching_support, prepared, device)
        training_arrays = _subject_arrays(prepared)
        keys = np.asarray(training_arrays["recording_keys"])
        for donor_index, donor in enumerate(donors):
            donor_key = donor.fit_recording_keys[0]
            selected = np.flatnonzero(keys == donor_key)
            if selected.size == 0:
                raise ValueError("wrong adapter donor has no outer-training support windows")
            donor_support = (training_arrays["observed"][selected], training_arrays["target"][selected], training_arrays["valid"][selected])
            adapters[f"DIFF-WRONG-{donor_index}"] = _fit_support_adapter(deterministic, donor, prepared.population_context, donor_support, prepared, device)
    started = time.perf_counter(); calls = 0
    for start in range(0, observed.shape[0], batch_size):
        stop = min(start + batch_size, observed.shape[0])
        y = torch.as_tensor(observed[start:stop], device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid[start:stop], device=device, dtype=torch.bool)
        pop_output = anchor(y, mask)
        chunks["RAW"].append(y.cpu().numpy()); chunks["POP"].append(pop_output.cpu().numpy()); chunks["DIFF-POP"].append(pop_output.cpu().numpy())
        match_condition = _runtime_condition(matching, y.shape[0], prepared, valid[start:stop], device)
        det_latent = deterministic(y, **{key: value for key, value in match_condition.items() if key not in {"support_sample_count", "support_artifact_spectrum"}})
        det_output, _ = full_c_population_residual_reconstruction(y, pop_output, det_latent, population_normalized_transfer=torch.as_tensor(prepared.population_context.normalized_transfer, device=device), subject_normalized_transfer=match_condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask, gain=1.0)
        chunks["DET-MATCH"].append(det_output.cpu().numpy())
        contexts = [("DIFF-MATCH", matching), *((f"DIFF-WRONG-{index}", donor) for index, donor in enumerate(donors))]
        for name, context in contexts:
            condition = _runtime_condition(context, y.shape[0], prepared, valid[start:stop], device)
            kwargs = dict(condition)
            if not isinstance(model, FullCFiLMDiffusion):
                kwargs.pop("support_sample_count"); kwargs.pop("support_artifact_spectrum")
            if route in {"P5_POSTERIOR_GUIDANCE", "P6_ANCHORED_SDEDIT"}:
                coordinates = torch.einsum("bce,bct->bet", condition["normalized_transfer"], y)
                anchor_latent = ((coordinates - latent_mean[None, :, None]) / latent_std[None, :, None]).clamp(-5.0, 5.0)
                standard_condition = {key: value for key, value in kwargs.items() if key not in {"support_sample_count", "support_artifact_spectrum"}}
                standard_condition["observed"] = y
                latent_value, posterior_samples, route_calls = structured_latent_samples(
                    model,
                    observation_anchor=anchor_latent,
                    sample_seeds=seeds,
                    condition=standard_condition,
                    mode=("posterior_guidance" if route == "P5_POSTERIOR_GUIDANCE" else "anchored_sdedit"),
                    guidance_strength=float(_mapping(config, "screen").get("guidance_strength", 0.1)),
                    sdedit_start_timestep=int(_mapping(config, "screen").get("sdedit_start_timestep", 250)),
                    ddim_steps=25,
                )
                calls += route_calls
            else:
                posterior = model.posterior_mean(observed=y, latent_mean=latent_mean, latent_standard_deviation=latent_std, sample_seeds=seeds, ddim_steps=25, record_trajectory=False, **kwargs)
                calls += int(posterior.network_calls)
                latent_value = posterior.standardized_latent_mean
                if posterior.standardized_latent_samples is None:
                    raise AssertionError("K1 requires explicit posterior samples")
                posterior_samples = posterior.standardized_latent_samples
            if route == "P4_SUPPORT_ADAPTER":
                adapter, summary = adapters[name]
                expanded = summary.expand(latent_value.shape[0], -1)
                latent_value = adapter(latent_value, expanded, mask)
                posterior_samples = torch.stack(tuple(adapter(sample, expanded, mask) for sample in posterior_samples))
            output, _ = full_c_population_residual_reconstruction(y, pop_output, latent_value, population_normalized_transfer=torch.as_tensor(prepared.population_context.normalized_transfer, device=device), subject_normalized_transfer=condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask, gain=1.0)
            if route == "P3_ACTIVITY_GATE" and name == "DIFF-MATCH":
                gate_path = _train_activity_gate_with_grad(
                    config, prepared, row, device
                )
                gate = AdaptiveActivityGate(y.shape[1]).to(device)
                gate.load_state_dict(torch.load(gate_path, map_location=device, weights_only=False)["gate"]); gate.eval()
                activity = gate(y, mask)
                output = pop_output + activity * (output - pop_output)
            chunks[name].append(output.cpu().numpy())
            if name == "DIFF-MATCH":
                output_k1, _ = full_c_population_residual_reconstruction(y, pop_output, posterior_samples[0], population_normalized_transfer=torch.as_tensor(prepared.population_context.normalized_transfer, device=device), subject_normalized_transfer=condition["normalized_transfer"], latent_mean=latent_mean, latent_standard_deviation=latent_std, valid_time_mask=mask, gain=1.0)
                chunks["DIFF-MATCH-K1"].append(output_k1.cpu().numpy())
    outputs = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    resources = {"runtime_seconds": time.perf_counter() - started, "network_calls": calls, "checkpoint": str(checkpoint), "population_checkpoint": str(old_checkpoint), "wrong_donors": [context.context_id for context in donors], "common_random_numbers": True, "K1_uses_own_posterior_sample_and_population_anchor": True}
    return outputs, resources


def _screen_task(config: Mapping[str, Any], run_dir: Path, task: Mapping[str, Any]) -> Mapping[str, Any]:
    route = str(task["route"])
    if route not in {"P1_FULL_C_RESIDUAL", "P2_FULL_C_FILM", "P3_ACTIVITY_GATE", "P4_SUPPORT_ADAPTER", "P5_POSTERIOR_GUIDANCE", "P6_ANCHORED_SDEDIT"}:
        raise ValueError(f"screen route is not implemented in the base array: {route}")
    base, root = _load(config)
    row = {key: task[key] for key in ("dataset", "fold_index", "seed")}
    prepared = _prepared_route(base, row)
    device = torch.device("cuda", 0)
    output_root = root / "route_screen" / route / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}"
    summary_path = output_root / "result_summary.json"
    if summary_path.is_file() and (output_root / "metrics.csv").is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed_full_real_route_screen":
            return {**prior, "resume_action": "skipped_completed"}
    rows: list[dict[str, Any]] = []
    arrays_root = root / "server_arrays" / route / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}"
    arrays_root.mkdir(parents=True, exist_ok=True)
    if row["dataset"] == "klados":
        for unit_key, mechanism, matching, _ in _klados_eval_records(base):
            matching_support = _klados_matching_support(base, prepared, mechanism, unit_key) if route == "P4_SUPPORT_ADAPTER" else None
            outputs, resources = _infer_route(config, prepared, row, route, unit_key=unit_key, observed=mechanism.observed_windows.astype(np.float32), valid=mechanism.valid_time_weight.astype(bool), matching=matching, matching_support=matching_support, device=device)
            np.savez_compressed(arrays_root / f"{unit_key}.npz", **{key.replace("-", "_"): value for key, value in outputs.items()})
            for method, output in outputs.items():
                rows.append({"route": route, "dataset": "klados", "unit_id": unit_key, "exact_cell": prepared.fold.layout_id, "training_seed": int(row["seed"]), "method": method, "status": "success", **_paired_metrics(mechanism.observed_windows, mechanism.clean_windows, output, mechanism.valid_time_weight.astype(bool)), "statistical_unit": "source_record", "screening_only": True, "query_information_used": False, "network_calls_total_for_unit": resources["network_calls"]})
    else:
        for unit_key, heldout in prepared.heldout.items():
            try:
                matching_support = _sge_matching_support(base, prepared, int(row["fold_index"]), unit_key) if route == "P4_SUPPORT_ADAPTER" else None
                outputs, resources = _infer_route(config, prepared, row, route, unit_key=unit_key, observed=heldout.query.observed, valid=heldout.query.valid_time_mask, matching=heldout.matching, matching_support=matching_support, device=device)
            except RouteBlockedError as error:
                for method in ("RAW", "POP", "DET-MATCH", "DIFF-POP", "DIFF-MATCH", "DIFF-MATCH-K1", "DIFF-WRONG-0", "DIFF-WRONG-1", "DIFF-WRONG-2"):
                    rows.append({
                        "route": route,
                        "dataset": "sgeyesub",
                        "unit_id": unit_key,
                        "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}",
                        "study": prepared.fold.study,
                        "training_seed": int(row["seed"]),
                        "method": method,
                        "method_id": method,
                        "status": "blocked_no_three_same_cell_wrong_donors",
                        "failure_reason": str(error),
                        "statistical_unit": "participant_stem",
                        "screening_only": True,
                        "outputs_frozen_before_query_scoring": False,
                    })
                continue
            archive = arrays_root / f"{unit_key.replace('/', '__')}.npz"
            np.savez_compressed(archive, **{key.replace("-", "_"): value for key, value in outputs.items()})
            # Query EOG/annotations are opened only after every method output
            # for this unit has been frozen to the route-specific server array.
            annotated = _annotation_opener(base, prepared, unit_key)()
            annotations = annotated.query_annotations
            if annotations is None:
                raise AssertionError("SGE query annotations were not opened after output freeze")
            observed_continuous = _continuous(heldout.query.observed)
            for method, output in outputs.items():
                metric = _evaluate_output(method_id=method, output=_continuous(output), observed=observed_continuous, matching_projector=heldout.matching.projector, population_projector=prepared.population_context.projector, query_eog=annotations.external_eog, artifactclasses=annotations.artifactclasses, predicted_contamination=None, trial_labels=annotations.trial_labels, samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz), minimum_trials_per_condition=2, status="success", operator_source=route, gamma=1.0, fallback_used=(method in {"POP", "DIFF-POP"}), uses_query_external_eog=False)
                rows.append({"route": route, "dataset": "sgeyesub", "unit_id": unit_key, "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}", "study": prepared.fold.study, "training_seed": int(row["seed"]), "method": method, **metric, "statistical_unit": "participant_stem", "screening_only": True, "outputs_frozen_before_query_scoring": True, "network_calls_total_for_unit": resources["network_calls"]})
    _write_csv(output_root / "metrics.csv", rows)
    summary = {"status": "completed_full_real_route_screen", **_implementation(), **dict(task), "route": route, "unit_count": len(set(row["unit_id"] for row in rows)), "metric_rows": len(rows), "screen_seed_count": 1, "scientific_role": "route_screening_not_final_claim", "result": str(output_root / "metrics.csv")}
    _write_json(summary_path, summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _aggregate_screen(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Create a compact one-seed route-screen audit without selecting a winner."""

    _, root = _load(config)
    metric_files = sorted((root / "route_screen").glob("*/*/fold_*/metrics.csv"))
    if not metric_files:
        raise FileNotFoundError("no completed route-screen metrics are available")
    rows: list[dict[str, Any]] = []
    for path in metric_files:
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    coverage: dict[tuple[str, str], set[str]] = {}
    success_coverage: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        route = str(row["route"])
        dataset = str(row["dataset"])
        method = str(row.get("method") or row.get("method_id"))
        grouped.setdefault((route, dataset, method), []).append(row)
        coverage.setdefault((route, dataset), set()).add(str(row["unit_id"]))
        if str(row.get("status", "")).startswith("success"):
            success_coverage.setdefault((route, dataset), set()).add(str(row["unit_id"]))

    metric_names = (
        "clean_waveform_RRMSE",
        "clean_waveform_correlation",
        "artifact_reconstruction_relative_error",
        "delta_SNR_db",
        "eog_coherence_reduction",
        "nonartifact_observation_preservation",
        "reference_free_psd_distortion",
        "reference_free_covariance_distortion",
        "condition_erp_observation_relative_preservation",
        "output_input_RMS_ratio",
        "observation_change_ratio",
    )
    method_summary: list[dict[str, Any]] = []
    for (route, dataset, method), values in sorted(grouped.items()):
        successful = [row for row in values if str(row.get("status", "")).startswith("success")]
        summary: dict[str, Any] = {
            "route": route,
            "dataset": dataset,
            "method": method,
            "success_units": len({str(row["unit_id"]) for row in successful}),
            "coverage_units": len(coverage[(route, dataset)]),
            "blocked_or_failed_units": len(coverage[(route, dataset)]) - len({str(row["unit_id"]) for row in successful}),
            "scientific_role": "one_seed_full_real_route_screen",
        }
        for metric in metric_names:
            numbers = [number for row in successful if (number := _float_or_none(row.get(metric))) is not None]
            if numbers:
                summary[f"mean_{metric}"] = float(np.mean(numbers))
                summary[f"median_{metric}"] = float(np.median(numbers))
        method_summary.append(summary)

    by_unit: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not str(row.get("status", "")).startswith("success"):
            continue
        key = (str(row["route"]), str(row["dataset"]), str(row["unit_id"]))
        by_unit.setdefault(key, {})[str(row.get("method") or row.get("method_id"))] = row
    effects: list[dict[str, Any]] = []
    for (route, dataset, unit), methods in sorted(by_unit.items()):
        if dataset == "klados":
            metric, sign = "clean_waveform_RRMSE", -1.0
        else:
            metric, sign = "eog_coherence_reduction", 1.0
        match = methods.get("DIFF-MATCH")
        if match is None:
            continue
        match_value = _float_or_none(match.get(metric))
        if match_value is None:
            continue
        comparisons: list[tuple[str, float]] = []
        for right in ("DIFF-POP", "DET-MATCH"):
            right_row = methods.get(right)
            right_value = None if right_row is None else _float_or_none(right_row.get(metric))
            if right_value is not None:
                comparisons.append((right, right_value))
        wrong_values = [
            value
            for index in range(3)
            if (wrong := methods.get(f"DIFF-WRONG-{index}")) is not None
            and (value := _float_or_none(wrong.get(metric))) is not None
        ]
        if len(wrong_values) == 3:
            comparisons.append(("MEAN-OF-3-DIFF-WRONG", float(np.mean(wrong_values))))
        for right, right_value in comparisons:
            effects.append({
                "route": route,
                "dataset": dataset,
                "unit_id": unit,
                "metric": metric,
                "left": "DIFF-MATCH",
                "right": right,
                "utility_effect_positive_is_better": sign * (match_value - right_value),
                "screening_only": True,
            })

    output = root / "screen_aggregation"
    _write_csv(output / "method_summary.csv", method_summary)
    _write_csv(output / "paired_effects.csv", effects)
    summary = {
        "status": "completed_parallel_route_screen_aggregation",
        **_implementation(),
        "metric_files": len(metric_files),
        "routes_observed": sorted({str(row["route"]) for row in rows}),
        "rows": len(rows),
        "paired_effect_rows": len(effects),
        "screening_seed": SCREEN_SEED,
        "top_route_selected": False,
        "three_seed_confirmation_run": False,
        "formal_FIR_diffusion_run": False,
        "results": {
            "method_summary": str(output / "method_summary.csv"),
            "paired_effects": str(output / "paired_effects.csv"),
        },
    }
    _write_json(output / "result_summary.json", summary)
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _screen_base_chunk(config: Mapping[str, Any], run_dir: Path, worker: int, chunks: int = 16) -> Mapping[str, Any]:
    tasks = base_screen_rows()
    indices = list(range(worker, len(tasks), chunks))
    results = [_screen_task(config, run_dir / f"task_{index:03d}", tasks[index]) for index in indices]
    summary = {"status": "completed_base_screen_chunk", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results), "routes": ["P1_FULL_C_RESIDUAL", "P2_FULL_C_FILM"]}
    _write_json(run_dir / "worker_summary.json", summary)
    return summary


def _screen_dependent_chunk(config: Mapping[str, Any], run_dir: Path, worker: int, chunks: int = 16) -> Mapping[str, Any]:
    tasks = dependent_screen_rows()
    indices = list(range(worker, len(tasks), chunks))
    results = [_screen_task(config, run_dir / f"task_{index:03d}", tasks[index]) for index in indices]
    summary = {"status": "completed_dependent_screen_chunk", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results), "routes": ["P3_ACTIVITY_GATE", "P4_SUPPORT_ADAPTER", "P5_POSTERIOR_GUIDANCE", "P6_ANCHORED_SDEDIT"]}
    _write_json(run_dir / "worker_summary.json", summary)
    return summary


def _j0(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    del base
    data = _mapping(config, "data")
    availability = {name: Path(str(path)).exists() for name, path in data.items() if name in {"klados", "sgeyesub"}}
    if not all(availability.values()):
        raise FileNotFoundError(f"required existing datasets are unavailable: {availability}")
    tasks = root / "task_lists"
    _write_csv(tasks / "p0_training.csv", fold_rows())
    _write_csv(tasks / "route_screen.csv", screen_rows())
    summary = {"status": "completed_J0", **_implementation(), "availability": availability, "p0_tasks": len(fold_rows()), "screen_tasks": len(screen_rows()), "seeds": list(SEEDS), "routes": list(ROUTES), "confirmation_data_opened": False}
    _write_json(run_dir / "result_summary.json", summary); _write_json(root / "j0_summary.json", summary)
    return summary


def _lag_design(latent: np.ndarray, valid: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Return all valid fixed-lag samples without choosing a candidate."""

    windows, coordinates, length = latent.shape
    offsets = tuple(range(-radius, radius + 1)) if radius else (0,)
    rows: list[np.ndarray] = []
    locations: list[tuple[int, int]] = []
    for window in range(windows):
        for sample in range(radius, length - radius):
            if not valid[window, sample] or any(not valid[window, sample + lag] for lag in offsets):
                continue
            rows.append(np.concatenate([latent[window, :, sample + lag] for lag in offsets]))
            locations.append((window, sample))
    if not rows:
        raise ValueError("fixed-lag candidate has no valid support samples")
    return np.stack(rows), np.asarray(locations, dtype=np.int64)


def _fir_direction(
    latent: np.ndarray,
    contamination: np.ndarray,
    valid: np.ndarray,
    fit_windows: np.ndarray,
    score_windows: np.ndarray,
    *,
    radius: int,
    ridge: float,
) -> float:
    design, locations = _lag_design(latent, valid, radius)
    fit = np.isin(locations[:, 0], fit_windows)
    score = np.isin(locations[:, 0], score_windows)
    if fit.sum() < design.shape[1] + 2 or score.sum() < 2:
        return float("nan")
    response = contamination[locations[:, 0], :, locations[:, 1]]
    gram = design[fit].T @ design[fit] + float(ridge) * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design[fit].T @ response[fit])
    residual = response[score] - design[score] @ weights
    baseline = response[score] - response[fit].mean(axis=0, keepdims=True)
    return float(np.mean(np.square(residual)) / max(np.mean(np.square(baseline)), np.finfo(float).eps))


def _fir_cache(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    rows: list[dict[str, Any]] = []
    candidates = tuple((radius, ridge) for radius in (0, 2, 4, 8) for ridge in (0.001, 0.01, 0.1))
    for fold in fold_rows((SCREEN_SEED,)):
        prepared = _prepared_route(base, fold)
        source = prepared.training
        mean = prepared.latent_normalizer.mean[None, :, None]
        scale = prepared.latent_normalizer.standard_deviation[None, :, None]
        physical = source.standardized_artifact_latent.astype(np.float64) * scale + mean
        contamination = np.einsum("nce,net->nct", source.normalized_transfer.astype(np.float64), physical)
        keys = np.asarray(source.recording_keys)
        unique = np.asarray(sorted(set(keys.tolist())))
        if unique.size > 1:
            fit_windows = np.flatnonzero(np.isin(keys, unique[::2]))
            score_windows = np.flatnonzero(np.isin(keys, unique[1::2]))
        else:
            fit_windows = np.arange(keys.size)[::2]
            score_windows = np.arange(keys.size)[1::2]
        if fit_windows.size == 0 or score_windows.size == 0:
            raise ValueError("FIR split-half cache cannot form two nonempty support halves")
        for radius, ridge in candidates:
            forward = _fir_direction(physical, contamination, source.valid_time_mask, fit_windows, score_windows, radius=radius, ridge=ridge)
            reverse = _fir_direction(physical, contamination, source.valid_time_mask, score_windows, fit_windows, radius=radius, ridge=ridge)
            rows.append({**dict(fold), "lag_radius_samples": radius, "ridge": ridge, "A_to_B_normalized_prediction_error": forward, "B_to_A_normalized_prediction_error": reverse, "mean_crossfit_error": float(np.nanmean((forward, reverse))), "selection_status": "cached_not_selected", "fit_windows": int(fit_windows.size), "score_windows": int(score_windows.size)})
    output = root / "fir_cache"
    _write_csv(output / "fixed_candidate_crossfit.csv", rows)
    summary = {"status": "completed_FIR_candidate_cache_without_selection", **_implementation(), "folds": len(fold_rows((SCREEN_SEED,))), "candidates_per_fold": len(candidates), "rows": len(rows), "formal_FIR_diffusion_submitted": False, "result": str(output / "fixed_candidate_crossfit.csv")}
    _write_json(output / "result_summary.json", summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def run_stage(config: Mapping[str, Any], run_dir: str | Path, stage: str, task_index: int | None) -> Mapping[str, Any]:
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    if stage == "j0":
        return _j0(config, run)
    if stage == "fir-cache":
        return _fir_cache(config, run)
    if stage == "technical":
        if task_index is None:
            raise ValueError("technical requires an array task")
        return _technical(config, run, task_index)
    if stage == "train-p0-worker":
        if task_index is None or not 0 <= task_index < 8:
            raise ValueError("P0 training worker requires array 0-7")
        return _train_worker(config, run, task_index)
    if stage == "train-p0-task":
        tasks = fold_rows()
        if task_index is None or not 0 <= task_index < len(tasks):
            raise ValueError("P0 training task requires array 0-77")
        return _train_p0(config, run, tasks[task_index])
    if stage == "train-p0-chunk":
        if task_index is None or not 0 <= task_index < 16:
            raise ValueError("P0 training chunk requires array 0-15")
        return _train_chunk(config, run, task_index)
    if stage == "screen-base-worker":
        if task_index is None or not 0 <= task_index < 16:
            raise ValueError("base screen chunk requires array 0-15")
        return _screen_base_chunk(config, run, task_index)
    if stage == "screen-dependent-worker":
        if task_index is None or not 0 <= task_index < 16:
            raise ValueError("dependent screen chunk requires array 0-15")
        return _screen_dependent_chunk(config, run, task_index)
    if stage == "screen-aggregate":
        return _aggregate_screen(config, run)
    raise ValueError(f"unsupported parallel route stage: {stage}")


__all__ = ["fold_rows", "run_stage", "screen_rows"]
