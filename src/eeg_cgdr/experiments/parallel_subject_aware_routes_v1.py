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
    _prepared,
    _split_indices,
)
from eeg_cgdr.experiments.subject_artifact_data import PreparedSubjectArtifactFold
from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)
from eeg_cgdr.models.parallel_subject_routes import (
    AdaptiveActivityGate,
    FullCFiLMDiffusion,
    canonical_target,
    full_c_population_residual_reconstruction,
    guided_latent_step,
    sdedit_initial_latent,
)


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_parallel_explore"))
PROTOCOL = "parallel_subject_aware_routes_v1"
SCREEN_SEED = 20260811
ROUTES = ("P1_FULL_C_RESIDUAL", "P2_FULL_C_FILM", "P3_ACTIVITY_GATE", "P4_SUPPORT_ADAPTER", "P5_POSTERIOR_GUIDANCE", "P6_ANCHORED_SDEDIT")


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


def _checkpoint_path(root: Path, row: Mapping[str, Any]) -> Path:
    return root / "checkpoints/P0" / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}" / f"seed_{int(row['seed'])}" / "model.pt"


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
    training = _mapping(config, "training")
    optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train_indices, validation_indices = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    start = 0; curve: list[dict[str, Any]] = []
    resume = checkpoint.parent / "resume.pt"
    if resume.is_file():
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
        ema = {name: value.to(device) for name, value in payload["ema"].items()}
        start = int(payload["step"]); curve = list(payload.get("curve", ()))
        rng.bit_generator.state = payload["numpy_state"]; generator.set_state(payload["cuda_generator_state"])
    batch_size = int(training["batch_size"])
    maximum = int(training["maximum_updates"])
    best_score = float("inf"); best_state: dict[str, Tensor] | None = None; best_step = 0
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
        decay = float(_mapping(config, "diffusion")["ema_decay"])
        with torch.no_grad():
            for name, value in model.state_dict().items():
                ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(gradient), **{key: float(value) for key, value in diagnostics.items()}})
        if step % int(training["validation_interval_updates"]) == 0 or step == maximum:
            validation_target, validation_condition = _batch(arrays, validation_indices, device)
            score = _ema_validation(model, ema, validation_target, validation_condition, seed + 99001)
            curve.append({"step": step, "ema_validation_x0_mse": score})
            if score < best_score:
                best_score, best_step = score, step
                best_state = {name: value.detach().cpu().clone() for name, value in ema.items()}
        if step % int(training["checkpoint_interval_updates"]) == 0:
            _save_checkpoint(resume, {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema, "curve": curve, "numpy_state": rng.bit_generator.state, "cuda_generator_state": generator.get_state()})
    if best_state is None:
        raise AssertionError("EMA validation did not select a P0 checkpoint")
    _save_checkpoint(checkpoint, {"protocol_id": PROTOCOL, "route": dict(row), "model_config": model_config.__dict__, "diffusion_config": diffusion_config.__dict__, "diffusion_ema": best_state, "best_step": best_step, "validation_x0_mse": best_score, "latent_mean": prepared.latent_normalizer.mean, "latent_standard_deviation": prepared.latent_normalizer.standard_deviation, "population_context": prepared.population_context})
    _write_csv(checkpoint.parent / "training_curve.csv", curve)
    summary = {"status": "completed_P0_population_training", **_implementation(), **dict(row), "checkpoint": str(checkpoint), "best_step": best_step, "validation_x0_mse": best_score, "runtime_seconds": time.perf_counter() - started, "target": "fixed_standardized_artifact_latent", "EMA_used_for_validation_and_checkpoint": True}
    _write_json(summary_path, summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def _train_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    tasks = fold_rows()
    indices = list(range(worker, len(tasks), 8))
    results = [_train_p0(config, run_dir / f"task_{index:03d}", tasks[index]) for index in indices]
    summary = {"status": "completed_P0_worker", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results)}
    _write_json(run_dir / "worker_summary.json", summary)
    return summary


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
    population = torch.as_tensor(prepared.population_context.full_transfer, dtype=torch.float32)
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
    raise ValueError(f"unsupported parallel route stage: {stage}")


__all__ = ["fold_rows", "run_stage", "screen_rows"]
