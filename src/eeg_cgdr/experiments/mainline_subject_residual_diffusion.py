"""Mainline subject-aware residual-diffusion experiment.

The module intentionally contains one model family, one information-matched
one-step comparator, and the minimum Klados/SGE execution and aggregation
stages required by the frozen protocol.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW

from eeg_cgdr.experiments.subject_artifact_data import (
    PreparedSubjectArtifactFold,
    RuntimeArtifactContext,
    prepare_subject_artifact_fold,
    validate_real_subject_artifact_inputs,
)
from eeg_cgdr.experiments.subject_artifact_development_eval import (
    _annotation_opener,
    _continuous,
)
from eeg_cgdr.experiments.sgeyesub_operator_specificity import _evaluate_output
from eeg_cgdr.models.subject_residual_diffusion import (
    BoundedResidual,
    OneStepResidualEstimator,
    PopulationAnchor,
    SubjectResidualConfig,
    SubjectResidualDiffusion,
    parameter_count,
)


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
PROTOCOL_ID = "mainline_subject_residual_diffusion_v1"
METHODS = ("RAW", "POP", "ONE-STEP-MATCH", "DIFF-POP", "DIFF-MATCH", "DIFF-SHUFFLED")
SEEDS = (20260811, 20260812, 20260813)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    if config.get("protocol_id") != PROTOCOL_ID or int(config.get("harness_level", -1)) != 1:
        raise ValueError("mainline protocol/harness changed")
    if tuple(int(value) for value in _mapping(config, "training")["seeds"]) != SEEDS:
        raise ValueError("the three initialization seeds changed")
    base_path = CODE_ROOT / str(config["base_subject_artifact_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base subject-artifact config is invalid")
    root = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    expected = CODE_ROOT / "results/cgdr/mainline_subject_residual_diffusion"
    if root != expected:
        raise ValueError("mainline output root changed")
    return base, root


def _implementation() -> dict[str, Any]:
    return {
        "git_sha": os.environ.get("DENOISENET_GIT_HEAD", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


def task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append({"task_index": len(rows), "dataset": "klados", "fold_index": 0, "seed": seed})
    for fold in range(25):
        for seed in SEEDS:
            rows.append({"task_index": len(rows), "dataset": "sgeyesub", "fold_index": fold, "seed": seed})
    if len(rows) != 78:
        raise AssertionError("mainline training task count changed")
    return rows


def _task(index: int) -> Mapping[str, Any]:
    rows = task_rows()
    if not 0 <= int(index) < len(rows):
        raise ValueError("task index must lie in [0,77]")
    return rows[int(index)]


def _prepared(base: Mapping[str, Any], dataset: str, fold: int) -> PreparedSubjectArtifactFold:
    if dataset == "sgeyesub":
        return prepare_subject_artifact_fold(base, fold)
    if dataset == "klados" and fold == 0:
        from eeg_cgdr.experiments.subject_artifact_klados_paired import prepare_klados_paired

        return prepare_klados_paired(base).prepared
    raise ValueError("unknown dataset/fold route")


def _training_arrays(prepared: PreparedSubjectArtifactFold) -> dict[str, np.ndarray]:
    source = prepared.training
    mean = prepared.latent_normalizer.mean[None, :, None]
    std = prepared.latent_normalizer.standard_deviation[None, :, None]
    physical = source.standardized_artifact_latent.astype(np.float64) * std + mean
    correction = np.einsum("nce,net->nct", source.normalized_transfer.astype(np.float64), physical)
    observed = source.observed.astype(np.float32)
    clean = (source.observed.astype(np.float64) - correction).astype(np.float32)
    population = np.repeat(
        prepared.population_context.full_transfer[None, :, :], observed.shape[0], axis=0
    ).astype(np.float32)
    return {
        "observed": observed,
        "clean": clean,
        "valid": source.valid_time_mask.astype(bool),
        "population_transfer": population,
        "subject_transfer": source.full_transfer.astype(np.float32),
        "rho": source.rho.astype(np.float32),
        "channel_mask": source.channel_mask.astype(bool),
    }


def _config(prepared: PreparedSubjectArtifactFold, config: Mapping[str, Any]) -> SubjectResidualConfig:
    model = _mapping(config, "model")
    diffusion = _mapping(config, "diffusion")
    return SubjectResidualConfig(
        eeg_channels=prepared.model_dimensions.eeg_channels,
        signal_length=prepared.model_dimensions.signal_length,
        base_channels=int(model["base_channels"]),
        num_timesteps=int(diffusion["timesteps"]),
        min_snr_gamma=float(diffusion["min_snr_gamma"]),
        context_dropout_probability=float(diffusion["context_dropout_probability"]),
        posterior_samples=int(diffusion["posterior_samples"]),
        ddim_steps=int(diffusion["ddim_steps"]),
    )


def _tensor(value: np.ndarray, device: torch.device, *, dtype: torch.dtype | None = None) -> Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    return {
        "observed": _tensor(arrays["observed"][indices], device, dtype=torch.float32),
        "clean": _tensor(arrays["clean"][indices], device, dtype=torch.float32),
        "valid_time_mask": _tensor(arrays["valid"][indices], device, dtype=torch.bool),
        "population_transfer": _tensor(arrays["population_transfer"][indices], device, dtype=torch.float32),
        "subject_transfer": _tensor(arrays["subject_transfer"][indices], device, dtype=torch.float32),
        "reliability": _tensor(arrays["rho"][indices], device, dtype=torch.float32),
        "channel_mask": _tensor(arrays["channel_mask"][indices], device, dtype=torch.bool),
    }


def _condition(batch: Mapping[str, Tensor], anchor: Tensor, present: Tensor) -> dict[str, Tensor]:
    return {
        "observed": batch["observed"],
        "population_anchor": anchor,
        "population_transfer": batch["population_transfer"],
        "subject_transfer": batch["subject_transfer"],
        "reliability": batch["reliability"],
        "channel_mask": batch["channel_mask"],
        "context_present": present,
        "valid_time_mask": batch["valid_time_mask"],
    }


def _masked_mse(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    weight = mask[:, None, :].to(predicted.dtype)
    return ((predicted - target).square() * weight).sum() / (weight.sum() * predicted.shape[1]).clamp_min(1)


def _split_indices(recording_keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    keys = sorted(set(recording_keys))
    validation_count = max(1, int(math.ceil(len(keys) * 0.2)))
    if len(keys) > 1:
        validation_keys = set(keys[-validation_count:])
        train = np.asarray([index for index, key in enumerate(recording_keys) if key not in validation_keys])
        validation = np.asarray([index for index, key in enumerate(recording_keys) if key in validation_keys])
    else:
        # Some exact cells contain one outer-training stem.  Keep source-level
        # separation when available; otherwise use fixed disjoint windows.
        all_indices = np.arange(len(recording_keys))
        validation = all_indices[::5]
        train = np.setdiff1d(all_indices, validation)
    if train.size == 0 or validation.size == 0:
        raise ValueError("training/validation split is empty")
    return train, validation


def _indices(generator: np.random.Generator, pool: np.ndarray, batch_size: int) -> np.ndarray:
    return generator.choice(pool, size=batch_size, replace=pool.size < batch_size)


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _train_task(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    base, root = _load(config)
    route = _task(task_index)
    existing = root / "checkpoints" / str(route["dataset"]) / f"fold_{int(route['fold_index']):02d}" / f"seed_{int(route['seed'])}/result_summary.json"
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("status") == "completed_mainline_training":
            return {**payload, "worker_resume_action": "skipped_already_completed"}
    prepared = _prepared(base, str(route["dataset"]), int(route["fold_index"]))
    arrays = _training_arrays(prepared)
    device = torch.device("cuda", 0)
    seed = int(route["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model_config = _config(prepared, config)
    anchor = PopulationAnchor(model_config).to(device)
    one_step = OneStepResidualEstimator(model_config).to(device)
    diffusion = SubjectResidualDiffusion(model_config).to(device)
    # Architectures are exactly parameter-matched because both comparators
    # instantiate the same residual backbone.
    if parameter_count(one_step) != parameter_count(diffusion):
        raise AssertionError("one-step and diffusion parameter counts differ")
    training = _mapping(config, "training")
    batch_size = int(training["batch_size"])
    train_indices, validation_indices = _split_indices(prepared.training.recording_keys)
    anchor_optimizer = AdamW(anchor.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    started = time.perf_counter()
    curve: list[dict[str, Any]] = []
    for step in range(1, int(training["anchor_updates"]) + 1):
        batch = _batch(arrays, _indices(rng, train_indices, batch_size), device)
        anchor_optimizer.zero_grad(set_to_none=True)
        predicted = anchor(batch["observed"], batch["valid_time_mask"])
        loss = _masked_mse(predicted, batch["clean"], batch["valid_time_mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(anchor.parameters(), float(training["gradient_clip_norm"]))
        anchor_optimizer.step()
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.append({"phase": "anchor", "step": step, "loss": float(loss.detach())})
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    # Freeze one shared residual scale after fitting the population anchor on
    # outer-training only.
    residual_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, train_indices.size, batch_size):
            batch = _batch(arrays, train_indices[start:start + batch_size], device)
            x_pop = anchor(batch["observed"], batch["valid_time_mask"])
            residual_chunks.append((batch["clean"] - x_pop).cpu().numpy())
    residual_values = np.concatenate(residual_chunks)
    thresholds = np.quantile(np.abs(residual_values), float(_mapping(config, "bounded_residual")["training_quantile"]), axis=(0, 2)).astype(np.float32)
    thresholds = np.maximum(thresholds, float(_mapping(config, "bounded_residual")["minimum_threshold"]))
    bound = BoundedResidual(torch.from_numpy(thresholds)).to(device)
    one_optimizer = AdamW(one_step.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    diff_optimizer = AdamW(diffusion.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    ema = {name: value.detach().clone() for name, value in diffusion.state_dict().items()}
    ema_decay = float(_mapping(config, "diffusion")["ema_decay"])
    best_score = float("inf")
    best_states: dict[str, Any] | None = None
    for step in range(1, int(training["comparator_updates"]) + 1):
        batch = _batch(arrays, _indices(rng, train_indices, batch_size), device)
        with torch.no_grad():
            x_pop = anchor(batch["observed"], batch["valid_time_mask"])
            residual_target = batch["clean"] - x_pop
        # One shared dropout draw is visible to both arms.
        present = (torch.rand(batch_size, device=device) >= model_config.context_dropout_probability).to(torch.float32)
        condition = _condition(batch, x_pop, present)
        one_optimizer.zero_grad(set_to_none=True)
        one_raw = one_step(**condition)
        one_pred, _ = bound(one_raw)
        one_loss = _masked_mse(one_pred, residual_target, batch["valid_time_mask"])
        one_loss.backward()
        torch.nn.utils.clip_grad_norm_(one_step.parameters(), float(training["gradient_clip_norm"]))
        one_optimizer.step()
        diff_optimizer.zero_grad(set_to_none=True)
        diff_loss, _ = diffusion.training_loss(residual_target, **condition)
        diff_loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), float(training["gradient_clip_norm"]))
        diff_optimizer.step()
        with torch.no_grad():
            for name, value in diffusion.state_dict().items():
                ema[name].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.extend([
                {"phase": "one_step", "step": step, "loss": float(one_loss.detach())},
                {"phase": "diffusion", "step": step, "loss": float(diff_loss.detach())},
            ])
        if step % int(training["validation_interval_updates"]) == 0 or step == int(training["comparator_updates"]):
            validation_batch = _batch(arrays, validation_indices, device)
            with torch.no_grad():
                x_pop = anchor(validation_batch["observed"], validation_batch["valid_time_mask"])
                target = validation_batch["clean"] - x_pop
                present_validation = torch.ones(target.shape[0], device=device)
                condition_validation = _condition(validation_batch, x_pop, present_validation)
                one_prediction, _ = bound(one_step(**condition_validation))
                one_validation = float(_masked_mse(one_prediction, target, validation_batch["valid_time_mask"]))
                # Diffusion validation uses fixed timesteps/noise through a
                # local RNG state so checkpoint selection is repeatable.
                state_before = torch.random.get_rng_state()
                torch.manual_seed(seed + 99173)
                diff_validation = float(diffusion.training_loss(target, **condition_validation)[0])
                torch.random.set_rng_state(state_before)
                score = one_validation + diff_validation
            curve.append({"phase": "validation", "step": step, "one_step_loss": one_validation, "diffusion_loss": diff_validation, "combined_score": score})
            if score < best_score:
                best_score = score
                best_states = {
                    "one_step": {name: value.detach().cpu().clone() for name, value in one_step.state_dict().items()},
                    "diffusion": {name: value.detach().cpu().clone() for name, value in diffusion.state_dict().items()},
                    "diffusion_ema": {name: value.detach().cpu().clone() for name, value in ema.items()},
                    "step": step,
                }
    if best_states is None:
        raise AssertionError("training did not produce a selected checkpoint")
    output = root / "checkpoints" / str(route["dataset"]) / f"fold_{int(route['fold_index']):02d}" / f"seed_{seed}"
    checkpoint = output / "models.pt"
    _save_checkpoint(checkpoint, {
        "protocol_id": PROTOCOL_ID,
        "route": dict(route),
        "model_config": model_config.__dict__,
        "anchor": {name: value.detach().cpu() for name, value in anchor.state_dict().items()},
        "one_step": best_states["one_step"],
        "diffusion": best_states["diffusion"],
        "diffusion_ema": best_states["diffusion_ema"],
        "thresholds": thresholds,
        "best_step": best_states["step"],
    })
    _write_csv(output / "training_curve.csv", curve)
    summary = {
        "status": "completed_mainline_training",
        **_implementation(),
        **dict(route),
        "checkpoint": str(checkpoint),
        "one_step_parameters": parameter_count(one_step),
        "diffusion_parameters": parameter_count(diffusion),
        "best_comparator_step": int(best_states["step"]),
        "runtime_seconds": time.perf_counter() - started,
        "query_EOG_or_labels_used": False,
    }
    _atomic_json(output / "result_summary.json", summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _train_worker(config: Mapping[str, Any], run_dir: Path, worker_index: int) -> Mapping[str, Any]:
    if not 0 <= int(worker_index) < 8:
        raise ValueError("training worker index must lie in [0,7]")
    indices = list(range(int(worker_index), 78, 8))
    results = [_train_task(config, run_dir / f"task_{index:02d}", index) for index in indices]
    summary = {
        "status": "completed_J2_worker",
        **_implementation(),
        "worker_index": int(worker_index),
        "task_indices": indices,
        "completed_task_count": len(results),
    }
    _atomic_json(run_dir / "worker_summary.json", summary)
    return summary


def _load_models(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, route: Mapping[str, Any], device: torch.device):
    _, root = _load(config)
    seed = int(route["seed"])
    checkpoint = root / "checkpoints" / str(route["dataset"]) / f"fold_{int(route['fold_index']):02d}" / f"seed_{seed}/models.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = SubjectResidualConfig(**payload["model_config"])
    anchor = PopulationAnchor(model_config).to(device)
    one = OneStepResidualEstimator(model_config).to(device)
    diffusion = SubjectResidualDiffusion(model_config).to(device)
    anchor.load_state_dict(payload["anchor"])
    one.load_state_dict(payload["one_step"])
    diffusion.load_state_dict(payload["diffusion_ema"])
    bound = BoundedResidual(torch.as_tensor(payload["thresholds"])).to(device)
    anchor.eval(); one.eval(); diffusion.eval(); bound.eval()
    return anchor, one, diffusion, bound, checkpoint


def _runtime_tensors(
    context: RuntimeArtifactContext,
    population: RuntimeArtifactContext,
    count: int,
    device: torch.device,
    *,
    reliability_override: float | None = None,
) -> dict[str, Tensor]:
    channels = context.full_transfer.shape[0]
    return {
        "population_transfer": torch.as_tensor(population.full_transfer, device=device, dtype=torch.float32)[None].expand(count, -1, -1),
        "subject_transfer": torch.as_tensor(context.full_transfer, device=device, dtype=torch.float32)[None].expand(count, -1, -1),
        "reliability": torch.full(
            (count,),
            float(context.rho if reliability_override is None else reliability_override),
            device=device,
        ),
        "channel_mask": torch.ones((count, channels), dtype=torch.bool, device=device),
    }


@torch.no_grad()
def _infer_six(anchor: PopulationAnchor, one: OneStepResidualEstimator, diffusion: SubjectResidualDiffusion, bound: BoundedResidual, *, observed: np.ndarray, valid: np.ndarray, population: RuntimeArtifactContext, matching: RuntimeArtifactContext, shuffled: RuntimeArtifactContext, seed: int, batch_size: int, device: torch.device) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, dict[str, float]]]:
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    clipping: dict[str, list[float]] = defaultdict(list)
    elapsed: dict[str, float] = defaultdict(float)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, start in enumerate(range(0, observed.shape[0], batch_size)):
        stop = min(start + batch_size, observed.shape[0])
        y = torch.as_tensor(observed[start:stop], device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid[start:stop], device=device, dtype=torch.bool)
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        x_pop = anchor(y, mask)
        if device.type == "cuda": torch.cuda.synchronize(device)
        anchor_elapsed = time.perf_counter() - started
        elapsed["POP"] += anchor_elapsed
        outputs["RAW"].append(y.cpu().numpy())
        outputs["POP"].append(x_pop.cpu().numpy())
        matching_context = _runtime_tensors(matching, population, y.shape[0], device)
        one_condition = {"observed": y, "population_anchor": x_pop, **matching_context, "context_present": torch.ones(y.shape[0], device=device), "valid_time_mask": mask}
        if device.type == "cuda": torch.cuda.synchronize(device)
        started = time.perf_counter()
        one_residual, one_fraction = bound(one(**one_condition))
        if device.type == "cuda": torch.cuda.synchronize(device)
        elapsed["ONE-STEP-MATCH"] += anchor_elapsed + time.perf_counter() - started
        outputs["ONE-STEP-MATCH"].append((x_pop + one_residual).cpu().numpy())
        clipping["ONE-STEP-MATCH"].append(float(one_fraction))
        base_seed = int(seed) * 1000003 + batch_index * 101
        sample_seeds = tuple(base_seed + value for value in range(8))
        for method, runtime, present in (
            ("DIFF-POP", population, 0.0),
            ("DIFF-MATCH", matching, 1.0),
            ("DIFF-SHUFFLED", shuffled, 1.0),
        ):
            # Every context arm retains the original query support reliability;
            # only C_s is replaced.  Population/null context therefore cannot
            # gain an artificial advantage from a different rho input.
            context = _runtime_tensors(
                runtime,
                population,
                y.shape[0],
                device,
                reliability_override=float(matching.rho),
            )
            condition = {"observed": y, "population_anchor": x_pop, **context, "context_present": torch.full((y.shape[0],), present, device=device), "valid_time_mask": mask}
            if device.type == "cuda": torch.cuda.synchronize(device)
            started = time.perf_counter()
            raw_residual, _calls = diffusion.sample(shape=tuple(y.shape), sample_seeds=sample_seeds, **condition)
            residual, fraction = bound(raw_residual)
            if device.type == "cuda": torch.cuda.synchronize(device)
            elapsed[method] += anchor_elapsed + time.perf_counter() - started
            outputs[method].append((x_pop + residual).cpu().numpy())
            clipping[method].append(float(fraction))
    peak = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0
    resources = {
        method: {
            "latency_seconds_per_window": float(elapsed.get(method, 0.0) / observed.shape[0]),
            "peak_memory_mb": 0.0 if method == "RAW" else peak,
            "function_evaluations_per_window": {
                "RAW": 0, "POP": 1, "ONE-STEP-MATCH": 2,
                "DIFF-POP": 401, "DIFF-MATCH": 401,
                "DIFF-SHUFFLED": 401,
            }[method],
        }
        for method in METHODS
    }
    return (
        {key: np.concatenate(value).astype(np.float32) for key, value in outputs.items()},
        {key: float(np.mean(value)) for key, value in clipping.items()},
        resources,
    )


def _paired_values(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return value.transpose(1, 0, 2)[:, mask].reshape(-1)


def _paired_metrics(observed: np.ndarray, clean: np.ndarray, output: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y = _paired_values(observed.astype(np.float64), mask)
    x = _paired_values(clean.astype(np.float64), mask)
    prediction = _paired_values(output.astype(np.float64), mask)
    artifact = y - x
    estimate = y - prediction
    epsilon = np.finfo(np.float64).eps
    noise_before = float(np.mean(np.square(y - x)))
    noise_after = float(np.mean(np.square(prediction - x)))
    return {
        "clean_waveform_RRMSE": float(np.linalg.norm(prediction - x) / max(np.linalg.norm(x), epsilon)),
        "clean_waveform_correlation": float(np.corrcoef(prediction, x)[0, 1]),
        "artifact_reconstruction_relative_error": float(np.linalg.norm(estimate - artifact) / max(np.linalg.norm(artifact), epsilon)),
        "delta_SNR_db": float(10.0 * np.log10(max(noise_before, epsilon) / max(noise_after, epsilon))),
        "output_input_RMS_ratio": float(np.sqrt(np.mean(prediction * prediction)) / max(np.sqrt(np.mean(y * y)), epsilon)),
        "observation_change_ratio": float(np.linalg.norm(prediction - y) / max(np.linalg.norm(y), epsilon)),
    }


def _klados_eval_records(base: Mapping[str, Any]):
    from eeg_cgdr.data.klados import load_klados_records
    from eeg_cgdr.data.mechanism import KLADOS_TRAIN_RECORDS, KLADOS_UNTOUCHED_RECORDS, fit_channel_normalizer, prepare_mechanism_record, select_records
    from eeg_cgdr.experiments.subject_artifact_klados_paired import EOG_ORDER, _fit_transfer, _loader_config, _mechanism_config, _raw_support_eog
    from eeg_cgdr.experiments.subject_artifact_data import _runtime, _support_rho

    mechanism_config = _mechanism_config(base)
    records = load_klados_records(_loader_config(mechanism_config))
    normalizer = fit_channel_normalizer(records, KLADOS_TRAIN_RECORDS)
    source_rate = int(_mapping(mechanism_config, "klados")["source_sampling_rate"])
    preprocessing = _mapping(mechanism_config, "preprocessing")
    target_rate = int(preprocessing["target_sampling_rate"])
    window_samples = int(preprocessing["window_samples"])
    selected = select_records(records, KLADOS_UNTOUCHED_RECORDS)
    result = []
    for record in selected:
        mechanism = prepare_mechanism_record(record, normalizer, source_rate=source_rate, target_rate=target_rate, window_samples=window_samples, calibration_seconds=10.0, guard_seconds=1.0)
        transfer = _fit_transfer(base, mechanism.calibration.eeg, _raw_support_eog(mechanism), fit_scope="support_only", fit_id=f"sim{record.record_id:02d}:support")
        shuffled_eog = np.roll(_raw_support_eog(mechanism), mechanism.calibration.eog.shape[1] // 2, axis=1)
        shuffled_transfer = _fit_transfer(base, mechanism.calibration.eeg, shuffled_eog, fit_scope="support_only", fit_id=f"sim{record.record_id:02d}:shuffled")
        rho = _support_rho(base, transfer)
        result.append((f"sim{record.record_id:02d}", mechanism, _runtime(transfer, role="matching", context_id=f"sim{record.record_id:02d}:matching", rho=rho, seconds=10.0, keys=(f"sim{record.record_id:02d}",)), _runtime(shuffled_transfer, role="shuffled_same_cell_severity_stratum", context_id=f"sim{record.record_id:02d}:shuffled", rho=rho, seconds=10.0, keys=(f"sim{record.record_id:02d}",))))
    return result


def _evaluate_klados(config: Mapping[str, Any], run_dir: Path, seed_index: int) -> Mapping[str, Any]:
    base, root = _load(config)
    route = task_rows()[seed_index]
    if route["dataset"] != "klados":
        raise ValueError("Klados evaluation index must be 0,1,2")
    output = root / "evaluation/klados" / f"seed_{route['seed']}"
    existing = output / "result_summary.json"
    if existing.is_file() and (output / "metrics.csv").is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("status") == "completed_klados_full_evaluation":
            return {**payload, "worker_resume_action": "skipped_already_completed"}
    from eeg_cgdr.experiments.subject_artifact_klados_paired import prepare_klados_paired

    paired = prepare_klados_paired(base)
    prepared = paired.prepared
    device = torch.device("cuda", 0)
    anchor, one, diffusion, bound, checkpoint = _load_models(config, prepared, route, device)
    rows: list[dict[str, Any]] = []
    example_saved = False
    for key, mechanism, matching, shuffled in _klados_eval_records(base):
        outputs, clipping, resources = _infer_six(
            anchor, one, diffusion, bound,
            observed=mechanism.observed_windows.astype(np.float32),
            valid=mechanism.valid_time_weight.astype(bool),
            population=prepared.population_context,
            matching=matching,
            shuffled=shuffled,
            seed=int(route["seed"]),
            batch_size=int(_mapping(config, "evaluation")["batch_size"]),
            device=device,
        )
        for method in METHODS:
            rows.append({
                "dataset": "klados", "unit_id": key, "exact_cell": prepared.fold.layout_id,
                "training_seed": int(route["seed"]), "method": method, "status": "success",
                **_paired_metrics(mechanism.observed_windows, mechanism.clean_windows, outputs[method], mechanism.valid_time_weight.astype(bool)),
                "clipping_fraction": clipping.get(method, 0.0),
                **resources[method],
                "statistical_unit": "source_record", "participant_claim": False,
            })
        if not example_saved and seed_index == 0:
            example = root / "server_arrays/klados_representative_sim37.npz"
            example.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(example, raw=mechanism.observed_windows[0], clean=mechanism.clean_windows[0], one_step=outputs["ONE-STEP-MATCH"][0], diffusion=outputs["DIFF-MATCH"][0])
            example_saved = True
    _write_csv(output / "metrics.csv", rows)
    summary = {"status": "completed_klados_full_evaluation", **_implementation(), "training_seed": route["seed"], "source_record_count": len(set(row["unit_id"] for row in rows)), "method_rows": len(rows), "checkpoint": str(checkpoint)}
    _atomic_json(output / "result_summary.json", summary); _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _sge_samples_per_trial(sampling_rate_hz: float) -> int:
    """The frozen release-internal SGE protocol uses eight-second trials."""

    value = int(round(8.0 * float(sampling_rate_hz)))
    if value <= 0:
        raise ValueError("SGE sampling rate must produce a positive trial length")
    return value


def _evaluate_sge(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    base, root = _load(config)
    route = _task(task_index + 3)
    if route["dataset"] != "sgeyesub":
        raise ValueError("SGE evaluation task must lie in [0,74]")
    output = root / "evaluation/sgeyesub" / f"fold_{int(route['fold_index']):02d}" / f"seed_{route['seed']}"
    existing = output / "result_summary.json"
    if existing.is_file() and (output / "metrics.csv").is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("status") == "completed_sge_full_evaluation":
            return {**payload, "worker_resume_action": "skipped_already_completed"}
    prepared = prepare_subject_artifact_fold(base, int(route["fold_index"]))
    device = torch.device("cuda", 0)
    anchor, one, diffusion, bound, checkpoint = _load_models(config, prepared, route, device)
    rows: list[dict[str, Any]] = []
    server_root = root / "server_arrays/sgeyesub" / f"fold_{int(route['fold_index']):02d}" / f"seed_{route['seed']}"
    server_root.mkdir(parents=True, exist_ok=True)
    for key, heldout in prepared.heldout.items():
        outputs, clipping, resources = _infer_six(
            anchor, one, diffusion, bound,
            observed=heldout.query.observed, valid=heldout.query.valid_time_mask,
            population=prepared.population_context, matching=heldout.matching,
            # The minimal negative control is a shuffled-subject context: a
            # support-derived C from another same-cell outer-training stem.
            # The legacy within-stem EOG permutation is deliberately not used.
            shuffled=heldout.wrong_same_cell, seed=int(route["seed"]),
            batch_size=int(_mapping(config, "evaluation")["batch_size"]), device=device,
        )
        # Freeze every method output before the evaluation-only EOG/annotation boundary.
        archive = server_root / f"{key.replace('/', '__')}_outputs.npz"
        np.savez_compressed(archive, **{name.replace("-", "_"): value for name, value in outputs.items()})
        annotated = _annotation_opener(base, prepared, key)()
        if annotated.query is None or annotated.query_annotations is None:
            raise AssertionError("SGE scoring fields were not opened after output freeze")
        observed_continuous = _continuous(heldout.query.observed)
        annotations = annotated.query_annotations
        for method in METHODS:
            output_continuous = _continuous(outputs[method])
            metric = _evaluate_output(
                method_id=method, output=output_continuous, observed=observed_continuous,
                matching_projector=heldout.matching.projector,
                population_projector=prepared.population_context.projector,
                query_eog=annotations.external_eog,
                artifactclasses=annotations.artifactclasses,
                predicted_contamination=None,
                trial_labels=annotations.trial_labels,
                samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz),
                minimum_trials_per_condition=2,
                status="success", operator_source="soft_support_context_only",
                gamma=None, fallback_used=False, uses_query_external_eog=False,
            )
            input_rms = np.sqrt(np.mean(np.square(observed_continuous)))
            output_rms = np.sqrt(np.mean(np.square(output_continuous)))
            rows.append({
                "dataset": "sgeyesub", "unit_id": key,
                "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}",
                "study": prepared.fold.study, "layout_id": prepared.fold.layout_id,
                "training_seed": int(route["seed"]), "method": method,
                **metric,
                "output_input_RMS_ratio": float(output_rms / max(input_rms, np.finfo(np.float64).eps)),
                "clipping_fraction": clipping.get(method, 0.0),
                **resources[method],
                "outputs_frozen_before_scoring": True,
            })
    _write_csv(output / "metrics.csv", rows)
    summary = {"status": "completed_sge_full_evaluation", **_implementation(), **dict(route), "heldout_stems": len(prepared.heldout), "method_rows": len(rows), "checkpoint": str(checkpoint)}
    _atomic_json(output / "result_summary.json", summary); _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _evaluate_sge_worker(config: Mapping[str, Any], run_dir: Path, worker_index: int) -> Mapping[str, Any]:
    if not 0 <= int(worker_index) < 8:
        raise ValueError("SGE worker index must lie in [0,7]")
    indices = list(range(int(worker_index), 75, 8))
    results = [_evaluate_sge(config, run_dir / f"task_{index:02d}", index) for index in indices]
    summary = {
        "status": "completed_J4_worker",
        **_implementation(),
        "worker_index": int(worker_index),
        "task_indices": indices,
        "completed_task_count": len(results),
    }
    _atomic_json(run_dir / "worker_summary.json", summary)
    return summary


def _read_csvs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    return rows


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group in groups.items():
        result = {key: value for key, value in zip(keys, group_key)}
        numeric_keys = set.intersection(*[
            {key for key, value in row.items() if key not in keys and _is_float(value)} for row in group
        ]) if group else set()
        for key in numeric_keys:
            values = [float(row[key]) for row in group if math.isfinite(float(row[key]))]
            if values:
                result[key] = float(np.mean(values))
        result["seed_count"] = len(set(str(row.get("training_seed")) for row in group))
        output.append(result)
    return output


def _is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _paired_effects(rows: Sequence[Mapping[str, Any]], *, metric: str, left: str, right: str, utility_sign: float) -> list[dict[str, Any]]:
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_unit[str(row["unit_id"])][str(row["method"])] = row
    result = []
    for unit, methods in sorted(by_unit.items()):
        if left in methods and right in methods:
            result.append({"unit_id": unit, "exact_cell": methods[left].get("exact_cell", ""), "effect": utility_sign * (float(methods[left][metric]) - float(methods[right][metric]))})
    return result


def _bootstrap(effects: Sequence[Mapping[str, Any]], *, stratified: bool, seed: int, replicates: int) -> dict[str, Any]:
    if not effects:
        return {"mean": None, "median": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    values = np.asarray([float(row["effect"]) for row in effects])
    samples = np.empty(replicates, dtype=np.float64)
    if stratified:
        strata: dict[str, np.ndarray] = {}
        for cell in sorted(set(str(row["exact_cell"]) for row in effects)):
            strata[cell] = np.asarray([float(row["effect"]) for row in effects if str(row["exact_cell"]) == cell])
        for index in range(replicates):
            draw = np.concatenate([rng.choice(value, size=value.size, replace=True) for value in strata.values()])
            samples[index] = draw.mean()
    else:
        for index in range(replicates):
            samples[index] = rng.choice(values, size=values.size, replace=True).mean()
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "n": int(values.size)}


def _sge_coverage(base: Mapping[str, Any], successful_units: int) -> dict[str, Any]:
    """Retain preblocked stems in feasibility, never in performance means."""

    sge = _mapping(_mapping(base, "data"), "sgeyesub")
    compatible = int(sge["compatible_stems"])
    blocked = [str(value) for value in sge.get("blocked_stems", ())]
    if successful_units != compatible:
        raise ValueError(
            f"SGE compatible coverage changed: success={successful_units}, expected={compatible}"
        )
    return {
        "available_participant_stems": compatible + len(blocked),
        "successful_compatible_participant_stems": successful_units,
        "blocked_participant_stems": len(blocked),
        "blocked_recording_keys": blocked,
    }


def _aggregate(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    klados_paths = sorted((root / "evaluation/klados").glob("seed_*/metrics.csv"))
    sge_paths = sorted((root / "evaluation/sgeyesub").glob("fold_*/seed_*/metrics.csv"))
    if len(klados_paths) != 3 or len(sge_paths) != 75:
        raise ValueError(f"incomplete evaluation coverage: Klados={len(klados_paths)}, SGE={len(sge_paths)}")
    raw_rows = _read_csvs([*klados_paths, *sge_paths])
    units = _mean_rows(raw_rows, ("dataset", "unit_id", "exact_cell", "method"))
    _write_csv(root / "unit_metrics.csv", units)
    method_summary = _mean_rows(units, ("dataset", "method"))
    _write_csv(root / "method_summary.csv", method_summary)
    klados = [row for row in units if row["dataset"] == "klados"]
    sge = [row for row in units if row["dataset"] == "sgeyesub"]
    h1_effects = _paired_effects(klados, metric="clean_waveform_RRMSE", left="DIFF-MATCH", right="ONE-STEP-MATCH", utility_sign=-1.0)
    h2_pop_effects = _paired_effects(sge, metric="eog_coherence_reduction", left="DIFF-MATCH", right="DIFF-POP", utility_sign=1.0)
    h2_shuffle_effects = _paired_effects(sge, metric="eog_coherence_reduction", left="DIFF-MATCH", right="DIFF-SHUFFLED", utility_sign=1.0)
    effect_rows = [
        *({**row, "estimand": "H1_DIFF_MATCH_minus_ONE_STEP_MATCH_clean_RRMSE_utility"} for row in h1_effects),
        *({**row, "estimand": "H2_DIFF_MATCH_minus_DIFF_POP_EOG_coherence_utility"} for row in h2_pop_effects),
        *({**row, "estimand": "H2_DIFF_MATCH_minus_DIFF_SHUFFLED_EOG_coherence_utility"} for row in h2_shuffle_effects),
    ]
    _write_csv(root / "paired_effects.csv", effect_rows)
    statistics = _mapping(config, "statistics")
    replicates = int(statistics["bootstrap_replicates"])
    bootstrap = {
        "H1": _bootstrap(h1_effects, stratified=False, seed=int(statistics["bootstrap_seed"]), replicates=replicates),
        "H2_population": _bootstrap(h2_pop_effects, stratified=True, seed=int(statistics["bootstrap_seed"]) + 1, replicates=replicates),
        "H2_shuffled": _bootstrap(h2_shuffle_effects, stratified=True, seed=int(statistics["bootstrap_seed"]) + 2, replicates=replicates),
    }
    margin = float(_mapping(config, "science_gates")["preservation_noninferiority_margin"])
    safety_specs = (
        ("nonartifact_observation_preservation", 1.0),
        ("condition_erp_observation_relative_preservation", 1.0),
        ("reference_free_psd_distortion", -1.0),
        ("reference_free_covariance_distortion", -1.0),
    )
    safety = {}
    for metric, sign in safety_specs:
        effects = _paired_effects(sge, metric=metric, left="DIFF-MATCH", right="POP", utility_sign=sign)
        summary = _bootstrap(effects, stratified=True, seed=int(statistics["bootstrap_seed"]) + len(safety) + 10, replicates=replicates)
        summary["noninferiority_passed"] = summary["ci95"][0] is not None and float(summary["ci95"][0]) >= margin
        safety[metric] = summary
    rms = [float(row["output_input_RMS_ratio"]) for row in sge if row["method"] == "DIFF-MATCH"]
    h1 = bootstrap["H1"]["ci95"][0] is not None and bootstrap["H1"]["ci95"][0] > 0
    h2 = all(bootstrap[key]["ci95"][0] is not None and bootstrap[key]["ci95"][0] > 0 for key in ("H2_population", "H2_shuffled"))
    h3 = all(value["noninferiority_passed"] for value in safety.values()) and bool(rms) and max(rms) <= float(_mapping(config, "science_gates")["maximum_output_input_RMS_ratio"])
    if h1 and h2 and h3:
        verdict = "mainline_subject_aware_residual_diffusion_supported_in_development"
    elif h1:
        verdict = "generic_residual_diffusion_supported_subject_aware_mainline_not_supported"
    elif h2:
        verdict = "support_context_useful_diffusion_increment_not_supported"
    else:
        verdict = "current_single_residual_diffusion_configuration_not_supported"
    bootstrap_payload = {"effects": bootstrap, "safety": safety, "replicates": replicates}
    _atomic_json(root / "bootstrap_summary.json", bootstrap_payload)
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    _plot_effects(figures / "h1_h2_paired_effect_forest.png", bootstrap)
    _plot_waveform(root, figures / "klados_representative_waveform.png")
    _plot_sge(sge, figures / "sge_artifact_reduction_vs_preservation.png")
    successful_sge_units = len(set(row["unit_id"] for row in sge))
    summary = {
        "status": "completed_mainline_aggregation", **_implementation(),
        "verdict": verdict, "H1_diffusion_utility": bool(h1),
        "H2_subject_awareness": bool(h2), "H3_natural_EEG_safety": bool(h3),
        "effects": bootstrap, "safety": safety,
        "coverage": {
            "klados_source_records": len(set(row["unit_id"] for row in klados)),
            "sge_participant_stems": successful_sge_units,
            **_sge_coverage(base, successful_sge_units),
            "klados_seed_files": len(klados_paths),
            "sge_fold_seed_files": len(sge_paths),
        },
        "scientific_scope": "Klados source-record paired evidence plus SGE participant/stem development evidence; not untouched confirmation",
    }
    _atomic_json(root / "result_summary.json", summary)
    _write_report(root, summary, method_summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _plot_effects(path: Path, bootstrap: Mapping[str, Mapping[str, Any]]) -> None:
    labels = ["H1: Diff−One", "H2: Match−Pop", "H2: Match−Shuffle"]
    values = [bootstrap[key]["mean"] for key in ("H1", "H2_population", "H2_shuffled")]
    intervals = [bootstrap[key]["ci95"] for key in ("H1", "H2_population", "H2_shuffled")]
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.errorbar(values, range(3), xerr=[[value - interval[0] for value, interval in zip(values, intervals)], [interval[1] - value for value, interval in zip(values, intervals)]], fmt="o")
    axis.axvline(0.0, color="black", linewidth=1); axis.set_yticks(range(3), labels); axis.set_xlabel("Paired utility effect (positive is better)"); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_waveform(root: Path, path: Path) -> None:
    data = np.load(root / "server_arrays/klados_representative_sim37.npz")
    figure, axis = plt.subplots(figsize=(10, 4))
    samples = min(512, data["raw"].shape[1])
    for name, color in (("raw", "0.6"), ("clean", "black"), ("one_step", "tab:blue"), ("diffusion", "tab:orange")):
        axis.plot(data[name][0, :samples], label=name, color=color, linewidth=1)
    axis.legend(ncol=4); axis.set_xlabel("Sample"); axis.set_ylabel("Normalized EEG"); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_sge(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    colors = {"POP": "tab:gray", "ONE-STEP-MATCH": "tab:blue", "DIFF-POP": "tab:green", "DIFF-MATCH": "tab:orange", "DIFF-SHUFFLED": "tab:red"}
    for method, color in colors.items():
        selected = [row for row in rows if row["method"] == method]
        axis.scatter([float(row["nonartifact_observation_preservation"]) for row in selected], [float(row["eog_coherence_reduction"]) for row in selected], s=14, alpha=0.55, label=method, color=color)
    axis.set_xlabel("Non-artifact preservation"); axis.set_ylabel("EOG coherence reduction"); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def _write_report(root: Path, summary: Mapping[str, Any], method_summary: Sequence[Mapping[str, Any]]) -> None:
    report = CODE_ROOT / "reports/mainline_subject_residual_diffusion.md"
    effects = _mapping(summary, "effects")
    lines = [
        "# Mainline subject-aware residual diffusion", "",
        f"Scientific decision: **{summary['verdict']}**.", "",
        "| Question | Passed | Mean effect | 95% interval |",
        "|---|---:|---:|---:|",
        f"| H1 DIFF-MATCH vs ONE-STEP-MATCH | {summary['H1_diffusion_utility']} | {effects['H1']['mean']:.6f} | [{effects['H1']['ci95'][0]:.6f}, {effects['H1']['ci95'][1]:.6f}] |",
        f"| H2 DIFF-MATCH vs DIFF-POP | {summary['H2_subject_awareness']} | {effects['H2_population']['mean']:.6f} | [{effects['H2_population']['ci95'][0]:.6f}, {effects['H2_population']['ci95'][1]:.6f}] |",
        f"| H2 DIFF-MATCH vs DIFF-SHUFFLED | {summary['H2_subject_awareness']} | {effects['H2_shuffled']['mean']:.6f} | [{effects['H2_shuffled']['ci95'][0]:.6f}, {effects['H2_shuffled']['ci95'][1]:.6f}] |",
        f"| H3 natural EEG safety | {summary['H3_natural_EEG_safety']} | — | frozen -0.02 margins |", "",
        "All six methods used the same query per unit. DIFF-POP, DIFF-MATCH and DIFF-SHUFFLED shared one checkpoint and common random numbers; query EOG and annotations were opened only after outputs were frozen.", "",
        f"Coverage: {summary['coverage']['klados_source_records']} Klados evaluation source records; "
        f"{summary['coverage']['successful_compatible_participant_stems']}/{summary['coverage']['available_participant_stems']} SGE participant stems produced performance results, with "
        f"{summary['coverage']['blocked_participant_stems']} preblocked singleton retained only in the feasibility denominator. Klados is paired source-record evidence; SGE is participant/stem development evidence, not untouched confirmation.", "",
        "## Six-method compact results", "",
        "| Dataset | Method | Clean RRMSE | EOG coherence reduction | Non-artifact preservation | Output/input RMS | Latency/window (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in method_summary:
        values = []
        for key in (
            "clean_waveform_RRMSE", "eog_coherence_reduction",
            "nonartifact_observation_preservation", "output_input_RMS_ratio",
            "latency_seconds_per_window",
        ):
            value = row.get(key)
            values.append("—" if value is None else f"{float(value):.6g}")
        lines.append(
            f"| {row['dataset']} | {row['method']} | " + " | ".join(values) + " |"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _technical_check(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    if not torch.cuda.is_available():
        raise RuntimeError("technical check requires scheduled CUDA")
    prepared = _prepared(base, "klados", 0)
    arrays = _training_arrays(prepared)
    device = torch.device("cuda", 0)
    cfg = _config(prepared, config)
    anchor = PopulationAnchor(cfg).to(device)
    one = OneStepResidualEstimator(cfg).to(device)
    diffusion = SubjectResidualDiffusion(cfg).to(device)
    bound = BoundedResidual(torch.full((cfg.eeg_channels,), 3.0)).to(device)
    indices = np.arange(min(4, arrays["observed"].shape[0]))
    batch = _batch(arrays, indices, device)
    x_pop = anchor(batch["observed"], batch["valid_time_mask"])
    condition = _condition(batch, x_pop, torch.ones(len(indices), device=device))
    target = batch["clean"] - x_pop.detach()
    one_raw = one(**condition); one_bounded, one_clip = bound(one_raw)
    diff_loss, _ = diffusion.training_loss(target, **condition)
    loss = _masked_mse(one_bounded, target, batch["valid_time_mask"]) + diff_loss
    loss.backward()
    resume_optimizer = AdamW(diffusion.parameters(), lr=1.0e-4)
    resume_optimizer.step()
    resume_optimizer.zero_grad(set_to_none=True)
    seeds = tuple(881000 + value for value in range(8))
    diffusion.eval()
    sampled, calls = diffusion.sample(shape=tuple(target.shape), sample_seeds=seeds, **condition)
    sampled_bounded, diff_clip = bound(sampled)
    population_condition = dict(condition); population_condition["subject_transfer"] = population_condition["population_transfer"]; population_condition["context_present"] = torch.zeros(len(indices), device=device)
    matching_again, _ = diffusion.sample(shape=tuple(target.shape), sample_seeds=seeds, **condition)
    population_sample, _ = diffusion.sample(shape=tuple(target.shape), sample_seeds=seeds, **population_condition)
    change = float(torch.linalg.vector_norm(sampled - population_sample) / torch.linalg.vector_norm(sampled).clamp_min(1e-8))
    repeat = float(torch.linalg.vector_norm(sampled - matching_again))
    checkpoint = root / "technical_check/checkpoint.pt"
    _save_checkpoint(checkpoint, {"model": diffusion.state_dict(), "optimizer": resume_optimizer.state_dict(), "completed_updates": 1})
    reloaded = SubjectResidualDiffusion(cfg).to(device)
    reloaded_optimizer = AdamW(reloaded.parameters(), lr=1.0e-4)
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(loaded["model"])
    reloaded_optimizer.load_state_dict(loaded["optimizer"])
    resume_ok = loaded["completed_updates"] == 1 and len(reloaded_optimizer.state_dict()["state"]) > 0
    checks = {
        "target_reconstruction_finite": bool(torch.isfinite(target).all()),
        "output_and_gradient_finite": bool(torch.isfinite(loss) and all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in list(one.parameters()) + list(diffusion.parameters()))),
        "checkpoint_save_reload_resume": resume_ok and all(torch.equal(left, right) for left, right in zip(diffusion.state_dict().values(), reloaded.state_dict().values())),
        "parameter_count_matched": parameter_count(one) == parameter_count(diffusion),
        "common_random_repeat_exact": repeat == 0.0,
        "ddim_network_calls": calls == 8 * 50,
        "bounded_not_zero": float(sampled_bounded.abs().max()) > 0 and float(diff_clip) < 1.0,
        "scale_safe": float(torch.sqrt(sampled_bounded.square().mean()) / torch.sqrt(batch["observed"].square().mean()).clamp_min(1e-8)) < 10.0,
        "context_swap_used": change > 1e-7,
        "query_eog_label_absent": set(SubjectResidualDiffusion.forbidden_input_fields).isdisjoint(SubjectResidualDiffusion.visible_input_fields),
    }
    result = {"status": "passed_technical_check" if all(checks.values()) else "failed_technical_check", **_implementation(), "checks": checks, "context_swap_relative_change": change, "one_step_clipping_fraction": float(one_clip), "diffusion_clipping_fraction": float(diff_clip), "one_step_parameters": parameter_count(one), "diffusion_parameters": parameter_count(diffusion)}
    _atomic_json(root / "technical_check/result_summary.json", result); _atomic_json(run_dir / "result_summary.json", result)
    if not all(checks.values()):
        raise RuntimeError("mainline technical check failed")
    return result


def run_stage(config: Mapping[str, Any], run_dir: str | Path, stage: str, task_index: int | None) -> Mapping[str, Any]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    base, root = _load(config)
    if stage == "j0":
        if task_index is not None:
            raise ValueError("J0 rejects arrays")
        validation = validate_real_subject_artifact_inputs(base)
        _write_csv(root / "task_list.csv", task_rows())
        result = {"status": "completed_J0", **_implementation(), "task_count": 78, "sge_fold_count": 25, "seeds": list(SEEDS), "real_input_validation": validation["status"], "historical_results_modified": False}
        _atomic_json(root / "j0_summary.json", result); _atomic_json(run / "result_summary.json", result)
        return result
    if stage == "j1":
        if task_index is not None: raise ValueError("J1 rejects arrays")
        return _technical_check(config, run)
    if stage == "j2-train":
        if task_index is None: raise ValueError("J2 requires an array")
        return _train_task(config, run, task_index)
    if stage == "j2-worker":
        if task_index is None: raise ValueError("J2 worker requires an array")
        return _train_worker(config, run, task_index)
    if stage == "j3-klados":
        if task_index is None: raise ValueError("J3 requires an array")
        return _evaluate_klados(config, run, task_index)
    if stage == "j4-sge":
        if task_index is None: raise ValueError("J4 requires an array")
        return _evaluate_sge(config, run, task_index)
    if stage == "j4-worker":
        if task_index is None: raise ValueError("J4 worker requires an array")
        return _evaluate_sge_worker(config, run, task_index)
    if stage == "j5-aggregate":
        if task_index is not None: raise ValueError("J5 rejects arrays")
        return _aggregate(config, run)
    if stage == "j6-finalize":
        if task_index is not None: raise ValueError("J6 rejects arrays")
        summary = json.loads((root / "result_summary.json").read_text(encoding="utf-8"))
        result = {"status": "completed_J6", **_implementation(), "scientific_verdict": summary["verdict"], "result_path": str(root), "report": str(CODE_ROOT / "reports/mainline_subject_residual_diffusion.md")}
        _atomic_json(root / "terminal_manifest.json", result); _atomic_json(run / "result_summary.json", result)
        return result
    raise ValueError(f"unsupported mainline stage: {stage}")


__all__ = ["run_stage", "task_rows"]
