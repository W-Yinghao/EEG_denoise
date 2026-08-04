"""TAAS experiment for support-calibrated artifact-subspace diffusion.

This runner deliberately reuses the frozen Klados/SGE data surfaces and keeps
one new diffusion family, one information-matched deterministic comparator,
and the minimum intervention/aggregation code required by the paper.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
import urllib.request
import inspect
import shutil
import subprocess
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

from eeg_cgdr.experiments.mainline_subject_residual_diffusion import (
    SEEDS,
    _annotation_opener,
    _continuous,
    _evaluate_output,
    _klados_eval_records,
    _paired_metrics,
    _prepared,
    _sge_samples_per_trial,
    _split_indices,
    _training_arrays,
)
from eeg_cgdr.experiments.subject_artifact_data import (
    PreparedSubjectArtifactFold,
    RuntimeArtifactContext,
    prepare_subject_artifact_fold,
    validate_real_subject_artifact_inputs,
)
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig,
    ArtifactSubspaceDiffusion,
    DeterministicSubspaceEstimator,
    aligned_artifact_basis,
    batched_aligned_bases,
    complement_consistency_error,
    parameter_count,
    participant_sample_seeds,
    reconstruct_from_subspace,
    training_tau,
)
from eeg_cgdr.models.subject_residual_diffusion import PopulationAnchor


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
PROTOCOL = "support_calibrated_artifact_subspace_diffusion_v1"
METHODS = (
    "RAW", "POP", "DIFF-POP", "DIFF-MATCH", "DIFF-WRONG-SAME-CELL", "DET-MATCH",
)
ABLATIONS = ("DIFF-MATCH-K1", "DIFF-MATCH-NO-IDENTITY")


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
    if config.get("protocol_id") != PROTOCOL or int(config.get("harness_level", -1)) != 1:
        raise ValueError("artifact-subspace protocol/harness changed")
    base = yaml.safe_load((CODE_ROOT / str(config["base_subject_artifact_config"])).read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base data protocol is invalid")
    root = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    return base, root


def _implementation() -> dict[str, Any]:
    return {
        "git_sha": os.environ.get("DENOISENET_GIT_HEAD", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


def task_rows() -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        rows.append({"task_index": len(rows), "dataset": "klados", "fold_index": 0, "seed": seed})
    for fold in range(25):
        for seed in SEEDS:
            rows.append({"task_index": len(rows), "dataset": "sgeyesub", "fold_index": fold, "seed": seed})
    if len(rows) != 78:
        raise AssertionError("training task count changed")
    return rows


def _task(index: int) -> Mapping[str, Any]:
    rows = task_rows()
    if not 0 <= index < len(rows):
        raise ValueError("task index must lie in [0,77]")
    return rows[index]


def _model_config(prepared: PreparedSubjectArtifactFold, config: Mapping[str, Any]) -> ArtifactSubspaceConfig:
    return ArtifactSubspaceConfig(
        eeg_channels=prepared.model_dimensions.eeg_channels,
        signal_length=prepared.model_dimensions.signal_length,
        base_channels=int(_mapping(config, "model")["base_channels"]),
        num_timesteps=int(_mapping(config, "diffusion")["timesteps"]),
        min_snr_gamma=float(_mapping(config, "diffusion")["min_snr_gamma"]),
        ddim_steps=int(_mapping(config, "diffusion")["ddim_steps"]),
        posterior_samples=int(_mapping(config, "diffusion")["posterior_samples_main"]),
    )


def _subspace_arrays(prepared: PreparedSubjectArtifactFold, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    common = _training_arrays(prepared)
    population, bases, singular, rank_masks = batched_aligned_bases(
        common["subject_transfer"], prepared.population_context.full_transfer
    )
    contamination = common["observed"].astype(np.float64) - common["clean"].astype(np.float64)
    coefficients = np.einsum("ncr,nct->nrt", bases.astype(np.float64), contamination)
    tau = training_tau(coefficients, quantile=float(_mapping(config, "target")["tau_quantile"]))
    active_u = np.tanh(coefficients / tau[None, :, None]).astype(np.float32)
    artifact_rms = np.sqrt(np.mean(np.square(contamination), axis=(1, 2)))
    identity_count = max(1, int(math.ceil(artifact_rms.size * 0.25)))
    identity_pool = np.argsort(artifact_rms)[:identity_count].astype(np.int64)
    return {
        **common,
        "contamination": contamination.astype(np.float32),
        "population_basis": population,
        "basis": bases,
        "singular_values_subspace": singular,
        "rank_mask": rank_masks,
        "active_u": active_u,
        "tau": tau,
        "identity_pool": identity_pool,
    }


def _batch(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device,
    *,
    identity: np.ndarray | None = None,
    population_probability: float = 0.0,
    generator: np.random.Generator | None = None,
) -> dict[str, Tensor]:
    observed = torch.as_tensor(arrays["observed"][indices], device=device, dtype=torch.float32)
    basis = np.asarray(arrays["basis"][indices]).copy()
    rank = np.asarray(arrays["rank_mask"][indices]).copy()
    if population_probability > 0:
        if generator is None:
            raise ValueError("population augmentation requires a generator")
        population_draw = generator.random(indices.size) < population_probability
        basis[population_draw] = np.asarray(arrays["population_basis"])
        rank[population_draw] = True
    contamination = np.asarray(arrays["contamination"][indices], dtype=np.float64)
    coefficient = np.einsum("ncr,nct->nrt", basis.astype(np.float64), contamination)
    target = np.tanh(coefficient / np.asarray(arrays["tau"])[None, :, None]).astype(np.float32)
    if identity is not None:
        target[np.asarray(identity, dtype=bool)] = 0.0
    return {
        "observed": observed,
        "basis": torch.as_tensor(basis, device=device, dtype=torch.float32),
        "reliability": torch.as_tensor(arrays["rho"][indices], device=device, dtype=torch.float32),
        "rank_mask": torch.as_tensor(rank, device=device, dtype=torch.bool),
        "valid_time_mask": torch.as_tensor(arrays["valid"][indices], device=device, dtype=torch.bool),
        "target_u": torch.as_tensor(target, device=device, dtype=torch.float32),
    }


def _condition(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: batch[key] for key in ("observed", "basis", "reliability", "rank_mask", "valid_time_mask")}


def _masked_mse(predicted: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    mask = valid[:, None].to(predicted.dtype)
    return ((predicted - target).square() * mask).sum() / (mask.sum() * predicted.shape[1]).clamp_min(1)


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _clone_state(module: torch.nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _train(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    base, root = _load(config)
    route = _task(task_index)
    output = root / "checkpoints" / str(route["dataset"]) / f"fold_{int(route['fold_index']):02d}" / f"seed_{int(route['seed'])}"
    summary_path = output / "result_summary.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed_subspace_training":
            return {**prior, "resume_action": "skipped_completed"}
    prepared = _prepared(base, str(route["dataset"]), int(route["fold_index"]))
    arrays = _subspace_arrays(prepared, config)
    cfg = _model_config(prepared, config)
    device = torch.device("cuda", 0)
    seed = int(route["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy_rng = np.random.default_rng(seed)
    torch_generator = torch.Generator(device=device).manual_seed(seed + 41)
    deterministic = DeterministicSubspaceEstimator(cfg).to(device)
    diffusion = ArtifactSubspaceDiffusion(cfg).to(device)
    no_identity = ArtifactSubspaceDiffusion(cfg).to(device)
    parameters = (parameter_count(deterministic), parameter_count(diffusion))
    if abs(parameters[0] - parameters[1]) / parameters[1] > 0.10:
        raise AssertionError("deterministic and diffusion capacity differ by more than 10%")
    training = _mapping(config, "training")
    optimizers = {
        "deterministic": AdamW(deterministic.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])),
        "diffusion": AdamW(diffusion.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])),
        "no_identity": AdamW(no_identity.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])),
    }
    ema_decay = float(_mapping(config, "diffusion")["ema_decay"])
    ema = {name: value.detach().clone() for name, value in diffusion.state_dict().items()}
    ema_no_identity = {name: value.detach().clone() for name, value in no_identity.state_dict().items()}
    train_indices, validation_indices = _split_indices(prepared.training.recording_keys)
    batch_size = int(training["batch_size"])
    best = {"deterministic": (float("inf"), None, 0), "diffusion": (float("inf"), None, 0), "no_identity": (float("inf"), None, 0)}
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    maximum = int(training["maximum_updates"])
    for step in range(1, maximum + 1):
        active_count = batch_size // 2
        identity_count = batch_size - active_count
        active = numpy_rng.choice(train_indices, size=active_count, replace=train_indices.size < active_count)
        identity_pool = np.intersect1d(train_indices, arrays["identity_pool"])
        if identity_pool.size == 0:
            identity_pool = train_indices[np.argsort(np.sqrt(np.mean(np.square(arrays["contamination"][train_indices]), axis=(1, 2))))[: max(1, train_indices.size // 4)]]
        identity = numpy_rng.choice(identity_pool, size=identity_count, replace=identity_pool.size < identity_count)
        indices = np.concatenate([active, identity])
        identity_flags = np.concatenate([np.zeros(active_count, dtype=bool), np.ones(identity_count, dtype=bool)])
        order = numpy_rng.permutation(batch_size)
        indices, identity_flags = indices[order], identity_flags[order]
        batch = _batch(
            arrays, indices, device, identity=identity_flags,
            population_probability=float(training["population_operator_probability"]), generator=numpy_rng,
        )
        condition = _condition(batch)
        optimizers["deterministic"].zero_grad(set_to_none=True)
        det_prediction = deterministic(**condition)
        det_loss = _masked_mse(det_prediction, batch["target_u"], batch["valid_time_mask"])
        det_loss.backward()
        torch.nn.utils.clip_grad_norm_(deterministic.parameters(), float(training["gradient_clip_norm"]))
        optimizers["deterministic"].step()
        optimizers["diffusion"].zero_grad(set_to_none=True)
        diff_loss, _ = diffusion.training_loss(batch["target_u"], generator=torch_generator, **condition)
        diff_loss.backward()
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), float(training["gradient_clip_norm"]))
        optimizers["diffusion"].step()
        active_batch = _batch(arrays, numpy_rng.choice(train_indices, size=batch_size, replace=train_indices.size < batch_size), device, population_probability=float(training["population_operator_probability"]), generator=numpy_rng)
        optimizers["no_identity"].zero_grad(set_to_none=True)
        noid_loss, _ = no_identity.training_loss(active_batch["target_u"], generator=torch_generator, **_condition(active_batch))
        noid_loss.backward()
        torch.nn.utils.clip_grad_norm_(no_identity.parameters(), float(training["gradient_clip_norm"]))
        optimizers["no_identity"].step()
        with torch.no_grad():
            for target, module in ((ema, diffusion), (ema_no_identity, no_identity)):
                for name, value in module.state_dict().items():
                    target[name].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
        if step == 1 or step % int(training["log_interval_updates"]) == 0:
            curve.append({"step": step, "deterministic_loss": float(det_loss), "diffusion_loss": float(diff_loss), "no_identity_diffusion_loss": float(noid_loss)})
        if step % int(training["validation_interval_updates"]) == 0 or step == maximum:
            validation = _batch(arrays, validation_indices, device)
            with torch.no_grad():
                det_score = float(_masked_mse(deterministic(**_condition(validation)), validation["target_u"], validation["valid_time_mask"]))
                validation_generator = torch.Generator(device=device).manual_seed(seed + 99173)
                diff_score = float(diffusion.training_loss(validation["target_u"], generator=validation_generator, **_condition(validation))[1]["u_mse"])
                validation_generator = torch.Generator(device=device).manual_seed(seed + 99173)
                noid_score = float(no_identity.training_loss(validation["target_u"], generator=validation_generator, **_condition(validation))[1]["u_mse"])
            scores = {"deterministic": det_score, "diffusion": diff_score, "no_identity": noid_score}
            curve.append({"step": step, **{f"validation_{key}": value for key, value in scores.items()}})
            states = {"deterministic": _clone_state(deterministic), "diffusion": {name: value.detach().cpu().clone() for name, value in ema.items()}, "no_identity": {name: value.detach().cpu().clone() for name, value in ema_no_identity.items()}}
            for key, score in scores.items():
                if score < best[key][0]:
                    best[key] = (score, states[key], step)
        if step % 1000 == 0:
            _save(output / "resume.pt", {"step": step, "models": {"deterministic": deterministic.state_dict(), "diffusion": diffusion.state_dict(), "no_identity": no_identity.state_dict()}, "optimizers": {key: value.state_dict() for key, value in optimizers.items()}, "ema": ema, "ema_no_identity": ema_no_identity, "cuda_generator_state": torch_generator.get_state(), "numpy_state": numpy_rng.bit_generator.state})
    if any(value[1] is None for value in best.values()):
        raise AssertionError("independent validation did not select every checkpoint")
    checkpoint = output / "models.pt"
    _save(checkpoint, {
        "protocol_id": PROTOCOL, "route": dict(route), "model_config": cfg.__dict__,
        "deterministic": best["deterministic"][1], "diffusion_ema": best["diffusion"][1],
        "no_identity_diffusion_ema": best["no_identity"][1], "tau": arrays["tau"],
        "population_basis": arrays["population_basis"],
        "best_steps": {key: int(value[2]) for key, value in best.items()},
        "common_update_endpoint": maximum,
    })
    _write_csv(output / "training_curve.csv", curve)
    summary = {
        "status": "completed_subspace_training", **_implementation(), **dict(route),
        "checkpoint": str(checkpoint), "runtime_seconds": time.perf_counter() - started,
        "deterministic_parameters": parameters[0], "diffusion_parameters": parameters[1],
        "best_steps": {key: int(value[2]) for key, value in best.items()},
        "identity_pair_fraction": 0.5, "query_information_used": False,
    }
    _atomic_json(summary_path, summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _train_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    indices = list(range(worker, 78, 8))
    results = [_train(config, run_dir / f"task_{index:02d}", index) for index in indices]
    summary = {"status": "completed_training_worker", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results)}
    _atomic_json(run_dir / "worker_summary.json", summary)
    return summary


def _load_models(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, route: Mapping[str, Any], device: torch.device):
    base, root = _load(config)
    checkpoint = root / "checkpoints" / str(route["dataset"]) / f"fold_{int(route['fold_index']):02d}" / f"seed_{int(route['seed'])}/models.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ArtifactSubspaceConfig(**payload["model_config"])
    det = DeterministicSubspaceEstimator(cfg).to(device)
    diff = ArtifactSubspaceDiffusion(cfg).to(device)
    noid = ArtifactSubspaceDiffusion(cfg).to(device)
    det.load_state_dict(payload["deterministic"])
    diff.load_state_dict(payload["diffusion_ema"])
    noid.load_state_dict(payload["no_identity_diffusion_ema"])
    det.eval(); diff.eval(); noid.eval()
    # The old population anchor is an explicitly retained external baseline.
    old_config = yaml.safe_load((CODE_ROOT / "configs/cgdr/mainline_subject_residual_diffusion.yaml").read_text(encoding="utf-8"))
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _load_models as _old_load
    anchor, _, _, _, _ = _old_load(old_config, prepared, route, device)
    anchor.eval()
    return anchor, det, diff, noid, torch.as_tensor(payload["tau"], device=device, dtype=torch.float32), checkpoint


def _runtime_basis(context: RuntimeArtifactContext, population_basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis, _, rank = aligned_artifact_basis(context.full_transfer, population_basis)
    return basis, rank


@torch.no_grad()
def _infer(
    anchor: PopulationAnchor,
    deterministic: DeterministicSubspaceEstimator,
    diffusion: ArtifactSubspaceDiffusion,
    no_identity: ArtifactSubspaceDiffusion,
    *, observed: np.ndarray, valid: np.ndarray, population: RuntimeArtifactContext,
    matching: RuntimeArtifactContext, wrong: RuntimeArtifactContext, tau: Tensor,
    training_seed: int, participant_key: str, batch_size: int, device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, np.ndarray]]:
    names = (*METHODS, *ABLATIONS)
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in names}
    resource: dict[str, dict[str, float]] = {name: {"seconds": 0.0, "saturation": 0.0} for name in names}
    uncertainties: dict[str, list[np.ndarray]] = {"DIFF-MATCH": []}
    population_basis, _, population_rank = aligned_artifact_basis(population.full_transfer)
    match_basis, match_rank = _runtime_basis(matching, population_basis)
    wrong_basis, wrong_rank = _runtime_basis(wrong, population_basis)
    rho_value = float(matching.rho)
    base_seeds = participant_sample_seeds(participant_key, training_seed)
    for batch_index, start in enumerate(range(0, observed.shape[0], batch_size)):
        stop = min(observed.shape[0], start + batch_size)
        y = torch.as_tensor(observed[start:stop], device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid[start:stop], device=device, dtype=torch.bool)
        count = y.shape[0]
        bases = {
            "population": torch.as_tensor(population_basis, device=device)[None].expand(count, -1, -1),
            "matching": torch.as_tensor(match_basis, device=device)[None].expand(count, -1, -1),
            "wrong": torch.as_tensor(wrong_basis, device=device)[None].expand(count, -1, -1),
        }
        ranks = {
            "population": torch.as_tensor(population_rank, device=device)[None].expand(count, -1),
            "matching": torch.as_tensor(match_rank, device=device)[None].expand(count, -1),
            "wrong": torch.as_tensor(wrong_rank, device=device)[None].expand(count, -1),
        }
        rho = torch.full((count,), rho_value, device=device)
        conditions = {key: {"observed": y, "basis": bases[key], "reliability": rho, "rank_mask": ranks[key], "valid_time_mask": mask} for key in bases}
        outputs["RAW"].append(y.cpu().numpy())
        started = time.perf_counter(); x_pop = anchor(y, mask)
        resource["POP"]["seconds"] += time.perf_counter() - started
        outputs["POP"].append(x_pop.cpu().numpy())
        sample_seeds = tuple(seed + batch_index * 104729 for seed in base_seeds)
        # Population branch is always constructed first. rho=0 returns here
        # without constructing/consuming matching random streams.
        started = time.perf_counter()
        pop_u, _, _, _ = diffusion.sample(sample_seeds=sample_seeds, **conditions["population"])
        pop_x, pop_delta = reconstruct_from_subspace(y, bases["population"], pop_u, tau, ranks["population"], mask)
        resource["DIFF-POP"]["seconds"] += time.perf_counter() - started
        outputs["DIFF-POP"].append(pop_x.cpu().numpy())
        resource["DIFF-POP"]["saturation"] += float((pop_u.abs() >= 0.999).float().mean())
        if rho_value <= 0.0:
            match_x = pop_x
            match_delta = pop_delta
            wrong_x = pop_x
            match_sd = torch.zeros_like(pop_u)
            match_k1 = pop_x
            noid_x = pop_x
            det_x = pop_x
        else:
            started = time.perf_counter()
            match_u, match_sd, _, _ = diffusion.sample(sample_seeds=sample_seeds, **conditions["matching"])
            _, subject_delta = reconstruct_from_subspace(y, bases["matching"], match_u, tau, ranks["matching"], mask)
            match_delta = rho_value * subject_delta + (1.0 - rho_value) * pop_delta
            match_x = y - match_delta
            resource["DIFF-MATCH"]["seconds"] += time.perf_counter() - started
            wrong_u, _, _, _ = diffusion.sample(sample_seeds=sample_seeds, **conditions["wrong"])
            _, wrong_subject_delta = reconstruct_from_subspace(y, bases["wrong"], wrong_u, tau, ranks["wrong"], mask)
            wrong_x = y - (rho_value * wrong_subject_delta + (1.0 - rho_value) * pop_delta)
            match_k1_u, _, _, _ = diffusion.sample(sample_seeds=(sample_seeds[0],), **conditions["matching"])
            _, match_k1_delta = reconstruct_from_subspace(y, bases["matching"], match_k1_u, tau, ranks["matching"], mask)
            match_k1 = y - (rho_value * match_k1_delta + (1.0 - rho_value) * pop_delta)
            noid_u, _, _, _ = no_identity.sample(sample_seeds=sample_seeds, **conditions["matching"])
            _, noid_delta = reconstruct_from_subspace(y, bases["matching"], noid_u, tau, ranks["matching"], mask)
            noid_x = y - (rho_value * noid_delta + (1.0 - rho_value) * pop_delta)
            det_pop_u = deterministic(**conditions["population"])
            det_match_u = deterministic(**conditions["matching"])
            _, det_pop_delta = reconstruct_from_subspace(y, bases["population"], det_pop_u, tau, ranks["population"], mask)
            _, det_match_delta = reconstruct_from_subspace(y, bases["matching"], det_match_u, tau, ranks["matching"], mask)
            det_x = y - (rho_value * det_match_delta + (1.0 - rho_value) * det_pop_delta)
        for method, value in (("DIFF-MATCH", match_x), ("DIFF-WRONG-SAME-CELL", wrong_x), ("DET-MATCH", det_x), ("DIFF-MATCH-K1", match_k1), ("DIFF-MATCH-NO-IDENTITY", noid_x)):
            outputs[method].append(value.cpu().numpy())
        resource["DIFF-MATCH"]["saturation"] += float((match_u.abs() >= 0.999).float().mean()) if rho_value > 0 else 0.0
        uncertainties["DIFF-MATCH"].append(torch.sqrt(torch.einsum("bcr,brt->bct", bases["matching"].square(), (tau[None, :, None] * match_sd).square()).clamp_min(0)).cpu().numpy())
    batches = max(1, math.ceil(observed.shape[0] / batch_size))
    return (
        {key: np.concatenate(value).astype(np.float32) for key, value in outputs.items()},
        {key: {"latency_seconds_per_window": value["seconds"] / observed.shape[0], "coefficient_saturation_fraction": value["saturation"] / batches} for key, value in resource.items()},
        {key: np.concatenate(value).astype(np.float32) for key, value in uncertainties.items()},
    )


def _paired_complement_error(observed: np.ndarray, output: np.ndarray, basis: np.ndarray, valid: np.ndarray) -> float:
    y = torch.as_tensor(observed)
    x = torch.as_tensor(output, dtype=y.dtype, device=y.device)
    operator = torch.as_tensor(basis, dtype=y.dtype, device=y.device)[None].expand(y.shape[0], -1, -1)
    return float(complement_consistency_error(y, x, operator, torch.as_tensor(valid, device=y.device)))


def _evaluate_klados(config: Mapping[str, Any], run_dir: Path, seed_index: int) -> Mapping[str, Any]:
    base, root = _load(config)
    route = _task(seed_index)
    prepared = _prepared(base, "klados", 0)
    device = torch.device("cuda", 0)
    anchor, det, diff, noid, tau, checkpoint = _load_models(config, prepared, route, device)
    output_root = root / "evaluation/klados" / f"seed_{route['seed']}"
    rows = []
    arrays_root = root / "server_arrays/klados" / f"seed_{route['seed']}"
    arrays_root.mkdir(parents=True, exist_ok=True)
    evaluation_records = _klados_eval_records(base)
    # Klados has source-record rather than reliable participant identity.  The
    # negative control therefore uses the next frozen evaluation source's
    # support operator, never a within-record EOG permutation.
    for record_index, (key, mechanism, matching, _legacy_shuffle) in enumerate(evaluation_records):
        wrong = evaluation_records[(record_index + 1) % len(evaluation_records)][2]
        outputs, resources, uncertainty = _infer(anchor, det, diff, noid, observed=mechanism.observed_windows.astype(np.float32), valid=mechanism.valid_time_weight.astype(bool), population=prepared.population_context, matching=matching, wrong=wrong, tau=tau, training_seed=int(route["seed"]), participant_key=key, batch_size=int(_mapping(config, "evaluation")["batch_size"]), device=device)
        population_basis, _, _ = aligned_artifact_basis(prepared.population_context.full_transfer)
        matching_basis, _, _ = aligned_artifact_basis(matching.full_transfer, population_basis)
        for method, value in outputs.items():
            basis = population_basis if method == "DIFF-POP" else matching_basis
            rows.append({"dataset": "klados", "unit_id": key, "exact_cell": prepared.fold.layout_id, "training_seed": int(route["seed"]), "method": method, "status": "success", **_paired_metrics(mechanism.observed_windows, mechanism.clean_windows, value, mechanism.valid_time_weight.astype(bool)), "orthogonal_complement_relative_change": _paired_complement_error(mechanism.observed_windows, value, basis, mechanism.valid_time_weight.astype(bool)) if method in {"DIFF-POP"} else float("nan"), **resources.get(method, {}), "statistical_unit": "source_record"})
        np.savez_compressed(arrays_root / f"{key}.npz", observed=mechanism.observed_windows, clean=mechanism.clean_windows, det=outputs["DET-MATCH"], diff=outputs["DIFF-MATCH"], diff_uncertainty=uncertainty["DIFF-MATCH"])
    _write_csv(output_root / "metrics.csv", rows)
    summary = {"status": "completed_klados_subspace_evaluation", **_implementation(), "source_records": len(set(row["unit_id"] for row in rows)), "checkpoint": str(checkpoint)}
    _atomic_json(output_root / "result_summary.json", summary); _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _continuous_segment_psd_distortion(output: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    indices = np.flatnonzero(mask)
    if indices.size < 64:
        return float("nan")
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = [run for run in np.split(indices, cuts) if run.size >= 64]
    if not runs:
        return float("nan")
    values = []
    for run in runs:
        length = min(256, run.size)
        for start in range(0, run.size - length + 1, max(1, length // 2)):
            segment = run[start:start + length]
            window = np.hanning(length)[None]
            left = np.abs(np.fft.rfft(output[:, segment] * window, axis=1)) ** 2
            right = np.abs(np.fft.rfft(observed[:, segment] * window, axis=1)) ** 2
            values.append(float(np.mean(np.linalg.norm(left - right, axis=1) / np.maximum(np.linalg.norm(right, axis=1), np.finfo(float).eps))))
    return float(np.mean(values))


def _evaluate_sge(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    base, root = _load(config)
    route = _task(task_index + 3)
    prepared = prepare_subject_artifact_fold(base, int(route["fold_index"]))
    device = torch.device("cuda", 0)
    anchor, det, diff, noid, tau, checkpoint = _load_models(config, prepared, route, device)
    output_root = root / "evaluation/sgeyesub" / f"fold_{int(route['fold_index']):02d}" / f"seed_{route['seed']}"
    server_root = root / "server_arrays/sgeyesub" / f"fold_{int(route['fold_index']):02d}" / f"seed_{route['seed']}"
    server_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, heldout in prepared.heldout.items():
        outputs, resources, uncertainty = _infer(anchor, det, diff, noid, observed=heldout.query.observed, valid=heldout.query.valid_time_mask, population=prepared.population_context, matching=heldout.matching, wrong=heldout.wrong_same_cell, tau=tau, training_seed=int(route["seed"]), participant_key=key, batch_size=int(_mapping(config, "evaluation")["batch_size"]), device=device)
        archive = server_root / f"{key.replace('/', '__')}_outputs.npz"
        np.savez_compressed(archive, **{name.replace("-", "_"): value for name, value in outputs.items()}, diff_uncertainty=uncertainty["DIFF-MATCH"])
        annotated = _annotation_opener(base, prepared, key)()
        observed_continuous = _continuous(heldout.query.observed)
        annotations = annotated.query_annotations
        if annotations is None:
            raise AssertionError("query annotations were not opened after output freeze")
        rest = np.asarray(annotations.artifactclasses == 6, dtype=bool)
        for method, value in outputs.items():
            continuous = _continuous(value)
            metric = _evaluate_output(method_id=method, output=continuous, observed=observed_continuous, matching_projector=heldout.matching.projector, population_projector=prepared.population_context.projector, query_eog=annotations.external_eog, artifactclasses=annotations.artifactclasses, predicted_contamination=None, trial_labels=annotations.trial_labels, samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz), minimum_trials_per_condition=2, status="success", operator_source="support_calibrated_artifact_subspace", gamma=None, fallback_used=False, uses_query_external_eog=False)
            metric["continuous_segment_welch_psd_distortion"] = _continuous_segment_psd_distortion(continuous, observed_continuous, rest)
            input_rms = float(np.sqrt(np.mean(np.square(observed_continuous))))
            rows.append({"dataset": "sgeyesub", "unit_id": key, "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}", "study": prepared.fold.study, "training_seed": int(route["seed"]), "method": method, **metric, "output_input_RMS_ratio": float(np.sqrt(np.mean(np.square(continuous))) / max(input_rms, np.finfo(float).eps)), **resources.get(method, {}), "outputs_frozen_before_scoring": True})
    _write_csv(output_root / "metrics.csv", rows)
    summary = {"status": "completed_sge_subspace_evaluation", **_implementation(), **dict(route), "heldout_stems": len(prepared.heldout), "checkpoint": str(checkpoint)}
    _atomic_json(output_root / "result_summary.json", summary); _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _evaluate_sge_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    indices = list(range(worker, 75, 8))
    results = [_evaluate_sge(config, run_dir / f"task_{index:02d}", index) for index in indices]
    summary = {"status": "completed_SGE_worker", **_implementation(), "worker": worker, "task_indices": indices, "completed": len(results)}
    _atomic_json(run_dir / "worker_summary.json", summary)
    return summary


def _read_csvs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    return rows


def _float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for key, group in groups.items():
        result = dict(zip(keys, key))
        fields = set.intersection(*[
            {
                field
                for field, value in row.items()
                if field not in keys and field not in {"training_seed", "seed_count"} and _float(value)
            }
            for row in group
        ])
        for field in fields:
            result[field] = float(np.mean([float(row[field]) for row in group]))
        if all(_float(row.get("seed_count")) for row in group):
            # A second aggregation (units -> methods) must preserve the number
            # of training seeds already averaged within each unit.  Averaging
            # the numeric seed labels would otherwise look like one seed.
            result["seed_count"] = int(min(float(row["seed_count"]) for row in group))
        else:
            result["seed_count"] = len(set(str(row.get("training_seed")) for row in group))
        output.append(result)
    return output


def _effects(rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str, sign: float) -> list[dict[str, Any]]:
    by_unit: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_unit[str(row["unit_id"])][str(row["method"])] = row
    output = []
    for unit, methods in sorted(by_unit.items()):
        if left in methods and right in methods and _float(methods[left].get(metric)) and _float(methods[right].get(metric)):
            output.append({"unit_id": unit, "exact_cell": methods[left].get("exact_cell", ""), "effect": sign * (float(methods[left][metric]) - float(methods[right][metric]))})
    return output


def _bootstrap(rows: Sequence[Mapping[str, Any]], *, stratified: bool, seed: int, replicates: int) -> dict[str, Any]:
    if not rows:
        return {"mean": None, "median": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    values = np.asarray([float(row["effect"]) for row in rows])
    samples = np.empty(replicates)
    strata = {cell: np.asarray([float(row["effect"]) for row in rows if str(row.get("exact_cell", "")) == cell]) for cell in sorted(set(str(row.get("exact_cell", "")) for row in rows))}
    for index in range(replicates):
        draw = np.concatenate([rng.choice(part, part.size, replace=True) for part in strata.values()]) if stratified else rng.choice(values, values.size, replace=True)
        samples[index] = draw.mean()
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "n": int(values.size)}


def _risk_coverage(root: Path) -> dict[str, Any]:
    units = []
    for record in range(37, 55):
        if record in (44, 45):
            continue
        key = f"sim{record:02d}"
        seed_arrays = []
        for seed in SEEDS:
            path = root / "server_arrays/klados" / f"seed_{seed}/{key}.npz"
            if path.is_file():
                seed_arrays.append(np.load(path))
        if len(seed_arrays) != 3:
            continue
        clean = seed_arrays[0]["clean"].astype(np.float64)
        diff = np.mean([value["diff"].astype(np.float64) for value in seed_arrays], axis=0)
        det_stack = np.stack([value["det"].astype(np.float64) for value in seed_arrays])
        det = det_stack.mean(0)
        diff_uncertainty = np.mean([value["diff_uncertainty"].astype(np.float64) for value in seed_arrays], axis=0)
        det_uncertainty = det_stack.std(0)
        for window in range(clean.shape[0]):
            units.append({"record": key, "diff_error": float(np.sqrt(np.mean(np.square(diff[window] - clean[window])))), "det_error": float(np.sqrt(np.mean(np.square(det[window] - clean[window])))), "diff_uncertainty": float(np.mean(diff_uncertainty[window])), "det_uncertainty": float(np.mean(det_uncertainty[:, window]))})
    def summary(error_key: str, uncertainty_key: str) -> dict[str, float]:
        error = np.asarray([row[error_key] for row in units])
        uncertainty = np.asarray([row[uncertainty_key] for row in units])
        if error.size < 2:
            return {"uncertainty_error_correlation": float("nan"), "risk_coverage_auc": float("nan")}
        order = np.argsort(uncertainty)
        risks = np.asarray([error[order[:count]].mean() for count in range(1, error.size + 1)])
        return {"uncertainty_error_correlation": float(np.corrcoef(uncertainty, error)[0, 1]), "risk_coverage_auc": float(risks.mean())}
    return {"window_count": len(units), "diffusion": summary("diff_error", "diff_uncertainty"), "deterministic_ensemble": summary("det_error", "det_uncertainty")}


def _aggregate(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    klados_paths = sorted((root / "evaluation/klados").glob("seed_*/metrics.csv"))
    sge_paths = sorted((root / "evaluation/sgeyesub").glob("fold_*/seed_*/metrics.csv"))
    if len(klados_paths) != 3 or len(sge_paths) != 75:
        raise ValueError(f"incomplete evaluation coverage: Klados={len(klados_paths)}, SGE={len(sge_paths)}")
    units = _mean_rows(_read_csvs([*klados_paths, *sge_paths]), ("dataset", "unit_id", "exact_cell", "method"))
    methods = _mean_rows(units, ("dataset", "method"))
    _write_csv(root / "unit_metrics.csv", units); _write_csv(root / "method_summary.csv", methods)
    klados = [row for row in units if row["dataset"] == "klados"]
    sge = [row for row in units if row["dataset"] == "sgeyesub"]
    estimands = {
        "subject_population": _effects(sge, "DIFF-MATCH", "DIFF-POP", "eog_coherence_reduction", 1.0),
        "subject_wrong": _effects(sge, "DIFF-MATCH", "DIFF-WRONG-SAME-CELL", "eog_coherence_reduction", 1.0),
        "diffusion_value": _effects(klados, "DIFF-MATCH", "DET-MATCH", "clean_waveform_RRMSE", -1.0),
    }
    effect_rows = [{**row, "estimand": key} for key, values in estimands.items() for row in values]
    _write_csv(root / "paired_effects.csv", effect_rows)
    stats = _mapping(config, "statistics")
    summaries = {key: _bootstrap(value, stratified=key.startswith("subject"), seed=int(stats["bootstrap_seed"]) + index, replicates=int(stats["bootstrap_replicates"])) for index, (key, value) in enumerate(estimands.items())}
    safety = {}
    for index, (metric, sign) in enumerate((("nonartifact_observation_preservation", 1.0), ("condition_erp_observation_relative_preservation", 1.0), ("continuous_segment_welch_psd_distortion", -1.0), ("reference_free_covariance_distortion", -1.0))):
        value = _bootstrap(_effects(sge, "DIFF-MATCH", "POP", metric, sign), stratified=True, seed=int(stats["bootstrap_seed"]) + 20 + index, replicates=int(stats["bootstrap_replicates"]))
        value["noninferiority_passed"] = value["ci95"][0] is not None and float(value["ci95"][0]) >= float(stats["preservation_noninferiority_margin"])
        safety[metric] = value
    risk = _risk_coverage(root)
    point_superiority = summaries["diffusion_value"]["ci95"][0] is not None and summaries["diffusion_value"]["ci95"][0] > 0
    point_noninferiority = summaries["diffusion_value"]["ci95"][0] is not None and summaries["diffusion_value"]["ci95"][0] >= float(stats["preservation_noninferiority_margin"])
    uncertainty_better = risk["window_count"] > 0 and risk["diffusion"]["uncertainty_error_correlation"] > risk["deterministic_ensemble"]["uncertainty_error_correlation"] and risk["diffusion"]["risk_coverage_auc"] < risk["deterministic_ensemble"]["risk_coverage_auc"]
    natural_safety = all(value["noninferiority_passed"] for value in safety.values())
    subject = all(summaries[key]["ci95"][0] is not None and summaries[key]["ci95"][0] > 0 for key in ("subject_population", "subject_wrong"))
    diffusion_value = point_superiority or (point_noninferiority and uncertainty_better)
    verdict = "support_calibrated_subject_aware_diffusion_supported_in_development" if subject and diffusion_value and natural_safety else "current_support_calibrated_artifact_subspace_diffusion_not_fully_supported"
    payload = {"effects": summaries, "safety": safety, "posterior_utility": risk, "point_estimate_superiority": point_superiority, "posterior_utility_supported": uncertainty_better, "subject_calibration_supported": subject, "natural_EEG_safety_supported": natural_safety}
    _atomic_json(root / "bootstrap_uncertainty_summary.json", payload)
    coverage_sge = len(set(row["unit_id"] for row in sge))
    blocked = tuple(_mapping(_mapping(base, "data"), "sgeyesub").get("blocked_stems", ()))
    summary = {"status": "completed_artifact_subspace_aggregation", **_implementation(), "verdict": verdict, **payload, "coverage": {"klados_source_records": len(set(row["unit_id"] for row in klados)), "sge_successful_stems": coverage_sge, "sge_available_stems": coverage_sge + len(blocked), "sge_blocked_stems": list(blocked), "eegeyenet": "blocked_official_public_drive_participant_file_access_after_two_attempts"}, "evidence_scope": "Klados source-record mechanism evidence and SGE participant/stem development evidence; independent EEGEyeNet evaluation blocked before outcomes were opened"}
    _atomic_json(root / "result_summary.json", summary)
    _figures(root, summaries, sge, risk)
    _report(root, summary, methods)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _figures(root: Path, effects: Mapping[str, Mapping[str, Any]], sge: Sequence[Mapping[str, Any]], risk: Mapping[str, Any]) -> None:
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    labels = ["Match−Pop", "Match−Wrong", "Diff−Det"]
    keys = ("subject_population", "subject_wrong", "diffusion_value")
    means = [effects[key]["mean"] for key in keys]; intervals = [effects[key]["ci95"] for key in keys]
    figure, axis = plt.subplots(figsize=(6.5, 3.5)); axis.errorbar(means, range(3), xerr=[[mean - interval[0] for mean, interval in zip(means, intervals)], [interval[1] - mean for mean, interval in zip(means, intervals)]], fmt="o"); axis.axvline(0, color="black", lw=1); axis.set_yticks(range(3), labels); axis.set_xlabel("Paired utility effect (positive is better)"); figure.tight_layout(); figure.savefig(figures / "subject_operator_paired_effects.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 5))
    for method, color in (("POP", "0.4"), ("DET-MATCH", "tab:blue"), ("DIFF-MATCH", "tab:orange"), ("DIFF-WRONG-SAME-CELL", "tab:red")):
        selected = [row for row in sge if row["method"] == method]
        axis.scatter([float(row["nonartifact_observation_preservation"]) for row in selected], [float(row["eog_coherence_reduction"]) for row in selected], s=14, alpha=.55, label=method, color=color)
    axis.set_xlabel("Low-artifact preservation"); axis.set_ylabel("Independent EOG coherence reduction"); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(figures / "artifact_reduction_preservation.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.5, 4)); labels = ["Diffusion K=8", "Deterministic ensemble"]; values = [risk["diffusion"]["risk_coverage_auc"], risk["deterministic_ensemble"]["risk_coverage_auc"]]; axis.bar(labels, values, color=["tab:orange", "tab:blue"]); axis.set_ylabel("Risk–coverage AUC (lower is better)"); figure.tight_layout(); figure.savefig(figures / "uncertainty_risk_coverage.png", dpi=180); plt.close(figure)
    # Method schematic is deliberately code-native and compact.
    figure, axis = plt.subplots(figsize=(9, 2.7)); axis.axis("off")
    boxes = [(0.02, "Support\nEEG/EOG"), (.23, "Ridge C + SVD\nProcrustes A_s"), (.48, "Diffusion in\nA_s^T y coordinates"), (.73, "Delta=A_s tau u\nx_hat=y-Delta")]
    for x, text in boxes:
        axis.text(x, .5, text, ha="left", va="center", transform=axis.transAxes, bbox={"boxstyle":"round", "fc":"white", "ec":"black"})
    for x in (.19, .44, .69): axis.annotate("", xy=(x+.025,.5), xytext=(x-.015,.5), xycoords="axes fraction", arrowprops={"arrowstyle":"->"})
    figure.tight_layout(); figure.savefig(figures / "method_schematic.png", dpi=180); plt.close(figure)


def _report(root: Path, summary: Mapping[str, Any], methods: Sequence[Mapping[str, Any]]) -> None:
    effects = summary["effects"]
    lines = ["# Support-calibrated artifact-subspace diffusion", "", f"Decision: **{summary['verdict']}**.", "", "| Question | Mean effect | 95% CI | Supported |", "|---|---:|---:|---:|", f"| Subject calibration: MATCH−POP | {effects['subject_population']['mean']:.6g} | {effects['subject_population']['ci95']} | {summary['subject_calibration_supported']} |", f"| Specificity: MATCH−WRONG | {effects['subject_wrong']['mean']:.6g} | {effects['subject_wrong']['ci95']} | {summary['subject_calibration_supported']} |", f"| Diffusion point estimate: DIFF−DET | {effects['diffusion_value']['mean']:.6g} | {effects['diffusion_value']['ci95']} | {summary['point_estimate_superiority']} |", f"| Posterior utility | — | risk–coverage/calibration | {summary['posterior_utility_supported']} |", f"| Natural EEG safety | — | frozen −0.02 margins | {summary['natural_EEG_safety_supported']} |", "", "The model diffuses only bounded rank-two artifact coefficients. The support basis defines query coordinates and reconstruction; query EOG, eye tracking, labels, outcomes, participant identity, and best-of-K selection are absent from inference. Orthogonal-complement consistency means preservation only relative to the estimated artifact basis, not preservation of all neural signal.", "", "## Core methods", "", "| Dataset | Method | Clean RRMSE | EOG coherence reduction | Low-artifact preservation | Continuous-segment PSD distortion |", "|---|---|---:|---:|---:|---:|"]
    for row in methods:
        values = [row.get(key) for key in ("clean_waveform_RRMSE", "eog_coherence_reduction", "nonartifact_observation_preservation", "continuous_segment_welch_psd_distortion")]
        lines.append(f"| {row['dataset']} | {row['method']} | " + " | ".join("—" if value is None else f"{float(value):.6g}" for value in values) + " |")
    lines.extend(["", f"Coverage: {summary['coverage']['klados_source_records']} Klados source records and {summary['coverage']['sge_successful_stems']}/{summary['coverage']['sge_available_stems']} SGE stems. Klados is paired source-record mechanism evidence; SGE is real-EEG development evidence, not independent confirmation."])
    report = CODE_ROOT / "reports/subject_artifact_subspace_diffusion.md"; report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _technical(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    prepared = _prepared(base, "klados", 0); arrays = _subspace_arrays(prepared, config); device = torch.device("cuda", 0); cfg = _model_config(prepared, config)
    det = DeterministicSubspaceEstimator(cfg).to(device); diff = ArtifactSubspaceDiffusion(cfg).to(device); optimizer = AdamW(list(det.parameters()) + list(diff.parameters()), lr=5e-4)
    indices = np.arange(min(4, arrays["observed"].shape[0])); batch = _batch(arrays, indices, device); condition = _condition(batch); generator = torch.Generator(device=device).manual_seed(41)
    initial = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True); det_loss = _masked_mse(det(**condition), batch["target_u"], batch["valid_time_mask"]); diff_loss, _ = diff.training_loss(batch["target_u"], generator=generator, timestep=torch.full((len(indices),), 250, device=device, dtype=torch.long), noise=torch.zeros_like(batch["target_u"]), **condition); loss = det_loss + diff_loss
        if initial is None: initial = float(loss.detach())
        loss.backward(); optimizer.step()
    seeds = participant_sample_seeds("technical_subject", 20260811); mean, sd, calls, trace = diff.sample(sample_seeds=seeds, record_trajectory=True, **condition); restored, _ = reconstruct_from_subspace(batch["observed"], batch["basis"], mean, torch.as_tensor(arrays["tau"], device=device), batch["rank_mask"], batch["valid_time_mask"])
    identity = torch.zeros_like(batch["target_u"]); identity_restored, correction = reconstruct_from_subspace(batch["observed"], batch["basis"], identity, torch.as_tensor(arrays["tau"], device=device), batch["rank_mask"], batch["valid_time_mask"])
    wrong_basis = batch["basis"].roll(1, 0); wrong_condition = {**condition, "basis": wrong_basis}; wrong, _, _, _ = diff.sample(sample_seeds=seeds, **wrong_condition)
    checkpoint = root / "technical_check/models.pt"; ema = _clone_state(diff); _save(checkpoint, {"deterministic": _clone_state(det), "diffusion_ema": ema, "optimizer": optimizer.state_dict(), "cuda_generator_state": generator.get_state(), "step": 200})
    det_reload = DeterministicSubspaceEstimator(cfg).to(device); reload = ArtifactSubspaceDiffusion(cfg).to(device); payload = torch.load(checkpoint, map_location=device, weights_only=False); det_reload.load_state_dict(payload["deterministic"]); reload.load_state_dict(payload["diffusion_ema"])
    resume_optimizer = AdamW(list(det_reload.parameters()) + list(reload.parameters()), lr=5e-4); resume_optimizer.load_state_dict(payload["optimizer"]); resume_optimizer.zero_grad(set_to_none=True); resume_generator = torch.Generator(device=device); resume_generator.set_state(payload["cuda_generator_state"].cpu()); resume_loss = _masked_mse(det_reload(**condition), batch["target_u"], batch["valid_time_mask"]) + reload.training_loss(batch["target_u"], generator=resume_generator, timestep=torch.full((len(indices),), 250, device=device, dtype=torch.long), noise=torch.zeros_like(batch["target_u"]), **condition)[0]; resume_loss.backward(); resume_optimizer.step()
    finite_trajectory = all(math.isfinite(value["u_rms"]) and value["ratio"] < 10.0 for value in trace)
    rng_state = payload["cuda_generator_state"]
    checks = {"same_bounded_target_fitted": float(loss.detach()) < float(initial), "identity_zero_correction": float(correction.abs().max()) == 0.0 and torch.equal(identity_restored, batch["observed"] * batch["valid_time_mask"][:, None]), "trajectory_finite": finite_trajectory and calls == 200, "bounded_u": float(mean.abs().max()) <= 1.0, "complement_consistency": float(complement_consistency_error(batch["observed"], restored, batch["basis"], batch["valid_time_mask"])) <= 1e-5, "operator_intervention_changes_output": float(torch.linalg.vector_norm(mean - wrong)) > 1e-7, "checkpoint_EMA_reload_resume": bool(torch.isfinite(resume_loss)), "explicit_cuda_generator_saved": rng_state.dtype == torch.uint8 and rng_state.numel() > 0, "participant_seeds_unique": seeds != participant_sample_seeds("technical_subject_2", 20260811), "query_fields_absent": set(diff.visible_input_fields).isdisjoint(diff.forbidden_input_fields)}
    result = {"status": "passed_technical_validity" if all(checks.values()) else "failed_technical_validity", **_implementation(), "checks": checks, "initial_loss": initial, "final_loss": float(loss.detach()), "trajectory_steps": len(trace)}
    _atomic_json(root / "technical_check/result_summary.json", result); _atomic_json(run_dir / "result_summary.json", result)
    if not all(checks.values()): raise RuntimeError("artifact-subspace technical validity failed")
    return result


def _j0(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config); validation = validate_real_subject_artifact_inputs(base); data = _mapping(config, "data"); eegeyenet = Path(str(data["eegeyenet"])); status = "present" if eegeyenet.exists() else "missing"
    osf_listing = None
    if status == "missing":
        try:
            with urllib.request.urlopen("https://api.osf.io/v2/nodes/ktv7m/files/osfstorage/", timeout=30) as response:
                value = json.load(response)
            osf_listing = [{"name": item.get("attributes", {}).get("name"), "kind": item.get("attributes", {}).get("kind"), "download": item.get("links", {}).get("download")} for item in value.get("data", [])]
            status = "missing_public_OSF_listing_reachable"
        except Exception as error:
            status = f"missing_OSF_probe_failed:{type(error).__name__}"
    _write_csv(root / "task_list.csv", task_rows())
    result = {"status": "completed_J0", **_implementation(), "real_input_validation": validation["status"], "training_tasks": 78, "eegeyenet_status": status, "eegeyenet_osf_top_level": osf_listing, "historical_8aca035_modified": False}
    _atomic_json(root / "j0_summary.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def _j1_real(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    base, root = _load(config)
    checks = []
    for dataset, fold in (("klados", 0), ("sgeyesub", 0)):
        prepared = _prepared(base, dataset, fold)
        arrays = _subspace_arrays(prepared, config)
        gram = np.einsum("ncr,ncs->nrs", arrays["basis"], arrays["basis"])
        disjoint = not (set(prepared.fold.training_recording_keys) & set(prepared.fold.heldout_recording_keys))
        checks.append({"dataset": dataset, "training_windows": int(arrays["observed"].shape[0]), "eeg_channels": int(arrays["observed"].shape[1]), "tau_0": float(arrays["tau"][0]), "tau_1": float(arrays["tau"][1]), "basis_orthonormal": bool(np.allclose(gram, np.eye(2)[None], atol=1e-5)), "finite": bool(all(np.isfinite(arrays[key]).all() for key in ("observed", "contamination", "basis", "active_u", "tau"))), "support_query_recording_disjoint": disjoint, "bounded_target": bool(np.max(np.abs(arrays["active_u"])) <= 1.0)})
    passed = all(row["basis_orthonormal"] and row["finite"] and row["support_query_recording_disjoint"] and row["bounded_target"] for row in checks)
    result = {"status": "passed_real_operator_target_validation" if passed else "failed_real_operator_target_validation", **_implementation(), "datasets": checks}
    _atomic_json(root / "j1_real_validation.json", result); _atomic_json(run_dir / "result_summary.json", result)
    if not passed: raise RuntimeError("real artifact-subspace operator/target validation failed")
    return result


def _osf_files(url: str, prefix: Path = Path()) -> list[dict[str, str]]:
    """Recursively list the public EEGEyeNet OSF node without signal reads."""

    output: list[dict[str, str]] = []
    next_url: str | None = url
    while next_url:
        with urllib.request.urlopen(next_url, timeout=60) as response:
            page = json.load(response)
        for item in page.get("data", []):
            attributes = item.get("attributes", {})
            name = str(attributes.get("name", ""))
            kind = str(attributes.get("kind", ""))
            if not name or name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("unsafe OSF member name")
            relative = prefix / name
            if kind == "file":
                download = item.get("links", {}).get("download")
                if not isinstance(download, str) or not download.startswith("https://"):
                    raise ValueError("OSF file lacks an HTTPS download link")
                output.append({"path": relative.as_posix(), "download": download})
            elif kind == "folder":
                related = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                if not isinstance(related, str):
                    raise ValueError("OSF folder lacks a child listing")
                output.extend(_osf_files(related, relative))
        next_value = page.get("links", {}).get("next")
        next_url = next_value if isinstance(next_value, str) and next_value else None
    return output


def _download_resume(url: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=180) as response:
        status = int(getattr(response, "status", 200))
        mode = "ab" if offset and status == 206 else "wb"
        with path.open(mode) as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    return path.stat().st_size


def _prepare_eegeyenet(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Download the public OSF release transactionally when it is absent."""

    _, root = _load(config)
    final = Path(str(_mapping(config, "data")["eegeyenet"]))
    partial = final.with_name(f"{final.name}.partial")
    if final.exists():
        result = {"status": "already_present_no_download", **_implementation(), "data_root": str(final)}
        _atomic_json(root / "eegeyenet_prepare.json", result); _atomic_json(run_dir / "result_summary.json", result); return result
    files = _osf_files("https://api.osf.io/v2/nodes/ktv7m/files/osfstorage/")
    if not files:
        raise RuntimeError("public EEGEyeNet OSF project exposed no downloadable files")
    rows = []
    for index, item in enumerate(files):
        destination = partial / item["path"]
        size = _download_resume(item["download"], destination)
        rows.append({"file_index": index, "relative_path": item["path"], "downloaded_bytes": size, "status": "downloaded_or_resumed"})
    if final.exists():
        raise RuntimeError("EEGEyeNet final directory appeared concurrently; publication stopped")
    os.replace(partial, final)
    _write_csv(root / "eegeyenet_file_index.csv", rows)
    result = {"status": "completed_public_EEGEyeNet_download", **_implementation(), "data_root": str(final), "file_count": len(rows)}
    _atomic_json(root / "eegeyenet_prepare.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def _eegeyenet_source_audit(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Resolve official public data endpoints without opening signal outcomes."""

    _, root = _load(config)
    readme = ""
    readme_url = None
    for branch in ("master", "main"):
        url = f"https://raw.githubusercontent.com/ardkastrati/EEGEyeNet/{branch}/README.md"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                readme = response.read().decode("utf-8", errors="replace")
            readme_url = url
            break
        except Exception:
            continue
    with urllib.request.urlopen("https://api.osf.io/v2/nodes/ktv7m/children/", timeout=60) as response:
        children_page = json.load(response)
    children = []
    for item in children_page.get("data", []):
        node_id = str(item.get("id", ""))
        title = str(item.get("attributes", {}).get("title", ""))
        child = {"node_id": node_id, "title": title, "files": []}
        if node_id:
            try:
                child["files"] = _osf_files(f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/")
            except Exception as error:
                child["listing_error"] = type(error).__name__
        children.append(child)
    urls = sorted(set(re.findall(r"https?://[^\s\])>\"']+", readme)))
    relevant_lines = [line.strip() for line in readme.splitlines() if any(word in line.lower() for word in ("download", "dataset", "data access", "osf", "zenodo", "figshare"))]
    result = {"status": "completed_official_source_audit", **_implementation(), "readme_url": readme_url, "readme_referenced_urls": urls, "readme_data_lines": relevant_lines, "osf_components": children}
    _atomic_json(root / "eegeyenet_source_audit.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def _eegeyenet_repository_audit(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Inspect only official repository text for the public data endpoint."""

    _, root = _load(config)
    with urllib.request.urlopen("https://api.github.com/repos/ardkastrati/EEGEyeNet/git/trees/master?recursive=1", timeout=60) as response:
        tree = json.load(response).get("tree", [])
    candidates = []
    for item in tree:
        path = str(item.get("path", ""))
        size = int(item.get("size") or 0)
        lowered = path.lower()
        if item.get("type") != "blob" or size > 300000 or not lowered.endswith((".md", ".txt", ".py", ".yaml", ".yml")):
            continue
        if not any(word in lowered for word in ("readme", "download", "data", "config", "setup")):
            continue
        raw_url = f"https://raw.githubusercontent.com/ardkastrati/EEGEyeNet/master/{path}"
        try:
            with urllib.request.urlopen(raw_url, timeout=30) as response:
                text_value = response.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        urls = sorted(set(re.findall(r"https?://[^\s\])>\"']+", text_value)))
        lines = [line.strip() for line in text_value.splitlines() if any(word in line.lower() for word in ("download", "dataset", "osf", "zenodo", "figshare", "research-collection"))]
        if urls or lines:
            candidates.append({"path": path, "urls": urls, "data_lines": lines[:80]})
    result = {"status": "completed_official_repository_data_endpoint_audit", **_implementation(), "candidate_files": candidates}
    _atomic_json(root / "eegeyenet_repository_audit.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def _eegeyenet_osf_metadata(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Inspect the official OSF description/wiki for large-file access notes."""

    _, root = _load(config)
    with urllib.request.urlopen("https://api.osf.io/v2/nodes/ktv7m/", timeout=60) as response:
        node = json.load(response)
    attributes = node.get("data", {}).get("attributes", {})
    descriptions = [str(attributes.get("description", ""))]
    wiki_rows = []
    try:
        with urllib.request.urlopen("https://api.osf.io/v2/nodes/ktv7m/wikis/", timeout=60) as response:
            wiki_page = json.load(response)
        for item in wiki_page.get("data", []):
            row = {"name": item.get("attributes", {}).get("name"), "urls": []}
            download = item.get("links", {}).get("download")
            if isinstance(download, str):
                try:
                    with urllib.request.urlopen(download, timeout=60) as response:
                        content = response.read().decode("utf-8", errors="replace")
                    row["urls"] = sorted(set(re.findall(r"https?://[^\s\])>\"']+", content)))
                    row["data_lines"] = [line.strip() for line in content.splitlines() if any(word in line.lower() for word in ("download", "data", "access", "request"))][:80]
                except Exception as error:
                    row["read_error"] = type(error).__name__
            wiki_rows.append(row)
    except Exception as error:
        wiki_rows.append({"listing_error": type(error).__name__})
    result = {"status": "completed_official_OSF_metadata_audit", **_implementation(), "title": attributes.get("title"), "description": descriptions[0], "description_urls": sorted(set(re.findall(r"https?://[^\s\])>\"']+", descriptions[0]))), "wikis": wiki_rows}
    _atomic_json(root / "eegeyenet_osf_metadata.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def _eegeyenet_gdrive_download(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Download only the public synchronized-data subfolder named by OSF.

    The Drive root also contains a large, irrelevant ``.git`` directory whose
    pack file is quota limited.  The official synchronized-data child folder
    is selected explicitly so a resume never traverses that repository cache.
    """

    _, root = _load(config)
    data_config = _mapping(config, "data")
    target = Path(str(data_config.get("eegeyenet_gdrive_target", Path(str(data_config["eegeyenet"])) / "google_drive")))
    target.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore
    except ImportError:
        # Do not mutate either registered Conda environment.  Import the
        # small pure-Python wheel from an untracked run cache instead.
        tool_root = CODE_ROOT / "runs/tools"
        tool_root.mkdir(parents=True, exist_ok=True)
        wheels = (("gdown", "gdown.whl"), ("beautifulsoup4", "beautifulsoup4.whl"), ("soupsieve", "soupsieve.whl"))
        for package_name, filename in wheels:
            wheel = tool_root / filename
            if not wheel.is_file():
                with urllib.request.urlopen(f"https://pypi.org/pypi/{package_name}/json", timeout=60) as response:
                    package = json.load(response)
                candidates = [item for item in package.get("urls", []) if str(item.get("filename", "")).endswith("-py3-none-any.whl")]
                if not candidates:
                    raise RuntimeError(f"PyPI exposed no pure-Python wheel for {package_name}")
                _download_resume(str(candidates[0]["url"]), wheel)
            sys.path.insert(0, str(wheel))
        import gdown  # type: ignore
    url = str(data_config.get("eegeyenet_gdrive_folder", "https://drive.google.com/drive/folders/1iHpnEE6kalLGHaw2Hd8EwJMdVE0K7rk7"))
    arguments: dict[str, Any] = {"url": url, "output": str(target), "quiet": False, "use_cookies": False}
    folder_parameters = inspect.signature(gdown.download_folder).parameters
    if "remaining_ok" in folder_parameters:
        arguments["remaining_ok"] = True
    if "resume" in folder_parameters:
        arguments["resume"] = True
    downloaded = gdown.download_folder(**arguments)
    files = [path for path in target.rglob("*") if path.is_file()]
    result = {"status": "completed_or_resumed_official_Google_Drive_download" if files else "blocked_public_folder_returned_no_files", **_implementation(), "source_folder": url, "target": str(target), "reported_download_entries": 0 if downloaded is None else len(downloaded), "local_file_count": len(files)}
    _atomic_json(root / "eegeyenet_gdrive_download.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def _eegeyenet_pdf_metadata(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Extract only release documentation needed to freeze a loader/split."""

    _, root = _load(config)
    data_root = Path(str(_mapping(config, "data")["eegeyenet"]))
    pdfs = (data_root / "DATA DESCRIPTION/Data  Structure Description.pdf", data_root / "DATA DESCRIPTION/Experimental Paradigms.pdf")
    rows = []
    for pdf in pdfs:
        if not pdf.is_file():
            raise FileNotFoundError(f"EEGEyeNet release document is missing: {pdf}")
        if shutil.which("pdftotext"):
            completed = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True, capture_output=True, text=True)
            text_value = completed.stdout
        else:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                tool_root = CODE_ROOT / "runs/tools"; tool_root.mkdir(parents=True, exist_ok=True); wheel = tool_root / "pypdf.whl"
                if not wheel.is_file():
                    with urllib.request.urlopen("https://pypi.org/pypi/pypdf/json", timeout=60) as response:
                        package = json.load(response)
                    candidates = [item for item in package.get("urls", []) if str(item.get("filename", "")).endswith("-py3-none-any.whl")]
                    if not candidates: raise RuntimeError("PyPI exposed no pure-Python pypdf wheel")
                    _download_resume(str(candidates[0]["url"]), wheel)
                sys.path.insert(0, str(wheel))
                from pypdf import PdfReader  # type: ignore
            text_value = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        relevant = [line.strip() for line in text_value.splitlines() if any(word in line.lower() for word in ("sampling", "channel", "eye", "eeg", "event", "participant", "subject", "wi1", "wi2", "structure"))]
        rows.append({"document": pdf.name, "relevant_lines": relevant[:400]})
    report = CODE_ROOT / "reports/eegeyenet_release_metadata.md"
    lines = ["# EEGEyeNet release metadata", "", "Documentation-only audit; no candidate test signal or outcome was opened.", ""]
    for row in rows:
        lines.extend([f"## {row['document']}", "", *[f"- {line}" for line in row["relevant_lines"]], ""])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {"status": "completed_EEGEyeNet_release_metadata", **_implementation(), "documents": [str(path) for path in pdfs], "report": str(report)}
    _atomic_json(root / "eegeyenet_release_metadata.json", result); _atomic_json(run_dir / "result_summary.json", result); return result


def run_stage(config: Mapping[str, Any], run_dir: str | Path, stage: str, task_index: int | None) -> Mapping[str, Any]:
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    if stage == "j0": return _j0(config, run)
    if stage == "j1-real": return _j1_real(config, run)
    if stage == "j0-eegeyenet-download": return _prepare_eegeyenet(config, run)
    if stage == "j0-eegeyenet-source": return _eegeyenet_source_audit(config, run)
    if stage == "j0-eegeyenet-repository": return _eegeyenet_repository_audit(config, run)
    if stage == "j0-eegeyenet-wiki": return _eegeyenet_osf_metadata(config, run)
    if stage == "j0-eegeyenet-gdrive": return _eegeyenet_gdrive_download(config, run)
    if stage == "j0-eegeyenet-pdf-metadata": return _eegeyenet_pdf_metadata(config, run)
    if stage == "j2-technical": return _technical(config, run)
    if stage == "j3-train-worker":
        if task_index is None or not 0 <= task_index < 8: raise ValueError("training worker requires array 0-7")
        return _train_worker(config, run, task_index)
    if stage == "j4-klados":
        if task_index is None or not 0 <= task_index < 3: raise ValueError("Klados evaluation requires array 0-2")
        return _evaluate_klados(config, run, task_index)
    if stage == "j4-sge-worker":
        if task_index is None or not 0 <= task_index < 8: raise ValueError("SGE worker requires array 0-7")
        return _evaluate_sge_worker(config, run, task_index)
    if stage == "j5-aggregate": return _aggregate(config, run)
    if stage == "j8-finalize":
        _, root = _load(config); summary = json.loads((root / "result_summary.json").read_text(encoding="utf-8")); result = {"status": "completed_J8", **_implementation(), "verdict": summary["verdict"], "result_path": str(root), "report": str(CODE_ROOT / "reports/subject_artifact_subspace_diffusion.md")}; _atomic_json(root / "terminal_manifest.json", result); _atomic_json(run / "result_summary.json", result); return result
    raise ValueError(f"unsupported artifact-subspace stage: {stage}")


__all__ = ["run_stage", "task_rows"]
