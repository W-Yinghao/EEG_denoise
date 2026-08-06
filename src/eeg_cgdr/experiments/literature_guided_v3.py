"""Literature-guided subject-aware diffusion exploration v3.

The module intentionally provides a small Slurm-facing surface.  It audits
only named datasets/repositories and never walks the 30 TB data root.  Model
screens are added as independent stages so a scientifically negative route
cannot cancel unrelated routes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW

from eeg_cgdr.experiments.mainline_subject_residual_diffusion import (
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
from eeg_cgdr.experiments.parallel_subject_aware_routes_v1 import (
    RouteBlockedError,
    _klados_matching_support,
    _sge_matching_support,
    _subject_arrays,
)
from eeg_cgdr.experiments.subject_artifact_data import PreparedSubjectArtifactFold
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusionConfig
from eeg_cgdr.models.artifact_latent_deterministic import ArtifactLatentModelConfig
from eeg_cgdr.models.artifact_latent_inference import canonical_artifact_delta
from eeg_cgdr.models.artifact_subspace_diffusion import participant_sample_seeds
from eeg_cgdr.models.literature_guided_v3 import (
    DirectSupportAdapter,
    RawSupportConfig,
    RawSupportTokenDeterministic,
    RawSupportTokenDiffusion,
    SupportStatisticControl,
)


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_lit_explore_v3"))
PROTOCOL = "literature_guided_subject_aware_exploration_v3"


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
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _git(args: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/usr/bin/git", *args], cwd=cwd, text=True, capture_output=True,
        check=False, timeout=600,
    )


def _repo_audit(name: str, url: str, root: Path) -> dict[str, Any]:
    destination = root / name
    status = "unknown"
    detail = ""
    if destination.is_dir() and (destination / ".git").is_dir():
        status = "existing_checkout"
        fetch = _git(["fetch", "--depth", "1", "origin"], cwd=destination)
        if fetch.returncode != 0:
            detail = f"fetch_failed: {fetch.stderr.strip()[:500]}"
    elif destination.exists():
        status = "blocked_non_git_destination"
        detail = "destination exists but is not a Git checkout"
    else:
        root.mkdir(parents=True, exist_ok=True)
        clone = _git(["clone", "--depth", "1", url, str(destination)])
        if clone.returncode == 0:
            status = "cloned"
        else:
            status = "clone_failed"
            detail = clone.stderr.strip()[:500]
    commit = ""
    if (destination / ".git").is_dir():
        head = _git(["rev-parse", "HEAD"], cwd=destination)
        commit = head.stdout.strip() if head.returncode == 0 else ""
    files = tuple(destination.rglob("*")) if destination.is_dir() else ()
    python_files = [path for path in files if path.is_file() and path.suffix == ".py"]
    matlab_files = [path for path in files if path.is_file() and path.suffix == ".m"]
    readmes = [path for path in files if path.is_file() and path.name.lower().startswith("readme")]
    train_like = [path for path in python_files if any(token in path.name.lower() for token in ("train", "main"))]
    data_like = [path for path in python_files if any(token in path.name.lower() for token in ("data", "loader", "dataset"))]
    checkpoint_like = [path for path in files if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    return {
        "repository": name,
        "url": url,
        "status": status,
        "commit": commit,
        "python_files": len(python_files),
        "matlab_files": len(matlab_files),
        "readme_files": len(readmes),
        "train_entry_candidates": len(train_like),
        "data_loader_candidates": len(data_like),
        "bundled_checkpoint_files": len(checkpoint_like),
        "detail": detail,
    }


def _dataset_candidate(root: Path, aliases: Sequence[str]) -> tuple[Path | None, list[Path]]:
    direct: list[Path] = []
    for alias in aliases:
        candidate = root / alias
        if candidate.exists():
            direct.append(candidate)
    # A bounded two-level name search catches registered version directories
    # without recursively traversing unrelated shared datasets.
    wanted = {Path(alias).name.lower() for alias in aliases}
    for first in root.iterdir():
        if first.name.lower() in wanted and first not in direct:
            direct.append(first)
        if first.is_dir() and first.name.lower() in {"datasets", "data", "derived"}:
            for second in first.iterdir():
                if second.name.lower() in wanted and second not in direct:
                    direct.append(second)
    return (direct[0] if direct else None), direct


def _bounded_dataset_facts(dataset_id: str, path: Path | None, candidates: Sequence[Path]) -> dict[str, Any]:
    if path is None:
        return {
            "dataset_id": dataset_id,
            "status": "missing_named_path",
            "selected_path": "",
            "candidate_paths": "",
            "top_level_entries": 0,
            "participant_like_entries": 0,
            "session_like_entries": 0,
            "signal_file_sample_count": 0,
        }
    children = tuple(path.iterdir()) if path.is_dir() else ()
    participant = [value for value in children if value.name.lower().startswith(("sub", "subject", "s0", "p0"))]
    session = [value for value in children if "session" in value.name.lower() or value.name.lower().startswith("ses")]
    signal_suffixes = {".set", ".fdt", ".mat", ".csv", ".edf", ".bdf", ".npy", ".npz"}
    sampled = []
    if path.is_dir():
        for value in path.rglob("*"):
            if value.is_file() and value.suffix.lower() in signal_suffixes:
                sampled.append(value)
                if len(sampled) >= 5000:
                    break
    return {
        "dataset_id": dataset_id,
        "status": "present_named_path",
        "selected_path": str(path),
        "candidate_paths": ";".join(str(value) for value in candidates),
        "top_level_entries": len(children),
        "participant_like_entries": len(participant),
        "session_like_entries": len(session),
        "signal_file_sample_count": len(sampled),
    }


def audit(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    data = _mapping(config, "data")
    output = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    root = Path(str(data["root"]))
    if root != Path("/projects/EEG-foundation-model") or not root.is_dir():
        raise ValueError("fixed EEG data root is unavailable")
    disk = shutil.disk_usage(root)
    dataset_rows = []
    for dataset_id, aliases in _mapping(data, "known_candidates").items():
        if not isinstance(aliases, Sequence) or isinstance(aliases, str):
            raise ValueError(f"dataset aliases are invalid: {dataset_id}")
        selected, candidates = _dataset_candidate(root, tuple(str(value) for value in aliases))
        dataset_rows.append(_bounded_dataset_facts(str(dataset_id), selected, candidates))
    _write_csv(output / "baseline_audit/data_availability.csv", dataset_rows)

    repositories = _mapping(_mapping(config, "external_repositories"), "repositories")
    external_root = CODE_ROOT / str(_mapping(config, "external_repositories")["root"])
    repo_rows = [_repo_audit(str(name), str(url), external_root) for name, url in repositories.items()]
    _write_csv(output / "baseline_audit/official_repository_audit.csv", repo_rows)
    summary = {
        "status": "completed_named_data_and_official_repository_audit",
        "protocol_id": PROTOCOL,
        "git_sha": os.environ.get("DENOISENET_GIT_HEAD", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "data_root_free_gib": round(disk.free / 2**30, 2),
        "datasets_present": sorted(row["dataset_id"] for row in dataset_rows if row["status"] == "present_named_path"),
        "repositories_available": sorted(row["repository"] for row in repo_rows if row["status"] in {"cloned", "existing_checkout"}),
        "repositories_failed": sorted(row["repository"] for row in repo_rows if row["status"] not in {"cloned", "existing_checkout"}),
        "whole_data_root_scan": False,
        "result_root": str(output),
    }
    _write_json(output / "baseline_audit/audit_summary.json", summary)
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in ("P_A_RAW_SUPPORT_TOKENS", "P_B_DIRECT_SUPPORT_ADAPTER", "P_D_SUPPORT_STAT_CONTROL"):
        rows.append({"task_index": len(rows), "route": route, "dataset": "klados", "fold_index": 0, "seed": 20260811})
        for fold in range(25):
            rows.append({"task_index": len(rows), "route": route, "dataset": "sgeyesub", "fold_index": fold, "seed": 20260811})
    return rows


def fold_rows() -> list[dict[str, Any]]:
    return ([{"task_index": 0, "dataset": "klados", "fold_index": 0, "seed": 20260811}]
            + [{"task_index": index + 1, "dataset": "sgeyesub", "fold_index": index, "seed": 20260811} for index in range(25)])


def _model_configs(prepared: PreparedSubjectArtifactFold, config: Mapping[str, Any]) -> tuple[ArtifactLatentModelConfig, ArtifactLatentDiffusionConfig, RawSupportConfig]:
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
            posterior_samples=int(diffusion["posterior_samples"]),
        ),
        RawSupportConfig(
            token_width=int(model["support_token_dim"]),
            token_count=int(model["support_token_count"]),
            encoder_layers=int(model["support_encoder_layers"]),
            context_dropout_probability=float(_mapping(config, "screen")["context_dropout_probability"]),
        ),
    )


def _prepared_v3(config: Mapping[str, Any], row: Mapping[str, Any]) -> PreparedSubjectArtifactFold:
    base_path = CODE_ROOT / str(config["base_subject_artifact_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base artifact config is invalid")
    return _prepared(base, str(row["dataset"]), int(row["fold_index"]))


def _base_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = yaml.safe_load((CODE_ROOT / str(config["base_subject_artifact_config"])).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("base subject-artifact config is invalid")
    return value


def _checkpoint(config: Mapping[str, Any], row: Mapping[str, Any]) -> Path:
    root = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    return root / "population_backbones/raw_support_tokens" / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}" / "seed_20260811/models.pt"


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _indices_by_key(recording_keys: Sequence[str], pool: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    allowed = set(int(value) for value in pool)
    for index, key in enumerate(recording_keys):
        if index in allowed:
            grouped.setdefault(str(key), []).append(index)
    return {key: np.asarray(values, dtype=np.int64) for key, values in grouped.items()}


def _episode_support(
    arrays: Mapping[str, np.ndarray],
    query_indices: np.ndarray,
    pool: np.ndarray,
    *,
    windows: int,
    rng: np.random.Generator,
    wrong: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = tuple(str(value) for value in arrays["recording_keys"])
    grouped = _indices_by_key(keys, pool)
    all_keys = sorted(grouped)
    eeg: list[np.ndarray] = []
    latent: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for query in query_indices:
        query_key = keys[int(query)]
        if wrong:
            candidates = [key for key in all_keys if key != query_key]
            if not candidates:
                raise RouteBlockedError("outer-training cell has no wrong support donor")
            selected_key = candidates[int(rng.integers(0, len(candidates)))]
            candidates_indices = grouped[selected_key]
        else:
            candidates_indices = grouped.get(query_key, np.empty(0, dtype=np.int64))
            candidates_indices = candidates_indices[candidates_indices != int(query)]
            if candidates_indices.size == 0:
                raise RouteBlockedError("training support/query cannot be made window-disjoint")
        selected = rng.choice(candidates_indices, size=windows, replace=candidates_indices.size < windows)
        if int(query) in set(int(value) for value in selected):
            raise AssertionError("support/query index overlap")
        eeg.append(arrays["observed"][selected])
        latent.append(arrays["target"][selected])
        valid.append(arrays["valid"][selected])
    return np.stack(eeg), np.stack(latent), np.stack(valid)


def _tensor_batch(arrays: Mapping[str, np.ndarray], indices: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(arrays["observed"][indices], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["target"][indices], device=device, dtype=torch.float32),
        torch.as_tensor(arrays["valid"][indices], device=device, dtype=torch.bool),
    )


def _support_tensors(value: tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(value[0], device=device, dtype=torch.float32),
        torch.as_tensor(value[1], device=device, dtype=torch.float32),
        torch.as_tensor(value[2], device=device, dtype=torch.bool),
    )


def _masked_mse(predicted: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    weight = valid[:, None].to(predicted.dtype)
    return ((predicted - target).square() * weight).sum() / (weight.sum() * target.shape[1]).clamp_min(1)


def _train_raw_support(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    row = fold_rows()[task_index]
    checkpoint = _checkpoint(config, row)
    summary_path = checkpoint.parent / "result_summary.json"
    if checkpoint.is_file() and summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed_raw_support_training":
            return {**prior, "resume_action": "skipped_completed"}
    prepared = _prepared_v3(config, row)
    arrays = _subject_arrays(prepared)
    train_indices, validation_indices = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    model_config, diffusion_config, support_config = _model_configs(prepared, config)
    device = torch.device("cuda", 0)
    seed = int(row["seed"])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 17)
    diffusion = RawSupportTokenDiffusion(model_config, diffusion_config, support_config).to(device)
    deterministic = RawSupportTokenDeterministic(model_config, support_config).to(device)
    training = _mapping(config, "training")
    optimizer_d = AdamW(diffusion.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    optimizer_u = AdamW(deterministic.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    ema = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}
    best_score = float("inf"); best_d = best_u = None; best_step = 0
    curve: list[dict[str, Any]] = []
    batch_size = int(training["batch_size"])
    support_windows = int(training["support_windows_per_episode"])
    margin = float(_mapping(config, "screen")["match_wrong_margin"])
    started = time.perf_counter()
    for step in range(1, int(training["maximum_updates"]) + 1):
        indices = rng.choice(train_indices, size=batch_size, replace=train_indices.size < batch_size)
        y, target, valid = _tensor_batch(arrays, indices, device)
        matching = _support_tensors(_episode_support(arrays, indices, train_indices, windows=support_windows, rng=rng, wrong=False), device)
        wrong = _support_tensors(_episode_support(arrays, indices, train_indices, windows=support_windows, rng=rng, wrong=True), device)
        present = (torch.rand(batch_size, device=device, generator=generator) >= support_config.context_dropout_probability).float()
        matching_drop = (matching[0] * present[:, None, None, None], matching[1] * present[:, None, None, None], matching[2])
        optimizer_d.zero_grad(set_to_none=True)
        loss_d, diagnostics = diffusion.training_loss(
            target, observed=y, support_eeg=matching_drop[0], support_artifact_latent=matching_drop[1],
            support_valid_time_mask=matching_drop[2], context_present=present,
            valid_time_mask=valid, generator=generator,
        )
        # A small fixed counterfactual objective makes ignoring support costly.
        fixed_t = torch.full((batch_size,), diffusion.num_timesteps // 2, device=device, dtype=torch.long)
        fixed_noise = torch.randn(target.shape, device=device, generator=generator)
        xt = diffusion.q_sample(target * valid[:, None], fixed_t, fixed_noise * valid[:, None])
        truth_v = diffusion.v_target(target * valid[:, None], fixed_noise * valid[:, None], fixed_t)
        match_v = diffusion.predict_v(
            xt, fixed_t, observed=y, support_eeg=matching[0], support_artifact_latent=matching[1],
            support_valid_time_mask=matching[2], context_present=torch.ones_like(present), valid_time_mask=valid,
        )
        wrong_v = diffusion.predict_v(
            xt, fixed_t, observed=y, support_eeg=wrong[0], support_artifact_latent=wrong[1],
            support_valid_time_mask=wrong[2], context_present=torch.ones_like(present), valid_time_mask=valid,
        )
        match_error = _masked_mse(match_v, truth_v, valid)
        wrong_error = _masked_mse(wrong_v, truth_v, valid)
        ranking_d = torch.relu(torch.as_tensor(margin, device=device) + match_error - wrong_error)
        total_d = loss_d + 0.1 * ranking_d
        total_d.backward()
        gradient_d = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer_d.step()
        optimizer_u.zero_grad(set_to_none=True)
        prediction_u = deterministic(
            observed=y, support_eeg=matching_drop[0], support_artifact_latent=matching_drop[1],
            support_valid_time_mask=matching_drop[2], context_present=present, valid_time_mask=valid,
        )
        match_u = deterministic(
            observed=y, support_eeg=matching[0], support_artifact_latent=matching[1],
            support_valid_time_mask=matching[2], context_present=torch.ones_like(present), valid_time_mask=valid,
        )
        wrong_u = deterministic(
            observed=y, support_eeg=wrong[0], support_artifact_latent=wrong[1],
            support_valid_time_mask=wrong[2], context_present=torch.ones_like(present), valid_time_mask=valid,
        )
        loss_u = _masked_mse(prediction_u, target, valid)
        ranking_u = torch.relu(torch.as_tensor(margin, device=device) + _masked_mse(match_u, target, valid) - _masked_mse(wrong_u, target, valid))
        total_u = loss_u + 0.1 * ranking_u
        total_u.backward()
        gradient_u = torch.nn.utils.clip_grad_norm_(deterministic.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True)
        optimizer_u.step()
        with torch.no_grad():
            decay = float(_mapping(config, "diffusion")["ema_decay"])
            for key, value in diffusion.state_dict().items():
                ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        if step == 1 or step % 100 == 0:
            curve.append({
                "step": step, "diffusion_loss": float(loss_d.detach()),
                "deterministic_loss": float(loss_u.detach()),
                "diffusion_ranking": float(ranking_d.detach()),
                "deterministic_ranking": float(ranking_u.detach()),
                "diffusion_gradient": float(gradient_d), "deterministic_gradient": float(gradient_u),
                **{key: float(value) for key, value in diagnostics.items()},
            })
        if step % int(training["validation_interval_updates"]) == 0 or step == int(training["maximum_updates"]):
            selected = validation_indices
            yv, targetv, validv = _tensor_batch(arrays, selected, device)
            supportv = _support_tensors(_episode_support(arrays, selected, validation_indices, windows=support_windows, rng=np.random.default_rng(seed + 9000 + step), wrong=False), device)
            live = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}
            diffusion.load_state_dict(ema); diffusion.eval(); deterministic.eval()
            validation_generator = torch.Generator(device=device).manual_seed(seed + 91000)
            with torch.no_grad():
                score_d = float(diffusion.training_loss(
                    targetv, observed=yv, support_eeg=supportv[0], support_artifact_latent=supportv[1],
                    support_valid_time_mask=supportv[2], context_present=torch.ones(targetv.shape[0], device=device),
                    valid_time_mask=validv, generator=validation_generator,
                )[1]["x0_mse"].cpu())
                predv = deterministic(
                    observed=yv, support_eeg=supportv[0], support_artifact_latent=supportv[1],
                    support_valid_time_mask=supportv[2], context_present=torch.ones(targetv.shape[0], device=device), valid_time_mask=validv,
                )
                score_u = float(_masked_mse(predv, targetv, validv).cpu())
            diffusion.load_state_dict(live); diffusion.train(); deterministic.train()
            score = score_d + score_u
            curve.append({"step": step, "ema_validation_x0_mse": score_d, "deterministic_validation_mse": score_u, "selection_score": score})
            if score < best_score:
                best_score = score; best_step = step
                best_d = {key: value.detach().cpu().clone() for key, value in ema.items()}
                best_u = {key: value.detach().cpu().clone() for key, value in deterministic.state_dict().items()}
    if best_d is None or best_u is None:
        raise AssertionError("raw-support validation selected no checkpoint")
    _save_checkpoint(checkpoint, {
        "protocol_id": PROTOCOL, "route": dict(row), "model_config": model_config.__dict__,
        "diffusion_config": diffusion_config.__dict__, "support_config": support_config.__dict__,
        "diffusion_ema": best_d, "deterministic": best_u, "best_step": best_step,
        "latent_mean": prepared.latent_normalizer.mean,
        "latent_standard_deviation": prepared.latent_normalizer.standard_deviation,
        "population_normalized_transfer": prepared.population_context.normalized_transfer,
    })
    _write_csv(checkpoint.parent / "training_curve.csv", curve)
    summary = {
        "status": "completed_raw_support_training", "route": "P_A_RAW_SUPPORT_TOKENS",
        **dict(row), "checkpoint": str(checkpoint), "best_step": best_step,
        "runtime_seconds": time.perf_counter() - started,
        "query_external_signal_used": False, "support_query_window_overlap": False,
    }
    _write_json(summary_path, summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def technical_check(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """One real-record GPU validity check; never used as scientific evidence."""

    row = fold_rows()[0]
    prepared = _prepared_v3(config, row)
    arrays = _subject_arrays(prepared)
    training, _ = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    batch_indices = training[: min(4, training.size)]
    if batch_indices.size < 2:
        raise RouteBlockedError("technical check needs two real training windows")
    device = torch.device("cuda", 0)
    model_config, diffusion_config, support_config = _model_configs(prepared, config)
    torch.manual_seed(20260811); torch.cuda.manual_seed_all(20260811)
    generator = torch.Generator(device=device).manual_seed(20260828)
    rng = np.random.default_rng(20260811)
    diffusion = RawSupportTokenDiffusion(model_config, diffusion_config, support_config).to(device)
    deterministic = RawSupportTokenDeterministic(model_config, support_config).to(device)
    y, target, valid = _tensor_batch(arrays, batch_indices, device)
    matching = _support_tensors(
        _episode_support(arrays, batch_indices, training, windows=4, rng=rng, wrong=False), device
    )
    wrong = _support_tensors(
        _episode_support(arrays, batch_indices, training, windows=4, rng=rng, wrong=True), device
    )
    present = torch.ones(y.shape[0], device=device)
    optimizer = AdamW((*diffusion.parameters(), *deterministic.parameters()), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss_d, _ = diffusion.training_loss(
        target, observed=y, support_eeg=matching[0], support_artifact_latent=matching[1],
        support_valid_time_mask=matching[2], context_present=present,
        valid_time_mask=valid, generator=generator,
    )
    estimate = deterministic(
        observed=y, support_eeg=matching[0], support_artifact_latent=matching[1],
        support_valid_time_mask=matching[2], context_present=present, valid_time_mask=valid,
    )
    loss_u = _masked_mse(estimate, target, valid)
    total = loss_d + loss_u
    total.backward()
    gradient = torch.nn.utils.clip_grad_norm_(
        (*diffusion.parameters(), *deterministic.parameters()), 1.0, error_if_nonfinite=True
    )
    optimizer.step()
    fixed_t = torch.full((y.shape[0],), diffusion.num_timesteps // 2, device=device, dtype=torch.long)
    fixed_state = torch.randn(target.shape, device=device, generator=generator) * valid[:, None]
    with torch.no_grad():
        matching_v = diffusion.predict_v(
            fixed_state, fixed_t, observed=y, support_eeg=matching[0],
            support_artifact_latent=matching[1], support_valid_time_mask=matching[2],
            context_present=present, valid_time_mask=valid,
        )
        wrong_v = diffusion.predict_v(
            fixed_state, fixed_t, observed=y, support_eeg=wrong[0],
            support_artifact_latent=wrong[1], support_valid_time_mask=wrong[2],
            context_present=present, valid_time_mask=valid,
        )
    context_difference = float((matching_v - wrong_v).square().mean().sqrt().cpu())
    if context_difference <= 1e-7:
        raise AssertionError("raw-support route ignores context above floating-point noise")
    temporary_checkpoint = run_dir / "technical_checkpoint.pt"
    _save_checkpoint(temporary_checkpoint, {
        "diffusion": diffusion.state_dict(), "deterministic": deterministic.state_dict(),
        "optimizer": optimizer.state_dict(),
    })
    reloaded = torch.load(temporary_checkpoint, map_location=device, weights_only=False)
    diffusion_clone = RawSupportTokenDiffusion(model_config, diffusion_config, support_config).to(device)
    deterministic_clone = RawSupportTokenDeterministic(model_config, support_config).to(device)
    diffusion_clone.load_state_dict(reloaded["diffusion"])
    deterministic_clone.load_state_dict(reloaded["deterministic"])
    if not math.isfinite(float(total.detach())) or not math.isfinite(float(gradient)):
        raise FloatingPointError("technical check loss/gradient is non-finite")
    summary = {
        "status": "passed_real_record_technical_check",
        "scientific_evidence": False,
        "real_dataset": "klados",
        "real_window_count": int(batch_indices.size),
        "loss": float(total.detach().cpu()),
        "gradient_norm": float(gradient),
        "context_swap_rms": context_difference,
        "checkpoint_reload": True,
        "query_external_signal_used": False,
    }
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _load_raw_support_models(config: Mapping[str, Any], row: Mapping[str, Any], device: torch.device):
    checkpoint = _checkpoint(config, row)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_config = ArtifactLatentModelConfig(**payload["model_config"])
    diffusion_config = ArtifactLatentDiffusionConfig(**payload["diffusion_config"])
    support_config = RawSupportConfig(**payload["support_config"])
    diffusion = RawSupportTokenDiffusion(model_config, diffusion_config, support_config).to(device)
    deterministic = RawSupportTokenDeterministic(model_config, support_config).to(device)
    diffusion.load_state_dict(payload["diffusion_ema"])
    deterministic.load_state_dict(payload["deterministic"])
    diffusion.eval(); deterministic.eval()
    return diffusion, deterministic, payload, checkpoint


def _fixed_support(
    support: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch: int,
    windows: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    eeg, latent, valid = support
    if eeg.shape[0] < 1:
        raise RouteBlockedError("support contains no complete window")
    selected = np.arange(windows) % eeg.shape[0]
    return (
        torch.as_tensor(eeg[selected], device=device, dtype=torch.float32)[None].expand(batch, -1, -1, -1),
        torch.as_tensor(latent[selected], device=device, dtype=torch.float32)[None].expand(batch, -1, -1, -1),
        torch.as_tensor(valid[selected], device=device, dtype=torch.bool)[None].expand(batch, -1, -1),
    )


def _training_donor_supports(prepared: PreparedSubjectArtifactFold, minimum: int = 3) -> list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    arrays = _subject_arrays(prepared)
    keys = np.asarray(arrays["recording_keys"])
    result = []
    for key in sorted(set(str(value) for value in keys)):
        selected = np.flatnonzero(keys == key)
        if selected.size:
            result.append((key, (arrays["observed"][selected], arrays["target"][selected], arrays["valid"][selected])))
        if len(result) >= minimum:
            break
    if len(result) < minimum:
        raise RouteBlockedError("exact cell has fewer than three training support donors")
    return result


def _population_support(prepared: PreparedSubjectArtifactFold, windows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = _subject_arrays(prepared)
    training, _ = _split_indices(tuple(str(value) for value in arrays["recording_keys"]))
    if training.size < 1:
        raise RouteBlockedError("population support pool is empty")
    # Deterministically spread the small support budget over outer-training data.
    positions = np.linspace(0, training.size - 1, windows).round().astype(np.int64)
    selected = training[positions]
    return arrays["observed"][selected], arrays["target"][selected], arrays["valid"][selected]


def _shuffled_support(support: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg, latent, valid = support
    if latent.shape[0] < 2:
        shifted = np.roll(latent, max(1, latent.shape[-1] // 3), axis=-1)
    else:
        shifted = np.roll(latent, 1, axis=0)
    return eeg.copy(), shifted.copy(), valid.copy()


def _population_anchor(config: Mapping[str, Any], prepared: PreparedSubjectArtifactFold, row: Mapping[str, Any], device: torch.device):
    value = yaml.safe_load((CODE_ROOT / "configs/cgdr/mainline_subject_residual_diffusion.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("mainline POP config is invalid")
    value = deepcopy(value)
    value["outputs"]["root"] = "/home/infres/yinwang/denoiseNet/results/cgdr/mainline_subject_residual_diffusion"
    anchor, _, _, _, checkpoint = _load_population_baseline(value, prepared, row, device)
    return anchor, checkpoint


def _restore_shared_population_coordinate(y: Tensor, latent: Tensor, valid: Tensor, payload: Mapping[str, Any]) -> Tensor:
    batch = y.shape[0]
    transfer = torch.as_tensor(payload["population_normalized_transfer"], device=y.device, dtype=y.dtype)
    transfer = transfer[None].expand(batch, -1, -1)
    mean = torch.as_tensor(payload["latent_mean"], device=y.device, dtype=y.dtype)[None].expand(batch, -1)
    scale = torch.as_tensor(payload["latent_standard_deviation"], device=y.device, dtype=y.dtype)[None].expand(batch, -1)
    output_mask = valid[:, None].to(y.dtype).expand(-1, y.shape[1], -1)
    delta = canonical_artifact_delta(
        latent, normalized_transfer=transfer, latent_mean=mean,
        latent_standard_deviation=scale, output_mask=output_mask,
    )
    return (y - delta) * output_mask


def _fit_adapters(
    diffusion: RawSupportTokenDiffusion,
    deterministic: RawSupportTokenDeterministic,
    support: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[DirectSupportAdapter, DirectSupportAdapter, dict[str, float]]:
    eeg, target, valid = support
    if eeg.shape[0] < 2:
        raise RouteBlockedError("direct adapter requires at least two support windows")
    split = max(1, eeg.shape[0] // 2)
    fit = np.arange(split)
    validation = np.arange(split, eeg.shape[0])
    if validation.size == 0:
        validation = fit
    y = torch.as_tensor(eeg, device=device, dtype=torch.float32)
    truth = torch.as_tensor(target, device=device, dtype=torch.float32)
    mask = torch.as_tensor(valid, device=device, dtype=torch.bool)
    support_windows = int(_mapping(config, "training")["support_windows_per_episode"])
    zero_support = (
        torch.zeros(y.shape[0], support_windows, y.shape[1], y.shape[2], device=device),
        torch.zeros(y.shape[0], support_windows, truth.shape[1], truth.shape[2], device=device),
        torch.ones(y.shape[0], support_windows, y.shape[2], device=device, dtype=torch.bool),
    )
    diff_adapter = DirectSupportAdapter(truth.shape[1], rank=int(_mapping(config, "model")["adapter_rank"])).to(device)
    det_adapter = DirectSupportAdapter(truth.shape[1], rank=int(_mapping(config, "model")["adapter_rank"])).to(device)
    optimizer_d = AdamW(diff_adapter.parameters(), lr=float(_mapping(config, "model")["adapter_learning_rate"]))
    optimizer_u = AdamW(det_adapter.parameters(), lr=float(_mapping(config, "model")["adapter_learning_rate"]))
    generator = torch.Generator(device=device).manual_seed(seed)
    for parameter in diffusion.parameters():
        parameter.requires_grad_(False)
    for parameter in deterministic.parameters():
        parameter.requires_grad_(False)
    diffusion.eval(); deterministic.eval()
    for _ in range(int(_mapping(config, "model")["adapter_updates"])):
        index = torch.as_tensor(fit, device=device)
        timestep = torch.randint(0, diffusion.num_timesteps, (fit.size,), device=device, generator=generator)
        noise = torch.randn(truth[index].shape, device=device, generator=generator)
        x0 = truth[index] * mask[index, None]
        xt = diffusion.q_sample(x0, timestep, noise * mask[index, None])
        v_truth = diffusion.v_target(x0, noise * mask[index, None], timestep)
        with torch.no_grad():
            base_v = diffusion.predict_v(
                xt, timestep, observed=y[index], support_eeg=zero_support[0][index],
                support_artifact_latent=zero_support[1][index], support_valid_time_mask=zero_support[2][index],
                context_present=torch.zeros(fit.size, device=device), valid_time_mask=mask[index],
            )
            base_u = deterministic(
                observed=y[index], support_eeg=zero_support[0][index],
                support_artifact_latent=zero_support[1][index], support_valid_time_mask=zero_support[2][index],
                context_present=torch.zeros(fit.size, device=device), valid_time_mask=mask[index],
            )
        optimizer_d.zero_grad(set_to_none=True)
        loss_d = _masked_mse(diff_adapter(base_v, mask[index]), v_truth, mask[index])
        loss_d.backward(); torch.nn.utils.clip_grad_norm_(diff_adapter.parameters(), 1.0, error_if_nonfinite=True); optimizer_d.step()
        optimizer_u.zero_grad(set_to_none=True)
        loss_u = _masked_mse(det_adapter(base_u, mask[index]), truth[index], mask[index])
        loss_u.backward(); torch.nn.utils.clip_grad_norm_(det_adapter.parameters(), 1.0, error_if_nonfinite=True); optimizer_u.step()
    with torch.no_grad():
        index = torch.as_tensor(validation, device=device)
        base_u = deterministic(
            observed=y[index], support_eeg=zero_support[0][index], support_artifact_latent=zero_support[1][index],
            support_valid_time_mask=zero_support[2][index], context_present=torch.zeros(validation.size, device=device), valid_time_mask=mask[index],
        )
        base_loss = float(_masked_mse(base_u, truth[index], mask[index]).cpu())
        adapted_loss = float(_masked_mse(det_adapter(base_u, mask[index]), truth[index], mask[index]).cpu())
    return diff_adapter.eval(), det_adapter.eval(), {"support_validation_base_mse": base_loss, "support_validation_adapted_mse": adapted_loss}


@torch.no_grad()
def _infer_context(
    route: str,
    diffusion: RawSupportTokenDiffusion,
    deterministic: RawSupportTokenDeterministic,
    payload: Mapping[str, Any],
    y: Tensor,
    valid: Tensor,
    support: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    sample_seeds: Sequence[int],
    config: Mapping[str, Any],
    diff_adapter: DirectSupportAdapter | None = None,
    det_adapter: DirectSupportAdapter | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    windows = int(_mapping(config, "training")["support_windows_per_episode"])
    if support is None:
        eeg = torch.zeros(y.shape[0], windows, y.shape[1], y.shape[2], device=y.device)
        latent = torch.zeros(y.shape[0], windows, diffusion.model_config.latent_channels, y.shape[2], device=y.device)
        support_mask = torch.ones(y.shape[0], windows, y.shape[2], device=y.device, dtype=torch.bool)
        present = torch.zeros(y.shape[0], device=y.device)
    else:
        eeg, latent, support_mask = _fixed_support(support, y.shape[0], windows, y.device)
        present = torch.ones(y.shape[0], device=y.device)
    if route == "P_D_SUPPORT_STAT_CONTROL" and support is not None:
        control = SupportStatisticControl()
        y_model = control(y, eeg, support_mask)
        eeg_model = control.normalize_support(eeg, support_mask)
    else:
        y_model, eeg_model = y, eeg
    det_latent = deterministic(
        observed=y_model, support_eeg=eeg_model, support_artifact_latent=latent,
        support_valid_time_mask=support_mask, context_present=present, valid_time_mask=valid,
    )
    if det_adapter is not None:
        det_latent = det_adapter(det_latent, valid)
    samples = diffusion.latent_samples(
        observed=y_model, support_eeg=eeg_model, support_artifact_latent=latent,
        support_valid_time_mask=support_mask, context_present=present,
        valid_time_mask=valid, sample_seeds=sample_seeds,
        ddim_steps=int(_mapping(config, "diffusion")["ddim_steps"]), v_adapter=diff_adapter,
    )
    diff_latent = samples.mean(dim=0)
    return _restore_shared_population_coordinate(y, det_latent, valid, payload), _restore_shared_population_coordinate(y, diff_latent, valid, payload), samples


@torch.no_grad()
def _infer_context_batched(
    route: str,
    diffusion: RawSupportTokenDiffusion,
    deterministic: RawSupportTokenDeterministic,
    payload: Mapping[str, Any],
    observed: np.ndarray,
    valid: np.ndarray,
    support: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    *,
    sample_seeds: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
    diff_adapter: DirectSupportAdapter | None = None,
    det_adapter: DirectSupportAdapter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(_mapping(config, "training")["batch_size"])
    deterministic_chunks: list[np.ndarray] = []
    diffusion_chunks: list[np.ndarray] = []
    sample_chunks: list[np.ndarray] = []
    for start in range(0, observed.shape[0], batch_size):
        stop = min(start + batch_size, observed.shape[0])
        y = torch.as_tensor(observed[start:stop], device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid[start:stop], device=device, dtype=torch.bool)
        deterministic_output, diffusion_output, samples = _infer_context(
            route, diffusion, deterministic, payload, y, mask, support,
            sample_seeds=sample_seeds, config=config,
            diff_adapter=diff_adapter, det_adapter=det_adapter,
        )
        deterministic_chunks.append(deterministic_output.cpu().numpy())
        diffusion_chunks.append(diffusion_output.cpu().numpy())
        sample_chunks.append(samples.cpu().numpy())
    return (
        np.concatenate(deterministic_chunks).astype(np.float32),
        np.concatenate(diffusion_chunks).astype(np.float32),
        np.concatenate(sample_chunks, axis=1).astype(np.float32),
    )


def _unit_outputs(
    config: Mapping[str, Any],
    prepared: PreparedSubjectArtifactFold,
    row: Mapping[str, Any],
    route: str,
    *,
    unit_key: str,
    observed: np.ndarray,
    valid: np.ndarray,
    matching_support: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    diffusion, deterministic, payload, checkpoint = _load_raw_support_models(config, row, device)
    anchor, population_checkpoint = _population_anchor(config, prepared, row, device)
    support_windows = int(_mapping(config, "training")["support_windows_per_episode"])
    population_support = _population_support(prepared, support_windows)
    wrong_supports = _training_donor_supports(prepared, minimum=int(_mapping(config, "screen")["wrong_donors"]))
    shuffled_support = _shuffled_support(matching_support)
    contexts: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None]] = [
        ("POP", population_support),
        ("MATCH", matching_support),
        *((f"WRONG-{index}", support) for index, (_, support) in enumerate(wrong_supports, 1)),
        ("SHUFFLED", shuffled_support),
        ("NO-SUPPORT", None),
    ]
    y = torch.as_tensor(observed, device=device, dtype=torch.float32)
    mask = torch.as_tensor(valid, device=device, dtype=torch.bool)
    with torch.no_grad():
        strong_population = anchor(y, mask).cpu().numpy().astype(np.float32)
    outputs: dict[str, np.ndarray] = {"RAW": observed.astype(np.float32), "STRONG-POP": strong_population}
    posterior_archives: dict[str, np.ndarray] = {}
    adapter_diagnostics: dict[str, Any] = {}
    started = time.perf_counter()
    common_seeds = participant_sample_seeds(unit_key, int(row["seed"]), count=8)
    for context_name, support in contexts:
        diff_adapter = det_adapter = None
        model_support = support
        if route == "P_B_DIRECT_SUPPORT_ADAPTER" and support is not None:
            diff_adapter, det_adapter, diagnostic = _fit_adapters(
                diffusion, deterministic, support, device, config,
                int(common_seeds[0] % (2**31 - 1)),
            )
            adapter_diagnostics[context_name] = diagnostic
            # Direct adaptation is applied to the population/no-support backbone;
            # support does not also enter through P-A's token conditioner.
            model_support = None
        deterministic_output, diffusion_output, samples = _infer_context_batched(
            route, diffusion, deterministic, payload, observed, valid, model_support,
            sample_seeds=common_seeds, config=config, device=device,
            diff_adapter=diff_adapter, det_adapter=det_adapter,
        )
        outputs[f"DET-{context_name}"] = deterministic_output
        outputs[f"DIFF-{context_name}"] = diffusion_output
        posterior_archives[context_name] = samples
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("route screen produced non-finite EEG")
    maximum_rms = max(
        float(np.sqrt(np.mean(value.astype(np.float64) ** 2)))
        for value in outputs.values()
    )
    raw_rms = float(np.sqrt(np.mean(observed.astype(np.float64) ** 2)))
    if maximum_rms > max(raw_rms, 1e-8) * 10.0:
        raise FloatingPointError("route screen output scale exceeds the 10x engineering bound")
    return outputs, {
        "checkpoint": str(checkpoint),
        "population_checkpoint": str(population_checkpoint),
        "wrong_donors": [name for name, _ in wrong_supports],
        "common_random_numbers": True,
        "runtime_seconds": time.perf_counter() - started,
        "adapter_diagnostics": adapter_diagnostics,
        "posterior_samples": posterior_archives,
    }


def _screen_route(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    task = task_rows()[task_index]
    route = str(task["route"])
    row = {key: task[key] for key in ("dataset", "fold_index", "seed")}
    if route == "P_A_RAW_SUPPORT_TOKENS":
        _train_raw_support(config, run_dir / "training", int(task_index))
    prepared = _prepared_v3(config, row)
    device = torch.device("cuda", 0)
    root = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    output_root = root / {
        "P_A_RAW_SUPPORT_TOKENS": "raw_support_tokens",
        "P_B_DIRECT_SUPPORT_ADAPTER": "support_lora",
        "P_D_SUPPORT_STAT_CONTROL": "support_stat_control",
    }[route] / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}"
    metrics_path = output_root / "metrics.csv"
    summary_path = output_root / "result_summary.json"
    if metrics_path.is_file() and summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed_full_real_one_seed_screen":
            return {**prior, "resume_action": "skipped_completed"}
    rows: list[dict[str, Any]] = []
    arrays_root = root / "server_arrays" / route / str(row["dataset"]) / f"fold_{int(row['fold_index']):02d}"
    arrays_root.mkdir(parents=True, exist_ok=True)
    base = _base_config(config)
    if row["dataset"] == "klados":
        for unit_key, mechanism, _, _ in _klados_eval_records(base):
            matching_support = _klados_matching_support(base, prepared, mechanism, unit_key)
            outputs, diagnostics = _unit_outputs(
                config, prepared, row, route, unit_key=unit_key,
                observed=mechanism.observed_windows.astype(np.float32),
                valid=mechanism.valid_time_weight.astype(bool), matching_support=matching_support,
                device=device,
            )
            np.savez_compressed(arrays_root / f"{unit_key}.npz", **{key.replace("-", "_"): value for key, value in outputs.items()})
            for method, output in outputs.items():
                rows.append({
                    "route": route, "dataset": "klados", "unit_id": unit_key,
                    "exact_cell": prepared.fold.layout_id, "training_seed": int(row["seed"]),
                    "method": method, "status": "success", "statistical_unit": "source_record",
                    "screening_only": True, "query_external_signal_used": False,
                    **_paired_metrics(mechanism.observed_windows, mechanism.clean_windows, output, mechanism.valid_time_weight.astype(bool)),
                })
    else:
        for unit_key, heldout in prepared.heldout.items():
            method_names = (
                "RAW", "STRONG-POP", "DET-POP", "DET-MATCH", "DIFF-POP", "DIFF-MATCH",
                "DIFF-WRONG-1", "DIFF-WRONG-2", "DIFF-WRONG-3", "DIFF-SHUFFLED", "DIFF-NO-SUPPORT",
            )
            try:
                matching_support = _sge_matching_support(base, prepared, int(row["fold_index"]), unit_key)
                outputs, diagnostics = _unit_outputs(
                    config, prepared, row, route, unit_key=unit_key,
                    observed=heldout.query.observed.astype(np.float32),
                    valid=heldout.query.valid_time_mask.astype(bool), matching_support=matching_support,
                    device=device,
                )
            except RouteBlockedError as error:
                for method in method_names:
                    rows.append({
                        "route": route, "dataset": "sgeyesub", "unit_id": unit_key,
                        "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}",
                        "study": prepared.fold.study, "training_seed": int(row["seed"]),
                        "method": method, "status": "blocked_incompatible_support", "failure_reason": str(error),
                        "statistical_unit": "participant_stem", "outputs_frozen_before_query_scoring": False,
                    })
                continue
            np.savez_compressed(arrays_root / f"{unit_key.replace('/', '__')}.npz", **{key.replace("-", "_"): value for key, value in outputs.items()})
            # The evaluator-only EOG and annotations are opened only after all
            # deployable outputs for this unit have been frozen.
            annotated = _annotation_opener(base, prepared, unit_key)()
            annotations = annotated.query_annotations
            if annotations is None:
                raise AssertionError("SGE query annotations unavailable after output freeze")
            observed_continuous = _continuous(heldout.query.observed)
            input_rms = float(np.sqrt(np.mean(observed_continuous.astype(np.float64) ** 2)))
            for method, output in outputs.items():
                output_continuous = _continuous(output)
                metric = _evaluate_output(
                    method_id=method, output=output_continuous, observed=observed_continuous,
                    matching_projector=heldout.matching.projector,
                    population_projector=prepared.population_context.projector,
                    query_eog=annotations.external_eog, artifactclasses=annotations.artifactclasses,
                    predicted_contamination=None, trial_labels=annotations.trial_labels,
                    samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz),
                    minimum_trials_per_condition=2, status="success", operator_source=route,
                    gamma=None, fallback_used=(method == "STRONG-POP"), uses_query_external_eog=False,
                )
                output_rms = float(np.sqrt(np.mean(output_continuous.astype(np.float64) ** 2)))
                rows.append({
                    "route": route, "dataset": "sgeyesub", "unit_id": unit_key,
                    "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}",
                    "study": prepared.fold.study, "training_seed": int(row["seed"]), "method": method,
                    **metric, "output_input_RMS_ratio": output_rms / max(input_rms, np.finfo(np.float64).eps),
                    "statistical_unit": "participant_stem", "screening_only": True,
                    "outputs_frozen_before_query_scoring": True,
                })
    _write_csv(metrics_path, rows)
    summary = {
        "status": "completed_full_real_one_seed_screen", "route": route, **dict(row),
        "unit_count": len({str(value["unit_id"]) for value in rows}), "metric_rows": len(rows),
        "scientific_role": "complete_real_development_route_screen_one_seed",
        "query_external_signal_used_for_inference": False, "result": str(metrics_path),
    }
    _write_json(summary_path, summary); _write_json(run_dir / "result_summary.json", summary)
    return summary


def write_tasks(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    output = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    rows = task_rows()
    _write_csv(output / "task_list.csv", rows)
    summary = {
        "status": "completed_frozen_task_list",
        "tasks": len(rows),
        "routes": sorted({row["route"] for row in rows}),
        "datasets": sorted({row["dataset"] for row in rows}),
        "array_spec": f"0-{len(rows) - 1}%8",
    }
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def run_stage(config: Mapping[str, Any], run_dir: Path, stage: str, task_index: int | None) -> Mapping[str, Any]:
    if config.get("protocol_id") != PROTOCOL or int(config.get("harness_level", -1)) != 1:
        raise ValueError("v3 protocol or harness level changed")
    if stage == "j0-audit":
        return audit(config, run_dir)
    if stage == "j1-tasks":
        return write_tasks(config, run_dir)
    if stage == "j2-tech":
        return technical_check(config, run_dir)
    route_ranges = {
        "j3-raw-support": (0, 25, "P_A_RAW_SUPPORT_TOKENS"),
        "j4-lora": (26, 51, "P_B_DIRECT_SUPPORT_ADAPTER"),
        "j6-control": (52, 77, "P_D_SUPPORT_STAT_CONTROL"),
    }
    if stage in route_ranges:
        if task_index is None:
            raise ValueError(f"{stage} requires a Slurm array task")
        lower, upper, route = route_ranges[stage]
        if not lower <= int(task_index) <= upper or task_rows()[int(task_index)]["route"] != route:
            raise ValueError(f"array task {task_index} does not belong to {stage}")
        return _screen_route(config, run_dir, int(task_index))
    raise ValueError(f"unsupported v3 stage: {stage}; task={task_index}")
