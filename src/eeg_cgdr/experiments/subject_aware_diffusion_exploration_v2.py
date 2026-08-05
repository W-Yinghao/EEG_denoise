"""Wide subject-aware diffusion exploration v2.

This is a development-only screen.  Historical result files are read-only;
all corrected audits and new route outputs live under the v2 result root.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import AdamW
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eeg_cgdr.experiments.subject_artifact_data import (
    _calibration_samples,
    _eog_order,
    _load_frozen_config,
    _loaded_record,
    _unified_fold_route,
)
from eeg_cgdr.experiments.sgeyesub_diffusion_runner import _prepare_fold
from eeg_cgdr.models.artifact_subspace_diffusion import aligned_artifact_basis
from eeg_cgdr.models.artifact_subspace_diffusion import (
    participant_sample_seeds,
    reconstruct_from_subspace,
)
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)
from eeg_cgdr.models.subject_aware_wide_v2 import (
    activity_gate_from_eeg_latent,
    canonical_eog_latent,
    fir_coefficients_from_lag_major,
    fir_coefficients_to_lag_major,
    fir_full_replacement,
    full_c_subject_residual,
    lazy_subject_residual,
    physical_eog_latent,
    SupportFiLMArtifactLatentDiffusion,
    SupportLoRAArtifactLatentDiffusion,
)


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet"))
PROTOCOL = "subject_aware_diffusion_wide_exploration_v2"
SEEDS = (20260811, 20260812, 20260813)
OLD_ROOT = CODE_ROOT / "results/cgdr/subject_artifact_subspace_diffusion"


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _load(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    if config.get("protocol_id") != PROTOCOL or int(config.get("harness_level", -1)) != 1:
        raise ValueError("wide exploration protocol or harness changed")
    base = yaml.safe_load((CODE_ROOT / str(config["base_subject_artifact_config"])).read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base subject-artifact config is invalid")
    root = CODE_ROOT / str(_mapping(config, "outputs")["root"])
    return base, root


def _implementation() -> dict[str, Any]:
    return {
        "git_sha": os.environ.get("DENOISENET_GIT_HEAD", "unknown"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


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


def _read_csvs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                converted: dict[str, Any] = {}
                for key, value in row.items():
                    try:
                        converted[key] = float(value) if value not in (None, "") else value
                    except ValueError:
                        converted[key] = value
                rows.append(converted)
    return rows


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if unique.size != values.size:
        for index, count in enumerate(counts):
            if count > 1:
                ranks[inverse == index] = ranks[inverse == index].mean()
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    left = _ranks(x) if rank else np.asarray(x, dtype=np.float64)
    right = _ranks(y) if rank else np.asarray(y, dtype=np.float64)
    left = left - left.mean(); right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def _risk_auc(errors: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    error = np.asarray(errors, dtype=np.float64)
    score = np.asarray(uncertainty, dtype=np.float64)
    order = np.argsort(score)
    risks = np.asarray([error[order[:count]].mean() for count in range(1, error.size + 1)])
    oracle_order = np.argsort(error)
    oracle = np.asarray([error[oracle_order[:count]].mean() for count in range(1, error.size + 1)])
    random_auc = float(error.mean())
    auc = float(risks.mean())
    scale = max(random_auc - float(oracle.mean()), 1.0e-12)
    return {
        "risk_coverage_auc": auc,
        "normalized_risk_coverage_auc": float((auc - float(oracle.mean())) / scale),
        "excess_risk_auc": float(auc - float(oracle.mean())),
        "random_ranking_auc": random_auc,
        "oracle_ranking_auc": float(oracle.mean()),
    }


def _bootstrap_sources(rows: Sequence[Mapping[str, Any]], key: str, seed: int, replicates: int) -> dict[str, Any]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        samples[index] = rng.choice(values, values.size, replace=True).mean()
    return {"mean": float(values.mean()), "median": float(np.median(values)), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "n_sources": int(values.size)}


def _j0_reaudit(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Correct the legacy axis bug without pretending missing sample arrays exist."""

    _, root = _load(config)
    output = root / "current_result_reaudit"
    rows: list[dict[str, Any]] = []
    for record in [f"sim{value:02d}" for value in range(37, 55) if value not in (44, 45)]:
        arrays = []
        for seed in SEEDS:
            path = OLD_ROOT / "server_arrays/klados" / f"seed_{seed}" / f"{record}.npz"
            if path.is_file():
                arrays.append(np.load(path))
        if len(arrays) != len(SEEDS):
            continue
        clean = np.asarray(arrays[0]["clean"], dtype=np.float64)
        diff = np.mean([np.asarray(value["diff"], dtype=np.float64) for value in arrays], axis=0)
        det_stack = np.stack([np.asarray(value["det"], dtype=np.float64) for value in arrays])
        det = det_stack.mean(0)
        legacy_diff_unc = np.mean([np.asarray(value["diff_uncertainty"], dtype=np.float64) for value in arrays], axis=0)
        det_unc = det_stack.std(0)
        diff_error = np.sqrt(np.mean(np.square(diff - clean), axis=(1, 2)))
        det_error = np.sqrt(np.mean(np.square(det - clean), axis=(1, 2)))
        diff_score = np.mean(legacy_diff_unc, axis=(1, 2))
        # Correct indexing: det_uncertainty is (window, channel, time).
        det_score = np.asarray([float(np.mean(det_unc[window])) for window in range(det_unc.shape[0])])
        diff_risk = _risk_auc(diff_error, diff_score)
        det_risk = _risk_auc(det_error, det_score)
        rows.append({
            "source_record": record,
            "window_count": int(clean.shape[0]),
            "diff_centered_pearson": _correlation(diff_score, diff_error),
            "diff_centered_spearman": _correlation(diff_score, diff_error, rank=True),
            "det_centered_pearson": _correlation(det_score, det_error),
            "det_centered_spearman": _correlation(det_score, det_error, rank=True),
            **{f"diff_{key}": value for key, value in diff_risk.items()},
            **{f"det_{key}": value for key, value in det_risk.items()},
        })
        for value in arrays:
            value.close()
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 old Klados records, found {len(rows)}")
    _write_csv(output / "corrected_legacy_source_uncertainty.csv", rows)

    metric_rows = _read_csvs(sorted((OLD_ROOT / "evaluation/klados").glob("seed_*/metrics.csv")))
    by_unit_seed: list[dict[str, Any]] = []
    for seed, path in zip(SEEDS, sorted((OLD_ROOT / "evaluation/klados").glob("seed_*/metrics.csv"))):
        current = _read_csvs([path])
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in current:
            groups[(str(row["unit_id"]), str(row["method"]))].append(row)
        for unit in sorted({key[0] for key in groups}):
            metrics = {}
            for method in ("POP", "DET-MATCH", "DIFF-MATCH", "DIFF-MATCH-K1"):
                selected = groups.get((unit, method), [])
                if selected:
                    metrics[method] = float(np.mean([float(item["clean_waveform_RRMSE"]) for item in selected]))
            if {"POP", "DET-MATCH", "DIFF-MATCH"}.issubset(metrics):
                by_unit_seed.append({"training_seed": seed, "source_record": unit, "diff_minus_det_utility": metrics["DET-MATCH"] - metrics["DIFF-MATCH"], "diff_minus_pop_utility": metrics["POP"] - metrics["DIFF-MATCH"], "legacy_K1_RRMSE": metrics.get("DIFF-MATCH-K1")})
    _write_csv(output / "per_seed_source_effects.csv", by_unit_seed)
    rng = np.random.default_rng(int(_mapping(config, "statistics")["bootstrap_seed"]))
    source_ids = sorted({str(row["source_record"]) for row in by_unit_seed})
    hierarchical = np.empty(int(_mapping(config, "statistics")["bootstrap_replicates"]), dtype=np.float64)
    lookup = defaultdict(list)
    for row in by_unit_seed:
        lookup[str(row["source_record"])].append(float(row["diff_minus_det_utility"]))
    for index in range(hierarchical.size):
        drawn_sources = rng.choice(source_ids, len(source_ids), replace=True)
        hierarchical[index] = np.mean([rng.choice(lookup[source]) for source in drawn_sources])

    checkpoint_rows = []
    for checkpoint in sorted((OLD_ROOT / "checkpoints").glob("**/models.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        summary_path = checkpoint.parent / "result_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        checkpoint_rows.append({"checkpoint": str(checkpoint), "contains_diffusion_ema": "diffusion_ema" in payload, "contains_raw_diffusion": "diffusion" in payload, "best_step_diffusion": summary.get("best_steps", {}).get("diffusion") if isinstance(summary.get("best_steps"), Mapping) else None, "validation_named_for_EMA": "ema" in json.dumps(summary).lower()})
    _write_csv(output / "ema_checkpoint_selection_audit.csv", checkpoint_rows)
    summary = {
        "status": "completed_current_result_reaudit_CPU",
        **_implementation(),
        "uncertainty_status": "invalid_due_to_window_axis_bug_and_noncomparable_uncertainty_definition",
        "corrected_axis_result_scope": "legacy descriptive only; diffusion score remains latent-derived while deterministic is seed variation",
        "full_final_EEG_uncertainty_status": "requires_checkpoint_replay_because_old_arrays_lack_training_seed_x_posterior_sample_axis",
        "source_level_uncertainty": {
            key: _bootstrap_sources(rows, key, 20260804 + index, 20000)
            for index, key in enumerate(("diff_centered_pearson", "diff_centered_spearman", "det_centered_pearson", "det_centered_spearman"))
        },
        "hierarchical_diff_minus_det": {"mean": float(np.mean([row["diff_minus_det_utility"] for row in by_unit_seed])), "ci95": [float(np.quantile(hierarchical, .025)), float(np.quantile(hierarchical, .975))], "sources": len(source_ids), "seeds": len(SEEDS)},
        "free_replay_requirements": ["DET-POP", "true_end_to_end_DIFF-MATCH-K1", "final_EEG_posterior_samples", "EMA_validation_replay"],
        "old_results_modified": False,
        "unused_metric_row_count": len(metric_rows),
    }
    replay_root = output / "final_EEG_checkpoint_replay"
    if (replay_root / "result_summary.json").is_file():
        comparable_rows = []
        method_rows = _read_csvs([replay_root / "metrics.csv"])
        for record in [f"sim{value:02d}" for value in range(37, 55) if value not in (44, 45)]:
            seed_arrays = []
            for seed in SEEDS:
                path = replay_root / f"seed_{seed}" / f"{record}.npz"
                if path.is_file(): seed_arrays.append(np.load(path))
            if len(seed_arrays) != 3: continue
            samples = np.stack([np.asarray(value["final_EEG_samples"], dtype=np.float64) for value in seed_arrays])
            clean = np.asarray(seed_arrays[0]["clean"], dtype=np.float64)
            det_stack = np.stack([np.asarray(value["DET_MATCH"], dtype=np.float64) for value in seed_arrays])
            seed_means = samples.mean(axis=1)
            final_mean = seed_means.mean(axis=0)
            posterior_variance = samples.var(axis=1).mean(axis=0)
            training_seed_variance = seed_means.var(axis=0)
            total_uncertainty = np.sqrt(posterior_variance + training_seed_variance)
            det_uncertainty = det_stack.std(axis=0)
            diff_error = np.sqrt(np.mean(np.square(final_mean - clean), axis=(1, 2)))
            det_error = np.sqrt(np.mean(np.square(det_stack.mean(0) - clean), axis=(1, 2)))
            diff_score = total_uncertainty.mean(axis=(1, 2)); det_score = det_uncertainty.mean(axis=(1, 2))
            comparable_rows.append({
                "source_record": record, "window_count": clean.shape[0],
                "diff_final_EEG_centered_pearson": _correlation(diff_score, diff_error),
                "diff_final_EEG_centered_spearman": _correlation(diff_score, diff_error, rank=True),
                "det_final_EEG_centered_pearson": _correlation(det_score, det_error),
                "det_final_EEG_centered_spearman": _correlation(det_score, det_error, rank=True),
                "posterior_variance_mean": float(posterior_variance.mean()),
                "training_seed_variance_mean": float(training_seed_variance.mean()),
                **{f"diff_{key}": value for key, value in _risk_auc(diff_error, diff_score).items()},
                **{f"det_{key}": value for key, value in _risk_auc(det_error, det_score).items()},
            })
            for value in seed_arrays: value.close()
        if len(comparable_rows) == 16:
            _write_csv(output / "comparable_final_EEG_uncertainty_by_source.csv", comparable_rows)
            comparable_summary = {
                "status": "recomputed_comparable_final_EEG_uncertainty",
                "array_axes": ["training_seed", "posterior_sample", "window", "channel", "time"],
                "source_level": {
                    key: _bootstrap_sources(comparable_rows, key, 20260840 + index, 20000)
                    for index, key in enumerate(("diff_final_EEG_centered_pearson", "diff_final_EEG_centered_spearman", "det_final_EEG_centered_pearson", "det_final_EEG_centered_spearman", "diff_normalized_risk_coverage_auc", "det_normalized_risk_coverage_auc", "diff_excess_risk_auc", "det_excess_risk_auc"))
                },
                "DET_POP_mean_RRMSE": float(np.mean([float(row["clean_RRMSE"]) for row in method_rows if row["method"] == "DET-POP"])),
                "true_K1_mean_RRMSE": float(np.mean([float(row["clean_RRMSE"]) for row in method_rows if row["method"] == "DIFF-MATCH-K1-TRUE-E2E"])),
                "K8_mean_RRMSE": float(np.mean([float(row["clean_RRMSE"]) for row in method_rows if row["method"] == "DIFF-MATCH-K8"])),
            }
            _atomic_json(output / "corrected_uncertainty.json", comparable_summary)
            summary["full_final_EEG_uncertainty_status"] = "recomputed_comparable_secondary_exploratory_evidence"
            summary["comparable_final_EEG_uncertainty"] = comparable_summary
    _atomic_json(output / "result_summary.json", summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


@torch.no_grad()
def _j0_replay(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Replay old checkpoints without training to recover comparable EEG uncertainty."""

    base, root = _load(config)
    prior_config = yaml.safe_load((CODE_ROOT / str(config["prior_artifact_subspace_config"])).read_text(encoding="utf-8"))
    if not isinstance(prior_config, Mapping):
        raise ValueError("prior artifact-subspace config is invalid")
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _prepared
    from eeg_cgdr.experiments.subject_artifact_subspace_diffusion import (
        _klados_eval_records,
        _load_models,
        _runtime_basis,
    )

    prepared = _prepared(base, "klados", 0)
    device = torch.device("cuda", 0)
    replay_root = root / "current_result_reaudit/final_EEG_checkpoint_replay"
    rows: list[dict[str, Any]] = []
    evaluation_records = _klados_eval_records(base)
    for seed_index, seed in enumerate(SEEDS):
        route = {"dataset": "klados", "fold_index": 0, "seed": seed}
        anchor, deterministic, diffusion, _, tau, checkpoint = _load_models(prior_config, prepared, route, device)
        del anchor
        population_basis, _, population_rank = aligned_artifact_basis(prepared.population_context.full_transfer)
        for key, mechanism, matching, _ in evaluation_records:
            match_basis, match_rank = _runtime_basis(matching, population_basis)
            y_all = np.asarray(mechanism.observed_windows, dtype=np.float32)
            valid_all = np.asarray(mechanism.valid_time_weight, dtype=bool)
            sample_outputs: list[np.ndarray] = []
            det_pop_parts: list[np.ndarray] = []
            det_match_parts: list[np.ndarray] = []
            k1_parts: list[np.ndarray] = []
            for batch_index, start in enumerate(range(0, y_all.shape[0], 32)):
                stop = min(y_all.shape[0], start + 32)
                y = torch.as_tensor(y_all[start:stop], device=device)
                mask = torch.as_tensor(valid_all[start:stop], device=device)
                count = y.shape[0]
                pop_basis = torch.as_tensor(population_basis, device=device)[None].expand(count, -1, -1)
                sub_basis = torch.as_tensor(match_basis, device=device)[None].expand(count, -1, -1)
                pop_rank = torch.as_tensor(population_rank, device=device)[None].expand(count, -1)
                sub_rank = torch.as_tensor(match_rank, device=device)[None].expand(count, -1)
                rho_value = float(matching.rho)
                rho = torch.full((count,), rho_value, device=device)
                common = {"observed": y, "reliability": rho, "valid_time_mask": mask}
                pop_condition = {**common, "basis": pop_basis, "rank_mask": pop_rank}
                match_condition = {**common, "basis": sub_basis, "rank_mask": sub_rank}
                det_pop_u = deterministic(**pop_condition)
                det_match_u = deterministic(**match_condition)
                _, det_pop_delta = reconstruct_from_subspace(y, pop_basis, det_pop_u, tau, pop_rank, mask)
                _, det_match_delta = reconstruct_from_subspace(y, sub_basis, det_match_u, tau, sub_rank, mask)
                det_pop_parts.append((y - det_pop_delta).cpu().numpy())
                det_match_parts.append((y - ((1.0 - rho_value) * det_pop_delta + rho_value * det_match_delta)).cpu().numpy())
                base_seeds = participant_sample_seeds(key, seed)
                common_seeds = tuple(value + batch_index * 104729 for value in base_seeds)
                batch_samples = []
                for sample_seed in common_seeds:
                    pop_u, _, _, _ = diffusion.sample(sample_seeds=(sample_seed,), **pop_condition)
                    _, pop_delta = reconstruct_from_subspace(y, pop_basis, pop_u, tau, pop_rank, mask)
                    if rho_value == 0.0:
                        final = y - pop_delta
                    else:
                        match_u, _, _, _ = diffusion.sample(sample_seeds=(sample_seed,), **match_condition)
                        _, match_delta = reconstruct_from_subspace(y, sub_basis, match_u, tau, sub_rank, mask)
                        final = y - ((1.0 - rho_value) * pop_delta + rho_value * match_delta)
                    batch_samples.append(final.cpu().numpy())
                stacked = np.stack(batch_samples, axis=0)
                sample_outputs.append(stacked)
                k1_parts.append(stacked[0])
            # concatenate along windows while retaining K first
            samples = np.concatenate(sample_outputs, axis=1).astype(np.float32)
            det_pop = np.concatenate(det_pop_parts).astype(np.float32)
            det_match = np.concatenate(det_match_parts).astype(np.float32)
            true_k1 = np.concatenate(k1_parts).astype(np.float32)
            clean = np.asarray(mechanism.clean_windows, dtype=np.float32)
            destination = replay_root / f"seed_{seed}" / f"{key}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, final_EEG_samples=samples, final_EEG_mean=samples.mean(0), final_EEG_sd=samples.std(0), DET_POP=det_pop, DET_MATCH=det_match, DIFF_MATCH_K1=true_k1, clean=clean, observed=y_all)
            def rrmse(value: np.ndarray) -> float:
                return float(np.linalg.norm(value - clean) / max(np.linalg.norm(clean), 1.0e-12))
            rows.extend([
                {"training_seed": seed, "source_record": key, "method": "DET-POP", "clean_RRMSE": rrmse(det_pop), "checkpoint": str(checkpoint)},
                {"training_seed": seed, "source_record": key, "method": "DET-MATCH", "clean_RRMSE": rrmse(det_match), "checkpoint": str(checkpoint)},
                {"training_seed": seed, "source_record": key, "method": "DIFF-MATCH-K1-TRUE-E2E", "clean_RRMSE": rrmse(true_k1), "checkpoint": str(checkpoint)},
                {"training_seed": seed, "source_record": key, "method": "DIFF-MATCH-K8", "clean_RRMSE": rrmse(samples.mean(0)), "checkpoint": str(checkpoint)},
            ])
    _write_csv(replay_root / "metrics.csv", rows)
    summary = {
        "status": "completed_old_checkpoint_final_EEG_replay",
        **_implementation(),
        "training_seeds": list(SEEDS), "posterior_samples": 8,
        "source_records": len(evaluation_records),
        "saved_array_axes": ["training_seed(file hierarchy)", "posterior_sample", "window", "channel", "time"],
        "population_subject_covariance_retained": True,
        "true_K1_uses_same_sample_for_population_and_matching": True,
        "training_performed": False,
    }
    _atomic_json(replay_root / "result_summary.json", summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _ridge(eeg: np.ndarray, eog_standardized: np.ndarray, ridge: float) -> np.ndarray:
    y = np.asarray(eeg, dtype=np.float64)
    e = np.asarray(eog_standardized, dtype=np.float64)
    y = y - y.mean(axis=1, keepdims=True)
    gram = e @ e.T + float(ridge) * np.eye(e.shape[0], dtype=np.float64)
    return np.linalg.solve(gram, (y @ e.T).T).T


def _standardize_full(eog: np.ndarray) -> np.ndarray:
    value = np.asarray(eog, dtype=np.float64)
    scale = value.std(axis=1, keepdims=True)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("constant EOG support coordinate")
    return (value - value.mean(axis=1, keepdims=True)) / scale


def _prediction_error(eeg: np.ndarray, eog: np.ndarray, transfer: np.ndarray) -> float:
    centered = np.asarray(eeg, dtype=np.float64) - np.asarray(eeg, dtype=np.float64).mean(axis=1, keepdims=True)
    residual = centered - np.asarray(transfer, dtype=np.float64) @ np.asarray(eog, dtype=np.float64)
    return float(np.linalg.norm(residual) / max(np.linalg.norm(centered), 1.0e-12))


def _projector(transfer: np.ndarray, rank: int = 2) -> np.ndarray:
    basis, _, mask = aligned_artifact_basis(transfer, rank=rank)
    active = basis[:, mask]
    return active @ active.T


def _principal_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    l, _, _ = np.linalg.svd(left, full_matrices=False)
    r, _, _ = np.linalg.svd(right, full_matrices=False)
    values = np.linalg.svd(l[:, :2].T @ r[:, :2], compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(values, -1.0, 1.0))).max())


def _fir_design(eog: np.ndarray, lags: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return a lag-major design: ``[lag0:E..., lag1:E..., ...]``."""

    e = np.asarray(eog, dtype=np.float64)
    lower = max(0, max(lags)); upper = e.shape[1] + min(0, min(lags))
    if upper - lower < 8:
        raise ValueError("FIR support is too short")
    fields = [e[:, lower - lag:upper - lag] for lag in lags]
    return np.concatenate(fields, axis=0), np.arange(lower, upper)


def _fir_lags(audit: Mapping[str, Any], sampling_rate_hz: float) -> tuple[int, ...]:
    if "fir_lags_milliseconds" in audit:
        milliseconds = tuple(float(value) for value in audit["fir_lags_milliseconds"])
        samples = tuple(int(round(value * float(sampling_rate_hz) / 1000.0)) for value in milliseconds)
    else:
        samples = tuple(int(value) for value in audit["fir_lags_samples"])
    if len(samples) < 1 or len(set(samples)) != len(samples) or 0 not in samples:
        raise ValueError("cell-specific FIR lags must be unique and contain zero")
    return samples


def _fit_fir(eeg: np.ndarray, eog: np.ndarray, lags: Sequence[int], ridge: float) -> np.ndarray:
    design, index = _fir_design(eog, lags)
    flat = _ridge(np.asarray(eeg)[:, index], design, ridge)
    return fir_coefficients_from_lag_major(
        flat,
        eeg_channels=np.asarray(eeg).shape[0],
        eog_channels=np.asarray(eog).shape[0],
        lag_count=len(lags),
    )


def _fir_prediction_error(eeg: np.ndarray, eog: np.ndarray, transfer: np.ndarray, lags: Sequence[int]) -> float:
    design, index = _fir_design(eog, lags)
    return _prediction_error(
        np.asarray(eeg)[:, index],
        design,
        fir_coefficients_to_lag_major(transfer),
    )


def _blocked_fir_crossfit(
    eeg: np.ndarray,
    eog: np.ndarray,
    population_fir: np.ndarray,
    lags: Sequence[int],
    ridge: float,
    alphas: Sequence[float],
) -> tuple[np.ndarray, float, dict[float, float], float]:
    """Select FIR shrinkage by A->B/B->A held-out support prediction."""

    midpoint = np.asarray(eog).shape[1] // 2
    halves = ((slice(0, midpoint), slice(midpoint, None)), (slice(midpoint, None), slice(0, midpoint)))
    half_transfers = [_fit_fir(eeg[:, fit], eog[:, fit], lags, ridge) for fit, _ in halves]
    scores: dict[float, float] = {}
    for alpha in alphas:
        values = []
        for (_, score), fitted in zip(halves, half_transfers):
            effective = population_fir + float(alpha) * (fitted - population_fir)
            values.append(_fir_prediction_error(eeg[:, score], eog[:, score], effective, lags))
        scores[float(alpha)] = float(np.mean(values))
    selected = min((float(value) for value in alphas), key=lambda value: (scores[value], value))
    full = _fit_fir(eeg, eog, lags, ridge)
    stability = float(np.linalg.norm(half_transfers[0] - half_transfers[1]))
    return full, selected, scores, stability


def _fit_state_transfer(
    eeg: np.ndarray,
    eog: np.ndarray,
    ridge: float,
    active_quantile: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    activity = np.sqrt(np.mean(np.square(eog), axis=0))
    threshold = float(np.quantile(activity, active_quantile))
    active = activity >= threshold
    if active.sum() < eog.shape[0] + 2 or (~active).sum() < eog.shape[0] + 2:
        raise ValueError("blocked state split lacks active or quiet excitation")
    return _ridge(eeg[:, active], eog[:, active], ridge), _ridge(eeg[:, ~active], eog[:, ~active], ridge), threshold


def _state_prediction_error(
    eeg: np.ndarray,
    eog: np.ndarray,
    active_transfer: np.ndarray,
    quiet_transfer: np.ndarray,
    threshold: float,
) -> float:
    activity = np.sqrt(np.mean(np.square(eog), axis=0))
    active = activity >= float(threshold)
    prediction = quiet_transfer @ eog
    prediction[:, active] = active_transfer @ eog[:, active]
    centered = eeg - eeg.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(centered - prediction) / max(np.linalg.norm(centered), 1.0e-12))


def _blocked_state_crossfit(
    eeg: np.ndarray,
    eog: np.ndarray,
    population_active: np.ndarray,
    population_quiet: np.ndarray,
    ridge: float,
    active_quantile: float,
    alphas: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, float, dict[float, float], float]:
    midpoint = eog.shape[1] // 2
    halves = ((slice(0, midpoint), slice(midpoint, None)), (slice(midpoint, None), slice(0, midpoint)))
    fitted = [_fit_state_transfer(eeg[:, fit], eog[:, fit], ridge, active_quantile) for fit, _ in halves]
    scores: dict[float, float] = {}
    for alpha in alphas:
        values = []
        for (_, score), (active, quiet, threshold) in zip(halves, fitted):
            active_effective = population_active + float(alpha) * (active - population_active)
            quiet_effective = population_quiet + float(alpha) * (quiet - population_quiet)
            values.append(_state_prediction_error(eeg[:, score], eog[:, score], active_effective, quiet_effective, threshold))
        scores[float(alpha)] = float(np.mean(values))
    selected = min((float(value) for value in alphas), key=lambda value: (scores[value], value))
    full_active, full_quiet, _ = _fit_state_transfer(eeg, eog, ridge, active_quantile)
    stability = float(
        np.linalg.norm(fitted[0][0] - fitted[1][0])
        + np.linalg.norm(fitted[0][1] - fitted[1][1])
    )
    return full_active, full_quiet, selected, scores, stability


def _j1_operator_audit(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """Audit all SGE stems with support-only static/FIR/state representations."""

    base, root = _load(config)
    frozen = _load_frozen_config(base)
    audit = _mapping(config, "operator_audit")
    ridge = float(audit["ridge_lambda"])
    alphas = tuple(float(value) for value in audit["alpha_candidates"])
    operator_rows: list[dict[str, Any]] = []
    crossfit_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unified_index in range(25):
        partition, local_index = _unified_fold_route(frozen, unified_index)
        prepared = _prepare_fold(frozen, partition, local_index)
        if prepared.population.outcome.transfer is None:
            raise RuntimeError("population transfer unexpectedly missing")
        population = np.asarray(prepared.population.outcome.transfer.transfer_matrix, dtype=np.float64)
        for key in prepared.fold.heldout_recording_keys:
            if key in seen:
                continue
            seen.add(key)
            loaded = _loaded_record(prepared, key)
            count, seconds = _calibration_samples(base, loaded.sampling_rate_hz, loaded.support.eeg.shape[1])
            lags = _fir_lags(audit, loaded.sampling_rate_hz)
            eeg = prepared.normalizer.transform(loaded.support.eeg)[:, :count]
            eog = _standardize_full(loaded.support.external_eog[:, :count])
            midpoint = count // 2
            halves = ((slice(0, midpoint), slice(midpoint, count)), (slice(midpoint, count), slice(0, midpoint)))
            full = _ridge(eeg, eog, ridge)
            singular = np.linalg.svd(full, compute_uv=False)
            p_full = _projector(full)
            p_pop = _projector(population)
            half_transfers = [_ridge(eeg[:, half], eog[:, half], ridge) for half, _ in halves]
            within_distance = float(np.linalg.norm(half_transfers[0] - half_transfers[1]))
            alpha_scores: dict[float, list[float]] = defaultdict(list)
            for (fit_slice, score_slice), half_transfer in zip(halves, half_transfers):
                for alpha in alphas:
                    shrunken = population + alpha * (half_transfer - population)
                    alpha_scores[alpha].append(_prediction_error(eeg[:, score_slice], eog[:, score_slice], shrunken))
            alpha = min(alphas, key=lambda value: (float(np.mean(alpha_scores[value])), value))
            selected = population + alpha * (full - population)
            # Build wrong controls from support information, not from the
            # lexicographically first training stems.  When possible one donor
            # is another held-out stem (its support only); the remaining
            # training donors span the near/far operator-distance range.
            candidate_rows = []
            candidate_keys = [
                *((str(value), "training_seen") for value in prepared.fold.training_recording_keys),
                *((str(value), "heldout_unseen") for value in prepared.fold.heldout_recording_keys if str(value) != key),
            ]
            for donor, donor_role in candidate_keys:
                donor_loaded = prepared.training[donor] if donor_role == "training_seen" else _loaded_record(prepared, donor)
                donor_count, _ = _calibration_samples(base, donor_loaded.sampling_rate_hz, donor_loaded.support.eeg.shape[1])
                donor_eeg = prepared.normalizer.transform(donor_loaded.support.eeg)[:, :donor_count]
                donor_eog = _standardize_full(donor_loaded.support.external_eog[:, :donor_count])
                donor_transfer = _ridge(donor_eeg, donor_eog, ridge)
                donor_midpoint = donor_count // 2
                donor_half_a = _ridge(donor_eeg[:, :donor_midpoint], donor_eog[:, :donor_midpoint], ridge)
                donor_half_b = _ridge(donor_eeg[:, donor_midpoint:], donor_eog[:, donor_midpoint:], ridge)
                donor_fir_e, donor_fir_index = _fir_design(donor_eog, lags)
                donor_fir = _fit_fir(donor_eeg, donor_eog, lags, ridge)
                donor_activity = np.sqrt(np.mean(np.square(donor_eog), axis=0))
                donor_threshold = float(np.quantile(donor_activity, float(audit["state_active_quantile"])))
                donor_active = donor_activity >= donor_threshold
                candidate_rows.append({
                    "key": donor, "role": donor_role, "full": donor_transfer,
                    "stability": float(np.linalg.norm(donor_half_a - donor_half_b)),
                    "severity": float(np.quantile(donor_activity, .9)),
                    "distance": float(np.linalg.norm(donor_transfer - full)),
                    "target_error": _prediction_error(eeg[:, midpoint:], eog[:, midpoint:], donor_transfer),
                    "fir": donor_fir, "fir_e": donor_fir_e,
                    "fir_eeg": donor_eeg[:, donor_fir_index],
                    "active_eeg": donor_eeg[:, donor_active], "active_eog": donor_eog[:, donor_active],
                    "quiet_eeg": donor_eeg[:, ~donor_active], "quiet_eog": donor_eog[:, ~donor_active],
                    "active_C": _ridge(donor_eeg[:, donor_active], donor_eog[:, donor_active], ridge),
                    "quiet_C": _ridge(donor_eeg[:, ~donor_active], donor_eog[:, ~donor_active], ridge),
                })
            training_candidates = sorted(
                (value for value in candidate_rows if value["role"] == "training_seen"),
                key=lambda value: (float(value["distance"]), str(value["key"])),
            )
            unseen_candidates = sorted(
                (value for value in candidate_rows if value["role"] == "heldout_unseen"),
                key=lambda value: (float(value["stability"]), str(value["key"])),
            )
            donor_rows = unseen_candidates[:1]
            needed = 3 - len(donor_rows)
            if len(training_candidates) < needed:
                raise RuntimeError("same-cell training donors are insufficient")
            positions = np.linspace(0, len(training_candidates) - 1, needed + 2)[1:-1]
            for position in positions:
                candidate = training_candidates[int(round(float(position)))]
                if candidate not in donor_rows:
                    donor_rows.append(candidate)
            for candidate in training_candidates:
                if len(donor_rows) >= 3:
                    break
                if candidate not in donor_rows:
                    donor_rows.append(candidate)
            donor_rows = donor_rows[:3]
            population_sources = [value for value in candidate_rows if value["role"] == "training_seen"]
            population_fir = _ridge(
                np.concatenate([np.asarray(value["fir_eeg"]) for value in population_sources], axis=1),
                np.concatenate([np.asarray(value["fir_e"]) for value in population_sources], axis=1),
                ridge,
            )
            population_fir = fir_coefficients_from_lag_major(
                population_fir,
                eeg_channels=eeg.shape[0],
                eog_channels=eog.shape[0],
                lag_count=len(lags),
            )
            population_active_transfer = _ridge(
                np.concatenate([np.asarray(value["active_eeg"]) for value in population_sources], axis=1),
                np.concatenate([np.asarray(value["active_eog"]) for value in population_sources], axis=1),
                ridge,
            )
            population_quiet_transfer = _ridge(
                np.concatenate([np.asarray(value["quiet_eeg"]) for value in population_sources], axis=1),
                np.concatenate([np.asarray(value["quiet_eog"]) for value in population_sources], axis=1),
                ridge,
            )
            fir, fir_alpha, fir_alpha_scores, fir_stability = _blocked_fir_crossfit(
                eeg, eog, population_fir, lags, ridge, alphas,
            )
            active_transfer, quiet_transfer, state_alpha, state_alpha_scores, state_stability = _blocked_state_crossfit(
                eeg,
                eog,
                population_active_transfer,
                population_quiet_transfer,
                ridge,
                float(audit["state_active_quantile"]),
                alphas,
            )
            cosine = 1.0 - float(np.sum(full * population) / max(np.linalg.norm(full) * np.linalg.norm(population), 1.0e-12))
            operator_rows.append({
                "dataset": "sgeyesub", "recording_key": key, "unified_fold": unified_index,
                "study": prepared.fold.study, "layout_id": prepared.records[key].layout_id,
                "sampling_rate_hz": loaded.sampling_rate_hz, "calibration_seconds": seconds,
                "calibration_samples": count, "effective_rank": int(np.linalg.matrix_rank(full)),
                "singular_1": float(singular[0]), "singular_2": float(singular[1]) if singular.size > 1 else 0.0,
                "projector_principal_angle_deg": _principal_angle_degrees(full, population),
                "projector_frobenius_distance": float(np.linalg.norm(p_full - p_pop)),
                "full_C_frobenius_distance": float(np.linalg.norm(full - population)),
                "full_C_cosine_distance": cosine, "transfer_column_scale_mean": float(np.linalg.norm(full, axis=0).mean()),
                "within_subject_half_distance": within_distance, "selected_alpha": alpha,
                "split_half_projector_distance": float(np.linalg.norm(_projector(half_transfers[0]) - _projector(half_transfers[1]))),
                "state_active_quiet_C_distance": float(np.linalg.norm(active_transfer - quiet_transfer)),
                "FIR_population_residual_norm": float(np.linalg.norm(fir - population_fir)),
                "FIR_selected_alpha": fir_alpha,
                "FIR_split_half_stability": fir_stability,
                "state_selected_alpha": state_alpha,
                "state_split_half_stability": state_stability,
                "rho_legacy": float(prepared.matching[key].outcome.diagnostics.get("rho", float("nan"))) if hasattr(prepared.matching[key].outcome, "diagnostics") else float("nan"),
            })
            crossfit_rows.append({
                "recording_key": key, "study": prepared.fold.study, "layout_id": prepared.records[key].layout_id,
                "matching_full_C_error": float(np.mean([_prediction_error(eeg[:, score], eog[:, score], fit) for (_, score), fit in zip(halves, half_transfers)])),
                "population_error": float(np.mean([_prediction_error(eeg[:, score], eog[:, score], population) for _, score in halves])),
                "shrunken_error": float(np.mean(alpha_scores[alpha])), "selected_alpha": alpha,
                "FIR_support_error": fir_alpha_scores[fir_alpha],
                "FIR_selected_alpha": fir_alpha,
                "state_specific_error": state_alpha_scores[state_alpha],
                "state_selected_alpha": state_alpha,
                **{f"wrong_{index+1}_key": donor["key"] for index, donor in enumerate(donor_rows)},
                **{f"wrong_{index+1}_role": donor["role"] for index, donor in enumerate(donor_rows)},
                **{f"wrong_{index+1}_error": donor["target_error"] for index, donor in enumerate(donor_rows)},
                **{f"wrong_{index+1}_stability": donor["stability"] for index, donor in enumerate(donor_rows)},
                **{f"wrong_{index+1}_severity": donor["severity"] for index, donor in enumerate(donor_rows)},
                **{f"wrong_{index+1}_operator_distance": donor["distance"] for index, donor in enumerate(donor_rows)},
                **{f"alpha_{value:g}_error": float(np.mean(alpha_scores[value])) for value in alphas},
                **{f"FIR_alpha_{value:g}_error": fir_alpha_scores[float(value)] for value in alphas},
                **{f"state_alpha_{value:g}_error": state_alpha_scores[float(value)] for value in alphas},
            })
            cache = root / "operator_cache/sgeyesub" / f"{key.replace('/', '__')}.npz"
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache, full_C=full, population_C=population, selected_C=selected,
                FIR=fir,
                population_FIR=population_fir,
                selected_FIR=population_fir + fir_alpha * (fir - population_fir),
                FIR_lags=np.asarray(lags), active_C=active_transfer,
                quiet_C=quiet_transfer, population_active_C=population_active_transfer,
                population_quiet_C=population_quiet_transfer,
                selected_alpha=np.asarray(alpha),
                selected_fir_alpha=np.asarray(fir_alpha),
                selected_state_alpha=np.asarray(state_alpha),
                wrong_C=np.stack([np.asarray(value["full"]) for value in donor_rows]),
                wrong_FIR=np.stack([np.asarray(value["fir"]) for value in donor_rows]),
                wrong_active_C=np.stack([np.asarray(value["active_C"]) for value in donor_rows]),
                wrong_quiet_C=np.stack([np.asarray(value["quiet_C"]) for value in donor_rows]),
                wrong_keys=np.asarray([value["key"] for value in donor_rows]),
                wrong_roles=np.asarray([value["role"] for value in donor_rows]),
            )
    if len(seen) != 58:
        raise RuntimeError(f"SGE operator audit expected 58 unique stems, found {len(seen)}")
    # The 16 frozen Klados evaluation records are source-record mechanism data,
    # never participant-level evidence.  Their query EOG is not used here.
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _prepared
    from eeg_cgdr.experiments.subject_artifact_subspace_diffusion import _klados_eval_records
    klados_prepared = _prepared(base, "klados", 0)
    lags = _fir_lags(audit, klados_prepared.fold.sampling_rate_hz)
    klados_population = np.asarray(klados_prepared.population_context.full_transfer, dtype=np.float64)
    klados_records = _klados_eval_records(base)
    training_target, training_coordinate_mean, training_coordinate_std = _canonical_training(klados_prepared)
    training_eog = (
        training_target * training_coordinate_std[None, :, None]
        + training_coordinate_mean[None, :, None]
    ).astype(np.float64)
    training_eeg = np.asarray(klados_prepared.training.observed, dtype=np.float64)
    training_valid = np.asarray(klados_prepared.training.valid_time_mask, dtype=bool)
    training_fir_e, training_fir_y = [], []
    active_eeg, active_eog, quiet_eeg, quiet_eog = [], [], [], []
    for window in range(training_eeg.shape[0]):
        fir_design, valid_fir_index = _fir_design(training_eog[window], lags)
        valid_fir = training_valid[window, valid_fir_index]
        training_fir_e.append(fir_design[:, valid_fir])
        training_fir_y.append(training_eeg[window][:, valid_fir_index[valid_fir]])
        activity = np.sqrt(np.mean(np.square(training_eog[window]), axis=0))
        threshold = float(np.quantile(activity[training_valid[window]], float(audit["state_active_quantile"])))
        active = training_valid[window] & (activity >= threshold)
        quiet = training_valid[window] & ~active
        active_eeg.append(training_eeg[window][:, active]); active_eog.append(training_eog[window][:, active])
        quiet_eeg.append(training_eeg[window][:, quiet]); quiet_eog.append(training_eog[window][:, quiet])
    klados_population_fir_flat = _ridge(
        np.concatenate(training_fir_y, axis=1),
        np.concatenate(training_fir_e, axis=1), ridge,
    )
    klados_population_fir = fir_coefficients_from_lag_major(
        klados_population_fir_flat,
        eeg_channels=training_eeg.shape[1],
        eog_channels=training_eog.shape[1],
        lag_count=len(lags),
    )
    klados_population_active = _ridge(np.concatenate(active_eeg, axis=1), np.concatenate(active_eog, axis=1), ridge)
    klados_population_quiet = _ridge(np.concatenate(quiet_eeg, axis=1), np.concatenate(quiet_eog, axis=1), ridge)
    for record_index, (key, mechanism, matching, _) in enumerate(klados_records):
        eeg = np.asarray(mechanism.calibration.eeg, dtype=np.float64)
        eog = np.asarray(mechanism.calibration.eog, dtype=np.float64)
        midpoint = eog.shape[1] // 2
        half_a = _ridge(eeg[:, :midpoint], eog[:, :midpoint], ridge)
        half_b = _ridge(eeg[:, midpoint:], eog[:, midpoint:], ridge)
        full = np.asarray(matching.full_transfer, dtype=np.float64)
        scores = {}
        for alpha in alphas:
            scores[alpha] = .5 * (
                _prediction_error(eeg[:, midpoint:], eog[:, midpoint:], klados_population + alpha * (half_a - klados_population))
                + _prediction_error(eeg[:, :midpoint], eog[:, :midpoint], klados_population + alpha * (half_b - klados_population))
            )
        alpha = min(alphas, key=lambda value: (scores[value], value))
        fir, fir_alpha, fir_alpha_scores, fir_stability = _blocked_fir_crossfit(
            eeg, eog, klados_population_fir, lags, ridge, alphas,
        )
        active_C, quiet_C, state_alpha, state_alpha_scores, state_stability = _blocked_state_crossfit(
            eeg,
            eog,
            klados_population_active,
            klados_population_quiet,
            ridge,
            float(audit["state_active_quantile"]),
            alphas,
        )
        singular = np.linalg.svd(full, compute_uv=False)
        wrong_contexts = [klados_records[(record_index + offset) % len(klados_records)][2] for offset in (1, 2, 3)]
        wrong_errors = [
            _prediction_error(eeg, eog, np.asarray(context.full_transfer, dtype=np.float64))
            for context in wrong_contexts
        ]
        operator_rows.append({
            "dataset": "klados", "recording_key": key, "study": "klados_v4_source_records",
            "layout_id": klados_prepared.fold.layout_id, "sampling_rate_hz": klados_prepared.fold.sampling_rate_hz,
            "calibration_seconds": 10.0, "calibration_samples": eog.shape[1],
            "effective_rank": int(np.linalg.matrix_rank(full)), "singular_1": float(singular[0]),
            "singular_2": float(singular[1]), "projector_principal_angle_deg": _principal_angle_degrees(full, klados_population),
            "projector_frobenius_distance": float(np.linalg.norm(_projector(full) - _projector(klados_population))),
            "full_C_frobenius_distance": float(np.linalg.norm(full - klados_population)),
            "within_subject_half_distance": float(np.linalg.norm(half_a - half_b)),
            "selected_alpha": alpha, "split_half_projector_distance": float(np.linalg.norm(_projector(half_a) - _projector(half_b))),
            "state_active_quiet_C_distance": float(np.linalg.norm(active_C - quiet_C)),
            "FIR_population_residual_norm": float(np.linalg.norm(fir[..., lags.index(0)] - klados_population)),
            "FIR_selected_alpha": fir_alpha, "FIR_split_half_stability": fir_stability,
            "state_selected_alpha": state_alpha, "state_split_half_stability": state_stability,
        })
        crossfit_rows.append({
            "recording_key": key, "study": "klados_v4_source_records", "layout_id": klados_prepared.fold.layout_id,
            "matching_full_C_error": .5 * (_prediction_error(eeg[:, midpoint:], eog[:, midpoint:], half_a) + _prediction_error(eeg[:, :midpoint], eog[:, :midpoint], half_b)),
            "population_error": .5 * (_prediction_error(eeg[:, midpoint:], eog[:, midpoint:], klados_population) + _prediction_error(eeg[:, :midpoint], eog[:, :midpoint], klados_population)),
            "shrunken_error": scores[alpha], "selected_alpha": alpha,
            "FIR_support_error": fir_alpha_scores[fir_alpha], "FIR_selected_alpha": fir_alpha,
            "state_specific_error": state_alpha_scores[state_alpha], "state_selected_alpha": state_alpha,
            **{f"wrong_{index + 1}_key": context.context_id for index, context in enumerate(wrong_contexts)},
            **{f"wrong_{index + 1}_error": error for index, error in enumerate(wrong_errors)},
            **{f"FIR_alpha_{value:g}_error": fir_alpha_scores[float(value)] for value in alphas},
            **{f"state_alpha_{value:g}_error": state_alpha_scores[float(value)] for value in alphas},
        })
        cache = root / "operator_cache/klados" / f"{key}.npz"; cache.parent.mkdir(parents=True, exist_ok=True)
        wrong_fir, wrong_active, wrong_quiet = [], [], []
        for offset in (1, 2, 3):
            _, wrong_mechanism, _, _ = klados_records[(record_index + offset) % len(klados_records)]
            wrong_eeg = np.asarray(wrong_mechanism.calibration.eeg, dtype=np.float64)
            wrong_eog = np.asarray(wrong_mechanism.calibration.eog, dtype=np.float64)
            wrong_design, wrong_index = _fir_design(wrong_eog, lags)
            wrong_fir.append(_fit_fir(wrong_eeg, wrong_eog, lags, ridge))
            wrong_activity = np.sqrt(np.mean(np.square(wrong_eog), axis=0))
            wrong_threshold = np.quantile(wrong_activity, float(audit["state_active_quantile"]))
            wrong_active.append(_ridge(wrong_eeg[:, wrong_activity >= wrong_threshold], wrong_eog[:, wrong_activity >= wrong_threshold], ridge))
            wrong_quiet.append(_ridge(wrong_eeg[:, wrong_activity < wrong_threshold], wrong_eog[:, wrong_activity < wrong_threshold], ridge))
        np.savez_compressed(
            cache, full_C=full, population_C=klados_population,
            selected_C=klados_population + alpha * (full - klados_population),
            FIR=fir, population_FIR=klados_population_fir,
            selected_FIR=klados_population_fir + fir_alpha * (fir - klados_population_fir),
            FIR_lags=np.asarray(lags), active_C=active_C, quiet_C=quiet_C,
            population_active_C=klados_population_active,
            population_quiet_C=klados_population_quiet,
            selected_alpha=np.asarray(alpha), selected_fir_alpha=np.asarray(fir_alpha),
            selected_state_alpha=np.asarray(state_alpha),
            wrong_C=np.stack([value.full_transfer for value in wrong_contexts]),
            wrong_FIR=np.stack(wrong_fir), wrong_active_C=np.stack(wrong_active),
            wrong_quiet_C=np.stack(wrong_quiet),
            wrong_keys=np.asarray([value.context_id for value in wrong_contexts]),
            wrong_roles=np.asarray(["source_record_control"] * 3),
        )
    _write_csv(root / "operator_audit.csv", operator_rows)
    _write_csv(root / "support_crossfit.csv", crossfit_rows)
    sge_crossfit = [row for row in crossfit_rows if row["study"] != "klados_v4_source_records"]
    klados_crossfit = [row for row in crossfit_rows if row["study"] == "klados_v4_source_records"]
    matching_better_population = np.asarray([row["matching_full_C_error"] < row["population_error"] for row in sge_crossfit])
    matching_better_wrong = np.asarray([row["matching_full_C_error"] < np.mean([row[f"wrong_{index}_error"] for index in (1, 2, 3)]) for row in sge_crossfit])
    summary = {
        "status": "completed_full_operator_data_suitability_audit",
        **_implementation(),
        "sge_successful_stems": len(seen), "sge_total_denominator": 59,
        "klados_source_records": len(klados_crossfit),
        "matching_full_C_beats_population_count": int(matching_better_population.sum()),
        "matching_full_C_beats_mean_three_wrong_count": int(matching_better_wrong.sum()),
        "median_selected_alpha": float(np.median([row["selected_alpha"] for row in sge_crossfit])),
        "representation_median_support_errors": {
            "full_C": float(np.median([row["matching_full_C_error"] for row in sge_crossfit])),
            "population": float(np.median([row["population_error"] for row in sge_crossfit])),
            "FIR": float(np.median([row["FIR_support_error"] for row in sge_crossfit])),
            "state_specific": float(np.median([row["state_specific_error"] for row in sge_crossfit])),
        },
        "query_EOG_used_by_denoiser": False,
        "scope": "support-only cross-fit development data-suitability evidence",
    }
    _atomic_json(root / "operator_audit_summary.json", summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _tensor_condition(prepared: Any, indices: np.ndarray, device: torch.device, *, support_summary: bool = False) -> dict[str, torch.Tensor]:
    source = prepared.training
    condition = {
        "observed": torch.as_tensor(np.array(source.observed[indices], copy=True), device=device),
        "full_transfer": torch.as_tensor(np.array(source.full_transfer[indices], copy=True), device=device),
        "normalized_transfer": torch.as_tensor(np.array(source.normalized_transfer[indices], copy=True), device=device),
        "transfer_scale": torch.as_tensor(np.array(source.transfer_scale[indices], copy=True), device=device),
        "singular_values": torch.as_tensor(np.array(source.singular_values[indices], copy=True), device=device),
        "rank": torch.as_tensor(np.array(source.rank[indices], copy=True), device=device),
        "rho": torch.as_tensor(np.array(source.rho[indices], copy=True), device=device),
        "calibration_duration_seconds": torch.as_tensor(np.array(source.calibration_duration_seconds[indices], copy=True), device=device),
        "channel_mask": torch.as_tensor(np.array(source.channel_mask[indices], copy=True), device=device),
        "valid_time_mask": torch.as_tensor(np.array(source.valid_time_mask[indices], copy=True), device=device),
    }
    return condition


def _j2_technical(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    """One real-record engineering check; no scientific route ranking."""

    base, root = _load(config)
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _prepared

    prepared = _prepared(base, "klados", 0)
    source = prepared.training
    target, coordinate_mean, coordinate_std = canonical_eog_latent(
        source.standardized_artifact_latent,
        prepared.latent_normalizer.mean,
        prepared.latent_normalizer.standard_deviation,
        source.transfer_scale,
        source.valid_time_mask,
    )
    indices = np.arange(min(4, source.observed.shape[0]))
    device = torch.device("cuda", 0)
    condition = _tensor_condition(prepared, indices, device)
    population_condition = _condition_from_indices(
        prepared, indices, device, population_probability=1.0,
        rng=np.random.default_rng(20260804),
    )
    target_tensor = torch.as_tensor(target[indices], device=device)
    model_config = ArtifactLatentModelConfig(
        eeg_channels=prepared.model_dimensions.eeg_channels,
        signal_length=prepared.model_dimensions.signal_length,
        latent_channels=prepared.model_dimensions.eog_coordinates,
        base_channels=32,
        time_sinusoidal_dim=64,
        time_embed_dim=256,
    )
    deterministic = DeterministicArtifactEstimator(model_config).to(device)
    diffusion = ArtifactLatentDiffusion(
        model_config,
        ArtifactLatentDiffusionConfig(
            num_timesteps=1000, min_snr_gamma=5.0,
            dynamic_threshold_quantile=.995,
            standardized_latent_absolute_clip=5.0,
            posterior_samples=8,
        ),
    ).to(device)
    optimizer = AdamW([*deterministic.parameters(), *diffusion.parameters()], lr=5.0e-4)
    generator = torch.Generator(device=device).manual_seed(20260804)
    valid = condition["valid_time_mask"][:, None].to(target_tensor.dtype)
    initial = None
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        det_prediction = deterministic(**condition)
        det_loss = ((det_prediction - target_tensor).square() * valid).sum() / (valid.sum() * target_tensor.shape[1]).clamp_min(1)
        diff_loss, _ = diffusion.training_loss(target_tensor, generator=generator, **condition)
        loss = det_loss + diff_loss
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
    assert initial is not None
    with torch.no_grad():
        det_prediction = deterministic(**condition)
        physical = physical_eog_latent(det_prediction, torch.as_tensor(coordinate_mean, device=device), torch.as_tensor(coordinate_std, device=device))
        population_transfer = torch.as_tensor(prepared.population_context.full_transfer, device=device, dtype=physical.dtype)[None].expand(len(indices), -1, -1)
        population_delta = torch.einsum("bce,bet->bct", population_transfer, physical)
        population_output = condition["observed"] - population_delta
        gate = activity_gate_from_eeg_latent(physical, float(np.quantile(np.abs(target), .75)))
        restored, correction = full_c_subject_residual(condition["observed"], population_output, physical, population_transfer, condition["full_transfer"], gate, condition["valid_time_mask"])
        calls = 0
        def forbidden_factory() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal calls
            calls += 1
            return restored, correction
        fallback, _, constructed = lazy_subject_residual(population_output, 0.0, forbidden_factory)
    checkpoint = root / "technical_check/models.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.partial")
    torch.save({"deterministic": deterministic.state_dict(), "diffusion": diffusion.state_dict(), "optimizer": optimizer.state_dict(), "generator": generator.get_state(), "coordinate_mean": coordinate_mean, "coordinate_std": coordinate_std}, temporary)
    os.replace(temporary, checkpoint)
    reloaded = DeterministicArtifactEstimator(model_config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["deterministic"])
    reloaded.eval(); deterministic.eval()
    with torch.no_grad():
        reload_equal = torch.equal(reloaded(**condition), deterministic(**condition))
    checks = {
        "real_EEG_finite": bool(torch.isfinite(restored).all() and torch.isfinite(loss)),
        "shared_canonical_target": target.shape == source.standardized_artifact_latent.shape,
        "overfit_loss_decreased": float(loss.detach()) < initial * .25,
        "finite_gradient": all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in deterministic.parameters()),
        "context_intervention_changes_output": float(torch.linalg.vector_norm(correction)) > 1.0e-7,
        "g0_short_circuit": torch.equal(fallback, population_output) and not constructed and calls == 0,
        "checkpoint_reload_exact": reload_equal,
        "query_fields_absent": True,
        "population_condition_dtype_safe": all(
            population_condition[name].dtype == condition[name].dtype
            for name in ("full_transfer", "normalized_transfer", "transfer_scale", "singular_values")
        ) and bool((population_condition["rho"] == 0).all()),
    }
    summary = {"status": "passed_wide_v2_technical_check" if all(checks.values()) else "failed_wide_v2_technical_check", **_implementation(), "checks": checks, "initial_loss": initial, "final_loss": float(loss.detach()), "checkpoint": str(checkpoint), "scientific_evidence": False}
    _atomic_json(root / "technical_check/result_summary.json", summary)
    _atomic_json(run_dir / "result_summary.json", summary)
    if not all(checks.values()):
        raise RuntimeError("wide v2 technical check failed")
    return summary


def _carrier_tasks() -> list[dict[str, Any]]:
    return [{"task": 0, "dataset": "klados", "fold": 0}] + [
        {"task": index + 1, "dataset": "sgeyesub", "fold": index}
        for index in range(25)
    ]


def _inner_split(keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    unique = tuple(dict.fromkeys(str(value) for value in keys))
    if len(unique) < 2:
        index = np.arange(len(keys)); return index[:-1], index[-1:]
    validation = set(unique[-max(1, len(unique) // 5):])
    train = np.asarray([index for index, key in enumerate(keys) if key not in validation], dtype=np.int64)
    valid = np.asarray([index for index, key in enumerate(keys) if key in validation], dtype=np.int64)
    if train.size < 1 or valid.size < 1:
        raise RuntimeError("recording-level inner split is empty")
    return train, valid


def _model_config(prepared: Any) -> ArtifactLatentModelConfig:
    return ArtifactLatentModelConfig(
        eeg_channels=prepared.model_dimensions.eeg_channels,
        signal_length=prepared.model_dimensions.signal_length,
        latent_channels=prepared.model_dimensions.eog_coordinates,
        base_channels=32,
        time_sinusoidal_dim=64,
        time_embed_dim=256,
    )


def _diffusion_config() -> ArtifactLatentDiffusionConfig:
    return ArtifactLatentDiffusionConfig(
        num_timesteps=1000, min_snr_gamma=5.0,
        dynamic_threshold_quantile=.995,
        standardized_latent_absolute_clip=5.0,
        posterior_samples=8,
    )


def _ema_update(ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _canonical_training(prepared: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = prepared.training
    return canonical_eog_latent(
        source.standardized_artifact_latent,
        prepared.latent_normalizer.mean,
        prepared.latent_normalizer.standard_deviation,
        source.transfer_scale,
        source.valid_time_mask,
    )


def _condition_from_indices(prepared: Any, indices: np.ndarray, device: torch.device, *, population_probability: float, rng: np.random.Generator, support_summary: bool = False) -> dict[str, torch.Tensor]:
    condition = _tensor_condition(prepared, indices, device, support_summary=support_summary)
    draw = rng.random(indices.size) < population_probability
    if draw.any():
        population = prepared.population_context
        replacements = {
            "full_transfer": population.full_transfer,
            "normalized_transfer": population.normalized_transfer,
            "transfer_scale": population.transfer_scale,
            "singular_values": population.singular_values,
        }
        for name, value in replacements.items():
            condition[name][torch.as_tensor(draw, device=device)] = torch.as_tensor(
                np.array(value, copy=True), device=device, dtype=condition[name].dtype
            )
        condition["rank"][torch.as_tensor(draw, device=device)] = int(population.rank)
        condition["rho"][torch.as_tensor(draw, device=device)] = 0.0
        condition["calibration_duration_seconds"][torch.as_tensor(draw, device=device)] = 0.0
    return condition


def _validation_loss(model: torch.nn.Module, prepared: Any, target: np.ndarray, indices: np.ndarray, device: torch.device, *, diffusion: bool, population_only: bool = False) -> float:
    model.eval(); values = []
    generator = torch.Generator(device=device).manual_seed(8143)
    with torch.no_grad():
        for start in range(0, indices.size, 32):
            chosen = indices[start:start + 32]
            condition = _tensor_condition(prepared, chosen, device, support_summary=isinstance(model, SupportFiLMArtifactLatentDiffusion))
            if population_only:
                population = prepared.population_context
                for name, value in {
                    "full_transfer": population.full_transfer,
                    "normalized_transfer": population.normalized_transfer,
                    "transfer_scale": population.transfer_scale,
                    "singular_values": population.singular_values,
                }.items():
                    condition[name][:] = torch.as_tensor(
                        np.array(value, copy=True), device=device, dtype=condition[name].dtype
                    )
                condition["rank"][:] = int(population.rank)
                condition["rho"][:] = 0.0
                condition["calibration_duration_seconds"][:] = 0.0
            truth = torch.as_tensor(target[chosen], device=device)
            mask = condition["valid_time_mask"][:, None].to(truth.dtype)
            if diffusion:
                timestep = torch.full((chosen.size,), 500, device=device, dtype=torch.long)
                noise = torch.zeros_like(truth)
                _, detail = model.training_loss(truth, timestep=timestep, noise=noise, generator=generator, **condition)  # type: ignore[attr-defined]
                values.append(float(detail["x0_mse"]))
            else:
                prediction = model(**condition)
                values.append(float((((prediction - truth).square() * mask).sum() / (mask.sum() * truth.shape[1]).clamp_min(1)).cpu()))
    model.train(); return float(np.mean(values))


def _train_carrier_fold(config: Mapping[str, Any], prepared: Any, dataset: str, fold: int, root: Path, device: torch.device, *, seed: int | None = None) -> tuple[DeterministicArtifactEstimator, ArtifactLatentDiffusion, np.ndarray, np.ndarray, Path]:
    seed = int(_mapping(config, "routes")["carrier_screen_seed"] if seed is None else seed)
    output = root / "carrier_screen/checkpoints" / dataset / f"fold_{fold:02d}" / f"seed_{seed}"
    checkpoint = output / "models.pt"
    target, coordinate_mean, coordinate_std = _canonical_training(prepared)
    cfg = _model_config(prepared)
    deterministic = DeterministicArtifactEstimator(cfg).to(device)
    diffusion = ArtifactLatentDiffusion(cfg, _diffusion_config()).to(device)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        deterministic.load_state_dict(payload["deterministic_best"])
        diffusion.load_state_dict(payload["diffusion_ema_best"])
        deterministic.eval(); diffusion.eval()
        return deterministic, diffusion, np.asarray(payload["coordinate_mean"]), np.asarray(payload["coordinate_std"]), checkpoint
    torch.manual_seed(seed + fold * 101)
    torch.cuda.manual_seed_all(seed + fold * 101)
    rng = np.random.default_rng(seed + fold * 101)
    generator = torch.Generator(device=device).manual_seed(seed + fold * 1009)
    det_opt = AdamW(deterministic.parameters(), lr=2e-4, weight_decay=1e-4)
    diff_opt = AdamW(diffusion.parameters(), lr=2e-4, weight_decay=1e-4)
    population_diffusion = ArtifactLatentDiffusion(cfg, _diffusion_config()).to(device)
    population_opt = AdamW(population_diffusion.parameters(), lr=2e-4, weight_decay=1e-4)
    ema = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}
    population_ema = {key: value.detach().clone() for key, value in population_diffusion.state_dict().items()}
    train_indices, validation_indices = _inner_split(prepared.training.recording_keys)
    maximum = int(_mapping(_mapping(config, "routes"), "training")["maximum_updates"])
    interval = int(_mapping(_mapping(config, "routes"), "training")["validation_interval_updates"])
    batch_size = int(_mapping(_mapping(config, "routes"), "training")["batch_size"])
    best_det = (float("inf"), None, 0); best_diff = (float("inf"), None, 0); best_population = (float("inf"), None, 0)
    curve = []; start_step = 1; resume_path = output / "resume.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        deterministic.load_state_dict(resume["deterministic"]); diffusion.load_state_dict(resume["diffusion"])
        population_diffusion.load_state_dict(resume["population_diffusion"])
        det_opt.load_state_dict(resume["det_optimizer"]); diff_opt.load_state_dict(resume["diff_optimizer"]); population_opt.load_state_dict(resume["population_optimizer"])
        ema = {key: value.to(device) for key, value in resume["ema"].items()}
        population_ema = {key: value.to(device) for key, value in resume["population_ema"].items()}
        best_det, best_diff, best_population = resume["best_det"], resume["best_diff"], resume["best_population"]
        rng.bit_generator.state = resume["numpy_rng_state"]; generator.set_state(resume["torch_generator_state"].cpu())
        curve = list(resume.get("curve", [])); start_step = int(resume["step"]) + 1
    for step in range(start_step, maximum + 1):
        chosen = rng.choice(train_indices, batch_size, replace=train_indices.size < batch_size)
        condition = _condition_from_indices(prepared, chosen, device, population_probability=.25, rng=rng)
        truth = torch.as_tensor(target[chosen], device=device)
        mask = condition["valid_time_mask"][:, None].to(truth.dtype)
        det_opt.zero_grad(set_to_none=True)
        det_prediction = deterministic(**condition)
        det_loss = ((det_prediction - truth).square() * mask).sum() / (mask.sum() * truth.shape[1]).clamp_min(1)
        det_loss.backward(); torch.nn.utils.clip_grad_norm_(deterministic.parameters(), 1.0); det_opt.step()
        diff_opt.zero_grad(set_to_none=True)
        diff_loss, detail = diffusion.training_loss(truth, generator=generator, **condition)
        diff_loss.backward(); torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0); diff_opt.step(); _ema_update(ema, diffusion, .999)
        population_condition = _condition_from_indices(prepared, chosen, device, population_probability=1.0, rng=rng)
        population_opt.zero_grad(set_to_none=True)
        population_loss, _ = population_diffusion.training_loss(truth, generator=generator, **population_condition)
        population_loss.backward(); torch.nn.utils.clip_grad_norm_(population_diffusion.parameters(), 1.0); population_opt.step(); _ema_update(population_ema, population_diffusion, .999)
        if step % interval == 0 or step == maximum:
            det_val = _validation_loss(deterministic, prepared, target, validation_indices, device, diffusion=False)
            raw = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}
            diffusion.load_state_dict(ema)
            diff_val = _validation_loss(diffusion, prepared, target, validation_indices, device, diffusion=True)
            ema_snapshot = _state_cpu(diffusion)
            diffusion.load_state_dict(raw)
            population_raw = {key: value.detach().clone() for key, value in population_diffusion.state_dict().items()}
            population_diffusion.load_state_dict(population_ema)
            population_val = _validation_loss(population_diffusion, prepared, target, validation_indices, device, diffusion=True, population_only=True)
            population_snapshot = _state_cpu(population_diffusion)
            population_diffusion.load_state_dict(population_raw)
            if det_val < best_det[0]: best_det = (det_val, _state_cpu(deterministic), step)
            if diff_val < best_diff[0]: best_diff = (diff_val, ema_snapshot, step)
            if population_val < best_population[0]: best_population = (population_val, population_snapshot, step)
            curve.append({"step": step, "det_train": float(det_loss.detach()), "diff_train": float(diff_loss.detach()), "population_diff_train": float(population_loss.detach()), "diff_x0": float(detail["x0_mse"]), "det_validation": det_val, "diffusion_EMA_validation": diff_val, "population_diffusion_EMA_validation": population_val})
            output.mkdir(parents=True, exist_ok=True); temporary_resume = resume_path.with_name(f".{resume_path.name}.{os.getpid()}.partial")
            torch.save({"step": step, "deterministic": _state_cpu(deterministic), "diffusion": _state_cpu(diffusion), "population_diffusion": _state_cpu(population_diffusion), "det_optimizer": det_opt.state_dict(), "diff_optimizer": diff_opt.state_dict(), "population_optimizer": population_opt.state_dict(), "ema": {key: value.detach().cpu() for key, value in ema.items()}, "population_ema": {key: value.detach().cpu() for key, value in population_ema.items()}, "best_det": best_det, "best_diff": best_diff, "best_population": best_population, "numpy_rng_state": rng.bit_generator.state, "torch_generator_state": generator.get_state(), "curve": curve}, temporary_resume); os.replace(temporary_resume, resume_path)
    if best_det[1] is None or best_diff[1] is None or best_population[1] is None:
        raise RuntimeError("carrier training produced no validation checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.partial")
    torch.save({"deterministic_best": best_det[1], "diffusion_ema_best": best_diff[1], "population_diffusion_ema_best": best_population[1], "coordinate_mean": coordinate_mean, "coordinate_std": coordinate_std, "best_det_step": best_det[2], "best_diff_step": best_diff[2], "best_population_step": best_population[2], "model_config": cfg.__dict__, "EMA_used_for_validation_and_inference": True}, temporary); os.replace(temporary, checkpoint)
    _write_csv(output / "training_curve.csv", curve)
    deterministic.load_state_dict(best_det[1]); diffusion.load_state_dict(best_diff[1]); deterministic.eval(); diffusion.eval()
    return deterministic, diffusion, coordinate_mean, coordinate_std, checkpoint


def _runtime_condition(observed: torch.Tensor, valid: torch.Tensor, transfer: np.ndarray, rho: float, duration: float) -> dict[str, torch.Tensor]:
    value = np.asarray(transfer, dtype=np.float64)
    scale = np.linalg.norm(value, axis=0); scale = np.maximum(scale, 1.0e-8)
    normalized = value / scale[None]
    singular = np.linalg.svd(value, compute_uv=False)
    batch = observed.shape[0]
    return {
        "observed": observed,
        "full_transfer": torch.as_tensor(value, device=observed.device, dtype=observed.dtype)[None].expand(batch, -1, -1),
        "normalized_transfer": torch.as_tensor(normalized, device=observed.device, dtype=observed.dtype)[None].expand(batch, -1, -1),
        "transfer_scale": torch.as_tensor(scale, device=observed.device, dtype=observed.dtype)[None].expand(batch, -1),
        "singular_values": torch.as_tensor(singular, device=observed.device, dtype=observed.dtype)[None].expand(batch, -1),
        "rank": torch.full((batch,), min(2, int(np.linalg.matrix_rank(value))), device=observed.device, dtype=torch.long),
        "rho": torch.full((batch,), float(rho), device=observed.device),
        "calibration_duration_seconds": torch.full((batch,), float(duration), device=observed.device),
        "channel_mask": torch.ones((batch, observed.shape[1]), device=observed.device, dtype=torch.bool),
        "valid_time_mask": valid,
    }


def _calibration_windows(
    eeg: np.ndarray,
    eog: np.ndarray,
    *,
    signal_length: int,
    coordinate_mean: np.ndarray,
    coordinate_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create contiguous support-only adapter examples without query fields."""

    y = np.asarray(eeg, dtype=np.float32)
    a = np.asarray(eog, dtype=np.float32)
    if y.ndim != 2 or a.ndim != 2 or y.shape[1] != a.shape[1]:
        raise ValueError("calibration EEG/EOG shapes differ")
    observed, target, masks = [], [], []
    for start in range(0, y.shape[1], signal_length):
        stop = min(y.shape[1], start + signal_length)
        if stop - start < max(16, signal_length // 4):
            continue
        current_y = np.zeros((y.shape[0], signal_length), dtype=np.float32)
        current_a = np.zeros((a.shape[0], signal_length), dtype=np.float32)
        mask = np.zeros(signal_length, dtype=bool)
        current_y[:, : stop - start] = y[:, start:stop]
        current_a[:, : stop - start] = (
            a[:, start:stop] - coordinate_mean[:, None]
        ) / coordinate_std[:, None]
        mask[: stop - start] = True
        observed.append(current_y); target.append(current_a); masks.append(mask)
    if not observed:
        raise RuntimeError("calibration support is too short for adapter fitting")
    return np.stack(observed), np.stack(target), np.stack(masks)


def _support_adapter_arrays(
    base: Mapping[str, Any], prepared: Any, dataset: str, recording_key: str,
    coordinate_mean: np.ndarray, coordinate_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = str(recording_key).split(":", 1)[0]
    if dataset == "sgeyesub":
        loaded = _loaded_record(prepared, key)
        count, _ = _calibration_samples(base, loaded.sampling_rate_hz, loaded.support.eeg.shape[1])
        eeg = prepared.normalizer.transform(loaded.support.eeg)[:, :count]
        eog = _standardize_full(loaded.support.external_eog[:, :count])
    else:
        from eeg_cgdr.experiments.subject_artifact_subspace_diffusion import _klados_eval_records
        records = {value[0]: value[1] for value in _klados_eval_records(base)}
        if key not in records:
            raise KeyError(f"Klados support record is unavailable: {key}")
        eeg = np.asarray(records[key].calibration.eeg, dtype=np.float32)
        eog = np.asarray(records[key].calibration.eog, dtype=np.float32)
    return _calibration_windows(
        eeg, eog, signal_length=prepared.model_dimensions.signal_length,
        coordinate_mean=coordinate_mean, coordinate_std=coordinate_std,
    )


def _fit_support_adapter(
    model: SupportLoRAArtifactLatentDiffusion,
    support: tuple[np.ndarray, np.ndarray, np.ndarray],
    transfer: np.ndarray,
    rho: float,
    duration: float,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> None:
    model.reset_support_adapter(); model.freeze_population_backbone(); model.eval()
    observed, target, valid = support
    adapter = _mapping(_mapping(config, "routes"), "support_adapter")
    updates = int(adapter["updates"]); batch_size = int(adapter["batch_size"])
    optimizer = AdamW(model.output_adapter.parameters(), lr=float(adapter["learning_rate"]), weight_decay=0.0)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 991)
    for _ in range(updates):
        chosen = rng.choice(observed.shape[0], batch_size, replace=observed.shape[0] < batch_size)
        y = torch.as_tensor(observed[chosen], device=device)
        mask = torch.as_tensor(valid[chosen], device=device)
        truth = torch.as_tensor(target[chosen], device=device)
        condition = _runtime_condition(y, mask, transfer, rho, duration)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.training_loss(truth, generator=generator, **condition)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.output_adapter.parameters(), 1.0); optimizer.step()
    if not all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.output_adapter.parameters()):
        raise RuntimeError("support-only adapter produced non-finite gradients")


@torch.no_grad()
def _predict_context_latents(
    deterministic: DeterministicArtifactEstimator,
    diffusion: ArtifactLatentDiffusion,
    *, observed: np.ndarray, valid: np.ndarray, transfers: Mapping[str, np.ndarray],
    rho: float, duration: float, coordinate_mean: np.ndarray,
    coordinate_std: np.ndarray, unit_key: str, training_seed: int,
    device: torch.device, config: Mapping[str, Any] | None = None,
    support_sets: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    det: dict[str, list[np.ndarray]] = defaultdict(list)
    diff: dict[str, list[np.ndarray]] = defaultdict(list)
    pop_parts: list[np.ndarray] = []
    base_seeds = participant_sample_seeds(unit_key, training_seed)
    if isinstance(diffusion, SupportLoRAArtifactLatentDiffusion):
        if config is None or support_sets is None:
            raise ValueError("support-only adapter inference requires calibration support sets")
        for role, transfer in transfers.items():
            if role == "population":
                diffusion.reset_support_adapter(); diffusion.freeze_population_backbone(); diffusion.eval()
            else:
                if role not in support_sets:
                    raise KeyError(f"support adapter calibration is missing for {role}")
                adapter_seed = training_seed + sum((index + 1) * ord(char) for index, char in enumerate(f"{unit_key}|{role}"))
                _fit_support_adapter(diffusion, support_sets[role], transfer, rho, duration, config, device, adapter_seed)
            for batch_index, start in enumerate(range(0, observed.shape[0], 32)):
                stop = min(observed.shape[0], start + 32)
                y = torch.as_tensor(observed[start:stop], device=device, dtype=torch.float32)
                mask = torch.as_tensor(valid[start:stop], device=device, dtype=torch.bool)
                condition = _runtime_condition(y, mask, transfer, rho if role != "population" else 0.0, duration if role != "population" else 0.0)
                det[role].append(deterministic(**condition).cpu().numpy())
                sample_seeds = tuple(value + batch_index * 104729 for value in base_seeds)
                posterior = diffusion.posterior_mean(
                    **condition,
                    latent_mean=torch.as_tensor(coordinate_mean, device=device),
                    latent_standard_deviation=torch.as_tensor(coordinate_std, device=device),
                    sample_seeds=sample_seeds, ddim_steps=25,
                )
                if posterior.standardized_latent_samples is None:
                    raise RuntimeError("adapter diffusion did not retain posterior samples")
                diff[role].append(posterior.standardized_latent_samples.cpu().numpy())
        return np.asarray(observed), {key: np.concatenate(value) for key, value in det.items()}, {key: np.concatenate(value, axis=1) for key, value in diff.items()}
    for batch_index, start in enumerate(range(0, observed.shape[0], 32)):
        stop = min(observed.shape[0], start + 32)
        y = torch.as_tensor(observed[start:stop], device=device, dtype=torch.float32)
        mask = torch.as_tensor(valid[start:stop], device=device, dtype=torch.bool)
        sample_seeds = tuple(value + batch_index * 104729 for value in base_seeds)
        for role, transfer in transfers.items():
            condition = _runtime_condition(y, mask, transfer, rho if role != "population" else 0.0, duration if role != "population" else 0.0)
            det[role].append(deterministic(**condition).cpu().numpy())
            posterior = diffusion.posterior_mean(
                **condition,
                latent_mean=torch.as_tensor(coordinate_mean, device=device),
                latent_standard_deviation=torch.as_tensor(coordinate_std, device=device),
                sample_seeds=sample_seeds, ddim_steps=25,
            )
            if posterior.standardized_latent_samples is None:
                raise RuntimeError("diffusion did not retain posterior samples")
            diff[role].append(posterior.standardized_latent_samples.cpu().numpy())
        pop_parts.append(y.cpu().numpy())
    return np.concatenate(pop_parts), {key: np.concatenate(value) for key, value in det.items()}, {key: np.concatenate(value, axis=1) for key, value in diff.items()}


def _carrier_output(
    route: str,
    observed: np.ndarray,
    population_output: np.ndarray,
    physical_latent: np.ndarray,
    cache: Mapping[str, np.ndarray],
    transfer: np.ndarray,
    gate: float,
    valid: np.ndarray,
) -> np.ndarray:
    y = torch.as_tensor(observed)
    x_pop = torch.as_tensor(population_output, dtype=y.dtype)
    latent = torch.as_tensor(physical_latent, dtype=y.dtype)
    c0 = torch.as_tensor(np.asarray(cache["population_C"]), dtype=y.dtype)
    cs = torch.as_tensor(transfer, dtype=y.dtype)
    mask = torch.as_tensor(valid)
    wrong_index = None
    if "wrong_C" in cache:
        for index, candidate in enumerate(np.asarray(cache["wrong_C"])):
            if np.allclose(transfer, candidate, atol=1.0e-6, rtol=1.0e-5):
                wrong_index = index
                break
    if route == "R1_full_C_residual":
        output, _ = full_c_subject_residual(y, x_pop, latent, c0, cs, gate, mask)
        return output.numpy()
    if route == "R2_FIR_residual":
        subject_fir_value = np.asarray(cache["FIR"] if wrong_index is None else cache["wrong_FIR"][wrong_index])
        subject_fir = torch.as_tensor(subject_fir_value, dtype=y.dtype)
        lags = tuple(int(value) for value in np.asarray(cache["FIR_lags"]).tolist())
        population_fir = torch.as_tensor(np.asarray(cache["population_FIR"]), dtype=y.dtype)
        restored, _, _ = fir_full_replacement(
            y,
            latent,
            population_fir,
            subject_fir,
            lags,
            float(gate),
            mask,
        )
        return restored.numpy()
    if route == "R3_state_gated_residual":
        activity = activity_gate_from_eeg_latent(latent, 1.0, temperature=.2).numpy()
        active = np.asarray(cache["active_C"] if wrong_index is None else cache["wrong_active_C"][wrong_index], dtype=np.float32)
        quiet = np.asarray(cache["quiet_C"] if wrong_index is None else cache["wrong_quiet_C"][wrong_index], dtype=np.float32)
        population_active = np.asarray(cache["population_active_C"], dtype=np.float32)
        population_quiet = np.asarray(cache["population_quiet_C"], dtype=np.float32)
        effective = quiet[None, :, :, None] + activity[:, None] * (active - quiet)[None, :, :, None]
        population_effective = population_quiet[None, :, :, None] + activity[:, None] * (population_active - population_quiet)[None, :, :, None]
        residual = np.einsum("bcet,bet->bct", effective - population_effective, physical_latent)
        return population_output - float(gate) * residual * valid[:, None]
    raise ValueError(f"unknown carrier route: {route}")


def _score_carrier_outputs(
    config: Mapping[str, Any], base: Mapping[str, Any], prepared: Any,
    dataset: str, fold: int, unit_key: str, observed: np.ndarray,
    clean: np.ndarray | None, valid: np.ndarray, outputs: Mapping[str, np.ndarray],
    output_archive: Path, server_only_arrays: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    server_only_arrays = dict(server_only_arrays or {})
    if clean is not None:
        server_only_arrays["clean_target"] = np.asarray(clean)
    archive_values = {key.replace("-", "_"): value for key, value in outputs.items()}
    if server_only_arrays:
        archive_values.update({f"server_only__{key.replace('-', '_')}": value for key, value in server_only_arrays.items()})
    np.savez_compressed(output_archive, **archive_values)
    rows = []
    if dataset == "klados":
        from eeg_cgdr.experiments.subject_artifact_subspace_diffusion import _paired_metrics
        assert clean is not None
        for method, value in outputs.items():
            rows.append({"dataset": dataset, "unit_id": unit_key, "exact_cell": prepared.fold.layout_id, "method": method, "status": "success", **_paired_metrics(observed, clean, value, valid), "statistical_unit": "source_record"})
        return rows
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import (
        _annotation_opener, _continuous, _evaluate_output, _sge_samples_per_trial,
    )
    heldout = prepared.heldout[unit_key]
    annotated = _annotation_opener(base, prepared, unit_key)()
    annotation = annotated.query_annotations
    if annotation is None:
        raise RuntimeError("SGE annotations failed to open after output freeze")
    observed_continuous = _continuous(observed)
    for method, value in outputs.items():
        metric = _evaluate_output(
            method_id=method, output=_continuous(value), observed=observed_continuous,
            matching_projector=heldout.matching.projector,
            population_projector=prepared.population_context.projector,
            query_eog=annotation.external_eog, artifactclasses=annotation.artifactclasses,
            predicted_contamination=None, trial_labels=annotation.trial_labels,
            samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz),
            minimum_trials_per_condition=2, status="success", operator_source="wide_v2_canonical_carrier",
            gamma=None, fallback_used=False, uses_query_external_eog=False,
        )
        output_continuous = _continuous(value)
        output_input_rms = float(
            np.sqrt(np.mean(np.square(output_continuous)))
            / max(np.sqrt(np.mean(np.square(observed_continuous))), np.finfo(float).eps)
        )
        rows.append({"dataset": dataset, "unit_id": unit_key, "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}", "study": prepared.fold.study, "method": method, **metric, "output_input_RMS_ratio": output_input_rms, "outputs_frozen_before_scoring": True})
    return rows


def _evaluate_carrier_fold(config: Mapping[str, Any], base: Mapping[str, Any], root: Path, prepared: Any, dataset: str, fold: int, deterministic: DeterministicArtifactEstimator, diffusion: ArtifactLatentDiffusion, coordinate_mean: np.ndarray, coordinate_std: np.ndarray, device: torch.device, *, carriers: Sequence[str] | None = None, conditioning: str = "structured", training_seed: int = 20260811, archive_scope: str = "carrier_screen") -> list[dict[str, Any]]:
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _load_models as _load_anchor
    old_config = yaml.safe_load((CODE_ROOT / "configs/cgdr/mainline_subject_residual_diffusion.yaml").read_text(encoding="utf-8"))
    route = {"dataset": dataset, "fold_index": fold, "seed": training_seed}
    anchor, _, _, _, _ = _load_anchor(old_config, prepared, route, device); anchor.eval()
    if dataset == "klados":
        from eeg_cgdr.experiments.subject_artifact_subspace_diffusion import _klados_eval_records
        units = [(key, mechanism.observed_windows.astype(np.float32), mechanism.clean_windows.astype(np.float32), mechanism.valid_time_weight.astype(bool), matching) for key, mechanism, matching, _ in _klados_eval_records(base)]
    else:
        units = [(key, value.query.observed, None, value.query.valid_time_mask, value.matching) for key, value in prepared.heldout.items()]
    all_rows = []
    for unit_key, observed, clean, valid, matching in units:
        cache_path = root / "operator_cache" / dataset / f"{unit_key.replace('/', '__')}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"operator cache missing: {cache_path}")
        with np.load(cache_path) as loaded:
            cache = {key: np.asarray(loaded[key]) for key in loaded.files}
        alpha = float(cache["selected_alpha"])
        wrong_values = np.asarray(cache["wrong_C"])
        transfers = {"population": np.asarray(cache["population_C"]), "matching": np.asarray(cache["full_C"]), **{f"wrong{index+1}": wrong_values[index] for index in range(min(3, wrong_values.shape[0]))}}
        support_sets = None
        if isinstance(diffusion, SupportLoRAArtifactLatentDiffusion):
            wrong_keys = [str(value) for value in np.asarray(cache["wrong_keys"]).tolist()]
            support_sets = {
                "matching": _support_adapter_arrays(base, prepared, dataset, unit_key, coordinate_mean, coordinate_std),
                **{
                    f"wrong{index + 1}": _support_adapter_arrays(base, prepared, dataset, wrong_key, coordinate_mean, coordinate_std)
                    for index, wrong_key in enumerate(wrong_keys[:3])
                },
            }
        with torch.no_grad():
            parts = []
            for start in range(0, observed.shape[0], 32):
                y = torch.as_tensor(observed[start:start+32], device=device)
                mask = torch.as_tensor(valid[start:start+32], device=device)
                parts.append(anchor(y, mask).cpu().numpy())
            x_pop = np.concatenate(parts)
        routes = tuple(carriers or ("R1_full_C_residual", "R2_FIR_residual", "R3_state_gated_residual"))
        gamma_values = tuple(float(value) for value in _mapping(config, "evaluation")["gamma_sweep"])
        outputs = {"RAW": observed, "POP": x_pop}
        server_only: dict[str, np.ndarray] = {}
        for carrier in routes:
            prefix = carrier if conditioning == "structured" else f"{carrier}|{conditioning}"
            if carrier == "R2_FIR_residual":
                deployment_alpha = float(cache["selected_fir_alpha"])
            elif carrier == "R3_state_gated_residual":
                deployment_alpha = float(cache["selected_state_alpha"])
            else:
                deployment_alpha = alpha
            _, det_latents, diff_latents = _predict_context_latents(
                deterministic, diffusion, observed=observed, valid=valid,
                transfers=transfers, rho=deployment_alpha,
                duration=float(matching.calibration_duration_seconds),
                coordinate_mean=coordinate_mean, coordinate_std=coordinate_std,
                unit_key=unit_key, training_seed=training_seed, device=device,
                config=config, support_sets=support_sets,
            )
            _, det_mechanism_latents, diff_mechanism_latents = _predict_context_latents(
                deterministic, diffusion, observed=observed, valid=valid,
                transfers=transfers, rho=1.0,
                duration=float(matching.calibration_duration_seconds),
                coordinate_mean=coordinate_mean, coordinate_std=coordinate_std,
                unit_key=unit_key, training_seed=training_seed, device=device,
                config=config, support_sets=support_sets,
            )
            det_physical = det_latents["matching"] * coordinate_std[None, :, None] + coordinate_mean[None, :, None]
            det_population_physical = det_latents["population"] * coordinate_std[None, :, None] + coordinate_mean[None, :, None]
            diff_match_samples = diff_latents["matching"] * coordinate_std[None, None, :, None] + coordinate_mean[None, None, :, None]
            diff_match = diff_match_samples.mean(0)
            diff_population_samples = diff_latents["population"] * coordinate_std[None, None, :, None] + coordinate_mean[None, None, :, None]
            diff_mechanism_match = diff_mechanism_latents["matching"].mean(0) * coordinate_std[None, :, None] + coordinate_mean[None, :, None]
            outputs[f"{prefix}|DET-MATCH|deployment"] = _carrier_output(carrier, observed, x_pop, det_physical, cache, transfers["matching"], deployment_alpha, valid)
            outputs[f"{prefix}|DET-POP"] = _carrier_output(carrier, observed, x_pop, det_population_physical, cache, transfers["population"], 0.0, valid)
            outputs[f"{prefix}|DIFF-POP"] = _carrier_output(carrier, observed, x_pop, diff_population_samples.mean(0), cache, transfers["population"], 0.0, valid)
            outputs[f"{prefix}|NO-SUPPORT"] = outputs[f"{prefix}|DIFF-POP"]
            outputs[f"{prefix}|RHO-ONLY-POP-BASIS"] = observed - deployment_alpha * (observed - x_pop)
            outputs[f"{prefix}|CONSTANT-RHO-0.5"] = _carrier_output(carrier, observed, x_pop, diff_match, cache, transfers["matching"], .5, valid)
            outputs[f"{prefix}|DIFF-MATCH-K8|deployment"] = _carrier_output(carrier, observed, x_pop, diff_match, cache, transfers["matching"], deployment_alpha, valid)
            outputs[f"{prefix}|DIFF-MATCH-K1|deployment"] = _carrier_output(carrier, observed, x_pop, diff_match_samples[0], cache, transfers["matching"], deployment_alpha, valid)
            outputs[f"{prefix}|DIFF-MATCH-K8|mechanism_g1"] = _carrier_output(carrier, observed, x_pop, diff_mechanism_match, cache, transfers["matching"], 1.0, valid)
            if carrier == "R1_full_C_residual":
                effective_transfer = np.asarray(cache["population_C"]) + alpha * (
                    np.asarray(cache["full_C"]) - np.asarray(cache["population_C"])
                )
                direct_correction = np.einsum("ce,bet->bct", effective_transfer, diff_match) * valid[:, None]
                outputs[f"{prefix}|DIFF-MATCH|diagnostic_full_replacement"] = observed - direct_correction
            server_only[f"{prefix}|DIFF-MATCH-K8|deployment_samples"] = np.stack([
                _carrier_output(carrier, observed, x_pop, sample, cache, transfers["matching"], deployment_alpha, valid)
                for sample in diff_match_samples
            ])
            for index in range(1, 4):
                key = f"wrong{index}"
                wrong_samples = diff_latents[key] * coordinate_std[None, None, :, None] + coordinate_mean[None, None, :, None]
                wrong_mechanism = diff_mechanism_latents[key].mean(0) * coordinate_std[None, :, None] + coordinate_mean[None, :, None]
                outputs[f"{prefix}|DIFF-WRONG-{index}|deployment"] = _carrier_output(carrier, observed, x_pop, wrong_samples.mean(0), cache, transfers[key], deployment_alpha, valid)
                outputs[f"{prefix}|DIFF-WRONG-{index}|mechanism_g1"] = _carrier_output(carrier, observed, x_pop, wrong_mechanism, cache, transfers[key], 1.0, valid)
            for gamma in gamma_values:
                outputs[f"{prefix}|DIFF-MATCH|gamma={gamma:g}"] = _carrier_output(carrier, observed, x_pop, diff_match, cache, transfers["matching"], min(gamma, 1.0), valid) if gamma <= 1 else x_pop + gamma * (_carrier_output(carrier, observed, x_pop, diff_match, cache, transfers["matching"], 1.0, valid) - x_pop)
        archive = root / archive_scope / "server_arrays" / conditioning / dataset / f"fold_{fold:02d}" / f"seed_{training_seed}" / f"{unit_key.replace('/', '__')}.npz"
        rows = _score_carrier_outputs(config, base, prepared, dataset, fold, unit_key, observed, clean, valid, outputs, archive, server_only)
        for row in rows: row["training_seed"] = training_seed; row["selected_alpha"] = alpha
        all_rows.extend(rows)
    return all_rows


def _j3_carrier_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    base, root = _load(config)
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _prepared
    device = torch.device("cuda", 0)
    tasks = _carrier_tasks()[worker::8]
    completed = []
    for task in tasks:
        dataset, fold = str(task["dataset"]), int(task["fold"])
        prepared = _prepared(base, dataset, fold)
        deterministic, diffusion, mean, std, checkpoint = _train_carrier_fold(config, prepared, dataset, fold, root, device)
        rows = _evaluate_carrier_fold(
            config, base, root, prepared, dataset, fold,
            deterministic, diffusion, mean, std, device,
            carriers=tuple(str(value) for value in _mapping(config, "routes")["carrier_candidates"]),
        )
        output = root / "carrier_screen/evaluation" / dataset / f"fold_{fold:02d}"
        _write_csv(output / "metrics.csv", rows)
        _atomic_json(output / "result_summary.json", {"status": "completed_carrier_fold", "dataset": dataset, "fold": fold, "units": len(set(row["unit_id"] for row in rows)), "checkpoint": str(checkpoint)})
        completed.append({"dataset": dataset, "fold": fold})
    summary = {"status": "completed_carrier_worker", **_implementation(), "worker": worker, "completed": completed}
    _atomic_json(run_dir / "result_summary.json", summary); return summary


def _numeric(value: Any) -> bool:
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False


def _mean_metric(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row.get("method") == method and _numeric(row.get(metric))]
    return float(np.mean(values)) if values else float("nan")


def _complete_query_stability_diagnostics(config: Mapping[str, Any], root: Path) -> None:
    """Append post-output-freeze data-suitability diagnostics to J1 rows."""

    base, _ = _load(config); frozen = _load_frozen_config(base)
    crossfit_path = root / "support_crossfit.csv"
    rows = _read_csvs([crossfit_path]); by_key = {str(row["recording_key"]): row for row in rows}
    audit = _mapping(config, "operator_audit"); ridge = float(audit["ridge_lambda"])
    bootstrap_replicates = int(audit["bootstrap_replicates"])
    from eeg_cgdr.experiments.subject_artifact_development_eval import _annotation_opener, _continuous
    seen = set()
    for unified_index in range(25):
        partition, local_index = _unified_fold_route(frozen, unified_index)
        prepared = _prepare_fold(frozen, partition, local_index)
        for key in prepared.fold.heldout_recording_keys:
            if key in seen:
                continue
            seen.add(key)
            loaded = _loaded_record(prepared, key)
            count, _ = _calibration_samples(base, loaded.sampling_rate_hz, loaded.support.eeg.shape[1])
            support_eeg = prepared.normalizer.transform(loaded.support.eeg)[:, :count]
            raw_support_eog = np.asarray(loaded.support.external_eog[:, :count], dtype=np.float64)
            support_mean = raw_support_eog.mean(axis=1, keepdims=True)
            support_std = raw_support_eog.std(axis=1, keepdims=True)
            support_eog = (raw_support_eog - support_mean) / np.maximum(support_std, 1.0e-8)
            support_C = _ridge(support_eeg, support_eog, ridge)
            annotated = _annotation_opener(base, prepared, key)()
            if annotated.query_annotations is None:
                raise RuntimeError("post-freeze query annotations are unavailable")
            # ``_prepare_fold`` deliberately exposes the sealed block-2 EEG as
            # ``SgeyesubQuerySignals.eeg``.  The windowed ``observed`` field
            # belongs to the later subject-artifact preparation surface and is
            # not present here.  Keep the offline diagnostic in the same
            # outer-training normalization as the support operator.
            query_eeg = prepared.normalizer.transform(
                prepared.heldout[key].query.eeg
            ).astype(np.float64)
            query_eog = (np.asarray(annotated.query_annotations.external_eog, dtype=np.float64) - support_mean) / np.maximum(support_std, 1.0e-8)
            length = min(query_eeg.shape[1], query_eog.shape[1])
            query_eeg, query_eog = query_eeg[:, :length], query_eog[:, :length]
            query_C = _ridge(query_eeg, query_eog, ridge)
            with np.load(root / "operator_cache/sgeyesub" / f"{key.replace('/', '__')}.npz") as cache:
                population = np.asarray(cache["population_C"], dtype=np.float64)
                wrong = np.asarray(cache["wrong_C"], dtype=np.float64)
            rng = np.random.default_rng(20260804 + unified_index)
            block = max(8, count // 8); bootstrap = []
            blocks = [np.arange(start, min(count, start + block)) for start in range(0, count, block)]
            for _ in range(bootstrap_replicates):
                sample = np.concatenate([blocks[int(value)] for value in rng.integers(0, len(blocks), len(blocks))])[:count]
                bootstrap.append(float(np.linalg.norm(_ridge(support_eeg[:, sample], support_eog[:, sample], ridge) - support_C)))
            row = by_key[str(key)]
            row.update({
                "bootstrap_C_variability": float(np.mean(bootstrap)),
                "support_to_later_query_C_distance": float(np.linalg.norm(support_C - query_C)),
                "later_query_matching_prediction_error": _prediction_error(query_eeg, query_eog, support_C),
                "later_query_population_prediction_error": _prediction_error(query_eeg, query_eog, population),
                "later_query_mean_three_wrong_prediction_error": float(np.mean([_prediction_error(query_eeg, query_eog, value) for value in wrong[:3]])),
                "query_EOG_role": "post_output_freeze_offline_data_suitability_scorer_only",
            })
    _write_csv(crossfit_path, list(by_key.values()))


def _j4_rank(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _, root = _load(config)
    _complete_query_stability_diagnostics(config, root)
    # Older retries may have produced the compact summary before Klados rows
    # were separated from the 58-stem SGE denominator.  Recompute these fields
    # from the lightweight CSV without rerunning the data audit.
    audit_path = root / "operator_audit_summary.json"
    crossfit_path = root / "support_crossfit.csv"
    if audit_path.is_file() and crossfit_path.is_file():
        audit_summary = json.loads(audit_path.read_text(encoding="utf-8"))
        audit_rows = _read_csvs([crossfit_path])
        operator_by_key = {str(row["recording_key"]): row for row in _read_csvs([root / "operator_audit.csv"])}
        sge_rows = [row for row in audit_rows if row.get("study") != "klados_v4_source_records"]
        klados_rows = [row for row in audit_rows if row.get("study") == "klados_v4_source_records"]
        audit_summary.update({
            "klados_source_records": len(klados_rows),
            "matching_full_C_beats_population_count": int(sum(float(row["matching_full_C_error"]) < float(row["population_error"]) for row in sge_rows)),
            "matching_full_C_beats_mean_three_wrong_count": int(sum(float(row["matching_full_C_error"]) < np.mean([float(row[f"wrong_{index}_error"]) for index in (1, 2, 3)]) for row in sge_rows)),
            "median_selected_alpha": float(np.median([float(row["selected_alpha"]) for row in sge_rows])),
            "representation_median_support_errors": {
                "full_C": float(np.median([float(row["matching_full_C_error"]) for row in sge_rows])),
                "population": float(np.median([float(row["population_error"]) for row in sge_rows])),
                "FIR": float(np.median([float(row["FIR_support_error"]) for row in sge_rows])),
                "state_specific": float(np.median([float(row["state_specific_error"]) for row in sge_rows])),
            },
            "support_to_later_query": {
                "matching_beats_population_count": int(sum(float(row["later_query_matching_prediction_error"]) < float(row["later_query_population_prediction_error"]) for row in sge_rows)),
                "matching_beats_mean_three_wrong_count": int(sum(float(row["later_query_matching_prediction_error"]) < float(row["later_query_mean_three_wrong_prediction_error"]) for row in sge_rows)),
                "median_operator_distance": float(np.median([float(row["support_to_later_query_C_distance"]) for row in sge_rows])),
            },
            "legacy_rho_audit": {
                "saturation_fraction_at_one": float(np.mean([float(operator_by_key[str(row["recording_key"])].get("rho_legacy", 0.0)) >= .999 for row in sge_rows])),
                "pearson_with_negative_split_half_distance": _correlation(
                    np.asarray([float(operator_by_key[str(row["recording_key"])].get("rho_legacy", 0.0)) for row in sge_rows]),
                    -np.asarray([float(operator_by_key[str(row["recording_key"])]["within_subject_half_distance"]) for row in sge_rows]),
                ),
            },
        })
        _atomic_json(audit_path, audit_summary)
    paths = sorted((root / "carrier_screen/evaluation").glob("**/metrics.csv"))
    if len(paths) != 26:
        raise RuntimeError(f"carrier screen coverage incomplete: {len(paths)}/26 folds")
    rows = _read_csvs(paths)
    route_rows = []
    candidates = tuple(str(value) for value in _mapping(config, "routes")["carrier_candidates"])
    for route in candidates:
        klados = [row for row in rows if row.get("dataset") == "klados"]
        sge = [row for row in rows if row.get("dataset") == "sgeyesub"]
        match = f"{route}|DIFF-MATCH-K8|deployment"
        match_mechanism = f"{route}|DIFF-MATCH-K8|mechanism_g1"
        det = f"{route}|DET-MATCH|deployment"
        wrongs = [f"{route}|DIFF-WRONG-{index}|deployment" for index in range(1, 4)]
        wrongs_mechanism = [f"{route}|DIFF-WRONG-{index}|mechanism_g1" for index in range(1, 4)]
        klados_diff = _mean_metric(klados, det, "clean_waveform_RRMSE") - _mean_metric(klados, match, "clean_waveform_RRMSE")
        sge_match = _mean_metric(sge, match, "eog_coherence_reduction")
        sge_pop = _mean_metric(sge, "POP", "eog_coherence_reduction")
        sge_diff_pop = _mean_metric(sge, f"{route}|DIFF-POP", "eog_coherence_reduction")
        wrong_value = float(np.mean([_mean_metric(sge, value, "eog_coherence_reduction") for value in wrongs]))
        sge_match_mechanism = _mean_metric(sge, match_mechanism, "eog_coherence_reduction")
        wrong_mechanism_value = float(np.mean([_mean_metric(sge, value, "eog_coherence_reduction") for value in wrongs_mechanism]))
        preservation = _mean_metric(sge, match, "nonartifact_observation_preservation")
        pop_preservation = _mean_metric(sge, "POP", "nonartifact_observation_preservation")
        gamma_points = []
        for gamma in _mapping(config, "evaluation")["gamma_sweep"]:
            method = f"{route}|DIFF-MATCH|gamma={float(gamma):g}"
            gamma_points.append((float(gamma), _mean_metric(sge, method, "eog_coherence_reduction"), _mean_metric(sge, method, "nonartifact_observation_preservation")))
        ordered = sorted((attenuation, preservation_value) for gamma, attenuation, preservation_value in gamma_points if gamma > 0 and math.isfinite(attenuation) and math.isfinite(preservation_value))
        pareto_auc = float(np.trapezoid([value[1] for value in ordered], [value[0] for value in ordered])) if len(ordered) > 1 else float("nan")
        by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in sge: by_unit[str(row["unit_id"])].append(row)
        equal_values = []
        for group in by_unit.values():
            pop_row = next((row for row in group if row["method"] == "POP"), None)
            gamma_rows = [row for row in group if str(row["method"]).startswith(f"{route}|DIFF-MATCH|gamma=") and not str(row["method"]).endswith("gamma=0")]
            if pop_row is None or len(gamma_rows) < 2: continue
            points = sorted((float(row["eog_coherence_reduction"]), float(row["nonartifact_observation_preservation"])) for row in gamma_rows)
            target = float(pop_row["eog_coherence_reduction"])
            if points[0][0] <= target <= points[-1][0]:
                equal_values.append(float(np.interp(target, [point[0] for point in points], [point[1] for point in points])) - float(pop_row["nonartifact_observation_preservation"]))
        equal_preservation = float(np.mean(equal_values)) if equal_values else float("nan")
        ranking_preservation = equal_preservation if math.isfinite(equal_preservation) else 0.0
        deployment_subject = sge_match - sge_diff_pop; deployment_specificity = sge_match - wrong_value
        mechanism_subject = sge_match_mechanism - sge_diff_pop; mechanism_specificity = sge_match_mechanism - wrong_mechanism_value
        raw_rrmse = _mean_metric(klados, "RAW", "clean_waveform_RRMSE")
        match_rrmse = _mean_metric(klados, match, "clean_waveform_RRMSE")
        match_delta_snr = _mean_metric(klados, match, "delta_SNR_db")
        output_scales = [float(row["output_input_RMS_ratio"]) for row in klados + sge if row.get("method") == match and _numeric(row.get("output_input_RMS_ratio"))]
        match_covariance = _mean_metric(sge, match, "reference_free_covariance_distortion")
        pop_covariance = _mean_metric(sge, "POP", "reference_free_covariance_distortion")
        jointly_dominated = bool(sge_match < sge_pop and preservation < pop_preservation and match_covariance > pop_covariance)
        absolute_valid = bool(
            output_scales
            and all(0.1 <= value <= 10.0 for value in output_scales)
            and match_rrmse < raw_rrmse
            and match_delta_snr > 0.0
            and not jointly_dominated
        )
        raw_score = .5 * (deployment_subject + mechanism_subject) + .5 * (deployment_specificity + mechanism_specificity) + klados_diff + .5 * ranking_preservation
        score = raw_score if absolute_valid else -1.0e9
        route_rows.append({"route": route, "absolute_validity": "passed" if absolute_valid else "failed", "klados_RAW_RRMSE": raw_rrmse, "klados_DIFF_MATCH_RRMSE": match_rrmse, "klados_DIFF_MATCH_delta_SNR_db": match_delta_snr, "output_scale_safe": bool(output_scales and all(0.1 <= value <= 10.0 for value in output_scales)), "sge_jointly_dominated_by_existing_POP": jointly_dominated, "sge_DIFF_MATCH_covariance_distortion": match_covariance, "sge_existing_POP_covariance_distortion": pop_covariance, "sge_existing_POP_eog_reduction": sge_pop, "sge_route_DIFF_POP_eog_reduction": sge_diff_pop, "klados_diff_minus_det_RRMSE_utility": klados_diff, "deployment_match_minus_DIFF_POP_eog_reduction": deployment_subject, "deployment_match_minus_three_wrong_eog_reduction": deployment_specificity, "mechanism_g1_match_minus_DIFF_POP_eog_reduction": mechanism_subject, "mechanism_g1_match_minus_three_wrong_eog_reduction": mechanism_specificity, "sge_match_minus_existing_POP_preservation": preservation - pop_preservation, "equal_attenuation_preservation_minus_existing_POP": equal_preservation, "equal_attenuation_comparable_units": len(equal_values), "attenuation_preservation_curve_AUC_nonzero_gamma": pareto_auc, "raw_ranking_score_before_absolute_gate": raw_score, "ranking_score": score})
    if len(candidates) > 1:
        old = json.loads((OLD_ROOT / "result_summary.json").read_text(encoding="utf-8"))
        route_rows.append({"route": "R0_projector", "klados_diff_minus_det_RRMSE_utility": float(old["effects"]["diffusion_value"]["mean"]), "sge_match_minus_POP_eog_reduction": float(old["effects"]["subject_population"]["mean"]), "sge_match_minus_three_wrong_eog_reduction": float(old["effects"]["subject_wrong"]["mean"]), "sge_match_minus_POP_preservation": float("nan"), "attenuation_preservation_curve_AUC": float("nan"), "ranking_score": float(old["effects"]["diffusion_value"]["mean"]) + float(old["effects"]["subject_population"]["mean"]) + float(old["effects"]["subject_wrong"]["mean"]), "historical_frozen_baseline": True})
    route_rows.sort(key=lambda row: float(row["ranking_score"]), reverse=True)
    top2 = [str(row["route"]) for row in route_rows if row["route"] != "R0_projector" and row.get("absolute_validity") == "passed"][:2]
    _write_csv(root / "route_screen.csv", route_rows)
    summary = {"status": "completed_carrier_ranking", **_implementation(), "top2": top2, "ranking": route_rows, "selection_role": "single_seed_full_real_development_route_screen_not_scientific_conclusion"}
    _atomic_json(root / "carrier_ranking.json", summary); _atomic_json(run_dir / "result_summary.json", summary); return summary


def _conditioning_tasks() -> list[dict[str, Any]]:
    tasks = []
    for conditioning in ("support_FiLM", "support_LoRA"):
        for route in _carrier_tasks():
            tasks.append({**route, "conditioning": conditioning, "task": len(tasks)})
    return tasks


def _train_condition_fold(config: Mapping[str, Any], prepared: Any, dataset: str, fold: int, conditioning: str, root: Path, device: torch.device, seed: int = 20260811) -> tuple[DeterministicArtifactEstimator, ArtifactLatentDiffusion, np.ndarray, np.ndarray, Path]:
    structured, structured_diffusion, mean, std, structured_checkpoint = _train_carrier_fold(config, prepared, dataset, fold, root, device, seed=seed)
    output = root / "conditioning_screen/checkpoints" / conditioning / dataset / f"fold_{fold:02d}" / f"seed_{seed}"
    checkpoint = output / "models.pt"
    model_config = _model_config(prepared)
    if conditioning == "support_LoRA":
        diffusion = SupportLoRAArtifactLatentDiffusion(model_config, _diffusion_config()).to(device)
        structured_payload = torch.load(structured_checkpoint, map_location=device, weights_only=False)
        missing, unexpected = diffusion.load_state_dict(structured_payload["population_diffusion_ema_best"], strict=False)
        if unexpected or set(missing) != {"output_adapter.down.weight", "output_adapter.up.weight"}:
            raise RuntimeError(f"population-to-adapter checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        diffusion.freeze_population_backbone(); diffusion.reset_support_adapter(); diffusion.eval()
        output.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.partial")
            torch.save({"population_backbone_checkpoint": str(structured_checkpoint), "support_adapter_fit_scope": "per_context_calibration_support_only", "coordinate_mean": mean, "coordinate_std": std}, temporary)
            os.replace(temporary, checkpoint)
        return structured, diffusion, mean, std, checkpoint
    diffusion = SupportFiLMArtifactLatentDiffusion(model_config, _diffusion_config()).to(device)
    diffusion.set_population_transfer(torch.as_tensor(prepared.population_context.full_transfer, device=device))
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        diffusion.load_state_dict(payload["diffusion_ema_best"]); diffusion.eval()
        return structured, diffusion, np.asarray(payload["coordinate_mean"]), np.asarray(payload["coordinate_std"]), checkpoint
    target, coordinate_mean, coordinate_std = _canonical_training(prepared)
    train_indices, validation_indices = _inner_split(prepared.training.recording_keys)
    rng = np.random.default_rng(seed + fold * 313 + (1 if conditioning == "support_LoRA" else 0))
    generator = torch.Generator(device=device).manual_seed(seed + fold * 2017)
    optimizer = AdamW(diffusion.parameters(), lr=2e-4, weight_decay=1e-4)
    ema = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}
    training = _mapping(_mapping(config, "routes"), "training")
    maximum, interval, batch_size = int(training["maximum_updates"]), int(training["validation_interval_updates"]), int(training["batch_size"])
    best = (float("inf"), None, 0); curve = []
    start_step = 1; resume_path = output / "resume.pt"
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        diffusion.load_state_dict(resume["diffusion"]); optimizer.load_state_dict(resume["optimizer"])
        ema = {key: value.to(device) for key, value in resume["ema"].items()}; best = resume["best"]
        rng.bit_generator.state = resume["numpy_rng_state"]; generator.set_state(resume["torch_generator_state"].cpu())
        curve = list(resume.get("curve", [])); start_step = int(resume["step"]) + 1
    for step in range(start_step, maximum + 1):
        chosen = rng.choice(train_indices, batch_size, replace=train_indices.size < batch_size)
        condition = _condition_from_indices(prepared, chosen, device, population_probability=.25, rng=rng, support_summary=True)
        truth = torch.as_tensor(target[chosen], device=device)
        timestep = torch.randint(0, 1000, (chosen.size,), device=device, generator=generator)
        noise = torch.randn(truth.shape, device=device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        matching_loss, detail = diffusion.training_loss(truth, timestep=timestep, noise=noise, generator=generator, **condition)
        wrong = dict(condition)
        for key in ("full_transfer", "normalized_transfer", "transfer_scale", "singular_values", "rank"):
            wrong[key] = condition[key].roll(1, 0)
        wrong_loss, _ = diffusion.training_loss(truth, timestep=timestep, noise=noise, generator=generator, **wrong)
        counterfactual = torch.relu(matching_loss - wrong_loss + .01)
        loss = matching_loss + .1 * counterfactual
        loss.backward(); torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0); optimizer.step(); _ema_update(ema, diffusion, .999)
        if step % interval == 0 or step == maximum:
            raw = {key: value.detach().clone() for key, value in diffusion.state_dict().items()}; diffusion.load_state_dict(ema)
            validation = _validation_loss(diffusion, prepared, target, validation_indices, device, diffusion=True)
            snapshot = _state_cpu(diffusion); diffusion.load_state_dict(raw)
            if validation < best[0]: best = (validation, snapshot, step)
            curve.append({"step": step, "matching_train": float(matching_loss.detach()), "wrong_train": float(wrong_loss.detach()), "counterfactual": float(counterfactual.detach()), "diffusion_EMA_validation": validation, "x0_mse": float(detail["x0_mse"])})
            output.mkdir(parents=True, exist_ok=True); temporary_resume = resume_path.with_name(f".{resume_path.name}.{os.getpid()}.partial")
            torch.save({"step": step, "diffusion": _state_cpu(diffusion), "optimizer": optimizer.state_dict(), "ema": {key: value.detach().cpu() for key, value in ema.items()}, "best": best, "numpy_rng_state": rng.bit_generator.state, "torch_generator_state": generator.get_state(), "curve": curve}, temporary_resume); os.replace(temporary_resume, resume_path)
    if best[1] is None: raise RuntimeError("conditioning screen selected no EMA checkpoint")
    output.mkdir(parents=True, exist_ok=True); temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.partial")
    torch.save({"diffusion_ema_best": best[1], "coordinate_mean": coordinate_mean, "coordinate_std": coordinate_std, "best_step": best[2], "conditioning": conditioning, "structured_checkpoint": str(structured_checkpoint), "EMA_used_for_validation_and_inference": True}, temporary); os.replace(temporary, checkpoint)
    _write_csv(output / "training_curve.csv", curve); diffusion.load_state_dict(best[1]); diffusion.eval()
    return structured, diffusion, coordinate_mean, coordinate_std, checkpoint


def _j5_conditioning_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    base, root = _load(config)
    ranking = json.loads((root / "carrier_ranking.json").read_text(encoding="utf-8")); top2 = tuple(ranking["top2"])
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _prepared
    device = torch.device("cuda", 0); completed = []; failures = []
    for task in _conditioning_tasks()[worker::8]:
        dataset, fold, conditioning = str(task["dataset"]), int(task["fold"]), str(task["conditioning"])
        output = root / "conditioning_screen/evaluation" / conditioning / dataset / f"fold_{fold:02d}"
        try:
            prepared = _prepared(base, dataset, fold)
            det, diffusion, mean, std, checkpoint = _train_condition_fold(config, prepared, dataset, fold, conditioning, root, device)
            rows = _evaluate_carrier_fold(config, base, root, prepared, dataset, fold, det, diffusion, mean, std, device, carriers=top2, conditioning=conditioning, archive_scope="conditioning_screen")
            _write_csv(output / "metrics.csv", rows); _atomic_json(output / "result_summary.json", {"status": "completed_conditioning_fold", "conditioning": conditioning, "dataset": dataset, "fold": fold, "checkpoint": str(checkpoint)})
            completed.append({"conditioning": conditioning, "dataset": dataset, "fold": fold})
        except Exception as error:
            failure = {"conditioning": conditioning, "dataset": dataset, "fold": fold, "error_type": type(error).__name__, "error": str(error)}
            failures.append(failure); _atomic_json(output / "failure.json", {"status": "technical_failure_route_continued", **failure})
    summary = {"status": "completed_conditioning_worker", **_implementation(), "worker": worker, "completed": completed, "failures": failures}
    _atomic_json(run_dir / "result_summary.json", summary); return summary


def _conditioning_selection(config: Mapping[str, Any], root: Path) -> list[dict[str, str]]:
    existing = root / "conditioning_ranking.json"
    if existing.is_file():
        return list(json.loads(existing.read_text(encoding="utf-8"))["top2"])
    carrier = json.loads((root / "carrier_ranking.json").read_text(encoding="utf-8")); routes = tuple(carrier["top2"])
    structured = _read_csvs(sorted((root / "carrier_screen/evaluation").glob("**/metrics.csv")))
    conditioned = _read_csvs(sorted((root / "conditioning_screen/evaluation").glob("**/metrics.csv")))
    candidates = []
    for route in routes:
        for conditioning in ("structured", "support_FiLM", "support_LoRA"):
            rows = structured if conditioning == "structured" else conditioned
            if conditioning != "structured":
                coverage = len(list((root / "conditioning_screen/evaluation" / conditioning).glob("**/metrics.csv")))
                if coverage != 26:
                    continue
            prefix = route if conditioning == "structured" else f"{route}|{conditioning}"
            klados = [row for row in rows if row.get("dataset") == "klados"]
            sge = [row for row in rows if row.get("dataset") == "sgeyesub"]
            match = f"{prefix}|DIFF-MATCH-K8|deployment"; det = f"{prefix}|DET-MATCH|deployment"
            wrongs = [f"{prefix}|DIFF-WRONG-{index}|deployment" for index in range(1, 4)]
            diff_value = _mean_metric(klados, det, "clean_waveform_RRMSE") - _mean_metric(klados, match, "clean_waveform_RRMSE")
            subject = _mean_metric(sge, match, "eog_coherence_reduction") - _mean_metric(sge, "POP", "eog_coherence_reduction")
            specificity = _mean_metric(sge, match, "eog_coherence_reduction") - float(np.mean([_mean_metric(sge, value, "eog_coherence_reduction") for value in wrongs]))
            preservation = _mean_metric(sge, match, "nonartifact_observation_preservation") - _mean_metric(sge, "POP", "nonartifact_observation_preservation")
            grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in sge: grouped[str(row["unit_id"])].append(row)
            equal_values = []
            for group in grouped.values():
                pop_row = next((row for row in group if row["method"] == "POP"), None)
                gamma_rows = [row for row in group if str(row["method"]).startswith(f"{prefix}|DIFF-MATCH|gamma=")]
                if pop_row is None or len(gamma_rows) < 2: continue
                points = sorted((float(row["eog_coherence_reduction"]), float(row["nonartifact_observation_preservation"])) for row in gamma_rows)
                target = float(pop_row["eog_coherence_reduction"])
                if points[0][0] <= target <= points[-1][0]:
                    equal_values.append(float(np.interp(target, [point[0] for point in points], [point[1] for point in points])) - float(pop_row["nonartifact_observation_preservation"]))
            equal_preservation = float(np.mean(equal_values)) if equal_values else float("nan")
            ranking_preservation = equal_preservation if math.isfinite(equal_preservation) else preservation
            candidates.append({"route": route, "conditioning": conditioning, "diffusion_value": diff_value, "subject_value": subject, "specificity": specificity, "preservation": preservation, "equal_attenuation_preservation": equal_preservation, "equal_attenuation_units": len(equal_values), "score": diff_value + subject + specificity + .5 * ranking_preservation})
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    summary = {"status": "completed_conditioning_ranking", "top2": [{"route": row["route"], "conditioning": row["conditioning"]} for row in candidates[:2]], "ranking": candidates}
    _atomic_json(root / "conditioning_ranking.json", summary)
    return list(summary["top2"])


def _final_tasks(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    top2 = _conditioning_selection(config, root)
    by_condition: dict[str, list[str]] = defaultdict(list)
    for item in top2: by_condition[str(item["conditioning"])].append(str(item["route"]))
    tasks = []
    for conditioning, carriers in sorted(by_condition.items()):
        for seed in tuple(int(value) for value in _mapping(config, "routes")["final_training_seeds"]):
            for route in _carrier_tasks():
                tasks.append({"task": len(tasks), "conditioning": conditioning, "carriers": tuple(carriers), "seed": seed, "dataset": route["dataset"], "fold": route["fold"]})
    return tasks


def _j6_final_worker(config: Mapping[str, Any], run_dir: Path, worker: int) -> Mapping[str, Any]:
    base, root = _load(config)
    from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _prepared
    device = torch.device("cuda", 0); completed = []
    for task in _final_tasks(config, root)[worker::8]:
        conditioning, seed = str(task["conditioning"]), int(task["seed"])
        dataset, fold, carriers = str(task["dataset"]), int(task["fold"]), tuple(task["carriers"])
        prepared = _prepared(base, dataset, fold)
        if conditioning == "structured":
            det, diffusion, mean, std, checkpoint = _train_carrier_fold(config, prepared, dataset, fold, root, device, seed=seed)
        else:
            det, diffusion, mean, std, checkpoint = _train_condition_fold(config, prepared, dataset, fold, conditioning, root, device, seed=seed)
        rows = _evaluate_carrier_fold(config, base, root, prepared, dataset, fold, det, diffusion, mean, std, device, carriers=carriers, conditioning=conditioning, training_seed=seed, archive_scope="final_top2")
        output = root / "final_top2/evaluation" / conditioning / dataset / f"fold_{fold:02d}" / f"seed_{seed}"
        _write_csv(output / "metrics.csv", rows); _atomic_json(output / "result_summary.json", {"status": "completed_final_top2_fold", "conditioning": conditioning, "carriers": list(carriers), "dataset": dataset, "fold": fold, "seed": seed, "checkpoint": str(checkpoint)})
        completed.append({"conditioning": conditioning, "dataset": dataset, "fold": fold, "seed": seed})
    summary = {"status": "completed_final_top2_worker", **_implementation(), "worker": worker, "completed": completed}
    _atomic_json(run_dir / "result_summary.json", summary); return summary


def _effects_by_seed(rows: Sequence[Mapping[str, Any]], left: str, right: str, metric: str, direction: float, label: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["unit_id"]), str(int(float(row["training_seed"]))))][str(row["method"])] = row
    output = []
    for (unit, seed), methods in sorted(grouped.items()):
        if left in methods and right in methods and _numeric(methods[left].get(metric)) and _numeric(methods[right].get(metric)):
            output.append({"estimand": label, "unit_id": unit, "training_seed": int(seed), "exact_cell": methods[left].get("exact_cell", ""), "effect": direction * (float(methods[left][metric]) - float(methods[right][metric]))})
    return output


def _hierarchical_bootstrap(rows: Sequence[Mapping[str, Any]], seed: int, replicates: int, stratified: bool) -> dict[str, Any]:
    if not rows:
        return {"status": "inconclusive_no_comparable_units", "mean": float("nan"), "median": float("nan"), "ci95": [float("nan"), float("nan")], "units": 0, "training_seeds": 0}
    by_unit: dict[str, list[float]] = defaultdict(list); cells = {}
    for row in rows:
        by_unit[str(row["unit_id"])].append(float(row["effect"])); cells[str(row["unit_id"])] = str(row.get("exact_cell", ""))
    rng = np.random.default_rng(seed); values = {unit: float(np.mean(parts)) for unit, parts in by_unit.items()}
    strata: dict[str, list[str]] = defaultdict(list)
    for unit in by_unit: strata[cells[unit] if stratified else "all"].append(unit)
    samples = np.empty(replicates)
    for index in range(replicates):
        draw = []
        for units in strata.values():
            for unit in rng.choice(units, len(units), replace=True):
                draw.append(float(rng.choice(by_unit[str(unit)])))
        samples[index] = np.mean(draw)
    unit_values = np.asarray(list(values.values()))
    return {"mean": float(unit_values.mean()), "median": float(np.median(unit_values)), "ci95": [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], "units": len(by_unit), "training_seeds": len(set(int(row["training_seed"]) for row in rows))}


def _final_uncertainty(root: Path, selection: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    results = {}
    records = [f"sim{value:02d}" for value in range(37, 55) if value not in (44, 45)]
    for item in selection:
        route, conditioning = str(item["route"]), str(item["conditioning"])
        prefix = route if conditioning == "structured" else f"{route}|{conditioning}"
        source_rows = []
        for record in records:
            samples_by_seed, det_by_seed, clean = [], [], None
            for seed in SEEDS:
                path = root / "final_top2/server_arrays" / conditioning / "klados/fold_00" / f"seed_{seed}" / f"{record}.npz"
                if not path.is_file(): continue
                with np.load(path) as value:
                    sample_key = f"server_only__{prefix}|DIFF_MATCH_K8|deployment_samples"
                    det_key = f"{prefix}|DET_MATCH|deployment"
                    samples_by_seed.append(np.asarray(value[sample_key], dtype=np.float64))
                    det_by_seed.append(np.asarray(value[det_key], dtype=np.float64))
                    clean = np.asarray(value["server_only__clean_target"], dtype=np.float64)
            if len(samples_by_seed) != 3 or clean is None: continue
            samples = np.stack(samples_by_seed); det_stack = np.stack(det_by_seed)
            seed_mean = samples.mean(1); output = seed_mean.mean(0)
            uncertainty = np.sqrt(samples.var(1).mean(0) + seed_mean.var(0))
            det_uncertainty = det_stack.std(0)
            error = np.sqrt(np.mean(np.square(output - clean), axis=(1, 2)))
            det_error = np.sqrt(np.mean(np.square(det_stack.mean(0) - clean), axis=(1, 2)))
            score = uncertainty.mean(axis=(1, 2)); det_score = det_uncertainty.mean(axis=(1, 2))
            source_rows.append({"source_record": record, "diff_pearson": _correlation(score, error), "diff_spearman": _correlation(score, error, rank=True), "det_pearson": _correlation(det_score, det_error), "det_spearman": _correlation(det_score, det_error, rank=True), **{f"diff_{key}": value for key, value in _risk_auc(error, score).items()}, **{f"det_{key}": value for key, value in _risk_auc(det_error, det_score).items()}})
        if source_rows:
            results[f"{route}|{conditioning}"] = {"status": "secondary_exploratory", "source_records": len(source_rows), "diff_centered_pearson": _bootstrap_sources(source_rows, "diff_pearson", 91, 20000), "det_centered_pearson": _bootstrap_sources(source_rows, "det_pearson", 92, 20000), "diff_normalized_risk_AUC": _bootstrap_sources(source_rows, "diff_normalized_risk_coverage_auc", 93, 20000), "det_normalized_risk_AUC": _bootstrap_sources(source_rows, "det_normalized_risk_coverage_auc", 94, 20000)}
    return results


def _j7_aggregate(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _, root = _load(config); selection = _conditioning_selection(config, root)
    all_effects = []; summaries = {}; method_rows = []; pareto_rows = []
    for combo_index, item in enumerate(selection):
        route, conditioning = str(item["route"]), str(item["conditioning"])
        paths = sorted((root / "final_top2/evaluation" / conditioning).glob("**/metrics.csv"))
        expected = 78
        if len(paths) != expected: raise RuntimeError(f"final {conditioning} coverage incomplete: {len(paths)}/{expected}")
        rows = _read_csvs(paths); prefix = route if conditioning == "structured" else f"{route}|{conditioning}"
        match = f"{prefix}|DIFF-MATCH-K8|deployment"; det = f"{prefix}|DET-MATCH|deployment"
        klados = [row for row in rows if row.get("dataset") == "klados"]
        sge = [row for row in rows if row.get("dataset") == "sgeyesub"]
        effects = {
            "diffusion_vs_deterministic": _effects_by_seed(klados, match, det, "clean_waveform_RRMSE", -1.0, "diffusion_vs_deterministic"),
            "match_vs_population": _effects_by_seed(sge, match, "POP", "eog_coherence_reduction", 1.0, "match_vs_population"),
            "mechanism_g1_match_vs_population": _effects_by_seed(sge, f"{prefix}|DIFF-MATCH-K8|mechanism_g1", "POP", "eog_coherence_reduction", 1.0, "mechanism_g1_match_vs_population"),
        }
        wrong_parts = [_effects_by_seed(sge, match, f"{prefix}|DIFF-WRONG-{index}|deployment", "eog_coherence_reduction", 1.0, f"match_vs_wrong_{index}") for index in range(1, 4)]
        wrong_lookup: dict[tuple[str, int], list[float]] = defaultdict(list)
        for part in wrong_parts:
            for row in part: wrong_lookup[(str(row["unit_id"]), int(row["training_seed"]))].append(float(row["effect"]))
        effects["match_vs_three_wrong"] = [{"estimand": "match_vs_three_wrong", "unit_id": key[0], "training_seed": key[1], "exact_cell": next(row["exact_cell"] for part in wrong_parts for row in part if row["unit_id"] == key[0] and row["training_seed"] == key[1]), "effect": float(np.mean(value))} for key, value in wrong_lookup.items()]
        mechanism_wrong_parts = [_effects_by_seed(sge, f"{prefix}|DIFF-MATCH-K8|mechanism_g1", f"{prefix}|DIFF-WRONG-{index}|mechanism_g1", "eog_coherence_reduction", 1.0, f"mechanism_g1_match_vs_wrong_{index}") for index in range(1, 4)]
        mechanism_wrong_lookup: dict[tuple[str, int], list[float]] = defaultdict(list)
        for part in mechanism_wrong_parts:
            for row in part: mechanism_wrong_lookup[(str(row["unit_id"]), int(row["training_seed"]))].append(float(row["effect"]))
        effects["mechanism_g1_match_vs_three_wrong"] = [{"estimand": "mechanism_g1_match_vs_three_wrong", "unit_id": key[0], "training_seed": key[1], "exact_cell": next(row["exact_cell"] for part in mechanism_wrong_parts for row in part if row["unit_id"] == key[0] and row["training_seed"] == key[1]), "effect": float(np.mean(value))} for key, value in mechanism_wrong_lookup.items()]
        combo_key = f"{route}|{conditioning}"
        summaries[combo_key] = {name: _hierarchical_bootstrap(value, 20260900 + combo_index * 10 + index, 20000, name != "diffusion_vs_deterministic") for index, (name, value) in enumerate(effects.items())}
        for name, values in effects.items(): all_effects.extend([{**row, "route": route, "conditioning": conditioning} for row in values])
        # Equal-attenuation preservation is evaluated within unit and seed.
        grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in sge: grouped[(str(row["unit_id"]), int(float(row["training_seed"])))].append(row)
        for (unit, seed), group in grouped.items():
            pop = next((row for row in group if row["method"] == "POP"), None)
            gamma = [row for row in group if str(row["method"]).startswith(f"{prefix}|DIFF-MATCH|gamma=")]
            if pop is None or len(gamma) < 2: continue
            points = sorted((float(row["eog_coherence_reduction"]), float(row["nonartifact_observation_preservation"])) for row in gamma)
            target = float(pop["eog_coherence_reduction"])
            if not points[0][0] <= target <= points[-1][0]:
                continue
            equal = float(np.interp(target, [point[0] for point in points], [point[1] for point in points]))
            auc = float(np.trapezoid([point[1] for point in points], [point[0] for point in points]))
            pareto_rows.append({"route": route, "conditioning": conditioning, "unit_id": unit, "training_seed": seed, "equal_attenuation_preservation_minus_POP": equal - float(pop["nonartifact_observation_preservation"]), "pareto_AUC": auc})
        # Compact per-method summaries.
        for dataset in ("klados", "sgeyesub"):
            for method in (
                "RAW", "POP", f"{prefix}|DET-POP", det,
                f"{prefix}|DIFF-POP", f"{prefix}|NO-SUPPORT",
                f"{prefix}|RHO-ONLY-POP-BASIS", f"{prefix}|CONSTANT-RHO-0.5",
                f"{prefix}|DIFF-MATCH-K1|deployment", match,
                *[f"{prefix}|DIFF-WRONG-{index}|deployment" for index in range(1,4)],
            ):
                selected = [row for row in rows if row["method"] == method and row["dataset"] == dataset]
                if not selected:
                    continue
                method_rows.append({"route": route, "conditioning": conditioning, "dataset": dataset, "method": method, **{metric: float(np.mean([float(row[metric]) for row in selected if _numeric(row.get(metric))])) for metric in ("clean_waveform_RRMSE", "eog_coherence_reduction", "nonartifact_observation_preservation", "condition_erp_observation_relative_preservation", "continuous_segment_welch_psd_distortion", "reference_free_covariance_distortion") if any(_numeric(row.get(metric)) for row in selected)}})
    _write_csv(root / "top2_unit_metrics.csv", _read_csvs(sorted((root / "final_top2/evaluation").glob("**/metrics.csv"))))
    _write_csv(root / "paired_effects.csv", all_effects); _write_csv(root / "method_summary.csv", method_rows); _write_csv(root / "equal_attenuation_pareto.csv", pareto_rows)
    uncertainty = _final_uncertainty(root, selection); _atomic_json(root / "corrected_uncertainty.json", {"status": "secondary_exploratory", "routes": uncertainty})
    summary = {"status": "completed_wide_v2_aggregation", **_implementation(), "selected_top2": selection, "effects": summaries, "equal_attenuation": {f"{item['route']}|{item['conditioning']}": _hierarchical_bootstrap([{**row, "effect": row["equal_attenuation_preservation_minus_POP"], "exact_cell": ""} for row in pareto_rows if row["route"] == item["route"] and row["conditioning"] == item["conditioning"]], 44, 20000, False) for item in selection}, "uncertainty": uncertainty, "coverage": {"klados_source_records": 16, "sge_successful_stems": 58, "sge_total_denominator": 59}, "scientific_role": "development_wide_exploration_not_confirmation"}
    _atomic_json(root / "result_summary.json", summary); _atomic_json(run_dir / "result_summary.json", summary)
    _wide_figures(root, summary)
    return summary


def _wide_figures(root: Path, summary: Mapping[str, Any]) -> None:
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    operator = _read_csvs([root / "operator_audit.csv"]); sge_operator = [row for row in operator if row.get("dataset") == "sgeyesub"]
    figure, axis = plt.subplots(figsize=(6, 4)); axis.scatter([float(row["within_subject_half_distance"]) for row in sge_operator], [float(row["full_C_frobenius_distance"]) for row in sge_operator], alpha=.65, s=18); axis.plot([0, max(float(row["within_subject_half_distance"]) for row in sge_operator)], [0, max(float(row["within_subject_half_distance"]) for row in sge_operator)], "k--", lw=1); axis.set(xlabel="Within-subject support-half C distance", ylabel="Subject-to-population C distance"); figure.tight_layout(); figure.savefig(figures / "within_vs_between_operator_distance.png", dpi=180); plt.close(figure)
    crossfit = _read_csvs([root / "support_crossfit.csv"]); labels = ["Projector/full-C", "Population", "FIR", "State"] ; values = [np.median([float(row[key]) for row in crossfit if _numeric(row.get(key))]) for key in ("matching_full_C_error", "population_error", "FIR_support_error", "state_specific_error")]; figure, axis = plt.subplots(figsize=(6,4)); axis.bar(labels, values); axis.set_ylabel("Support cross-fit prediction error (lower better)"); axis.tick_params(axis="x", rotation=20); figure.tight_layout(); figure.savefig(figures / "operator_carrier_crossfit.png", dpi=180); plt.close(figure)
    effects = summary["effects"]; labels=[]; means=[]; lows=[]; highs=[]
    for combo, values_map in effects.items():
        for name in ("match_vs_population", "match_vs_three_wrong", "diffusion_vs_deterministic"):
            value=values_map[name]; labels.append(f"{combo}\n{name}"); means.append(value["mean"]); lows.append(value["mean"]-value["ci95"][0]); highs.append(value["ci95"][1]-value["mean"])
    figure, axis=plt.subplots(figsize=(9,5)); axis.errorbar(means, range(len(means)), xerr=[lows,highs], fmt="o"); axis.axvline(0,color="black",lw=1); axis.set_yticks(range(len(labels)), labels, fontsize=7); axis.set_xlabel("Paired utility (positive better)"); figure.tight_layout(); figure.savefig(figures / "route_subject_and_diffusion_effects.png",dpi=180); plt.close(figure)
    carrier_rows = _read_csvs(sorted((root / "carrier_screen/evaluation").glob("**/metrics.csv")))
    carrier_effects = []
    for route in ("R1_full_C_residual", "R2_FIR_residual", "R3_state_gated_residual"):
        sge = [row for row in carrier_rows if row.get("dataset") == "sgeyesub"]
        match = f"{route}|DIFF-MATCH-K8|deployment"
        pop_effect = _effects_by_seed(sge, match, "POP", "eog_coherence_reduction", 1.0, "MATCH-POP")
        wrong = [_effects_by_seed(sge, match, f"{route}|DIFF-WRONG-{index}|deployment", "eog_coherence_reduction", 1.0, "MATCH-WRONG") for index in range(1,4)]
        wrong_lookup: dict[str, list[float]] = defaultdict(list)
        for part in wrong:
            for row in part: wrong_lookup[str(row["unit_id"])].append(float(row["effect"]))
        carrier_effects.extend([(route, "MATCH-POP", float(row["effect"])) for row in pop_effect])
        carrier_effects.extend([(route, "MATCH-3WRONG", float(np.mean(value))) for _, value in wrong_lookup.items()])
    figure, axis = plt.subplots(figsize=(8,4)); labels=[]; positions=[]
    for position, (route, contrast) in enumerate((route, contrast) for route in ("R1_full_C_residual", "R2_FIR_residual", "R3_state_gated_residual") for contrast in ("MATCH-POP", "MATCH-3WRONG")):
        values = np.asarray([value for candidate, kind, value in carrier_effects if candidate == route and kind == contrast])
        labels.append(f"{route}\n{contrast}"); positions.append(position)
        axis.scatter(np.full(values.size, position), values, alpha=.22, s=9, color="tab:blue")
        axis.scatter([position], [values.mean()], color="black", s=28, zorder=3)
    axis.axhline(0,color="black",lw=1); axis.set_xticks(positions,labels,rotation=20,ha="right",fontsize=7); axis.set_ylabel("Single-seed participant/stem paired utility"); figure.tight_layout(); figure.savefig(figures / "all_carrier_match_population_wrong_effects.png",dpi=180); plt.close(figure)
    pareto=_read_csvs([root / "equal_attenuation_pareto.csv"]); figure,axis=plt.subplots(figsize=(6,4));
    for combo in sorted({f"{row['route']}|{row['conditioning']}" for row in pareto}):
        chosen=[row for row in pareto if f"{row['route']}|{row['conditioning']}"==combo]; axis.scatter([float(row["pareto_AUC"]) for row in chosen],[float(row["equal_attenuation_preservation_minus_POP"]) for row in chosen],s=14,alpha=.45,label=combo)
    axis.axhline(0,color="black",lw=1); axis.set(xlabel="Attenuation–preservation curve AUC",ylabel="Equal-attenuation preservation minus POP"); axis.legend(fontsize=6); figure.tight_layout(); figure.savefig(figures / "attenuation_preservation_pareto.png",dpi=180); plt.close(figure)
    paired=_read_csvs([root / "paired_effects.csv"]); klados=[row for row in paired if row["estimand"]=="diffusion_vs_deterministic"]; figure,axis=plt.subplots(figsize=(7,4));
    for combo in sorted({f"{row['route']}|{row['conditioning']}" for row in klados}):
        chosen=[row for row in klados if f"{row['route']}|{row['conditioning']}"==combo]; by=defaultdict(list)
        for row in chosen: by[str(row["unit_id"])].append(float(row["effect"])); axis.plot(range(len(by)),[np.mean(value) for _,value in sorted(by.items())],marker="o",label=combo)
    axis.axhline(0,color="black",lw=1); axis.set(xlabel="Klados source record",ylabel="DIFF−DET clean-RRMSE utility"); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(figures / "diffusion_vs_deterministic_klados.png",dpi=180); plt.close(figure)
    units=_read_csvs([root / "top2_unit_metrics.csv"]); figure,axis=plt.subplots(figsize=(6,4))
    for item in summary["selected_top2"]:
        route,conditioning=item["route"],item["conditioning"]; prefix=route if conditioning=="structured" else f"{route}|{conditioning}"; by=defaultdict(dict)
        for row in units:
            if row.get("dataset")=="klados": by[(str(row["unit_id"]),int(float(row["training_seed"])))][str(row["method"])]=row
        raw=[]; gain=[]
        for methods in by.values():
            if "RAW" in methods and f"{prefix}|DIFF-MATCH-K8|deployment" in methods and f"{prefix}|DET-MATCH|deployment" in methods:
                raw.append(float(methods["RAW"]["clean_waveform_RRMSE"])); gain.append(float(methods[f"{prefix}|DET-MATCH|deployment"]["clean_waveform_RRMSE"])-float(methods[f"{prefix}|DIFF-MATCH-K8|deployment"]["clean_waveform_RRMSE"]))
        axis.scatter(raw,gain,s=15,alpha=.5,label=f"{route}|{conditioning}")
    axis.axhline(0,color="black",lw=1); axis.set(xlabel="RAW clean RRMSE (artifact severity)",ylabel="DIFF−DET gain"); axis.legend(fontsize=7); figure.tight_layout(); figure.savefig(figures / "artifact_severity_vs_diffusion_gain.png",dpi=180); plt.close(figure)
    uncertainty=summary.get("uncertainty",{}); figure,axis=plt.subplots(figsize=(7,4)); labels=[]; values=[]
    for combo,value in uncertainty.items(): labels.extend([f"{combo}\nDiff",f"{combo}\nDet"]); values.extend([value["diff_normalized_risk_AUC"]["mean"],value["det_normalized_risk_AUC"]["mean"]])
    axis.bar(labels,values); axis.set_ylabel("Normalized risk–coverage AUC (lower better)"); axis.tick_params(axis="x",labelsize=6); figure.tight_layout(); figure.savefig(figures / "corrected_uncertainty_risk_coverage.png",dpi=180); plt.close(figure)


def _j8_finalize(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _, root = _load(config); summary=json.loads((root/"result_summary.json").read_text(encoding="utf-8")); audit=json.loads((root/"operator_audit_summary.json").read_text(encoding="utf-8")); reaudit=json.loads((root/"current_result_reaudit/result_summary.json").read_text(encoding="utf-8")); carrier=json.loads((root/"carrier_ranking.json").read_text(encoding="utf-8")); conditioning=json.loads((root/"conditioning_ranking.json").read_text(encoding="utf-8"))
    lines=["# Subject-aware diffusion wide exploration v2","","This is a development wide screen, not confirmation or a frozen submission protocol.","","## Corrected current-result audit","",f"The old uncertainty verdict is invalid under its original definition. Comparable final-EEG replay gives diffusion normalized risk–coverage AUC {reaudit['comparable_final_EEG_uncertainty']['source_level']['diff_normalized_risk_coverage_auc']['mean']:.4f} versus deterministic {reaudit['comparable_final_EEG_uncertainty']['source_level']['det_normalized_risk_coverage_auc']['mean']:.4f}; uncertainty is secondary and provides no current advantage. True end-to-end K=1 and DET-POP are reported in the re-audit directory.","","## Data suitability","",f"SGE full-C support cross-fit beat population in {audit['matching_full_C_beats_population_count']}/58 stems and the mean of three same-cell wrong donors in {audit['matching_full_C_beats_mean_three_wrong_count']}/58. Median errors were {audit['representation_median_support_errors']}. Post-output-freeze support-to-query evidence was {audit.get('support_to_later_query', {})}; legacy-rho diagnostics were {audit.get('legacy_rho_audit', {})}.","","## Complete carrier ranking","","| Rank | Carrier | Score | DIFF−DET | Deployment MATCH−POP | Deployment MATCH−WRONG | Equal-attenuation preservation |","|---:|---|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(carrier["ranking"], start=1):
        lines.append(f"| {rank} | {row['route']} | {float(row['ranking_score']):.5f} | {float(row.get('klados_diff_minus_det_RRMSE_utility', float('nan'))):.5f} | {float(row.get('deployment_match_minus_POP_eog_reduction', row.get('sge_match_minus_POP_eog_reduction', float('nan')))):.5f} | {float(row.get('deployment_match_minus_three_wrong_eog_reduction', row.get('sge_match_minus_three_wrong_eog_reduction', float('nan')))):.5f} | {float(row.get('equal_attenuation_preservation_minus_POP', float('nan'))):.5f} |")
    lines.extend(["","Conditioning variants were ranked only after complete 26-fold coverage: " + str(conditioning["ranking"]),"","## Final two routes","","| Route | Deployment MATCH−POP | Deployment MATCH−3 WRONG | Mechanism g=1 MATCH−POP | Mechanism g=1 MATCH−3 WRONG | DIFF−DET | Equal-attenuation preservation |","|---|---:|---:|---:|---:|---:|---:|"])
    for item in summary["selected_top2"]:
        key=f"{item['route']}|{item['conditioning']}"; effect=summary["effects"][key]; equal=summary["equal_attenuation"][key]; lines.append(f"| {key} | {effect['match_vs_population']['mean']:.5f} {effect['match_vs_population']['ci95']} | {effect['match_vs_three_wrong']['mean']:.5f} {effect['match_vs_three_wrong']['ci95']} | {effect['mechanism_g1_match_vs_population']['mean']:.5f} {effect['mechanism_g1_match_vs_population']['ci95']} | {effect['mechanism_g1_match_vs_three_wrong']['mean']:.5f} {effect['mechanism_g1_match_vs_three_wrong']['ci95']} | {effect['diffusion_vs_deterministic']['mean']:.5f} {effect['diffusion_vs_deterministic']['ci95']} | {equal['mean']:.5f} {equal['ci95']} |")
    lines.extend(["","The ranking separates diffusion incremental value, carrier value, support-only reliability, and equal-attenuation natural-EEG trade-offs. The corrected uncertainty remains secondary. Klados remains 16-source-record mechanism evidence; SGE is 58/59-stem development evidence. R0 is a frozen historical baseline and not a full-C representation. No Eye-BCI confirmation outcome, EEGEyeNet retry, TAAS edit, or family-wide claim was made."])
    report=CODE_ROOT/str(_mapping(config,"outputs")["report"]); report.parent.mkdir(parents=True,exist_ok=True); report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    result={"status":"completed_wide_v2_finalize",**_implementation(),"report":str(report),"result_root":str(root),"selected_top2":summary["selected_top2"]}; _atomic_json(root/"terminal_manifest.json",result); _atomic_json(run_dir/"result_summary.json",result); return result


def run_stage(config: Mapping[str, Any], run_dir: str | Path, stage: str, task_index: int | None) -> Mapping[str, Any]:
    run = Path(run_dir); run.mkdir(parents=True, exist_ok=True)
    if stage == "j0-reaudit":
        if task_index is not None: raise ValueError("J0 rejects arrays")
        return _j0_reaudit(config, run)
    if stage == "j1-operator-audit":
        if task_index is not None: raise ValueError("J1 rejects arrays")
        return _j1_operator_audit(config, run)
    if stage == "j0-replay":
        if task_index is not None: raise ValueError("J0 replay rejects arrays")
        return _j0_replay(config, run)
    if stage == "j2-technical":
        if task_index is not None: raise ValueError("J2 rejects arrays")
        return _j2_technical(config, run)
    if stage == "j3-carrier-worker":
        if task_index is None or not 0 <= task_index < 8: raise ValueError("J3 requires worker 0-7")
        return _j3_carrier_worker(config, run, task_index)
    if stage == "j4-rank":
        if task_index is not None: raise ValueError("J4 rejects arrays")
        return _j4_rank(config, run)
    if stage == "j5-conditioning-worker":
        if task_index is None or not 0 <= task_index < 8: raise ValueError("J5 requires worker 0-7")
        return _j5_conditioning_worker(config, run, task_index)
    if stage == "j6-final-worker":
        if task_index is None or not 0 <= task_index < 8: raise ValueError("J6 requires worker 0-7")
        return _j6_final_worker(config, run, task_index)
    if stage == "j7-aggregate":
        if task_index is not None: raise ValueError("J7 rejects arrays")
        return _j7_aggregate(config, run)
    if stage == "j8-finalize":
        if task_index is not None: raise ValueError("J8 rejects arrays")
        return _j8_finalize(config, run)
    raise ValueError(f"unsupported wide exploration stage: {stage}")


__all__ = ["run_stage"]
