"""Fold-local EB audit and population artifact-subspace validity for SGE v8.

The runner deliberately keeps the two development questions independent.  The
operator audit re-fits every quantity inside its outer fold.  The GPU route
uses the bounded rank-two artifact-subspace model rather than the invalidated
v6 dynamic-transfer backbone.
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
from scipy.signal import coherence
from torch.optim import AdamW

from eeg_cgdr.data.sgeyesub import load_sgeyesub_signal_record
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import (
    _metadata, _record_pairs, _support_eog_stats, _support_transfer,
    _natural_metrics, apply_dynamic_transfer, fit_dynamic_transfer,
)
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig, ArtifactSubspaceDiffusion,
    DeterministicSubspaceEstimator, aligned_artifact_basis,
    reconstruct_from_subspace, training_tau,
)
from eeg_cgdr.models.artifact_subspace_score_lora import inject_score_lora, lora_state_dict


PROTOCOL = "SGE-SCORE-LORA-v8"


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL or int(value.get("harness_level", -1)) != 1:
        raise ValueError("wrong v8 protocol or harness")
    return value


def _folds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return json.loads((Path(str(config["v6_root"])) / "frozen_grouped_folds.json").read_text(encoding="utf-8"))["folds"]


def _frequency_response(transfer: np.ndarray, nfft: int = 256) -> np.ndarray:
    response = np.fft.rfft(np.asarray(transfer, np.float64), n=nfft, axis=-1)
    scale = np.sqrt(np.mean(np.abs(response) ** 2)).clip(1e-12)
    return response / scale


def _operator_distances(candidate: np.ndarray, reference: np.ndarray, common_eog: np.ndarray) -> dict[str, float]:
    left = np.asarray(candidate, np.float64); right = np.asarray(reference, np.float64)
    left_normal = left / max(np.linalg.norm(left), 1e-12)
    right_normal = right / max(np.linalg.norm(right), 1e-12)
    left_response = _frequency_response(left); right_response = _frequency_response(right)
    phase = np.angle(np.exp(1j * (np.angle(left_response) - np.angle(right_response))))
    common_left = apply_dynamic_transfer(left, common_eog)
    common_right = apply_dynamic_transfer(right, common_eog)
    return {
        "normalized_frobenius": float(np.linalg.norm(left_normal - right_normal)),
        "frequency_magnitude": float(np.mean(np.abs(np.abs(left_response) - np.abs(right_response)))),
        "frequency_phase": float(np.mean(np.abs(phase))),
        "common_eog_response": float(np.linalg.norm(common_left - common_right) / max(np.linalg.norm(common_right), 1e-12)),
    }


def _distance_utility(candidate: np.ndarray, reference: np.ndarray, common_eog: np.ndarray) -> float:
    value = _operator_distances(candidate, reference, common_eog)
    return value["frequency_magnitude"] + 0.1 * value["frequency_phase"] + value["common_eog_response"]


def _query_generator(loaded: Any, record: Any, normal_mean: np.ndarray, normal_std: np.ndarray,
                     eog_mean: np.ndarray, eog_std: np.ndarray, taps: int, ridge: float) -> tuple[np.ndarray, dict[str, Any]]:
    # Reuse only the frozen, model-blind trial-role assignment.  Refit H_G below
    # using this fold/budget's exact EEG and EOG normalization.
    values = _record_pairs(loaded, record, normal_mean, normal_std, taps=taps, ridge=ridge,
                           window_seconds=2.0, return_role_metadata=True)
    metadata = values[-1]
    query_eeg = (np.asarray(loaded.query.eeg, np.float64) - normal_mean) / normal_std
    query_eog = (np.asarray(loaded.query_annotations.external_eog, np.float64) - eog_mean) / eog_std
    trial = int(record.samples_per_trial)
    indices = metadata["generator_trial_ids"]
    eeg = np.concatenate([query_eeg[:, i * trial:(i + 1) * trial] for i in indices], axis=1)
    eog = np.concatenate([query_eog[:, i * trial:(i + 1) * trial] for i in indices], axis=1)
    return fit_dynamic_transfer(eeg, eog, taps=taps, ridge=ridge), metadata


def _support_features(loaded: Any, support: np.ndarray, population: np.ndarray, reliability: float,
                      budget: float, samples: int) -> np.ndarray:
    flat = support.reshape(support.shape[0], -1)
    singular = np.linalg.svd(flat, compute_uv=False)[:4]
    singular = singular / max(np.linalg.norm(singular), 1e-12)
    singular = np.pad(singular, (0, 4 - singular.size))
    eog = np.asarray(loaded.support.external_eog[:, :samples], np.float64)
    eeg = np.asarray(loaded.support.eeg[:, :samples], np.float64)
    cov = float(np.linalg.norm(eeg @ eog.T / max(samples, 1)) / max(np.linalg.norm(eeg), 1e-12))
    condition = float(singular[0] / max(singular[-1], 1e-6))
    delta = float(np.linalg.norm(support - population) / max(np.linalg.norm(population), 1e-12))
    return np.asarray([reliability, math.log1p(budget), delta, math.log1p(condition), math.log1p(cov), *singular], np.float64)


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> float:
    mean = train_x.mean(0); std = train_x.std(0).clip(1e-6)
    design = np.column_stack((np.ones(len(train_x)), (train_x - mean) / std))
    coef = np.linalg.solve(design.T @ design + 0.05 * np.eye(design.shape[1]), design.T @ train_y)
    return float(np.clip(np.r_[1.0, (test_x - mean) / std] @ coef, 0, 1))


def stage_eb_headroom(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Re-fit Hs/Hpop/Hgen/wrongs inside every fold for every support budget."""
    root = Path(str(config["eb_result_root"])); layouts, records = _metadata(config)
    data_root = Path(str(config["data_root"])); folds = _folds(config)
    rows: list[dict[str, Any]] = []; grid = np.linspace(0.0, 1.0, 101)
    for budget in map(float, config["support_budgets_seconds"]):
        for fold_index, fold in enumerate(folds):
            rate = float(fold["sampling_rate_hz"]); samples = int(round(budget * rate))
            taps = 2 * int(round(float(config["fir_lag_ms"]) * rate / 1000.0)) + 1
            keys = list(fold["training"]) + list(fold["heldout"])
            loaded = {key: load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=True, include_query_annotations=True) for key in keys}
            eligible = {key: loaded[key].support.eeg.shape[1] >= samples + int(round(float(config["guard_seconds"]) * rate)) for key in keys}
            train_eligible = [key for key in fold["training"] if eligible[key]]
            if not train_eligible:
                rows.extend({"budget_seconds": budget, "fold_id": fold["fold_id"], "fold_cluster": fold_index,
                             "study": fold["study"], "recording_key": key, "eligible": False,
                             "calibration_samples": 0, "status": "ineligible_no_outer_training_support_at_budget"}
                            for key in fold["heldout"])
                continue
            train_eeg = np.concatenate([loaded[key].support.eeg[:, :samples] for key in train_eligible], axis=1).astype(np.float64)
            normal_mean = train_eeg.mean(1, keepdims=True); normal_std = train_eeg.std(1, keepdims=True).clip(1e-6)
            support: dict[str, np.ndarray] = {}; generator: dict[str, np.ndarray] = {}; reliability: dict[str, float] = {}; features: dict[str, np.ndarray] = {}
            for key in keys:
                if not eligible[key]:
                    continue
                eog_mean, eog_std = _support_eog_stats(loaded[key], samples)
                support[key], reliability[key] = _support_transfer(loaded[key], eog_mean, eog_std, normal_mean, normal_std, samples, taps, float(config["ridge_lambda"]))
                generator[key], _ = _query_generator(loaded[key], records[key], normal_mean, normal_std, eog_mean, eog_std, taps, float(config["ridge_lambda"]))
            population = np.mean(np.stack([support[key] for key in train_eligible]), axis=0)
            common_eog = np.random.default_rng(int(config["pair_seed"]) + fold_index + int(budget)).standard_normal((population.shape[1], 4096))
            for key in support:
                features[key] = _support_features(loaded[key], support[key], population, reliability[key], budget, samples)
            train_oracle = []
            for key in train_eligible:
                scores = [_distance_utility(population + lam * (support[key] - population), generator[key], common_eog) for lam in grid]
                train_oracle.append(float(grid[int(np.argmin(scores))]))
            train_x = np.stack([features[key] for key in train_eligible]); train_y = np.asarray(train_oracle)
            for key in fold["heldout"]:
                base = {"budget_seconds": budget, "fold_id": fold["fold_id"], "fold_cluster": fold_index, "study": fold["study"], "recording_key": key,
                        "eligible": bool(eligible[key]), "calibration_samples": samples if eligible[key] else 0}
                if not eligible[key]:
                    rows.append({**base, "status": "ineligible_support_too_short"}); continue
                scores = np.asarray([_distance_utility(population + lam * (support[key] - population), generator[key], common_eog) for lam in grid])
                oracle_lambda = float(grid[int(np.argmin(scores))]); deployable_lambda = _ridge_predict(train_x, train_y, features[key])
                candidates = {
                    "MATCH": support[key], "POP": population,
                    "EB_ORACLE": population + oracle_lambda * (support[key] - population),
                    "EB_DEPLOYABLE": population + deployable_lambda * (support[key] - population),
                }
                wrong = [donor for donor in fold["heldout"] if donor != key and donor in support]
                wrong_scores = [_distance_utility(support[donor], generator[key], common_eog) for donor in wrong]
                for arm, candidate in candidates.items():
                    distance = _operator_distances(candidate, generator[key], common_eog)
                    rows.append({**base, "status": "success", "arm": arm, "support_reliability": reliability[key],
                                 "oracle_lambda": oracle_lambda, "deployable_lambda": deployable_lambda,
                                 "composite_distance": _distance_utility(candidate, generator[key], common_eog), **distance,
                                 "wrong_donor_count": len(wrong), "mean_wrong_composite_distance": float(np.mean(wrong_scores)) if wrong_scores else float("nan")})
    _csv(root / "unit_operator_metrics.csv", rows)
    success = [row for row in rows if row.get("status") == "success"]
    effects: list[dict[str, Any]] = []
    for budget in map(float, config["support_budgets_seconds"]):
        units = sorted({row["recording_key"] for row in success if row["budget_seconds"] == budget})
        for key in units:
            unit = [row for row in success if row["budget_seconds"] == budget and row["recording_key"] == key]
            by = {row["arm"]: row for row in unit}
            effects.append({"budget_seconds": budget, "fold_id": by["POP"]["fold_id"], "fold_cluster": by["POP"]["fold_cluster"], "study": by["POP"]["study"], "recording_key": key,
                            "oracle_relative_improvement": (by["POP"]["composite_distance"] - by["EB_ORACLE"]["composite_distance"]) / max(by["POP"]["composite_distance"], 1e-12),
                            "deployable_relative_improvement": (by["POP"]["composite_distance"] - by["EB_DEPLOYABLE"]["composite_distance"]) / max(by["POP"]["composite_distance"], 1e-12),
                            "match_relative_improvement": (by["POP"]["composite_distance"] - by["MATCH"]["composite_distance"]) / max(by["POP"]["composite_distance"], 1e-12),
                            "match_vs_wrong_relative_improvement": (by["POP"]["mean_wrong_composite_distance"] - by["MATCH"]["composite_distance"]) / max(by["POP"]["mean_wrong_composite_distance"], 1e-12),
                            "reliability": by["MATCH"]["support_reliability"], "oracle_lambda": by["MATCH"]["oracle_lambda"], "deployable_lambda": by["MATCH"]["deployable_lambda"]})
    _csv(root / "paired_effects.csv", effects)
    bootstrap = _cluster_bootstrap(effects, int(config["statistics"]["bootstrap_replicates"]), int(config["statistics"]["bootstrap_seed"]))
    _csv(root / "bootstrap_summary.csv", bootstrap)
    budget_summaries = {}
    for budget in map(float, config["support_budgets_seconds"]):
        subset = [row for row in effects if row["budget_seconds"] == budget]
        budget_summaries[str(budget)] = {"eligible_stems": len(subset), **({m: float(np.mean([r[m] for r in subset])) for m in ("oracle_relative_improvement", "deployable_relative_improvement", "match_relative_improvement", "match_vs_wrong_relative_improvement")} if subset else {m: None for m in ("oracle_relative_improvement", "deployable_relative_improvement", "match_relative_improvement", "match_vs_wrong_relative_improvement")})}
    summary = {"status": "completed_fold_local_eb_headroom", "folds": len(folds), "availability_denominator": 59,
               "compatible_stems": 58, "budgets_seconds": list(config["support_budgets_seconds"]),
               "interpretation": "operator_shrinkage_headroom_only_not_score_lora_gate",
               "budget_summaries": budget_summaries}
    _json(root / "result_summary.json", summary); _json(run_dir / "result_summary.json", summary)
    _write_eb_report(root, summary, bootstrap)
    return summary


def _cluster_bootstrap(rows: Sequence[Mapping[str, Any]], replicates: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed); result = []
    metrics = ("oracle_relative_improvement", "deployable_relative_improvement", "match_relative_improvement", "match_vs_wrong_relative_improvement")
    for budget in sorted({float(row["budget_seconds"]) for row in rows}):
        subset = [row for row in rows if float(row["budget_seconds"]) == budget]
        studies = sorted({row["study"] for row in subset})
        for metric in metrics:
            draws = []
            for _ in range(replicates):
                values = []
                for study in studies:
                    clusters = sorted({int(row["fold_cluster"]) for row in subset if row["study"] == study})
                    for cluster in rng.choice(clusters, size=len(clusters), replace=True):
                        units = [float(row[metric]) for row in subset if int(row["fold_cluster"]) == int(cluster)]
                        values.extend(rng.choice(units, size=len(units), replace=True))
                draws.append(float(np.mean(values)))
            observed = np.asarray([float(row[metric]) for row in subset])
            result.append({"budget_seconds": budget, "metric": metric, "mean": float(observed.mean()), "median": float(np.median(observed)),
                           "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                           "positive_count": int((observed > 0).sum()), "denominator": len(observed), "bootstrap_replicates": replicates,
                           "resampling": "study_stratified_outer_fold_cluster_then_participant"})
    return result


def _write_eb_report(root: Path, summary: Mapping[str, Any], bootstrap: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Fold-local empirical-Bayes operator headroom", "", "Every support, population, wrong, and evaluator-only generator transfer was re-fitted inside the same outer fold with identical EEG/EOG normalization and compatibility cell. Query outcomes were not used by the deployable lambda predictor.", "", "| budget | effect | mean | 95% cluster CI | positive |", "|---:|---|---:|---:|---:|"]
    for row in bootstrap:
        lines.append(f"| {row['budget_seconds']:.0f}s | {row['metric']} | {row['mean']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['positive_count']}/{row['denominator']} |")
    lines += ["", "This audit determines only whether the H/shrinkage branch has development headroom. It is not a hard gate for score-space LoRA."]
    path = Path("reports/fold_local_eb_operator_headroom.md"); path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _expanded_record_pairs(loaded: Any, record: Any, normal_mean: np.ndarray, normal_std: np.ndarray,
                           *, taps: int, ridge: float, window_seconds: float, cap: int,
                           seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create participant-balanced, disjoint real EEG/EOG-backed pairs."""
    support_samples = int(round(30.0 * record.sampling_rate_hz))
    eog_mean, eog_std = _support_eog_stats(loaded, support_samples)
    eeg = (np.asarray(loaded.query.eeg, np.float64) - normal_mean) / normal_std
    eog = (np.asarray(loaded.query_annotations.external_eog, np.float64) - eog_mean) / eog_std
    labels = np.asarray(loaded.query_annotations.artifactclasses).reshape(-1)
    trial = int(record.samples_per_trial); window = int(round(window_seconds * record.sampling_rate_hz))
    trial_count = int(record.trial_count)
    clean_by_trial: dict[int, list[np.ndarray]] = {}; ocular_by_trial: dict[int, list[np.ndarray]] = {}
    for index in range(trial_count):
        clean: list[np.ndarray] = []; ocular: list[np.ndarray] = []
        for start in range(0, trial - window + 1, window):
            absolute = index * trial + start; label = labels[absolute:absolute + window]
            if np.mean(label == 6) >= .95:
                clean.append(eeg[:, absolute:absolute + window])
            if np.mean(np.isin(label, np.arange(1, 6))) >= .25:
                ocular.append(eog[:, absolute:absolute + window])
        if clean: clean_by_trial[index] = clean
        if ocular: ocular_by_trial[index] = ocular
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    assignments = 0
    for clean_trial, clean_windows in clean_by_trial.items():
        for artifact_trial, ocular_windows in ocular_by_trial.items():
            if artifact_trial == clean_trial:
                continue
            generator_trials = [i for i in range(trial_count) if i not in {clean_trial, artifact_trial}]
            if not generator_trials:
                continue
            gen_eeg = np.concatenate([eeg[:, i * trial:(i + 1) * trial] for i in generator_trials], axis=1)
            gen_eog = np.concatenate([eog[:, i * trial:(i + 1) * trial] for i in generator_trials], axis=1)
            hgen = fit_dynamic_transfer(gen_eeg, gen_eog, taps=taps, ridge=ridge)
            assignments += 1
            candidates.extend((clean_window, apply_dynamic_transfer(hgen, ocular_window)) for clean_window in clean_windows for ocular_window in ocular_windows)
    if not candidates:
        raise ValueError("expanded record has no eligible disjoint pair")
    rng = np.random.default_rng(seed); order = rng.permutation(len(candidates))[:cap]
    chosen = [candidates[int(i)] for i in order]
    x = np.stack([item[0] for item in chosen]); a = np.stack([item[1] for item in chosen])
    return x.astype(np.float32), (x + a).astype(np.float32), a.astype(np.float32), {
        "clean_windows": sum(map(len, clean_by_trial.values())), "artifact_windows": sum(map(len, ocular_by_trial.values())),
        "trial_role_assignments": assignments, "cartesian_candidates": len(candidates),
        "selected_pairs": len(chosen), "trial_roles_disjoint": True,
    }


def _pad(values: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[-1] > length:
        raise ValueError("cannot shrink a real window")
    result = np.zeros((*values.shape[:-1], length), np.float32); result[..., :values.shape[-1]] = values
    valid = np.zeros((values.shape[0], length), bool); valid[:, :values.shape[-1]] = True
    return result, valid


def stage_prepare_base(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    fold_id = str(config["diagnostic_folds"][task_index]); fold = next(row for row in _folds(config) if row["fold_id"] == fold_id)
    root = Path(str(config["result_root"])); destination = root / "prepared_base" / fold_id
    layouts, records = _metadata(config); data_root = Path(str(config["data_root"])); rate = float(fold["sampling_rate_hz"])
    support_samples = int(round(30.0 * rate)); taps = 2 * int(round(float(config["fir_lag_ms"]) * rate / 1000.0)) + 1
    loaded = {key: load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=True, include_query_annotations=True) for key in fold["training"]}
    train_support = np.concatenate([item.support.eeg[:, :support_samples] for item in loaded.values()], axis=1).astype(np.float64)
    normal_mean = train_support.mean(1, keepdims=True); normal_std = train_support.std(1, keepdims=True).clip(1e-6)
    transfers = []
    for key, item in loaded.items():
        eog_mean, eog_std = _support_eog_stats(item, support_samples)
        transfer, _ = _support_transfer(item, eog_mean, eog_std, normal_mean, normal_std, support_samples, taps, float(config["ridge_lambda"]))
        transfers.append(transfer.reshape(transfer.shape[0], -1))
    population_transfer = np.mean(np.stack(transfers), axis=0)
    population_basis, _, _ = aligned_artifact_basis(population_transfer)
    xs: list[np.ndarray] = []; ys: list[np.ndarray] = []; artifacts: list[np.ndarray] = []; keys: list[str] = []; manifest = []
    for offset, (key, item) in enumerate(loaded.items()):
        x, y, artifact, audit = _expanded_record_pairs(item, records[key], normal_mean, normal_std, taps=taps,
            ridge=float(config["ridge_lambda"]), window_seconds=float(config["window_seconds"]),
            cap=int(config["pair_cap_per_participant"]), seed=int(config["pair_seed"]) + 101 * task_index + offset)
        xs.append(x); ys.append(y); artifacts.append(artifact); keys.extend([key] * len(x)); manifest.append({"fold_id": fold_id, "recording_key": key, **audit})
    x = np.concatenate(xs); y = np.concatenate(ys); artifact = np.concatenate(artifacts)
    raw_length = x.shape[-1]; padded_length = int(math.ceil(raw_length / 8.0) * 8)
    x, valid = _pad(x, padded_length); y, _ = _pad(y, padded_length); artifact, _ = _pad(artifact, padded_length)
    coefficient = np.einsum("cr,nct->nrt", population_basis.astype(np.float64), artifact.astype(np.float64))
    tau = training_tau(coefficient); target_u = np.tanh(coefficient / tau[None, :, None]).astype(np.float32)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "training_pairs.npz", x=x, y=y, a=artifact, target_u=target_u, valid=valid,
                        key=np.asarray(keys), population_basis=population_basis, tau=tau, normal_mean=normal_mean.astype(np.float32),
                        normal_std=normal_std.astype(np.float32), raw_length=np.int64(raw_length))
    _csv(destination / "expanded_pair_manifest.csv", manifest)
    summary = {"status": "prepared_expanded_real_pairs", "fold_id": fold_id, "participants": len(loaded), "unique_pairs": len(x),
               "legacy_pair_count": int(np.load(Path(str(config["v6_root"])) / "prepared" / fold_id / "training_pairs.npz")["y"].shape[0]),
               "pair_expansion_ratio": float(len(x) / max(1, np.load(Path(str(config["v6_root"])) / "prepared" / fold_id / "training_pairs.npz")["y"].shape[0])),
               "raw_length": raw_length, "padded_length": padded_length, "trial_roles_disjoint": all(row["trial_roles_disjoint"] for row in manifest)}
    _json(destination / "result_summary.json", summary); _json(run_dir / "result_summary.json", summary); return summary


def _condition(y: torch.Tensor, basis: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = y.shape[0]
    return {"observed": y, "basis": basis.expand(batch, -1, -1), "reliability": torch.ones(batch, device=y.device),
            "rank_mask": torch.ones((batch, 2), dtype=torch.bool, device=y.device), "valid_time_mask": valid}


def _masked_u_mse(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[:, None].to(predicted); return ((predicted - target).square() * mask).sum() / (mask.sum() * predicted.shape[1]).clamp_min(1)


def _rrmse(value: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(value - target) / max(np.linalg.norm(target), 1e-12))


@torch.no_grad()
def _oracle_roundtrip(model: ArtifactSubspaceDiffusion, target: torch.Tensor, noise: torch.Tensor) -> float:
    sequence = model._sequence(); state = noise.clone()
    # At every state, the analytic v corresponding to the same target/noise
    # exactly preserves the DDIM non-Markovian noise path.
    for index, step in enumerate(sequence):
        alpha = model.alpha_bar[step]
        state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        v = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
        x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * v
        epsilon = (1 - alpha).sqrt() * state + alpha.sqrt() * v
        if index + 1 < len(sequence):
            next_alpha = model.alpha_bar[sequence[index + 1]]
            state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
        else:
            state = x0
    return float(torch.linalg.norm(state - target) / torch.linalg.norm(target).clamp_min(1e-12))


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def stage_base_validity(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    fold_id = str(config["diagnostic_folds"][task_index]); root = Path(str(config["result_root"])); folder = root / "prepared_base" / fold_id
    arrays = np.load(folder / "training_pairs.npz"); device = torch.device("cuda"); seed = int(config["training"]["seed"])
    torch.manual_seed(seed + task_index); torch.cuda.manual_seed_all(seed + task_index); rng = np.random.default_rng(seed + task_index)
    cfg = ArtifactSubspaceConfig(eeg_channels=arrays["y"].shape[1], signal_length=arrays["y"].shape[2], base_channels=int(config["model"]["base_channels"]),
                                 num_timesteps=int(config["model"]["timesteps"]), min_snr_gamma=float(config["model"]["min_snr_gamma"]),
                                 ddim_steps=int(config["model"]["ddim_steps"]), posterior_samples=int(config["model"]["posterior_samples"]))
    basis = torch.tensor(arrays["population_basis"][None], device=device); tau = torch.tensor(arrays["tau"], device=device)
    # Ladder 1: analytic sampler, then a real single-window fixed target.
    probe = ArtifactSubspaceDiffusion(cfg).to(device); target_one = torch.tensor(arrays["target_u"][:1], device=device)
    noise_one = torch.randn(target_one.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed + 77))
    oracle_error = _oracle_roundtrip(probe, target_one, noise_one)
    y_one = torch.tensor(arrays["y"][:1], device=device); valid_one = torch.tensor(arrays["valid"][:1], device=device)
    fixed_t = torch.full((1,), 750, device=device, dtype=torch.long); fixed_noise = noise_one.clone(); fixed_opt = AdamW(probe.parameters(), lr=5e-4)
    fixed_curve = []
    for step in range(1, int(config["training"]["fixed_window_updates"]) + 1):
        fixed_opt.zero_grad(set_to_none=True); loss, details = probe.training_loss(target_one, generator=torch.Generator(device=device).manual_seed(seed), timestep=fixed_t, noise=fixed_noise, **_condition(y_one, basis, valid_one)); loss.backward(); fixed_opt.step()
        if step == 1 or step % 100 == 0: fixed_curve.append({"step": step, "loss": float(loss.detach()), "x0_mse": float(details["u_mse"])})
    _csv(run_dir / "fixed_window_overfit.csv", fixed_curve); _csv(folder / "fixed_window_overfit.csv", fixed_curve)
    fixed_pass = fixed_curve[-1]["x0_mse"] <= 1e-4
    # Ladder 2+: population DET and DIFF on expanded real pairs.
    det = DeterministicSubspaceEstimator(cfg).to(device); diff = ArtifactSubspaceDiffusion(cfg).to(device)
    det_opt = AdamW(det.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    diff_opt = AdamW(diff.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    generator = torch.Generator(device=device).manual_seed(seed + 991); ema = _clone_state(diff); curves = []
    batch_size = int(config["training"]["batch_size"]); updates = int(config["training"]["successful_updates"])
    for step in range(1, updates + 1):
        index = rng.integers(0, len(arrays["y"]), size=batch_size)
        y = torch.tensor(arrays["y"][index], device=device); target = torch.tensor(arrays["target_u"][index], device=device); valid = torch.tensor(arrays["valid"][index], device=device); condition = _condition(y, basis, valid)
        det_opt.zero_grad(set_to_none=True); det_loss = _masked_u_mse(det(**condition), target, valid); det_loss.backward(); det_grad = float(torch.nn.utils.clip_grad_norm_(det.parameters(), float(config["training"]["gradient_clip_norm"]))); det_opt.step()
        diff_opt.zero_grad(set_to_none=True); diff_loss, detail = diff.training_loss(target, generator=generator, **condition); diff_loss.backward(); diff_grad = float(torch.nn.utils.clip_grad_norm_(diff.parameters(), float(config["training"]["gradient_clip_norm"]))); diff_opt.step()
        with torch.no_grad():
            for name, value in diff.state_dict().items(): ema[name].mul_(float(config["training"]["ema_decay"])).add_(value.detach().cpu(), alpha=1-float(config["training"]["ema_decay"]))
        if step == 1 or step % 100 == 0: curves.append({"step": step, "det_loss": float(det_loss.detach()), "diff_loss": float(diff_loss.detach()), "diff_x0_mse": float(detail["u_mse"]), "det_grad": det_grad, "diff_grad": diff_grad})
        if step % int(config["training"]["checkpoint_interval"]) == 0:
            checkpoint = folder / "checkpoint.pt"; torch.save({"step": step, "det": det.state_dict(), "diff": diff.state_dict(), "diff_ema": ema, "det_optimizer": det_opt.state_dict(), "diff_optimizer": diff_opt.state_dict(), "generator_state": generator.get_state(), "numpy_state": rng.bit_generator.state, "model_config": cfg.__dict__, "tau": arrays["tau"], "population_basis": arrays["population_basis"]}, checkpoint)
    _csv(run_dir / "training_curve.csv", curves); _csv(folder / "training_curve.csv", curves); diff.load_state_dict(ema); det.eval(); diff.eval()
    metrics, timestep_rows, k_rows, trajectory_rows = _evaluate_base(config, fold_id, arrays, det, diff, basis, tau, device, seed)
    for destination in (run_dir, folder):
        _csv(destination / "base_metrics.csv", metrics); _csv(destination / "timestep_x0_metrics.csv", timestep_rows); _csv(destination / "k_convergence.csv", k_rows); _csv(destination / "sampler_trajectory.csv", trajectory_rows)
    held = [row for row in metrics if row["split"] == "heldout" and row["method"] == "DIFF-POP"]
    det_held = {row["recording_key"]: row for row in metrics if row["split"] == "heldout" and row["method"] == "DET-POP"}
    raw_held = {row["recording_key"]: row for row in metrics if row["split"] == "heldout" and row["method"] == "RAW"}
    mean_diff = float(np.mean([r["rrmse"] for r in held])); mean_det = float(np.mean([det_held[r["recording_key"]]["rrmse"] for r in held])); mean_raw = float(np.mean([raw_held[r["recording_key"]]["rrmse"] for r in held]))
    high_noise = [row for row in timestep_rows if row["timestep"] >= 750 and row["split"] == "heldout"]
    high_noise_valid = bool(high_noise and np.isfinite([row["x0_rrmse"] for row in high_noise]).all() and max(row["x0_rms_ratio"] for row in high_noise) < 10)
    natural = [row for row in metrics if row["split"] == "natural" and row["method"] == "DIFF-POP"]
    preservation = float(np.nanmean([row["nonartifact_preservation"] for row in natural])) if natural else float("nan")
    psd = float(np.nanmean([row["psd_distortion"] for row in natural])) if natural else float("nan")
    covariance = float(np.nanmean([row["covariance_distortion"] for row in natural])) if natural else float("nan")
    summary = {"status": "completed_population_base_validity", "fold_id": fold_id, "oracle_roundtrip_error": oracle_error, "oracle_roundtrip_pass": oracle_error < 1e-4,
               "fixed_window_x0_mse": fixed_curve[-1]["x0_mse"], "fixed_window_overfit_pass": fixed_pass, "expanded_pairs": len(arrays["y"]),
               "mean_diff_pop_rrmse": mean_diff, "mean_det_pop_rrmse": mean_det, "mean_raw_rrmse": mean_raw,
               "diff_better_than_raw": mean_diff < mean_raw, "diff_relative_to_det": mean_diff / max(mean_det, 1e-12) - 1,
               "high_noise_scale_valid": high_noise_valid, "natural_preservation": preservation, "natural_psd_distortion": psd,
               "natural_covariance_distortion": covariance, "checkpoint": str(folder / "checkpoint.pt")}
    _json(folder / "validity_summary.json", summary); _json(run_dir / "result_summary.json", summary); return summary


@torch.no_grad()
def _evaluate_base(config: Mapping[str, Any], fold_id: str, arrays: Any, det: DeterministicSubspaceEstimator,
                   diff: ArtifactSubspaceDiffusion, basis: torch.Tensor, tau: torch.Tensor, device: torch.device,
                   seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []; timestep_rows: list[dict[str, Any]] = []; k_rows: list[dict[str, Any]] = []; trajectory_rows: list[dict[str, Any]] = []
    root = Path(str(config["v6_root"])) / "prepared" / fold_id; raw_length = int(arrays["raw_length"])
    sets = [("training", "training", arrays["x"][:16], arrays["y"][:16], arrays["a"][:16])]
    for path in sorted(root.glob("paired_*.npz")):
        item = np.load(path); key = path.stem.removeprefix("paired_").replace("__", "/"); sets.append(("heldout", key, item["x"], item["y"], item["a"]))
    for split, key, x_np, y_np, a_np in sets:
        length = arrays["y"].shape[2]; x_pad, valid_np = _pad(np.asarray(x_np), length); y_pad, _ = _pad(np.asarray(y_np), length); a_pad, _ = _pad(np.asarray(a_np), length)
        y = torch.tensor(y_pad, device=device); valid = torch.tensor(valid_np, device=device); condition = _condition(y, basis, valid)
        det_u = det(**condition); det_x, det_a = reconstruct_from_subspace(y, condition["basis"], det_u, tau, condition["rank_mask"], valid)
        values = {"RAW": y_pad[..., :raw_length], "DET-POP": det_x.cpu().numpy()[..., :raw_length]}
        for k in (1, 8, 32):
            sample_seeds = tuple(seed * 100003 + 97 * i + sum(ord(c) for c in key) for i in range(k))
            u, variance, calls, trajectory = diff.sample(sample_seeds=sample_seeds, record_trajectory=(k == 8), **condition)
            restored, correction = reconstruct_from_subspace(y, condition["basis"], u, tau, condition["rank_mask"], valid)
            rrmse = _rrmse(restored.cpu().numpy()[..., :raw_length], np.asarray(x_np))
            k_rows.append({"fold_id": fold_id, "split": split, "recording_key": key, "K": k, "rrmse": rrmse, "sample_variance": float(variance.mean()), "network_calls": calls})
            if k == 8:
                trajectory_rows.extend({"fold_id": fold_id, "split": split, "recording_key": key, **row} for row in trajectory)
            if k == 8: values["DIFF-POP"] = restored.cpu().numpy()[..., :raw_length]
        for method, value in values.items():
            metrics.append({"fold_id": fold_id, "split": split, "recording_key": key, "method": method, "rrmse": _rrmse(value, np.asarray(x_np)),
                            "artifact_correlation": float(np.corrcoef((np.asarray(y_np) - value).ravel(), np.asarray(a_np).ravel())[0, 1]),
                            "artifact_rms_ratio": float(np.sqrt(np.mean((np.asarray(y_np) - value) ** 2)) / max(np.sqrt(np.mean(np.asarray(a_np) ** 2)), 1e-12))})
        target_u = torch.tanh(torch.einsum("bcr,bct->brt", condition["basis"], torch.tensor(a_pad, device=device)) / tau[None, :, None])
        fixed_noise = torch.randn(target_u.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed + 123))
        for timestep in (999, 950, 750, 500, 250, 0):
            t = torch.full((len(y),), timestep, device=device, dtype=torch.long); alpha = diff.alpha_bar[timestep]
            state = alpha.sqrt() * target_u + (1 - alpha).sqrt() * fixed_noise; predicted_v = diff.backbone(state, t, **condition)
            x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * predicted_v
            timestep_rows.append({"fold_id": fold_id, "split": split, "recording_key": key, "timestep": timestep,
                                  "x0_rrmse": float(torch.linalg.norm(x0 - target_u) / torch.linalg.norm(target_u).clamp_min(1e-12)),
                                  "x0_rms_ratio": float(torch.sqrt(x0.square().mean()) / torch.sqrt(target_u.square().mean()).clamp_min(1e-12)),
                                  "artifact_correlation": float(np.corrcoef(x0.cpu().numpy().ravel(), target_u.cpu().numpy().ravel())[0, 1])})
    # Natural query is evaluated only after the deployable output is formed.
    for path in sorted(root.glob("natural_input_*.npz")):
        key = path.stem.removeprefix("natural_input_").replace("__", "/"); item = np.load(path)
        raw = np.asarray(item["y"], np.float32); raw_length = int(arrays["raw_length"]); usable = raw.shape[1] // raw_length * raw_length
        windows = raw[:, :usable].reshape(raw.shape[0], -1, raw_length).transpose(1, 0, 2); padded, valid_np = _pad(windows, arrays["y"].shape[2])
        output = []
        for start in range(0, len(padded), 16):
            y = torch.tensor(padded[start:start + 16], device=device); valid = torch.tensor(valid_np[start:start + 16], device=device); condition = _condition(y, basis, valid)
            seeds = tuple(seed * 100003 + 97 * i + sum(ord(c) for c in key) for i in range(8))
            u, _, _, _ = diff.sample(sample_seeds=seeds, **condition); restored, _ = reconstruct_from_subspace(y, condition["basis"], u, tau, condition["rank_mask"], valid)
            output.append(restored.cpu().numpy()[..., :raw_length])
        continuous = np.concatenate(output).transpose(1, 0, 2).reshape(raw.shape[0], usable)
        evaluator = np.load(root / f"natural_evaluator_{key.replace('/', '__')}.npz")
        natural = _natural_metrics(raw[:, :usable], continuous, evaluator["eog"], evaluator["labels"], float(next(f["sampling_rate_hz"] for f in _folds(config) if f["fold_id"] == fold_id)))
        metrics.append({"fold_id": fold_id, "split": "natural", "recording_key": key, "method": "DIFF-POP", **natural})
    return metrics, timestep_rows, k_rows, trajectory_rows


def stage_base_gate(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = Path(str(config["result_root"])); rows = [json.loads((root / "prepared_base" / fold / "validity_summary.json").read_text()) for fold in config["diagnostic_folds"]]
    pass_roundtrip = all(row["oracle_roundtrip_pass"] for row in rows); pass_fixed = all(row["fixed_window_overfit_pass"] for row in rows)
    pass_raw = float(np.mean([row["mean_diff_pop_rrmse"] for row in rows])) < float(np.mean([row["mean_raw_rrmse"] for row in rows])) and sum(row["diff_better_than_raw"] for row in rows) >= 2
    pass_det = float(np.mean([row["mean_diff_pop_rrmse"] for row in rows])) <= 1.10 * float(np.mean([row["mean_det_pop_rrmse"] for row in rows]))
    pass_scale = all(row["high_noise_scale_valid"] for row in rows)
    pass_natural = all(row["natural_preservation"] >= .75 and row["natural_psd_distortion"] <= .25 and row["natural_covariance_distortion"] <= .25 for row in rows)
    gate = pass_roundtrip and pass_fixed and pass_raw and pass_det and pass_scale and pass_natural
    summary = {"status": "passed" if gate else "failed", "decision": "POPULATION_DIFFUSION_BASE_VALID" if gate else "POPULATION_DIFFUSION_BASE_GATE_FAILED",
               "score_lora_authorized": gate, "analytic_roundtrip": pass_roundtrip, "single_window_fixed_overfit": pass_fixed,
               "mean_diff_pop_beats_raw_and_2of3": pass_raw, "relative_to_det_within_10_percent": pass_det, "high_noise_scale_valid": pass_scale,
               "natural_safety": pass_natural,
               "folds": rows, "failure_scope": None if gate else "population_backbone_or_expanded_data_or_sampling_variance_not_subject_adaptation"}
    _json(root / "routing_decision.json", summary); _json(run_dir / "result_summary.json", summary); return summary


def _support_pseudo_pairs(loaded: Any, record: Any, normal_mean: np.ndarray, normal_std: np.ndarray,
                          population_basis: np.ndarray, tau: np.ndarray, *, budget: float, taps: int,
                          ridge: float, shuffled: bool, split: str, seed: int) -> dict[str, np.ndarray]:
    samples = int(round(budget * record.sampling_rate_hz)); half = samples // 2
    start, stop = (0, half) if split == "adapt" else (half, samples)
    eeg = (np.asarray(loaded.support.eeg[:, :samples], np.float64) - normal_mean) / normal_std
    eog_mean, eog_std = _support_eog_stats(loaded, samples); eog = (np.asarray(loaded.support.external_eog[:, :samples], np.float64) - eog_mean) / eog_std
    labels = np.asarray(loaded.support.artifactclasses[:samples]).reshape(-1)
    bounds = np.linspace(start, stop, 4, dtype=int); chunks = [(bounds[i], bounds[i + 1]) for i in range(3)]
    best = None
    import itertools
    window = int(round(2.0 * record.sampling_rate_hz))
    for fit_i, clean_i, artifact_i in itertools.permutations(range(3)):
        clean_count = sum(np.mean(labels[s:s + window] == 6) >= .95 for s in range(chunks[clean_i][0], chunks[clean_i][1] - window + 1, window))
        artifact_count = sum(np.mean(np.isin(labels[s:s + window], np.arange(1, 6))) >= .25 for s in range(chunks[artifact_i][0], chunks[artifact_i][1] - window + 1, window))
        score = clean_count * artifact_count
        if best is None or score > best[0]: best = (score, fit_i, clean_i, artifact_i)
    if best is None or best[0] == 0:
        raise ValueError("support half lacks disjoint FIR/clean/artifact pseudo-pair roles")
    _, fit_i, clean_i, artifact_i = best; fit_start, fit_stop = chunks[fit_i]
    fit_eog = eog[:, fit_start:fit_stop].copy()
    if shuffled:
        fit_eog = fit_eog[:, np.random.default_rng(seed).permutation(fit_eog.shape[1])]
    transfer = fit_dynamic_transfer(eeg[:, fit_start:fit_stop], fit_eog, taps=taps, ridge=ridge)
    clean = [eeg[:, s:s + window] for s in range(chunks[clean_i][0], chunks[clean_i][1] - window + 1, window) if np.mean(labels[s:s + window] == 6) >= .95]
    ocular = [eog[:, s:s + window] for s in range(chunks[artifact_i][0], chunks[artifact_i][1] - window + 1, window) if np.mean(np.isin(labels[s:s + window], np.arange(1, 6))) >= .25]
    pairs = [(x, apply_dynamic_transfer(transfer, eye)) for x in clean for eye in ocular]
    pairs = pairs[:16]; x = np.stack([item[0] for item in pairs]).astype(np.float32); a = np.stack([item[1] for item in pairs]).astype(np.float32); y = x + a
    length = int(math.ceil(window / 8) * 8); x, valid = _pad(x, length); y, _ = _pad(y, length); a, _ = _pad(a, length)
    coefficient = np.einsum("cr,nct->nrt", population_basis.astype(np.float64), a.astype(np.float64)); target_u = np.tanh(coefficient / tau[None, :, None]).astype(np.float32)
    return {"x": x, "y": y, "a": a, "target_u": target_u, "valid": valid}


def _fresh_adapted(checkpoint: Mapping[str, Any], kind: str, device: torch.device, rank: int) -> tuple[Any, Any]:
    cfg = ArtifactSubspaceConfig(**checkpoint["model_config"])
    model = DeterministicSubspaceEstimator(cfg).to(device) if kind == "det" else ArtifactSubspaceDiffusion(cfg).to(device)
    model.load_state_dict(checkpoint[kind if kind == "det" else "diff_ema"])
    summary = inject_score_lora(model.backbone, rank=rank)
    return model, summary


def _adapt_lora(model: Any, kind: str, pairs: Mapping[str, np.ndarray], basis: torch.Tensor,
                updates: int, learning_rate: float, seed: int, device: torch.device) -> None:
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed); generator = torch.Generator(device=device).manual_seed(seed)
    model.train()
    for _ in range(updates):
        index = rng.integers(0, len(pairs["y"]), size=min(8, len(pairs["y"])))
        y = torch.tensor(pairs["y"][index], device=device); target = torch.tensor(pairs["target_u"][index], device=device); valid = torch.tensor(pairs["valid"][index], device=device); condition = _condition(y, basis, valid)
        optimizer.zero_grad(set_to_none=True)
        loss = _masked_u_mse(model(**condition), target, valid) if kind == "det" else model.training_loss(target, generator=generator, **condition)[0]
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optimizer.step()
    model.eval()


@torch.no_grad()
def _score_support(model: Any, kind: str, pairs: Mapping[str, np.ndarray], basis: torch.Tensor, tau: torch.Tensor,
                   seed: int, device: torch.device) -> tuple[float, float]:
    y = torch.tensor(pairs["y"], device=device); valid = torch.tensor(pairs["valid"], device=device); condition = _condition(y, basis, valid)
    if kind == "det": u = model(**condition)
    else: u = model.sample(sample_seeds=tuple(seed + 97 * i for i in range(8)), **condition)[0]
    restored, _ = reconstruct_from_subspace(y, condition["basis"], u, tau, condition["rank_mask"], valid)
    rrmse = _rrmse(restored.cpu().numpy(), pairs["x"]); scale = float(torch.sqrt(restored.square().mean()) / torch.sqrt(y.square().mean()).clamp_min(1e-12))
    return rrmse, scale


def stage_support_lora(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    routing = json.loads((Path(str(config["result_root"])) / "routing_decision.json").read_text())
    if not routing.get("score_lora_authorized"):
        summary = {"status": "not_run_blocked_by_population_base_gate", "task_index": task_index}; _json(run_dir / "result_summary.json", summary); return summary
    fold_id = str(config["diagnostic_folds"][task_index]); fold = next(row for row in _folds(config) if row["fold_id"] == fold_id); root = Path(str(config["result_root"]))
    checkpoint = torch.load(root / "prepared_base" / fold_id / "checkpoint.pt", map_location="cpu", weights_only=False); device = torch.device("cuda")
    arrays = np.load(root / "prepared_base" / fold_id / "training_pairs.npz"); basis_np = np.asarray(checkpoint["population_basis"], np.float32); tau_np = np.asarray(checkpoint["tau"], np.float32)
    basis = torch.tensor(basis_np[None], device=device); tau = torch.tensor(tau_np, device=device)
    layouts, records = _metadata(config); data_root = Path(str(config["data_root"])); normal_mean = arrays["normal_mean"]; normal_std = arrays["normal_std"]
    rows = []
    for position, key in enumerate(fold["heldout"]):
        loaded = load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=False, include_query_annotations=False)
        donor_key = fold["heldout"][(position + 1) % len(fold["heldout"])]
        donor = load_sgeyesub_signal_record(data_root, records[donor_key], layouts[records[donor_key].layout_id], include_query=False, include_query_annotations=False)
        selected = None
        for budget in (float(config["score_lora_support_budget_seconds"]),):
            try:
                adapt = _support_pseudo_pairs(loaded, records[key], normal_mean, normal_std, basis_np, tau_np, budget=budget, taps=2*int(round(float(config["fir_lag_ms"])*float(fold["sampling_rate_hz"])/1000))+1, ridge=float(config["ridge_lambda"]), shuffled=False, split="adapt", seed=int(config["training"]["seed"]))
                validation = _support_pseudo_pairs(loaded, records[key], normal_mean, normal_std, basis_np, tau_np, budget=budget, taps=2*int(round(float(config["fir_lag_ms"])*float(fold["sampling_rate_hz"])/1000))+1, ridge=float(config["ridge_lambda"]), shuffled=False, split="validation", seed=int(config["training"]["seed"]))
                wrong = _support_pseudo_pairs(donor, records[donor_key], normal_mean, normal_std, basis_np, tau_np, budget=budget, taps=2*int(round(float(config["fir_lag_ms"])*float(fold["sampling_rate_hz"])/1000))+1, ridge=float(config["ridge_lambda"]), shuffled=False, split="adapt", seed=int(config["training"]["seed"]))
                shuffled = _support_pseudo_pairs(loaded, records[key], normal_mean, normal_std, basis_np, tau_np, budget=budget, taps=2*int(round(float(config["fir_lag_ms"])*float(fold["sampling_rate_hz"])/1000))+1, ridge=float(config["ridge_lambda"]), shuffled=True, split="adapt", seed=int(config["training"]["seed"]))
                selected = (budget, adapt, validation, wrong, shuffled); break
            except ValueError:
                continue
        if selected is None:
            rows.append({"fold_id": fold_id, "recording_key": key, "status": "blocked_support_pair_coverage"}); continue
        budget, adapt, validation, wrong, shuffled = selected; models = {}
        for arm, kind, training_pairs in (("DET-MATCH", "det", adapt), ("DIFF-MATCH", "diff", adapt), ("DIFF-WRONG", "diff", wrong), ("DIFF-SHUFFLED", "diff", shuffled)):
            model, info = _fresh_adapted(checkpoint, kind, device, int(config["score_lora"]["rank"])); _adapt_lora(model, kind, training_pairs, basis, int(config["score_lora"]["support_updates"]), float(config["score_lora"]["learning_rate"]), int(config["training"]["seed"]) + position, device); models[arm] = model
            score, scale = _score_support(model, kind, validation, basis, tau, int(config["training"]["seed"]) + position * 1000, device)
            rows.append({"fold_id": fold_id, "recording_key": key, "status": "success", "arm": arm, "support_budget_seconds": budget, "validation_rrmse": score, "output_input_rms": scale, "lora_rank": info.rank, "trainable_parameters": info.trainable_parameters, "wrong_donor": donor_key if "WRONG" in arm else ""})
        base_diff = ArtifactSubspaceDiffusion(ArtifactSubspaceConfig(**checkpoint["model_config"])).to(device); base_diff.load_state_dict(checkpoint["diff_ema"]); base_diff.eval(); score, scale = _score_support(base_diff, "diff", validation, basis, tau, int(config["training"]["seed"]) + position * 1000, device)
        rows.append({"fold_id": fold_id, "recording_key": key, "status": "success", "arm": "DIFF-NOADAPT", "support_budget_seconds": budget, "validation_rrmse": score, "output_input_rms": scale})
    _csv(run_dir / "support_inner_metrics.csv", rows); _csv(root / "support_inner" / f"{fold_id}.csv", rows)
    success = [r for r in rows if r["status"] == "success"]; by = {(r["recording_key"], r["arm"]): r for r in success}; keys = sorted({r["recording_key"] for r in success})
    effects = [{"key": key, "match_vs_noadapt": by[(key,"DIFF-NOADAPT")]["validation_rrmse"]-by[(key,"DIFF-MATCH")]["validation_rrmse"], "match_vs_wrong": by[(key,"DIFF-WRONG")]["validation_rrmse"]-by[(key,"DIFF-MATCH")]["validation_rrmse"], "match_vs_shuffled": by[(key,"DIFF-SHUFFLED")]["validation_rrmse"]-by[(key,"DIFF-MATCH")]["validation_rrmse"], "diff_vs_det": by[(key,"DET-MATCH")]["validation_rrmse"]-by[(key,"DIFF-MATCH")]["validation_rrmse"], "scale_safe": .5 <= by[(key,"DIFF-MATCH")]["output_input_rms"] <= 2.0} for key in keys]
    effect_names = ("match_vs_noadapt","match_vs_wrong","match_vs_shuffled","diff_vs_det")
    summary = {"status": "completed_support_inner" if keys else "blocked_support_pair_coverage", "fold_id": fold_id, "eligible": len(keys),
               "effects": {name: float(np.mean([r[name] for r in effects])) if effects else 0.0 for name in effect_names},
               "scale_safe": bool(effects) and all(r["scale_safe"] for r in effects)}
    _json(run_dir / "result_summary.json", summary); _json(root / "support_inner" / f"{fold_id}.json", summary); return summary


def stage_support_gate(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = Path(str(config["result_root"])); base = json.loads((root / "routing_decision.json").read_text())
    if not base.get("score_lora_authorized"):
        summary = {"status": "not_run_blocked_by_population_base_gate", "decision": "SCORE_LORA_NOT_STARTED", "later_query_authorized": False}
        _json(root / "support_inner_decision.json", summary); _json(run_dir / "result_summary.json", summary); return summary
    rows = [json.loads((root / "support_inner" / f"{fold}.json").read_text()) for fold in config["diagnostic_folds"]]
    effects = {name: float(np.mean([row["effects"][name] for row in rows])) for name in ("match_vs_noadapt","match_vs_wrong","match_vs_shuffled","diff_vs_det")}
    gate = all(value > 0 for value in effects.values()) and all(row["scale_safe"] for row in rows)
    summary = {"status": "passed" if gate else "failed", "decision": "SCORE_LORA_SUPPORT_INNER_VALID" if gate else "SCORE_LORA_SUPPORT_INNER_FAILED", "later_query_authorized": gate, "effects": effects, "folds": rows}
    _json(root / "support_inner_decision.json", summary); _json(run_dir / "result_summary.json", summary); return summary


def _load_base_models(checkpoint: Mapping[str, Any], device: torch.device) -> tuple[DeterministicSubspaceEstimator, ArtifactSubspaceDiffusion]:
    cfg = ArtifactSubspaceConfig(**checkpoint["model_config"]); det = DeterministicSubspaceEstimator(cfg).to(device); diff = ArtifactSubspaceDiffusion(cfg).to(device)
    det.load_state_dict(checkpoint["det"]); diff.load_state_dict(checkpoint["diff_ema"]); det.eval(); diff.eval(); return det, diff


@torch.no_grad()
def _query_output(model: Any, kind: str, observed: np.ndarray, basis: torch.Tensor, tau: torch.Tensor,
                  seeds: tuple[int, ...], device: torch.device) -> np.ndarray:
    length = model.config.signal_length; padded, valid_np = _pad(np.asarray(observed, np.float32), length); chunks = []
    for start in range(0, len(padded), 16):
        y = torch.tensor(padded[start:start + 16], device=device); valid = torch.tensor(valid_np[start:start + 16], device=device); condition = _condition(y, basis, valid)
        u = model(**condition) if kind == "det" else model.sample(sample_seeds=seeds, **condition)[0]
        restored, _ = reconstruct_from_subspace(y, condition["basis"], u, tau, condition["rank_mask"], valid); chunks.append(restored.cpu().numpy()[..., :observed.shape[-1]])
    return np.concatenate(chunks)


def stage_full_eval(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    root = Path(str(config["result_root"])); support_gate = json.loads((root / "support_inner_decision.json").read_text())
    if not support_gate.get("later_query_authorized"):
        summary = {"status": "not_run_blocked_by_support_inner_gate", "task_index": task_index}; _json(run_dir / "result_summary.json", summary); return summary
    fold = _folds(config)[task_index]; fold_id = fold["fold_id"]; local_config = dict(config); local_config["diagnostic_folds"] = [fold_id]
    prepared = root / "prepared_base" / fold_id
    if not (prepared / "training_pairs.npz").exists(): stage_prepare_base(local_config, 0, run_dir / "prepare")
    if not (prepared / "checkpoint.pt").exists(): stage_base_validity(local_config, 0, run_dir / "population")
    checkpoint = torch.load(prepared / "checkpoint.pt", map_location="cpu", weights_only=False); arrays = np.load(prepared / "training_pairs.npz"); device = torch.device("cuda")
    basis_np = np.asarray(checkpoint["population_basis"], np.float32); tau_np = np.asarray(checkpoint["tau"], np.float32); basis = torch.tensor(basis_np[None], device=device); tau = torch.tensor(tau_np, device=device)
    layouts, records = _metadata(config); data_root = Path(str(config["data_root"])); rate = float(fold["sampling_rate_hz"]); taps = 2*int(round(float(config["fir_lag_ms"])*rate/1000))+1
    rows: list[dict[str, Any]] = []; natural_rows: list[dict[str, Any]] = []; output_dir = root / "full_outputs" / fold_id; output_dir.mkdir(parents=True, exist_ok=True)
    base_det, base_diff = _load_base_models(checkpoint, device)
    for position, key in enumerate(fold["heldout"]):
        target_loaded = load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=False, include_query_annotations=False)
        budget = float(config["score_lora_support_budget_seconds"])
        adapt = _support_pseudo_pairs(target_loaded, records[key], arrays["normal_mean"], arrays["normal_std"], basis_np, tau_np, budget=budget, taps=taps, ridge=float(config["ridge_lambda"]), shuffled=False, split="adapt", seed=int(config["training"]["seed"]))
        shuffled = _support_pseudo_pairs(target_loaded, records[key], arrays["normal_mean"], arrays["normal_std"], basis_np, tau_np, budget=budget, taps=taps, ridge=float(config["ridge_lambda"]), shuffled=True, split="adapt", seed=int(config["training"]["seed"]))
        det_match, _ = _fresh_adapted(checkpoint, "det", device, int(config["score_lora"]["rank"])); _adapt_lora(det_match, "det", adapt, basis, int(config["score_lora"]["support_updates"]), float(config["score_lora"]["learning_rate"]), int(config["training"]["seed"])+position, device)
        diff_match, _ = _fresh_adapted(checkpoint, "diff", device, int(config["score_lora"]["rank"])); _adapt_lora(diff_match, "diff", adapt, basis, int(config["score_lora"]["support_updates"]), float(config["score_lora"]["learning_rate"]), int(config["training"]["seed"])+position, device)
        diff_shuffle, _ = _fresh_adapted(checkpoint, "diff", device, int(config["score_lora"]["rank"])); _adapt_lora(diff_shuffle, "diff", shuffled, basis, int(config["score_lora"]["support_updates"]), float(config["score_lora"]["learning_rate"]), int(config["training"]["seed"])+position, device)
        wrong_models: list[tuple[str, Any, Any]] = []
        for donor_index, donor_key in enumerate([donor for donor in fold["heldout"] if donor != key]):
            donor_loaded = load_sgeyesub_signal_record(data_root, records[donor_key], layouts[records[donor_key].layout_id], include_query=False, include_query_annotations=False)
            donor_pairs = _support_pseudo_pairs(donor_loaded, records[donor_key], arrays["normal_mean"], arrays["normal_std"], basis_np, tau_np, budget=budget, taps=taps, ridge=float(config["ridge_lambda"]), shuffled=False, split="adapt", seed=int(config["training"]["seed"])+donor_index)
            det_wrong, _ = _fresh_adapted(checkpoint, "det", device, int(config["score_lora"]["rank"])); _adapt_lora(det_wrong, "det", donor_pairs, basis, int(config["score_lora"]["support_updates"]), float(config["score_lora"]["learning_rate"]), int(config["training"]["seed"])+donor_index, device)
            diff_wrong, _ = _fresh_adapted(checkpoint, "diff", device, int(config["score_lora"]["rank"])); _adapt_lora(diff_wrong, "diff", donor_pairs, basis, int(config["score_lora"]["support_updates"]), float(config["score_lora"]["learning_rate"]), int(config["training"]["seed"])+donor_index, device)
            wrong_models.append((donor_key, det_wrong, diff_wrong))
        pair = np.load(Path(str(config["v6_root"])) / "prepared" / fold_id / f"paired_{key.replace('/','__')}.npz"); seed_base = int(config["training"]["seed"])*100003 + sum(ord(c) for c in key); seeds = tuple(seed_base + 97*i for i in range(8))
        outputs = {"RAW": pair["y"], "DET-NOADAPT": _query_output(base_det,"det",pair["y"],basis,tau,seeds,device), "DET-MATCH-LoRA": _query_output(det_match,"det",pair["y"],basis,tau,seeds,device), "DIFF-NOADAPT": _query_output(base_diff,"diff",pair["y"],basis,tau,seeds,device), "DIFF-MATCH-LoRA": _query_output(diff_match,"diff",pair["y"],basis,tau,seeds,device), "DIFF-SHUFFLED-LoRA": _query_output(diff_shuffle,"diff",pair["y"],basis,tau,seeds,device)}
        for donor_index, (donor_key, det_wrong, diff_wrong) in enumerate(wrong_models):
            outputs[f"DET-WRONG-{donor_index}"] = _query_output(det_wrong,"det",pair["y"],basis,tau,seeds,device); outputs[f"DIFF-WRONG-{donor_index}"] = _query_output(diff_wrong,"diff",pair["y"],basis,tau,seeds,device)
        np.savez_compressed(output_dir / f"paired_{key.replace('/','__')}.npz", **outputs)
        for method, value in outputs.items():
            rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,"rrmse":_rrmse(value,pair["x"]),"correlation":float(np.corrcoef(value.ravel(),pair["x"].ravel())[0,1]),"delta_snr":float(20*np.log10(max(np.linalg.norm(pair["y"]-pair["x"]),1e-12)/max(np.linalg.norm(value-pair["x"]),1e-12)))})
        natural_input = np.load(Path(str(config["v6_root"])) / "prepared" / fold_id / f"natural_input_{key.replace('/','__')}.npz"); raw = natural_input["y"]; raw_length = int(arrays["raw_length"]); usable = raw.shape[1]//raw_length*raw_length; windows = raw[:,:usable].reshape(raw.shape[0],-1,raw_length).transpose(1,0,2)
        natural_outputs = {method: _query_output(model,kind,windows,basis,tau,seeds,device).transpose(1,0,2).reshape(raw.shape[0],usable) for method,model,kind in (("DET-NOADAPT",base_det,"det"),("DET-MATCH-LoRA",det_match,"det"),("DIFF-NOADAPT",base_diff,"diff"),("DIFF-MATCH-LoRA",diff_match,"diff"),("DIFF-SHUFFLED-LoRA",diff_shuffle,"diff"))}
        for donor_index, (_, det_wrong, diff_wrong) in enumerate(wrong_models): natural_outputs[f"DET-WRONG-{donor_index}"]=_query_output(det_wrong,"det",windows,basis,tau,seeds,device).transpose(1,0,2).reshape(raw.shape[0],usable); natural_outputs[f"DIFF-WRONG-{donor_index}"]=_query_output(diff_wrong,"diff",windows,basis,tau,seeds,device).transpose(1,0,2).reshape(raw.shape[0],usable)
        evaluator=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_evaluator_{key.replace('/','__')}.npz")
        for method,value in natural_outputs.items(): natural_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,**_natural_metrics(raw[:,:usable],value,evaluator["eog"],evaluator["labels"],rate)})
        torch.save({"det_match_lora":lora_state_dict(det_match),"diff_match_lora":lora_state_dict(diff_match),"diff_shuffled_lora":lora_state_dict(diff_shuffle)},output_dir/f"lora_{key.replace('/','__')}.pt")
    _csv(root/"full_fold_metrics"/f"{fold_id}_paired.csv",rows); _csv(root/"full_fold_metrics"/f"{fold_id}_natural.csv",natural_rows)
    summary={"status":"completed_full_fold_score_lora","fold_id":fold_id,"heldout_stems":len(fold["heldout"]),"seed":int(config["training"]["seed"])};_json(run_dir/"result_summary.json",summary);return summary


def stage_aggregate(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root=Path(str(config["result_root"])); support=json.loads((root/"support_inner_decision.json").read_text())
    if not support.get("later_query_authorized"):
        summary={"status":"not_run_blocked_by_support_inner_gate","route_decision":"SCORE_LORA_NOT_EVALUATED_ON_LATER_QUERY"};_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary
    paired=[];natural=[]
    for path in sorted((root/"full_fold_metrics").glob("*_paired.csv")):
        with path.open(newline="",encoding="utf-8") as stream:paired.extend(csv.DictReader(stream))
    for path in sorted((root/"full_fold_metrics").glob("*_natural.csv")):
        with path.open(newline="",encoding="utf-8") as stream:natural.extend(csv.DictReader(stream))
    _csv(root/"unit_metrics.csv",paired); by={(r["recording_key"],r["method"]):r for r in paired}; effects=[]
    for key in sorted({r["recording_key"] for r in paired}):
        wrong=[r for r in paired if r["recording_key"]==key and r["method"].startswith("DIFF-WRONG")]
        effects.append({"recording_key":key,"fold_id":by[(key,"DIFF-MATCH-LoRA")]["fold_id"],"study":by[(key,"DIFF-MATCH-LoRA")]["study"],"U_D":float(by[(key,"DET-MATCH-LoRA")]["rrmse"])-float(by[(key,"DIFF-MATCH-LoRA")]["rrmse"]),"U_P":float(by[(key,"DIFF-NOADAPT")]["rrmse"])-float(by[(key,"DIFF-MATCH-LoRA")]["rrmse"]),"U_W":float(np.mean([float(r["rrmse"])-float(by[(key,"DIFF-MATCH-LoRA")]["rrmse"]) for r in wrong])),"U_S":float(by[(key,"DIFF-SHUFFLED-LoRA")]["rrmse"])-float(by[(key,"DIFF-MATCH-LoRA")]["rrmse"]),"wrong_donors":len(wrong)})
    _csv(root/"paired_effects.csv",effects); bootstrap=_cluster_effect_bootstrap(effects,int(config["statistics"]["bootstrap_replicates"]),int(config["statistics"]["bootstrap_seed"]));_csv(root/"bootstrap_summary.csv",bootstrap)
    methods=[]
    for method in sorted({r["method"] for r in paired}):
        subset=[r for r in paired if r["method"]==method];methods.append({"method":method,"units":len(subset),"rrmse_mean":float(np.mean([float(r["rrmse"]) for r in subset])),"correlation_mean":float(np.mean([float(r["correlation"]) for r in subset])),"delta_snr_mean":float(np.mean([float(r["delta_snr"]) for r in subset]))})
    _csv(root/"method_summary.csv",methods); means={m:float(np.mean([r[m] for r in effects])) for m in ("U_D","U_P","U_W","U_S")}; win=float(np.mean([r["U_D"]>0 for r in effects])); studies=sorted({r["study"] for r in effects}); directions={m:sum(np.mean([r[m] for r in effects if r["study"]==s])>=0 for s in studies) for m in means}
    natural_by={(r["recording_key"],r["method"]):r for r in natural};safety=[]
    for key in sorted({r["recording_key"] for r in natural}):
        match=natural_by[(key,"DIFF-MATCH-LoRA")];pop=natural_by[(key,"DIFF-NOADAPT")];safety.append({"preservation":float(match["nonartifact_preservation"])-float(pop["nonartifact_preservation"]),"psd":float(pop["psd_distortion"])-float(match["psd_distortion"]),"covariance":float(pop["covariance_distortion"])-float(match["covariance_distortion"])})
    safety_means={k:float(np.nanmean([r[k] for r in safety])) for k in ("preservation","psd","covariance")};gate=all(v>0 for v in means.values()) and win>=.55 and all(v>=4 for v in directions.values()) and all(v>=float(config["statistics"]["safety_margin"]) for v in safety_means.values())
    decision="SCORE_LORA_ONE_SEED_PASS_ADDITIONAL_SEEDS_AUTHORIZED" if gate else "CURRENT_SCORE_LORA_INSTANCE_NO_GO"
    summary={"status":"completed_one_seed","route_decision":decision,"additional_seeds_authorized":gate,"coverage":len(effects),"availability_denominator":59,**means,"diffusion_win_fraction":win,"nonnegative_study_counts":directions,"natural_safety_margins":safety_means};_json(root/"routing_decision_final.json",summary);_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def _cluster_effect_bootstrap(rows:Sequence[Mapping[str,Any]],replicates:int,seed:int)->list[dict[str,Any]]:
    rng=np.random.default_rng(seed);result=[]
    for metric in ("U_D","U_P","U_W","U_S"):
        draws=[];studies=sorted({r["study"] for r in rows})
        for _ in range(replicates):
            values=[]
            for study in studies:
                clusters=sorted({r["fold_id"] for r in rows if r["study"]==study})
                for cluster in rng.choice(clusters,size=len(clusters),replace=True):
                    unit=[float(r[metric]) for r in rows if r["fold_id"]==cluster];values.extend(rng.choice(unit,size=len(unit),replace=True))
            draws.append(float(np.mean(values)))
        observed=np.asarray([float(r[metric]) for r in rows]);result.append({"effect":metric,"mean":float(observed.mean()),"median":float(np.median(observed)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"positive_count":int((observed>0).sum()),"denominator":len(observed),"bootstrap_replicates":replicates})
    return result


def stage_finalize(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = Path(str(config["result_root"])); eb_root = Path(str(config["eb_result_root"])); base_path = root / "routing_decision.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else {"decision": "NOT_COMPLETED"}
    support_path = root / "support_inner_decision.json"; support = json.loads(support_path.read_text()) if support_path.exists() else {"decision": "NOT_RUN"}
    final_path = root / "routing_decision_final.json"; final = json.loads(final_path.read_text()) if final_path.exists() else {"route_decision": "NOT_RUN"}
    eb = json.loads((eb_root / "result_summary.json").read_text()) if (eb_root / "result_summary.json").exists() else {"status": "NOT_COMPLETED"}
    base_rows = []
    for fold in config["diagnostic_folds"]:
        path = root / "prepared_base" / fold / "validity_summary.json"
        if path.exists(): base_rows.append(json.loads(path.read_text()))
    if base_rows and not (root / "method_summary.csv").exists():
        methods = []
        for method, field in (("RAW","mean_raw_rrmse"),("DET-POP","mean_det_pop_rrmse"),("DIFF-POP","mean_diff_pop_rrmse")):
            methods.append({"method":method,"diagnostic_folds":len(base_rows),"paired_rrmse_mean":float(np.mean([row[field] for row in base_rows])),"scope":"three_fold_population_validity_not_subject_science"})
        _csv(root / "method_summary.csv", methods)
    if not (root / "paired_effects.csv").exists():
        _csv(root / "paired_effects.csv", [{"status":"not_run","reason":support.get("decision","support_inner_not_completed")}])
    if not (root / "bootstrap_summary.csv").exists():
        _csv(root / "bootstrap_summary.csv", [{"status":"not_run","reason":support.get("decision","support_inner_not_completed")}])
    # Keep only compact, decision-level figures in Git.  Checkpoints and the
    # per-window arrays remain server-side.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    eb_rows: list[dict[str, str]] = []
    eb_bootstrap = eb_root / "bootstrap_summary.csv"
    if eb_bootstrap.exists():
        with eb_bootstrap.open(newline="", encoding="utf-8") as stream:
            eb_rows = list(csv.DictReader(stream))
        labels = {
            "oracle_relative_improvement": "Oracle EB ceiling",
            "deployable_relative_improvement": "Deployable EB",
            "match_relative_improvement": "Unshrunk support",
            "match_vs_wrong_relative_improvement": "Support vs wrong",
        }
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for metric, label in labels.items():
            selected = sorted((r for r in eb_rows if r["metric"] == metric), key=lambda r: float(r["budget_seconds"]))
            x = np.asarray([float(r["budget_seconds"]) for r in selected]); y = np.asarray([float(r["mean"]) for r in selected])
            low = np.asarray([float(r["ci_low"]) for r in selected]); high = np.asarray([float(r["ci_high"]) for r in selected])
            ax.errorbar(x, y, yerr=np.vstack((y-low, high-y)), marker="o", capsize=3, label=label)
        ax.axhline(0, color="black", linewidth=.8); ax.set(xlabel="Support budget (s)", ylabel="Relative response-distance improvement", title="Fold-local EB operator headroom")
        ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(figures / "eb_operator_headroom.png", dpi=180); plt.close(fig)
    if base_rows:
        labels = [row["fold_id"].replace("_layout_", "\nlayout ").replace("_heldout_", "/h") for row in base_rows]
        x = np.arange(len(base_rows)); width = .25
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        for offset, field, label in ((-width,"mean_raw_rrmse","RAW"),(0,"mean_det_pop_rrmse","DET-POP"),(width,"mean_diff_pop_rrmse","DIFF-POP")):
            ax.bar(x+offset, [row[field] for row in base_rows], width, label=label)
        ax.set_xticks(x, labels, fontsize=8); ax.set(ylabel="Paired RRMSE (lower is better)", title="Population validity on three diagnostic folds")
        ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures / "population_base_rrmse.png", dpi=180); plt.close(fig)
        k_values: dict[int, list[float]] = {1: [], 8: [], 32: []}
        for fold in config["diagnostic_folds"]:
            path = root / "prepared_base" / fold / "k_convergence.csv"
            if not path.exists(): continue
            with path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row["split"] == "heldout": k_values[int(row["K"])].append(float(row["rrmse"]))
        if all(k_values.values()):
            fig, ax = plt.subplots(figsize=(5.8, 4.0)); ks = sorted(k_values)
            means = [float(np.mean(k_values[k])) for k in ks]
            ax.plot(ks, means, marker="o"); ax.set_xscale("log", base=2); ax.set_xticks(ks, [str(k) for k in ks])
            ax.set(xlabel="Posterior samples K", ylabel="Mean paired RRMSE", title="K convergence (diagnostic only)")
            fig.tight_layout(); fig.savefig(figures / "posterior_k_convergence.png", dpi=180); plt.close(fig)
    base_means = {
        field: float(np.mean([row[field] for row in base_rows])) if base_rows else float("nan")
        for field in ("mean_raw_rrmse", "mean_det_pop_rrmse", "mean_diff_pop_rrmse")
    }
    eb_headlines = {
        int(float(row["budget_seconds"])): row for row in eb_rows
        if row["metric"] == "deployable_relative_improvement"
    }
    fold_table = "\n".join(
        f"| {row['fold_id']} | {row['expanded_pairs']} | {row['mean_raw_rrmse']:.4f} | {row['mean_det_pop_rrmse']:.4f} | {row['mean_diff_pop_rrmse']:.4f} | "
        f"{'pass' if row['fixed_window_overfit_pass'] else 'fail'} | {row['natural_preservation']:.4f} | {row['natural_psd_distortion']:.4f} | {row['natural_covariance_distortion']:.4f} |"
        for row in base_rows
    )
    eb_table = "\n".join(
        f"| {budget}s | {float(row['mean']):+.4f} | [{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] | {row['positive_count']}/{row['denominator']} |"
        for budget, row in sorted(eb_headlines.items())
    )
    report = Path("reports/sge_score_lora_v8.md"); report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# SGE score-LoRA v8\n\n"
        "## Scientific scope\n\nThis is development evidence. The v6 status is corrected to `CURRENT_V6_BACKBONE_AND_K8_ESTIMATOR_NOT_VALIDATED / CAUSE_NOT_FULLY_IDENTIFIED`; its sampler roundtrip passed, but its K=8 end-to-end estimator was not validated. No v6 seeds, sampler/objective repairs, confirmation data, or MobileBCI data were run.\n\n"
        "## Fold-local EB operator headroom\n\n"
        f"Status: `{eb.get('status')}`. All 58 compatible stems were evaluated; the availability denominator remains 59. Operators and normalization were re-fitted inside each outer fold. The deployable lambda predictor used outer-training support features only.\n\n"
        "| support | deployable relative improvement | 95% fold-cluster CI | positive stems |\n|---:|---:|---:|---:|\n" + eb_table + "\n\n"
        "The 60 s and 120 s budgets show development headroom for fold-local empirical-Bayes shrinkage; 30 s is directionally positive but its interval crosses zero. This supports the H/shrinkage operator branch only and does not establish score-space personalization.\n\n"
        "## Population diffusion implementation validity and utility\n\n"
        f"Decision: `{base.get('decision')}`. All {len(base_rows)}/3 diagnostic folds completed with 32x pair expansion where the legacy fold contained enough participants (256/1664/1664 real paired samples). The analytic roundtrip and high-noise scale checks passed on every fold. Across folds, mean RRMSE was RAW={base_means['mean_raw_rrmse']:.4f}, DET-POP={base_means['mean_det_pop_rrmse']:.4f}, and DIFF-POP={base_means['mean_diff_pop_rrmse']:.4f}. This aggregate signal does not override the frozen per-fold validity and safety gate.\n\n"
        "| diagnostic fold | pairs | RAW | DET-POP | DIFF-POP | fixed-window overfit | preservation | PSD dist. | covariance dist. |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n" + fold_table + "\n\n"
        "The base failed because study04 missed the fixed-window x0-MSE threshold (1.215e-4 versus 1e-4), study01 did not beat RAW and failed preservation/covariance safety, and study02 preservation was 0.7482 versus the frozen 0.75 threshold. Therefore population diffusion utility is promising on two folds but implementation eligibility is not established. K=32 remains diagnostic and does not replace primary K=8.\n\n"
        "## Score-space subject adaptation\n\n"
        f"Support-inner decision: `{support.get('decision')}`. Later-query decision: `{final.get('route_decision')}`. Because the population base gate failed, no participant LoRA parameters were optimized and no later-query subject-adaptation comparison was run. The implemented rank-4 LoRA sits inside frozen U-Net ResBlock score convolutions; it is not the historical output-space or global transfer adapter. Its scientific effect remains untested.\n\n"
        "## Evidence boundary\n\nThe EB result is support/operator headroom; the three-fold GPU result is population-backbone development validity. Neither is confirmation evidence. The failure closes this population-backbone instance before personalization and is not a family-wide conclusion about diffusion, score-LoRA, or personalization.\n",
        encoding="utf-8",
    )
    summary = {"status":"completed_v8_finalization","eb_operator_headroom":eb.get("status"),"eb_deployable_headlines":{str(k):{"mean":float(v["mean"]),"ci_low":float(v["ci_low"]),"ci_high":float(v["ci_high"]),"positive_count":int(v["positive_count"]),"denominator":int(v["denominator"])} for k,v in eb_headlines.items()},"population_diffusion":base.get("decision"),"population_mean_rrmse":base_means,"population_gate_checks":{key:base.get(key) for key in ("analytic_roundtrip","single_window_fixed_overfit","mean_diff_pop_beats_raw_and_2of3","relative_to_det_within_10_percent","high_noise_scale_valid","natural_safety")},"support_inner":support.get("decision"),"later_query":final.get("route_decision"),"confirmation_evidence":False,"branch":"codex/sge-score-lora-v8"}
    _json(root / "result_summary.json", summary); _json(run_dir / "result_summary.json", summary); return summary


def run_stage(config_path: Path, stage: str, run_dir: Path, *, task_index: int = 0) -> dict[str, Any]:
    config = _config(config_path)
    if stage == "eb-headroom":
        return stage_eb_headroom(config, run_dir)
    if stage == "prepare-base":
        return stage_prepare_base(config, task_index, run_dir)
    if stage == "base-validity":
        return stage_base_validity(config, task_index, run_dir)
    if stage == "base-gate":
        return stage_base_gate(config, run_dir)
    if stage == "support-lora":
        return stage_support_lora(config, task_index, run_dir)
    if stage == "support-gate":
        return stage_support_gate(config, run_dir)
    if stage == "full-eval":
        return stage_full_eval(config, task_index, run_dir)
    if stage == "aggregate":
        return stage_aggregate(config, run_dir)
    if stage == "finalize":
        return stage_finalize(config, run_dir)
    raise ValueError(f"unknown v8 stage: {stage}")


__all__ = ["run_stage", "stage_eb_headroom", "_operator_distances", "_cluster_bootstrap"]
