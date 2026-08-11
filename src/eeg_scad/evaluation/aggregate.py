from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIRED_DIRECTIONS = {
    "rrmse_temporal": -1,
    "rrmse_spectral": -1,
    "correlation": 1,
    "snr_improvement": 1,
    "artifact_rmse": -1,
    "artifact_correlation": 1,
    "clean_output_rms_ratio": -1,
}
NATURAL_DIRECTIONS = {
    "heldout_eog_remaining_ratio": -1,
    "artifact_attenuation_db": 1,
    "preservation": 1,
    "psd_distortion": -1,
    "covariance_distortion": -1,
    "output_input_rms_ratio": -1,
    "observation_change_ratio": -1,
    "eeg_eog_coherence_reduction": 1,
    "frontal_residual_topography": -1,
    "erp_proxy": 1,
    "ssvep_proxy": 1,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def participant_metrics(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, int], panel: str) -> list[dict[str, Any]]:
    # window/pair -> session/task -> participant, independently within seed and method.
    unit: dict[tuple[str, str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        unit[(str(row["participant"]), str(row["session"]), str(row["task"]), int(row["seed"]), str(row["method"]))].append(row)
    task: dict[tuple[str, str, int, str], list[dict[str, float]]] = defaultdict(list)
    for (participant, _session, task_name, seed, method), values in unit.items():
        task[(participant, task_name, seed, method)].append({metric: float(np.mean([float(v[metric]) for v in values])) for metric in metrics})
    participant: dict[tuple[str, int, str], list[dict[str, float]]] = defaultdict(list)
    for (participant_id, _task, seed, method), values in task.items():
        participant[(participant_id, seed, method)].append({metric: float(np.mean([v[metric] for v in values])) for metric in metrics})
    result: list[dict[str, Any]] = []
    for (participant_id, seed, method), values in sorted(participant.items()):
        row: dict[str, Any] = {"panel": panel, "participant": participant_id, "seed": seed, "method": method}
        row.update({metric: float(np.mean([v[metric] for v in values])) for metric in metrics})
        result.append(row)
    return result


def _bootstrap(values: np.ndarray, seed: int, repetitions: int = 20000) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def method_summary(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    # Seeds are averaged within participant before the scientific n=15 summary.
    per_participant: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in metrics:
            per_participant[(str(row["participant"]), str(row["method"]), metric)].append(float(row[metric]))
    for (participant, method, metric), values in per_participant.items():
        grouped[(method, metric)].append({"participant": participant, "value": float(np.mean(values))})
    result = []
    for (method, metric), values in sorted(grouped.items()):
        vector = np.asarray([float(v["value"]) for v in values])
        lo, hi = _bootstrap(vector, 20260824 + sum(map(ord, method + metric)))
        result.append({"method": method, "metric": metric, "direction": metrics[metric], "participants": len(vector), "mean": float(np.mean(vector)), "median": float(np.median(vector)), "bootstrap_low": lo, "bootstrap_high": hi})
    return result


def effect_rows(rows: Sequence[Mapping[str, Any]], panel: str, primary_metric: str, direction: int) -> list[dict[str, Any]]:
    values: dict[tuple[str, int, str], float] = {(str(row["participant"]), int(row["seed"]), str(row["method"])): float(row[primary_metric]) for row in rows}
    contrasts = {
        "DET_MATCH_vs_POP": ("DET_MATCH", "DET_POP"),
        "DET_MATCH_vs_WRONG": ("DET_MATCH", "DET_WRONG"),
        "SCAD_MATCH_vs_POP": ("SCAD_MATCH", "SCAD_POP"),
        "SCAD_MATCH_vs_WRONG": ("SCAD_MATCH", "SCAD_WRONG"),
        "SCAD_K1_vs_DET1_MATCH": ("SCAD_MATCH", "DET_MATCH"),
    }
    result = []
    participants = sorted({key[0] for key in values})
    seeds = sorted({key[1] for key in values})
    for name, (match, comparator) in contrasts.items():
        for participant in participants:
            seed_effects = []
            for seed in seeds:
                if (participant, seed, match) in values and (participant, seed, comparator) in values:
                    seed_effects.append(direction * (values[(participant, seed, match)] - values[(participant, seed, comparator)]))
            if seed_effects:
                result.append({"panel": panel, "contrast": name, "metric": primary_metric, "participant": participant, "effect_positive_is_better": float(np.mean(seed_effects)), "seeds": len(seed_effects)})
    return result


def seed_summary(rows: Sequence[Mapping[str, Any]], primary_metric: str, direction: int) -> list[dict[str, Any]]:
    values = {(str(r["participant"]), int(r["seed"]), str(r["method"])): float(r[primary_metric]) for r in rows}
    result = []
    for seed in sorted({k[1] for k in values}):
        participants = sorted({k[0] for k in values if k[1] == seed})
        for name, match, comparator in (("MATCH_vs_POP", "SCAD_MATCH", "SCAD_POP"), ("MATCH_vs_WRONG", "SCAD_MATCH", "SCAD_WRONG"), ("SCAD_vs_DET", "SCAD_MATCH", "DET_MATCH")):
            vector = [direction * (values[(p, seed, match)] - values[(p, seed, comparator)]) for p in participants if (p, seed, match) in values and (p, seed, comparator) in values]
            if vector:
                result.append({"seed": seed, "contrast": name, "metric": primary_metric, "mean": float(np.mean(vector)), "median": float(np.median(vector)), "positive_count": int(np.sum(np.asarray(vector) > 0)), "participants": len(vector)})
    return result


def _effect_diagnosis(effects: Sequence[Mapping[str, Any]], contrast: str) -> dict[str, Any]:
    vector = np.asarray([float(r["effect_positive_is_better"]) for r in effects if r["contrast"] == contrast])
    if not len(vector):
        return {"status": "not_interpretable", "n": 0}
    lo, hi = _bootstrap(vector, 20260825 + sum(map(ord, contrast)))
    return {"n": len(vector), "mean": float(np.mean(vector)), "median": float(np.median(vector)), "positive_count": int(np.sum(vector > 0)), "bootstrap_low": lo, "bootstrap_high": hi}


def diagnose(paired_effects: Sequence[Mapping[str, Any]], natural_effects: Sequence[Mapping[str, Any]], sanity: Mapping[str, Any]) -> dict[str, Any]:
    paired_p = _effect_diagnosis(paired_effects, "SCAD_MATCH_vs_POP")
    paired_w = _effect_diagnosis(paired_effects, "SCAD_MATCH_vs_WRONG")
    natural_p = _effect_diagnosis(natural_effects, "SCAD_MATCH_vs_POP")
    natural_w = _effect_diagnosis(natural_effects, "SCAD_MATCH_vs_WRONG")
    diff = _effect_diagnosis(paired_effects, "SCAD_K1_vs_DET1_MATCH")
    if sanity.get("status") != "PASS":
        engineering = "invalid"
    else:
        engineering = "valid"
    positive_subject = [d.get("mean", 0) > 0 and d.get("median", 0) > 0 for d in (paired_p, paired_w, natural_p, natural_w)]
    if all(positive_subject): subject = "clear_development_signal"
    elif any(positive_subject): subject = "weak_or_heterogeneous_signal"
    elif any(d.get("mean", 0) < 0 for d in (paired_p, paired_w)): subject = "context_harmful"
    else: subject = "context_inert"
    if diff.get("mean", 0) > 0 and diff.get("positive_count", 0) >= 10: diffusion = "clear_development_signal"
    elif diff.get("mean", 0) > 0: diffusion = "small_signal"
    elif abs(diff.get("mean", 0)) < .002: diffusion = "deterministic_equivalent"
    else: diffusion = "deterministic_better"
    preservation = _effect_diagnosis(natural_effects, "SCAD_MATCH_vs_POP")
    tradeoff = "promising" if preservation.get("mean", 0) > 0 and natural_p.get("mean", 0) > 0 else "mixed"
    recommendation = "A. continue SCAD full development" if subject == "clear_development_signal" else "B. improve context representation" if engineering == "valid" else "F. revise baseline/reproduction first"
    return {
        "engineering_validity": engineering,
        "baseline_reproduction": {"EEGDfus": "architecture_reimplementation", "D4PM": "blocked_incomplete_release"},
        "subject_context_evidence": subject,
        "diffusion_incremental_value": diffusion,
        "natural_EEG_tradeoff": tradeoff,
        "next_step": recommendation,
        "paired_SCAD_MATCH_POP": paired_p,
        "paired_SCAD_MATCH_WRONG": paired_w,
        "natural_SCAD_MATCH_POP": natural_p,
        "natural_SCAD_MATCH_WRONG": natural_w,
        "paired_SCAD_K1_DET1": diff,
        "K8_vs_DET8": "not_tested",
        "development_only": True,
    }


def figures(root: Path, paired_summary: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]], natural_summary: Sequence[Mapping[str, Any]], latency: Sequence[Mapping[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    methods = ["DET_POP", "DET_MATCH", "DET_WRONG", "SCAD_POP", "SCAD_MATCH", "SCAD_WRONG"]
    means = {str(r["method"]): float(r["mean"]) for r in paired_summary if r["metric"] == "rrmse_temporal"}
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([m for m in methods if m in means], [means[m] for m in methods if m in means]); ax.set_ylabel("participant-first RRMSE"); ax.tick_params(axis="x", rotation=35); fig.tight_layout(); fig.savefig(root/"paired_method_comparison.png", dpi=180); plt.close(fig)
    groups = defaultdict(list)
    for row in effects: groups[str(row["contrast"])].append(float(row["effect_positive_is_better"]))
    fig, ax = plt.subplots(figsize=(8, 4)); names=list(groups); ax.boxplot([groups[n] for n in names], tick_labels=names, showmeans=True); ax.axhline(0,color="black",lw=.8); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("utility (positive better)"); fig.tight_layout(); fig.savefig(root/"context_effect_forest.png",dpi=180); fig.savefig(root/"context_swap_effects.png",dpi=180); plt.close(fig)
    remaining={str(r["method"]):float(r["mean"]) for r in natural_summary if r["metric"]=="heldout_eog_remaining_ratio"};pres={str(r["method"]):float(r["mean"]) for r in natural_summary if r["metric"]=="preservation"}
    fig,ax=plt.subplots(figsize=(6,5))
    for method in sorted(set(remaining)&set(pres)):ax.scatter(1-remaining[method],pres[method]);ax.annotate(method,(1-remaining[method],pres[method]),fontsize=7)
    ax.set_xlabel("artifact attenuation utility (1 - remaining)");ax.set_ylabel("preservation");fig.tight_layout();fig.savefig(root/"attenuation_preservation_scatter.png",dpi=180);plt.close(fig)
    if latency:
        fig,ax=plt.subplots(figsize=(6,5))
        for row in latency:ax.scatter(float(row["milliseconds_per_window"]),means.get(str(row["method"]),np.nan));ax.annotate(str(row["method"]),(float(row["milliseconds_per_window"]),means.get(str(row["method"]),np.nan)),fontsize=7)
        ax.set_xlabel("latency ms/window");ax.set_ylabel("paired RRMSE");fig.tight_layout();fig.savefig(root/"quality_latency_curve.png",dpi=180);plt.close(fig)


def engineering_figures(root: Path, result: Path, sanity: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    trace = list(sanity.get("trajectory", []))
    if trace:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([v["step"] for v in trace], [v["state_rms"] for v in trace], label="state")
        ax.plot([v["step"] for v in trace], [v["x0_rms"] for v in trace], label="predicted x0")
        ax.set_xlabel("reverse step"); ax.set_ylabel("RMS"); ax.legend(); fig.tight_layout(); fig.savefig(root/"diffusion_trajectory_rms.png", dpi=180); plt.close(fig)
    curves = []
    for method in ("det", "scad"):
        for path in sorted((result/method).glob("fold_*_seed_*.json")):
            value = json.loads(path.read_text())
            for point in value.get("curve", []): curves.append({"method":method,"step":point["step"],"validation":point["validation_artifact_mse"]})
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ("det", "scad"):
            steps=sorted({int(v["step"]) for v in curves if v["method"]==method})
            means=[np.mean([float(v["validation"]) for v in curves if v["method"]==method and int(v["step"])==step]) for step in steps]
            ax.plot(steps,means,label=method)
        ax.set_xlabel("update");ax.set_ylabel("mean validation artifact MSE");ax.legend();fig.tight_layout();fig.savefig(root/"training_curves.png",dpi=180);plt.close(fig)
    paired_effects=read_csv(result/"participant_effects.csv") if (result/"participant_effects.csv").is_file() else []
    if paired_effects:
        fig,ax=plt.subplots(figsize=(7,4))
        groups=defaultdict(list)
        for row in paired_effects:
            if row["panel"]=="paired":groups[row["contrast"]].append(float(row["effect_positive_is_better"]))
        names=list(groups)
        ax.boxplot([groups[name] for name in names],tick_labels=names,showmeans=True);ax.axhline(0,color="black",lw=.8);ax.tick_params(axis="x",rotation=35);ax.set_ylabel("paired utility");fig.tight_layout();fig.savefig(root/"participant_effects_forest.png",dpi=180);plt.close(fig)


def aggregate_all(derived: Path, result: Path, figure_root: Path, sanity: Mapping[str, Any]) -> dict[str, Any]:
    paired_raw=read_csv(derived/"metrics/paired_window_metrics.csv");natural_raw=read_csv(derived/"metrics/natural_window_metrics.csv")
    paired=participant_metrics(paired_raw,PAIRED_DIRECTIONS,"paired");natural=participant_metrics(natural_raw,NATURAL_DIRECTIONS,"natural")
    write_csv(result/"paired_evaluation/participant_metrics.csv",paired);write_csv(result/"natural_evaluation/participant_metrics.csv",natural)
    paired_summary=method_summary(paired,PAIRED_DIRECTIONS);natural_summary=method_summary(natural,NATURAL_DIRECTIONS);summary=paired_summary+natural_summary;write_csv(result/"method_summary.csv",summary)
    paired_effects=effect_rows(paired,"paired","rrmse_temporal",-1);natural_effects=effect_rows(natural,"natural","heldout_eog_remaining_ratio",-1);effects=paired_effects+natural_effects;write_csv(result/"participant_effects.csv",effects)
    seeds=seed_summary(paired,"rrmse_temporal",-1)+seed_summary(natural,"heldout_eog_remaining_ratio",-1);write_csv(result/"seed_effects.csv",seeds)
    latency=[]
    for path in sorted((derived/"predictions/paired").glob("*_latency.csv")):latency.extend(read_csv(path))
    by_method=defaultdict(list)
    for row in latency:by_method[str(row["method"])].append(row)
    latency_summary=[{"method":m,"milliseconds_per_window":float(np.mean([float(v["milliseconds_per_window"]) for v in rows])),"nfe":int(float(rows[0]["nfe"])),"peak_memory_mb":float(np.mean([float(v["peak_memory_mb"]) for v in rows]))} for m,rows in sorted(by_method.items())]
    write_csv(result/"latency_summary.csv",latency_summary)
    diagnosis=diagnose(paired_effects,natural_effects,sanity);(result/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    figures(figure_root,paired_summary,effects,natural_summary,latency_summary)
    engineering_figures(figure_root,result,sanity)
    return diagnosis
