"""Participant-first V30 aggregation and absolute/relative interpretation."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


LOWER = {
    "rrmse_temporal", "rrmse_spectral", "artifact_rmse", "artifact_rrmse", "identity_change",
    "heldout_eog_remaining_ratio", "low_eog_observation_change", "psd_distortion",
    "covariance_distortion", "observation_change_ratio",
}


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(csv.DictReader(path.open()))
    return rows


def participant_first(rows: Iterable[Mapping[str, Any]], metrics: Iterable[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["panel"]), str(row["participant"]), str(row["method"]))].append(row)
    result = []
    for (panel, participant, method), values in sorted(groups.items()):
        record: dict[str, Any] = {"panel": panel, "participant": participant, "method": method}
        for metric in metrics:
            vector = np.asarray([float(row[metric]) for row in values if str(row.get(metric, "")) not in ("", "nan", "None")])
            record[metric] = float(vector.mean()) if len(vector) else np.nan
        result.append(record)
    return result


def bootstrap(values: np.ndarray, seed: int = 20260932) -> dict[str, Any]:
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": np.nan, "median": np.nan, "positive": 0, "participants": 0, "bootstrap_low": np.nan, "bootstrap_high": np.nan}
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    samples = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(20000)])
    return {"mean": float(values.mean()), "median": float(np.median(values)), "positive": int(np.sum(values > 0)), "participants": len(values), "bootstrap_low": float(np.quantile(samples, .025)), "bootstrap_high": float(np.quantile(samples, .975))}


def method_summary(rows: list[dict[str, Any]], metrics: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    for panel in sorted({row["panel"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["panel"] == panel}):
            chosen = [row for row in rows if row["panel"] == panel and row["method"] == method]
            for metric in metrics:
                result.append({"panel": panel, "method": method, "metric": metric, **bootstrap(np.asarray([row[metric] for row in chosen]))})
    return result


def contrast(rows: list[dict[str, Any]], first: str, second: str, metric: str) -> list[dict[str, Any]]:
    values = {(row["participant"], row["method"]): float(row[metric]) for row in rows if np.isfinite(float(row[metric]))}
    participants = sorted({participant for participant, method in values if method == first} & {participant for participant, method in values if method == second})
    sign = -1 if metric in LOWER else 1
    return [{"participant": participant, "first": first, "second": second, "metric": metric, "effect": sign * (values[participant, first] - values[participant, second])} for participant in participants]


def classify(
    paired_summary: list[dict[str, Any]],
    natural_summary: list[dict[str, Any]],
    donor_group: list[dict[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    def value(rows, method, metric):
        match = next((row for row in rows if row["method"] == method and row["metric"] == metric), None)
        return np.nan if match is None else float(match["mean"])
    candidate = str(selection["selected_candidate"])
    if candidate == "none":
        return {
            "engineering": selection["engineering"],
            "correct_context_specificity": selection["correct_context_specificity"],
            "absolute_paired_denoising": selection["absolute_paired_denoising"],
            "absolute_natural_artifact": selection["absolute_natural_artifact"],
            "observation_retention": selection["observation_retention"],
            "revision_readiness": selection["revision_readiness"],
            "selected_candidate": "none",
            "next_route": selection["next_route"],
            "selection_rationale": selection["rationale"],
            "development_only": True,
        }
    specificity = next((row for row in donor_group if row["method"] == candidate), None)
    paired = value(paired_summary, candidate, "rrmse_temporal")
    standard = value(paired_summary, "STANDARD", "rrmse_temporal")
    remaining = value(natural_summary, candidate, "heldout_eog_remaining_ratio")
    retention = value(natural_summary, candidate, "low_eog_observation_retention")
    engineering = "valid"
    correct = "mixed" if specificity is None else "clear" if specificity["correct_top3"] >= 12 and specificity["mean_correct_minus_mean_wrong"] > 0 else "weak" if specificity["mean_correct_minus_mean_wrong"] > 0 else "absent"
    absolute_paired = "weak" if not np.isfinite(paired) else "strong" if paired < .65 else "competitive" if paired < standard - .005 else "near_standard" if paired <= standard + .005 else "weak"
    absolute_natural = "mixed" if not np.isfinite(remaining) else "attenuating" if remaining < 1 else "no_attenuation" if remaining < 1.05 else "harmful"
    observation = "mixed" if not np.isfinite(retention) else "acceptable" if retention >= .95 else "concern"
    readiness = str(selection["revision_readiness"])
    return {"engineering": engineering, "correct_context_specificity": correct, "absolute_paired_denoising": absolute_paired, "absolute_natural_artifact": absolute_natural, "observation_retention": observation, "revision_readiness": readiness, "selected_candidate": candidate, "next_route": selection["next_route"], "selection_rationale": selection["rationale"], "development_only": True}


__all__ = ["LOWER", "bootstrap", "classify", "contrast", "method_summary", "participant_first", "read_rows"]
