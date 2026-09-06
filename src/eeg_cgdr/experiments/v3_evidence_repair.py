"""Non-destructive correction of v3 route names and comparator semantics."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _number(row: Mapping[str, Any], metric: str) -> float | None:
    try:
        value = float(row[metric])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def run(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    v3 = Path(str(config["v3_result_root"]))
    source = _read_csv(v3 / "aggregate/unit_metrics.csv")
    output = CODE_ROOT / str(config["v3_repair_root"])
    route_names = {
        "P_A_RAW_SUPPORT_TOKENS": "support_moment_summary_FiLM",
        "P_B_DIRECT_SUPPORT_ADAPTER": "support_fitted_output_space_residual_adapter",
        "P_D_SUPPORT_STAT_CONTROL": "inference_only_normalization_OOD_hybrid",
    }
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in source:
        if str(row.get("status", "")).startswith("success"):
            grouped[(row["route"], row["dataset"], row["unit_id"])][row["method"]] = row
    for (route, dataset, unit), methods in sorted(grouped.items()):
        match = methods.get("DIFF-MATCH")
        if match is None:
            continue
        metric = "clean_waveform_RRMSE" if dataset == "klados" else "eog_coherence_reduction"
        sign = -1.0 if dataset == "klados" else 1.0
        comparators = {
            "MATCH_minus_POP_EXEMPLAR": methods.get("DIFF-POP"),
            "MATCH_minus_NO_SUPPORT": methods.get("DIFF-NO-SUPPORT"),
            "MATCH_minus_STRONG_POP": methods.get("STRONG-POP"),
        }
        wrong = [methods.get(f"DIFF-WRONG-{index}") for index in (1, 2, 3)]
        match_value = _number(match, metric)
        for estimand, comparator in comparators.items():
            right = _number(comparator or {}, metric)
            if match_value is not None and right is not None:
                rows.append({
                    "route_original": route, "route_corrected": route_names.get(route, route),
                    "dataset": dataset, "unit_id": unit, "estimand": estimand,
                    "metric": metric, "utility": sign * (match_value - right),
                    "scientific_ranking_valid": route != "P_D_SUPPORT_STAT_CONTROL",
                })
        wrong_values = [_number(value or {}, metric) for value in wrong]
        wrong_values = [value for value in wrong_values if value is not None]
        if match_value is not None and wrong_values:
            rows.append({
                "route_original": route, "route_corrected": route_names.get(route, route),
                "dataset": dataset, "unit_id": unit, "estimand": "MATCH_minus_mean_WRONG",
                "metric": metric, "utility": sign * (match_value - float(np.mean(wrong_values))),
                "scientific_ranking_valid": route != "P_D_SUPPORT_STAT_CONTROL",
            })
    _write_csv(output / "corrected_paired_effects.csv", rows)
    summaries: list[dict[str, Any]] = []
    for key in sorted({(row["route_corrected"], row["dataset"], row["estimand"]) for row in rows}):
        values = np.asarray([float(row["utility"]) for row in rows
                             if (row["route_corrected"], row["dataset"], row["estimand"]) == key])
        summaries.append({
            "route": key[0], "dataset": key[1], "estimand": key[2], "units": int(values.size),
            "mean_utility": float(values.mean()), "median_utility": float(np.median(values)),
            "positive_count": int(np.sum(values > 0)),
            "scientific_ranking_valid": key[0] != "inference_only_normalization_OOD_hybrid",
        })
    _write_csv(output / "corrected_effect_summary.csv", summaries)
    summary = {
        "status": "completed_non_destructive_v3_evidence_correction",
        "historical_outputs_modified": False,
        "route_corrections": {
            "P-A": "support moment-summary FiLM; not raw waveform tokens or cross-attention",
            "P-B": "support-fitted output-space residual adapter; not LoRA/internal adapter",
            "P-D": "invalid_for_scientific_route_ranking; OOD_robustness_diagnostic_only",
        },
        "old_pop_comparator": "POP-EXEMPLAR_four_training_windows_not_population_no_support",
        "corrected_estimands": ["MATCH_minus_NO_SUPPORT", "MATCH_minus_STRONG_POP", "MATCH_minus_mean_WRONG"],
        "accurate_scope": (
            "The tested moment-summary FiLM, output residual adapter, and OOD normalization hybrid do not advance; "
            "raw temporal support, internal LoRA adaptation, and the subject-aware diffusion family remain untested."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = CODE_ROOT / "reports/v3_evidence_correction.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# v3 evidence correction\n\n"
        "Historical files were preserved. P-A is **support moment-summary FiLM**, not a raw waveform/token route. "
        "P-B is a **support-fitted output-space residual adapter**, not LoRA. P-D is an inference-only normalization "
        "OOD hybrid and is excluded from scientific route ranking.\n\n"
        "The former POP arm used four training-window exemplars. It is now named `POP-EXEMPLAR`; fair effects are "
        "reported separately against `NO-SUPPORT`, `STRONG-POP`, and the mean of three WRONG donors. Therefore the "
        "published v3 P-A Klados +0.1141 effect is not a fair population subject-utility estimate.\n\n"
        "The narrow corrected conclusion is that these three concrete instances do not advance. This does not test "
        "the raw temporal-support route, internal LoRA, or the diffusion/personalization families.\n",
        encoding="utf-8",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary

