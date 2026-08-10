"""Participant-first closure of the frozen BCI2b raw-support experiment."""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _exact(values: np.ndarray, *, one_sided: bool) -> float:
    observed = float(np.mean(values))
    means = np.asarray([np.mean(values * np.asarray(signs)) for signs in itertools.product((-1, 1), repeat=len(values))])
    if one_sided:
        return float(np.mean(means >= observed - 1e-15))
    return float(np.mean(np.abs(means) >= abs(observed) - 1e-15))


def _summary(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "positive": int(np.sum(x > 0)), "n": int(len(x)),
        "one_sided_exact_sign_flip": _exact(x, one_sided=True),
        "two_sided_exact_sign_flip": _exact(x, one_sided=False),
        "participant_values": [float(v) for v in x],
    }


def aggregate(source_root: Path, output_root: Path) -> dict[str, Any]:
    paired = _read(source_root / "raw_support_models" / "paired_metrics.csv")
    natural = _read(source_root / "raw_support_models" / "natural_safety.csv")
    seeds = sorted({int(row["seed"]) for row in paired})
    if seeds != [20260808]:
        raise RuntimeError(f"closure must use frozen one-seed outputs, got {seeds}")
    effects: list[dict[str, Any]] = []
    donor_pairs: list[dict[str, Any]] = []
    donor_lodo: list[dict[str, Any]] = []
    for k in (1, 8):
        for participant in range(1, 10):
            rows = [row for row in paired if int(row["participant"]) == participant]
            by_method: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                by_method[row["method"]].append(float(row["rrmse"]))
            match_name = f"DIFF-CLEAN-MATCH-K{k}"
            pop_name = f"DIFF-CLEAN-POP-K{k}"
            match = float(np.mean(by_method[match_name])); pop = float(np.mean(by_method[pop_name]))
            wrong_names = []
            for name in sorted(by_method):
                prefix, suffix = "DIFF-CLEAN-WRONG-", f"-K{k}"
                if name.startswith(prefix) and name.endswith(suffix):
                    donor = name[len(prefix):-len(suffix)]
                    if donor.isdigit():
                        wrong_names.append(name)
            # These are raw EEG support-set donors. No operator/shrinkage WRONG aliases are admitted.
            donor_values = []
            for name in wrong_names:
                donor = name.removeprefix("DIFF-CLEAN-WRONG-").removesuffix(f"-K{k}")
                wrong = float(np.mean(by_method[name])); utility = wrong - match; donor_values.append((donor, wrong, utility))
                donor_pairs.append({"K": k, "participant": participant, "donor": donor, "wrong_rrmse": wrong, "match_rrmse": match, "utility": utility, "win": int(utility > 0), "raw_support_donor": 1})
            utilities = np.asarray([value[2] for value in donor_values], dtype=float)
            for excluded, _, _ in donor_values:
                kept = [value for donor, _, value in donor_values if donor != excluded]
                donor_lodo.append({"K": k, "participant": participant, "excluded_donor": excluded, "utility": float(np.mean(kept)) if kept else float("nan"), "remaining_donors": len(kept)})
            det = float(np.mean(by_method["DET-CLEAN-MATCH"]))
            effects.append({
                "K": k, "participant": participant,
                "U_P": pop - match,
                "U_W_donor_mean": float(np.mean(utilities)),
                "U_W_donor_median": float(np.median(utilities)),
                "DET_minus_DIFF": det - match,
                "K8_minus_K1": float("nan"),
                "donors": len(donor_values),
            })
    k1 = {int(row["participant"]): row for row in effects if int(row["K"]) == 1}
    k8 = {int(row["participant"]): row for row in effects if int(row["K"]) == 8}
    for participant in range(1, 10):
        # Protocol-first mean is required; compute from every protocol row, not the first row.
        match1 = float(np.mean([float(row["rrmse"]) for row in paired if int(row["participant"]) == participant and row["method"] == "DIFF-CLEAN-MATCH-K1"]))
        match8 = float(np.mean([float(row["rrmse"]) for row in paired if int(row["participant"]) == participant and row["method"] == "DIFF-CLEAN-MATCH-K8"]))
        k1[participant]["K8_minus_K1"] = match1 - match8
        k8[participant]["K8_minus_K1"] = match1 - match8
    summaries = {}
    for k in (1, 8):
        take = [row for row in effects if int(row["K"]) == k]
        summaries[f"K{k}"] = {key: _summary([float(row[key]) for row in take]) for key in ("U_P", "U_W_donor_mean", "U_W_donor_median", "DET_minus_DIFF")}
    summaries["K8_minus_K1"] = _summary([float(k8[p]["K8_minus_K1"]) for p in range(1, 10)])
    donor_summary = {
        f"K{k}": {
            "donor_recipient_pairs": len([row for row in donor_pairs if int(row["K"]) == k]),
            "pair_win_rate": float(np.mean([int(row["win"]) for row in donor_pairs if int(row["K"]) == k])),
            "mean_leave_one_donor_out": float(np.nanmean([float(row["utility"]) for row in donor_lodo if int(row["K"]) == k])),
        } for k in (1, 8)
    }
    natural_participant = []
    for method in ("DIFF-CLEAN-POP-K8", "DIFF-CLEAN-MATCH-K8"):
        for participant in range(1, 10):
            take = [row for row in natural if row["method"] == method and int(row["participant"]) == participant]
            natural_participant.append({"method": method, "participant": participant, **{key: float(np.mean([float(row[key]) for row in take])) for key in ("eog_attenuation", "preservation", "mi_band_distortion", "covariance", "mi_kappa", "erd_preservation")}})
    labels = [
        "SUPPORT_EOG_ASSISTED_RAW_TEMPORAL_CONTEXT",
        "MATCH_OVER_STRONG_POP_NOT_ESTABLISHED",
        "DONOR_SPECIFICITY_SUGGESTIVE",
        "MULTISAMPLE_AVERAGING_GAIN_PRESENT",
        "DIFFUSION_OVER_COMPUTE_MATCHED_DETERMINISTIC_NOT_TESTED",
        "RELATIVE_MATCH_VS_POP_SAFETY_PASSED",
        "ABSOLUTE_NATURAL_SAFETY_NOT_ESTABLISHED",
    ]
    result = {
        "labels": labels, "effects": summaries, "donor_robustness": donor_summary,
        "availability": {"eligible_protocol_units": 26, "denominator": 27, "participants": 9},
        "aggregation": "protocol -> participant; n=9", "training_seeds": seeds,
        "resume_validity": "NOT_TESTED: checkpoint fields and reload equality do not establish interrupted-training resume equality",
        "historical_results_overwritten": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "participant_effects_k1_k8.csv", effects)
    _write_csv(output_root / "raw_donor_recipient_effects.csv", donor_pairs)
    _write_csv(output_root / "leave_one_donor_out.csv", donor_lodo)
    _write_csv(output_root / "participant_natural_safety.csv", natural_participant)
    (output_root / "result_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "route_decision.json").write_text(json.dumps({"labels": labels, "development_only": True, "additional_training": False}, indent=2) + "\n", encoding="utf-8")
    return result


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = ["# BCI2b raw-support clean diffusion: final scientific closure", "", "This is a no-training participant-first reaggregation of frozen outputs. The route is support-EOG-assisted because EOG was used to exclude gross-artifact support patches; query inference itself receives EEG only.", "", "## Frozen labels", ""]
    lines.extend(f"- `{label}`" for label in result["labels"])
    lines += ["", "## Effects", ""]
    for panel, metrics in result["effects"].items():
        if panel == "K8_minus_K1":
            lines.append(f"- {panel}: {metrics['mean']:+.5f} mean, {metrics['median']:+.5f} median, {metrics['positive']}/9.")
            continue
        for name, values in metrics.items():
            lines.append(f"- {panel} {name}: {values['mean']:+.5f} mean, {values['median']:+.5f} median, {values['positive']}/9, two-sided exact p={values['two_sided_exact_sign_flip']:.6f}.")
    lines += ["", "WRONG includes only actual raw-support donor arms. Donors are scored separately before recipient-level mean/median aggregation; donor-recipient pair wins and leave-one-donor-out sensitivity are saved separately.", "", "Checkpoint raw/EMA/optimizer/RNG fields exist and deterministic reload was tested, but interrupted-training resume equality was not actually executed. No full resume-validity claim is made.", "", "The relative MATCH-versus-POP safety margins passed, while absolute natural safety was not established. This development closure neither changes the historical model nor supports a family-wide conclusion."]
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines) + "\n", encoding="utf-8")
