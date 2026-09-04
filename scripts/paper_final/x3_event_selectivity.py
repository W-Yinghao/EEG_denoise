#!/usr/bin/env python3
"""WAVE-6 E3 — is the value of a propagation difference selective to the eye
movement that is happening right now? (CPU; reads the E2 donor sweep)

Frozen design: reports/prereg_wave6_propagation_FROZEN.md section 4.

Two layers of evidence, as the plan requires, because the paired data are built
by injecting `A_gen e` and an operator-difference effect there is partly
structural:

  layer 1 (controlled restoration)  the E2 sweep, stratified by event type
  layer 2 (natural prediction)      the earlier calibration operator predicts a
        spatial pattern for future vertical / horizontal eye activity; that
        pattern is then estimated INDEPENDENTLY from the future natural EEG and
        compared.  The observed side never uses `A e` - it is a fresh regression
        of recorded EEG on recorded EOG in windows the calibration never saw.

Outputs results/paper_final/wave6/e3_event_selectivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from pf_common import OUT, stat

V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
WAVE6 = OUT / "wave6"
E2_UNITS = WAVE6 / "e2_units"
NATURAL_METRIC = "attenuation_db"       # higher is more ocular activity removed
PAIRED_METRIC = "rrmse_temporal"        # lower is better


def _load_e2():
    rows = []
    for path in sorted(E2_UNITS.glob("fold_*_seed_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("complete"):
            rows.extend(payload["rows"])
    return rows


def _labels(rows, train_participants):
    """Event labels; cut points fixed on TRAINING participants only."""
    natural = [r for r in rows if r["kind"] == "natural" and r["arm"] == "OWN"]
    train = [r for r in natural if r["participant"] in train_participants]
    pool = train or natural            # fall back only if no training rows exist
    ratios = np.array([r["veog_rms"] / max(r["veog_rms"] + r["heog_rms"], 1e-12)
                       for r in pool])
    energies = np.array([r["eog_rms"] for r in pool])
    cuts = {"ratio_lo": float(np.quantile(ratios, 1 / 3)),
            "ratio_hi": float(np.quantile(ratios, 2 / 3)),
            "energy_low": float(np.quantile(energies, 1 / 4)),
            "source": "train" if train else "all(no train rows in E2 units)",
            "n_cut_rows": len(pool)}

    def label(row):
        ratio = row["veog_rms"] / max(row["veog_rms"] + row["heog_rms"], 1e-12)
        if row["eog_rms"] <= cuts["energy_low"]:
            return "LOW"
        if ratio >= cuts["ratio_hi"]:
            return "V"
        if ratio <= cuts["ratio_lo"]:
            return "H"
        return "MIXED"

    return cuts, label


def _operator_bank():
    npz = np.load(WAVE6 / "e1_operators.npz", allow_pickle=False)
    keys = [str(k) for k in npz["cell_keys"]]
    return {k: npz["eb"][i] for i, k in enumerate(keys)}


def _donor_direction_class(ops, recipient_cell, donor_cell):
    """Which column of (A_i - A_j) carries the difference? Calibration only."""
    a, b = ops.get(recipient_cell), ops.get(donor_cell)
    if a is None or b is None:
        return None, None
    delta = a - b
    v = float(np.linalg.norm(delta[:, 0]))
    h = float(np.linalg.norm(delta[:, 1]))
    return ("V" if v >= h else "H"), v / max(v + h, 1e-12)


def _natural_coupling(registry30, data, key, up):
    """Independent coupling estimated from the FUTURE natural EEG.

    Regresses the recorded EEG of the evaluation windows on the recorded EOG of
    those same windows. Nothing about the earlier calibration enters here, so
    comparing it with the calibration operator is a genuine prediction test and
    not an algebraic identity.
    """
    windows = list(up._natural_windows(registry30, data, key))
    if not windows:
        return None
    y = np.concatenate([w[1] for w in windows], axis=1)
    e = np.concatenate([w[2] for w in windows], axis=1)
    gram = e @ e.T
    ridge = 1e-6 * float(np.trace(gram)) / gram.shape[0]
    return (y @ e.T) @ np.linalg.inv(gram + ridge * np.eye(gram.shape[0]))


def _cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def layer2(ops):
    """Does the earlier calibration predict the LATER natural coupling, and does
    it predict the owner's better than a stranger's?"""
    sys.path.insert(0, str(V44_SRC))
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli import run_v44 as up
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry

    data, folds, _ = configs()
    observed = {}
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        for key in sorted(registry30.cells):
            if key[0] not in fold["test"]:
                continue
            coupling = _natural_coupling(registry30, data, key, up)
            if coupling is not None:
                observed["|".join(key)] = coupling
    rows = []
    for cell, obs in sorted(observed.items()):
        own = ops.get(cell)
        if own is None:
            continue
        session_task = cell.split("|", 1)[1]
        others = [c for c in ops if c != cell and c.split("|", 1)[1] == session_task]
        for column, name in ((0, "vertical"), (1, "horizontal")):
            own_cos = _cosine(own[:, column], obs[:, column])
            stranger = [_cosine(ops[c][:, column], obs[:, column]) for c in others]
            rows.append({"cell": cell, "participant": cell.split("|")[0],
                         "column": name, "own_cosine": own_cos,
                         "stranger_cosine_mean": float(np.mean(stranger)) if stranger else float("nan"),
                         "own_minus_stranger": own_cos - float(np.mean(stranger)) if stranger else float("nan"),
                         "own_rank_among": int(sum(s > own_cos for s in stranger)),
                         "n_strangers": len(stranger)})
    summary = {}
    for name in ("vertical", "horizontal"):
        vals = [r["own_minus_stranger"] for r in rows
                if r["column"] == name and np.isfinite(r["own_minus_stranger"])]
        per: dict[str, list[float]] = {}
        for r in rows:
            if r["column"] == name and np.isfinite(r["own_minus_stranger"]):
                per.setdefault(r["participant"], []).append(r["own_minus_stranger"])
        summary[name] = {
            "cells": len(vals),
            "own_minus_stranger_by_participant": stat(
                [float(np.mean(v)) for v in per.values()]) if per else None,
            "own_cosine_median": float(np.median(
                [r["own_cosine"] for r in rows if r["column"] == name])),
        }
    return {"rows": rows, "summary": summary}


def main() -> None:
    sys.path.insert(0, str(V44_SRC))
    from eeg_scad.cli.run_v43 import configs

    data, folds, _ = configs()
    rows = _load_e2()
    if not rows:
        raise SystemExit("no complete E2 units yet — run x2_donor_swap.py first")
    ops = _operator_bank()
    test_of_fold = {i: set(f["test"]) for i, f in enumerate(folds)}
    train_participants = {p for f in folds for p in f["train"]}
    cuts, label = _labels(rows, train_participants)

    # ---------- layer 1: the E2 sweep stratified by event type ----------------
    own_by = {}
    for r in rows:
        if r["kind"] == "natural" and r["arm"] == "OWN":
            own_by[(r["fold"], r["seed"], r["participant"], r["session"],
                    r["task"], r["start"])] = r[NATURAL_METRIC]
    graded = []
    for r in rows:
        if r["kind"] != "natural" or not r["arm"].startswith("DONOR_"):
            continue
        unit = (r["fold"], r["seed"], r["participant"], r["session"], r["task"], r["start"])
        if unit not in own_by:
            continue
        recipient_cell = "|".join((r["participant"], r["session"], r["task"]))
        donor_cell = "|".join((r["donor"], r["session"], r["task"]))
        direction, share = _donor_direction_class(ops, recipient_cell, donor_cell)
        if direction is None:
            continue
        graded.append({"participant": r["participant"], "event": label(r),
                       "donor_direction": direction, "direction_share": share,
                       "eog_rms": r["eog_rms"],
                       "delta": r[NATURAL_METRIC] - own_by[unit]})

    # energy matching: keep only windows in the middle energy band shared by the
    # V-dominant and H-dominant strata, so the interaction cannot be an energy effect
    v_e = [g["eog_rms"] for g in graded if g["event"] == "V"]
    h_e = [g["eog_rms"] for g in graded if g["event"] == "H"]
    band = None
    if v_e and h_e:
        lo = max(np.quantile(v_e, .1), np.quantile(h_e, .1))
        hi = min(np.quantile(v_e, .9), np.quantile(h_e, .9))
        band = [float(lo), float(hi)]
    matched = [g for g in graded
               if band and band[0] <= g["eog_rms"] <= band[1]] if band else graded

    def cell_mean(subset, event, direction):
        per: dict[str, list[float]] = {}
        for g in subset:
            if g["event"] == event and g["donor_direction"] == direction:
                per.setdefault(g["participant"], []).append(g["delta"])
        return {p: float(np.mean(v)) for p, v in per.items()}

    interaction_per_participant = {}
    for subset, tag in ((matched, "energy_matched"), (graded, "all_windows")):
        cells = {(e, d): cell_mean(subset, e, d)
                 for e in ("V", "H") for d in ("V", "H")}
        common = set.intersection(*[set(c) for c in cells.values()]) if all(cells.values()) else set()
        values = {p: (cells[("V", "V")][p] + cells[("H", "H")][p]
                      - cells[("H", "V")][p] - cells[("V", "H")][p]) for p in sorted(common)}
        interaction_per_participant[tag] = {
            "per_participant": values,
            "summary": stat(list(values.values())) if values else None,
            "cell_means": {f"event{e}_donor{d}": (float(np.mean(list(cells[(e, d)].values())))
                                                  if cells[(e, d)] else None)
                           for e in ("V", "H") for d in ("V", "H")},
            "n_participants": len(values)}

    # ---------- secondary: energy first, then direction --------------------
    def variance_explained(subset, keys):
        if len(subset) < 20:
            return None
        design = np.column_stack([np.ones(len(subset))]
                                 + [np.asarray([g[k] for g in subset], float) for k in keys])
        target = np.asarray([g["delta"] for g in subset], float)
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ beta
        total = float(np.var(target)) or 1.0
        return float(1 - np.var(residual) / total)

    energy_only = variance_explained(matched, ["eog_rms"])
    plus_direction = variance_explained(matched, ["eog_rms", "direction_share"])

    strata = {}
    for event in ("V", "H", "MIXED", "LOW"):
        vals = [g["delta"] for g in graded if g["event"] == event]
        strata[event] = {"n": len(vals),
                         "mean_delta": float(np.mean(vals)) if vals else None,
                         "median_eog_rms": float(np.median(
                             [g["eog_rms"] for g in graded if g["event"] == event]))
                         if vals else None}

    decision = {
        "frozen": "reports/prereg_wave6_propagation_FROZEN.md#4",
        "event_cuts": cuts, "event_strata": strata,
        "energy_match_band": band,
        "layer1_interaction": interaction_per_participant,
        "layer1_variance_explained": {"energy_only": energy_only,
                                      "energy_plus_direction": plus_direction},
        "layer2_natural_prediction": layer2(ops),
        "n_e2_rows": len(rows), "n_graded": len(graded), "n_matched": len(matched),
        "note": "layer 1 is a controlled-restoration check (the paired corpus is "
                "built by injecting A_gen e); layer 2 is the load-bearing "
                "prediction test on recorded EEG the calibration never saw",
    }
    WAVE6.mkdir(parents=True, exist_ok=True)
    (WAVE6 / "e3_event_selectivity.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"interaction": interaction_per_participant["energy_matched"]["summary"],
                      "layer2": decision["layer2_natural_prediction"]["summary"],
                      "ve": decision["layer1_variance_explained"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
