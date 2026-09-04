#!/usr/bin/env python3
"""WAVE-6 analyzers for E2, E4 and E5 (CPU).

One committed analyzer per experiment, producing exactly the endpoints frozen in
reports/prereg_wave6_propagation_FROZEN.md — so the numbers are not produced by
ad-hoc inspection. Each writes a neutral JSON with its own QC block.

usage: x_analyze.py {e2,e4,e5}
"""
from __future__ import annotations

import argparse
import json
from itertools import product

import numpy as np

from pf_common import OUT, stat

WAVE6 = OUT / "wave6"


# --------------------------------------------------------------- shared utils

def _units(sub):
    rows, units = [], []
    for path in sorted((WAVE6 / sub).glob("fold_*_seed_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("complete"):
            rows.extend(payload["rows"])
            units.append({"file": path.name, "n_rows": payload.get("n_rows", len(payload["rows"])),
                          "n_unique_keys": payload.get("n_unique_keys")})
    return rows, units


def _operators():
    npz = np.load(WAVE6 / "e1_operators.npz", allow_pickle=False)
    keys = [str(k) for k in npz["cell_keys"]]
    probes = {str(k): npz["probe_" + str(k)] for k in npz["probe_keys"]}
    return {k: npz["eb"][i] for i, k in enumerate(keys)}, probes


def _spearman(x, y):
    if len(x) < 4:
        return float("nan")
    rx = np.argsort(np.argsort(np.asarray(x, float)))
    ry = np.argsort(np.argsort(np.asarray(y, float)))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _distance(ops, probes, cell_a, cell_b):
    a, b = ops.get(cell_a), ops.get(cell_b)
    if a is None or b is None:
        return None
    st = cell_a.split("|", 1)[1]
    probe = probes.get(st)
    delta = a - b
    out = {"matrix": float(np.linalg.norm(delta)),
           "direction": float(np.linalg.norm(
               a / max(np.linalg.norm(a), 1e-12) - b / max(np.linalg.norm(b), 1e-12))),
           "gain_log": float(abs(np.log(max(np.linalg.norm(a), 1e-12)
                                        / max(np.linalg.norm(b), 1e-12))))}
    out["probe"] = float(np.linalg.norm(delta @ probe)) if probe is not None else out["matrix"]
    return out


# ------------------------------------------------------------------------- E2

def e2() -> None:
    rows, units = _units("e2_units")
    if not rows:
        raise SystemExit("no complete E2 units")
    ops, probes = _operators()

    # participant-first means, per arm, per kind
    def per_participant(kind, metric):
        acc: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            if r["kind"] != kind:
                continue
            acc.setdefault((r["participant"], r["arm"]), []).append(r[metric])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    paired = per_participant("paired", "rrmse_temporal")
    natural = {m: per_participant("natural", m) for m in
               ("attenuation_db", "low_eog_observation_retention", "coherence_reduction")}

    participants = sorted({p for p, _ in paired})
    arms = sorted({a for _, a in paired})

    # donor cell for each (participant, arm) — read off the rows, never inferred
    donor_of: dict[tuple[str, str], set] = {}
    for r in rows:
        donor_of.setdefault((r["participant"], r["arm"]), set()).add(
            r["donor"] if "|" in str(r["donor"])
            else "|".join((str(r["donor"]), r["session"], r["task"])))

    # primary: within-recipient Spearman(D_probe, dR) over cross-participant donors
    per_recipient, deltas = {}, []
    for p in participants:
        own = paired.get((p, "OWN"))
        if own is None:
            continue
        pairs = []
        for a in arms:
            if not a.startswith("DONOR_"):
                continue
            value = paired.get((p, a))
            if value is None:
                continue
            cells = donor_of.get((p, a), set())
            dist = [_distance(ops, probes, "|".join((p,) + tuple(c.split("|")[1:])), c)
                    for c in cells]
            dist = [d for d in dist if d]
            if not dist:
                continue
            d_probe = float(np.mean([d["probe"] for d in dist]))
            d_dir = float(np.mean([d["direction"] for d in dist]))
            d_gain = float(np.mean([d["gain_log"] for d in dist]))
            delta = value - own
            pairs.append((d_probe, d_dir, d_gain, delta, a))
            deltas.append({"participant": p, "arm": a, "d_probe": d_probe,
                           "d_direction": d_dir, "d_gain": d_gain, "delta": delta})
        if len(pairs) >= 4:
            per_recipient[p] = {
                "rho_probe": _spearman([q[0] for q in pairs], [q[3] for q in pairs]),
                "rho_direction": _spearman([q[1] for q in pairs], [q[3] for q in pairs]),
                "rho_gain": _spearman([q[2] for q in pairs], [q[3] for q in pairs]),
                "n_donors": len(pairs), "own_rrmse": own,
                "mean_delta": float(np.mean([q[3] for q in pairs]))}

    # tertiles of D_probe, cut per recipient on calibration distance alone
    tertile = {}
    for name, index in (("near", 0), ("mid", 1), ("far", 2)):
        values = []
        for p in participants:
            mine = sorted([d for d in deltas if d["participant"] == p],
                          key=lambda d: d["d_probe"])
            if len(mine) < 3:
                continue
            chunk = np.array_split(mine, 3)[index]
            values.append(float(np.mean([c["delta"] for c in chunk])))
        tertile[name] = stat(values) if values else None

    # OWN_OTHER: identity held constant, compatibility varied (amendment W6-1a)
    own_other = {}
    for p in participants:
        own = paired.get((p, "OWN"))
        vals = [paired[(p, a)] - own for a in arms
                if a.startswith("OWN_OTHER_") and (p, a) in paired and own is not None]
        if vals:
            own_other[p] = float(np.mean(vals))
    near_stranger = {}
    for p in participants:
        mine = sorted([d for d in deltas if d["participant"] == p], key=lambda d: d["d_probe"])
        if mine:
            near_stranger[p] = float(np.mean([c["delta"] for c in mine[:max(1, len(mine) // 3)]]))
    common = sorted(set(own_other) & set(near_stranger))

    decision = {
        "frozen": "reports/prereg_wave6_propagation_FROZEN.md#3",
        "primary_rho_probe": stat([v["rho_probe"] for v in per_recipient.values()
                                   if np.isfinite(v["rho_probe"])]) if per_recipient else None,
        "rho_direction": stat([v["rho_direction"] for v in per_recipient.values()
                               if np.isfinite(v["rho_direction"])]) if per_recipient else None,
        "rho_gain": stat([v["rho_gain"] for v in per_recipient.values()
                          if np.isfinite(v["rho_gain"])]) if per_recipient else None,
        "per_recipient": per_recipient,
        "donor_tertiles_delta_rrmse": tertile,
        "own_other_minus_own": stat(list(own_other.values())) if own_other else None,
        "near_stranger_minus_own": stat(list(near_stranger.values())) if near_stranger else None,
        "near_stranger_minus_own_other": stat(
            [near_stranger[p] - own_other[p] for p in common]) if common else None,
        "arm_means_paired": {a: float(np.mean([paired[(p, a)] for p in participants
                                               if (p, a) in paired]))
                             for a in arms},
        "arm_means_natural": {m: {a: float(np.mean([d[(p, a)] for p in participants
                                                    if (p, a) in d]))
                                  for a in arms} for m, d in natural.items()},
        "qc": {"units": units, "n_rows": len(rows), "n_participants": len(participants),
               "n_arms": len(arms), "n_delta_points": len(deltas)},
    }
    (WAVE6 / "e2_results.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({k: decision[k] for k in
                      ("primary_rho_probe", "donor_tertiles_delta_rrmse",
                       "own_other_minus_own", "near_stranger_minus_own",
                       "near_stranger_minus_own_other")}, indent=1, default=float))


# ------------------------------------------------------------------------- E4

def e4() -> None:
    rows, units = _units("e4_units")
    if not rows:
        raise SystemExit("no complete E4 units")
    metric = "attenuation_db"
    by_window: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_window.setdefault((r["participant"], r["session"], r["task"], r["start"],
                              r["fold"], r["seed"]), {})[r["arm"]] = r

    points = []
    for unit, arms in by_window.items():
        if "A0_RAW" not in arms or "AT_RAW" not in arms:
            continue
        a0, at = arms["A0_RAW"], arms["AT_RAW"]
        points.append({"participant": unit[0], "elapsed_s": at["elapsed_s"],
                       "operator_displacement": at["operator_displacement"],
                       "activated_displacement": at["activated_displacement"],
                       "eog_rms": at["eog_rms"],
                       "delta_recent_minus_initial": at[metric] - a0[metric],
                       "delta_pop_minus_initial": (arms["POP"][metric] - a0[metric])
                       if "POP" in arms else None,
                       "delta_owneb_minus_initial": (arms["OWN_EB"][metric] - a0[metric])
                       if "OWN_EB" in arms else None})

    def per_participant(field, subset=None):
        acc: dict[str, list[float]] = {}
        for p in (subset if subset is not None else points):
            if p[field] is None:
                continue
            acc.setdefault(p["participant"], []).append(p[field])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    # strata by how much of the drift the current activity actually uses
    act = [p["activated_displacement"] for p in points]
    lo, hi = (float(np.quantile(act, 1 / 3)), float(np.quantile(act, 2 / 3))) if act else (0, 0)
    strata = {}
    for name, keep in (("low_activated", lambda v: v <= lo),
                       ("mid_activated", lambda v: lo < v <= hi),
                       ("high_activated", lambda v: v > hi)):
        subset = [p for p in points if keep(p["activated_displacement"])]
        vals = per_participant("delta_recent_minus_initial", subset)
        strata[name] = {"n_windows": len(subset),
                        "median_operator_displacement": float(np.median(
                            [p["operator_displacement"] for p in subset])) if subset else None,
                        "median_activated_displacement": float(np.median(
                            [p["activated_displacement"] for p in subset])) if subset else None,
                        "recent_minus_initial": stat(list(vals.values())) if vals else None}

    elapsed_bins = {}
    if points:
        edges = np.quantile([p["elapsed_s"] for p in points], [0, .25, .5, .75, 1.0])
        for i in range(4):
            subset = [p for p in points if edges[i] <= p["elapsed_s"] <= edges[i + 1]]
            vals = per_participant("delta_recent_minus_initial", subset)
            elapsed_bins[f"q{i}"] = {
                "elapsed_range_s": [float(edges[i]), float(edges[i + 1])],
                "n_windows": len(subset),
                "median_operator_displacement": float(np.median(
                    [p["operator_displacement"] for p in subset])) if subset else None,
                "recent_minus_initial": stat(list(vals.values())) if vals else None}

    overall = per_participant("delta_recent_minus_initial")
    pop = per_participant("delta_pop_minus_initial")
    decision = {
        "frozen": "reports/prereg_wave6_propagation_FROZEN.md#5",
        "metric": metric,
        "primary_recent_minus_initial": stat(list(overall.values())) if overall else None,
        "pop_minus_initial": stat(list(pop.values())) if pop else None,
        "by_activated_displacement": strata,
        "by_elapsed_time": elapsed_bins,
        "correlation_displacement_vs_effect": _spearman(
            [p["operator_displacement"] for p in points],
            [p["delta_recent_minus_initial"] for p in points]) if points else None,
        "correlation_activated_vs_effect": _spearman(
            [p["activated_displacement"] for p in points],
            [p["delta_recent_minus_initial"] for p in points]) if points else None,
        "qc": {"units": units, "n_rows": len(rows), "n_windows": len(points)},
        "note": "compare the displacement figures against the E1 R1 short-term "
                "repeatability band before calling any of this drift",
    }
    (WAVE6 / "e4_results.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({k: decision[k] for k in
                      ("primary_recent_minus_initial", "correlation_displacement_vs_effect",
                       "correlation_activated_vs_effect")}, indent=1, default=float))


# ------------------------------------------------------------------------- E5

def e5() -> None:
    rows, units = _units("e5_units")
    if not rows:
        raise SystemExit("no complete E5 units")
    metric = "attenuation_db"

    ratios = np.array([r["veog_rms"] / max(r["veog_rms"] + r["heog_rms"], 1e-12)
                       for r in rows if r["arm"] == "OWN_EB"])
    energies = np.array([r["eog_rms"] for r in rows if r["arm"] == "OWN_EB"])
    cuts = {"ratio_lo": float(np.quantile(ratios, 1 / 3)),
            "ratio_hi": float(np.quantile(ratios, 2 / 3)),
            "energy_low": float(np.quantile(energies, 1 / 4))}

    def event(r):
        ratio = r["veog_rms"] / max(r["veog_rms"] + r["heog_rms"], 1e-12)
        if r["eog_rms"] <= cuts["energy_low"]:
            return "LOW"
        return "V" if ratio >= cuts["ratio_hi"] else ("H" if ratio <= cuts["ratio_lo"] else "MIXED")

    def composition(arm):
        return arm.rsplit("_", 1)[0] if arm != "OWN_EB" else "OWN_EB"

    acc: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r["participant"], composition(r["arm"]), event(r)), []).append(r[metric])
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    participants = sorted({k[0] for k in means})

    grid = {}
    for comp, ev in product(("VHEAVY", "HHEAVY", "BALANCED", "OWN_EB"), ("V", "H", "MIXED", "LOW")):
        vals = [means[(p, comp, ev)] for p in participants if (p, comp, ev) in means]
        grid[f"{comp}|{ev}"] = {"mean": float(np.mean(vals)) if vals else None,
                                "n_participants": len(vals)}

    interaction = {}
    for p in participants:
        need = [(p, "VHEAVY", "V"), (p, "HHEAVY", "H"), (p, "VHEAVY", "H"), (p, "HHEAVY", "V")]
        if all(k in means for k in need):
            interaction[p] = (means[need[0]] + means[need[1]]
                              - means[need[2]] - means[need[3]])

    evenness = {}
    for comp in ("VHEAVY", "HHEAVY", "BALANCED"):
        spread = []
        for p in participants:
            vals = [means[(p, comp, e)] for e in ("V", "H") if (p, comp, e) in means]
            if len(vals) == 2:
                spread.append(abs(vals[0] - vals[1]))
        evenness[comp] = stat(spread) if spread else None

    decision = {
        "frozen": "prereg_wave6 amendment W6-1b",
        "metric": metric, "event_cuts": cuts,
        "composition_by_event_grid": grid,
        "primary_interaction": stat(list(interaction.values())) if interaction else None,
        "per_participant_interaction": interaction,
        "evenness_abs_V_minus_H": evenness,
        "achieved_composition": {
            arm: {"mean_ratio": float(np.mean([r.get("mean_ratio", np.nan) for r in rows
                                               if r["arm"] == arm and "mean_ratio" in r])),
                  "mean_energy": float(np.mean([r.get("mean_energy", np.nan) for r in rows
                                                if r["arm"] == arm and "mean_energy" in r]))}
            for arm in sorted({r["arm"] for r in rows if r["arm"] != "OWN_EB"})},
        "qc": {"units": units, "n_rows": len(rows), "n_participants": len(participants)},
    }
    (WAVE6 / "e5_results.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"primary_interaction": decision["primary_interaction"],
                      "evenness": evenness}, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["e2", "e4", "e5"])
    args = parser.parse_args()
    {"e2": e2, "e4": e4, "e5": e5}[args.which]()


if __name__ == "__main__":
    main()
