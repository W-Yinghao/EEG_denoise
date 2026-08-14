"""V44-S2: ownership guard, honest re-baselining, online refinement.

Inference-only on the frozen V44-S1 checkpoints.  Rules frozen in the V44-S2
addendum of reports/v44_preregistration.md.  The guard's Mahalanobis metric is
the U0-b EB posterior variance (recomputed entrywise on this branch from the
same hierarchy); its threshold is the 95th percentile of own-operator
split-half scores (support halves only — no query outcomes).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_scad.cli.run_v43 import bootstrap_draws, configs, holm
from eeg_scad.cli.run_v44 import (ROOT, RESULT, REPORT, S1_SEEDS, _gated_assets,
                                  _natural_metrics_full, _natural_windows, _participant_means,
                                  _stat, natural_noise_seed, noise_seed, sample_bank_eog)
from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler, TransferRegistry,
                                                  bipolar_eog, ridge_transfer)
from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
from eeg_scad.data.v24_coordinate_contract import robust_center_scale
from eeg_scad.evaluation.paired_metrics import paired_metrics


TV_PRIMARY = 30
TV_SENSITIVITY = (10, 60)
RLS_GRID = (10, 30, 60, 120, 240)
RLS_FORGET = 0.999
RLS_FORGET_SENS = 0.99
OG1_MARGIN = 0.005
OG1_DETECTION = 0.90
OG2_FALSE_ALARM = 0.10
OG2_COST = 0.005
ORACLE_GAP_S1 = 0.24413


# ------------------------------------------------------------ guard machinery

def _posterior_variance(registry30, eb120, fold) -> dict[tuple, np.ndarray]:
    """Entrywise EB posterior variance of the gated operator per cell (U0-b)."""
    groups: dict[tuple, np.ndarray] = {}
    for group in registry30.population_transfer:
        train_full = np.stack([eb120.cells[key].transfer for key in sorted(eb120.cells)
                               if key[0] in fold["train"] and key[1:] == group])
        groups[group] = train_full.var(axis=0, ddof=1).clip(1e-12)
    out = {}
    for key, cell in eb120.cells.items():
        if key[1:] not in groups:
            continue
        _, blocks, _, _ = eb120._fit(key, 12000, 100)
        v = np.stack(blocks).var(axis=0, ddof=1).clip(1e-12)
        tau2 = groups[key[1:]]
        out[key] = 1.0 / (1.0 / tau2 + len(blocks) / v)
    return out


def _score(c_pres: np.ndarray, c_probe: np.ndarray, post_var: np.ndarray) -> float:
    return float(np.sqrt(np.mean((c_pres - c_probe) ** 2 / post_var)))


def _probe_operator(registry30, key, t_v: int, start: int) -> np.ndarray:
    """Deployment-legal probe: ridge fit on the first t_v seconds of the query
    stream, using the recipient's support normalization (V41R contract)."""
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    cell = registry30.cells[key]
    stop = start + t_v * 100
    latent = (eog[:, start:stop] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
    scaled = eeg[:, start:stop] / registry30.eeg_scale[:, None]
    return ridge_transfer(scaled, latent, registry30.ridge_ratio)[0]


def _split_half_threshold(registry30, eb120, post_var) -> float:
    scores = []
    for key in sorted(eb120.cells):
        if key not in post_var:
            continue
        eeg, eye, names = registry30._load(*key)
        eog = bipolar_eog(eye, names)
        cell = registry30.cells[key]
        halves = []
        for start, stop in ((0, 6000), (6000, 12000)):
            latent = (eog[:, start:stop] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
            scaled = eeg[:, start:stop] / registry30.eeg_scale[:, None]
            halves.append(ridge_transfer(scaled, latent, registry30.ridge_ratio)[0])
        scores.append(_score(halves[0], halves[1], post_var[key]))
    return float(np.percentile(scores, 95))


def _rls_trajectory(registry30, key, init: np.ndarray, grid, start: int,
                    forget: float = RLS_FORGET) -> dict[int, np.ndarray]:
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    cell = registry30.cells[key]
    stop = start + max(grid) * 100
    latent = (eog[:, start:stop] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
    scaled = eeg[:, start:stop] / registry30.eeg_scale[:, None]
    operator = np.asarray(init, np.float64).copy()
    p = np.eye(2) * 1e3
    snapshots = {}
    marks = {int(t * 100): t for t in grid}
    for t in range(latent.shape[1]):
        e = latent[:, t]
        gain = p @ e / (forget + e @ p @ e)
        operator += np.outer(scaled[:, t] - operator @ e, gain)
        p = (p - np.outer(gain, e @ p)) / forget
        if t + 1 in marks:
            snapshots[marks[t + 1]] = operator.copy()
    return snapshots


# ------------------------------------------------------------------- RB-1

def rb1() -> dict[str, Any]:
    rows = []
    for fold_id in range(5):
        for seed in S1_SEEDS:
            rows += json.loads((RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                                / "stage1_result.json").read_text())["rows"]
    frame = pd.DataFrame(rows)
    per = {arm: _participant_means(frame, arm) for arm in ("MATCH_gated", "NO_A0", "POP")}
    participants = per["MATCH_gated"].index
    vs_noa0 = (per["NO_A0"] - per["MATCH_gated"]).loc[participants]
    vs_pop = (per["POP"] - per["MATCH_gated"]).loc[participants]
    anchor = (per["POP"] - per["NO_A0"]).loc[participants]
    return {"gain_vs_NO_A0": _stat(vs_noa0), "gain_vs_POP": _stat(vs_pop),
            "bad_anchor_worse_than_none_POP_minus_NO_A0": _stat(anchor),
            "condition_means": {arm: float(series.mean()) for arm, series in per.items()}}


# ------------------------------------------------------------------ stage2 cell

def stage2(fold_id: int, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    result_dir = RESULT / "stage2" / f"fold_{fold_id}_seed_{seed}"
    result_path = result_dir / "stage2_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "complete"}))
        return
    source = json.loads((RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                         / "train_curve.json").read_text())
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    post_var = _posterior_variance(registry30, eb120, fold)
    threshold = _split_half_threshold(registry30, eb120, post_var)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    from eeg_scad.cli.run_v44 import _bank_drives
    drives = _bank_drives(assets, bank)
    wrongs = [sampler.condition_signature(meta, "WRONG")[1] for meta in bank["meta"]]
    qgen_start = int(data["qgen_start"])
    qnat_start = int(data["qnatural_start"])

    probes = {}
    guard_rows = []
    for key in sorted({(m["participant"], m["session"], m["task"]) for m in bank["meta"]}):
        probes[key] = {t_v: _probe_operator(registry30, key, t_v, qgen_start)
                       for t_v in (TV_PRIMARY,) + TV_SENSITIVITY}
    flags_match, flags_wrong = {}, {}
    for meta, wrong in zip(bank["meta"], wrongs):
        key = (meta["participant"], meta["session"], meta["task"])
        wkey = (wrong, meta["session"], meta["task"])
        for t_v in (TV_PRIMARY,) + TV_SENSITIVITY:
            score_match = _score(assets[key]["C_gated"], probes[key][t_v], post_var[key])
            score_wrong = _score(assets[wkey]["C_gated"], probes[key][t_v], post_var[key])
            guard_rows.append({"participant": key[0], "cell": "|".join(key), "t_v": t_v,
                               "wrong_owner": wrong, "score_match": score_match,
                               "score_wrong": score_wrong, "threshold": threshold,
                               "flag_match": int(score_match > threshold),
                               "flag_wrong": int(score_wrong > threshold)})
            if t_v == TV_PRIMARY:
                flags_match[key] = score_match > threshold
                flags_wrong[(key, wkey)] = score_wrong > threshold

    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    ns = noise_seed(fold_id, seed)

    def run_arm(name, a0_stack, sig_stack, owners, extra=None):
        output = sample_bank_eog(model, schedule, bank["y"], np.stack(a0_stack),
                                 np.stack(sig_stack), device, ns)
        for clean, observed, artifact, prediction, meta, owner in zip(
                bank["x"], bank["y"], bank["artifact"], output, bank["meta"], owners):
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite V44-S2 output in {name}")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": name, "context_owner": owner,
                         "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         **(extra or {}),
                         **paired_metrics(clean, observed, artifact, observed - prediction)})

    rows: list[dict] = []

    def stacks(arm):
        a0s, sigs, owners = [], [], []
        for meta, drive, wrong in zip(bank["meta"], drives, wrongs):
            key = (meta["participant"], meta["session"], meta["task"])
            wkey = (wrong, meta["session"], meta["task"])
            if arm == "MATCH_guard":
                fired = flags_match[key]
                a0s.append(np.zeros_like(assets[key]["C_gated"] @ drive) if fired
                           else assets[key]["C_gated"] @ drive)
                sigs.append(assets[key]["sig_pop"] if fired else assets[key]["sig_gated"])
                owners.append(meta["participant"])
            elif arm == "WRONG_guard":
                fired = flags_wrong[(key, wkey)]
                a0s.append(np.zeros_like(assets[key]["C_gated"] @ drive) if fired
                           else assets[wkey]["C_gated"] @ drive)
                sigs.append(assets[key]["sig_pop"] if fired else assets[wkey]["sig_gated"])
                owners.append(wrong)
            elif arm == "MATCH_guard_popfallback":
                fired = flags_match[key]
                a0s.append(assets[key]["C0"] @ drive if fired else assets[key]["C_gated"] @ drive)
                sigs.append(assets[key]["sig_pop"] if fired else assets[key]["sig_gated"])
                owners.append(meta["participant"])
            elif arm == "WRONG_guard_popfallback":
                fired = flags_wrong[(key, wkey)]
                a0s.append(assets[key]["C0"] @ drive if fired
                           else assets[wkey]["C_gated"] @ drive)
                sigs.append(assets[key]["sig_pop"] if fired else assets[wkey]["sig_gated"])
                owners.append(wrong)
            else:
                raise ValueError(arm)
        return a0s, sigs, owners

    for arm in ("MATCH_guard", "WRONG_guard", "MATCH_guard_popfallback",
                "WRONG_guard_popfallback"):
        a0s, sigs, owners = stacks(arm)
        run_arm(arm, a0s, sigs, owners)

    # ------------------------------------------------------------- OR (RLS)
    trajectories: dict[tuple, dict[str, dict[int, np.ndarray]]] = {}
    for key in probes:
        wrong = next(w for m, w in zip(bank["meta"], wrongs)
                     if (m["participant"], m["session"], m["task"]) == key)
        wkey = (wrong, key[1], key[2])
        trajectories[key] = {
            "warm": _rls_trajectory(registry30, key, assets[key]["C_gated"], RLS_GRID, qgen_start),
            "coldzero": _rls_trajectory(registry30, key, np.zeros_like(assets[key]["C_gated"]),
                                        RLS_GRID, qgen_start),
            "coldpop": _rls_trajectory(registry30, key, assets[key]["C0"], (RLS_GRID[-1],),
                                       qgen_start),
            "wrongwarm": _rls_trajectory(registry30, key, assets[wkey]["C_gated"], RLS_GRID,
                                         qgen_start),
            "warm_sens": _rls_trajectory(registry30, key, assets[key]["C_gated"],
                                         (RLS_GRID[-1],), qgen_start, forget=RLS_FORGET_SENS),
        }

    def rls_arm(name, init, t):
        a0s, sigs, owners = [], [], []
        for meta, drive in zip(bank["meta"], drives):
            key = (meta["participant"], meta["session"], meta["task"])
            operator = trajectories[key][init][t]
            a0s.append(operator @ drive)
            sig = assets[key]["sig_gated"] if init in ("warm", "wrongwarm", "warm_sens") \
                else assets[key]["sig_pop"]
            if init == "wrongwarm":
                wrong = next(w for m, w in zip(bank["meta"], wrongs)
                             if (m["participant"], m["session"], m["task"]) == key)
                sig = assets[(wrong, key[1], key[2])]["sig_gated"]
            sigs.append(sig)
            owners.append(meta["participant"])
        run_arm(name, a0s, sigs, owners, extra={"rls_init": init, "rls_t": t})

    for t in RLS_GRID:
        rls_arm(f"RLS_warm_{t}s", "warm", t)
        rls_arm(f"RLS_coldzero_{t}s", "coldzero", t)
        rls_arm(f"RLS_wrongwarm_{t}s", "wrongwarm", t)
    rls_arm(f"RLS_coldpop_{RLS_GRID[-1]}s", "coldpop", RLS_GRID[-1])
    rls_arm(f"RLS_warm099_{RLS_GRID[-1]}s", "warm_sens", RLS_GRID[-1])

    # ------------------------------------------------------------- natural
    natural_rows = []
    nns = natural_noise_seed(fold_id, seed)
    for participant, session, task in itertools.product(fold["test"], data["sessions"],
                                                        data["tasks"]):
        key = (participant, session, task)
        if key not in assets:
            continue
        wrong = sorted(c for c in {k[0] for k in registry30.cells}
                       if c != participant and (c, session, task) in assets)[0]
        wkey = (wrong, session, task)
        nat_probe = _probe_operator(registry30, key, TV_PRIMARY, qnat_start)
        flag_match = _score(assets[key]["C_gated"], nat_probe, post_var[key]) > threshold
        flag_wrong = _score(assets[wkey]["C_gated"], nat_probe, post_var[key]) > threshold
        warm_end = _rls_trajectory(registry30, key, assets[key]["C_gated"],
                                   (RLS_GRID[-1],), qnat_start)[RLS_GRID[-1]]
        windows = list(_natural_windows(registry30, data, key))
        y_stack = np.stack([w[1] for w in windows])
        arm_specs = {
            "NAT_MATCH_guard": [((np.zeros_like(assets[key]["C_gated"] @ w[2])
                                  if flag_match else assets[key]["C_gated"] @ w[2]),
                                 assets[key]["sig_pop"] if flag_match
                                 else assets[key]["sig_gated"]) for w in windows],
            "NAT_WRONG_guard": [((np.zeros_like(assets[key]["C_gated"] @ w[2])
                                  if flag_wrong else assets[wkey]["C_gated"] @ w[2]),
                                 assets[key]["sig_pop"] if flag_wrong
                                 else assets[wkey]["sig_gated"]) for w in windows],
            "NAT_RLS_warm_end": [((warm_end @ w[2]), assets[key]["sig_gated"])
                                 for w in windows],
        }
        for arm, spec in arm_specs.items():
            output = sample_bank_eog(model, schedule, y_stack,
                                     np.stack([s[0] for s in spec]),
                                     np.stack([s[1] for s in spec]), device, nns)
            for (start, y, drive), out in zip(windows, output):
                if not np.isfinite(out).all():
                    raise FloatingPointError(f"nonfinite V44-S2 natural output in {arm}")
                natural_rows.append({"fold": fold_id, "seed": seed, "participant": participant,
                                     "session": session, "task": task, "start": start,
                                     "condition": arm, "flag_match": int(flag_match),
                                     "flag_wrong": int(flag_wrong),
                                     **_natural_metrics_full(y, drive, np.asarray(out, np.float64))})

    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "threshold": threshold,
        "guard_rows": guard_rows, "rows": rows, "natural_rows": natural_rows,
        "noise_seed": ns, "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "threshold": round(threshold, 4),
                      "arms": len({r['condition'] for r in rows})}))


# ------------------------------------------------------------------ aggregate

def aggregate() -> dict[str, Any]:
    rows, guard_rows, natural_rows = [], [], []
    s1_rows, s1_natural = [], []
    for fold_id in range(5):
        for seed in S1_SEEDS:
            payload = json.loads((RESULT / "stage2" / f"fold_{fold_id}_seed_{seed}"
                                  / "stage2_result.json").read_text())
            rows += payload["rows"]
            guard_rows += payload["guard_rows"]
            natural_rows += payload["natural_rows"]
            s1 = json.loads((RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                             / "stage1_result.json").read_text())
            s1_rows += s1["rows"]
            s1_natural += s1["natural_rows"]
    frame = pd.DataFrame(rows)
    s1_frame = pd.DataFrame(s1_rows)
    per = lambda f, c: _participant_means(f, c)
    match_s1 = per(s1_frame, "MATCH_gated")
    noa0_s1 = per(s1_frame, "NO_A0")
    oracle_s1 = per(s1_frame, "ORACLE")
    participants = match_s1.index

    rb = rb1()
    guard = pd.DataFrame(guard_rows)
    primary = guard[guard.t_v == TV_PRIMARY]
    detection = float(primary.flag_wrong.mean())
    false_alarm = float(primary.flag_match.mean())
    og1_delta = (per(frame, "WRONG_guard") - noa0_s1).loc[participants]
    og1_stat = _stat(og1_delta)
    og1 = {"contrast": "WRONG_gated_guard_minus_NO_A0", **og1_stat, "margin": OG1_MARGIN,
           "detection_rate": detection, "detection_required": OG1_DETECTION,
           "pass": bool(og1_delta.mean() <= OG1_MARGIN and detection >= OG1_DETECTION)}
    og2_delta = (per(frame, "MATCH_guard") - match_s1).loc[participants]
    og2_stat = _stat(og2_delta)
    og2 = {"contrast": "MATCH_gated_guard_minus_MATCH_gated", **og2_stat, "margin": OG2_COST,
           "false_alarm_rate": false_alarm, "false_alarm_max": OG2_FALSE_ALARM,
           "pass": bool(false_alarm <= OG2_FALSE_ALARM and og2_delta.mean() <= OG2_COST)}
    p_raw = {"OG-1": float(np.mean(bootstrap_draws(og1_delta.to_numpy()) >= OG1_MARGIN)),
             "OG-2": float(np.mean(bootstrap_draws(og2_delta.to_numpy()) >= OG2_COST))}
    sensitivity = guard.groupby("t_v").agg(detection=("flag_wrong", "mean"),
                                           false_alarm=("flag_match", "mean")).reset_index()

    warm_end = per(frame, f"RLS_warm_{RLS_GRID[-1]}s")
    or1_delta = (match_s1 - warm_end).loc[participants]
    or1_stat = _stat(or1_delta)
    closure = float(or1_delta.mean() / max(ORACLE_GAP_S1, 1e-9))
    or1 = {"contrast": "static_MATCH_gated_minus_warm_RLS_end", **or1_stat,
           "pass": bool(or1_stat["mean"] > 0 and or1_stat["bootstrap_low"] > 0),
           "oracle_gap_s1": ORACLE_GAP_S1, "gap_closure_fraction": closure,
           "warm099_end_mean": float(per(frame, f"RLS_warm099_{RLS_GRID[-1]}s").mean()),
           "coldpop_end_mean": float(per(frame, f"RLS_coldpop_{RLS_GRID[-1]}s").mean())}
    half_life = []
    for t in RLS_GRID:
        warm = per(frame, f"RLS_warm_{t}s")
        cold = per(frame, f"RLS_coldzero_{t}s")
        wrong = per(frame, f"RLS_wrongwarm_{t}s")
        half_life.append({"t_seconds": t,
                          "warm_mean_rrmse": float(warm.mean()),
                          "coldzero_mean_rrmse": float(cold.mean()),
                          "wrongwarm_mean_rrmse": float(wrong.mean()),
                          "calibration_value_cold_minus_warm": float((cold - warm).mean())})

    nat = pd.DataFrame(natural_rows)
    s1n = pd.DataFrame(s1_natural)
    nat_per = lambda f, c, m: f[f.condition == c].groupby("participant")[m].mean()
    natural = {}
    for metric in ("attenuation_db", "low_eog_observation_retention"):
        match_ref = nat_per(s1n, "MATCH_gated", metric)
        natural[metric] = {
            "MATCH_guard_minus_MATCH": _stat((nat_per(nat, "NAT_MATCH_guard", metric)
                                              - match_ref).dropna()),
            "WRONG_guard_mean": float(nat_per(nat, "NAT_WRONG_guard", metric).mean()),
            "NO_A0_ref_mean": float(nat_per(s1n, "NO_A0", metric).mean()),
            "RLS_warm_end_minus_MATCH": _stat((nat_per(nat, "NAT_RLS_warm_end", metric)
                                               - match_ref).dropna())}

    decision = {
        "preregistration": "reports/v44_preregistration.md (V44-S2 addendum)",
        "stage": "V44_S2_ownership_guard",
        "RB-1": rb, "OG-1": og1, "OG-2": og2,
        "holm": {"p_raw": p_raw, "p_adjusted": holm(p_raw), "alpha": 0.05},
        "OR-1": or1, "OR-2_half_life": half_life,
        "OR-3_wrongwarm_note": "see half-life table wrongwarm column",
        "guard_sensitivity_by_t_v": sensitivity.to_dict("records"),
        "natural": natural,
        "condition_means": {c: float(per(frame, c).mean())
                            for c in sorted(frame.condition.unique())},
        "sealed_reads": 0,
    }
    (RESULT / "stage2" / "decision.json").write_text(json.dumps(decision, indent=2,
                                                                sort_keys=True) + "\n")
    (REPORT / "v44_stage2.md").write_text(
        "# V44 Stage 2 — ownership guard, re-baselining, online refinement\n\n"
        "Inference-only on the frozen V44-S1 checkpoints; addendum rules frozen before "
        "submission. S1 verdicts unrevised.\n\n"
        f"Decision: OG-1 **{og1['pass']}**, OG-2 **{og2['pass']}**, OR-1 **{or1['pass']}**.\n\n"
        "## RB-1 honest re-baselining\n\n```json\n" + json.dumps(rb, indent=2, sort_keys=True)
        + "\n```\n\n## Ownership guard\n\n```json\n"
        + json.dumps({"OG-1": og1, "OG-2": og2, "holm": decision["holm"],
                      "sensitivity": decision["guard_sensitivity_by_t_v"]},
                     indent=2, sort_keys=True)
        + "\n```\n\n## Online refinement\n\n```json\n"
        + json.dumps({"OR-1": or1}, indent=2, sort_keys=True) + "\n```\n\n"
        "### Calibration half-life (OR-2/OR-3)\n\n"
        + pd.DataFrame(half_life).round(6).to_markdown(index=False)
        + "\n\n## Natural panel\n\n```json\n"
        + json.dumps(natural, indent=2, sort_keys=True) + "\n```\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("rb1")
    cell = sub.add_parser("stage2")
    cell.add_argument("--fold", type=int, required=True)
    cell.add_argument("--seed", type=int, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.stage == "rb1":
        print(json.dumps(rb1(), indent=2, sort_keys=True))
    elif args.stage == "stage2":
        stage2(args.fold, args.seed)
    else:
        print(json.dumps(aggregate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
