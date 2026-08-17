#!/usr/bin/env python3
"""IRIS F2-R — spatially-gated subtraction rerun of the dots fight.

Preregistered as amendment F-1 in reports/iris_prereg_f.md. Per-channel abstention:
a target channel receives subtraction only if BOTH split-half directions on FIT give
validated r^2 >= 0.10 AND validated post-saccadic retention >= 0.84. Fail-closed.
E1/E2/E3 and both readings exactly as F2-a. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/iris"))
import w_stage as W          # noqa: E402
import f2_fight as F         # noqa: E402

W.REPAIRED = True
OUT = REPO / "results/iris/f2"
R2_MIN = 0.10
ARMS = F.ARMS


def gated_row(p: dict) -> dict:
    fs = p["fs"]
    lags5 = [int(round(ms / 1000.0 * fs)) for ms in W.LAGS_MS]
    kernel_op, fir_lags, trains = F._fir_kernel(p)
    conv = F._conv_trains(kernel_op, fir_lags, trains)
    static = W._z(np.stack([p["veog"], p["heog"]]), p["fit"])
    gaze = W._z(p["gaze"], p["fit"])
    typed = np.concatenate([W._z(conv, p["fit"]), gaze], axis=0)
    designs = {
        "INCUMBENT_NATIVE": static,
        "INCUMBENT_SAMEREF": np.concatenate([static, gaze], axis=0),
        "IRIS_TYPED": np.concatenate([static, typed], axis=0),
    }
    target = np.concatenate([p["frontal"], p["posterior"]], axis=0)
    n_front = p["frontal"].shape[0]
    ev = p["eval"]
    rich = F._rich_mask(p)
    post_mask_eval = F._post_saccadic_mask(p)
    drive_eval = np.stack([p["veog"], p["heog"]])[:, ev]

    # split-half FIT structure: the two constituent thirds
    n = p["n"]
    third = n // 3
    halves = (np.r_[0:third], np.r_[2 * third:n])

    # post-saccadic mask over the full recording (for FIT-side validation)
    span = int(F.E3_WINDOW_S * fs)
    full_mask = np.zeros(n, bool)
    for i, kind in enumerate(p["events"]["type"]):
        if "_saccade" not in kind:
            continue
        onset = p["events"]["latency"][i]
        if np.isfinite(onset) and 0 <= int(onset) < n:
            full_mask[int(onset):min(int(onset) + span, n)] = True

    out = {}
    for arm, ref in designs.items():
        lagged = W._lagged(ref, lags5)
        eligible = np.ones(target.shape[0], bool)
        for a, b in (halves, halves[::-1]):
            op = W._ridge_fit(target[:, a], lagged[:, a])
            pred = W._predict(op, lagged)[:, b]
            tgt = W._center(target[:, b])
            resid = tgt - pred
            r2 = 1.0 - resid.var(axis=1) / np.maximum(tgt.var(axis=1), 1e-18)
            pm = full_mask[b]
            keep = np.linalg.norm(pred[:, pm], axis=1) \
                <= (1 - F.E3_BAR) * np.maximum(
                    np.linalg.norm(tgt[:, pm], axis=1), 1e-12)
            # amendment F-2: the retention screen applies to posterior channels
            # only; frontal channels are gated by validated r^2 alone
            keep[:n_front] = True
            eligible &= (r2 >= R2_MIN) & keep
        op_full = W._ridge_fit(target[:, p["fit"]], lagged[:, p["fit"]])
        est = W._predict(op_full, lagged)[:, ev]
        est[~eligible] = 0.0
        y_front = W._center(p["frontal"][:, ev])
        est_front = est[:n_front]
        resid = y_front - est_front
        att = 10 * np.log10(max(float(np.var(y_front[:, rich])), 1e-18)
                            / max(float(np.var(resid[:, rich])), 1e-18))
        nat = F._v44_natural_metrics(y_front, drive_eval, est_front)
        y_post = W._center(p["posterior"][:, ev])
        est_post = est[n_front:]
        e3 = 1 - float(np.linalg.norm(est_post[:, post_mask_eval])
                       / max(np.linalg.norm(y_post[:, post_mask_eval]), 1e-12))
        out[arm] = {"attenuation_db": float(att),
                    "retention": nat["low_eog_observation_retention"],
                    "e3_post_saccadic_retention": float(e3),
                    "eligible_frontal": int(eligible[:n_front].sum()),
                    "eligible_posterior": int(eligible[n_front:].sum())}
    return out


def main() -> None:
    rows = []
    for subject_dir in sorted(W.ROOT.iterdir()):
        for path in sorted(subject_dir.glob("*.mat")):
            arms = gated_row(F.prepare_dots(path))
            rows.append({"recording": f"{subject_dir.name}/{path.name}",
                         "participant": subject_dir.name, "arms": arms})
            print(json.dumps({"rec": rows[-1]["recording"],
                              **{a: [round(arms[a]["attenuation_db"], 2),
                                     arms[a]["eligible_frontal"],
                                     arms[a]["eligible_posterior"]]
                                 for a in ARMS}}), flush=True)

    def pf(metric, arm):
        per = {}
        for r in rows:
            per.setdefault(r["participant"], []).append(r["arms"][arm][metric])
        return W._stat([float(np.mean(v)) for v in per.values()])

    def pf_delta(metric, a, b):
        per = {}
        for r in rows:
            per.setdefault(r["participant"], []).append(
                r["arms"][a][metric] - r["arms"][b][metric])
        return W._stat([float(np.mean(v)) for v in per.values()])

    e3 = {arm: pf("e3_post_saccadic_retention", arm) for arm in ARMS}
    e3_pass = {arm: bool(e3[arm]["mean"] >= F.E3_BAR) for arm in ARMS}
    readings = {}
    for label, incumbent in (("same_reference", "INCUMBENT_SAMEREF"),
                             ("native_reference", "INCUMBENT_NATIVE")):
        d_att = pf_delta("attenuation_db", "IRIS_TYPED", incumbent)
        d_ret = pf_delta("retention", "IRIS_TYPED", incumbent)
        valid = e3_pass["IRIS_TYPED"] and e3_pass[incumbent] \
            and d_ret["mean"] >= F.E2_NONINF
        verdict = ("INVALID(E3/E2)" if not valid else
                   "WIN" if d_att["bootstrap_low"] > 0 else
                   "LOSS" if d_att["bootstrap_high"] < 0 else "TIE")
        readings[label] = {"incumbent": incumbent,
                           "attenuation_delta_db": d_att,
                           "retention_delta": d_ret, "verdict": verdict}
    (OUT / "f2r_decision.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_f.md (amendment F-1)",
        "r2_min": R2_MIN,
        "arm_levels": {arm: {"attenuation_db": pf("attenuation_db", arm),
                             "retention": pf("retention", arm),
                             "e3": e3[arm], "e3_pass": e3_pass[arm],
                             "eligible_frontal": pf("eligible_frontal", arm),
                             "eligible_posterior": pf("eligible_posterior", arm)}
                       for arm in ARMS},
        "readings": readings, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: r["verdict"] for k, r in readings.items()}))


if __name__ == "__main__":
    main()
