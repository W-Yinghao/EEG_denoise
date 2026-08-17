#!/usr/bin/env python3
"""IRIS F2 — EEGEyeNet dual-reading fight (F2-a, dots) + sealed-fight qualifier
(F2-b, antisaccade dev). Preregistered in reports/iris_prereg_f.md. CPU only.

F2-a arms: INCUMBENT_NATIVE (static EOG), INCUMBENT_SAMEREF (static EOG+gaze,
no typed structure), IRIS_TYPED (static ∪ typed additive). Endpoints: E1 rich-window
attenuation (frontal 8), E2 V44 low-EOG retention verbatim (frontal 8; the V44
quantile-.3 low mask governs, per the VERBATIM clause), E3 post-saccadic posterior
retention >= 0.84 hard gate (posterior 8). Fit target = frontal ∪ posterior.

F2-b: T2 increment machinery with event-train-only typed drives on the 28 antisaccade
dev recordings (periocular-interpolated recordings INCLUDED, status carried; 2-recording
intact companion).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/iris"))
import w_stage as W  # noqa: E402

W.REPAIRED = True                      # repaired gaze encoding per prereg
ANTI_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_min/antisaccade_min")
OUT = REPO / "results/iris/f2"
E3_BAR = 0.84
E3_WINDOW_S = 0.300
E2_NONINF = -0.02
POSTERIOR_N = 8
ARMS = ("INCUMBENT_NATIVE", "INCUMBENT_SAMEREF", "IRIS_TYPED")
PERI_NUMS = {int(lab[1:]) for lab in W.ALL_PERI}


def _stat(values):
    return W._stat(values)


# ------------------------------------------------------------------ shared

def _fir_kernel(p: dict) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Per-type FIR templates fit on FIT vs the frontal block (W3 construction,
    referee-free: the referee gated oracle claims, not the fight)."""
    fs = p["fs"]
    fir_lags = [int(round(ms / 1000.0 * fs)) for ms in W.FIR_LAGS_MS]
    trains = W._event_trains(p)
    design = W._lagged(trains, fir_lags)
    if p.get("gaze") is not None:
        lags5 = [int(round(ms / 1000.0 * fs)) for ms in W.LAGS_MS]
        gaze2 = W._z(p["gaze"][:2], p["fit"])
        design = np.concatenate([design, W._lagged(gaze2, lags5)], axis=0)
    kernel_op = W._ridge_fit(p["frontal"][:, p["fit"]], design[:, p["fit"]])
    return kernel_op, fir_lags, trains


def _conv_trains(kernel_op: np.ndarray, fir_lags: list[int],
                 trains: np.ndarray) -> np.ndarray:
    n_fir = len(fir_lags)
    kernel = kernel_op[:, :2 * n_fir].mean(axis=0)
    conv = np.zeros_like(trains)
    for t_i in range(2):
        for lag, c in zip(fir_lags, kernel[t_i::2][:n_fir]):
            conv[t_i] += c * np.roll(trains[t_i], lag)
    return conv


def _rich_mask(p: dict, fraction: float = W.ARTIFACT_RICH) -> np.ndarray:
    fs = p["fs"]
    ev = p["eval"]
    window = int(W.WINDOW_S * fs)
    n_win = len(ev) // window
    energy = (p["veog"][ev][:n_win * window].reshape(n_win, window) ** 2).mean(axis=1)
    rich = np.argsort(energy)[-max(int(round(n_win * fraction)), 1):]
    mask = np.zeros(len(ev), bool)
    for w_i in rich:
        mask[w_i * window:(w_i + 1) * window] = True
    return mask


def _v44_natural_metrics(y: np.ndarray, drive: np.ndarray,
                         estimate: np.ndarray) -> dict[str, float]:
    """V44 run_v44._natural_metrics, ported verbatim (retention is the endpoint)."""
    out = y - estimate
    energy = np.sqrt(np.mean(drive * drive, axis=0))
    low = energy <= np.quantile(energy, .3)
    high = energy >= np.quantile(energy, .7)
    rms = lambda v: float(np.sqrt(np.mean(v * v)))  # noqa: E731
    return {
        "attenuation_db_v44": float(20 * np.log10(
            max(rms(y[:, high]), 1e-12) / max(rms(out[:, high]), 1e-12))),
        "low_eog_observation_retention": 1 - float(
            np.linalg.norm(estimate[:, low]) / max(np.linalg.norm(y[:, low]), 1e-12)),
    }


def _post_saccadic_mask(p: dict) -> np.ndarray:
    fs = p["fs"]
    ev = p["eval"]
    span = int(E3_WINDOW_S * fs)
    mask = np.zeros(len(ev), bool)
    start0 = ev[0]
    for i, kind in enumerate(p["events"]["type"]):
        if "_saccade" not in kind:
            continue
        onset = p["events"]["latency"][i]
        if not np.isfinite(onset):
            continue
        rel = int(onset) - start0
        if 0 <= rel < len(ev):
            mask[rel:min(rel + span, len(ev))] = True
    return mask


# ------------------------------------------------------------------ F2-a

def prepare_dots(path: Path) -> dict:
    rec = W.load_full(path)
    p = W.prepare(rec)
    labels = rec["labels"]
    eligible = [i for i, lab in enumerate(labels)
                if lab not in W.NON_EEG and lab not in W.ALL_PERI]
    posterior_idx = sorted(eligible, key=lambda i: rec["x_coord"][i])[:POSTERIOR_N]
    p["posterior"] = W._bandpass(rec["data"][posterior_idx], 0.5, 40.0, p["fs"])
    return p


def f2a_row(p: dict) -> dict:
    fs = p["fs"]
    lags5 = [int(round(ms / 1000.0 * fs)) for ms in W.LAGS_MS]
    kernel_op, fir_lags, trains = _fir_kernel(p)
    conv = _conv_trains(kernel_op, fir_lags, trains)
    static = W._z(np.stack([p["veog"], p["heog"]]), p["fit"])
    gaze = W._z(p["gaze"], p["fit"])
    typed = np.concatenate([W._z(conv, p["fit"]), gaze], axis=0)
    designs = {
        "INCUMBENT_NATIVE": static,
        "INCUMBENT_SAMEREF": np.concatenate([static, gaze], axis=0),
        "IRIS_TYPED": np.concatenate([static, typed], axis=0),
    }
    target = np.concatenate([p["frontal"], p["posterior"]], axis=0)
    ev = p["eval"]
    rich = _rich_mask(p)
    post_mask = _post_saccadic_mask(p)
    drive_eval = np.stack([p["veog"], p["heog"]])[:, ev]
    n_front = p["frontal"].shape[0]
    out = {}
    for arm, ref in designs.items():
        op = W._ridge_fit(target[:, p["fit"]], W._lagged(ref, lags5)[:, p["fit"]])
        est = W._predict(op, W._lagged(ref, lags5))[:, ev]
        y_front = W._center(p["frontal"][:, ev])
        est_front = est[:n_front]
        resid = y_front - est_front
        att = 10 * np.log10(max(float(np.var(y_front[:, rich])), 1e-18)
                            / max(float(np.var(resid[:, rich])), 1e-18))
        nat = _v44_natural_metrics(y_front, drive_eval, est_front)
        y_post = W._center(p["posterior"][:, ev])
        est_post = est[n_front:]
        e3 = 1 - float(np.linalg.norm(est_post[:, post_mask])
                       / max(np.linalg.norm(y_post[:, post_mask]), 1e-12))
        out[arm] = {"attenuation_db": float(att),
                    "retention": nat["low_eog_observation_retention"],
                    "e3_post_saccadic_retention": float(e3)}
    return out


# ------------------------------------------------------------------ F2-b

def load_full_anti(path: Path) -> dict:
    import h5py
    from k3_instrument import _h5_num, _h5_text
    with h5py.File(path, "r") as handle:
        eeg = handle["EEG"]
        srate = float(eeg["srate"][0, 0])
        labels = [_h5_text(handle, r)
                  for r in np.asarray(eeg["chanlocs"]["labels"]).ravel()]
        x_coord = np.asarray([_h5_num(handle, r)
                              for r in np.asarray(eeg["chanlocs"]["X"]).ravel()])
        idx = {lab: i for i, lab in enumerate(labels)}
        eligible = [i for i, lab in enumerate(labels) if lab not in W.ALL_PERI]
        frontal_idx = sorted(eligible, key=lambda i: -x_coord[i])[:W.FRONTAL_N]
        want = sorted(set([idx[lab] for lab in W.ALL_PERI] + frontal_idx))
        data_cols = eeg["data"][:, want].astype(float).T     # (chan, samples)
        col = {c: j for j, c in enumerate(want)}
        get = lambda lab: data_cols[col[idx[lab]]]           # noqa: E731
        veog = W._bandpass((get("E25") - get("E127") + get("E8") - get("E126")) / 2.0,
                           0.5, 8.0, srate)
        heog = W._bandpass(get("E125") - get("E128"), 0.5, 20.0, srate)
        frontal = W._bandpass(np.stack([data_cols[col[i]] for i in frontal_idx]),
                              0.5, 40.0, srate)
        events = {}
        ev_grp = eeg["event"]
        for field in ("type", "latency", "duration", "endtime", "sac_amplitude",
                      "sac_startpos_x", "sac_startpos_y", "sac_endpos_x",
                      "sac_endpos_y"):
            refs = np.asarray(ev_grp[field]).ravel()
            events[field] = ([_h5_text(handle, r) for r in refs] if field == "type"
                             else np.asarray([_h5_num(handle, r) for r in refs]))
        bad = set()
        auto = handle.get("automagic")
        if auto is not None:
            for key, sub in (("finalBadChans", None), ("interpolation", "channels")):
                node = auto.get(key)
                if node is not None and sub is not None:
                    node = node.get(sub) if hasattr(node, "get") else None
                if node is not None and hasattr(node, "shape"):
                    values = np.asarray(node[()]).ravel()
                    if values.dtype.kind in "fiu":
                        bad |= {int(v) for v in values if np.isfinite(v) and v > 0}
        n = frontal.shape[1]
        third = n // 3
        return {"fs": srate, "n": n, "veog": veog, "heog": heog, "frontal": frontal,
                "gaze": None, "fit": np.r_[0:third, 2 * third:n],
                "eval": np.r_[third:2 * third], "events": events,
                "periocular_interpolated": sorted(bad & PERI_NUMS)}


def f2b_row(p: dict) -> dict:
    fs = p["fs"]
    lags5 = [int(round(ms / 1000.0 * fs)) for ms in W.LAGS_MS]
    kernel_op, fir_lags, trains = _fir_kernel(p)
    conv = _conv_trains(kernel_op, fir_lags, trains)
    static = W._z(np.stack([p["veog"], p["heog"]]), p["fit"])
    typed = W._z(conv, p["fit"])
    lo = int(5 * fs)
    shift = int(W._T2_RNG.integers(lo, max(p["n"] - lo, lo + 1)))
    refs = {"static": static,
            "combined": np.concatenate([static, typed], axis=0),
            "combined_shuffled": np.concatenate(
                [static, np.roll(typed, shift, axis=1)], axis=0)}
    ev = p["eval"]
    mask = _rich_mask(p)
    out = {}
    for name, ref in refs.items():
        op = W._ridge_fit(p["frontal"][:, p["fit"]], W._lagged(ref, lags5)[:, p["fit"]])
        pred = W._predict(op, W._lagged(ref, lags5))[:, ev]
        tgt = W._center(p["frontal"][:, ev])
        out[name] = float(np.mean((tgt[:, mask] - pred[:, mask]) ** 2))
    return {"delta_inc": float((out["static"] - out["combined"])
                               / max(out["static"], 1e-12)),
            "delta_null": float((out["static"] - out["combined_shuffled"])
                                / max(out["static"], 1e-12)),
            "resid_static": out["static"]}


# ------------------------------------------------------------------ main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- F2-a
    rows = []
    for subject_dir in sorted(W.ROOT.iterdir()):
        for path in sorted(subject_dir.glob("*.mat")):
            arms = f2a_row(prepare_dots(path))
            rows.append({"recording": f"{subject_dir.name}/{path.name}",
                         "participant": subject_dir.name, "arms": arms})
            print(json.dumps({"rec": rows[-1]["recording"],
                              **{a: round(arms[a]["attenuation_db"], 2)
                                 for a in ARMS}}), flush=True)

    def pf(metric, arm):
        per = {}
        for r in rows:
            per.setdefault(r["participant"], []).append(r["arms"][arm][metric])
        return _stat([float(np.mean(v)) for v in per.values()])

    def pf_delta(metric, a, b):
        per = {}
        for r in rows:
            per.setdefault(r["participant"], []).append(
                r["arms"][a][metric] - r["arms"][b][metric])
        return _stat([float(np.mean(v)) for v in per.values()])

    e3 = {arm: pf("e3_post_saccadic_retention", arm) for arm in ARMS}
    e3_pass = {arm: bool(e3[arm]["mean"] >= E3_BAR) for arm in ARMS}
    readings = {}
    for label, incumbent in (("same_reference", "INCUMBENT_SAMEREF"),
                             ("native_reference", "INCUMBENT_NATIVE")):
        d_att = pf_delta("attenuation_db", "IRIS_TYPED", incumbent)
        d_ret = pf_delta("retention", "IRIS_TYPED", incumbent)
        valid = e3_pass["IRIS_TYPED"] and e3_pass[incumbent] \
            and d_ret["mean"] >= E2_NONINF
        verdict = ("INVALID(E3/E2)" if not valid else
                   "WIN" if d_att["bootstrap_low"] > 0 else
                   "LOSS" if d_att["bootstrap_high"] < 0 else "TIE")
        readings[label] = {"incumbent": incumbent,
                           "attenuation_delta_db": d_att,
                           "retention_delta": d_ret, "verdict": verdict}
    (OUT / "f2a_decision.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_f.md (F2-a)",
        "arm_levels": {arm: {"attenuation_db": pf("attenuation_db", arm),
                             "retention": pf("retention", arm),
                             "e3": e3[arm], "e3_pass": e3_pass[arm]}
                       for arm in ARMS},
        "e3_bar": E3_BAR, "e2_noninferiority": E2_NONINF,
        "readings": readings, "rows": rows}, indent=2, sort_keys=True) + "\n")

    # ---- F2-b
    brows = []
    for subject_dir in sorted(ANTI_ROOT.iterdir()):
        for path in sorted(subject_dir.glob("*.mat")):
            p = load_full_anti(path)
            row = {"recording": f"{subject_dir.name}/{path.name}",
                   "participant": subject_dir.name,
                   "periocular_interpolated": p["periocular_interpolated"],
                   **f2b_row(p)}
            brows.append(row)
            print(json.dumps({"rec": row["recording"],
                              "inc": round(row["delta_inc"], 4),
                              "null": round(row["delta_null"], 4),
                              "peri_interp": row["periocular_interpolated"]}),
                  flush=True)
    inc = _stat([r["delta_inc"] for r in brows])
    null = _stat([r["delta_null"] for r in brows])
    diff = _stat([r["delta_inc"] - r["delta_null"] for r in brows])
    intact = [r for r in brows if not r["periocular_interpolated"]]
    gate = bool(inc["bootstrap_low"] > 0 and inc["mean"] >= W.T_GATE_MEAN
                and diff["bootstrap_low"] > 0)
    (OUT / "f2b_decision.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_f.md (F2-b)",
        "note": "participant == recording on this task (one recording per subject)",
        "delta_inc": inc, "delta_null": null, "inc_minus_null": diff,
        "intact_companion": {"n": len(intact),
                             "delta_inc": [round(r["delta_inc"], 4)
                                           for r in intact]},
        "gate": {"pass": gate,
                 "verdict": ("sealed-fight plan carries a typed arm" if gate else
                             "typed leg dead on the sealed panel class; any sealed "
                             "opening proposal shrinks to incumbent-class "
                             "confirmation")},
        "rows": brows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"f2b_inc": round(inc["mean"], 4), "f2b_pass": gate}))


if __name__ == "__main__":
    main()
