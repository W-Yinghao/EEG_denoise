#!/usr/bin/env python3
"""IRIS W/T stage — the four instrument items + the typed-information kill, on dots.

Preregistered in reports/iris_prereg_w.md (frozen before execution). Dots dev only
(instrument VALID per K3); sealed untouched; CPU only. Writes
results/iris/w/{w1_a4,w2_kappa,w3_readout,t_typed_info}.json + per-recording rows.

Implementation constants not fixed by the prereg (documented, gate-neutral):
reference channels z-scored by FIT-third std; W3 event-train FIR lags −100..+400 ms in
10 ms steps; saccade train = boxcar over event duration × signed horizontal amplitude
(deg); T's per-type 1-D template = frontal-mean of the W3 FIR kernel; kernel_ridge RBF
centers = 16, rng seed 3 (T6 verbatim).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/iris"))
from k3_instrument import ALL_PERI, _sc_num, _sc_text  # noqa: E402

ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_min/dots_min")
OUT = REPO / "results/iris/w"
RIDGE = 0.05
LAGS_MS = (-100, -50, 0, 50, 100)
FIR_LAGS_MS = tuple(range(-100, 401, 10))
FRONTAL_N = 8
REFEREE_R = 0.5
ARTIFACT_RICH = 0.20
WINDOW_S = 1.0
SACCADE_MIN_DEG = 0.0            # W3/T use ALL saccades; K3's 2-deg cut was gate-only
KAPPA_BAR = 0.60
READOUT_BOUND = 0.03
T_GATE_MEAN = 0.05
BOOT_SEED, BOOT_DRAWS = 420, 5000
NON_EEG = {"TIME", "L-GAZE-X", "L-GAZE-Y", "L-AREA", "R-GAZE-X", "R-GAZE-Y", "R-AREA"}


def _ridge_fit(target: np.ndarray, drive: np.ndarray, ratio: float = RIDGE) -> np.ndarray:
    y = target - target.mean(axis=1, keepdims=True)
    e = drive - drive.mean(axis=1, keepdims=True)
    gram = e @ e.T
    ridge = float(ratio) * max(float(np.trace(gram) / len(gram)), np.finfo(float).eps)
    return (y @ e.T) @ np.linalg.inv(gram + ridge * np.eye(len(gram)))


def _predict(operator: np.ndarray, drive: np.ndarray) -> np.ndarray:
    return operator @ (drive - drive.mean(axis=1, keepdims=True))


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def _stat(values, seed: int = BOOT_SEED) -> dict:
    series = np.asarray(list(values), float)
    series = series[np.isfinite(series)]
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(series, len(series), replace=True).mean()
                        for _ in range(BOOT_DRAWS)])
    return {"mean": float(series.mean()), "median": float(np.median(series)),
            "n": int(len(series)), "positive_count": int((series > 0).sum()),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def _bandpass(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _lagged(drive: np.ndarray, lags_samples) -> np.ndarray:
    return np.concatenate([np.roll(drive, lag, axis=1) for lag in lags_samples], axis=0)


def load_full(path: Path) -> dict:
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    eeg = mat["sEEG"]
    while isinstance(eeg, np.ndarray):
        eeg = eeg.ravel()[0]
    data = np.asarray(eeg.data, float)
    chanlocs = np.asarray(eeg.chanlocs).ravel()
    labels = [_sc_text(c.labels) for c in chanlocs]
    x_coord = np.asarray([_sc_num(getattr(c, "X", np.nan)) for c in chanlocs])
    raw_events = np.asarray(eeg.event).ravel()
    events = {"type": [_sc_text(e.type) for e in raw_events]}
    for field in ("latency", "duration", "endtime", "sac_amplitude",
                  "sac_startpos_x", "sac_startpos_y", "sac_endpos_x", "sac_endpos_y"):
        events[field] = np.asarray(
            [_sc_num(getattr(e, field, np.nan)) for e in raw_events])
    return {"data": data, "labels": labels, "x_coord": x_coord, "events": events,
            "srate": float(np.asarray(eeg.srate).ravel()[0])}


def prepare(rec: dict) -> dict:
    fs = rec["srate"]
    labels = rec["labels"]
    idx = {lab: i for i, lab in enumerate(labels)}
    data = rec["data"]
    n = data.shape[1]
    veog = _bandpass((data[idx["E25"]] - data[idx["E127"]]
                      + data[idx["E8"]] - data[idx["E126"]]) / 2.0, 0.5, 8.0, fs)
    heog = _bandpass(data[idx["E125"]] - data[idx["E128"]], 0.5, 20.0, fs)
    eligible = [i for i, lab in enumerate(labels)
                if lab not in NON_EEG and lab not in ALL_PERI]
    frontal_idx = sorted(eligible, key=lambda i: -rec["x_coord"][i])[:FRONTAL_N]
    frontal = _bandpass(data[frontal_idx], 0.5, 40.0, fs)
    gaze = np.stack([data[idx["L-GAZE-X"]], data[idx["L-GAZE-Y"]],
                     data[idx["L-AREA"]]])
    third = n // 3
    fit = np.r_[0:third, 2 * third:n]
    ev = np.r_[third:2 * third]
    return {"fs": fs, "n": n, "veog": veog, "heog": heog, "frontal": frontal,
            "frontal_labels": [labels[i] for i in frontal_idx], "gaze": gaze,
            "fit": fit, "eval": ev, "events": rec["events"]}


def _z(x: np.ndarray, ref_idx: np.ndarray) -> np.ndarray:
    scale = np.std(x[:, ref_idx], axis=1, keepdims=True)
    return x / np.maximum(scale, 1e-12)


def _intervals(events: dict, tokens: tuple[str, ...], n: int) -> list[tuple[int, int]]:
    out = []
    for i, kind in enumerate(events["type"]):
        if not any(t in kind for t in tokens):
            continue
        start = events["latency"][i]
        end = events["endtime"][i]
        if not np.isfinite(start):
            continue
        if not np.isfinite(end):
            dur = events["duration"][i]
            end = start + (dur if np.isfinite(dur) else 0.0)
        out.append((max(int(start), 0), min(int(end), n - 1)))
    return out


def w1_row(p: dict) -> dict:
    fs = p["fs"]
    lags = [int(round(ms / 1000.0 * fs)) for ms in LAGS_MS]
    eog = _z(np.stack([p["veog"], p["heog"]]), p["fit"])
    opt = _z(p["gaze"], p["fit"])
    ops = {name: _ridge_fit(p["frontal"][:, p["fit"]],
                            _lagged(ref, lags)[:, p["fit"]])
           for name, ref in (("eog", eog), ("opt", opt))}
    est = {name: _predict(ops[name], _lagged(ref, lags))[:, p["eval"]]
           for name, ref in (("eog", eog), ("opt", opt))}
    diff = est["eog"] - est["opt"]
    denom = float(np.sqrt(np.mean(est["eog"] ** 2)))
    corr = float(np.corrcoef(est["eog"].ravel(), est["opt"].ravel())[0, 1])
    return {"d_rms": float(np.sqrt(np.mean(diff ** 2)) / max(denom, 1e-12)),
            "d_corr": 1.0 - corr}


def w2_events(p: dict) -> list[tuple[str, str]]:
    fs = p["fs"]
    pad = int(0.100 * fs)
    pairs = []
    for i, kind in enumerate(p["events"]["type"]):
        is_blink = "_blink" in kind
        is_sacc = "_saccade" in kind
        if not (is_blink or is_sacc):
            continue
        start = p["events"]["latency"][i]
        if not np.isfinite(start) or int(start) not in range(p["eval"][0],
                                                            p["eval"][-1]):
            continue
        end = p["events"]["endtime"][i]
        if not np.isfinite(end):
            dur = p["events"]["duration"][i]
            end = start + (dur if np.isfinite(dur) else 0.0)
        lo = max(int(start) - pad, 0)
        hi = min(int(end) + pad, p["n"] - 1)
        v = p["veog"][lo:hi]
        h = p["heog"][lo:hi]
        if v.size < 3:
            continue
        prominence = float(v.max() - np.median(v))
        step = float(h.max() - h.min())
        ratio = prominence / max(step, 1e-12)
        peak = int(np.argmax(v))
        half = prominence / 2.0 + np.median(v)
        left = peak
        while left > 0 and v[left] > half:
            left -= 1
        right = peak
        while right < v.size - 1 and v[right] > half:
            right += 1
        width_ok = (right - left) / fs >= 0.050
        predicted = "blink" if (ratio > 1.0 and width_ok) else "saccade"
        pairs.append(("blink" if is_blink else "saccade", predicted))
    return pairs


def _event_trains(p: dict) -> np.ndarray:
    n = p["n"]
    blink = np.zeros(n)
    for start, end in _intervals(p["events"], ("_blink",), n):
        blink[start:end + 1] = 1.0
    sacc = np.zeros(n)
    ev = p["events"]
    for i, kind in enumerate(ev["type"]):
        if "_saccade" not in kind:
            continue
        dx = ev["sac_endpos_x"][i] - ev["sac_startpos_x"][i]
        dy = ev["sac_endpos_y"][i] - ev["sac_startpos_y"][i]
        norm = float(np.hypot(dx, dy))
        amp = ev["sac_amplitude"][i]
        if not (np.isfinite(amp) and norm > 0 and np.isfinite(ev["latency"][i])):
            continue
        h_amp = amp * dx / norm
        if abs(h_amp) < SACCADE_MIN_DEG:
            continue
        start = int(ev["latency"][i])
        end = ev["endtime"][i]
        end = int(end) if np.isfinite(end) else start
        sacc[max(start, 0):min(end, n - 1) + 1] = h_amp
    return np.stack([blink, sacc])


def w3_row(p: dict) -> dict | None:
    fs = p["fs"]
    fir_lags = [int(round(ms / 1000.0 * fs)) for ms in FIR_LAGS_MS]
    lags5 = [int(round(ms / 1000.0 * fs)) for ms in LAGS_MS]
    trains = _event_trains(p)
    gaze2 = _z(p["gaze"][:2], p["fit"])
    design = np.concatenate([_lagged(trains, fir_lags), _lagged(gaze2, lags5)], axis=0)
    kernel_op = _ridge_fit(p["frontal"][:, p["fit"]], design[:, p["fit"]])
    a_ref = _predict(kernel_op, design)
    referee = float(np.corrcoef(a_ref[:, p["eval"]].mean(axis=0),
                                p["veog"][p["eval"]])[0, 1])
    if not (np.isfinite(referee) and referee >= REFEREE_R):
        return {"excluded_referee_r": referee}

    eog = _z(np.stack([p["veog"], p["heog"]]), p["fit"])
    rbf_state = {}

    def kernel_design(e):
        if "centers" not in rbf_state:
            idx = np.random.default_rng(3).choice(e.shape[1], size=16, replace=False)
            rbf_state["centers"] = e[:, idx]
        d = ((e[:, None, :] - rbf_state["centers"][:, :, None]) ** 2).sum(axis=0)
        return np.exp(-d / (2 * np.median(d) + 1e-9))

    designs = {
        "indicator_linear": lambda e: e,
        "rank3_derivative": lambda e: np.concatenate(
            (e, np.gradient(e[0])[None]), axis=0),
        "fir_lagged": lambda e: np.concatenate(
            [np.roll(e, l, axis=1) for l in (-2, -1, 0, 1, 2)], axis=0),
        "amplitude_gain": lambda e: np.concatenate(
            (e, e * np.sqrt(np.mean(e ** 2, axis=0, keepdims=True))), axis=0),
        "kernel_ridge": kernel_design,
    }
    residuals = {}
    target_fit = a_ref[:, p["fit"]]
    target_eval = _center(a_ref[:, p["eval"]])
    for name, fn in designs.items():
        op = _ridge_fit(target_fit, fn(eog)[:, p["fit"]])
        pred = _predict(op, fn(eog))[:, p["eval"]]
        residuals[name] = float(np.mean((target_eval - pred) ** 2))
    base = residuals["indicator_linear"]
    gains = {k: float((base - v) / max(base, 1e-12)) for k, v in residuals.items()}
    best = max((k for k in gains if k != "indicator_linear"), key=gains.get)
    return {"referee_r": referee, "residuals": residuals, "gains": gains,
            "best_family": best, "best_gain": gains[best],
            "blink_kernel": kernel_op, "fir_lags": fir_lags}


def t_row(p: dict, w3: dict) -> dict:
    fs = p["fs"]
    lags5 = [int(round(ms / 1000.0 * fs)) for ms in LAGS_MS]
    fir_lags = w3["fir_lags"]
    trains = _event_trains(p)
    # 1-D per-type template = frontal-mean of the W3 FIR kernel for that train.
    # _lagged interleaves rows as (lag, train): [lag0-blink, lag0-sacc, lag1-blink, ...]
    n_fir = len(fir_lags)
    kernel = w3["blink_kernel"][:, :2 * n_fir].mean(axis=0)
    conv = np.zeros_like(trains)
    for t_i in range(2):
        coeffs = kernel[t_i::2][:n_fir]
        for lag, c in zip(fir_lags, coeffs):
            conv[t_i] += c * np.roll(trains[t_i], lag)
    typed = np.concatenate([_z(conv, p["fit"]), _z(p["gaze"], p["fit"])], axis=0)
    static = _z(np.stack([p["veog"], p["heog"]]), p["fit"])

    window = int(WINDOW_S * fs)
    ev = p["eval"]
    n_win = len(ev) // window
    energy = (p["veog"][ev][:n_win * window].reshape(n_win, window) ** 2).mean(axis=1)
    rich = np.argsort(energy)[-max(int(round(n_win * ARTIFACT_RICH)), 1):]
    mask = np.zeros(len(ev), bool)
    for w in rich:
        mask[w * window:(w + 1) * window] = True

    out = {}
    for name, ref in (("static", static), ("typed", typed)):
        op = _ridge_fit(p["frontal"][:, p["fit"]], _lagged(ref, lags5)[:, p["fit"]])
        pred = _predict(op, _lagged(ref, lags5))[:, ev]
        target = _center(p["frontal"][:, ev])
        out[name] = float(np.mean((target[:, mask] - pred[:, mask]) ** 2))
    delta = (out["static"] - out["typed"]) / max(out["static"], 1e-12)
    return {"resid_static": out["static"], "resid_typed": out["typed"],
            "delta": float(delta), "n_rich_windows": int(len(rich))}


def main() -> None:
    w1_rows, w2_pairs, w3_rows, t_rows = [], [], [], []
    participants = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
    for participant in participants:
        for path in sorted((ROOT / participant).glob("*.mat")):
            name = f"{participant}/{path.name}"
            p = prepare(load_full(path))
            row1 = {"recording": name, "participant": participant, **w1_row(p)}
            w1_rows.append(row1)
            w2_pairs.extend((participant, true, pred)
                            for true, pred in w2_events(p))
            row3 = w3_row(p)
            row3_out = {"recording": name, "participant": participant,
                        **{k: v for k, v in (row3 or {}).items()
                           if k not in ("blink_kernel", "fir_lags")}}
            w3_rows.append(row3_out)
            if row3 and "gains" in row3:
                t_rows.append({"recording": name, "participant": participant,
                               **t_row(p, row3)})
            print(json.dumps({"recording": name,
                              "d_rms": round(row1["d_rms"], 4),
                              "w3": row3_out.get("best_gain",
                                                 row3_out.get("excluded_referee_r")),
                              "t_delta": (round(t_rows[-1]["delta"], 4)
                                          if t_rows and t_rows[-1]["recording"] == name
                                          else None)}), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    def participant_first(rows, field):
        per: dict[str, list] = {}
        for r in rows:
            if field in r and np.isfinite(r[field]):
                per.setdefault(r["participant"], []).append(r[field])
        return _stat([np.mean(v) for v in per.values()]), \
            _stat([r[field] for r in rows if field in r and np.isfinite(r[field])])

    pf, rl = participant_first(w1_rows, "d_rms")
    pfc, rlc = participant_first(w1_rows, "d_corr")
    (OUT / "w1_a4.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_w.md (W1)",
        "d_rms_participant_first": pf, "d_rms_recording_level": rl,
        "d_corr_participant_first": pfc, "d_corr_recording_level": rlc,
        "banked_wave4_axis_limited": {"d_rms": 1.7673, "d_corr": 0.9000,
                                      "note": "enters BESIDE, never over"},
        "rows": w1_rows}, indent=2, sort_keys=True) + "\n")

    truth = np.asarray([1 if t == "blink" else 0 for _, t, _ in w2_pairs])
    pred = np.asarray([1 if p_ == "blink" else 0 for _, _, p_ in w2_pairs])
    po = float((truth == pred).mean())
    pe = float(truth.mean() * pred.mean() + (1 - truth.mean()) * (1 - pred.mean()))
    kappa = (po - pe) / max(1 - pe, 1e-12)
    per_participant = {}
    for participant in participants:
        sub = [(t, p_) for pp, t, p_ in w2_pairs if pp == participant]
        if len(sub) < 10:
            continue
        st = np.asarray([1 if t == "blink" else 0 for t, _ in sub])
        sp = np.asarray([1 if p_ == "blink" else 0 for _, p_ in sub])
        so = float((st == sp).mean())
        se = float(st.mean() * sp.mean() + (1 - st.mean()) * (1 - sp.mean()))
        per_participant[participant] = (so - se) / max(1 - se, 1e-12)
    (OUT / "w2_kappa.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_w.md (W2)",
        "n_events": int(len(w2_pairs)), "pooled_kappa": kappa,
        "accuracy": po, "blink_prevalence": float(truth.mean()),
        "participant_kappa": _stat(list(per_participant.values())),
        "bar": KAPPA_BAR, "pass": bool(kappa >= KAPPA_BAR),
        "wave3_banked": -0.25}, indent=2, sort_keys=True) + "\n")

    included = [r for r in w3_rows if "best_gain" in r]
    excluded = [r for r in w3_rows if "best_gain" not in r]
    pf3, rl3 = participant_first(included, "best_gain")
    from collections import Counter
    (OUT / "w3_readout.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_w.md (W3)",
        "included": len(included), "excluded_referee": len(excluded),
        "best_gain_participant_first": pf3, "best_gain_recording_level": rl3,
        "best_family_counts": dict(Counter(r["best_family"] for r in included)),
        "bound": READOUT_BOUND,
        "verdict": ("SIZED" if pf3["mean"] >= READOUT_BOUND else "BOUNDED"),
        "wave4_banked": {"gain": 0.0620, "n_subjects": 3},
        "rows": w3_rows}, indent=2, sort_keys=True) + "\n")

    pft, rlt = participant_first(t_rows, "delta")
    gate = bool(pft["bootstrap_low"] > 0 and pft["mean"] >= T_GATE_MEAN)
    (OUT / "t_typed_info.json").write_text(json.dumps({
        "prereg": "reports/iris_prereg_w.md (T)",
        "delta_participant_first": pft, "delta_recording_level": rlt,
        "gate": {"ci_low_positive": bool(pft["bootstrap_low"] > 0),
                 "mean_floor": T_GATE_MEAN, "pass": gate,
                 "verdict": ("typed information REAL -> typed family proceeds to F2"
                             if gate else
                             "family-finality extends to the rich reference; "
                             "typed-family leg DROPPED (first-class negative)")},
        "rows": t_rows}, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"w1_d_rms": round(pf["mean"], 4), "w2_kappa": round(kappa, 4),
                      "w3_best_gain": round(pf3["mean"], 4),
                      "t_delta": round(pft["mean"], 4), "t_pass": gate}))


if __name__ == "__main__":
    main()
