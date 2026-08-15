#!/usr/bin/env python3
"""WAVE4-E2: the Tier-S-authorized measurements, in the authorized order M2 -> M4 -> M3.

Protocol frozen in reports/wave4_e2_preregistration.md (and Part 1/2 of
reports/wave4_preregistration.md). Tier-E is locked, so NO per-event blink label enters
any statistic here: every measurement is segment-level by construction.

Each recording is loaded once; the three measurements are computed and each is written to
disk immediately after its own stage, in the authorized order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from wave4_e1_align import EEG_ROOT, POSTERIOR, VALIDITY_CUT, read_eeg, read_tobii
from wave4_e1r import blink_candidates

REPO = Path(__file__).resolve().parents[1]
E1R = REPO / "results/wave4_optical/e1r/correspondence_table.csv"
OUT = REPO / "results/wave4_optical"

EEG_RATE = 1000.0
RIDGE = 0.05
LAGS_MS = (-100.0, -50.0, 0.0, 50.0, 100.0)
WINDOW_MS = 1000.0
FRONTAL = ("FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F8")
ARTIFACT_RICH = 0.20
MIN_RICH_WINDOWS = 20
FIX_VELOCITY = 30.0
FIX_SWEEP = (20.0, 30.0, 50.0)
FIX_MIN_MS = 100.0
FIX_MIN_TOTAL_MS = 60_000.0
OPERA_LEAKAGE = 0.055
READOUT_BOUND = 0.03
N_BOOT = 5000
SEED = 420


def _ridge_fit(target: np.ndarray, drive: np.ndarray, ratio: float = RIDGE) -> np.ndarray:
    """Program convention (run_wave3._ridge): centred ridge, trace-normalised penalty."""
    y = target - target.mean(axis=1, keepdims=True)
    e = drive - drive.mean(axis=1, keepdims=True)
    gram = e @ e.T
    penalty = float(ratio) * max(float(np.trace(gram) / len(gram)), np.finfo(float).eps)
    return (y @ e.T) @ np.linalg.inv(gram + penalty * np.eye(len(gram)))


def _bootstrap(values, seed: int = SEED, draws: int = N_BOOT) -> dict[str, Any]:
    series = np.asarray(list(values), float)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return {"mean": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(series, len(series), replace=True).mean()
                        for _ in range(draws)])
    return {"mean": float(series.mean()), "median": float(np.median(series)),
            "n": int(len(series)), "bootstrap_low": float(np.quantile(means, 0.025)),
            "bootstrap_high": float(np.quantile(means, 0.975))}


def _participant_first(rows: list[dict], key: str, seed: int = SEED) -> dict[str, Any]:
    """Primary: mean within participant, then bootstrap over participants."""
    by_subject: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            by_subject.setdefault(row["subject"], []).append(float(value))
    per = {s: float(np.mean(v)) for s, v in by_subject.items()}
    primary = _bootstrap(list(per.values()), seed=seed)
    primary["per_subject"] = per
    secondary = _bootstrap([row[key] for row in rows
                            if row.get(key) is not None and np.isfinite(row[key])], seed=seed)
    return {"participant_first_primary": primary,
            "recording_level_secondary_declared": secondary}


def _lagged(design: np.ndarray) -> np.ndarray:
    """Frozen FIR lag set applied to every regressor row."""
    shifts = [int(round(lag * EEG_RATE / 1000.0)) for lag in LAGS_MS]
    return np.concatenate([np.roll(design, s, axis=1) for s in shifts], axis=0)


def optical_block(eeg: dict, tob: dict, slope: float, intercept: float) -> dict[str, Any]:
    """Optical regressors resampled onto the EEG clock (frozen definition)."""
    time_eeg = eeg["time_s"] * 1000.0
    stamp = tob["stamp_ms"]
    valid = (tob["validity_left"] <= VALIDITY_CUT) & (tob["validity_right"] <= VALIDITY_CUT)
    # gaze in degrees of visual angle from ADCSmm + eye distance
    distance = np.nanmean(np.stack([tob["distance_left"], tob["distance_right"]]), axis=0)
    distance = np.where(np.isfinite(distance) & (distance > 1.0), distance, np.nan)
    if np.isfinite(distance).any():
        distance = np.where(np.isfinite(distance), distance,
                            np.nanmedian(distance[np.isfinite(distance)]))
    else:
        distance = np.full(len(stamp), 600.0)
    def _deg(mm):
        centred = mm - (np.nanmedian(mm[np.isfinite(mm)]) if np.isfinite(mm).any() else 0.0)
        return np.degrees(np.arctan2(centred, distance))
    gx, gy = _deg(tob["gaze_mm_x"]), _deg(tob["gaze_mm_y"])
    # sample-and-hold through invalid stretches
    def _hold(series):
        out = series.copy()
        bad = ~np.isfinite(out)
        if bad.all():
            return np.zeros_like(out)
        index = np.where(~bad, np.arange(len(out)), 0)
        np.maximum.accumulate(index, out=index)
        out = out[index]
        out[~np.isfinite(out)] = 0.0
        return out
    gx, gy = _hold(gx), _hold(gy)
    dt = np.gradient(stamp) / 1000.0
    speed = np.sqrt(np.gradient(gx) ** 2 + np.gradient(gy) ** 2) / np.maximum(dt, 1e-6)
    pupil = np.nanmean(np.stack([tob["pupil_left"], tob["pupil_right"]]), axis=0)
    pupil = _hold(pupil)
    # map Tobii samples onto the EEG clock, then sample-and-hold at 1000 Hz
    tob_in_eeg = slope * stamp + intercept
    index = np.clip(np.searchsorted(tob_in_eeg, time_eeg, side="right") - 1, 0, len(stamp) - 1)
    covered = (time_eeg >= tob_in_eeg[0]) & (time_eeg <= tob_in_eeg[-1])
    inval = (~valid)[index].astype(float)
    block = np.stack([inval, gx[index], gy[index], speed[index], pupil[index]])
    finite = np.isfinite(block)
    block = np.where(finite, block, 0.0)
    std = block.std(axis=1, keepdims=True)
    block = (block - block.mean(axis=1, keepdims=True)) / np.where(std > 1e-9, std, 1.0)
    # blink-candidate mask (segment-level exclusion only; never an event label)
    blink_mask = np.zeros(len(time_eeg), bool)
    for onset, duration in blink_candidates(tob):
        lo = np.searchsorted(time_eeg, slope * onset + intercept, side="left")
        hi = np.searchsorted(time_eeg, slope * (onset + duration) + intercept, side="right")
        blink_mask[lo:hi] = True
    return {"block": block, "covered": covered, "speed_eeg": speed[index],
            "valid_eeg": valid[index], "blink_mask": blink_mask,
            "inval_eeg": inval}


def _windows(n_samples: int):
    size = int(WINDOW_MS * EEG_RATE / 1000.0)
    count = n_samples // size
    return np.arange(count) * size, size


def _cv_residual(target: np.ndarray, design: np.ndarray) -> float:
    """2-fold CV MSE over disjoint contiguous halves (banked convention)."""
    half = design.shape[1] // 2
    if half < 10:
        return float("nan")
    out = []
    for a, b in ((slice(0, half), slice(half, design.shape[1])),
                 (slice(half, design.shape[1]), slice(0, half))):
        operator = _ridge_fit(target[:, a], design[:, a])
        xb = design[:, b] - design[:, b].mean(axis=1, keepdims=True)
        yb = target[:, b] - target[:, b].mean(axis=1, keepdims=True)
        out.append(float(np.mean(np.square(yb - operator @ xb))))
    return float(np.mean(out))


def _cv_predict(target: np.ndarray, design: np.ndarray) -> np.ndarray | None:
    """Held-out predictions from the same 2-fold split."""
    half = design.shape[1] // 2
    if half < 10:
        return None
    prediction = np.zeros_like(target)
    for a, b in ((slice(0, half), slice(half, design.shape[1])),
                 (slice(half, design.shape[1]), slice(0, half))):
        operator = _ridge_fit(target[:, a], design[:, a])
        xb = design[:, b] - design[:, b].mean(axis=1, keepdims=True)
        prediction[:, b] = operator @ xb
    return prediction


def load_all() -> list[dict[str, Any]]:
    import csv

    eligible = [r for r in csv.DictReader(E1R.open()) if r["eligible"] == "True"]
    loaded = []
    for row in eligible:
        subject, session, name = row["subject"], row["session"], row["name"]
        slope, intercept = float(row["clock_slope"]), 0.0
        # intercept is not in the E1R table; recover it from the banked E1 artifact
        banked = json.loads((REPO / "results/wave4_optical/alignment/e1_pilot.json").read_text())
        for entry in banked["per_recording"]:
            if entry["recording"] == row["recording"]:
                slope, intercept = float(entry["slope"]), float(entry["intercept_ms"])
                break
        eeg = read_eeg(subject, session, name, channels=FRONTAL + POSTERIOR)
        tob = read_tobii(subject, session, name)
        optical = optical_block(eeg, tob, slope, intercept)
        loaded.append({"recording": row["recording"], "subject": subject,
                       "session": session, "name": name, "eeg": eeg, "tob": tob,
                       "optical": optical})
        print(f"loaded {row['recording']}", flush=True)
    return loaded


# ------------------------------------------------------------------ M2

def run_m2(records: list[dict]) -> dict[str, Any]:
    rows = []
    for record in records:
        eeg, optical = record["eeg"], record["optical"]
        channels = [c for c in FRONTAL if c in eeg["extra"]]
        if not channels or eeg["heo"] is None:
            rows.append({"recording": record["recording"], "subject": record["subject"],
                         "excluded": "no frontal block or HEO"})
            continue
        target_full = np.stack([eeg["extra"][c] for c in channels])
        heo_full = eeg["heo"][None, :]
        starts, size = _windows(target_full.shape[1])
        inval = optical["inval_eeg"]
        covered = optical["covered"]
        keep = []
        for start in starts:
            sl = slice(start, start + size)
            if covered[sl].mean() > 0.99 and inval[sl].mean() >= ARTIFACT_RICH:
                keep.append(sl)
        if len(keep) < MIN_RICH_WINDOWS:
            rows.append({"recording": record["recording"], "subject": record["subject"],
                         "artifact_rich_windows": len(keep),
                         "excluded": f"only {len(keep)} artifact-rich windows"})
            continue
        index = np.concatenate([np.arange(sl.start, sl.stop) for sl in keep])
        target = np.nan_to_num(target_full[:, index])
        # lags are built on the FULL timeline, then subset, so no seam of the
        # concatenated artifact-rich windows leaks into a lagged regressor
        design_eog = _lagged(np.nan_to_num(heo_full))[:, index]
        design_opt = _lagged(optical["block"])[:, index]
        a_eog = _cv_predict(target, design_eog)
        a_opt = _cv_predict(target, design_opt)
        if a_eog is None or a_opt is None:
            rows.append({"recording": record["recording"], "subject": record["subject"],
                         "excluded": "insufficient samples for 2-fold CV"})
            continue
        per_channel = []
        for row_index, channel in enumerate(channels):
            u, v = a_eog[row_index], a_opt[row_index]
            denominator = float(np.sqrt(np.mean(u ** 2)))
            d_rms = float(np.sqrt(np.mean((u - v) ** 2)) / denominator) if denominator > 0 \
                else float("nan")
            if u.std() > 0 and v.std() > 0:
                d_corr = float(1.0 - np.corrcoef(u, v)[0, 1])
            else:
                d_corr = float("nan")
            per_channel.append({"channel": channel, "D_rms": d_rms, "D_corr": d_corr})
        rows.append({
            "recording": record["recording"], "subject": record["subject"],
            "artifact_rich_windows": len(keep), "channels": len(channels),
            "D_rms": float(np.nanmean([c["D_rms"] for c in per_channel])),
            "D_corr": float(np.nanmean([c["D_corr"] for c in per_channel])),
            "per_channel": per_channel, "excluded": ""})
    included = [r for r in rows if not r.get("excluded")]
    payload = {
        "measurement": "M2 — A4 reference-channel-error row",
        "authorized_by": "E1R TIER-S",
        "frozen": {"artifact_rich_invalidity": ARTIFACT_RICH,
                   "min_rich_windows": MIN_RICH_WINDOWS, "ridge": RIDGE,
                   "lags_ms": list(LAGS_MS), "frontal_block": list(FRONTAL)},
        "recordings_included": len(included), "recordings_excluded": len(rows) - len(included),
        "D_rms": _participant_first(included, "D_rms"),
        "D_corr": _participant_first(included, "D_corr"),
        "interpretation": ("UPPER BOUND on reference-channel error: contains EOG "
                           "measurement noise and neural crosstalk together with the "
                           "optical reference's own limitations, which this panel cannot "
                           "separate. Never a point estimate of EOG error."),
        "cross_panel_caveat": ("measured on Eye-BCI; enters the MobileBCI ledger as an "
                               "order-of-magnitude BOUND with an explicit cross-panel "
                               "label; no comparability check was performed"),
        "per_recording": rows,
    }
    (OUT / "m2").mkdir(parents=True, exist_ok=True)
    (OUT / "m2/m2_reference_error.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"M2_D_rms_mean": payload["D_rms"]["participant_first_primary"]["mean"],
                      "included": len(included)}))
    return payload


# ------------------------------------------------------------------ M4

def _fixation_mask(record: dict, threshold: float) -> np.ndarray:
    optical = record["optical"]
    base = (optical["valid_eeg"] & (optical["speed_eeg"] < threshold)
            & ~optical["blink_mask"] & optical["covered"])
    # enforce the >=100 ms sustained requirement
    minimum = int(FIX_MIN_MS * EEG_RATE / 1000.0)
    mask = np.zeros_like(base)
    start = None
    for index, flag in enumerate(np.append(base, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                mask[start:index] = True
            start = None
    return mask


def run_m4(records: list[dict]) -> dict[str, Any]:
    def measure(threshold: float) -> list[dict]:
        rows = []
        for record in records:
            eeg = record["eeg"]
            channels = [c for c in POSTERIOR if c in eeg["extra"]]
            mask = _fixation_mask(record, threshold)
            duration_ms = float(mask.sum() * 1000.0 / EEG_RATE)
            if eeg["heo"] is None or not channels or duration_ms < FIX_MIN_TOTAL_MS:
                rows.append({"recording": record["recording"], "subject": record["subject"],
                             "fixation_ms": duration_ms,
                             "excluded": f"fixation {duration_ms:.0f} ms < {FIX_MIN_TOTAL_MS:.0f} ms"
                             if duration_ms < FIX_MIN_TOTAL_MS else "no posterior block or HEO"})
                continue
            posterior = np.nan_to_num(np.stack([eeg["extra"][c] for c in channels])[:, mask])
            heo = np.nan_to_num(eeg["heo"][None, mask])
            operator = _ridge_fit(heo, posterior)
            centred = posterior - posterior.mean(axis=1, keepdims=True)
            target = heo - heo.mean(axis=1, keepdims=True)
            prediction = operator @ centred
            total = float(np.mean(target ** 2))
            r2_in = float(1.0 - np.mean((target - prediction) ** 2) / total) if total > 0 \
                else float("nan")
            cv_residual = _cv_residual(heo, posterior)
            r2_cv = float(1.0 - cv_residual / total) if total > 0 and np.isfinite(cv_residual) \
                else float("nan")
            rows.append({"recording": record["recording"], "subject": record["subject"],
                         "fixation_ms": duration_ms, "channels": len(channels),
                         "R2_in_sample": r2_in, "R2_cv": r2_cv, "excluded": ""})
        return rows

    rows = measure(FIX_VELOCITY)
    included = [r for r in rows if not r.get("excluded")]
    sweep = {}
    for threshold in FIX_SWEEP:
        swept = [r for r in measure(threshold) if not r.get("excluded")]
        sweep[f"{threshold:g}deg_s"] = {
            "included": len(swept),
            "R2_in_sample": _participant_first(swept, "R2_in_sample")["participant_first_primary"],
            "R2_cv": _participant_first(swept, "R2_cv")["participant_first_primary"]}
    payload = {
        "measurement": "M4 — exogeneity / neural crosstalk into the EOG reference",
        "authorized_by": "E1R TIER-S",
        "frozen": {"velocity_deg_s": FIX_VELOCITY, "sustained_ms": FIX_MIN_MS,
                   "min_fixation_ms": FIX_MIN_TOTAL_MS, "ridge": RIDGE,
                   "posterior_block": list(POSTERIOR)},
        "mask_note": ("segment mask from directly measured validity+velocity; NOT a "
                      "kappa-validated event label (M1 is locked by TIER-E)"),
        "recordings_included": len(included), "recordings_excluded": len(rows) - len(included),
        "R2_in_sample": _participant_first(included, "R2_in_sample"),
        "R2_cv": _participant_first(included, "R2_cv"),
        "opera_reference_leakage_r2": OPERA_LEAKAGE,
        "velocity_sweep": sweep,
        "cross_panel_caveat": ("measured on Eye-BCI; enters the MobileBCI ledger as an "
                               "order-of-magnitude BOUND with an explicit cross-panel label"),
        "per_recording": rows,
    }
    (OUT / "m4").mkdir(parents=True, exist_ok=True)
    (OUT / "m4/m4_exogeneity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"M4_R2_in": payload["R2_in_sample"]["participant_first_primary"]["mean"],
                      "M4_R2_cv": payload["R2_cv"]["participant_first_primary"]["mean"],
                      "included": len(included)}))
    return payload


# ------------------------------------------------------------------ M3

def run_m3(records: list[dict]) -> dict[str, Any]:
    rows = []
    for record in records:
        eeg, optical = record["eeg"], record["optical"]
        channels = [c for c in FRONTAL if c in eeg["extra"]]
        covered = optical["covered"]
        if not channels or covered.sum() < 2 * int(WINDOW_MS):
            rows.append({"recording": record["recording"], "subject": record["subject"],
                         "excluded": "no frontal block or insufficient optical coverage"})
            continue
        index = np.flatnonzero(covered)
        target = np.nan_to_num(np.stack([eeg["extra"][c] for c in channels])[:, index])
        full = optical["block"]
        # every family design is built on the FULL timeline, then subset to covered
        block = full[:, index]
        derivative = np.concatenate((full, np.gradient(full[0])[None]), axis=0)[:, index]
        lagged = _lagged(full)[:, index]
        amplitude = np.concatenate(
            (full, full * np.sqrt(np.mean(full ** 2, axis=0, keepdims=True))), axis=0)[:, index]
        families = {"indicator_linear": _cv_residual(target, block),
                    "rank3_derivative": _cv_residual(target, derivative),
                    "fir_lagged": _cv_residual(target, lagged),
                    "amplitude_gain": _cv_residual(target, amplitude)}
        rng = np.random.default_rng(3)
        centers = block[:, rng.choice(block.shape[1], size=16, replace=False)]
        distances = ((block[:, None, :] - centers[:, :, None]) ** 2).sum(axis=0)
        families["kernel_ridge"] = _cv_residual(
            target, np.exp(-distances / (2 * np.median(distances) + 1e-9)))
        linear = families["indicator_linear"]
        analytic_names = [k for k in families if k != "indicator_linear"
                          and np.isfinite(families[k])]
        if not np.isfinite(linear) or not analytic_names:
            rows.append({"recording": record["recording"], "subject": record["subject"],
                         "excluded": "readout families not evaluable"})
            continue
        best = min(analytic_names, key=lambda k: families[k])
        gain = float((linear - families[best]) / max(linear, 1e-12))
        rows.append({"recording": record["recording"], "subject": record["subject"],
                     "residual_linear": linear, "residual_analytic": families[best],
                     "best_analytic_family": best, "relative_gain": gain,
                     **{f"cv_{k}": v for k, v in families.items()}, "excluded": ""})
    included = [r for r in rows if not r.get("excluded")]
    stats = _participant_first(included, "relative_gain")
    mean_gain = stats["participant_first_primary"]["mean"]
    sized = bool(np.isfinite(mean_gain) and mean_gain > READOUT_BOUND)
    counts: dict[str, int] = {}
    for row in included:
        counts[row["best_analytic_family"]] = counts.get(row["best_analytic_family"], 0) + 1
    payload = {
        "measurement": "M3 — readout bound, SEGMENT-LEVEL variant",
        "authorized_by": "E1R TIER-S (event-level/full M3 remains locked by TIER-E)",
        "diff_class_readout": "NOT-MEASURABLE-THIS-PANEL (no V44-class checkpoint for the "
                              "62-channel Neuroscan montage; checkpoint porting prohibited)",
        "frozen": {"threshold_relative_gain": READOUT_BOUND, "ridge": RIDGE,
                   "linear_family": "indicator_linear",
                   "analytic_families": ["rank3_derivative", "fir_lagged",
                                         "amplitude_gain", "kernel_ridge"]},
        "recordings_included": len(included), "recordings_excluded": len(rows) - len(included),
        "relative_gain": stats,
        "best_analytic_family_counts": counts,
        "verdict": "SIZED" if sized else "BOUNDED",
        "readout_row": (f"sized at {mean_gain:.4f}" if sized
                        else f"bounded at {READOUT_BOUND}"),
        "cross_panel_caveat": ("measured on Eye-BCI; enters the MobileBCI ledger as an "
                               "order-of-magnitude BOUND with an explicit cross-panel label"),
        "per_recording": rows,
    }
    (OUT / "m3").mkdir(parents=True, exist_ok=True)
    (OUT / "m3/m3_readout_bound.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"M3_relative_gain": mean_gain, "verdict": payload["verdict"],
                      "included": len(included)}))
    return payload


def main() -> None:
    records = load_all()
    print(f"--- E2 substrate: {len(records)} eligible recordings", flush=True)
    print("--- M2", flush=True)
    run_m2(records)
    print("--- M4", flush=True)
    run_m4(records)
    print("--- M3", flush=True)
    run_m3(records)


if __name__ == "__main__":
    main()
