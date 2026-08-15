#!/usr/bin/env python3
"""WAVE4-E1R: the one registered repair of the optical-correspondence instrument.

Protocol frozen in reports/wave4_e1r_preregistration.md. The clock layer is NOT
recomputed: per-recording slope/intercept and the clock-valid set are read verbatim from
the banked E1 artifact. Only the correspondence layer is rebuilt.

  REPAIR 1  physiological lag window [-20, +120] ms (lag = VEOG peak - tracking loss)
  REPAIR 2  fragmentation-robust segmentation (merge <=40 ms, duration 50-500 ms)
  REPAIR 3  reverse VEOG-anchored instrument vs circular-shift null (co-primary)
  REPAIR 4  eligibility at Tobii validity fraction >= 0.40 (sweep 0.30-0.60 reported)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from wave4_e1_align import (EEG_ROOT, VALIDITY_CUT, read_eeg, read_tobii, tobii_root,
                            veog_deflections)

REPO = Path(__file__).resolve().parents[1]
BANKED = REPO / "results/wave4_optical/alignment/e1_pilot.json"
OUT = REPO / "results/wave4_optical/e1r"

LAG_LOW_MS, LAG_HIGH_MS = -20.0, 120.0     # REPAIR 1 (frozen from blink kinematics)
MERGE_GAP_MS = 40.0                        # REPAIR 2
BLINK_MIN_MS, BLINK_MAX_MS = 50.0, 500.0   # REPAIR 2
ELIGIBILITY = 0.40                         # REPAIR 4 (frozen; sweep reported)
SWEEP = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
N_SHIFTS = 200
N_BOOT = 2000
TIER_S_FRACTION = 0.60
TIER_E_MEDIAN = 0.70
TIER_E_P = 0.01
TIER_E_FRACTION = 0.60
RNG_SEED = 20260815


def _circular_shifts(train: np.ndarray, low: float, high: float) -> list[np.ndarray]:
    """N_SHIFTS circular shifts of a point train inside [low, high]."""
    span = max(high - low, 1.0)
    out = []
    for k in range(1, N_SHIFTS + 1):
        delta = span * k / (N_SHIFTS + 1.0)
        out.append(low + np.sort(((train - low) + delta) % span))
    return out


def blink_candidates(tob: dict[str, Any]) -> list[tuple[float, float]]:
    """REPAIR 2: merge fragmented invalid runs, then keep 50-500 ms candidates.

    The E1 '>=50 ms valid flanking' clause is deliberately absent: at ~59% tracker
    invalidity it rejects real blinks. Replaced by the merge+duration criterion.
    """
    stamp = tob["stamp_ms"]
    valid = (tob["validity_left"] <= VALIDITY_CUT) & (tob["validity_right"] <= VALIDITY_CUT)
    invalid = ~valid
    runs = []
    start = None
    for index, flag in enumerate(invalid):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append([start, index])
            start = None
    if start is not None:
        runs.append([start, len(invalid)])
    merged: list[list[int]] = []
    for run in runs:
        if merged and stamp[run[0]] - stamp[merged[-1][1] - 1] <= MERGE_GAP_MS:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))
    out = []
    for begin, end in merged:
        if end <= begin or end > len(stamp):
            continue
        duration = stamp[end - 1] - stamp[begin]
        if BLINK_MIN_MS <= duration <= BLINK_MAX_MS:
            out.append((float(stamp[begin]), float(duration)))
    return out


def _forward_rate(onsets_eeg: np.ndarray, deflections: np.ndarray) -> float:
    """Fraction of blink candidates with a VEOG peak inside the frozen lag window."""
    if len(onsets_eeg) == 0 or len(deflections) == 0:
        return float("nan")
    lo = np.searchsorted(deflections, onsets_eeg + LAG_LOW_MS, side="left")
    hi = np.searchsorted(deflections, onsets_eeg + LAG_HIGH_MS, side="right")
    return float(np.mean(hi > lo))


def _reverse_stat(centres_eeg: np.ndarray, slope: float, intercept: float,
                  stamp: np.ndarray, invalid_cumsum: np.ndarray) -> float:
    """Mean Tobii invalidity inside the implied tracking-loss window of each VEOG blink."""
    if len(centres_eeg) == 0:
        return float("nan")
    lo_eeg = centres_eeg - LAG_HIGH_MS
    hi_eeg = centres_eeg - LAG_LOW_MS
    lo = np.searchsorted(stamp, (lo_eeg - intercept) / slope, side="left")
    hi = np.searchsorted(stamp, (hi_eeg - intercept) / slope, side="right")
    counts = hi - lo
    keep = counts > 0
    if not keep.any():
        return float("nan")
    sums = invalid_cumsum[hi[keep]] - invalid_cumsum[lo[keep]]
    return float(np.mean(sums / counts[keep]))


def _reverse_per_window(centres_eeg: np.ndarray, slope: float, intercept: float,
                        stamp: np.ndarray, invalid_cumsum: np.ndarray) -> np.ndarray:
    lo_eeg = centres_eeg - LAG_HIGH_MS
    hi_eeg = centres_eeg - LAG_LOW_MS
    lo = np.searchsorted(stamp, (lo_eeg - intercept) / slope, side="left")
    hi = np.searchsorted(stamp, (hi_eeg - intercept) / slope, side="right")
    counts = hi - lo
    keep = counts > 0
    if not keep.any():
        return np.asarray([])
    return (invalid_cumsum[hi[keep]] - invalid_cumsum[lo[keep]]) / counts[keep]


def measure(subject: str, session: str, name: str, slope: float, intercept: float,
            rng: np.random.Generator) -> dict[str, Any]:
    eeg = read_eeg(subject, session, name)
    tob = read_tobii(subject, session, name)
    stamp = tob["stamp_ms"]
    valid = (tob["validity_left"] <= VALIDITY_CUT) & (tob["validity_right"] <= VALIDITY_CUT)
    validity_fraction = float(np.mean(valid)) if len(valid) else 0.0
    invalid_cumsum = np.concatenate([[0.0], np.cumsum((~valid).astype(float))])
    deflections = np.sort(veog_deflections(eeg))
    candidates = blink_candidates(tob)
    onsets_tob = np.asarray([c[0] for c in candidates])
    onsets_eeg = slope * onsets_tob + intercept if len(onsets_tob) else np.asarray([])

    row: dict[str, Any] = {
        "recording": f"{subject}/{session}/{name}", "subject": subject,
        "session": session, "name": name,
        "validity_fraction": validity_fraction,
        "n_veog": int(len(deflections)), "n_blink_candidates": int(len(candidates)),
        "forward_match_rate": float("nan"), "forward_p": float("nan"),
        "reverse_observed": float("nan"), "reverse_null_mean": float("nan"),
        "reverse_elevation": float("nan"), "reverse_ci_low": float("nan"),
        "reverse_ci_high": float("nan"), "reverse_p": float("nan"),
        "reverse_positive": False, "pupil_collapse_frac": float("nan"),
    }

    # --- forward instrument (repaired segmentation + physiological window)
    if len(onsets_eeg) and len(deflections):
        observed = _forward_rate(onsets_eeg, deflections)
        nulls = [_forward_rate(shift, deflections) for shift in
                 _circular_shifts(onsets_eeg, float(eeg["time_s"][0] * 1000.0),
                                  float(eeg["time_s"][-1] * 1000.0))]
        nulls = np.asarray([n for n in nulls if np.isfinite(n)])
        row["forward_match_rate"] = observed
        if len(nulls):
            row["forward_p"] = float((np.sum(nulls >= observed) + 1) / (len(nulls) + 1))

    # --- reverse instrument (co-primary, segmentation-free)
    if len(deflections) and len(stamp):
        per_window = _reverse_per_window(deflections, slope, intercept, stamp, invalid_cumsum)
        if len(per_window):
            observed = float(np.mean(per_window))
            nulls = np.asarray([
                _reverse_stat(shift, slope, intercept, stamp, invalid_cumsum)
                for shift in _circular_shifts(deflections,
                                              float(eeg["time_s"][0] * 1000.0),
                                              float(eeg["time_s"][-1] * 1000.0))])
            nulls = nulls[np.isfinite(nulls)]
            null_mean = float(np.mean(nulls)) if len(nulls) else float("nan")
            draws = rng.integers(0, len(per_window), size=(N_BOOT, len(per_window)))
            boot = per_window[draws].mean(axis=1) - null_mean
            row.update({
                "reverse_observed": observed, "reverse_null_mean": null_mean,
                "reverse_elevation": observed - null_mean,
                "reverse_ci_low": float(np.percentile(boot, 2.5)),
                "reverse_ci_high": float(np.percentile(boot, 97.5)),
                "reverse_p": float((np.sum(nulls >= observed) + 1) / (len(nulls) + 1))
                if len(nulls) else float("nan")})
            row["reverse_positive"] = bool(row["reverse_ci_low"] > 0)

    # --- auxiliary only (enters no gate): pupil collapse across blink candidates
    pupil = np.nanmean(np.stack([tob["pupil_left"], tob["pupil_right"]]), axis=0)
    finite = np.isfinite(pupil)
    if finite.any() and candidates:
        baseline = float(np.median(pupil[finite]))
        collapsed = 0
        for onset, duration in candidates:
            lo = int(np.searchsorted(stamp, onset, side="left"))
            hi = int(np.searchsorted(stamp, onset + duration, side="right"))
            window = pupil[lo:hi]
            if len(window) == 0:
                continue
            good = window[np.isfinite(window)]
            if len(good) == 0 or float(np.min(good)) <= 0.5 * baseline:
                collapsed += 1
        row["pupil_collapse_frac"] = collapsed / len(candidates)
    return row


def _tiers(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    eligible = [r for r in rows if r["validity_fraction"] >= threshold]
    rev = [r for r in eligible if np.isfinite(r["reverse_elevation"])]
    positives = [r for r in rev if r["reverse_positive"]]
    frac_positive = len(positives) / len(rev) if rev else 0.0
    pooled_low = pooled = float("nan")
    if rev:
        values = np.asarray([r["reverse_elevation"] for r in rev])
        rng = np.random.default_rng(RNG_SEED + 1)
        draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))
        boot = values[draws].mean(axis=1)
        pooled = float(values.mean())
        pooled_low = float(np.percentile(boot, 2.5))
    tier_s = bool(rev and frac_positive >= TIER_S_FRACTION and pooled_low > 0)

    fwd = [r for r in eligible if np.isfinite(r["forward_match_rate"])]
    median_forward = float(np.median([r["forward_match_rate"] for r in fwd])) if fwd else float("nan")
    sig = [r for r in fwd if np.isfinite(r["forward_p"]) and r["forward_p"] < TIER_E_P]
    frac_sig = len(sig) / len(fwd) if fwd else 0.0
    tier_e = bool(fwd and median_forward >= TIER_E_MEDIAN and frac_sig >= TIER_E_FRACTION)
    return {"threshold": threshold, "eligible_n": len(eligible),
            "reverse_evaluable_n": len(rev), "reverse_positive_n": len(positives),
            "reverse_positive_fraction": frac_positive,
            "pooled_elevation": pooled, "pooled_elevation_ci_low": pooled_low,
            "tier_s": tier_s,
            "forward_evaluable_n": len(fwd), "forward_median": median_forward,
            "forward_p_lt_0.01_n": len(sig), "forward_p_fraction": frac_sig,
            "tier_e": tier_e}


def main() -> None:
    banked = json.loads(BANKED.read_text())
    todo = [r for r in banked["per_recording"] if r.get("clock_gate_passed")]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for entry in todo:
        row = measure(entry["subject"], entry["session"], entry["name"],
                      float(entry["slope"]), float(entry["intercept_ms"]), rng)
        row["clock_slope"] = float(entry["slope"])
        row["clock_residual_ms"] = float(entry["drift_residual_ms"])
        rows.append(row)
        print(json.dumps({k: row[k] for k in
                          ("recording", "validity_fraction", "n_blink_candidates",
                           "forward_match_rate", "reverse_elevation", "reverse_ci_low")}),
              flush=True)
    for row in rows:
        row["eligible"] = bool(row["validity_fraction"] >= ELIGIBILITY)
        row["exclusion_reason"] = "" if row["eligible"] else \
            f"validity {row['validity_fraction']:.3f} < {ELIGIBILITY}"

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["recording", "subject", "session", "name", "validity_fraction", "eligible",
              "exclusion_reason", "clock_slope", "clock_residual_ms", "n_veog",
              "n_blink_candidates", "forward_match_rate", "forward_p", "reverse_observed",
              "reverse_null_mean", "reverse_elevation", "reverse_ci_low",
              "reverse_ci_high", "reverse_p", "reverse_positive", "pupil_collapse_frac"]
    with (OUT / "correspondence_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    frozen = _tiers(rows, ELIGIBILITY)
    decision = {
        "instrument": "wave4-e1r",
        "preregistration": "reports/wave4_e1r_preregistration.md",
        "clock_layer": "frozen; reused verbatim from banked E1 (never recomputed)",
        "recordings_measured": len(rows),
        "frozen_eligibility": ELIGIBILITY,
        "lag_window_ms": [LAG_LOW_MS, LAG_HIGH_MS],
        "tier_s": frozen["tier_s"], "tier_e": frozen["tier_e"],
        "eligible_n": frozen["eligible_n"],
        "close_fired": bool(not frozen["tier_s"]),
        "verdict_at_frozen_threshold": frozen,
        "sweep": [_tiers(rows, t) for t in SWEEP],
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: decision[k] for k in
                      ("tier_s", "tier_e", "eligible_n", "close_fired")}))


if __name__ == "__main__":
    main()
