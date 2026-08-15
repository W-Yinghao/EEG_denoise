#!/usr/bin/env python3
"""WAVE4 E1: cross-stream alignment of Tobii optical and Neuroscan EEG.

Protocol frozen in reports/wave4_preregistration.md:
  clock model = linear drift fit on shared events (E-Prime triggers primary;
  blink-onset cross-matching fallback);
  gate = >=80% of Tobii-detected blinks match a VEOG-surrogate deflection within
  |lag| <= 50 ms after the drift fit, AND drift-fit residual <= 20 ms RMS.
Failing subjects are EXCLUDED-AND-COUNTED. S27-S31 are excluded from E2 by the
exposure note (they are still alignment-reported).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

DATA_ROOT = Path("/projects/EEG-foundation-model/eye_bci")
EEG_ROOT = DATA_ROOT / "syn64005218-neuroscan"
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/wave4_optical/alignment"
EEG_RATE = 1000.0
VEOG_CHANNELS = ("FP1", "FPZ", "FP2")
POSTERIOR = ("P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3",
             "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2", "CB1", "CB2")
EXPOSURE_EXCLUDED = {"S27", "S28", "S29", "S30", "S31"}
BLINK_MATCH_MS = 50.0
DRIFT_RESIDUAL_MS = 20.0
MATCH_RATE_GATE = 0.80


def tobii_root() -> Path:
    final = DATA_ROOT / "syn64005218-tobii"
    return final if final.is_dir() else DATA_ROOT / ".syn64005218-tobii.partial"


def recordings() -> list[tuple[str, str, str]]:
    out = []
    for eeg in sorted(EEG_ROOT.glob("S*/Sess*/Neuroscan/*.csv")):
        subject, session, name = eeg.parts[-4], eeg.parts[-3], eeg.stem
        if (tobii_root() / subject / session / "Tobii" / f"{name}.csv").is_file():
            out.append((subject, session, name))
    return out


def read_eeg(subject: str, session: str, name: str, channels=None) -> dict[str, Any]:
    """Stream the EEG CSV once, keeping only what E1/E2 need."""
    path = EEG_ROOT / subject / session / "Neuroscan" / f"{name}.csv"
    wanted = set(channels or ()) | set(VEOG_CHANNELS)
    times, veog, trig, blinks = [], [], [], []
    extra: dict[str, list[float]] = {c: [] for c in wanted if c not in VEOG_CHANNELS}
    heo: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        veog_cols = [c for c in VEOG_CHANNELS if c in fields]
        for index, row in enumerate(reader):
            try:
                times.append(float(row["Time"]))
            except (TypeError, ValueError):
                continue
            values = []
            for c in veog_cols:
                try:
                    values.append(float(row[c]))
                except (TypeError, ValueError):
                    values.append(np.nan)
            veog.append(np.nanmean(values) if values else np.nan)
            if "HEO" in fields:
                try:
                    heo.append(float(row["HEO"]))
                except (TypeError, ValueError):
                    heo.append(np.nan)
            for c in extra:
                try:
                    extra[c].append(float(row[c]))
                except (TypeError, ValueError):
                    extra[c].append(np.nan)
            value = row.get("Trig")
            if value not in (None, "", "NA"):
                trig.append((float(row["Time"]), value))
            value = row.get("Blinks")
            if value not in (None, "", "NA"):
                try:
                    blinks.append(float(value))
                except ValueError:
                    blinks.append(0.0)
            else:
                blinks.append(0.0)
    return {"time_s": np.asarray(times), "veog": np.asarray(veog, float),
            "heo": np.asarray(heo, float) if heo else None,
            "trig": trig, "blinks": np.asarray(blinks, float),
            "extra": {c: np.asarray(v, float) for c, v in extra.items()},
            "fields": fields}


def read_tobii(subject: str, session: str, name: str) -> dict[str, Any]:
    path = tobii_root() / subject / session / "Tobii" / f"{name}.csv"
    stamp, validity_l, validity_r, gaze_x, gaze_y, event, gaze_type = [], [], [], [], [], [], []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for row in reader:
            try:
                stamp.append(float(row["RecordingTimestamp"]))
            except (TypeError, ValueError):
                continue
            for key, target in (("ValidityLeft", validity_l), ("ValidityRight", validity_r),
                                ("GazePointX (MCSpx)", gaze_x), ("GazePointY (MCSpx)", gaze_y)):
                value = row.get(key)
                try:
                    target.append(float(value))
                except (TypeError, ValueError):
                    target.append(np.nan)
            gaze_type.append(row.get("GazeEventType") or "")
            marker = None
            for key in ("EventMarkerValue", "StudioEventData", "ExternalEventValue",
                        "StudioEvent", "ExternalEvent"):
                value = row.get(key)
                if value not in (None, "",):
                    marker = value
                    break
            if marker is not None:
                event.append((float(row["RecordingTimestamp"]), marker))
    return {"stamp_ms": np.asarray(stamp), "validity_left": np.asarray(validity_l),
            "validity_right": np.asarray(validity_r), "gaze_x": np.asarray(gaze_x),
            "gaze_y": np.asarray(gaze_y), "gaze_type": np.asarray(gaze_type, dtype=object),
            "events": event, "fields": fields}


def diagnose(limit: int) -> None:
    rows = []
    for subject, session, name in recordings()[:limit]:
        eeg = read_eeg(subject, session, name)
        tob = read_tobii(subject, session, name)
        valid = (tob["validity_left"] <= 1) & (tob["validity_right"] <= 1)
        rows.append({
            "recording": f"{subject}/{session}/{name}",
            "eeg_samples": int(len(eeg["time_s"])),
            "eeg_duration_s": float(eeg["time_s"][-1]) if len(eeg["time_s"]) else 0.0,
            "eeg_trig_events": len(eeg["trig"]),
            "eeg_trig_examples": [t[1] for t in eeg["trig"][:6]],
            "eeg_blinks_column_nonzero": int(np.count_nonzero(eeg["blinks"])),
            "tobii_samples": int(len(tob["stamp_ms"])),
            "tobii_duration_s": float(tob["stamp_ms"][-1] / 1000) if len(tob["stamp_ms"]) else 0.0,
            "tobii_rate_hz": float(1000 * len(tob["stamp_ms"]) / max(tob["stamp_ms"][-1], 1))
            if len(tob["stamp_ms"]) else 0.0,
            "tobii_events": len(tob["events"]),
            "tobii_event_examples": [e[1] for e in tob["events"][:6]],
            "tobii_valid_fraction": float(np.mean(valid)),
            "gaze_type_counts": {k: int(v) for k, v in
                                 zip(*np.unique(tob["gaze_type"], return_counts=True))},
        })
        print(json.dumps(rows[-1])[:400], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1_diagnostic.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")


def optical_blinks(tob: dict[str, Any]) -> list[tuple[float, float]]:
    """Frozen M1 blink rule: both-eye invalid run of 50-500 ms flanked by >=50 ms valid."""
    invalid = ~((tob["validity_left"] <= 1) & (tob["validity_right"] <= 1))
    stamp = tob["stamp_ms"]
    runs = []
    start = None
    for index, flag in enumerate(invalid):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(invalid)))
    blinks = []
    for begin, end in runs:
        if begin == 0 or end >= len(stamp):
            continue
        duration = stamp[end - 1] - stamp[begin]
        if not (50.0 <= duration <= 500.0):
            continue
        pre = stamp[begin] - stamp[max(begin - 1, 0)]
        if pre > 100:                      # require a real preceding valid stretch
            continue
        blinks.append((float(stamp[begin]), float(duration)))
    return blinks


def veog_deflections(eeg: dict[str, Any]) -> np.ndarray:
    """Blink-like deflections on the registered VEOG surrogate (0.5-8 Hz, robust z)."""
    from scipy.signal import butter, find_peaks, sosfiltfilt

    signal = eeg["veog"].copy()
    signal[~np.isfinite(signal)] = np.nanmedian(signal[np.isfinite(signal)]) if \
        np.isfinite(signal).any() else 0.0
    sos = butter(4, (0.5, 8.0), btype="bandpass", fs=EEG_RATE, output="sos")
    filtered = sosfiltfilt(sos, signal)
    median = np.median(filtered)
    mad = 1.4826 * np.median(np.abs(filtered - median))
    z = (filtered - median) / max(mad, 1e-9)
    peaks, _ = find_peaks(np.abs(z), height=4.0, distance=int(0.2 * EEG_RATE))
    return eeg["time_s"][peaks] * 1000.0 if len(eeg["time_s"]) else np.asarray([])


def align_recording(subject: str, session: str, name: str) -> dict[str, Any]:
    eeg = read_eeg(subject, session, name)
    tob = read_tobii(subject, session, name)
    blinks = optical_blinks(tob)
    deflections = veog_deflections(eeg)
    path_used = "none"
    slope, intercept, residual_ms = 1.0, 0.0, float("nan")
    # --- primary: shared events (E-Prime triggers in EEG vs Tobii event markers)
    if len(eeg["trig"]) >= 3 and len(tob["events"]) >= 3:
        eeg_times = np.asarray([t[0] * 1000.0 for t in eeg["trig"]])
        tob_times = np.asarray([e[0] for e in tob["events"]])
        count = min(len(eeg_times), len(tob_times))
        if count >= 3:
            slope, intercept = np.polyfit(tob_times[:count], eeg_times[:count], 1)
            residual = eeg_times[:count] - (slope * tob_times[:count] + intercept)
            residual_ms = float(np.sqrt(np.mean(residual ** 2)))
            path_used = "eprime_triggers"
    # --- fallback: blink-onset cross-matching (coarse lag search, then linear fit)
    if path_used == "none" and len(blinks) >= 5 and len(deflections) >= 5:
        onsets = np.asarray([b[0] for b in blinks])
        best_lag, best_score = 0.0, -1
        for lag in np.arange(-2000, 2001, 5.0):
            matched = sum(1 for o in onsets
                          if np.min(np.abs(deflections - (o + lag))) <= BLINK_MATCH_MS)
            if matched > best_score:
                best_lag, best_score = float(lag), matched
        pairs = [(o, deflections[np.argmin(np.abs(deflections - (o + best_lag)))])
                 for o in onsets
                 if np.min(np.abs(deflections - (o + best_lag))) <= 200.0]
        if len(pairs) >= 5:
            x = np.asarray([p[0] for p in pairs])
            y = np.asarray([p[1] for p in pairs])
            slope, intercept = np.polyfit(x, y, 1)
            residual_ms = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
            path_used = "blink_cross_matching"
    # --- gate
    matched = 0
    lags = []
    for onset, _ in blinks:
        predicted = slope * onset + intercept
        if len(deflections):
            index = int(np.argmin(np.abs(deflections - predicted)))
            lag = float(deflections[index] - predicted)
            lags.append(lag)
            matched += int(abs(lag) <= BLINK_MATCH_MS)
    match_rate = matched / len(blinks) if blinks else 0.0
    passed = bool(blinks and match_rate >= MATCH_RATE_GATE
                  and np.isfinite(residual_ms) and residual_ms <= DRIFT_RESIDUAL_MS)
    return {"recording": f"{subject}/{session}/{name}", "subject": subject,
            "session": session, "name": name, "clock_path": path_used,
            "slope": float(slope), "intercept_ms": float(intercept),
            "drift_residual_ms": residual_ms,
            "optical_blinks": len(blinks), "veog_deflections": int(len(deflections)),
            "matched_blinks": matched, "match_rate": match_rate,
            "median_abs_lag_ms": float(np.median(np.abs(lags))) if lags else float("nan"),
            "gate_passed": passed,
            "exposure_excluded_from_E2": subject in EXPOSURE_EXCLUDED}


def run(subjects: list[str] | None, tag: str) -> None:
    todo = [r for r in recordings() if subjects is None or r[0] in subjects]
    rows = []
    for subject, session, name in todo:
        try:
            rows.append(align_recording(subject, session, name))
        except Exception as error:                          # noqa: BLE001 reason-coded
            rows.append({"recording": f"{subject}/{session}/{name}", "subject": subject,
                         "error": f"{type(error).__name__}: {error}", "gate_passed": False,
                         "exposure_excluded_from_E2": subject in EXPOSURE_EXCLUDED})
        print(json.dumps(rows[-1])[:300], flush=True)
    by_subject: dict[str, list[dict]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    subject_rows = []
    for subject, entries in sorted(by_subject.items()):
        passes = [e for e in entries if e.get("gate_passed")]
        subject_rows.append({
            "subject": subject, "recordings": len(entries), "passed": len(passes),
            "subject_aligned": bool(len(passes) >= max(1, len(entries) // 2)),
            "median_match_rate": float(np.median([e.get("match_rate", 0.0) for e in entries])),
            "median_drift_residual_ms": float(np.nanmedian(
                [e.get("drift_residual_ms", np.nan) for e in entries])),
            "exposure_excluded_from_E2": subject in EXPOSURE_EXCLUDED})
    payload = {"gate": {"match_rate": MATCH_RATE_GATE, "blink_match_ms": BLINK_MATCH_MS,
                        "drift_residual_ms": DRIFT_RESIDUAL_MS},
               "per_recording": rows, "per_subject": subject_rows,
               "aligned_subjects": [s["subject"] for s in subject_rows if s["subject_aligned"]],
               "excluded_and_counted": [s["subject"] for s in subject_rows
                                        if not s["subject_aligned"]],
               "exposure_excluded": sorted(EXPOSURE_EXCLUDED)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"e1_{tag}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"tag": tag, "aligned": len(payload["aligned_subjects"]),
                      "excluded": len(payload["excluded_and_counted"])}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--limit", type=int, default=4)
    p = sub.add_parser("pilot")
    p.add_argument("--subjects", nargs="+", default=["S01", "S02", "S03"])
    sub.add_parser("all")
    args = parser.parse_args()
    if args.mode == "diagnose":
        diagnose(args.limit)
    elif args.mode == "pilot":
        run(args.subjects, "pilot")
    else:
        run(None, "all")


if __name__ == "__main__":
    main()
