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
VALIDITY_CUT = 1          # declared (prereg froze "both-eye invalid", not the code cut)


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


def _runs(times_ms: np.ndarray, labels: np.ndarray, gap_ms: float = 100.0):
    """Collapse a sparse, run-held marker channel into (onset_ms, label) events."""
    events = []
    if len(times_ms) == 0:
        return events
    start = 0
    for index in range(1, len(times_ms) + 1):
        broken = (index == len(times_ms)
                  or times_ms[index] - times_ms[index - 1] > gap_ms
                  or labels[index] != labels[start])
        if broken:
            events.append((float(times_ms[start]), str(labels[start])))
            start = index
    return events


def read_eeg(subject: str, session: str, name: str, channels=None) -> dict[str, Any]:
    """Read only the columns E1/E2 need (pandas usecols; the CSVs are ~370 MB)."""
    import pandas as pd

    path = EEG_ROOT / subject / session / "Neuroscan" / f"{name}.csv"
    header = pd.read_csv(path, nrows=0)
    fields = list(header.columns)
    wanted = ["Time"] + [c for c in VEOG_CHANNELS if c in fields]
    for column in ("HEO", "Trig", "Cues", "Blinks", "RecordingTimestamp"):
        if column in fields:
            wanted.append(column)
    for column in (channels or ()):
        if column in fields and column not in wanted:
            wanted.append(column)
    frame = pd.read_csv(path, usecols=wanted, low_memory=False,
                        dtype={c: str for c in ("Trig", "Cues", "RecordingTimestamp")
                               if c in fields})
    time_s = pd.to_numeric(frame["Time"], errors="coerce").to_numpy(float)
    veog_cols = [c for c in VEOG_CHANNELS if c in frame.columns]
    veog = frame[veog_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float).mean(axis=1) \
        if veog_cols else np.full(len(frame), np.nan)
    events = []
    for column in ("Trig", "Cues"):
        if column not in frame.columns:
            continue
        values = frame[column].fillna("NA").to_numpy(dtype=object)
        mask = ~np.isin(values, ["NA", "", "nan", None])
        if mask.any():
            events.extend([(t, f"{column}:{v}")
                           for t, v in _runs(time_s[mask] * 1000.0, values[mask])])
    events.sort()
    extra = {}
    for column in (channels or ()):
        if column in frame.columns:
            extra[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    blinks = pd.to_numeric(frame["Blinks"], errors="coerce").fillna(0).to_numpy(float) \
        if "Blinks" in frame.columns else np.zeros(len(frame))
    # Primary clock source (prereg): the EEG export carries the TOBII recording-clock
    # value on the handful of rows that bear a trial marker; everything else is "NA".
    # Each populated row is therefore an exact (tobii_ms, eeg_ms) correspondence.
    clock_pairs = np.zeros((0, 2))
    if "RecordingTimestamp" in frame.columns:
        stamps = pd.to_numeric(frame["RecordingTimestamp"], errors="coerce").to_numpy(float)
        good = np.isfinite(stamps) & np.isfinite(time_s)
        if good.any():
            clock_pairs = np.stack([stamps[good], time_s[good] * 1000.0], axis=1)
    return {"time_s": time_s, "veog": veog, "clock_pairs": clock_pairs,
            "heo": pd.to_numeric(frame["HEO"], errors="coerce").to_numpy(float)
            if "HEO" in frame.columns else None,
            "events": events, "blinks": blinks, "extra": extra, "fields": fields}


def read_tobii(subject: str, session: str, name: str) -> dict[str, Any]:
    """Tobii export: 300 Hz, RecordingTimestamp in ms. Sync events are the 42
    EventMarkerOn/Off pulses in StudioEvent (EventMarkerValue is a held 0/1 channel)."""
    import pandas as pd

    path = tobii_root() / subject / session / "Tobii" / f"{name}.csv"
    columns = ["RecordingTimestamp", "ValidityLeft", "ValidityRight", "GazePointX (MCSpx)",
               "GazePointY (MCSpx)", "GazeEventType", "GazeEventDuration", "PupilLeft",
               "PupilRight", "StudioEvent", "EventMarkerValue",
               # E2 geometry: visual angle needs gaze in mm plus eye-to-tracker distance
               "StrictAverageGazePointX (ADCSmm)", "StrictAverageGazePointY (ADCSmm)",
               "DistanceLeft", "DistanceRight"]
    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in columns if c in header.columns]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False,
                        dtype={c: str for c in ("GazeEventType", "StudioEvent") if c in usecols})
    numeric = lambda c: (pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
                         if c in frame.columns else np.full(len(frame), np.nan))
    stamp = numeric("RecordingTimestamp")
    events = []
    if "StudioEvent" in frame.columns:
        marks = frame["StudioEvent"].fillna("").to_numpy(dtype=object)
        for index in np.flatnonzero(marks == "EventMarkerOn"):
            events.append((float(stamp[index]), "EventMarkerOn"))
    if not events and "EventMarkerValue" in frame.columns:
        marker = numeric("EventMarkerValue")
        rising = np.flatnonzero((marker[1:] > 0.5) & (marker[:-1] <= 0.5)) + 1
        events = [(float(stamp[i]), "MarkerRise") for i in rising]
    return {"stamp_ms": stamp, "validity_left": numeric("ValidityLeft"),
            "validity_right": numeric("ValidityRight"),
            "gaze_x": numeric("GazePointX (MCSpx)"), "gaze_y": numeric("GazePointY (MCSpx)"),
            "pupil_left": numeric("PupilLeft"), "pupil_right": numeric("PupilRight"),
            "gaze_type": (frame["GazeEventType"].fillna("").to_numpy(dtype=object)
                          if "GazeEventType" in frame.columns
                          else np.full(len(frame), "", dtype=object)),
            "gaze_event_duration": numeric("GazeEventDuration"),
            "gaze_mm_x": numeric("StrictAverageGazePointX (ADCSmm)"),
            "gaze_mm_y": numeric("StrictAverageGazePointY (ADCSmm)"),
            "distance_left": numeric("DistanceLeft"),
            "distance_right": numeric("DistanceRight"),
            "events": events, "fields": list(frame.columns)}


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
            "eeg_events": len(eeg["events"]),
            "eeg_event_examples": [t[1] for t in eeg["events"][:6]],
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


def optical_blinks(tob: dict[str, Any], validity_cut: int = VALIDITY_CUT
                   ) -> list[tuple[float, float]]:
    """Frozen M1 blink rule, implemented literally: a both-eye-invalid contiguous run of
    50-500 ms FLANKED BY >=50 ms of valid samples on BOTH sides.

    `validity_cut` is not fixed by the preregistration (it froze "both-eye validity
    invalid" without naming the Tobii code cutoff); VALIDITY_CUT is the declared choice
    and run() reports the full 0..3 sensitivity alongside it.
    """
    valid = (tob["validity_left"] <= validity_cut) & (tob["validity_right"] <= validity_cut)
    invalid = ~valid
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
        # walk back/forward over the contiguous VALID stretches that flank this run
        pre_start = begin - 1
        while pre_start > 0 and valid[pre_start - 1]:
            pre_start -= 1
        post_end = end
        while post_end < len(stamp) - 1 and valid[post_end + 1]:
            post_end += 1
        if stamp[begin - 1] - stamp[pre_start] < 50.0:
            continue
        if stamp[post_end] - stamp[end] < 50.0:
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


def _lag_votes(eeg_times: np.ndarray, tob_times: np.ndarray, bin_ms: float = 4.0
               ) -> tuple[float, int]:
    """Modal (eeg - tobii) offset by pairwise-difference voting.

    Unlike a hit-count lag scan, this requires MANY event pairs to agree on ONE offset,
    so a dense EEG marker train cannot manufacture a match: chance votes stay at the
    background density while a true correspondence concentrates in a single bin.
    """
    if len(eeg_times) == 0 or len(tob_times) == 0:
        return 0.0, 0
    diffs = (eeg_times[:, None] - tob_times[None, :]).ravel()
    bins = np.round(diffs / bin_ms).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    top = int(np.argmax(counts))
    return float(values[top] * bin_ms), int(counts[top])


def _fit_at_lag(eeg_times: np.ndarray, tob_times: np.ndarray, lag: float,
                tolerance: float = 100.0):
    """Pair each Tobii pulse with its nearest EEG event at the modal lag, then fit."""
    pairs = []
    for t in tob_times:
        predicted = t + lag
        index = int(np.clip(np.searchsorted(eeg_times, predicted), 0, len(eeg_times) - 1))
        for probe in {max(index - 1, 0), index, min(index + 1, len(eeg_times) - 1)}:
            if abs(eeg_times[probe] - predicted) <= tolerance:
                pairs.append((float(t), float(eeg_times[probe])))
                break
    if len(pairs) < 3:
        return None
    x = np.asarray([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
    return float(slope), float(intercept), residual, len(pairs)


def align_recording(subject: str, session: str, name: str) -> dict[str, Any]:
    eeg = read_eeg(subject, session, name)
    tob = read_tobii(subject, session, name)
    blinks = optical_blinks(tob)
    deflections = veog_deflections(eeg)
    path_used = "none"
    slope, intercept, residual_ms = 1.0, 0.0, float("nan")
    event_counts = (len(eeg["events"]), len(tob["events"]))
    clock_n = int(len(eeg["clock_pairs"]))
    null_votes = real_votes = 0
    collinear_frac = float("nan")
    # --- PRIMARY (prereg): the shared RecordingTimestamp. The EEG export writes the
    # Tobii recording-clock value onto each trial-marker row, giving exact pairs.
    if clock_n >= 3:
        x = eeg["clock_pairs"][:, 0]
        y = eeg["clock_pairs"][:, 1]
        slope, intercept = np.polyfit(x, y, 1)
        residual_ms = float(np.sqrt(np.mean((y - (slope * x + intercept)) ** 2)))
        path_used = f"shared_recording_timestamp n={clock_n}"
        # diagnostic only (never gates): largest subset consistent with one line, so a
        # failure from a stream discontinuity is distinguishable from one from noise.
        offsets = np.round((y - x) / 4.0).astype(np.int64)
        _, counts = np.unique(offsets, return_counts=True)
        collinear_frac = float(counts.max() / len(offsets))
    # --- SECONDARY (prereg): E-Prime-corroborated trigger matching, for the recordings
    # whose EEG export lacks the timestamp pair.
    if path_used == "none" and event_counts[0] >= 3 and event_counts[1] >= 3:
        tob_times = np.asarray([e[0] for e in tob["events"]])
        families: dict[str, list[float]] = {}
        for time_ms, label in eeg["events"]:
            families.setdefault(label, []).append(time_ms)
        best = None
        for label, times in families.items():
            candidate = np.asarray(sorted(times))
            if len(candidate) < 10:
                continue
            lag, votes = _lag_votes(candidate, tob_times)
            # null control: the same estimator on circularly shifted pulse trains
            span = max(tob_times[-1] - tob_times[0], 1.0)
            null = max(_lag_votes(candidate, tob_times[0] + np.sort(
                ((tob_times - tob_times[0]) + shift * span / 7.0) % span))[1]
                for shift in range(1, 7))
            if votes < 10 or votes <= null:
                continue
            fit = _fit_at_lag(candidate, tob_times, lag)
            if fit is None:
                continue
            fit_slope, fit_intercept, residual, n_pairs = fit
            if best is None or (votes, -residual) > best[0]:
                best = ((votes, -residual), label, fit_slope, fit_intercept,
                        residual, n_pairs, votes, null)
        if best is not None:
            _, label, slope, intercept, residual_ms, n_pairs, real_votes, null_votes = best
            path_used = f"eprime_triggers[{label}] n={n_pairs}"
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
    clock_ok = bool(np.isfinite(residual_ms) and residual_ms <= DRIFT_RESIDUAL_MS)
    passed = bool(blinks and match_rate >= MATCH_RATE_GATE and clock_ok)
    # sensitivity over the one parameter the preregistration left unfrozen
    sensitivity = {}
    for cut in (0, 1, 2, 3):
        alt = optical_blinks(tob, validity_cut=cut)
        hits = 0
        for onset, _ in alt:
            predicted = slope * onset + intercept
            if len(deflections):
                index = int(np.argmin(np.abs(deflections - predicted)))
                hits += int(abs(deflections[index] - predicted) <= BLINK_MATCH_MS)
        sensitivity[f"cut<={cut}"] = {"blinks": len(alt), "matched": hits,
                                      "match_rate": hits / len(alt) if alt else 0.0}
    return {"recording": f"{subject}/{session}/{name}", "subject": subject,
            "session": session, "name": name, "clock_path": path_used,
            "eeg_events": event_counts[0], "tobii_events": event_counts[1],
            "clock_pairs": clock_n, "clock_collinear_frac": collinear_frac,
            "secondary_votes": real_votes, "secondary_null_votes": null_votes,
            "slope": float(slope), "intercept_ms": float(intercept),
            "drift_residual_ms": residual_ms, "clock_gate_passed": clock_ok,
            "optical_blinks": len(blinks), "veog_deflections": int(len(deflections)),
            "matched_blinks": matched, "match_rate": match_rate,
            "median_abs_lag_ms": float(np.median(np.abs(lags))) if lags else float("nan"),
            "validity_cut_sensitivity": sensitivity,
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
            "clock_gate_passed": sum(int(bool(e.get("clock_gate_passed"))) for e in entries),
            "total_optical_blinks": sum(int(e.get("optical_blinks", 0)) for e in entries),
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
