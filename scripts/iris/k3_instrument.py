#!/usr/bin/env python3
"""IRIS K3 — EEGEyeNet true-VEOG instrument validity (dev subjects only).

Preregistered in reports/iris_prereg_k.md (frozen before execution). Ports the E1R
correspondence instrument to the panel with a real vertical periocular axis and
official <=2 ms EyeLink synchronization. Gates: G-K3a forward blink median >= 0.70,
G-K3b reverse elevation CI-low > 0 (both on antisaccade, the sealed-fight panel;
dots reported alongside), G-K3c saccade median >= 0.70 (gates the saccade-typed EOG
drive only). Sealed tree is never touched.

Frozen derivations: VEOG_L = E25−E127, VEOG_R = E8−E126, VEOG = mean; HEOG = right
minus left outer canthus among {E125, E128}, sides resolved by each file's own
chanlocs Y coordinates (geometry-verified). Filters: VEOG 0.5–8 Hz, HEOG 0.5–20 Hz.
Blink peak: local max > 3×MAD, half-max width >= 50 ms. Windows: forward [start−100,
end+100] ms; reverse dilation ±100 ms; saccade ±50 ms. Null: 200 circular shifts;
bootstrap 5000 draws; seed 20260817.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_min")
OUT = REPO / "results/iris/k/k3_instrument.json"
PERIOCULAR = {"upper": ("E25", "E8"), "lower": ("E127", "E126"),
              "canthi": ("E125", "E128")}
ALL_PERI = ("E8", "E25", "E125", "E126", "E127", "E128")
SEED = 20260817
N_SHIFTS = 200
N_BOOT = 5000
FORWARD_PAD_S = 0.100
SACCADE_PAD_S = 0.050
MIN_BLINK_WIDTH_S = 0.050
MIN_PEAK_SEP_S = 0.200
MIN_SHIFT_S = 5.0
SACCADE_MIN_DEG = 2.0
GATE_FORWARD = 0.70
GATE_SACCADE = 0.70


# ---------------------------------------------------------------- loaders

def _h5_text(handle, node) -> str:
    import h5py
    if isinstance(node, h5py.Reference):
        node = handle[node]
    values = np.asarray(node[()]).ravel()
    if values.size == 0 or values.dtype.kind not in "uif":
        return ""
    return "".join(chr(int(c)) for c in values if int(c) > 0)


def _h5_num(handle, node) -> float:
    import h5py
    if isinstance(node, h5py.Reference):
        node = handle[node]
    values = np.asarray(node[()]).ravel()
    if values.size == 0:
        return float("nan")
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return float("nan")


def load_antisaccade(path: Path) -> dict:
    import h5py
    with h5py.File(path, "r") as handle:
        eeg = handle["EEG"]
        srate = float(eeg["srate"][0, 0])
        labels = [_h5_text(handle, r)
                  for r in np.asarray(eeg["chanlocs"]["labels"]).ravel()]
        coords = {}
        for axis in ("X", "Y", "Z"):
            if axis in eeg["chanlocs"]:
                coords[axis] = np.asarray(
                    [_h5_num(handle, r)
                     for r in np.asarray(eeg["chanlocs"][axis]).ravel()])
        idx = {lab: i for i, lab in enumerate(labels)}
        want = sorted(idx[lab] for lab in ALL_PERI)
        data = eeg["data"]                      # (samples, channels)
        cols = data[:, want]                    # h5py needs increasing indices
        chans = {lab: cols[:, want.index(idx[lab])].astype(float)
                 for lab in ALL_PERI}
        event = eeg["event"]
        fields = list(event.keys())
        events = {}
        for field in ("type", "latency", "duration", "endtime", "sac_amplitude",
                      "sac_startpos_x", "sac_startpos_y", "sac_endpos_x",
                      "sac_endpos_y"):
            if field not in fields:
                continue
            refs = np.asarray(event[field]).ravel()
            if field == "type":
                events[field] = [_h5_text(handle, r) for r in refs]
            else:
                events[field] = np.asarray([_h5_num(handle, r) for r in refs])
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
        return {"srate": srate, "labels": labels, "coords": coords, "chans": chans,
                "events": events, "bad_chans": sorted(bad), "gaze": None,
                "n_samples": int(data.shape[0])}


def _sc_text(value) -> str:
    while isinstance(value, np.ndarray) and value.size >= 1:
        value = value.ravel()[0]
    return "" if isinstance(value, np.ndarray) else str(value)


def _sc_num(value) -> float:
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return float("nan")
    try:
        return float(arr[0])
    except (TypeError, ValueError):
        return float("nan")


def load_dots(path: Path) -> dict:
    from scipy.io import loadmat
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    eeg = mat["sEEG"]
    while isinstance(eeg, np.ndarray):
        eeg = eeg.ravel()[0]
    srate = float(np.asarray(eeg.srate).ravel()[0])
    data = np.asarray(eeg.data, float)          # (channels, samples)
    chanlocs = np.asarray(eeg.chanlocs).ravel()
    labels = [_sc_text(c.labels) for c in chanlocs]
    coords = {}
    for axis in ("X", "Y", "Z"):
        vals = [_sc_num(getattr(c, axis, np.nan)) for c in chanlocs]
        coords[axis] = np.asarray(vals)
    idx = {lab: i for i, lab in enumerate(labels)}
    chans = {lab: data[idx[lab]] for lab in ALL_PERI}
    gaze = {lab: data[idx[lab]] for lab in ("L-GAZE-X", "L-GAZE-Y", "L-AREA")
            if lab in idx}
    raw_events = np.asarray(eeg.event).ravel()
    events: dict = {"type": [_sc_text(e.type) for e in raw_events]}
    for field in ("latency", "duration", "endtime", "sac_amplitude",
                  "sac_startpos_x", "sac_startpos_y", "sac_endpos_x", "sac_endpos_y"):
        events[field] = np.asarray(
            [_sc_num(getattr(e, field, np.nan)) for e in raw_events])
    return {"srate": srate, "labels": labels, "coords": coords, "chans": chans,
            "events": events, "bad_chans": [], "gaze": gaze,
            "n_samples": int(data.shape[1])}


# ---------------------------------------------------------------- geometry

def verify_geometry(rec: dict) -> dict:
    coords, labels = rec["coords"], rec["labels"]
    if not coords or not np.isfinite(coords.get("X", np.asarray([np.nan]))).any():
        return {"verified": False, "reason": "no coordinates in file; "
                "standard GSN-HydroCel-129 montage assumed (automagic sfp)"}
    idx = {lab: i for i, lab in enumerate(labels)}
    x, y, z = coords["X"], coords["Y"], coords["Z"]
    peri = [idx[lab] for lab in ALL_PERI]
    anterior_rank = float(np.mean([np.mean(x > x[i]) for i in peri]))
    upper_ok = all(z[idx[u]] > z[idx[lo]] for u, lo in
                   zip(PERIOCULAR["upper"], PERIOCULAR["lower"]))
    canthi = list(PERIOCULAR["canthi"])
    ys = {lab: y[idx[lab]] for lab in canthi}
    lateral_ok = (np.sign(ys[canthi[0]]) != np.sign(ys[canthi[1]])
                  and all(abs(ys[lab]) >= max(abs(y[idx[e]]) for e in ALL_PERI[:2])
                          for lab in canthi))
    # EEGLAB convention: +Y = left ear -> the RIGHT canthus has negative Y
    right = min(canthi, key=lambda lab: ys[lab])
    left = max(canthi, key=lambda lab: ys[lab])
    return {"verified": bool(upper_ok and lateral_ok and anterior_rank < 0.25),
            "anterior_rank": anterior_rank, "upper_above_lower": upper_ok,
            "canthi_lateral": bool(lateral_ok), "heog_right": right,
            "heog_left": left}


# ---------------------------------------------------------------- detection

def _bandpass(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def blink_peaks(veog: np.ndarray, fs: float) -> np.ndarray:
    mad = float(np.median(np.abs(veog - np.median(veog))))
    threshold = 3.0 * mad
    peaks, _ = signal.find_peaks(veog, height=threshold,
                                 distance=int(MIN_PEAK_SEP_S * fs))
    keep = []
    half = np.asarray(veog) >= (threshold / 2.0)
    for p in peaks:
        lo = p
        while lo > 0 and half[lo - 1]:
            lo -= 1
        hi = p
        while hi < len(half) - 1 and half[hi + 1]:
            hi += 1
        if (hi - lo) / fs >= MIN_BLINK_WIDTH_S:
            keep.append(p)
    return np.asarray(keep, int)


def _intervals(events: dict, kinds: tuple[str, ...], fs: float,
               n_samples: int) -> list[tuple[int, int]]:
    out = []
    types = events["type"]
    latency = events.get("latency")
    duration = events.get("duration", np.full(len(types), np.nan))
    endtime = events.get("endtime", np.full(len(types), np.nan))
    # units sanity: latencies are in samples per the data description; a duration
    # column whose median is < 10 would indicate seconds — convert if so
    dur_med = float(np.nanmedian(duration)) if np.isfinite(duration).any() else np.nan
    dur_scale = fs if (np.isfinite(dur_med) and dur_med < 10) else 1.0
    for i, kind in enumerate(types):
        if not any(kind.startswith(k) for k in kinds):
            continue
        start = latency[i]
        if not np.isfinite(start):
            continue
        end = endtime[i] if np.isfinite(endtime[i]) else start + (
            duration[i] * dur_scale if np.isfinite(duration[i]) else 0.0)
        out.append((max(int(start), 0), min(int(end), n_samples - 1)))
    return out


def _inside(points: np.ndarray, intervals: list[tuple[int, int]], pad: int,
            n_samples: int) -> np.ndarray:
    mask = np.zeros(n_samples, bool)
    for start, end in intervals:
        mask[max(start - pad, 0):min(end + pad + 1, n_samples)] = True
    return mask[points]


def process(rec: dict, name: str, group: str) -> dict:
    fs = rec["srate"]
    n = rec["n_samples"]
    geometry = verify_geometry(rec)
    excluded_bad = sorted(set(rec["bad_chans"])
                          & {int(lab[1:]) for lab in ALL_PERI})
    row = {"recording": name, "group": group, "srate": fs,
           "geometry": geometry, "periocular_bad_chans": excluded_bad,
           "excluded": bool(excluded_bad)}
    if excluded_bad:
        return row
    veog = _bandpass((rec["chans"]["E25"] - rec["chans"]["E127"]
                      + rec["chans"]["E8"] - rec["chans"]["E126"]) / 2.0,
                     0.5, 8.0, fs)
    right = geometry.get("heog_right", "E125")
    left = geometry.get("heog_left", "E128")
    heog = _bandpass(rec["chans"][right] - rec["chans"][left], 0.5, 20.0, fs)

    peaks = blink_peaks(veog, fs)
    blinks = _intervals(rec["events"], ("L_blink", "R_blink"), fs, n)
    row["n_eyelink_blinks"] = len(blinks)
    row["n_veog_peaks"] = int(len(peaks))
    pad = int(FORWARD_PAD_S * fs)

    if blinks:
        hits = sum(1 for start, end in blinks
                   if np.any((peaks >= start - pad) & (peaks <= end + pad)))
        row["forward_match"] = hits / len(blinks)

    gap_intervals = list(blinks)
    if rec["gaze"] and "L-AREA" in rec["gaze"]:
        area = rec["gaze"]["L-AREA"]
        lost = (~np.isfinite(area)) | (area <= 0)
        edges = np.flatnonzero(np.diff(lost.astype(int)))
        bounds = np.concatenate([[0], edges + 1, [n]])
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if lost[lo] and (hi - lo) / fs >= MIN_BLINK_WIDTH_S:
                gap_intervals.append((int(lo), int(hi - 1)))
    if len(peaks) and gap_intervals:
        observed = float(_inside(peaks, gap_intervals, pad, n).mean())
        rng = np.random.default_rng(SEED)
        null_rates = []
        for _ in range(N_SHIFTS):
            shift = int(rng.uniform(MIN_SHIFT_S * fs, n - MIN_SHIFT_S * fs))
            null_rates.append(float(_inside((peaks + shift) % n, gap_intervals,
                                            pad, n).mean()))
        row["reverse_rate"] = observed
        row["reverse_null_mean"] = float(np.mean(null_rates))
        row["reverse_elevation"] = observed - float(np.mean(null_rates))

    events = rec["events"]
    types = events["type"]
    onsets, signs = [], []
    for i, kind in enumerate(types):
        if "_saccade" not in kind:
            continue
        dx = events["sac_endpos_x"][i] - events["sac_startpos_x"][i]
        dy = events["sac_endpos_y"][i] - events["sac_startpos_y"][i]
        norm = float(np.hypot(dx, dy))
        amp = events["sac_amplitude"][i]
        if not (np.isfinite(amp) and norm > 0):
            continue
        if amp * abs(dx) / norm >= SACCADE_MIN_DEG:
            onsets.append(int(events["latency"][i]))
            signs.append(np.sign(dx))
    row["n_horizontal_saccades"] = len(onsets)
    if onsets:
        d = np.abs(np.diff(heog))
        threshold = 3.0 * float(np.median(np.abs(d - np.median(d))))
        spad = int(SACCADE_PAD_S * fs)
        hits, sign_agreement = 0, []
        dh = np.diff(heog)
        for onset, want in zip(onsets, signs):
            lo, hi = max(onset - spad, 0), min(onset + spad, n - 2)
            window = d[lo:hi]
            if window.size and window.max() > threshold:
                hits += 1
                sign_agreement.append(
                    float(np.sign(dh[lo + int(np.argmax(window))]) == want))
        row["saccade_match"] = hits / len(onsets)
        row["saccade_sign_consistency"] = (
            float(np.mean(sign_agreement)) if sign_agreement else float("nan"))
    return row


# ---------------------------------------------------------------- gates

def _boot_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean()
                        for _ in range(N_BOOT)])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def gates(rows: list[dict]) -> dict:
    usable = [r for r in rows if not r.get("excluded")]
    forward = np.asarray([r["forward_match"] for r in usable
                          if "forward_match" in r])
    elevation = np.asarray([r["reverse_elevation"] for r in usable
                            if "reverse_elevation" in r])
    saccade = np.asarray([r["saccade_match"] for r in usable
                          if "saccade_match" in r])
    out = {"n_recordings": len(rows), "n_excluded_bad_periocular":
           sum(1 for r in rows if r.get("excluded")), "n_usable": len(usable)}
    if forward.size:
        out["G_K3a_forward"] = {"median": float(np.median(forward)),
                                "mean": float(forward.mean()),
                                "n": int(forward.size), "bar": GATE_FORWARD,
                                "pass": bool(np.median(forward) >= GATE_FORWARD)}
    if elevation.size:
        lo, hi = _boot_ci(elevation)
        out["G_K3b_reverse"] = {"pooled_elevation": float(elevation.mean()),
                                "ci_low": lo, "ci_high": hi,
                                "n": int(elevation.size), "pass": bool(lo > 0)}
    if saccade.size:
        out["G_K3c_saccade"] = {"median": float(np.median(saccade)),
                                "mean": float(saccade.mean()),
                                "n": int(saccade.size), "bar": GATE_SACCADE,
                                "pass": bool(np.median(saccade) >= GATE_SACCADE)}
    return out


def main() -> None:
    rows = []
    for subject_dir in sorted((ROOT / "antisaccade_min").iterdir()):
        for path in sorted(subject_dir.glob("*.mat")):
            rows.append(process(load_antisaccade(path),
                                f"{subject_dir.name}/{path.name}", "antisaccade"))
            print(json.dumps({k: rows[-1].get(k) for k in
                              ("recording", "forward_match", "reverse_elevation",
                               "saccade_match", "excluded")}), flush=True)
    for subject_dir in sorted((ROOT / "dots_min").iterdir()):
        for path in sorted(subject_dir.glob("*.mat")):
            rows.append(process(load_dots(path),
                                f"{subject_dir.name}/{path.name}", "dots"))
            print(json.dumps({k: rows[-1].get(k) for k in
                              ("recording", "forward_match", "reverse_elevation",
                               "saccade_match", "excluded")}), flush=True)

    anti = gates([r for r in rows if r["group"] == "antisaccade"])
    dots = gates([r for r in rows if r["group"] == "dots"])
    valid = bool(anti.get("G_K3a_forward", {}).get("pass")
                 and anti.get("G_K3b_reverse", {}).get("pass"))
    payload = {
        "prereg": "reports/iris_prereg_k.md (K3)",
        "verdict": {"instrument_valid": valid,
                    "basis": "G-K3a AND G-K3b on antisaccade (sealed-fight panel)",
                    "saccade_typed_eog_drive":
                        bool(anti.get("G_K3c_saccade", {}).get("pass"))},
        "antisaccade": anti, "dots": dots,
        "constants": {"forward_pad_s": FORWARD_PAD_S, "saccade_pad_s": SACCADE_PAD_S,
                      "min_blink_width_s": MIN_BLINK_WIDTH_S,
                      "min_peak_sep_s": MIN_PEAK_SEP_S, "n_shifts": N_SHIFTS,
                      "n_boot": N_BOOT, "seed": SEED,
                      "saccade_min_deg": SACCADE_MIN_DEG,
                      "mad_multiplier": 3.0, "mad_scaling": "unscaled (prereg literal)"},
        "per_recording": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"valid": valid,
                      "anti_forward_median":
                          anti.get("G_K3a_forward", {}).get("median"),
                      "anti_reverse_ci_low":
                          anti.get("G_K3b_reverse", {}).get("ci_low"),
                      "anti_saccade_median":
                          anti.get("G_K3c_saccade", {}).get("median")}))


if __name__ == "__main__":
    main()
