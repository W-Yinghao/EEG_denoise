#!/usr/bin/env python3
"""E2 / E3 / E6 arrays from the natural-pass shards (CPU, job 2 of 2).

Reads the 90 per-cell shards written by export_natural_pass.py
(/projects/EEG-foundation-model/derived/denoiseNet/sg_natural_pass/<sub>|<ses>|<task>.npz)
and writes three files under results/paper_final/paper_final_arrays/:

E2  natural_residual_corr.npz
    corr       (condition, window, 46, 2)  |Pearson r| between each corrected channel
               and the raw bipolar VEOG / HEOG of the window, over all 512 samples
    corr_high  the same over the high-EOG samples H (sg_common.eog_masks, top 30 %)
    window -> recording / participant index arrays, coverage (condition, window),
    corr_recording / corr_high_recording (mean over the 4 windows of a recording)

E3  natural_psd_ratio.npz
    ratio_db   (condition, recording, electrode, freq) = mean over the 4 windows of
               10*log10(PSD_corrected / PSD_uncorrected) on the LOW-EOG samples L of
               each window (bottom 30 %), electrodes = sg_common.NINE + EXTRA_ROW.
    METHOD DEVIATION (also in the 'method_note' key): the doc's 2-s window / 1-s
    overlap Welch cannot be honoured on ~154 non-contiguous low samples per window;
    sg_common.welch is run on the concatenated low samples with
    nperseg = min(200, min n_low) (one Hann-tapered segment per window; noverlap =
    min(100, nperseg//2)), the recording mean over 4 windows and the participant mean
    over 6 recordings do the smoothing.  ratio_db_window keeps the per-window values.

E6  natural_band_power.npz
    ratio      (condition, recording, band, electrode) = mean over windows of
               band power after / before on the low-EOG samples (same PSD as E3,
               bins f in [lo, hi) summed x df), bands = sg_common.BANDS; ratio_db =
               mean over windows of 10*log10 of the per-window ratio; ratio_window
               keeps the per-window linear ratios.

Existing keys of a file are kept; missing arms are NaN with coverage False.

Usage:  export_natural_post.py            full export
        export_natural_post.py --smoke    synthetic shards -> scratch arrays (login-node check)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402
from sg_common import BANDS, EEG_NAMES, EXPORT_LABEL, EXTRA_ROW, IDX, NINE, WINDOW  # noqa: E402

SHARD_DIR = Path("/projects/EEG-foundation-model/derived/denoiseNet/sg_natural_pass")
CONDITIONS = ("RAW", "MATCH_gated", "POP", "NO_A0", "LINEAR", "WRONG_gated", "SHUFFLED",
              "ICA", "ASR", "SGEYESUB")
ELECTRODES = list(NINE) + list(EXTRA_ROW)
N_WINDOWS = 4
def label(condition: str) -> str:
    """sg_common.EXPORT_LABEL keyed by the D-wave arm name (MATCH_gated -> MATCH)."""
    return EXPORT_LABEL[condition] if condition in EXPORT_LABEL else EXPORT_LABEL[condition.replace("_gated", "")]


E2_NAME, E3_NAME, E6_NAME = "natural_residual_corr.npz", "natural_psd_ratio.npz", "natural_band_power.npz"


# ---------------------------------------------------------------- shards

def load_shards(shard_dir: Path):
    paths = sorted(p for p in shard_dir.glob("sub-*.npz") if not p.name.endswith(".tmp.npz"))
    assert paths, f"no shards under {shard_dir}"
    rec = {"participant": [], "session": [], "task": [], "wrong_donor": [], "lambda": [],
           "hard_gate": [], "fold": [], "ica_n_excluded": [], "asr_error": []}
    corrected, coverage, drive, eog_raw, starts, metrics = [], [], [], [], [], []
    eeg_scale = None
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            cond = tuple(str(c) for c in d["condition"])
            assert cond == CONDITIONS, (path.name, cond)
            assert [str(n) for n in d["eeg_names"]] == EEG_NAMES
            assert d["corrected"].shape == (len(CONDITIONS), N_WINDOWS, 46, WINDOW), path.name
            corrected.append(np.asarray(d["corrected"], np.float32))
            coverage.append(np.asarray(d["coverage"], bool))
            drive.append(np.asarray(d["drive"], np.float64))
            eog_raw.append(np.asarray(d["eog_raw"], np.float64))
            starts.append(np.asarray(d["starts"], np.int64))
            metrics.append(np.asarray(d["metrics"], np.float64))
            for k in rec:
                rec[k].append(d[k].item() if d[k].ndim == 0 else d[k])
            scale = np.asarray(d["eeg_scale"], np.float64)
            eeg_scale = scale if eeg_scale is None else eeg_scale
            expected = "|".join((str(d["participant"]), str(d["session"]), str(d["task"])))
            assert path.stem == expected, (path.name, expected)
    out = {
        "corrected": np.concatenate(corrected, axis=1),            # (cond, window, 46, 512)
        "coverage": np.concatenate(coverage, axis=1),              # (cond, window)
        "drive": np.concatenate(drive, axis=0),                    # (window, 2, 512)
        "eog_raw": np.concatenate(eog_raw, axis=0),                # (window, 2, 512)
        "starts": np.concatenate(starts), "metrics": np.concatenate(metrics, axis=1),
        "n_recording": len(paths), "eeg_scale": eeg_scale,
    }
    for k, v in rec.items():
        out["recording_" + k] = np.asarray(v)
    n_rec = len(paths)
    out["window_recording"] = np.repeat(np.arange(n_rec), N_WINDOWS)
    for k in ("participant", "session", "task"):
        out["window_" + k] = np.repeat(out["recording_" + k], N_WINDOWS)
    participants = sorted(set(out["recording_participant"].tolist()))
    out["participants"] = np.asarray(participants)
    out["recording_participant_index"] = np.asarray([participants.index(p) for p in out["recording_participant"]])
    out["window_participant_index"] = np.repeat(out["recording_participant_index"], N_WINDOWS)
    out["recording_movement"] = np.asarray([sg.SESSION_CONDITION[s] for s in out["recording_session"]])
    out["window_movement"] = np.repeat(out["recording_movement"], N_WINDOWS)
    return out


def gate_text() -> str:
    path = SHARD_DIR / "gate.json"
    return path.read_text() if path.is_file() else "{}"


def common_keys(S, extra_note: str):
    n_cond = len(CONDITIONS)
    return dict(
        condition=np.asarray(CONDITIONS), condition_label=np.asarray([label(c) for c in CONDITIONS]),
        recording_participant=S["recording_participant"], recording_session=S["recording_session"],
        recording_task=S["recording_task"], recording_movement=S["recording_movement"],
        recording_participant_index=S["recording_participant_index"],
        recording_wrong_donor=S["recording_wrong_donor"], recording_fold=S["recording_fold"],
        **{"recording_lambda": S["recording_lambda"]}, recording_hard_gate=S["recording_hard_gate"],
        participants=S["participants"], seed=np.asarray(20261201), k_trajectories=np.asarray(1),
        source=np.asarray(str(SHARD_DIR)), gate=np.asarray(gate_text()),
        arm_note=np.asarray("diffusion arms re-sampled with the fold checkpoints (seed 20261201, "
                            "K=1) on run_v44._natural_windows, gate-checked bit-identical to the "
                            "stored V44-S1 natural rows on the first cell; ICA/ASR/SGEYESUB from "
                            "cpu_rows' per-cell fitters; " + extra_note),
        n_condition=np.asarray(n_cond),
    )


# ---------------------------------------------------------------- E2

def abs_corr(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """|Pearson r| between every row of x (..., C, n) and every row of e (2, n) -> (..., C, 2)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    ec = e - e.mean(axis=-1, keepdims=True)
    num = xc @ ec.T
    den = np.linalg.norm(xc, axis=-1)[..., None] * np.linalg.norm(ec, axis=-1)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.abs(r)


def recording_mean(values: np.ndarray, coverage: np.ndarray, n_rec: int):
    """nanmean over the 4 windows of each recording along axis 1 -> (cond, rec, ...) plus n covered."""
    shape = values.shape
    v = values.reshape(shape[0], n_rec, N_WINDOWS, *shape[2:])
    cov = coverage.reshape(shape[0], n_rec, N_WINDOWS)
    v = np.where(cov.reshape(cov.shape + (1,) * (v.ndim - 3)), v, np.nan)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # empty slice = uncovered recording
        mean = np.nanmean(v, axis=2)
    return mean, cov.sum(axis=2), cov.all(axis=2)


def e2(S) -> dict:
    n_cond, n_win = S["coverage"].shape
    corr = np.full((n_cond, n_win, 46, 2), np.nan)
    corr_high = np.full((n_cond, n_win, 46, 2), np.nan)
    n_high = np.zeros(n_win, np.int64)
    for w in range(n_win):
        x = S["corrected"][:, w].astype(np.float64)
        e = S["eog_raw"][w]
        _, high = sg.eog_masks(S["drive"][w])
        n_high[w] = int(high.sum())
        corr[:, w] = abs_corr(x, e)
        corr_high[:, w] = abs_corr(x[..., high], e[:, high])
    cov = S["coverage"]
    corr[~cov] = np.nan
    corr_high[~cov] = np.nan
    corr_rec, n_cov, cov_rec = recording_mean(corr, cov, S["n_recording"])
    corr_high_rec, _, _ = recording_mean(corr_high, cov, S["n_recording"])
    return dict(
        corr=corr, corr_high=corr_high, coverage=cov, n_high=n_high,
        corr_recording=corr_rec, corr_high_recording=corr_high_rec,
        coverage_recording=cov_rec, n_windows_covered=n_cov,
        window_recording=S["window_recording"], window_participant=S["window_participant"],
        window_participant_index=S["window_participant_index"], window_session=S["window_session"],
        window_task=S["window_task"], window_movement=S["window_movement"], window_start=S["starts"],
        eog_names=np.asarray(["VEOG", "HEOG"]), eog_axis=np.asarray(["vertical", "horizontal"]),
        method_note=np.asarray("|Pearson r| between each corrected channel and the raw bipolar "
                               "VEOG (VEOGU-VEOGL) / HEOG (HEOGL-HEOGR) of the same 5.12-s window; "
                               "corr over all 512 samples, corr_high over the top-30% EOG-energy "
                               "samples (sg_common.eog_masks on the latent drive, exactly "
                               "run_v44._natural_metrics' H); channels with zero variance -> NaN"),
        **common_keys(S, "E2"),
    )


# ---------------------------------------------------------------- E3 / E6

def psd_products(S):
    n_cond, n_win = S["coverage"].shape
    el = [IDX[n] for n in ELECTRODES]
    i_raw = CONDITIONS.index("RAW")
    n_low = np.zeros(n_win, np.int64)
    lows = []
    for w in range(n_win):
        low, _ = sg.eog_masks(S["drive"][w])
        lows.append(low)
        n_low[w] = int(low.sum())
    nperseg = int(min(sg.WELCH_NPERSEG, n_low.min()))
    noverlap = int(min(sg.WELCH_NOVERLAP, nperseg // 2))
    freqs = None
    ratio_db = psd = None
    band_names = list(BANDS)
    band_edges = np.asarray([BANDS[b] for b in band_names], np.float64)
    bp = None
    for w in range(n_win):
        low = lows[w]
        x = S["corrected"][:, w][:, el][..., low].astype(np.float64)      # (cond, 12, n_low)
        f, p = sg.welch(x, nperseg)
        if freqs is None:
            freqs = f
            ratio_db = np.full((n_cond, n_win, len(el), len(f)), np.nan)
            psd = np.full((n_cond, n_win, len(el), len(f)), np.nan)
            bp = np.full((n_cond, n_win, len(band_names), len(el)), np.nan)
        assert np.array_equal(f, freqs), "frequency grid changed between windows"
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_db[:, w] = 10.0 * np.log10(p / p[i_raw][None])
        psd[:, w] = p
        df = float(f[1] - f[0])
        for b, (lo, hi) in enumerate(band_edges):
            sel = (f >= lo) & (f < hi)
            assert sel.any(), (band_names[b], lo, hi, f)
            bp[:, w, b] = p[..., sel].sum(axis=-1) * df
    cov = S["coverage"]
    ratio_db[~cov] = np.nan
    psd[~cov] = np.nan
    bp[~cov] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        bp_ratio = bp / bp[i_raw][None]
    return dict(freqs=freqs, nperseg=nperseg, noverlap=noverlap, n_low=n_low, ratio_db=ratio_db,
                psd=psd, bp=bp, bp_ratio=bp_ratio, band_names=band_names, band_edges=band_edges,
                electrode_index=np.asarray(el))


def e3(S, P) -> dict:
    cov = S["coverage"]
    ratio_rec, n_cov, cov_rec = recording_mean(P["ratio_db"], cov, S["n_recording"])
    psd_rec, _, _ = recording_mean(P["psd"], cov, S["n_recording"])
    note = ("Welch on the concatenated LOW-EOG samples of each 5.12-s window (bottom 30% EOG "
            "energy per sg_common.eog_masks on the latent drive, exactly run_v44._natural_metrics' "
            f"L; n_low = {int(P['n_low'].min())}..{int(P['n_low'].max())} non-contiguous samples). "
            "The doc's 2-s window / 1-s overlap cannot be honoured on ~150 samples, so "
            f"sg_common.welch runs with nperseg = min(200, min n_low) = {P['nperseg']} "
            f"(Hann, noverlap {P['noverlap']}, 100 Hz, {sg.FMIN}-{sg.FMAX} Hz kept), i.e. one "
            "tapered segment per window; ratio_db_window = 10*log10(PSD_corrected/PSD_raw) per "
            "window, ratio_db = its mean over the 4 windows of the recording (mean of dB); the "
            "participant mean over 6 recordings adds the remaining smoothing. Concatenating "
            "non-contiguous samples introduces discontinuities shared by numerator and denominator.")
    return dict(
        ratio_db=ratio_rec, ratio_db_window=P["ratio_db"], psd=psd_rec, psd_window=P["psd"],
        freqs=P["freqs"], nperseg=np.asarray(P["nperseg"]), noverlap=np.asarray(P["noverlap"]),
        n_low=P["n_low"], electrodes=np.asarray(ELECTRODES), electrode_index=P["electrode_index"],
        nine=np.asarray(NINE), extra_row=np.asarray(EXTRA_ROW),
        coverage=cov_rec, coverage_window=cov, n_windows_covered=n_cov,
        window_recording=S["window_recording"], window_start=S["starts"],
        psd_units=np.asarray("fold-scaled units^2 / Hz (prepared microvolt / eeg_scale per channel)"),
        method_note=np.asarray(note), **common_keys(S, "E3"),
    )


def e6(S, P) -> dict:
    cov = S["coverage"]
    ratio_rec, n_cov, cov_rec = recording_mean(P["bp_ratio"], cov, S["n_recording"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_db_win = 10.0 * np.log10(P["bp_ratio"])
    ratio_db_rec, _, _ = recording_mean(ratio_db_win, cov, S["n_recording"])
    power_rec, _, _ = recording_mean(P["bp"], cov, S["n_recording"])
    note = ("band power = sum of the E3 low-EOG-sample Welch PSD bins with lo <= f < hi times df "
            f"(nperseg {P['nperseg']}, see natural_psd_ratio.npz method_note); ratio_window = "
            "power after / power before (RAW) per window; ratio = mean over the 4 windows "
            "(linear), ratio_db = mean over windows of 10*log10(ratio_window)")
    return dict(
        ratio=ratio_rec, ratio_db=ratio_db_rec, ratio_window=P["bp_ratio"], power=power_rec,
        bands=np.asarray(P["band_names"]), band_edges_hz=P["band_edges"],
        electrodes=np.asarray(ELECTRODES), electrode_index=P["electrode_index"],
        coverage=cov_rec, coverage_window=cov, n_windows_covered=n_cov,
        window_recording=S["window_recording"], window_start=S["starts"],
        method_note=np.asarray(note), **common_keys(S, "E6"),
    )


# ---------------------------------------------------------------- write

def write(name: str, arrays: dict) -> Path:
    path = sg.ARRAYS_OUT / name
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            kept = {k: old[k] for k in old.files if k not in arrays}
        if kept:
            print(json.dumps({"kept_existing_keys": sorted(kept), "file": name}), flush=True)
        arrays = {**kept, **arrays}
    return sg.save_npz(name, **arrays)


def run(shard_dir: Path) -> None:
    t0 = time.time()
    S = load_shards(shard_dir)
    n_cond, n_win = S["coverage"].shape
    print(json.dumps({"shards": int(S["n_recording"]), "windows": int(n_win),
                      "coverage": {c: int(S["coverage"][i].sum()) for i, c in enumerate(CONDITIONS)},
                      "t": round(time.time() - t0, 1)}), flush=True)
    A2 = e2(S)
    write(E2_NAME, A2)
    P = psd_products(S)
    A3 = e3(S, P)
    write(E3_NAME, A3)
    A6 = e6(S, P)
    write(E6_NAME, A6)
    summary = {
        "E2_corr_mean_over_covered": {c: [round(float(np.nanmean(A2["corr"][i][..., a])), 3) for a in (0, 1)]
                                      for i, c in enumerate(CONDITIONS)},
        "E3_shape": list(A3["ratio_db"].shape), "E3_nperseg": int(A3["nperseg"]),
        "E3_freqs": [round(float(f), 2) for f in A3["freqs"]],
        "E3_mean_db_Fz": {c: round(float(np.nanmean(A3["ratio_db"][i, :, ELECTRODES.index("Fz")])), 2)
                          for i, c in enumerate(CONDITIONS)},
        "E6_shape": list(A6["ratio"].shape),
        "E6_mean_ratio_by_band": {c: [round(float(np.nanmean(A6["ratio"][i, :, b])), 3)
                                      for b in range(len(A6["bands"]))] for i, c in enumerate(CONDITIONS)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(summary), flush=True)


def smoke() -> None:
    """Synthetic shards (3 cells, 2 participants) -> scratch arrays; no data loads."""
    scratch = Path("/tmp/claude-34987/-home-infres-yinwang-denoiseNet/11cbfe34-6c44-4498-bfb8-deafdf46e607/scratchpad")
    shard_dir = scratch / "sg_natural_pass_smoke"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for p in shard_dir.glob("*.npz"):
        p.unlink()
    rng = np.random.default_rng(0)
    cells = [("sub-02", "ses-02", "ERP"), ("sub-02", "ses-03", "SSVEP"), ("sub-05", "ses-04", "ERP")]
    for ci, key in enumerate(cells):
        t = np.arange(WINDOW) / 100.0
        drive = rng.standard_normal((N_WINDOWS, 2, WINDOW))
        drive[:, 0, 100:160] += 8.0                                   # a blink
        eog_raw = drive * 40.0 + 3.0
        A = rng.standard_normal((46, 2)) * 0.3
        eeg = rng.standard_normal((N_WINDOWS, 46, WINDOW)) + np.sin(2 * np.pi * 10 * t)[None, None]
        raw = eeg + np.einsum("ce,wet->wct", A, drive)
        corrected = np.full((len(CONDITIONS), N_WINDOWS, 46, WINDOW), np.nan, np.float32)
        coverage = np.zeros((len(CONDITIONS), N_WINDOWS), bool)
        for i, c in enumerate(CONDITIONS):
            if c == "ASR" and ci == 0:
                continue                                              # a failed fitter
            frac = {"RAW": 0.0, "MATCH_gated": 0.95, "POP": 0.8, "NO_A0": 0.6, "LINEAR": 0.9,
                    "WRONG_gated": 0.5, "SHUFFLED": 0.1, "ICA": 0.85, "ASR": 0.7, "SGEYESUB": 0.9}[c]
            corrected[i] = (raw - frac * np.einsum("ce,wet->wct", A, drive)
                            + 0.05 * frac * rng.standard_normal(raw.shape)).astype(np.float32)
            coverage[i] = True
        np.savez_compressed(shard_dir / ("|".join(key) + ".npz"),
                            corrected=corrected, coverage=coverage, condition=np.asarray(CONDITIONS),
                            eeg_names=np.asarray(EEG_NAMES), drive=drive.astype(np.float32),
                            eog_raw=eog_raw.astype(np.float32), starts=np.arange(N_WINDOWS) * 5000 + 30000,
                            metrics=np.zeros((len(CONDITIONS), N_WINDOWS, 7)),
                            participant=np.asarray(key[0]), session=np.asarray(key[1]), task=np.asarray(key[2]),
                            wrong_donor=np.asarray("sub-03"), fold=np.asarray(0), ica_n_excluded=np.asarray(2),
                            asr_error=np.asarray("" if ci else "smoke failure"), hard_gate=np.asarray(0),
                            eeg_scale=np.ones(46), **{"lambda": np.asarray(0.5)})
    sg.ARRAYS_OUT = scratch / "paper_final_arrays_smoke"
    run(shard_dir)
    with np.load(sg.ARRAYS_OUT / E2_NAME, allow_pickle=False) as d:
        corr = d["corr"]
        i_raw, i_match, i_asr = (CONDITIONS.index(c) for c in ("RAW", "MATCH_gated", "ASR"))
        assert corr.shape == (len(CONDITIONS), 12, 46, 2)
        assert np.nanmean(corr[i_match]) < np.nanmean(corr[i_raw]), "synthetic correction did not lower |r|"
        assert np.isnan(corr[i_asr, :4]).all() and not d["coverage"][i_asr, :4].any()
        assert np.isfinite(corr[i_asr, 4:]).all()
    with np.load(sg.ARRAYS_OUT / E3_NAME, allow_pickle=False) as d:
        assert d["ratio_db"].shape == (len(CONDITIONS), 3, 12, len(d["freqs"]))
        assert np.allclose(d["ratio_db"][CONDITIONS.index("RAW")], 0.0)
        assert np.isnan(d["ratio_db"][CONDITIONS.index("ASR"), 0]).all()
    with np.load(sg.ARRAYS_OUT / E6_NAME, allow_pickle=False) as d:
        assert d["ratio"].shape == (len(CONDITIONS), 3, 3, 12)
        assert np.allclose(d["ratio"][CONDITIONS.index("RAW")], 1.0)
    print(json.dumps({"smoke": True, "arrays": str(sg.ARRAYS_OUT), "shards": str(shard_dir)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shard-dir", default=str(SHARD_DIR))
    args = parser.parse_args()
    smoke() if args.smoke else run(Path(args.shard_dir))
