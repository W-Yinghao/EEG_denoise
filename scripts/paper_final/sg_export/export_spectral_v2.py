#!/usr/bin/env python
"""SERVER_INSTRUCTIONS_SPECTRAL_COST_V2.md — items A and C (CPU).

A. Band-pass-then-mask spectral cost, no concatenation anywhere: each corrected
   window (10 conditions x 46 x 512, fold-scaled, as stored by the natural pass)
   is filtered as one contiguous 512-sample window with a zero-phase Butterworth
   (sosfiltfilt), and the mean square is taken over the masked samples only
   (L = bottom 30 % EOG energy, H = top 30 %, exactly sg_common.eog_masks on the
   latent drive).  Replaces the concatenated-Welch estimator of E3/E6, whose
   step discontinuities leak broadband power that does not cancel in the ratio.
   -> natural_band_power_v2.npz, natural_psd_ratio_v2.npz

C. Natural endpoints for all ten arms from the same pass via run_v44._natural_metrics
   -> natural_endpoints_all_arms.npz

Gates are asserted and printed as single lines for the commit message.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sg_common as sg  # noqa: E402
from export_natural_post import (CONDITIONS, ELECTRODES, N_WINDOWS, NINE, SHARD_DIR,  # noqa: E402
                                 common_keys, load_shards, recording_mean, write)

FS = 100.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "broad": (0.5, 15.0)}
SOS = {b: butter(4, [lo, hi], btype="band", fs=FS, output="sos") for b, (lo, hi) in BANDS.items()}
CENTRES = list(range(1, 15))
BANK = {k: butter(2, [k - 0.5, k + 0.5], btype="band", fs=FS, output="sos") for k in CENTRES}
N_MASK = 154
I_RAW = CONDITIONS.index("RAW")
EL = np.asarray([sg.IDX[n] for n in ELECTRODES])
NINE_IDX = np.asarray([ELECTRODES.index(n) for n in NINE])


def band_power_on_mask(x: np.ndarray, mask: np.ndarray, sos_dict: dict) -> dict:
    """x: (..., n) contiguous windows; mask: (n,) bool -> {band: mean square over masked samples}."""
    return {b: (sosfiltfilt(s, x, axis=-1)[..., mask] ** 2).mean(-1) for b, s in sos_dict.items()}


def participant_first(rec_values: np.ndarray, rec_participant_index: np.ndarray) -> np.ndarray:
    """(cond, rec, ...) -> (cond, ...): nanmean over each participant's recordings, then mean over participants."""
    import warnings
    parts = np.unique(rec_participant_index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per = np.stack([np.nanmean(rec_values[:, rec_participant_index == p], axis=1) for p in parts], axis=1)
        return np.nanmean(per, axis=1)


# ------------------------------------------------------------------------- A

def item_a(S) -> dict:
    n_cond, n_win = S["coverage"].shape
    n_rec = S["n_recording"]
    shape_b = (n_cond, n_win, len(BANDS), len(ELECTRODES))
    shape_k = (n_cond, n_win, len(CENTRES), len(ELECTRODES))
    P_L, P_H = np.full(shape_b, np.nan), np.full(shape_b, np.nan)
    K_L, K_H = np.full(shape_k, np.nan), np.full(shape_k, np.nan)
    n_low = np.zeros(n_win, int); n_high = np.zeros(n_win, int)
    for w in range(n_win):
        low, high = sg.eog_masks(S["drive"][w])
        n_low[w], n_high[w] = int(low.sum()), int(high.sum())
        x = S["corrected"][:, w][:, EL].astype(np.float64)          # (cond, 12, 512) contiguous
        for mask, P, K in ((low, P_L, K_L), (high, P_H, K_H)):
            bp = band_power_on_mask(x, mask, SOS)
            for j, b in enumerate(BANDS):
                P[:, w, j] = bp[b]
            bk = band_power_on_mask(x, mask, BANK)
            for j, k in enumerate(CENTRES):
                K[:, w, j] = bk[k]
    assert (n_low == N_MASK).all() and (n_high == N_MASK).all(), (n_low.min(), n_low.max(), n_high.min(), n_high.max())
    cov = S["coverage"]
    covf = cov[..., None, None]

    def ratios(P):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = 10.0 * np.log10(P / P[I_RAW][None])
        return np.where(covf, r, np.nan)

    R_L, R_H, RK_L, RK_H = ratios(P_L), ratios(P_H), ratios(K_L), ratios(K_H)
    ratio_rec_L, n_cov, cov_rec = recording_mean(R_L, cov, n_rec)
    ratio_rec_H, _, _ = recording_mean(R_H, cov, n_rec)
    power_rec_L, _, _ = recording_mean(np.where(covf, P_L, np.nan), cov, n_rec)
    power_rec_H, _, _ = recording_mean(np.where(covf, P_H, np.nan), cov, n_rec)
    kr_rec_L, _, _ = recording_mean(RK_L, cov, n_rec)
    kr_rec_H, _, _ = recording_mean(RK_H, cov, n_rec)

    estimator = ("fourth-order zero-phase Butterworth band-pass on the contiguous 512-sample window "
                 "(scipy sosfiltfilt), mean square over the masked samples; no concatenation")
    mask_note = (f"L = bottom 30% of per-sample EOG energy of the latent drive (sg_common.eog_masks, "
                 f"run_v44._natural_metrics' L), H = top 30%; n_low = n_high = {N_MASK} in every window")
    book = dict(electrodes=np.asarray(ELECTRODES), electrode_index=EL, coverage=cov_rec, coverage_window=cov,
                n_windows_covered=n_cov, window_recording=S["window_recording"], window_start=S["starts"],
                window_participant=S["window_participant"], window_participant_index=S["window_participant_index"],
                window_session=S["window_session"], window_task=S["window_task"], window_movement=S["window_movement"],
                estimator=np.asarray(estimator), mask_note=np.asarray(mask_note))
    e6 = dict(ratio_db=ratio_rec_L, ratio_db_window=R_L, power=power_rec_L,
              ratio_db_high=ratio_rec_H, ratio_db_window_high=R_H, power_high=power_rec_H,
              bands=np.asarray(list(BANDS)), band_edges_hz=np.asarray([BANDS[b] for b in BANDS], float),
              **book, **common_keys(S, "E6b band-pass-then-mask"))
    e3 = dict(ratio_db=kr_rec_L, ratio_db_window=RK_L, ratio_db_high=kr_rec_H, ratio_db_window_high=RK_H,
              centres_hz=np.asarray(CENTRES, float), bandwidth_hz=np.asarray(1.0),
              filter_note=np.asarray("second-order zero-phase Butterworth band-pass, 1 Hz wide, centres 1..14 Hz"),
              **book, **common_keys(S, "E3b band-pass-then-mask"))
    write("natural_band_power_v2.npz", e6)
    write("natural_psd_ratio_v2.npz", e3)

    # ---- gates
    pf_L = participant_first(ratio_rec_L[..., NINE_IDX].mean(-1), S["recording_participant_index"])   # (cond, band)
    pf_H = participant_first(ratio_rec_H[..., NINE_IDX].mean(-1), S["recording_participant_index"])
    i_ng = CONDITIONS.index("NO_A0")
    g_unguided = np.abs(pf_L[i_ng, :3])
    g_high = {c: float(pf_H[CONDITIONS.index(c), 0]) for c in ("MATCH_gated", "ICA", "SGEYESUB")}
    banked = np.load(sg.ARRAYS_OUT / "natural_band_power.npz", allow_pickle=False)
    b_ratio = banked["ratio_db"]                                                             # (10, 90, 3, 12)
    pf_banked = participant_first(b_ratio[..., NINE_IDX].mean(-1), S["recording_participant_index"])
    lines = [
        f"GATE n_low == n_high == {N_MASK} in every window: PASS ({n_win} windows)",
        f"GATE unguided L nine-electrode participant-first |delta,theta,alpha| = "
        f"{g_unguided[0]:.4f}/{g_unguided[1]:.4f}/{g_unguided[2]:.4f} dB (< 0.05): {'PASS' if (g_unguided < 0.05).all() else 'FAIL'}",
        f"GATE H-mask delta subject/ICA/eye-subspace = {g_high['MATCH_gated']:.3f}/{g_high['ICA']:.3f}/{g_high['SGEYESUB']:.3f} dB (< -3): "
        f"{'PASS' if all(v < -3 for v in g_high.values()) else 'FAIL'}",
        "TABLE L nine-electrode participant-first dB (v2 band-pass | banked concatenated) delta/theta/alpha:"]
    for i, c in enumerate(CONDITIONS):
        lines.append(f"  {c:12s} v2 {pf_L[i,0]:+.3f}/{pf_L[i,1]:+.3f}/{pf_L[i,2]:+.3f} | banked "
                     f"{pf_banked[i,0]:+.3f}/{pf_banked[i,1]:+.3f}/{pf_banked[i,2]:+.3f}")
    out = {"gate_lines": lines, "pf_L": pf_L.tolist(), "pf_H": pf_H.tolist(), "pf_banked": pf_banked.tolist()}
    (sg.ARRAYS_OUT / "spectral_v2_gates.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines), flush=True)
    return out


# ------------------------------------------------------------------------- C

def item_c(S) -> dict:
    from eeg_scad.cli import run_v44 as up
    from pf_common import stored_stage1_natural_rows
    n_cond, n_win = S["coverage"].shape
    names = ("attenuation_db", "low_eog_observation_retention", "coherence_reduction")
    M = {k: np.full((n_cond, n_win), np.nan) for k in names}
    stored_names = None
    max_dev = 0.0
    for w in range(n_win):
        y = S["corrected"][I_RAW, w].astype(np.float64)
        drive = S["drive"][w]
        for c in range(n_cond):
            if not S["coverage"][c, w]:
                continue
            x = S["corrected"][c, w].astype(np.float64)
            m = up._natural_metrics(y, drive, y - x)
            for k in names:
                M[k][c, w] = m[k]
    # self-check against the metrics the pass stored alongside the waveforms
    shard0 = sorted(SHARD_DIR.glob("sub-*.npz"))[0]
    with np.load(shard0, allow_pickle=False) as d:
        stored_names = [str(n) for n in d["metric_names"]]
    for k in names:
        j = stored_names.index(k)
        stored = S["metrics"][:, :, j]
        ok = S["coverage"]
        max_dev = max(max_dev, float(np.nanmax(np.abs(M[k][ok] - stored[ok]))))
    pidx = S["recording_participant_index"]
    n_rec = S["n_recording"]
    pf = {}
    for k in names:
        rec, _, _ = recording_mean(M[k], S["coverage"], n_rec)
        pf[k] = participant_first(rec, pidx)                                                  # (cond,)
    write("natural_endpoints_all_arms.npz", dict(
        attenuation_db=M["attenuation_db"], retention=M["low_eog_observation_retention"],
        coherence_reduction=M["coherence_reduction"], coverage=S["coverage"],
        window_recording=S["window_recording"], window_start=S["starts"],
        window_participant=S["window_participant"], window_participant_index=S["window_participant_index"],
        window_session=S["window_session"], window_task=S["window_task"], window_movement=S["window_movement"],
        participant_first_attenuation_db=pf["attenuation_db"], participant_first_retention=pf["low_eog_observation_retention"],
        participant_first_coherence_reduction=pf["coherence_reduction"],
        metric_note=np.asarray("run_v44._natural_metrics(y, drive, y - corrected) per window; recording mean over "
                               "4 windows, participant mean over recordings, cohort mean over participants"),
        **common_keys(S, "C natural endpoints all arms")))

    # ---- gates: (i) stored seed-20261201 V44-S1 rows (same seed, same windows -> should be exact)
    rows = stored_stage1_natural_rows(seeds=(20261201,))
    stored_pf = {}
    for cond in ("MATCH_gated", "POP", "NO_A0", "WRONG_gated", "SHUFFLED"):
        per = {}
        for r in rows:
            if r["condition"] == cond:
                per.setdefault(r["participant"], []).append([r[k] for k in names])
        stored_pf[cond] = np.mean([np.mean(v, axis=0) for v in per.values()], axis=0)
    t5 = json.load(open(sg.OUT / "t5_natural_plane.json"))["five_condition_table"]
    cpu = json.load(open(sg.OUT / "cpu_reference_rows.json"))
    lines = [f"SELFCHECK recomputed _natural_metrics vs the pass's stored metrics: max |dev| = {max_dev:.2e}"]
    worst_seed = 0.0
    for cond in stored_pf:
        i = CONDITIONS.index(cond)
        ours = np.array([pf[k][i] for k in names]); dev = np.abs(ours - stored_pf[cond]).max(); worst_seed = max(worst_seed, dev)
        t5v = np.array([t5[cond][k] for k in names]) if cond in t5 else None
        lines.append(f"GATE {cond:12s} ours {ours[0]:.4f}/{ours[1]:.4f}/{ours[2]:.4f} | stored seed-20261201 rows "
                     f"{stored_pf[cond][0]:.4f}/{stored_pf[cond][1]:.4f}/{stored_pf[cond][2]:.4f} (max dev {dev:.1e})"
                     + (f" | T5 3-seed table {t5v[0]:.4f}/{t5v[1]:.4f}/{t5v[2]:.4f}" if t5v is not None else ""))
    lines.append(f"GATE five diffusion arms vs stored seed-20261201 rows: max dev {worst_seed:.1e} ({'PASS' if worst_seed < 1e-6 else 'FAIL'} at 1e-6)")
    worst_cpu = 0.0
    for cond, key in (("ICA", "ica"), ("ASR", "asr"), ("SGEYESUB", "sgeyesub")):
        i = CONDITIONS.index(cond)
        ref = np.array([cpu[key]["natural"][k]["mean"] for k in names])
        ours = np.array([pf[k][i] for k in names]); dev = np.abs(ours - ref).max(); worst_cpu = max(worst_cpu, dev)
        lines.append(f"GATE {cond:12s} ours {ours[0]:.4f}/{ours[1]:.4f}/{ours[2]:.4f} | cpu_reference_rows {ref[0]:.4f}/{ref[1]:.4f}/{ref[2]:.4f} (max dev {dev:.1e})")
    lines.append(f"GATE reference methods vs cpu_reference_rows: max dev {worst_cpu:.1e} ({'PASS' if worst_cpu < 1e-6 else 'FAIL'} at 1e-6)")
    i = CONDITIONS.index("LINEAR")
    lines.append(f"REPORT LINEAR (calibrated linear regression) attenuation/retention/coherence = "
                 f"{pf['attenuation_db'][i]:.4f} dB / {pf['low_eog_observation_retention'][i]:.4f} / {pf['coherence_reduction'][i]:.4f}")
    lines.append("TABLE all arms participant-first attenuation_db / retention / coherence_reduction:")
    for j, c in enumerate(CONDITIONS):
        lines.append(f"  {c:12s} {pf['attenuation_db'][j]:+.4f} / {pf['low_eog_observation_retention'][j]:.4f} / {pf['coherence_reduction'][j]:.4f}")
    out = {"gate_lines": lines, "participant_first": {k: pf[k].tolist() for k in names}}
    (sg.ARRAYS_OUT / "natural_endpoints_gates.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(lines), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["a", "c", "ac"])
    args = parser.parse_args()
    S = load_shards(SHARD_DIR)
    if "a" in args.mode:
        item_a(S)
    if "c" in args.mode:
        item_c(S)


if __name__ == "__main__":
    main()
