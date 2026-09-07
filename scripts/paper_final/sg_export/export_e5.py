#!/usr/bin/env python3
"""E5 export — one natural window per movement condition (natural_exemplars.npz).

SERVER_EXPORT_FIGURE_ARRAYS_v4.md, item E5.  CPU only: nothing is re-sampled,
every corrected trace is read from the banked D-wave ERP tiles
(results/paper_final/dwave/denoised/d2_<cell>.npz and cpuarms_<cell>.npz via
sg_common.dwave_tiles), so the exemplar is exactly the tile the Table-8 / E4
endpoints already used.

Participant : sub-02 (accepted in all three sessions, hard_gate 0), task ERP.
Conditions  : one tile per session ses-02 / ses-03 / ses-04, labelled ONLY
              through sg_common.SESSION_CONDITION (standing / slow_walking /
              fast_walking); CONDITION_NOTE travels with the file.

Tile selection (calibration-independent activity only, no operator involved):
  rank_v  percentile rank within the cell of the stored per-tile `veog_rms`
          (d2 file: RMS of the latent-scaled vertical drive over the tile)
  rank_h  percentile rank of `heog_rms` recomputed here from the prepared
          record's raw bipolar HEOG (sg_common.record -> eog[1], tile = start:start+512)
  candidates = tiles with BOTH rms values above the cell's 70th percentile
  chosen     = argmax min(rank_v, rank_h); ties -> larger rank_v + rank_h, then
               earlier start; the next three candidates are the runner-ups.

Gates asserted at runtime (alignment of the stored tiles with the record):
  G1  the d2/cpuarms starts equal dwave.d2_tiles(record length) and agree
  G2  the stored veog_rms reproduces RMS(drive[0, tile]) to 1e-6 for every tile
  G3  the stored RAW tile equals record[:, start:start+512] (float32) for every tile
  G4  the stored LINEAR tile equals RAW - C_gated @ drive for the chosen tile (1e-5)
  G5  hard_gate is 0 in the d2 file AND in the frozen EB registry for every cell
  G6  every candidate tile has both rms values above the cell's 70th percentile

Keys (t6_waveform_exemplar_dev.npz names kept): contaminated, matched, unguided,
linear_regression, eog_drive, eeg_names, fixed_channels; ADDED: population,
ica, asr, eye_subspace (each (3, 46, 512), NaN + coverage False when the
cpuarms arm is absent for that cell), eog_raw_bipolar (3, 2, 512) in the units
of the prepared record (microvolt), condition, session, speed_mps, start,
tile_index, rank_v, rank_h, score, runner_up_*, veog_rms_all / heog_rms_all
(NaN-padded), coverage (3, 8) + coverage_arms, eeg_scale (46,) so every trace
can be put back into microvolt (multiply), lambda, hard_gate, fold.

Usage:  export_e5.py            full export (CPU, minutes: builds the fold-0 registries)
        export_e5.py --smoke    one d2 file, selection logic only (login-node check)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402
from sg_common import (CALIB, DWAVE_ARMS, EEG_NAMES, EXEMPLAR_STACK, EXPORT_LABEL,  # noqa: E402
                       IDX, WINDOW, dwave_tiles, save_npz)

sys.path.insert(0, str(sg.PF))
from dwave import d2_tiles  # noqa: E402

NAME = "natural_exemplars.npz"
PARTICIPANT, TASK = "sub-02", "ERP"
SESSIONS = ("ses-02", "ses-03", "ses-04")
PERCENTILE = 0.70
N_RUNNER_UP = 3
FIXED_CHANNELS = ("Fp1", "Fp2", "AFz")          # t6_waveform_exemplar_dev convention
# D-wave arm -> npz key (t6 names where they exist)
ARM_KEY = {"RAW": "contaminated", "MATCH": "matched", "POP": "population",
           "NO_A0": "unguided", "LINEAR": "linear_regression", "ICA": "ica",
           "ASR": "asr", "SGEYESUB": "eye_subspace"}
assert tuple(ARM_KEY) == DWAVE_ARMS


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=-1))


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank in (0, 1]; 1 = largest."""
    from scipy.stats import rankdata
    return rankdata(values, method="average") / float(len(values))


def select_tile(starts: np.ndarray, veog_rms: np.ndarray, heog_rms: np.ndarray):
    """Chosen tile index + ordered candidate list (see module docstring)."""
    n = len(starts)
    assert len(veog_rms) == len(heog_rms) == n
    rank_v, rank_h = percentile_rank(veog_rms), percentile_rank(heog_rms)
    thr_v, thr_h = np.quantile(veog_rms, PERCENTILE), np.quantile(heog_rms, PERCENTILE)
    candidate = (veog_rms > thr_v) & (heog_rms > thr_h)
    idx = np.flatnonzero(candidate)
    assert len(idx) >= 1, "no tile has both VEOG and HEOG activity above the 70th percentile"
    score = np.minimum(rank_v, rank_h)
    order = idx[np.lexsort((starts[idx], -(rank_v[idx] + rank_h[idx]), -score[idx]))]
    # G6
    assert (veog_rms[order] > thr_v).all() and (heog_rms[order] > thr_h).all()
    return {"chosen": int(order[0]), "order": order, "rank_v": rank_v, "rank_h": rank_h,
            "score": score, "n_candidates": int(len(idx)),
            "threshold_v": float(thr_v), "threshold_h": float(thr_h)}


def smoke() -> None:
    """One d2 file: starts arithmetic and the selection logic (no registry)."""
    tiles = dwave_tiles(PARTICIPANT, SESSIONS[0], TASK)
    starts = np.asarray(tiles["starts"], np.int64)
    veog = np.asarray(tiles["veog_rms"], float)
    assert starts[0] == CALIB and np.all(np.diff(starts) == WINDOW)
    assert int(tiles["hard_gate"]) == 0
    # stand-in for the HEOG rms: the RAW tile's Fp1-Fp2 rms (smoke only, never exported)
    heog_proxy = _rms(tiles["RAW"][:, IDX["Fp1"]] - tiles["RAW"][:, IDX["Fp2"]])
    sel = select_tile(starts, veog, heog_proxy)
    print(json.dumps({"smoke": True, "cell": f"{PARTICIPANT}|{SESSIONS[0]}|{TASK}",
                      "tiles": int(len(starts)), "arms_present": sorted(
                          a for a in DWAVE_ARMS if a in tiles),
                      "chosen_start": int(starts[sel["chosen"]]),
                      "n_candidates": sel["n_candidates"],
                      "runner_ups": [int(s) for s in starts[sel["order"][1:1 + N_RUNNER_UP]]]}))


def main() -> None:
    from eeg_scad.cli.run_v44 import _gated_assets

    t0 = time.time()
    fold_id, fold = sg.fold_of(PARTICIPANT)
    data, r30, eb120 = sg.registry(fold)
    assets = _gated_assets(r30, eb120)
    print(json.dumps({"fold": fold_id, "test": sorted(fold["test"]),
                      "t": round(time.time() - t0, 1)}), flush=True)
    eeg_scale = np.asarray(r30.eeg_scale, np.float64)
    assert eeg_scale.shape == (46,)

    n_cond = len(SESSIONS)
    traces = {arm: np.full((n_cond, 46, WINDOW), np.nan, np.float32) for arm in DWAVE_ARMS}
    coverage = np.zeros((n_cond, len(DWAVE_ARMS)), bool)
    eog_drive = np.full((n_cond, 2, WINDOW), np.nan, np.float32)
    eog_raw = np.full((n_cond, 2, WINDOW), np.nan, np.float32)
    n_max = 0
    per_cell = []
    for session in SESSIONS:
        key = (PARTICIPANT, session, TASK)
        tiles = dwave_tiles(*key)
        eeg, drive, eog = sg.record(r30, key)
        length = min(eeg.shape[1], eog.shape[1])
        starts = np.asarray(tiles["starts"], np.int64)
        # G1: the tile grid is the D-wave grid on this record
        expected = np.asarray(d2_tiles(length), np.int64)
        assert np.array_equal(starts, expected), (key, starts[:3], expected[:3], len(starts), len(expected))
        # G5: accepted calibration in the stored file and in the frozen registry
        cell = eb120.cells[key]
        assert int(tiles["hard_gate"]) == 0 and not bool(cell.hard_gate), (key, "hard_gate")
        # G2: the stored veog_rms is RMS of the latent vertical drive on every tile
        veog_stored = np.asarray(tiles["veog_rms"], np.float64)
        veog_re = np.asarray([_rms(drive[0, s:s + WINDOW]) for s in starts])
        g2 = float(np.abs(veog_stored - veog_re).max())
        assert g2 < 1e-6, (key, "G2 veog_rms mismatch", g2)
        # G3: the stored RAW tiles are the record (float32 cast)
        raw_stack = np.stack([eeg[:, s:s + WINDOW] for s in starts]).astype(np.float32)
        g3 = float(np.abs(raw_stack - tiles["RAW"]).max())
        assert g3 < 1e-5, (key, "G3 RAW tile mismatch", g3)
        # calibration-independent H activity: raw bipolar HEOG (microvolt) per tile
        heog_rms = np.asarray([_rms(eog[1, s:s + WINDOW]) for s in starts])
        veog_rms_raw = np.asarray([_rms(eog[0, s:s + WINDOW]) for s in starts])
        sel = select_tile(starts, veog_stored, heog_rms)
        i, s = sel["chosen"], int(starts[sel["chosen"]])
        # G4: LINEAR tile == RAW - C_gated @ drive on the chosen tile
        lin = raw_stack[i].astype(np.float64) - assets[key]["C_gated"] @ drive[:, s:s + WINDOW]
        g4 = float(np.abs(lin - tiles["LINEAR"][i]).max())
        assert g4 < 1e-5, (key, "G4 LINEAR tile mismatch", g4)

        c = SESSIONS.index(session)
        for a, arm in enumerate(DWAVE_ARMS):
            if arm in tiles:
                assert tiles[arm].shape == (len(starts), 46, WINDOW), (key, arm, tiles[arm].shape)
                traces[arm][c] = tiles[arm][i]
                coverage[c, a] = np.isfinite(tiles[arm][i]).all()
        eog_drive[c] = drive[:, s:s + WINDOW]
        eog_raw[c] = eog[:, s:s + WINDOW]
        n_max = max(n_max, len(starts))
        order = sel["order"]
        runner = order[1:1 + N_RUNNER_UP]
        # informational: is the blink / horizontal movement visible, and does MATCH remove it
        fp1 = IDX["Fp1"]
        r_raw = abs(np.corrcoef(traces["RAW"][c, fp1], eog_raw[c, 0])[0, 1])
        r_match = abs(np.corrcoef(traces["MATCH"][c, fp1], eog_raw[c, 0])[0, 1])
        per_cell.append({
            "key": key, "session": session, "start": s, "tile_index": i,
            "n_tiles": int(len(starts)), "n_candidates": sel["n_candidates"],
            "rank_v": float(sel["rank_v"][i]), "rank_h": float(sel["rank_h"][i]),
            "score": float(sel["score"][i]),
            "runner_up_start": [int(starts[j]) for j in runner],
            "runner_up_index": [int(j) for j in runner],
            "runner_up_rank_v": [float(sel["rank_v"][j]) for j in runner],
            "runner_up_rank_h": [float(sel["rank_h"][j]) for j in runner],
            "runner_up_score": [float(sel["score"][j]) for j in runner],
            "threshold_v": sel["threshold_v"], "threshold_h": sel["threshold_h"],
            "veog_rms_all": veog_stored, "heog_rms_all": heog_rms,
            "veog_rms_raw_all": veog_rms_raw, "starts_all": starts,
            "veog_peak_uv": float(np.abs(eog_raw[c, 0]).max()),
            "heog_peak_uv": float(np.abs(eog_raw[c, 1]).max()),
            "lambda": float(cell.lam), "hard_gate": int(cell.hard_gate),
            "gates": {"G2_veog_rms_max_abs": g2, "G3_raw_max_abs": g3, "G4_linear_max_abs": g4},
            "info_fp1_abs_r_veog_raw": float(r_raw), "info_fp1_abs_r_veog_match": float(r_match),
        })
        print(json.dumps({"cell": "|".join(key), "condition": sg.SESSION_CONDITION[session],
                          "start": s, "tiles": int(len(starts)), "candidates": sel["n_candidates"],
                          "rank_v": round(float(sel["rank_v"][i]), 3),
                          "rank_h": round(float(sel["rank_h"][i]), 3),
                          "runner_ups": [int(starts[j]) for j in runner],
                          "arms": [a for a in DWAVE_ARMS if coverage[c, DWAVE_ARMS.index(a)]],
                          "veog_peak_uv": round(per_cell[-1]["veog_peak_uv"], 1),
                          "heog_peak_uv": round(per_cell[-1]["heog_peak_uv"], 1),
                          "fp1_|r|_veog_raw_vs_match": [round(r_raw, 3), round(r_match, 3)],
                          "gates": {k: float(f"{v:.2e}") for k, v in per_cell[-1]["gates"].items()}}),
              flush=True)
    del r30, eb120, assets

    def padded(field, fill):
        out = np.full((n_cond, n_max), fill, dtype=np.float64 if fill is np.nan else np.int64)
        for c, row in enumerate(per_cell):
            out[c, :len(row[field])] = row[field]
        return out

    def stacked(field, dtype=np.float64):
        return np.asarray([row[field] for row in per_cell], dtype=dtype)

    sessions = np.asarray(SESSIONS)
    arrays = dict(
        participant=np.asarray(PARTICIPANT), task=np.asarray(TASK), fold=np.asarray(int(fold_id)),
        session=sessions,
        condition=np.asarray([sg.SESSION_CONDITION[s] for s in SESSIONS]),
        speed_mps=np.asarray([sg.SESSION_SPEED_MPS[s] for s in SESSIONS]),
        start=stacked("start", np.int64), tile_index=stacked("tile_index", np.int64),
        n_tiles=stacked("n_tiles", np.int64), n_candidates=stacked("n_candidates", np.int64),
        rank_v=stacked("rank_v"), rank_h=stacked("rank_h"), score=stacked("score"),
        threshold_v=stacked("threshold_v"), threshold_h=stacked("threshold_h"),
        runner_up_start=stacked("runner_up_start", np.int64),
        runner_up_index=stacked("runner_up_index", np.int64),
        runner_up_rank_v=stacked("runner_up_rank_v"), runner_up_rank_h=stacked("runner_up_rank_h"),
        runner_up_score=stacked("runner_up_score"),
        veog_rms_all=padded("veog_rms_all", np.nan), heog_rms_all=padded("heog_rms_all", np.nan),
        veog_rms_raw_all=padded("veog_rms_raw_all", np.nan), starts_all=padded("starts_all", -1),
        veog_peak_uv=stacked("veog_peak_uv"), heog_peak_uv=stacked("heog_peak_uv"),
        hard_gate=stacked("hard_gate", np.int64), **{"lambda": stacked("lambda")},
        eog_drive=eog_drive, eog_raw_bipolar=eog_raw,
        eog_names=np.asarray(["VEOG", "HEOG"]),
        eeg_names=np.asarray(EEG_NAMES), fixed_channels=np.asarray(FIXED_CHANNELS),
        stack_channels=np.asarray(EXEMPLAR_STACK),
        eeg_scale=eeg_scale,
        coverage=coverage, coverage_arms=np.asarray(DWAVE_ARMS),
        arm_keys=np.asarray([ARM_KEY[a] for a in DWAVE_ARMS]),
        arm_labels=np.asarray([EXPORT_LABEL[a] for a in DWAVE_ARMS]),
        selection_rule=np.asarray(
            "tile = argmax min(rank_v, rank_h) over tiles with veog_rms (stored, latent V drive) "
            "and heog_rms (raw bipolar HEOG, recomputed) both above the cell's 70th percentile; "
            "ties -> larger rank sum, then earlier start; no operator or calibration involved"),
        units=np.asarray("EEG traces are in fold-scaled units (prepared microvolt / eeg_scale per "
                         "channel); multiply by eeg_scale[:, None] for microvolt. eog_drive is the "
                         "latent (centred/scaled) drive; eog_raw_bipolar is [VEOGU-VEOGL, HEOGL-HEOGR] "
                         "in the prepared record's microvolt"),
        source=np.asarray(str(sg.OUT / "dwave/denoised")),
        gates=np.asarray(json.dumps({row["session"]: row["gates"] for row in per_cell})),
        info_fp1_abs_r_veog=np.asarray([[row["info_fp1_abs_r_veog_raw"], row["info_fp1_abs_r_veog_match"]]
                                        for row in per_cell]),
    )
    for arm in DWAVE_ARMS:
        arrays[ARM_KEY[arm]] = traces[arm]
    # every stored trace with coverage True must be finite; NaN exactly where coverage is False
    for a, arm in enumerate(DWAVE_ARMS):
        finite = np.isfinite(traces[arm]).all(axis=(1, 2))
        assert np.array_equal(finite, coverage[:, a]), (arm, finite, coverage[:, a])
    assert coverage[:, :5].all(), "a diffusion / RAW / LINEAR arm is missing"

    # keep the keys of an earlier version of the file, if any (rule 2)
    path = sg.ARRAYS_OUT / NAME
    if path.is_file():
        with np.load(path, allow_pickle=False) as old:
            kept = {k: old[k] for k in old.files if k not in arrays}
        if kept:
            print(json.dumps({"kept_existing_keys": sorted(kept)}), flush=True)
        arrays = {**kept, **arrays}
    save_npz(NAME, **arrays)
    print(json.dumps({"coverage": {arm: coverage[:, a].tolist() for a, arm in enumerate(DWAVE_ARMS)},
                      "starts": arrays["start"].tolist(), "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else main()
