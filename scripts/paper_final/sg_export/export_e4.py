#!/usr/bin/env python3
"""E4 — ERP grand averages and P300 peak topographies (erp_topographies.npz).

Source: the banked D-wave ERP tiles (results/paper_final/dwave/denoised/d2_* and
cpuarms_*), every ERP cell of both cohorts (dev 15 + sealed 8).  Epochs are cut
with scripts/paper_final/dwave_deep._erp_epochs, i.e. exactly the epochs of the
D-wave ERP endpoints: -0.2..0.8 s around each event at 100 Hz, an epoch must lie
inside one 512-sample tile after the 120-s calibration prefix, label 1 = target
(event value 2), and a cell with < 40 usable events is dropped.

Pooling (mirrors dwave_deep._collect): a participant's epochs are pooled across
the participant's usable cells (ses-02/03/04); an arm counts for a participant
only when it is present in EVERY usable cell, so the epoch set is identical
across arms within a participant (ASR is absent from some cells: asrpy
calibration bug, see dwave.cpu_arms).  Missing arms -> NaN + coverage False.

Units: the tiles are in fold-scaled units (eeg / fold eeg_scale, per channel).
The exported ERPs are multiplied back by that scale so they are in the units of
the prepared records (microvolt); `eeg_scale` (cohort, participant, 46) is
stored so the tile-unit version is exactly recoverable (divide).

Baseline: per epoch and channel, the mean of the 20 pre-stimulus samples
(-0.2..0 s) is subtracted (the same baseline rule dwave._lda_auc applies).

Usage:  export_e4.py            full export (CPU, ~minutes)
        export_e4.py --smoke    one cell's epoch mapping only (login-node check)
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
from sg_common import (DWAVE_ARMS, EEG_NAMES, EXPORT_LABEL, IDX, RATE, SEALED,  # noqa: E402
                       dwave_tiles, save_npz)

sys.path.insert(0, str(sg.PF))
from dwave import DEN, MERGED_ROOT, load_events  # noqa: E402
from dwave_deep import EPOCH_POST, EPOCH_PRE, _erp_epochs  # noqa: E402

NAME = "erp_topographies.npz"
COHORTS = ("dev", "heldout")
CLASSES = ("nontarget", "target")
N_TIME = EPOCH_PRE + EPOCH_POST                 # 100 samples, -0.2..0.79 s
TIME_S = (np.arange(N_TIME) - EPOCH_PRE) / RATE
P300_WINDOW_S = (0.25, 0.60)
P300_ELECTRODE = "Pz"
MIN_EVENTS = 40                                 # dwave_deep._erp_epochs threshold


def _configs():
    from eeg_scad.cli.run_v43 import configs
    data, folds, _ = configs()
    return data, folds


def _scales(data, folds, dev: list[str]) -> dict[tuple[str, str], np.ndarray]:
    """Per (cohort, participant) fold eeg_scale, built exactly as dwave.fold_contexts:
    dev -> the fold whose test set holds the participant (source root);
    heldout -> fold 99, train = every dev participant, merged sealed root."""
    from eeg_scad.data.counterfactual_pairs import fold_eeg_scale
    out = {}
    cache = {}
    for fold in folds:
        for p in fold["test"]:
            if p in dev:
                key = tuple(fold["train"])
                if key not in cache:
                    cache[key] = fold_eeg_scale(data, fold["train"])
                out[("dev", p)] = cache[key]
    data_h = dict(data)
    data_h["v19_derived_root"] = str(MERGED_ROOT)
    scale_h = fold_eeg_scale(data_h, list(data["participants"]))
    for p in SEALED:
        out[("heldout", p)] = scale_h
    return out


def _check_channel_order(data) -> None:
    """The tile channel axis is the prepared order; assert it equals sg.EEG_NAMES."""
    from eeg_scad.data.counterfactual_pairs import _prepared
    p = data["participants"][0]
    path = _prepared(Path(data["v19_derived_root"]), p, data["sessions"][0], "ERP")
    with np.load(path, allow_pickle=False) as z:
        names = [str(n) for n in z["eeg_names"]]
    if names != EEG_NAMES:
        raise RuntimeError(f"prepared channel order differs from sg_common.EEG_NAMES: {names}")


def _cell_epochs(participant: str, session: str):
    """(meta, labels, tiles) for one ERP cell, or (None, reason, None) when dropped."""
    path = DEN / f"d2_{participant}|{session}|ERP.npz"
    if not path.is_file():
        return None, "no D-wave tile file", None
    tiles = dwave_tiles(participant, session, "ERP")
    meta, labels = _erp_epochs(path, tiles)
    if meta is None:
        n_events = len(load_events(participant, session, "ERP"))
        return None, f"<{MIN_EVENTS} usable events ({n_events} events in the record)", None
    return meta, labels, tiles


def _class_sums(tiles_arm: np.ndarray, meta, labels: np.ndarray):
    """Baseline-corrected epochs -> per-class (sum over epochs, count)."""
    epochs = np.stack([tiles_arm[t][:, o:o + N_TIME] for t, o in meta]).astype(np.float64)
    epochs -= epochs[:, :, :EPOCH_PRE].mean(axis=2, keepdims=True)
    sums = np.stack([epochs[labels == c].sum(axis=0) for c in (0, 1)])   # (2, 46, T)
    counts = np.asarray([(labels == c).sum() for c in (0, 1)], dtype=int)
    return sums, counts


def export() -> Path:
    t0 = time.time()
    data, folds = _configs()
    _check_channel_order(data)
    dev = list(data["participants"])
    if len(dev) != 15 or len(SEALED) != 8:
        raise RuntimeError(f"unexpected cohort sizes dev={len(dev)} sealed={len(SEALED)}")
    sessions = list(data["sessions"])
    cohort_participants = {"dev": dev, "heldout": list(SEALED)}
    scales = _scales(data, folds, dev)
    n_p = max(len(v) for v in cohort_participants.values())
    n_arm, n_ses = len(DWAVE_ARMS), len(sessions)

    participant_avg = np.full((2, n_arm, n_p, 2, 46, N_TIME), np.nan, np.float32)
    coverage = np.zeros((2, n_arm, n_p), bool)
    n_epochs = np.zeros((2, n_arm, n_p, 2), int)
    n_epochs_cell = np.zeros((2, n_p, n_ses, 2), int)
    cell_used = np.zeros((2, n_p, n_ses), bool)
    arm_in_cell = np.zeros((2, n_arm, n_p, n_ses), bool)
    eeg_scale = np.full((2, n_p, 46), np.nan, np.float64)
    participants = np.full((2, n_p), "", dtype="<U8")
    dropped = []

    for ci, cohort in enumerate(COHORTS):
        for pi, p in enumerate(cohort_participants[cohort]):
            participants[ci, pi] = p
            scale = scales[(cohort, p)]
            eeg_scale[ci, pi] = scale
            sums = np.zeros((n_arm, 2, 46, N_TIME))
            counts = np.zeros((n_arm, 2), int)
            for si, ses in enumerate(sessions):
                meta, labels, tiles = _cell_epochs(p, ses)
                if meta is None:
                    dropped.append(f"{cohort}|{p}|{ses}|ERP: {labels}")
                    print(json.dumps({"dropped": f"{p}|{ses}|ERP", "reason": labels}), flush=True)
                    continue
                cell_used[ci, pi, si] = True
                for c in (0, 1):
                    n_epochs_cell[ci, pi, si, c] = int((labels == c).sum())
                for ai, arm in enumerate(DWAVE_ARMS):
                    if arm not in tiles:
                        continue
                    arm_in_cell[ci, ai, pi, si] = True
                    s, n = _class_sums(tiles[arm], meta, labels)
                    sums[ai] += s
                    counts[ai] += n
                del tiles
            used = cell_used[ci, pi]
            if not used.any():
                print(json.dumps({"participant_without_usable_cell": p, "cohort": cohort}),
                      flush=True)
                continue
            for ai, arm in enumerate(DWAVE_ARMS):
                # dwave_deep rule: the arm must be present in every usable cell
                if not arm_in_cell[ci, ai, pi][used].all():
                    continue
                if (counts[ai] == 0).any():
                    continue
                avg = sums[ai] / counts[ai][:, None, None]          # tile units
                participant_avg[ci, ai, pi] = (avg * scale[None, :, None]).astype(np.float32)
                coverage[ci, ai, pi] = True
                n_epochs[ci, ai, pi] = counts[ai]
            print(json.dumps({"cohort": cohort, "participant": p,
                              "cells_used": int(used.sum()),
                              "epochs": counts[DWAVE_ARMS.index("RAW")].tolist(),
                              "arms": [a for ai, a in enumerate(DWAVE_ARMS)
                                       if coverage[ci, ai, pi]],
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    # participant-first grand averages (mean over covered participants)
    grand = np.full((2, n_arm, 2, 46, N_TIME), np.nan, np.float32)
    for ci in range(2):
        for ai in range(n_arm):
            cov = coverage[ci, ai]
            if cov.any():
                grand[ci, ai] = participant_avg[ci, ai, cov].mean(axis=0)

    # P300 peak latency: argmax of the RAW grand-average target-minus-nontarget at Pz
    raw_i, pz = DWAVE_ARMS.index("RAW"), IDX[P300_ELECTRODE]
    win = (TIME_S >= P300_WINDOW_S[0]) & (TIME_S <= P300_WINDOW_S[1])
    peak_index = np.zeros(2, int)
    peak_latency = np.zeros(2, float)
    peak_value = np.zeros(2, float)
    for ci in range(2):
        diff = grand[ci, raw_i, 1, pz] - grand[ci, raw_i, 0, pz]
        idx = int(np.flatnonzero(win)[np.nanargmax(diff[win])])
        peak_index[ci], peak_latency[ci], peak_value[ci] = idx, TIME_S[idx], diff[idx]

    # ---- structural gates (assert), then honest QC prints (no assertion on the science)
    for ci, cohort in enumerate(COHORTS):
        n = len(cohort_participants[cohort])
        covered_any = coverage[ci].any(axis=0)
        assert coverage[ci, raw_i, :n].sum() == covered_any[:n].sum(), "RAW must cover every usable participant"
        for pi in range(n):
            cnt = n_epochs[ci, :, pi][coverage[ci, :, pi]]
            if len(cnt):
                assert (cnt == cnt[0]).all(), f"epoch sets differ across arms for {cohort} p{pi}"
                assert (cnt[0] > 0).all()
        assert np.isfinite(participant_avg[ci][coverage[ci]]).all(), "non-finite ERP"
        assert np.isfinite(grand[ci][coverage[ci].any(axis=1)]).all()
    assert (peak_latency >= P300_WINDOW_S[0]).all() and (peak_latency <= P300_WINDOW_S[1]).all()

    qc = {
        "coverage_per_arm": {cohort: {arm: int(coverage[ci, ai].sum()) for ai, arm in enumerate(DWAVE_ARMS)}
                             for ci, cohort in enumerate(COHORTS)},
        "participants_covered_raw": {cohort: int(coverage[ci, raw_i].sum()) for ci, cohort in enumerate(COHORTS)},
        "dropped_cells": dropped,
        "p300_peak_latency_s": peak_latency.tolist(),
        "p300_peak_value_uV_raw_Pz": peak_value.tolist(),
        "p300_peak_positive": [bool(v > 0) for v in peak_value],
        "epochs_per_participant_raw": {cohort: n_epochs[ci, raw_i, :len(cohort_participants[cohort])].tolist()
                                       for ci, cohort in enumerate(COHORTS)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps({"qc": qc}, indent=1), flush=True)

    path = save_npz(
        NAME,
        grand_average=grand,
        participant_target=participant_avg[:, :, :, 1],
        participant_nontarget=participant_avg[:, :, :, 0],
        coverage=coverage,
        participants=participants,
        n_participants=np.asarray([len(cohort_participants[c]) for c in COHORTS]),
        participants_dev=np.asarray(dev),
        participants_heldout=np.asarray(list(SEALED)),
        cohorts=np.asarray(COHORTS),
        arms=np.asarray(DWAVE_ARMS),
        conditions=np.asarray([EXPORT_LABEL[a] for a in DWAVE_ARMS]),
        classes=np.asarray(CLASSES),
        time_s=TIME_S.astype(np.float64),
        p300_peak_latency_s=peak_latency,
        p300_peak_index=peak_index,
        p300_peak_value_uV=peak_value,
        p300_window_s=np.asarray(P300_WINDOW_S),
        p300_electrode=np.asarray(P300_ELECTRODE),
        n_epochs=n_epochs,
        n_epochs_cell=n_epochs_cell,
        cell_used=cell_used,
        arm_in_cell=arm_in_cell,
        sessions=np.asarray(sessions),
        session_condition=np.asarray([sg.SESSION_CONDITION[s] for s in sessions]),
        dropped_cells=np.asarray(dropped if dropped else [""]),
        eeg_scale=eeg_scale,
        units=np.asarray("microvolt: tile units (eeg / fold eeg_scale) multiplied back by "
                         "eeg_scale[cohort, participant, channel]; divide to recover tile units"),
        baseline_window_s=np.asarray([-EPOCH_PRE / RATE, 0.0]),
        epoch_rule=np.asarray("dwave_deep._erp_epochs: -0.2..0.8 s at 100 Hz, epoch inside one 512-sample "
                              "tile after the 120-s calibration prefix, label 1 = event value 2 (target), "
                              f"cells with < {MIN_EVENTS} usable events dropped"),
        pooling_rule=np.asarray("participant average pools epochs over the participant's usable cells; an arm "
                                "counts only when present in every usable cell (dwave_deep._collect rule); "
                                "grand_average = mean over covered participants (participant-first)"),
        qc=np.asarray(json.dumps(qc)),
    )
    return path


def smoke() -> None:
    """Login-node check: one cell's epoch mapping (reads only 'starts' + the events TSV)."""
    t0 = time.time()
    p, ses = "sub-02", "ses-02"
    path = DEN / f"d2_{p}|{ses}|ERP.npz"
    with np.load(path, allow_pickle=False) as d:
        meta, labels = _erp_epochs(path, {"starts": d["starts"]})
    assert meta is not None and len(meta) == len(labels) >= MIN_EVENTS
    assert set(np.unique(labels)) <= {0, 1} and labels.sum() > 0
    assert all(0 <= o and o + N_TIME <= sg.WINDOW for _, o in meta)
    assert IDX[P300_ELECTRODE] < 32 and len(TIME_S) == N_TIME
    print(json.dumps({"smoke": f"{p}|{ses}|ERP", "epochs": len(meta),
                      "targets": int(labels.sum()), "elapsed_s": round(time.time() - t0, 2)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        export()


if __name__ == "__main__":
    main()
