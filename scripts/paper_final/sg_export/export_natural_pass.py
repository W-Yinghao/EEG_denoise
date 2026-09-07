#!/usr/bin/env python3
"""Natural pass for E2 / E3 / E6 — rerun T5's EXACT 360 natural windows (GPU).

SERVER_EXPORT_FIGURE_ARRAYS_v4.md, items E2/E3/E6, job 1 of 2.  The V44-S1 stage-1
evaluation kept only scalar natural rows, so the corrected windows are re-sampled
here with the frozen fold checkpoints (seed 20261201 only, K = 1 trajectory), on
the windows run_v44._natural_windows yields for every development cell (4 per
cell, 15 participants x 6 cells = 90 cells = 360 windows).

Published diffusion arms, built EXACTLY as run_v44.stage1_eval's natural loop:
  MATCH_gated / POP / NO_A0 / WRONG_gated / SHUFFLED through run_v44._arm_inputs,
  wrong donor = sorted first other participant holding that (session, task) cell,
  shuffled drive = permutation from default_rng(740000 + fold*100 + seed%100)
  consumed in stage1_eval's order (one permutation per ARM_SET arm per window),
  noise from sample_bank_eog with natural_noise_seed(fold, seed), episode_batch 2.
Closed-form arms: RAW (identity) and LINEAR (y - C_gated @ drive).
Reference methods on the same windows, cpu_rows' per-cell fitters verbatim:
  SGEYESUB  rank-2 projection from the 120-s calibration ridge fit (eb120 RAW operator)
  ASR       asrpy calibrated on the 120-s calibration prefix (guarded per cell)
  ICA       mne fastica on the 1-Hz high-passed continuous record, EOG-corr exclusion
            (cpu_rows fits ICA on the whole record, not the prefix; kept as is so
            the rows equal the published cpu_reference_rows)
Failures -> NaN + coverage False.

GATE (assert, non-zero exit): on the first cell of fold 0 the rerun's
MATCH_gated / POP / NO_A0 natural metrics (run_v44._natural_metrics_full, a
superset of _natural_metrics) must equal the STORED V44-S1 rows
(pf_common.stored_stage1_natural_rows(seeds=(20261201,)), matched on
participant/session/task/start/condition) to 1e-9.  WRONG_gated / SHUFFLED are
compared the same way and reported (warning, not fatal).  SGEYESUB / ICA / ASR
are compared to cpu_rows_units/*.json (informational).

Output: one shard per cell under
  /projects/EEG-foundation-model/derived/denoiseNet/sg_natural_pass/<sub>|<ses>|<task>.npz
  corrected (condition, window, 46, 512) float32 in fold-scaled units, drive
  (window, 2, 512) latent, eog_raw (window, 2, 512) microvolt bipolar, starts,
  condition, coverage (condition, window), metrics (condition, window, 7) + names,
  eeg_scale, wrong_donor, lambda, hard_gate, ica_n_excluded, asr_error.
Resume-safe per cell (complete shards are skipped; the shuffle rng is still
advanced for skipped cells so later cells keep the published permutations).

Usage:  export_natural_pass.py            full pass (GPU)
        export_natural_pass.py --smoke    login-node check: imports + stored-row index
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402
from sg_common import CALIB, EEG_NAMES, EXPORT_LABEL, RATE, SEED, WINDOW  # noqa: E402

sys.path.insert(0, str(sg.PF))
from pf_common import OUT, load_model, stored_stage1_natural_rows  # noqa: E402

SHARD_DIR = Path("/projects/EEG-foundation-model/derived/denoiseNet/sg_natural_pass")
GATE_PATH = OUT / "sg_natural_pass_gate.json"
UNIT_DIR = OUT / "cpu_rows_units"
CONDITIONS = ("RAW", "MATCH_gated", "POP", "NO_A0", "LINEAR", "WRONG_gated", "SHUFFLED",
              "ICA", "ASR", "SGEYESUB")
DIFFUSION_ARMS = ("MATCH_gated", "POP", "NO_A0", "WRONG_gated", "SHUFFLED")
GATE_ARMS = ("MATCH_gated", "POP", "NO_A0")
GATE_TOL = 1e-9
REFERENCE_TOL = 1e-6
METRIC_NAMES = ("attenuation_db", "coherence_reduction", "low_eog_observation_retention",
                "psd_distortion_proxy", "output_input_rms", "psd_distortion",
                "covariance_distortion")
CPU_ROWS_CONDITION = {"SGEYESUB": ("sgeyesub", "SGEYESUB_style"), "ICA": ("ica", "ICA_eog_corr"),
                      "ASR": ("asr", "ASR")}
def label(condition: str) -> str:
    """sg_common.EXPORT_LABEL keyed by the D-wave arm name (MATCH_gated -> MATCH)."""
    return EXPORT_LABEL[condition] if condition in EXPORT_LABEL else EXPORT_LABEL[condition.replace("_gated", "")]


assert all(label(c) for c in CONDITIONS)


def shard_path(key) -> Path:
    return SHARD_DIR / ("|".join(key) + ".npz")


def shard_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as d:
            ok = ("corrected" in d.files and "condition" in d.files
                  and tuple(str(c) for c in d["condition"]) == CONDITIONS
                  and d["corrected"].shape == (len(CONDITIONS), 4, 46, WINDOW))
    except Exception:
        ok = False
    return bool(ok)


def cell_order(fold, data, assets):
    """Cells in the exact order of run_v44.stage1_eval's natural loop."""
    return [(p, s, t) for p, s, t in itertools.product(fold["test"], data["sessions"], data["tasks"])
            if (p, s, t) in assets]


def wrong_donor(registry30, assets, key) -> str:
    participant, session, task = key
    return sorted(candidate for candidate in {k[0] for k in registry30.cells}
                  if candidate != participant and (candidate, session, task) in assets)[0]


def shuffled_permutations(rng, arm_set, n_windows: int, n_samples: int):
    """Advance the natural shuffle rng exactly as stage1_eval does (one permutation per
    ARM_SET arm per window) and return the permutations the SHUFFLED arm consumed."""
    kept = None
    for arm in arm_set:
        perms = [rng.permutation(n_samples) for _ in range(n_windows)]
        if arm == "SHUFFLED":
            kept = perms
    assert kept is not None
    return kept


def stored_index():
    rows = stored_stage1_natural_rows(seeds=(SEED,))
    index = {(r["participant"], r["session"], r["task"], int(r["start"]), r["condition"]): r
             for r in rows}
    assert len(index) == len(rows), "duplicate stored natural rows"
    return index


def reference_index():
    """cpu_rows_units rows keyed by (participant, start) -> list of rows (no session key)."""
    out = {}
    for arm, (unit, condition) in CPU_ROWS_CONDITION.items():
        path = UNIT_DIR / f"{unit}.json"
        if not path.is_file():
            continue
        rows = [r for r in json.loads(path.read_text())["natural_rows"] if r["condition"] == condition]
        by = {}
        for r in rows:
            by.setdefault((r["participant"], int(r["start"])), []).append(r)
        out[arm] = by
    return out


def compare_metrics(mine: dict, stored: dict, names) -> float:
    return max(abs(float(mine[n]) - float(stored[n])) for n in names if n in stored)


# ---------------------------------------------------------------- reference fitters (cpu_rows verbatim)

def ica_clean_record(eeg: np.ndarray, eog: np.ndarray):
    """cpu_rows.ica: fit on the 1-Hz high-passed continuous record, EOG-correlation
    exclusion, applied to the unfiltered record. Returns (cleaned eeg 46xT raw units, n_excluded)."""
    import mne
    mne.set_log_level("ERROR")
    length = min(eeg.shape[1], eog.shape[1])
    stacked = np.vstack([eeg[:, :length], eog[:, :length]])
    eeg_names = [f"E{i:02d}" for i in range(46)]
    info = mne.create_info(eeg_names + ["VEOG", "HEOG"], float(RATE), ["eeg"] * 46 + ["eog", "eog"])
    raw = mne.io.RawArray(stacked, info, verbose="ERROR")
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    decomposition = mne.preprocessing.ICA(n_components=0.999999, method="fastica",
                                          random_state=SEED, max_iter=1000)
    decomposition.fit(raw_filt, picks="eeg")
    bads, _ = decomposition.find_bads_eog(raw_filt)
    decomposition.exclude = sorted(set(bads))
    cleaned = decomposition.apply(raw.copy())
    return cleaned.get_data(picks="eeg"), len(decomposition.exclude)


def asr_cleaner(calib: np.ndarray):
    """cpu_rows.asr: asrpy calibrated on the 120-s prefix (fold-scaled units).
    Returns (cleaner or None, error string)."""
    try:
        import asrpy
    except ImportError as error:
        return None, f"asrpy import failed: {error}"[:200]
    import mne
    mne.set_log_level("ERROR")
    try:  # asrpy has a data-dependent reshape bug in calibration
        cleaner = asrpy.ASR(sfreq=float(RATE))
        cleaner.fit(_to_raw(calib))
        return cleaner, ""
    except Exception as error:
        return None, str(error)[:200]


def _to_raw(array):
    import mne
    eeg_names = [f"E{i:02d}" for i in range(46)]
    info = mne.create_info(eeg_names, float(RATE), ["eeg"] * 46)
    return mne.io.RawArray(np.asarray(array, np.float64), info, verbose="ERROR")


def sgeyesub_projector(eb120, key) -> np.ndarray:
    c_full = eb120.operator(*key, "RAW")
    q, _ = np.linalg.qr(c_full)
    return q @ q.T


# ---------------------------------------------------------------- one cell

def process_cell(up, fold_id, fold, data, registry30, eb120, assets, model, schedule, device,
                 key, perms):
    t0 = time.time()
    participant, session, task = key
    wrong = wrong_donor(registry30, assets, key)
    wrong_key = (wrong, session, task)
    windows = list(up._natural_windows(registry30, data, key))
    assert len(windows) == 4 and len(perms) == 4, (key, len(windows))
    starts = np.asarray([w[0] for w in windows], np.int64)
    y_stack = np.stack([w[1] for w in windows])                    # (4, 46, 512) fold-scaled
    drives = np.stack([w[2] for w in windows])                     # (4, 2, 512) latent
    eeg_raw, eye, names = registry30._load(*key)
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
    eog_full = bipolar_eog(eye, names)
    eog_raw = np.stack([eog_full[:, s:s + WINDOW] for s in starts])  # (4, 2, 512) microvolt
    scale = registry30.eeg_scale[:, None]
    cell = registry30.cells[key]
    # sanity: the windows are the record slices they claim to be
    assert np.allclose(np.stack([eeg_raw[:, s:s + WINDOW] / scale for s in starts]), y_stack)
    assert np.allclose(np.stack([(eog_full[:, s:s + WINDOW] - cell.eog_center[:, None])
                                 / cell.eog_scale[:, None] for s in starts]), drives)

    n_cond = len(CONDITIONS)
    corrected = np.full((n_cond, 4, 46, WINDOW), np.nan, np.float32)
    coverage = np.zeros((n_cond, 4), bool)
    metrics = np.full((n_cond, 4, len(METRIC_NAMES)), np.nan, np.float64)
    info = {"ica_n_excluded": -1, "asr_error": ""}

    def store(arm, outputs):
        c = CONDITIONS.index(arm)
        for i, (y, drive, out) in enumerate(zip(y_stack, drives, outputs)):
            if out is None or not np.isfinite(out).all():
                continue
            out = np.asarray(out, np.float64)
            corrected[c, i] = out.astype(np.float32)
            coverage[c, i] = True
            m = up._natural_metrics_full(y, drive, out)
            metrics[c, i] = [m[n] for n in METRIC_NAMES]

    # closed-form arms
    store("RAW", [y for y in y_stack])
    store("LINEAR", [y - assets[key]["C_gated"] @ drive for y, drive in zip(y_stack, drives)])
    # diffusion arms, exactly as stage1_eval's natural loop
    for arm in DIFFUSION_ARMS:
        a0_stack, sig_stack = [], []
        for drive, perm in zip(drives, perms):
            shuffled_drive = drive[:, perm]
            a0, sig = up._arm_inputs(assets, key, wrong_key, drive, shuffled_drive)[arm]
            a0_stack.append(a0)
            sig_stack.append(sig)
        output = up.sample_bank_eog(model, schedule, y_stack, np.stack(a0_stack),
                                    np.stack(sig_stack), device, up.natural_noise_seed(fold_id, SEED))
        if not np.isfinite(output).all():
            raise FloatingPointError(f"nonfinite natural output {key} {arm}")
        store(arm, list(output))
    t_gpu = time.time() - t0
    # reference methods (cpu_rows fitters)
    projector = sgeyesub_projector(eb120, key)
    store("SGEYESUB", [y - projector @ y for y in y_stack])
    try:
        cleaned, n_excluded = ica_clean_record(eeg_raw, eog_full)
        info["ica_n_excluded"] = int(n_excluded)
        store("ICA", [cleaned[:, s:s + WINDOW] / scale for s in starts])
    except Exception as error:
        info["ica_error"] = str(error)[:200]
        print(json.dumps({"ica_failed": "|".join(key), "error": info["ica_error"]}), flush=True)
    cleaner, asr_error = asr_cleaner(eeg_raw[:, :CALIB] / scale)
    info["asr_error"] = asr_error
    if cleaner is not None:
        outs = []
        for y in y_stack:
            try:
                outs.append(cleaner.transform(_to_raw(y)).get_data())
            except Exception as error:
                info["asr_error"] = f"transform: {str(error)[:180]}"
                outs.append(None)
        store("ASR", outs)
    if info["asr_error"]:
        print(json.dumps({"asr_failed": "|".join(key), "error": info["asr_error"]}), flush=True)

    eb_cell = eb120.cells[key]
    shard = dict(
        corrected=corrected, coverage=coverage, condition=np.asarray(CONDITIONS),
        condition_label=np.asarray([label(c) for c in CONDITIONS]),
        metrics=metrics, metric_names=np.asarray(METRIC_NAMES),
        drive=drives.astype(np.float32), eog_raw=eog_raw.astype(np.float32),
        eog_names=np.asarray(["VEOG", "HEOG"]), starts=starts,
        participant=np.asarray(participant), session=np.asarray(session), task=np.asarray(task),
        movement=np.asarray(sg.SESSION_CONDITION[session]), fold=np.asarray(int(fold_id)),
        seed=np.asarray(int(SEED)), natural_noise_seed=np.asarray(int(up.natural_noise_seed(fold_id, SEED))),
        k_trajectories=np.asarray(1), wrong_donor=np.asarray(wrong),
        eeg_scale=np.asarray(registry30.eeg_scale, np.float64), eeg_names=np.asarray(EEG_NAMES),
        **{"lambda": np.asarray(float(eb_cell.lam))}, hard_gate=np.asarray(int(eb_cell.hard_gate)),
        ica_n_excluded=np.asarray(int(info["ica_n_excluded"])), asr_error=np.asarray(info["asr_error"]),
        ica_error=np.asarray(info.get("ica_error", "")),
        units=np.asarray("corrected EEG in fold-scaled units (prepared microvolt / eeg_scale per "
                         "channel); drive = latent (centred/scaled) bipolar EOG; eog_raw = "
                         "[VEOGU-VEOGL, HEOGL-HEOGR] microvolt"),
        condition_note=np.asarray(sg.CONDITION_NOTE),
    )
    path = shard_path(key)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **shard)
    os.replace(tmp, path)
    print(json.dumps({"cell": "|".join(key), "wrong_donor": wrong, "starts": starts.tolist(),
                      "coverage": {c: int(coverage[i].sum()) for i, c in enumerate(CONDITIONS)},
                      "ica_excluded": info["ica_n_excluded"], "asr_error": info["asr_error"][:60],
                      "lambda": round(float(eb_cell.lam), 4), "hard_gate": int(eb_cell.hard_gate),
                      "t_gpu_s": round(t_gpu, 1), "t_s": round(time.time() - t0, 1)}), flush=True)
    return shard


# ---------------------------------------------------------------- gate

def gate(shard, stored, reference) -> dict:
    key = (str(shard["participant"]), str(shard["session"]), str(shard["task"]))
    report = {"cell": "|".join(key), "tolerance": GATE_TOL, "arms": {}, "reference": {}}
    for arm in DIFFUSION_ARMS:
        c = CONDITIONS.index(arm)
        worst = 0.0
        for i, start in enumerate(shard["starts"].tolist()):
            row = stored.get((*key, int(start), arm))
            assert row is not None, f"no stored V44-S1 natural row for {key} start {start} {arm}"
            mine = dict(zip(METRIC_NAMES, shard["metrics"][c, i]))
            worst = max(worst, compare_metrics(mine, row, METRIC_NAMES))
        report["arms"][arm] = {"max_abs_diff": float(worst), "pass": bool(worst <= GATE_TOL)}
    for arm, by in reference.items():
        c = CONDITIONS.index(arm)
        worst, n = 0.0, 0
        for i, start in enumerate(shard["starts"].tolist()):
            if not shard["coverage"][c, i]:
                continue
            rows = by.get((key[0], int(start)), [])
            if not rows:
                continue
            mine = dict(zip(METRIC_NAMES, shard["metrics"][c, i]))
            names = [n_ for n_ in METRIC_NAMES if n_ in rows[0]]
            worst = max(worst, min(compare_metrics(mine, r, names) for r in rows))
            n += 1
        report["reference"][arm] = {"max_abs_diff_best_match": float(worst), "windows_compared": n,
                                    "within_1e-6": bool(worst <= REFERENCE_TOL) if n else None}
    report["pass"] = all(report["arms"][a]["pass"] for a in GATE_ARMS)
    return report


# ---------------------------------------------------------------- main

def main() -> None:
    import torch
    torch.backends.cudnn.allow_tf32 = False          # no-op on V100 (the published GPU), guards Ampere+
    torch.backends.cuda.matmul.allow_tf32 = False
    from eeg_scad.cli import run_v44 as up
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data, folds, _ = configs()
    stored = stored_index()
    reference = reference_index()
    device = torch.device("cuda")
    print(json.dumps({"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "stored_rows": len(stored), "reference_arms": sorted(reference),
                      "shard_dir": str(SHARD_DIR)}), flush=True)
    gate_report = None
    done, skipped = [], []
    for fold_id, fold in enumerate(folds):
        assert int(fold["fold"]) == fold_id
        _, registry30, eb120 = sg.registry(fold)
        assets = up._gated_assets(registry30, eb120)
        model = load_model(fold_id, device, SEED)
        schedule = LinearX0Schedule().to(device)
        rng = np.random.default_rng(740000 + fold_id * 100 + SEED % 100)
        keys = cell_order(fold, data, assets)
        print(json.dumps({"fold": fold_id, "test": sorted(fold["test"]), "cells": len(keys),
                          "t": round(time.time() - t0, 1)}), flush=True)
        for idx, key in enumerate(keys):
            perms = shuffled_permutations(rng, up.ARM_SET, 4, WINDOW)
            is_gate_cell = fold_id == 0 and idx == 0
            if shard_complete(shard_path(key)) and not is_gate_cell:
                skipped.append("|".join(key))
                continue
            shard = process_cell(up, fold_id, fold, data, registry30, eb120, assets, model,
                                 schedule, device, key, perms)
            done.append("|".join(key))
            if is_gate_cell:
                gate_report = gate(shard, stored, reference)
                gate_report["device"] = torch.cuda.get_device_name(0)
                GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                GATE_PATH.write_text(json.dumps(gate_report, indent=1) + "\n")
                (SHARD_DIR / "gate.json").write_text(json.dumps(gate_report, indent=1) + "\n")
                print(json.dumps({"GATE": gate_report}), flush=True)
                if not gate_report["pass"]:
                    print("GATE FAILED: the rerun is not bit-identical to the stored V44-S1 "
                          "natural pass on MATCH_gated/POP/NO_A0 — stopping before the fleet",
                          file=sys.stderr, flush=True)
                    sys.exit(2)
                for arm in ("WRONG_gated", "SHUFFLED"):
                    if not gate_report["arms"][arm]["pass"]:
                        print(json.dumps({"WARNING": f"{arm} not bit-identical to the stored pass",
                                          "max_abs_diff": gate_report["arms"][arm]["max_abs_diff"]}),
                              flush=True)
        del model, assets, registry30, eb120
        torch.cuda.empty_cache()
    shards = sorted(p.name for p in SHARD_DIR.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    manifest = {"cells": len(shards), "done_this_run": done, "skipped_complete": skipped,
                "gate": gate_report, "conditions": list(CONDITIONS), "seed": SEED,
                "k_trajectories": 1, "elapsed_s": round(time.time() - t0, 1)}
    (SHARD_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    assert len(shards) == 90, f"expected 90 cell shards, found {len(shards)}"
    print(json.dumps({"cells": len(shards), "done": len(done), "skipped": len(skipped),
                      "elapsed_s": manifest["elapsed_s"]}), flush=True)


def smoke() -> None:
    """Login-node check (<30 s, one JSON per fold): imports, stored-row index, first cell."""
    from eeg_scad.cli import run_v44 as up
    from eeg_scad.cli.run_v43 import configs
    data, folds, _ = configs()
    stored = stored_index()
    reference = reference_index()
    first = [k for k in stored if k[-1] == "MATCH_gated"][0]
    keys0 = [(p, s, t) for p, s, t in itertools.product(folds[0]["test"], data["sessions"], data["tasks"])]
    rng = np.random.default_rng(740000 + 0 * 100 + SEED % 100)
    perms = shuffled_permutations(rng, up.ARM_SET, 4, WINDOW)
    assert up.ARM_SET.index("SHUFFLED") == 4 and len(perms) == 4 and sorted(perms[0]) == list(range(WINDOW))
    print(json.dumps({"smoke": True, "stored_rows": len(stored), "first_stored_key": list(first),
                      "fold0_first_cell": "|".join(keys0[0]), "arm_set": list(up.ARM_SET),
                      "reference_arms": {a: len(b) for a, b in reference.items()},
                      "qnatural_start": int(data["qnatural_start"]),
                      "natural_noise_seed_fold0": up.natural_noise_seed(0, SEED),
                      "shard_dir_exists": SHARD_DIR.is_dir()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke() if args.smoke else main()
