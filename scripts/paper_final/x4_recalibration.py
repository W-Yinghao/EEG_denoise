#!/usr/bin/env python3
"""WAVE-6 E4 — does propagation drift actually invalidate the old calibration?

Frozen design: reports/prereg_wave6_propagation_FROZEN.md section 5.

The existing T4 analysis shows the operator moves away from its first estimate
while the paired gain does not fall in step.  That is compatible with three
different stories, and they are only separable by putting the old and the new
calibration on the SAME evaluation window.  This runner does that:

  A0_RAW   ridge fit on the record's first 120 s (the calibration segment)
  AT_RAW   ridge fit on the 120 s immediately preceding the evaluation window
           (never overlapping it)
  POP      the fold's population operator
  OWN_EB   the deployed EB-shrunk 0-120 s operator, as a reference point

and records, per window, how far the operator moved (||A_t - A_0||) and how much
of that movement the current eye movement actually activates
(||(A_t - A_0) e_q||_F) next to the restoration difference it produced.

modes: probe (QC gates, exits non-zero on failure) | run (one fold+seed unit)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from pf_common import OUT

V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
WAVE6 = OUT / "wave6"
UNITS = WAVE6 / "e4_units"
SEEDS = (20261201, 20261202, 20261203)
CALIB_SPAN_S = 120
ARMS = ("A0_RAW", "AT_RAW", "POP", "OWN_EB")


def _v44():
    if str(V44_SRC) not in sys.path:
        sys.path.insert(0, str(V44_SRC))
    from eeg_scad.cli import run_v44 as up
    return up


def _fit(registry30, eeg, eog, start, span):
    from eeg_scad.data.artifact_transfer_v41r import ridge_transfer
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale
    seg = eog[:, start:start + span]
    center, scale = robust_center_scale(seg)
    latent = (seg - center[:, None]) / scale[:, None]
    scaled = eeg[:, start:start + span] / registry30.eeg_scale[:, None]
    operator, _ = ridge_transfer(scaled, latent, registry30.ridge_ratio)
    return operator


def _setup(fold_id: int, seed: int):
    import torch
    up = _v44()
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    data, folds, _ = configs()
    fold = folds[fold_id]
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = up._gated_assets(registry30, eb120)
    source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                         / "train_curve.json").read_text())
    device = torch.device("cuda")
    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    return up, data, fold, registry30, assets, model, LinearX0Schedule().to(device), device


def _cell_rows(up, fold_id, seed, data, registry30, assets, model, schedule,
               device, key, limit_windows=None):
    """One cell: all natural windows x the four calibration arms."""
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
    rate = int(data.get("sampling_rate", 100))
    span = CALIB_SPAN_S * rate
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)

    windows = list(up._natural_windows(registry30, data, key))
    if limit_windows:
        windows = windows[:limit_windows]
    if not windows:
        return []
    a0_raw = _fit(registry30, eeg, eog, 0, span)

    operators, meta = {}, []
    for start, _, _ in windows:
        recent_start = start - span
        if recent_start < 0:                       # no non-overlapping history
            meta.append(None)
            continue
        meta.append(_fit(registry30, eeg, eog, recent_start, span))

    y_stack = np.stack([w[1] for w in windows])
    drives = np.stack([w[2] for w in windows])
    starts = [w[0] for w in windows]
    keep = [i for i, m in enumerate(meta) if m is not None]
    if not keep:
        return []
    y_stack, drives = y_stack[keep], drives[keep]
    starts = [starts[i] for i in keep]
    at_raw = [meta[i] for i in keep]

    base_norm = max(np.linalg.norm(a0_raw), 1e-12)
    geometry = []
    for i, drive in enumerate(drives):
        delta = at_raw[i] - a0_raw
        geometry.append({
            "operator_displacement": float(np.linalg.norm(delta) / base_norm),
            "activated_displacement": float(np.linalg.norm(delta @ drive)
                                            / max(np.linalg.norm(a0_raw @ drive), 1e-12)),
            "eog_rms": float(np.sqrt(np.mean(drive ** 2))),
            "veog_rms": float(np.sqrt(np.mean(drive[0] ** 2))),
            "heog_rms": float(np.sqrt(np.mean(drive[1] ** 2))),
            "elapsed_s": starts[i] / rate,
        })

    rows = []
    for arm in ARMS:
        a0 = []
        for i, drive in enumerate(drives):
            operator = {"A0_RAW": a0_raw, "AT_RAW": at_raw[i],
                        "POP": assets[key]["C0"], "OWN_EB": assets[key]["C_gated"]}[arm]
            a0.append(operator @ drive)
        sig = np.stack([assets[key]["sig_pop"]] * len(drives))
        output = up.sample_bank_eog(model, schedule, y_stack, np.stack(a0), sig,
                                    device, up.natural_noise_seed(fold_id, seed))
        for i, (y, drive, prediction) in enumerate(zip(y_stack, drives, output)):
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite E4 {arm} {key} {i}")
            rows.append({"fold": fold_id, "seed": seed, "participant": key[0],
                         "session": key[1], "task": key[2], "start": starts[i],
                         "arm": arm, **geometry[i],
                         **up._natural_metrics(y, drive, y - prediction)})
    return rows


def probe(fold_id: int = 0, seed: int = SEEDS[0]) -> None:
    up, data, fold, registry30, assets, model, schedule, device = _setup(fold_id, seed)
    key = next(k for k in sorted(assets) if k[0] in fold["test"])
    rows = _cell_rows(up, fold_id, seed, data, registry30, assets, model, schedule,
                      device, key, limit_windows=2)
    rows2 = _cell_rows(up, fold_id, seed, data, registry30, assets, model, schedule,
                       device, key, limit_windows=2)
    checks: dict[str, object] = {"fold": fold_id, "seed": seed, "cell": "|".join(key),
                                 "n_rows": len(rows)}

    by = lambda rs, arm: [r["attenuation_db"] for r in rs if r["arm"] == arm]
    p1 = bool(rows) and all(a["attenuation_db"] == b["attenuation_db"]
                            for a, b in zip(rows, rows2))
    checks["P1_determinism"] = p1

    a0v, atv = by(rows, "A0_RAW"), by(rows, "AT_RAW")
    delta = max(abs(a - b) for a, b in zip(a0v, atv)) if a0v and atv else 0.0
    checks["P2_max_abs_delta_A0_vs_AT"] = float(delta)
    p2 = delta > 1e-9

    # the recent calibration window must never touch the evaluation window
    rate = int(data.get("sampling_rate", 100))
    p3 = all(r["start"] - CALIB_SPAN_S * rate >= 0 for r in rows)
    checks["P3_no_calibration_overlap"] = bool(p3)

    disp = [r["operator_displacement"] for r in rows]
    act = [r["activated_displacement"] for r in rows]
    p4 = bool(disp) and all(np.isfinite(disp)) and all(np.isfinite(act)) and max(disp) > 0
    checks["P4_geometry_sanity"] = p4
    checks["P4_displacement_range"] = [float(min(disp)), float(max(disp))] if disp else []

    own = by(rows, "OWN_EB")
    p5 = bool(own) and all(np.isfinite(own))
    checks["P5_own_arm_finite"] = p5
    checks["all_gates_pass"] = bool(p1 and p2 and p3 and p4 and p5)
    WAVE6.mkdir(parents=True, exist_ok=True)
    (WAVE6 / "e4_probe.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=1))
    if not checks["all_gates_pass"]:
        raise SystemExit("WAVE6 E4 PROBE GATE FAILED — fleet must not launch")


def run(fold_id: int, seed: int) -> None:
    UNITS.mkdir(parents=True, exist_ok=True)
    out = UNITS / f"fold_{fold_id}_seed_{seed}.json"
    if out.is_file() and json.loads(out.read_text()).get("complete"):
        print(json.dumps({"skipped": str(out)}))
        return
    up, data, fold, registry30, assets, model, schedule, device = _setup(fold_id, seed)
    rows = []
    for participant, session, task in itertools.product(
            sorted(fold["test"]), data["sessions"], data["tasks"]):
        key = (participant, session, task)
        if key not in assets:
            continue
        rows.extend(_cell_rows(up, fold_id, seed, data, registry30, assets, model,
                               schedule, device, key))
        print(json.dumps({"cell": "|".join(key), "rows": len(rows)}), flush=True)
    keys = {(r["participant"], r["session"], r["task"], r["start"], r["arm"]) for r in rows}
    out.write_text(json.dumps({"fold": fold_id, "seed": seed, "arms": list(ARMS),
                               "n_rows": len(rows), "n_unique_keys": len(keys),
                               "complete": True, "rows": rows,
                               "frozen": "reports/prereg_wave6_propagation_FROZEN.md#5"},
                              sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "n_rows": len(rows),
                      "n_unique_keys": len(keys)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "run"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    args = parser.parse_args()
    (probe if args.mode == "probe" else run)(args.fold, args.seed)


if __name__ == "__main__":
    main()
