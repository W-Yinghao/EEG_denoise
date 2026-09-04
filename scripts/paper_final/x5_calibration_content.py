#!/usr/bin/env python3
"""WAVE-6 E5 — does WHAT the calibration contains decide what it is good for?

Frozen design: reports/prereg_wave6_propagation_FROZEN.md amendment W6-1b.

Inside each cell's pre-evaluation region the EOG is cut into 2-s blocks, blocks
are ranked by v / (v + h), and three EQUAL-DURATION calibration sets are
assembled - V-heavy, H-heavy and balanced - with total EOG energy equalised as
far as the available blocks allow.  Three independent draws guard against a
lucky segment.  Every set is fitted with the same estimator and shrunk with the
SAME lambda (the cell's deployed value), so no composition can be advantaged by
silently falling back to a different rule.  All of them then restore the SAME
later natural windows.

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
UNITS = WAVE6 / "e5_units"
SEEDS = (20261201, 20261202, 20261203)
BLOCK_S = 2               # calibration block length
SET_SECONDS = 60          # every composition gets exactly this much data
DRAWS = 3
COMPOSITIONS = ("VHEAVY", "HHEAVY", "BALANCED")


def _v44():
    if str(V44_SRC) not in sys.path:
        sys.path.insert(0, str(V44_SRC))
    from eeg_scad.cli import run_v44 as up
    return up


def _blocks(eog, limit, rate):
    """(index, v_rms, h_rms, ratio) for every 2-s block before `limit`."""
    span = BLOCK_S * rate
    out = []
    for start in range(0, limit - span + 1, span):
        seg = eog[:, start:start + span]
        v = float(np.sqrt(np.mean(seg[0] ** 2)))
        h = float(np.sqrt(np.mean(seg[1] ** 2)))
        out.append({"start": start, "v": v, "h": h,
                    "ratio": v / max(v + h, 1e-12),
                    "energy": float(np.sqrt(np.mean(seg ** 2)))})
    return out


def _compose(blocks, kind, draw, rate, rng):
    """Pick SET_SECONDS worth of blocks of the requested composition.

    Energy matching: candidates are ordered by composition preference, then the
    selection walks that order keeping the running mean energy as close as it can
    to the cell's overall mean - so a composition cannot win merely by carrying
    more EOG energy.
    """
    need = SET_SECONDS // BLOCK_S
    if len(blocks) < need:
        return None, {}
    target = float(np.mean([b["energy"] for b in blocks]))
    order = sorted(blocks, key=lambda b: b["ratio"])
    # DEFECT W6-D1 (2026-09-05): the first implementation varied draws by rotating
    # the whole ranking, which moved draws 1 and 2 into the middle of the ranking
    # and collapsed every composition onto the cell average (separation +0.015 and
    # +0.001 versus +0.368 for draw 0).  Draws must vary WITHIN the composition's
    # own region, never across it.
    width = min(len(order), max(need * 2, need + 1))
    if kind == "VHEAVY":
        region = order[::-1][:width]           # the most vertical blocks
    elif kind == "HHEAVY":
        region = order[:width]                 # the most horizontal blocks
    else:                                      # BALANCED: stratified across the rank
        idx = np.linspace(0, len(order) - 1, width).astype(int)
        region = [order[i] for i in dict.fromkeys(idx)]
    if draw:
        pick = rng.permutation(len(region))[:need]
        pool = [region[i] for i in sorted(pick)]
        pool += [b for b in region if b not in pool]
    else:
        pool = list(region)
    chosen, running = [], []
    for b in pool:
        if len(chosen) >= need:
            break
        trial = running + [b["energy"]]
        # keep a block unless it pushes the running mean further from target and
        # there is still enough pool left to be choosy
        if (len(chosen) < need - (len(pool) - len(chosen)) // 2
                or abs(np.mean(trial) - target) <= abs(np.mean(running) - target)
                or not running):
            chosen.append(b)
            running = trial
    for b in pool:                          # top up if the filter was too strict
        if len(chosen) >= need:
            break
        if b not in chosen:
            chosen.append(b)
    chosen = chosen[:need]
    stats = {"n_blocks": len(chosen),
             "mean_ratio": float(np.mean([b["ratio"] for b in chosen])),
             "mean_energy": float(np.mean([b["energy"] for b in chosen])),
             "cell_mean_energy": target,
             "seconds": len(chosen) * BLOCK_S}
    return chosen, stats


def _fit_composition(registry30, eeg, eog, chosen, rate, lam, pop):
    """Same estimator for every composition; the SAME lambda for every one."""
    from eeg_scad.data.artifact_transfer_v41r import ridge_transfer
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale
    span = BLOCK_S * rate
    seg_eog = np.concatenate([eog[:, b["start"]:b["start"] + span] for b in chosen], axis=1)
    seg_eeg = np.concatenate([eeg[:, b["start"]:b["start"] + span] for b in chosen], axis=1)
    center, scale = robust_center_scale(seg_eog)
    latent = (seg_eog - center[:, None]) / scale[:, None]
    scaled = seg_eeg / registry30.eeg_scale[:, None]
    raw, _ = ridge_transfer(scaled, latent, registry30.ridge_ratio)
    return pop + lam * (raw - pop), raw


def _cell_rows(up, fold_id, seed, data, registry30, eb120, assets, model, schedule,
               device, key, limit_windows=None):
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
    rate = int(data.get("sampling_rate", 100))
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    windows = list(up._natural_windows(registry30, data, key))
    if limit_windows:
        windows = windows[:limit_windows]
    if not windows:
        return [], {}
    limit = min(int(data["qnatural_start"]), eeg.shape[1], eog.shape[1])
    blocks = _blocks(eog, limit, rate)
    if len(blocks) < SET_SECONDS // BLOCK_S:
        return [], {}

    lam = float(eb120.cells[key].lam)
    pop = registry30.population_transfer[key[1:]]
    rng = np.random.default_rng(abs(hash(("e5",) + key)) % (2 ** 31))
    operators, composition_meta = {}, {}
    for kind, draw in itertools.product(COMPOSITIONS, range(DRAWS)):
        chosen, stats = _compose(blocks, kind, draw, rate, rng)
        if chosen is None:
            continue
        arm = f"{kind}_{draw}"
        operators[arm], _ = _fit_composition(registry30, eeg, eog, chosen, rate, lam, pop)
        composition_meta[arm] = stats
    operators["OWN_EB"] = assets[key]["C_gated"]

    y_stack = np.stack([w[1] for w in windows])
    drives = np.stack([w[2] for w in windows])
    starts = [w[0] for w in windows]
    activity = [{"veog_rms": float(np.sqrt(np.mean(d[0] ** 2))),
                 "heog_rms": float(np.sqrt(np.mean(d[1] ** 2))),
                 "eog_rms": float(np.sqrt(np.mean(d ** 2)))} for d in drives]

    rows = []
    for arm, operator in operators.items():
        a0 = np.stack([operator @ d for d in drives])
        sig = np.stack([assets[key]["sig_pop"]] * len(drives))
        output = up.sample_bank_eog(model, schedule, y_stack, a0, sig, device,
                                    up.natural_noise_seed(fold_id, seed))
        for i, (y, drive, prediction) in enumerate(zip(y_stack, drives, output)):
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite E5 {arm} {key} {i}")
            rows.append({"fold": fold_id, "seed": seed, "participant": key[0],
                         "session": key[1], "task": key[2], "start": starts[i],
                         "arm": arm, "lam": lam, **activity[i],
                         **composition_meta.get(arm, {}),
                         **up._natural_metrics(y, drive, y - prediction)})
    return rows, composition_meta


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
    return (up, data, fold, registry30, eb120, assets, model,
            LinearX0Schedule().to(device), device)


def probe(fold_id: int = 0, seed: int = SEEDS[0]) -> None:
    up, data, fold, registry30, eb120, assets, model, schedule, device = _setup(fold_id, seed)
    key = next(k for k in sorted(assets) if k[0] in fold["test"])
    rows, meta = _cell_rows(up, fold_id, seed, data, registry30, eb120, assets,
                            model, schedule, device, key, limit_windows=2)
    rows2, _ = _cell_rows(up, fold_id, seed, data, registry30, eb120, assets,
                          model, schedule, device, key, limit_windows=2)
    checks: dict[str, object] = {"fold": fold_id, "seed": seed,
                                 "cell": "|".join(key), "n_rows": len(rows),
                                 "compositions": meta}
    p1 = bool(rows) and all(a["attenuation_db"] == b["attenuation_db"]
                            for a, b in zip(rows, rows2))
    checks["P1_determinism"] = p1

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["attenuation_db"])
    vh = [abs(a - b) for a, b in zip(by_arm.get("VHEAVY_0", []), by_arm.get("HHEAVY_0", []))]
    checks["P2_max_abs_delta_V_vs_H"] = float(max(vh)) if vh else 0.0
    p2 = bool(vh) and max(vh) > 1e-9

    # every composition must have the SAME duration and a comparable energy
    seconds = {m["seconds"] for m in meta.values()}
    ratios = {a: m["mean_ratio"] for a, m in meta.items()}
    p3 = len(seconds) == 1 and seconds.pop() == SET_SECONDS
    checks["P3_equal_duration"] = bool(p3)
    checks["P3_mean_ratio_by_arm"] = ratios
    energies = [m["mean_energy"] for m in meta.values()]
    target = next(iter(meta.values()))["cell_mean_energy"] if meta else 1.0
    spread = float(max(energies) - min(energies)) / max(target, 1e-12) if energies else 1.0
    checks["P4_energy_spread_over_cell_mean"] = spread

    # every DRAW must separate on its own — a single separated draw hiding two
    # collapsed ones is exactly defect W6-D1
    per_draw = {}
    for draw in range(DRAWS):
        v = ratios.get(f"VHEAVY_{draw}")
        h = ratios.get(f"HHEAVY_{draw}")
        per_draw[draw] = (v - h) if (v is not None and h is not None) else None
    checks["P4_separation_by_draw"] = per_draw
    values = [x for x in per_draw.values() if x is not None]
    p4 = bool(values) and len(values) == DRAWS and min(values) > 0.05
    checks["P4_composition_separated"] = p4
    p5 = "OWN_EB" in by_arm and all(np.isfinite(by_arm["OWN_EB"]))
    checks["P5_reference_arm_finite"] = p5
    checks["all_gates_pass"] = bool(p1 and p2 and p3 and p4 and p5)
    WAVE6.mkdir(parents=True, exist_ok=True)
    (WAVE6 / "e5_probe.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=1))
    if not checks["all_gates_pass"]:
        raise SystemExit("WAVE6 E5 PROBE GATE FAILED — fleet must not launch")


def run(fold_id: int, seed: int) -> None:
    UNITS.mkdir(parents=True, exist_ok=True)
    out = UNITS / f"fold_{fold_id}_seed_{seed}.json"
    if out.is_file() and json.loads(out.read_text()).get("complete"):
        print(json.dumps({"skipped": str(out)}))
        return
    up, data, fold, registry30, eb120, assets, model, schedule, device = _setup(fold_id, seed)
    rows, comps = [], {}
    for participant, session, task in itertools.product(
            sorted(fold["test"]), data["sessions"], data["tasks"]):
        key = (participant, session, task)
        if key not in assets:
            continue
        cell_rows, meta = _cell_rows(up, fold_id, seed, data, registry30, eb120,
                                     assets, model, schedule, device, key)
        rows.extend(cell_rows)
        comps["|".join(key)] = meta
        print(json.dumps({"cell": "|".join(key), "rows": len(rows)}), flush=True)
    keys = {(r["participant"], r["session"], r["task"], r["start"], r["arm"]) for r in rows}
    out.write_text(json.dumps({"fold": fold_id, "seed": seed, "n_rows": len(rows),
                               "n_unique_keys": len(keys), "complete": True,
                               "compositions": comps, "rows": rows,
                               "frozen": "prereg_wave6 amendment W6-1b"},
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
