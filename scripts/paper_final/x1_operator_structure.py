#!/usr/bin/env python3
"""WAVE-6 E1 — repeatable structure in the propagation relation (CPU only).

Frozen design: reports/prereg_wave6_propagation_FROZEN.md section 2.

Banks, for every dev cell and every non-overlapping 120-s window, the ridge
operator (the estimator t4_staleness.cpu() already uses), then computes the four
comparison classes R1-R4 under a probe-based distance whose probe bank is drawn
from TRAINING participants only and fixed before any distance is computed.

Also banks the deployed EB operator and its RAW counterpart per cell, so that
similarity manufactured by shrinkage toward a shared population mean can be told
apart from similarity of the underlying fits, and so that E2/E3/E4 read their
donor distances from one place.

Outputs
  results/paper_final/wave6/e1_operators.npz   operators + metadata (arrays)
  results/paper_final/wave6/e1_structure.json  distance classes + endpoints
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from pf_common import OUT, stat

WAVE6 = OUT / "wave6"
WINDOW_SECONDS = 120
CALIB_SECONDS = 120           # E1 probe bank is taken from calibration windows


def _fit_windows(registry30, key, rate):
    """Per-window ridge operators for one cell, in the shared latent coordinate.

    Mirrors t4_staleness.cpu() exactly (same scaling, same ridge ratio) so the
    two analyses cannot disagree about what the operator is.
    """
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog, ridge_transfer
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale

    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    length = min(eeg.shape[1], eog.shape[1])
    span = WINDOW_SECONDS * rate
    fits, latents = {}, {}
    for start in range(0, length - span + 1, span):
        seg = eog[:, start:start + span]
        center, scale = robust_center_scale(seg)
        latent = (seg - center[:, None]) / scale[:, None]
        scaled = eeg[:, start:start + span] / registry30.eeg_scale[:, None]
        fits[start], _ = ridge_transfer(scaled, latent, registry30.ridge_ratio)
        latents[start] = latent
    return fits, latents


def _probe_bank(latents_by_cell, train_participants, rate, rng):
    """Fixed EOG probe per (session, task) from TRAINING participants only.

    Concatenates the first calibration window of every training cell of that
    (session, task); the probe is therefore independent of any evaluation window
    and of every recipient.
    """
    probes: dict[tuple[str, str], list[np.ndarray]] = {}
    for key, latents in sorted(latents_by_cell.items()):
        if key[0] not in train_participants or not latents:
            continue
        first = min(latents)
        probes.setdefault(key[1:], []).append(latents[first][:, :CALIB_SECONDS * rate])
    return {st: np.concatenate(v, axis=1) for st, v in probes.items() if v}


def _distances(a, b, probe):
    diff = a - b
    return {
        "probe": float(np.linalg.norm(diff @ probe)),
        "matrix": float(np.linalg.norm(diff)),
        "direction": float(np.linalg.norm(
            a / max(np.linalg.norm(a), 1e-12) - b / max(np.linalg.norm(b), 1e-12))),
        "gain_log": float(abs(np.log(max(np.linalg.norm(a), 1e-12)
                                     / max(np.linalg.norm(b), 1e-12)))),
    }


def main() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, folds, _ = configs()
    rate = int(data.get("sampling_rate", 100))
    WAVE6.mkdir(parents=True, exist_ok=True)

    # ---- per-fold banking (a cell's operators come from the fold that owns it)
    win_ops: dict[tuple, dict[int, np.ndarray]] = {}
    latents: dict[tuple, dict[int, np.ndarray]] = {}
    eb_ops, raw_ops, cell_meta = {}, {}, {}
    train_of_fold: dict[int, list[str]] = {}
    for fold_id, fold in enumerate(folds):
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        train_of_fold[fold_id] = sorted(fold["train"])
        for key in sorted(registry30.cells):
            if key[0] not in fold["test"]:
                continue          # each cell banked once, by its own test fold
            fits, lat = _fit_windows(registry30, key, rate)
            if not fits:
                continue
            win_ops[key], latents[key] = fits, lat
            eb_ops[key] = eb120.operator(*key, "EB")
            raw_ops[key] = eb120.operator(*key, "RAW")
            cell = eb120.cells[key]
            quality = np.asarray(getattr(cell, "quality", np.nan), float).ravel()
            cell_meta[key] = {"fold": fold_id, "lam": float(cell.lam),
                              "hard_gate": bool(getattr(cell, "hard_gate", False)),
                              "quality": [float(q) for q in quality],
                              "quality_mean": float(np.nanmean(quality)),
                              "n_windows": len(fits)}
        # training cells only supply the probe bank
        for key in sorted(registry30.cells):
            if key[0] in fold["train"] and key not in latents:
                _, lat = _fit_windows(registry30, key, rate)
                latents.setdefault(key, lat)
    print(json.dumps({"cells_banked": len(win_ops),
                      "cells_with_latents": len(latents)}), flush=True)

    # ---- probe bank: fixed, training-only, per (session, task) ----------------
    all_train = sorted({p for v in train_of_fold.values() for p in v})
    probes = _probe_bank(latents, all_train, rate, np.random.default_rng(20260905))
    scale_ref = {}
    for st, probe in probes.items():
        norms = [np.linalg.norm(op @ probe) for k, op in eb_ops.items() if k[1:] == st]
        scale_ref[st] = float(np.mean(norms)) if norms else 1.0
    mat_ref = float(np.mean([np.linalg.norm(op) for op in eb_ops.values()]))

    # ---- comparison classes ---------------------------------------------------
    def norm_row(d, st):
        return {"probe": d["probe"] / max(scale_ref[st], 1e-12),
                "matrix": d["matrix"] / max(mat_ref, 1e-12),
                "direction": d["direction"], "gain_log": d["gain_log"]}

    rows = []
    for key, fits in sorted(win_ops.items()):
        st = key[1:]
        if st not in probes:
            continue
        probe = probes[st]
        starts = sorted(fits)
        # R1 adjacent non-overlapping windows of the same cell
        for a, b in zip(starts, starts[1:]):
            rows.append({"cls": "R1_adjacent_repeat", "participant": key[0],
                         "cell": "|".join(key), "other": "|".join(key),
                         "gap_s": (b - a) // rate,
                         **norm_row(_distances(fits[a], fits[b], probe), st)})
        # R2 first window vs every later window of the same cell
        for b in starts[1:]:
            rows.append({"cls": "R2_within_record_time", "participant": key[0],
                         "cell": "|".join(key), "other": "|".join(key),
                         "gap_s": (b - starts[0]) // rate,
                         **norm_row(_distances(fits[starts[0]], fits[b], probe), st)})

    # R3 same participant, different (session, task); R4 different participants,
    # same (session, task) — both on the deployed EB operator and on RAW.
    for variant, ops in (("EB", eb_ops), ("RAW", raw_ops)):
        for ka, kb in itertools.combinations(sorted(ops), 2):
            if ka[0] == kb[0] and ka[1:] != kb[1:]:
                cls, st = "R3_across_condition", ka[1:]
            elif ka[0] != kb[0] and ka[1:] == kb[1:]:
                cls, st = "R4_across_participant", ka[1:]
            else:
                continue
            if st not in probes:
                continue
            rows.append({"cls": f"{cls}_{variant}", "participant": ka[0],
                         "cell": "|".join(ka), "other": "|".join(kb), "gap_s": -1,
                         **norm_row(_distances(ops[ka], ops[kb], probes[st]), st)})

    # ---- endpoints ------------------------------------------------------------
    def medians_by_participant(cls, metric):
        per: dict[str, list[float]] = {}
        for r in rows:
            if r["cls"] == cls:
                per.setdefault(r["participant"], []).append(r[metric])
        return {p: float(np.median(v)) for p, v in per.items()}

    endpoints = {}
    for metric in ("probe", "matrix", "direction", "gain_log"):
        r1 = medians_by_participant("R1_adjacent_repeat", metric)
        for variant in ("EB", "RAW"):
            r4 = medians_by_participant(f"R4_across_participant_{variant}", metric)
            r3 = medians_by_participant(f"R3_across_condition_{variant}", metric)
            common = sorted(set(r1) & set(r4))
            endpoints[f"{metric}_R4_minus_R1_{variant}"] = stat(
                [r4[p] - r1[p] for p in common])
            common3 = sorted(set(r3) & set(r4))
            endpoints[f"{metric}_R4_minus_R3_{variant}"] = stat(
                [r4[p] - r3[p] for p in common3])
        r2 = medians_by_participant("R2_within_record_time", metric)
        common2 = sorted(set(r1) & set(r2))
        endpoints[f"{metric}_R2_minus_R1"] = stat([r2[p] - r1[p] for p in common2])

    class_summary = {}
    for cls in sorted({r["cls"] for r in rows}):
        vals = [r["probe"] for r in rows if r["cls"] == cls]
        class_summary[cls] = {"n": len(vals), "median": float(np.median(vals)),
                              "mean": float(np.mean(vals)),
                              "q25": float(np.quantile(vals, .25)),
                              "q75": float(np.quantile(vals, .75))}

    # shrinkage / reliability confound diagnostics
    lam = {"|".join(k): v["lam"] for k, v in cell_meta.items()}
    r4_by_cell: dict[str, list[float]] = {}
    for r in rows:
        if r["cls"] == "R4_across_participant_EB":
            r4_by_cell.setdefault(r["cell"], []).append(r["probe"])
    paired = [(lam[c], float(np.median(v))) for c, v in r4_by_cell.items() if c in lam]
    confound = {"n_cells": len(paired)}
    if len(paired) >= 3:
        a = np.array(paired)
        confound["pearson_lambda_vs_R4_probe"] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])

    decision = {
        "frozen": "reports/prereg_wave6_propagation_FROZEN.md#2",
        "primary_endpoint": "probe_R4_minus_R1_EB",
        "endpoints": endpoints, "class_summary": class_summary,
        "shrinkage_confound": confound,
        "lambda_by_cell": lam,
        "n_rows": len(rows), "n_cells": len(win_ops),
        "probe_bank": {"|".join(k): list(v.shape) for k, v in probes.items()},
    }
    (WAVE6 / "e1_structure.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")

    keys = sorted(eb_ops)
    np.savez_compressed(
        WAVE6 / "e1_operators.npz",
        cell_keys=np.asarray(["|".join(k) for k in keys]),
        eb=np.stack([eb_ops[k] for k in keys]),
        raw=np.stack([raw_ops[k] for k in keys]),
        meta=np.asarray(json.dumps({"|".join(k): cell_meta[k] for k in keys})),
        probe_keys=np.asarray(["|".join(k) for k in sorted(probes)]),
        **{"probe_" + "|".join(k): probes[k] for k in sorted(probes)},
        window_ops=np.asarray(json.dumps(
            {"|".join(k): {str(s): v.tolist() for s, v in f.items()}
             for k, f in win_ops.items()})),
    )
    print(json.dumps({"primary": endpoints["probe_R4_minus_R1_EB"],
                      "classes": {k: v["median"] for k, v in class_summary.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
