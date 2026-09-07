#!/usr/bin/env python3
"""E1 export — propagation matrices per development recording (CPU only).

SERVER_EXPORT_FIGURE_ARRAYS_v4.md, item E1.  Reads the wave-6 operator bank
(results/paper_final/wave6/e1_operators.npz: eb = A_tilde, raw = A_hat, meta =
lam / hard_gate per cell) and ADDS, per cell:

  A_pop    population matrix of that (session, task) for the fold whose TEST set
           holds the participant  ->  fold-train mean, participant excluded
  A_later  the E3 layer-2 object: ridge regression of the four natural windows'
           scaled EEG on their EOG drive (x3_event_selectivity._natural_coupling,
           imported, not re-typed, so the two analyses cannot disagree)
  lambda / accepted (= not hard_gate) / participant / session / task / condition

Gates asserted at runtime:
  G1  A_tilde == A_pop + lambda * (A_hat - A_pop)   to 1e-9, every cell
  G2  A_hat   == eb120.operator(key, "RAW") and A_tilde == eb120.operator(key, "EB")
      re-derived from the frozen registries, to 1e-9
  G3  lambda / hard_gate / fold in the stored meta == the frozen EB cell
  G4  cos(A_tilde[:, c], A_later[:, c]) reproduces the banked E3 layer-2
      own_cosine for every (cell, column) to 1e-6 (skipped only if the JSON is absent)

Output: results/paper_final/paper_final_arrays/e1_operators.npz (via sg_common.save_npz)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402

WAVE6 = sg.OUT / "wave6"
SOURCE = WAVE6 / "e1_operators.npz"
E3_JSON = WAVE6 / "e3_event_selectivity.json"
OUT_NAME = "e1_operators.npz"
ARM_NAMES = ("A_tilde", "A_hat", "A_pop", "A_later")


def load_wave6():
    """cell_keys, eb (A_tilde), raw (A_hat), meta dict — one file, no registry."""
    with np.load(SOURCE, allow_pickle=False) as d:
        keys = [str(k) for k in d["cell_keys"]]
        eb = np.asarray(d["eb"], np.float64)
        raw = np.asarray(d["raw"], np.float64)
        meta = json.loads(str(d["meta"]))
    assert eb.shape == raw.shape == (len(keys), 46, 2), (eb.shape, raw.shape)
    assert all(k in meta for k in keys), "meta is missing cells"
    return keys, eb, raw, meta


def main() -> None:
    from eeg_scad.cli import run_v44 as up
    from x3_event_selectivity import _natural_coupling, _cosine

    keys, eb, raw, meta = load_wave6()
    n = len(keys)
    parts = [k.split("|") for k in keys]
    participant = np.asarray([p[0] for p in parts])
    session = np.asarray([p[1] for p in parts])
    task = np.asarray([p[2] for p in parts])
    condition = np.asarray([sg.SESSION_CONDITION.get(s, "unknown") for s in session])
    lam = np.asarray([float(meta[k]["lam"]) for k in keys])
    hard_gate = np.asarray([bool(meta[k]["hard_gate"]) for k in keys])
    fold_meta = np.asarray([int(meta[k]["fold"]) for k in keys])
    accepted = ~hard_gate

    A_pop = np.full((n, 46, 2), np.nan)
    A_later = np.full((n, 46, 2), np.nan)
    later_starts = np.full((n, up.NATURAL_WINDOWS_PER_CELL), -1, dtype=np.int64)
    fold_used = np.full(n, -1, dtype=np.int64)

    t0 = time.time()
    for fold_id in sorted(set(fold_meta.tolist())):
        idx = np.flatnonzero(fold_meta == fold_id)
        fid, fold = sg.fold_of(str(participant[idx[0]]))
        assert fid == fold_id, (fid, fold_id)                         # G3 (fold)
        data, r30, eb120 = sg.registry(fold)
        print(json.dumps({"fold": fold_id, "cells": int(len(idx)),
                          "test": sorted(fold["test"]), "t": round(time.time() - t0, 1)}),
              flush=True)
        for i in idx:
            key = (str(participant[i]), str(session[i]), str(task[i]))
            assert key[0] in fold["test"], (key, fold_id)             # G3 (fold)
            assert key[0] not in fold["train"], (key, fold_id)        # participant excluded
            cell = eb120.cells[key]
            assert abs(float(cell.lam) - lam[i]) < 1e-12, (key, cell.lam, lam[i])   # G3
            assert bool(cell.hard_gate) == bool(hard_gate[i]), key                  # G3
            # G2: the banked operators are the frozen registry's operators
            assert np.allclose(raw[i], eb120.operator(*key, "RAW"), atol=1e-9, rtol=0), key
            assert np.allclose(eb[i], eb120.operator(*key, "EB"), atol=1e-9, rtol=0), key
            A_pop[i] = r30.population_transfer[key[1:]]
            coupling = _natural_coupling(r30, data, key, up)
            if coupling is not None:
                A_later[i] = coupling
                starts = [s for s, _, _ in up._natural_windows(r30, data, key)]
                later_starts[i, :len(starts)] = starts
            fold_used[i] = fold_id
        del data, r30, eb120

    # G1: the shrinkage identity, every cell, 1e-9
    recon = A_pop + lam[:, None, None] * (raw - A_pop)
    err = np.abs(recon - eb).max(axis=(1, 2))
    assert np.isfinite(A_pop).all(), "A_pop missing for some cell"
    assert err.max() < 1e-9, f"G1 shrinkage identity violated: max err {err.max():.3e}"
    print(json.dumps({"G1_max_abs_err": float(err.max())}), flush=True)

    coverage = np.stack([np.isfinite(eb).all(axis=(1, 2)), np.isfinite(raw).all(axis=(1, 2)),
                         np.isfinite(A_pop).all(axis=(1, 2)), np.isfinite(A_later).all(axis=(1, 2))])
    print(json.dumps({"coverage": {a: int(c.sum()) for a, c in zip(ARM_NAMES, coverage)},
                      "accepted": int(accepted.sum()), "cells": n}), flush=True)

    # G4: reproduce the banked E3 layer-2 own_cosine per (cell, column)
    own_cos = np.full((n, 2), np.nan)
    for i in range(n):
        if coverage[3, i]:
            own_cos[i] = [_cosine(eb[i][:, c], A_later[i][:, c]) for c in (0, 1)]
    if E3_JSON.is_file():
        rows = json.loads(E3_JSON.read_text())["layer2_natural_prediction"]["rows"]
        banked = {(r["cell"], r["column"]): float(r["own_cosine"]) for r in rows}
        col = {"vertical": 0, "horizontal": 1}
        checked, worst = 0, 0.0
        for (cell, column), value in banked.items():
            i = keys.index(cell)
            assert coverage[3, i], f"E3 has a coupling for {cell} but this export has none"
            diff = abs(own_cos[i, col[column]] - value)
            worst = max(worst, diff)
            assert diff < 1e-6, f"G4 own_cosine mismatch {cell} {column}: {diff:.3e}"
            checked += 1
        print(json.dumps({"G4_checked": checked, "G4_worst_abs_diff": worst}), flush=True)
    else:
        print(json.dumps({"G4": "skipped (E3 json absent)"}), flush=True)

    sg.save_npz(
        OUT_NAME,
        participant=participant, session=session, task=task, condition=condition,
        cell_keys=np.asarray(keys),
        A_tilde=eb, A_hat=raw, A_pop=A_pop, A_later=A_later,
        eb=eb, raw=raw,                                   # wave-6 aliases
        **{"lambda": lam},
        hard_gate=hard_gate, accepted=accepted, fold=fold_used,
        later_window_starts=later_starts,
        own_cosine_tilde_later=own_cos,
        coverage=coverage, coverage_names=np.asarray(ARM_NAMES),
        meta=np.asarray(json.dumps(meta)),
        source=np.asarray(str(SOURCE)),
        units=np.asarray("46x2 operators map latent (centred/scaled) bipolar EOG "
                         "[V, H] to fold-scaled EEG; A_pop = fold-train mean with "
                         "the participant excluded; A_later = ridge fit of the 4 "
                         "natural evaluation windows (x3 layer 2)"),
    )
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
