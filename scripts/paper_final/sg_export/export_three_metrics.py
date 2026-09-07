#!/usr/bin/env python
"""Item D — temporal RRMSE, spectral RRMSE and Pearson correlation for every row
of the paired-reconstruction table, on exactly the windows behind the banked
temporal numbers.  Banks results/paper_final/gapfill/g11_three_metrics.json.

Development cohort (rows re-read from the campaign's own per-window records,
all of which were written by eeg_scad.evaluation.paired_metrics and therefore
already carry the three metrics):
  RAW / WRONG / WRONG_gated / SHUFFLED / MATCH_gated / POP / NO_A0  <- V44 stage-1
      stage1_result.json, 5 folds x 3 seeds (the g6 windows)
  det_twin_matched (DET_MATCH_gated)                              <- V44 stage-1 det_result.json,
      seeds 20261201/02 (+ 20261203 when item B has landed)
  det_pop_retrained (DET_POP)                                     <- V43 stage-2 det_result.json
  linear_calibrated (arm C_gated)                                 <- V44 stage-0 paired_arm_rows.csv
Held-out cohort: RAW / POP / NO_A0 / MATCH_gated on the eight windows per sealed
participant, re-scored by replaying the M35 C-1 evaluator (fold-99 bank, sampler
seed 20269001, digest-verified frozen sealed_outputs.npz) — the g1 windows.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PF = REPO / "scripts/paper_final"
sys.path.insert(0, str(PF))
from pf_common import SEALED, V44_ROOT  # noqa: E402

V43_ROOT = Path("/home/infres/yinwang/denoiseNet_rgcc_v43")
FLAGSHIP = Path("/home/infres/yinwang/denoiseNet_flagship_m0")
CONS = Path("/home/infres/yinwang/denoiseNet_consolidated")
SEALED_NPZ = FLAGSHIP / "results/flagship_m35/c1_sealed/sealed_outputs.npz"
SEALED_SHA = "857a6713a7a6453429bb7d8821593187b23f062fa04d57e04be268c3df22ff60"
OUT = REPO / "results/paper_final/gapfill/g11_three_metrics.json"
METRICS = ("rrmse_temporal", "rrmse_spectral", "correlation")
SEEDS = (20261201, 20261202, 20261203)
TARGETS = {("dev", "det_pop_retrained"): 0.5043, ("dev", "RAW"): 0.6076, ("dev", "WRONG_gated"): 0.6896, ("dev", "WRONG"): 1.0807,
           ("dev", "SHUFFLED"): 0.7511, ("dev", "det_twin_matched"): 0.4968,
           ("dev", "linear_calibrated"): 0.4353, ("heldout", "RAW"): 0.7753,
           ("heldout", "POP"): 0.7661, ("heldout", "NO_A0"): 0.6888, ("heldout", "MATCH_gated"): 0.5351}


def row(cohort, arm, r, window, seed=None, source=""):
    return {"cohort": cohort, "arm": arm, "participant": r["participant"], "fold": int(r["fold"]),
            "seed": int(seed if seed is not None else r["seed"]), "window": int(window),
            "zero_artifact": int(r["zero_artifact"]),
            **{m: float(r[m]) for m in METRICS}, "source": source}


def dev_rows() -> tuple[list[dict], dict]:
    rows, notes = [], {}
    # V44 stage-1 diffusion arms, all three seeds
    for fold in range(5):
        for seed in SEEDS:
            p = V44_ROOT / f"results/rgcc_eog_v44/stage1/fold_{fold}_seed_{seed}/stage1_result.json"
            counter = defaultdict(int)
            for r in json.loads(p.read_text())["rows"]:
                c = r["condition"]
                if c in ("RAW", "WRONG", "WRONG_gated", "SHUFFLED", "MATCH_gated", "POP", "NO_A0"):
                    rows.append(row("dev", c, r, counter[c], source=str(p.relative_to(V44_ROOT.parent))))
                    counter[c] += 1
    # deterministic twin (V44 stage-1 det); seed 3 from item B when present
    def complete_seeds(roots, pattern):
        ok = []
        for seed in SEEDS:
            if all(any((root / pattern.format(fold=f, seed=seed)).is_file() for root in roots) for f in range(5)):
                ok.append(seed)
        return ok
    twin_roots = (V44_ROOT / "results/rgcc_eog_v44", CONS / "results/rgcc_eog_v44")
    twin_seeds = complete_seeds(twin_roots, "stage1/det_fold_{fold}_seed_{seed}/det_result.json")
    found = []
    for fold in range(5):
        for seed in twin_seeds:
            for root in twin_roots:
                p = root / f"stage1/det_fold_{fold}_seed_{seed}/det_result.json"
                if p.is_file():
                    break
            else:
                continue
            found.append(seed)
            counter = defaultdict(int)
            for r in json.loads(p.read_text())["rows"]:
                if r["condition"] == "DET_MATCH_gated":
                    rows.append(row("dev", "det_twin_matched", r, counter["d"], source=str(p)))
                    counter["d"] += 1
    notes["det_twin_matched_seeds"] = sorted(set(found))
    pop_roots = (V43_ROOT / "results/rgcc_v43", CONS / "results/rgcc_v43")
    pop_seeds = complete_seeds(pop_roots, "stage2/det_fold_{fold}_seed_{seed}/det_result.json")
    found = []
    for fold in range(5):
        for seed in pop_seeds:
            for root in pop_roots:
                p = root / f"stage2/det_fold_{fold}_seed_{seed}/det_result.json"
                if p.is_file():
                    break
            else:
                continue
            found.append(seed)
            counter = defaultdict(int)
            for r in json.loads(p.read_text())["rows"]:
                if r["condition"] == "DET_POP":
                    rows.append(row("dev", "det_pop_retrained", r, counter["d"], source=str(p)))
                    counter["d"] += 1
    notes["det_pop_retrained_seeds"] = sorted(set(found))
    # calibrated linear regression, V44 stage-0 (closed form), arm C_gated
    p = V44_ROOT / "results/rgcc_eog_v44/stage0/paired_arm_rows.csv"
    counter = defaultdict(int)
    with p.open() as fh:
        for r in csv.DictReader(fh):
            if r["arm"] != "C_gated":
                continue
            key = (r["fold"], r["seed"])
            rows.append(row("dev", "linear_calibrated", r, counter[key], source=str(p.relative_to(V44_ROOT.parent))))
            counter[key] += 1
    return rows, notes


def heldout_rows() -> list[dict]:
    from t1_heldout_uq import _fold99_context
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    sha = hashlib.sha256(SEALED_NPZ.read_bytes()).hexdigest()
    assert sha == SEALED_SHA, sha
    data, fold, registry30, eb120, assets, sampler = _fold99_context()
    bank = sampler.sample_balanced(8)
    rows = []
    per_subject_local = defaultdict(int)
    with np.load(SEALED_NPZ, allow_pickle=False) as archive:
        for i, (clean, observed, artifact, meta) in enumerate(zip(bank["x"], bank["y"], bank["artifact"], bank["meta"])):
            subject = meta["participant"]
            local = per_subject_local[subject]; per_subject_local[subject] += 1
            base = {"participant": subject, "fold": 99, "seed": 20261201, "zero_artifact": int(meta["zero_artifact"])}
            m = paired_metrics(clean, observed, artifact, np.zeros_like(artifact))
            rows.append(row("heldout", "RAW", {**base, **m}, local, source="bank (contaminated input)"))
            for arm in ("MATCH_gated", "NO_A0", "POP"):
                prediction = np.asarray(archive[f"paired_{subject}_{arm}"])[local]
                m = paired_metrics(clean, observed, artifact, observed - prediction)
                rows.append(row("heldout", arm, {**base, **m}, local, source=str(SEALED_NPZ)))
    return rows


def participant_first(rows, cohort, arm, seeds=None):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["cohort"] == cohort and r["arm"] == arm and (seeds is None or r["seed"] in seeds):
            for m in METRICS:
                per[r["participant"]][m].append(r[m])
    if not per:
        return None
    return {m: float(np.mean([np.mean(v[m]) for v in per.values()])) for m in METRICS} | {"participants": len(per)}


def main() -> None:
    dev, notes = dev_rows()
    held = heldout_rows()
    rows = dev + held
    pooled, lines = {}, []
    for cohort in ("dev", "heldout"):
        arms = sorted({r["arm"] for r in rows if r["cohort"] == cohort})
        for arm in arms:
            pf = participant_first(rows, cohort, arm)
            pooled[f"{cohort}/{arm}"] = pf
            target = TARGETS.get((cohort, arm))
            gate = ""
            if target is not None:
                gate_pf = participant_first(rows, cohort, arm, {20261201, 20261202}) if arm.startswith("det_") else pf
                ok = abs(gate_pf["rrmse_temporal"] - target) < 5e-5
                gate = f" | banked {target:.4f} {'PASS' if ok else 'FAIL'}"
            lines.append(f"{cohort:8s} {arm:18s} temporal {pf['rrmse_temporal']:.4f} spectral {pf['rrmse_spectral']:.4f} "
                         f"correlation {pf['correlation']:.4f} (n={pf['participants']}){gate}")
    # det twin on the two shared seeds (the manuscript's comparison) and on three seeds
    for label, seeds in (("two shared seeds", {20261201, 20261202}), ("three seeds", set(SEEDS))):
        pf = participant_first(rows, "dev", "det_twin_matched", seeds)
        if pf:
            pooled[f"dev/det_twin_matched[{label}]"] = pf
            lines.append(f"dev      det_twin_matched[{label}] temporal {pf['rrmse_temporal']:.4f} spectral {pf['rrmse_spectral']:.4f} correlation {pf['correlation']:.4f}")
    fails = [l for l in lines if l.endswith("FAIL")]
    payload = {"rows": rows, "pooled_check": pooled, "gate_lines": lines, "targets": {f"{c}/{a}": v for (c, a), v in TARGETS.items()},
               "notes": {**notes, "metrics": "eeg_scad.evaluation.paired_metrics: rrmse_temporal, rrmse_spectral (Welch spectrum of the window), correlation (Pearson with the reference)",
                         "pooling": "participant-first: mean over a participant's windows across folds/seeds, then mean over participants",
                         "heldout": f"fold-99 bank (t1_heldout_uq._fold99_context, sampler seed 20269001, sample_balanced(8)); predictions from {SEALED_NPZ} sha256 {SEALED_SHA}",
                         "sealed_reads": "none beyond the cells the original pass logged"}}
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print("\n".join(lines)); print(json.dumps({"rows": len(rows), "fails": len(fails), "notes": notes}))
    if fails:
        raise SystemExit("GATE FAILED")


if __name__ == "__main__":
    main()
