#!/usr/bin/env python
"""SERVER_INSTRUCTIONS_SPECTRAL_COST_V2.md item B — append the seed-20261203
deterministic-twin rows to results/paper_final/gapfill/g4_estimators.json and
report the twin-vs-diffusion comparison on all three seeds.

Rows are built exactly like the existing det_twin_matched rows (condition
DET_MATCH_gated of det_result.json, campaign rgcc_eog_v44_stage1) and, when the
V43 population-trained det results exist for the seed, like det_pop_retrained
(condition DET_POP of the V43 stage-2 det_result.json). Pooling is participant-
first, as the file's note states.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CONS = Path("/home/infres/yinwang/denoiseNet_consolidated")
G4 = REPO / "results/paper_final/gapfill/g4_estimators.json"
SEED = 20261203
sys.path.insert(0, str(REPO / "scripts/paper_final"))
from pf_common import stat  # noqa: E402


def rows_from(det_path: Path, condition: str, arm: str, campaign: str) -> list[dict]:
    d = json.loads(det_path.read_text())
    out = []
    for r in d["rows"]:
        if r["condition"] != condition:
            continue
        out.append({"arm": arm, "campaign": campaign, "condition": condition,
                    "participant": r["participant"], "fold": r["fold"], "seed": r["seed"],
                    "rrmse_temporal": r["rrmse_temporal"], "zero_artifact": r["zero_artifact"],
                    "source": str(det_path.relative_to(CONS.parent))})
    return out


def participant_first(rows: list[dict], arm: str, seeds=None) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    for r in rows:
        if r["arm"] == arm and (seeds is None or r["seed"] in seeds):
            per.setdefault(r["participant"], []).append(r["rrmse_temporal"])
    return {p: float(np.mean(v)) for p, v in per.items()}


def main() -> None:
    d = json.loads(G4.read_text())
    rows = [r for r in d["rows"] if not (r["seed"] == SEED and r["arm"] in ("det_twin_matched", "det_pop_retrained"))]
    added = {}
    twin = [CONS / f"results/rgcc_eog_v44/stage1/det_fold_{f}_seed_{SEED}/det_result.json" for f in range(5)]
    assert all(p.is_file() for p in twin), [str(p) for p in twin if not p.is_file()]
    new = [r for p in twin for r in rows_from(p, "DET_MATCH_gated", "det_twin_matched", "rgcc_eog_v44_stage1")]
    assert len(new) == 120, len(new)
    rows += new; added["det_twin_matched"] = len(new)
    pop = [CONS / f"results/rgcc_v43/stage2/det_fold_{f}_seed_{SEED}/det_result.json" for f in range(5)]
    if all(p.is_file() for p in pop):
        newp = [r for p in pop for r in rows_from(p, "DET_POP", "det_pop_retrained", "rgcc_v43_stage2")]
        assert len(newp) == 120, len(newp)
        rows += newp; added["det_pop_retrained"] = len(newp)
    d["rows"] = rows
    lines = []
    # seed-3 twin on its own
    t3 = participant_first(rows, "det_twin_matched", {SEED}); m3 = participant_first(rows, "diffusion_matched", {SEED})
    lines.append(f"seed {SEED}: det_twin_matched participant-first RRMSE {np.mean(list(t3.values())):.4f} "
                 f"vs diffusion_matched {np.mean(list(m3.values())):.4f} (n={len(t3)})")
    # three-seed twin vs three-seed diffusion, paired within participant
    seeds3 = {20261201, 20261202, SEED}
    tw = participant_first(rows, "det_twin_matched", seeds3); df = participant_first(rows, "diffusion_matched", seeds3)
    common = sorted(set(tw) & set(df))
    paired = stat([tw[p] - df[p] for p in common])
    lines.append(f"three seeds: det_twin_matched {np.mean([tw[p] for p in common]):.4f} vs diffusion_matched "
                 f"{np.mean([df[p] for p in common]):.4f}; paired twin-minus-diffusion {paired['mean']:+.4f} "
                 f"[{paired['bootstrap_low']:+.4f},{paired['bootstrap_high']:+.4f}], {paired['positive_count']}/{paired['participants']} twin worse")
    two = {20261201, 20261202}
    tw2 = participant_first(rows, "det_twin_matched", two); df2 = participant_first(rows, "diffusion_matched", two)
    lines.append(f"two shared seeds (as in the manuscript): twin {np.mean(list(tw2.values())):.4f} vs diffusion {np.mean(list(df2.values())):.4f}")
    if "det_pop_retrained" in added:
        p3 = participant_first(rows, "det_pop_retrained", {SEED}); q3 = participant_first(rows, "diffusion_pop_retrained", {SEED})
        lines.append(f"seed {SEED}: det_pop_retrained {np.mean(list(p3.values())):.4f} vs diffusion_pop_retrained {np.mean(list(q3.values())):.4f}")
    for arm in ("det_twin_matched", "det_pop_retrained"):
        pf = participant_first(rows, arm)
        d["per_participant_means"][arm] = pf
        d["pooled_check"].setdefault(arm, {})
        d["pooled_check"][arm]["participant_first_mean_all_seeds"] = float(np.mean(list(pf.values()))) if pf else None
        d["pooled_check"][arm]["seeds"] = sorted({r["seed"] for r in rows if r["arm"] == arm})
    d["note"] = str(d["note"]) + (f" | 2026-09-07: seed {SEED} deterministic-twin rows appended (det_twin_matched"
                                  + (", det_pop_retrained" if "det_pop_retrained" in added else "")
                                  + f"), trained with the identical stage-1 det recipe (80000 updates) from the consolidated tree; "
                                  "pooled_check now carries all-seed participant-first means for the det arms.")
    d["source_files"] = list(d["source_files"]) + [str(p) for p in twin] + ([str(p) for p in pop] if "det_pop_retrained" in added else [])
    d["seed3_report"] = lines
    G4.write_text(json.dumps(d, indent=1) + "\n")
    print(json.dumps({"added": added}))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
