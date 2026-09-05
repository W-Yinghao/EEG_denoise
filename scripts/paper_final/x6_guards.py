#!/usr/bin/env python3
"""WAVE-6 guards — two confound checks the plan demands before its results can
be read (CPU).

G-A (E5 identifiability).  `added_exp.md`, experiment 5: "不能把另一列参数不可辨
认这一直接结果当作主要发现".  A composition dominated by one EOG direction may
leave the OTHER column of the operator unidentifiable, and a bad column hurts
every natural window.  For every cell and every composition arm this recomputes
the calibration set and reports the conditioning of the latent EOG design and
the per-column norm of the resulting operator.

G-B (E2 OWN_OTHER confound).  The OWN_OTHER arm applies an operator fitted in a
different (session, task) cell, where the EOG latent is robust-scaled
separately.  Its penalty could therefore be a latent-scale mismatch rather than
a condition effect.  This decomposes the OWN_OTHER and near-stranger distances
into their gain and direction parts so the two explanations can be told apart.

Output results/paper_final/wave6/guards.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from pf_common import OUT, stat

sys.path.insert(0, str(Path(__file__).resolve().parent))
V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
WAVE6 = OUT / "wave6"


def guard_a():
    import x5_calibration_content as x5
    sys.path.insert(0, str(V44_SRC))
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry, bipolar_eog
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale

    data, folds, _ = configs()
    rate = int(data.get("sampling_rate", 100))
    span = x5.BLOCK_S * rate
    rows = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        for key in sorted(registry30.cells):
            if key[0] not in fold["test"]:
                continue
            eeg, eye, names = registry30._load(*key)
            eog = bipolar_eog(eye, names)
            limit = min(int(data["qnatural_start"]), eeg.shape[1], eog.shape[1])
            blocks = x5._blocks(eog, limit, rate)
            if len(blocks) < x5.SET_SECONDS // x5.BLOCK_S:
                continue
            lam = float(eb120.cells[key].lam)
            pop = registry30.population_transfer[key[1:]]
            rng = np.random.default_rng(abs(hash(("e5",) + key)) % (2 ** 31))
            for kind, draw in itertools.product(x5.COMPOSITIONS, range(x5.DRAWS)):
                chosen, stats = x5._compose(blocks, kind, draw, rate, rng)
                if chosen is None:
                    continue
                seg = np.concatenate([eog[:, b["start"]:b["start"] + span]
                                      for b in chosen], axis=1)
                center, scale = robust_center_scale(seg)
                latent = (seg - center[:, None]) / scale[:, None]
                sv = np.linalg.svd(latent, compute_uv=False)
                operator, _ = x5._fit_composition(registry30, eeg, eog, chosen,
                                                  rate, lam, pop)
                rows.append({
                    "cell": "|".join(key), "arm": f"{kind}_{draw}",
                    "composition": kind, "draw": draw,
                    "mean_ratio": stats["mean_ratio"],
                    "singular_ratio": float(sv[0] / max(sv[-1], 1e-12)),
                    "v_row_energy": float(np.sqrt(np.mean(latent[0] ** 2))),
                    "h_row_energy": float(np.sqrt(np.mean(latent[1] ** 2))),
                    "op_v_col_norm": float(np.linalg.norm(operator[:, 0])),
                    "op_h_col_norm": float(np.linalg.norm(operator[:, 1])),
                })
    summary = {}
    for kind in x5.COMPOSITIONS:
        sub = [r for r in rows if r["composition"] == kind]
        summary[kind] = {
            "n": len(sub),
            "singular_ratio_median": float(np.median([r["singular_ratio"] for r in sub])),
            "singular_ratio_max": float(np.max([r["singular_ratio"] for r in sub])),
            "op_v_col_norm_median": float(np.median([r["op_v_col_norm"] for r in sub])),
            "op_h_col_norm_median": float(np.median([r["op_h_col_norm"] for r in sub])),
            "mean_ratio": float(np.mean([r["mean_ratio"] for r in sub])),
        }
    return {"rows": rows, "summary": summary,
            "reads": "a composition whose singular ratio is much larger than the "
                     "others has a poorly identified second column; the plan "
                     "forbids reporting that as a content finding"}


def guard_b():
    npz = np.load(WAVE6 / "e1_operators.npz", allow_pickle=False)
    keys = [str(k) for k in npz["cell_keys"]]
    ops = {k: npz["eb"][i] for i, k in enumerate(keys)}
    probes = {str(k): npz["probe_" + str(k)] for k in npz["probe_keys"]}

    def parts(a, b, st):
        delta = a - b
        probe = probes.get(st)
        return {
            "probe": float(np.linalg.norm(delta @ probe)) if probe is not None
            else float(np.linalg.norm(delta)),
            "direction": float(np.linalg.norm(
                a / max(np.linalg.norm(a), 1e-12) - b / max(np.linalg.norm(b), 1e-12))),
            "gain_log": float(abs(np.log(max(np.linalg.norm(a), 1e-12)
                                         / max(np.linalg.norm(b), 1e-12)))),
        }

    own_other, stranger = [], []
    for cell in sorted(ops):
        participant, st = cell.split("|", 1)
        for other in sorted(ops):
            if other == cell:
                continue
            same_person = other.split("|")[0] == participant
            same_cond = other.split("|", 1)[1] == st
            d = parts(ops[cell], ops[other], st)
            d.update({"cell": cell, "other": other, "participant": participant})
            if same_person and not same_cond:
                own_other.append(d)
            elif not same_person and same_cond:
                stranger.append(d)
    near = []
    for cell in sorted({d["cell"] for d in stranger}):
        mine = sorted([d for d in stranger if d["cell"] == cell], key=lambda d: d["probe"])
        near.extend(mine[:max(1, len(mine) // 3)])

    def per_participant(rows, field):
        acc: dict[str, list[float]] = {}
        for r in rows:
            acc.setdefault(r["participant"], []).append(r[field])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    out = {}
    for field in ("probe", "direction", "gain_log"):
        oo, nn = per_participant(own_other, field), per_participant(near, field)
        common = sorted(set(oo) & set(nn))
        out[field] = {
            "own_other_mean": float(np.mean(list(oo.values()))),
            "near_stranger_mean": float(np.mean(list(nn.values()))),
            "own_other_minus_near_stranger": stat([oo[p] - nn[p] for p in common]),
        }
    return {"by_component": out, "n_own_other_pairs": len(own_other),
            "n_near_stranger_pairs": len(near),
            "reads": "if own-other exceeds near-stranger only on gain_log, the "
                     "OWN_OTHER penalty is consistent with the per-cell EOG "
                     "latent scaling and must not be called a condition effect"}


def main() -> None:
    decision = {"guard_a_e5_identifiability": guard_a(),
                "guard_b_e2_own_other_confound": guard_b()}
    WAVE6.mkdir(parents=True, exist_ok=True)
    (WAVE6 / "guards.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"A": decision["guard_a_e5_identifiability"]["summary"],
                      "B": decision["guard_b_e2_own_other_confound"]["by_component"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
