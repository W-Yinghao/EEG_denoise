"""FLAGSHIP-M35 execution CLI. Rules frozen in reports/m35_preregistration.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_chart.analytic import canonical_clean
from eeg_chart.geodesic import rho_eb, transport_family
from eeg_chart.run_m0 import _canon_path, _load_panel, _stat, transport_context
from eeg_chart.run_m13 import PANEL_TRANSPORT


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/flagship_m35"
REPORT = ROOT / "reports"
HARD_GATE_MIN_SECONDS = 60
RHO_GRID = (("0.00", 0.0), ("0.25", 0.25), ("0.50", 0.5), ("0.75", 0.75),
            ("EB", None), ("1.00", 1.0))
SUPPORT_SECONDS = {"klados": 15.0, "bci2b": 300.0}   # nominal; bci2b support >> 60 s


def _anchor_operators(cells):
    """Frozen λ-gated anchor operators per cell (S3c closed form, 2-half within)."""
    pop_by_cell, lam_by_cell = {}, {}
    withins = {cell.cell: float(np.mean(np.square(cell.a_halves[0] - cell.a_halves[1])) / 2.0)
               for cell in cells}
    threshold = float(np.percentile(list(withins.values()), 95))
    for cell in cells:
        others = [c.a_support for c in cells if c.subject != cell.subject]
        pop = np.mean(others, axis=0)
        tau2 = float(np.mean([np.mean(np.square(c.a_support - pop)) for c in cells
                              if c.subject != cell.subject]))
        support_seconds = SUPPORT_SECONDS[cell.panel]
        if support_seconds < HARD_GATE_MIN_SECONDS or withins[cell.cell] > threshold:
            lam = 0.0
        else:
            lam = rho_eb(tau2, withins[cell.cell] * 2.0)   # within of a single fit
        pop_by_cell[cell.cell] = pop
        lam_by_cell[cell.cell] = lam
    return pop_by_cell, lam_by_cell


def u1() -> None:
    from eeg_scad.cli.run_v43 import bootstrap_draws, holm

    canon = np.load(_canon_path())["u_canon"]
    out = {}
    for panel in ("klados", "bci2b"):
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT[panel],
                                    split_half_abstain=True)
        pop_ops, lams = _anchor_operators(cells)
        ordered = sorted(context["per_cell"].items())
        unit_rows: dict[str, dict[str, list[float]]] = {}
        for index, (cell_id, entry) in enumerate(ordered):
            cell = entry["cell"]
            donor = next(other for _, other in ordered[index + 1:] + ordered[:index]
                         if other["cell"].subject != cell.subject)
            transports = {
                "T-POP": transport_family(context["lift"], context["lift_pinv"],
                                          context["sigma_bar"], None, entry["base"],
                                          entry["base"], 0.0),
                "T-MATCH": transport_family(context["lift"], context["lift_pinv"],
                                            context["sigma_bar"], cell.sigma_support,
                                            entry["rotation"], entry["base"], entry["rho"],
                                            whitening=PANEL_TRANSPORT[panel]),
                "T-WRONG": transport_family(context["lift"], context["lift_pinv"],
                                            context["sigma_bar"],
                                            donor["cell"].sigma_support, donor["rotation"],
                                            entry["base"], donor["rho"],
                                            whitening=PANEL_TRANSPORT[panel]),
            }
            c0 = pop_ops[cell.cell]
            c_match = c0 + lams[cell.cell] * (cell.a_support - c0)
            donor_c = pop_ops[cell.cell] + lams[donor["cell"].cell] \
                * (donor["cell"].a_support - pop_ops[cell.cell])
            unit = cell.subject if panel != "klados" else cell.cell
            for episode in cell.episodes:
                anchors = {"A0-none": np.zeros_like(episode["y"]),
                           "A0-POP": c0 @ episode["drive"],
                           "A0-MATCH": c_match @ episode["drive"],
                           "A0-WRONGg": donor_c @ episode["drive"]}
                for t_name, arm in transports.items():
                    for a_name, anchor in anchors.items():
                        if t_name == "T-WRONG" and a_name != "A0-MATCH":
                            continue                       # descriptive row only
                        if a_name == "A0-WRONGg" and t_name != "T-MATCH":
                            continue                       # descriptive row only
                        cleaned = canonical_clean(arm, context["u_canon"],
                                                  context["sigma_bar_inv"],
                                                  episode["y"] - anchor)
                        rrmse = float(np.linalg.norm(cleaned - episode["x"])
                                      / max(np.linalg.norm(episode["x"]), 1e-12))
                        unit_rows.setdefault(f"{t_name}|{a_name}", {}) \
                            .setdefault(unit, []).append(rrmse)
        per = {arm: {u: float(np.mean(v)) for u, v in units.items()}
               for arm, units in unit_rows.items()}
        common = sorted(set.intersection(*(set(v) for v in per.values())))
        def delta(a, b):
            return np.asarray([per[a][u] - per[b][u] for u in common])
        uf1 = delta("T-POP|A0-MATCH", "T-MATCH|A0-MATCH")
        uf2 = delta("T-MATCH|A0-POP", "T-MATCH|A0-MATCH")
        base = np.asarray([per["T-POP|A0-POP"][u] for u in common])
        joint = np.asarray([per["T-MATCH|A0-MATCH"][u] for u in common])
        leg_t = np.asarray([per["T-MATCH|A0-POP"][u] for u in common])
        leg_a = np.asarray([per["T-POP|A0-MATCH"][u] for u in common])
        best_single = np.minimum(leg_t, leg_a)
        uf3 = best_single - joint
        gain_joint = base - joint
        gain_sum = (base - leg_t) + (base - leg_a)
        additivity = float(gain_joint.mean() / gain_sum.mean()) if abs(gain_sum.mean()) > 1e-9 \
            else float("nan")
        rng_draws = {name: bootstrap_draws(values) for name, values in
                     (("UF-1", uf1), ("UF-2", uf2), ("UF-3", uf3))}
        add_draws = []
        rng = np.random.default_rng(420)
        for _ in range(5000):
            pick = rng.integers(0, len(common), len(common))
            gs = gain_sum[pick].mean()
            add_draws.append(gain_joint[pick].mean() / gs if abs(gs) > 1e-9 else np.nan)
        add_draws = np.asarray([v for v in add_draws if np.isfinite(v)])
        out[panel] = {
            "arm_means": {arm: float(np.mean(list(v.values()))) for arm, v in per.items()},
            "UF-1": {**_stat(uf1), "pass": bool(uf1.mean() > 0
                                                and _stat(uf1)["bootstrap_low"] > 0),
                     "p_raw": float(np.mean(rng_draws["UF-1"] <= 0))},
            "UF-2": {**_stat(uf2), "pass": bool(uf2.mean() > 0
                                                and _stat(uf2)["bootstrap_low"] > 0),
                     "p_raw": float(np.mean(rng_draws["UF-2"] <= 0))},
            "UF-3": {**_stat(uf3), "pass": bool(uf3.mean() > 0
                                                and _stat(uf3)["bootstrap_low"] > 0),
                     "p_raw": float(np.mean(rng_draws["UF-3"] <= 0)),
                     "additivity_index": additivity,
                     "additivity_ci": [float(np.quantile(add_draws, .025)),
                                       float(np.quantile(add_draws, .975))]
                     if len(add_draws) else None},
            "descriptive": {"T-WRONG|A0-MATCH_minus_joint":
                            _stat(delta("T-WRONG|A0-MATCH", "T-MATCH|A0-MATCH")),
                            "T-MATCH|A0-WRONGg_minus_joint":
                            _stat(delta("T-MATCH|A0-WRONGg", "T-MATCH|A0-MATCH"))},
            "anchor_lambda_abstained_fraction":
                float(np.mean([lams[c.cell] == 0.0 for c in cells])),
            "units": len(common),
        }
        p_raw = {name: out[panel][name]["p_raw"] for name in ("UF-1", "UF-2", "UF-3")}
        out[panel]["holm"] = {"p_raw": p_raw, "p_adjusted": holm(p_raw)}
    target = RESULT / "u1_factorial"
    target.mkdir(parents=True, exist_ok=True)
    (target / "decision.json").write_text(json.dumps(
        {"preregistration": "reports/m35_preregistration.md", "panels": out,
         "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({panel: {name: out[panel][name]["pass"]
                              for name in ("UF-1", "UF-2", "UF-3")} for panel in out}))


def p1() -> None:
    """Transport-state linkage at the ρ grid (descriptive; two-leg privacy story)."""
    from eeg_chart.transport import minimal_rotation, ordered_frame, spd_power
    from eeg_scad.evaluation.linkage_diagnostic import linkage

    canon = np.load(_canon_path())["u_canon"]
    rows = []
    for panel in ("klados", "bci2b"):
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT[panel],
                                    split_half_abstain=True)
        bar_root = spd_power(context["sigma_bar"], 0.5)
        bar_inv_root = spd_power(context["sigma_bar"], -0.5)

        def state_feature(cell, half_index: int, rho: float):
            a_half = cell.a_halves[half_index]
            sigma_half = cell.sigma_support   # covariance summary shared; frame varies
            whitened = bar_inv_root @ sigma_half @ bar_inv_root
            sigma_rho = bar_root @ spd_power(whitened, rho) @ bar_root
            eigs = np.sort(np.linalg.eigvalsh(sigma_rho))[-20:]
            rotation = minimal_rotation(ordered_frame(lift @ a_half), canon)
            base = context["per_cell"][cell.cell]["base"]
            frame_feat = rho * ((rotation @ base.T - np.eye(len(base))) @ canon).reshape(-1)
            return np.concatenate((np.log(np.clip(eigs, 1e-12, None)) * rho, frame_feat))

        for label, rho_value in RHO_GRID:
            features = {}
            for cell in cells:
                unit = cell.subject if panel != "klados" else cell.cell
                if unit in features:
                    continue
                rho = context["per_cell"][cell.cell]["rho"] if rho_value is None else rho_value
                features[unit] = (state_feature(cell, 0, rho), state_feature(cell, 1, rho))
            metrics = linkage(features)[0]
            rows.append({"panel": panel, "rho_label": label,
                         "rho_mean": float(np.mean([context["per_cell"][c.cell]["rho"]
                                                    for c in cells]))
                         if rho_value is None else rho_value, **metrics})
    target = RESULT / "p1_privacy"
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "transport_state_privacy.csv", index=False)
    (target / "decision.json").write_text(json.dumps(
        {"descriptive_only": True, "rows": rows, "sealed_reads": 0},
        indent=2, sort_keys=True) + "\n")
    print(frame.groupby(["panel", "rho_label"])[["top1_accuracy", "same_different_auroc"]]
          .mean().round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    for name in ("u1", "p1"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.unit == "u1":
        u1()
    elif args.unit == "p1":
        p1()


if __name__ == "__main__":
    main()
