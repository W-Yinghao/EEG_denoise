"""FLAGSHIP-M13 execution CLI. Rules frozen in reports/m13_preregistration.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_chart.analytic import canonical_clean
from eeg_chart.geodesic import transport_family
from eeg_chart.run_m0 import (PANELS, STRATA, _arms, _load_panel, _canon_path, _stat,
                              _strata_masks, transport_context)


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/flagship_m13"
REPORT = ROOT / "reports"
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
KAPPA_TARGET = 100.0


def _panel_probe(cells, context) -> dict[str, Any]:
    """U1-a-style analytic arm probe under the current context settings."""
    masks = _strata_masks(cells)
    ordered = sorted(context["per_cell"].items())
    unit_rows: dict[str, dict[str, dict[str, list[float]]]] = {}
    for index, (cell_id, entry) in enumerate(ordered):
        donor = next(other for _, other in ordered[index + 1:] + ordered[:index]
                     if other["cell"].subject != entry["cell"].subject)
        arms = _arms(context, entry, donor, gauge_seed=97000 + index)
        unit = entry["cell"].subject if entry["cell"].panel != "klados" else entry["cell"].cell
        for episode in entry["cell"].episodes:
            flags = masks(episode)
            for arm_name, arm in arms.items():
                cleaned = canonical_clean(arm, context["u_canon"], context["sigma_bar_inv"],
                                          episode["y"])
                rrmse = float(np.linalg.norm(cleaned - episode["x"])
                              / max(np.linalg.norm(episode["x"]), 1e-12))
                for stratum, flag in flags.items():
                    if flag:
                        unit_rows.setdefault(stratum, {}).setdefault(arm_name, {}) \
                            .setdefault(unit, []).append(rrmse)
    out = {}
    for stratum in STRATA:
        arms_mean = {arm: {unit: float(np.mean(values)) for unit, values in units.items()}
                     for arm, units in unit_rows.get(stratum, {}).items()}
        common = sorted(set.intersection(*(set(v) for v in arms_mean.values())))
        out[stratum] = {
            "arm_means": {arm: float(np.mean(list(v.values()))) for arm, v in arms_mean.items()},
            "ceiling_pop_minus_oracle": _stat([arms_mean["T-POP"][u] - arms_mean["T-ORACLE"][u]
                                               for u in common]),
            "deployable_match_minus_pop_gain": _stat([arms_mean["T-POP"][u] - arms_mean["T-MATCH"][u]
                                                      for u in common]),
            "gauge_null_minus_pop": _stat([arms_mean["GAUGE-NULL"][u] - arms_mean["T-POP"][u]
                                           for u in common]),
            "wrong_minus_pop": _stat([arms_mean["T-WRONG"][u] - arms_mean["T-POP"][u]
                                      for u in common]),
            "units": len(common)}
    return out


def w1() -> None:
    canon = np.load(_canon_path())["u_canon"]
    target = RESULT / "w1_repair"
    target.mkdir(parents=True, exist_ok=True)
    geometry, probes = {}, {}
    for panel in PANELS:
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening="truncated",
                                    split_half_abstain=True)
        roundtrip, kappas, angles = [], [], []
        identity_ok = True
        for cell_id, entry in sorted(context["per_cell"].items()):
            cell = entry["cell"]
            for rho in (0.0, entry["rho"], 1.0):
                arm = transport_family(context["lift"], context["lift_pinv"],
                                       context["sigma_bar"], cell.sigma_support,
                                       entry["rotation"], entry["base"], rho,
                                       whitening="truncated")
                roundtrip.append(float(np.max(np.abs(arm.pinv @ arm.transport
                                                     - np.eye(arm.transport.shape[1])))))
                kappas.append(float(np.linalg.cond(arm.transport)))
            pop_arm = transport_family(context["lift"], context["lift_pinv"],
                                       context["sigma_bar"], None, entry["base"],
                                       entry["base"], 0.0)
            zero_arm = transport_family(context["lift"], context["lift_pinv"],
                                        context["sigma_bar"], cell.sigma_support,
                                        entry["rotation"], entry["base"], 0.0,
                                        whitening="truncated")
            identity_ok &= bool(np.array_equal(zero_arm.transport, pop_arm.transport)
                                and np.array_equal(zero_arm.pinv, pop_arm.pinv))
            match_arm = transport_family(context["lift"], context["lift_pinv"],
                                         context["sigma_bar"], cell.sigma_support,
                                         entry["rotation"], entry["base"], entry["rho"],
                                         whitening="truncated")
            transported = match_arm.transport @ cell.a_query
            from eeg_chart.transport import ordered_frame
            angle = np.degrees(np.arccos(np.clip(np.linalg.svd(
                ordered_frame(transported).T @ context["u_canon"], compute_uv=False),
                -1, 1))).max()
            angles.append(float(angle))
        within = [entry["split_half_distance"] for entry in context["per_cell"].values()]
        between = [entry["cohort_distance"] for entry in context["per_cell"].values()]
        geometry[panel] = {
            "cells": len(context["per_cell"]),
            "roundtrip_max": float(np.max(roundtrip)),
            "roundtrip_gate": bool(np.max(roundtrip) <= 1e-10),
            "rho0_bit_identity": bool(identity_ok),
            "kappa_max": float(np.max(kappas)), "kappa_median": float(np.median(kappas)),
            "kappa_within_target_fraction": float(np.mean(np.asarray(kappas)
                                                          <= KAPPA_TARGET * 50)),
            "frame_angle_within_15deg_fraction": float(np.mean(np.asarray(angles) <= 15.0)),
            "frame_angle_p50": float(np.median(angles)),
            "abstentions": int(sum(entry["abstained"] for entry in context["per_cell"].values())),
            "rho_mean_nonabstained": float(np.mean([entry["rho"] for entry
                                                    in context["per_cell"].values()
                                                    if not entry["abstained"]] or [0.0])),
            "split_half_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "diagnosis": ("estimation_noise_dominated" if np.median(within) >= np.median(between)
                          else "heterogeneity_dominated"),
        }
        probes[panel] = _panel_probe(cells, context)
    (target / "geometry.json").write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n")
    (target / "u1a_rerun.json").write_text(json.dumps(probes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({panel: {"kappa_max": geometry[panel]["kappa_max"],
                              "abstentions": geometry[panel]["abstentions"],
                              "deployable_all": probes[panel]["all"]
                              ["deployable_match_minus_pop_gain"]["mean"]}
                      for panel in PANELS}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    sub.add_parser("w1")
    args = parser.parse_args()
    if args.unit == "w1":
        w1()


if __name__ == "__main__":
    main()
