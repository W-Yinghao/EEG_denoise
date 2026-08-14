"""FLAGSHIP-M0 execution CLI: U0-a/b/c preconditions and the U1 ceiling probes.

Adjudication rules are frozen in reports/m0_preregistration.md.  V42R/V43/V44
artifacts are consumed read-only.  No canonical-prior training happens here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_chart.analytic import canonical_clean, gauge_null_rotation
from eeg_chart.geodesic import (ANGLE_CAP, TransportArm, max_principal_angle, rho_eb,
                                rotation_distance, transport_family)
from eeg_chart.panels import (PanelCell, build_bci2b_panel, build_klados_panel,
                              build_mobilebci_panel)
from eeg_chart.transport import (airm_frechet_mean, minimal_rotation, ordered_frame,
                                 orth, spd_power)


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/flagship_m0"
REPORT = ROOT / "reports"
V44_S0_DECISION = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44/"
                       "stage0/decision.json")
V42R_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_saddpm_cond_v42r")
PANELS = ("mobilebci", "klados", "bci2b")
STRATA = ("all", "high_severity", "high_eog")
GO_MEAN, GO_CI_LOW = 0.020, 0.005
KAPPA_CAP = 1e3
CONDITIONING_BANKED = {"mean": 0.006267, "bootstrap_low": -0.001607, "bootstrap_high": 0.019680,
                       "panel": "mobilebci", "source": "V42R ORACLE-MATCH (banked, not re-measured)"}


def _stat(values) -> dict[str, Any]:
    from eeg_scad.cli.run_v43 import bootstrap_draws

    series = np.asarray(list(values), float)
    draws = bootstrap_draws(series)
    return {"mean": float(series.mean()), "median": float(np.median(series)),
            "positive_count": int((series > 0).sum()), "n": int(len(series)),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def _load_panel(name: str) -> tuple[list[PanelCell], np.ndarray]:
    return {"mobilebci": build_mobilebci_panel, "klados": build_klados_panel,
            "bci2b": build_bci2b_panel}[name]()


def _canon_path() -> Path:
    return RESULT / "u_canon.npz"


def transport_context(cells: list[PanelCell], lift: np.ndarray,
                      u_canon: np.ndarray | None) -> dict[str, Any]:
    lift_pinv = np.linalg.pinv(lift)
    sigma_bar = airm_frechet_mean([cell.sigma_support for cell in cells])
    if u_canon is None:  # the MobileBCI dev cohort defines the global canonical frame
        u_canon = ordered_frame(lift @ np.mean([cell.a_support for cell in cells], axis=0))
    bar_root = spd_power(sigma_bar, 0.5)
    per_cell: dict[str, dict[str, Any]] = {}
    for cell in cells:
        others = [c.a_support for c in cells if c.subject != cell.subject]
        a_pop = np.mean(others, axis=0) if others else np.mean([c.a_support for c in cells], axis=0)
        base = minimal_rotation(ordered_frame(lift @ a_pop), u_canon)
        align_full = bar_root @ spd_power(cell.sigma_support, -0.5)
        rotation = minimal_rotation(ordered_frame(align_full @ lift @ cell.a_support), u_canon)
        halves = tuple(minimal_rotation(ordered_frame(align_full @ lift @ half), u_canon)
                       for half in cell.a_halves)
        align_query = bar_root @ spd_power(cell.sigma_query, -0.5)
        rotation_oracle = minimal_rotation(ordered_frame(align_query @ lift @ cell.a_query), u_canon)
        per_cell[cell.cell] = {"cell": cell, "a_pop": a_pop, "base": base, "rotation": rotation,
                               "halves": halves, "rotation_oracle": rotation_oracle}
    tau2 = float(np.mean([rotation_distance(entry["rotation"], entry["base"]) ** 2
                          for entry in per_cell.values()]))
    for entry in per_cell.values():
        v_s = rotation_distance(entry["halves"][0], entry["halves"][1]) ** 2
        angle = max_principal_angle(entry["rotation"], entry["base"])
        entry["abstained"] = bool(angle > ANGLE_CAP)
        entry["rho"] = 0.0 if entry["abstained"] else rho_eb(tau2, v_s)
        entry["split_half_distance"] = float(np.sqrt(v_s))
        entry["cohort_distance"] = rotation_distance(entry["rotation"], entry["base"])
        entry["max_angle"] = float(angle)
    return {"lift": lift, "lift_pinv": lift_pinv, "sigma_bar": sigma_bar,
            "sigma_bar_inv": spd_power(sigma_bar, -1.0), "u_canon": u_canon,
            "per_cell": per_cell, "tau2": tau2}


def _arms(context: dict[str, Any], entry: dict[str, Any], donor: dict[str, Any],
          gauge_seed: int) -> dict[str, TransportArm]:
    lift, lift_pinv = context["lift"], context["lift_pinv"]
    sigma_bar = context["sigma_bar"]
    cell: PanelCell = entry["cell"]
    arms = {
        "T-POP": transport_family(lift, lift_pinv, sigma_bar, None,
                                  entry["base"], entry["base"], 0.0),
        "T-MATCH": transport_family(lift, lift_pinv, sigma_bar, cell.sigma_support,
                                    entry["rotation"], entry["base"], entry["rho"]),
        "T-ORACLE": transport_family(lift, lift_pinv, sigma_bar, cell.sigma_query,
                                     entry["rotation_oracle"], entry["base"], 1.0),
        "T-WRONG": transport_family(lift, lift_pinv, sigma_bar, donor["cell"].sigma_support,
                                    donor["rotation"], entry["base"], donor["rho"]),
    }
    match = arms["T-MATCH"]
    if match.rho > 0.0:
        gauge_rotation = gauge_null_rotation(match.rotation, entry["base"], gauge_seed)
        arms["GAUGE-NULL"] = TransportArm(
            match.rho, gauge_rotation @ match.align @ lift,
            lift_pinv @ np.linalg.inv(match.align) @ gauge_rotation.T,
            gauge_rotation, match.align, False)
    else:
        arms["GAUGE-NULL"] = arms["T-POP"]
    return arms


def _strata_masks(cells: list[PanelCell]):
    severities = [e["severity"] for cell in cells for e in cell.episodes if e["severity"] > 1e-6]
    energies = [e["eog_energy"] for cell in cells for e in cell.episodes]
    severity_cut = float(np.quantile(severities, 2 / 3)) if severities else np.inf
    energy_cut = float(np.quantile(energies, 2 / 3)) if energies else np.inf

    def masks(episode):
        return {"all": True,
                "high_severity": episode["severity"] >= severity_cut,
                "high_eog": episode["eog_energy"] >= energy_cut}
    return masks


# ---------------------------------------------------------------------- U0-a

def u0a() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    from eeg_chart.atlas import sgeyesub_layout_rows

    payload: dict[str, Any] = {"panels": {}, "preregistration": "reports/m0_preregistration.md"}
    u_canon = None
    contexts = {}
    for panel in PANELS:
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, u_canon)
        if panel == "mobilebci":
            u_canon = context["u_canon"]
            np.savez(_canon_path(), u_canon=u_canon)
        contexts[panel] = (cells, context)
        rng = np.random.default_rng(20260814)
        roundtrip, identity_ok, locality, angles, kappas = [], True, [], [], []
        for index, (cell_id, entry) in enumerate(sorted(context["per_cell"].items())):
            cell: PanelCell = entry["cell"]
            for rho in (0.0, entry["rho"], 1.0):
                arm = transport_family(context["lift"], context["lift_pinv"], context["sigma_bar"],
                                       cell.sigma_support, entry["rotation"], entry["base"], rho)
                roundtrip.append(float(np.max(np.abs(arm.pinv @ arm.transport
                                                     - np.eye(arm.transport.shape[1])))))
                kappas.append(float(np.linalg.cond(arm.transport)))
            pop_arm = transport_family(context["lift"], context["lift_pinv"], context["sigma_bar"],
                                       None, entry["base"], entry["base"], 0.0)
            zero_arm = transport_family(context["lift"], context["lift_pinv"], context["sigma_bar"],
                                        cell.sigma_support, entry["rotation"], entry["base"], 0.0)
            identity_ok &= bool(np.array_equal(zero_arm.transport, pop_arm.transport)
                                and np.array_equal(zero_arm.pinv, pop_arm.pinv))
            if cell.episodes and identity_ok:
                y = cell.episodes[0]["y"]
                identity_ok &= bool(np.array_equal(
                    canonical_clean(zero_arm, context["u_canon"], context["sigma_bar_inv"], y),
                    canonical_clean(pop_arm, context["u_canon"], context["sigma_bar_inv"], y)))
            match_arm = transport_family(context["lift"], context["lift_pinv"], context["sigma_bar"],
                                         cell.sigma_support, entry["rotation"], entry["base"],
                                         entry["rho"])
            union = orth(np.concatenate((
                match_arm.rotation @ entry["base"].T @ context["u_canon"],
                context["u_canon"], entry["base"] @ np.zeros_like(context["u_canon"])), axis=1))
            # locality: deviation of Q(rho) from base restricted to the complement of
            # the union of the involved ocular frames
            frames = np.concatenate((entry["rotation"].T @ context["u_canon"],
                                     entry["base"].T @ context["u_canon"],
                                     context["u_canon"]), axis=1)
            union = orth(frames)
            projector = np.eye(len(union)) - union @ union.T
            deviation = (match_arm.rotation @ entry["base"].T - np.eye(len(union))) @ projector
            locality.append(float(np.max(np.abs(deviation))))
            transported = match_arm.transport @ cell.a_query
            angle = np.degrees(np.arccos(np.clip(np.linalg.svd(
                ordered_frame(transported).T @ context["u_canon"], compute_uv=False), -1, 1))).max()
            angles.append(float(angle))
        within = [entry["split_half_distance"] for entry in context["per_cell"].values()]
        between = [entry["cohort_distance"] for entry in context["per_cell"].values()]
        payload["panels"][panel] = {
            "cells": len(context["per_cell"]),
            "roundtrip_max": float(np.max(roundtrip)), "roundtrip_gate": bool(np.max(roundtrip) <= 1e-10),
            "rho0_bit_identity": bool(identity_ok),
            "locality_max": float(np.max(locality)), "locality_gate": bool(np.max(locality) <= 1e-10),
            "frame_angle_deg_p50": float(np.median(angles)),
            "frame_angle_within_15deg_fraction": float(np.mean(np.asarray(angles) <= 15.0)),
            "frame_gate": bool(np.mean(np.asarray(angles) <= 15.0) >= 0.80),
            "split_half_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "split_half_lt_between": bool(np.median(within) < np.median(between)),
            "kappa_max": float(np.max(kappas)), "kappa_flagged": int(np.sum(np.asarray(kappas) > KAPPA_CAP)),
            "abstentions": int(sum(entry["abstained"] for entry in context["per_cell"].values())),
            "rho_mean": float(np.mean([entry["rho"] for entry in context["per_cell"].values()])),
            "tau2": context["tau2"],
        }
    payload["sgeyesub_stress"] = sgeyesub_layout_rows(ROOT)
    core_gates = ["roundtrip_gate", "rho0_bit_identity", "locality_gate", "frame_gate",
                  "split_half_lt_between"]
    payload["passed"] = bool(all(payload["panels"][panel][gate]
                                 for panel in PANELS for gate in core_gates))
    (RESULT / "u0a_geometry.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                         default=str) + "\n")
    (REPORT / "m0_u0a.md").write_text(
        "# M0 U0-a — transport geometry gates\n\nProp-5' corrected family; "
        f"gates fail-closed per preregistration. Overall passed: **{payload['passed']}**.\n\n```json\n"
        + json.dumps({panel: payload["panels"][panel] for panel in PANELS},
                     indent=2, sort_keys=True) + "\n```\n\nSGEYESUB montage stress (lift-level):\n\n```json\n"
        + json.dumps(payload["sgeyesub_stress"], indent=2, default=str) + "\n```\n")
    print(json.dumps({"passed": payload["passed"],
                      **{panel: {gate: payload["panels"][panel][gate] for gate in core_gates}
                         for panel in PANELS}}))


# ---------------------------------------------------------------------- U0-b

def u0b() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_chart.posterior import block_coverage

    data, folds, _ = configs()
    fold_rows = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        fits = {key: eb120._fit(key, 12000, 100) for key in registry30.cells}
        coverages: dict[str, list[float]] = {}
        for group in registry30.population_transfer:
            train_full = np.stack([fits[key][0] for key in sorted(fits)
                                   if key[0] in fold["train"] and key[1:] == group])
            pop = registry30.population_transfer[group]
            tau2 = train_full.var(axis=0, ddof=1).clip(1e-12)
            for key in sorted(fits):
                if key[1:] != group:
                    continue
                coverage = block_coverage(pop, tau2, fits[key][1])
                coverages.setdefault(key[0], []).append(coverage)
        participant_first = {participant: float(np.mean(values))
                             for participant, values in coverages.items()}
        mean_coverage = float(np.mean(list(participant_first.values())))
        fold_rows.append({"fold": fold["fold"], "coverage_mean": mean_coverage,
                          "participants": len(participant_first),
                          "in_band": bool(0.70 <= mean_coverage <= 0.90),
                          "per_participant": participant_first})
    payload = {"nominal": 0.80, "band": [0.70, 0.90], "folds": fold_rows,
               "passed": bool(all(row["in_band"] for row in fold_rows))}
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "u0b_coverage.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": payload["passed"],
                      "fold_coverage": [row["coverage_mean"] for row in fold_rows]}))


# ---------------------------------------------------------------------- U0-c

def u0c() -> None:
    from eeg_chart.atlas import write_atlas

    all_cells = []
    for panel in PANELS:
        cells, _ = _load_panel(panel)
        all_cells.extend(cells)
    frame = write_atlas(all_cells, ROOT, RESULT / "atlas")
    print(json.dumps({"atlas_rows": int(len(frame)),
                      "datasets": sorted(frame.dataset.unique().tolist())}))


# ---------------------------------------------------------------------- U1-a

def u1a() -> None:
    canon = np.load(_canon_path())["u_canon"] if _canon_path().is_file() else None
    results = {}
    for panel in PANELS:
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, None if panel == "mobilebci" else canon)
        if panel == "mobilebci" and canon is None:
            canon = context["u_canon"]
            np.savez(_canon_path(), u_canon=canon)
        masks = _strata_masks(cells)
        ordered = sorted(context["per_cell"].items())
        unit_rows: dict[str, dict[str, dict[str, list[float]]]] = {}
        for index, (cell_id, entry) in enumerate(ordered):
            donor = next(other for _, other in ordered[index + 1:] + ordered[:index]
                         if other["cell"].subject != entry["cell"].subject)
            arms = _arms(context, entry, donor, gauge_seed=97000 + index)
            unit = entry["cell"].subject if panel != "klados" else entry["cell"].cell
            for episode in entry["cell"].episodes:
                stratum_flags = masks(episode)
                for arm_name, arm in arms.items():
                    cleaned = canonical_clean(arm, context["u_canon"], context["sigma_bar_inv"],
                                              episode["y"])
                    rrmse = float(np.linalg.norm(cleaned - episode["x"])
                                  / max(np.linalg.norm(episode["x"]), 1e-12))
                    for stratum, flag in stratum_flags.items():
                        if flag:
                            unit_rows.setdefault(stratum, {}).setdefault(arm_name, {}) \
                                .setdefault(unit, []).append(rrmse)
        panel_out = {}
        for stratum in STRATA:
            arms_mean = {arm: {unit: float(np.mean(values)) for unit, values in units.items()}
                         for arm, units in unit_rows.get(stratum, {}).items()}
            common = sorted(set.intersection(*(set(v) for v in arms_mean.values())))
            ceiling = [arms_mean["T-POP"][u] - arms_mean["T-ORACLE"][u] for u in common]
            deploy = [arms_mean["T-POP"][u] - arms_mean["T-MATCH"][u] for u in common]
            gauge = [arms_mean["GAUGE-NULL"][u] - arms_mean["T-POP"][u] for u in common]
            wrong = [arms_mean["T-WRONG"][u] - arms_mean["T-POP"][u] for u in common]
            panel_out[stratum] = {
                "arm_means": {arm: float(np.mean(list(values.values())))
                              for arm, values in arms_mean.items()},
                "ceiling_pop_minus_oracle": _stat(ceiling),
                "deployable_match_minus_pop_gain": _stat(deploy),
                "gauge_null_minus_pop": _stat(gauge),
                "wrong_minus_pop": _stat(wrong), "units": len(common)}
        results[panel] = panel_out
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "u1a_transport_ceiling.json").write_text(json.dumps(results, indent=2,
                                                                  sort_keys=True) + "\n")
    print(json.dumps({panel: results[panel]["all"]["ceiling_pop_minus_oracle"]["mean"]
                      for panel in PANELS}))


# ---------------------------------------------------------------------- U1-b

def u1b() -> None:
    consumed = json.loads(V44_S0_DECISION.read_text()) if V44_S0_DECISION.is_file() else None
    results: dict[str, Any] = {"consumed_v44_s0": {
        "path": str(V44_S0_DECISION), "present": consumed is not None,
        "G0-1": consumed["G0-1"] if consumed else None,
        "note": "V44-S0 gated-vs-population subtraction on MobileBCI; oracle row degenerate "
                "on generated pairs"} if True else None}
    for panel in PANELS:
        cells, _ = _load_panel(panel)
        masks = _strata_masks(cells)
        unit_rows: dict[str, dict[str, dict[str, list[float]]]] = {}
        pop_ops = {}
        for cell in cells:
            others = [c.a_support for c in cells if c.subject != cell.subject]
            pop_ops[cell.cell] = np.mean(others, axis=0) if others else cell.a_support
        for cell in cells:
            unit = cell.subject if panel != "klados" else cell.cell
            arms = {"C0": pop_ops[cell.cell], "C-MATCH": cell.a_support, "C-ORACLE": cell.a_query}
            for episode in cell.episodes:
                flags = masks(episode)
                for arm_name, operator in arms.items():
                    cleaned = episode["y"] - operator @ episode["drive"]
                    rrmse = float(np.linalg.norm(cleaned - episode["x"])
                                  / max(np.linalg.norm(episode["x"]), 1e-12))
                    for stratum, flag in flags.items():
                        if flag:
                            unit_rows.setdefault(stratum, {}).setdefault(arm_name, {}) \
                                .setdefault(unit, []).append(rrmse)
        panel_out = {}
        for stratum in STRATA:
            arms_mean = {arm: {unit: float(np.mean(values)) for unit, values in units.items()}
                         for arm, units in unit_rows.get(stratum, {}).items()}
            common = sorted(set.intersection(*(set(v) for v in arms_mean.values())))
            ceiling = [arms_mean["C0"][u] - arms_mean["C-ORACLE"][u] for u in common]
            deploy = [arms_mean["C0"][u] - arms_mean["C-MATCH"][u] for u in common]
            panel_out[stratum] = {"arm_means": {arm: float(np.mean(list(v.values())))
                                                for arm, v in arms_mean.items()},
                                  "ceiling_c0_minus_oracle": _stat(ceiling),
                                  "deployable_c0_minus_match_gain": _stat(deploy),
                                  "units": len(common)}
        results[panel] = panel_out
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "u1b_likelihood_ceiling.json").write_text(json.dumps(results, indent=2,
                                                                   sort_keys=True) + "\n")
    print(json.dumps({panel: results[panel]["all"]["ceiling_c0_minus_oracle"]["mean"]
                      for panel in PANELS}))


# ---------------------------------------------------------------------- U1-c

def u1c(fold_id: int, updates: int = 2000) -> None:
    import torch
    import torch.nn.functional as F
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule
    from eeg_scad.training.train_v42r import _conditions, sample_bank
    from eeg_chart.lora_probe import inject_score_lora, lora_parameters

    seed = 20261201
    out_dir = RESULT / "u1c"
    out_path = out_dir / f"fold_{fold_id}.json"
    if out_path.is_file():
        print(json.dumps({"fold": fold_id, "skipped": "already complete"}))
        return
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    test_sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = test_sampler.sample_balanced(8)
    checkpoint = V42R_DERIVED / "job_941770" / f"fold_{fold_id}_seed_{seed}" / "best.pt"
    frozen = json.loads((ROOT / "results/rgcc_v43/stage1" / f"fold_{fold_id}_seed_{seed}"
                         / "stage1_result.json").read_text())
    pop_rows = [row for row in frozen["rows"] if row["condition"] == "POP"]
    baseline: dict[str, list[float]] = {}
    for row in pop_rows:
        baseline.setdefault(row["participant"], []).append(row["rrmse_temporal"])
    schedule = LinearX0Schedule().to(device)
    pop_signature, _ = _conditions(test_sampler, bank["meta"], "POP")
    subjects = sorted(set(meta["participant"] for meta in bank["meta"]))
    results = {}
    for subject in subjects:
        torch.manual_seed(seed + fold_id)
        model = CalibSADDPMCond().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["ema"])
        summary = inject_score_lora(model, rank=4)
        model.train()
        tune_sampler = TransferEpisodeSampler(data, fold, "test", seed + 9, registry30)
        keys = sorted(key for key in registry30.cells
                      if key[0] == subject and key[1:] in registry30.population_transfer)
        optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(seed + 7001 + fold_id)
        for step in range(updates):
            tune_bank = tune_sampler.sample(8, recipient_keys=[keys[step % len(keys)]] * 8)
            signature, _ = _conditions(tune_sampler, tune_bank["meta"], "POP")
            clean = torch.from_numpy(np.asarray(tune_bank["x"], np.float32)).to(device)
            observed = torch.from_numpy(np.asarray(tune_bank["y"], np.float32)).to(device)
            condition = torch.from_numpy(np.asarray(signature, np.float32)).to(device)
            noisy, timestep, _ = schedule.forward_sample(clean, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(noisy, observed, timestep, condition), clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite U1-c loss at step {step}")
            loss.backward()
            optimizer.step()
        model.eval()
        output = sample_bank(model, schedule, bank["y"], pop_signature, device,
                             420000 + fold_id * 100 + seed % 100)
        adapted = []
        for clean, prediction, meta in zip(bank["x"], output, bank["meta"]):
            if meta["participant"] != subject:
                continue
            adapted.append(float(np.linalg.norm(prediction - clean)
                                 / max(np.linalg.norm(clean), 1e-12)))
        results[subject] = {
            "baseline_pop_rrmse": float(np.mean(baseline[subject])),
            "oracle_lora_rrmse": float(np.mean(adapted)),
            "ceiling_pop_minus_lora": float(np.mean(baseline[subject]) - np.mean(adapted)),
            "adapted_convolutions": summary.adapted_convolutions,
            "trainable_parameters": summary.trainable_parameters,
            "updates": updates, "nondeployable": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fold": fold_id, "seed": seed, "subjects": results,
                                    "supervision": "ORACLE_operator_synthesized_generative_truth",
                                    "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id,
                      "ceilings": {s: r["ceiling_pop_minus_lora"] for s, r in results.items()}}))


# ------------------------------------------------------------------- decision

def decision() -> None:
    u0a_payload = json.loads((RESULT / "u0a_geometry.json").read_text())
    u0b_payload = json.loads((RESULT / "u0b_coverage.json").read_text())
    u1a_payload = json.loads((RESULT / "u1a_transport_ceiling.json").read_text())
    u1b_payload = json.loads((RESULT / "u1b_likelihood_ceiling.json").read_text())
    lora = {}
    for fold_id in range(5):
        path = RESULT / "u1c" / f"fold_{fold_id}.json"
        if path.is_file():
            for subject, row in json.loads(path.read_text())["subjects"].items():
                lora[subject] = row["ceiling_pop_minus_lora"]
    rows = []
    go_flags = {}

    def add(channel, panel, stratum, stat, note=""):
        if stat is None:
            rows.append({"channel": channel, "panel": panel, "stratum": stratum,
                         "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                         "go": False, "note": note or "not measured"})
            return
        go = bool(stat["mean"] >= GO_MEAN and stat["bootstrap_low"] > GO_CI_LOW)
        go_flags[(channel, panel, stratum)] = go
        rows.append({"channel": channel, "panel": panel, "stratum": stratum,
                     "mean": stat["mean"], "ci_low": stat["bootstrap_low"],
                     "ci_high": stat["bootstrap_high"], "go": go, "note": note})

    add("conditioning", "mobilebci", "all",
        {"mean": CONDITIONING_BANKED["mean"], "bootstrap_low": CONDITIONING_BANKED["bootstrap_low"],
         "bootstrap_high": CONDITIONING_BANKED["bootstrap_high"]},
        CONDITIONING_BANKED["source"])
    for panel in PANELS:
        for stratum in STRATA:
            add("transport", panel, stratum, u1a_payload[panel][stratum]["ceiling_pop_minus_oracle"])
            add("likelihood", panel, stratum, u1b_payload[panel][stratum]["ceiling_c0_minus_oracle"],
                "oracle operator on generated/near-exact pairs; see degeneracy note")
    if lora:
        add("weight_space", "mobilebci", "all", _stat(list(lora.values())),
            "ORACLE-LoRA, generative-truth supervision, non-deployable")
    matrix = pd.DataFrame(rows)
    matrix.to_csv(RESULT / "ceiling_matrix.csv", index=False)
    k1 = not any(go_flags.values())
    payload = {
        "preregistration": "reports/m0_preregistration.md",
        "go_flags": {f"{c}|{p}|{s}": v for (c, p, s), v in go_flags.items()},
        "K1_fired": bool(k1),
        "u0a_geometry_passed": u0a_payload["passed"],
        "u0b_coverage_passed": u0b_payload["passed"],
        "transport_deployable_effect": {panel: u1a_payload[panel]["all"]
                                        ["deployable_match_minus_pop_gain"] for panel in PANELS},
        "gauge_null": {panel: u1a_payload[panel]["all"]["gauge_null_minus_pop"]
                       for panel in PANELS},
        "weight_space_per_subject": lora,
        "consumed": {"v44_s0": u1b_payload.get("consumed_v44_s0"),
                     "v43_s2": "read-only; not required for the matrix"},
        "sealed_reads": 0,
    }
    (RESULT / "decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (REPORT / "m0_report.md").write_text(
        "# FLAGSHIP-M0 — channel-ceiling matrix\n\n"
        f"K1 fired: **{k1}**. U0-a geometry passed: **{u0a_payload['passed']}**; "
        f"U0-b coverage passed: **{u0b_payload['passed']}**.\n\n"
        "GO rule per cell: mean >= +0.020 and bootstrap CI-low > +0.005.\n\n"
        + matrix.round(6).to_markdown(index=False)
        + "\n\n## GAUGE-NULL rows (must not GO)\n\n```json\n"
        + json.dumps(payload["gauge_null"], indent=2, sort_keys=True) + "\n```\n\n"
        "## Transport deployable effect (T-MATCH - T-POP, descriptive)\n\n```json\n"
        + json.dumps(payload["transport_deployable_effect"], indent=2, sort_keys=True) + "\n```\n\n"
        "## Weight-space per-subject ceilings (non-deployable supervision)\n\n```json\n"
        + json.dumps(lora, indent=2, sort_keys=True) + "\n```\n")
    print(json.dumps({"K1_fired": k1, "go_flags": payload["go_flags"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    for unit in ("u0a", "u0b", "u0c", "u1a", "u1b", "decision"):
        sub.add_parser(unit)
    lora_parser = sub.add_parser("u1c")
    lora_parser.add_argument("--fold", type=int, required=True)
    lora_parser.add_argument("--updates", type=int, default=2000)
    args = parser.parse_args()
    if args.unit == "u1c":
        u1c(args.fold, args.updates)
    else:
        {"u0a": u0a, "u0b": u0b, "u0c": u0c, "u1a": u1a, "u1b": u1b,
         "decision": decision}[args.unit]()


if __name__ == "__main__":
    main()
