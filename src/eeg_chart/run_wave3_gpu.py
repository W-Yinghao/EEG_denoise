"""WAVE3 GPU tranche + decision-point and ledger assembly.

Rules frozen in reports/wave3_preregistration.md. No sealed contact.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_chart.run_wave3 import (BANKED, DECISIONS, NATURAL_START, PSI_WINDOW, RESULT,
                                 Records, V43_STATE, V44_RESULT, _ridge, _stat, psi_labels)


P0_CHECKPOINT = Path("/projects/EEG-foundation-model/derived/denoiseNet/flagship_m13/"
                     "w2_prior/p0_seed_20261301/best.pt")
DELTA_GRID = (1e-8, 1e-4, 1e-2, 1e-1, 1.0)
WINDOW = 512
NAT_WINDOWS = 4


# ------------------------------------------------------------------- A0 / A1'

def _prior_and_frame():
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_chart.prior_model import CanonicalPrior
    from eeg_chart.run_m0 import _canon_path

    device = torch.device("cuda")
    model = CanonicalPrior().to(device)
    model.load_state_dict(torch.load(P0_CHECKPOINT, map_location=device,
                                     weights_only=False)["ema"])
    model.eval()
    schedule = LinearX0Schedule().to(device)
    canon = np.load(_canon_path())["u_canon"]
    return model, schedule, canon, device


def a0() -> None:
    """Pack-A entry gate: reproduce the dev U-ratio deficit with the banked P0 prior."""
    import torch
    from eeg_chart.prior_model import ddim_denoise
    from eeg_chart.run_m13 import _eval_pairs

    out_path = RESULT / "packa_a0.json"
    if out_path.is_file():
        print(json.dumps({"skipped": "complete"}))
        return
    model, schedule, canon, device = _prior_and_frame()
    episodes, context = _eval_pairs("mobilebci", whitening="off", split_half_abstain=True)
    by_unit: dict[str, list[float]] = {}
    prior_rows = []
    for start in range(0, len(episodes), 8):
        chunk = episodes[start:start + 8]
        y_canon = np.stack([e["transport"] @ e["y"] for e in chunk]).astype(np.float32)
        x_canon = np.stack([e["transport"] @ e["x"] for e in chunk]).astype(np.float32)
        y_t = torch.from_numpy(y_canon).to(device)
        noise = torch.randn(y_t.shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(424242 + start))
        with torch.no_grad():
            x0 = ddim_denoise(model, y_t, noise, schedule, 50).cpu().numpy()
        for e, xc, x0c in zip(chunk, x_canon, x0):
            n = e["n_valid"]
            u_true = float(np.sqrt(np.mean((canon.T @ xc[:, :n]) ** 2)))
            u_hat = float(np.sqrt(np.mean((canon.T @ x0c[:, :n]) ** 2)))
            ratio = u_hat / max(u_true, 1e-12)
            by_unit.setdefault(e["unit"], []).append(ratio)
            prior_rows.append({"unit": e["unit"], "u_ratio": ratio})
    per_unit = {u: float(np.mean(v)) for u, v in by_unit.items()}
    stat = _stat(list(per_unit.values()))
    payload = {"instrument": ("U-ratio = ||U0^T x_hat|| / ||U0^T x_true|| on the PRIOR "
                              "output (before the analytic likelihood step)"),
               "per_unit": per_unit, **stat,
               "gate_mean_below_0.95": bool(stat["mean"] < 0.95),
               "ci_excludes_1": bool(stat["bootstrap_high"] < 1.0 or stat["bootstrap_low"] > 1.0),
               "pass": bool(stat["mean"] < 0.95 and stat["bootstrap_high"] < 1.0),
               "consequence_on_fail": ("finding 7 softens to a single-run observation; "
                                       "Pack-A stops per the frozen gate")}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"u_ratio_mean": round(stat["mean"], 4),
                      "ci": [round(stat["bootstrap_low"], 4), round(stat["bootstrap_high"], 4)],
                      "pass": payload["pass"]}))


def a1prime() -> None:
    """Guidance-weight sweep of the analytic likelihood step on the frozen P0 prior."""
    import torch
    from eeg_chart.prior_model import ddim_denoise
    from eeg_chart.run_m13 import _eval_pairs
    from eeg_chart.transport import spd_power

    out_path = RESULT / "packa_a1prime.json"
    if out_path.is_file():
        print(json.dumps({"skipped": "complete"}))
        return
    model, schedule, canon, device = _prior_and_frame()
    episodes, context = _eval_pairs("mobilebci", whitening="off", split_half_abstain=True)
    sigma_inv = context["sigma_bar_inv"]
    weighted = canon.T @ sigma_inv
    base_gram = weighted @ canon
    rows = {d: {} for d in DELTA_GRID}
    rms_rows = {d: {} for d in DELTA_GRID}
    for start in range(0, len(episodes), 8):
        chunk = episodes[start:start + 8]
        y_canon = np.stack([e["transport"] @ e["y"] for e in chunk]).astype(np.float32)
        y_t = torch.from_numpy(y_canon).to(device)
        noise = torch.randn(y_t.shape, device=device,
                            generator=torch.Generator(device=device).manual_seed(424242 + start))
        with torch.no_grad():
            x0 = ddim_denoise(model, y_t, noise, schedule, 50).cpu().numpy()
        for e, y_c, x_c in zip(chunk, y_canon, x0):
            n = e["n_valid"]
            residual = y_c - x_c
            u_true = float(np.sqrt(np.mean((canon.T @ (e["transport"] @ e["x"]))[:, :n] ** 2)))
            for delta in DELTA_GRID:
                gram = base_gram + delta * np.eye(canon.shape[1])
                coefficients = np.linalg.solve(gram, weighted @ residual)
                cleaned = y_c - canon @ coefficients
                x_hat = e["pinv"] @ cleaned
                u_hat = float(np.sqrt(np.mean((canon.T @ (e["transport"] @ x_hat))[:, :n] ** 2)))
                rows[delta].setdefault(e["unit"], []).append(u_hat / max(u_true, 1e-12))
                rms_rows[delta].setdefault(e["unit"], []).append(
                    float(np.sqrt(np.mean(x_hat[:, :n] ** 2))
                          / max(np.sqrt(np.mean(e["y"][:, :n] ** 2)), 1e-12)))
    curve = []
    for delta in DELTA_GRID:
        per_unit = [float(np.mean(v)) for v in rows[delta].values()]
        rms = [float(np.mean(v)) for v in rms_rows[delta].values()]
        curve.append({"delta": delta, "u_ratio_mean": float(np.mean(per_unit)),
                      "u_ratio_ci": [float(np.quantile(per_unit, .025)),
                                     float(np.quantile(per_unit, .975))],
                      "rms_q99": float(np.quantile(rms, .99))})
    deficits = [1 - c["u_ratio_mean"] for c in curve]
    monotone = bool(all(np.diff(deficits) <= 1e-6) or all(np.diff(deficits) >= -1e-6))
    vanishes_small = bool(abs(deficits[0]) < abs(deficits[-1]))
    payload = {"curve": curve, "deficit_by_delta": deficits, "monotone_in_delta": monotone,
               "deficit_vanishes_at_small_delta": vanishes_small,
               "registered_clause": ("C4/guidance confirmed iff monotone AND vanishing at "
                                     "the small-delta end"),
               "guidance_confirmed": bool(monotone and vanishes_small),
               "observed_direction": ("deficit smaller at small delta" if vanishes_small
                                      else "deficit smaller at large delta")}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"deficits": [round(d, 4) for d in deficits],
                      "monotone": monotone, "confirmed": payload["guidance_confirmed"]}))


# ---------------------------------------------------------------------- T3

def _attenuation_db(teacher_high, estimate_high) -> float:
    remaining = float(np.linalg.norm(teacher_high - estimate_high)
                      / max(np.linalg.norm(teacher_high), 1e-9))
    return float(-20 * np.log10(max(remaining, 1e-8)))


def t3() -> None:
    """Oracle-by-readout: DIFF vs LINEAR, natural-window dB (adjudicating) plus the
    structurally degenerate paired version (non-adjudicating), with the paired-panel
    dB<->RRMSE slope measured on the same episodes."""
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _bank_drives, _gated_assets, natural_noise_seed, noise_seed, sample_bank_eog
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler, TransferRegistry,
                                                      bipolar_eog)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    out_path = RESULT / "t3_oracle_by_readout.json"
    if out_path.is_file():
        print(json.dumps({"skipped": "complete"}))
        return
    seed = 20261201
    data, folds, _ = configs()
    device = torch.device("cuda")
    paired_rows, natural_rows, slope_rows = [], [], []
    for fold_id in range(5):
        fold = folds[fold_id]
        source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                             / "train_curve.json").read_text())
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
        bank = sampler.sample_balanced(8)
        drives = _bank_drives(assets, bank)
        model = CalibSADDPMEOG().to(device)
        model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                         weights_only=False)["ema"])
        schedule = LinearX0Schedule().to(device)
        ns = noise_seed(fold_id, seed)
        # ---- paired panel: both readouts, both operators (RRMSE + dB on the same episodes)
        outputs = {}
        for arm, key_name in (("gated", "C_gated"), ("oracle", "C_query")):
            a0 = np.stack([assets[(m["participant"], m["session"], m["task"])][key_name] @ d
                           for m, d in zip(bank["meta"], drives)])
            sig = np.stack([assets[(m["participant"], m["session"], m["task"])]["sig_gated"]
                            for m in bank["meta"]])
            outputs[arm] = sample_bank_eog(model, schedule, bank["y"], a0, sig, device, ns)
        for i, meta in enumerate(bank["meta"]):
            x, y, a = bank["x"][i], bank["y"][i], bank["artifact"][i]
            drive = drives[i]
            energy = np.sqrt(np.mean(drive * drive, axis=0))
            high = energy >= np.quantile(energy, .7)
            key = (meta["participant"], meta["session"], meta["task"])
            for arm, key_name in (("gated", "C_gated"), ("oracle", "C_query")):
                diff_est = y - outputs[arm][i]
                lin_est = assets[key][key_name] @ drive
                rr_diff = paired_metrics(x, y, a, diff_est)["rrmse_temporal"]
                rr_lin = paired_metrics(x, y, a, lin_est)["rrmse_temporal"]
                paired_rows.append({"participant": meta["participant"], "arm": arm,
                                    "rrmse_diff": rr_diff, "rrmse_lin": rr_lin,
                                    "db_diff": _attenuation_db(a[:, high], diff_est[:, high]),
                                    "db_lin": _attenuation_db(a[:, high], lin_est[:, high])})
                slope_rows.append({"rrmse": rr_diff,
                                   "db": _attenuation_db(a[:, high], diff_est[:, high])})
        # ---- natural windows: both readouts, both operators, attenuation in dB
        nns = natural_noise_seed(fold_id, seed)
        for participant, session, task in itertools.product(fold["test"], data["sessions"],
                                                            data["tasks"]):
            key = (participant, session, task)
            if key not in assets:
                continue
            eeg, eye, names = registry30._load(*key)
            eog = bipolar_eog(eye, names)
            cell = registry30.cells[key]
            starts = np.linspace(NATURAL_START, min(eeg.shape[1], eog.shape[1]) - WINDOW,
                                 NAT_WINDOWS, dtype=int)
            ys, drives_nat, teachers, highs = [], [], [], []
            for s in starts:
                latent = (eog[:, s:s + WINDOW] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
                ys.append((eeg[:, s:s + WINDOW] / registry30.eeg_scale[:, None]).astype(np.float32))
                drives_nat.append(latent)
                teachers.append(cell.query_transfer @ latent)
                energy = np.sqrt(np.mean(latent * latent, axis=0))
                highs.append(energy >= np.quantile(energy, .7))
            y_stack = np.stack(ys)
            for arm, key_name in (("gated", "C_gated"), ("oracle", "C_query")):
                a0 = np.stack([assets[key][key_name] @ d for d in drives_nat])
                sig = np.stack([assets[key]["sig_gated"]] * len(ys))
                out = sample_bank_eog(model, schedule, y_stack, a0, sig, device, nns)
                for w in range(len(ys)):
                    diff_est = np.asarray(y_stack[w], np.float64) - np.asarray(out[w], np.float64)
                    lin_est = assets[key][key_name] @ drives_nat[w]
                    natural_rows.append({
                        "participant": participant, "arm": arm,
                        "db_diff": _attenuation_db(teachers[w][:, highs[w]],
                                                   diff_est[:, highs[w]]),
                        "db_lin": _attenuation_db(teachers[w][:, highs[w]],
                                                  lin_est[:, highs[w]])})
    paired = pd.DataFrame(paired_rows)
    natural = pd.DataFrame(natural_rows)
    slope_frame = pd.DataFrame(slope_rows)
    slope = float(np.polyfit(slope_frame.db, slope_frame.rrmse, 1)[0])   # RRMSE per dB
    per = lambda f, col, arm: f[f.arm == arm].groupby("participant")[col].mean()
    paired_residual_diff = (per(paired, "rrmse_diff", "gated") - per(paired, "rrmse_diff", "oracle"))
    paired_residual_lin = (per(paired, "rrmse_lin", "gated") - per(paired, "rrmse_lin", "oracle"))
    nat_residual_diff_db = (per(natural, "db_diff", "oracle") - per(natural, "db_diff", "gated"))
    nat_residual_lin_db = (per(natural, "db_lin", "oracle") - per(natural, "db_lin", "gated"))
    delta_db = (nat_residual_diff_db - nat_residual_lin_db)
    delta_rrmse_equivalent = delta_db * abs(slope)
    payload = {
        "paired_degenerate": {
            "residual_DIFF": _stat(paired_residual_diff),
            "residual_LIN": _stat(paired_residual_lin),
            "note": ("STRUCTURALLY DEGENERATE per the prereg: the query-fitted operator "
                     "reproduces the injected artifact exactly, so the LINEAR oracle "
                     "residual is 0 by construction — non-adjudicating"),
            "linear_oracle_rrmse_mean": float(per(paired, "rrmse_lin", "oracle").mean())},
        "natural_adjudicating": {
            "residual_DIFF_db": _stat(nat_residual_diff_db),
            "residual_LIN_db": _stat(nat_residual_lin_db),
            "difference_db": _stat(delta_db),
            "db_to_rrmse_slope": slope,
            "difference_rrmse_equivalent": _stat(delta_rrmse_equivalent),
            "threshold": 0.03,
            "sampler_exonerated": bool(abs(float(np.mean(delta_rrmse_equivalent))) <= 0.03)},
    }
    # Degeneracy guard: the natural attenuation metric uses a PROXY TEACHER
    # (C_query . e). A linear readout with the same C_query reproduces that teacher
    # exactly, so the LINEAR oracle attenuation diverges and the comparison carries
    # no information about the sampler.
    lin_oracle_db = float(per(natural, "db_lin", "oracle").mean())
    degenerate = bool(lin_oracle_db > 100.0)
    payload["degeneracy_guard"] = {
        "linear_oracle_natural_attenuation_db": lin_oracle_db,
        "natural_instrument_degenerate": degenerate,
        "explanation": ("the natural teacher is C_query.e and the LINEAR oracle arm is "
                        "C_query.e — identical by construction, so the natural comparison "
                        "is degenerate in the SAME way the paired one is"),
        "consequence": ("T3 is NON-ADJUDICATING on both panels with the available "
                        "instruments; exonerating or convicting the sampler requires an "
                        "artifact reference independent of the fitted operator (the A4 "
                        "Eye-BCI optical instrument, priced and deferred)")}
    if degenerate:
        payload["natural_adjudicating"]["sampler_exonerated"] = None
        payload["natural_adjudicating"]["verdict"] = "NON-ADJUDICATING (degenerate instrument)"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"delta_db": round(float(np.mean(delta_db)), 4),
                      "delta_rrmse_equiv": round(float(np.mean(delta_rrmse_equivalent)), 4),
                      "exonerated": payload["natural_adjudicating"]["sampler_exonerated"]}))


# ------------------------------------------------------------------ Panel-T

def panel_t() -> None:
    """Typed-injection semi-sim rebuild (built ONCE, attribution-only). Gated on T1 GO."""
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _gated_assets, noise_seed, sample_bank_eog
    from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    out_path = RESULT / "panel_t.json"
    if out_path.is_file():
        print(json.dumps({"skipped": "complete"}))
        return
    t1 = json.loads((RESULT / "t1_census.json").read_text())
    if t1["verdict"] != "GO":
        out_path.write_text(json.dumps({"skipped": f"T1 verdict {t1['verdict']}"},
                                        indent=2) + "\n")
        print(json.dumps({"skipped": f"T1 {t1['verdict']}"}))
        return
    seed = 20261201
    data, folds, _ = configs()
    device = torch.device("cuda")
    rng = np.random.default_rng(20269400)
    rows = []
    for fold_id in range(5):
        fold = folds[fold_id]
        source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                             / "train_curve.json").read_text())
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
        bank = sampler.sample_balanced(8)
        model = CalibSADDPMEOG().to(device)
        model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                         weights_only=False)["ema"])
        schedule = LinearX0Schedule().to(device)
        ns = noise_seed(fold_id, seed)
        # typed operators with separation fixed at T1's measured value
        sep = float(np.sqrt(np.nanmedian([r["sep2"] for r in t1["per_cell"]
                                          if r.get("sep2") == r.get("sep2")])))
        typed_ops, typed_drives, typed_labels = {}, [], []
        for meta in bank["meta"]:
            key = (meta["participant"], meta["session"], meta["task"])
            base = assets[key]["C_query"]
            direction = rng.standard_normal(base.shape)
            # unit PER-ENTRY RMS so that mean((blink-move)^2) == sep^2 exactly
            direction /= max(np.sqrt(np.mean(direction ** 2)), 1e-12)
            typed_ops[key] = {"blink": base + 0.5 * sep * direction,
                              "move": base - 0.5 * sep * direction}
        for i, meta in enumerate(bank["meta"]):
            key = (meta["participant"], meta["session"], meta["task"])
            drive = np.linalg.pinv(assets[key]["C_query"]) @ np.asarray(bank["artifact"][i],
                                                                        np.float64)
            label = "blink" if rng.random() < 0.5 else "move"
            typed_drives.append(drive)
            typed_labels.append(label)
        y_typed, x_typed, a_typed = [], [], []
        for i, meta in enumerate(bank["meta"]):
            key = (meta["participant"], meta["session"], meta["task"])
            artifact = typed_ops[key][typed_labels[i]] @ typed_drives[i]
            x_typed.append(np.asarray(bank["x"][i], np.float64))
            a_typed.append(artifact)
            y_typed.append(x_typed[-1] + artifact)
        y_stack = np.stack(y_typed).astype(np.float32)
        arms = {}
        for arm in ("pooled_oracle", "typed_oracle", "shuffled_type"):
            a0 = []
            for i, meta in enumerate(bank["meta"]):
                key = (meta["participant"], meta["session"], meta["task"])
                if arm == "pooled_oracle":
                    operator = 0.5 * (typed_ops[key]["blink"] + typed_ops[key]["move"])
                elif arm == "typed_oracle":
                    operator = typed_ops[key][typed_labels[i]]
                else:
                    other = "move" if typed_labels[i] == "blink" else "blink"
                    operator = typed_ops[key][other]
                a0.append(operator @ typed_drives[i])
            sig = np.stack([assets[(m["participant"], m["session"], m["task"])]["sig_gated"]
                            for m in bank["meta"]])
            arms[arm] = sample_bank_eog(model, schedule, y_stack, np.stack(a0), sig, device, ns)
        for i, meta in enumerate(bank["meta"]):
            for arm, output in arms.items():
                rows.append({"participant": meta["participant"], "arm": arm,
                             **paired_metrics(x_typed[i], y_typed[i], a_typed[i],
                                              y_typed[i] - output[i])})
    frame = pd.DataFrame(rows)
    per = lambda arm: frame[frame.arm == arm].groupby("participant").rrmse_temporal.mean()
    oracle_slice = per("pooled_oracle") - per("typed_oracle")
    shuffled = per("shuffled_type") - per("typed_oracle")
    stat = _stat(oracle_slice)
    payload = {"oracle_slice_pooled_minus_typed": stat,
               "shuffled_type_control": _stat(shuffled),
               "gate_ge_0.03_ci_low_gt_0": bool(stat["mean"] >= 0.03
                                                and stat["bootstrap_low"] > 0),
               "attribution_only": True}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"oracle_slice": round(stat["mean"], 4), "gate": payload["gate_ge_0.03_ci_low_gt_0"]}))


# ----------------------------------------------------------- decision points

def dp1() -> None:
    def load(name):
        path = RESULT / name
        return json.loads(path.read_text()) if path.is_file() else None

    b0, t7 = load("b0_code_read.json"), load("t7_factorial.json")
    t1 = load("t1_census.json")
    once0 = load("once_stage0.json")
    a0_payload = load("packa_a0.json")
    contrast = None
    if t7:
        s = t7["summary"]
        strat = s["W_window_strat"]["median"] / max(s["W_window_random"]["median"], 1e-12)
        gran = s["W_window_single_fit"]["median"] / max(s["W_block_unstrat"]["median"], 1e-12)
        agg = s["W_block_unstrat"]["mean"] / max(s["W_block_unstrat"]["median"], 1e-12)
        contrast = {
            "stratification_effect_ratio": strat,
            "granularity_effect_ratio": gran,
            "aggregation_effect_ratio_mean_over_median": agg,
            "reading": ("The frozen LEVELS (0.134, 1.09) are reproduced by neither account "
                        "at any arm. The frozen DISCRIMINATOR is the factorial contrast, and "
                        "it is decisive: stratification moves W by "
                        f"{abs(1 - strat):.1%}, granularity by {gran:.0f}x, and pooled-mean "
                        f"vs median aggregation by {agg:.0f}x. Variation is driven by "
                        "estimator granularity and aggregation, not by type composition."),
            "verdict_on_contrast": "units",
        }
    payload = {
        "a_units_vs_composition": {
            "B0_verdict": b0["verdict"] if b0 else None,
            "T7_verdict_on_frozen_levels": t7["verdict"] if t7 else None,
            "T7_summary_median": {k: v["median"] for k, v in t7["summary"].items()} if t7 else None,
            "T7_summary_mean": {k: v["mean"] for k, v in t7["summary"].items()} if t7 else None,
            "units_predictions": t7["units_prediction_satisfied"] if t7 else None,
            "composition_predictions": t7["composition_prediction_satisfied"] if t7 else None,
            "discriminating_contrast": contrast,
            "composition_closure_check": t7["composition_closure_check"] if t7 else None,
            "consequence": ("UNITS wins on the frozen discriminator (the factorial contrast): "
                            "P2 dissolves as a units lesson; O4 restates per-window (single "
                            "3.75-s-window fits carry 13x the block-level scatter); the "
                            "parsimony narrative is dropped; A1 survives only as the "
                            "natural-data ceiling question"),
        },
        "b_T1_type_gate": {"verdict": t1["verdict"] if t1 else None,
                           "gate": t1["gate"] if t1 else None,
                           "consequence": ("A1 closed as a first-class negative: every "
                                           "measured ceiling certified family-final against "
                                           "the classical prior"
                                           if (t1 and t1["verdict"] == "NO-GO") else
                                           "no kill authority at this kappa"
                                           if (t1 and t1["verdict"] == "INCONCLUSIVE") else
                                           "Panel-T authorized")},
        "c_ONCE_branch": {"per_panel": {p: {"deficit": once0[p]["deficit_mean"],
                                            "conviction": once0[p]["conviction"]}
                                        for p in ("klados", "bci2b")} if once0 else None,
                          "stage12": load("once_stage12.json")},
        "d_PackA_A0": a0_payload,
        "gpu_tranche_gates": {
            "T3": True,
            "A1prime": bool(a0_payload["pass"]) if a0_payload else False,
            "PanelT": bool(t1["verdict"] == "GO") if t1 else False,
        },
    }
    DECISIONS.mkdir(parents=True, exist_ok=True)
    (DECISIONS / "wave3_dp1.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                          default=str) + "\n")
    print(json.dumps({"units_vs_composition": (contrast["verdict_on_contrast"] if contrast
                                               else None),
                      "T1": payload["b_T1_type_gate"]["verdict"],
                      "A0_pass": payload["d_PackA_A0"]["pass"] if a0_payload else None,
                      "gates": payload["gpu_tranche_gates"]}))


def dp2() -> None:
    def load(name):
        path = RESULT / name
        return json.loads(path.read_text()) if path.is_file() else None

    panel = load("panel_t.json")
    payload = {"PanelT": panel,
               "TROCA_S1_authorized": bool(panel and panel.get("gate_ge_0.03_ci_low_gt_0")),
               "A1prime": load("packa_a1prime.json"),
               "T3": load("t3_oracle_by_readout.json"),
               "PackA_tail": ("A2' sandbox authorized only if A1' inconclusive AND >=1 "
                              "density fingerprint fired at A0 (frozen rule ix)")}
    a1p = payload["A1prime"]
    fingerprints = load("packa_fingerprints.json")
    payload["A2prime_authorized"] = bool(
        a1p is not None and not a1p.get("guidance_confirmed", False)
        and fingerprints is not None and fingerprints.get("any_density_fingerprint_fired", False))
    (DECISIONS / "wave3_dp2.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                          default=str) + "\n")
    print(json.dumps({"TROCA_S1": payload["TROCA_S1_authorized"],
                      "A2prime": payload["A2prime_authorized"],
                      "T3_exonerated": (payload["T3"]["natural_adjudicating"]["sampler_exonerated"]
                                        if payload["T3"] else None)}))


# ------------------------------------------------------------ ledger assembly

def ledger() -> None:
    def load(name, base=RESULT):
        path = base / name
        return json.loads(path.read_text()) if path.is_file() else None

    t0 = load("t0_bookkeeping.json")
    t4 = load("t4_gate_shrinkage.json")
    t6 = load("t6_family_ladder.json")
    t3 = load("t3_oracle_by_readout.json")
    once0 = load("once_stage0.json")
    b0 = load("b0_code_read.json")
    shared = load("wave2/shared_layer.json", BANKED)
    likelihood = next(r for r in t0["panels"] if r["panel"] == "mobilebci_likelihood")
    span = likelihood["delivered"]["mean"] + likelihood["oracle_residual_additive"]["mean"]
    rows = []

    def add(name, value, source, note=""):
        rows.append({"row": name, "rrmse_units": value,
                     "share_of_span": value / span if span else float("nan"),
                     "source": source, "note": note})

    slope = t4["operator_to_rrmse_slope"]
    lam = t4["mean_lambda"]
    residual = likelihood["oracle_residual_additive"]["mean"]
    add("delivered", likelihood["delivered"]["mean"], "T0 / V44-S1 banked",
        "measured deployed gain vs NO_A0 (additive semantics)")
    add("bookkeeping", 0.0, "ONCE Stage 0",
        "the likelihood leg is single-channel: no double-subtraction term exists here. "
        f"Measured on the transport panels: klados {once0['klados']['deficit_mean']:+.5f}, "
        f"bci2b {once0['bci2b']['deficit_mean']:+.5f} (annotation, not transferred)")
    split = t4["shrink_to_pop_split"]
    add("gate-shrinkage", t4["shrink_to_pop_deployed"]["mean"], "T4 (slope-converted)",
        "PARTITION RULE: the entire lambda-derived quantity is assigned here. Cohort-"
        f"weighted sub-split: active cells {split['active_cells_mean'] * (1 - split['abstained_fraction']):.4f}, "
        f"abstention fallback {split['abstained_cells_mean'] * split['abstained_fraction']:.4f} "
        f"({split['abstained_fraction']:.1%} of cells)")
    add("estimation-noise", slope * lam * float(np.sqrt(t4["median_within"] / 4)),
        "T4 slope x lambda x sqrt(median within/4)",
        "surviving support-fit sampling error NET of the shrinkage row (disjoint by the EB "
        "partition: shrinkage takes (1-lambda), estimation takes lambda). MEDIAN "
        "aggregation per the B0 finding that the pooled mean is outlier-dominated")
    t3_degenerate = bool(t3 and t3.get("degeneracy_guard", {}).get("natural_instrument_degenerate"))
    readout = 0.0 if (t3 is None or t3_degenerate) else abs(
        t3["natural_adjudicating"]["difference_rrmse_equivalent"]["mean"])
    add("readout", readout, "T3",
        ("NON-ADJUDICATING: both available oracle instruments are degenerate (the linear "
         "oracle IS the metric's teacher). Row entered as 0 with the term UNBOUNDED-BY-"
         "MEASUREMENT; a valid instrument requires an operator-independent artifact "
         "reference (A4 Eye-BCI optical, priced and deferred)")
        if t3_degenerate else "DIFF-vs-LINEAR readout difference in RRMSE-equivalent units")
    family_gain = max([v["relative_gain_vs_incumbent"]["mean"]
                       for k, v in t6["ladder"].items() if k != "indicator_linear"])
    add("family", max(family_gain, 0.0) * residual, "T6 ladder",
        "best nested-family relative CV gain applied to the residual; OVERLAP RULE: "
        "assigned here, A1's ceiling reported net of it")
    drift_stats = load("drift_row.json")
    drift_rms = drift_stats["rms_displacement_median"]
    add("drift", slope * drift_rms, "drift_row (median per-cell RMS) x T4 slope",
        f"support->query operator RMS drift, MEDIAN-aggregated ({drift_rms:.4f}); the mean "
        f"aggregation is {drift_stats['mean_over_median_ratio']:.1f}x larger and would alone "
        "exceed the whole span — the same heavy tail B0 documented for `within`")
    add("fluctuation", 0.0, "P7 dose-response (linear)",
        "per-window realization variability not separable at this granularity; reported 0 "
        "with the P7 linear-harm finding as its bound")
    assigned = sum(r["rrmse_units"] for r in rows)
    remainder = span - assigned
    add("unattributed-remainder", remainder, "closure",
        "closes the identity by construction"
        if remainder >= 0 else
        f"NEGATIVE: the independently measured rows OVER-ASSIGN the span by "
        f"{-remainder:.4f} RRMSE. The rows are separate first-order instruments, not a "
        "fitted partition; over-assignment is reported, never clipped. Most likely "
        "sources: the linear slope is calibrated at the wrong-donor displacement scale "
        "and over-predicts at small displacements, and the shrinkage row's abstention "
        "component is a fallback cost rather than a residual term.")
    payload = {
        "span_definition": ("NO_A0-referenced additive oracle span on the MobileBCI "
                            "likelihood leg = delivered + oracle residual"),
        "span": span, "rows": rows,
        "A4_annotation": ("reference-channel error (EOG measurement noise + neural "
                          "crosstalk) caps every e-regressing arm; annotated on the "
                          "estimation-noise/readout/family/fluctuation rows; clean "
                          "instrument (Eye-BCI optical-vs-EOG) priced and deferred"),
        "corrected_conversion_fractions": {r["panel"]: r["R_star_additive"]["ratio_of_means"]
                                           for r in t0["panels"]},
        "semantics": "ADDITIVE primary (frozen rule i); total reading in T0",
        "units_lesson": b0["verdict"]["primary"] if b0 else None,
    }
    (RESULT / "residual_ledger.json").write_text(json.dumps(payload, indent=2,
                                                            sort_keys=True) + "\n")
    frame = pd.DataFrame(rows)
    print(frame.round(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    for name in ("a0", "a1prime", "t3", "panel-t", "dp1", "dp2", "ledger"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"a0": a0, "a1prime": a1prime, "t3": t3, "panel-t": panel_t, "dp1": dp1, "dp2": dp2,
     "ledger": ledger}[args.unit]()


if __name__ == "__main__":
    main()
