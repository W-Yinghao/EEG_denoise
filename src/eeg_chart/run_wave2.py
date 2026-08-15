"""WAVE2-T1 execution CLI. All rules frozen in reports/wave2_preregistration.md.

No sealed byte is read anywhere in this module (dev cohort, Klados, BCI2b,
MobileBCI v4 motion blocks only).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/wave2"
DECISIONS = ROOT / "decisions"
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
MOTION_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/mobile_bci_headroom_v4")
BANKED_LAMBDA = (0.8754, 0.8752, 0.9179, 0.9024, 0.8883)   # V43 fold manifests
V44_DELIVERED = 0.14280771381963858                        # RB-1 vs NO_A0
V44_ORACLE_GAP = 0.24413                                   # ORACLE residual (additive)
V44_NOA0_MINUS_ORACLE = 0.5738 - 0.1868                    # total reading denominator


def _stat(values) -> dict[str, Any]:
    from eeg_scad.cli.run_v43 import bootstrap_draws

    series = np.asarray(list(values), float)
    draws = bootstrap_draws(series)
    return {"mean": float(series.mean()), "median": float(np.median(series)),
            "positive_count": int((series > 0).sum()), "n": int(len(series)),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


# ------------------------------------------------------------- shared layer

def shared_layer() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, folds, _ = configs()
    per_fold = []
    drift_vectors = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        dev = set(fold["train"] + fold["validation"] + fold["test"])
        supports, queries, withins = [], [], []
        for key in sorted(eb120.cells):
            if key[0] not in dev:
                continue
            cell = eb120.cells[key]
            c_query = registry30.cells[key].query_transfer
            supports.append(cell.transfer)
            queries.append(c_query)
            withins.append(cell.within)
            drift_vectors.append((c_query - cell.transfer).reshape(-1))
        supports = np.stack(supports)
        queries = np.stack(queries)
        mean_op = supports.mean(axis=0)
        tau2 = float(np.mean(np.square(supports - mean_op[None])))
        w = float(np.mean(withins))
        d_raw = float(np.mean(np.square(queries - supports)))
        # query-side within (4 Qgen sub-block scatter) approximated by W (same
        # estimator class); the debiased reading subtracts both estimation floors
        d_deb = float(max(0.0, d_raw - w / 4 - w / 4))
        per_fold.append({"fold": fold["fold"], "tau2": tau2, "W": w,
                         "D_raw_total": d_raw, "D_debiased_additive": d_deb,
                         "D_over_tau2_raw": d_raw / tau2, "D_over_tau2_deb": d_deb / tau2,
                         "lambda_pred": float(tau2 / (tau2 + w / 4)),
                         "cells": int(len(supports))})
    sigma_drift = np.var(np.stack(drift_vectors), axis=0, ddof=1).reshape(46, 2)

    # cross-panel rows (support halves as W proxy; a_query as the drift target)
    panel_rows = []
    from eeg_chart.panels import build_bci2b_panel, build_klados_panel
    for name, builder in (("klados", build_klados_panel), ("bci2b", build_bci2b_panel)):
        cells, _ = builder()
        supports = np.stack([c.a_support for c in cells])
        queries = np.stack([c.a_query for c in cells])
        w = float(np.mean([np.mean(np.square(c.a_halves[0] - c.a_halves[1])) / 2
                           for c in cells]))
        tau2 = float(np.mean(np.square(supports - supports.mean(axis=0)[None])))
        d_raw = float(np.mean(np.square(queries - supports)))
        panel_rows.append({"panel": name, "tau2": tau2, "W": w, "D_raw_total": d_raw,
                           "D_debiased_additive": float(max(0.0, d_raw - w)),
                           "D_over_tau2_raw": d_raw / tau2, "cells": len(cells)})
    RESULT.mkdir(parents=True, exist_ok=True)
    payload = {"mobilebci_per_fold": per_fold, "cross_panels": panel_rows,
               "banked_lambda_mean": float(np.mean(BANKED_LAMBDA)),
               "semantics": "D reported in BOTH readings per the language rule"}
    (RESULT / "shared_layer.json").write_text(json.dumps(payload, indent=2,
                                                         sort_keys=True) + "\n")
    np.savez(RESULT / "sigma_drift.npz", sigma_drift=sigma_drift)
    print(json.dumps({"tau2_mean": float(np.mean([r["tau2"] for r in per_fold])),
                      "W_mean": float(np.mean([r["W"] for r in per_fold])),
                      "D_raw_over_tau2": float(np.mean([r["D_over_tau2_raw"]
                                                        for r in per_fold])),
                      "lambda_pred_mean": float(np.mean([r["lambda_pred"]
                                                         for r in per_fold]))}))


def p2() -> None:
    layer = json.loads((RESULT / "shared_layer.json").read_text())
    lambda_pred = float(np.mean([r["lambda_pred"] for r in layer["mobilebci_per_fold"]]))
    lambda_measured = layer["banked_lambda_mean"]
    additive = V44_DELIVERED / (V44_DELIVERED + V44_ORACLE_GAP)
    total = V44_DELIVERED / V44_NOA0_MINUS_ORACLE
    lambda_ok = bool(abs(lambda_pred - lambda_measured) <= 0.03)
    conversion_ok = bool(0.3 <= additive <= 0.5 or 0.3 <= total <= 0.5)
    verdict = "PASS" if (lambda_ok and conversion_ok) else "SOFT-FAIL"
    d_ratios = [r["D_over_tau2_raw"] for r in layer["mobilebci_per_fold"]]
    payload = {"lambda_pred": lambda_pred, "lambda_measured_banked": lambda_measured,
               "lambda_within_0.03": lambda_ok,
               "conversion_additive": additive, "conversion_total": total,
               "conversion_in_band": conversion_ok,
               "D_over_tau2_raw_mean": float(np.mean(d_ratios)),
               "D_band_language": "D in [tau2, 2.4tau2]; D~tau2 is the optimistic endpoint",
               "verdict": verdict,
               "consequence": (None if verdict == "PASS" else
                               "Sigma_drift downgraded to an empirical object; "
                               "rho-hat prediction language stripped from sealed plans")}
    (RESULT / "p2_ledger.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "lambda_pred": round(lambda_pred, 4),
                      "additive": round(additive, 4), "total": round(total, 4)}))


# ------------------------------------------------------------------- MOKA

def moka() -> None:
    from scipy.signal import butter, sosfiltfilt
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale

    step_zero = {"motion_blocks_exist": MOTION_ROOT.is_dir(),
                 "example_readable": False,
                 "closure_wording": ("v5 closure LABEL = PROTOCOL_CORE_VALID / "
                                     "SSVEP_SAFETY_PREVIOUSLY_INVALID / ONE-SEED_RAW_"
                                     "TEMPORAL_ROUTE_NO_GO / DIFFUSION_FAMILY_NOT_TESTED "
                                     "(commit 49a43a7) — covers temporal-support/diffusion "
                                     "routes only; motion-operator conditioning not barred")}
    sos = butter(4, (0.5, 15.0), btype="bandpass", fs=100.0, output="sos")
    fractions, mb_rows = [], []
    subjects = sorted(p.name for p in MOTION_ROOT.iterdir() if p.name.startswith("sub-"))
    for subject in subjects:
        for session_dir in sorted((MOTION_ROOT / subject).iterdir()):
            for task_dir in sorted(session_dir.iterdir()):
                eeg_path = task_dir / "eeg.npy"
                imu_path = task_dir / "imu.npy"
                proxy_path = task_dir / "clean_proxy.npy"
                if not (eeg_path.is_file() and imu_path.is_file()):
                    continue
                eeg = np.asarray(np.load(eeg_path), np.float64)
                imu = np.asarray(np.load(imu_path), np.float64)
                step_zero["example_readable"] = True
                eeg_f = sosfiltfilt(sos, eeg, axis=-1)
                _, iscale = robust_center_scale(imu)
                imu_s = imu / iscale[:, None]
                half = eeg.shape[1] // 2
                gram = imu_s[:, :half] @ imu_s[:, :half].T
                ridge = 0.05 * float(np.trace(gram)) / len(gram)
                operator = (eeg_f[:, :half] @ imu_s[:, :half].T) \
                    @ np.linalg.inv(gram + ridge * np.eye(len(gram)))
                predicted = operator @ imu_s[:, half:]
                target = eeg_f[:, half:]
                fraction = float(1 - np.sum((target - predicted) ** 2)
                                 / max(np.sum(target ** 2), 1e-12))
                fractions.append({"subject": subject, "session": session_dir.name,
                                  "task": task_dir.name, "fraction": max(fraction, 0.0)})
                if proxy_path.is_file():
                    proxy = np.asarray(np.load(proxy_path), np.float64)
                    mb_rows.append({"subject": subject, "session": session_dir.name,
                                    "task": task_dir.name, "eeg_path": str(eeg_path)})
    frame = pd.DataFrame(fractions)
    per_subject = frame.groupby("subject").fraction.median()
    ma_median = float(per_subject.median())
    ma_go = bool(ma_median >= 0.10)
    result: dict[str, Any] = {"step_zero": step_zero,
                              "M-A": {"median_fraction": ma_median,
                                      "per_subject_median": per_subject.round(4).to_dict(),
                                      "go": ma_go, "records": len(frame)}}
    if ma_go:
        # M-B: own vs population motion operator, V19-style paired on real
        # (clean_proxy, eeg) pairs; subtraction arm y - C·imu_features
        operators, keys = {}, []
        for row in mb_rows:
            task_dir = Path(row["eeg_path"]).parent
            eeg = sosfiltfilt(sos, np.asarray(np.load(task_dir / "eeg.npy"), np.float64),
                              axis=-1)
            proxy = sosfiltfilt(sos, np.asarray(np.load(task_dir / "clean_proxy.npy"),
                                                np.float64), axis=-1)
            imu = np.asarray(np.load(task_dir / "imu.npy"), np.float64)
            _, iscale = robust_center_scale(imu)
            imu_s = imu / iscale[:, None]
            artifact = eeg - proxy
            half = eeg.shape[1] // 2
            gram = imu_s[:, :half] @ imu_s[:, :half].T
            ridge = 0.05 * float(np.trace(gram)) / len(gram)
            operator = (artifact[:, :half] @ imu_s[:, :half].T) \
                @ np.linalg.inv(gram + ridge * np.eye(len(gram)))
            key = (row["subject"], row["task"])
            operators.setdefault(key, []).append(
                {"operator": operator, "task_dir": task_dir, "half": half})
            keys.append(key)
        gains_by_task: dict[str, list[float]] = {}
        for (subject, task), entries in operators.items():
            pop = np.mean([o["operator"] for (s, t), os_ in operators.items()
                           if s != subject and t == task for o in os_], axis=0)
            own = np.mean([o["operator"] for o in entries], axis=0)
            per_gains = []
            for entry in entries:
                task_dir = entry["task_dir"]
                eeg = sosfiltfilt(sos, np.asarray(np.load(task_dir / "eeg.npy"),
                                                  np.float64), axis=-1)
                proxy = sosfiltfilt(sos, np.asarray(np.load(task_dir / "clean_proxy.npy"),
                                                    np.float64), axis=-1)
                imu = np.asarray(np.load(task_dir / "imu.npy"), np.float64)
                _, iscale = robust_center_scale(imu)
                imu_s = imu / iscale[:, None]
                half = entry["half"]
                y, x, e = eeg[:, half:], proxy[:, half:], imu_s[:, half:]
                def rrmse(operator_):
                    est = y - operator_ @ e
                    return float(np.linalg.norm(est - x) / max(np.linalg.norm(x), 1e-12))
                per_gains.append(rrmse(pop) - rrmse(own))
            gains_by_task.setdefault(task, []).append(float(np.mean(per_gains)))
        mb = {}
        any_go = False
        for task, gains in gains_by_task.items():
            stat = _stat(gains)
            go = bool(stat["mean"] >= 0.02 and stat["bootstrap_low"] > 0)
            any_go |= go
            mb[task] = {**stat, "go": go}
        result["M-B"] = {"per_protocol": mb, "go": bool(any_go)}
    else:
        result["M-B"] = {"skipped": "M-A NO-GO"}
    (RESULT / "moka.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"M-A_median": ma_median, "M-A_go": ma_go,
                      "M-B_go": result["M-B"].get("go", False)}))


# ------------------------------------------------------------------ OPERA A0

def opera_a0() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry, bipolar_eog

    data, folds, _ = configs()
    posterior_names = {"P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
                       "O1", "Oz", "O2"}
    fold = folds[0]
    registry30 = TransferRegistry(data, fold, 30, .05)
    leakages = []
    audit_rows = []
    for key in sorted(registry30.cells):
        eeg, eye, names = registry30._load(*key)
        eog = bipolar_eog(eye, names)
        cell = registry30.cells[key]
        latent = (eog - cell.eog_center[:, None]) / cell.eog_scale[:, None]
        with np.load(Path(registry30.root) / "prepared" / key[0]
                     / f"{key[1]}_{key[2]}.npz", allow_pickle=False) as archive:
            eeg_names = [str(v) for v in archive["eeg_names"]]
        posterior_index = [i for i, n in enumerate(eeg_names) if n in posterior_names]
        energy = np.sqrt(np.mean(latent * latent, axis=0))
        low = energy <= np.quantile(energy, .3)
        post = eeg[np.ix_(posterior_index, np.flatnonzero(low))]
        target = latent[:, low]
        gram = post @ post.T + 1e-6 * np.trace(post @ post.T) / len(post) * np.eye(len(post))
        beta = np.linalg.solve(gram, post @ target.T)
        fitted = beta.T @ post
        r2 = float(np.sum(fitted ** 2) / max(np.sum(target ** 2), 1e-12))
        leakages.append(r2)
        audit_rows.append({"cell": "|".join(key), "leakage_r2": r2,
                           "low_window_fraction": float(low.mean())})
    leak = float(np.median(leakages))
    payload = {"exogeneity_leakage_r2_median": leak, "hard_gate_bound": 0.15,
               "pass": bool(leak <= 0.15), "per_cell": audit_rows,
               "note": "Eye-BCI optical reference is the only clean escape if this fails",
               "corpus_audit": {"panels": ["mobilebci", "klados", "bci2b"],
                                "contamination_model": "C_query @ gain*latent (V19)"}}
    (RESULT / "opera_a0.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"leakage_r2_median": round(leak, 4), "pass": payload["pass"]}))


# --------------------------------------------------------------- DT-Gibbs G0

def gibbs_g0() -> None:
    from scipy import stats
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.cli.run_v44_s2 import _posterior_variance

    data, folds, _ = configs()
    sigma_drift = np.load(RESULT / "sigma_drift.npz")["sigma_drift"]
    coverages = {}
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        post_var = _posterior_variance(registry30, eb120, fold)
        for key in sorted(eb120.cells):
            if key not in post_var:
                continue
            c_query = registry30.cells[key].query_transfer
            center = eb120.cells[key].transfer
            sd = np.sqrt(post_var[key] + sigma_drift)
            inside = np.abs(c_query - center) <= stats.norm.ppf(0.9) * sd
            coverages.setdefault(key[0], []).append(float(inside.mean()))
    per_participant = {p: float(np.mean(v)) for p, v in coverages.items()}
    mean_cov = float(np.mean(list(per_participant.values())))
    payload = {"nominal": 0.80, "band": [0.70, 0.90], "coverage_mean": mean_cov,
               "per_participant": per_participant,
               "pass": bool(0.70 <= mean_cov <= 0.90)}
    (RESULT / "gibbs_g0.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"coverage": round(mean_cov, 4), "pass": payload["pass"]}))


# ---------------------------------------------------------------- THRESH T0

def thresh_t0() -> None:
    layer = json.loads((RESULT / "shared_layer.json").read_text())
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, folds, _ = configs()
    deviations = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        mean_op = np.mean([c.transfer for c in eb120.cells.values()], axis=0)
        for key, cell in eb120.cells.items():
            deviations.append((cell.transfer - mean_op).reshape(-1))
    deviation_cov = np.cov(np.stack(deviations).T)
    spectrum = np.sort(np.linalg.eigvalsh(deviation_cov))[::-1].clip(0)
    d_eff = float(spectrum.sum() ** 2 / max(np.sum(spectrum ** 2), 1e-12))
    ambient = int(deviation_cov.shape[0])
    harm_survives = bool(d_eff <= 20)
    tau2 = float(np.mean([r["tau2"] for r in layer["mobilebci_per_fold"]]))
    w = float(np.mean([r["W"] for r in layer["mobilebci_per_fold"]]))
    n_star = d_eff * w / max(tau2, 1e-12)   # closed-form crossover under anisotropy
    payload = {"d_eff": d_eff, "ambient_dim": ambient,
               "spectrum_top10": [float(s) for s in spectrum[:10]],
               "harm_prediction_survives_on_paper": harm_survives,
               "consequence": (None if harm_survives else
                               "T1a demo shrinks to transition-only (frozen rule)"),
               "closed_form_crossover_context_length_ratio": n_star,
               "task_prior": {"tau2": tau2, "W": w}}
    (RESULT / "thresh_t0.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"d_eff": round(d_eff, 2), "harm_survives": harm_survives}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    for name in ("shared-layer", "p2", "moka", "opera-a0", "gibbs-g0", "thresh-t0"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"shared-layer": shared_layer, "p2": p2, "moka": moka, "opera-a0": opera_a0,
     "gibbs-g0": gibbs_g0, "thresh-t0": thresh_t0}[args.unit]()


if __name__ == "__main__":
    main()
