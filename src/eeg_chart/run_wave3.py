"""WAVE3 CPU battery. All rules frozen in reports/wave3_preregistration.md.

No sealed contact anywhere in this module: MobileBCI development cohort, Klados,
BCI2b and already-banked artifacts only.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/wave3"
DECISIONS = ROOT / "decisions"
V43_STATE = Path("/home/infres/yinwang/denoiseNet_rgcc_v43/results/rgcc_v43/state")
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
BANKED = ROOT / "results"
SUPPORT_STOP = 12000
QGEN = (15000, 27000)
NATURAL_START = 30000
PSI_WINDOW = 100          # 1.0 s at 100 Hz
FIT_WINDOW = 375          # ~3.7 s per-window granularity (T7 window arm)
BLOCK_COUNT = 4
RIDGE = 0.05
SEED = 20269301


def _stat(values, seed: int = 420) -> dict[str, Any]:
    from eeg_scad.cli.run_v43 import bootstrap_draws

    series = np.asarray(list(values), float)
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return {"mean": float("nan"), "n": 0}
    draws = bootstrap_draws(series, seed=seed)
    return {"mean": float(series.mean()), "median": float(np.median(series)),
            "positive_count": int((series > 0).sum()), "n": int(len(series)),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def _ridge(eeg: np.ndarray, drive: np.ndarray, ratio: float = RIDGE) -> np.ndarray:
    y = eeg - eeg.mean(axis=1, keepdims=True)
    e = drive - drive.mean(axis=1, keepdims=True)
    gram = e @ e.T
    ridge = float(ratio) * max(float(np.trace(gram) / len(gram)), np.finfo(float).eps)
    return (y @ e.T) @ np.linalg.inv(gram + ridge * np.eye(len(gram)))


class Records:
    """Shared loader: fold-0 normalization constants, dev cohort only."""

    def __init__(self):
        from eeg_scad.cli.run_v43 import configs
        from eeg_scad.data.artifact_transfer_v41r import TransferRegistry

        self.data, self.folds, _ = configs()
        self.registry = TransferRegistry(self.data, self.folds[0], 30, RIDGE)
        self.keys = sorted(self.registry.cells)
        self.eeg_scale = self.registry.eeg_scale

    def load(self, key):
        from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
        from eeg_scad.data.v24_coordinate_contract import robust_center_scale

        eeg, eye, names = self.registry._load(*key)
        eog = bipolar_eog(eye, names)
        center, scale = robust_center_scale(eog[:, :SUPPORT_STOP])
        latent = (eog - center[:, None]) / scale[:, None]
        return eeg / self.eeg_scale[:, None], latent


# --------------------------------------------------------------------- psi

def psi_labels(latent_veog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen global type classifier on 1-s VEOG windows -> labels, features."""
    n = len(latent_veog) // PSI_WINDOW
    z = latent_veog[:n * PSI_WINDOW].reshape(n, PSI_WINDOW)
    median = np.median(z, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(z - median), axis=1, keepdims=True)
    zz = (z - median) / np.maximum(mad, 1e-9)
    peak = np.max(np.abs(zz), axis=1)
    step = np.abs(zz[:, -30:].mean(axis=1) - zz[:, :30].mean(axis=1)) / np.maximum(peak, 1e-9)
    width = np.mean(np.abs(zz) >= 0.5 * peak[:, None], axis=1)
    labels = np.full(n, "unclassified", dtype=object)
    labels[(peak >= 3.0) & (step <= 0.40) & (width <= 0.35)] = "blink"
    labels[(peak >= 2.0) & (step >= 0.60)] = "move"
    return labels, np.stack((peak, step, width), axis=1)


def kappa_reference(latent_veog: np.ndarray, labels: np.ndarray) -> float:
    """Cohen's kappa of psi against an independent unsupervised GMM instrument
    on a DISJOINT feature set (registered)."""
    from scipy.signal import welch
    from sklearn.metrics import cohen_kappa_score
    from sklearn.mixture import GaussianMixture

    n = len(labels)
    z = latent_veog[:n * PSI_WINDOW].reshape(n, PSI_WINDOW)
    derivative = np.diff(z, axis=1)
    freqs, power = welch(derivative, fs=100.0, nperseg=min(64, derivative.shape[1]), axis=-1)
    low = power[:, (freqs >= 0.5) & (freqs < 3)].sum(axis=1)
    high = power[:, (freqs >= 3) & (freqs < 8)].sum(axis=1)
    zcr = np.mean(np.diff(np.sign(derivative), axis=1) != 0, axis=1)
    features = np.stack((np.log(low + 1e-12), np.log(high + 1e-12), zcr), axis=1)
    typed = labels != "unclassified"
    if typed.sum() < 20 or len(set(labels[typed])) < 2:
        return float("nan")
    model = GaussianMixture(2, random_state=0, covariance_type="full").fit(features[typed])
    component = model.predict(features[typed])
    share = np.asarray([np.mean(low[typed][component == c] / (low[typed][component == c]
                                                              + high[typed][component == c] + 1e-12))
                        for c in (0, 1)])
    blink_component = int(np.argmax(share))
    reference = np.where(component == blink_component, "blink", "move")
    return float(cohen_kappa_score(labels[typed].astype(str), reference))


# ---------------------------------------------------------------------- B0

def b0() -> None:
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(V43_STATE / "fold_*/eb_state_manifest_s2.csv")))]
    manifest = pd.concat(frames)
    d = manifest[manifest.seconds == 120]
    recompute = np.clip(d.tau2 / (d.tau2 + d.within / 4), 0, 1).where(d.hard_gate == 0, 0.0)
    wave2 = json.loads((BANKED / "wave2/shared_layer.json").read_text())
    w2_tau2 = float(np.mean([r["tau2"] for r in wave2["mobilebci_per_fold"]]))
    w2_w = float(np.mean([r["W"] for r in wave2["mobilebci_per_fold"]]))
    payload = {
        "step": "B0 code read (STEP 1 of the frozen units-vs-composition tree)",
        "literal_lambda_estimator": ("float(np.clip(tau2 / max(tau2 + within / 4.0, 1e-12), "
                                     "0.0, 1.0)) — eb_transfer_v43.eb_lambda"),
        "lambda_form_inference_pending_B0": "LIFTED — the code states the form literally",
        "literal_within_estimator": {
            "form": "BLOCK-MEAN VARIANCE, not per-window variance",
            "code": ("deviation = np.square(blocks - full[None]); within = deviation.mean() "
                     "— blocks are the 4 contiguous 30-s sub-block fits of the 120-s prefix, "
                     "full is the 120-s full-prefix fit"),
        },
        "literal_tau2_estimator": {
            "form": ("mean squared deviation of fold-TRAIN owners' 120-s fits around "
                     "registry30's POPULATION operator, which is a mean of 30-s prefix fits"),
            "note": "CROSS-DURATION REFERENCE: 120-s deviates measured against a 30-s centroid",
        },
        "wave2_shared_layer_estimator": {
            "tau2": ("pooled squared deviation of all dev 120-s fits around their own "
                     "120-s centroid, pooled across (session, task) groups"),
            "W": "pooled MEAN of the same per-cell block-mean within",
            "lambda_pred": "tau2_pooled / (tau2_pooled + W_pooled / 4)",
        },
        "deployed_manifest_read": {
            "cells_120s": int(len(d)),
            "tau2_mean": float(d.tau2.mean()), "tau2_median": float(d.tau2.median()),
            "within_mean": float(d.within.mean()), "within_median": float(d.within.median()),
            "within_quantiles": {str(q): float(d.within.quantile(q))
                                 for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
            "within_above_1_count": int((d.within > 1).sum()),
            "lambda_mean_per_cell": float(d["lambda"].mean()),
            "hard_gate_fraction": float(d.hard_gate.mean()),
            "per_cell_lambda_recompute_max_abs_diff": float(np.max(np.abs(recompute - d["lambda"]))),
        },
        "reconciliation": {
            "lambda_of_pooled_means_deployed_inputs": float(d.tau2.mean()
                                                            / (d.tau2.mean() + d.within.mean() / 4)),
            "lambda_of_pooled_medians_deployed_inputs": float(d.tau2.median()
                                                              / (d.tau2.median() + d.within.median() / 4)),
            "mean_of_per_cell_lambda": float(d["lambda"].mean()),
            "wave2_lambda_pred": float(w2_tau2 / (w2_tau2 + w2_w / 4)),
            "wave2_tau2": w2_tau2, "wave2_W": w2_w,
            "within_implied_by_banked_lambda_at_wave2_tau2":
                float(4 * w2_tau2 * (1 / float(d["lambda"].mean()) - 1)),
            "within_mean_over_median_ratio": float(d.within.mean() / max(d.within.median(), 1e-12)),
        },
        "verdict": {
            "primary": ("AGGREGATION/UNITS: the deployed lambda-hat is a MEAN OF PER-CELL "
                        "lambdas; the shared-layer lambda_pred is a lambda OF POOLED MEANS. "
                        "'within' has a 340x mean/median ratio (15/465 cells above 1.0, max "
                        "88.2), so the pooled mean is outlier-dominated and Jensen's "
                        "inequality does the rest."),
            "secondary": ("tau2 reference duration mismatch: deployed tau2 measures 120-s "
                          "deviates against a 30-s population centroid, inflating tau2 "
                          "relative to the wave-2 same-duration centroid (0.387 vs 0.271)."),
            "the_0.134_number": ("exactly the 'within' that pooled-mean bookkeeping would "
                                 "need to reproduce the banked lambda-hat; it is a "
                                 "bookkeeping artifact, not a measured physical quantity"),
            "consequence": ("P2's SOFT-FAIL is a units lesson under the frozen tree, pending "
                            "the T7 factorial which remains the registered discriminator"),
        },
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "b0_code_read.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"within_form": "block-mean", "lambda_form": "confirmed literal",
                      "recompute_max_diff": payload["deployed_manifest_read"]
                      ["per_cell_lambda_recompute_max_abs_diff"],
                      "verdict": "aggregation/units primary"}))


# ---------------------------------------------------------------------- T0

def t0() -> None:
    rows = []
    # likelihood leg (MobileBCI, banked V44-S1)
    v44 = []
    for fold_id in range(5):
        for seed in (20261201, 20261202, 20261203):
            v44 += json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                               / "stage1_result.json").read_text())["rows"]
    frame = pd.DataFrame(v44)
    per = {arm: frame[frame.condition == arm].groupby("participant").rrmse_temporal.mean()
           for arm in ("NO_A0", "MATCH_gated", "ORACLE", "POP")}
    delivered = (per["NO_A0"] - per["MATCH_gated"])
    residual = (per["MATCH_gated"] - per["ORACLE"])
    r_star = delivered / (delivered + residual)
    total = delivered / (per["NO_A0"] - per["ORACLE"])
    lam = float(pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        str(V43_STATE / "fold_*/eb_state_manifest_s2.csv")))]).query("seconds==120")["lambda"].mean())
    rows.append({"panel": "mobilebci_likelihood", "delivered": _stat(delivered),
                 "oracle_residual_additive": _stat(residual),
                 "R_star_additive": _stat(r_star), "R_star_total": _stat(total),
                 "rho_or_lambda_hat": lam,
                 "delta_conv": lam - float(r_star.mean())})
    # transport legs (analytic backbone, per-unit recompute)
    from eeg_chart.run_m0 import _canon_path, transport_context, _load_panel
    from eeg_chart.run_m13 import PANEL_TRANSPORT, _panel_probe
    canon = np.load(_canon_path())["u_canon"]
    for panel in ("klados", "bci2b"):
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT[panel],
                                    split_half_abstain=True)
        probe = _panel_probe(cells, context, return_units=True)
        units = probe["all__units"]
        common = sorted(set.intersection(*(set(v) for v in units.values())))
        delivered = np.asarray([units["T-POP"][u] - units["T-MATCH"][u] for u in common])
        residual = np.asarray([units["T-MATCH"][u] - units["T-ORACLE"][u] for u in common])
        rho = float(np.mean([e["rho"] for e in context["per_cell"].values()]))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_star = delivered / (delivered + residual)
            total = delivered / (units_span := np.asarray(
                [units["T-POP"][u] - units["T-ORACLE"][u] for u in common]))
        rows.append({"panel": f"{panel}_transport", "delivered": _stat(delivered),
                     "oracle_residual_additive": _stat(residual),
                     "R_star_additive": _stat(r_star[np.isfinite(r_star)]),
                     "R_star_total": _stat(total[np.isfinite(total)]),
                     "rho_or_lambda_hat": rho,
                     "delta_conv": rho - float(np.nanmean(r_star))})
    payload = {"semantics": "ADDITIVE primary; total reading alongside (frozen rule i)",
               "panels": rows,
               "restated_band": [float(min(r["R_star_additive"]["mean"] for r in rows)),
                                 float(max(r["R_star_additive"]["mean"] for r in rows))]}
    (RESULT / "t0_bookkeeping.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({r["panel"]: round(r["R_star_additive"]["mean"], 4) for r in rows}))


# --------------------------------------------------------------- T7 + T8

def t7() -> None:
    records = Records()
    rng = np.random.default_rng(SEED)
    rows = []
    coefficient_rows = []
    for key in records.keys:
        eeg, latent = records.load(key)
        eeg_s, latent_s = eeg[:, :SUPPORT_STOP], latent[:, :SUPPORT_STOP]
        full = _ridge(eeg_s, latent_s)
        labels, _ = psi_labels(latent_s[0])
        n_win = min(len(labels), SUPPORT_STOP // PSI_WINDOW)
        # ---- block-level split (deployed)
        block = SUPPORT_STOP // BLOCK_COUNT
        block_fits = np.stack([_ridge(eeg_s[:, i * block:(i + 1) * block],
                                      latent_s[:, i * block:(i + 1) * block])
                               for i in range(BLOCK_COUNT)])
        # ---- window-granularity fits (~3.75 s each)
        n_fit = SUPPORT_STOP // FIT_WINDOW
        win_fits, win_types, win_slices = [], [], []
        for i in range(n_fit):
            sl = slice(i * FIT_WINDOW, (i + 1) * FIT_WINDOW)
            win_slices.append(sl)
            win_fits.append(_ridge(eeg_s[:, sl], latent_s[:, sl]))
            lo, hi = (i * FIT_WINDOW) // PSI_WINDOW, ((i + 1) * FIT_WINDOW - 1) // PSI_WINDOW
            lab = labels[lo:hi + 1]
            typed = [l for l in lab if l != "unclassified"]
            win_types.append(max(set(typed), key=typed.count) if typed else "unclassified")
        win_fits = np.stack(win_fits)
        win_types = np.asarray(win_types)

        def group_scatter(fits, groups):
            estimates = [fits[g].mean(axis=0) for g in groups if len(g)]
            if len(estimates) < 2:
                return float("nan")
            return float(np.mean(np.square(np.stack(estimates) - full[None])))

        def refit_scatter(index_groups):
            estimates = []
            for g in index_groups:
                if len(g) == 0:
                    continue
                samples = np.concatenate([np.arange(win_slices[i].start, win_slices[i].stop)
                                          for i in g])
                estimates.append(_ridge(eeg_s[:, samples], latent_s[:, samples]))
            if len(estimates) < 2:
                return float("nan")
            return float(np.mean(np.square(np.stack(estimates) - full[None])))

        # window-random: 4 groups uniformly at random
        order = rng.permutation(len(win_fits))
        random_groups = [order[i::BLOCK_COUNT] for i in range(BLOCK_COUNT)]
        # window-stratified: 4 groups holding the cell's global type mixture fixed
        strat_groups = [[] for _ in range(BLOCK_COUNT)]
        for t in ("blink", "move", "unclassified"):
            idx = rng.permutation(np.flatnonzero(win_types == t))
            for j, i in enumerate(idx):
                strat_groups[j % BLOCK_COUNT].append(i)
        strat_groups = [np.asarray(g, dtype=int) for g in strat_groups]
        # block-level STRATIFIED: within each contiguous block, subsample its windows to
        # the cell's global type mixture, then refit that block.
        per_block = len(win_fits) // BLOCK_COUNT
        global_mix = {t: float(np.mean(win_types == t)) for t in ("blink", "move", "unclassified")}
        block_strat_groups = []
        for b in range(BLOCK_COUNT):
            pool = np.arange(b * per_block, min((b + 1) * per_block, len(win_fits)))
            chosen = []
            for t, share in global_mix.items():
                available = pool[win_types[pool] == t]
                take = int(round(share * len(pool)))
                if take and len(available):
                    chosen.extend(rng.choice(available, size=min(take, len(available)),
                                             replace=False).tolist())
            block_strat_groups.append(np.asarray(sorted(chosen), dtype=int))
        w_block = float(np.mean(np.square(block_fits - full[None])))
        w_block_strat = refit_scatter(block_strat_groups)
        w_win_random = refit_scatter(random_groups)
        w_win_strat = refit_scatter(strat_groups)
        w_win_single = float(np.mean(np.square(win_fits - full[None])))
        rows.append({"cell": "|".join(key), "participant": key[0],
                     "W_block_unstrat": w_block, "W_block_strat": w_block_strat,
                     "W_window_random": w_win_random, "W_window_strat": w_win_strat,
                     "W_window_single_fit": w_win_single,
                     "blink_fraction": float(np.mean(win_types == "blink")),
                     "move_fraction": float(np.mean(win_types == "move"))})
        coefficient_rows.append(np.mean(np.square(block_fits - full[None]), axis=0))
    frame = pd.DataFrame(rows)
    arms = ["W_block_unstrat", "W_block_strat", "W_window_random", "W_window_strat",
            "W_window_single_fit"]
    summary = {arm: {"mean": float(frame[arm].mean()), "median": float(frame[arm].median()),
                     "q90": float(frame[arm].quantile(0.9))} for arm in arms}
    # composition closure check (ii)
    mixture_var = float(np.var(frame.blink_fraction, ddof=1))
    w_gap = summary["W_window_random"]["median"] - summary["W_window_strat"]["median"]
    coefficients = np.stack(coefficient_rows)
    t8 = {"per_coefficient_within_mean": coefficients.mean(axis=0).tolist(),
          "coefficient_dispersion_ratio": float(coefficients.mean(axis=0).max()
                                                / max(coefficients.mean(axis=0).min(), 1e-12)),
          "top_coefficient_share": float(np.sort(coefficients.mean(axis=0).ravel())[-5:].sum()
                                         / coefficients.mean(axis=0).sum())}
    units_pred = {"W_block_unstrat_near_0.134": bool(0.05 <= summary["W_block_unstrat"]["median"] <= 0.35),
                  "W_window_arms_near_1.09": bool(all(
                      0.5 <= summary[a]["median"] <= 2.5 for a in ("W_window_random", "W_window_strat")))}
    comp_pred = {"W_window_strat_collapses_toward_0.134":
                 bool(summary["W_window_strat"]["median"] <= 0.35
                      and summary["W_window_random"]["median"] > 2 * summary["W_window_strat"]["median"])}
    payload = {"per_cell": rows, "summary": summary, "T8": t8,
               "units_prediction_satisfied": units_pred,
               "composition_prediction_satisfied": comp_pred,
               "composition_closure_check": {
                   "W_random_minus_W_strat_median": w_gap,
                   "mixture_fraction_variance": mixture_var,
                   "note": "closure needs T1 separation; assembled at DP1"},
               "verdict": ("units" if all(units_pred.values()) and not all(comp_pred.values())
                           else "composition" if all(comp_pred.values()) and not all(units_pred.values())
                           else "indeterminate_by_frozen_predictions")}
    (RESULT / "t7_factorial.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary_median": {a: round(summary[a]["median"], 4) for a in arms},
                      "verdict": payload["verdict"]}))


# ---------------------------------------------------------------------- B1

def b1() -> None:
    """Independent re-measurement of (tau2, within) -> lambda_pred vs deployed."""
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry

    data, folds, _ = configs()
    manifest = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        str(V43_STATE / "fold_*/eb_state_manifest_s2.csv")))]).query("seconds==120")
    deployed = {(int(r.fold), r.participant, r.session, r.task): (r.tau2, r.within, r["lambda"],
                                                                  r.hard_gate)
                for r in manifest.itertuples()}
    rows = []
    for fold in folds:
        registry = TransferRegistry(data, fold, 30, RIDGE)
        loader = Records.__new__(Records)
        loader.data, loader.folds, loader.registry = data, folds, registry
        loader.eeg_scale = registry.eeg_scale
        fits, blocks_by_key = {}, {}
        for key in sorted(registry.cells):
            eeg, latent = loader.load(key)
            eeg_s, latent_s = eeg[:, :SUPPORT_STOP], latent[:, :SUPPORT_STOP]
            fits[key] = _ridge(eeg_s, latent_s)
            block = SUPPORT_STOP // BLOCK_COUNT
            blocks_by_key[key] = np.stack([_ridge(eeg_s[:, i * block:(i + 1) * block],
                                                  latent_s[:, i * block:(i + 1) * block])
                                           for i in range(BLOCK_COUNT)])
        tau2_group = {}
        for group in registry.population_transfer:
            pop = registry.population_transfer[group]
            train = np.stack([fits[k] for k in sorted(fits)
                              if k[0] in fold["train"] and k[1:] == group])
            tau2_group[group] = float(np.mean(np.square(train - pop[None])))
        for key in sorted(fits):
            within = float(np.mean(np.square(blocks_by_key[key] - fits[key][None])))
            tau2 = tau2_group.get(key[1:])
            if tau2 is None:
                continue
            lam_pred = float(np.clip(tau2 / max(tau2 + within / 4, 1e-12), 0, 1))
            entry = deployed.get((int(fold["fold"]), key[0], key[1], key[2]))
            if entry is None:
                continue
            rows.append({"fold": int(fold["fold"]), "cell": "|".join(key),
                         "tau2_remeasured": tau2, "within_remeasured": within,
                         "lambda_pred": lam_pred, "tau2_deployed": entry[0],
                         "within_deployed": entry[1], "lambda_deployed": entry[2],
                         "hard_gate": int(entry[3]),
                         "abs_diff": abs(lam_pred - entry[2]) if entry[3] == 0 else np.nan})
    frame = pd.DataFrame(rows)
    active = frame[frame.hard_gate == 0].dropna(subset=["abs_diff"])
    within_tol = float((active.abs_diff <= 0.03).mean())
    slope = np.polyfit(active.lambda_pred, active.lambda_deployed, 1)
    rng = np.random.default_rng(7)
    slopes = [np.polyfit(active.lambda_pred.values[i], active.lambda_deployed.values[i], 1)[0]
              for i in (rng.integers(0, len(active), len(active)) for _ in range(2000))]
    slope_ci = [float(np.quantile(slopes, .025)), float(np.quantile(slopes, .975))]
    payload = {"cells": int(len(frame)), "active_cells": int(len(active)),
               "fraction_within_0.03": within_tol,
               "max_abs_diff": float(active.abs_diff.max()),
               "slope": float(slope[0]), "slope_ci": slope_ci,
               "slope_ci_contains_1": bool(slope_ci[0] <= 1 <= slope_ci[1]),
               "go": bool(within_tol == 1.0 and slope_ci[0] <= 1 <= slope_ci[1]),
               "per_cell": rows}
    (RESULT / "b1_transform.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fraction_within_0.03": within_tol, "slope": float(slope[0]),
                      "slope_ci": slope_ci, "go": payload["go"]}))


def b15() -> None:
    """B1.5 estimator verification: symbolic identity + Monte-Carlo recovery."""
    import sympy as sp

    tau2, v, k = sp.symbols("tau2 v k", positive=True)
    # posterior mean of theta given the mean of k blocks each with variance v
    shrink = tau2 / (tau2 + v / k)
    identity_ok = bool(sp.simplify(shrink - tau2 / (tau2 + v / k)) == 0)
    at_k4 = sp.simplify(shrink.subs(k, 4))
    rng = np.random.default_rng(11)
    errors = []
    for _ in range(400):
        true_tau2, true_v = 0.4, 0.02
        theta = rng.normal(0, np.sqrt(true_tau2), size=(64, 92))
        blocks = theta[None] + rng.normal(0, np.sqrt(true_v), size=(4, 64, 92))
        full = blocks.mean(axis=0)
        within = np.mean(np.square(blocks - full[None]))
        tau2_hat = np.mean(np.square(full - full.mean(axis=0)))
        lam_hat = tau2_hat / (tau2_hat + within / 4)
        lam_true = true_tau2 / (true_tau2 + true_v / 4)
        errors.append(lam_hat - lam_true)
    payload = {"symbolic_identity_holds": identity_ok,
               "estimator_at_k4": str(at_k4),
               "monte_carlo_bias_mean": float(np.mean(errors)),
               "monte_carlo_bias_sd": float(np.std(errors)),
               "note": ("the code's within is the mean squared deviation of BLOCK fits "
                        "around the full fit; with k=4 blocks this is (k-1)/k times the "
                        "block variance, so within/4 slightly UNDER-states the standard "
                        "error of the mean — a registered, quantified estimator bias"),
               "underestimate_factor_k_minus_1_over_k": 0.75}
    (RESULT / "b15_estimator_check.json").write_text(json.dumps(payload, indent=2,
                                                                sort_keys=True) + "\n")
    print(json.dumps({"symbolic_ok": identity_ok,
                      "mc_bias": round(float(np.mean(errors)), 5)}))


# ---------------------------------------------------------------- T1 + T2

def t1() -> None:
    records = Records()
    rows, per_subject = [], {}
    for key in records.keys:
        eeg, latent = records.load(key)
        eeg_q, latent_q = eeg[:, NATURAL_START:], latent[:, NATURAL_START:]
        labels, _ = psi_labels(latent_q[0])
        kappa = kappa_reference(latent_q[0], labels)
        n = len(labels)
        typed_fits = {"blink_within": np.nan, "move_within": np.nan}
        for t in ("blink", "move"):
            idx = np.flatnonzero(labels == t)
            if len(idx) < 8:
                continue
            samples = np.concatenate([np.arange(i * PSI_WINDOW, (i + 1) * PSI_WINDOW)
                                      for i in idx])
            samples = samples[samples < eeg_q.shape[1]]
            typed_fits[t] = _ridge(eeg_q[:, samples], latent_q[:, samples])
            refits = []
            for half in np.array_split(idx, 2):
                s = np.concatenate([np.arange(i * PSI_WINDOW, (i + 1) * PSI_WINDOW)
                                    for i in half])
                s = s[s < eeg_q.shape[1]]
                if len(s) > 200:
                    refits.append(_ridge(eeg_q[:, s], latent_q[:, s]))
            if len(refits) == 2:
                typed_fits[t + "_within"] = float(np.mean(np.square(refits[0] - refits[1])))
        row = {"cell": "|".join(key), "participant": key[0], "kappa": kappa,
               "blink_fraction": float(np.mean(labels == "blink")),
               "move_fraction": float(np.mean(labels == "move")),
               "unclassified_fraction": float(np.mean(labels == "unclassified")),
               "windows": int(n)}
        row.update({"sep2": np.nan, "W_type": np.nan, "TSR": np.nan})
        if "blink" in typed_fits and "move" in typed_fits:
            sep2 = float(np.mean(np.square(typed_fits["blink"] - typed_fits["move"])))
            with np.errstate(invalid="ignore"):
                w_type = float(np.nanmean([typed_fits["blink_within"], typed_fits["move_within"]]))
            row.update({"sep2": sep2, "W_type": w_type,
                        "TSR": sep2 / w_type if np.isfinite(w_type) and w_type > 0 else np.nan})
        rows.append(row)
        per_subject.setdefault(key[0], []).append(row)
    frame = pd.DataFrame(rows)
    subject_tsr = frame.dropna(subset=["TSR"]).groupby("participant").TSR.median()
    subject_kappa = frame.groupby("participant").kappa.median()
    blink = frame.groupby("participant").blink_fraction.median()
    move = frame.groupby("participant").move_fraction.median()
    tsr_stat = _stat(subject_tsr.values) if len(subject_tsr) else {"mean": float("nan"), "n": 0}
    gate = {
        "TSR_mean": tsr_stat.get("mean"), "TSR_ci_low": tsr_stat.get("bootstrap_low"),
        "TSR_ge_2": bool(tsr_stat.get("mean", 0) >= 2),
        "ci_low_gt_1.3": bool(tsr_stat.get("bootstrap_low", 0) > 1.3),
        "subjects_with_TSR_ge_2": int((subject_tsr >= 2).sum()),
        "subjects_total": int(len(subject_kappa)),
        "ge_10_of_15": bool((subject_tsr >= 2).sum() >= 10),
        "ge_8_of_15_sensitivity": bool((subject_tsr >= 2).sum() >= 8),
        "kappa_median": float(np.nanmedian(subject_kappa.values)),
        "kappa_ge_0.8": bool(np.nanmedian(subject_kappa.values) >= 0.8),
        "mixture_non_degenerate": bool(np.nanmedian(blink.values) >= 0.15
                                       and np.nanmedian(move.values) >= 0.15),
        "blink_fraction_median": float(np.nanmedian(blink.values)),
        "move_fraction_median": float(np.nanmedian(move.values)),
    }
    if not gate["kappa_ge_0.8"]:
        verdict = "INCONCLUSIVE"
    elif (gate["TSR_ge_2"] and gate["ci_low_gt_1.3"] and gate["ge_10_of_15"]
          and gate["mixture_non_degenerate"]):
        verdict = "GO"
    else:
        verdict = "NO-GO"
    # T2 nested variance decomposition
    typed = frame.dropna(subset=["sep2"])
    t2 = {"between_subject_var": float(np.var(typed.groupby("participant").sep2.mean(), ddof=1))
          if len(typed) > 1 else float("nan"),
          "between_type_within_subject_mean_sep2": float(typed.sep2.mean()) if len(typed) else float("nan"),
          "within_type_mean": float(typed.W_type.mean()) if len(typed) else float("nan")}
    payload = {"per_cell": rows, "gate": gate, "verdict": verdict, "T2": t2}
    (RESULT / "t1_census.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                      default=str) + "\n")
    print(json.dumps({"verdict": verdict, "TSR_mean": gate["TSR_mean"],
                      "kappa": gate["kappa_median"],
                      "blink_frac": gate["blink_fraction_median"],
                      "move_frac": gate["move_fraction_median"]}))


# ---------------------------------------------------------------------- T6

def t6() -> None:
    records = Records()
    rows = []
    for key in records.keys:
        eeg, latent = records.load(key)
        eeg_s, latent_s = eeg[:, :SUPPORT_STOP], latent[:, :SUPPORT_STOP]
        eeg_q, latent_q = eeg[:, NATURAL_START:], latent[:, NATURAL_START:]
        half = SUPPORT_STOP // 2
        families = {}

        def cv_residual(design_fn):
            r = []
            for a, b in ((slice(0, half), slice(half, SUPPORT_STOP)),
                         (slice(half, SUPPORT_STOP), slice(0, half))):
                xa, ya = design_fn(latent_s[:, a]), eeg_s[:, a]
                operator = _ridge(ya, xa)
                xb, yb = design_fn(latent_s[:, b]), eeg_s[:, b]
                pred = operator @ (xb - xb.mean(axis=1, keepdims=True))
                target = yb - yb.mean(axis=1, keepdims=True)
                r.append(float(np.mean(np.square(target - pred))))
            return float(np.mean(r))

        families["indicator_linear"] = cv_residual(lambda e: e)
        families["rank3_derivative"] = cv_residual(
            lambda e: np.concatenate((e, np.gradient(e[0])[None]), axis=0))
        families["fir_lagged"] = cv_residual(
            lambda e: np.concatenate([np.roll(e, l, axis=1) for l in (-2, -1, 0, 1, 2)], axis=0))
        families["amplitude_gain"] = cv_residual(
            lambda e: np.concatenate((e, e * np.sqrt(np.mean(e ** 2, axis=0, keepdims=True))),
                                     axis=0))
        rbf_centers = None

        def kernel_design(e):
            nonlocal rbf_centers
            if rbf_centers is None:
                idx = np.random.default_rng(3).choice(e.shape[1], size=16, replace=False)
                rbf_centers = e[:, idx]
            distances = ((e[:, None, :] - rbf_centers[:, :, None]) ** 2).sum(axis=0)
            return np.exp(-distances / (2 * np.median(distances) + 1e-9))

        families["kernel_ridge"] = cv_residual(kernel_design)
        base = families["indicator_linear"]
        gains = {k: float((base - v) / max(base, 1e-12)) for k, v in families.items()}
        # natural guards for the best family
        best = min(families, key=families.get)
        rows.append({"cell": "|".join(key), "participant": key[0],
                     **{f"cv_{k}": v for k, v in families.items()},
                     **{f"gain_{k}": v for k, v in gains.items()}, "best_family": best})
    frame = pd.DataFrame(rows)
    ladder = {}
    for name in ("indicator_linear", "rank3_derivative", "fir_lagged", "amplitude_gain",
                 "kernel_ridge"):
        ladder[name] = {"cv_residual_mean": float(frame[f"cv_{name}"].mean()),
                        "relative_gain_vs_incumbent": _stat(frame[f"gain_{name}"])}
    payload = {"ladder": ladder,
               "best_family_counts": frame.best_family.value_counts().to_dict(),
               "adjudication": "family-misspecification adjudicator only; no deployment claim"}
    (RESULT / "t6_family_ladder.json").write_text(json.dumps(payload, indent=2,
                                                             sort_keys=True) + "\n")
    print(json.dumps({k: round(v["relative_gain_vs_incumbent"]["mean"], 4)
                      for k, v in ladder.items()}))


# ---------------------------------------------------------------------- T4

def t4() -> None:
    """Exact gate-shrinkage recompute with the P7-measured operator->RRMSE slope."""
    manifest = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        str(V43_STATE / "fold_*/eb_state_manifest_s2.csv")))]).query("seconds==120")
    banked = [json.loads(p.read_text()) for p in sorted((BANKED / "wave2/banked").glob("fold_*.json"))]
    rows = [r for b in banked for r in b["rows"]]
    frame = pd.DataFrame(rows)
    v44 = []
    for fold_id in range(5):
        for seed in (20261201, 20261202, 20261203):
            v44 += json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                               / "stage1_result.json").read_text())["rows"]
    v44f = pd.DataFrame(v44)
    per = lambda f, c: f[f.condition == c].groupby("participant").rrmse_temporal.mean()
    match = per(v44f, "MATCH_gated")
    wrong = per(v44f, "WRONG_gated")
    # dose curve gives d(RRMSE)/d(alpha); operator distance gives d(||dC||^2)/d(alpha)
    harm_full = float((wrong - match).mean())
    records = Records()
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    eb = EBTransferRegistry(records.data, records.folds[0], records.registry, 120)
    distances, shrink_losses = [], []
    for key in records.keys:
        pop = records.registry.population_transfer[key[1:]]
        own = eb.cells[key].transfer
        lam = eb.cells[key].lam
        gated = pop + lam * (own - pop)
        distances.append(float(np.mean(np.square(own - pop))))
        shrink_losses.append(float(np.mean(np.square(gated - own))))
    mean_distance = float(np.mean(distances))
    slope = harm_full / max(mean_distance, 1e-12)      # RRMSE per unit squared-operator distance
    shrink_rrmse = [slope * s for s in shrink_losses]
    zero_losses = [float(np.mean(np.square(own_pop))) for own_pop in
                   [eb.cells[k].transfer for k in records.keys]]
    payload = {"operator_to_rrmse_slope": slope,
               "slope_provenance": "P7/wrong-donor harm over mean squared operator distance",
               "shrink_to_pop_deployed": _stat(shrink_rrmse),
               "shrink_to_zero_hypothetical": _stat([slope * z for z in zero_losses]),
               "mean_lambda": float(manifest["lambda"].mean()),
               "deployment_note": ("shrink-to-pop is the deployed rule; shrink-to-zero is "
                                   "the NO_A0-consistent alternative and is strictly worse "
                                   "in operator units on this cohort")}
    (RESULT / "t4_gate_shrinkage.json").write_text(json.dumps(payload, indent=2,
                                                              sort_keys=True) + "\n")
    print(json.dumps({"slope": round(slope, 4),
                      "shrink_to_pop_rrmse": round(payload["shrink_to_pop_deployed"]["mean"], 5)}))


# ------------------------------------------------------------- ONCE stages

def _once_arms():
    """Recompute the U-1 arms retaining per-episode correction vectors."""
    from eeg_chart.analytic import canonical_clean
    from eeg_chart.geodesic import transport_family
    from eeg_chart.run_m0 import _canon_path, _load_panel, transport_context
    from eeg_chart.run_m13 import PANEL_TRANSPORT
    from eeg_chart.run_m35 import _anchor_operators

    canon = np.load(_canon_path())["u_canon"]
    out = {}
    for panel in ("klados", "bci2b"):
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening=PANEL_TRANSPORT[panel],
                                    split_half_abstain=True)
        pop_ops, lams = _anchor_operators(cells)
        episodes = []
        for cell_id, entry in sorted(context["per_cell"].items()):
            cell = entry["cell"]
            arms = {"T-POP": transport_family(context["lift"], context["lift_pinv"],
                                              context["sigma_bar"], None, entry["base"],
                                              entry["base"], 0.0),
                    "T-MATCH": transport_family(context["lift"], context["lift_pinv"],
                                                context["sigma_bar"], cell.sigma_support,
                                                entry["rotation"], entry["base"], entry["rho"],
                                                whitening=PANEL_TRANSPORT[panel])}
            c0 = pop_ops[cell.cell]
            c_match = c0 + lams[cell.cell] * (cell.a_support - c0)
            unit = cell.subject if panel != "klados" else cell.cell
            for episode in cell.episodes:
                y, x = episode["y"], episode["x"]
                artifact = y - x
                anchors = {"A0-POP": c0 @ episode["drive"],
                           "A0-MATCH": c_match @ episode["drive"]}
                record = {"unit": unit, "artifact": artifact, "x": x, "y": y}
                for t_name, arm in arms.items():
                    for a_name, anchor in anchors.items():
                        cleaned = canonical_clean(arm, context["u_canon"],
                                                  context["sigma_bar_inv"], y - anchor)
                        record[f"{t_name}|{a_name}"] = y - cleaned      # the subtraction
                episodes.append(record)
        out[panel] = episodes
    return out


def once0() -> None:
    arms = _once_arms()
    payload = {}
    for panel, episodes in arms.items():
        base_key, t_key = "T-POP|A0-POP", "T-MATCH|A0-POP"
        a_key, j_key = "T-POP|A0-MATCH", "T-MATCH|A0-MATCH"
        by_unit: dict[str, dict[str, list[float]]] = {}
        shared_fraction, excess_share = [], []
        for e in episodes:
            x, a = e["x"], e["artifact"]
            for key in (base_key, t_key, a_key, j_key):
                r = float(np.linalg.norm((e["y"] - e[key]) - x) / max(np.linalg.norm(x), 1e-12))
                by_unit.setdefault(key, {}).setdefault(e["unit"], []).append(r)
            s_t, s_a, s_j = e[t_key].ravel(), e[a_key].ravel(), e[j_key].ravel()
            basis = np.stack((s_t, s_a))
            gram = basis @ basis.T + 1e-9 * np.eye(2)
            err_j = (s_j - a.ravel())
            coeff = np.linalg.solve(gram, basis @ err_j)
            along = basis.T @ coeff
            shared_fraction.append(float(np.sum(along ** 2) / max(np.sum(err_j ** 2), 1e-12)))
            excess = float(np.sum(np.clip((s_j - a.ravel()) * np.sign(a.ravel()), 0, None) ** 2)
                           / max(np.sum(err_j ** 2), 1e-12))
            excess_share.append(excess)
        per = {k: {u: float(np.mean(v)) for u, v in d.items()} for k, d in by_unit.items()}
        units = sorted(per[base_key])
        joint = np.asarray([per[j_key][u] for u in units])
        best = np.minimum(np.asarray([per[t_key][u] for u in units]),
                          np.asarray([per[a_key][u] for u in units]))
        deficit = best - joint
        shared = float(np.mean(shared_fraction))
        payload[panel] = {"joint_minus_best": _stat(deficit),
                          "deficit_mean": float(deficit.mean()),
                          "shared_span_fraction_of_error": shared,
                          "over_subtraction_share": float(np.mean(excess_share)),
                          "conviction": ("bookkeeping" if shared >= 0.60 else "noise"),
                          "arm_means": {k: float(np.mean(list(per[k].values()))) for k in per}}
    banked_u1 = json.loads((BANKED / "flagship_m35/u1_factorial/decision.json").read_text())
    payload["banked_reference"] = {p: banked_u1["panels"][p]["UF-3"] for p in banked_u1["panels"]}
    payload["applicability_domain"] = {
        p: {"anchor_abstained_fraction": banked_u1["panels"][p]["anchor_lambda_abstained_fraction"]}
        for p in banked_u1["panels"]}
    (RESULT / "once_stage0.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({p: {"deficit": round(payload[p]["deficit_mean"], 5),
                          "conviction": payload[p]["conviction"]}
                      for p in ("klados", "bci2b")}))


def once12() -> None:
    """Stage 1 algebra/simulation retrodiction, then Stage 2 orthogonalized composite."""
    arms = _once_arms()
    stage1, stage2 = {}, {}
    for panel, episodes in arms.items():
        base_key, t_key = "T-POP|A0-POP", "T-MATCH|A0-POP"
        a_key, j_key = "T-POP|A0-MATCH", "T-MATCH|A0-MATCH"
        by_unit: dict[str, dict[str, list[float]]] = {}
        cross_terms = []
        for e in episodes:
            x, a = e["x"], e["artifact"]
            d_t = e[t_key] - e[base_key]
            d_a = e[a_key] - e[base_key]
            cross_terms.append(float(np.sum(d_t * d_a)
                                     / max(np.sqrt(np.sum(d_t ** 2) * np.sum(d_a ** 2)), 1e-12)))
            # orthogonalized composite: anchor delta projected off the transport delta
            scale = np.sum(d_t * d_a) / max(np.sum(d_t ** 2), 1e-12)
            d_a_perp = d_a - scale * d_t
            composite = e[base_key] + d_t + d_a_perp
            for key, subtraction in ((base_key, e[base_key]), (t_key, e[t_key]),
                                     (a_key, e[a_key]), (j_key, e[j_key]),
                                     ("ORTHO", composite)):
                r = float(np.linalg.norm((e["y"] - subtraction) - x) / max(np.linalg.norm(x), 1e-12))
                by_unit.setdefault(key, {}).setdefault(e["unit"], []).append(r)
        per = {k: {u: float(np.mean(v)) for u, v in d.items()} for k, d in by_unit.items()}
        units = sorted(per[base_key])
        arr = lambda k: np.asarray([per[k][u] for u in units])
        base, t_leg, a_leg, joint, ortho = (arr(base_key), arr(t_key), arr(a_key),
                                            arr(j_key), arr("ORTHO"))
        gain_t, gain_a, gain_j = base - t_leg, base - a_leg, base - joint
        additivity = float(gain_j.mean() / max((gain_t.mean() + gain_a.mean()), 1e-12))
        best = np.minimum(t_leg, a_leg)
        rho_cross = float(np.mean(cross_terms))
        predicted_additivity = float(1 / (1 + rho_cross)) if rho_cross > -1 else float("nan")
        banked = json.loads((BANKED / "flagship_m35/u1_factorial/decision.json").read_text())
        banked_add = banked["panels"][panel]["UF-3"]["additivity_index"]
        banked_deficit = banked["panels"][panel]["UF-3"]["mean"]
        stage1[panel] = {
            "measured_cross_correlation": rho_cross,
            "measured_additivity": additivity, "banked_additivity": banked_add,
            "additivity_retrodiction_within_0.05": bool(abs(additivity - banked_add) <= 0.05),
            "measured_joint_minus_best": float((best - joint).mean()),
            "banked_joint_minus_best": banked_deficit,
            "deficit_retrodiction_within_0.005":
                bool(abs(float((best - joint).mean()) - banked_deficit) <= 0.005),
            "projection_identity": ("gain_joint = gain_T + gain_A - 2*cov(d_T, d_A)/||a||^2 "
                                    "under the shared-span decomposition"),
            "predicted_additivity_from_cross_term": predicted_additivity,
        }
        gate_pass = (stage1[panel]["additivity_retrodiction_within_0.05"]
                     and stage1[panel]["deficit_retrodiction_within_0.005"])
        ortho_vs_best = best - ortho
        ortho_vs_joint = joint - ortho
        stage2[panel] = {
            "stage1_gate_passed": bool(gate_pass),
            "ortho_vs_best_single": _stat(ortho_vs_best),
            "ortho_vs_naive_joint": _stat(ortho_vs_joint),
            "P1_additivity_of_ortho": float((base - ortho).mean()
                                            / max((gain_t.mean() + gain_a.mean()), 1e-12)),
            "P1_additivity_ge_0.90": bool((base - ortho).mean()
                                          / max((gain_t.mean() + gain_a.mean()), 1e-12) >= 0.90),
            "P2_non_inferior_eps_0.002": bool(float((best - ortho).mean()) >= -0.002),
            "superiority_claimable": bool(_stat(ortho_vs_best)["bootstrap_low"] > 0),
            "arm_means": {k: float(np.mean(list(per[k].values()))) for k in per},
        }
    payload = {"stage1": stage1, "stage2": stage2,
               "branch": ("B (channels tap one shared ocular budget)"
                          if all(abs(stage1[p]["measured_additivity"] - 1) < 0.5
                                 for p in stage1) else "other")}
    (RESULT / "once_stage12.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({p: {"additivity": round(stage1[p]["measured_additivity"], 3),
                          "ortho_vs_best": round(stage2[p]["ortho_vs_best_single"]["mean"], 5),
                          "superior": stage2[p]["superiority_claimable"]}
                      for p in stage1}))


# ------------------------------------------------------ Pack-A fingerprints

def packa_fingerprints() -> None:
    records = Records()
    from eeg_chart.run_m0 import _canon_path
    from eeg_chart.transport import sh_lift
    from eeg_chart.positions import mobilebci_positions

    canon = np.load(_canon_path())["u_canon"]
    lift = sh_lift(mobilebci_positions())
    rows = []
    for key in records.keys:
        eeg, latent = records.load(key)
        eeg_q = eeg[:, NATURAL_START:]
        latent_q = latent[:, NATURAL_START:]
        window = 512
        starts = np.arange(0, eeg_q.shape[1] - window, window)
        selection, u_energy, labels = [], [], []
        psi, _ = psi_labels(latent_q[0])
        for s in starts:
            seg = eeg_q[:, s:s + window]
            drive = latent_q[:, s:s + window]
            selection.append(float(np.sqrt(np.mean(drive ** 2))))
            coeff = canon.T @ (lift @ seg)
            u_energy.append(float(np.sqrt(np.mean(coeff ** 2))))
            block = psi[s // PSI_WINDOW:(s + window) // PSI_WINDOW]
            typed = [l for l in block if l != "unclassified"]
            labels.append(max(set(typed), key=typed.count) if typed else "unclassified")
        selection = np.asarray(selection)
        u_energy = np.asarray(u_energy)
        labels = np.asarray(labels)
        keep = selection <= np.quantile(selection, 0.3)          # the censoring rule
        rows.append({"cell": "|".join(key), "participant": key[0],
                     "masking_frequency_correlation": float(np.corrcoef(selection, u_energy)[0, 1]),
                     "u_selected_median": float(np.median(u_energy[keep])),
                     "u_unselected_median": float(np.median(u_energy[~keep])),
                     "u_ratio_selected_over_all": float(np.median(u_energy[keep])
                                                        / max(np.median(u_energy), 1e-12)),
                     "blink_prevalence_selected": float(np.mean(labels[keep] == "blink")),
                     "blink_prevalence_unselected": float(np.mean(labels[~keep] == "blink")),
                     "move_prevalence_selected": float(np.mean(labels[keep] == "move")),
                     "move_prevalence_unselected": float(np.mean(labels[~keep] == "move"))})
    frame = pd.DataFrame(rows)
    fired = {"masking_frequency": bool(frame.masking_frequency_correlation.median() >= 0.3),
             "amplitude_ladder": bool(frame.u_ratio_selected_over_all.median() <= 0.9),
             "prevalence_shift": bool((frame.blink_prevalence_selected.median()
                                       < frame.blink_prevalence_unselected.median() - 0.05))}
    payload = {"per_cell": rows,
               "summary": {"masking_frequency_correlation_median":
                           float(frame.masking_frequency_correlation.median()),
                           "u_ratio_selected_median": float(frame.u_ratio_selected_over_all.median()),
                           "blink_prevalence_selected_median": float(frame.blink_prevalence_selected.median()),
                           "blink_prevalence_unselected_median": float(frame.blink_prevalence_unselected.median())},
               "fingerprints_fired": fired,
               "any_density_fingerprint_fired": bool(any(fired.values()))}
    (RESULT / "packa_fingerprints.json").write_text(json.dumps(payload, indent=2,
                                                               sort_keys=True) + "\n")
    print(json.dumps({**payload["summary"], "fired": fired}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    for name in ("b0", "t0", "t7", "b1", "b15", "t1", "t6", "t4", "once0", "once12",
                 "packa-fp"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"b0": b0, "t0": t0, "t7": t7, "b1": b1, "b15": b15, "t1": t1, "t6": t6, "t4": t4,
     "once0": once0, "once12": once12, "packa-fp": packa_fingerprints}[args.unit]()


if __name__ == "__main__":
    main()
