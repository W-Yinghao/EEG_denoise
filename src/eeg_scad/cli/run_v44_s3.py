"""V44-S3: drift-decomposed ownership rescoring (CPU-only micro-stage).

The guard is a ROUTER between already-sampled frozen S1 outputs (MATCH_gated /
WRONG_gated vs NO_A0 per episode, common noise), so every endpoint recomputes
on CPU.  Rules frozen in the V44-S3 addendum: drift-calibrated null (R1) from
training owners' own support→query operator pairs; spatial principal-angle vs
gain norm-ratio score families (R2), with the spatial-separates prediction
registered up front.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from eeg_scad.cli.run_v43 import bootstrap_draws, configs, holm
from eeg_scad.cli.run_v44 import RESULT, REPORT, S1_SEEDS, _gated_assets, _participant_means, _stat
from eeg_scad.cli.run_v44_s2 import TV_PRIMARY, _probe_operator
from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry


OG1_MARGIN, OG1_DETECTION = 0.005, 0.90
OG2_FALSE_ALARM, OG2_COST = 0.10, 0.005


def _spatial_angles(c_a: np.ndarray, c_b: np.ndarray) -> tuple[float, float]:
    qa, _ = np.linalg.qr(np.asarray(c_a, np.float64))
    qb, _ = np.linalg.qr(np.asarray(c_b, np.float64))
    cosines = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), -1.0, 1.0)
    angles = np.arccos(cosines)
    return float(angles.max()), float(angles.mean())


def _gain_score(c_a: np.ndarray, c_b: np.ndarray) -> float:
    global_ratio = abs(np.log(max(np.linalg.norm(c_a), 1e-12)
                              / max(np.linalg.norm(c_b), 1e-12)))
    rows_a = np.linalg.norm(c_a, axis=1).clip(1e-12)
    rows_b = np.linalg.norm(c_b, axis=1).clip(1e-12)
    return float(global_ratio + np.mean(np.abs(np.log(rows_a / rows_b))))


def rescore() -> dict:
    data, folds, _ = configs()
    score_rows = []
    null_rows = {"spatial": [], "gain": []}
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        qgen_start = int(data["qgen_start"])
        # R1 null: TRAIN owners' own support(gated)→query-probe discrepancy
        for key in sorted(assets):
            if key[0] not in fold["train"]:
                continue
            probe = _probe_operator(registry30, key, TV_PRIMARY, qgen_start)
            null_rows["spatial"].append(_spatial_angles(assets[key]["C_gated"], probe)[0])
            null_rows["gain"].append(_gain_score(assets[key]["C_gated"], probe))
        # scores for the S1 panel presentations (recompute probes; deterministic)
        for seed in S1_SEEDS:
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            seen = set()
            for meta in bank["meta"]:
                key = (meta["participant"], meta["session"], meta["task"])
                wrong = sampler.condition_signature(meta, "WRONG")[1]
                wkey = (wrong, meta["session"], meta["task"])
                pair = (fold["fold"], seed, "|".join(key))
                if pair in seen:
                    continue
                seen.add(pair)
                probe = _probe_operator(registry30, key, TV_PRIMARY, qgen_start)
                sp_m, sp_m_mean = _spatial_angles(assets[key]["C_gated"], probe)
                sp_w, sp_w_mean = _spatial_angles(assets[wkey]["C_gated"], probe)
                score_rows.append({
                    "fold": fold["fold"], "seed": seed, "participant": key[0],
                    "cell": "|".join(key), "wrong_owner": wrong,
                    "spatial_match": sp_m, "spatial_match_mean": sp_m_mean,
                    "spatial_wrong": sp_w, "spatial_wrong_mean": sp_w_mean,
                    "gain_match": _gain_score(assets[key]["C_gated"], probe),
                    "gain_wrong": _gain_score(assets[wkey]["C_gated"], probe)})
    thresholds = {family: float(np.percentile(values, 95))
                  for family, values in null_rows.items()}
    return {"score_rows": score_rows, "thresholds": thresholds,
            "null_sizes": {k: len(v) for k, v in null_rows.items()}}


def _s1_rows():
    rows, natural = [], []
    for fold_id in range(5):
        for seed in S1_SEEDS:
            payload = json.loads((RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                                  / "stage1_result.json").read_text())
            rows += payload["rows"]
            natural += payload["natural_rows"]
    return pd.DataFrame(rows), pd.DataFrame(natural)


def _s2_scores():
    rows = []
    for fold_id in range(5):
        for seed in S1_SEEDS:
            payload = json.loads((RESULT / "stage2" / f"fold_{fold_id}_seed_{seed}"
                                  / "stage2_result.json").read_text())
            rows += [dict(row, fold=fold_id, seed=seed) for row in payload["guard_rows"]
                     if row["t_v"] == TV_PRIMARY]
    return pd.DataFrame(rows)


def _route(frame: pd.DataFrame, flags: dict, arm: str, fallback: str, flag_key: str):
    """Per-episode routing between frozen arm rows by the cell-level flag."""
    arm_rows = frame[frame.condition == arm].reset_index(drop=True)
    fallback_rows = frame[frame.condition == fallback].reset_index(drop=True)
    routed = []
    for (_, a_row), (_, f_row) in zip(arm_rows.iterrows(), fallback_rows.iterrows()):
        cell = (a_row["fold"], a_row["seed"], a_row["participant"])
        fired = flags.get(cell + (flag_key,), False)
        routed.append(f_row if fired else a_row)
    return pd.DataFrame(routed)


def aggregate() -> dict:
    scored = rescore()
    scores = pd.DataFrame(scored["score_rows"])
    s2_scores = _s2_scores()
    frame, natural = _s1_rows()

    families = {
        "spatial": ("spatial_match", "spatial_wrong", scored["thresholds"]["spatial"]),
        "gain": ("gain_match", "gain_wrong", scored["thresholds"]["gain"]),
    }
    results = {}
    roc_tables = {}
    for family, (match_col, wrong_col, threshold) in families.items():
        cell_flags = {}
        for _, row in scores.iterrows():
            base = (row["fold"], row["seed"], row["participant"])
            cell_flags[base + ("match",)] = cell_flags.get(base + ("match",), False) \
                or bool(row[match_col] > threshold)
            cell_flags[base + ("wrong",)] = cell_flags.get(base + ("wrong",), False) \
                or bool(row[wrong_col] > threshold)
        detection = float(np.mean(scores[wrong_col] > threshold))
        false_alarm = float(np.mean(scores[match_col] > threshold))
        match_routed = _route(frame, cell_flags, "MATCH_gated", "NO_A0", "match")
        wrong_routed = _route(frame, cell_flags, "WRONG_gated", "NO_A0", "wrong")
        per_match = match_routed.groupby("participant").rrmse_temporal.mean()
        per_wrong = wrong_routed.groupby("participant").rrmse_temporal.mean()
        noa0 = _participant_means(frame, "NO_A0")
        match_ref = _participant_means(frame, "MATCH_gated")
        participants = noa0.index
        og1_delta = (per_wrong - noa0).loc[participants]
        og2_delta = (per_match - match_ref).loc[participants]
        og1_stat, og2_stat = _stat(og1_delta), _stat(og2_delta)
        results[family] = {
            "threshold": threshold,
            "detection_rate": detection, "false_alarm_rate": false_alarm,
            "OG-1p": {**og1_stat, "margin": OG1_MARGIN,
                      "pass": bool(og1_delta.mean() <= OG1_MARGIN
                                   and detection >= OG1_DETECTION)},
            "OG-2p": {**og2_stat, "margin": OG2_COST,
                      "pass": bool(false_alarm <= OG2_FALSE_ALARM
                                   and og2_delta.mean() <= OG2_COST)},
            "p_raw": {"OG-1p": float(np.mean(bootstrap_draws(og1_delta.to_numpy()) >= OG1_MARGIN)),
                      "OG-2p": float(np.mean(bootstrap_draws(og2_delta.to_numpy()) >= OG2_COST))},
        }
        grid = np.quantile(np.concatenate([scores[match_col], scores[wrong_col]]),
                           np.linspace(0.0, 1.0, 41))
        roc = []
        for value in grid:
            roc.append({"threshold": float(value),
                        "detection": float(np.mean(scores[wrong_col] > value)),
                        "false_alarm": float(np.mean(scores[match_col] > value))})
        roc_tables[family] = roc

    s2_roc = []
    grid = np.quantile(np.concatenate([s2_scores.score_match, s2_scores.score_wrong]),
                       np.linspace(0.0, 1.0, 41))
    for value in grid:
        s2_roc.append({"threshold": float(value),
                       "detection": float(np.mean(s2_scores.score_wrong > value)),
                       "false_alarm": float(np.mean(s2_scores.score_match > value))})
    roc_tables["s2_mahalanobis"] = s2_roc

    def auc(table):
        points = sorted([(row["false_alarm"], row["detection"]) for row in table])
        xs = [0.0] + [p[0] for p in points] + [1.0]
        ys = [points[0][1] if points else 0.0] + [p[1] for p in points] + [1.0]
        return float(np.trapz(ys, xs))

    aucs = {family: auc(table) for family, table in roc_tables.items()}
    spatial_pass = results["spatial"]["OG-1p"]["pass"] and results["spatial"]["OG-2p"]["pass"]
    p_raw = results["spatial"]["p_raw"]
    decision = {
        "preregistration": "reports/v44_preregistration.md (V44-S3 addendum)",
        "stage": "V44_S3_drift_decomposed_ownership",
        "thresholds_R1": scored["thresholds"], "null_sizes": scored["null_sizes"],
        "families": results, "roc_auc": aucs,
        "holm_spatial": {"p_raw": p_raw, "p_adjusted": holm(p_raw), "alpha": 0.05},
        "prediction_check": {
            "registered": "SPATIAL separates (AUC materially above S2); GAIN does not",
            "spatial_auc": aucs["spatial"], "gain_auc": aucs["gain"],
            "s2_auc": aucs["s2_mahalanobis"]},
        "ownership_closed": bool(not spatial_pass),
        "closure_statement": (None if spatial_pass else
                              "ownership verification from operator features is CLOSED for "
                              "the likelihood leg; complete two-family negative with the "
                              "drift diagnosis"),
        "roc_tables": roc_tables, "sealed_reads": 0,
    }
    target = RESULT / "stage3"
    target.mkdir(parents=True, exist_ok=True)
    (target / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (REPORT / "v44_stage3.md").write_text(
        "# V44 Stage 3 — drift-decomposed ownership rescoring\n\n"
        "CPU-only re-routing of frozen S1 outputs; addendum frozen before rescoring. "
        "No further ownership attempts after this stage regardless of outcome.\n\n"
        f"SPATIAL: OG-1' **{results['spatial']['OG-1p']['pass']}**, "
        f"OG-2' **{results['spatial']['OG-2p']['pass']}** "
        f"(detection {results['spatial']['detection_rate']:.3f}, "
        f"false-alarm {results['spatial']['false_alarm_rate']:.3f}); "
        f"GAIN (negative control): detection {results['gain']['detection_rate']:.3f}, "
        f"false-alarm {results['gain']['false_alarm_rate']:.3f}.\n\n"
        f"ROC AUC: spatial {aucs['spatial']:.4f}, gain {aucs['gain']:.4f}, "
        f"S2 Mahalanobis {aucs['s2_mahalanobis']:.4f}.\n\n"
        "```json\n" + json.dumps({k: v for k, v in decision.items() if k != "roc_tables"},
                                 indent=2, sort_keys=True) + "\n```\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["aggregate"])
    parser.parse_args()
    payload = aggregate()
    print(json.dumps({"spatial": {k: payload["families"]["spatial"][k]
                                  for k in ("detection_rate", "false_alarm_rate")},
                      "gain": {k: payload["families"]["gain"][k]
                               for k in ("detection_rate", "false_alarm_rate")},
                      "auc": payload["roc_auc"],
                      "ownership_closed": payload["ownership_closed"]}, indent=2))


if __name__ == "__main__":
    main()
