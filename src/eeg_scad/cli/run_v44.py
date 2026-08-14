"""V44 RGCC-EOG execution CLI: gated subject operators in the EOG-guided class.

Query EOG (the two registered bipolar regressors) is a DECLARED RUNTIME INPUT
in this deployment class — the information boundary differs from V42R/V43 by
design and is stated wherever results are reported.  The Qgen-fitted operator
stays evaluator-only (ORACLE).  The V43 EB gate is reused frozen (no retuning).
All adjudication rules are frozen in reports/v44_preregistration.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eeg_scad.cli.run_v43 import _stat, bootstrap_draws, configs, holm
from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry, bipolar_eog
from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
from eeg_scad.evaluation.paired_metrics import paired_metrics


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/rgcc_eog_v44"
REPORT = ROOT / "reports"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/rgcc_eog_v44")
S1_SEEDS = (20261201, 20261202, 20261203)
DET_SEEDS = (20261201, 20261202)
SUBTRACTION_ARMS = ("C0", "C_gated", "C_wrong", "C_wrong_g", "C_query")
G01_MARGIN = 0.010
G02_MARGIN = 0.010
G2_WRONG_GATED_MARGIN = 0.005
NATURAL_RETENTION_BAR = 0.75
WINDOW = 512
NATURAL_WINDOWS_PER_CELL = 4


def noise_seed(fold_id: int, seed: int) -> int:
    return 420000 + fold_id * 100 + seed % 100


def natural_noise_seed(fold_id: int, seed: int) -> int:
    return 610000 + fold_id * 100 + seed % 100


def _rrmse(target: np.ndarray, value: np.ndarray) -> float:
    return float(np.linalg.norm(value - target) / max(np.linalg.norm(target), 1e-12))


def _coherence(signal: np.ndarray, drive: np.ndarray) -> float:
    """Fraction of signal energy linearly explainable by the EOG drive."""
    gram = drive @ drive.T + 1e-8 * np.eye(len(drive))
    fitted = (signal @ drive.T) @ np.linalg.inv(gram) @ drive
    return float(np.sum(fitted * fitted) / max(np.sum(signal * signal), 1e-12))


def _operators(registry30, eb120, key: tuple[str, str, str], wrong: str) -> dict[str, np.ndarray]:
    session, task = key[1], key[2]
    return {"C0": registry30.population_transfer[(session, task)],
            "C_gated": eb120.operator(*key, "EB"),
            "C_wrong": eb120.operator(wrong, session, task, "RAW"),
            "C_wrong_g": eb120.operator(wrong, session, task, "EB"),
            "C_query": registry30.cells[key].query_transfer}


def _natural_windows(registry30, data, key: tuple[str, str, str]):
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    cell = registry30.cells[key]
    qstart = int(data["qnatural_start"])
    limit = min(eeg.shape[1], eog.shape[1]) - WINDOW
    starts = np.linspace(qstart, limit, NATURAL_WINDOWS_PER_CELL, dtype=int)
    for start in starts:
        y = eeg[:, start:start + WINDOW] / registry30.eeg_scale[:, None]
        drive = (eog[:, start:start + WINDOW] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
        yield int(start), y, drive


def _natural_metrics(y: np.ndarray, drive: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    out = y - estimate
    energy = np.sqrt(np.mean(drive * drive, axis=0))
    low = energy <= np.quantile(energy, .3)
    high = energy >= np.quantile(energy, .7)
    rms = lambda value: float(np.sqrt(np.mean(value * value)))
    return {
        "attenuation_db": float(20 * np.log10(max(rms(y[:, high]), 1e-12) / max(rms(out[:, high]), 1e-12))),
        "coherence_reduction": _coherence(y, drive) - _coherence(out, drive),
        "low_eog_observation_retention": 1 - float(np.linalg.norm(estimate[:, low])
                                                   / max(np.linalg.norm(y[:, low]), 1e-12)),
        "psd_distortion_proxy": float(np.abs(np.log(max(rms(out[:, low]), 1e-12) / max(rms(y[:, low]), 1e-12)))),
        "output_input_rms": rms(out) / max(rms(y), 1e-12),
    }


# ------------------------------------------------------------------- stage 0

def stage0() -> None:
    data, folds, _ = configs()
    paired_rows, natural_rows = [], []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        pinv_cache: dict[tuple[str, str, str], np.ndarray] = {}
        for seed in S1_SEEDS:
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
                key = (meta["participant"], meta["session"], meta["task"])
                if key not in pinv_cache:
                    pinv_cache[key] = np.linalg.pinv(registry30.cells[key].query_transfer)
                drive = pinv_cache[key] @ np.asarray(artifact, np.float64)
                wrong = sampler.condition_signature(meta, "WRONG")[1]
                operators = _operators(registry30, eb120, key, wrong)
                zero = int(meta["zero_artifact"])
                energy = np.sqrt(np.mean(drive * drive, axis=0))
                high = None if zero else energy >= np.quantile(energy, .7)
                base = {"fold": fold["fold"], "seed": seed, "participant": meta["participant"],
                        "session": meta["session"], "task": meta["task"], "wrong_owner": wrong,
                        "zero_artifact": zero, "gain": meta["gain"],
                        "query_eog_runtime_input": 1}
                paired_rows.append({**base, "arm": "RAW", "masked_rrmse_temporal": np.nan,
                                    **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
                for arm, operator in operators.items():
                    predicted = operator @ drive
                    out = observed - predicted
                    masked = np.nan if zero else _rrmse(clean[:, high], out[:, high])
                    paired_rows.append({**base, "arm": arm, "masked_rrmse_temporal": masked,
                                        **paired_metrics(clean, observed, artifact, predicted)})
        for participant, session, task in itertools.product(fold["test"], data["sessions"], data["tasks"]):
            key = (participant, session, task)
            if key not in registry30.cells:
                continue
            wrong = sorted(candidate for candidate in {k[0] for k in registry30.cells}
                           if candidate != participant and (candidate, session, task) in registry30.cells)[0]
            operators = _operators(registry30, eb120, key, wrong)
            for start, y, drive in _natural_windows(registry30, data, key):
                for arm in ("C0", "C_gated"):
                    natural_rows.append({"fold": fold["fold"], "participant": participant,
                                         "session": session, "task": task, "start": start, "arm": arm,
                                         **_natural_metrics(y, drive, operators[arm] @ drive)})

    frame = pd.DataFrame(paired_rows)
    per = {arm: frame[frame.arm == arm].groupby("participant").rrmse_temporal.mean()
           for arm in ("RAW",) + SUBTRACTION_ARMS}
    masked_per = {arm: frame[frame.arm == arm].groupby("participant").masked_rrmse_temporal.mean()
                  for arm in ("RAW",) + SUBTRACTION_ARMS}
    participants = per["C0"].index
    gain = (per["C0"] - per["C_gated"]).loc[participants]
    gate_safety = (per["C_wrong_g"] - per["C0"]).loc[participants]
    wrong_harm = (per["C_wrong"] - per["C0"]).loc[participants]
    gain_stat = _stat(gain)
    go = bool(gain_stat["mean"] >= G01_MARGIN and gain_stat["bootstrap_low"] > 0)
    g02_stat = _stat(gate_safety)
    g02_pass = bool(g02_stat["mean"] <= G02_MARGIN)
    natural_frame = pd.DataFrame(natural_rows)
    natural_summary = natural_frame.groupby(["arm", "participant"], as_index=False).mean(numeric_only=True) \
        .groupby("arm").mean(numeric_only=True)[["attenuation_db", "coherence_reduction",
                                                 "low_eog_observation_retention", "output_input_rms"]]
    decision = {
        "preregistration": "reports/v44_preregistration.md",
        "stage": "V44_S0_subtraction_probe",
        "information_boundary": "query EOG (VEOG/HEOG bipolar) is a declared runtime input in this class",
        "G0-1": {"contrast": "RRMSE(y-C0*e) - RRMSE(y-C_gated*e)", **gain_stat,
                 "margin": G01_MARGIN, "go": go},
        "G0-2": {"contrast": "RRMSE(y-C_wrong_g*e) - RRMSE(y-C0*e)", **g02_stat,
                 "margin": G02_MARGIN, "pass": g02_pass},
        "G0-3": {"ungated_wrong_harm": _stat(wrong_harm),
                 "oracle_row": {"mean_rrmse": float(per["C_query"].mean()),
                                "note": "degenerate on the paired panel: the Qgen operator "
                                        "reproduces the generative artifact exactly"}},
        "G0-4_natural_descriptive": {arm: {metric: float(natural_summary.loc[arm, metric])
                                           for metric in natural_summary.columns}
                                     for arm in natural_summary.index},
        "condition_means_full_window": {arm: float(series.mean()) for arm, series in per.items()},
        "condition_means_masked_top30": {arm: float(series.mean()) for arm, series in masked_per.items()},
        "participants": int(len(participants)), "sealed_reads": 0,
        "decision": "GO_to_S1" if go else "NO_GO_stop",
    }
    target = RESULT / "stage0"
    target.mkdir(parents=True, exist_ok=True)
    (target / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    frame.to_csv(target / "paired_arm_rows.csv", index=False)
    natural_frame.to_csv(target / "natural_arm_rows.csv", index=False)
    arm_table = pd.DataFrame([{"arm": arm, "full_window_rrmse": float(per[arm].mean()),
                               "masked_top30_rrmse": float(masked_per[arm].mean())}
                              for arm in ("RAW",) + SUBTRACTION_ARMS])
    (REPORT / "v44_stage0.md").write_text(
        "# V44 Stage 0 — subtraction probe (CPU)\n\n"
        "Query EOG is a declared runtime input in this deployment class. Operators use the "
        "V43-frozen EB gate unchanged. Paired panel, full-window temporal RRMSE vs clean, "
        "participant-first n=15; masked top-30% rows are a V19-comparability secondary.\n\n"
        f"Decision: **{decision['decision']}** — G0-1 mean {gain_stat['mean']:+.6f} "
        f"(margin {G01_MARGIN:+.3f}), CI [{gain_stat['bootstrap_low']:+.6f}, "
        f"{gain_stat['bootstrap_high']:+.6f}]; G0-2 pass **{g02_pass}** "
        f"(mean {g02_stat['mean']:+.6f}).\n\n## Arm means\n\n"
        + arm_table.round(6).to_markdown(index=False)
        + "\n\n## Endpoints\n\n```json\n"
        + json.dumps({key: decision[key] for key in ("G0-1", "G0-2", "G0-3")},
                     indent=2, sort_keys=True) + "\n```\n\n"
        "## G0-4 natural windows (descriptive)\n\n"
        + natural_summary.round(6).to_markdown() + "\n")
    print(json.dumps({"decision": decision["decision"], "G0-1": decision["G0-1"],
                      "G0-2": {"pass": g02_pass, "mean": g02_stat["mean"]}}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("stage0")
    args = parser.parse_args()
    if args.stage == "stage0":
        stage0()


if __name__ == "__main__":
    main()
