"""V43 RGCC execution CLI: EB state build, S1 floor probe, S1.5 ceiling probe, aggregation.

All adjudication rules are frozen in reports/v43_preregistration.md (committed
before the first submission).  V42R artifacts (checkpoints, results tree) are
read-only inputs.  The S1.5 route trains on query-fitted (generative-truth)
conditioning and is non-deployable by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
from eeg_scad.evaluation.paired_metrics import paired_metrics


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/rgcc_v43"
REPORT = ROOT / "reports"
V42R_RESULT = ROOT / "results/calib_saddpm_cond_v42r"          # read-only
V42R_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_saddpm_cond_v42r")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/rgcc_v43")
CHECKPOINT_RUN = "job_941770"
SEEDS = (20261201, 20261202)
STAGE15_CELLS = ((0, 20261201), (2, 20261201))
EB_ARM_VARIANTS = {"MATCH_EB120": ("eb120", "recipient"), "MATCH_RAW120": ("raw120", "recipient"),
                   "MATCH_EB10": ("eb10", "recipient"), "WRONG_EB120": ("eb120", "wrong"),
                   "MATCH_EB120_PERROW": ("perrow120", "recipient")}
MARGINS = {"F1": 0.010, "F2": 0.010, "F3": 0.005}
S15_MEAN_MARGIN, S15_CI_LOW = 0.020, 0.005


def configs():
    data = yaml.safe_load((ROOT / "configs/setcalibdiff_v25/data.yaml").read_text())
    data.update(yaml.safe_load((ROOT / "configs/calib_saddpm_cond_v42r/data.yaml").read_text()))
    data["v19_derived_root"] = data["source_root"]
    folds = yaml.safe_load((ROOT / "configs/setcalibdiff_v25/folds.yaml").read_text())["folds"]
    training = yaml.safe_load((ROOT / "configs/calib_saddpm_cond_v42r/training.yaml").read_text())
    return data, folds, training


def noise_seed(fold_id: int, seed: int) -> int:
    return 420000 + fold_id * 100 + seed % 100


def bootstrap_draws(values, seed: int = 420, draws: int = 5000) -> np.ndarray:
    value = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    return np.asarray([rng.choice(value, len(value), replace=True).mean() for _ in range(draws)])


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - rank) * p))
        adjusted[key] = running
    return adjusted


# ---------------------------------------------------------------- state build

def build_state() -> None:
    data, folds, _ = configs()
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        eb10 = EBTransferRegistry(data, fold, registry30, 10)
        keys = sorted(registry30.cells)
        arrays: dict[str, list] = {name: [] for name in ("eb120", "raw120", "perrow120", "eb10")}
        pop_identity_checked = 0
        for key in keys:
            pop = registry30.signature(*key, "POP")
            arrays["eb120"].append(eb120.signature(*key, "EB"))
            arrays["raw120"].append(eb120.signature(*key, "RAW"))
            arrays["perrow120"].append(eb120.signature(*key, "PERROW"))
            ten = eb10.signature(*key, "EB")
            # Preregistered bypass identity: the 10-s hard gate must reproduce POP exactly.
            if not eb10.cells[key].hard_gate or not np.array_equal(ten, pop):
                raise AssertionError(f"10-s hard gate did not reproduce POP for {key}")
            if eb120.cells[key].lam == 0.0 and not np.array_equal(arrays["eb120"][-1], pop):
                raise AssertionError(f"gated 120-s lambda=0 state differs from POP for {key}")
            arrays["eb10"].append(ten)
            pop_identity_checked += 1
        target = RESULT / "state" / f"fold_{fold['fold']}"
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "eb_signatures.npz",
            keys=np.asarray(["|".join(key) for key in keys]),
            **{name: np.stack(value).astype(np.float32) for name, value in arrays.items()},
            lambda120=np.asarray([eb120.cells[key].lam for key in keys]),
            hard_gate120=np.asarray([int(eb120.cells[key].hard_gate) for key in keys]),
            lambda10=np.asarray([eb10.cells[key].lam for key in keys]),
        )
        manifest = eb120.manifest_rows() + eb10.manifest_rows()
        pd.DataFrame(manifest).to_csv(target / "eb_state_manifest.csv", index=False)
        frame = pd.DataFrame(eb120.manifest_rows())
        summary = {"fold": fold["fold"], "cells": len(keys), "pop_identity_checked": pop_identity_checked,
                   "within_threshold_120": eb120.within_threshold, "within_threshold_10": eb10.within_threshold,
                   "hard_gate_count_120": int(frame.hard_gate.sum()),
                   "lambda120_min": float(frame["lambda"].min()), "lambda120_mean": float(frame["lambda"].mean()),
                   "lambda120_max": float(frame["lambda"].max())}
        (target / "state_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary))


# ---------------------------------------------------------------------- S1

def stage1(fold_id: int, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule
    from eeg_scad.training.train_v42r import _conditions, sample_bank

    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    with np.load(RESULT / "state" / f"fold_{fold_id}" / "eb_signatures.npz", allow_pickle=False) as archive:
        state = {name: np.asarray(archive[name]) for name in archive.files}
    index = {str(key): position for position, key in enumerate(state["keys"])}
    checkpoint = V42R_DERIVED / CHECKPOINT_RUN / f"fold_{fold_id}_seed_{seed}" / "best.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CalibSADDPMCond().to(device)
    model.load_state_dict(payload["ema"])
    schedule = LinearX0Schedule().to(device)
    frozen = json.loads((checkpoint.parent / "result.json").read_text())

    rows = []
    for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"], "condition": "RAW",
                     "context_owner": "NONE", "lambda": np.nan, "hard_gate": 0,
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})

    wrong_owners = [sampler.condition_signature(meta, "WRONG")[1] for meta in bank["meta"]]
    recipient_keys = ["|".join((meta["participant"], meta["session"], meta["task"])) for meta in bank["meta"]]
    wrong_keys = ["|".join((owner, meta["session"], meta["task"]))
                  for owner, meta in zip(wrong_owners, bank["meta"])]
    outputs: dict[str, np.ndarray] = {}
    arm_context: dict[str, list[tuple[str, float, int]]] = {}
    for condition in ("POP", "MATCH"):
        signature, owners = _conditions(sampler, bank["meta"], condition)
        arm_context[condition] = [(owner, np.nan, 0) for owner in owners]
        outputs[condition] = sample_bank(model, schedule, bank["y"], signature, device,
                                         noise_seed(fold_id, seed))
    for condition, (variant, target) in EB_ARM_VARIANTS.items():
        keys = wrong_keys if target == "wrong" else recipient_keys
        owners = wrong_owners if target == "wrong" else [meta["participant"] for meta in bank["meta"]]
        positions = [index[key] for key in keys]
        signature = np.stack([state[variant][position] for position in positions])
        lam_field = "lambda10" if variant == "eb10" else "lambda120"
        lams = [1.0 if condition == "MATCH_RAW120" else float(state[lam_field][position])
                for position in positions]
        gates = [int(state["hard_gate120"][position]) if variant != "eb10" else 1 for position in positions]
        arm_context[condition] = list(zip(owners, lams, gates))
        outputs[condition] = sample_bank(model, schedule, bank["y"], signature, device,
                                         noise_seed(fold_id, seed))
    for condition, output in outputs.items():
        for clean, observed, artifact, prediction, meta, (owner, lam, gate) in zip(
                bank["x"], bank["y"], bank["artifact"], output, bank["meta"], arm_context[condition]):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V43 stage1 output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": condition, "context_owner": owner, "lambda": lam, "hard_gate": gate,
                         "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         **paired_metrics(clean, observed, artifact, observed - prediction)})

    anchor = {}
    for condition in ("POP", "MATCH"):
        frozen_rows = [row["rrmse_temporal"] for row in frozen["paired_metrics"] if row["condition"] == condition]
        ours = [row["rrmse_temporal"] for row in rows if row["condition"] == condition]
        if len(frozen_rows) != len(ours):
            raise AssertionError(f"anchor episode count mismatch for {condition}")
        anchor[condition] = {"max_abs_diff": float(np.max(np.abs(np.asarray(ours) - np.asarray(frozen_rows)))),
                             "mean_frozen": float(np.mean(frozen_rows)), "mean_replay": float(np.mean(ours))}
        if anchor[condition]["max_abs_diff"] > 2e-3:
            raise AssertionError(f"{condition} anchor diverged from frozen replay: {anchor[condition]}")
    eb10_delta = float(np.max(np.abs(outputs["MATCH_EB10"] - outputs["POP"])))
    if eb10_delta > 1e-6:
        raise AssertionError(f"MATCH_EB10 bypass is not the POP route (max delta {eb10_delta})")

    target = RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    result = {"fold": fold_id, "seed": seed, "checkpoint": str(checkpoint),
              "checkpoint_best_step": payload["step"], "noise_seed": noise_seed(fold_id, seed),
              "anchor_qc": anchor, "eb10_equals_pop_output": bool(eb10_delta == 0.0),
              "eb10_pop_output_max_abs_diff": eb10_delta, "sealed_reads": 0,
              "query_eog_inference_reads": 0, "rows": rows}
    (target / "stage1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "anchor_qc": anchor, "eb10_delta": eb10_delta}))


# --------------------------------------------------------------------- S1.5

def train_oracle(model, schedule, train_sampler, validation_bank, validation_sampler,
                 device, seed: int, updates: int, batch_size: int, validation_interval: int,
                 runtime: Path) -> list[dict[str, float]]:
    """Registered clone of train_v42r.train_joint with ONE change: the training
    conditioning state is the query-fitted (Qgen) ORACLE signature of each
    episode's owner cell.  The 20% context dropout to POP, the optimizer, EMA,
    validation-POP checkpoint rule, and the RNG consumption order are identical,
    so training episodes match the V42R cells draw-for-draw."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    from eeg_scad.training.train_v42r import EMA, participant_pop_score

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model, .999)
    generator = torch.Generator(device=device).manual_seed(seed + 7001)
    curve, best = [], float("inf")
    best_path, last_path = runtime / "best.pt", runtime / "last.pt"
    for step in range(1, updates + 1):
        bank = train_sampler.sample(batch_size)
        signature = np.stack([train_sampler.condition_signature(meta, "ORACLE")[0]
                              for meta in bank["meta"]])
        dropped = train_sampler.rng.random(batch_size) < .20
        for position in np.flatnonzero(dropped):
            signature[position], _ = train_sampler.condition_signature(bank["meta"][int(position)], "POP")
        clean = torch.from_numpy(np.asarray(bank["x"], np.float32)).to(device)
        observed = torch.from_numpy(np.asarray(bank["y"], np.float32)).to(device)
        condition = torch.from_numpy(np.asarray(signature, np.float32)).to(device)
        noisy, timestep, _ = schedule.forward_sample(clean, generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, observed, timestep, condition)
        loss = F.smooth_l1_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite V43 oracle loss at update {step}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        transfer_gradient = torch.stack([parameter.grad.detach().norm()
                                         for parameter in model.transfer_parameters()
                                         if parameter.grad is not None]).sum()
        if not torch.isfinite(gradient) or not torch.isfinite(transfer_gradient):
            raise FloatingPointError(f"nonfinite V43 oracle gradient at update {step}")
        optimizer.step()
        ema.update(model)
        if step % validation_interval == 0 or step == updates:
            score = participant_pop_score(ema.model, schedule, validation_bank, validation_sampler,
                                          device, seed + 17001)
            curve.append({"step": step, "train_smooth_l1": float(loss.detach()),
                          "validation_pop_rrmse": score, "gradient_norm": float(gradient),
                          "transfer_gradient_norm": float(transfer_gradient),
                          "context_dropout_fraction": float(dropped.mean())})
            payload = {"model": model.state_dict(), "ema": ema.state_dict(), "step": step,
                       "seed": seed, "curve": curve, "best_validation_pop_rrmse": min(best, score),
                       "conditioning": "ORACLE_query_fitted_nondeployable"}
            runtime.mkdir(parents=True, exist_ok=True)
            torch.save(payload, last_path)
            if score < best:
                best = score
                payload["best_validation_pop_rrmse"] = score
                torch.save(payload, best_path)
    if not best_path.is_file():
        raise RuntimeError("V43 oracle training created no checkpoint")
    return curve


def stage15(fold_id: int, seed: int, updates: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule
    from eeg_scad.training.train_v42r import _conditions, sample_bank

    result_path = RESULT / "stage15" / f"fold_{fold_id}_seed_{seed}" / "stage15_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "result already complete"}))
        return
    data, folds, training = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = DERIVED / "stage15" / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    test_sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    test_bank = test_sampler.sample_balanced(8)
    model = CalibSADDPMCond().to(device)
    schedule = LinearX0Schedule().to(device)
    curve = train_oracle(model, schedule, train_sampler, validation_bank, validation_sampler,
                         device, seed, updates, training["effective_batch"],
                         training["validation_interval"], runtime)
    payload = torch.load(runtime / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    rows = []
    for clean, observed, artifact, meta in zip(test_bank["x"], test_bank["y"], test_bank["artifact"], test_bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"], "condition": "RAW",
                     "context_owner": "NONE", "oracle_non_deployable": 0,
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    for condition in ("POP", "ORACLE"):
        signature, owners = _conditions(test_sampler, test_bank["meta"], condition)
        output = sample_bank(model, schedule, test_bank["y"], signature, device,
                             noise_seed(fold_id, seed))
        for clean, observed, artifact, prediction, meta, owner in zip(
                test_bank["x"], test_bank["y"], test_bank["artifact"], output, test_bank["meta"], owners):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V43 stage15 output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": condition, "context_owner": owner,
                         "oracle_non_deployable": int(condition == "ORACLE"),
                         "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         **paired_metrics(clean, observed, artifact, observed - prediction)})
    target = RESULT / "stage15" / f"fold_{fold_id}_seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    result = {"fold": fold_id, "seed": seed, "updates": updates,
              "training_conditioning": "ORACLE_query_fitted_generative_truth_nondeployable",
              "checkpoint": str(runtime / "best.pt"), "checkpoint_best_step": payload["step"],
              "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"],
              "noise_seed": noise_seed(fold_id, seed), "training_curve": curve,
              "sealed_reads": 0, "rows": rows}
    (target / "stage15_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "best_step": payload["step"],
                      "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"]}))


# ---------------------------------------------------------------- aggregate

def _participant_means(frame: pd.DataFrame, condition: str) -> pd.Series:
    block = frame[frame.condition == condition]
    return block.groupby("participant").rrmse_temporal.mean()


def _stat(values: pd.Series, seed: int = 420) -> dict[str, Any]:
    draws = bootstrap_draws(values.to_numpy(), seed=seed)
    return {"mean": float(values.mean()), "median": float(values.median()),
            "positive_count": int((values > 0).sum()), "participants": int(len(values)),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def aggregate_stage1() -> dict[str, Any]:
    rows = []
    for fold_id in range(5):
        for seed in SEEDS:
            path = RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}" / "stage1_result.json"
            payload = json.loads(path.read_text())
            rows += payload["rows"]
    frame = pd.DataFrame(rows)
    frozen = pd.read_csv(V42R_RESULT / "paired_metrics.csv")
    per = {condition: _participant_means(frame, condition)
           for condition in ("RAW", "POP", "MATCH", "MATCH_EB120", "MATCH_RAW120",
                             "MATCH_EB10", "WRONG_EB120", "MATCH_EB120_PERROW")}
    frozen_per = {condition: _participant_means(frozen, condition) for condition in ("POP", "WRONG")}
    participants = per["POP"].index

    d_wrong = (per["WRONG_EB120"] - per["POP"]).loc[participants]
    frozen_harm = (frozen_per["WRONG"] - frozen_per["POP"]).loc[participants]
    reduction = frozen_harm - d_wrong
    d_eb10 = (per["MATCH_EB10"] - per["POP"]).loc[participants]
    d_eb120 = (per["MATCH_EB120"] - per["POP"]).loc[participants]
    gain = -d_eb120

    draws = {"F1": bootstrap_draws(d_wrong.to_numpy()), "F2": bootstrap_draws(d_eb10.to_numpy()),
             "F3": bootstrap_draws(d_eb120.to_numpy())}
    p_raw = {name: float(np.mean(draws[name] >= MARGINS[name])) for name in MARGINS}
    p_adjusted = holm(p_raw)
    reduction_stat = _stat(reduction)

    f1 = {"contrast": "WRONG_EB120_minus_POP", **_stat(d_wrong), "margin": MARGINS["F1"],
          "reduction_vs_frozen_wrong_harm": reduction_stat,
          "frozen_wrong_harm_mean": float(frozen_harm.mean()),
          "pass": bool(d_wrong.mean() <= MARGINS["F1"] and reduction_stat["bootstrap_low"] > 0)}
    f2 = {"contrast": "MATCH_EB10_minus_POP", **_stat(d_eb10), "margin": MARGINS["F2"],
          "pass": bool(d_eb10.mean() <= MARGINS["F2"])}
    f3 = {"contrast": "MATCH_EB120_minus_POP", **_stat(d_eb120), "margin": MARGINS["F3"],
          "pass": bool(d_eb120.mean() <= MARGINS["F3"])}
    anchors = {"raw_mean": float(per["RAW"].mean()), "pop_mean": float(per["POP"].mean()),
               "match_minus_pop_mean": float((per["MATCH"] - per["POP"]).mean()),
               "frozen_reference": {"raw": 0.714933, "pop": 0.632308, "match_minus_pop": 0.0000268},
               "per_cell_anchor_qc_max_abs_diff": float(max(
                   json.loads((RESULT / "stage1" / f"fold_{f}_seed_{s}" / "stage1_result.json").read_text())
                   ["anchor_qc"][condition]["max_abs_diff"]
                   for f in range(5) for s in SEEDS for condition in ("POP", "MATCH")))}
    decision = {
        "preregistration": "reports/v43_preregistration.md",
        "stage": "S1_frozen_checkpoint_floor_probe",
        "F1": f1, "F2": f2, "F3": f3,
        "holm": {"p_raw": p_raw, "p_adjusted": p_adjusted, "alpha": 0.05,
                 "note": "one-sided bootstrap probability that the endpoint mean exceeds its margin"},
        "gain_reading_nonadjudicating": {
            "contrast": "POP_minus_MATCH_EB120", **_stat(gain),
            "note": "non-adjudicating in S1: checkpoint was trained on 30-s states"},
        "secondaries": {"MATCH_RAW120_minus_POP": _stat((per["MATCH_RAW120"] - per["POP"]).loc[participants]),
                        "MATCH_EB120_PERROW_minus_POP": _stat((per["MATCH_EB120_PERROW"] - per["POP"]).loc[participants])},
        "anchors": anchors, "sealed_reads": 0,
    }
    target = RESULT / "stage1"
    (target / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    condition_table = pd.DataFrame([{"condition": name, "participant_mean_rrmse_temporal": float(series.mean())}
                                    for name, series in per.items()])
    contrast_table = pd.DataFrame([
        {"endpoint": "F1", **{k: v for k, v in f1.items() if k != "reduction_vs_frozen_wrong_harm"}},
        {"endpoint": "F1_reduction", "contrast": "frozen_harm_minus_new_harm", **reduction_stat},
        {"endpoint": "F2", **f2}, {"endpoint": "F3", **f3}])
    (REPORT / "v43_stage1.md").write_text(
        "# V43 Stage 1 — frozen-checkpoint floor probe\n\n"
        "Preregistration: `reports/v43_preregistration.md` (frozen before submission). "
        "Frozen V42R checkpoints (job_941770), identical test banks, common noise across arms. "
        "The MATCH_EB120-POP gain reading is non-adjudicating in S1.\n\n"
        f"Decision: F1 **{f1['pass']}**, F2 **{f2['pass']}**, F3 **{f3['pass']}**.\n\n"
        "## Participant-first condition means (temporal RRMSE, n=15)\n\n"
        + condition_table.round(6).to_markdown(index=False) + "\n\n## Preregistered endpoints\n\n"
        + contrast_table.round(6).to_markdown(index=False) + "\n\n## Holm over {F1, F2, F3}\n\n"
        f"raw p: {json.dumps(p_raw)}; Holm-adjusted: {json.dumps(p_adjusted)}\n\n"
        "## Anchors\n\n```json\n" + json.dumps(anchors, indent=2, sort_keys=True) + "\n```\n\n"
        "## Non-adjudicating gain reading and secondaries\n\n```json\n"
        + json.dumps({"gain_reading": decision["gain_reading_nonadjudicating"],
                      "secondaries": decision["secondaries"]}, indent=2, sort_keys=True) + "\n```\n")
    return decision


def aggregate_stage15() -> dict[str, Any]:
    rows = []
    for fold_id, seed in STAGE15_CELLS:
        path = RESULT / "stage15" / f"fold_{fold_id}_seed_{seed}" / "stage15_result.json"
        rows += json.loads(path.read_text())["rows"]
    frame = pd.DataFrame(rows)
    pop = _participant_means(frame, "POP")
    oracle = _participant_means(frame, "ORACLE")
    raw = _participant_means(frame, "RAW")
    delta = (pop - oracle).loc[pop.index]
    stat = _stat(delta)
    go = bool(stat["mean"] >= S15_MEAN_MARGIN and stat["bootstrap_low"] > S15_CI_LOW)
    decision = {
        "preregistration": "reports/v43_preregistration.md",
        "stage": "S1.5_oracle_trained_ceiling_probe_nondeployable",
        "go_rule": {"mean_margin": S15_MEAN_MARGIN, "ci_low_threshold": S15_CI_LOW},
        "contrast": "POP_minus_ORACLE", **stat, "go": go,
        "per_participant": {participant: float(value) for participant, value in delta.items()},
        "cells": [{"fold": fold_id, "seed": seed} for fold_id, seed in STAGE15_CELLS],
        "condition_means": {"RAW": float(raw.mean()), "POP": float(pop.mean()), "ORACLE": float(oracle.mean())},
        "interpretation": ("GO: trainable conditioning headroom exists on this panel"
                           if go else
                           "NO-GO: waveform-level gain claim dead on this panel; V43 proceeds floor-only"),
        "sealed_reads": 0,
    }
    target = RESULT / "stage15"
    (target / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (REPORT / "v43_stage15.md").write_text(
        "# V43 Stage 1.5 — oracle-trained ceiling probe (non-deployable)\n\n"
        "Training conditioning is the query-fitted (Qgen, generative-truth) transfer signature; "
        "this route is an oracle diagnostic and is non-deployable by construction. "
        "Held-out test participants of folds 0 and 2, common noise, same episode banks as S1.\n\n"
        f"Decision: **{'GO' if go else 'NO-GO'}** "
        f"(mean POP-ORACLE = {stat['mean']:+.6f}, CI [{stat['bootstrap_low']:+.6f}, "
        f"{stat['bootstrap_high']:+.6f}]; GO requires mean >= +0.020 and CI-low > +0.005).\n\n"
        "```json\n" + json.dumps(decision, indent=2, sort_keys=True) + "\n```\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("build-state")
    one = sub.add_parser("stage1")
    one.add_argument("--fold", type=int, required=True)
    one.add_argument("--seed", type=int, required=True)
    fifteen = sub.add_parser("stage15")
    fifteen.add_argument("--fold", type=int, required=True)
    fifteen.add_argument("--seed", type=int, required=True)
    fifteen.add_argument("--updates", type=int, required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--which", choices=("1", "15", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "build-state":
        build_state()
    elif args.stage == "stage1":
        stage1(args.fold, args.seed)
    elif args.stage == "stage15":
        stage15(args.fold, args.seed, args.updates)
    else:
        if args.which in ("1", "all"):
            print(json.dumps(aggregate_stage1(), indent=2, sort_keys=True))
        if args.which in ("15", "all"):
            print(json.dumps(aggregate_stage15(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
