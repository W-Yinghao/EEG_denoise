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
DURATIONS = (10, 30, 60, 120)
S2_SEEDS = (20261201, 20261202, 20261203)
DET_SEEDS = (20261201, 20261202)
S2B_CELLS = ((1, 20261201), (3, 20261201), (4, 20261201))
D_MARGINS = {"D-F1": 0.005, "D-F3": 0.002, "D-F3_upper": 0.005, "D-F4": 0.002}


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


# ------------------------------------------------------------------ S2 state

def build_state_s2() -> None:
    from eeg_scad.data.eb_transfer_v43 import HARD_GATE_MIN_SECONDS

    data, folds, _ = configs()
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        registries = {seconds: EBTransferRegistry(data, fold, registry30, seconds)
                      for seconds in DURATIONS}
        keys = sorted(registry30.cells)
        payload: dict[str, np.ndarray] = {"keys": np.asarray(["|".join(key) for key in keys])}
        manifest = []
        for seconds, registry in registries.items():
            signatures = []
            for key in keys:
                signature = registry.signature(*key, "EB")
                if seconds < HARD_GATE_MIN_SECONDS:
                    # D-F2 construction check: <60-s support routes to bit-identical POP.
                    pop = registry30.signature(*key, "POP")
                    if not registry.cells[key].hard_gate or not np.array_equal(signature, pop):
                        raise AssertionError(f"D-F2 construction check failed for {key} at {seconds}s")
                signatures.append(signature)
            payload[f"sig_eb{seconds}"] = np.stack(signatures).astype(np.float32)
            payload[f"lambda{seconds}"] = np.asarray([registry.cells[key].lam for key in keys])
            payload[f"hard_gate{seconds}"] = np.asarray([int(registry.cells[key].hard_gate) for key in keys])
            manifest += registry.manifest_rows()
        payload["transfer_eb120"] = np.stack([
            registry30.population_transfer[key[1:]] + registries[120].cells[key].lam *
            (registries[120].cells[key].transfer - registry30.population_transfer[key[1:]])
            for key in keys])
        target = RESULT / "state" / f"fold_{fold['fold']}"
        with np.load(target / "eb_signatures.npz", allow_pickle=False) as archive:
            if (not np.array_equal(np.asarray(archive["eb120"]), payload["sig_eb120"]) or
                    not np.array_equal(np.asarray(archive["eb10"]), payload["sig_eb10"])):
                raise AssertionError(f"S2 state build disagrees with the S1 signature bank (fold {fold['fold']})")
        np.savez_compressed(target / "eb_signatures_s2.npz", **payload)
        pd.DataFrame(manifest).to_csv(target / "eb_state_manifest_s2.csv", index=False)
        summary = {"fold": fold["fold"], "cells": len(keys), "df2_construction_check": "pass",
                   "s1_signature_bank_consistent": True}
        for seconds in DURATIONS:
            lam = payload[f"lambda{seconds}"]
            summary[f"lambda{seconds}"] = {"min": float(lam.min()), "mean": float(lam.mean()),
                                           "max": float(lam.max()),
                                           "hard_gate_count": int(payload[f"hard_gate{seconds}"].sum())}
        (target / "state_summary_s2.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary))


def _load_s2_state(fold_id: int):
    with np.load(RESULT / "state" / f"fold_{fold_id}" / "eb_signatures_s2.npz",
                 allow_pickle=False) as archive:
        state = {name: np.asarray(archive[name]) for name in archive.files}
    index = {str(key): position for position, key in enumerate(state["keys"])}
    return state, index


def _meta_key(meta: Mapping[str, Any]) -> str:
    return "|".join((str(meta["participant"]), str(meta["session"]), str(meta["task"])))


# ------------------------------------------------------------- S2a training

def train_gated(model, schedule, train_sampler, validation_bank, validation_sampler,
                device, seed: int, updates: int, batch_size: int, validation_interval: int,
                runtime: Path, state, index) -> list[dict[str, float]]:
    """Registered clone of train_v42r.train_joint with ONE change: the training
    conditioning state is the EB-gated signature of the episode's owner cell at a
    support duration drawn uniformly from {10, 30, 60, 120} s (hard gate included,
    so 10/30-s draws train the POP route through the gate).  Optimizer, EMA, 20%
    POP context dropout, and the validation-POP checkpoint rule are identical."""
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
        duration_draw = train_sampler.rng.integers(0, len(DURATIONS), batch_size)
        signature = np.stack([
            state[f"sig_eb{DURATIONS[int(draw)]}"][index[_meta_key(meta)]]
            for draw, meta in zip(duration_draw, bank["meta"])])
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
            raise FloatingPointError(f"nonfinite V43 gated loss at update {step}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        transfer_gradient = torch.stack([parameter.grad.detach().norm()
                                         for parameter in model.transfer_parameters()
                                         if parameter.grad is not None]).sum()
        if not torch.isfinite(gradient) or not torch.isfinite(transfer_gradient):
            raise FloatingPointError(f"nonfinite V43 gated gradient at update {step}")
        optimizer.step()
        ema.update(model)
        if step % validation_interval == 0 or step == updates:
            score = participant_pop_score(ema.model, schedule, validation_bank, validation_sampler,
                                          device, seed + 17001)
            curve.append({"step": step, "train_smooth_l1": float(loss.detach()),
                          "validation_pop_rrmse": score, "gradient_norm": float(gradient),
                          "transfer_gradient_norm": float(transfer_gradient),
                          "context_dropout_fraction": float(dropped.mean()),
                          "mean_duration_drawn": float(np.mean([DURATIONS[int(d)] for d in duration_draw]))})
            payload = {"model": model.state_dict(), "ema": ema.state_dict(), "step": step,
                       "seed": seed, "curve": curve, "best_validation_pop_rrmse": min(best, score),
                       "conditioning": "EB_gated_duration_randomized_10_30_60_120"}
            runtime.mkdir(parents=True, exist_ok=True)
            torch.save(payload, last_path)
            if score < best:
                best = score
                payload["best_validation_pop_rrmse"] = score
                torch.save(payload, best_path)
    if not best_path.is_file():
        raise RuntimeError("V43 gated training created no checkpoint")
    return curve


def stage2_train(fold_id: int, seed: int, updates: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule

    result_dir = RESULT / "stage2" / f"fold_{fold_id}_seed_{seed}"
    curve_path = result_dir / "train_curve.json"
    if curve_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "training already complete"}))
        return
    data, folds, training = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = DERIVED / "stage2" / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    state, index = _load_s2_state(fold_id)
    model = CalibSADDPMCond().to(device)
    schedule = LinearX0Schedule().to(device)
    curve = train_gated(model, schedule, train_sampler, validation_bank, validation_sampler,
                        device, seed, updates, training["effective_batch"],
                        training["validation_interval"], runtime, state, index)
    payload = torch.load(runtime / "best.pt", map_location="cpu", weights_only=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "updates": updates,
        "conditioning": "EB_gated_duration_randomized_10_30_60_120",
        "checkpoint": str(runtime / "best.pt"), "checkpoint_best_step": payload["step"],
        "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"],
        "training_curve": curve, "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "best_step": payload["step"],
                      "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"]}))


def stage2_eval(fold_id: int, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule
    from eeg_scad.training.train_v42r import _conditions, sample_bank

    result_dir = RESULT / "stage2" / f"fold_{fold_id}_seed_{seed}"
    result_path = result_dir / "stage2_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "eval already complete"}))
        return
    source = json.loads((result_dir / "train_curve.json").read_text())
    checkpoint = Path(source["checkpoint"])
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    state, index = _load_s2_state(fold_id)
    model = CalibSADDPMCond().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    schedule = LinearX0Schedule().to(device)
    ns = noise_seed(fold_id, seed)

    rows = []
    for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"], "condition": "RAW",
                     "context_owner": "NONE", "lambda": np.nan, "hard_gate": 0,
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    wrong_owners = [sampler.condition_signature(meta, "WRONG")[1] for meta in bank["meta"]]
    recipient_keys = [_meta_key(meta) for meta in bank["meta"]]
    wrong_keys = ["|".join((owner, meta["session"], meta["task"]))
                  for owner, meta in zip(wrong_owners, bank["meta"])]
    outputs: dict[str, np.ndarray] = {}
    context: dict[str, list[tuple[str, float, int]]] = {}
    for condition in ("POP", "WRONG", "NO_TRANSFER_BRANCH"):
        signature, owners = _conditions(sampler, bank["meta"], condition)
        context[condition] = [(owner, np.nan, 0) for owner in owners]
        outputs[condition] = sample_bank(model, schedule, bank["y"], signature, device, ns,
                                         transfer_enabled=condition != "NO_TRANSFER_BRANCH")
    for seconds in DURATIONS:
        positions = [index[key] for key in recipient_keys]
        signature = np.stack([state[f"sig_eb{seconds}"][position] for position in positions])
        context[f"MATCH_EB{seconds}"] = [
            (meta["participant"], float(state[f"lambda{seconds}"][position]),
             int(state[f"hard_gate{seconds}"][position]))
            for meta, position in zip(bank["meta"], positions)]
        outputs[f"MATCH_EB{seconds}"] = sample_bank(model, schedule, bank["y"], signature, device, ns)
    positions = [index[key] for key in wrong_keys]
    signature = np.stack([state["sig_eb120"][position] for position in positions])
    context["WRONG_EB120"] = [(owner, float(state["lambda120"][position]),
                               int(state["hard_gate120"][position]))
                              for owner, position in zip(wrong_owners, positions)]
    outputs["WRONG_EB120"] = sample_bank(model, schedule, bank["y"], signature, device, ns)
    for condition, output in outputs.items():
        for clean, observed, artifact, prediction, meta, (owner, lam, gate) in zip(
                bank["x"], bank["y"], bank["artifact"], output, bank["meta"], context[condition]):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V43 stage2 output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": condition, "context_owner": owner, "lambda": lam, "hard_gate": gate,
                         "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         **paired_metrics(clean, observed, artifact, observed - prediction)})
    bypass_qc = {}
    for seconds in (10, 30):
        delta = float(np.max(np.abs(outputs[f"MATCH_EB{seconds}"] - outputs["POP"])))
        bypass_qc[f"eb{seconds}_pop_output_max_abs_diff"] = delta
        if delta > 1e-6:
            raise AssertionError(f"MATCH_EB{seconds} bypass is not the POP route (max delta {delta})")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "checkpoint": str(checkpoint),
        "checkpoint_best_step": source["checkpoint_best_step"], "noise_seed": ns,
        "bypass_qc": bypass_qc, "sealed_reads": 0, "query_eog_inference_reads": 0,
        "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "bypass_qc": bypass_qc}))


# ------------------------------------------------------------- S2a DET twin

def stage2_det(fold_id: int, seed: int, updates: int) -> None:
    """Capacity-matched deterministic twin: same backbone, x_t := y, fixed t=0
    embedding, direct MSE to clean; identical optimizer/updates/EMA/dropout and
    the same duration-randomized EB conditioning.  Descriptive positioning only."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond
    from eeg_scad.training.train_v42r import EMA, _conditions

    result_dir = RESULT / "stage2" / f"det_fold_{fold_id}_seed_{seed}"
    result_path = result_dir / "det_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "det already complete"}))
        return
    data, folds, training = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = DERIVED / "stage2_det" / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    test_sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    test_bank = test_sampler.sample_balanced(8)
    state, index = _load_s2_state(fold_id)
    model = CalibSADDPMCond().to(device)

    def det_forward(net, observed_np, signature_np, batch: int = 8):
        pieces = []
        with torch.no_grad():
            for start in range(0, len(observed_np), batch):
                observed = torch.from_numpy(np.asarray(observed_np[start:start + batch], np.float32)).to(device)
                signature = torch.from_numpy(np.asarray(signature_np[start:start + batch], np.float32)).to(device)
                timestep = torch.zeros(len(observed), dtype=torch.long, device=device)
                pieces.append(net(observed, observed, timestep, signature).cpu().numpy())
        return np.concatenate(pieces)

    def det_pop_score(net) -> float:
        signature, _ = _conditions(validation_sampler, validation_bank["meta"], "POP")
        prediction = det_forward(net, validation_bank["y"], signature)
        values: dict[str, list[float]] = {}
        for clean, output, meta in zip(validation_bank["x"], prediction, validation_bank["meta"]):
            values.setdefault(meta["participant"], []).append(
                float(np.linalg.norm(output - clean) / max(np.linalg.norm(clean), 1e-12)))
        return float(np.mean([np.mean(v) for v in values.values()]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model, .999)
    curve, best = [], float("inf")
    batch_size = training["effective_batch"]
    for step in range(1, updates + 1):
        bank = train_sampler.sample(batch_size)
        duration_draw = train_sampler.rng.integers(0, len(DURATIONS), batch_size)
        signature = np.stack([
            state[f"sig_eb{DURATIONS[int(draw)]}"][index[_meta_key(meta)]]
            for draw, meta in zip(duration_draw, bank["meta"])])
        dropped = train_sampler.rng.random(batch_size) < .20
        for position in np.flatnonzero(dropped):
            signature[position], _ = train_sampler.condition_signature(bank["meta"][int(position)], "POP")
        clean = torch.from_numpy(np.asarray(bank["x"], np.float32)).to(device)
        observed = torch.from_numpy(np.asarray(bank["y"], np.float32)).to(device)
        condition = torch.from_numpy(np.asarray(signature, np.float32)).to(device)
        timestep = torch.zeros(len(observed), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(observed, observed, timestep, condition)
        loss = F.mse_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite V43 DET loss at update {step}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"nonfinite V43 DET gradient at update {step}")
        optimizer.step()
        ema.update(model)
        if step % training["validation_interval"] == 0 or step == updates:
            score = det_pop_score(ema.model)
            curve.append({"step": step, "train_mse": float(loss.detach()),
                          "validation_pop_rrmse": score, "gradient_norm": float(gradient)})
            payload = {"ema": ema.state_dict(), "step": step, "seed": seed, "curve": curve,
                       "best_validation_pop_rrmse": min(best, score),
                       "conditioning": "EB_gated_duration_randomized_10_30_60_120",
                       "architecture": "DET_twin_one_step_t0"}
            torch.save(payload, runtime / "last.pt")
            if score < best:
                best = score
                payload["best_validation_pop_rrmse"] = score
                torch.save(payload, runtime / "best.pt")
    payload = torch.load(runtime / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    model.eval()
    rows = []
    for clean, observed, artifact, meta in zip(test_bank["x"], test_bank["y"], test_bank["artifact"], test_bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"], "condition": "RAW",
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    arm_signatures = {"DET_POP": _conditions(test_sampler, test_bank["meta"], "POP")[0],
                      "DET_MATCH_EB120": np.stack([state["sig_eb120"][index[_meta_key(meta)]]
                                                   for meta in test_bank["meta"]])}
    for condition, signature in arm_signatures.items():
        output = det_forward(model, test_bank["y"], signature)
        for clean, observed, artifact, prediction, meta in zip(
                test_bank["x"], test_bank["y"], test_bank["artifact"], output, test_bank["meta"]):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V43 DET output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": condition, "zero_artifact": meta["zero_artifact"],
                         "gain": meta["gain"],
                         **paired_metrics(clean, observed, artifact, observed - prediction)})
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "updates": updates, "checkpoint": str(runtime / "best.pt"),
        "checkpoint_best_step": payload["step"],
        "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"],
        "training_curve": curve, "sealed_reads": 0, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "best_step": payload["step"],
                      "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"]}))


# ------------------------------------------------------ S2a LINEAR reference

def stage2_linear() -> None:
    """LINEAR-EOG reference: clean_hat = y - C_gated @ E, where E is the exact
    latent EOG drive recovered from the generative artifact (teacher/evaluator
    role).  Requires query EOG at inference; not information-matched."""
    data, folds, _ = configs()
    all_rows = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        state, index = _load_s2_state(int(fold["fold"]))
        pinv_cache = {}
        for seed in S2_SEEDS:
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
                key = (meta["participant"], meta["session"], meta["task"])
                if key not in pinv_cache:
                    pinv_cache[key] = np.linalg.pinv(registry30.cells[key].query_transfer)
                drive = pinv_cache[key] @ np.asarray(artifact, np.float64)
                gated = state["transfer_eb120"][index[_meta_key(meta)]]
                predicted = gated @ drive
                all_rows.append({"fold": int(fold["fold"]), "seed": seed,
                                 "participant": meta["participant"], "condition": "LINEAR_EB120",
                                 "uses_query_eog_at_inference": 1, "information_matched": 0,
                                 "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                                 **paired_metrics(clean, observed, artifact, predicted)})
    target = RESULT / "stage2"
    target.mkdir(parents=True, exist_ok=True)
    (target / "linear_result.json").write_text(json.dumps({
        "label": "requires query EOG at inference; not information-matched",
        "rows": all_rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"linear_rows": len(all_rows)}))


# ---------------------------------------------------------- S2c privacy curve

def stage2c_privacy() -> None:
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog, ridge_transfer
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale
    from eeg_scad.evaluation.linkage_diagnostic import linkage

    data, folds, _ = configs()
    lambdas = (("0.00", 0.0), ("0.25", 0.25), ("0.50", 0.5), ("0.75", 0.75),
               ("EB", None), ("1.00", 1.0))
    rows = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        participants = sorted(set(fold["train"] + fold["validation"] + fold["test"]))
        halves, lam_hat = {}, {}
        for participant in participants:
            key = next((k for k in sorted(registry30.cells) if k[0] == participant), None)
            if key is None:
                continue
            eeg, eye, names = registry30._load(*key)
            eog = bipolar_eog(eye, names)
            pair = []
            for start, stop in ((0, 6000), (6000, 12000)):
                center, scale = robust_center_scale(eog[:, start:stop])
                latent = (eog[:, start:stop] - center[:, None]) / scale[:, None]
                scaled_eeg = eeg[:, start:stop] / registry30.eeg_scale[:, None]
                transfer, diagnostics = ridge_transfer(scaled_eeg, latent, registry30.ridge_ratio)
                rms = np.sqrt(np.mean((eog[:, start:stop] - center[:, None]) ** 2, axis=1)).clip(1e-8)
                quality = np.array([np.log(rms[0]), np.log(rms[1]), diagnostics["fit_r2"],
                                    np.log1p(diagnostics["condition_number"])])
                pair.append((transfer, quality))
            halves[participant] = (pair, key)
            lam_hat[participant] = eb120.cells[key].lam
        for label, lam_value in lambdas:
            features = {}
            for participant, (pair, key) in halves.items():
                lam = lam_hat[participant] if lam_value is None else lam_value
                pop_transfer = registry30.population_transfer[key[1:]]
                pop_quality = registry30.population_quality[key[1:]]
                halves_features = []
                for transfer, quality in pair:
                    clamped = np.clip(quality, eb120.quality_min, eb120.quality_max)
                    gated_transfer = pop_transfer + lam * (transfer - pop_transfer)
                    gated_quality = pop_quality + lam * (clamped - pop_quality)
                    continuous = ((registry30._continuous(gated_transfer, gated_quality)
                                   - registry30.continuous_center) / registry30.continuous_scale)
                    halves_features.append(continuous.reshape(-1))
                features[participant] = (halves_features[0], halves_features[1])
            metrics = linkage(features)[0]
            rows.append({"fold": fold["fold"], "lambda_label": label,
                         "lambda_mean": float(np.mean([lam_hat[p] if lam_value is None else lam_value
                                                       for p in features])),
                         "state_bytes_float32": 46 * 53 * 4, **metrics})
    target = RESULT / "stage2c"
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "lambda_privacy.csv", index=False)
    summary = frame.groupby("lambda_label", as_index=False)[
        ["lambda_mean", "top1_accuracy", "same_different_auroc"]].mean()
    print(summary.to_string(index=False))


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


def ceiling_panel() -> dict[str, Any]:
    data, folds, _ = configs()
    rows = []
    for fold_id in range(5):
        path = RESULT / "stage15" / f"fold_{fold_id}_seed_20261201" / "stage15_result.json"
        rows += json.loads(path.read_text())["rows"]
    frame = pd.DataFrame(rows)
    pop = _participant_means(frame, "POP")
    oracle = _participant_means(frame, "ORACLE")
    delta = (pop - oracle).loc[pop.index]
    fold_of = {participant: fold["fold"] for fold in folds for participant in fold["test"]}
    fold_means = {str(fold_id): float(np.mean([value for participant, value in delta.items()
                                               if fold_of[participant] == fold_id]))
                  for fold_id in range(5)}
    payload = {
        "descriptive_only": True,
        "no_go_final": "the S1.5 NO-GO is final; this panel completion cannot reopen the gain leg",
        "contrast": "POP_minus_ORACLE", **_stat(delta),
        "per_participant": {participant: float(value) for participant, value in delta.items()},
        "fold_mean": fold_means,
        "sign_heterogeneity": {"positive_folds": sorted(k for k, v in fold_means.items() if v > 0),
                               "negative_folds": sorted(k for k, v in fold_means.items() if v <= 0)},
        "cells": [{"fold": fold_id, "seed": 20261201} for fold_id in range(5)],
        "condition_means": {"RAW": float(_participant_means(frame, "RAW").mean()),
                            "POP": float(pop.mean()), "ORACLE": float(oracle.mean())},
        "sealed_reads": 0,
    }
    target = RESULT / "stage2b"
    target.mkdir(parents=True, exist_ok=True)
    (target / "ceiling_panel.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def aggregate_stage2() -> dict[str, Any]:
    rows, bypass = [], {}
    for fold_id in range(5):
        for seed in S2_SEEDS:
            payload = json.loads((RESULT / "stage2" / f"fold_{fold_id}_seed_{seed}"
                                  / "stage2_result.json").read_text())
            rows += payload["rows"]
            for key, value in payload["bypass_qc"].items():
                bypass[f"fold_{fold_id}_seed_{seed}_{key}"] = value
    frame = pd.DataFrame(rows)
    conditions = ["RAW", "POP", "WRONG", "NO_TRANSFER_BRANCH", "WRONG_EB120"] + \
                 [f"MATCH_EB{seconds}" for seconds in DURATIONS]
    per = {condition: _participant_means(frame, condition) for condition in conditions}
    participants = per["POP"].index

    d_wrong_eb = (per["WRONG_EB120"] - per["POP"]).loc[participants]
    reduction = (per["WRONG"] - per["WRONG_EB120"]).loc[participants]
    d_match = {seconds: (per[f"MATCH_EB{seconds}"] - per["POP"]).loc[participants]
               for seconds in DURATIONS}
    draws_f1 = bootstrap_draws(d_wrong_eb.to_numpy())
    draws_f3 = bootstrap_draws(d_match[120].to_numpy())
    draws_f4 = {seconds: bootstrap_draws(d_match[seconds].to_numpy()) for seconds in DURATIONS}
    reduction_stat = _stat(reduction)

    df1 = {"contrast": "WRONG_EB120_minus_POP", **_stat(d_wrong_eb), "margin": D_MARGINS["D-F1"],
           "reduction_vs_ungated_wrong": reduction_stat,
           "ungated_wrong_harm_mean": float((per["WRONG"] - per["POP"]).mean()),
           "pass": bool(d_wrong_eb.mean() <= D_MARGINS["D-F1"] and reduction_stat["bootstrap_low"] > 0)}
    state_ok = all(json.loads((RESULT / "state" / f"fold_{fold_id}" / "state_summary_s2.json")
                              .read_text())["df2_construction_check"] == "pass" for fold_id in range(5))
    bypass_ok = all(value <= 1e-6 for value in bypass.values())
    df2 = {"construction_check": "hard gate routes <60-s support to the bit-identical POP state",
           "state_build_assert": state_ok, "output_bypass_assert": bypass_ok,
           "max_output_bypass_diff": float(max(bypass.values())),
           "pass": bool(state_ok and bypass_ok)}
    upper95 = float(np.quantile(draws_f3, 0.95))
    df3 = {"contrast": "MATCH_EB120_minus_POP", **_stat(d_match[120]), "margin": D_MARGINS["D-F3"],
           "one_sided_upper95": upper95, "upper_margin": D_MARGINS["D-F3_upper"],
           "pass": bool(d_match[120].mean() <= D_MARGINS["D-F3"] and upper95 <= D_MARGINS["D-F3_upper"])}
    df4 = {"contrast": "MATCH_EBd_minus_POP for every d", "margin": D_MARGINS["D-F4"],
           "per_duration": {str(seconds): _stat(d_match[seconds]) for seconds in DURATIONS},
           "note": "duration curve shape is descriptive; no monotone-benefit claim",
           "pass": bool(all(d_match[seconds].mean() <= D_MARGINS["D-F4"] for seconds in DURATIONS))}
    p_raw = {"D-F1": float(np.mean(draws_f1 >= D_MARGINS["D-F1"])),
             "D-F3": float(np.mean(draws_f3 >= D_MARGINS["D-F3"])),
             "D-F4": float(max(np.mean(draws_f4[seconds] >= D_MARGINS["D-F4"])
                               for seconds in DURATIONS))}
    p_adjusted = holm(p_raw)
    branch = {"NO_TRANSFER_minus_MATCH_EB120":
              _stat((per["NO_TRANSFER_BRANCH"] - per["MATCH_EB120"]).loc[participants]),
              "NO_TRANSFER_minus_POP":
              _stat((per["NO_TRANSFER_BRANCH"] - per["POP"]).loc[participants])}

    det_rows = []
    for fold_id in range(5):
        for seed in DET_SEEDS:
            det_rows += json.loads((RESULT / "stage2" / f"det_fold_{fold_id}_seed_{seed}"
                                    / "det_result.json").read_text())["rows"]
    det_frame = pd.DataFrame(det_rows)
    det_pop = _participant_means(det_frame, "DET_POP")
    det_match = _participant_means(det_frame, "DET_MATCH_EB120")
    det = {"positioning_only": True, "DET_POP_mean": float(det_pop.mean()),
           "DET_MATCH_EB120_mean": float(det_match.mean()),
           "DET_MATCH_EB120_minus_DET_POP": _stat((det_match - det_pop).loc[det_pop.index]),
           "DIFFUSION_POP_minus_DET_POP": _stat((per["POP"] - det_pop).loc[participants])}
    linear_payload = json.loads((RESULT / "stage2" / "linear_result.json").read_text())
    linear_per = _participant_means(pd.DataFrame(linear_payload["rows"]), "LINEAR_EB120")
    linear = {"label": linear_payload["label"], "LINEAR_EB120_mean": float(linear_per.mean()),
              "LINEAR_EB120_minus_POP": _stat((linear_per - per["POP"]).loc[participants])}

    manifests = pd.concat([pd.read_csv(RESULT / "state" / f"fold_{fold_id}" / "eb_state_manifest_s2.csv")
                           for fold_id in range(5)], ignore_index=True)
    lambda_distribution = {}
    for seconds in DURATIONS:
        block = manifests[manifests.seconds == seconds]
        lambda_distribution[str(seconds)] = {
            "mean": float(block["lambda"].mean()), "min": float(block["lambda"].min()),
            "max": float(block["lambda"].max()), "hard_gate_fraction": float(block.hard_gate.mean())}

    decision = {
        "preregistration": "reports/v43_preregistration.md (V43-S2 addendum)",
        "stage": "S2_floor_definitive_gated_retraining",
        "D-F1": df1, "D-F2": df2, "D-F3": df3, "D-F4": df4,
        "holm": {"p_raw": p_raw, "p_adjusted": p_adjusted, "alpha": 0.05,
                 "note": "one-sided bootstrap probability that the endpoint mean exceeds its margin; "
                         "D-F4 uses the max over durations (intersection-union)"},
        "condition_means": {condition: float(series.mean()) for condition, series in per.items()},
        "branch_necessity": branch, "det_positioning": det, "linear_positioning": linear,
        "lambda_distribution": lambda_distribution, "cells": 15, "sealed_reads": 0,
        "no_gain_endpoints": True,
    }
    target = RESULT / "stage2"
    (target / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    panel = ceiling_panel()
    privacy = pd.read_csv(RESULT / "stage2c" / "lambda_privacy.csv")
    privacy_summary = privacy.groupby("lambda_label", as_index=False)[
        ["lambda_mean", "top1_accuracy", "top3_accuracy", "same_different_auroc"]].mean()

    condition_table = pd.DataFrame([{"condition": condition,
                                     "participant_mean_rrmse_temporal": float(series.mean())}
                                    for condition, series in per.items()])
    endpoint_table = pd.DataFrame([
        {"endpoint": "D-F1", "contrast": df1["contrast"], "mean": df1["mean"],
         "bootstrap_low": df1["bootstrap_low"], "bootstrap_high": df1["bootstrap_high"],
         "margin": df1["margin"], "pass": df1["pass"]},
        {"endpoint": "D-F1_reduction", "contrast": "WRONG_minus_WRONG_EB120", **reduction_stat},
        {"endpoint": "D-F3", "contrast": df3["contrast"], "mean": df3["mean"],
         "bootstrap_low": df3["bootstrap_low"], "bootstrap_high": df3["bootstrap_high"],
         "margin": df3["margin"], "one_sided_upper95": upper95, "pass": df3["pass"]}] + [
        {"endpoint": f"D-F4[{seconds}s]", "contrast": f"MATCH_EB{seconds}_minus_POP",
         **df4["per_duration"][str(seconds)], "margin": D_MARGINS["D-F4"]}
        for seconds in DURATIONS])
    duration_table = pd.DataFrame([
        {"support_seconds": seconds,
         "MATCH_EBd_mean_rrmse": float(per[f"MATCH_EB{seconds}"].mean()),
         "delta_vs_POP": float(d_match[seconds].mean())} for seconds in DURATIONS])
    forest_table = pd.DataFrame([{"participant": participant, "pop_minus_oracle": value}
                                 for participant, value in panel["per_participant"].items()])
    (REPORT / "v43_stage2.md").write_text(
        "# V43 Stage 2 — floor-definitive round\n\n"
        "Preregistration: V43-S2 addendum in `reports/v43_preregistration.md` (frozen before "
        "submission). Retrained gated model (duration-randomized EB conditioning), 15 cells; "
        "no gain endpoints anywhere; the S1.5 NO-GO stands.\n\n"
        f"Decision: D-F1 **{df1['pass']}**, D-F2 **{df2['pass']}**, D-F3 **{df3['pass']}**, "
        f"D-F4 **{df4['pass']}**.\n\n"
        "## Participant-first condition means (temporal RRMSE, n=15)\n\n"
        + condition_table.round(6).to_markdown(index=False)
        + "\n\n## Definitive floor endpoints\n\n"
        + endpoint_table.round(6).to_markdown(index=False)
        + "\n\nD-F2 (construction check): " + json.dumps(df2) + "\n\n"
        f"Holm over {{D-F1, D-F3, D-F4}}: raw p {json.dumps(p_raw)}; adjusted {json.dumps(p_adjusted)}\n\n"
        "## Support-duration curve (descriptive)\n\n"
        + duration_table.round(6).to_markdown(index=False)
        + "\n\n## Branch necessity on the retrained model\n\n```json\n"
        + json.dumps(branch, indent=2, sort_keys=True) + "\n```\n\n"
        "## Positioning (descriptive only, no superiority claims)\n\n```json\n"
        + json.dumps({"DET_twin": det, "LINEAR_EOG": linear}, indent=2, sort_keys=True) + "\n```\n\n"
        "## Retrained-state builder lambda distribution\n\n```json\n"
        + json.dumps(lambda_distribution, indent=2, sort_keys=True) + "\n```\n\n"
        "## S2b — ceiling completion (descriptive; NO-GO final)\n\n"
        f"Pooled n=15 POP-ORACLE: mean {panel['mean']:+.6f}, CI [{panel['bootstrap_low']:+.6f}, "
        f"{panel['bootstrap_high']:+.6f}], positive {panel['positive_count']}/15. "
        f"Fold means: {json.dumps(panel['fold_mean'])}.\n\n"
        + forest_table.round(6).to_markdown(index=False)
        + "\n\n## S2c — lambda-privacy curve (descriptive; no privacy-safe claim)\n\n"
        + privacy_summary.round(6).to_markdown(index=False) + "\n")
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
    sub.add_parser("build-state-s2")
    two_train = sub.add_parser("stage2-train")
    two_train.add_argument("--fold", type=int, required=True)
    two_train.add_argument("--seed", type=int, required=True)
    two_train.add_argument("--updates", type=int, required=True)
    two_eval = sub.add_parser("stage2-eval")
    two_eval.add_argument("--fold", type=int, required=True)
    two_eval.add_argument("--seed", type=int, required=True)
    two_det = sub.add_parser("stage2-det")
    two_det.add_argument("--fold", type=int, required=True)
    two_det.add_argument("--seed", type=int, required=True)
    two_det.add_argument("--updates", type=int, required=True)
    sub.add_parser("stage2-linear")
    sub.add_parser("stage2c")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--which", choices=("1", "15", "2", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "build-state":
        build_state()
    elif args.stage == "build-state-s2":
        build_state_s2()
    elif args.stage == "stage1":
        stage1(args.fold, args.seed)
    elif args.stage == "stage15":
        stage15(args.fold, args.seed, args.updates)
    elif args.stage == "stage2-train":
        stage2_train(args.fold, args.seed, args.updates)
    elif args.stage == "stage2-eval":
        stage2_eval(args.fold, args.seed)
    elif args.stage == "stage2-det":
        stage2_det(args.fold, args.seed, args.updates)
    elif args.stage == "stage2-linear":
        stage2_linear()
    elif args.stage == "stage2c":
        stage2c_privacy()
    else:
        if args.which in ("1", "all"):
            print(json.dumps(aggregate_stage1(), indent=2, sort_keys=True))
        if args.which in ("15", "all"):
            print(json.dumps(aggregate_stage15(), indent=2, sort_keys=True))
        if args.which == "2":
            print(json.dumps(aggregate_stage2(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
