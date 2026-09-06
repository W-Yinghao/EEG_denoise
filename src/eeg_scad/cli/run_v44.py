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


# ------------------------------------------------------------------- stage 1

ARM_SET = ("POP", "MATCH_gated", "WRONG", "WRONG_gated", "SHUFFLED", "NO_A0", "ORACLE")


def _gated_assets(registry30, eb120) -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    assets = {}
    for key in registry30.cells:
        if key[1:] not in registry30.population_transfer:
            continue
        assets[key] = {
            "pinv_query": np.linalg.pinv(registry30.cells[key].query_transfer),
            "C_query": registry30.cells[key].query_transfer,
            "C_gated": eb120.operator(*key, "EB"),
            "C_raw": eb120.operator(*key, "RAW"),
            "C0": registry30.population_transfer[key[1:]],
            "sig_gated": eb120.signature(*key, "EB"),
            "sig_raw": eb120.signature(*key, "RAW"),
            "sig_pop": registry30.signature(*key, "POP"),
        }
    return assets


def _bank_drives(assets, bank) -> np.ndarray:
    drives = []
    for meta, artifact in zip(bank["meta"], bank["artifact"]):
        key = (meta["participant"], meta["session"], meta["task"])
        drives.append(assets[key]["pinv_query"] @ np.asarray(artifact, np.float64))
    return np.stack(drives)


def sample_bank_eog(model, schedule, observed, a0, transfer, device, seed: int,
                    transfer_enabled: bool = True, episode_batch: int = 2) -> np.ndarray:
    import torch
    from eeg_scad.models.calib_saddpm_eog_v44 import ddim_sample_eog

    model.eval()
    outputs = []
    for start in range(0, len(observed), episode_batch):
        stop = min(len(observed), start + episode_batch)
        y = torch.from_numpy(np.asarray(observed[start:stop], np.float32)).to(device)
        anchor = torch.from_numpy(np.asarray(a0[start:stop], np.float32)).to(device)
        condition = torch.from_numpy(np.asarray(transfer[start:stop], np.float32)).to(device)
        generator = torch.Generator(device=device).manual_seed(seed + start * 1009)
        noise = torch.randn(y.shape, device=device, generator=generator)
        outputs.append(ddim_sample_eog(model, y, anchor, condition, noise, schedule, 50,
                                       transfer_enabled).cpu().numpy())
    return np.concatenate(outputs)


def _pop_score_eog(model, schedule, bank, assets, device, seed: int) -> float:
    drives = _bank_drives(assets, bank)
    a0, signature = [], []
    for meta, drive in zip(bank["meta"], drives):
        key = (meta["participant"], meta["session"], meta["task"])
        a0.append(assets[key]["C0"] @ drive)
        signature.append(assets[key]["sig_pop"])
    prediction = sample_bank_eog(model, schedule, bank["y"], np.stack(a0), np.stack(signature),
                                 device, seed)
    values: dict[str, list[float]] = {}
    for clean, output, meta in zip(bank["x"], prediction, bank["meta"]):
        values.setdefault(meta["participant"], []).append(
            float(np.linalg.norm(output - clean) / max(np.linalg.norm(clean), 1e-12)))
    return float(np.mean([np.mean(v) for v in values.values()]))


def stage1_train(fold_id: int, seed: int, updates: int) -> None:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG
    from eeg_scad.training.train_v42r import EMA

    result_dir = RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
    curve_path = result_dir / "train_curve.json"
    if curve_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "training already complete"}))
        return
    data, folds, training = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = DERIVED / "stage1" / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    model = CalibSADDPMEOG().to(device)
    schedule = LinearX0Schedule().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model, .999)
    generator = torch.Generator(device=device).manual_seed(seed + 7001)
    batch_size = training["effective_batch"]
    curve, best = [], float("inf")
    for step in range(1, updates + 1):
        bank = train_sampler.sample(batch_size)
        drives = _bank_drives(assets, bank)
        a0, signature = [], []
        for meta, drive in zip(bank["meta"], drives):
            key = (meta["participant"], meta["session"], meta["task"])
            a0.append(assets[key]["C_gated"] @ drive)
            signature.append(assets[key]["sig_gated"].copy())
        a0 = np.stack(a0)
        signature = np.stack(signature)
        a0_dropped = train_sampler.rng.random(batch_size) < .30
        a0[a0_dropped] = 0.0
        context_dropped = train_sampler.rng.random(batch_size) < .20
        for position in np.flatnonzero(context_dropped):
            meta = bank["meta"][int(position)]
            signature[position] = assets[(meta["participant"], meta["session"], meta["task"])]["sig_pop"]
        clean = torch.from_numpy(np.asarray(bank["x"], np.float32)).to(device)
        observed = torch.from_numpy(np.asarray(bank["y"], np.float32)).to(device)
        anchor = torch.from_numpy(np.asarray(a0, np.float32)).to(device)
        condition = torch.from_numpy(np.asarray(signature, np.float32)).to(device)
        noisy, timestep, _ = schedule.forward_sample(clean, generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, observed, anchor, timestep, condition)
        loss = F.smooth_l1_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite V44 loss at update {step}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        transfer_gradient = torch.stack([parameter.grad.detach().norm()
                                         for parameter in model.transfer_parameters()
                                         if parameter.grad is not None]).sum()
        if not torch.isfinite(gradient) or not torch.isfinite(transfer_gradient):
            raise FloatingPointError(f"nonfinite V44 gradient at update {step}")
        optimizer.step()
        ema.update(model)
        if step % training["validation_interval"] == 0 or step == updates:
            score = _pop_score_eog(ema.model, schedule, validation_bank, assets, device, seed + 17001)
            curve.append({"step": step, "train_smooth_l1": float(loss.detach()),
                          "validation_pop_rrmse": score, "gradient_norm": float(gradient),
                          "transfer_gradient_norm": float(transfer_gradient),
                          "a0_dropout_fraction": float(a0_dropped.mean()),
                          "context_dropout_fraction": float(context_dropped.mean())})
            payload = {"model": model.state_dict(), "ema": ema.state_dict(), "step": step,
                       "seed": seed, "curve": curve, "best_validation_pop_rrmse": min(best, score),
                       "conditioning": "EOG_guided_a0_Cgated_30pct_dropout"}
            runtime.mkdir(parents=True, exist_ok=True)
            torch.save(payload, runtime / "last.pt")
            if score < best:
                best = score
                payload["best_validation_pop_rrmse"] = score
                torch.save(payload, runtime / "best.pt")
    if not (runtime / "best.pt").is_file():
        raise RuntimeError("V44 training created no checkpoint")
    payload = torch.load(runtime / "best.pt", map_location="cpu", weights_only=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "updates": updates,
        "checkpoint": str(runtime / "best.pt"), "checkpoint_best_step": payload["step"],
        "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"],
        "training_curve": curve, "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "best_step": payload["step"],
                      "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"]}))


def _natural_metrics_full(y: np.ndarray, drive: np.ndarray, out: np.ndarray) -> dict[str, float]:
    from scipy import signal as scipy_signal

    estimate = y - out
    metrics = _natural_metrics(y, drive, estimate)
    energy = np.sqrt(np.mean(drive * drive, axis=0))
    low = energy <= np.quantile(energy, .3)
    frequencies, p0 = scipy_signal.welch(y[:, low], fs=100, nperseg=min(128, int(low.sum())), axis=-1)
    _, p1 = scipy_signal.welch(out[:, low], fs=100, nperseg=min(128, int(low.sum())), axis=-1)
    keep = (frequencies >= 1) & (frequencies <= 15)
    covariance = np.cov(y[:, low])
    metrics["psd_distortion"] = float(np.mean(np.abs(np.log(p0[:, keep] + 1e-8)
                                                     - np.log(p1[:, keep] + 1e-8))))
    metrics["covariance_distortion"] = float(np.linalg.norm(np.cov(out[:, low]) - covariance)
                                             / max(np.linalg.norm(covariance), 1e-8))
    return metrics


def _arm_inputs(assets, key, wrong_key, drive, shuffled_drive):
    zero = np.zeros((len(assets[key]["C0"]), drive.shape[1]))
    return {
        "POP": (assets[key]["C0"] @ drive, assets[key]["sig_pop"]),
        "MATCH_gated": (assets[key]["C_gated"] @ drive, assets[key]["sig_gated"]),
        "WRONG": (assets[wrong_key]["C_raw"] @ drive, assets[wrong_key]["sig_raw"]),
        "WRONG_gated": (assets[wrong_key]["C_gated"] @ drive, assets[wrong_key]["sig_gated"]),
        "SHUFFLED": (assets[key]["C_gated"] @ shuffled_drive, assets[key]["sig_gated"]),
        "NO_A0": (zero, assets[key]["sig_gated"]),
        "ORACLE": (assets[key]["C_query"] @ drive, assets[key]["sig_gated"]),
    }


def stage1_eval(fold_id: int, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    result_dir = RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
    result_path = result_dir / "stage1_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "eval already complete"}))
        return
    source = json.loads((result_dir / "train_curve.json").read_text())
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    drives = _bank_drives(assets, bank)
    shuffle_rng = np.random.default_rng(730000 + fold_id * 100 + seed % 100)
    shuffled = np.stack([drive[:, shuffle_rng.permutation(drive.shape[1])] for drive in drives])
    wrongs = [sampler.condition_signature(meta, "WRONG")[1] for meta in bank["meta"]]

    rows = []
    for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                     "condition": "RAW", "context_owner": "NONE", "oracle_non_deployable": 0,
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    for arm in ARM_SET:
        a0_stack, sig_stack, owners = [], [], []
        for meta, drive, shuffled_drive, wrong in zip(bank["meta"], drives, shuffled, wrongs):
            key = (meta["participant"], meta["session"], meta["task"])
            wrong_key = (wrong, meta["session"], meta["task"])
            a0, sig = _arm_inputs(assets, key, wrong_key, drive, shuffled_drive)[arm]
            a0_stack.append(a0)
            sig_stack.append(sig)
            owners.append(wrong if arm.startswith("WRONG") else meta["participant"])
        output = sample_bank_eog(model, schedule, bank["y"], np.stack(a0_stack),
                                 np.stack(sig_stack), device, noise_seed(fold_id, seed))
        for clean, observed, artifact, prediction, meta, owner in zip(
                bank["x"], bank["y"], bank["artifact"], output, bank["meta"], owners):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V44 stage1 output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": arm, "context_owner": owner,
                         "oracle_non_deployable": int(arm == "ORACLE"),
                         "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         **paired_metrics(clean, observed, artifact, observed - prediction)})

    natural_rows = []
    natural_shuffle_rng = np.random.default_rng(740000 + fold_id * 100 + seed % 100)
    for participant, session, task in itertools.product(fold["test"], data["sessions"], data["tasks"]):
        key = (participant, session, task)
        if key not in assets:
            continue
        wrong = sorted(candidate for candidate in {k[0] for k in registry30.cells}
                       if candidate != participant and (candidate, session, task) in assets)[0]
        wrong_key = (wrong, session, task)
        windows = list(_natural_windows(registry30, data, key))
        y_stack = np.stack([window[1] for window in windows])
        for arm in ARM_SET:
            a0_stack, sig_stack = [], []
            for _, y, drive in windows:
                shuffled_drive = drive[:, natural_shuffle_rng.permutation(drive.shape[1])]
                a0, sig = _arm_inputs(assets, key, wrong_key, drive, shuffled_drive)[arm]
                a0_stack.append(a0)
                sig_stack.append(sig)
            output = sample_bank_eog(model, schedule, y_stack, np.stack(a0_stack),
                                     np.stack(sig_stack), device,
                                     natural_noise_seed(fold_id, seed))
            for (start, y, drive), out in zip(windows, output):
                if not np.isfinite(out).all():
                    raise FloatingPointError("nonfinite V44 natural output")
                natural_rows.append({"fold": fold_id, "seed": seed, "participant": participant,
                                     "session": session, "task": task, "start": start,
                                     "condition": arm, "oracle_non_deployable": int(arm == "ORACLE"),
                                     **_natural_metrics_full(y, drive, np.asarray(out, np.float64))})
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "checkpoint": source["checkpoint"],
        "checkpoint_best_step": source["checkpoint_best_step"],
        "noise_seed": noise_seed(fold_id, seed), "natural_noise_seed": natural_noise_seed(fold_id, seed),
        "query_eog_runtime_input": 1, "sealed_reads": 0,
        "rows": rows, "natural_rows": natural_rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "rows": len(rows),
                      "natural_rows": len(natural_rows)}))


def stage1_det(fold_id: int, seed: int, updates: int) -> None:
    """Capacity-matched DET-EOG twin: x_t := (y - a0), fixed t=0, direct MSE;
    conditioning identical to the diffusion training.  Positioning only."""
    import torch
    from torch import nn
    import torch.nn.functional as F
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG
    from eeg_scad.training.train_v42r import EMA

    result_dir = RESULT / "stage1" / f"det_fold_{fold_id}_seed_{seed}"
    result_path = result_dir / "det_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "det already complete"}))
        return
    data, folds, training = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = DERIVED / "stage1_det" / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    test_sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    test_bank = test_sampler.sample_balanced(8)
    model = CalibSADDPMEOG().to(device)

    def det_forward(net, observed_np, a0_np, signature_np, batch: int = 8):
        pieces = []
        with torch.no_grad():
            for start in range(0, len(observed_np), batch):
                observed = torch.from_numpy(np.asarray(observed_np[start:start + batch], np.float32)).to(device)
                anchor = torch.from_numpy(np.asarray(a0_np[start:start + batch], np.float32)).to(device)
                signature = torch.from_numpy(np.asarray(signature_np[start:start + batch], np.float32)).to(device)
                timestep = torch.zeros(len(observed), dtype=torch.long, device=device)
                pieces.append(net(observed - anchor, observed, anchor, timestep, signature).cpu().numpy())
        return np.concatenate(pieces)

    def det_pop_score(net) -> float:
        drives = _bank_drives(assets, validation_bank)
        a0 = np.stack([assets[(m["participant"], m["session"], m["task"])]["C0"] @ d
                       for m, d in zip(validation_bank["meta"], drives)])
        signature = np.stack([assets[(m["participant"], m["session"], m["task"])]["sig_pop"]
                              for m in validation_bank["meta"]])
        prediction = det_forward(net, validation_bank["y"], a0, signature)
        values: dict[str, list[float]] = {}
        for clean, output, meta in zip(validation_bank["x"], prediction, validation_bank["meta"]):
            values.setdefault(meta["participant"], []).append(
                float(np.linalg.norm(output - clean) / max(np.linalg.norm(clean), 1e-12)))
        return float(np.mean([np.mean(v) for v in values.values()]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model, .999)
    batch_size = training["effective_batch"]
    curve, best = [], float("inf")
    for step in range(1, updates + 1):
        bank = train_sampler.sample(batch_size)
        drives = _bank_drives(assets, bank)
        a0, signature = [], []
        for meta, drive in zip(bank["meta"], drives):
            key = (meta["participant"], meta["session"], meta["task"])
            a0.append(assets[key]["C_gated"] @ drive)
            signature.append(assets[key]["sig_gated"].copy())
        a0 = np.stack(a0)
        signature = np.stack(signature)
        a0_dropped = train_sampler.rng.random(batch_size) < .30
        a0[a0_dropped] = 0.0
        context_dropped = train_sampler.rng.random(batch_size) < .20
        for position in np.flatnonzero(context_dropped):
            meta = bank["meta"][int(position)]
            signature[position] = assets[(meta["participant"], meta["session"], meta["task"])]["sig_pop"]
        clean = torch.from_numpy(np.asarray(bank["x"], np.float32)).to(device)
        observed = torch.from_numpy(np.asarray(bank["y"], np.float32)).to(device)
        anchor = torch.from_numpy(np.asarray(a0, np.float32)).to(device)
        condition = torch.from_numpy(np.asarray(signature, np.float32)).to(device)
        timestep = torch.zeros(len(observed), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(observed - anchor, observed, anchor, timestep, condition)
        loss = F.mse_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite V44 DET loss at update {step}")
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"nonfinite V44 DET gradient at update {step}")
        optimizer.step()
        ema.update(model)
        if step % training["validation_interval"] == 0 or step == updates:
            score = det_pop_score(ema.model)
            curve.append({"step": step, "train_mse": float(loss.detach()),
                          "validation_pop_rrmse": score, "gradient_norm": float(gradient)})
            payload = {"ema": ema.state_dict(), "step": step, "seed": seed, "curve": curve,
                       "best_validation_pop_rrmse": min(best, score),
                       "architecture": "DET_EOG_twin_one_step_t0"}
            torch.save(payload, runtime / "last.pt")
            if score < best:
                best = score
                payload["best_validation_pop_rrmse"] = score
                torch.save(payload, runtime / "best.pt")
    payload = torch.load(runtime / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    model.eval()
    drives = _bank_drives(assets, test_bank)
    rows = []
    for clean, observed, artifact, meta in zip(test_bank["x"], test_bank["y"],
                                               test_bank["artifact"], test_bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                     "condition": "RAW", "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    for arm, operator_key, sig_key in (("DET_POP", "C0", "sig_pop"),
                                       ("DET_MATCH_gated", "C_gated", "sig_gated")):
        a0 = np.stack([assets[(m["participant"], m["session"], m["task"])][operator_key] @ d
                       for m, d in zip(test_bank["meta"], drives)])
        signature = np.stack([assets[(m["participant"], m["session"], m["task"])][sig_key]
                              for m in test_bank["meta"]])
        output = det_forward(model, test_bank["y"], a0, signature)
        for clean, observed, artifact, prediction, meta in zip(
                test_bank["x"], test_bank["y"], test_bank["artifact"], output, test_bank["meta"]):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V44 DET output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": arm, "zero_artifact": meta["zero_artifact"],
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


# ----------------------------------------------------------------- aggregate

def _participant_means(frame: pd.DataFrame, condition: str, metric: str = "rrmse_temporal") -> pd.Series:
    block = frame[frame.condition == condition]
    return block.groupby("participant")[metric].mean()


def aggregate_stage1() -> dict[str, Any]:
    rows, natural_rows = [], []
    for fold_id in range(5):
        for seed in S1_SEEDS:
            payload = json.loads((RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                                  / "stage1_result.json").read_text())
            rows += payload["rows"]
            natural_rows += payload["natural_rows"]
    frame = pd.DataFrame(rows)
    per = {arm: _participant_means(frame, arm) for arm in ("RAW",) + ARM_SET}
    participants = per["POP"].index

    g1 = (per["POP"] - per["MATCH_gated"]).loc[participants]
    g2_wrong_gated = (per["WRONG_gated"] - per["POP"]).loc[participants]
    g2_shuffled = (per["SHUFFLED"] - per["MATCH_gated"]).loc[participants]
    wrong_harm = (per["WRONG"] - per["POP"]).loc[participants]
    oracle_gap = (per["MATCH_gated"] - per["ORACLE"]).loc[participants]
    bridge = (per["NO_A0"] - per["POP"]).loc[participants]
    draws_g1 = bootstrap_draws(g1.to_numpy())
    draws_wg = bootstrap_draws(g2_wrong_gated.to_numpy())
    draws_sh = bootstrap_draws(g2_shuffled.to_numpy())
    g1_stat = _stat(g1)
    g1_pass = bool(g1_stat["mean"] > 0 and g1_stat["bootstrap_low"] > 0)
    wg_stat = _stat(g2_wrong_gated)
    wg_pass = bool(wg_stat["mean"] <= G2_WRONG_GATED_MARGIN)
    sh_stat = _stat(g2_shuffled)
    sh_pass = bool(sh_stat["mean"] > 0 and sh_stat["bootstrap_low"] > 0)
    p_raw = {"G1": float(np.mean(draws_g1 <= 0)),
             "G2-wrong-gated": float(np.mean(draws_wg >= G2_WRONG_GATED_MARGIN)),
             "G2-shuffled": float(np.mean(draws_sh <= 0))}
    p_adjusted = holm(p_raw)

    det_rows = []
    for fold_id in range(5):
        for seed in DET_SEEDS:
            det_rows += json.loads((RESULT / "stage1" / f"det_fold_{fold_id}_seed_{seed}"
                                    / "det_result.json").read_text())["rows"]
    det_frame = pd.DataFrame(det_rows)
    det_pop = _participant_means(det_frame, "DET_POP")
    det_match = _participant_means(det_frame, "DET_MATCH_gated")
    stage0_decision = json.loads((RESULT / "stage0" / "decision.json").read_text())
    g3 = {"wording": "competitive; no superiority claim in either direction (C05)",
          "DET_POP_mean": float(det_pop.mean()), "DET_MATCH_gated_mean": float(det_match.mean()),
          "DET_MATCH_gated_minus_DET_POP": _stat((det_match - det_pop).loc[det_pop.index]),
          "LINEAR_rows_from_S0": stage0_decision["condition_means_full_window"]}

    natural_frame = pd.DataFrame(natural_rows)
    natural_per = {}
    flags = {}
    for arm in ARM_SET:
        block = natural_frame[natural_frame.condition == arm].groupby("participant").mean(numeric_only=True)
        natural_per[arm] = block
        flags[arm] = {"attenuation_db_mean": float(block.attenuation_db.mean()),
                      "retention_mean": float(block.low_eog_observation_retention.mean()),
                      "meets_validity_bar": bool(block.attenuation_db.mean() > 0 and
                                                 block.low_eog_observation_retention.mean() >= NATURAL_RETENTION_BAR)}
    g4_valid = flags["MATCH_gated"]["meets_validity_bar"] and flags["POP"]["meets_validity_bar"]
    g4 = {"validity_bar": {"attenuation_db": "> 0", "retention": f">= {NATURAL_RETENTION_BAR}"},
          "flags": flags, "utilities_adjudicable": bool(g4_valid)}
    if g4_valid:
        common = natural_per["MATCH_gated"].index.intersection(natural_per["POP"].index)
        g4["natural_utilities_MATCH_gated_minus_POP"] = {
            metric: _stat((natural_per["MATCH_gated"].loc[common, metric]
                           - natural_per["POP"].loc[common, metric]))
            for metric in ("attenuation_db", "coherence_reduction", "low_eog_observation_retention")}
    else:
        g4["note"] = "validity bar not met; natural rows are descriptive only"

    decision = {
        "preregistration": "reports/v44_preregistration.md",
        "stage": "V44_S1_EOG_guided_diffusion",
        "information_boundary": "query EOG is a declared runtime input in this deployment class",
        "G1": {"contrast": "RRMSE(POP) - RRMSE(MATCH_gated)", **g1_stat, "pass": g1_pass},
        "G2": {"wrong_gated": {"contrast": "WRONG_gated - POP", **wg_stat,
                               "margin": G2_WRONG_GATED_MARGIN, "pass": wg_pass},
               "shuffled": {"contrast": "SHUFFLED - MATCH_gated", **sh_stat, "pass": sh_pass},
               "ungated_wrong_harm_descriptive": _stat(wrong_harm),
               "oracle_residual_gap": _stat(oracle_gap),
               "no_a0_bridge_row_descriptive": _stat(bridge)},
        "G3": g3, "G4": g4,
        "holm": {"p_raw": p_raw, "p_adjusted": p_adjusted, "alpha": 0.05},
        "condition_means": {arm: float(series.mean()) for arm, series in per.items()},
        "cells": 15, "sealed_reads": 0,
    }
    target = RESULT / "stage1"
    (target / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    arm_table = pd.DataFrame([{"condition": arm, "participant_mean_rrmse_temporal": float(series.mean())}
                              for arm, series in per.items()])
    forest = pd.DataFrame([{"participant": participant, "pop_minus_match_gated": float(value)}
                           for participant, value in g1.items()])
    natural_table = pd.DataFrame([{"condition": arm, **flags[arm]} for arm in ARM_SET])
    (REPORT / "v44_stage1.md").write_text(
        "# V44 Stage 1 — EOG-guided diffusion\n\n"
        "Query EOG is a declared runtime input in this deployment class. Any gain enters "
        "through the operator anchor; the claim is subject-aware SYSTEM gain, never score-"
        "network personalization and never diffusion superiority (C05). DET/LINEAR rows are "
        "reported wherever the diffusion arms are.\n\n"
        f"Decision: G1 **{g1_pass}**, G2-wrong-gated **{wg_pass}**, G2-shuffled **{sh_pass}**, "
        f"G4 utilities adjudicable **{bool(g4_valid)}**.\n\n"
        "## Paired arm means (temporal RRMSE, participant-first n=15)\n\n"
        + arm_table.round(6).to_markdown(index=False)
        + "\n\n## G1 per-participant forest data\n\n"
        + forest.round(6).to_markdown(index=False)
        + "\n\n## Endpoints and controls\n\n```json\n"
        + json.dumps({"G1": decision["G1"], "G2": decision["G2"], "holm": decision["holm"]},
                     indent=2, sort_keys=True) + "\n```\n\n"
        "The NO_A0 bridge row (a0 = 0) is the conditioning-class behavior inside the V44 "
        "system; its near-zero utility vs POP connects to the V43 null.\n\n"
        "## G3 positioning (descriptive; competitive, no superiority claim)\n\n```json\n"
        + json.dumps(g3, indent=2, sort_keys=True) + "\n```\n\n"
        "## G4 natural windows\n\n" + natural_table.round(6).to_markdown(index=False)
        + "\n\n```json\n" + json.dumps(g4, indent=2, sort_keys=True) + "\n```\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("stage0")
    train = sub.add_parser("stage1-train")
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--updates", type=int, required=True)
    evaluate = sub.add_parser("stage1-eval")
    evaluate.add_argument("--fold", type=int, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    det = sub.add_parser("stage1-det")
    det.add_argument("--fold", type=int, required=True)
    det.add_argument("--seed", type=int, required=True)
    det.add_argument("--updates", type=int, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "stage1-train":
        stage1_train(args.fold, args.seed, args.updates)
    elif args.stage == "stage1-eval":
        stage1_eval(args.fold, args.seed)
    elif args.stage == "stage1-det":
        stage1_det(args.fold, args.seed, args.updates)
    else:
        print(json.dumps(aggregate_stage1(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
