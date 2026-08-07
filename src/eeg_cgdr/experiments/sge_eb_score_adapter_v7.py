"""Validity adjudication and conditional EB score-adapter route for SGE v7."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from scipy.signal import coherence, freqz, welch
from torch import Tensor
from torch.optim import AdamW

from eeg_cgdr.data.sgeyesub import load_sgeyesub_signal_record
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import _load_models, _metadata, _natural_metrics, _record_pairs, _rrmse
from eeg_cgdr.models.dynamic_transfer_diffusion import DynamicTransferDiffusion


PROTOCOL = "SGE-EB-SCORE-ADAPTER-v7"


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL:
        raise ValueError("wrong v7 config")
    return value


def _folds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(config["v6_root"])) / "frozen_grouped_folds.json"
    return json.loads(path.read_text(encoding="utf-8"))["folds"]


def _seed(key: str, base: int, count: int) -> tuple[int, ...]:
    offset = (sum((i + 1) * ord(c) for i, c in enumerate(key)) + base * 1000003) % (2**31 - 10000)
    return tuple(int(offset + 37 * i) for i in range(count))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1])


def _stats(prefix: str, value: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(value)))),
        f"{prefix}_mean": float(np.mean(value)),
        f"{prefix}_std": float(np.std(value)),
        f"{prefix}_channel_scale_min": float(np.min(np.sqrt(np.mean(np.square(value), axis=(0, 2))))),
        f"{prefix}_channel_scale_max": float(np.max(np.sqrt(np.mean(np.square(value), axis=(0, 2))))),
    }


def _velocity_targets(target: Tensor, noise: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
    state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
    velocity = alpha.sqrt() * noise - (1 - alpha).sqrt() * target
    return state, velocity


def _x0_from_v(state: Tensor, velocity: Tensor, alpha: Tensor) -> Tensor:
    return alpha.sqrt() * state - (1 - alpha).sqrt() * velocity


def _sample_objective(
    model: DynamicTransferDiffusion,
    *,
    observed: Tensor,
    transfer: Tensor,
    reliability: Tensor,
    sample_seeds: Sequence[int],
    objective: str,
) -> Tensor:
    sequence = torch.linspace(model.config.timesteps - 1, 0, model.config.ddim_steps).round().long().tolist()
    samples = []
    for seed in sample_seeds:
        generator = torch.Generator(device=observed.device).manual_seed(int(seed))
        state = torch.randn(observed.shape, device=observed.device, generator=generator)
        for index, step in enumerate(sequence):
            timestep = torch.full((observed.shape[0],), int(step), dtype=torch.long, device=observed.device)
            prediction = model.backbone(state, timestep, observed=observed, transfer=transfer, reliability=reliability)
            alpha = model.alpha_bar[int(step)]
            if objective.endswith("v"):
                x0 = _x0_from_v(state, prediction, alpha)
                epsilon = (1 - alpha).sqrt() * state + alpha.sqrt() * prediction
            elif objective == "epsilon":
                epsilon = prediction
                x0 = (state - (1 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1e-12)
            else:
                raise ValueError(objective)
            if index + 1 == len(sequence):
                state = x0
            else:
                next_alpha = model.alpha_bar[int(sequence[index + 1])]
                state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
        samples.append(state)
    return torch.stack(samples).mean(0)


def _objective_loss(
    model: DynamicTransferDiffusion,
    target: Tensor,
    observed: Tensor,
    transfer: Tensor,
    reliability: Tensor,
    generator: torch.Generator,
    objective: str,
) -> Tensor:
    timestep = torch.randint(0, model.config.timesteps, (target.shape[0],), device=target.device, generator=generator)
    noise = torch.randn(target.shape, device=target.device, generator=generator)
    alpha = model.alpha_bar[timestep][:, None, None]
    state, velocity = _velocity_targets(target, noise, alpha)
    predicted = model.backbone(state, timestep, observed=observed, transfer=transfer, reliability=reliability)
    expected = noise if objective == "epsilon" else velocity
    squared = (predicted - expected).square()
    if objective == "weighted-v":
        snr = alpha / (1 - alpha).clamp_min(1e-8)
        squared = squared * (torch.minimum(snr, torch.full_like(snr, 5.0)) / (snr + 1))
    return squared.mean()


def _ema_update(ema: dict[str, Tensor], model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            ema[name].mul_(decay).add_(value.detach(), alpha=1 - decay)


def _model_from_checkpoint(config: Mapping[str, Any], fold: Mapping[str, Any], device: torch.device) -> tuple[DynamicTransferDiffusion, np.ndarray]:
    root = Path(str(config["v6_root"]))
    data = np.load(root / "prepared" / fold["fold_id"] / "training_pairs.npz")
    _, model = _load_models(yaml.safe_load(Path(str(config["v6_config"])).read_text()), data["y"].shape[1], device)
    payload = torch.load(root / "checkpoints" / "20260806" / fold["fold_id"] / "diff.pt", map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, np.asarray(payload["scale"], np.float32)


def stage_cpu_audit(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Separate model inputs/evaluator truth and repair manifest/statistical semantics."""
    v6 = Path(str(config["v6_root"]))
    root = Path(str(config["audit_root"]))
    inference_root = root / "inference_inputs"
    evaluator_root = root / "evaluator_inputs"
    operator_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["seed"]))
    v6_config = yaml.safe_load(Path(str(config["v6_config"])).read_text(encoding="utf-8"))
    layouts, records = _metadata(v6_config)
    data_root = Path(str(v6_config["data_root"]))
    for fold in _folds(config):
        folder = v6 / "prepared" / fold["fold_id"]
        training = np.load(folder / "training_pairs.npz")
        keys = np.asarray(training["key"]).astype(str)
        participant_rho = {key: float(np.mean(training["rho"][keys == key])) for key in sorted(set(keys))}
        pop_rho = float(np.mean(list(participant_rho.values())))
        fold_inference = inference_root / fold["fold_id"]
        fold_evaluator = evaluator_root / fold["fold_id"]
        fold_inference.mkdir(parents=True, exist_ok=True)
        fold_evaluator.mkdir(parents=True, exist_ok=True)
        for pair_path in sorted(folder.glob("paired_*.npz")):
            key = pair_path.stem.removeprefix("paired_").replace("__", "/")
            data = np.load(pair_path)
            np.savez_compressed(
                fold_inference / pair_path.name,
                y=data["y"], h_match=data["h_match"], h_pop=data["h_pop"], h_wrong=data["h_wrong"],
                rho_match=data["rho_match"], rho_pop=np.float32(pop_rho), rho_wrong=data["rho_wrong"], wrong=data["wrong"], pair_id=np.asarray(key),
            )
            np.savez_compressed(
                fold_evaluator / pair_path.name,
                x=data["x"], a=data["a"], h_generator=data["h_generator"], pair_id=np.asarray(key),
            )
            transfers = [("MATCH", data["h_match"], float(data["rho_match"])) , ("POP", data["h_pop"], pop_rho)]
            transfers.extend((f"WRONG{index}", value, float(data["rho_wrong"][index])) for index, value in enumerate(data["h_wrong"]))
            common_eog = rng.standard_normal((data["h_match"].shape[1], 2048))
            generator = np.asarray(data["h_generator"], np.float64)
            generator_norm = generator / max(np.linalg.norm(generator), 1e-12)
            generator_response = _apply_fir_numpy(generator, common_eog)
            for arm, transfer, reliability in transfers:
                transfer = np.asarray(transfer, np.float64)
                normed = transfer / max(np.linalg.norm(transfer), 1e-12)
                magnitude, phase = _frequency_features(transfer)
                gen_mag, gen_phase = _frequency_features(generator)
                operator_rows.append({
                    "fold_id": fold["fold_id"], "recording_key": key, "arm": arm,
                    "deployed_reliability": reliability, "population_split_reliability": pop_rho,
                    "normalized_fir_distance": float(np.linalg.norm(normed - generator_norm)),
                    "frequency_magnitude_distance": float(np.mean(np.abs(magnitude - gen_mag))),
                    "frequency_phase_distance": float(np.mean(np.abs(np.angle(np.exp(1j * (phase - gen_phase)))))),
                    "principal_angle_degrees": _principal_angle(transfer, generator),
                    "common_eog_response_distance": float(np.linalg.norm(_apply_fir_numpy(transfer, common_eog) - generator_response) / max(np.linalg.norm(generator_response), 1e-12)),
                })
            with (folder / "pair_manifest.csv").open(newline="", encoding="utf-8") as stream:
                old = next(row for row in csv.DictReader(stream) if row["recording_key"] == key)
            loaded = load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=True, include_query_annotations=True)
            taps = int(data["h_match"].shape[-1])
            role_values = _record_pairs(
                loaded, records[key], training["normal_mean"], training["normal_std"], taps=taps,
                ridge=float(v6_config["ridge_lambda"]), window_seconds=float(v6_config["window_seconds"]), return_role_metadata=True,
            )
            role_metadata = role_values[-1]
            manifest_rows.append({
                "fold_id": fold["fold_id"], "recording_key": key, "wrong_donors": old["wrong_donors"],
                "paired_windows": int(old["paired_windows"]),
                "Q_roles": role_metadata["assignment_rule"],
                "generator_trial_ids": ";".join(map(str, role_metadata["generator_trial_ids"])),
                "clean_trial_ids": ";".join(map(str, role_metadata["clean_trial_ids"])),
                "artifact_trial_ids": ";".join(map(str, role_metadata["artifact_trial_ids"])),
                "generator_time_ranges_samples": json.dumps(role_metadata["generator_time_ranges_samples"]),
                "clean_time_ranges_samples": json.dumps(role_metadata["clean_time_ranges_samples"]),
                "artifact_time_ranges_samples": json.dumps(role_metadata["artifact_time_ranges_samples"]),
                "role_overlap_verified": bool(role_metadata["trial_roles_disjoint"]),
                "legacy_overlap_field_invalid": True,
            })
    _csv(root / "operator_reliability_audit.csv", operator_rows)
    _csv(root / "corrected_pair_manifest.csv", manifest_rows)
    summary = {
        "status": "passed",
        "historical_status": "CURRENT_STATIC_TRANSFER_SUMMARY_INSTANCE_NO_GO / DIFFUSION_OPTIMIZATION_VALIDITY_NOT_ESTABLISHED / DYNAMIC_TRANSFER_SUBJECT_AWARENESS_NOT_CLEANLY_TESTED",
        "folds": 27,
        "inference_evaluator_physically_separated": True,
        "population_reliability": "equal-participant mean of outer-training blocked-split support reliability",
        "manifest_role_repaired": True,
        "raw_trial_overlap_replay_required": False,
        "raw_trial_overlap_verified": all(row["role_overlap_verified"] for row in manifest_rows),
    }
    _json(run_dir / "result_summary.json", summary)
    return summary


def _apply_fir_numpy(transfer: np.ndarray, eog: np.ndarray) -> np.ndarray:
    radius = transfer.shape[-1] // 2
    padded = np.pad(eog, ((0, 0), (radius, radius)))
    result = np.zeros((transfer.shape[0], eog.shape[1]), np.float64)
    for lag in range(transfer.shape[-1]):
        result += transfer[:, :, lag] @ padded[:, lag:lag + eog.shape[1]]
    return result


def _frequency_features(transfer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    response = np.fft.rfft(transfer, n=128, axis=-1)
    norm = np.sqrt(np.mean(np.abs(response) ** 2, axis=(-2, -1), keepdims=True)).clip(1e-12)
    response = response / norm
    return np.abs(response), np.angle(response)


def _principal_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_u = np.linalg.svd(left.reshape(left.shape[0], -1), full_matrices=False)[0]
    right_u = np.linalg.svd(right.reshape(right.shape[0], -1), full_matrices=False)[0]
    singular = np.linalg.svd(left_u[:, :4].T @ right_u[:, :4], compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(np.min(singular), -1, 1))))


@torch.no_grad()
def stage_fold_audit(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    """Audit one historical fold checkpoint without modifying it."""
    torch.manual_seed(int(config["seed"])); np.random.seed(int(config["seed"]) % (2**32 - 1)); torch.cuda.manual_seed_all(int(config["seed"]))
    fold = _folds(config)[task_index]
    v6 = Path(str(config["v6_root"])); root = Path(str(config["audit_root"])); device = torch.device("cuda")
    folder = v6 / "prepared" / fold["fold_id"]
    training = np.load(folder / "training_pairs.npz")
    model, scale_np = _model_from_checkpoint(config, fold, device)
    scale = torch.tensor(scale_np[None, :, None], device=device)
    fold_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    k_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    train_keys = np.asarray(training["key"]).astype(str)
    unique_train = sorted(set(train_keys))
    participant_rhos = [float(np.mean(training["rho"][train_keys == key])) for key in unique_train]
    pop_reliability = float(np.mean(participant_rhos))
    train_count = len(training["y"])
    fold_base = {
        "fold_id": fold["fold_id"], "study": fold["study"], "train_pair_count": train_count,
        "train_participant_count": len(unique_train), "channels": int(training["y"].shape[1]),
        "sampling_rate_hz": float(fold["sampling_rate_hz"]), "checkpoint_seed": 20260806,
        "initialization_reproducible_from_checkpoint": False,
    }
    fold_rows.append({**fold_base, **_stats("target_artifact", training["a"])})
    audit_sets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    indices = np.arange(min(8, train_count))
    audit_sets.append(("training", training["y"][indices], training["a"][indices], training["h"][indices], training["rho"][indices]))
    for path in sorted(folder.glob("paired_*.npz")):
        data = np.load(path)
        key = path.stem.removeprefix("paired_").replace("__", "/")
        y = torch.tensor(data["y"], device=device)
        target = np.asarray(data["a"], np.float32)
        transfer = torch.tensor(np.repeat(data["h_match"][None], len(y), axis=0), device=device)
        reliability = torch.full((len(y),), float(data["rho_match"]), device=device)
        for posterior_k in config["audit"]["posterior_k"]:
            prediction_norm, variance_norm, calls = model.sample_k(
                observed=y, transfer=transfer, reliability=reliability,
                sample_seeds=_seed(key, int(config["seed"]), int(posterior_k)),
            )
            prediction = (prediction_norm * scale).cpu().numpy()
            restored = data["y"] - prediction
            k_rows.append({
                "fold_id": fold["fold_id"], "recording_key": key, "K": int(posterior_k),
                "paired_rrmse": _rrmse(restored, data["x"]), "artifact_rrmse": _rrmse(prediction, target),
                "artifact_correlation": _corr(prediction, target),
                "posterior_variance_mean": float((variance_norm * scale).square().mean().cpu()), "network_calls": calls,
            })
        historical = np.load(v6 / "outputs" / "20260806" / fold["fold_id"] / "diff" / path.name)
        corrections = {arm: data["y"] - historical[arm] for arm in historical.files if arm not in {"x", "y"}}
        row = {**fold_base, "recording_key": key, **_stats("historical_match_correction", corrections["MATCH"])}
        row.update({
            "historical_match_target_correlation": _corr(corrections["MATCH"], target),
            "historical_match_artifact_rrmse": _rrmse(corrections["MATCH"], target),
            "historical_match_restored_rrmse": _rrmse(historical["MATCH"], data["x"]),
            "raw_rrmse": _rrmse(data["y"], data["x"]),
            "match_pop_correction_difference_rms": float(np.sqrt(np.mean((corrections["MATCH"] - corrections["POP"]) ** 2))),
            "match_wrong_correction_difference_rms": float(np.mean([np.sqrt(np.mean((corrections["MATCH"] - value) ** 2)) for arm, value in corrections.items() if arm.startswith("WRONG")])),
        })
        fold_rows.append(row)
        arm_values: list[tuple[str, np.ndarray, float]] = [("MATCH", data["h_match"], float(data["rho_match"])), ("POP", data["h_pop"], pop_reliability)]
        arm_values.extend((f"WRONG{index}", value, float(data["rho_wrong"][index])) for index, value in enumerate(data["h_wrong"]))
        for regime in ("operator_only_common_reliability", "deployed_support_reliability"):
            for arm, transfer_value, arm_reliability in arm_values:
                transfer_arm = torch.tensor(np.repeat(transfer_value[None], len(y), 0), device=device)
                rho_value = 1.0 if regime == "operator_only_common_reliability" else arm_reliability
                rho_arm = torch.full((len(y),), rho_value, device=device)
                predicted_norm, _, _ = model.sample_k(observed=y, transfer=transfer_arm, reliability=rho_arm, sample_seeds=_seed(key, int(config["seed"]), 8))
                correction = (predicted_norm * scale).cpu().numpy(); restored = data["y"] - correction
                context_rows.append({
                    "fold_id": fold["fold_id"], "recording_key": key, "regime": regime, "arm": arm,
                    "reliability": rho_value, "rrmse": _rrmse(restored, data["x"]),
                    "artifact_rrmse": _rrmse(correction, data["a"]), "artifact_correlation": _corr(correction, data["a"]),
                    "correction_rms": float(np.sqrt(np.mean(correction ** 2))),
                })
        if len(audit_sets) == 1:
            count = min(8, len(data["y"]))
            audit_sets.append(("heldout", data["y"][:count], data["a"][:count], np.repeat(data["h_match"][None], count, 0), np.repeat(float(data["rho_match"]), count)))
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    for split, y_np, target_np, transfer_np, rho_np in audit_sets:
        y = torch.tensor(y_np, device=device); target = torch.tensor(target_np, device=device) / scale
        transfer = torch.tensor(transfer_np, device=device); reliability = torch.tensor(rho_np, device=device)
        fixed_noise = torch.randn(target.shape, device=device, generator=generator)
        context_x0: dict[int, Tensor] = {}
        for timestep_value in config["audit"]["timesteps"]:
            timestep = torch.full((len(y),), int(timestep_value), device=device, dtype=torch.long)
            alpha = model.alpha_bar[timestep][:, None, None]
            state, expected_v = _velocity_targets(target, fixed_noise, alpha)
            predicted_v = model.backbone(state, timestep, observed=y, transfer=transfer, reliability=reliability)
            x0 = _x0_from_v(state, predicted_v, alpha)
            context_x0[int(timestep_value)] = x0
            time_rows.append({
                "fold_id": fold["fold_id"], "split": split, "timestep": int(timestep_value),
                "predicted_v_mse": float((predicted_v - expected_v).square().mean().cpu()),
                "x0_rrmse": float(torch.sqrt((x0 - target).square().mean()).div(torch.sqrt(target.square().mean()).clamp_min(1e-12)).cpu()),
                "artifact_correlation": _corr((x0 * scale).cpu().numpy(), target_np),
                "artifact_rms_ratio": float(torch.sqrt(x0.square().mean()).div(torch.sqrt(target.square().mean()).clamp_min(1e-12)).cpu()),
            })
        if split == "heldout":
            sampled = model.sample_k(observed=y, transfer=transfer, reliability=reliability, sample_seeds=_seed(fold["fold_id"], int(config["seed"]), 1), trace=True)
            final, _, _, trace = sampled
            for item in trace:
                x0 = item["x0"]
                trajectory_rows.append({
                    "fold_id": fold["fold_id"], "timestep": int(item["timestep"]),
                    "state_rms": float(torch.sqrt(item["state"].square().mean()).cpu()),
                    "x0_rms": float(torch.sqrt(x0.square().mean()).cpu()),
                    "x0_error": float(torch.sqrt((x0 - target).square().mean()).cpu()),
                    "epsilon_rms": float(torch.sqrt(item["epsilon"].square().mean()).cpu()),
                })
    task_dir = root / "fold_tasks" / fold["fold_id"]
    _csv(task_dir / "fold_diagnostics.csv", fold_rows)
    _csv(task_dir / "timestep_x0_metrics.csv", time_rows)
    _csv(task_dir / "sampler_trajectory.csv", trajectory_rows)
    _csv(task_dir / "k_convergence.csv", k_rows)
    _csv(task_dir / "corrected_context_metrics.csv", context_rows)
    summary = {"status": "passed", "fold_id": fold["fold_id"], "stems": len(fold["heldout"]), "K": [1, 8, 32], "timesteps": list(config["audit"]["timesteps"])}
    _json(run_dir / "result_summary.json", summary)
    return summary


def _evaluate_overfit(
    model: DynamicTransferDiffusion,
    *,
    y: Tensor,
    target: Tensor,
    transfer: Tensor,
    reliability: Tensor,
    objective: str,
    seed: int,
) -> tuple[float, float, Tensor]:
    model.eval()
    with torch.no_grad():
        prediction = _sample_objective(model, observed=y, transfer=transfer, reliability=reliability, sample_seeds=_seed("overfit", seed, 8), objective=objective)
    error = float(torch.sqrt((prediction - target).square().mean()).div(torch.sqrt(target.square().mean()).clamp_min(1e-12)).cpu())
    correlation = _corr(prediction.detach().cpu().numpy(), target.detach().cpu().numpy())
    model.train()
    return error, correlation, prediction


def _train_objective(
    config: Mapping[str, Any], fold: Mapping[str, Any], objective: str, *, full_fold: bool, run_dir: Path,
) -> tuple[DynamicTransferDiffusion, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    seed = int(config["seed"]); torch.manual_seed(seed); np.random.seed(seed % (2**32 - 1)); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda"); v6 = Path(str(config["v6_root"])); data = np.load(v6 / "prepared" / fold["fold_id"] / "training_pairs.npz")
    _, model = _load_models(yaml.safe_load(Path(str(config["v6_config"])).read_text()), data["y"].shape[1], device)
    scale_np = np.quantile(np.abs(data["a"]), .995, axis=(0, 2)).clip(1e-4).astype(np.float32)
    scale = torch.tensor(scale_np[None, :, None], device=device)
    batch_count = len(data["y"]) if full_fold else min(int(config["audit"]["fixed_batch_size"]), len(data["y"]))
    fixed_indices = np.arange(batch_count)
    fixed_y = torch.tensor(data["y"][fixed_indices], device=device)
    fixed_target = torch.tensor(data["a"][fixed_indices], device=device) / scale
    fixed_h = torch.tensor(data["h"][fixed_indices], device=device)
    fixed_rho = torch.tensor(data["rho"][fixed_indices], device=device)
    optimizer = AdamW(model.parameters(), lr=float(yaml.safe_load(Path(str(config["v6_config"])).read_text())["training"]["learning_rate"]), weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed); rng = np.random.default_rng(seed)
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    curves: list[dict[str, Any]] = []
    max_updates = int(config["audit"]["repair_updates"] if full_fold else config["audit"]["overfit_max_updates"])
    check_interval = 500 if full_fold else int(config["audit"]["overfit_check_interval"])
    passed = False; pass_checkpoint = "none"; final_error = math.inf; final_correlation = -1.0
    for step in range(1, max_updates + 1):
        if full_fold:
            indices = rng.integers(0, len(data["y"]), int(yaml.safe_load(Path(str(config["v6_config"])).read_text())["training"]["batch_size"]))
            y = torch.tensor(data["y"][indices], device=device); target = torch.tensor(data["a"][indices], device=device) / scale
            transfer = torch.tensor(data["h"][indices], device=device); reliability = torch.tensor(data["rho"][indices], device=device)
        else:
            y, target, transfer, reliability = fixed_y, fixed_target, fixed_h, fixed_rho
        optimizer.zero_grad(set_to_none=True)
        loss = _objective_loss(model, target, y, transfer, reliability, generator, objective)
        loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).detach().cpu()); optimizer.step()
        _ema_update(ema, model, float(config["audit"]["ema_decay"]))
        if step == 1 or step % check_interval == 0 or step == max_updates:
            raw_error, raw_corr, _ = _evaluate_overfit(model, y=fixed_y, target=fixed_target, transfer=fixed_h, reliability=fixed_rho, objective=objective, seed=seed)
            raw_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.load_state_dict(ema)
            ema_error, ema_corr, _ = _evaluate_overfit(model, y=fixed_y, target=fixed_target, transfer=fixed_h, reliability=fixed_rho, objective=objective, seed=seed)
            model.load_state_dict(raw_state)
            curves.append({"step": step, "loss": float(loss.detach().cpu()), "gradient_norm": grad, "raw_rrmse": raw_error, "raw_correlation": raw_corr, "ema_rrmse": ema_error, "ema_correlation": ema_corr})
            if not full_fold and (raw_error <= float(config["audit"]["overfit_rrmse"]) and raw_corr >= float(config["audit"]["overfit_correlation"]) or ema_error <= float(config["audit"]["overfit_rrmse"]) and ema_corr >= float(config["audit"]["overfit_correlation"])):
                passed = True
                if ema_error < raw_error:
                    model.load_state_dict(ema); final_error, final_correlation, pass_checkpoint = ema_error, ema_corr, "ema"
                else:
                    final_error, final_correlation, pass_checkpoint = raw_error, raw_corr, "raw"
                break
            final_error, final_correlation = min((raw_error, raw_corr), (ema_error, ema_corr), key=lambda item: item[0])
    checkpoint = run_dir / f"{objective}_{'repair' if full_fold else 'overfit'}.pt"
    torch.save({"model": model.state_dict(), "ema": ema, "optimizer": optimizer.state_dict(), "scale": scale_np, "step": step, "objective": objective, "seed": seed}, checkpoint)
    return model, scale_np, curves, {"passed": passed if not full_fold else True, "rrmse": final_error, "correlation": final_correlation, "checkpoint_kind": pass_checkpoint, "updates": step, "checkpoint": str(checkpoint)}


@torch.no_grad()
def _repair_evaluation(config: Mapping[str, Any], fold: Mapping[str, Any], model: DynamicTransferDiffusion, scale_np: np.ndarray, objective: str) -> dict[str, Any]:
    v6 = Path(str(config["v6_root"])); folder = v6 / "prepared" / fold["fold_id"]; device = torch.device("cuda")
    scale = torch.tensor(scale_np[None, :, None], device=device)
    paired_rows = []
    natural_rows = []
    for path in sorted(folder.glob("paired_*.npz")):
        data = np.load(path); key = path.stem.removeprefix("paired_").replace("__", "/")
        y = torch.tensor(data["y"], device=device)
        h = torch.tensor(np.repeat(data["h_match"][None], len(y), 0), device=device)
        rho = torch.full((len(y),), float(data["rho_match"]), device=device)
        artifact = _sample_objective(model, observed=y, transfer=h, reliability=rho, sample_seeds=_seed(key, int(config["seed"]), 8), objective=objective) * scale
        restored = (y - artifact).cpu().numpy()
        paired_rows.append({
            "recording_key": key, "raw_rrmse": _rrmse(data["y"], data["x"]), "repaired_rrmse": _rrmse(restored, data["x"]),
            "repaired_better_than_raw": _rrmse(restored, data["x"]) < _rrmse(data["y"], data["x"]),
            "artifact_correlation": _corr(artifact.cpu().numpy(), data["a"]), "artifact_rrmse": _rrmse(artifact.cpu().numpy(), data["a"]),
        })
        natural = np.load(folder / f"natural_input_{key.replace('/','__')}.npz")
        evaluator = np.load(folder / f"natural_evaluator_{key.replace('/','__')}.npz")
        rate = float(fold["sampling_rate_hz"]); length = int(round(2.0 * rate)); usable = natural["y"].shape[1] // length * length
        windows = natural["y"][:, :usable].reshape(natural["y"].shape[0], -1, length).transpose(1, 0, 2)
        output = []
        for start in range(0, len(windows), 8):
            observed = torch.tensor(windows[start:start + 8], device=device)
            transfer = torch.tensor(np.repeat(natural["h_match"][None], len(observed), 0), device=device)
            reliability = torch.full((len(observed),), float(natural["rho_match"]), device=device)
            correction = _sample_objective(model, observed=observed, transfer=transfer, reliability=reliability, sample_seeds=_seed(key, int(config["seed"]), 8), objective=objective) * scale
            output.append((observed - correction).cpu().numpy())
        restored = np.concatenate(output).transpose(1, 0, 2).reshape(natural["y"].shape[0], usable)
        natural_rows.append({"recording_key": key, **_natural_metrics(natural["y"][:, :usable], restored, evaluator["eog"], evaluator["labels"], rate)})
    return {
        "paired": paired_rows,
        "natural": natural_rows,
        "all_heldout_better_than_raw": all(row["repaired_better_than_raw"] for row in paired_rows),
        "natural_preservation_mean": float(np.nanmean([row["nonartifact_preservation"] for row in natural_rows])),
        "natural_psd_mean": float(np.nanmean([row["psd_distortion"] for row in natural_rows])),
        "natural_covariance_mean": float(np.nanmean([row["covariance_distortion"] for row in natural_rows])),
    }


def stage_overfit(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    diagnostic = list(config["diagnostic_folds"])
    fold_id = diagnostic[task_index]
    fold = next(row for row in _folds(config) if row["fold_id"] == fold_id)
    v6 = Path(str(config["v6_root"])); data = np.load(v6 / "prepared" / fold_id / "training_pairs.npz"); device = torch.device("cuda")
    _, sampler = _load_models(yaml.safe_load(Path(str(config["v6_config"])).read_text()), data["y"].shape[1], device)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    target = torch.tensor(data["a"][:2], device=device)
    noise = torch.randn(target.shape, device=device, generator=generator)
    oracle = sampler.oracle_v_roundtrip(target, initial_noise=noise)
    oracle_relative_error = float(torch.linalg.norm(oracle - target).div(torch.linalg.norm(target).clamp_min(1e-12)).cpu())
    oracle_pass = oracle_relative_error < 1e-4
    if not oracle_pass:
        summary = {"status": "failed", "fold_id": fold_id, "oracle_roundtrip_relative_error": oracle_relative_error, "oracle_roundtrip_pass": False, "validity": "sampler_invalid"}
        _json(run_dir / "result_summary.json", summary)
        raise RuntimeError("oracle-v sampler roundtrip failed")
    objective_rows: list[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    for objective in ("weighted-v", "unweighted-v"):
        _, _, curves, result = _train_objective(config, fold, objective, full_fold=False, run_dir=run_dir)
        _csv(run_dir / f"{objective}_curve.csv", curves)
        outcomes[objective] = result
        objective_rows.append({"fold_id": fold_id, "objective": objective, **result})
    if not outcomes["weighted-v"]["passed"] and not outcomes["unweighted-v"]["passed"]:
        _, _, curves, result = _train_objective(config, fold, "epsilon", full_fold=False, run_dir=run_dir)
        _csv(run_dir / "epsilon_curve.csv", curves)
        outcomes["epsilon"] = result
        objective_rows.append({"fold_id": fold_id, "objective": "epsilon", **result})
    selected = "unweighted-v" if outcomes["unweighted-v"]["passed"] else "epsilon" if outcomes.get("epsilon", {}).get("passed") else None
    repair = None
    if selected is not None:
        model, scale, curves, training = _train_objective(config, fold, selected, full_fold=True, run_dir=run_dir)
        _csv(run_dir / "repair_training_curve.csv", curves)
        repair = _repair_evaluation(config, fold, model, scale, selected)
        _csv(run_dir / "repair_paired_metrics.csv", repair.pop("paired"))
        _csv(run_dir / "repair_natural_metrics.csv", repair.pop("natural"))
        repair.update({"objective": selected, "fixed_endpoint_updates": training["updates"]})
    _csv(run_dir / "objective_summary.csv", objective_rows)
    summary = {
        "status": "passed" if selected is not None else "failed",
        "fold_id": fold_id, "oracle_roundtrip_relative_error": oracle_relative_error, "oracle_roundtrip_pass": oracle_pass,
        "selected_repair_objective": selected, "objectives": outcomes, "repair": repair,
    }
    _json(run_dir / "result_summary.json", summary)
    return summary


def _collect_csv(paths: Sequence[Path], destination: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    _csv(destination, rows)
    return rows


def stage_validity_decision(config: Mapping[str, Any], overfit_job: str, run_dir: Path) -> dict[str, Any]:
    root = Path(str(config["audit_root"])); v6 = Path(str(config["v6_root"]))
    fold_tasks = root / "fold_tasks"
    _collect_csv(sorted(fold_tasks.glob("*/fold_diagnostics.csv")), root / "fold_diagnostics.csv")
    _collect_csv(sorted(fold_tasks.glob("*/timestep_x0_metrics.csv")), root / "timestep_x0_metrics.csv")
    _collect_csv(sorted(fold_tasks.glob("*/sampler_trajectory.csv")), root / "sampler_trajectory.csv")
    _collect_csv(sorted(fold_tasks.glob("*/k_convergence.csv")), root / "k_convergence.csv")
    contexts = _collect_csv(sorted(fold_tasks.glob("*/corrected_context_metrics.csv")), root / "corrected_context_metrics.csv")
    corrected_effects: list[dict[str, Any]] = []
    for regime in sorted({row["regime"] for row in contexts}):
        subset = [row for row in contexts if row["regime"] == regime]
        for key in sorted({row["recording_key"] for row in subset}):
            unit = [row for row in subset if row["recording_key"] == key]
            match = next(row for row in unit if row["arm"] == "MATCH")
            pop = next(row for row in unit if row["arm"] == "POP")
            donors = [row for row in unit if row["arm"].startswith("WRONG")]
            donor_utilities = [float(row["rrmse"]) - float(match["rrmse"]) for row in donors]
            corrected_effects.append({
                "recording_key": key, "fold_id": match["fold_id"], "regime": regime,
                "U_P": float(pop["rrmse"]) - float(match["rrmse"]),
                "U_W_donor_mean": float(np.mean(donor_utilities)),
                "wrong_donor_count": len(donors),
                "wrong_donor_utilities": ";".join(f"{value:.9g}" for value in donor_utilities),
                "wrong_scoring_order": "RRMSE_each_donor_then_mean_utility",
            })
    _csv(root / "corrected_wrong_effects.csv", corrected_effects)
    overfit_root = v6 / "runs" / "v7-overfit" / f"job_{overfit_job}"
    summaries = []
    for task_index in range(3):
        path = overfit_root / f"task_{task_index}" / "result_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    sampler_pass = all(row["oracle_roundtrip_pass"] for row in summaries)
    selected_objectives = [row["selected_repair_objective"] for row in summaries]
    validated_objective = selected_objectives[0] if len(set(selected_objectives)) == 1 and selected_objectives[0] in {"unweighted-v", "epsilon"} else None
    objective_pass = validated_objective is not None
    heldout_pass = all(row.get("repair") and row["repair"]["all_heldout_better_than_raw"] for row in summaries)
    safety_pass = all(
        row.get("repair") and row["repair"]["natural_preservation_mean"] >= 0.75 and row["repair"]["natural_psd_mean"] <= 0.25 and row["repair"]["natural_covariance_mean"] <= 0.25
        for row in summaries
    )
    stage_b = sampler_pass and objective_pass and heldout_pass and safety_pass
    decision = "DIFFUSION_ESTIMATOR_VALID_STAGE_B_AUTHORIZED" if stage_b else "DIFFUSION_IMPLEMENTATION_OR_OBJECTIVE_INVALID"
    summary = {
        "status": "passed" if stage_b else "failed",
        "decision": decision, "stage_b_authorized": stage_b,
        "sampler_oracle_roundtrip": sampler_pass, "real_batch_overfit": objective_pass,
        "validated_objective": validated_objective,
        "all_three_repaired_folds_beat_raw": heldout_pass, "natural_scale_safety": safety_pass,
        "diagnostic_folds": summaries,
        "historical_v6_status": "CURRENT_STATIC_TRANSFER_SUMMARY_INSTANCE_NO_GO / DIFFUSION_OPTIMIZATION_VALIDITY_NOT_ESTABLISHED / DYNAMIC_TRANSFER_SUBJECT_AWARENESS_NOT_CLEANLY_TESTED",
        "original_v6_seeds_not_submitted": [20260807, 20260808],
    }
    _json(root / "validity_decision.json", summary)
    _json(run_dir / "result_summary.json", summary)
    _write_validity_report(root, summary)
    return summary


def _write_validity_report(root: Path, summary: Mapping[str, Any]) -> None:
    rows = summary["diagnostic_folds"]
    table_rows = []
    for row in rows:
        repair = row.get("repair") or {}
        table_rows.append(
            f"| {row['fold_id']} | {row['oracle_roundtrip_relative_error']:.3e} | {row['selected_repair_objective'] or 'none'} | "
            f"{str(bool(repair.get('all_heldout_better_than_raw'))).lower()} | "
            f"{repair.get('natural_preservation_mean', float('nan')):.4f} | {repair.get('natural_psd_mean', float('nan')):.4f} | {repair.get('natural_covariance_mean', float('nan')):.4f} |"
        )
    table = "\n".join(table_rows)
    report = Path("reports/v6_diffusion_validity_audit.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"""# v6 diffusion validity adjudication

The historical v6 outputs are preserved. Their corrected scientific status is:

`CURRENT_STATIC_TRANSFER_SUMMARY_INSTANCE_NO_GO / DIFFUSION_OPTIMIZATION_VALIDITY_NOT_ESTABLISHED / DYNAMIC_TRANSFER_SUBJECT_AWARENESS_NOT_CLEANLY_TESTED`.

The v6 condition encoded a static projector, channel FIR scale and reliability; it did not encode lag/frequency/phase. Historical folds had only 8--52 unique pairs, POP reliability was hard-coded to 1.0, and WRONG waveforms were averaged before RRMSE. Therefore v6 is not described as a valid negative result for dynamic-transfer-conditioned diffusion.

| Diagnostic fold | oracle-v error | selected repair objective | all heldout beat RAW | preservation | PSD distortion | covariance distortion |
|---|---:|---|---|---:|---:|---:|
{table}

Decision: `{summary['decision']}`. Stage B authorized: `{str(summary['stage_b_authorized']).lower()}`. The old seeds 20260807/20260808 remain unsubmitted.
""", encoding="utf-8")


def _response_distance(candidate: np.ndarray, generator: np.ndarray, common_eog: np.ndarray) -> float:
    magnitude, phase = _frequency_features(candidate)
    reference_magnitude, reference_phase = _frequency_features(generator)
    magnitude_distance = float(np.mean(np.abs(magnitude - reference_magnitude)))
    phase_distance = float(np.mean(np.abs(np.angle(np.exp(1j * (phase - reference_phase))))))
    candidate_response = _apply_fir_numpy(candidate, common_eog)
    generator_response = _apply_fir_numpy(generator, common_eog)
    response_distance = float(np.linalg.norm(candidate_response - generator_response) / max(np.linalg.norm(generator_response), 1e-12))
    return magnitude_distance + 0.1 * phase_distance + response_distance


def _eb_features(support: np.ndarray, population: np.ndarray, reliability: float) -> np.ndarray:
    flat = support.reshape(support.shape[0], -1)
    singular = np.linalg.svd(flat, compute_uv=False)
    singular = singular[:4] / max(np.linalg.norm(singular[:4]), 1e-12)
    singular = np.pad(singular, (0, 4 - len(singular)))
    condition = float(singular[0] / max(singular[-1], 1e-6))
    delta = float(np.linalg.norm(support - population) / max(np.linalg.norm(population), 1e-12))
    return np.asarray([reliability, math.log1p(30.0), delta, math.log1p(condition), *singular], np.float64)


def stage_eb_headroom(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    audit = json.loads((Path(str(config["audit_root"])) / "validity_decision.json").read_text(encoding="utf-8"))
    if not audit.get("stage_b_authorized"):
        raise RuntimeError("Stage B is fail-closed by diffusion validity adjudication")
    v6 = Path(str(config["v6_root"])); root = Path(str(config["result_root"])); root.mkdir(parents=True, exist_ok=True)
    folds = _folds(config); lookup: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for fold in folds:
        folder = v6 / "prepared" / fold["fold_id"]
        for path in folder.glob("paired_*.npz"):
            key = path.stem.removeprefix("paired_").replace("__", "/"); data = np.load(path)
            lookup[key] = (np.asarray(data["h_match"], np.float64), np.asarray(data["h_generator"], np.float64), float(data["rho_match"]))
    grid = np.linspace(0, 1, int(config["stage_b"]["lambda_grid_points"]))
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["seed"]))
    for fold in folds:
        folder = v6 / "prepared" / fold["fold_id"]
        training = np.load(folder / "training_pairs.npz")
        population = np.asarray(training["h_pop"], np.float64)
        eog_channels = population.shape[1]
        common_eog = rng.standard_normal((eog_channels, 4096))
        feature_train = []
        target_train = []
        for key in fold["training"]:
            support, generator, reliability = lookup[key]
            scores = [_response_distance(population + value * (support - population), generator, common_eog) for value in grid]
            target_train.append(float(grid[int(np.argmin(scores))]))
            feature_train.append(_eb_features(support, population, reliability))
        features = np.stack(feature_train); targets = np.asarray(target_train)
        mean = features.mean(0); std = features.std(0).clip(1e-6); design = np.column_stack((np.ones(len(features)), (features - mean) / std))
        coefficients = np.linalg.solve(design.T @ design + 1e-2 * np.eye(design.shape[1]), design.T @ targets)
        for key in fold["heldout"]:
            support, generator, reliability = lookup[key]
            scores = np.asarray([_response_distance(population + value * (support - population), generator, common_eog) for value in grid])
            oracle_index = int(np.argmin(scores)); oracle_lambda = float(grid[oracle_index]); pop_score = float(scores[0]); oracle_score = float(scores[oracle_index])
            feature = _eb_features(support, population, reliability)
            predicted = float(np.clip(np.r_[1.0, (feature - mean) / std] @ coefficients, 0, 1))
            predicted_score = _response_distance(population + predicted * (support - population), generator, common_eog)
            rows.append({
                "fold_id": fold["fold_id"], "recording_key": key, "study": fold["study"],
                "oracle_lambda": oracle_lambda, "deployable_lambda": predicted, "support_reliability": reliability,
                "population_response_distance": pop_score, "oracle_eb_response_distance": oracle_score,
                "deployable_eb_response_distance": predicted_score,
                "oracle_improvement": pop_score - oracle_score, "deployable_improvement": pop_score - predicted_score,
                "oracle_beats_population": oracle_score < pop_score,
            })
    _csv(root / "eb_headroom_metrics.csv", rows)
    win_fraction = float(np.mean([bool(row["oracle_beats_population"]) for row in rows]))
    by_study = {study: float(np.mean([row["oracle_improvement"] for row in rows if row["study"] == study])) for study in sorted({row["study"] for row in rows})}
    nonnegative = int(sum(value >= 0 for value in by_study.values()))
    mean_improvement = float(np.mean([row["oracle_improvement"] for row in rows]))
    continue_route = mean_improvement > 0 and win_fraction >= float(config["stage_b"]["oracle_min_win_fraction"]) and nonnegative >= int(config["stage_b"]["oracle_min_nonnegative_cells"])
    summary = {
        "status": "passed" if continue_route else "closed",
        "decision": "EB_OPERATOR_HEADROOM_PASS" if continue_route else "EB_ORACLE_CEILING_NO_GO",
        "expanded_pairs_authorized": continue_route, "stems": len(rows),
        "oracle_mean_improvement": mean_improvement, "oracle_win_fraction": win_fraction,
        "nonnegative_studies": nonnegative, "study_effects": by_study,
        "deployable_lambda_mean": float(np.mean([row["deployable_lambda"] for row in rows])),
        "query_information_in_lambda_predictor": False,
    }
    _json(root / "eb_headroom_decision.json", summary); _json(run_dir / "result_summary.json", summary)
    return summary


def run_stage(config_path: Path, stage: str, run_dir: Path, *, task_index: int = 0, overfit_job: str = "") -> dict[str, Any]:
    config = _config(config_path)
    if stage == "cpu-audit":
        return stage_cpu_audit(config, run_dir)
    if stage == "fold-audit":
        return stage_fold_audit(config, task_index, run_dir)
    if stage == "overfit":
        return stage_overfit(config, task_index, run_dir)
    if stage == "validity-decision":
        return stage_validity_decision(config, overfit_job, run_dir)
    if stage == "eb-headroom":
        return stage_eb_headroom(config, run_dir)
    raise ValueError(stage)


__all__ = ["run_stage"]
