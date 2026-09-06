"""Joint official-semantics POP/Calib-EEGDfus training and paired replay."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from scipy import signal

from eeg_scad.data.artifact_transfer_v41r import (
    TransferEpisodeSampler, TransferRegistry, flatten_channels, flatten_signatures, reassemble_channels,
)
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.calib_eegdfus_v41r import CalibEEGDfus, OfficialLinearSchedule, ancestral_sample


CONDITIONS = ("POP", "MATCH", "WRONG", "SHUFFLED", "ORACLE", "CHANNEL_ONLY")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(np.asarray(value, np.float32)).to(device)


def _episode_signatures(sampler: TransferEpisodeSampler, meta: list[dict[str, Any]], condition: str) -> tuple[np.ndarray, list[str]]:
    values, owners = [], []
    for row in meta:
        signature, owner = sampler.condition_signature(row, condition)
        values.append(signature)
        owners.append(owner)
    return np.stack(values), owners


def _validation_score(model: CalibEEGDfus, schedule: OfficialLinearSchedule, bank: Mapping[str, Any],
                      device: torch.device, seed: int, chunk: int = 184) -> float:
    model.eval()
    clean = flatten_channels(bank["x"])
    observed = flatten_channels(bank["y"])
    transfer = flatten_signatures(bank["signature"])
    generator = torch.Generator(device=device).manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(clean), chunk):
            x, y, c = _tensor(clean[start:start + chunk], device), _tensor(observed[start:start + chunk], device), _tensor(transfer[start:start + chunk], device)
            noisy, noise, level = schedule.training_sample(x, generator)
            loss = torch.nn.functional.l1_loss(model(noisy, y, level, c), noise, reduction="sum")
            total += float(loss)
            count += noise.numel()
    model.train()
    return total / max(count, 1)


def train_model(model: CalibEEGDfus, schedule: OfficialLinearSchedule, train_sampler: TransferEpisodeSampler,
                validation_bank: Mapping[str, Any], device: torch.device, seed: int, updates: int,
                batch_episodes: int, validation_interval: int, checkpoint: Path) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    curve, best, best_step = [], float("inf"), 0
    model.train()
    generator = torch.Generator(device=device).manual_seed(seed + 711)
    for step in range(1, updates + 1):
        bank = train_sampler.sample(batch_episodes)
        signature = bank["signature"].copy()
        # Episode-level context dropout trains the exact population signature in the same model.
        dropped = train_sampler.rng.random(batch_episodes) < 0.20
        for index in np.flatnonzero(dropped):
            row = bank["meta"][int(index)]
            signature[index], _ = train_sampler.condition_signature(row, "POP")
        x = _tensor(flatten_channels(bank["x"]), device)
        y = _tensor(flatten_channels(bank["y"]), device)
        condition = _tensor(flatten_signatures(signature), device)
        noisy, noise, level = schedule.training_sample(x, generator)
        predicted = model(noisy, y, level, condition)
        loss = torch.nn.functional.l1_loss(predicted, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite V41R loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gradient):
            raise FloatingPointError(f"nonfinite V41R gradient at update {step}")
        optimizer.step()
        if step % validation_interval == 0 or step == updates:
            score = _validation_score(model, schedule, validation_bank, device, seed + 1701)
            transfer_grad = float(sum((parameter.grad.detach().norm() for name, parameter in model.named_parameters()
                                       if "transfer_" in name and parameter.grad is not None), torch.tensor(0.0, device=device)))
            curve.append({"step": step, "train_epsilon_l1": float(loss.detach()),
                          "validation_epsilon_l1": score, "gradient_norm": float(gradient),
                          "transfer_gradient_norm": transfer_grad,
                          "context_dropout_fraction": float(dropped.mean())})
            if score < best:
                best, best_step = score, step
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "step": step, "best_validation_epsilon_l1": score,
                    "seed": seed, "train_sampler_rng": train_sampler.rng.bit_generator.state,
                    "torch_generator_state": generator.get_state().cpu(),
                    "contract": {"channels": 1, "samples": 512, "eog_regressors": 2,
                                 "prediction": "epsilon", "T": 500, "context_dropout": 0.20},
                }, checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError("validation did not create a checkpoint")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if int(payload["step"]) != best_step:
        raise RuntimeError("checkpoint selection mismatch")
    return curve


@torch.no_grad()
def _sample_clean(model: CalibEEGDfus, schedule: OfficialLinearSchedule, observed: np.ndarray,
                  signatures: np.ndarray, device: torch.device, seed: int, episode_batch: int = 3) -> np.ndarray:
    model.eval()
    outputs = []
    for episode_start in range(0, len(observed), episode_batch):
        episode_stop = min(len(observed), episode_start + episode_batch)
        flat_y = flatten_channels(observed[episode_start:episode_stop])
        flat_c = flatten_signatures(signatures[episode_start:episode_stop])
        # The seed depends only on the frozen query batch, not the intervention.
        output = ancestral_sample(model, _tensor(flat_y, device), _tensor(flat_c, device),
                                  seed + episode_start * 1009, schedule)
        outputs.append(reassemble_channels(output.cpu().numpy(), episode_stop - episode_start))
    return np.concatenate(outputs)


def evaluate_paired(model: CalibEEGDfus, schedule: OfficialLinearSchedule, bank: Mapping[str, Any],
                    sampler: TransferEpisodeSampler, device: torch.device, fold: int, seed: int,
                    conditions: tuple[str, ...] = CONDITIONS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, channel_rows = [], []
    for x, y, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
        values = paired_metrics(x, y, artifact, np.zeros_like(artifact))
        rows.append({"fold": fold, "seed": seed, "participant": meta["participant"], "session": meta["session"],
                     "task": meta["task"], "condition": "RAW", "context_owner": "NONE",
                     "oracle_non_deployable": 0, "gain": meta["gain"],
                     "severity": "zero" if meta["zero_artifact"] else ("mild" if meta["gain"] < 0.55 else "medium" if meta["gain"] < 0.95 else "severe"),
                     "zero_artifact": meta["zero_artifact"], "output_input_rms": 1.0,
                     "query_eog_inference_reads": 0, "query_transfer_in_condition": 0, **values})
    for condition in conditions:
        signatures, context_owners = _episode_signatures(sampler, bank["meta"], condition)
        clean = _sample_clean(model, schedule, bank["y"], signatures, device, 410000 + fold * 100 + seed % 100)
        for index, (x, y, artifact, predicted, meta) in enumerate(zip(bank["x"], bank["y"], bank["artifact"], clean, bank["meta"])):
            if not np.isfinite(predicted).all():
                raise FloatingPointError("nonfinite V41R paired output")
            predicted_artifact = y - predicted
            values = paired_metrics(x, y, artifact, predicted_artifact)
            rows.append({
                "fold": fold, "seed": seed, "participant": meta["participant"], "session": meta["session"],
                "task": meta["task"], "condition": condition, "context_owner": context_owners[index],
                "oracle_non_deployable": int(condition == "ORACLE"), "gain": meta["gain"],
                "severity": "zero" if meta["zero_artifact"] else ("mild" if meta["gain"] < 0.55 else "medium" if meta["gain"] < 0.95 else "severe"),
                "zero_artifact": meta["zero_artifact"], "output_input_rms": float(np.sqrt(np.mean(predicted ** 2)) / max(np.sqrt(np.mean(y ** 2)), 1e-8)),
                "query_eog_inference_reads": 0, "query_transfer_in_condition": int(condition == "ORACLE"),
                **values,
            })
            for channel in range(len(x)):
                denom = max(float(np.linalg.norm(x[channel])), 1e-12)
                channel_rows.append({
                    "fold": fold, "seed": seed, "participant": meta["participant"], "condition": condition,
                    "channel": channel, "channel_rrmse_temporal": float(np.linalg.norm(predicted[channel] - x[channel]) / denom),
                    "channel_artifact_rmse": float(np.sqrt(np.mean((predicted_artifact[channel] - artifact[channel]) ** 2))),
                })
    return rows, channel_rows


def duration_evaluation(model: CalibEEGDfus, schedule: OfficialLinearSchedule, bank: Mapping[str, Any],
                        samplers: Mapping[int, TransferEpisodeSampler], device: torch.device,
                        fold: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for seconds in (0, 10, 30):
        sampler = samplers[30] if seconds == 0 else samplers[seconds]
        condition = "POP" if seconds == 0 else "MATCH"
        signatures, _ = _episode_signatures(sampler, bank["meta"], condition)
        predicted = _sample_clean(model, schedule, bank["y"], signatures, device, 510000 + fold * 100 + seed % 100)
        for x, y, artifact, output, meta in zip(bank["x"], bank["y"], bank["artifact"], predicted, bank["meta"]):
            rows.append({
                "fold": fold, "seed": seed, "participant": meta["participant"], "support_seconds": seconds,
                "effective_seconds": seconds, "window_count": seconds // 2,
                "rrmse_temporal": paired_metrics(x, y, artifact, y - output)["rrmse_temporal"],
                "same_query": 1, "same_checkpoint": 1, "same_noise": 1,
            })
    return rows


def run_fold(result_root: Path, data: Mapping[str, Any], fold: Mapping[str, Any], seed: int,
             device: torch.device, updates: int = 12000, batch_episodes: int = 4,
             validation_interval: int = 250, run_id: str = "runtime") -> dict[str, Any]:
    seed_all(seed)
    fold_id = int(fold["fold"])
    runtime = result_root / run_id / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, 0.05)
    registry10 = TransferRegistry(data, fold, 10, 0.05)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    test_sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    validation_bank = validation_sampler.sample_balanced(4)
    test_bank = test_sampler.sample_balanced(8)
    model = CalibEEGDfus().to(device)
    schedule = OfficialLinearSchedule().to(device)
    checkpoint = runtime / "calib_eegdfus.pt"
    curve = train_model(model, schedule, train_sampler, validation_bank, device, seed, updates,
                        batch_episodes, validation_interval, checkpoint)
    paired, channel = evaluate_paired(model, schedule, test_bank, test_sampler, device, fold_id, seed)
    sampler10 = TransferEpisodeSampler(data, fold, "test", seed + 3, registry10)
    duration = duration_evaluation(model, schedule, test_bank, {10: sampler10, 30: test_sampler}, device, fold_id, seed)
    # Bind support-derived inference states so natural inference never opens query EOG.
    test_keys = sorted(key for key in registry30.cells if key[0] in fold["test"])
    match = np.stack([registry30.signature(*key, "MATCH") for key in test_keys])
    population = np.stack([registry30.signature(*key, "POP") for key in test_keys])
    wrong_values, wrong_owners = [], []
    for key in test_keys:
        value, owner = test_sampler.condition_signature({"participant": key[0], "session": key[1], "task": key[2]}, "WRONG")
        wrong_values.append(value); wrong_owners.append(owner)
    signature_path = runtime / "inference_signatures.npz"
    np.savez_compressed(signature_path, keys=np.asarray(["|".join(key) for key in test_keys]),
                        match=match, population=population, wrong=np.stack(wrong_values),
                        wrong_owners=np.asarray(wrong_owners), eeg_scale=registry30.eeg_scale)
    result = {
        "fold": fold_id, "seed": seed, "paired_metrics": paired, "channel_metrics": channel,
        "support_duration": duration, "transfer_manifest": registry30.manifest_rows(), "training_curve": curve,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "fold": fold_id, "seed": seed,
                       "model": "Calib-EEGDfus", "best_criterion": "validation_epsilon_l1",
                       "best_step": int(torch.load(checkpoint, map_location="cpu", weights_only=False)["step"])},
        "inference_signature_binding": {"path": str(signature_path), "sha256": sha256(signature_path),
                                        "query_eog_reads": 0, "state_source": "support_prefix_only"},
        "participant_coverage": sorted(set(row["participant"] for row in paired)),
        "sealed_reads": 0, "query_eog_inference_reads": 0,
    }
    (runtime / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _natural_inference_bank(data: Mapping[str, Any], fold: Mapping[str, Any], signature_path: Path,
                            windows_per_cell: int = 4) -> dict[str, Any]:
    with np.load(signature_path, allow_pickle=False) as archive:
        lookup = {str(key): index for index, key in enumerate(archive["keys"])}
        match, population, wrong = np.asarray(archive["match"]), np.asarray(archive["population"]), np.asarray(archive["wrong"])
        wrong_owners, eeg_scale = [str(value) for value in archive["wrong_owners"]], np.asarray(archive["eeg_scale"])
    arrays = {key: [] for key in ("y", "MATCH", "POP", "WRONG")}; meta = []
    source_root = Path(data["v19_derived_root"])
    length, qstart = int(data["window_samples"]), int(data["qnatural_start"])
    for participant, session, task in itertools.product(fold["test"], data["sessions"], data["tasks"]):
        encoded = "|".join((participant, session, task))
        if encoded not in lookup:
            continue
        path = source_root / "prepared" / participant / f"{session}_{task}.npz"
        if not path.is_file():
            continue
        # Read EEG only. The eog member is not opened by the inference stage.
        with np.load(path, allow_pickle=False) as archive:
            eeg = np.asarray(archive["eeg"], np.float64)
        starts = np.linspace(qstart, eeg.shape[1] - length, windows_per_cell, dtype=int)
        index = lookup[encoded]
        for start in starts:
            arrays["y"].append((eeg[:, start:start + length] / eeg_scale[:, None]).astype(np.float32))
            arrays["MATCH"].append(match[index]); arrays["POP"].append(population[index]); arrays["WRONG"].append(wrong[index])
            meta.append({"participant": participant, "session": session, "task": task, "start": int(start),
                         "wrong_owner": wrong_owners[index], "query_eog_inference_reads": 0})
    return {**{key: np.stack(value) for key, value in arrays.items()}, "meta": meta}


def natural_output_freeze(result_root: Path, data: Mapping[str, Any], fold: Mapping[str, Any], seed: int,
                          device: torch.device, paired_result: Path, run_id: str) -> dict[str, Any]:
    source = json.loads(paired_result.read_text())
    checkpoint = Path(source["checkpoint"]["path"]); signature_path = Path(source["inference_signature_binding"]["path"])
    if sha256(checkpoint) != source["checkpoint"]["sha256"] or sha256(signature_path) != source["inference_signature_binding"]["sha256"]:
        raise RuntimeError("V41R natural binding checksum mismatch")
    model = CalibEEGDfus().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    schedule = OfficialLinearSchedule().to(device)
    bank = _natural_inference_bank(data, fold, signature_path)
    outputs = {}
    for condition in ("POP", "MATCH", "WRONG"):
        outputs[condition] = _sample_clean(model, schedule, bank["y"], bank[condition], device,
                                           610000 + int(fold["fold"]) * 100 + seed % 100)
    runtime = result_root / run_id / f"fold_{fold['fold']}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    freeze = runtime / "output_freeze.npz"
    np.savez_compressed(freeze, y=bank["y"], pop=outputs["POP"], match=outputs["MATCH"], wrong=outputs["WRONG"],
                        participant=np.asarray([row["participant"] for row in bank["meta"]]),
                        session=np.asarray([row["session"] for row in bank["meta"]]),
                        task=np.asarray([row["task"] for row in bank["meta"]]),
                        start=np.asarray([row["start"] for row in bank["meta"]]),
                        wrong_owner=np.asarray([row["wrong_owner"] for row in bank["meta"]]))
    manifest = {"path": str(freeze), "sha256": sha256(freeze), "rows": len(bank["meta"]),
                "conditions": ["POP", "MATCH", "WRONG"], "query_eog_inference_reads": 0,
                "query_operator_inference_reads": 0, "sealed_reads": 0, "evaluator_opened": False}
    (runtime / "output_freeze.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def natural_evaluator(result_root: Path, data: Mapping[str, Any], fold: Mapping[str, Any], seed: int,
                      freeze_manifest: Path) -> list[dict[str, Any]]:
    manifest = json.loads(freeze_manifest.read_text()); freeze = Path(manifest["path"])
    if sha256(freeze) != manifest["sha256"] or manifest["evaluator_opened"]:
        raise RuntimeError("natural output freeze is not immutable")
    registry = TransferRegistry(data, fold, 30, include_query_transfer=True)
    with np.load(freeze, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    rows = []
    root = Path(data["v19_derived_root"])
    for index in range(len(payload["y"])):
        participant, session, task, start = str(payload["participant"][index]), str(payload["session"][index]), str(payload["task"][index]), int(payload["start"][index])
        path = root / "prepared" / participant / f"{session}_{task}.npz"
        with np.load(path, allow_pickle=False) as archive:
            eye = np.asarray(archive["eog"], np.float64)
            names = [str(value) for value in archive["eog_names"]]
        from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
        eog = bipolar_eog(eye[:, start:start + 512], names)
        cell = registry.cells[(participant, session, task)]
        latent = (eog - cell.eog_center[:, None]) / cell.eog_scale[:, None]
        teacher = cell.query_transfer @ latent
        energy = np.sqrt(np.mean(latent * latent, axis=0)); low = energy <= np.quantile(energy, .3); high = energy >= np.quantile(energy, .7)
        y = payload["y"][index]
        for condition, key in (("POP", "pop"), ("MATCH", "match"), ("WRONG", "wrong")):
            output = payload[key][index]; estimate = y - output
            remaining = float(np.linalg.norm(teacher[:, high] - estimate[:, high]) / max(np.linalg.norm(teacher[:, high]), 1e-8))
            retention = 1 - float(np.linalg.norm(estimate[:, low]) / max(np.linalg.norm(y[:, low]), 1e-8))
            f, p0 = signal.welch(y[:, low], fs=100, nperseg=min(128, int(low.sum())), axis=-1)
            _, p1 = signal.welch(output[:, low], fs=100, nperseg=min(128, int(low.sum())), axis=-1); keep = (f >= 1) & (f <= 15)
            covariance = np.cov(y[:, low])
            rows.append({"fold": fold["fold"], "seed": seed, "participant": participant, "session": session,
                         "task": task, "condition": condition, "heldout_eog_remaining_ratio": remaining,
                         "artifact_attenuation_db": float(-20 * np.log10(max(remaining, 1e-8))),
                         "low_eog_observation_retention": retention,
                         "psd_distortion": float(np.mean(np.abs(np.log(p0[:, keep] + 1e-8) - np.log(p1[:, keep] + 1e-8)))),
                         "covariance_distortion": float(np.linalg.norm(np.cov(output[:, low]) - covariance) / max(np.linalg.norm(covariance), 1e-8)),
                         "output_input_rms": float(np.sqrt(np.mean(output ** 2)) / max(np.sqrt(np.mean(y ** 2)), 1e-8)),
                         "query_eog_inference_reads": 0, "evaluator_query_eog_reads": 1})
    result = freeze_manifest.parent / "natural_result.json"
    result.write_text(json.dumps({"natural_metrics": rows, "output_freeze_sha256": manifest["sha256"],
                                  "evaluator_opened_after_freeze": True}, indent=2, sort_keys=True) + "\n")
    return rows


__all__ = ["CONDITIONS", "duration_evaluation", "evaluate_paired", "natural_evaluator",
           "natural_output_freeze", "run_fold", "train_model"]
