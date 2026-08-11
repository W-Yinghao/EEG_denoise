from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from eeg_scad.data.eog_latent_streams import EOGStreamSampler, generate_bank
from eeg_scad.models.pa_el_det import decode_deviation
from eeg_scad.models.pa_el_scad import PAELResidualDiffusion, PAELSCADConfig
from eeg_scad.models.population_anchor_v24 import PopulationAnchorV24
from eeg_scad.models.temporal_eog_net import TemporalEOGNet
from eeg_scad.training.checkpoint import EMA


def _t(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def _ema_model(model: nn.Module, ema: EMA) -> nn.Module:
    value = copy.deepcopy(model).eval()
    holder = EMA(value)
    holder.load_state_dict(ema.state_dict())
    holder.copy_to(value)
    return value


def _projection(c0: Tensor, q0: Tensor) -> Tensor:
    return torch.einsum("bcd,bdt->bct", c0, q0)


def _latent_metrics(predicted: Tensor, target: Tensor) -> tuple[float, float]:
    rmse = float((predicted - target).square().mean().sqrt())
    p = predicted.flatten(1); t = target.flatten(1)
    p = p - p.mean(1, keepdim=True); t = t - t.mean(1, keepdim=True)
    correlation = float(((p * t).sum(1) / (torch.linalg.vector_norm(p, dim=1) * torch.linalg.vector_norm(t, dim=1)).clamp_min(1e-8)).mean())
    return rmse, correlation


@torch.no_grad()
def validate_anchor(model: PopulationAnchorV24, bank: Mapping[str, Any], device: torch.device) -> dict[str, float]:
    result = {}
    for stream in ("paired", "natural"):
        b = bank[stream]
        y = _t(b["y"], device); c0 = _t(b["c0"], device); q0 = _t(b["q0"], device)
        target = _t(b["artifact"] if stream == "paired" else b["teacher_artifact"], device)
        predicted = model(y, q0, _projection(c0, q0))
        result[f"{stream}_artifact_mse"] = float((predicted - target).square().mean())
    return result


@torch.no_grad()
def validate_temporal(model: TemporalEOGNet, anchor: PopulationAnchorV24, bank: Mapping[str, Any], device: torch.device) -> dict[str, float]:
    result = {}
    for stream in ("paired", "natural"):
        b = bank[stream]
        y = _t(b["y"], device); c0 = _t(b["c0"], device); q0 = _t(b["q0"], device); ds = _t(b["ds"], device)
        target_e = _t(b["latent"], device)
        target_a = _t(b["artifact"] if stream == "paired" else b["teacher_artifact"], device)
        a0 = anchor(y, q0, _projection(c0, q0)); predicted_e = model(y, a0, q0); predicted_a = decode_deviation(a0, ds, predicted_e)
        rmse, correlation = _latent_metrics(predicted_e, target_e)
        result[f"{stream}_latent_rmse"] = rmse
        result[f"{stream}_latent_correlation"] = correlation
        result[f"{stream}_artifact_mse"] = float((predicted_a - target_a).square().mean())
    return result


def _mix(sampler: EOGStreamSampler, batch_size: int, natural_fraction: float) -> dict[str, Any]:
    if sampler.rng.random() < natural_fraction:
        return sampler.sample_natural(batch_size)
    return sampler.sample_paired(batch_size)


def train_anchor(fold: int, seed: int, cfg: Mapping[str, Any], data: Mapping[str, Any], fold_cfg: Mapping[str, Any], checkpoint_root: Path, resume: bool = False) -> dict[str, Any]:
    device = torch.device("cuda")
    sampler = EOGStreamSampler(data, fold_cfg, "train", seed)
    validation = generate_bank(EOGStreamSampler(data, fold_cfg, "validation", seed + 11), 256, seed + 900, 0.30)
    model = PopulationAnchorV24(width=int(cfg["base_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["maximum_updates"]))
    ema = EMA(model, float(cfg["ema"])); curves = []
    best_paired = best_natural = best_joint = float("inf"); bad = 0; start = 0
    last = checkpoint_root / "last.pt"
    if resume and last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); ema.load_state_dict(state["ema"])
        sampler.set_state(state["data_rng"]); start = int(state["step"]); curves = state["curves"]; best_paired = state["best_paired"]; best_natural = state["best_natural"]; best_joint = state["best_joint"]; bad = state["bad"]
    started = time.time(); interval = int(cfg["validation_interval"]); maximum = int(cfg["maximum_updates"])
    for step in range(start, maximum):
        batch = _mix(sampler, int(cfg["batch_size"]), 0.30)
        y = _t(batch["y"], device); c0 = _t(batch["c0"], device); q0 = _t(batch["q0"], device)
        target = _t(batch["artifact"] if batch["stream"] == "paired" else batch["teacher_artifact"], device)
        optimizer.zero_grad(set_to_none=True); prediction = model(y, q0, _projection(c0, q0)); base = (prediction - target).square().mean()
        zero = torch.linalg.vector_norm(target.flatten(1), dim=1) == 0
        identity = prediction[zero].square().mean() if batch["stream"] == "paired" and bool(zero.any()) else base * 0
        loss = base + float(cfg["lambda_identity"]) * identity; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); scheduler.step(); ema.update(model)
        if (step + 1) % interval == 0:
            metrics = validate_anchor(_ema_model(model, ema), validation, device); joint = metrics["paired_artifact_mse"] + metrics["natural_artifact_mse"]
            curves.append({"step": step + 1, "training_loss": float(loss.detach()), **metrics, "joint_validation": joint})
            if metrics["paired_artifact_mse"] < best_paired:
                best_paired = metrics["paired_artifact_mse"]; _save(checkpoint_root / "best_paired.pt", {"model": model.state_dict(), "ema": ema.state_dict(), "config": dict(cfg), "fold": fold, "seed": seed, "step": step + 1, "kind": "population_anchor"})
            if metrics["natural_artifact_mse"] < best_natural:
                best_natural = metrics["natural_artifact_mse"]; _save(checkpoint_root / "best_natural.pt", {"model": model.state_dict(), "ema": ema.state_dict(), "config": dict(cfg), "fold": fold, "seed": seed, "step": step + 1, "kind": "population_anchor"})
            if joint < best_joint - 1e-7:
                best_joint = joint; bad = 0; _save(checkpoint_root / "best_joint.pt", {"model": model.state_dict(), "ema": ema.state_dict(), "config": dict(cfg), "fold": fold, "seed": seed, "step": step + 1, "kind": "population_anchor"})
            else:
                bad += 1
            _save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "ema": ema.state_dict(), "data_rng": sampler.state(), "stream_rng": sampler.state(), "step": step + 1, "curves": curves, "best_paired": best_paired, "best_natural": best_natural, "best_joint": best_joint, "bad": bad, "config": dict(cfg), "kind": "population_anchor"})
            if step + 1 >= int(cfg["minimum_updates"]) and bad >= int(cfg["early_stopping_patience"]):
                break
    return {"kind": "population_anchor", "fold": fold, "seed": seed, "updates": step + 1, "best_paired": best_paired, "best_natural": best_natural, "best_joint": best_joint, "curve": curves, "checkpoint": str(checkpoint_root / "best_joint.pt"), "last_checkpoint": str(last), "training_seconds": time.time() - started, "parameters": sum(p.numel() for p in model.parameters()), "device": torch.cuda.get_device_name(0)}


def load_anchor(path: Path, device: torch.device) -> tuple[PopulationAnchorV24, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = PopulationAnchorV24(width=int(state["config"]["base_channels"])).to(device)
    holder = EMA(model); holder.load_state_dict(state["ema"]); holder.copy_to(model); model.eval()
    return model, state


def _spectral_l1(predicted: Tensor, target: Tensor) -> Tensor:
    p = torch.stft(predicted.flatten(0, 1), n_fft=64, hop_length=16, return_complex=True)
    t = torch.stft(target.flatten(0, 1), n_fft=64, hop_length=16, return_complex=True)
    return (torch.log(p.abs() + 1e-6) - torch.log(t.abs() + 1e-6)).abs().mean()


def train_temporal(fold: int, seed: int, cfg: Mapping[str, Any], data: Mapping[str, Any], fold_cfg: Mapping[str, Any], checkpoint_root: Path, anchor_path: Path, resume: bool = False) -> dict[str, Any]:
    device = torch.device("cuda"); anchor, _ = load_anchor(anchor_path, device)
    for parameter in anchor.parameters(): parameter.requires_grad_(False)
    sampler = EOGStreamSampler(data, fold_cfg, "train", seed)
    validation = generate_bank(EOGStreamSampler(data, fold_cfg, "validation", seed + 11), 256, seed + 901, 0.30)
    model = TemporalEOGNet(width=int(cfg["hidden_channels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["maximum_updates"])); ema = EMA(model, float(cfg["ema"])); curves = []
    best_eog = best_paired = best_natural = best_joint = float("inf"); bad = 0; start = 0; last = checkpoint_root / "last.pt"
    if resume and last.is_file():
        state = torch.load(last, map_location=device, weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); ema.load_state_dict(state["ema"]); sampler.set_state(state["data_rng"]); start = state["step"]; curves = state["curves"]; best_eog = state["best_eog"]; best_paired = state["best_paired"]; best_natural = state["best_natural"]; best_joint = state["best_joint"]; bad = state["bad"]
    started = time.time(); interval = int(cfg["validation_interval"]); maximum = int(cfg["maximum_updates"])
    for step in range(start, maximum):
        batch = _mix(sampler, int(cfg["batch_size"]), 0.30)
        y = _t(batch["y"], device); c0 = _t(batch["c0"], device); q0 = _t(batch["q0"], device); ds = _t(batch["ds"], device); dw = _t(batch["dw"], device); target_e = _t(batch["latent"], device); target_a = _t(batch["artifact"] if batch["stream"] == "paired" else batch["teacher_artifact"], device)
        with torch.no_grad(): a0 = anchor(y, q0, _projection(c0, q0))
        optimizer.zero_grad(set_to_none=True); predicted_e = model(y, a0, q0); predicted_a = decode_deviation(a0, ds, predicted_e)
        latent_loss = torch.nn.functional.smooth_l1_loss(predicted_e, target_e); derivative = torch.nn.functional.l1_loss(torch.diff(predicted_e), torch.diff(target_e)); spectral = _spectral_l1(predicted_e, target_e); artifact = (predicted_a - target_a).square().mean(); zero = torch.linalg.vector_norm(target_a.flatten(1), dim=1) == 0; identity = predicted_a[zero].square().mean() if bool(zero.any()) else artifact * 0; rank = artifact * 0
        if batch["stream"] == "paired" and bool((~zero).any()):
            wrong_a = decode_deviation(a0, dw, predicted_e); per_match = (predicted_a - target_a).square().mean((1,2)); per_wrong = (wrong_a - target_a).square().mean((1,2)); rank = torch.relu(float(cfg["context_margin"]) + per_match[~zero] - per_wrong[~zero]).mean()
        loss = float(cfg["lambda_e"]) * latent_loss + float(cfg["lambda_delta"]) * derivative + float(cfg["lambda_spec"]) * spectral + float(cfg["lambda_artifact"]) * artifact + float(cfg["lambda_identity"]) * identity + float(cfg["lambda_ctx"]) * rank
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); scheduler.step(); ema.update(model)
        if (step + 1) % interval == 0:
            metrics = validate_temporal(_ema_model(model, ema), anchor, validation, device); joint = metrics["paired_artifact_mse"] + metrics["natural_artifact_mse"] + metrics["paired_latent_rmse"] + metrics["natural_latent_rmse"]
            curves.append({"step": step + 1, "latent_loss": float(latent_loss.detach()), "derivative_loss": float(derivative.detach()), "spectral_loss": float(spectral.detach()), "artifact_loss": float(artifact.detach()), "ranking_loss": float(rank.detach()), **metrics, "joint_validation": joint})
            payload = {"model": model.state_dict(), "ema": ema.state_dict(), "config": dict(cfg), "fold": fold, "seed": seed, "step": step + 1, "kind": "temporal_eog", "anchor": str(anchor_path)}
            eog_value = metrics["paired_latent_rmse"] + metrics["natural_latent_rmse"]
            if eog_value < best_eog: best_eog = eog_value; _save(checkpoint_root / "best_eog.pt", payload)
            if metrics["paired_artifact_mse"] < best_paired: best_paired = metrics["paired_artifact_mse"]; _save(checkpoint_root / "best_paired.pt", payload)
            if metrics["natural_artifact_mse"] < best_natural: best_natural = metrics["natural_artifact_mse"]; _save(checkpoint_root / "best_natural.pt", payload)
            if joint < best_joint - 1e-7: best_joint = joint; bad = 0; _save(checkpoint_root / "best_joint.pt", payload)
            else: bad += 1
            _save(last, {**payload, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "data_rng": sampler.state(), "role_rng": sampler.state(), "stream_rng": sampler.state(), "curves": curves, "best_eog": best_eog, "best_paired": best_paired, "best_natural": best_natural, "best_joint": best_joint, "bad": bad})
            if step + 1 >= int(cfg["minimum_updates"]) and bad >= int(cfg["early_stopping_patience"]): break
    return {"kind": "temporal_eog", "fold": fold, "seed": seed, "updates": step + 1, "best_eog": best_eog, "best_paired": best_paired, "best_natural": best_natural, "best_joint": best_joint, "curve": curves, "checkpoint": str(checkpoint_root / "best_joint.pt"), "last_checkpoint": str(last), "training_seconds": time.time() - started, "parameters": sum(p.numel() for p in model.parameters()), "device": torch.cuda.get_device_name(0)}


def load_temporal(path: Path, device: torch.device) -> tuple[TemporalEOGNet, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False); model = TemporalEOGNet(width=int(state["config"]["hidden_channels"])).to(device); holder = EMA(model); holder.load_state_dict(state["ema"]); holder.copy_to(model); model.eval(); return model, state


def train_diffusion(fold: int, seed: int, cfg: Mapping[str, Any], data: Mapping[str, Any], fold_cfg: Mapping[str, Any], checkpoint_root: Path, anchor_path: Path, temporal_path: Path, resume: bool = False) -> dict[str, Any]:
    device = torch.device("cuda"); anchor, _ = load_anchor(anchor_path, device); temporal, _ = load_temporal(temporal_path, device)
    for module in (anchor, temporal):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    sampler = EOGStreamSampler(data, fold_cfg, "train", seed); validation = generate_bank(EOGStreamSampler(data, fold_cfg, "validation", seed + 11), 128, seed + 902, 0.30)
    config = PAELSCADConfig(base_channels=int(cfg["base_channels"]), timesteps=int(cfg["diffusion_steps"]), ddim_steps=int(cfg["ddim_steps"])); model = PAELResidualDiffusion(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["maximum_updates"])); ema = EMA(model, float(cfg["ema"])); generator = torch.Generator(device=device).manual_seed(seed + 88); curves = []; best = float("inf"); bad = 0; start = 0; last = checkpoint_root / "last.pt"
    if resume and last.is_file():
        state = torch.load(last, map_location=device, weights_only=False); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"]); ema.load_state_dict(state["ema"]); generator.set_state(state["diffusion_rng"].cpu()); sampler.set_state(state["data_rng"]); start = state["step"]; curves = state["curves"]; best = state["best"]; bad = state["bad"]
    started = time.time(); interval = int(cfg["validation_interval"]); maximum = int(cfg["maximum_updates"])
    for step in range(start, maximum):
        batch = _mix(sampler, int(cfg["batch_size"]), 0.30); y = _t(batch["y"], device); c0 = _t(batch["c0"], device); q0 = _t(batch["q0"], device); ds = _t(batch["ds"], device); target_e = _t(batch["latent"], device); target_a = _t(batch["artifact"] if batch["stream"] == "paired" else batch["teacher_artifact"], device)
        with torch.no_grad(): a0 = anchor(y, q0, _projection(c0, q0)); zdet = temporal(y, a0, q0)
        residual = target_e - zdet; optimizer.zero_grad(set_to_none=True); base, extra = model.training_loss(residual, y, a0, q0, zdet, generator); artifact = decode_deviation(a0, ds, zdet + extra["predicted_x0"]); decoded = (artifact - target_a).square().mean(); loss = base + float(cfg["lambda_decoded"]) * decoded; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); scheduler.step(); ema.update(model)
        if (step + 1) % interval == 0:
            evaluation = _ema_model(model, ema); scores = []
            for stream in ("paired", "natural"):
                b = validation[stream]; vy = _t(b["y"], device); vc0 = _t(b["c0"], device); vq0 = _t(b["q0"], device); vds = _t(b["ds"], device); va = _t(b["artifact"] if stream == "paired" else b["teacher_artifact"], device)
                with torch.no_grad(): va0 = anchor(vy, vq0, _projection(vc0, vq0)); vz = temporal(vy, va0, vq0); noise = torch.randn(vz.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed + step + (0 if stream == "paired" else 1))); correction, _ = evaluation.sample(vy, va0, vq0, vz, noise, int(cfg["ddim_steps"])); pred = decode_deviation(va0, vds, vz + correction); scores.append(float((pred - va).square().mean()))
            joint = sum(scores); curves.append({"step": step + 1, "residual_loss": float(base.detach()), "decoded_loss": float(decoded.detach()), "paired_sampling_mse": scores[0], "natural_sampling_mse": scores[1], "joint_validation": joint})
            payload = {"model": model.state_dict(), "ema": ema.state_dict(), "config": dict(cfg), "fold": fold, "seed": seed, "step": step + 1, "kind": "pa_el_scad", "anchor": str(anchor_path), "temporal": str(temporal_path)}
            if joint < best - 1e-7: best = joint; bad = 0; _save(checkpoint_root / "best_sampling.pt", payload)
            else: bad += 1
            _save(last, {**payload, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "diffusion_rng": generator.get_state(), "data_rng": sampler.state(), "role_rng": sampler.state(), "stream_rng": sampler.state(), "curves": curves, "best": best, "bad": bad})
            if step + 1 >= int(cfg["minimum_updates"]) and bad >= int(cfg["early_stopping_patience"]): break
    return {"kind": "pa_el_scad", "fold": fold, "seed": seed, "updates": step + 1, "best_joint": best, "curve": curves, "checkpoint": str(checkpoint_root / "best_sampling.pt"), "last_checkpoint": str(last), "training_seconds": time.time() - started, "parameters": sum(p.numel() for p in model.parameters()), "device": torch.cuda.get_device_name(0)}


def load_diffusion(path: Path, device: torch.device) -> tuple[PAELResidualDiffusion, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False); cfg = state["config"]; model = PAELResidualDiffusion(PAELSCADConfig(base_channels=int(cfg["base_channels"]), timesteps=int(cfg["diffusion_steps"]), ddim_steps=int(cfg["ddim_steps"]))).to(device); holder = EMA(model); holder.load_state_dict(state["ema"]); holder.copy_to(model); model.eval(); return model, state


__all__ = ["load_anchor", "load_diffusion", "load_temporal", "train_anchor", "train_diffusion", "train_temporal", "validate_anchor", "validate_temporal"]
