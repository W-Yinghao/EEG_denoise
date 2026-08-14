"""Clean-room V42R native sanity, joint x0 training, and frozen evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule, TRANSFER_DIM, ddim_sample


CONDITIONS = ("POP", "MATCH", "WRONG", "SHUFFLED", "ORACLE", "NO_TRANSFER_BRANCH")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, np.float32)).to(device)


class EMA:
    def __init__(self, model: nn.Module, decay: float = .999) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for target, source in zip(self.model.parameters(), model.parameters()):
            target.lerp_(source.detach(), 1 - self.decay)
        for target, source in zip(self.model.buffers(), model.buffers()):
            target.copy_(source)

    def state_dict(self):
        return self.model.state_dict()


def _conditions(sampler: TransferEpisodeSampler, meta: list[dict[str, Any]], condition: str) -> tuple[np.ndarray, list[str]]:
    values, owners = [], []
    route = "POP" if condition == "NO_TRANSFER_BRANCH" else condition
    for row in meta:
        value, owner = sampler.condition_signature(row, route)
        values.append(value); owners.append(owner)
    return np.stack(values), owners


@torch.no_grad()
def sample_bank(model: CalibSADDPMCond, schedule: LinearX0Schedule, observed: np.ndarray,
                transfer: np.ndarray, device: torch.device, seed: int,
                transfer_enabled: bool = True, episode_batch: int = 2) -> np.ndarray:
    model.eval(); outputs = []
    for start in range(0, len(observed), episode_batch):
        stop = min(len(observed), start + episode_batch)
        y = _tensor(observed[start:stop], device); c = _tensor(transfer[start:stop], device)
        generator = torch.Generator(device=device).manual_seed(seed + start * 1009)
        noise = torch.randn(y.shape, device=device, generator=generator)
        output = ddim_sample(model, y, c, noise, schedule, 50, transfer_enabled)
        outputs.append(output.cpu().numpy())
    return np.concatenate(outputs)


def participant_pop_score(model: CalibSADDPMCond, schedule: LinearX0Schedule,
                          bank: Mapping[str, Any], sampler: TransferEpisodeSampler,
                          device: torch.device, seed: int) -> float:
    signature, _ = _conditions(sampler, bank["meta"], "POP")
    prediction = sample_bank(model, schedule, bank["y"], signature, device, seed)
    values: dict[str, list[float]] = {}
    for clean, output, row in zip(bank["x"], prediction, bank["meta"]):
        values.setdefault(row["participant"], []).append(
            float(np.linalg.norm(output - clean) / max(np.linalg.norm(clean), 1e-12)))
    return float(np.mean([np.mean(rows) for rows in values.values()]))


def train_joint(model: CalibSADDPMCond, schedule: LinearX0Schedule,
                train_sampler: TransferEpisodeSampler, validation_bank: Mapping[str, Any],
                validation_sampler: TransferEpisodeSampler, device: torch.device, seed: int,
                updates: int, batch_size: int, validation_interval: int, runtime: Path) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model, .999)
    # Registered single engineering repair: the canonical AMP pilot produced a
    # nonfinite gradient at update 1905.  Full-float training changes no science.
    amp_enabled = False
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator(device=device).manual_seed(seed + 7001)
    curve, best = [], float("inf")
    best_path, last_path = runtime / "best.pt", runtime / "last.pt"
    for step in range(1, updates + 1):
        bank = train_sampler.sample(batch_size)
        signature = bank["signature"].copy()
        dropped = train_sampler.rng.random(batch_size) < .20
        for index in np.flatnonzero(dropped):
            signature[index], _ = train_sampler.condition_signature(bank["meta"][int(index)], "POP")
        clean, observed, condition = _tensor(bank["x"], device), _tensor(bank["y"], device), _tensor(signature, device)
        noisy, timestep, _ = schedule.forward_sample(clean, generator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            prediction = model(noisy, observed, timestep, condition)
            loss = F.smooth_l1_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite V42R loss at update {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        transfer_gradient = torch.stack([parameter.grad.detach().norm() for parameter in model.transfer_parameters()
                                         if parameter.grad is not None]).sum()
        if not torch.isfinite(gradient) or not torch.isfinite(transfer_gradient):
            raise FloatingPointError(f"nonfinite V42R gradient at update {step}")
        scaler.step(optimizer); scaler.update(); ema.update(model)
        if step % validation_interval == 0 or step == updates:
            score = participant_pop_score(ema.model, schedule, validation_bank, validation_sampler,
                                          device, seed + 17001)
            row = {"step": step, "train_smooth_l1": float(loss.detach()), "validation_pop_rrmse": score,
                   "gradient_norm": float(gradient), "transfer_gradient_norm": float(transfer_gradient),
                   "context_dropout_fraction": float(dropped.mean())}
            curve.append(row)
            payload = {"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                       "scaler": scaler.state_dict(), "step": step, "seed": seed, "curve": curve,
                       "best_validation_pop_rrmse": min(best, score),
                       "data_rng": train_sampler.rng.bit_generator.state,
                       "diffusion_rng": generator.get_state().cpu(),
                       "contract": {"channels": 46, "samples": 512, "prediction": "x0",
                                    "diffusion_steps": 1000, "ddim_steps": 50, "context_dropout": .20}}
            runtime.mkdir(parents=True, exist_ok=True); torch.save(payload, last_path)
            if score < best:
                best = score; payload["best_validation_pop_rrmse"] = score; torch.save(payload, best_path)
    if not best_path.is_file():
        raise RuntimeError("V42R validation created no checkpoint")
    return curve


def evaluate_joint(model: CalibSADDPMCond, schedule: LinearX0Schedule, bank: Mapping[str, Any],
                   sampler: TransferEpisodeSampler, device: torch.device, fold: int, seed: int):
    rows = []
    for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
        values = paired_metrics(clean, observed, artifact, np.zeros_like(artifact))
        rows.append({"fold": fold, "seed": seed, "participant": meta["participant"], "condition": "RAW",
                     "context_owner": "NONE", "oracle_non_deployable": 0, "query_eog_inference_reads": 0,
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"], "output_input_rms": 1., **values})
    for condition in CONDITIONS:
        signature, owners = _conditions(sampler, bank["meta"], condition)
        output = sample_bank(model, schedule, bank["y"], signature, device,
                             420000 + fold * 100 + seed % 100,
                             transfer_enabled=condition != "NO_TRANSFER_BRANCH")
        for clean, observed, artifact, prediction, meta, owner in zip(bank["x"], bank["y"], bank["artifact"], output, bank["meta"], owners):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V42R paired output")
            values = paired_metrics(clean, observed, artifact, observed - prediction)
            rows.append({"fold": fold, "seed": seed, "participant": meta["participant"], "condition": condition,
                         "context_owner": owner, "oracle_non_deployable": int(condition == "ORACLE"),
                         "query_eog_inference_reads": 0, "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         "output_input_rms": float(np.sqrt(np.mean(prediction ** 2)) / max(np.sqrt(np.mean(observed ** 2)), 1e-8)), **values})
    return rows


def duration_joint(model, schedule, bank, samplers, device, fold, seed):
    rows = []
    for seconds in (0, 10, 30):
        sampler = samplers[30] if seconds == 0 else samplers[seconds]
        condition = "POP" if seconds == 0 else "MATCH"
        signature, _ = _conditions(sampler, bank["meta"], condition)
        output = sample_bank(model, schedule, bank["y"], signature, device, 520000 + fold * 100 + seed % 100)
        for clean, observed, artifact, prediction, meta in zip(bank["x"], bank["y"], bank["artifact"], output, bank["meta"]):
            rows.append({"fold": fold, "seed": seed, "participant": meta["participant"], "support_seconds": seconds,
                         "effective_seconds": seconds, "window_count": seconds // 2,
                         "rrmse_temporal": paired_metrics(clean, observed, artifact, observed - prediction)["rrmse_temporal"],
                         "same_checkpoint": 1, "same_query": 1, "same_noise": 1})
    return rows


def run_cell(result_root: Path, data: Mapping[str, Any], fold: Mapping[str, Any], seed: int,
             device: torch.device, updates: int, run_id: str, batch_size: int = 8,
             validation_interval: int = 2000) -> dict[str, Any]:
    torch.manual_seed(seed); np.random.seed(seed)
    runtime = result_root / run_id / f"fold_{fold['fold']}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    registry10 = TransferRegistry(data, fold, 10, .05)
    train_sampler = TransferEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    test_sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    test_bank = test_sampler.sample_balanced(8)
    model = CalibSADDPMCond().to(device); schedule = LinearX0Schedule().to(device)
    curve = train_joint(model, schedule, train_sampler, validation_bank, validation_sampler, device,
                        seed, updates, batch_size, validation_interval, runtime)
    checkpoint = runtime / "best.pt"; payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["ema"])
    paired = evaluate_joint(model, schedule, test_bank, test_sampler, device, int(fold["fold"]), seed)
    sampler10 = TransferEpisodeSampler(data, fold, "test", seed + 3, registry10)
    duration = duration_joint(model, schedule, test_bank, {10: sampler10, 30: test_sampler}, device, int(fold["fold"]), seed)
    # Persist only support-derived states for frozen natural inference.  This
    # deliberately excludes the independent query-generating transfer and all
    # query EOG, so the inference/evaluator boundary can be audited by digest.
    test_keys = sorted(key for key in registry30.cells if key[0] in fold["test"])
    match = np.stack([registry30.signature(*key, "MATCH") for key in test_keys])
    population = np.stack([registry30.signature(*key, "POP") for key in test_keys])
    wrong_values, wrong_owners = [], []
    for key in test_keys:
        value, owner = test_sampler.condition_signature(
            {"participant": key[0], "session": key[1], "task": key[2]}, "WRONG"
        )
        wrong_values.append(value); wrong_owners.append(owner)
    signature_path = runtime / "inference_signatures.npz"
    np.savez_compressed(
        signature_path,
        keys=np.asarray(["|".join(key) for key in test_keys]),
        match=match,
        population=population,
        wrong=np.stack(wrong_values),
        wrong_owners=np.asarray(wrong_owners),
        eeg_scale=registry30.eeg_scale,
    )
    result = {"fold": fold["fold"], "seed": seed, "updates": updates, "paired_metrics": paired,
              "support_duration": duration, "training_curve": curve,
              "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "best_step": payload["step"],
                             "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"]},
              "inference_signature_binding": {"path": str(signature_path), "sha256": sha256(signature_path),
                                                "query_eog_reads": 0, "state_source": "support_prefix_only"},
              "participants": sorted(set(row["participant"] for row in paired)), "sealed_reads": 0,
              "query_eog_inference_reads": 0}
    (runtime / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def native_sanity(data_root: Path, result_root: Path, device: torch.device, seed: int = 20261200,
                  updates: int = 10000) -> dict[str, Any]:
    def load(name):
        with np.load(data_root / f"{name}.npz", allow_pickle=False) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    train, validation, test = load("train"), load("validation"), load("test")
    model = CalibSADDPMCond(channels=1).to(device); schedule = LinearX0Schedule().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model); rng = np.random.Generator(np.random.PCG64DXSM(seed)); generator = torch.Generator(device=device).manual_seed(seed + 1)
    condition = lambda count: torch.zeros((count, 1, TRANSFER_DIM), device=device)
    curve=[]; best=float("inf"); result_root.mkdir(parents=True, exist_ok=True); checkpoint=result_root/"native_best.pt"
    for step in range(1, updates + 1):
        index=rng.integers(0,len(train["clean"]),size=64); clean=_tensor(train["clean"][index],device); observed=_tensor(train["noisy"][index],device)
        noisy,timestep,_=schedule.forward_sample(clean,generator); prediction=model(noisy,observed,timestep,condition(len(clean)))
        loss=F.smooth_l1_loss(prediction,clean); optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.update(model)
        if step%1000==0 or step==updates:
            probe=np.arange(min(64,len(validation["clean"]))); vy=_tensor(validation["noisy"][probe],device); vx=_tensor(validation["clean"][probe],device)
            noise=torch.randn(vy.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+100+step)); out=ddim_sample(ema.model,vy,condition(len(vy)),noise,schedule,50)
            score=float(torch.linalg.vector_norm(out-vx,dim=(1,2)).div(torch.linalg.vector_norm(vx,dim=(1,2)).clamp_min(1e-8)).mean())
            curve.append({"step":step,"train_smooth_l1":float(loss.detach()),"validation_rrmse":score})
            if score<best:best=score;torch.save({"ema":ema.state_dict(),"step":step,"score":score},checkpoint)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["ema"]); rows=[]
    for start in range(0,len(test["clean"]),64):
        y=_tensor(test["noisy"][start:start+64],device); x=test["clean"][start:start+64]
        noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+9000+start));out=ddim_sample(model,y,condition(len(y)),noise,schedule,50).cpu().numpy()
        for clean,observed,prediction in zip(x,test["noisy"][start:start+64],out):rows.append(paired_metrics(clean,observed,observed-clean,observed-prediction))
    summary={metric:float(np.mean([row[metric] for row in rows])) for metric in ("rrmse_temporal","rrmse_spectral","correlation")}
    result={"status":"valid" if np.isfinite(list(summary.values())).all() and summary["rrmse_temporal"]<1 else "invalid",
            "updates":updates,"best_validation_rrmse":best,"test":summary,"curve":curve,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint)}
    (result_root/"native_result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");return result


__all__ = ["CONDITIONS", "EMA", "evaluate_joint", "native_sanity", "run_cell", "sample_bank", "train_joint"]
