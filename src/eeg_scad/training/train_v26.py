"""Training and checkpoint loading for V26 matched refiners."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.models.calib_refine_det import CalibRefineDET, PopRefineDET
from eeg_scad.models.calib_sdedit import CalibSDEdit, PopSDEdit, sigma_to_timestep
from eeg_scad.training.checkpoint import EMA
from eeg_scad.training.train_v24 import load_anchor
from eeg_scad.training.train_v25 import load_det


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(value), path)


def _ema_model(model: nn.Module, ema: EMA) -> nn.Module:
    result = copy.deepcopy(model).eval()
    holder = EMA(result)
    holder.load_state_dict(ema.state_dict())
    holder.copy_to(result)
    return result


def _spectral(left: Tensor, right: Tensor) -> Tensor:
    a = torch.fft.rfft(left, dim=-1).abs().clamp_min(1e-6).log()
    b = torch.fft.rfft(right, dim=-1).abs().clamp_min(1e-6).log()
    return (a-b).abs().mean()


@torch.no_grad()
def frozen_outputs(batch: Mapping[str, Any], anchor: nn.Module, det: nn.Module, device: torch.device) -> dict[str, Tensor]:
    y = _tensor(batch["y"], device); q0 = _tensor(batch["q0"], device); c0 = _tensor(batch["c0"], device)
    pop = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0))
    match = det(y, pop, q0, _tensor(batch["support_eeg"], device), _tensor(batch["support_eog"], device))
    wrong = det(y, pop, q0, _tensor(batch["wrong_support_eeg"], device), _tensor(batch["wrong_support_eog"], device))
    target = _tensor(batch["artifact"] if batch["stream"] == "paired" else batch["teacher_artifact"], device)
    return {"y": y, "q0": q0, "pop": pop, "match": match["artifact"], "wrong": wrong["artifact"], "match_context": match["context"], "wrong_context": wrong["context"], "target": target}


def _sample(sampler: SupportSetEpisodeSampler, size: int, natural_fraction: float) -> dict[str, Any]:
    return sampler.sample_natural(size) if sampler.rng.random() < natural_fraction else sampler.sample_paired(size)


def _validation(sampler: SupportSetEpisodeSampler) -> dict[str, dict[str, Any]]:
    state = sampler.state(); paired = sampler.sample_paired(96); natural = sampler.sample_natural(96); sampler.set_state(state)
    return {"paired": paired, "natural": natural}


def _one_step(model: nn.Module, kind: str, frozen: Mapping[str, Tensor]) -> Tensor:
    return model(frozen["y"], frozen["pop"]) if kind == "pop_refine_det" else model(frozen["y"], frozen["match"], frozen["pop"], frozen["match_context"])


@torch.no_grad()
def validate_one_step(model: nn.Module, kind: str, bank: Mapping[str, dict[str, Any]], anchor: nn.Module, det: nn.Module, device: torch.device) -> dict[str, float]:
    metrics = {}
    for stream, batch in bank.items():
        frozen = frozen_outputs(batch, anchor, det, device); prediction = _one_step(model, kind, frozen)
        metrics[f"{stream}_artifact_mse"] = float((prediction-frozen["target"]).square().mean())
        metrics[f"{stream}_clean_rrmse"] = float(torch.linalg.vector_norm(prediction-frozen["target"]) / torch.linalg.vector_norm(frozen["y"]-frozen["target"]).clamp_min(1e-8))
        metrics[f"{stream}_identity"] = float(prediction.square().mean().sqrt()) if stream == "paired" else 0.0
        metrics[f"{stream}_refinement_rms"] = float((prediction-(frozen["pop"] if kind == "pop_refine_det" else frozen["match"])).square().mean().sqrt())
    return metrics


def train_one_step(kind: str, fold: int, seed: int, cfg: Mapping[str, Any], data: Mapping[str, Any], fold_cfg: Mapping[str, Any], output: Path, anchor_path: Path, det_path: Path, resume: bool = False) -> dict[str, Any]:
    if kind not in ("calib_refine_det", "pop_refine_det"): raise ValueError(kind)
    device = torch.device("cuda"); anchor, _ = load_anchor(anchor_path, device); det, _ = load_det(det_path, device)
    for module in (anchor, det):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    model = (CalibRefineDET if kind == "calib_refine_det" else PopRefineDET)(int(cfg["width"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["maximum_updates"])); ema = EMA(model, float(cfg["ema"])); sampler = SupportSetEpisodeSampler(data, fold_cfg, "train", seed); validation = _validation(SupportSetEpisodeSampler(data, fold_cfg, "validation", seed+19)); curves=[]; best={"paired":float("inf"),"natural":float("inf"),"joint":float("inf")}; bad=0; start=0; last=output/"last.pt"
    if resume and last.is_file():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);ema.load_state_dict(state["ema"]);sampler.set_state(state["data_rng"]);curves=state["curves"];best=state["best"];bad=state["bad"];start=int(state["step"])
    begun=time.time(); maximum=int(cfg["maximum_updates"]); interval=int(cfg["validation_interval"])
    for step in range(start, maximum):
        batch=_sample(sampler,int(cfg["batch_size"]),float(cfg["natural_fraction"])); frozen=frozen_outputs(batch,anchor,det,device); prediction=_one_step(model,kind,frozen); target=frozen["target"]
        artifact=(prediction-target).abs().mean()+.5*(prediction-target).square().mean(); clean=((frozen["y"]-prediction)-(frozen["y"]-target)).abs().mean(); spectral=_spectral(prediction,target); loss=artifact+float(cfg["lambda_clean"])*clean+float(cfg["lambda_spec"])*spectral
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            metrics=validate_one_step(_ema_model(model,ema),kind,validation,anchor,det,device);joint=metrics["paired_artifact_mse"]+metrics["natural_artifact_mse"];curves.append({"step":step+1,"loss":float(loss.detach()),"joint":joint,**metrics});payload={"model":model.state_dict(),"ema":ema.state_dict(),"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed,"step":step+1,"anchor":str(anchor_path),"det":str(det_path)}
            for key,value,name in (("paired",metrics["paired_artifact_mse"],"best_paired.pt"),("natural",metrics["natural_artifact_mse"],"best_natural.pt"),("joint",joint,"best_joint.pt")):
                if value<best[key]:best[key]=value;_save(output/name,payload);bad=0 if key=="joint" else bad
                elif key=="joint":bad+=1
            _save(last,{**payload,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"amp_scaler":{},"data_rng":sampler.state(),"support_rng":sampler.state(),"wrong_owner_rng":sampler.state(),"curves":curves,"best":best,"bad":bad})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":kind,"fold":fold,"seed":seed,"updates":step+1,"best":best,"curve":curves,"checkpoint":str(output/"best_joint.pt"),"parameters":sum(p.numel() for p in model.parameters()),"training_seconds":time.time()-begun,"device":torch.cuda.get_device_name(0)}


def load_one_step(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    state=torch.load(path,map_location=device,weights_only=False);cls=CalibRefineDET if state["kind"]=="calib_refine_det" else PopRefineDET;model=cls(int(state["config"]["width"])).to(device);ema=EMA(model);ema.load_state_dict(state["ema"]);ema.copy_to(model);return model.eval(),state


def _sd_sample(model: nn.Module, kind: str, frozen: Mapping[str, Tensor], noise: Tensor, sigma: float, steps: int) -> tuple[Tensor,list[dict[str,float]]]:
    if kind=="pop_sdedit":return model.sample(frozen["y"],frozen["pop"],noise,sigma,steps)
    return model.sample(frozen["y"],frozen["match"],frozen["pop"],frozen["match_context"],noise,sigma,steps)


@torch.no_grad()
def validate_sdedit(model: nn.Module, kind: str, bank: Mapping[str, dict[str, Any]], anchor: nn.Module, det: nn.Module, device: torch.device, sigma: float, steps: int, seed: int) -> dict[str,float]:
    metrics={}
    for stream,batch in bank.items():
        frozen=frozen_outputs(batch,anchor,det,device);base=frozen["pop"] if kind=="pop_sdedit" else frozen["match"];noise=torch.randn(base.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+(0 if stream=="paired" else 1)));prediction,_=_sd_sample(model,kind,frozen,noise,sigma,steps);metrics[f"{stream}_artifact_mse"]=float((prediction-frozen["target"]).square().mean());metrics[f"{stream}_clean_rrmse"]=float(torch.linalg.vector_norm(prediction-frozen["target"])/torch.linalg.vector_norm(frozen["y"]-frozen["target"]).clamp_min(1e-8));metrics[f"{stream}_refinement_rms"]=float((prediction-base).square().mean().sqrt())
    return metrics


def train_sdedit(kind: str, fold: int, seed: int, cfg: Mapping[str, Any], data: Mapping[str, Any], fold_cfg: Mapping[str, Any], output: Path, anchor_path: Path, det_path: Path, resume: bool=False) -> dict[str,Any]:
    if kind not in ("calib_sdedit","pop_sdedit"):raise ValueError(kind)
    device=torch.device("cuda");anchor,_=load_anchor(anchor_path,device);det,_=load_det(det_path,device)
    for module in (anchor,det):
        for parameter in module.parameters():parameter.requires_grad_(False)
    model=(CalibSDEdit if kind=="calib_sdedit" else PopSDEdit)(int(cfg["width"]),int(cfg["diffusion_steps"])).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));sampler=SupportSetEpisodeSampler(data,fold_cfg,"train",seed);validation=_validation(SupportSetEpisodeSampler(data,fold_cfg,"validation",seed+23));generator=torch.Generator(device=device).manual_seed(seed+707);curves=[];best={"paired":float("inf"),"natural":float("inf"),"joint":float("inf"),"full_sampling":float("inf")};bad=0;start=0;last=output/"last.pt";tmax=sigma_to_timestep(model.alpha_bar,.35)
    if resume and last.is_file():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);ema.load_state_dict(state["ema"]);generator.set_state(state["diffusion_rng"]);sampler.set_state(state["data_rng"]);curves=state["curves"];best=state["best"];bad=state["bad"];start=int(state["step"])
    begun=time.time();maximum=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"]);sigma=float(cfg["sigma_start"]);ddim=int(cfg["ddim_steps"])
    for step in range(start,maximum):
        batch=_sample(sampler,int(cfg["batch_size"]),float(cfg["natural_fraction"]));frozen=frozen_outputs(batch,anchor,det,device);target=frozen["target"]
        if kind=="pop_sdedit":loss,prediction,_=model.training_loss(target,frozen["y"],frozen["pop"],tmax,generator);base=frozen["pop"]
        else:loss,prediction,_=model.training_loss(target,frozen["y"],frozen["match"],frozen["pop"],frozen["match_context"],tmax,generator);base=frozen["match"]
        clean=((frozen["y"]-prediction)-(frozen["y"]-target)).abs().mean();spectral=_spectral(prediction,target);anchor_penalty=(prediction-base).abs().mean();total=loss+float(cfg["lambda_clean"])*clean+float(cfg["lambda_spec"])*spectral+float(cfg["lambda_anchor"])*anchor_penalty
        optimizer.zero_grad(set_to_none=True);total.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            metrics=validate_sdedit(_ema_model(model,ema),kind,validation,anchor,det,device,sigma,ddim,seed+step);joint=metrics["paired_artifact_mse"]+metrics["natural_artifact_mse"];curves.append({"step":step+1,"loss":float(total.detach()),"joint":joint,**metrics});payload={"model":model.state_dict(),"ema":ema.state_dict(),"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed,"step":step+1,"anchor":str(anchor_path),"det":str(det_path)}
            for key,value,name in (("paired",metrics["paired_artifact_mse"],"best_paired.pt"),("natural",metrics["natural_artifact_mse"],"best_natural.pt"),("joint",joint,"best_joint.pt"),("full_sampling",joint,"best_full_sampling.pt")):
                if value<best[key]:best[key]=value;_save(output/name,payload);bad=0 if key=="joint" else bad
                elif key=="joint":bad+=1
            _save(last,{**payload,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"amp_scaler":{},"diffusion_rng":generator.get_state(),"data_rng":sampler.state(),"support_rng":sampler.state(),"wrong_owner_rng":sampler.state(),"curves":curves,"best":best,"bad":bad})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":kind,"fold":fold,"seed":seed,"updates":step+1,"best":best,"curve":curves,"checkpoint":str(output/"best_joint.pt"),"parameters":sum(p.numel() for p in model.parameters()),"training_seconds":time.time()-begun,"device":torch.cuda.get_device_name(0),"sigma_start":sigma,"t0":sigma_to_timestep(model.alpha_bar,sigma),"alpha_bar_t0":float(model.alpha_bar[sigma_to_timestep(model.alpha_bar,sigma)]),"ddim_steps":ddim}


def load_sdedit(path:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);cls=CalibSDEdit if state["kind"]=="calib_sdedit" else PopSDEdit;model=cls(int(state["config"]["width"]),int(state["config"]["diffusion_steps"])).to(device);ema=EMA(model);ema.load_state_dict(state["ema"]);ema.copy_to(model);return model.eval(),state


__all__=["train_one_step","train_sdedit","load_one_step","load_sdedit","frozen_outputs","validate_one_step","validate_sdedit"]
