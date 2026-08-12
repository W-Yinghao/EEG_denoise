"""Training and exact checkpoint resume for V28 clean conditional models."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.energy.projector import projector
from eeg_scad.models.pop_clean_cdm import PopCleanCDM
from eeg_scad.models.pop_clean_det import PopCleanDET
from eeg_scad.models.support_clean_cdm import SupportCleanCDM
from eeg_scad.models.support_clean_det import SupportCleanDET
from eeg_scad.training.checkpoint import EMA
from eeg_scad.training.train_v25 import load_det


KINDS = {"pop_det": PopCleanDET, "support_det": SupportCleanDET, "pop_cdm": PopCleanCDM, "support_cdm": SupportCleanCDM}


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(dict(value), path)


def _ema_model(model: nn.Module, ema: EMA) -> nn.Module:
    result=copy.deepcopy(model).eval();holder=EMA(result);holder.load_state_dict(ema.state_dict());holder.copy_to(result);return result


def _spectral(left: Tensor, right: Tensor) -> Tensor:
    a=torch.fft.rfft(left,dim=-1).abs().clamp_min(1e-6).log();b=torch.fft.rfft(right,dim=-1).abs().clamp_min(1e-6).log();return (a-b).abs().mean()


def _correlation_loss(left: Tensor, right: Tensor) -> Tensor:
    left=left.flatten(1)-left.flatten(1).mean(1,keepdim=True);right=right.flatten(1)-right.flatten(1).mean(1,keepdim=True);return 1-(left*right).sum(1).div(left.norm(dim=1)*right.norm(dim=1)+1e-8).mean()


@torch.no_grad()
def support_features(support_model: nn.Module, batch: Mapping[str, Any], device: torch.device, wrong: bool=False) -> tuple[Tensor,Tensor]:
    prefix="wrong_" if wrong else "";encoded=support_model.encode_support(_tensor(batch[prefix+"support_eeg"],device),_tensor(batch[prefix+"support_eog"],device));return encoded["context"],projector(encoded["basis"])


def _natural_losses(prediction: Tensor, y: Tensor, latent: Tensor, pi: Tensor | None) -> tuple[Tensor,Tensor]:
    energy=latent.square().mean(1).sqrt();threshold=torch.quantile(energy,.3,dim=1,keepdim=True);low=(energy<=threshold).float()[:,None];delta=prediction-y;low_loss=(delta.abs()*low).sum()/low.sum().clamp_min(1)/delta.shape[1]
    if pi is None:return low_loss,torch.zeros((),device=y.device)
    q=torch.eye(y.shape[1],device=y.device)[None]-pi;complement=torch.einsum("bij,bjt->bit",q,delta);return low_loss,complement.abs().mean()


def _predict_training(model: nn.Module, kind: str, batch: Mapping[str,Any], support_model: nn.Module, generator: torch.Generator, device: torch.device) -> tuple[Tensor,Tensor,Tensor|None]:
    y=_tensor(batch["y"],device);context=pi=None
    if kind.startswith("support"):
        context,pi=support_features(support_model,batch,device)
    if kind=="pop_det":prediction=model(y)
    elif kind=="support_det":prediction=model(y,context)
    elif kind=="pop_cdm":prediction,_,_=model.training_prediction(_tensor(batch["x"],device) if batch["stream"]=="paired" else y,y,generator)
    else:prediction,_,_=model.training_prediction(_tensor(batch["x"],device) if batch["stream"]=="paired" else y,y,context,generator)
    return prediction,y,pi


@torch.no_grad()
def predict(model: nn.Module, kind: str, batch: Mapping[str,Any], support_model: nn.Module, device: torch.device, seed: int, steps: int=25, wrong: bool=False, null: bool=False) -> Tensor:
    y=_tensor(batch["y"],device)
    if kind=="pop_det":return model(y)
    if kind=="pop_cdm":return model.sample(y,torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed)),steps)[0]
    if null:context=torch.zeros((len(y),128),device=device)
    else:context,_=support_features(support_model,batch,device,wrong)
    if kind=="support_det":return model(y,context)
    noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed));return model.sample(y,context,noise,steps)[0]


@torch.no_grad()
def validate(model: nn.Module, kind: str, bank: Mapping[str,dict[str,Any]], support_model: nn.Module, device: torch.device, seed: int, steps: int) -> dict[str,float]:
    paired=bank["paired"];prediction=predict(model,kind,paired,support_model,device,seed,steps);target=_tensor(paired["x"],device);denominator=target.norm().clamp_min(1e-8);result={"paired_clean_rrmse":float((prediction-target).norm()/denominator),"paired_spectral":float(_spectral(prediction,target)),"paired_correlation_loss":float(_correlation_loss(prediction,target))}
    natural=bank["natural"];npred=predict(model,kind,natural,support_model,device,seed+1,steps);ny=_tensor(natural["y"],device);latent=_tensor(natural["latent"],device);pi=None
    if kind.startswith("support"):_,pi=support_features(support_model,natural,device)
    low,q=_natural_losses(npred,ny,latent,pi);result.update({"natural_low_change":float(low),"natural_complement_change":float(q),"joint":result["paired_clean_rrmse"]+float(low)+float(q)})
    return result


def train(kind: str, fold: int, seed: int, cfg: Mapping[str,Any], data: Mapping[str,Any], fold_cfg: Mapping[str,Any], output: Path, support_checkpoint: Path, resume: bool=False) -> dict[str,Any]:
    if kind not in KINDS:raise ValueError(kind)
    device=torch.device("cuda");support_model,support_state=load_det(support_checkpoint,device)
    for parameter in support_model.parameters():parameter.requires_grad_(False)
    model=KINDS[kind](int(cfg["width"]),int(cfg["diffusion_steps"])) if kind.endswith("cdm") else KINDS[kind](int(cfg["width"]));model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));sampler=SupportSetEpisodeSampler(data,fold_cfg,"train",seed);validation_sampler=SupportSetEpisodeSampler(data,fold_cfg,"validation",seed+91);validation={"paired":validation_sampler.sample_paired(48,float(cfg["identity_fraction"])),"natural":validation_sampler.sample_natural(48)};generator=torch.Generator(device=device).manual_seed(seed+811);curves=[];best={"paired":float("inf"),"natural":float("inf"),"joint":float("inf"),"full_ddim":float("inf")};bad=0;start=0;last=output/"last.pt"
    if resume and last.is_file():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);ema.load_state_dict(state["ema"]);sampler.set_state(state["data_rng"]);generator.set_state(state["diffusion_rng"].cpu());curves=state["curves"];best=state["best"];bad=state["bad"];start=int(state["step"])
    maximum=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"]);started=time.time()
    for step in range(start,maximum):
        natural=bool(sampler.rng.random()<float(cfg["natural_fraction"]));batch=sampler.sample_natural(int(cfg["batch_size"])) if natural else sampler.sample_paired(int(cfg["batch_size"]),float(cfg["identity_fraction"]));prediction,y,pi=_predict_training(model,kind,batch,support_model,generator,device)
        if natural:
            low,q=_natural_losses(prediction,y,_tensor(batch["latent"],device),pi);loss=float(cfg["lambda_low"])*low+float(cfg["lambda_Q"])*q
        else:
            target=_tensor(batch["x"],device);loss=F.smooth_l1_loss(prediction,target)+float(cfg["lambda_spec"])*_spectral(prediction,target)+float(cfg["lambda_corr"])*_correlation_loss(prediction,target);identity=torch.as_tensor([bool(m["zero_artifact"]) for m in batch["meta"]],device=device)
            if identity.any():loss=loss+float(cfg["lambda_id"])*(prediction[identity]-y[identity]).abs().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            metrics=validate(_ema_model(model,ema),kind,validation,support_model,device,seed+step,int(cfg["ddim_steps"]));curves.append({"step":step+1,"loss":float(loss.detach()),**metrics});payload={"model":model.state_dict(),"ema":ema.state_dict(),"kind":kind,"fold":fold,"seed":seed,"step":step+1,"config":dict(cfg),"support_checkpoint":str(support_checkpoint),"support_checkpoint_sha":support_state.get("checkpoint_sha")}
            for key,value,name in (("paired",metrics["paired_clean_rrmse"],"best_paired_x0.pt"),("natural",metrics["natural_low_change"]+metrics["natural_complement_change"],"best_natural_consistency.pt"),("joint",metrics["joint"],"best_joint.pt"),("full_ddim",metrics["joint"],"best_full_ddim.pt")):
                if value<best[key]:best[key]=value;_save(output/name,payload);bad=0 if key=="joint" else bad
                elif key=="joint":bad+=1
            _save(last,{**payload,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"amp_scaler":{},"data_rng":sampler.state(),"support_rng":sampler.state(),"wrong_support_rng":sampler.state(),"diffusion_rng":generator.get_state(),"stream_rng":sampler.state(),"curves":curves,"best":best,"bad":bad})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":kind,"fold":fold,"seed":seed,"updates":step+1,"best":best,"curve":curves,"checkpoint":str(output/"best_joint.pt"),"support_checkpoint":str(support_checkpoint),"parameters":sum(p.numel() for p in model.parameters()),"training_seconds":time.time()-started,"device":torch.cuda.get_device_name(0)}


def load(path:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);kind=state["kind"];cfg=state["config"];model=KINDS[kind](int(cfg["width"]),int(cfg["diffusion_steps"])) if kind.endswith("cdm") else KINDS[kind](int(cfg["width"]));model=model.to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);return model.eval(),state


__all__=["train","load","predict","support_features","validate","KINDS"]
