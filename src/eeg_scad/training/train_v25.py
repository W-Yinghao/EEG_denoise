from __future__ import annotations
import copy,time
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import torch
from torch import Tensor,nn
from eeg_scad.context.learned_spatial_decoder import basis_decorrelation,decode_residual,ridge_latent
from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.models.setcalib_det import SetCalibDET
from eeg_scad.models.setcalib_diff import SetCalibDiffConfig,SetCalibResidualDiffusion
from eeg_scad.training.checkpoint import EMA
from eeg_scad.training.train_v24 import load_anchor

def _t(value:np.ndarray,device:torch.device)->Tensor:return torch.as_tensor(value,dtype=torch.float32,device=device)
def _projection(c0:Tensor,q0:Tensor)->Tensor:return torch.einsum("bcd,bdt->bct",c0,q0)
def _save(path:Path,value:Mapping[str,Any])->None:path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(value),path)
def _ema(model:nn.Module,ema:EMA)->nn.Module:
    output=copy.deepcopy(model).eval();holder=EMA(output);holder.load_state_dict(ema.state_dict());holder.copy_to(output);return output
def _batch(sampler:SupportSetEpisodeSampler,size:int,natural_fraction:float)->dict[str,Any]:return sampler.sample_natural(size) if sampler.rng.random()<natural_fraction else sampler.sample_paired(size)
def _anchor(batch:Mapping[str,Any],anchor:nn.Module,device:torch.device)->tuple[Tensor,Tensor,Tensor,Tensor]:
    y=_t(batch["y"],device);q0=_t(batch["q0"],device);c0=_t(batch["c0"],device)
    with torch.no_grad():a0=anchor(y,q0,_projection(c0,q0))
    target=_t(batch["artifact"] if batch["stream"]=="paired" else batch["teacher_artifact"],device);return y,q0,a0,target
def _validation(sampler:SupportSetEpisodeSampler)->dict[str,dict[str,Any]]:
    state=sampler.state();paired=sampler.sample_paired(96);natural=sampler.sample_natural(96);sampler.set_state(state);return {"paired":paired,"natural":natural}

@torch.no_grad()
def validate_det(model:SetCalibDET,anchor:nn.Module,bank:Mapping[str,dict[str,Any]],device:torch.device)->dict[str,float]:
    result={}
    for stream,b in bank.items():
        y,q0,a0,target=_anchor(b,anchor,device);output=model(y,a0,q0,_t(b["support_eeg"],device),_t(b["support_eog"],device));wrong=model(y,a0,q0,_t(b["wrong_support_eeg"],device),_t(b["wrong_support_eog"],device));result[f"{stream}_artifact_mse"]=float((output["artifact"]-target).square().mean());result[f"{stream}_wrong_mse"]=float((wrong["artifact"]-target).square().mean())
    return result

def train_det(fold:int,seed:int,encoder:str,cfg:Mapping[str,Any],data:Mapping[str,Any],fold_cfg:Mapping[str,Any],checkpoint_root:Path,anchor_path:Path,resume:bool=False)->dict[str,Any]:
    device=torch.device("cuda");anchor,_=load_anchor(anchor_path,device)
    for p in anchor.parameters():p.requires_grad_(False)
    sampler=SupportSetEpisodeSampler(data,fold_cfg,"train",seed);validation=_validation(SupportSetEpisodeSampler(data,fold_cfg,"validation",seed+19));model=SetCalibDET(encoder=encoder,rank=8).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));curves=[];best={"paired":float("inf"),"natural":float("inf"),"joint":float("inf")};bad=0;start=0;last=checkpoint_root/"last.pt"
    if resume and last.is_file():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);ema.load_state_dict(state["ema"]);sampler.set_state(state["support_window_rng"]);start=state["step"];curves=state["curves"];best=state["best"];bad=state["bad"]
    started=time.time();maximum=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"])
    for step in range(start,maximum):
        b=_batch(sampler,int(cfg["batch_size"]),float(cfg["natural_fraction"]));y,q0,a0,target=_anchor(b,anchor,device);seeg=_t(b["support_eeg"],device);seog=_t(b["support_eog"],device);weeg=_t(b["wrong_support_eeg"],device);weog=_t(b["wrong_support_eog"],device);optimizer.zero_grad(set_to_none=True);out=model(y,a0,q0,seeg,seog);wrong=model(y,a0,q0,weeg,weog);error=(out["artifact"]-target).abs().mean()+.5*(out["artifact"]-target).square().mean();clean=(out["clean"]-(y-target)).abs().mean();per=(out["artifact"]-target).square().mean((1,2));wper=(wrong["artifact"]-target).square().mean((1,2));rank=torch.relu(float(cfg["margin"])+per-wper).mean();operator=(out["operator"]-_t(b["cquery"],device)).square().mean();basis=basis_decorrelation(out["basis"]);loss=error+clean+float(cfg["lambda_rank"])*rank+float(cfg["lambda_operator"])*operator+float(cfg["lambda_basis"])*basis;loss.backward();finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters());
        if not finite:raise RuntimeError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            metrics=validate_det(_ema(model,ema),anchor,validation,device);joint=metrics["paired_artifact_mse"]+metrics["natural_artifact_mse"];curves.append({"step":step+1,"loss":float(loss.detach()),"rank":float(rank.detach()),"operator":float(operator.detach()),**metrics,"joint":joint});payload={"model":model.state_dict(),"ema":ema.state_dict(),"config":dict(cfg),"encoder":encoder,"fold":fold,"seed":seed,"step":step+1,"anchor":str(anchor_path)}
            for key,value,name in (("paired",metrics["paired_artifact_mse"],"best_paired.pt"),("natural",metrics["natural_artifact_mse"],"best_natural.pt"),("joint",joint,"best_joint.pt")):
                if value<best[key]:best[key]=value;_save(checkpoint_root/name,payload);bad=0 if key=="joint" else bad
                elif key=="joint":bad+=1
            _save(last,{**payload,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"support_window_rng":sampler.state(),"wrong_support_rng":sampler.state(),"paired_mixture_rng":sampler.state(),"natural_stream_rng":sampler.state(),"curves":curves,"best":best,"bad":bad})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":"setcalib_det","encoder":encoder,"fold":fold,"seed":seed,"updates":step+1,"best":best,"curve":curves,"checkpoint":str(checkpoint_root/"best_joint.pt"),"training_seconds":time.time()-started,"parameters":sum(p.numel() for p in model.parameters()),"device":torch.cuda.get_device_name(0)}

def load_det(path:Path,device:torch.device)->tuple[SetCalibDET,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);model=SetCalibDET(encoder=state["encoder"],rank=8).to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);model.eval();return model,state

def train_diff(fold:int,seed:int,cfg:Mapping[str,Any],data:Mapping[str,Any],fold_cfg:Mapping[str,Any],checkpoint_root:Path,anchor_path:Path,det_path:Path,resume:bool=False)->dict[str,Any]:
    device=torch.device("cuda");anchor,_=load_anchor(anchor_path,device);det,_=load_det(det_path,device)
    for module in (anchor,det):
        for p in module.parameters():p.requires_grad_(False)
    sampler=SupportSetEpisodeSampler(data,fold_cfg,"train",seed);validation=_validation(SupportSetEpisodeSampler(data,fold_cfg,"validation",seed+23));config=SetCalibDiffConfig(rank=8,base_channels=int(cfg["base_channels"]),timesteps=int(cfg["diffusion_steps"]),ddim_steps=int(cfg["ddim_steps"]));model=SetCalibResidualDiffusion(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));generator=torch.Generator(device=device).manual_seed(seed+707);curves=[];best=float("inf");bad=0;started=time.time();maximum=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"])
    for step in range(maximum):
        b=_batch(sampler,int(cfg["batch_size"]),.3);y,q0,a0,target=_anchor(b,anchor,device)
        with torch.no_grad():out=det(y,a0,q0,_t(b["support_eeg"],device),_t(b["support_eog"],device));hstar=ridge_latent(target-a0,out["basis"]);residual=hstar-out["coefficient"]
        optimizer.zero_grad(set_to_none=True);base,pred=model.training_loss(residual,y,a0,q0,out["coefficient"],out["context"],generator);decoded=decode_residual(a0,out["basis"],out["coefficient"]+pred);field=(decoded-target).square().mean();loss=base+float(cfg["lambda_decoded"])*field;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            eval_model=_ema(model,ema);scores=[]
            for stream,v in validation.items():
                vy,vq,va0,vtarget=_anchor(v,anchor,device);vo=det(vy,va0,vq,_t(v["support_eeg"],device),_t(v["support_eog"],device));noise=torch.randn(vo["coefficient"].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+step));correction=eval_model.sample(vy,va0,vq,vo["coefficient"],vo["context"],noise);prediction=decode_residual(va0,vo["basis"],vo["coefficient"]+correction);scores.append(float((prediction-vtarget).square().mean()))
            joint=sum(scores);curves.append({"step":step+1,"residual_loss":float(base.detach()),"decoded_loss":float(field.detach()),"paired_sampling_mse":scores[0],"natural_sampling_mse":scores[1],"joint":joint});payload={"model":model.state_dict(),"ema":ema.state_dict(),"config":dict(cfg),"fold":fold,"seed":seed,"step":step+1,"det":str(det_path),"anchor":str(anchor_path)}
            if joint<best:best=joint;bad=0;_save(checkpoint_root/"best_joint.pt",payload)
            else:bad+=1
            _save(checkpoint_root/"last.pt",{**payload,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"diffusion_rng":generator.get_state(),"support_window_rng":sampler.state(),"wrong_support_rng":sampler.state(),"paired_mixture_rng":sampler.state(),"natural_stream_rng":sampler.state(),"curves":curves,"best":best,"bad":bad})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":"setcalib_diff","fold":fold,"seed":seed,"updates":step+1,"best_joint":best,"curve":curves,"checkpoint":str(checkpoint_root/"best_joint.pt"),"training_seconds":time.time()-started,"parameters":sum(p.numel() for p in model.parameters()),"device":torch.cuda.get_device_name(0)}

def load_diff(path:Path,device:torch.device)->tuple[SetCalibResidualDiffusion,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);cfg=state["config"];model=SetCalibResidualDiffusion(SetCalibDiffConfig(rank=8,base_channels=int(cfg["base_channels"]),timesteps=int(cfg["diffusion_steps"]),ddim_steps=int(cfg["ddim_steps"]))).to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);model.eval();return model,state

__all__=["train_det","train_diff","load_det","load_diff","validate_det"]
