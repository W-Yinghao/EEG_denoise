"""V29 adapter-only training around frozen V28 population models."""
from __future__ import annotations

import copy, time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.energy.projector import projector
from eeg_scad.models.pop_adapter_cdm import PopAdapterCDM
from eeg_scad.models.pop_adapter_det import PopAdapterDET
from eeg_scad.models.support_adapter_cdm import SupportAdapterCDM
from eeg_scad.models.support_adapter_det import SupportAdapterDET
from eeg_scad.models.support_adapter_common import freeze
from eeg_scad.training.checkpoint import EMA
from eeg_scad.training.train_v25 import load_det as load_support
from eeg_scad.training.train_v28 import load as load_population

KINDS={"support_adapter_det":SupportAdapterDET,"pop_adapter_det":PopAdapterDET,"support_adapter_cdm":SupportAdapterCDM,"pop_adapter_cdm":PopAdapterCDM}


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:return torch.as_tensor(value,dtype=torch.float32,device=device)
def _save(path:Path,value:Mapping[str,Any])->None:path.parent.mkdir(parents=True,exist_ok=True);torch.save(dict(value),path)
def _ema_model(model:nn.Module,ema:EMA)->nn.Module:
    result=copy.deepcopy(model).eval();holder=EMA(result);holder.load_state_dict(ema.state_dict());holder.copy_to(result);return result
def _spectral(left:Tensor,right:Tensor)->Tensor:
    a=torch.fft.rfft(left,dim=-1).abs().clamp_min(1e-6).log();b=torch.fft.rfft(right,dim=-1).abs().clamp_min(1e-6).log();return (a-b).abs().mean()
def _corr(left:Tensor,right:Tensor)->Tensor:
    a=left.flatten(1)-left.flatten(1).mean(1,keepdim=True);b=right.flatten(1)-right.flatten(1).mean(1,keepdim=True);return 1-(a*b).sum(1).div(a.norm(dim=1)*b.norm(dim=1)+1e-8).mean()


@torch.no_grad()
def contexts(support:nn.Module,batch:Mapping[str,Any],device:torch.device)->tuple[Tensor,Tensor,Tensor]:
    match=support.encode_support(_tensor(batch["support_eeg"],device),_tensor(batch["support_eog"],device));wrong=support.encode_support(_tensor(batch["wrong_support_eeg"],device),_tensor(batch["wrong_support_eog"],device));return match["context"],wrong["context"],projector(match["basis"])


def _population(pop:nn.Module,kind:str,y:Tensor,state:Tensor|None=None,timestep:Tensor|None=None)->Tensor:
    with torch.no_grad():
        return pop(y) if kind.endswith("det") else pop.predict_x0(state,y,pop.context(len(y)),timestep)


def _training_predictions(model:nn.Module,pop:nn.Module,kind:str,batch:Mapping[str,Any],support:nn.Module,device:torch.device,generator:torch.Generator):
    y=_tensor(batch["y"],device);match=wrong=pi=None;state=timestep=None
    if kind.startswith("support"):
        match,wrong,pi=contexts(support,batch,device)
    if kind.endswith("cdm"):
        target=_tensor(batch["x"],device) if batch["stream"]=="paired" else y
        timestep=torch.randint(0,len(pop.alpha_bar),(len(y),),device=device,generator=generator);noise=torch.randn(target.shape,device=device,generator=generator);alpha=pop.alpha_bar[timestep].view(-1,1,1);state=alpha.sqrt()*target+(1-alpha).sqrt()*noise
    population=_population(pop,kind,y,state,timestep)
    if kind=="support_adapter_det":
        matched=model(y,population,match);wrong_pred=model(y,population,wrong);bypass=model(y,population,match,bypass=True)
    elif kind=="pop_adapter_det":matched=model(y,population);wrong_pred=bypass=population
    elif kind=="support_adapter_cdm":
        matched=model.predict_x0(state,y,population,match,timestep);wrong_pred=model.predict_x0(state,y,population,wrong,timestep);bypass=model.predict_x0(state,y,population,match,timestep,bypass=True)
    else:
        matched=model.predict_x0(state,y,population,model.context(len(y)),timestep);wrong_pred=bypass=population
    return matched,wrong_pred,bypass,population,y,pi


def _paired_loss(pred:Tensor,target:Tensor,cfg:Mapping[str,Any])->Tensor:
    return F.smooth_l1_loss(pred,target)+float(cfg["lambda_spec"])*_spectral(pred,target)+float(cfg["lambda_corr"])*_corr(pred,target)


def _rank(match:Tensor,wrong:Tensor,pop:Tensor,target:Tensor,tau:float)->Tensor:
    denominator=target.abs().sum((1,2)).clamp_min(1e-8);lm=(match-target).abs().sum((1,2))/denominator;lw=(wrong-target).abs().sum((1,2))/denominator;lp=(pop-target).abs().sum((1,2))/denominator
    return (F.softplus((lm-lw)/tau)+F.softplus((lm-lp)/tau)).mean()


def _natural_increment_loss(pred:Tensor,pop:Tensor,y:Tensor,latent:Tensor,pi:Tensor,cfg:Mapping[str,Any])->tuple[Tensor,Tensor]:
    increment=pred-pop;energy=latent.square().mean(1).sqrt();threshold=torch.quantile(energy,.3,dim=1,keepdim=True);low=(energy<=threshold).float()[:,None];low_loss=(increment.abs()*low).sum()/low.sum().clamp_min(1)/increment.shape[1];q=torch.eye(y.shape[1],device=y.device)[None]-pi;complement=torch.einsum("bij,bjt->bit",q,increment).abs().mean();return low_loss,complement


@torch.no_grad()
def predict(model:nn.Module,pop:nn.Module,kind:str,batch:Mapping[str,Any],support:nn.Module,device:torch.device,seed:int,steps:int=10,wrong:bool=False,bypass:bool=False)->Tensor:
    y=_tensor(batch["y"],device)
    if kind.endswith("det"):
        population=pop(y)
        if bypass:return population
        if kind.startswith("pop"):return model(y,population)
        match,other,_=contexts(support,batch,device);return model(y,population,other if wrong else match)
    noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
    if bypass:return pop.sample(y,noise,steps)[0]
    if kind.startswith("pop"):return model.sample(pop,y,noise,steps)[0]
    match,other,_=contexts(support,batch,device);return model.sample(pop,y,other if wrong else match,noise,steps)[0]


@torch.no_grad()
def validate(model:nn.Module,pop:nn.Module,kind:str,bank:Mapping[str,dict[str,Any]],support:nn.Module,device:torch.device,seed:int,steps:int)->dict[str,float]:
    paired=bank["paired"];match=predict(model,pop,kind,paired,support,device,seed,steps);wrong=predict(model,pop,kind,paired,support,device,seed,steps,wrong=True);bypass=predict(model,pop,kind,paired,support,device,seed,steps,bypass=True);target=_tensor(paired["x"],device);den=target.norm().clamp_min(1e-8);paired_score=float((match-target).norm()/den);wrong_effect=float((wrong-target).norm()/den)-paired_score;population_effect=float((bypass-target).norm()/den)-paired_score
    natural=bank["natural"];nmatch=predict(model,pop,kind,natural,support,device,seed+1,steps);nbypass=predict(model,pop,kind,natural,support,device,seed+1,steps,bypass=True);ny=_tensor(natural["y"],device);latent=_tensor(natural["latent"],device);_,_,pi=contexts(support,natural,device);low,q=_natural_increment_loss(nmatch,nbypass,ny,latent,pi,{"lambda_low":1,"lambda_Q":1})
    return {"paired_clean_rrmse":paired_score,"match_wrong_utility":wrong_effect,"match_population_utility":population_effect,"natural_increment_low":float(low),"natural_increment_complement":float(q),"joint":paired_score+float(low)+float(q)-.05*(wrong_effect+population_effect)}


def train(kind:str,fold:int,seed:int,cfg:Mapping[str,Any],data:Mapping[str,Any],fold_cfg:Mapping[str,Any],output:Path,pop_checkpoint:Path,support_checkpoint:Path,resume:bool=False)->dict[str,Any]:
    if kind not in KINDS:raise ValueError(kind)
    device=torch.device("cuda");pop,pop_state=load_population(pop_checkpoint,device);support,support_state=load_support(support_checkpoint,device);freeze(pop);freeze(support)
    model=KINDS[kind](int(cfg["width"])).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));sampler=SupportSetEpisodeSampler(data,fold_cfg,"train",seed);validation_sampler=SupportSetEpisodeSampler(data,fold_cfg,"validation",seed+109);validation={"paired":validation_sampler.sample_paired(48,float(cfg["identity_fraction"])),"natural":validation_sampler.sample_natural(48)};generator=torch.Generator(device=device).manual_seed(seed+929);curves=[];best={"paired":float("inf"),"context":float("inf"),"natural":float("inf"),"joint":float("inf")};bad=0;start=0;last=output/"last.pt"
    if resume and last.is_file():
        state=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"]);ema.load_state_dict(state["ema"]);sampler.set_state(state["paired_natural_rng"]);generator.set_state(state["diffusion_rng"].cpu());curves=state["curves"];best=state["best"];bad=state["bad"];start=int(state["step"])
    started=time.time();maximum=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"])
    for step in range(start,maximum):
        natural=bool(sampler.rng.random()<float(cfg["natural_fraction"]));batch=sampler.sample_natural(int(cfg["batch_size"])) if natural else sampler.sample_paired(int(cfg["batch_size"]),float(cfg["identity_fraction"]));match,wrong,bypass,population,y,pi=_training_predictions(model,pop,kind,batch,support,device,generator)
        if natural:
            if kind.startswith("support"):
                low,q=_natural_increment_loss(match,population,y,_tensor(batch["latent"],device),pi,cfg);loss=float(cfg["lambda_low"])*low+float(cfg["lambda_Q"])*q
            else:loss=(match-population).abs().mean()*float(cfg["lambda_low"])
        else:
            target=_tensor(batch["x"],device);loss=_paired_loss(match,target,cfg)
            if kind.startswith("support"):loss=loss+float(cfg["lambda_ctx"])*_rank(match,wrong,bypass,target,float(cfg["tau"]))
            identity=torch.as_tensor([bool(m["zero_artifact"]) for m in batch["meta"]],device=device)
            if identity.any():loss=loss+float(cfg["lambda_id"])*(match[identity]-population[identity]).abs().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise RuntimeError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            metrics=validate(_ema_model(model,ema),pop,kind,validation,support,device,seed+step,int(cfg["ddim_steps"]));curves.append({"step":step+1,"loss":float(loss.detach()),**metrics});payload={"model":model.state_dict(),"ema":ema.state_dict(),"kind":kind,"fold":fold,"seed":seed,"step":step+1,"config":dict(cfg),"population_checkpoint":str(pop_checkpoint),"population_checkpoint_sha":pop_state.get("checkpoint_sha"),"support_checkpoint":str(support_checkpoint),"support_checkpoint_sha":support_state.get("checkpoint_sha")}
            scores={"paired":metrics["paired_clean_rrmse"],"context":-(metrics["match_wrong_utility"]+metrics["match_population_utility"]),"natural":metrics["natural_increment_low"]+metrics["natural_increment_complement"],"joint":metrics["joint"]}
            for key,name in (("paired","best_paired.pt"),("context","best_context.pt"),("natural","best_natural.pt"),("joint","best_joint.pt")):
                if scores[key]<best[key]:best[key]=scores[key];_save(output/name,payload);bad=0 if key=="joint" else bad
                elif key=="joint":bad+=1
            _save(last,{**payload,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"global_step":step+1,"paired_natural_rng":sampler.state(),"support_rng":sampler.state(),"wrong_support_rng":sampler.state(),"diffusion_rng":generator.get_state(),"curves":curves,"best":best,"bad":bad})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":kind,"fold":fold,"seed":seed,"updates":step+1,"best":best,"curve":curves,"checkpoint":str(output/"best_joint.pt"),"population_checkpoint":str(pop_checkpoint),"support_checkpoint":str(support_checkpoint),"adapter_parameters":sum(p.numel() for p in model.parameters()),"frozen_population_parameters":sum(p.numel() for p in pop.parameters()),"training_seconds":time.time()-started,"device":torch.cuda.get_device_name(0)}


def load(path:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);model=KINDS[state["kind"]](int(state["config"]["width"])).to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);return model.eval(),state


__all__=["train","load","predict","contexts","validate","KINDS"]
