from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any,Mapping

import numpy as np
import torch
from torch import nn

from eeg_scad.data.counterfactual_pairs import load_training_split
from eeg_scad.models.deterministic_artifact_unet import DeterministicArtifactEstimator
from eeg_scad.models.scad_artifact_diffusion import SCADArtifactDiffusion,SCADConfig
from .checkpoint import EMA,clone_with_ema
from .losses import ranking_loss,zero_identity_loss


def _batch(arrays:Mapping[str,np.ndarray],indices:np.ndarray,device:torch.device)->dict[str,torch.Tensor]:return {k:torch.from_numpy(np.asarray(v[indices])).to(device) for k,v in arrays.items()}
def parameter_count(model:nn.Module)->int:return sum(v.numel() for v in model.parameters() if v.requires_grad)


def _model(kind:str,cfg:Mapping[str,Any])->nn.Module:
    common=dict(channels=46,base_channels=int(cfg["base_channels"]),context_input_dim=int(cfg["context_input_dim"]),context_hidden_dim=int(cfg["context_hidden_dim"]),context_dim=int(cfg["context_dim"]))
    if kind=="det":return DeterministicArtifactEstimator(**common)
    return SCADArtifactDiffusion(SCADConfig(**common,timesteps=int(cfg["diffusion_steps"]),ddim_steps=int(cfg["ddim_steps"]),parameterization=str(cfg["artifact_parameterization"])))


@torch.no_grad()
def _validation(model:nn.Module,kind:str,arrays:Mapping[str,np.ndarray],device:torch.device,seed:int)->float:
    n=min(len(arrays["y"]),64);idx=np.arange(n);b=_batch(arrays,idx,device);context=b["context_match"]
    if kind=="det":pred=model(b["y"],context)
    else:
        generator=torch.Generator(device=device).manual_seed(seed+991);t=torch.full((n,),500,device=device,dtype=torch.long);noise=torch.randn(b["artifact"].shape,device=device,generator=generator);_,extra=model.training_loss(b["artifact"],b["y"],context,generator,t,noise);pred=extra["predicted_x0"]
    return float((pred-b["artifact"]).square().mean())


def train_fold(kind:str,fold:int,seed:int,cfg:Mapping[str,Any],derived:Path,checkpoint:Path,resume:bool=False)->dict[str,Any]:
    device=torch.device("cuda");torch.manual_seed(seed+fold*1000);np_rng=np.random.Generator(np.random.PCG64DXSM(seed+fold*1000));torch_generator=torch.Generator(device=device).manual_seed(seed+fold*1000+7)
    train=load_training_split(derived,fold,"train");validation=load_training_split(derived,fold,"validation");model=_model(kind,cfg).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));ema=EMA(model,float(cfg["ema"]));start=0;best=float("inf");curves=[]
    if resume and checkpoint.is_file():
        state=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);ema.load_state_dict(state["ema"]);start=int(state["step"]);best=float(state["best"]);np_rng.bit_generator.state=state["np_rng"];torch_generator.set_state(state["torch_generator"]);torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
    batch_size=int(cfg["batch_size"]);updates=int(cfg["updates"]);started=time.time();model.train()
    for step in range(start,updates):
        indices=np_rng.integers(0,len(train["y"]),size=batch_size);b=_batch(train,indices,device);choice=np_rng.random(batch_size);context=b["context_match"].clone();context[choice>=.4]=b["context_pop"][choice>=.4];context[choice>=.65]=b["context_wrong"][choice>=.65];context[choice>=.9]=0
        drop=np_rng.random(batch_size)<float(cfg["context_dropout"]);context[torch.as_tensor(drop,device=device)]=0;optimizer.zero_grad(set_to_none=True)
        if kind=="det":
            pred=model(b["y"],context);correct=model(b["y"],b["context_match"]);wrong=model(b["y"],b["context_wrong"]);base=(pred-b["artifact"]).square().mean()
        else:
            base,extra=model.training_loss(b["artifact"],b["y"],context,torch_generator);t=extra["timestep"];noise=extra["noise"];state=extra["state"];correct=model.predict(state,b["y"],t,b["context_match"])[1];wrong=model.predict(state,b["y"],t,b["context_wrong"])[1];pred=extra["predicted_x0"]
        rank=ranking_loss(correct,wrong,b["artifact"],float(cfg["context_margin"]));identity=zero_identity_loss(pred,b["artifact"]);loss=base+float(cfg["lambda_ctx"])*rank+float(cfg["lambda_id"])*identity
        loss.backward();grad_finite=all(p.grad is None or bool(torch.all(torch.isfinite(p.grad))) for p in model.parameters());
        if not grad_finite:raise FloatingPointError("nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();ema.update(model)
        if (step+1)%int(cfg["validation_interval"])==0 or step+1==updates:
            eval_model=clone_with_ema(model,ema).to(device).eval();val=_validation(eval_model,kind,validation,device,seed);del eval_model;curves.append({"step":step+1,"train_loss":float(loss.detach()),"base_loss":float(base.detach()),"ranking_loss":float(rank.detach()),"identity_loss":float(identity.detach()),"validation_artifact_mse":val})
            best=min(best,val);checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"kind":kind,"fold":fold,"seed":seed,"config":dict(cfg),"model":model.state_dict(),"optimizer":optimizer.state_dict(),"ema":ema.state_dict(),"step":step+1,"best":best,"np_rng":np_rng.bit_generator.state,"torch_generator":torch_generator.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"curves":curves},checkpoint)
    return {"kind":kind,"fold":fold,"seed":seed,"updates":updates,"parameters":parameter_count(model),"best_validation_artifact_mse":best,"training_seconds":time.time()-started,"checkpoint":str(checkpoint),"curve":curves,"finite":True,"device":torch.cuda.get_device_name(0)}


def load_ema_model(kind:str,checkpoint:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(checkpoint,map_location=device,weights_only=False);model=_model(kind,state["config"]).to(device);EMA(model).load_state_dict(state["ema"]);ema=EMA(model);ema.load_state_dict(state["ema"]);ema.copy_to(model);model.eval();return model,state

