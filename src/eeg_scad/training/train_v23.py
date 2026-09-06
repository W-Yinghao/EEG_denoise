from __future__ import annotations
import copy,json,time
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import torch
from torch import Tensor,nn
from eeg_scad.context.operator_factorization import decode_torch
from eeg_scad.data.online_counterfactual import OnlineCounterfactualSampler,generate_validation_bank
from eeg_scad.models.of_deterministic import OFDeterministic,PopulationMarginalDET
from eeg_scad.models.of_residual_diffusion import OFResidualDiffusion,PopulationMarginalSCAD,OFSCADConfig
from eeg_scad.models.v22_fixed_fullfield import V22FixedFullField
from eeg_scad.models.scad_artifact_diffusion import SCADConfig
from eeg_scad.training.checkpoint import EMA


def parameter_count(model:nn.Module)->int:return sum(p.numel() for p in model.parameters() if p.requires_grad)
def _t(value:np.ndarray,device:torch.device)->Tensor:return torch.from_numpy(np.asarray(value)).to(device)
def _rms(value:Tensor)->Tensor:return value.square().mean().sqrt()


def coefficient_stats(sampler:OnlineCounterfactualSampler,samples:int=512)->tuple[np.ndarray,np.ndarray]:
    bank=generate_validation_bank(sampler,samples,sampler.seed+701);values=np.concatenate((bank["z_match"],bank["z_pop"],bank["z_wrong"]),axis=0);mean=np.mean(values,axis=(0,2));std=np.maximum(np.std(values,axis=(0,2)),1e-4);return mean.astype(np.float32),std.astype(np.float32)


def _standard(value:Tensor,mean:Tensor,std:Tensor)->Tensor:return (value-mean[:,:,None])/std[:,:,None]
def _inverse(value:Tensor,mean:Tensor,std:Tensor)->Tensor:return value*std[:,:,None]+mean[:,:,None]


def build_model(kind:str,cfg:Mapping[str,Any])->nn.Module:
    if kind=="v22_fixed":return V22FixedFullField(SCADConfig(base_channels=int(cfg["base_channels"])))
    if kind=="of_det":return OFDeterministic(int(cfg["base_channels"]))
    if kind=="pop_marginal_det":return PopulationMarginalDET(int(cfg["base_channels"]))
    dc=OFSCADConfig(base=int(cfg["base_channels"]),timesteps=int(cfg["diffusion_steps"]),ddim_steps=int(cfg["ddim_steps"]))
    return OFResidualDiffusion(dc) if kind=="of_scad" else PopulationMarginalSCAD(dc)


def _of_inputs(batch:Mapping[str,np.ndarray],label:str,device:torch.device,mean:Tensor,std:Tensor)->dict[str,Tensor]:
    return {"y":_t(batch["y"],device),"artifact":_t(batch["artifact"],device),"basis":_t(batch[f"basis_{label}"],device),"summary":_t(batch[f"summary_{label}"],device),"q":_standard(_t(batch[f"q_{label}"],device),mean,std),"projected":_t(batch[f"projected_{label}"],device),"target":_standard(_t(batch[f"z_{label}"],device),mean,std)}


@torch.no_grad()
def validate_det(model:nn.Module,bank:Mapping[str,np.ndarray],kind:str,device:torch.device,mean:Tensor,std:Tensor,batch_size:int=32)->dict[str,float]:
    label="pop" if kind=="pop_marginal_det" else "match";values=[];coef=[]
    for start in range(0,len(bank["y"]),batch_size):
        subset={k:v[start:start+batch_size] if isinstance(v,np.ndarray) else v for k,v in bank.items()};b=_of_inputs(subset,label,device,mean,std);z=model(b["y"],b["q"],b["projected"],b["summary"]);raw=_inverse(z,mean,std);artifact=decode_torch(b["basis"],raw);values.append(float((artifact-b["artifact"]).square().sum()));coef.append(float((z-b["target"]).square().sum()))
    denom=sum(np.prod(v.shape[1:]) for v in np.array_split(bank["artifact"],range(batch_size,len(bank["artifact"]),batch_size)));return {"artifact_mse":sum(values)/denom,"coefficient_mse":sum(coef)/(len(bank["y"])*8*bank["y"].shape[-1])}


@torch.no_grad()
def validate_scad(model:OFResidualDiffusion,anchor:nn.Module,bank:Mapping[str,np.ndarray],kind:str,device:torch.device,mean:Tensor,std:Tensor,seed:int,batch_size:int=24)->dict[str,float]:
    label="pop" if kind=="pop_marginal_scad" else "match";sse=0.;coef_sse=0.;count=0
    for start in range(0,len(bank["y"]),batch_size):
        subset={k:v[start:start+batch_size] if isinstance(v,np.ndarray) else v for k,v in bank.items()};b=_of_inputs(subset,label,device,mean,std);zdet=anchor(b["y"],b["q"],b["projected"],b["summary"]);noise=torch.randn(zdet.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+start));res,_=model.sample(b["y"],b["q"],b["projected"],zdet,b["summary"],noise,25);z=zdet+res;artifact=decode_torch(b["basis"],_inverse(z,mean,std));sse+=float((artifact-b["artifact"]).square().sum());coef_sse+=float((z-b["target"]).square().sum());count+=artifact.numel()
    return {"full_sampling_artifact_mse":sse/count,"full_sampling_coefficient_mse":coef_sse/(len(bank["y"])*8*bank["y"].shape[-1])}


def _save(path:Path,payload:dict[str,Any])->None:path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)


def train_v22_fixed(fold:int,seed:int,cfg:Mapping[str,Any],data:Mapping[str,Any],fold_cfg:Mapping[str,Any],checkpoint_root:Path,resume:bool=False)->dict[str,Any]:
    device=torch.device("cuda");sampler=OnlineCounterfactualSampler(data,fold_cfg,"train",seed);validation=generate_validation_bank(OnlineCounterfactualSampler(data,fold_cfg,"validation",seed+11),192,seed+900);model=build_model("v22_fixed",cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));gen=torch.Generator(device=device).manual_seed(seed+33);curves=[];best=float("inf");bad=0;start=0;last=checkpoint_root/"last.pt"
    if resume and last.is_file():
        s=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(s["model"]);opt.load_state_dict(s["optimizer"]);scheduler.load_state_dict(s["scheduler"]);ema.load_state_dict(s["ema"]);gen.set_state(s["diffusion_rng"].cpu());sampler.set_state(s["data_rng"]);start=s["step"];best=s["best"];curves=s["curves"];bad=s["bad"]
    started=time.time();max_updates=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"])
    for step in range(start,max_updates):
        match=sampler.sample(int(cfg["batch_size"])//2);pop=sampler.sample(int(cfg["batch_size"])-int(cfg["batch_size"])//2,pop_consistent=True);y=_t(np.concatenate((match["y"],pop["y"])),device);a=_t(np.concatenate((match["artifact"],pop["artifact"])),device);ctx=_t(np.concatenate((match["context_match"],pop["context_pop"])),device);opt.zero_grad(set_to_none=True);base,extra=model.training_loss(a,y,ctx,gen);n=len(match["y"]);state=extra["state"][:n];t=extra["timestep"][:n];correct=model.predict(state,y[:n],t,_t(match["context_match"],device))[1];wrong=model.predict(state,y[:n],t,_t(match["context_wrong"],device))[1];mse_c=(correct-_t(match["artifact"],device)).square().mean((1,2));mse_w=(wrong-_t(match["artifact"],device)).square().mean((1,2));rank=torch.relu(float(cfg["context_margin"])+mse_c-mse_w).mean();zero=torch.linalg.vector_norm(a.flatten(1),dim=1)==0;identity=extra["predicted_x0"][zero].square().mean() if bool(zero.any()) else base*0;loss=base+float(cfg["lambda_ctx"])*rank+float(cfg["lambda_id"])*identity;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            eval_model=copy.deepcopy(model).eval();EMA(eval_model).load_state_dict(ema.state_dict());temp=EMA(eval_model);temp.load_state_dict(ema.state_dict());temp.copy_to(eval_model);vb={k:_t(v[:96],device) for k,v in validation.items() if isinstance(v,np.ndarray)};noise=torch.randn(vb["artifact"].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+1900));pred=eval_model.sample(vb["y"],vb["context_match"],noise,25)[0];val=float((pred-vb["artifact"]).square().mean());curves.append({"step":step+1,"base_loss":float(base.detach()),"ranking_loss":float(rank.detach()),"weighted_ranking":float(cfg["lambda_ctx"])*float(rank.detach()),"validation_sampling_mse":val});improved=val<best-1e-7
            if improved:best=val;bad=0;_save(checkpoint_root/"best_sampling.pt",{"model":model.state_dict(),"ema":ema.state_dict(),"config":dict(cfg),"kind":"v22_fixed","fold":fold,"seed":seed,"step":step+1})
            else:bad+=1
            payload={"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"diffusion_rng":gen.get_state(),"data_rng":sampler.state(),"wrong_owner_rng":sampler.state(),"mixture_rng":sampler.state(),"step":step+1,"best":best,"bad":bad,"curves":curves,"config":dict(cfg),"kind":"v22_fixed","fold":fold,"seed":seed};_save(last,payload)
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":"v22_fixed","fold":fold,"seed":seed,"updates":step+1,"parameters":parameter_count(model),"best_validation":best,"curve":curves,"checkpoint":str(checkpoint_root/"best_sampling.pt"),"last_checkpoint":str(last),"training_seconds":time.time()-started,"wrong_base_loss_proportion":0.,"match_base_loss_proportion":.5,"pop_base_loss_proportion":.5,"device":torch.cuda.get_device_name(0)}


def train_det(kind:str,fold:int,seed:int,cfg:Mapping[str,Any],data:Mapping[str,Any],fold_cfg:Mapping[str,Any],checkpoint_root:Path,resume:bool=False)->dict[str,Any]:
    device=torch.device("cuda");sampler=OnlineCounterfactualSampler(data,fold_cfg,"train",seed);mean_np,std_np=coefficient_stats(sampler);mean=_t(mean_np[None],device);std=_t(std_np[None],device);validation=generate_validation_bank(OnlineCounterfactualSampler(data,fold_cfg,"validation",seed+11),192,seed+901);model=build_model(kind,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));curves=[];best_coef=best_art=float("inf");bad=0;start=0;last=checkpoint_root/"last.pt"
    if resume and last.is_file():
        s=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(s["model"]);opt.load_state_dict(s["optimizer"]);scheduler.load_state_dict(s["scheduler"]);ema.load_state_dict(s["ema"]);sampler.set_state(s["data_rng"]);start=s["step"];best_coef=s["best_coef"];best_art=s["best_art"];curves=s["curves"];bad=s["bad"]
    started=time.time();max_updates=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"]);label="pop" if kind=="pop_marginal_det" else "match"
    for step in range(start,max_updates):
        batch=sampler.sample(int(cfg["batch_size"]));b=_of_inputs(batch,label,device,mean,std);opt.zero_grad(set_to_none=True);z=model(b["y"],b["q"],b["projected"],b["summary"]);artifact=decode_torch(b["basis"],_inverse(z,mean,std));coef=(z-b["target"]).square().mean();art=(artifact-b["artifact"]).square().mean();zero=torch.linalg.vector_norm(b["artifact"].flatten(1),dim=1)==0;identity=artifact[zero].square().mean() if bool(zero.any()) else art*0;rank=art*0
        if kind=="of_det":
            w=_of_inputs(batch,"wrong",device,mean,std);zw=model(w["y"],w["q"],w["projected"],w["summary"]);aw=decode_torch(w["basis"],_inverse(zw,mean,std));per_c=(artifact-b["artifact"]).square().mean((1,2));per_w=(aw-b["artifact"]).square().mean((1,2));keep=~zero;rank=torch.relu(float(cfg["context_margin"])+per_c[keep]-per_w[keep]).mean() if bool(keep.any()) else art*0
        loss=coef+float(cfg["lambda_artifact"])*art+float(cfg.get("lambda_ctx",0))*rank+float(cfg["lambda_id"])*identity;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            em=copy.deepcopy(model).eval();holder=EMA(em);holder.load_state_dict(ema.state_dict());holder.copy_to(em);metrics=validate_det(em,validation,kind,device,mean,std);curves.append({"step":step+1,"coefficient_loss":float(coef.detach()),"artifact_loss":float(art.detach()),"ranking_loss":float(rank.detach()),**metrics});improved=metrics["artifact_mse"]<best_art-1e-7
            if metrics["coefficient_mse"]<best_coef:best_coef=metrics["coefficient_mse"];_save(checkpoint_root/"best_coefficient.pt",{"model":model.state_dict(),"ema":ema.state_dict(),"mean":mean_np,"std":std_np,"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed,"step":step+1})
            if improved:best_art=metrics["artifact_mse"];bad=0;_save(checkpoint_root/"best_artifact.pt",{"model":model.state_dict(),"ema":ema.state_dict(),"mean":mean_np,"std":std_np,"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed,"step":step+1})
            else:bad+=1
            _save(last,{"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"mean":mean_np,"std":std_np,"data_rng":sampler.state(),"mixture_rng":sampler.state(),"wrong_owner_rng":sampler.state(),"diffusion_rng":None,"step":step+1,"best_coef":best_coef,"best_art":best_art,"bad":bad,"curves":curves,"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":kind,"fold":fold,"seed":seed,"updates":step+1,"parameters":parameter_count(model),"best_validation_coefficient":best_coef,"best_validation_artifact":best_art,"curve":curves,"checkpoint":str(checkpoint_root/"best_artifact.pt"),"last_checkpoint":str(last),"training_seconds":time.time()-started,"wrong_base_loss_proportion":0.,"device":torch.cuda.get_device_name(0)}


def load_det_checkpoint(path:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);model=build_model(state["kind"],state["config"]).to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);model.eval();return model,state


def train_scad(kind:str,fold:int,seed:int,cfg:Mapping[str,Any],data:Mapping[str,Any],fold_cfg:Mapping[str,Any],checkpoint_root:Path,anchor_path:Path,resume:bool=False)->dict[str,Any]:
    device=torch.device("cuda");anchor,anchor_state=load_det_checkpoint(anchor_path,device);mean_np=np.asarray(anchor_state["mean"]);std_np=np.asarray(anchor_state["std"]);mean=_t(mean_np[None],device);std=_t(std_np[None],device);sampler=OnlineCounterfactualSampler(data,fold_cfg,"train",seed);validation=generate_validation_bank(OnlineCounterfactualSampler(data,fold_cfg,"validation",seed+11),96,seed+902);model=build_model(kind,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"]));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,int(cfg["maximum_updates"]));ema=EMA(model,float(cfg["ema"]));gen=torch.Generator(device=device).manual_seed(seed+88);curves=[];best=float("inf");bad=0;start=0;last=checkpoint_root/"last.pt";label="pop" if kind=="pop_marginal_scad" else "match"
    if resume and last.is_file():
        s=torch.load(last,map_location=device,weights_only=False);model.load_state_dict(s["model"]);opt.load_state_dict(s["optimizer"]);scheduler.load_state_dict(s["scheduler"]);ema.load_state_dict(s["ema"]);gen.set_state(s["diffusion_rng"].cpu());sampler.set_state(s["data_rng"]);start=s["step"];best=s["best"];curves=s["curves"];bad=s["bad"]
    started=time.time();max_updates=int(cfg["maximum_updates"]);interval=int(cfg["validation_interval"])
    for step in range(start,max_updates):
        batch=sampler.sample(int(cfg["batch_size"]));b=_of_inputs(batch,label,device,mean,std)
        with torch.no_grad():zdet=anchor(b["y"],b["q"],b["projected"],b["summary"])
        residual=b["target"]-zdet;opt.zero_grad(set_to_none=True);base,extra=model.training_loss(residual,b["y"],b["q"],b["projected"],zdet,b["summary"],gen);z=zdet+extra["predicted_x0"];artifact=decode_torch(b["basis"],_inverse(z,mean,std));art=(artifact-b["artifact"]).square().mean();loss=base+float(cfg["lambda_decoded_artifact"])*art;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();scheduler.step();ema.update(model)
        if (step+1)%interval==0:
            em=copy.deepcopy(model).eval();holder=EMA(em);holder.load_state_dict(ema.state_dict());holder.copy_to(em);metrics=validate_scad(em,anchor,validation,kind,device,mean,std,seed);curves.append({"step":step+1,"residual_loss":float(base.detach()),"decoded_artifact_loss":float(art.detach()),**metrics});val=metrics["full_sampling_artifact_mse"];improved=val<best-1e-7
            if improved:best=val;bad=0;_save(checkpoint_root/"best_sampling.pt",{"model":model.state_dict(),"ema":ema.state_dict(),"mean":mean_np,"std":std_np,"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed,"step":step+1,"anchor":str(anchor_path)})
            else:bad+=1
            _save(last,{"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),"ema":ema.state_dict(),"mean":mean_np,"std":std_np,"diffusion_rng":gen.get_state(),"data_rng":sampler.state(),"mixture_rng":sampler.state(),"wrong_owner_rng":sampler.state(),"step":step+1,"best":best,"bad":bad,"curves":curves,"config":dict(cfg),"kind":kind,"fold":fold,"seed":seed,"anchor":str(anchor_path)})
            if step+1>=int(cfg["minimum_updates"]) and bad>=int(cfg["early_stopping_patience"]):break
    return {"kind":kind,"fold":fold,"seed":seed,"updates":step+1,"parameters":parameter_count(model),"anchor_parameters":parameter_count(anchor),"best_validation_sampling_artifact":best,"curve":curves,"checkpoint":str(checkpoint_root/"best_sampling.pt"),"last_checkpoint":str(last),"training_seconds":time.time()-started,"device":torch.cuda.get_device_name(0)}


def load_scad_checkpoint(path:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);model=build_model(state["kind"],state["config"]).to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);model.eval();return model,state

