"""Conditional one-seed temporal-support diffusion screen, route-gated by J5."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any,Mapping

import numpy as np
import torch

from eeg_cgdr.experiments.mobile_bci_headroom_runner import (_cache,_normalization,_read_csv,_score,_support,_training_records,_write_csv)
from eeg_cgdr.models.temporal_support_diffusion import TemporalDiffusionConfig,TemporalSupportCorrectionDiffusion


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _ema_update(ema:torch.nn.Module,model:torch.nn.Module,decay:float=.999)->None:
    with torch.no_grad():
        for left,right in zip(ema.parameters(),model.parameters()):left.mul_(decay).add_(right,alpha=1-decay)


def _kwargs(support_tuple:tuple[np.ndarray,np.ndarray,np.ndarray],device:torch.device,present:float=1.0)->dict[str,torch.Tensor]:
    eeg,imu,eog=support_tuple;return {"support_eeg":torch.tensor(eeg[None],device=device),"support_imu":torch.tensor(imu[None],device=device),"support_eog":torch.tensor(eog[None],device=device),"modality_present":torch.tensor([[1.,1.,0.]],device=device),"context_present":torch.tensor([present],device=device)}


def _sample(model:TemporalSupportCorrectionDiffusion,query:np.ndarray,support_tuple:tuple[np.ndarray,np.ndarray,np.ndarray],mean:np.ndarray,std:np.ndarray,device:torch.device,seed:int,present:float=1.0,k:int=8)->tuple[np.ndarray,np.ndarray]:
    window=512;outputs=[];variances=[];generator=torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for start in range(0,query.shape[1],window):
        length=min(window,query.shape[1]-start);value=query[:,start:start+length]
        if length<window:value=np.pad(value,((0,0),(0,window-length)))
        observed=torch.tensor(((value-mean[:,None])/std[:,None])[None],device=device)
        samples=model.sample(observed,_kwargs(support_tuple,device,present),generator,k=k)[:,0].cpu().numpy()*std[None,:,None]
        correction=samples.mean(0);outputs.append((value[:,:length]+correction[:,:length]).astype(np.float32));variances.append(samples[:,:,:length].std(0).astype(np.float32))
    return np.concatenate(outputs,axis=1),np.concatenate(variances,axis=1)


def _validation(model:TemporalSupportCorrectionDiffusion,config:Mapping[str,Any],participants:list[str],mean:np.ndarray,std:np.ndarray,device:torch.device,seed:int)->float:
    rng=np.random.default_rng(seed);records=_training_records(config,participants);window=512;support_length=int(float(config["preprocessing"]["support_budget_seconds"])*float(config["preprocessing"]["target_sampling_rate_hz"]));values=[]
    for index in range(8):
        participant,session,task=records[int(rng.integers(len(records)))];eeg=_cache(config,participant,session,task,"eeg");target=_cache(config,participant,session,task,"clean_proxy");start=int(rng.integers(0,max(1,eeg.shape[1]-window)));query=np.asarray(eeg[:,start:start+window]);truth=np.asarray(target[:,start:start+window]);support=_support(config,participant,"ses-02",task,support_length,mean,std);prediction,_=_sample(model,query,support,mean,std,device,seed+index,k=8);values.append(float(np.mean((prediction-truth)**2)))
    return float(np.mean(values))


def _train(config:Mapping[str,Any],fold:int,device:torch.device)->tuple[Path,dict[str,Any]]:
    routing=json.loads((CODE_ROOT/str(config["output_root"])/"aggregate/routing_decision.json").read_text())
    if not routing.get("temporal_diffusion_one_seed_authorized"):raise RuntimeError("J5 did not authorize temporal diffusion")
    fold_rows=_read_csv(CODE_ROOT/str(config["output_root"])/"metadata/development_cv_folds.csv");training=sorted(r["participant"] for r in fold_rows if int(r["fold"])==fold and r["split"]=="training");validation=training[-2:];fit=training[:-2];records=_training_records(config,fit);mean,std=_normalization(config,fit);seed=int(config["temporal_diffusion"]["first_seed"]);torch.manual_seed(seed+fold);torch.cuda.manual_seed_all(seed+fold);np.random.seed(seed+fold);random.seed(seed+fold)
    diffusion_config=TemporalDiffusionConfig(timesteps=int(config["temporal_diffusion"]["diffusion_timesteps"]),ddim_steps=int(config["temporal_diffusion"]["ddim_steps"]),posterior_samples=int(config["temporal_diffusion"]["posterior_samples"]));model=TemporalSupportCorrectionDiffusion(diffusion_config).to(device);ema=TemporalSupportCorrectionDiffusion(diffusion_config).to(device);ema.load_state_dict(model.state_dict());ema.requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["headroom"]["learning_rate"]));generator=torch.Generator(device=device).manual_seed(seed+fold);checkpoint=CODE_ROOT/"results/cgdr/temporal_support_diffusion_v4"/f"checkpoints/fold_{fold:02d}/training.pt";checkpoint.parent.mkdir(parents=True,exist_ok=True);start=0;best_loss=float("inf");best_state=None;curve=[]
    if checkpoint.is_file():
        payload=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(payload["model"]);ema.load_state_dict(payload["ema"]);optimizer.load_state_dict(payload["optimizer"]);start=int(payload["step"]);best_loss=float(payload["best_loss"]);best_state=payload.get("best_state");generator.set_state(payload["generator_state"])
    rng=np.random.default_rng(seed+fold);window=512;support_length=int(float(config["preprocessing"]["support_budget_seconds"])*float(config["preprocessing"]["target_sampling_rate_hz"]));maximum=int(config["headroom"]["maximum_updates"]);interval=int(config["headroom"]["validation_interval"]);started=time.perf_counter()
    for step in range(start+1,maximum+1):
        queries=[];targets=[];seegs=[];simus=[];seogs=[];present=[]
        for _ in range(4):
            participant,session,task=records[int(rng.integers(len(records)))];eeg=_cache(config,participant,session,task,"eeg");clean=_cache(config,participant,session,task,"clean_proxy");offset=int(rng.integers(0,max(1,eeg.shape[1]-window)));queries.append((np.asarray(eeg[:,offset:offset+window])-mean[:,None])/std[:,None]);targets.append((np.asarray(clean[:,offset:offset+window])-np.asarray(eeg[:,offset:offset+window]))/std[:,None]);support=_support(config,participant,"ses-02",task,support_length,mean,std);seegs.append(support[0]);simus.append(support[1]);seogs.append(support[2]);present.append(float(rng.random()>=.25))
        observed=torch.tensor(np.stack(queries),device=device);target=torch.tensor(np.stack(targets),device=device);kwargs={"support_eeg":torch.tensor(np.stack(seegs),device=device),"support_imu":torch.tensor(np.stack(simus),device=device),"support_eog":torch.tensor(np.stack(seogs),device=device),"modality_present":torch.tensor([[1.,1.,0.]]*4,device=device),"context_present":torch.tensor(present,device=device)}
        optimizer.zero_grad(set_to_none=True);loss=model.training_loss(target,observed,kwargs,generator);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();_ema_update(ema,model)
        if step%interval==0 or step==maximum:
            validation_loss=_validation(ema,config,validation,mean,std,device,seed+fold+step);curve.append({"step":step,"training_loss":float(loss),"full_ddim_k8_validation_mse":validation_loss})
            if validation_loss<best_loss:best_loss=validation_loss;best_state={key:value.detach().cpu() for key,value in ema.state_dict().items()}
            torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"best_loss":best_loss,"best_state":best_state,"generator_state":generator.get_state()},checkpoint)
    best=checkpoint.parent/"best_ema.pt";torch.save({"ema":best_state,"mean":mean,"std":std,"seed":seed,"fold":fold,"best_validation":best_loss},best);_write_csv(checkpoint.parent/"training_curve.csv",curve);return best,{"checkpoint":str(best),"runtime_seconds":time.perf_counter()-started,"best_validation":best_loss,"fit":fit,"validation":validation}


def run_fold(config:Mapping[str,Any],run_dir:Path,fold:int)->Mapping[str,Any]:
    device=torch.device("cuda",0);checkpoint,training=_train(config,fold,device);payload=torch.load(checkpoint,map_location=device,weights_only=False);diffusion_config=TemporalDiffusionConfig(timesteps=int(config["temporal_diffusion"]["diffusion_timesteps"]),ddim_steps=int(config["temporal_diffusion"]["ddim_steps"]),posterior_samples=int(config["temporal_diffusion"]["posterior_samples"]));model=TemporalSupportCorrectionDiffusion(diffusion_config).to(device);model.load_state_dict(payload["ema"]);mean=np.asarray(payload["mean"]);std=np.asarray(payload["std"]);fold_rows=_read_csv(CODE_ROOT/str(config["output_root"])/"metadata/development_cv_folds.csv");heldout=sorted(r["participant"] for r in fold_rows if int(r["fold"])==fold and r["split"]=="validation");protocols=_read_csv(CODE_ROOT/str(config["output_root"])/"evaluator/frozen_protocol_units.csv");rows=[];output_root=CODE_ROOT/"results/cgdr/temporal_support_diffusion_v4"/f"fold_{fold:02d}";arrays=output_root/"server_arrays";arrays.mkdir(parents=True,exist_ok=True);rate=float(config["preprocessing"]["target_sampling_rate_hz"])
    for protocol in protocols:
        participant=protocol["participant"]
        if participant not in heldout or protocol["status"]!="eligible":continue
        task=protocol["task"];support_session=protocol["support_session"];query_session=protocol["query_session"];support_length=max(512,int(float(protocol["support_end"])*rate));query_start=int(float(protocol.get("query_start",0) or 0)*rate);query=np.asarray(_cache(config,participant,query_session,task,"eeg"),dtype=np.float32)[:,query_start:];matching=_support(config,participant,support_session,task,support_length,mean,std);base_seed=2026080600+int(participant[-2:])*100+fold
        outputs={};uncertainties={};outputs["TEMPORAL-DIFF-MATCH"],uncertainties["MATCH"]=_sample(model,query,matching,mean,std,device,base_seed,present=1,k=8);outputs["TEMPORAL-DIFF-NULL"],uncertainties["NULL"]=_sample(model,query,matching,mean,std,device,base_seed,present=0,k=8)
        for index in (1,2,3):
            donor=protocol[f"wrong_donor_{index}"];path=Path(str(config["derived_root"]))/donor/support_session/task/"eeg.npy"
            if not path.is_file():continue
            outputs[f"TEMPORAL-DIFF-WRONG-{index}"],uncertainties[f"WRONG-{index}"]=_sample(model,query,_support(config,donor,support_session,task,support_length,mean,std),mean,std,device,base_seed,present=1,k=8)
        outputs["TEMPORAL-DIFF-SHUFFLED"],uncertainties["SHUFFLED"]=_sample(model,query,_support(config,participant,support_session,task,support_length,mean,std,reverse=True),mean,std,device,base_seed,present=1,k=8);stem=f"{participant}_{protocol['protocol']}_{task}_{query_session}";np.savez_compressed(arrays/f"{stem}.npz",**outputs)
        imu=np.asarray(_cache(config,participant,query_session,task,"imu"),dtype=np.float32)[:,query_start:]
        for method,output in outputs.items():rows.append({"fold":fold,"participant":participant,"protocol":protocol["protocol"],"task":task,"query_session":query_session,"method":method,"status":"success","query_imu_used_for_inference":False,"outputs_frozen_before_query_imu_scoring":True,"posterior_sd_rms":float(np.sqrt(np.mean(uncertainties[method.replace('TEMPORAL-DIFF-','')]**2))) if method.replace('TEMPORAL-DIFF-','') in uncertainties else float("nan"),**_score(output,query,imu)})
    _write_csv(output_root/"metrics.csv",rows);summary={"status":"completed_one_seed_full_temporal_diffusion_fold","fold":fold,"heldout":heldout,"rows":len(rows),"training":training,"sealed_signal_opened":False};(output_root/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
