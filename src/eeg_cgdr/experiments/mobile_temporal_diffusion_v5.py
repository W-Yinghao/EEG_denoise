"""Direct deterministic/diffusion interaction screen on repaired MobileBCI v5."""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy import signal

from eeg_cgdr.experiments.mobile_bci_headroom_v4 import _erp_readout,_ssvep_readout
from eeg_cgdr.models.temporal_support_conditioner import PopulationCleaner,TemporalSupportCleaner
from eeg_cgdr.models.temporal_support_diffusion import TemporalDiffusionConfig,TemporalSupportCorrectionDiffusion


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_diffusion_v5"))


def _seed_all(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _read_csv(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def _write_csv(path:Path,rows:list[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _cache(config:Mapping[str,Any],participant:str,session:str,task:str,name:str,mmap:bool=True)->np.ndarray:
    return np.load(Path(str(config["derived_root"]))/participant/session/task/f"{name}.npy",mmap_mode="r" if mmap else None)


def _fold_sets(config:Mapping[str,Any],fold:int)->tuple[list[str],list[str],list[str]]:
    development=list(config["split"]["development"]);heldout=list(config["split"]["outer_folds"][fold]);inner=list(config["split"]["inner_validation"][fold]);fit=[p for p in development if p not in heldout and p not in inner];return fit,inner,heldout


def _protocols(config:Mapping[str,Any],participants:list[str])->list[dict[str,str]]:
    rows=_read_csv(CODE_ROOT/str(config["output_root"])/"protocol/frozen_protocol_units.csv")
    result=[row for row in rows if row["status"]=="eligible" and row["participant"] in participants]
    if not result:raise ValueError("no eligible protocol episodes")
    return result


def _normalization(config:Mapping[str,Any],participants:list[str])->tuple[np.ndarray,np.ndarray]:
    total=np.zeros(46);square=np.zeros(46);count=0
    for participant in participants:
        for session in ("ses-02","ses-03","ses-04"):
            for task in ("ERP","SSVEP"):
                path=Path(str(config["derived_root"]))/participant/session/task/"eeg.npy"
                if not path.is_file():continue
                value=np.load(path,mmap_mode="r");total+=value.sum(1,dtype=np.float64);square+=np.square(value,dtype=np.float64).sum(1);count+=value.shape[1]
    if count==0:raise ValueError("no normalization samples")
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,1e-8));return mean.astype(np.float32),std.astype(np.float32)


def _support(config:Mapping[str,Any],participant:str,session:str,task:str,mean:np.ndarray,std:np.ndarray,reverse:bool=False)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    length=int(float(config["preprocessing"]["support_budget_seconds"])*float(config["preprocessing"]["target_sampling_rate_hz"]));eeg=np.asarray(_cache(config,participant,session,task,"eeg")[:,:length],dtype=np.float32);imu=np.asarray(_cache(config,participant,session,task,"imu")[:,:length],dtype=np.float32)
    if eeg.shape[1]!=length or imu.shape[1]!=length:raise ValueError("support does not provide exactly 60 seconds")
    eeg=(eeg-mean[:,None])/std[:,None];imu=(imu-imu.mean(1,keepdims=True))/np.maximum(imu.std(1,keepdims=True),1e-6)
    if reverse:eeg=eeg[:,::-1].copy();imu=imu[:,::-1].copy()
    return eeg,imu,np.zeros((4,length),dtype=np.float32)


def _kwargs(support:tuple[np.ndarray,np.ndarray,np.ndarray],device:torch.device,present:float,batch:int=1)->dict[str,torch.Tensor]:
    eeg,imu,eog=support
    return {"support_eeg":torch.tensor(np.repeat(eeg[None],batch,axis=0),device=device),"support_imu":torch.tensor(np.repeat(imu[None],batch,axis=0),device=device),"support_eog":torch.tensor(np.repeat(eog[None],batch,axis=0),device=device),"modality_present":torch.tensor([[1.,1.,0.]]*batch,device=device),"context_present":torch.full((batch,),present,device=device)}


def _fixed_validation(config:Mapping[str,Any],participants:list[str])->list[tuple[dict[str,str],int]]:
    window=int(float(config["preprocessing"]["window_seconds"])*float(config["preprocessing"]["target_sampling_rate_hz"]));count=int(config["training"]["validation_windows_per_protocol"]);result=[]
    for episode in _protocols(config,participants):
        query_start=int(float(episode["query_start"])*float(config["preprocessing"]["target_sampling_rate_hz"]));length=_cache(config,episode["participant"],episode["query_session"],episode["task"],"eeg").shape[1]
        for offset in range(count):
            start=query_start+offset*window
            if start+window<=length:result.append((episode,start))
    if not result:raise ValueError("no fixed validation windows")
    return result


def _batch(config:Mapping[str,Any],episodes:list[dict[str,str]],mean:np.ndarray,std:np.ndarray,rng:np.random.Generator)->tuple[np.ndarray,np.ndarray,list[tuple[np.ndarray,np.ndarray,np.ndarray]],np.ndarray]:
    batch=int(config["training"]["batch_size"]);rate=float(config["preprocessing"]["target_sampling_rate_hz"]);window=int(float(config["preprocessing"]["window_seconds"])*rate);queries=[];targets=[];supports=[];present=[]
    # Round-robin protocol strata within every batch.
    strata={name:[row for row in episodes if row["protocol"]==name] for name in ("S0_STATIC_XSESSION","S1_MOTION_WITHIN_SESSION","S2_MOTION_XSPEED")}
    for index in range(batch):
        name=tuple(strata)[index%3];available=strata[name] or episodes;episode=available[int(rng.integers(len(available)))];participant=episode["participant"];session=episode["query_session"];task=episode["task"];eeg=_cache(config,participant,session,task,"eeg");clean=_cache(config,participant,session,task,"clean_proxy");minimum=int(float(episode["query_start"])*rate);maximum=max(minimum+1,eeg.shape[1]-window+1);start=int(rng.integers(minimum,maximum));value=np.asarray(eeg[:,start:start+window]);truth=np.asarray(clean[:,start:start+window]);queries.append((value-mean[:,None])/std[:,None]);targets.append((truth-value)/std[:,None]);supports.append(_support(config,participant,episode["support_session"],task,mean,std));present.append(float(rng.random()>=float(config["training"]["context_dropout_probability"])))
    return np.stack(queries).astype(np.float32),np.stack(targets).astype(np.float32),supports,np.asarray(present,dtype=np.float32)


def _stack_kwargs(supports:list[tuple[np.ndarray,np.ndarray,np.ndarray]],present:np.ndarray,device:torch.device)->dict[str,torch.Tensor]:
    return {"support_eeg":torch.tensor(np.stack([v[0] for v in supports]),device=device),"support_imu":torch.tensor(np.stack([v[1] for v in supports]),device=device),"support_eog":torch.tensor(np.stack([v[2] for v in supports]),device=device),"modality_present":torch.tensor([[1.,1.,0.]]*len(supports),device=device),"context_present":torch.tensor(present,device=device)}


def _validate_one(model:torch.nn.Module,kind:str,config:Mapping[str,Any],windows:list[tuple[dict[str,str],int]],mean:np.ndarray,std:np.ndarray,device:torch.device,seed:int)->float:
    window=int(float(config["preprocessing"]["window_seconds"])*float(config["preprocessing"]["target_sampling_rate_hz"]));values=[];model.eval()
    with torch.no_grad():
        for offset,(episode,start) in enumerate(windows):
            eeg=np.asarray(_cache(config,episode["participant"],episode["query_session"],episode["task"],"eeg")[:,start:start+window]);clean=np.asarray(_cache(config,episode["participant"],episode["query_session"],episode["task"],"clean_proxy")[:,start:start+window]);observed=torch.tensor(((eeg-mean[:,None])/std[:,None])[None],device=device);target=(clean-eeg)/std[:,None]
            if kind=="population":prediction=model(observed)[0].cpu().numpy()
            else:
                support=_support(config,episode["participant"],episode["support_session"],episode["task"],mean,std);kwargs=_kwargs(support,device,1.)
                if kind=="deterministic":prediction=model(observed,**kwargs)[0].cpu().numpy()
                else:
                    generator=torch.Generator(device=device).manual_seed(seed+offset);prediction=model.sample(observed,kwargs,generator,k=int(config["diffusion"]["posterior_samples"]))[:,0].mean(0).cpu().numpy()
            values.append(float(np.mean((prediction-target)**2)))
    model.train();return float(np.mean(values))


def _train_model(kind:str,config:Mapping[str,Any],fold:int,seed:int,fit:list[str],inner:list[str],mean:np.ndarray,std:np.ndarray,device:torch.device)->tuple[Path,dict[str,Any]]:
    base=CODE_ROOT/str(config["output_root"])/f"checkpoints/seed_{seed}/fold_{fold:02d}/{kind}";base.mkdir(parents=True,exist_ok=True);checkpoint=base/"resume.pt";best_path=base/"best.pt"
    if kind=="population":model:torch.nn.Module=PopulationCleaner().to(device)
    elif kind=="deterministic":model=TemporalSupportCleaner().to(device)
    else:model=TemporalSupportCorrectionDiffusion(TemporalDiffusionConfig(timesteps=int(config["diffusion"]["timesteps"]),ddim_steps=int(config["diffusion"]["ddim_steps"]),posterior_samples=int(config["diffusion"]["posterior_samples"]),min_snr_gamma=float(config["diffusion"]["min_snr_gamma"]),correction_clip=float(config["diffusion"]["correction_clip"]))).to(device)
    ema=copy.deepcopy(model).requires_grad_(False) if kind=="diffusion" else None;optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["training"]["learning_rate"]));start=0;best=float("inf");best_step=0;bad=0;curve=[];generator=torch.Generator(device=device).manual_seed(seed+fold*100+{"population":1,"deterministic":2,"diffusion":3}[kind]);rng=np.random.default_rng(seed+fold*1000+{"population":1,"deterministic":2,"diffusion":3}[kind]);episodes=_protocols(config,fit);validation=_fixed_validation(config,inner)
    if checkpoint.is_file():
        payload=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"]);start=int(payload["step"]);best=float(payload["best"]);best_step=int(payload["best_step"]);bad=int(payload["bad"]);curve=list(payload["curve"]);generator.set_state(payload["generator_state"]);
        if "numpy_rng_state" in payload:rng.bit_generator.state=payload["numpy_rng_state"]
        if ema is not None:ema.load_state_dict(payload["ema"])
    maximum=int(config["training"]["maximum_updates"]);interval=int(config["training"]["validation_interval"]);patience=int(config["training"]["patience_intervals"]);started=time.perf_counter()
    for step in range(start+1,maximum+1):
        query,target,supports,present=_batch(config,episodes,mean,std,rng);observed=torch.tensor(query,device=device);truth=torch.tensor(target,device=device);optimizer.zero_grad(set_to_none=True)
        if kind=="population":prediction=model(observed);loss=torch.mean((prediction-truth)**2)
        elif kind=="deterministic":prediction=model(observed,**_stack_kwargs(supports,present,device));loss=torch.mean((prediction-truth)**2)
        else:loss=model.training_loss(truth,observed,_stack_kwargs(supports,present,device),generator)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for left,right in zip(ema.parameters(),model.parameters()):left.mul_(.999).add_(right,alpha=.001)
        if step%interval==0 or step==maximum:
            selected=ema if ema is not None else model;validation_loss=_validate_one(selected,kind,config,validation,mean,std,device,seed+fold*10000+step);curve.append({"step":step,"training_loss":float(loss.detach()),"restoration_validation_mse":validation_loss,"kind":kind})
            improved=validation_loss<best-1e-8
            if improved:best=validation_loss;best_step=step;bad=0;torch.save({"state":{key:value.detach().cpu() for key,value in selected.state_dict().items()},"mean":mean,"std":std,"kind":kind,"best_step":best_step,"best_validation":best,"fit":fit,"inner_validation":inner},best_path)
            else:bad+=1
            torch.save({"model":model.state_dict(),"ema":ema.state_dict() if ema is not None else None,"optimizer":optimizer.state_dict(),"step":step,"best":best,"best_step":best_step,"bad":bad,"curve":curve,"generator_state":generator.get_state(),"numpy_rng_state":rng.bit_generator.state},checkpoint)
            if bad>=patience:break
    _write_csv(base/"training_curve.csv",curve);return best_path,{"kind":kind,"best_step":best_step,"best_validation":best,"updates":curve[-1]["step"] if curve else start,"runtime_seconds":time.perf_counter()-started}


def _window_values(query:np.ndarray,window:int=512)->tuple[list[np.ndarray],list[int]]:
    values=[];lengths=[]
    for start in range(0,query.shape[1],window):
        length=min(window,query.shape[1]-start);value=query[:,start:start+length]
        if length<window:value=np.pad(value,((0,0),(0,window-length)))
        values.append(value);lengths.append(length)
    return values,lengths


def _predict_one(model:torch.nn.Module,kind:str,query:np.ndarray,support:tuple[np.ndarray,np.ndarray,np.ndarray],mean:np.ndarray,std:np.ndarray,device:torch.device,present:float=1.)->np.ndarray:
    values,lengths=_window_values(query);outputs=[];model.eval()
    with torch.no_grad():
        tokens=model.encode_support(**_kwargs(support,device,present)) if kind=="deterministic" else None
        for begin in range(0,len(values),32):
            batch=np.stack(values[begin:begin+32]);observed=torch.tensor((batch-mean[None,:,None])/std[None,:,None],device=device)
            if kind=="population":correction=model(observed).cpu().numpy()*std[None,:,None]
            else:correction=model.forward_with_tokens(observed,tokens.expand(observed.shape[0],-1,-1)).cpu().numpy()*std[None,:,None]
            for offset,(value,length) in enumerate(zip(values[begin:begin+32],lengths[begin:begin+32])):outputs.append((value[:,:length]+correction[offset,:,:length]).astype(np.float32))
    return np.concatenate(outputs,axis=1)


def _sample(model:TemporalSupportCorrectionDiffusion,query:np.ndarray,support:tuple[np.ndarray,np.ndarray,np.ndarray],mean:np.ndarray,std:np.ndarray,device:torch.device,seed:int,present:float=1.)->tuple[np.ndarray,float]:
    values,lengths=_window_values(query);outputs=[];spread=[];model.eval();generator=torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for value,length in zip(values,lengths):
            observed=torch.tensor(((value-mean[:,None])/std[:,None])[None],device=device);samples=model.sample(observed,_kwargs(support,device,present),generator,k=int(model.config.posterior_samples))[:,0].cpu().numpy()*std[None,:,None];outputs.append((value[:,:length]+samples.mean(0)[:,:length]).astype(np.float32));spread.append(float(np.sqrt(np.mean(samples[:,:,:length].var(0)))))
    return np.concatenate(outputs,axis=1),float(np.mean(spread))


def _motion(value:np.ndarray,imu:np.ndarray,rate:float)->float:
    length=min(value.shape[1],imu.shape[1]);value=value[:,:length];imu=imu[:,:length];a=(value-value.mean(1,keepdims=True))/np.maximum(value.std(1,keepdims=True),1e-8);b=(imu-imu.mean(1,keepdims=True))/np.maximum(imu.std(1,keepdims=True),1e-8);frequencies,coherence=signal.coherence(a[:12].mean(0),b.mean(0),fs=rate,nperseg=min(512,length));keep=(frequencies>=.5)&(frequencies<=8.);return float(np.mean(coherence[keep]))


def _continuous_true_runs(mask:np.ndarray,minimum:int)->list[slice]:
    padded=np.pad(np.asarray(mask,dtype=np.int8),(1,1));edges=np.flatnonzero(np.diff(padded));return [slice(int(start),int(stop)) for start,stop in edges.reshape(-1,2) if stop-start>=minimum]


def _continuous_welch_distortion(output:np.ndarray,observed:np.ndarray,mask:np.ndarray,rate:float)->float:
    sections=_continuous_true_runs(mask,max(128,int(rate)))
    if not sections:return float("nan")
    nperseg=min(512,min(section.stop-section.start for section in sections));spectra_output=[];spectra_observed=[]
    for section in sections:
        _,left=signal.welch(output[:,section],fs=rate,nperseg=nperseg,axis=-1)
        _,right=signal.welch(observed[:,section],fs=rate,nperseg=nperseg,axis=-1)
        spectra_output.append(left);spectra_observed.append(right)
    left=np.mean(spectra_output,axis=0);right=np.mean(spectra_observed,axis=0)
    return float(np.linalg.norm(left-right)/max(np.linalg.norm(right),1e-12))


def _score(output:np.ndarray,observed:np.ndarray,imu:np.ndarray,rate:float)->dict[str,float]:
    length=min(output.shape[1],observed.shape[1],imu.shape[1]);output=output[:,:length];observed=observed[:,:length];imu=imu[:,:length];window=512;count=length//window
    if count<1:raise ValueError("query is shorter than one evaluator window")
    activity=np.asarray([np.sqrt(np.mean(imu[:,i*window:(i+1)*window]**2)) for i in range(count)]);threshold=np.quantile(activity,.25);low=np.repeat(activity<=threshold,window);low=np.pad(low,(0,length-low.size),constant_values=False);den=max(np.linalg.norm(observed[:,low]),1e-12);pres=1-np.linalg.norm(output[:,low]-observed[:,low])/den;psd=_continuous_welch_distortion(output,observed,low,rate);cov=np.linalg.norm(np.cov(output[:,low])-np.cov(observed[:,low]),ord="fro")/max(np.linalg.norm(np.cov(observed[:,low]),ord="fro"),1e-12);return {"motion_coherence_reduction":_motion(observed,imu,rate)-_motion(output,imu,rate),"nonartifact_observation_preservation":float(pres),"reference_free_psd_distortion":float(psd),"reference_free_covariance_distortion":float(cov),"output_input_RMS_ratio":float(np.sqrt(np.mean(output**2))/max(np.sqrt(np.mean(observed**2)),1e-12)),"observation_change_ratio":float(np.linalg.norm(output-observed)/max(np.linalg.norm(observed),1e-12))}


def technical_check(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    seed=20260811;_seed_all(seed);device=torch.device("cuda",0);fit,_,_=_fold_sets(config,0);mean,std=_normalization(config,fit);episodes=_protocols(config,fit);rng=np.random.default_rng(seed);query,target,supports,present=_batch(config,episodes,mean,std,rng);observed=torch.tensor(query[:2],device=device);truth=torch.tensor(target[:2],device=device);kwargs=_stack_kwargs(supports[:2],present[:2],device);det=TemporalSupportCleaner().to(device);small=TemporalDiffusionConfig(timesteps=32,ddim_steps=4,posterior_samples=2,min_snr_gamma=float(config["diffusion"]["min_snr_gamma"]),correction_clip=float(config["diffusion"]["correction_clip"]));diff=TemporalSupportCorrectionDiffusion(small).to(device);optimizer=torch.optim.AdamW(list(det.parameters())+list(diff.parameters()),lr=1e-4);generator=torch.Generator(device=device).manual_seed(seed);det_loss=torch.mean((det(observed,**kwargs)-truth)**2);diff_loss=diff.training_loss(truth,observed,kwargs,generator);loss=det_loss+diff_loss;loss.backward();optimizer.step()
    run_dir.mkdir(parents=True,exist_ok=True);checkpoint=run_dir/"technical_checkpoint.pt";torch.save({"det":det.state_dict(),"diff":diff.state_dict(),"optimizer":optimizer.state_dict(),"generator":generator.get_state()},checkpoint);det_reload=TemporalSupportCleaner().to(device);diff_reload=TemporalSupportCorrectionDiffusion(small).to(device);payload=torch.load(checkpoint,map_location=device,weights_only=False);det_reload.load_state_dict(payload["det"]);diff_reload.load_state_dict(payload["diff"])
    match_kwargs={key:value.clone() for key,value in kwargs.items()};null_kwargs={key:value.clone() for key,value in kwargs.items()};null_kwargs["context_present"].zero_();reverse_kwargs={key:value.clone() for key,value in kwargs.items()};reverse_kwargs["support_eeg"]=reverse_kwargs["support_eeg"].flip(-1);reverse_kwargs["support_imu"]=reverse_kwargs["support_imu"].flip(-1)
    with torch.no_grad():
        match_det=det_reload(observed,**match_kwargs);null_det=det_reload(observed,**null_kwargs);reverse_det=det_reload(observed,**reverse_kwargs);first=diff_reload.sample(observed,match_kwargs,torch.Generator(device=device).manual_seed(seed+1),k=2);second=diff_reload.sample(observed,match_kwargs,torch.Generator(device=device).manual_seed(seed+1),k=2)
    context_change=max(float((match_det-null_det).abs().max()),float((match_det-reverse_det).abs().max()));finite=bool(torch.isfinite(loss) and torch.isfinite(first).all());common=bool(torch.equal(first,second));scale=float(torch.sqrt(torch.mean(first.square()))/torch.sqrt(torch.mean(observed.square())).clamp_min(1e-8));passed=finite and common and context_change>1e-8 and scale<10 and bool(torch.all(kwargs["modality_present"][:,2]==0))
    summary={"status":"passed" if passed else "failed_technical_validity","real_batch":True,"finite_gradient_and_output":finite,"checkpoint_reload":True,"common_random_numbers_exact":common,"context_output_max_change":context_change,"sample_to_observation_rms_ratio":scale,"support_seconds":60,"eog_modality_masked":bool(torch.all(kwargs["modality_present"][:,2]==0)),"sealed_signal_opened":False};(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary


def run_fold(config:Mapping[str,Any],run_dir:Path,fold:int,seed:int)->Mapping[str,Any]:
    _seed_all(seed+fold*10000);device=torch.device("cuda",0);fit,inner,heldout=_fold_sets(config,fold);mean,std=_normalization(config,fit);training={};paths={}
    for kind in ("population","deterministic","diffusion"):paths[kind],training[kind]=_train_model(kind,config,fold,seed,fit,inner,mean,std,device)
    models={}
    for kind,path in paths.items():
        payload=torch.load(path,map_location=device,weights_only=False)
        if kind=="population":model=PopulationCleaner().to(device)
        elif kind=="deterministic":model=TemporalSupportCleaner().to(device)
        else:model=TemporalSupportCorrectionDiffusion(TemporalDiffusionConfig(timesteps=int(config["diffusion"]["timesteps"]),ddim_steps=int(config["diffusion"]["ddim_steps"]),posterior_samples=int(config["diffusion"]["posterior_samples"]),min_snr_gamma=float(config["diffusion"]["min_snr_gamma"]),correction_clip=float(config["diffusion"]["correction_clip"]))).to(device)
        model.load_state_dict(payload["state"]);models[kind]=model
    protocols=_protocols(config,heldout);rows=[];pareto=[];output_root=CODE_ROOT/str(config["output_root"])/f"factorial/seed_{seed}/fold_{fold:02d}";arrays=output_root/"server_arrays";arrays.mkdir(parents=True,exist_ok=True);rate=float(config["preprocessing"]["target_sampling_rate_hz"])
    for protocol in protocols:
        participant=protocol["participant"];task=protocol["task"];support_session=protocol["support_session"];query_session=protocol["query_session"];query_start=int(float(protocol["query_start"])*rate);query=np.asarray(_cache(config,participant,query_session,task,"eeg"),dtype=np.float32)[:,query_start:];match=_support(config,participant,support_session,task,mean,std);base_seed=seed*10000+int(participant[-2:])*100+fold;outputs={"RAW":query,"POP":_predict_one(models["population"],"population",query,match,mean,std,device),"DET-NULL":_predict_one(models["deterministic"],"deterministic",query,match,mean,std,device,0.),"DET-MATCH":_predict_one(models["deterministic"],"deterministic",query,match,mean,std,device,1.)};diff_null,unc_null=_sample(models["diffusion"],query,match,mean,std,device,base_seed,0.);outputs["DIFF-NULL"]=diff_null;outputs["DIFF-POP"]=diff_null;diff_match,unc_match=_sample(models["diffusion"],query,match,mean,std,device,base_seed,1.);outputs["DIFF-MATCH"]=diff_match
        for index in (1,2,3):
            wrong=_support(config,protocol[f"wrong_donor_{index}"],support_session,task,mean,std);outputs[f"DET-WRONG-{index}"]=_predict_one(models["deterministic"],"deterministic",query,wrong,mean,std,device,1.);outputs[f"DIFF-WRONG-{index}"],_=_sample(models["diffusion"],query,wrong,mean,std,device,base_seed,1.)
        shuffled=_support(config,participant,support_session,task,mean,std,reverse=True);outputs["DET-SHUFFLED"]=_predict_one(models["deterministic"],"deterministic",query,shuffled,mean,std,device,1.);outputs["DIFF-SHUFFLED"],_=_sample(models["diffusion"],query,shuffled,mean,std,device,base_seed,1.)
        stem=f"{participant}_{protocol['protocol']}_{task}_{query_session}";np.savez_compressed(arrays/f"{stem}.npz",**outputs)
        # Query IMU/events open only after every deployable output is frozen.
        imu=np.asarray(_cache(config,participant,query_session,task,"imu"),dtype=np.float32)[:,query_start:];onsets=np.asarray(_cache(config,participant,query_session,task,"event_onsets",False))-query_start/rate;durations=np.asarray(_cache(config,participant,query_session,task,"event_durations",False));labels=json.loads((Path(str(config["derived_root"]))/participant/query_session/task/"event_labels.json").read_text());raw_neural=_erp_readout(query,onsets,labels,rate) if task=="ERP" else _ssvep_readout(query,onsets,labels,durations,rate)
        for method,output in outputs.items():
            neural=_erp_readout(output,onsets,labels,rate) if task=="ERP" else _ssvep_readout(output,onsets,labels,durations,rate);safety={"erp_amplitude_relative_preservation":float("nan"),"erp_latency_relative_preservation":float("nan"),"ssvep_snr_relative_preservation":float("nan"),"ssvep_phase_relative_preservation":float("nan")}
            if task=="ERP" and raw_neural.get("erp_status")==neural.get("erp_status")=="success":safety["erp_amplitude_relative_preservation"]=1-abs(float(neural["erp_amplitude_uv"])-float(raw_neural["erp_amplitude_uv"]))/max(abs(float(raw_neural["erp_amplitude_uv"])),1e-8);safety["erp_latency_relative_preservation"]=1-abs(float(neural["erp_peak_latency_seconds"])-float(raw_neural["erp_peak_latency_seconds"]))/0.8
            if task=="SSVEP" and raw_neural.get("ssvep_status")==neural.get("ssvep_status")=="success":safety["ssvep_snr_relative_preservation"]=1-abs(float(neural["ssvep_snr_db"])-float(raw_neural["ssvep_snr_db"]))/max(abs(float(raw_neural["ssvep_snr_db"])),1.);safety["ssvep_phase_relative_preservation"]=1-abs(float(neural["ssvep_phase_consistency"])-float(raw_neural["ssvep_phase_consistency"]))
            metric={"fold":fold,"seed":seed,"participant":participant,"protocol":protocol["protocol"],"task":task,"query_session":query_session,"method":method,"status":"success","support_seconds":60,"query_imu_event_used_for_inference":False,"common_random_numbers":method.startswith("DIFF"),"posterior_samples":8 if method.startswith("DIFF") else 0,"posterior_sd_rms":unc_match if method=="DIFF-MATCH" else unc_null if method in ("DIFF-NULL","DIFF-POP") else float("nan"),**_score(output,query,imu,rate),**safety};rows.append(metric)
            for gamma in config["evaluation"]["pareto_strengths"]:
                value=query+float(gamma)*(output-query);pareto.append({"fold":fold,"seed":seed,"participant":participant,"protocol":protocol["protocol"],"task":task,"method":method,"gamma":gamma,**_score(value,query,imu,rate)})
    _write_csv(output_root/"metrics.csv",rows);_write_csv(output_root/"pareto_metrics.csv",pareto);summary={"status":"completed_real_mobile_factorial","fold":fold,"seed":seed,"heldout":heldout,"rows":len(rows),"training":training,"sealed_signal_opened":False};(output_root/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
