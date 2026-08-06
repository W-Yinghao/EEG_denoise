"""Full-development temporal deterministic support headroom screen."""

from __future__ import annotations

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

from eeg_cgdr.models.temporal_support_conditioner import PopulationCleaner, TemporalSupportCleaner


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _read_csv(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def _write_csv(path:Path,rows:list[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _cache(config:Mapping[str,Any],participant:str,session:str,task:str,name:str,mmap:bool=True)->np.ndarray:
    path=Path(str(config["derived_root"]))/participant/session/task/f"{name}.npy"
    return np.load(path,mmap_mode="r" if mmap else None)


def _normalization(config:Mapping[str,Any],participants:list[str])->tuple[np.ndarray,np.ndarray]:
    total=np.zeros(46); square=np.zeros(46); count=0
    for participant in participants:
        for session in ("ses-02","ses-03","ses-04"):
            for task in ("ERP","SSVEP"):
                path=Path(str(config["derived_root"]))/participant/session/task/"eeg.npy"
                if not path.is_file():continue
                value=np.load(path,mmap_mode="r")
                # Exact streaming moments over complete outer-training records.
                total+=np.sum(value,dtype=np.float64,axis=1); square+=np.sum(np.square(value,dtype=np.float64),axis=1); count+=value.shape[1]
    if count==0:raise ValueError("no outer-training cached samples")
    mean=total/count; std=np.sqrt(np.maximum(square/count-mean*mean,1e-8)); return mean.astype(np.float32),std.astype(np.float32)


def _training_records(config:Mapping[str,Any],participants:list[str])->list[tuple[str,str,str]]:
    root=Path(str(config["derived_root"]));result=[]
    for participant in participants:
        for session in ("ses-03","ses-04"):
            for task in ("ERP","SSVEP"):
                if (root/participant/session/task/"eeg.npy").is_file() and (root/participant/"ses-02"/task/"eeg.npy").is_file():result.append((participant,session,task))
    if not result:raise ValueError("no compatible outer-training query/support records")
    return result


def _support(config:Mapping[str,Any],participant:str,session:str,task:str,length:int,mean:np.ndarray,std:np.ndarray,reverse:bool=False)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    eeg=np.asarray(_cache(config,participant,session,task,"eeg")[:,:length],dtype=np.float32)
    imu=np.asarray(_cache(config,participant,session,task,"imu")[:,:length],dtype=np.float32)
    if eeg.shape[1]<length:
        eeg=np.pad(eeg,((0,0),(0,length-eeg.shape[1]))); imu=np.pad(imu,((0,0),(0,length-imu.shape[1])))
    eeg=(eeg-mean[:,None])/std[:,None]; imu=(imu-imu.mean(axis=1,keepdims=True))/np.maximum(imu.std(axis=1,keepdims=True),1e-6)
    if reverse: eeg=eeg[:,::-1].copy(); imu=imu[:,::-1].copy()
    return eeg,imu,np.zeros((4,length),dtype=np.float32)


def _checkpoint_payload(model:torch.nn.Module,population:torch.nn.Module,optimizer:torch.optim.Optimizer,pop_optimizer:torch.optim.Optimizer,step:int,mean:np.ndarray,std:np.ndarray,best:Mapping[str,Any])->dict[str,Any]:
    return {"model":model.state_dict(),"population":population.state_dict(),"optimizer":optimizer.state_dict(),"population_optimizer":pop_optimizer.state_dict(),"step":step,"mean":mean,"std":std,"best":dict(best),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate()}


def _validation_loss(model:TemporalSupportCleaner,population:PopulationCleaner,config:Mapping[str,Any],participants:list[str],mean:np.ndarray,std:np.ndarray,device:torch.device,seed:int)->tuple[float,float]:
    rng=np.random.default_rng(seed); records=_training_records(config,participants);values=[]; pops=[]; window=int(config["preprocessing"]["window_seconds"]*config["preprocessing"]["target_sampling_rate_hz"]); support=int(config["preprocessing"]["support_budget_seconds"]*config["preprocessing"]["target_sampling_rate_hz"])
    model.eval();population.eval()
    with torch.no_grad():
        for _ in range(48):
            participant,session,task=records[int(rng.integers(len(records)))]
            eeg=_cache(config,participant,session,task,"eeg");target=_cache(config,participant,session,task,"clean_proxy");start=int(rng.integers(0,max(1,eeg.shape[1]-window)))
            query=torch.tensor(((np.asarray(eeg[:,start:start+window])-mean[:,None])/std[:,None])[None],device=device)
            truth=torch.tensor(((np.asarray(target[:,start:start+window])-np.asarray(eeg[:,start:start+window]))/std[:,None])[None],device=device)
            seeg,simu,seog=_support(config,participant,"ses-02",task,support,mean,std)
            kwargs={"support_eeg":torch.tensor(seeg[None],device=device),"support_imu":torch.tensor(simu[None],device=device),"support_eog":torch.tensor(seog[None],device=device),"modality_present":torch.tensor([[1.,1.,0.]],device=device),"context_present":torch.ones(1,device=device)}
            values.append(float(torch.mean((model(query,**kwargs)-truth)**2)));pops.append(float(torch.mean((population(query)-truth)**2)))
    model.train();population.train();return float(np.mean(values)),float(np.mean(pops))


def _train(config:Mapping[str,Any],fold:int,run_dir:Path,device:torch.device)->tuple[Path,dict[str,Any]]:
    fold_rows=_read_csv(CODE_ROOT/str(config["output_root"])/"metadata/development_cv_folds.csv")
    training=sorted(row["participant"] for row in fold_rows if int(row["fold"])==fold and row["split"]=="training")
    validation=training[-2:];fit=training[:-2];seed=int(config["headroom"]["training_seed"]);torch.manual_seed(seed+fold);torch.cuda.manual_seed_all(seed+fold);np.random.seed(seed+fold);random.seed(seed+fold)
    mean,std=_normalization(config,fit);model=TemporalSupportCleaner().to(device);population=PopulationCleaner().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["headroom"]["learning_rate"]));pop_optimizer=torch.optim.AdamW(population.parameters(),lr=float(config["headroom"]["learning_rate"]));checkpoint=CODE_ROOT/str(config["output_root"])/f"checkpoints/fold_{fold:02d}/models.pt";checkpoint.parent.mkdir(parents=True,exist_ok=True)
    start_step=0;best={"model_loss":float("inf"),"population_loss":float("inf"),"model_state":None,"population_state":None,"step":0};curve=[]
    if checkpoint.is_file():
        payload=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(payload["model"]);population.load_state_dict(payload["population"]);optimizer.load_state_dict(payload["optimizer"]);pop_optimizer.load_state_dict(payload["population_optimizer"]);start_step=int(payload["step"]);mean=np.asarray(payload["mean"]);std=np.asarray(payload["std"]);best=dict(payload["best"]);torch.set_rng_state(payload["torch_rng"].cpu());torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]]);np.random.set_state(payload["numpy_rng"]);random.setstate(payload["python_rng"])
    fit_records=_training_records(config,fit);rng=np.random.default_rng(seed+fold);window=int(config["preprocessing"]["window_seconds"]*config["preprocessing"]["target_sampling_rate_hz"]);support=int(config["preprocessing"]["support_budget_seconds"]*config["preprocessing"]["target_sampling_rate_hz"]);batch=4;maximum=int(config["headroom"]["maximum_updates"]);interval=int(config["headroom"]["validation_interval"]);started=time.perf_counter()
    for step in range(start_step+1,maximum+1):
        queries=[];truths=[];seegs=[];simus=[];seogs=[];contexts=[]
        for _ in range(batch):
            participant,session,task=fit_records[int(rng.integers(len(fit_records)))];eeg=_cache(config,participant,session,task,"eeg");target=_cache(config,participant,session,task,"clean_proxy");start=int(rng.integers(0,max(1,eeg.shape[1]-window)))
            queries.append((np.asarray(eeg[:,start:start+window])-mean[:,None])/std[:,None]);truths.append((np.asarray(target[:,start:start+window])-np.asarray(eeg[:,start:start+window]))/std[:,None]);seeg,simu,seog=_support(config,participant,"ses-02",task,support,mean,std);seegs.append(seeg);simus.append(simu);seogs.append(seog);contexts.append(float(rng.random()>=0.25))
        query=torch.tensor(np.stack(queries),device=device);truth=torch.tensor(np.stack(truths),device=device);kwargs={"support_eeg":torch.tensor(np.stack(seegs),device=device),"support_imu":torch.tensor(np.stack(simus),device=device),"support_eog":torch.tensor(np.stack(seogs),device=device),"modality_present":torch.tensor([[1.,1.,0.]]*batch,device=device),"context_present":torch.tensor(contexts,device=device)}
        optimizer.zero_grad(set_to_none=True);prediction=model(query,**kwargs);loss=torch.mean((prediction-truth)**2);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        pop_optimizer.zero_grad(set_to_none=True);pop_prediction=population(query);pop_loss=torch.mean((pop_prediction-truth)**2);pop_loss.backward();torch.nn.utils.clip_grad_norm_(population.parameters(),1.0);pop_optimizer.step()
        if step%interval==0 or step==maximum:
            validation_loss,pop_validation=_validation_loss(model,population,config,validation,mean,std,device,seed+fold+step);curve.append({"step":step,"train_loss":float(loss),"population_train_loss":float(pop_loss),"validation_loss":validation_loss,"population_validation_loss":pop_validation})
            if validation_loss<best["model_loss"]:best.update({"model_loss":validation_loss,"model_state":{key:value.detach().cpu() for key,value in model.state_dict().items()},"step":step})
            if pop_validation<best["population_loss"]:best.update({"population_loss":pop_validation,"population_state":{key:value.detach().cpu() for key,value in population.state_dict().items()}})
            torch.save(_checkpoint_payload(model,population,optimizer,pop_optimizer,step,mean,std,best),checkpoint)
    if best["model_state"] is not None:model.load_state_dict(best["model_state"])
    if best["population_state"] is not None:population.load_state_dict(best["population_state"])
    torch.save({"model":model.state_dict(),"population":population.state_dict(),"mean":mean,"std":std,"best_step":best["step"],"seed":seed,"fold":fold},checkpoint.parent/"best.pt")
    _write_csv(checkpoint.parent/"training_curve.csv",curve)
    return checkpoint.parent/"best.pt",{"fit_participants":fit,"validation_participants":validation,"best_step":best["step"],"runtime_seconds":time.perf_counter()-started,"checkpoint":str(checkpoint.parent/"best.pt")}


def _predict(model:TemporalSupportCleaner,query:np.ndarray,support_tuple:tuple[np.ndarray,np.ndarray,np.ndarray],mean:np.ndarray,std:np.ndarray,device:torch.device,present:float=1.0)->np.ndarray:
    seeg,simu,seog=support_tuple;window=512;outputs=[];model.eval()
    with torch.no_grad():
        for start in range(0,query.shape[1],window):
            length=min(window,query.shape[1]-start);value=query[:,start:start+length]
            if length<window:value=np.pad(value,((0,0),(0,window-length)))
            tensor=torch.tensor(((value-mean[:,None])/std[:,None])[None],device=device);kwargs={"support_eeg":torch.tensor(seeg[None],device=device),"support_imu":torch.tensor(simu[None],device=device),"support_eog":torch.tensor(seog[None],device=device),"modality_present":torch.tensor([[1.,1.,0.]],device=device),"context_present":torch.tensor([present],device=device)}
            correction=model(tensor,**kwargs)[0].cpu().numpy()*std[:,None];outputs.append((value+correction[:,:length]).astype(np.float32))
    return np.concatenate(outputs,axis=1)


def _population_predict(model:PopulationCleaner,query:np.ndarray,mean:np.ndarray,std:np.ndarray,device:torch.device)->np.ndarray:
    window=512;outputs=[];model.eval()
    with torch.no_grad():
        for start in range(0,query.shape[1],window):
            length=min(window,query.shape[1]-start);value=query[:,start:start+length]
            if length<window:value=np.pad(value,((0,0),(0,window-length)))
            tensor=torch.tensor(((value-mean[:,None])/std[:,None])[None],device=device);correction=model(tensor)[0].cpu().numpy()*std[:,None];outputs.append((value+correction[:,:length]).astype(np.float32))
    return np.concatenate(outputs,axis=1)


def _motion_coherence(value:np.ndarray,imu:np.ndarray,rate:float=128.0)->float:
    eeg_z=(value-value.mean(1,keepdims=True))/np.maximum(value.std(1,keepdims=True),1e-8);imu_z=(imu-imu.mean(1,keepdims=True))/np.maximum(imu.std(1,keepdims=True),1e-8)
    frequencies,coherence=signal.coherence(eeg_z[:12].mean(0),imu_z.mean(0),fs=rate,nperseg=min(512,value.shape[1]));keep=(frequencies>=0.5)&(frequencies<=8.0);return float(np.mean(coherence[keep]))


def _score(output:np.ndarray,observed:np.ndarray,imu:np.ndarray)->dict[str,float]:
    length=min(output.shape[1],observed.shape[1],imu.shape[1]);output=output[:,:length];observed=observed[:,:length];imu=imu[:,:length];raw_coherence=_motion_coherence(observed,imu);out_coherence=_motion_coherence(output,imu)
    window=512;counts=length//window;activity=np.asarray([np.sqrt(np.mean(imu[:,i*window:(i+1)*window]**2)) for i in range(counts)]);threshold=np.quantile(activity,0.25);low=np.repeat(activity<=threshold,window);low=np.pad(low,(0,length-low.size),constant_values=False)
    denominator=max(np.linalg.norm(observed[:,low]),1e-12);preservation=1-np.linalg.norm(output[:,low]-observed[:,low])/denominator
    left=np.abs(np.fft.rfft(output[:,low],axis=-1));right=np.abs(np.fft.rfft(observed[:,low],axis=-1));psd=np.linalg.norm(left-right)/max(np.linalg.norm(right),1e-12);cov=np.linalg.norm(np.cov(output[:,low])-np.cov(observed[:,low]),ord="fro")/max(np.linalg.norm(np.cov(observed[:,low]),ord="fro"),1e-12)
    return {"motion_coherence_reduction":raw_coherence-out_coherence,"nonartifact_observation_preservation":float(preservation),"reference_free_psd_distortion":float(psd),"reference_free_covariance_distortion":float(cov),"output_input_RMS_ratio":float(np.sqrt(np.mean(output**2))/max(np.sqrt(np.mean(observed**2)),1e-12)),"observation_change_ratio":float(np.linalg.norm(output-observed)/max(np.linalg.norm(observed),1e-12))}


def run_fold(config:Mapping[str,Any],run_dir:Path,fold:int)->Mapping[str,Any]:
    device=torch.device("cuda",0);checkpoint,training=_train(config,fold,run_dir,device);payload=torch.load(checkpoint,map_location=device,weights_only=False);model=TemporalSupportCleaner().to(device);population=PopulationCleaner().to(device);model.load_state_dict(payload["model"]);population.load_state_dict(payload["population"]);mean=np.asarray(payload["mean"]);std=np.asarray(payload["std"])
    fold_rows=_read_csv(CODE_ROOT/str(config["output_root"])/"metadata/development_cv_folds.csv");heldout=sorted(row["participant"] for row in fold_rows if int(row["fold"])==fold and row["split"]=="validation");protocols=_read_csv(CODE_ROOT/str(config["output_root"])/"evaluator/frozen_protocol_units.csv");rows=[];array_root=CODE_ROOT/str(config["output_root"])/f"server_arrays/fold_{fold:02d}";array_root.mkdir(parents=True,exist_ok=True);rate=float(config["preprocessing"]["target_sampling_rate_hz"])
    for protocol in protocols:
        participant=protocol["participant"]
        if participant not in heldout or protocol["status"]!="eligible":continue
        task=protocol["task"];support_session=protocol["support_session"];query_session=protocol["query_session"];support_length=max(512,int(float(protocol["support_end"])*rate));query=np.asarray(_cache(config,participant,query_session,task,"eeg"),dtype=np.float32);query_start=int(float(protocol.get("query_start",0) or 0)*rate);query=query[:,query_start:]
        matching=_support(config,participant,support_session,task,support_length,mean,std);outputs={"RAW":query,"STRONG-POP":_population_predict(population,query,mean,std,device),"DET-NULL":_predict(model,query,matching,mean,std,device,present=0.0),"DET-MATCH":_predict(model,query,matching,mean,std,device)}
        for index in (1,2,3):
            donor=protocol[f"wrong_donor_{index}"]
            donor_path=Path(str(config["derived_root"]))/donor/support_session/task/"eeg.npy"
            if not donor_path.is_file():raise FileNotFoundError(f"frozen compatible wrong donor unavailable: {donor_path}")
            wrong=_support(config,donor,support_session,task,support_length,mean,std);outputs[f"DET-WRONG-{index}"]=_predict(model,query,wrong,mean,std,device)
        outputs["TEMPORAL-SHUFFLED"]=_predict(model,query,_support(config,participant,support_session,task,support_length,mean,std,reverse=True),mean,std,device)
        normalized=(query-mean[:,None])/std[:,None];outputs["ASR-LIKE"]=(np.clip(normalized,-5,5)*std[:,None]+mean[:,None]).astype(np.float32)
        stem=f"{participant}_{protocol['protocol']}_{task}_{query_session}";path=array_root/f"{stem}.npz";np.savez_compressed(path,**outputs)
        # Evaluator-side IMU opens only after all deployable outputs are frozen.
        imu=np.asarray(_cache(config,participant,query_session,task,"imu"),dtype=np.float32)[:,query_start:]
        imu_z=(imu-imu.mean(1,keepdims=True))/np.maximum(imu.std(1,keepdims=True),1e-8);design=np.concatenate((np.ones((1,imu.shape[1])),imu_z),axis=0);coef=np.linalg.solve(design@design.T+1e-2*np.eye(design.shape[0]),(query@design.T).T).T;outputs["ORACLE-QUERY-IMU"]=(query-coef@design).astype(np.float32)
        onsets=np.asarray(_cache(config,participant,query_session,task,"event_onsets",mmap=False),dtype=np.float64)-query_start/rate
        durations=np.asarray(_cache(config,participant,query_session,task,"event_durations",mmap=False),dtype=np.float64)
        labels=json.loads((Path(str(config["derived_root"]))/participant/query_session/task/"event_labels.json").read_text())
        from eeg_cgdr.experiments.mobile_bci_headroom_v4 import _erp_readout,_ssvep_readout
        raw_neural=_erp_readout(query,onsets,labels,rate) if task=="ERP" else _ssvep_readout(query,onsets,labels,durations,rate)
        for method,output in outputs.items():
            neural=_erp_readout(output,onsets,labels,rate) if task=="ERP" else _ssvep_readout(output,onsets,labels,durations,rate)
            safety={"erp_amplitude_relative_preservation":float("nan"),"erp_latency_relative_preservation":float("nan"),"ssvep_snr_relative_preservation":float("nan"),"ssvep_phase_relative_preservation":float("nan")}
            if task=="ERP" and raw_neural.get("erp_status")==neural.get("erp_status")=="success":
                safety["erp_amplitude_relative_preservation"]=1-abs(float(neural["erp_amplitude_uv"])-float(raw_neural["erp_amplitude_uv"]))/max(abs(float(raw_neural["erp_amplitude_uv"])),1e-8)
                safety["erp_latency_relative_preservation"]=1-abs(float(neural["erp_peak_latency_seconds"])-float(raw_neural["erp_peak_latency_seconds"]))/0.8
            if task=="SSVEP" and raw_neural.get("ssvep_status")==neural.get("ssvep_status")=="success":
                safety["ssvep_snr_relative_preservation"]=1-abs(float(neural["ssvep_snr_db"])-float(raw_neural["ssvep_snr_db"]))/max(abs(float(raw_neural["ssvep_snr_db"])),1.0)
                safety["ssvep_phase_relative_preservation"]=1-abs(float(neural["ssvep_phase_consistency"])-float(raw_neural["ssvep_phase_consistency"]))
            rows.append({"fold":fold,"participant":participant,"protocol":protocol["protocol"],"task":task,"query_session":query_session,"method":method,"status":"success","query_imu_used_for_inference":False,"query_event_used_for_inference":False,"oracle_nondeployable":method=="ORACLE-QUERY-IMU","outputs_frozen_before_query_imu_scoring":True,**_score(output,query,imu),**safety})
    output=CODE_ROOT/str(config["output_root"])/f"headroom/fold_{fold:02d}";_write_csv(output/"metrics.csv",rows);summary={"status":"completed_full_fold_temporal_deterministic_headroom","fold":fold,"heldout_participants":heldout,"metric_rows":len(rows),"training":training,"sealed_signal_opened":False};(output/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary


def _bootstrap(values:np.ndarray,repetitions:int,seed:int)->tuple[float,float]:
    rng=np.random.default_rng(seed); draws=np.empty(repetitions)
    for index in range(repetitions):draws[index]=values[rng.integers(0,values.size,values.size)].mean()
    return float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def aggregate_headroom(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    all_rows=[];root=CODE_ROOT/str(config["output_root"])
    for fold in range(4):all_rows.extend(_read_csv(root/f"headroom/fold_{fold:02d}/metrics.csv"))
    successful=[row for row in all_rows if row["status"]=="success"]
    # Average tasks and query sessions within each participant/protocol/method.
    metrics=("motion_coherence_reduction","nonartifact_observation_preservation","reference_free_psd_distortion","reference_free_covariance_distortion","output_input_RMS_ratio","erp_amplitude_relative_preservation","erp_latency_relative_preservation","ssvep_snr_relative_preservation","ssvep_phase_relative_preservation")
    units=[]
    for key in sorted({(r["participant"],r["protocol"],r["method"]) for r in successful}):
        selected=[r for r in successful if (r["participant"],r["protocol"],r["method"])==key];row={"participant":key[0],"protocol":key[1],"method":key[2],"record_rows":len(selected)}
        for metric in metrics:
            values=[]
            for value in selected:
                try:number=float(value[metric])
                except (ValueError,KeyError):continue
                if np.isfinite(number):values.append(number)
            row[metric]=float(np.mean(values)) if values else float("nan")
        units.append(row)
    _write_csv(root/"aggregate/unit_metrics.csv",units)
    effects=[];decisions={}
    for protocol in ("S0_STATIC_XSESSION","S1_MOTION_WITHIN_SESSION","S2_MOTION_XSPEED"):
        by_participant={}
        for row in units:
            if row["protocol"]==protocol:by_participant.setdefault(row["participant"],{})[row["method"]]=row
        for estimand,comparator in (("MATCH_minus_NULL","DET-NULL"),("MATCH_minus_STRONG_POP","STRONG-POP"),("MATCH_minus_TEMPORAL_SHUFFLED","TEMPORAL-SHUFFLED")):
            values=[]
            for participant,methods in by_participant.items():
                if "DET-MATCH" in methods and comparator in methods:values.append((participant,float(methods["DET-MATCH"]["motion_coherence_reduction"])-float(methods[comparator]["motion_coherence_reduction"])))
            array=np.asarray([value for _,value in values]);ci=_bootstrap(array,int(config["evaluation"]["bootstrap_repetitions"]),20260806) if array.size else (float("nan"),float("nan"))
            effects.append({"protocol":protocol,"estimand":estimand,"participants":array.size,"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0)),"participant_effects_json":json.dumps(dict(values),sort_keys=True)})
        wrong_values=[]
        for participant,methods in by_participant.items():
            if "DET-MATCH" not in methods:continue
            wrong=[float(methods[f"DET-WRONG-{i}"]["motion_coherence_reduction"]) for i in (1,2,3) if f"DET-WRONG-{i}" in methods]
            if len(wrong)==3:wrong_values.append((participant,float(methods["DET-MATCH"]["motion_coherence_reduction"])-float(np.mean(wrong))))
        array=np.asarray([value for _,value in wrong_values]);ci=_bootstrap(array,int(config["evaluation"]["bootstrap_repetitions"]),20260807) if array.size else (float("nan"),float("nan"));effects.append({"protocol":protocol,"estimand":"MATCH_minus_mean_WRONG","participants":array.size,"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0)),"participant_effects_json":json.dumps(dict(wrong_values),sort_keys=True)})
        effect_map={row["estimand"]:row for row in effects if row["protocol"]==protocol};required=("MATCH_minus_NULL","MATCH_minus_STRONG_POP","MATCH_minus_mean_WRONG")
        points=all(float(effect_map[name]["mean_utility"])>0 for name in required);clear=points and all(float(effect_map[name]["ci95_low"])>0 for name in required);majority=all(int(effect_map[name]["positive_count"])>=9 for name in required)
        safety=[]
        for participant,methods in by_participant.items():
            if "DET-MATCH" in methods and "STRONG-POP" in methods:
                left=methods["DET-MATCH"];right=methods["STRONG-POP"]
                safety.append((float(left["nonartifact_observation_preservation"])-float(right["nonartifact_observation_preservation"]),float(right["reference_free_psd_distortion"])-float(left["reference_free_psd_distortion"]),float(right["reference_free_covariance_distortion"])-float(left["reference_free_covariance_distortion"]),float(left["erp_amplitude_relative_preservation"])-float(right["erp_amplitude_relative_preservation"]) if np.isfinite(float(left["erp_amplitude_relative_preservation"])) and np.isfinite(float(right["erp_amplitude_relative_preservation"])) else np.nan,float(left["ssvep_snr_relative_preservation"])-float(right["ssvep_snr_relative_preservation"]) if np.isfinite(float(left["ssvep_snr_relative_preservation"])) and np.isfinite(float(right["ssvep_snr_relative_preservation"])) else np.nan))
        safety_array=np.asarray(safety);margin=float(config["evaluation"]["noninferiority_margin"]);safety_pass=bool(safety and np.nanmean(safety_array,axis=0)[:3].min()>=margin and np.nanmean(safety_array,axis=0)[3]>=margin and np.nanmean(safety_array,axis=0)[4]>=margin)
        decisions[protocol]={"point_effects_positive":points,"all_primary_ci_lower_positive":clear,"majority_participants":majority,"safety_passed":safety_pass,"safety_mean_deltas":np.nanmean(safety_array,axis=0).tolist() if safety else [],"route":"CLEAR_SUBJECT_HEADROOM" if clear and majority and safety_pass else "TENTATIVE_SUBJECT_HEADROOM" if points and majority and safety_pass else "SUBJECT_HEADROOM_NO_GO"}
    _write_csv(root/"aggregate/paired_effects.csv",effects)
    routes=[value["route"] for value in decisions.values()];overall="CLEAR_SUBJECT_HEADROOM" if "CLEAR_SUBJECT_HEADROOM" in routes else "TENTATIVE_SUBJECT_HEADROOM" if "TENTATIVE_SUBJECT_HEADROOM" in routes else "SUBJECT_HEADROOM_NO_GO"
    name="subject-and-activity-aware diffusion" if overall!="SUBJECT_HEADROOM_NO_GO" and decisions["S0_STATIC_XSESSION"]["route"]=="SUBJECT_HEADROOM_NO_GO" else "static subject calibration" if overall!="SUBJECT_HEADROOM_NO_GO" else "no_new_diffusion_training"
    summary={"status":"completed_mobile_headroom_routing","routing_decision":overall,"protocol_decisions":decisions,"authorized_next_model_name":name,"temporal_diffusion_one_seed_authorized":overall in ("CLEAR_SUBJECT_HEADROOM","TENTATIVE_SUBJECT_HEADROOM"),"sealed_signal_opened":False,"development_participants":16,"confirmation_eligibility":False}
    (root/"aggregate/routing_decision.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");report=CODE_ROOT/"reports/mobile_bci_headroom_v4.md";report.write_text("# MobileBCI participant headroom v4\n\n"+f"Routing decision: `{overall}`. All statistics use 16 development participants; the sealed eight were not opened. Protocol decisions: `{json.dumps(decisions,sort_keys=True)}`.\n",encoding="utf-8");return summary
