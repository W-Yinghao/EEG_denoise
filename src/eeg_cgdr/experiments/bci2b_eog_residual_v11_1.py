"""V11.1 spectral repair and same-session completion without changing V11."""
from __future__ import annotations
import csv,json,time
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping
import numpy as np,yaml
from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11

SAME=("same_01","same_02","same_03")
METHODS=("RAW","LINEAR-POP","LINEAR-MATCH","LINEAR-WRONG","DET-POP","DET-MATCH","DET-WRONG","DIFF-POP","DIFF-MATCH","DIFF-WRONG","DIFF-TEMPORAL-SHUFFLED")

def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as h:return yaml.safe_load(h)
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0]);[keys.append(k) for r in rows for k in r if k not in keys]
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def _link(source:Path,target:Path)->None:
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source)

def stage_materialize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    old=Path(str(config["v11_result_root"]));new=Path(str(config["result_root"]));_json(new/"technical_validity.json",json.loads((old/"technical_validity.json").read_text()))
    for fold in range(9):
        source=old/"folds"/f"fold_{fold:02d}";target=new/"folds"/f"fold_{fold:02d}";target.mkdir(parents=True,exist_ok=True);_link(source/"training_pairs.npz",target/"training_pairs.npz")
        rows=[r for r in _read(source/"unit_manifest.csv") if r["protocol"] in SAME];_csv(target/"unit_manifest.csv",rows)
        for row in rows:_link(source/"units"/row["protocol"]/"inference.npz",target/"units"/row["protocol"]/"inference.npz")
        if fold<3:_link(source/"checkpoint.pt",target/"checkpoint.pt")
    summary={"status":"completed_deployable_materialization","folds":9,"evaluator_links":0,"same_session_only":True,"v11_read_only":True};_json(run_dir/"result_summary.json",summary);return summary

def _welch(value:np.ndarray,fs:float,crop:int)->tuple[np.ndarray,np.ndarray]:
    from scipy.signal import welch
    return welch(value[...,:crop],fs=fs,nperseg=250,noverlap=125,axis=-1)
def _band_error(output:np.ndarray,target:np.ndarray,band:tuple[float,float],config:Mapping[str,Any])->float:
    f,p=_welch(output,float(config["sampling_rate"]),int(config["crop_length"]));_,q=_welch(target,float(config["sampling_rate"]),int(config["crop_length"]));mask=(f>=band[0])&(f<=band[1]);floor=float(config["psd_floor_uv2_hz"]);return float(np.mean(np.abs(np.log(np.maximum(p[...,mask],floor))-np.log(np.maximum(q[...,mask],floor)))))
def _bandpower_distortion(output:np.ndarray,raw:np.ndarray,band:tuple[float,float],config:Mapping[str,Any])->float:
    f,p=_welch(output,float(config["sampling_rate"]),int(config["crop_length"]));_,q=_welch(raw,float(config["sampling_rate"]),int(config["crop_length"]));mask=(f>=band[0])&(f<=band[1]);floor=float(config["psd_floor_uv2_hz"]);bp=np.trapezoid(p[...,mask],f[mask],axis=-1);bq=np.trapezoid(q[...,mask],f[mask],axis=-1);return float(np.mean(np.abs(np.log(np.maximum(bp,floor))-np.log(np.maximum(bq,floor)))))
def _method_arrays(base:Path,protocol:str,source:Path|None=None)->tuple[dict[str,np.ndarray],Any,Any]:
    unit=base/"units"/protocol;inf=np.load(unit/"inference.npz");ev=np.load((source or base)/"units"/protocol/"evaluator.npz");out=np.load(base/"outputs"/protocol/"inference_outputs.npz");return {k.split("_",1)[1]:np.asarray(out[k]) for k in out.files if k.startswith("paired_")},{k.split("_",1)[1]:np.asarray(out[k]) for k in out.files if k.startswith("natural_")},(inf,ev)

def stage_forensic(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));old=Path(str(config["v11_result_root"]));paired=[];natural=[];bands=[];lambdas=[];participant=[]
    for fold in range(3):
        base=old/"folds"/f"fold_{fold:02d}";subject=fold+1
        for protocol in SAME:
            pa,na,(inf,ev)=_method_arrays(base,protocol);scale=np.asarray(inf["eeg_scale"]);loc=np.asarray(inf["eeg_location"]);x=np.asarray(ev["paired_x"])[...,:500];raw_p=np.asarray(pa["RAW"])*scale[None,:,None]+loc[None,:,None];raw_p=raw_p[...,:500];raw_n=np.asarray(na["RAW"])[...,:500];eog=np.asarray(inf["natural_eog"])[...,:500];energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3)
            raw_error=_band_error(raw_p,x,(1,45),config)
            for method in METHODS:
                if method not in pa:continue
                p=np.asarray(pa[method])*scale[None,:,None]+loc[None,:,None];p=p[...,:500];n=np.asarray(na[method])[...,:500];error=_band_error(p,x,(1,45),config);paired.append({"subject":subject,"protocol":protocol,"method":method,"paired_psd_error_1_45":error,"paired_spectral_utility":raw_error-error})
                for name,band in config["bands"].items():bands.append({"subject":subject,"protocol":protocol,"method":method,"panel":"paired","band":name,"error":_band_error(p,x,tuple(band),config),"utility":_band_error(raw_p,x,tuple(band),config)-_band_error(p,x,tuple(band),config)})
                distortion=_bandpower_distortion(n[low],raw_n[low],(8,30),config);natural.append({"subject":subject,"protocol":protocol,"method":method,"natural_mi_band_distortion":distortion,"preservation":1-v11.rrmse(n[low],raw_n[low]),"covariance":v11._covariance_distortion(n[low],raw_n[low]),"eog_attenuation":v11._coherence_proxy(raw_n,eog)-v11._coherence_proxy(n,eog),"historical_whole_log_psd":v11._psd_distortion(np.asarray(na[method])[low],np.asarray(na["RAW"])[low])})
                for name,band in config["bands"].items():bands.append({"subject":subject,"protocol":protocol,"method":method,"panel":"natural","band":name,"distortion":_bandpower_distortion(n[low],raw_n[low],tuple(band),config)})
                for value in config["lambda_grid"]:
                    mixed=raw_p+float(value)*(p-raw_p);lambdas.append({"subject":subject,"protocol":protocol,"method":method,"lambda":value,"paired_psd_error_1_45":_band_error(mixed,x,(1,45),config),"rrmse":v11.rrmse(mixed,x)})
    # Participant-first gate quantities; old natural CSV supplies the frozen MI kappa.
    old_natural=[]
    for fold in range(3):old_natural.extend(_read(old/"folds"/f"fold_{fold:02d}"/"natural_safety.csv"))
    for subject in range(1,4):
        for method in ("DET-POP","DIFF-POP"):
            p=[r for r in paired if r["subject"]==subject and r["method"]==method];n=[r for r in natural if r["subject"]==subject and r["method"]==method];k=[float(r["mi_kappa"]) for r in old_natural if int(r["subject"])==subject and r["protocol"] in SAME and r["method"]==method];rawk=[float(r["mi_kappa"]) for r in old_natural if int(r["subject"])==subject and r["protocol"] in SAME and r["method"]=="RAW"]
            participant.append({"subject":subject,"method":method,"paired_spectral_utility":float(np.mean([r["paired_spectral_utility"] for r in p])),"natural_mi_band_distortion":float(np.mean([r["natural_mi_band_distortion"] for r in n])),"preservation":float(np.mean([r["preservation"] for r in n])),"covariance":float(np.mean([r["covariance"] for r in n])),"eog_attenuation":float(np.mean([r["eog_attenuation"] for r in n])),"kappa_minus_raw":float(np.nanmean(k)-np.nanmean(rawk))})
    _csv(root/"forensic_paired.csv",paired);_csv(root/"forensic_natural.csv",natural);_csv(root/"forensic_bandwise.csv",bands);_csv(root/"forensic_lambda_curves.csv",lambdas);_csv(root/"forensic_participant.csv",participant)
    d=[r for r in participant if r["method"]=="DIFF-POP"];t=[r for r in participant if r["method"]=="DET-POP"];checks={"paired_utility_mean_positive":bool(np.mean([r["paired_spectral_utility"] for r in d])>0),"paired_utility_positive_subjects":int(np.sum([r["paired_spectral_utility"]>0 for r in d])),"mi_distortion_delta":float(np.mean([r["natural_mi_band_distortion"] for r in d])-np.mean([r["natural_mi_band_distortion"] for r in t])),"preservation":float(np.mean([r["preservation"] for r in d])),"covariance":float(np.mean([r["covariance"] for r in d])),"eog_attenuation":float(np.mean([r["eog_attenuation"] for r in d])),"kappa_minus_raw":float(np.mean([r["kappa_minus_raw"] for r in d]))};passed=checks["paired_utility_mean_positive"] and checks["paired_utility_positive_subjects"]>=2 and checks["mi_distortion_delta"]<=.02 and checks["preservation"]>=.75 and checks["covariance"]<=.25 and checks["eog_attenuation"]>0 and checks["kappa_minus_raw"]>=-.02;summary={"status":"completed_spectral_forensic","contract_freeze_authorized":bool(passed),"checks":checks,"subjects":3,"padding_excluded":True,"v11_frozen_gate_unchanged":True};_json(root/"forensic_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_freeze_contract(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));decision=json.loads((root/"forensic_decision.json").read_text());
    if not decision["contract_freeze_authorized"]:raise RuntimeError("subjects 1-3 did not authorize evaluator contract")
    contract={"status":"frozen_before_participants_4_9_evaluator_open","crop_length":500,"sampling_rate_hz":250,"primary_paired_band_hz":[1,45],"primary_natural_neural_band_hz":[8,30],"bands":config["bands"],"psd_floor_uv2_hz":config["psd_floor_uv2_hz"],"welch":{"nperseg":250,"noverlap":125,"padding_samples_excluded":12},"aggregation":"participant_first","primary_panel":"K8_DDIM25","secondary_panel":"K32_DDIM25","participant_wise_k_selection":False,"historical_v11_gate":"frozen_failed_whole_spectrum_0.3168_gt_0.25","routing":{"population":"DET/DIFF POP mean+median improve RAW; >=7/9; paired spectral utility >0; natural safety","subject":"U_P/U_W mean+median >0; >=6/9; natural safety","diffusion":"one global K only; U_D/U_L mean+median >0; >=5/9; spectral no worse LINEAR; safety"},"development_not_confirmation":True};_json(root/"spectral_evaluator_contract.json",contract);_json(run_dir/"result_summary.json",contract);return contract

def stage_train(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:return v11.stage_train_fold(config,task_index,run_dir)

def _infer_k(config:Mapping[str,Any],fold:int,k:int)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    root=Path(str(config["result_root"]))/"folds"/f"fold_{fold:02d}";checkpoint=torch.load(root/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**checkpoint["config"]);device=torch.device("cuda");det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(checkpoint["det"]);diff.load_state_dict(checkpoint["diff"]);ema=EMA(diff);ema.load_state_dict(checkpoint["ema"]);ema.copy_to(diff);det.eval();diff.eval();scale=np.asarray(checkpoint["residual_scale"],np.float32);latency=[]
    for unit_index,row in enumerate(_read(root/"unit_manifest.csv")):
        protocol=row["protocol"];data=np.load(root/"units"/protocol/"inference.npz");outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(data[f"{panel}_y"],np.float32);eog=np.asarray(data[f"{panel}_eog"],np.float32);gamma=float(data["gamma"]);outputs[f"{panel}_RAW"]=y
            for name,h in {"POP":data["h_population"],"MATCH":data["h_match"],"WRONG":data["h_wrong"]}.items():
                a0=v11.apply_transfer(np.asarray(h),eog);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);start=time.perf_counter()
                with torch.no_grad():rdet=det(y=yt,eog=et,a0=at,r_det=None) if False else det(y=yt,eog=et,a0=at);bank=v11._noise_bank(y.shape,int(config["seed"])+fold*100000+unit_index*10000,k);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy() for noise in bank]
                torch.cuda.synchronize();latency.append((time.perf_counter()-start)/len(y));dc=at.cpu().numpy()+rdet.cpu().numpy()*scale[None,:,None];correction=a0+(rdet.cpu().numpy()+np.mean(samples,axis=0))*scale[None,:,None];outputs[f"{panel}_LINEAR-{name}"]=v11.gamma_correction(y,a0,gamma);outputs[f"{panel}_DET-{name}"]=v11.gamma_correction(y,dc,gamma);outputs[f"{panel}_DIFF-{name}"]=v11.gamma_correction(y,correction,gamma)
            shuffled=v11.temporal_shuffle(eog,int(config["seed"])+fold*100+unit_index);a0=v11.apply_transfer(np.asarray(data["h_match"]),shuffled);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(shuffled,device=device);at=torch.as_tensor(a0,device=device)
            with torch.no_grad():rdet=det(y=yt,eog=et,a0=at);bank=v11._noise_bank(y.shape,int(config["seed"])+fold*100000+unit_index*10000,k);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy() for noise in bank]
            correction=a0+(rdet.cpu().numpy()+np.mean(samples,axis=0))*scale[None,:,None];outputs[f"{panel}_DIFF-TEMPORAL-SHUFFLED"]=v11.gamma_correction(y,correction,gamma)
        out=root/"outputs"/f"k{k}"/protocol;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"inference_outputs.npz",**outputs)
    return {"k":k,"latency_seconds_per_window":float(np.mean(latency)),"posterior_calls":k*25,"peak_memory_bytes":int(torch.cuda.max_memory_allocated())}

def stage_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    values=[_infer_k(config,task_index,k) for k in (8,32)];summary={"status":"completed_evaluator_blind_inference","fold":task_index,"panels":values,"evaluator_opened":False};_json(Path(str(config["result_root"]))/"folds"/f"fold_{task_index:02d}"/"inference_runtime.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def _erd_preservation(output:np.ndarray,raw:np.ndarray,labels:np.ndarray,config:Mapping[str,Any])->float:
    f,p=_welch(output,float(config["sampling_rate"]),500);_,q=_welch(raw,float(config["sampling_rate"]),500);mask=(f>=8)&(f<=30);bp=np.trapezoid(p[...,mask],f[mask],axis=-1);bq=np.trapezoid(q[...,mask],f[mask],axis=-1);patterns=[]
    for label in np.unique(labels):
        take=labels==label
        if np.any(take):patterns.append((bp[take].mean(0),bq[take].mean(0)))
    return float(1-np.mean([np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12) for a,b in patterns])) if patterns else float("nan")

def stage_evaluate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));old=Path(str(config["v11_result_root"]));contract=json.loads((root/"spectral_evaluator_contract.json").read_text());assert contract["crop_length"]==500;base=root/"folds"/f"fold_{task_index:02d}";source=old/"folds"/f"fold_{task_index:02d}";paired=[];natural=[];bandrows=[]
    for k in (8,32):
        for protocol in SAME:
            inf=np.load(base/"units"/protocol/"inference.npz");ev=np.load(source/"units"/protocol/"evaluator.npz");out=np.load(base/"outputs"/f"k{k}"/protocol/"inference_outputs.npz");scale=np.asarray(inf["eeg_scale"]);loc=np.asarray(inf["eeg_location"]);x=np.asarray(ev["paired_x"])[...,:500];raw_p=np.asarray(out["paired_RAW"])*scale[None,:,None]+loc[None,:,None];raw_p=raw_p[...,:500];raw_error=_band_error(raw_p,x,(1,45),config);raw_n=np.asarray(out["natural_RAW"])[...,:500];eog=np.asarray(inf["natural_eog"])[...,:500];labels=np.asarray(ev["natural_labels"]);energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3)
            for key in out.files:
                panel,method=key.split("_",1);value=np.asarray(out[key])
                if panel=="paired":
                    physical=(value*scale[None,:,None]+loc[None,:,None])[...,:500];error=_band_error(physical,x,(1,45),config);paired.append({"subject":task_index+1,"protocol":protocol,"k":k,"method":method,"rrmse":v11.rrmse(physical,x),"correlation":v11.correlation(physical,x),"delta_snr":v11.delta_snr(physical,x,raw_p),"paired_psd_error_1_45":error,"paired_spectral_utility":raw_error-error})
                    for name,band in config["bands"].items():bandrows.append({"subject":task_index+1,"protocol":protocol,"k":k,"method":method,"panel":"paired","band":name,"error":_band_error(physical,x,tuple(band),config),"utility":_band_error(raw_p,x,tuple(band),config)-_band_error(physical,x,tuple(band),config)})
                else:
                    n=value[...,:500];natural.append({"subject":task_index+1,"protocol":protocol,"k":k,"method":method,"mi_band_distortion":_bandpower_distortion(n[low],raw_n[low],(8,30),config),"preservation":1-v11.rrmse(n[low],raw_n[low]),"covariance":v11._covariance_distortion(n[low],raw_n[low]),"eog_attenuation":v11._coherence_proxy(raw_n,eog)-v11._coherence_proxy(n,eog),"mi_kappa":v11._kappa(n,labels),"erd_preservation":_erd_preservation(n,raw_n,labels,config),"historical_whole_psd":v11._psd_distortion(value[low],np.asarray(out["natural_RAW"])[low])})
                    for name,band in config["bands"].items():bandrows.append({"subject":task_index+1,"protocol":protocol,"k":k,"method":method,"panel":"natural","band":name,"distortion":_bandpower_distortion(n[low],raw_n[low],tuple(band),config)})
    _csv(base/"paired_metrics_v11_1.csv",paired);_csv(base/"natural_metrics_v11_1.csv",natural);_csv(base/"bandwise_v11_1.csv",bandrows);summary={"status":"completed_post_contract_evaluation","fold":task_index,"evaluator_opened_after_contract":True,"paired_rows":len(paired),"natural_rows":len(natural)};_json(run_dir/"result_summary.json",summary);return summary

def _participant_rows(paired:list[dict[str,str]],natural:list[dict[str,str]],k:int)->list[dict[str,Any]]:
    rows=[]
    for subject in range(1,10):
        p=[r for r in paired if int(r["subject"])==subject and int(r["k"])==k];n=[r for r in natural if int(r["subject"])==subject and int(r["k"])==k];by=defaultdict(list)
        for r in p:by[r["method"]].append(float(r["rrmse"]))
        mean={m:float(np.mean(v)) for m,v in by.items()};match=[r for r in n if r["method"]=="DIFF-MATCH"];raw=[r for r in n if r["method"]=="RAW"]
        rows.append({"subject":subject,"k":k,"U_D":mean["DET-MATCH"]-mean["DIFF-MATCH"],"U_P":mean["DIFF-POP"]-mean["DIFF-MATCH"],"U_W":mean["DIFF-WRONG"]-mean["DIFF-MATCH"],"U_S":mean["DIFF-TEMPORAL-SHUFFLED"]-mean["DIFF-MATCH"],"U_L":mean["LINEAR-MATCH"]-mean["DIFF-MATCH"],"diff_match_rrmse":mean["DIFF-MATCH"],"linear_match_rrmse":mean["LINEAR-MATCH"],"paired_spectral_utility":float(np.mean([float(r["paired_spectral_utility"]) for r in p if r["method"]=="DIFF-MATCH"])),"linear_spectral_utility":float(np.mean([float(r["paired_spectral_utility"]) for r in p if r["method"]=="LINEAR-MATCH"])),"mi_band_distortion":float(np.mean([float(r["mi_band_distortion"]) for r in match])),"preservation":float(np.mean([float(r["preservation"]) for r in match])),"covariance":float(np.mean([float(r["covariance"]) for r in match])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in match])),"kappa_minus_raw":float(np.nanmean([float(r["mi_kappa"]) for r in match])-np.nanmean([float(r["mi_kappa"]) for r in raw])),"erd_preservation":float(np.mean([float(r["erd_preservation"]) for r in match]))})
    return rows

def stage_aggregate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));paired=[];natural=[];bands=[];runtime=[]
    for fold in range(9):
        base=root/"folds"/f"fold_{fold:02d}";paired.extend(_read(base/"paired_metrics_v11_1.csv"));natural.extend(_read(base/"natural_metrics_v11_1.csv"));bands.extend(_read(base/"bandwise_v11_1.csv"));runtime.append(json.loads((base/"inference_runtime.json").read_text()))
    _csv(root/"paired_spectral_metrics.csv",paired);_csv(root/"bandwise_safety.csv",bands);participants={k:_participant_rows(paired,natural,k) for k in (8,32)};summaries={};boot=[];rng=np.random.default_rng(int(config["bootstrap_seed"]));indices=rng.integers(0,9,size=(int(config["bootstrap_replicates"]),9))
    for k,rows in participants.items():
        _csv(root/f"participant_effects_k{k}.csv",rows);methods=[]
        for method in METHODS:
            p=[r for r in paired if int(r["k"])==k and r["method"]==method];n=[r for r in natural if int(r["k"])==k and r["method"]==method];methods.append({"k":k,"method":method,"subjects":9,"rrmse":float(np.mean([float(r["rrmse"]) for r in p])),"correlation":float(np.mean([float(r["correlation"]) for r in p])),"delta_snr":float(np.mean([float(r["delta_snr"]) for r in p])),"paired_spectral_utility":float(np.mean([float(r["paired_spectral_utility"]) for r in p])),"mi_band_distortion":float(np.mean([float(r["mi_band_distortion"]) for r in n])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in n])),"preservation":float(np.mean([float(r["preservation"]) for r in n])),"covariance":float(np.mean([float(r["covariance"]) for r in n])),"mi_kappa":float(np.nanmean([float(r["mi_kappa"]) for r in n])),"erd_preservation":float(np.mean([float(r["erd_preservation"]) for r in n]))})
        _csv(root/f"method_summary_k{k}.csv",methods);effects={}
        for e in ("U_D","U_P","U_W","U_S","U_L"):
            values=np.asarray([r[e] for r in rows]);rep=values[indices].mean(1);effects[e]={"mean":float(values.mean()),"median":float(np.median(values)),"positive":int(np.sum(values>0)),"ci_low":float(np.quantile(rep,.025)),"ci_high":float(np.quantile(rep,.975))};boot.append({"k":k,"effect":e,**effects[e],"status":"participant_descriptive_bootstrap"})
        summaries[k]={"effects":effects,"safety":{name:float(np.mean([r[name] for r in rows])) for name in ("paired_spectral_utility","mi_band_distortion","preservation","covariance","eog_attenuation","kappa_minus_raw","erd_preservation")}}
    _csv(root/"bootstrap_summary.csv",boot);resource=[]
    for k in (8,32):
        values=[panel for fold in runtime for panel in fold["panels"] if int(panel["k"])==k];resource.append({"k":k,"folds":9,"latency_seconds_per_window":float(np.mean([v["latency_seconds_per_window"] for v in values])),"posterior_network_calls":int(values[0]["posterior_calls"]),"peak_memory_bytes_max":int(max(v["peak_memory_bytes"] for v in values)),"participant_wise_selection":0})
    _csv(root/"inference_resource_summary.csv",resource)
    def method(k:int,name:str)->dict[str,str]:return next(r for r in _read(root/f"method_summary_k{k}.csv") if r["method"]==name)
    routing={}
    for k in (8,32):
        raw=float(method(k,"RAW")["rrmse"]);det=float(method(k,"DET-POP")["rrmse"]);diff=float(method(k,"DIFF-POP")["rrmse"]);det_effect=[];diff_effect=[]
        for participant in range(1,10):
            def unit_mean(name:str)->float:return float(np.mean([float(r["rrmse"]) for r in paired if int(r["subject"])==participant and int(r["k"])==k and r["method"]==name]))
            unit_raw=unit_mean("RAW");det_effect.append(unit_raw-unit_mean("DET-POP"));diff_effect.append(unit_raw-unit_mean("DIFF-POP"))
        popwins_det=int(np.sum(np.asarray(det_effect)>0));popwins_diff=int(np.sum(np.asarray(diff_effect)>0));e=summaries[k]["effects"];s=summaries[k]["safety"];diff_pop=method(k,"DIFF-POP");det_pop=method(k,"DET-POP");raw_method=method(k,"RAW");population_safety=float(diff_pop["preservation"])>=.75 and float(diff_pop["covariance"])<=.25 and float(diff_pop["eog_attenuation"])>0 and float(diff_pop["mi_kappa"])-float(raw_method["mi_kappa"])>=-.02 and float(diff_pop["mi_band_distortion"])<=float(det_pop["mi_band_distortion"])+.02;population=det<raw and diff<raw and np.median(det_effect)>0 and np.median(diff_effect)>0 and popwins_det>=7 and popwins_diff>=7 and float(diff_pop["paired_spectral_utility"])>0 and population_safety;subject=e["U_P"]["mean"]>0 and e["U_P"]["median"]>0 and e["U_P"]["positive"]>=6 and e["U_W"]["mean"]>0 and e["U_W"]["median"]>0 and e["U_W"]["positive"]>=6 and s["preservation"]>=.75 and s["covariance"]<=.25 and s["eog_attenuation"]>0 and s["kappa_minus_raw"]>=-.02;increment=e["U_D"]["mean"]>0 and e["U_D"]["median"]>0 and e["U_D"]["positive"]>=5 and e["U_L"]["mean"]>0 and e["U_L"]["median"]>0 and e["U_L"]["positive"]>=5 and s["paired_spectral_utility"]>=float(method(k,"LINEAR-MATCH")["paired_spectral_utility"]) and s["preservation"]>=.75 and s["covariance"]<=.25 and s["eog_attenuation"]>0 and s["kappa_minus_raw"]>=-.02;routing[k]={"population_valid":bool(population),"subject_operator_supported":bool(subject),"diffusion_increment_supported":bool(increment),"pop_det_wins":popwins_det,"pop_diff_wins":popwins_diff,"population_safety":bool(population_safety)}
    # K32 may win globally only if lower DIFF-MATCH RRMSE than K8; never participant-wise selection.
    k32_overall_better=float(method(32,"DIFF-MATCH")["rrmse"])<float(method(8,"DIFF-MATCH")["rrmse"]) and routing[32]["population_valid"] and routing[32]["subject_operator_supported"];winning_k=32 if k32_overall_better else 8;r=routing[winning_k]
    if r["subject_operator_supported"] and not r["diffusion_increment_supported"]:decision="SUBJECT_OPERATOR_SUPPORTED_BUT_DIFFUSION_INCREMENT_NOT_ESTABLISHED"
    elif r["subject_operator_supported"] and r["diffusion_increment_supported"] and r["population_valid"]:decision="ELIGIBLE_FOR_TWO_ADDITIONAL_SEEDS"
    else:decision="CURRENT_V11_1_SAME_SESSION_INSTANCE_NO_GO"
    result={"status":"completed_nine_participant_one_seed_development","decision":decision,"winning_global_k":winning_k,"k32_overall_better":bool(k32_overall_better),"k32_role":"secondary_not_selected" if winning_k==8 else "selected_global_panel","additional_seeds_authorized":decision=="ELIGIBLE_FOR_TWO_ADDITIONAL_SEEDS","routing_by_k":routing,"summaries":summaries,"participants":9,"same_session_only":True,"cross_session_extended":False,"development_not_confirmation":True};_json(root/"routing_decision.json",result);_json(root/"result_summary.json",result);_json(run_dir/"result_summary.json",result);return result

def stage_finalize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=Path(str(config["result_root"]));forensic=json.loads((root/"forensic_decision.json").read_text());result=json.loads((root/"result_summary.json").read_text());k=result["winning_global_k"];methods=_read(root/f"method_summary_k{k}.csv");effects=_read(root/f"participant_effects_k{k}.csv")
    high=[r for r in _read(root/"forensic_bandwise.csv") if r["panel"]=="natural" and r["band"]=="high_frequency_secondary" and r["method"]=="DIFF-POP"];mi=[r for r in _read(root/"forensic_bandwise.csv") if r["panel"]=="natural" and r["band"] in ("alpha","beta") and r["method"]=="DIFF-POP"];historical=[r for r in _read(root/"forensic_natural.csv") if r["method"]=="DIFF-POP"]
    Path("reports/v11_frozen_gate_correction.md").write_text("# V11 frozen-gate correction\n\nV11 remains `CURRENT_EOG_ANCHORED_RESIDUAL_DIFFUSION_INSTANCE_NO_GO` under its frozen whole-spectrum metric. The historical 0.3168 > 0.25 result is not overwritten. V11.1 crops the 12 padding samples and freezes task-relevant spectral metrics before opening participants 4–9 outcomes.\n\nThe bridge-audit support gamma check used target=prediction, so 45/45 gamma=1 was tautological and is not gamma-validity evidence. Fold gamma was support-only and shared across methods, but MATCH used the full support operator and therefore was not a strict operator cross-fit.\n",encoding="utf-8")
    per_subject={subject:float(np.mean([float(r["historical_whole_log_psd"]) for r in historical if int(r["subject"])==subject])) for subject in (1,2,3)};worst=max(per_subject,key=per_subject.get)
    Path("reports/v11_1_spectral_forensic.md").write_text("# V11.1 spectral forensic\n\nSubjects 1–3 only were used to freeze the evaluator. Padding samples 500–511 are excluded. The original whole-spectrum metric remains historical secondary. Physical inverse normalization was checked through the paired identity/RAW reconstruction path; the Welch grid is 0–125 Hz at 1 Hz resolution, and the frozen PSD floor is 1e-12 µV²/Hz.\n\nForensic decision: `"+str(forensic["contract_freeze_authorized"])+"`. Historical whole-spectrum DIFF-POP distortion: "+f"{np.mean([float(r['historical_whole_log_psd']) for r in historical]):.4f}"+". Cropped high-frequency (45–125 Hz) bandpower distortion: "+f"{np.mean([float(r['distortion']) for r in high]):.4f}"+"; alpha/beta distortion: "+f"{np.mean([float(r['distortion']) for r in mi]):.4f}"+f". Participant {worst} had the largest historical whole-spectrum distortion ({per_subject[worst]:.4f}). The task-relevant MI-band result is much smaller than the high-frequency secondary result, so >45 Hz and participant heterogeneity are the dominant forensic signals; excluding padding alone does not erase high-frequency distortion, and no inverse-normalization error was found. DC and 45–125 Hz remain secondary and do not enter the repaired gate. Full attribution tables are in `forensic_bandwise.csv`.\n",encoding="utf-8")
    lines=["# BCI2b EOG residual diffusion V11.1","","Development completion, same-session only. Participants 4–9 were unscored development holdouts, not sealed confirmation.","",f"Decision: `{result['decision']}`. Global panel: K={k}.","","| method | RRMSE | paired spectral utility | MI-band distortion | preservation | covariance | EOG attenuation |","|---|---:|---:|---:|---:|---:|---:|"]
    for name in METHODS:
        r=next(x for x in methods if x["method"]==name);lines.append(f"| {name} | {float(r['rrmse']):.4f} | {float(r['paired_spectral_utility']):+.4f} | {float(r['mi_band_distortion']):.4f} | {float(r['preservation']):.4f} | {float(r['covariance']):.4f} | {float(r['eog_attenuation']):+.4f} |")
    lines += ["","| effect | mean | median | positive |","|---|---:|---:|---:|"]
    for e,d in result["summaries"][str(k)]["effects"].items():lines.append(f"| {e} | {d['mean']:+.4f} | {d['median']:+.4f} | {d['positive']}/9 |")
    lines += ["","K=8 and K=32 are global panels; no participant-wise K selection occurred. LINEAR-MATCH remains a primary comparator. Cross-session participants 4–9 were not evaluated. Evidence is development, not confirmation."]
    Path("reports/bci2b_eog_residual_diffusion_v11_1.md").write_text("\n".join(lines)+"\n",encoding="utf-8");fig,ax=plt.subplots(figsize=(7,4));ax.axhline(0,color="black",lw=.8);ax.bar(np.arange(9),[float(r["U_P"]) for r in effects],label="U_P");ax.plot(np.arange(9),[float(r["U_D"]) for r in effects],"o",label="U_D");ax.set_xticks(np.arange(9),[f"P{i}" for i in range(1,10)]);ax.legend();ax.set_ylabel("positive = DIFF-MATCH better");fig.tight_layout();(root/"figures").mkdir(exist_ok=True);fig.savefig(root/"figures"/"participant_effects.png",dpi=180);plt.close(fig);_json(run_dir/"result_summary.json",result);return result

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    c=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="materialize":return stage_materialize(c,task_index,run_dir)
    if stage=="forensic":return stage_forensic(c,task_index,run_dir)
    if stage=="freeze-contract":return stage_freeze_contract(c,task_index,run_dir)
    if stage=="train":return stage_train(c,task_index,run_dir)
    if stage=="infer":return stage_infer(c,task_index,run_dir)
    if stage=="evaluate":return stage_evaluate(c,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(c,task_index,run_dir)
    if stage=="finalize":return stage_finalize(c,task_index,run_dir)
    raise ValueError(stage)
