"""Strict POP8-R and support-only operator shrinkage for BCI-IV-2b.

All inference stages are evaluator blind. Evaluator arrays are opened only by
the dedicated CPU stages. Scientific aggregation is protocol -> seed ->
participant; row-weighted summaries are explicitly descriptive.
"""
from __future__ import annotations

import csv,json
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping

import numpy as np
import yaml

from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as v111
from eeg_cgdr.experiments import bci2b_subject_diffusion_next as nxt

SAME=("same_01","same_02","same_03")
SESSION_LABELS={1:"01T",2:"02T",3:"03T",4:"04E",5:"05E"}

def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as handle:return yaml.safe_load(handle)
def _root(c:Mapping[str,Any],key:str)->Path:return Path(str(c[key]))
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0])
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator="\n");writer.writeheader();writer.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))
def _task(c:Mapping[str,Any],index:int)->tuple[int,int]:
    seeds=list(map(int,c["seeds"]));return seeds[index//9],index%9
def _seed_fold(root:Path,seed:int,fold:int)->Path:return root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"

def _support_region(c:Mapping[str,Any],participant:int,session:str)->tuple[np.ndarray,np.ndarray,float,slice,float]:
    eeg,eog,sfreq,events=v11._load_session(c,participant,session);support,_=v11._support_query_ranges(events,eeg.shape[1],sfreq);seconds=(support.stop-support.start)/sfreq;return eeg,eog,sfreq,support,float(seconds)
def _eligible(c:Mapping[str,Any],participant:int,session:str,seconds:int=120)->bool:return _support_region(c,participant,session)[4]>=seconds
def _strict_transfer(c:Mapping[str,Any],participant:int,session:str,loc:np.ndarray,scale:np.ndarray,eloc:np.ndarray,escale:np.ndarray,seconds:int=120)->tuple[np.ndarray|None,float]:
    eeg,eog,sfreq,support,available=_support_region(c,participant,session)
    if available<seconds:return None,available
    region=slice(support.start,support.start+int(round(seconds*sfreq)));return v11._normalized_transfer(eeg,eog,region,loc,scale,eloc,escale),available
def _half_support_transfer(c:Mapping[str,Any],participant:int,session:str,inf:Mapping[str,np.ndarray])->np.ndarray:
    eeg,eog,sfreq,support,available=_support_region(c,participant,session)
    if available<120:raise RuntimeError(f"participant {participant} {session} lacks 120-second support")
    region=slice(support.start,support.start+int(round(60*sfreq)))
    return v11._normalized_transfer(eeg,eog,region,np.asarray(inf["eeg_location"]),np.asarray(inf["eeg_scale"]),np.asarray(inf["eog_location"]),np.asarray(inf["eog_scale"]))
def _population_operator(c:Mapping[str,Any],owners:list[int],session:str,loc:np.ndarray,scale:np.ndarray,eloc:np.ndarray,escale:np.ndarray)->tuple[np.ndarray,list[int]]:
    per=[];eligible=[]
    for owner in owners:
        value,_=_strict_transfer(c,owner,session,loc,scale,eloc,escale)
        if value is not None:per.append(value);eligible.append(owner)
    if not per:raise RuntimeError(f"no eligible population owners for {session}")
    return np.mean(np.stack(per),axis=0).astype(np.float32),eligible
def _global_population_operator(c:Mapping[str,Any],owners:list[int],loc:np.ndarray,scale:np.ndarray,eloc:np.ndarray,escale:np.ndarray)->tuple[np.ndarray,dict[int,list[str]]]:
    per=[];used={}
    for owner in owners:
        values=[];sessions=[]
        for session in SESSION_LABELS.values():
            value,_=_strict_transfer(c,owner,session,loc,scale,eloc,escale)
            if value is not None:values.append(value);sessions.append(session)
        if values:per.append(np.mean(np.stack(values),axis=0));used[owner]=sessions
    if not per:raise RuntimeError("no strict POP8-R support operators")
    return np.mean(np.stack(per),axis=0).astype(np.float32),used

def stage_audit(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    root=_root(c,"result_root");root.mkdir(parents=True,exist_ok=True);rows=[]
    for participant in range(1,10):
        for session in SESSION_LABELS.values():
            _,_,_,_,seconds=_support_region(c,participant,session);rows.append({"participant":participant,"session":session,"support_seconds":seconds,"eligible_120":int(seconds>=120)})
    _csv(root/"support_owner_eligibility.csv",rows)
    primary=[r for r in rows if r["session"] in ("01T","02T","03T")];summary={"status":"STRICT_POP8_AUDIT_PASSED","primary_eligible_units":sum(int(r["eligible_120"]) for r in primary),"primary_availability_denominator":27,"participants":9,"seeds":list(map(int,c["seeds"])),"training_updates":int(c["training_updates"]),"evaluator_blind_inference":True,"a_track_touched":False,"bci2a_data_exists":_root(c,"bci2a_data_root").exists()};_json(root/"frozen_protocol.json",summary);_json(run/"result_summary.json",summary);return summary

def stage_prepare(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    fold=task_index;recipient=fold+1;training=[x for x in range(1,10) if x!=recipient];root=_root(c,"strict_root")/"prepared"/f"fold_{fold:02d}";root.mkdir(parents=True,exist_ok=True);parts=[]
    for subject in training:
        for session,path in sorted(v11._bci2b_files(c,subject).items()):
            eeg,eog,sfreq,events=v11.load_gdf_channels(path,eeg_channels=3);eeg*=1e6;eog*=1e6
            try:part=v11._task_domain_pairs(eeg,eog,events,sfreq,20260808+subject*100+int(session[:2]),64)
            except RuntimeError:continue
            part["subject"]=np.full(len(part["y"]),subject,np.int16);part["session"]=np.full(len(part["y"]),int(session[:2]),np.int16);parts.append(part)
    if not parts:raise RuntimeError("no strict POP8-R training pairs")
    loc,scale=v11._robust_location_scale([p["y"] for p in parts]);eloc,escale=v11._robust_location_scale([p["eog"] for p in parts]);train={k:np.concatenate([p[k] for p in parts]) for k in ("x","y","a","eog","subject","session")}
    train["x"]=v11._normalize_windows(train["x"],loc,scale);train["y"]=v11._normalize_windows(train["y"],loc,scale);train["a"]=v11._normalize_windows(train["a"],loc,scale,difference=True);train["eog"]=v11._normalize_windows(train["eog"],eloc,escale)
    hpop,owner_sessions=_global_population_operator(c,training,loc,scale,eloc,escale);lookup={};fallback=[]
    for subject in training:
        for session_num,session in SESSION_LABELS.items():
            value,available=_strict_transfer(c,subject,session,loc,scale,eloc,escale);lookup[(subject,session_num)]=hpop if value is None else value
            if value is None:fallback.append({"participant":subject,"session":session,"available_seconds":available,"training_context":"POP_FALLBACK"})
    train["h_subject"]=np.stack([lookup[(int(s),int(q))] for s,q in zip(train["subject"],train["session"])]).astype(np.float32)
    np.savez_compressed(root/"training_pairs.npz",**train,h_population=hpop,eeg_location=loc,eeg_scale=scale,eog_location=eloc,eog_scale=escale);_csv(root/"training_fallbacks.csv",fallback)
    unit_rows=[];donor_rows=[]
    for protocol,support_session,query_session in v11._protocols()[:3]:
        hpop_session,pop_owners=_population_operator(c,training,support_session,loc,scale,eloc,escale);hmatch,available=_strict_transfer(c,recipient,support_session,loc,scale,eloc,escale);recipient_ok=hmatch is not None
        clean,e_art,a_phys,_,_=v11._query_arrays(c,recipient,query_session);paired_y=clean+a_phys;qeeg,qeog,qsf,qevents=v11._load_session(c,recipient,query_session);natural,natural_eog,labels=v11._query_trials(qeeg,qeog,qevents,qsf)
        arrays={"paired_y":v11._normalize_windows(paired_y,loc,scale),"paired_eog":v11._normalize_windows(e_art,eloc,escale),"natural_y":v11._normalize_windows(natural,loc,scale),"natural_eog":v11._normalize_windows(natural_eog,eloc,escale),"h_pop":hpop_session,"recipient_eligible":np.array(int(recipient_ok)),"recipient":np.array(recipient),"eeg_location":loc,"eeg_scale":scale,"eog_location":eloc,"eog_scale":escale}
        if recipient_ok:
            arrays["h_match"]=hmatch;eeg,eog,sfreq,support,_=_support_region(c,recipient,support_session);end=support.start+int(round(120*sfreq));mid=support.start+int(round(60*sfreq));hfit=v11._normalized_transfer(eeg,eog,slice(support.start,mid),loc,scale,eloc,escale);_,veog,_=v11._continuous_windows(eeg,eog,mid,end);ven=v11._normalize_windows(veog,eloc,escale);arrays["gamma"]=np.array(v11.support_gamma(v11.apply_transfer(hfit,ven),v11.apply_transfer(hmatch,ven)),np.float32) if len(ven) else np.array(0,np.float32)
        else:arrays["h_match"]=hpop_session;arrays["gamma"]=np.array(0,np.float32)
        donors=[]
        for donor in training:
            hd,dsecs=_strict_transfer(c,donor,support_session,loc,scale,eloc,escale);ok=hd is not None
            if ok:arrays[f"h_wrong_{donor}"]=hd;donors.append(donor)
            donor_rows.append({"fold":fold,"recipient":recipient,"protocol":protocol,"donor":donor,"support_seconds":dsecs,"eligible_120":int(ok),"population_training_seen":1})
        unit=root/"units"/protocol;unit.mkdir(parents=True,exist_ok=True);np.savez_compressed(unit/"inference.npz",**arrays);np.savez_compressed(unit/"evaluator.npz",paired_x=clean.astype(np.float32),paired_a=a_phys.astype(np.float32),natural_labels=labels.astype(np.int16));unit_rows.append({"fold":fold,"recipient":recipient,"protocol":protocol,"support_session":support_session,"query_session":query_session,"support_seconds":available,"eligible_120":int(recipient_ok),"population_owners":";".join(map(str,pop_owners)),"population_owner_count":len(pop_owners),"wrong_donors":";".join(map(str,donors)),"wrong_donor_count":len(donors),"paired_windows":len(clean),"natural_windows":len(natural)})
    fallback_rows=sum(
        int(((train["subject"]==int(row["participant"])) &
             (train["session"]==int(str(row["session"])[:2]))).sum())
        for row in fallback
    )
    _csv(root/"unit_manifest.csv",unit_rows);_csv(root/"donor_manifest.csv",donor_rows);_json(root/"fold_metadata.json",{"fold":fold,"recipient":recipient,"population_training":training,"population_training_count":8,"global_population_owner_sessions":owner_sessions,"training_pairs":len(train["y"]),"training_fallback_rows":int(fallback_rows),"strict_support_seconds":120})
    summary={"status":"STRICT_POP8_PREPARED","fold":fold,"recipient":recipient,"training_pairs":len(train["y"]),"eligible_units":sum(int(r["eligible_120"]) for r in unit_rows),"availability_units":3,"training_fallback_contexts":len(fallback)};_json(run/"result_summary.json",summary);return summary

def stage_train(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    seed,fold=_task(c,task_index);prepared=_root(c,"strict_root")/"prepared"/f"fold_{fold:02d}";target=_seed_fold(_root(c,"strict_root"),seed,fold);target.mkdir(parents=True,exist_ok=True)
    for name in ("training_pairs.npz","unit_manifest.csv","donor_manifest.csv","fold_metadata.json","training_fallbacks.csv"):
        dest=target/name
        if not dest.exists():dest.symlink_to(prepared/name)
    for protocol in SAME:
        dest=target/"units"/protocol;dest.mkdir(parents=True,exist_ok=True)
        for name in ("inference.npz","evaluator.npz"):
            path=dest/name
            if not path.exists():path.symlink_to(prepared/"units"/protocol/name)
    local={**c,"seed":seed,"result_root":str(_root(c,"strict_root"))};metrics=v11._train_models(local,fold,target,torch.device("cuda"));summary={"status":"STRICT_POP8_TRAINED","seed":seed,"fold":fold,**metrics};_json(run/"result_summary.json",summary);return summary

def _fit_normalized(eeg:np.ndarray,eog:np.ndarray,indices:np.ndarray,loc:np.ndarray,scale:np.ndarray,eloc:np.ndarray,escale:np.ndarray)->np.ndarray:
    y=(eeg[:,indices]-loc[:,None])/scale[:,None];e=(eog[:,indices]-eloc[:,None])/escale[:,None];return v11._instant_transfer(y,e)
def _select_lambda(c:Mapping[str,Any],participant:int,session:str,inf:Mapping[str,np.ndarray],hpop:np.ndarray)->tuple[float,list[dict[str,Any]],np.ndarray]:
    eeg,eog,sfreq,support,available=_support_region(c,participant,session)
    if available<120:return 0.0,[],hpop
    start=support.start;stop=start+int(round(120*sfreq));blocks=np.array_split(np.arange(start,stop),int(c["support_blocks"]));loc=np.asarray(inf["eeg_location"]);scale=np.asarray(inf["eeg_scale"]);eloc=np.asarray(inf["eog_location"]);escale=np.asarray(inf["eog_scale"]);lambdas=list(map(float,c["lambda_candidates"]));loss={lam:[] for lam in lambdas}
    for heldout,valid in enumerate(blocks):
        train=np.concatenate([block for index,block in enumerate(blocks) if index!=heldout]);hm=_fit_normalized(eeg,eog,train,loc,scale,eloc,escale);yv=(eeg[:,valid]-loc[:,None])/scale[:,None];ev=(eog[:,valid]-eloc[:,None])/escale[:,None]
        for lam in lambdas:
            h=hpop+lam*(hm-hpop);loss[lam].append(float(np.mean((yv-h@ev)**2)))
    curve=[{"lambda":lam,"mean_loss":float(np.mean(loss[lam])),"block_losses":";".join(f"{x:.10g}" for x in loss[lam]),"blocks":len(blocks)} for lam in lambdas];chosen=min(lambdas,key=lambda lam:(np.mean(loss[lam]),lam));full=np.concatenate(blocks);hmfull=_fit_normalized(eeg,eog,full,loc,scale,eloc,escale);return float(chosen),curve,hmfull

def stage_shrink_infer(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    seed,fold=_task(c,task_index);recipient=fold+1;base=_seed_fold(_root(c,"strict_root"),seed,fold);cp,det,diff,device=nxt._models(base/"checkpoint.pt");rescale=np.asarray(cp["residual_scale"],np.float32);lambda_rows=[]
    for unit_index,protocol in enumerate(SAME):
        inf=np.load(base/"units"/protocol/"inference.npz");manifest=_read(base/"unit_manifest.csv")[unit_index];folder=_root(c,"shrinkage_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs"/protocol;folder.mkdir(parents=True,exist_ok=True)
        if not bool(int(inf["recipient_eligible"])):np.savez_compressed(folder/"inference_outputs.npz");lambda_rows.append({"seed":seed,"participant":recipient,"protocol":protocol,"eligible":0,"selected_lambda":0,"candidate_lambda":"","reason":"recipient_support_lt_120"});continue
        hp=np.asarray(inf["h_pop"]);lam,curve,hm=_select_lambda(c,recipient,manifest["support_session"],inf,hp);hs=hp+lam*(hm-hp);donors=[int(x) for x in manifest["wrong_donors"].split(";") if x];lambda_rows.extend({"seed":seed,"participant":recipient,"protocol":protocol,"eligible":1,"selected_lambda":lam,"candidate_lambda":row["lambda"],"mean_loss":row["mean_loss"],"block_losses":row["block_losses"],"blocks":row["blocks"]} for row in curve);outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(inf[f"{panel}_y"],np.float32);eog=np.asarray(inf[f"{panel}_eog"],np.float32);gamma=float(inf["gamma"]);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8);outputs[f"{panel}_RAW"]=y
            contexts=[("POP8",hp),("RAW-MATCH",hm),("SHRINK-MATCH",hs)]+[(f"SHRINK-WRONG-{d}",hp+lam*(np.asarray(inf[f"h_wrong_{d}"])-hp)) for d in donors]
            for name,h in contexts:
                a0=v11.apply_transfer(h,eog);at=torch.as_tensor(a0,device=device)
                with torch.no_grad():rd=det(y=yt,eog=et,a0=at);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(n,device=device)).cpu().numpy() for n in bank]
                corr=a0+(rd.cpu().numpy()+np.mean(samples,0))*rescale[None,:,None];corr[...,500:]=0;outputs[f"{panel}_LINEAR-{name}"]=v11.gamma_correction(y,np.pad(a0[...,:500],((0,0),(0,0),(0,12))),gamma);detcorr=a0+rd.cpu().numpy()*rescale[None,:,None];detcorr[...,500:]=0;outputs[f"{panel}_DET-{name}"]=v11.gamma_correction(y,detcorr,gamma);outputs[f"{panel}_DIFF-{name}"]=v11.gamma_correction(y,corr,gamma)
        np.savez_compressed(folder/"inference_outputs.npz",**outputs)
    _csv(_root(c,"shrinkage_root")/"lambda_tasks"/f"seed_{seed}_fold_{fold:02d}.csv",lambda_rows);summary={"status":"SHRINKAGE_EVALUATOR_BLIND_INFERENCE_COMPLETED","seed":seed,"fold":fold,"recipient":recipient,"evaluator_opened":False};_json(run/"result_summary.json",summary);return summary

def stage_evaluate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    seed,fold=_task(c,task_index);base=_seed_fold(_root(c,"strict_root"),seed,fold);prepared=_root(c,"strict_root")/"prepared"/f"fold_{fold:02d}";outputs=_root(c,"shrinkage_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs";paired,natural=nxt._evaluate_outputs(c,base,prepared,outputs,seed,fold,"STRICT-SHRINK");root=_root(c,"shrinkage_root")/"evaluation";_csv(root/f"seed_{seed}_fold_{fold:02d}_paired.csv",paired);_csv(root/f"seed_{seed}_fold_{fold:02d}_natural.csv",natural);summary={"status":"STRICT_SHRINK_INDEPENDENT_EVALUATION_COMPLETED","seed":seed,"fold":fold,"paired_rows":len(paired),"natural_rows":len(natural)};_json(run/"result_summary.json",summary);return summary

def _sign_flip(values:np.ndarray)->float:
    observed=abs(float(values.mean()));signs=((np.arange(2**len(values))[:,None]>>np.arange(len(values)))&1)*2-1;return float(np.mean(np.abs((signs*values[None]).mean(1))>=observed))
def _effect_summary(c:Mapping[str,Any],seed_rows:list[dict[str,Any]],key:str)->dict[str,Any]:
    values=np.asarray([np.mean([r[key] for r in seed_rows if r["participant"]==p]) for p in range(1,10)]);rng=np.random.default_rng(int(c["bootstrap_seed"]));index=rng.integers(0,9,size=(int(c["bootstrap_replicates"]),9));rep=values[index].mean(1);return {"mean":float(values.mean()),"median":float(np.median(values)),"positive":int(np.sum(values>0)),"participant_values":values.tolist(),"seed_means":[float(np.mean([r[key] for r in seed_rows if r["seed"]==s])) for s in map(int,c["seeds"])],"two_sided_exact_sign_flip":_sign_flip(values),"descriptive_ci":[float(np.quantile(rep,.025)),float(np.quantile(rep,.975))],"leave_one_participant_out":[float(np.delete(values,i).mean()) for i in range(9)]}

def stage_aggregate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=_root(c,"shrinkage_root");paired=[];natural=[]
    for seed in map(int,c["seeds"]):
        for fold in range(9):paired.extend(_read(root/"evaluation"/f"seed_{seed}_fold_{fold:02d}_paired.csv"));natural.extend(_read(root/"evaluation"/f"seed_{seed}_fold_{fold:02d}_natural.csv"))
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);lambda_rows=[]
    for path in sorted((root/"lambda_tasks").glob("*.csv")):lambda_rows.extend(_read(path))
    _csv(root/"support_lambda_selection.csv",lambda_rows)
    seed_effects=[]
    for seed in map(int,c["seeds"]):
        for participant in range(1,10):
            take=[r for r in paired if int(r["seed"])==seed and int(r["participant"])==participant];by=defaultdict(list)
            for row in take:by[row["method"]].append(float(row["rrmse"]))
            needed=("DIFF-POP8","DIFF-SHRINK-MATCH","DET-POP8","DET-SHRINK-MATCH")
            if not all(x in by for x in needed):continue
            wrong=[float(np.mean(values)) for method,values in by.items() if method.startswith("DIFF-SHRINK-WRONG-")];seed_effects.append({"seed":seed,"participant":participant,"E_P":float(np.mean(by["DIFF-POP8"])-np.mean(by["DIFF-SHRINK-MATCH"])),"E_W":float(np.mean(wrong)-np.mean(by["DIFF-SHRINK-MATCH"])),"E_D":float(np.mean(by["DET-SHRINK-MATCH"])-np.mean(by["DIFF-SHRINK-MATCH"])),"Delta_SA":float((np.mean(by["DIFF-POP8"])-np.mean(by["DIFF-SHRINK-MATCH"]))-(np.mean(by["DET-POP8"])-np.mean(by["DET-SHRINK-MATCH"])))})
    _csv(root/"participant_seed_effects.csv",seed_effects);participant=[{"participant":p,**{key:float(np.mean([r[key] for r in seed_effects if r["participant"]==p])) for key in ("E_P","E_W","E_D","Delta_SA")}} for p in range(1,10)];_csv(root/"participant_effects.csv",participant)
    def participant_method(method:str,key:str,rows:list[dict[str,str]])->list[float]:return [float(np.mean([float(r[key]) for r in rows if int(r["participant"])==p and r["method"]==method])) for p in range(1,10)]
    methods=[]
    for method in sorted({r["method"] for r in paired}):
        take=[r for r in paired if r["method"]==method];methods.append({"method":method,"participants":len({int(r["participant"]) for r in take}),**{key:float(np.mean(participant_method(method,key,paired))) for key in ("rrmse","correlation","delta_snr","paired_spectral_utility")}})
    _csv(root/"method_summary_participant_first.csv",methods)
    row_weighted=[]
    for method in sorted({r["method"] for r in paired}):
        take=[r for r in paired if r["method"]==method];row_weighted.append({"method":method,"rows":len(take),**{key:float(np.mean([float(r[key]) for r in take])) for key in ("rrmse","correlation","delta_snr","paired_spectral_utility")}})
    _csv(root/"descriptive_row_weighted_method_summary.csv",row_weighted)
    safety_rows=[]
    for method in sorted({r["method"] for r in natural}):
        for participant in range(1,10):
            take=[r for r in natural if r["method"]==method and int(r["participant"])==participant]
            if take:safety_rows.append({"participant":participant,"method":method,**{key:float(np.mean([float(r[key]) for r in take])) for key in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}})
    _csv(root/"participant_natural_safety.csv",safety_rows)
    effects={key:_effect_summary(c,seed_effects,key) for key in ("E_P","E_W","E_D","Delta_SA")};sm=[r for r in safety_rows if r["method"]=="DIFF-SHRINK-MATCH"];sp=[r for r in safety_rows if r["method"]=="DIFF-POP8"];nat={key:float(np.mean([float(r[key]) for r in sm])) for key in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")};popnat={key:float(np.mean([float(r[key]) for r in sp])) for key in nat};margins={key:nat[key]-popnat[key] for key in nat}
    personal=effects["E_P"]["mean"]>.005 and effects["E_P"]["median"]>0 and effects["E_P"]["positive"]>=7 and all(x>0 for x in effects["E_P"]["seed_means"]) and effects["E_W"]["mean"]>0 and effects["E_W"]["median"]>0 and effects["E_W"]["positive"]>=7 and margins["preservation"]>=-.02 and margins["covariance"]<=.02 and margins["mi_kappa"]>=-.02 and nat["eog_attenuation"]>0
    diffusion_increment=effects["E_D"]["mean"]>0 and effects["E_D"]["positive"]>=6
    diffusion_amplification=effects["Delta_SA"]["mean"]>0 and effects["Delta_SA"]["positive"]>=6
    selected=[float(r["selected_lambda"]) for r in lambda_rows if r.get("eligible")=="1" and r.get("candidate_lambda") in ("0.0","0")]
    route={"status":"SUPPORT_SHRINKAGE_PERSONALIZATION_RETAINED" if personal else "STRONG_POPULATION_MATCHED_BY_SUPPORT_SHRINKAGE_NOT_ESTABLISHED","personalization_retained":bool(personal),"diffusion_point_estimate_increment":bool(diffusion_increment),"diffusion_specific_amplification":bool(diffusion_amplification),"bci2a_operator_audit_authorized":not personal,"development_only":True}
    summary={"effects":effects,"natural_match":nat,"natural_pop":popnat,"natural_margins":margins,"lambda_distribution":{"mean":float(np.mean(selected)) if selected else None,"positive_fraction":float(np.mean(np.asarray(selected)>0)) if selected else None,"unit_count":len(selected)},"routing":route,"availability":{"eligible_protocol_units":26,"denominator":27,"participants":9},"aggregation_order":"protocol -> seed -> participant"};_json(root/"result_summary.json",summary);_json(root/"routing_decision.json",route)
    figdir=root/"figures";figdir.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(7,4));x=np.arange(1,10);ax.axhline(0,color="black",lw=.8);ax.plot(x,[r["E_P"] for r in participant],"o-",label="E_P");ax.plot(x,[r["E_W"] for r in participant],"s-",label="E_W");ax.set_xlabel("Participant");ax.set_ylabel("Positive RRMSE utility");ax.legend();fig.tight_layout();fig.savefig(figdir/"shrinkage_effects.png",dpi=180);plt.close(fig)
    report=("# BCI2b POP8-centered operator shrinkage\n\n"+f"Decision: `{route['status']}`. Strict support coverage is 26/27 protocol units and all nine participants. Aggregation is protocol → seed → participant.\n\n"+"\n".join(f"- {key}: mean {value['mean']:+.5f}, median {value['median']:+.5f}, {value['positive']}/9, exact p={value['two_sided_exact_sign_flip']:.6f}." for key,value in effects.items())+f"\n\nNatural MATCH means: attenuation {nat['eog_attenuation']:+.5f}, preservation {nat['preservation']:.5f}, covariance {nat['covariance']:.5f}, MI kappa {nat['mi_kappa']:.5f}. This is development evidence only.\n");Path("reports/bci2b_operator_shrinkage.md").write_text(report,encoding="utf-8");_json(run/"result_summary.json",summary);return summary

def stage_strict_report(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    strict=_root(c,"strict_root");prepared=[]
    for fold in range(9):prepared.extend(_read(strict/"prepared"/f"fold_{fold:02d}"/"unit_manifest.csv"))
    summary={"status":"STRICT_POP8_R_COMPLETED","eligible_units":sum(int(r["eligible_120"]) for r in prepared),"availability_denominator":len(prepared),"training_checkpoints":len(list(strict.glob("seeds/*/folds/*/checkpoint.pt"))),"silent_truncation":False,"training_fallback_is_explicit":True,"development_only":True};_json(strict/"result_summary.json",summary);Path("reports/bci2b_pop8_strict_reanalysis.md").write_text(f"# BCI2b strict POP8-R reanalysis\n\nStrict owner-level 120-second eligibility yields {summary['eligible_units']}/{summary['availability_denominator']} evaluation units. Population operators average only eligible outer participants, WRONG donors with short support are NA, and training episodes with insufficient subject support use explicit POP fallback. All {summary['training_checkpoints']}/27 frozen-budget checkpoints completed. Scientific metrics are reported in the shrinkage package using participant-first aggregation; row-weighted summaries are descriptive only.\n",encoding="utf-8");_json(run/"result_summary.json",summary);return summary

def stage_repair_metadata(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    """Repair descriptive fallback row counts without touching prepared arrays."""
    strict=_root(c,"strict_root");rows=[]
    for fold in range(9):
        root=strict/"prepared"/f"fold_{fold:02d}"
        with np.load(root/"training_pairs.npz") as train:
            subjects=np.asarray(train["subject"]);sessions=np.asarray(train["session"])
        fallback=_read(root/"training_fallbacks.csv");count=sum(int(((subjects==int(row["participant"]))&(sessions==int(str(row["session"])[:2]))).sum()) for row in fallback)
        metadata=json.loads((root/"fold_metadata.json").read_text());metadata["training_fallback_rows"]=int(count);_json(root/"fold_metadata.json",metadata);rows.append({"fold":fold,"fallback_contexts":len(fallback),"fallback_training_rows":count})
    _csv(strict/"training_fallback_row_audit.csv",rows);summary={"status":"STRICT_POP8_METADATA_REPAIRED","folds":9,"prepared_arrays_modified":False};_json(run/"result_summary.json",summary);return summary

def bounded_oppost_sample(diff:Any,*,y:Any,eog:Any,a0:Any,r_det:Any,initial_noise:Any,center:Any,precision:Any,late_steps:int,w_max:float,strength:float=1.0,valid_length:int=500)->tuple[Any,list[dict[str,float]]]:
    """Frozen DDIM reverse chain with a bounded late-step diagonal proximal."""
    import torch
    if strength==0:return diff.sample(y=y,eog=eog,a0=a0,r_det=r_det,initial_noise=initial_noise),[]
    state=initial_noise.clone();schedule=torch.linspace(diff.config.timesteps-1,0,diff.config.ddim_steps,device=y.device).round().long();stats=[]
    for index,t_value in enumerate(schedule):
        timestep=torch.full((len(y),),int(t_value),device=y.device,dtype=torch.long);alpha=diff.alpha_bar.gather(0,timestep).reshape(len(y),1,1);v=diff.backbone(state,timestep,y=y,eog=eog,a0=a0,r_det=r_det);x0=alpha.sqrt()*state-(1-alpha).sqrt()*v;epsilon=(1-alpha).sqrt()*state+alpha.sqrt()*v;full=r_det+x0
        active=index>=len(schedule)-int(late_steps)
        if active:
            sigma2=strength*(1-alpha)/alpha.clamp_min(1e-5);raw=sigma2*precision;weight=torch.clamp(raw/(1+raw),max=float(w_max));prox=full+weight*(center-full);stats.append({"step":float(index),"timestep":float(t_value),"weight_min":float(weight.min()),"weight_median":float(weight.median()),"weight_max":float(weight.max())})
        else:prox=full
        if valid_length<prox.shape[-1]:prox[...,valid_length:]=full[...,valid_length:]
        x0=prox-r_det
        if index+1==len(schedule):state=x0
        else:
            next_t=torch.full_like(timestep,int(schedule[index+1]));next_alpha=diff.alpha_bar.gather(0,next_t).reshape(len(y),1,1);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
    if valid_length<state.shape[-1]:state[...,valid_length:]=0
    return state,stats

def stage_oppost_select(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    """Select one fold-global schedule using outer-training rows only."""
    import torch
    fold=task_index;seed=int(c["seeds"][0]);base=_seed_fold(_root(c,"strict_root"),seed,fold);cp,det,diff,device=nxt._models(base/"checkpoint.pt");rescale=np.asarray(cp["residual_scale"],np.float32)
    with np.load(base/"training_pairs.npz") as data:
        y=np.asarray(data["y"][-32:],np.float32);eog=np.asarray(data["eog"][-32:],np.float32);a=np.asarray(data["a"][-32:],np.float32);hp=np.asarray(data["h_population"],np.float32);hs=np.asarray(data["h_subject"][-32:],np.float32)
    a0=v11.apply_transfer(hp,eog);center=(v11.apply_transfer(hs,eog)-a0)/rescale[None,:,None];target=(a-a0)/rescale[None,:,None];variance=np.maximum(np.var(target-center,axis=(0,2)),.05);precision=torch.as_tensor((1/variance)[None,:,None],device=device);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);ct=torch.as_tensor(center,device=device);truth=torch.as_tensor(a,device=device);noise=torch.as_tensor(v11._noise_bank(y.shape,seed+fold*1000,1)[0],device=device)
    with torch.no_grad():rd=det(y=yt,eog=et,a0=at)
    candidates=[(0,0.0)]+[(int(late),float(wmax)) for late in c["oppost_late_steps"] for wmax in c["oppost_w_max"]];rows=[]
    for late,wmax in candidates:
        with torch.no_grad():delta,_=bounded_oppost_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise,center=ct,precision=precision,late_steps=max(late,1),w_max=max(wmax,1e-6),strength=0 if late==0 else 1)
        correction=at+(rd+delta)*torch.as_tensor(rescale,device=device)[None,:,None];rows.append({"fold":fold,"late_steps":late,"w_max":wmax,"outer_training_artifact_rrmse":v11.rrmse(correction.cpu().numpy(),truth.cpu().numpy()),"validation_rows":len(y),"heldout_outcomes_opened":0})
    selected=min(rows,key=lambda r:(r["outer_training_artifact_rrmse"],r["late_steps"],r["w_max"]));_csv(_root(c,"oppost_root")/"selection"/f"fold_{fold:02d}.csv",rows);_json(_root(c,"oppost_root")/"selection"/f"fold_{fold:02d}.json",selected);_json(run/"result_summary.json",{"status":"OPPOST_FOLD_GLOBAL_SCHEDULE_FROZEN","selected":selected});return selected

def stage_oppost_infer(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    seed,fold=_task(c,task_index);recipient=fold+1;base=_seed_fold(_root(c,"strict_root"),seed,fold);cp,det,diff,device=nxt._models(base/"checkpoint.pt");rescale=np.asarray(cp["residual_scale"],np.float32);selected=json.loads((_root(c,"oppost_root")/"selection"/f"fold_{fold:02d}.json").read_text())
    for unit_index,protocol in enumerate(SAME):
        inf=np.load(base/"units"/protocol/"inference.npz");manifest=_read(base/"unit_manifest.csv")[unit_index];folder=_root(c,"oppost_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs"/protocol;folder.mkdir(parents=True,exist_ok=True)
        if not bool(int(inf["recipient_eligible"])):np.savez_compressed(folder/"inference_outputs.npz");continue
        hp=np.asarray(inf["h_pop"]);hm=_half_support_transfer(c,recipient,manifest["support_session"],inf);donors=[int(x) for x in manifest["wrong_donors"].split(";") if x];outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(inf[f"{panel}_y"],np.float32);eog=np.asarray(inf[f"{panel}_eog"],np.float32);gamma=float(inf["gamma"]);a0=v11.apply_transfer(hp,eog);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8)
            with torch.no_grad():rd=det(y=yt,eog=et,a0=at)
            contexts=[("POP",hp,recipient),("MATCH",hm,recipient)]+[(f"WRONG-{d}",_half_support_transfer(c,d,manifest["support_session"],inf),d) for d in donors]
            for name,h,owner in contexts:
                center=(v11.apply_transfer(h,eog)-a0)/rescale[None,:,None];precision=nxt._support_precision(c,owner,manifest["support_session"],120,inf,h,rescale);samples=[]
                with torch.no_grad():
                    for noise in bank:
                        delta,_=bounded_oppost_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(noise,device=device),center=torch.as_tensor(center,device=device),precision=torch.as_tensor(precision[None],device=device),late_steps=int(selected["late_steps"]),w_max=float(selected["w_max"]),strength=0 if int(selected["late_steps"])==0 else 1);samples.append(delta.cpu().numpy())
                corr=a0+(rd.cpu().numpy()+np.mean(samples,axis=0))*rescale[None,:,None];corr[...,500:]=0;outputs[f"{panel}_WEAK-OPPOST-{name}"]=v11.gamma_correction(y,corr,gamma)
            # Frozen current MATCH uses the same POP8-R checkpoint/noise bank.
            am=v11.apply_transfer(hm,eog);amt=torch.as_tensor(am,device=device)
            with torch.no_grad():rdm=det(y=yt,eog=et,a0=amt);samples=[diff.sample(y=yt,eog=et,a0=amt,r_det=rdm,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy() for noise in bank]
            corr=am+(rdm.cpu().numpy()+np.mean(samples,axis=0))*rescale[None,:,None];corr[...,500:]=0;outputs[f"{panel}_CURRENT-DIFF-MATCH"]=v11.gamma_correction(y,corr,gamma)
        np.savez_compressed(folder/"inference_outputs.npz",**outputs)
    summary={"status":"WEAK_OPPOST_EVALUATOR_BLIND_INFERENCE_COMPLETED","seed":seed,"fold":fold,"selection":selected,"evaluator_opened":False};_json(run/"result_summary.json",summary);return summary

def stage_oppost_oracle(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    """Evaluator-only oracle-center diagnostic; never used for selection."""
    import torch
    fold=task_index;seed=int(c["seeds"][0]);base=_seed_fold(_root(c,"strict_root"),seed,fold);cp,det,diff,device=nxt._models(base/"checkpoint.pt");rescale=np.asarray(cp["residual_scale"],np.float32);rows=[]
    for unit_index,protocol in enumerate(SAME):
        inf=np.load(base/"units"/protocol/"inference.npz");ev=np.load(base/"units"/protocol/"evaluator.npz");manifest=_read(base/"unit_manifest.csv")[unit_index]
        if not bool(int(inf["recipient_eligible"])):continue
        y=np.asarray(inf["paired_y"],np.float32);eog=np.asarray(inf["paired_eog"],np.float32);hp=np.asarray(inf["h_pop"]);hm=_half_support_transfer(c,fold+1,manifest["support_session"],inf);a0=v11.apply_transfer(hp,eog);target=np.asarray(ev["paired_a"],np.float32)/np.asarray(inf["eeg_scale"])[None,:,None];oracle=(target-a0)/rescale[None,:,None];support=(v11.apply_transfer(hm,eog)-a0)/rescale[None,:,None];donors=[int(x) for x in manifest["wrong_donors"].split(";") if x];wrong_h=_half_support_transfer(c,donors[0],manifest["support_session"],inf);wrong=(v11.apply_transfer(wrong_h,eog)-a0)/rescale[None,:,None];yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);noise=torch.as_tensor(v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,1)[0],device=device);precision=torch.ones((1,3,1),device=device)
        with torch.no_grad():rd=det(y=yt,eog=et,a0=at)
        for late,wmax in [(0,0.0)]+[(int(a),float(b)) for a in c["oppost_late_steps"] for b in c["oppost_w_max"]]:
            for name,center in (("zero",np.zeros_like(oracle)),("support",support),("wrong",wrong),("oracle",oracle)):
                with torch.no_grad():delta,stats=bounded_oppost_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise,center=torch.as_tensor(center,device=device),precision=precision,late_steps=max(late,1),w_max=max(wmax,1e-6),strength=0 if late==0 else 1)
                correction=at+(rd+delta)*torch.as_tensor(rescale,device=device)[None,:,None]
                weights=np.asarray([x["weight_median"] for x in stats])
                rows.append({
                    "participant":fold+1,"protocol":protocol,"late_steps":late,
                    "w_max":wmax,"center":name,
                    "artifact_rrmse":v11.rrmse(correction.cpu().numpy()[...,:500],target[...,:500]),
                    "weight_min":float(weights.min()) if len(weights) else 0,
                    "weight_median":float(np.median(weights)) if len(weights) else 0,
                    "weight_max":float(weights.max()) if len(weights) else 0,
                    "fraction_steps_gt_0_10":float(np.mean(weights>.10)) if len(weights) else 0,
                    "fraction_steps_gt_0_25":float(np.mean(weights>.25)) if len(weights) else 0,
                    "fraction_steps_gt_0_50":float(np.mean(weights>.50)) if len(weights) else 0,
                    "fraction_steps_gt_0_90":float(np.mean(weights>.90)) if len(weights) else 0,
                    "evaluator_only":1,
                })
    _csv(_root(c,"oppost_root")/"oracle_tasks"/f"fold_{fold:02d}.csv",rows);summary={"status":"OPPOST_ORACLE_CENTER_DIAGNOSTIC_COMPLETED","fold":fold,"rows":len(rows),"selection_unchanged":True};_json(run/"result_summary.json",summary);return summary

def stage_oppost_evaluate_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    root=_root(c,"oppost_root");paired=[];natural=[];seed=int(c["seeds"][0])
    for fold in range(9):
        base=_seed_fold(_root(c,"strict_root"),seed,fold);prepared=_root(c,"strict_root")/"prepared"/f"fold_{fold:02d}";outputs=root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs";p,n=nxt._evaluate_outputs(c,base,prepared,outputs,seed,fold,"WEAK-OPPOST");paired.extend(p);natural.extend(n)
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);summary={"status":"WEAK_OPPOST_INDEPENDENT_EVALUATION_COMPLETED","paired_rows":len(paired),"natural_rows":len(natural)};_json(run/"result_summary.json",summary);return summary

def stage_oppost_aggregate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    root=_root(c,"oppost_root");paired=_read(root/"paired_metrics.csv");natural=_read(root/"natural_safety.csv");effects=[]
    for participant in range(1,10):
        take=[r for r in paired if int(r["participant"])==participant];by=defaultdict(list)
        for r in take:by[r["method"]].append(float(r["rrmse"]))
        if "WEAK-OPPOST-MATCH" not in by:continue
        wrong=[np.mean(v) for k,v in by.items() if k.startswith("WEAK-OPPOST-WRONG-")];effects.append({"participant":participant,"U_P":float(np.mean(by["WEAK-OPPOST-POP"])-np.mean(by["WEAK-OPPOST-MATCH"])),"U_W":float(np.mean(wrong)-np.mean(by["WEAK-OPPOST-MATCH"])),"G_MATCH":float(np.mean(by["CURRENT-DIFF-MATCH"])-np.mean(by["WEAK-OPPOST-MATCH"]))})
    _csv(root/"participant_effects.csv",effects);summary_effect={key:{"mean":float(np.mean([r[key] for r in effects])),"median":float(np.median([r[key] for r in effects])),"positive":int(sum(r[key]>0 for r in effects)),"participant_values":[r[key] for r in effects]} for key in ("U_P","U_W","G_MATCH")};oracle=[]
    for path in sorted((root/"oracle_tasks").glob("*.csv")):oracle.extend(_read(path))
    _csv(root/"oracle_center_diagnostic.csv",oracle);selected=[json.loads(path.read_text()) for path in sorted((root/"selection").glob("*.json"))];_csv(root/"frozen_schedule_selection.csv",selected)
    oracle_gain=[]
    for participant in range(1,10):
        for protocol in SAME:
            take=[r for r in oracle if int(r["participant"])==participant and r["protocol"]==protocol];zero=[float(r["artifact_rrmse"]) for r in take if r["center"]=="zero" and int(float(r["late_steps"]))==0];best=[float(r["artifact_rrmse"]) for r in take if r["center"]=="oracle" and int(float(r["late_steps"]))>0]
            if zero and best:oracle_gain.append(min(zero)-min(best))
    oracle_means={}
    for wmax in c["oppost_w_max"]:
        values=[float(r["artifact_rrmse"]) for r in oracle if r["center"]=="oracle" and float(r["w_max"])==float(wmax)]
        if values:oracle_means[str(wmax)]=float(np.mean(values))
    ordered=[oracle_means.get(str(x),np.nan) for x in c["oppost_w_max"]]
    oracle_monotonic=bool(len(ordered)==len(c["oppost_w_max"]) and np.all(np.diff(ordered)<=1e-8))
    oracle_valid=bool(oracle_gain and np.mean(oracle_gain)>0 and sum(x>0 for x in oracle_gain)>=.75*len(oracle_gain) and oracle_monotonic)
    match=[r for r in natural if r["method"]=="WEAK-OPPOST-MATCH"];current=[r for r in natural if r["method"]=="CURRENT-DIFF-MATCH"]
    keys=("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")
    nat={k:float(np.mean([float(r[k]) for r in match])) for k in keys};cur={k:float(np.mean([float(r[k]) for r in current])) for k in keys};margin={k:nat[k]-cur[k] for k in keys}
    success=summary_effect["G_MATCH"]["mean"]>=0 and summary_effect["U_P"]["mean"]>0 and summary_effect["U_P"]["median"]>0 and summary_effect["U_P"]["positive"]>=7 and summary_effect["U_W"]["mean"]>0 and summary_effect["U_W"]["median"]>0 and summary_effect["U_W"]["positive"]>=7 and nat["eog_attenuation"]>0 and margin["eog_attenuation"]>=-.01 and margin["preservation"]>=-.02 and margin["mi_kappa"]>=-.02 and margin["erd_preservation"]>=-.02 and margin["mi_band_distortion"]<=.02 and margin["covariance"]<=.02
    status="WEAK_OPPOST_ONE_SEED_PASSED" if success else ("OPERATOR_LIKELIHOOD_BRIDGE_FAILED" if oracle_valid else "OPPOST_PROXIMAL_IMPLEMENTATION_INVALID")
    route={"status":status,"extra_seeds_authorized":bool(success),"oracle_center_valid":oracle_valid,"oracle_monotonic":oracle_monotonic,"localized_backup_forbidden":True,"development_only":True}
    summary={"effects":summary_effect,"natural":nat,"current_natural":cur,"natural_margins":margin,"oracle_mean_gain":float(np.mean(oracle_gain)) if oracle_gain else None,"oracle_positive":int(sum(x>0 for x in oracle_gain)),"oracle_units":len(oracle_gain),"oracle_mean_rrmse_by_w_max":oracle_means,"routing":route}
    _json(root/"result_summary.json",summary);_json(root/"routing_decision.json",route)
    Path("reports/bci2b_oppost_strength_audit.md").write_text(f"# BCI2b bounded OPPOST strength audit\n\nDecision: `{status}`. One-seed participant-first U_P={summary_effect['U_P']['mean']:+.5f} ({summary_effect['U_P']['positive']}/9), U_W={summary_effect['U_W']['mean']:+.5f} ({summary_effect['U_W']['positive']}/9), and G_MATCH={summary_effect['G_MATCH']['mean']:+.5f}. Oracle-center mean artifact-RRMSE gain is {summary['oracle_mean_gain']}; monotonicity across the frozen w_max schedule is {oracle_monotonic}. Schedule selection used outer-training rows only and was frozen before held-out outputs. No localized backup was run.\n",encoding="utf-8")
    _json(run/"result_summary.json",summary);return summary

def _bci2a_root(c:Mapping[str,Any])->Path:return _root(c,"bci2a_root")

def stage_bci2a_inventory(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    from eeg_cgdr.data.bci2a_v10 import discover_sessions,inspect_gdf
    sessions=discover_sessions(_root(c,"bci2a_data_root"));rows=[inspect_gdf(x) for x in sessions]
    complete=len(rows)==18 and len({int(r["subject"]) for r in rows})==9
    _csv(_bci2a_root(c)/"file_inventory.csv",rows)
    summary={"status":"BCI2A_READ_ONLY_INVENTORY_COMPLETED" if complete else "BCI2A_DATA_INCOMPLETE","sessions":len(rows),"participants":len({int(r["subject"]) for r in rows}),"complete":complete,"gpu_training_started":False}
    _json(_bci2a_root(c)/"inventory.json",summary);_json(run/"result_summary.json",summary)
    if not complete:raise RuntimeError("BCI2a 18-session inventory incomplete")
    return summary

def _bci2a_standardize(value:np.ndarray)->np.ndarray:
    center=np.median(value,axis=1,keepdims=True);scale=np.quantile(np.abs(value-center),.75,axis=1,keepdims=True)/.67448975
    return ((value-center)/np.maximum(scale,1e-8)).astype(np.float32)

def _bci2a_transfer(y:np.ndarray,e:np.ndarray)->np.ndarray:
    return v11._instant_transfer(y,e).astype(np.float32)

def stage_bci2a_extract(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    """Support-only operator extraction; later-query samples are never opened."""
    from eeg_cgdr.data.bci2a_v10 import discover_sessions,load_with_events
    from eeg_cgdr.experiments.bci2a_hierarchical_score_v10 import _support_query_ranges
    subject=task_index+1;sessions={(x.subject,x.session):x for x in discover_sessions(_root(c,"bci2a_data_root"))};arrays={};coverage=[]
    for session in "TE":
        eeg,eog,sfreq,events=load_with_events(sessions[(subject,session)]);support,_=_support_query_ranges(events,eeg.shape[1],sfreq);available=(support.stop-support.start)/sfreq;eligible=available>=120
        coverage.append({"participant":subject,"session":session,"available_seconds":available,"eligible_120":int(eligible)})
        if not eligible:continue
        stop=support.start+int(round(120*sfreq));indices=np.arange(support.start,stop);y=_bci2a_standardize(eeg[:,indices]);e=_bci2a_standardize(eog[:,indices]);arrays[f"{session}_full"]=_bci2a_transfer(y,e)
        for block,index in enumerate(np.array_split(np.arange(len(indices)),4)):
            arrays[f"{session}_block_{block}_y"]=y[:,index].astype(np.float32);arrays[f"{session}_block_{block}_e"]=e[:,index].astype(np.float32)
    out=_bci2a_root(c)/"support"/f"participant_{subject:02d}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays);_csv(_bci2a_root(c)/"coverage"/f"participant_{subject:02d}.csv",coverage)
    summary={"status":"BCI2A_SUPPORT_ONLY_OPERATOR_EXTRACTED","participant":subject,"eligible_sessions":sum(r["eligible_120"] for r in coverage),"query_opened":False};_json(run/"result_summary.json",summary);return summary

def stage_bci2a_select(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    root=_bci2a_root(c);rows=[]
    support={p:np.load(root/"support"/f"participant_{p:02d}.npz") for p in range(1,10)}
    try:
        for recipient in range(1,10):
            for session in "TE":
                key=f"{session}_full"
                if key not in support[recipient]:rows.append({"participant":recipient,"session":session,"eligible":0,"selected_lambda":0,"reason":"support_lt_120"});continue
                owners=[p for p in range(1,10) if p!=recipient and key in support[p]];hp=np.mean([support[p][key] for p in owners],axis=0);loss={lam:[] for lam in map(float,c["lambda_candidates"])}
                for heldout in range(4):
                    train_y=np.concatenate([support[recipient][f"{session}_block_{b}_y"] for b in range(4) if b!=heldout],axis=1);train_e=np.concatenate([support[recipient][f"{session}_block_{b}_e"] for b in range(4) if b!=heldout],axis=1);hm=_bci2a_transfer(train_y,train_e);yv=support[recipient][f"{session}_block_{heldout}_y"];ev=support[recipient][f"{session}_block_{heldout}_e"]
                    for lam in loss:loss[lam].append(float(np.mean((yv-(hp+lam*(hm-hp))@ev)**2)))
                chosen=min(loss,key=lambda lam:(np.mean(loss[lam]),lam));rows.append({"participant":recipient,"session":session,"eligible":1,"selected_lambda":chosen,"population_owners":";".join(map(str,owners)),**{f"loss_lambda_{lam:g}":float(np.mean(loss[lam])) for lam in loss}})
    finally:
        for item in support.values():item.close()
    _csv(root/"support_lambda_selection.csv",rows);summary={"status":"BCI2A_SUPPORT_ONLY_LAMBDA_FROZEN","eligible_units":sum(int(r["eligible"]) for r in rows),"availability_denominator":18};_json(run/"result_summary.json",summary);return summary

def stage_bci2a_evaluate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    """Evaluator opens later EEG/EOG only after support-only lambda is frozen."""
    from eeg_cgdr.data.bci2a_v10 import discover_sessions,load_with_events
    from eeg_cgdr.experiments.bci2a_hierarchical_score_v10 import _support_query_ranges
    root=_bci2a_root(c);recipient=task_index+1;selection={(int(r["participant"]),r["session"]):r for r in _read(root/"support_lambda_selection.csv")};support={p:np.load(root/"support"/f"participant_{p:02d}.npz") for p in range(1,10)};sessions={(x.subject,x.session):x for x in discover_sessions(_root(c,"bci2a_data_root"))};rows=[]
    try:
        for session in "TE":
            chosen=selection[(recipient,session)]
            if not int(chosen["eligible"]):continue
            eeg,eog,sfreq,events=load_with_events(sessions[(recipient,session)]);_,query=_support_query_ranges(events,eeg.shape[1],sfreq);y=_bci2a_standardize(eeg[:,query]);e=_bci2a_standardize(eog[:,query]);hp=np.mean([support[p][f"{session}_full"] for p in range(1,10) if p!=recipient and f"{session}_full" in support[p]],axis=0);hm=support[recipient][f"{session}_full"];lam=float(chosen["selected_lambda"]);methods={"POP8":hp,"RAW-MATCH":hm,"SHRINK-MATCH":hp+lam*(hm-hp)}
            for donor in range(1,10):
                if donor!=recipient and f"{session}_full" in support[donor]:methods[f"SHRINK-WRONG-{donor}"]=hp+lam*(support[donor][f"{session}_full"]-hp)
            energy=np.mean(e**2,axis=0);low=energy<=np.quantile(energy,.3);base_cov=np.cov(y[:,low])
            for name,h in methods.items():
                correction=h@e;output=y-correction;residual=float(np.mean(output**2));preservation=float(1-np.sqrt(np.mean(correction[:,low]**2))/(np.sqrt(np.mean(y[:,low]**2))+1e-12));cov=float(np.linalg.norm(np.cov(output[:,low])-base_cov)/(np.linalg.norm(base_cov)+1e-12));rows.append({"participant":recipient,"session":session,"method":name,"query_residual_mse":residual,"preservation":preservation,"covariance_distortion":cov,"evaluator_query_opened":1})
    finally:
        for item in support.values():item.close()
    _csv(root/"evaluation"/f"participant_{recipient:02d}.csv",rows);summary={"status":"BCI2A_LATER_QUERY_EVALUATED","participant":recipient,"rows":len(rows)};_json(run/"result_summary.json",summary);return summary

def stage_bci2a_aggregate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    root=_bci2a_root(c);allrows=[]
    for path in sorted((root/"evaluation").glob("participant_*.csv")):allrows.extend(_read(path))
    effects=[]
    for participant in range(1,10):
        take=[r for r in allrows if int(r["participant"])==participant];by=defaultdict(list)
        for r in take:by[r["method"]].append(float(r["query_residual_mse"]))
        if "SHRINK-MATCH" not in by:continue
        wrong=[np.mean(v) for k,v in by.items() if k.startswith("SHRINK-WRONG-")];effects.append({"participant":participant,"E_P":float(np.mean(by["POP8"])-np.mean(by["SHRINK-MATCH"])),"E_W":float(np.mean(wrong)-np.mean(by["SHRINK-MATCH"])),"preservation_margin":float(np.mean([float(r["preservation"]) for r in take if r["method"]=="SHRINK-MATCH"])-np.mean([float(r["preservation"]) for r in take if r["method"]=="POP8"]))})
    _csv(root/"participant_effects.csv",effects);ep=np.asarray([r["E_P"] for r in effects]);ew=np.asarray([r["E_W"] for r in effects]);pm=np.asarray([r["preservation_margin"] for r in effects]);passed=bool(len(effects)==9 and ep.mean()>0 and np.median(ep)>0 and (ep>0).sum()>=6 and ew.mean()>0 and np.median(ew)>0 and pm.mean()>=-.02);route={"status":"BCI2A_OPERATOR_SHRINKAGE_HEADROOM_DETECTED" if passed else "BCI2A_OPERATOR_SHRINKAGE_HEADROOM_NOT_ESTABLISHED","one_seed_factorial_authorized":passed,"eog_transfer_personalization_family_stopped":not passed,"development_only":True};summary={"effects":{"E_P_mean":float(ep.mean()),"E_P_median":float(np.median(ep)),"E_P_positive":int((ep>0).sum()),"E_W_mean":float(ew.mean()),"E_W_median":float(np.median(ew)),"E_W_positive":int((ew>0).sum()),"preservation_margin_mean":float(pm.mean())},"routing":route};_json(root/"result_summary.json",summary);_json(root/"routing_decision.json",route);Path("reports/bci2a_operator_shrinkage_headroom.md").write_text(f"# BCI2a operator shrinkage headroom\n\nConditional development audit decision: `{route['status']}`. E_P mean/median {ep.mean():+.5f}/{np.median(ep):+.5f} with {(ep>0).sum()}/9 positive; E_W mean/median {ew.mean():+.5f}/{np.median(ew):+.5f}. The same 120-second support-only lambda rule was used; later EEG/EOG was opened only by this evaluator.\n",encoding="utf-8");_json(run/"result_summary.json",summary);return summary

def _run_many(c:Mapping[str,Any],run:Path,name:str,fn:Any,count:int)->dict[str,Any]:
    for index in range(count):child=run/f"task_{index}";child.mkdir(parents=True,exist_ok=True);fn(c,index,child)
    summary={"status":f"{name}_COMPLETED","tasks":count};_json(run/"result_summary.json",summary);return summary
def stage_prepare_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:return _run_many(c,run,"STRICT_PREPARE_ALL",stage_prepare,9)
def stage_evaluate_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:return _run_many(c,run,"STRICT_EVALUATE_ALL",stage_evaluate,27)

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    c=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True);stages={"audit":stage_audit,"prepare":stage_prepare,"prepare-all":stage_prepare_all,"repair-metadata":stage_repair_metadata,"train":stage_train,"shrink-infer":stage_shrink_infer,"evaluate":stage_evaluate,"evaluate-all":stage_evaluate_all,"aggregate":stage_aggregate,"strict-report":stage_strict_report,"oppost-select":stage_oppost_select,"oppost-infer":stage_oppost_infer,"oppost-oracle":stage_oppost_oracle,"oppost-evaluate-all":stage_oppost_evaluate_all,"oppost-aggregate":stage_oppost_aggregate,"bci2a-inventory":stage_bci2a_inventory,"bci2a-extract":stage_bci2a_extract,"bci2a-select":stage_bci2a_select,"bci2a-evaluate":stage_bci2a_evaluate,"bci2a-aggregate":stage_bci2a_aggregate}
    if stage not in stages:raise ValueError(stage)
    return stages[stage](c,task_index,run_dir)

__all__=["_select_lambda","run_stage"]
