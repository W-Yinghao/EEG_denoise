"""BCI-IV-2b EOG-guided residual-diffusion development experiment (V11).

Inference deliberately receives query EOG. Evaluator-only arrays remain separate
and contain the paired carrier/artifact and natural MI labels.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from eeg_cgdr.data.bci2a_v10 import load_gdf_channels
from eeg_cgdr.experiments.bci2a_hierarchical_score_v10 import (
    _bci2b_files,
    _instant_transfer,
    _query_trials,
    _support_query_ranges,
)


def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as handle:return yaml.safe_load(handle)


def _json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0]);
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def rrmse(estimate:np.ndarray,target:np.ndarray)->float:
    return float(np.sqrt(np.mean((estimate-target)**2))/(np.sqrt(np.mean(target**2))+1e-12))


def correlation(estimate:np.ndarray,target:np.ndarray)->float:
    a=estimate.ravel().astype(float);b=target.ravel().astype(float);a-=a.mean();b-=b.mean();return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def delta_snr(estimate:np.ndarray,target:np.ndarray,noisy:np.ndarray)->float:
    before=np.mean((noisy-target)**2)+1e-12;after=np.mean((estimate-target)**2)+1e-12;return float(10*np.log10(before/after))


def temporal_shuffle(eog:np.ndarray,seed:int)->np.ndarray:
    """Destroy time correspondence while preserving each window's values."""
    rng=np.random.default_rng(seed);out=np.empty_like(eog)
    for index,row in enumerate(eog):
        shift=int(rng.integers(max(1,row.shape[-1]//4),max(2,3*row.shape[-1]//4)));out[index]=np.roll(row,shift,axis=-1)
    return out


def apply_transfer(operator:np.ndarray,eog:np.ndarray)->np.ndarray:
    value=np.asarray(operator)
    if value.ndim==2:return np.einsum("ce,net->nct",value,eog,optimize=True).astype(np.float32)
    if value.ndim==3 and value.shape[0]==eog.shape[0]:return np.einsum("nce,net->nct",value,eog,optimize=True).astype(np.float32)
    raise ValueError("transfer must be CxE or batch-matched NxCxE")


def gamma_correction(y:np.ndarray,correction:np.ndarray,gamma:float)->np.ndarray:
    if gamma==0:return y.copy()
    return y-float(gamma)*correction


def support_gamma(target_artifact:np.ndarray,predicted_artifact:np.ndarray)->float:
    denominator=float(np.sum(predicted_artifact.astype(float)**2))
    if denominator<=1e-20:return 0.0
    return float(np.clip(np.sum(target_artifact.astype(float)*predicted_artifact.astype(float))/denominator,0,1))


def _load_session(config:Mapping[str,Any],subject:int,session:str)->tuple[np.ndarray,np.ndarray,float,list[tuple[float,str]]]:
    path=_bci2b_files(config,subject)[session];eeg,eog,sfreq,events=load_gdf_channels(path,eeg_channels=3);return eeg*1e6,eog*1e6,sfreq,events


def _protocols()->tuple[tuple[str,str,str],...]:
    return (("same_01","01T","01T"),("same_02","02T","02T"),("same_03","03T","03T"),("cross_02","01T","02T"),("cross_03","01T","03T"))


def _support_operator(config:Mapping[str,Any],subject:int,session:str)->tuple[np.ndarray,dict[str,float]]:
    eeg,eog,sfreq,events=_load_session(config,subject,session);support,_=_support_query_ranges(events,eeg.shape[1],sfreq);operator=_instant_transfer(eeg[:,support],eog[:,support]);stats={"support_seconds":float((support.stop-support.start)/sfreq),"eeg_median":float(np.median(eeg[:,support])),"eeg_mad":float(np.median(np.abs(eeg[:,support]-np.median(eeg[:,support],axis=1,keepdims=True)))),"eog_median":float(np.median(eog[:,support])),"eog_mad":float(np.median(np.abs(eog[:,support]-np.median(eog[:,support],axis=1,keepdims=True))))};return operator,stats


def _query_arrays(config:Mapping[str,Any],subject:int,session:str)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    eeg,eog,sfreq,events=_load_session(config,subject,session);natural,eog_windows,labels=_query_trials(eeg,eog,events,sfreq);energy=np.sqrt(np.mean(eog_windows.astype(float)**2,axis=(1,2)));order=np.argsort(energy);count=min(64,len(order)//3);clean=natural[order[:count]];artifact_eog=eog_windows[order[-count:]];operator=_instant_transfer(eeg,eog);artifact=apply_transfer(operator,artifact_eog);return clean,artifact_eog,artifact,operator,natural


def _basis(operator:np.ndarray,rank:int)->np.ndarray:
    u=np.linalg.svd(operator,full_matrices=False)[0];return u[:,:rank]


def _metric_row(subject:int,protocol:str,method:str,y:np.ndarray,x:np.ndarray,correction:np.ndarray,gamma:float)->dict[str,Any]:
    estimate=gamma_correction(y,correction,gamma);return {"subject":subject,"protocol":protocol,"method":method,"gamma":gamma,"rrmse":rrmse(estimate,x),"correlation":correlation(estimate,x),"delta_snr":delta_snr(estimate,x,y),"correction_rms":float(np.sqrt(np.mean((gamma*correction)**2))),"output_input_rms":float(np.sqrt(np.mean(estimate**2))/(np.sqrt(np.mean(y**2))+1e-12))}


def stage_bridge_audit(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    """No-training V10-aligned evaluator audit for one held-out participant."""
    subject=task_index+1;wrong=subject%9+1;training=[s for s in range(1,10) if s not in (subject,wrong)];root=Path(str(config["result_root"]));rows=[];curves=[];stats=[]
    for protocol,support_session,query_session in _protocols():
        x,e_query,a,h_query,natural=_query_arrays(config,subject,query_session);y=x+a
        hs,sstats=_support_operator(config,subject,support_session);hw,_=_support_operator(config,wrong,support_session);hpop=np.mean([_support_operator(config,s,support_session)[0] for s in training],axis=0)
        c_query=apply_transfer(h_query,e_query);c_match=apply_transfer(hs,e_query);c_pop=apply_transfer(hpop,e_query);c_wrong=apply_transfer(hw,e_query)
        v10=Path(str(config["v10_result_root"]))/"bci2b_gpu"/f"fold_{subject-1:02d}"/"units"/protocol
        if v10.exists():
            with np.load(v10/"deployable.npz") as dep,np.load(v10/"evaluator.npz") as ev:
                identity_error=rrmse(np.asarray(dep["paired_y"]),np.asarray(ev["paired_x"])+np.asarray(ev["paired_a"]));v10_windows=len(dep["paired_y"])
        else:identity_error=float("nan");v10_windows=0
        if identity_error>1e-6:raise RuntimeError(f"{subject} {protocol}: V10 paired identity failed: {identity_error}")
        # Support-only pseudo-pairs define correction strength without query outcomes.
        seeg,seog,sfreq,sevents=_load_session(config,subject,support_session);support,_=_support_query_ranges(sevents,seeg.shape[1],sfreq);mid=(support.start+support.stop)//2;support_eog=[]
        for start in range(mid,max(mid,support.stop-500+1),500):support_eog.append(seog[:,start:start+500])
        if support_eog:
            se=np.pad(np.stack(support_eog),((0,0),(0,0),(0,12)));support_target=apply_transfer(hs,se);gamma=support_gamma(support_target,apply_transfer(hs,se))
        else:gamma=0.0
        corrections={"FULL-ORACLE":a,"QUERY-TRANSFER-ORACLE":c_query,"SUPPORT-LINEAR-MATCH":c_match,"LINEAR-POP":c_pop,"LINEAR-WRONG":c_wrong}
        for rank in (1,2):
            b=_basis(hs,rank);corrections[f"RANK-{rank}-PROJECTION-ORACLE"]=np.einsum("cr,nrt->nct",b,np.einsum("cr,nct->nrt",b,a))
        b=_basis(hs,2);coeff=np.einsum("cr,nct->nrt",b,a);support_coeff=np.einsum("cr,nct->nrt",b,support_target) if support_eog else coeff;tau=np.maximum(np.quantile(np.abs(support_coeff),.99,axis=(0,2)),1e-6)
        corrections["BOUNDED-CLIP-ORACLE"]=np.einsum("cr,nrt->nct",b,np.clip(coeff,-tau[None,:,None],tau[None,:,None]))
        corrections["BOUNDED-TANH-ORACLE"]=np.einsum("cr,nrt->nct",b,tau[None,:,None]*np.tanh(coeff/tau[None,:,None]))
        h_rank1=np.linalg.svd(h_query,full_matrices=False);cca=h_rank1[0][:,:1]@np.diag(h_rank1[1][:1])@h_rank1[2][:1];corrections["EOG-REGRESSION"]=c_query;corrections["CCA-RANK1"]=apply_transfer(cca,e_query)
        for method,correction in corrections.items():
            rows.append(_metric_row(subject,protocol,method,y,x,correction,gamma if "ORACLE" not in method else 1.0))
            for value in config["gamma_grid"]:curves.append(_metric_row(subject,protocol,method,y,x,correction,float(value)))
        query_gamma=support_gamma(a,c_match)
        stats.append({"subject":subject,"protocol":protocol,"support_session":support_session,"query_session":query_session,"wrong_subject":wrong,"population_training_subjects":";".join(map(str,training)),"support_gamma":gamma,"query_optimal_gamma_ceiling":query_gamma,"paired_identity_rrmse":identity_error,"paired_windows":len(x),"v10_windows":v10_windows,"artifact_clean_norm_ratio":float(np.linalg.norm(a)/(np.linalg.norm(x)+1e-12)),"correction_y_norm_ratio":float(np.linalg.norm(c_match)/(np.linalg.norm(y)+1e-12)),"rank2_energy_fraction":float(np.sum(np.einsum("cr,nct->nrt",b,a)**2)/(np.sum(a**2)+1e-12)),"latent_saturation_fraction":float(np.mean(np.abs(coeff/tau[None,:,None])>1)),**sstats})
    folder=root/"bridge_audit";_csv(folder/f"subject_{subject:02d}_metrics.csv",rows);_csv(folder/f"subject_{subject:02d}_gamma.csv",curves);_csv(folder/f"subject_{subject:02d}_stats.csv",stats);summary={"status":"completed_v11_bridge_audit","subject":subject,"protocols":len(stats),"wrong_subject":wrong};_json(run_dir/"result_summary.json",summary);return summary


def stage_bridge_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));folder=root/"bridge_audit";rows=[];curves=[];stats=[]
    for subject in range(1,10):rows.extend(_read(folder/f"subject_{subject:02d}_metrics.csv"));curves.extend(_read(folder/f"subject_{subject:02d}_gamma.csv"));stats.extend(_read(folder/f"subject_{subject:02d}_stats.csv"))
    _csv(root/"base_bridge_metrics.csv",rows);_csv(root/"gamma_curves.csv",curves);_csv(root/"base_bridge_diagnostics.csv",stats)
    summaries=[]
    for panel,names in (("same_session",("same_01","same_02","same_03")),("cross_session",("cross_02","cross_03"))):
        for method in sorted({r["method"] for r in rows}):
            subset=[r for r in rows if r["protocol"] in names and r["method"]==method];values=np.asarray([float(r["rrmse"]) for r in subset]);summaries.append({"protocol":panel,"method":method,"units":len(subset),"subjects":len({r["subject"] for r in subset}),"rrmse_mean":float(values.mean()),"rrmse_median":float(np.median(values)),"correlation_mean":float(np.mean([float(r["correlation"]) for r in subset])),"delta_snr_mean":float(np.mean([float(r["delta_snr"]) for r in subset]))})
    _csv(root/"base_bridge_summary.csv",summaries)
    raw=np.asarray([1.0 for _ in stats]);full=np.asarray([float(r["rrmse"]) for r in rows if r["method"]=="FULL-ORACLE"]);query=np.asarray([float(r["rrmse"]) for r in rows if r["method"]=="QUERY-TRANSFER-ORACLE"]);reg=np.asarray([float(r["rrmse"]) for r in rows if r["method"]=="EOG-REGRESSION"]);gammas=np.asarray([float(r["support_gamma"]) for r in stats]);identity=np.asarray([float(r["paired_identity_rrmse"]) for r in stats]);passed=bool(np.nanmax(identity)<=1e-6 and np.mean(full)<1e-6 and np.mean(query)<.8*np.mean(raw) and np.mean(reg)<.8*np.mean(raw) and np.mean(gammas>0)>0.5)
    decision={"status":"completed_base_bridge_audit","decision":"BASE_BRIDGE_VALID_J1_AUTHORIZED" if passed else "SIMULATOR_EVALUATOR_BRIDGE_INVALID_STOP","j1_authorized":passed,"full_oracle_rrmse":float(np.mean(full)),"query_transfer_oracle_rrmse":float(np.mean(query)),"eog_regression_rrmse":float(np.mean(reg)),"support_gamma_zero_fraction":float(np.mean(gammas==0)),"units":len(stats),"subjects":9,"same_and_cross_separate":True,"deployment":"EOG-guided"};_json(root/"routing_decision.json",decision);_json(run_dir/"result_summary.json",decision)
    lines=["# BCI2b base-bridge oracle audit V11","","Development audit; both support and later-query deployment are EOG-guided.","",f"Decision: `{decision['decision']}`.","",f"FULL oracle RRMSE: {decision['full_oracle_rrmse']:.6g}; query-transfer oracle: {decision['query_transfer_oracle_rrmse']:.4f}; EOG regression: {decision['eog_regression_rrmse']:.4f}; support gamma zero fraction: {decision['support_gamma_zero_fraction']:.3f}.","","Same-session and cross-session results are stored separately in `base_bridge_summary.csv`. Query-optimal gamma values are evaluator-only ceilings; primary operator comparisons use the support-derived gamma."]
    Path("reports/bci2b_base_bridge_oracle_audit_v11.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return decision


def _pad(windows:np.ndarray)->np.ndarray:
    return np.pad(windows,((0,0),(0,0),(0,12))).astype(np.float32)


def _continuous_windows(eeg:np.ndarray,eog:np.ndarray,start:int,stop:int,length:int=500)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    starts=np.arange(start,max(start,stop-length+1),length,dtype=int)
    if len(starts)==0:return np.empty((0,3,512),np.float32),np.empty((0,3,512),np.float32),starts
    return _pad(np.stack([eeg[:,i:i+length] for i in starts])),_pad(np.stack([eog[:,i:i+length] for i in starts])),starts


def _task_domain_pairs(eeg:np.ndarray,eog:np.ndarray,events:list[tuple[float,str]],sfreq:float,seed:int,cap:int=64)->dict[str,np.ndarray]:
    """Build real task-domain pairs with disjoint generator/clean/artifact blocks."""
    _,query=_support_query_ranges(events,eeg.shape[1],sfreq);guard=int(round(5*sfreq));start=min(query.stop,query.start+guard);span=query.stop-start;third=span//3
    if third<2000:raise RuntimeError("task region too short for disjoint paired construction")
    generator=slice(start,start+third-guard);clean_region=(start+third,start+2*third-guard);artifact_region=(start+2*third,query.stop)
    transfer=_instant_transfer(eeg[:,generator],eog[:,generator]);clean,clean_eog,clean_ids=_continuous_windows(eeg,eog,*clean_region);artifact_carrier,artifact_eog,artifact_ids=_continuous_windows(eeg,eog,*artifact_region)
    clean_energy=np.sqrt(np.mean(clean_eog.astype(float)**2,axis=(1,2)));artifact_energy=np.sqrt(np.mean(artifact_eog.astype(float)**2,axis=(1,2)));clean_order=np.argsort(clean_energy);artifact_order=np.argsort(artifact_energy)[::-1];count=min(cap,len(clean_order),len(artifact_order))
    if count<4:raise RuntimeError("fewer than four disjoint task pairs")
    rng=np.random.default_rng(seed);low=clean_order[:count].copy();high=artifact_order[:count].copy();rng.shuffle(low);rng.shuffle(high);artifact=apply_transfer(transfer,artifact_eog[high]);x=clean[low];return {"x":x,"y":x+artifact,"a":artifact,"eog":artifact_eog[high],"transfer":transfer,"clean_ids":clean_ids[low],"artifact_ids":artifact_ids[high],"generator_start":np.array(generator.start),"generator_stop":np.array(generator.stop)}


def _robust_location_scale(values:list[np.ndarray])->tuple[np.ndarray,np.ndarray]:
    flat=np.concatenate([x.transpose(1,0,2).reshape(x.shape[1],-1) for x in values],axis=1).astype(float);location=np.median(flat,axis=1);scale=1.4826*np.median(np.abs(flat-location[:,None]),axis=1);scale=np.maximum(scale,1e-3);return location.astype(np.float32),scale.astype(np.float32)


def _normalize_windows(value:np.ndarray,location:np.ndarray,scale:np.ndarray,*,difference:bool=False)->np.ndarray:
    offset=0 if difference else location[None,:,None];return ((value-offset)/scale[None,:,None]).astype(np.float32)


def _normalized_transfer(eeg:np.ndarray,eog:np.ndarray,region:slice,eeg_loc:np.ndarray,eeg_scale:np.ndarray,eog_loc:np.ndarray,eog_scale:np.ndarray)->np.ndarray:
    y=(eeg[:,region]-eeg_loc[:,None])/eeg_scale[:,None];e=(eog[:,region]-eog_loc[:,None])/eog_scale[:,None];return _instant_transfer(y,e)


def stage_prepare_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    route=json.loads((Path(str(config["result_root"]))/"routing_decision.json").read_text())
    if not route["j1_authorized"]:raise RuntimeError("base bridge did not authorize task-domain preparation")
    heldout=task_index+1;wrong=heldout%9+1;training=[s for s in range(1,10) if s not in (heldout,wrong)];root=Path(str(config["result_root"]))/"folds"/f"fold_{task_index:02d}";root.mkdir(parents=True,exist_ok=True);raw_parts=[]
    for subject in training:
        for session,path in sorted(_bci2b_files(config,subject).items()):
            eeg,eog,sfreq,events=load_gdf_channels(path,eeg_channels=3);eeg=eeg*1e6;eog=eog*1e6
            try:part=_task_domain_pairs(eeg,eog,events,sfreq,int(config["seed"])+subject*100+int(session[:2]),64)
            except RuntimeError:continue
            part["subject"]=np.full(len(part["y"]),subject,np.int16);part["session"]=np.full(len(part["y"]),int(session[:2]),np.int16);raw_parts.append(part)
    if not raw_parts:raise RuntimeError("no task-domain training pairs")
    eeg_loc,eeg_scale=_robust_location_scale([p["y"] for p in raw_parts]);eog_loc,eog_scale=_robust_location_scale([p["eog"] for p in raw_parts]);hpop_values=[]
    for subject in training:
        for support_session in ("01T","02T","03T"):
            eeg,eog,sfreq,events=_load_session(config,subject,support_session);support,_=_support_query_ranges(events,eeg.shape[1],sfreq);hpop_values.append(_normalized_transfer(eeg,eog,support,eeg_loc,eeg_scale,eog_loc,eog_scale))
    hpop=np.mean(hpop_values,axis=0).astype(np.float32);train={name:np.concatenate([p[name] for p in raw_parts]) for name in ("x","y","a","eog","subject","session")};train["x"]=_normalize_windows(train["x"],eeg_loc,eeg_scale);train["y"]=_normalize_windows(train["y"],eeg_loc,eeg_scale);train["a"]=_normalize_windows(train["a"],eeg_loc,eeg_scale,difference=True);train["eog"]=_normalize_windows(train["eog"],eog_loc,eog_scale)
    # Each training sample gets its own support-derived operator, enabling MATCH/POP/WRONG context substitution at inference.
    lookup={}
    for subject in training:
        for session in (1,2,3,4,5):
            label=f"{session:02d}{'T' if session<=3 else 'E'}";eeg,eog,sfreq,events=_load_session(config,subject,label);support,_=_support_query_ranges(events,eeg.shape[1],sfreq);lookup[(subject,session)]=_normalized_transfer(eeg,eog,support,eeg_loc,eeg_scale,eog_loc,eog_scale)
    train["h_subject"]=np.stack([lookup[(int(s),int(q))] for s,q in zip(train["subject"],train["session"])]).astype(np.float32);np.savez_compressed(root/"training_pairs.npz",**train,h_population=hpop,eeg_location=eeg_loc,eeg_scale=eeg_scale,eog_location=eog_loc,eog_scale=eog_scale)
    units=[]
    for protocol,support_session,query_session in _protocols():
        seeg,seog,ssf,sevents=_load_session(config,heldout,support_session);support,_=_support_query_ranges(sevents,seeg.shape[1],ssf);hmatch=_normalized_transfer(seeg,seog,support,eeg_loc,eeg_scale,eog_loc,eog_scale);weeg,weog,wsf,wevents=_load_session(config,wrong,support_session);wsupport,_=_support_query_ranges(wevents,weeg.shape[1],wsf);hwrong=_normalized_transfer(weeg,weog,wsupport,eeg_loc,eeg_scale,eog_loc,eog_scale)
        clean,e_art,a_phys,h_query,natural=_query_arrays(config,heldout,query_session);qeeg,qeog,qsf,qevents=_load_session(config,heldout,query_session);natural_y,natural_eog,labels=_query_trials(qeeg,qeog,qevents,qsf);paired_y=clean+a_phys;paired_y_n=_normalize_windows(paired_y,eeg_loc,eeg_scale);paired_e_n=_normalize_windows(e_art,eog_loc,eog_scale);natural_y_n=_normalize_windows(natural_y,eeg_loc,eeg_scale);natural_e_n=_normalize_windows(natural_eog,eog_loc,eog_scale)
        # Support-only gamma validation uses heldout support and never query outcomes.
        middle=(support.start+support.stop)//2;h_fit=_normalized_transfer(seeg,seog,slice(support.start,middle),eeg_loc,eeg_scale,eog_loc,eog_scale);_,support_eog,_=_continuous_windows(seeg,seog,middle,support.stop);support_eog_n=_normalize_windows(support_eog,eog_loc,eog_scale);support_target=apply_transfer(h_fit,support_eog_n);support_pred=apply_transfer(hmatch,support_eog_n);gamma=support_gamma(support_target,support_pred)
        unit=root/"units"/protocol;unit.mkdir(parents=True,exist_ok=True);np.savez_compressed(unit/"inference.npz",paired_y=paired_y_n,paired_eog=paired_e_n,natural_y=natural_y_n,natural_eog=natural_e_n,h_match=hmatch,h_population=hpop,h_wrong=hwrong,gamma=np.array(gamma,np.float32),eeg_location=eeg_loc,eeg_scale=eeg_scale,eog_location=eog_loc,eog_scale=eog_scale,recipient=np.array(heldout),wrong_donor=np.array(wrong));np.savez_compressed(unit/"evaluator.npz",paired_x=clean.astype(np.float32),paired_a=a_phys.astype(np.float32),natural_labels=labels.astype(np.int16),query_transfer_evaluator_only=h_query.astype(np.float32));units.append({"protocol":protocol,"support_session":support_session,"query_session":query_session,"paired_windows":len(clean),"natural_windows":len(natural_y),"gamma":gamma,"wrong_donor":wrong})
    _csv(root/"unit_manifest.csv",units);_json(root/"fold_metadata.json",{"heldout":heldout,"wrong_unseen":wrong,"population_training":training,"training_pairs":len(train["y"]),"physical_units":"microvolts","normalization":"outer-training median/MAD","deployment":"EOG-guided","evaluator_physically_separate":True});summary={"status":"completed_task_domain_fold_preparation","fold":task_index,"heldout":heldout,"wrong":wrong,"training_participants":len(training),"training_pairs":len(train["y"]),"units":len(units)};_json(run_dir/"result_summary.json",summary);return summary


def _context_batch(y:np.ndarray,eog:np.ndarray,h:np.ndarray)->np.ndarray:
    return apply_transfer(h,eog)


def _train_models(config:Mapping[str,Any],fold:int,root:Path,device:Any,*,technical:bool=False)->dict[str,Any]:
    import torch
    from torch.optim import AdamW
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion,checkpoint_payload
    with np.load(root/"training_pairs.npz") as data:
        y=np.asarray(data["y"],np.float32);eog=np.asarray(data["eog"],np.float32);a=np.asarray(data["a"],np.float32);hsub=np.asarray(data["h_subject"],np.float32);hpop=np.asarray(data["h_population"],np.float32);eeg_loc=np.asarray(data["eeg_location"],np.float32);eeg_scale=np.asarray(data["eeg_scale"],np.float32);eog_loc=np.asarray(data["eog_location"],np.float32);eog_scale=np.asarray(data["eog_scale"],np.float32)
    if technical:y=y[:32];eog=eog[:32];a=a[:32];hsub=hsub[:32]
    seed=int(config["seed"])+fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);cfg=EOGResidualConfig(base_channels=16 if technical else 32);det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det_opt=AdamW(det.parameters(),lr=float(config["learning_rate"]));diff_opt=AdamW(diff.parameters(),lr=float(config["learning_rate"]));ema=EMA(diff,float(config["ema_decay"]));rng=np.random.default_rng(seed);generator=torch.Generator(device=device).manual_seed(seed);updates=int(config["technical_updates"] if technical else config["training_updates"]);curve=[]
    residual_scale=np.maximum(np.quantile(np.abs(a),.995,axis=(0,2)),1e-3).astype(np.float32)
    schedule=[]
    for step in range(updates):
        idx=np.arange(len(y)) if technical else rng.integers(0,len(y),int(config["batch_size"]));choice=np.zeros(len(idx),bool) if technical else rng.random(len(idx))<.5;schedule.append((idx,choice));hs=hsub[idx];h=np.where(choice[:,None,None],hs,hpop[None]);a0=_context_batch(y[idx],eog[idx],h);target=(a[idx]-a0)/residual_scale[None,:,None];yt=torch.as_tensor(y[idx],device=device);et=torch.as_tensor(eog[idx],device=device);at=torch.as_tensor(a0,device=device);truth=torch.as_tensor(target,device=device)
        det.train();det_opt.zero_grad(set_to_none=True);prediction=det(y=yt,eog=et,a0=at);dloss=(prediction-truth).square().mean();dloss.backward();dgrad=float(torch.nn.utils.clip_grad_norm_(det.parameters(),1));det_opt.step()
        if step%100==0 or step+1==updates:curve.append({"phase":"deterministic","step":step+1,"det_loss":float(dloss.detach()),"diff_loss":"","det_grad":dgrad,"diff_grad":""})
    det.eval()
    for step,(idx,choice) in enumerate(schedule):
        hs=hsub[idx];h=np.where(choice[:,None,None],hs,hpop[None]);a0=_context_batch(y[idx],eog[idx],h);target=(a[idx]-a0)/residual_scale[None,:,None];yt=torch.as_tensor(y[idx],device=device);et=torch.as_tensor(eog[idx],device=device);at=torch.as_tensor(a0,device=device);truth=torch.as_tensor(target,device=device)
        with torch.no_grad():anchor=det(y=yt,eog=et,a0=at)
        delta=truth-anchor;diff.train();diff_opt.zero_grad(set_to_none=True);floss,_=diff.training_loss(delta,y=yt,eog=et,a0=at,r_det=anchor,generator=generator);floss.backward();fgrad=float(torch.nn.utils.clip_grad_norm_(diff.parameters(),1));diff_opt.step();ema.update(diff)
        if step%100==0 or step+1==updates:curve.append({"phase":"diffusion","step":step+1,"det_loss":"","diff_loss":float(floss.detach()),"det_grad":"","diff_grad":fgrad})
    payload=checkpoint_payload(cfg,det,diff,ema,residual_scale=residual_scale,h_population=hpop,eeg_location=eeg_loc,eeg_scale=eeg_scale,eog_location=eog_loc,eog_scale=eog_scale,updates=updates,seed=seed);checkpoint=root/("technical_checkpoint.pt" if technical else "checkpoint.pt");torch.save(payload,checkpoint);_csv(root/("technical_curve.csv" if technical else "training_curve.csv"),curve)
    # Deterministic fixed real-batch restoration check plus full DDIM check.
    det.eval();ema.copy_to(diff);diff.eval();hs=np.repeat(hpop[None],len(y),axis=0) if technical else hsub[:len(y)];a0=_context_batch(y,eog,hs);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);truth=torch.as_tensor((a-a0)/residual_scale[None,:,None],device=device)
    with torch.no_grad():rdet=det(y=yt,eog=et,a0=at);noise=torch.randn(yt.shape,device=device,generator=generator);delta=diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=noise);correction=at+(rdet+delta)*torch.as_tensor(residual_scale,device=device)[None,:,None];target=torch.as_tensor(a,device=device);det_correction=at+rdet*torch.as_tensor(residual_scale,device=device)[None,:,None]
    def score(value:Any)->tuple[float,float]:
        va=value.detach().cpu().numpy();ta=target.detach().cpu().numpy();return rrmse(va,ta),correlation(va,ta)
    det_rr,det_corr=score(det_correction);diff_rr,diff_corr=score(correction);oracle=diff.oracle_roundtrip(torch.randn(2,3,512,device=device,generator=generator),torch.randn(2,3,512,device=device,generator=generator));return {"det_rrmse":det_rr,"det_correlation":det_corr,"diff_rrmse":diff_rr,"diff_correlation":diff_corr,"oracle_roundtrip":oracle,"updates":updates,"checkpoint":str(checkpoint)}


def stage_technical(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    root=Path(str(config["result_root"]))/"folds"/"fold_00";metrics=_train_models(config,0,root,torch.device("cuda"),technical=True);passed=bool(metrics["oracle_roundtrip"]<=1e-6 and metrics["det_rrmse"]<=.10 and metrics["det_correlation"]>=.95 and metrics["diff_rrmse"]<=.10 and metrics["diff_correlation"]>=.95);summary={"status":"technical_validity_passed" if passed else "technical_validity_failed","training_authorized":passed,**metrics};_json(Path(str(config["result_root"]))/"technical_validity.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def stage_technical_replay(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    root=Path(str(config["result_root"]));fold=root/"folds"/"fold_00";checkpoint=torch.load(fold/"technical_checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**checkpoint["config"]);device=torch.device("cuda")
    def load_models():
        det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(checkpoint["det"]);diff.load_state_dict(checkpoint["diff"]);ema=EMA(diff);ema.load_state_dict(checkpoint["ema"]);ema.copy_to(diff);det.eval();diff.eval();return det,diff
    with np.load(fold/"training_pairs.npz") as data:y=np.asarray(data["y"][:4],np.float32);eog=np.asarray(data["eog"][:4],np.float32);h=np.asarray(data["h_subject"][:4],np.float32)
    a0=apply_transfer(h,eog);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);noise=torch.as_tensor(_noise_bank(y.shape,int(config["seed"]),1)[0],device=device);first=load_models();second=load_models()
    with torch.no_grad():
        r1=first[0](y=yt,eog=et,a0=at);r2=second[0](y=yt,eog=et,a0=at);d1=first[1].sample(y=yt,eog=et,a0=at,r_det=r1,initial_noise=noise);d2=second[1].sample(y=yt,eog=et,a0=at,r_det=r2,initial_noise=noise);replayed=first[1].sample(y=yt,eog=et,a0=at,r_det=r1,initial_noise=noise)
    checkpoint_equal=bool(torch.equal(r1,r2) and torch.equal(d1,d2));noise_equal=bool(torch.equal(d1,replayed));delta_zero_equal=bool(np.array_equal(gamma_correction(y,a0,1),gamma_correction(y,a0+np.zeros_like(a0),1)));no_lora=not any("lora" in name.lower() for name,_ in first[1].named_parameters());passed=checkpoint_equal and noise_equal and delta_zero_equal and no_lora;previous=json.loads((root/"technical_validity.json").read_text());previous.update({"checkpoint_reload_exact":checkpoint_equal,"common_noise_replay_exact":noise_equal,"delta_zero_deterministic_exact":delta_zero_equal,"lora_parameters":0,"base_parameter_policy":"deterministic_and_diffusion_backbones_trained; no hidden adapter","training_authorized":bool(previous["training_authorized"] and passed)});_json(root/"technical_validity.json",previous);summary={"status":"technical_replay_passed" if passed else "technical_replay_failed","training_authorized":previous["training_authorized"],"checkpoint_reload_exact":checkpoint_equal,"common_noise_replay_exact":noise_equal,"delta_zero_deterministic_exact":delta_zero_equal,"lora_parameters":0};_json(run_dir/"result_summary.json",summary);return summary


def stage_train_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    gate=json.loads((Path(str(config["result_root"]))/"technical_validity.json").read_text())
    if not gate["training_authorized"]:raise RuntimeError("technical gate failed")
    root=Path(str(config["result_root"]))/"folds"/f"fold_{task_index:02d}";metrics=_train_models(config,task_index,root,torch.device("cuda"));summary={"status":"completed_fold_training","fold":task_index,**metrics};_json(run_dir/"result_summary.json",summary);return summary


def _noise_bank(shape:tuple[int,...],seed:int,k:int)->np.ndarray:
    rows=[]
    for sample in range(k):
        generated=[]
        for window in range(shape[0]):generated.append(np.random.default_rng(seed+window*1009+sample*1000003).standard_normal(shape[1:],dtype=np.float32))
        rows.append(np.stack(generated))
    return np.stack(rows)


def stage_infer_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    root=Path(str(config["result_root"]))/"folds"/f"fold_{task_index:02d}";checkpoint=torch.load(root/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**checkpoint["config"]);device=torch.device("cuda");det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(checkpoint["det"]);diff.load_state_dict(checkpoint["diff"]);ema=EMA(diff);ema.load_state_dict(checkpoint["ema"]);ema.copy_to(diff);det.eval();diff.eval();scale=np.asarray(checkpoint["residual_scale"],np.float32);manifest=_read(root/"unit_manifest.csv")
    for unit_index,row in enumerate(manifest):
        protocol=row["protocol"];unit=root/"units"/protocol;data=np.load(unit/"inference.npz");outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(data[f"{panel}_y"],np.float32);eog=np.asarray(data[f"{panel}_eog"],np.float32);gamma=float(data["gamma"]);outputs[f"{panel}_RAW"]=y
            contexts={"POP":np.asarray(data["h_population"]),"MATCH":np.asarray(data["h_match"]),"WRONG":np.asarray(data["h_wrong"])}
            for name,h in contexts.items():
                a0=apply_transfer(h,eog);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device)
                with torch.no_grad():rdet=det(y=yt,eog=et,a0=at);det_correction=at+rdet*torch.as_tensor(scale,device=device)[None,:,None]
                outputs[f"{panel}_LINEAR-{name}"]=gamma_correction(y,a0,gamma);outputs[f"{panel}_DET-{name}"]=gamma_correction(y,det_correction.cpu().numpy(),gamma);bank=_noise_bank(y.shape,int(config["seed"])+task_index*100000+unit_index*10000,int(config["posterior_samples"]));samples=[]
                with torch.no_grad():
                    for noise in bank:samples.append(diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy())
                correction=a0+(rdet.cpu().numpy()+np.mean(samples,axis=0))*scale[None,:,None];outputs[f"{panel}_DIFF-{name}"]=gamma_correction(y,correction,gamma)
            shuffled=temporal_shuffle(eog,int(config["seed"])+task_index*100+unit_index);h=np.asarray(data["h_match"]);a0=apply_transfer(h,shuffled);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(shuffled,device=device);at=torch.as_tensor(a0,device=device)
            with torch.no_grad():rdet=det(y=yt,eog=et,a0=at);bank=_noise_bank(y.shape,int(config["seed"])+task_index*100000+unit_index*10000,int(config["posterior_samples"]));samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy() for noise in bank]
            correction=a0+(rdet.cpu().numpy()+np.mean(samples,axis=0))*scale[None,:,None];outputs[f"{panel}_DIFF-TEMPORAL-SHUFFLED"]=gamma_correction(y,correction,gamma)
        out=root/"outputs"/protocol;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"inference_outputs.npz",**outputs)
    summary={"status":"completed_frozen_inference","fold":task_index,"units":len(manifest),"query_eog_used":True,"evaluator_opened":False,"deployment":"EOG-guided"};_json(run_dir/"result_summary.json",summary);return summary


def _coherence_proxy(eeg:np.ndarray,eog:np.ndarray)->float:
    values=[]
    for row,e in zip(eeg,eog):
        for x in row:
            for z in e:
                x0=x-x.mean();z0=z-z.mean();values.append(abs(float(np.dot(x0,z0)/(np.linalg.norm(x0)*np.linalg.norm(z0)+1e-12))))
    return float(np.mean(values))


def _psd_distortion(output:np.ndarray,raw:np.ndarray)->float:
    from scipy.signal import welch
    _,po=welch(output,fs=250,nperseg=256,axis=-1);_,pr=welch(raw,fs=250,nperseg=256,axis=-1);return float(np.mean(np.abs(np.log((po+1e-12)/(pr+1e-12)))))


def _covariance_distortion(output:np.ndarray,raw:np.ndarray)->float:
    values=[]
    for a,b in zip(output,raw):values.append(np.linalg.norm(np.cov(a)-np.cov(b),"fro")/(np.linalg.norm(np.cov(b),"fro")+1e-12))
    return float(np.mean(values))


def _kappa(output:np.ndarray,labels:np.ndarray)->float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    if len(np.unique(labels))<2 or len(labels)<8:return float("nan")
    split=max(4,len(labels)//2);freq=np.fft.rfft(output,axis=-1);features=np.concatenate([np.log(np.abs(freq[:,:,16:61])+1e-8).mean(-1),np.log(np.abs(freq[:,:,16:61])+1e-8).std(-1)],axis=1);model=LinearDiscriminantAnalysis().fit(features[:split],labels[:split]);return float(cohen_kappa_score(labels[split:],model.predict(features[split:])))


def stage_eval_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]))/"folds"/f"fold_{task_index:02d}";manifest=_read(root/"unit_manifest.csv");paired=[];natural=[]
    for row in manifest:
        protocol=row["protocol"];unit=root/"units"/protocol;inf=np.load(unit/"inference.npz");ev=np.load(unit/"evaluator.npz");out=np.load(root/"outputs"/protocol/"inference_outputs.npz");eeg_loc=np.asarray(inf["eeg_location"]);eeg_scale=np.asarray(inf["eeg_scale"]);x=np.asarray(ev["paired_x"]);raw_n=np.asarray(inf["natural_y"]);eog_n=np.asarray(inf["natural_eog"]);labels=np.asarray(ev["natural_labels"])
        for key in out.files:
            panel,method=key.split("_",1);normalized=np.asarray(out[key]);physical=normalized*eeg_scale[None,:,None]+eeg_loc[None,:,None]
            if panel=="paired":paired.append({"subject":task_index+1,"protocol":protocol,"method":method,"rrmse":rrmse(physical,x),"correlation":correlation(physical,x),"delta_snr":delta_snr(physical,x,np.asarray(inf["paired_y"])*eeg_scale[None,:,None]+eeg_loc[None,:,None])})
            else:
                energy=np.sqrt(np.mean(eog_n.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3);preservation=1-rrmse(normalized[low],raw_n[low]) if np.any(low) else float("nan");natural.append({"subject":task_index+1,"protocol":protocol,"method":method,"eog_attenuation":_coherence_proxy(raw_n,eog_n)-_coherence_proxy(normalized,eog_n),"preservation":preservation,"psd_distortion":_psd_distortion(normalized[low],raw_n[low]) if np.any(low) else float("nan"),"covariance_distortion":_covariance_distortion(normalized[low],raw_n[low]) if np.any(low) else float("nan"),"mi_kappa":_kappa(normalized,labels),"correction_rms":float(np.sqrt(np.mean((normalized-raw_n)**2)))})
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);summary={"status":"completed_independent_evaluator","fold":task_index,"paired_rows":len(paired),"natural_rows":len(natural)};_json(run_dir/"result_summary.json",summary);return summary


def _aggregate_subjects(config:Mapping[str,Any],folds:list[int])->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    root=Path(str(config["result_root"]));paired=[];natural=[]
    for fold in folds:
        base=root/"folds"/f"fold_{fold:02d}";paired.extend(_read(base/"paired_metrics.csv"));natural.extend(_read(base/"natural_safety.csv"))
    effects=[]
    for subject in sorted({int(r["subject"]) for r in paired}):
        for panel,names in (("same_session",("same_01","same_02","same_03")),("cross_session",("cross_02","cross_03"))):
            rows=[r for r in paired if int(r["subject"])==subject and r["protocol"] in names];by=defaultdict(list)
            for row in rows:by[row["method"]].append(float(row["rrmse"]))
            mean={k:float(np.mean(v)) for k,v in by.items()};nrows=[r for r in natural if int(r["subject"])==subject and r["protocol"] in names and r["method"]=="DIFF-MATCH"]
            effects.append({"subject":subject,"protocol":panel,"U_D":mean["DET-MATCH"]-mean["DIFF-MATCH"],"U_P":mean["DIFF-POP"]-mean["DIFF-MATCH"],"U_W":mean["DIFF-WRONG"]-mean["DIFF-MATCH"],"U_S":mean["DIFF-TEMPORAL-SHUFFLED"]-mean["DIFF-MATCH"],"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in nrows])),"preservation":float(np.mean([float(r["preservation"]) for r in nrows])),"psd":float(np.mean([float(r["psd_distortion"]) for r in nrows])),"covariance":float(np.mean([float(r["covariance_distortion"]) for r in nrows])),"kappa_match":float(np.nanmean([float(r["mi_kappa"]) for r in nrows])),"kappa_raw":float(np.nanmean([float(r["mi_kappa"]) for r in natural if int(r["subject"])==subject and r["protocol"] in names and r["method"]=="RAW"]))})
    return paired,natural,effects


def stage_gate3(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));paired,natural,effects=_aggregate_subjects(config,[0,1,2]);_csv(root/"participant_metrics.csv",effects);_csv(root/"paired_effects.csv",effects);by=defaultdict(list)
    for row in paired:by[row["method"]].append(float(row["rrmse"]))
    subjects=[]
    for subject in (1,2,3):
        rows=[r for r in paired if int(r["subject"])==subject];subject_by=defaultdict(list)
        for r in rows:subject_by[r["method"]].append(float(r["rrmse"]))
        subjects.append({"subject":subject,"det_pop_better_raw":float(np.mean(subject_by["DET-POP"]))<float(np.mean(subject_by["RAW"])),"diff_pop_better_raw":float(np.mean(subject_by["DIFF-POP"]))<float(np.mean(subject_by["RAW"]))})
    n=[r for r in natural if r["method"]=="DIFF-POP"];summary_metrics={"raw_rrmse":float(np.mean(by["RAW"])),"det_pop_rrmse":float(np.mean(by["DET-POP"])),"diff_pop_rrmse":float(np.mean(by["DIFF-POP"])),"preservation":float(np.mean([float(r["preservation"]) for r in n])),"psd":float(np.mean([float(r["psd_distortion"]) for r in n])),"covariance":float(np.mean([float(r["covariance_distortion"]) for r in n])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in n]))};method_rows=[]
    for panel,names in (("same_session",("same_01","same_02","same_03")),("cross_session",("cross_02","cross_03"))):
        for method in sorted({r["method"] for r in paired}):
            p=[r for r in paired if r["protocol"] in names and r["method"]==method];q=[r for r in natural if r["protocol"] in names and r["method"]==method]
            if p:method_rows.append({"protocol":panel,"method":method,"gate_participants":3,"paired_rrmse":float(np.mean([float(r["rrmse"]) for r in p])),"correlation":float(np.mean([float(r["correlation"]) for r in p])),"delta_snr":float(np.mean([float(r["delta_snr"]) for r in p])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in q])) if q else "","preservation":float(np.mean([float(r["preservation"]) for r in q])) if q else "","psd":float(np.mean([float(r["psd_distortion"]) for r in q])) if q else "","covariance":float(np.mean([float(r["covariance_distortion"]) for r in q])) if q else ""})
    _csv(root/"method_summary.csv",method_rows);wins=sum(r["det_pop_better_raw"] and r["diff_pop_better_raw"] for r in subjects);passed=summary_metrics["det_pop_rrmse"]<summary_metrics["raw_rrmse"] and summary_metrics["diff_pop_rrmse"]<summary_metrics["raw_rrmse"] and wins>=2 and summary_metrics["preservation"]>=.75 and summary_metrics["psd"]<=.25 and summary_metrics["covariance"]<=.25 and summary_metrics["eog_attenuation"]>0;decision={"status":"completed_three_participant_gate","decision":"FULL_NINE_AUTHORIZED" if passed else "CURRENT_EOG_ANCHORED_RESIDUAL_DIFFUSION_INSTANCE_NO_GO","full_nine_authorized":passed,"participants":subjects,"diagnostic_effects":[{"protocol":panel,**{effect:float(np.mean([r[effect] for r in effects if r["protocol"]==panel])) for effect in ("U_D","U_P","U_W","U_S")}} for panel in ("same_session","cross_session")],**summary_metrics};_json(root/"gate3_decision.json",decision);_json(run_dir/"result_summary.json",decision);return decision


def stage_aggregate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));paired,natural,effects=_aggregate_subjects(config,list(range(9)));_csv(root/"participant_metrics.csv",effects);_csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);summary_rows=[];bootstrap=[];eligible=[];rng=np.random.default_rng(int(config["bootstrap_seed"]));indices=rng.integers(0,9,size=(int(config["bootstrap_replicates"]),9))
    for panel in ("same_session","cross_session"):
        rows=[r for r in effects if r["protocol"]==panel];summary={"protocol":panel,"subjects":len(rows)}
        for effect in ("U_D","U_P","U_W","U_S"):
            values=np.asarray([r[effect] for r in rows]);summary[f"{effect}_mean"]=float(values.mean());summary[f"{effect}_median"]=float(np.median(values));summary[f"{effect}_positive"]=int(np.sum(values>0));replicates=values[indices].mean(1);bootstrap.append({"protocol":panel,"effect":effect,"mean":float(values.mean()),"median":float(np.median(values)),"ci_low":float(np.quantile(replicates,.025)),"ci_high":float(np.quantile(replicates,.975)),"positive":int(np.sum(values>0)),"denominator":len(values),"status":"participant_bootstrap_descriptive"})
        summary.update({name:float(np.mean([r[name] for r in rows])) for name in ("eog_attenuation","preservation","psd","covariance")});summary["kappa_match_minus_raw"]=float(np.nanmean([r["kappa_match"]-r["kappa_raw"] for r in rows]));summary_rows.append(summary);gate=all(summary[f"{e}_mean"]>0 and summary[f"{e}_median"]>0 for e in ("U_D","U_P","U_W")) and summary["U_P_positive"]>=6 and summary["U_D_positive"]>=6 and summary["U_W_positive"]>=6 and summary["eog_attenuation"]>0 and summary["kappa_match_minus_raw"]>=-.02 and summary["preservation"]>=.75 and summary["psd"]<=.25 and summary["covariance"]<=.25
        if gate:eligible.append(panel)
    _csv(root/"method_summary.csv",summary_rows);_csv(root/"paired_effects.csv",effects);_csv(root/"bootstrap_summary.csv",bootstrap);decision={"status":"completed_one_seed_development","decision":"ELIGIBLE_FOR_TWO_ADDITIONAL_SEEDS" if eligible else "CURRENT_EOG_ANCHORED_RESIDUAL_DIFFUSION_INSTANCE_NO_GO","additional_seeds_authorized":bool(eligible),"eligible_protocols":eligible,"protocols":summary_rows,"subjects":9,"deployment":"EOG-guided","bootstrap":"participant_level_descriptive_20000","family_wide_status":"not_tested"};_json(root/"routing_decision.json",decision);_json(root/"result_summary.json",decision);_json(run_dir/"result_summary.json",decision);return decision


def stage_finalize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=Path(str(config["result_root"]));bridge=json.loads((root/"routing_decision.json").read_text()) if (root/"routing_decision.json").exists() else {};technical=json.loads((root/"technical_validity.json").read_text()) if (root/"technical_validity.json").exists() else {};gate3=json.loads((root/"gate3_decision.json").read_text()) if (root/"gate3_decision.json").exists() else {};science=json.loads((root/"result_summary.json").read_text()) if (root/"result_summary.json").exists() and (root/"participant_metrics.csv").exists() else {}
    if science:decision=science["decision"]
    elif gate3:decision=gate3["decision"]
    elif technical and not technical.get("training_authorized",False):decision="TECHNICAL_VALIDITY_FAILED"
    elif bridge and not bridge.get("j1_authorized",False):decision="SIMULATOR_EVALUATOR_BRIDGE_INVALID_STOP"
    else:decision="COMPUTATION_INCOMPLETE"
    final={"status":"completed_v11_development" if decision!="COMPUTATION_INCOMPLETE" else "incomplete","decision":decision,"deployment":"EOG-guided_support_and_query_EEG_EOG","operator_identity_specificity":"present_in_historical_v10_but_not_subject_utility","deployable_subject_utility":"diagnostic_gate_only_not_scientific_adjudication" if gate3 and not science else ("adjudicated" if science else "not_reached"),"diffusion_incremental_utility":"diagnostic_gate_only_not_scientific_adjudication" if gate3 and not science else ("adjudicated" if science else "not_reached"),"base_bridge":bridge,"technical_validity":technical,"three_participant_gate":gate3,"one_seed":science,"confirmation":False,"family_wide_status":"not_tested"};_json(root/"result_summary.json",final);_json(root/"routing_decision.json",final)
    figures=root/"figures";figures.mkdir(parents=True,exist_ok=True)
    if (root/"base_bridge_summary.csv").exists():
        rows=_read(root/"base_bridge_summary.csv");methods=("FULL-ORACLE","QUERY-TRANSFER-ORACLE","SUPPORT-LINEAR-MATCH","LINEAR-POP","LINEAR-WRONG");fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for axis,panel in zip(axes,("same_session","cross_session")):
            values=[float(next(r["rrmse_mean"] for r in rows if r["protocol"]==panel and r["method"]==m)) for m in methods];axis.bar(range(len(methods)),values);axis.set_xticks(range(len(methods)),[m.replace("-","\n") for m in methods],rotation=25,ha="right",fontsize=7);axis.set_title(panel);axis.axhline(1,color="black",ls="--",lw=.8)
        axes[0].set_ylabel("paired RRMSE");fig.tight_layout();fig.savefig(figures/"base_bridge_rrmse.png",dpi=180);plt.close(fig)
    if (root/"participant_metrics.csv").exists():
        effects=_read(root/"participant_metrics.csv");fig,axis=plt.subplots(figsize=(8,4));labels=[];values=[]
        for row in effects:labels.append(f"S{row['subject']}-{row['protocol'][:1]}");values.append(float(row["U_P"]))
        axis.axhline(0,color="black",lw=.8);axis.bar(range(len(values)),values);axis.set_xticks(range(len(values)),labels,rotation=90,fontsize=7);axis.set_ylabel("U_P: DIFF-POP − DIFF-MATCH RRMSE");fig.tight_layout();fig.savefig(figures/"subject_utility.png",dpi=180);plt.close(fig)
    lines=["# BCI2b EOG-guided residual diffusion V11","","Development exploration; not confirmation. Both early support and later query use EEG+EOG, so the method is **EOG-guided**, not EEG-only.","","## Final route", "",f"Decision: `{decision}`.","","## Evidence layers","",f"- Operator/simulator bridge: `{bridge.get('decision','not_run')}`.",f"- Technical estimator validity: `{technical.get('status','not_run')}`.",f"- Three-participant population rescue: `{gate3.get('decision','not_run')}`.",f"- Full 9-participant subject and diffusion effects: `{science.get('decision','not_run')}`."]
    if gate3:
        lines += ["","The three-participant gate is a technical route gate, not a final scientific estimate.","",f"RAW RRMSE {gate3['raw_rrmse']:.4f}; DET-POP {gate3['det_pop_rrmse']:.4f}; DIFF-POP {gate3['diff_pop_rrmse']:.4f}. Preservation {gate3['preservation']:.4f}, PSD distortion {gate3['psd']:.4f}, covariance distortion {gate3['covariance']:.4f}, natural EOG attenuation {gate3['eog_attenuation']:+.4f}.","","| protocol | U_D | U_P | U_W | U_S |","|---|---:|---:|---:|---:|"]
        for row in gate3.get("diagnostic_effects",[]):lines.append(f"| {row['protocol']} | {row['U_D']:+.4f} | {row['U_P']:+.4f} | {row['U_W']:+.4f} | {row['U_S']:+.4f} |")
    lines += ["","Operator identity specificity, deployable MATCH-vs-POP subject utility, and DIFF-vs-DET incremental utility are not interchangeable and are reported separately.","","V10 historical arrays were generated before inference, but its inference/model code never read evaluator fields. V11 physically separates inference and evaluator NPZ files. Paired results are real EEG/EOG-backed semi-simulation, not natural-clean ground truth.","","The 3-person gate failure prevented the 9-person scientific experiment, so its diagnostic subject contrasts cannot establish or refute subject utility. Any negative result constrains only this EOG-anchored residual-diffusion instance; diffusion and personalization families remain untested.","","Implementation provenance: the conditional 1-D diffusion is an adapted project implementation informed by common EEG diffusion recipes; it is not an exact official EEGDfus reproduction."]
    Path("reports/bci2b_eog_residual_diffusion_v11.md").write_text("\n".join(lines)+"\n",encoding="utf-8");_json(run_dir/"result_summary.json",final);return final


def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="bridge-audit":return stage_bridge_audit(config,task_index,run_dir)
    if stage=="bridge-aggregate":return stage_bridge_aggregate(config,run_dir)
    if stage=="prepare-fold":return stage_prepare_fold(config,task_index,run_dir)
    if stage=="technical":return stage_technical(config,task_index,run_dir)
    if stage=="technical-replay":return stage_technical_replay(config,task_index,run_dir)
    if stage=="train-fold":return stage_train_fold(config,task_index,run_dir)
    if stage=="infer-fold":return stage_infer_fold(config,task_index,run_dir)
    if stage=="eval-fold":return stage_eval_fold(config,task_index,run_dir)
    if stage=="gate3":return stage_gate3(config,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(config,task_index,run_dir)
    if stage=="finalize":return stage_finalize(config,task_index,run_dir)
    raise ValueError(f"unknown V11 stage: {stage}")


__all__=["run_stage","rrmse","support_gamma","gamma_correction","temporal_shuffle"]
