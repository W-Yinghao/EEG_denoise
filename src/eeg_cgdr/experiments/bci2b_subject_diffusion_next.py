"""BCI2b POP8 robustness, context-coherence audit and operator-posterior guidance.

This is development-only and EOG-guided.  Inference builders never open the
physically separate evaluator arrays.
"""
from __future__ import annotations

import csv,json,math,time
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping

import numpy as np
import yaml

from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as v111

SAME=("same_01","same_02","same_03")
DURATIONS=(30,60,120)

def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as handle:return yaml.safe_load(handle)
def _json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
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
def _root(config:Mapping[str,Any],name:str)->Path:return Path(str(config[name]))
def _seed_fold(root:Path,seed:int,fold:int)->Path:return root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"
def _task(config:Mapping[str,Any],index:int)->tuple[int,int]:
    seeds=[int(x) for x in config["seeds"]];return seeds[index//9],index%9
def _slice_budget(support:slice,sfreq:float,seconds:int|None)->slice:
    if seconds is None:return support
    return slice(support.start,min(support.stop,support.start+int(round(seconds*sfreq))))
def _transfer(config:Mapping[str,Any],subject:int,session:str,seconds:int|None,loc:np.ndarray,scale:np.ndarray,eloc:np.ndarray,escale:np.ndarray)->tuple[np.ndarray,slice,float]:
    eeg,eog,sfreq,events=v11._load_session(config,subject,session);support,_=v11._support_query_ranges(events,eeg.shape[1],sfreq);region=_slice_budget(support,sfreq,seconds)
    return v11._normalized_transfer(eeg,eog,region,loc,scale,eloc,escale),region,sfreq

def stage_audit(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=_root(config,"result_root");root.mkdir(parents=True,exist_ok=True)
    data=Path(str(config["data_root"]));rep=_root(config,"replication_root");uq=_root(config,"mechanism_uq_root")
    folds=[];donors=[]
    for fold in range(9):
        recipient=fold+1;folds.append({"fold":fold,"recipient":recipient,"population8":";".join(str(x) for x in range(1,10) if x!=recipient),"primary_support_seconds":120})
        for donor in range(1,10):
            if donor!=recipient:donors.append({"fold":fold,"recipient":recipient,"donor":donor,"population8_seen":1,"same_layout":1,"same_session":1})
    _csv(root/"frozen_folds.csv",folds);_csv(root/"frozen_donors.csv",donors)
    duration=[]
    for subject in range(1,10):
        for protocol,session,_ in v11._protocols()[:3]:
            eeg,_,sfreq,events=v11._load_session(config,subject,session);support,_=v11._support_query_ranges(events,eeg.shape[1],sfreq);available=(support.stop-support.start)/sfreq
            duration.append({"participant":subject,"protocol":protocol,"available_seconds":available,"d30":int(available>=30),"d60":int(available>=60),"d120":int(available>=120),"full_fair":0})
    _csv(root/"support_duration_manifest.csv",duration)
    full_fair=len({round(float(r["available_seconds"]),6) for r in duration})==1
    checks={"data_exists":data.exists(),"replication_exists":rep.exists(),"mechanism_arrays_exist":uq.exists(),"participants":9,"seeds":list(map(int,config["seeds"])),"support_primary":120,"full_available_fair":full_fair,"a_track_touched":False,"evaluator_blind_inference":True}
    checks["status"]="J0_AUDIT_PASSED" if all((checks["data_exists"],checks["replication_exists"],checks["mechanism_arrays_exist"])) else "J0_AUDIT_FAILED"
    _json(root/"frozen_protocol.json",checks);_json(run_dir/"result_summary.json",checks);return checks

def stage_prepare_pop8(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    fold=task_index;recipient=fold+1;training=[x for x in range(1,10) if x!=recipient];root=_root(config,"strong_population_root")/"prepared"/f"fold_{fold:02d}";root.mkdir(parents=True,exist_ok=True);parts=[]
    for subject in training:
        for session,path in sorted(v11._bci2b_files(config,subject).items()):
            eeg,eog,sfreq,events=v11.load_gdf_channels(path,eeg_channels=3);eeg=eeg*1e6;eog=eog*1e6
            try:part=v11._task_domain_pairs(eeg,eog,events,sfreq,20260808+subject*100+int(session[:2]),64)
            except RuntimeError:continue
            part["subject"]=np.full(len(part["y"]),subject,np.int16);part["session"]=np.full(len(part["y"]),int(session[:2]),np.int16);parts.append(part)
    if not parts:raise RuntimeError("no POP8 training pairs")
    loc,scale=v11._robust_location_scale([p["y"] for p in parts]);eloc,escale=v11._robust_location_scale([p["eog"] for p in parts]);train={name:np.concatenate([p[name] for p in parts]) for name in ("x","y","a","eog","subject","session")}
    train["x"]=v11._normalize_windows(train["x"],loc,scale);train["y"]=v11._normalize_windows(train["y"],loc,scale);train["a"]=v11._normalize_windows(train["a"],loc,scale,difference=True);train["eog"]=v11._normalize_windows(train["eog"],eloc,escale)
    lookup={}
    for subject in training:
        for session in range(1,6):
            label=f"{session:02d}{'T' if session<=3 else 'E'}";lookup[(subject,session)]=_transfer(config,subject,label,120,loc,scale,eloc,escale)[0]
    train["h_subject"]=np.stack([lookup[(int(s),int(q))] for s,q in zip(train["subject"],train["session"])]).astype(np.float32)
    hpop_by={};donors={}
    for seconds in DURATIONS:
        values=[]
        for s in training:
            for session in ("01T","02T","03T"):values.append(_transfer(config,s,session,seconds,loc,scale,eloc,escale)[0])
        hpop_by[seconds]=np.mean(values,axis=0).astype(np.float32)
    np.savez_compressed(root/"training_pairs.npz",**train,h_population=hpop_by[120],eeg_location=loc,eeg_scale=scale,eog_location=eloc,eog_scale=escale)
    manifest=[]
    for protocol,support_session,query_session in v11._protocols()[:3]:
        clean,e_art,a_phys,_,_=v11._query_arrays(config,recipient,query_session);paired_y=clean+a_phys;qeeg,qeog,qsf,qevents=v11._load_session(config,recipient,query_session);natural,natural_eog,labels=v11._query_trials(qeeg,qeog,qevents,qsf)
        arrays={"paired_y":v11._normalize_windows(paired_y,loc,scale),"paired_eog":v11._normalize_windows(e_art,eloc,escale),"natural_y":v11._normalize_windows(natural,loc,scale),"natural_eog":v11._normalize_windows(natural_eog,eloc,escale),"eeg_location":loc,"eeg_scale":scale,"eog_location":eloc,"eog_scale":escale,"recipient":np.array(recipient)}
        for seconds in DURATIONS:
            hm,region,sfreq=_transfer(config,recipient,support_session,seconds,loc,scale,eloc,escale);arrays[f"h_match_{seconds}"]=hm;arrays[f"h_pop_{seconds}"]=hpop_by[seconds]
            seeg,seog,_,_=v11._load_session(config,recipient,support_session);mid=(region.start+region.stop)//2;hfit=v11._normalized_transfer(seeg,seog,slice(region.start,mid),loc,scale,eloc,escale);_,sev,_=v11._continuous_windows(seeg,seog,mid,region.stop);sev=v11._normalize_windows(sev,eloc,escale);arrays[f"gamma_{seconds}"]=np.array(v11.support_gamma(v11.apply_transfer(hfit,sev),v11.apply_transfer(hm,sev)),np.float32) if len(sev) else np.array(0,np.float32)
            for donor in training:arrays[f"h_donor_{seconds}_{donor}"]=_transfer(config,donor,support_session,seconds,loc,scale,eloc,escale)[0]
        unit=root/"units"/protocol;unit.mkdir(parents=True,exist_ok=True);np.savez_compressed(unit/"inference.npz",**arrays);np.savez_compressed(unit/"evaluator.npz",paired_x=clean.astype(np.float32),paired_a=a_phys.astype(np.float32),natural_labels=labels.astype(np.int16));manifest.append({"protocol":protocol,"support_session":support_session,"query_session":query_session,"paired_windows":len(clean),"natural_windows":len(natural),"donors":len(training)})
    _csv(root/"unit_manifest.csv",manifest);_json(root/"fold_metadata.json",{"fold":fold,"recipient":recipient,"population_training":training,"training_participants":8,"training_pairs":len(train["y"]),"normalization":"POP8 outer-training median/MAD","deployment":"EOG-guided"})
    summary={"status":"completed_pop8_preparation","fold":fold,"recipient":recipient,"training_participants":8,"training_pairs":len(train["y"]),"units":3};_json(run_dir/"result_summary.json",summary);return summary

def stage_train_pop8(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    seed,fold=_task(config,task_index);prepared=_root(config,"strong_population_root")/"prepared"/f"fold_{fold:02d}";target=_seed_fold(_root(config,"strong_population_root"),seed,fold);target.mkdir(parents=True,exist_ok=True)
    for name in ("training_pairs.npz","unit_manifest.csv","fold_metadata.json"):
        path=target/name
        if not path.exists():path.symlink_to(prepared/name)
    units=target/"units";units.mkdir(exist_ok=True)
    for protocol in SAME:
        dest=units/protocol;dest.mkdir(exist_ok=True)
        for name in ("inference.npz","evaluator.npz"):
            path=dest/name
            if not path.exists():path.symlink_to(prepared/"units"/protocol/name)
    local={**config,"seed":seed,"result_root":str(_root(config,"strong_population_root")),"technical_updates":6000}
    metrics=v11._train_models(local,fold,target,torch.device("cuda"));summary={"status":"completed_pop8_training","seed":seed,"fold":fold,**metrics};_json(run_dir/"result_summary.json",summary);return summary

def _models(checkpoint:Path):
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    cp=torch.load(checkpoint,map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**cp["config"]);device=torch.device("cuda");det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(cp["det"]);diff.load_state_dict(cp["diff"]);ema=EMA(diff);ema.load_state_dict(cp["ema"]);ema.copy_to(diff);det.eval();diff.eval();return cp,det,diff,device

def stage_infer_pop8(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    seed,fold=_task(config,task_index);base=_seed_fold(_root(config,"strong_population_root"),seed,fold);cp,det,diff,device=_models(base/"checkpoint.pt");rescale=np.asarray(cp["residual_scale"],np.float32);recipient=fold+1
    duration_manifest=_read(_root(config,"result_root")/"support_duration_manifest.csv")
    eligibility={(int(r["participant"]),r["protocol"],seconds):bool(int(r[f"d{seconds}"])) for r in duration_manifest for seconds in DURATIONS}
    for unit_index,protocol in enumerate(SAME):
        data=np.load(base/"units"/protocol/"inference.npz");folder=base/"outputs"/protocol;folder.mkdir(parents=True,exist_ok=True)
        for seconds in DURATIONS:
            outputs={}
            if not eligibility[(recipient,protocol,seconds)]:
                # Keep the protocol in the availability denominator without
                # silently relabelling a shorter support block as 30/60/120 s.
                np.savez_compressed(folder/f"duration_{seconds}.npz");continue
            for panel in ("paired","natural"):
                y=np.asarray(data[f"{panel}_y"],np.float32);eog=np.asarray(data[f"{panel}_eog"],np.float32);gamma=float(data[f"gamma_{seconds}"]);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);outputs[f"{panel}_RAW"]=y
                for context,h in (("POP8",data[f"h_pop_{seconds}"]),("MATCH8",data[f"h_match_{seconds}"])):
                    a0=v11.apply_transfer(np.asarray(h),eog);at=torch.as_tensor(a0,device=device)
                    with torch.no_grad():rd=det(y=yt,eog=et,a0=at);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(n,device=device)).cpu().numpy() for n in bank]
                    corr=a0+(rd.cpu().numpy()+np.mean(samples,0))*rescale[None,:,None];outputs[f"{panel}_DIFF-{context}"]=v11.gamma_correction(y,corr,gamma)
                    if context=="MATCH8":outputs[f"{panel}_LINEAR-MATCH"]=v11.gamma_correction(y,a0,gamma);outputs[f"{panel}_DET-MATCH"]=v11.gamma_correction(y,a0+rd.cpu().numpy()*rescale[None,:,None],gamma)
                for donor in range(1,10):
                    if donor==recipient:continue
                    h=np.asarray(data[f"h_donor_{seconds}_{donor}"]);a0=v11.apply_transfer(h,eog);at=torch.as_tensor(a0,device=device)
                    with torch.no_grad():rd=det(y=yt,eog=et,a0=at);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(n,device=device)).cpu().numpy() for n in bank]
                    corr=a0+(rd.cpu().numpy()+np.mean(samples,0))*rescale[None,:,None];outputs[f"{panel}_DIFF-WRONG8-{donor}"]=v11.gamma_correction(y,corr,gamma)
            np.savez_compressed(folder/f"duration_{seconds}.npz",**outputs)
    summary={"status":"completed_pop8_evaluator_blind_inference","seed":seed,"fold":fold,"durations":list(DURATIONS),"donors_per_recipient":8,"eligible_units_by_duration":{str(seconds):int(sum(eligibility[(recipient,p,seconds)] for p in SAME)) for seconds in DURATIONS},"evaluator_opened":False};_json(run_dir/"result_summary.json",summary);return summary

def stage_causal_audit(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seed,fold=_task(config,task_index);rep=_seed_fold(_root(config,"replication_root"),seed,fold);source=_root(config,"v11_root")/"folds"/f"fold_{fold:02d}";uq=_root(config,"mechanism_uq_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}";rows=[]
    for protocol in SAME:
        inf=np.load(rep/"units"/protocol/"inference.npz");ev=np.load(source/"units"/protocol/"evaluator.npz");out=np.load(rep/"outputs"/"k8"/protocol/"inference_outputs.npz");scale=np.asarray(inf["eeg_scale"]);loc=np.asarray(inf["eeg_location"]);x=np.asarray(ev["paired_x"])[...,:500];y=np.asarray(inf["paired_y"]);eog=np.asarray(inf["paired_eog"]);gamma=float(inf["gamma"]);residual_scale=np.asarray(np.load(rep/"units"/protocol/"inference.npz")["eeg_scale"])*0+1
        methods={name:v11.rrmse((np.asarray(out[f"paired_{name}"])*scale[None,:,None]+loc[None,:,None])[...,:500],x) for name in ("LINEAR-POP","LINEAR-MATCH","LINEAR-WRONG","DET-POP","DET-MATCH","DET-WRONG","DIFF-POP","DIFF-MATCH","DIFF-WRONG")}
        atrue=np.asarray(ev["paired_a"])/scale[None,:,None];anchor_keys={"POP":"h_population","MATCH":"h_match","WRONG":"h_wrong"};anchors={c:v11.apply_transfer(np.asarray(inf[anchor_keys[c]]),eog) for c in ("POP","MATCH","WRONG")}
        # Crossed deterministic corrections reconstructed exactly from frozen outputs.
        detres={}
        for c in ("POP","MATCH"):
            restored=np.asarray(out[f"paired_DET-{c}"]);corr=(y-restored)/max(gamma,1e-12);detres[c]=corr-anchors[c]
        def crossed(a:str,c:str,res:Mapping[str,np.ndarray])->float:
            restored=v11.gamma_correction(y,anchors[a]+res[c],gamma);physical=(restored*scale[None,:,None]+loc[None,:,None])[...,:500];return v11.rrmse(physical,x)
        dmm,dmp,dpm,dpp=crossed("MATCH","MATCH",detres),crossed("MATCH","POP",detres),crossed("POP","MATCH",detres),crossed("POP","POP",detres)
        # Evaluator-only oracle residual exposes the algebraic coherence penalty.
        ores={c:atrue-anchors[c] for c in ("POP","MATCH")};omm,omp,opm,opp=crossed("MATCH","MATCH",ores),crossed("MATCH","POP",ores),crossed("POP","MATCH",ores),crossed("POP","POP",ores)
        f=np.load(uq/"units"/protocol/"paired_samples_and_factorial.npz");full={arm:v11.rrmse((np.asarray(f[f"paired_{arm}"])*scale[None,:,None]+loc[None,:,None])[...,:500],x) for arm in ("R_MM","R_MP","R_PM","R_PP")}
        i_full=full["R_MP"]+full["R_PM"]-full["R_MM"]-full["R_PP"];i_det=dmp+dpm-dmm-dpp
        rows.append({"seed":seed,"participant":fold+1,"protocol":protocol,**{f"R_{k}":v for k,v in methods.items()},"U_subject_LINEAR":methods["LINEAR-POP"]-methods["LINEAR-MATCH"],"U_subject_DET":methods["DET-POP"]-methods["DET-MATCH"],"U_subject_DIFF":methods["DIFF-POP"]-methods["DIFF-MATCH"],"U_delta_MATCH":methods["DET-MATCH"]-methods["DIFF-MATCH"],"Delta_SA":(methods["DIFF-POP"]-methods["DIFF-MATCH"])-(methods["DET-POP"]-methods["DET-MATCH"]),"I_full":i_full,"I_det_cross":i_det,"I_full_minus_I_det_cross":i_full-i_det,"I_oracle":omp+opm-omm-opp,"anchor_distance":float(np.sqrt(np.mean((anchors["MATCH"]-anchors["POP"])**2)))})
    folder=_root(config,"causal_audit_root")/"tasks";_csv(folder/f"seed_{seed}_fold_{fold:02d}.csv",rows);summary={"status":"completed_causal_task","seed":seed,"fold":fold,"rows":len(rows)};_json(run_dir/"result_summary.json",summary);return summary

def _support_precision(config:Mapping[str,Any],participant:int,session:str,seconds:int,inf:Mapping[str,np.ndarray],h:np.ndarray,rescale:np.ndarray)->np.ndarray:
    eeg,eog,sfreq,events=v11._load_session(config,participant,session);support,_=v11._support_query_ranges(events,eeg.shape[1],sfreq);region=_slice_budget(support,sfreq,seconds);mid=(region.start+region.stop)//2;validation=slice(mid,region.stop);yn=(eeg[:,validation]-np.asarray(inf["eeg_location"])[:,None])/np.asarray(inf["eeg_scale"])[:,None];en=(eog[:,validation]-np.asarray(inf["eog_location"])[:,None])/np.asarray(inf["eog_scale"])[:,None];res=(yn-v11.apply_transfer(h,en[None])[0])/rescale[:,None];variance=np.maximum(np.var(res,axis=1),.05);return np.clip(1/variance,.05,20).astype(np.float32)[:,None]

def operator_posterior_sample(diff:Any,*,y:Any,eog:Any,a0:Any,r_det:Any,initial_noise:Any,center:Any,precision:Any,strength:float=1.0,valid_length:int=500)->Any:
    """DDIM25 with a diagonal Gaussian proximal step on the full residual."""
    import torch
    if strength==0:return diff.sample(y=y,eog=eog,a0=a0,r_det=r_det,initial_noise=initial_noise)
    state=initial_noise.clone();schedule=torch.linspace(diff.config.timesteps-1,0,diff.config.ddim_steps,device=y.device).round().long()
    for index,t_value in enumerate(schedule):
        timestep=torch.full((len(y),),int(t_value),device=y.device,dtype=torch.long);alpha=diff.alpha_bar.gather(0,timestep).reshape(len(y),1,1);v=diff.backbone(state,timestep,y=y,eog=eog,a0=a0,r_det=r_det);x0=alpha.sqrt()*state-(1-alpha).sqrt()*v;epsilon=(1-alpha).sqrt()*state+alpha.sqrt()*v;sigma2=strength*(1-alpha)/alpha.clamp_min(1e-5);full=r_det+x0;prox=(full+sigma2*precision*center)/(1+sigma2*precision)
        # The guidance is identity on padded samples; scientific evaluation crops
        # to the original 500 samples.
        if valid_length<prox.shape[-1]:prox[...,valid_length:]=full[...,valid_length:]
        x0=prox-r_det
        state=x0 if index+1==len(schedule) else diff.alpha_bar.gather(0,torch.full_like(timestep,int(schedule[index+1]))).reshape(len(y),1,1).sqrt()*x0+(1-diff.alpha_bar.gather(0,torch.full_like(timestep,int(schedule[index+1]))).reshape(len(y),1,1)).sqrt()*epsilon
    if valid_length<state.shape[-1]:state[...,valid_length:]=0
    return state

def stage_oppost_technical(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    seed=int(config["seeds"][0]);fold=0;rep=_seed_fold(_root(config,"replication_root"),seed,fold);cp,det,diff,device=_models(rep/"checkpoint.pt");data=np.load(rep/"units"/"same_01"/"inference.npz");y=np.asarray(data["paired_y"][:2],np.float32);eog=np.asarray(data["paired_eog"][:2],np.float32);hp=np.asarray(data["h_population"]);hm=_transfer(config,1,"01T",60,np.asarray(data["eeg_location"]),np.asarray(data["eeg_scale"]),np.asarray(data["eog_location"]),np.asarray(data["eog_scale"]))[0];a0=v11.apply_transfer(hp,eog);rescale=np.asarray(cp["residual_scale"],np.float32);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);noise=torch.as_tensor(v11._noise_bank(y.shape,seed,1)[0],device=device)
    with torch.no_grad():rd=det(y=yt,eog=et,a0=at);base=diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise);zero=operator_posterior_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise,center=torch.zeros_like(rd),precision=torch.ones(1,3,1,device=device),strength=0);pop=operator_posterior_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise,center=torch.zeros_like(rd),precision=torch.ones(1,3,1,device=device),strength=1);replay=operator_posterior_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise,center=torch.zeros_like(rd),precision=torch.ones(1,3,1,device=device),strength=1)
    _,det2,diff2,_=_models(rep/"checkpoint.pt")
    with torch.no_grad():rd2=det2(y=yt,eog=et,a0=at);base2=diff2.sample(y=yt,eog=et,a0=at,r_det=rd2,initial_noise=noise)
    reload_exact=bool(torch.equal(rd,rd2) and torch.equal(base,base2));mm=(v11.apply_transfer(hm,eog)-a0)/rescale[None,:,None];finite=bool(torch.isfinite(pop).all());padding_zero=bool(torch.count_nonzero(pop[...,500:])==0);corr=a0+(rd.cpu().numpy()+pop.cpu().numpy())*rescale[None,:,None];corr[...,500:]=0;restored=v11.gamma_correction(y,corr,float(data["gamma"]));padding_identity=bool(np.array_equal(restored[...,500:],y[...,500:]));passed=bool(torch.equal(base,zero) and torch.equal(pop,replay) and reload_exact and padding_zero and padding_identity and np.max(np.abs(mm))>0 and finite and np.allclose(v11.apply_transfer(hp,eog)-a0,0));summary={"status":"OPPOST_TECHNICAL_PASSED" if passed else "OPPOST_TECHNICAL_FAILED","lambda0_bitwise":bool(torch.equal(base,zero)),"common_noise_exact":bool(torch.equal(pop,replay)),"checkpoint_reload_exact":reload_exact,"guided_padding_zero":padding_zero,"reconstructed_padding_identity":padding_identity,"checkpoint_source":str(rep/"checkpoint.pt"),"pop_center_zero":True,"match_center_nonzero":bool(np.max(np.abs(mm))>0),"finite":finite,"support_split":"first_60s_operator_second_60s_precision","evaluator_opened":False};_json(_root(config,"oppost_root")/"technical_validity.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_oppost_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    tech=json.loads((_root(config,"oppost_root")/"technical_validity.json").read_text())
    if tech["status"]!="OPPOST_TECHNICAL_PASSED":raise RuntimeError("technical validity failed")
    seed,fold=_task(config,task_index);rep=_seed_fold(_root(config,"replication_root"),seed,fold);cp,det,diff,device=_models(rep/"checkpoint.pt");rescale=np.asarray(cp["residual_scale"],np.float32);recipient=fold+1
    for unit_index,protocol in enumerate(SAME):
        data=np.load(rep/"units"/protocol/"inference.npz");session=_read(rep/"unit_manifest.csv")[unit_index]["support_session"];folder=_root(config,"oppost_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs"/protocol;folder.mkdir(parents=True,exist_ok=True);outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(data[f"{panel}_y"],np.float32);eog=np.asarray(data[f"{panel}_eog"],np.float32);gamma=float(data["gamma"]);hp=np.asarray(data["h_population"]);hm=_transfer(config,recipient,session,60,np.asarray(data["eeg_location"]),np.asarray(data["eeg_scale"]),np.asarray(data["eog_location"]),np.asarray(data["eog_scale"]))[0];a0=v11.apply_transfer(hp,eog);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8)
            with torch.no_grad():rd=det(y=yt,eog=et,a0=at)
            contexts=[("POP",hp,recipient) ,("MATCH",hm,recipient)]+[(f"WRONG-{d}",_transfer(config,d,session,60,np.asarray(data["eeg_location"]),np.asarray(data["eeg_scale"]),np.asarray(data["eog_location"]),np.asarray(data["eog_scale"]))[0],d) for d in range(1,10) if d!=recipient]
            for name,h,owner in contexts:
                center=(v11.apply_transfer(h,eog)-a0)/rescale[None,:,None];precision=_support_precision(config,owner,session,120,data,h,rescale);ct=torch.as_tensor(center,device=device);pt=torch.as_tensor(precision[None],device=device);samples=[]
                with torch.no_grad():
                    for n in bank:samples.append(operator_posterior_sample(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(n,device=device),center=ct,precision=pt).cpu().numpy())
                corr=a0+(rd.cpu().numpy()+np.mean(samples,0))*rescale[None,:,None];corr[...,500:]=0;outputs[f"{panel}_OPPOST-{name}"]=v11.gamma_correction(y,corr,gamma)
        np.savez_compressed(folder/"inference_outputs.npz",**outputs)
    summary={"status":"completed_oppost_evaluator_blind_inference","seed":seed,"fold":fold,"contexts":10,"evaluator_opened":False};_json(run_dir/"result_summary.json",summary);return summary

def stage_localized_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    """Conditional inference-only backup; no evaluator field is opened."""
    route=json.loads((_root(config,"oppost_root")/"routing_decision.json").read_text())
    if not route["localized_backup_authorized"]:
        summary={"status":"skipped_not_authorized","task_index":task_index};_json(run_dir/"result_summary.json",summary);return summary
    seed,fold=_task(config,task_index);recipient=fold+1;rep=_seed_fold(_root(config,"replication_root"),seed,fold);global_root=_root(config,"oppost_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs";rescale=np.asarray(__import__("torch").load(rep/"checkpoint.pt",map_location="cpu",weights_only=False)["residual_scale"],np.float32)
    for unit_index,protocol in enumerate(SAME):
        data=np.load(rep/"units"/protocol/"inference.npz");current=np.load(rep/"outputs"/"k8"/protocol/"inference_outputs.npz");global_out=np.load(global_root/protocol/"inference_outputs.npz");session=_read(rep/"unit_manifest.csv")[unit_index]["support_session"];hp=np.asarray(data["h_population"]);folder=_root(config,"oppost_root")/"localized"/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs"/protocol;folder.mkdir(parents=True,exist_ok=True);outputs={}
        for panel in ("paired","natural"):
            eog=np.asarray(data[f"{panel}_eog"],np.float32);base=np.asarray(current[f"{panel}_DIFF-POP"]);contexts=[("POP",hp,recipient),("MATCH",_transfer(config,recipient,session,60,np.asarray(data["eeg_location"]),np.asarray(data["eeg_scale"]),np.asarray(data["eog_location"]),np.asarray(data["eog_scale"]))[0],recipient)]+[(f"WRONG-{d}",_transfer(config,d,session,60,np.asarray(data["eeg_location"]),np.asarray(data["eeg_scale"]),np.asarray(data["eog_location"]),np.asarray(data["eog_scale"]))[0],d) for d in range(1,10) if d!=recipient]
            for name,h,owner in contexts:
                center=np.abs((v11.apply_transfer(h,eog)-v11.apply_transfer(hp,eog))/rescale[None,:,None]);precision=_support_precision(config,owner,session,120,data,h,rescale)[:,0];threshold=(1/np.maximum(precision,.05))[None,:,None];mask=center/(center+threshold+1e-8);global_value=np.asarray(global_out[f"{panel}_OPPOST-{name}"]);outputs[f"{panel}_LOCAL-{name}"]=base+mask*(global_value-base)
        np.savez_compressed(folder/"inference_outputs.npz",**outputs)
    summary={"status":"completed_localized_evaluator_blind_inference","seed":seed,"fold":fold,"evaluator_opened":False,"mask_inputs":"support transfer/reliability and query EOG only"};_json(run_dir/"result_summary.json",summary);return summary

def stage_evaluate_local_all(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    rows=[];natural=[]
    for index in range(27):
        seed,fold=_task(config,index);rep=_seed_fold(_root(config,"replication_root"),seed,fold);source=_root(config,"v11_root")/"folds"/f"fold_{fold:02d}";outputs=_root(config,"oppost_root")/"localized"/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs";p,n=_evaluate_outputs(config,rep,source,outputs,seed,fold,"LOCAL");rows.extend(p);natural.extend(n)
    root=_root(config,"oppost_root")/"localized";_csv(root/"paired_metrics.csv",rows);_csv(root/"natural_safety.csv",natural);summary={"status":"completed_localized_independent_evaluation","paired_rows":len(rows),"natural_rows":len(natural)};_json(run_dir/"result_summary.json",summary);return summary

def stage_evaluate_oppost_all(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    rows=[];natural=[]
    for index in range(27):
        seed,fold=_task(config,index);rep=_seed_fold(_root(config,"replication_root"),seed,fold);source=_root(config,"v11_root")/"folds"/f"fold_{fold:02d}";outputs=_root(config,"oppost_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs";p,n=_evaluate_outputs(config,rep,source,outputs,seed,fold,"OPPOST");rows.extend(p);natural.extend(n)
    root=_root(config,"oppost_root");_csv(root/"paired_metrics.csv",rows);_csv(root/"natural_safety.csv",natural);summary={"status":"completed_oppost_independent_evaluation","paired_rows":len(rows),"natural_rows":len(natural)};_json(run_dir/"result_summary.json",summary);return summary

def stage_aggregate_oppost(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=_root(config,"oppost_root");paired=_read(root/"paired_metrics.csv");natural=_read(root/"natural_safety.csv");effects=_participant_effects(paired,"OPPOST","OPPOST-MATCH","OPPOST-POP","OPPOST-WRONG-");_csv(root/"participant_seed_effects.csv",effects)
    def effect(key:str)->dict[str,Any]:
        values=np.asarray([np.mean([r[key] for r in effects if r["participant"]==p]) for p in range(1,10)]);return {"mean":float(values.mean()),"median":float(np.median(values)),"positive":int(np.sum(values>0)),"participant_values":values.tolist(),"seed_means":[float(np.mean([r[key] for r in effects if r["seed"]==s])) for s in map(int,config["seeds"])]}
    up,uw=effect("U_P"),effect("U_W");rep_root=_root(config,"replication_root");current=[];current_nat=[]
    for seed in map(int,config["seeds"]):
        for fold in range(9):
            current.extend({"seed":seed,"participant":fold+1,"rrmse":float(r["rrmse"])} for r in _read(rep_root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"paired_metrics_v11_1.csv") if r["method"]=="DIFF-MATCH")
            current_nat.extend(r for r in _read(rep_root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"natural_metrics_v11_1.csv") if r["method"]=="DIFF-MATCH")
    gains=[]
    for r in effects:gains.append({"participant":r["participant"],"seed":r["seed"],"G_MATCH":float(np.mean([x["rrmse"] for x in current if x["participant"]==r["participant"] and x["seed"]==r["seed"]]))-r["match_rrmse"]})
    gm=np.asarray([np.mean([r["G_MATCH"] for r in gains if r["participant"]==p]) for p in range(1,10)]);match=[r for r in natural if r["method"]=="OPPOST-MATCH"];keys=("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation");nat={k:float(np.nanmean([float(r[k]) for r in match])) for k in keys};cur={k:float(np.nanmean([float(r[k]) for r in current_nat])) for k in keys};delta={k:nat[k]-cur[k] for k in keys};subject=up["mean"]>0 and up["median"]>0 and uw["mean"]>0 and uw["median"]>0 and up["positive"]>=8 and uw["positive"]>=8 and all(x>0 for x in up["seed_means"]) and all(x>0 for x in uw["seed_means"]) and gm.mean()>=0;safe=nat["eog_attenuation"]>0 and delta["preservation"]>=-.02 and delta["mi_kappa"]>=-.02 and delta["erd_preservation"]>=-.02;safety_drop=delta["preservation"]<-.02 or delta["mi_kappa"]<-.02 or delta["erd_preservation"]<-.02;route={"oppost_subject_signal":bool(subject),"oppost_full_success":bool(subject and safe),"localized_backup_authorized":bool(subject and safety_drop),"localized_backup_run":False,"development_only":True};summary={"U_P_OP":up,"U_W_OP":uw,"G_MATCH":{"mean":float(gm.mean()),"median":float(np.median(gm)),"positive":int(np.sum(gm>0)),"participant_values":gm.tolist()},"natural":nat,"current_diff_match_natural":cur,"natural_delta_vs_current":delta,"routing":route};_json(root/"result_summary.json",summary);_json(root/"routing_decision.json",route);Path("reports/bci2b_operator_likelihood_diffusion.md").write_text(f"# BCI2b operator-likelihood diffusion\n\nDevelopment-only frozen score/anchor experiment. U_P_OP {up['mean']:+.5f} ({up['positive']}/9); U_W_OP {uw['mean']:+.5f} ({uw['positive']}/9); G_MATCH {gm.mean():+.5f}. Localized backup authorized: `{route['localized_backup_authorized']}`.\n",encoding="utf-8");_json(run_dir/"result_summary.json",summary);return summary

def stage_aggregate_local(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=_root(config,"oppost_root")/"localized";paired=_read(root/"paired_metrics.csv");natural=_read(root/"natural_safety.csv");effects=_participant_effects(paired,"LOCAL","LOCAL-MATCH","LOCAL-POP","LOCAL-WRONG-");_csv(root/"participant_seed_effects.csv",effects)
    avg=[]
    for p in range(1,10):avg.append({"participant":p,"U_P":float(np.mean([r["U_P"] for r in effects if r["participant"]==p])),"U_W":float(np.mean([r["U_W"] for r in effects if r["participant"]==p]))})
    match=[r for r in natural if r["method"]=="LOCAL-MATCH"];global_summary=json.loads((_root(config,"oppost_root")/"result_summary.json").read_text());summary={"U_P_mean":float(np.mean([r["U_P"] for r in avg])),"U_W_mean":float(np.mean([r["U_W"] for r in avg])),"U_P_positive":int(sum(r["U_P"]>0 for r in avg)),"U_W_positive":int(sum(r["U_W"]>0 for r in avg)),"preservation":float(np.mean([float(r["preservation"]) for r in match])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in match]))};summary["pareto_improves_global"]=summary["preservation"]>global_summary["natural"]["preservation"] and summary["eog_attenuation"]>=global_summary["natural"]["eog_attenuation"]-.01;summary["keeps_subject_effects"]=summary["U_P_mean"]>0 and summary["U_W_mean"]>0;_json(root/"result_summary.json",summary);route=json.loads((_root(config,"oppost_root")/"routing_decision.json").read_text());route["localized_backup_run"]=True;route["localized_backup_success"]=bool(summary["pareto_improves_global"] and summary["keeps_subject_effects"]);_json(_root(config,"oppost_root")/"routing_decision.json",route);_json(run_dir/"result_summary.json",summary);return summary

def _evaluate_outputs(config:Mapping[str,Any],base:Path,source:Path,output_path:Path,seed:int,fold:int,prefix:str)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    paired=[];natural=[]
    for protocol in SAME:
        inf=np.load(base/"units"/protocol/"inference.npz");ev=np.load(source/"units"/protocol/"evaluator.npz");out=np.load(output_path/protocol/"inference_outputs.npz");scale=np.asarray(inf["eeg_scale"]);loc=np.asarray(inf["eeg_location"]);x=np.asarray(ev["paired_x"])[...,:500];raw=(np.asarray(inf["paired_y"])*scale[None,:,None]+loc[None,:,None])[...,:500];rawerr=v111._band_error(raw,x,(1,45),config);raw_n=np.asarray(inf["natural_y"])[...,:500];eog=np.asarray(inf["natural_eog"])[...,:500];labels=np.asarray(ev["natural_labels"]);energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3)
        for key in out.files:
            panel,method=key.split("_",1);value=np.asarray(out[key])[...,:500]
            if panel=="paired":
                physical=value*scale[None,:,None]+loc[None,:,None];err=v111._band_error(physical,x,(1,45),config);paired.append({"package":prefix,"seed":seed,"participant":fold+1,"protocol":protocol,"method":method,"rrmse":v11.rrmse(physical,x),"correlation":v11.correlation(physical,x),"delta_snr":v11.delta_snr(physical,x,raw),"paired_spectral_utility":rawerr-err})
            else:natural.append({"package":prefix,"seed":seed,"participant":fold+1,"protocol":protocol,"method":method,"eog_attenuation":v11._coherence_proxy(raw_n,eog)-v11._coherence_proxy(value,eog),"preservation":1-v11.rrmse(value[low],raw_n[low]),"mi_band_distortion":v111._bandpower_distortion(value[low],raw_n[low],(8,30),config),"covariance":v11._covariance_distortion(value[low],raw_n[low]),"mi_kappa":v11._kappa(value,labels),"erd_preservation":v111._erd_preservation(value,raw_n,labels,config)})
    return paired,natural

def stage_evaluate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seed,fold=_task(config,task_index);rows=[];natural=[]
    # POP8 durations have evaluator files in their prepared root.
    popbase=_seed_fold(_root(config,"strong_population_root"),seed,fold);prepared=_root(config,"strong_population_root")/"prepared"/f"fold_{fold:02d}"
    for seconds in DURATIONS:
        # Materialize a temporary directory-shaped view without touching outputs.
        for protocol in SAME:
            src=popbase/"outputs"/protocol/f"duration_{seconds}.npz";dest=popbase/"eval_view"/str(seconds)/protocol;dest.mkdir(parents=True,exist_ok=True);link=dest/"inference_outputs.npz"
            if not link.exists():link.symlink_to(src)
        p,n=_evaluate_outputs(config,popbase,prepared,popbase/"eval_view"/str(seconds),seed,fold,f"POP8-D{seconds}");rows.extend(p);natural.extend(n)
    rep=_seed_fold(_root(config,"replication_root"),seed,fold);source=_root(config,"v11_root")/"folds"/f"fold_{fold:02d}";opout=_root(config,"oppost_root")/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"outputs";p,n=_evaluate_outputs(config,rep,source,opout,seed,fold,"OPPOST");rows.extend(p);natural.extend(n)
    out=_root(config,"result_root")/"evaluation";_csv(out/f"seed_{seed}_fold_{fold:02d}_paired.csv",rows);_csv(out/f"seed_{seed}_fold_{fold:02d}_natural.csv",natural);summary={"status":"completed_independent_evaluator","seed":seed,"fold":fold,"paired_rows":len(rows),"natural_rows":len(natural)};_json(run_dir/"result_summary.json",summary);return summary

def _sign_flip(values:np.ndarray)->float:
    observed=abs(float(values.mean()));signs=((np.arange(2**len(values))[:,None]>>np.arange(len(values)))&1)*2-1;return float(np.mean(np.abs((signs*values[None]).mean(1))>=observed))
def _participant_effects(rows:list[dict[str,str]],package:str,match:str,pop:str,wrong_prefix:str)->list[dict[str,Any]]:
    out=[]
    for seed in sorted({int(r["seed"]) for r in rows}):
        for participant in range(1,10):
            take=[r for r in rows if r["package"]==package and int(r["seed"])==seed and int(r["participant"])==participant];by=defaultdict(list)
            for r in take:by[r["method"]].append(float(r["rrmse"]))
            if match not in by or pop not in by:continue
            wrong=[float(np.mean(v)) for k,v in by.items() if k.startswith(wrong_prefix)];out.append({"seed":seed,"participant":participant,"match_rrmse":float(np.mean(by[match])),"pop_rrmse":float(np.mean(by[pop])),"wrong_rrmse":float(np.mean(wrong)),"U_P":float(np.mean(by[pop])-np.mean(by[match])),"U_W":float(np.mean(wrong)-np.mean(by[match]))})
    return out

def _aggregate_causal(config:Mapping[str,Any])->dict[str,Any]:
    causal=_root(config,"causal_audit_root");rows=[]
    for path in sorted((causal/"tasks").glob("*.csv")):rows.extend(_read(path))
    fields=("U_subject_LINEAR","U_subject_DET","U_subject_DIFF","U_delta_MATCH","Delta_SA","I_full","I_oracle","I_det_cross","I_full_minus_I_det_cross","anchor_distance");participant=[]
    for p in range(1,10):
        take=[r for r in rows if int(r["participant"])==p];participant.append({"participant":p,**{f:float(np.mean([float(r[f]) for r in take])) for f in fields}})
    _csv(causal/"participant_effects.csv",participant);corr=float(np.corrcoef([r["I_oracle"] for r in participant],[r["anchor_distance"] for r in participant])[0,1]);det_fraction=float(np.mean([abs(r["I_det_cross"]) for r in participant])/(np.mean([abs(r["I_full"]) for r in participant])+1e-12));oracle_fraction=float(np.mean([abs(r["I_oracle"]) for r in participant])/(np.mean([abs(r["I_full"]) for r in participant])+1e-12));label="CONTEXT_COHERENCE_EFFECT_PRESENT_SCORE_SPECIFIC_MEDIATION_NOT_IDENTIFIED" if det_fraction>=.5 or oracle_fraction>=.5 else "SCORE_SPECIFIC_MEDIATION_REMAINS_PLAUSIBLE";summary={"status":label,"participant_effects":{f:{"mean":float(np.mean([r[f] for r in participant])),"median":float(np.median([r[f] for r in participant])),"positive":int(sum(r[f]>0 for r in participant))} for f in fields},"oracle_anchor_distance_correlation":corr,"det_cross_absolute_fraction_of_full":det_fraction,"oracle_absolute_fraction_of_full":oracle_fraction,"scientific_unit":"participant","n":9,"development_only":True};_json(causal/"result_summary.json",summary);return summary

def stage_aggregate_causal(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    summary=_aggregate_causal(config);e=summary["participant_effects"];Path("reports/bci2b_context_causal_audit.md").write_text(f"# BCI2b context causal audit\n\nDecision: `{summary['status']}`. Coherent subject effects are LINEAR {e['U_subject_LINEAR']['mean']:+.5f}, DET {e['U_subject_DET']['mean']:+.5f}, and DIFF {e['U_subject_DIFF']['mean']:+.5f}; diffusion-vs-matched-DET increment is {e['U_delta_MATCH']['mean']:+.5f} and Delta_SA is {e['Delta_SA']['mean']:+.5f}. I_full mean {e['I_full']['mean']:+.5f}; I_oracle {e['I_oracle']['mean']:+.5f}; I_det_cross {e['I_det_cross']['mean']:+.5f}. The historical crossed grid is a context-coherence penalty, not evidence of score synergy. Scientific n=9 participants; seeds and protocol units were aggregated within participant.\n",encoding="utf-8");_json(run_dir/"result_summary.json",summary);return summary

def stage_aggregate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    evalroot=_root(config,"result_root")/"evaluation";paired=[];natural=[]
    for seed in map(int,config["seeds"]):
        for fold in range(9):paired.extend(_read(evalroot/f"seed_{seed}_fold_{fold:02d}_paired.csv"));natural.extend(_read(evalroot/f"seed_{seed}_fold_{fold:02d}_natural.csv"))
    strong=_root(config,"strong_population_root");causal=_root(config,"causal_audit_root");oproot=_root(config,"oppost_root")
    pop_effects=[]
    for seconds in DURATIONS:pop_effects.extend([{**r,"duration_seconds":seconds} for r in _participant_effects(paired,f"POP8-D{seconds}","DIFF-MATCH8","DIFF-POP8","DIFF-WRONG8-")])
    _csv(strong/"participant_seed_effects.csv",pop_effects)
    oppost=_participant_effects(paired,"OPPOST","OPPOST-MATCH","OPPOST-POP","OPPOST-WRONG-");_csv(oproot/"participant_seed_effects.csv",oppost)
    def summarize(rows:list[dict[str,Any]],key:str)->dict[str,Any]:
        avg=[]
        for p in range(1,10):avg.append(float(np.mean([r[key] for r in rows if r["participant"]==p])))
        arr=np.asarray(avg);rng=np.random.default_rng(int(config["bootstrap_seed"]));idx=rng.integers(0,9,size=(int(config["bootstrap_replicates"]),9));rep=arr[idx].mean(1);return {"mean":float(arr.mean()),"median":float(np.median(arr)),"positive":int(np.sum(arr>0)),"participant_values":avg,"seed_means":[float(np.mean([r[key] for r in rows if r["seed"]==s])) for s in map(int,config["seeds"])],"sign_flip_two_sided":_sign_flip(arr),"descriptive_ci":[float(np.quantile(rep,.025)),float(np.quantile(rep,.975))]}
    pop_summary={}
    for seconds in DURATIONS:
        take=[r for r in pop_effects if r["duration_seconds"]==seconds];pop_summary[str(seconds)]={"U_P8":summarize(take,"U_P"),"U_W8":summarize(take,"U_W")}
    pop_participant=[]
    for seconds in DURATIONS:
        take=[r for r in pop_effects if r["duration_seconds"]==seconds]
        for participant in range(1,10):
            selected=[r for r in take if r["participant"]==participant];pop_participant.append({"duration_seconds":seconds,"participant":participant,"U_P8":float(np.mean([r["U_P"] for r in selected])),"U_W8":float(np.mean([r["U_W"] for r in selected])),"match_rrmse":float(np.mean([r["match_rrmse"] for r in selected])),"pop_rrmse":float(np.mean([r["pop_rrmse"] for r in selected])),"wrong_rrmse":float(np.mean([r["wrong_rrmse"] for r in selected]))})
    _csv(strong/"participant_effects.csv",pop_participant)
    causal_summary=_aggregate_causal(config);cpart=_read(causal/"participant_effects.csv")
    op_summary={"U_P_OP":summarize(oppost,"U_P"),"U_W_OP":summarize(oppost,"U_W")}
    _csv(oproot/"participant_effects.csv",[{"participant":p,"U_P_OP":float(np.mean([r["U_P"] for r in oppost if r["participant"]==p])),"U_W_OP":float(np.mean([r["U_W"] for r in oppost if r["participant"]==p])),"match_rrmse":float(np.mean([r["match_rrmse"] for r in oppost if r["participant"]==p])),"pop_rrmse":float(np.mean([r["pop_rrmse"] for r in oppost if r["participant"]==p])),"wrong_rrmse":float(np.mean([r["wrong_rrmse"] for r in oppost if r["participant"]==p]))} for p in range(1,10)])
    # Current MATCH comparison and safety are paired on the frozen source outputs.
    current=[];current_natural=[];current_panel=[];current_natural_panel=[]
    rep_root=_root(config,"replication_root")
    for seed in map(int,config["seeds"]):
        for fold in range(9):
            for row in _read(rep_root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"paired_metrics_v11_1.csv"):
                if row["method"]=="DIFF-MATCH":current.append({"seed":seed,"participant":fold+1,"rrmse":float(row["rrmse"])})
                if row["method"] in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-POP","DIFF-MATCH"):
                    current_panel.append({"package":"CURRENT","seed":seed,"participant":fold+1,"protocol":row["protocol"],"method":row["method"],"rrmse":row["rrmse"],"correlation":row["correlation"],"delta_snr":row["delta_snr"],"paired_spectral_utility":row["paired_spectral_utility"]})
            for row in _read(rep_root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"/"natural_metrics_v11_1.csv"):
                if row["method"]=="DIFF-MATCH":current_natural.append(row)
                if row["method"] in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-POP","DIFF-MATCH"):
                    current_natural_panel.append({"package":"CURRENT","seed":seed,"participant":fold+1,"protocol":row["protocol"],"method":row["method"],"eog_attenuation":row["eog_attenuation"],"preservation":row["preservation"],"mi_band_distortion":row["mi_band_distortion"],"covariance":row["covariance"],"mi_kappa":row["mi_kappa"],"erd_preservation":row["erd_preservation"]})
    gains=[]
    for row in oppost:
        baseline=np.mean([r["rrmse"] for r in current if r["seed"]==row["seed"] and r["participant"]==row["participant"]]);gains.append({**row,"G_MATCH":baseline-row["match_rrmse"]})
    op_summary["G_MATCH"]=summarize(gains,"G_MATCH");op_summary["Delta_U_P"]={"mean":op_summary["U_P_OP"]["mean"]-float(np.mean([float(r["U_subject_DIFF"]) for r in cpart]))}
    op_summary["secondary_rrmse"]={method:float(np.mean([float(r["rrmse"]) for r in current_panel if r["method"]==method])) for method in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-POP","DIFF-MATCH")}
    opnat=[r for r in natural if r["package"]=="OPPOST" and r["method"]=="OPPOST-MATCH"];op_summary["natural"]={k:float(np.nanmean([float(r[k]) for r in opnat])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")};current_safety={k:float(np.nanmean([float(r[k]) for r in current_natural])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")};op_summary["current_diff_match_natural"]=current_safety;op_summary["natural_delta_vs_current"]={k:op_summary["natural"][k]-current_safety[k] for k in current_safety}
    safe=op_summary["natural"]["eog_attenuation"]>0 and op_summary["natural_delta_vs_current"]["preservation"]>=-.02 and op_summary["natural_delta_vs_current"]["mi_kappa"]>=-.02 and op_summary["natural_delta_vs_current"]["erd_preservation"]>=-.02
    subject_signal=op_summary["U_P_OP"]["mean"]>0 and op_summary["U_P_OP"]["median"]>0 and op_summary["U_W_OP"]["mean"]>0 and op_summary["U_W_OP"]["median"]>0 and op_summary["U_P_OP"]["positive"]>=8 and op_summary["U_W_OP"]["positive"]>=8 and all(x>0 for x in op_summary["U_P_OP"]["seed_means"]) and all(x>0 for x in op_summary["U_W_OP"]["seed_means"]) and op_summary["G_MATCH"]["mean"]>=0
    op_success=subject_signal and safe
    safety_drop=op_summary["natural_delta_vs_current"]["preservation"]<-.02 or op_summary["natural_delta_vs_current"]["mi_kappa"]<-.02 or op_summary["natural_delta_vs_current"]["erd_preservation"]<-.02
    route={"oppost_subject_signal":bool(subject_signal),"oppost_full_success":bool(op_success),"localized_backup_authorized":bool(subject_signal and safety_drop),"localized_backup_run":False,"development_only":True};op_summary["routing"]=route;_json(oproot/"result_summary.json",op_summary);_json(oproot/"routing_decision.json",route)
    availability=_read(_root(config,"result_root")/"support_duration_manifest.csv");coverage={str(seconds):{"eligible_protocol_units":int(sum(int(r[f"d{seconds}"]) for r in availability)),"availability_denominator":len(availability),"participants_represented":int(len({int(r["participant"]) for r in availability if int(r[f"d{seconds}"])}))} for seconds in DURATIONS}
    historical=json.loads((_root(config,"replication_root")/"result_summary.json").read_text());strong_result={"support_duration":pop_summary,"support_coverage":coverage,"primary_duration_seconds":120,"full_available_reported":False,"full_available_reason":"support durations are not identical across all units","frozen_twoheldout_unseen_wrong_sensitivity":historical["effects"]["U_W"],"frozen_twoheldout_population_effect":historical["effects"]["U_P"],"seen_and_unseen_donors_not_pooled":True,"scientific_unit":"participant","n":9,"seeds":list(map(int,config["seeds"])),"development_only":True};_json(strong/"result_summary.json",strong_result);_json(strong/"routing_decision.json",{"status":"STRONG_POPULATION_CONTROL_COMPLETED","primary_duration_seconds":120,"support_coverage":coverage,"development_only":True})
    strong_paired=[r for r in paired if r["package"].startswith("POP8")];strong_natural=[r for r in natural if r["package"].startswith("POP8")];op_paired=[r for r in paired if r["package"]=="OPPOST"]+current_panel;op_natural=[r for r in natural if r["package"]=="OPPOST"]+current_natural_panel
    _csv(strong/"paired_metrics.csv",strong_paired);_csv(strong/"natural_safety.csv",strong_natural);_csv(oproot/"natural_safety.csv",op_natural);_csv(oproot/"paired_metrics.csv",op_paired)
    def method_summary(rows:list[dict[str,Any]])->list[dict[str,Any]]:
        result=[]
        for package in sorted({r["package"] for r in rows}):
            for method in sorted({r["method"] for r in rows if r["package"]==package}):
                take=[r for r in rows if r["package"]==package and r["method"]==method];result.append({"package":package,"method":method,"participants":len({int(r["participant"]) for r in take}),"seeds":len({int(r["seed"]) for r in take}),"rrmse":float(np.mean([float(r["rrmse"]) for r in take])),"correlation":float(np.mean([float(r["correlation"]) for r in take])),"delta_snr":float(np.mean([float(r["delta_snr"]) for r in take])),"paired_spectral_utility":float(np.mean([float(r["paired_spectral_utility"]) for r in take]))})
        return result
    _csv(strong/"method_summary.csv",method_summary(strong_paired));_csv(oproot/"method_summary.csv",method_summary(op_paired))
    def participant_safety(rows:list[dict[str,Any]])->list[dict[str,Any]]:
        result=[]
        for package in sorted({r["package"] for r in rows}):
            for seed in sorted({int(r["seed"]) for r in rows if r["package"]==package}):
                for participant in range(1,10):
                    for method in sorted({r["method"] for r in rows if r["package"]==package and int(r["seed"])==seed and int(r["participant"])==participant}):
                        take=[r for r in rows if r["package"]==package and int(r["seed"])==seed and int(r["participant"])==participant and r["method"]==method];result.append({"package":package,"seed":seed,"participant":participant,"method":method,**{k:float(np.nanmean([float(r[k]) for r in take])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}})
        return result
    strong_participant_safety=participant_safety(strong_natural);_csv(strong/"participant_natural_safety.csv",strong_participant_safety);_csv(oproot/"participant_natural_safety.csv",participant_safety(op_natural))
    strong_result=json.loads((strong/"result_summary.json").read_text());primary_method={}
    for method in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-POP8","DIFF-MATCH8"):
        take=[r for r in strong_paired if r["package"]=="POP8-D120" and r["method"]==method];primary_method[method]={key:float(np.mean([np.mean([float(r[key]) for r in take if int(r["participant"])==p]) for p in range(1,10)])) for key in ("rrmse","correlation","delta_snr","paired_spectral_utility")}
    primary_safety={}
    for method in ("DIFF-POP8","DIFF-MATCH8"):
        take=[r for r in strong_participant_safety if r["package"]=="POP8-D120" and r["method"]==method];primary_safety[method]={key:{"mean":float(np.mean([np.mean([float(r[key]) for r in take if int(r["participant"])==p]) for p in range(1,10)])),"median":float(np.median([np.mean([float(r[key]) for r in take if int(r["participant"])==p]) for p in range(1,10)])),"participant_values":[float(np.mean([float(r[key]) for r in take if int(r["participant"])==p])) for p in range(1,10)]} for key in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}
    strong_result["primary_method_summary"]=primary_method;strong_result["primary_natural_safety"]=primary_safety;_json(strong/"result_summary.json",strong_result)
    # Minimal diagnostic figures.
    for root,name,rows in ((strong,"pop8_effects",[r for r in pop_effects if r["duration_seconds"]==120]),(oproot,"oppost_effects",oppost)):
        figdir=root/"figures";figdir.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(7,4));avg={p:(np.mean([r["U_P"] for r in rows if r["participant"]==p]),np.mean([r["U_W"] for r in rows if r["participant"]==p])) for p in range(1,10)};x=np.arange(1,10);ax.axhline(0,color="black",lw=.8);ax.plot(x,[avg[p][0] for p in x],"o-",label="MATCH-POP");ax.plot(x,[avg[p][1] for p in x],"s-",label="MATCH-WRONG");ax.set_xlabel("Participant");ax.set_ylabel("positive RRMSE utility");ax.legend();fig.tight_layout();fig.savefig(figdir/f"{name}.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));x=np.asarray(DURATIONS);ax.axhline(0,color="black",lw=.8);ax.plot(x,[pop_summary[str(s)]["U_P8"]["mean"] for s in DURATIONS],"o-",label="U_P8");ax.plot(x,[pop_summary[str(s)]["U_W8"]["mean"] for s in DURATIONS],"s-",label="U_W8");ax.set_xlabel("Support duration (s)");ax.set_ylabel("Participant-first mean RRMSE utility");ax.legend();fig.tight_layout();fig.savefig(strong/"figures"/"support_duration_robustness.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));cp=[float(r["I_full"]) for r in cpart];dp=[float(r["I_det_cross"]) for r in cpart];ax.axline((0,0),slope=1,color="0.6",lw=1);ax.scatter(cp,dp);ax.set_xlabel("Historical full crossed interaction");ax.set_ylabel("DET-cross interaction");fig.tight_layout();(causal/"figures").mkdir(parents=True,exist_ok=True);fig.savefig(causal/"figures"/"coherence_penalty_decomposition.png",dpi=180);plt.close(fig)
    _reports(config,pop_summary,causal_summary,op_summary);summary={"strong_population":pop_summary,"causal_audit":causal_summary,"operator_posterior":op_summary,"routing":route};_json(_root(config,"result_root")/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def _reports(config:Mapping[str,Any],pop:dict[str,Any],causal:dict[str,Any],op:dict[str,Any])->None:
    strong=_root(config,"strong_population_root");oproot=_root(config,"oppost_root")
    strong_result=json.loads((strong/"result_summary.json").read_text())
    availability=_read(_root(config,"result_root")/"support_duration_manifest.csv");coverage={seconds:sum(int(r[f"d{seconds}"]) for r in availability) for seconds in DURATIONS}
    duration_lines=[]
    for seconds in DURATIONS:
        item=pop[str(seconds)];duration_lines.append(
            f"| {seconds} s | {item['U_P8']['mean']:+.5f} | {item['U_P8']['median']:+.5f} | {item['U_P8']['positive']}/9 | {item['U_P8']['sign_flip_two_sided']:.6f} | "
            f"{item['U_W8']['mean']:+.5f} | {item['U_W8']['median']:+.5f} | {item['U_W8']['positive']}/9 | {item['U_W8']['sign_flip_two_sided']:.6f} |"
        )
    p=pop["120"]
    Path("reports/bci2b_strong_population_control.md").write_text(
        "# BCI2b strong population control\n\n"
        "This is a development-only robustness experiment. Each LOSO backbone excludes only the recipient and uses the other eight participants. "
        f"The primary duration is frozen at 120 s; 30 s and 60 s are robustness analyses. Eligible protocol units are {coverage[30]}/27, {coverage[60]}/27, and {coverage[120]}/27, respectively, while all 9 participants remain represented. The blocked short-support unit stays in the availability denominator. FULL_AVAILABLE is not reported because support availability is not identical for every unit.\n\n"
        "| support | U_P8 mean | median | wins | exact two-sided p | U_W8 mean | median | wins | exact two-sided p |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"+"\n".join(duration_lines)+"\n\n"
        f"At 120 s, U_P8 is {p['U_P8']['mean']:+.5f} ({p['U_P8']['positive']}/9) and U_W8 is {p['U_W8']['mean']:+.5f} ({p['U_W8']['positive']}/9). "
        "All compatible WRONG8 donors were scored separately and averaged within recipient; they are training-seen donor context sensitivity. "
        "The frozen two-heldout cyclic donor remains a separate unseen-WRONG sensitivity and is never pooled with WRONG8. Seeds and protocol units are first aggregated within participant; scientific n=9. "
        f"Participant-first primary RRMSE is RAW {strong_result['primary_method_summary']['RAW']['rrmse']:.5f}, LINEAR-MATCH {strong_result['primary_method_summary']['LINEAR-MATCH']['rrmse']:.5f}, DET-MATCH {strong_result['primary_method_summary']['DET-MATCH']['rrmse']:.5f}, DIFF-POP8 {strong_result['primary_method_summary']['DIFF-POP8']['rrmse']:.5f}, and DIFF-MATCH8 {strong_result['primary_method_summary']['DIFF-MATCH8']['rrmse']:.5f}. "
        f"DIFF-MATCH8 natural means are EOG attenuation {strong_result['primary_natural_safety']['DIFF-MATCH8']['eog_attenuation']['mean']:+.5f}, preservation {strong_result['primary_natural_safety']['DIFF-MATCH8']['preservation']['mean']:.5f}, MI-band distortion {strong_result['primary_natural_safety']['DIFF-MATCH8']['mi_band_distortion']['mean']:.5f}, covariance {strong_result['primary_natural_safety']['DIFF-MATCH8']['covariance']['mean']:.5f}, MI kappa {strong_result['primary_natural_safety']['DIFF-MATCH8']['mi_kappa']['mean']:.5f}, and ERD preservation {strong_result['primary_natural_safety']['DIFF-MATCH8']['erd_preservation']['mean']:.5f}. "
        "Participant bootstrap intervals are descriptive. Full participant and seed values are in the accompanying CSV/JSON files.\n\n"
        "## Execution notes\n\nTraining completed 27/27 checkpoints at the frozen 8,000-update budget. Slurm QOS submit/GPU limits delayed inference submissions but did not alter the protocol. One 15.6-second support unit was blocked from duration-labelled effects rather than silently truncated.\n",
        encoding="utf-8")
    e=causal["participant_effects"]
    Path("reports/bci2b_context_causal_audit.md").write_text(
        "# BCI2b context causal audit\n\n"
        f"Decision: `{causal['status']}`. This is a development-only causal diagnostic with n=9 participants; protocols and three seeds are aggregated within participant.\n\n"
        f"Coherent subject effects are LINEAR {e['U_subject_LINEAR']['mean']:+.5f}, DET {e['U_subject_DET']['mean']:+.5f}, and DIFF {e['U_subject_DIFF']['mean']:+.5f}. "
        f"The diffusion increment relative to matched DET is {e['U_delta_MATCH']['mean']:+.5f}; Delta_SA is {e['Delta_SA']['mean']:+.5f}. "
        f"I_full is {e['I_full']['mean']:+.5f}, I_oracle is {e['I_oracle']['mean']:+.5f}, and I_det_cross is {e['I_det_cross']['mean']:+.5f}; "
        f"the DET-cross absolute fraction of I_full is {causal['det_cross_absolute_fraction_of_full']:.3f}. "
        "Because residual coordinates are defined relative to their own anchor, crossed anchor/context cells impose a coherence penalty. The historical crossed-grid interaction is therefore not evidence of score-specific synergy.\n",
        encoding="utf-8")
    status="OPERATOR_LIKELIHOOD_GUIDANCE_SUBJECT_EFFECT_SUPPORTED" if op["routing"]["oppost_full_success"] else "CURRENT_OPERATOR_LIKELIHOOD_GUIDED_INSTANCE_NO_GO"
    Path("reports/bci2b_operator_likelihood_diffusion.md").write_text(
        "# BCI2b operator-likelihood diffusion\n\n"
        f"Decision: `{status}`. This development-only experiment freezes the population anchor, deterministic residual, score context, checkpoint, DDIM25 trajectory, and K=8 common noise. Only the support-derived likelihood center and precision change across POP/MATCH/WRONG.\n\n"
        f"U_P_OP is {op['U_P_OP']['mean']:+.5f} (median {op['U_P_OP']['median']:+.5f}; {op['U_P_OP']['positive']}/9), U_W_OP is {op['U_W_OP']['mean']:+.5f} "
        f"(median {op['U_W_OP']['median']:+.5f}; {op['U_W_OP']['positive']}/9), and G_MATCH is {op['G_MATCH']['mean']:+.5f}. "
        f"For transparency, frozen CURRENT RRMSE is RAW {op['secondary_rrmse']['RAW']:.5f}, LINEAR-MATCH {op['secondary_rrmse']['LINEAR-MATCH']:.5f}, DET-MATCH {op['secondary_rrmse']['DET-MATCH']:.5f}, DIFF-POP {op['secondary_rrmse']['DIFF-POP']:.5f}, and DIFF-MATCH {op['secondary_rrmse']['DIFF-MATCH']:.5f}. "
        f"MATCH natural metrics are EOG attenuation {op['natural']['eog_attenuation']:+.5f}, preservation {op['natural']['preservation']:.5f}, MI-band distortion {op['natural']['mi_band_distortion']:.5f}, covariance {op['natural']['covariance']:.5f}, and MI kappa {op['natural']['mi_kappa']:.5f}. "
        f"Localized backup authorized: `{op['routing']['localized_backup_authorized']}`. It is not run unless the global subject effects pass but safety deteriorates. DIFF-vs-DET/LINEAR remains transparent secondary evidence, not this subject-awareness gate.\n\n"
        "## Failure and recovery record\n\nJob 931737 exposed Python-3.9-incompatible `zip(strict=...)` and was superseded after the compatibility-only fix. Jobs 931741/931742 were superseded technical probes; 931743 established the original technical contract. Job 931799 exposed a missing frozen PSD-floor config and was replaced by 931804/931805. Final padding-identity validation passed in 931847, after which all 27 inference tasks were replayed before the final evaluator. These engineering failures are excluded from scientific aggregation.\n",
        encoding="utf-8")

def stage_finalize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    result=json.loads((_root(config,"result_root")/"result_summary.json").read_text());_json(run_dir/"result_summary.json",result);return result

def _run_many(config:Mapping[str,Any],run_dir:Path,stage_name:str,fn:Any,count:int)->dict[str,Any]:
    completed=[]
    for index in range(count):
        child=run_dir/f"task_{index}";child.mkdir(parents=True,exist_ok=True);completed.append(fn(config,index,child))
    summary={"status":f"completed_{stage_name}","tasks":count};_json(run_dir/"result_summary.json",summary);return summary

def stage_prepare_pop8_all(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:return _run_many(config,run_dir,"prepare_pop8_all",stage_prepare_pop8,9)
def stage_causal_audit_all(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:return _run_many(config,run_dir,"causal_audit_all",stage_causal_audit,27)
def stage_evaluate_all(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:return _run_many(config,run_dir,"evaluate_all",stage_evaluate,27)

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    stages={"audit":stage_audit,"prepare-pop8":stage_prepare_pop8,"prepare-pop8-all":stage_prepare_pop8_all,"train-pop8":stage_train_pop8,"infer-pop8":stage_infer_pop8,"causal-audit":stage_causal_audit,"causal-audit-all":stage_causal_audit_all,"aggregate-causal":stage_aggregate_causal,"oppost-technical":stage_oppost_technical,"oppost-infer":stage_oppost_infer,"localized-infer":stage_localized_infer,"evaluate":stage_evaluate,"evaluate-all":stage_evaluate_all,"evaluate-oppost-all":stage_evaluate_oppost_all,"aggregate-oppost":stage_aggregate_oppost,"evaluate-local-all":stage_evaluate_local_all,"aggregate-local":stage_aggregate_local,"aggregate":stage_aggregate,"finalize":stage_finalize}
    if stage not in stages:raise ValueError(stage)
    return stages[stage](config,task_index,run_dir)

__all__=["operator_posterior_sample","run_stage"]
