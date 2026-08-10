"""BrainID Gate-01R: one-shot nuisance-invariant construct-validity repair.

This module intentionally exposes no denoiser, diffusion, bridge, CacheKV, or
identity-guided restoration training stage.  Day-200 and sealed PhysioMotion
access fail before signal data are dereferenced.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt, welch
from scipy.stats import kurtosis, trim_mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from eeg_cgdr.experiments.brainid_gate_v17 import (
    _bootstrap,
    _csv,
    _eer,
    _json,
    _read_csv,
    _sign_flip_p,
    _target_channels,
    fold_members,
    guard_physio,
)


ROLE_TO_DATASET = {"R": "Day_1", "T": "Day_7", "G": "Day_80"}
FORBIDDEN_DATASET = "Day_200"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _root(c: Mapping[str, Any]) -> Path:
    return Path(c["result_root"])


def _data(c: Mapping[str, Any]) -> Path:
    return Path(c["data_root"])


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def guard_role(role: str) -> None:
    if role not in ROLE_TO_DATASET:
        raise PermissionError(f"future/Day-200 access refused: {role}")


def _participant_file(c: Mapping[str, Any], participant: int) -> Path:
    path = _data(c) / "files" / f"S{participant}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _refs(handle: h5py.File, name: str) -> list[h5py.Dataset]:
    if name == FORBIDDEN_DATASET:
        raise PermissionError("Day-200 signal loader is fail-closed")
    return [handle[reference] for reference in np.asarray(handle[name]).reshape(-1)]


def _physio_development(c: Mapping[str, Any]) -> list[int]:
    rows = _read_csv(Path(c["physiomotion_split"]))
    values = sorted(int(row["participant"]) for row in rows if row["role"] == "development")
    if len(values) != 20:
        raise RuntimeError(f"frozen PhysioMotion development set changed: {values}")
    for participant in values:
        guard_physio(c, participant)
    return values


def _freeze_control_usage(c: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Freeze 52 anonymous negative-only block pairs without inventing IDs."""
    available=[]
    for source_participant in range(1,16):
        with h5py.File(_participant_file(c,source_participant),"r") as handle:
            count=int(np.asarray(handle["GroupB"]).size)
        for start in range(0,count-1,2):
            available.append((source_participant,start,start+1))
    rng=np.random.default_rng(int(c["split_seed"]))
    chosen=np.sort(rng.choice(len(available),52,replace=False))
    rows=[]
    for owner,index in enumerate(chosen,1):
        source,a,b=available[int(index)]
        rows.append({"control_owner":owner,"source_file":f"S{source}.mat","groupb_ref_index_a":a,"groupb_ref_index_b":b,"identity_mapping":"anonymous_unverified","positive_pair_allowed":0,"allowed_use":"negative_and_nuisance_calibration_only"})
    return rows


def freeze_protocol(c: Mapping[str, Any]) -> dict[str, Any]:
    old=_load_json(Path(c["immutable_gate01"]))
    expected={"M1_verifier":"FAIL","M0_actionability":"INSUFFICIENT","PASS_01":False}
    if any(old.get(key)!=value for key,value in expected.items()):
        raise RuntimeError(f"immutable Gate-01 decision changed: {old}")
    root=_root(c); frozen=root/"frozen"; frozen.mkdir(parents=True,exist_ok=True)
    split=[]; sessions=[]
    for participant in range(1,16):
        heldout=(participant-1)%5
        for fold in range(5):
            split.append({"outer_fold":fold,"participant":participant,"role":"evaluation" if fold==heldout else "outer_training"})
        for role,day in (("R","Day-1"),("T","Day-7"),("G","Day-80"),("F","Day-200")):
            sessions.append({"participant":participant,"role":role,"acquisition":day,"loader_access":"forbidden" if role=="F" else "allowed"})
    controls=_freeze_control_usage(c)
    _csv(frozen/"split_manifest.csv",split); _csv(frozen/"session_role_manifest.csv",sessions); _csv(frozen/"control_usage_manifest.csv",controls)
    names,points=_target_channels(c)
    _csv(frozen/"channel_mapping.csv",[{"index":i,"channel":name,"x2d":points[i,0],"y2d":points[i,1]} for i,name in enumerate(names)])
    # Freeze only artifact mask geometry from the already opened PhysioMotion
    # development set.  No PhysioMotion waveform is read.
    source_root=Path(c["physiomotion_fairness_root"])/"fair_materialized";templates=[]
    for owner in _physio_development(c):
        with np.load(source_root/f"masks_{owner:02d}.npz",allow_pickle=False) as data:
            for family,mask in zip(data["families"],data["masks"]): templates.append((owner,str(family),np.asarray(mask,bool)))
    rng=np.random.default_rng(int(c["split_seed"]));chosen=np.sort(rng.choice(len(templates),min(int(c["mask_templates"]),len(templates)),replace=False));distance=np.linalg.norm(points[:,None]-points[None],axis=2);length=int(round((float(c["epoch_end_seconds"])-float(c["epoch_start_seconds"]))*int(c["model_sampling_rate"])));masks=[];mask_rows=[]
    for source_index in chosen:
        owner,family,source=templates[int(source_index)];active_t=np.flatnonzero(source.any(0));active_c=np.flatnonzero(source.any(1))
        if not len(active_t) or not len(active_c): continue
        start=int(round(active_t[0]/source.shape[1]*length));end=min(length,max(start+1,int(round((active_t[-1]+1)/source.shape[1]*length))));count=min(max(1,round(len(active_c)/source.shape[0]*len(names))),max(1,round(.35*len(names))));anchor=int(np.random.SeedSequence([int(c["split_seed"]),owner,int(source_index)]).generate_state(1)[0]%len(names));channels=np.argsort(distance[anchor],kind="stable")[:count];target=np.zeros((len(names),length),bool);target[channels,start:end]=True;masks.append(target);mask_rows.append({"mask_id":len(masks)-1,"physiomotion_development_owner":owner,"family":family,"target_channels":count,"target_start":start,"target_end":end,"physiomotion_waveform_used":0})
    np.savez_compressed(frozen/"corruption_masks.npz",masks=np.asarray(masks,np.uint8));_csv(frozen/"corruption_manifest.csv",mask_rows)
    schema={
        "frozen_before_results":True,
        "physiological_view":{"band_hz":[0.5,30.0],"notch_hz":50.0,"baseline_seconds":0.05,"robust_aggregate":"coordinate-wise 12.5% trimmed mean","normalization":"outer-fold channel median/MAD","feature_groups_equal_weight":["ERP amplitude/latency/slope","channel-demeaned spatial topography","relative 1-4/4-8/8-13/13-30 Hz power"]},
        "artifact_view":{"band_hz":[30.0,100.0],"features":["relative high-band power","first difference","kurtosis","peak rate","saturation/flatline","bad-channel distribution"],"forbidden_from_physiological_input":True},
        "negative_pair_matching":["ERP condition","aggregate trial count","outer-fold artifact-energy bin"],
        "wrong_condition_criterion":"wrong-condition same-subject margin <= 0.5 * correct-condition margin",
        "loo_auroc_definition":"mean heldout participant AUROC after leaving one participant out"
    }
    _json(frozen/"feature_schema.json",schema)
    prereg={
        "name":"BrainID Gate-01R","immutable_original_gate":"FAIL","roles":{"R":"Day-1","T":"Day-7","G":"Day-80","F":"Day-200 sealed"},"outer_folds":5,
        "physiological_view":schema["physiological_view"],"artifact_negative_control":schema["artifact_view"],"verifier_a_r":{"parameters_max":1000000,"embedding_dim":64,"objective":"cross-day supervised contrastive","nuisance_heads":{"session":0.10,"artifact_energy_bin":0.10,"rereference_type":0.10}},
        "verifier_b_r":"fixed equal-weight physiological feature cosine metric; evaluator only","thresholds":dict(c["m1r_gate"]),"m0r":dict(c["m0r"]),"results_may_not_change_protocol":True
    }
    (root/"gate01r_preregistration.yaml").write_text(yaml.safe_dump(prereg,sort_keys=False),encoding="utf-8")
    sealed={"day200_opened":False,"day200_loader":"fail_closed","physiomotion_sealed_opened":False,"physiomotion_development_ids":_physio_development(c),"shu_day4_day5_opened":False,"original_gate01_sha":c["immutable_gate01_commit"]}
    _json(frozen/"sealed_guard.json",sealed)
    return {"status":"GATE01R_PROTOCOL_FROZEN","participants":15,"controls_negative_only":52,"channels":len(names),**sealed}


def _events(dataset:h5py.Dataset,c:Mapping[str,Any],seed_parts:list[int],cap:int=64)->tuple[np.ndarray,np.ndarray,bool]:
    shape=dataset.shape
    if len(shape)!=2 or min(shape)!=58: raise RuntimeError(f"unexpected block shape {shape}")
    samples_first=shape[1]==58
    trigger=np.asarray(dataset[:,57] if samples_first else dataset[57,:]).reshape(-1)
    rounded=np.rint(trigger).astype(int); by={label:np.flatnonzero(rounded==label) for label in (1,2)}; count=min(len(by[1]),len(by[2]),cap)
    rng=np.random.default_rng(np.random.SeedSequence([int(c["split_seed"]),*seed_parts])); selected=np.sort(np.concatenate([rng.choice(by[label],count,replace=False) for label in (1,2)]))
    return selected,trigger,samples_first


def _extract_views(dataset:h5py.Dataset,c:Mapping[str,Any],seed_parts:list[int],cap:int=64)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    events,trigger,samples_first=_events(dataset,c,seed_parts,cap)
    source_fs=int(c["source_sampling_rate"]); model_fs=int(c["model_sampling_rate"])
    pre=int(round(float(c["epoch_start_seconds"])*source_fs)); post=int(round(float(c["epoch_end_seconds"])*source_fs))
    phys_sos=butter(4,list(map(float,c["physiological_band_hz"])),btype="bandpass",fs=source_fs,output="sos")
    art_sos=butter(4,list(map(float,c["artifact_band_hz"])),btype="bandpass",fs=source_fs,output="sos")
    notch_b,notch_a=iirnotch(float(c["line_frequency_hz"]),30,fs=source_fs)
    phys=[]; artifact=[]; qc=[]; labels=[]
    for event in events:
        start,end=event+pre,event+post
        if start<0 or end>len(trigger): continue
        raw=np.asarray(dataset[start:end,:57] if samples_first else dataset[:57,start:end].T,np.float64).T
        if not np.isfinite(raw).all(): continue
        raw-=raw.mean(axis=0,keepdims=True)
        notched=filtfilt(notch_b,notch_a,raw,axis=-1)
        low=resample_poly(sosfiltfilt(phys_sos,notched,axis=-1),model_fs,source_fs,axis=-1)
        high=resample_poly(sosfiltfilt(art_sos,notched,axis=-1),model_fs,source_fs,axis=-1)
        baseline=max(1,int(round(float(c["baseline_seconds"])*model_fs))); low-=low[...,:baseline].mean(axis=-1,keepdims=True)
        scale=np.median(np.abs(raw-np.median(raw,axis=-1,keepdims=True)),axis=-1)+1e-12
        difference=np.diff(raw,axis=-1)
        features=np.concatenate([
            np.log(np.mean(high**2,axis=-1)+1e-12),
            np.log(np.mean(difference**2,axis=-1)+1e-12),
            np.nan_to_num(kurtosis(raw,axis=-1,fisher=True),nan=0,posinf=0,neginf=0),
            np.mean(np.abs(difference)>6*scale[:,None],axis=-1),
            np.mean(np.abs(difference)<1e-4*scale[:,None],axis=-1),
        ]).astype(np.float32)
        phys.append(low.astype(np.float32)); artifact.append(high.astype(np.float32)); qc.append(features); labels.append(int(round(trigger[event])))
    return np.asarray(phys),np.asarray(artifact),np.asarray(qc),np.asarray(labels,np.int8)


def _balanced(phys:np.ndarray,artifact:np.ndarray,qc:np.ndarray,labels:np.ndarray,c:Mapping[str,Any],seed_parts:list[int])->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    minimum=min(np.sum(labels==1),np.sum(labels==2)); rng=np.random.default_rng(np.random.SeedSequence([int(c["split_seed"]),*seed_parts])); keep=[]
    for label in (1,2): keep.extend(rng.choice(np.flatnonzero(labels==label),minimum,replace=False))
    keep=np.sort(keep); return phys[keep],artifact[keep],qc[keep],labels[keep]


def prepare_longitudinal(c:Mapping[str,Any],participant:int)->dict[str,Any]:
    if participant not in range(1,16): raise ValueError(participant)
    phys_arrays={}; artifact_arrays={}; inventory=[]
    with h5py.File(_participant_file(c,participant),"r") as handle:
        if FORBIDDEN_DATASET not in handle: raise RuntimeError("Day-200 metadata absent")
        for role,name in ROLE_TO_DATASET.items():
            guard_role(role); px=[];ax=[];qx=[];ly=[]
            for block,dataset in enumerate(_refs(handle,name)):
                p,a,q,y=_extract_views(dataset,c,[participant,ord(role),block]); px.append(p);ax.append(a);qx.append(q);ly.append(y)
            p,a,q,y=_balanced(np.concatenate(px),np.concatenate(ax),np.concatenate(qx),np.concatenate(ly),c,[participant,ord(role)])
            phys_arrays[f"{role}_epochs"]=p.astype(np.float16); phys_arrays[f"{role}_condition"]=y
            artifact_arrays[f"artifact_{role}_epochs"]=a.astype(np.float16); artifact_arrays[f"artifact_{role}_condition"]=y; artifact_arrays[f"artifact_{role}_qc"]=q.astype(np.float32)
            inventory.append({"participant":participant,"role":role,"trials":len(y),"condition1":int(np.sum(y==1)),"condition2":int(np.sum(y==2)),"channels":p.shape[1],"samples":p.shape[2],"day200_opened":0})
    server=_root(c)/"server_arrays"; (server/"physiological").mkdir(parents=True,exist_ok=True); (server/"artifact").mkdir(parents=True,exist_ok=True)
    np.savez_compressed(server/"physiological"/f"subject_{participant:02d}.npz",**phys_arrays); np.savez_compressed(server/"artifact"/f"subject_{participant:02d}.npz",**artifact_arrays)
    _csv(_root(c)/"inventory"/f"subject_{participant:02d}.csv",inventory)
    return {"status":"PREPARED","participant":participant,"roles":["R","T","G"],"day200_opened":False}


def prepare_control(c:Mapping[str,Any],owner:int)->dict[str,Any]:
    rows=_read_csv(_root(c)/"frozen"/"control_usage_manifest.csv"); row=next(r for r in rows if int(r["control_owner"])==owner)
    source=int(Path(row["source_file"]).stem[1:]); phys=[];artifact=[];qc=[];labels=[]
    with h5py.File(_participant_file(c,source),"r") as handle:
        refs=np.asarray(handle["GroupB"]).reshape(-1)
        for local,index in enumerate((int(row["groupb_ref_index_a"]),int(row["groupb_ref_index_b"]))):
            p,a,q,y=_extract_views(handle[refs[index]],c,[9000,owner,local],cap=32); phys.append(p);artifact.append(a);qc.append(q);labels.append(y)
    p,a,q,y=_balanced(np.concatenate(phys),np.concatenate(artifact),np.concatenate(qc),np.concatenate(labels),c,[9000,owner])
    server=_root(c)/"server_arrays"; (server/"controls_physiological").mkdir(parents=True,exist_ok=True);(server/"controls_artifact").mkdir(parents=True,exist_ok=True)
    np.savez_compressed(server/"controls_physiological"/f"control_{owner:02d}.npz",C_epochs=p.astype(np.float16),C_condition=y)
    np.savez_compressed(server/"controls_artifact"/f"control_{owner:02d}.npz",artifact_C_epochs=a.astype(np.float16),artifact_C_condition=y,artifact_C_qc=q.astype(np.float32))
    return {"status":"CONTROL_PREPARED_NEGATIVE_ONLY","control_owner":owner,"trials":len(y),"positive_pair_allowed":False}


def _robust_aggregate(x:np.ndarray,y:np.ndarray,c:Mapping[str,Any],seed_parts:list[int])->tuple[np.ndarray,np.ndarray]:
    size=int(c["aggregate_trials"]); cap=int(c["aggregates_per_condition"]); values=[]; labels=[]
    for condition in (1,2):
        ids=np.flatnonzero(y==condition); rng=np.random.default_rng(np.random.SeedSequence([int(c["split_seed"]),*seed_parts,condition])); ids=rng.permutation(ids); groups=min(len(ids)//size,cap)
        for group in range(groups):
            chunk=np.asarray(x[ids[group*size:(group+1)*size]],np.float32); values.append(trim_mean(chunk,float(c["trim_fraction"]),axis=0));labels.append(condition)
    return np.asarray(values,np.float32),np.asarray(labels,np.int8)


def load_role(c:Mapping[str,Any],participant:int,role:str,view:str="physiological")->tuple[np.ndarray,np.ndarray,np.ndarray|None]:
    guard_role(role); server=_root(c)/"server_arrays"
    if view=="physiological":
        with np.load(server/view/f"subject_{participant:02d}.npz") as data: x=np.asarray(data[f"{role}_epochs"],np.float32); y=np.asarray(data[f"{role}_condition"])
        aggregated,labels=_robust_aggregate(x,y,c,[participant,ord(role),101]); return aggregated,labels,None
    if view=="artifact":
        with np.load(server/view/f"subject_{participant:02d}.npz") as data: x=np.asarray(data[f"artifact_{role}_epochs"],np.float32);y=np.asarray(data[f"artifact_{role}_condition"]);q=np.asarray(data[f"artifact_{role}_qc"],np.float32)
        aggregated,labels=_robust_aggregate(x,y,c,[participant,ord(role),101]); qagg,_=_robust_aggregate(q[...,None],y,c,[participant,ord(role),101]); return aggregated,labels,qagg[...,0]
    raise ValueError(view)


def load_control(c:Mapping[str,Any],owner:int,view:str="physiological")->tuple[np.ndarray,np.ndarray,np.ndarray|None]:
    server=_root(c)/"server_arrays"
    if view=="physiological":
        with np.load(server/"controls_physiological"/f"control_{owner:02d}.npz") as data: x=np.asarray(data["C_epochs"],np.float32);y=np.asarray(data["C_condition"])
        a,l=_robust_aggregate(x,y,c,[9000,owner,101]); return a,l,None
    with np.load(server/"controls_artifact"/f"control_{owner:02d}.npz") as data: x=np.asarray(data["artifact_C_epochs"],np.float32);y=np.asarray(data["artifact_C_condition"]);q=np.asarray(data["artifact_C_qc"],np.float32)
    a,l=_robust_aggregate(x,y,c,[9000,owner,101]);qa,_=_robust_aggregate(q[...,None],y,c,[9000,owner,101]);return a,l,qa[...,0]


def aggregate_inventory(c:Mapping[str,Any])->dict[str,Any]:
    rows=[]
    for participant in range(1,16): rows.extend(_read_csv(_root(c)/"inventory"/f"subject_{participant:02d}.csv"))
    controls=list((_root(c)/"server_arrays"/"controls_physiological").glob("control_*.npz")); _csv(_root(c)/"data_inventory.csv",rows)
    complete=sum(all(any(int(r["participant"])==p and r["role"]==role for r in rows) for role in ("R","T","G")) for p in range(1,16)); channels=min(int(r["channels"]) for r in rows)
    passed=complete>=13 and channels>=24 and len(controls)==52
    result={"status":"DATA_PROTOCOL_STILL_VALID" if passed else "DATA_PROTOCOL_INSUFFICIENT","PASS":passed,"participants_with_RTG":complete,"common_channels":channels,"controls_opened_for_negative_calibration":len(controls)==52,"controls_prepared":len(controls),"controls_positive_pairs":0,"day200_opened":False,"physiomotion_sealed_opened":False}
    _json(_root(c)/"data_protocol_decision.json",result); return result


def _component_features(x:np.ndarray,fs:int=250)->np.ndarray:
    """Frozen ERP amplitude, latency, and slope groups."""
    windows=((.08,.18),(.18,.30),(.30,.50),(.50,.70)); groups=[]
    event_zero=int(round(.05*fs))
    for start_s,end_s in windows:
        start=event_zero+int(round(start_s*fs)); end=min(x.shape[-1],event_zero+int(round(end_s*fs))); segment=x[...,start:end]
        amplitude=segment.mean(-1); peak=np.argmax(np.abs(segment),axis=-1)/max(1,segment.shape[-1]-1); slope=(segment[...,-1]-segment[...,0])/max(1,segment.shape[-1]-1)
        groups.extend((amplitude,peak,slope))
    return np.concatenate(groups,axis=1).astype(np.float32)


def _topography_features(x:np.ndarray,fs:int=250)->np.ndarray:
    event_zero=int(round(.05*fs)); windows=((.08,.18),(.18,.30),(.30,.50),(.50,.70)); values=[]
    for start_s,end_s in windows:
        start=event_zero+int(round(start_s*fs));end=min(x.shape[-1],event_zero+int(round(end_s*fs))); topo=x[...,start:end].mean(-1); topo-=topo.mean(1,keepdims=True); topo/=np.maximum(np.linalg.norm(topo,axis=1,keepdims=True),1e-8); values.append(topo)
    return np.concatenate(values,axis=1).astype(np.float32)


def _spectral_features(x:np.ndarray,bands:Iterable[Iterable[float]],fs:int=250)->np.ndarray:
    f,p=welch(x,fs=fs,nperseg=min(128,x.shape[-1]),axis=-1); total=np.maximum(trapezoid(p[...,((f>=1)&(f<=30))],f[(f>=1)&(f<=30)],axis=-1),1e-12); values=[]
    for low,high in bands:
        keep=(f>=float(low))&(f<float(high)); values.append(np.log(np.maximum(trapezoid(p[...,keep],f[keep],axis=-1)/total,1e-12)))
    return np.concatenate(values,axis=1).astype(np.float32)


def physiological_feature_groups(x:np.ndarray,c:Mapping[str,Any])->tuple[np.ndarray,np.ndarray,np.ndarray]:
    return _component_features(x),_topography_features(x),_spectral_features(x,c["feature_bands_hz"])


class VerifierBR:
    """Evaluator-only equal-weight fixed physiological cosine metric."""
    def __init__(self,means:list[np.ndarray],scales:list[np.ndarray]): self.means=means;self.scales=scales
    def embed(self,x:np.ndarray,c:Mapping[str,Any])->np.ndarray:
        output=[]
        for features,center,scale in zip(physiological_feature_groups(x,c),self.means,self.scales):
            z=(features-center)/scale; z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8); output.append(z)
        joined=np.concatenate(output,axis=1)/math.sqrt(3); return joined.astype(np.float32)
    def save(self,path:Path)->None:
        path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**{f"mean_{i}":v for i,v in enumerate(self.means)},**{f"scale_{i}":v for i,v in enumerate(self.scales)})
    @classmethod
    def load(cls,path:Path)->"VerifierBR":
        with np.load(path) as data: return cls([np.asarray(data[f"mean_{i}"]) for i in range(3)],[np.asarray(data[f"scale_{i}"]) for i in range(3)])


def fit_verifier_b_r(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    training,evaluation=fold_members(fold); groups=[[],[],[]]
    for participant in training:
        for role in ("R","T","G"):
            x,_,_=load_role(c,participant,role)
            for index,value in enumerate(physiological_feature_groups(x,c)): groups[index].append(value)
    means=[];scales=[]
    for values in groups:
        joined=np.concatenate(values); center=np.median(joined,axis=0); scale=np.median(np.abs(joined-center),axis=0)/.67448975; positive=scale[scale>0]; scale=np.maximum(scale,(np.median(positive) if len(positive) else 1.)*1e-4); means.append(center.astype(np.float32));scales.append(scale.astype(np.float32))
    model=VerifierBR(means,scales); path=_root(c)/"server_checkpoints"/f"fold_{fold:02d}"/"verifier_b_r.npz";model.save(path)
    result={"status":"VERIFIER_B_R_FROZEN","fold":fold,"training":training,"evaluation":evaluation,"feature_dimensions":[len(v) for v in means],"equal_group_weights":True,"imported_by_training":False};_json(run_dir/"result_summary.json",result);return result


def _cosine_embed(x:np.ndarray,bins:int=16)->np.ndarray:
    length=x.shape[-1]; edges=np.linspace(0,length,bins+1,dtype=int); features=np.concatenate([x[...,edges[i]:edges[i+1]].mean(-1) for i in range(bins)],axis=1); features-=features.mean(1,keepdims=True); return (features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)).astype(np.float32)


def _band_view(x:np.ndarray,band:tuple[float,float],fs:int=250)->np.ndarray:
    high=min(float(band[1]),fs/2-1); low=float(band[0]); sos=butter(4,[low,high],btype="bandpass",fs=fs,output="sos");return sosfiltfilt(sos,x,axis=-1).astype(np.float32)


def _pair_rows(query:dict[tuple[int,int],np.ndarray],gallery:dict[tuple[int,int],np.ndarray],thresholds:dict[int,float]|None=None)->tuple[list[dict[str,Any]],np.ndarray,np.ndarray]:
    rows=[]; all_scores=[];all_labels=[]
    participants=sorted({p for p,_ in query})
    for participant,condition in sorted(query):
        q=query[(participant,condition)]; ids=participants; g=np.stack([gallery[(p,condition)] for p in ids]); similarity=q@g.T; own=ids.index(participant); positive=similarity[:,own]; negative=np.delete(similarity,own,axis=1); margin=positive-np.median(negative,axis=1); rank=np.argmax(similarity,axis=1)==own
        scores=np.concatenate([positive,negative.reshape(-1)]);labels=np.concatenate([np.ones(len(positive)),np.zeros(negative.size)]); threshold=float(thresholds[condition] if thresholds else np.quantile(negative,.95))
        rows.append({"participant":participant,"condition":condition,"identity_margin":float(np.mean(margin)),"auroc":float(roc_auc_score(labels,scores)),"rank1":float(np.mean(rank)),"tar_at_far5":float(np.mean(positive>=threshold))});all_scores.extend(scores);all_labels.extend(labels)
    return rows,np.asarray(all_scores),np.asarray(all_labels)


def _representations(c:Mapping[str,Any],participant:int,role:str)->dict[str,np.ndarray]:
    phys,labels,_=load_role(c,participant,role,"physiological"); artifact,artifact_labels,_=load_role(c,participant,role,"artifact")
    if not np.array_equal(labels,artifact_labels): raise RuntimeError("view condition mismatch")
    template=np.stack([phys[labels==condition].mean(0) for condition in labels]); induced=phys-template
    amplitude=phys.mean(-1,keepdims=True)*np.ones_like(phys)
    topography=phys.mean(-1,keepdims=True); topography-=topography.mean(1,keepdims=True); topography=np.repeat(topography,phys.shape[-1],axis=-1)
    morphology=phys/np.maximum(np.sqrt(np.mean(phys**2,axis=(1,2),keepdims=True)),1e-8)
    result={"physiological":phys,"phase_locked_erp":template,"induced_residual":induced,"artifact_acquisition_only":artifact,"amplitude_only":amplitude,"topography_only":topography,"latency_morphology_only":morphology}
    for low,high in ((.5,4),(4,8),(8,13),(13,30)): result[f"band_{low:g}_{high:g}"]=_band_view(phys,(low,high))
    result["band_30_45"]=_band_view(artifact,(30,45));result["band_45_100"]=_band_view(artifact,(45,100))
    return result


def run_carrier_forensic(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    training,evaluation=fold_members(fold); all_data={(p,role):_representations(c,p,role) for p in training+evaluation for role in ("R","G")}; rows=[];variance=[]
    panel_names=list(next(iter(all_data.values())))
    for panel in panel_names:
        embeddings={}
        for key,reps in all_data.items(): embeddings[key]=_cosine_embed(reps[panel])
        query={};gallery={}
        for participant in evaluation:
            _,labels,_=load_role(c,participant,"R")
            _,glabels,_=load_role(c,participant,"G")
            for condition in (1,2): query[(participant,condition)]=embeddings[(participant,"R")][labels==condition];gallery[(participant,condition)]=embeddings[(participant,"G")][glabels==condition].mean(0)
        local,scores,labels_binary=_pair_rows(query,gallery)
        for item in local: rows.append({"fold":fold,"metric":"cosine","panel":panel,**item})
        # Descriptive variance partition on outer training embeddings.
        z=[];owners=[];sessions=[];conditions=[];artifact_energy=[]
        for participant in training:
            for session,role in enumerate(("R","G")):
                _,labels,_=load_role(c,participant,role);_,artifact_labels,local_qc=load_role(c,participant,role,"artifact")
                if not np.array_equal(labels,artifact_labels): raise RuntimeError("forensic label mismatch")
                values=embeddings[(participant,role)];z.extend(values);owners.extend([participant]*len(values));sessions.extend([session]*len(values));conditions.extend(labels);artifact_energy.extend(_artifact_scalar(local_qc))
        z=np.asarray(z); total=float(np.mean(np.var(z,axis=0)))+1e-12;artifact_bins=np.digitize(artifact_energy,np.quantile(artifact_energy,[1/3,2/3]))
        def between(groups): return float(np.mean(np.var(np.stack([z[np.asarray(groups)==value].mean(0) for value in sorted(set(groups))]),axis=0))/total)
        variance.append({"fold":fold,"panel":panel,"identity_fraction":between(owners),"session_fraction":between(sessions),"condition_fraction":between(conditions),"artifact_fraction":between(artifact_bins)})
    # Preserve original Gate-01 verifier metrics as immutable full-input references.
    old_root=Path(c["immutable_gate01"]).parent
    for name in ("a","b"):
        for item in _read_csv(old_root/f"m1_verifier_{name}_subject_metrics.csv"):
            if (int(item["participant"])-1)%5==fold: rows.append({"fold":fold,"metric":f"old_verifier_{name.upper()}","panel":"full_original","participant":int(item["participant"]),"condition":"aggregate","identity_margin":float(item["identity_margin"]),"auroc":float(item["auroc"]),"rank1":float(item["rank1"]),"tar_at_far5":float(item["tar_at_far5"])})
    out=_root(c)/"j1"/f"fold_{fold:02d}";_csv(out/"carrier_metrics.csv",rows);_csv(out/"variance_partition.csv",variance);result={"status":"J1_FORENSIC_COMPLETE","fold":fold,"panels":panel_names,"scientific_gate":False};_json(run_dir/"result_summary.json",result);return result


def aggregate_forensic(c:Mapping[str,Any])->dict[str,Any]:
    rows=[];variance=[]
    for fold in range(5): rows.extend(_read_csv(_root(c)/"j1"/f"fold_{fold:02d}"/"carrier_metrics.csv"));variance.extend(_read_csv(_root(c)/"j1"/f"fold_{fold:02d}"/"variance_partition.csv"))
    grouped=defaultdict(list)
    for row in rows: grouped[(row["metric"],row["panel"],int(row["participant"]))].append(row)
    participant=[]
    for (metric,panel,p),values in grouped.items(): participant.append({"metric":metric,"panel":panel,"participant":p,"auroc":mean(float(v["auroc"]) for v in values),"identity_margin":mean(float(v["identity_margin"]) for v in values),"rank1":mean(float(v["rank1"]) for v in values),"tar_at_far5":mean(float(v["tar_at_far5"]) for v in values)})
    summary=[]
    for key in sorted({(r["metric"],r["panel"]) for r in participant}):
        selected=[r for r in participant if (r["metric"],r["panel"])==key];summary.append({"metric":key[0],"panel":key[1],"participants":len(selected),"auroc_mean":mean(r["auroc"] for r in selected),"identity_margin_mean":mean(r["identity_margin"] for r in selected),"positive":sum(r["identity_margin"]>0 for r in selected)})
    root=_root(c);_csv(root/"j1_carrier_metrics.csv",participant);_csv(root/"j1_band_ablation.csv",[r for r in summary if str(r["panel"]).startswith("band_")]);_csv(root/"j1_variance_partition.csv",variance)
    report=Path(__file__).parents[3]/"reports"/"j1_forensic_report.md";report.parent.mkdir(parents=True,exist_ok=True);lines=["# BrainID Gate-01R identity-carrier forensic","","Descriptive causal forensic only; no panel selected the J2 representation. J2 remained frozen to the 0.5–30 Hz physiological view.","","| Metric | Panel | N | AUROC | Margin | Positive |","|---|---|---:|---:|---:|---:|"]
    for row in summary: lines.append(f"| {row['metric']} | {row['panel']} | {row['participants']} | {row['auroc_mean']:.4f} | {row['identity_margin_mean']:.4f} | {row['positive']} |")
    report.write_text("\n".join(lines)+"\n",encoding="utf-8");return {"status":"J1_FORENSIC_AGGREGATED","rows":len(participant),"panels":len(summary)}


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx:Any,x:torch.Tensor)->torch.Tensor: return x.view_as(x)
    @staticmethod
    def backward(ctx:Any,gradient:torch.Tensor)->tuple[torch.Tensor]: return (-gradient,)


class BrainprintVerifierAR(nn.Module):
    def __init__(self,channels:int=57,embedding_dim:int=64,adjacency:torch.Tensor|None=None):
        super().__init__(); self.register_buffer("adjacency",torch.eye(channels) if adjacency is None else adjacency)
        self.temporal=nn.Sequential(nn.Conv1d(channels,96,15,padding=7,bias=False),nn.BatchNorm1d(96),nn.GELU(),nn.Conv1d(96,128,9,padding=4,stride=2,bias=False),nn.BatchNorm1d(128),nn.GELU(),nn.Conv1d(128,128,7,padding=3,stride=2,bias=False),nn.BatchNorm1d(128),nn.GELU())
        self.embedding=nn.Linear(128,embedding_dim); self.session_head=nn.Linear(embedding_dim,3);self.artifact_head=nn.Linear(embedding_dim,3);self.reference_head=nn.Linear(embedding_dim,2)
    def encode(self,x:torch.Tensor)->torch.Tensor:
        x=torch.einsum("ij,bjt->bit",self.adjacency,x);return F.normalize(self.embedding(self.temporal(x).mean(-1)),dim=-1)
    def nuisance(self,z:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        reverse=_GradientReverse.apply(z);return self.session_head(reverse),self.artifact_head(reverse),self.reference_head(reverse)


def _adjacency(c:Mapping[str,Any])->torch.Tensor:
    _,points=_target_channels(c);distance=np.linalg.norm(points[:,None]-points[None],axis=2);sigma=np.median(np.sort(distance,axis=1)[:,1:5]);matrix=np.exp(-(distance**2)/(2*sigma**2));matrix[distance>np.sort(distance,axis=1)[:,6,None]]=0;matrix+=np.eye(len(matrix));matrix/=matrix.sum(1,keepdims=True);return torch.tensor(matrix,dtype=torch.float32)


def _fit_signal_scale(c:Mapping[str,Any],participants:list[int])->tuple[np.ndarray,np.ndarray]:
    values=[]
    for participant in participants:
        for role in ("R","T","G"):
            x,_,_=load_role(c,participant,role);values.append(x)
    joined=np.concatenate(values);center=np.median(joined,axis=(0,2),keepdims=True);scale=np.median(np.abs(joined-center),axis=(0,2),keepdims=True)/.67448975;positive=scale[scale>0];scale=np.maximum(scale,(np.median(positive) if len(positive) else 1.)*1e-4);return center.astype(np.float32),scale.astype(np.float32)


def _standardize(x:np.ndarray,center:np.ndarray,scale:np.ndarray)->np.ndarray: return ((x-center)/scale).astype(np.float32)


def _augment_ar(x:torch.Tensor,generator:torch.Generator,c:Mapping[str,Any])->tuple[torch.Tensor,torch.Tensor]:
    batch,channels,length=x.shape;gain=1-float(c["verifier_a_r"]["gain_fraction"])+2*float(c["verifier_a_r"]["gain_fraction"])*torch.rand((batch,1,1),generator=generator,device=x.device);out=x*gain
    reference_type=torch.randint(0,2,(batch,),generator=generator,device=x.device); reference_ids=torch.randint(0,channels,(batch,),generator=generator,device=x.device);reference=out[torch.arange(batch,device=x.device),reference_ids].unsqueeze(1);out=out-reference_type[:,None,None]*reference
    count=max(1,int(round(float(c["verifier_a_r"]["dropout_fraction"])*channels)))
    for index in range(batch): out[index,torch.randperm(channels,generator=generator,device=x.device)[:count]]=0
    shifts=torch.randint(-int(c["verifier_a_r"]["jitter_samples"]),int(c["verifier_a_r"]["jitter_samples"])+1,(batch,),generator=generator,device=x.device);shifted=torch.zeros_like(out)
    for index,value in enumerate(shifts):
        shift=int(value.item())
        if shift>0: shifted[index,:,shift:]=out[index,:,:length-shift]
        elif shift<0: shifted[index,:,:length+shift]=out[index,:,-shift:]
        else: shifted[index]=out[index]
    return shifted,reference_type


def _artifact_scalar(qc:np.ndarray)->np.ndarray: return np.mean(qc[:,:57],axis=1)


def train_verifier_a_r(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    training,evaluation=fold_members(fold);center,scale=_fit_signal_scale(c,training);records={};artifact_values=[]
    for participant in training:
        for session,role in enumerate(("R","T","G")):
            x,y,_=load_role(c,participant,role);_,ay,q=load_role(c,participant,role,"artifact");
            if not np.array_equal(y,ay): raise RuntimeError("physiology/artifact labels diverged")
            records[(participant,session)]=(x,y,_artifact_scalar(q));artifact_values.extend(_artifact_scalar(q))
    controls={}
    for owner in range(1,53):
        x,y,_=load_control(c,owner);_,ay,q=load_control(c,owner,"artifact");controls[owner]=(x,y,_artifact_scalar(q));artifact_values.extend(_artifact_scalar(q))
    cuts=np.quantile(artifact_values,[1/3,2/3]);device=torch.device("cuda");seed=int(c["training_seed"])+fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    model=BrainprintVerifierAR(57,int(c["verifier_a_r"]["embedding_dim"]),_adjacency(c)).to(device);parameters=sum(v.numel() for v in model.parameters());
    if parameters>int(c["verifier_a_r"]["max_parameters"]): raise RuntimeError(parameters)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(c["verifier_a_r"]["learning_rate"]),weight_decay=float(c["verifier_a_r"]["weight_decay"]));generator=torch.Generator(device=device).manual_seed(seed+1000);rng=np.random.default_rng(seed+2000);losses=[]
    model.train();participants=list(training);temperature=float(c["verifier_a_r"]["temperature"]);nuisance_weight=float(c["verifier_a_r"]["nuisance_weight_each"])
    for epoch in range(int(c["verifier_a_r"]["epochs"])):
        local=[]
        for step in range(int(c["verifier_a_r"]["steps_per_epoch"])):
            condition=1+(step+epoch)%2;desired=(step+epoch)%3;x1=[];x2=[];s1=[];s2=[];bins1=[];bins2=[]
            for participant in participants:
                session_a,session_b=rng.choice(3,2,replace=False);samples=[]
                for session in (session_a,session_b):
                    x,y,q=records[(participant,int(session))];ids=np.flatnonzero(y==condition);bins=np.digitize(q[ids],cuts);near=ids[np.argmin(np.abs(bins-desired))];samples.append((x[near],int(session),int(np.digitize(q[near],cuts))))
                x1.append(samples[0][0]);s1.append(samples[0][1]);bins1.append(samples[0][2]);x2.append(samples[1][0]);s2.append(samples[1][1]);bins2.append(samples[1][2])
            control_x=[]
            for owner in rng.choice(np.arange(1,53),len(participants),replace=False):
                x,y,q=controls[int(owner)];ids=np.flatnonzero(y==condition);bins=np.digitize(q[ids],cuts);control_x.append(x[ids[np.argmin(np.abs(bins-desired))]])
            a=torch.tensor(_standardize(np.asarray(x1),center,scale),device=device);p=torch.tensor(_standardize(np.asarray(x2),center,scale),device=device);n=torch.tensor(_standardize(np.asarray(control_x),center,scale),device=device)
            a,ref_a=_augment_ar(a,generator,c);p,ref_p=_augment_ar(p,generator,c);n,_=_augment_ar(n,generator,c);za=model.encode(a);zp=model.encode(p);zn=model.encode(n)
            logits=torch.cat([za@zp.T,za@zn.T],dim=1)/temperature;target=torch.arange(len(participants),device=device);contrastive=.5*(F.cross_entropy(logits,target)+F.cross_entropy(zp@za.T/temperature,target))
            session_logits_a,artifact_logits_a,reference_logits_a=model.nuisance(za);session_logits_p,artifact_logits_p,reference_logits_p=model.nuisance(zp)
            nuisance=(F.cross_entropy(session_logits_a,torch.tensor(s1,device=device))+F.cross_entropy(session_logits_p,torch.tensor(s2,device=device)))/2+(F.cross_entropy(artifact_logits_a,torch.tensor(bins1,device=device))+F.cross_entropy(artifact_logits_p,torch.tensor(bins2,device=device)))/2+(F.cross_entropy(reference_logits_a,ref_a)+F.cross_entropy(reference_logits_p,ref_p))/2
            loss=contrastive+nuisance_weight*nuisance;optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();local.append(float(loss.detach()))
        losses.append(float(np.mean(local)))
    output=_root(c)/"server_checkpoints"/f"fold_{fold:02d}";output.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"center":center,"scale":scale,"artifact_cuts":cuts,"training":training,"evaluation":evaluation,"parameters":parameters,"losses":losses},output/"verifier_a_r.pt")
    result={"status":"VERIFIER_A_R_TRAINED","fold":fold,"parameters":parameters,"loss_initial":losses[0],"loss_final":losses[-1],"cross_day_positive":True,"controls_positive_pairs":0,"nuisance_weight_each":nuisance_weight};_json(run_dir/"result_summary.json",result);return result


def _load_a_r(c:Mapping[str,Any],fold:int,device:torch.device)->tuple[BrainprintVerifierAR,np.ndarray,np.ndarray,np.ndarray]:
    payload=torch.load(_root(c)/"server_checkpoints"/f"fold_{fold:02d}"/"verifier_a_r.pt",map_location=device,weights_only=False);model=BrainprintVerifierAR(57,int(c["verifier_a_r"]["embedding_dim"]),_adjacency(c)).to(device);model.load_state_dict(payload["model"]);model.eval();return model,np.asarray(payload["center"]),np.asarray(payload["scale"]),np.asarray(payload["artifact_cuts"])


@torch.no_grad()
def _embed_a_r(model:BrainprintVerifierAR,x:np.ndarray,center:np.ndarray,scale:np.ndarray,device:torch.device)->np.ndarray:
    values=[];standard=_standardize(x,center,scale)
    for start in range(0,len(standard),128): values.append(model.encode(torch.tensor(standard[start:start+128],device=device)).cpu().numpy())
    return np.concatenate(values)


def _counterfactual(x:np.ndarray,kind:str,seed:int,qc:np.ndarray|None=None)->np.ndarray:
    output=np.asarray(x,np.float32).copy();rng=np.random.default_rng(seed);channels=output.shape[1]
    if kind=="normal": return output
    if kind=="rereference":
        for index in range(len(output)):
            reference=int(rng.integers(0,channels));output[index]-=output[index,reference:reference+1]
        return output
    if kind.startswith("dropout"):
        fraction=float(kind.replace("dropout",""))/100
        for index in range(len(output)): output[index,rng.choice(channels,max(1,int(round(fraction*channels))),replace=False)]=0
        return output
    if kind=="gain_normalized":
        rms=np.sqrt(np.mean(output**2,axis=(1,2),keepdims=True));return output/np.maximum(rms,1e-8)
    if kind=="statistics_only":
        if qc is None: raise ValueError("QC required")
        energy=np.asarray(qc[:,:57],np.float32);energy-=energy.mean(1,keepdims=True);energy/=np.maximum(np.std(energy,axis=1,keepdims=True),1e-8);return np.repeat(energy[...,None],output.shape[-1],axis=-1)
    raise ValueError(kind)


def _outer_threshold(embeddings:dict[tuple[int,int,str],np.ndarray],training:list[int])->dict[int,float]:
    thresholds={}
    for condition in (1,2):
        negative=[]
        for participant in training:
            query=embeddings[(participant,condition,"R")];gallery=np.stack([embeddings[(other,condition,"G")].mean(0) for other in training if other!=participant]);negative.extend((query@gallery.T).reshape(-1))
        thresholds[condition]=float(np.quantile(negative,.95))
    return thresholds


def _session_probe(z:np.ndarray,session:np.ndarray,groups:np.ndarray,seed:int)->tuple[float,float]:
    predicted=np.empty(len(z),int);permuted=np.random.default_rng(seed).permutation(session);predicted_perm=np.empty(len(z),int)
    for train,test in GroupKFold(4).split(z,session,groups):
        predicted[test]=LogisticRegression(max_iter=1000,random_state=seed).fit(z[train],session[train]).predict(z[test]);predicted_perm[test]=LogisticRegression(max_iter=1000,random_state=seed).fit(z[train],permuted[train]).predict(z[test])
    return float(balanced_accuracy_score(session,predicted)),float(balanced_accuracy_score(permuted,predicted_perm))


def evaluate_m1r_fold(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    training,evaluation=fold_members(fold);device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model_a,center,scale,cuts=_load_a_r(c,fold,device);model_b=VerifierBR.load(_root(c)/"server_checkpoints"/f"fold_{fold:02d}"/"verifier_b_r.npz")
    normal={};artifact={};qc={};conditions={}
    for participant in training+evaluation:
        for role in ("R","T","G"):
            x,y,_=load_role(c,participant,role);a,ay,q=load_role(c,participant,role,"artifact")
            if not np.array_equal(y,ay): raise RuntimeError("view labels diverged")
            normal[(participant,role)]=x;artifact[(participant,role)]=a;qc[(participant,role)]=q;conditions[(participant,role)]=y
    subject_rows=[];nuisance=[]
    for name,embed in (("A-R",lambda x:_embed_a_r(model_a,x,center,scale,device)),("B-R",lambda x:model_b.embed(x,c))):
        normal_embeddings={(p,condition,role):embed(normal[(p,role)][conditions[(p,role)]==condition]) for p in training+evaluation for role in ("R","T","G") for condition in (1,2)}
        threshold=_outer_threshold(normal_embeddings,training)
        transforms={"physiological":{},"artifact_acquisition_only":{},"rereference":{},"dropout10":{},"dropout20":{},"gain_normalized":{},"statistics_only":{}}
        for participant in evaluation:
            for role in ("R","G"):
                y=conditions[(participant,role)]
                for condition in (1,2):
                    keep=y==condition;base=normal[(participant,role)][keep];artifact_base=artifact[(participant,role)][keep];local_qc=qc[(participant,role)][keep]
                    transforms["physiological"][(participant,condition,role)]=normal_embeddings[(participant,condition,role)]
                    transforms["artifact_acquisition_only"][(participant,condition,role)]=embed(artifact_base)
                    for kind in ("rereference","dropout10","dropout20","gain_normalized","statistics_only"):
                        value=_counterfactual(base,kind,int(c["training_seed"])+fold*10000+participant*100+condition*10+(0 if role=="R" else 1),local_qc)
                        transforms[kind][(participant,condition,role)]=embed(value)
        panel_rows={};panel_scores={}
        for panel,values in transforms.items():
            query={(p,condition):values[(p,condition,"R")] for p in evaluation for condition in (1,2)};gallery={(p,condition):values[(p,condition,"G")].mean(0) for p in evaluation for condition in (1,2)};rows,scores,labels=_pair_rows(query,gallery,threshold if panel=="physiological" else None);panel_rows[panel]=rows;panel_scores[panel]=(scores,labels)
            for row in rows: subject_rows.append({"fold":fold,"verifier":name,"panel":panel,**row})
        # Same-subject wrong-condition counterfactual.
        query={(p,condition):transforms["physiological"][(p,condition,"R")] for p in evaluation for condition in (1,2)};wrong_gallery={(p,condition):transforms["physiological"][(p,3-condition,"G")].mean(0) for p in evaluation for condition in (1,2)};wrong_rows,_,_=_pair_rows(query,wrong_gallery)
        for row in wrong_rows: subject_rows.append({"fold":fold,"verifier":name,"panel":"wrong_condition_same_subject",**row})
        # Frozen repeated identity permutation null.
        permutation_scores=[];permutation_labels=[];rng=np.random.default_rng(np.random.SeedSequence([int(c["training_seed"]),fold,ord(name[0]),991]))
        query={(p,condition):transforms["physiological"][(p,condition,"R")] for p in evaluation for condition in (1,2)};base_gallery={(p,condition):transforms["physiological"][(p,condition,"G")].mean(0) for p in evaluation for condition in (1,2)}
        for _ in range(64):
            shuffled=rng.permutation(evaluation);gallery={(p,condition):base_gallery[(int(shuffled[evaluation.index(p)]),condition)] for p in evaluation for condition in (1,2)};_,scores,labels=_pair_rows(query,gallery);permutation_scores.extend(scores);permutation_labels.extend(labels)
        # Session probe uses outer participants and frozen physiological embeddings.
        z=[];session=[];groups=[]
        for participant in training:
            for session_index,role in enumerate(("R","T","G")):
                for condition in (1,2):
                    value=normal_embeddings[(participant,condition,role)];z.extend(value);session.extend([session_index]*len(value));groups.extend([participant]*len(value))
        session_acc,session_perm=_session_probe(np.asarray(z),np.asarray(session),np.asarray(groups),int(c["training_seed"])+fold)
        normal_scores,normal_labels=panel_scores["physiological"]
        nuisance.extend([
            {"fold":fold,"verifier":name,"metric":"eer","value":_eer(normal_scores,normal_labels)},
            {"fold":fold,"verifier":name,"metric":"identity_permutation_auc","value":roc_auc_score(permutation_labels,permutation_scores)},
            {"fold":fold,"verifier":name,"metric":"session_balanced_accuracy","value":session_acc},
            {"fold":fold,"verifier":name,"metric":"session_label_permutation_balanced_accuracy","value":session_perm},
            {"fold":fold,"verifier":name,"metric":"artifact_energy_cut_1","value":float(cuts[0])},{"fold":fold,"verifier":name,"metric":"artifact_energy_cut_2","value":float(cuts[1])},
        ])
    output=_root(c)/"m1r"/f"fold_{fold:02d}";_csv(output/"subject_metrics.csv",subject_rows);_csv(output/"nuisance_metrics.csv",nuisance);result={"status":"M1R_FOLD_EVALUATED","fold":fold,"evaluation":evaluation,"day200_opened":False,"controls_positive_pairs":0,"verifier_b_r_training_import":False};_json(run_dir/"result_summary.json",result);return result


def aggregate_m1r(c:Mapping[str,Any])->dict[str,Any]:
    subject=[];nuisance=[]
    for fold in range(5): subject.extend(_read_csv(_root(c)/"m1r"/f"fold_{fold:02d}"/"subject_metrics.csv"));nuisance.extend(_read_csv(_root(c)/"m1r"/f"fold_{fold:02d}"/"nuisance_metrics.csv"))
    participant_rows=[];decisions={}
    for name in ("A-R","B-R"):
        selected=[row for row in subject if row["verifier"]==name];panels=sorted(set(row["panel"] for row in selected));by_panel={}
        for panel in panels:
            rows=[]
            for participant in range(1,16):
                local=[r for r in selected if r["panel"]==panel and int(r["participant"])==participant]
                rows.append({"verifier":name,"panel":panel,"participant":participant,"auroc":mean(float(r["auroc"]) for r in local),"identity_margin":mean(float(r["identity_margin"]) for r in local),"rank1":mean(float(r["rank1"]) for r in local),"tar_at_far5":mean(float(r["tar_at_far5"]) for r in local)})
            participant_rows.extend(rows);by_panel[panel]=rows
        normal=by_panel["physiological"];artifact=by_panel["artifact_acquisition_only"];reref=by_panel["rereference"];drop10=by_panel["dropout10"];drop20=by_panel["dropout20"];wrong=by_panel["wrong_condition_same_subject"]
        aurocs={r["participant"]:r["auroc"] for r in normal};ci=_bootstrap(aurocs,int(c["bootstrap_seed"])+(0 if name=="A-R" else 1),int(c["bootstrap_replicates"]));local_n=[r for r in nuisance if r["verifier"]==name];metric=lambda key:mean(float(r["value"]) for r in local_n if r["metric"]==key)
        normal_auc=mean(r["auroc"] for r in normal);artifact_auc=mean(r["auroc"] for r in artifact);normal_margin=mean(r["identity_margin"] for r in normal);wrong_margin=mean(r["identity_margin"] for r in wrong);loo=min(mean(v for p,v in aurocs.items() if p!=left) for left in aurocs)
        decisions[name]={"physiological_auroc":normal_auc,"participant_bootstrap_ci":ci,"eer":metric("eer"),"tar_at_far5":mean(r["tar_at_far5"] for r in normal),"rank1":mean(r["rank1"] for r in normal),"positive":sum(r["identity_margin"]>0 for r in normal),"artifact_auroc":artifact_auc,"physiological_minus_artifact":normal_auc-artifact_auc,"identity_permutation_auc":metric("identity_permutation_auc"),"session_balanced_accuracy":metric("session_balanced_accuracy"),"session_permutation_accuracy":metric("session_label_permutation_balanced_accuracy"),"rereference_auroc":mean(r["auroc"] for r in reref),"dropout10_auroc":mean(r["auroc"] for r in drop10),"dropout20_auroc":mean(r["auroc"] for r in drop20),"wrong_condition_margin":wrong_margin,"correct_condition_margin":normal_margin,"wrong_condition_fraction":wrong_margin/max(normal_margin,1e-8),"leave_one_participant_out_auroc_min":loo}
    a={r["participant"]:r["identity_margin"] for r in participant_rows if r["verifier"]=="A-R" and r["panel"]=="physiological"};b={r["participant"]:r["identity_margin"] for r in participant_rows if r["verifier"]=="B-R" and r["panel"]=="physiological"};direction=sum((a[p]>0)==(b[p]>0) for p in a)
    gate=c["m1r_gate"];criteria=[]
    for name,d in decisions.items():
        criteria.extend([(f"{name}_auroc",d["physiological_auroc"]>=float(gate["auroc_min"])),(f"{name}_ci_low",d["participant_bootstrap_ci"][0]>=float(gate["ci_low_min"])),(f"{name}_eer",d["eer"]<=float(gate["eer_max"])),(f"{name}_tar",d["tar_at_far5"]>=float(gate["tar_far5_min"])),(f"{name}_positive",d["positive"]>=int(gate["positive_participants_min"])),(f"{name}_artifact",d["artifact_auroc"]<=float(gate["artifact_auroc_max"])),(f"{name}_gap",d["physiological_minus_artifact"]>=float(gate["physiological_minus_artifact_min"])),(f"{name}_identity_permutation",float(gate["permutation_auc_range"][0])<=d["identity_permutation_auc"]<=float(gate["permutation_auc_range"][1])),(f"{name}_session",d["session_balanced_accuracy"]<=1/3+float(gate["session_probe_margin"])),(f"{name}_rereference",d["rereference_auroc"]>=d["physiological_auroc"]-float(gate["rereference_drop_max"])),(f"{name}_dropout10",d["dropout10_auroc"]>=d["physiological_auroc"]-float(gate["dropout10_drop_max"])),(f"{name}_wrong_condition",d["wrong_condition_fraction"]<=float(gate["wrong_condition_fraction_max"])),(f"{name}_loo",d["leave_one_participant_out_auroc_min"]>=float(gate["loo_auroc_min"]))])
    criteria.append(("participant_direction_agreement",direction>=int(gate["direction_agreement_min"])));passed=all(value for _,value in criteria)
    _csv(_root(c)/"m1r_verifier_subject_metrics.csv",participant_rows);_csv(_root(c)/"m1r_nuisance_probe_metrics.csv",nuisance);result={"status":"M1R_PHYSIOLOGICAL_VERIFIER_VALID" if passed else "M1R_PHYSIOLOGICAL_VERIFIER_FAIL","PASS":passed,"verifiers":decisions,"participant_direction_agreement":direction,"failed_criteria":[name for name,value in criteria if not value],"participants":15,"controls_positive_pairs":0,"day200_opened":False};_json(_root(c)/"m1r_decision.json",result);return result


def future_preregistration()->dict[str,Any]:
    return {"name":"BrainID Bridge post-Gate-01R preregistration","execution_authorized":False,"methods":["population degraded-to-clean direct bridge (K1 primary)","parameter/update-matched DET-noRef","DET-Ref","CacheKV-MATCH","CacheKV-POP","CacheKV-WRONG"],"staging":["CacheKV without physiological loss","timestep-scaled physiological loss only after CacheKV adjudication"],"support_memory":"shared denoiser tower once; cache multi-layer K/V","forbidden":["single 64-D token compression","old raw-support cross-attention"],"verifiers":{"A-R":"training constraint allowed","B-R":"evaluation only, never training or selection"},"sampling":{"primary":"K1","secondary":"K8 versus DET ensemble"},"success":"physiological-reference utility beats POP/WRONG with noninferior waveform, ERP task, and observation fidelity"}


def gate01r(c:Mapping[str,Any])->dict[str,Any]:
    original=_load_json(Path(c["immutable_gate01"]));data=_load_json(_root(c)/"data_protocol_decision.json");m1=_load_json(_root(c)/"m1r_decision.json") if (_root(c)/"m1r_decision.json").exists() else None;m0=_load_json(_root(c)/"m0r_decision.json") if (_root(c)/"m0r_decision.json").exists() else None
    original_ok=original.get("PASS_01") is False and original.get("M1_verifier")=="FAIL";passed=bool(original_ok and data.get("PASS") and m1 and m1.get("PASS") and m0 and m0.get("PASS"));failed=[]
    if not data.get("PASS"): failed.extend(f"DATA:{v}" for v in data.get("failed_criteria",[data.get("status")]))
    if not m1 or not m1.get("PASS"): failed.extend(f"M1R:{v}" for v in (m1.get("failed_criteria",[]) if m1 else ["not_run"]))
    if not m0 or not m0.get("PASS"): failed.extend(f"M0R:{v}" for v in (m0.get("failed_criteria",[]) if m0 else ["not_run_after_M1R_failure"]))
    decision={"original_gate01":"FAIL_IMMUTABLE","data_protocol":"PASS" if data.get("PASS") else ("INSUFFICIENT" if "INSUFFICIENT" in data.get("status","") else "FAIL"),"M1R_verifier":"PASS" if m1 and m1.get("PASS") else ("FAIL" if m1 else "INSUFFICIENT"),"M0R_actionability":"PASS" if m0 and m0.get("PASS") else ("FAIL" if m0 else "NOT_RUN"),"PASS_01R":passed,"failed_criteria":failed,"controls_opened_for_negative_calibration":True,"day200_opened":False,"physiomotion_sealed_opened":False,"denoiser_or_diffusion_trained":False}
    _json(_root(c)/"gate01r_decision.json",decision)
    if passed: (_root(c)/"future_brainid_bridge_preregistration.yaml").write_text(yaml.safe_dump(future_preregistration(),sort_keys=False),encoding="utf-8")
    return decision


def write_report(c:Mapping[str,Any])->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=_root(c);data=_load_json(root/"data_protocol_decision.json");m1=_load_json(root/"m1r_decision.json") if (root/"m1r_decision.json").exists() else None;m0=_load_json(root/"m0r_decision.json") if (root/"m0r_decision.json").exists() else None;gate=_load_json(root/"gate01r_decision.json")
    figures=root/"figures";figures.mkdir(parents=True,exist_ok=True)
    if m1:
        rows=_read_csv(root/"m1r_verifier_subject_metrics.csv");fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for axis,name in zip(axes,("A-R","B-R")):
            normal=[r for r in rows if r["verifier"]==name and r["panel"]=="physiological"];artifact=[r for r in rows if r["verifier"]==name and r["panel"]=="artifact_acquisition_only"];ids=[int(r["participant"]) for r in normal];axis.axhline(0,color="black",linewidth=.8);axis.plot(ids,[float(r["identity_margin"]) for r in normal],"o-",label="physiological");axis.plot(ids,[float(r["identity_margin"]) for r in artifact],"s--",label="artifact-only");axis.set(title=name,xlabel="held-out participant",xticks=ids);axis.legend(frameon=False)
        axes[0].set_ylabel("R→G identity margin");fig.tight_layout();fig.savefig(figures/"m1r_construct_validity.png",dpi=180);plt.close(fig)
    report=Path(__file__).parents[3]/"reports"/"brainid_gate_v17r.md";lines=["# BrainID Gate-01R","","One-shot nuisance-invariant construct-validity repair. The immutable Gate-01 result remains FAIL; no denoiser, diffusion, I²SB, CacheKV, or identity-guided model was trained.","","## Data protocol","",f"- `{data['status']}`: {data['participants_with_RTG']}/15 longitudinal participants, {data['common_channels']} channels, {data['controls_prepared']}/52 anonymous controls used only as negatives/nuisance calibration.","- R/T/G remained Day-1/Day-7/Day-80; Day-200 and PhysioMotion sealed signals remained unopened.","","## J1 identity-carrier forensic","",f"- Descriptive panel report: `{Path(__file__).parents[3]/'reports'/'j1_forensic_report.md'}`. No panel was selected; J2 stayed frozen to 0.5–30 Hz.","","## M1R physiological verifier"," "]
    if m1:
        for name,d in m1["verifiers"].items(): lines.append(f"- {name}: physiological AUROC {d['physiological_auroc']:.4f} (descriptive CI {d['participant_bootstrap_ci'][0]:.4f}–{d['participant_bootstrap_ci'][1]:.4f}), artifact-only AUROC {d['artifact_auroc']:.4f}, gap {d['physiological_minus_artifact']:.4f}, EER {d['eer']:.4f}, TAR@FAR5 {d['tar_at_far5']:.4f}, positive {d['positive']}/15.")
        lines.append(f"- Decision: `{m1['status']}`; failed criteria: {', '.join(m1['failed_criteria']) or 'none'}.")
    else: lines.append("- Not run because upstream protocol was insufficient.")
    lines += ["","## M0R no-training actionability","",("- "+m0["status"] if m0 else "- NOT_RUN because M1R failed; no restoration model or analytical reference intervention was run."),"","## Gate-01R decision","",f"```json\n{json.dumps(gate,indent=2,sort_keys=True)}\n```","","This is development evidence. Any failure constrains only the frozen Gate-01R construct/actionability instance and is not a family-wide negative."]
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text("\n".join(lines)+"\n",encoding="utf-8");result={"status":"REPORT_WRITTEN","PASS_01R":gate["PASS_01R"],"report":str(report),"figures":sorted(str(p) for p in figures.glob("*.png"))};_json(root/"result_summary.json",result);return result
