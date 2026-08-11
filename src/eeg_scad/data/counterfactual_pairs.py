from __future__ import annotations

import csv
import hashlib
import itertools
from pathlib import Path
from typing import Any,Mapping,Sequence

import numpy as np

from eeg_scad.context.operator_normalization import canonical_operator_features,canonicalize_operator,robust_scale


def _sha_array(value:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _write_csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for row in rows for k in row})
    with path.open("w",encoding="utf-8",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _prepared(root:Path,participant:str,session:str,task:str)->Path:return root/"prepared"/participant/f"{session}_{task}.npz"
def _raw_operator(root:Path,participant:str,session:str,task:str)->Path:return root/"operators/raw"/participant/f"{session}_{task}.npz"
def _query_operator(root:Path,participant:str,session:str,task:str)->Path:return root/"evaluator"/participant/f"{session}_{task}.npz"


def _load_signal(root:Path,p:str,s:str,t:str)->tuple[np.ndarray,np.ndarray]:
    with np.load(_prepared(root,p,s,t),allow_pickle=False) as z:return np.asarray(z["eeg"],np.float64),np.asarray(z["eog"],np.float64)


def _load_raw(root:Path,p:str,s:str,t:str,tasks:Sequence[str])->tuple[dict[str,np.ndarray],bool]:
    path=_raw_operator(root,p,s,t);fallback=False
    if not path.is_file():path=_raw_operator(root,p,s,next(v for v in tasks if v!=t));fallback=True
    if not path.is_file():raise FileNotFoundError(path)
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k],np.float64) for k in z.files},fallback


def fold_eeg_scale(data:Mapping[str,Any],train:Sequence[str])->np.ndarray:
    root=Path(data["v19_derived_root"]);values=[]
    for p,s,t in itertools.product(train,data["sessions"],data["tasks"]):
        if not _prepared(root,p,s,t).is_file():continue
        eeg,_=_load_signal(root,p,s,t);q=eeg[:,int(data["qnatural_start"]):];_,scale=robust_scale(q);values.append(scale)
    if not values:raise RuntimeError("no fold normalization data")
    return np.maximum(np.median(np.stack(values),axis=0),1e-3)


def _owner_B(data:Mapping[str,Any],owner:str,session:str,task:str,eeg_scale:np.ndarray)->tuple[np.ndarray,np.ndarray,bool]:
    root=Path(data["v19_derived_root"]);raw,fallback=_load_raw(root,owner,session,task,data["tasks"]);_,eog=_load_signal(root,owner,session,task if not fallback else next(v for v in data["tasks"] if v!=task));_,escale=robust_scale(eog[:,:int(data["support_samples"])]);B=canonicalize_operator(raw["C_raw"],escale,eeg_scale)
    blocks=np.stack([canonicalize_operator(c,escale,eeg_scale) for c in raw["C_blocks"]]);return B,blocks,fallback


def context_package(data:Mapping[str,Any],fold:Mapping[str,Any],recipient:str,owner:str|None,session:str,task:str,eeg_scale:np.ndarray)->tuple[np.ndarray,np.ndarray,str,bool]:
    train=list(fold["train"]);train_B=[];fallback=False
    for p in train:
        b,_,fb=_owner_B(data,p,session,task,eeg_scale);train_B.append(b);fallback|=fb
    pop=np.mean(train_B,axis=0);tau=float(np.mean(np.square(np.stack(train_B)-pop[None])))
    if owner is None:return pop,canonical_operator_features(pop),"POP",fallback
    B,blocks,fb=_owner_B(data,owner,session,task,eeg_scale);within=float(np.mean(np.square(blocks-B[None])));alpha=float(np.clip(tau/max(tau+within/4,1e-12),0,1));shrunk=pop+alpha*(B-pop)
    return shrunk,canonical_operator_features(shrunk),owner,fb


def _windows(eeg:np.ndarray,eog:np.ndarray,start:int,length:int)->tuple[list[np.ndarray],list[np.ndarray],np.ndarray]:
    x=eeg[:,start:];e=eog[:,start:];count=min(x.shape[1],e.shape[1])//length;x=x[:,:count*length].reshape(x.shape[0],count,length).transpose(1,0,2);e=e[:,:count*length].reshape(e.shape[0],count,length).transpose(1,0,2);energy=np.sqrt(np.mean(e*e,axis=(1,2)));return list(x),list(e),energy


def _candidate(data:Mapping[str,Any],p:str,s:str,t:str)->tuple[list[np.ndarray],list[np.ndarray],np.ndarray]:
    root=Path(data["v19_derived_root"]);eeg,eog=_load_signal(root,p,s,t);return _windows(eeg,eog,int(data["qnatural_start"]),int(data["window_samples"]))


def _records(data:Mapping[str,Any],fold:Mapping[str,Any],split:str,eeg_scale:np.ndarray,seed:int)->tuple[dict[str,np.ndarray],list[dict[str,Any]]]:
    root=Path(data["v19_derived_root"]);recipients=list(fold[split]);train=list(fold["train"]);rng=np.random.Generator(np.random.PCG64DXSM(seed+int(fold["fold"])*100+{"train":0,"validation":1,"test":2}[split]));arrays={k:[] for k in ("x","y","artifact","context_match","context_pop","context_wrong")};meta=[]
    for recipient,session,task in itertools.product(recipients,data["sessions"],data["tasks"]):
        qp=_query_operator(root,recipient,session,task)
        if not qp.is_file():continue
        with np.load(qp,allow_pickle=False) as z:cquery=np.asarray(z["C_query"],np.float64)
        _,ctx_match,_,fm=context_package(data,fold,recipient,recipient,session,task,eeg_scale);_,ctx_pop,_,fp=context_package(data,fold,recipient,None,session,task,eeg_scale)
        wrong_pool=[p for p in (recipients if split!="train" else train) if p!=recipient]
        if not wrong_pool:wrong_pool=[str(data["auxiliary_support_owner"])]
        candidates=[p for p in train if p!=recipient]
        if len(candidates)<2:continue
        cache={p:_candidate(data,p,session,task) for p in candidates}
        count=int(data["paired_windows_per_unit"])
        for index in range(count):
            xowner=candidates[index%len(candidates)];eowner=candidates[(index+1)%len(candidates)]
            if eowner==xowner:eowner=candidates[(index+2)%len(candidates)]
            xw,_,xe=cache[xowner];_,ew,ee=cache[eowner];xi=np.flatnonzero(xe<=np.quantile(xe,float(data["low_eog_quantile"])));ei=np.flatnonzero(ee>=np.quantile(ee,float(data["high_eog_quantile"])))
            if not len(xi) or not len(ei):continue
            x=np.asarray(xw[int(xi[index%len(xi)])],np.float64);e=np.asarray(ew[int(ei[index%len(ei)])],np.float64);e=e-np.mean(e,axis=1,keepdims=True);temporal=np.sqrt(np.mean(e*e,axis=0));mask=temporal>=np.quantile(temporal,float(data["artifact_mask_quantile"]));artifact=cquery@(e*mask[None])
            zero=bool(rng.random()<.10);artifact=np.zeros_like(artifact) if zero else artifact;y=x+artifact;x=x/eeg_scale[:,None];y=y/eeg_scale[:,None];artifact=artifact/eeg_scale[:,None]
            wrong_owner=wrong_pool[index%len(wrong_pool)];_,ctx_wrong,_,fw=context_package(data,fold,recipient,wrong_owner,session,task,eeg_scale)
            for key,value in (("x",x),("y",y),("artifact",artifact),("context_match",ctx_match),("context_pop",ctx_pop),("context_wrong",ctx_wrong)):arrays[key].append(np.asarray(value,np.float32))
            meta.append({"fold":fold["fold"],"split":split,"participant":recipient,"session":session,"task":task,"pair":index,"x_owner":xowner,"eog_owner":eowner,"operator_recipient":recipient,"wrong_owner":wrong_owner,"three_way_disjoint":int(len({recipient,xowner,eowner})==3),"zero_artifact":int(zero),"common_xy_hash":_sha_array(np.concatenate((x,y,artifact))) ,"support_query_disjoint":1,"query_operator_evaluator_only":1,"fallback":int(fm or fp or fw)})
    return {k:np.stack(v) for k,v in arrays.items()},meta


def _natural(data:Mapping[str,Any],fold:Mapping[str,Any],eeg_scale:np.ndarray)->tuple[dict[str,np.ndarray],dict[str,np.ndarray],list[dict[str,Any]]]:
    root=Path(data["v19_derived_root"]);inf={k:[] for k in ("y","context_match","context_pop","context_wrong")};ev={k:[] for k in ("eog","C_query")};meta=[]
    for recipient,session,task in itertools.product(fold["test"],data["sessions"],data["tasks"]):
        if not _prepared(root,recipient,session,task).is_file() or not _query_operator(root,recipient,session,task).is_file():continue
        eeg,eog=_load_signal(root,recipient,session,task);yw,ew,energy=_windows(eeg,eog,int(data["qnatural_start"]),int(data["window_samples"]));indices=np.linspace(0,len(yw)-1,min(int(data["natural_windows_per_unit"]),len(yw)),dtype=int)
        _,cm,_,_=context_package(data,fold,recipient,recipient,session,task,eeg_scale);_,cp,_,_=context_package(data,fold,recipient,None,session,task,eeg_scale);wrong_pool=[p for p in fold["test"] if p!=recipient];wrong=wrong_pool[0];_,cw,_,_=context_package(data,fold,recipient,wrong,session,task,eeg_scale)
        with np.load(_query_operator(root,recipient,session,task),allow_pickle=False) as z:cq=np.asarray(z["C_query"],np.float32)
        for index in indices:
            inf["y"].append((np.asarray(yw[index])/eeg_scale[:,None]).astype(np.float32));inf["context_match"].append(cm);inf["context_pop"].append(cp);inf["context_wrong"].append(cw);ev["eog"].append(np.asarray(ew[index],np.float32));ev["C_query"].append(cq)
            meta.append({"fold":fold["fold"],"participant":recipient,"session":session,"task":task,"window":int(index),"wrong_owner":wrong,"query_eog_in_inference":0,"query_operator_in_inference":0})
    return {k:np.stack(v) for k,v in inf.items()},{k:np.stack(v) for k,v in ev.items()},meta


def build_fold_assets(data:Mapping[str,Any],fold:Mapping[str,Any],derived:Path,result:Path)->dict[str,Any]:
    fold_id=int(fold["fold"]);scale=fold_eeg_scale(data,fold["train"]);target=derived/f"fold_{fold_id}";target.mkdir(parents=True,exist_ok=True);np.save(target/"eeg_scale.npy",scale)
    all_meta=[]
    for split in ("train","validation","test"):
        arrays,meta=_records(data,fold,split,scale,20260808);all_meta+=meta
        if split=="test":
            np.savez_compressed(target/"paired_test_inference.npz",y=arrays["y"],context_match=arrays["context_match"],context_pop=arrays["context_pop"],context_wrong=arrays["context_wrong"])
            np.savez_compressed(target/"paired_test_evaluator.npz",x=arrays["x"],artifact=arrays["artifact"])
        else:np.savez_compressed(target/f"paired_{split}.npz",**arrays)
        _write_csv(result/f"fold_{fold_id}_{split}_roles.csv",meta)
    ni,ne,nm=_natural(data,fold,scale);np.savez_compressed(target/"natural_inference.npz",**ni);np.savez_compressed(target/"natural_evaluator.npz",**ne);_write_csv(result/f"fold_{fold_id}_natural_roles.csv",nm)
    return {"fold":fold_id,"train_samples":sum(1 for r in all_meta if r["split"]=="train"),"validation_samples":sum(1 for r in all_meta if r["split"]=="validation"),"test_samples":sum(1 for r in all_meta if r["split"]=="test"),"natural_samples":len(nm),"sealed_reads":0,"three_way_primary_fraction":float(np.mean([r["three_way_disjoint"] for r in all_meta]))}


def load_training_split(derived:Path,fold:int,split:str)->dict[str,np.ndarray]:
    with np.load(derived/f"fold_{fold}"/f"paired_{split}.npz",allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

