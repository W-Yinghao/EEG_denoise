"""Online counterfactual mixtures for V23.

Roles are sampled for every minibatch. The clean carrier, EOG waveform and
operator recipient are distinct whenever the frozen fold permits it. Query
operators generate targets but are never returned as model inputs.
"""
from __future__ import annotations
import hashlib,itertools
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
from eeg_scad.context.operator_normalization import canonicalize_operator,canonical_operator_features,robust_scale
from eeg_scad.context.operator_factorization import factorize_operator,population_basis,operator_summary
from eeg_scad.context.projection_features import project_numpy,ridge_target_numpy
from eeg_scad.data.counterfactual_pairs import _load_signal,_owner_B,_query_operator,fold_eeg_scale


def _hash(value:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


class OnlineCounterfactualSampler:
    def __init__(self,data:Mapping[str,Any],fold:Mapping[str,Any],split:str,seed:int,coef_ratio:float=.001,q_ratio:float=.01)->None:
        self.data=data;self.fold=fold;self.split=split;self.seed=seed;self.rng=np.random.Generator(np.random.PCG64DXSM(seed+int(fold["fold"])*1000+{"train":0,"validation":1,"test":2}[split]));self.coef_ratio=coef_ratio;self.q_ratio=q_ratio
        self.participants=list(fold[split]);self.training=list(fold["train"]);self.root=Path(data["v19_derived_root"]);self.eeg_scale=fold_eeg_scale(data,self.training);self.records=[];self.operators={};self.pop={}
        available=set(self.training+self.participants+[str(data["auxiliary_support_owner"])])
        for session,task in itertools.product(data["sessions"],data["tasks"]):
            train_b=[]
            for owner in self.training:
                try:b,_,_=_owner_B(data,owner,session,task,self.eeg_scale);train_b.append(b);self.operators[(owner,session,task)]=b
                except FileNotFoundError:continue
            if not train_b:continue
            self.pop[(session,task)]=np.mean(train_b,axis=0)
            for owner in available:
                try:b,_,_=_owner_B(data,owner,session,task,self.eeg_scale);self.operators[(owner,session,task)]=b
                except FileNotFoundError:pass
            for recipient in self.participants:
                qp=_query_operator(self.root,recipient,session,task)
                if not qp.is_file() or (recipient,session,task) not in self.operators:continue
                eeg,eog=_load_signal(self.root,recipient,session,task);_,support_scale=robust_scale(eog[:,:int(data["support_samples"])]);cq_raw=np.load(qp,allow_pickle=False)["C_query"];cq=canonicalize_operator(cq_raw,support_scale,self.eeg_scale)
                self.records.append((recipient,session,task,cq,np.asarray(cq_raw,np.float64)))
        if not self.records:raise RuntimeError(f"no online records for fold {fold['fold']} {split}")
        self.source_records=[]
        for owner,session,task in itertools.product(self.training,data["sessions"],data["tasks"]):
            try:eeg,eog=_load_signal(self.root,owner,session,task)
            except FileNotFoundError:continue
            if eeg.shape[1]>int(data["qnatural_start"])+int(data["window_samples"]):self.source_records.append((owner,session,task,eeg,eog))

    def state(self)->dict[str,Any]:return self.rng.bit_generator.state
    def set_state(self,state:dict[str,Any])->None:self.rng.bit_generator.state=state
    def _source(self,exclude:set[str],session:str,task:str)->tuple[str,np.ndarray,np.ndarray]:
        cells=[v for v in self.source_records if v[0] not in exclude and v[1]==session and v[2]==task]
        if not cells:cells=[v for v in self.source_records if v[0] not in exclude]
        value=cells[int(self.rng.integers(len(cells)))];return value[0],value[3],value[4]
    def _window(self,value:np.ndarray)->np.ndarray:
        length=int(self.data["window_samples"]);lo=int(self.data["qnatural_start"]);start=int(self.rng.integers(lo,max(lo+1,value.shape[1]-length+1)));return np.asarray(value[:,start:start+length],np.float64)
    def sample(self,batch_size:int,zero_proportion:float=.10,pop_consistent:bool=False)->dict[str,np.ndarray]:
        arrays={k:[] for k in ("x","y","artifact","basis_match","basis_pop","basis_wrong","basis_query","summary_match","summary_pop","summary_wrong","context_match","context_pop","context_wrong","q_match","q_pop","q_wrong","projected_match","projected_pop","projected_wrong","z_match","z_pop","z_wrong","z_query")};meta=[]
        for _ in range(batch_size):
            recipient,session,task,cquery,cquery_raw=self.records[int(self.rng.integers(len(self.records)))];c0=self.pop[(session,task)];cs=self.operators[(recipient,session,task)]
            wrong_pool=[p for p in self.participants+self.training+[str(self.data["auxiliary_support_owner"])] if p!=recipient and (p,session,task) in self.operators];wrong=wrong_pool[int(self.rng.integers(len(wrong_pool)))];cw=self.operators[(wrong,session,task)]
            xowner,xeeg,_=self._source({recipient},session,task);eowner,_,eeog=self._source({recipient,xowner},session,task);x=self._window(xeeg);e=self._window(eeog);e=e-np.mean(e,axis=1,keepdims=True)
            gain=float(self.rng.choice([.35,.7,1.15])*self.rng.uniform(.85,1.15));generating=c0 if pop_consistent else cquery;artifact=(generating@(gain*e))/self.eeg_scale[:,None];zero=bool(self.rng.random()<zero_proportion);artifact=np.zeros_like(artifact) if zero else artifact;x=x/self.eeg_scale[:,None];y=x+artifact
            bm,sm,dm=factorize_operator(c0,cs);bp,sp,dp=population_basis(c0);bw,sw,dw=factorize_operator(c0,cw);bq,sq,dq=factorize_operator(c0,cquery)
            arrays["basis_query"].append(bq)
            arrays["context_match"].append(canonical_operator_features(cs));arrays["context_pop"].append(canonical_operator_features(c0));arrays["context_wrong"].append(canonical_operator_features(cw))
            for label,basis,scales in (("match",bm,sm),("pop",bp,sp),("wrong",bw,sw),("query",bq,sq)):
                z,_,_=ridge_target_numpy(artifact,basis,self.coef_ratio)
                if label!="query":q,projected,_=project_numpy(y,basis,self.q_ratio);arrays[f"basis_{label}"].append(basis);arrays[f"summary_{label}"].append(operator_summary(basis,scales));arrays[f"q_{label}"].append(q);arrays[f"projected_{label}"].append(projected)
                arrays[f"z_{label}"].append(z)
            for key,value in (("x",x),("y",y),("artifact",artifact)):arrays[key].append(value)
            signal=np.linalg.norm(x);noise=np.linalg.norm(artifact);meta.append({"participant":recipient,"session":session,"task":task,"clean_owner":xowner,"eog_owner":eowner,"operator_recipient":recipient,"wrong_owner":wrong,"strict_three_way":int(len({recipient,xowner,eowner})==3),"gain":gain,"input_snr_db":float(20*np.log10(max(signal,1e-8)/max(noise,1e-8))) if not zero else np.nan,"zero_artifact":int(zero),"common_xy_hash":_hash(np.concatenate((x,y,artifact))),"projection_match":dm.deviation_energy_ratio,"projection_wrong":dw.deviation_energy_ratio})
        return {**{k:np.asarray(v,np.float32) for k,v in arrays.items()},"meta":meta}


def generate_validation_bank(sampler:OnlineCounterfactualSampler,samples:int,seed:int)->dict[str,np.ndarray]:
    old=sampler.state();sampler.rng=np.random.Generator(np.random.PCG64DXSM(seed));value=sampler.sample(samples);sampler.set_state(old);return value
