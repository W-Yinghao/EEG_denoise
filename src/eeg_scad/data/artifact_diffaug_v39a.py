"""Leakage-safe V39A artifact targets and exact 30-second support context."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any,Mapping
import numpy as np
import torch

from eeg_scad.context.deepsets_encoder import DeepSetsSupportEncoder
from eeg_scad.data.counterfactual_pairs import _load_signal
from eeg_scad.data.eog_latent_streams import EOGStreamSampler
from eeg_scad.data.v24_coordinate_contract import robust_center_scale


V25_CHECKPOINTS=Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25/checkpoints/det/deepsets")


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def exact_support(data:Mapping[str,Any],fold:Mapping[str,Any],owner:str,session:str,task:str)->tuple[np.ndarray,np.ndarray,list[int],str]:
    root=Path(data["v19_derived_root"]);actual=task
    try:eeg,eog=_load_signal(root,owner,session,task)
    except FileNotFoundError:
        actual=next(value for value in data["tasks"] if value!=task);eeg,eog=_load_signal(root,owner,session,actual)
    prefix=3000;length=200;starts=list(range(0,prefix-length+1,length));assert len(starts)==15
    scale_path=Path(data["v24_derived_root"])/f"fold_{fold['fold']}"/"eeg_scale.npy"
    eeg_scale=np.load(scale_path);center,scale=robust_center_scale(eog[:,:prefix])
    seeg=np.stack([eeg[:,s:s+length]/eeg_scale[:,None] for s in starts]).astype(np.float32)
    seog=np.stack([(eog[:,s:s+length]-center[:,None])/scale[:,None] for s in starts]).astype(np.float32)
    _load_signal.cache_clear()
    return seeg,seog,starts,actual


def load_support_encoder(fold:int,seed:int,device:torch.device)->tuple[DeepSetsSupportEncoder,dict[str,object]]:
    path=V25_CHECKPOINTS/f"fold_{fold}"/f"seed_{seed}"/"best_joint.pt"
    if not path.is_file():raise FileNotFoundError(path)
    payload=torch.load(path,map_location=device,weights_only=True);model=DeepSetsSupportEncoder(rank=8).to(device)
    state=payload["ema"]["shadow"];prefix="support.";model.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    model.eval();[p.requires_grad_(False) for p in model.parameters()]
    return model,{"path":str(path),"sha256":sha256(path),"fold":fold,"seed":seed,"model":"V25_DeepSets_support_encoder"}


@torch.no_grad()
def context_bank(data:Mapping[str,Any],fold:Mapping[str,Any],seed:int,device:torch.device)->tuple[dict[tuple[str,str,str],np.ndarray],list[dict[str,object]],dict[str,object]]:
    encoder,binding=load_support_encoder(int(fold["fold"]),seed,device);result={};rows=[]
    owners=sorted(set(fold["train"]+fold["validation"]+fold["test"]))
    for owner in owners:
        for session in data["sessions"]:
            for task in data["tasks"]:
                try:eeg,eog,starts,actual=exact_support(data,fold,owner,session,task)
                except FileNotFoundError:continue
                encoded=encoder(torch.from_numpy(eeg[None]).to(device),torch.from_numpy(eog[None]).to(device));result[(owner,session,task)]=encoded["context"][0].cpu().numpy()
                rows.append({"fold":fold["fold"],"participant":owner,"session":session,"task":task,"actual_task":actual,"support_seconds":30,"window_count":15,"starts":";".join(map(str,starts)),"overlap_samples":0,"repeated_samples":0,"query_samples":0,"normalization_prefix_seconds":30})
    return result,rows,binding


@torch.no_grad()
def intervention_contexts(data:Mapping[str,Any],fold:Mapping[str,Any],contexts:dict[tuple[str,str,str],np.ndarray],seed:int,device:torch.device)->dict[str,dict[tuple[str,str,str],np.ndarray]]:
    """Registered fold-training population, mean-wrong, and time-shuffled controls."""
    encoder,_=load_support_encoder(int(fold["fold"]),seed,device);population={};wrong={};shuffled={};rng=np.random.Generator(np.random.PCG64DXSM(20260939))
    for owner in fold["test"]:
        for session in data["sessions"]:
            for task in data["tasks"]:
                key=(owner,session,task)
                if key not in contexts:continue
                train_values=[contexts[(other,session,task)] for other in fold["train"] if (other,session,task) in contexts]
                wrong_values=[value for (other,current_session,current_task),value in contexts.items() if other!=owner and current_session==session and current_task==task]
                population[key]=np.mean(train_values,axis=0).astype(np.float32);wrong[key]=np.mean(wrong_values,axis=0).astype(np.float32)
                eeg,eog,_,_=exact_support(data,fold,owner,session,task);permutation=rng.permutation(len(eog));encoded=encoder(torch.from_numpy(eeg[None]).to(device),torch.from_numpy(eog[None,permutation]).to(device));shuffled[key]=encoded["context"][0].cpu().numpy()
    return {"correct":contexts,"population_context":population,"mean_wrong_support":wrong,"registered_shuffled_support":shuffled}


def sample_targets(data:Mapping[str,Any],fold:Mapping[str,Any],split:str,seed:int,paired:int,natural:int,contexts:dict[tuple[str,str,str],np.ndarray])->dict[str,np.ndarray|list[dict[str,object]]]:
    sampler=EOGStreamSampler(data,fold,split,seed);parts=[]
    if paired:parts.append(sampler.sample_paired(paired,zero_proportion=0.0))
    if natural:parts.append(sampler.sample_natural(natural,evaluator=True))
    artifacts=[];clean=[];observed=[];condition=[];metadata=[];latent=[]
    for part in parts:
        proxy=part["stream"]=="natural";target=part["teacher_artifact"] if proxy else part["artifact"]
        for i,meta in enumerate(part["meta"]):
            key=(meta["participant"],meta["session"],meta["task"]);a=target[i];severity=float(np.sqrt(np.mean(a*a)))
            artifacts.append(a);observed.append(part["y"][i]);clean.append(part["y"][i]-a if proxy else part["x"][i]);condition.append(contexts[key]);latent.append(part["latent"][i]);metadata.append({**meta,"stream":part["stream"],"artifact_type":"ocular_EOG","severity":severity,"channel_layout":"SGEYESUB_46ch","source_method":"synchronized_EOG_query_operator_proxy" if proxy else "corrected_coordinate_known_injection","teacher_provenance_status":"proxy" if proxy else "known_generating_process","support_block":"0:3000","query_block":"qnatural>=30000"})
    return {"artifact":np.asarray(artifacts,np.float32),"clean":np.asarray(clean,np.float32),"y":np.asarray(observed,np.float32),"context":np.asarray(condition,np.float32),"latent":np.asarray(latent,np.float32),"meta":metadata}


__all__=["context_bank","exact_support","intervention_contexts","load_support_encoder","sample_targets"]
