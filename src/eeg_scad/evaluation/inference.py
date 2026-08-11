from __future__ import annotations
import csv,json,time
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
import torch
from eeg_scad.training.train import load_ema_model


def _write_csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r})
    with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)


def paired_inference(fold:int,seed:int,derived:Path,checkpoint_root:Path)->dict[str,Any]:
    device=torch.device("cuda");det,_=load_ema_model("det",checkpoint_root/"det"/f"fold_{fold}"/f"seed_{seed}.pt",device);scad,_=load_ema_model("scad",checkpoint_root/"scad"/f"fold_{fold}"/f"seed_{seed}.pt",device)
    with np.load(derived/f"fold_{fold}/paired_test_inference.npz",allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files}
    outputs={};lat=[];batch=16
    for model_name,model in (("DET",det),("SCAD",scad)):
        for context_name in ("pop","match","wrong"):
            values=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            for start in range(0,len(data["y"]),batch):
                y=torch.from_numpy(data["y"][start:start+batch]).to(device);ctx=torch.from_numpy(data[f"context_{context_name}"][start:start+batch]).to(device)
                if model_name=="DET":pred=model(y,ctx)
                else:
                    generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start);noise=torch.randn(y.shape,device=device,generator=generator);pred=scad.sample(y,ctx,noise,25)[0]
                values.append(pred.detach().cpu().numpy().astype(np.float32))
            outputs[f"{model_name}_{context_name.upper()}"]=np.concatenate(values);elapsed=time.perf_counter()-started;lat.append({"fold":fold,"seed":seed,"method":f"{model_name}_{context_name.upper()}","milliseconds_per_window":1000*elapsed/len(data["y"]),"nfe":1 if model_name=="DET" else 25,"peak_memory_mb":torch.cuda.max_memory_allocated()/2**20})
    ctx=torch.zeros_like(torch.from_numpy(data["context_match"])).to(device);values=[]
    for start in range(0,len(data["y"]),batch):
        y=torch.from_numpy(data["y"][start:start+batch]).to(device);generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start);noise=torch.randn(y.shape,device=device,generator=generator);values.append(scad.sample(y,ctx[start:start+len(y)],noise,25)[0].cpu().numpy().astype(np.float32))
    outputs["SCAD_NO_CONTEXT"]=np.concatenate(values);target=derived/"predictions/paired"/f"fold_{fold}_seed_{seed}.npz";target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**outputs);_write_csv(derived/"predictions/paired"/f"fold_{fold}_seed_{seed}_latency.csv",lat)
    if seed==20260808:
        for tag,label in (("scad_no_rank","SCAD_NO_RANK_MATCH"),("scad_v","SCAD_V_MATCH"),("scad_eegdus_unified","EEGDFUS_UNIFIED")):
            path=checkpoint_root/tag/f"fold_{fold}"/f"seed_{seed}.pt"
            if not path.is_file():continue
            variant,_=load_ema_model("scad",path,device);values=[]
            for start in range(0,len(data["y"]),batch):
                y=torch.from_numpy(data["y"][start:start+batch]).to(device);ctx=torch.from_numpy(data["context_match"][start:start+batch]).to(device);ctx=torch.zeros_like(ctx) if tag=="scad_eegdus_unified" else ctx;noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start));values.append(variant.sample(y,ctx,noise,25)[0].cpu().numpy().astype(np.float32))
            outputs[label]=np.concatenate(values)
        np.savez_compressed(target,**outputs)
    return {"fold":fold,"seed":seed,"samples":len(data["y"]),"methods":len(outputs),"prediction_path":str(target),"query_eog_reads":0,"sealed_reads":0}


def natural_inference(fold:int,seed:int,derived:Path,checkpoint_root:Path)->dict[str,Any]:
    device=torch.device("cuda");det,_=load_ema_model("det",checkpoint_root/"det"/f"fold_{fold}"/f"seed_{seed}.pt",device);scad,_=load_ema_model("scad",checkpoint_root/"scad"/f"fold_{fold}"/f"seed_{seed}.pt",device)
    with np.load(derived/f"fold_{fold}/natural_inference.npz",allow_pickle=False) as z:data={k:np.asarray(z[k]) for k in z.files};outputs={};batch=16
    for model_name,model in (("DET",det),("SCAD",scad)):
        for context_name in ("pop","match","wrong"):
            values=[]
            for start in range(0,len(data["y"]),batch):
                y=torch.from_numpy(data["y"][start:start+batch]).to(device);ctx=torch.from_numpy(data[f"context_{context_name}"][start:start+batch]).to(device)
                if model_name=="DET":pred=model(y,ctx)
                else:
                    generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start);noise=torch.randn(y.shape,device=device,generator=generator);pred=scad.sample(y,ctx,noise,25)[0]
                values.append(pred.cpu().numpy().astype(np.float32))
            outputs[f"{model_name}_{context_name.upper()}"]=np.concatenate(values)
    target=derived/"predictions/natural"/f"fold_{fold}_seed_{seed}.npz";target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**outputs)
    if seed==20260808:
        for tag,label in (("scad_no_rank","SCAD_NO_RANK_MATCH"),("scad_v","SCAD_V_MATCH"),("scad_eegdus_unified","EEGDFUS_UNIFIED")):
            path=checkpoint_root/tag/f"fold_{fold}"/f"seed_{seed}.pt"
            if not path.is_file():continue
            variant,_=load_ema_model("scad",path,device);values=[]
            for start in range(0,len(data["y"]),batch):
                y=torch.from_numpy(data["y"][start:start+batch]).to(device);ctx=torch.from_numpy(data["context_match"][start:start+batch]).to(device);ctx=torch.zeros_like(ctx) if tag=="scad_eegdus_unified" else ctx;noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start));values.append(variant.sample(y,ctx,noise,25)[0].cpu().numpy().astype(np.float32))
            outputs[label]=np.concatenate(values)
        np.savez_compressed(target,**outputs)
    return {"fold":fold,"seed":seed,"samples":len(data["y"]),"prediction_path":str(target),"query_eog_reads":0,"query_event_reads":0,"query_operator_reads":0,"sealed_reads":0}
