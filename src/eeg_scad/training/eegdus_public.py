"""Clean-room EEGDfus-style single-channel public benchmark reproduction."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_scad.models.scad_artifact_diffusion import SCADArtifactDiffusion, SCADConfig
from eeg_scad.training.checkpoint import EMA, clone_with_ema


def _load(path: Path) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as value:return {key:np.asarray(value[key]) for key in value.files}


def train_and_evaluate(data_root: Path, checkpoint: Path, seed: int=20260808, updates: int=2000) -> dict[str,Any]:
    device=torch.device("cuda");torch.manual_seed(seed);rng=np.random.Generator(np.random.PCG64DXSM(seed));generator=torch.Generator(device=device).manual_seed(seed+1);train=_load(data_root/"train.npz");validation=_load(data_root/"validation.npz");test=_load(data_root/"test.npz")
    config=SCADConfig(channels=1,base_channels=32,context_input_dim=1,context_hidden_dim=64,context_dim=64,timesteps=500,ddim_steps=25,parameterization="x0");model=SCADArtifactDiffusion(config).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);ema=EMA(model,.999);curve=[];started=time.time();batch=64
    for step in range(updates):
        index=rng.integers(0,len(train["noisy"]),size=batch);noisy=torch.from_numpy(train["noisy"][index]).to(device);clean=torch.from_numpy(train["clean"][index]).to(device);context=torch.zeros((batch,1),device=device);optimizer.zero_grad(set_to_none=True);loss,_=model.training_loss(clean,noisy,context,generator);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();ema.update(model)
        if (step+1)%200==0 or step+1==updates:
            with torch.no_grad():
                probe=np.arange(min(64,len(validation["noisy"])));vy=torch.from_numpy(validation["noisy"][probe]).to(device);vx=torch.from_numpy(validation["clean"][probe]).to(device);vc=torch.zeros((len(probe),1),device=device);t=torch.full((len(probe),),250,device=device,dtype=torch.long);noise=torch.randn(vx.shape,device=device,generator=generator);_,extra=model.training_loss(vx,vy,vc,generator,t,noise);val=float((extra["predicted_x0"]-vx).square().mean())
            curve.append({"step":step+1,"train_loss":float(loss.detach()),"validation_x0_mse":val})
    checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"ema":ema.state_dict(),"config":config.__dict__,"seed":seed,"updates":updates,"curve":curve},checkpoint);eval_model=clone_with_ema(model,ema).to(device).eval();rows=[]
    with torch.no_grad():
        for start in range(0,len(test["noisy"]),64):
            noisy=torch.from_numpy(test["noisy"][start:start+64]).to(device);clean=torch.from_numpy(test["clean"][start:start+64]).to(device);context=torch.zeros((len(noisy),1),device=device);noise=torch.randn(noisy.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+1000+start));prediction=eval_model.sample(noisy,context,noise,25)[0];rrmse=torch.linalg.vector_norm(prediction-clean,dim=(1,2))/torch.linalg.vector_norm(clean,dim=(1,2)).clamp_min(1e-8);corr=torch.nn.functional.cosine_similarity((prediction-prediction.mean(-1,keepdim=True)).flatten(1),(clean-clean.mean(-1,keepdim=True)).flatten(1),dim=1)
            for offset in range(len(noisy)):rows.append({"snr_db":float(test["snr_db"][start+offset]),"source_record":int(test["source_record"][start+offset]),"rrmse":float(rrmse[offset]),"correlation":float(corr[offset])})
    summary=[]
    for snr in sorted({r["snr_db"] for r in rows}):
        values=[r for r in rows if r["snr_db"]==snr];summary.append({"snr_db":snr,"rrmse_mean":float(np.mean([r["rrmse"] for r in values])),"correlation_mean":float(np.mean([r["correlation"] for r in values])),"source_records":len({r["source_record"] for r in values})})
    return {"classification":"architecture_reimplementation","seed":seed,"updates":updates,"parameters":sum(p.numel() for p in model.parameters()),"training_seconds":time.time()-started,"checkpoint":str(checkpoint),"curve":curve,"test_summary":summary,"finite":bool(all(np.isfinite(r["rrmse"]) for r in rows)),"official_native":False,"source_record_grouped":True,"device":torch.cuda.get_device_name(0)}
