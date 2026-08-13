"""Two-stage participant-grouped EEGDfus-MC training and frozen-context evaluation."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy import signal
from torch import Tensor, nn

from eeg_scad.data.eog_latent_streams import EOGStreamSampler
from eeg_scad.data.official_support_v40r import exact_support, validate_support_episode
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.eegdfus_mc_v40r import CompactSupportEncoder, EEGDfusMC, LinearSchedule, ddim_sample


CONDITIONS = ("POP", "MATCH", "WRONG", "SHUFFLED", "POP_MEAN", "ADAPTER_DISABLED")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch(bank: Mapping[str, Any], size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, len(bank["y"]), size=size)


def _support_bank(data: Mapping[str, Any], fold: Mapping[str, Any], seconds: int = 30) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    bank, rows = {}, []
    for owner in sorted(set(fold["train"] + fold["validation"] + fold["test"])):
        for session in data["sessions"]:
            for task in data["tasks"]:
                try: episode = exact_support(data, fold, owner, session, task, seconds)
                except FileNotFoundError: continue
                if not validate_support_episode(episode, seconds): raise RuntimeError("invalid support contract")
                assert episode is not None
                key = owner, session, task; bank[key] = episode
                rows.append({"fold": fold["fold"], "participant": owner, "session": session, "task": task,
                             "actual_task": episode["actual_task"], "support_seconds": seconds,
                             "window_count": len(episode["starts"]), "starts": ";".join(map(str, episode["starts"])),
                             "overlap_samples": 0, "repeated_samples": 0, "query_samples": 0,
                             "normalization_prefix_seconds": seconds})
    return bank, rows


def _sample(data: Mapping[str, Any], fold: Mapping[str, Any], split: str, seed: int, paired: int, natural: int = 0) -> dict[str, Any]:
    sampler = EOGStreamSampler(data, fold, split, seed)
    parts = []
    if paired: parts.append(sampler.sample_paired(paired, zero_proportion=.15))
    if natural: parts.append(sampler.sample_natural(natural, evaluator=True))
    result = {"x": [], "y": [], "artifact": [], "latent": [], "meta": [], "stream": []}
    for part in parts:
        proxy = part["stream"] == "natural"
        artifact = part["teacher_artifact"] if proxy else part["artifact"]
        clean = part["y"] - artifact if proxy else part["x"]
        result["x"].extend(clean); result["y"].extend(part["y"]); result["artifact"].extend(artifact)
        result["latent"].extend(part["latent"]); result["meta"].extend(part["meta"]); result["stream"].extend([part["stream"]] * len(clean))
    for key in ("x", "y", "artifact", "latent"): result[key] = np.asarray(result[key], np.float32)
    return result


def _contexts(encoder: CompactSupportEncoder, support: Mapping[tuple[str, str, str], dict[str, Any]], device: torch.device) -> dict[tuple[str, str, str], Tensor]:
    encoder.eval(); output = {}
    with torch.no_grad():
        for key, episode in support.items():
            eeg = torch.from_numpy(episode["eeg"])[None].to(device); eog = torch.from_numpy(episode["eog"])[None].to(device)
            output[key] = encoder(eeg, eog)[0].detach()
    return output


def _train_population(model: EEGDfusMC, schedule: LinearSchedule, train: Mapping[str, Any], val: Mapping[str, Any], device: torch.device, seed: int, updates: int, batch: int) -> list[dict[str, float]]:
    # The official 1e-3 LR assumes batch 512/single channel. The 46-channel port uses
    # 2e-4 at batch 16 after the registered first run produced nonfinite gradients.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4); rng = np.random.default_rng(seed); curve=[]; best=float("inf"); best_state=None
    model.train()
    for step in range(1, updates + 1):
        idx = _batch(train, batch, rng); x=torch.from_numpy(train["x"][idx]).to(device); y=torch.from_numpy(train["y"][idx]).to(device)
        t=torch.randint(0,500,(batch,),device=device); noise=torch.randn_like(x); xt=schedule.q_sample(x,t,noise); pred=model(xt,y,schedule.alpha_bar[t].sqrt()[:,None],bypass=True)
        loss=torch.nn.functional.l1_loss(pred,noise)
        if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite population loss at step {step}")
        optimizer.zero_grad(); loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),5)
        if not torch.isfinite(norm):raise FloatingPointError(f"nonfinite population gradient at step {step}")
        optimizer.step()
        if step % 250 == 0:
            vi=np.arange(min(64,len(val["x"]))); vx=torch.from_numpy(val["x"][vi]).to(device); vy=torch.from_numpy(val["y"][vi]).to(device); vt=torch.full((len(vi),),250,device=device,dtype=torch.long); vn=torch.randn_like(vx)
            model.eval()
            with torch.no_grad(): score=float(torch.nn.functional.l1_loss(model(schedule.q_sample(vx,vt,vn),vy,schedule.alpha_bar[vt].sqrt()[:,None],bypass=True),vn))
            model.train(); curve.append({"step":step,"train_loss":float(loss.detach()),"validation_epsilon_l1":score})
            if score < best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return curve


def _train_adapter(model: EEGDfusMC, encoder: CompactSupportEncoder, schedule: LinearSchedule, train: Mapping[str, Any], support: Mapping[tuple[str,str,str],dict[str,Any]], device: torch.device, seed: int, updates: int, batch: int) -> list[dict[str,float]]:
    model.freeze_population(); encoder.train(); parameters=[p for p in list(model.parameters())+list(encoder.parameters()) if p.requires_grad]; optimizer=torch.optim.Adam(parameters,lr=2e-4); rng=np.random.default_rng(seed); curve=[]
    for step in range(1,updates+1):
        idx=_batch(train,batch,rng); x=torch.from_numpy(train["x"][idx]).to(device); y=torch.from_numpy(train["y"][idx]).to(device); episodes=[support[(train["meta"][i]["participant"],train["meta"][i]["session"],train["meta"][i]["task"])] for i in idx]
        eeg=torch.from_numpy(np.stack([e["eeg"] for e in episodes])).to(device);eog=torch.from_numpy(np.stack([e["eog"] for e in episodes])).to(device);context=encoder(eeg,eog)
        t=torch.randint(0,500,(batch,),device=device);noise=torch.randn_like(x);xt=schedule.q_sample(x,t,noise);pred=model(xt,y,schedule.alpha_bar[t].sqrt()[:,None],context=context);loss=torch.nn.functional.l1_loss(pred,noise)
        if not torch.isfinite(loss):raise FloatingPointError(f"nonfinite adapter loss at step {step}")
        optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(parameters,5)
        if not torch.isfinite(norm):raise FloatingPointError(f"nonfinite adapter gradient at step {step}")
        optimizer.step()
        if step%250==0:curve.append({"step":step,"train_epsilon_l1":float(loss.detach()),"context_norm":float(context.norm(dim=1).mean()),"adapter_gradient_norm":float(sum((p.grad.norm() for p in parameters if p.grad is not None),torch.tensor(0.,device=device)))})
    return curve


def _condition_map(meta: list[dict[str,Any]], context: Mapping[tuple[str,str,str],Tensor], fold: Mapping[str,Any], condition: str, device: torch.device) -> Tensor | None:
    if condition in ("POP", "ADAPTER_DISABLED"): return None
    train_values=torch.stack([value for key,value in context.items() if key[0] in fold["train"]]);mean=train_values.mean(0);owners=sorted({key[0] for key in context});rng=np.random.default_rng(20261040);perm=dict(zip(owners,np.asarray(owners)[rng.permutation(len(owners))]))
    values=[]
    for row in meta:
        owner,session,task=row["participant"],row["session"],row["task"];key=(owner,session,task)
        if condition=="MATCH":value=context[key]
        elif condition=="POP_MEAN":value=mean
        elif condition=="WRONG":
            other=next(candidate for candidate in owners if candidate!=owner and (candidate,session,task) in context);value=context[(other,session,task)]
        elif condition=="SHUFFLED":
            other=perm[owner]
            if other==owner or (other,session,task) not in context:other=next(candidate for candidate in owners if candidate!=owner and (candidate,session,task) in context)
            value=context[(other,session,task)]
        else:raise ValueError(condition)
        values.append(value)
    return torch.stack(values).to(device)


@torch.no_grad()
def _evaluate(model: EEGDfusMC, encoder: CompactSupportEncoder, schedule: LinearSchedule, bank: Mapping[str,Any], support: Mapping[tuple[str,str,str],dict[str,Any]], fold_cfg: Mapping[str,Any], device: torch.device, fold:int, seed:int, condition:str, steps:int=25) -> list[dict[str,Any]]:
    model.eval();encoder.eval();context_bank=_contexts(encoder,support,device);context=_condition_map(bank["meta"],context_bank,fold_cfg,condition,device);bypass=condition in ("POP","ADAPTER_DISABLED");noise=torch.from_numpy(np.random.default_rng(20261040+fold).standard_normal(np.asarray(bank["y"]).shape).astype(np.float32)).to(device);pred=[]
    for start in range(0,len(bank["y"]),16):
        y=torch.from_numpy(bank["y"][start:start+16]).to(device);c=None if context is None else context[start:start+16];pred.append(ddim_sample(model,y,noise[start:start+16],steps,c,bypass,schedule).cpu().numpy())
    pred=np.concatenate(pred);rows=[]
    for i,(x,y,a,meta,stream) in enumerate(zip(bank["x"],bank["y"],bank["artifact"],bank["meta"],bank["stream"])):
        common={"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":"SC-EEGDfus" if condition!="POP" else "EEGDfus-MC-POP","condition":condition,"sampler_steps":steps,"query_eog_inference_reads":0}
        if not np.isfinite(pred[i]).all():raise FloatingPointError("nonfinite diffusion output")
        output_ratio=float(np.sqrt(np.mean(pred[i]**2))/max(np.sqrt(np.mean(y**2)),1e-8))
        if output_ratio>3:raise FloatingPointError(f"output scale collapse ratio={output_ratio:.3f}")
        if stream=="paired":rows.append({**common,"stream":stream,**paired_metrics(x,y,a,y-pred[i])})
        else:
            latent=np.asarray(bank["latent"][i]);energy=np.sqrt(np.mean(latent*latent,axis=0));low=energy<=np.quantile(energy,.3);high=energy>=np.quantile(energy,.7);estimate=y-pred[i];remaining=float(np.linalg.norm(a[:,high]-estimate[:,high])/max(np.linalg.norm(a[:,high]),1e-8));retention=1-float(np.linalg.norm(estimate[:,low])/max(np.linalg.norm(y[:,low]),1e-8));f,p0=signal.welch(y[:,low],fs=100,nperseg=min(128,int(low.sum())),axis=-1);_,p1=signal.welch(pred[i][:,low],fs=100,nperseg=min(128,int(low.sum())),axis=-1);keep=(f>=1)&(f<=15);cov=np.cov(y[:,low]);rows.append({**common,"stream":stream,"heldout_eog_remaining_ratio":remaining,"artifact_attenuation_db":float(-20*np.log10(max(remaining,1e-8))),"eeg_eog_coherence_reduction":float(1-remaining),"low_eog_observation_retention":retention,"psd_distortion":float(np.mean(np.abs(np.log(p0[:,keep]+1e-8)-np.log(p1[:,keep]+1e-8)))),"covariance_distortion":float(np.linalg.norm(np.cov(pred[i][:,low])-cov)/max(np.linalg.norm(cov),1e-8)),"output_input_rms":float(np.sqrt(np.mean(pred[i]**2))/max(np.sqrt(np.mean(y**2)),1e-8))})
    return rows


def run_fold(result_root: Path, data: Mapping[str,Any], fold_cfg: Mapping[str,Any], seed:int, device:torch.device, population_updates:int=20000, adapter_updates:int=5000, run_id:str="runtime") -> dict[str,Any]:
    seed_all(seed);fold=int(fold_cfg["fold"]);runtime=result_root/run_id/f"fold_{fold}_seed_{seed}";runtime.mkdir(parents=True,exist_ok=False);support,support_rows=_support_bank(data,fold_cfg,30)
    train=_sample(data,fold_cfg,"train",seed+1,768);val=_sample(data,fold_cfg,"validation",seed+2,192);paired=_sample(data,fold_cfg,"test",seed+3,384);natural=_sample(data,fold_cfg,"test",seed+4,0,192)
    model=EEGDfusMC().to(device);schedule=LinearSchedule().to(device);pop_curve=_train_population(model,schedule,train,val,device,seed,population_updates,16);pop_path=runtime/"population.pt";torch.save({"model":model.state_dict(),"curve":pop_curve},pop_path)
    encoder=CompactSupportEncoder().to(device);adapter_curve=_train_adapter(model,encoder,schedule,train,support,device,seed+100,adapter_updates,16);adapter_path=runtime/"support.pt";torch.save({"model":model.state_dict(),"support_encoder":encoder.state_dict(),"curve":adapter_curve},adapter_path)
    rows=[]
    for condition in CONDITIONS:rows+=_evaluate(model,encoder,schedule,paired,support,fold_cfg,device,fold,seed,condition);rows+=_evaluate(model,encoder,schedule,natural,support,fold_cfg,device,fold,seed,condition)
    duration=[]
    for seconds in (0,10,30):
        if seconds==0:duration_rows=_evaluate(model,encoder,schedule,paired,support,fold_cfg,device,fold,seed,"POP")
        else:
            duration_support,_=_support_bank(data,fold_cfg,seconds);duration_rows=_evaluate(model,encoder,schedule,paired,duration_support,fold_cfg,device,fold,seed,"MATCH")
        for row in duration_rows:
            if row["stream"]=="paired":duration.append({"fold":fold,"seed":seed,"participant":row["participant"],"support_seconds":seconds,"effective_seconds":seconds,"window_count":seconds//2,"rrmse_temporal":row["rrmse_temporal"]})
    payload={"fold":fold,"seed":seed,"run_id":run_id,"support_manifest":support_rows,"metrics":rows,"support_duration":duration,"checkpoints":[{"model":"EEGDfus-MC-POP","path":str(pop_path),"sha256":sha256(pop_path)},{"model":"SC-EEGDfus","path":str(adapter_path),"sha256":sha256(adapter_path)}],"population_curve":pop_curve,"adapter_curve":adapter_curve,"sealed_reads":0,"query_eog_inference_reads":0,"repair_used":False}
    (runtime/"result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");return payload


__all__=["CONDITIONS","run_fold"]
