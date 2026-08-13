"""V34P exact-fiber selection, refit, and evaluation protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, pairwise_distances
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bci2a import outer_folds
from .consolidation import _privacy_balance
from .experiment import STRENGTHS, _ece, encode, evaluate_representation, load_cached, seed_all, sha256
from .fiber import FiberOneStep, FiberSANDiff, HeadFiber
from .leace import LEACE
from .models import EEGNetRepresentation


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _load_eegnet(path: Path, device: torch.device) -> EEGNetRepresentation:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = EEGNetRepresentation().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def replace_fiber(model: nn.Module, kind: str, h: np.ndarray, fiber_dim: int, device: torch.device, seed: int) -> np.ndarray:
    model.eval(); outputs=[]
    generator = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, len(h), 256):
        condition = torch.from_numpy(np.ascontiguousarray(h[start:start+256])).float().to(device)
        if kind == "Fiber-OneStep":
            prediction = model(condition)
        else:
            noise = torch.randn((len(condition), fiber_dim), device=device, generator=generator)
            prediction = model.sample(condition, reverse_steps=10, noise=noise)
        outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs)


def sanitize_fiber(model: nn.Module, kind: str, geometry: HeadFiber, z: np.ndarray, device: torch.device, seed: int, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_head, u, h = geometry.decompose(z)
    replacement = replace_fiber(model, kind, h, geometry.fiber_dim, device, seed)
    mixed = (1.0 - alpha) * u + alpha * replacement
    return geometry.compose(z_head, mixed), u, mixed


def exact_preservation(geometry: HeadFiber, before: np.ndarray, after: np.ndarray, task: np.ndarray) -> dict[str, float | int]:
    logits_before = before.astype(np.float64) @ geometry.weight.T + geometry.bias
    logits_after = after.astype(np.float64) @ geometry.weight.T + geometry.bias
    centered_before = logits_before - logits_before.mean(1, keepdims=True)
    centered_after = logits_after - logits_after.mean(1, keepdims=True)
    prob_before = np.exp(centered_before - centered_before.max(1, keepdims=True)); prob_before /= prob_before.sum(1, keepdims=True)
    prob_after = np.exp(centered_after - centered_after.max(1, keepdims=True)); prob_after /= prob_after.sum(1, keepdims=True)
    pred_before = prob_before.argmax(1); pred_after = prob_after.argmax(1)
    return {
        "max_centered_logit_error": float(np.abs(centered_after - centered_before).max(initial=0.0)),
        "max_softmax_probability_error": float(np.abs(prob_after - prob_before).max(initial=0.0)),
        "prediction_mismatch_count": int((pred_after != pred_before).sum()),
        "fixed_head_ba_difference": float(balanced_accuracy_score(task, pred_after) - balanced_accuracy_score(task, pred_before)),
        "fixed_head_ece_difference": float(_ece(prob_after, task) - _ece(prob_before, task)),
    }


def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(2 * pairwise_distances(x, y).mean() - pairwise_distances(x, x).mean() - pairwise_distances(y, y).mean())


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    joined = np.concatenate([x, y])
    distances = pairwise_distances(joined, squared=True)
    nonzero = distances[distances > 0]
    bandwidth = float(np.median(nonzero)) if len(nonzero) else 1.0
    kernel = np.exp(-distances / max(2 * bandwidth, 1e-8))
    n=len(x); return float(kernel[:n,:n].mean()+kernel[n:,n:].mean()-2*kernel[:n,n:].mean())


def distribution_fidelity(u: np.ndarray, replacement: np.ndarray, h: np.ndarray, seed: int) -> dict[str, float]:
    """Class/logit-stratified same-sample fiber distribution diagnostics."""
    predicted = h.argmax(1); confidence = np.linalg.norm(h, axis=1)
    rng=np.random.default_rng(seed); covariance=[];energy=[];mmd=[]
    for task in range(4):
        task_mask=predicted==task
        if task_mask.sum()<8: continue
        cuts=np.quantile(confidence[task_mask],[1/3,2/3])
        bins=np.digitize(confidence,cuts)
        for level in range(3):
            indices=np.flatnonzero(task_mask&(bins==level))
            if len(indices)<8: continue
            if len(indices)>96: indices=np.sort(rng.choice(indices,96,replace=False))
            x=u[indices].astype(np.float64);y=replacement[indices].astype(np.float64)
            scale=np.std(x,axis=0,ddof=1);scale=np.where(scale>1e-6,scale,1.0);x=x/scale;y=y/scale
            cx=np.cov(x,rowvar=False);cy=np.cov(y,rowvar=False)
            covariance.append(float(np.linalg.norm(cy-cx,"fro")/max(np.linalg.norm(cx,"fro"),1e-8)))
            energy.append(_energy_distance(x,y));mmd.append(_mmd_rbf(x,y))
    raw_variance=float(np.var(u,axis=0,ddof=1).sum());replacement_variance=float(np.var(replacement,axis=0,ddof=1).sum())
    return {
        "conditional_covariance_relative_frobenius": float(np.mean(covariance)) if covariance else float("nan"),
        "conditional_energy_distance": float(np.mean(energy)) if energy else float("nan"),
        "conditional_mmd_rbf": float(np.mean(mmd)) if mmd else float("nan"),
        "fiber_variance_retained": replacement_variance/max(raw_variance,1e-8),
        "raw_fiber_variance": raw_variance,
        "replacement_fiber_variance": replacement_variance,
    }


def _selection_balance(row: dict[str, object], fidelity: dict[str, float]) -> float:
    # Fixed-head utility is an identity; selection rewards recoverable task utility,
    # lower adaptive leakage, verification near chance, and gross distribution fidelity.
    base = _privacy_balance(row)
    covariance_penalty = min(float(fidelity["conditional_covariance_relative_frobenius"]), 10.0)
    return float(base - 0.01 * covariance_penalty)


def _train_epoch(model: nn.Module, kind: str, h: np.ndarray, u: np.ndarray, device: torch.device, seed: int, epoch: int, optimizer) -> float:
    dataset=TensorDataset(torch.from_numpy(h).float(),torch.from_numpy(u).float())
    loader=DataLoader(dataset,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch))
    model.train();losses=[]
    for condition,target in loader:
        condition,target=condition.to(device),target.to(device)
        if kind=="Fiber-OneStep":prediction=model(condition)
        else:
            t=torch.randint(0,len(model.alpha_bar),(len(target),),device=device)
            noise=torch.randn_like(target);prediction=model(model.q_sample(target,t,noise),condition,t)
        loss=nn.functional.mse_loss(prediction,target)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_stage_a(kind: str, model: nn.Module, geometry: HeadFiber, z: dict[str,np.ndarray], task: dict[str,np.ndarray], subject: dict[str,np.ndarray], head: nn.Module, device: torch.device, seed: int, fold: int, output: Path) -> dict[str, object]:
    _,u_train,h_train=geometry.decompose(z["train"]);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=-1e9;best_epoch=10;best_state=None;curve=[]
    for epoch in range(80):
        train_loss=_train_epoch(model,kind,h_train,u_train,device,seed,epoch,optimizer)
        if (epoch+1)%10:continue
        sanitized={};fibers={}
        for index,key in enumerate(("train","gallery","query")):
            sanitized[key],original,replacement=sanitize_fiber(model,kind,geometry,z[key],device,seed+10000+index,1.0);fibers[key]=(original,replacement,geometry.decompose(z[key])[2])
        row,_=evaluate_representation(kind,seed,sanitized["train"],task["train"],sanitized["gallery"],task["gallery"],subject["gallery"],sanitized["query"],task["query"],subject["query"],head,device,fold,"strong")
        fidelity=distribution_fidelity(*fibers["query"],seed+epoch);balance=_selection_balance(row,fidelity)
        curve.append({"epoch":epoch+1,"train_x0_mse":train_loss,"validation_balance":balance,**row,**fidelity})
        if balance>best:best=balance;best_epoch=epoch+1;best_state=_cpu_state(model)
    output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":best_state,"selected_epoch":best_epoch,"validation_full_output_balance":best,"rule":"full_output" if kind=="Fiber-OneStep" else "full_10_step"},output)
    return {"selected_epoch":best_epoch,"validation_balance":best,"curve":curve}


def train_exact(kind: str, fiber_dim: int, h: np.ndarray, u: np.ndarray, device: torch.device, seed: int, epochs: int, output: Path) -> nn.Module:
    seed_all(seed);model=(FiberOneStep(fiber_dim) if kind=="Fiber-OneStep" else FiberSANDiff(fiber_dim)).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    curve=[]
    for epoch in range(epochs):curve.append({"epoch":epoch+1,"train_x0_mse":_train_epoch(model,kind,h,u,device,seed,epoch,optimizer)})
    state=_cpu_state(model);output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":state,"epochs":epochs,"selection_stage":"B_refit","kind":kind},output)
    return model


@torch.no_grad()
def _latency(model: nn.Module, kind: str, h: np.ndarray, fiber_dim: int, device: torch.device, batch_size: int, seed: int) -> dict[str, float | int]:
    condition=torch.from_numpy(h[:batch_size]).float().to(device);noise=torch.randn((len(condition),fiber_dim),device=device,generator=torch.Generator(device=device).manual_seed(seed))
    def call():return model(condition) if kind=="Fiber-OneStep" else model.sample(condition,reverse_steps=10,noise=noise)
    for _ in range(20):call()
    if device.type=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    samples=[]
    for _ in range(100):
        start=time.perf_counter();call()
        if device.type=="cuda":torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return {"batch_size":batch_size,"median_ms":float(np.median(samples)),"p95_ms":float(np.percentile(samples,95)),"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()) if device.type=="cuda" else 0,"parameters":sum(p.numel() for p in model.parameters())}


def run_fiber_fold(v33_root: Path, result_root: Path, fold: int, seed: int, device: torch.device) -> dict[str, object]:
    split=outer_folds()[fold];run=result_root/"runtime"/f"fold_{fold}_seed_{seed}";run.mkdir(parents=True,exist_ok=True)
    source=v33_root/"results"/"sandiff_v33p"/"runtime"/f"fold_{fold}_seed_{seed}";cache=v33_root/"results"/"sandiff_v33p"/"runtime"/"bci2a_trials.npz"
    stage_a_model=_load_eegnet(source/"stage_a_eegnet.pt",device);final_model=_load_eegnet(source/"eegnet_full_pool.pt",device)
    train_a=load_cached(cache,split["train_subjects"],"T");val_gallery=load_cached(cache,split["validation_subjects"],"T");val_query=load_cached(cache,split["validation_subjects"],"E")
    za_train,_=encode(stage_a_model,train_a,device);za_gallery,_=encode(stage_a_model,val_gallery,device);za_query,_=encode(stage_a_model,val_query,device);geometry_a=HeadFiber.from_linear(stage_a_model.task_head)
    z_a={"train":za_train,"gallery":za_gallery,"query":za_query};task_a={"train":train_a.task,"gallery":val_gallery.task,"query":val_query.task};subject_a={"train":train_a.subject,"gallery":val_gallery.subject,"query":val_query.subject}
    selections={}
    for offset,kind in ((1000,"Fiber-OneStep"),(2000,"Fiber-SANDiff")):
        seed_all(seed+offset);model=(FiberOneStep(geometry_a.fiber_dim) if kind=="Fiber-OneStep" else FiberSANDiff(geometry_a.fiber_dim)).to(device)
        selections[kind]=train_stage_a(kind,model,geometry_a,z_a,task_a,subject_a,stage_a_model.task_head,device,seed+offset,fold,run/"stage_a"/f"{kind}.pt")
    full_subjects=sorted(split["train_subjects"]+split["validation_subjects"]);full=load_cached(cache,full_subjects,"T");gallery=load_cached(cache,split["test_subjects"],"T");query=load_cached(cache,split["test_subjects"],"E")
    z_full,_=encode(final_model,full,device);z_gallery,_=encode(final_model,gallery,device);z_query,_=encode(final_model,query,device);geometry=HeadFiber.from_linear(final_model.task_head)
    z_head_full,u_full,h_full=geometry.decompose(z_full);z_head_gallery,u_gallery,h_gallery=geometry.decompose(z_gallery);z_head_query,u_query,h_query=geometry.decompose(z_query)
    models={};bindings=[]
    for offset,kind in ((1000,"Fiber-OneStep"),(2000,"Fiber-SANDiff")):
        path=run/"stage_b"/f"{kind}.pt";models[kind]=train_exact(kind,geometry.fiber_dim,h_full,u_full,device,seed+offset,int(selections[kind]["selected_epoch"]),path)
        bindings.append({"fold":fold,"seed":seed,"model":kind,"path":str(path.resolve()),"sha256":sha256(path),"training_subjects":";".join(map(str,full_subjects)),"selected_epoch":selections[kind]["selected_epoch"]})
    for name,path in (("V33P_stage_A_EEGNet",source/"stage_a_eegnet.pt"),("V33P_full_pool_EEGNet",source/"eegnet_full_pool.pt")):
        bindings.append({"fold":fold,"seed":seed,"model":name,"path":str(path.resolve()),"sha256":sha256(path),"training_subjects":";".join(map(str,split["train_subjects"] if "stage_A" in name else full_subjects)),"selected_epoch":"frozen"})
    rows=[];participants=[];preservation=[];fidelity=[];latency=[]
    def add(name,strength,zt,zg,zq):
        row,part=evaluate_representation(name,seed,zt,full.task,zg,gallery.task,gallery.subject,zq,query.task,query.subject,final_model.task_head,device,fold,strength);rows.append(row);participants.extend(part)
    add("RAW","na",z_full,z_gallery,z_query);add("HEAD_ONLY","na",z_head_full,z_head_gallery,z_head_query)
    leace=LEACE.fit(z_full,full.subject);add("LEACE","na",leace.transform(z_full),leace.transform(z_gallery),leace.transform(z_query))
    preservation.append({"fold":fold,"seed":seed,"method":"HEAD_ONLY","strength":"na",**exact_preservation(geometry,z_query,z_head_query,query.task)})
    for index,(kind,model) in enumerate(models.items()):
        for strength,alpha in STRENGTHS.items():
            sanitized={};fibers={}
            for set_index,(key,z) in enumerate((("train",z_full),("gallery",z_gallery),("query",z_query))):
                sanitized[key],original,replacement=sanitize_fiber(model,kind,geometry,z,device,seed+5000+index*100+set_index,alpha);fibers[key]=(original,replacement,geometry.decompose(z)[2])
            add(kind,strength,sanitized["train"],sanitized["gallery"],sanitized["query"])
            preservation.append({"fold":fold,"seed":seed,"method":kind,"strength":strength,**exact_preservation(geometry,z_query,sanitized["query"],query.task)})
            fidelity.append({"fold":fold,"seed":seed,"method":kind,"strength":strength,**distribution_fidelity(*fibers["query"],seed+index)})
        for batch in (1,64):
            item=_latency(model,kind,h_query,geometry.fiber_dim,device,batch,seed);item.update({"fold":fold,"seed":seed,"method":kind});latency.append(item)
    geometry_row={"fold":fold,"seed":seed,**geometry.diagnostics(),"stage_A_head_rank":geometry_a.rank,"stage_A_fiber_dim":geometry_a.fiber_dim}
    selection_rows=[{"fold":fold,"seed":seed,"model":kind,"selected_epoch":value["selected_epoch"],"validation_balance":value["validation_balance"],"selection_train_subjects":";".join(map(str,split["train_subjects"])),"selection_validation_subjects":";".join(map(str,split["validation_subjects"])),"final_refit_subjects":";".join(map(str,full_subjects)),"selection_rule":"full_output" if kind=="Fiber-OneStep" else "full_10_step"} for kind,value in selections.items()]
    payload={"fold":fold,"seed":seed,"split":split,"metrics":rows,"participant_effects":participants,"exact_preservation":preservation,"distribution_fidelity":fidelity,"latency":latency,"checkpoint_binding":bindings,"fiber_geometry":geometry_row,"selection_summary":selection_rows,"selection_curves":{key:value["curve"] for key,value in selections.items()},"waveform_sealed_reads":0,"outer_test_used_for_selection":False}
    (run/"fold_result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");return payload


__all__=["run_fiber_fold","train_stage_a","train_exact","sanitize_fiber","exact_preservation","distribution_fidelity"]
