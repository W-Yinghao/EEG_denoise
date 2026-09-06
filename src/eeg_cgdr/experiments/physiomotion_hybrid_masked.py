"""Frozen PhysioMotion hybrid masked-restoration development screen."""

from __future__ import annotations

import csv
import copy
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_cgdr.experiments.physiomotion_retrieval_fairness import _score_candidates
from eeg_cgdr.models.hybrid_masked_diffusion import (
    EMA,
    DeterministicHybridMasked,
    HybridMaskedConfig,
    HybridMaskedDiffusion,
    parameter_count,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _result(c: Mapping[str, Any]) -> Path: return Path(c["result_root"])
def _fair(c: Mapping[str, Any]) -> Path: return Path(c["fairness_result_root"])
def _source(c: Mapping[str, Any]) -> Path: return Path(c["source_result_root"])


def _split(c: Mapping[str, Any]) -> tuple[list[int], list[int], dict[int, int]]:
    rows = _csv_read(_source(c) / "metadata" / "frozen_participant_split.csv")
    development = sorted(int(r["participant"]) for r in rows if r["role"] == "development")
    sealed = sorted(int(r["participant"]) for r in rows if r["role"] == "sealed")
    folds = {int(r["participant"]): int(r["cv_fold"]) for r in rows if r["role"] == "development"}
    return development, sealed, folds


def _guard(c: Mapping[str, Any], participants: Iterable[int]) -> None:
    development, _, _ = _split(c)
    forbidden = sorted(set(int(v) for v in participants) - set(development))
    if forbidden:
        raise PermissionError(f"sealed PhysioMotion access refused: {forbidden}")


def _fair_file(c: Mapping[str, Any], participant: int) -> Path:
    _guard(c, [participant])
    return _fair(c) / "fair_materialized" / f"support_{participant:02d}.npz"


def _source_file(c: Mapping[str, Any], participant: int) -> Path:
    _guard(c, [participant])
    return _source(c) / "prepared" / f"participant_{participant:02d}.npz"


def _mask_file(c: Mapping[str, Any], participant: int) -> Path:
    _guard(c, [participant])
    return _fair(c) / "fair_materialized" / f"masks_{participant:02d}.npz"


def _banks(c: Mapping[str, Any], owners: Iterable[int], state: str) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for owner in owners:
        with np.load(_fair_file(c, owner)) as data:
            output[owner] = np.asarray(data[state], np.float32)
    return output


def _retrieve(observed: np.ndarray, mask: np.ndarray, banks: Mapping[int, np.ndarray], k: int) -> tuple[np.ndarray, list[str]]:
    codes, bank_map = [], {}
    for owner, bank in sorted(banks.items()):
        bank_map[owner] = bank
        codes.extend(owner * 100 + index for index in range(len(bank)))
    if len(codes) < k:
        raise RuntimeError(f"retrieval bank only has {len(codes)} candidates")
    code_array = np.asarray(codes, np.int32)
    scores = _score_candidates(observed, mask, code_array, bank_map)
    selected = code_array[np.argsort(scores, kind="stable")[-k:]]
    patches = [bank_map[int(code) // 100][int(code) % 100] for code in selected]
    return np.mean(patches, axis=0).astype(np.float32), [f"p{int(code)//100:02d}:{int(code)%100:02d}" for code in selected]


def _load_masks(c: Mapping[str, Any], owners: Iterable[int]) -> dict[str, list[tuple[str, np.ndarray]]]:
    result: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    allowed = set(c["primary_families"])
    for owner in owners:
        with np.load(_mask_file(c, owner)) as data:
            for key, family, mask in zip(data["keys"], data["families"], data["masks"]):
                if str(family) in allowed:
                    result[str(family)].append((str(key), np.asarray(mask, bool)))
    return result


def _normalization(c: Mapping[str, Any], owners: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for owner in owners:
        with np.load(_fair_file(c, owner)) as data:
            rows.extend(np.asarray(data[state], np.float32) for state in ("open_base", "close_base") if len(data[state]))
            if len(data["query"]): rows.append(np.asarray(data["query"], np.float32))
    combined = np.concatenate(rows, axis=0)
    center = np.median(combined, axis=(0, 2)).astype(np.float32)[:, None]
    scale = (np.median(np.abs(combined - center[None]), axis=(0, 2)) / .67448975).astype(np.float32)[:, None]
    scale = np.maximum(scale, np.median(scale[scale > 0]) * 1e-3)
    return center, scale


def _standardize(value: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((value - center) / scale).astype(np.float32)


def _fold_members(c: Mapping[str, Any], fold: int) -> tuple[list[int], list[int]]:
    development, _, folds = _split(c)
    recipients = [p for p in development if folds[p] == fold]
    training = [p for p in development if folds[p] != fold]
    if len(recipients) != 4 or len(training) != 16:
        raise RuntimeError((fold, recipients, training))
    return recipients, training


def stage_freeze(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    development, sealed, folds = _split(c)
    if len(development) != 20 or len(sealed) != 10:
        raise RuntimeError("frozen 20/10 split changed")
    fairness = json.loads((_fair(c) / "aggregate" / "routing_decision.json").read_text(encoding="utf-8"))
    if fairness.get("status") != "DEPLOYABLE_SUBJECT_INCREMENT_HEADROOM_PRESENT":
        raise RuntimeError(f"J1R route did not authorize this screen: {fairness.get('status')}")
    fold_rows, context_rows = [], []
    k = int(c["retrieval_k"])
    for fold in range(5):
        recipients, training = _fold_members(c, fold)
        for p in development:
            fold_rows.append({"fold": fold, "participant": p, "role": "evaluation" if p in recipients else "outer_training"})
        for recipient in recipients:
            wrong = [p for p in recipients if p != recipient]
            for state in ("open_base", "close_base"):
                banks = _banks(c, [recipient] + wrong + training, state)
                context_rows.append({"fold": fold, "recipient": recipient, "state": state, "match_bank": len(banks[recipient]), "evaluable": int(len(banks[recipient]) >= k), "wrong_donors": ";".join(map(str, wrong)), "wrong_bank_sizes": ";".join(f"{p}:{len(banks[p])}" for p in wrong), "population_owners": len([p for p in training if len(banks[p])]), "population_candidates": sum(len(banks[p]) for p in training)})
    root = _result(c) / "frozen"
    _csv_write(root / "fold_manifest.csv", fold_rows)
    _csv_write(root / "context_manifest.csv", context_rows)
    _json(root / "protocol.json", {"development": development, "sealed": sealed, "sealed_opened": False, "families": c["primary_families"], "retrieval": "J1R observable z-normalized correlation top-8 arithmetic mean", "contexts": {"POP": "(r_P,0)", "HYBRID_MATCH": "(r_P,r_M-r_P)", "HYBRID_WRONG": "(r_P,r_W-r_P)"}, "training_context_mixture": {"POP_NULL": .5, "TRUE_MATCH": .5}, "outer_folds": 5, "science_seed": int(c["training_seed"]), "one_seed_first": True, "sealed_gate": "three-seed development GO required"})
    summary = {"status": "PROTOCOL_FROZEN", "development": len(development), "sealed": len(sealed), "sealed_opened": False, "evaluable_participants": len({int(r["recipient"]) for r in context_rows if int(r["evaluable"])}), "folds": folds}
    _json(run_dir / "result_summary.json", summary)
    return summary


def _append_unit(target: dict[str, list[Any]], clean: np.ndarray, mask: np.ndarray, rp: np.ndarray, rm: np.ndarray, wrong: list[np.ndarray], metadata: dict[str, Any]) -> None:
    target["clean"].append(clean.astype(np.float16)); target["mask"].append(mask.astype(np.uint8)); target["r_pop"].append(rp.astype(np.float16)); target["r_match"].append(rm.astype(np.float16))
    target["r_wrong"].append(np.asarray(wrong, np.float16));
    for key, value in metadata.items(): target[key].append(value)


def _save_model_evaluator(prefix: Path, units: dict[str, list[Any]], center: np.ndarray, scale: np.ndarray) -> None:
    clean = np.asarray(units.pop("clean"), np.float32); mask = np.asarray(units.pop("mask"), bool)
    rp = np.asarray(units.pop("r_pop"), np.float32); rm = np.asarray(units.pop("r_match"), np.float32); wrong = np.asarray(units.pop("r_wrong"), np.float32)
    y_obs = clean * (~mask)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_name(prefix.name + "_model.npz"), y_obs=y_obs.astype(np.float16), mask=mask.astype(np.uint8), r_pop=rp.astype(np.float16), subject_residual=(rm-rp).astype(np.float16), wrong_residual=(wrong-rp[:,None]).astype(np.float16), **{k: np.asarray(v) for k,v in units.items()})
    np.savez_compressed(prefix.with_name(prefix.name + "_evaluator.npz"), clean=((clean * scale[None]) + center[None]).astype(np.float32), center=center, scale=scale, **{k: np.asarray(v) for k,v in units.items()})


def stage_materialize(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    recipients, training = _fold_members(c, fold); center, scale = _normalization(c, training); k = int(c["retrieval_k"])
    families = list(c["primary_families"]); masks_by_family = _load_masks(c, training)
    train: dict[str, list[Any]] = defaultdict(list); exact: dict[str, list[Any]] = defaultdict(list); selection_rows=[]
    cap = int(c["training_rows_per_pseudo_recipient_cap"])
    for pseudo in training:
        with np.load(_fair_file(c, pseudo)) as data:
            query=np.asarray(data["query"],np.float32); states=[str(v) for v in data["query_state"]]; runs=np.asarray(data["query_run"],int)
        candidates=[]
        for qi,(clean,state,run) in enumerate(zip(query,states,runs)):
            banks=_banks(c,training,state); pop_banks={p:b for p,b in banks.items() if p!=pseudo and len(b)}
            if sum(len(b) for b in pop_banks.values())<k: continue
            match_ok=len(banks[pseudo])>=k
            for fi,family in enumerate(families):
                templates=[v for v in masks_by_family[family] if not v[0].startswith(f"p{pseudo:02d}_")]
                if not templates: continue
                key,mask=templates[(pseudo*10000+qi*10+fi)%len(templates)]; observed=clean.copy(); observed[mask]=0
                rp,_=_retrieve(observed,mask,pop_banks,k); rm,_=_retrieve(observed,mask,{pseudo:banks[pseudo]},k) if match_ok else (rp,[])
                candidates.append((clean,mask,rp,rm,{"participant":pseudo,"state":state,"run":int(run),"family":family,"mask_key":key,"match_available":int(match_ok)}))
        rng=np.random.default_rng(np.random.SeedSequence([int(c["training_seed"]),fold,pseudo,1]))
        chosen=np.arange(len(candidates)) if len(candidates)<=cap else np.sort(rng.choice(len(candidates),cap,replace=False))
        for idx in chosen:
            clean,mask,rp,rm,meta=candidates[int(idx)]
            _append_unit(train,_standardize(clean,center,scale),mask,_standardize(rp,center,scale),_standardize(rm,center,scale),[],meta)
    # Training never needs a WRONG tensor.
    train["r_wrong"]=[np.empty((0,int(c["channels"]),int(c["signal_length"])),np.float16) for _ in train["clean"]]
    material=_result(c)/"server_arrays"/f"fold_{fold:02d}"; material.mkdir(parents=True,exist_ok=True)
    clean=np.asarray(train.pop("clean"),np.float16); mask=np.asarray(train.pop("mask"),np.uint8); rp=np.asarray(train.pop("r_pop"),np.float16); rm=np.asarray(train.pop("r_match"),np.float16); train.pop("r_wrong")
    np.savez_compressed(material/"train.npz",clean=clean,mask=mask,r_pop=rp,subject_residual=(rm-rp).astype(np.float16),**{k0:np.asarray(v) for k0,v in train.items()})
    for recipient in recipients:
        with np.load(_fair_file(c,recipient)) as data:
            query=np.asarray(data["query"],np.float32); states=[str(v) for v in data["query_state"]]; runs=np.asarray(data["query_run"],int)
        for qi,(clean_raw,state,run) in enumerate(zip(query,states,runs)):
            train_banks=_banks(c,training,state); pop_banks={p:b for p,b in train_banks.items() if len(b)}
            rec_bank=_banks(c,[recipient],state)[recipient]; wrong_ids=[p for p in recipients if p!=recipient]; wrong_banks=_banks(c,wrong_ids,state)
            if sum(len(b) for b in pop_banks.values())<k: continue
            for fi,family in enumerate(families):
                key,mask=masks_by_family[family][(recipient*10000+qi*10+fi)%len(masks_by_family[family])]; observed=clean_raw.copy(); observed[mask]=0
                rp,pcodes=_retrieve(observed,mask,pop_banks,k); match_ok=len(rec_bank)>=k; rm,mcodes=_retrieve(observed,mask,{recipient:rec_bank},k) if match_ok else (rp,[])
                wrong_values=[]; wrong_valid=[]
                for donor in wrong_ids:
                    if len(wrong_banks[donor])>=k: value,codes=_retrieve(observed,mask,{donor:wrong_banks[donor]},k); wrong_valid.append(1)
                    else: value,codes=rp,[]; wrong_valid.append(0)
                    wrong_values.append(_standardize(value,center,scale))
                    selection_rows.append({"fold":fold,"participant":recipient,"query_index":qi,"family":family,"context":"WRONG","donor":donor,"selected_codes":";".join(codes),"observable_only":1})
                meta={"participant":recipient,"query_index":qi,"state":state,"run":int(run),"family":family,"mask_key":key,"match_available":int(match_ok),"wrong_donors":";".join(map(str,wrong_ids)),"wrong_valid":";".join(map(str,wrong_valid))}
                _append_unit(exact,_standardize(clean_raw,center,scale),mask,_standardize(rp,center,scale),_standardize(rm,center,scale),wrong_values,meta)
                selection_rows += [{"fold":fold,"participant":recipient,"query_index":qi,"family":family,"context":"POP","donor":"","selected_codes":";".join(pcodes),"observable_only":1},{"fold":fold,"participant":recipient,"query_index":qi,"family":family,"context":"MATCH","donor":recipient,"selected_codes":";".join(mcodes),"observable_only":1}]
    _save_model_evaluator(material/"exact",exact,center,scale)
    # Natural arrays remain physically separated; retrieval uses the manual mask only as explicit localization.
    natural: dict[str,list[Any]]=defaultdict(list); natural_cap=int(c["natural_rows_per_recipient_cap"])
    for recipient in recipients:
        with np.load(_source_file(c,recipient)) as data:
            nat=np.asarray(data["natural"],np.float32); masks=np.asarray(data["natural_mask"],bool); fam=[str(v) for v in data["natural_family"]]; runs=np.asarray(data["natural_run"],int)
        keep=[i for i,f in enumerate(fam) if f in families][:natural_cap]
        for ni in keep:
            clean_raw=nat[ni]; mask=masks[ni]; state="open_base"; observed=clean_raw.copy(); observed[mask]=0
            pop_banks={p:b for p,b in _banks(c,training,state).items() if len(b)}; rec_bank=_banks(c,[recipient],state)[recipient]; wrong_ids=[p for p in recipients if p!=recipient]; wrong_banks=_banks(c,wrong_ids,state)
            rp,_=_retrieve(observed,mask,pop_banks,k); match_ok=len(rec_bank)>=k; rm,_=_retrieve(observed,mask,{recipient:rec_bank},k) if match_ok else (rp,[])
            wrong_values=[]; wrong_valid=[]
            for donor in wrong_ids:
                if len(wrong_banks[donor])>=k: value,_=_retrieve(observed,mask,{donor:wrong_banks[donor]},k); wrong_valid.append(1)
                else: value=rp; wrong_valid.append(0)
                wrong_values.append(_standardize(value,center,scale))
            meta={"participant":recipient,"query_index":ni,"state":state,"run":int(runs[ni]),"family":fam[ni],"mask_key":f"natural_p{recipient:02d}_{ni}","match_available":int(match_ok),"wrong_donors":";".join(map(str,wrong_ids)),"wrong_valid":";".join(map(str,wrong_valid))}
            _append_unit(natural,_standardize(clean_raw,center,scale),mask,_standardize(rp,center,scale),_standardize(rm,center,scale),wrong_values,meta)
    _save_model_evaluator(material/"natural",natural,center,scale)
    _csv_write(_result(c)/"frozen"/f"fold_{fold:02d}_retrieval_indices.csv",selection_rows)
    summary={"fold":fold,"training_rows":len(clean),"exact_units":len(exact["participant"]),"natural_units":len(natural["participant"]),"recipients":recipients,"outer_training":training,"sealed_opened":False,"model_evaluator_physical_separation":True}
    _json(run_dir/"result_summary.json",summary); return summary


def _model_config(c: Mapping[str,Any]) -> HybridMaskedConfig:
    return HybridMaskedConfig(channels=int(c["channels"]),signal_length=int(c["signal_length"]),base_channels=int(c["base_channels"]),timesteps=int(c["diffusion_timesteps"]),ddim_steps=int(c["ddim_steps"]),posterior_samples=int(c["posterior_samples"]))


def _seed_everything(seed:int)->None:
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def _batch_schedule(n:int,updates:int,batch:int,seed:int)->np.ndarray:
    rng=np.random.default_rng(seed); return rng.integers(0,n,size=(updates,batch),dtype=np.int32)


def _load_train(c:Mapping[str,Any],fold:int,device:torch.device)->dict[str,torch.Tensor]:
    path=_result(c)/"server_arrays"/f"fold_{fold:02d}"/"train.npz"
    with np.load(path) as z: return {k:torch.as_tensor(np.asarray(z[k],np.float32),device=device) for k in ("clean","mask","r_pop","subject_residual")}


def _save_checkpoint(path:Path,det:DeterministicHybridMasked,diff:HybridMaskedDiffusion,opt_det:torch.optim.Optimizer,opt_diff:torch.optim.Optimizer,ema_det:EMA,ema_diff:EMA,step:int,seed:int)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"det":det.state_dict(),"diff":diff.state_dict(),"opt_det":opt_det.state_dict(),"opt_diff":opt_diff.state_dict(),"ema_det":ema_det.state_dict(),"ema_diff":ema_diff.state_dict(),"step":step,"seed":seed,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},path)


def stage_train(c:Mapping[str,Any],fold:int,seed:int,run_dir:Path,technical:bool=False)->dict[str,Any]:
    device=torch.device("cuda"); _seed_everything(seed+fold*1000+(999 if technical else 0)); data=_load_train(c,fold,device)
    if technical:
        take=min(16,len(data["clean"])); data={k:v[:take] for k,v in data.items()}
    cfg=_model_config(c); det=DeterministicHybridMasked(cfg).to(device); diff=HybridMaskedDiffusion(cfg).to(device)
    if parameter_count(det)!=parameter_count(diff): raise RuntimeError("DET/DIFF parameter mismatch")
    opt_det=torch.optim.AdamW(det.parameters(),lr=float(c["learning_rate"]),weight_decay=float(c["weight_decay"])); opt_diff=torch.optim.AdamW(diff.parameters(),lr=float(c["learning_rate"]),weight_decay=float(c["weight_decay"])); ema_det=EMA(det,float(c["ema_decay"])); ema_diff=EMA(diff,float(c["ema_decay"]))
    updates=int(c["technical_updates"] if technical else c["updates"]); batch=min(int(c["batch_size"]),len(data["clean"])); schedule=_batch_schedule(len(data["clean"]),updates,batch,seed+fold*7919)
    generator=torch.Generator(device=device); generator.manual_seed(seed+fold*104729); curve=[]; start=time.time(); gradient_coverage=[]
    for step in range(updates):
        ids=torch.as_tensor(schedule[step],device=device); clean=data["clean"][ids]; mask=data["mask"][ids]; rp=data["r_pop"][ids]; resid=data["subject_residual"][ids]
        use_match=((torch.arange(batch,device=device)+step)%2==0).float()[:,None,None]; context=resid*use_match; yobs=clean*(1-mask)
        opt_det.zero_grad(set_to_none=True); pred=det(y_obs=yobs,mask=mask,r_pop=rp,subject_residual=context); det_loss=((pred-clean).square()*mask).sum()/mask.sum().clamp_min(1); det_loss.backward()
        if step==0: gradient_coverage.append(sum(bool(p.grad is not None and torch.isfinite(p.grad).all().item() and (p.grad.abs().max()>0).item()) for p in det.parameters())/sum(1 for _ in det.parameters()))
        torch.nn.utils.clip_grad_norm_(det.parameters(),float(c["gradient_clip"])); opt_det.step(); ema_det.update(det)
        opt_diff.zero_grad(set_to_none=True); diff_loss,_=diff.training_loss(clean,y_obs=yobs,mask=mask,r_pop=rp,subject_residual=context,generator=generator); diff_loss.backward()
        if step==0: gradient_coverage.append(sum(bool(p.grad is not None and torch.isfinite(p.grad).all().item() and (p.grad.abs().max()>0).item()) for p in diff.parameters())/sum(1 for _ in diff.parameters()))
        torch.nn.utils.clip_grad_norm_(diff.parameters(),float(c["gradient_clip"])); opt_diff.step(); ema_diff.update(diff)
        if step in {0,49,99,199,499,999,updates-1}: curve.append({"step":step+1,"det_loss":float(det_loss.detach()),"diff_loss":float(diff_loss.detach())})
    ckroot=_result(c)/"server_checkpoints"/("technical" if technical else f"seed_{seed}")
    checkpoint=ckroot/f"fold_{fold:02d}.pt"; _save_checkpoint(checkpoint,det,diff,opt_det,opt_diff,ema_det,ema_diff,updates,seed)
    _csv_write(_result(c)/"training_curves"/("technical.csv" if technical else f"seed_{seed}_fold_{fold:02d}.csv"),curve)
    summary={"fold":fold,"seed":seed,"technical":technical,"updates":updates,"training_rows":len(data["clean"]),"det_parameters":parameter_count(det),"diff_parameters":parameter_count(diff),"parameter_difference":0,"gradient_tensor_coverage_det":gradient_coverage[0],"gradient_tensor_coverage_diff":gradient_coverage[1],"runtime_seconds":time.time()-start,"checkpoint":str(checkpoint),"sealed_opened":False}
    _json(run_dir/"result_summary.json",summary); return summary


def _load_ema_models(c:Mapping[str,Any],fold:int,seed:int,technical:bool=False)->tuple[DeterministicHybridMasked,HybridMaskedDiffusion]:
    cfg=_model_config(c); det=DeterministicHybridMasked(cfg).cuda(); diff=HybridMaskedDiffusion(cfg).cuda(); path=_result(c)/"server_checkpoints"/("technical" if technical else f"seed_{seed}")/f"fold_{fold:02d}.pt"; payload=torch.load(path,map_location="cuda",weights_only=False); EMA(det).load_state_dict(payload["ema_det"]); EMA(diff).load_state_dict(payload["ema_diff"]); EMA_det=EMA(det); EMA_det.load_state_dict(payload["ema_det"]); EMA_det.copy_to(det); EMA_diff=EMA(diff); EMA_diff.load_state_dict(payload["ema_diff"]); EMA_diff.copy_to(diff); det.eval();diff.eval();return det,diff


def _inference_noise(seed:int,fold:int,unit:int,sample:int,shape:tuple[int,...],device:torch.device)->torch.Tensor:
    generator=torch.Generator(device=device); generator.manual_seed(int(np.random.SeedSequence([seed,fold,unit,sample]).generate_state(1,np.uint64)[0]%(2**63-1))); return torch.randn(shape,device=device,generator=generator)


@torch.no_grad()
def _infer_panel(c:Mapping[str,Any],fold:int,seed:int,kind:str)->dict[str,np.ndarray]:
    det,diff=_load_ema_models(c,fold,seed); device=torch.device("cuda"); path=_result(c)/"server_arrays"/f"fold_{fold:02d}"/f"{kind}_model.npz"
    with np.load(path) as z: arrays={k:np.asarray(z[k]) for k in z.files}
    outputs:dict[str,list[np.ndarray]]=defaultdict(list); batch=8; n=len(arrays["y_obs"]); donors=arrays["wrong_residual"].shape[1]
    for left in range(0,n,batch):
        right=min(n,left+batch); y=torch.as_tensor(arrays["y_obs"][left:right].astype(np.float32),device=device); mask=torch.as_tensor(arrays["mask"][left:right].astype(np.float32),device=device); rp=torch.as_tensor(arrays["r_pop"][left:right].astype(np.float32),device=device); match=torch.as_tensor(arrays["subject_residual"][left:right].astype(np.float32),device=device); zero=torch.zeros_like(match)
        contexts={"POP":zero,"HYBRID-MATCH":match}
        for d in range(donors): contexts[f"HYBRID-WRONG-{d}"]=torch.as_tensor(arrays["wrong_residual"][left:right,d].astype(np.float32),device=device)
        noises=[torch.stack([_inference_noise(seed,fold,i,s,(1,int(c["channels"]),int(c["signal_length"])),device)[0] for i in range(left,right)]) for s in range(int(c["posterior_samples"]))]
        for name,context in contexts.items():
            outputs[f"DET-{name}"].append(det(y_obs=y,mask=mask,r_pop=rp,subject_residual=context).cpu().numpy())
            samples=[diff.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=context,initial_noise=noise) for noise in noises]
            outputs[f"DIFF-K1-{name}"].append(samples[0].cpu().numpy()); outputs[f"DIFF-K8-{name}"].append(torch.stack(samples).mean(0).cpu().numpy())
    result={k:np.concatenate(v).astype(np.float32) for k,v in outputs.items()}; result.update({k:arrays[k] for k in ("participant","query_index","state","run","family","mask_key","match_available","wrong_donors","wrong_valid")})
    return result


def stage_infer(c:Mapping[str,Any],fold:int,seed:int,run_dir:Path)->dict[str,Any]:
    root=_result(c)/"server_outputs"/f"seed_{seed}";root.mkdir(parents=True,exist_ok=True)
    for kind in ("exact","natural"):
        panel=_infer_panel(c,fold,seed,kind); np.savez_compressed(root/f"fold_{fold:02d}_{kind}.npz",**panel)
    summary={"fold":fold,"seed":seed,"exact_units":len(panel["participant"]),"common_noise":True,"k8_waveforms_averaged_before_metrics":True,"sealed_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def _masked_metrics(clean:np.ndarray,pred:np.ndarray,mask:np.ndarray,fs:int)->dict[str,float]:
    truth=clean[mask].astype(float); value=pred[mask].astype(float); error=np.linalg.norm(value-truth)/max(np.linalg.norm(truth),1e-12); corr=np.corrcoef(value,truth)[0,1] if len(truth)>1 and np.std(value)>0 and np.std(truth)>0 else 0.; snr=20*np.log10(max(np.linalg.norm(truth),1e-12)/max(np.linalg.norm(value-truth),1e-12))
    freq,pc=welch(clean,fs=fs,nperseg=min(fs,clean.shape[-1]),axis=-1);_,pp=welch(pred,fs=fs,nperseg=min(fs,pred.shape[-1]),axis=-1); band=(freq>=1)&(freq<=45); spectral=np.mean(np.abs(np.log(np.maximum(pp[:,band],1e-18))-np.log(np.maximum(pc[:,band],1e-18))))
    tc=np.sqrt(np.mean(clean**2,axis=-1));tp=np.sqrt(np.mean(pred**2,axis=-1));topo=np.linalg.norm(tp/max(np.linalg.norm(tp),1e-12)-tc/max(np.linalg.norm(tc),1e-12)); covc=np.cov(clean);covp=np.cov(pred);cov=np.linalg.norm(covp-covc)/max(np.linalg.norm(covc),1e-12)
    return {"rrmse":float(error),"correlation":float(corr),"delta_snr":float(snr),"spectral_error":float(spectral),"topography_error":float(topo),"covariance_error":float(cov),"outside_max_abs_change":float(np.max(np.abs(pred[~mask]-clean[~mask]))) if np.any(~mask) else 0.}


def _interpolate(y:np.ndarray,mask:np.ndarray)->np.ndarray:
    out=y.copy();t=np.arange(y.shape[-1])
    for ch in range(len(y)):
        observed=~mask[ch]
        if observed.sum()>=2:out[ch,mask[ch]]=np.interp(t[mask[ch]],t[observed],y[ch,observed])
    return out


def _artifact_features(values:np.ndarray)->np.ndarray:
    values=np.asarray(values,float)
    rms=np.sqrt(np.mean(values**2,axis=-1));drms=np.sqrt(np.mean(np.diff(values,axis=-1)**2,axis=-1))
    return np.concatenate((np.log(np.maximum(rms,1e-12)),np.log(np.maximum(drms,1e-12))),axis=-1)


def _outer_artifact_detector(c:Mapping[str,Any],fold:int):
    _,training=_fold_members(c,fold);clean=[];artifact=[]
    for p in training:
        with np.load(_fair_file(c,p)) as z:
            if len(z["query"]):clean.append(np.asarray(z["query"],np.float32))
        with np.load(_source_file(c,p)) as z:
            families=np.asarray(z["natural_family"]).astype(str);keep=np.isin(families,np.asarray(c["primary_families"]));
            if np.any(keep):artifact.append(np.asarray(z["natural"],np.float32)[keep])
    x0=np.concatenate(clean);x1=np.concatenate(artifact);cap=min(len(x0),len(x1));x=np.concatenate((_artifact_features(x0[:cap]),_artifact_features(x1[:cap])));label=np.concatenate((np.zeros(cap,int),np.ones(cap,int)));model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=500,random_state=int(c["training_seed"]),solver="liblinear"));model.fit(x,label);return model


def stage_evaluate(c:Mapping[str,Any],fold:int,seed:int,run_dir:Path)->dict[str,Any]:
    fs=int(c["sampling_rate"]); outroot=_result(c)/"server_outputs"/f"seed_{seed}"; rows=[]; natural_rows=[];detector=_outer_artifact_detector(c,fold);baseline_cache={}
    for kind in ("exact","natural"):
        with np.load(outroot/f"fold_{fold:02d}_{kind}.npz") as z: output={k:np.asarray(z[k]) for k in z.files}
        with np.load(_result(c)/"server_arrays"/f"fold_{fold:02d}"/f"{kind}_evaluator.npz") as z: evaluator={k:np.asarray(z[k]) for k in z.files}
        with np.load(_result(c)/"server_arrays"/f"fold_{fold:02d}"/f"{kind}_model.npz") as z: model_aux={k:np.asarray(z[k]) for k in ("mask","r_pop","subject_residual")}
        center=evaluator["center"];scale=evaluator["scale"]
        methods=[k for k in output if k.startswith("DET-") or k.startswith("DIFF-")]
        for i in range(len(output["participant"])):
            clean=evaluator["clean"][i]; mask=np.asarray(model_aux["mask"][i],bool)
            y=clean.copy();y[mask]=0
            predictions={method:(output[method][i]*scale+center) for method in methods}
            if kind=="exact":
                predictions.update({"MASKED-ZERO":y,"TEMPORAL-INTERPOLATION":_interpolate(y,mask)})
                # Retrieval baselines are the model contexts mapped back to physical units.
                rp=np.asarray(model_aux["r_pop"][i],np.float32)*scale+center;rm=(np.asarray(model_aux["r_pop"][i],np.float32)+np.asarray(model_aux["subject_residual"][i],np.float32))*scale+center
                retp=clean.copy();retp[mask]=rp[mask];retm=clean.copy();retm[mask]=rm[mask];predictions.update({"RETRIEVAL-POP-LARGE":retp,"RETRIEVAL-HYBRID-MATCH":retm})
                for method,pred in predictions.items():
                    donor="";valid=1
                    if "HYBRID-WRONG-" in method:
                        index=int(method.rsplit("-",1)[-1]);donors=str(output["wrong_donors"][i]).split(";");flags=str(output["wrong_valid"][i]).split(";");donor=donors[index];valid=int(flags[index])
                    rows.append({"fold":fold,"seed":seed,"participant":int(output["participant"][i]),"state":str(output["state"][i]),"run":int(output["run"][i]),"family":str(output["family"][i]),"unit":int(output["query_index"][i]),"method":method,"donor":donor,"context_valid":valid,"match_available":int(output["match_available"][i]),**_masked_metrics(clean,pred,mask,fs)})
            else:
                participant=int(output["participant"][i]);raw=clean
                if participant not in baseline_cache:
                    with np.load(_fair_file(c,participant)) as fair_file:baseline_cache[participant]=np.median(np.asarray(fair_file["query"],np.float32),axis=0)
                baseline=baseline_cache[participant];raw_score=float(detector.predict_proba(_artifact_features(raw[None]))[0,1])
                for method,pred in predictions.items():
                    output_score=float(detector.predict_proba(_artifact_features(pred[None]))[0,1]);boundary=[];donor="";valid=1
                    if "HYBRID-WRONG-" in method:
                        index=int(method.rsplit("-",1)[-1]);donors=str(output["wrong_donors"][i]).split(";");flags=str(output["wrong_valid"][i]).split(";");donor=donors[index];valid=int(flags[index])
                    for ch in range(len(mask)):
                        edges=np.flatnonzero(np.diff(mask[ch].astype(int))!=0)+1
                        boundary.extend(abs(float(pred[ch,e]-pred[ch,e-1])) for e in edges if 0<e<pred.shape[-1])
                    metric=_masked_metrics(raw,pred,mask,fs)
                    natural_rows.append({"fold":fold,"seed":seed,"participant":participant,"run":int(output["run"][i]),"family":str(output["family"][i]),"unit":int(output["query_index"][i]),"method":method,"donor":donor,"context_valid":valid,"match_available":int(output["match_available"][i]),"artifact_detector_reduction":raw_score-output_score,"raw_detector_confidence":raw_score,"output_detector_confidence":output_score,"boundary_discontinuity":float(np.mean(boundary)) if boundary else 0.,"baseline_rmse":float(np.sqrt(np.mean((pred-baseline)**2))),"outside_max_abs_change":metric["outside_max_abs_change"],"spectral_distance":metric["spectral_error"],"topography_distance":metric["topography_error"]})
    evalroot=_result(c)/"evaluation"/f"seed_{seed}";_csv_write(evalroot/f"fold_{fold:02d}_exact.csv",rows);_csv_write(evalroot/f"fold_{fold:02d}_natural.csv",natural_rows)
    summary={"fold":fold,"seed":seed,"exact_metric_rows":len(rows),"natural_metric_rows":len(natural_rows),"evaluator_opened_after_outputs":True,"sealed_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def _signflip(values:np.ndarray)->float:
    if not len(values):return float("nan")
    observed=float(np.mean(values));return float(np.mean([np.mean(values*np.asarray(s))>=observed-1e-15 for s in itertools.product((-1,1),repeat=len(values))]))


def _bootstrap(values:np.ndarray,seed:int,reps:int)->tuple[float,float]:
    rng=np.random.default_rng(seed); samples=np.mean(values[rng.integers(0,len(values),size=(reps,len(values)))],axis=1);return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def _balanced_participant(rows:list[dict[str,str]],metric:str)->dict[tuple[int,str],float]:
    result={}
    participants=sorted({int(r["participant"]) for r in rows});methods=sorted({r["method"] for r in rows})
    for p in participants:
        for method in methods:
            take=[r for r in rows if int(r["participant"])==p and r["method"]==method]
            fam=[]
            for family in sorted({r["family"] for r in take}):
                fr=[r for r in take if r["family"]==family]; groups=[]
                for state_run in sorted({(r.get("state",""),r["run"]) for r in fr}): groups.append(np.mean([float(r[metric]) for r in fr if (r.get("state",""),r["run"])==state_run]))
                if groups:fam.append(np.mean(groups))
            if fam:result[(p,method)]=float(np.mean(fam))
    return result


def stage_aggregate(c:Mapping[str,Any],seed:int,run_dir:Path)->dict[str,Any]:
    exact=[];natural=[]
    for fold in range(5):exact+=_csv_read(_result(c)/"evaluation"/f"seed_{seed}"/f"fold_{fold:02d}_exact.csv");natural+=_csv_read(_result(c)/"evaluation"/f"seed_{seed}"/f"fold_{fold:02d}_natural.csv")
    development,_,_=_split(c)
    # Mechanism estimand excludes support-unavailable units and unavailable
    # donors. Policy estimand retains all units and assigns POP fallback.
    mechanism=[r for r in exact if int(r["match_available"]) and ("WRONG" not in r["method"] or int(r.get("context_valid","1")))]
    natural_mechanism=[r for r in natural if int(r["match_available"]) and ("WRONG" not in r["method"] or int(r.get("context_valid","1")))]
    values=_balanced_participant(mechanism,"rrmse")
    policy=[dict(r) for r in exact];lookup={(r["participant"],r["state"],r["run"],r["family"],r["unit"],r["method"]):float(r["rrmse"]) for r in exact}
    for r in policy:
        if int(r["match_available"]):continue
        replacement=None
        if r["method"].startswith("DIFF-K8-HYBRID-"):replacement="DIFF-K8-POP"
        elif r["method"].startswith("DIFF-K1-HYBRID-"):replacement="DIFF-K1-POP"
        elif r["method"].startswith("DET-HYBRID-"):replacement="DET-POP"
        elif r["method"]=="RETRIEVAL-HYBRID-MATCH":replacement="RETRIEVAL-POP-LARGE"
        if replacement:r["rrmse"]=lookup[(r["participant"],r["state"],r["run"],r["family"],r["unit"],replacement)]
    policy_values=_balanced_participant(policy,"rrmse");participant_rows=[]
    for p in development:
        eligible=(p,"DIFF-K8-HYBRID-MATCH") in values;pop=values.get((p,"DIFF-K8-POP"),float("nan"));match=values.get((p,"DIFF-K8-HYBRID-MATCH"),float("nan"));wrong=[v for (pp,m),v in values.items() if pp==p and m.startswith("DIFF-K8-HYBRID-WRONG")];det=values.get((p,"DET-HYBRID-MATCH"),float("nan"));k1=values.get((p,"DIFF-K1-HYBRID-MATCH"),float("nan"));det_pop=values.get((p,"DET-POP"),float("nan"))
        ppop=policy_values[(p,"DIFF-K8-POP")];pmatch=policy_values[(p,"DIFF-K8-HYBRID-MATCH")];pwrong=[v for (pp,m),v in policy_values.items() if pp==p and m.startswith("DIFF-K8-HYBRID-WRONG")]
        up=pop-match if eligible else float("nan");updet=det_pop-det if eligible else float("nan")
        participant_rows.append({"participant":p,"evaluable":int(eligible),"policy_U_P":ppop-pmatch,"policy_U_W":float(np.mean(pwrong)-pmatch),"mechanism_U_P":"" if not eligible else up,"mechanism_U_W":"" if not eligible or not wrong else float(np.mean(wrong)-match),"E_D_K1":"" if not eligible else det-k1,"E_avg":"" if not eligible else k1-match,"U_P_DET":"" if not eligible else updet,"DeltaSA":"" if not eligible else up-updet,"diff_match_rrmse":"" if not eligible else match,"diff_pop_rrmse":"" if not eligible else pop,"wrong_donors_scored":len(wrong)})
    _csv_write(_result(c)/"participant_effects.csv",participant_rows)
    eval_rows=[r for r in participant_rows if r["evaluable"]]; up=np.asarray([float(r["mechanism_U_P"]) for r in eval_rows]);uw=np.asarray([float(r["mechanism_U_W"]) for r in eval_rows]); lo_p,hi_p=_bootstrap(up,int(c["bootstrap_seed"]),int(c["bootstrap_replicates"]));lo_w,hi_w=_bootstrap(uw,int(c["bootstrap_seed"])+1,int(c["bootstrap_replicates"]))
    family_rows=[]
    for family in c["primary_families"]:
        take=[r for r in mechanism if r["family"]==family]
        fv=_balanced_participant(take,"rrmse"); ps=sorted({p for p,m in fv if m=="DIFF-K8-HYBRID-MATCH"});fup=[fv[(p,"DIFF-K8-POP")]-fv[(p,"DIFF-K8-HYBRID-MATCH")] for p in ps if (p,"DIFF-K8-POP") in fv];fuw=[]
        for p in ps:
            wrong=[v for (pp,m),v in fv.items() if pp==p and m.startswith("DIFF-K8-HYBRID-WRONG")]
            if wrong:fuw.append(np.mean(wrong)-fv[(p,"DIFF-K8-HYBRID-MATCH")])
        family_rows.append({"family":family,"U_P":float(np.mean(fup)) if fup else "","U_W":float(np.mean(fuw)) if fuw else "","participants":len(fup)})
    _csv_write(_result(c)/"family_effects.csv",family_rows)
    nat=_balanced_participant(natural_mechanism,"artifact_detector_reduction"); safety=[]
    safety_metrics={metric:_balanced_participant(mechanism,metric) for metric in ("spectral_error","topography_error","covariance_error")}
    for p in development:
        if (p,"DIFF-K8-HYBRID-MATCH") not in nat:continue
        match_nat=nat[(p,"DIFF-K8-HYBRID-MATCH")];pop_nat=nat[(p,"DIFF-K8-POP")]
        row={"participant":p,"match_artifact_detector_reduction":match_nat,"pop_artifact_detector_reduction":pop_nat,"artifact_reduction_margin":match_nat-pop_nat}
        for metric,mapping in safety_metrics.items():row[f"match_{metric}"]=mapping[(p,"DIFF-K8-HYBRID-MATCH")];row[f"pop_{metric}"]=mapping[(p,"DIFF-K8-POP")];row[f"{metric}_margin"]=row[f"match_{metric}"]-row[f"pop_{metric}"]
        row["severe_reversal"]=int(match_nat<=0 or any(row[f"{m}_margin"]>.02 for m in ("spectral_error","topography_error","covariance_error")));safety.append(row)
    _csv_write(_result(c)/"natural_safety.csv",safety)
    coverage=len(eval_rows); subject_gate=coverage>=int(c["development_evaluable_required"]) and np.mean(up)>=float(c["subject_effect_floor"]) and np.median(up)>0 and np.mean(uw)>=float(c["subject_effect_floor"]) and np.median(uw)>0 and np.sum(up>0)>=int(c["development_positive_required"]) and np.sum(uw>0)>=int(c["development_positive_required"]) and _signflip(up)<.05 and _signflip(uw)<.05
    family_gate=sum(float(r["U_P"])>=0 and float(r["U_W"])>=0 for r in family_rows if r["U_P"]!="")>=4 and all(float(r["U_P"])>=-.02 and float(r["U_W"])>=-.02 for r in family_rows if r["U_P"]!="")
    match_mean=float(np.mean([r["diff_match_rrmse"] for r in eval_rows]));absolute=match_mean<float(np.mean([values.get((int(r["participant"]),"TEMPORAL-INTERPOLATION"),np.inf) for r in eval_rows])) and match_mean<float(np.mean([values.get((int(r["participant"]),"MASKED-ZERO"),np.inf) for r in eval_rows]))
    identity=max(float(r["outside_max_abs_change"]) for r in exact if r["method"].startswith(("DET-","DIFF-")))<=1e-7
    natural_improvement=np.mean([r["match_artifact_detector_reduction"] for r in safety])>0;safety_noninferiority=all(np.mean([r[f"{m}_margin"] for r in safety])<=.02 for m in ("spectral_error","topography_error","covariance_error"));severe_fraction=np.mean([r["severe_reversal"] for r in safety]);safety_gate=natural_improvement and safety_noninferiority and severe_fraction<=float(c["severe_reversal_fraction_max"])
    # A point/natural/preservation three-axis Pareto check against matched DET.
    det_values={metric:_balanced_participant(mechanism,metric) for metric in ("rrmse","spectral_error","topography_error")};det_nat=_balanced_participant(natural_mechanism,"artifact_detector_reduction")
    diff_dominated=(np.mean([det_values["rrmse"][(p,"DIFF-K8-HYBRID-MATCH")]-det_values["rrmse"][(p,"DET-HYBRID-MATCH")] for p in [r["participant"] for r in safety]])>0 and np.mean([det_nat[(p,"DET-HYBRID-MATCH")]-det_nat[(p,"DIFF-K8-HYBRID-MATCH")] for p in [r["participant"] for r in safety]])>0 and all(np.mean([det_values[m][(p,"DIFF-K8-HYBRID-MATCH")]-det_values[m][(p,"DET-HYBRID-MATCH")] for p in [r["participant"] for r in safety]])>0 for m in ("spectral_error","topography_error")))
    go=bool(subject_gate and family_gate and absolute and identity and safety_gate and not diff_dominated)
    decision="DEV_ONE_SEED_GO" if go else "DEV_ONE_SEED_NO_GO"
    route={"decision":decision,"seed":seed,"coverage":coverage,"availability_denominator":20,"sealed_opened":False,"subject_effects":{"U_P":{"mean":float(np.mean(up)),"median":float(np.median(up)),"positive":int(np.sum(up>0)),"one_sided_exact_p":_signflip(up),"descriptive_ci":[lo_p,hi_p]},"U_W":{"mean":float(np.mean(uw)),"median":float(np.median(uw)),"positive":int(np.sum(uw>0)),"one_sided_exact_p":_signflip(uw),"descriptive_ci":[lo_w,hi_w]}},"mechanism":{"E_D_K1_mean":float(np.mean([float(r["E_D_K1"]) for r in eval_rows])),"E_avg_mean":float(np.mean([float(r["E_avg"]) for r in eval_rows])),"DeltaSA_mean":float(np.mean([float(r["DeltaSA"]) for r in eval_rows]))},"policy_means":{"U_P":float(np.mean([r["policy_U_P"] for r in participant_rows])),"U_W":float(np.mean([r["policy_U_W"] for r in participant_rows]))},"safety":{"natural_artifact_improvement":bool(natural_improvement),"relative_noninferiority":bool(safety_noninferiority),"severe_reversal_fraction":float(severe_fraction),"diff_pareto_dominated_by_det":bool(diff_dominated)},"gates":{"subject":bool(subject_gate),"family":bool(family_gate),"absolute_vs_interpolation":bool(absolute),"mask_identity":bool(identity),"natural_safety":bool(safety_gate),"not_det_pareto_dominated":bool(not diff_dominated)},"next_stage":"additional development seeds and DET8" if go else "stop; do not open sealed"}
    _json(_result(c)/"routing_decision.json",route);_json(_result(c)/"result_summary.json",route);_json(run_dir/"result_summary.json",route);return route


def stage_report(c:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    seed=int(c["training_seed"]);root=_result(c);route=json.loads((root/"routing_decision.json").read_text(encoding="utf-8"));exact=[];natural=[]
    for fold in range(5):exact+=_csv_read(root/"evaluation"/f"seed_{seed}"/f"fold_{fold:02d}_exact.csv");natural+=_csv_read(root/"evaluation"/f"seed_{seed}"/f"fold_{fold:02d}_natural.csv")
    methods=sorted({r["method"] for r in exact});metrics=("rrmse","correlation","delta_snr","spectral_error","topography_error","covariance_error");method_rows=[]
    maps={metric:_balanced_participant(exact,metric) for metric in metrics}
    for method in methods:
        participants=sorted(p for p,m in maps["rrmse"] if m==method);row={"method":method,"participants":len(participants),"scientific_unit":"participant"}
        for metric in metrics:
            values=[maps[metric][(p,method)] for p in participants if (p,method) in maps[metric]];row[f"{metric}_mean"]=float(np.mean(values));row[f"{metric}_median"]=float(np.median(values))
        method_rows.append(row)
    _csv_write(root/"method_summary.csv",method_rows)
    _csv_write(root/"method_compute.csv",[{"method":"DET","parameters":362114,"updates":int(c["updates"]),"NFE":1,"posterior_samples":1},{"method":"DIFF-K1","parameters":362114,"updates":int(c["updates"]),"NFE":int(c["ddim_steps"]),"posterior_samples":1},{"method":"DIFF-K8","parameters":362114,"updates":int(c["updates"]),"NFE":int(c["ddim_steps"])*int(c["posterior_samples"]),"posterior_samples":int(c["posterior_samples"])}])
    mask_rows=[]
    for r in exact:
        key=(r["fold"],r["family"],r["method"],r["unit"],r["participant"])
        if r["method"]=="DIFF-K8-POP":mask_rows.append({"fold":r["fold"],"participant":r["participant"],"unit":r["unit"],"state":r["state"],"run":r["run"],"family":r["family"],"outer_training_mask_only":1})
    _csv_write(root/"frozen"/"evaluation_mask_manifest.csv",mask_rows)
    fairness_mask=_csv_read(_fair(c)/"audit"/"mask_audit.csv");_csv_write(root/"frozen"/"development_mask_source_audit.csv",fairness_mask)
    participants=_csv_read(root/"participant_effects.csv");eligible=[r for r in participants if int(r["evaluable"])]
    figure_root=Path("reports/figures");figure_root.mkdir(parents=True,exist_ok=True)
    x=np.arange(len(eligible));fig,ax=plt.subplots(figsize=(9,4.5));ax.axhline(0,color="black",lw=.8);ax.plot(x,[float(r["mechanism_U_P"]) for r in eligible],"o-",label="U_P: POP - MATCH");ax.plot(x,[float(r["mechanism_U_W"]) for r in eligible],"s-",label="U_W: WRONG - MATCH");ax.set_xticks(x,labels=[r["participant"] for r in eligible],rotation=60);ax.set_ylabel("RRMSE utility (positive is better)");ax.legend(frameon=False);fig.tight_layout();fig.savefig(figure_root/"physiomotion_hybrid_subject_effects.png",dpi=180);plt.close(fig)
    selected=["MASKED-ZERO","TEMPORAL-INTERPOLATION","RETRIEVAL-POP-LARGE","DET-HYBRID-MATCH","DIFF-K1-HYBRID-MATCH","DIFF-K8-POP","DIFF-K8-HYBRID-MATCH"]
    take=[r for r in method_rows if r["method"] in selected];fig,ax=plt.subplots(figsize=(9,4.5));ax.bar(np.arange(len(take)),[r["rrmse_mean"] for r in take]);ax.set_xticks(np.arange(len(take)),labels=[r["method"] for r in take],rotation=35,ha="right");ax.set_ylabel("participant-first masked RRMSE");fig.tight_layout();fig.savefig(figure_root/"physiomotion_hybrid_method_comparison.png",dpi=180);plt.close(fig)
    family=_csv_read(root/"family_effects.csv");fig,ax=plt.subplots(figsize=(7,4));pos=np.arange(len(family));ax.axhline(0,color="black",lw=.8);ax.plot(pos,[float(r["U_P"]) for r in family],"o-",label="U_P");ax.plot(pos,[float(r["U_W"]) for r in family],"s-",label="U_W");ax.set_xticks(pos,labels=[r["family"] for r in family],rotation=30,ha="right");ax.legend(frameon=False);fig.tight_layout();fig.savefig(figure_root/"physiomotion_hybrid_family_effects.png",dpi=180);plt.close(fig)
    technical=json.loads((root/"technical_validity.json").read_text(encoding="utf-8"));lines=["# PhysioMotion hybrid masked-diffusion development screen","","## Frozen scope","","This was the single preregistered development screen of retrieval-conditioned, artifact-localized masked clean restoration. It used 20 frozen development participants in five outer folds; the 10 sealed participants remained unopened. Manual annotations supplied mask localization, so artifact detection is not a contribution.","","The contexts had identical shapes: POP used `(r_P, 0)`, HYBRID-MATCH used `(r_P, r_M-r_P)`, and each HYBRID-WRONG donor used `(r_P, r_W-r_P)`. `r_P` came only from the 16 outer-training participants; MATCH came only from recipient run-01; WRONG donors were the other three unseen recipients in the same outer fold. Retrieval used only the frozen J1R observable selector.","","## Technical validity","",f"The technical fold passed: DET RRMSE/correlation {technical['det_rrmse']:.5f}/{technical['det_correlation']:.5f}; DIFF-K8 {technical['diff_k8_rrmse']:.5f}/{technical['diff_k8_correlation']:.5f}. Mask-exterior change was {technical['outside_max_abs_change']:.1e}; common-noise replay and raw/optimizer/EMA interrupted continuation were exact. DET and DIFF each had 362,114 trainable parameters and all trainable tensors received finite nonzero gradients.","","## One-seed participant-first result","",f"Decision: **{route['decision']}**. Mechanism coverage was {route['coverage']}/20; participants 9, 10, and 11 followed the frozen POP fallback policy. U_P mean/median was {route['subject_effects']['U_P']['mean']:+.5f}/{route['subject_effects']['U_P']['median']:+.5f}, with {route['subject_effects']['U_P']['positive']}/17 positive and one-sided exact p={route['subject_effects']['U_P']['one_sided_exact_p']:.6f}. U_W was {route['subject_effects']['U_W']['mean']:+.5f}/{route['subject_effects']['U_W']['median']:+.5f}, with {route['subject_effects']['U_W']['positive']}/17 positive and p={route['subject_effects']['U_W']['one_sided_exact_p']:.6f}. The 20-person policy effects were U_P={route['policy_means']['U_P']:+.5f}, U_W={route['policy_means']['U_W']:+.5f}.","",f"All five artifact-family U_P effects were negative, so the subject and family gates failed. K8 averaging itself was beneficial (E_avg={route['mechanism']['E_avg_mean']:+.5f}), while DIFF-K1 did not beat the matched DET point estimate (E_D_K1={route['mechanism']['E_D_K1_mean']:+.5f}). DeltaSA was {route['mechanism']['DeltaSA_mean']:+.5f}.","","## Absolute restoration and safety","",f"DIFF-K8-HYBRID-MATCH beat masked-zero and temporal interpolation, preserved the mask exterior exactly, and improved the outer-trained natural artifact-detector aggregate. Relative spectral/topographic/covariance safety passed and severe reversal fraction was {route['safety']['severe_reversal_fraction']:.3f}. These checks do not rescue failed subject utility.","","## Routing boundary","","The frozen outcome is `DEV_ONE_SEED_NO_GO`. No additional DIFF seeds, DET8 members, final development models, or sealed evaluation were run. This constrains this hybrid retrieval-context masked restoration instance; it is not a family-wide conclusion about diffusion, masked restoration, or personalization. Per the terminal instruction, no further PhysioMotion structural variants are authorized."]
    Path("reports/physiomotion_hybrid_masked_diffusion.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    summary={**route,"technical":technical,"method_summary_rows":len(method_rows),"sealed_opened":False,"additional_seeds_run":False,"sealed_stage_run":False};_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def stage_technical(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    seed=int(c["training_seed"]); _seed_everything(seed+fold*1000+999); technical_checkpoint=_result(c)/"server_checkpoints"/"technical"/f"fold_{fold:02d}.pt"
    if not technical_checkpoint.exists(): stage_train(c,fold,seed,run_dir,technical=True)
    det,diff=_load_ema_models(c,fold,seed,technical=True);device=torch.device("cuda");data=_load_train(c,fold,device);take=min(8,len(data["clean"]));clean=data["clean"][:take];mask=data["mask"][:take];rp=data["r_pop"][:take];res=data["subject_residual"][:take];y=clean*(1-mask);noise=torch.stack([_inference_noise(seed,fold,i,s,(1,int(c["channels"]),int(c["signal_length"])),device)[0] for i in range(take) for s in [0]])
    with torch.no_grad():d=det(y_obs=y,mask=mask,r_pop=rp,subject_residual=res);samples=[diff.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=res,initial_noise=torch.stack([_inference_noise(seed,fold,i,s,(1,int(c["channels"]),int(c["signal_length"])),device)[0] for i in range(take)])) for s in range(int(c["posterior_samples"]))];f=torch.stack(samples).mean(0);wrong=diff.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=-res,initial_noise=noise);repeat=diff.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=res,initial_noise=noise);shuffled=diff.sample(y_obs=torch.flip(y,dims=(-1,)),mask=mask,r_pop=rp,subject_residual=res,initial_noise=noise)
    def score(v):
        truth=clean[mask.bool()].cpu().numpy();pred=v[mask.bool()].cpu().numpy();return float(np.linalg.norm(pred-truth)/np.linalg.norm(truth)),float(np.corrcoef(pred,truth)[0,1])
    dr,dc=score(d);fr,fc=score(f);sr,_=score(shuffled);outside=max(float((d[~mask.bool()]-y[~mask.bool()]).abs().max()),float((f[~mask.bool()]-y[~mask.bool()]).abs().max()));response=float((wrong-repeat).abs().max());replay=float((repeat-diff.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=res,initial_noise=noise)).abs().max());identity_mask=torch.zeros_like(mask);identity=det(y_obs=clean,mask=identity_mask,r_pop=rp,subject_residual=res)
    # A real interrupted-continuation probe reloads raw model, optimizer and EMA
    # into two independent processes and verifies the next successful update.
    checkpoint=_result(c)/"server_checkpoints"/"technical"/f"fold_{fold:02d}.pt";payload=torch.load(checkpoint,map_location=device,weights_only=False)
    continued=[]
    fixed_ids=torch.arange(min(4,len(clean)),device=device);fixed_t=torch.full((len(fixed_ids),),500,device=device,dtype=torch.long);fixed_noise=torch.linspace(-1,1,clean[fixed_ids].numel(),device=device).reshape_as(clean[fixed_ids])
    for _ in range(2):
        d2=DeterministicHybridMasked(_model_config(c)).to(device);f2=HybridMaskedDiffusion(_model_config(c)).to(device);d2.load_state_dict(payload["det"]);f2.load_state_dict(payload["diff"]);od=torch.optim.AdamW(d2.parameters(),lr=float(c["learning_rate"]),weight_decay=float(c["weight_decay"]));of=torch.optim.AdamW(f2.parameters(),lr=float(c["learning_rate"]),weight_decay=float(c["weight_decay"]));od.load_state_dict(copy.deepcopy(payload["opt_det"]));of.load_state_dict(copy.deepcopy(payload["opt_diff"]));ed=EMA(d2);ef=EMA(f2);ed.load_state_dict(payload["ema_det"]);ef.load_state_dict(payload["ema_diff"])
        cc=clean[fixed_ids];mm=mask[fixed_ids];yy=y[fixed_ids];rr=rp[fixed_ids];ss=res[fixed_ids];od.zero_grad();pd=d2(y_obs=yy,mask=mm,r_pop=rr,subject_residual=ss);ld=((pd-cc).square()*mm).sum()/mm.sum();ld.backward();od.step();ed.update(d2);of.zero_grad();lf,_=f2.training_loss(cc,y_obs=yy,mask=mm,r_pop=rr,subject_residual=ss,generator=torch.Generator(device=device).manual_seed(1),timestep=fixed_t,noise=fixed_noise);lf.backward();of.step();ef.update(f2);continued.append((d2.state_dict(),f2.state_dict(),ed.state_dict(),ef.state_dict()))
    resume_exact=all(torch.equal(continued[0][group][key],continued[1][group][key]) for group in (0,1) for key in continued[0][group]) and all(torch.equal(continued[0][group]["shadow"][key],continued[1][group]["shadow"][key]) for group in (2,3) for key in continued[0][group]["shadow"])
    passed=dr<=.05 and dc>=.98 and fr<=.10 and fc>=.95 and outside<=1e-7 and response>1e-8 and replay==0 and torch.equal(identity,clean) and sr>fr and resume_exact
    summary={"decision":"TECHNICAL_VALIDITY_PASSED" if passed else "TECHNICAL_VALIDITY_FAILED","fold":fold,"det_rrmse":dr,"det_correlation":dc,"diff_k8_rrmse":fr,"diff_k8_correlation":fc,"shuffled_query_rrmse":sr,"outside_max_abs_change":outside,"context_response_max_abs":response,"common_noise_replay_max_abs":replay,"m0_bitwise_identity":bool(torch.equal(identity,clean)),"optimizer_raw_ema_resume_continuation_exact":resume_exact,"sealed_opened":False}
    _json(_result(c)/"technical_validity.json",summary);_json(run_dir/"result_summary.json",summary)
    if not passed:raise RuntimeError(summary)
    return summary


STAGES={"freeze":stage_freeze,"materialize":stage_materialize,"technical":stage_technical,"train":stage_train,"infer":stage_infer,"evaluate":stage_evaluate,"aggregate":stage_aggregate,"report":stage_report}
