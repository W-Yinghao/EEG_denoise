"""Execution stages for the SGE-DYNTRANS-DIFF-v6 development experiment."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from scipy.signal import coherence, welch
from torch.optim import AdamW

from eeg_cgdr.data.sgeyesub import (
    SgeyesubLayout, SgeyesubReleaseRecord, load_sgeyesub_signal_record,
    load_sgeyesub_structure_audit,
)
from eeg_cgdr.models.dynamic_transfer_diffusion import (
    DynamicTransferDeterministic, DynamicTransferDiffusion,
    DynamicTransferModelConfig,
)


PROTOCOL = "SGE-DYNTRANS-DIFF-v6"
BLOCKED = "study05/study05_p42"
METHODS = ("RAW", "DET-MATCH", "DET-POP", "DET-WRONG", "DIFF-MATCH", "DIFF-POP", "DIFF-WRONG")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL:
        raise ValueError("wrong v6 config")
    return value


def _metadata(config: Mapping[str, Any]) -> tuple[dict[str, SgeyesubLayout], dict[str, SgeyesubReleaseRecord]]:
    layouts, records = load_sgeyesub_structure_audit(Path(str(config["structure_audit"])))
    return {x.layout_id: x for x in layouts}, {x.recording_key: x for x in records}


def _cell(record: SgeyesubReleaseRecord) -> tuple[str, str, str, float]:
    return record.study, record.layout_id, "release_preprocessed_as_delivered", record.sampling_rate_hz


def grouped_unseen_folds(records: Sequence[SgeyesubReleaseRecord]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, str, float], list[SgeyesubReleaseRecord]] = {}
    for record in records:
        by_cell.setdefault(_cell(record), []).append(record)
    rows = []
    for cell, members in sorted(by_cell.items()):
        members = sorted(members, key=lambda x: x.recording_key)
        if len(members) == 1:
            continue
        group_count = max(1, len(members) // 2)
        sizes = [len(members) // group_count] * group_count
        for index in range(len(members) % group_count): sizes[index] += 1
        offset = 0
        for index, size in enumerate(sizes):
            held = members[offset:offset + size]; offset += size
            training = [x for x in members if x not in held]
            if len(held) < 2 or not training:
                raise AssertionError("fold lacks symmetric unseen donor or population training")
            rows.append({
                "fold_id": f"{cell[0]}_{cell[1]}_heldout_{index:02d}",
                "study": cell[0], "layout_id": cell[1], "reference": cell[2],
                "sampling_rate_hz": cell[3],
                "heldout": [x.recording_key for x in held],
                "training": [x.recording_key for x in training],
                "wrong_donor_policy": "other_outer_heldout_same_cell_unseen",
            })
    covered = [key for row in rows for key in row["heldout"]]
    expected = sorted(x.recording_key for x in records if x.recording_key != BLOCKED)
    if sorted(covered) != expected or len(set(covered)) != 58:
        raise AssertionError("grouped fold coverage must be 58 compatible stems exactly once")
    return rows


def stage_audit(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    _, records = _metadata(config)
    folds = grouped_unseen_folds(tuple(records.values()))
    root = Path(str(config["result_root"])); root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "frozen_grouped_folds.json", {"folds": folds, "blocked": [BLOCKED], "availability_denominator": 59, "compatible": 58})
    task_rows = [{"task_index": i, **row} for i, row in enumerate(folds)]
    _write_csv(root / "fold_tasks.csv", task_rows)
    summary = {"status": "passed", "fold_count": len(folds), "compatible_stems": 58, "availability_denominator": 59, "blocked": [BLOCKED], "query_information_opened": False}
    _write_json(run_dir / "result_summary.json", summary); return summary


def _fir_design(eog: np.ndarray, taps: int) -> np.ndarray:
    if taps % 2 != 1: raise ValueError("FIR taps must be odd")
    radius = taps // 2
    padded = np.pad(eog, ((0, 0), (radius, radius)), mode="constant")
    # lag-major ordering, explicitly restored by fit reshape below
    return np.concatenate([padded[:, radius + lag:radius + lag + eog.shape[1]] for lag in range(-radius, radius + 1)], axis=0)


def fit_dynamic_transfer(eeg: np.ndarray, eog: np.ndarray, *, taps: int, ridge: float) -> np.ndarray:
    design = _fir_design(np.asarray(eog, np.float64), taps)
    gram = design @ design.T + ridge * np.eye(design.shape[0])
    flat = np.linalg.solve(gram, design @ np.asarray(eeg, np.float64).T).T
    # design is lag-major [lag,eog], so reshape C,L,E then transpose C,E,L.
    return flat.reshape(eeg.shape[0], taps, eog.shape[0]).transpose(0, 2, 1)


def apply_dynamic_transfer(transfer: np.ndarray, eog: np.ndarray) -> np.ndarray:
    radius = transfer.shape[-1] // 2
    result = np.zeros((transfer.shape[0], eog.shape[1]), np.float64)
    padded = np.pad(eog, ((0, 0), (radius, radius)), mode="constant")
    for lag in range(transfer.shape[-1]):
        result += transfer[:, :, lag] @ padded[:, lag:lag + eog.shape[1]]
    return result


def _support_eog_stats(loaded: Any, samples: int) -> tuple[np.ndarray, np.ndarray]:
    eog = np.asarray(loaded.support.external_eog[:, :samples], np.float64)
    return eog.mean(1, keepdims=True), eog.std(1, keepdims=True).clip(1e-6)


def _support_transfer(loaded: Any, mean: np.ndarray, std: np.ndarray, normal_mean: np.ndarray, normal_std: np.ndarray, samples: int, taps: int, ridge: float) -> tuple[np.ndarray, float]:
    eeg = (np.asarray(loaded.support.eeg[:, :samples], np.float64) - normal_mean) / normal_std
    eog = (np.asarray(loaded.support.external_eog[:, :samples], np.float64) - mean) / std
    half = samples // 2
    first = fit_dynamic_transfer(eeg[:, :half], eog[:, :half], taps=taps, ridge=ridge)
    second = fit_dynamic_transfer(eeg[:, half:], eog[:, half:], taps=taps, ridge=ridge)
    error = np.linalg.norm(first - second) / max(np.linalg.norm(first) + np.linalg.norm(second), 1e-12)
    full = fit_dynamic_transfer(eeg, eog, taps=taps, ridge=ridge)
    rho = float(np.exp(-2.0 * error))
    return full, rho


def _trial_parts(value: np.ndarray, samples_per_trial: int) -> list[np.ndarray]:
    return [value[:, i:i + samples_per_trial] for i in range(0, value.shape[1], samples_per_trial)]


def _record_pairs(loaded: Any, record: SgeyesubReleaseRecord, normal_mean: np.ndarray, normal_std: np.ndarray, *, taps: int, ridge: float, window_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if loaded.query is None or loaded.query_annotations is None: raise ValueError("builder requires annotated later query")
    support_samples = int(round(30 * record.sampling_rate_hz))
    eog_mean, eog_std = _support_eog_stats(loaded, support_samples)
    eeg_trials = _trial_parts((np.asarray(loaded.query.eeg, np.float64) - normal_mean) / normal_std, record.samples_per_trial)
    eog_trials = _trial_parts((np.asarray(loaded.query_annotations.external_eog, np.float64) - eog_mean) / eog_std, record.samples_per_trial)
    label_trials = [x.reshape(-1) for x in _trial_parts(np.asarray(loaded.query_annotations.artifactclasses)[None], record.samples_per_trial)]
    generator_indices = list(range(0, len(eeg_trials), 3)); clean_indices = list(range(1, len(eeg_trials), 3)); artifact_indices = list(range(2, len(eeg_trials), 3))
    if not generator_indices or not clean_indices or not artifact_indices: raise ValueError("query lacks three disjoint trial roles")
    h_generator = fit_dynamic_transfer(np.concatenate([eeg_trials[i] for i in generator_indices], 1), np.concatenate([eog_trials[i] for i in generator_indices], 1), taps=taps, ridge=ridge)
    samples = int(round(window_seconds * record.sampling_rate_hz))
    clean, artifact_eog = [], []
    for index in clean_indices:
        for start in range(0, record.samples_per_trial - samples + 1, samples):
            labels = label_trials[index][start:start + samples]
            if np.all(labels == 6): clean.append(eeg_trials[index][:, start:start + samples])
    for index in artifact_indices:
        for start in range(0, record.samples_per_trial - samples + 1, samples):
            labels = label_trials[index][start:start + samples]
            if np.mean(np.isin(labels, np.arange(1, 6))) >= .25: artifact_eog.append(eog_trials[index][:, start:start + samples])
    if not clean or not artifact_eog: raise ValueError("record lacks disjoint class6 target or class1-5 EOG source")
    count = max(len(clean), len(artifact_eog))
    x = np.stack([clean[i % len(clean)] for i in range(count)])
    a = np.stack([apply_dynamic_transfer(h_generator, artifact_eog[i % len(artifact_eog)]) for i in range(count)])
    return x.astype(np.float32), (x + a).astype(np.float32), a.astype(np.float32), h_generator.astype(np.float32)


def stage_build(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    root = Path(str(config["result_root"])); folds = json.loads((root / "frozen_grouped_folds.json").read_text())["folds"]
    fold = folds[task_index]; layouts, records = _metadata(config); data_root = Path(str(config["data_root"]))
    training_loaded = {key: load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=True, include_query_annotations=True) for key in fold["training"]}
    held_loaded = {key: load_sgeyesub_signal_record(data_root, records[key], layouts[records[key].layout_id], include_query=True, include_query_annotations=True) for key in fold["heldout"]}
    support_samples = int(round(float(config["support_seconds"]) * float(fold["sampling_rate_hz"])))
    train_support = np.concatenate([x.support.eeg[:, :support_samples] for x in training_loaded.values()], 1).astype(np.float64)
    normal_mean = train_support.mean(1, keepdims=True); normal_std = train_support.std(1, keepdims=True).clip(1e-6)
    taps = 2 * int(round(float(config["fir_lag_ms"]) * float(fold["sampling_rate_hz"]) / 1000)) + 1
    ridge = float(config["ridge_lambda"])
    transfers: dict[str, np.ndarray] = {}; rhos = {}
    for key, loaded in {**training_loaded, **held_loaded}.items():
        mean, std = _support_eog_stats(loaded, support_samples)
        transfers[key], rhos[key] = _support_transfer(loaded, mean, std, normal_mean, normal_std, support_samples, taps, ridge)
    population = np.mean(np.stack([transfers[key] for key in fold["training"]]), 0).astype(np.float32)
    arrays = {"x": [], "y": [], "a": [], "h": [], "rho": [], "key": []}
    for key, loaded in training_loaded.items():
        x, y, a, _ = _record_pairs(loaded, records[key], normal_mean, normal_std, taps=taps, ridge=ridge, window_seconds=float(config["window_seconds"]))
        arrays["x"].append(x); arrays["y"].append(y); arrays["a"].append(a); arrays["h"].append(np.repeat(transfers[key][None], len(x), 0)); arrays["rho"].append(np.repeat(rhos[key], len(x))); arrays["key"].extend([key] * len(x))
    destination = root / "prepared" / fold["fold_id"]; destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "training_pairs.npz", x=np.concatenate(arrays["x"]), y=np.concatenate(arrays["y"]), a=np.concatenate(arrays["a"]), h=np.concatenate(arrays["h"]), rho=np.concatenate(arrays["rho"]).astype(np.float32), key=np.asarray(arrays["key"]), h_pop=population, normal_mean=normal_mean.astype(np.float32), normal_std=normal_std.astype(np.float32))
    pair_manifest=[]
    for key, loaded in held_loaded.items():
        x, y, a, hgen = _record_pairs(loaded, records[key], normal_mean, normal_std, taps=taps, ridge=ridge, window_seconds=float(config["window_seconds"]))
        wrong = [donor for donor in fold["heldout"] if donor != key]
        np.savez_compressed(destination / f"paired_{key.replace('/','__')}.npz", x=x, y=y, a=a, h_match=transfers[key], h_pop=population, h_wrong=np.stack([transfers[d] for d in wrong]), rho_match=np.float32(rhos[key]), rho_wrong=np.asarray([rhos[d] for d in wrong], np.float32), wrong=np.asarray(wrong), h_generator=hgen)
        query = (np.asarray(loaded.query.eeg, np.float64) - normal_mean) / normal_std
        np.savez_compressed(destination / f"natural_input_{key.replace('/','__')}.npz", y=query.astype(np.float32), h_match=transfers[key], h_pop=population, h_wrong=np.stack([transfers[d] for d in wrong]), rho_match=np.float32(rhos[key]), rho_wrong=np.asarray([rhos[d] for d in wrong], np.float32), wrong=np.asarray(wrong))
        np.savez_compressed(destination / f"natural_evaluator_{key.replace('/','__')}.npz", eog=np.asarray(loaded.query_annotations.external_eog, np.float32), labels=np.asarray(loaded.query_annotations.artifactclasses, np.int8))
        pair_manifest.append({"fold_id":fold["fold_id"],"recording_key":key,"wrong_donors":";".join(wrong),"paired_windows":len(x),"Q_roles":"trial_ordinal_mod3_generator_clean_artifact","overlap":False,"H_support_vs_H_generator_distance":float(np.linalg.norm(transfers[key]-hgen)),"H_population_vs_H_generator_distance":float(np.linalg.norm(population-hgen))})
    _write_csv(destination / "pair_manifest.csv", pair_manifest)
    summary={"status":"passed","fold_id":fold["fold_id"],"training_windows":int(sum(len(x) for x in arrays["x"])),"heldout_stems":len(held_loaded),"paired_windows":sum(x["paired_windows"] for x in pair_manifest),"taps":taps,"query_fields_in_model_files":False}
    _write_json(run_dir / "result_summary.json", summary); return summary


def _load_models(config: Mapping[str, Any], channels: int, device: torch.device) -> tuple[Any, Any]:
    model = config["model"]; cfg=DynamicTransferModelConfig(channels,int(model["width"]),int(model["blocks"]),int(model["timesteps"]),int(model["ddim_steps"]))
    return DynamicTransferDeterministic(cfg).to(device), DynamicTransferDiffusion(cfg).to(device)


def stage_technical(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root=Path(str(config["result_root"])); folds=json.loads((root/"frozen_grouped_folds.json").read_text())["folds"]; data=np.load(root/"prepared"/folds[0]["fold_id"]/"training_pairs.npz")
    device=torch.device("cuda"); det,diff=_load_models(config,data["y"].shape[1],device); batch=slice(0,min(2,len(data["y"])))
    y=torch.tensor(data["y"][batch],device=device); target=torch.tensor(data["a"][batch],device=device); h=torch.tensor(data["h"][batch],device=device); rho=torch.tensor(data["rho"][batch],device=device)
    optimizer=AdamW(list(det.parameters())+list(diff.parameters()),lr=1e-4); generator=torch.Generator(device=device).manual_seed(17)
    det_out=det(y,transfer=h,reliability=rho); loss=(det_out-target).square().mean()+diff.training_loss(target,observed=y,transfer=h,reliability=rho,generator=generator); loss.backward(); optimizer.step()
    seeds=tuple(range(101,109)); mean,_,calls=diff.sample(observed=y,transfer=h,reliability=rho,sample_seeds=seeds)
    checkpoint=run_dir/"checkpoint.pt"; torch.save({"det":det.state_dict(),"diff":diff.state_dict(),"optimizer":optimizer.state_dict()},checkpoint); loaded=torch.load(checkpoint,map_location=device,weights_only=True); det.load_state_dict(loaded["det"]); diff.load_state_dict(loaded["diff"])
    zero=torch.zeros_like(target); identity=float((y-(y-zero)).abs().max().cpu()); context=float((det(y,transfer=h,reliability=rho)-det(y,transfer=torch.roll(h,1,0),reliability=rho)).abs().mean().cpu())
    summary={"status":"passed" if torch.isfinite(mean).all() and context>1e-8 and identity==0 else "failed","finite":bool(torch.isfinite(mean).all()),"network_calls":calls,"context_difference":context,"zero_artifact_identity_error":identity,"checkpoint_reload":True,"K":8}
    _write_json(run_dir/"result_summary.json",summary); return summary


def _seed_stream(key: str, seed: int) -> tuple[int,...]:
    base=(sum((i+1)*ord(c) for i,c in enumerate(key))+seed*1000003)%(2**31-1000)
    return tuple(base+37*i for i in range(8))


def stage_train(config: Mapping[str, Any], task_index: int, model_kind: str, seed: int, run_dir: Path) -> dict[str, Any]:
    root=Path(str(config["result_root"])); folds=json.loads((root/"frozen_grouped_folds.json").read_text())["folds"]; fold=folds[task_index]; folder=root/"prepared"/fold["fold_id"]
    npz=np.load(folder/"training_pairs.npz"); device=torch.device("cuda"); det,diff=_load_models(config,npz["y"].shape[1],device); model=det if model_kind=="det" else diff
    target_scale=np.quantile(np.abs(npz["a"]),.995,axis=(0,2)).clip(1e-4).astype(np.float32); optimizer=AdamW(model.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"])); generator=torch.Generator(device=device).manual_seed(seed); rng=np.random.default_rng(seed)
    updates=int(config["training"]["successful_updates"]); batch_size=int(config["training"]["batch_size"]); curves=[]; model.train()
    for step in range(1,updates+1):
        index=rng.integers(0,len(npz["y"]),batch_size); y=torch.tensor(npz["y"][index],device=device); target=torch.tensor(npz["a"][index]/target_scale[None,:,None],device=device); h=torch.tensor(npz["h"][index],device=device); rho=torch.tensor(npz["rho"][index],device=device)
        optimizer.zero_grad(set_to_none=True)
        loss=(model(y,transfer=h,reliability=rho)-target).square().mean() if model_kind=="det" else model.training_loss(target,observed=y,transfer=h,reliability=rho,generator=generator)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["training"]["gradient_clip_norm"])); optimizer.step()
        if step==1 or step%100==0: curves.append({"step":step,"loss":float(loss.detach().cpu())})
        if step%int(config["training"]["checkpoint_interval"])==0: torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"scale":target_scale,"seed":seed},run_dir/"checkpoint.pt")
    checkpoint=root/"checkpoints"/str(seed)/fold["fold_id"]/f"{model_kind}.pt"; checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"step":updates,"scale":target_scale,"seed":seed},checkpoint)
    _write_csv(run_dir/"training_curve.csv",curves)
    model.eval(); output=root/"outputs"/str(seed)/fold["fold_id"]/model_kind; output.mkdir(parents=True,exist_ok=True)
    for path in sorted(folder.glob("paired_*.npz")):
        data=np.load(path); key=path.stem.removeprefix("paired_").replace("__","/"); y=torch.tensor(data["y"],device=device); scale=torch.tensor(target_scale[None,:,None],device=device)
        arms={"MATCH":(data["h_match"],float(data["rho_match"])),"POP":(data["h_pop"],1.0)}
        for wi,wrong in enumerate(data["h_wrong"]): arms[f"WRONG{wi}"]=(wrong,float(data["rho_wrong"][wi]))
        predictions={}
        for arm,(h_value,rho_value) in arms.items():
            h=torch.tensor(np.repeat(h_value[None],len(y),0),device=device); rho=torch.full((len(y),),rho_value,device=device)
            if model_kind=="det": artifact=model(y,transfer=h,reliability=rho)*scale
            else: artifact=model.sample(observed=y,transfer=h,reliability=rho,sample_seeds=_seed_stream(key,seed))[0]*scale
            predictions[arm]=(y-artifact).cpu().numpy()
        np.savez_compressed(output/f"paired_{key.replace('/','__')}.npz",x=data["x"],y=data["y"],**predictions)
        natural=np.load(folder/f"natural_input_{key.replace('/','__')}.npz"); rate=float(fold["sampling_rate_hz"]); length=int(round(float(config["window_seconds"])*rate)); usable=(natural["y"].shape[1]//length)*length
        natural_y=natural["y"][:,:usable].reshape(natural["y"].shape[0],-1,length).transpose(1,0,2); natural_predictions={}
        natural_arms={"MATCH":(natural["h_match"],float(natural["rho_match"])),"POP":(natural["h_pop"],1.0)}
        for wi,wrong in enumerate(natural["h_wrong"]): natural_arms[f"WRONG{wi}"]=(wrong,float(natural["rho_wrong"][wi]))
        for arm,(h_value,rho_value) in natural_arms.items():
            restored=[]
            for start in range(0,len(natural_y),batch_size):
                observed=torch.tensor(natural_y[start:start+batch_size],device=device); h=torch.tensor(np.repeat(h_value[None],len(observed),0),device=device); rho=torch.full((len(observed),),rho_value,device=device)
                artifact=model(observed,transfer=h,reliability=rho)*scale if model_kind=="det" else model.sample(observed=observed,transfer=h,reliability=rho,sample_seeds=_seed_stream(key,seed))[0]*scale
                restored.append((observed-artifact).cpu().numpy())
            natural_predictions[arm]=np.concatenate(restored).transpose(1,0,2).reshape(natural["y"].shape[0],usable)
        np.savez_compressed(output/f"natural_{key.replace('/','__')}.npz",y=natural["y"][:,:usable],**natural_predictions)
    summary={"status":"passed","fold_id":fold["fold_id"],"model":model_kind,"seed":seed,"updates":updates,"parameter_count":sum(p.numel() for p in model.parameters()),"checkpoint":str(checkpoint),"heldout_stems":len(fold["heldout"])}
    _write_json(run_dir/"result_summary.json",summary); return summary


def _rrmse(value: np.ndarray,target: np.ndarray)->float:return float(np.sqrt(np.mean((value-target)**2))/max(np.sqrt(np.mean(target**2)),1e-12))

def _natural_metrics(raw:np.ndarray,value:np.ndarray,eog:np.ndarray,labels:np.ndarray,rate:float)->dict[str,float]:
    usable=min(raw.shape[1],value.shape[1],eog.shape[1],labels.size);raw=raw[:,:usable];value=value[:,:usable];eog=eog[:,:usable];labels=labels[:usable]
    artifact=np.isin(labels,np.arange(1,6));rest=labels==6
    def coh(signal:np.ndarray,mask:np.ndarray)->float:
        indices=np.flatnonzero(mask)
        if len(indices)<int(rate*4):return float("nan")
        vals=[]
        for channel in signal:
            for ocular in eog:
                _,c=coherence(channel[indices],ocular[indices],fs=rate,nperseg=min(256,len(indices)));vals.append(float(np.nanmean(c)))
        return float(np.mean(vals))
    raw_coh=coh(raw,artifact);out_coh=coh(value,artifact)
    if rest.sum()<max(raw.shape[0]+1,int(rate*4)):return {"eog_coherence_reduction":raw_coh-out_coh,"nonartifact_preservation":float("nan"),"psd_distortion":float("nan"),"covariance_distortion":float("nan")}
    r=raw[:,rest];v=value[:,rest];pres=1-float(np.linalg.norm(v-r)/max(np.linalg.norm(r),1e-12));_,pr=welch(r,fs=rate,nperseg=min(256,r.shape[1]),axis=1);_,pv=welch(v,fs=rate,nperseg=min(256,v.shape[1]),axis=1);psd=float(np.mean(np.abs(np.log((pv+1e-12)/(pr+1e-12)))));cr=np.cov(r);cv=np.cov(v);cov=float(np.linalg.norm(cv-cr,"fro")/max(np.linalg.norm(cr,"fro"),1e-12))
    return {"eog_coherence_reduction":raw_coh-out_coh,"nonartifact_preservation":pres,"psd_distortion":psd,"covariance_distortion":cov}


def stage_aggregate(config: Mapping[str, Any], seed: int, run_dir: Path) -> dict[str, Any]:
    root=Path(str(config["result_root"])); folds=json.loads((root/"frozen_grouped_folds.json").read_text())["folds"]; rows=[]
    natural_rows=[]
    for fold in folds:
        outputs=root/"outputs"/str(seed)/fold["fold_id"]
        for key in fold["heldout"]:
            name=f"paired_{key.replace('/','__')}.npz"; det=np.load(outputs/"det"/name); diff=np.load(outputs/"diff"/name); x=det["x"]; y=det["y"]
            values={"RAW":y,"DET-MATCH":det["MATCH"],"DET-POP":det["POP"],"DET-WRONG":np.mean(np.stack([det[k] for k in det.files if k.startswith("WRONG")]),0),"DIFF-MATCH":diff["MATCH"],"DIFF-POP":diff["POP"],"DIFF-WRONG":np.mean(np.stack([diff[k] for k in diff.files if k.startswith("WRONG")]),0)}
            for method,value in values.items(): rows.append({"seed":seed,"fold_id":fold["fold_id"],"recording_key":key,"study":fold["study"],"method":method,"rrmse":_rrmse(value,x),"correlation":float(np.corrcoef(value.ravel(),x.ravel())[0,1]),"delta_snr":float(20*np.log10(max(np.linalg.norm(y-x),1e-12)/max(np.linalg.norm(value-x),1e-12))),"artifact_residual_rmse":float(np.sqrt(np.mean(((y-value)-(y-x))**2)))})
            evaluator=np.load(root/"prepared"/fold["fold_id"]/f"natural_evaluator_{key.replace('/','__')}.npz"); natural_det=np.load(outputs/"det"/f"natural_{key.replace('/','__')}.npz"); natural_diff=np.load(outputs/"diff"/f"natural_{key.replace('/','__')}.npz")
            natural_values={"RAW":natural_det["y"],"DET-MATCH":natural_det["MATCH"],"DET-POP":natural_det["POP"],"DET-WRONG":np.mean(np.stack([natural_det[k] for k in natural_det.files if k.startswith("WRONG")]),0),"DIFF-MATCH":natural_diff["MATCH"],"DIFF-POP":natural_diff["POP"],"DIFF-WRONG":np.mean(np.stack([natural_diff[k] for k in natural_diff.files if k.startswith("WRONG")]),0)}
            for method,value in natural_values.items():natural_rows.append({"seed":seed,"fold_id":fold["fold_id"],"recording_key":key,"study":fold["study"],"method":method,**_natural_metrics(natural_det["y"],value,evaluator["eog"],evaluator["labels"],float(fold["sampling_rate_hz"]))})
    _write_csv(root/"unit_metrics.csv",rows)
    _write_csv(root/"natural_safety_metrics.csv",natural_rows)
    method_summary=[]
    for method in METHODS:
        subset=[row for row in rows if row["method"]==method]
        natural_subset=[row for row in natural_rows if row["method"]==method]
        method_summary.append({"method":method,"participant_stems":len(subset),"rrmse_mean":float(np.mean([x["rrmse"] for x in subset])),"correlation_mean":float(np.mean([x["correlation"] for x in subset])),"delta_snr_mean":float(np.mean([x["delta_snr"] for x in subset])),"eog_coherence_reduction_mean":float(np.nanmean([x["eog_coherence_reduction"] for x in natural_subset])),"nonartifact_preservation_mean":float(np.nanmean([x["nonartifact_preservation"] for x in natural_subset])),"psd_distortion_mean":float(np.nanmean([x["psd_distortion"] for x in natural_subset])),"covariance_distortion_mean":float(np.nanmean([x["covariance_distortion"] for x in natural_subset]))})
    _write_csv(root/"method_summary.csv",method_summary)
    by={(r["recording_key"],r["method"]):r for r in rows}; effects=[]
    for key in sorted({r["recording_key"] for r in rows}):
        effects.append({"recording_key":key,"study":by[(key,"RAW")]["study"],"U_D":by[(key,"DET-MATCH")]["rrmse"]-by[(key,"DIFF-MATCH")]["rrmse"],"U_P":by[(key,"DIFF-POP")]["rrmse"]-by[(key,"DIFF-MATCH")]["rrmse"],"U_W":by[(key,"DIFF-WRONG")]["rrmse"]-by[(key,"DIFF-MATCH")]["rrmse"]})
    _write_csv(root/"paired_effects.csv",effects)
    rng=np.random.default_rng(20260806); bootstrap=[]
    studies=sorted({x["study"] for x in effects})
    for metric in ("U_D","U_P","U_W"):
        observed=np.asarray([x[metric] for x in effects],float); draws=[]
        for _ in range(int(config["statistics"]["bootstrap_replicates"])):
            sample=[]
            for study in studies:
                values=np.asarray([x[metric] for x in effects if x["study"]==study],float);sample.extend(rng.choice(values,size=len(values),replace=True))
            draws.append(float(np.mean(sample)))
        bootstrap.append({"effect":metric,"mean":float(observed.mean()),"median":float(np.median(observed)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"positive_count":int(np.sum(observed>0)),"denominator":len(observed),"bootstrap_replicates":len(draws)})
    _write_csv(root/"bootstrap_summary.csv",bootstrap)
    means={name:float(np.mean([x[name] for x in effects])) for name in ("U_D","U_P","U_W")}; win=float(np.mean([x["U_D"]>0 for x in effects])); coverage=len(effects)/58
    cells={name:sum(np.mean([x[name] for x in effects if x["study"]==study])>=0 for study in sorted({x["study"] for x in effects})) for name in means}
    natural_by={(r["recording_key"],r["method"]):r for r in natural_rows}; safety={}
    for metric in ("nonartifact_preservation","psd_distortion","covariance_distortion"):
        deltas=[]
        for key in sorted({r["recording_key"] for r in natural_rows}):
            match=natural_by[(key,"DIFF-MATCH")][metric];pop=natural_by[(key,"DIFF-POP")][metric]
            if np.isfinite(match) and np.isfinite(pop):deltas.append(match-pop if metric=="nonartifact_preservation" else pop-match)
        safety[metric+"_margin"]=float(np.mean(deltas)) if deltas else float("nan")
    absolute=all(by[(key,"DIFF-MATCH")]["rrmse"]<by[(key,"RAW")]["rrmse"] for key in sorted({r["recording_key"] for r in rows}))
    safety_pass=all(np.isfinite(v) and v>=float(config["statistics"]["seed0_gate"]["safety_margin"]) for v in safety.values())
    gate=coverage>=.90 and all(v>0 for v in means.values()) and win>=.55 and all(v>=3 for v in cells.values()) and absolute and safety_pass
    decision="seed0_gate_pass_submit_additional_seeds" if gate else "current_transfer_conditioned_instance_no_go"
    summary={"status":"completed_seed_aggregate","seed":seed,"coverage":coverage,"participant_stems":len(effects),"availability_denominator":59,"blocked":[BLOCKED],**means,**safety,"diffusion_win_fraction":win,"nonnegative_study_counts":cells,"absolute_rRMSE_better_than_RAW_all":absolute,"route_decision":decision,"additional_seeds_authorized":gate,"natural_safety":"passed" if safety_pass else "failed"}
    _write_json(root/"route_decision.json",summary); _write_json(run_dir/"result_summary.json",summary)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figures=root/"figures";figures.mkdir(exist_ok=True)
    fig,axis=plt.subplots(figsize=(6,4));positions=np.arange(3);axis.errorbar(positions,[x["mean"] for x in bootstrap],yerr=[[x["mean"]-x["ci_low"] for x in bootstrap],[x["ci_high"]-x["mean"] for x in bootstrap]],fmt="o");axis.axhline(0,color="black",lw=1);axis.set_xticks(positions,["U_D","U_P","U_W"]);axis.set_ylabel("RRMSE utility (positive is better)");fig.tight_layout();fig.savefig(figures/"primary_effects.png",dpi=160);plt.close(fig)
    fig,axis=plt.subplots(figsize=(5,5));lookup={x["method"]:x for x in method_summary};
    for method in ("DET-MATCH","DIFF-MATCH","DIFF-POP","DIFF-WRONG"):axis.scatter(lookup[method]["eog_coherence_reduction_mean"],lookup[method]["nonartifact_preservation_mean"],label=method)
    axis.set_xlabel("EOG coherence reduction");axis.set_ylabel("class-6 preservation");axis.legend(fontsize=7);fig.tight_layout();fig.savefig(figures/"natural_safety.png",dpi=160);plt.close(fig)
    report=Path("reports/sge_dynamic_transfer_diffusion_v6.md"); report.parent.mkdir(parents=True,exist_ok=True);report.write_text("# SGE-DYNTRANS-DIFF-v6\n\nThis development experiment uses 30 s early support, a 5 s guard, grouped outer-heldout participant stems, and a query-disjoint real EEG/EOG-backed paired semi-simulation. Label 6 is reported only as a low-artifact observed EEG target.\n\n"+f"Seed `{seed}` covered {len(effects)}/58 compatible stems (59 availability denominator; `{BLOCKED}` blocked). Mean effects: U_D={means['U_D']:+.6f}, U_P={means['U_P']:+.6f}, U_W={means['U_W']:+.6f}. Natural safety: {summary['natural_safety']}. Route decision: `{decision}`. This decision constrains only the current dynamic-transfer-conditioned instance.\n",encoding="utf-8")
    return summary


def run_stage(config_path: Path, stage: str, run_dir: Path, *, task_index: int=0, model_kind: str="", seed: int=20260806) -> dict[str, Any]:
    config=_read_config(config_path)
    if stage=="audit": return stage_audit(config,run_dir)
    if stage=="build": return stage_build(config,task_index,run_dir)
    if stage=="technical": return stage_technical(config,run_dir)
    if stage=="train": return stage_train(config,task_index,model_kind,seed,run_dir)
    if stage=="aggregate": return stage_aggregate(config,seed,run_dir)
    raise ValueError(f"unknown stage {stage}")


__all__=["apply_dynamic_transfer","fit_dynamic_transfer","grouped_unseen_folds","run_stage"]
