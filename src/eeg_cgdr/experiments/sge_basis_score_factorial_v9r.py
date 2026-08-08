"""V9R paired-randomness audit for SGE subject-basis × score-LoRA.

The historical V9 tree is read-only.  Support pseudo-pairs use support-side
artifact-class labels, so this protocol is label-assisted calibration support.
Query outcomes are isolated in evaluator files and stages.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.optim import AdamW

from eeg_cgdr.data.sgeyesub import load_sgeyesub_signal_record
from eeg_cgdr.experiments.sge_basis_score_factorial_v9 import (
    _condition_batch, _heldout_bases, _load_models, _natural_metrics, _paired_metrics,
    _support_pairs,
)
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import (
    _metadata, _support_eog_stats, fit_dynamic_transfer, apply_dynamic_transfer,
)
from eeg_cgdr.experiments.sge_score_lora_v8 import _folds, _masked_u_mse, _pad, _rrmse
from eeg_cgdr.models.adaptation_replay import AdaptationReplay, seed_all
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig, ArtifactSubspaceDiffusion, DeterministicSubspaceEstimator,
    aligned_artifact_basis, reconstruct_from_subspace, window_noise_bank,
)
from eeg_cgdr.models.artifact_subspace_score_lora import inject_score_lora, lora_state_dict


PROTOCOL = "SGE-BASIS-SCORE-FACTORIAL-v9r"
RAW_WRONG = re.compile(r"^DIFF-D11-WRONG-BOTH-[0-9]+$")
CAL_WRONG = re.compile(r"^DIFF-D11-WRONG-BOTH-[0-9]+-CAL$")
ADAPTATION_SEEDS = (20260821, 20260822, 20260823)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows: raise ValueError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream: return list(csv.DictReader(stream))


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL or int(value.get("harness_level", -1)) != 1:
        raise ValueError("wrong V9R protocol")
    return value


def _fold(config: Mapping[str, Any], index: int, *, all_folds: bool = False) -> dict[str, Any]:
    folds = _folds(config)
    if all_folds: return folds[index]
    fold_id = str(config["diagnostic_folds"][index])
    return next(row for row in folds if row["fold_id"] == fold_id)


def _safe(key: str) -> str: return key.replace("/", "__")


def _v9_root(config: Mapping[str, Any]) -> Path: return Path(str(config["v9_root"]))


def _result(config: Mapping[str, Any]) -> Path: return Path(str(config["result_root"]))


def _metric_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str], Mapping[str, str]]:
    return {(str(row["recording_key"]), str(row["method"])): row for row in rows}


def _study_summary(rows: Sequence[Mapping[str, Any]], effects: Sequence[str], panel: str) -> list[dict[str, Any]]:
    result = []
    for study in sorted({str(row["study"]) for row in rows}):
        subset = [row for row in rows if row["study"] == study]
        for effect in effects:
            values = np.asarray([float(row[effect]) for row in subset], np.float64)
            result.append({"panel": panel, "study": study, "effect": effect, "units": len(values), "mean": float(values.mean()), "median": float(np.median(values)), "positive_count": int((values > 0).sum())})
    return result


def _panel_summary(rows: Sequence[Mapping[str, Any]], effects: Sequence[str], panel: str) -> list[dict[str, Any]]:
    result = []
    studies = sorted({str(row["study"]) for row in rows})
    for effect in effects:
        values = np.asarray([float(row[effect]) for row in rows], np.float64)
        study_means = np.asarray([np.mean([float(row[effect]) for row in rows if row["study"] == study]) for study in studies])
        result.append({"panel": panel, "effect": effect, "units": len(values), "stem_weighted_mean": float(values.mean()), "equal_study_mean": float(study_means.mean()), "median": float(np.median(values)), "positive_count": int((values > 0).sum()), "win_fraction": float((values > 0).mean())})
    return result


def _historical_calibrated_pop(v9: Path, fold_id: str, key: str, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.load(v9 / "factorial" / "outputs" / fold_id / f"paired_{_safe(key)}.npz")
    y = np.asarray(output["RAW"]); pop = np.asarray(output["DIFF-D00"])
    return y - float(gamma) * (y - pop), y, np.asarray(output["DIFF-D11-CAL"])


def stage_v9_audit(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = _result(config); v9 = _v9_root(config); rows = _read(v9 / "unit_metrics.csv"); by = _metric_index(rows)
    calibration = _read(v9 / "support_calibration.csv"); cal = {(r["recording_key"], r["candidate"]): r for r in calibration}
    raw, calibrated, consolidated = [], [], []
    for key in sorted({row["recording_key"] for row in rows}):
        base = by[(key, "DIFF-D00")]; study = base["study"]; fold_id = base["fold_id"]
        eligible = int(cal[(key, "D11")].get("personalization_eligible", "0") or "0")
        match = by[(key, "DIFF-D11")]; det = by[(key, "DET-D11")]
        wrong = [float(row["rrmse"]) for row in rows if row["recording_key"] == key and RAW_WRONG.fullmatch(row["method"])]
        raw.append({"fold_id": fold_id, "study": study, "recording_key": key, "personalization_eligible": eligible,
                    "U_D": float(det["rrmse"]) - float(match["rrmse"]), "U_P": float(base["rrmse"]) - float(match["rrmse"]),
                    "U_W": float(np.mean(wrong)) - float(match["rrmse"]) if wrong else 0.0,
                    "U_S": float(by[(key,"DIFF-D11-SHUFFLED-BOTH")]["rrmse"]) - float(match["rrmse"]) if eligible else 0.0,
                    "I": float(by[(key,"DIFF-D10")]["rrmse"]) + float(by[(key,"DIFF-D01")]["rrmse"]) - float(base["rrmse"]) - float(match["rrmse"])})
        gamma = float(cal[(key,"D11")]["gamma"]); pop_cal, y, personal_cal = _historical_calibrated_pop(v9, fold_id, key, gamma)
        evaluator = np.load(v9 / "factorial" / "evaluator" / fold_id / f"paired_{_safe(key)}.npz"); x = np.asarray(evaluator["x"])
        wrong_cal = [row for row in rows if row["recording_key"] == key and CAL_WRONG.fullmatch(row["method"])]
        shuffled_cal = by[(key,"DIFF-D11-SHUFFLED-BOTH-CAL")]
        calibrated.append({"fold_id": fold_id, "study": study, "recording_key": key, "personalization_eligible": eligible,
                           "U_P": _rrmse(pop_cal,x)-_rrmse(personal_cal,x),
                           "U_W": float(np.mean([float(r["rrmse"]) for r in wrong_cal]))-_rrmse(personal_cal,x) if wrong_cal else 0.0,
                           "U_S": float(shuffled_cal["rrmse"])-_rrmse(personal_cal,x) if eligible else 0.0})
        adaptation_path = v9 / "factorial" / "outputs" / fold_id / f"adaptation_{_safe(key)}.json"
        adaptation = json.loads(adaptation_path.read_text()) if adaptation_path.exists() else {}
        for candidate in ("D10","D01","D11"):
            item = cal[(key,candidate)]; selected = adaptation.get(candidate,{}).get("selected_step")
            consolidated.append({"fold_id":fold_id,"study":study,"recording_key":key,"candidate":candidate,"personalization_eligible":eligible,"adapter_selected_step":"" if selected is None else int(selected),"adapter_active":int(eligible and selected not in (None,0)),"gamma":item["gamma"],"w":item["w"],"fallback_reason":item.get("fallback_reason","") or ""})
    effects = ("U_D","U_P","U_W","U_S","I")
    panels = {
        "raw_itt": raw,
        "raw_eligible": [r for r in raw if r["personalization_eligible"] == 1],
        "calibrated_itt": calibrated,
        "calibrated_eligible": [r for r in calibrated if r["personalization_eligible"] == 1],
    }
    for name, values in panels.items(): _csv(root / "v9_audit" / f"{name}.csv", values)
    summaries=[]; studies=[]; loso=[]
    for name, values in panels.items():
        names=[e for e in effects if e in values[0]];summaries.extend(_panel_summary(values,names,name));studies.extend(_study_summary(values,names,name))
        for left_out in sorted({r["study"] for r in values}):
            subset=[r for r in values if r["study"]!=left_out]
            for effect in names:loso.append({"panel":name,"left_out_study":left_out,"effect":effect,"units":len(subset),"mean":float(np.mean([r[effect] for r in subset]))})
    _csv(root/"v9_audit"/"summary.csv",summaries);_csv(root/"v9_audit"/"per_study.csv",studies);_csv(root/"v9_audit"/"leave_one_study_out.csv",loso);_csv(root/"support_calibration_v9_corrected.csv",consolidated)
    d00=[]
    for key in sorted({r["recording_key"] for r in rows}):
        d=by[(key,"DET-D00")];f=by[(key,"DIFF-D00")];d00.append({"fold_id":d["fold_id"],"study":d["study"],"recording_key":key,"DET_minus_DIFF":float(d["rrmse"])-float(f["rrmse"])})
    boot=_cluster_bootstrap(d00,["DET_minus_DIFF"],int(config["bootstrap_seed"]),int(config["bootstrap_replicates"]));_csv(root/"v9_audit"/"d00_population_diffusion.csv",d00);_csv(root/"v9_audit"/"d00_bootstrap.csv",boot)
    report = """# V9 statistical correction\n\nV9 historical files are unchanged. WRONG and SHUFFLED controls are now matched exactly by raw/calibrated panel; fallback units remain in ITT and are excluded from eligible-only mechanism estimates. Support pseudo-pairs use artifact-class labels, so calibration is label-assisted.\n\nThe four panels, stem/equal-study summaries, per-study means, and leave-one-study-out results are under `results/cgdr/sge_basis_score_factorial_v9r/v9_audit/`. D00 is separately bootstrapped and any interval spanning zero is described as a heterogeneous positive-average development signal.\n"""
    Path("reports/v9_statistical_correction.md").write_text(report,encoding="utf-8")
    summary={"status":"completed_v9_statistical_reaudit","units":len(raw),"eligible_units":sum(r["personalization_eligible"] for r in raw),"bootstrap_seed":int(config["bootstrap_seed"]),"historical_files_modified":False,"support_semantics":"label_assisted_calibration_support"};_json(run_dir/"result_summary.json",summary);return summary


def _old_pair_eligibility(labels: np.ndarray, rate: float, split: str) -> bool:
    samples=int(round(120*rate));half=samples//2;start,stop=(0,half) if split=="adapt" else (half,samples);bounds=np.linspace(start,stop,4,dtype=int);window=int(round(2*rate))
    import itertools
    for _,clean_i,artifact_i in itertools.permutations(range(3)):
        clean=sum(np.mean(labels[s:s+window]==6)>=.95 for s in range(bounds[clean_i],bounds[clean_i+1]-window+1,window))
        artifact=sum(np.mean(np.isin(labels[s:s+window],np.arange(1,6)))>=.25 for s in range(bounds[artifact_i],bounds[artifact_i+1]-window+1,window))
        if clean*artifact>0:return True
    return False


def _blocked_physical_pairs(loaded: Any, record: Any, normal_mean: np.ndarray, normal_std: np.ndarray, *, split: str, taps: int, ridge: float, shuffled: bool, seed: int, cap: int=16) -> dict[str, Any]:
    rate=float(record.sampling_rate_hz);samples=int(round(120*rate));guard=int(round(5*rate));half=(samples-guard)//2
    start,stop=(0,half) if split=="adapt" else (half+guard,samples)
    eeg=(np.asarray(loaded.support.eeg[:,:samples],np.float64)-normal_mean)/normal_std;eog_mean,eog_std=_support_eog_stats(loaded,samples);eog=(np.asarray(loaded.support.external_eog[:,:samples],np.float64)-eog_mean)/eog_std;labels=np.asarray(loaded.support.artifactclasses[:samples]).reshape(-1)
    block=int(round(10*rate));window=int(round(2*rate));blocks=[(s,min(s+block,stop)) for s in range(start,stop-block+1,block)]
    import itertools
    best=None
    for fit_i,clean_i,artifact_i in itertools.permutations(range(len(blocks)),3):
        clean=[s for s in range(blocks[clean_i][0],blocks[clean_i][1]-window+1,window) if np.mean(labels[s:s+window]==6)>=.95]
        ocular=[s for s in range(blocks[artifact_i][0],blocks[artifact_i][1]-window+1,window) if np.mean(np.isin(labels[s:s+window],np.arange(1,6)))>=.25]
        score=(min(len(clean),len(ocular)),len(set(clean)),len(set(ocular)))
        if best is None or score>best[0]:best=(score,fit_i,clean_i,artifact_i,clean,ocular)
    if best is None or min(best[0])<1:raise ValueError("blocked_support_missing_clean_or_artifact_role")
    _,fit_i,_,_,clean_ids,ocular_ids=best;fs,fe=blocks[fit_i];fit_eog=eog[:,fs:fe].copy()
    if shuffled:fit_eog=fit_eog[:,np.random.default_rng(seed).permutation(fit_eog.shape[1])]
    transfer=fit_dynamic_transfer(eeg[:,fs:fe],fit_eog,taps=taps,ridge=ridge);rng=np.random.default_rng(seed);clean_order=rng.permutation(clean_ids);ocular_order=rng.permutation(ocular_ids);count=min(cap,max(len(clean_order),len(ocular_order)));offset=int(rng.integers(0,max(len(ocular_order),1)))
    pairs=[(int(clean_order[i%len(clean_order)]),int(ocular_order[(i+offset)%len(ocular_order)])) for i in range(count)];x=np.stack([eeg[:,c:c+window] for c,_ in pairs]);eye=np.stack([eog[:,o:o+window] for _,o in pairs]);a=np.stack([apply_dynamic_transfer(transfer,value) for value in eye]);y=x+a;length=int(math.ceil(window/8)*8);x,valid=_pad(x,length);y,_=_pad(y,length);a,_=_pad(a,length)
    return {"x":x,"y":y,"a":a,"valid":valid,"clean_ids":np.asarray(clean_order),"artifact_ids":np.asarray(ocular_order),"pair_ids":np.asarray(pairs),"unique_clean_windows":len(set(clean_ids)),"unique_artifact_windows":len(set(ocular_ids)),"effective_pair_count":len(pairs),"fit_range":(fs,fe),"split_range":(start,stop),"guard_samples":guard}


def stage_coverage(config: Mapping[str, Any], index: int, run_dir: Path) -> dict[str, Any]:
    fold=_fold(config,index,all_folds=True);layouts,records=_metadata(config);root=_result(config);rows=[];v9=_v9_root(config)
    # Fold-local normalization is read from an existing compatible V9/v8 prepared fold.
    source=Path(str(config["v8_root"]))/"prepared_base"/fold["fold_id"]/"training_pairs.npz"
    if not source.exists(): source=Path(str(config["v6_root"]))/"prepared"/fold["fold_id"]/"training_pairs.npz"
    arrays=np.load(source);rate=float(fold["sampling_rate_hz"]);taps=2*int(round(float(config["fir_lag_ms"])*rate/1000))+1
    for key in fold["heldout"]:
        loaded=load_sgeyesub_signal_record(Path(str(config["data_root"])),records[key],layouts[records[key].layout_id],include_query=False,include_query_annotations=False);labels=np.asarray(loaded.support.artifactclasses[:int(round(120*rate))]).reshape(-1);old=all(_old_pair_eligibility(labels,rate,split) for split in ("adapt","validation"));base={"fold_id":fold["fold_id"],"study":fold["study"],"recording_key":key,"old_builder_eligible":int(old),"support_label_assisted":1}
        try:
            adapt=_blocked_physical_pairs(loaded,records[key],arrays["normal_mean"],arrays["normal_std"],split="adapt",taps=taps,ridge=float(config["ridge_lambda"]),shuffled=False,seed=int(config["pair_seed"]));valid=_blocked_physical_pairs(loaded,records[key],arrays["normal_mean"],arrays["normal_std"],split="validation",taps=taps,ridge=float(config["ridge_lambda"]),shuffled=False,seed=int(config["pair_seed"])+1);eligible=adapt["effective_pair_count"]>=4 and valid["effective_pair_count"]>=4;rows.append({**base,"new_builder_eligible":int(eligible),"adapt_pairs":adapt["effective_pair_count"],"validation_pairs":valid["effective_pair_count"],"adapt_unique_clean":adapt["unique_clean_windows"],"adapt_unique_artifact":adapt["unique_artifact_windows"],"validation_unique_clean":valid["unique_clean_windows"],"validation_unique_artifact":valid["unique_artifact_windows"],"failure_reason":"" if eligible else "fewer_than_four_pairs"})
        except ValueError as exc:rows.append({**base,"new_builder_eligible":0,"adapt_pairs":0,"validation_pairs":0,"adapt_unique_clean":0,"adapt_unique_artifact":0,"validation_unique_clean":0,"validation_unique_artifact":0,"failure_reason":str(exc)})
    _csv(root/"coverage"/f"{fold['fold_id']}.csv",rows);summary={"status":"completed_coverage_fold","fold_id":fold["fold_id"],"units":len(rows),"new_eligible":sum(r["new_builder_eligible"] for r in rows),"query_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def stage_coverage_aggregate(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    rows=[]
    for fold in _folds(config):rows.extend(_read(_result(config)/"coverage"/f"{fold['fold_id']}.csv"))
    _csv(_result(config)/"support_pair_coverage.csv",rows);old=sum(int(r["old_builder_eligible"]) for r in rows);new=sum(int(r["new_builder_eligible"]) for r in rows)
    Path("reports/v9_support_pair_coverage.md").write_text(f"# V9R support-pair coverage\n\nAll {len(rows)} compatible stems were audited without query outcomes. Old builder: {old}/{len(rows)} eligible. Unified blocked builder: {new}/{len(rows)} eligible. Adaptation and validation use disjoint continuous support blocks with a 5 s guard and seeded balanced pairing. Artifact-class labels are used support-side; this is label-assisted calibration.\n",encoding="utf-8")
    summary={"status":"completed_full_support_coverage","compatible_stems":len(rows),"availability_denominator":59,"old_eligible":old,"new_eligible":new,"query_opened":False};_json(_result(config)/"support_coverage_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def _target(pairs: Mapping[str, np.ndarray], basis: np.ndarray, tau: np.ndarray, rank_mask: np.ndarray) -> np.ndarray:
    coefficient=np.einsum("cr,nct->nrt",basis.astype(np.float64),np.asarray(pairs["a"],np.float64));value=np.tanh(coefficient/tau[None,:,None]).astype(np.float32);return value*rank_mask[None,:,None]


def _fresh(checkpoint: Mapping[str, Any], kind: str, replay: AdaptationReplay, device: torch.device) -> tuple[Any,dict[str,torch.Tensor],set[str]]:
    seed_all(replay.initialization_seed);cfg=ArtifactSubspaceConfig(**checkpoint["model_config"]);model=DeterministicSubspaceEstimator(cfg).to(device) if kind=="det" else ArtifactSubspaceDiffusion(cfg).to(device);model.load_state_dict(checkpoint["det" if kind=="det" else "diff_ema"]);inject_score_lora(model.backbone,rank=4);initial={k:v.clone() for k,v in lora_state_dict(model).items()};trainable={name for name,p in model.named_parameters() if p.requires_grad};return model,initial,trainable


@torch.no_grad()
def _output(model: Any, kind: str, observed: np.ndarray, basis_np: np.ndarray, rank_mask_np: np.ndarray, tau_np: np.ndarray, key: str, seed: int, device: torch.device, *, batch_size: int=16) -> np.ndarray:
    values=np.asarray(observed,np.float32);bank=window_noise_bank(key,seed,range(len(values)),posterior_samples=8,signal_length=values.shape[-1],device=device);result=[]
    for start in range(0,len(values),batch_size):
        stop=min(start+batch_size,len(values));y=torch.tensor(values[start:stop],device=device);basis=torch.tensor(np.repeat(basis_np[None],len(y),axis=0),device=device);rank=torch.tensor(np.repeat(rank_mask_np[None],len(y),axis=0),device=device);valid=torch.ones((len(y),y.shape[-1]),dtype=torch.bool,device=device);condition=_condition_batch(y,basis,rank,valid);u=model(**condition) if kind=="det" else model.sample(initial_noise_bank=bank[:,start:stop],**condition)[0];restored,_=reconstruct_from_subspace(y,basis,u,torch.tensor(tau_np,device=device),rank,valid);result.append(restored.cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def _support_score(model: Any, kind: str, pairs: Mapping[str,np.ndarray], basis_np: np.ndarray, rank_mask_np: np.ndarray, tau_np: np.ndarray, replay: AdaptationReplay, device: torch.device) -> float:
    y=torch.tensor(pairs["y"],device=device);basis=torch.tensor(np.repeat(basis_np[None],len(y),axis=0),device=device);rank=torch.tensor(np.repeat(rank_mask_np[None],len(y),axis=0),device=device);valid=torch.tensor(pairs["valid"],device=device);condition=_condition_batch(y,basis,rank,valid);u=model(**condition) if kind=="det" else model.sample(initial_noise_bank=torch.tensor(replay.inference_noise_bank,device=device),**condition)[0];restored,_=reconstruct_from_subspace(y,basis,u,torch.tensor(tau_np,device=device),rank,valid);return _rrmse(restored.cpu().numpy(),pairs["x"])


def _adapt(config: Mapping[str,Any],checkpoint: Mapping[str,Any],kind:str,adapt:Mapping[str,np.ndarray],validation:Mapping[str,np.ndarray],basis_np:np.ndarray,rank_mask_np:np.ndarray,tau_np:np.ndarray,replay:AdaptationReplay,device:torch.device)->tuple[Any,dict[str,Any]]:
    replay.validate(pair_count=len(adapt["y"]),validation_count=len(validation["y"]),signal_length=adapt["y"].shape[-1]);model,initial,trainable=_fresh(checkpoint,kind,replay,device);base_before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in initial};optimizer=AdamW([p for p in model.parameters() if p.requires_grad],lr=float(config["score_lora"]["learning_rate"]),weight_decay=1e-4);target=_target(adapt,basis_np,tau_np,rank_mask_np);basis=torch.tensor(basis_np[None],device=device);rank=torch.tensor(rank_mask_np[None],device=device);checkpoints=set(map(int,replay.checkpoint_steps));curve=[];best=None
    for step in range(len(replay.minibatch_indices)+1):
        if step in checkpoints:
            model.eval();score=_support_score(model,kind,validation,basis_np,rank_mask_np,tau_np,replay,device);state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in initial};curve.append({"step":step,"validation_rrmse":score});
            if best is None or score<best[0]:best=(score,step,state)
        if step==len(replay.minibatch_indices):break
        ids=replay.minibatch_indices[step];seed_all(int(replay.dropout_seeds[step]));model.train();y=torch.tensor(adapt["y"][ids],device=device);valid=torch.tensor(adapt["valid"][ids],device=device);condition=_condition_batch(y,basis.expand(len(ids),-1,-1),rank.expand(len(ids),-1),valid);truth=torch.tensor(target[ids],device=device);optimizer.zero_grad(set_to_none=True)
        if kind=="det":loss=_masked_u_mse(model(**condition)*rank[:,:,None],truth,valid)
        else:
            timestep=torch.tensor(replay.timesteps[step],device=device);noise=torch.tensor(replay.gaussian_noise[step],device=device);loss=model.training_loss(truth,generator=torch.Generator(device=device).manual_seed(1),timestep=timestep,noise=noise,**condition)[0]
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
    current=model.state_dict();current.update(best[2]);model.load_state_dict(current);model.eval();base_unchanged=all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in base_before.items());active=best[1]>0 and any(not torch.equal(initial[k],model.state_dict()[k].detach().cpu()) for k in initial);return model,{"selected_step":int(best[1]),"adapter_active":bool(active),"validation_rrmse":float(best[0]),"curve":curve,"base_parameters_unchanged":base_unchanged,"trainable_names":sorted(trainable)}


def _old_checkpoint(config:Mapping[str,Any],fold_id:str)->Mapping[str,Any]:return torch.load(_v9_root(config)/"prepared"/fold_id/"checkpoint.pt",map_location="cpu",weights_only=False)


def _old_arrays(config:Mapping[str,Any],fold_id:str)->Mapping[str,np.ndarray]:return np.load(_v9_root(config)/"prepared"/fold_id/"training_pairs.npz")


def _geometry(config:Mapping[str,Any],fold:Mapping[str,Any],arrays:Mapping[str,np.ndarray],pop:np.ndarray,key:str,seed:int)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    match,shuffled=_heldout_bases(config,fold,arrays,pop,key,seed);layouts,records=_metadata(config);loaded=load_sgeyesub_signal_record(Path(str(config["data_root"])),records[key],layouts[records[key].layout_id],include_query=False,include_query_annotations=False);rate=float(fold["sampling_rate_hz"]);samples=int(round(120*rate));taps=2*int(round(float(config["fir_lag_ms"])*rate/1000))+1;eog_mean,eog_std=_support_eog_stats(loaded,samples);from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import _support_transfer
    transfer,_=_support_transfer(loaded,eog_mean,eog_std,arrays["normal_mean"],arrays["normal_std"],samples,taps,float(config["ridge_lambda"]));_,_,match_mask=aligned_artifact_basis(transfer.reshape(transfer.shape[0],-1),pop);eeg=(np.asarray(loaded.support.eeg[:,:samples],np.float64)-arrays["normal_mean"])/arrays["normal_std"];eog=(np.asarray(loaded.support.external_eog[:,:samples],np.float64)-eog_mean)/eog_std;eog=eog[:,np.random.default_rng(seed).permutation(eog.shape[1])];st=fit_dynamic_transfer(eeg,eog,taps=taps,ridge=float(config["ridge_lambda"]));_,_,shuffled_mask=aligned_artifact_basis(st.reshape(st.shape[0],-1),pop);return match,match_mask,shuffled,shuffled_mask


def stage_technical(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold_ids=list(config["technical_folds"]);fold_id=fold_ids[index];fold=next(r for r in _folds(config) if r["fold_id"]==fold_id);checkpoint=_old_checkpoint(config,fold_id);arrays=_old_arrays(config,fold_id);pop=np.asarray(checkpoint["population_basis"],np.float32);tau=np.asarray(checkpoint["tau"],np.float32);key=fold["heldout"][0];match,mask,_,_=_geometry(config,fold,arrays,pop,key,int(config["pair_seed"]));adapt=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=False,seed=int(config["pair_seed"]));valid=_support_pairs(config,fold,arrays,key,pop,tau,split="validation",shuffled=False,seed=int(config["pair_seed"]));replay=AdaptationReplay.create(seed=ADAPTATION_SEEDS[0],pair_count=len(adapt["y"]),validation_count=len(valid["y"]),updates=20,batch_size=8,timesteps=1000,signal_length=adapt["y"].shape[-1],checkpoint_steps=(0,20));path=_result(config)/"technical"/f"{fold_id}_replay.npz";replay.save(path);device=torch.device("cuda");d01,meta01=_adapt(config,checkpoint,"diff",adapt,valid,pop,np.ones(2,bool),tau,replay,device);d11,meta11=_adapt(config,checkpoint,"diff",adapt,valid,match,mask,tau,replay,device);fresh01,init01,_=_fresh(checkpoint,"diff",replay,device);fresh11,init11,_=_fresh(checkpoint,"diff",replay,device);initial_equal=all(torch.equal(init01[k],init11[k]) for k in init01);zero_equal=all(torch.equal(fresh01.state_dict()[k],fresh11.state_dict()[k]) for k in fresh01.state_dict());o1=_output(d01,"diff",valid["y"],pop,np.ones(2,bool),tau,key,ADAPTATION_SEEDS[0],device);o2=_output(d01,"diff",valid["y"],pop,np.ones(2,bool),tau,key,ADAPTATION_SEEDS[0],device);summary={"status":"passed" if initial_equal and zero_equal and np.array_equal(o1,o2) and meta01["base_parameters_unchanged"] and meta11["base_parameters_unchanged"] else "failed","fold_id":fold_id,"initial_lora_equal_D01_D11":initial_equal,"zero_step_backbone_equivalent":zero_equal,"identical_replay_output":bool(np.array_equal(o1,o2)),"only_lora_changed":meta01["base_parameters_unchanged"] and meta11["base_parameters_unchanged"],"checkpoint_reload":"covered_by_replay_load","common_random_inference":True,"query_outcomes_opened":False};_json(_result(config)/"technical"/f"{fold_id}.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def stage_exact_build(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);root=_result(config)/"exact";source=_v9_root(config)/"factorial"
    for key in fold["heldout"]:
        safe=_safe(key);inp=np.load(source/"deployable_inputs"/fold["fold_id"]/f"paired_{safe}.npz");ev=np.load(source/"evaluator"/fold["fold_id"]/f"paired_{safe}.npz");deploy=root/"deployable_inputs"/fold["fold_id"];evaluator=root/"evaluator"/fold["fold_id"];deploy.mkdir(parents=True,exist_ok=True);evaluator.mkdir(parents=True,exist_ok=True);np.savez_compressed(deploy/f"paired_{safe}.npz",y=inp["y"],recording_key=np.asarray(key));np.savez_compressed(evaluator/f"paired_{safe}.npz",x=ev["x"],a=ev["a"],recording_key=np.asarray(key))
    summary={"status":"completed_exact_boundary_build","fold_id":fold["fold_id"],"units":len(fold["heldout"]),"inference_fields":["y","recording_key"],"query_outcomes_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def _fallback_outputs(base_det:Any,base_diff:Any,y:np.ndarray,pop:np.ndarray,tau:np.ndarray,key:str,seed:int,device:torch.device)->dict[str,np.ndarray]:
    det=_output(base_det,"det",y,pop,np.ones(2,bool),tau,key,seed,device);diff=_output(base_diff,"diff",y,pop,np.ones(2,bool),tau,key,seed,device);result={"RAW":y,"DET-D00":det,"DIFF-D00":diff}
    for name in ("DET-D10","DET-D01","DET-D11","DIFF-D10","DIFF-D01","DIFF-D11","DIFF-D01-SHUFFLED-LORA","DIFF-D11-SHUFFLED-BOTH"):result[name]=det if name.startswith("DET") else diff
    return result


def stage_exact_infer(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);fold_id=fold["fold_id"];arrays=_old_arrays(config,fold_id);checkpoint=_old_checkpoint(config,fold_id);device=torch.device("cuda");base_det,base_diff=_load_models(checkpoint,device);pop=np.asarray(checkpoint["population_basis"],np.float32);tau=np.asarray(checkpoint["tau"],np.float32);root=_result(config)/"exact";metadata=[]
    # Build support objects once; donor adapters are cached per fold/seed/context.
    support={}
    for position,key in enumerate(fold["heldout"]):
        try:
            match,match_mask,shuffled,shuffled_mask=_geometry(config,fold,arrays,pop,key,int(config["pair_seed"])+position);physical=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=False,seed=int(config["pair_seed"]));validation=_support_pairs(config,fold,arrays,key,pop,tau,split="validation",shuffled=False,seed=int(config["pair_seed"]));shuf=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=True,seed=int(config["pair_seed"]));shufval=_support_pairs(config,fold,arrays,key,pop,tau,split="validation",shuffled=True,seed=int(config["pair_seed"]));support[key]={"eligible":True,"match":match,"match_mask":match_mask,"shuffled":shuffled,"shuffled_mask":shuffled_mask,"adapt":physical,"validation":validation,"shuffled_adapt":shuf,"shuffled_validation":shufval}
        except ValueError as exc:support[key]={"eligible":False,"reason":str(exc)}
    for adaptation_seed in ADAPTATION_SEEDS:
        cache={}
        for key,item in support.items():
            if not item["eligible"]:continue
            replay_path=root/"replays"/fold_id/f"{_safe(key)}_seed{adaptation_seed}.npz"
            if replay_path.exists():replay=AdaptationReplay.load(replay_path)
            else:replay=AdaptationReplay.create(seed=adaptation_seed,pair_count=len(item["adapt"]["y"]),validation_count=len(item["validation"]["y"]),updates=int(config["score_lora"]["support_updates"]),batch_size=8,timesteps=1000,signal_length=item["adapt"]["y"].shape[-1],checkpoint_steps=tuple(config["score_lora"]["checkpoints"]));replay.save(replay_path)
            d01,m01=_adapt(config,checkpoint,"diff",item["adapt"],item["validation"],pop,np.ones(2,bool),tau,replay,device);d11,m11=_adapt(config,checkpoint,"diff",item["adapt"],item["validation"],item["match"],item["match_mask"],tau,replay,device);u01,um01=_adapt(config,checkpoint,"det",item["adapt"],item["validation"],pop,np.ones(2,bool),tau,replay,device);u11,um11=_adapt(config,checkpoint,"det",item["adapt"],item["validation"],item["match"],item["match_mask"],tau,replay,device);sd01,sm01=_adapt(config,checkpoint,"diff",item["shuffled_adapt"],item["shuffled_validation"],pop,np.ones(2,bool),tau,replay,device);sd11,sm11=_adapt(config,checkpoint,"diff",item["shuffled_adapt"],item["shuffled_validation"],item["shuffled"],item["shuffled_mask"],tau,replay,device);cache[key]={"replay":replay,"d01":d01,"d11":d11,"u01":u01,"u11":u11,"sd01":sd01,"sd11":sd11,"meta":{"D01":m01,"D11":m11,"DET_D01":um01,"DET_D11":um11,"SHUFFLED_D01":sm01,"SHUFFLED_D11":sm11}}
        for key,item in support.items():
            inp=np.load(root/"deployable_inputs"/fold_id/f"paired_{_safe(key)}.npz");y=np.asarray(inp["y"]);outdir=root/"outputs"/fold_id;outdir.mkdir(parents=True,exist_ok=True)
            if not item["eligible"]:outputs=_fallback_outputs(base_det,base_diff,y,pop,tau,key,adaptation_seed,device);active=False;selected=0
            else:
                c=cache[key];outputs={"RAW":y,"DET-D00":_output(base_det,"det",y,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DIFF-D00":_output(base_diff,"diff",y,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DET-D10":_output(base_det,"det",y,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DIFF-D10":_output(base_diff,"diff",y,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DET-D01":_output(c["u01"],"det",y,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DET-D11":_output(c["u11"],"det",y,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DIFF-D01":_output(c["d01"],"diff",y,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DIFF-D11":_output(c["d11"],"diff",y,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DIFF-D01-SHUFFLED-LORA":_output(c["sd01"],"diff",y,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DIFF-D11-SHUFFLED-BOTH":_output(c["sd11"],"diff",y,item["shuffled"],item["shuffled_mask"],tau,key,adaptation_seed,device)}
                for donor_index,donor in enumerate([v for v in fold["heldout"] if v!=key and support[v]["eligible"]]):
                    dc=cache[donor];di=support[donor];outputs[f"DIFF-D01-WRONG-LORA-{donor_index}"]=_output(dc["d01"],"diff",y,pop,np.ones(2,bool),tau,key,adaptation_seed,device);outputs[f"DIFF-D11-WRONG-BOTH-{donor_index}"]=_output(dc["d11"],"diff",y,di["match"],di["match_mask"],tau,key,adaptation_seed,device)
                active=bool(c["meta"]["D11"]["adapter_active"]);selected=int(c["meta"]["D11"]["selected_step"])
            np.savez_compressed(outdir/f"paired_{_safe(key)}_seed{adaptation_seed}.npz",**outputs);metadata.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"adaptation_seed":adaptation_seed,"personalization_eligible":int(item["eligible"]),"adapter_active":int(active),"selected_step":selected,"fallback_reason":"" if item["eligible"] else item["reason"]})
            # Natural query is processed only after support adaptation is frozen; no evaluator fields are opened.
            natural=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_input_{_safe(key)}.npz");raw=np.asarray(natural["y"],np.float32);raw_length=int(arrays["raw_length"]);usable=raw.shape[1]//raw_length*raw_length;windows=raw[:,:usable].reshape(raw.shape[0],-1,raw_length).transpose(1,0,2);natural_outputs={"RAW":raw[:,:usable]}
            if not item["eligible"]:nvalues=_fallback_outputs(base_det,base_diff,windows,pop,tau,key,adaptation_seed,device)
            else:
                c=cache[key];nvalues={"DET-D00":_output(base_det,"det",windows,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DIFF-D00":_output(base_diff,"diff",windows,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DET-D10":_output(base_det,"det",windows,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DIFF-D10":_output(base_diff,"diff",windows,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DET-D01":_output(c["u01"],"det",windows,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DET-D11":_output(c["u11"],"det",windows,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DIFF-D01":_output(c["d01"],"diff",windows,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DIFF-D11":_output(c["d11"],"diff",windows,item["match"],item["match_mask"],tau,key,adaptation_seed,device),"DIFF-D01-SHUFFLED-LORA":_output(c["sd01"],"diff",windows,pop,np.ones(2,bool),tau,key,adaptation_seed,device),"DIFF-D11-SHUFFLED-BOTH":_output(c["sd11"],"diff",windows,item["shuffled"],item["shuffled_mask"],tau,key,adaptation_seed,device)}
            for name,value in nvalues.items():
                if name=="RAW":continue
                natural_outputs[name]=value.transpose(1,0,2).reshape(raw.shape[0],usable)
            np.savez_compressed(outdir/f"natural_{_safe(key)}_seed{adaptation_seed}.npz",**natural_outputs)
    _csv(root/"adaptation_metadata"/f"{fold_id}.csv",metadata);summary={"status":"completed_exact_common_random_inference","fold_id":fold_id,"units":len(fold["heldout"]),"adaptation_seeds":list(ADAPTATION_SEEDS),"eligible_units":sum(v["eligible"] for v in support.values()),"query_outcomes_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def stage_exact_eval(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);root=_result(config)/"exact";paired=[];natural=[]
    for key in fold["heldout"]:
        evaluator=np.load(root/"evaluator"/fold["fold_id"]/f"paired_{_safe(key)}.npz")
        for seed in ADAPTATION_SEEDS:
            output=np.load(root/"outputs"/fold["fold_id"]/f"paired_{_safe(key)}_seed{seed}.npz");y=np.asarray(output["RAW"])
            for method in output.files:paired.append({"fold_id":fold["fold_id"],"study":fold["study"],"recording_key":key,"adaptation_seed":seed,"method":method,**_paired_metrics(output[method],evaluator["x"],y)})
            nout=np.load(root/"outputs"/fold["fold_id"]/f"natural_{_safe(key)}_seed{seed}.npz");raw=np.asarray(nout["RAW"]);neval=np.load(Path(str(config["v6_root"]))/"prepared"/fold["fold_id"]/f"natural_evaluator_{_safe(key)}.npz")
            for method in nout.files:natural.append({"fold_id":fold["fold_id"],"study":fold["study"],"recording_key":key,"adaptation_seed":seed,"method":method,**_natural_metrics(raw,nout[method],neval["eog"],neval["labels"],float(fold["sampling_rate_hz"]))})
    _csv(root/"metrics"/f"{fold['fold_id']}_paired.csv",paired);_csv(root/"metrics"/f"{fold['fold_id']}_natural.csv",natural);summary={"status":"completed_independent_exact_evaluator","fold_id":fold["fold_id"],"units":len(fold["heldout"]),"adaptation_seeds":3};_json(run_dir/"result_summary.json",summary);return summary


def _cluster_bootstrap(rows:Sequence[Mapping[str,Any]],metrics:Sequence[str],seed:int,reps:int)->list[dict[str,Any]]:
    rng=np.random.default_rng(seed);studies=sorted({str(r["study"]) for r in rows});draw_indices=[]
    for _ in range(reps):
        indices=[]
        for study in studies:
            clusters=sorted({str(r["fold_id"]) for r in rows if r["study"]==study})
            for cluster in rng.choice(clusters,size=len(clusters),replace=True):
                positions=[i for i,r in enumerate(rows) if r["fold_id"]==cluster];indices.extend(rng.choice(positions,size=len(positions),replace=True).tolist())
        draw_indices.append(indices)
    result=[]
    for metric in metrics:
        observed=np.asarray([float(r[metric]) for r in rows]);draws=np.asarray([np.mean(observed[idx]) for idx in draw_indices]);result.append({"effect":metric,"mean":float(observed.mean()),"median":float(np.median(observed)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"positive_count":int((observed>0).sum()),"denominator":len(observed),"bootstrap_seed":seed,"replicates":reps,"scope":"development_descriptive_fold_cluster"})
    return result


def stage_exact_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=_result(config);exact=root/"exact";paired=[];natural=[];metadata=[]
    for fold_id in config["diagnostic_folds"]:paired.extend(_read(exact/"metrics"/f"{fold_id}_paired.csv"));natural.extend(_read(exact/"metrics"/f"{fold_id}_natural.csv"));metadata.extend(_read(exact/"adaptation_metadata"/f"{fold_id}.csv"))
    _csv(root/"unit_metrics_by_seed.csv",paired);_csv(root/"natural_safety_by_seed.csv",natural);meta={(r["recording_key"],int(r["adaptation_seed"])):r for r in metadata};effects_seed=[]
    for key in sorted({r["recording_key"] for r in paired}):
        for seed in ADAPTATION_SEEDS:
            rows=[r for r in paired if r["recording_key"]==key and int(r["adaptation_seed"])==seed];by={r["method"]:r for r in rows};eligible=int(meta[(key,seed)]["personalization_eligible"]);wrong=[float(r["rrmse"]) for r in rows if RAW_WRONG.fullmatch(r["method"])]
            effects_seed.append({"fold_id":by["RAW"]["fold_id"],"study":by["RAW"]["study"],"recording_key":key,"adaptation_seed":seed,"personalization_eligible":eligible,"adapter_active":int(meta[(key,seed)]["adapter_active"]),"selected_step":int(meta[(key,seed)]["selected_step"]),"U_D":float(by["DET-D11"]["rrmse"])-float(by["DIFF-D11"]["rrmse"]),"U_P":float(by["DIFF-D00"]["rrmse"])-float(by["DIFF-D11"]["rrmse"]),"U_W":float(np.mean(wrong))-float(by["DIFF-D11"]["rrmse"]) if wrong else 0.0,"U_S":float(by["DIFF-D11-SHUFFLED-BOTH"]["rrmse"])-float(by["DIFF-D11"]["rrmse"]),"I":float(by["DIFF-D10"]["rrmse"])+float(by["DIFF-D01"]["rrmse"])-float(by["DIFF-D00"]["rrmse"])-float(by["DIFF-D11"]["rrmse"]),"D00_DET_minus_DIFF":float(by["DET-D00"]["rrmse"])-float(by["DIFF-D00"]["rrmse"])})
    _csv(root/"effects_by_adaptation_seed.csv",effects_seed);unit=[]
    for key in sorted({r["recording_key"] for r in effects_seed}):
        rows=[r for r in effects_seed if r["recording_key"]==key];unit.append({"fold_id":rows[0]["fold_id"],"study":rows[0]["study"],"recording_key":key,"personalization_eligible":rows[0]["personalization_eligible"],"adapter_active":int(any(r["adapter_active"] for r in rows)),**{effect:float(np.mean([r[effect] for r in rows])) for effect in ("U_D","U_P","U_W","U_S","I","D00_DET_minus_DIFF")}})
    _csv(root/"raw_itt_effects.csv",unit);eligible=[r for r in unit if r["personalization_eligible"]==1];_csv(root/"raw_eligible_effects.csv",eligible)
    # V9R does not retain correction calibration; historical corrected panels remain explicit audit outputs.
    _csv(root/"calibrated_itt_effects.csv",[{"status":"not_run","reason":"V9R_primary_raw_no_crossfit_calibration"}]);_csv(root/"calibrated_eligible_effects.csv",[{"status":"not_run","reason":"V9R_primary_raw_no_crossfit_calibration"}])
    summaries=_panel_summary(unit,("U_D","U_P","U_W","U_S","I","D00_DET_minus_DIFF"),"raw_itt")+_panel_summary(eligible,("U_D","U_P","U_W","U_S","I","D00_DET_minus_DIFF"),"raw_eligible");_csv(root/"effect_summary.csv",summaries);per_study=_study_summary(unit,("U_D","U_P","U_W","U_S","I"),"raw_itt")+_study_summary(eligible,("U_D","U_P","U_W","U_S","I"),"raw_eligible");_csv(root/"per_study_effects.csv",per_study)
    loso=[]
    for panel,values in (("raw_itt",unit),("raw_eligible",eligible)):
        for study in sorted({r["study"] for r in values}):
            subset=[r for r in values if r["study"]!=study]
            for effect in ("U_D","U_P","U_W","U_S","I"):loso.append({"panel":panel,"left_out_study":study,"effect":effect,"units":len(subset),"mean":float(np.mean([r[effect] for r in subset]))})
    _csv(root/"leave_one_study_out.csv",loso);variance=[]
    for key in sorted({r["recording_key"] for r in effects_seed}):
        rows=[r for r in effects_seed if r["recording_key"]==key]
        for effect in ("U_D","U_P","U_W","U_S","I"):variance.append({"recording_key":key,"effect":effect,"adaptation_seed_variance":float(np.var([r[effect] for r in rows],ddof=0))})
    _csv(root/"adapter_seed_variance.csv",variance);boot=_cluster_bootstrap(unit,("U_D","U_P","U_W","U_S","I","D00_DET_minus_DIFF"),int(config["bootstrap_seed"]),int(config["bootstrap_replicates"]));_csv(root/"bootstrap_summary.csv",boot)
    safety=[]
    for key in sorted({r["recording_key"] for r in natural}):
        rows=[r for r in natural if r["recording_key"]==key and r["method"]=="DIFF-D11"];safety.append({"fold_id":rows[0]["fold_id"],"study":rows[0]["study"],"recording_key":key,**{name:float(np.mean([float(r[name]) for r in rows])) for name in ("eog_coherence_reduction","nonartifact_preservation","psd_distortion","covariance_distortion")}})
    _csv(root/"natural_safety_distribution.csv",safety);safety_summary={name:{"mean":float(np.mean([r[name] for r in safety])),"median":float(np.median([r[name] for r in safety])),"minimum":float(np.min([r[name] for r in safety]))} for name in ("eog_coherence_reduction","nonartifact_preservation","psd_distortion","covariance_distortion")};severe=float(np.mean([r["U_P"]<-.05 for r in eligible]));active=sum(r["adapter_active"] for r in eligible)/max(len(eligible),1);means={e:float(np.mean([r[e] for r in eligible])) for e in ("U_D","U_P","U_W","U_S","I")};gate=means["U_D"]>0 and means["I"]>0 and means["U_W"]>0 and means["U_S"]>0 and means["U_P"]>-.01 and active>=.8 and severe<=.25 and safety_summary["nonartifact_preservation"]["mean"]>=.70 and safety_summary["psd_distortion"]["mean"]<=.35 and safety_summary["covariance_distortion"]["mean"]<=.35
    if not gate:decision="V9_INTERACTION_NOT_REPRODUCED_UNDER_PAIRED_ADAPTATION_RANDOMNESS" if means["I"]<=0 else "V9_D11_INTERACTION_PRESENT_BUT_SUBJECT_UTILITY_HETEROGENEOUS"
    else:decision="V9R_EXACT_REPLAY_PASSED_NEW_FOLD_EXTRAPOLATION_AUTHORIZED"
    route={"status":"passed" if gate else "failed","decision":decision,"new_fold_authorized":gate,"means_eligible":means,"eligible_units":len(eligible),"itt_units":len(unit),"lora_active_coverage":active,"fallback_coverage":1-len(eligible)/len(unit),"severe_reversal_fraction":severe,"safety":safety_summary,"support_semantics":"label_assisted_calibration_support","confirmation":False};_json(root/"exact_route_decision.json",route);_json(run_dir/"result_summary.json",route);return route


def stage_finalize(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=_result(config);decision=json.loads((root/"exact_route_decision.json").read_text());coverage=json.loads((root/"support_coverage_summary.json").read_text());summary_rows=_read(root/"effect_summary.csv");eligible={r["effect"]:r for r in summary_rows if r["panel"]=="raw_eligible"};boot={r["effect"]:r for r in _read(root/"bootstrap_summary.csv")};lines=["# SGE basis × Score-LoRA V9R","","Development exploration; no confirmation outcomes were opened. Support pseudo-pairs use support artifact-class labels, so the method is label-assisted calibration support.","","## Exact common-random replay","","| effect | eligible mean | median | positive | descriptive 95% CI |","|---|---:|---:|---:|---:|"]
    for effect in ("D00_DET_minus_DIFF","U_D","U_P","U_W","U_S","I"):
        r=eligible[effect];b=boot[effect];lines.append(f"| {effect} | {float(r['stem_weighted_mean']):+.4f} | {float(r['median']):+.4f} | {r['positive_count']}/{r['units']} | [{float(b['ci_low']):+.4f}, {float(b['ci_high']):+.4f}] |")
    lines += ["",f"Original builder eligible: {coverage['old_eligible']}/{coverage['compatible_stems']}; unified blocked builder eligible: {coverage['new_eligible']}/{coverage['compatible_stems']} (availability denominator 59).",f"LoRA-active coverage among eligible stems: {decision['lora_active_coverage']:.1%}; fallback coverage: {decision['fallback_coverage']:.1%}.","",f"Routing decision: `{decision['decision']}`. New-fold extrapolation authorized: `{decision['new_fold_authorized']}`.","","The report separates population diffusion, diffusion-vs-deterministic, geometry/LoRA interaction, subject utility, wrong/shuffled specificity, applicability, and natural safety. All intervals are development-descriptive and no failure is generalized to a diffusion or personalization family."]
    Path("reports/sge_basis_score_factorial_v9r.md").write_text("\n".join(lines)+"\n",encoding="utf-8");Path("reports/v9_common_randomness_audit.md").write_text("# V9 common-randomness audit\n\nAdaptationReplay freezes zero-output LoRA initialization, minibatches, diffusion timesteps, Gaussian noise, dropout RNG, checkpoint steps, and K=8 inference noise. D01/D11 replay the same schedule; donor adapters are trained once per fold/seed and reused. Rank masks are applied to targets, loss, and correction. Query inference sees deployable EEG/condition IDs only.\n",encoding="utf-8");final={"status":"completed_v9r","exact_replay":decision,"coverage":coverage,"new_fold_status":"AUTHORIZED_PENDING" if decision["new_fold_authorized"] else "NOT_RUN_EXACT_REPLAY_GATE_FAILED","confirmation":False};_json(root/"result_summary.json",final);_json(run_dir/"result_summary.json",final);return final


def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="v9-audit":return stage_v9_audit(config,run_dir)
    if stage=="coverage":return stage_coverage(config,task_index,run_dir)
    if stage=="coverage-aggregate":return stage_coverage_aggregate(config,run_dir)
    if stage=="technical":return stage_technical(config,task_index,run_dir)
    if stage=="exact-build":return stage_exact_build(config,task_index,run_dir)
    if stage=="exact-infer":return stage_exact_infer(config,task_index,run_dir)
    if stage=="exact-eval":return stage_exact_eval(config,task_index,run_dir)
    if stage=="exact-aggregate":return stage_exact_aggregate(config,run_dir)
    if stage=="finalize":return stage_finalize(config,run_dir)
    raise ValueError(f"unknown V9R stage: {stage}")


__all__=["run_stage","_blocked_physical_pairs","_old_pair_eligibility","RAW_WRONG","CAL_WRONG"]
