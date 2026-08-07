"""SGE V9 subject-basis exposure by true internal score-LoRA exploration.

Historical v8/v8.1 files are read-only.  Query inference and evaluator surfaces
are kept separate, while this intentionally remains a research-light runner.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.optim import AdamW

from eeg_cgdr.data.sgeyesub import load_sgeyesub_signal_record
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import _metadata, _support_eog_stats, _support_transfer, fit_dynamic_transfer
from eeg_cgdr.experiments.sge_score_lora_v8 import (
    _clone_state, _condition, _expanded_record_pairs, _folds, _masked_u_mse,
    _natural_metrics, _pad, _rrmse, _support_pseudo_pairs,
    stage_prepare_base as _v8_prepare_base,
)
from eeg_cgdr.experiments.sge_eb_bridge_v8_1 import _basis_from_transfer, _model_output, _paired_metrics
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig, ArtifactSubspaceDiffusion,
    DeterministicSubspaceEstimator, aligned_artifact_basis,
    reconstruct_from_subspace, training_tau, window_noise_bank,
)
from eeg_cgdr.models.artifact_subspace_score_lora import inject_score_lora, lora_state_dict


PROTOCOL="SGE-BASIS-SCORE-FACTORIAL-v9"


def _json_default(value:Any)->Any:
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)


def _json(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(value),indent=2,allow_nan=False,default=_json_default)+"\n",encoding="utf-8")


def _csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    if not rows: raise ValueError(f"empty table: {path}")
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _config(path:Path)->dict[str,Any]:
    value=yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("protocol_id")!=PROTOCOL or int(value.get("harness_level",-1))!=1: raise ValueError("wrong V9 protocol")
    return value


def _fold(config:Mapping[str,Any],index:int)->dict[str,Any]:
    fold_id=str(config["diagnostic_folds"][index])
    return next(row for row in _folds(config) if row["fold_id"]==fold_id)


def _folder(config:Mapping[str,Any],fold_id:str)->Path:
    return Path(str(config["result_root"]))/"prepared"/fold_id


def _support_geometries(config:Mapping[str,Any],fold:Mapping[str,Any],arrays:Mapping[str,np.ndarray])->dict[str,Any]:
    layouts,records=_metadata(config);data_root=Path(str(config["data_root"]));rate=float(fold["sampling_rate_hz"])
    samples=int(round(float(config["support_budget_seconds"])*rate));taps=2*int(round(float(config["fir_lag_ms"])*rate/1000))+1
    transfers=[];keys=[]
    for key in fold["training"]:
        loaded=load_sgeyesub_signal_record(data_root,records[key],layouts[records[key].layout_id],include_query=False,include_query_annotations=False)
        eog_mean,eog_std=_support_eog_stats(loaded,samples)
        transfer,_=_support_transfer(loaded,eog_mean,eog_std,arrays["normal_mean"],arrays["normal_std"],samples,taps,float(config["ridge_lambda"]))
        transfers.append(transfer.reshape(transfer.shape[0],-1));keys.append(key)
    population_transfer=np.mean(np.stack(transfers),axis=0);population_basis,_,population_mask=aligned_artifact_basis(population_transfer)
    bases=[];masks=[];singular=[]
    for transfer in transfers:
        basis,values,mask=aligned_artifact_basis(transfer,population_basis);bases.append(basis);masks.append(mask);singular.append(values)
    return {"keys":np.asarray(keys),"transfers":np.stack(transfers).astype(np.float32),"population_transfer":population_transfer.astype(np.float32),"population_basis":population_basis,"population_mask":population_mask,"bases":np.stack(bases),"masks":np.stack(masks),"singular":np.stack(singular),"support_samples":samples,"taps":taps}


def stage_audit(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    folds=[_fold(config,i) for i in range(len(config["diagnostic_folds"]))];rows=[]
    for fold in folds:
        for position,key in enumerate(fold["heldout"]):
            donors=[v for v in fold["heldout"] if v!=key]
            rows.append({"fold_id":fold["fold_id"],"study":fold["study"],"recording_key":key,"support_seconds":120,"wrong_donors":";".join(donors),"wrong_donor_count":len(donors),"geometry_primary":"RAW_MATCH120","geometry_secondary":"FIXED_EB120","query_outcomes_allowed_in_inference":False})
    root=Path(str(config["result_root"]));_csv(root/"frozen_protocol_units.csv",rows)
    correction={"historical_v8_1_decision":"BRIDGE_GATE_FAILED_SCORE_LORA_NOT_RUN","historical_decision_unchanged":True,"score_lora":"NOT_TESTED","eb_operator_proxy":"not_equivalent_to_denoising_geometry","diff_match120":"candidate_signal_not_established_advantage","bootstrap_seed":int(config["bootstrap_seed"]),"primary_geometry":"RAW_MATCH120","baseline_geometry":"POP","secondary_geometry":"FIXED_LAMBDA_EB120","full_feature_eb_predictor":"STOPPED","representation_fallback":"support_pair_coverage_only","confirmation_evidence":False}
    _json(root/"v8_1_scientific_correction.json",correction);Path("reports/v8_1_scientific_correction.md").write_text("# v8.1 scientific correction for V9\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in correction.items())+"\n",encoding="utf-8")
    summary={"status":"completed_v9_audit","diagnostic_folds":len(folds),"diagnostic_units":len(rows),"studies":sorted({r["study"] for r in rows}),"bootstrap_seed":int(config["bootstrap_seed"])};_json(run_dir/"result_summary.json",summary);return summary


def stage_prepare(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);fold_id=fold["fold_id"];root=Path(str(config["result_root"]));destination=_folder(config,fold_id)
    # Reuse the verified real-pair builder, but write only into the V9 result root.
    local=dict(config);local["result_root"]=str(root);local["diagnostic_folds"]=list(config["diagnostic_folds"])
    _v8_prepare_base(local,index,run_dir)
    source=root/"prepared_base"/fold_id
    destination.mkdir(parents=True,exist_ok=True)
    for name in ("training_pairs.npz","expanded_pair_manifest.csv","result_summary.json"):
        (destination/name).write_bytes((source/name).read_bytes())
    arrays=np.load(destination/"training_pairs.npz");geometry=_support_geometries(config,fold,arrays)
    coefficient=np.einsum("cr,nct->nrt",geometry["population_basis"].astype(np.float64),arrays["a"].astype(np.float64));tau=training_tau(coefficient)
    np.savez_compressed(destination/"subject_geometry.npz",**geometry,tau=tau)
    counts={str(k):int(np.sum(arrays["key"]==k)) for k in geometry["keys"]}
    summary={"status":"prepared_subject_basis_exposure","fold_id":fold_id,"training_participants":len(geometry["keys"]),"pairs":len(arrays["y"]),"support_seconds":120,"basis_sampling":{"POP":.5,"SUBJECT":.5},"pair_counts":counts,"query_outcomes_opened":False};_json(destination/"v9_prepare_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def _cfg(config:Mapping[str,Any],arrays:Mapping[str,np.ndarray])->ArtifactSubspaceConfig:
    return ArtifactSubspaceConfig(eeg_channels=arrays["y"].shape[1],signal_length=arrays["y"].shape[2],base_channels=int(config["model"]["base_channels"]),num_timesteps=int(config["model"]["timesteps"]),min_snr_gamma=float(config["model"]["min_snr_gamma"]),ddim_steps=25,posterior_samples=8)


def _basis_batch(indices:np.ndarray,keys:np.ndarray,geometry:Mapping[str,np.ndarray],use_subject:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    lookup={str(k):i for i,k in enumerate(geometry["keys"])};bases=[];masks=[]
    for row,subject in zip(indices,use_subject):
        if subject:
            position=lookup[str(keys[int(row)])];bases.append(geometry["bases"][position]);masks.append(geometry["masks"][position])
        else:
            bases.append(geometry["population_basis"]);masks.append(geometry["population_mask"])
    return np.stack(bases).astype(np.float32),np.stack(masks).astype(bool)


def _condition_batch(y:torch.Tensor,basis:torch.Tensor,mask:torch.Tensor,valid:torch.Tensor)->dict[str,torch.Tensor]:
    return {"observed":y,"basis":basis,"reliability":torch.ones(len(y),device=y.device),"rank_mask":mask,"valid_time_mask":valid}


def stage_train(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);folder=_folder(config,fold["fold_id"]);arrays=np.load(folder/"training_pairs.npz");geometry=np.load(folder/"subject_geometry.npz");device=torch.device("cuda");seed=int(config["training"]["seed"])+index
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed);cfg=_cfg(config,arrays);det=DeterministicSubspaceEstimator(cfg).to(device);diff=ArtifactSubspaceDiffusion(cfg).to(device)
    det_opt=AdamW(det.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"]));diff_opt=AdamW(diff.parameters(),lr=float(config["training"]["learning_rate"]),weight_decay=float(config["training"]["weight_decay"]));noise_rng=torch.Generator(device=device).manual_seed(seed+991);ema=_clone_state(diff);curve=[];start=0
    checkpoint=folder/"checkpoint.pt"
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location="cpu",weights_only=False);start=int(state["step"]);det.load_state_dict(state["det"]);diff.load_state_dict(state["diff"]);det_opt.load_state_dict(state["det_optimizer"]);diff_opt.load_state_dict(state["diff_optimizer"]);ema=state["diff_ema"];noise_rng.set_state(state["generator_state"]);rng.bit_generator.state=state["numpy_state"]
    batch=int(config["training"]["batch_size"]);updates=int(config["training"]["successful_updates"]);tau=torch.tensor(geometry["tau"],device=device)
    for step in range(start+1,updates+1):
        ids=rng.integers(0,len(arrays["y"]),size=batch);subject=rng.random(batch)<.5;basis_np,mask_np=_basis_batch(ids,arrays["key"],geometry,subject);y=torch.tensor(arrays["y"][ids],device=device);a=torch.tensor(arrays["a"][ids],device=device);valid=torch.tensor(arrays["valid"][ids],device=device);basis=torch.tensor(basis_np,device=device);mask=torch.tensor(mask_np,device=device);target=torch.tanh(torch.einsum("bcr,bct->brt",basis,a)/tau[None,:,None]);condition=_condition_batch(y,basis,mask,valid)
        det_opt.zero_grad(set_to_none=True);dloss=_masked_u_mse(det(**condition),target,valid);dloss.backward();dg=float(torch.nn.utils.clip_grad_norm_(det.parameters(),1));det_opt.step()
        diff_opt.zero_grad(set_to_none=True);floss,detail=diff.training_loss(target,generator=noise_rng,**condition);floss.backward();fg=float(torch.nn.utils.clip_grad_norm_(diff.parameters(),1));diff_opt.step()
        with torch.no_grad():
            for name,value in diff.state_dict().items():ema[name].mul_(float(config["training"]["ema_decay"])).add_(value.detach().cpu(),alpha=1-float(config["training"]["ema_decay"]))
        if step==1 or step%100==0:curve.append({"step":step,"det_loss":float(dloss.detach()),"diff_loss":float(floss.detach()),"x0_mse":float(detail["u_mse"]),"det_grad":dg,"diff_grad":fg,"subject_basis_fraction":float(subject.mean())})
        if step%int(config["training"]["checkpoint_interval"])==0 or step==updates:
            torch.save({"step":step,"det":det.state_dict(),"diff":diff.state_dict(),"diff_ema":ema,"det_optimizer":det_opt.state_dict(),"diff_optimizer":diff_opt.state_dict(),"generator_state":noise_rng.get_state(),"numpy_state":rng.bit_generator.state,"model_config":cfg.__dict__,"tau":geometry["tau"],"population_basis":geometry["population_basis"],"protocol":PROTOCOL},checkpoint)
    _csv(folder/"training_curve.csv",curve);summary={"status":"completed_basis_exposed_training","fold_id":fold["fold_id"],"updates":updates,"pairs":len(arrays["y"]),"checkpoint":str(checkpoint),"basis_sampling":"POP_0.5_SUBJECT_0.5"};_json(run_dir/"result_summary.json",summary);return summary


def stage_technical(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    fold=_fold(config,1);folder=_folder(config,fold["fold_id"]);arrays=np.load(folder/"training_pairs.npz");geometry=np.load(folder/"subject_geometry.npz");device=torch.device("cuda");cfg=_cfg(config,arrays);seed=int(config["training"]["seed"])
    keys=list(map(str,geometry["keys"][:3]));ids=np.asarray([int(np.flatnonzero(arrays["key"]==key)[0]) for key in keys]);basis_np=np.asarray(geometry["bases"][:len(ids)]);mask_np=np.asarray(geometry["masks"][:len(ids)]);y=torch.tensor(arrays["y"][ids],device=device);a=torch.tensor(arrays["a"][ids],device=device);valid=torch.tensor(arrays["valid"][ids],device=device);basis=torch.tensor(basis_np,device=device);mask=torch.tensor(mask_np,device=device);tau=torch.tensor(geometry["tau"],device=device);target=torch.tanh(torch.einsum("bcr,bct->brt",basis,a)/tau[None,:,None]);model=ArtifactSubspaceDiffusion(cfg).to(device);condition=_condition_batch(y,basis,mask,valid);t=torch.full((len(y),),750,device=device,dtype=torch.long);noise=torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed));opt=AdamW(model.parameters(),lr=5e-4);model.eval();initial=float(model.training_loss(target,generator=torch.Generator(device=device).manual_seed(seed),timestep=t,noise=noise,**condition)[0]);model.train()
    for _ in range(3000):opt.zero_grad(set_to_none=True);loss,_=model.training_loss(target,generator=torch.Generator(device=device).manual_seed(seed),timestep=t,noise=noise,**condition);loss.backward();opt.step()
    model.eval();final=float(model.training_loss(target,generator=torch.Generator(device=device).manual_seed(seed),timestep=t,noise=noise,**condition)[0]);with_pop=dict(condition);with_pop["basis"]=torch.tensor(np.repeat(geometry["population_basis"][None],len(y),axis=0),device=device);response=float(torch.linalg.norm(model.backbone(torch.zeros_like(target),t,**condition)-model.backbone(torch.zeros_like(target),t,**with_pop)))
    lora_model=ArtifactSubspaceDiffusion(cfg).to(device);lora=inject_score_lora(lora_model.backbone,rank=4);initial_lora={k:v.clone() for k,v in lora_state_dict(lora_model).items()};lopt=AdamW([p for p in lora_model.parameters() if p.requires_grad],lr=1e-4);lopt.zero_grad(set_to_none=True);lloss,_=lora_model.training_loss(target,generator=torch.Generator(device=device).manual_seed(seed),**condition);lloss.backward();lopt.step();updated=any(not torch.equal(v,lora_state_dict(lora_model)[k]) for k,v in initial_lora.items())
    bank=window_noise_bank(keys[0],seed,range(len(y)),posterior_samples=8,signal_length=y.shape[-1],device=device);split=torch.cat((window_noise_bank(keys[0],seed,range(1),posterior_samples=8,signal_length=y.shape[-1],device=device),window_noise_bank(keys[0],seed,range(1,len(y)),posterior_samples=8,signal_length=y.shape[-1],device=device)),dim=1);torch.testing.assert_close(bank,split)
    probe=run_dir/"technical_checkpoint.pt";torch.save({"model":model.state_dict()},probe);reloaded=ArtifactSubspaceDiffusion(cfg).to(device);reloaded.load_state_dict(torch.load(probe,map_location=device,weights_only=True)["model"])
    summary={"status":"passed" if final<=.01*initial and response>1e-6 and updated else "failed","real_training_subject_bases":keys,"fixed_batch_loss_reduction":1-final/max(initial,1e-12),"basis_context_response":response,"lora":{"rank":lora.rank,"adapted_convolutions":lora.adapted_convolutions,"trainable_parameters":lora.trainable_parameters,"updated":updated},"unique_window_rng":True,"checkpoint_reload":True,"query_outcomes_opened":False};_json(run_dir/"result_summary.json",summary);_json(Path(str(config["result_root"]))/"technical_validity.json",summary);return summary


def _track_a_fold(config:Mapping[str,Any],index:int)->dict[str,Any]:
    fold_id=str(config["track_a_folds"][index]);return next(row for row in _folds(config) if row["fold_id"]==fold_id)


def _v8_checkpoint_folder(config:Mapping[str,Any],fold_id:str)->Path:
    bridge=Path(str(config["v8_1_root"]))/"prepared_base"/fold_id
    return bridge if (bridge/"checkpoint.pt").exists() else Path(str(config["v8_root"]))/"prepared_base"/fold_id


def stage_track_a_build(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_track_a_fold(config,index);root=Path(str(config["result_root"]))/"track_a";source=Path(str(config["v6_root"]))/"prepared"/fold["fold_id"]
    for key in fold["heldout"]:
        safe=key.replace("/","__");pair=np.load(source/f"paired_{safe}.npz");deploy=root/"deployable_inputs"/fold["fold_id"];evaluator=root/"evaluator"/fold["fold_id"];deploy.mkdir(parents=True,exist_ok=True);evaluator.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(deploy/f"paired_{safe}.npz",y=pair["y"],recording_key=np.asarray(key));np.savez_compressed(evaluator/f"paired_{safe}.npz",x=pair["x"],a=pair["a"],recording_key=np.asarray(key))
    summary={"status":"completed_track_a_boundary_build","fold_id":fold["fold_id"],"units":len(fold["heldout"]),"deployable_fields":["y","recording_key"],"evaluator_fields":["x","a","recording_key"]};_json(run_dir/"result_summary.json",summary);return summary


def _load_models(checkpoint:Mapping[str,Any],device:torch.device)->tuple[Any,Any]:
    cfg=ArtifactSubspaceConfig(**checkpoint["model_config"]);det=DeterministicSubspaceEstimator(cfg).to(device);diff=ArtifactSubspaceDiffusion(cfg).to(device);det.load_state_dict(checkpoint["det"]);diff.load_state_dict(checkpoint["diff_ema"]);det.eval();diff.eval();return det,diff


def stage_track_a_infer(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_track_a_fold(config,index);fold_id=fold["fold_id"];root=Path(str(config["result_root"]))/"track_a";folder=_v8_checkpoint_folder(config,fold_id);checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=False);arrays=np.load(folder/"training_pairs.npz");device=torch.device("cuda");det,diff=_load_models(checkpoint,device);tau=np.asarray(checkpoint["tau"],np.float32);pop_basis=np.asarray(checkpoint["population_basis"],np.float32);eb=Path(str(config["v8_1_root"]))/"eb";builder=eb/"inference_builder"/"120s"/fold_id;heldout=eb/"heldout_inference"/"120s"/fold_id;seed=int(config["training"]["seed"])
    for key in fold["heldout"]:
        safe=key.replace("/","__");inp=np.load(root/"deployable_inputs"/fold_id/f"paired_{safe}.npz");candidate=np.load(heldout/f"{safe}.npz");support=np.load(builder/f"{safe}.npz");basis={"POP":pop_basis,"MATCH120":_basis_from_transfer(support["support"],pop_basis),"EB-FIXED120":_basis_from_transfer(candidate["fixed"],pop_basis),"RAW-SHUFFLED120":_basis_from_transfer(support["shuffled"],pop_basis)}
        donors=[]
        for donor in fold["heldout"]:
            if donor==key:continue
            donor_item=np.load(builder/f"{donor.replace('/','__')}.npz");donors.append((donor,_basis_from_transfer(donor_item["support"],pop_basis)))
        output={"RAW":np.asarray(inp["y"])}
        for name,b in basis.items():
            output[f"DET-{name}"]=_model_output(det,"det",inp["y"],b,tau,key,seed,device,k=8);output[f"DIFF-{name}"]=_model_output(diff,"diff",inp["y"],b,tau,key,seed,device,k=8)
        for donor_index,(donor,b) in enumerate(donors):output[f"DIFF-RAW-WRONG120-{donor_index}"]=_model_output(diff,"diff",inp["y"],b,tau,key,seed,device,k=8)
        destination=root/"outputs"/fold_id;destination.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination/f"paired_{safe}.npz",**output)
        natural=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_input_{safe}.npz");raw=np.asarray(natural["y"],np.float32);raw_length=int(arrays["raw_length"]);usable=raw.shape[1]//raw_length*raw_length;windows=raw[:,:usable].reshape(raw.shape[0],-1,raw_length).transpose(1,0,2);natural_output={"RAW":raw[:,:usable]}
        for name,b in basis.items():natural_output[f"DET-{name}"]=_model_output(det,"det",windows,b,tau,key,seed,device,k=8).transpose(1,0,2).reshape(raw.shape[0],usable);natural_output[f"DIFF-{name}"]=_model_output(diff,"diff",windows,b,tau,key,seed,device,k=8).transpose(1,0,2).reshape(raw.shape[0],usable)
        for donor_index,(donor,b) in enumerate(donors):natural_output[f"DIFF-RAW-WRONG120-{donor_index}"]=_model_output(diff,"diff",windows,b,tau,key,seed,device,k=8).transpose(1,0,2).reshape(raw.shape[0],usable)
        np.savez_compressed(destination/f"natural_{safe}.npz",**natural_output)
    summary={"status":"completed_track_a_inference","fold_id":fold_id,"units":len(fold["heldout"]),"K":8,"outcomes_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def stage_track_a_eval(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_track_a_fold(config,index);fold_id=fold["fold_id"];root=Path(str(config["result_root"]))/"track_a";paired_rows=[];natural_rows=[];strength_rows=[]
    for key in fold["heldout"]:
        safe=key.replace("/","__");outputs=np.load(root/"outputs"/fold_id/f"paired_{safe}.npz");evaluator=np.load(root/"evaluator"/fold_id/f"paired_{safe}.npz");y=np.asarray(outputs["RAW"])
        for method in outputs.files:
            value=np.asarray(outputs[method]);paired_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,**_paired_metrics(value,evaluator["x"],y)})
        natural=np.load(root/"outputs"/fold_id/f"natural_{safe}.npz");raw=np.asarray(natural["RAW"]);natural_eval=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_evaluator_{safe}.npz")
        for method in natural.files:
            value=np.asarray(natural[method]);metrics=_natural_metrics(raw,value,natural_eval["eog"],natural_eval["labels"],float(fold["sampling_rate_hz"]));natural_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,**metrics})
            if method.startswith("DIFF-"):
                correction=raw-value
                for gamma in config["correction_strengths"]:
                    scaled=raw-float(gamma)*correction;strength_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,"gamma":gamma,**_natural_metrics(raw,scaled,natural_eval["eog"],natural_eval["labels"],float(fold["sampling_rate_hz"]))})
    destination=root/"metrics";_csv(destination/f"{fold_id}_paired.csv",paired_rows);_csv(destination/f"{fold_id}_natural.csv",natural_rows);_csv(destination/f"{fold_id}_strength.csv",strength_rows);summary={"status":"completed_track_a_evaluator","fold_id":fold_id,"units":len(fold["heldout"]),"gamma_diagnostic_only":True};_json(run_dir/"result_summary.json",summary);return summary


def stage_factorial_build(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);root=Path(str(config["result_root"]))/"factorial";source=Path(str(config["v6_root"]))/"prepared"/fold["fold_id"]
    for key in fold["heldout"]:
        safe=key.replace("/","__");pair=np.load(source/f"paired_{safe}.npz");deploy=root/"deployable_inputs"/fold["fold_id"];evaluator=root/"evaluator"/fold["fold_id"];deploy.mkdir(parents=True,exist_ok=True);evaluator.mkdir(parents=True,exist_ok=True);np.savez_compressed(deploy/f"paired_{safe}.npz",y=pair["y"],recording_key=np.asarray(key));np.savez_compressed(evaluator/f"paired_{safe}.npz",x=pair["x"],a=pair["a"],recording_key=np.asarray(key))
    summary={"status":"completed_factorial_boundary_build","fold_id":fold["fold_id"],"units":len(fold["heldout"]),"outcomes_in_inference":False};_json(run_dir/"result_summary.json",summary);return summary


def _heldout_bases(config:Mapping[str,Any],fold:Mapping[str,Any],arrays:Mapping[str,np.ndarray],population_basis:np.ndarray,key:str,seed:int)->tuple[np.ndarray,np.ndarray]:
    layouts,records=_metadata(config);loaded=load_sgeyesub_signal_record(Path(str(config["data_root"])),records[key],layouts[records[key].layout_id],include_query=False,include_query_annotations=False);rate=float(fold["sampling_rate_hz"]);samples=int(round(120*rate));taps=2*int(round(float(config["fir_lag_ms"])*rate/1000))+1;eog_mean,eog_std=_support_eog_stats(loaded,samples);transfer,_=_support_transfer(loaded,eog_mean,eog_std,arrays["normal_mean"],arrays["normal_std"],samples,taps,float(config["ridge_lambda"]));match=_basis_from_transfer(transfer,population_basis);eeg=(np.asarray(loaded.support.eeg[:,:samples],np.float64)-arrays["normal_mean"])/arrays["normal_std"];eog=(np.asarray(loaded.support.external_eog[:,:samples],np.float64)-eog_mean)/eog_std;eog=eog[:,np.random.default_rng(seed).permutation(eog.shape[1])];shuffled=_basis_from_transfer(fit_dynamic_transfer(eeg,eog,taps=taps,ridge=float(config["ridge_lambda"])),population_basis);return match,shuffled


def _support_pairs(config:Mapping[str,Any],fold:Mapping[str,Any],arrays:Mapping[str,np.ndarray],key:str,basis:np.ndarray,tau:np.ndarray,*,split:str,shuffled:bool,seed:int)->dict[str,np.ndarray]:
    layouts,records=_metadata(config);loaded=load_sgeyesub_signal_record(Path(str(config["data_root"])),records[key],layouts[records[key].layout_id],include_query=False,include_query_annotations=False);taps=2*int(round(float(config["fir_lag_ms"])*float(fold["sampling_rate_hz"])/1000))+1
    return _support_pseudo_pairs(loaded,records[key],arrays["normal_mean"],arrays["normal_std"],basis,tau,budget=120,taps=taps,ridge=float(config["ridge_lambda"]),shuffled=shuffled,split=split,seed=seed)


def _fresh_lora(checkpoint:Mapping[str,Any],kind:str,device:torch.device)->tuple[Any,Any]:
    cfg=ArtifactSubspaceConfig(**checkpoint["model_config"]);model=DeterministicSubspaceEstimator(cfg).to(device) if kind=="det" else ArtifactSubspaceDiffusion(cfg).to(device);model.load_state_dict(checkpoint["det" if kind=="det" else "diff_ema"]);info=inject_score_lora(model.backbone,rank=4);return model,info


@torch.no_grad()
def _support_score(model:Any,kind:str,pairs:Mapping[str,np.ndarray],basis_np:np.ndarray,tau_np:np.ndarray,key:str,seed:int,device:torch.device)->float:
    output=_model_output(model,kind,pairs["y"],basis_np,tau_np,key,seed,device,k=8);return _rrmse(output,pairs["x"])


def _adapt_selected(config:Mapping[str,Any],checkpoint:Mapping[str,Any],kind:str,adapt:Mapping[str,np.ndarray],validation:Mapping[str,np.ndarray],basis_np:np.ndarray,tau_np:np.ndarray,key:str,seed:int,device:torch.device)->tuple[Any,dict[str,Any]]:
    model,info=_fresh_lora(checkpoint,kind,device);basis=torch.tensor(basis_np[None],device=device);optimizer=AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);rng=np.random.default_rng(seed);generator=torch.Generator(device=device).manual_seed(seed);checkpoints=set(map(int,config["score_lora"]["checkpoints"]));curve=[];best=None
    for step in range(0,1001):
        if step in checkpoints:
            model.eval();score=_support_score(model,kind,validation,basis_np,tau_np,key,seed+7000,device);curve.append({"step":step,"validation_rrmse":score});state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if ".down." in k or ".up." in k}
            if best is None or score<best[0]:best=(score,step,state)
        if step==1000:break
        model.train();ids=rng.integers(0,len(adapt["y"]),size=min(8,len(adapt["y"])));y=torch.tensor(adapt["y"][ids],device=device);target=torch.tensor(adapt["target_u"][ids],device=device);valid=torch.tensor(adapt["valid"][ids],device=device);condition=_condition(y,basis,valid);optimizer.zero_grad(set_to_none=True);loss=_masked_u_mse(model(**condition),target,valid) if kind=="det" else model.training_loss(target,generator=generator,**condition)[0];loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
    current=model.state_dict();current.update(best[2]);model.load_state_dict(current);model.eval();return model,{"selected_step":best[1],"validation_rrmse":best[0],"curve":curve,"trainable_parameters":info.trainable_parameters}


def _calibrate(pop:np.ndarray,personal:np.ndarray,y:np.ndarray,x:np.ndarray,points:int)->tuple[float,float,float]:
    c0=y-pop;c1=y-personal;best=None
    for gamma in np.linspace(0,1,points):
        for weight in np.linspace(0,1,points):
            value=y-gamma*((1-weight)*c0+weight*c1);score=_rrmse(value,x)
            if best is None or score<best[0]:best=(score,float(gamma),float(weight))
    return best[1],best[2],best[0]


def _calibrated(y:np.ndarray,pop:np.ndarray,personal:np.ndarray,gamma:float,weight:float)->np.ndarray:
    return y-gamma*((1-weight)*(y-pop)+weight*(y-personal))


def stage_factorial_infer(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);fold_id=fold["fold_id"];folder=_folder(config,fold_id);arrays=np.load(folder/"training_pairs.npz");checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=False);device=torch.device("cuda");base_det,base_diff=_load_models(checkpoint,device);pop=np.asarray(checkpoint["population_basis"],np.float32);tau=np.asarray(checkpoint["tau"],np.float32);root=Path(str(config["result_root"]))/"factorial";seed=int(config["training"]["seed"])+index*1000;calibration_rows=[]
    for position,key in enumerate(fold["heldout"]):
        match,shuffled=_heldout_bases(config,fold,arrays,pop,key,seed+position);adapt_pop=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=False,seed=seed);valid_pop=_support_pairs(config,fold,arrays,key,pop,tau,split="validation",shuffled=False,seed=seed);adapt_match=_support_pairs(config,fold,arrays,key,match,tau,split="adapt",shuffled=False,seed=seed);valid_match=_support_pairs(config,fold,arrays,key,match,tau,split="validation",shuffled=False,seed=seed)
        d01,meta_d01=_adapt_selected(config,checkpoint,"diff",adapt_pop,valid_pop,pop,tau,key,seed+11,device);u01,meta_u01=_adapt_selected(config,checkpoint,"det",adapt_pop,valid_pop,pop,tau,key,seed+12,device);d11,meta_d11=_adapt_selected(config,checkpoint,"diff",adapt_match,valid_match,match,tau,key,seed+13,device);u11,meta_u11=_adapt_selected(config,checkpoint,"det",adapt_match,valid_match,match,tau,key,seed+14,device)
        shuffled_pop=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=True,seed=seed+20);shuffled_basis=_support_pairs(config,fold,arrays,key,shuffled,tau,split="adapt",shuffled=True,seed=seed+20);d01s,_=_adapt_selected(config,checkpoint,"diff",shuffled_pop,valid_pop,pop,tau,key,seed+21,device);d11s,_=_adapt_selected(config,checkpoint,"diff",shuffled_basis,valid_match,shuffled,tau,key,seed+22,device)
        inp=np.load(root/"deployable_inputs"/fold_id/f"paired_{key.replace('/','__')}.npz");y=np.asarray(inp["y"]);outputs={"RAW":y,"DET-D00":_model_output(base_det,"det",y,pop,tau,key,seed,device),"DET-D10":_model_output(base_det,"det",y,match,tau,key,seed,device),"DET-D01":_model_output(u01,"det",y,pop,tau,key,seed,device),"DET-D11":_model_output(u11,"det",y,match,tau,key,seed,device),"DIFF-D00":_model_output(base_diff,"diff",y,pop,tau,key,seed,device),"DIFF-D10":_model_output(base_diff,"diff",y,match,tau,key,seed,device),"DIFF-D01":_model_output(d01,"diff",y,pop,tau,key,seed,device),"DIFF-D11":_model_output(d11,"diff",y,match,tau,key,seed,device),"DIFF-D10-SHUFFLED-GEOMETRY":_model_output(base_diff,"diff",y,shuffled,tau,key,seed,device),"DIFF-D01-SHUFFLED-LORA":_model_output(d01s,"diff",y,pop,tau,key,seed,device),"DIFF-D11-SHUFFLED-BOTH":_model_output(d11s,"diff",y,shuffled,tau,key,seed,device)}
        wrong_names=[]
        for donor_index,donor in enumerate([v for v in fold["heldout"] if v!=key]):
            wrong,_=_heldout_bases(config,fold,arrays,pop,donor,seed+100+donor_index);wrong_pop=_support_pairs(config,fold,arrays,donor,pop,tau,split="adapt",shuffled=False,seed=seed+100);wrong_basis=_support_pairs(config,fold,arrays,donor,wrong,tau,split="adapt",shuffled=False,seed=seed+100);dw0,_=_adapt_selected(config,checkpoint,"diff",wrong_pop,valid_pop,pop,tau,donor,seed+110+donor_index,device);dw1,_=_adapt_selected(config,checkpoint,"diff",wrong_basis,valid_match,wrong,tau,donor,seed+120+donor_index,device);outputs[f"DIFF-D10-WRONG-GEOMETRY-{donor_index}"]=_model_output(base_diff,"diff",y,wrong,tau,key,seed,device);outputs[f"DIFF-D01-WRONG-LORA-{donor_index}"]=_model_output(dw0,"diff",y,pop,tau,key,seed,device);outputs[f"DIFF-D11-WRONG-BOTH-{donor_index}"]=_model_output(dw1,"diff",y,wrong,tau,key,seed,device);wrong_names.append(donor)
        # Support-only calibration is fitted on matching validation pseudo-pairs.
        vpop=_model_output(base_diff,"diff",valid_match["y"],pop,tau,key,seed+5000,device);personal_validation={"D10":_model_output(base_diff,"diff",valid_match["y"],match,tau,key,seed+5000,device),"D01":_model_output(d01,"diff",valid_match["y"],pop,tau,key,seed+5000,device),"D11":_model_output(d11,"diff",valid_match["y"],match,tau,key,seed+5000,device)};cal={}
        for candidate,value in personal_validation.items():gamma,weight,score=_calibrate(vpop,value,valid_match["y"],valid_match["x"],int(config["support_calibration_grid_points"]));cal[candidate]=(gamma,weight);outputs[f"DIFF-{candidate}-CAL"]=_calibrated(y,outputs["DIFF-D00"],outputs[f"DIFF-{candidate}"],gamma,weight);calibration_rows.append({"fold_id":fold_id,"recording_key":key,"candidate":candidate,"gamma":gamma,"w":weight,"support_validation_rrmse":score})
        gamma,weight=cal["D11"]
        for name in list(outputs):
            if "WRONG" in name or "SHUFFLED" in name:outputs[name+"-CAL"]=_calibrated(y,outputs["DIFF-D00"],outputs[name],gamma,weight)
        destination=root/"outputs"/fold_id;destination.mkdir(parents=True,exist_ok=True);np.savez_compressed(destination/f"paired_{key.replace('/','__')}.npz",**outputs);_json(destination/f"adaptation_{key.replace('/','__')}.json",{"recording_key":key,"wrong_donors":wrong_names,"D01":meta_d01,"D11":meta_d11,"DET_D01":meta_u01,"DET_D11":meta_u11,"calibration":{k:{"gamma":v[0],"w":v[1]} for k,v in cal.items()}})
        natural=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_input_{key.replace('/','__')}.npz");raw=np.asarray(natural["y"],np.float32);raw_length=int(arrays["raw_length"]);usable=raw.shape[1]//raw_length*raw_length;windows=raw[:,:usable].reshape(raw.shape[0],-1,raw_length).transpose(1,0,2);natural_output={"RAW":raw[:,:usable]}
        models={"DET-D00":(base_det,"det",pop),"DET-D10":(base_det,"det",match),"DET-D01":(u01,"det",pop),"DET-D11":(u11,"det",match),"DIFF-D00":(base_diff,"diff",pop),"DIFF-D10":(base_diff,"diff",match),"DIFF-D01":(d01,"diff",pop),"DIFF-D11":(d11,"diff",match),"DIFF-D10-SHUFFLED-GEOMETRY":(base_diff,"diff",shuffled),"DIFF-D01-SHUFFLED-LORA":(d01s,"diff",pop),"DIFF-D11-SHUFFLED-BOTH":(d11s,"diff",shuffled)}
        for name,(model,kind,basis) in models.items():natural_output[name]=_model_output(model,kind,windows,basis,tau,key,seed,device).transpose(1,0,2).reshape(raw.shape[0],usable)
        for candidate,(gamma,weight) in cal.items():natural_output[f"DIFF-{candidate}-CAL"]=_calibrated(natural_output["RAW"],natural_output["DIFF-D00"],natural_output[f"DIFF-{candidate}"],gamma,weight)
        for name in list(natural_output):
            if "SHUFFLED" in name:natural_output[name+"-CAL"]=_calibrated(natural_output["RAW"],natural_output["DIFF-D00"],natural_output[name],cal["D11"][0],cal["D11"][1])
        np.savez_compressed(destination/f"natural_{key.replace('/','__')}.npz",**natural_output)
    _csv(root/"support_calibration"/f"{fold_id}.csv",calibration_rows);summary={"status":"completed_factorial_inference","fold_id":fold_id,"units":len(fold["heldout"]),"score_lora_rank":4,"support_seconds":120,"outcomes_opened":False};_json(run_dir/"result_summary.json",summary);return summary


def stage_factorial_eval(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);fold_id=fold["fold_id"];root=Path(str(config["result_root"]))/"factorial";paired=[];natural=[]
    for key in fold["heldout"]:
        safe=key.replace("/","__");output=np.load(root/"outputs"/fold_id/f"paired_{safe}.npz");evaluation=np.load(root/"evaluator"/fold_id/f"paired_{safe}.npz");y=np.asarray(output["RAW"])
        for method in output.files:paired.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,**_paired_metrics(output[method],evaluation["x"],y)})
        nout=np.load(root/"outputs"/fold_id/f"natural_{safe}.npz");raw=np.asarray(nout["RAW"]);neval=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_evaluator_{safe}.npz")
        for method in nout.files:natural.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,**_natural_metrics(raw,nout[method],neval["eog"],neval["labels"],float(fold["sampling_rate_hz"]))})
    destination=root/"metrics";_csv(destination/f"{fold_id}_paired.csv",paired);_csv(destination/f"{fold_id}_natural.csv",natural);summary={"status":"completed_factorial_evaluator","fold_id":fold_id,"units":len(fold["heldout"])};_json(run_dir/"result_summary.json",summary);return summary


def _read_csvs(paths:Sequence[Path])->list[dict[str,str]]:
    rows=[]
    for path in paths:
        with path.open(newline="",encoding="utf-8") as stream:rows.extend(csv.DictReader(stream))
    return rows


def _bootstrap(rows:Sequence[Mapping[str,Any]],metrics:Sequence[str],seed:int,replicates:int)->list[dict[str,Any]]:
    rng=np.random.default_rng(seed);studies=sorted({str(r["study"]) for r in rows});result=[]
    for metric in metrics:
        draws=[]
        for _ in range(replicates):
            values=[]
            for study in studies:
                clusters=sorted({str(r["fold_id"]) for r in rows if r["study"]==study})
                for cluster in rng.choice(clusters,size=len(clusters),replace=True):
                    units=[float(r[metric]) for r in rows if r["fold_id"]==cluster];values.extend(rng.choice(units,size=len(units),replace=True))
            draws.append(float(np.mean(values)))
        observed=np.asarray([float(r[metric]) for r in rows]);result.append({"effect":metric,"mean":float(observed.mean()),"median":float(np.median(observed)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"positive_count":int((observed>0).sum()),"denominator":len(observed),"bootstrap_seed":seed,"replicates":replicates,"scope":"development_descriptive"})
    return result


def stage_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));factor=root/"factorial";folds=list(map(str,config["diagnostic_folds"]));paired=_read_csvs([factor/"metrics"/f"{fold}_paired.csv" for fold in folds]);natural=_read_csvs([factor/"metrics"/f"{fold}_natural.csv" for fold in folds]);calibration=_read_csvs([factor/"support_calibration"/f"{fold}.csv" for fold in folds]);_csv(root/"unit_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);_csv(root/"support_calibration.csv",calibration)
    methods=[]
    for method in sorted({r["method"] for r in paired}):
        subset=[r for r in paired if r["method"]==method];methods.append({"method":method,"units":len(subset),"rrmse_mean":float(np.mean([float(r["rrmse"]) for r in subset])),"correlation_mean":float(np.mean([float(r["correlation"]) for r in subset])),"delta_snr_mean":float(np.mean([float(r["delta_snr"]) for r in subset]))})
    _csv(root/"method_summary.csv",methods);by={(r["recording_key"],r["method"]):r for r in paired};effects=[]
    for key in sorted({r["recording_key"] for r in paired}):
        base=float(by[(key,"DIFF-D00")]["rrmse"]);row0=by[(key,"DIFF-D00")];row={"fold_id":row0["fold_id"],"study":row0["study"],"recording_key":key}
        for candidate in ("D10","D01","D11"):
            personal=float(by[(key,f"DIFF-{candidate}")]["rrmse"]);det=float(by[(key,f"DET-{candidate}")]["rrmse"]);row[f"{candidate}_U_D"]=det-personal;row[f"{candidate}_U_P"]=base-personal
            wrong=[float(r["rrmse"])-personal for r in paired if r["recording_key"]==key and r["method"].startswith(f"DIFF-{candidate}-WRONG")];shuffled_name={"D10":"DIFF-D10-SHUFFLED-GEOMETRY","D01":"DIFF-D01-SHUFFLED-LORA","D11":"DIFF-D11-SHUFFLED-BOTH"}[candidate];row[f"{candidate}_U_W"]=float(np.mean(wrong)) if wrong else float("nan");row[f"{candidate}_U_S"]=float(by[(key,shuffled_name)]["rrmse"])-personal
            calibrated=float(by[(key,f"DIFF-{candidate}-CAL")]["rrmse"]);row[f"{candidate}_CAL_U_P"]=base-calibrated
        r00=base;r10=float(by[(key,"DIFF-D10")]["rrmse"]);r01=float(by[(key,"DIFF-D01")]["rrmse"]);r11=float(by[(key,"DIFF-D11")]["rrmse"]);row.update({"G":r00-r10,"A":r00-r01,"C":r00-r11,"I":r10+r01-r00-r11});effects.append(row)
    _csv(root/"factorial_effects.csv",effects);_csv(root/"paired_effects.csv",effects)
    bootstrap=_bootstrap(effects,["G","A","C","I",*[f"{c}_{m}" for c in ("D10","D01","D11") for m in ("U_D","U_P","U_W","U_S")]],int(config["bootstrap_seed"]),int(config["bootstrap_replicates"]));_csv(root/"bootstrap_summary.csv",bootstrap)
    natural_by={(r["recording_key"],r["method"]):r for r in natural};cal_by={(r["recording_key"],r["candidate"]):r for r in calibration};routes=[]
    for candidate in ("D10","D01","D11"):
        metrics={name:float(np.nanmean([r[f"{candidate}_{name}"] for r in effects])) for name in ("U_D","U_P","U_W","U_S")};subject_values=[metrics[n] for n in ("U_P","U_W","U_S")];study_counts={name:int(sum(np.nanmean([r[f"{candidate}_{name}"] for r in effects if r["study"]==study])>=0 for study in sorted({r["study"] for r in effects}))) for name in ("U_P","U_W","U_S")};cal_rows=[r for r in calibration if r["candidate"]==candidate];mean_w=float(np.mean([float(r["w"]) for r in cal_rows]));active=float(np.mean([float(r["w"])>=.25 for r in cal_rows]));method=f"DIFF-{candidate}-CAL";nrows=[r for r in natural if r["method"]==method];pres=float(np.nanmean([float(r["nonartifact_preservation"]) for r in nrows]));psd=float(np.nanmean([float(r["psd_distortion"]) for r in nrows]));cov=float(np.nanmean([float(r["covariance_distortion"]) for r in nrows]));severe=float(np.mean([r[f"{candidate}_CAL_U_P"]<-.05 for r in effects]));coverage=len(cal_rows)/max(len(effects),1);qualifies=coverage>=.8 and metrics["U_D"]>0 and np.mean(subject_values)>0 and sum(v>0 for v in subject_values)>=2 and min(subject_values)>=-.01 and max(study_counts.values())>=3 and severe<=.2 and pres>=.70 and psd<=.35 and cov<=.35;routes.append({"route":candidate,"coverage":coverage,**metrics,"mean_subject_effect":float(np.mean(subject_values)),"nonnegative_study_counts":json.dumps(study_counts,sort_keys=True),"severe_reversal_fraction":severe,"preservation":pres,"psd_distortion":psd,"covariance_distortion":cov,"mean_w":mean_w,"w_ge_0_25_fraction":active,"diagnostic_qualifies":qualifies})
    _csv(root/"route_summary.csv",routes);ranked=sorted(routes,key=lambda r:(r["diagnostic_qualifies"],r["mean_subject_effect"],r["U_D"]),reverse=True);promoted=[r["route"] for r in ranked if r["diagnostic_qualifies"]][:2];decision={"status":"completed_six_fold_diagnostic","promoted_routes":promoted,"full_one_seed_authorized":bool(promoted),"diagnostic_units":len(effects),"availability_denominator":59,"score_lora_tested":True,"routes":routes,"bootstrap_seed":int(config["bootstrap_seed"]),"confirmation_evidence":False};_json(root/"route_decision.json",decision);_json(run_dir/"result_summary.json",decision);return decision


def stage_finalize(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));decision=json.loads((root/"route_decision.json").read_text());methods=list(csv.DictReader((root/"method_summary.csv").open()));routes=list(csv.DictReader((root/"route_summary.csv").open()));lines=["# SGE subject-basis × score-LoRA V9","","Development exploration only. Historical v8.1 remains `BRIDGE_GATE_FAILED_SCORE_LORA_NOT_RUN`; V9 directly tests Score-LoRA and does not revise that result.","","## Six-fold diagnostic routes","","| route | U_D | U_P | U_W | U_S | preservation | PSD | covariance | mean w | promoted |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in routes:lines.append(f"| {r['route']} | {float(r['U_D']):+.4f} | {float(r['U_P']):+.4f} | {float(r['U_W']):+.4f} | {float(r['U_S']):+.4f} | {float(r['preservation']):.4f} | {float(r['psd_distortion']):.4f} | {float(r['covariance_distortion']):.4f} | {float(r['mean_w']):.3f} | {r['diagnostic_qualifies']} |")
    lines += ["","Geometry is D10, score-LoRA is D01, and their joint/interaction route is D11. Raw and support-calibrated outputs are reported separately. WRONG donors were scored individually before utility averaging.","",f"Promoted routes: `{decision['promoted_routes']}`.","","All evidence is exposed development evidence, not confirmation. No result is a family-wide diffusion or personalization conclusion."]
    Path("reports/sge_basis_score_factorial_v9.md").write_text("\n".join(lines)+"\n",encoding="utf-8");summary={"status":"completed_v9","diagnostic_decision":decision,"full_one_seed_status":"AUTHORIZED_PENDING" if decision["full_one_seed_authorized"] else "NOT_RUN_NO_DIAGNOSTIC_CANDIDATE","confirmation_evidence":False};_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path);run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="audit":return stage_audit(config,run_dir)
    if stage=="prepare":return stage_prepare(config,task_index,run_dir)
    if stage=="technical":return stage_technical(config,run_dir)
    if stage=="train":return stage_train(config,task_index,run_dir)
    if stage=="track-a-build":return stage_track_a_build(config,task_index,run_dir)
    if stage=="track-a-infer":return stage_track_a_infer(config,task_index,run_dir)
    if stage=="track-a-eval":return stage_track_a_eval(config,task_index,run_dir)
    if stage=="factorial-build":return stage_factorial_build(config,task_index,run_dir)
    if stage=="factorial-infer":return stage_factorial_infer(config,task_index,run_dir)
    if stage=="factorial-eval":return stage_factorial_eval(config,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(config,run_dir)
    if stage=="finalize":return stage_finalize(config,run_dir)
    raise ValueError(f"unknown V9 stage: {stage}")


__all__=["run_stage","_basis_batch"]
