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


def _write_population_fallback(config:Mapping[str,Any],fold:Mapping[str,Any],arrays:Mapping[str,np.ndarray],key:str,base_det:Any,base_diff:Any,pop:np.ndarray,tau:np.ndarray,seed:int,device:torch.device,destination:Path)->list[dict[str,Any]]:
    """Preserve an ineligible support unit with the predeclared POP fallback."""
    safe=key.replace('/','__');inp=np.load(Path(str(config["result_root"]))/"factorial"/"deployable_inputs"/fold["fold_id"]/f"paired_{safe}.npz");y=np.asarray(inp["y"]);det_pop=_model_output(base_det,"det",y,pop,tau,key,seed,device);diff_pop=_model_output(base_diff,"diff",y,pop,tau,key,seed,device);outputs={"RAW":y,"DET-D00":det_pop,"DIFF-D00":diff_pop}
    for candidate in ("D10","D01","D11"):
        outputs[f"DET-{candidate}"]=det_pop;outputs[f"DIFF-{candidate}"]=diff_pop;outputs[f"DIFF-{candidate}-CAL"]=diff_pop
    controls=("DIFF-D10-SHUFFLED-GEOMETRY","DIFF-D01-SHUFFLED-LORA","DIFF-D11-SHUFFLED-BOTH")
    for name in controls:outputs[name]=diff_pop;outputs[name+"-CAL"]=diff_pop
    donors=[value for value in fold["heldout"] if value!=key]
    for donor_index,_ in enumerate(donors):
        for candidate,label in (("D10","WRONG-GEOMETRY"),("D01","WRONG-LORA"),("D11","WRONG-BOTH")):
            name=f"DIFF-{candidate}-{label}-{donor_index}";outputs[name]=diff_pop;outputs[name+"-CAL"]=diff_pop
    np.savez_compressed(destination/f"paired_{safe}.npz",**outputs)
    natural=np.load(Path(str(config["v6_root"]))/"prepared"/fold["fold_id"]/f"natural_input_{safe}.npz");raw=np.asarray(natural["y"],np.float32);raw_length=int(arrays["raw_length"]);usable=raw.shape[1]//raw_length*raw_length;windows=raw[:,:usable].reshape(raw.shape[0],-1,raw_length).transpose(1,0,2);det_n=_model_output(base_det,"det",windows,pop,tau,key,seed,device).transpose(1,0,2).reshape(raw.shape[0],usable);diff_n=_model_output(base_diff,"diff",windows,pop,tau,key,seed,device).transpose(1,0,2).reshape(raw.shape[0],usable);natural_output={"RAW":raw[:,:usable],"DET-D00":det_n,"DIFF-D00":diff_n}
    for candidate in ("D10","D01","D11"):natural_output[f"DET-{candidate}"]=det_n;natural_output[f"DIFF-{candidate}"]=diff_n;natural_output[f"DIFF-{candidate}-CAL"]=diff_n
    for name in controls:natural_output[name]=diff_n;natural_output[name+"-CAL"]=diff_n
    np.savez_compressed(destination/f"natural_{safe}.npz",**natural_output);_json(destination/f"adaptation_{safe}.json",{"recording_key":key,"status":"support_pair_coverage_insufficient_pop_fallback","personalization_eligible":False,"wrong_donors":donors,"calibration":{candidate:{"gamma":1.0,"w":0.0} for candidate in ("D10","D01","D11")}})
    return [{"fold_id":fold["fold_id"],"recording_key":key,"candidate":candidate,"gamma":1.0,"w":0.0,"support_validation_rrmse":None,"personalization_eligible":0,"fallback_reason":"support_pair_coverage_insufficient"} for candidate in ("D10","D01","D11")]


def stage_factorial_infer(config:Mapping[str,Any],index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,index);fold_id=fold["fold_id"];folder=_folder(config,fold_id);arrays=np.load(folder/"training_pairs.npz");checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=False);device=torch.device("cuda");base_det,base_diff=_load_models(checkpoint,device);pop=np.asarray(checkpoint["population_basis"],np.float32);tau=np.asarray(checkpoint["tau"],np.float32);root=Path(str(config["result_root"]))/"factorial";seed=int(config["training"]["seed"])+index*1000;calibration_rows=[];fallback_units=0
    for position,key in enumerate(fold["heldout"]):
        destination=root/"outputs"/fold_id;destination.mkdir(parents=True,exist_ok=True);safe=key.replace('/','__');paired_path=destination/f"paired_{safe}.npz";natural_path=destination/f"natural_{safe}.npz";adaptation_path=destination/f"adaptation_{safe}.json"
        if paired_path.exists() and natural_path.exists() and adaptation_path.exists():
            adaptation=json.loads(adaptation_path.read_text(encoding="utf-8"))
            eligible=int(adaptation.get("personalization_eligible",True));fallback_units+=int(not eligible)
            for candidate,values in adaptation.get("calibration",{}).items():calibration_rows.append({"fold_id":fold_id,"recording_key":key,"candidate":candidate,"gamma":values["gamma"],"w":values["w"],"support_validation_rrmse":adaptation.get("support_validation_rrmse",{}).get(candidate),"personalization_eligible":eligible,"fallback_reason":"" if eligible else "support_pair_coverage_insufficient"})
            continue
        try:
            match,shuffled=_heldout_bases(config,fold,arrays,pop,key,seed+position);adapt_pop=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=False,seed=seed);valid_pop=_support_pairs(config,fold,arrays,key,pop,tau,split="validation",shuffled=False,seed=seed);adapt_match=_support_pairs(config,fold,arrays,key,match,tau,split="adapt",shuffled=False,seed=seed);valid_match=_support_pairs(config,fold,arrays,key,match,tau,split="validation",shuffled=False,seed=seed)
        except ValueError:
            calibration_rows.extend(_write_population_fallback(config,fold,arrays,key,base_det,base_diff,pop,tau,seed,device,destination));fallback_units+=1;continue
        d01,meta_d01=_adapt_selected(config,checkpoint,"diff",adapt_pop,valid_pop,pop,tau,key,seed+11,device);u01,meta_u01=_adapt_selected(config,checkpoint,"det",adapt_pop,valid_pop,pop,tau,key,seed+12,device);d11,meta_d11=_adapt_selected(config,checkpoint,"diff",adapt_match,valid_match,match,tau,key,seed+13,device);u11,meta_u11=_adapt_selected(config,checkpoint,"det",adapt_match,valid_match,match,tau,key,seed+14,device)
        shuffled_pop=_support_pairs(config,fold,arrays,key,pop,tau,split="adapt",shuffled=True,seed=seed+20);shuffled_pop_valid=_support_pairs(config,fold,arrays,key,pop,tau,split="validation",shuffled=True,seed=seed+20);shuffled_basis=_support_pairs(config,fold,arrays,key,shuffled,tau,split="adapt",shuffled=True,seed=seed+20);shuffled_basis_valid=_support_pairs(config,fold,arrays,key,shuffled,tau,split="validation",shuffled=True,seed=seed+20);d01s,_=_adapt_selected(config,checkpoint,"diff",shuffled_pop,shuffled_pop_valid,pop,tau,key,seed+21,device);d11s,_=_adapt_selected(config,checkpoint,"diff",shuffled_basis,shuffled_basis_valid,shuffled,tau,key,seed+22,device)
        inp=np.load(root/"deployable_inputs"/fold_id/f"paired_{key.replace('/','__')}.npz");y=np.asarray(inp["y"]);outputs={"RAW":y,"DET-D00":_model_output(base_det,"det",y,pop,tau,key,seed,device),"DET-D10":_model_output(base_det,"det",y,match,tau,key,seed,device),"DET-D01":_model_output(u01,"det",y,pop,tau,key,seed,device),"DET-D11":_model_output(u11,"det",y,match,tau,key,seed,device),"DIFF-D00":_model_output(base_diff,"diff",y,pop,tau,key,seed,device),"DIFF-D10":_model_output(base_diff,"diff",y,match,tau,key,seed,device),"DIFF-D01":_model_output(d01,"diff",y,pop,tau,key,seed,device),"DIFF-D11":_model_output(d11,"diff",y,match,tau,key,seed,device),"DIFF-D10-SHUFFLED-GEOMETRY":_model_output(base_diff,"diff",y,shuffled,tau,key,seed,device),"DIFF-D01-SHUFFLED-LORA":_model_output(d01s,"diff",y,pop,tau,key,seed,device),"DIFF-D11-SHUFFLED-BOTH":_model_output(d11s,"diff",y,shuffled,tau,key,seed,device)}
        wrong_names=[]
        for donor_index,donor in enumerate([v for v in fold["heldout"] if v!=key]):
            wrong,_=_heldout_bases(config,fold,arrays,pop,donor,seed+100+donor_index);wrong_pop=_support_pairs(config,fold,arrays,donor,pop,tau,split="adapt",shuffled=False,seed=seed+100);wrong_pop_valid=_support_pairs(config,fold,arrays,donor,pop,tau,split="validation",shuffled=False,seed=seed+100);wrong_basis=_support_pairs(config,fold,arrays,donor,wrong,tau,split="adapt",shuffled=False,seed=seed+100);wrong_basis_valid=_support_pairs(config,fold,arrays,donor,wrong,tau,split="validation",shuffled=False,seed=seed+100);dw0,_=_adapt_selected(config,checkpoint,"diff",wrong_pop,wrong_pop_valid,pop,tau,donor,seed+110+donor_index,device);dw1,_=_adapt_selected(config,checkpoint,"diff",wrong_basis,wrong_basis_valid,wrong,tau,donor,seed+120+donor_index,device);outputs[f"DIFF-D10-WRONG-GEOMETRY-{donor_index}"]=_model_output(base_diff,"diff",y,wrong,tau,key,seed,device);outputs[f"DIFF-D01-WRONG-LORA-{donor_index}"]=_model_output(dw0,"diff",y,pop,tau,key,seed,device);outputs[f"DIFF-D11-WRONG-BOTH-{donor_index}"]=_model_output(dw1,"diff",y,wrong,tau,key,seed,device);wrong_names.append(donor)
        # Support-only calibration is fitted on matching validation pseudo-pairs.
        vpop=_model_output(base_diff,"diff",valid_match["y"],pop,tau,key,seed+5000,device);personal_validation={"D10":_model_output(base_diff,"diff",valid_match["y"],match,tau,key,seed+5000,device),"D01":_model_output(d01,"diff",valid_match["y"],pop,tau,key,seed+5000,device),"D11":_model_output(d11,"diff",valid_match["y"],match,tau,key,seed+5000,device)};cal={}
        for candidate,value in personal_validation.items():gamma,weight,score=_calibrate(vpop,value,valid_match["y"],valid_match["x"],int(config["support_calibration_grid_points"]));cal[candidate]=(gamma,weight);outputs[f"DIFF-{candidate}-CAL"]=_calibrated(y,outputs["DIFF-D00"],outputs[f"DIFF-{candidate}"],gamma,weight);calibration_rows.append({"fold_id":fold_id,"recording_key":key,"candidate":candidate,"gamma":gamma,"w":weight,"support_validation_rrmse":score,"personalization_eligible":1,"fallback_reason":""})
        for name in list(outputs):
            if "WRONG" in name or "SHUFFLED" in name:
                candidate=next(value for value in ("D10","D01","D11") if f"DIFF-{value}-" in name);gamma,weight=cal[candidate];outputs[name+"-CAL"]=_calibrated(y,outputs["DIFF-D00"],outputs[name],gamma,weight)
        np.savez_compressed(paired_path,**outputs);_json(adaptation_path,{"recording_key":key,"status":"personalization_active","personalization_eligible":True,"wrong_donors":wrong_names,"D01":meta_d01,"D11":meta_d11,"DET_D01":meta_u01,"DET_D11":meta_u11,"calibration":{k:{"gamma":v[0],"w":v[1]} for k,v in cal.items()},"support_validation_rrmse":{r["candidate"]:r["support_validation_rrmse"] for r in calibration_rows if r["recording_key"]==key}})
        natural=np.load(Path(str(config["v6_root"]))/"prepared"/fold_id/f"natural_input_{key.replace('/','__')}.npz");raw=np.asarray(natural["y"],np.float32);raw_length=int(arrays["raw_length"]);usable=raw.shape[1]//raw_length*raw_length;windows=raw[:,:usable].reshape(raw.shape[0],-1,raw_length).transpose(1,0,2);natural_output={"RAW":raw[:,:usable]}
        models={"DET-D00":(base_det,"det",pop),"DET-D10":(base_det,"det",match),"DET-D01":(u01,"det",pop),"DET-D11":(u11,"det",match),"DIFF-D00":(base_diff,"diff",pop),"DIFF-D10":(base_diff,"diff",match),"DIFF-D01":(d01,"diff",pop),"DIFF-D11":(d11,"diff",match),"DIFF-D10-SHUFFLED-GEOMETRY":(base_diff,"diff",shuffled),"DIFF-D01-SHUFFLED-LORA":(d01s,"diff",pop),"DIFF-D11-SHUFFLED-BOTH":(d11s,"diff",shuffled)}
        for name,(model,kind,basis) in models.items():natural_output[name]=_model_output(model,kind,windows,basis,tau,key,seed,device).transpose(1,0,2).reshape(raw.shape[0],usable)
        for candidate,(gamma,weight) in cal.items():natural_output[f"DIFF-{candidate}-CAL"]=_calibrated(natural_output["RAW"],natural_output["DIFF-D00"],natural_output[f"DIFF-{candidate}"],gamma,weight)
        for name in list(natural_output):
            if "SHUFFLED" in name:
                candidate=next(value for value in ("D10","D01","D11") if f"DIFF-{value}-" in name);natural_output[name+"-CAL"]=_calibrated(natural_output["RAW"],natural_output["DIFF-D00"],natural_output[name],cal[candidate][0],cal[candidate][1])
        np.savez_compressed(natural_path,**natural_output)
    _csv(root/"support_calibration"/f"{fold_id}.csv",calibration_rows);summary={"status":"completed_factorial_inference","fold_id":fold_id,"units":len(fold["heldout"]),"personalization_eligible_units":len(fold["heldout"])-fallback_units,"population_fallback_units":fallback_units,"score_lora_rank":4,"support_seconds":120,"outcomes_opened":False};_json(run_dir/"result_summary.json",summary);return summary


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


def _personalization_coverage(rows:Sequence[Mapping[str,Any]],denominator:int)->float:
    return sum(int(row.get("personalization_eligible",1))==1 for row in rows)/max(int(denominator),1)


def stage_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));factor=root/"factorial";folds=list(map(str,config["diagnostic_folds"]));paired=_read_csvs([factor/"metrics"/f"{fold}_paired.csv" for fold in folds]);natural=_read_csvs([factor/"metrics"/f"{fold}_natural.csv" for fold in folds]);calibration=_read_csvs([factor/"support_calibration"/f"{fold}.csv" for fold in folds]);_csv(root/"unit_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);_csv(root/"support_calibration.csv",calibration)
    track_a_root=root/"track_a";track_a_folds=list(map(str,config["track_a_folds"]));track_a_paired=_read_csvs([track_a_root/"metrics"/f"{fold}_paired.csv" for fold in track_a_folds]);track_a_natural=_read_csvs([track_a_root/"metrics"/f"{fold}_natural.csv" for fold in track_a_folds]);track_a_strength=_read_csvs([track_a_root/"metrics"/f"{fold}_strength.csv" for fold in track_a_folds]);_csv(track_a_root/"unit_metrics.csv",track_a_paired);_csv(track_a_root/"natural_safety.csv",track_a_natural);_csv(track_a_root/"strength_diagnostic.csv",track_a_strength)
    track_a_methods=[]
    for method in sorted({r["method"] for r in track_a_paired}):
        subset=[r for r in track_a_paired if r["method"]==method];track_a_methods.append({"method":method,"units":len(subset),"rrmse_mean":float(np.mean([float(r["rrmse"]) for r in subset])),"correlation_mean":float(np.mean([float(r["correlation"]) for r in subset])),"delta_snr_mean":float(np.mean([float(r["delta_snr"]) for r in subset]))})
    _csv(track_a_root/"method_summary.csv",track_a_methods);ta={(r["recording_key"],r["method"]):r for r in track_a_paired};track_a_effects=[]
    oracle_rows=_read_csvs([Path(str(config["v8_1_root"]))/"oracle_ceiling_metrics.csv"]);oracle={(r["recording_key"],r["basis"]):r for r in oracle_rows if r["ceiling"]=="ORACLE_PROJECTION"}
    for key in sorted({r["recording_key"] for r in track_a_paired}):
        base=ta[(key,"DIFF-POP")];match=ta[(key,"DIFF-MATCH120")];eb=ta[(key,"DIFF-EB-FIXED120")];wrong=[float(r["rrmse"])-float(match["rrmse"]) for r in track_a_paired if r["recording_key"]==key and r["method"].startswith("DIFF-RAW-WRONG120-")];row={"fold_id":base["fold_id"],"study":base["study"],"recording_key":key,"match_minus_pop_utility":float(base["rrmse"])-float(match["rrmse"]),"match_minus_eb_utility":float(eb["rrmse"])-float(match["rrmse"]),"match_minus_wrong_utility":float(np.mean(wrong)) if wrong else float("nan"),"match_minus_shuffled_utility":float(ta[(key,"DIFF-RAW-SHUFFLED120")]["rrmse"])-float(match["rrmse"])}
        if (key,"A_POP") in oracle and (key,"A_MATCH120") in oracle:row["oracle_match_minus_pop_ceiling"]=float(oracle[(key,"A_POP")]["rrmse"])-float(oracle[(key,"A_MATCH120")]["rrmse"])
        if (key,"A_EB120") in oracle and (key,"A_MATCH120") in oracle:row["oracle_match_minus_eb_ceiling"]=float(oracle[(key,"A_EB120")]["rrmse"])-float(oracle[(key,"A_MATCH120")]["rrmse"])
        track_a_effects.append(row)
    _csv(track_a_root/"paired_effects.csv",track_a_effects)
    def development_corr(left:str,right:str)->float:
        values=[(float(r[left]),float(r[right])) for r in track_a_effects if left in r and right in r and np.isfinite(float(r[left])) and np.isfinite(float(r[right]))];return float(np.corrcoef(np.asarray(values).T)[0,1]) if len(values)>=3 else float("nan")
    track_a_correlation={"oracle_match_minus_pop_vs_actual":development_corr("oracle_match_minus_pop_ceiling","match_minus_pop_utility"),"oracle_match_minus_eb_vs_actual":development_corr("oracle_match_minus_eb_ceiling","match_minus_eb_utility"),"scope":"development_correlation_only","shared_RAW_correlation_reported":False};_json(track_a_root/"oracle_actual_correlations.json",track_a_correlation)
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
    natural_by={(r["recording_key"],r["method"]):r for r in natural};cal_by={(r["recording_key"],r["candidate"]):r for r in calibration};routes=[];per_study=[]
    for candidate in ("D10","D01","D11"):
        metrics={name:float(np.nanmean([r[f"{candidate}_{name}"] for r in effects])) for name in ("U_D","U_P","U_W","U_S")};subject_values=[metrics[n] for n in ("U_P","U_W","U_S")];study_counts={name:int(sum(np.nanmean([r[f"{candidate}_{name}"] for r in effects if r["study"]==study])>=0 for study in sorted({r["study"] for r in effects}))) for name in ("U_P","U_W","U_S")};subject_nonnegative_studies=0
        for study in sorted({r["study"] for r in effects}):
            subset=[r for r in effects if r["study"]==study];values={name:float(np.nanmean([r[f"{candidate}_{name}"] for r in subset])) for name in ("U_D","U_P","U_W","U_S")};subject_mean=float(np.nanmean([values[n] for n in ("U_P","U_W","U_S")]));subject_nonnegative_studies+=int(subject_mean>=0);per_study.append({"route":candidate,"study":study,"units":len(subset),**values,"mean_subject_effect":subject_mean})
        cal_rows=[r for r in calibration if r["candidate"]==candidate];eligible_rows=[r for r in cal_rows if int(r.get("personalization_eligible",1))==1];mean_w=float(np.mean([float(r["w"]) for r in cal_rows]));active=float(np.mean([float(r["w"])>=.25 for r in cal_rows]));method=f"DIFF-{candidate}-CAL";nrows=[r for r in natural if r["method"]==method];pres=float(np.nanmean([float(r["nonartifact_preservation"]) for r in nrows]));psd=float(np.nanmean([float(r["psd_distortion"]) for r in nrows]));cov=float(np.nanmean([float(r["covariance_distortion"]) for r in nrows]));eog=float(np.nanmean([float(r["eog_coherence_reduction"]) for r in nrows]));cal_u_p=float(np.nanmean([r[f"{candidate}_CAL_U_P"] for r in effects]));severe=float(np.mean([r[f"{candidate}_CAL_U_P"]<-.05 for r in effects]));coverage=_personalization_coverage(cal_rows,len(effects));failures=[]
        if coverage<.8:failures.append("personalization_coverage_lt_0.80")
        if metrics["U_D"]<=0:failures.append("U_D_nonpositive")
        if np.mean(subject_values)<=0:failures.append("mean_subject_effect_nonpositive")
        if sum(v>0 for v in subject_values)<2:failures.append("fewer_than_two_positive_subject_effects")
        if min(subject_values)<-.01:failures.append("subject_effect_below_minus_0.01")
        if subject_nonnegative_studies<3:failures.append("fewer_than_three_nonnegative_studies")
        if severe>.2:failures.append("severe_reversal_fraction_gt_0.20")
        if pres<.70:failures.append("preservation_below_0.70")
        if psd>.35:failures.append("psd_distortion_above_0.35")
        if cov>.35:failures.append("covariance_distortion_above_0.35")
        qualifies=not failures;routes.append({"route":candidate,"coverage":coverage,"fallback_units":len(cal_rows)-len(eligible_rows),**metrics,"calibrated_U_P":cal_u_p,"mean_subject_effect":float(np.mean(subject_values)),"nonnegative_study_counts":json.dumps(study_counts,sort_keys=True),"subject_nonnegative_studies":subject_nonnegative_studies,"severe_reversal_fraction":severe,"eog_coherence_reduction":eog,"preservation":pres,"psd_distortion":psd,"covariance_distortion":cov,"mean_w":mean_w,"w_ge_0_25_fraction":active,"diagnostic_qualifies":qualifies,"qualification_failures":";".join(failures)})
    _csv(root/"per_study_effects.csv",per_study)
    _csv(root/"route_summary.csv",routes);ranked=sorted(routes,key=lambda r:(r["diagnostic_qualifies"],r["mean_subject_effect"],r["U_D"]),reverse=True);promoted=[r["route"] for r in ranked if r["diagnostic_qualifies"]][:2];decision={"status":"completed_six_fold_diagnostic","promoted_routes":promoted,"full_one_seed_authorized":bool(promoted),"diagnostic_units":len(effects),"availability_denominator":59,"score_lora_tested":True,"routes":routes,"track_a_oracle_actual_correlations":track_a_correlation,"bootstrap_seed":int(config["bootstrap_seed"]),"confirmation_evidence":False};_json(root/"route_decision.json",decision);_json(run_dir/"result_summary.json",decision);return decision


def stage_finalize(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));decision=json.loads((root/"route_decision.json").read_text());methods=list(csv.DictReader((root/"method_summary.csv").open()));routes=list(csv.DictReader((root/"route_summary.csv").open()));track_a_methods=list(csv.DictReader((root/"track_a"/"method_summary.csv").open()));track_a_natural=list(csv.DictReader((root/"track_a"/"natural_safety.csv").open()));track_a_corr=json.loads((root/"track_a"/"oracle_actual_correlations.json").read_text());lines=["# SGE subject-basis × score-LoRA V9","","Development exploration only. Historical v8.1 remains `BRIDGE_GATE_FAILED_SCORE_LORA_NOT_RUN`; V9 directly tests Score-LoRA and does not revise that result. Score-LoRA was therefore tested for the first time here, independently of the historical v8.1 bridge gate.","","## V8.1 no-training Track A","","Raw MATCH120 is the primary geometry control, POP is the baseline, fixed-λ EB is secondary, and full-feature EB is stopped. WRONG donors are scored before averaging. The correction-strength curve is diagnostic only and was not used to select V9.","","| method | units | mean RRMSE | mean correlation | mean ΔSNR |","|---|---:|---:|---:|---:|"]
    for r in track_a_methods:lines.append(f"| {r['method']} | {r['units']} | {float(r['rrmse_mean']):.4f} | {float(r['correlation_mean']):.4f} | {float(r['delta_snr_mean']):+.4f} |")
    lines += ["",f"Oracle MATCH−POP ceiling vs actual MATCH−POP correlation: `{track_a_corr['oracle_match_minus_pop_vs_actual']:.4f}`; oracle MATCH−EB vs actual MATCH−EB: `{track_a_corr['oracle_match_minus_eb_vs_actual']:.4f}`. These are development correlations only; the mathematically coupled RAW-shared correlation is not reported.","","Track A natural-query raw operating points:","","| method | units | EOG reduction | preservation | PSD | covariance |","|---|---:|---:|---:|---:|---:|"]
    for method in ("DIFF-POP","DIFF-MATCH120","DIFF-EB-FIXED120","DIFF-RAW-SHUFFLED120"):
        subset=[r for r in track_a_natural if r["method"]==method]
        if subset:lines.append(f"| {method} | {len(subset)} | {np.mean([float(r['eog_coherence_reduction']) for r in subset]):.4f} | {np.mean([float(r['nonartifact_preservation']) for r in subset]):.4f} | {np.mean([float(r['psd_distortion']) for r in subset]):.4f} | {np.mean([float(r['covariance_distortion']) for r in subset]):.4f} |")
    lines += ["","The full γ={0.50, 0.65, 0.75, 0.85, 0.925, 1.00} diagnostic is preserved in `track_a/strength_diagnostic.csv`; it was not used for method selection.","","## Six-fold diagnostic routes","","Positive U denotes lower paired RRMSE. U_D is diffusion−matched deterministic; U_P is personal−population; U_W and U_S are wrong/shuffled specificity. Safety values use the support-calibrated output, while the primary U values are raw outputs.","","| route | U_D | raw U_P | cal U_P | U_W | U_S | EOG red. | preservation | PSD | covariance | coverage | fallback | mean w | promoted |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in routes:lines.append(f"| {r['route']} | {float(r['U_D']):+.4f} | {float(r['U_P']):+.4f} | {float(r['calibrated_U_P']):+.4f} | {float(r['U_W']):+.4f} | {float(r['U_S']):+.4f} | {float(r['eog_coherence_reduction']):.4f} | {float(r['preservation']):.4f} | {float(r['psd_distortion']):.4f} | {float(r['covariance_distortion']):.4f} | {100*float(r['coverage']):.1f}% | {r['fallback_units']} | {float(r['mean_w']):.3f} | {r['diagnostic_qualifies']} |")
    lines += ["","Absolute paired performance:","","| method | units | RRMSE | correlation | ΔSNR |","|---|---:|---:|---:|---:|"]
    for name in ("RAW","DET-D00","DIFF-D00","DET-D10","DIFF-D10","DIFF-D10-CAL","DET-D01","DIFF-D01","DIFF-D01-CAL","DET-D11","DIFF-D11","DIFF-D11-CAL"):
        row=next((r for r in methods if r["method"]==name),None)
        if row:lines.append(f"| {name} | {row['units']} | {float(row['rrmse_mean']):.4f} | {float(row['correlation_mean']):.4f} | {float(row['delta_snr_mean']):+.4f} |")
    factorial=list(csv.DictReader((root/"factorial_effects.csv").open()));means={name:float(np.nanmean([float(r[name]) for r in factorial])) for name in ("G","A","C","I")};technical=json.loads((root/"technical_validity.json").read_text());lines += ["",f"Raw factorial effects: geometry G `{means['G']:+.4f}`, LoRA A `{means['A']:+.4f}`, combined C `{means['C']:+.4f}`, interaction I `{means['I']:+.4f}`.",f"Technical validity: `{technical['status']}`; three real subject bases; fixed-batch loss reduction `{technical['fixed_batch_loss_reduction']:.6f}`; basis response `{technical['basis_context_response']:.6f}`; {technical['lora']['adapted_convolutions']} internal ResBlock convolutions and {technical['lora']['trainable_parameters']} rank-4 parameters.","","Geometry is D10, score-LoRA is D01, and their joint/interaction route is D11. WRONG donors were scored individually before utility averaging. Study heterogeneity is in `per_study_effects.csv`; the bootstrap is development-descriptive, clustered by outer fold within study, and uses seed 20260811.","","No route was promoted. D11 was the strongest combined signal but failed the frozen diagnostic rule because personalization coverage was 11/14 (78.6%, below 80%). Its raw U_P was small and heterogeneous: only 4/14 units were positive and only 2/5 study means were nonnegative. D10 also had negative U_P; D01 had negative U_P and a 21.4% severe-reversal rate.","","The study05 units lacked sufficient disjoint support pseudo-pair roles and therefore used the predeclared support-only POP fallback (w=0); all three remain in the denominator. Consequently J5–J7, the 27-fold one-seed extension, and extra seeds were not run.","","Recovery record: job 928440 was cancelled before output after the candidate-specific calibration-control fix. Array task 928421_5 then failed on study05 support coverage, making 928422 DependencyNeverSatisfied; recovery jobs 928860 and 928862 applied the predeclared POP fallback, followed by successful aggregation/finalization. Final tests passed in job 928867. The first clean-checkout job 928868 exposed a node without SLURM_TMPDIR; the fallback path was repaired and the check resubmitted without changing science outputs.","",f"Promoted routes: `{decision['promoted_routes']}`.","","Narrow conclusion: the population artifact-subspace diffusion is a valid and useful estimator in these diagnostic folds and diffusion beats its information-matched deterministic estimator on average for D10/D01/D11. Geometry or LoRA alone did not establish utility over POP. Their combined D11 interaction is a development candidate signal, not an established subject-aware advantage and not eligible for expansion under the frozen rule. No result is confirmation or a family-wide diffusion/personalization conclusion."]
    Path("reports/sge_basis_score_factorial_v9.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    figures=root/"figures";figures.mkdir(parents=True,exist_ok=True);labels=[r["route"] for r in routes];x=np.arange(len(labels));width=.18;fig,ax=plt.subplots(figsize=(8,4.5))
    for offset,effect in enumerate(("U_D","U_P","U_W","U_S")):ax.bar(x+(offset-1.5)*width,[float(r[effect]) for r in routes],width,label=effect)
    ax.axhline(0,color="black",linewidth=.8);ax.set_xticks(x,labels);ax.set_ylabel("RRMSE utility (positive is better)");ax.legend(ncol=4);fig.tight_layout();fig.savefig(figures/"factorial_effects.png",dpi=180);plt.close(fig)
    natural=list(csv.DictReader((root/"natural_safety.csv").open()));fig,ax=plt.subplots(figsize=(6,5))
    for route in labels:
        rows=[r for r in natural if r["method"]==f"DIFF-{route}-CAL"];ax.scatter([float(r["nonartifact_preservation"]) for r in rows],[float(r["eog_coherence_reduction"]) for r in rows],label=route,alpha=.75)
    ax.set_xlabel("non-artifact preservation");ax.set_ylabel("EOG coherence reduction");ax.legend();fig.tight_layout();fig.savefig(figures/"attenuation_preservation.png",dpi=180);plt.close(fig)
    calibration=list(csv.DictReader((root/"support_calibration.csv").open()));fig,axes=plt.subplots(1,2,figsize=(8,3.8))
    for route in labels:
        rows=[r for r in calibration if r["candidate"]==route];axes[0].hist([float(r["gamma"]) for r in rows],bins=np.linspace(0,1,11),alpha=.45,label=route);axes[1].hist([float(r["w"]) for r in rows],bins=np.linspace(0,1,11),alpha=.45,label=route)
    axes[0].set_title("support-only gamma");axes[1].set_title("personalization weight w");axes[1].legend();fig.tight_layout();fig.savefig(figures/"support_calibration.png",dpi=180);plt.close(fig)
    summary={"status":"completed_v9","diagnostic_decision":decision,"full_one_seed_status":"AUTHORIZED_PENDING" if decision["full_one_seed_authorized"] else "NOT_RUN_NO_DIAGNOSTIC_CANDIDATE","confirmation_evidence":False};_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


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
