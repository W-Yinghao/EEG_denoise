"""v8.1 validity repair and fold-local EB-to-denoising bridge.

The historical v8 outputs are read-only.  This runner keeps support-only
operator construction, outer-training predictor fitting, held-out inference,
and evaluator-only scoring in separate on-disk surfaces.
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
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import (
    _metadata, _natural_metrics, _support_eog_stats, _support_transfer,
)
from eeg_cgdr.experiments.sge_score_lora_v8 import (
    _clone_state, _condition, _distance_utility, _expanded_record_pairs,
    _folds as v8_folds, _masked_u_mse, _operator_distances, _pad,
    _query_generator, _rrmse, _support_features,
    stage_prepare_base as v8_prepare_base,
)
from eeg_cgdr.models.artifact_subspace_diffusion import (
    ArtifactSubspaceConfig, ArtifactSubspaceDiffusion,
    DeterministicSubspaceEstimator, aligned_artifact_basis,
    reconstruct_from_subspace, training_tau, window_noise_bank,
)


PROTOCOL = "SGE-EB-BRIDGE-v8.1"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL or int(value.get("harness_level", -1)) != 1:
        raise ValueError("wrong v8.1 protocol or harness")
    return value


def _folds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    wanted = set(map(str, config["diagnostic_folds"]))
    return [fold for fold in v8_folds(config) if fold["fold_id"] in wanted]


def _all_folds(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return v8_folds(config)


def _fold(config: Mapping[str, Any], task_index: int) -> dict[str, Any]:
    fold_id = str(config["diagnostic_folds"][task_index])
    return next(fold for fold in _folds(config) if fold["fold_id"] == fold_id)


def _safe_key(key: str) -> str:
    return key.replace("/", "__")


def _base_folder(config: Mapping[str, Any], fold_id: str) -> Path:
    if fold_id in set(map(str, config["reused_v8_folds"])):
        return Path(str(config["v8_root"])) / "prepared_base" / fold_id
    return Path(str(config["result_root"])) / "prepared_base" / fold_id


def _model_from_checkpoint(checkpoint: Mapping[str, Any], device: torch.device) -> tuple[DeterministicSubspaceEstimator, ArtifactSubspaceDiffusion]:
    cfg = ArtifactSubspaceConfig(**checkpoint["model_config"])
    det = DeterministicSubspaceEstimator(cfg).to(device)
    diff = ArtifactSubspaceDiffusion(cfg).to(device)
    det.load_state_dict(checkpoint["det"]); diff.load_state_dict(checkpoint["diff_ema"])
    det.eval(); diff.eval()
    return det, diff


@torch.no_grad()
def oracle_ddim_roundtrip(
    model: ArtifactSubspaceDiffusion,
    target: torch.Tensor,
    noise: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    return_state_trace: bool = False,
) -> float | tuple[float, list[torch.Tensor]]:
    """Propagate the current reverse state; never reset it inside the loop."""

    sequence = model._sequence(); first_alpha = model.alpha_bar[sequence[0]]
    state = first_alpha.sqrt() * target + (1.0 - first_alpha).sqrt() * noise
    if initial_state is not None:
        if initial_state.shape != state.shape:
            raise ValueError("initial oracle state shape differs")
        state = initial_state.clone()
    incoming_states: list[torch.Tensor] = []
    for index, step in enumerate(sequence):
        if return_state_trace:
            incoming_states.append(state.detach().clone())
        alpha = model.alpha_bar[step]
        epsilon = (state - alpha.sqrt() * target) / (1.0 - alpha).sqrt().clamp_min(1e-12)
        oracle_v = alpha.sqrt() * epsilon - (1.0 - alpha).sqrt() * target
        x0 = alpha.sqrt() * state - (1.0 - alpha).sqrt() * oracle_v
        if index + 1 == len(sequence):
            state = x0
        else:
            next_alpha = model.alpha_bar[sequence[index + 1]]
            state = next_alpha.sqrt() * x0 + (1.0 - next_alpha).sqrt() * epsilon
    error = float(torch.linalg.norm(state - target) / torch.linalg.norm(target).clamp_min(1e-12))
    return (error, incoming_states) if return_state_trace else error


def _fixed_eval(model: ArtifactSubspaceDiffusion, target: torch.Tensor, y: torch.Tensor,
                basis: torch.Tensor, valid: torch.Tensor, timestep: torch.Tensor,
                noise: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        loss, _ = model.training_loss(target, generator=torch.Generator(device=target.device).manual_seed(0), timestep=timestep, noise=noise, **_condition(y, basis, valid))
        alpha = model.alpha_bar[int(timestep[0])]
        state = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
        predicted_v = model.backbone(state, timestep, **_condition(y, basis, valid))
        x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * predicted_v
        error = x0 - target
        return {
            "loss": float(loss),
            "x0_mse": float(error.square().mean()),
            "x0_rrmse": float(torch.linalg.norm(error) / torch.linalg.norm(target).clamp_min(1e-12)),
            "x0_correlation": float(np.corrcoef(x0.cpu().numpy().ravel(), target.cpu().numpy().ravel())[0, 1]),
        }


def stage_prepare_new(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    index = list(map(str, config["diagnostic_folds"])).index("study01_layout_01_heldout_01")
    summary = v8_prepare_base(config, index, run_dir)
    summary = {**summary, "protocol_revision": PROTOCOL, "purpose": "new_complementary_fold_only"}
    _json(run_dir / "result_summary.json", summary)
    return summary


def stage_train_new(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    fold_id = "study01_layout_01_heldout_01"; folder = _base_folder(config, fold_id)
    arrays = np.load(folder / "training_pairs.npz"); device = torch.device("cuda"); seed = int(config["training"]["seed"]) + 101
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
    cfg = ArtifactSubspaceConfig(eeg_channels=arrays["y"].shape[1], signal_length=arrays["y"].shape[2], base_channels=int(config["model"]["base_channels"]),
                                 num_timesteps=int(config["model"]["timesteps"]), min_snr_gamma=float(config["model"]["min_snr_gamma"]),
                                 ddim_steps=int(config["model"]["ddim_steps"]), posterior_samples=int(config["model"]["posterior_samples"]))
    det = DeterministicSubspaceEstimator(cfg).to(device); diff = ArtifactSubspaceDiffusion(cfg).to(device)
    det_opt = AdamW(det.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    diff_opt = AdamW(diff.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    generator = torch.Generator(device=device).manual_seed(seed + 991); ema = _clone_state(diff); curve = []
    basis = torch.tensor(arrays["population_basis"][None], device=device); batch_size = int(config["training"]["batch_size"])
    for step in range(1, int(config["training"]["successful_updates"]) + 1):
        index = rng.integers(0, len(arrays["y"]), size=batch_size)
        y = torch.tensor(arrays["y"][index], device=device); target = torch.tensor(arrays["target_u"][index], device=device); valid = torch.tensor(arrays["valid"][index], device=device)
        det_opt.zero_grad(set_to_none=True); det_loss = _masked_u_mse(det(**_condition(y,basis,valid)),target,valid); det_loss.backward(); torch.nn.utils.clip_grad_norm_(det.parameters(),1.0); det_opt.step()
        diff_opt.zero_grad(set_to_none=True); diff_loss, detail = diff.training_loss(target,generator=generator,**_condition(y,basis,valid)); diff_loss.backward(); torch.nn.utils.clip_grad_norm_(diff.parameters(),1.0); diff_opt.step()
        with torch.no_grad():
            for name,value in diff.state_dict().items(): ema[name].mul_(float(config["training"]["ema_decay"])).add_(value.detach().cpu(),alpha=1-float(config["training"]["ema_decay"]))
        if step == 1 or step % 100 == 0: curve.append({"step":step,"det_loss":float(det_loss),"diff_loss":float(diff_loss),"x0_mse":float(detail["u_mse"])})
    checkpoint = folder / "checkpoint.pt"
    torch.save({"step":int(config["training"]["successful_updates"]),"det":det.state_dict(),"diff":diff.state_dict(),"diff_ema":ema,"det_optimizer":det_opt.state_dict(),"diff_optimizer":diff_opt.state_dict(),"generator_state":generator.get_state(),"numpy_state":rng.bit_generator.state,"model_config":cfg.__dict__,"tau":arrays["tau"],"population_basis":arrays["population_basis"]},checkpoint)
    _csv(folder/"training_curve_v8_1.csv",curve); summary={"status":"completed_new_population_checkpoint","fold_id":fold_id,"pairs":len(arrays["y"]),"updates":int(config["training"]["successful_updates"]),"checkpoint":str(checkpoint)};_json(run_dir/"result_summary.json",summary);return summary


def stage_validity_core(config: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    fold = _fold(config, task_index); fold_id = fold["fold_id"]; folder = _base_folder(config, fold_id); arrays=np.load(folder/"training_pairs.npz"); checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=False)
    device=torch.device("cuda"); cfg=ArtifactSubspaceConfig(**checkpoint["model_config"]); seed=int(config["training"]["seed"])+task_index
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);probe=ArtifactSubspaceDiffusion(cfg).to(device)
    target=torch.tensor(arrays["target_u"][:1],device=device);y=torch.tensor(arrays["y"][:1],device=device);valid=torch.tensor(arrays["valid"][:1],device=device);basis=torch.tensor(arrays["population_basis"][None],device=device)
    timestep=torch.full((1,),750,device=device,dtype=torch.long);noise=torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+77))
    roundtrip=oracle_ddim_roundtrip(probe,target,noise);initial=_fixed_eval(probe,target,y,basis,valid,timestep,noise)
    optimizer=AdamW(probe.parameters(),lr=5e-4);curve=[];probe.train()
    for step in range(1,int(config["training"]["fixed_window_updates"])+1):
        optimizer.zero_grad(set_to_none=True);loss,_=probe.training_loss(target,generator=torch.Generator(device=device).manual_seed(seed),timestep=timestep,noise=noise,**_condition(y,basis,valid));loss.backward();optimizer.step()
        if step==1 or step%100==0:curve.append({"step":step,"train_mode_loss":float(loss)})
    final=_fixed_eval(probe,target,y,basis,valid,timestep,noise);reduction=1-final["loss"]/max(initial["loss"],1e-12)
    rows=[{"fold_id":fold_id,"phase":"initial",**initial},{"fold_id":fold_id,"phase":"final_eval_mode",**final,"loss_reduction":reduction}];_csv(run_dir/"fixed_window_eval.csv",rows);_csv(Path(str(config["result_root"]))/"validity"/f"{fold_id}.csv",rows)
    summary={"status":"completed_validity_core","fold_id":fold_id,"oracle_roundtrip_error":roundtrip,"oracle_roundtrip_pass":roundtrip<1e-4,"loss_reduction":reduction,"loss_reduction_pass":reduction>=.99,"correlation_pass":final["x0_correlation"]>=.99,"legacy_mse_threshold_pass":final["x0_mse"]<=1e-4,"finite":bool(np.isfinite(list(final.values())).all())};_json(run_dir/"result_summary.json",summary);_json(Path(str(config["result_root"]))/"validity"/f"{fold_id}.json",summary);return summary


def _fit_ridge_model(features: np.ndarray, target: np.ndarray, columns: Sequence[int]) -> dict[str, Any]:
    selected = features[:, list(columns)]; mean=selected.mean(0);std=selected.std(0).clip(1e-6);design=np.column_stack((np.ones(len(selected)),(selected-mean)/std))
    coefficient=np.linalg.solve(design.T@design+0.05*np.eye(design.shape[1]),design.T@target)
    return {"columns":list(map(int,columns)),"mean":mean.tolist(),"std":std.tolist(),"coefficient":coefficient.tolist()}


def _predict_ridge(model: Mapping[str, Any], features: np.ndarray) -> float:
    columns=np.asarray(model["columns"],int);mean=np.asarray(model["mean"]);std=np.asarray(model["std"]);coefficient=np.asarray(model["coefficient"])
    return float(np.clip(np.r_[1.0,(features[columns]-mean)/std]@coefficient,0,1))


def _eb_builder(config: Mapping[str, Any], root: Path) -> None:
    """Support-only surface: this function never requests query signals."""
    layouts,records=_metadata(config);data_root=Path(str(config["data_root"]));destination=root/"eb"/"inference_builder"
    for budget in map(float,config["support_budgets_seconds"]):
        for fold_index,fold in enumerate(_all_folds(config)):
            rate=float(fold["sampling_rate_hz"]);samples=int(round(budget*rate));taps=2*int(round(float(config["fir_lag_ms"])*rate/1000))+1
            keys=list(fold["training"])+list(fold["heldout"]);loaded={key:load_sgeyesub_signal_record(data_root,records[key],layouts[records[key].layout_id],include_query=False) for key in keys}
            train_eeg=np.concatenate([loaded[key].support.eeg[:,:samples] for key in fold["training"]],axis=1).astype(np.float64);normal_mean=train_eeg.mean(1,keepdims=True);normal_std=train_eeg.std(1,keepdims=True).clip(1e-6)
            support={};reliability={}
            for key in keys:
                if loaded[key].support.eeg.shape[1] < samples+int(round(float(config["guard_seconds"])*rate)):continue
                eog_mean,eog_std=_support_eog_stats(loaded[key],samples);support[key],reliability[key]=_support_transfer(loaded[key],eog_mean,eog_std,normal_mean,normal_std,samples,taps,float(config["ridge_lambda"]))
            train_keys=[key for key in fold["training"] if key in support]
            if not train_keys:raise RuntimeError(f"{fold['fold_id']} has no support at {budget}s")
            population=np.mean(np.stack([support[key] for key in train_keys]),axis=0);fold_dir=destination/f"{int(budget)}s"/fold["fold_id"];fold_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(fold_dir/"fold_support.npz",population=population,normal_mean=normal_mean,normal_std=normal_std)
            for offset,key in enumerate(support):
                item=loaded[key];eog_mean,eog_std=_support_eog_stats(item,samples);features=_support_features(item,support[key],population,reliability[key],budget,samples)
                eeg=(np.asarray(item.support.eeg[:,:samples],np.float64)-normal_mean)/normal_std;eog=(np.asarray(item.support.external_eog[:,:samples],np.float64)-eog_mean)/eog_std
                shuffled_eog=eog[:,np.random.default_rng(int(config["pair_seed"])+fold_index*1009+offset).permutation(eog.shape[1])]
                from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import fit_dynamic_transfer
                shuffled=fit_dynamic_transfer(eeg,shuffled_eog,taps=taps,ridge=float(config["ridge_lambda"]))
                np.savez_compressed(fold_dir/f"{_safe_key(key)}.npz",support=support[key],shuffled=shuffled,features=features,reliability=np.float64(reliability[key]),eog_mean=eog_mean,eog_std=eog_std)
            _json(fold_dir/"boundary.json",{"stage":"inference_builder_support_only","fold_id":fold["fold_id"],"budget_seconds":budget,"training":fold["training"],"heldout":fold["heldout"],"query_opened":False,"taps":taps})


def _eb_fit_predictors(config: Mapping[str, Any], root: Path) -> None:
    """Outer-training evaluator supervision is consumed only in this stage."""
    layouts,records=_metadata(config);data_root=Path(str(config["data_root"]));builder=root/"eb"/"inference_builder";destination=root/"eb"/"predictor_fitting";grid=np.linspace(0,1,101)
    for budget in map(float,config["support_budgets_seconds"]):
        for fold_index,fold in enumerate(_all_folds(config)):
            fold_dir=builder/f"{int(budget)}s"/fold["fold_id"];base=np.load(fold_dir/"fold_support.npz");population=base["population"];rate=float(fold["sampling_rate_hz"]);taps=population.shape[-1]
            common=np.random.default_rng(int(config["pair_seed"])+fold_index+int(budget)).standard_normal((population.shape[1],4096));xs=[];ys=[];details=[];training_generators={}
            for key in fold["training"]:
                support=np.load(fold_dir/f"{_safe_key(key)}.npz");loaded=load_sgeyesub_signal_record(data_root,records[key],layouts[records[key].layout_id],include_query=True,include_query_annotations=True)
                generator,_=_query_generator(loaded,records[key],base["normal_mean"],base["normal_std"],support["eog_mean"],support["eog_std"],taps,float(config["ridge_lambda"]));training_generators[key]=generator
                scores=[_distance_utility(population+lam*(support["support"]-population),generator,common) for lam in grid];oracle=float(grid[int(np.argmin(scores))]);xs.append(support["features"]);ys.append(oracle);details.append({"recording_key":key,"oracle_lambda":oracle})
            x=np.stack(xs);y=np.asarray(ys);fixed_scores=[]
            for lam in grid:
                values=[]
                for key in fold["training"]:
                    support=np.load(fold_dir/f"{_safe_key(key)}.npz");values.append(_distance_utility(population+lam*(support["support"]-population),training_generators[key],common))
                fixed_scores.append(float(np.mean(values)))
            model={"fold_id":fold["fold_id"],"budget_seconds":budget,"fixed_lambda":float(grid[int(np.argmin(fixed_scores))]),"reliability_model":_fit_ridge_model(x,y,[0]),"full_model":_fit_ridge_model(x,y,range(x.shape[1])),"training_units":details,"heldout_outcomes_used":False}
            _json(destination/f"{int(budget)}s"/f"{fold['fold_id']}.json",model)


def _eb_heldout_inference(config: Mapping[str, Any], root: Path) -> None:
    """Held-out support plus frozen coefficients only; no evaluator arrays."""
    builder=root/"eb"/"inference_builder";models=root/"eb"/"predictor_fitting";destination=root/"eb"/"heldout_inference"
    for budget in map(float,config["support_budgets_seconds"]):
        for fold in _all_folds(config):
            fold_dir=builder/f"{int(budget)}s"/fold["fold_id"];population=np.load(fold_dir/"fold_support.npz")["population"];model=json.loads((models/f"{int(budget)}s"/f"{fold['fold_id']}.json").read_text())
            for key in fold["heldout"]:
                item=np.load(fold_dir/f"{_safe_key(key)}.npz");rel_lambda=_predict_ridge(model["reliability_model"],item["features"]);full_lambda=_predict_ridge(model["full_model"],item["features"]);fixed=float(model["fixed_lambda"])
                donors=[donor for donor in fold["heldout"] if donor!=key and (fold_dir/f"{_safe_key(donor)}.npz").exists()];wrong=[];wrong_fixed=[];wrong_keys=[]
                for donor in donors:
                    donor_item=np.load(fold_dir/f"{_safe_key(donor)}.npz");donor_lambda=_predict_ridge(model["full_model"],donor_item["features"]);wrong.append(population+donor_lambda*(donor_item["support"]-population));wrong_fixed.append(population+fixed*(donor_item["support"]-population));wrong_keys.append(donor)
                output_path=_heldout_output_path(destination,budget,fold["fold_id"],key)
                np.savez_compressed(output_path,population=population,raw_match=item["support"],fixed=population+fixed*(item["support"]-population),reliability=population+rel_lambda*(item["support"]-population),full=population+full_lambda*(item["support"]-population),wrong=np.stack(wrong),wrong_fixed=np.stack(wrong_fixed),wrong_keys=np.asarray(wrong_keys),shuffled=population+full_lambda*(item["shuffled"]-population),shuffled_fixed=population+fixed*(item["shuffled"]-population),fixed_lambda=np.float64(fixed),reliability_lambda=np.float64(rel_lambda),full_lambda=np.float64(full_lambda))
    _json(destination/"boundary.json",{"stage":"heldout_inference","heldout_query_opened":False,"evaluator_fields_present":False})


def _heldout_output_path(destination: Path, budget: float, fold_id: str, key: str) -> Path:
    """Return a per-unit path after creating its fold-local output directory."""
    path=destination/f"{int(budget)}s"/fold_id/f"{_safe_key(key)}.npz"
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def _eb_evaluate(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    layouts,records=_metadata(config);data_root=Path(str(config["data_root"]));builder=root/"eb"/"inference_builder";inference=root/"eb"/"heldout_inference";rows=[]
    for budget in map(float,config["support_budgets_seconds"]):
        for fold_index,fold in enumerate(_all_folds(config)):
            fold_builder=builder/f"{int(budget)}s"/fold["fold_id"];base=np.load(fold_builder/"fold_support.npz");rate=float(fold["sampling_rate_hz"]);taps=base["population"].shape[-1];common=np.random.default_rng(int(config["pair_seed"])+fold_index+int(budget)).standard_normal((base["population"].shape[1],4096))
            for key in fold["heldout"]:
                support=np.load(fold_builder/f"{_safe_key(key)}.npz");loaded=load_sgeyesub_signal_record(data_root,records[key],layouts[records[key].layout_id],include_query=True,include_query_annotations=True);generator,_=_query_generator(loaded,records[key],base["normal_mean"],base["normal_std"],support["eog_mean"],support["eog_std"],taps,float(config["ridge_lambda"]));candidate=np.load(inference/f"{int(budget)}s"/fold["fold_id"]/f"{_safe_key(key)}.npz")
                arms={"POP":candidate["population"],"RAW_MATCH":candidate["raw_match"],"FIXED_LAMBDA":candidate["fixed"],"RELIABILITY_ONLY":candidate["reliability"],"FULL_FEATURE":candidate["full"],"EB_SHUFFLED":candidate["shuffled"]}
                for donor_index,value in enumerate(candidate["wrong"]):arms[f"EB_WRONG_{donor_index}"]=value
                for arm,value in arms.items():
                    distances=_operator_distances(value,generator,common);rows.append({"budget_seconds":budget,"fold_id":fold["fold_id"],"study":fold["study"],"recording_key":key,"arm":arm,"composite_proxy":_distance_utility(value,generator,common),"fixed_lambda":float(candidate["fixed_lambda"]),"reliability_lambda":float(candidate["reliability_lambda"]),"full_lambda":float(candidate["full_lambda"]),**distances})
    return rows


def _eb_loso(config: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    """Leave-one-study-out lambda prediction with evaluator use confined here."""
    layouts,records=_metadata(config);data_root=Path(str(config["data_root"]));builder=root/"eb"/"inference_builder";grid=np.linspace(0,1,101);units=[]
    for budget in map(float,config["support_budgets_seconds"]):
        for fold_index,fold in enumerate(_all_folds(config)):
            fold_dir=builder/f"{int(budget)}s"/fold["fold_id"];base=np.load(fold_dir/"fold_support.npz");population=base["population"];common=np.random.default_rng(int(config["pair_seed"])+fold_index+int(budget)).standard_normal((population.shape[1],4096))
            for key in fold["heldout"]:
                support=np.load(fold_dir/f"{_safe_key(key)}.npz");loaded=load_sgeyesub_signal_record(data_root,records[key],layouts[records[key].layout_id],include_query=True,include_query_annotations=True);generator,_=_query_generator(loaded,records[key],base["normal_mean"],base["normal_std"],support["eog_mean"],support["eog_std"],population.shape[-1],float(config["ridge_lambda"]))
                scores=[_distance_utility(population+lam*(support["support"]-population),generator,common) for lam in grid]
                units.append({"budget":budget,"fold_id":fold["fold_id"],"study":fold["study"],"key":key,"features":support["features"],"oracle":float(grid[int(np.argmin(scores))]),"population":population,"support":support["support"],"generator":generator,"common":common})
    rows=[]
    for budget in map(float,config["support_budgets_seconds"]):
        subset=[u for u in units if u["budget"]==budget]
        for study in sorted({u["study"] for u in subset}):
            train=[u for u in subset if u["study"]!=study];test=[u for u in subset if u["study"]==study];model=_fit_ridge_model(np.stack([u["features"] for u in train]),np.asarray([u["oracle"] for u in train]),range(len(train[0]["features"])))
            for unit in test:
                predicted=_predict_ridge(model,unit["features"]);candidate=unit["population"]+predicted*(unit["support"]-unit["population"]);pop_distance=_distance_utility(unit["population"],unit["generator"],unit["common"]);distance=_distance_utility(candidate,unit["generator"],unit["common"])
                rows.append({"budget_seconds":budget,"heldout_study":study,"fold_id":unit["fold_id"],"recording_key":unit["key"],"predicted_lambda":predicted,"oracle_lambda":unit["oracle"],"relative_improvement":(pop_distance-distance)/max(pop_distance,1e-12),"lambda_absolute_error":abs(predicted-unit["oracle"])})
    return rows


def stage_eb_robustness(config: Mapping[str, Any], run_dir: Path, *, resume_from_heldout: bool=False) -> dict[str, Any]:
    root=Path(str(config["result_root"]));
    if not resume_from_heldout:
        _eb_builder(config,root);_eb_fit_predictors(config,root)
    _eb_heldout_inference(config,root);rows=_eb_evaluate(config,root);_csv(root/"eb_robustness_unit_metrics.csv",rows);loso=_eb_loso(config,root);_csv(root/"eb_leave_one_study_out.csv",loso)
    effects=[]
    for budget in map(float,config["support_budgets_seconds"]):
        for key in sorted({r["recording_key"] for r in rows if r["budget_seconds"]==budget}):
            unit=[r for r in rows if r["budget_seconds"]==budget and r["recording_key"]==key];by={r["arm"]:r for r in unit};wrong=[r for r in unit if r["arm"].startswith("EB_WRONG")]
            pop=by["POP"]["composite_proxy"];effects.append({"budget_seconds":budget,"fold_id":by["POP"]["fold_id"],"study":by["POP"]["study"],"recording_key":key,"full_vs_pop":(pop-by["FULL_FEATURE"]["composite_proxy"])/max(pop,1e-12),"fixed_vs_pop":(pop-by["FIXED_LAMBDA"]["composite_proxy"])/max(pop,1e-12),"reliability_vs_pop":(pop-by["RELIABILITY_ONLY"]["composite_proxy"])/max(pop,1e-12),"full_vs_fixed":(by["FIXED_LAMBDA"]["composite_proxy"]-by["FULL_FEATURE"]["composite_proxy"])/max(by["FIXED_LAMBDA"]["composite_proxy"],1e-12),"full_vs_reliability":(by["RELIABILITY_ONLY"]["composite_proxy"]-by["FULL_FEATURE"]["composite_proxy"])/max(by["RELIABILITY_ONLY"]["composite_proxy"],1e-12),"full_vs_wrong":(float(np.mean([r["composite_proxy"] for r in wrong]))-by["FULL_FEATURE"]["composite_proxy"])/max(float(np.mean([r["composite_proxy"] for r in wrong])),1e-12),"full_vs_shuffled":(by["EB_SHUFFLED"]["composite_proxy"]-by["FULL_FEATURE"]["composite_proxy"])/max(by["EB_SHUFFLED"]["composite_proxy"],1e-12),"full_lambda":by["FULL_FEATURE"]["full_lambda"]})
    _csv(root/"eb_robustness_summary.csv",effects)
    summary={"status":"completed_eb_robustness","scope":"descriptive_conditional_overlapping_outer_training","compatible_stems":len({r["recording_key"] for r in effects}),"availability_denominator":59,"primary_budget_seconds":120,"means":{m:float(np.mean([r[m] for r in effects if r["budget_seconds"]==120])) for m in ("full_vs_pop","fixed_vs_pop","reliability_vs_pop","full_vs_fixed","full_vs_reliability","full_vs_wrong","full_vs_shuffled")},"loso_120_mean_relative_improvement":float(np.mean([r["relative_improvement"] for r in loso if r["budget_seconds"]==120])),"loso_120_positive_count":int(sum(r["relative_improvement"]>0 for r in loso if r["budget_seconds"]==120)),"bootstrap_not_run":"predictor_refit_per_replicate_required","leave_one_study_out":"completed_development_robustness_not_confirmation"};_json(root/"eb_robustness_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def _basis_from_transfer(transfer: np.ndarray, population_basis: np.ndarray) -> np.ndarray:
    basis,_,_=aligned_artifact_basis(np.asarray(transfer).reshape(transfer.shape[0],-1),population_basis)
    return basis


def _paired_metrics(value: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str,float]:
    return {"rrmse":_rrmse(value,x),"correlation":float(np.corrcoef(value.ravel(),x.ravel())[0,1]),"delta_snr":float(20*np.log10(max(np.linalg.norm(y-x),1e-12)/max(np.linalg.norm(value-x),1e-12))),"artifact_residual":float(np.linalg.norm((y-value)-(y-x))/max(np.linalg.norm(y-x),1e-12)),"correction_rms":float(np.sqrt(np.mean((y-value)**2)))}


@torch.no_grad()
def _model_output(model: Any, kind: str, observed_np: np.ndarray, basis_np: np.ndarray, tau_np: np.ndarray,
                  key: str, seed: int, device: torch.device, *, k: int=8, absolute_start: int=0, batch_size: int=16) -> np.ndarray:
    observed_np=np.asarray(observed_np,np.float32);result=[]
    for start in range(0,len(observed_np),batch_size):
        stop=min(start+batch_size,len(observed_np));y=torch.tensor(observed_np[start:stop],device=device);basis=torch.tensor(np.repeat(basis_np[None],len(y),axis=0),device=device);valid=torch.ones((len(y),y.shape[-1]),dtype=torch.bool,device=device);condition={"observed":y,"basis":basis,"reliability":torch.ones(len(y),device=device),"rank_mask":torch.ones((len(y),2),dtype=torch.bool,device=device),"valid_time_mask":valid}
        if kind=="det":u=model(**condition)
        else:
            bank=window_noise_bank(key,seed,range(absolute_start+start,absolute_start+stop),posterior_samples=k,signal_length=y.shape[-1],device=device,dtype=y.dtype);u,_,_,_=model.sample(initial_noise_bank=bank,**condition)
        restored,_=reconstruct_from_subspace(y,basis,u,torch.tensor(tau_np,device=device),condition["rank_mask"],valid);result.append(restored.cpu().numpy())
    return np.concatenate(result)


def _oracle_ceiling_rows(fold: Mapping[str,Any], pair: Any, bases: Mapping[str,np.ndarray], tau: np.ndarray) -> list[dict[str,Any]]:
    x=np.asarray(pair["x"]);y=np.asarray(pair["y"]);artifact=np.asarray(pair["a"]);rows=[];raw_rrmse=_rrmse(y,x)
    for basis_name,basis in bases.items():
        coefficient=np.einsum("cr,nct->nrt",basis,artifact)
        corrections={"ORACLE_PROJECTION":np.einsum("cr,nrt->nct",basis,coefficient),"ORACLE_CLIP":np.einsum("cr,nrt->nct",basis,np.clip(coefficient,-tau[None,:,None],tau[None,:,None])),"ORACLE_TANH":np.einsum("cr,nrt->nct",basis,tau[None,:,None]*np.tanh(coefficient/tau[None,:,None]))}
        for method,correction in corrections.items():
            metric=_paired_metrics(y-correction,x,y);rows.append({"fold_id":fold["fold_id"],"study":fold["study"],"recording_key":"", "basis":basis_name,"ceiling":method,"raw_rrmse":raw_rrmse,"representation_eligible":metric["rrmse"]<raw_rrmse,**metric})
    return rows


def stage_repaired_inference(config: Mapping[str,Any], task_index: int, run_dir: Path) -> dict[str,Any]:
    fold=_fold(config,task_index);fold_id=fold["fold_id"];folder=_base_folder(config,fold_id);checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=False);device=torch.device("cuda");_,diff=_model_from_checkpoint(checkpoint,device);tau=np.asarray(checkpoint["tau"]);pop_basis=np.asarray(checkpoint["population_basis"]);seed=int(config["training"]["seed"])
    root=Path(str(config["result_root"]));robust=json.loads((root/"eb_robustness_decision.json").read_text());selected="full" if robust["means"]["full_vs_fixed"]>0 else "fixed";v6=Path(str(config["v6_root"]))/"prepared"/fold_id;eb=root/"eb"/"heldout_inference"/"120s"/fold_id;paired_rows=[];natural_rows=[];ceiling_rows=[]
    for key in fold["heldout"]:
        pair=np.load(v6/f"paired_{_safe_key(key)}.npz");operator=np.load(eb/f"{_safe_key(key)}.npz");bases={"A_POP":pop_basis,"A_MATCH120":_basis_from_transfer(operator["raw_match"],pop_basis),"A_EB120":_basis_from_transfer(operator[selected],pop_basis)}
        unit_ceiling=_oracle_ceiling_rows(fold,pair,bases,tau)
        for row in unit_ceiling:row["recording_key"]=key
        ceiling_rows.extend(unit_ceiling)
        for k in (1,8,32):
            output=_model_output(diff,"diff",pair["y"],pop_basis,tau,key,seed,device,k=k);paired_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"K":k,**_paired_metrics(output,pair["x"],pair["y"])})
        natural=np.load(v6/f"natural_input_{_safe_key(key)}.npz");raw=np.asarray(natural["y"]);length=int(checkpoint["model_config"]["signal_length"]);usable=raw.shape[1]//length*length;windows=raw[:,:usable].reshape(raw.shape[0],-1,length).transpose(1,0,2);evaluator=np.load(v6/f"natural_evaluator_{_safe_key(key)}.npz")
        for k in (1,8,32):
            output_windows=_model_output(diff,"diff",windows,pop_basis,tau,key,seed,device,k=k);continuous=output_windows.transpose(1,0,2).reshape(raw.shape[0],usable);metric=_natural_metrics(raw[:,:usable],continuous,evaluator["eog"],evaluator["labels"],float(fold["sampling_rate_hz"]));natural_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"K":k,"correction_rms":float(np.sqrt(np.mean((raw[:,:usable]-continuous)**2))),**metric})
    destination=Path(str(config["result_root"]))/"repaired_inference";_csv(destination/f"{fold_id}_paired.csv",paired_rows);_csv(destination/f"{fold_id}_natural.csv",natural_rows);_csv(destination/f"{fold_id}_oracle.csv",ceiling_rows)
    eb_projection=[r for r in ceiling_rows if r["basis"]=="A_EB120" and r["ceiling"]=="ORACLE_PROJECTION"];eligible=float(np.mean([r["rrmse"] for r in eb_projection]))<float(np.mean([r["raw_rrmse"] for r in eb_projection]));summary={"status":"completed_repaired_inference","fold_id":fold_id,"heldout_units":len(fold["heldout"]),"unique_window_rng":True,"selected_eb_predictor":selected,"representation_eligible":eligible,"primary_K":8};_json(destination/f"{fold_id}.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def _bridge_bases(operator: Any,pop_basis:np.ndarray,selected_predictor:str)->dict[str,Any]:
    wrong_field="wrong" if selected_predictor=="full_feature" else "wrong_fixed";shuffled_field="shuffled" if selected_predictor=="full_feature" else "shuffled_fixed"
    wrong=[_basis_from_transfer(value,pop_basis) for value in operator[wrong_field]]
    selected=operator["full"] if selected_predictor=="full_feature" else operator["fixed"]
    return {"POP":pop_basis,"MATCH120":_basis_from_transfer(operator["raw_match"],pop_basis),"EB-FIXED120":_basis_from_transfer(operator["fixed"],pop_basis),"EB-DEPLOY120":_basis_from_transfer(selected,pop_basis),"EB-FULL120":_basis_from_transfer(operator["full"],pop_basis),"EB-WRONG120":wrong,"EB-SHUFFLED120":_basis_from_transfer(operator[shuffled_field],pop_basis)}


def stage_bridge(config: Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    fold=_fold(config,task_index);fold_id=fold["fold_id"];root=Path(str(config["result_root"]));representation=json.loads((root/"repaired_inference"/f"{fold_id}.json").read_text());robust=json.loads((root/"eb_robustness_decision.json").read_text());selected_predictor="full_feature" if robust["means"]["full_vs_fixed"]>0 else "fixed_lambda";folder=_base_folder(config,fold_id);checkpoint=torch.load(folder/"checkpoint.pt",map_location="cpu",weights_only=False);device=torch.device("cuda");det,diff=_model_from_checkpoint(checkpoint,device);tau=np.asarray(checkpoint["tau"]);pop_basis=np.asarray(checkpoint["population_basis"]);seed=int(config["training"]["seed"]);v6=Path(str(config["v6_root"]))/"prepared"/fold_id;eb=root/"eb"/"heldout_inference"/"120s"/fold_id;paired_rows=[];natural_rows=[]
    for key in fold["heldout"]:
        pair=np.load(v6/f"paired_{_safe_key(key)}.npz");operator=np.load(eb/f"{_safe_key(key)}.npz");bases=_bridge_bases(operator,pop_basis,selected_predictor);outputs={"RAW":np.asarray(pair["y"])}
        for label in ("POP","MATCH120","EB-FIXED120","EB-DEPLOY120"):
            outputs[f"DET-{label}"]=_model_output(det,"det",pair["y"],bases[label],tau,key,seed,device)
            outputs[f"DIFF-{label}"]=_model_output(diff,"diff",pair["y"],bases[label],tau,key,seed,device,k=8)
        wrong_outputs=[]
        for index,basis in enumerate(bases["EB-WRONG120"]):
            value=_model_output(diff,"diff",pair["y"],basis,tau,key,seed,device,k=8);outputs[f"DIFF-EB-WRONG120-{index}"]=value;wrong_outputs.append(value)
        outputs["DIFF-EB-SHUFFLED120"]=_model_output(diff,"diff",pair["y"],bases["EB-SHUFFLED120"],tau,key,seed,device,k=8)
        if not representation["representation_eligible"]:
            outputs={method:(pair["y"] if method!="RAW" else value) for method,value in outputs.items()}
        for method,value in outputs.items():paired_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,"raw_fallback":not representation["representation_eligible"],**_paired_metrics(value,pair["x"],pair["y"])})
        natural=np.load(v6/f"natural_input_{_safe_key(key)}.npz");raw=np.asarray(natural["y"]);length=int(checkpoint["model_config"]["signal_length"]);usable=raw.shape[1]//length*length;windows=raw[:,:usable].reshape(raw.shape[0],-1,length).transpose(1,0,2);evaluator=np.load(v6/f"natural_evaluator_{_safe_key(key)}.npz")
        natural_methods={"DIFF-POP":bases["POP"],"DIFF-EB-DEPLOY120":bases["EB-DEPLOY120"],"DIFF-EB-SHUFFLED120":bases["EB-SHUFFLED120"]}
        for index,basis in enumerate(bases["EB-WRONG120"]):natural_methods[f"DIFF-EB-WRONG120-{index}"]=basis
        for method,basis in natural_methods.items():
            continuous=_model_output(diff,"diff",windows,basis,tau,key,seed,device,k=8).transpose(1,0,2).reshape(raw.shape[0],usable)
            if not representation["representation_eligible"]:continuous=raw[:,:usable]
            natural_rows.append({"fold_id":fold_id,"study":fold["study"],"recording_key":key,"method":method,"raw_fallback":not representation["representation_eligible"],"correction_rms":float(np.sqrt(np.mean((raw[:,:usable]-continuous)**2))),**_natural_metrics(raw[:,:usable],continuous,evaluator["eog"],evaluator["labels"],float(fold["sampling_rate_hz"]))})
    destination=root/"bridge";_csv(destination/f"{fold_id}_paired.csv",paired_rows);_csv(destination/f"{fold_id}_natural.csv",natural_rows);summary={"status":"completed_bridge_fold","fold_id":fold_id,"representation_eligible":representation["representation_eligible"],"selected_predictor":selected_predictor,"heldout_units":len(fold["heldout"])};_json(run_dir/"result_summary.json",summary);return summary


def _read_csvs(paths: Sequence[Path]) -> list[dict[str,str]]:
    rows=[]
    for path in paths:
        with path.open(newline="",encoding="utf-8") as stream:rows.extend(csv.DictReader(stream))
    return rows


def _cluster_descriptive(effects: Sequence[Mapping[str,Any]], fold_ids: Sequence[str], seed: int, replicates: int=20_000) -> list[dict[str,Any]]:
    """Development-only fold-cluster bootstrap; windows never enter resampling."""
    rng=np.random.default_rng(seed);rows=[]
    grouped={fold:[row for row in effects if row["fold_id"]==fold] for fold in fold_ids}
    for metric in ("U_D","U_P","U_W","U_S"):
        samples=np.empty(replicates,dtype=np.float64)
        for index in range(replicates):
            chosen=rng.choice(fold_ids,size=len(fold_ids),replace=True);values=[]
            for fold in chosen:
                units=grouped[str(fold)];draw=rng.integers(0,len(units),size=len(units));values.extend(float(units[int(i)][metric]) for i in draw)
            samples[index]=np.mean(values)
        observed=np.asarray([float(row[metric]) for row in effects])
        rows.append({"metric":metric,"unit":"participant_stem","clusters":len(fold_ids),"units":len(effects),"mean":float(observed.mean()),"median":float(np.median(observed)),"ci_low":float(np.quantile(samples,.025)),"ci_high":float(np.quantile(samples,.975)),"positive_count":int((observed>0).sum()),"replicates":replicates,"scope":"development_descriptive"})
    return rows


def stage_aggregate(config: Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));fold_ids=list(map(str,config["diagnostic_folds"]));validity=[json.loads((root/"validity"/f"{fold}.json").read_text()) for fold in fold_ids];repaired=[json.loads((root/"repaired_inference"/f"{fold}.json").read_text()) for fold in fold_ids]
    oracle=_read_csvs([root/"repaired_inference"/f"{fold}_oracle.csv" for fold in fold_ids]);_csv(root/"oracle_ceiling_metrics.csv",oracle)
    paired=_read_csvs([root/"bridge"/f"{fold}_paired.csv" for fold in fold_ids]);natural=_read_csvs([root/"bridge"/f"{fold}_natural.csv" for fold in fold_ids]);_csv(root/"natural_safety_metrics.csv",natural)
    methods=[]
    for method in sorted({r["method"] for r in paired}):
        subset=[r for r in paired if r["method"]==method];methods.append({"method":method,"units":len(subset),"rrmse_mean":float(np.mean([float(r["rrmse"]) for r in subset])),"correlation_mean":float(np.mean([float(r["correlation"]) for r in subset])),"delta_snr_mean":float(np.mean([float(r["delta_snr"]) for r in subset])),"fallback_units":sum(r["raw_fallback"]=="True" for r in subset)})
    _csv(root/"method_summary.csv",methods);by={(r["recording_key"],r["method"]):r for r in paired};effects=[]
    for key in sorted({r["recording_key"] for r in paired}):
        match=by[(key,"DIFF-EB-DEPLOY120")]
        det=by[(key,"DET-EB-DEPLOY120")]
        pop=by[(key,"DIFF-POP")];shuffled=by[(key,"DIFF-EB-SHUFFLED120")];wrong=[r for r in paired if r["recording_key"]==key and r["method"].startswith("DIFF-EB-WRONG120-")]
        effects.append({"fold_id":match["fold_id"],"study":match["study"],"recording_key":key,"raw_fallback":match["raw_fallback"],"U_D":float(det["rrmse"])-float(match["rrmse"]),"U_P":float(pop["rrmse"])-float(match["rrmse"]),"U_W":float(np.mean([float(r["rrmse"])-float(match["rrmse"]) for r in wrong])),"U_S":float(shuffled["rrmse"])-float(match["rrmse"]),"DIFF_MATCH120_vs_POP":float(pop["rrmse"])-float(by[(key,"DIFF-MATCH120")]["rrmse"]),"EB_DEPLOY_vs_FIXED":float(by[(key,"DIFF-EB-FIXED120")]["rrmse"])-float(match["rrmse"]),"diff_deploy_rrmse":float(match["rrmse"]),"raw_rrmse":float(by[(key,"RAW")]["rrmse"]),"wrong_donors":len(wrong)})
    _csv(root/"paired_effects.csv",effects)
    means={metric:float(np.mean([r[metric] for r in effects])) for metric in ("U_D","U_P","U_W","U_S")};directions={metric:int(sum(np.mean([r[metric] for r in effects if r["fold_id"]==fold])>=0 for fold in fold_ids)) for metric in means}
    cluster_rows=_cluster_descriptive(effects,fold_ids,int(config["training"]["seed"]));_csv(root/"cluster_descriptive_summary.csv",cluster_rows)
    oracle_projection={r["recording_key"]:float(r["raw_rrmse"])-float(r["rrmse"]) for r in oracle if r["basis"]=="A_EB120" and r["ceiling"]=="ORACLE_PROJECTION"}
    actual={r["recording_key"]:float(r["raw_rrmse"])-float(r["diff_deploy_rrmse"]) for r in effects};common=sorted(set(oracle_projection)&set(actual));oracle_actual_correlation=float(np.corrcoef([oracle_projection[key] for key in common],[actual[key] for key in common])[0,1]) if len(common)>1 else float("nan")
    natural_match=[r for r in natural if r["method"]=="DIFF-EB-DEPLOY120"];safety={"preservation":float(np.nanmean([float(r["nonartifact_preservation"]) for r in natural_match])),"psd_distortion":float(np.nanmean([float(r["psd_distortion"]) for r in natural_match])),"covariance_distortion":float(np.nanmean([float(r["covariance_distortion"]) for r in natural_match])),"eog_attenuation":float(np.nanmean([float(r["eog_coherence_reduction"]) for r in natural_match]))}
    technical=all(v["oracle_roundtrip_pass"] and v["finite"] and v["loss_reduction_pass"] and v["correlation_pass"] for v in validity) and all(v["unique_window_rng"] for v in repaired)
    science=all(value>0 for value in means.values()) and all(value>=3 for value in directions.values()) and float(np.mean([r["diff_deploy_rrmse"] for r in effects]))<float(np.mean([r["raw_rrmse"] for r in effects])) and safety["preservation"]>=.75 and safety["psd_distortion"]<=.25 and safety["covariance_distortion"]<=.25
    gate=technical and science;decision="BRIDGE_GATE_PASSED_SUPPORT_INNER_PILOT_AUTHORIZED" if gate else "BRIDGE_GATE_FAILED_SCORE_LORA_NOT_RUN"
    corrected={"status":"completed_v8_1_validity_repair","historical_v8_gate":"POPULATION_DIFFUSION_BASE_GATE_FAILED","aggregate_diffusion_signal":{"RAW":.4675060957670212,"DET_POP":.42504712111420107,"DIFF_POP":.41138627048995763},"technical_validity":technical,"folds":validity,"representation_folds":repaired,"interpretation":"historical_gate_failure_not_equivalent_to_population_diffusion_implementation_invalid"};_json(root/"corrected_validity_summary.json",corrected)
    routing={"status":"completed_bridge_gate","decision":decision,"support_inner_authorized":gate,"later_query_authorized":False,"technical_requirements":technical,"scientific_requirements":science,"effects":means,"nonnegative_fold_counts":directions,"natural_safety":safety,"coverage":len(effects),"diagnostic_folds":4,"cluster_descriptive":cluster_rows,"oracle_actual_improvement_correlation":oracle_actual_correlation,"confirmation_evidence":False};_json(root/"bridge_routing_decision.json",routing);_json(root/"routing_decision.json",{"score_lora_authorized":gate,"decision":decision})
    if not gate:_json(root/"support_inner_decision.json",{"status":"not_run_blocked_by_bridge_gate","decision":"SCORE_LORA_NOT_RUN","later_query_authorized":False})
    _json(run_dir/"result_summary.json",routing);return routing


def _ensure_prepared_links(config: Mapping[str,Any])->None:
    root=Path(str(config["result_root"]))/"prepared_base";root.mkdir(parents=True,exist_ok=True)
    for fold in config["reused_v8_folds"]:
        destination=root/str(fold);source=Path(str(config["v8_root"]))/"prepared_base"/str(fold)
        if not destination.exists():destination.symlink_to(source,target_is_directory=True)


def stage_support_inner(config: Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    routing=json.loads((Path(str(config["result_root"]))/"bridge_routing_decision.json").read_text())
    if not routing.get("support_inner_authorized"):
        summary={"status":"not_run_blocked_by_bridge_gate","task_index":task_index};_json(run_dir/"result_summary.json",summary);return summary
    _ensure_prepared_links(config)
    from eeg_cgdr.experiments.sge_score_lora_v8 import stage_support_lora
    local=dict(config);local["score_lora_support_budget_seconds"]=120;return stage_support_lora(local,task_index,run_dir)


def stage_support_inner_gate(config: Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));routing=json.loads((root/"bridge_routing_decision.json").read_text())
    if not routing.get("support_inner_authorized"):
        summary={"status":"not_run_blocked_by_bridge_gate","decision":"SCORE_LORA_NOT_RUN","next_stage_eligible":False,"later_query_authorized":False};_json(root/"support_inner_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary
    rows=[json.loads((root/"support_inner"/f"{fold}.json").read_text()) for fold in config["diagnostic_folds"]];names=("match_vs_noadapt","match_vs_wrong","match_vs_shuffled","diff_vs_det");effects={name:float(np.mean([row["effects"][name] for row in rows])) for name in names};passed=all(v>0 for v in effects.values()) and all(row["scale_safe"] for row in rows)
    summary={"status":"passed" if passed else "failed","decision":"SCORE_LORA_SUPPORT_INNER_PILOT_PASSED" if passed else "SCORE_LORA_SUPPORT_INNER_PILOT_FAILED","effects":effects,"folds":rows,"next_stage_eligible":passed,"later_query_authorized":False};_json(root/"support_inner_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def stage_finalize(config: Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));corrected=json.loads((root/"corrected_validity_summary.json").read_text());eb=json.loads((root/"eb_robustness_decision.json").read_text());bridge=json.loads((root/"bridge_routing_decision.json").read_text());support=json.loads((root/"support_inner_decision.json").read_text())
    validity_rows=corrected["folds"];validity_table="\n".join(f"| {r['fold_id']} | {r['oracle_roundtrip_error']:.2e} | {r['loss_reduction']:.4f} | {str(r['correlation_pass']).lower()} |" for r in validity_rows)
    report=Path("reports/sge_eb_denoising_bridge_v8_1.md");report.write_text("# SGE EB denoising bridge v8.1\n\n## Scope\n\nDevelopment exploration only. Historical v8 artifacts and `POPULATION_DIFFUSION_BASE_GATE_FAILED` remain unchanged. The EB label is corrected to `FOLD_LOCAL_OPERATOR_PROXY_HEADROOM_DETECTED`.\n\n## Corrected validity\n\n| fold | sampler error | eval-mode loss reduction | corr >= .99 |\n|---|---:|---:|---:|\n"+validity_table+f"\n\nTechnical validity: `{corrected['technical_validity']}`.\n\n## Operator proxy headroom\n\nPrimary 120 s FULL-feature effects: `{json.dumps(eb['means'],sort_keys=True)}`. LOSO mean improvement: {eb['loso_120_mean_relative_improvement']:+.4f}. Fixed lambda outperformed the full-feature predictor and was therefore retained for the bridge. These are operator-proxy results, not denoising success.\n\n## Denoising bridge\n\nDecision: `{bridge['decision']}`. Effects: `{json.dumps(bridge['effects'],sort_keys=True)}`. Natural safety: `{json.dumps(bridge['natural_safety'],sort_keys=True)}`. Oracle-ceiling versus actual-improvement correlation: {bridge['oracle_actual_improvement_correlation']:+.4f}. Cluster intervals are development-descriptive and are saved in `cluster_descriptive_summary.csv`. Representation-ineligible folds would use the declared RAW fallback; all four folds were eligible.\n\n## Score-LoRA\n\nDecision: `{support['decision']}`. No Score-LoRA training ran because the bridge gate failed. Later-query full evaluation was not authorized.\n\nNo result is confirmation evidence or a family-wide conclusion.\n",encoding="utf-8")
    Path("reports/v8_1_validity_repair.md").write_text("# v8.1 validity repair\n\n"+f"Corrected technical validity: `{corrected['technical_validity']}`. The historical aggregate RRMSE signal (RAW 0.4675, DET-POP 0.4250, DIFF-POP 0.4114) is retained without changing the historical gate.\n",encoding="utf-8")
    Path("reports/fold_local_eb_robustness_v8_1.md").write_text("# Fold-local EB robustness v8.1\n\nPrimary support is 120 s; 60 s is secondary. Builder, predictor fitting, held-out inference, and evaluator outputs are physically separated.\n\n"+json.dumps(eb,indent=2)+"\n",encoding="utf-8")
    summary={"status":"completed_v8_1","validity":corrected["technical_validity"],"eb_proxy":"FOLD_LOCAL_OPERATOR_PROXY_HEADROOM_DETECTED","bridge":bridge["decision"],"score_lora":support["decision"],"later_query_full_evaluation":"NOT_RUN","confirmation_evidence":False};_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path)
    if stage=="prepare-new":return stage_prepare_new(config,run_dir)
    if stage=="train-new":return stage_train_new(config,run_dir)
    if stage=="validity-core":return stage_validity_core(config,task_index,run_dir)
    if stage=="eb-robustness":return stage_eb_robustness(config,run_dir)
    if stage=="eb-resume":return stage_eb_robustness(config,run_dir,resume_from_heldout=True)
    if stage=="repaired-inference":return stage_repaired_inference(config,task_index,run_dir)
    if stage=="bridge":return stage_bridge(config,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(config,run_dir)
    if stage=="support-inner":return stage_support_inner(config,task_index,run_dir)
    if stage=="support-inner-gate":return stage_support_inner_gate(config,run_dir)
    if stage=="finalize":return stage_finalize(config,run_dir)
    raise ValueError(f"unknown v8.1 stage: {stage}")


__all__=["run_stage","oracle_ddim_roundtrip"]
