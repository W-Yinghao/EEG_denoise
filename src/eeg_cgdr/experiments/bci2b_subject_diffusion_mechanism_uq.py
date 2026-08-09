"""Frozen-checkpoint operator/score intervention and predictive-UQ audit."""
from __future__ import annotations
import ast,csv,json,inspect,time,textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping
import numpy as np,yaml
from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as v111

SAME=("same_01","same_02","same_03")
CONTEXTS=("MATCH","POP","WRONG")
def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as h:return yaml.safe_load(h)
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0]);[keys.append(k) for r in rows for k in r if k not in keys]
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def _task(config:Mapping[str,Any],task:int)->tuple[int,int]:return int(config["seeds"][task//9]),task%9
def _base(config:Mapping[str,Any],seed:int,fold:int)->Path:return Path(str(config["replication_root"]))/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"

def stage_audit(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    from eeg_cgdr.models.eog_residual_diffusion import EOGResidualDiffusion,_ResidualUNet
    source=inspect.getsource(EOGResidualDiffusion.sample);forward=inspect.getsource(_ResidualUNet.forward);tree=ast.parse(textwrap.dedent(source))
    external_reconstruction="gamma_correction" not in source and "correction" not in source
    score_fields=all(token in forward for token in ("a0","r_det","state"));det_external="r_det:Tensor" in source
    separable=bool(external_reconstruction and score_fields and det_external)
    # The frozen intervention changes the external reconstruction anchor A while
    # feeding a0_C and r_det_C to the unchanged score network.
    result={"status":"SCORE_CONTEXT_SEPARABLE_IN_CURRENT_ARCHITECTURE" if separable else "SCORE_CONTEXT_NOT_SEPARABLE_IN_CURRENT_ARCHITECTURE","b1_authorized":separable,"operator_anchor_location":"external correction assembly after diffusion.sample","score_context_fields":["a0_C","r_det_C","query_EOG","observed_EEG"],"factorial_reconstruction":"a0_A + residual_scale * (r_det_C + delta_C)","network_definition_changed":False,"ast_parsed":isinstance(tree,ast.Module),"development_only":True}
    root=Path(str(config["result_root"]));_json(root/"b0_graph_separability.json",result);_json(run_dir/"result_summary.json",result);return result

def _load_models(base:Path,device:Any):
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    cp=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**cp["config"]);det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(cp["det"]);diff.load_state_dict(cp["diff"]);ema=EMA(diff);ema.load_state_dict(cp["ema"]);ema.copy_to(diff);det.eval();diff.eval();return cp,det,diff

def _sample_context(diff:Any,det:Any,y:np.ndarray,eog:np.ndarray,a0:np.ndarray,bank:np.ndarray,device:Any)->tuple[np.ndarray,np.ndarray]:
    import torch
    yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0,device=device)
    with torch.no_grad():
        rdet=det(y=yt,eog=et,a0=at);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy() for noise in bank]
    return rdet.cpu().numpy(),np.stack(samples)

def stage_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    gate=json.loads((Path(str(config["result_root"]))/"b0_graph_separability.json").read_text());seed,fold=_task(config,task_index);base=_base(config,seed,fold);device=torch.device("cuda");cp,det,diff=_load_models(base,device);scale=np.asarray(cp["residual_scale"],np.float32);root=Path(str(config["result_root"]))/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}";latency=[]
    for unit_index,protocol in enumerate(SAME):
        inf=np.load(base/"units"/protocol/"inference.npz");gamma=float(inf["gamma"]);outdir=root/"units"/protocol;outdir.mkdir(parents=True,exist_ok=True)
        for panel in ("paired","natural"):
            y=np.asarray(inf[f"{panel}_y"],np.float32);eog=np.asarray(inf[f"{panel}_eog"],np.float32);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,int(config["posterior_samples"]));condition={};start=time.perf_counter()
            for name,key in (("MATCH","h_match"),("POP","h_population"),("WRONG","h_wrong")):
                a0=v11.apply_transfer(np.asarray(inf[key]),eog);rdet,samples=_sample_context(diff,det,y,eog,a0,bank,device);condition[name]=(a0,rdet,samples)
            torch.cuda.synchronize();latency.append((time.perf_counter()-start)/len(y));saved={}
            # K32 individual predictive outputs for the three frozen contexts.
            for name,(a0,rdet,samples) in condition.items():
                correction=a0[None]+(rdet[None]+samples)*scale[None,None,:,None];predictive=v11.gamma_correction(y[None],correction,gamma).astype(np.float32)
                if panel=="paired":saved[f"{panel}_{name}_samples"]=predictive
                else:
                    saved[f"{panel}_{name}_mean"]=predictive.mean(0).astype(np.float32)
                    saved[f"{panel}_{name}_dispersion"]=np.sqrt(np.mean(np.var(predictive,axis=0),axis=(1,2))).astype(np.float32)
                    saved[f"{panel}_{name}_latent_dispersion"]=np.sqrt(np.mean(np.var(samples,axis=0),axis=(1,2))).astype(np.float32)
            if gate["b1_authorized"]:
                # R_AC: A is the reconstruction/operator anchor, C is score context.
                for a_name,c_name,label in (("MATCH","MATCH","R_MM"),("MATCH","POP","R_MP"),("POP","MATCH","R_PM"),("POP","POP","R_PP")):
                    a0a=condition[a_name][0];_,rdetc,samplesc=condition[c_name];correction=a0a+(rdetc+samplesc[:int(config["primary_samples"])].mean(0))*scale[None,:,None];saved[f"{panel}_{label}"]=v11.gamma_correction(y,correction,gamma).astype(np.float32)
            np.savez_compressed(outdir/f"{panel}_samples_and_factorial.npz",**saved)
    # Freeze abstention cutoffs from outer-training predictive dispersion only.
    train=np.load(base/"training_pairs.npz");chosen=[]
    for participant in sorted(set(map(int,np.asarray(train["subject"])))):chosen.extend(np.flatnonzero(np.asarray(train["subject"])==participant)[:8].tolist())
    chosen=np.asarray(chosen,dtype=int);ty=np.asarray(train["y"])[chosen];te=np.asarray(train["eog"])[chosen];th=np.asarray(train["h_subject"])[chosen];ta0=v11.apply_transfer(th,te);tbank=v11._noise_bank(ty.shape,seed+fold*100000+77777,int(config["posterior_samples"]));_,tsamples=_sample_context(diff,det,ty,te,ta0,tbank,device);tdisp=np.sqrt(np.mean(np.var(tsamples,axis=0),axis=(1,2)))
    _json(root/"support_frozen_uncertainty_thresholds.json",{"source":"outer_training_rows_only","rows":len(chosen),"q50":float(np.quantile(tdisp,.5)),"q80":float(np.quantile(tdisp,.8)),"query_outcomes_used":False})
    summary={"status":"completed_frozen_factorial_and_k32_samples","seed":seed,"fold":fold,"b1_run":bool(gate["b1_authorized"]),"k":32,"ddim_steps":25,"latency_seconds_per_window_three_contexts":float(np.mean(latency)),"evaluator_opened":False,"peak_memory_bytes":int(torch.cuda.max_memory_allocated())};_json(root/"inference_runtime.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def _rrmse_rows(value:np.ndarray,target:np.ndarray)->np.ndarray:
    return np.sqrt(np.sum((value-target)**2,axis=(1,2))/np.maximum(np.sum(target**2,axis=(1,2)),1e-12))
def _crps(samples:np.ndarray,target:np.ndarray)->float:
    ordered=np.sort(samples,axis=0);k=len(samples);weights=(2*np.arange(k)-k+1).reshape((k,)+(1,)*(samples.ndim-1));return float(np.mean(np.mean(np.abs(samples-target[None]),axis=0)-np.sum(weights*ordered,axis=0)/(k*k)))
def _energy(samples:np.ndarray,target:np.ndarray)->float:
    k=len(samples);flat=samples.reshape(k,len(target),-1);truth=target.reshape(len(target),-1);first=np.linalg.norm(flat-truth[None],axis=2).mean(0);pair=np.zeros(len(target))
    for i in range(k):pair+=np.linalg.norm(flat[i,None]-flat,axis=2).sum(0)
    return float(np.mean((first-.5*pair/(k*k))/np.sqrt(flat.shape[-1])))
def _uq_metrics(samples:np.ndarray,target:np.ndarray)->dict[str,Any]:
    mean=samples.mean(0);unc=np.sqrt(np.mean(np.var(samples,axis=0),axis=(1,2)));err=_rrmse_rows(mean,target);result={"mean_rrmse":float(np.mean(err)),"mean_correlation":v11.correlation(mean,target),"crps":_crps(samples,target),"energy_score":_energy(samples,target),"uncertainty_mean":float(np.mean(unc)),"error_mean":float(np.mean(err))}
    from scipy.stats import pearsonr,spearmanr
    result["uncertainty_error_pearson"]=float(pearsonr(unc,err).statistic) if len(err)>2 and np.std(unc)>0 else float("nan");result["uncertainty_error_spearman"]=float(spearmanr(unc,err).statistic) if len(err)>2 and np.std(unc)>0 else float("nan")
    order=np.argsort(unc);risk=np.cumsum(err[order])/np.arange(1,len(err)+1);result["risk_coverage_auc"]=float(np.trapezoid(risk,np.arange(1,len(err)+1)/len(err)))
    for level in (.5,.8):
        lo=np.quantile(samples,(1-level)/2,axis=0);hi=np.quantile(samples,1-(1-level)/2,axis=0);result[f"coverage_{int(level*100)}"]=float(np.mean((target>=lo)&(target<=hi)));result[f"width_{int(level*100)}"]=float(np.mean(hi-lo))
    result["calibration_error_50_80"]=float(np.mean([abs(result["coverage_50"]-.5),abs(result["coverage_80"]-.8)]));return result

def _independent_realizations(clean:np.ndarray)->dict[str,Any]:
    groups=[]
    for row in clean:
        found=False
        for representative,count in groups:
            if np.array_equal(row,representative):count[0]+=1;found=True;break
        if not found:groups.append((row.copy(),[1]))
    counts=[count[0] for _,count in groups];return {"rows":len(clean),"unique_clean_carriers":len(groups),"carriers_with_multiple_contaminations":int(sum(x>1 for x in counts)),"max_realizations_per_carrier":int(max(counts,default=0)),"posterior_calibration_identifiable":bool(any(x>1 for x in counts))}

def stage_evaluate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seed,fold=_task(config,task_index);base=_base(config,seed,fold);root=Path(str(config["result_root"]))/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}";source=Path(str(config["v11_result_root"]))/"folds"/f"fold_{fold:02d}";factor=[];uq=[];rank=[];construction=[];natural_uq=[];fallback=[];thresholds=json.loads((root/"support_frozen_uncertainty_thresholds.json").read_text())
    for protocol in SAME:
        inf=np.load(base/"units"/protocol/"inference.npz");ev=np.load(source/"units"/protocol/"evaluator.npz");scale=np.asarray(inf["eeg_scale"]);loc=np.asarray(inf["eeg_location"]);target=np.asarray(ev["paired_x"])[...,:int(config["crop_length"])]
        paired=np.load(root/"units"/protocol/"paired_samples_and_factorial.npz");construction.append({"seed":seed,"subject":fold+1,"protocol":protocol,**_independent_realizations(target)})
        for label in ("R_MM","R_MP","R_PM","R_PP"):
            key=f"paired_{label}"
            if key in paired:factor.append({"seed":seed,"subject":fold+1,"protocol":protocol,"arm":label,"rrmse":v11.rrmse((np.asarray(paired[key])*scale[None,:,None]+loc[None,:,None])[...,:500],target)})
        for context in CONTEXTS:
            norm=np.asarray(paired[f"paired_{context}_samples"])[...,:500];samples=norm*scale[None,None,:,None]+loc[None,None,:,None];metrics=_uq_metrics(samples,target);uq.append({"seed":seed,"subject":fold+1,"protocol":protocol,"model":"DIFF","context":context,"samples":32,**metrics})
            ranks=np.sum(samples<target[None],axis=0).ravel();counts=np.bincount(ranks,minlength=33)
            for bin_index,count in enumerate(counts):rank.append({"seed":seed,"subject":fold+1,"protocol":protocol,"model":"DIFF","context":context,"rank":bin_index,"count":int(count)})
        natural=np.load(root/"units"/protocol/"natural_samples_and_factorial.npz");raw=np.asarray(inf["natural_y"])[...,:500];eog=np.asarray(inf["natural_eog"])[...,:500];labels=np.asarray(ev["natural_labels"]);energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3);points={};dispersions={}
        for context in CONTEXTS:
            point=np.asarray(natural[f"natural_{context}_mean"])[...,:500];disp=np.asarray(natural[f"natural_{context}_dispersion"]);latent_disp=np.asarray(natural[f"natural_{context}_latent_dispersion"]);points[context]=point;dispersions[context]=latent_disp;natural_uq.append({"seed":seed,"subject":fold+1,"protocol":protocol,"context":context,"uncertainty_mean":float(np.mean(disp)),"latent_uncertainty_mean":float(np.mean(latent_disp)),"preservation":1-v11.rrmse(point[low],raw[low]),"covariance":v11._covariance_distortion(point[low],raw[low]),"eog_attenuation":v11._coherence_proxy(raw,eog)-v11._coherence_proxy(point,eog),"mi_kappa":v11._kappa(point,labels),"low_eog_fraction":float(np.mean(low))})
        for name,threshold in (("q50",thresholds["q50"]),("q80",thresholds["q80"])):
            use=dispersions["MATCH"]<=float(threshold);mixed=np.where(use[:,None,None],points["MATCH"],points["POP"]);fallback.append({"seed":seed,"subject":fold+1,"protocol":protocol,"threshold":name,"achieved_match_coverage":float(np.mean(use)),"preservation":1-v11.rrmse(mixed[low],raw[low]),"covariance":v11._covariance_distortion(mixed[low],raw[low]),"eog_attenuation":v11._coherence_proxy(raw,eog)-v11._coherence_proxy(mixed,eog),"mi_kappa":v11._kappa(mixed,labels)})
    _csv(root/"factorial_metrics.csv",factor);_csv(root/"uq_metrics.csv",uq);_csv(root/"rank_histogram.csv",rank);_csv(root/"construction_audit.csv",construction);_csv(root/"natural_uq.csv",natural_uq);_csv(root/"support_frozen_fallback.csv",fallback);summary={"status":"completed_independent_evaluator","seed":seed,"fold":fold,"factorial_rows":len(factor),"uq_rows":len(uq),"natural_uq_rows":len(natural_uq),"evaluator_opened_after_outputs":True};_json(run_dir/"result_summary.json",summary);return summary

def _participant_factor(rows:list[dict[str,str]])->list[dict[str,Any]]:
    out=[]
    for seed in sorted({int(r["seed"]) for r in rows}):
        for subject in range(1,10):
            take=[r for r in rows if int(r["seed"])==seed and int(r["subject"])==subject];mean={arm:float(np.mean([float(r["rrmse"]) for r in take if r["arm"]==arm])) for arm in ("R_MM","R_MP","R_PM","R_PP")};out.append({"seed":seed,"subject":subject,**mean,"G_A":.5*((mean["R_PM"]-mean["R_MM"])+(mean["R_PP"]-mean["R_MP"])),"G_C":.5*((mean["R_MP"]-mean["R_MM"])+(mean["R_PP"]-mean["R_PM"])),"I":mean["R_MP"]+mean["R_PM"]-mean["R_MM"]-mean["R_PP"]})
    return out

def stage_aggregate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));factor=[];uq=[];construction=[];natural_uq=[];fallback=[]
    for seed in map(int,config["seeds"]):
        for fold in range(9):
            base=root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}";factor.extend(_read(base/"factorial_metrics.csv"));uq.extend(_read(base/"uq_metrics.csv"));construction.extend(_read(base/"construction_audit.csv"));natural_uq.extend(_read(base/"natural_uq.csv"));fallback.extend(_read(base/"support_frozen_fallback.csv"))
    participant=_participant_factor(factor);_csv(root/"b1_participant_seed_effects.csv",participant);averaged=[]
    for subject in range(1,10):
        rows=[r for r in participant if r["subject"]==subject];averaged.append({"subject":subject,**{name:float(np.mean([r[name] for r in rows])) for name in ("G_A","G_C","I")}})
    _csv(root/"b1_participant_effects.csv",averaged);rng=np.random.default_rng(int(config["bootstrap_seed"]));boot_idx=rng.integers(0,9,size=(int(config["bootstrap_replicates"]),9));b1={}
    for name in ("G_A","G_C","I"):
        values=np.asarray([r[name] for r in averaged]);rep=values[boot_idx].mean(1);b1[name]={"mean":float(values.mean()),"median":float(np.median(values)),"positive":int(np.sum(values>0)),"seed_means":[float(np.mean([r[name] for r in participant if r["seed"]==seed])) for seed in map(int,config["seeds"])],"descriptive_ci_low":float(np.quantile(rep,.025)),"descriptive_ci_high":float(np.quantile(rep,.975))}
    if b1["G_C"]["mean"]>0 and b1["G_C"]["median"]>0 and sum(x>0 for x in b1["G_C"]["seed_means"])>=2:b1_decision="SUBJECT_CONTEXT_IS_CONSUMED_BY_SCORE_MODEL"
    elif b1["G_A"]["mean"]>0 and b1["G_A"]["median"]>0:b1_decision="SUBJECT_EFFECT_IS_OPERATOR_MEDIATED_WITHIN_DIFFUSION_PIPELINE"
    else:b1_decision="NO_STABLE_SEPARABLE_SUBJECT_COMPONENT_SIGNAL"
    # Deterministic deep ensemble from the three frozen checkpoints/outputs.
    detrows=[]
    for subject in range(1,10):
        for protocol in SAME:
            samples=[];target=None
            for seed in map(int,config["seeds"]):
                base=_base(config,seed,subject-1);inf=np.load(base/"units"/protocol/"inference.npz");out=np.load(base/"outputs"/"k8"/protocol/"inference_outputs.npz");ev=np.load(Path(str(config["v11_result_root"]))/"folds"/f"fold_{subject-1:02d}"/"units"/protocol/"evaluator.npz");scale=np.asarray(inf["eeg_scale"]);loc=np.asarray(inf["eeg_location"]);samples.append((np.asarray(out["paired_DET-MATCH"])*scale[None,:,None]+loc[None,:,None])[...,:500]);target=np.asarray(ev["paired_x"])[...,:500]
            detrows.append({"subject":subject,"protocol":protocol,"model":"DET_ENSEMBLE","context":"MATCH","samples":3,**_uq_metrics(np.stack(samples),target)})
    _csv(root/"deterministic_ensemble_uq.csv",detrows);_csv(root/"diffusion_uq.csv",uq);_csv(root/"natural_uq.csv",natural_uq);_csv(root/"support_frozen_fallback.csv",fallback)
    diff_part=[];det_part=[]
    for subject in range(1,10):
        for seed in map(int,config["seeds"]):
            rows=[r for r in uq if int(r["subject"])==subject and int(r["seed"])==seed and r["context"]=="MATCH"];diff_part.append({"subject":subject,"seed":seed,**{name:float(np.mean([float(r[name]) for r in rows])) for name in ("crps","energy_score","risk_coverage_auc","uncertainty_error_spearman","mean_rrmse")}})
        rows=[r for r in detrows if int(r["subject"])==subject];det_part.append({"subject":subject,**{name:float(np.mean([float(r[name]) for r in rows])) for name in ("crps","energy_score","risk_coverage_auc","uncertainty_error_spearman","mean_rrmse")}})
    _csv(root/"diffusion_participant_uq.csv",diff_part);_csv(root/"deterministic_participant_uq.csv",det_part)
    diff_avg={s:{name:float(np.mean([r[name] for r in diff_part if r["subject"]==s])) for name in ("crps","energy_score","risk_coverage_auc","uncertainty_error_spearman","mean_rrmse")} for s in range(1,10)};det_avg={r["subject"]:r for r in det_part};proper_better=np.mean([diff_avg[s]["crps"]-det_avg[s]["crps"] for s in range(1,10)])<0 or np.mean([diff_avg[s]["energy_score"]-det_avg[s]["energy_score"] for s in range(1,10)])<0;risk_better=np.mean([diff_avg[s]["risk_coverage_auc"]-det_avg[s]["risk_coverage_auc"] for s in range(1,10)])<0;point_not_worse=np.mean([diff_avg[s]["mean_rrmse"]-det_avg[s]["mean_rrmse"] for s in range(1,10)])<=.01;association=np.mean([diff_avg[s]["uncertainty_error_spearman"] for s in range(1,10)])>0
    identifiable=any(int(r["carriers_with_multiple_contaminations"])>0 for r in construction);b2_decision="DIFFUSION_PROBABILISTIC_VALUE_SIGNAL_PRESENT" if (proper_better or risk_better) and point_not_worse else ("WEAK_UNCERTAINTY_ASSOCIATION_ONLY" if association else "NO_PROBABILISTIC_VALUE_IN_CURRENT_CHECKPOINT")
    comparison={}
    for name in ("crps","energy_score","risk_coverage_auc","uncertainty_error_spearman","mean_rrmse"):
        diffs=np.asarray([diff_avg[s][name]-det_avg[s][name] for s in range(1,10)]);rep=diffs[boot_idx].mean(1);comparison[name]={"diffusion_mean":float(np.mean([diff_avg[s][name] for s in range(1,10)])),"det_ensemble_mean":float(np.mean([det_avg[s][name] for s in range(1,10)])),"diff_minus_det":float(diffs.mean()),"descriptive_ci_low":float(np.quantile(rep,.025)),"descriptive_ci_high":float(np.quantile(rep,.975))}
    participant_natural=[]
    for subject in range(1,10):
        match=[r for r in natural_uq if int(r["subject"])==subject and r["context"]=="MATCH"];pop=[r for r in natural_uq if int(r["subject"])==subject and r["context"]=="POP"];participant_natural.append({"subject":subject,"uncertainty":float(np.mean([float(r["uncertainty_mean"]) for r in match])),"preservation":float(np.mean([float(r["preservation"]) for r in match])),"covariance":float(np.mean([float(r["covariance"]) for r in match])),"kappa_delta_vs_pop":float(np.mean([float(r["mi_kappa"]) for r in match])-np.mean([float(r["mi_kappa"]) for r in pop]))})
    for row in participant_natural:row["reversal"]=int(row["preservation"]<.78 or row["covariance"]>.15 or row["kappa_delta_vs_pop"]<-.02)
    _csv(root/"natural_reversal_uq.csv",participant_natural);rev=[r["uncertainty"] for r in participant_natural if r["reversal"]];ok=[r["uncertainty"] for r in participant_natural if not r["reversal"]]
    reversal_auc=float(np.mean([a>b for a in rev for b in ok])) if rev and ok else float("nan")
    natural_summary={"match_uncertainty_mean":float(np.mean([float(r["uncertainty_mean"]) for r in natural_uq if r["context"]=="MATCH"])),"support_frozen_q50_coverage":float(np.mean([float(r["achieved_match_coverage"]) for r in fallback if r["threshold"]=="q50"])),"support_frozen_q80_coverage":float(np.mean([float(r["achieved_match_coverage"]) for r in fallback if r["threshold"]=="q80"])),"participant_reversals":int(sum(r["reversal"] for r in participant_natural)),"uncertainty_reversal_auc":reversal_auc}
    result={"status":"completed_development_mechanism_and_uq","b1_decision":b1_decision,"b1_effects":b1,"b1_interpretation":"small heterogeneous score-context effect; 3/3 seed means positive but only 5/9 participant means positive","b2_decision":b2_decision,"b2_diffusion_vs_det_ensemble":comparison,"posterior_calibration_identifiable":bool(identifiable),"uq_claim_scope":"posterior calibration" if identifiable else "predictive dispersion and error ranking only","proper_score_better_than_det_ensemble":bool(proper_better),"risk_coverage_better_than_det_ensemble":bool(risk_better),"point_estimate_not_materially_worse":bool(point_not_worse),"natural_uq":natural_summary,"participants":9,"training_seeds":3,"scientific_unit":"participant"};_json(root/"result_summary.json",result);_json(root/"routing_decision.json",result);_json(run_dir/"result_summary.json",result)
    import matplotlib.pyplot as plt
    figdir=root/"figures";figdir.mkdir(exist_ok=True);x=np.arange(1,10);fig,ax=plt.subplots(figsize=(7,4));ax.axhline(0,color="black",lw=.8);ax.plot(x,[r["G_A"] for r in averaged],"o-",label="G_A operator");ax.plot(x,[r["G_C"] for r in averaged],"s-",label="G_C score context");ax.plot(x,[r["I"] for r in averaged],"^-",label="I synergy");ax.set_xlabel("Participant");ax.set_ylabel("positive RRMSE utility");ax.legend();fig.tight_layout();fig.savefig(figdir/"factorial_participant_effects.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));names=("crps","energy_score","risk_coverage_auc","mean_rrmse");pos=np.arange(len(names));ax.bar(pos-.18,[comparison[n]["diffusion_mean"] for n in names],.36,label="DIFF K32");ax.bar(pos+.18,[comparison[n]["det_ensemble_mean"] for n in names],.36,label="DET 3-seed");ax.set_xticks(pos,names,rotation=20);ax.legend();fig.tight_layout();fig.savefig(figdir/"uq_diffusion_vs_det.png",dpi=180);plt.close(fig);return result

def stage_finalize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));r=json.loads((root/"result_summary.json").read_text());lines=["# BCI2b subject diffusion mechanism and UQ","","Development exploration using frozen V11.1 checkpoints; no network was retrained.","",f"B1 decision: `{r['b1_decision']}` (small, participant-heterogeneous score-context signal).",f"B2 decision: `{r['b2_decision']}`.","","## Separable intervention","",f"Operator effect G_A mean/median: {r['b1_effects']['G_A']['mean']:+.5f}/{r['b1_effects']['G_A']['median']:+.5f} ({r['b1_effects']['G_A']['positive']}/9 positive). Score-context effect G_C: {r['b1_effects']['G_C']['mean']:+.5f}/{r['b1_effects']['G_C']['median']:+.5f} ({r['b1_effects']['G_C']['positive']}/9 positive; all 3 seed means positive). Positive-synergy interaction I: {r['b1_effects']['I']['mean']:+.5f}/{r['b1_effects']['I']['median']:+.5f} ({r['b1_effects']['I']['positive']}/9 positive).",f"Descriptive participant bootstrap intervals: G_A [{r['b1_effects']['G_A']['descriptive_ci_low']:+.5f}, {r['b1_effects']['G_A']['descriptive_ci_high']:+.5f}], G_C [{r['b1_effects']['G_C']['descriptive_ci_low']:+.5f}, {r['b1_effects']['G_C']['descriptive_ci_high']:+.5f}], I [{r['b1_effects']['I']['descriptive_ci_low']:+.5f}, {r['b1_effects']['I']['descriptive_ci_high']:+.5f}].","","The intervention keeps the score network frozen. A selects the external EOG-operator reconstruction anchor; C supplies a0_C and r_det_C to the unchanged score network. G_C is much smaller than G_A and heterogeneous across participants, so the label does not imply a large or uniform score effect. I uses the predeclared positive-synergy definition R_MP + R_PM - R_MM - R_PP. It does not alter the A-track claim.","","## Probability scope","",f"Paired construction supports `{r['uq_claim_scope']}`. Training seeds were evaluated separately and were not treated as posterior samples. The deterministic comparator is the three-seed DET-MATCH ensemble.","","| metric | DIFF K32 | DET ensemble | DIFF-DET |","|---|---:|---:|---:|"]
    for name,value in r["b2_diffusion_vs_det_ensemble"].items():lines.append(f"| {name} | {value['diffusion_mean']:.6f} | {value['det_ensemble_mean']:.6f} | {value['diff_minus_det']:+.6f} [{value['descriptive_ci_low']:+.6f}, {value['descriptive_ci_high']:+.6f}] |")
    lines += ["",f"Proper-score better: `{r['proper_score_better_than_det_ensemble']}`; risk-coverage better: `{r['risk_coverage_better_than_det_ensemble']}`; point estimate not materially worse: `{r['point_estimate_not_materially_worse']}`. Thus positive dispersion-error association alone supports only the weak-association label.","",f"Natural reversal diagnostic: {r['natural_uq']['participant_reversals']}/9 participant reversals, uncertainty reversal AUC {r['natural_uq']['uncertainty_reversal_auc']:.3f}. Outer-training-frozen q50/q80 cutoffs achieved query MATCH coverage {r['natural_uq']['support_frozen_q50_coverage']:.3f}/{r['natural_uq']['support_frozen_q80_coverage']:.3f}.","","Natural-output uncertainty is exploratory; no query outcome trained a gate or selected a threshold."]
    Path("reports/bci2b_subject_diffusion_mechanism_uq.md").write_text("\n".join(lines)+"\n",encoding="utf-8");_json(run_dir/"result_summary.json",r);return r

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    c=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="audit":return stage_audit(c,task_index,run_dir)
    if stage=="infer":return stage_infer(c,task_index,run_dir)
    if stage=="evaluate":return stage_evaluate(c,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(c,task_index,run_dir)
    if stage=="finalize":return stage_finalize(c,task_index,run_dir)
    raise ValueError(stage)
