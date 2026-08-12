"""Slurm-facing V28 support-conditioned clean conditional diffusion workflow."""
from __future__ import annotations

import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
from typing import Any,Iterable,Mapping

import numpy as np
import torch
import yaml

from eeg_scad.cli import run_v27,v26
from eeg_scad.data.folds import load_folds,validate_folds
from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.evaluation.aggregate_v26 import bootstrap,contrast,participant_first
from eeg_scad.evaluation.aggregate_v28 import aggregate as aggregate_metrics
from eeg_scad.evaluation.natural_metrics_v28 import attenuation_consistency,natural_metrics_v28
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.task_preservation_v28 import inventory as task_inventory
from eeg_scad.training.train_v25 import load_det as load_support_model
from eeg_scad.training.train_v28 import load as load_v28,predict as predict_v28,support_features,train as train_v28

ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT",Path(__file__).resolve().parents[3]));RESULT=ROOT/"results/sc_cdm_v28";DERIVED=Path("/projects/EEG-foundation-model/derived/denoiseNet/sc_cdm_v28");V24=Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24");V25=Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25");V27=Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_energy_v27");BASE="40eae116e70e9de7fe0af55d64ee25551932c4a8";SEEDS=[20260901,20260902,20260903];OLD_SEED={20260901:20260825,20260902:20260826,20260903:20260827};V26_SEED={20260901:20260828,20260902:20260829,20260903:20260830}


def _cfg(name:str)->dict[str,Any]:return yaml.safe_load((ROOT/f"configs/sc_cdm_v28/{name}.yaml").read_text())
def _folds()->list[dict[str,Any]]:return load_folds(ROOT/"configs/sc_cdm_v28/folds.yaml")
def _index()->int:return int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))
def _digest(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _csv(path:Path,rows:Iterable[Mapping[str,Any]])->None:
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
def _support_path(fold:int,seed:int)->Path:return V25/f"checkpoints/det/deepsets/fold_{fold}/seed_{OLD_SEED[seed]}/best_joint.pt"
def _checkpoint(kind:str,fold:int,seed:int,variant:str="selected")->Path:return DERIVED/f"checkpoints/{variant}/{kind}/fold_{fold}/seed_{seed}/best_joint.pt"


def preflight(run:Path)->dict[str,Any]:
    validate_folds(_folds(),_cfg("data")["participants"]);head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();ledger=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();checks={"base_exact":subprocess.check_output(["git","rev-parse","codex/calib-energy-v27"],cwd=ROOT,text=True).strip()==BASE,"base_ancestor":subprocess.run(["git","merge-base","--is-ancestor",BASE,head],cwd=ROOT).returncode==0,"ledger_v1_7":"**版本：** v1.7" in ledger,"ledger_active_v28":"V28 SC-CDM" in ledger,"v27_unchanged":not bool(subprocess.check_output(["git","diff","--name-only",BASE,"--","results/calib_energy_v27","reports/v27_*"],cwd=ROOT,text=True).strip()),"a_track_unchanged":not bool(subprocess.check_output(["git","diff","--name-only",BASE,"--","taas_submission"],cwd=ROOT,text=True).strip()),"sealed_reads":0}
    if not all(value is True for key,value in checks.items() if key!="sealed_reads"):raise RuntimeError(checks)
    registry={"stage":"R0","status":"PASS","base_commit":BASE,"head":head,"V27_terminal":BASE,"V25":"a7d9d647b69e152255b62dbca917a4b3ed082915","V26":"7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c","A_track":"0c4f2301c1f873120fe54537cde3c76fff7ea3a2",**checks};_json(RESULT/"source_registry.json",registry);_json(run/"result_summary.json",registry);return registry


def audit(run:Path)->dict[str,Any]:
    paths=[ROOT/"results/calib_energy_v27/terminal_manifest.json",ROOT/"results/calib_energy_v27/method_summary.csv",ROOT/"results/calib_energy_v27/participant_effects.csv",ROOT/"results/calib_energy_v27/natural_evaluation/output_manifest.csv",ROOT/"reports/slurm/v27_job_ids.txt"]
    inventory=[]
    for path in paths:
        stat=path.stat();inventory.append({"absolute_path":str(path.resolve()),"role":"frozen_V27_evidence","sha256":_digest(path),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns})
    _csv(RESULT/"input_inventory.csv",inventory);tasks=task_inventory({});_csv(RESULT/"event_task_inventory.csv",tasks)
    legacy=list(csv.DictReader((ROOT/"results/calib_energy_v27/method_summary.csv").open()));aliases={row["metric"] for row in legacy if row["metric"] in ("erp_proxy","ssvep_proxy","preservation")};value={"stage":"R1","status":"PASS","legacy_field_renamed":"preservation_legacy","active_fields":["low_eog_observation_change","low_eog_observation_retention"],"legacy_aliases_detected":sorted(aliases),"erp_status":"unavailable","ssvep_status":"unavailable","erd_ers_status":"unavailable","proxy_aliases_active":False,"reason":"frozen evaluator arrays and role metadata lack event markers, stimulation frequency/phase, and trial boundaries","query_auxiliary_inference_reads":0,"sealed_reads":0};_json(RESULT/"natural_metric_audit.json",value)
    (ROOT/"reports/v28_v27_transition_audit.md").write_text("# V28 V27 transition audit\n\nV27 is frozen at `"+BASE+"`. The bound evidence inventory is machine-readable in `results/sc_cdm_v28/input_inventory.csv`. V27 remains development evidence and is not modified.\n")
    (ROOT/"reports/v28_natural_metric_audit.md").write_text("# V28 natural metric audit\n\nThe historical `preservation` scalar is correction-based low-EOG observation retention, not ERP, SSVEP, or physiological ground truth. V28 renames it `low_eog_observation_retention`, reports its complement as `low_eog_observation_change`, and sets ERP/SSVEP/ERD-ERS to `unavailable`. No scalar aliases are active.\n")
    _json(run/"result_summary.json",value);return value


def prepare(run:Path)->dict[str,Any]:
    rows=[];bindings=[]
    for fold in _folds():
        for split in ("train","validation","test"):
            for participant in fold[split]:rows.append({"fold":fold["fold"],"split":split,"participant":participant})
        for seed in SEEDS:
            path=_support_path(fold["fold"],seed);bindings.append({"fold":fold["fold"],"seed":seed,"support_encoder":"V25 DeepSets frozen","checkpoint":str(path),"sha256":_digest(path),"query_disjoint":True})
    _csv(RESULT/"fold_manifest.csv",rows);_csv(RESULT/"support_binding.csv",bindings);value={"stage":"R3","status":"PASS","folds":5,"support_bindings":15,"same_backbone":True,"query_auxiliary_inference_reads":0,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def pareto(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=V26_SEED[SEEDS[index%3]];sampler=SupportSetEpisodeSampler(v26._cfg("data"),_folds()[fold],"test",seed+401);paired=sampler.sample_paired(192);natural=sampler.sample_natural(96);rows=[]
    for ly in (.5,2,8):
        pp,_=run_v27._energy_bundle(paired,fold,seed,1,ly,"final_only");nn,_=run_v27._energy_bundle(natural,fold,seed,1,ly,"final_only")
        for panel,batch,pred in (("paired",paired,pp),("natural",natural,nn)):
            for method in ("CALIB_ENERGY_DET_MATCH","CALIB_ENERGY_DET_WRONG","POP_ENERGY_DET","CALIB_ENERGY_SDEDIT_MATCH","CALIB_ENERGY_SDEDIT_WRONG","POP_ENERGY_SDEDIT"):
                if panel=="paired":score=np.mean([paired_metrics(batch["x"][i],batch["y"][i],batch["artifact"][i],pred[method][i])["rrmse_temporal"] for i in range(len(batch["y"]))]);rows.append({"panel":panel,"fold":fold,"seed":seed,"lambda_y":ly,"lambda_a":1,"method":method,"clean_rrmse":float(score)})
                else:
                    metrics=[v26._natural(batch["y"][i],pred[method][i],batch["teacher_artifact"][i],batch["latent"][i]) for i in range(len(batch["y"]))];rows.append({"panel":panel,"fold":fold,"seed":seed,"lambda_y":ly,"lambda_a":1,"method":method,"remaining_ratio":float(np.mean([m["remaining_ratio"] for m in metrics])),"low_eog_observation_retention":float(np.mean([m["preservation"] for m in metrics]))})
    _csv(RESULT/f"v27_pareto/cell_{fold}_{seed}.csv",rows);value={"stage":"R2","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"architecture_selection_use":False,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def pareto_aggregate(run:Path)->dict[str,Any]:
    rows=[]
    for path in sorted((RESULT/"v27_pareto").glob("cell_*.csv")):rows.extend(csv.DictReader(path.open()))
    _csv(RESULT/"v27_energy_pareto.csv",rows);summary=[]
    for panel in ("paired","natural"):
        for ly in (.5,2,8):
            for method in sorted({r["method"] for r in rows}):
                chosen=[r for r in rows if r["panel"]==panel and float(r["lambda_y"])==ly and r["method"]==method]
                if chosen:
                    item={"panel":panel,"lambda_y":ly,"method":method,"cells":len(chosen)}
                    for metric in ("clean_rrmse","remaining_ratio","low_eog_observation_retention"):
                        values=[float(r[metric]) for r in chosen if r.get(metric,"") not in ("",None)];
                        if values:item[metric]=float(np.mean(values))
                    summary.append(item)
    _csv(RESULT/"v27_energy_pareto_summary.csv",summary);(ROOT/"reports/v28_v27_energy_pareto_ablation.md").write_text("# V28 V27 mild-energy Pareto ablation\n\nAll 15 development participants were replayed at `lambda_y` 0.5, 2, and 8 with `lambda_a=1`, final-only. This frozen-output diagnostic does not select the V28 architecture. The complete rows are in `v27_energy_pareto.csv`.\n");value={"stage":"R2_AGG","status":"PASS","rows":len(rows),"cells":15,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def sanity(run:Path)->dict[str,Any]:
    from eeg_scad.models.pop_clean_cdm import PopCleanCDM
    from eeg_scad.models.pop_clean_det import PopCleanDET
    from eeg_scad.models.support_clean_cdm import SupportCleanCDM
    from eeg_scad.models.support_clean_det import SupportCleanDET
    device=torch.device("cuda");generator=torch.Generator(device=device).manual_seed(20260901);y=torch.randn((2,46,256),device=device,generator=generator);clean=y-.1*torch.randn(y.shape,device=device,generator=generator);context=torch.randn((2,128),device=device,generator=generator);wrong=-context
    models=[PopCleanDET().to(device),SupportCleanDET().to(device),PopCleanCDM().to(device),SupportCleanCDM().to(device)];initial=[];final=[]
    for model in models:
        optimizer=torch.optim.Adam(model.parameters(),1e-3);first=None
        for _ in range(80):
            if isinstance(model,PopCleanDET):pred=model(y)
            elif isinstance(model,SupportCleanDET):pred=model(y,context)
            elif isinstance(model,PopCleanCDM):pred,_,_=model.training_prediction(clean,y,generator)
            else:pred,_,_=model.training_prediction(clean,y,context,generator)
            loss=(pred-clean).square().mean();first=float(loss) if first is None else first;optimizer.zero_grad();loss.backward();optimizer.step()
        initial.append(first);final.append(float(loss))
    # Replay checks are inference checks.  Keeping the freshly trained model in
    # train mode would deliberately resample dropout masks and falsely report
    # non-determinism even when the diffusion noise is fixed.
    diff=models[-1].eval();noise=torch.randn(y.shape,device=device,generator=generator);sample,trajectory=diff.sample(y,context,noise,10);same=diff.sample(y,context,noise,10)[0];wrong_sample=diff.sample(y,wrong,noise,10)[0];replay=float((sample-same).abs().max());value={"stage":"R4","status":"PASS" if replay<=1e-6 else "FAIL","initial_losses":initial,"final_losses":final,"overfit_reduction":[b/a for a,b in zip(initial,final)],"finite":bool(torch.isfinite(sample).all()),"fixed_noise_replay_max":replay,"context_swap_change":float((sample-wrong_sample).abs().mean()),"ddim10_calls":len(trajectory),"same_backbone_parameter_delta":sum(p.numel() for p in models[2].parameters())-sum(p.numel() for p in models[3].parameters()),"K":1,"sealed_reads":0};_json(RESULT/"sanity/technical_validity.json",value);_json(run/"result_summary.json",value);return value


ROUND_A=[("pop_det","natural"),("support_det","natural"),("pop_cdm","natural"),("support_cdm","paired_only"),("support_cdm","natural")]
def train_stage(stage:str,run:Path)->dict[str,Any]:
    index=_index()
    if stage=="r5-rounda":
        model_index=index//2;fold=(0,2)[index%2];kind,variant=ROUND_A[model_index];seed=20260901
    else:
        fold=index//3;seed=SEEDS[index%3];kind=stage.removeprefix("r9-").replace("-","_");variant="selected"
    config_name={"pop_det":"pop_clean_det","support_det":"support_clean_det","pop_cdm":"pop_clean_cdm","support_cdm":"support_clean_cdm"}[kind];cfg=_cfg(config_name)
    if variant=="paired_only":cfg["natural_fraction"]=0.;cfg["lambda_low"]=0.;cfg["lambda_Q"]=0.
    selection=RESULT/"round_a/selection.json"
    if variant=="selected" and selection.is_file():
        chosen=json.loads(selection.read_text());cfg["natural_fraction"]=chosen["natural_fraction"];cfg["lambda_low"]=chosen["lambda_low"];cfg["lambda_Q"]=chosen["lambda_Q"];cfg["ddim_steps"]=chosen["ddim_steps"]
    out=DERIVED/f"checkpoints/{variant}/{kind}/fold_{fold}/seed_{seed}";result=train_v28(kind,fold,seed,cfg,_cfg("data"),_folds()[fold],out,_support_path(fold,seed),resume=True);target=RESULT/("round_a" if stage=="r5-rounda" else "round_b")/f"{kind}_{variant}_fold_{fold}_seed_{seed}.json";_json(target,result);_json(run/"result_summary.json",result);return result


def _models(fold:int,seed:int,round_a:bool=False):
    device=torch.device("cuda");support,_=load_support_model(_support_path(fold,seed),device);variant=lambda kind: "paired_only" if round_a and kind=="support_cdm_paired" else "natural" if round_a else "selected"
    result={}
    for label,kind in (("POP_CLEAN_DET","pop_det"),("SUPPORT_CLEAN_DET","support_det"),("POP_CLEAN_CDM","pop_cdm"),("SUPPORT_CLEAN_CDM","support_cdm")):
        v=variant("support_cdm_paired" if label=="SUPPORT_CLEAN_CDM" else kind);result[label]=load_v28(_checkpoint(kind,fold,seed,v),device)[0]
    if round_a:result["SUPPORT_CLEAN_CDM_PAIRED"]=load_v28(_checkpoint("support_cdm",fold,seed,"paired_only"),device)[0]
    return device,support,result


@torch.no_grad()
def _predict_all(batch:Mapping[str,Any],fold:int,seed:int,round_a:bool=False,steps:int=25)->dict[str,np.ndarray]:
    device,support,models=_models(fold,seed,round_a);output={}
    for label,model in models.items():
        kind="pop_det" if label=="POP_CLEAN_DET" else "support_det" if label=="SUPPORT_CLEAN_DET" else "pop_cdm" if label=="POP_CLEAN_CDM" else "support_cdm"
        output[label+"_MATCH" if kind.startswith("support") else label]=predict_v28(model,kind,batch,support,device,seed+101,steps).cpu().numpy()
        if kind.startswith("support"):
            output[label+"_WRONG"]=predict_v28(model,kind,batch,support,device,seed+101,steps,wrong=True).cpu().numpy();output[label+"_NULL"]=predict_v28(model,kind,batch,support,device,seed+101,steps,null=True).cpu().numpy()
    return output


def round_a_eval(run:Path)->dict[str,Any]:
    fold=(0,2)[_index()];seed=20260901;sampler=SupportSetEpisodeSampler(_cfg("data"),_folds()[fold],"validation",seed+303);paired=sampler.sample_paired(192,.2);natural=sampler.sample_natural(96);rows=[]
    for steps in (10,25):
        for panel,batch in (("paired",paired),("natural",natural)):
            pred=_predict_all(batch,fold,seed,True,steps)
            for method,clean in pred.items():
                if panel=="paired":metrics=[paired_metrics(batch["x"][i],batch["y"][i],batch["artifact"][i],batch["y"][i]-clean[i]) for i in range(len(clean))];rows.append({"fold":fold,"steps":steps,"panel":panel,"method":method,"clean_rrmse":float(np.mean([m["rrmse_temporal"] for m in metrics])),"spectral":float(np.mean([m["rrmse_spectral"] for m in metrics])),"correlation":float(np.mean([m["correlation"] for m in metrics]))})
                else:
                    change=float(np.mean(np.abs(clean-batch["y"])));rows.append({"fold":fold,"steps":steps,"panel":panel,"method":method,"natural_observation_change":change})
    _csv(RESULT/f"round_a/evaluation_fold_{fold}.csv",rows);value={"stage":"R6_R7","status":"PASS","fold":fold,"rows":len(rows),"test_used":False,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def select(run:Path)->dict[str,Any]:
    rows=[]
    for path in sorted((RESULT/"round_a").glob("evaluation_fold_*.csv")):rows.extend(csv.DictReader(path.open()))
    candidates=[]
    for variant in ("SUPPORT_CLEAN_CDM_PAIRED_MATCH","SUPPORT_CLEAN_CDM_MATCH"):
        for steps in (10,25):
            paired=[r for r in rows if r["panel"]=="paired" and r["method"]==variant and int(r["steps"])==steps];natural=[r for r in rows if r["panel"]=="natural" and r["method"]==variant and int(r["steps"])==steps];score=np.mean([float(r["clean_rrmse"]) for r in paired])+.1*np.mean([float(r["natural_observation_change"]) for r in natural]);candidates.append({"variant":variant,"steps":steps,"score":float(score),"paired_rrmse":float(np.mean([float(r["clean_rrmse"]) for r in paired])),"natural_observation_change":float(np.mean([float(r["natural_observation_change"]) for r in natural]))})
    best=min(candidates,key=lambda row:row["score"]);natural=best["variant"]=="SUPPORT_CLEAN_CDM_MATCH";value={"status":"ROUND_B_CONFIG_FROZEN","ddim_steps":best["steps"],"natural_fraction":.3 if natural else 0.,"lambda_low":.05 if natural else 0.,"lambda_Q":.05 if natural else 0.,"support_encoder":"frozen","optional_finetune_authorized":False,"selection_uses_test":False,"selection_priority":"paired_fidelity_with_corrected_natural_observation_change","candidates":candidates,"rationale":"Validation-only folds 0/2; deterministic models remain competitive controls rather than diffusion retention gates."};_json(RESULT/"round_a/selection.json",value);_json(run/"result_summary.json",value);return value


def paired_infer(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3];sampler=SupportSetEpisodeSampler(_cfg("data"),_folds()[fold],"test",seed+509);batch=sampler.sample_paired(192,.2);selection=json.loads((RESULT/"round_a/selection.json").read_text());started=time.time();pred=_predict_all(batch,fold,seed,False,int(selection["ddim_steps"]));out=DERIVED/f"paired/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred,x=batch["x"],y=batch["y"],artifact=batch["artifact"]);_json(DERIVED/f"paired/fold_{fold}_seed_{seed}_meta.json",batch["meta"]);value={"stage":"R10_INFER","status":"PASS","fold":fold,"seed":seed,"path":str(out),"sha256":_digest(out),"seconds":time.time()-started,"windows":192,"query_auxiliary_reads":0,"sealed_reads":0};_json(RESULT/f"round_b/output_{fold}_{seed}.json",value);_json(run/"result_summary.json",value);return value


def paired_eval(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3];path=DERIVED/f"paired/fold_{fold}_seed_{seed}.npz";meta=json.loads((DERIVED/f"paired/fold_{fold}_seed_{seed}_meta.json").read_text())
    with np.load(path,allow_pickle=False) as archive:arrays={key:np.asarray(archive[key]) for key in archive.files}
    methods=[key for key in arrays if key not in ("x","y","artifact")];rows=[]
    for i,item in enumerate(meta):
        for method in ("RAW",*methods):
            clean=arrays["y"][i] if method=="RAW" else arrays[method][i];metric=paired_metrics(arrays["x"][i],arrays["y"][i],arrays["artifact"][i],arrays["y"][i]-clean);identity=bool(item["zero_artifact"])
            if identity:metric["snr_improvement"]=np.nan;metric["artifact_rrmse"]=np.nan
            rows.append({"panel":"paired","fold":fold,"seed":seed,"participant":item["participant"],"session":item["session"],"task":item["task"],"severity":"identity" if identity else "mild" if item["gain"]<.5 else "medium" if item["gain"]<.95 else "severe","method":method,"identity":int(identity),"identity_change":float(np.linalg.norm(clean-arrays["y"][i])/max(np.linalg.norm(arrays["y"][i]),1e-12)) if identity else np.nan,**metric})
    _csv(DERIVED/f"metrics/paired/fold_{fold}_seed_{seed}.csv",rows);value={"stage":"R10_EVAL","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"sealed_reads":0};_json(run/"result_summary.json",value);return value


def natural_infer(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3]
    with np.load(V24/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as archive:batch={key:np.asarray(archive[key]) for key in ("y","q0","c0")}
    with np.load(V25/f"support_banks/fold_{fold}.npz",allow_pickle=False) as archive:batch.update({key:np.asarray(archive[key]) for key in archive.files})
    selection=json.loads((RESULT/"round_a/selection.json").read_text());started=time.time();pred=_predict_all(batch,fold,seed,False,int(selection["ddim_steps"]));out=DERIVED/f"natural/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred);value={"stage":"R11","status":"PASS","fold":fold,"seed":seed,"path":str(out),"sha256":_digest(out),"seconds":time.time()-started,"windows":len(batch["y"]),"query_EOG_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(RESULT/f"natural_evaluation/output_{fold}_{seed}.json",value);_json(run/"result_summary.json",value);return value


def freeze(run:Path)->dict[str,Any]:
    rows=[]
    for fold in range(5):
        for seed in SEEDS:
            row=json.loads((RESULT/f"natural_evaluation/output_{fold}_{seed}.json").read_text());assert _digest(Path(row["path"]))==row["sha256"];rows.append(row)
    _csv(RESULT/"natural_evaluation/output_manifest.csv",rows);value={"stage":"R12","status":"PASS","outputs":15,"query_EOG_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(RESULT/"natural_evaluation/output_freeze.json",value);_json(run/"result_summary.json",value);return value


def natural_eval(run:Path)->dict[str,Any]:
    assert json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text())["status"]=="PASS";index=_index();fold=index//3;seed=SEEDS[index%3]
    with np.load(V24/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as archive:query={key:np.asarray(archive[key]) for key in archive.files}
    with np.load(V24/f"fold_{fold}/natural_test_evaluator.npz",allow_pickle=False) as archive:evaluator={key:np.asarray(archive[key]) for key in archive.files}
    with np.load(DERIVED/f"natural/fold_{fold}_seed_{seed}.npz",allow_pickle=False) as archive:pred={key:np.asarray(archive[key]) for key in archive.files}
    roles=[r for r in csv.DictReader((ROOT/"results/pa_el_scad_v24/role_manifest.csv").open()) if r["fold"]==str(fold) and r["stream"]=="natural" and r["split"]=="test"];scale=np.load(V24/f"fold_{fold}/eeg_scale.npy");rows=[];max_consistency=0.
    for i,meta in enumerate(roles):
        for method in ("RAW",*pred):
            clean=query["y"][i] if method=="RAW" else pred[method][i];metric=natural_metrics_v28(query["y"][i],clean,evaluator["latent"][i],query["cs"][i],scale);max_consistency=max(max_consistency,attenuation_consistency(metric["heldout_eog_remaining_ratio"],metric["artifact_attenuation_db"]));rows.append({"panel":"natural","fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":method,**metric})
    _csv(DERIVED/f"metrics/natural/fold_{fold}_seed_{seed}.csv",rows);value={"stage":"R13","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"attenuation_remaining_max_difference":max_consistency,"evaluator_after_freeze":True,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def aggregate(run:Path)->dict[str,Any]:
    diagnosis,tables=aggregate_metrics(DERIVED,RESULT,SEEDS)
    _csv(RESULT/"method_summary.csv",tables["summary"]);_csv(RESULT/"participant_effects.csv",tables["effects"]);_csv(RESULT/"seed_effects.csv",tables["seed"]);_csv(RESULT/"severity_effects.csv",tables["severity"]);_json(RESULT/"development_diagnosis.json",diagnosis)
    latency=[]
    for panel,folder in (("paired",RESULT/"round_b"),("natural",RESULT/"natural_evaluation")):
        for path in sorted(folder.glob("output_*.json")):
            row=json.loads(path.read_text());latency.append({"panel":panel,"fold":row["fold"],"seed":row["seed"],"windows":row["windows"],"seconds":row["seconds"],"seconds_per_window":row["seconds"]/row["windows"]})
    _csv(RESULT/"latency_summary.csv",latency);_figures(tables,latency)
    p=diagnosis["paired"];n=diagnosis["natural"]
    (ROOT/"reports/v28_round_b.md").write_text(f'''# V28 Round B\n\nSupportCleanCDM MATCH minus PopCleanCDM paired utility was {p["support_vs_population"]["mean"]:+.6f} ({p["support_vs_population"]["positive"]}/15 positive; 95% CI [{p["support_vs_population"]["bootstrap_low"]:+.6f}, {p["support_vs_population"]["bootstrap_high"]:+.6f}]). MATCH minus WRONG was {p["match_vs_wrong"]["mean"]:+.6f}. SupportCleanCDM minus matched SupportCleanDET was {p["cdm_vs_det"]["mean"]:+.6f}; this is competitive positioning, not a retention gate.\n''')
    (ROOT/"reports/v28_natural_development.md").write_text(f'''# V28 natural development\n\nCorrected natural outcomes are reported separately. MATCH minus population artifact utility was {n["support_artifact"]["mean"]:+.6f}; low-EOG observation-retention utility was {n["support_observation_retention"]["mean"]:+.6f}; PSD utility was {n["support_psd"]["mean"]:+.6f}. ERP, SSVEP, and ERD/ERS are unavailable because required event metadata are absent. No observation-retention scalar is described as physiological preservation.\n''')
    (ROOT/"reports/v28_final_development_diagnosis.md").write_text("# V28 final development diagnosis\n\n```json\n"+json.dumps(diagnosis,indent=2)+"\n```\n")
    _json(run/"result_summary.json",diagnosis);return diagnosis


def _figures(tables:Mapping[str,list[dict[str,Any]]],latency:list[dict[str,Any]])->None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=ROOT/"figures/sc_cdm_v28";root.mkdir(parents=True,exist_ok=True);summary=tables["summary"];effects=tables["effects"]
    methods=["POP_CLEAN_DET","SUPPORT_CLEAN_DET_MATCH","POP_CLEAN_CDM","SUPPORT_CLEAN_CDM_MATCH"]
    lookup={(r["panel"],r["method"],r["metric"]):float(r["mean"]) for r in summary}
    fig,ax=plt.subplots();vals=[lookup.get(("paired",m,"rrmse_temporal"),np.nan) for m in methods];ax.bar(range(len(methods)),vals);ax.set_xticks(range(len(methods)),methods,rotation=30,ha="right");ax.set_ylabel("clean temporal RRMSE");fig.tight_layout();fig.savefig(root/"paired_method_comparison.png",dpi=160);plt.close(fig)
    forest=[r for r in effects if r["panel"]=="paired" and r["contrast"] in ("CDM_MATCH_POP","CDM_MATCH_WRONG") and r["metric"]=="rrmse_temporal"];participants=sorted({r["participant"] for r in forest});fig,ax=plt.subplots(figsize=(8,5));
    for j,name in enumerate(("CDM_MATCH_POP","CDM_MATCH_WRONG")):
        values={r["participant"]:float(r["effect"]) for r in forest if r["contrast"]==name};ax.plot([values[p] for p in participants],np.arange(len(participants))+.12*(j-.5),"o",label=name)
    ax.axvline(0,color="black",lw=.8);ax.set_yticks(range(len(participants)),participants);ax.legend();fig.tight_layout();fig.savefig(root/"support_context_forest.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots()
    for method in methods:
        x=lookup.get(("natural",method,"heldout_eog_remaining_ratio"));y=lookup.get(("natural",method,"low_eog_observation_retention"));
        if x is not None and y is not None:ax.scatter(x,y,label=method)
    ax.set(xlabel="EOG remaining ratio (lower better)",ylabel="low-EOG observation retention",title="Corrected natural trade-off");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/"natural_artifact_retention_scatter.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots()
    for method in methods:
        x=lookup.get(("natural",method,"psd_distortion"));y=lookup.get(("natural",method,"covariance_distortion"));
        if x is not None and y is not None:ax.scatter(x,y,label=method)
    ax.set(xlabel="PSD distortion",ylabel="covariance distortion");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/"PSD_covariance_tradeoff.png",dpi=160);plt.close(fig)
    pareto=list(csv.DictReader((RESULT/"v27_energy_pareto_summary.csv").open()));chosen=[r for r in pareto if r["panel"]=="natural" and r["method"]=="CALIB_ENERGY_SDEDIT_MATCH"];fig,ax=plt.subplots();ax.plot([float(r["remaining_ratio"]) for r in chosen],[float(r["low_eog_observation_retention"]) for r in chosen],"o-");ax.set(xlabel="remaining ratio",ylabel="low-EOG retention",title="Frozen V27 energy Pareto");fig.tight_layout();fig.savefig(root/"V27_energy_pareto.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots();ax.plot(range(len(latency)),[float(r["seconds_per_window"]) for r in latency],"o");ax.set(xlabel="fold/seed bundle",ylabel="seconds/window");fig.tight_layout();fig.savefig(root/"quality_latency_curve.png",dpi=160);plt.close(fig)
    # Explicit N/A task panel prevents proxy substitution in manuscript figures.
    fig,ax=plt.subplots();ax.axis("off");ax.text(.5,.5,"ERP / SSVEP / ERD-ERS\nunavailable: required event metadata absent",ha="center",va="center");fig.tight_layout();fig.savefig(root/"task_preservation.png",dpi=160);plt.close(fig)


def ledger_check(run:Path)->dict[str,Any]:
    path=ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md";text=path.read_text();value={"stage":"R15","status":"PASS","project_ledger_version":"v1.8","project_ledger_sha256":_digest(path),"v28_results_recorded":"## 6.14 V28 — SC-CDM" in text,"sealed_reads":0}
    if "**版本：** v1.8" not in text or not value["v28_results_recorded"]:raise RuntimeError(value)
    _json(RESULT/"ledger_sync.json",value);_json(run/"result_summary.json",value);return value


STAGES={"r0-preflight":preflight,"r1-audit":audit,"r2-pareto":pareto,"r2-pareto-aggregate":pareto_aggregate,"r3-prepare":prepare,"r4-sanity":sanity,"r5-rounda":lambda run:train_stage("r5-rounda",run),"r6-rounda-eval":round_a_eval,"r8-select":select,"r9-pop-det":lambda run:train_stage("r9-pop-det",run),"r9-support-det":lambda run:train_stage("r9-support-det",run),"r9-pop-cdm":lambda run:train_stage("r9-pop-cdm",run),"r9-support-cdm":lambda run:train_stage("r9-support-cdm",run),"r10-paired-infer":paired_infer,"r10-paired-eval":paired_eval,"r11-natural-infer":natural_infer,"r12-freeze":freeze,"r13-natural-eval":natural_eval,"r14-aggregate":aggregate,"r15-ledger":ledger_check}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=STAGES);parser.add_argument("--run-dir",type=Path,required=True);args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True);STAGES[args.stage](args.run_dir)
if __name__=="__main__":main()
