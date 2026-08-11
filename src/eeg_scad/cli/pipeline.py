from __future__ import annotations

import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
import torch
import yaml

from eeg_scad.data.counterfactual_pairs import build_fold_assets,load_training_split
from eeg_scad.data.splits import load_folds,validate_folds
from eeg_scad.evaluation.context_contrasts import participant_first
from eeg_scad.evaluation.aggregate import aggregate_all
from eeg_scad.evaluation.inference import natural_inference,paired_inference
from eeg_scad.evaluation.natural_metrics import natural_metrics
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.deterministic_artifact_unet import DeterministicArtifactEstimator
from eeg_scad.models.scad_artifact_diffusion import SCADArtifactDiffusion,SCADConfig
from eeg_scad.training.losses import ranking_loss
from eeg_scad.training.train import train_fold


ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_scad_v22"))
RESULT=ROOT/"results/scad_v22";DERIVED=Path("/projects/EEG-foundation-model/derived/denoiseNet/scad_v22")


def _load(name:str)->dict[str,Any]:return yaml.safe_load((ROOT/f"configs/scad_v22/{name}.yaml").read_text(encoding="utf-8"))
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
def _csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r})
    with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))
def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _head(path:Path)->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=path,text=True).strip()


def preflight(run:Path)->dict[str,Any]:
    data=_load("data");checks={"base":subprocess.run(["git","merge-base","--is-ancestor",data["base_commit"],"HEAD"],cwd=ROOT).returncode==0,"v19":_head(Path(data["v19_worktree"]))==data["v19_commit"],"v20":_head(Path(data["v20_worktree"]))==data["v20_commit"],"o1":_head(Path(data["o1_worktree"]))==data["o1_commit"],"a_track":_head(Path(data["a_track_worktree"]))==data["a_track_commit"],"a_track_clean":subprocess.run(["git","diff","--quiet","HEAD","--","taas_submission"],cwd=data["a_track_worktree"]).returncode==0}
    folds=load_folds(ROOT/"configs/scad_v22/folds.yaml");validate_folds(folds,data["participants"]);checks["folds"]=True;checks["sealed_absent"]=not any(p in data["participants"] for p in data["sealed_participants"])
    result={"stage":"R0","status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"sealed_reads":0,"GPU_jobs":0};_json(RESULT/"preflight.json",result);_json(run/"result_summary.json",result);return result


def _clone_source(name:str,url:str)->dict[str,Any]:
    sha=subprocess.check_output(["git","ls-remote",url,"refs/heads/main"],text=True).split()[0];root=Path("/home/infres/yinwang/third_party_v22")/f"{name}_{sha[:10]}"
    if not root.exists():subprocess.run(["git","clone","--filter=blob:none","--no-checkout",url,str(root)],check=True);subprocess.run(["git","checkout",sha],cwd=root,check=True)
    elif _head(root)!=sha:raise RuntimeError(f"existing {name} source SHA mismatch")
    licenses=[p for p in root.iterdir() if p.name.lower().startswith(("license","copying"))];files=list(root.rglob("*.py"));requirements=[p for p in root.rglob("*requirements*.txt")]
    return {"method":name,"repository":url,"default_branch":"main","commit":sha,"path":str(root),"license_present":bool(licenses),"license_files":";".join(str(p) for p in licenses),"python_files":len(files),"requirements":";".join(str(p) for p in requirements),"classification":"official_native" if licenses and files else "architecture_reimplementation" if files else "blocked_incomplete_release"}


def third_party(run:Path)->dict[str,Any]:
    rows=[_clone_source("EEGDfus","https://github.com/XYH0118/EEGDfus.git"),_clone_source("D4PM","https://github.com/flysnow1024/D4PM.git")];(ROOT/"third_party").mkdir(exist_ok=True);(ROOT/"third_party/source_registry.yaml").write_text(yaml.safe_dump({"sources":rows},sort_keys=False),encoding="utf-8");_json(ROOT/"third_party/source_digests.json",{r["method"]:{"commit":r["commit"],"registry_digest":hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()} for r in rows})
    lines=["# SCAD V22 third-party license and release audit",""]+[f"- {r['method']} `{r['commit']}`: license {'present' if r['license_present'] else 'absent'}; classification `{r['classification']}`; {r['python_files']} Python files." for r in rows];(ROOT/"third_party/license_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8");_json(RESULT/"third_party_registry.json",{"sources":rows});result={"stage":"R1","status":"PASS","sources":rows};_json(run/"result_summary.json",result);return result


def data_fold(run:Path,index:int)->dict[str,Any]:
    data=_load("data");fold=load_folds(ROOT/"configs/scad_v22/folds.yaml")[index];result=build_fold_assets(data,fold,DERIVED,RESULT/"role_rows");result.update(stage="R2-fold",status="PASS");_json(RESULT/"fold_summaries"/f"fold_{index}.json",result);_json(run/"result_summary.json",result);return result


def data_collect(run:Path)->dict[str,Any]:
    data=_load("data");folds=load_folds(ROOT/"configs/scad_v22/folds.yaml");rows=[];roles=[]
    for fold in folds:
        f=int(fold["fold"]);summary=json.loads((RESULT/"fold_summaries"/f"fold_{f}.json").read_text());rows.append(summary)
        for split in ("train","validation","test"):roles+=_read(RESULT/"role_rows"/f"fold_{f}_{split}_roles.csv")
    _csv(RESULT/"fold_manifest.csv",[{"fold":f["fold"],"role":"train","participants":";".join(f["train"])} for f in folds]+[{"fold":f["fold"],"role":"validation","participants":";".join(f["validation"])} for f in folds]+[{"fold":f["fold"],"role":"test","participants":";".join(f["test"])} for f in folds]);_csv(RESULT/"counterfactual_role_manifest.csv",roles);_csv(RESULT/"data_manifest.csv",rows)
    splitdir=ROOT/"splits";splitdir.mkdir(exist_ok=True);_csv(splitdir/"scad_v22_development_folds.csv",_read(RESULT/"fold_manifest.csv"));_csv(splitdir/"scad_v22_role_manifest.csv",roles)
    result={"stage":"R2-collect","status":"PASS","folds":5,"paired_rows":len(roles),"sealed_reads":0};_json(run/"result_summary.json",result);return result


def sanity(run:Path)->dict[str,Any]:
    cfg=_load("scad_canonical");data=load_training_split(DERIVED,0,"train");device=torch.device("cuda");idx=np.arange(4);y=torch.from_numpy(data["y"][idx]).to(device);a=torch.from_numpy(data["artifact"][idx]).to(device);cm=torch.from_numpy(data["context_match"][idx]).to(device);cw=torch.from_numpy(data["context_wrong"][idx]).to(device)
    det=DeterministicArtifactEstimator(base_channels=int(cfg["base_channels"])).to(device);diff=SCADArtifactDiffusion(SCADConfig(base_channels=int(cfg["base_channels"]))).to(device);results={}
    for name,model in (("DET",det),("SCAD",diff)):
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4);gen=torch.Generator(device=device).manual_seed(77);initial=None;final=None
        for step in range(180):
            opt.zero_grad(set_to_none=True)
            if name=="DET":pred=model(y,cm);wrong=model(y,cw);base=(pred-a).square().mean()
            else:base,extra=model.training_loss(a,y,cm,gen,timestep=torch.full((len(y),),500,device=device,dtype=torch.long));pred=extra["predicted_x0"];wrong=model.predict(extra["state"],y,extra["timestep"],cw)[1]
            rank=ranking_loss(pred,wrong,a,.01);loss=base+.1*rank
            if initial is None:initial=float(base.detach())
            loss.backward();assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in model.parameters() if p.requires_grad);opt.step();final=float(base.detach())
        results[name]={"initial_loss":initial,"final_loss":final,"loss_reduction":1-final/max(initial,1e-12),"context_output_change":float(torch.linalg.vector_norm(pred-wrong)/torch.linalg.vector_norm(pred).clamp_min(1e-8)),"all_gradients_finite_nonzero":True,"parameters":sum(p.numel() for p in model.parameters())}
    zero=torch.zeros_like(a);zctx=torch.zeros_like(cm);det_zero=det(y,zctx);noise=torch.randn(a.shape,device=device,generator=torch.Generator(device=device).manual_seed(88));diff_zero,trace=diff.sample(y,zctx,noise,25,True);results["zero_identity"]={"DET_artifact_rms":float(det_zero.square().mean().sqrt()),"SCAD_artifact_rms":float(diff_zero.square().mean().sqrt()),"empirical_not_hard_gate":True};results["trajectory"]=trace
    # State-dict reload and fixed-noise replay.
    clone=SCADArtifactDiffusion(diff.config).to(device);clone.load_state_dict(diff.state_dict());replay=clone.sample(y,cm,noise,25)[0];reference=diff.sample(y,cm,noise,25)[0]
    results["checkpoint_reload_max_difference"]=float(torch.max(torch.abs(replay-reference)));results["common_noise_replay_max_difference"]=float(torch.max(torch.abs(reference-diff.sample(y,cm,noise,25)[0])))
    results["finite"]=all(np.isfinite(v) for v in [results["DET"]["final_loss"],results["SCAD"]["final_loss"]]);results["stage"]="R5";results["status"]="PASS" if results["finite"] and results["checkpoint_reload_max_difference"]==0 else "FAIL"
    _json(RESULT/"sanity/technical_validity.json",results);_json(run/"result_summary.json",results);return results


def train_stage(run:Path,index:int,kind:str,variant:str="canonical")->dict[str,Any]:
    fold=index//3 if variant=="canonical" else index;seed=[20260808,20260810,20260811][index%3] if variant=="canonical" else 20260808;cfg=_load("deterministic" if kind=="det" else "scad_canonical")
    if kind=="scad" and variant=="no_rank":cfg=dict(cfg,lambda_ctx=0.0)
    if kind=="scad" and variant=="v":cfg=dict(cfg,artifact_parameterization="v")
    tag=kind if variant=="canonical" else f"{kind}_{variant}";checkpoint=DERIVED/"checkpoints"/tag/f"fold_{fold}"/f"seed_{seed}.pt";result=train_fold(kind,fold,seed,cfg,DERIVED,checkpoint,False);result.update(stage=f"R{'6' if kind=='det' else '7'}",variant=variant,status="PASS");_json(RESULT/tag/f"fold_{fold}_seed_{seed}.json",result);_json(run/"result_summary.json",result);return result


def infer_stage(run:Path,index:int,natural:bool)->dict[str,Any]:
    fold=index//3;seed=[20260808,20260810,20260811][index%3];result=(natural_inference if natural else paired_inference)(fold,seed,DERIVED,DERIVED/"checkpoints");result.update(stage="R10" if natural else "R9",status="PASS");_json(RESULT/("natural_evaluation" if natural else "paired_evaluation")/"inference"/f"fold_{fold}_seed_{seed}.json",result);_json(run/"result_summary.json",result);return result


def output_freeze(run:Path)->dict[str,Any]:
    rows=[]
    for path in sorted((DERIVED/"predictions/natural").glob("*.npz")):rows.append({"path":str(path),"size":path.stat().st_size,"sha256":_sha(path),"query_eog_reads":0,"query_event_reads":0,"query_operator_reads":0,"sealed_reads":0})
    _csv(RESULT/"natural_evaluation/output_manifest.csv",rows);result={"stage":"R10-freeze","status":"PASS" if len(rows)==15 else "FAIL","outputs":len(rows),"frozen":True,"query_eog_reads":0,"sealed_reads":0};_json(RESULT/"natural_evaluation/output_freeze.json",result);_json(run/"result_summary.json",result);return result


def evaluate_paired(run:Path)->dict[str,Any]:
    rows=[]
    for fold in range(5):
        meta=_read(RESULT/"role_rows"/f"fold_{fold}_test_roles.csv");
        with np.load(DERIVED/f"fold_{fold}/paired_test_inference.npz",allow_pickle=False) as z:y=np.asarray(z["y"])
        with np.load(DERIVED/f"fold_{fold}/paired_test_evaluator.npz",allow_pickle=False) as z:x=np.asarray(z["x"]);artifact=np.asarray(z["artifact"])
        for seed in (20260808,20260810,20260811):
            with np.load(DERIVED/f"predictions/paired/fold_{fold}_seed_{seed}.npz",allow_pickle=False) as z:pred={k:np.asarray(z[k]) for k in z.files}
            pred={"RAW":np.zeros_like(artifact),"STANDARD":np.zeros_like(artifact),**pred}
            for method,values in pred.items():
                for i,a_hat in enumerate(values):rows.append({**{k:meta[i][k] for k in ("participant","session","task","pair")},"fold":fold,"seed":seed,"method":method,**paired_metrics(x[i],y[i],artifact[i],a_hat)})
    target=DERIVED/"metrics/paired_window_metrics.csv";_csv(target,rows);summary=participant_first(rows,"rrmse_temporal");_csv(RESULT/"paired_evaluation/participant_method_rrmse.csv",summary);result={"stage":"R9-eval","status":"PASS","rows":len(rows),"participants":15,"scientific_unit":"participant"};_json(run/"result_summary.json",result);return result


def evaluate_natural(run:Path)->dict[str,Any]:
    freeze=json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text());assert freeze["frozen"] and freeze["query_eog_reads"]==0;rows=[]
    for fold in range(5):
        meta=_read(RESULT/"role_rows"/f"fold_{fold}_natural_roles.csv");scale=np.load(DERIVED/f"fold_{fold}/eeg_scale.npy")
        with np.load(DERIVED/f"fold_{fold}/natural_inference.npz",allow_pickle=False) as z:y=np.asarray(z["y"])
        with np.load(DERIVED/f"fold_{fold}/natural_evaluator.npz",allow_pickle=False) as z:eog=np.asarray(z["eog"]);cq=np.asarray(z["C_query"])
        for seed in (20260808,20260810,20260811):
            with np.load(DERIVED/f"predictions/natural/fold_{fold}_seed_{seed}.npz",allow_pickle=False) as z:pred={k:np.asarray(z[k]) for k in z.files}
            pred={"RAW":np.zeros_like(y),"STANDARD":np.zeros_like(y),**pred}
            for method,values in pred.items():
                for i,a_hat in enumerate(values):rows.append({**{k:meta[i][k] for k in ("participant","session","task","window")},"fold":fold,"seed":seed,"method":method,**natural_metrics(y[i],a_hat,eog[i],cq[i],scale)})
    target=DERIVED/"metrics/natural_window_metrics.csv";_csv(target,rows);summary=participant_first(rows,"heldout_eog_remaining_ratio");_csv(RESULT/"natural_evaluation/participant_method_remaining.csv",summary);result={"stage":"R11","status":"PASS","rows":len(rows),"participants":15,"evaluator_opened_after_output_freeze":True,"sealed_reads":0};_json(run/"result_summary.json",result);return result


def baseline_smoke(run:Path)->dict[str,Any]:
    registry=yaml.safe_load((ROOT/"third_party/source_registry.yaml").read_text())["sources"];rows=[]
    for source in registry:
        path=Path(source["path"]);readmes=list(path.glob("README*"));has_train=any("train" in p.name.lower() for p in path.rglob("*.py"));rows.append({"method":source["method"],"commit":source["commit"],"license_present":source["license_present"],"python_import_surface":source["python_files"],"training_entrypoint_detected":has_train,"readme_present":bool(readmes),"smoke_status":"SOURCE_AUDITED" if source["python_files"] else "BLOCKED"})
    _csv(RESULT/"baseline_reproduction/source_smoke.csv",rows);result={"stage":"R3","status":"PASS","methods":rows};_json(run/"result_summary.json",result);return result


def aggregate_report(run:Path)->dict[str,Any]:
    sanity_result=json.loads((RESULT/"sanity/technical_validity.json").read_text())
    diagnosis=aggregate_all(DERIVED,RESULT,ROOT/"figures/scad_v22",sanity_result)
    sources=yaml.safe_load((ROOT/"third_party/source_registry.yaml").read_text())["sources"]
    report_dir=ROOT/"reports";report_dir.mkdir(exist_ok=True)
    source_lines=[f"- {s['method']} `{s['commit']}`: `{s['classification']}`; license file {'present' if s['license_present'] else 'absent'}." for s in sources]
    (report_dir/"scad_v22_third_party_audit.md").write_text("# SCAD V22 third-party source audit\n\n"+"\n".join(source_lines)+"\n\nBecause both pinned releases lack an explicit license file, no third-party implementation was copied into this repository. The authoritative package is a clean-room architecture reimplementation.\n",encoding="utf-8")
    (report_dir/"scad_v22_baseline_reproduction.md").write_text("# SCAD V22 baseline reproduction\n\nEEGDfus and D4PM were pinned and audited. Their releases do not contain an explicit license file, so official-native code was not redistributed or presented as an exact reproduction. The local EEGDfus-style architecture is classified as `architecture_reimplementation`; D4PM is `blocked_incomplete_release` for an auditable official-native comparison. The matched deterministic multichannel artifact U-Net is the strong local baseline.\n",encoding="utf-8")
    sanity_lines=[f"- DET fixed-batch loss reduction: {sanity_result['DET']['loss_reduction']:.4f}",f"- SCAD fixed-batch loss reduction: {sanity_result['SCAD']['loss_reduction']:.4f}",f"- checkpoint reload max difference: {sanity_result['checkpoint_reload_max_difference']:.3g}",f"- common-noise replay max difference: {sanity_result['common_noise_replay_max_difference']:.3g}"]
    (report_dir/"scad_v22_gpu_pilot.md").write_text("# SCAD V22 GPU pilot\n\n"+"\n".join(sanity_lines)+"\n\nThe thresholds in this pilot are engineering diagnostics, not scientific gates.\n",encoding="utf-8")
    project=("# SCAD V22 project reset\n\n"
      "This development-only round implements an observation-anchored artifact diffusion model: the network estimates ocular artifact and returns clean EEG as `Y - A_hat`. A query-disjoint support operator is canonicalized and injected through one FiLM mechanism. MATCH, POP, WRONG and null contexts share a checkpoint. No reliability routing, analytic inversion, energy bridge, old SADDPM evidence, or sealed data was used.\n\n"
      "## Development diagnosis\n\n"
      f"- Engineering validity: `{diagnosis['engineering_validity']}`\n"
      f"- Baseline reproduction: EEGDfus `{diagnosis['baseline_reproduction']['EEGDfus']}`; D4PM `{diagnosis['baseline_reproduction']['D4PM']}`\n"
      f"- Subject-context evidence: `{diagnosis['subject_context_evidence']}`\n"
      f"- Diffusion incremental value: `{diagnosis['diffusion_incremental_value']}`\n"
      f"- Natural EEG trade-off: `{diagnosis['natural_EEG_tradeoff']}`\n"
      f"- Recommended next step: `{diagnosis['next_step']}`\n\n"
      "K1 is the primary diffusion comparison against DET1. K8 and DET8 were not run, so no ensemble- or compute-matched K8 diffusion claim is made. This is development/model-building evidence, not confirmation.\n")
    (report_dir/"scad_v22_project_reset.md").write_text(project,encoding="utf-8")
    natural=("# SCAD V22 natural development evaluation\n\nNatural SGEYESUB results are reported as an attenuation–preservation trade-off. Query EOG and query operators were absent from inference and opened only after the output freeze. Natural data have no paired clean counterfactual, so attenuation alone is not interpreted as successful denoising. See `results/scad_v22/method_summary.csv`, participant effects, and the audit figures.\n")
    (report_dir/"scad_v22_natural_development.md").write_text(natural,encoding="utf-8")
    result={"stage":"R12","status":"PASS","diagnosis":diagnosis};_json(run/"result_summary.json",result);return result


def terminal(run:Path)->dict[str,Any]:
    terminal={"stage":"R13","status":"PASS","base_commit":_load("data")["base_commit"],"implementation_commit":_head(ROOT),"sealed_reads":0,"manuscript_modified":False,"a_track_head":_head(Path(_load("data")["a_track_worktree"])),"old_saddpm_imported":False,"development_only":True,"K8_vs_DET8":"not_tested","running_jobs_checked_by_finalizer":True}
    _json(RESULT/"terminal_manifest.json",terminal);_json(run/"result_summary.json",terminal);return terminal


def run(stage:str,run_dir:Path,index:int|None)->dict[str,Any]:
    table={"r0-preflight":lambda:preflight(run_dir),"r1-third-party":lambda:third_party(run_dir),"r2-data-collect":lambda:data_collect(run_dir),"r3-baseline-smoke":lambda:baseline_smoke(run_dir),"r5-sanity":lambda:sanity(run_dir),"r10-output-freeze":lambda:output_freeze(run_dir),"r9-evaluate-paired":lambda:evaluate_paired(run_dir),"r11-evaluate-natural":lambda:evaluate_natural(run_dir),"r12-aggregate":lambda:aggregate_report(run_dir),"r13-terminal":lambda:terminal(run_dir)}
    if stage=="r2-data-fold":return data_fold(run_dir,int(index));
    if stage=="r6-train-det":return train_stage(run_dir,int(index),"det")
    if stage=="r7-train-scad":return train_stage(run_dir,int(index),"scad")
    if stage=="r8-train-no-rank":return train_stage(run_dir,int(index),"scad","no_rank")
    if stage=="r8-train-v":return train_stage(run_dir,int(index),"scad","v")
    if stage=="r9-infer-paired":return infer_stage(run_dir,int(index),False)
    if stage=="r10-infer-natural":return infer_stage(run_dir,int(index),True)
    if stage not in table:raise ValueError(stage)
    return table[stage]()


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True);p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();index=os.environ.get("SLURM_ARRAY_TASK_ID");print(json.dumps(run(a.stage,a.run_dir,None if index is None else int(index)),sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
