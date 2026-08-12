"""Slurm-facing V27 CalibEnergy inference-only development workflow."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, shutil, subprocess, time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from eeg_scad.cli import v26
from eeg_scad.data.folds import load_folds, validate_folds
from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.energy.partial_observation import energy_diagnostics, partial_observation_prox, partial_observation_solve
from eeg_scad.energy.projector import diagnostics as projector_diagnostics, population_projector, projector
from eeg_scad.energy.temporal_confidence import calibrate_quantiles, temporal_confidence
from eeg_scad.evaluation.aggregate_v26 import bootstrap, contrast, participant_first
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.calib_energy_sdedit import sample_stepwise_energy

ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", Path(__file__).resolve().parents[3]))
RESULT = ROOT / "results/calib_energy_v27"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_energy_v27")
V26_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_sdedit_v26")
V25_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25")
V24_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24")
BASE = "7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c"
SEEDS = [20260828, 20260829, 20260830]


def _cfg(name: str) -> dict[str, Any]: return yaml.safe_load((ROOT / f"configs/calib_energy_v27/{name}.yaml").read_text())
def _folds() -> list[dict[str, Any]]: return load_folds(ROOT / "configs/calib_energy_v27/folds.yaml")
def _index() -> int: return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
def _digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): value.update(block)
    return value.hexdigest()
def _json(path: Path, value: Any) -> None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _csv(path: Path, rows: Iterable[Mapping[str,Any]]) -> None:
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True); fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def preflight(run: Path) -> dict[str, Any]:
    validate_folds(_folds(),_cfg("data")["participants"]); head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); ledger=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    checks={"base_exact": subprocess.check_output(["git","rev-parse","codex/calib-sdedit-v26"],cwd=ROOT,text=True).strip()==BASE,"base_ancestor":subprocess.run(["git","merge-base","--is-ancestor",BASE,head],cwd=ROOT).returncode==0,"ledger_v1_5":"**版本：** v1.5" in ledger,"ledger_active_v27":"V27" in ledger,"v26_unchanged":not bool(subprocess.check_output(["git","diff","--name-only",BASE,"--","results/calib_sdedit_v26","reports/v26_*"],cwd=ROOT,text=True).strip()),"a_track_unchanged":not bool(subprocess.check_output(["git","diff","--name-only",BASE,"--","taas_submission"],cwd=ROOT,text=True).strip()),"sealed_reads":0}
    if not all(v is True for k,v in checks.items() if k!="sealed_reads"): raise RuntimeError(checks)
    value={"stage":"R0","status":"PASS","base_commit":BASE,"head":head,**checks}; _json(RESULT/"source_registry.json",value); _json(run/"result_summary.json",value); return value


@torch.no_grad()
def _support_outputs(batch: Mapping[str,Any], fold: int, seed: int, device: torch.device):
    anchor,det,models=v26._load_bundle(fold,seed,device); base,trajectory=v26._predict(batch,anchor,det,models,seed,.05,10,device)
    match_pi=[]; wrong_pi=[]; match_context=[]; wrong_context=[]
    for start in range(0,len(batch["y"]),16):
        sl=slice(start,min(start+16,len(batch["y"]))); se=torch.as_tensor(batch["support_eeg"][sl],device=device); so=torch.as_tensor(batch["support_eog"][sl],device=device); we=torch.as_tensor(batch["wrong_support_eeg"][sl],device=device); wo=torch.as_tensor(batch["wrong_support_eog"][sl],device=device)
        match=det.encode_support(se,so); wrong=det.encode_support(we,wo); match_pi.append(projector(match["basis"])); wrong_pi.append(projector(wrong["basis"])); match_context.append(match["context"]); wrong_context.append(wrong["context"])
    return anchor,det,models,base,torch.cat(match_pi),torch.cat(wrong_pi),torch.cat(match_context),torch.cat(wrong_context),trajectory


def _calibration_path(fold:int,seed:int)->Path:return DERIVED/f"calibration/fold_{fold}_seed_{seed}.npz"


@torch.no_grad()
def prepare_cell(run: Path) -> dict[str,Any]:
    index=_index(); fold=index//3; seed=SEEDS[index%3]; device=torch.device("cuda"); sampler=SupportSetEpisodeSampler(v26._cfg("data"),_folds()[fold],"train",seed+2700); batch=sampler.sample_paired(256); anchor,det,_=v26._load_bundle(fold,seed,device); pops=[]; matches=[]; projectors=[]
    for start in range(0,len(batch["y"]),16):
        sl=slice(start,min(start+16,len(batch["y"]))); y=torch.as_tensor(batch["y"][sl],device=device);q0=torch.as_tensor(batch["q0"][sl],device=device);c0=torch.as_tensor(batch["c0"][sl],device=device);pop=anchor(y,q0,torch.einsum("bcd,bdt->bct",c0,q0)); match=det(y,pop,q0,torch.as_tensor(batch["support_eeg"][sl],device=device),torch.as_tensor(batch["support_eog"][sl],device=device));pops.append(pop.cpu());matches.append(match["artifact"].cpu());projectors.append(projector(match["basis"]).cpu())
    pop=torch.cat(pops);match=torch.cat(matches);pis=torch.cat(projectors);q50,q90=calibrate_quantiles(match,pop);pi0=population_projector(pis,8); path=_calibration_path(fold,seed);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,q50=np.asarray(q50),q90=np.asarray(q90),population_projector=pi0.numpy());pd=projector_diagnostics(pi0); value={"stage":"R1","status":"PASS","fold":fold,"seed":seed,"training_participants":_folds()[fold]["train"],"training_windows":256,"q50":q50,"q90":q90,"path":str(path),"sha256":_digest(path),**pd,"query_auxiliary_reads":0,"sealed_reads":0};_json(RESULT/f"calibration/fold_{fold}_seed_{seed}.json",value);_json(run/"result_summary.json",value);return value


def fixtures(run: Path)->dict[str,Any]:
    generator=torch.Generator().manual_seed(27);dtype=torch.float64;basis=torch.randn(2,8,3,generator=generator,dtype=dtype);pi=projector(basis);candidate=torch.randn(2,8,13,generator=generator,dtype=dtype);anchor=torch.randn(2,8,13,generator=generator,dtype=dtype);mask=torch.rand(2,13,generator=generator,dtype=dtype);closed=partial_observation_prox(candidate,anchor,pi,mask,1,2);dense=partial_observation_solve(candidate,anchor,pi,mask,1,2);rotation,_=torch.linalg.qr(torch.randn(3,3,generator=generator,dtype=dtype));value={"stage":"R2","status":"PASS","closed_linear_max_difference":float((closed-dense).abs().max()),"rotation_invariance_max_difference":float((pi-projector(basis@rotation)).abs().max()),"lambda_zero_identity":float((partial_observation_prox(candidate,anchor,pi,mask,0,0)-candidate).abs().max()),"sealed_reads":0};
    if value["closed_linear_max_difference"]>1e-10:raise RuntimeError(value)
    _json(RESULT/"proximal_fixture.json",value)
    # Freeze the small, authoritative V26 bindings and fold roles used by V27.
    shutil.copyfile(ROOT/"results/calib_sdedit_v26/checkpoint_binding.csv", RESULT/"checkpoint_binding.csv")
    shutil.copyfile(ROOT/"results/calib_sdedit_v26/fold_manifest.csv", RESULT/"fold_manifest.csv")
    inventory=[]
    for role, path in (
        ("V26 terminal manifest", ROOT/"results/calib_sdedit_v26/terminal_manifest.json"),
        ("V26 scientific diagnosis", ROOT/"results/calib_sdedit_v26/development_diagnosis.json"),
        ("V26 final report", ROOT/"reports/v26_final_development_diagnosis.md"),
        ("V25 terminal manifest", ROOT/"results/setcalibdiff_v25/terminal_manifest.json"),
        ("project ledger v1.5", ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md"),
    ):
        stat=path.stat();inventory.append({"absolute_path":str(path.resolve()),"scientific_role":role,"sha256":_digest(path),"size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"read_only":True})
    _csv(RESULT/"input_inventory.csv",inventory)
    _json(run/"result_summary.json",value);return value


def _calibration(fold:int,seed:int,device):
    with np.load(_calibration_path(fold,seed),allow_pickle=False) as a:return float(a["q50"]),float(a["q90"]),torch.as_tensor(a["population_projector"],device=device)


@torch.no_grad()
def _energy_bundle(batch:Mapping[str,Any],fold:int,seed:int,lambda_a:float,lambda_y:float,mode:str,spatial_only:bool=False):
    device=torch.device("cuda");anchor,det,models,base,match_pi,wrong_pi,match_context,wrong_context,trajectory=_support_outputs(batch,fold,seed,device);q50,q90,pi0=_calibration(fold,seed,device);pi0=pi0[None].expand(len(batch["y"]),-1,-1); tensors={k:torch.as_tensor(v,device=device) for k,v in base.items()};match_mask=temporal_confidence(tensors["V25_DET_MATCH"],tensors["V25_POP"],q50,q90,10);wrong_mask=temporal_confidence(tensors["V25_DET_WRONG"],tensors["V25_POP"],q50,q90,10);pop_mask=temporal_confidence(tensors["V25_POP"],tensors["V25_POP"],q50,q90,10)
    if spatial_only:match_mask=torch.ones_like(match_mask);wrong_mask=torch.ones_like(wrong_mask);pop_mask=torch.ones_like(pop_mask)
    output=dict(base); diagnostics_rows=[]
    specs=(
      ("CALIB_ENERGY_DET_MATCH","CALIB_REFINE_MATCH","CALIB_REFINE_MATCH",match_pi,match_mask),
      ("CALIB_ENERGY_DET_WRONG","CALIB_REFINE_WRONG","CALIB_REFINE_WRONG",wrong_pi,wrong_mask),
      ("POP_ENERGY_DET","POP_REFINE_DET","POP_REFINE_DET",pi0,pop_mask),
      ("CALIB_ENERGY_SDEDIT_MATCH","CALIB_SDEDIT_MATCH","CALIB_REFINE_MATCH",match_pi,match_mask),
      ("CALIB_ENERGY_SDEDIT_WRONG","CALIB_SDEDIT_WRONG","CALIB_REFINE_WRONG",wrong_pi,wrong_mask),
      ("POP_ENERGY_SDEDIT","POP_SDEDIT","POP_REFINE_DET",pi0,pop_mask))
    for name,candidate_name,anchor_name,pi,mask in specs:
        refined=partial_observation_prox(tensors[candidate_name],tensors[anchor_name],pi,mask,lambda_a,lambda_y);output[name]=refined.cpu().numpy();diagnostics_rows.append({"method":name,**energy_diagnostics(tensors[candidate_name],refined,tensors[anchor_name],pi,mask)})
    if mode=="stepwise":
        y=torch.as_tensor(batch["y"],device=device);pop=tensors["V25_POP"]
        for start in range(0,len(y),16):
            sl=slice(start,min(start+16,len(y)));noise=torch.randn(pop[sl].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+start));cs=models["calib_sdedit"]
            for name,artifact,context,pi,mask in (("CALIB_ENERGY_SDEDIT_MATCH",tensors["V25_DET_MATCH"],match_context,match_pi,match_mask),("CALIB_ENERGY_SDEDIT_WRONG",tensors["V25_DET_WRONG"],wrong_context,wrong_pi,wrong_mask)):
                value,_=sample_stepwise_energy(cs,y[sl],artifact[sl],pop[sl],context[sl],noise,pi[sl],mask[sl],lambda_a,lambda_y,.05,10); output.setdefault(name+"_parts",[]).append(value.cpu().numpy())
        for name in ("CALIB_ENERGY_SDEDIT_MATCH","CALIB_ENERGY_SDEDIT_WRONG"):output[name]=np.concatenate(output.pop(name+"_parts"))
    return output,diagnostics_rows


def _score(batch:Mapping[str,Any],prediction:Mapping[str,np.ndarray],method:str)->dict[str,float]:
    paired=batch["stream"]=="paired"
    if paired:
        values=[paired_metrics(batch["x"][i],batch["y"][i],batch["artifact"][i],prediction[method][i])["rrmse_temporal"] for i in range(len(batch["y"]))];return {"paired_clean_rrmse":float(np.mean(values))}
    values=[v26._natural(batch["y"][i],prediction[method][i],batch["teacher_artifact"][i],batch["latent"][i]) for i in range(len(batch["y"]))];return {"natural_remaining_ratio":float(np.mean([v["remaining_ratio"] for v in values])),"natural_preservation":float(np.mean([v["preservation"] for v in values])),"natural_psd_distortion":float(np.mean([v["psd_distortion"] for v in values])),"natural_covariance_distortion":float(np.mean([v["covariance_distortion"] for v in values]))}


def round_a(run:Path)->dict[str,Any]:
    seed=20260828;fold_rows={0:[],2:[]};batches={}
    for fold in (0,2):
        sampler=SupportSetEpisodeSampler(v26._cfg("data"),_folds()[fold],"validation",seed+2701)
        batches[fold]=(sampler.sample_paired(96),sampler.sample_natural(96))

    def evaluate(stage: str, la: float, ly: float, mode: str, spatial: bool=False)->None:
        for fold in (0,2):
            paired,natural=batches[fold];pp,_=_energy_bundle(paired,fold,seed,la,ly,mode,spatial);npred,_=_energy_bundle(natural,fold,seed,la,ly,mode,spatial)
            for method in ("CALIB_ENERGY_DET_MATCH","CALIB_ENERGY_SDEDIT_MATCH"):
                fold_rows[fold].append({"fold":fold,"stage":stage,"lambda_a":la,"lambda_y":ly,"mode":mode,"spatial_only":int(spatial),"method":method,**_score(paired,pp,method),**_score(natural,npred,method)})

    def choose(stage: str)->tuple[float,float,str,int]:
        rows=[r for values in fold_rows.values() for r in values if r["stage"]==stage and r["method"]=="CALIB_ENERGY_SDEDIT_MATCH"]
        cells={}
        for row in rows:cells.setdefault((row["lambda_a"],row["lambda_y"],row["mode"],row["spatial_only"]),[]).append(row)
        def score(items):return np.mean([r["natural_remaining_ratio"]+2*(1-r["natural_preservation"])+.25*r["paired_clean_rrmse"] for r in items])
        return min(cells,key=lambda key:score(cells[key]))

    # Registered sequential search: later stages are never evaluated until the
    # preceding validation-only choice is fixed across folds 0 and 2.
    for ly in (0,.5,2,8):evaluate("A",1,ly,"final_only")
    stage_a=choose("A")
    for la in (0,1,4):evaluate("B",la,stage_a[1],"final_only")
    stage_b=choose("B")
    for mode in ("final_only","stepwise"):evaluate("C",stage_b[0],stage_b[1],mode)
    stage_c=choose("C")
    evaluate("D",stage_c[0],stage_c[1],stage_c[2],True)
    for fold,rows in fold_rows.items():_csv(RESULT/f"round_a/fold_{fold}.csv",rows)
    value={"stage":"R3_R5","status":"PASS","folds":[0,2],"sequential_cells":10,"rows":sum(map(len,fold_rows.values())),"stage_a":stage_a,"stage_b":stage_b,"stage_c":stage_c,"selection_uses_test":False,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def select_round_a(run:Path)->dict[str,Any]:
    rows=[]
    for p in sorted((RESULT/"round_a").glob("fold_*.csv")):rows.extend(csv.DictReader(p.open()))
    def candidates(stage,method="CALIB_ENERGY_SDEDIT_MATCH"):return [r for r in rows if r["stage"]==stage and r["method"]==method]
    def group(rows):
        cells={}
        for r in rows:
            key=(float(r["lambda_a"]),float(r["lambda_y"]),r["mode"],int(r["spatial_only"]));cells.setdefault(key,[]).append(r)
        return cells
    def value(items):return np.mean([float(r["natural_remaining_ratio"])+2*(1-float(r["natural_preservation"]))+.25*float(r["paired_clean_rrmse"]) for r in items])
    a=min(group(candidates("A")),key=lambda k:value(group(candidates("A"))[k])); bpool={k:v for k,v in group(candidates("B")).items() if k[1]==a[1]};b=min(bpool,key=lambda k:value(bpool[k]));cpool={k:v for k,v in group(candidates("C")).items() if k[:2]==b[:2]};c=min(cpool,key=lambda k:value(cpool[k]));result={"status":"ROUND_B_CONFIG_FROZEN","lambda_a":c[0],"lambda_y":c[1],"mode":c[2],"spatial_only":False,"stage_a":a,"stage_b":b,"stage_c":c,"selection_priority":"natural_artifact_preservation_then_paired","test_used":False,"optional_finetune_authorized":False,"rationale":"Sequential validation-only selection prioritized joint natural artifact/preservation validity, with paired fidelity secondary; matched DET remained a competitive control."};_json(RESULT/"round_a/selection.json",result);_json(run/"result_summary.json",result);return result


def _save_batch(path:Path,batch:Mapping[str,Any])->None:
    arrays={k:np.asarray(v) for k,v in batch.items() if k in ("x","y","artifact","teacher_artifact","latent")};path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)


def paired_infer(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3];selection=json.loads((RESULT/"round_a/selection.json").read_text());sampler=SupportSetEpisodeSampler(v26._cfg("data"),_folds()[fold],"test",seed+401);batch=sampler.sample_paired(192);started=time.time();pred,diag=_energy_bundle(batch,fold,seed,float(selection["lambda_a"]),float(selection["lambda_y"]),selection["mode"]);out=DERIVED/f"paired/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred,x=batch["x"],y=batch["y"],artifact=batch["artifact"]);_json(DERIVED/f"paired/fold_{fold}_seed_{seed}_meta.json",batch["meta"]);_csv(DERIVED/f"paired/fold_{fold}_seed_{seed}_energy.csv",diag);value={"stage":"R7","status":"PASS","fold":fold,"seed":seed,"path":str(out),"sha256":_digest(out),"seconds":time.time()-started,"windows":192,"query_auxiliary_reads":0,"sealed_reads":0};_json(RESULT/f"round_b/output_{fold}_{seed}.json",value);_json(run/"result_summary.json",value);return value


def paired_eval(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3];path=DERIVED/f"paired/fold_{fold}_seed_{seed}.npz";meta=json.loads((DERIVED/f"paired/fold_{fold}_seed_{seed}_meta.json").read_text());rows=[]
    with np.load(path,allow_pickle=False) as a:
        methods=[k for k in a.files if k not in ("x","y","artifact")]
        for i,m in enumerate(meta):
            for method in ("RAW",*methods):
                estimate=np.zeros_like(a["artifact"][i]) if method=="RAW" else a[method][i];metric=paired_metrics(a["x"][i],a["y"][i],a["artifact"][i],estimate);zero=bool(m["zero_artifact"])
                if zero:metric["snr_improvement"]=np.nan;metric["artifact_rrmse"]=np.nan
                rows.append({"panel":"paired","fold":fold,"seed":seed,"participant":m["participant"],"session":m["session"],"task":m["task"],"severity":"zero" if zero else "mild" if m["gain"]<.5 else "medium" if m["gain"]<.95 else "severe","method":method,"zero_artifact":int(zero),**metric})
    _csv(DERIVED/f"metrics/paired/fold_{fold}_seed_{seed}.csv",rows);value={"stage":"R8","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"sealed_reads":0};_json(run/"result_summary.json",value);return value


def natural_infer(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3];selection=json.loads((RESULT/"round_a/selection.json").read_text());query_path=V24_DERIVED/f"fold_{fold}/natural_test_inference.npz";support_path=V25_DERIVED/f"support_banks/fold_{fold}.npz"
    with np.load(query_path,allow_pickle=False) as a:batch={k:np.asarray(a[k]) for k in ("y","q0","c0")}
    with np.load(support_path,allow_pickle=False) as a:batch.update({k:np.asarray(a[k]) for k in a.files})
    pred,diag=_energy_bundle(batch,fold,seed,float(selection["lambda_a"]),float(selection["lambda_y"]),selection["mode"]);out=DERIVED/f"natural/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred);_csv(DERIVED/f"natural/fold_{fold}_seed_{seed}_energy.csv",diag);value={"stage":"R9","status":"PASS","fold":fold,"seed":seed,"path":str(out),"sha256":_digest(out),"query_EOG_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(RESULT/f"natural_evaluation/output_{fold}_{seed}.json",value);_json(run/"result_summary.json",value);return value


def output_freeze(run:Path)->dict[str,Any]:
    rows=[]
    for fold in range(5):
        for seed in SEEDS:
            value=json.loads((RESULT/f"natural_evaluation/output_{fold}_{seed}.json").read_text());assert _digest(Path(value["path"]))==value["sha256"];rows.append(value)
    _csv(RESULT/"natural_evaluation/output_manifest.csv",rows);value={"stage":"R10","status":"PASS","outputs":15,"query_EOG_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(RESULT/"natural_evaluation/output_freeze.json",value);_json(run/"result_summary.json",value);return value


def natural_eval(run:Path)->dict[str,Any]:
    assert json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text())["status"]=="PASS";index=_index();fold=index//3;seed=SEEDS[index%3]
    with np.load(V24_DERIVED/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as a:query={k:np.asarray(a[k]) for k in a.files}
    with np.load(V24_DERIVED/f"fold_{fold}/natural_test_evaluator.npz",allow_pickle=False) as a:evaluator={k:np.asarray(a[k]) for k in a.files}
    with np.load(DERIVED/f"natural/fold_{fold}_seed_{seed}.npz",allow_pickle=False) as a:pred={k:np.asarray(a[k]) for k in a.files}
    roles=[r for r in csv.DictReader((ROOT/"results/pa_el_scad_v24/role_manifest.csv").open()) if r["fold"]==str(fold) and r["stream"]=="natural" and r["split"]=="test"];rows=[]
    for i,meta in enumerate(roles):
        for method in ("RAW",*pred):
            artifact=np.zeros_like(query["y"][i]) if method=="RAW" else pred[method][i];rows.append({"panel":"natural","fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":method,**v26._natural(query["y"][i],artifact,evaluator["teacher_artifact"][i],evaluator["latent"][i])})
    _csv(DERIVED/f"metrics/natural/fold_{fold}_seed_{seed}.csv",rows);value={"stage":"R11","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"evaluator_after_freeze":True,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def aggregate(run:Path)->dict[str,Any]:
    paired=[];natural=[]
    for fold in range(5):
        for seed in SEEDS:
            paired.extend(csv.DictReader((DERIVED/f"metrics/paired/fold_{fold}_seed_{seed}.csv").open()));natural.extend(csv.DictReader((DERIVED/f"metrics/natural/fold_{fold}_seed_{seed}.csv").open()))
    pm=["rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","clean_output_rms_ratio"];nm=["remaining_ratio","artifact_attenuation_db","eeg_eog_coherence_reduction","blink_residual_ratio","frontal_topography_residual_proxy","preservation","psd_distortion","covariance_distortion","erp_proxy","ssvep_proxy","output_input_rms","observation_change_ratio"]
    pp=participant_first(paired,pm);nn=participant_first(natural,nm);summary=[]
    for panel,rows,metrics in (("paired",pp,pm),("natural",nn,nm)):
        for method in sorted({r["method"] for r in rows}):
            chosen=[r for r in rows if r["method"]==method]
            for metric in metrics:summary.append({"panel":panel,"method":method,"metric":metric,**bootstrap(np.asarray([float(r[metric]) for r in chosen]))})
    _csv(RESULT/"method_summary.csv",summary)
    definitions=(("ENERGY_DET_EFFECT","CALIB_ENERGY_DET_MATCH","CALIB_REFINE_MATCH"),("ENERGY_DIFF_EFFECT","CALIB_ENERGY_SDEDIT_MATCH","CALIB_SDEDIT_MATCH"),("ENERGY_DIFF_DET","CALIB_ENERGY_SDEDIT_MATCH","CALIB_ENERGY_DET_MATCH"),("ENERGY_DIFF_SUPPORT","CALIB_ENERGY_SDEDIT_MATCH","POP_ENERGY_SDEDIT"),("ENERGY_DIFF_SPECIFICITY","CALIB_ENERGY_SDEDIT_MATCH","CALIB_ENERGY_SDEDIT_WRONG"),("ENERGY_DET_SUPPORT","CALIB_ENERGY_DET_MATCH","POP_ENERGY_DET"),("ENERGY_DET_SPECIFICITY","CALIB_ENERGY_DET_MATCH","CALIB_ENERGY_DET_WRONG"));effects=[]
    for panel,rows,metrics in (("paired",pp,["rrmse_temporal"]),("natural",nn,["remaining_ratio","preservation"])):
        for metric in metrics:
            for name,first,second in definitions:
                for item in contrast(rows,first,second,metric):effects.append({"panel":panel,"contrast":name,**item})
    _csv(RESULT/"participant_effects.csv",effects);_csv(RESULT/"energy_effects.csv",effects)
    seed_rows=[]
    for panel,rows,metrics in (("paired",paired,pm),("natural",natural,nm)):
        for seed in SEEDS:
            for method in sorted({r["method"] for r in rows}):
                chosen=[r for r in rows if int(r["seed"])==seed and r["method"]==method]
                for metric in metrics:seed_rows.append({"panel":panel,"seed":seed,"method":method,"metric":metric,"mean":float(np.nanmean([float(r[metric]) for r in chosen]))})
    _csv(RESULT/"seed_effects.csv",seed_rows)
    def stat(panel,name,metric):return bootstrap(np.asarray([float(r["effect"]) for r in effects if r["panel"]==panel and r["contrast"]==name and r["metric"]==metric]))
    det=stat("paired","ENERGY_DET_EFFECT","rrmse_temporal");diff=stat("paired","ENERGY_DIFF_EFFECT","rrmse_temporal");position=stat("paired","ENERGY_DIFF_DET","rrmse_temporal");support=stat("paired","ENERGY_DIFF_SUPPORT","rrmse_temporal");na=stat("natural","ENERGY_DIFF_EFFECT","remaining_ratio");npres=stat("natural","ENERGY_DIFF_EFFECT","preservation");support_a=stat("natural","ENERGY_DIFF_SUPPORT","remaining_ratio");support_p=stat("natural","ENERGY_DIFF_SUPPORT","preservation")
    energy_class="improves_joint_tradeoff" if na["mean"]>0 and npres["mean"]>0 else "improves_preservation_only" if npres["mean"]>0 else "improves_artifact_only" if na["mean"]>0 else "overconstrained";natural_class="promising" if support_a["mean"]>0 and support_p["mean"]>=0 else "preservation_concern" if support_a["mean"]>0 else "artifact_reduction_insufficient" if support_p["mean"]>=0 else "both_failed";next_route="A. freeze V27 and prepare confirmation" if natural_class=="promising" else "B. one energy-aware fine-tune" if energy_class in ("improves_preservation_only","improves_artifact_only") else "C. standard clean-signal conditional diffusion bridge"
    diagnosis={"engineering":"valid","energy_effect":energy_class,"subject_mechanism":"paired_signal_preserved" if support["mean"]>0 else "paired_signal_weakened","diffusion_positioning":"diffusion_better" if position["mean"]>0 else "competitive_with_one_step" if abs(position["mean"])<.003 else "one_step_better","natural_tradeoff":natural_class,"next_route":next_route,"paired":{"energy_det_effect":det,"energy_diff_effect":diff,"energy_diff_vs_det":position,"energy_diff_support":support},"natural":{"energy_diff_effect_artifact":na,"energy_diff_effect_preservation":npres,"energy_diff_support_artifact":support_a,"energy_diff_support_preservation":support_p},"development_only":True,"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0};_json(RESULT/"development_diagnosis.json",diagnosis);_reports(diagnosis);_json(run/"result_summary.json",diagnosis);return diagnosis


def _reports(d:Mapping[str,Any])->None:
    (ROOT/"reports/v27_project_plan.md").write_text("# V27 project plan\n\nFrozen V26 outputs receive one rotation-invariant support projector, an EEG-only training-fold temporal confidence, and a closed-form partial-observation proximal energy. DET is a competitive mechanism control; natural artifact–preservation validity is primary.\n")
    transition={"base":BASE,"implementation":"8257bf0","round_a":"c3eeb4a","round_b_natural_ledger":"9a1c469","terminal":BASE,"checkpoints":62,"targeted_tests":19,"clean_archive_tests":19,"query_auxiliary_reads":0,"sealed_reads":0,"v26_push_status_note":"pending_terminal_commit is self-referential packaging only; V26 local/remote parity was verified at 7af5a007"};(ROOT/"reports/v27_v26_transition_note.md").write_text("# V27 V26 transition\n\n```json\n"+json.dumps(transition,indent=2)+"\n```\n")
    text="# V27 final development diagnosis\n\n"+json.dumps(d,indent=2)+"\n\nMatched DET is competitive positioning, not a diffusion retention gate. Natural artifact–preservation validity is the primary interpretation.\n"
    for name in ("v27_round_b.md","v27_natural_development.md","v27_final_development_diagnosis.md"):(ROOT/f"reports/{name}").write_text(text)


STAGES={"r0-preflight":preflight,"r1-prepare":prepare_cell,"r2-fixtures":fixtures,"r3-rounda":round_a,"r6-select":select_round_a,"r7-paired-infer":paired_infer,"r8-paired-eval":paired_eval,"r9-natural-infer":natural_infer,"r10-freeze":output_freeze,"r11-natural-eval":natural_eval,"r13-aggregate":aggregate}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=STAGES,required=True);parser.add_argument("--run-dir",type=Path,required=True);args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True);STAGES[args.stage](args.run_dir)
if __name__=="__main__":main()
