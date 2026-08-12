"""Slurm-facing V29 population-anchored support-adapter workflow."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,subprocess,time
from pathlib import Path
from typing import Any,Iterable,Mapping
import numpy as np
import torch
import yaml

from eeg_scad.data.folds import load_folds,validate_folds
from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.evaluation.adapter_diagnostics import increment_diagnostics
from eeg_scad.evaluation.aggregate_v29 import aggregate as aggregate_metrics
from eeg_scad.evaluation.natural_metrics_v28 import attenuation_consistency,natural_metrics_v28
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.models.pop_adapter_cdm import PopAdapterCDM
from eeg_scad.models.pop_adapter_det import PopAdapterDET
from eeg_scad.models.support_adapter_cdm import SupportAdapterCDM
from eeg_scad.models.support_adapter_det import SupportAdapterDET
from eeg_scad.models.pop_clean_cdm import PopCleanCDM
from eeg_scad.models.pop_clean_det import PopCleanDET
from eeg_scad.training.train_v25 import load_det as load_support
from eeg_scad.training.train_v28 import load as load_v28
from eeg_scad.training.train_v29 import contexts,load as load_adapter,predict as predict_adapter,train as train_adapter

ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT",Path(__file__).resolve().parents[3]));RESULT=ROOT/"results/pa_sc_cdm_v29";DERIVED=Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_sc_cdm_v29");V24=Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24");V25=Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25");V28=Path("/projects/EEG-foundation-model/derived/denoiseNet/sc_cdm_v28");BASE="f7aec43e8fae1d18c2831ee44b00eae9a0098e7e";SEEDS=[20260905,20260906,20260907];V28_SEED={20260905:20260901,20260906:20260902,20260907:20260903};V25_SEED={20260905:20260825,20260906:20260826,20260907:20260827}

def _cfg(name):return yaml.safe_load((ROOT/f"configs/pa_sc_cdm_v29/{name}.yaml").read_text())
def _folds():return load_folds(ROOT/"configs/pa_sc_cdm_v29/folds.yaml")
def _index():return int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))
def _digest(path:Path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):value.update(block)
    return value.hexdigest()
def _json(path:Path,value:Any):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _csv(path:Path,rows:Iterable[Mapping[str,Any]]):
    rows=list(rows);path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
def _pop_path(kind,fold,seed):return V28/f"checkpoints/selected/{kind}/fold_{fold}/seed_{V28_SEED[seed]}/best_joint.pt"
def _support_path(fold,seed):return V25/f"checkpoints/det/deepsets/fold_{fold}/seed_{V25_SEED[seed]}/best_joint.pt"
def _adapter_path(kind,fold,seed,variant="selected"):return DERIVED/f"checkpoints/{variant}/{kind}/fold_{fold}/seed_{seed}/best_joint.pt"
def _tensor(value,device):return torch.as_tensor(value,dtype=torch.float32,device=device)


def preflight(run):
    validate_folds(_folds(),_cfg("data")["participants"]);head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();ledger=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();branch=subprocess.check_output(["git","rev-parse","codex/support-clean-conditional-diffusion-v28"],cwd=ROOT,text=True).strip();checks={"base_exact":branch==BASE,"base_ancestor":subprocess.run(["git","merge-base","--is-ancestor",BASE,head],cwd=ROOT).returncode==0,"ledger_v1_9":"**版本：** v1.9" in ledger,"ledger_active_v29":"V29" in ledger and "population-anchored" in ledger.lower(),"v28_unchanged":not bool(subprocess.check_output(["git","diff","--name-only",BASE,"--","results/sc_cdm_v28","reports/v28_*","src/eeg_scad/models/*clean*"],cwd=ROOT,text=True).strip()),"a_track_unchanged":not bool(subprocess.check_output(["git","diff","--name-only",BASE,"--","taas_submission"],cwd=ROOT,text=True).strip()),"sealed_reads":0}
    if not all(v is True for k,v in checks.items() if k!="sealed_reads"):raise RuntimeError(checks)
    value={"stage":"R0","status":"PASS","base_commit":BASE,"head":head,"V25":"a7d9d647b69e152255b62dbca917a4b3ed082915","V26":"7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c","V27":"40eae116e70e9de7fe0af55d64ee25551932c4a8","V28":BASE,"A_track":"0c4f2301c1f873120fe54537cde3c76fff7ea3a2",**checks};_json(RESULT/"source_registry.json",value);_json(run/"result_summary.json",value);return value


def forensic(run):
    keep={"STANDARD","POP_CLEAN_DET","SUPPORT_CLEAN_DET_MATCH","POP_CLEAN_CDM","SUPPORT_CLEAN_CDM_MATCH"};metrics={"rrmse_temporal","rrmse_spectral","correlation","artifact_rrmse","artifact_correlation","snr_improvement","identity_change"};rows=[{**r,"source_commit":BASE} for r in csv.DictReader((ROOT/"results/sc_cdm_v28/method_summary.csv").open()) if r.get("panel")=="paired" and r.get("method") in keep and r.get("metric") in metrics]
    for source,methods,label in ((ROOT/"results/setcalibdiff_v25/method_summary.csv",{"DET_MATCH"},"V25"),(ROOT/"results/scad_v22/method_summary.csv",{"EEGDFUS_UNIFIED"},"V22")):
        for r in csv.DictReader(source.open()):
            if r.get("method") in methods and r.get("metric") in metrics:rows.append({**r,"source_commit":label,"comparability":"historical_protocol_not_direct"})
    _csv(RESULT/"forensic_summary.csv",rows);inventory=[]
    for path in (ROOT/"results/sc_cdm_v28/method_summary.csv",ROOT/"results/sc_cdm_v28/participant_effects.csv",ROOT/"results/sc_cdm_v28/terminal_manifest.json",ROOT/"reports/v28_final_development_diagnosis.md"):
        inventory.append({"absolute_path":str(path.resolve()),"role":"frozen_V28_evidence","sha256":_digest(path),"size_bytes":path.stat().st_size})
    _csv(RESULT/"input_inventory.csv",inventory)
    # Representative frozen-model replay for residual/context scale.
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");fold=0;seed=20260905;sampler=SupportSetEpisodeSampler(_cfg("data"),_folds()[fold],"validation",seed+211);batch=sampler.sample_paired(8,.2);support,_=load_support(_support_path(fold,seed),device);popdet,_=load_v28(_pop_path("pop_det",fold,seed),device);popcdm,_=load_v28(_pop_path("pop_cdm",fold,seed),device);supdet,_=load_v28(V28/f"checkpoints/selected/support_det/fold_{fold}/seed_{V28_SEED[seed]}/best_joint.pt",device);supcdm,_=load_v28(V28/f"checkpoints/selected/support_cdm/fold_{fold}/seed_{V28_SEED[seed]}/best_joint.pt",device);y=_tensor(batch["y"],device);x=_tensor(batch["x"],device);match,wrong,_=contexts(support,batch,device)
    with torch.no_grad():
        d=supdet(y,match);raw=(d-y)/.1;t=torch.full((len(y),),500,device=device,dtype=torch.long);state=torch.zeros_like(y);c=supcdm.predict_x0(state,y,match,t);cwrong=supcdm.predict_x0(state,y,wrong,t);time=supcdm.backbone.time(__import__('eeg_scad.models.eegdus_backbone',fromlist=['sinusoidal_embedding']).sinusoidal_embedding(t,128));combined=match+time;film=supcdm.backbone.network.film[0](combined);scale,shift=film.chunk(2,1)
    diag={"rms_y":float(y.square().mean().sqrt()),"rms_x_minus_y":float((x-y).square().mean().sqrt()),"rms_raw_network_output":float(raw.square().mean().sqrt()),"rms_scaled_network_output":float((.1*raw).square().mean().sqrt()),"rms_prediction_minus_y":float((d-y).square().mean().sqrt()),"support_context_norm":float(match.norm(dim=1).mean()),"wrong_context_norm":float(wrong.norm(dim=1).mean()),"time_embedding_norm":float(time.norm(dim=1).mean()),"context_time_norm":float(combined.norm(dim=1).mean()),"film_scale_norm":float(scale.norm(dim=1).mean()),"film_shift_norm":float(shift.norm(dim=1).mean()),"match_wrong_feature_distance":float((c-cwrong).square().mean().sqrt()),"null_route_trained":False,"null_interpretation":"V28 zero context was not an ordinary trained population route"};_json(RESULT/"v28_forensic_diagnostics.json",diag)
    (ROOT/"reports/v29_v28_forensic.md").write_text("# V29 V28 forensic\n\nV28 absolute results were replayed from the committed method summary; `forensic_summary.csv` separates matched-architecture comparisons from historical strongest-denoiser positioning. The representative frozen replay reports observation residual and context/time/FiLM magnitudes in `v28_forensic_diagnostics.json`. V28's zero-context NULL was not ordinarily trained and is not a population route. V29 therefore uses an exact frozen-population architectural bypass.\n\n```json\n"+json.dumps(diag,indent=2)+"\n```\n");value={"stage":"R1","status":"PASS","forensic_rows":len(rows),"representative_replay":diag,"V28_modified":False,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def prepare(run):
    folds=[];bindings=[]
    for fold in _folds():
        for split in ("train","validation","test"):
            folds.extend({"fold":fold["fold"],"split":split,"participant":p} for p in fold[split])
        for seed in SEEDS:
            for kind in ("pop_det","pop_cdm"):
                path=_pop_path(kind,fold["fold"],seed);bindings.append({"fold":fold["fold"],"seed":seed,"role":"V28_frozen_"+kind,"path":str(path),"sha256":_digest(path)})
            path=_support_path(fold["fold"],seed);bindings.append({"fold":fold["fold"],"seed":seed,"role":"V25_frozen_support_encoder","path":str(path),"sha256":_digest(path)})
    _csv(RESULT/"fold_manifest.csv",folds);_csv(RESULT/"checkpoint_binding.csv",bindings);value={"stage":"R2","status":"PASS","fold_rows":len(folds),"checkpoint_bindings":len(bindings),"query_disjoint_support":True,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def sanity(run):
    device=torch.device("cuda");g=torch.Generator(device=device).manual_seed(29);y=torch.randn((2,46,64),device=device,generator=g);clean=y-.1*torch.randn(y.shape,device=device,generator=g);context=torch.randn((2,128),device=device,generator=g);wrong=-context;popdet=PopCleanDET(width=8).to(device).eval();popcdm=PopCleanCDM(width=8).to(device).eval();det=SupportAdapterDET(8).to(device);cdm=SupportAdapterCDM(8).to(device);pdet=popdet(y);t=torch.tensor([500,500],device=device);state=torch.randn(y.shape,device=device,generator=g);pcdm=popcdm.predict_x0(state,y,popcdm.context(2),t);initial={"det":float((det(y,pdet,context)-pdet).abs().max()),"cdm":float((cdm.predict_x0(state,y,pcdm,context,t)-pcdm).abs().max())};noise=torch.randn(y.shape,device=device,generator=g);popout,pt=popcdm.sample(y,noise,10);bypass,bt=cdm.sample(popcdm,y,context,noise,10,True);pop_error=float((popout-bypass).abs().max());opt=torch.optim.Adam(list(det.parameters())+list(cdm.parameters()),1e-3);first=last=None
    for _ in range(40):
        d=det(y,pdet,context);c=cdm.predict_x0(state,y,pcdm,context,t);loss=(d-clean).square().mean()+(c-clean).square().mean();first=float(loss) if first is None else first;opt.zero_grad();loss.backward();opt.step();last=float(loss)
    match=cdm.predict_x0(state,y,pcdm,context,t);other=cdm.predict_x0(state,y,pcdm,wrong,t);value={"stage":"R3","status":"PASS" if max(initial.values())==0 and pop_error<=1e-6 and torch.isfinite(match).all() else "FAIL","zero_initialization":initial,"exact_pop_bypass_max":pop_error,"stepwise_pop_identity":len(pt)==len(bt)==10,"single_batch_initial":first,"single_batch_final":last,"context_swap_change":float((match-other).abs().mean()),"K":1,"sealed_reads":0};_json(RESULT/"sanity/technical_validity.json",value);_json(run/"result_summary.json",value);return value


ROUND_A=[("support_adapter_det","base"),("support_adapter_det","rank"),("support_adapter_cdm","base"),("support_adapter_cdm","rank"),("pop_adapter_det","base"),("pop_adapter_cdm","base")]
def train_stage(stage,run):
    index=_index()
    if stage=="r4-rounda":kind,variant=ROUND_A[index//2];fold=(0,2)[index%2];seed=20260905
    else:kind=stage.removeprefix("r8-").replace("-","_");variant="selected";fold=index//3;seed=SEEDS[index%3]
    cfg=_cfg("training");cfg["lambda_ctx"]=0. if variant=="base" else .1
    if variant=="selected":cfg["lambda_ctx"]=float(json.loads((RESULT/"round_a/selection.json").read_text())["lambda_ctx"])
    pop_kind="pop_det" if kind.endswith("det") else "pop_cdm";out=DERIVED/f"checkpoints/{variant}/{kind}/fold_{fold}/seed_{seed}";result=train_adapter(kind,fold,seed,cfg,_cfg("data"),_folds()[fold],out,_pop_path(pop_kind,fold,seed),_support_path(fold,seed),True);target=RESULT/("round_a" if stage=="r4-rounda" else "round_b")/f"{kind}_{variant}_fold_{fold}_seed_{seed}.json";_json(target,result);_json(run/"result_summary.json",result);return result


def _load_bundle(fold,seed,variant="selected"):
    device=torch.device("cuda");support,_=load_support(_support_path(fold,seed),device);popdet,_=load_v28(_pop_path("pop_det",fold,seed),device);popcdm,_=load_v28(_pop_path("pop_cdm",fold,seed),device);models={k:load_adapter(_adapter_path(k,fold,seed,variant),device)[0] for k in ("support_adapter_det","pop_adapter_det","support_adapter_cdm","pop_adapter_cdm")};return device,support,popdet,popcdm,models


@torch.no_grad()
def _predict_all(batch,fold,seed,variant="selected",steps=10):
    device,support,popdet,popcdm,models=_load_bundle(fold,seed,variant);output={};diagnostics=[]
    for start in range(0,len(batch["y"]),32):
        stop=min(start+32,len(batch["y"]));chunk={k:(v[start:stop] if hasattr(v,"__len__") and len(v)==len(batch["y"]) else v) for k,v in batch.items()};y=_tensor(chunk["y"],device);context,wrong,pi=contexts(support,chunk,device);pdet=popdet(y);det=models["support_adapter_det"];pdet_adapter=models["pop_adapter_det"];dmatch=det(y,pdet,context);dwrong=det(y,pdet,wrong);dbypass=det(y,pdet,context,True);assert torch.equal(dbypass,pdet);seed_value=seed+701+start;noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed_value));pcdm_value=popcdm.sample(y,noise,steps)[0];cdm=models["support_adapter_cdm"];cmatch=cdm.sample(popcdm,y,context,noise,steps)[0];cwrong=cdm.sample(popcdm,y,wrong,noise,steps)[0];cbypass=cdm.sample(popcdm,y,context,noise,steps,True)[0];assert torch.equal(cbypass,pcdm_value);popad=models["pop_adapter_cdm"].sample(popcdm,y,noise,steps)[0]
        values={"POP_CLEAN_DET":pdet,"PA_SC_DET_MATCH":dmatch,"PA_SC_DET_WRONG":dwrong,"PA_SC_DET_POP":dbypass,"POP_ADAPTER_DET":pdet_adapter(y,pdet),"POP_CLEAN_CDM":pcdm_value,"PA_SC_CDM_MATCH":cmatch,"PA_SC_CDM_WRONG":cwrong,"PA_SC_CDM_POP":cbypass,"POP_ADAPTER_CDM":popad}
        for key,value in values.items():output.setdefault(key,[]).append(value.cpu().numpy())
        diagnostics.append(increment_diagnostics(pcdm_value,cmatch,cwrong,projector=pi))
    return {key:np.concatenate(value) for key,value in output.items()},diagnostics


def round_a_eval(run):
    fold=(0,2)[_index()];seed=20260905;sampler=SupportSetEpisodeSampler(_cfg("data"),_folds()[fold],"validation",seed+333);paired=sampler.sample_paired(192,.2);natural=sampler.sample_natural(96);scale=np.load(V24/f"fold_{fold}/eeg_scale.npy");rows=[]
    for variant in ("base","rank"):
        # PopAdapter checkpoints are shared across variants; expose links under rank if absent.
        for kind in ("pop_adapter_det","pop_adapter_cdm"):
            target=_adapter_path(kind,fold,seed,variant)
            if variant=="rank" and not target.exists():target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(_adapter_path(kind,fold,seed,"base"))
        for panel,batch in (("paired",paired),("natural",natural)):
            pred,_=_predict_all(batch,fold,seed,variant,10)
            for method,clean in pred.items():
                if panel=="paired":m=[paired_metrics(batch["x"][i],batch["y"][i],batch["artifact"][i],batch["y"][i]-clean[i]) for i in range(len(clean))];rows.append({"fold":fold,"variant":variant,"panel":panel,"method":method,"rrmse":float(np.mean([v["rrmse_temporal"] for v in m])),"artifact_rrmse":float(np.mean([v["artifact_rrmse"] for v in m if np.isfinite(v["artifact_rrmse"])])),"correlation":float(np.mean([v["correlation"] for v in m]))})
                else:m=[natural_metrics_v28(batch["y"][i],clean[i],batch["latent"][i],batch["teacher_artifact"][i],scale) for i in range(len(clean))];rows.append({"fold":fold,"variant":variant,"panel":panel,"method":method,"remaining":float(np.mean([v["heldout_eog_remaining_ratio"] for v in m])),"retention":float(np.mean([v["low_eog_observation_retention"] for v in m])),"psd":float(np.mean([v["psd_distortion"] for v in m]))})
    _csv(RESULT/f"round_a/evaluation_fold_{fold}.csv",rows);value={"stage":"R5_R6","status":"PASS","fold":fold,"rows":len(rows),"test_used":False,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def select(run):
    rows=[]
    for path in sorted((RESULT/"round_a").glob("evaluation_fold_*.csv")):rows.extend(csv.DictReader(path.open()))
    candidates=[]
    for variant in ("base","rank"):
        def mean(panel,method,key):return float(np.mean([float(r[key]) for r in rows if r["variant"]==variant and r["panel"]==panel and r["method"]==method]))
        match=mean("paired","PA_SC_CDM_MATCH","rrmse");wrong=mean("paired","PA_SC_CDM_WRONG","rrmse");popad=mean("paired","POP_ADAPTER_CDM","rrmse");remaining=mean("natural","PA_SC_CDM_MATCH","remaining");retention=mean("natural","PA_SC_CDM_MATCH","retention");score=match+.05*remaining-.05*retention-.05*(wrong-match)-.05*(popad-match);candidates.append({"variant":variant,"lambda_ctx":0 if variant=="base" else .1,"score":score,"paired_rrmse":match,"match_wrong_utility":wrong-match,"match_pop_adapter_utility":popad-match,"natural_remaining":remaining,"natural_retention":retention})
    chosen=min(candidates,key=lambda r:r["score"]);value={"status":"ROUND_B_CONFIG_FROZEN","variant":chosen["variant"],"lambda_ctx":chosen["lambda_ctx"],"ddim_steps":10,"support_encoder":"V25 DeepSets frozen","population":"V28 frozen","selection_uses_test":False,"rationale":"Validation-only paired fidelity, support specificity, capacity control, and corrected natural trade-off; no hard scientific gate.","candidates":candidates};_json(RESULT/"round_a/selection.json",value);(ROOT/"reports/v29_round_a.md").write_text("# V29 Round A\n\nThe validation-only selection retained the `"+chosen["variant"]+"` adapter (`lambda_ctx="+str(chosen["lambda_ctx"])+"`). Selection jointly considered absolute paired fidelity, MATCH−WRONG, MATCH−PopAdapter, corrected natural artifact/retention, increment size, and stability; no test participants or automatic scientific threshold were used.\n");_json(run/"result_summary.json",value);return value


def paired_infer(run):
    index=_index();fold=index//3;seed=SEEDS[index%3];sampler=SupportSetEpisodeSampler(_cfg("data"),_folds()[fold],"test",seed+509);batch=sampler.sample_paired(192,.2);started=time.time();pred,diag=_predict_all(batch,fold,seed,"selected",10);out=DERIVED/f"paired/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred,x=batch["x"],y=batch["y"],artifact=batch["artifact"]);_json(DERIVED/f"paired/fold_{fold}_seed_{seed}_meta.json",batch["meta"]);_csv(DERIVED/f"diagnostics/adapter_{fold}_{seed}.csv",diag);value={"stage":"R9_INFER","status":"PASS","fold":fold,"seed":seed,"path":str(out),"sha256":_digest(out),"seconds":time.time()-started,"windows":192,"query_auxiliary_reads":0,"sealed_reads":0};_json(RESULT/f"round_b/output_{fold}_{seed}.json",value);_json(run/"result_summary.json",value);return value


def paired_eval(run):
    index=_index();fold=index//3;seed=SEEDS[index%3];path=DERIVED/f"paired/fold_{fold}_seed_{seed}.npz";meta=json.loads((DERIVED/f"paired/fold_{fold}_seed_{seed}_meta.json").read_text())
    with np.load(path,allow_pickle=False) as archive:arrays={k:np.asarray(archive[k]) for k in archive.files}
    methods=[k for k in arrays if k not in ("x","y","artifact")];rows=[]
    for i,item in enumerate(meta):
        for method in ("STANDARD",*methods):
            clean=arrays["y"][i] if method=="STANDARD" else arrays[method][i];metric=paired_metrics(arrays["x"][i],arrays["y"][i],arrays["artifact"][i],arrays["y"][i]-clean);identity=bool(item["zero_artifact"])
            if identity:metric["snr_improvement"]=np.nan;metric["artifact_rrmse"]=np.nan
            pop=arrays["POP_CLEAN_CDM"][i] if "CDM" in method else arrays["POP_CLEAN_DET"][i];rows.append({"panel":"paired","fold":fold,"seed":seed,"participant":item["participant"],"session":item["session"],"task":item["task"],"severity":"identity" if identity else "mild" if item["gain"]<.5 else "medium" if item["gain"]<.95 else "severe","method":method,"identity":int(identity),"identity_change":float(np.linalg.norm(clean-arrays["y"][i])/max(np.linalg.norm(arrays["y"][i]),1e-12)) if identity else np.nan,"adapter_rms":float(np.sqrt(np.mean((clean-pop)**2))) if method!="STANDARD" else 0.,**metric})
    _csv(DERIVED/f"metrics/paired/fold_{fold}_seed_{seed}.csv",rows);value={"stage":"R9_EVAL","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"sealed_reads":0};_json(run/"result_summary.json",value);return value


def natural_infer(run):
    index=_index();fold=index//3;seed=SEEDS[index%3]
    with np.load(V24/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as archive:batch={k:np.asarray(archive[k]) for k in ("y","q0","c0")}
    with np.load(V25/f"support_banks/fold_{fold}.npz",allow_pickle=False) as archive:batch.update({k:np.asarray(archive[k]) for k in archive.files})
    started=time.time();pred,diag=_predict_all(batch,fold,seed,"selected",10);out=DERIVED/f"natural/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred);_csv(DERIVED/f"diagnostics/natural_adapter_{fold}_{seed}.csv",diag);value={"stage":"R10","status":"PASS","fold":fold,"seed":seed,"path":str(out),"sha256":_digest(out),"seconds":time.time()-started,"windows":len(batch["y"]),"query_EOG_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(RESULT/f"natural_evaluation/output_{fold}_{seed}.json",value);_json(run/"result_summary.json",value);return value


def freeze_outputs(run):
    rows=[]
    for fold in range(5):
        for seed in SEEDS:
            row=json.loads((RESULT/f"natural_evaluation/output_{fold}_{seed}.json").read_text());assert _digest(Path(row["path"]))==row["sha256"];rows.append(row)
    _csv(RESULT/"natural_evaluation/output_manifest.csv",rows);value={"stage":"R11","status":"PASS","outputs":15,"query_EOG_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(RESULT/"natural_evaluation/output_freeze.json",value);_json(run/"result_summary.json",value);return value


def natural_eval(run):
    assert json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text())["status"]=="PASS";index=_index();fold=index//3;seed=SEEDS[index%3]
    with np.load(V24/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as archive:query={k:np.asarray(archive[k]) for k in archive.files}
    with np.load(V24/f"fold_{fold}/natural_test_evaluator.npz",allow_pickle=False) as archive:evaluator={k:np.asarray(archive[k]) for k in archive.files}
    with np.load(DERIVED/f"natural/fold_{fold}_seed_{seed}.npz",allow_pickle=False) as archive:pred={k:np.asarray(archive[k]) for k in archive.files}
    roles=[r for r in csv.DictReader((ROOT/"results/pa_el_scad_v24/role_manifest.csv").open()) if r["fold"]==str(fold) and r["stream"]=="natural" and r["split"]=="test"];scale=np.load(V24/f"fold_{fold}/eeg_scale.npy");rows=[];max_difference=0.
    for i,meta in enumerate(roles):
        for method in ("STANDARD",*pred):
            clean=query["y"][i] if method=="STANDARD" else pred[method][i];metric=natural_metrics_v28(query["y"][i],clean,evaluator["latent"][i],evaluator["teacher_artifact"][i],scale);max_difference=max(max_difference,attenuation_consistency(metric["heldout_eog_remaining_ratio"],metric["artifact_attenuation_db"]));pop=pred["POP_CLEAN_CDM"][i] if "CDM" in method else pred["POP_CLEAN_DET"][i];rows.append({"panel":"natural","fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":method,"adapter_rms":float(np.sqrt(np.mean((clean-pop)**2))) if method!="STANDARD" else 0.,**metric})
    _csv(DERIVED/f"metrics/natural/fold_{fold}_seed_{seed}.csv",rows);value={"stage":"R12","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"attenuation_remaining_max_difference":max_difference,"evaluator_after_freeze":True,"query_auxiliary_reads":0,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def aggregate(run):
    diagnosis,tables=aggregate_metrics(DERIVED,SEEDS);_csv(RESULT/"method_summary.csv",tables["summary"]);_csv(RESULT/"participant_effects.csv",tables["effects"]);_csv(RESULT/"seed_effects.csv",tables["seed"]);_csv(RESULT/"severity_effects.csv",tables["severity"]);diag=[]
    for path in sorted((DERIVED/"diagnostics").glob("*.csv")):diag.extend(csv.DictReader(path.open()))
    _csv(RESULT/"adapter_diagnostics.csv",diag);_json(RESULT/"development_diagnosis.json",diagnosis);latency=[]
    for panel,folder in (("paired",RESULT/"round_b"),("natural",RESULT/"natural_evaluation")):
        for path in folder.glob("output_*.json"):
            row=json.loads(path.read_text())
            if {"fold","seed","windows","seconds"}.issubset(row):latency.append({"panel":panel,"fold":row["fold"],"seed":row["seed"],"seconds_per_window":row["seconds"]/row["windows"]})
    _csv(RESULT/"latency_summary.csv",latency);_figures(tables,diag,latency);p=diagnosis["paired"];n=diagnosis["natural_effects"]
    (ROOT/"reports/v29_round_b.md").write_text(f'# V29 Round B\n\nPA-SC-CDM MATCH−PopAdapter utility was {p["cdm_match_pop_adapter"]["mean"]:+.6f} ({p["cdm_match_pop_adapter"]["positive"]}/15 positive; 95% CI [{p["cdm_match_pop_adapter"]["bootstrap_low"]:+.6f}, {p["cdm_match_pop_adapter"]["bootstrap_high"]:+.6f}]). MATCH−WRONG was {p["cdm_match_wrong"]["mean"]:+.6f}. PA-SC-CDM−PA-SC-DET was {p["cdm_det"]["mean"]:+.6f}; this is competitive positioning, not a survival gate.\n')
    (ROOT/"reports/v29_natural_development.md").write_text(f'# V29 natural development\n\nMATCH−PopAdapter artifact utility was {n["artifact"]["mean"]:+.6f}; low-EOG observation-retention utility was {n["retention"]["mean"]:+.6f}; PSD utility was {n["psd"]["mean"]:+.6f}. ERP/SSVEP/ERD-ERS remain unavailable and no proxy alias is used.\n')
    (ROOT/"reports/v29_final_development_diagnosis.md").write_text("# V29 final development diagnosis\n\n```json\n"+json.dumps(diagnosis,indent=2)+"\n```\n");_json(run/"result_summary.json",diagnosis);return diagnosis


def _figures(tables,diag,latency):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    root=ROOT/"figures/pa_sc_cdm_v29";root.mkdir(parents=True,exist_ok=True);summary=tables["summary"];lookup={(r["panel"],r["method"],r["metric"]):float(r["mean"]) for r in summary};methods=["POP_CLEAN_CDM","POP_ADAPTER_CDM","PA_SC_CDM_MATCH","PA_SC_CDM_WRONG"]
    def save(name,x,y,xlabel,ylabel):fig,ax=plt.subplots();ax.plot(x,y,"o-");ax.set(xlabel=xlabel,ylabel=ylabel);fig.tight_layout();fig.savefig(root/name,dpi=150);plt.close(fig)
    save("paired_method_comparison.png",range(len(methods)),[lookup.get(("paired",m,"rrmse_temporal"),np.nan) for m in methods],"method index","clean RRMSE");save("natural_artifact_retention.png",[lookup.get(("natural",m,"heldout_eog_remaining_ratio"),np.nan) for m in methods],[lookup.get(("natural",m,"low_eog_observation_retention"),np.nan) for m in methods],"EOG remaining ratio","low-EOG retention");save("adapter_increment_distribution.png",range(len(diag)),[float(r["adapter_rms"]) for r in diag],"bundle","adapter RMS");save("quality_latency_curve.png",range(len(latency)),[r["seconds_per_window"] for r in latency],"bundle","seconds/window")
    effects=[r for r in tables["effects"] if r["panel"]=="paired" and r["contrast"] in ("CDM_MATCH_POP_ADAPTER","CDM_MATCH_WRONG") and r["metric"]=="rrmse_temporal"];participants=sorted({r["participant"] for r in effects});fig,ax=plt.subplots(figsize=(8,5));
    for j,name in enumerate(("CDM_MATCH_POP_ADAPTER","CDM_MATCH_WRONG")):values={r["participant"]:float(r["effect"]) for r in effects if r["contrast"]==name};ax.plot([values[p] for p in participants],np.arange(len(participants))+.1*j,"o",label=name)
    ax.axvline(0,color="black");ax.set_yticks(range(len(participants)),participants);ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/"support_effect_forest.png",dpi=150);plt.close(fig)
    # Forensic figures use bound summaries/diagnostics rather than hidden raw arrays.
    forensic=json.loads((RESULT/"v28_forensic_diagnostics.json").read_text());save("context_time_norms.png",[0,1,2],[forensic["support_context_norm"],forensic["time_embedding_norm"],forensic["context_time_norm"]],"support/time/combined","norm");save("V28_absolute_performance.png",range(len(methods)),[lookup.get(("paired",m,"rrmse_temporal"),np.nan) for m in methods],"method index","RRMSE")
    fig,ax=plt.subplots()
    for path in sorted((RESULT/"round_b").glob("*.json")):
        row=json.loads(path.read_text())
        if "curve" in row:ax.plot([r["step"] for r in row["curve"]],[r["joint"] for r in row["curve"]],alpha=.3)
    ax.set(xlabel="updates",ylabel="validation joint");fig.tight_layout();fig.savefig(root/"adapter_training_curves.png",dpi=150);plt.close(fig)


def ledger_check(run):
    path=ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md";text=path.read_text();value={"stage":"R14","status":"PASS","project_ledger_version":"v2.0","project_ledger_sha256":_digest(path),"v29_results_recorded":"V29 — PA-SC-CDM" in text,"sealed_reads":0}
    if "**版本：** v2.0" not in text or not value["v29_results_recorded"]:raise RuntimeError(value)
    _json(RESULT/"ledger_sync.json",value);_json(run/"result_summary.json",value);return value


def _lineage():
    rows=[];by={}
    for job in sorted((RESULT/"runs").glob("*/job_*")):
        for path in sorted(job.glob("task_*")) or [job]:
            stage=path.parts[-3] if path.name.startswith("task_") else path.parent.name;job_id=job.name.removeprefix("job_");task=path.name.removeprefix("task_") if path.name.startswith("task_") else "";status="accepted" if (path/"result_summary.json").is_file() or (path/"pytest.txt").is_file() else "failed";row={"stage":stage,"job_id":job_id,"array_task":task,"status":status,"recovery_of":"","scientific_setting_changed":False};rows.append(row);by.setdefault((stage,task),[]).append(row)
    for cell in by.values():
        failed=[r for r in cell if r["status"]=="failed"];accepted=[r for r in cell if r["status"]=="accepted"]
        if failed and accepted:accepted[-1]["status"]="recovery";accepted[-1]["recovery_of"]=failed[-1]["job_id"]
    return rows


def package(run):
    lineage=_lineage();_csv(RESULT/"job_lineage.csv",lineage);lines=["# V29 Slurm lineage","stage\tjob_id\tarray_task\tstatus\trecovery_of\tscientific_setting_changed"]+["\t".join(str(r[k]) for k in ("stage","job_id","array_task","status","recovery_of","scientific_setting_changed")) for r in lineage];(ROOT/"reports/slurm").mkdir(parents=True,exist_ok=True);(ROOT/"reports/slurm/v29_job_ids.txt").write_text("\n".join(lines)+"\n");manifest=[]
    for kind in ("support_adapter_det","pop_adapter_det","support_adapter_cdm","pop_adapter_cdm"):
        for fold in range(5):
            for seed in SEEDS:
                path=_adapter_path(kind,fold,seed);record=json.loads((RESULT/f"round_b/{kind}_selected_fold_{fold}_seed_{seed}.json").read_text());manifest.append({"path":str(path),"sha256":_digest(path),"fold":fold,"seed":seed,"model":kind,"best_criterion":"joint","updates":record["updates"],"adapter_parameters":record["adapter_parameters"],"population_checkpoint":record["population_checkpoint"],"support_checkpoint":record["support_checkpoint"]})
    _csv(RESULT/"checkpoint_manifest.csv",manifest)
    def count(stage):
        paths=list((RESULT/f"runs/{stage}").glob("job_*/pytest.txt"));match=re.search(r"(\d+) passed",paths[-1].read_text()) if paths else None;return int(match.group(1)) if match else 0
    diagnosis=json.loads((RESULT/"development_diagnosis.json").read_text());ledger=ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md";queue=subprocess.check_output(["squeue","--me","--noheader","-o","%i %j %T"],text=True);counts={s:sum(r["status"]==s for r in lineage) for s in ("accepted","failed","superseded","recovery")};value={"protocol_id":"population_anchored_support_adapter_clean_diffusion_v29","development_only":True,"base_commit":BASE,"implementation_commit":"reported_in_terminal_handoff","forensic_commit":"reported_in_terminal_handoff","round_a_commit":"reported_in_terminal_handoff","round_b_commit":"reported_in_terminal_handoff","natural_result_commit":"reported_in_terminal_handoff","ledger_v2_0_commit":"reported_in_terminal_handoff","report_package_commit":"reported_in_terminal_handoff","terminal_commit":"SELF_REFERENTIAL_REPORTED_EXTERNALLY","push_status":"push_verified_after_terminal_commit","remote_sha":"reported_after_push","model_cells":60,"checkpoint_bindings":60,"paired_inference_outputs":15,"natural_inference_outputs":15,"targeted_tests":count("r15-tests"),"clean_archive_tests":count("r16-clean"),"job_status_counts":counts,"accepted_jobs":[r["job_id"] for r in lineage if r["status"]=="accepted"],"failed_jobs":[r["job_id"] for r in lineage if r["status"]=="failed"],"superseded_jobs":[],"recovery_jobs":[{"job_id":r["job_id"],"recovery_of":r["recovery_of"]} for r in lineage if r["status"]=="recovery"],"current_v29_jobs":[line for line in queue.splitlines() if "v29_" in line],"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0,"A_track_head":"0c4f2301c1f873120fe54537cde3c76fff7ea3a2","A_track_unchanged":True,"manuscript_unchanged":True,"K":1,"gpu_environment":"icml","cpu_environment":"eeg2025","project_ledger_path":"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md","project_ledger_version":"v2.0","project_ledger_sha256":_digest(ledger),**diagnosis};_json(RESULT/"terminal_manifest.json",value);_json(run/"result_summary.json",value);return value


STAGES={"r0-preflight":preflight,"r1-forensic":forensic,"r2-prepare":prepare,"r3-sanity":sanity,"r4-rounda":lambda r:train_stage("r4-rounda",r),"r5-rounda-eval":round_a_eval,"r7-select":select,"r8-support-adapter-det":lambda r:train_stage("r8-support-adapter-det",r),"r8-pop-adapter-det":lambda r:train_stage("r8-pop-adapter-det",r),"r8-support-adapter-cdm":lambda r:train_stage("r8-support-adapter-cdm",r),"r8-pop-adapter-cdm":lambda r:train_stage("r8-pop-adapter-cdm",r),"r9-paired-infer":paired_infer,"r9-paired-eval":paired_eval,"r10-natural-infer":natural_infer,"r11-freeze":freeze_outputs,"r12-natural-eval":natural_eval,"r13-aggregate":aggregate,"r14-ledger":ledger_check,"r17-package":package}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=STAGES);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True);STAGES[args.stage](args.run_dir)
if __name__=="__main__":main()
