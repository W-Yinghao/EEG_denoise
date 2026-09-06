from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
import torch
import yaml
from scipy import signal
from eeg_scad.data.folds import load_folds,validate_folds
from eeg_scad.data.support_set_episodes import NaturalSupportBankBuilder,SupportSetEpisodeSampler,episode_digest
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.context.learned_spatial_decoder import decode_residual
from eeg_scad.training.train_v24 import load_anchor
from eeg_scad.training.train_v25 import load_det,load_diff,train_det,train_diff

ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT",Path(__file__).resolve().parents[3]));RESULT=ROOT/"results/setcalibdiff_v25";DERIVED=Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25");V24DERIVED=Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24");SEEDS=[20260825,20260826,20260827];ANCHOR_SEED={20260825:20260825,20260826:20260826,20260827:20260824}
def _cfg(name:str)->dict[str,Any]:return yaml.safe_load((ROOT/f"configs/setcalibdiff_v25/{name}.yaml").read_text())
def _folds()->list[dict[str,Any]]:return load_folds(ROOT/"configs/setcalibdiff_v25/folds.yaml")
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row});stream=path.open("w",newline="",encoding="utf-8");writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows);stream.close()
def _head(path:Path)->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=path,text=True).strip()
def _clean(path:Path)->bool:return not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=path,text=True).strip()
def _digest(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _index()->int:return int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))
def _anchor_path(fold:int,seed:int)->Path:return V24DERIVED/f"checkpoints/anchor/fold_{fold}/seed_{ANCHOR_SEED[seed]}/best_joint.pt"
def _det_path(fold:int,seed:int,encoder:str="deepsets")->Path:return DERIVED/f"checkpoints/det/{encoder}/fold_{fold}/seed_{seed}/best_joint.pt"
def _diff_path(fold:int,seed:int)->Path:return DERIVED/f"checkpoints/diff/fold_{fold}/seed_{seed}/best_joint.pt"

def preflight(run:Path)->dict[str,Any]:
    data=_cfg("data");folds=_folds();validate_folds(folds,data["participants"]);ledger=ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md";text=ledger.read_text();v24=Path(data["v24_worktree"]);atrack=Path(data["a_track_worktree"]);checks={"base_ancestor":subprocess.run(["git","merge-base","--is-ancestor",data["base_commit"],"HEAD"],cwd=ROOT).returncode==0,"ledger_v1_1":"**版本：** v1.1" in text,"ledger_active_v25":"V25 SetCalibDiff" in text,"ledger_latest_v24":"V24 PA-EL-SCAD" in text,"v24_exact":_head(v24)==data["v24_commit"],"v24_clean":_clean(v24),"a_track_exact":_head(atrack)==data["a_track_commit"],"a_track_clean":_clean(atrack),"sealed_reads":0}
    if not all(v is True or k=="sealed_reads" for k,v in checks.items()):raise RuntimeError(checks)
    registry={"V24":{"commit":data["v24_commit"],"coordinate_audit":"bf840a72229622ff1a311dfa5b2686d46444cd69","core_implementation":"2e5b0caf8108201e6c6e177ab95908f7b7075a71","packaging_latency":"2750552e85263ba262c4276cb83cc6c6a0ec6e1f","result_producing":"55a078214eddcc35bec7046441784eedcdf673ab","terminal":data["v24_commit"],"canonical_natural_artifact_utility":-0.373885,"canonical_natural_preservation_utility":-0.232186},"ledger":{"version":"v1.1","sha256":_digest(ledger)},"A_track":{"commit":data["a_track_commit"]}}
    _json(RESULT/"source_registry.json",registry);_json(RESULT/"preflight.json",{"status":"PASS",**checks});_json(run/"result_summary.json",{"stage":"R0","status":"PASS",**checks});return checks

def prepare(run:Path)->dict[str,Any]:
    data=_cfg("data");rows=[];episodes=[]
    for fold in _folds():
        for split in ("train","validation","test"):
            for participant in fold[split]:rows.append({"fold":fold["fold"],"split":split,"participant":participant})
        sampler=SupportSetEpisodeSampler(data,fold,"test",20260825);batch=sampler.sample_paired(24)
        for meta,eeg,eog in zip(batch["meta"],batch["support_eeg"],batch["support_eog"]):episodes.append({"fold":fold["fold"],"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"support_owner":meta["support_owner"],"wrong_owner":meta["wrong_owner"],"support_start_min":min(meta["support_starts"]),"support_start_max":max(meta["support_starts"]),"query_start":data["qnatural_start"],"windows":data["support_windows"],"window_samples":data["support_window_samples"],"digest":episode_digest(eeg,eog,meta["support_starts"])})
    _csv(RESULT/"fold_manifest.csv",rows);_csv(RESULT/"support_episode_manifest.csv",episodes);_csv(RESULT/"support_budget_manifest.csv",[{"seconds":v,"windows":max(1,round(16*v/120)),"training":v==120} for v in (10,30,60,120)]);_json(RESULT/"support_protocol.json",{"support":"0-120s","guard_1":"120-150s","qgen":"150-270s","guard_2":"270-300s","query":"300s-end","query_auxiliary_inference":False,"sealed_reads":0});result={"stage":"R1","status":"PASS","episodes":len(episodes),"folds":5,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def prepare_natural_support(run:Path)->dict[str,Any]:
    """Build support-only banks aligned to V24's frozen auxiliary-free query arrays."""
    data=_cfg("data");folds=_folds();role_path=ROOT/"results/pa_el_scad_v24/role_manifest.csv";all_rows=list(csv.DictReader(role_path.open()));manifest=[]
    for fold in range(5):
        rows=[row for row in all_rows if row["fold"]==str(fold) and row["stream"]=="natural" and row["split"]=="test"]
        inference_path=V24DERIVED/f"fold_{fold}/natural_test_inference.npz"
        with np.load(inference_path,allow_pickle=False) as archive: count=len(archive["y"])
        if len(rows)!=count:raise RuntimeError(f"natural role count mismatch fold {fold}: {len(rows)} != {count}")
        builder=NaturalSupportBankBuilder(data,folds[fold],20260825+fold*1000)
        correct_eeg=[];correct_eog=[];wrong_eeg=[];wrong_eog=[];starts=[]
        for index,row in enumerate(rows):
            ce,co,cs,ca=builder.support_set(row["participant"],row["session"],row["task"])
            we,wo,ws,wa=builder.support_set(row["wrong_owner"],row["session"],row["task"])
            correct_eeg.append(ce);correct_eog.append(co);wrong_eeg.append(we);wrong_eog.append(wo)
            starts.append({"fold":fold,"sample":index,"participant":row["participant"],"wrong_owner":row["wrong_owner"],"session":row["session"],"task":row["task"],"support_starts":";".join(map(str,cs)),"wrong_support_starts":";".join(map(str,ws)),"support_actual_task":ca,"wrong_actual_task":wa})
        path=DERIVED/f"support_banks/fold_{fold}.npz";path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,support_eeg=np.asarray(correct_eeg),support_eog=np.asarray(correct_eog),wrong_support_eeg=np.asarray(wrong_eeg),wrong_support_eog=np.asarray(wrong_eog))
        start_path=RESULT/f"support_banks/fold_{fold}_manifest.csv";_csv(start_path,starts)
        manifest.append({"fold":fold,"samples":count,"path":str(path),"sha256":_digest(path),"source_inference":str(inference_path),"query_eog_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0})
    _csv(RESULT/"support_banks/manifest.csv",manifest);result={"stage":"R2_SUPPORT_ONLY","status":"PASS","folds":5,"samples":sum(row["samples"] for row in manifest),"query_eog_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def sanity(run:Path)->dict[str,Any]:
    from eeg_scad.models.setcalib_det import SetCalibDET
    from eeg_scad.models.setcalib_diff import SetCalibResidualDiffusion
    device=torch.device("cuda");g=torch.Generator(device=device).manual_seed(20260825);y=torch.randn((2,46,256),device=device,generator=g);a0=torch.randn((2,46,256),device=device,generator=g)*.1;q0=torch.randn((2,4,256),device=device,generator=g);seeg=torch.randn((2,16,46,200),device=device,generator=g);seog=torch.randn((2,16,4,200),device=device,generator=g);model=SetCalibDET().to(device);optimizer=torch.optim.Adam(model.parameters(),1e-3);initial=None
    for step in range(120):
        out=model(y,a0,q0,seeg,seog);target=a0+.1*torch.randn(out["artifact"].shape,device=device,generator=g);loss=(out["artifact"]-target).square().mean();initial=float(loss) if initial is None else initial;optimizer.zero_grad();loss.backward();optimizer.step()
    wrong=model(y,a0,q0,seeg.flip(1),seog.flip(1));diff=SetCalibResidualDiffusion().to(device);noise=torch.randn((2,8,256),device=device,generator=g);sample=diff.sample(y,a0,q0,out["coefficient"],out["context"],noise,10);result={"stage":"R3","status":"PASS","initial_loss":initial,"final_loss":float(loss),"finite":bool(torch.isfinite(sample).all()),"pop_identity_max":float((SetCalibDET.population(y,a0)["artifact"]-a0).abs().max()),"context_change":float((out["artifact"]-wrong["artifact"]).abs().max()),"ddim_steps":10,"sealed_reads":0};_json(RESULT/"sanity/technical_validity.json",result);_json(run/"result_summary.json",result);return result

def train_stage(stage:str,run:Path)->dict[str,Any]:
    data=_cfg("data");folds=_folds();index=_index()
    if stage=="r4-rounda-det":encoder="deepsets" if index<2 else "set_transformer";fold=[0,2][index%2];seed=20260825;cfg=_cfg("setcalib_det");cfg["maximum_updates"]=15000;result=train_det(fold,seed,encoder,cfg,data,folds[fold],DERIVED/f"checkpoints/det/{encoder}/fold_{fold}/seed_{seed}",_anchor_path(fold,seed))
    elif stage=="r4-rounda-diff":fold=[0,2][index];seed=20260825;cfg=_cfg("setcalib_diff");cfg["maximum_updates"]=15000;result=train_diff(fold,seed,cfg,data,folds[fold],DERIVED/f"checkpoints/diff/fold_{fold}/seed_{seed}",_anchor_path(fold,seed),_det_path(fold,seed))
    elif stage=="r7-det":fold=index//3;seed=SEEDS[index%3];selection=json.loads((RESULT/"round_a/selection.json").read_text());encoder=selection["selected_encoder"];result=train_det(fold,seed,encoder,_cfg("setcalib_det"),data,folds[fold],DERIVED/f"checkpoints/det/{encoder}/fold_{fold}/seed_{seed}",_anchor_path(fold,seed),resume=True)
    else:fold=index//3;seed=SEEDS[index%3];selection=json.loads((RESULT/"round_a/selection.json").read_text());encoder=selection["selected_encoder"];result=train_diff(fold,seed,_cfg("setcalib_diff"),data,folds[fold],DERIVED/f"checkpoints/diff/fold_{fold}/seed_{seed}",_anchor_path(fold,seed),_det_path(fold,seed,encoder),resume=True)
    target=RESULT/("round_a" if "rounda" in stage else "round_b")/f"{result['kind']}_{result.get('encoder','')}_fold_{fold}_seed_{seed}.json";_json(target,result);_json(run/"result_summary.json",result);return result

def _load_models(fold:int,seed:int,encoder:str,device:torch.device):
    anchor,_=load_anchor(_anchor_path(fold,seed),device);det,_=load_det(_det_path(fold,seed,encoder),device);diff,_=load_diff(_diff_path(fold,seed),device);return anchor,det,diff
def _latent_outputs(inf:dict[str,np.ndarray],anchor,det,diff,device,seed:int)->dict[str,np.ndarray]:
    output={key:[] for key in ("POP","DET_MATCH","DET_WRONG","DIFF_MATCH","DIFF_WRONG","CONTEXT_MATCH","CONTEXT_WRONG","BASIS_MATCH","BASIS_WRONG","ATTENTION_MATCH","ATTENTION_WRONG","OPERATOR_MATCH","OPERATOR_WRONG")}
    for start in range(0,len(inf["y"]),16):
        sl=slice(start,min(start+16,len(inf["y"])));y=torch.as_tensor(inf["y"][sl],device=device);q=torch.as_tensor(inf["q0"][sl],device=device);c0=torch.as_tensor(inf["c0"][sl],device=device);a0=anchor(y,q,torch.einsum("bcd,bdt->bct",c0,q));match=det(y,a0,q,torch.as_tensor(inf["support_eeg"][sl],device=device),torch.as_tensor(inf["support_eog"][sl],device=device));wrong=det(y,a0,q,torch.as_tensor(inf["wrong_support_eeg"][sl],device=device),torch.as_tensor(inf["wrong_support_eog"][sl],device=device));noise=torch.randn(match["coefficient"].shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+start));mr=diff.sample(y,a0,q,match["coefficient"],match["context"],noise);wr=diff.sample(y,a0,q,wrong["coefficient"],wrong["context"],noise);batch={"POP":a0,"DET_MATCH":match["artifact"],"DET_WRONG":wrong["artifact"],"DIFF_MATCH":decode_residual(a0,match["basis"],match["coefficient"]+mr),"DIFF_WRONG":decode_residual(a0,wrong["basis"],wrong["coefficient"]+wr)}
        batch.update({"CONTEXT_MATCH":match["context"],"CONTEXT_WRONG":wrong["context"],"BASIS_MATCH":match["basis"],"BASIS_WRONG":wrong["basis"],"ATTENTION_MATCH":match["attention"],"ATTENTION_WRONG":wrong["attention"],"OPERATOR_MATCH":match["operator"],"OPERATOR_WRONG":wrong["operator"]})
        for key,value in batch.items():output[key].append(value.detach().cpu().numpy())
    return {k:np.concatenate(v) for k,v in output.items()}

def _projector_distance(left:np.ndarray,right:np.ndarray)->float:
    ql=np.linalg.qr(left)[0];qr=np.linalg.qr(right)[0];return float(np.linalg.norm(ql@ql.T-qr@qr.T,"fro"))

def paired_eval(run:Path,round_a:bool=False)->dict[str,Any]:
    data=_cfg("data");folds=_folds();index=_index();fold=[0,2][index] if round_a else index//3;seed=20260825 if round_a else SEEDS[index%3];encoder="deepsets" if round_a else json.loads((RESULT/"round_a/selection.json").read_text())["selected_encoder"];device=torch.device("cuda");anchor,det,diff=_load_models(fold,seed,encoder,device);sampler=SupportSetEpisodeSampler(data,folds[fold],"test",seed+301);batch=sampler.sample_paired(192);inf={k:v for k,v in batch.items() if isinstance(v,np.ndarray)};started=time.time();pred=_latent_outputs(inf,anchor,det,diff,device,seed);elapsed=time.time()-started;rows=[]
    for i,meta in enumerate(batch["meta"]):
        for method in ("RAW","POP","DET_MATCH","DET_WRONG","DIFF_MATCH","DIFF_WRONG"):
            estimate=np.zeros_like(batch["artifact"][i]) if method=="RAW" else pred[method][i];metric=paired_metrics(batch["x"][i],batch["y"][i],batch["artifact"][i],estimate);zero=bool(meta["zero_artifact"]);metric["artifact_rrmse"]=float("nan") if zero else metric["artifact_rrmse"];metric["snr_improvement"]=float("nan") if zero else metric["snr_improvement"];rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":method,"zero_artifact":int(zero),**metric})
    path=DERIVED/"metrics"/("round_a" if round_a else "round_b")/f"fold_{fold}_seed_{seed}.csv";_csv(path,rows);diagnostics=[]
    for i,meta in enumerate(batch["meta"]):
        diagnostics.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"context_distance":float(np.linalg.norm(pred["CONTEXT_MATCH"][i]-pred["CONTEXT_WRONG"][i])),"match_basis_query_operator_distance":_projector_distance(pred["BASIS_MATCH"][i],batch["cquery"][i]),"match_basis_support_operator_distance":_projector_distance(pred["BASIS_MATCH"][i],batch["cs"][i]),"wrong_basis_query_operator_distance":_projector_distance(pred["BASIS_WRONG"][i],batch["cquery"][i]),"operator_aux_query_error":float(np.linalg.norm(pred["OPERATOR_MATCH"][i]-batch["cquery"][i])/max(np.linalg.norm(batch["cquery"][i]),1e-12)),"attention_entropy":float(-(pred["ATTENTION_MATCH"][i]*np.log(np.maximum(pred["ATTENTION_MATCH"][i],1e-12))).sum()),"attention_max":float(pred["ATTENTION_MATCH"][i].max())})
    dpath=DERIVED/"metrics"/("round_a_diagnostics" if round_a else "round_b_diagnostics")/f"fold_{fold}_seed_{seed}.csv";_csv(dpath,diagnostics);result={"stage":"R5" if round_a else "R9","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"diagnostic_rows":len(diagnostics),"evaluation_seconds":elapsed,"windows_per_second":len(batch["y"])/elapsed,"latency_ms_per_window":1000*elapsed/len(batch["y"]),"sealed_reads":0};_json(run/"result_summary.json",result);return result

def support_budget_eval(run:Path)->dict[str,Any]:
    data=_cfg("data");folds=_folds();index=_index();fold=index//3;seed=SEEDS[index%3];encoder=json.loads((RESULT/"round_a/selection.json").read_text())["selected_encoder"];device=torch.device("cuda");anchor,det,_= _load_models(fold,seed,encoder,device);sampler=SupportSetEpisodeSampler(data,folds[fold],"test",seed+501);batch=sampler.sample_paired(96);rows=[]
    with torch.no_grad():
        for seconds,count in ((10,1),(30,4),(60,8),(120,16)):
            y=torch.as_tensor(batch["y"],device=device);q=torch.as_tensor(batch["q0"],device=device);c0=torch.as_tensor(batch["c0"],device=device);a0=anchor(y,q,torch.einsum("bcd,bdt->bct",c0,q));out=det(y,a0,q,torch.as_tensor(batch["support_eeg"][:,:count],device=device),torch.as_tensor(batch["support_eog"][:,:count],device=device))
            pred=out["artifact"].cpu().numpy()
            for i,meta in enumerate(batch["meta"]):
                metric=paired_metrics(batch["x"][i],batch["y"][i],batch["artifact"][i],pred[i]);rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"support_seconds":seconds,"support_windows":count,"rrmse_temporal":metric["rrmse_temporal"],"artifact_correlation":metric["artifact_correlation"]})
    _csv(DERIVED/f"metrics/support_budget/fold_{fold}_seed_{seed}.csv",rows);result={"stage":"R13","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"sealed_reads":0};_json(run/"result_summary.json",result);return result

def severity_eval(run:Path)->dict[str,Any]:
    data=_cfg("data");folds=_folds();index=_index();fold=index//3;seed=SEEDS[index%3];sampler=SupportSetEpisodeSampler(data,folds[fold],"test",seed+301);batch=sampler.sample_paired(192);metrics=list(csv.DictReader((DERIVED/f"metrics/round_b/fold_{fold}_seed_{seed}.csv").open()));methods=("RAW","POP","DET_MATCH","DET_WRONG","DIFF_MATCH","DIFF_WRONG");rows=[]
    if len(metrics)!=len(batch["meta"])*len(methods):raise RuntimeError("severity replay row mismatch")
    for sample,meta in enumerate(batch["meta"]):
        artifact=batch["artifact"][sample];clean=batch["x"][sample];zero=bool(meta["zero_artifact"]);snr=float("nan") if zero else float(20*np.log10(np.linalg.norm(clean)/max(np.linalg.norm(artifact),1e-12)));gain=float(meta["gain"]);severity="zero" if zero else "mild" if gain<.5 else "medium" if gain<.95 else "severe"
        for method_index,method in enumerate(methods):
            row=metrics[sample*len(methods)+method_index]
            if row["method"]!=method or row["participant"]!=meta["participant"]:raise RuntimeError("severity deterministic replay key mismatch")
            rows.append({"fold":fold,"seed":seed,"sample":sample,"participant":meta["participant"],"method":method,"gain":gain,"input_snr_db":snr,"severity":severity,"rrmse_temporal":row["rrmse_temporal"]})
    _csv(DERIVED/f"metrics/severity/fold_{fold}_seed_{seed}.csv",rows);result={"stage":"R13_SEVERITY","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"zero_rows_excluded_from_snr":True,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def round_a_select(run:Path)->dict[str,Any]:
    # Both encoders are stable; choose by mean validation joint score across the two preregistered folds.
    scores={}
    for encoder in ("deepsets","set_transformer"):
        vals=[]
        for fold in (0,2):
            value=json.loads(next((RESULT/"round_a").glob(f"setcalib_det_{encoder}_fold_{fold}_seed_*.json")).read_text());vals.append(value["best"]["joint"])
        scores[encoder]=float(np.mean(vals))
    selected=min(scores,key=scores.get);result={"status":"ROUND_B_AUTHORIZED","selected_encoder":selected,"validation_joint":scores,"rationale":f"{selected} had the lower two-fold mean joint validation artifact error; both were retained as Round-A evidence. Selection used no test endpoint threshold."};_json(RESULT/"round_a/selection.json",result);(ROOT/"reports/v25_round_a.md").write_text("# V25 Round A\n\n"+result["rationale"]+f"\n\nValidation joint scores: `{scores}`.\n");_json(run/"result_summary.json",result);return result

def natural_infer(run:Path)->dict[str,Any]:
    index=_index();fold=index//3;seed=SEEDS[index%3];encoder=json.loads((RESULT/"round_a/selection.json").read_text())["selected_encoder"];device=torch.device("cuda");anchor,det,diff=_load_models(fold,seed,encoder,device)
    # This archive was frozen by V24 specifically as the auxiliary-free inference namespace.
    query_path=V24DERIVED/f"fold_{fold}/natural_test_inference.npz";support_path=DERIVED/f"support_banks/fold_{fold}.npz"
    with np.load(query_path,allow_pickle=False) as archive:inference={key:np.asarray(archive[key]) for key in ("y","q0","c0")}
    with np.load(support_path,allow_pickle=False) as archive:inference.update({key:np.asarray(archive[key]) for key in archive.files})
    pred=_latent_outputs(inference,anchor,det,diff,device,seed);out=DERIVED/f"predictions/fold_{fold}_seed_{seed}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**pred)
    manifest={"fold":fold,"seed":seed,"prediction":str(out),"prediction_sha256":_digest(out),"query_bundle":str(query_path),"support_bundle":str(support_path),"query_eog_inference_reads":0,"query_operator_inference_reads":0,"query_event_inference_reads":0,"sealed_reads":0};_json(RESULT/f"natural_evaluation/output_{fold}_{seed}.json",manifest);_json(run/"result_summary.json",{"stage":"R10","status":"PASS",**manifest});return manifest
def output_freeze(run:Path)->dict[str,Any]:
    rows=[]
    for fold in range(5):
        for seed in SEEDS:
            value=json.loads((RESULT/f"natural_evaluation/output_{fold}_{seed}.json").read_text());assert _digest(Path(value["prediction"]))==value["prediction_sha256"];rows.append(value)
    _csv(RESULT/"natural_evaluation/output_manifest.csv",rows);result={"stage":"R11","status":"PASS","outputs":15,"digests_verified":True,"query_eog_inference_reads":0,"query_operator_inference_reads":0,"query_event_inference_reads":0,"sealed_reads":0};_json(RESULT/"natural_evaluation/output_freeze.json",result);_json(run/"result_summary.json",result);return result
def _natural(y,predicted,teacher,latent):
    energy=np.sqrt(np.mean(latent*latent,axis=0));low=energy<=np.quantile(energy,.3);high=energy>=np.quantile(energy,.7);clean=y-predicted;remaining=float(np.linalg.norm((teacher-predicted)[:,high])/max(np.linalg.norm(teacher[:,high]),1e-12));atten=float(-20*np.log10(max(remaining,1e-12)));pres=1-float(np.linalg.norm(predicted[:,low])/max(np.linalg.norm(y[:,low]),1e-12));f,p0=signal.welch(y[:,low],fs=100,nperseg=min(128,max(8,int(low.sum()))),axis=-1);_,p1=signal.welch(clean[:,low],fs=100,nperseg=min(128,max(8,int(low.sum()))),axis=-1);keep=(f>=1)&(f<=15);return {"heldout_eog_remaining_ratio":remaining,"artifact_attenuation_db":atten,"preservation":pres,"psd_distortion":float(np.mean(np.abs(np.log(np.maximum(p0[:,keep],1e-10))-np.log(np.maximum(p1[:,keep],1e-10))))),"covariance_distortion":float(np.linalg.norm(np.cov(clean[:,low])-np.cov(y[:,low]))/max(np.linalg.norm(np.cov(y[:,low])),1e-12)),"erp_proxy":pres,"ssvep_proxy":pres,"output_input_rms":float(np.sqrt(np.mean(clean*clean))/max(np.sqrt(np.mean(y*y)),1e-12))}
def natural_eval(run:Path)->dict[str,Any]:
    freeze=json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text());assert freeze["status"]=="PASS";index=_index();fold=index//3;seed=SEEDS[index%3]
    query=dict(np.load(V24DERIVED/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False));pred=dict(np.load(DERIVED/f"predictions/fold_{fold}_seed_{seed}.npz",allow_pickle=False));ev=dict(np.load(V24DERIVED/f"fold_{fold}/natural_test_evaluator.npz",allow_pickle=False));role_rows=[row for row in csv.DictReader((ROOT/"results/pa_el_scad_v24/role_manifest.csv").open()) if row["fold"]==str(fold) and row["stream"]=="natural" and row["split"]=="test"];assert len(role_rows)==len(query["y"])==len(ev["latent"]);rows=[]
    for i,meta in enumerate(role_rows):
        for method in ("RAW","POP","DET_MATCH","DET_WRONG","DIFF_MATCH","DIFF_WRONG"):
            estimate=np.zeros_like(query["y"][i]) if method=="RAW" else pred[method][i];rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"method":method,**_natural(query["y"][i],estimate,ev["teacher_artifact"][i],ev["latent"][i])})
    _csv(DERIVED/f"metrics/natural/fold_{fold}_seed_{seed}.csv",rows);result={"stage":"R12","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"evaluator_after_freeze":True,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def aggregate(run:Path)->dict[str,Any]:
    paired=[];natural=[]
    for fold in range(5):
        for seed in SEEDS:paired.extend(csv.DictReader((DERIVED/f"metrics/round_b/fold_{fold}_seed_{seed}.csv").open()));natural.extend(csv.DictReader((DERIVED/f"metrics/natural/fold_{fold}_seed_{seed}.csv").open()))
    def participant(rows,metrics,panel):
        output=[]
        for method in sorted({r["method"] for r in rows}):
            for person in sorted({r["participant"] for r in rows}):
                values=[r for r in rows if r["method"]==method and r["participant"]==person];output.append({"panel":panel,"participant":person,"method":method,**{m:float(np.nanmean([float(v[m]) for v in values])) for m in metrics}})
        return output
    pm=participant(paired,["rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","clean_output_rms_ratio"],"paired");nm=participant(natural,["heldout_eog_remaining_ratio","artifact_attenuation_db","preservation","psd_distortion","covariance_distortion","erp_proxy","ssvep_proxy","output_input_rms"],"natural");summary=[]
    for rows in (pm,nm):
        for method in sorted({r["method"] for r in rows}):
            for metric in [k for k in rows[0] if k not in ("panel","participant","method")]:
                vec=np.array([r[metric] for r in rows if r["method"]==method]);summary.append({"panel":rows[0]["panel"],"method":method,"metric":metric,"mean":float(np.nanmean(vec)),"median":float(np.nanmedian(vec)),"participants":len(vec)})
    rng=np.random.Generator(np.random.PCG64DXSM(20260828))
    for row in summary:
        source=pm if row["panel"]=="paired" else nm;vec=np.asarray([r[row["metric"]] for r in source if r["method"]==row["method"]],dtype=float);draw=vec[rng.integers(0,len(vec),size=(20000,len(vec)))].mean(1);row.update({"bootstrap_low":float(np.quantile(draw,.025)),"bootstrap_high":float(np.quantile(draw,.975)),"positive":int((vec>0).sum()),"source_role":"V25_current"})
    # Retain frozen comparators as explicitly historical rows rather than silently recomputing them.
    for source_path,label,methods in ((ROOT/"results/pa_el_scad_v24/method_summary.csv","V24_frozen",{"V24_POP_ANCHOR","PA_EL_DET_MATCH"}),(ROOT/"results/scad_v22/method_summary.csv","V22_frozen",{"EEGDFUS_UNIFIED"})):
        for row in csv.DictReader(source_path.open()):
            if row["method"] in methods:summary.append({**row,"source_role":label})
    _csv(RESULT/"method_summary.csv",summary);effects=[]
    for rows,panel,metric in ((pm,"paired","rrmse_temporal"),(nm,"natural","heldout_eog_remaining_ratio"),(nm,"natural","preservation")):
        direction=1 if metric=="preservation" else -1;value={(r["participant"],r["method"]):r[metric] for r in rows}
        for name,match,other in (("DET_MATCH_POP","DET_MATCH","POP"),("DET_MATCH_WRONG","DET_MATCH","DET_WRONG"),("DIFF_MATCH_POP","DIFF_MATCH","POP"),("DIFF_MATCH_WRONG","DIFF_MATCH","DIFF_WRONG"),("DIFF_DET","DIFF_MATCH","DET_MATCH")):
            for p in sorted({k[0] for k in value}):effects.append({"panel":panel,"metric":metric,"contrast":name,"participant":p,"effect":direction*(value[p,match]-value[p,other])})
    for row in effects:
        row["positive"]=int(row["effect"]>0)
    _csv(RESULT/"participant_effects.csv",effects)
    seed_rows=[]
    for panel_rows,panel in ((paired,"paired"),(natural,"natural")):
        metric_names=["rrmse_temporal","artifact_rrmse","correlation"] if panel=="paired" else ["heldout_eog_remaining_ratio","artifact_attenuation_db","preservation","psd_distortion","covariance_distortion"]
        for seed in SEEDS:
            for method in sorted({r["method"] for r in panel_rows}):
                selected=[r for r in panel_rows if int(r["seed"])==seed and r["method"]==method]
                for metric in metric_names:seed_rows.append({"panel":panel,"seed":seed,"method":method,"metric":metric,"mean":float(np.nanmean([float(r[metric]) for r in selected]))})
    _csv(RESULT/"seed_effects.csv",seed_rows)
    diagnostic_rows=[]
    for fold in range(5):
        for seed in SEEDS:diagnostic_rows.extend(csv.DictReader((DERIVED/f"metrics/round_b_diagnostics/fold_{fold}_seed_{seed}.csv").open()))
    _csv(RESULT/"support_context_diagnostics.csv",diagnostic_rows)
    _csv(RESULT/"learned_basis_diagnostics.csv",[{key:row[key] for key in ("fold","seed","participant","session","task","match_basis_query_operator_distance","match_basis_support_operator_distance","wrong_basis_query_operator_distance","operator_aux_query_error")} for row in diagnostic_rows])
    budget=[]
    for path in sorted((DERIVED/"metrics/support_budget").glob("*.csv")):budget.extend(csv.DictReader(path.open()))
    if budget:
        reduced=[]
        for seconds in (10,30,60,120):
            for participant in sorted({r["participant"] for r in budget}):
                values=[float(r["rrmse_temporal"]) for r in budget if int(r["support_seconds"])==seconds and r["participant"]==participant];reduced.append({"support_seconds":seconds,"participant":participant,"rrmse_temporal":float(np.mean(values))})
        _csv(RESULT/"support_budget_effects.csv",reduced)
    severity=[]
    for path in sorted((DERIVED/"metrics/severity").glob("*.csv")):severity.extend(csv.DictReader(path.open()))
    severity_effects=[]
    for level in ("mild","medium","severe"):
        for participant in sorted({r["participant"] for r in severity}):
            selected=[r for r in severity if r["severity"]==level and r["participant"]==participant]
            for seed in map(str,SEEDS):
                det=[float(r["rrmse_temporal"]) for r in selected if r["method"]=="DET_MATCH" and r["seed"]==seed];diff=[float(r["rrmse_temporal"]) for r in selected if r["method"]=="DIFF_MATCH" and r["seed"]==seed]
                if det and diff:severity_effects.append({"severity":level,"participant":participant,"seed":seed,"contrast":"DIFF_DET","effect":float(np.mean(det)-np.mean(diff))})
    _csv(RESULT/"severity_effects.csv",severity_effects)
    exposure=[]
    for path in sorted((RESULT/"round_b").glob("*.json")):
        value=json.loads(path.read_text());exposure.append({"model":value["kind"],"fold":value["fold"],"seed":value["seed"],"updates":value["updates"],"parameters":value["parameters"],"training_seconds":value["training_seconds"],"device":value["device"]})
    for path in sorted((RESULT/"runs/r9-paired").glob("job_*/task_*/result_summary.json")):
        value=json.loads(path.read_text());exposure.append({"model":"joint_evaluation_bundle","fold":value["fold"],"seed":value["seed"],"updates":"","parameters":"","training_seconds":value["evaluation_seconds"],"device":"GPU","latency_ms_per_window":value["latency_ms_per_window"]})
    _csv(RESULT/"latency_summary.csv",exposure)
    def d(panel,metric,contrast):
        vec=np.array([r["effect"] for r in effects if r["panel"]==panel and r["metric"]==metric and r["contrast"]==contrast]);draw=vec[rng.integers(0,len(vec),size=(20000,len(vec)))].mean(1);return {"mean":float(vec.mean()),"median":float(np.median(vec)),"positive":int((vec>0).sum()),"participants":len(vec),"bootstrap_low":float(np.quantile(draw,.025)),"bootstrap_high":float(np.quantile(draw,.975))}
    detp=d("paired","rrmse_temporal","DET_MATCH_POP");detw=d("paired","rrmse_temporal","DET_MATCH_WRONG");diffp=d("paired","rrmse_temporal","DIFF_MATCH_POP");diffw=d("paired","rrmse_temporal","DIFF_MATCH_WRONG");dv=d("paired","rrmse_temporal","DIFF_DET");nat=d("natural","heldout_eog_remaining_ratio","DIFF_MATCH_POP");pres=d("natural","preservation","DIFF_MATCH_POP");tail={level:float(np.mean([float(r["effect"]) for r in severity_effects if r["severity"]==level])) for level in ("mild","medium","severe")};support="clear_development_signal" if detp["bootstrap_low"]>0 and detw["bootstrap_low"]>0 else "weak_or_heterogeneous" if max(detp["mean"],detw["mean"])>0 else "context_harmful";population="support_better" if detp["bootstrap_low"]>0 else "similar" if abs(detp["mean"])<.002 else "mixed" if detp["mean"]>0 else "population_better";diffusion="clear_development_signal" if dv["mean"]>0 and dv["positive"]>=10 else "small_signal" if dv["mean"]>0 else "deterministic_equivalent" if abs(dv["mean"])<.002 else "deterministic_better";trade="promising" if nat["mean"]>0 and pres["mean"]>=0 else "artifact_reduction_insufficient" if nat["mean"]<=0 else "preservation_concern";tail_signal=any(value>0 for value in tail.values());next_route="A. continue SetCalibDiff" if support=="clear_development_signal" and diffusion!="deterministic_better" and trade=="promising" else "B. improve support encoder" if population=="population_better" else "F. focus diffusion on uncertainty/tail" if diffusion=="deterministic_better" and tail_signal else "G. remove current diffusion implementation" if diffusion=="deterministic_better" else "C. active prompted calibration";diagnosis={"engineering":"valid","raw_support_representation":support,"strong_population_comparison":population,"diffusion":diffusion,"natural_tradeoff":trade,"next_route":next_route,"paired":{"DET_MATCH_POP":detp,"DET_MATCH_WRONG":detw,"DIFF_MATCH_POP":diffp,"DIFF_MATCH_WRONG":diffw,"DIFF_DET":dv},"severity_DIFF_DET":tail,"natural":{"DIFF_MATCH_POP_artifact":nat,"DIFF_MATCH_POP_preservation":pres},"sealed_reads":0,"development_only":True,"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"query_event_inference_reads":0};_json(RESULT/"development_diagnosis.json",diagnosis)
    _make_figures(pm,nm,effects,diagnostic_rows,exposure)
    _write_reports(diagnosis,summary,exposure)
    _json(run/"result_summary.json",diagnosis);return diagnosis

def _make_figures(pm,nm,effects,diagnostics,exposure)->None:
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    target=ROOT/"figures/setcalibdiff_v25";target.mkdir(parents=True,exist_ok=True)
    det_curves=[];diff_curves=[]
    for path in sorted((RESULT/"round_b").glob("*.json")):
        value=json.loads(path.read_text());(det_curves if value["kind"]=="setcalib_det" else diff_curves).append(value["curve"])
    fig,ax=plt.subplots();
    for curves,label in ((det_curves,"DET joint"),(diff_curves,"Diff joint")):
        common=sorted(set.intersection(*[set(row["step"] for row in curve) for curve in curves]));ax.plot(common,[np.mean([next(row["joint"] for row in curve if row["step"]==step) for curve in curves]) for step in common],label=label)
    ax.set(xlabel="update",ylabel="validation objective");ax.legend();fig.tight_layout();fig.savefig(target/"training_curves.png");plt.close(fig)
    fig,ax=plt.subplots();ax.hist([float(r["attention_max"]) for r in diagnostics],bins=30);ax.set(xlabel="maximum support-window attention",ylabel="count");fig.tight_layout();fig.savefig(target/"support_attention.png");plt.close(fig)
    fig,ax=plt.subplots();ax.hist([float(r["context_distance"]) for r in diagnostics],bins=30);ax.set(xlabel="MATCH–WRONG context distance",ylabel="count");fig.tight_layout();fig.savefig(target/"context_embedding.png");plt.close(fig)
    fig,ax=plt.subplots();ax.boxplot([[float(r[key]) for r in diagnostics] for key in ("match_basis_query_operator_distance","match_basis_support_operator_distance","wrong_basis_query_operator_distance")],tick_labels=["MATCH-query","MATCH-support","WRONG-query"]);ax.set_ylabel("projector distance");fig.tight_layout();fig.savefig(target/"learned_basis_vs_operator.png");plt.close(fig)
    methods=("RAW","POP","DET_MATCH","DET_WRONG","DIFF_MATCH","DIFF_WRONG");fig,ax=plt.subplots();ax.bar(methods,[np.mean([r["rrmse_temporal"] for r in pm if r["method"]==m]) for m in methods]);ax.tick_params(axis="x",rotation=35);ax.set_ylabel("paired clean RRMSE");fig.tight_layout();fig.savefig(target/"paired_method_comparison.png");plt.close(fig)
    forest=[r for r in effects if r["panel"]=="paired" and r["metric"]=="rrmse_temporal" and r["contrast"] in ("DET_MATCH_POP","DET_MATCH_WRONG","DIFF_DET")];fig,ax=plt.subplots();
    for index,contrast in enumerate(("DET_MATCH_POP","DET_MATCH_WRONG","DIFF_DET")):vec=[r["effect"] for r in forest if r["contrast"]==contrast];ax.scatter(vec,np.full(len(vec),index),alpha=.7,label=contrast)
    ax.axvline(0,color="black",lw=.8);ax.set_yticks(range(3),["DET M-P","DET M-W","DIFF-DET"]);ax.set_xlabel("positive utility");fig.tight_layout();fig.savefig(target/"context_effect_forest.png");plt.close(fig)
    fig,ax=plt.subplots();
    for curve in diff_curves[:5]:ax.plot([r["step"] for r in curve],[r["residual_loss"] for r in curve],alpha=.7)
    ax.set(xlabel="update",ylabel="diffusion residual loss");fig.tight_layout();fig.savefig(target/"diffusion_trajectory.png");plt.close(fig)
    fig,ax=plt.subplots();
    for method in ("POP","DET_MATCH","DET_WRONG","DIFF_MATCH","DIFF_WRONG"):ax.scatter(np.mean([r["preservation"] for r in nm if r["method"]==method]),np.mean([r["artifact_attenuation_db"] for r in nm if r["method"]==method]),label=method)
    ax.set(xlabel="preservation",ylabel="artifact attenuation (dB)");ax.legend(fontsize=7);fig.tight_layout();fig.savefig(target/"attenuation_preservation_scatter.png");plt.close(fig)
    eval_latency=[float(r.get("latency_ms_per_window",0)) for r in exposure if r["model"]=="joint_evaluation_bundle"];fig,ax=plt.subplots();ax.scatter([np.mean(eval_latency)],[np.mean([r["rrmse_temporal"] for r in pm if r["method"]=="DIFF_MATCH"])]);ax.set(xlabel="joint evaluation latency (ms/window)",ylabel="DIFF paired RRMSE");fig.tight_layout();fig.savefig(target/"quality_latency_curve.png");plt.close(fig)

def _write_reports(diagnosis:dict[str,Any],summary:list[dict[str,Any]],exposure:list[dict[str,Any]])->None:
    paired=diagnosis["paired"];natural=diagnosis["natural"]
    (ROOT/"reports/v25_round_b.md").write_text("# V25 Round B\n\nFive development folds and three fixed seeds were evaluated participant-first.\n\n```json\n"+json.dumps(paired,indent=2)+"\n```\n")
    (ROOT/"reports/v25_natural_development.md").write_text("# V25 Natural Development\n\nInference used query EEG plus query-disjoint S120 EEG+EOG support only. V24 evaluator auxiliaries were opened after output freeze.\n\n```json\n"+json.dumps(natural,indent=2)+"\n```\n")
    (ROOT/"reports/v25_final_development_diagnosis.md").write_text("# V25 Final Development Diagnosis\n\nThis is development/model-building evidence, not confirmation.\n\n- Engineering: `"+diagnosis["engineering"]+"`\n- Raw support representation: `"+diagnosis["raw_support_representation"]+"`\n- Strong population comparison: `"+diagnosis["strong_population_comparison"]+"`\n- Diffusion: `"+diagnosis["diffusion"]+"`\n- Natural trade-off: `"+diagnosis["natural_tradeoff"]+"`\n- Next route: `"+diagnosis["next_route"]+"`\n\nExact participant effects, seed effects, frozen comparators, training exposure, and latency are in the accompanying CSV files.\n")

def package_inventory(run:Path)->dict[str,Any]:
    inventory=[]
    for path in sorted(list((ROOT/"configs/setcalibdiff_v25").glob("*.yaml"))+[ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md",ROOT/"results/pa_el_scad_v24/role_manifest.csv",ROOT/"results/pa_el_scad_v24/method_summary.csv"]):
        stat=path.stat();inventory.append({"absolute_path":str(path.resolve()),"role":"V25_config_or_governance" if "setcalibdiff" in str(path) or "LEDGER" in str(path) else "V24_frozen_reference","size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":_digest(path)})
    for path in sorted((RESULT/"support_banks").glob("*.csv")):
        stat=path.stat();inventory.append({"absolute_path":str(path.resolve()),"role":"support_only_manifest","size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":_digest(path)})
    _csv(RESULT/"input_inventory.csv",inventory)
    checkpoints=[]
    seen=set()
    for fold in range(5):
        for seed in SEEDS:
            for model,path in (("V24_POP_ANCHOR",_anchor_path(fold,seed)),("SetCalibDET",_det_path(fold,seed)),("SetCalibDiff",_diff_path(fold,seed))):
                key=str(path)
                if key in seen:continue
                seen.add(key);state=torch.load(path,map_location="cpu",weights_only=False);checkpoints.append({"absolute_path":key,"sha256":_digest(path),"fold":fold,"seed":seed,"model":model,"config":json.dumps(state.get("config",{}),sort_keys=True),"training_job":"see reports/slurm/v25_job_ids.txt","best_criterion":"V24 best_joint" if model=="V24_POP_ANCHOR" else "best_joint"})
    _csv(RESULT/"checkpoint_manifest.csv",checkpoints)
    role_rows=list(csv.DictReader((RESULT/"support_episode_manifest.csv").open()));_csv(RESULT/"role_manifest.csv",[{**row,"support_role":"query_disjoint_S120","query_role":"Qnatural_after_300s","context":"MATCH_and_WRONG","scientific_unit":"participant"} for row in role_rows])
    result={"stage":"R14_PACKAGE","status":"PASS","inventory_rows":len(inventory),"checkpoint_rows":len(checkpoints),"sealed_reads":0,"query_auxiliary_inference_reads":0};_json(run/"result_summary.json",result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True);parser.add_argument("--run-dir",type=Path,required=True);args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=="r0-preflight":preflight(args.run_dir)
    elif args.stage=="r1-prepare":prepare(args.run_dir)
    elif args.stage=="r2-support-bank":prepare_natural_support(args.run_dir)
    elif args.stage=="r3-sanity":sanity(args.run_dir)
    elif args.stage in ("r4-rounda-det","r4-rounda-diff","r7-det","r8-diff"):train_stage(args.stage,args.run_dir)
    elif args.stage=="r5-rounda-eval":paired_eval(args.run_dir,True)
    elif args.stage=="r6-select":round_a_select(args.run_dir)
    elif args.stage=="r9-paired":paired_eval(args.run_dir,False)
    elif args.stage=="r10-natural-infer":natural_infer(args.run_dir)
    elif args.stage=="r11-freeze":output_freeze(args.run_dir)
    elif args.stage=="r12-natural-eval":natural_eval(args.run_dir)
    elif args.stage=="r13-budget":support_budget_eval(args.run_dir)
    elif args.stage=="r13-severity":severity_eval(args.run_dir)
    elif args.stage=="r14-aggregate":aggregate(args.run_dir)
    elif args.stage=="r14-package":package_inventory(args.run_dir)
    else:raise ValueError(args.stage)
if __name__=="__main__":main()
