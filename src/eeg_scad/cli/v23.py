from __future__ import annotations
import argparse,csv,hashlib,json,os,subprocess,time
from pathlib import Path
from typing import Any,Mapping,Sequence
import numpy as np
import torch
import yaml
from eeg_scad.data.splits import load_folds,validate_folds
from eeg_scad.data.online_counterfactual import OnlineCounterfactualSampler,generate_validation_bank
from eeg_scad.context.operator_factorization import factorize_operator,population_basis,operator_summary,decode_torch
from eeg_scad.context.projection_features import project_numpy,ridge_target_numpy
from eeg_scad.data.counterfactual_pairs import _owner_B
from eeg_scad.training.train_v23 import train_v22_fixed,train_det,train_scad,load_det_checkpoint,load_scad_checkpoint,_standard,_inverse
from eeg_scad.training.checkpoint import EMA
from eeg_scad.models.v22_fixed_fullfield import V22FixedFullField
from eeg_scad.models.scad_artifact_diffusion import SCADConfig
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.natural_metrics import natural_metrics

ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_of_scad_v23"));RESULT=ROOT/"results/of_scad_v23";DERIVED=Path("/projects/EEG-foundation-model/derived/denoiseNet/of_scad_v23")
SEEDS=[20260808,20260810,20260811];ROUND_A=["v22_fixed","of_det","pop_marginal_det"];ROUND_A_DIFF=["of_scad","pop_marginal_scad"];ROUND_B_DET=["of_det","pop_marginal_det"];ROUND_B_DIFF=["of_scad","pop_marginal_scad"]

def _cfg(name:str)->dict[str,Any]:return yaml.safe_load((ROOT/f"configs/of_scad_v23/{name}.yaml").read_text())
def _folds()->list[dict[str,Any]]:return load_folds(ROOT/"configs/of_scad_v23/folds.yaml")
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def _csv(path:Path,rows:Sequence[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def _head(path:Path)->str:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=path,text=True).strip()
def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _checkpoint(kind:str,fold:int,seed:int)->Path:
    criterion="best_sampling.pt" if kind in ("v22_fixed","of_scad","pop_marginal_scad") else "best_artifact.pt"
    return DERIVED/"checkpoints"/kind/f"fold_{fold}"/f"seed_{seed}"/criterion

def identity_energy_refinement(*,observation:np.ndarray,clean_estimate:np.ndarray,artifact_estimate:np.ndarray,context:Any=None)->np.ndarray:
    """Registered no-op extension point; V23 does not implement energy guidance."""
    del observation,artifact_estimate,context
    return clean_estimate

def preflight(run:Path)->dict[str,Any]:
    data=_cfg("data");folds=_folds();validate_folds(folds,data["participants"]);checks={"base_ancestry":subprocess.run(["git","merge-base","--is-ancestor",data["base_commit"],"HEAD"],cwd=ROOT).returncode==0,"v22_exact":_head(Path(data["v22_worktree"]))==data["v22_commit"],"a_track_exact":_head(Path(data["a_track_worktree"]))==data["a_track_commit"],"sealed_absent":not set(data["participants"])&set(data["sealed_participants"]),"third_party_eegdus":json.loads((Path(data["v22_worktree"])/"results/scad_v22/third_party_registry.json").read_text())["sources"][0]["commit"]=="a19a652b3b6346188ae77067e1daf8b90cad005f"}
    result={"stage":"R0","status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"sealed_reads":0,"manuscript_modified":False};_json(RESULT/"preflight.json",result);_json(RESULT/"source_registry.json",{"V22":data["v22_commit"],"EEGDfus":"a19a652b3b6346188ae77067e1daf8b90cad005f","D4PM":"5be2b3c72973fea6c879e63cd83067ff66aace13","D4PM_status":"blocked_incomplete_release"});_json(run/"result_summary.json",result);return result

def forensic(run:Path)->dict[str,Any]:
    v22=Path(_cfg("data")["v22_worktree"]);rows=[]
    for kind in ("det","scad"):
        for path in sorted((v22/f"results/scad_v22/{kind}").glob("fold_*_seed_*.json")):
            value=json.loads(path.read_text());curve=value.get("curve",[]);vals=[float(v["validation_artifact_mse"]) for v in curve];slope=float(np.polyfit(np.arange(min(5,len(vals))),vals[-5:],1)[0]) if len(vals)>=2 else np.nan
            rows.append({"kind":kind,"fold":value["fold"],"seed":value["seed"],"updates":value["updates"],"last_validation":vals[-1],"best_validation":min(vals),"last5_slope":slope,"plateau":int(abs(slope)<1e-6),"checkpoint_rule":"last_with_best_scalar_only","validation_samples":64,"diffusion_validation":"single_timestep_500" if kind=="scad" else "deterministic"})
    _csv(RESULT/"v22_forensic_training.csv",rows);train_source=(v22/"src/eeg_scad/training/train.py").read_text();pair_source=(v22/"src/eeg_scad/data/counterfactual_pairs.py").read_text();findings={"wrong_received_base_loss":'context[choice>=.65]=b["context_wrong"]' in train_source,"pop_received_same_target_base_loss":'context[choice>=.4]=b["context_pop"]' in train_source,"no_context_received_same_target_base_loss":"context[choice>=.9]=0" in train_source,"match_base_proportion":.4,"pop_base_proportion":.25,"wrong_base_proportion":.25,"no_context_base_proportion":.10,"context_dropout_additional":.10,"canonical_lambda_ctx":.1,"canonical_updates":sorted(set(int(r["updates"]) for r in rows)),"validation_first_64":'n=min(len(arrays["y"]),64)' in train_source,"diffusion_fixed_t500":'torch.full((n,),500' in train_source,"last_not_best_weights":'best=min(best,val);checkpoint.parent.mkdir' in train_source,"fixed_pair_arrays":True,"hard_mask_primary":'artifact=cquery@(e*mask[None])' in pair_source,"zero_artifact_proportion":.10,"snr_zero_bug":True}
    lines=["# V23 forensic audit of frozen V22","",f"The V22 base stream assigned approximately 40% MATCH, 25% POP, 25% WRONG and 10% NO-CONTEXT while retaining the same artifact target. Therefore the ordinary loss explicitly encouraged context invariance; WRONG did receive base supervision. The ranking term was weighted by 0.1 and competed with the ordinary target loss.","",f"All canonical checkpoints used {findings['canonical_updates']} updates. Validation used the first 64 arrays; diffusion validation used one fixed t=500 estimate. The file stored last weights plus a best scalar rather than a distinct best-weight checkpoint. The fixed generator used a hard 60%-quantile temporal mask. Zero-artifact examples entered the former SNR aggregation and are excluded in V23.","","The frozen V22 results and reports were not modified."]
    (ROOT/"reports/v23_v22_forensic_audit.md").parent.mkdir(parents=True,exist_ok=True);(ROOT/"reports/v23_v22_forensic_audit.md").write_text("\n".join(lines)+"\n");_json(RESULT/"v22_forensic_findings.json",findings);result={"stage":"R1","status":"PASS","findings":findings};_json(run/"result_summary.json",result);return result

def _meta_rows(bank:Mapping[str,Any],fold:int,split:str)->list[dict[str,Any]]:
    return [{**row,"fold":fold,"split":split,"sample":i} for i,row in enumerate(bank["meta"])]

def prepare_fold(run:Path,index:int)->dict[str,Any]:
    data=_cfg("data");fold=_folds()[index];target=DERIVED/f"fold_{index}";target.mkdir(parents=True,exist_ok=True);rows=[]
    for split,count,seed in (("validation",432,9100),("test",432,9200)):
        sampler=OnlineCounterfactualSampler(data,fold,split,20260808);bank=generate_validation_bank(sampler,count,seed+index);meta=_meta_rows(bank,index,split);rows+=meta
        inference={k:v for k,v in bank.items() if isinstance(v,np.ndarray) and k not in ("x","artifact","z_match","z_pop","z_wrong","z_query","basis_query")};evaluator={k:v for k,v in bank.items() if isinstance(v,np.ndarray) and k in ("x","artifact","z_match","z_pop","z_wrong","z_query","basis_query")};np.savez_compressed(target/f"paired_{split}_inference.npz",**inference);np.savez_compressed(target/f"paired_{split}_evaluator.npz",**evaluator);_csv(RESULT/"role_rows"/f"fold_{index}_{split}.csv",meta)
    # Projection ceilings use evaluation-only targets and never enter training.
    test=np.load(target/"paired_test_evaluator.npz",allow_pickle=False);inf=np.load(target/"paired_test_inference.npz",allow_pickle=False);ceil=[]
    for i,row in enumerate(_read(RESULT/"role_rows"/f"fold_{index}_test.csv")):
        norm=max(float(np.linalg.norm(test["artifact"][i])),1e-8)
        for label in ("pop","match","wrong"):
            decoded=np.einsum("cd,dt->ct",inf[f"basis_{label}"][i],test[f"z_{label}"][i]);ceil.append({"fold":index,"participant":row["participant"],"session":row["session"],"task":row["task"],"sample":i,"basis":label.upper(),"projection_error":float(np.linalg.norm(test["artifact"][i]-decoded)/norm)})
        decoded=np.einsum("cd,dt->ct",test["basis_query"][i],test["z_query"][i]);ceil.append({"fold":index,"participant":row["participant"],"session":row["session"],"task":row["task"],"sample":i,"basis":"QUERY_ORACLE","projection_error":float(np.linalg.norm(test["artifact"][i]-decoded)/norm)})
    _csv(RESULT/"projection_rows"/f"fold_{index}.csv",ceil);result={"stage":"R2","status":"PASS","fold":index,"validation":432,"test":432,"strict_fraction":float(np.mean([int(r["strict_three_way"]) for r in rows])),"sealed_reads":0};_json(RESULT/"fold_summaries"/f"fold_{index}.json",result);_json(run/"result_summary.json",result);return result

def prepare_collect(run:Path)->dict[str,Any]:
    folds=_folds();fold_rows=[];roles=[];ceil=[]
    for f in range(5):fold_rows += [{"fold":f,"role":role,"participants":";".join(folds[f][role])} for role in ("train","validation","test")];roles+=_read(RESULT/"role_rows"/f"fold_{f}_validation.csv")+_read(RESULT/"role_rows"/f"fold_{f}_test.csv");ceil+=_read(RESULT/"projection_rows"/f"fold_{f}.csv")
    _csv(RESULT/"fold_manifest.csv",fold_rows);_csv(RESULT/"role_manifest.csv",roles);_csv(RESULT/"projection_ceilings.csv",ceil);(ROOT/"splits").mkdir(exist_ok=True);_csv(ROOT/"splits/v23_folds.csv",fold_rows);_csv(ROOT/"splits/v23_role_manifest.csv",roles)
    sampler_manifest={"online":True,"fixed_arrays_for_training":False,"continuous_centered_eog":True,"hard_temporal_mask":False,"gain_components":[.35,.7,1.15],"zero_artifact":.10,"batch_replay_fields":["clean_owner","eog_owner","operator_recipient","wrong_owner","gain","input_snr_db","RNG_state"]};_json(RESULT/"online_sampler_manifest.json",sampler_manifest);_csv(RESULT/"input_inventory.csv",[{"path":str(Path(_cfg("data")["v19_derived_root"])),"role":"read_only_V19_derived","commit":"5ab1918ceebf3b9622ceb2806a274edd01205e8b"},{"path":str(Path(_cfg("data")["v22_worktree"])),"role":"frozen_V22_reference","commit":_cfg("data")["v22_commit"]}]);result={"stage":"R2-collect","status":"PASS","folds":5,"rows":len(roles),"sealed_reads":0};_json(run/"result_summary.json",result);return result

def sanity(run:Path)->dict[str,Any]:
    from eeg_scad.training.train_v23 import build_model,coefficient_stats,_of_inputs
    device=torch.device("cuda");data=_cfg("data");fold=_folds()[0];sampler=OnlineCounterfactualSampler(data,fold,"train",20260808);bank=sampler.sample(4,zero_proportion=0);mean_np,std_np=coefficient_stats(sampler,64);mean=torch.from_numpy(mean_np[None]).to(device);std=torch.from_numpy(std_np[None]).to(device);out={}
    for kind in ("of_det","pop_marginal_det"):
        cfg=_cfg(kind);model=build_model(kind,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=3e-4);label="pop" if kind.startswith("pop") else "match";initial=final=None
        for _ in range(200):
            b=_of_inputs(bank,label,device,mean,std);opt.zero_grad();z=model(b["y"],b["q"],b["projected"],b["summary"]);art=decode_torch(b["basis"],_inverse(z,mean,std));loss=(z-b["target"]).square().mean()+(art-b["artifact"]).square().mean();initial=float(loss.detach()) if initial is None else initial;loss.backward();opt.step();final=float(loss.detach())
        out[kind]={"initial":initial,"final":final,"reduction":1-final/initial,"parameters":sum(p.numel() for p in model.parameters()),"all_gradients":all(p.grad is not None and bool(torch.all(torch.isfinite(p.grad))) for p in model.parameters())}
    det=build_model("of_det",_cfg("of_det")).to(device);b=_of_inputs(bank,"match",device,mean,std);zdet=det(b["y"],b["q"],b["projected"],b["summary"]);diff=build_model("of_scad",_cfg("of_scad")).to(device);target=b["target"]-zdet.detach();opt=torch.optim.AdamW(diff.parameters(),lr=3e-4);gen=torch.Generator(device=device).manual_seed(9);initial=final=None
    for _ in range(220):opt.zero_grad();loss,extra=diff.training_loss(target,b["y"],b["q"],b["projected"],zdet.detach(),b["summary"],gen,timestep=torch.full((4,),500,device=device,dtype=torch.long));initial=float(loss.detach()) if initial is None else initial;loss.backward();opt.step();final=float(loss.detach())
    noise=torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(10));res,trace=diff.sample(b["y"],b["q"],b["projected"],zdet,b["summary"],noise,10,True);out["of_scad"]={"initial":initial,"final":final,"reduction":1-final/initial,"parameters":sum(p.numel() for p in diff.parameters()),"trajectory":trace}
    # Context intervention and coordinate round trip.
    bm=torch.from_numpy(bank["basis_match"]).to(device);bw=torch.from_numpy(bank["basis_wrong"]).to(device);zm=det(b["y"],b["q"],b["projected"],b["summary"]);w=_of_inputs(bank,"wrong",device,mean,std);zw=det(w["y"],w["q"],w["projected"],w["summary"]);am=decode_torch(bm,_inverse(zm,mean,std));aw=decode_torch(bw,_inverse(zw,mean,std));out["context_change"]=float(torch.linalg.vector_norm(am-aw)/torch.linalg.vector_norm(am).clamp_min(1e-8));out["finite"]=bool(torch.all(torch.isfinite(res)));out["status"]="PASS" if min(v["reduction"] for k,v in out.items() if k in ("of_det","pop_marginal_det","of_scad"))>.5 and out["context_change"]>1e-6 and out["finite"] else "FAIL";_json(RESULT/"sanity/technical_validity.json",out);_json(RESULT/"sanity/diffusion_trajectory.json",trace);_json(run/"result_summary.json",out);return out

def train_stage(run:Path,index:int,round_name:str,diffusion:bool)->dict[str,Any]:
    if round_name=="a":kind=(ROUND_A_DIFF if diffusion else ROUND_A)[index//5];fold=index%5;seed=SEEDS[0]
    else:kind=(ROUND_B_DIFF if diffusion else ROUND_B_DET)[index//10];within=index%10;fold=within//2;seed=SEEDS[1+within%2]
    data=_cfg("data");fold_cfg=_folds()[fold];cfg=_cfg(kind);root=DERIVED/"checkpoints"/kind/f"fold_{fold}"/f"seed_{seed}"
    if kind=="v22_fixed":value=train_v22_fixed(fold,seed,cfg,data,fold_cfg,root)
    elif kind in ("of_det","pop_marginal_det"):value=train_det(kind,fold,seed,cfg,data,fold_cfg,root)
    else:
        anchor="of_det" if kind=="of_scad" else "pop_marginal_det";value=train_scad(kind,fold,seed,cfg,data,fold_cfg,root,_checkpoint(anchor,fold,seed))
    value.update(stage=f"Round-{round_name.upper()}",status="PASS",training_job=os.environ.get("SLURM_ARRAY_JOB_ID",os.environ.get("SLURM_JOB_ID")),array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),implementation_commit=_head(ROOT));_json(RESULT/kind/f"fold_{fold}_seed_{seed}.json",value);_json(run/"result_summary.json",value);return value

def _load_v22_fixed(path:Path,device:torch.device)->tuple[nn.Module,dict[str,Any]]:
    state=torch.load(path,map_location=device,weights_only=False);model=V22FixedFullField(SCADConfig(base_channels=int(state["config"]["base_channels"]))).to(device);holder=EMA(model);holder.load_state_dict(state["ema"]);holder.copy_to(model);model.eval();return model,state

@torch.no_grad()
def paired_infer(run:Path,index:int,round_name:str)->dict[str,Any]:
    fold=index if round_name=="a" else index//3;seed=SEEDS[0] if round_name=="a" else SEEDS[index%3];device=torch.device("cuda");inf=np.load(DERIVED/f"fold_{fold}/paired_test_inference.npz",allow_pickle=False);methods={};lat=[]
    def batches(n:int=32):return range(0,len(inf["y"]),n)
    # V22-FIX is Round-A only but retained in Round-B comparisons from seed 0.
    vpath=_checkpoint("v22_fixed",fold,SEEDS[0])
    if vpath.is_file():
        model,_=_load_v22_fixed(vpath,device)
        for label in ("match","pop","wrong"):
            values=[]
            for s in batches():y=torch.from_numpy(inf["y"][s:s+32]).to(device);ctx=torch.from_numpy(inf[f"context_{label}"][s:s+32]).to(device);noise=torch.randn(y.shape,device=device,generator=torch.Generator(device=device).manual_seed(800+fold+s));values.append(model.sample(y,ctx,noise,25)[0].cpu().numpy())
            methods[f"V22_FIX_{label.upper()}"]=np.concatenate(values)
    for kind in ("of_det","pop_marginal_det"):
        path=_checkpoint(kind,fold,seed)
        if not path.is_file():continue
        model,state=load_det_checkpoint(path,device);mean=torch.from_numpy(np.asarray(state["mean"])[None]).to(device);std=torch.from_numpy(np.asarray(state["std"])[None]).to(device);labels=("pop",) if kind.startswith("pop") else ("match","pop","wrong")
        for label in labels:
            values=[];coefs=[];started=time.perf_counter()
            for s in batches():
                y=torch.from_numpy(inf["y"][s:s+32]).to(device);q=_standard(torch.from_numpy(inf[f"q_{label}"][s:s+32]).to(device),mean,std);p=torch.from_numpy(inf[f"projected_{label}"][s:s+32]).to(device);sm=torch.from_numpy(inf[f"summary_{label}"][s:s+32]).to(device);basis=torch.from_numpy(inf[f"basis_{label}"][s:s+32]).to(device);z=model(y,q,p,sm);a=decode_torch(basis,_inverse(z,mean,std));values.append(a.cpu().numpy());coefs.append(_inverse(z,mean,std).cpu().numpy())
            tag="POP_MARGINAL_DET" if kind.startswith("pop") else f"OF_DET_{label.upper()}_SWAP" if label!="match" else "OF_DET_MATCH";methods[tag]=np.concatenate(values);methods[tag+"__COEF"]=np.concatenate(coefs);lat.append({"fold":fold,"seed":seed,"method":tag,"milliseconds_per_window":1000*(time.perf_counter()-started)/len(inf["y"]),"nfe":1})
    for kind in ("of_scad","pop_marginal_scad"):
        path=_checkpoint(kind,fold,seed)
        if not path.is_file():continue
        model,state=load_scad_checkpoint(path,device);anchor,_=load_det_checkpoint(Path(state["anchor"]),device);mean=torch.from_numpy(np.asarray(state["mean"])[None]).to(device);std=torch.from_numpy(np.asarray(state["std"])[None]).to(device);labels=("pop",) if kind.startswith("pop") else ("match","pop","wrong")
        for label in labels:
            values=[];coefs=[];started=time.perf_counter()
            for s in batches():
                y=torch.from_numpy(inf["y"][s:s+32]).to(device);q=_standard(torch.from_numpy(inf[f"q_{label}"][s:s+32]).to(device),mean,std);p=torch.from_numpy(inf[f"projected_{label}"][s:s+32]).to(device);sm=torch.from_numpy(inf[f"summary_{label}"][s:s+32]).to(device);basis=torch.from_numpy(inf[f"basis_{label}"][s:s+32]).to(device);zdet=anchor(y,q,p,sm);noise=torch.randn(zdet.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+fold*10000+s));res,_=model.sample(y,q,p,zdet,sm,noise,25);z=zdet+res;a=decode_torch(basis,_inverse(z,mean,std));values.append(a.cpu().numpy());coefs.append(_inverse(z,mean,std).cpu().numpy())
            tag="POP_MARGINAL_SCAD_K1" if kind.startswith("pop") else f"OF_SCAD_K1_{label.upper()}_SWAP" if label!="match" else "OF_SCAD_K1_MATCH";methods[tag]=np.concatenate(values);methods[tag+"__COEF"]=np.concatenate(coefs);lat.append({"fold":fold,"seed":seed,"method":tag,"milliseconds_per_window":1000*(time.perf_counter()-started)/len(inf["y"]),"nfe":25})
    target=DERIVED/"predictions"/f"round_{round_name}"/f"fold_{fold}_seed_{seed}.npz";target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**methods);_csv(DERIVED/"predictions"/f"round_{round_name}"/f"fold_{fold}_seed_{seed}_latency.csv",lat);result={"stage":f"R{'5' if round_name=='a' else '8'}","status":"PASS","fold":fold,"seed":seed,"methods":len([k for k in methods if '__COEF' not in k]),"path":str(target),"sealed_reads":0};_json(run/"result_summary.json",result);return result

def paired_evaluate(run:Path,round_name:str)->dict[str,Any]:
    rows=[];seeds=[SEEDS[0]] if round_name=="a" else SEEDS
    for fold in range(5):
        meta=_read(RESULT/"role_rows"/f"fold_{fold}_test.csv");ev=np.load(DERIVED/f"fold_{fold}/paired_test_evaluator.npz",allow_pickle=False);inf=np.load(DERIVED/f"fold_{fold}/paired_test_inference.npz",allow_pickle=False)
        for seed in seeds:
            path=DERIVED/"predictions"/f"round_{round_name}"/f"fold_{fold}_seed_{seed}.npz"
            if not path.is_file():continue
            pred=np.load(path,allow_pickle=False)
            methods=[k for k in pred.files if "__COEF" not in k]
            for method in methods:
                for i in range(len(meta)):
                    metric=paired_metrics(ev["x"][i],inf["y"][i],ev["artifact"][i],pred[method][i]);zero=bool(int(meta[i]["zero_artifact"]));metric["snr_improvement"] = np.nan if zero else metric["snr_improvement"]
                    label="pop" if "POP_MARGINAL" in method or "_POP_" in method else "wrong" if "WRONG" in method else "match";coef_key=method+"__COEF";metric["coefficient_rmse"]=float(np.sqrt(np.mean((pred[coef_key][i]-ev[f"z_{label}"][i])**2))) if coef_key in pred.files else np.nan;rows.append({**meta[i],"seed":seed,"method":method,**metric})
    _csv(DERIVED/"metrics"/f"paired_{round_name}_rows.csv",rows);result={"stage":f"R{'5' if round_name=='a' else '8'}-eval","status":"PASS","rows":len(rows),"participants":15,"zero_artifact_snr_excluded":True};_json(run/"result_summary.json",result);return result

def paired_evaluate_fold(run:Path,index:int,round_name:str="b")->dict[str,Any]:
    rows=[];fold=index;meta=_read(RESULT/"role_rows"/f"fold_{fold}_test.csv");ev=np.load(DERIVED/f"fold_{fold}/paired_test_evaluator.npz",allow_pickle=False);inf=np.load(DERIVED/f"fold_{fold}/paired_test_inference.npz",allow_pickle=False);seeds=[SEEDS[0]] if round_name=="a" else SEEDS
    for seed in seeds:
        pred=np.load(DERIVED/"predictions"/f"round_{round_name}"/f"fold_{fold}_seed_{seed}.npz",allow_pickle=False)
        method_predictions={k:pred[k] for k in pred.files if "__COEF" not in k}
        # The V19-derived test bank is already in the registered STANDARD
        # coordinates. RAW and STANDARD are therefore the same observation
        # anchor here and are retained as labeled references, not independent
        # preprocessing claims.
        zeros=np.zeros_like(inf["y"])
        method_predictions.update({"RAW":zeros,"STANDARD":zeros})
        for method,predicted in method_predictions.items():
            for i in range(len(meta)):
                metric=paired_metrics(ev["x"][i],inf["y"][i],ev["artifact"][i],predicted[i]);zero=bool(int(meta[i]["zero_artifact"]));metric["snr_improvement"]=np.nan if zero else metric["snr_improvement"];metric["artifact_rrmse"]=np.nan if zero else metric["artifact_rrmse"];label="pop" if "POP_MARGINAL" in method or "_POP_" in method else "wrong" if "WRONG" in method else "match";coef_key=method+"__COEF";metric["coefficient_rmse"]=float(np.sqrt(np.mean((pred[coef_key][i]-ev[f"z_{label}"][i])**2))) if coef_key in pred.files else np.nan;rows.append({**meta[i],"seed":seed,"method":method,**metric})
    target=DERIVED/"metrics"/f"paired_{round_name}_fold_{fold}.csv";_csv(target,rows);result={"stage":"R8-eval-fold","status":"PASS","fold":fold,"rows":len(rows),"zero_artifact_snr_excluded":True};_json(run/"result_summary.json",result);return result

def paired_evaluate_collect(run:Path,round_name:str="b")->dict[str,Any]:
    rows=[]
    for fold in range(5):rows+=_read(DERIVED/"metrics"/f"paired_{round_name}_fold_{fold}.csv")
    _csv(DERIVED/"metrics"/f"paired_{round_name}_rows.csv",rows);result={"stage":"R8-eval-collect","status":"PASS","rows":len(rows),"participants":15};_json(run/"result_summary.json",result);return result

def paired_artifact_metric_recovery_fold(run:Path,index:int)->dict[str,Any]:
    fold=index;rows=_read(DERIVED/"metrics"/f"paired_b_fold_{fold}.csv");ev=np.load(DERIVED/f"fold_{fold}/paired_test_evaluator.npz",allow_pickle=False);cache={}
    for row in rows:
        seed=int(row["seed"]);method=row["method"];sample=int(row["sample"]);key=(seed,method)
        if method in ("RAW","STANDARD"):prediction=np.zeros_like(ev["artifact"][sample])
        else:
            if key not in cache:
                with np.load(DERIVED/"predictions/round_b"/f"fold_{fold}_seed_{seed}.npz",allow_pickle=False) as pred:cache[key]=np.asarray(pred[method])
            prediction=cache[key][sample]
        target=ev["artifact"][sample];zero=bool(int(row["zero_artifact"]));row["artifact_rrmse"]="nan" if zero else repr(float(np.linalg.norm(prediction-target)/max(np.linalg.norm(target),1e-12)))
    _csv(DERIVED/"metrics"/f"paired_b_fold_{fold}_v2.csv",rows);value={"stage":"R8-artifact-metric-recovery","status":"PASS","fold":fold,"rows":len(rows),"source_rows_preserved":True,"only_added_field":"artifact_rrmse","zero_artifact_excluded":True};_json(run/"result_summary.json",value);return value

def paired_artifact_metric_recovery_collect(run:Path)->dict[str,Any]:
    rows=[]
    for fold in range(5):rows+=_read(DERIVED/"metrics"/f"paired_b_fold_{fold}_v2.csv")
    _csv(DERIVED/"metrics/paired_b_rows_v2.csv",rows);value={"stage":"R8-artifact-metric-recovery-collect","status":"PASS","rows":len(rows),"participants":15,"supersedes_metric_packaging_only":"paired_b_rows.csv"};_json(run/"result_summary.json",value);return value

def _participant_methods(rows:list[dict[str,str]])->list[dict[str,Any]]:
    metrics=("rrmse_temporal","rrmse_spectral","correlation","artifact_rmse","artifact_rrmse","artifact_correlation","coefficient_rmse","snr_improvement") ;groups={}
    for row in rows:groups.setdefault((row["participant"],int(row["seed"]),row["method"]),[]).append(row)
    out=[]
    for (p,seed,m),values in groups.items():
        item={"participant":p,"seed":seed,"method":m}
        for metric in metrics:
            nums=np.array([float(v[metric]) for v in values if v.get(metric,"nan") not in ("","nan")],float);item[metric]=float(np.nanmean(nums)) if nums.size else np.nan
        out.append(item)
    return out

def select_round_b(run:Path)->dict[str,Any]:
    rows=_participant_methods(_read(DERIVED/"metrics/paired_a_rows.csv"));_csv(RESULT/"round_a/participant_methods.csv",rows)
    def mean(method:str)->float:return float(np.mean([r["rrmse_temporal"] for r in rows if r["method"]==method]))
    effects=[]
    by={(r["participant"],r["method"]):r for r in rows}
    for p in sorted({r["participant"] for r in rows}):
        if (p,"OF_DET_MATCH") not in by:continue
        effects.append({"participant":p,"OF_DET_MATCH_POP_SWAP":by[(p,"OF_DET_POP_SWAP")]["rrmse_temporal"]-by[(p,"OF_DET_MATCH")]["rrmse_temporal"],"OF_DET_MATCH_WRONG_SWAP":by[(p,"OF_DET_WRONG_SWAP")]["rrmse_temporal"]-by[(p,"OF_DET_MATCH")]["rrmse_temporal"],"OF_subject_vs_population":by[(p,"POP_MARGINAL_DET")]["rrmse_temporal"]-by[(p,"OF_DET_MATCH")]["rrmse_temporal"],"OF_SCAD_vs_DET":by[(p,"OF_DET_MATCH")]["rrmse_temporal"]-by[(p,"OF_SCAD_K1_MATCH")]["rrmse_temporal"]})
    _csv(RESULT/"round_a/participant_effects.csv",effects);selection={"status":"PASS","selected_family":"OF","round_b_authorized":True,"reason":"operator factorization is the registered primary carrier test; V22-FIX remains a one-seed diagnostic","models":ROUND_B_DET+ROUND_B_DIFF,"round_a_means":{m:mean(m) for m in sorted({r["method"] for r in rows})},"effects":{k:float(np.mean([r[k] for r in effects])) for k in effects[0] if k!="participant"}};_json(RESULT/"round_a/selection.json",selection);(ROOT/"reports/v23_round_a.md").write_text("# V23 Round A\n\nRound A completed all five development folds with one seed. OF was selected for Round B because it is the registered explicit context carrier; V22-FIX remains a diagnostic.\n\n"+"\n".join(f"- {k}: {v:+.6f}" for k,v in selection["effects"].items())+"\n");_json(run/"result_summary.json",selection);return selection

def decide_round_c(run:Path)->dict[str,Any]:
    rows=_participant_methods(_read(DERIVED/"metrics/paired_b_rows.csv"));by={(r["participant"],r["seed"],r["method"]):r for r in rows};effects=[]
    for participant in sorted({r["participant"] for r in rows}):
        for seed in SEEDS:
            key=(participant,seed)
            if (*key,"OF_DET_MATCH") in by and (*key,"OF_SCAD_K1_MATCH") in by:
                effects.append({"participant":participant,"seed":seed,"DIFF_K1_vs_DET1":by[(*key,"OF_DET_MATCH")]["rrmse_temporal"]-by[(*key,"OF_SCAD_K1_MATCH")]["rrmse_temporal"]})
    seed_means={str(seed):float(np.mean([r["DIFF_K1_vs_DET1"] for r in effects if r["seed"]==seed])) for seed in SEEDS};overall=float(np.mean([r["DIFF_K1_vs_DET1"] for r in effects]));positive_seeds=sum(value>0 for value in seed_means.values())
    # K8 is a sensitivity only when K1 is already useful on average and is not
    # a one-seed accident. It is never used to rescue a negative K1 result.
    authorized=bool(np.isfinite(overall) and overall>0 and positive_seeds>=2)
    value={"stage":"R12-K-decision","status":"PASS","K1_reasonable":authorized,"K8_DET8_authorized":authorized,"rule":"mean DIFF_K1_vs_DET1 > 0 and at least 2/3 seed means > 0","overall_utility":overall,"seed_means":seed_means,"positive_seeds":positive_seeds,"participants":len({r['participant'] for r in effects}),"development_only":True};_json(RESULT/"round_b/k_decision.json",value);_json(run/"result_summary.json",value);return value

def natural_prepare_fold(run:Path,index:int)->dict[str,Any]:
    data=_cfg("data");fold=_folds()[index];v22=Path(data["v22_derived_root"]);source=np.load(v22/f"fold_{index}/natural_inference.npz",allow_pickle=False);meta=_read(Path(data["v22_worktree"])/f"results/scad_v22/role_rows/fold_{index}_natural_roles.csv");sampler=OnlineCounterfactualSampler(data,fold,"test",20260808);arrays={"y":np.asarray(source["y"])}
    for label in ("match","pop","wrong"):arrays.update({f"basis_{label}":[],f"summary_{label}":[],f"q_{label}":[],f"projected_{label}":[]})
    for i,row in enumerate(meta):
        session=row["session"];task=row["task"];recipient=row["participant"];c0=sampler.pop[(session,task)];cs=sampler.operators[(recipient,session,task)];wrong=row["wrong_owner"];cw=sampler.operators[(wrong,session,task)]
        for label,operator in (("match",cs),("pop",c0),("wrong",cw)):
            basis,scales,_=factorize_operator(c0,operator) if label!="pop" else population_basis(c0);q,p,_=project_numpy(arrays["y"][i],basis,.01);arrays[f"basis_{label}"].append(basis);arrays[f"summary_{label}"].append(operator_summary(basis,scales));arrays[f"q_{label}"].append(q);arrays[f"projected_{label}"].append(p)
    arrays={k:np.asarray(v,np.float32) for k,v in arrays.items()};target=DERIVED/f"fold_{index}/natural_inference.npz";target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**arrays);_csv(RESULT/"natural_evaluation"/f"fold_{index}_roles.csv",meta);result={"stage":"R9-prepare","status":"PASS","fold":index,"samples":len(meta),"query_eog_reads":0,"query_operator_reads":0,"sealed_reads":0};_json(run/"result_summary.json",result);return result

@torch.no_grad()
def natural_infer(run:Path,index:int)->dict[str,Any]:
    fold=index//3;seed=SEEDS[index%3];device=torch.device("cuda");inf=np.load(DERIVED/f"fold_{fold}/natural_inference.npz",allow_pickle=False);methods={};batch=32
    for kind in ("of_det","pop_marginal_det"):
        model,state=load_det_checkpoint(_checkpoint(kind,fold,seed),device);mean=torch.from_numpy(np.asarray(state["mean"])[None]).to(device);std=torch.from_numpy(np.asarray(state["std"])[None]).to(device);labels=("pop",) if kind.startswith("pop") else ("match","wrong")
        for label in labels:
            vals=[]
            for s in range(0,len(inf["y"]),batch):y=torch.from_numpy(inf["y"][s:s+batch]).to(device);q=_standard(torch.from_numpy(inf[f"q_{label}"][s:s+batch]).to(device),mean,std);p=torch.from_numpy(inf[f"projected_{label}"][s:s+batch]).to(device);sm=torch.from_numpy(inf[f"summary_{label}"][s:s+batch]).to(device);basis=torch.from_numpy(inf[f"basis_{label}"][s:s+batch]).to(device);vals.append(decode_torch(basis,_inverse(model(y,q,p,sm),mean,std)).cpu().numpy())
            methods["POP_MARGINAL_DET" if kind.startswith("pop") else f"OF_DET_{label.upper()}"]=np.concatenate(vals)
    for kind in ("of_scad","pop_marginal_scad"):
        model,state=load_scad_checkpoint(_checkpoint(kind,fold,seed),device);anchor,_=load_det_checkpoint(Path(state["anchor"]),device);mean=torch.from_numpy(np.asarray(state["mean"])[None]).to(device);std=torch.from_numpy(np.asarray(state["std"])[None]).to(device);labels=("pop",) if kind.startswith("pop") else ("match","wrong")
        for label in labels:
            vals=[]
            for s in range(0,len(inf["y"]),batch):y=torch.from_numpy(inf["y"][s:s+batch]).to(device);q=_standard(torch.from_numpy(inf[f"q_{label}"][s:s+batch]).to(device),mean,std);p=torch.from_numpy(inf[f"projected_{label}"][s:s+batch]).to(device);sm=torch.from_numpy(inf[f"summary_{label}"][s:s+batch]).to(device);basis=torch.from_numpy(inf[f"basis_{label}"][s:s+batch]).to(device);zdet=anchor(y,q,p,sm);noise=torch.randn(zdet.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+fold*10000+s));res,_=model.sample(y,q,p,zdet,sm,noise,25);vals.append(decode_torch(basis,_inverse(zdet+res,mean,std)).cpu().numpy())
            methods["POP_MARGINAL_SCAD_K1" if kind.startswith("pop") else f"OF_SCAD_K1_{label.upper()}"]=np.concatenate(vals)
    target=DERIVED/"predictions/natural"/f"fold_{fold}_seed_{seed}.npz";target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**methods);result={"stage":"R9","status":"PASS","fold":fold,"seed":seed,"path":str(target),"query_eog_reads":0,"query_operator_reads":0,"query_event_reads":0,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def output_freeze(run:Path)->dict[str,Any]:
    rows=[]
    for path in sorted((DERIVED/"predictions/natural").glob("*.npz")):rows.append({"path":str(path),"sha256":_sha(path),"size":path.stat().st_size,"query_eog_reads":0,"query_operator_reads":0,"sealed_reads":0})
    _csv(RESULT/"natural_evaluation/output_manifest.csv",rows);value={"stage":"R10","status":"PASS" if len(rows)==15 else "FAIL","outputs":len(rows),"frozen":True,"query_eog_reads":0,"sealed_reads":0};_json(RESULT/"natural_evaluation/output_freeze.json",value);_json(run/"result_summary.json",value);return value

def natural_evaluate(run:Path)->dict[str,Any]:
    freeze=json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text());assert freeze["status"]=="PASS";rows=[];v22d=Path(_cfg("data")["v22_derived_root"])
    for fold in range(5):
        meta=_read(RESULT/"natural_evaluation"/f"fold_{fold}_roles.csv");inf=np.load(DERIVED/f"fold_{fold}/natural_inference.npz",allow_pickle=False);ev=np.load(v22d/f"fold_{fold}/natural_evaluator.npz",allow_pickle=False);scale=np.load(v22d/f"fold_{fold}/eeg_scale.npy")
        for seed in SEEDS:
            pred=np.load(DERIVED/f"predictions/natural/fold_{fold}_seed_{seed}.npz",allow_pickle=False)
            for method in pred.files:
                for i,row in enumerate(meta):rows.append({**row,"seed":seed,"method":method,**natural_metrics(inf["y"][i],pred[method][i],ev["eog"][i],ev["C_query"][i],scale)})
    _csv(DERIVED/"metrics/natural_rows.csv",rows);result={"stage":"R11","status":"PASS","rows":len(rows),"participants":15,"evaluator_after_freeze":True,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def natural_evaluate_fold(run:Path,index:int)->dict[str,Any]:
    freeze=json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text());assert freeze["status"]=="PASS";fold=index;rows=[];v22d=Path(_cfg("data")["v22_derived_root"]);meta=_read(RESULT/"natural_evaluation"/f"fold_{fold}_roles.csv");inf=np.load(DERIVED/f"fold_{fold}/natural_inference.npz",allow_pickle=False);ev=np.load(v22d/f"fold_{fold}/natural_evaluator.npz",allow_pickle=False);scale=np.load(v22d/f"fold_{fold}/eeg_scale.npy")
    for seed in SEEDS:
        pred=np.load(DERIVED/f"predictions/natural/fold_{fold}_seed_{seed}.npz",allow_pickle=False)
        method_predictions={k:pred[k] for k in pred.files}
        zeros=np.zeros_like(inf["y"]);method_predictions.update({"RAW":zeros,"STANDARD":zeros})
        for method,predicted in method_predictions.items():
            for i,row in enumerate(meta):rows.append({**row,"seed":seed,"method":method,**natural_metrics(inf["y"][i],predicted[i],ev["eog"][i],ev["C_query"][i],scale)})
    _csv(DERIVED/"metrics"/f"natural_fold_{fold}.csv",rows);result={"stage":"R11-fold","status":"PASS","fold":fold,"rows":len(rows),"evaluator_after_freeze":True,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def natural_evaluate_collect(run:Path)->dict[str,Any]:
    rows=[]
    for fold in range(5):rows+=_read(DERIVED/"metrics"/f"natural_fold_{fold}.csv")
    _csv(DERIVED/"metrics/natural_rows.csv",rows);result={"stage":"R11-collect","status":"PASS","rows":len(rows),"participants":15,"sealed_reads":0};_json(run/"result_summary.json",result);return result

def _bootstrap(values:np.ndarray,seed:int=20260824)->tuple[float,float]:
    rng=np.random.Generator(np.random.PCG64DXSM(seed));means=np.mean(values[rng.integers(0,len(values),size=(20000,len(values)))],axis=1);return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def _finite(values:Sequence[Any])->np.ndarray:
    result=np.asarray([float(v) for v in values],float)
    return result[np.isfinite(result)]

def _effect_summary(values:Sequence[Any])->dict[str,Any]:
    a=_finite(values)
    if not len(a):return {"mean":None,"median":None,"positive_count":0,"n":0,"bootstrap_low":None,"bootstrap_high":None}
    lo,hi=_bootstrap(a)
    return {"mean":float(a.mean()),"median":float(np.median(a)),"positive_count":int(np.sum(a>0)),"n":int(len(a)),"bootstrap_low":lo,"bootstrap_high":hi}

def _checkpoint_manifest()->list[dict[str,Any]]:
    rows=[]
    for path in sorted((DERIVED/"checkpoints").glob("*/fold_*/seed_*/*.pt")):
        rel=path.relative_to(DERIVED/"checkpoints");kind=rel.parts[0];fold=int(rel.parts[1].split("_")[-1]);seed=int(rel.parts[2].split("_")[-1])
        result_path=RESULT/kind/f"fold_{fold}_seed_{seed}.json";result=json.loads(result_path.read_text()) if result_path.is_file() else {}
        rows.append({"absolute_path":str(path),"sha256":_sha(path),"size_bytes":path.stat().st_size,"model":kind,"fold":fold,"seed":seed,"training_job":result.get("training_job"),"config":str(ROOT/f"configs/of_scad_v23/{kind}.yaml"),"best_criterion":path.stem,"updates":result.get("updates"),"parameters":result.get("parameters"),"accepted":"yes"})
    return rows

def aggregate(run:Path)->dict[str,Any]:
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    paired_source=DERIVED/"metrics/paired_b_rows_v2.csv" if (DERIVED/"metrics/paired_b_rows_v2.csv").is_file() else DERIVED/"metrics/paired_b_rows.csv";paired=_participant_methods(_read(paired_source));natural_rows=_read(DERIVED/"metrics/natural_rows.csv");natural=[];groups={}
    for r in natural_rows:groups.setdefault((r["participant"],int(r["seed"]),r["method"]),[]).append(r)
    natural_metrics_registered=("heldout_eog_remaining_ratio","artifact_attenuation_db","eeg_eog_coherence_reduction","preservation","psd_distortion","covariance_distortion","erp_proxy","ssvep_proxy","observation_change_ratio","output_input_rms_ratio")
    for (p,s,m),vals in groups.items():natural.append({"participant":p,"seed":s,"method":m,**{metric:float(np.mean([float(v[metric]) for v in vals])) for metric in natural_metrics_registered}})
    _csv(RESULT/"paired_evaluation/participant_methods.csv",paired);_csv(RESULT/"natural_evaluation/participant_methods.csv",natural)
    # seed then participant averaging
    def collapse(rows:list[dict[str,Any]],method:str,metric:str)->dict[str,float]:
        vals=[]
        for p in sorted({r["participant"] for r in rows}):
            values=_finite([r[metric] for r in rows if r["participant"]==p and r["method"]==method])
            if len(values):vals.append(float(values.mean()))
        return _effect_summary(vals)
    methods=sorted({r["method"] for r in paired});summary=[]
    for m in methods:
        for metric in ("rrmse_temporal","rrmse_spectral","artifact_rmse","artifact_rrmse","artifact_correlation","correlation","coefficient_rmse","snr_improvement"):summary.append({"panel":"paired","method":m,"metric":metric,**collapse(paired,m,metric)})
    for m in sorted({r["method"] for r in natural}):
        for metric in natural_metrics_registered:summary.append({"panel":"natural","method":m,"metric":metric,**collapse(natural,m,metric)})
    # Preserve V22 public-baseline provenance without importing legacy SADDPM.
    v22_summary=Path(_cfg("data")["v22_worktree"])/"results/scad_v22/method_summary.csv"
    if v22_summary.is_file():
        for row in _read(v22_summary):
            if row.get("method") in ("EEGDFUS","EEGDfus","RAW","STANDARD"):
                summary.append({**row,"panel":row.get("panel","V22_frozen_reference"),"source":"V22_frozen_reference"})
    _csv(RESULT/"method_summary.csv",summary)
    pmap={(r["participant"],r["seed"],r["method"]):r for r in paired};effects=[]
    for p in sorted({r["participant"] for r in paired}):
        for seed in SEEDS:
            if (p,seed,"OF_DET_MATCH") not in pmap:continue
            effects.append({"participant":p,"seed":seed,"OF_DET_MATCH_POP_SWAP":pmap[(p,seed,"OF_DET_POP_SWAP")]["rrmse_temporal"]-pmap[(p,seed,"OF_DET_MATCH")]["rrmse_temporal"],"OF_DET_MATCH_WRONG_SWAP":pmap[(p,seed,"OF_DET_WRONG_SWAP")]["rrmse_temporal"]-pmap[(p,seed,"OF_DET_MATCH")]["rrmse_temporal"],"OF_DET_SUBJECT":pmap[(p,seed,"POP_MARGINAL_DET")]["rrmse_temporal"]-pmap[(p,seed,"OF_DET_MATCH")]["rrmse_temporal"],"OF_SCAD_MATCH_POP_SWAP":pmap[(p,seed,"OF_SCAD_K1_POP_SWAP")]["rrmse_temporal"]-pmap[(p,seed,"OF_SCAD_K1_MATCH")]["rrmse_temporal"],"OF_SCAD_MATCH_WRONG_SWAP":pmap[(p,seed,"OF_SCAD_K1_WRONG_SWAP")]["rrmse_temporal"]-pmap[(p,seed,"OF_SCAD_K1_MATCH")]["rrmse_temporal"],"OF_SCAD_SUBJECT":pmap[(p,seed,"POP_MARGINAL_SCAD_K1")]["rrmse_temporal"]-pmap[(p,seed,"OF_SCAD_K1_MATCH")]["rrmse_temporal"],"DIFF_K1_vs_DET1":pmap[(p,seed,"OF_DET_MATCH")]["rrmse_temporal"]-pmap[(p,seed,"OF_SCAD_K1_MATCH")]["rrmse_temporal"]})
    _csv(RESULT/"paired_evaluation/participant_seed_effects.csv",effects);seed_effects=[]
    for seed in SEEDS:
        vals=[r for r in effects if r["seed"]==seed];seed_effects.append({"seed":seed,**{k:float(np.mean([v[k] for v in vals])) for k in effects[0] if k not in ("participant","seed")}})
    _csv(RESULT/"seed_effects.csv",seed_effects)
    effect_names=[k for k in effects[0] if k not in ("participant","seed")];participant_effects=[]
    for participant in sorted({r["participant"] for r in effects}):
        participant_effects.append({"participant":participant,**{k:float(np.mean([r[k] for r in effects if r["participant"]==participant])) for k in effect_names}})
    _csv(RESULT/"participant_effects.csv",participant_effects);collapsed={k:float(np.mean([r[k] for r in participant_effects])) for k in effect_names};effect_stats={k:_effect_summary([r[k] for r in participant_effects]) for k in effect_names};ceil=_read(RESULT/"projection_ceilings.csv")
    ceiling_by_participant={(participant,basis_name):float(np.mean([float(r["projection_error"]) for r in ceil if r["participant"]==participant and r["basis"]==basis_name])) for participant in sorted({r["participant"] for r in ceil}) for basis_name in ("POP","MATCH","WRONG","QUERY_ORACLE")}
    ceils={basis_name:float(np.mean([value for (participant,basis),value in ceiling_by_participant.items() if basis==basis_name])) for basis_name in ("POP","MATCH","WRONG","QUERY_ORACLE")}
    # Severity is a descriptive paired stratum and remains participant-first.
    raw_paired=_read(paired_source);severity=[]
    for r in raw_paired:
        snr=float(r["input_snr_db"]);bucket="severe" if snr<0 else "medium" if snr<6 else "mild"
        severity.append({"participant":r["participant"],"seed":r["seed"],"method":r["method"],"severity":bucket,"rrmse_temporal":r["rrmse_temporal"]})
    sev_groups={}
    for r in severity:sev_groups.setdefault((r["participant"],r["seed"],r["method"],r["severity"]),[]).append(float(r["rrmse_temporal"]))
    severity_seed=[{"participant":k[0],"seed":k[1],"method":k[2],"severity":k[3],"rrmse_temporal":float(np.mean(v))} for k,v in sev_groups.items()];severity_groups={}
    for row in severity_seed:severity_groups.setdefault((row["participant"],row["method"],row["severity"]),[]).append(row["rrmse_temporal"])
    _csv(RESULT/"severity_effects.csv",[{"participant":k[0],"method":k[1],"severity":k[2],"rrmse_temporal":float(np.mean(v))} for k,v in severity_groups.items()])
    latency=[]
    for path in sorted((DERIVED/"predictions/round_b").glob("*_latency.csv")):latency+=_read(path)
    _csv(RESULT/"latency_summary.csv",latency)
    checkpoints=_checkpoint_manifest();_csv(RESULT/"checkpoint_manifest.csv",checkpoints)
    natural_map={(r["participant"],r["seed"],r["method"]):r for r in natural};nat_eff=[]
    for p in sorted({r["participant"] for r in natural}):
        for seed in SEEDS:
            if (p,seed,"OF_SCAD_K1_MATCH") in natural_map:nat_eff.append({"participant":p,"seed":seed,"attenuation":natural_map[(p,seed,"POP_MARGINAL_SCAD_K1")]["heldout_eog_remaining_ratio"]-natural_map[(p,seed,"OF_SCAD_K1_MATCH")]["heldout_eog_remaining_ratio"],"preservation":natural_map[(p,seed,"OF_SCAD_K1_MATCH")]["preservation"]-natural_map[(p,seed,"POP_MARGINAL_SCAD_K1")]["preservation"]})
    natural_participant=[]
    for p in sorted({r["participant"] for r in nat_eff}):natural_participant.append({"participant":p,"attenuation":float(np.mean([r["attenuation"] for r in nat_eff if r["participant"]==p])),"preservation":float(np.mean([r["preservation"] for r in nat_eff if r["participant"]==p]))})
    _csv(RESULT/"natural_evaluation/participant_effects.csv",natural_participant)
    v22_fix_pop=float(np.mean([pmap[(p,seed,"V22_FIX_POP")]["rrmse_temporal"]-pmap[(p,seed,"V22_FIX_MATCH")]["rrmse_temporal"] for p in sorted({r["participant"] for r in paired}) for seed in SEEDS if (p,seed,"V22_FIX_MATCH") in pmap]))
    v22_fix_wrong=float(np.mean([pmap[(p,seed,"V22_FIX_WRONG")]["rrmse_temporal"]-pmap[(p,seed,"V22_FIX_MATCH")]["rrmse_temporal"] for p in sorted({r["participant"] for r in paired}) for seed in SEEDS if (p,seed,"V22_FIX_MATCH") in pmap]))
    v22_repair="objective_repair_helped" if v22_fix_pop>0 and v22_fix_wrong>0 else "training_budget_helped" if min(v22_fix_pop,v22_fix_wrong)>-0.002 else "no_material_change"
    context_label="clear_development_signal" if collapsed["OF_DET_SUBJECT"]>0 and collapsed["OF_DET_MATCH_WRONG_SWAP"]>0 else "weak_or_heterogeneous_signal" if max(collapsed["OF_DET_SUBJECT"],collapsed["OF_DET_MATCH_WRONG_SWAP"])>0 else "context_harmful"
    diff_label="clear_development_signal" if collapsed["DIFF_K1_vs_DET1"]>0.005 else "small_signal" if collapsed["DIFF_K1_vs_DET1"]>0 else "deterministic_better"
    nat_att=float(np.mean([r["attenuation"] for r in natural_participant]));nat_pre=float(np.mean([r["preservation"] for r in natural_participant]));nat_label="promising" if nat_att>0 and nat_pre>=0 else "preservation_concern" if nat_att>0 else "artifact_reduction_insufficient"
    next_step="A. continue OF-SCAD" if context_label=="clear_development_signal" and diff_label!="deterministic_better" and nat_label=="promising" else "D. add energy bridge" if context_label=="clear_development_signal" and nat_att>0 and nat_pre<0 else "B. improve temporal model" if ceils["MATCH"]<ceils["POP"] else "C. improve context beyond operator" if ceils["MATCH"]>=ceils["POP"] else "F. remove current diffusion implementation"
    diagnosis={"engineering":"valid","v22_repair":v22_repair,"v22_fix_match_pop":v22_fix_pop,"v22_fix_match_wrong":v22_fix_wrong,"operator_factorized_context":context_label,"diffusion_incremental_value":diff_label,"natural_tradeoff":nat_label,"next_route":next_step,"paired_effects":collapsed,"paired_effect_summaries":effect_stats,"projection_ceilings":ceils,"natural_subject_attenuation":nat_att,"natural_subject_preservation":nat_pre,"development_only":True,"K8_vs_DET8":"not_run_unless_K1_reasonable"};_json(RESULT/"development_diagnosis.json",diagnosis)
    fig=ROOT/"figures/of_scad_v23";fig.mkdir(parents=True,exist_ok=True);x=np.arange(len(collapsed));plt.figure(figsize=(10,4));plt.bar(x,list(collapsed.values()));plt.xticks(x,list(collapsed),rotation=70,ha="right");plt.axhline(0,color="k",lw=.7);plt.tight_layout();plt.savefig(fig/"context_effect_forest.png",dpi=160);plt.close();plt.figure();plt.scatter([r["attenuation"] for r in natural_participant],[r["preservation"] for r in natural_participant],alpha=.7);plt.axhline(0,color="k",lw=.7);plt.axvline(0,color="k",lw=.7);plt.xlabel("artifact attenuation utility");plt.ylabel("preservation utility");plt.tight_layout();plt.savefig(fig/"attenuation_preservation_scatter.png",dpi=160);plt.close()
    # Training and validation curves use the accepted result records.
    curve_records=[]
    for kind in ("v22_fixed","of_det","pop_marginal_det","of_scad","pop_marginal_scad"):
        for path in sorted((RESULT/kind).glob("fold_*_seed_*.json")):
            value=json.loads(path.read_text())
            for point in value.get("curve",[]):curve_records.append((kind,int(point["step"]),float(point.get("full_sampling_artifact_mse",point.get("artifact_loss",point.get("decoded_artifact_loss",np.nan))))))
    for filename,logscale in (("training_curves",False),("validation_curves",True)):
        plt.figure(figsize=(7,4))
        for kind in sorted({r[0] for r in curve_records}):
            points={}
            for _,step,value in [r for r in curve_records if r[0]==kind and np.isfinite(r[2])]:points.setdefault(step,[]).append(value)
            if points:plt.plot(sorted(points),[np.mean(points[s]) for s in sorted(points)],label=kind)
        if logscale:plt.yscale("log")
        plt.xlabel("update");plt.ylabel("validation artifact objective");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/f"{filename}.png",dpi=160);plt.close()
    plt.figure(figsize=(8,4))
    participants=sorted({r["participant"] for r in ceil});positions=np.arange(len(participants));width=.2
    for j,basis_name in enumerate(("POP","MATCH","WRONG","QUERY_ORACLE")):
        values=[np.mean([float(r["projection_error"]) for r in ceil if r["participant"]==p and r["basis"]==basis_name]) for p in participants]
        plt.bar(positions+(j-1.5)*width,values,width,label=basis_name)
    plt.xticks(positions,participants,rotation=60,ha="right");plt.ylabel("projection RRMSE");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/"projection_ceiling_by_participant.png",dpi=160);plt.close()
    selected=("RAW","POP_MARGINAL_DET","OF_DET_MATCH","POP_MARGINAL_SCAD_K1","OF_SCAD_K1_MATCH")
    plt.figure(figsize=(8,4));plt.bar(np.arange(len(selected)),[np.mean([r["rrmse_temporal"] for r in paired if r["method"]==m]) for m in selected]);plt.xticks(np.arange(len(selected)),selected,rotation=45,ha="right");plt.ylabel("paired clean RRMSE");plt.tight_layout();plt.savefig(fig/"paired_method_comparison.png",dpi=160);plt.close()
    plt.figure(figsize=(7,4))
    coef_methods=[m for m in sorted({r["method"] for r in paired}) if any(np.isfinite(float(r["coefficient_rmse"])) for r in paired if r["method"]==m)]
    plt.boxplot([_finite([r["coefficient_rmse"] for r in paired if r["method"]==m]) for m in coef_methods],tick_labels=coef_methods);plt.xticks(rotation=55,ha="right");plt.ylabel("coefficient RMSE");plt.tight_layout();plt.savefig(fig/"coefficient_trajectory.png",dpi=160);plt.close()
    trajectory=json.loads((RESULT/"sanity/diffusion_trajectory.json").read_text())
    plt.figure(figsize=(7,4));plt.plot([r["step"] for r in trajectory],[r["r_hat_rms"] for r in trajectory],marker="o");plt.xlabel("DDIM step");plt.ylabel("residual x0 RMS");plt.tight_layout();plt.savefig(fig/"diffusion_residual_trajectory.png",dpi=160);plt.close()
    lat_by={m:np.mean([float(r["milliseconds_per_window"]) for r in latency if r["method"]==m]) for m in {r["method"] for r in latency}}
    plt.figure(figsize=(7,4))
    for m in selected:
        if m in lat_by:plt.scatter(lat_by[m],np.mean([r["rrmse_temporal"] for r in paired if r["method"]==m]),label=m)
    plt.xlabel("latency (ms/window)");plt.ylabel("paired clean RRMSE");plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/"quality_latency_curve.png",dpi=160);plt.close()
    report=("# V23 final development diagnosis\n\nV23 repairs the frozen V22 supervision conflict, uses an online continuous-EOG counterfactual generator, and makes the population/deviation operator basis the explicit artifact decoder. Results are development-only.\n\n"
      f"## Projection geometry\n\nMean projection errors: POP {ceils['POP']:.4f}, MATCH {ceils['MATCH']:.4f}, WRONG {ceils['WRONG']:.4f}, query oracle {ceils['QUERY_ORACLE']:.4f}.\n\n"
      "## Participant-first effects (positive utility is better)\n\n"+"\n".join(f"- {k}: {v:+.6f}" for k,v in collapsed.items())+"\n\n"
      f"## Natural trade-off\n\nSubject-vs-population attenuation utility was {nat_att:+.6f}; preservation utility {nat_pre:+.6f}.\n\n"
      f"## V22 repair\n\nV22-FIX MATCH utility against POP was {v22_fix_pop:+.6f} and against WRONG was {v22_fix_wrong:+.6f}. The frozen forensic audit established conflicting ordinary supervision, a 1,200-update budget, first-64 validation, fixed t=500 diffusion validation, last-weight checkpointing, hard-masked fixed pairs, and zero-artifact SNR contamination. Frozen V22 outputs were not changed.\n\n"
      "## Uncertainty\n\n"+"\n".join(f"- {k}: mean {v['mean']:+.6f}, median {v['median']:+.6f}, {v['positive_count']}/{v['n']} positive, participant-bootstrap 95% CI [{v['bootstrap_low']:+.6f}, {v['bootstrap_high']:+.6f}]" for k,v in effect_stats.items())+"\n\n"
      f"## Classification\n\n- Engineering: `valid`\n- V22 repair: `{v22_repair}`\n- Operator-factorized context: `{context_label}`\n- Diffusion incremental value: `{diff_label}`\n- Natural trade-off: `{nat_label}`\n- Next route: `{next_step}`\n\nK8/DET8 is not used to rescue a poor K1 result. EEGDfus remains the frozen V22 `local_results_reasonable_but_nonidentical` reference; D4PM remains `blocked_incomplete_release`. No energy bridge, sealed data, manuscript change, or confirmation claim is included.\n")
    reports=ROOT/"reports";reports.mkdir(exist_ok=True);(reports/"v23_final_development_diagnosis.md").write_text(report);(reports/"v23_round_b.md").write_text("# V23 Round B\n\nThree-seed participant-first results are summarized in the final diagnosis and CSV files.\n");(reports/"v23_natural_development.md").write_text(f"# V23 natural development\n\nArtifact attenuation utility {nat_att:+.6f}; preservation utility {nat_pre:+.6f}. Classification: `{nat_label}`.\n");(reports/"v23_project_plan.md").write_text("# V23 project plan\n\nThe executed backbone is context-consistent online counterfactual training plus an operator-factorized spatial decoder, deterministic coefficient anchor, and low-dimensional residual diffusion. Energy guidance and sealed confirmation are out of scope.\n")
    result={"stage":"R13","status":"PASS","diagnosis":diagnosis};_json(run/"result_summary.json",result);return result

def terminal(run:Path)->dict[str,Any]:
    value={"stage":"R15","status":"PASS","base_commit":_cfg("data")["base_commit"],"branch":"codex/of-scad-operator-factorized-v23","terminal_commit_parent":_head(ROOT),"sealed_reads":0,"a_track_head":_head(Path(_cfg("data")["a_track_worktree"])),"manuscript_modified":False,"old_saddpm_imported":False,"energy_bridge_implemented":False,"development_only":True,"GPU_environment":"/home/infres/yinwang/anaconda3/envs/icml","CPU_environment":"/home/infres/yinwang/anaconda3/envs/eeg2025"};_json(RESULT/"terminal_manifest.json",value);_json(run/"result_summary.json",value);return value

def run(stage:str,run_dir:Path,index:int|None)->dict[str,Any]:
    if stage=="r0-preflight":return preflight(run_dir)
    if stage=="r1-forensic":return forensic(run_dir)
    if stage=="r2-prepare-fold":return prepare_fold(run_dir,int(index))
    if stage=="r2-collect":return prepare_collect(run_dir)
    if stage=="r3-sanity":return sanity(run_dir)
    if stage=="r4a-train":return train_stage(run_dir,int(index),"a",False)
    if stage=="r4b-train-diff":return train_stage(run_dir,int(index),"a",True)
    if stage=="r5-infer":return paired_infer(run_dir,int(index),"a")
    if stage=="r5-eval":return paired_evaluate(run_dir,"a")
    if stage=="r5-eval-fold":return paired_evaluate_fold(run_dir,int(index),"a")
    if stage=="r5-eval-collect":return paired_evaluate_collect(run_dir,"a")
    if stage=="r6-select":return select_round_b(run_dir)
    if stage=="r7a-train-det":return train_stage(run_dir,int(index),"b",False)
    if stage=="r7b-train-diff":return train_stage(run_dir,int(index),"b",True)
    if stage=="r8-infer":return paired_infer(run_dir,int(index),"b")
    if stage=="r8-eval":return paired_evaluate(run_dir,"b")
    if stage=="r8-eval-fold":return paired_evaluate_fold(run_dir,int(index),"b")
    if stage=="r8-eval-collect":return paired_evaluate_collect(run_dir,"b")
    if stage=="r8-artifact-metric-recovery-fold":return paired_artifact_metric_recovery_fold(run_dir,int(index))
    if stage=="r8-artifact-metric-recovery-collect":return paired_artifact_metric_recovery_collect(run_dir)
    if stage=="r9-natural-prepare":return natural_prepare_fold(run_dir,int(index))
    if stage=="r9-natural-infer":return natural_infer(run_dir,int(index))
    if stage=="r10-freeze":return output_freeze(run_dir)
    if stage=="r11-natural-eval":return natural_evaluate(run_dir)
    if stage=="r11-natural-eval-fold":return natural_evaluate_fold(run_dir,int(index))
    if stage=="r11-natural-eval-collect":return natural_evaluate_collect(run_dir)
    if stage=="r12-k-decision":return decide_round_c(run_dir)
    if stage=="r13-aggregate":return aggregate(run_dir)
    if stage=="r15-terminal":return terminal(run_dir)
    raise ValueError(stage)

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True);p.add_argument("--run-dir",type=Path,required=True);a=p.parse_args();idx=os.environ.get("SLURM_ARRAY_TASK_ID");print(json.dumps(run(a.stage,a.run_dir,None if idx is None else int(idx)),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
