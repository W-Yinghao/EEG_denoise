"""Capacity-matched diffusion and clean-neural prior development audits.

Inference stages never open evaluator arrays.  Scientific aggregation is
protocol -> seed -> participant, with participants as the sole sample unit.
"""
from __future__ import annotations

import csv,json,time
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping

import numpy as np
import yaml

from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_operator_shrinkage as shrink
from eeg_cgdr.experiments import bci2b_subject_diffusion_next as nxt

SAME=("same_01","same_02","same_03")

def _config(path:Path)->dict[str,Any]:return yaml.safe_load(path.read_text(encoding="utf-8"))
def _root(c:Mapping[str,Any],key:str)->Path:return Path(str(c[key]))
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0])
    for row in rows:
        for key in row:
            if key not in keys:keys.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,lineterminator="\n");writer.writeheader();writer.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))
def _task(c:Mapping[str,Any],index:int)->tuple[int,int]:
    seeds=list(map(int,c["seeds"]));return seeds[index//9],index%9
def _seed_fold(root:Path,seed:int,fold:int)->Path:return root/"seeds"/str(seed)/"folds"/f"fold_{fold:02d}"

def stage_audit(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    strict=_root(c,"strict_root");checkpoints=list(strict.glob("seeds/*/folds/*/checkpoint.pt"));units=[]
    for fold in range(9):units.extend(_read(strict/"prepared"/f"fold_{fold:02d}"/"unit_manifest.csv"))
    summary={"status":"FROZEN_CAPACITY_AND_NEURAL_PROTOCOL","strict_checkpoints":len(checkpoints),"eligible_protocol_units":sum(int(r["eligible_120"]) for r in units),"availability_denominator":len(units),"participants":9,"seeds":list(map(int,c["seeds"])),"aggregation":"protocol -> seed -> participant","primary_context":"POP8-R","a_track_touched":False,"development_only":True}
    if len(checkpoints)!=27 or summary["eligible_protocol_units"]!=26 or len(units)!=27:raise RuntimeError(summary)
    _json(_root(c,"result_root")/"frozen_protocol.json",summary);_json(run/"result_summary.json",summary);return summary

def _load_base(base:Path,device:Any):
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    cp=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**cp["config"]);det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(cp["det"]);diff.load_state_dict(cp["diff"]);ema=EMA(diff);ema.load_state_dict(cp["ema"]);ema.copy_to(diff);det.eval();diff.eval();return cp,det,diff,cfg

def _train_det2(c:Mapping[str,Any],seed:int,fold:int,device:Any,*,technical:bool=False)->dict[str,Any]:
    import torch
    from torch.optim import AdamW
    from eeg_cgdr.models.eog_residual_diffusion import CapacityMatchedDeterministic,EMA
    torch.manual_seed(seed+fold);torch.cuda.manual_seed_all(seed+fold);np.random.seed(seed+fold)
    base=_seed_fold(_root(c,"strict_root"),seed,fold);cp,det,diff,cfg=_load_base(base,device)
    with np.load(base/"training_pairs.npz") as data:
        y=np.asarray(data["y"],np.float32);eog=np.asarray(data["eog"],np.float32);a=np.asarray(data["a"],np.float32);hsub=np.asarray(data["h_subject"],np.float32);hpop=np.asarray(data["h_population"],np.float32)
    if technical:y=y[:8];eog=eog[:8];a=a[:8];hsub=hsub[:8]
    # Base checkpoint loading instantiates modules, so reseed immediately before
    # DET2 construction to make its initialization independent of that detail.
    torch.manual_seed(seed+fold);torch.cuda.manual_seed_all(seed+fold)
    model=CapacityMatchedDeterministic(cfg).to(device);parameters_diff=sum(p.numel() for p in diff.backbone.parameters());parameters_det2=sum(p.numel() for p in model.backbone.parameters())
    if parameters_diff!=parameters_det2:raise RuntimeError("DET2/diffusion parameter mismatch")
    # Recreate the frozen diffusion-stage batch/context schedule exactly.
    rng=np.random.default_rng(seed+fold);updates=int(c["technical_updates"] if technical else c["training_updates"]);batch=int(c["batch_size"]);schedule=[]
    for _ in range(updates):
        idx=np.arange(len(y)) if technical else rng.integers(0,len(y),batch);choice=np.zeros(len(idx),bool) if technical else rng.random(len(idx))<.5;schedule.append((idx,choice))
    opt=AdamW(model.parameters(),lr=float(c["learning_rate"]));ema=EMA(model,float(c["ema_decay"]));rescale=np.asarray(cp["residual_scale"],np.float32);curve=[];started=time.monotonic()
    for step,(idx,choice) in enumerate(schedule):
        h=np.where(choice[:,None,None],hsub[idx],hpop[None]);a0=v11.apply_transfer(h,eog[idx]);yt=torch.as_tensor(y[idx],device=device);et=torch.as_tensor(eog[idx],device=device);at=torch.as_tensor(a0,device=device);target=torch.as_tensor((a[idx]-a0)/rescale[None,:,None],device=device)
        with torch.no_grad():rd=det(y=yt,eog=et,a0=at)
        truth=target-rd;model.train();opt.zero_grad(set_to_none=True);pred=model(y=yt,eog=et,a0=at,r_det=rd);loss=(pred-truth).square().mean();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));opt.step();ema.update(model)
        if step%100==0 or step+1==updates:curve.append({"step":step+1,"loss":float(loss.detach()),"gradient_norm":grad})
    elapsed=time.monotonic()-started;raw_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()};ema.copy_to(model);model.eval();out=_root(c,"capacity_root")/("technical" if technical else "det2")/str(seed)/f"fold_{fold:02d}";out.mkdir(parents=True,exist_ok=True);torch.save({"config":cp["config"],"raw_det2":raw_state,"det2":model.state_dict(),"ema":ema.state_dict(),"seed":seed,"fold":fold,"updates":updates,"residual_scale":rescale},out/"checkpoint.pt");_csv(out/"training_curve.csv",curve)
    # Fixed real-batch overfit/reload and deterministic replay checks.
    idx=np.arange(min(32,len(y)));a0=v11.apply_transfer(hpop,eog[idx]);yt=torch.as_tensor(y[idx],device=device);et=torch.as_tensor(eog[idx],device=device);at=torch.as_tensor(a0,device=device)
    with torch.no_grad():rd=det(y=yt,eog=et,a0=at);p1=model(y=yt,eog=et,a0=at,r_det=rd);p2=model(y=yt,eog=et,a0=at,r_det=rd)
    saved=torch.load(out/"checkpoint.pt",map_location=device,weights_only=False);reload=CapacityMatchedDeterministic(cfg).to(device);reload.load_state_dict(saved["det2"]);reload.eval();raw=CapacityMatchedDeterministic(cfg).to(device);raw.load_state_dict(saved["raw_det2"]);raw.eval()
    with torch.no_grad():p3=reload(y=yt,eog=et,a0=at,r_det=rd);praw=raw(y=yt,eog=et,a0=at,r_det=rd)
    truth=torch.as_tensor((a[idx]-a0)/rescale[None,:,None],device=device)-rd;rr=float(torch.linalg.vector_norm(p1-truth)/torch.linalg.vector_norm(truth).clamp_min(1e-12));corr=float(np.corrcoef(p1.cpu().numpy().ravel(),truth.cpu().numpy().ravel())[0,1]);exact=bool(torch.equal(p1,p2) and torch.equal(p1,p3))
    raw_rr=float(torch.linalg.vector_norm(praw-truth)/torch.linalg.vector_norm(truth).clamp_min(1e-12));raw_corr=float(np.corrcoef(praw.cpu().numpy().ravel(),truth.cpu().numpy().ravel())[0,1]);metrics={"status":"DET2_TRAINED","seed":seed,"fold":fold,"technical":technical,"updates":updates,"parameters_det2":parameters_det2,"parameters_diffusion":parameters_diff,"parameter_difference":parameters_det2-parameters_diff,"training_seconds":elapsed,"fixed_batch_rrmse":rr,"fixed_batch_correlation":corr,"raw_fixed_batch_rrmse":raw_rr,"raw_fixed_batch_correlation":raw_corr,"scientific_checkpoint":"EMA","deterministic_replay_exact":exact,"checkpoint_reload_exact":bool(torch.equal(p1,p3)),"training_schedule":"exact frozen diffusion batch/context schedule","evaluator_opened":False}
    _json(out/"metrics.json",metrics);return metrics

def stage_technical(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    result=_train_det2(c,int(c["seeds"][0]),0,torch.device("cuda"),technical=True);overfit=(result["fixed_batch_rrmse"]<=.10 and result["fixed_batch_correlation"]>=.95) or (result["raw_fixed_batch_rrmse"]<=.10 and result["raw_fixed_batch_correlation"]>=.95);passed=result["parameter_difference"]==0 and result["deterministic_replay_exact"] and overfit
    result.update({"technical_validity_passed":bool(passed),"fixed_batch_is_technical_only":True});_json(_root(c,"capacity_root")/"technical_validity.json",result);_json(run/"result_summary.json",result)
    if not passed:raise RuntimeError(result)
    return result

def stage_train_det2(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    if not json.loads((_root(c,"capacity_root")/"technical_validity.json").read_text())["technical_validity_passed"]:raise RuntimeError("technical gate failed")
    seed,fold=_task(c,task_index);result=_train_det2(c,seed,fold,torch.device("cuda"));_json(run/"result_summary.json",result);return result

def _first_x0(diff:Any,*,y:Any,eog:Any,a0:Any,r_det:Any,initial_noise:Any)->Any:
    import torch
    state=initial_noise;t=torch.full((len(y),),diff.config.timesteps-1,device=y.device,dtype=torch.long);alpha=diff.alpha_bar.gather(0,t).reshape(len(y),1,1);v=diff.backbone(state,t,y=y,eog=eog,a0=a0,r_det=r_det);return alpha.sqrt()*state-(1-alpha).sqrt()*v

def stage_infer(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import CapacityMatchedDeterministic
    seed,fold=_task(c,task_index);base=_seed_fold(_root(c,"strict_root"),seed,fold);cp,det,diff,cfg=_load_base(base,torch.device("cuda"));d2path=_root(c,"capacity_root")/"det2"/str(seed)/f"fold_{fold:02d}"/"checkpoint.pt";d2=CapacityMatchedDeterministic(cfg).cuda();d2.load_state_dict(torch.load(d2path,map_location="cuda",weights_only=False)["det2"]);d2.eval();rescale=np.asarray(cp["residual_scale"],np.float32);rows=[]
    for unit_index,protocol in enumerate(SAME):
        inf=np.load(base/"units"/protocol/"inference.npz");folder=_root(c,"capacity_root")/"inference"/str(seed)/f"fold_{fold:02d}"/protocol;folder.mkdir(parents=True,exist_ok=True)
        if not int(inf["recipient_eligible"]):np.savez_compressed(folder/"inference_outputs.npz");continue
        hpop=np.asarray(inf["h_pop"]);manifest=_read(base/"unit_manifest.csv")[unit_index];lam,_,hm=shrink._select_lambda(c,fold+1,manifest["support_session"],inf,hpop);hmatch=hpop+lam*(hm-hpop);outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(inf[f"{panel}_y"],np.float32);eog=np.asarray(inf[f"{panel}_eog"],np.float32);gamma=float(inf["gamma"]);yt=torch.as_tensor(y,device="cuda");et=torch.as_tensor(eog,device="cuda");bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8)
            for context,h in (("POP8",hpop),("SHRINK-MATCH",hmatch)):
                a0=v11.apply_transfer(h,eog);at=torch.as_tensor(a0,device="cuda")
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
                with torch.no_grad():rd=det(y=yt,eog=et,a0=at)
                torch.cuda.synchronize();det1_seconds=time.monotonic()-started;started=time.monotonic()
                with torch.no_grad():delta2=d2(y=yt,eog=et,a0=at,r_det=rd)
                torch.cuda.synchronize();det2_stage_seconds=time.monotonic()-started;det2_seconds=det1_seconds+det2_stage_seconds;n0=torch.as_tensor(bank[0],device="cuda");started=time.monotonic()
                with torch.no_grad():d1=_first_x0(diff,y=yt,eog=et,a0=at,r_det=rd,initial_noise=n0)
                torch.cuda.synchronize();diff1_seconds=time.monotonic()-started;started=time.monotonic()
                with torch.no_grad():d25=diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=n0)
                torch.cuda.synchronize();diff25_seconds=time.monotonic()-started;started=time.monotonic()
                with torch.no_grad():samples=[d25]+[diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(n,device="cuda")) for n in bank[1:]]
                torch.cuda.synchronize();diffk8_seconds=diff25_seconds+(time.monotonic()-started);peak_mb=torch.cuda.max_memory_allocated()/2**20
                methods={"DET1":rd.cpu().numpy(),"DET2":(rd+delta2).cpu().numpy(),"DIFF1-K1":(rd+d1).cpu().numpy(),"DIFF25-K1":(rd+d25).cpu().numpy(),"DIFF25-K8":(rd+torch.stack(samples).mean(0)).cpu().numpy()}
                for name,residual in methods.items():
                    correction=a0+residual*rescale[None,:,None];correction[...,500:]=0;outputs[f"{panel}_{name}-{context}"]=v11.gamma_correction(y,correction,gamma)
        np.savez_compressed(folder/"inference_outputs.npz",**outputs);rows.append({"seed":seed,"fold":fold,"participant":fold+1,"protocol":protocol,"lambda":lam,"DET1_seconds_last_panel":det1_seconds,"DET2_cascade_seconds_last_panel":det2_seconds,"DIFF1_K1_cascade_seconds_last_panel":det1_seconds+diff1_seconds,"DIFF25_K1_cascade_seconds_last_panel":det1_seconds+diff25_seconds,"DIFF25_K8_cascade_seconds_last_panel":det1_seconds+diffk8_seconds,"peak_gpu_memory_mb":peak_mb,"K8_calls":200,"K1_calls":25,"DDIM1_calls":1,"evaluator_opened":0})
    _csv(_root(c,"capacity_root")/"inference_manifest"/f"seed_{seed}_fold_{fold:02d}.csv",rows);summary={"status":"CAPACITY_EVALUATOR_BLIND_INFERENCE_COMPLETED","seed":seed,"fold":fold,"units":len(rows)};_json(run/"result_summary.json",summary);return summary

def stage_evaluate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    seed,fold=_task(c,task_index);base=_seed_fold(_root(c,"strict_root"),seed,fold);source=_root(c,"strict_root")/"prepared"/f"fold_{fold:02d}";outputs=_root(c,"capacity_root")/"inference"/str(seed)/f"fold_{fold:02d}";paired,natural=nxt._evaluate_outputs(c,base,source,outputs,seed,fold,"CAPACITY")
    root=_root(c,"capacity_root")/"evaluation";_csv(root/f"seed_{seed}_fold_{fold:02d}_paired.csv",paired);_csv(root/f"seed_{seed}_fold_{fold:02d}_natural.csv",natural);summary={"status":"CAPACITY_INDEPENDENT_EVALUATION_COMPLETED","seed":seed,"fold":fold,"paired_rows":len(paired),"natural_rows":len(natural)};_json(run/"result_summary.json",summary);return summary

def _sign_flip(values:np.ndarray,*,one_sided:bool=False)->float:
    signs=((np.arange(2**len(values))[:,None]>>np.arange(len(values)))&1)*2-1;rep=(signs*values[None]).mean(1);observed=float(values.mean());return float(np.mean(rep>=observed)) if one_sided else float(np.mean(np.abs(rep)>=abs(observed)))

def _effect(c:Mapping[str,Any],rows:list[dict[str,Any]],key:str)->dict[str,Any]:
    values=np.asarray([np.mean([float(r[key]) for r in rows if int(r["participant"])==p]) for p in range(1,10)]);rng=np.random.default_rng(int(c["bootstrap_seed"]));idx=rng.integers(0,9,size=(int(c["bootstrap_replicates"]),9));rep=values[idx].mean(1)
    return {"mean":float(values.mean()),"median":float(np.median(values)),"positive":int(np.sum(values>0)),"participant_values":values.tolist(),"seed_means":[float(np.mean([float(r[key]) for r in rows if int(r["seed"])==s])) for s in map(int,c["seeds"])],"two_sided_exact_sign_flip":_sign_flip(values),"descriptive_ci":[float(np.quantile(rep,.025)),float(np.quantile(rep,.975))],"leave_one_participant_out":[float(np.delete(values,i).mean()) for i in range(9)]}

def stage_aggregate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=_root(c,"capacity_root");paired=[];natural=[]
    for seed in map(int,c["seeds"]):
        for fold in range(9):paired.extend(_read(root/"evaluation"/f"seed_{seed}_fold_{fold:02d}_paired.csv"));natural.extend(_read(root/"evaluation"/f"seed_{seed}_fold_{fold:02d}_natural.csv"))
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);effects=[]
    for seed in map(int,c["seeds"]):
        for participant in range(1,10):
            take=[r for r in paired if int(r["seed"])==seed and int(r["participant"])==participant];by=defaultdict(list)
            for r in take:by[r["method"]].append(float(r["rrmse"]))
            needed=("DET2-POP8","DIFF1-K1-POP8","DIFF25-K1-POP8","DIFF25-K8-POP8")
            if not all(x in by for x in needed):continue
            row={"seed":seed,"participant":participant,"E_cap":float(np.mean(by["DET2-POP8"])-np.mean(by["DIFF25-K8-POP8"])),"E_iter":float(np.mean(by["DIFF1-K1-POP8"])-np.mean(by["DIFF25-K1-POP8"])),"E_avg":float(np.mean(by["DIFF25-K1-POP8"])-np.mean(by["DIFF25-K8-POP8"]))}
            if "DIFF25-K8-SHRINK-MATCH" in by and "DET2-SHRINK-MATCH" in by:
                row.update({"E_P_DIFF":float(np.mean(by["DIFF25-K8-POP8"])-np.mean(by["DIFF25-K8-SHRINK-MATCH"])),"E_P_DET2":float(np.mean(by["DET2-POP8"])-np.mean(by["DET2-SHRINK-MATCH"]))});row["DeltaSA2"]=row["E_P_DIFF"]-row["E_P_DET2"]
            effects.append(row)
    _csv(root/"participant_seed_effects.csv",effects);participant_rows=[{"participant":p,**{key:float(np.mean([r[key] for r in effects if r["participant"]==p and key in r])) for key in ("E_cap","E_iter","E_avg","E_P_DIFF","E_P_DET2","DeltaSA2")}} for p in range(1,10)];_csv(root/"participant_effects.csv",participant_rows)
    summaries={key:_effect(c,effects,key) for key in ("E_cap","E_iter","E_avg","E_P_DIFF","E_P_DET2","DeltaSA2")}
    safety=[]
    for method in sorted({r["method"] for r in natural}):
        for participant in range(1,10):
            take=[r for r in natural if r["method"]==method and int(r["participant"])==participant]
            if take:safety.append({"method":method,"participant":participant,**{k:float(np.mean([float(r[k]) for r in take])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}})
    _csv(root/"participant_natural_safety.csv",safety)
    def mean_safety(method:str)->dict[str,float]:return {k:float(np.mean([r[k] for r in safety if r["method"]==method])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}
    sd=mean_safety("DIFF25-K8-POP8");s2=mean_safety("DET2-POP8");margin={k:sd[k]-s2[k] for k in sd};ec=summaries["E_cap"];ei=summaries["E_iter"]
    cap=ec["mean"]>=.005 and ec["median"]>0 and ec["positive"]>=7 and all(x>0 for x in ec["seed_means"]) and ec["two_sided_exact_sign_flip"]<.05 and ec["descriptive_ci"][0]>0 and margin["preservation"]>=-.02 and margin["covariance"]<=.02 and margin["mi_kappa"]>=-.02
    iterative=cap and ei["mean"]>0 and ei["median"]>0
    status="ITERATIVE_DIFFUSION_INCREMENT_SUPPORTED" if iterative else ("TWO_STAGE_DIFFUSION_TRAINED_ESTIMATOR_SUPPORTED" if cap else "EXTRA_CAPACITY_CONFOUNDED")
    compute=[]
    for path in sorted((root/"det2").glob("*/fold_*/metrics.json")):
        item=json.loads(path.read_text());compute.append({k:item[k] for k in ("seed","fold","parameters_det2","parameters_diffusion","parameter_difference","updates","training_seconds")})
    _csv(root/"training_compute_by_fold.csv",compute);runtime=[]
    for path in sorted((root/"inference_manifest").glob("*.csv")):runtime.extend(_read(path))
    parameter_count=int(compute[0]["parameters_diffusion"]);method_compute=[
        {"method":"DET2-POP8","parameters_second_stage":parameter_count,"successful_updates":int(c["training_updates"]),"NFE_second_stage":1,"mean_seconds_per_last_natural_unit":float(np.mean([float(r["DET2_cascade_seconds_last_panel"]) for r in runtime])),"mean_peak_gpu_memory_mb":float(np.mean([float(r["peak_gpu_memory_mb"]) for r in runtime]))},
        {"method":"DIFF-POP8-DDIM1-K1","parameters_second_stage":parameter_count,"successful_updates":int(c["training_updates"]),"NFE_second_stage":1,"mean_seconds_per_last_natural_unit":float(np.mean([float(r["DIFF1_K1_cascade_seconds_last_panel"]) for r in runtime])),"mean_peak_gpu_memory_mb":float(np.mean([float(r["peak_gpu_memory_mb"]) for r in runtime]))},
        {"method":"DIFF-POP8-DDIM25-K1","parameters_second_stage":parameter_count,"successful_updates":int(c["training_updates"]),"NFE_second_stage":25,"mean_seconds_per_last_natural_unit":float(np.mean([float(r["DIFF25_K1_cascade_seconds_last_panel"]) for r in runtime])),"mean_peak_gpu_memory_mb":float(np.mean([float(r["peak_gpu_memory_mb"]) for r in runtime]))},
        {"method":"DIFF-POP8-DDIM25-K8","parameters_second_stage":parameter_count,"successful_updates":int(c["training_updates"]),"NFE_second_stage":200,"mean_seconds_per_last_natural_unit":float(np.mean([float(r["DIFF25_K8_cascade_seconds_last_panel"]) for r in runtime])),"mean_peak_gpu_memory_mb":float(np.mean([float(r["peak_gpu_memory_mb"]) for r in runtime]))},
    ];_csv(root/"method_compute_manifest.csv",method_compute);route={"status":status,"capacity_matched_increment":bool(cap),"iterative_increment":bool(iterative),"posterior_averaging_effect":summaries["E_avg"],"old_E_D_status":"retained_only_as_capacity_confounded_historical_signal" if not cap else "capacity_confound_addressed","development_only":True};summary={"effects":summaries,"natural_diffusion":sd,"natural_det2":s2,"natural_margins_diff_minus_det2":margin,"routing":route,"availability":{"eligible_units":26,"denominator":27,"participants":9},"scientific_unit":"participant"};_json(root/"result_summary.json",summary);_json(root/"route_decision.json",route)
    figdir=root/"figures";figdir.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(7,4));x=np.arange(1,10)
    for key,marker in (("E_cap","o"),("E_iter","s"),("E_avg","^")):ax.plot(x,[r[key] for r in participant_rows],marker+"-",label=key)
    ax.axhline(0,color="black",lw=.8);ax.set(xlabel="Participant",ylabel="Positive RRMSE utility");ax.legend();fig.tight_layout();fig.savefig(figdir/"capacity_effects.png",dpi=180);plt.close(fig)
    Path("reports/diffusion_capacity_matched_audit.md").write_text("# Capacity-matched diffusion audit\n\n"+f"Decision: `{status}`. This is development evidence with n=9 participants; protocol units and seeds were aggregated within participant.\n\n"+"\n".join(f"- {k}: mean {v['mean']:+.5f}, median {v['median']:+.5f}, {v['positive']}/9, exact two-sided p={v['two_sided_exact_sign_flip']:.6f}, descriptive CI {v['descriptive_ci']}." for k,v in summaries.items())+"\n",encoding="utf-8");_json(run/"result_summary.json",summary);return summary

# ---------------------------------------------------------------------------
# Clean-neural support-only headroom.  Support extraction and later-query
# evaluation are physically separate Slurm stages.

def _bandpass(eeg:np.ndarray,sfreq:float)->np.ndarray:
    from scipy.signal import butter,sosfiltfilt
    sos=butter(4,(1,45),btype="bandpass",fs=sfreq,output="sos");return sosfiltfilt(sos,eeg,axis=-1).astype(np.float64)

def _covariance(x:np.ndarray)->np.ndarray:
    x=np.asarray(x,float);center=np.median(x,axis=1,keepdims=True);mad=np.median(np.abs(x-center),axis=1,keepdims=True)/.67448975;z=np.clip((x-center)/np.maximum(mad,1e-12),-8,8);cov=np.cov(z);target=np.trace(cov)/len(cov);return (.9*cov+.1*target*np.eye(len(cov))).astype(np.float64)

def _log_spd(cov:np.ndarray)->np.ndarray:
    values,vectors=np.linalg.eigh(cov);return (vectors*np.log(np.maximum(values,1e-10)))@vectors.T

def _airm(left:np.ndarray,right:np.ndarray)->float:
    from scipy.linalg import eigvalsh
    # The generalized symmetric eigensystem is required here.  Applying
    # np.linalg.eigvalsh to left^{-1}right is invalid because that product is
    # generally not symmetric in Euclidean coordinates.
    values=eigvalsh(right,left,check_finite=True);return float(np.linalg.norm(np.log(np.maximum(values,1e-10))))

def _logeuclidean(left:np.ndarray,right:np.ndarray)->float:return float(np.linalg.norm(_log_spd(left)-_log_spd(right)))

def _psd_profile(x:np.ndarray,sfreq:float)->np.ndarray:
    from scipy.signal import welch
    f,p=welch(x,fs=sfreq,nperseg=min(500,x.shape[-1]),axis=-1);keep=(f>=1)&(f<=45);return np.log(np.maximum(p[:,keep],1e-20)).mean(1)

def _clean_support(eeg:np.ndarray,eog:np.ndarray,sfreq:float,region:slice,seconds:float)->tuple[np.ndarray,np.ndarray,float]:
    available=(region.stop-region.start)/sfreq
    if available<seconds:return np.empty((eeg.shape[0],0)),np.empty((eog.shape[0],0)),available
    stop=region.start+int(round(seconds*sfreq));eeg=eeg[:,region.start:stop];eog=eog[:,region.start:stop];window=int(round(2*sfreq));starts=range(0,eeg.shape[1]-window+1,window);energy=np.asarray([np.sqrt(np.mean(eog[:,s:s+window].astype(float)**2)) for s in starts]);threshold=np.quantile(energy,.95);kept=[eeg[:,s:s+window] for s,q in zip(starts,energy) if q<=threshold];return np.concatenate(kept,axis=1),eog,available

def _neural_session(c:Mapping[str,Any],dataset:str,subject:int,session:str):
    if dataset=="bci2a":
        from eeg_cgdr.data.bci2a_v10 import discover_sessions,load_with_events
        from eeg_cgdr.experiments.bci2a_hierarchical_score_v10 import _support_query_ranges
        found={(x.subject,x.session):x for x in discover_sessions(_root(c,"data_root"))};eeg,eog,sfreq,events=load_with_events(found[(subject,session)]);support,query=_support_query_ranges(events,eeg.shape[1],sfreq);return eeg,eog,sfreq,support,query
    eeg,eog,sfreq,events=v11._load_session(c,subject,session);support,query=v11._support_query_ranges(events,eeg.shape[1],sfreq);return eeg,eog,sfreq,support,query

def stage_neural_extract(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    dataset=("bci2a","bci2b")[task_index//9];subject=task_index%9+1;sessions=("T","E") if dataset=="bci2a" else ("01T","02T","03T");arrays={};coverage=[]
    for session in sessions:
        eeg,eog,sfreq,support,_=_neural_session(c,dataset,subject,session);clean,_,available=_clean_support(eeg,eog,sfreq,support,float(c["neural_support_seconds"]));eligible=clean.shape[1]>0;coverage.append({"dataset":dataset,"participant":subject,"session":session,"available_seconds":available,"eligible":int(eligible),"support_eog_used_only_for_gross_exclusion":1})
        if not eligible:continue
        filtered=_bandpass(clean,sfreq);halves=np.array_split(filtered,2,axis=1);cov=_covariance(filtered);c1=_covariance(halves[0]);c2=_covariance(halves[1]);off=np.triu_indices(len(cov),1);r1=np.corrcoef(c1);r2=np.corrcoef(c2);reliability=float(np.corrcoef(r1[off],r2[off])[0,1]) if len(off[0])>=3 else float(np.corrcoef(c1.ravel(),c2.ravel())[0,1]);arrays[f"{session}_cov"]=cov;arrays[f"{session}_psd"]=_psd_profile(filtered,sfreq);arrays[f"{session}_reliability"]=np.array(reliability);arrays[f"{session}_sfreq"]=np.array(sfreq)
    root=_root(c,"neural_root")/dataset;(root/"support").mkdir(parents=True,exist_ok=True);np.savez_compressed(root/"support"/f"participant_{subject:02d}.npz",**arrays);_csv(root/"coverage"/f"participant_{subject:02d}.csv",coverage);summary={"status":"CLEAN_NEURAL_SUPPORT_EXTRACTED","dataset":dataset,"participant":subject,"eligible_sessions":sum(r["eligible"] for r in coverage),"query_opened":False};_json(run/"result_summary.json",summary);return summary

def stage_neural_evaluate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    dataset=("bci2a","bci2b")[task_index//9];subject=task_index%9+1;sessions=("T","E") if dataset=="bci2a" else ("01T","02T","03T");root=_root(c,"neural_root")/dataset;support={p:np.load(root/"support"/f"participant_{p:02d}.npz") for p in range(1,10)};rows=[]
    try:
        for session in sessions:
            key=f"{session}_cov"
            if key not in support[subject]:continue
            eeg,eog,sfreq,_,query=_neural_session(c,dataset,subject,session);window=int(round(2*sfreq));starts=list(range(query.start,query.stop-window+1,window));energy=np.asarray([np.sqrt(np.mean(eog[:,s:s+window].astype(float)**2)) for s in starts]);threshold=np.quantile(energy,.3);kept=[eeg[:,s:s+window] for s,q in zip(starts,energy) if q<=threshold]
            if not kept:continue
            query_x=_bandpass(np.concatenate(kept,axis=1),sfreq);cq=_covariance(query_x);pq=_psd_profile(query_x,sfreq);owners=[p for p in range(1,10) if p!=subject and key in support[p]];c0=np.mean([support[p][key] for p in owners],axis=0);p0=np.mean([support[p][f"{session}_psd"] for p in owners],axis=0);cs=np.asarray(support[subject][key]);ps=np.asarray(support[subject][f"{session}_psd"])
            pop_airm=_airm(cq,c0);match_airm=_airm(cq,cs);pop_log=_logeuclidean(cq,c0);match_log=_logeuclidean(cq,cs);wrong_airm=[_airm(cq,np.asarray(support[p][key])) for p in owners];wrong_log=[_logeuclidean(cq,np.asarray(support[p][key])) for p in owners];wrong_psd=[float(np.linalg.norm(pq-np.asarray(support[p][f"{session}_psd"]))) for p in owners]
            rows.append({"dataset":dataset,"participant":subject,"session":session,"eligible":1,"H_P_airm":pop_airm-match_airm,"H_W_airm":float(np.mean(wrong_airm))-match_airm,"H_P_logeuclidean":pop_log-match_log,"H_W_logeuclidean":float(np.mean(wrong_log))-match_log,"H_P_psd":float(np.linalg.norm(pq-p0)-np.linalg.norm(pq-ps)),"H_W_psd":float(np.mean(wrong_psd)-np.linalg.norm(pq-ps)),"support_reliability":float(support[subject][f"{session}_reliability"]),"wrong_donors":len(owners),"query_eog_used_evaluator_only":1})
    finally:
        for item in support.values():item.close()
    _csv(root/"evaluation"/f"participant_{subject:02d}.csv",rows);summary={"status":"CLEAN_NEURAL_LATER_QUERY_EVALUATED","dataset":dataset,"participant":subject,"rows":len(rows)};_json(run/"result_summary.json",summary);return summary

def stage_neural_aggregate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=_root(c,"neural_root");all_summary={};authorized=None
    for dataset in ("bci2a","bci2b"):
        folder=root/dataset;rows=[];coverage=[]
        for path in sorted((folder/"evaluation").glob("participant_*.csv")):rows.extend(_read(path))
        for path in sorted((folder/"coverage").glob("participant_*.csv")):coverage.extend(_read(path))
        participant=[]
        for p in range(1,10):
            take=[r for r in rows if int(r["participant"])==p]
            participant.append({"dataset":dataset,"participant":p,**{key:float(np.mean([float(r[key]) for r in take])) if take else float("nan") for key in ("H_P_airm","H_W_airm","H_P_logeuclidean","H_W_logeuclidean","H_P_psd","H_W_psd","support_reliability")}})
        _csv(folder/"participant_headroom.csv",participant);valid=[r for r in participant if np.isfinite(r["H_P_airm"])];metrics={}
        for key in ("H_P_airm","H_W_airm","H_P_logeuclidean","H_W_logeuclidean","H_P_psd","H_W_psd"):
            values=np.asarray([r[key] for r in valid]);metrics[key]={"mean":float(values.mean()),"median":float(np.median(values)),"positive":int(np.sum(values>0)),"participant_values":values.tolist(),"one_sided_exact_sign_flip":_sign_flip(values,one_sided=True)}
        reliability=np.asarray([r["support_reliability"] for r in valid]);loo=[float(np.delete(reliability,i).mean()) for i in range(len(reliability))];passed=len(valid)==9 and all(metrics[key]["mean"]>0 and metrics[key]["median"]>0 and metrics[key]["positive"]>=7 and metrics[key]["one_sided_exact_sign_flip"]<.05 for key in ("H_P_airm","H_W_airm","H_P_logeuclidean","H_W_logeuclidean")) and reliability.mean()>0 and min(loo)>0
        summary={"status":"CLEAN_NEURAL_COVARIANCE_HEADROOM_DETECTED" if passed else "CLEAN_NEURAL_COVARIANCE_HEADROOM_NOT_ESTABLISHED","metrics":metrics,"support_reliability":{"mean":float(reliability.mean()),"median":float(np.median(reliability)),"positive":int(np.sum(reliability>0)),"leave_one_participant_out":loo},"coverage":{"evaluated_participants":len(valid),"availability_denominator":9,"eligible_session_units":sum(int(r["eligible"]) for r in coverage),"session_denominator":len(coverage)},"aligned_diffusion_authorized":bool(passed),"development_only":True};_json(folder/"result_summary.json",summary);all_summary[dataset]=summary
        if passed and (authorized is None or dataset=="bci2a"):authorized=dataset
    route={"status":"CLEAN_NEURAL_ALIGNED_DIFFUSION_AUTHORIZED" if authorized else "CLEAN_NEURAL_SUBJECT_HEADROOM_NOT_ESTABLISHED","authorized_dataset":authorized,"bci2a_priority_enforced":True,"eog_transfer_personalization_family":"closed","development_only":True};_json(root/"route_decision.json",route);_json(root/"result_summary.json",{"datasets":all_summary,"routing":route})
    figdir=root/"figures";figdir.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,dataset in zip(axes,("bci2a","bci2b")):
        rows=_read(root/dataset/"participant_headroom.csv");x=np.arange(1,10);ax.axhline(0,color="black",lw=.8);ax.plot(x,[float(r["H_P_airm"]) for r in rows],"o-",label="MATCH−POP");ax.plot(x,[float(r["H_W_airm"]) for r in rows],"s-",label="MATCH−WRONG");ax.set_title(dataset.upper());ax.set_xlabel("Participant")
    axes[0].set_ylabel("Positive AIRM headroom");axes[1].legend();fig.tight_layout();fig.savefig(figdir/"clean_neural_headroom.png",dpi=180);plt.close(fig)
    lines=["# Clean-neural subject headroom","",f"Routing: `{route['status']}`; authorized dataset: `{authorized}`. Support EOG was used only to exclude gross ocular support windows and was not a conditioning feature. Query EOG and later-query covariance were opened only by the evaluator.",""]
    for dataset,summary in all_summary.items():
        lines.append(f"## {dataset.upper()}");lines.append(f"Decision: `{summary['status']}`. Evaluated {summary['coverage']['evaluated_participants']}/9 participants.")
        for key in ("H_P_airm","H_W_airm","H_P_logeuclidean","H_W_logeuclidean"):m=summary["metrics"][key];lines.append(f"- {key}: mean {m['mean']:+.5f}, median {m['median']:+.5f}, {m['positive']}/9, exact one-sided p={m['one_sided_exact_sign_flip']:.6f}.")
        lines.append("")
    Path("reports/clean_neural_subject_headroom.md").write_text("\n".join(lines)+"\n",encoding="utf-8");_json(run/"result_summary.json",{"datasets":all_summary,"routing":route});return route

def _invsqrt(cov:np.ndarray,epsilon:float=1e-3)->np.ndarray:
    values,vectors=np.linalg.eigh(np.asarray(cov,float)+float(epsilon)*np.eye(len(cov)));return ((vectors/np.sqrt(np.maximum(values,1e-8)))@vectors.T).astype(np.float32)

def _align(value:np.ndarray,matrix:np.ndarray)->np.ndarray:
    matrix=np.asarray(matrix)
    if matrix.ndim==2:return np.einsum("ij,njt->nit",matrix,value,optimize=True).astype(np.float32)
    return np.einsum("nij,njt->nit",matrix,value,optimize=True).astype(np.float32)

def _alignment_lookup(c:Mapping[str,Any],training:list[int])->tuple[dict[tuple[int,int],np.ndarray],np.ndarray]:
    root=_root(c,"neural_root")/"bci2b"/"support";covs=[];lookup={}
    for subject in training:
        with np.load(root/f"participant_{subject:02d}.npz") as data:
            for session in (1,2,3):
                key=f"{session:02d}T_cov"
                if key in data:lookup[(subject,session)]=np.asarray(data[key]);covs.append(np.asarray(data[key]))
    c0=np.mean(np.stack(covs),axis=0);return lookup,_invsqrt(c0)

def _train_neural_models(c:Mapping[str,Any],seed:int,fold:int,device:Any,*,technical:bool=False)->dict[str,Any]:
    import torch
    from torch.optim import AdamW
    from eeg_cgdr.models.eog_residual_diffusion import CapacityMatchedDeterministic,DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    route=json.loads((_root(c,"neural_root")/"route_decision.json").read_text())
    if route["authorized_dataset"]!="bci2b":raise RuntimeError("BCI2b clean-neural headroom did not authorize training")
    base=_seed_fold(_root(c,"strict_root"),seed,fold)
    with np.load(base/"training_pairs.npz") as data:
        y=np.asarray(data["y"],np.float32);eog=np.asarray(data["eog"],np.float32);a=np.asarray(data["a"],np.float32);subjects=np.asarray(data["subject"],int);sessions=np.asarray(data["session"],int);hpop=np.asarray(data["h_population"],np.float32)
    if technical:y=y[:32];eog=eog[:32];a=a[:32];subjects=subjects[:32];sessions=sessions[:32]
    training=[p for p in range(1,10) if p!=fold+1];lookup,a_pop=_alignment_lookup(c,training);row_match=np.stack([_invsqrt(lookup[(int(s),int(q))]) if (int(s),int(q)) in lookup else a_pop for s,q in zip(subjects,sessions)])
    torch.manual_seed(seed+fold);torch.cuda.manual_seed_all(seed+fold);np.random.seed(seed+fold);cfg=EOGResidualConfig();det1=DeterministicEOGResidual(cfg).to(device);det2=CapacityMatchedDeterministic(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);diff.backbone.load_state_dict(det2.backbone.state_dict())
    p2=sum(p.numel() for p in det2.backbone.parameters());pd=sum(p.numel() for p in diff.backbone.parameters())
    if p2!=pd:raise RuntimeError("neural DET2/diff capacity mismatch")
    updates=int(c["neural_technical_updates"] if technical else c["training_updates"]);batch=int(c["batch_size"]);rng=np.random.default_rng(seed+fold);schedule=[]
    for _ in range(updates):
        idx=np.arange(len(y)) if technical else rng.integers(0,len(y),batch);use_match=np.ones(len(idx),bool) if technical else rng.random(len(idx))<.5;schedule.append((idx,use_match))
    # Population anchor is fixed; only the neural coordinate system is episodic.
    a0_native=v11.apply_transfer(hpop,eog);matrices=np.where(np.ones((len(y),1,1),bool),row_match,a_pop)
    # Robust scale in aligned artifact-residual coordinates.
    aligned_a=_align(a,matrices);aligned_a0=_align(a0_native,matrices);rescale=np.maximum(np.quantile(np.abs(aligned_a-aligned_a0),.995,axis=(0,2)),1e-3).astype(np.float32)
    opts=[AdamW(det1.parameters(),lr=float(c["learning_rate"])),AdamW(det2.parameters(),lr=float(c["learning_rate"])),AdamW(diff.parameters(),lr=float(c["learning_rate"]))];ema2=EMA(det2,float(c["ema_decay"]));emad=EMA(diff,float(c["ema_decay"]));curve=[];generator=torch.Generator(device=device).manual_seed(seed+fold);started=time.monotonic()
    def batch_fields(idx,use_match):
        mats=np.where(use_match[:,None,None],row_match[idx],a_pop[None]);yz=_align(y[idx],mats);az=_align(a[idx],mats);a0z=_align(a0_native[idx],mats);return torch.as_tensor(yz,device=device),torch.as_tensor(eog[idx],device=device),torch.as_tensor(a0z,device=device),torch.as_tensor((az-a0z)/rescale[None,:,None],device=device)
    for step,(idx,use_match) in enumerate(schedule):
        yt,et,at,target=batch_fields(idx,use_match);det1.train();opts[0].zero_grad(set_to_none=True);rd=det1(y=yt,eog=et,a0=at);loss=(rd-target).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(det1.parameters(),1);opts[0].step()
        if step%200==0:curve.append({"phase":"DET1","step":step+1,"loss":float(loss.detach())})
    det1.eval()
    for phase,model,opt,ema in (("DET2",det2,opts[1],ema2),("DIFF",diff,opts[2],emad)):
        for step,(idx,use_match) in enumerate(schedule):
            yt,et,at,target=batch_fields(idx,use_match)
            with torch.no_grad():rd=det1(y=yt,eog=et,a0=at)
            truth=target-rd;model.train();opt.zero_grad(set_to_none=True)
            if phase=="DET2":loss=(model(y=yt,eog=et,a0=at,r_det=rd)-truth).square().mean()
            else:loss,_=model.training_loss(truth,y=yt,eog=et,a0=at,r_det=rd,generator=generator)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ema.update(model)
            if step%200==0:curve.append({"phase":phase,"step":step+1,"loss":float(loss.detach())})
    ema2.copy_to(det2);emad.copy_to(diff);det1.eval();det2.eval();diff.eval();root=_root(c,"neural_diffusion_root")/("technical" if technical else "models")/str(seed)/f"fold_{fold:02d}";root.mkdir(parents=True,exist_ok=True);payload={"config":cfg.__dict__,"det1":det1.state_dict(),"det2":det2.state_dict(),"diff":diff.state_dict(),"residual_scale":rescale,"a_pop":a_pop,"h_population":hpop,"updates":updates,"seed":seed,"fold":fold};torch.save(payload,root/"checkpoint.pt");_csv(root/"training_curve.csv",curve)
    idx=np.arange(min(32,len(y)));use=np.ones(len(idx),bool);yt,et,at,target=batch_fields(idx,use)
    with torch.no_grad():rd=det1(y=yt,eog=et,a0=at);p2out=det2(y=yt,eog=et,a0=at,r_det=rd);noise=torch.randn(yt.shape,device=device,generator=generator);dout=diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=noise)
    def score(delta):
        estimate=rd+delta;rr=float(torch.linalg.vector_norm(estimate-target)/torch.linalg.vector_norm(target).clamp_min(1e-12));co=float(np.corrcoef(estimate.cpu().numpy().ravel(),target.cpu().numpy().ravel())[0,1]);return rr,co
    r2,c2=score(p2out);rdiff,cdiff=score(dout);identity=float(np.max(np.abs(_align(_align(y[idx],row_match[idx]),np.linalg.inv(row_match[idx]))-y[idx])));result={"status":"NEURAL_MODELS_TRAINED","seed":seed,"fold":fold,"technical":technical,"det2_rrmse":r2,"det2_correlation":c2,"diff_rrmse":rdiff,"diff_correlation":cdiff,"alignment_roundtrip_max_abs":identity,"parameters_det2":p2,"parameters_diffusion":pd,"updates":updates,"training_seconds":time.monotonic()-started,"evaluator_opened":False};_json(root/"metrics.json",result);return result

def stage_neural_technical(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    result=_train_neural_models(c,int(c["neural_seeds"][0]),0,torch.device("cuda"),technical=True);passed=result["alignment_roundtrip_max_abs"]<1e-4 and result["parameters_det2"]==result["parameters_diffusion"] and np.isfinite(result["diff_rrmse"]);result["technical_validity_passed"]=bool(passed);_json(_root(c,"neural_diffusion_root")/"technical_validity.json",result);_json(run/"result_summary.json",result)
    if not passed:raise RuntimeError(result)
    return result

def stage_neural_train(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    seeds=list(map(int,c["neural_seeds"]));seed=seeds[task_index//9];fold=task_index%9;result=_train_neural_models(c,seed,fold,torch.device("cuda"));_json(run/"result_summary.json",result);return result

def stage_neural_infer(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import CapacityMatchedDeterministic,DeterministicEOGResidual,EOGResidualConfig,EOGResidualDiffusion
    seeds=list(map(int,c["neural_seeds"]));seed=seeds[task_index//9];fold=task_index%9;modelroot=_root(c,"neural_diffusion_root")/"models"/str(seed)/f"fold_{fold:02d}";cp=torch.load(modelroot/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**cp["config"]);device=torch.device("cuda");det1=DeterministicEOGResidual(cfg).to(device);det2=CapacityMatchedDeterministic(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det1.load_state_dict(cp["det1"]);det2.load_state_dict(cp["det2"]);diff.load_state_dict(cp["diff"]);det1.eval();det2.eval();diff.eval();rescale=np.asarray(cp["residual_scale"],np.float32);a_pop=np.asarray(cp["a_pop"],np.float32)
    base=_seed_fold(_root(c,"strict_root"),seed,fold);supportroot=_root(c,"neural_root")/"bci2b"/"support";support={p:np.load(supportroot/f"participant_{p:02d}.npz") for p in range(1,10)};rows=[]
    try:
        for unit_index,protocol in enumerate(SAME):
            inf=np.load(base/"units"/protocol/"inference.npz");folder=_root(c,"neural_diffusion_root")/"inference"/str(seed)/f"fold_{fold:02d}"/protocol;folder.mkdir(parents=True,exist_ok=True)
            if not int(inf["recipient_eligible"]):np.savez_compressed(folder/"inference_outputs.npz");continue
            session=int(protocol[-2:]);key=f"{session:02d}T_cov";recipient=fold+1;amatch=_invsqrt(np.asarray(support[recipient][key]));donors=[p for p in range(1,10) if p!=recipient and key in support[p]];contexts=[("POP",a_pop),("MATCH",amatch)]+[(f"WRONG-{p}",_invsqrt(np.asarray(support[p][key]))) for p in donors];hpop=np.asarray(inf["h_pop"]);outputs={}
            for panel in ("paired","natural"):
                y=np.asarray(inf[f"{panel}_y"],np.float32);eog=np.asarray(inf[f"{panel}_eog"],np.float32);gamma=float(inf["gamma"]);a0native=v11.apply_transfer(hpop,eog);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8);outputs[f"{panel}_RAW"]=y;outputs[f"{panel}_LINEAR-POP"]=v11.gamma_correction(y,np.pad(a0native[...,:500],((0,0),(0,0),(0,12))),gamma)
                for name,matrix in contexts:
                    yz=_align(y,matrix);a0z=_align(a0native,matrix);yt=torch.as_tensor(yz,device=device);et=torch.as_tensor(eog,device=device);at=torch.as_tensor(a0z,device=device)
                    with torch.no_grad():rd=det1(y=yt,eog=et,a0=at);d2=det2(y=yt,eog=et,a0=at,r_det=rd);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rd,initial_noise=torch.as_tensor(n,device=device)) for n in bank]
                    inv=np.linalg.inv(matrix);detcorr=_align(a0z+(rd+d2).cpu().numpy()*rescale[None,:,None],inv);diffcorr=_align(a0z+(rd+torch.stack(samples).mean(0)).cpu().numpy()*rescale[None,:,None],inv);detcorr[...,500:]=0;diffcorr[...,500:]=0;outputs[f"{panel}_DET2-NEURAL-{name}"]=v11.gamma_correction(y,detcorr,gamma);outputs[f"{panel}_DIFF-NEURAL-{name}"]=v11.gamma_correction(y,diffcorr,gamma)
            np.savez_compressed(folder/"inference_outputs.npz",**outputs);rows.append({"seed":seed,"fold":fold,"participant":recipient,"protocol":protocol,"wrong_donors":";".join(map(str,donors)),"fixed_population_eog_operator":1,"evaluator_opened":0})
    finally:
        for item in support.values():item.close()
    _csv(_root(c,"neural_diffusion_root")/"inference_manifest"/f"seed_{seed}_fold_{fold:02d}.csv",rows);summary={"status":"NEURAL_ALIGNED_EVALUATOR_BLIND_INFERENCE_COMPLETED","seed":seed,"fold":fold,"units":len(rows)};_json(run/"result_summary.json",summary);return summary

def stage_neural_diff_evaluate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    seeds=list(map(int,c["neural_seeds"]));seed=seeds[task_index//9];fold=task_index%9;base=_seed_fold(_root(c,"strict_root"),seed,fold);source=_root(c,"strict_root")/"prepared"/f"fold_{fold:02d}";outputs=_root(c,"neural_diffusion_root")/"inference"/str(seed)/f"fold_{fold:02d}";paired,natural=nxt._evaluate_outputs(c,base,source,outputs,seed,fold,"NEURAL")
    root=_root(c,"neural_diffusion_root")/"evaluation";_csv(root/f"seed_{seed}_fold_{fold:02d}_paired.csv",paired);_csv(root/f"seed_{seed}_fold_{fold:02d}_natural.csv",natural);summary={"status":"NEURAL_ALIGNED_INDEPENDENT_EVALUATION_COMPLETED","seed":seed,"fold":fold,"paired_rows":len(paired),"natural_rows":len(natural)};_json(run/"result_summary.json",summary);return summary

def stage_neural_diff_aggregate(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=_root(c,"neural_diffusion_root");seeds=list(map(int,c["neural_seeds"]));paired=[];natural=[]
    for seed in seeds:
        for fold in range(9):paired.extend(_read(root/"evaluation"/f"seed_{seed}_fold_{fold:02d}_paired.csv"));natural.extend(_read(root/"evaluation"/f"seed_{seed}_fold_{fold:02d}_natural.csv"))
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);effects=[]
    for seed in seeds:
        for participant in range(1,10):
            take=[r for r in paired if int(r["seed"])==seed and int(r["participant"])==participant];by=defaultdict(list)
            for r in take:by[r["method"]].append(float(r["rrmse"]))
            needed=("DIFF-NEURAL-POP","DIFF-NEURAL-MATCH","DET2-NEURAL-POP","DET2-NEURAL-MATCH")
            if not all(x in by for x in needed):continue
            wrong=[np.mean(v) for k,v in by.items() if k.startswith("DIFF-NEURAL-WRONG-")];effects.append({"seed":seed,"participant":participant,"U_P":float(np.mean(by["DIFF-NEURAL-POP"])-np.mean(by["DIFF-NEURAL-MATCH"])),"U_W":float(np.mean(wrong)-np.mean(by["DIFF-NEURAL-MATCH"])),"U_D2":float(np.mean(by["DET2-NEURAL-MATCH"])-np.mean(by["DIFF-NEURAL-MATCH"])),"DeltaSA_neural":float((np.mean(by["DIFF-NEURAL-POP"])-np.mean(by["DIFF-NEURAL-MATCH"]))-(np.mean(by["DET2-NEURAL-POP"])-np.mean(by["DET2-NEURAL-MATCH"])))})
    _csv(root/"participant_seed_effects.csv",effects);summaries={k:_effect({**c,"seeds":seeds},effects,k) for k in ("U_P","U_W","U_D2","DeltaSA_neural")};participant=[{"participant":p,**{k:float(np.mean([r[k] for r in effects if r["participant"]==p])) for k in summaries}} for p in range(1,10)];_csv(root/"participant_effects.csv",participant)
    safety=[]
    for method in sorted({r["method"] for r in natural}):
        for p in range(1,10):
            take=[r for r in natural if r["method"]==method and int(r["participant"])==p]
            if take:safety.append({"method":method,"participant":p,**{k:float(np.mean([float(r[k]) for r in take])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}})
    _csv(root/"participant_natural_safety.csv",safety)
    def sm(method):return {k:float(np.mean([r[k] for r in safety if r["method"]==method])) for k in ("eog_attenuation","preservation","mi_band_distortion","covariance","mi_kappa","erd_preservation")}
    match=sm("DIFF-NEURAL-MATCH");pop=sm("DIFF-NEURAL-POP");margin={k:match[k]-pop[k] for k in match};raw=float(np.mean([float(r["rrmse"]) for r in paired if r["method"]=="RAW"]));match_rr=float(np.mean([float(r["rrmse"]) for r in paired if r["method"]=="DIFF-NEURAL-MATCH"]));up=summaries["U_P"];uw=summaries["U_W"];passed=up["mean"]>0 and up["median"]>0 and up["positive"]>=7 and up["two_sided_exact_sign_flip"]<.05 and uw["mean"]>0 and uw["median"]>0 and uw["positive"]>=7 and uw["two_sided_exact_sign_flip"]<.05 and match_rr<raw and margin["preservation"]>=-.02 and margin["mi_kappa"]>=-.02 and margin["erd_preservation"]>=-.02 and margin["mi_band_distortion"]<=.02 and margin["covariance"]<=.02
    route={"status":"CLEAN_NEURAL_ALIGNED_DIFFUSION_ONE_SEED_PASSED" if passed else "CLEAN_NEURAL_ALIGNED_DIFFUSION_INSTANCE_NO_GO","additional_seeds_authorized":bool(passed and len(seeds)==1),"U_D2_not_a_subject_gate":True,"DeltaSA_not_a_subject_gate":True,"development_only":True};summary={"effects":summaries,"natural_match":match,"natural_pop":pop,"natural_margins":margin,"raw_rrmse":raw,"diff_match_rrmse":match_rr,"routing":route};_json(root/"result_summary.json",summary);_json(root/"route_decision.json",route)
    figdir=root/"figures";figdir.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(7,4));x=np.arange(1,10);ax.axhline(0,color="black",lw=.8);ax.plot(x,[r["U_P"] for r in participant],"o-",label="U_P");ax.plot(x,[r["U_W"] for r in participant],"s-",label="U_W");ax.legend();ax.set(xlabel="Participant",ylabel="Positive RRMSE utility");fig.tight_layout();fig.savefig(figdir/"neural_aligned_effects.png",dpi=180);plt.close(fig)
    Path("reports/clean_neural_aligned_diffusion.md").write_text("# Clean-neural aligned diffusion\n\n"+f"Decision: `{route['status']}` on BCI2b same-session development data.\n\n"+"\n".join(f"- {k}: mean {v['mean']:+.5f}, median {v['median']:+.5f}, {v['positive']}/9, exact p={v['two_sided_exact_sign_flip']:.6f}." for k,v in summaries.items())+"\n\nThe EOG-transfer personalization family remains closed; all contexts used the same population EOG operator, so MATCH/POP/WRONG differed only in the support-derived clean-neural alignment.\n",encoding="utf-8");_json(run/"result_summary.json",summary);return summary

def stage_final(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:
    capacity=json.loads((_root(c,"capacity_root")/"result_summary.json").read_text());neural=json.loads((_root(c,"neural_root")/"result_summary.json").read_text());aligned_path=_root(c,"neural_diffusion_root")/"result_summary.json";aligned=json.loads(aligned_path.read_text()) if aligned_path.exists() else None;route={"eog_transfer_personalization_family":"closed","capacity_matched_diffusion_increment":capacity["routing"]["status"],"clean_neural_headroom":neural["routing"]["status"],"clean_neural_aligned_diffusion":aligned["routing"]["status"] if aligned else "NOT_RUN","clean_neural_diffusion_run":aligned is not None,"development_only":True,"family_wide_negative_forbidden":True};_json(_root(c,"result_root")/"route_decision.json",route);_json(_root(c,"result_root")/"result_summary.json",{"capacity":capacity,"neural_headroom":neural,"neural_aligned_diffusion":aligned,"routing":route});_json(run/"result_summary.json",route);return route

def _run_many(c:Mapping[str,Any],run:Path,name:str,fn:Any,count:int)->dict[str,Any]:
    for i in range(count):child=run/f"task_{i}";child.mkdir(parents=True,exist_ok=True);fn(c,i,child)
    result={"status":name,"tasks":count};_json(run/"result_summary.json",result);return result

def stage_evaluate_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:return _run_many(c,run,"CAPACITY_EVALUATION_COMPLETED",stage_evaluate,27)
def stage_neural_extract_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:return _run_many(c,run,"NEURAL_SUPPORT_EXTRACTION_COMPLETED",stage_neural_extract,18)
def stage_neural_evaluate_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:return _run_many(c,run,"NEURAL_EVALUATION_COMPLETED",stage_neural_evaluate,18)
def stage_neural_diff_evaluate_all(c:Mapping[str,Any],task_index:int,run:Path)->dict[str,Any]:return _run_many(c,run,"NEURAL_DIFFUSION_EVALUATION_COMPLETED",stage_neural_diff_evaluate,9*len(c["neural_seeds"]))

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    c=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True);stages={"audit":stage_audit,"technical":stage_technical,"train-det2":stage_train_det2,"infer":stage_infer,"evaluate":stage_evaluate,"evaluate-all":stage_evaluate_all,"aggregate":stage_aggregate,"neural-extract":stage_neural_extract,"neural-extract-all":stage_neural_extract_all,"neural-evaluate":stage_neural_evaluate,"neural-evaluate-all":stage_neural_evaluate_all,"neural-aggregate":stage_neural_aggregate,"neural-technical":stage_neural_technical,"neural-train":stage_neural_train,"neural-infer":stage_neural_infer,"neural-diff-evaluate":stage_neural_diff_evaluate,"neural-diff-evaluate-all":stage_neural_diff_evaluate_all,"neural-diff-aggregate":stage_neural_diff_aggregate,"final":stage_final}
    if stage not in stages:raise ValueError(stage)
    return stages[stage](c,task_index,run_dir)

__all__=["_airm","_covariance","_logeuclidean","_sign_flip","run_stage"]
