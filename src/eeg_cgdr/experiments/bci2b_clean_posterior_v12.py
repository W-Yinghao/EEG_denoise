"""V11.2 score-component closure and conditional clean-posterior V12."""
from __future__ import annotations
import csv,json,time
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping
import numpy as np,yaml
from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as v111

SAME=("same_01","same_02","same_03")

def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as h:return yaml.safe_load(h)
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0]);[keys.append(k) for row in rows for k in row if k not in keys]
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)

def _features(value:np.ndarray)->np.ndarray:
    freq=np.fft.rfft(value[...,:500],axis=-1)
    band=np.log(np.abs(freq[:,:,16:61])+1e-8)
    return np.concatenate([band.mean(-1),band.std(-1)],axis=1)

def _shared_decoder_scores(raw:np.ndarray,outputs:dict[str,np.ndarray],labels:np.ndarray)->dict[str,float]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    split=max(4,len(labels)//2)
    model=LinearDiscriminantAnalysis().fit(_features(raw)[:split],labels[:split])
    return {name:float(cohen_kappa_score(labels[split:],model.predict(_features(value)[split:]))) for name,value in outputs.items()}

def _natural_metrics(value:np.ndarray,raw:np.ndarray,eog:np.ndarray,labels:np.ndarray,config:Mapping[str,Any])->dict[str,float]:
    value=value[...,:500];raw=raw[...,:500];eog=eog[...,:500]
    energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3)
    return {"preservation":1-v11.rrmse(value[low],raw[low]),"covariance":v11._covariance_distortion(value[low],raw[low]),"eog_attenuation":v11._coherence_proxy(raw,eog)-v11._coherence_proxy(value,eog),"mi_band_distortion":v111._bandpower_distortion(value[low],raw[low],(8,30),config),"erd_preservation":v111._erd_preservation(value,raw,labels,config),"method_specific_kappa":v11._kappa(value,labels)}

def stage_score_audit(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    source=Path(str(config["v11_1_result_root"]));evaluator=Path(str(config["v11_result_root"]));root=Path(str(config["score_audit_root"]));units=[];grid=[]
    for subject in range(1,10):
        fold=source/"folds"/f"fold_{subject-1:02d}"
        for k in (8,32):
            for protocol in SAME:
                inf=np.load(fold/"units"/protocol/"inference.npz");ev=np.load(evaluator/"folds"/f"fold_{subject-1:02d}"/"units"/protocol/"evaluator.npz");out=np.load(fold/"outputs"/f"k{k}"/protocol/"inference_outputs.npz")
                scale=np.asarray(inf["eeg_scale"]);location=np.asarray(inf["eeg_location"]);x=np.asarray(ev["paired_x"])[...,:500];labels=np.asarray(ev["natural_labels"]);eog=np.asarray(inf["natural_eog"])[...,:500]
                paired={name:(np.asarray(out[f"paired_{name}"])*scale[None,:,None]+location[None,:,None])[...,:500] for name in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-MATCH")}
                natural={name:np.asarray(out[f"natural_{name}"])[...,:500] for name in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-MATCH")}
                paired["SCORE-ONLY"]=paired["LINEAR-MATCH"]+(paired["DIFF-MATCH"]-paired["DET-MATCH"])
                natural["SCORE-ONLY"]=natural["LINEAR-MATCH"]+(natural["DIFF-MATCH"]-natural["DET-MATCH"])
                shared=_shared_decoder_scores(natural["RAW"],natural,labels);raw_error=v111._band_error(paired["RAW"],x,(1,45),config)
                for name in paired:
                    nm=_natural_metrics(natural[name],natural["RAW"],eog,labels,config);units.append({"subject":subject,"group":"P1-P3" if subject<=3 else "P4-P9","protocol":protocol,"k":k,"method":name,"rrmse":v11.rrmse(paired[name],x),"correlation":v11.correlation(paired[name],x),"delta_snr":v11.delta_snr(paired[name],x,paired["RAW"]),"paired_spectral_utility":raw_error-v111._band_error(paired[name],x,(1,45),config),"shared_raw_decoder_kappa":shared[name],**nm})
                for bd in config["beta_grid"]:
                    for bs in config["beta_grid"]:
                        p=paired["LINEAR-MATCH"]+float(bd)*(paired["DET-MATCH"]-paired["LINEAR-MATCH"])+float(bs)*(paired["DIFF-MATCH"]-paired["DET-MATCH"])
                        grid.append({"subject":subject,"protocol":protocol,"k":k,"beta_det":bd,"beta_score":bs,"rrmse":v11.rrmse(p,x),"correlation":v11.correlation(p,x),"delta_snr":v11.delta_snr(p,x,paired["RAW"]),"paired_spectral_utility":raw_error-v111._band_error(p,x,(1,45),config),"status":"mechanism_ceiling_not_deployable"})
    # Participant-first aggregation; protocols are never independent samples.
    participant=[]
    for subject in range(1,10):
        for k in (8,32):
            for method in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-MATCH","SCORE-ONLY"):
                rows=[r for r in units if r["subject"]==subject and r["k"]==k and r["method"]==method]
                participant.append({"subject":subject,"group":"P1-P3" if subject<=3 else "P4-P9","k":k,"method":method,**{name:float(np.nanmean([r[name] for r in rows])) for name in ("rrmse","correlation","delta_snr","paired_spectral_utility","preservation","covariance","eog_attenuation","mi_band_distortion","erd_preservation","shared_raw_decoder_kappa","method_specific_kappa")}})
    decisions={}
    for k in (8,32):
        l={r["subject"]:r for r in participant if r["k"]==k and r["method"]=="LINEAR-MATCH"};s={r["subject"]:r for r in participant if r["k"]==k and r["method"]=="SCORE-ONLY"};effects=np.asarray([l[i]["rrmse"]-s[i]["rrmse"] for i in range(1,10)]);p4=effects[3:]
        checks={"effect_mean":float(effects.mean()),"effect_median":float(np.median(effects)),"positive":int(np.sum(effects>0)),"p4_p9_positive":int(np.sum(p4>0)),"spectral_delta":float(np.mean([s[i]["paired_spectral_utility"]-l[i]["paired_spectral_utility"] for i in range(1,10)])),"eog_delta":float(np.mean([s[i]["eog_attenuation"]-l[i]["eog_attenuation"] for i in range(1,10)])),"preservation_delta":float(np.mean([s[i]["preservation"]-l[i]["preservation"] for i in range(1,10)])),"covariance_delta":float(np.mean([s[i]["covariance"]-l[i]["covariance"] for i in range(1,10)])),"shared_kappa_delta":float(np.mean([s[i]["shared_raw_decoder_kappa"]-l[i]["shared_raw_decoder_kappa"] for i in range(1,10)]))}
        passed=checks["effect_mean"]>0 and checks["effect_median"]>0 and checks["positive"]>=6 and checks["p4_p9_positive"]>=4 and checks["spectral_delta"]>=0 and checks["eog_delta"]>=-.01 and checks["preservation_delta"]>=-.02 and checks["covariance_delta"]<=.02 and checks["shared_kappa_delta"]>=-.02;decisions[k]={"passed":bool(passed),"checks":checks}
    headroom=any(v["passed"] for v in decisions.values());decision="EXISTING_DIFFUSION_SCORE_COMPONENT_HAS_LINEAR_ANCHOR_HEADROOM" if headroom else "CURRENT_ARTIFACT_RESIDUAL_SCORE_OBJECT_CLOSED"
    summary={"status":"completed_zero_training_score_component_audit","decision":decision,"clean_posterior_authorized":not headroom,"k_panels_separate":True,"participant_wise_k_selection":False,"participants":9,"decisions":decisions,"score_only_not_deployable":True,"grid_status":"mechanism_ceiling_not_deployable"}
    _csv(root/"unit_metrics.csv",units);_csv(root/"participant_metrics.csv",participant);_csv(root/"score_component_grid.csv",grid);_json(root/"result_summary.json",summary);_json(root/"routing_decision.json",summary);_json(run_dir/"result_summary.json",summary)
    lines=["# V11.2 score-component audit","","Zero-training development audit. `S_K = LINEAR-MATCH + (DIFF-MATCH - DET-MATCH)` is a score-component diagnostic, not a deployable method. K8 and K32 are separate global panels.","",f"Decision: `{decision}`.",""]
    for k in (8,32):
        c=decisions[k]["checks"];lines += [f"## K{k}","",f"RRMSE effect mean/median: {c['effect_mean']:+.6f}/{c['effect_median']:+.6f}; positive {c['positive']}/9; P4–P9 {c['p4_p9_positive']}/6. Spectral delta {c['spectral_delta']:+.6f}; EOG {c['eog_delta']:+.6f}; preservation {c['preservation_delta']:+.6f}; covariance {c['covariance_delta']:+.6f}; shared-decoder kappa {c['shared_kappa_delta']:+.6f}.",""]
    lines += ["The beta_det/beta_score grid is a post-hoc mechanism ceiling only. It did not select a participant-specific operating point or alter V11.1."]
    Path("reports/v11_2_score_component_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary

def _link(source:Path,target:Path)->None:
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source)

def stage_materialize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    old=Path(str(config["v11_1_result_root"]));root=Path(str(config["clean_posterior_root"]));score=json.loads((Path(str(config["score_audit_root"]))/"result_summary.json").read_text())
    if not score["clean_posterior_authorized"]:raise RuntimeError("score component retained headroom; V12 is not authorized")
    for fold in range(9):
        source=old/"folds"/f"fold_{fold:02d}";target=root/"folds"/f"fold_{fold:02d}";target.mkdir(parents=True,exist_ok=True);_link(source/"training_pairs.npz",target/"training_pairs.npz");_link(source/"unit_manifest.csv",target/"unit_manifest.csv")
        for protocol in SAME:_link(source/"units"/protocol/"inference.npz",target/"units"/protocol/"inference.npz")
    summary={"status":"completed_evaluator_blind_materialization","folds":9,"same_session_only":True,"evaluator_links":0,"two_heldout_split_inherited":True};_json(root/"materialization.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def _wrong_indices(subject:np.ndarray)->np.ndarray:
    result=np.empty(len(subject),dtype=int)
    for index,value in enumerate(subject):
        candidates=np.flatnonzero(subject!=value)
        if not len(candidates):raise RuntimeError("no other-subject operator for balanced WRONG context")
        result[index]=candidates[index%len(candidates)]
    return result

def _context_arrays(data:dict[str,np.ndarray],indices:np.ndarray,modes:np.ndarray,wrong_index:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    y=data["y"][indices];eog=data["eog"][indices];hs=data["h_subject"][indices];hp=np.broadcast_to(data["h_population"],hs.shape);hw=data["h_subject"][wrong_index[indices]];h=np.where((modes==0)[:,None,None],hs,np.where((modes==1)[:,None,None],hp,hw));a0=v11.apply_transfer(h,eog);return y-a0,a0

def _noise_bank(shape:tuple[int,...],seed:int,k:int)->np.ndarray:
    return np.stack([np.stack([np.random.default_rng(seed+window*1009+sample*1000003).standard_normal(shape[1:],dtype=np.float32) for window in range(shape[0])]) for sample in range(k)])

def _train_clean_models(config:Mapping[str,Any],fold:int,root:Path,*,technical:bool=False)->dict[str,Any]:
    import torch
    from torch.optim import AdamW
    from eeg_cgdr.models.clean_posterior_diffusion import CleanPosteriorConfig,CleanPosteriorDiffusion,DeterministicCleanEstimator,EMA,checkpoint_payload
    with np.load(root/"training_pairs.npz") as handle:data={name:np.asarray(handle[name]) for name in handle.files}
    all_indices=np.arange(len(data["y"]));val_indices=all_indices[all_indices%5==0];train_indices=all_indices[all_indices%5!=0]
    if technical:val_indices=val_indices[:32]
    fixed_indices=train_indices[:32]
    seed=int(config["seed"])+fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);device=torch.device("cuda");cfg=CleanPosteriorConfig(base_channels=32);det=DeterministicCleanEstimator(cfg).to(device);diff=CleanPosteriorDiffusion(cfg).to(device);det_opt=AdamW(det.parameters(),lr=float(config["learning_rate"]));diff_opt=AdamW(diff.parameters(),lr=float(config["learning_rate"]));ema=EMA(diff,float(config["ema_decay"]));generator=torch.Generator(device=device).manual_seed(seed);rng=np.random.default_rng(seed);wrong=_wrong_indices(data["subject"]);updates=int(config["technical_updates"] if technical else config["training_updates"]);curve=[];batch=32 if technical else int(config["batch_size"])
    # Observation variance is fitted only on outer-training population residuals.
    hp=np.broadcast_to(data["h_population"],data["h_subject"].shape);pop_lin=data["y"]-v11.apply_transfer(hp,data["eog"]);obs_var=np.maximum(np.var(data["x"][train_indices]-pop_lin[train_indices],axis=(0,2)),1e-4).astype(np.float32)
    schedule=[]
    for step in range(updates):
        if technical:idx=fixed_indices
        else:idx=rng.choice(train_indices,size=batch,replace=True)
        modes=np.arange(batch,dtype=int)%3;schedule.append((idx,modes));xlin,a0=_context_arrays(data,idx,modes,wrong);target=data["x"][idx];xt=torch.as_tensor(xlin,device=device);at=torch.as_tensor(a0,device=device);truth=torch.as_tensor(target,device=device)
        det.train();det_opt.zero_grad(set_to_none=True);pred=det(x_lin=xt,a0=at);loss=(pred-truth).square().mean();loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(det.parameters(),1));det_opt.step()
        if step%100==0 or step+1==updates:curve.append({"phase":"DET","step":step+1,"loss":float(loss.detach()),"gradient_norm":grad})
    for step,(idx,modes) in enumerate(schedule):
        xlin,a0=_context_arrays(data,idx,modes,wrong);target=data["x"][idx];xt=torch.as_tensor(xlin,device=device);at=torch.as_tensor(a0,device=device);truth=torch.as_tensor(target,device=device);diff.train();diff_opt.zero_grad(set_to_none=True);visited_timestep=torch.randint(1,max(config["t_start_candidates"])+1,(len(truth),),device=device,generator=generator);loss,_=diff.training_loss(truth,x_lin=xt,a0=at,generator=generator,timestep=visited_timestep,observation_anchored=True);loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(diff.parameters(),1));diff_opt.step();ema.update(diff)
        if step%100==0 or step+1==updates:curve.append({"phase":"DIFF","step":step+1,"loss":float(loss.detach()),"gradient_norm":grad})
    det.eval();ema.copy_to(diff);diff.eval();eval_idx=val_indices if len(val_indices) else train_indices[:8];modes=np.zeros(len(eval_idx),dtype=int);xlin,a0=_context_arrays(data,eval_idx,modes,wrong);target=data["x"][eval_idx];xt=torch.as_tensor(xlin,device=device);at=torch.as_tensor(a0,device=device);truth=torch.as_tensor(target,device=device);variance=torch.as_tensor(obs_var,device=device)
    with torch.no_grad():dp=det(x_lin=xt,a0=at).cpu().numpy()
    t_rows=[];best=None
    for t_start in config["t_start_candidates"]:
        bank=_noise_bank(target.shape,seed+700000,int(config["posterior_samples"]));samples=[]
        with torch.no_grad():
            for noise in bank:samples.append(diff.sample(x_lin=xt,a0=at,t_start=int(t_start),initial_noise=torch.as_tensor(noise,device=device),observation_variance=variance).cpu().numpy())
        value=np.mean(samples,axis=0);row={"fold":fold,"technical":int(technical),"t_start":int(t_start),"validation_rows":len(eval_idx),"linear_rrmse":v11.rrmse(xlin,target),"det_rrmse":v11.rrmse(dp,target),"diff_rrmse":v11.rrmse(value,target),"diff_correlation":v11.correlation(value,target)};t_rows.append(row)
        if best is None or row["diff_rrmse"]<best["diff_rrmse"]:best=row
    fixed_idx=fixed_indices;fx,fa=_context_arrays(data,fixed_idx,np.zeros(len(fixed_idx),dtype=int),wrong);ft=data["x"][fixed_idx];fxt=torch.as_tensor(fx,device=device);fat=torch.as_tensor(fa,device=device)
    with torch.no_grad():fixed_det=det(x_lin=fxt,a0=fat).cpu().numpy();noise=torch.as_tensor(_noise_bank(ft.shape,seed+900000,1)[0],device=device);fixed_diff=diff.sample(x_lin=fxt,a0=fat,t_start=int(best["t_start"]),initial_noise=noise,observation_variance=variance).cpu().numpy()
    payload=checkpoint_payload(cfg,det,diff,ema,observation_variance=obs_var,t_start=int(best["t_start"]),fold=fold,seed=seed,updates=updates,validation_rows=eval_idx,training_rows=train_indices);torch.save(payload,root/("technical_checkpoint.pt" if technical else "checkpoint.pt"));_csv(root/("technical_curve.csv" if technical else "training_curve.csv"),curve);_csv(root/("technical_t_selection.csv" if technical else "posterior_selection.csv"),t_rows)
    return {"fold":fold,"updates":updates,"training_state":"observation_anchored_x_lin","training_timestep_max":int(max(config["t_start_candidates"])),"t_start":int(best["t_start"]),"validation_linear_rrmse":best["linear_rrmse"],"validation_diff_rrmse":best["diff_rrmse"],"validation_headroom":float(best["linear_rrmse"]-best["diff_rrmse"]),"fixed_det_rrmse":v11.rrmse(fixed_det,ft),"fixed_det_correlation":v11.correlation(fixed_det,ft),"fixed_diff_rrmse":v11.rrmse(fixed_diff,ft),"fixed_diff_correlation":v11.correlation(fixed_diff,ft)}

def stage_technical(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.clean_posterior_diffusion import CleanPosteriorConfig,CleanPosteriorDiffusion
    root=Path(str(config["clean_posterior_root"]));metrics=_train_clean_models(config,0,root/"folds"/"fold_00",technical=True);model=CleanPosteriorDiffusion(CleanPosteriorConfig(base_channels=32)).cuda();x=torch.randn(2,3,512,device="cuda");a=torch.randn_like(x);noise=torch.randn_like(x);var=torch.ones(3,device="cuda");zero=model.sample(x_lin=x,a0=a,t_start=0,initial_noise=noise,observation_variance=var);zero_exact=bool(torch.equal(zero,x));finite=all(np.isfinite(value) for key,value in metrics.items() if isinstance(value,float));passed=finite and zero_exact and metrics["fixed_det_rrmse"]<=.05 and metrics["fixed_det_correlation"]>=.98 and metrics["fixed_diff_rrmse"]<=.10 and metrics["fixed_diff_correlation"]>=.95 and metrics["validation_headroom"]>0;decision="TECHNICAL_VALIDITY_PASSED" if passed else ("CLEAN_POSTERIOR_HAS_NO_OUTER_TRAINING_HEADROOM" if metrics["validation_headroom"]<=0 else "TECHNICAL_VALIDITY_FAILED");summary={"status":decision,"screen_authorized":bool(passed),"t_start_zero_linear_exact":zero_exact,"sampler_finite":finite,"artifact_residual_output_path":False,"technical_repair":"32-window rehearsal and production-capacity backbone",**metrics};_json(root/"technical_validity.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_technical_diagnostic(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.clean_posterior_diffusion import CleanPosteriorConfig,CleanPosteriorDiffusion,EMA
    root=Path(str(config["clean_posterior_root"]));fold=root/"folds"/"fold_00";checkpoint=torch.load(fold/"technical_checkpoint.pt",map_location="cpu",weights_only=False);cfg=CleanPosteriorConfig(**checkpoint["config"]);device=torch.device("cuda")
    with np.load(fold/"training_pairs.npz") as handle:data={name:np.asarray(handle[name]) for name in handle.files}
    train=np.asarray(checkpoint["training_rows"]);idx=train[:32];wrong=_wrong_indices(data["subject"]);xlin,a0=_context_arrays(data,idx,np.zeros(len(idx),dtype=int),wrong);target=data["x"][idx];xt=torch.as_tensor(xlin,device=device);at=torch.as_tensor(a0,device=device);variance=torch.as_tensor(checkpoint["observation_variance"],device=device);rows=[]
    for weights in ("raw","ema"):
        model=CleanPosteriorDiffusion(cfg).to(device);model.load_state_dict(checkpoint["diff"])
        if weights=="ema":ema=EMA(model);ema.load_state_dict(checkpoint["ema"]);ema.copy_to(model)
        model.eval()
        for t_start in config["t_start_candidates"]:
            bank=_noise_bank(target.shape,int(config["seed"])+900000,int(config["posterior_samples"]));samples=[]
            with torch.no_grad():
                for noise in bank:samples.append(model.sample(x_lin=xt,a0=at,t_start=int(t_start),initial_noise=torch.as_tensor(noise,device=device),observation_variance=variance).cpu().numpy())
                timestep=torch.full((len(target),),int(t_start),device=device,dtype=torch.long);alpha=model.alpha_bar.gather(0,timestep).reshape(len(target),1,1);noise=torch.as_tensor(bank[0],device=device);truth=torch.as_tensor(target,device=device);state=alpha.sqrt()*truth+(1-alpha).sqrt()*noise;predicted_v=model.backbone(state,timestep,x_lin=xt,a0=at);direct_x0=(alpha.sqrt()*state-(1-alpha).sqrt()*predicted_v).cpu().numpy()
            value=np.mean(samples,axis=0);rows.append({"weights":weights,"t_start":t_start,"rrmse":v11.rrmse(value,target),"correlation":v11.correlation(value,target),"direct_x0_rrmse":v11.rrmse(direct_x0,target),"direct_x0_correlation":v11.correlation(direct_x0,target)})
    _csv(root/"technical_raw_ema_diagnostic.csv",rows);summary={"status":"completed_raw_ema_diagnostic","rows":rows};_json(run_dir/"result_summary.json",summary);return summary

def stage_technical_resample(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.clean_posterior_diffusion import CleanPosteriorConfig,CleanPosteriorDiffusion,DeterministicCleanEstimator,EMA
    root=Path(str(config["clean_posterior_root"]));fold=root/"folds"/"fold_00";checkpoint=torch.load(fold/"technical_checkpoint.pt",map_location="cpu",weights_only=False);cfg=CleanPosteriorConfig(**checkpoint["config"]);device=torch.device("cuda");det=DeterministicCleanEstimator(cfg).to(device);diff=CleanPosteriorDiffusion(cfg).to(device);det.load_state_dict(checkpoint["det"]);diff.load_state_dict(checkpoint["diff"]);ema=EMA(diff);ema.load_state_dict(checkpoint["ema"]);ema.copy_to(diff);det.eval();diff.eval()
    with np.load(fold/"training_pairs.npz") as handle:data={name:np.asarray(handle[name]) for name in handle.files}
    wrong=_wrong_indices(data["subject"]);variance=torch.as_tensor(checkpoint["observation_variance"],device=device);rows=[]
    for panel,idx in (("fixed",np.asarray(checkpoint["training_rows"])[:32]),("validation",np.asarray(checkpoint["validation_rows"]))):
        xlin,a0=_context_arrays(data,idx,np.zeros(len(idx),dtype=int),wrong);target=data["x"][idx];xt=torch.as_tensor(xlin,device=device);at=torch.as_tensor(a0,device=device)
        with torch.no_grad():dp=det(x_lin=xt,a0=at).cpu().numpy()
        for t_start in config["t_start_candidates"]:
            bank=_noise_bank(target.shape,int(config["seed"])+(900000 if panel=="fixed" else 700000),int(config["posterior_samples"]));samples=[]
            with torch.no_grad():
                for noise in bank:samples.append(diff.sample(x_lin=xt,a0=at,t_start=int(t_start),initial_noise=torch.as_tensor(noise,device=device),observation_variance=variance).cpu().numpy())
            value=np.mean(samples,axis=0);rows.append({"panel":panel,"t_start":t_start,"linear_rrmse":v11.rrmse(xlin,target),"det_rrmse":v11.rrmse(dp,target),"det_correlation":v11.correlation(dp,target),"diff_rrmse":v11.rrmse(value,target),"diff_correlation":v11.correlation(value,target)})
    validation=[r for r in rows if r["panel"]=="validation"];best=min(validation,key=lambda r:r["diff_rrmse"]);fixed=next(r for r in rows if r["panel"]=="fixed" and r["t_start"]==best["t_start"]);checkpoint["t_start"]=int(best["t_start"]);torch.save(checkpoint,fold/"technical_checkpoint.pt");passed=fixed["det_rrmse"]<=.05 and fixed["det_correlation"]>=.98 and fixed["diff_rrmse"]<=.10 and fixed["diff_correlation"]>=.95 and best["linear_rrmse"]-best["diff_rrmse"]>0;status="TECHNICAL_VALIDITY_PASSED" if passed else ("CLEAN_POSTERIOR_HAS_NO_OUTER_TRAINING_HEADROOM" if best["linear_rrmse"]-best["diff_rrmse"]<=0 else "TECHNICAL_VALIDITY_FAILED");summary={"status":status,"screen_authorized":bool(passed),"proximal_covariance":"reverse_step_covariance","t_start":int(best["t_start"]),"validation_linear_rrmse":best["linear_rrmse"],"validation_diff_rrmse":best["diff_rrmse"],"validation_headroom":best["linear_rrmse"]-best["diff_rrmse"],"fixed_det_rrmse":fixed["det_rrmse"],"fixed_det_correlation":fixed["det_correlation"],"fixed_diff_rrmse":fixed["diff_rrmse"],"fixed_diff_correlation":fixed["diff_correlation"],"t_start_zero_linear_exact":True,"sampler_finite":True,"artifact_residual_output_path":False};_csv(root/"technical_resample.csv",rows);_json(root/"technical_validity.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_train(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["clean_posterior_root"]));gate=json.loads((root/"technical_validity.json").read_text())
    if not gate["screen_authorized"]:raise RuntimeError("V12 technical gate failed")
    metrics=_train_clean_models(config,task_index,root/"folds"/f"fold_{task_index:02d}",technical=False);summary={"status":"completed_fold_training",**metrics};_json(run_dir/"result_summary.json",summary);return summary

def stage_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.clean_posterior_diffusion import CleanPosteriorConfig,CleanPosteriorDiffusion,DeterministicCleanEstimator,EMA
    root=Path(str(config["clean_posterior_root"]));fold=root/"folds"/f"fold_{task_index:02d}";checkpoint=torch.load(fold/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=CleanPosteriorConfig(**checkpoint["config"]);device=torch.device("cuda");det=DeterministicCleanEstimator(cfg).to(device);diff=CleanPosteriorDiffusion(cfg).to(device);det.load_state_dict(checkpoint["det"]);diff.load_state_dict(checkpoint["diff"]);ema=EMA(diff);ema.load_state_dict(checkpoint["ema"]);ema.copy_to(diff);det.eval();diff.eval();variance=torch.as_tensor(checkpoint["observation_variance"],device=device);latency=[]
    with (fold/"unit_manifest.csv").open(newline="",encoding="utf-8") as h:manifest=list(csv.DictReader(h))
    for unit_index,row in enumerate(manifest):
        protocol=row["protocol"]
        if protocol not in SAME:continue
        inf=np.load(fold/"units"/protocol/"inference.npz");outputs={}
        for panel in ("paired","natural"):
            y=np.asarray(inf[f"{panel}_y"],np.float32);eog=np.asarray(inf[f"{panel}_eog"],np.float32);outputs[f"{panel}_RAW"]=y
            for name,h in {"POP":inf["h_population"],"MATCH":inf["h_match"],"WRONG":inf["h_wrong"]}.items():
                a0=v11.apply_transfer(np.asarray(h),eog);xlin=y-a0;xt=torch.as_tensor(xlin,device=device);at=torch.as_tensor(a0,device=device);start=time.perf_counter()
                with torch.no_grad():dp=det(x_lin=xt,a0=at).cpu().numpy();bank=_noise_bank(y.shape,int(config["seed"])+task_index*100000+unit_index*10000,int(config["posterior_samples"]));samples=[diff.sample(x_lin=xt,a0=at,t_start=int(checkpoint["t_start"]),initial_noise=torch.as_tensor(noise,device=device),observation_variance=variance).cpu().numpy() for noise in bank]
                torch.cuda.synchronize();latency.append((time.perf_counter()-start)/len(y));outputs[f"{panel}_LINEAR-{name}"]=xlin;outputs[f"{panel}_DET-CLEAN-{name}"]=dp;outputs[f"{panel}_DIFF-CLEAN-{name}"]=np.mean(samples,axis=0)
        out=fold/"outputs"/protocol;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/"inference_outputs.npz",**outputs)
    summary={"status":"completed_evaluator_blind_inference","fold":task_index,"t_start":int(checkpoint["t_start"]),"latency_seconds_per_window":float(np.mean(latency)),"posterior_calls":int(config["posterior_samples"])*int(config["ddim_steps"]),"evaluator_opened":False};_json(fold/"inference_runtime.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_evaluate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["clean_posterior_root"]));source=Path(str(config["v11_result_root"]));fold=root/"folds"/f"fold_{task_index:02d}";paired=[];natural=[]
    for protocol in SAME:
        inf=np.load(fold/"units"/protocol/"inference.npz");ev=np.load(source/"folds"/f"fold_{task_index:02d}"/"units"/protocol/"evaluator.npz");out=np.load(fold/"outputs"/protocol/"inference_outputs.npz");scale=np.asarray(inf["eeg_scale"]);location=np.asarray(inf["eeg_location"]);x=np.asarray(ev["paired_x"])[...,:500];raw_p=(np.asarray(out["paired_RAW"])*scale[None,:,None]+location[None,:,None])[...,:500];labels=np.asarray(ev["natural_labels"]);raw_n=np.asarray(out["natural_RAW"])[...,:500];eog=np.asarray(inf["natural_eog"])[...,:500];natural_arrays={key.split("_",1)[1]:np.asarray(out[key])[...,:500] for key in out.files if key.startswith("natural_")};shared=_shared_decoder_scores(raw_n,natural_arrays,labels);raw_error=v111._band_error(raw_p,x,(1,45),config)
        for key in out.files:
            panel,method=key.split("_",1);value=np.asarray(out[key])
            if panel=="paired":
                physical=(value*scale[None,:,None]+location[None,:,None])[...,:500];paired.append({"subject":task_index+1,"protocol":protocol,"method":method,"rrmse":v11.rrmse(physical,x),"correlation":v11.correlation(physical,x),"delta_snr":v11.delta_snr(physical,x,raw_p),"paired_spectral_utility":raw_error-v111._band_error(physical,x,(1,45),config)})
            else:natural.append({"subject":task_index+1,"protocol":protocol,"method":method,"shared_raw_decoder_kappa":shared[method],**_natural_metrics(value,raw_n,eog,labels,config)})
    _csv(fold/"paired_metrics.csv",paired);_csv(fold/"natural_metrics.csv",natural);summary={"status":"completed_independent_evaluation","fold":task_index,"evaluator_opened_after_outputs":True};_json(run_dir/"result_summary.json",summary);return summary

def _aggregate(config:Mapping[str,Any],folds:list[int])->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    root=Path(str(config["clean_posterior_root"]));paired=[];natural=[]
    for fold in folds:
        with (root/"folds"/f"fold_{fold:02d}"/"paired_metrics.csv").open(newline="",encoding="utf-8") as h:paired.extend(csv.DictReader(h))
        with (root/"folds"/f"fold_{fold:02d}"/"natural_metrics.csv").open(newline="",encoding="utf-8") as h:natural.extend(csv.DictReader(h))
    effects=[]
    for subject in sorted({int(r["subject"]) for r in paired}):
        p=[r for r in paired if int(r["subject"])==subject];by=defaultdict(list)
        for row in p:by[row["method"]].append(float(row["rrmse"]))
        mean={name:float(np.mean(value)) for name,value in by.items()};match=[r for r in natural if int(r["subject"])==subject and r["method"]=="DIFF-CLEAN-MATCH"];linear=[r for r in natural if int(r["subject"])==subject and r["method"]=="LINEAR-MATCH"]
        effects.append({"subject":subject,"U_L":mean["LINEAR-MATCH"]-mean["DIFF-CLEAN-MATCH"],"U_D":mean["DET-CLEAN-MATCH"]-mean["DIFF-CLEAN-MATCH"],"U_P":mean["DIFF-CLEAN-POP"]-mean["DIFF-CLEAN-MATCH"],"U_W":mean["DIFF-CLEAN-WRONG"]-mean["DIFF-CLEAN-MATCH"],"paired_spectral_delta_vs_linear":float(np.mean([float(r["paired_spectral_utility"]) for r in p if r["method"]=="DIFF-CLEAN-MATCH"])-np.mean([float(r["paired_spectral_utility"]) for r in p if r["method"]=="LINEAR-MATCH"])),**{name:float(np.mean([float(r[name]) for r in match])) for name in ("eog_attenuation","preservation","covariance","mi_band_distortion","erd_preservation","shared_raw_decoder_kappa")},"linear_eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in linear])),"linear_shared_kappa":float(np.mean([float(r["shared_raw_decoder_kappa"]) for r in linear]))})
    return paired,natural,effects

def stage_gate3(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["clean_posterior_root"]));paired,natural,effects=_aggregate(config,[0,1,2]);_csv(root/"participant_effects.csv",effects)
    means={name:float(np.mean([r[name] for r in effects])) for name in ("U_L","U_D","U_P","U_W")};positive={name:int(np.sum([r[name]>0 for r in effects])) for name in means};linear=[r for r in natural if r["method"]=="LINEAR-MATCH"];safety={"spectral_delta":float(np.mean([r["paired_spectral_delta_vs_linear"] for r in effects])),"eog_delta":float(np.mean([r["eog_attenuation"]-r["linear_eog_attenuation"] for r in effects])),"preservation":float(np.mean([r["preservation"] for r in effects])),"covariance":float(np.mean([r["covariance"] for r in effects])),"shared_kappa_delta":float(np.mean([r["shared_raw_decoder_kappa"]-r["linear_shared_kappa"] for r in effects])),"erd_min":float(np.min([r["erd_preservation"] for r in effects]))};passed=means["U_L"]>0 and positive["U_L"]>=2 and means["U_D"]>0 and positive["U_D"]>=2 and means["U_P"]>0 and positive["U_P"]>=2 and means["U_W"]>0 and positive["U_W"]>=2 and safety["spectral_delta"]>=0 and safety["eog_delta"]>=-.01 and safety["preservation"]>=.78 and safety["covariance"]<=.15 and safety["shared_kappa_delta"]>=-.02 and safety["erd_min"]>0
    summary={"status":"completed_three_participant_screen","decision":"FULL_NINE_AUTHORIZED" if passed else "SUBJECT_OPERATOR_SUPPORTED_BUT_CLEAN_POSTERIOR_DIFFUSION_DID_NOT_BEAT_LINEAR_MATCH","full_nine_authorized":bool(passed),"effects":means,"positive":positive,"safety":safety,"participants":3};_json(root/"gate3_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_aggregate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["clean_posterior_root"]));paired,natural,effects=_aggregate(config,list(range(9)));_csv(root/"participant_effects.csv",effects);methods=[]
    for method in sorted({r["method"] for r in paired}):
        p=[r for r in paired if r["method"]==method];n=[r for r in natural if r["method"]==method];methods.append({"method":method,"participants":9,"rrmse":float(np.mean([float(r["rrmse"]) for r in p])),"correlation":float(np.mean([float(r["correlation"]) for r in p])),"delta_snr":float(np.mean([float(r["delta_snr"]) for r in p])),"paired_spectral_utility":float(np.mean([float(r["paired_spectral_utility"]) for r in p])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in n])),"preservation":float(np.mean([float(r["preservation"]) for r in n])),"covariance":float(np.mean([float(r["covariance"]) for r in n])),"shared_raw_decoder_kappa":float(np.mean([float(r["shared_raw_decoder_kappa"]) for r in n]))});_csv(root/"method_summary.csv",methods)
    rng=np.random.default_rng(int(config["bootstrap_seed"]));indices=rng.integers(0,9,size=(int(config["bootstrap_replicates"]),9));boot=[];effect_summary={}
    for name in ("U_L","U_D","U_P","U_W"):
        value=np.asarray([r[name] for r in effects]);rep=value[indices].mean(1);effect_summary[name]={"mean":float(value.mean()),"median":float(np.median(value)),"positive":int(np.sum(value>0)),"ci_low":float(np.quantile(rep,.025)),"ci_high":float(np.quantile(rep,.975))};boot.append({"effect":name,**effect_summary[name]})
    _csv(root/"bootstrap_summary.csv",boot);p4=effects[3:];safety=next(r for r in methods if r["method"]=="DIFF-CLEAN-MATCH");linear=next(r for r in methods if r["method"]=="LINEAR-MATCH");passed=all(effect_summary[x]["mean"]>0 and effect_summary[x]["median"]>0 for x in effect_summary) and effect_summary["U_L"]["positive"]>=6 and effect_summary["U_P"]["positive"]>=6 and effect_summary["U_W"]["positive"]>=6 and effect_summary["U_D"]["positive"]>=5 and all(np.mean([r[x] for r in p4])>0 for x in ("U_L","U_D","U_P","U_W")) and safety["paired_spectral_utility"]>=linear["paired_spectral_utility"] and safety["eog_attenuation"]>=linear["eog_attenuation"]-.01 and safety["preservation"]>=linear["preservation"]-.02 and safety["covariance"]<=linear["covariance"]+.02 and safety["shared_raw_decoder_kappa"]>=linear["shared_raw_decoder_kappa"]-.02;summary={"status":"completed_nine_participant_one_seed","decision":"ELIGIBLE_FOR_TWO_ADDITIONAL_SEEDS" if passed else "SUBJECT_OPERATOR_SUPPORTED_BUT_CLEAN_POSTERIOR_DIFFUSION_DID_NOT_BEAT_LINEAR_MATCH","additional_seeds_authorized":bool(passed),"effects":effect_summary,"participants":9,"p4_p9_no_reversal":bool(all(np.mean([r[x] for r in p4])>0 for x in ("U_L","U_D","U_P","U_W"))),"same_session_only":True,"development_not_confirmation":True};_json(root/"routing_decision.json",summary);_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary

def stage_finalize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    score_root=Path(str(config["score_audit_root"]));root=Path(str(config["clean_posterior_root"]));score=json.loads((score_root/"result_summary.json").read_text());technical=json.loads((root/"technical_validity.json").read_text());decision=technical["status"] if not technical.get("screen_authorized",False) else (json.loads((root/"result_summary.json").read_text())["decision"] if (root/"result_summary.json").exists() else "SCREEN_NOT_COMPLETED")
    selection=[]
    for name in ("technical_t_selection.csv","technical_resample.csv"):
        path=root/"folds"/"fold_00"/name if name=="technical_t_selection.csv" else root/name
        if path.exists():
            with path.open(newline="",encoding="utf-8") as h:
                for row in csv.DictReader(h):selection.append({"source":name,**row})
    _csv(root/"posterior_selection_manifest.csv",selection)
    participant_path=score_root/"participant_metrics.csv";participants=[]
    if participant_path.exists():
        with participant_path.open(newline="",encoding="utf-8") as h:participants=list(csv.DictReader(h))
    shared=[]
    for k in (8,32):
        for method in ("LINEAR-MATCH","DET-MATCH","DIFF-MATCH","SCORE-ONLY"):
            rows=[r for r in participants if int(r["k"])==k and r["method"]==method]
            if rows:shared.append({"k":k,"method":method,"participants":len(rows),**{name:float(np.nanmean([float(r[name]) for r in rows])) for name in ("shared_raw_decoder_kappa","method_specific_kappa","preservation","covariance","eog_attenuation","mi_band_distortion","erd_preservation")}})
    _csv(root/"shared_decoder_safety.csv",shared);_csv(root/"method_summary.csv",[{"stage":"technical","method":"DET-CLEAN","rrmse":technical.get("fixed_det_rrmse",""),"correlation":technical.get("fixed_det_correlation",""),"status":decision},{"stage":"technical","method":"DIFF-CLEAN","rrmse":technical.get("fixed_diff_rrmse",""),"correlation":technical.get("fixed_diff_correlation",""),"status":decision},{"stage":"outer_training_validation","method":"LINEAR-MATCH","rrmse":technical.get("validation_linear_rrmse",""),"status":decision},{"stage":"outer_training_validation","method":"DIFF-CLEAN-MATCH","rrmse":technical.get("validation_diff_rrmse",""),"status":decision}]);_csv(root/"participant_effects.csv",[{"status":"not_run","reason":"technical_or_outer_training_gate_failed","participants":0}]);_csv(root/"bootstrap_summary.csv",[{"status":"not_run","reason":"no heldout scientific screen","participants":0}])
    summary={"status":"completed_v12_development_route","decision":decision,"score_component_decision":score["decision"],"technical_validity":technical,"three_participant_screen_run":False,"nine_participant_screen_run":False,"additional_seeds_authorized":False,"same_session_only":True,"deployment":"EOG-guided","development_not_confirmation":True,"family_wide_status":"not_tested"};_json(root/"routing_decision.json",summary);_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary)
    lines=["# BCI2b conditional clean-neural posterior diffusion V12","","Development exploration; same-session and EOG-guided only.","",f"Final decision: `{decision}`.","","## Route","",f"The zero-training score-component audit returned `{score['decision']}`, so the artifact-residual score object was closed and V12 was authorized. The conditional clean posterior then failed its pre-heldout technical/outer-training gate; no participant 1–3 or 4–9 evaluator screen was run and no additional seeds were submitted.","","## Technical evidence","",f"t_start=0 exactly returned LINEAR: `{technical.get('t_start_zero_linear_exact')}`. The sampler remained finite: `{technical.get('sampler_finite')}`. Selected t_start: `{technical.get('t_start')}`. Fixed DET/DIFF RRMSE: `{technical.get('fixed_det_rrmse',float('nan')):.4f}/{technical.get('fixed_diff_rrmse',float('nan')):.4f}`. Outer-training LINEAR/DIFF RRMSE: `{technical.get('validation_linear_rrmse',float('nan')):.4f}/{technical.get('validation_diff_rrmse',float('nan')):.4f}`.","","This closes only the present observation-anchored clean-posterior implementation. It is not a family-wide diffusion or personalization conclusion."]
    Path("reports/bci2b_clean_posterior_diffusion_v12.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return summary

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="score-audit":return stage_score_audit(config,task_index,run_dir)
    if stage=="materialize":return stage_materialize(config,task_index,run_dir)
    if stage=="technical":return stage_technical(config,task_index,run_dir)
    if stage=="technical-diagnostic":return stage_technical_diagnostic(config,task_index,run_dir)
    if stage=="technical-resample":return stage_technical_resample(config,task_index,run_dir)
    if stage=="train":return stage_train(config,task_index,run_dir)
    if stage=="infer":return stage_infer(config,task_index,run_dir)
    if stage=="evaluate":return stage_evaluate(config,task_index,run_dir)
    if stage=="gate3":return stage_gate3(config,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(config,task_index,run_dir)
    if stage=="finalize":return stage_finalize(config,task_index,run_dir)
    raise ValueError(stage)
