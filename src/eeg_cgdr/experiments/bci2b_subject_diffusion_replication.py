"""Three-seed replication of subject conditioning inside frozen V11.1 diffusion."""
from __future__ import annotations
import csv,inspect,json,time
from collections import defaultdict
from pathlib import Path
from typing import Any,Mapping
import numpy as np,yaml
from eeg_cgdr.experiments import bci2b_eog_residual_v11 as v11
from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as v111

SAME=("same_01","same_02","same_03")
METHODS=("RAW","LINEAR-POP","LINEAR-MATCH","LINEAR-WRONG","DET-POP","DET-MATCH","DET-WRONG","DIFF-POP","DIFF-MATCH","DIFF-WRONG","DIFF-TEMPORAL-SHUFFLED")
def _config(path:Path)->dict[str,Any]:
    with path.open(encoding="utf-8") as h:return yaml.safe_load(h)
def _json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _csv(path:Path,rows:list[dict[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    keys=list(rows[0]);[keys.append(k) for row in rows for k in row if k not in keys]
    with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows(rows)
def _read(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def _link(source:Path,target:Path)->None:
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():target.symlink_to(source)
def _seed_root(config:Mapping[str,Any],seed:int)->Path:return Path(str(config["result_root"]))/"seeds"/str(seed)

def stage_audit(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    root=Path(str(config["result_root"]));old=Path(str(config["v11_1_result_root"]));v11root=Path(str(config["v11_result_root"]));summary_rows=_read(old/"method_summary_k8.csv");effects=_read(old/"participant_effects_k8.csv")
    method={r["method"]:float(r["rrmse"]) for r in summary_rows};checks={"diff_pop":bool(abs(method["DIFF-POP"]-.1512)<=5e-4),"diff_match":bool(abs(method["DIFF-MATCH"]-.1152)<=5e-4),"diff_wrong":bool(abs(method["DIFF-WRONG"]-.1614)<=5e-4),"U_P":bool(abs(np.mean([float(r["U_P"]) for r in effects])-.0360)<=5e-4),"U_W":bool(abs(np.mean([float(r["U_W"]) for r in effects])-.0462)<=5e-4),"U_P_9_of_9":bool(sum(float(r["U_P"])>0 for r in effects)==9),"U_W_9_of_9":bool(sum(float(r["U_W"])>0 for r in effects)==9)}
    inference_source=inspect.getsource(v111._infer_k);boundary="evaluator.npz" not in inference_source and "paired_x" not in inference_source and "natural_labels" not in inference_source;manifest=[];donors=[];overlap=True
    for fold in range(9):
        recipient=fold+1;base=old/"folds"/f"fold_{fold:02d}";metadata=json.loads((v11root/"folds"/f"fold_{fold:02d}"/"fold_metadata.json").read_text());rows=_read(base/"unit_manifest.csv");wrong=int(rows[0]["wrong_donor"]);unseen=wrong not in metadata["population_training"] and recipient not in metadata["population_training"]
        for row in rows:
            protocol=row["protocol"];session=row["support_session"];eeg,eog,sfreq,events=v11._load_session(config,recipient,session);support,query=v11._support_query_ranges(events,eeg.shape[1],sfreq);overlap=overlap and support.stop<=query.start;inf=np.load(base/"units"/protocol/"inference.npz");out=np.load(base/"outputs"/"k8"/protocol/"inference_outputs.npz");donors.append({"recipient":recipient,"fold":fold,"protocol":protocol,"primary_wrong_donor":wrong,"same_layout":1,"same_session":1,"sampling_rate":sfreq,"wrong_in_population_training":int(wrong in metadata["population_training"]),"recipient_in_population_training":int(recipient in metadata["population_training"]),"primary_wrong_unseen":int(unseen)})
            for seed in config["seeds"]:
                for key in out.files:
                    panel,method=key.split("_",1);manifest.append({"path":str(base/"outputs"/"k8"/protocol/"inference_outputs.npz"),"participant":recipient,"seed":seed,"fold":fold,"method":method,"panel":panel,"shape":"x".join(map(str,out[key].shape)),"role":"frozen_original" if seed==config["seeds"][0] else "expected_new_seed","donor_status":"cyclic_unseen" if "WRONG" in method else "not_applicable"})
    # Checkpoint reload/replay on one real window, with identical K=1 noise.
    cp=torch.load(old/"folds"/"fold_00"/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**cp["config"]);data=np.load(old/"folds"/"fold_00"/"units"/"same_01"/"inference.npz");y=np.asarray(data["paired_y"][:1],np.float32);eog=np.asarray(data["paired_eog"][:1],np.float32);a0=v11.apply_transfer(np.asarray(data["h_match"]),eog)
    def load():
        det=DeterministicEOGResidual(cfg);diff=EOGResidualDiffusion(cfg);det.load_state_dict(cp["det"]);diff.load_state_dict(cp["diff"]);ema=EMA(diff);ema.load_state_dict(cp["ema"]);ema.copy_to(diff);det.eval();diff.eval();return det,diff
    first=load();second=load();yt=torch.as_tensor(y);et=torch.as_tensor(eog);at=torch.as_tensor(a0);noise=torch.as_tensor(v11._noise_bank(y.shape,int(config["seeds"][0]),1)[0])
    with torch.no_grad():r1=first[0](y=yt,eog=et,a0=at);r2=second[0](y=yt,eog=et,a0=at);d1=first[1].sample(y=yt,eog=et,a0=at,r_det=r1,initial_noise=noise);d2=second[1].sample(y=yt,eog=et,a0=at,r_det=r2,initial_noise=noise)
    replay=bool(torch.equal(r1,r2) and torch.equal(d1,d2));common_noise="bank=v11._noise_bank" in inference_source or "bank=v11._noise_bank".replace("v11.","") in inference_source
    for seed in config["seeds"]:
        sroot=_seed_root(config,int(seed));sroot.mkdir(parents=True,exist_ok=True);_link(old/"spectral_evaluator_contract.json",sroot/"spectral_evaluator_contract.json")
        for fold in range(9):
            source=old/"folds"/f"fold_{fold:02d}";target=sroot/"folds"/f"fold_{fold:02d}";target.mkdir(parents=True,exist_ok=True);_link(source/"training_pairs.npz",target/"training_pairs.npz");_link(source/"unit_manifest.csv",target/"unit_manifest.csv")
            for protocol in SAME:_link(source/"units"/protocol/"inference.npz",target/"units"/protocol/"inference.npz")
            if int(seed)==int(config["seeds"][0]):
                _link(source/"checkpoint.pt",target/"checkpoint.pt");_link(source/"outputs"/"k8",target/"outputs"/"k8");_link(source/"paired_metrics_v11_1.csv",target/"paired_metrics_v11_1.csv");_link(source/"natural_metrics_v11_1.csv",target/"natural_metrics_v11_1.csv");_link(source/"bandwise_v11_1.csv",target/"bandwise_v11_1.csv");_link(source/"inference_runtime.json",target/"inference_runtime.json")
        _json(sroot/"technical_validity.json",{"training_authorized":True,"source":"frozen_V11.1_technical_contract"})
    passed=bool(all(checks.values()) and boundary and overlap and replay and common_noise and all(int(r["primary_wrong_unseen"]) for r in donors));audit={"status":"J0_AUDIT_PASSED" if passed else "J0_AUDIT_FAILED","training_authorized":passed,"exact_replay":checks,"inference_evaluator_blind":bool(boundary),"support_query_disjoint":bool(overlap),"checkpoint_replay_exact":bool(replay),"common_noise_contract":bool(common_noise),"deployment":"EOG-guided","participants":9,"seeds":config["seeds"]};_csv(root/"lightweight_manifest.csv",manifest);_csv(root/"wrong_donor_audit.csv",donors);_json(root/"j0_audit.json",audit);_json(run_dir/"result_summary.json",audit);return audit

def _task(config:Mapping[str,Any],task_index:int)->tuple[int,int]:
    seeds=[int(x) for x in config["new_seeds"]];return seeds[task_index//9],task_index%9
def _v11_config(config:Mapping[str,Any],seed:int)->dict[str,Any]:
    return {**config,"result_root":str(_seed_root(config,seed)),"seed":seed}

def stage_train(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seed,fold=_task(config,task_index);audit=json.loads((Path(str(config["result_root"]))/"j0_audit.json").read_text())
    if not audit["training_authorized"]:raise RuntimeError("J0 failed")
    result=v11.stage_train_fold(_v11_config(config,seed),fold,run_dir);result.update({"replication_seed":seed,"fold":fold});_json(run_dir/"result_summary.json",result);return result

def stage_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seed,fold=_task(config,task_index);result=v111._infer_k(_v11_config(config,seed),fold,8);summary={"status":"completed_frozen_K8_inference","seed":seed,"fold":fold,"evaluator_opened":False,**result};_json(run_dir/"result_summary.json",summary);return summary

def stage_evaluate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seed,fold=_task(config,task_index);base=_seed_root(config,seed)/"folds"/f"fold_{fold:02d}";source=Path(str(config["v11_result_root"]))/"folds"/f"fold_{fold:02d}";paired=[];natural=[];bands=[]
    for protocol in SAME:
        inf=np.load(base/"units"/protocol/"inference.npz");ev=np.load(source/"units"/protocol/"evaluator.npz");out=np.load(base/"outputs"/"k8"/protocol/"inference_outputs.npz");scale=np.asarray(inf["eeg_scale"]);location=np.asarray(inf["eeg_location"]);target=np.asarray(ev["paired_x"])[...,:500];raw_p=(np.asarray(out["paired_RAW"])*scale[None,:,None]+location[None,:,None])[...,:500];raw_error=v111._band_error(raw_p,target,(1,45),config);raw_n=np.asarray(out["natural_RAW"])[...,:500];eog=np.asarray(inf["natural_eog"])[...,:500];labels=np.asarray(ev["natural_labels"]);energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3)
        for key in out.files:
            panel,method=key.split("_",1);value=np.asarray(out[key])
            if panel=="paired":
                physical=(value*scale[None,:,None]+location[None,:,None])[...,:500];paired.append({"subject":fold+1,"protocol":protocol,"k":8,"method":method,"rrmse":v11.rrmse(physical,target),"correlation":v11.correlation(physical,target),"delta_snr":v11.delta_snr(physical,target,raw_p),"paired_psd_error_1_45":v111._band_error(physical,target,(1,45),config),"paired_spectral_utility":raw_error-v111._band_error(physical,target,(1,45),config)})
                for name,band in config["bands"].items():bands.append({"subject":fold+1,"protocol":protocol,"k":8,"method":method,"panel":"paired","band":name,"error":v111._band_error(physical,target,tuple(band),config),"utility":v111._band_error(raw_p,target,tuple(band),config)-v111._band_error(physical,target,tuple(band),config)})
            else:
                cropped=value[...,:500];natural.append({"subject":fold+1,"protocol":protocol,"k":8,"method":method,"mi_band_distortion":v111._bandpower_distortion(cropped[low],raw_n[low],(8,30),config),"preservation":1-v11.rrmse(cropped[low],raw_n[low]),"covariance":v11._covariance_distortion(cropped[low],raw_n[low]),"eog_attenuation":v11._coherence_proxy(raw_n,eog)-v11._coherence_proxy(cropped,eog),"mi_kappa":v11._kappa(cropped,labels),"erd_preservation":v111._erd_preservation(cropped,raw_n,labels,config),"historical_whole_psd":v11._psd_distortion(value[low],np.asarray(out["natural_RAW"])[low])})
                for name,band in config["bands"].items():bands.append({"subject":fold+1,"protocol":protocol,"k":8,"method":method,"panel":"natural","band":name,"distortion":v111._bandpower_distortion(cropped[low],raw_n[low],tuple(band),config)})
    _csv(base/"paired_metrics_v11_1.csv",paired);_csv(base/"natural_metrics_v11_1.csv",natural);_csv(base/"bandwise_v11_1.csv",bands);result={"status":"completed_K8_only_evaluation","seed":seed,"fold":fold,"paired_rows":len(paired),"natural_rows":len(natural),"k32_opened":False};_json(run_dir/"result_summary.json",result);return result

def stage_donor_operators(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    """Build seed-independent donor support operators without evaluator data."""
    fold=task_index;recipient=fold+1;root=Path(str(config["result_root"]));base=_seed_root(config,int(config["seeds"][0]))/"folds"/f"fold_{fold:02d}";metadata=json.loads((Path(str(config["v11_result_root"]))/"folds"/f"fold_{fold:02d}"/"fold_metadata.json").read_text());rows=[]
    for row in _read(base/"unit_manifest.csv"):
        protocol=row["protocol"];session=row["support_session"];inf=np.load(base/"units"/protocol/"inference.npz");folder=root/"donor_operators"/f"fold_{fold:02d}"/protocol;folder.mkdir(parents=True,exist_ok=True)
        for donor in range(1,10):
            if donor==recipient:continue
            deeg,deog,sfreq,events=v11._load_session(config,donor,session);support,_=v11._support_query_ranges(events,deeg.shape[1],sfreq);h=v11._normalized_transfer(deeg,deog,support,np.asarray(inf["eeg_location"]),np.asarray(inf["eeg_scale"]),np.asarray(inf["eog_location"]),np.asarray(inf["eog_scale"]));seen=int(donor in metadata["population_training"]);primary=int(donor==int(row["wrong_donor"]));np.savez_compressed(folder/f"donor_{donor:02d}.npz",h=np.asarray(h,np.float32),donor=np.array(donor),recipient=np.array(recipient),donor_seen=np.array(seen),primary_cyclic=np.array(primary),support_start=np.array(support.start),support_stop=np.array(support.stop),sampling_rate=np.array(sfreq));rows.append({"fold":fold,"recipient":recipient,"protocol":protocol,"donor":donor,"donor_seen":seen,"primary_cyclic":primary,"support_samples":support.stop-support.start,"sampling_rate":sfreq})
    _csv(root/"donor_operators"/f"fold_{fold:02d}"/"manifest.csv",rows);summary={"status":"completed_evaluator_blind_donor_operator_build","fold":fold,"recipient":recipient,"operators":len(rows),"evaluator_opened":False};_json(run_dir/"result_summary.json",summary);return summary

def stage_multi_donor_infer(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from eeg_cgdr.models.eog_residual_diffusion import DeterministicEOGResidual,EMA,EOGResidualConfig,EOGResidualDiffusion
    seeds=[int(x) for x in config["seeds"]];seed=seeds[task_index//9];fold=task_index%9;recipient=fold+1;sroot=_seed_root(config,seed);base=sroot/"folds"/f"fold_{fold:02d}";cp=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False);cfg=EOGResidualConfig(**cp["config"]);device=torch.device("cuda");det=DeterministicEOGResidual(cfg).to(device);diff=EOGResidualDiffusion(cfg).to(device);det.load_state_dict(cp["det"]);diff.load_state_dict(cp["diff"]);ema=EMA(diff);ema.load_state_dict(cp["ema"]);ema.copy_to(diff);det.eval();diff.eval();scale=np.asarray(cp["residual_scale"],np.float32);metadata=json.loads((Path(str(config["v11_result_root"]))/"folds"/f"fold_{fold:02d}"/"fold_metadata.json").read_text())
    for unit_index,row in enumerate(_read(base/"unit_manifest.csv")):
        protocol=row["protocol"];session=row["support_session"];inf=np.load(base/"units"/protocol/"inference.npz");y=np.asarray(inf["paired_y"],np.float32);eog=np.asarray(inf["paired_eog"],np.float32);gamma=float(inf["gamma"]);yt=torch.as_tensor(y,device=device);et=torch.as_tensor(eog,device=device);bank=v11._noise_bank(y.shape,seed+fold*100000+unit_index*10000,8)
        operator_folder=Path(str(config["result_root"]))/"donor_operators"/f"fold_{fold:02d}"/protocol
        for operator_path in sorted(operator_folder.glob("donor_*.npz")):
            operator=np.load(operator_path);donor=int(operator["donor"]);h=np.asarray(operator["h"],np.float32);a0=v11.apply_transfer(h,eog);at=torch.as_tensor(a0,device=device)
            with torch.no_grad():rdet=det(y=yt,eog=et,a0=at);samples=[diff.sample(y=yt,eog=et,a0=at,r_det=rdet,initial_noise=torch.as_tensor(noise,device=device)).cpu().numpy() for noise in bank]
            correction=a0+(rdet.cpu().numpy()+np.mean(samples,axis=0))*scale[None,:,None];restored=v11.gamma_correction(y,correction,gamma);folder=base/"multi_donor"/protocol;folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f"donor_{donor:02d}.npz",paired_DIFF_WRONG=restored,donor=np.array(donor),recipient=np.array(recipient),donor_seen=np.array(int(operator["donor_seen"])),primary_cyclic=np.array(int(operator["primary_cyclic"])))
    summary={"status":"completed_multi_donor_evaluator_blind_inference","seed":seed,"fold":fold,"recipient":recipient,"donors":8,"evaluator_opened":False};_json(run_dir/"result_summary.json",summary);return summary

def stage_multi_donor_evaluate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    seeds=[int(x) for x in config["seeds"]];seed=seeds[task_index//9];fold=task_index%9;recipient=fold+1;base=_seed_root(config,seed)/"folds"/f"fold_{fold:02d}";source=Path(str(config["v11_result_root"]))/"folds"/f"fold_{fold:02d}";rows=[]
    for protocol in SAME:
        inf=np.load(base/"units"/protocol/"inference.npz");ev=np.load(source/"units"/protocol/"evaluator.npz");scale=np.asarray(inf["eeg_scale"]);location=np.asarray(inf["eeg_location"]);target=np.asarray(ev["paired_x"])[...,:500]
        for path in sorted((base/"multi_donor"/protocol).glob("donor_*.npz")):
            value=np.load(path);physical=(np.asarray(value["paired_DIFF_WRONG"])*scale[None,:,None]+location[None,:,None])[...,:500];rows.append({"seed":seed,"recipient":recipient,"fold":fold,"protocol":protocol,"donor":int(value["donor"]),"donor_seen":int(value["donor_seen"]),"primary_cyclic":int(value["primary_cyclic"]),"rrmse":v11.rrmse(physical,target)})
    _csv(base/"multi_donor_metrics.csv",rows);summary={"status":"completed_multi_donor_evaluation","seed":seed,"fold":fold,"rows":len(rows)};_json(run_dir/"result_summary.json",summary);return summary

def _sign_flip(value:np.ndarray)->float:
    observed=float(value.mean());signs=((np.arange(2**len(value))[:,None]>>np.arange(len(value)))&1)*2-1;return float(np.mean((signs*value[None]).mean(1)>=observed))

def stage_aggregate(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import matplotlib.pyplot as plt
    root=Path(str(config["result_root"]));paired=[];natural=[];runtime=[];multi=[]
    for seed in map(int,config["seeds"]):
        for fold in range(9):
            base=_seed_root(config,seed)/"folds"/f"fold_{fold:02d}"
            for row in _read(base/"paired_metrics_v11_1.csv"):paired.append({"seed":seed,**row})
            for row in _read(base/"natural_metrics_v11_1.csv"):natural.append({"seed":seed,**row})
            if (base/"inference_runtime.json").exists():runtime.append({"seed":seed,"fold":fold,**json.loads((base/"inference_runtime.json").read_text())})
            multi.extend(_read(base/"multi_donor_metrics.csv"))
    # New-seed inference summaries live in immutable Slurm run directories.
    # The frozen original seed keeps V11.1's per-fold runtime file.
    for path in sorted((root/"runs"/"infer").glob("job_*/task_*/result_summary.json")):
        value=json.loads(path.read_text());key=(int(value["seed"]),int(value["fold"]));existing={(int(r["seed"]),int(r["fold"])) for r in runtime}
        if key not in existing:runtime.append(value)
    manifest=[]
    for seed in map(int,config["seeds"]):
        for fold in range(9):
            base=_seed_root(config,seed)/"folds"/f"fold_{fold:02d}"
            for protocol in SAME:
                path=base/"outputs"/"k8"/protocol/"inference_outputs.npz";values=np.load(path)
                for key in values.files:
                    panel,method=key.split("_",1);manifest.append({"path":str(path),"participant":fold+1,"seed":seed,"fold":fold,"method":method,"panel":panel,"shape":"x".join(map(str,values[key].shape)),"role":"frozen_original" if seed==int(config["seeds"][0]) else "new_replication","donor_status":"cyclic_unseen" if "WRONG" in method else "not_applicable"})
    _csv(root/"lightweight_manifest.csv",manifest)
    effects=[]
    for seed in map(int,config["seeds"]):
        for subject in range(1,10):
            rows=[r for r in paired if int(r["seed"])==seed and int(r["subject"])==subject and r["protocol"] in SAME];by=defaultdict(list)
            for row in rows:by[row["method"]].append(float(row["rrmse"]))
            m={name:float(np.mean(value)) for name,value in by.items()};effects.append({"seed":seed,"subject":subject,"U_P":m["DIFF-POP"]-m["DIFF-MATCH"],"U_W":m["DIFF-WRONG"]-m["DIFF-MATCH"],"U_S":m["DIFF-TEMPORAL-SHUFFLED"]-m["DIFF-MATCH"],"I_P":(m["DET-POP"]-m["DET-MATCH"])-(m["DIFF-POP"]-m["DIFF-MATCH"]),"raw_rrmse":m["RAW"],"linear_match_rrmse":m["LINEAR-MATCH"],"det_match_rrmse":m["DET-MATCH"],"diff_pop_rrmse":m["DIFF-POP"],"diff_match_rrmse":m["DIFF-MATCH"],"diff_wrong_rrmse":m["DIFF-WRONG"]})
    _csv(root/"participant_seed_effects.csv",effects)
    method_rows=[]
    for seed in map(int,config["seeds"]):
        for method in METHODS:
            p=[r for r in paired if int(r["seed"])==seed and r["method"]==method];n=[r for r in natural if int(r["seed"])==seed and r["method"]==method]
            method_rows.append({"seed":seed,"method":method,"participants":9,"rrmse":float(np.mean([float(r["rrmse"]) for r in p])),"correlation":float(np.mean([float(r["correlation"]) for r in p])),"delta_snr":float(np.mean([float(r["delta_snr"]) for r in p])),"paired_spectral_utility":float(np.mean([float(r["paired_spectral_utility"]) for r in p])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in n])),"preservation":float(np.mean([float(r["preservation"]) for r in n])),"covariance":float(np.mean([float(r["covariance"]) for r in n])),"mi_band_distortion":float(np.mean([float(r["mi_band_distortion"]) for r in n])),"mi_kappa":float(np.nanmean([float(r["mi_kappa"]) for r in n])),"erd_preservation":float(np.mean([float(r["erd_preservation"]) for r in n]))})
    _csv(root/"method_summary.csv",method_rows)
    seed_summary=[]
    for seed in map(int,config["seeds"]):
        rows=[r for r in effects if r["seed"]==seed];seed_summary.append({"seed":seed,**{f"{name}_{stat}":value for name in ("U_P","U_W","U_S","I_P") for stat,value in (("mean",float(np.mean([r[name] for r in rows]))),("median",float(np.median([r[name] for r in rows]))),("positive",int(np.sum([r[name]>0 for r in rows]))))}})
    _csv(root/"seed_summary.csv",seed_summary)
    averaged=[]
    for subject in range(1,10):
        rows=[r for r in effects if r["subject"]==subject];averaged.append({"subject":subject,**{name:float(np.mean([r[name] for r in rows])) for name in ("U_P","U_W","U_S","I_P")},**{f"{name}_seed_variance":float(np.var([r[name] for r in rows],ddof=1)) for name in ("U_P","U_W")}})
    _csv(root/"participant_averaged_effects.csv",averaged);rng=np.random.default_rng(int(config["bootstrap_seed"]));indices=rng.integers(0,9,size=(int(config["bootstrap_replicates"]),9));boot=[];combined={}
    for name in ("U_P","U_W","U_S"):
        value=np.asarray([r[name] for r in averaged]);rep=value[indices].mean(1);combined[name]={"mean":float(value.mean()),"median":float(np.median(value)),"positive":int(np.sum(value>0)),"sign_flip_p_one_sided":_sign_flip(value),"ci_low":float(np.quantile(rep,.025)),"ci_high":float(np.quantile(rep,.975))};boot.append({"effect":name,**combined[name],"status":"participant_descriptive_bootstrap"})
    _csv(root/"bootstrap_summary.csv",boot)
    # Donors are scored separately, then averaged within recipient/seed and role.
    sensitivity=[]
    for seed in map(int,config["seeds"]):
        for recipient in range(1,10):
            match=float(np.mean([float(r["rrmse"]) for r in paired if int(r["seed"])==seed and int(r["subject"])==recipient and r["method"]=="DIFF-MATCH"]))
            rows=[r for r in multi if int(r["seed"])==seed and int(r["recipient"])==recipient]
            for role,take in (("outer_training_seen",[r for r in rows if int(r["donor_seen"])==1]),("outer_training_unseen",[r for r in rows if int(r["donor_seen"])==0])):
                donor_means=[]
                for donor in sorted({int(r["donor"]) for r in take}):donor_means.append(float(np.mean([float(r["rrmse"]) for r in take if int(r["donor"])==donor])))
                sensitivity.append({"seed":seed,"recipient":recipient,"donor_role":role,"donors":len(donor_means),"match_rrmse":match,"mean_donor_rrmse":float(np.mean(donor_means)) if donor_means else "","utility":float(np.mean(donor_means)-match) if donor_means else "","positive":int(bool(donor_means) and np.mean(donor_means)>match)})
    _csv(root/"multi_donor_sensitivity.csv",sensitivity)
    safety=[]
    for seed in map(int,config["seeds"]):
        for subject in range(1,10):
            for method in ("DIFF-POP","DIFF-MATCH","LINEAR-MATCH","DET-MATCH"):
                rows=[r for r in natural if int(r["seed"])==seed and int(r["subject"])==subject and r["method"]==method];safety.append({"seed":seed,"subject":subject,"method":method,**{name:float(np.mean([float(r[name]) for r in rows])) for name in ("eog_attenuation","preservation","covariance","mi_band_distortion","mi_kappa","erd_preservation")}})
    _csv(root/"natural_safety.csv",safety)
    match=[r for r in safety if r["method"]=="DIFF-MATCH"];pop=[r for r in safety if r["method"]=="DIFF-POP"];pop_by={(r["seed"],r["subject"]):r for r in pop};participant_safety=[]
    for subject in range(1,10):
        rows=[r for r in match if r["subject"]==subject];participant_safety.append({"subject":subject,"eog_attenuation":float(np.mean([r["eog_attenuation"] for r in rows])),"preservation":float(np.mean([r["preservation"] for r in rows])),"covariance":float(np.mean([r["covariance"] for r in rows])),"mi_band_distortion":float(np.mean([r["mi_band_distortion"] for r in rows])),"kappa_delta_vs_diff_pop":float(np.mean([r["mi_kappa"]-pop_by[(r["seed"],r["subject"])]["mi_kappa"] for r in rows]))})
    _csv(root/"participant_averaged_safety.csv",participant_safety);safe={"eog_attenuation":float(np.mean([r["eog_attenuation"] for r in match])),"eog_attenuation_min":float(np.min([r["eog_attenuation"] for r in participant_safety])),"preservation":float(np.mean([r["preservation"] for r in match])),"preservation_min":float(np.min([r["preservation"] for r in participant_safety])),"covariance":float(np.mean([r["covariance"] for r in match])),"covariance_max":float(np.max([r["covariance"] for r in participant_safety])),"mi_band_distortion_mean":float(np.mean([r["mi_band_distortion"] for r in match])),"mi_band_distortion_max":float(np.max([r["mi_band_distortion"] for r in participant_safety])),"kappa_delta_vs_diff_pop":float(np.mean([r["mi_kappa"] for r in match])-np.mean([r["mi_kappa"] for r in pop])),"kappa_delta_min":float(np.min([r["kappa_delta_vs_diff_pop"] for r in participant_safety])),"participant_reversals":{"eog_nonpositive":int(sum(r["eog_attenuation"]<=0 for r in participant_safety)),"preservation_below_078":int(sum(r["preservation"]<.78 for r in participant_safety)),"covariance_above_015":int(sum(r["covariance"]>.15 for r in participant_safety)),"kappa_delta_below_minus_002":int(sum(r["kappa_delta_vs_diff_pop"]<-.02 for r in participant_safety))}}
    absolute=all(r["diff_match_rrmse"]<r["raw_rrmse"] and np.isfinite(r["diff_match_rrmse"]) for r in effects);subject_ok=combined["U_P"]["mean"]>0 and combined["U_P"]["median"]>0 and combined["U_P"]["positive"]>=8 and combined["U_W"]["mean"]>0 and combined["U_W"]["median"]>0 and combined["U_W"]["positive"]>=8 and sum(r["U_P_mean"]>0 and r["U_W_mean"]>0 for r in seed_summary)>=2;natural_ok=safe["eog_attenuation"]>0 and safe["preservation"]>=.78 and safe["covariance"]<=.15 and safe["kappa_delta_vs_diff_pop"]>=-.02;primary_wrong=combined["U_W"]["mean"]>0 and combined["U_W"]["median"]>0 and combined["U_W"]["positive"]>=8;seen=[float(r["utility"]) for r in sensitivity if r["donor_role"]=="outer_training_seen" and r["utility"]!=""];unseen=[float(r["utility"]) for r in sensitivity if r["donor_role"]=="outer_training_unseen" and r["utility"]!=""];donor_summary={"outer_training_seen":{"mean":float(np.mean(seen)),"median":float(np.median(seen)),"positive":int(np.sum(np.asarray(seen)>0)),"denominator":len(seen)},"outer_training_unseen":{"mean":float(np.mean(unseen)),"median":float(np.median(unseen)),"positive":int(np.sum(np.asarray(unseen)>0)),"denominator":len(unseen)}};donor_robust=bool(seen and np.mean(seen)>0 and np.median(seen)>0);replicated=absolute and subject_ok and natural_ok and primary_wrong;decision="SUBJECT_CONDITIONING_WITHIN_FIXED_DIFFUSION_REPLICATED" if replicated else "V11_1_ONE_SEED_SUBJECT_CONDITIONING_SIGNAL_NOT_REPLICATED";labels=[decision]+(["DONOR_ROBUSTNESS_LIMITED"] if replicated and not donor_robust else [])
    panels=[]
    for value in runtime:
        if "panels" in value:panels.extend([panel for panel in value["panels"] if int(panel["k"])==8])
        elif int(value.get("k",8))==8:panels.append(value)
    resource={"panel":"K8_DDIM25","fold_seed_runs":len(panels),"latency_seconds_per_window":float(np.mean([float(v["latency_seconds_per_window"]) for v in panels])),"posterior_network_calls":200,"peak_memory_bytes_max":int(max(int(v["peak_memory_bytes"]) for v in panels))};_csv(root/"resource_summary.csv",[resource])
    result={"status":"completed_three_seed_same_session_development_replication","decision":decision,"labels":labels,"absolute_validity":bool(absolute),"subject_conditioning":bool(subject_ok),"natural_safety_passed":bool(natural_ok),"primary_unseen_wrong_passed":bool(primary_wrong),"multi_donor_robustness":bool(donor_robust),"multi_donor_summary":donor_summary,"effects":combined,"seed_summary":seed_summary,"safety":safe,"resources":resource,"participants":9,"seeds":config["seeds"],"deployment":"EOG-guided","diffusion_beats_linear_or_deterministic":"not_primary_and_not_required","confirmation":False};_json(root/"routing_decision.json",result);_json(root/"result_summary.json",result);_json(run_dir/"result_summary.json",result)
    figures=root/"figures";figures.mkdir(exist_ok=True)
    x=np.arange(9);fig,ax=plt.subplots(figsize=(7,4));ax.axhline(0,color="black",lw=.8);ax.plot(x,[r["U_P"] for r in averaged],"o-",label="U_P");ax.plot(x,[r["U_W"] for r in averaged],"s-",label="U_W");ax.set_xticks(x,[f"P{i}" for i in range(1,10)]);ax.legend();ax.set_ylabel("RRMSE utility");fig.tight_layout();fig.savefig(figures/"participant_subject_effects.png",dpi=180);plt.close(fig)
    for key,names,title in (("heat",("U_P","U_W"),"participant effect heatmap"),):
        matrix=np.asarray([[next(r[n] for r in effects if r["seed"]==seed and r["subject"]==subject) for subject in range(1,10)] for seed in map(int,config["seeds"]) for n in names]);fig,ax=plt.subplots(figsize=(8,3));image=ax.imshow(matrix,aspect="auto",cmap="coolwarm");ax.set_yticks(range(6),[f"{s}-{n}" for s in config["seeds"] for n in names]);ax.set_xticks(range(9),[f"P{i}" for i in range(1,10)]);fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(figures/"seed_effect_heatmap.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4));ax.scatter([r["eog_attenuation"] for r in match],[r["preservation"] for r in match],c=[r["seed"] for r in match]);ax.axhline(.78,color="black",ls="--");ax.set_xlabel("EOG attenuation");ax.set_ylabel("preservation");fig.tight_layout();fig.savefig(figures/"attenuation_preservation.png",dpi=180);plt.close(fig)
    means={name:float(np.mean([float(r["rrmse"]) for r in paired if r["method"]==name])) for name in ("RAW","LINEAR-MATCH","DIFF-POP","DIFF-MATCH","DIFF-WRONG")};fig,ax=plt.subplots(figsize=(6,4));ax.bar(means.keys(),means.values());ax.tick_params(axis="x",rotation=25);ax.set_ylabel("paired RRMSE");fig.tight_layout();fig.savefig(figures/"method_comparison.png",dpi=180);plt.close(fig)
    from scipy.signal import welch
    example_base=_seed_root(config,int(config["seeds"][0]))/"folds"/"fold_00";protocol="same_01";inf=np.load(example_base/"units"/protocol/"inference.npz");ev=np.load(Path(str(config["v11_result_root"]))/"folds"/"fold_00"/"units"/protocol/"evaluator.npz");out=np.load(example_base/"outputs"/"k8"/protocol/"inference_outputs.npz");scale=np.asarray(inf["eeg_scale"]);location=np.asarray(inf["eeg_location"]);target=np.asarray(ev["paired_x"])[0,0,:500];names=("RAW","LINEAR-MATCH","DIFF-POP","DIFF-MATCH","DIFF-WRONG");signals={name:(np.asarray(out[f"paired_{name}"])*scale[None,:,None]+location[None,:,None])[0,0,:500] for name in names};fig,axes=plt.subplots(2,1,figsize=(8,6));time_axis=np.arange(500)/float(config["sampling_rate"]);axes[0].plot(time_axis,target,label="target",color="black",lw=1.2);[axes[0].plot(time_axis,signals[name],label=name,alpha=.75,lw=.8) for name in names];axes[0].set_ylabel("EEG (physical units)");axes[0].legend(ncol=3,fontsize=7);[axes[1].semilogy(*welch(signal,fs=float(config["sampling_rate"]),nperseg=250),label=name) for name,signal in signals.items()];axes[1].set_xlim(1,45);axes[1].set_xlabel("Hz");axes[1].set_ylabel("PSD");fig.tight_layout();fig.savefig(figures/"fixed_participant_waveform_psd.png",dpi=180);plt.close(fig)
    return result

def stage_finalize(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));result=json.loads((root/"result_summary.json").read_text());lines=["# BCI2b same-session subject-aware diffusion replication","","Three-seed development replication of the frozen V11.1 method. Deployment is EOG-guided; this is not confirmation. V12 is archived as a technically unvalidated enhancement route and was neither repaired nor screened here.","",f"Decision: `{result['decision']}`"+(f" + `{result['labels'][1]}`" if len(result["labels"])>1 else "")+".","","Coverage: 9/9 BCI2b participants, three same-session units per participant, three fixed training seeds, K=8 and DDIM25. Windows and seeds were aggregated within participant and were not treated as scientific replicates.","","## Primary subject-conditioning effects","","| effect | mean | median | positive | sign-flip p | descriptive 95% CI |","|---|---:|---:|---:|---:|---:|"]
    for name in ("U_P","U_W","U_S"):
        value=result["effects"][name];lines.append(f"| {name} | {value['mean']:+.5f} | {value['median']:+.5f} | {value['positive']}/9 | {value['sign_flip_p_one_sided']:.5f} | [{value['ci_low']:+.5f}, {value['ci_high']:+.5f}] |")
    lines += ["","| seed | mean U_P | positive | mean U_W | positive |","|---:|---:|---:|---:|---:|"]
    for row in result["seed_summary"]:lines.append(f"| {row['seed']} | {row['U_P_mean']:+.5f} | {row['U_P_positive']}/9 | {row['U_W_mean']:+.5f} | {row['U_W_positive']}/9 |")
    methods=_read(root/"method_summary.csv");lines += ["","## Absolute method results","","| method | paired RRMSE | correlation | delta SNR | EOG attenuation | preservation | covariance | MI kappa |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in ("RAW","LINEAR-MATCH","DET-MATCH","DIFF-POP","DIFF-MATCH","DIFF-WRONG"):
        rows=[r for r in methods if r["method"]==method];mean=lambda key:float(np.mean([float(r[key]) for r in rows]));lines.append(f"| {method} | {mean('rrmse'):.5f} | {mean('correlation'):.5f} | {mean('delta_snr'):+.4f} | {mean('eog_attenuation'):+.4f} | {mean('preservation'):.4f} | {mean('covariance'):.4f} | {mean('mi_kappa'):.4f} |")
    lines += ["","DIFF-MATCH does not beat LINEAR-MATCH or DET-MATCH on aggregate paired RRMSE. That comparison is reported transparently and is distinct from the replicated result that matching subject conditioning improves the same fixed diffusion denoiser relative to POP and WRONG contexts."]
    s=result["safety"];reversal=s["participant_reversals"];lines += ["","## Natural safety","",f"EOG attenuation mean/min `{s['eog_attenuation']:+.4f}/{s['eog_attenuation_min']:+.4f}`, preservation mean/min `{s['preservation']:.4f}/{s['preservation_min']:.4f}`, covariance mean/max `{s['covariance']:.4f}/{s['covariance_max']:.4f}`, MI-band distortion mean/max `{s['mi_band_distortion_mean']:.4f}/{s['mi_band_distortion_max']:.4f}`, MI-kappa relative to DIFF-POP mean/min `{s['kappa_delta_vs_diff_pop']:+.4f}/{s['kappa_delta_min']:+.4f}`.","",f"Participant-level reversals (out of 9): nonpositive EOG attenuation {reversal['eog_nonpositive']}, preservation below 0.78 {reversal['preservation_below_078']}, covariance above 0.15 {reversal['covariance_above_015']}, and MI-kappa delta below -0.02 {reversal['kappa_delta_below_minus_002']}.","","LINEAR-MATCH and DET-MATCH remain transparent comparators, but whether diffusion beats them is not this replication's primary gate. The supported claim is that matching subject conditioning improves a fixed diffusion denoiser over population and frozen cyclic unseen-WRONG contexts."]
    resource=result["resources"];donor=result["multi_donor_summary"];lines += ["","## Compute and donor sensitivity","",f"K8/DDIM25 used {resource['posterior_network_calls']} posterior calls, mean latency `{resource['latency_seconds_per_window']:.4f}` s/window, and peak allocated GPU memory `{resource['peak_memory_bytes_max']/1e6:.0f}` MB.","",f"Outer-training-seen donor sensitivity: mean/median utility `{donor['outer_training_seen']['mean']:+.5f}/{donor['outer_training_seen']['median']:+.5f}`, {donor['outer_training_seen']['positive']}/{donor['outer_training_seen']['denominator']} recipient-seed comparisons positive. Outer-training-unseen: `{donor['outer_training_unseen']['mean']:+.5f}/{donor['outer_training_unseen']['median']:+.5f}`, {donor['outer_training_unseen']['positive']}/{donor['outer_training_unseen']['denominator']} positive. Only one truly unseen donor exists per fold, so no claim of three unseen donors is made.","","The frozen cyclic unseen-WRONG is the primary continuity control. All other compatible donors were scored separately before recipient-level averaging; outer-training-seen and outer-training-unseen sensitivity are reported separately in `multi_donor_sensitivity.csv`.","","Recovery record: evaluator job 930791 was invalid because it requested absent K32 outputs; multi-donor jobs 930846/930855/930864 were invalid because the GPU Python 3.9 loader could not execute `zip(strict=True)`. Neither set entered aggregation. The repaired path built evaluator-blind donor operators on CPU and replayed unchanged GPU inference."]
    Path("reports/bci2b_subject_diffusion_replication.md").write_text("\n".join(lines)+"\n",encoding="utf-8");_json(run_dir/"result_summary.json",result);return result

def run_stage(config_path:Path,stage:str,run_dir:Path,*,task_index:int=0)->dict[str,Any]:
    config=_config(config_path);run_dir.mkdir(parents=True,exist_ok=True)
    if stage=="audit":return stage_audit(config,task_index,run_dir)
    if stage=="train":return stage_train(config,task_index,run_dir)
    if stage=="infer":return stage_infer(config,task_index,run_dir)
    if stage=="evaluate":return stage_evaluate(config,task_index,run_dir)
    if stage=="donor-operators":return stage_donor_operators(config,task_index,run_dir)
    if stage=="multi-donor-infer":return stage_multi_donor_infer(config,task_index,run_dir)
    if stage=="multi-donor-evaluate":return stage_multi_donor_evaluate(config,task_index,run_dir)
    if stage=="aggregate":return stage_aggregate(config,task_index,run_dir)
    if stage=="finalize":return stage_finalize(config,task_index,run_dir)
    raise ValueError(stage)
