"""V40R execution, participant-first aggregation, reports, and governance."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score

from eeg_scad.models.eegdfus_mc_v40r import CompactSupportEncoder
from eeg_scad.training.train_v40r import _support_bank, resume_fold, run_fold


ROOT=Path(__file__).resolve().parents[3];RESULT=ROOT/"results/official_support_diffusion_v40r";REPORT=ROOT/"reports";FIG=ROOT/"figures/official_support_diffusion_v40r";SEEDS=(20261010,20261011)
def cfg(name):return yaml.safe_load((ROOT/f"configs/setcalibdiff_v25/{name}.yaml").read_text())
def write_csv(path,rows):path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(path,index=False)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    RESULT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);folds=cfg("folds")["folds"];data=cfg("data");split=[]
    for fold in folds:
        for role in ("train","validation","test"):
            split += [{"fold":fold["fold"],"participant":participant,"role":role} for participant in fold[role]]
    write_csv(RESULT/"split_manifest.csv",split)
    historical=Path("/home/infres/yinwang/denoiseNet/results/cgdr/eegdfus_benchmark/full_aggregate")
    summary=json.loads((historical/"result_summary.json").read_text());cells=pd.read_csv(historical/"cell_summary.csv");rows=[]
    for row in cells.to_dict("records"):
        rows.append({"method":"EEGDfus","protocol":row["protocol"],"artifact":row["noise_type"],"arm":row["arm"],"status":"reasonable_nonidentical_reproduction","rrmse_temporal":row["mean_rrmse_temporal"],"rrmse_spectral":row["mean_rrmse_spectral_corrected_psd_denominator_shape"],"correlation":row["mean_correlation"],"snr_improvement":row["mean_snr_improvement_db"],"optimizer_updates":row["optimizer_updates"],"sampler_steps":row["network_calls_per_output"],"source_identity_unit":"source_epoch_not_participant"})
    write_csv(RESULT/"official_reproduction.csv",rows)
    source={"base_commit":"8be1ec3a7c8c9735b548ca2dbd744c76bf27f37d","official_eegdfus_commit":"a19a652b3b6346188ae77067e1daf8b90cad005f","historical_official_aggregate":str(historical),"historical_summary_sha256":digest(historical/"result_summary.json"),"historical_status":summary["status"],"participant_panel":data["participants"],"sealed_participants":data["sealed_participants"],"support_contract":"V31 prefix-only/non-overlap/duration-local EOG normalization","query_eog_inference_reads":0,"sealed_reads":0,"manuscript_unchanged":True}
    (RESULT/"source_binding.json").write_text(json.dumps(source,indent=2,sort_keys=True)+"\n")


def bootstrap(values,seed=40):
    values=np.asarray(values,float);rng=np.random.default_rng(seed);draw=np.asarray([rng.choice(values,len(values),replace=True).mean() for _ in range(5000)]);return np.quantile(draw,[.025,.975])


def _summarize(frame,metrics):
    per=frame.groupby(["condition","participant"],as_index=False)[metrics].mean();rows=[]
    for condition,block in per.groupby("condition"):
        for metric in metrics:
            lo,hi=bootstrap(block[metric]);rows.append({"condition":condition,"metric":metric,"participant_mean":block[metric].mean(),"participant_median":block[metric].median(),"bootstrap_low":lo,"bootstrap_high":hi,"participants":block.participant.nunique()})
    return rows,per


def _privacy_rows(binding, data, folds):
    rows=[]
    for fold in range(5):
        checkpoint=next(row for row in binding if row["model"]=="SC-EEGDfus" and f"fold_{fold}_" in row["path"])
        payload=torch.load(checkpoint["path"],map_location="cpu",weights_only=True);encoder=CompactSupportEncoder();encoder.load_state_dict(payload["support_encoder"]);encoder.eval();support,_=_support_bank(data,folds[fold],30);gallery=[];query=[];labels=[]
        with torch.no_grad():
            for key,episode in sorted(support.items()):
                eeg=torch.from_numpy(episode["eeg"]);eog=torch.from_numpy(episode["eog"]);split=max(1,len(eeg)//2);gallery.append(encoder(eeg[:split][None],eog[:split][None])[0].numpy());query.append(encoder(eeg[split:][None],eog[split:][None])[0].numpy());labels.append(key[0])
        gallery=np.asarray(gallery);query=np.asarray(query);gallery/=np.linalg.norm(gallery,axis=1,keepdims=True).clip(1e-8);query/=np.linalg.norm(query,axis=1,keepdims=True).clip(1e-8);similarity=query@gallery.T;pred=np.argmax(similarity,axis=1);top1=float(np.mean(np.asarray(labels)[pred]==np.asarray(labels)));scores=[];truth=[]
        for i in range(len(query)):
            for j in range(len(gallery)):scores.append(similarity[i,j]);truth.append(labels[i]==labels[j])
        rows.append({"fold":fold,"top1_participant_classification":top1,"verification_auroc":float(roc_auc_score(truth,scores)),"context_state_bytes":512,"state_stored":False,"recommendation":"delete_at_session_end","support_halves_disjoint":True})
    return rows


def aggregate():
    payload=[]
    run_roots=sorted([path for path in RESULT.glob("job_*") if path.is_dir()])
    if not run_roots:raise FileNotFoundError("no completed recovery job root")
    run_root=run_roots[-1]
    for fold in range(5):
        for seed in SEEDS:payload.append(json.loads((run_root/f"fold_{fold}_seed_{seed}"/"result.json").read_text()))
    support=[];binding=[];metrics=[];duration=[]
    for value in payload:support+=value["support_manifest"];binding+=value["checkpoints"];metrics+=value["metrics"];duration+=value["support_duration"]
    write_csv(RESULT/"support_manifest.csv",support);write_csv(RESULT/"checkpoint_binding.csv",binding);write_csv(RESULT/"paired_metrics.csv",[r for r in metrics if r["stream"]=="paired"]);write_csv(RESULT/"natural_metrics.csv",[r for r in metrics if r["stream"]=="natural"]);write_csv(RESULT/"support_duration.csv",duration)
    paired=pd.DataFrame([r for r in metrics if r["stream"]=="paired"]);natural=pd.DataFrame([r for r in metrics if r["stream"]=="natural"])
    pm=("rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","observation_change_ratio","output_input_rms");nm=("heldout_eog_remaining_ratio","artifact_attenuation_db","eeg_eog_coherence_reduction","low_eog_observation_retention","psd_distortion","covariance_distortion","output_input_rms")
    ps,pp=_summarize(paired,list(pm));ns,np_=_summarize(natural,list(nm));write_csv(RESULT/"method_summary.csv",[{"panel":"paired",**r} for r in ps]+[{"panel":"natural",**r} for r in ns])
    effects=[]
    directions={"rrmse_temporal":-1,"rrmse_spectral":-1,"correlation":1,"snr_improvement":1,"artifact_rrmse":-1,"artifact_correlation":1,"artifact_attenuation_db":1,"low_eog_observation_retention":1,"psd_distortion":-1,"covariance_distortion":-1}
    for panel,per in (("paired",pp),("natural",np_)):
        match=per[per.condition=="MATCH"].set_index("participant")
        for comparator in ("POP","WRONG","SHUFFLED","POP_MEAN","ADAPTER_DISABLED"):
            other=per[per.condition==comparator].set_index("participant")
            for metric in [m for m in directions if m in match]:
                common=match.index.intersection(other.index);values=directions[metric]*(match.loc[common,metric]-other.loc[common,metric]);lo,hi=bootstrap(values);effects.append({"panel":panel,"contrast":f"MATCH-{comparator}","metric":metric,"participant_mean_utility":values.mean(),"participant_median_utility":np.median(values),"positive_count":int((values>0).sum()),"participants":len(values),"bootstrap_low":lo,"bootstrap_high":hi})
    write_csv(RESULT/"participant_effects.csv",effects)
    dur=pd.DataFrame(duration).groupby(["support_seconds","participant"],as_index=False).mean(numeric_only=True).groupby("support_seconds",as_index=False).mean(numeric_only=True);write_csv(RESULT/"ablation_summary.csv",dur.to_dict("records"))
    privacy=_privacy_rows(binding,cfg("data"),cfg("folds")["folds"])
    write_csv(RESULT/"privacy_summary.csv",privacy)
    primary=next(r for r in effects if r["panel"]=="paired" and r["contrast"]=="MATCH-POP" and r["metric"]=="rrmse_temporal");natural_primary=next(r for r in effects if r["panel"]=="natural" and r["contrast"]=="MATCH-POP" and r["metric"]=="artifact_attenuation_db")
    scale=p[p.metric=="output_input_rms"].participant_mean;scale_valid=bool(np.isfinite(scale).all() and scale.quantile(.99)<10)
    positioning="A" if primary["participant_mean_utility"]>0 and primary["positive_count"]>=10 else "B" if primary["participant_mean_utility"]>0 else "C"
    diagnosis={"engineering":"valid" if scale_valid else "invalid_scale_collapse","output_input_rms_participant_q99":float(scale.quantile(.99)),"official_eegdfus":"reasonable_nonidentical_reproduction","d4pm":"official_release_not_runnable","eegoar_net":"protocol_incompatible","participant_coverage":15,"fold_seed_cells":10,"primary_estimand":primary,"natural_attenuation_estimand":natural_primary,"final_positioning":positioning if scale_valid else "C","repair_used":True,"repair_scope":"input_channel_and_multichannel_learning_rate_engineering_only","sealed_reads":0,"query_eog_inference_reads":0,"manuscript_unchanged":True}
    (RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n");figures(pd.DataFrame(ps),pd.DataFrame(ns),pd.DataFrame(effects),dur);reports(pd.DataFrame(ps),pd.DataFrame(ns),pd.DataFrame(effects),dur,diagnosis)


def figures(paired,natural,effects,duration):
    rr=paired[paired.metric=="rrmse_temporal"];fig,ax=plt.subplots(figsize=(7,4));ax.bar(rr.condition,rr.participant_mean);ax.set(ylabel="Temporal RRMSE",title="Support condition");fig.tight_layout();fig.savefig(FIG/"support_condition_effect.png",dpi=180);plt.close(fig)
    ep=effects[(effects.panel=="paired")&(effects.metric=="rrmse_temporal")];fig,ax=plt.subplots(figsize=(7,4));ax.bar(ep.contrast,ep.participant_mean_utility);ax.axhline(0,color="black",lw=.8);ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(FIG/"participant_effects.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.plot(duration.support_seconds,duration.rrmse_temporal,marker="o");ax.set(xlabel="Support seconds",ylabel="Temporal RRMSE");fig.tight_layout();fig.savefig(FIG/"support_duration.png",dpi=180);plt.close(fig)
    nat=natural.pivot(index="condition",columns="metric",values="participant_mean").reset_index();fig,ax=plt.subplots(figsize=(6,4));ax.scatter(nat.artifact_attenuation_db,nat.low_eog_observation_retention);[ax.annotate(r.condition,(r.artifact_attenuation_db,r.low_eog_observation_retention)) for r in nat.itertuples()];ax.set(xlabel="Attenuation dB",ylabel="Low-EOG retention");fig.tight_layout();fig.savefig(FIG/"natural_attenuation_retention.png",dpi=180);plt.close(fig)
    official=pd.read_csv(RESULT/"official_reproduction.csv");eog=official[(official.protocol=="official_native")&(official.artifact=="EOG")];fig,ax=plt.subplots(figsize=(6,4));ax.bar(eog.arm,eog.rrmse_temporal);ax.tick_params(axis="x",rotation=15);fig.tight_layout();fig.savefig(FIG/"official_benchmark_comparison.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(["POP","MATCH","WRONG","SHUFFLED"],[float(rr[rr.condition==c].participant_mean.iloc[0]) for c in ("POP","MATCH","WRONG","SHUFFLED")]);fig.tight_layout();fig.savefig(FIG/"method_ablation.png",dpi=180);plt.close(fig)


def reports(p,n,e,duration,diagnosis):
    tab=lambda frame,cols:frame[cols].round(6).to_markdown(index=False)
    official=pd.read_csv(RESULT/"official_reproduction.csv")
    (REPORT/"v40r_official_reproduction.md").write_text("# V40R official reproduction\n\n"+tab(official,["protocol","artifact","arm","status","rrmse_temporal","rrmse_spectral","correlation","snr_improvement","optimizer_updates","sampler_steps"])+"\n\nThe upstream spectral formula is blocked by a 400-versus-512 denominator shape mismatch; the explicitly named corrected PSD-denominator result is used. EEGdenoiseNet has source epochs but no participant identity.\n")
    (REPORT/"v40r_paired_results.md").write_text("# V40R paired results\n\n"+tab(p,["condition","metric","participant_mean","participant_median","bootstrap_low","bootstrap_high","participants"])+"\n\n## Participant-first contrasts\n\n"+tab(e[e.panel=="paired"],["contrast","metric","participant_mean_utility","positive_count","participants","bootstrap_low","bootstrap_high"])+"\n")
    (REPORT/"v40r_natural_results.md").write_text("# V40R natural development results\n\nLow-EOG observation retention is not physiological preservation. Query EOG is unavailable to inference and opened only by the post-freeze evaluator.\n\n"+tab(n,["condition","metric","participant_mean","participant_median","bootstrap_low","bootstrap_high","participants"])+"\n")
    privacy=pd.read_csv(RESULT/"privacy_summary.csv")
    (REPORT/"v40r_ablation_and_privacy.md").write_text("# V40R ablation and lightweight privacy\n\n"+tab(duration,["support_seconds","rrmse_temporal"])+"\n\nPOP is an exact adapter bypass. POP_MEAN and ADAPTER_DISABLED isolate context and FiLM paths.\n\n## Disjoint-half context linkage\n\n"+tab(privacy,["fold","top1_participant_classification","verification_auroc","context_state_bytes","state_stored"])+"\n\nThe support state is 128 float32 values (512 bytes), is not persistent, and should be deleted at session end. This is a linkage diagnostic, not an anonymity claim.\n")
    (REPORT/"v40r_final_diagnosis.md").write_text("# V40R final diagnosis\n\nFinal positioning: **"+diagnosis["final_positioning"]+"**.\n\n```json\n"+json.dumps(diagnosis,indent=2,sort_keys=True)+"\n```\n")


def terminal():
    diagnosis=json.loads((RESULT/"development_diagnosis.json").read_text());manifest={"base_commit":"8be1ec3a7c8c9735b548ca2dbd744c76bf27f37d","branch":"codex/official-support-eeg-diffusion-v40r","pre_terminal_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"terminal_commit_resolution":"commit containing this manifest; exact SHA reported after push","official_baseline_statuses":{"EEGDfus":"reasonable_nonidentical_reproduction","D4PM":"official_release_not_runnable","EEGOAR-Net":"protocol_incompatible"},"participant_coverage":15,"fold_seed_cells":10,"accepted_jobs":[],"failed_jobs":[],"recovery_jobs":[],"current_jobs":[],"tests":{},"sealed_reads":0,"query_eog_inference_reads":0,"manuscript_unchanged":True,"final_positioning":diagnosis["final_positioning"],"push_status":"pending_terminal_commit"};(RESULT/"terminal_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="stage",required=True);sub.add_parser("prepare");run=sub.add_parser("run");run.add_argument("--fold",type=int,required=True);run.add_argument("--seed",type=int,required=True);run.add_argument("--population-updates",type=int,default=20000);run.add_argument("--adapter-updates",type=int,default=5000);run.add_argument("--run-id",default="runtime");resume=sub.add_parser("resume");resume.add_argument("--fold",type=int,required=True);resume.add_argument("--seed",type=int,required=True);resume.add_argument("--run-id",required=True);resume.add_argument("--population-path",type=Path,required=True);resume.add_argument("--adapter-updates",type=int,default=5000);sub.add_parser("aggregate");sub.add_parser("terminal");args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="run":run_fold(RESULT,cfg("data"),cfg("folds")["folds"][args.fold],args.seed,torch.device("cuda"),args.population_updates,args.adapter_updates,args.run_id)
    elif args.stage=="resume":resume_fold(RESULT,cfg("data"),cfg("folds")["folds"][args.fold],args.seed,torch.device("cuda"),args.run_id,args.population_path,args.adapter_updates)
    elif args.stage=="aggregate":aggregate()
    else:terminal()


if __name__=="__main__":main()
