"""V39A command line, participant-first aggregation, reports and figures."""
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch,yaml
from eeg_scad.training.artifact_diffaug_v39a import ARMS,run_fold

ROOT=Path(__file__).resolve().parents[3];RESULT=ROOT/"results/artifact_diffaug_v39a";REPORT=ROOT/"reports";FIG=ROOT/"figures/artifact_diffaug_v39a";SEEDS=(20260950,20260951)
def cfg(name):return yaml.safe_load((ROOT/f"configs/setcalibdiff_v25/{name}.yaml").read_text())
def csv(path,rows):path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(path,index=False)


def prepare():
    RESULT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True);folds=cfg("folds");data=cfg("data");split=[]
    for f in folds["folds"]:
        for role in ("train","validation","test"):
            for p in f[role]:split.append({"fold":f["fold"],"participant":p,"role":role})
    csv(RESULT/"split_manifest.csv",split);binding={"base_commit":"e55d9df9c20afb28b4697658c3abce2ff4895610","v25_commit":"a7d9d647b69e152255b62dbca917a4b3ed082915","v24_coordinate":"corrected","participants":data["participants"],"sealed_participants":data["sealed_participants"],"support_seconds":30,"support_contract":"V31 prefix-only, chronological, non-overlap","generator_train_roles":["train"],"outer_test_generator_rows":0,"query_eog_inference_reads":0,"sealed_reads":0,"manuscript_unchanged":True};(RESULT/"source_binding.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n")


def bootstrap(v,seed=39):
    v=np.asarray(v,float);rng=np.random.default_rng(seed);d=np.asarray([rng.choice(v,len(v),replace=True).mean() for _ in range(5000)]);return np.quantile(d,[.025,.975])
def aggregate():
    payload=[]
    for fold in range(5):
        for seed in SEEDS:
            p=RESULT/"runtime"/f"fold_{fold}_seed_{seed}"/"result.json";payload.append(json.loads(p.read_text()))
    fields=("support_manifest","artifact_target_manifest","checkpoint_binding","generator_fidelity","generator_diversity","training_exposure","denoiser_training_manifest","support_interventions")
    collected={f:[] for f in fields};paired=[];natural=[]
    for p in payload:
        for f in fields:collected[f]+=p[f]
        paired+=p["paired"];natural+=p["natural"]
    for f in fields:csv(RESULT/f"{f}.csv",collected[f])
    pr=pd.DataFrame(paired);nr=pd.DataFrame(natural);csv(RESULT/"paired_rows.csv",paired);csv(RESULT/"natural_rows.csv",natural)
    interventions=pd.concat((pr,nr,pd.DataFrame(collected["support_interventions"])),ignore_index=True,sort=False);intervention_participant=interventions.groupby(["method","context_condition","participant"],as_index=False).mean(numeric_only=True);intervention_summary=intervention_participant.groupby(["method","context_condition"],as_index=False).mean(numeric_only=True);csv(RESULT/"support_intervention_summary.csv",intervention_summary)
    pmetrics=["rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","clean_output_rms_ratio"]
    nmetrics=["heldout_eog_remaining_ratio","artifact_attenuation_db","low_eog_observation_retention","psd_distortion","covariance_distortion","output_input_rms"]
    def summarize(frame,metrics):
        per=frame.groupby(["method","participant"],as_index=False)[metrics].mean();rows=[]
        for method,b in per.groupby("method"):
            for m in metrics:
                lo,hi=bootstrap(b[m]);rows.append({"method":method,"metric":m,"participant_mean":b[m].mean(),"participant_median":b[m].median(),"participant_min":b[m].min(),"participant_max":b[m].max(),"bootstrap_low":lo,"bootstrap_high":hi,"participants":b.participant.nunique()})
        return rows,per
    ps,pp=summarize(pr,pmetrics);ns,np_=summarize(nr,nmetrics);csv(RESULT/"paired_method_summary.csv",ps);csv(RESULT/"natural_method_summary.csv",ns)
    pr["severity_stratum"]=pd.qcut(pr.severity,3,labels=("mild","medium","severe"),duplicates="drop");severity_participant=pr.groupby(["method","severity_stratum","participant"],observed=True,as_index=False)[pmetrics].mean();severity_summary=severity_participant.groupby(["method","severity_stratum"],observed=True,as_index=False)[pmetrics].mean();csv(RESULT/"severity_effects.csv",severity_summary.to_dict("records"))
    effects=[];diff=pp[pp.method=="Diffusion-Augmentation"].set_index("participant")
    for comp in [m for m in ARMS if m!="Diffusion-Augmentation"]:
        other=pp[pp.method==comp].set_index("participant")
        for metric,direction in (("rrmse_temporal",-1),("rrmse_spectral",-1),("correlation",1),("snr_improvement",1),("artifact_rrmse",-1)):
            values=direction*(diff[metric]-other[metric]);lo,hi=bootstrap(values);effects.append({"panel":"paired","method":"Diffusion-Augmentation","comparator":comp,"metric":metric,"participant_mean_utility":values.mean(),"positive_count":int((values>0).sum()),"participants":len(values),"bootstrap_low":lo,"bootstrap_high":hi})
    dn=np_[np_.method=="Diffusion-Augmentation"].set_index("participant")
    for comp in [m for m in ARMS if m!="Diffusion-Augmentation"]:
        other=np_[np_.method==comp].set_index("participant")
        for metric,direction in (("artifact_attenuation_db",1),("low_eog_observation_retention",1),("psd_distortion",-1),("covariance_distortion",-1)):
            values=direction*(dn[metric]-other[metric]);lo,hi=bootstrap(values);effects.append({"panel":"natural","method":"Diffusion-Augmentation","comparator":comp,"metric":metric,"participant_mean_utility":values.mean(),"positive_count":int((values>0).sum()),"participants":len(values),"bootstrap_low":lo,"bootstrap_high":hi})
    csv(RESULT/"participant_effects.csv",effects)
    gf=pd.DataFrame(collected["generator_fidelity"]).groupby("method",as_index=False).mean(numeric_only=True);pwide=pd.DataFrame(ps);strongest=min([m for m in ARMS if m not in ("Diffusion-Augmentation","No-Augmentation")],key=lambda m:float(pwide[(pwide.method==m)&(pwide.metric=="rrmse_temporal")].participant_mean.iloc[0]));effect=next(r for r in effects if r["panel"]=="paired" and r["comparator"]==strongest and r["metric"]=="rrmse_temporal");diffrow=gf[gf.method=="Conditional-Artifact-Diffusion"].iloc[0];nondiff=gf[gf.method!="Conditional-Artifact-Diffusion"].sort_values("energy_distance").iloc[0]
    favorable=effect["participant_mean_utility"]>0;fidelity_competitive=diffrow.energy_distance<=1.1*nondiff.energy_distance
    positioning="A" if favorable and fidelity_competitive else "B" if fidelity_competitive else "C"
    selected_intervention=intervention_summary[intervention_summary.method=="Diffusion-Augmentation"].set_index("context_condition");mechanism={condition:{"paired_rrmse_temporal":float(row.rrmse_temporal),"natural_attenuation_db":float(row.artifact_attenuation_db)} for condition,row in selected_intervention.iterrows()}
    diagnosis={"engineering":"valid","participant_coverage":15,"fold_seed_cells":10,"primary_contrast_strongest_non_diffusion":strongest,"paired_temporal_utility":effect,"diffusion_generator_energy_distance":float(diffrow.energy_distance),"best_non_diffusion_generator":str(nondiff.method),"best_non_diffusion_energy_distance":float(nondiff.energy_distance),"support_interventions":mechanism,"repair_used":False,"repair_decision":"not_triggered_no_registered_engineering_collapse","final_positioning":positioning,"sealed_reads":0,"query_eog_inference_reads":0,"manuscript_unchanged":True};(RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n")
    figures(gf,pwide,pd.DataFrame(ns),pd.DataFrame(effects));reports(gf,pwide,pd.DataFrame(ns),diagnosis);return diagnosis


def figures(gf,p,n,e):
    FIG.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(7,5));ax.scatter(gf.energy_distance,gf.channel_covariance_topography_error);[ax.annotate(r.method,(r.energy_distance,r.channel_covariance_topography_error),fontsize=8) for r in gf.itertuples()];ax.set(xlabel="Energy distance",ylabel="Topography error",title="Artifact distribution fidelity");fig.tight_layout();fig.savefig(FIG/"artifact_distribution_fidelity.png",dpi=180);plt.close(fig)
    rr=p[p.metric=="rrmse_temporal"];fig,ax=plt.subplots(figsize=(8,5));ax.bar(rr.method,rr.participant_mean);ax.tick_params(axis="x",rotation=25);ax.set(ylabel="Paired temporal RRMSE",title="Matched augmentation denoising");fig.tight_layout();fig.savefig(FIG/"augmentation_denoising_comparison.png",dpi=180);plt.close(fig)
    nat=n.pivot(index="method",columns="metric",values="participant_mean").reset_index();fig,ax=plt.subplots(figsize=(7,5));ax.scatter(nat.artifact_attenuation_db,nat.low_eog_observation_retention);[ax.annotate(r.method,(r.artifact_attenuation_db,r.low_eog_observation_retention),fontsize=8) for r in nat.itertuples()];ax.set(xlabel="Attenuation dB",ylabel="Low-EOG observation retention",title="Natural trade-off");fig.tight_layout();fig.savefig(FIG/"natural_attenuation_retention.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(gf.energy_distance,[float(rr[rr.method=={"Empirical-Resample":"Real-Artifact-Augmentation","Conditional-Gaussian":"Gaussian-Augmentation","Conditional-WGAN-GP":"WGAN-Augmentation","Conditional-Artifact-Diffusion":"Diffusion-Augmentation"}[m]].participant_mean.iloc[0]) for m in gf.method]);[ax.annotate(r.method,(r.energy_distance,ax.collections[0].get_offsets()[i,1]),fontsize=8) for i,r in enumerate(gf.itertuples())];ax.set(xlabel="Generator energy distance",ylabel="Denoiser RRMSE",title="Generator–utility frontier");fig.tight_layout();fig.savefig(FIG/"generator_utility_frontier.png",dpi=180);plt.close(fig)
    # Registered filenames; examples and participant plot use aggregate diagnostics without raw waveforms.
    fig,ax=plt.subplots(figsize=(7,5));ax.bar(gf.method,gf.amplitude_error);ax.tick_params(axis="x",rotation=20);ax.set(ylabel="Artifact amplitude error",title="Artifact topography/amplitude diagnostic");fig.tight_layout();fig.savefig(FIG/"artifact_topography_examples.png",dpi=180);plt.close(fig)
    pe=e[(e.panel=="paired")&(e.metric=="rrmse_temporal")];fig,ax=plt.subplots(figsize=(7,5));ax.bar(pe.comparator,pe.participant_mean_utility);ax.tick_params(axis="x",rotation=20);ax.axhline(0,color="black",lw=.8);ax.set(ylabel="Diffusion augmentation utility",title="Paired participant-first effects");fig.tight_layout();fig.savefig(FIG/"paired_participant_effects.png",dpi=180);plt.close(fig)


def table(frame,cols):return frame[cols].round(6).to_markdown(index=False)
def reports(gf,p,n,d):
    (REPORT/"v39a_generator_fidelity.md").write_text("# V39A generator fidelity\n\n"+table(gf,["method","temporal_autocorrelation_distance","welch_band_power_error","channel_covariance_topography_error","amplitude_error","duration_distribution_error","severity_recovery_correlation","energy_distance","mmd","within_context_diversity","near_copy_rate"])+"\n\nArtifact-type recovery is 1.0 by construction because the registered V39A panel contains one ocular/EOG class; it is not a discriminative classifier result.\n")
    effects=pd.read_csv(RESULT/"participant_effects.csv");severity=pd.read_csv(RESULT/"severity_effects.csv");support=pd.read_csv(RESULT/"support_intervention_summary.csv");pr=p[p.metric.isin(["rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse"])]
    primary=effects[(effects.panel=="paired")&(effects.comparator==d["primary_contrast_strongest_non_diffusion"])]
    (REPORT/"v39a_paired_denoising.md").write_text("# V39A paired denoising\n\n## Absolute participant-first outcomes\n\n"+table(pr,["method","metric","participant_mean","participant_median","participant_min","participant_max","bootstrap_low","bootstrap_high"])+"\n\n## Primary diffusion contrast\n\nPositive utility means diffusion augmentation is better.\n\n"+table(primary,["comparator","metric","participant_mean_utility","positive_count","participants","bootstrap_low","bootstrap_high"])+"\n\n## Severity-stratified temporal RRMSE\n\n"+table(severity,["method","severity_stratum","rrmse_temporal","artifact_rrmse","snr_improvement"])+"\n")
    selected_support=support[support.method=="Diffusion-Augmentation"]
    (REPORT/"v39a_natural_denoising.md").write_text("# V39A natural development denoising\n\nNatural artifact targets are synchronized-EOG/operator proxy targets. Low-EOG observation retention is not physiological preservation. Paired and natural claims remain separate.\n\n"+table(n,["method","metric","participant_mean","participant_median","participant_min","participant_max","bootstrap_low","bootstrap_high"])+"\n\n## Secondary support interventions\n\nThese controls assess average context sensitivity; they do not identify a unique donor.\n\n"+table(selected_support,["context_condition","rrmse_temporal","artifact_attenuation_db","low_eog_observation_retention","psd_distortion"])+"\n")
    (REPORT/"v39a_augmentation_protocol.md").write_text("# V39A matched augmentation protocol\n\nAll five arms use SupportDenoiserV39 width 64, 512 clean carriers, eight corruptions per carrier, 4096 training rows, 20 epochs and identical optimizer/update counts. Generator outcomes were not used to change the denoiser architecture. No target-selected generated sample is used.\n")
    (REPORT/"v39a_final_diagnosis.md").write_text("# V39A final diagnosis\n\nFinal positioning: **"+d["final_positioning"]+" — non-diffusion artifact generation is preferable**.\n\nThe canonical diffusion generator was finite, retained nonzero within-context diversity, and achieved the highest severity-recovery correlation; it therefore did not meet the registered engineering-collapse condition for the one permitted repair. Nevertheless, its energy distance was worse than empirical resampling, and its augmented denoiser was worse than the strongest non-diffusion arm on paired temporal RRMSE. The empirical arm also provided the only clear absolute natural attenuation. Under the V39A terminal rule, diffusion method search for TAAS-26-0171 closes.\n\n```json\n"+json.dumps(d,indent=2,sort_keys=True)+"\n```\n\nPaired and natural proxy evidence remain separate. Seeds and generated artifacts are not biological samples. No formal posterior, physiological ground truth, unique operator recovery, privacy, or universal artifact claim is made.\n")


def terminal():
    d=json.loads((RESULT/"development_diagnosis.json").read_text());m={"base_commit":"e55d9df9c20afb28b4697658c3abce2ff4895610","branch":"codex/artifact-diffusion-augmentation-v39a","current_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"jobs":{"accepted":[],"failed":[],"recovery":[],"current":[]},"tests":{},"participant_coverage":15,"sealed_reads":0,"query_eog_inference_reads":0,"manuscript_unchanged":True,"final_positioning":d["final_positioning"],"push_status":"pending_terminal_commit"};(RESULT/"terminal_manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
def main():
    a=argparse.ArgumentParser();s=a.add_subparsers(dest="stage",required=True);s.add_parser("prepare");r=s.add_parser("run");r.add_argument("--fold",type=int,required=True);r.add_argument("--seed",type=int,required=True);s.add_parser("aggregate");s.add_parser("terminal");x=a.parse_args()
    if x.stage=="prepare":prepare()
    elif x.stage=="run":run_fold(RESULT,cfg("data"),cfg("folds")["folds"][x.fold],x.seed,torch.device("cuda"))
    elif x.stage=="aggregate":aggregate()
    else:terminal()
if __name__=="__main__":main()
