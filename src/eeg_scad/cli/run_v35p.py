"""CLI for frozen V35P exact-fiber channel validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.fiber_channel import run_validation_fold


ROOT=Path(__file__).resolve().parents[3];RESULT=ROOT/"results"/"fiber_channel_v35p"
V34=Path("/home/infres/yinwang/denoiseNet_fiber_sandiff_v34p");V33=Path("/home/infres/yinwang/denoiseNet_sandiff_consolidation_v33p");SEEDS=(20260920,20260921)


def _csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    fields=[]
    for row in rows:
        for field in row:
            if field not in fields:fields.append(field)
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def _group(rows,fields):
    grouped={}
    for row in rows:grouped.setdefault(tuple(row[field] for field in fields),[]).append(row)
    return grouped


def prepare():
    manifest=V34/"results"/"fiber_sandiff_v34p"/"terminal_manifest.json"
    if not manifest.is_file():raise FileNotFoundError(manifest)
    binding={"v34p_terminal_commit":"e10dd40100e60f5e47c4d1a917ec4515880fc9ca","v34p_terminal_manifest":str(manifest),"v34p_terminal_manifest_payload":json.loads(manifest.read_text()),"v34p_results_read_only":True,"latency_benchmark_authorized":False}
    RESULT.mkdir(parents=True,exist_ok=True);(RESULT/"v34p_binding.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def aggregate():
    payloads=[json.loads((RESULT/"runtime"/f"fold_{fold}_seed_{seed}"/"fold_result.json").read_text()) for fold in range(3) for seed in SEEDS]
    names=("checkpoint_binding","resample_coverage","exact_preservation","head_aware_attacks","conditional_fiber_leakage","distribution_fidelity","multisample_diversity","participant_effects")
    combined={name:[row for payload in payloads for row in payload[name]] for name in names}
    for name,rows in combined.items():_csv(RESULT/f"{name}.csv",rows)
    metrics=[row for payload in payloads for row in payload["metrics"]];_csv(RESULT/"task_utility.csv",metrics)
    utility=[]
    keys=("fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","calibration_error","worst_participant_accuracy","between_participant_variance")
    for (method,strength),rows in _group(metrics,("method","strength")).items():
        item={"method":method,"strength":strength}
        for key in keys:item[key]=float(np.mean([float(row[key]) for row in rows]))
        utility.append(item)
    _csv(RESULT/"task_utility_summary.csv",utility)
    attacks=[]
    for (method,attacker,feature),rows in _group(combined["head_aware_attacks"],("method","attacker","feature")).items():
        attacks.append({"method":method,"attacker":attacker,"feature":feature,"balanced_accuracy":float(np.mean([float(row["balanced_accuracy"]) for row in rows])),"cross_entropy":float(np.mean([float(row["cross_entropy"]) for row in rows])),"same_different_verification_auroc":float(np.mean([float(row["same_different_verification_auroc"]) for row in rows]))})
    _csv(RESULT/"head_aware_attack_summary.csv",attacks)
    primary=[]
    for (fold,seed,method,attacker),rows in _group(combined["head_aware_attacks"],("fold","seed","method","attacker")).items():
        eligible=[row for row in rows if row["feature"] in ("A_H","A_Z","A_HZ")];best=max(eligible,key=lambda row:float(row["balanced_accuracy"]))
        primary.append({"fold":fold,"seed":seed,"method":method,"attacker":attacker,"primary_finite_threat_balanced_accuracy":best["balanced_accuracy"],"maximizing_feature":best["feature"],"privacy_semantics":"cannot claim below A_H boundary"})
    _csv(RESULT/"primary_finite_threat.csv",primary)
    fidelity=[]
    for method,rows in _group(combined["distribution_fidelity"],("method",)).items():
        item={"method":method[0]}
        for key in ("conditional_covariance_relative_frobenius","conditional_energy_distance","conditional_mmd_rbf","fiber_variance_retained"):item[key]=float(np.mean([float(row[key]) for row in rows]))
        fidelity.append(item)
    _csv(RESULT/"distribution_fidelity_summary.csv",fidelity)
    diversity=[]
    for method,rows in _group(combined["multisample_diversity"],("method",)).items():
        item={"method":method[0]}
        for key in ("within_H_sample_variance","between_H_variance","conditional_covariance_trace","nearest_training_fiber_distance","duplicate_rate","sample_diversity"):item[key]=float(np.mean([float(row[key]) for row in rows]))
        diversity.append(item)
    _csv(RESULT/"multisample_diversity_summary.csv",diversity)
    conditional=[]
    for (method,attacker),rows in _group(combined["conditional_fiber_leakage"],("method","attacker")).items():
        conditional.append({"method":method,"attacker":attacker,"CE_A_H":float(np.mean([float(row["CE_A_H"]) for row in rows])),"CE_A_HU":float(np.mean([float(row["CE_A_HU"]) for row in rows])),"conditional_fiber_leakage":float(np.mean([float(row["conditional_fiber_leakage"]) for row in rows])),"interpretation":"finite cross-session closure diagnostic, not CMI"})
    _csv(RESULT/"conditional_fiber_leakage_summary.csv",conditional)
    fl={row["method"]:row for row in fidelity};ut={row["method"]:row for row in utility};sand=fl["Fiber-SANDiff"];resample=fl["Fiber-Stratified-Resample"]
    comparison={"covariance":sand["conditional_covariance_relative_frobenius"]<resample["conditional_covariance_relative_frobenius"],"energy":sand["conditional_energy_distance"]<resample["conditional_energy_distance"],"mmd":sand["conditional_mmd_rbf"]<resample["conditional_mmd_rbf"],"variance_closeness":abs(1-sand["fiber_variance_retained"])<abs(1-resample["fiber_variance_retained"])}
    better=sum(comparison.values());retrained_delta=ut["Fiber-SANDiff"]["retrained_head_balanced_accuracy"]-ut["Fiber-Stratified-Resample"]["retrained_head_balanced_accuracy"]
    if better>=3 and retrained_delta>=-0.005:position="A. Fiber-SANDiff retains a diffusion-specific advantage"
    elif better<=1 and retrained_delta<=0:position="C. Fiber-Stratified-Resample is clearly preferable"
    else:position="B. Fiber-SANDiff and population resampling are equivalent"
    exact_valid=all(int(float(row["prediction_mismatch_count"]))==0 and abs(float(row["fixed_head_ba_difference"]))<1e-12 for row in combined["exact_preservation"])
    diagnosis={"status":"development_complete","final_positioning":position,"exact_preservation":exact_valid,"strong_endpoint_privacy_semantics":"I(Z_prime;S|Y)=I(H;S|Y)","finite_attacker_not_mutual_information":True,"fiber_sandiff_vs_resample_distribution":comparison,"fiber_sandiff_vs_resample_retrained_head_delta":retrained_delta,"latency_used":False,"latency_benchmark_run":False,"waveform_sealed_reads":0}
    (RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n",encoding="utf-8");make_figures(attacks,conditional,fidelity,diversity,primary)


def make_figures(attacks,conditional,fidelity,diversity,primary):
    import matplotlib.pyplot as plt
    out=ROOT/"figures"/"fiber_channel_v35p";out.mkdir(parents=True,exist_ok=True)
    rows=[row for row in attacks if row["attacker"]=="adaptive_mlp" and row["feature"] in ("A_H","A_Z","A_HZ")];methods=[]
    for row in rows:
        if row["method"] not in methods:methods.append(row["method"])
    fig,ax=plt.subplots(figsize=(10,5));width=.25;x=np.arange(len(methods))
    for index,feature in enumerate(("A_H","A_Z","A_HZ")):ax.bar(x+(index-1)*width,[next(r["balanced_accuracy"] for r in rows if r["method"]==m and r["feature"]==feature) for m in methods],width,label=feature)
    ax.set_xticks(x,methods,rotation=25,ha="right");ax.set_ylabel("Adaptive subject BA");ax.legend();fig.tight_layout();fig.savefig(out/"head_aware_privacy.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));labels=[f"{r['method']}\n{r['attacker']}" for r in conditional];ax.bar(labels,[r["conditional_fiber_leakage"] for r in conditional]);ax.axhline(0,color="k",lw=1);ax.tick_params(axis="x",rotation=30);ax.set_ylabel("CE(A_H) − CE(A_HU)");fig.tight_layout();fig.savefig(out/"conditional_fiber_leakage.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,4));keys=("conditional_covariance_relative_frobenius","conditional_energy_distance","conditional_mmd_rbf","fiber_variance_retained")
    for ax,key in zip(axes,keys):ax.bar([r["method"] for r in fidelity],[r[key] for r in fidelity]);ax.tick_params(axis="x",rotation=25);ax.set_title(key.replace("_"," "),fontsize=8)
    fig.tight_layout();fig.savefig(out/"stochastic_distribution_fidelity.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,4));keys=("within_H_sample_variance","duplicate_rate","sample_diversity")
    for ax,key in zip(axes,keys):ax.bar([r["method"] for r in diversity],[r[key] for r in diversity]);ax.tick_params(axis="x",rotation=25);ax.set_title(key.replace("_"," "),fontsize=8)
    fig.tight_layout();fig.savefig(out/"multisample_diversity.png",dpi=180);plt.close(fig)
    adaptive=[row for row in primary if row["attacker"]=="adaptive_mlp"];privacy={}
    for method,rows in _group(adaptive,("method",)).items():privacy[method[0]]=float(np.mean([float(r["primary_finite_threat_balanced_accuracy"]) for r in rows]))
    energy={row["method"]:row["conditional_energy_distance"] for row in fidelity};common=[m for m in energy if m in privacy]
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter([privacy[m] for m in common],[energy[m] for m in common]);
    for m in common:ax.annotate(m,(privacy[m],energy[m]),fontsize=8)
    ax.set(xlabel="Primary finite-threat adaptive BA",ylabel="Conditional energy distance",title="Privacy–distribution positioning");fig.tight_layout();fig.savefig(out/"privacy_distribution_frontier.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","run","aggregate"));parser.add_argument("--fold",type=int);parser.add_argument("--seed",type=int);args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="run":
        if args.fold not in range(3) or args.seed not in SEEDS:parser.error("registered fold/seed required")
        prepare();run_validation_fold(V34,V33,RESULT,args.fold,args.seed,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:aggregate()
    return 0


if __name__=="__main__":raise SystemExit(main())
