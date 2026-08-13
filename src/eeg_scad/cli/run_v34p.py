"""CLI for V34P exact task-head fiber consolidation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.bci2a import outer_folds
from eeg_scad.privacy.experiment import sha256
from eeg_scad.privacy.fiber_experiment import run_fiber_fold


ROOT=Path(__file__).resolve().parents[3]
RESULT=ROOT/"results"/"fiber_sandiff_v34p"
V33=Path("/home/infres/yinwang/denoiseNet_sandiff_consolidation_v33p")
SEEDS=(20260920,20260921)


def _csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    fields=[]
    for row in rows:
        for field in row:
            if field not in fields:fields.append(field)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def _read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def prepare():
    if not V33.exists():raise FileNotFoundError(V33)
    inventory=_read_csv(V33/"results"/"sandiff_v33p"/"checkpoint_binding.csv")
    eegnet=[row for row in inventory if row["model"]=="EEGNet"]
    if len(eegnet)!=6:raise RuntimeError(f"expected 6 V33P EEGNet checkpoints, found {len(eegnet)}")
    for row in eegnet:
        path=Path(row["path"])
        if not path.is_file() or sha256(path)!=row["sha256"]:raise RuntimeError(f"checkpoint mismatch: {path}")
    rows=[]
    for split in outer_folds():
        for subject in range(1,10):
            role="outer_test" if subject in split["test_subjects"] else ("stage_A_train__stage_B_refit" if subject in split["train_subjects"] else "stage_A_validation__stage_B_refit")
            rows.append({"fold":split["fold"],"subject":subject,"role":role})
    _csv(RESULT/"split_manifest.csv",rows)


def _group(rows,fields):
    grouped={}
    for row in rows:grouped.setdefault(tuple(row[field] for field in fields),[]).append(row)
    return grouped


def _float(row,key):return float(row[key])


def aggregate():
    payloads=[json.loads((RESULT/"runtime"/f"fold_{fold}_seed_{seed}"/"fold_result.json").read_text()) for fold in range(3) for seed in SEEDS]
    metrics=[row for payload in payloads for row in payload["metrics"]];participants=[row for payload in payloads for row in payload["participant_effects"]]
    preservation=[row for payload in payloads for row in payload["exact_preservation"]];fidelity=[row for payload in payloads for row in payload["distribution_fidelity"]]
    latency=[row for payload in payloads for row in payload["latency"]];bindings=[row for payload in payloads for row in payload["checkpoint_binding"]]
    geometry=[payload["fiber_geometry"] for payload in payloads];selection=[row for payload in payloads for row in payload["selection_summary"]]
    # Frozen V33P comparisons use the same full-pool EEGNet, test groups, and seeds.
    for row in _read_csv(V33/"results"/"sandiff_v33p"/"privacy_attacks.csv"):
        if row["strength"]=="strong" and row["method"] in ("one_step","SANDiff"):
            copied=dict(row);copied["method"]="V33P_"+row["method"]+"_strong";metrics.append(copied)
    for row in _read_csv(V33/"results"/"sandiff_v33p"/"participant_effects.csv"):
        if row["strength"]=="strong" and row["method"] in ("one_step","SANDiff"):
            copied=dict(row);copied["method"]="V33P_"+row["method"]+"_strong";participants.append(copied)
    _csv(RESULT/"checkpoint_binding.csv",bindings);_csv(RESULT/"fiber_geometry.csv",geometry);_csv(RESULT/"exact_preservation.csv",preservation);_csv(RESULT/"privacy_attacks.csv",metrics);_csv(RESULT/"task_utility.csv",metrics);_csv(RESULT/"distribution_fidelity.csv",fidelity);_csv(RESULT/"participant_effects.csv",participants);_csv(RESULT/"latency_summary.csv",latency);_csv(RESULT/"selection_summary.csv",selection)
    metric_keys=("fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","calibration_error","worst_participant_accuracy","between_participant_variance","linear_subject_probe_balanced_accuracy","adaptive_subject_attack_balanced_accuracy","cross_session_verification_balanced_accuracy","cross_session_same_different_auroc")
    summary=[]
    for (method,strength),rows in _group(metrics,("method","strength")).items():
        item={"method":method,"strength":strength,"folds":len({row["fold"] for row in rows}),"seeds":len({row["seed"] for row in rows})}
        for key in metric_keys:item[key]=float(np.mean([_float(row,key) for row in rows]))
        summary.append(item)
    _csv(RESULT/"method_summary.csv",summary)
    fidelity_summary=[]
    for (method,strength),rows in _group(fidelity,("method","strength")).items():
        item={"method":method,"strength":strength}
        for key in ("conditional_covariance_relative_frobenius","conditional_energy_distance","conditional_mmd_rbf","fiber_variance_retained"):item[key]=float(np.mean([_float(row,key) for row in rows]))
        fidelity_summary.append(item)
    _csv(RESULT/"distribution_fidelity_summary.csv",fidelity_summary)
    averaged={}
    for key,rows in _group(participants,("method","strength","participant")).items():
        averaged[key]={field:float(np.mean([_float(row,field) for row in rows])) for field in ("fixed_head_balanced_accuracy","adaptive_subject_attack_recall","cross_session_same_different_auroc")}
    participant_ids=range(1,10)
    def vector(method,strength,field):return np.asarray([averaged[(method,strength,p)][field] for p in participant_ids])
    specs=[
        ("Fiber_SANDiff_vs_RAW_fixed",vector("Fiber-SANDiff","strong","fixed_head_balanced_accuracy")-vector("RAW","na","fixed_head_balanced_accuracy")),
        ("Fiber_SANDiff_privacy_vs_RAW",vector("RAW","na","adaptive_subject_attack_recall")-vector("Fiber-SANDiff","strong","adaptive_subject_attack_recall")),
        ("Fiber_SANDiff_verification_vs_RAW",vector("RAW","na","cross_session_same_different_auroc")-vector("Fiber-SANDiff","strong","cross_session_same_different_auroc")),
        ("Fiber_SANDiff_privacy_gap_to_HEAD_ONLY",vector("Fiber-SANDiff","strong","adaptive_subject_attack_recall")-vector("HEAD_ONLY","na","adaptive_subject_attack_recall")),
        ("Fiber_SANDiff_minus_OneStep_fixed",vector("Fiber-SANDiff","strong","fixed_head_balanced_accuracy")-vector("Fiber-OneStep","strong","fixed_head_balanced_accuracy")),
        ("Fiber_SANDiff_privacy_vs_OneStep",vector("Fiber-OneStep","strong","adaptive_subject_attack_recall")-vector("Fiber-SANDiff","strong","adaptive_subject_attack_recall")),
        ("Fiber_SANDiff_privacy_vs_V33P",vector("V33P_SANDiff_strong","strong","adaptive_subject_attack_recall")-vector("Fiber-SANDiff","strong","adaptive_subject_attack_recall")),
    ]
    rng=np.random.default_rng(34);contrasts=[]
    for name,values in specs:
        boot=np.mean(rng.choice(values,(50000,len(values)),replace=True),axis=1)
        contrasts.append({"contrast":name,"participant_mean":float(values.mean()),"participant_median":float(np.median(values)),"ci95_low":float(np.quantile(boot,.025)),"ci95_high":float(np.quantile(boot,.975)),"positive_count":int((values>0).sum()),"participants":len(values)})
    _csv(RESULT/"contrast_summary.csv",contrasts)
    lookup={(row["method"],row["strength"]):row for row in summary};fiber=lookup[("Fiber-SANDiff","strong")];one=lookup[("Fiber-OneStep","strong")];raw=lookup[("RAW","na")];head=lookup[("HEAD_ONLY","na")];v33=lookup[("V33P_SANDiff_strong","strong")]
    fl={(row["method"],row["strength"]):row for row in fidelity_summary};fiber_f=fl[("Fiber-SANDiff","strong")];one_f=fl[("Fiber-OneStep","strong")]
    exact_valid=all(int(float(row["prediction_mismatch_count"]))==0 and abs(float(row["fixed_head_ba_difference"]))<1e-12 for row in preservation)
    sand_better_fidelity=sum(fiber_f[key]<one_f[key] for key in ("conditional_covariance_relative_frobenius","conditional_energy_distance","conditional_mmd_rbf"))
    head_dominates=head["adaptive_subject_attack_balanced_accuracy"]>0.40 and abs(fiber["adaptive_subject_attack_balanced_accuracy"]-head["adaptive_subject_attack_balanced_accuracy"])<0.02
    if head_dominates:position="D. head-transmitted leakage dominates"
    elif sand_better_fidelity>=2 and fiber["retrained_head_balanced_accuracy"]>=one["retrained_head_balanced_accuracy"]-0.005:position="A. Fiber-SANDiff positive method"
    elif one["adaptive_subject_attack_balanced_accuracy"]+0.02<fiber["adaptive_subject_attack_balanced_accuracy"] and one["retrained_head_balanced_accuracy"]>=fiber["retrained_head_balanced_accuracy"]:position="B. Fiber-OneStep preferable"
    else:position="C. Fiber methods equivalent"
    diagnosis={"status":"development_complete","exact_preservation":exact_valid,"final_method_positioning":position,"strong_fiber_sandiff_vs_raw":{"fixed_head_delta":fiber["fixed_head_balanced_accuracy"]-raw["fixed_head_balanced_accuracy"],"adaptive_privacy_utility":raw["adaptive_subject_attack_balanced_accuracy"]-fiber["adaptive_subject_attack_balanced_accuracy"],"verification_auroc_reduction":raw["cross_session_same_different_auroc"]-fiber["cross_session_same_different_auroc"]},"head_only_floor":{"adaptive_subject_attack_balanced_accuracy":head["adaptive_subject_attack_balanced_accuracy"],"verification_auroc":head["cross_session_same_different_auroc"],"fiber_gap_adaptive":fiber["adaptive_subject_attack_balanced_accuracy"]-head["adaptive_subject_attack_balanced_accuracy"]},"fiber_sandiff_vs_one_step":{"retrained_head_delta":fiber["retrained_head_balanced_accuracy"]-one["retrained_head_balanced_accuracy"],"adaptive_privacy_utility":one["adaptive_subject_attack_balanced_accuracy"]-fiber["adaptive_subject_attack_balanced_accuracy"],"verification_auroc_reduction":one["cross_session_same_different_auroc"]-fiber["cross_session_same_different_auroc"],"distribution_metrics_better_count":sand_better_fidelity},"fiber_sandiff_vs_v33p":{"fixed_head_delta":fiber["fixed_head_balanced_accuracy"]-v33["fixed_head_balanced_accuracy"],"adaptive_privacy_utility":v33["adaptive_subject_attack_balanced_accuracy"]-fiber["adaptive_subject_attack_balanced_accuracy"]},"K":1,"reverse_steps":10,"waveform_sealed_reads":0,"outer_test_used_for_selection":False}
    (RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    make_figures(summary,fidelity_summary,contrasts,latency)


def make_figures(summary,fidelity,contrasts,latency):
    import matplotlib.pyplot as plt
    out=ROOT/"figures"/"fiber_sandiff_v34p";out.mkdir(parents=True,exist_ok=True)
    selected=[row for row in summary if (row["strength"] in ("na","strong"))]
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter([r["adaptive_subject_attack_balanced_accuracy"] for r in selected],[r["fixed_head_balanced_accuracy"] for r in selected]);
    for row in selected:ax.annotate(row["method"],(row["adaptive_subject_attack_balanced_accuracy"],row["fixed_head_balanced_accuracy"]),fontsize=7)
    ax.set(xlabel="Adaptive subject BA (lower is private)",ylabel="Fixed-head balanced accuracy",title="V34P privacy–utility frontier");fig.tight_layout();fig.savefig(out/"privacy_utility_frontier.png",dpi=180);plt.close(fig)
    lookup={(r["method"],r["strength"]):r for r in summary};methods=["RAW","HEAD_ONLY","Fiber-OneStep","Fiber-SANDiff"];strength={"RAW":"na","HEAD_ONLY":"na","Fiber-OneStep":"strong","Fiber-SANDiff":"strong"}
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(methods,[lookup[(m,strength[m])]["adaptive_subject_attack_balanced_accuracy"] for m in methods]);ax.axhline(1/3,color="k",ls="--",lw=1);ax.set(ylabel="Adaptive subject BA",title="Head-transmitted leakage floor");fig.tight_layout();fig.savefig(out/"head_floor_comparison.png",dpi=180);plt.close(fig)
    strong=[r for r in fidelity if r["strength"]=="strong"];fig,axes=plt.subplots(1,3,figsize=(10,4));keys=("conditional_covariance_relative_frobenius","conditional_energy_distance","fiber_variance_retained")
    for ax,key in zip(axes,keys):ax.bar([r["method"] for r in strong],[r[key] for r in strong]);ax.set_title(key.replace("_"," "),fontsize=8);ax.tick_params(axis="x",rotation=20)
    fig.tight_layout();fig.savefig(out/"fiber_distribution_fidelity.png",dpi=180);plt.close(fig)
    privacy=next(r for r in contrasts if r["contrast"]=="Fiber_SANDiff_privacy_vs_RAW");fig,ax=plt.subplots(figsize=(6,4));ax.bar(["mean"],[privacy["participant_mean"]],yerr=[[privacy["participant_mean"]-privacy["ci95_low"]],[privacy["ci95_high"]-privacy["participant_mean"]]]);ax.axhline(0,color="k",lw=1);ax.set(ylabel="RAW − Fiber-SANDiff adaptive recall",title="Participant-first privacy effect");fig.tight_layout();fig.savefig(out/"participant_privacy_effects.png",dpi=180);plt.close(fig)
    grouped=[]
    for (method,batch),rows in _group(latency,("method","batch_size")).items():grouped.append((method,int(batch),float(np.mean([float(r["median_ms"]) for r in rows]))))
    fig,ax=plt.subplots(figsize=(7,4));labels=[f"{m}\nB{b}" for m,b,_ in grouped];ax.bar(labels,[v for _,_,v in grouped]);ax.set(ylabel="Median latency (ms)",title="Fiber sanitizer latency");fig.tight_layout();fig.savefig(out/"latency_comparison.png",dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","run","aggregate"));parser.add_argument("--fold",type=int);parser.add_argument("--seed",type=int);args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="run":
        if args.fold not in range(3) or args.seed not in SEEDS:parser.error("registered --fold and --seed required")
        prepare();run_fiber_fold(V33,RESULT,args.fold,args.seed,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:aggregate()
    return 0


if __name__=="__main__":raise SystemExit(main())
