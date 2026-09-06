"""CLI for V33P full-pool SANDiff consolidation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.bci2a import outer_folds
from eeg_scad.privacy.consolidation import run_consolidation_fold
from eeg_scad.privacy.experiment import prepare_cache


ROOT=Path(__file__).resolve().parents[3];RESULT=ROOT/"results"/"sandiff_v33p";CACHE=RESULT/"runtime"/"bci2a_trials.npz";SEEDS=(20260920,20260921)


def _csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text("",encoding="utf-8");return
    fields=[]
    for row in rows:
        for field in row:
            if field not in fields:fields.append(field)
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def prepare():
    prepare_cache(Path("/projects/EEG-foundation-model/BCI-IV"),CACHE);rows=[]
    for split in outer_folds():
        full=sorted(split["train_subjects"]+split["validation_subjects"])
        for subject in range(1,10):
            if subject in split["train_subjects"]:role="stage_A_train__stage_B_refit"
            elif subject in split["validation_subjects"]:role="stage_A_validation__stage_B_refit"
            else:role="outer_test_only"
            rows.append({"fold":split["fold"],"subject":subject,"role":role,"stage_A_train":"yes" if subject in split["train_subjects"] else "no","stage_A_validation":"yes" if subject in split["validation_subjects"] else "no","stage_B_full_pool":"yes" if subject in full else "no","outer_test":"yes" if subject in split["test_subjects"] else "no"})
    _csv(RESULT/"split_manifest.csv",rows)


def aggregate():
    payloads=[json.loads((RESULT/"runtime"/f"fold_{fold}_seed_{seed}"/"fold_result.json").read_text()) for fold in range(3) for seed in SEEDS]
    metrics=[row for p in payloads for row in p["metrics"]];participants=[row for p in payloads for row in p["participant_effects"]];latency=[row for p in payloads for row in p["latency"]];bindings=[row for p in payloads for row in p["checkpoint_binding"]];selection=[row for p in payloads for row in p["selection_summary"]]
    _csv(RESULT/"selection_summary.csv",selection);_csv(RESULT/"checkpoint_binding.csv",bindings);_csv(RESULT/"privacy_attacks.csv",metrics);_csv(RESULT/"task_utility.csv",metrics);_csv(RESULT/"privacy_utility_curve.csv",metrics);_csv(RESULT/"participant_effects.csv",participants);_csv(RESULT/"latency_summary.csv",latency)
    keys=["fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","calibration_error","worst_participant_accuracy","between_participant_variance","linear_subject_probe_balanced_accuracy","adaptive_subject_attack_balanced_accuracy","cross_session_verification_balanced_accuracy","cross_session_same_different_auroc"]
    grouped={}
    for row in metrics:grouped.setdefault((row["method"],row["strength"]),[]).append(row)
    summary=[]
    for (method,strength),rows in grouped.items():
        item={"method":method,"strength":strength,"folds":len({r['fold'] for r in rows}),"seeds":len({r['seed'] for r in rows})}
        for key in keys:item[key]=float(np.mean([float(r[key]) for r in rows]))
        summary.append(item)
    _csv(RESULT/"method_summary.csv",summary)
    seed_rows=[]
    for (method,strength,seed),rows in _group(metrics,("method","strength","seed")).items():
        item={"method":method,"strength":strength,"seed":seed}
        for key in keys:item[key]=float(np.mean([float(row[key]) for row in rows]))
        seed_rows.append(item)
    _csv(RESULT/"seed_effects.csv",seed_rows)
    biological={key:rows for key,rows in _group(participants,("method","strength","participant")).items()}
    averaged={key:{field:float(np.mean([float(row[field]) for row in rows])) for field in ("fixed_head_balanced_accuracy","adaptive_subject_attack_recall","cross_session_same_different_auroc")} for key,rows in biological.items()}
    def vector(method,strength,field):return np.asarray([averaged[(method,strength,participant)][field] for participant in range(1,10)])
    contrast_specs=[
        ("SANDiff_strong_minus_RAW_fixed",vector("SANDiff","strong","fixed_head_balanced_accuracy")-vector("RAW","na","fixed_head_balanced_accuracy")),
        ("SANDiff_strong_privacy_minus_RAW",vector("RAW","na","adaptive_subject_attack_recall")-vector("SANDiff","strong","adaptive_subject_attack_recall")),
        ("SANDiff_strong_AUROC_reduction_minus_RAW",vector("RAW","na","cross_session_same_different_auroc")-vector("SANDiff","strong","cross_session_same_different_auroc")),
        ("SANDiff_strong_minus_LEACE_fixed",vector("SANDiff","strong","fixed_head_balanced_accuracy")-vector("LEACE","na","fixed_head_balanced_accuracy")),
        ("SANDiff_strong_minus_one_step_fixed",vector("SANDiff","strong","fixed_head_balanced_accuracy")-vector("one_step","strong","fixed_head_balanced_accuracy")),
        ("SANDiff_strong_privacy_minus_one_step",vector("one_step","strong","adaptive_subject_attack_recall")-vector("SANDiff","strong","adaptive_subject_attack_recall")),
        ("full_sampler_minus_single_fixed",vector("SANDiff","strong","fixed_head_balanced_accuracy")-vector("SANDiff_single_checkpoint","strong","fixed_head_balanced_accuracy")),
        ("full_sampler_privacy_minus_single",vector("SANDiff_single_checkpoint","strong","adaptive_subject_attack_recall")-vector("SANDiff","strong","adaptive_subject_attack_recall")),
    ]
    rng=np.random.default_rng(330);contrast_rows=[]
    for name,values in contrast_specs:
        bootstrap=np.mean(rng.choice(values,(50000,len(values)),replace=True),axis=1)
        contrast_rows.append({"contrast":name,"participant_mean":float(values.mean()),"participant_median":float(np.median(values)),"ci95_low":float(np.quantile(bootstrap,.025)),"ci95_high":float(np.quantile(bootstrap,.975)),"positive_count":int((values>0).sum()),"participants":len(values)})
    _csv(RESULT/"contrast_summary.csv",contrast_rows)
    lookup={(r["method"],r["strength"]):r for r in summary};raw=lookup[("RAW","na")];strong=lookup[("SANDiff","strong")];one=lookup[("one_step","strong")];single=lookup[("SANDiff_single_checkpoint","strong")]
    diagnosis={"status":"development_complete","full_pool_subjects_per_fold":6,"primary_operating_point":"strong","sandiff_reverse_steps":10,"K":1,"strong_sandiff_vs_raw":{"fixed_head_delta":strong["fixed_head_balanced_accuracy"]-raw["fixed_head_balanced_accuracy"],"adaptive_privacy_utility":raw["adaptive_subject_attack_balanced_accuracy"]-strong["adaptive_subject_attack_balanced_accuracy"],"verification_auroc_reduction":raw["cross_session_same_different_auroc"]-strong["cross_session_same_different_auroc"]},"strong_sandiff_vs_one_step":{"fixed_head_delta":strong["fixed_head_balanced_accuracy"]-one["fixed_head_balanced_accuracy"],"adaptive_privacy_utility":one["adaptive_subject_attack_balanced_accuracy"]-strong["adaptive_subject_attack_balanced_accuracy"]},"full_sampler_vs_single_checkpoint":{"fixed_head_delta":strong["fixed_head_balanced_accuracy"]-single["fixed_head_balanced_accuracy"],"adaptive_privacy_utility":single["adaptive_subject_attack_balanced_accuracy"]-strong["adaptive_subject_attack_balanced_accuracy"]},"waveform_sealed_reads":0,"selection_uses_outer_test":False}
    if diagnosis["strong_sandiff_vs_raw"]["adaptive_privacy_utility"]>0 and diagnosis["strong_sandiff_vs_raw"]["fixed_head_delta"]>=0:position="SANDiff positive advantage retained"
    elif abs(diagnosis["strong_sandiff_vs_one_step"]["fixed_head_delta"])<0.005 and abs(diagnosis["strong_sandiff_vs_one_step"]["adaptive_privacy_utility"])<0.01:position="SANDiff and one-step practically equivalent"
    else:position="one-step clearly preferable"
    diagnosis["final_method_positioning"]=position
    (RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def _group(rows,fields):
    result={}
    for row in rows:result.setdefault(tuple(row[field] for field in fields),[]).append(row)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","run","aggregate"));parser.add_argument("--fold",type=int);parser.add_argument("--seed",type=int);args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="run":
        if args.fold not in range(3) or args.seed not in SEEDS:parser.error("registered --fold and --seed required")
        prepare();run_consolidation_fold(CACHE,RESULT,args.fold,args.seed,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:aggregate()
    return 0


if __name__=="__main__":raise SystemExit(main())
