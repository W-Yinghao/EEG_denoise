"""Command-line entry point for the V32P SANDiff pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.bci2a import outer_folds
from eeg_scad.privacy.experiment import prepare_cache, run_fold


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results" / "sandiff_v32p"
CACHE = RESULT / "runtime" / "bci2a_trials.npz"


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields=[]
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def prepare() -> None:
    metadata=prepare_cache(Path("/projects/EEG-foundation-model/BCI-IV"),CACHE)
    rows=[]
    for split in outer_folds():
        for subject in split["train_subjects"]:rows.append({"fold":split["fold"],"subject":subject,"role":"train","session_T":"model_train","session_E":"unused"})
        for subject in split["validation_subjects"]:rows.append({"fold":split["fold"],"subject":subject,"role":"validation","session_T":"unused","session_E":"model_selection"})
        for subject in split["test_subjects"]:rows.append({"fold":split["fold"],"subject":subject,"role":"outer_test","session_T":"adaptive_attack_gallery","session_E":"task_and_attack_query"})
    _csv(RESULT/"split_manifest.csv",rows)
    (RESULT/"runtime"/"data_inventory.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def aggregate() -> None:
    payloads=[json.loads((RESULT/"runtime"/f"fold_{fold}"/"fold_result.json").read_text()) for fold in range(3)]
    metrics=[row for payload in payloads for row in payload["metrics"]];participants=[row for payload in payloads for row in payload["participant_effects"]];latency=[row for payload in payloads for row in payload["latency"]];bindings=[row for payload in payloads for row in payload["checkpoint_binding"]]
    _csv(RESULT/"baseline_summary.csv",[r for r in metrics if r["method"] in {"RAW","LEACE","DANN"}])
    _csv(RESULT/"privacy_attacks.csv",metrics);_csv(RESULT/"task_utility.csv",metrics);_csv(RESULT/"privacy_utility_curve.csv",metrics);_csv(RESULT/"participant_effects.csv",participants);_csv(RESULT/"latency_summary.csv",latency);_csv(RESULT/"checkpoint_binding.csv",bindings)
    summary=[]
    keys=["fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","calibration_error","worst_participant_accuracy","between_participant_variance","linear_subject_probe_balanced_accuracy","adaptive_subject_attack_balanced_accuracy","cross_session_verification_balanced_accuracy","cross_session_same_different_auroc"]
    groups={}
    for row in metrics:groups.setdefault((row["method"],row["strength"]),[]).append(row)
    for (method,strength),rows in groups.items():
        item={"method":method,"strength":strength,"folds":len({r['fold'] for r in rows}),"seeds":len({r['seed'] for r in rows})}
        for key in keys:item[key]=float(np.mean([float(r[key]) for r in rows]))
        summary.append(item)
    _csv(RESULT/"method_summary.csv",summary)
    # Candidate selection is finalized after full curves exist; this provisional rule is transparent.
    candidates=[r for r in summary if r["method"] in {"SANDiff","one_step"}]
    for r in candidates:r["selection_balance"]=(r["fixed_head_balanced_accuracy"]+r["retrained_head_balanced_accuracy"])/2-0.25*r["adaptive_subject_attack_balanced_accuracy"]-0.10*abs(r["cross_session_same_different_auroc"]-0.5)
    selected=max(candidates,key=lambda r:r["selection_balance"])
    diagnosis={"status":"development_complete","selected_positive_candidate":selected["method"],"selected_strength":selected["strength"],"selection_is_human_reviewable_not_hard_gate":True,"waveform_interaction":"deferred_not_comparable","waveform_sealed_reads":0,"formal_anonymity_claim":False,"cross_dataset_claim":False}
    (RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","fold","aggregate"));parser.add_argument("--fold",type=int);args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="fold":
        if args.fold not in (0,1,2):parser.error("--fold 0, 1, or 2 required")
        prepare();run_fold(CACHE,RESULT,args.fold,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:aggregate()
    return 0


if __name__=="__main__":raise SystemExit(main())
