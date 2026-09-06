"""V42R preflight and server execution CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch
import yaml

from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
from eeg_scad.evaluation.aggregate_v42r import aggregate_cells, aggregate_natural
from eeg_scad.training.train_v42r import natural_evaluator, natural_output_freeze, native_sanity, paired_channel_evaluator, run_cell


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/calib_saddpm_cond_v42r"
REPORT = ROOT / "reports"
FIGURE = ROOT / "figures/calib_saddpm_cond_v42r"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_saddpm_cond_v42r")
SEEDS = (20261201, 20261202)


def _yaml(path): return yaml.safe_load(path.read_text())


def configs():
    data = _yaml(ROOT / "configs/setcalibdiff_v25/data.yaml")
    data.update(_yaml(ROOT / "configs/calib_saddpm_cond_v42r/data.yaml"))
    data["v19_derived_root"] = data["source_root"]
    folds = _yaml(ROOT / "configs/setcalibdiff_v25/folds.yaml")["folds"]
    training = _yaml(ROOT / "configs/calib_saddpm_cond_v42r/training.yaml")
    return data, folds, training


def sha256(path: Path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def prepare():
    data, folds, _ = configs(); RESULT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"fold":f["fold"],"participant":p,"role":role} for f in folds for role in ("train","validation","test") for p in f[role]]).to_csv(RESULT/"split_manifest.csv",index=False)
    rows=[]
    for fold in folds: rows += TransferRegistry(data,fold,30).manifest_rows()
    pd.DataFrame(rows).to_csv(RESULT/"support_manifest.csv",index=False);pd.DataFrame(rows).to_csv(RESULT/"transfer_signature_manifest.csv",index=False)
    official=Path("/home/infres/yinwang/v40r_third_party/EEGDfus")
    official_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=official,text=True).strip()
    if official_sha!="a19a652b3b6346188ae77067e1daf8b90cad005f" or subprocess.check_output(["git","status","--short"],cwd=official,text=True).strip():raise RuntimeError("official EEGDfus binding changed")
    binding={"base_commit":"8931ad7c036863976b4693f9f0721e11ab04857a","official_eegdfus_commit":official_sha,
             "official_status":"reasonable_nonidentical_reproduction","official_native_EOG":{"rrmse_temporal":.296527,"rrmse_spectral":.302818,"correlation":.953041},
             "d4pm":"official_release_not_runnable","eegoar_net":"protocol_incompatible_primary_paired_panel",
             "cleanroom":True,"reused_internal":["artifact_transfer_v41r","paired_metrics","V25 folds","V31 duration contract"],
             "prohibited_source_imports":[],"v41r_tree":subprocess.check_output(["git","rev-parse","HEAD:results/calib_eegdfus_v41r"],cwd=ROOT,text=True).strip(),
             "manuscript_tree":subprocess.check_output(["git","rev-parse","HEAD:taas_submission"],cwd=ROOT,text=True).strip(),"sealed_reads":0}
    (RESULT/"source_binding.json").write_text(json.dumps(binding,indent=2,sort_keys=True)+"\n")


def native():
    data,_,_=configs();native_sanity(Path(data["native_root"]),DERIVED/"native_sanity",torch.device("cuda"))


def run(fold_id,seed,updates,run_id):
    data,folds,training=configs();result=run_cell(DERIVED,data,folds[fold_id],seed,torch.device("cuda"),updates,run_id,training["effective_batch"],training["validation_interval"])
    if updates==training["pilot_updates"]:
        curve=result["training_curve"];pop=[r for r in result["paired_metrics"] if r["condition"]=="POP"]
        validity={"finite":all(torch.isfinite(torch.tensor([r["validation_pop_rrmse"] for r in curve]))),
                  "loss_reduction":curve[-1]["train_smooth_l1"]<curve[0]["train_smooth_l1"],
                  "transfer_gradient_nonzero":max(r["transfer_gradient_norm"] for r in curve)>0,
                  "output_scale_plausible":max(r["output_input_rms"] for r in pop)<3,
                  "identity_head_initialized":True,"engineering_repair_used":True,
                  "engineering_repair":"disable_AMP_after_nonfinite_gradient_at_pilot_update_1905"}
        (DERIVED/run_id/"pilot_validity.json").write_text(json.dumps(validity,indent=2,sort_keys=True)+"\n")
        if not all(validity[k] for k in ("finite","transfer_gradient_nonzero","output_scale_plausible","identity_head_initialized")):raise RuntimeError("V42R pilot engineering boundary failed")


def aggregate(run_id):
    data, folds, _ = configs(); payload = []
    for fold in range(5):
        for seed in SEEDS:
            path = DERIVED / run_id / f"fold_{fold}_seed_{seed}" / "result.json"
            if not path.is_file(): raise FileNotFoundError(path)
            cell = json.loads(path.read_text()); channel = path.parent / "channel_metrics.json"
            if channel.is_file(): cell["channel_metrics"] = json.loads(channel.read_text())["channel_metrics"]
            payload.append(cell)
    return aggregate_cells(payload, RESULT, REPORT, FIGURE, data, folds, run_id)


def natural_aggregate(run_id):
    rows = []
    for fold in range(5):
        for seed in SEEDS:
            path = DERIVED / run_id / f"fold_{fold}_seed_{seed}" / "natural_result.json"
            if not path.is_file(): raise FileNotFoundError(path)
            rows += json.loads(path.read_text())["natural_metrics"]
    aggregate_natural(rows, RESULT, REPORT, FIGURE)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="stage",required=True);sub.add_parser("prepare");sub.add_parser("native")
    runp=sub.add_parser("run");runp.add_argument("--fold",type=int,required=True);runp.add_argument("--seed",type=int,required=True);runp.add_argument("--updates",type=int,required=True);runp.add_argument("--run-id",required=True)
    aggregatep=sub.add_parser("aggregate");aggregatep.add_argument("--run-id",required=True)
    freeze=sub.add_parser("natural-freeze");freeze.add_argument("--fold",type=int,required=True);freeze.add_argument("--seed",type=int,required=True);freeze.add_argument("--paired-run-id",required=True);freeze.add_argument("--run-id",required=True)
    evaluate=sub.add_parser("natural-evaluate");evaluate.add_argument("--fold",type=int,required=True);evaluate.add_argument("--seed",type=int,required=True);evaluate.add_argument("--run-id",required=True)
    channel=sub.add_parser("paired-channel");channel.add_argument("--fold",type=int,required=True);channel.add_argument("--seed",type=int,required=True);channel.add_argument("--paired-run-id",required=True)
    natural_aggregatep=sub.add_parser("natural-aggregate");natural_aggregatep.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage=="native":native()
    elif args.stage=="run":run(args.fold,args.seed,args.updates,args.run_id)
    elif args.stage=="aggregate":aggregate(args.run_id)
    elif args.stage=="natural-freeze":
        data,folds,_=configs();natural_output_freeze(DERIVED,data,folds[args.fold],args.seed,torch.device("cuda"),DERIVED/args.paired_run_id/f"fold_{args.fold}_seed_{args.seed}"/"result.json",args.run_id)
    elif args.stage=="natural-evaluate":
        data,folds,_=configs();natural_evaluator(DERIVED,data,folds[args.fold],args.seed,DERIVED/args.run_id/f"fold_{args.fold}_seed_{args.seed}"/"output_freeze.json")
    elif args.stage=="paired-channel":
        data,folds,_=configs();paired_channel_evaluator(data,folds[args.fold],args.seed,torch.device("cuda"),DERIVED/args.paired_run_id/f"fold_{args.fold}_seed_{args.seed}"/"result.json")
    else:natural_aggregate(args.run_id)


if __name__=="__main__":main()
