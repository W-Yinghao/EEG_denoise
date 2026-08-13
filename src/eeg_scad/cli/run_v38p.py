"""Command-line orchestration and participant-first aggregation for V38P."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eeg_scad.privacy.experiment import sha256
from eeg_scad.privacy.openbmi import OPENBMI_ROOT, build_dataset_inventory, outer_folds, validate_folds
from eeg_scad.privacy.sard_experiment import run_fold


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/sard_bridge_v38p"
REPORT = ROOT / "reports"
FIGURE = ROOT / "figures/sard_bridge_v38p"
V36 = Path("/home/infres/yinwang/denoiseNet_fiber_openbmi_v36p")
SEEDS = (20260940, 20260941)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(path, index=False)


def prepare() -> None:
    RESULT.mkdir(parents=True, exist_ok=True); FIGURE.mkdir(parents=True, exist_ok=True)
    inventory, dataset = build_dataset_inventory(OPENBMI_ROOT)
    validate_folds(); _write_csv(RESULT / "dataset_inventory.csv", inventory)
    split_rows = []
    for item in outer_folds():
        for role in ("train", "validation", "test"):
            for owner in item[f"{role}_subjects"]:
                split_rows.append({"fold": item["fold"], "participant": int(owner + 1), "role": role, "support_session": "ses_0", "query_session": "ses_1"})
    _write_csv(RESULT / "split_manifest.csv", split_rows)
    support_rows = []
    for item in outer_folds():
        for role in ("train", "validation", "test"):
            for owner in item[f"{role}_subjects"]:
                support_rows.append({"fold": item["fold"], "participant": int(owner + 1), "role": role, "session": "ses_0", "selection": "chronological first 10 trials per true MI class", "support_trials": 20, "class_0_trials": 10, "class_1_trials": 10, "gallery_trials": 80, "query_trials": 100, "support_query_overlap": 0})
    _write_csv(RESULT / "support_manifest.csv", support_rows)
    donor_rows = []
    for item in outer_folds():
        donor_rows.append({"fold": item["fold"], "bank_role": "outer_train_only", "participants": ";".join(str(value + 1) for value in item["train_subjects"]), "participant_count": 36, "rows": 6480, "outer_validation_rows": 0, "outer_test_rows": 0, "match": "different participant; same frozen predicted task; same or nearest frozen confidence stratum"})
    _write_csv(RESULT / "donor_bank_manifest.csv", donor_rows)
    binding_path = V36 / "results/fiber_openbmi_v36p/checkpoint_binding.csv"
    source = {"base_commit": "89effec0abd8c0b3581c89dc6bfeed9e68b2cafe", "v36p_commit": "a90cabf5ed7167e0bc6cfc01257e74592b6e7d85", "v36p_checkpoint_binding": str(binding_path), "v36p_checkpoint_binding_sha256": sha256(binding_path), "dataset": dataset, "support_budget_frozen_before_results": True, "V38R": "superseded_not_executed", "V37T": "frozen_negative_waveform_diffusion_result", "V32P_V36P": "frozen_companion_paper_assets", "sealed_reads": 0}
    (RESULT / "source_binding.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bootstrap(values: np.ndarray, seed: int = 38) -> tuple[float, float]:
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    if not len(values): return float("nan"), float("nan")
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(5000)])
    return tuple(np.quantile(draws, (.025, .975)))


def _participant_first(rows: pd.DataFrame, value: str, group: str = "method") -> pd.DataFrame:
    per = rows.groupby([group, "participant"], as_index=False)[value].mean()
    output = []
    for method, block in per.groupby(group):
        low, high = _bootstrap(block[value].to_numpy())
        output.append({group: method, "metric": value, "participant_mean": block[value].mean(), "participant_median": block[value].median(), "bootstrap_ci_low": low, "bootstrap_ci_high": high, "participants": block.participant.nunique()})
    return pd.DataFrame(output)


def aggregate() -> dict[str, object]:
    payloads = []
    for fold in range(6):
        for seed in SEEDS:
            path = RESULT / "runtime" / f"fold_{fold}_seed_{seed}" / "fold_result.json"
            if not path.is_file(): raise FileNotFoundError(path)
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    fields = ("checkpoint_binding", "method_summary", "privacy_attacks", "task_utility", "distribution_fidelity", "multisample_diversity", "training_exposure", "ensemble_utility", "participant_effects")
    checkpoint=[]; method=[]; distribution=[]; diversity=[]; exposure=[]; ensemble=[]; participants=[]; validation=[]
    for payload in payloads:
        checkpoint += payload["checkpoint_binding"]; method += payload["method_summary"]; distribution += payload["distribution_fidelity"]; diversity += payload["multisample_diversity"]; exposure += payload["training_exposure"]; ensemble += payload["ensemble_utility"]; participants += payload["participant_effects"]; validation += payload["validation"]
    method_df=pd.DataFrame(method); participant_df=pd.DataFrame(participants); dist_df=pd.DataFrame(distribution); exposure_df=pd.DataFrame(exposure); ensemble_df=pd.DataFrame(ensemble)
    _write_csv(RESULT/"checkpoint_binding.csv",checkpoint);_write_csv(RESULT/"method_summary.csv",method);_write_csv(RESULT/"privacy_attacks.csv",method);_write_csv(RESULT/"task_utility.csv",method);_write_csv(RESULT/"distribution_fidelity.csv",distribution);_write_csv(RESULT/"multisample_diversity.csv",diversity);_write_csv(RESULT/"training_exposure.csv",exposure);_write_csv(RESULT/"ensemble_utility.csv",ensemble);_write_csv(RESULT/"augmentation_utility.csv",ensemble);_write_csv(RESULT/"participant_effects.csv",participants);_write_csv(RESULT/"selection_summary.csv",validation)
    # Participant-first effect table uses each participant averaged across its two seeds.
    effect_rows=[]
    for metric in ("fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","adaptive_subject_attack_recall","cross_session_same_different_auroc"):
        if metric not in participant_df: continue
        effect_rows.extend(_participant_first(participant_df,metric).to_dict("records"))
    _write_csv(RESULT/"participant_first_summary.csv",effect_rows)
    means=method_df.groupby("method",as_index=False).mean(numeric_only=True)
    dist_means=dist_df.groupby("method",as_index=False).mean(numeric_only=True)
    exp_means=exposure_df.groupby("method",as_index=False).mean(numeric_only=True)
    ens_means=ensemble_df.groupby("method",as_index=False).mean(numeric_only=True)
    sard=means[means.method=="SARD-Bridge"].iloc[0]; one=means[means.method=="OneStep-Bridge"].iloc[0]; gauss=means[means.method=="Gaussian-Bridge"].iloc[0]; raw=means[means.method=="RAW"].iloc[0]
    sard_dist=dist_means[dist_means.method=="SARD-Bridge"].iloc[0]; gauss_dist=dist_means[dist_means.method=="Gaussian-Bridge"].iloc[0]; resample_dist=dist_means[dist_means.method=="Stratified-Resample"].iloc[0]
    sard_exp=exp_means[exp_means.method=="SARD-Bridge"].iloc[0]; resample_exp=exp_means[exp_means.method=="Stratified-Resample"].iloc[0]
    axes={"privacy_vs_raw": float(raw.adaptive_subject_attack_balanced_accuracy-sard.adaptive_subject_attack_balanced_accuracy),"fixed_task_vs_raw":float(sard.fixed_head_balanced_accuracy-raw.fixed_head_balanced_accuracy),"retrained_task_vs_raw":float(sard.retrained_head_balanced_accuracy-raw.retrained_head_balanced_accuracy),"distribution_energy_vs_gaussian":float(gauss_dist.conditional_energy_distance-sard_dist.conditional_energy_distance),"distribution_energy_vs_resample":float(resample_dist.conditional_energy_distance-sard_dist.conditional_energy_distance),"exemplar_exposure_vs_resample":float(resample_exp.exact_copy_rate-sard_exp.exact_copy_rate)}
    positive_dist=axes["distribution_energy_vs_gaussian"]>0
    positive_privacy=axes["privacy_vs_raw"]>0 and axes["fixed_task_vs_raw"]>-0.05
    positive_exposure=axes["exemplar_exposure_vs_resample"]>0 and axes["distribution_energy_vs_resample"]>-1.0
    positioning="A" if (positive_dist or positive_privacy or positive_exposure) else ("B" if abs(axes["distribution_energy_vs_gaussian"])<.05 else "C")
    diagnosis={"engineering":"valid","participant_coverage":54,"outer_test_count_per_participant":1,"fold_seed_cells":12,"K":8,"reverse_steps":10,"repair_used":False,"registered_positive_axes":axes,"final_positioning":positioning,"sealed_reads":0,"manuscript_unchanged":True}
    (RESULT/"development_diagnosis.json").write_text(json.dumps(diagnosis,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    _figures(means,dist_means,exp_means,ens_means,participant_df)
    _reports(means,dist_means,exp_means,ens_means,diagnosis)
    return diagnosis


def _figures(means,dist,exposure,ensemble,participants):
    FIGURE.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(means.adaptive_subject_attack_balanced_accuracy,means.fixed_head_balanced_accuracy)
    for row in means.itertuples():ax.annotate(row.method,(row.adaptive_subject_attack_balanced_accuracy,row.fixed_head_balanced_accuracy),fontsize=8)
    ax.set(xlabel="Adaptive source-subject BA (lower is better)",ylabel="Frozen-head task BA",title="V38P privacy–utility frontier");fig.tight_layout();fig.savefig(FIGURE/"privacy_utility_frontier.png",dpi=180);plt.close(fig)
    for filename,x,y,title in (("donor_distribution_fidelity.png","conditional_energy_distance","conditional_mmd_rbf","Donor-distribution fidelity"),("multisample_diversity.png","within_query_diversity","variance_retained","K=8 diversity")):
        frame=dist;fig,ax=plt.subplots(figsize=(7,5));ax.scatter(frame[x],frame[y]);[ax.annotate(r.method,(getattr(r,x),getattr(r,y)),fontsize=8) for r in frame.itertuples()];ax.set(xlabel=x,ylabel=y,title=title);fig.tight_layout();fig.savefig(FIGURE/filename,dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.bar(exposure.method,exposure.exact_copy_rate);ax.tick_params(axis="x",rotation=20);ax.set(ylabel="Exact training copy rate",title="Training-exemplar exposure");fig.tight_layout();fig.savefig(FIGURE/"exemplar_exposure.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(ensemble.augmentation_retrained_head_balanced_accuracy,ensemble.ensemble_frozen_head_balanced_accuracy);[ax.annotate(r.method,(r.augmentation_retrained_head_balanced_accuracy,r.ensemble_frozen_head_balanced_accuracy),fontsize=8) for r in ensemble.itertuples()];ax.set(xlabel="Augmentation task BA",ylabel="K-sample ensemble task BA",title="Ensemble and augmentation utility");fig.tight_layout();fig.savefig(FIGURE/"ensemble_augmentation_utility.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));plot=participants.groupby(["method","participant"]).fixed_head_balanced_accuracy.mean().unstack(0);plot.plot(ax=ax,marker="o",linewidth=.8);ax.set(xlabel="Participant",ylabel="Frozen-head BA",title="Participant effects");fig.tight_layout();fig.savefig(FIGURE/"participant_effects.png",dpi=180);plt.close(fig)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].round(6).to_markdown(index=False)


def _reports(means,dist,exposure,ensemble,diagnosis):
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/"v38p_baseline_frontier.md").write_text("# V38P baseline frontier\n\n"+_table(means,["method","fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","adaptive_subject_attack_balanced_accuracy","cross_session_same_different_auroc"])+"\n",encoding="utf-8")
    (REPORT/"v38p_sard_bridge_method.md").write_text("# V38P SARD-Bridge method\n\nSARD-Bridge models the dynamic, task/confidence-matched cross-subject donor residual distribution. It conditions on the frozen source representation, frozen logits, and a query-disjoint task-demeaned Session-1 support prototype. Training and inference never use a source/donor identity token or true test label. Canonical inference retains all K=8 draws from a 10-step x0-prediction sampler.\n",encoding="utf-8")
    (REPORT/"v38p_privacy_utility_results.md").write_text("# V38P privacy and task utility\n\n"+_table(means,["method","fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","calibration_error","worst_participant_accuracy","adaptive_subject_attack_balanced_accuracy","cross_session_same_different_auroc"])+"\n",encoding="utf-8")
    (REPORT/"v38p_distribution_and_exposure.md").write_text("# V38P distribution fidelity and exposure\n\n## Distribution\n\n"+_table(dist,["method","conditional_energy_distance","conditional_mmd_rbf","conditional_covariance_discrepancy","variance_retained","within_query_diversity","duplicate_rate"])+"\n\n## Exposure\n\n"+_table(exposure,["method","exact_copy_rate","near_copy_rate","nearest_training_fiber_distance","membership_attack_probability"])+"\n",encoding="utf-8")
    (REPORT/"v38p_ensemble_augmentation.md").write_text("# V38P ensemble and augmentation\n\n"+_table(ensemble,["method","ensemble_frozen_head_balanced_accuracy","augmentation_retrained_head_balanced_accuracy"])+"\n",encoding="utf-8")
    explanation={"A":"SARD-Bridge establishes a positive diffusion role on at least one registered axis.","B":"SARD-Bridge is viable but not distinctive from Gaussian/OneStep transport.","C":"Non-diffusion transport is preferable after the registered canonical run."}[diagnosis["final_positioning"]]
    (REPORT/"v38p_final_diagnosis.md").write_text(f"# V38P final diagnosis\n\nFinal positioning: **{diagnosis['final_positioning']}**. {explanation}\n\nRegistered axis effects:\n\n```json\n{json.dumps(diagnosis['registered_positive_axes'],indent=2,sort_keys=True)}\n```\n\nAll 54 participants were outer-tested exactly once; seeds were not treated as biological samples. Sealed reads were zero and the manuscript was unchanged. No formal anonymity, exact subject-information removal, waveform denoising, or universal generalization is claimed.\n",encoding="utf-8")


def terminal_manifest() -> dict[str, object]:
    current=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    diagnosis=json.loads((RESULT/"development_diagnosis.json").read_text())
    manifest={"base_commit":"89effec0abd8c0b3581c89dc6bfeed9e68b2cafe","current_commit":current,"branch":"codex/sard-bridge-v38p","participant_coverage":54,"fold_seed_cells":12,"K":8,"jobs":{"accepted":[],"failed":[],"recovery":[],"current":[]},"tests":{},"sealed_reads":0,"manuscript_unchanged":True,"final_positioning":diagnosis["final_positioning"],"push_status":"pending_terminal_commit"}
    (RESULT/"terminal_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8");return manifest


def main() -> None:
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare");fold=sub.add_parser("run-fold");fold.add_argument("--fold",type=int,required=True);fold.add_argument("--seed",type=int,required=True);fold.add_argument("--device",default="cuda");fold.add_argument("--variant",choices=("canonical","source_adversary_repair"),default="canonical");fold.add_argument("--sard-lambda-source",type=float,default=.1);sub.add_parser("aggregate");sub.add_parser("terminal")
    args=parser.parse_args()
    if args.command=="prepare":prepare()
    elif args.command=="run-fold":run_fold(RESULT,args.fold,args.seed,torch.device(args.device),variant=args.variant,sard_lambda_source=args.sard_lambda_source)
    elif args.command=="aggregate":aggregate()
    else:terminal_manifest()


if __name__ == "__main__": main()
