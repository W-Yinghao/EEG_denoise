"""V41R preflight, execution, participant-first aggregation, and reporting."""
from __future__ import annotations

import argparse
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

from eeg_scad.data.artifact_transfer_v41r import TransferRegistry, bipolar_eog, ridge_transfer
from eeg_scad.training.train_v41r import natural_evaluator, natural_output_freeze, run_fold


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/calib_eegdfus_v41r"
REPORT = ROOT / "reports"
FIGURE = ROOT / "figures/calib_eegdfus_v41r"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_eegdfus_v41r")
SEEDS = (20261110, 20261111)


def read_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def configs():
    data = read_yaml(ROOT / "configs/setcalibdiff_v25/data.yaml")
    data.update(read_yaml(ROOT / "configs/calib_eegdfus_v41r/data.yaml"))
    data["v19_derived_root"] = data["source_root"]
    folds = read_yaml(ROOT / "configs/setcalibdiff_v25/folds.yaml")["folds"]
    training = read_yaml(ROOT / "configs/calib_eegdfus_v41r/training.yaml")
    return data, folds, training


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def prepare() -> None:
    data, folds, _ = configs(); RESULT.mkdir(parents=True, exist_ok=True); FIGURE.mkdir(parents=True, exist_ok=True)
    split = [{"fold": fold["fold"], "participant": participant, "role": role}
             for fold in folds for role in ("train", "validation", "test") for participant in fold[role]]
    write_csv(RESULT / "split_manifest.csv", split)
    support, transfer = [], []
    for fold in folds:
        registry = TransferRegistry(data, fold, 30)
        rows = registry.manifest_rows(); support.extend(rows); transfer.extend(rows)
    write_csv(RESULT / "support_manifest.csv", support); write_csv(RESULT / "transfer_signature_manifest.csv", transfer)
    official_checkout = Path("/home/infres/yinwang/v40r_third_party/EEGDfus")
    official_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=official_checkout, text=True).strip()
    official_dirty = subprocess.check_output(["git", "status", "--short"], cwd=official_checkout, text=True).strip()
    if official_sha != "a19a652b3b6346188ae77067e1daf8b90cad005f" or official_dirty:
        raise RuntimeError("official EEGDfus binding changed")
    source = {
        "base_commit": "ade827ebc587f4edf8c4eede11a5d4472116338f",
        "official_eegdfus_repository": "https://github.com/XYH0118/EEGDfus",
        "official_eegdfus_commit": official_sha, "official_checkout_unchanged": True,
        "official_reproduction_status": "reasonable_nonidentical_reproduction",
        "official_eog_temporal_rrmse": 0.296527, "official_eog_spectral_rrmse": 0.302818,
        "official_eog_correlation": 0.953041, "d4pm": "official_release_not_runnable",
        "paired_resource": "V19/V24 participant-session counterfactual operator panel",
        "klados_participant_identity_status": "not_recoverable_not_silently_relabelled",
        "v40r_result_tree": "a9d6c300d44aebee2693129675167b24842f5cb4",
        "manuscript_tree": "b15768f122dd20da3ae460bfaaee41275cb7eacb",
        "eog_electrodes": ["HEOGL", "HEOGR", "VEOGU", "VEOGL"],
        "bipolar_regressors": ["VEOGU-VEOGL", "HEOGL-HEOGR"],
        "sealed_reads": 0, "query_eog_inference_reads": 0,
    }
    (RESULT / "source_binding.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")


def bootstrap(values, seed=411, draws=5000):
    value = np.asarray(values, float); rng = np.random.default_rng(seed)
    sampled = np.asarray([rng.choice(value, len(value), replace=True).mean() for _ in range(draws)])
    return np.quantile(sampled, (.025, .975))


def _privacy(data, folds):
    rows = []
    for fold in folds:
        registry = TransferRegistry(data, fold, 30)
        gallery, query, labels = [], [], []
        for participant in sorted(set(fold["train"] + fold["validation"] + fold["test"])):
            key = next((key for key in sorted(registry.cells) if key[0] == participant), None)
            if key is None: continue
            eeg, eye, names = registry._load(*key); eog = bipolar_eog(eye, names); cell = registry.cells[key]
            halves = []
            for start, stop in ((0, 1500), (1500, 3000)):
                standardized = (eog[:, start:stop] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
                transfer, diagnostic = ridge_transfer(eeg[:, start:stop] / registry.eeg_scale[:, None], standardized, .05)
                quality = np.array([np.log(np.sqrt(np.mean((eog[0, start:stop]-cell.eog_center[0])**2)).clip(1e-8)),
                                    np.log(np.sqrt(np.mean((eog[1, start:stop]-cell.eog_center[1])**2)).clip(1e-8)),
                                    diagnostic["fit_r2"], np.log1p(diagnostic["condition_number"])])
                halves.append(((registry._continuous(transfer, quality)-registry.continuous_center)/registry.continuous_scale).reshape(-1))
            gallery.append(halves[0]); query.append(halves[1]); labels.append(participant)
        gallery, query = np.asarray(gallery), np.asarray(query)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True).clip(1e-8); query /= np.linalg.norm(query, axis=1, keepdims=True).clip(1e-8)
        similarity = query @ gallery.T; predicted = np.argmax(similarity, axis=1)
        truth, scores = [], []
        for i in range(len(query)):
            for j in range(len(gallery)): truth.append(i == j); scores.append(similarity[i, j])
        rows.append({"fold": fold["fold"], "top1": float(np.mean(predicted == np.arange(len(labels)))),
                     "verification_auroc": float(roc_auc_score(truth, scores)), "participants": len(labels),
                     "signature_bytes": 46 * 53 * 4, "state_stored": False,
                     "recommended_deletion": "session_end", "claim": "linkage_risk_not_anonymity"})
    return rows


def aggregate(run_id: str) -> dict:
    data, folds, _ = configs(); payload = []
    for fold in range(5):
        for seed in SEEDS:
            path = DERIVED / run_id / f"fold_{fold}_seed_{seed}" / "result.json"
            if not path.is_file(): raise FileNotFoundError(path)
            payload.append(json.loads(path.read_text()))
    paired = [row for result in payload for row in result["paired_metrics"]]
    channel = [row for result in payload for row in result["channel_metrics"]]
    duration = [row for result in payload for row in result["support_duration"]]
    binding = [result["checkpoint"] | {"training_run": run_id,
               "signature_path": result["inference_signature_binding"]["path"],
               "signature_sha256": result["inference_signature_binding"]["sha256"]} for result in payload]
    frame = pd.DataFrame(paired); frame.loc[frame.zero_artifact == 1, ["snr_improvement", "artifact_rrmse", "artifact_correlation"]] = np.nan
    write_csv(RESULT / "paired_metrics.csv", frame.to_dict("records")); write_csv(RESULT / "checkpoint_binding.csv", binding)
    write_csv(RESULT / "support_duration.csv", duration)
    channel_summary = pd.DataFrame(channel).groupby(["condition", "participant", "channel"], as_index=False).mean(numeric_only=True)
    write_csv(RESULT / "ablation_summary.csv", channel_summary.to_dict("records"))
    metrics = ["rrmse_temporal", "rrmse_spectral", "correlation", "snr_improvement", "artifact_rmse",
               "artifact_rrmse", "artifact_correlation", "clean_output_rms_ratio", "observation_change_ratio", "output_input_rms"]
    per = frame.groupby(["condition", "participant"], as_index=False)[metrics].mean()
    summary = []
    for condition, block in per.groupby("condition"):
        for metric in metrics:
            value = block[metric].dropna(); lo, hi = bootstrap(value)
            summary.append({"panel": "paired", "condition": condition, "metric": metric,
                            "participant_mean": value.mean(), "participant_median": value.median(),
                            "bootstrap_low": lo, "bootstrap_high": hi, "participants": len(value)})
    official = [{"panel": "official_external", "condition": "OFFICIAL_EEGDFUS", "metric": "rrmse_temporal", "participant_mean": .296527},
                {"panel": "official_external", "condition": "OFFICIAL_EEGDFUS", "metric": "rrmse_spectral", "participant_mean": .302818},
                {"panel": "official_external", "condition": "OFFICIAL_EEGDFUS", "metric": "correlation", "participant_mean": .953041}]
    write_csv(RESULT / "method_summary.csv", summary + official)
    effects = []
    directions = {"rrmse_temporal": -1, "rrmse_spectral": -1, "correlation": 1, "snr_improvement": 1,
                  "artifact_rrmse": -1, "artifact_correlation": 1, "observation_change_ratio": -1}
    match = per[per.condition == "MATCH"].set_index("participant")
    for comparator in ("POP", "WRONG", "SHUFFLED", "ORACLE", "CHANNEL_ONLY"):
        other = per[per.condition == comparator].set_index("participant"); common = match.index.intersection(other.index)
        for metric, direction in directions.items():
            value = direction * (match.loc[common, metric] - other.loc[common, metric]); lo, hi = bootstrap(value)
            effects.append({"contrast": f"MATCH-{comparator}", "metric": metric, "participant_mean_utility": value.mean(),
                            "participant_median_utility": value.median(), "positive_count": int((value > 0).sum()),
                            "participants": len(value), "bootstrap_low": lo, "bootstrap_high": hi})
    write_csv(RESULT / "participant_effects.csv", effects)
    privacy = _privacy(data, folds); write_csv(RESULT / "privacy_summary.csv", privacy)
    pop_rr = next(row["participant_mean"] for row in summary if row["condition"] == "POP" and row["metric"] == "rrmse_temporal")
    raw_rr = next(row["participant_mean"] for row in summary if row["condition"] == "RAW" and row["metric"] == "rrmse_temporal")
    pop_snr = next(row["participant_mean"] for row in summary if row["condition"] == "POP" and row["metric"] == "snr_improvement")
    pop_scale = per[per.condition == "POP"].output_input_rms.quantile(.99)
    pop_valid = bool(np.isfinite(pop_rr) and pop_rr < raw_rr and pop_snr > 0 and pop_scale < 3 and pop_rr < 1.192548)
    primary = next(row for row in effects if row["contrast"] == "MATCH-POP" and row["metric"] == "rrmse_temporal")
    if not pop_valid: positioning = "D_base_model_not_established"
    elif primary["participant_mean_utility"] > 0 and primary["positive_count"] >= 10: positioning = "A_clear_increment"
    elif primary["participant_mean_utility"] > 0: positioning = "B_heterogeneous_increment"
    else: positioning = "C_no_increment"
    diagnosis = {"engineering": "valid", "population_valid": pop_valid, "pop_temporal_rrmse": pop_rr,
                 "raw_temporal_rrmse": raw_rr, "pop_snr_improvement": pop_snr,
                 "pop_output_input_rms_q99": pop_scale, "v40r_pop_temporal_rrmse": 1.192548,
                 "primary_estimand": primary, "participant_coverage": int(per.participant.nunique()),
                 "fold_seed_cells": len(payload), "final_positioning": positioning,
                 "natural_authorized": pop_valid, "sealed_reads": 0, "query_eog_inference_reads": 0,
                 "manuscript_unchanged": True, "run_id": run_id}
    (RESULT / "development_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    figures(pd.DataFrame(summary), pd.DataFrame(effects), pd.DataFrame(duration)); reports(pd.DataFrame(summary), pd.DataFrame(effects), pd.DataFrame(duration), pd.DataFrame(privacy), diagnosis)
    return diagnosis


def aggregate_natural(run_id: str) -> None:
    rows = []
    for fold in range(5):
        for seed in SEEDS:
            path = DERIVED / run_id / f"fold_{fold}_seed_{seed}" / "natural_result.json"
            if path.is_file(): rows += json.loads(path.read_text())["natural_metrics"]
    if not rows:
        (RESULT / "natural_metrics.csv").write_text("status\nnot_run_population_invalid_or_pending\n")
        (REPORT / "v41r_natural_results.md").write_text("# V41R natural results\n\nNot run or not interpreted because population validity was not established.\n")
        return
    write_csv(RESULT / "natural_metrics.csv", rows); frame = pd.DataFrame(rows)
    per = frame.groupby(["condition", "participant"], as_index=False).mean(numeric_only=True); summary=[]
    for condition, block in per.groupby("condition"):
        for metric in ("heldout_eog_remaining_ratio", "artifact_attenuation_db", "low_eog_observation_retention", "psd_distortion", "covariance_distortion", "output_input_rms"):
            lo, hi = bootstrap(block[metric]); summary.append({"condition": condition, "metric": metric, "participant_mean": block[metric].mean(), "participant_median": block[metric].median(), "bootstrap_low": lo, "bootstrap_high": hi, "participants": len(block)})
    text = pd.DataFrame(summary).round(6).to_markdown(index=False)
    (REPORT / "v41r_natural_results.md").write_text("# V41R natural results\n\nLow-EOG retention is observation retention, not physiological preservation. Inference outputs were digest-frozen before the evaluator opened query EOG.\n\n" + text + "\n")
    pivot = pd.DataFrame(summary).pivot(index="condition", columns="metric", values="participant_mean").reset_index(); fig, ax = plt.subplots(figsize=(6,4)); ax.scatter(pivot.artifact_attenuation_db, pivot.low_eog_observation_retention); [ax.annotate(row.condition, (row.artifact_attenuation_db,row.low_eog_observation_retention)) for row in pivot.itertuples()]; fig.tight_layout(); fig.savefig(FIG/"natural_attenuation_retention.png",dpi=180); plt.close(fig)


def figures(summary, effects, duration):
    rr = summary[(summary.panel == "paired") & (summary.metric == "rrmse_temporal")]
    fig, ax = plt.subplots(figsize=(7,4)); ax.bar(rr.condition, rr.participant_mean); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Temporal RRMSE"); fig.tight_layout(); fig.savefig(FIG/"official_vs_paired_population.png",dpi=180); plt.close(fig)
    primary = effects[effects.metric == "rrmse_temporal"]; fig, ax = plt.subplots(figsize=(7,4)); ax.bar(primary.contrast, primary.participant_mean_utility); ax.axhline(0,color="black",lw=.8); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(FIG/"transfer_condition_effect.png",dpi=180); plt.close(fig)
    dur = duration.groupby(["support_seconds","participant"],as_index=False).mean(numeric_only=True).groupby("support_seconds",as_index=False).mean(numeric_only=True); fig, ax=plt.subplots(figsize=(6,4)); ax.plot(dur.support_seconds,dur.rrmse_temporal,marker="o"); ax.set(xlabel="Support seconds",ylabel="Temporal RRMSE"); fig.tight_layout(); fig.savefig(FIG/"support_duration.png",dpi=180); plt.close(fig)
    oracle = effects[(effects.metric=="rrmse_temporal") & effects.contrast.isin(["MATCH-POP","MATCH-ORACLE"])]; fig,ax=plt.subplots(figsize=(5,4)); ax.bar(oracle.contrast,oracle.participant_mean_utility); ax.axhline(0,color="black",lw=.8); fig.tight_layout(); fig.savefig(FIG/"oracle_headroom.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(primary.contrast,primary.positive_count); ax.set(ylabel="Positive participants"); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(FIG/"participant_effects.png",dpi=180); plt.close(fig)


def reports(summary, effects, duration, privacy, diagnosis):
    table = lambda frame: frame.round(6).to_markdown(index=False)
    (REPORT/"v41r_official_semantics_backbone.md").write_text("# V41R official-semantics backbone\n\nThe shared B×C -> [B*C,1,512] model retains the official dual branch, epsilon prediction, 500-step linear schedule, and conditional observation at every ancestral step. It has no 46-channel convolution.\n\n```json\n"+json.dumps({k:diagnosis[k] for k in ("population_valid","pop_temporal_rrmse","raw_temporal_rrmse","pop_snr_improvement","pop_output_input_rms_q99","v40r_pop_temporal_rrmse")},indent=2)+"\n```\n")
    (REPORT/"v41r_paired_results.md").write_text("# V41R paired participant-first results\n\n"+table(summary[summary.panel=="paired"])+"\n\n## Contrasts\n\n"+table(effects)+"\n")
    dur=duration.groupby(["support_seconds","participant"],as_index=False).mean(numeric_only=True).groupby("support_seconds",as_index=False).mean(numeric_only=True)
    (REPORT/"v41r_support_duration.md").write_text("# V41R support duration\n\nThe 0-second condition is the exact population signature. Prefix-only normalization and non-overlapping chronological windows are used.\n\n"+table(dur)+"\n")
    (REPORT/"v41r_privacy_note.md").write_text("# V41R transfer-signature linkage note\n\nThis is a linkage-risk diagnostic, not anonymity. The state is not stored and should be deleted at session end.\n\n"+table(privacy)+"\n")
    (REPORT/"v41r_final_diagnosis.md").write_text("# V41R final diagnosis\n\nFinal positioning: **"+diagnosis["final_positioning"]+"**.\n\n```json\n"+json.dumps(diagnosis,indent=2,sort_keys=True)+"\n```\n")


def terminal() -> None:
    diagnosis = json.loads((RESULT/"development_diagnosis.json").read_text())
    manifest = {"base_commit":"ade827ebc587f4edf8c4eede11a5d4472116338f","branch":"codex/calib-eegdfus-artifact-transfer-v41r","pre_terminal_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"official_eegdfus_commit":"a19a652b3b6346188ae77067e1daf8b90cad005f","jobs":{"accepted":[],"failed":[],"recovery":[],"current":[]},"tests":{},"participant_coverage":diagnosis["participant_coverage"],"population_valid":diagnosis["population_valid"],"final_positioning":diagnosis["final_positioning"],"sealed_reads":0,"query_eog_inference_reads":0,"manuscript_unchanged":True,"push_status":"pending_terminal_commit"}
    (RESULT/"terminal_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="stage",required=True)
    sub.add_parser("prepare")
    run=sub.add_parser("run"); run.add_argument("--fold",type=int,required=True);run.add_argument("--seed",type=int,required=True);run.add_argument("--run-id",required=True);run.add_argument("--updates",type=int,default=12000)
    aggregate_parser=sub.add_parser("aggregate");aggregate_parser.add_argument("--run-id",required=True)
    freeze=sub.add_parser("natural-freeze");freeze.add_argument("--fold",type=int,required=True);freeze.add_argument("--seed",type=int,required=True);freeze.add_argument("--paired-run-id",required=True);freeze.add_argument("--run-id",required=True)
    evaluate=sub.add_parser("natural-evaluate");evaluate.add_argument("--fold",type=int,required=True);evaluate.add_argument("--seed",type=int,required=True);evaluate.add_argument("--run-id",required=True)
    natural_aggregate=sub.add_parser("natural-aggregate");natural_aggregate.add_argument("--run-id",required=True)
    sub.add_parser("terminal"); args=parser.parse_args();data,folds,training=configs()
    if args.stage=="prepare":prepare()
    elif args.stage=="run":run_fold(DERIVED,data,folds[args.fold],args.seed,torch.device("cuda"),args.updates,training["batch_episodes"],training["validation_interval"],args.run_id)
    elif args.stage=="aggregate":aggregate(args.run_id)
    elif args.stage=="natural-freeze":natural_output_freeze(DERIVED,data,folds[args.fold],args.seed,torch.device("cuda"),DERIVED/args.paired_run_id/f"fold_{args.fold}_seed_{args.seed}"/"result.json",args.run_id)
    elif args.stage=="natural-evaluate":natural_evaluator(DERIVED,data,folds[args.fold],args.seed,DERIVED/args.run_id/f"fold_{args.fold}_seed_{args.seed}"/"output_freeze.json")
    elif args.stage=="natural-aggregate":aggregate_natural(args.run_id)
    else:terminal()


if __name__ == "__main__":main()
