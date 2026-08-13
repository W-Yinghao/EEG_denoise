"""CLI for V36P OpenBMI exact-fiber external replication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.openbmi import OPENBMI_ROOT, build_dataset_inventory, outer_folds, validate_folds
from eeg_scad.privacy.openbmi_experiment import recover_retrained_participant_utility, run_openbmi_fold


ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results" / "fiber_openbmi_v36p"
FIGURE = ROOT / "figures" / "fiber_openbmi_v36p"
SEEDS = (20260940, 20260941)
BASE = "096b43fcb902e745811c953f1049b3e63fd90726"


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group(rows, fields):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def prepare() -> None:
    validate_folds()
    RESULT.mkdir(parents=True, exist_ok=True)
    inventory, manifest = build_dataset_inventory(OPENBMI_ROOT)
    _csv(RESULT / "dataset_manifest.csv", inventory)
    split_rows = []
    for split in outer_folds():
        for role in ("train", "validation", "test"):
            for subject in split[f"{role}_subjects"]:
                split_rows.append({
                    "fold": split["fold"], "participant": int(subject + 1), "internal_subject": subject,
                    "role": role, "selection_session": "ses_1" if role == "validation" else "none",
                    "refit_session": "ses_0" if role != "test" else "none",
                    "outer_gallery_session": "ses_0" if role == "test" else "none",
                    "outer_query_session": "ses_1" if role == "test" else "none",
                })
    _csv(RESULT / "split_manifest.csv", split_rows)
    (RESULT / "dataset_inventory.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean_rows(rows, group_fields, value_fields):
    output = []
    for key, values in _group(rows, group_fields).items():
        item = dict(zip(group_fields, key))
        for field in value_fields:
            item[field] = float(np.mean([float(row[field]) for row in values]))
        output.append(item)
    return output


def aggregate() -> None:
    payloads = []
    recoveries = []
    for fold in range(6):
        for seed in SEEDS:
            path = RESULT / "runtime" / f"fold_{fold}_seed_{seed}" / "fold_result.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
            recovery_path = path.parent / "retrained_participant_recovery.json"
            if not recovery_path.is_file():
                raise FileNotFoundError(recovery_path)
            recoveries.append(json.loads(recovery_path.read_text(encoding="utf-8")))
    mappings = {
        "checkpoint_binding": "checkpoint_binding.csv",
        "exact_preservation": "exact_preservation.csv",
        "head_aware_attacks": "head_aware_attacks.csv",
        "conditional_fiber_leakage": "conditional_fiber_leakage.csv",
        "training_exposure": "training_exposure.csv",
        "distribution_fidelity": "distribution_fidelity.csv",
        "multisample_diversity": "multisample_diversity.csv",
        "metrics": "task_utility.csv",
    }
    combined = {}
    for key, filename in mappings.items():
        combined[key] = [row for payload in payloads for row in payload[key]]
        if key == "training_exposure":
            # Supersede the primary-array threshold implementation for the
            # bank channel. Resampler.sample returns training rows verbatim;
            # its structural exact-copy rate is therefore exactly one. The
            # original immutable fold JSON remains available for provenance.
            for row in combined[key]:
                if row["method"] == "Fiber-Stratified-Resample":
                    row["exact_copy_rate"] = 1.0
                    row["exact_copy_definition"] = "structural_bytewise_training_bank_membership"
                    row["exact_copy_supersedes_primary_threshold_metric"] = True
        _csv(RESULT / filename, combined[key])
    _csv(RESULT / "resample_coverage.csv", [row for payload in payloads for row in payload["resample_coverage"]])
    _csv(RESULT / "gaussian_coverage.csv", [row for payload in payloads for row in payload["gaussian_coverage"]])

    # Long participant-first table; seeds are repeated measurements, never biological samples.
    participant = []
    for payload in payloads:
        for row in payload["participant_effects"]:
            for metric in ("fixed_head_balanced_accuracy", "linear_subject_probe_recall", "adaptive_subject_attack_recall", "cross_session_same_different_auroc"):
                participant.append({"fold": row["fold"], "seed": row["seed"], "participant": row["participant"], "method": row["method"], "family": "task_or_legacy_privacy", "metric": metric, "value": row[metric]})
        for row in payload["attack_participant_effects"]:
            for metric in ("subject_recall", "cross_entropy", "same_different_verification_auroc"):
                participant.append({"fold": row["fold"], "seed": row["seed"], "participant": row["participant"], "method": row["method"], "family": f"head_aware_{row['attacker']}_{row['feature']}", "metric": metric, "value": row[metric]})
        for row in payload["exposure_participant_effects"]:
            for metric in ("exact_copy_rate", "near_copy_rate", "nearest_training_fiber_distance", "membership_attack_probability"):
                value = 1.0 if row["method"] == "Fiber-Stratified-Resample" and metric == "exact_copy_rate" else row[metric]
                participant.append({"fold": row["fold"], "seed": row["seed"], "participant": row["participant"], "method": row["method"], "family": "training_exposure", "metric": metric, "value": value})
    recovered_participant = [row for payload in recoveries for row in payload["participant_rows"]]
    _csv(RESULT / "retrained_participant_utility.csv", recovered_participant)
    for row in recovered_participant:
        participant.append({"fold": row["fold"], "seed": row["seed"], "participant": row["participant"], "method": row["method"], "family": "retrained_task", "metric": "retrained_head_balanced_accuracy", "value": row["retrained_head_balanced_accuracy"]})
    _csv(RESULT / "participant_effects.csv", participant)

    task_summary = _mean_rows(combined["metrics"], ("method", "strength"), ("fixed_head_balanced_accuracy", "retrained_head_balanced_accuracy", "calibration_error", "worst_participant_accuracy", "between_participant_variance", "adaptive_subject_attack_balanced_accuracy", "cross_session_same_different_auroc"))
    recovered_summary = _mean_rows([row for payload in recoveries for row in payload["summary"]], ("method",), ("retrained_head_balanced_accuracy", "worst_participant_retrained_head_balanced_accuracy", "between_participant_retrained_variance"))
    recovered_lookup = {row["method"]: row for row in recovered_summary}
    for row in task_summary:
        recovered = recovered_lookup[row["method"]]
        row["retrained_head_balanced_accuracy"] = recovered["retrained_head_balanced_accuracy"]
        row["worst_participant_retrained_head_balanced_accuracy"] = recovered["worst_participant_retrained_head_balanced_accuracy"]
        row["between_participant_retrained_variance"] = recovered["between_participant_retrained_variance"]
        row["worst_participant_accuracy_legacy_semantics"] = "fixed_head; superseded for retrained utility"
    attack_summary = _mean_rows(combined["head_aware_attacks"], ("method", "attacker", "feature"), ("balanced_accuracy", "cross_entropy", "same_different_verification_auroc"))
    fidelity_summary = _mean_rows(combined["distribution_fidelity"], ("method",), ("conditional_covariance_relative_frobenius", "conditional_energy_distance", "conditional_mmd_rbf", "fiber_variance_retained"))
    exposure_summary = _mean_rows(combined["training_exposure"], ("method",), ("exact_copy_rate", "near_copy_rate", "nearest_training_fiber_distance", "nearest_heldout_fiber_distance", "membership_attack_probability", "membership_attack_positive_rate_0_5", "nearest_training_donor_max_share"))
    diversity_summary = _mean_rows(combined["multisample_diversity"], ("method",), ("within_H_sample_variance", "between_H_variance", "nearest_training_fiber_distance", "duplicate_rate", "sample_diversity"))
    _csv(RESULT / "task_utility_summary.csv", task_summary)
    _csv(RESULT / "head_aware_attack_summary.csv", attack_summary)
    _csv(RESULT / "distribution_fidelity_summary.csv", fidelity_summary)
    _csv(RESULT / "training_exposure_summary.csv", exposure_summary)
    _csv(RESULT / "multisample_diversity_summary.csv", diversity_summary)

    primary = []
    for (fold, seed, method, attacker), rows in _group(combined["head_aware_attacks"], ("fold", "seed", "method", "attacker")).items():
        eligible = [row for row in rows if row["feature"] in ("A_H", "A_Z", "A_HZ")]
        maximum = max(eligible, key=lambda row: float(row["balanced_accuracy"]))
        head = next(row for row in rows if row["feature"] == "A_H")
        primary.append({"fold": fold, "seed": seed, "method": method, "attacker": attacker, "primary_finite_threat_balanced_accuracy": maximum["balanced_accuracy"], "maximizing_feature": maximum["feature"], "A_H_balanced_accuracy": head["balanced_accuracy"], "privacy_semantics": "cannot claim below H-visible boundary"})
    _csv(RESULT / "primary_finite_threat.csv", primary)

    exact_valid = all(int(float(row["prediction_mismatch_count"])) == 0 and abs(float(row["fixed_head_ba_difference"])) < 1e-12 for row in combined["exact_preservation"])
    participants = sorted({int(row["participant"]) for row in participant})
    if participants != list(range(1, 55)):
        raise ValueError("participant-first aggregate does not cover all 54 participants")
    diagnosis = {
        "status": "external_development_complete",
        "base_commit": BASE,
        "dataset": "OpenBMI / Lee2019_MI",
        "participant_coverage": 54,
        "outer_test_count_per_participant": 1,
        "seeds": list(SEEDS),
        "seeds_are_biological_samples": False,
        "exact_function_preservation": exact_valid,
        "head_visible_privacy_boundary": "I(Z_prime;S|Y)=I(H;S|Y) for strong source-fiber-independent channels",
        "finite_attacker_not_mutual_information": True,
        "sandiff_deployment_requires_training_fiber_bank": False,
        "resample_deployment_requires_training_fiber_bank": True,
        "latency_benchmark_run": False,
        "waveform_sealed_reads": 0,
        "final_positioning": "PENDING_EVIDENCE_REVIEW_A_B_C_D",
    }
    (RESULT / "development_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figures(task_summary, attack_summary, fidelity_summary, exposure_summary, participant, primary)


def make_figures(task, attacks, fidelity, exposure, participant, primary) -> None:
    import matplotlib.pyplot as plt

    FIGURE.mkdir(parents=True, exist_ok=True)
    adaptive = [row for row in primary if row["attacker"] == "adaptive_mlp"]
    privacy = _mean_rows(adaptive, ("method",), ("primary_finite_threat_balanced_accuracy", "A_H_balanced_accuracy"))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(privacy)); width = 0.38
    ax.bar(x - width / 2, [row["primary_finite_threat_balanced_accuracy"] for row in privacy], width, label="max(A_H,A_Z,A_HZ)")
    ax.bar(x + width / 2, [row["A_H_balanced_accuracy"] for row in privacy], width, label="A_H")
    ax.set_xticks(x, [row["method"] for row in privacy], rotation=25, ha="right"); ax.set_ylabel("Adaptive subject BA"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE / "external_privacy_replication.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, ("conditional_covariance_relative_frobenius", "conditional_energy_distance", "conditional_mmd_rbf", "fiber_variance_retained")):
        ax.bar([row["method"] for row in fidelity], [row[metric] for row in fidelity]); ax.tick_params(axis="x", rotation=25); ax.set_title(metric.replace("_", " "), fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE / "model_only_distribution_fidelity.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, metric in zip(axes, ("exact_copy_rate", "nearest_training_fiber_distance", "membership_attack_probability")):
        ax.bar([row["method"] for row in exposure], [row[metric] for row in exposure]); ax.tick_params(axis="x", rotation=25); ax.set_title(metric.replace("_", " "), fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE / "training_exemplar_exposure.png", dpi=180); plt.close(fig)
    rows = [row for row in participant if row["family"] == "training_exposure" and row["metric"] == "membership_attack_probability"]
    methods = sorted({row["method"] for row in rows}); fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        values = _mean_rows([row for row in rows if row["method"] == method], ("participant",), ("value",)); ax.plot([row["participant"] for row in values], [row["value"] for row in values], marker="o", label=method)
    ax.set(xlabel="Participant", ylabel="Membership/exposure probability"); ax.legend(); fig.tight_layout(); fig.savefig(FIGURE / "participant_effects.png", dpi=180); plt.close(fig)
    privacy_map = {row["method"]: row["primary_finite_threat_balanced_accuracy"] for row in privacy}
    fidelity_map = {row["method"]: row["conditional_energy_distance"] for row in fidelity}; common = sorted(set(privacy_map) & set(fidelity_map))
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter([privacy_map[m] for m in common], [fidelity_map[m] for m in common])
    for method in common: ax.annotate(method, (privacy_map[method], fidelity_map[method]), fontsize=8)
    ax.set(xlabel="Primary finite-threat adaptive BA", ylabel="Conditional energy distance"); fig.tight_layout(); fig.savefig(FIGURE / "privacy_distribution_frontier.png", dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "task-recovery", "aggregate"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage in ("run", "task-recovery"):
        if args.fold not in range(6) or args.seed not in SEEDS:
            parser.error("registered fold/seed required")
        prepare()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.stage == "run":
            run_openbmi_fold(RESULT, args.fold, args.seed, device)
        else:
            recover_retrained_participant_utility(RESULT, args.fold, args.seed, device)
    else:
        aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
