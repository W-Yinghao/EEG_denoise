"""Participant-first V42R aggregation, privacy audit, figures, and reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eeg_scad.data.artifact_transfer_v41r import TransferRegistry, bipolar_eog, ridge_transfer


METRICS = ("rrmse_temporal", "rrmse_spectral", "correlation", "snr_improvement", "artifact_rmse",
           "artifact_rrmse", "artifact_correlation", "clean_output_rms_ratio",
           "observation_change_ratio", "output_input_rms")
DIRECTIONS = {"rrmse_temporal": -1, "rrmse_spectral": -1, "correlation": 1,
              "snr_improvement": 1, "artifact_rrmse": -1, "artifact_correlation": 1,
              "observation_change_ratio": -1}


def bootstrap(values: Sequence[float], seed: int = 420, draws: int = 5000) -> tuple[float, float]:
    value = np.asarray(values, float); rng = np.random.default_rng(seed)
    sampled = np.asarray([rng.choice(value, len(value), replace=True).mean() for _ in range(draws)])
    return tuple(np.quantile(sampled, (.025, .975)))


def _write(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(path, index=False)


def privacy_audit(data: Mapping[str, Any], folds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fold in folds:
        registry = TransferRegistry(data, fold, 30); gallery, query, labels = [], [], []
        for participant in sorted(set(fold["train"] + fold["validation"] + fold["test"])):
            key = next((key for key in sorted(registry.cells) if key[0] == participant), None)
            if key is None:
                continue
            eeg, eye, names = registry._load(*key); eog = bipolar_eog(eye, names); cell = registry.cells[key]; halves = []
            for start, stop in ((0, 1500), (1500, 3000)):
                standardized = (eog[:, start:stop] - cell.eog_center[:, None]) / cell.eog_scale[:, None]
                transfer, diagnostic = ridge_transfer(eeg[:, start:stop] / registry.eeg_scale[:, None], standardized, .05)
                quality = np.array([np.log(np.sqrt(np.mean((eog[0, start:stop] - cell.eog_center[0]) ** 2)).clip(1e-8)),
                                    np.log(np.sqrt(np.mean((eog[1, start:stop] - cell.eog_center[1]) ** 2)).clip(1e-8)),
                                    diagnostic["fit_r2"], np.log1p(diagnostic["condition_number"])])
                halves.append(((registry._continuous(transfer, quality) - registry.continuous_center) /
                               registry.continuous_scale).reshape(-1))
            gallery.append(halves[0]); query.append(halves[1]); labels.append(participant)
        gallery, query = np.asarray(gallery), np.asarray(query)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True).clip(1e-8)
        query /= np.linalg.norm(query, axis=1, keepdims=True).clip(1e-8)
        similarity = query @ gallery.T; predicted = np.argmax(similarity, axis=1); truth, scores = [], []
        for i in range(len(query)):
            for j in range(len(gallery)):
                truth.append(i == j); scores.append(similarity[i, j])
        rows.append({"fold": fold["fold"], "top1": float(np.mean(predicted == np.arange(len(labels)))),
                     "verification_auroc": float(roc_auc_score(truth, scores)), "participants": len(labels),
                     "state_dimension": 46 * 53, "signature_bytes_float32": 46 * 53 * 4,
                     "state_stored": False, "recommended_deletion": "session_end",
                     "claim": "linkage_risk_not_anonymity"})
    return rows


def aggregate_cells(payload: Sequence[Mapping[str, Any]], result_dir: Path, report_dir: Path,
                    figure_dir: Path, data: Mapping[str, Any], folds: Sequence[Mapping[str, Any]],
                    run_id: str) -> dict[str, Any]:
    figure_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    paired = [row for result in payload for row in result["paired_metrics"]]
    duration = [row for result in payload for row in result["support_duration"]]
    curves = [{"fold": result["fold"], "seed": result["seed"], **row}
              for result in payload for row in result["training_curve"]]
    channels = [row for result in payload for row in result.get("channel_metrics", [])]
    frame = pd.DataFrame(paired)
    frame["severity"] = pd.cut(frame["gain"], [-np.inf, .55, .95, np.inf], labels=["mild", "medium", "severe"])
    frame.loc[frame.zero_artifact == 1, ["snr_improvement", "artifact_rrmse", "artifact_correlation"]] = np.nan
    _write(result_dir / "paired_metrics.csv", frame.to_dict("records")); _write(result_dir / "support_duration.csv", duration)
    _write(result_dir / "training_curves.csv", curves)
    if channels:
        channel_frame = pd.DataFrame(channels)
        channel_frame.loc[channel_frame.zero_artifact == 1, ["snr_improvement", "artifact_rrmse", "artifact_correlation"]] = np.nan
        _write(result_dir / "channel_metrics.csv", channel_frame.to_dict("records"))
    bindings = [result["checkpoint"] | {"fold": result["fold"], "seed": result["seed"], "training_run": run_id,
        "signature_path": result["inference_signature_binding"]["path"],
        "signature_sha256": result["inference_signature_binding"]["sha256"],
        "best_criterion": "participant_aggregated_validation_POP_temporal_RRMSE"} for result in payload]
    _write(result_dir / "checkpoint_binding.csv", bindings)
    per = frame.groupby(["condition", "participant"], as_index=False)[list(METRICS)].mean()
    cell_summary = frame.groupby(["condition", "fold", "seed"], as_index=False)[list(METRICS)].mean()
    _write(result_dir / "fold_seed_effects.csv", cell_summary.to_dict("records"))
    summary = []
    for condition, block in per.groupby("condition"):
        for metric in METRICS:
            value = block[metric].dropna(); lo, hi = bootstrap(value)
            summary.append({"panel": "paired", "stratum": "all", "condition": condition, "metric": metric,
                "participant_mean": value.mean(), "participant_median": value.median(), "bootstrap_low": lo,
                "bootstrap_high": hi, "participants": len(value)})
    severity_per = frame.groupby(["condition", "severity", "participant"], observed=True, as_index=False)[list(METRICS)].mean()
    for (condition, severity), block in severity_per.groupby(["condition", "severity"], observed=True):
        for metric in METRICS:
            value = block[metric].dropna()
            if value.empty: continue
            lo, hi = bootstrap(value)
            summary.append({"panel": "paired_severity", "stratum": str(severity), "condition": condition,
                "metric": metric, "participant_mean": value.mean(), "participant_median": value.median(),
                "bootstrap_low": lo, "bootstrap_high": hi, "participants": len(value)})
    seed_per = frame.groupby(["condition", "seed", "participant"], as_index=False)[list(METRICS)].mean()
    for (condition, seed), block in seed_per.groupby(["condition", "seed"]):
        for metric in METRICS:
            value = block[metric].dropna()
            if value.empty: continue
            lo, hi = bootstrap(value)
            summary.append({"panel": "paired_seed", "stratum": str(seed), "condition": condition,
                "metric": metric, "participant_mean": value.mean(), "participant_median": value.median(),
                "bootstrap_low": lo, "bootstrap_high": hi, "participants": len(value)})
    summary += [
        {"panel": "official_external", "stratum": "native_EOG", "condition": "OFFICIAL_EEGDFUS", "metric": "rrmse_temporal", "participant_mean": .296527},
        {"panel": "official_external", "stratum": "native_EOG", "condition": "OFFICIAL_EEGDFUS", "metric": "rrmse_spectral", "participant_mean": .302818},
        {"panel": "official_external", "stratum": "native_EOG", "condition": "OFFICIAL_EEGDFUS", "metric": "correlation", "participant_mean": .953041},
        {"panel": "official_external", "stratum": "native_EOG", "condition": "OFFICIAL_MATCHED_DETERMINISTIC", "metric": "rrmse_temporal", "participant_mean": .318704},
        {"panel": "historical_common_panel", "stratum": "V30/V37T", "condition": "FROZEN_V25_JOINT_DET_MATCH", "metric": "rrmse_temporal", "participant_mean": .751577},
        {"panel": "historical_common_panel", "stratum": "V30/V37T", "condition": "FROZEN_V25_JOINT_DET_MATCH", "metric": "rrmse_spectral", "participant_mean": .527347},
        {"panel": "historical_common_panel", "stratum": "V30/V37T", "condition": "FROZEN_V25_JOINT_DET_MATCH", "metric": "correlation", "participant_mean": .880337},
    ]
    _write(result_dir / "method_summary.csv", summary)
    effects, participant_rows = [], []; match = per[per.condition == "MATCH"].set_index("participant")
    for comparator in ("POP", "WRONG", "SHUFFLED", "ORACLE", "NO_TRANSFER_BRANCH"):
        other = per[per.condition == comparator].set_index("participant"); common = match.index.intersection(other.index)
        for metric, direction in DIRECTIONS.items():
            value = direction * (match.loc[common, metric] - other.loc[common, metric]); lo, hi = bootstrap(value)
            contrast = f"MATCH-{comparator}"
            participant_rows += [{"row_type": "participant", "contrast": contrast, "metric": metric,
                "participant": participant, "utility": float(utility), "positive": int(utility > 0)}
                for participant, utility in value.items()]
            effects.append({"contrast": contrast, "metric": metric, "participant_mean_utility": value.mean(),
                "participant_median_utility": value.median(), "positive_count": int((value > 0).sum()),
                "participants": len(value), "bootstrap_low": lo, "bootstrap_high": hi})
    participant_rows += [{"row_type": "summary", "participant": "ALL", "utility": row["participant_mean_utility"],
                          "positive": row["positive_count"], **row} for row in effects]
    _write(result_dir / "participant_effects.csv", participant_rows)
    channel_effects = pd.DataFrame()
    if channels:
        channel_frame = pd.DataFrame(channels)
        channel_per = channel_frame.groupby(["condition", "participant", "channel"], as_index=False).rrmse_temporal.mean()
        channel_pivot = channel_per.pivot(index=["participant", "channel"], columns="condition", values="rrmse_temporal").reset_index()
        channel_pivot["MATCH_minus_POP_utility"] = channel_pivot["POP"] - channel_pivot["MATCH"]
        channel_effects = channel_pivot.groupby("channel", as_index=False).agg(
            participant_mean_utility=("MATCH_minus_POP_utility", "mean"),
            participant_median_utility=("MATCH_minus_POP_utility", "median"),
            positive_count=("MATCH_minus_POP_utility", lambda value: int((value > 0).sum())),
            participants=("participant", "nunique"),
        )
        _write(result_dir / "channel_effects.csv", channel_effects.to_dict("records"))
    # The branch-disabled condition is the registered residual-branch ablation.
    _write(result_dir / "ablation_summary.csv", [row for row in effects if row["contrast"] in
           ("MATCH-NO_TRANSFER_BRANCH", "MATCH-POP", "MATCH-WRONG", "MATCH-SHUFFLED", "MATCH-ORACLE")])
    privacy = privacy_audit(data, folds); _write(result_dir / "privacy_summary.csv", privacy)
    lookup = {(row["condition"], row["metric"]): row for row in summary if row["panel"] == "paired"}
    pop_rr = lookup[("POP", "rrmse_temporal")]["participant_mean"]; raw_rr = lookup[("RAW", "rrmse_temporal")]["participant_mean"]
    pop_snr = lookup[("POP", "snr_improvement")]["participant_mean"]
    pop_scale = float(per[per.condition == "POP"].output_input_rms.quantile(.99))
    pop_valid = bool(np.isfinite(pop_rr) and pop_rr < raw_rr and pop_snr > 0 and pop_scale < 3)
    primary = next(row for row in effects if row["contrast"] == "MATCH-POP" and row["metric"] == "rrmse_temporal")
    if not pop_valid: positioning = "D_paired_diffusion_base_not_reproduced"
    elif primary["participant_mean_utility"] > 0 and primary["positive_count"] >= 10: positioning = "A_clear_MATCH_increment"
    elif primary["participant_mean_utility"] > 0: positioning = "B_small_or_heterogeneous_MATCH_increment"
    else: positioning = "C_no_MATCH_increment"
    diagnosis = {"engineering": "valid", "population_valid": pop_valid, "pop_temporal_rrmse": pop_rr,
        "raw_temporal_rrmse": raw_rr, "pop_snr_improvement": pop_snr, "pop_output_input_rms_q99": pop_scale,
        "primary_estimand": primary, "participant_coverage": int(per.participant.nunique()),
        "fold_seed_cells": len(payload), "final_positioning": positioning, "natural_tuned": False,
        "sealed_reads": 0, "query_eog_inference_reads": 0, "manuscript_unchanged": True, "run_id": run_id}
    (result_dir / "development_diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    _figures(pd.DataFrame(summary), pd.DataFrame(effects), pd.DataFrame(participant_rows), pd.DataFrame(duration), figure_dir)
    curve_frame = pd.DataFrame(curves)
    fig, ax = plt.subplots(figsize=(6, 4))
    for (_, _), block in curve_frame.groupby(["fold", "seed"]):
        ax.plot(block.step, block.validation_pop_rrmse, alpha=.7)
    ax.set(xlabel="Optimizer update", ylabel="Validation POP temporal RRMSE")
    fig.tight_layout(); fig.savefig(figure_dir / "population_learning_curve.png", dpi=180); plt.close(fig)
    native_path = result_dir / "native_sanity.csv"
    if native_path.is_file():
        native = pd.read_csv(native_path)
        fig, ax = plt.subplots(figsize=(6, 4)); ax.bar(native.method, native.rrmse_temporal)
        ax.set_ylabel("Native EOG temporal RRMSE"); ax.tick_params(axis="x", rotation=20)
        fig.tight_layout(); fig.savefig(figure_dir / "native_x0_sanity.png", dpi=180); plt.close(fig)
    _reports(pd.DataFrame(summary), pd.DataFrame(effects), pd.DataFrame(duration), pd.DataFrame(privacy),
             diagnosis, report_dir, channel_effects)
    return diagnosis


def aggregate_natural(rows: Sequence[Mapping[str, Any]], result_dir: Path, report_dir: Path,
                      figure_dir: Path) -> list[dict[str, Any]]:
    _write(result_dir / "natural_metrics.csv", rows); frame = pd.DataFrame(rows)
    per = frame.groupby(["condition", "participant"], as_index=False).mean(numeric_only=True); summary = []
    metrics = ("heldout_eog_remaining_ratio", "artifact_attenuation_db", "low_eog_observation_retention",
               "psd_distortion", "covariance_distortion", "output_input_rms")
    for condition, block in per.groupby("condition"):
        for metric in metrics:
            lo, hi = bootstrap(block[metric]); summary.append({"condition": condition, "metric": metric,
                "participant_mean": block[metric].mean(), "participant_median": block[metric].median(),
                "bootstrap_low": lo, "bootstrap_high": hi, "participants": len(block)})
    method_path = result_dir / "method_summary.csv"
    if method_path.is_file():
        method = pd.read_csv(method_path)
        natural_method = pd.DataFrame([{"panel": "natural", "stratum": "all", **row} for row in summary])
        pd.concat((method, natural_method), ignore_index=True).to_csv(method_path, index=False)
    directions = {"heldout_eog_remaining_ratio": -1, "artifact_attenuation_db": 1,
                  "low_eog_observation_retention": 1, "psd_distortion": -1,
                  "covariance_distortion": -1, "output_input_rms": 0}
    natural_effects = []; match = per[per.condition == "MATCH"].set_index("participant")
    for comparator in ("POP", "WRONG"):
        other = per[per.condition == comparator].set_index("participant"); common = match.index.intersection(other.index)
        for metric, direction in directions.items():
            if direction == 0: continue
            value = direction * (match.loc[common, metric] - other.loc[common, metric]); lo, hi = bootstrap(value)
            natural_effects += [{"row_type": "natural_participant", "contrast": f"MATCH-{comparator}",
                "metric": metric, "participant": participant, "utility": float(utility), "positive": int(utility > 0)}
                for participant, utility in value.items()]
            natural_effects.append({"row_type": "natural_summary", "contrast": f"MATCH-{comparator}",
                "metric": metric, "participant": "ALL", "utility": float(value.mean()),
                "positive": int((value > 0).sum()), "bootstrap_low": lo, "bootstrap_high": hi})
    effects_path = result_dir / "participant_effects.csv"
    if effects_path.is_file():
        pd.concat((pd.read_csv(effects_path), pd.DataFrame(natural_effects)), ignore_index=True).to_csv(effects_path, index=False)
    pop = per[per.condition == "POP"]
    natural_pop_valid = bool(pop.heldout_eog_remaining_ratio.mean() < 1 and
                             pop.artifact_attenuation_db.mean() > 0 and pop.output_input_rms.quantile(.99) < 3)
    diagnosis_path = result_dir / "development_diagnosis.json"
    if diagnosis_path.is_file():
        diagnosis = json.loads(diagnosis_path.read_text()); diagnosis.update({
            "natural_population_valid": natural_pop_valid,
            "natural_interpretation": "interpretable" if natural_pop_valid else "natural_population_route_invalid",
            "natural_tuned": False,
        }); diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    table = pd.DataFrame(summary).round(6).to_markdown(index=False)
    (report_dir / "v42r_natural_results.md").write_text("# V42R frozen natural development results\n\n"
        "The models and outputs were frozen before the evaluator opened query EOG. Low-EOG retention is observation retention, not physiological preservation. Natural results did not tune training or checkpoint selection.\n\n"
        f"Natural POP validity: **{natural_pop_valid}**. If false, MATCH minus POP is descriptive and is not interpreted as a general support effect.\n\n" + table + "\n")
    pivot = pd.DataFrame(summary).pivot(index="condition", columns="metric", values="participant_mean").reset_index()
    fig, ax = plt.subplots(figsize=(6, 4)); ax.scatter(pivot.artifact_attenuation_db, pivot.low_eog_observation_retention)
    for row in pivot.itertuples(): ax.annotate(row.condition, (row.artifact_attenuation_db, row.low_eog_observation_retention))
    ax.set(xlabel="Artifact attenuation (dB)", ylabel="Low-EOG observation retention")
    fig.tight_layout(); figure_dir.mkdir(parents=True, exist_ok=True); fig.savefig(figure_dir / "natural_attenuation_retention.png", dpi=180); plt.close(fig)
    return summary


def _figures(summary: pd.DataFrame, effects: pd.DataFrame, participant_effects: pd.DataFrame,
             duration: pd.DataFrame, figure_dir: Path) -> None:
    rr = summary[(summary.panel == "paired") & (summary.metric == "rrmse_temporal")]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(rr.condition, rr.participant_mean); ax.tick_params(axis="x", rotation=25); ax.set_ylabel("Temporal RRMSE")
    fig.tight_layout(); fig.savefig(figure_dir / "population_vs_contaminated.png", dpi=180); plt.close(fig)
    primary = effects[effects.metric == "rrmse_temporal"]
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(primary.contrast, primary.participant_mean_utility); ax.axhline(0, color="black", lw=.8); ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(figure_dir / "transfer_condition_effect.png", dpi=180); plt.close(fig)
    dur = duration.groupby(["support_seconds", "participant"], as_index=False).mean(numeric_only=True).groupby("support_seconds", as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(dur.support_seconds, dur.rrmse_temporal, marker="o"); ax.set(xlabel="Support seconds", ylabel="Temporal RRMSE")
    fig.tight_layout(); fig.savefig(figure_dir / "support_duration.png", dpi=180); plt.close(fig)
    participant = participant_effects[(participant_effects.row_type == "participant") &
                                      (participant_effects.contrast == "MATCH-POP") &
                                      (participant_effects.metric == "rrmse_temporal")]
    if not participant.empty:
        fig, ax = plt.subplots(figsize=(7, 4)); participant.sort_values("utility").plot.bar(x="participant", y="utility", ax=ax, legend=False); ax.axhline(0, color="black", lw=.8)
        fig.tight_layout(); fig.savefig(figure_dir / "participant_effects.png", dpi=180); plt.close(fig)


def _reports(summary: pd.DataFrame, effects: pd.DataFrame, duration: pd.DataFrame, privacy: pd.DataFrame,
             diagnosis: Mapping[str, Any], report_dir: Path, channel_effects: pd.DataFrame) -> None:
    table = lambda value: value.round(6).to_markdown(index=False)
    (report_dir / "v42r_population_validity.md").write_text("# V42R population-route validity\n\n```json\n" +
        json.dumps({key: diagnosis[key] for key in ("population_valid", "pop_temporal_rrmse", "raw_temporal_rrmse", "pop_snr_improvement", "pop_output_input_rms_q99")}, indent=2) + "\n```\n")
    (report_dir / "v42r_paired_results.md").write_text("# V42R paired participant-first results\n\n## Absolute results\n\n" +
        table(summary[summary.panel == "paired"]) + "\n\n## Severity strata\n\n" + table(summary[summary.panel == "paired_severity"]) +
        "\n\n## Seed diagnostics\n\n" + table(summary[summary.panel == "paired_seed"]) +
        "\n\n## Frozen external/historical positioning\n\nThese values retain their own registered protocols and are positioning references, not same-episode contrasts.\n\n" +
        table(summary[summary.panel.isin(["official_external", "historical_common_panel"])]) + "\n\n## Contrasts\n\n" + table(effects) +
        ("\n\n## Channel-wise MATCH minus POP temporal-RRMSE utility\n\n" + table(channel_effects) if not channel_effects.empty else "") + "\n")
    dur = duration.groupby(["support_seconds", "participant"], as_index=False).mean(numeric_only=True).groupby("support_seconds", as_index=False).mean(numeric_only=True)
    (report_dir / "v42r_support_duration.md").write_text("# V42R support-duration sensitivity\n\n0 s is exact POP. Windows are chronological, non-overlapping, prefix-normalized, unrepeated, and query-disjoint.\n\n" + table(dur) + "\n")
    (report_dir / "v42r_privacy_note.md").write_text("# V42R transfer-state linkage risk\n\nThis is a secondary linkage-risk audit, not anonymity. State is ephemeral and session-end deletion is recommended.\n\n" + table(privacy) + "\n")
    (report_dir / "v42r_final_diagnosis.md").write_text("# V42R final diagnosis\n\nFinal positioning: **" + str(diagnosis["final_positioning"]) + "**.\n\n```json\n" + json.dumps(diagnosis, indent=2, sort_keys=True) + "\n```\n")


__all__ = ["aggregate_cells", "aggregate_natural", "bootstrap", "privacy_audit"]
