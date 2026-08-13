from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/taas_waveform_v37t"
FIGURE = ROOT / "figures/taas_waveform_v37t"
FIGURE.mkdir(parents=True, exist_ok=True)


natural = list(csv.DictReader((RESULT / "common_natural_summary.csv").open()))
lookup = {(row["method"], row["metric"]): float(row["mean"]) for row in natural if row.get("mean")}
methods = ["V27_ENERGY_SDEDIT_L05", "V27_ENERGY_SDEDIT_L2", "V27_ENERGY_SDEDIT_L8"]
labels = ["L0.5", "L2", "L8"]
fig, ax = plt.subplots(figsize=(5.3, 4.2))
for method, label in zip(methods, labels):
    x = lookup[method, "low_eog_observation_retention"]
    y = lookup[method, "artifact_attenuation_db"]
    ax.scatter(x, y, s=55); ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points")
ax.set(xlabel="Low-EOG observation retention", ylabel="Artifact attenuation (dB)", title="Frozen V27 energy trade-off")
fig.tight_layout(); fig.savefig(FIGURE / "attenuation_retention_pareto.png", dpi=180); plt.close(fig)


uncertainty = list(csv.DictReader((RESULT / "uncertainty_summary.csv").open()))
method_order = ["V26_CALIB_SDEDIT_MATCH", "V26_POP_SDEDIT", "V27_ENERGY_SDEDIT_L05"]
short = ["CalibSDEdit", "PopSDEdit", "EnergySDEdit L0.5"]
levels = [.5, .8, .9]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for method, name in zip(method_order, short):
    means = [np.mean([float(row["coverage"]) for row in uncertainty if row["method"] == method and row["reference"] == "ensemble_quantiles" and float(row["level"]) == level]) for level in levels]
    ax.plot(levels, means, "o-", label=name)
ax.plot(levels, levels, "k--", label="nominal")
ax.set(xlabel="Nominal interval", ylabel="Empirical coverage", title="Uncalibrated K=16 intervals"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGURE / "uncertainty_coverage.png", dpi=180); plt.close(fig)


fig, ax = plt.subplots(figsize=(6.5, 4.2))
rhos = [[float(row["error_dispersion_spearman"]) for row in uncertainty if row["method"] == method and row["reference"] == "ensemble_quantiles" and float(row["level"]) == .8] for method in method_order]
ax.boxplot(rhos, labels=short, showmeans=True); ax.axhline(0, color="black", lw=.8)
ax.set(ylabel="Participant error–dispersion Spearman rho", title="Dispersion informativeness")
fig.tight_layout(); fig.savefig(FIGURE / "error_dispersion.png", dpi=180); plt.close(fig)


points = list(csv.DictReader((RESULT / "participant_effects.csv").open()))
fig, ax = plt.subplots(figsize=(6.7, 4.2))
summaries = ["single_draw", "sample_mean", "sample_median", "matched_deterministic"]
width = .22
for index, method in enumerate(("V26_CALIB_SDEDIT_MATCH", "V27_ENERGY_SDEDIT_L05")):
    values = []
    for summary in summaries:
        selected = [float(row["rrmse_temporal"]) for row in points if row["method"] == method and row["summary"] == summary]
        values.append(np.mean(selected) if selected else np.nan)
    ax.bar(np.arange(len(summaries)) + (index-.5)*width, values, width, label=short[index*2])
ax.set_xticks(range(len(summaries)), ["single", "mean", "median", "matched DET"]); ax.set_ylabel("Paired temporal RRMSE"); ax.legend()
fig.tight_layout(); fig.savefig(FIGURE / "stochastic_point_summaries.png", dpi=180); plt.close(fig)
