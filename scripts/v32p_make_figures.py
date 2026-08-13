#!/usr/bin/env python
"""Create the registered V32P development figures from committed CSV summaries."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "sandiff_v32p"
OUT = ROOT / "figures" / "sandiff_v32p"
OUT.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(RESULT / "method_summary.csv")


def save(name: str) -> None:
    plt.tight_layout(); plt.savefig(OUT / name, dpi=180); plt.close()


plt.figure(figsize=(6.2, 4.4))
for method, marker in (("one_step", "o"), ("SANDiff", "s")):
    part = summary[summary.method == method].copy()
    order = pd.Categorical(part.strength, ["weak", "medium", "strong"], ordered=True)
    part = part.assign(order=order).sort_values("order")
    plt.plot(1 - part.adaptive_subject_attack_balanced_accuracy, part.fixed_head_balanced_accuracy, marker=marker, label=method)
    for _, row in part.iterrows(): plt.annotate(row.strength, (1-row.adaptive_subject_attack_balanced_accuracy,row.fixed_head_balanced_accuracy), xytext=(4,4), textcoords="offset points", fontsize=8)
for method, marker in (("RAW", "x"), ("LEACE", "^"), ("DANN", "v")):
    row=summary[summary.method==method].iloc[0];plt.scatter(1-row.adaptive_subject_attack_balanced_accuracy,row.fixed_head_balanced_accuracy,marker=marker,s=65,label=method)
plt.xlabel("Adaptive privacy utility (1 − subject balanced accuracy)");plt.ylabel("Fixed-head MI balanced accuracy");plt.legend(fontsize=8);plt.grid(alpha=.25);save("privacy_utility_frontier.png")

strong=summary[((summary.method.isin(["one_step","SANDiff"]))&(summary.strength=="strong"))|summary.method.isin(["RAW","LEACE","DANN"])]
plt.figure(figsize=(6.8,4.1));plt.bar(strong.method, strong.adaptive_subject_attack_balanced_accuracy,color=["#777777","#4c78a8","#f58518","#72b7b2","#e45756"]);plt.axhline(1/3,color="black",ls="--",lw=1,label="3-class chance");plt.ylabel("Adaptive subject attack balanced accuracy");plt.xticks(rotation=20);plt.legend();plt.grid(axis="y",alpha=.25);save("adaptive_attack_comparison.png")

plt.figure(figsize=(5.2,4.5));
for _,row in summary.iterrows():
    if row.method in {"one_step","SANDiff"} and row.strength!="strong":continue
    label=row.method+(f"-{row.strength}" if row.strength not in {"na","medium"} else "")
    plt.scatter(row.fixed_head_balanced_accuracy,row.retrained_head_balanced_accuracy,s=55,label=label)
plt.axline((.25,.25),slope=1,color="black",lw=1,ls="--");plt.xlabel("Fixed-head balanced accuracy");plt.ylabel("Retrained-head balanced accuracy");plt.legend(fontsize=8);plt.grid(alpha=.25);save("fixed_vs_retrained_utility.png")

participants=pd.read_csv(RESULT/"participant_effects.csv");selection=pd.read_csv(RESULT/"operating_point_selection.csv")
chosen=participants.merge(selection,left_on=["fold","method","seed","strength"],right_on=["fold","method","seed","selected_strength"])
chosen=chosen[chosen.method=="SANDiff"].groupby("participant").mean(numeric_only=True)
raw=participants[participants.method=="RAW"].set_index("participant")
effect=raw.adaptive_subject_attack_recall-chosen.adaptive_subject_attack_recall
plt.figure(figsize=(6.3,3.9));plt.bar([f"A{i:02d}" for i in effect.index],effect,color=np.where(effect>=0,"#4c78a8","#e45756"));plt.axhline(0,color="black",lw=1);plt.ylabel("Adaptive privacy gain vs RAW\n(positive = lower subject recall)");plt.grid(axis="y",alpha=.25);save("participant_privacy_effects.png")

latency=pd.read_csv(RESULT/"latency_summary.csv").groupby("method").median(numeric_only=True)
plt.figure(figsize=(4.8,3.8));plt.bar(latency.index,latency.median_ms,color=["#f58518","#4c78a8"]);plt.yscale("log");plt.ylabel("Batch-64 latency (ms, log scale)");plt.grid(axis="y",alpha=.25);save("method_latency.png")
