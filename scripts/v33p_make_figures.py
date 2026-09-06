#!/usr/bin/env python
"""Generate registered V33P figures and the V32P comparison table."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1];R33=ROOT/"results"/"sandiff_v33p";R32=ROOT/"results"/"sandiff_v32p";OUT=ROOT/"figures"/"sandiff_v33p";OUT.mkdir(parents=True,exist_ok=True)
s33=pd.read_csv(R33/"method_summary.csv");s32=pd.read_csv(R32/"method_summary.csv")


def save(name):plt.tight_layout();plt.savefig(OUT/name,dpi=180);plt.close()


rows=[]
for method,strength in (("RAW","na"),("LEACE","na"),("DANN","medium"),("one_step","strong"),("SANDiff","strong")):
    old=s32[(s32.method==method)&(s32.strength==strength)].iloc[0];new=s33[(s33.method==method)&(s33.strength==strength)].iloc[0]
    for metric in ("fixed_head_balanced_accuracy","retrained_head_balanced_accuracy","adaptive_subject_attack_balanced_accuracy","cross_session_same_different_auroc"):
        rows.append({"method":method,"strength":strength,"metric":metric,"v32p":old[metric],"v33p":new[metric],"v33p_minus_v32p":new[metric]-old[metric]})
pd.DataFrame(rows).to_csv(R33/"v32p_vs_v33p.csv",index=False)

plt.figure(figsize=(6.2,4.3))
for method,marker in (("one_step","o"),("SANDiff","s")):
    part=s33[s33.method==method].copy();part["order"]=part.strength.map({"weak":0,"medium":1,"strong":2});part=part.sort_values("order")
    plt.plot(1-part.adaptive_subject_attack_balanced_accuracy,part.fixed_head_balanced_accuracy,marker=marker,label=method)
    for _,row in part.iterrows():plt.annotate(row.strength,(1-row.adaptive_subject_attack_balanced_accuracy,row.fixed_head_balanced_accuracy),xytext=(4,4),textcoords="offset points",fontsize=8)
for method,marker in (("RAW","x"),("LEACE","^"),("DANN","v")):
    row=s33[s33.method==method].iloc[0];plt.scatter(1-row.adaptive_subject_attack_balanced_accuracy,row.fixed_head_balanced_accuracy,s=65,marker=marker,label=method)
plt.xlabel("Adaptive privacy utility (1 − subject BA)");plt.ylabel("Fixed-head MI balanced accuracy");plt.legend(fontsize=8);plt.grid(alpha=.25);save("privacy_utility_frontier.png")

methods=["RAW","LEACE","one_step","SANDiff"]
old=[];new=[]
for method in methods:
    strength="na" if method in {"RAW","LEACE"} else "strong";old.append(s32[(s32.method==method)&(s32.strength==strength)].fixed_head_balanced_accuracy.iloc[0]);new.append(s33[(s33.method==method)&(s33.strength==strength)].fixed_head_balanced_accuracy.iloc[0])
x=np.arange(len(methods));plt.figure(figsize=(6,4));plt.bar(x-.18,old,.36,label="V32P");plt.bar(x+.18,new,.36,label="V33P");plt.xticks(x,methods);plt.ylabel("Fixed-head MI balanced accuracy");plt.legend();plt.grid(axis="y",alpha=.25);save("v32p_vs_v33p.png")

primary=s33[((s33.method.isin(["one_step","SANDiff"]))&(s33.strength=="strong"))|s33.method.isin(["RAW","LEACE","DANN"])]
plt.figure(figsize=(6.7,4));plt.bar(primary.method,primary.adaptive_subject_attack_balanced_accuracy,color=["#777777","#4c78a8","#f58518","#72b7b2","#e45756"]);plt.axhline(1/3,color="black",ls="--",lw=1);plt.ylabel("Adaptive subject attack balanced accuracy");plt.xticks(rotation=20);plt.grid(axis="y",alpha=.25);save("adaptive_attack_comparison.png")

part=pd.read_csv(R33/"participant_effects.csv").groupby(["method","strength","participant"],as_index=False).mean(numeric_only=True);raw=part[(part.method=="RAW")].set_index("participant");sand=part[(part.method=="SANDiff")&(part.strength=="strong")].set_index("participant");effect=raw.adaptive_subject_attack_recall-sand.adaptive_subject_attack_recall
plt.figure(figsize=(6.3,3.8));plt.bar([f"A{i:02d}" for i in effect.index],effect,color=np.where(effect>=0,"#4c78a8","#e45756"));plt.axhline(0,color="black",lw=1);plt.ylabel("SANDiff privacy utility vs RAW\n(positive = lower subject recall)");plt.grid(axis="y",alpha=.25);save("participant_effects.png")

lat=pd.read_csv(R33/"latency_summary.csv").groupby(["method","batch_size"],as_index=False).median(numeric_only=True);plt.figure(figsize=(5.3,3.8))
for method in ("one_step","SANDiff"):
    subset=lat[lat.method==method];plt.plot(subset.batch_size,subset.median_ms,marker="o",label=method)
plt.yscale("log");plt.xticks([1,64]);plt.xlabel("Batch size");plt.ylabel("Latency (ms, log scale)");plt.legend();plt.grid(alpha=.25);save("latency_comparison.png")
