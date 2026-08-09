#!/usr/bin/env python3
"""Rebuild frozen TAAS evidence and submission figures from committed summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.signal import welch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/cgdr/bci2b_subject_diffusion_replication"
OUT = ROOT / "results/cgdr/taas_subject_diffusion_freeze"
FIG = ROOT / "taas_submission/figures"
SERVER = Path("/home/infres/yinwang/denoiseNet_subject_diffusion_replication/results/cgdr/bci2b_subject_diffusion_replication")
V11 = Path("/home/infres/yinwang/denoiseNet_bci2b_eog_residual_v11/results/cgdr/bci2b_eog_residual_v11")


def rows(name: str) -> list[dict[str, str]]:
    with (SRC / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def validate() -> dict:
    result = json.loads((SRC / "result_summary.json").read_text())
    expected = {
        "U_P": 0.03472997751776819,
        "U_W": 0.0483602481537763,
        "eog": 0.10430522797781798,
        "preservation": 0.7982431411007305,
    }
    actual = {
        "U_P": result["effects"]["U_P"]["mean"],
        "U_W": result["effects"]["U_W"]["mean"],
        "eog": result["safety"]["eog_attenuation"],
        "preservation": result["safety"]["preservation"],
    }
    for key, value in expected.items():
        if not np.isclose(actual[key], value, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"frozen {key} changed: {actual[key]} != {value}")
    methods = rows("method_summary.csv")
    means: dict[str, float] = {}
    for method in ("DIFF-MATCH", "DET-MATCH", "LINEAR-MATCH"):
        values = [float(r["rrmse"]) for r in methods if r["method"] == method]
        means[method] = float(np.mean(values))
    expected_methods = {"DIFF-MATCH": 0.11350142567154066,
                        "DET-MATCH": 0.11225142642665012,
                        "LINEAR-MATCH": 0.08759881297333373}
    for key, value in expected_methods.items():
        if not np.isclose(means[key], value, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"frozen method mean changed for {key}")
    return {"result": result, "means": means}


def evidence_manifest(valid: dict) -> None:
    result = valid["result"]
    evidence = [
        ("scientific_unit", "participant", "n=9", "result_summary.json"),
        ("development_scope", "three-seed stability", "same cohort; not confirmation", "result_summary.json"),
        ("U_P", "mean RRMSE effect", f"{result['effects']['U_P']['mean']:.8f}", "result_summary.json"),
        ("U_W", "mean RRMSE effect", f"{result['effects']['U_W']['mean']:.8f}", "result_summary.json"),
        ("U_P_signs", "participant positive count", "9/9", "participant_averaged_effects.csv"),
        ("U_W_signs", "participant positive count", "9/9", "participant_averaged_effects.csv"),
        ("sign_flip_one_sided", "exact participant sign flip", "0.001953", "participant_averaged_effects.csv"),
        ("sign_flip_two_sided", "exact sensitivity", "0.003906", "participant_averaged_effects.csv"),
        ("diff_match_rrmse", "three-seed mean", f"{valid['means']['DIFF-MATCH']:.5f}", "method_summary.csv"),
        ("det_match_rrmse", "three-seed mean", f"{valid['means']['DET-MATCH']:.5f}", "method_summary.csv"),
        ("linear_match_rrmse", "three-seed mean", f"{valid['means']['LINEAR-MATCH']:.5f}", "method_summary.csv"),
        ("eog_attenuation", "natural mean", f"{result['safety']['eog_attenuation']:.4f}", "result_summary.json"),
        ("preservation", "natural mean", f"{result['safety']['preservation']:.4f}", "result_summary.json"),
        ("preservation_reversal", "participant exceptions", "1/9", "participant_averaged_safety.csv"),
        ("covariance_reversal", "participant exceptions", "2/9", "participant_averaged_safety.csv"),
        ("eog_reversal", "participant exceptions", "0/9", "participant_averaged_safety.csv"),
        ("kappa_reversal", "participant exceptions", "0/9", "participant_averaged_safety.csv"),
        ("population_training", "frozen split", "7 subjects; recipient+cyclic donor held out", "wrong_donor_audit.csv"),
        ("primary_wrong", "specificity control", "one cyclic unseen donor per fold", "wrong_donor_audit.csv"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "evidence_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("claim_id", "quantity", "frozen_value", "source"))
        writer.writerows(evidence)


def method_protocol() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    def box(x: float, y: float, w: float, text: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, .72, boxstyle="round,pad=.04",
                                    facecolor=color, edgecolor="#34495e", linewidth=1.1))
        ax.text(x+w/2, y+.36, text, ha="center", va="center", fontsize=9)
    def arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=12,
                                     linewidth=1.1,color="#34495e"))
        if label: ax.text((x1+x2)/2,(y1+y2)/2+.12,label,ha="center",fontsize=8)
    box(.15,1.85,1.8,"Early support\nEEG + EOG","#d6eaf8")
    box(2.55,1.85,1.8,"Ridge transfer\n$H_s$","#fdebd0")
    arrow(1.95,2.21,2.55,2.21)
    box(.15,.65,1.8,"Later query\nEEG $y$ + EOG $e$","#d6eaf8")
    box(2.55,.65,1.8,"Linear anchor\n$a_0=H_s e$","#fdebd0")
    box(5.0,.65,1.8,"Deterministic\nresidual $r_{det}$","#e8daef")
    box(7.45,.65,1.8,"Residual diffusion\nDDIM25, $K=8$","#d5f5e3")
    box(9.9,.65,1.8,"Subtract correction\n" + r"$\hat{x}=y-\gamma\hat{a}$","#fcf3cf")
    arrow(1.95,1.01,2.55,1.01); arrow(4.35,1.01,5.0,1.01)
    arrow(6.8,1.01,7.45,1.01); arrow(9.25,1.01,9.9,1.01)
    arrow(3.45,1.85,3.45,1.37,"subject context")
    box(7.45,2.0,2.8,"Evaluator only\nclean target / labels / outcomes","#f5b7b1")
    arrow(10.25,2.0,10.95,1.38,"score only")
    ax.text(.15,.18,"Inference never reads the evaluator-only fields. Query EOG is a deployment input.",
            fontsize=8.5, color="#5d6d7e")
    save(fig,"method_protocol")


def participant_figures() -> None:
    averaged = rows("participant_averaged_effects.csv")
    subjects = np.array([int(r["subject"]) for r in averaged])
    up = np.array([float(r["U_P"]) for r in averaged]); uw = np.array([float(r["U_W"]) for r in averaged])
    fig, axes = plt.subplots(1,2,figsize=(9.2,3.7),sharey=True)
    for ax, value, label, color in zip(axes,(up,uw),(r"$U_P$: POP $-$ MATCH",r"$U_W$: WRONG $-$ MATCH"),("#2874a6","#b03a2e")):
        ax.axvline(0,color="black",lw=.8); ax.scatter(value,subjects,color=color,s=36,zorder=3)
        for y,x in zip(subjects,value): ax.plot([0,x],[y,y],color=color,alpha=.45,lw=1)
        ax.axvline(value.mean(),color=color,ls="--",lw=1.3,label=f"mean={value.mean():.4f}")
        ax.set_xlabel("RRMSE utility (positive favors MATCH)"); ax.set_title(label); ax.legend(frameon=False,fontsize=8)
        ax.grid(axis="x",alpha=.18)
    axes[0].set_yticks(subjects,[f"P{s}" for s in subjects]); axes[0].set_ylabel("Participant")
    fig.tight_layout(); save(fig,"participant_effects")

    seed = rows("participant_seed_effects.csv")
    seeds = sorted({int(r["seed"]) for r in seed})
    fig, axes = plt.subplots(1,2,figsize=(9.3,3.7),sharey=True)
    for ax,key,title in zip(axes,("U_P","U_W"),(r"$U_P$",r"$U_W$")):
        mat=np.array([[float(next(r[key] for r in seed if int(r["seed"])==s and int(r["subject"])==p)) for s in seeds] for p in subjects])
        im=ax.imshow(mat,aspect="auto",cmap="YlGn",vmin=0)
        ax.set_xticks(range(3),[str(s) for s in seeds],rotation=20); ax.set_title(title); ax.set_xlabel("Training seed")
        fig.colorbar(im,ax=ax,shrink=.8,label="RRMSE utility")
    axes[0].set_yticks(range(9),[f"P{s}" for s in subjects]); axes[0].set_ylabel("Participant")
    fig.tight_layout(); save(fig,"seed_effect_heatmap")


def safety_figure() -> None:
    match = rows("participant_averaged_safety.csv")
    x=np.array([float(r["eog_attenuation"]) for r in match]); y=np.array([float(r["preservation"]) for r in match])
    ids=[int(r["subject"]) for r in match]
    fig,ax=plt.subplots(figsize=(5.2,4.1)); ax.axvline(0,color="black",lw=.8,ls="--"); ax.axhline(.78,color="#a93226",lw=1,ls="--")
    ax.scatter(x,y,c="#2874a6",s=42)
    for a,b,s in zip(x,y,ids): ax.annotate(f"P{s}",(a,b),xytext=(4,3),textcoords="offset points",fontsize=8)
    ax.set_xlabel("EOG attenuation (higher is better)"); ax.set_ylabel("Preservation (higher is better)"); ax.grid(alpha=.16)
    fig.tight_layout(); save(fig,"attenuation_preservation")


def waveform_figure() -> None:
    # Frozen before inspection: seed 20260808, participant 1/fold 0,
    # same_01, first paired window, first EEG channel.
    fold=SERVER/"seeds/20260808/folds/fold_00"; protocol="same_01"
    with np.load(fold/"units"/protocol/"inference.npz") as inf:
        scale=np.asarray(inf["eeg_scale"]); location=np.asarray(inf["eeg_location"])
    with np.load(V11/"folds/fold_00/units"/protocol/"evaluator.npz") as ev:
        target=np.asarray(ev["paired_x"])[0,0,:500]
    with np.load(fold/"outputs/k8"/protocol/"inference_outputs.npz") as out:
        names=("RAW","LINEAR-MATCH","DIFF-POP","DIFF-MATCH","DIFF-WRONG")
        signals={n:(np.asarray(out[f"paired_{n}"])*scale[None,:,None]+location[None,:,None])[0,0,:500] for n in names}
    time=np.arange(500)/250.0
    fig,axes=plt.subplots(2,1,figsize=(9.0,5.8))
    colors={"RAW":"#7f8c8d","LINEAR-MATCH":"#1f618d","DIFF-POP":"#ca6f1e","DIFF-MATCH":"#117864","DIFF-WRONG":"#922b21"}
    axes[0].plot(time,target,color="black",lw=1.5,label="clean target")
    for n in names: axes[0].plot(time,signals[n],lw=.85,alpha=.82,color=colors[n],label=n)
    axes[0].set_ylabel("EEG (physical units)"); axes[0].legend(ncol=3,fontsize=7,frameon=False); axes[0].set_xlim(0,time[-1])
    f,p=welch(target,fs=250,nperseg=250); mask=(f>=1)&(f<=45); axes[1].semilogy(f[mask],p[mask],color="black",lw=1.5,label="clean target")
    for n in names:
        f,p=welch(signals[n],fs=250,nperseg=250); axes[1].semilogy(f[mask],p[mask],lw=.85,alpha=.82,color=colors[n],label=n)
    axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("PSD"); axes[1].grid(alpha=.15)
    fig.tight_layout(); save(fig,"waveform_psd")


def main() -> None:
    valid=validate(); evidence_manifest(valid); method_protocol(); participant_figures(); safety_figure(); waveform_figure()
    result=valid["result"]
    frozen={
        "scientific_status":"MATCHED_SUBJECT_OPERATOR_EFFECT_IS_THREE_SEED_STABLE_WITHIN_THE_FIXED_EOG_GUIDED_DIFFUSION_PIPELINE",
        "evidence_scope":"three-seed stability on the same nine-participant development cohort",
        "confirmation":False,
        "deployment":"EOG-guided",
        "scientific_unit":"participant",
        "n":9,
        "participant_seed_rows_are_independent_n":False,
        "effects":{"U_P":result["effects"]["U_P"],"U_W":result["effects"]["U_W"]},
        "exact_sign_flip":{"one_sided":0.001953125,"two_sided_sensitivity":0.00390625},
        "participant_bootstrap":"descriptive",
        "method_rrmse":valid["means"],
        "diffusion_over_deterministic":"not_supported",
        "diffusion_over_linear":"not_supported",
        "safety":{"wording":"Mean natural-signal proxy criteria were met, with disclosed participant-level exceptions.",
                  "eog_attenuation":result["safety"]["eog_attenuation"],
                  "preservation":result["safety"]["preservation"],
                  "participant_reversals":result["safety"]["participant_reversals"]},
        "population_control":"frozen twoheldout seven-subject population training",
        "primary_specificity":"one frozen cyclic unseen-WRONG donor per fold",
        "seen_donors":"sensitivity analysis, not unseen-donor replication",
        "scientific_sources_unchanged":True,
    }
    (OUT/"result_summary.json").write_text(json.dumps(frozen,indent=2)+"\n")
    (OUT/"build_summary.json").write_text(json.dumps({"status":"TAAS_FREEZE_ASSETS_BUILT","scientific_sources_unchanged":True,"figures":5},indent=2)+"\n")


if __name__ == "__main__":
    main()
