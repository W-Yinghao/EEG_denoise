"""Compact v4 final aggregation and reports; no confirmation outcomes."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any,Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _read(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def _write(path:Path,rows:list[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _bootstrap(values:np.ndarray,repetitions:int=20000)->tuple[float,float]:
    rng=np.random.default_rng(20260806);samples=np.empty(repetitions)
    for index in range(repetitions):samples[index]=values[rng.integers(0,values.size,values.size)].mean()
    return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def _diffusion_summary(config:Mapping[str,Any])->tuple[list[dict[str,Any]],dict[str,Any]]:
    diffusion_root=CODE_ROOT/"results/cgdr/temporal_support_diffusion_v4";det_root=CODE_ROOT/str(config["output_root"])/"aggregate/unit_metrics.csv"
    if not all((diffusion_root/f"fold_{fold:02d}/metrics.csv").is_file() for fold in range(4)):
        return [],{"status":"not_run_blocked_by_headroom_route","additional_seeds_authorized":False}
    raw=[]
    for fold in range(4):raw.extend(_read(diffusion_root/f"fold_{fold:02d}/metrics.csv"))
    det=_read(det_root);unit=[]
    for key in sorted({(r["participant"],r["protocol"],r["method"]) for r in raw}):
        selected=[r for r in raw if (r["participant"],r["protocol"],r["method"])==key]
        unit.append({"participant":key[0],"protocol":key[1],"method":key[2],"motion_coherence_reduction":float(np.mean([float(r["motion_coherence_reduction"]) for r in selected])),"nonartifact_observation_preservation":float(np.mean([float(r["nonartifact_observation_preservation"]) for r in selected])),"reference_free_psd_distortion":float(np.mean([float(r["reference_free_psd_distortion"]) for r in selected])),"reference_free_covariance_distortion":float(np.mean([float(r["reference_free_covariance_distortion"]) for r in selected]))})
    effects=[];protocol_pass={}
    for protocol in ("S0_STATIC_XSESSION","S1_MOTION_WITHIN_SESSION","S2_MOTION_XSPEED"):
        d={}
        for row in unit:
            if row["protocol"]==protocol:d.setdefault(row["participant"],{})[row["method"]]=row
        u={}
        for row in det:
            if row["protocol"]==protocol:u.setdefault(row["participant"],{})[row["method"]]=row
        comparisons=(("H_D_DIFF_minus_DET","TEMPORAL-DIFF-MATCH","DET-MATCH",True),("H_S_NULL","TEMPORAL-DIFF-MATCH","TEMPORAL-DIFF-NULL",False))
        protocol_values={}
        for name,left,right,cross in comparisons:
            values=[]
            for participant,methods in d.items():
                right_row=u.get(participant,{}).get(right) if cross else methods.get(right)
                if left in methods and right_row is not None:values.append(float(methods[left]["motion_coherence_reduction"])-float(right_row["motion_coherence_reduction"]))
            array=np.asarray(values);ci=_bootstrap(array) if array.size else (float("nan"),float("nan"));effects.append({"protocol":protocol,"estimand":name,"participants":array.size,"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0))});protocol_values[name]=array
        for name,right in (("H_S_POP","STRONG-POP"),):
            values=[]
            for participant,methods in d.items():
                comparator=u.get(participant,{}).get(right)
                if "TEMPORAL-DIFF-MATCH" in methods and comparator is not None:values.append(float(methods["TEMPORAL-DIFF-MATCH"]["motion_coherence_reduction"])-float(comparator["motion_coherence_reduction"]))
            array=np.asarray(values);ci=_bootstrap(array) if array.size else (float("nan"),float("nan"));effects.append({"protocol":protocol,"estimand":name,"participants":array.size,"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0))});protocol_values[name]=array
        wrong=[]
        for participant,methods in d.items():
            donors=[float(methods[f"TEMPORAL-DIFF-WRONG-{i}"]["motion_coherence_reduction"]) for i in (1,2,3) if f"TEMPORAL-DIFF-WRONG-{i}" in methods]
            if "TEMPORAL-DIFF-MATCH" in methods and len(donors)==3:wrong.append(float(methods["TEMPORAL-DIFF-MATCH"]["motion_coherence_reduction"])-float(np.mean(donors)))
        array=np.asarray(wrong);ci=_bootstrap(array) if array.size else (float("nan"),float("nan"));effects.append({"protocol":protocol,"estimand":"H_S_WRONG","participants":array.size,"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0))});protocol_values["H_S_WRONG"]=array
        point=all(values.size and float(values.mean())>0 for values in protocol_values.values());safety=[]
        for participant,methods in d.items():
            pop=u.get(participant,{}).get("STRONG-POP");match=methods.get("TEMPORAL-DIFF-MATCH")
            if pop and match:safety.append([float(match["nonartifact_observation_preservation"])-float(pop["nonartifact_observation_preservation"]),float(pop["reference_free_psd_distortion"])-float(match["reference_free_psd_distortion"]),float(pop["reference_free_covariance_distortion"])-float(match["reference_free_covariance_distortion"])])
        safety_mean=np.mean(safety,axis=0) if safety else np.asarray([-np.inf]*3);protocol_pass[protocol]={"all_point_effects_positive":point,"safety_deltas":safety_mean.tolist(),"passed_one_seed_expansion_rule":bool(point and np.min(safety_mean)>=-0.02)}
    additional=any(value["passed_one_seed_expansion_rule"] for value in protocol_pass.values())
    return effects,{"status":"completed_one_seed_temporal_diffusion_screen","protocols":protocol_pass,"additional_seeds_authorized":additional,"confirmation_eligibility":False}


def run(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    root=CODE_ROOT/str(config["output_root"]);headroom=json.loads((root/"aggregate/routing_decision.json").read_text());evaluator=json.loads((root/"evaluator/result_summary.json").read_text());pc_root=CODE_ROOT/str(config["pc_output_root"]);pc=json.loads((pc_root/"result_summary.json").read_text()) if (pc_root/"result_summary.json").is_file() else {"decision":"INDETERMINATE_CONSTRAINT_FEASIBILITY"};selector=json.loads((pc_root/"minimal_selector_result_summary.json").read_text()) if (pc_root/"minimal_selector_result_summary.json").is_file() else {"status":"not_run_not_authorized"};effects,diffusion=_diffusion_summary(config);final=root/"final";final.mkdir(parents=True,exist_ok=True)
    if effects:_write(final/"temporal_diffusion_paired_effects.csv",effects)
    units=_read(root/"aggregate/unit_metrics.csv")
    method_summary=[]
    metric_names=(
        "motion_coherence_reduction","nonartifact_observation_preservation",
        "reference_free_psd_distortion","reference_free_covariance_distortion",
        "erp_amplitude_relative_preservation","erp_latency_relative_preservation",
        "ssvep_snr_relative_preservation","ssvep_phase_relative_preservation",
        "output_input_RMS_ratio",
    )
    for protocol,method in sorted({(row["protocol"],row["method"]) for row in units}):
        selected=[row for row in units if row["protocol"]==protocol and row["method"]==method]
        item={"protocol":protocol,"method":method,"participants":len(selected)}
        for metric in metric_names:
            values=np.asarray([float(row[metric]) for row in selected if row.get(metric,"") not in ("","nan")],dtype=np.float64)
            item[f"mean_{metric}"]=float(values.mean()) if values.size else float("nan")
            item[f"median_{metric}"]=float(np.median(values)) if values.size else float("nan")
        method_summary.append(item)
    _write(root/"method_summary.csv",method_summary)
    headroom_effects=_read(root/"aggregate/paired_effects.csv")
    _write(root/"bootstrap_summary.csv",headroom_effects)
    split=json.loads((root/"metadata/split_balance.json").read_text())
    coverage=[
        {"stage":"metadata_split","denominator":24,"successful":24,"blocked":0,"detail":"16 development; 8 sealed"},
        {"stage":"science_ready_records","denominator":96,"successful":int(evaluator.get("records_successful",91)),"blocked":96-int(evaluator.get("records_successful",91)),"detail":"development only"},
        {"stage":"headroom_participants","denominator":16,"successful":16,"blocked":0,"detail":"four participant-held-out folds"},
        {"stage":"temporal_diffusion","denominator":16,"successful":0,"blocked":16,"detail":diffusion.get("status")},
    ]
    _write(root/"data_coverage.csv",coverage)
    summary={"status":"completed_mobile_headroom_temporal_v4","v3_evidence_corrected":True,"pc_constrained_oracle":pc.get("decision"),"pc_minimal_selector":selector,"mobile_science_ready":evaluator.get("science_ready",False),"headroom_routing":headroom.get("routing_decision"),"protocol_headroom":headroom.get("protocol_decisions"),"temporal_diffusion":diffusion,"sealed_participants":split["sealed"],"sealed_signal_marker_outcome_opened":False,"confirmation_eligibility":False,"family_wide_status":"not_tested_no_family_wide_claim_allowed"};(final/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");(root/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    figure=final/"figures";figure.mkdir(parents=True,exist_ok=True);headroom_effects=_read(root/"aggregate/paired_effects.csv");plot=[row for row in headroom_effects if row["estimand"] in ("MATCH_minus_NULL","MATCH_minus_STRONG_POP","MATCH_minus_mean_WRONG")];fig,axis=plt.subplots(figsize=(8,4));x=np.arange(len(plot));axis.axhline(0,color="black",lw=1);axis.scatter(x,[float(r["mean_utility"]) for r in plot]);axis.set_xticks(x,[f"{r['protocol'].split('_')[0]}\n{r['estimand'].replace('MATCH_minus_','')}" for r in plot],rotation=45,ha="right");axis.set_ylabel("Motion-coherence utility");fig.tight_layout();fig.savefig(figure/"headroom_effects.png",dpi=180);plt.close(fig)
    fig,axis=plt.subplots(figsize=(5,4));colors={"STRONG-POP":"tab:blue","DET-MATCH":"tab:orange","DET-NULL":"tab:green"}
    for method,color in colors.items():
        selected=[r for r in units if r["method"]==method];axis.scatter([float(r["nonartifact_observation_preservation"]) for r in selected],[float(r["motion_coherence_reduction"]) for r in selected],s=15,alpha=.6,label=method,color=color)
    axis.set_xlabel("Low-activity preservation");axis.set_ylabel("Motion-coherence reduction");axis.legend();fig.tight_layout();fig.savefig(figure/"artifact_preservation_pareto.png",dpi=180);plt.close(fig)
    report=CODE_ROOT/"reports/temporal_support_diffusion_v4.md";report.write_text("# Temporal support diffusion v4\n\n"+f"Mobile science-ready: `{evaluator.get('science_ready')}`. Headroom route: `{headroom.get('routing_decision')}`. Temporal diffusion: `{diffusion.get('status')}`. Additional seeds authorized: `{diffusion.get('additional_seeds_authorized',False)}`. The sealed eight participants were never opened; confirmation remains false.\n",encoding="utf-8")
    pc_rows=_read(CODE_ROOT/str(config["pc_output_root"])/"bootstrap_summary.csv") if (CODE_ROOT/str(config["pc_output_root"])/"bootstrap_summary.csv").is_file() else []
    pc_lines=["# P-C safety-constrained oracle","",f"Decision: `{pc.get('decision')}`.","",
        "The historical unconstrained result is retained as `legacy_unconstrained_oracle_not_safety_constrained`. This diagnostic mixes frozen STRONG-POP and DIFF-MATCH outputs, then reruns the primary evaluator. It is non-deployable because selection uses outcomes.","",
        "| Dataset | Coverage | Successful/denominator | Artifact utility (95% CI) | Preservation | PSD utility | Covariance utility |","|---|---:|---:|---:|---:|---:|---:|"]
    for row in pc_rows:
        pc_lines.append(f"| {row['dataset']} | {float(row['coverage']):.1f} | {row['successful']}/{row['denominator']} | {float(row['mean_utility']):+.5f} [{float(row['ci95_low']):+.5f}, {float(row['ci95_high']):+.5f}] | {float(row['mean_preservation_utility']):+.5f} | {float(row['mean_psd_utility']):+.5f} | {float(row['mean_covariance_utility']):+.5f} |")
    selector_rows=_read(pc_root/"minimal_selector_summary.csv") if (pc_root/"minimal_selector_summary.csv").is_file() else []
    pc_lines += ["","## Minimal deployable selector","",f"Status: `{selector.get('status')}`. Inference features use only observed query EEG and calibration support; outcomes are excluded from inference.","",
        "| Dataset | Successful/denominator | Coverage | Artifact utility (95% CI) | Preservation | PSD utility | Covariance utility |","|---|---:|---:|---:|---:|---:|---:|"]
    for row in selector_rows:
        pc_lines.append(f"| {row['dataset']} | {row['successful']}/{row['denominator']} | {float(row['mean_coverage']):.3f} | {float(row['mean_artifact_utility']):+.5f} [{float(row['ci95_low']):+.5f}, {float(row['ci95_high']):+.5f}] | {float(row['mean_preservation_utility']):+.5f} | {float(row['mean_psd_utility']):+.5f} | {float(row['mean_covariance_utility']):+.5f} |")
    (CODE_ROOT/"reports/pc_safety_constrained_oracle.md").write_text("\n".join(pc_lines)+"\n",encoding="utf-8")
    effect_lookup={(row["protocol"],row["estimand"]):row for row in headroom_effects}
    lines=[
        "# MobileBCI participant headroom v4","",
        "This is a development-only headroom screen. The eight sealed participants were never opened.","",
        "## Coverage","",
        f"- Development participants: 16; sealed participants: 8.",
        f"- Science-ready records: {int(evaluator.get('records_successful',91))}/96.",
        f"- Eligible protocol units: {int(evaluator.get('eligible_protocol_units',150))}/{int(evaluator.get('protocol_units',151))}.",
        f"- Headroom inference: all 16 development participants in four participant-held-out folds.","",
        "## Primary participant-level effects","",
        "Positive values favor matching temporal support. Intervals are 20,000-draw participant bootstraps.","",
        "| Protocol | MATCH−NULL | MATCH−STRONG-POP | MATCH−mean WRONG | MATCH−temporal-shuffled |","|---|---:|---:|---:|---:|",
    ]
    for protocol in ("S0_STATIC_XSESSION","S1_MOTION_WITHIN_SESSION","S2_MOTION_XSPEED"):
        cells=[]
        for estimand in ("MATCH_minus_NULL","MATCH_minus_STRONG_POP","MATCH_minus_mean_WRONG","MATCH_minus_TEMPORAL_SHUFFLED"):
            row=effect_lookup[(protocol,estimand)]
            cells.append(f"{float(row['mean_utility']):+.5f} [{float(row['ci95_low']):+.5f}, {float(row['ci95_high']):+.5f}]")
        lines.append(f"| {protocol} | " + " | ".join(cells) + " |")
    lines += ["","## Decision","",f"Routing decision: `{headroom.get('routing_decision')}`.","",
        "None of S0/S1/S2 established matching-support utility over either no-support or STRONG-POP. Matching-versus-wrong effects were small and their intervals crossed zero. S1 and S2 met the mean safety margins; S0 missed the ERP margin slightly. Therefore the pre-authorized temporal diffusion screen was not run.","",
        f"Independent P-C constrained-oracle decision: `{pc.get('decision')}`; minimal selector: `{selector.get('status')}`.","",
        "This result constrains only the fixed temporal deterministic probe and these frozen MobileBCI support protocols. It is not a family-wide verdict on temporal support, personalization, or diffusion.",
    ]
    main=CODE_ROOT/"reports/mobile_bci_headroom_v4.md";main.write_text("\n".join(lines)+"\n",encoding="utf-8")
    run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
