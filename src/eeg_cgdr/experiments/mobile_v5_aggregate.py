"""Participant-level aggregation and pre-specified v5 routing."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any,Mapping

import numpy as np


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_diffusion_v5"))


def _read(path:Path)->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as stream:return list(csv.DictReader(stream))


def _write(path:Path,rows:list[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _bootstrap(values:np.ndarray,repetitions:int,seed:int)->tuple[float,float]:
    if not values.size:return float("nan"),float("nan")
    rng=np.random.default_rng(seed);draw=np.asarray([values[rng.integers(0,values.size,values.size)].mean() for _ in range(repetitions)]);return float(np.quantile(draw,.025)),float(np.quantile(draw,.975))


def _mean_rows(rows:list[dict[str,str]],keys:tuple[str,...],metrics:tuple[str,...])->list[dict[str,Any]]:
    result=[]
    for key in sorted({tuple(row[name] for name in keys) for row in rows}):
        selected=[row for row in rows if tuple(row[name] for name in keys)==key];item=dict(zip(keys,key));item["record_rows"]=len(selected)
        for metric in metrics:
            values=[]
            for row in selected:
                try:value=float(row[metric])
                except (KeyError,ValueError):continue
                if np.isfinite(value):values.append(value)
            item[metric]=float(np.mean(values)) if values else float("nan")
        result.append(item)
    return result


def aggregate(config:Mapping[str,Any],run_dir:Path,seeds:list[int])->Mapping[str,Any]:
    root=CODE_ROOT/str(config["output_root"]);raw=[];pareto=[]
    for seed in seeds:
        for fold in range(4):
            raw.extend(_read(root/f"factorial/seed_{seed}/fold_{fold:02d}/metrics.csv"));pareto.extend(_read(root/f"factorial/seed_{seed}/fold_{fold:02d}/pareto_metrics.csv"))
    successful=[row for row in raw if row["status"]=="success"]
    protocol_rows=_read(root/"protocol/frozen_protocol_units.csv");coverage=[]
    for protocol in ("S0_STATIC_XSESSION","S1_MOTION_WITHIN_SESSION","S2_MOTION_XSPEED"):
        for task in ("ERP","SSVEP"):
            selected=[row for row in protocol_rows if row["protocol"]==protocol and row["task"]==task]
            coverage.append({"protocol":protocol,"task":task,"denominator":len(selected),"eligible":sum(row["status"]=="eligible" for row in selected),"blocked_no_later_block":sum(row["status"]=="blocked_no_later_block" for row in selected),"missing":sum(row["status"]=="missing" for row in selected),"sealed_signal_opened":False})
    metrics=("motion_coherence_reduction","nonartifact_observation_preservation","reference_free_psd_distortion","reference_free_covariance_distortion","output_input_RMS_ratio","observation_change_ratio","erp_amplitude_relative_preservation","erp_latency_relative_preservation","ssvep_snr_relative_preservation","ssvep_phase_relative_preservation")
    # Average task/session and optimization seeds inside participant first.
    units=_mean_rows(successful,("participant","protocol","method"),metrics);output=root/"aggregate";_write(output/"unit_metrics.csv",units);_write(output/"data_coverage.csv",coverage)
    effects=[];decisions={};repetitions=int(config["evaluation"]["bootstrap_repetitions"]);margin=float(config["evaluation"]["noninferiority_margin"])
    comparisons=(("H_D_DIFF_MATCH_minus_DET_MATCH","DIFF-MATCH","DET-MATCH"),("H_S_NULL","DIFF-MATCH","DIFF-NULL"),("H_S_POP","DIFF-MATCH","DIFF-POP"),("H_S_SHUFFLED","DIFF-MATCH","DIFF-SHUFFLED"))
    for protocol in ("S0_STATIC_XSESSION","S1_MOTION_WITHIN_SESSION","S2_MOTION_XSPEED"):
        by={}
        for row in units:
            if row["protocol"]==protocol:by.setdefault(row["participant"],{})[row["method"]]=row
        values_by={}
        for name,left,right in comparisons:
            values=[(participant,float(methods[left]["motion_coherence_reduction"])-float(methods[right]["motion_coherence_reduction"])) for participant,methods in by.items() if left in methods and right in methods];array=np.asarray([value for _,value in values]);ci=_bootstrap(array,repetitions,20260806+len(effects));effects.append({"protocol":protocol,"estimand":name,"participants":len(values),"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0)),"participant_effects_json":json.dumps(dict(values),sort_keys=True)});values_by[name]=array
        wrong=[]
        for participant,methods in by.items():
            donors=[float(methods[f"DIFF-WRONG-{index}"]["motion_coherence_reduction"]) for index in (1,2,3) if f"DIFF-WRONG-{index}" in methods]
            if "DIFF-MATCH" in methods and len(donors)==3:wrong.append((participant,float(methods["DIFF-MATCH"]["motion_coherence_reduction"])-float(np.mean(donors))))
        array=np.asarray([value for _,value in wrong]);ci=_bootstrap(array,repetitions,20260877);effects.append({"protocol":protocol,"estimand":"H_S_WRONG","participants":len(wrong),"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0)),"participant_effects_json":json.dumps(dict(wrong),sort_keys=True)});values_by["H_S_WRONG"]=array
        interaction=[]
        for participant,methods in by.items():
            if all(name in methods for name in ("DIFF-MATCH","DIFF-NULL","DET-MATCH","DET-NULL")):interaction.append((participant,(float(methods["DIFF-MATCH"]["motion_coherence_reduction"])-float(methods["DIFF-NULL"]["motion_coherence_reduction"]))-(float(methods["DET-MATCH"]["motion_coherence_reduction"])-float(methods["DET-NULL"]["motion_coherence_reduction"]))))
        array=np.asarray([value for _,value in interaction]);ci=_bootstrap(array,repetitions,20260888);effects.append({"protocol":protocol,"estimand":"DIFFUSION_x_SUPPORT_INTERACTION","participants":len(interaction),"mean_utility":float(array.mean()) if array.size else float("nan"),"median_utility":float(np.median(array)) if array.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(array>0)),"participant_effects_json":json.dumps(dict(interaction),sort_keys=True)})
        safety=[]
        for methods in by.values():
            if "DIFF-MATCH" not in methods or "POP" not in methods:continue
            left=methods["DIFF-MATCH"];right=methods["POP"]
            safety.append([float(left["nonartifact_observation_preservation"])-float(right["nonartifact_observation_preservation"]),float(right["reference_free_psd_distortion"])-float(left["reference_free_psd_distortion"]),float(right["reference_free_covariance_distortion"])-float(left["reference_free_covariance_distortion"]),float(left["erp_amplitude_relative_preservation"])-float(right["erp_amplitude_relative_preservation"]) if np.isfinite(float(left["erp_amplitude_relative_preservation"])) and np.isfinite(float(right["erp_amplitude_relative_preservation"])) else np.nan,float(left["erp_latency_relative_preservation"])-float(right["erp_latency_relative_preservation"]) if np.isfinite(float(left["erp_latency_relative_preservation"])) and np.isfinite(float(right["erp_latency_relative_preservation"])) else np.nan,float(left["ssvep_snr_relative_preservation"])-float(right["ssvep_snr_relative_preservation"]) if np.isfinite(float(left["ssvep_snr_relative_preservation"])) and np.isfinite(float(right["ssvep_snr_relative_preservation"])) else np.nan,float(left["ssvep_phase_relative_preservation"])-float(right["ssvep_phase_relative_preservation"]) if np.isfinite(float(left["ssvep_phase_relative_preservation"])) and np.isfinite(float(right["ssvep_phase_relative_preservation"])) else np.nan,abs(float(right["output_input_RMS_ratio"])-1)-abs(float(left["output_input_RMS_ratio"])-1),float(right["observation_change_ratio"])-float(left["observation_change_ratio"])])
        safety_mean=np.nanmean(np.asarray(safety),axis=0) if safety else np.full(9,np.nan);safety_pass=bool(safety and np.all(np.isfinite(safety_mean)) and np.min(safety_mean)>=margin)
        # Mean Pareto operating points, excluding gamma=0 identity.
        p=[row for row in pareto if row["protocol"]==protocol and float(row["gamma"])>0 and row["method"] in ("POP","DIFF-MATCH")];points=_mean_rows(p,("method","gamma"),("motion_coherence_reduction","nonartifact_observation_preservation"));match=[row for row in points if row["method"]=="DIFF-MATCH"];population=[row for row in points if row["method"]=="POP"];dominated=bool(match and population and all(any(float(pop["motion_coherence_reduction"])>=float(candidate["motion_coherence_reduction"]) and float(pop["nonartifact_observation_preservation"])>=float(candidate["nonartifact_observation_preservation"]) for pop in population) for candidate in match))
        required=(values_by["H_D_DIFF_MATCH_minus_DET_MATCH"],values_by["H_S_NULL"],values_by["H_S_WRONG"]);point=all(value.size and float(value.mean())>0 for value in required);majority=all(int(np.sum(value>0))>=int(config["evaluation"]["minimum_positive_participants"]) for value in required);expand=bool(point and majority and safety_pass and not dominated)
        decisions[protocol]={"additional_seeds_authorized":expand,"mean_effects":{"diffusion_increment":float(required[0].mean()) if required[0].size else None,"support_utility":float(required[1].mean()) if required[1].size else None,"specificity":float(required[2].mean()) if required[2].size else None},"positive_counts":[int(np.sum(value>0)) for value in required],"safety_names":["low_motion_preservation","PSD_utility","covariance_utility","ERP_amplitude","ERP_latency","SSVEP_SNR","SSVEP_phase","output_scale","observation_change"],"safety_mean_deltas":safety_mean.tolist(),"safety_passed":safety_pass,"pareto_dominated_by_population":dominated}
    _write(output/"paired_effects.csv",effects);_write(output/"method_summary.csv",_mean_rows(units,("protocol","method"),metrics));_write(output/"pareto_summary.csv",_mean_rows([row for row in pareto if float(row["gamma"])>0],("protocol","method","gamma"),("motion_coherence_reduction","nonartifact_observation_preservation","reference_free_psd_distortion","reference_free_covariance_distortion")))
    # Backup only when population diffusion is better than deterministic POP and support context collapses.
    population_effects=[]
    for protocol in decisions:
        by={}
        for row in units:
            if row["protocol"]==protocol:by.setdefault(row["participant"],{})[row["method"]]=row
        values=[float(methods["DIFF-POP"]["motion_coherence_reduction"])-float(methods["POP"]["motion_coherence_reduction"]) for methods in by.values() if "DIFF-POP" in methods and "POP" in methods];population_effects.extend(values)
    pop_array=np.asarray(population_effects);population_diffusion_signal=bool(pop_array.size and float(pop_array.mean())>0 and int(np.sum(pop_array>0))>=9)
    context_collapse=all((decision["mean_effects"]["support_utility"] or 0)<=0 or (decision["mean_effects"]["specificity"] or 0)<=0 for decision in decisions.values());backup=bool(population_diffusion_signal and context_collapse)
    expand=any(value["additional_seeds_authorized"] for value in decisions.values());summary={"status":"completed_v5_one_seed_factorial_aggregation" if len(seeds)==1 else "completed_v5_three_seed_stability","seeds":seeds,"protocol_decisions":decisions,"protocol_denominator":sum(row["denominator"] for row in coverage),"eligible_protocol_units":sum(row["eligible"] for row in coverage),"blocked_no_later_block":sum(row["blocked_no_later_block"] for row in coverage),"missing_protocol_units":sum(row["missing"] for row in coverage),"additional_seeds_authorized":expand,"score_adapter_backup_authorized":backup,"population_diffusion_signal":population_diffusion_signal,"context_collapse":context_collapse,"sealed_signal_opened":False,"confirmation_eligibility":False,"family_wide_status":"not_tested"};(output/"routing_decision.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
