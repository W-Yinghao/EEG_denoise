"""Participant-first aggregation and non-binary V29 classifications."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any
import numpy as np
from eeg_scad.evaluation.aggregate_v26 import bootstrap,participant_first
from eeg_scad.evaluation.aggregate_v28 import PAIRED_METRICS,NATURAL_METRICS,LOWER_IS_BETTER


def _rows(paths):
    result=[]
    for path in paths:result.extend(csv.DictReader(path.open()))
    return result
def contrast(rows,first,second,metric):
    values={(str(r["participant"]),str(r["method"])):float(r[metric]) for r in rows};sign=-1 if metric in LOWER_IS_BETTER else 1;participants=sorted({p for p,m in values if m==first}&{p for p,m in values if m==second});return [{"participant":p,"first":first,"second":second,"metric":metric,"effect":sign*(values[p,first]-values[p,second])} for p in participants]


def aggregate(derived:Path,seeds:list[int],bootstrap_seed:int=20260908)->tuple[dict[str,Any],dict[str,list[dict[str,Any]]]]:
    paired=_rows(sorted((derived/"metrics/paired").glob("*.csv")));natural=_rows(sorted((derived/"metrics/natural").glob("*.csv")));pp=participant_first(paired,PAIRED_METRICS+["adapter_rms"]);nn=participant_first(natural,NATURAL_METRICS+["adapter_rms"]);summary=[]
    for panel,rows,metrics in (("paired",pp,PAIRED_METRICS+["adapter_rms"]),("natural",nn,NATURAL_METRICS+["adapter_rms"])):
        for method in sorted({r["method"] for r in rows}):
            chosen=[r for r in rows if r["method"]==method]
            for metric in metrics:
                vector=np.asarray([float(r[metric]) for r in chosen]);vector=vector[np.isfinite(vector)]
                if len(vector):summary.append({"panel":panel,"method":method,"metric":metric,**bootstrap(vector,bootstrap_seed)})
    definitions=(("CDM_MATCH_POP_ADAPTER","PA_SC_CDM_MATCH","POP_ADAPTER_CDM"),("CDM_MATCH_WRONG","PA_SC_CDM_MATCH","PA_SC_CDM_WRONG"),("CDM_MATCH_FROZEN_POP","PA_SC_CDM_MATCH","POP_CLEAN_CDM"),("DET_MATCH_POP_ADAPTER","PA_SC_DET_MATCH","POP_ADAPTER_DET"),("DET_MATCH_WRONG","PA_SC_DET_MATCH","PA_SC_DET_WRONG"),("DET_MATCH_FROZEN_POP","PA_SC_DET_MATCH","POP_CLEAN_DET"),("CDM_DET","PA_SC_CDM_MATCH","PA_SC_DET_MATCH"));effects=[]
    for panel,rows,metrics in (("paired",pp,["rrmse_temporal","rrmse_spectral","correlation"]),("natural",nn,["heldout_eog_remaining_ratio","artifact_attenuation_db","low_eog_observation_retention","psd_distortion","covariance_distortion"])):
        for name,first,second in definitions:
            for metric in metrics:
                effects.extend({"panel":panel,"contrast":name,**row} for row in contrast(rows,first,second,metric))
    seed_rows=[]
    for panel,rows,metrics in (("paired",paired,PAIRED_METRICS),("natural",natural,NATURAL_METRICS)):
        for seed in seeds:
            for method in sorted({r["method"] for r in rows}):
                chosen=[r for r in rows if int(r["seed"])==seed and r["method"]==method]
                for metric in metrics:
                    v=np.asarray([float(r[metric]) for r in chosen]);v=v[np.isfinite(v)]
                    if len(v):seed_rows.append({"panel":panel,"seed":seed,"method":method,"metric":metric,"mean":float(v.mean())})
    severity=[]
    for level in sorted({r["severity"] for r in paired}):
        for method in sorted({r["method"] for r in paired}):
            v=[float(r["rrmse_temporal"]) for r in paired if r["severity"]==level and r["method"]==method]
            if v:severity.append({"severity":level,"method":method,"clean_rrmse":float(np.mean(v)),"rows":len(v)})
    def stat(panel,name,metric):return bootstrap(np.asarray([float(r["effect"]) for r in effects if r["panel"]==panel and r["contrast"]==name and r["metric"]==metric]),bootstrap_seed)
    cdm_pop=stat("paired","CDM_MATCH_POP_ADAPTER","rrmse_temporal");cdm_wrong=stat("paired","CDM_MATCH_WRONG","rrmse_temporal");cdm_det=stat("paired","CDM_DET","rrmse_temporal");det_pop=stat("paired","DET_MATCH_POP_ADAPTER","rrmse_temporal");natural_art=stat("natural","CDM_MATCH_POP_ADAPTER","heldout_eog_remaining_ratio");natural_ret=stat("natural","CDM_MATCH_POP_ADAPTER","low_eog_observation_retention");natural_psd=stat("natural","CDM_MATCH_POP_ADAPTER","psd_distortion")
    support="clear_paired_signal" if cdm_pop["mean"]>0 and cdm_wrong["mean"]>0 else "specificity_only" if cdm_wrong["mean"]>0 else "weak_paired_signal" if cdm_pop["mean"]>0 else "harmful" if cdm_pop["mean"]<0 else "absent";capacity="support_beats_pop_adapter" if cdm_pop["mean"]>0 and det_pop["mean"]>0 else "adapter_capacity_explains_gain" if cdm_pop["mean"]<=0 and det_pop["mean"]<=0 else "mixed";diffusion="competitive_with_det" if abs(cdm_det["mean"])<=.01 else "diffusion_better" if cdm_det["mean"]>0 else "det_better";natural_class="artifact_promising_retention_acceptable" if natural_art["mean"]>0 and natural_ret["mean"]>=0 else "artifact_insufficient" if natural_art["mean"]<=0 and natural_ret["mean"]>=0 else "retention_concern" if natural_art["mean"]>0 else "mixed";absolute="improved_over_v28" if cdm_pop["mean"]>0 else "similar_to_v28" if abs(cdm_pop["mean"])<.005 else "worse";next_route="A. freeze method and complete revision experiments" if support=="clear_paired_signal" and natural_class=="artifact_promising_retention_acceptable" else "B. one global residual-head refinement" if support in ("clear_paired_signal","weak_paired_signal","specificity_only") else "D. consult AE before confirmation"
    diagnosis={"engineering":"valid","absolute_denoising":absolute,"support_mechanism":support,"capacity_control":capacity,"diffusion_positioning":diffusion,"natural":natural_class,"next_route":next_route,"development_only":True,"paired":{"cdm_match_pop_adapter":cdm_pop,"cdm_match_wrong":cdm_wrong,"det_match_pop_adapter":det_pop,"cdm_det":cdm_det},"natural_effects":{"artifact":natural_art,"retention":natural_ret,"psd":natural_psd},"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0}
    return diagnosis,{"summary":summary,"effects":effects,"seed":seed_rows,"severity":severity,"paired_participant":pp,"natural_participant":nn}


__all__=["aggregate"]
