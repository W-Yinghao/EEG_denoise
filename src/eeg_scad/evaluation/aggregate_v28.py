"""Participant-first V28 aggregation and development classification."""
from __future__ import annotations

import csv,json
from pathlib import Path
from typing import Any,Mapping

import numpy as np

from eeg_scad.evaluation.aggregate_v26 import bootstrap,contrast,participant_first


PAIRED_METRICS=["rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","clean_output_rms_ratio","identity_change"]
NATURAL_METRICS=["heldout_eog_remaining_ratio","artifact_attenuation_db","low_eog_observation_change","low_eog_observation_retention","psd_distortion","covariance_distortion","output_input_rms_ratio","observation_change_ratio","eeg_eog_coherence_reduction","frontal_residual_topography"]


def _rows(paths):
    values=[]
    for path in paths:values.extend(csv.DictReader(path.open()))
    return values


def aggregate(derived:Path,output:Path,seeds:list[int],bootstrap_seed:int=20260904)->tuple[dict[str,Any],dict[str,list[dict[str,Any]]]]:
    paired=_rows(sorted((derived/"metrics/paired").glob("*.csv")));natural=_rows(sorted((derived/"metrics/natural").glob("*.csv")));pp=participant_first(paired,PAIRED_METRICS);nn=participant_first(natural,NATURAL_METRICS);summary=[]
    for panel,rows,metrics in (("paired",pp,PAIRED_METRICS),("natural",nn,NATURAL_METRICS)):
        for method in sorted({r["method"] for r in rows}):
            chosen=[r for r in rows if r["method"]==method]
            for metric in metrics:
                vector=np.asarray([float(r[metric]) for r in chosen]);vector=vector[np.isfinite(vector)]
                if len(vector):summary.append({"panel":panel,"method":method,"metric":metric,**bootstrap(vector,bootstrap_seed)})
    definitions=(
        ("CDM_MATCH_POP","SUPPORT_CLEAN_CDM_MATCH","POP_CLEAN_CDM"),
        ("CDM_MATCH_WRONG","SUPPORT_CLEAN_CDM_MATCH","SUPPORT_CLEAN_CDM_WRONG"),
        ("CDM_MATCH_NULL","SUPPORT_CLEAN_CDM_MATCH","SUPPORT_CLEAN_CDM_NULL"),
        ("CDM_DET","SUPPORT_CLEAN_CDM_MATCH","SUPPORT_CLEAN_DET_MATCH"),
        ("DET_MATCH_POP","SUPPORT_CLEAN_DET_MATCH","POP_CLEAN_DET"),
        ("DET_MATCH_WRONG","SUPPORT_CLEAN_DET_MATCH","SUPPORT_CLEAN_DET_WRONG"),
    );effects=[]
    for panel,rows,metrics in (("paired",pp,["rrmse_temporal","rrmse_spectral","correlation"]),("natural",nn,["heldout_eog_remaining_ratio","artifact_attenuation_db","low_eog_observation_change","low_eog_observation_retention","psd_distortion","covariance_distortion"])):
        for name,first,second in definitions:
            for metric in metrics:
                for row in contrast(rows,first,second,metric):effects.append({"panel":panel,"contrast":name,**row})
    seed_rows=[]
    for panel,rows,metrics in (("paired",paired,PAIRED_METRICS),("natural",natural,NATURAL_METRICS)):
        for seed in seeds:
            for method in sorted({r["method"] for r in rows}):
                chosen=[r for r in rows if int(r["seed"])==seed and r["method"]==method]
                for metric in metrics:
                    vector=np.asarray([float(r[metric]) for r in chosen]);vector=vector[np.isfinite(vector)]
                    if len(vector):seed_rows.append({"panel":panel,"seed":seed,"method":method,"metric":metric,"mean":float(vector.mean())})
    severity=[]
    for severity_name in sorted({r["severity"] for r in paired}):
        chosen=[r for r in paired if r["severity"]==severity_name]
        for method in sorted({r["method"] for r in chosen}):
            vector=[float(r["rrmse_temporal"]) for r in chosen if r["method"]==method];severity.append({"severity":severity_name,"method":method,"clean_rrmse":float(np.mean(vector)),"rows":len(vector)})
    def stat(panel,name,metric):
        vector=np.asarray([float(r["effect"]) for r in effects if r["panel"]==panel and r["contrast"]==name and r["metric"]==metric]);return bootstrap(vector,bootstrap_seed)
    paired_support=stat("paired","CDM_MATCH_POP","rrmse_temporal");specificity=stat("paired","CDM_MATCH_WRONG","rrmse_temporal");cdm_det=stat("paired","CDM_DET","rrmse_temporal");natural_artifact=stat("natural","CDM_MATCH_POP","heldout_eog_remaining_ratio");natural_retention=stat("natural","CDM_MATCH_POP","low_eog_observation_retention");natural_psd=stat("natural","CDM_MATCH_POP","psd_distortion")
    clean_class="competitive" if abs(cdm_det["mean"])<=.02 else "weak_but_usable" if cdm_det["mean"]>-.05 else "undertrained";support_class="paired_signal_clear" if paired_support["mean"]>0 and specificity["mean"]>0 else "paired_signal_weak" if paired_support["mean"]>0 or specificity["mean"]>0 else "paired_signal_harmful" if paired_support["mean"]<0 else "paired_signal_absent";artifact_class="promising" if natural_artifact["mean"]>0 else "mixed" if natural_artifact["positive"]>=7 else "insufficient";retention_class="acceptable" if natural_retention["mean"]>=0 and natural_psd["mean"]>=-.02 else "mixed" if natural_retention["positive"]>=7 else "concern";next_route="A. freeze method and complete revision experiments" if support_class=="paired_signal_clear" and artifact_class=="promising" and retention_class=="acceptable" else "B. one small training refinement" if support_class in ("paired_signal_clear","paired_signal_weak") else "D. narrow claim and consult AE"
    diagnosis={"engineering":"valid","clean_conditional_diffusion":clean_class,"support_mechanism":support_class,"natural_artifact":artifact_class,"natural_observation_retention":retention_class,"task_valid_preservation":"unavailable","next_route":next_route,"paired":{"support_vs_population":paired_support,"match_vs_wrong":specificity,"cdm_vs_det":cdm_det},"natural":{"support_artifact":natural_artifact,"support_observation_retention":natural_retention,"support_psd":natural_psd},"interpretation":"DET/CNN are competitive positioning controls, not diffusion survival gates; natural metrics are separated and no retention scalar is called physiological preservation.","development_only":True,"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0}
    return diagnosis,{"summary":summary,"effects":effects,"seed":seed_rows,"severity":severity,"paired_participant":pp,"natural_participant":nn}


__all__=["aggregate","PAIRED_METRICS","NATURAL_METRICS"]
