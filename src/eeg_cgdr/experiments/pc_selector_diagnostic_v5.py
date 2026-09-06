"""Unit-weighted P-C selector diagnostics on frozen v3 arrays."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any,Mapping

import numpy as np

from eeg_cgdr.experiments.pc_minimal_selector import _klados_units,_sge_units


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_diffusion_v5"))


def _write(path:Path,rows:list[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _auc(labels:np.ndarray,scores:np.ndarray)->tuple[float,float]:
    positive=labels>0.5;negative=~positive
    if not positive.any() or not negative.any():return float("nan"),float("nan")
    order=np.argsort(scores);ranks=np.empty(scores.size);ranks[order]=np.arange(1,scores.size+1);auroc=(ranks[positive].sum()-positive.sum()*(positive.sum()+1)/2)/(positive.sum()*negative.sum());descending=np.argsort(scores)[::-1];truth=positive[descending];precision=np.cumsum(truth)/np.arange(1,len(truth)+1);auprc=float(np.sum(precision*truth)/positive.sum());return float(auroc),auprc


def _fit(units:list[dict[str,Any]],feature_name:str)->tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    xs=[];ys=[];weights=[]
    for unit in units:
        value=unit[feature_name];xs.append(value);ys.append(unit["labels"]);weights.append(np.full(value.shape[0],1/max(value.shape[0],1)))
    x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(weights);mean=np.average(x,axis=0,weights=weight);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=weight));scale[scale<1e-8]=1.;design=np.column_stack((np.ones(x.shape[0]),(x-mean)/scale));root=np.sqrt(weight)[:,None];regularizer=np.eye(design.shape[1]);regularizer[0,0]=0.;coef=np.linalg.solve((design*root).T@(design*root)+regularizer,(design*root).T@(y*np.sqrt(weight)));scores=design@coef;coverage=float(np.mean([np.mean(unit["labels"]) for unit in units]));threshold=float(np.quantile(scores,1-coverage)) if coverage>0 else float("inf");return coef,mean,scale,threshold


def _predict(model:tuple[np.ndarray,np.ndarray,np.ndarray,float],x:np.ndarray)->np.ndarray:
    coef,mean,scale,_=model;return np.column_stack((np.ones(x.shape[0]),(x-mean)/scale))@coef


def _finite_mean(values:list[Any])->float:
    finite=np.asarray([float(value) for value in values if np.isfinite(float(value))]);return float(finite.mean()) if finite.size else float("nan")


def run(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    arrays=Path(str(config["v3_result_root"]))/"server_arrays";margin=float(config["evaluation"]["noninferiority_margin"]);units=_klados_units(config,arrays,margin)+_sge_units(config,arrays,margin)
    # M1 adds frozen-output disagreement; it is diagnostic and not an inference-free baseline.
    for unit in units:
        observed=np.asarray(unit["observed"],dtype=np.float64);pop=np.asarray(unit["pop"],dtype=np.float64);match=np.asarray(unit["match"],dtype=np.float64);den=np.maximum(np.sqrt(np.mean(observed**2,axis=(1,2))),1e-12);disagreement=np.column_stack((np.sqrt(np.mean((match-pop)**2,axis=(1,2)))/den,np.sqrt(np.mean((match-observed)**2,axis=(1,2)))/den,np.sqrt(np.mean((pop-observed)**2,axis=(1,2)))/den));unit["M0"]=unit["features"];unit["M1"]=np.column_stack((unit["features"],disagreement))
    rows=[]
    for test in units:
        train=[unit for unit in units if unit["dataset"]==test["dataset"] and unit["exact_cell"]==test["exact_cell"] and unit["unit_id"]!=test["unit_id"]]
        if not train:
            for route in ("M0","M1"):rows.append({"dataset":test["dataset"],"unit_id":test["unit_id"],"exact_cell":test["exact_cell"],"route":route,"status":"blocked_singleton_exact_cell"})
            continue
        donors=[unit for unit in train if unit["features"].shape[1]==test["features"].shape[1]]
        wrong=donors[sum(ord(c) for c in test["unit_id"])%len(donors)]
        for route in ("M0","M1"):
            model=_fit(train,route);scores=_predict(model,test[route]);threshold=model[3];actions=scores>=threshold;auroc,auprc=_auc(np.asarray(test["labels"]),scores)
            wrong_x=test[route].copy();wrong_x[:,4:7]=np.resize(wrong[route][:,4:7].mean(0),(wrong_x.shape[0],3));wrong_scores=_predict(model,wrong_x);wrong_actions=wrong_scores>=threshold
            rows.append({"dataset":test["dataset"],"unit_id":test["unit_id"],"exact_cell":test["exact_cell"],"route":route,"status":"success_LOO_unit_weighted","label_prevalence":float(np.mean(test["labels"])),"score_finite_rate":float(np.isfinite(scores).mean()),"score_q01":float(np.quantile(scores,.01)),"score_q50":float(np.quantile(scores,.5)),"score_q99":float(np.quantile(scores,.99)),"auroc":auroc,"auprc":auprc,"coverage":float(actions.mean()),"selected_safe_benefit_rate":float(np.mean(test["labels"][actions])) if actions.any() else 0.,"wrong_support_coverage":float(wrong_actions.mean()),"wrong_support_selected_safe_benefit_rate":float(np.mean(test["labels"][wrong_actions])) if wrong_actions.any() else 0.,"threshold_calibration":"training_unit_mean_achieved_coverage","infeasible_action":"abstain_to_POP"})
    output=CODE_ROOT/str(config["pc_output_root"]);_write(output/"loo_diagnostics.csv",rows);summaries=[]
    for dataset in ("klados","sgeyesub"):
        for route in ("M0","M1"):
            selected=[row for row in rows if row["dataset"]==dataset and row["route"]==route and str(row["status"]).startswith("success")];summaries.append({"dataset":dataset,"route":route,"denominator":16 if dataset=="klados" else 59,"successful":len(selected),"mean_label_prevalence":float(np.mean([row["label_prevalence"] for row in selected])) if selected else float("nan"),"mean_auroc":_finite_mean([row["auroc"] for row in selected]),"mean_auprc":_finite_mean([row["auprc"] for row in selected]),"mean_coverage":float(np.mean([row["coverage"] for row in selected])) if selected else 0.,"matching_selected_safe_rate":float(np.mean([row["selected_safe_benefit_rate"] for row in selected])) if selected else 0.,"wrong_support_selected_safe_rate":float(np.mean([row["wrong_support_selected_safe_benefit_rate"] for row in selected])) if selected else 0.})
    _write(output/"summary.csv",summaries);supported=any(all(row["matching_selected_safe_rate"]>row["wrong_support_selected_safe_rate"] and row["matching_selected_safe_rate"]>0 for row in summaries if row["route"]==route) for route in ("M0","M1"));summary={"status":"completed_bounded_candidate_selector_diagnostic","routes":["M0_seven_features_coverage_calibrated","M1_plus_output_disagreement"],"unit_weighting":True,"infeasible_units":"abstain_to_POP_full_denominator","selector_supported_across_datasets":supported,"scientific_role":"CPU_diagnostic_nonblocking","sealed_mobile_data_opened":False};(output/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
