"""One development-only deployable selector after the corrected P-C ceiling.

Inference features are restricted to observed query EEG and calibration-support
statistics. Query EOG, artifact classes, clean targets, and outcomes are opened
only to construct leave-one-unit-out development labels and final scores.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from eeg_cgdr.experiments.literature_guided_v3 import (
    _annotation_opener, _base_config, _continuous, _klados_eval_records,
    _prepared_v3, _sge_matching_support, _sge_samples_per_trial,
)
from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _evaluate_output, _paired_metrics


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _features(observed: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Query-EEG and support-only features; no candidate output or outcome."""
    epsilon = 1e-12
    centered = observed.astype(np.float64) - observed.mean(axis=-1, keepdims=True)
    power = np.mean(centered * centered, axis=(1, 2))
    rms = np.sqrt(power)
    kurtosis = np.mean(centered ** 4, axis=(1, 2)) / np.maximum(power * power, epsilon)
    line = np.mean(np.abs(np.diff(centered, axis=-1)), axis=(1, 2)) / np.maximum(rms, epsilon)
    spectrum = np.abs(np.fft.rfft(centered, axis=-1))
    high_ratio = spectrum[..., spectrum.shape[-1] // 2 :].mean(axis=(1, 2)) / np.maximum(spectrum.mean(axis=(1, 2)), epsilon)
    support_abs = np.abs(np.asarray(support, dtype=np.float64)).reshape(-1)
    summary = np.asarray([
        np.quantile(support_abs, 0.50), np.quantile(support_abs, 0.90),
        np.sqrt(np.mean(support_abs * support_abs)),
    ]) if support_abs.size else np.zeros(3)
    return np.column_stack((rms, kurtosis, line, high_ratio, *(np.full(rms.shape, value) for value in summary)))


def _fit(train_x: np.ndarray, train_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(0); scale = train_x.std(0); scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(train_x.shape[0]), (train_x - mean) / scale))
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return weights, mean, scale


def _predict(model: tuple[np.ndarray, np.ndarray, np.ndarray], value: np.ndarray) -> np.ndarray:
    weights, mean, scale = model
    return np.column_stack((np.ones(value.shape[0]), (value - mean) / scale)) @ weights


def _corr(eeg: np.ndarray, eog: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for channel in eeg:
        for external in np.atleast_2d(eog):
            if mask.sum() >= 3 and np.std(channel[mask]) > 1e-10 and np.std(external[mask]) > 1e-10:
                values.append(abs(float(np.corrcoef(channel[mask], external[mask])[0, 1])))
    return float(np.mean(values)) if values else 0.0


def _psd(value: np.ndarray, reference: np.ndarray) -> float:
    if value.shape[-1] < 3: return float("inf")
    return float(np.linalg.norm(np.abs(np.fft.rfft(value, axis=-1)) - np.abs(np.fft.rfft(reference, axis=-1))) / max(np.linalg.norm(np.abs(np.fft.rfft(reference, axis=-1))), 1e-12))


def _cov(value: np.ndarray, reference: np.ndarray) -> float:
    if value.shape[-1] < 3: return float("inf")
    left = np.cov(value); right = np.cov(reference)
    return float(np.linalg.norm(left - right, ord="fro") / max(np.linalg.norm(right, ord="fro"), 1e-12))


def _klados_units(config: Mapping[str, Any], arrays: Path, margin: float) -> list[dict[str, Any]]:
    base = _base_config(config); prepared = _prepared_v3(config, {"dataset":"klados","fold_index":0,"seed":20260811}); units=[]
    for unit, mechanism, _, _ in _klados_eval_records(base):
        archive=np.load(arrays/"P_A_RAW_SUPPORT_TOKENS/klados/fold_00"/f"{unit}.npz")
        observed=mechanism.observed_windows.astype(np.float64); clean=mechanism.clean_windows.astype(np.float64)
        pop=archive["STRONG_POP"].astype(np.float64); match=archive["DIFF_MATCH"].astype(np.float64); valid=mechanism.valid_time_weight.astype(bool)
        labels=[]
        for index in range(observed.shape[0]):
            keep=valid[index]; target=clean[index][:,keep]; p=pop[index][:,keep]; m=match[index][:,keep]
            p_error=np.linalg.norm(p-target)/max(np.linalg.norm(target),1e-12); m_error=np.linalg.norm(m-target)/max(np.linalg.norm(target),1e-12)
            p_corr=float(np.mean([np.corrcoef(p[c],target[c])[0,1] for c in range(p.shape[0]) if np.std(p[c])>1e-10 and np.std(target[c])>1e-10]))
            m_corr=float(np.mean([np.corrcoef(m[c],target[c])[0,1] for c in range(m.shape[0]) if np.std(m[c])>1e-10 and np.std(target[c])>1e-10]))
            safe=(m_corr-p_corr>=margin and _psd(p,target)-_psd(m,target)>=margin and _cov(p,target)-_cov(m,target)>=margin and .5<=np.sqrt(np.mean(m*m))/max(np.sqrt(np.mean(observed[index][:,keep]**2)),1e-12)<=2.)
            labels.append(float(m_error<p_error and safe))
        units.append({"dataset":"klados","unit_id":unit,"exact_cell":prepared.fold.layout_id,"observed":observed,"pop":pop,"match":match,"clean":clean,"valid":valid,"features":_features(observed,np.asarray(mechanism.calibration.eog)),"labels":np.asarray(labels)})
    return units


def _sge_units(config: Mapping[str, Any], arrays: Path, margin: float) -> list[dict[str, Any]]:
    base=_base_config(config); units=[]
    for fold in range(25):
        prepared=_prepared_v3(config,{"dataset":"sgeyesub","fold_index":fold,"seed":20260811}); directory=arrays/f"P_A_RAW_SUPPORT_TOKENS/sgeyesub/fold_{fold:02d}"
        for unit,heldout in prepared.heldout.items():
            path=directory/f"{unit.replace('/','__')}.npz"
            if not path.is_file(): continue
            archive=np.load(path); observed=heldout.query.observed.astype(np.float64); pop=archive["STRONG_POP"].astype(np.float64); match=archive["DIFF_MATCH"].astype(np.float64)
            annotated=_annotation_opener(base,prepared,unit)(); annotations=annotated.query_annotations
            if annotations is None: continue
            classes=np.asarray(annotations.artifactclasses); eog=np.asarray(annotations.external_eog); samples=observed.shape[-1]; labels=[]
            for index in range(observed.shape[0]):
                section=slice(index*samples,(index+1)*samples); local=classes[section]; local_eog=eog[...,section]
                artifact=(local>=1)&(local<=5); rest=local==6
                if local.size!=samples or not np.any(artifact) or rest.sum()<3: labels.append(0.0); continue
                benefit=_corr(pop[index],local_eog,artifact)-_corr(match[index],local_eog,artifact)
                p_change=np.linalg.norm((pop[index]-observed[index])[:,rest])/max(np.linalg.norm(observed[index][:,rest]),1e-12)
                m_change=np.linalg.norm((match[index]-observed[index])[:,rest])/max(np.linalg.norm(observed[index][:,rest]),1e-12)
                p=pop[index][:,rest]; m=match[index][:,rest]; reference=observed[index][:,rest]
                safe=(p_change-m_change>=margin and _psd(p,reference)-_psd(m,reference)>=margin and _cov(p,reference)-_cov(m,reference)>=margin and .5<=np.sqrt(np.mean(match[index]**2))/max(np.sqrt(np.mean(observed[index]**2)),1e-12)<=2.)
                labels.append(float(benefit>0 and safe))
            support=_sge_matching_support(base,prepared,fold,unit)[1]
            units.append({"dataset":"sgeyesub","unit_id":unit,"exact_cell":f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}","prepared":prepared,"heldout":heldout,"annotated":annotated,"observed":observed,"pop":pop,"match":match,"features":_features(observed,support),"labels":np.asarray(labels)})
    return units


def _bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[float,float]:
    if not values.size: return float("nan"),float("nan")
    rng=np.random.default_rng(seed); samples=np.asarray([values[rng.integers(0,values.size,values.size)].mean() for _ in range(repetitions)])
    return float(np.quantile(samples,.025)),float(np.quantile(samples,.975))


def run(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    pc=CODE_ROOT/str(config["pc_output_root"]); ceiling=json.loads((pc/"result_summary.json").read_text())
    if ceiling.get("decision")!="GO_MINIMAL_SELECTOR": raise RuntimeError("minimal selector requires corrected GO_MINIMAL_SELECTOR ceiling")
    arrays=Path(str(config["v3_result_root"]))/"server_arrays"; margin=float(config["evaluation"]["noninferiority_margin"])
    units=_klados_units(config,arrays,margin)+_sge_units(config,arrays,margin); rows=[]
    for test in units:
        train=[unit for unit in units if unit["dataset"]==test["dataset"] and unit["exact_cell"]==test["exact_cell"] and unit["unit_id"]!=test["unit_id"]]
        if not train:
            rows.append({"dataset":test["dataset"],"unit_id":test["unit_id"],"exact_cell":test["exact_cell"],"status":"blocked_singleton_exact_cell","coverage":0.0,"query_outcome_feature_used":False}); continue
        model=_fit(np.concatenate([unit["features"] for unit in train]),np.concatenate([unit["labels"] for unit in train])); actions=_predict(model,test["features"])>=.5
        mixed=test["pop"].copy(); mixed[actions]=test["match"][actions]
        if test["dataset"]=="klados":
            pop_metric=_paired_metrics(test["observed"],test["clean"],test["pop"],test["valid"]); metric=_paired_metrics(test["observed"],test["clean"],mixed,test["valid"])
            artifact=float(pop_metric["clean_waveform_RRMSE"]-metric["clean_waveform_RRMSE"]); preservation=float(metric["clean_waveform_correlation"]-pop_metric["clean_waveform_correlation"])
            pop_flat=np.concatenate([test["pop"][i][:,test["valid"][i]] for i in range(test["pop"].shape[0])],axis=1); mix_flat=np.concatenate([mixed[i][:,test["valid"][i]] for i in range(mixed.shape[0])],axis=1); clean_flat=np.concatenate([test["clean"][i][:,test["valid"][i]] for i in range(test["clean"].shape[0])],axis=1)
            psd=_psd(pop_flat,clean_flat)-_psd(mix_flat,clean_flat); cov=_cov(pop_flat,clean_flat)-_cov(mix_flat,clean_flat)
        else:
            annotated=test["annotated"]; annotations=annotated.query_annotations; observed=_continuous(test["observed"]); pop=_continuous(test["pop"]); output=_continuous(mixed); length=min(observed.shape[1],len(annotations.artifactclasses),annotations.external_eog.shape[-1]); observed=observed[:,:length];pop=pop[:,:length];output=output[:,:length]
            def evaluate(name:str,value:np.ndarray)->Mapping[str,Any]: return _evaluate_output(method_id=name,output=value,observed=observed,matching_projector=test["heldout"].matching.projector,population_projector=test["prepared"].population_context.projector,query_eog=annotations.external_eog[...,:length],artifactclasses=annotations.artifactclasses[:length],predicted_contamination=None,trial_labels=annotations.trial_labels,samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz),minimum_trials_per_condition=2,status="success",operator_source="frozen_v3",gamma=None,fallback_used=name=="POP",uses_query_external_eog=False)
            p=evaluate("POP",pop);m=evaluate("SELECTOR",output);artifact=float(m["eog_coherence_reduction"]-p["eog_coherence_reduction"]);preservation=float(m["nonartifact_observation_preservation"]-p["nonartifact_observation_preservation"]);psd=float(p["reference_free_psd_distortion"]-m["reference_free_psd_distortion"]);cov=float(p["reference_free_covariance_distortion"]-m["reference_free_covariance_distortion"])
        rows.append({"dataset":test["dataset"],"unit_id":test["unit_id"],"exact_cell":test["exact_cell"],"status":"success_leave_one_unit_out_exact_cell","coverage":float(actions.mean()),"artifact_utility":artifact,"preservation_utility":preservation,"psd_utility":psd,"covariance_utility":cov,"query_outcome_feature_used":False,"query_external_signal_feature_used":False})
    _write_csv(pc/"minimal_selector_unit_metrics.csv",rows); summaries=[]
    for dataset,denominator in (("klados",16),("sgeyesub",59)):
        selected=[row for row in rows if row["dataset"]==dataset and str(row["status"]).startswith("success")]; values=np.asarray([float(row["artifact_utility"]) for row in selected]);ci=_bootstrap(values,int(config["evaluation"]["bootstrap_repetitions"]),20260806)
        summaries.append({"dataset":dataset,"denominator":denominator,"successful":len(selected),"mean_artifact_utility":float(values.mean()) if values.size else float("nan"),"median_artifact_utility":float(np.median(values)) if values.size else float("nan"),"ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(values>0)),"mean_coverage":float(np.mean([float(row["coverage"]) for row in selected])) if selected else 0.0,"mean_preservation_utility":float(np.mean([float(row["preservation_utility"]) for row in selected])) if selected else float("nan"),"mean_psd_utility":float(np.mean([float(row["psd_utility"]) for row in selected])) if selected else float("nan"),"mean_covariance_utility":float(np.mean([float(row["covariance_utility"]) for row in selected])) if selected else float("nan")})
    _write_csv(pc/"minimal_selector_summary.csv",summaries); supported=all(row["mean_artifact_utility"]>0 and row["mean_preservation_utility"]>=margin and row["mean_psd_utility"]>=margin and row["mean_covariance_utility"]>=margin for row in summaries)
    summary={"status":"completed_single_minimal_deployable_selector","selector_supported_across_klados_and_sge":supported,"inference_features":"observed_query_EEG_and_calibration_support_only","query_EOG_artifactclass_clean_target_outcome_features":False,"evaluation":"leave_one_source_or_stem_out_exact_cell_development","klados_denominator":16,"sge_denominator":59}
    (pc/"minimal_selector_result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");run_dir.mkdir(parents=True,exist_ok=True);(run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");return summary
