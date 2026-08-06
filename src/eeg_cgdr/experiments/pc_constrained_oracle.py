"""Safety-constrained, post-output P-C oracle on frozen v3 arrays."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from eeg_cgdr.experiments.literature_guided_v3 import (
    _annotation_opener, _base_config, _continuous, _klados_eval_records,
    _prepared_v3, _sge_samples_per_trial,
)
from eeg_cgdr.experiments.mainline_subject_residual_diffusion import _evaluate_output, _paired_metrics


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _bootstrap(values: np.ndarray, repetitions: int, seed: int, strata: list[str] | None = None) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return float("nan"), float("nan")
    samples = np.empty(repetitions, dtype=np.float64)
    if strata is None:
        for index in range(repetitions):
            samples[index] = values[rng.integers(0, values.size, values.size)].mean()
    else:
        groups = [np.flatnonzero(np.asarray(strata) == name) for name in sorted(set(strata))]
        for index in range(repetitions):
            draw = np.concatenate([group[rng.integers(0, group.size, group.size)] for group in groups])
            samples[index] = values[draw].mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _covariance_distortion(output: np.ndarray, reference: np.ndarray) -> float:
    left = np.cov(output, bias=False); right = np.cov(reference, bias=False)
    return float(np.linalg.norm(left - right, ord="fro") / max(np.linalg.norm(right, ord="fro"), 1e-12))


def _psd_distortion(output: np.ndarray, reference: np.ndarray) -> float:
    left = np.abs(np.fft.rfft(output, axis=-1)); right = np.abs(np.fft.rfft(reference, axis=-1))
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1e-12))


def _candidate_masks(benefit: np.ndarray, risk: np.ndarray, count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if count <= 0:
        return [np.zeros(benefit.size, dtype=bool)]
    scales = np.linspace(0.0, 8.0, 65)
    for scale in scales:
        score = benefit - scale * risk
        chosen = np.argsort(score)[::-1][:count]
        mask = np.zeros(benefit.size, dtype=bool); mask[chosen] = True
        masks.append(mask)
    # Safety-first candidate makes constrained infeasibility explicit.
    chosen = np.argsort(risk)[:count]
    mask = np.zeros(benefit.size, dtype=bool); mask[chosen] = True; masks.append(mask)
    return masks


def _choose(
    benefit: np.ndarray, risk: np.ndarray, count: int,
    evaluator: Callable[[np.ndarray], Mapping[str, float]], margin: float,
) -> tuple[np.ndarray | None, Mapping[str, float] | None]:
    best_mask = None; best_metrics = None; best_utility = -np.inf
    for mask in _candidate_masks(benefit, risk, count):
        metrics = evaluator(mask)
        safety = all(float(metrics[key]) >= margin for key in ("preservation_utility", "psd_utility", "covariance_utility"))
        safety = safety and 0.5 <= float(metrics["output_input_RMS_ratio"]) <= 2.0
        if safety and float(metrics["artifact_utility"]) > best_utility:
            best_mask, best_metrics, best_utility = mask, metrics, float(metrics["artifact_utility"])
    return best_mask, best_metrics


def _klados_units(config: Mapping[str, Any], arrays_root: Path, coverages: list[float], margin: float) -> list[dict[str, Any]]:
    base = _base_config(config)
    rows: list[dict[str, Any]] = []
    for unit, mechanism, _, _ in _klados_eval_records(base):
        archive = np.load(arrays_root / "P_A_RAW_SUPPORT_TOKENS/klados/fold_00" / f"{unit}.npz")
        pop = archive["STRONG_POP"].astype(np.float64); match = archive["DIFF_MATCH"].astype(np.float64)
        clean = mechanism.clean_windows.astype(np.float64); observed = mechanism.observed_windows.astype(np.float64)
        valid = mechanism.valid_time_weight.astype(bool)
        epsilon = np.finfo(np.float64).eps
        benefit = np.asarray([
            np.linalg.norm(pop[index, :, valid[index]] - clean[index, :, valid[index]])
            - np.linalg.norm(match[index, :, valid[index]] - clean[index, :, valid[index]])
            for index in range(pop.shape[0])
        ])
        risk = np.asarray([
            np.linalg.norm(match[index, :, valid[index]] - clean[index, :, valid[index]])
            / max(np.linalg.norm(clean[index, :, valid[index]]), epsilon)
            for index in range(pop.shape[0])
        ])
        pop_metrics = _paired_metrics(observed, clean, pop, valid)
        pop_flat = np.concatenate([pop[i][:, valid[i]] for i in range(pop.shape[0])], axis=1)
        clean_flat = np.concatenate([clean[i][:, valid[i]] for i in range(clean.shape[0])], axis=1)
        pop_psd = _psd_distortion(pop_flat, clean_flat); pop_cov = _covariance_distortion(pop_flat, clean_flat)
        def evaluate(mask: np.ndarray) -> Mapping[str, float]:
            mixed = pop.copy(); mixed[mask] = match[mask]
            metrics = _paired_metrics(observed, clean, mixed, valid)
            mixed_flat = np.concatenate([mixed[i][:, valid[i]] for i in range(mixed.shape[0])], axis=1)
            return {
                "artifact_utility": pop_metrics["clean_waveform_RRMSE"] - metrics["clean_waveform_RRMSE"],
                "preservation_utility": metrics["clean_waveform_correlation"] - pop_metrics["clean_waveform_correlation"],
                "psd_utility": pop_psd - _psd_distortion(mixed_flat, clean_flat),
                "covariance_utility": pop_cov - _covariance_distortion(mixed_flat, clean_flat),
                "erp_utility": float("nan"), "output_input_RMS_ratio": metrics["output_input_RMS_ratio"],
            }
        maximum = 0.0
        for coverage in coverages:
            count = int(round(coverage * benefit.size))
            chosen, metrics = _choose(benefit, risk, count, evaluate, margin)
            if chosen is not None:
                maximum = max(maximum, float(chosen.mean()))
            rows.append({
                "dataset": "klados", "unit_id": unit, "exact_cell": "klados_19ch_256hz",
                "coverage_requested": coverage, "coverage_achieved": float(chosen.mean()) if chosen is not None else 0.0,
                "maximum_safe_coverage": maximum, "status": "success" if chosen is not None else "infeasible_safety_constraint",
                **(dict(metrics) if metrics is not None else {key: float("nan") for key in (
                    "artifact_utility", "preservation_utility", "psd_utility", "covariance_utility", "erp_utility", "output_input_RMS_ratio")}),
            })
    return rows


def _sge_units(config: Mapping[str, Any], arrays_root: Path, coverages: list[float], margin: float) -> list[dict[str, Any]]:
    base = _base_config(config); rows: list[dict[str, Any]] = []
    for fold in range(25):
        prepared = _prepared_v3(config, {"dataset": "sgeyesub", "fold_index": fold, "seed": 20260811})
        directory = arrays_root / f"P_A_RAW_SUPPORT_TOKENS/sgeyesub/fold_{fold:02d}"
        for unit, heldout in prepared.heldout.items():
            path = directory / f"{unit.replace('/', '__')}.npz"
            if not path.is_file():
                for coverage in coverages:
                    rows.append({"dataset": "sgeyesub", "unit_id": unit, "exact_cell": prepared.fold.layout_id,
                                 "coverage_requested": coverage, "coverage_achieved": 0.0, "maximum_safe_coverage": 0.0,
                                 "status": "blocked_missing_frozen_output"})
                continue
            archive = np.load(path); pop_w = archive["STRONG_POP"].astype(np.float64); match_w = archive["DIFF_MATCH"].astype(np.float64)
            observed_w = heldout.query.observed.astype(np.float64)
            annotated = _annotation_opener(base, prepared, unit)(); annotations = annotated.query_annotations
            if annotations is None:
                raise ValueError("SGE annotations unavailable after frozen output open")
            pop = _continuous(pop_w); match = _continuous(match_w); observed = _continuous(observed_w)
            artifact = np.asarray((annotations.artifactclasses >= 1) & (annotations.artifactclasses <= 5), dtype=bool)
            rest = np.asarray(annotations.artifactclasses == 6, dtype=bool)
            length = min(observed.shape[1], artifact.size, annotations.external_eog.shape[-1])
            observed = observed[:, :length]; pop = pop[:, :length]; match = match[:, :length]
            artifact = artifact[:length]; rest = rest[:length]
            eog = np.asarray(annotations.external_eog)[..., :length]
            window = observed_w.shape[-1]; windows = min(pop_w.shape[0], length // window)
            if windows < 1 or not np.any(rest) or not np.any(artifact):
                for coverage in coverages:
                    rows.append({"dataset": "sgeyesub", "unit_id": unit, "exact_cell": prepared.fold.layout_id,
                                 "coverage_requested": coverage, "coverage_achieved": 0.0, "maximum_safe_coverage": 0.0,
                                 "status": "infeasible_missing_rest_or_artifact_coverage"})
                continue
            def corr(value: np.ndarray, ext: np.ndarray, mask: np.ndarray) -> float:
                vals=[]
                for x in value:
                    for z in np.atleast_2d(ext):
                        if np.std(x[mask]) > 1e-10 and np.std(z[mask]) > 1e-10:
                            vals.append(abs(float(np.corrcoef(x[mask], z[mask])[0, 1])))
                return float(np.mean(vals)) if vals else 0.0
            benefit=[]; risk=[]
            for index in range(windows):
                section = slice(index * window, (index + 1) * window)
                local_artifact = artifact[section]; local_rest = rest[section]
                benefit.append((corr(pop[:, section], eog[:, section], local_artifact) - corr(match[:, section], eog[:, section], local_artifact)) if np.any(local_artifact) else 0.0)
                risk.append(np.linalg.norm((match-pop)[:, section][:, local_rest]) / max(np.linalg.norm(pop[:, section][:, local_rest]), 1e-12) if np.any(local_rest) else 1.0)
            benefit=np.asarray(benefit); risk=np.asarray(risk)
            pop_eval = _evaluate_output(method_id="STRONG-POP", output=pop, observed=observed,
                matching_projector=heldout.matching.projector, population_projector=prepared.population_context.projector,
                query_eog=eog, artifactclasses=annotations.artifactclasses[:length], predicted_contamination=None,
                trial_labels=annotations.trial_labels, samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz),
                minimum_trials_per_condition=2, status="success", operator_source="frozen_v3", gamma=None,
                fallback_used=True, uses_query_external_eog=False)
            def evaluate(mask: np.ndarray) -> Mapping[str, float]:
                mixed_w=pop_w.copy(); selected_indices=np.flatnonzero(mask); mixed_w[selected_indices]=match_w[selected_indices]; mixed=_continuous(mixed_w)[:, :length]
                metric = _evaluate_output(method_id="PC-MIXED", output=mixed, observed=observed,
                    matching_projector=heldout.matching.projector, population_projector=prepared.population_context.projector,
                    query_eog=eog, artifactclasses=annotations.artifactclasses[:length], predicted_contamination=None,
                    trial_labels=annotations.trial_labels, samples_per_trial=_sge_samples_per_trial(annotated.sampling_rate_hz),
                    minimum_trials_per_condition=2, status="success", operator_source="oracle_diagnostic", gamma=None,
                    fallback_used=False, uses_query_external_eog=False)
                return {
                    "artifact_utility": float(metric["eog_coherence_reduction"])-float(pop_eval["eog_coherence_reduction"]),
                    "preservation_utility": float(metric["nonartifact_observation_preservation"])-float(pop_eval["nonartifact_observation_preservation"]),
                    "psd_utility": float(pop_eval["reference_free_psd_distortion"])-float(metric["reference_free_psd_distortion"]),
                    "covariance_utility": float(pop_eval["reference_free_covariance_distortion"])-float(metric["reference_free_covariance_distortion"]),
                    "erp_utility": float(metric["condition_erp_observation_relative_preservation"])-float(pop_eval["condition_erp_observation_relative_preservation"]),
                    "output_input_RMS_ratio": float(np.sqrt(np.mean(mixed*mixed))/max(np.sqrt(np.mean(observed*observed)),1e-12)),
                }
            maximum=0.0
            for coverage in coverages:
                chosen, metrics = _choose(benefit, risk, int(round(coverage*windows)), evaluate, margin)
                if chosen is not None: maximum=max(maximum,float(chosen.mean()))
                rows.append({"dataset":"sgeyesub","unit_id":unit,"exact_cell":f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}",
                    "coverage_requested":coverage,"coverage_achieved":float(chosen.mean()) if chosen is not None else 0.0,
                    "maximum_safe_coverage":maximum,"status":"success" if chosen is not None else "infeasible_safety_constraint",
                    **(dict(metrics) if metrics is not None else {key:float("nan") for key in ("artifact_utility","preservation_utility","psd_utility","covariance_utility","erp_utility","output_input_RMS_ratio")})})
    return rows


def run(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    output = CODE_ROOT / str(config["pc_output_root"]); arrays = Path(str(config["v3_result_root"])) / "server_arrays"
    coverages=[float(v) for v in config["evaluation"]["pc_coverages"]]; margin=float(config["evaluation"]["noninferiority_margin"])
    rows=_klados_units(config,arrays,coverages,margin)+_sge_units(config,arrays,coverages,margin)
    _write_csv(output/"unit_metrics.csv",rows)
    summaries=[]; go=False; feasibility=True
    for dataset in ("klados","sgeyesub"):
        for coverage in coverages:
            selected=[row for row in rows if row["dataset"]==dataset and float(row["coverage_requested"])==coverage]
            successful=[row for row in selected if row["status"]=="success"]
            values=np.asarray([float(row["artifact_utility"]) for row in successful])
            ci=_bootstrap(values,int(config["evaluation"]["bootstrap_repetitions"]),20260806,
                          [str(row["exact_cell"]) for row in successful] if dataset=="sgeyesub" else None)
            summaries.append({"dataset":dataset,"coverage":coverage,"denominator":len(selected),"successful":len(successful),
                "mean_utility":float(values.mean()) if values.size else float("nan"),"median_utility":float(np.median(values)) if values.size else float("nan"),
                "ci95_low":ci[0],"ci95_high":ci[1],"positive_count":int(np.sum(values>0)),
                "mean_preservation_utility":float(np.mean([float(r["preservation_utility"]) for r in successful])) if successful else float("nan"),
                "mean_psd_utility":float(np.mean([float(r["psd_utility"]) for r in successful])) if successful else float("nan"),
                "mean_covariance_utility":float(np.mean([float(r["covariance_utility"]) for r in successful])) if successful else float("nan")})
            feasibility &= bool(successful) if coverage in (0.5,0.8) else True
    _write_csv(output/"bootstrap_summary.csv",summaries)
    for coverage in coverages:
        pair=[r for r in summaries if r["coverage"]==coverage]
        if len(pair)==2 and all(float(r["mean_utility"])>0 and float(r["mean_preservation_utility"])>=margin and float(r["mean_psd_utility"])>=margin and float(r["mean_covariance_utility"])>=margin for r in pair): go=True
    decision="GO_MINIMAL_SELECTOR" if go else "NO_GO_CONSTRAINED_CEILING" if feasibility else "INDETERMINATE_CONSTRAINT_FEASIBILITY"
    summary={"status":"completed_safety_constrained_oracle","decision":decision,"legacy_result":"legacy_unconstrained_oracle_not_safety_constrained",
             "coverages":coverages,"klados_denominator":16,"sge_denominator":59,"query_outcomes_role":"oracle_diagnostic_only_not_deployable",
             "mixed_outputs_reevaluated_with_primary_evaluator":True,"minimal_selector_authorized":go}
    output.mkdir(parents=True,exist_ok=True); (output/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    report=CODE_ROOT/"reports/pc_safety_constrained_oracle.md"; report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text("# P-C safety-constrained oracle\n\n"+f"Decision: `{decision}`. Historical unconstrained output remains available and is now named `legacy_unconstrained_oracle_not_safety_constrained`. Mixed POP/MATCH waveforms were re-evaluated under clean-target (Klados) or artifactclass-specific/rest-only (SGE) metrics.\n",encoding="utf-8")
    run_dir.mkdir(parents=True,exist_ok=True); (run_dir/"result_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary
