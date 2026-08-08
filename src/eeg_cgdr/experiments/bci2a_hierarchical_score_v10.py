"""BCI-IV-2a calibration-conditioned hierarchical Score-LoRA V10."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from eeg_cgdr.data.bci2a_v10 import discover_sessions, inspect_gdf, load_gdf_channels, load_with_events, loso_manifest
from eeg_cgdr.models.artifact_subspace_diffusion import aligned_artifact_basis
from eeg_cgdr.models.artifact_subspace_diffusion import ArtifactSubspaceConfig, ArtifactSubspaceDiffusion, DeterministicSubspaceEstimator, bounded_subspace_target, reconstruct_from_subspace, training_tau, window_noise_bank
from eeg_cgdr.models.hierarchical_score_lora import inject_hierarchical_score_lora, set_hierarchical_alpha, shared_direction_parameters


def _config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stage_inventory(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    data_root = Path(str(config["data_root"]))
    sessions = discover_sessions(data_root)
    rows = [inspect_gdf(session) for session in sessions]
    complete = len(rows) == 18 and {(int(r["subject"]), str(r["session"])) for r in rows} == {(s, x) for s in range(1, 10) for x in "TE"}
    registry = {
        "dataset": "BCI Competition IV-2a / BNCI 001-2014",
        "official_source": "https://www.bbci.de/competition/iv/download/",
        "data_root": str(data_root),
        "sessions": len(rows),
        "subjects": len({int(r["subject"]) for r in rows}),
        "complete_18_gdf": complete,
        "expected_channels": {"EEG": 22, "EOG": 3},
        "expected_sampling_rate": 250,
        "sessions_detail": rows,
    }
    _json(Path("datasets/registry/bci2a_001_2014.json"), registry)
    _csv(Path("datasets/splits/bci2a_loso_v10.csv"), loso_manifest())
    status = "completed_inventory" if complete else "blocked_incomplete_official_gdf"
    summary = {"status": status, "sessions": len(rows), "subjects": registry["subjects"], "split_frozen_before_query_outcomes": True}
    _json(run_dir / "result_summary.json", summary)
    if not complete:
        raise RuntimeError(f"BCI2a inventory incomplete: {len(rows)}/18 sessions")
    return summary


def _method(rows: list[dict[str, str]], name: str) -> float:
    values = [float(r["rrmse"]) for r in rows if r["method"] == name]
    return float(np.mean(values)) if values else float("nan")


def stage_sge_closure(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    source = Path(str(config["v9r_root"]))
    new = source / "new_fold_extrapolation"
    paired: list[dict[str, str]] = []
    natural: list[dict[str, str]] = []
    repaired = Path("results/cgdr/sge_v9r_mechanism_closure/factorial_replay/metrics")
    metric_root = repaired if list(repaired.glob("*_paired.csv")) else new / "metrics"
    for path in sorted(metric_root.glob("*_paired.csv")): paired.extend(_read(path))
    for path in sorted((new / "metrics").glob("*_natural.csv")): natural.extend(_read(path))
    coverage = {r["recording_key"]: r for r in _read(source / "support_pair_coverage.csv")}
    effects: list[dict[str, object]] = []
    for key in sorted({r["recording_key"] for r in paired}):
        seed_effects = []
        for seed in sorted({int(r["adaptation_seed"]) for r in paired if r["recording_key"] == key}):
            rows = [r for r in paired if r["recording_key"] == key and int(r["adaptation_seed"]) == seed]
            d00, d10, d01, d11 = (_method(rows, name) for name in ("DIFF-D00", "DIFF-D10", "DIFF-D01", "DIFF-D11"))
            wrong = [float(r["rrmse"]) for r in rows if r["method"].startswith("DIFF-D11-WRONG-")]
            shuffled = _method(rows, "DIFF-D11-SHUFFLED-BOTH")
            seed_effects.append({"G": d00-d10, "A": d00-d01, "C": d00-d11, "I": d10+d01-d00-d11, "U_W": float(np.mean(wrong))-d11 if wrong else np.nan, "U_S": shuffled-d11})
        c = coverage.get(key, {})
        strict = all(int(c.get(field, 0) or 0) >= 4 for field in ("adapt_unique_clean","adapt_unique_artifact","validation_unique_clean","validation_unique_artifact"))
        effects.append({"recording_key": key, "study": key.split("/")[0], "strict_support_diversity": int(strict), **{name: float(np.nanmean([r[name] for r in seed_effects])) for name in ("G","A","C","I","U_W","U_S")}, "wrong_specificity_available": int(any(np.isfinite(r["U_W"]) for r in seed_effects))})
    _csv(Path("results/cgdr/sge_v9r_mechanism_closure/factorial_effects.csv"), effects)
    sensitivity=[]
    for key in sorted({r["recording_key"] for r in effects}):
        item=coverage.get(key,{})
        sensitivity.append({
            "recording_key":key,
            "study":key.split("/")[0],
            "old_builder_eligible":int(item.get("old_builder_eligible",0) or 0),
            "blocked_builder_eligible":int(item.get("new_builder_eligible",0) or 0),
            "adapt_unique_clean":int(item.get("adapt_unique_clean",0) or 0),
            "adapt_unique_artifact":int(item.get("adapt_unique_artifact",0) or 0),
            "validation_unique_clean":int(item.get("validation_unique_clean",0) or 0),
            "validation_unique_artifact":int(item.get("validation_unique_artifact",0) or 0),
        })
    _csv(Path("results/cgdr/sge_v9r_mechanism_closure/builder_sensitivity.csv"),sensitivity)
    eligible_safety = []
    eligible_keys = {r["recording_key"] for r in effects if r["strict_support_diversity"] == 1}
    for key in sorted(eligible_keys):
        rows = [r for r in natural if r["recording_key"] == key]
        if rows:
            eligible_safety.append({"recording_key": key, **{field: float(np.mean([float(r[field]) for r in rows])) for field in ("eog_coherence_reduction","nonartifact_preservation","psd_distortion","covariance_distortion")}})
    _csv(Path("results/cgdr/sge_v9r_mechanism_closure/eligible_safety.csv"), eligible_safety)
    means = {
        name: float(np.mean([float(r[name]) for r in effects if np.isfinite(float(r[name]))]))
        if any(np.isfinite(float(r[name])) for r in effects) else float("nan")
        for name in ("G", "A", "C", "I")
    }
    if means["G"] <= 0 and means["A"] <= 0: label = "BOTH_NEGATIVE"
    elif means["G"] <= 0: label = "GEOMETRY_FAILURE"
    elif means["A"] <= 0: label = "SCORE_ADAPTER_FAILURE"
    else: label = "NONADDITIVE_BUT_NOT_PERSONALIZED"
    safety_summary = {field: {"mean": float(np.mean([r[field] for r in eligible_safety])), "p95": float(np.quantile([r[field] for r in eligible_safety], .95)), "maximum": float(np.max([r[field] for r in eligible_safety]))} for field in ("psd_distortion","covariance_distortion")} if eligible_safety else {}
    summary = {"status": "completed_sge_mechanism_closure", "decision": label, "means": means, "units": len(effects), "strict_support_diversity_units": len(eligible_keys), "wrong_missing_units": sum(1-int(r["wrong_specificity_available"]) for r in effects), "eligible_only_distortion": safety_summary, "builder_label": "support_diversity_experiment_not_coverage_repair", "v9r_no_go_unchanged": True}
    _json(Path("results/cgdr/sge_v9r_mechanism_closure/result_summary.json"), summary)
    Path("reports/sge_v9r_mechanism_closure.md").write_text(f"# SGE V9R mechanism closure\n\nThis support-only common-random replay reused the frozen V9R population backbones and new-fold prepared arrays; it retrained only the support adapters needed to expose D10/D01. The V9R no-go remains unchanged. The blocked builder is a **support-diversity experiment**, not a coverage repair.\n\nDecision: `{label}`. Mean geometry G={means['G']:+.4f}, score-adapter A={means['A']:+.4f}, combined C={means['C']:+.4f}, interaction I={means['I']:+.4f}. WRONG specificity is NA when no donor exists and excluded from its denominator. Eligible-only distortion risk is reported with p95/maximum in the result JSON. No SGE expansion is authorized, and this closure is not a family-wide negative.\n", encoding="utf-8")
    _json(run_dir / "result_summary.json", summary)
    return summary


def _v9r_config(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(config["v9r_config"]))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def stage_sge_factorial_replay(
    config: Mapping[str, Any], task_index: int, run_dir: Path
) -> dict[str, Any]:
    """Fill the new-fold D10/D01 cells with paired adaptation randomness.

    Historical V9R checkpoints and arrays remain read-only.  This replay uses
    the unified blocked support builder and writes only into the V10 closure
    root.  Evaluator fields are not opened in this GPU stage.
    """

    import torch
    from eeg_cgdr.experiments import sge_basis_score_factorial_v9r as v9r
    from eeg_cgdr.models.adaptation_replay import AdaptationReplay

    vcfg = _v9r_config(config)
    fold = v9r._new_fold(vcfg, task_index)
    fold_id = fold["fold_id"]
    arrays = v9r._new_arrays(vcfg, fold_id)
    checkpoint = v9r._new_checkpoint(vcfg, fold_id)
    device = torch.device("cuda")
    base_det, base_diff = v9r._load_models(checkpoint, device)
    pop = np.asarray(checkpoint["population_basis"], np.float32)
    tau = np.asarray(checkpoint["tau"], np.float32)
    historical = v9r._new_fold_root(vcfg)
    root = Path("results/cgdr/sge_v9r_mechanism_closure/factorial_replay")
    support: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, object]] = []
    for position, key in enumerate(fold["heldout"]):
        try:
            support[key] = v9r._new_support(
                vcfg, fold, arrays, pop, tau, key, int(vcfg["pair_seed"]) + position
            )
        except ValueError as exc:
            support[key] = {"eligible": False, "reason": str(exc)}

    for adaptation_seed in v9r.ADAPTATION_SEEDS:
        cache: dict[str, dict[str, Any]] = {}
        for key, item in support.items():
            if not item["eligible"]:
                continue
            replay_path = historical / "replays" / fold_id / (
                f"{v9r._safe(key)}_seed{adaptation_seed}.npz"
            )
            replay = AdaptationReplay.load(replay_path)
            d01, m01 = v9r._adapt(
                vcfg, checkpoint, "diff", item["adapt"], item["validation"],
                pop, np.ones(2, bool), tau, replay, device,
            )
            d11, m11 = v9r._adapt(
                vcfg, checkpoint, "diff", item["adapt"], item["validation"],
                item["match"], item["match_mask"], tau, replay, device,
            )
            u01, _ = v9r._adapt(
                vcfg, checkpoint, "det", item["adapt"], item["validation"],
                pop, np.ones(2, bool), tau, replay, device,
            )
            u11, _ = v9r._adapt(
                vcfg, checkpoint, "det", item["adapt"], item["validation"],
                item["match"], item["match_mask"], tau, replay, device,
            )
            sd11, _ = v9r._adapt(
                vcfg, checkpoint, "diff", item["shuffled_adapt"],
                item["shuffled_validation"], item["shuffled"],
                item["shuffled_mask"], tau, replay, device,
            )
            cache[key] = {
                "d01": d01, "d11": d11, "u01": u01, "u11": u11,
                "sd11": sd11, "m01": m01, "m11": m11,
            }

        for key, item in support.items():
            inp = np.load(
                historical / "deployable_inputs" / fold_id /
                f"paired_{v9r._safe(key)}.npz"
            )
            y = np.asarray(inp["y"])
            if not item["eligible"]:
                outputs = v9r._fallback_outputs(
                    base_det, base_diff, y, pop, tau, key, adaptation_seed, device
                )
                active, selected = False, 0
            else:
                c = cache[key]
                rank_pop = np.ones(2, bool)
                outputs = {
                    "RAW": y,
                    "DET-D00": v9r._output(base_det, "det", y, pop, rank_pop, tau, key, adaptation_seed, device),
                    "DET-D10": v9r._output(base_det, "det", y, item["match"], item["match_mask"], tau, key, adaptation_seed, device),
                    "DET-D01": v9r._output(c["u01"], "det", y, pop, rank_pop, tau, key, adaptation_seed, device),
                    "DET-D11": v9r._output(c["u11"], "det", y, item["match"], item["match_mask"], tau, key, adaptation_seed, device),
                    "DIFF-D00": v9r._output(base_diff, "diff", y, pop, rank_pop, tau, key, adaptation_seed, device),
                    "DIFF-D10": v9r._output(base_diff, "diff", y, item["match"], item["match_mask"], tau, key, adaptation_seed, device),
                    "DIFF-D01": v9r._output(c["d01"], "diff", y, pop, rank_pop, tau, key, adaptation_seed, device),
                    "DIFF-D11": v9r._output(c["d11"], "diff", y, item["match"], item["match_mask"], tau, key, adaptation_seed, device),
                    "DIFF-D11-SHUFFLED-BOTH": v9r._output(c["sd11"], "diff", y, item["shuffled"], item["shuffled_mask"], tau, key, adaptation_seed, device),
                }
                donors = [d for d in fold["heldout"] if d != key and support[d]["eligible"]]
                for donor_index, donor in enumerate(donors):
                    donor_item, donor_cache = support[donor], cache[donor]
                    outputs[f"DIFF-D11-WRONG-BOTH-{donor_index}"] = v9r._output(
                        donor_cache["d11"], "diff", y, donor_item["match"],
                        donor_item["match_mask"], tau, key, adaptation_seed, device,
                    )
                active = bool(c["m11"]["adapter_active"])
                selected = int(c["m11"]["selected_step"])
            outdir = root / "outputs" / fold_id
            outdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                outdir / f"paired_{v9r._safe(key)}_seed{adaptation_seed}.npz",
                **outputs,
            )
            metadata.append({
                "fold_id": fold_id, "study": fold["study"],
                "recording_key": key, "adaptation_seed": adaptation_seed,
                "personalization_eligible": int(item["eligible"]),
                "adapter_active": int(active), "selected_step": selected,
                "fallback_reason": "" if item["eligible"] else item["reason"],
            })
    _csv(root / "adaptation_metadata" / f"{fold_id}.csv", metadata)
    summary = {
        "status": "completed_sge_factorial_replay", "fold_id": fold_id,
        "units": len(fold["heldout"]),
        "eligible_units": sum(int(v["eligible"]) for v in support.values()),
        "adaptation_seeds": list(v9r.ADAPTATION_SEEDS),
        "inactive_rank_loss_masked": True, "query_outcomes_opened": False,
    }
    _json(run_dir / "result_summary.json", summary)
    return summary


def stage_sge_factorial_eval(
    config: Mapping[str, Any], task_index: int, run_dir: Path
) -> dict[str, Any]:
    from eeg_cgdr.experiments import sge_basis_score_factorial_v9r as v9r

    vcfg = _v9r_config(config)
    fold = v9r._new_fold(vcfg, task_index)
    fold_id = fold["fold_id"]
    historical = v9r._new_fold_root(vcfg)
    root = Path("results/cgdr/sge_v9r_mechanism_closure/factorial_replay")
    rows: list[dict[str, object]] = []
    for key in fold["heldout"]:
        evaluator = np.load(
            historical / "evaluator" / fold_id / f"paired_{v9r._safe(key)}.npz"
        )
        for seed in v9r.ADAPTATION_SEEDS:
            output = np.load(
                root / "outputs" / fold_id / f"paired_{v9r._safe(key)}_seed{seed}.npz"
            )
            raw = np.asarray(output["RAW"])
            for method in output.files:
                rows.append({
                    "fold_id": fold_id, "study": fold["study"],
                    "recording_key": key, "adaptation_seed": seed,
                    "method": method,
                    **v9r._paired_metrics(output[method], evaluator["x"], raw),
                })
    _csv(root / "metrics" / f"{fold_id}_paired.csv", rows)
    summary = {"status": "completed_sge_factorial_evaluator", "fold_id": fold_id, "units": len(fold["heldout"])}
    _json(run_dir / "result_summary.json", summary)
    return summary


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0))
RIDGES = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)


def _standardize(value: np.ndarray) -> np.ndarray:
    center = np.median(value, axis=1, keepdims=True)
    scale = np.quantile(np.abs(value-center), .75, axis=1, keepdims=True) / .67448975
    return ((value-center)/np.maximum(scale, 1.0e-8)).astype(np.float64)


def _band_transfer(eeg: np.ndarray, eog: np.ndarray, sfreq: float, ridge: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    result=[]
    for low, high in BANDS:
        sos=butter(4,(low,high),btype="bandpass",fs=sfreq,output="sos")
        y=sosfiltfilt(sos,_standardize(eeg),axis=1);e=sosfiltfilt(sos,_standardize(eog),axis=1)
        gram=e@e.T + float(ridge)*e.shape[1]*np.eye(e.shape[0])
        result.append((y@e.T)@np.linalg.inv(gram))
    return np.stack(result).astype(np.float32)


def _support_query_ranges(events: list[tuple[float,str]], n_times: int, sfreq: float) -> tuple[slice,slice]:
    first_trial=min(onset for onset,description in events if description=="768")
    boundary=int(round(first_trial*sfreq))
    # Five seconds prevent any trial-onset/filter edge from entering support.
    support_stop=max(1,boundary-int(round(5*sfreq)))
    return slice(0,support_stop),slice(boundary,n_times)


def _distance(left: np.ndarray, right: np.ndarray) -> tuple[float,float,float]:
    a=np.asarray(left,float);b=np.asarray(right,float)
    fro=float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-12))
    magnitude=float(np.linalg.norm(np.abs(a)-np.abs(b))/(np.linalg.norm(np.abs(b))+1e-12))
    cosine=float(1-np.dot(a.ravel(),b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    return fro,magnitude,cosine


def stage_headroom_extract(config: Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    subject=task_index+1;sessions={(s.subject,s.session):s for s in discover_sessions(Path(str(config["data_root"])))};payload={}
    for session in "TE":
        eeg,eog,sfreq,events=load_with_events(sessions[(subject,session)]);support,query=_support_query_ranges(events,eeg.shape[1],sfreq);s_mid=(support.start+support.stop)//2
        values={}
        for ridge in RIDGES:
            values[str(ridge)]={"support":_band_transfer(eeg[:,support],eog[:,support],sfreq,ridge),"half_a":_band_transfer(eeg[:,support.start:s_mid],eog[:,support.start:s_mid],sfreq,ridge),"half_b":_band_transfer(eeg[:,s_mid:support.stop],eog[:,s_mid:support.stop],sfreq,ridge),"query":_band_transfer(eeg[:,query],eog[:,query],sfreq,ridge)}
        payload[session]={"support_samples":support.stop-support.start,"query_samples":query.stop-query.start,"sfreq":sfreq,"values":values}
    out=Path(str(config["result_root"]))/"headroom"/"operators"/f"subject_{subject:02d}.npz";out.parent.mkdir(parents=True,exist_ok=True)
    arrays={}
    metadata={"subject":subject,"sessions":{}}
    for session,item in payload.items():
        metadata["sessions"][session]={k:item[k] for k in ("support_samples","query_samples","sfreq")}
        for ridge,values in item["values"].items():
            for role,value in values.items():arrays[f"{session}_{ridge}_{role}"]=value
    np.savez_compressed(out,metadata=np.array(json.dumps(metadata)),**arrays);summary={"status":"completed_headroom_extract","subject":subject,"sessions":2,"query_outcomes_used_for_model":False};_json(run_dir/"result_summary.json",summary);return summary


def _load_ops(root:Path,subject:int)->dict[str,np.ndarray]:
    with np.load(root/"headroom"/"operators"/f"subject_{subject:02d}.npz",allow_pickle=False) as data:return {k:np.asarray(data[k]) for k in data.files if k!="metadata"}


def stage_headroom_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));ops={s:_load_ops(root,s) for s in range(1,10)};rows=[];ridge_rows=[]
    for heldout in range(1,10):
        training=[s for s in range(1,10) if s!=heldout]
        # Select one ridge using only outer-training support split stability.
        scores={ridge:float(np.mean([_distance(ops[s][f"T_{ridge}_half_a"],ops[s][f"T_{ridge}_half_b"])[0]+_distance(ops[s][f"E_{ridge}_half_a"],ops[s][f"E_{ridge}_half_b"])[0] for s in training])) for ridge in map(str,RIDGES)}
        ridge=min(scores,key=scores.get);ridge_rows.append({"heldout_subject":heldout,"ridge":ridge,"outer_training_split_stability":scores[ridge]})
        for protocol,support_session,query_session in (("same_session_T","T","T"),("same_session_E","E","E"),("cross_session","T","E")):
            match=ops[heldout][f"{support_session}_{ridge}_support"];target=ops[heldout][f"{query_session}_{ridge}_query"];pop=np.mean([ops[s][f"{support_session}_{ridge}_support"] for s in training],axis=0);wrong=[ops[s][f"{support_session}_{ridge}_support"] for s in training]
            dmatch=_distance(match,target);dpop=_distance(pop,target);dwrong=[_distance(value,target) for value in wrong];shuffled=np.flip(match,axis=0);dshuffle=_distance(shuffled,target)
            rows.append({"heldout_subject":heldout,"protocol":protocol,"ridge":ridge,"match_distance":dmatch[0],"pop_distance":dpop[0],"mean_wrong_distance":float(np.mean([d[0] for d in dwrong])),"shuffled_distance":dshuffle[0],"match_minus_pop_utility":dpop[0]-dmatch[0],"match_minus_wrong_utility":float(np.mean([d[0] for d in dwrong]))-dmatch[0],"match_minus_shuffled_utility":dshuffle[0]-dmatch[0],"support_half_stability":_distance(ops[heldout][f"{support_session}_{ridge}_half_a"],ops[heldout][f"{support_session}_{ridge}_half_b"])[0],"frequency_magnitude_match":dmatch[1],"cosine_match":dmatch[2],"wrong_donors":len(wrong),"query_operator_evaluator_only":1})
    _csv(root/"headroom"/"participant_effects.csv",rows);_csv(root/"headroom"/"ridge_selection.csv",ridge_rows)
    protocol_summary=[]
    for protocol in ("same_session_T","same_session_E","cross_session"):
        subset=[r for r in rows if r["protocol"]==protocol];wrong=np.asarray([r["match_minus_wrong_utility"] for r in subset],float);pop=np.asarray([r["match_minus_pop_utility"] for r in subset],float)
        protocol_summary.append({"protocol":protocol,"subjects":len(subset),"match_wrong_mean":float(wrong.mean()),"match_wrong_median":float(np.median(wrong)),"match_wrong_positive":int((wrong>0).sum()),"match_pop_mean":float(pop.mean()),"match_pop_median":float(np.median(pop)),"match_pop_positive":int((pop>0).sum())})
    _csv(root/"headroom"/"protocol_summary.csv",protocol_summary)
    eligible=[r for r in protocol_summary if r["match_wrong_mean"]>0 and r["match_wrong_positive"]>=6]
    status="BCI2A_SUBJECT_IDENTIFIABILITY_DETECTED" if eligible else "BCI2A_IDENTIFIABILITY_NOT_DETECTED"
    summary={"status":"completed_identifiability_audit","decision":status,"gpu_training_authorized":bool(eligible),"eligible_protocols":[r["protocol"] for r in eligible],"protocols":protocol_summary,"subjects":9,"query_operator_role":"evaluator_only_ceiling","confirmation":False};_json(root/"headroom"/"routing_decision.json",summary);_json(run_dir/"result_summary.json",summary)
    lines=["# BCI2a subject identifiability V10","","All nine participants were evaluated. Query operators are evaluator-only and never enter deployment inference.","","| protocol | MATCH−WRONG mean | positive | MATCH−POP mean |","|---|---:|---:|---:|"]
    for r in protocol_summary:lines.append(f"| {r['protocol']} | {r['match_wrong_mean']:+.4f} | {r['match_wrong_positive']}/9 | {r['match_pop_mean']:+.4f} |")
    lines += ["",f"Decision: `{status}`. GPU training authorized: `{bool(eligible)}`. This is development headroom, not denoising success or confirmation."]
    Path("reports/bci2a_subject_identifiability_v10.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return summary


def _bci2b_files(config:Mapping[str,Any],subject:int)->dict[str,Path]:
    root=Path(str(config["data_root"]));result={}
    for path in root.rglob(f"B{subject:02d}???.gdf"):
        stem=path.stem
        if len(stem)==6 and stem[:3]==f"B{subject:02d}" and stem[3:5].isdigit() and stem[5] in "TE":result[stem[3:]]=path
    return result


def stage_bci2b_extract(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    subject=task_index+1;files=_bci2b_files(config,subject)
    if len(files)!=5:raise RuntimeError(f"B{subject:02d}: expected five sessions, got {sorted(files)}")
    arrays={};metadata={"subject":subject,"sessions":sorted(files)}
    for session,path in sorted(files.items()):
        eeg,eog,sfreq,events=load_gdf_channels(path,eeg_channels=3);support,query=_support_query_ranges(events,eeg.shape[1],sfreq);mid=(support.start+support.stop)//2
        for ridge in RIDGES:
            for role,region in (("support",support),("half_a",slice(support.start,mid)),("half_b",slice(mid,support.stop)),("query",query)):
                arrays[f"{session}_{ridge}_{role}"]=_band_transfer(eeg[:,region],eog[:,region],sfreq,ridge)
    out=Path(str(config["result_root"]))/"headroom_2b"/"operators"/f"subject_{subject:02d}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,metadata=np.array(json.dumps(metadata)),**arrays);summary={"status":"completed_bci2b_extract","subject":subject,"sessions":5};_json(run_dir/"result_summary.json",summary);return summary


def stage_bci2b_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));folder=root/"headroom_2b"/"operators"
    ops={}
    for subject in range(1,10):
        with np.load(folder/f"subject_{subject:02d}.npz",allow_pickle=False) as data:ops[subject]={k:np.asarray(data[k]) for k in data.files if k!="metadata"}
    rows=[]
    for heldout in range(1,10):
        training=[s for s in range(1,10) if s!=heldout]
        scores={ridge:float(np.mean([_distance(ops[s][f"01T_{ridge}_half_a"],ops[s][f"01T_{ridge}_half_b"])[0] for s in training])) for ridge in map(str,RIDGES)};ridge=min(scores,key=scores.get)
        protocols=[("same_session",session,session) for session in ("01T","02T","03T","04E","05E")]+[("cross_session","03T","04E"),("cross_session","03T","05E")]
        temporary=defaultdict(list)
        for protocol,support_session,query_session in protocols:
            match=ops[heldout][f"{support_session}_{ridge}_support"];target=ops[heldout][f"{query_session}_{ridge}_query"];pop=np.mean([ops[s][f"{support_session}_{ridge}_support"] for s in training],axis=0);wrong=[ops[s][f"{support_session}_{ridge}_support"] for s in training]
            temporary[protocol].append((_distance(pop,target)[0]-_distance(match,target)[0],float(np.mean([_distance(x,target)[0] for x in wrong]))-_distance(match,target)[0]))
        for protocol,values in temporary.items():rows.append({"heldout_subject":heldout,"protocol":protocol,"match_pop_utility":float(np.mean([v[0] for v in values])),"match_wrong_utility":float(np.mean([v[1] for v in values])),"sessions":len(values)})
    _csv(root/"headroom_2b"/"participant_effects.csv",rows);summary_rows=[]
    for protocol in ("same_session","cross_session"):
        subset=[r for r in rows if r["protocol"]==protocol];v=np.asarray([r["match_wrong_utility"] for r in subset]);p=np.asarray([r["match_pop_utility"] for r in subset]);summary_rows.append({"protocol":protocol,"subjects":9,"match_wrong_mean":float(v.mean()),"match_wrong_median":float(np.median(v)),"match_wrong_positive":int((v>0).sum()),"match_pop_mean":float(p.mean()),"match_pop_positive":int((p>0).sum())})
    _csv(root/"headroom_2b"/"protocol_summary.csv",summary_rows);eligible=[r for r in summary_rows if r["match_wrong_mean"]>0 and r["match_wrong_positive"]>=6];decision="BCI2B_SUBJECT_IDENTIFIABILITY_DETECTED" if eligible else "CONTAMINATION_TRANSFER_REPRESENTATION_STOPPED_AFTER_BCI2A_AND_BCI2B"
    summary={"status":"completed_bci2b_identifiability_audit","decision":decision,"gpu_training_authorized":bool(eligible),"protocols":summary_rows,"subjects":9,"bci2a_gpu_training_remains_blocked":True,"family_wide_status":"not_tested"};_json(root/"headroom_2b"/"routing_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def _instant_transfer(eeg:np.ndarray,eog:np.ndarray)->np.ndarray:
    y=eeg-np.mean(eeg,axis=1,keepdims=True);e=eog-np.mean(eog,axis=1,keepdims=True);return ((y@e.T)@np.linalg.inv(e@e.T+1e-2*e.shape[1]*np.eye(e.shape[0]))).astype(np.float32)


def _window_set(eeg:np.ndarray,eog:np.ndarray,start:int,stop:int,*,length:int=500)->tuple[np.ndarray,np.ndarray]:
    starts=np.arange(start,max(start,stop-length+1),length,dtype=int)
    if starts.size==0:return np.empty((0,eeg.shape[0],length),np.float32),np.empty((0,eog.shape[0],length),np.float32)
    return np.stack([eeg[:,i:i+length] for i in starts]),np.stack([eog[:,i:i+length] for i in starts])


def _pairs_from_region(eeg:np.ndarray,eog:np.ndarray,start:int,stop:int,seed:int,cap:int)->dict[str,np.ndarray]:
    clean,eogs=_window_set(eeg,eog,start,stop);energy=np.sqrt(np.mean(eogs.astype(float)**2,axis=(1,2)));order=np.argsort(energy);count=max(4,int(len(order)*.3));low=order[:count];high=order[-count:];rng=np.random.default_rng(seed);rng.shuffle(low);rng.shuffle(high);n=min(len(low),len(high),cap);low=low[:n];high=np.roll(high[:n],seed%max(n,1));transfer=_instant_transfer(eeg[:,start:stop],eog[:,start:stop]);artifact=np.einsum("ce,net->nct",transfer,eogs[high]);x=clean[low];y=x+artifact
    def pad(value:np.ndarray)->np.ndarray:return np.pad(value,((0,0),(0,0),(0,12))).astype(np.float32)
    return {"x":pad(x),"y":pad(y),"a":pad(artifact),"eog":pad(eogs[high]),"unique_low":np.array(len(set(map(int,low)))) ,"unique_high":np.array(len(set(map(int,high)))) ,"transfer":transfer}


def _query_trials(eeg:np.ndarray,eog:np.ndarray,events:list[tuple[float,str]],sfreq:float)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    rows=[]
    for onset,description in events:
        if description in ("769","770"):
            start=int(round((onset+.5)*sfreq));stop=start+500
            if stop<=eeg.shape[1]:rows.append((eeg[:,start:stop],eog[:,start:stop],int(description)-769))
    if not rows:return np.empty((0,eeg.shape[0],512),np.float32),np.empty((0,eog.shape[0],512),np.float32),np.empty(0,int)
    return np.pad(np.stack([r[0] for r in rows]),((0,0),(0,0),(0,12))).astype(np.float32),np.pad(np.stack([r[1] for r in rows]),((0,0),(0,0),(0,12))).astype(np.float32),np.asarray([r[2] for r in rows],int)


def stage_prepare_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    route=json.loads((Path(str(config["result_root"]))/"headroom_2b"/"routing_decision.json").read_text())
    if not route["gpu_training_authorized"]:raise RuntimeError("2b identifiability gate did not authorize preparation")
    heldout=task_index+1;training=[s for s in range(1,10) if s!=heldout];root=Path(str(config["result_root"]))/"bci2b_gpu"/f"fold_{task_index:02d}";root.mkdir(parents=True,exist_ok=True);train_parts=[];transfers=[]
    for subject in training:
        for session,path in sorted(_bci2b_files(config,subject).items()):
            if not session.endswith("T"):continue
            eeg,eog,sfreq,events=load_gdf_channels(path,eeg_channels=3);eeg=eeg*1.0e6;eog=eog*1.0e6;support,_=_support_query_ranges(events,eeg.shape[1],sfreq);pairs=_pairs_from_region(eeg,eog,support.start,support.stop,int(config["seed"])+subject*100+int(session[:2]),64);pairs["subject"]=np.full(len(pairs["y"]),subject,np.int16);train_parts.append(pairs);transfers.append(pairs["transfer"])
    population_transfer=np.mean(transfers,axis=0);basis,_,rank=aligned_artifact_basis(population_transfer);training_arrays={name:np.concatenate([p[name] for p in train_parts]) for name in ("x","y","a","subject")};np.savez_compressed(root/"training_pairs.npz",**training_arrays,population_basis=basis,rank_mask=rank)
    units=[]
    heldout_files=_bci2b_files(config,heldout)
    for protocol,support_session,query_session in (("same_01","01T","01T"),("same_02","02T","02T"),("same_03","03T","03T"),("cross_02","01T","02T"),("cross_03","01T","03T")):
        seeg,seog,ssf,sevents=load_gdf_channels(heldout_files[support_session],eeg_channels=3);seeg=seeg*1.0e6;seog=seog*1.0e6;support,_=_support_query_ranges(sevents,seeg.shape[1],ssf);mid=(support.start+support.stop)//2;adapt=_pairs_from_region(seeg,seog,support.start,mid,int(config["seed"])+heldout,32);valid=_pairs_from_region(seeg,seog,mid,support.stop,int(config["seed"])+heldout+1,32);qeeg,qeog,qsf,qevents=load_gdf_channels(heldout_files[query_session],eeg_channels=3);qeeg=qeeg*1.0e6;qeog=qeog*1.0e6;natural,eog_windows,labels=_query_trials(qeeg,qeog,qevents,qsf);energy=np.sqrt(np.mean(eog_windows.astype(float)**2,axis=(1,2)));order=np.argsort(energy);n=min(64,len(order)//3);clean=natural[order[:n]];artifact_eog=eog_windows[order[-n:]];qtransfer=_instant_transfer(qeeg,qeog);artifact=np.einsum("ce,net->nct",qtransfer,artifact_eog);paired_y=clean+artifact
        unit=root/"units"/protocol;unit.mkdir(parents=True,exist_ok=True);np.savez_compressed(unit/"support.npz",adapt_y=adapt["y"],adapt_a=adapt["a"],valid_y=valid["y"],valid_a=valid["a"],unique_low=adapt["unique_low"],unique_high=adapt["unique_high"]);np.savez_compressed(unit/"deployable.npz",paired_y=paired_y,natural_y=natural);np.savez_compressed(unit/"evaluator.npz",paired_x=clean,paired_a=artifact,natural_eog=eog_windows,labels=labels);units.append({"protocol":protocol,"support_session":support_session,"query_session":query_session,"adapt_pairs":len(adapt["y"]),"validation_pairs":len(valid["y"]),"paired_windows":len(clean),"natural_trials":len(natural)})
    _csv(root/"unit_manifest.csv",units);summary={"status":"completed_bci2b_fold_preparation","fold":task_index,"heldout_subject":heldout,"training_pairs":len(training_arrays["y"]),"units":len(units),"query_outcomes_in_deployable":False};_json(run_dir/"result_summary.json",summary);return summary


def _condition(y, basis, rank, device):
    import torch
    batch=len(y);return {"observed":torch.as_tensor(y,device=device),"basis":torch.as_tensor(np.repeat(basis[None],batch,axis=0),device=device),"reliability":torch.ones(batch,device=device),"rank_mask":torch.as_tensor(np.repeat(rank[None],batch,axis=0),device=device),"valid_time_mask":torch.ones(batch,1,y.shape[-1],dtype=torch.bool,device=device)}


def stage_train_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    from torch.optim import AdamW
    torch.manual_seed(int(config["seed"])+task_index);np.random.seed(int(config["seed"])+task_index);device=torch.device("cuda");root=Path(str(config["result_root"]))/"bci2b_gpu"/f"fold_{task_index:02d}"
    with np.load(root/"training_pairs.npz") as data:y=np.asarray(data["y"],np.float32);a=np.asarray(data["a"],np.float32);subjects=np.asarray(data["subject"],int);basis=np.asarray(data["population_basis"],np.float32);rank=np.asarray(data["rank_mask"],bool)
    coefficients=np.einsum("cr,nct->nrt",basis,a);tau=training_tau(coefficients);target=bounded_subspace_target(a,np.repeat(basis[None],len(a),axis=0),tau);cfg=ArtifactSubspaceConfig(eeg_channels=3,signal_length=512,base_channels=16);det=DeterministicSubspaceEstimator(cfg).to(device);diff=ArtifactSubspaceDiffusion(cfg).to(device);det_opt=AdamW(det.parameters(),lr=2e-4);diff_opt=AdamW(diff.parameters(),lr=2e-4);rng=np.random.default_rng(int(config["seed"])+task_index);generator=torch.Generator(device=device).manual_seed(int(config["seed"])+task_index);curve=[]
    for step in range(3000):
        idx=rng.integers(0,len(y),32);condition=_condition(y[idx],basis,rank,device);truth=torch.as_tensor(target[idx],device=device);det_opt.zero_grad(set_to_none=True);prediction=det(**condition);dloss=((prediction-truth)**2).mean();dloss.backward();torch.nn.utils.clip_grad_norm_(det.parameters(),1);det_opt.step();diff_opt.zero_grad(set_to_none=True);floss,_=diff.training_loss(truth,generator=generator,**condition);floss.backward();torch.nn.utils.clip_grad_norm_(diff.parameters(),1);diff_opt.step()
        if step%250==0:curve.append({"step":step,"det_loss":float(dloss.detach()),"diff_loss":float(floss.detach())})
    det_summary=inject_hierarchical_score_lora(det.backbone);diff_summary=inject_hierarchical_score_lora(diff.backbone);subject_ids=sorted(set(map(int,subjects)));lookup={s:i for i,s in enumerate(subject_ids)};det_embed=torch.nn.Parameter(torch.randn(len(subject_ids),4,device=device)*.05);diff_embed=torch.nn.Parameter(torch.randn(len(subject_ids),4,device=device)*.05);det_shared=AdamW(shared_direction_parameters(det)+[det_embed],lr=1e-4);diff_shared=AdamW(shared_direction_parameters(diff)+[diff_embed],lr=1e-4)
    for step in range(1000):
        idx=rng.integers(0,len(y),32);condition=_condition(y[idx],basis,rank,device);truth=torch.as_tensor(target[idx],device=device);row=torch.as_tensor([lookup[int(s)] for s in subjects[idx]],device=device);da=torch.tanh(det_embed[row]);fa=torch.tanh(diff_embed[row]);set_hierarchical_alpha(det,da);det_shared.zero_grad(set_to_none=True);dloss=((det(**condition)-truth)**2).mean()+1e-3*da.square().mean();dloss.backward();det_shared.step();set_hierarchical_alpha(diff,fa);diff_shared.zero_grad(set_to_none=True);floss,_=diff.training_loss(truth,generator=generator,**condition);floss=floss+1e-3*fa.square().mean();floss.backward();diff_shared.step()
    set_hierarchical_alpha(det,None);set_hierarchical_alpha(diff,None);checkpoint={"config":cfg.__dict__,"det":det.state_dict(),"diff":diff.state_dict(),"basis":basis,"rank_mask":rank,"tau":tau,"det_prior_cov":np.cov(torch.tanh(det_embed).detach().cpu().numpy().T)+1e-3*np.eye(4),"diff_prior_cov":np.cov(torch.tanh(diff_embed).detach().cpu().numpy().T)+1e-3*np.eye(4),"det_shared_parameters":det_summary.shared_parameters,"diff_shared_parameters":diff_summary.shared_parameters};torch.save(checkpoint,root/"checkpoint.pt");_csv(root/"training_curve.csv",curve)
    # Real fixed-batch technical checks.
    diff.eval();condition=_condition(y[:8],basis,rank,device);set_hierarchical_alpha(diff,torch.zeros(8,4,device=device));zero=diff.backbone(torch.zeros(8,2,512,device=device),torch.zeros(8,dtype=torch.long,device=device),**condition);set_hierarchical_alpha(diff,None);pop=diff.backbone(torch.zeros(8,2,512,device=device),torch.zeros(8,dtype=torch.long,device=device),**condition);technical=bool(torch.equal(zero,pop) and torch.isfinite(pop).all());summary={"status":"completed_fold_training" if technical else "technical_validity_failed","fold":task_index,"training_pairs":len(y),"alpha_zero_exact":technical,"shared_directions":4,"det_shared_parameters":det_summary.shared_parameters,"diff_shared_parameters":diff_summary.shared_parameters};_json(run_dir/"result_summary.json",summary);return summary


def _load_hierarchical(checkpoint,kind,device):
    model=DeterministicSubspaceEstimator(ArtifactSubspaceConfig(**checkpoint["config"])).to(device) if kind=="det" else ArtifactSubspaceDiffusion(ArtifactSubspaceConfig(**checkpoint["config"])).to(device);inject_hierarchical_score_lora(model.backbone);model.load_state_dict(checkpoint[kind]);return model


def _adapt_alpha(model,kind,y,a,valid_y,valid_a,basis,rank,tau,prior_cov,seed,device):
    import torch
    from torch.optim import Adam
    alpha=torch.nn.Parameter(torch.zeros(4,device=device));opt=Adam([alpha],lr=5e-2);target=bounded_subspace_target(a,np.repeat(basis[None],len(a),axis=0),tau);valid_target=bounded_subspace_target(valid_a,np.repeat(basis[None],len(valid_a),axis=0),tau);generator=torch.Generator(device=device).manual_seed(seed);model.eval();best=(float("inf"),torch.zeros(4,device=device));fixed_noise=torch.randn(valid_target.shape,generator=torch.Generator(device=device).manual_seed(seed+991),device=device);fixed_t=torch.full((len(valid_y),),500,dtype=torch.long,device=device)
    precision=torch.as_tensor(np.linalg.inv(prior_cov),device=device,dtype=torch.float32)
    for step in range(251):
        if step in (0,50,100,200,250):
            bounded=torch.tanh(alpha);set_hierarchical_alpha(model,bounded);condition=_condition(valid_y,basis,rank,device);truth=torch.as_tensor(valid_target,device=device)
            with torch.no_grad():score=float((((model(**condition)-truth)**2).mean() if kind=="det" else model.training_loss(truth,generator=generator,timestep=fixed_t,noise=fixed_noise,**condition)[0]).cpu())
            if score<best[0]:best=(score,bounded.detach().clone())
        if step==250:break
        idx=torch.arange(min(16,len(y)),device=device);condition=_condition(y[idx.cpu().numpy()],basis,rank,device);truth=torch.as_tensor(target[idx.cpu().numpy()],device=device);bounded=torch.tanh(alpha);set_hierarchical_alpha(model,bounded);opt.zero_grad(set_to_none=True);loss=((model(**condition)-truth)**2).mean() if kind=="det" else model.training_loss(truth,generator=generator,**condition)[0];loss=loss+1e-3*(bounded@(precision@bounded));loss.backward();opt.step()
    value=best[1];set_hierarchical_alpha(model,None);return value


def _predict(model,kind,y,basis,rank,tau,alpha,key,device):
    import torch
    outputs=[];model.eval()
    for start in range(0,len(y),32):
        stop=min(start+32,len(y));condition=_condition(y[start:stop],basis,rank,device);set_hierarchical_alpha(model,alpha)
        with torch.no_grad():
            if kind=="det":u=model(**condition)
            else:bank=window_noise_bank(key,int(20260830),range(start,stop),posterior_samples=8,signal_length=512,device=device);u=model.sample(initial_noise_bank=bank,**condition)[0]
            restored,_=reconstruct_from_subspace(condition["observed"],condition["basis"],u,torch.as_tensor(tau,device=device),condition["rank_mask"],condition["valid_time_mask"]);outputs.append(restored.cpu().numpy())
    set_hierarchical_alpha(model,None);return np.concatenate(outputs)


def stage_infer_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    import torch
    device=torch.device("cuda");root=Path(str(config["result_root"]))/"bci2b_gpu"/f"fold_{task_index:02d}";checkpoint=torch.load(root/"checkpoint.pt",map_location="cpu",weights_only=False);basis=np.asarray(checkpoint["basis"]);rank=np.asarray(checkpoint["rank_mask"]);tau=np.asarray(checkpoint["tau"]);det=_load_hierarchical(checkpoint,"det",device);diff=_load_hierarchical(checkpoint,"diff",device);manifest=_read(root/"unit_manifest.csv");training=np.load(root/"training_pairs.npz");donors=sorted(set(map(int,training["subject"])))[:3]
    for unit in manifest:
        name=unit["protocol"];support=np.load(root/"units"/name/"support.npz");deploy=np.load(root/"units"/name/"deployable.npz");eligible=min(int(support["unique_low"]),int(support["unique_high"]))>=4
        if eligible:
            det_alpha=_adapt_alpha(det,"det",support["adapt_y"],support["adapt_a"],support["valid_y"],support["valid_a"],basis,rank,tau,checkpoint["det_prior_cov"],int(config["seed"])+task_index,device);diff_alpha=_adapt_alpha(diff,"diff",support["adapt_y"],support["adapt_a"],support["valid_y"],support["valid_a"],basis,rank,tau,checkpoint["diff_prior_cov"],int(config["seed"])+task_index,device);shuffle_a=np.asarray(support["adapt_a"])[::-1].copy();shuffle_valid=np.asarray(support["valid_a"])[::-1].copy();shuffle_alpha=_adapt_alpha(diff,"diff",support["adapt_y"],shuffle_a,support["valid_y"],shuffle_valid,basis,rank,tau,checkpoint["diff_prior_cov"],int(config["seed"])+task_index,device)
        else:det_alpha=diff_alpha=shuffle_alpha=torch.zeros(4,device=device)
        donor_alphas=[]
        for donor in donors:
            idx=np.flatnonzero(training["subject"]==donor)[:64];donor_alphas.append(_adapt_alpha(diff,"diff",training["y"][idx[:32]],training["a"][idx[:32]],training["y"][idx[32:]],training["a"][idx[32:]],basis,rank,tau,checkpoint["diff_prior_cov"],int(config["seed"])+donor,device))
        for split in ("paired_y","natural_y"):
            y=np.asarray(deploy[split]);outputs={"RAW":y,"DET-POP":_predict(det,"det",y,basis,rank,tau,torch.zeros(4,device=device),f"{task_index}-{name}-{split}",device),"DET-MATCH":_predict(det,"det",y,basis,rank,tau,det_alpha,f"{task_index}-{name}-{split}",device),"DIFF-POP":_predict(diff,"diff",y,basis,rank,tau,torch.zeros(4,device=device),f"{task_index}-{name}-{split}",device),"DIFF-MATCH":_predict(diff,"diff",y,basis,rank,tau,diff_alpha,f"{task_index}-{name}-{split}",device),"DIFF-SHUFFLED":_predict(diff,"diff",y,basis,rank,tau,shuffle_alpha,f"{task_index}-{name}-{split}",device)}
            for index,alpha in enumerate(donor_alphas):outputs[f"DIFF-WRONG-{index}"]=_predict(diff,"diff",y,basis,rank,tau,alpha,f"{task_index}-{name}-{split}",device)
            out=root/"outputs"/name;out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f"{split}.npz",**outputs)
        _json(root/"outputs"/name/"metadata.json",{"eligible":eligible,"det_alpha":det_alpha.cpu().tolist(),"diff_alpha":diff_alpha.cpu().tolist(),"wrong_donors":donors,"query_outcomes_opened":False})
    summary={"status":"completed_fold_inference","fold":task_index,"units":len(manifest)};_json(run_dir/"result_summary.json",summary);return summary


def _rrmse(output:np.ndarray,target:np.ndarray)->float:return float(np.linalg.norm(output-target)/(np.linalg.norm(target)+1e-12))


def _corr(output:np.ndarray,target:np.ndarray)->float:
    a=output.ravel()-output.mean();b=target.ravel()-target.mean();return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))


def _coherence_proxy(eeg:np.ndarray,eog:np.ndarray)->float:
    values=[]
    for c in range(eeg.shape[1]):
        for e in range(eog.shape[1]):values.append(abs(_corr(eeg[:,c],eog[:,e])))
    return float(np.mean(values))


def _distortions(output:np.ndarray,raw:np.ndarray)->tuple[float,float]:
    from scipy.signal import welch
    _,p0=welch(raw,fs=250,nperseg=256,axis=-1);_,p1=welch(output,fs=250,nperseg=256,axis=-1);psd=float(np.linalg.norm(np.log(p1+1e-12)-np.log(p0+1e-12))/(np.linalg.norm(np.log(p0+1e-12))+1e-12));c0=np.cov(raw.transpose(1,0,2).reshape(raw.shape[1],-1));c1=np.cov(output.transpose(1,0,2).reshape(output.shape[1],-1));cov=float(np.linalg.norm(c1-c0)/(np.linalg.norm(c0)+1e-12));return psd,cov


def _features(value:np.ndarray)->np.ndarray:
    from scipy.signal import butter,sosfiltfilt
    filtered=sosfiltfilt(butter(4,(8,30),btype="bandpass",fs=250,output="sos"),value,axis=-1);return np.log(np.var(filtered,axis=-1)+1e-12)


def stage_eval_fold(config:Mapping[str,Any],task_index:int,run_dir:Path)->dict[str,Any]:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import cohen_kappa_score
    root=Path(str(config["result_root"]))/"bci2b_gpu"/f"fold_{task_index:02d}";manifest=_read(root/"unit_manifest.csv");train_eval=np.load(root/"units"/"same_01"/"evaluator.npz");train_raw=np.load(root/"outputs"/"same_01"/"natural_y.npz")["RAW"];split=max(2,len(train_raw)//2);classifier=LinearDiscriminantAnalysis().fit(_features(train_raw[:split]),train_eval["labels"][:split]);paired_rows=[];natural_rows=[];wrong_rows=[]
    for unit in manifest:
        name=unit["protocol"];evaluation=np.load(root/"units"/name/"evaluator.npz");paired=np.load(root/"outputs"/name/"paired_y.npz");natural=np.load(root/"outputs"/name/"natural_y.npz");raw=natural["RAW"];eog=evaluation["natural_eog"];energy=np.sqrt(np.mean(eog.astype(float)**2,axis=(1,2)));low=energy<=np.quantile(energy,.3);base_coh=_coherence_proxy(raw,eog);metadata=json.loads((root/"outputs"/name/"metadata.json").read_text())
        for method in paired.files:
            output=paired[method];r=_rrmse(output,evaluation["paired_x"]);paired_rows.append({"fold":task_index,"heldout_subject":task_index+1,"protocol":name,"method":method,"rrmse":r,"correlation":_corr(output,evaluation["paired_x"]),"delta_snr":float(20*np.log10((_rrmse(paired["RAW"],evaluation["paired_x"])+1e-12)/(r+1e-12))),"eligible":int(metadata["eligible"])})
            if method.startswith("DIFF-WRONG-"):wrong_rows.append({"fold":task_index,"heldout_subject":task_index+1,"protocol":name,"donor_index":method.rsplit("-",1)[-1],"rrmse":r})
        for method in natural.files:
            output=natural[method];psd,cov=_distortions(output[low],raw[low]);pres=float(1-_rrmse(output[low],raw[low]));indices=np.arange(split,len(raw)) if name=="same_01" else np.arange(len(raw));pred=classifier.predict(_features(output[indices]));kappa=float(cohen_kappa_score(evaluation["labels"][indices],pred));natural_rows.append({"fold":task_index,"heldout_subject":task_index+1,"protocol":name,"method":method,"eog_attenuation":base_coh-_coherence_proxy(output,eog),"nonartifact_preservation":pres,"psd_distortion":psd,"covariance_distortion":cov,"mi_kappa":kappa,"eligible":int(metadata["eligible"]),"alpha_active":int(np.linalg.norm(metadata["diff_alpha"])>1e-3)})
    _csv(root/"paired_metrics.csv",paired_rows);_csv(root/"natural_safety.csv",natural_rows);_csv(root/"wrong_donor_effects.csv",wrong_rows);summary={"status":"completed_independent_fold_evaluator","fold":task_index,"protocol_units":len(manifest)};_json(run_dir/"result_summary.json",summary);return summary


def stage_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));paired=[];natural=[];wrong=[]
    for fold in range(9):
        base=root/"bci2b_gpu"/f"fold_{fold:02d}";paired.extend(_read(base/"paired_metrics.csv"));natural.extend(_read(base/"natural_safety.csv"));wrong.extend(_read(base/"wrong_donor_effects.csv"))
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);_csv(root/"wrong_donor_effects.csv",wrong);participant=[]
    for subject in range(1,10):
        for panel,names in (("same_session",("same_01","same_02","same_03")),("cross_session",("cross_02","cross_03"))):
            rows=[r for r in paired if int(r["heldout_subject"])==subject and r["protocol"] in names];by=defaultdict(list)
            for r in rows:by[r["method"]].append(float(r["rrmse"]))
            mean={k:float(np.mean(v)) for k,v in by.items()};wrong_values=[v for k,values in by.items() if k.startswith("DIFF-WRONG-") for v in values];nrows=[r for r in natural if int(r["heldout_subject"])==subject and r["protocol"] in names and r["method"]=="DIFF-MATCH"]
            participant.append({"subject":subject,"protocol":panel,"U_D":mean["DET-MATCH"]-mean["DIFF-MATCH"],"U_P":mean["DIFF-POP"]-mean["DIFF-MATCH"],"U_W":float(np.mean(wrong_values))-mean["DIFF-MATCH"],"U_S":mean["DIFF-SHUFFLED"]-mean["DIFF-MATCH"],"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in nrows])),"preservation":float(np.mean([float(r["nonartifact_preservation"]) for r in nrows])),"psd":float(np.mean([float(r["psd_distortion"]) for r in nrows])),"covariance":float(np.mean([float(r["covariance_distortion"]) for r in nrows])),"kappa_match":float(np.mean([float(r["mi_kappa"]) for r in nrows])),"kappa_pop":float(np.mean([float(r["mi_kappa"]) for r in natural if int(r["heldout_subject"])==subject and r["protocol"] in names and r["method"]=="DIFF-POP"])),"kappa_raw":float(np.mean([float(r["mi_kappa"]) for r in natural if int(r["heldout_subject"])==subject and r["protocol"] in names and r["method"]=="RAW"])),"alpha_active":int(any(int(r["alpha_active"]) for r in nrows))})
    _csv(root/"participant_effects.csv",participant);summaries=[]
    for protocol in ("same_session","cross_session"):
        rows=[r for r in participant if r["protocol"]==protocol];entry={"protocol":protocol,"subjects":9}
        for effect in ("U_D","U_P","U_W","U_S"):values=np.asarray([r[effect] for r in rows]);entry[f"{effect}_mean"]=float(values.mean());entry[f"{effect}_median"]=float(np.median(values));entry[f"{effect}_positive"]=int((values>0).sum())
        entry.update({"eog_attenuation":float(np.mean([r["eog_attenuation"] for r in rows])),"preservation":float(np.mean([r["preservation"] for r in rows])),"psd":float(np.mean([r["psd"] for r in rows])),"covariance":float(np.mean([r["covariance"] for r in rows])),"kappa_match_minus_pop":float(np.mean([r["kappa_match"]-r["kappa_pop"] for r in rows])),"kappa_match_minus_raw":float(np.mean([r["kappa_match"]-r["kappa_raw"] for r in rows])),"active_alpha":int(sum(r["alpha_active"] for r in rows))});summaries.append(entry)
    _csv(root/"method_summary.csv",summaries);eligible=[]
    for r in summaries:
        gate=all(r[f"{e}_mean"]>0 and r[f"{e}_median"]>0 for e in ("U_D","U_P","U_W","U_S")) and r["U_P_positive"]>=6 and r["U_W_positive"]>=6 and r["eog_attenuation"]>0 and r["kappa_match_minus_pop"]>=-.02 and r["kappa_match_minus_raw"]>=-.02 and r["psd"]<=.25 and r["covariance"]<=.25 and r["active_alpha"]>=7
        if gate:eligible.append(r["protocol"])
    decision="ELIGIBLE_FOR_TWO_ADDITIONAL_MODEL_SEEDS" if eligible else "CURRENT_BCI2B_HIERARCHICAL_SCORE_LORA_INSTANCE_NO_GO";summary={"status":"completed_one_seed_development","decision":decision,"additional_seeds_authorized":bool(eligible),"eligible_protocols":eligible,"protocols":summaries,"bci2a":"identifiability_gate_failed_gpu_not_run","bci2b":"contingency_development","confirmation":False,"family_wide_status":"not_tested"};_json(root/"routing_decision.json",summary);_json(root/"result_summary.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def stage_diagnostic_aggregate(config:Mapping[str,Any],run_dir:Path)->dict[str,Any]:
    root=Path(str(config["result_root"]));paired=[];natural=[]
    for fold in range(3):
        base=root/"bci2b_gpu"/f"fold_{fold:02d}";paired.extend(_read(base/"paired_metrics.csv"));natural.extend(_read(base/"natural_safety.csv"))
    by=defaultdict(list)
    for r in paired:by[r["method"]].append(float(r["rrmse"]))
    diff_pop=float(np.mean(by["DIFF-POP"]));raw=float(np.mean(by["RAW"]));diff_match=float(np.mean(by["DIFF-MATCH"]));det_match=float(np.mean(by["DET-MATCH"]));nrows=[r for r in natural if r["method"]=="DIFF-MATCH"];safety={"preservation":float(np.mean([float(r["nonartifact_preservation"]) for r in nrows])),"psd":float(np.mean([float(r["psd_distortion"]) for r in nrows])),"covariance":float(np.mean([float(r["covariance_distortion"]) for r in nrows]))};finite=all(np.isfinite(v) for v in (diff_pop,raw,diff_match,det_match,*safety.values()));gate=finite and diff_pop<raw and safety["preservation"]>.5 and safety["psd"]<.5 and safety["covariance"]<.5;summary={"status":"completed_three_fold_technical_efficacy","full_nine_fold_authorized":gate,"raw_rrmse":raw,"det_match_rrmse":det_match,"diff_pop_rrmse":diff_pop,"diff_match_rrmse":diff_match,"safety":safety,"finite":finite};_json(root/"technical_route_decision.json",summary);_json(run_dir/"result_summary.json",summary);return summary


def stage_finalize(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Freeze the route at its completed diagnostic stage."""
    root=Path(str(config["result_root"]));paired=[];natural=[];wrong=[];completed=[]
    for fold in range(9):
        base=root/"bci2b_gpu"/f"fold_{fold:02d}"
        if not (base/"paired_metrics.csv").exists():continue
        completed.append(fold);paired.extend(_read(base/"paired_metrics.csv"));natural.extend(_read(base/"natural_safety.csv"));wrong.extend(_read(base/"wrong_donor_effects.csv"))
    if not completed:raise RuntimeError("no independently evaluated BCI2b diagnostic folds")
    _csv(root/"paired_metrics.csv",paired);_csv(root/"natural_safety.csv",natural);_csv(root/"wrong_donor_effects.csv",wrong)
    units=[]
    for subject in sorted({int(r["heldout_subject"]) for r in paired}):
        for panel,names in (("same_session",("same_01","same_02","same_03")),("cross_session",("cross_02","cross_03"))):
            prows=[r for r in paired if int(r["heldout_subject"])==subject and r["protocol"] in names]
            if not prows:continue
            by=defaultdict(list)
            for row in prows:by[row["method"]].append(float(row["rrmse"]))
            mean={name:float(np.mean(values)) for name,values in by.items()};wrong_values=[value for name,values in by.items() if name.startswith("DIFF-WRONG-") for value in values];nrows=[r for r in natural if int(r["heldout_subject"])==subject and r["protocol"] in names and r["method"]=="DIFF-MATCH"]
            units.append({"subject":subject,"protocol":panel,"U_D":mean["DET-MATCH"]-mean["DIFF-MATCH"],"U_P":mean["DIFF-POP"]-mean["DIFF-MATCH"],"U_W":float(np.mean(wrong_values))-mean["DIFF-MATCH"],"U_S":mean["DIFF-SHUFFLED"]-mean["DIFF-MATCH"],"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in nrows])),"preservation":float(np.mean([float(r["nonartifact_preservation"]) for r in nrows])),"psd_distortion":float(np.mean([float(r["psd_distortion"]) for r in nrows])),"covariance_distortion":float(np.mean([float(r["covariance_distortion"]) for r in nrows])),"mi_kappa":float(np.mean([float(r["mi_kappa"]) for r in nrows])),"alpha_active":int(any(int(r["alpha_active"]) for r in nrows))})
    _csv(root/"participant_effects.csv",units)
    methods=[]
    for panel,names in (("same_session",("same_01","same_02","same_03")),("cross_session",("cross_02","cross_03"))):
        for method in sorted({r["method"] for r in paired}):
            p=[r for r in paired if r["protocol"] in names and r["method"]==method];n=[r for r in natural if r["protocol"] in names and r["method"]==method]
            if not p:continue
            methods.append({"protocol":panel,"method":method,"diagnostic_subjects":len({r["heldout_subject"] for r in p}),"paired_rrmse":float(np.mean([float(r["rrmse"]) for r in p])),"paired_correlation":float(np.mean([float(r["correlation"]) for r in p])),"delta_snr":float(np.mean([float(r["delta_snr"]) for r in p])),"eog_attenuation":float(np.mean([float(r["eog_attenuation"]) for r in n])) if n else "","preservation":float(np.mean([float(r["nonartifact_preservation"]) for r in n])) if n else "","psd_distortion":float(np.mean([float(r["psd_distortion"]) for r in n])) if n else "","covariance_distortion":float(np.mean([float(r["covariance_distortion"]) for r in n])) if n else "","mi_kappa":float(np.mean([float(r["mi_kappa"]) for r in n])) if n else ""})
    _csv(root/"method_summary.csv",methods);_csv(root/"mi_preservation.csv",[{"protocol":r["protocol"],"method":r["method"],"diagnostic_subjects":r["diagnostic_subjects"],"mi_kappa":r["mi_kappa"]} for r in methods if r["mi_kappa"]!=""])
    ident2a=json.loads((root/"headroom"/"routing_decision.json").read_text());ident2b=json.loads((root/"headroom_2b"/"routing_decision.json").read_text());technical=json.loads((root/"technical_route_decision.json").read_text())
    decision={"status":"completed_development_route","decision":"CURRENT_BCI2B_HIERARCHICAL_SCORE_LORA_INSTANCE_NO_GO","bci2a_identifiability":ident2a["decision"],"bci2a_gpu_training_run":False,"bci2b_identifiability":ident2b["decision"],"bci2b_diagnostic_folds":completed,"population_diffusion_valid_estimator":False,"failure_localization":"population_estimator_and_absolute_safety_failed_before_adapter_adjudication","full_nine_fold_run":False,"additional_seeds_authorized":False,"technical_route":technical,"confirmation":False,"family_wide_status":"not_tested"};_json(root/"routing_decision.json",decision);_json(root/"result_summary.json",decision)
    h=["# BCI2a subject identifiability V10","","All nine BCI-IV-2a participants were audited. Query operators were evaluator-only and never entered deployment inference.","","| protocol | MATCH-WRONG mean | positive | MATCH-POP mean |","|---|---:|---:|---:|"]
    for row in ident2a["protocols"]:h.append(f"| {row['protocol']} | {row['match_wrong_mean']:+.4f} | {row['match_wrong_positive']}/9 | {row['match_pop_mean']:+.4f} |")
    h += ["",f"BCI2a decision: `{ident2a['decision']}`; the frozen 6/9 identifiability gate was not met, so no BCI2a GPU denoiser was trained.","","BCI-IV-2b was used only as the pre-specified contamination-transfer contingency:","","| protocol | MATCH-WRONG mean | positive | MATCH-POP mean |","|---|---:|---:|---:|"]
    for row in ident2b["protocols"]:h.append(f"| {row['protocol']} | {row['match_wrong_mean']:+.4f} | {row['match_wrong_positive']}/9 | {row['match_pop_mean']:+.4f} |")
    h += ["",f"BCI2b decision: `{ident2b['decision']}`. This development headroom cannot be relabeled as a BCI2a result."];Path("reports/bci2a_subject_identifiability_v10.md").write_text("\n".join(h)+"\n",encoding="utf-8")
    diagnostic=[]
    for panel in ("same_session","cross_session"):
        rows=[r for r in units if r["protocol"]==panel]
        diagnostic.append({"protocol":panel,**{effect:float(np.mean([r[effect] for r in rows])) for effect in ("U_D","U_P","U_W","U_S")}})
    report=["# BCI2a hierarchical Score-LoRA diffusion V10","","Development exploration only; no confirmation or sealed outcomes were opened.","","## Routing outcome","",f"BCI2a did not meet the frozen subject-identifiability gate (`{ident2a['decision']}`), so the hierarchical diffusion factorial was not run on BCI2a. The pre-specified BCI2b audit did meet identifiability and entered a three-subject real-data technical/efficacy ladder.","",f"After correcting the physical-unit mismatch and retraining from regenerated microvolt-scale arrays, the BCI2b ladder still failed absolute validity: RAW RRMSE {technical['raw_rrmse']:.4f}, DET-MATCH {technical['det_match_rrmse']:.4f}, DIFF-POP {technical['diff_pop_rrmse']:.4f}, DIFF-MATCH {technical['diff_match_rrmse']:.4f}; preservation {technical['safety']['preservation']:.4f}, PSD distortion {technical['safety']['psd']:.4f}, covariance distortion {technical['safety']['covariance']:.4f}.","","The diagnostic contrasts below are reported for transparency but are not promoted to scientific effects because the population estimator failed absolute validity:","","| protocol | U_D | U_P | U_W | U_S |","|---|---:|---:|---:|---:|"]
    for row in diagnostic:report.append(f"| {row['protocol']} | {row['U_D']:+.4f} | {row['U_P']:+.4f} | {row['U_W']:+.4f} | {row['U_S']:+.4f} |")
    report += ["", "Same-session and cross-session both show a negative diffusion-vs-deterministic contrast and negative temporal-shuffle specificity; the tiny MATCH−POP differences cannot establish subject utility under the failed absolute estimator.","","The full 9-fold factorial and extra seeds were not authorized. This localizes the stopped route to the current population estimator/absolute-safety implementation before a fair adapter-success adjudication; it is not a hierarchical-adapter, diffusion, personalization, BCI2a, or BCI2b family-wide negative.","","## Evidence boundaries","","- BCI2a: complete 9/9 participant identifiability audit; no GPU denoising result.","- BCI2b: complete 9/9 participant identifiability audit; real-data diagnostic GPU ladder on 3/9 participants only.","- Paired targets are EOG-backed semi-simulation, not natural-clean ground truth.","- Natural evaluator fields were opened only after cleaner outputs were frozen."]
    Path("reports/bci2a_hierarchical_score_diffusion_v10.md").write_text("\n".join(report)+"\n",encoding="utf-8");_json(run_dir/"result_summary.json",decision);return decision


def run_stage(config_path: Path, stage: str, run_dir: Path, *, task_index: int = 0) -> dict[str, Any]:
    config = _config(config_path); run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "inventory": return stage_inventory(config, run_dir)
    if stage == "sge-closure": return stage_sge_closure(config, run_dir)
    if stage == "sge-factorial-replay": return stage_sge_factorial_replay(config, task_index, run_dir)
    if stage == "sge-factorial-eval": return stage_sge_factorial_eval(config, task_index, run_dir)
    if stage == "headroom-extract": return stage_headroom_extract(config, task_index, run_dir)
    if stage == "headroom-aggregate": return stage_headroom_aggregate(config, run_dir)
    if stage == "bci2b-extract": return stage_bci2b_extract(config, task_index, run_dir)
    if stage == "bci2b-aggregate": return stage_bci2b_aggregate(config, run_dir)
    if stage == "prepare-fold": return stage_prepare_fold(config, task_index, run_dir)
    if stage == "train-fold": return stage_train_fold(config, task_index, run_dir)
    if stage == "infer-fold": return stage_infer_fold(config, task_index, run_dir)
    if stage == "eval-fold": return stage_eval_fold(config, task_index, run_dir)
    if stage == "aggregate": return stage_aggregate(config, run_dir)
    if stage == "diagnostic-aggregate": return stage_diagnostic_aggregate(config, run_dir)
    if stage == "finalize": return stage_finalize(config, run_dir)
    raise ValueError(f"unknown V10 stage: {stage}")


__all__ = [
    "run_stage", "stage_inventory", "stage_sge_closure",
    "stage_sge_factorial_replay", "stage_sge_factorial_eval",
]
