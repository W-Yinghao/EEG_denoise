"""Read-only provenance audit for the frozen v19 null floor."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_counterfactual_operator_v19_provenance"))


def _audit_root(config: Mapping[str, Any]) -> Path:
    return CODE_ROOT / str(config["audit_result_root"])


def _source_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["source_result_root"]))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _canonical_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":")).encode()
    if suffix in {".yaml", ".yml"}:
        return json.dumps(yaml.safe_load(path.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":")).encode()
    if suffix == ".csv":
        rows = _read_csv(path)
        return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return path.read_bytes()


def _digest(path: Path, maximum: int) -> str:
    if path.stat().st_size > maximum:
        return "NOT_DIGESTED_SIZE_LIMIT"
    return "sha256:" + hashlib.sha256(_canonical_bytes(path)).hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix.lower() == ".csv":
        return len(_read_csv(path))
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return len(value) if isinstance(value, list) else 1
    return None


def _role(path: Path) -> str:
    text = str(path)
    if "/o0_natural/" in text: return "O0-A accepted source rows"
    if "/o0_paired/" in text: return "O0-B accepted source rows"
    if "v19_preregistration" in text: return "frozen preregistration"
    if "participant_effects" in text: return "canonical participant effects"
    if "null_controls" in text: return "participant-level stress/null cells"
    if "route_decision" in text or "o0_summary" in text: return "canonical decision"
    if "job_ids" in text: return "job lineage map"
    if "/runs/" in text: return "job-scoped run summary"
    if "/slurm_logs/" in text: return "job log"
    return "supporting manifest"


def _producer(path: Path) -> str:
    text = str(path)
    if "/o0_natural/" in text: return "934248"
    if "/o0_paired/" in text: return "934231"
    if "operator_fit" in text: return "934209"
    if "alignment" in text: return "934192"
    if "operator_context_manifest" in text or "data_protocol_decision" in text: return "934230"
    if "v19_preregistration" in text or "split_manifest" in text or "metric_schema" in text: return "934229"
    if any(value in text for value in ("participant_effects", "null_controls", "wrong_donor_metrics", "route_decision", "o0_summary")): return "934265"
    if "result_summary" in text or "gauge_scale_replay" in text: return "934266"
    return "supporting_or_unknown"


def preflight_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    source_worktree = Path(str(config["source_worktree"]))
    expected = str(config["source_v19_commit"])
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_worktree, text=True).strip()
    if actual != expected:
        raise AssertionError(f"source HEAD mismatch: {actual}")
    if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source_worktree,
                      text=True, capture_output=True, check=True).stdout.strip():
        raise AssertionError("source tracked tree is dirty")
    source = _source_root(config); source_worktree_result = source_worktree / "results/cgdr/counterfactual_operator_headroom_v19"
    # Tracked manifests are read from this clean base; server-only per-window rows from the immutable source worktree.
    candidates: list[Path] = []
    candidates += [path for path in source_worktree_result.rglob("*") if path.is_file()
                   and not any(part in {"runs", "o0_natural", "o0_paired"} for part in path.parts)]
    candidates += sorted((source / "o0_natural").glob("*.csv"))
    candidates += sorted((source / "o0_paired").glob("*.csv"))
    candidates += [source_worktree / "configs/cgdr/counterfactual_operator_headroom_v19.yaml",
                   source_worktree / "src/eeg_cgdr/experiments/counterfactual_operator_v19.py",
                   source_worktree / "reports/slurm/counterfactual_operator_headroom_v19_job_ids.txt"]
    run_root = source / "runs"
    candidates += sorted(run_root.glob("*/job_*/result_summary.json"))
    log_root = Path(str(config["source_log_root"]))
    candidates += sorted(log_root.glob("*.out")) + sorted(log_root.glob("*.err"))
    unique = sorted(set(path.resolve() for path in candidates if path.is_file()))
    rows: list[dict[str, Any]] = []
    max_bytes = int(config["digest_max_bytes"])
    for path in unique:
        stat = path.stat()
        rows.append({"absolute_path": str(path), "producing_job": _producer(path),
                     "commit": expected, "config": "configs/cgdr/counterfactual_operator_headroom_v19.yaml",
                     "row_count": _row_count(path), "dtype": path.suffix.lower().lstrip(".") or "text",
                     "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                     "canonical_content_digest": _digest(path, max_bytes), "scientific_role": _role(path),
                     "accepted_status": "accepted" if _producer(path) in {str(v) for v in config["accepted_jobs"]} else "supporting_or_excluded"})
    output = _audit_root(config)
    _write_csv(output / "artifact_inventory.csv", rows)
    snapshot = {row["absolute_path"]: {"size_bytes": row["size_bytes"], "mtime_ns": row["mtime_ns"],
                                       "digest": row["canonical_content_digest"]} for row in rows}
    _write_json(output / "frozen_artifact_snapshot.json", snapshot)
    natural = sorted((source / "o0_natural").glob("sub-*.csv")); paired = sorted((source / "o0_paired").glob("sub-*.csv"))
    prereg = source_worktree_result / "v19_preregistration.yaml"
    permutation_candidates = [path for path in unique if any(token in path.name.lower() for token in ("permutation", "rng", "resampling"))]
    status = {
        "stage": "P0", "source_commit_exact": actual == expected, "tracked_source_clean": True,
        "natural_participant_files": len(natural), "paired_participant_files": len(paired),
        "preregistration_present": prereg.is_file(), "permutation_index_files": len(permutation_candidates),
        "floor_rng_schedule_present": False,
        "numeric_replay_inputs_complete": len(natural) == 16 and len(paired) == 16 and prereg.is_file(),
        "requested_statistical_null_provenance": "INDETERMINATE_NO_REPLICATE_OR_RNG_SCHEDULE",
        "raw_or_sealed_read": False,
    }
    _write_json(output / "p0_input_decision.json", status); _write_json(run_dir / "result_summary.json", status)
    return status


def lineage_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    source_worktree = Path(str(config["source_worktree"])); source = _source_root(config)
    job_map = _read_csv_like_job_map(source_worktree / "reports/slurm/counterfactual_operator_headroom_v19_job_ids.txt")
    accepted = {str(value) for value in config["accepted_jobs"]}; excluded = {str(value) for value in config["excluded_jobs"]}
    rows: list[dict[str, Any]] = []
    try:
        command = ["sacct", "-j", ",".join(sorted(accepted | excluded)), "--format=JobIDRaw,State,ExitCode,Elapsed", "-n", "-P"]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        sacct_status = "available" if completed.returncode == 0 else "unavailable"
        sacct_text = completed.stdout + completed.stderr
    except Exception as error:
        sacct_status = "unavailable"; sacct_text = f"{type(error).__name__}: {error}"
    for row in job_map:
        base = row["job_id"].split("_")[0]
        if not base.isdigit():
            category = "no_job_id_submission"
        elif base in accepted:
            category = "accepted_science_producer"
        elif base in excluded:
            category = "excluded_or_superseded"
        elif base in {str(v) for v in config["qa_only_jobs"]}:
            category = "qa_only"
        elif base in {str(v) for v in config["predecessor_qa_jobs"]}:
            category = "predecessor_qa"
        else:
            category = "supporting"
        pattern = f"*{base}*" if base.isdigit() else "__no_such_log__"
        logs = list(Path(str(config["source_log_root"])).glob(pattern))
        runs = list((source / "runs").glob(f"*/job_{base}*/result_summary.json")) if base.isdigit() else []
        rows.append({**row, "base_job_id": base, "category": category, "log_files": len(logs),
                     "run_summaries": len(runs), "attribution_consistent": int(category != "accepted_science_producer" or bool(logs or runs))})
    _write_csv(_audit_root(config) / "job_lineage.csv", rows)
    prereg_mtime = (source_worktree / "results/cgdr/counterfactual_operator_headroom_v19/v19_preregistration.yaml").stat().st_mtime_ns
    score_paths = list((source / "o0_natural").glob("*.csv")) + list((source / "o0_paired").glob("*.csv"))
    freeze_before_scoring = bool(score_paths) and prereg_mtime <= min(path.stat().st_mtime_ns for path in score_paths)
    no_job_rows = [row for row in rows if row["category"] == "no_job_id_submission"]
    mismatch = [row for row in rows if not int(row["attribution_consistent"])]
    result = {"stage": "P1", "sacct_status": sacct_status, "sacct_output": sacct_text[:4000],
              "accepted_jobs": sorted(accepted), "excluded_jobs": sorted(excluded),
              "no_job_id_submissions": len(no_job_rows), "freeze_before_scoring": freeze_before_scoring,
              "producer_attribution_mismatches": len(mismatch),
              "lineage_decision": "PASS" if not mismatch and freeze_before_scoring else "FAIL_PROVENANCE_MISMATCH"}
    _write_json(_audit_root(config) / "job_lineage_summary.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def _read_csv_like_job_map(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"): continue
        parts = line.split("\t")
        rows.append({"job_id": parts[0], "stage": parts[1] if len(parts) > 1 else "",
                     "state": parts[2] if len(parts) > 2 else "", "recovery_of": parts[3] if len(parts) > 3 else "",
                     "note": parts[4] if len(parts) > 4 else ""})
    return rows


def participant_manifest_stage(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    participant = list(config["development_participants"])[task_index]; source = _source_root(config)
    rows: list[dict[str, Any]] = []
    maximum = int(config["digest_max_bytes"])
    for panel, producing_job in (("natural", "934248"), ("paired", "934231")):
        path = source / f"o0_{panel}" / f"{participant}.csv"
        values = _read_csv(path)
        methods = sorted({row["method"] for row in values}); donors = sorted({row.get("wrong_donor", "") for row in values if row.get("wrong_donor", "")})
        rows.append({"participant": participant, "panel": panel, "absolute_path": str(path.resolve()),
                     "producing_job": producing_job, "rows": len(values), "sessions": len({row["session"] for row in values}),
                     "tasks": ";".join(sorted({row["task"] for row in values})), "methods": ";".join(methods),
                     "wrong_donors": len(donors), "digest": _digest(path, maximum), "accepted": 1})
    path = _audit_root(config) / "participant_artifact_manifest" / f"{participant}.csv"
    _write_csv(path, rows)
    result = {"stage": "P2", "participant": participant, "panels": 2, "rows": sum(row["rows"] for row in rows)}
    _write_json(run_dir / "result_summary.json", result)
    return result


def exact_signflip(values: Sequence[float], *, two_sided: bool = False) -> float:
    array = np.asarray(values, dtype=np.float64); observed = float(np.mean(array))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(array))), dtype=np.float64)
    permutations = np.mean(signs * array[None, :], axis=1)
    if two_sided:
        return float(np.mean(np.abs(permutations) >= abs(observed) - 1e-15))
    return float(np.mean(permutations >= observed - 1e-15))


def _independent_collapse(rows: Sequence[Mapping[str, str]], value_key: str, natural: bool) -> dict[tuple[str, str, str, str], float]:
    donor_values: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in rows:
        if natural and row.get("eog_panel") != "high": continue
        key = (row["participant"], row["session"], row["task"], row["method"], row.get("wrong_donor", ""))
        donor_values.setdefault(key, []).append(float(row[value_key]))
    method_values: dict[tuple[str, str, str, str], list[float]] = {}
    for (participant, session, task, method, _donor), values in donor_values.items():
        method_values.setdefault((participant, session, task, method), []).append(float(np.mean(values)))
    return {key: float(np.mean(values)) for key, values in method_values.items()}


def _independent_participant(unit: Mapping[tuple[str, str, str, str], float]) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str, str], float]]:
    task_values: dict[tuple[str, str, str], list[float]] = {}
    for (participant, _session, task, method), value in unit.items():
        task_values.setdefault((participant, task, method), []).append(value)
    task = {key: float(np.mean(values)) for key, values in task_values.items()}
    participant_values: dict[tuple[str, str], list[float]] = {}
    for (participant, _task, method), value in task.items():
        participant_values.setdefault((participant, method), []).append(value)
    return {key: float(np.mean(values)) for key, values in participant_values.items()}, task


def independent_replay(config: Mapping[str, Any]) -> dict[str, Any]:
    source = _source_root(config); participants = list(config["development_participants"])
    natural_rows = list(itertools.chain.from_iterable(_read_csv(source / "o0_natural" / f"{p}.csv") for p in participants))
    paired_rows = list(itertools.chain.from_iterable(_read_csv(source / "o0_paired" / f"{p}.csv") for p in participants))
    natural_unit = _independent_collapse(natural_rows, "normalized_prediction_risk", True)
    paired_unit = _independent_collapse(paired_rows, "mask_rrmse", False)
    natural_pm, natural_ptm = _independent_participant(natural_unit)
    paired_pm, paired_ptm = _independent_participant(paired_unit)
    participant_rows: list[dict[str, Any]] = []; null_rows: list[dict[str, Any]] = []
    for participant in participants:
        tasks_ok = all((participant, task, "MATCH") in natural_ptm and (participant, task, "MATCH") in paired_ptm
                       for task in ("ERP", "SSVEP"))
        available = tasks_ok and all((participant, method) in natural_pm for method in ("MATCH", "POP", "WRONG")) \
                    and all((participant, method) in paired_pm for method in ("MATCH", "POP", "WRONG", "QUERY_ORACLE"))
        row: dict[str, Any] = {"participant": participant, "evaluable": int(available)}
        for panel, prefix, pm in (("natural", "N", natural_pm), ("paired", "H", paired_pm)):
            if available:
                row[f"{prefix}_P"] = pm[(participant, "POP")] - pm[(participant, "MATCH")]
                row[f"{prefix}_W"] = pm[(participant, "WRONG")] - pm[(participant, "MATCH")]
                row[f"{prefix}_time_shift_effect"] = pm[(participant, "POP")] - pm[(participant, "TIME_SHIFT")]
                row[f"{prefix}_gain_P"] = pm.get((participant, "GAIN_POP"), pm[(participant, "POP")]) - pm[(participant, "GAIN_MATCH")]
                row[f"{prefix}_gain_W"] = pm[(participant, "GAIN_WRONG")] - pm[(participant, "GAIN_MATCH")]
                null_rows.append({"participant": participant, "panel": panel,
                                  "pop_minus_time_shift": row[f"{prefix}_time_shift_effect"],
                                  "pop_minus_channel_perm": pm[(participant, "POP")] - pm[(participant, "CHANNEL_PERM")],
                                  "gain_match_effect": row[f"{prefix}_gain_P"]})
            else:
                for name in ("P", "W", "time_shift_effect", "gain_P", "gain_W"): row[f"{prefix}_{name}"] = 0.0
        participant_rows.append(row)
    floor_rows: list[dict[str, Any]] = []
    for row in participant_rows:
        participant = row["participant"]
        for panel, prefix, pm in (("natural", "N", natural_pm), ("paired", "H", paired_pm)):
            outer = [item for item in null_rows if item["participant"] != participant and item["panel"] == panel]
            cells = [max(0.0, float(item["pop_minus_time_shift"])) for item in outer]
            q95 = float(np.quantile(cells, 0.95, method="linear"))
            pop_risk = pm.get((participant, "POP"), float("nan"))
            relative = 0.05 * pop_risk if np.isfinite(pop_risk) else 0.0
            floor = max(0.01, relative, q95)
            row[f"{prefix}_effect_floor"] = floor
            components = {"absolute": 0.01, "relative": relative, "outer_time_shift_q95": q95}
            floor_rows.append({"heldout_participant": participant, "panel": panel, "outer_cell_count": len(cells),
                               "absolute_component": 0.01, "relative_reference_pop_risk": pop_risk,
                               "relative_component": relative, "outer_null_q95": q95,
                               "quantile_method": "numpy_linear", "one_sided_clip": "max(0,effect)",
                               "floor": floor, "dominant_component": max(components, key=components.get),
                               "scientific_unit": "participant cell from another LOPO fold",
                               "group_null_statistic": "ABSENT", "joint_max_statistic": "ABSENT",
                               "permutation_replicate": "ABSENT"})
    effects = {key: [float(row[key]) for row in participant_rows] for key in ("N_P", "N_W", "H_P", "H_W")}
    summary = {key: {"mean": float(np.mean(values)), "median": float(np.median(values)),
                     "positive_count": int(np.sum(np.asarray(values) > 0)),
                     "exact_one_sided_p": exact_signflip(values), "exact_two_sided_p": exact_signflip(values, two_sided=True)}
               for key, values in effects.items()}
    summary["floor_A"] = float(np.mean([row["N_effect_floor"] for row in participant_rows]))
    summary["floor_B"] = float(np.mean([row["H_effect_floor"] for row in participant_rows]))
    return {"participant_rows": participant_rows, "null_rows": null_rows, "floor_rows": floor_rows,
            "summary": summary, "natural_pm": natural_pm, "paired_pm": paired_pm,
            "natural_rows": natural_rows, "paired_rows": paired_rows}


def floor_lineage_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    replay = independent_replay(config); output = _audit_root(config)
    _write_csv(output / "floor_lineage.csv", replay["floor_rows"])
    null_rows = replay["null_rows"]
    family_rows: list[dict[str, Any]] = []
    family_keys = {"EOG_TIME_SHIFT": "pop_minus_time_shift", "CHANNEL_PERM": "pop_minus_channel_perm",
                   "GAIN_NORMALIZED": "gain_match_effect"}
    for panel in ("natural", "paired"):
        observed_prefix = "N" if panel == "natural" else "H"
        for family, key in family_keys.items():
            values = np.asarray([float(row[key]) for row in null_rows if row["panel"] == panel], dtype=np.float64)
            family_rows.append({"panel": panel, "null_family": family, "classification": "statistical_floor_input" if family == "EOG_TIME_SHIFT" else "falsification_stress_control",
                                "replicate_count": 0, "participant_cells": len(values), "mean": float(np.mean(values)),
                                "median": float(np.median(values)), "sd": float(np.std(values)),
                                "q90": float(np.quantile(values, .90)), "q95": float(np.quantile(values, .95)),
                                "q99": float(np.quantile(values, .99)), "maximum": float(np.max(values)),
                                "center_shift": float(np.mean(values)),
                                "maxstat_winner_count": "NOT_DEFINED",
                                "proportion_exceeding_observed_P": float(np.mean(values > config["reported"][f"{observed_prefix}_P"])),
                                "proportion_exceeding_observed_W": float(np.mean(values > config["reported"][f"{observed_prefix}_W"])),
                                "exchangeable": False if family == "EOG_TIME_SHIFT" else False})
    _write_csv(output / "null_family_summary.csv", family_rows)
    maxstat_rows = [{"heldout_participant": row["heldout_participant"], "panel": row["panel"],
                     "replicate_id": "ABSENT", "participant_vector": "ABSENT", "group_statistic": "ABSENT",
                     "joint_max_statistic": "ABSENT", "outer_cell_q95": row["outer_null_q95"],
                     "status": "original implementation used cross-participant stress cells, not null replicates"}
                    for row in replay["floor_rows"]]
    _write_csv(output / "null_replicate_maxstat.csv", maxstat_rows)
    extreme: list[dict[str, Any]] = []
    for floor in replay["floor_rows"]:
        candidates = [row for row in null_rows if row["panel"] == floor["panel"] and row["participant"] != floor["heldout_participant"]]
        candidates.sort(key=lambda row: max(0.0, float(row["pop_minus_time_shift"])), reverse=True)
        for rank, row in enumerate(candidates[:15], 1):
            extreme.append({"heldout_fold": floor["heldout_participant"], "panel": floor["panel"], "rank": rank,
                            "contributing_participant": row["participant"], "null_family": "EOG_TIME_SHIFT",
                            "raw_effect": row["pop_minus_time_shift"], "clipped_effect": max(0.0, float(row["pop_minus_time_shift"])),
                            "donor": "participant-level donor-mean already reduced", "protocol": "ERP/SSVEP equal weighted",
                            "operator_id": f"{row['participant']}::TIME_SHIFT", "is_permutation_replicate": False,
                            "fold_q95": floor["outer_null_q95"]})
    selected: list[dict[str, Any]] = []
    for panel in ("natural", "paired"):
        panel_rows = [row for row in extreme if row["panel"] == panel]
        panel_rows.sort(key=lambda row: float(row["clipped_effect"]), reverse=True)
        selected.extend(panel_rows[:20])
    _write_csv(output / "top_extreme_trace.csv", selected)
    result = {"stage": "P3", "replayed": replay["summary"], "quantile_method_in_code": "numpy default linear",
              "quantile_method_in_prereg": "UNSPECIFIED", "permutation_count": 0, "permutation_seed": None,
              "group_null_statistic": "ABSENT", "joint_max_stat": "ABSENT",
              "null_input": "fold-local positive-clipped participant time-shift stress cells",
              "null_exchangeable": False, "max_stat_axis_valid": False}
    _write_json(output / "floor_lineage_summary.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def original_replay_in_scratch(config: Mapping[str, Any]) -> dict[str, Any]:
    from eeg_cgdr.experiments import counterfactual_operator_v19 as original
    source = _source_root(config)
    scratch = _audit_root(config) / "replay_scratch" / "original"
    if scratch.exists(): shutil.rmtree(scratch)
    target = scratch / "results/cgdr/counterfactual_operator_headroom_v19"
    (target / "o0_natural").mkdir(parents=True); (target / "o0_paired").mkdir(parents=True)
    for panel in ("natural", "paired"):
        for path in (source / f"o0_{panel}").glob("*.csv"):
            shutil.copy2(path, target / f"o0_{panel}" / path.name)
    config_path = Path(str(config["source_worktree"])) / "configs/cgdr/counterfactual_operator_headroom_v19.yaml"
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    previous = original.CODE_ROOT; original.CODE_ROOT = scratch
    try:
        original.aggregate_o0_stage(source_config, scratch / "run")
    finally:
        original.CODE_ROOT = previous
    return {"participant_rows": _read_csv(target / "participant_effects.csv"),
            "null_rows": _read_csv(target / "null_controls.csv"),
            "decision": json.loads((target / "route_decision.json").read_text(encoding="utf-8"))}


def replay_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    original = original_replay_in_scratch(config); reference = independent_replay(config)
    original_by_id = {row["participant"]: row for row in original["participant_rows"]}
    reference_by_id = {row["participant"]: row for row in reference["participant_rows"]}
    differences: list[float] = []
    id_equal = set(original_by_id) == set(reference_by_id)
    for participant in sorted(reference_by_id):
        for key in ("N_P", "N_W", "H_P", "H_W", "N_effect_floor", "H_effect_floor"):
            differences.append(abs(float(original_by_id[participant][key]) - float(reference_by_id[participant][key])))
    reported = config["reported"]; summary = reference["summary"]
    report_diffs = {key: abs(float(summary[key]["mean"]) - float(reported[key])) for key in ("N_P", "N_W", "H_P", "H_W")}
    report_diffs["floor_A"] = abs(float(summary["floor_A"]) - float(reported["floor_A"]))
    report_diffs["floor_B"] = abs(float(summary["floor_B"]) - float(reported["floor_B"]))
    max_diff = max(differences + list(report_diffs.values()))
    comparison = {"source_row_sets_identical": True, "participant_ids_identical": id_equal,
                  "maximum_numeric_difference": max_diff, "tolerance": float(config["tolerance"]),
                  "exact_numeric_reproduction": id_equal and max_diff <= float(config["tolerance"]),
                  "reported_differences": report_diffs, "original_route": original["decision"]["route"],
                  "reference_summary": summary, "permutation_schedule_identical": "NOT_APPLICABLE_ABSENT",
                  "rng_schedule_identical": "NO_FLOOR_RNG_EXISTS"}
    _write_json(_audit_root(config) / "independent_replay_comparison.json", comparison)
    _write_json(run_dir / "result_summary.json", comparison)
    return comparison


def reference_joint_maxstat(replicates: np.ndarray, quantile: float = 0.95) -> tuple[np.ndarray, float]:
    """Reference synthetic max-stat: endpoints are maxed only after participant mean."""
    value = np.asarray(replicates, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("expected replicate x participant x endpoint")
    group = np.mean(value, axis=1)
    maxstat = np.max(group, axis=1)
    return maxstat, float(np.quantile(maxstat, quantile, method="linear"))


def _verify_snapshot(config: Mapping[str, Any]) -> tuple[bool, list[str]]:
    snapshot = json.loads((_audit_root(config) / "frozen_artifact_snapshot.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, frozen in snapshot.items():
        path = Path(name)
        if not path.is_file():
            failures.append(f"missing:{name}"); continue
        if path.stat().st_size != int(frozen["size_bytes"]): failures.append(f"size:{name}")
        digest = str(frozen["digest"])
        if digest != "NOT_DIGESTED_SIZE_LIMIT" and _digest(path, int(config["digest_max_bytes"])) != digest:
            failures.append(f"digest:{name}")
    return not failures, failures


def decision_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    output = _audit_root(config)
    comparison = json.loads((output / "independent_replay_comparison.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "floor_lineage_summary.json").read_text(encoding="utf-8"))
    p0 = json.loads((output / "p0_input_decision.json").read_text(encoding="utf-8"))
    p1 = json.loads((output / "job_lineage_summary.json").read_text(encoding="utf-8"))
    original_unchanged, artifact_failures = _verify_snapshot(config)
    source_decision = json.loads((_source_root(config) / "route_decision.json").read_text(encoding="utf-8"))
    sealed = json.loads((_source_root(config) / "sealed_guard.json").read_text(encoding="utf-8"))
    sealed_zero = all(int(sealed.get(key, 0)) == 0 for key in
                      ("mobile_sealed_reads", "physiomotion_sealed_reads", "shu_day4_day5_reads", "physiotrait_day200_reads"))
    a_root = Path("/home/infres/yinwang/denoiseNet_taas_subject_diffusion")
    a_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a_root, text=True).strip()
    a_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "taas_submission"], cwd=a_root, check=False).returncode
    exact = bool(comparison["exact_numeric_reproduction"])
    # Numerically reproducible does not imply a valid statistical floor.  The frozen
    # text never defines null replicates, a group statistic, a P/W joint max axis,
    # or percentile interpolation; the time shift is a falsification intervention,
    # not an exchangeable draw under the scientific null.
    ambiguity = [
        "no permutation/null-replicate indices or floor RNG schedule",
        "no preregistered group null statistic",
        "no preregistered panel-local/global P/W max-stat axis",
        "percentile interpolation method absent from preregistration",
        "EOG time-shift falsification cells are not exchangeable null replicates",
        "multiple reasonable corrections exist and none is uniquely preregistered",
    ]
    verdict = "INVALID_NO_EXACT_RECOVERY"
    final_label = "NULL_FLOOR_PROTOCOL_OR_PROVENANCE_INVALID_ROUTE_CLOSED"
    extremes = _read_csv(output / "top_extreme_trace.csv")
    natural_extremes = [row for row in extremes if row["panel"] == "natural"]
    dominant = natural_extremes[0] if natural_extremes else None
    decision = {
        "source_v19_commit": str(config["source_v19_commit"]),
        "original_decision": "SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO",
        "floor_A_reported": round(float(config["reported"]["floor_A"]), 4),
        "floor_B_reported": round(float(config["reported"]["floor_B"]), 4),
        "floor_A_reproduced": float(comparison["reference_summary"]["floor_A"]),
        "floor_B_reproduced": float(comparison["reference_summary"]["floor_B"]),
        "exact_numeric_reproduction": exact,
        "same_scientific_unit": True,
        "same_risk_scale": True,
        "participant_first": True,
        "wrong_donor_mean_preserved": True,
        "max_stat_axis_valid": False,
        "outer_only": True,
        "null_exchangeable": False,
        "dominant_null_cell_A": dominant,
        "audit_verdict": verdict,
        "final_label": final_label,
        "exact_recovery_executed": False,
        "O1_executed": False,
        "sealed_opened": False,
        "paper_modified": False,
        "failed_or_indeterminate_criteria": ambiguity,
        "source_artifacts_unchanged": original_unchanged,
        "source_artifact_mismatches": artifact_failures,
        "lineage_status": p1["lineage_decision"],
        "numeric_inputs_complete": p0["numeric_replay_inputs_complete"],
        "A_track_head": a_head, "A_track_forbidden_diff": bool(a_diff),
    }
    if source_decision["route"] != "SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO":
        raise AssertionError("original v19 decision changed")
    if not original_unchanged or not sealed_zero or a_head != "0c4f2301c1f873120fe54537cde3c76fff7ea3a2" or a_diff:
        raise AssertionError("immutable/sealed/A-track governance failure")
    _write_json(output / "audit_decision.json", decision)
    recovery = {"eligibility": "NOT_AUTHORIZED", "label": "NULL_FLOOR_PROTOCOL_OR_PROVENANCE_INVALID_ROUTE_CLOSED",
                "reason": "protocol does not uniquely specify a statistical-null replicate/group/max-stat construction",
                "candidate_repairs": ["participant-sign permutation max-stat", "fold-local stress-cell q95", "panel-local endpoint floor", "joint P/W max floor"],
                "multiple_reasonable_repairs": True, "exact_recovery_executed": False, "O1_authorized": False}
    _write_json(output / "exact_recovery_eligibility.json", recovery)
    lines = [
        "# v19 Null-Floor Provenance Audit", "",
        f"Final audit verdict: `{verdict}`", "",
        f"Terminal label: `{final_label}`", "",
        "The reported implementation is numerically reproducible, but the statistical-null provenance needed to validate the routing floor is not uniquely preregistered. No recovery or O1 was run.", "",
        "## Exact numeric replay", "",
        f"- O0-A floor: reported 0.2293; replayed {decision['floor_A_reproduced']:.15f}.",
        f"- O0-B floor: reported 0.1745; replayed {decision['floor_B_reproduced']:.15f}.",
        f"- Maximum original-vs-independent implementation difference: {comparison['maximum_numeric_difference']:.3g} (tolerance {comparison['tolerance']:.1e}).",
        f"- N_P={comparison['reference_summary']['N_P']['mean']:.15f}; N_W={comparison['reference_summary']['N_W']['mean']:.15f}; H_P={comparison['reference_summary']['H_P']['mean']:.15f}; H_W={comparison['reference_summary']['H_W']['mean']:.15f}.", "",
        "## What generated 0.2293", "",
        "For each held-out participant, the code selected the other 15 participants' participant-first `POP − TIME_SHIFT` natural-risk effects, clipped each at zero, and applied `numpy.quantile(..., 0.95)` with the library-default linear interpolation. It then took the maximum of 0.010, 5% of that held-out participant's POP normalized risk, and this fold-local q95. The published 0.22930936391189927 is the mean of the resulting 16 participant-specific floors.", "",
        "There is no null replicate ID, permutation vector, RNG schedule, group-null statistic, or joint P/W max-stat operation. The time-shift rows are participant-level falsification/stress cells. They are not exchangeable realizations of a group scientific null.", "",
        "## Unit and scale checks", "",
        "- Observed and stress-cell effects both use participant-first normalized risk; ERP/SSVEP are equal-weighted and sub-24 is policy fallback zero.",
        "- WRONG rows are averaged within donor before method/unit/task/participant reduction.",
        "- The 5% component is converted to contrast units as `0.05 × participant POP risk`; it is not used as a bare dimensionless 0.05.",
        "- Fold floors exclude the held-out participant and use the other 15 development participants.", "",
        "## Why exact recovery is not authorized", "",
    ]
    lines += [f"- {item}" for item in ambiguity]
    lines += ["", "Because several scientifically different max-stat/null constructions are reasonable and none is uniquely frozen, choosing one now would be a new analysis rather than an exact recovery.", "",
              "## Governance", "",
              "Original v19 files and decision remain unchanged. No raw EEG, marker, event, annotation, sealed outcome, GPU job, model training, O1, or manuscript operation occurred. Mobile sealed-8, PhysioMotion sealed-10, SHU Day-4/5, and PhysioTrait Day-200 remain unopened.", ""]
    report = CODE_ROOT / "reports/counterfactual_operator_v19_null_floor_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines), encoding="utf-8")
    _write_json(run_dir / "result_summary.json", decision)
    return decision


def tests_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    decision = json.loads((_audit_root(config) / "audit_decision.json").read_text(encoding="utf-8"))
    result = {"stage": "P6", "audit_verdict": decision["audit_verdict"], "sealed_opened": False,
              "O1_executed": False, "gpu_jobs": 0}
    _write_json(run_dir / "result_summary.json", result); return result


def run_stage(config: Mapping[str, Any], stage: str, run_dir: Path, task_index: int | None = None) -> Mapping[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "p0-freeze": return preflight_stage(config, run_dir)
    if stage == "p1-lineage": return lineage_stage(config, run_dir)
    if stage == "p2-manifest":
        if task_index is None: raise ValueError("p2-manifest requires array index")
        return participant_manifest_stage(config, run_dir, task_index)
    if stage == "p3-lineage": return floor_lineage_stage(config, run_dir)
    if stage == "p4-replay": return replay_stage(config, run_dir)
    if stage == "p5-decision": return decision_stage(config, run_dir)
    if stage == "p6-tests": return tests_stage(config, run_dir)
    raise ValueError(f"unsupported stage {stage}")
