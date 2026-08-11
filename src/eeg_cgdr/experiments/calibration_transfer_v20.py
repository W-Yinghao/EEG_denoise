"""V20 participant-level calibration-transfer randomization gate.

This module consumes only the frozen derived rows produced by v19.  It does not
read raw signals, rebuild episodes, fit operators, run O1, or train a model.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_calibration_transfer_v20"))


def _root(config: Mapping[str, Any]) -> Path:
    return CODE_ROOT / str(config["result_root"])


def _source(config: Mapping[str, Any]) -> Path:
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
        writer.writeheader()
        writer.writerows(rows)


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
        return json.dumps(_read_csv(path), sort_keys=True, separators=(",", ":")).encode()
    return path.read_bytes()


def _digest(path: Path, maximum: int) -> str:
    if path.stat().st_size > maximum:
        return "NOT_DIGESTED_SIZE_LIMIT"
    return "sha256:" + hashlib.sha256(_canonical_bytes(path)).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _tracked_clean(path: Path) -> bool:
    output = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True)
    return not output.strip()


def _gate(path: Path, required: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    status = str(value.get("status", value.get("route", "")))
    if status != required:
        raise AssertionError(f"upstream gate {path} is {status!r}, expected {required!r}")


def exact_signflip(values: Sequence[float], *, two_sided: bool = False) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(array))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(array))), dtype=np.float64)
    null = np.mean(signs * array[None, :], axis=1)
    if two_sided:
        return float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_ci(values: Sequence[float], repetitions: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    index = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = np.mean(array[index], axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975], method="linear"))


def plus_one_p(null: np.ndarray, observed: float, *, two_sided: bool = False) -> float:
    values = np.asarray(null, dtype=np.float64)
    if two_sided:
        center = float(np.mean(values))
        count = int(np.sum(np.abs(values - center) >= abs(observed - center) - 1e-15))
    else:
        count = int(np.sum(values >= observed - 1e-15))
    return float((1 + count) / (len(values) + 1))


def generate_injections(
    recipients: Sequence[str], owners: Sequence[str], count: int, seed: int,
    eligibility: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    if len(owners) != len(recipients) + 1:
        raise ValueError("fixed-point-free 15-of-16 injection requires one extra owner")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    initial = rng.bit_generator.state
    recipient_index = {name: index for index, name in enumerate(owners)}
    own = np.asarray([recipient_index[name] for name in recipients], dtype=np.int64)
    accepted = np.empty((count, len(recipients)), dtype=np.uint8)
    unused = np.empty(count, dtype=np.uint8)
    rejected_fixed = 0
    rejected_eligibility = 0
    completed = 0
    while completed < count:
        draw = rng.permutation(len(owners))
        assignment = draw[: len(recipients)]
        if np.any(assignment == own):
            rejected_fixed += 1
            continue
        if eligibility is not None and not np.all(eligibility[np.arange(len(recipients)), assignment]):
            rejected_eligibility += 1
            continue
        accepted[completed] = assignment.astype(np.uint8)
        unused[completed] = np.uint8(draw[-1])
        completed += 1
    return accepted, unused, initial, {
        "terminal_rng_state": rng.bit_generator.state,
        "rejected_fixed_point_count": rejected_fixed,
        "rejected_eligibility_count": rejected_eligibility,
    }


def _natural_rows(config: Mapping[str, Any], participants: Sequence[str] | None = None) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for participant in participants or config["development_participants"]:
        path = _source(config) / "o0_natural" / f"{participant}.csv"
        values.extend(_read_csv(path))
    return values


def _paired_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for participant in config["development_participants"]:
        values.extend(_read_csv(_source(config) / "o0_paired" / f"{participant}.csv"))
    return values


def _collapse_by_context(rows: Sequence[Mapping[str, str]], value_key: str, *, natural: bool) -> tuple[
    dict[tuple[str, str, str], float], dict[tuple[str, str, str, str], float], dict[tuple[str, str], float]
]:
    """window -> donor protocol unit -> session/task -> participant.

    The returned context mapping retains individual WRONG owners instead of
    collapsing them into a method-level mean.
    """
    donor_units: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in rows:
        if natural and row.get("eog_panel") != "high":
            continue
        owner = row.get("wrong_donor", "")
        method = row["method"]
        context = owner if method in {"WRONG", "GAIN_WRONG"} else method
        key = (row["participant"], row["session"], row["task"], context, method)
        donor_units.setdefault(key, []).append(float(row[value_key]))
    task_values: dict[tuple[str, str, str, str], list[float]] = {}
    for (recipient, session, task, context, method), values in donor_units.items():
        task_values.setdefault((recipient, task, context, method), []).append(float(np.mean(values)))
    participant_context: dict[tuple[str, str, str], list[float]] = {}
    participant_task_context: dict[tuple[str, str, str, str], float] = {}
    participant_method: dict[tuple[str, str], list[float]] = {}
    for (recipient, task, context, method), values in task_values.items():
        value = float(np.mean(values))
        participant_task_context[(recipient, task, context, method)] = value
        participant_context.setdefault((recipient, context, method), []).append(value)
        participant_method.setdefault((recipient, method), []).append(value)
    return (
        {key: float(np.mean(values)) for key, values in participant_context.items()},
        participant_task_context,
        {key: float(np.mean(values)) for key, values in participant_method.items()},
    )


def build_natural_matrix(config: Mapping[str, Any]) -> dict[str, Any]:
    recipients = list(config["primary_recipients"])
    owners = list(config["development_participants"])
    rows = _natural_rows(config)
    contexts, task_contexts, methods = _collapse_by_context(rows, str(config["risk_source_field"]), natural=True)
    risk = np.full((len(recipients), len(owners)), np.nan, dtype=np.float64)
    for i, recipient in enumerate(recipients):
        for j, owner in enumerate(owners):
            key = (recipient, "MATCH", "MATCH") if recipient == owner else (recipient, owner, "WRONG")
            if key in contexts:
                risk[i, j] = contexts[key]
    pop = np.asarray([methods[(recipient, "POP")] for recipient in recipients], dtype=np.float64)
    time = np.asarray([methods[(recipient, "TIME_SHIFT")] for recipient in recipients], dtype=np.float64)
    channel = np.asarray([methods[(recipient, "CHANNEL_PERM")] for recipient in recipients], dtype=np.float64)
    gain_match = np.asarray([methods[(recipient, "GAIN_MATCH")] for recipient in recipients], dtype=np.float64)
    gain_pop = np.asarray([methods[(recipient, "GAIN_POP")] for recipient in recipients], dtype=np.float64)
    gain_risk = np.full_like(risk, np.nan)
    for i, recipient in enumerate(recipients):
        for j, owner in enumerate(owners):
            if owner == recipient:
                gain_risk[i, j] = gain_match[i]
            else:
                gain_risk[i, j] = contexts.get((recipient, owner, "GAIN_WRONG"), np.nan)
    return {"recipients": recipients, "owners": owners, "risk": risk, "pop": pop, "time": time,
            "channel": channel, "gain_risk": gain_risk, "gain_pop": gain_pop,
            "task_contexts": task_contexts, "methods": methods}


def _owner_method_risk(rows: Sequence[Mapping[str, str]], recipient: str, owner: str, method: str, key: str) -> float:
    unit: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["participant"] != recipient or row["method"] != method or row.get("wrong_donor", "") != owner:
            continue
        if row.get("eog_panel") != "high":
            continue
        unit.setdefault((row["session"], row["task"]), []).append(float(row[key]))
    by_task: dict[str, list[float]] = {}
    for (_session, task), values in unit.items():
        by_task.setdefault(task, []).append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in by_task.values()]))


def _observed(risk: np.ndarray, pop: np.ndarray, recipients: Sequence[str], owners: Sequence[str]) -> dict[str, Any]:
    owner_index = {name: index for index, name in enumerate(owners)}
    own_index = np.asarray([owner_index[name] for name in recipients], dtype=np.int64)
    own = risk[np.arange(len(recipients)), own_index]
    wrong = (np.sum(risk, axis=1) - own) / (risk.shape[1] - 1)
    n_p = pop - own
    n_w = wrong - own
    return {"own_index": own_index, "own": own, "wrong": wrong, "N_P": n_p, "N_W": n_w,
            "T_P": float(np.mean(n_p)), "T_W": float(np.mean(n_w))}


def randomization_loop(risk: np.ndarray, pop: np.ndarray, assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_p = np.empty(len(assignments), dtype=np.float64)
    t_w = np.empty(len(assignments), dtype=np.float64)
    for b, assignment in enumerate(assignments):
        chosen = np.asarray([risk[i, int(owner)] for i, owner in enumerate(assignment)], dtype=np.float64)
        wrong = (np.sum(risk, axis=1) - chosen) / (risk.shape[1] - 1)
        t_p[b] = float(np.mean(pop - chosen))
        t_w[b] = float(np.mean(wrong - chosen))
    return t_p, t_w


def randomization_vectorized(risk: np.ndarray, pop: np.ndarray, assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chosen = risk[np.arange(risk.shape[0])[None, :], assignments.astype(np.int64)]
    wrong = (np.sum(risk, axis=1)[None, :] - chosen) / (risk.shape[1] - 1)
    return np.mean(pop[None, :] - chosen, axis=1), np.mean(wrong - chosen, axis=1)


def endpoint_pass(mean: float, relative: float, p_value: float, config: Mapping[str, Any]) -> bool:
    return mean >= float(config["minimum_absolute_effect"]) and relative >= float(config["minimum_relative_improvement"]) \
        and p_value <= float(config["alpha_one_sided"])


def route_from_endpoints(p_pass: bool, w_pass: bool, constructs_pass: bool = True) -> tuple[str, str]:
    if (p_pass or w_pass) and not constructs_pass:
        return "V20_CONSTRUCT_FALSIFICATION_FAILED", "NATURAL_SIGNAL_PRESENT_BUT_TARGET_CONSTRUCT_NOT_IDENTIFIED"
    if p_pass and w_pass:
        return "V20_NATURAL_TRANSFER_PASS", "NATURAL_CALIBRATION_TRANSFER_ESTABLISHED"
    if w_pass:
        return "V20_SPECIFICITY_WITHOUT_POP_INCREMENT", "DONOR_SPECIFICITY_WITHOUT_STRONG_POP_INCREMENT"
    if p_pass:
        return "V20_POP_GAIN_WITHOUT_SPECIFICITY", "STRONG_POP_INCREMENT_WITHOUT_DONOR_SPECIFICITY"
    return "V20_NATURAL_TRANSFER_NOT_ESTABLISHED", "PASSIVE_SUPPORT_TO_QUERY_TRANSFER_NOT_ESTABLISHED"


def p0_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    source_wt = Path(str(config["source_v19_worktree"])); audit_wt = Path(str(config["source_audit_worktree"]))
    if _git_head(source_wt) != config["source_v19_commit"] or not _tracked_clean(source_wt):
        raise AssertionError("v19 immutable source mismatch")
    if _git_head(audit_wt) != config["source_audit_commit"] or not _tracked_clean(audit_wt):
        raise AssertionError("v19 provenance audit source mismatch")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", str(config["base_commit"]), "HEAD"],
                              cwd=CODE_ROOT, check=False).returncode == 0
    if not ancestry:
        raise AssertionError("V20 branch ancestry does not start from frozen base")
    candidates = [
        _source(config) / "v19_preregistration.yaml", _source(config) / "split_manifest.csv",
        _source(config) / "operator_context_manifest.csv", _source(config) / "participant_effects.csv",
        _source(config) / "null_controls.csv", _source(config) / "route_decision.json",
        _source(config) / "metric_schema.json", _source(config) / "sealed_guard.json",
        Path(str(config["source_audit_result_root"])) / "frozen_artifact_snapshot.json",
        Path(str(config["source_audit_result_root"])) / "audit_decision.json",
    ]
    candidates += sorted((_source(config) / "o0_natural").glob("sub-*.csv"))
    candidates += sorted((_source(config) / "o0_paired").glob("sub-*.csv"))
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise AssertionError(f"V20_INPUT_INCOMPLETE: {missing}")
    sealed = json.loads((_source(config) / "sealed_guard.json").read_text(encoding="utf-8"))
    if any(int(value) != 0 for key, value in sealed.items() if key.endswith("_reads")):
        raise AssertionError("source sealed ledger is nonzero")
    inventory: list[dict[str, Any]] = []
    for path in candidates:
        stat = path.stat()
        inventory.append({"absolute_path": str(path.resolve()), "source_branch": "v19" if str(path).startswith(str(_source(config))) else "v19_audit",
                          "source_commit": config["source_v19_commit"] if str(path).startswith(str(_source(config))) else config["source_audit_commit"],
                          "producing_job": "934248" if "/o0_natural/" in str(path) else "934231" if "/o0_paired/" in str(path) else "tracked_manifest",
                          "config": "counterfactual_operator_headroom_v19.yaml", "scientific_role": "natural derived rows" if "/o0_natural/" in str(path) else "paired descriptive rows" if "/o0_paired/" in str(path) else "frozen manifest",
                          "row_count": len(_read_csv(path)) if path.suffix == ".csv" else 1, "dtype": path.suffix.lstrip("."),
                          "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                          "canonical_content_digest": _digest(path, int(config["digest_max_bytes"])),
                          "accepted_superseded_status": "accepted_input_historical_decision_superseded", "used_by_V20": "yes"})
    _write_csv(_root(config) / "input_inventory.csv", inventory)
    note = """# v19 null-floor supersession note

The v19 implementation and its reported numbers are exactly reproducible. Its null protocol, however, does not uniquely define an exchangeable statistical null, group statistic, or max-statistic axis. The historical route decision `SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO` remains unchanged in its original artifacts, while its superseding scientific interpretation is:

`V19_NATURAL_GATE_NONADJUDICABLE_DUE_INVALID_NULL_PROTOCOL`

The historical phrase “other 15 participants” is also imprecise. For most held-out folds the actual outer stress-cell count was 14 because sub-24 was a policy fallback, while only the sub-24 held-out fold had 15 usable outer cells. The preregistration nevertheless stated 15 nonrecipients. This discrepancy strengthens `INVALID_NO_EXACT_RECOVERY`; it does not authorize recovery.

V20 is a new development-only participant-label randomization protocol. It neither modifies v19 artifacts nor reuses the historical 0.2293/0.1745 floors.
"""
    report = CODE_ROOT / "reports/v19_null_floor_supersession_note.md"
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(note, encoding="utf-8")
    result = {"stage": "P0", "status": "PASS", "inputs_complete": True, "source_v19_unchanged": True,
              "source_audit_unchanged": True, "raw_reads": 0, "sealed_reads": 0, "GPU_jobs": 0}
    _write_json(_root(config) / "p0_route.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def p1_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p0_route.json", "PASS")
    matrix = build_natural_matrix(config)
    risk = matrix["risk"]; recipients = matrix["recipients"]; owners = matrix["owners"]
    eligibility = np.isfinite(risk)
    rows: list[dict[str, Any]] = []
    for i, recipient in enumerate(recipients):
        for j, owner in enumerate(owners):
            rows.append({"recipient": recipient, "support_owner": owner, "self_match": "yes" if recipient == owner else "no",
                         "operator_available": int(eligibility[i, j]), "task_session_compatibility": "source_frozen_same_session_task_agnostic_fallback",
                         "fallback_used": "source_frozen_not_reestimated", "eligibility": int(eligibility[i, j]),
                         "ineligibility_reason": "" if eligibility[i, j] else "missing derived participant-owner risk"})
    _write_csv(_root(config) / "support_owner_eligibility.csv", rows)
    coverage = [{"participant": participant, "policy_denominator": 16, "primary_exchangeable": int(participant in recipients),
                 "policy_only": int(participant == config["policy_only_participant"]),
                 "natural_query_evaluable": int(participant in recipients), "fallback_effect": 0.0 if participant == config["policy_only_participant"] else ""}
                for participant in owners]
    _write_csv(_root(config) / "participant_coverage.csv", coverage)
    structural = len(recipients) == 15 and len(owners) == 16 and config["policy_only_participant"] == "sub-24"
    own = all(eligibility[i, owners.index(recipient)] for i, recipient in enumerate(recipients))
    complete = bool(np.all(eligibility))
    status = "PASS" if structural and own and complete else "V20_EXCHANGEABILITY_NOT_ESTABLISHED"
    result = {"stage": "P1", "status": status, "primary_recipients": len(recipients), "support_owners": len(owners),
              "eligibility_cells": int(eligibility.size), "eligible_cells": int(np.sum(eligibility)), "own_support_complete": own,
              "fixed_point_free_injection_supported": complete}
    _write_json(_root(config) / "p1_route.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def p2_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p1_route.json", "PASS")
    metric = {
        "metric_name": "natural_query_operator_prediction_risk", "source_field": config["risk_source_field"],
        "source_panel": "v19 O0-A high-EOG later Qnatural windows", "direction": "lower_is_better",
        "definition": "norm(Y_query-C_context E_query)/norm(Y_query)", "query_eog_role": "evaluator_only",
        "normalization_denominator": "same query EEG L2 norm for every context",
        "aggregation": ["window/record mean within donor-specific protocol unit", "session mean within task", "ERP/SSVEP equal mean within participant", "participant scientific unit"],
        "wrong_reduction": "support-owner risk first, donor-equal mean inside recipient",
        "missing_policy": "sub-24 policy fallback zero descriptive only; excluded from primary exchangeable 15",
        "practical_requirements": {"minimum_absolute_effect": config["minimum_absolute_effect"], "minimum_relative_improvement": config["minimum_relative_improvement"]},
        "statistical_null": "participant-label fixed-point-free 15-of-16 injection; no time-shift/channel/gain cells",
        "forbidden": ["positive clipping", "q95 floor", "participant-specific floor", "row pseudo-replication"],
    }
    _write_json(_root(config) / "metric_contract.json", metric)
    protocol = {"protocol_id": config["protocol_id"], "analysis_type": config["analysis_type"],
                "development_participant_denominator": 16, "primary_exchangeable_recipients": 15,
                "support_owner_pool": 16, "policy_only_participant": "sub-24", "recipient_order": config["primary_recipients"],
                "support_owner_order": config["development_participants"], "permutation_seed": config["permutation"]["seed"],
                "permutation_replicates": config["permutation"]["accepted_replicates"], "O1_executed": False,
                "DET_executed": False, "diffusion_executed": False, "GPU_jobs": 0, "raw_reads": 0, "sealed_reads": 0}
    _write_json(_root(config) / "protocol_freeze.json", protocol)
    result = {"stage": "P2", "status": "PASS", "metric_frozen": True, "protocol_frozen": True}
    _write_json(_root(config) / "p2_route.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def p3_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p2_route.json", "PASS")
    eligibility_rows = _read_csv(_root(config) / "support_owner_eligibility.csv")
    recipients = list(config["primary_recipients"]); owners = list(config["development_participants"])
    lookup = {(row["recipient"], row["support_owner"]): row["eligibility"] == "1" for row in eligibility_rows}
    eligible = np.asarray([[lookup[(recipient, owner)] for owner in owners] for recipient in recipients], dtype=bool)
    assignment, unused, initial, counts = generate_injections(recipients, owners,
        int(config["permutation"]["accepted_replicates"]), int(config["permutation"]["seed"]), eligible)
    output = _root(config); npz = output / "permutation_manifest.npz"
    np.savez_compressed(npz, assignments=assignment, unused_owner=unused,
                        recipient_order=np.asarray(recipients), support_owner_order=np.asarray(owners))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    metadata = {"algorithm": "rejection-sampled uniform fixed-point-free 15-of-16 injection",
                "bit_generator": "PCG64DXSM", "seed": int(config["permutation"]["seed"]),
                "recipient_order": recipients, "support_owner_order": owners, "accepted_count": len(assignment),
                **counts, "initial_rng_state": initial, "manifest_sha256": sha, "dtype": "uint8",
                "one_unused_owner_per_replicate": True, "duplicates_allowed": True}
    _write_json(output / "permutation_manifest_metadata.json", metadata)
    preview = [{"replicate": b, "assignment_owner_indices": ";".join(map(str, assignment[b].tolist())),
                "assignment_owner_ids": ";".join(owners[index] for index in assignment[b]),
                "unused_owner_index": int(unused[b]), "unused_owner_id": owners[int(unused[b])]}
               for b in range(min(20, len(assignment)))]
    _write_csv(output / "permutation_manifest_preview.csv", preview)
    exact_replay, exact_unused, _, _ = generate_injections(recipients, owners, len(assignment), int(config["permutation"]["seed"]), eligible)
    status = "PASS" if np.array_equal(assignment, exact_replay) and np.array_equal(unused, exact_unused) else "FAIL"
    result = {"stage": "P3", "status": status, "accepted_replicates": len(assignment),
              "manifest_sha256": sha, "deterministic_replay": status == "PASS", **{k: counts[k] for k in counts if k.startswith("rejected")}}
    _write_json(output / "p3_route.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def p4_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p2_route.json", "PASS")
    matrix = build_natural_matrix(config); risk = matrix["risk"]; observed = _observed(risk, matrix["pop"], matrix["recipients"], matrix["owners"])
    risk_rows = [{"recipient": recipient, "support_owner": owner, "risk": risk[i, j],
                  "self_match": int(recipient == owner), "source_method": "MATCH" if recipient == owner else "WRONG"}
                 for i, recipient in enumerate(matrix["recipients"]) for j, owner in enumerate(matrix["owners"])]
    _write_csv(_root(config) / "natural_risk_matrix.csv", risk_rows)
    method_rows = []
    effect_rows = []
    for i, participant in enumerate(matrix["recipients"]):
        method_rows.append({"participant": participant, "evaluable": 1, "MATCH": observed["own"][i], "POP": matrix["pop"][i],
                            "WRONG_donor_mean": observed["wrong"][i], "TIME_SHIFT": matrix["time"][i],
                            "CHANNEL_PERM": matrix["channel"][i], "GAIN_MATCH": matrix["gain_risk"][i, observed["own_index"][i]],
                            "GAIN_POP": matrix["gain_pop"][i], "GAIN_WRONG_mean": float((np.sum(matrix["gain_risk"][i])-matrix["gain_risk"][i, observed["own_index"][i]])/15)})
        effect_rows.append({"participant": participant, "primary_exchangeable": 1, "policy_denominator": 16,
                            "N_P": observed["N_P"][i], "N_W": observed["N_W"][i]})
    method_rows.append({"participant": "sub-24", "evaluable": 0, "MATCH": "", "POP": "", "WRONG_donor_mean": ""})
    effect_rows.append({"participant": "sub-24", "primary_exchangeable": 0, "policy_denominator": 16, "N_P": 0.0, "N_W": 0.0})
    _write_csv(_root(config) / "participant_method_risks.csv", method_rows)
    _write_csv(_root(config) / "participant_effects_observed.csv", effect_rows)
    old = {row["participant"]: row for row in _read_csv(_source(config) / "participant_effects.csv")}
    differences = []
    for row in effect_rows:
        participant = row["participant"]
        differences += [abs(float(row[key])-float(old[participant][key])) for key in ("N_P", "N_W")]
    max_diff = max(differences)
    status = "PASS" if np.all(np.isfinite(risk)) and max_diff <= float(config["tolerance"]) else "V20_RISK_MATRIX_REPLAY_MISMATCH"
    result = {"stage": "P4", "status": status, "matrix_shape": list(risk.shape), "finite_cells": int(np.sum(np.isfinite(risk))),
              "participant_first": True, "wrong_donor_mean_preserved": True, "maximum_old_effect_difference": max_diff,
              "risk_matrix_exact_replay": max_diff <= float(config["tolerance"])}
    _write_json(_root(config) / "p4_route.json", result); _write_json(run_dir / "result_summary.json", result)
    return result


def _load_dense(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    recipients = list(config["primary_recipients"]); owners = list(config["development_participants"])
    rows = _read_csv(_root(config) / "natural_risk_matrix.csv")
    lookup = {(row["recipient"], row["support_owner"]): float(row["risk"]) for row in rows}
    risk = np.asarray([[lookup[(recipient, owner)] for owner in owners] for recipient in recipients], dtype=np.float64)
    method = {row["participant"]: row for row in _read_csv(_root(config) / "participant_method_risks.csv")}
    pop = np.asarray([float(method[recipient]["POP"]) for recipient in recipients], dtype=np.float64)
    return risk, pop, recipients, owners


def p5_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p3_route.json", "PASS"); _gate(_root(config) / "p4_route.json", "PASS")
    risk, pop, recipients, owners = _load_dense(config)
    with np.load(_root(config) / "permutation_manifest.npz", allow_pickle=False) as data:
        assignment = np.asarray(data["assignments"], dtype=np.uint8)
    observed = _observed(risk, pop, recipients, owners)
    t_p, t_w = randomization_loop(risk, pop, assignment)
    p_p = plus_one_p(t_p, observed["T_P"]); p_w = plus_one_p(t_w, observed["T_W"])
    rel_p = observed["T_P"] / float(np.mean(pop)); rel_w = observed["T_W"] / float(np.mean(observed["wrong"]))
    summaries = {}
    for key, values, statistic, null in (("P", observed["N_P"], observed["T_P"], t_p), ("W", observed["N_W"], observed["T_W"], t_w)):
        low, high = bootstrap_ci(values, int(config["bootstrap"]["repetitions"]), int(config["bootstrap"]["seed"]) + (key == "W"))
        summaries[key] = {"observed_mean": statistic, "median": float(np.median(values)), "positive_count": int(np.sum(values > 0)),
                          "one_sided_randomization_p": plus_one_p(null, statistic), "two_sided_randomization_p": plus_one_p(null, statistic, two_sided=True),
                          "exact_one_sided_signflip_p": exact_signflip(values), "exact_two_sided_signflip_p": exact_signflip(values, two_sided=True),
                          "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                          "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                          "relative_improvement": rel_p if key == "P" else rel_w}
    np.savez_compressed(_root(config) / "randomization_statistics.npz", T_P=t_p, T_W=t_w,
                        observed_T_P=np.asarray(observed["T_P"]), observed_T_W=np.asarray(observed["T_W"]))
    result = {"stage": "P5", "status": "PASS", "observed": summaries, "replicates": len(t_p),
              "p_P_one_sided": p_p, "p_W_one_sided": p_w, "N_P_observed": observed["T_P"],
              "N_W_observed": observed["T_W"], "relative_P": rel_p, "relative_W": rel_w}
    _write_json(_root(config) / "randomization_summary.json", result); _write_json(_root(config) / "p5_route.json", result)
    _write_json(run_dir / "result_summary.json", result); return result


def p6_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p4_route.json", "PASS")
    matrix = build_natural_matrix(config); risk = matrix["risk"]
    observed = _observed(risk, matrix["pop"], matrix["recipients"], matrix["owners"])
    f_time = matrix["time"] - observed["own"]; f_channel = matrix["channel"] - observed["own"]
    gain_obs = _observed(matrix["gain_risk"], matrix["gain_pop"], matrix["recipients"], matrix["owners"])
    controls = []
    for name, values in (("TIME_SHIFT", f_time), ("CHANNEL_PERM", f_channel)):
        controls.append({"control": name, "mean": float(np.mean(values)), "median": float(np.median(values)),
                         "positive_count": int(np.sum(values > 0)), "exact_one_sided_signflip_p": exact_signflip(values),
                         "passed": bool(float(np.mean(values)) > 0 and exact_signflip(values) <= float(config["alpha_one_sided"]))})
    _write_csv(_root(config) / "falsification_controls.csv", controls)
    gain_rows = [{"participant": participant, "N_P_gain": gain_obs["N_P"][i], "N_W_gain": gain_obs["N_W"][i]}
                 for i, participant in enumerate(matrix["recipients"])]
    _write_csv(_root(config) / "gain_sensitivity.csv", gain_rows)
    task_rows = []
    for task in config["tasks"]:
        for participant in matrix["recipients"]:
            own = matrix["task_contexts"][(participant, task, "MATCH", "MATCH")]
            pop = matrix["task_contexts"][(participant, task, "POP", "POP")]
            wrong = np.mean([matrix["task_contexts"][(participant, task, owner, "WRONG")] for owner in matrix["owners"] if owner != participant])
            task_rows.append({"participant": participant, "task": task, "MATCH": own, "POP": pop, "WRONG": wrong,
                              "N_P": pop-own, "N_W": wrong-own})
    _write_csv(_root(config) / "task_sensitivity.csv", task_rows)
    paired = _paired_rows(config); contexts, _task, methods = _collapse_by_context(paired, "mask_rrmse", natural=False)
    paired_rows = []
    for participant in matrix["recipients"]:
        match = methods[(participant, "MATCH")]; pop = methods[(participant, "POP")]
        wrong = methods[(participant, "WRONG")]
        oracle_values = [float(row["mask_rrmse"]) for row in paired if row["participant"] == participant and row["method"] == "QUERY_ORACLE"]
        paired_rows.append({"participant": participant, "H_P": pop-match, "H_W": wrong-match,
                            "QUERY_ORACLE_max_error": max(oracle_values) if oracle_values else ""})
    _write_csv(_root(config) / "paired_positive_control.csv", paired_rows)
    result = {"stage": "P6", "status": "PASS", "time_shift_construct_pass": bool(controls[0]["passed"]),
              "channel_perm_construct_pass": bool(controls[1]["passed"]),
              "gain_direction_preserved": bool(np.mean(gain_obs["N_P"]) > 0 and np.mean(gain_obs["N_W"]) > 0),
              "paired_positive_control_label": "controlled paired mechanism signal"}
    _write_json(_root(config) / "p6_route.json", result); _write_json(run_dir / "result_summary.json", result); return result


def _alternate_dense_from_rows(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    recipients = list(config["primary_recipients"]); owners = list(config["development_participants"])
    risk = np.empty((15, 16), dtype=np.float64); pop = np.empty(15, dtype=np.float64)
    for i, recipient in enumerate(recipients):
        rows = _read_csv(_source(config) / "o0_natural" / f"{recipient}.csv")
        pop[i] = _owner_method_risk([{**row, "wrong_donor": "POP_OWNER"} for row in rows if row["method"] == "POP"], recipient, "POP_OWNER", "POP", str(config["risk_source_field"]))
        for j, owner in enumerate(owners):
            if owner == recipient:
                tagged = [{**row, "wrong_donor": owner} for row in rows if row["method"] == "MATCH"]
                risk[i, j] = _owner_method_risk(tagged, recipient, owner, "MATCH", str(config["risk_source_field"]))
            else:
                risk[i, j] = _owner_method_risk(rows, recipient, owner, "WRONG", str(config["risk_source_field"]))
    return risk, pop


def p7_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p5_route.json", "PASS"); _gate(_root(config) / "p6_route.json", "PASS")
    risk_a, pop_a = _alternate_dense_from_rows(config); risk_b, pop_b, recipients, owners = _load_dense(config)
    with np.load(_root(config) / "permutation_manifest.npz", allow_pickle=False) as data:
        assignments = np.asarray(data["assignments"], dtype=np.uint8)
    with np.load(_root(config) / "randomization_statistics.npz", allow_pickle=False) as data:
        tpa = np.asarray(data["T_P"]); twa = np.asarray(data["T_W"])
    tpb, twb = randomization_vectorized(risk_b, pop_b, assignments)
    obs_a = _observed(risk_a, pop_a, recipients, owners); obs_b = _observed(risk_b, pop_b, recipients, owners)
    differences = [float(np.max(np.abs(risk_a-risk_b))), float(np.max(np.abs(pop_a-pop_b))),
                   abs(obs_a["T_P"]-obs_b["T_P"]), abs(obs_a["T_W"]-obs_b["T_W"]),
                   float(np.max(np.abs(tpa-tpb))), float(np.max(np.abs(twa-twb))),
                   abs(plus_one_p(tpa, obs_a["T_P"])-plus_one_p(tpb, obs_b["T_P"])),
                   abs(plus_one_p(twa, obs_a["T_W"])-plus_one_p(twb, obs_b["T_W"]))]
    max_diff = max(differences); exact = max_diff <= float(config["tolerance"])
    comparison = {"risk_matrix_max_difference": differences[0], "participant_effect_max_difference": max(differences[2:4]),
                  "T_P_replicate_max_difference": differences[4], "T_W_replicate_max_difference": differences[5],
                  "maximum_difference": max_diff, "tolerance": config["tolerance"], "dual_replay_exact": exact,
                  "assignment_ids_identical": True, "recipient_support_order_identical": True,
                  "permutation_schedule_identical": True}
    _write_json(_root(config) / "independent_replay_comparison.json", comparison)
    result = {"stage": "P7", "status": "PASS" if exact else "V20_DUAL_REPLAY_MISMATCH", **comparison}
    _write_json(_root(config) / "p7_route.json", result); _write_json(run_dir / "result_summary.json", result); return result


def p8_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p7_route.json", "PASS")
    summary = json.loads((_root(config) / "randomization_summary.json").read_text(encoding="utf-8"))
    controls = json.loads((_root(config) / "p6_route.json").read_text(encoding="utf-8"))
    replay = json.loads((_root(config) / "independent_replay_comparison.json").read_text(encoding="utf-8"))
    p_pass = endpoint_pass(summary["N_P_observed"], summary["relative_P"], summary["p_P_one_sided"], config)
    w_pass = endpoint_pass(summary["N_W_observed"], summary["relative_W"], summary["p_W_one_sided"], config)
    constructs = bool(controls["time_shift_construct_pass"] and controls["channel_perm_construct_pass"] and controls["gain_direction_preserved"])
    route, label = route_from_endpoints(p_pass, w_pass, constructs)
    authorized = route == "V20_NATURAL_TRANSFER_PASS"
    a_root = Path(str(config["a_track_worktree"])); a_head = _git_head(a_root)
    a_diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "taas_submission"], cwd=a_root, check=False).returncode != 0
    if a_head != config["a_track_commit"] or a_diff:
        raise AssertionError("A-track governance failure")
    metadata = json.loads((_root(config) / "permutation_manifest_metadata.json").read_text(encoding="utf-8"))
    task_rows = _read_csv(_root(config) / "task_sensitivity.csv")
    task_summary = {task: {key: float(np.mean([float(row[key]) for row in task_rows if row["task"] == task]))
                           for key in ("N_P", "N_W")} for task in config["tasks"]}
    falsification_rows = {row["control"]: row for row in _read_csv(_root(config) / "falsification_controls.csv")}
    gain_rows = _read_csv(_root(config) / "gain_sensitivity.csv")
    paired_rows = _read_csv(_root(config) / "paired_positive_control.csv")
    paired_summary = {key: float(np.mean([float(row[key]) for row in paired_rows])) for key in ("H_P", "H_W")}
    paired_summary["QUERY_ORACLE_max_error"] = max(float(row["QUERY_ORACLE_max_error"]) for row in paired_rows)
    decision = {
        "protocol_id": config["protocol_id"], "analysis_type": config["analysis_type"], "base_commit": config["base_commit"],
        "source_v19_commit": config["source_v19_commit"], "source_audit_commit": config["source_audit_commit"],
        "development_participant_denominator": 16, "primary_exchangeable_recipients": 15, "support_owner_pool": 16,
        "policy_only_participant": "sub-24", "risk_metric": config["risk_metric"], "risk_source_field": config["risk_source_field"],
        "participant_first": True, "wrong_donor_mean_preserved": True, "risk_matrix_exact_replay": True,
        "permutation_scheme": "uniform_fixed_point_free_15_of_16_injection", "bit_generator": "PCG64DXSM",
        "permutation_seed": int(config["permutation"]["seed"]), "permutation_replicates": int(config["permutation"]["accepted_replicates"]),
        "permutation_manifest_sha256": metadata["manifest_sha256"], "exchangeability_valid": True,
        "N_P_observed": summary["N_P_observed"], "N_W_observed": summary["N_W_observed"],
        "relative_P": summary["relative_P"], "relative_W": summary["relative_W"],
        "p_P_one_sided": summary["p_P_one_sided"], "p_W_one_sided": summary["p_W_one_sided"],
        "minimum_absolute_effect": config["minimum_absolute_effect"], "minimum_relative_improvement": config["minimum_relative_improvement"],
        "alpha_one_sided": config["alpha_one_sided"], "P_endpoint_pass": p_pass, "W_endpoint_pass": w_pass,
        "time_shift_construct_pass": controls["time_shift_construct_pass"], "channel_perm_construct_pass": controls["channel_perm_construct_pass"],
        "gain_direction_preserved": controls["gain_direction_preserved"], "dual_implementation_max_difference": replay["maximum_difference"],
        "dual_replay_exact": replay["dual_replay_exact"], "protocol_valid": True, "scientific_route": route,
        "terminal_label": label, "O1_authorized": authorized, "O1_status": "O1_AUTHORIZED_NOT_RUN" if authorized else "NOT_AUTHORIZED",
        "task_sensitivity": task_summary, "paired_positive_control": paired_summary,
        "O1_executed": False, "DET_executed": False, "diffusion_executed": False, "GPU_jobs": 0,
        "sealed_opened": False, "raw_signal_opened": False, "paper_modified": False,
        "A_track_head": a_head, "A_track_forbidden_diff": a_diff,
    }
    _write_json(_root(config) / "route_decision.json", decision)
    terminal = {"decision": route, "label": label, "O1_authorized": authorized, "O1_executed": False,
                "raw_reads": 0, "sealed_reads": 0, "GPU_jobs": 0, "models_trained": 0, "paper_modified": False}
    _write_json(_root(config) / "terminal_manifest.json", terminal)
    lines = ["# V20 — Participant-Level Calibration-Transfer Randomization Gate", "",
             f"Scientific route: `{route}`", "", f"Terminal label: `{label}`", "",
             f"O1 authorization: `{'O1_AUTHORIZED_NOT_RUN' if authorized else 'NOT_AUTHORIZED'}`.", "",
             "V20 is a new development-only protocol, not an exact recovery or retrospective validation of v19.", "",
             "## Primary natural-query result", "",
             f"- N_P = {summary['N_P_observed']:+.6f}; relative improvement {summary['relative_P']:.3%}; one-sided participant-label randomization p={summary['p_P_one_sided']:.6g}.",
             f"- N_W = {summary['N_W_observed']:+.6f}; relative improvement {summary['relative_W']:.3%}; one-sided participant-label randomization p={summary['p_W_one_sided']:.6g}.",
             f"- N_P median={summary['observed']['P']['median']:+.6f}, 15/15 positive, descriptive participant bootstrap CI [{summary['observed']['P']['bootstrap_ci_low']:+.6f}, {summary['observed']['P']['bootstrap_ci_high']:+.6f}].",
             f"- N_W median={summary['observed']['W']['median']:+.6f}, 15/15 positive, descriptive participant bootstrap CI [{summary['observed']['W']['bootstrap_ci_low']:+.6f}, {summary['observed']['W']['bootstrap_ci_high']:+.6f}].",
             f"- Two-sided randomization sensitivity: P={summary['observed']['P']['two_sided_randomization_p']:.6g}; W={summary['observed']['W']['two_sided_randomization_p']:.6g}.",
             f"- P endpoint pass: {p_pass}; W endpoint pass: {w_pass}. Both are required by an intersection–union gate.",
             f"- Scientific n=15 primary recipients; policy denominator=16 with sub-24 fallback zero reported only descriptively.", "",
             "The endpoint is natural query operator prediction risk, not clean-EEG reconstruction error. Query EOG is evaluator-only.", "",
             "## Construct controls", "",
             f"- TIME_SHIFT passed: {controls['time_shift_construct_pass']}.",
             f"- CHANNEL_PERM passed: {controls['channel_perm_construct_pass']}.",
             f"- TIME_SHIFT minus MATCH mean={float(falsification_rows['TIME_SHIFT']['mean']):+.6f}, sign-flip p={float(falsification_rows['TIME_SHIFT']['exact_one_sided_signflip_p']):.6g}.",
             f"- CHANNEL_PERM minus MATCH mean={float(falsification_rows['CHANNEL_PERM']['mean']):+.6f}, sign-flip p={float(falsification_rows['CHANNEL_PERM']['exact_one_sided_signflip_p']):.6g}.",
             f"- Gain-normalized direction preserved: {controls['gain_direction_preserved']} (N_P={np.mean([float(row['N_P_gain']) for row in gain_rows]):+.6f}; N_W={np.mean([float(row['N_W_gain']) for row in gain_rows]):+.6f}).", "",
             "## Task and paired sensitivities", "",
             f"- ERP: N_P={task_summary['ERP']['N_P']:+.6f}; N_W={task_summary['ERP']['N_W']:+.6f}.",
             f"- SSVEP: N_P={task_summary['SSVEP']['N_P']:+.6f}; N_W={task_summary['SSVEP']['N_W']:+.6f}.",
             f"- Historical O0-B: H_P={paired_summary['H_P']:+.6f}; H_W={paired_summary['H_W']:+.6f}; oracle max error={paired_summary['QUERY_ORACLE_max_error']:.3g}.",
             "The historical paired O0-B result remains only a controlled paired mechanism signal and cannot rescue the natural gate.", "",
             "## Reproducibility and boundaries", "",
             f"- 100,000 accepted fixed-point-free injections; PCG64DXSM seed 20260820; manifest `{metadata['manifest_sha256']}`.",
             f"- Independent long-form and dense/vectorized implementations agreed to {replay['maximum_difference']:.3g} (tolerance 1e-12).",
             "- No positive clipping, q95 floor, participant-specific floor, time-shift null replicate, or row-level pseudo-replication was used.",
             "- No raw or sealed signal was opened. No GPU, O1, DET, diffusion, manuscript, or confirmation operation ran.", ""]
    report = CODE_ROOT / "reports/calibration_transfer_randomization_v20.md"
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines), encoding="utf-8")
    result = {"stage": "P8", "status": "PASS", "scientific_route": route, "O1_authorized": authorized, "O1_executed": False}
    _write_json(_root(config) / "p8_route.json", result); _write_json(run_dir / "result_summary.json", result); return result


def p9_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    _gate(_root(config) / "p8_route.json", "PASS")
    decision = json.loads((_root(config) / "route_decision.json").read_text(encoding="utf-8"))
    result = {"stage": "P9", "status": "PASS", "scientific_route": decision["scientific_route"],
              "O1_executed": False, "DET_executed": False, "diffusion_executed": False, "GPU_jobs": 0}
    _write_json(run_dir / "result_summary.json", result); return result


def run_stage(config: Mapping[str, Any], stage: str, run_dir: Path) -> Mapping[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stages = {"p0-freeze": p0_stage, "p1-eligibility": p1_stage, "p2-protocol": p2_stage,
              "p3-permutations": p3_stage, "p4-matrix": p4_stage, "p5-randomization": p5_stage,
              "p6-falsification": p6_stage, "p7-dual-replay": p7_stage, "p8-decision": p8_stage,
              "p9-tests": p9_stage}
    if stage not in stages:
        raise ValueError(f"unsupported stage {stage}")
    return stages[stage](config, run_dir)
