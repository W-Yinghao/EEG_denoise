"""Counterfactual Operator-Swap Headroom v19.

CPU-only, development-only experiment.  Inference packages never contain the
later evaluator operator; the latter is materialized beneath an evaluator-only
directory and is opened only by O0 scoring stages.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal

from eeg_cgdr.data.mobile_bci import (
    SealedParticipantAccessError,
    assert_development_access,
    metadata_inventory,
    read_development_record,
    read_source_eeg_eog,
)


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_counterfactual_operator_v19"))


def _root(config: Mapping[str, Any]) -> Path:
    return CODE_ROOT / str(config["output_root"])


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _ledger_path(config: Mapping[str, Any], participant: str) -> Path:
    return _root(config) / "resolved_path_ledger" / f"{participant}.csv"


def guarded_paths(
    config: Mapping[str, Any], participant: str, session: str, task: str,
    *, roles: Iterable[str], write_ledger: bool = True,
) -> list[dict[str, str]]:
    """Resolve paths only after an exact development allowlist check.

    This function is deliberately the only v19 path gateway.  Sealed IDs fail
    before Path.exists, Path.open, or MNE can be invoked.
    """
    development = tuple(str(value) for value in config["development_participants"])
    assert_development_access(participant, development)
    data_root = Path(str(config["data_root"]))
    processed = data_root / participant / session / "eeg"
    source = data_root / "sourcedata" / participant / session / "eeg"
    stem = f"{participant}_{session}_task-{task}_eeg"
    role_to_path = {
        "processed_header": processed / f"{stem}.vhdr",
        "processed_binary": processed / f"{stem}.eeg",
        "processed_marker": processed / f"{stem}.vmrk",
        "processed_channels": processed / f"{participant}_{session}_task-{task}_channels.tsv",
        "source_header": source / f"{stem}.vhdr",
        "source_binary": source / f"{stem}.eeg",
        "source_marker": source / f"{stem}.vmrk",
        "source_channels": source / f"{participant}_{session}_task-{task}_channels.tsv",
    }
    rows = [{
        "participant": participant, "session": session, "task": task,
        "role": role, "resolved_path": str(role_to_path[role]), "sealed": "0",
    } for role in roles]
    if write_ledger:
        path = _ledger_path(config, participant)
        old = _read_csv(path) if path.is_file() else []
        unique = {(row["session"], row["task"], row["role"], row["resolved_path"]): row for row in old + rows}
        _write_csv(path, list(unique.values()), fields=("participant", "session", "task", "role", "resolved_path", "sealed"))
    return rows


def _available(config: Mapping[str, Any], participant: str, session: str, task: str) -> bool:
    rows = guarded_paths(config, participant, session, task, roles=(
        "processed_header", "processed_binary", "processed_marker", "processed_channels",
        "source_header", "source_binary", "source_marker", "source_channels",
    ), write_ledger=False)
    return all(Path(row["resolved_path"]).is_file() for row in rows)


def _preregistration(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol_id": config["protocol_id"],
        "scientific_scope": "participant-session calibration; paired operator-swap semi-simulation",
        "forbidden_interpretations": [
            "stable brain physiology", "natural EEG counterfactual", "validated subject-aware denoising",
        ],
        "development_participants": list(config["development_participants"]),
        "sealed_participants": list(config["sealed_participants"]),
        "roles": {
            "S120": [0.0, 120.0], "guard_1": [120.0, 150.0],
            "Qgen": [150.0, 270.0], "guard_2": [270.0, 300.0],
            "Qnatural": [300.0, "record_end"],
        },
        "operator": dict(config["operator"]),
        "preprocessing": dict(config["preprocessing"]),
        "strong_population": "recipient-excluded participant-equal mean over all 15 other development participants",
        "missing_donor_record_rule": "same-session task-agnostic operator fallback: use the donor's other task S120; missing recipient query units remain unavailable",
        "query_forward": "EEG-only in O1; query EOG is evaluator-only in O0 and safety",
        "aggregation": "window/donor/unit -> task/session -> participant; policy n=16",
        "outer_null_floor": "fold-local q95 of max(0, POP-minus-EOG-time-shift) among the 15 nonrecipients, separately for O0-A/B",
        "o0_gate": dict(config["o0"]),
        "o1_gate": dict(config["o1"]),
        "frozen_before_scientific_signal": True,
    }


def preflight_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    output = _root(config)
    inventory = metadata_inventory(Path(str(config["data_root"])))
    dev = set(config["development_participants"])
    sealed = set(config["sealed_participants"])
    if dev & sealed or len(dev) != 16 or len(sealed) != 8:
        raise AssertionError("frozen development/sealed split is invalid")
    novelty = {
        "decision": "NOVEL_ESTIMAND_CONTINUE",
        "compared_experiments": ["mobile_bci_headroom_v4", "mobile_temporal_diffusion_v5"],
        "required_new_features": {
            "later_query_C_query": True,
            "byte_identical_x_e_y_mask_across_arms": True,
            "strong_pop_all_15_nonrecipients": True,
            "query_forward_eeg_only": True,
            "gain_time_shift_channel_geometry_nulls": True,
        },
        "prior_estimands": {
            "v4": "processed EEG+IMU temporal-support deterministic proxy; no later evaluator C_query",
            "v5": "learned temporal-support factorial; source EOG disabled; no paired operator swap",
        },
    }
    _write_json(output / "estimand_novelty_audit.json", novelty)
    import yaml
    (output / "v19_preregistration.yaml").parent.mkdir(parents=True, exist_ok=True)
    (output / "v19_preregistration.yaml").write_text(yaml.safe_dump(_preregistration(config), sort_keys=False), encoding="utf-8")
    split_rows = [{"participant": p, "role": "development", "policy_denominator": 16} for p in sorted(dev)]
    split_rows += [{"participant": p, "role": "sealed", "policy_denominator": 0} for p in sorted(sealed)]
    _write_csv(output / "split_manifest.csv", split_rows)
    sealed_guard = {
        "allowlist_checked_before_path_resolution_or_open": True,
        "development": sorted(dev), "sealed": sorted(sealed),
        "mobile_sealed_reads": 0, "physiomotion_sealed_reads": 0,
        "shu_day4_day5_reads": 0, "physiotrait_day200_reads": 0,
    }
    _write_json(output / "sealed_guard.json", sealed_guard)
    metric_schema = {
        "scientific_unit": "participant", "denominator": 16,
        "risk_direction": "lower_is_better", "effect_direction": "positive_is_match_better",
        "primary": {"O0_A": ["N_P", "N_W"], "O0_B": ["H_P", "H_W"]},
        "participant_first_order": ["window", "wrong_donor", "protocol_unit", "task", "participant"],
    }
    _write_json(output / "metric_schema.json", metric_schema)
    # Metadata facts only; signal files remain unopened at J0.
    development_rows = [row for row in inventory if row["participant"] in dev and row["session"] in config["sessions"]]
    facts = {
        "source_expected_channels": {"EEG": 46, "EOG": 4},
        "processed_expected_channels": {"EEG": 46, "EOG": 0, "IMU_typical": 27},
        "historical_coverage": {"records": "91/96", "protocol_units": "150/151"},
        "metadata_records_present": sum(bool(row["header_exists"] and row["data_exists"] and row["marker_exists"]) for row in development_rows),
        "metadata_record_denominator": 96,
    }
    _write_json(output / "data_facts.json", facts)
    summary = {"stage": "j0-preflight", "status": novelty["decision"], "sealed_reads": 0, **facts}
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _winsor(value: np.ndarray, multiple: float) -> np.ndarray:
    median = np.median(value, axis=1, keepdims=True)
    mad = np.median(np.abs(value - median), axis=1, keepdims=True)
    limit = multiple * np.maximum(1.4826 * mad, 1e-6)
    return np.clip(value, median - limit, median + limit)


def _preprocess(eeg: np.ndarray, eog: np.ndarray, rate: float, config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prep = config["preprocessing"]
    eeg = np.asarray(eeg, dtype=np.float64)
    eog = np.asarray(eog, dtype=np.float64)
    eeg = eeg - np.mean(eeg, axis=0, keepdims=True)
    eeg = _winsor(eeg, float(prep["winsor_mad"]))
    eog = _winsor(eog, float(prep["winsor_mad"]))
    sos = signal.butter(int(prep["filter_order"]), tuple(prep["bandpass_hz"]), btype="bandpass", fs=rate, output="sos")
    eeg = signal.sosfiltfilt(sos, eeg, axis=-1)
    eog = signal.sosfiltfilt(sos, eog, axis=-1)
    target = float(prep["target_sampling_rate_hz"])
    from fractions import Fraction
    ratio = Fraction(target / rate).limit_denominator(1000)
    eeg = signal.resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1)
    eog = signal.resample_poly(eog, ratio.numerator, ratio.denominator, axis=-1)
    length = min(eeg.shape[1], eog.shape[1])
    return eeg[:, :length].astype(np.float32), eog[:, :length].astype(np.float32)


def _marker_times(path: Path, sampling_rate: float) -> np.ndarray:
    times: list[float] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        fields = line.split("=", 1)[1].split(",")
        if len(fields) < 3 or fields[0].strip().lower() == "new segment":
            continue
        try:
            position = int(fields[2])
        except ValueError:
            continue
        times.append((position - 1) / sampling_rate)
    return np.asarray(times, dtype=np.float64)


def prepare_stage(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    participants = list(config["development_participants"])
    if not 0 <= task_index < len(participants):
        raise IndexError(task_index)
    participant = participants[task_index]
    data_root = Path(str(config["data_root"]))
    derived = Path(str(config["derived_root"])) / "prepared" / participant
    derived.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for session, task in itertools.product(config["sessions"], config["tasks"]):
        try:
            if not _available(config, participant, session, task):
                raise FileNotFoundError("source_or_processed_asset_missing")
            resolved = guarded_paths(config, participant, session, task, roles=(
                "processed_header", "processed_binary", "processed_marker", "processed_channels",
                "source_header", "source_binary", "source_marker", "source_channels",
            ))
            source = read_source_eeg_eog(data_root, participant, session, task, allowlist=participants)
            processed = read_development_record(data_root, participant, session, task, allowlist=participants)
            common = [name for name in source["eeg_names"] if name in set(processed["eeg_names"])]
            resolved_by_role = {row["role"]: Path(row["resolved_path"]) for row in resolved}
            source_markers = _marker_times(resolved_by_role["source_marker"], float(source["sampling_rate_hz"]))
            processed_markers = _marker_times(resolved_by_role["processed_marker"], float(processed["sampling_rate_hz"]))
            marker_count = min(len(source_markers), len(processed_markers))
            marker_error = np.abs(source_markers[:marker_count] - processed_markers[:marker_count]) if marker_count else np.asarray([])
            eeg, eog = _preprocess(source["eeg"], source["eog"], float(source["sampling_rate_hz"]), config)
            duration = eeg.shape[1] / float(config["preprocessing"]["target_sampling_rate_hz"])
            required = (float(config["windows"]["support_seconds"]) + 2 * float(config["windows"]["guard_seconds"])
                        + float(config["windows"]["query_generator_seconds"]))
            status = "eligible" if duration >= required + 2.0 else "blocked_insufficient_duration"
            target = derived / f"{session}_{task}.npz"
            np.savez_compressed(target, eeg=eeg, eog=eog,
                                eeg_names=np.asarray(source["eeg_names"]), eog_names=np.asarray(source["eog_names"]),
                                sampling_rate=np.float64(config["preprocessing"]["target_sampling_rate_hz"]))
            rows.append({
                "participant": participant, "session": session, "task": task, "status": status,
                "duration_seconds": duration, "source_rate_hz": source["sampling_rate_hz"],
                "processed_rate_hz": processed["sampling_rate_hz"],
                "source_eeg_channels": eeg.shape[0], "source_eog_channels": eog.shape[0],
                "processed_eeg_channels": processed["eeg"].shape[0], "processed_imu_channels": processed["imu"].shape[0],
                "common_eeg_channels": len(common),
                "duration_alignment_seconds": abs(duration - float(processed["duration_seconds"])),
                "source_marker_count": len(source_markers), "processed_marker_count": len(processed_markers),
                "aligned_marker_count": marker_count,
                "marker_time_median_error_seconds": float(np.median(marker_error)) if marker_count else float("nan"),
                "marker_time_p95_error_seconds": float(np.quantile(marker_error, 0.95)) if marker_count else float("nan"),
                "prepared_path": str(target), "sealed_reads": 0,
            })
        except Exception as error:
            rows.append({"participant": participant, "session": session, "task": task,
                         "status": "blocked_prepare_failure", "failure": f"{type(error).__name__}: {error}", "sealed_reads": 0})
    _write_csv(_root(config) / "alignment" / f"{participant}.csv", rows)
    summary = {"stage": "j2-source-processed-alignment", "participant": participant,
               "eligible_units": sum(row["status"] == "eligible" for row in rows), "unit_denominator": 6,
               "sealed_reads": 0}
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _fit_operator(eeg: np.ndarray, eog: np.ndarray, ridge_ratio: float) -> np.ndarray:
    y = np.asarray(eeg, dtype=np.float64)
    e = np.asarray(eog, dtype=np.float64)
    y = y - np.mean(y, axis=1, keepdims=True)
    e = e - np.mean(e, axis=1, keepdims=True)
    gram = e @ e.T / max(1, e.shape[1])
    ridge = ridge_ratio * max(float(np.trace(gram) / gram.shape[0]), 1e-9)
    return (y @ e.T / max(1, e.shape[1])) @ np.linalg.inv(gram + ridge * np.eye(gram.shape[0]))


def _unit_slices(length: int, rate: float, config: Mapping[str, Any]) -> tuple[slice, slice, slice] | None:
    support = int(round(float(config["windows"]["support_seconds"]) * rate))
    guard = int(round(float(config["windows"]["guard_seconds"]) * rate))
    qgen = int(round(float(config["windows"]["query_generator_seconds"]) * rate))
    second = int(round(float(config["windows"]["second_guard_seconds"]) * rate))
    a = slice(0, support); b = slice(support + guard, support + guard + qgen)
    c = slice(support + guard + qgen + second, length)
    if c.start + int(2 * rate) > length:
        return None
    return a, b, c


def fit_raw_stage(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    participant = list(config["development_participants"])[task_index]
    prepared = Path(str(config["derived_root"])) / "prepared" / participant
    operator_dir = Path(str(config["derived_root"])) / "operators" / "raw" / participant
    evaluator_dir = Path(str(config["derived_root"])) / "evaluator" / participant
    operator_dir.mkdir(parents=True, exist_ok=True); evaluator_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rate = float(config["preprocessing"]["target_sampling_rate_hz"])
    ridge = float(config["operator"]["ridge_ratio"])
    shift = int(round(float(config["operator"]["shift_seconds"]) * rate))
    for session, task in itertools.product(config["sessions"], config["tasks"]):
        path = prepared / f"{session}_{task}.npz"
        if not path.is_file():
            rows.append({"participant": participant, "session": session, "task": task, "status": "blocked_missing_prepared"})
            continue
        with np.load(path, allow_pickle=False) as data:
            eeg = np.asarray(data["eeg"], dtype=np.float64); eog = np.asarray(data["eog"], dtype=np.float64)
        slices = _unit_slices(eeg.shape[1], rate, config)
        if slices is None:
            rows.append({"participant": participant, "session": session, "task": task, "status": "blocked_insufficient_duration"})
            continue
        support, qgen, qnatural = slices
        c_raw = _fit_operator(eeg[:, support], eog[:, support], ridge)
        block_length = (support.stop - support.start) // 4
        blocks = np.stack([_fit_operator(eeg[:, slice(i * block_length, (i + 1) * block_length)],
                                         eog[:, slice(i * block_length, (i + 1) * block_length)], ridge) for i in range(4)])
        shifted = np.roll(eog[:, support], shift=shift, axis=1)
        c_shift = _fit_operator(eeg[:, support], shifted, ridge)
        c_query = _fit_operator(eeg[:, qgen], eog[:, qgen], ridge)
        inference_path = operator_dir / f"{session}_{task}.npz"
        evaluator_path = evaluator_dir / f"{session}_{task}.npz"
        np.savez_compressed(inference_path, C_raw=c_raw.astype(np.float32), C_blocks=blocks.astype(np.float32),
                            C_shift_raw=c_shift.astype(np.float32), qnatural_start=np.int64(qnatural.start))
        np.savez_compressed(evaluator_path, C_query=c_query.astype(np.float32))
        rows.append({"participant": participant, "session": session, "task": task, "status": "eligible",
                     "support_seconds": 120.0, "guard_seconds": 30.0, "qgen_seconds": 120.0,
                     "qnatural_seconds": (eeg.shape[1] - qnatural.start) / rate,
                     "inference_contains_C_query": False, "evaluator_contains_C_query": True})
    _write_csv(_root(config) / "operator_fit" / f"{participant}.csv", rows)
    summary = {"stage": "j3-fit-raw-operator", "participant": participant,
               "eligible_units": sum(row["status"] == "eligible" for row in rows), "unit_denominator": 6}
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _raw_operator(config: Mapping[str, Any], participant: str, session: str, task: str) -> dict[str, np.ndarray] | None:
    path = Path(str(config["derived_root"])) / "operators" / "raw" / participant / f"{session}_{task}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def build_contexts_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    participants = list(config["development_participants"])
    derived = Path(str(config["derived_root"]))
    context_root = derived / "operators" / "contexts"
    rows: list[dict[str, Any]] = []
    for recipient, session, task in itertools.product(participants, config["sessions"], config["tasks"]):
        raw = _raw_operator(config, recipient, session, task)
        other_task = next(value for value in config["tasks"] if value != task)
        donor_records: dict[str, tuple[dict[str, np.ndarray], bool]] = {}
        for donor in participants:
            if donor == recipient:
                continue
            exact = _raw_operator(config, donor, session, task)
            fallback = False
            if exact is None:
                exact = _raw_operator(config, donor, session, other_task)
                fallback = exact is not None
            if exact is not None:
                donor_records[donor] = (exact, fallback)
        donors = sorted(donor_records)
        if raw is None or len(donors) != 15:
            rows.append({"participant": recipient, "session": session, "task": task,
                         "status": "blocked_missing_unit_or_pop15", "eligible_donors": len(donors)})
            continue
        c0 = np.mean([donor_records[donor][0]["C_raw"] for donor in donors], axis=0)
        deviations = np.stack([donor_records[donor][0]["C_raw"] - c0 for donor in donors])
        tau2 = float(np.mean(np.square(deviations)))
        within = float(np.mean(np.square(raw["C_blocks"] - raw["C_raw"][None])))
        alpha = float(np.clip(tau2 / max(tau2 + within / 4.0, 1e-15), 0.0, 1.0))
        match = c0 + alpha * (raw["C_raw"] - c0)
        shift = c0 + alpha * (raw["C_shift_raw"] - c0)
        wrong_values: list[np.ndarray] = []
        wrong_alpha: list[float] = []
        for donor in donors:
            donor_raw = donor_records[donor][0]
            donor_within = float(np.mean(np.square(donor_raw["C_blocks"] - donor_raw["C_raw"][None])))
            alpha_d = float(np.clip(tau2 / max(tau2 + donor_within / 4.0, 1e-15), 0.0, 1.0))
            wrong_values.append(c0 + alpha_d * (donor_raw["C_raw"] - c0))
            wrong_alpha.append(alpha_d)
        c0_norm = max(float(np.linalg.norm(c0)), 1e-12)
        match_gain = match * (c0_norm / max(float(np.linalg.norm(match)), 1e-12))
        wrong_gain = np.stack([value * (c0_norm / max(float(np.linalg.norm(value)), 1e-12)) for value in wrong_values])
        channel_perm = np.roll(match, shift=1, axis=0)
        target = context_root / recipient / f"{session}_{task}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, C_pop=c0.astype(np.float32), C_match=match.astype(np.float32),
                            C_shift=shift.astype(np.float32), C_match_gain=match_gain.astype(np.float32),
                            C_wrong=np.stack(wrong_values).astype(np.float32), C_wrong_gain=wrong_gain.astype(np.float32),
                            C_channel_perm=channel_perm.astype(np.float32), donors=np.asarray(donors),
                            alpha=np.float64(alpha), wrong_alpha=np.asarray(wrong_alpha))
        rows.append({"participant": recipient, "session": session, "task": task, "status": "eligible",
                     "eligible_donors": len(donors), "alpha": alpha, "within_variance": within,
                     "outer_between_variance": tau2, "pop_participant_count": len(donors),
                     "donor_task_fallback_count": sum(donor_records[donor][1] for donor in donors),
                     "context_path": str(target), "contains_C_query": False})
    _write_csv(_root(config) / "operator_context_manifest.csv", rows)
    evaluable = {participant for participant in participants
                 if all(any(row["participant"] == participant and row["task"] == task and row["status"] == "eligible" for row in rows)
                        for task in config["tasks"])}
    status = "PASS" if len(evaluable) >= int(config["o0"]["minimum_evaluable_participants"]) else "INSUFFICIENT"
    summary = {"stage": "j3-build-contexts", "data_protocol": status,
               "evaluable_participants": len(evaluable), "policy_denominator": 16,
               "eligible_units": sum(row["status"] == "eligible" for row in rows),
               "unit_denominator": 96, "pop_definition": "recipient-excluded participant-equal POP15"}
    _write_json(_root(config) / "data_protocol_decision.json", summary)
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _window_indices(eog: np.ndarray, rate: float, seconds: float) -> tuple[list[slice], np.ndarray]:
    length = int(round(rate * seconds))
    windows = [slice(start, start + length) for start in range(0, eog.shape[1] - length + 1, length)]
    energy = np.asarray([float(np.sqrt(np.mean(np.square(eog[:, window])))) for window in windows])
    return windows, energy


def _context(config: Mapping[str, Any], participant: str, session: str, task: str) -> dict[str, np.ndarray] | None:
    path = Path(str(config["derived_root"])) / "operators" / "contexts" / participant / f"{session}_{task}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _risk(y: np.ndarray, e: np.ndarray, operator: np.ndarray) -> float:
    yy = y - np.mean(y, axis=1, keepdims=True)
    ee = e - np.mean(e, axis=1, keepdims=True)
    return float(np.linalg.norm(yy - operator @ ee) / max(np.linalg.norm(yy), 1e-12))


def natural_stage(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    participant = list(config["development_participants"])[task_index]
    prepared_root = Path(str(config["derived_root"])) / "prepared" / participant
    rate = float(config["preprocessing"]["target_sampling_rate_hz"])
    rows: list[dict[str, Any]] = []
    for session, task in itertools.product(config["sessions"], config["tasks"]):
        path = prepared_root / f"{session}_{task}.npz"; context = _context(config, participant, session, task)
        if not path.is_file() or context is None:
            continue
        with np.load(path, allow_pickle=False) as data:
            eeg = np.asarray(data["eeg"], dtype=np.float64); eog = np.asarray(data["eog"], dtype=np.float64)
        slices = _unit_slices(eeg.shape[1], rate, config)
        if slices is None:
            continue
        qnatural = slices[2]; y = eeg[:, qnatural]; e = eog[:, qnatural]
        windows, energy = _window_indices(e, rate, float(config["windows"]["natural_window_seconds"]))
        if len(windows) < 4:
            continue
        high = energy >= np.quantile(energy, float(config["o0"]["high_eog_quantile"]))
        low = energy <= np.quantile(energy, 0.10)
        methods: list[tuple[str, np.ndarray, str]] = [
            ("MATCH", context["C_match"], ""), ("POP", context["C_pop"], ""),
            ("TIME_SHIFT", context["C_shift"], ""), ("CHANNEL_PERM", context["C_channel_perm"], ""),
            ("GAIN_MATCH", context["C_match_gain"], ""), ("GAIN_POP", context["C_pop"], ""),
            ("NULL", np.zeros_like(context["C_pop"]), ""),
        ]
        for donor, operator in zip(context["donors"].astype(str), context["C_wrong"]):
            methods.append(("WRONG", operator, donor))
        for donor, operator in zip(context["donors"].astype(str), context["C_wrong_gain"]):
            methods.append(("GAIN_WRONG", operator, donor))
        for window_index, (window, is_high, is_low) in enumerate(zip(windows, high, low)):
            if not (is_high or is_low):
                continue
            for method, operator, donor in methods:
                correction = operator @ (e[:, window] - np.mean(e[:, window], axis=1, keepdims=True))
                rows.append({"participant": participant, "session": session, "task": task,
                             "window": window_index, "eog_panel": "high" if is_high else "low",
                             "method": method, "wrong_donor": donor,
                             "normalized_prediction_risk": _risk(y[:, window], e[:, window], operator),
                             "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
                             "eog_rms": float(energy[window_index])})
    _write_csv(_root(config) / "o0_natural" / f"{participant}.csv", rows)
    summary = {"stage": "j4-o0-natural", "participant": participant,
               "rows": len(rows), "eligible_units": len({(row["session"], row["task"]) for row in rows})}
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _rrmse(target: np.ndarray, estimate: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        left = target.ravel(); right = estimate.ravel()
    else:
        selected = np.broadcast_to(mask[None, :], target.shape)
        left = target[selected]; right = estimate[selected]
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))


def _corr(target: np.ndarray, estimate: np.ndarray) -> float:
    a = target.ravel() - float(np.mean(target)); b = estimate.ravel() - float(np.mean(estimate))
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _spectral_error(target: np.ndarray, estimate: np.ndarray, rate: float) -> float:
    freq, p0 = signal.welch(target, fs=rate, nperseg=min(target.shape[-1], 200), axis=-1)
    _, p1 = signal.welch(estimate, fs=rate, nperseg=min(estimate.shape[-1], 200), axis=-1)
    keep = (freq >= 1.0) & (freq <= 15.0)
    a = np.log(np.maximum(p0[:, keep], 1e-12)); b = np.log(np.maximum(p1[:, keep], 1e-12))
    return float(np.mean(np.abs(a - b)))


def _digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _candidate_windows(config: Mapping[str, Any], participant: str, session: str, task: str) -> tuple[np.ndarray, np.ndarray, list[slice], np.ndarray] | None:
    path = Path(str(config["derived_root"])) / "prepared" / participant / f"{session}_{task}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        eeg = np.asarray(data["eeg"], dtype=np.float64); eog = np.asarray(data["eog"], dtype=np.float64)
    rate = float(config["preprocessing"]["target_sampling_rate_hz"])
    slices = _unit_slices(eeg.shape[1], rate, config)
    if slices is None:
        return None
    q = slices[2]; y = eeg[:, q]; e = eog[:, q]
    windows, energy = _window_indices(e, rate, float(config["windows"]["natural_window_seconds"]))
    return y, e, windows, energy


def paired_stage(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    participants = list(config["development_participants"]); recipient = participants[task_index]
    rate = float(config["preprocessing"]["target_sampling_rate_hz"])
    rows: list[dict[str, Any]] = []
    for session, task in itertools.product(config["sessions"], config["tasks"]):
        context = _context(config, recipient, session, task)
        evaluator_path = Path(str(config["derived_root"])) / "evaluator" / recipient / f"{session}_{task}.npz"
        if context is None or not evaluator_path.is_file():
            continue
        with np.load(evaluator_path, allow_pickle=False) as evaluator:
            c_query = np.asarray(evaluator["C_query"], dtype=np.float64)
        others = [value for value in participants if value != recipient and _candidate_windows(config, value, session, task) is not None]
        if len(others) < 2:
            continue
        methods: list[tuple[str, np.ndarray, str]] = [
            ("QUERY_ORACLE", c_query, ""), ("MATCH", context["C_match"], ""),
            ("POP", context["C_pop"], ""), ("TIME_SHIFT", context["C_shift"], ""),
            ("CHANNEL_PERM", context["C_channel_perm"], ""),
            ("GAIN_MATCH", context["C_match_gain"], ""), ("GAIN_POP", context["C_pop"], ""),
            ("NULL", np.zeros_like(c_query), ""),
        ]
        methods += [("WRONG", operator, donor) for donor, operator in zip(context["donors"].astype(str), context["C_wrong"])]
        methods += [("GAIN_WRONG", operator, donor) for donor, operator in zip(context["donors"].astype(str), context["C_wrong_gain"])]
        count = int(config["o0"]["paired_windows_per_unit"])
        for pair_index in range(count):
            x_donor = others[pair_index % len(others)]
            e_donor = others[(pair_index + 1) % len(others)]
            if x_donor == e_donor:
                raise AssertionError("x/e donors must be distinct")
            x_pack = _candidate_windows(config, x_donor, session, task)
            e_pack = _candidate_windows(config, e_donor, session, task)
            assert x_pack is not None and e_pack is not None
            x_y, _, x_windows, x_energy = x_pack; _, e_all, e_windows, e_energy = e_pack
            x_indices = np.flatnonzero(x_energy <= np.quantile(x_energy, float(config["o0"]["low_eog_quantile"])))
            e_indices = np.flatnonzero(e_energy >= np.quantile(e_energy, float(config["o0"]["high_eog_quantile"])))
            if not len(x_indices) or not len(e_indices):
                continue
            x = x_y[:, x_windows[int(x_indices[pair_index % len(x_indices)])]].copy()
            e = e_all[:, e_windows[int(e_indices[pair_index % len(e_indices)])]].copy()
            e = e - np.mean(e, axis=1, keepdims=True)
            temporal_energy = np.sqrt(np.mean(np.square(e), axis=0))
            mask = temporal_energy >= np.quantile(temporal_energy, 0.60)
            artifact = c_query @ (e * mask[None, :])
            y = x + artifact
            common_hash = f"{_digest(x)}:{_digest(e)}:{_digest(y)}:{_digest(mask)}"
            for method, operator, donor in methods:
                estimate = y - operator @ (e * mask[None, :])
                outside = float(np.max(np.abs(estimate[:, ~mask] - y[:, ~mask]))) if np.any(~mask) else 0.0
                rows.append({"participant": recipient, "session": session, "task": task, "pair": pair_index,
                             "x_donor": x_donor, "e_donor": e_donor, "operator_recipient": recipient,
                             "three_way_disjoint": int(len({recipient, x_donor, e_donor}) == 3),
                             "method": method, "wrong_donor": donor, "common_input_hash": common_hash,
                             "mask_rrmse": _rrmse(x, estimate, mask), "full_rrmse": _rrmse(x, estimate),
                             "correlation": _corr(x, estimate), "spectral_error_1_15": _spectral_error(x, estimate, rate),
                             "topography_error": float(np.linalg.norm(np.std(x, axis=1) - np.std(estimate, axis=1)) /
                                                        max(np.linalg.norm(np.std(x, axis=1)), 1e-12)),
                             "outside_mask_max_change": outside})
    _write_csv(_root(config) / "o0_paired" / f"{recipient}.csv", rows)
    oracle = [row["mask_rrmse"] for row in rows if row["method"] == "QUERY_ORACLE"]
    summary = {"stage": "j5-o0-paired", "participant": recipient, "rows": len(rows),
               "oracle_max_rrmse": max(oracle) if oracle else None,
               "all_three_way_disjoint": bool(rows and all(row["three_way_disjoint"] for row in rows)),
               "common_input_hashes_per_pair": "one"}
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def exact_signflip_p(values: Sequence[float], *, alternative: str = "greater") -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(array))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(array))), dtype=np.float64)
    permutations = np.mean(signs * array[None, :], axis=1)
    if alternative == "greater":
        return float(np.mean(permutations >= observed - 1e-15))
    if alternative == "two-sided":
        return float(np.mean(np.abs(permutations) >= abs(observed) - 1e-15))
    raise ValueError(alternative)


def _bootstrap(values: Sequence[float], repetitions: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _collapse_method_rows(
    rows: Sequence[Mapping[str, str]], *, value_key: str, panel: str,
) -> tuple[dict[tuple[str, str, str, str], float], list[dict[str, Any]]]:
    # windows/pairs -> donor -> protocol unit. WRONG is then donor-equal.
    by_donor: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in rows:
        if panel == "natural" and row.get("eog_panel") != "high":
            continue
        key = (row["participant"], row["session"], row["task"], row["method"], row.get("wrong_donor", ""))
        by_donor.setdefault(key, []).append(float(row[value_key]))
    donor_rows: list[dict[str, Any]] = []
    by_method: dict[tuple[str, str, str, str], list[float]] = {}
    for (participant, session, task, method, donor), values in by_donor.items():
        mean = float(np.mean(values))
        donor_rows.append({"panel": panel, "participant": participant, "session": session, "task": task,
                           "method": method, "wrong_donor": donor, "risk": mean, "replicates": len(values)})
        by_method.setdefault((participant, session, task, method), []).append(mean)
    return ({key: float(np.mean(values)) for key, values in by_method.items()}, donor_rows)


def _participant_method(unit: Mapping[tuple[str, str, str, str], float]) -> dict[tuple[str, str], float]:
    # protocol unit -> task equal -> participant. Sessions are first averaged inside task.
    by_task: dict[tuple[str, str, str], list[float]] = {}
    for (participant, session, task, method), value in unit.items():
        by_task.setdefault((participant, task, method), []).append(value)
    by_participant: dict[tuple[str, str], list[float]] = {}
    for (participant, _task, method), values in by_task.items():
        by_participant.setdefault((participant, method), []).append(float(np.mean(values)))
    return {key: float(np.mean(values)) for key, values in by_participant.items()}


def _participant_task_method(unit: Mapping[tuple[str, str, str, str], float]) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], list[float]] = {}
    for (participant, _session, task, method), value in unit.items():
        values.setdefault((participant, task, method), []).append(value)
    return {key: float(np.mean(item)) for key, item in values.items()}


def _effect_summary(values: Sequence[float], config: Mapping[str, Any], seed_offset: int = 0) -> dict[str, Any]:
    low, high = _bootstrap(values, int(config["o0"]["bootstrap_repetitions"]),
                           int(config["o0"]["bootstrap_seed"]) + seed_offset)
    array = np.asarray(values, dtype=np.float64)
    loo = [float(np.mean(np.delete(array, index))) for index in range(len(array))]
    return {"mean": float(np.mean(array)), "median": float(np.median(array)),
            "positive_count": int(np.sum(array > 0)), "denominator": len(array),
            "exact_one_sided_p": exact_signflip_p(array), "bootstrap_ci_low": low,
            "bootstrap_ci_high": high, "leave_one_participant_out_min_mean": min(loo)}


def aggregate_o0_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    output = _root(config); participants = list(config["development_participants"])
    natural_rows = list(itertools.chain.from_iterable(
        _read_csv(output / "o0_natural" / f"{p}.csv") for p in participants
        if (output / "o0_natural" / f"{p}.csv").is_file()))
    paired_rows = list(itertools.chain.from_iterable(
        _read_csv(output / "o0_paired" / f"{p}.csv") for p in participants
        if (output / "o0_paired" / f"{p}.csv").is_file()))
    natural_unit, natural_donors = _collapse_method_rows(natural_rows, value_key="normalized_prediction_risk", panel="natural")
    paired_unit, paired_donors = _collapse_method_rows(paired_rows, value_key="mask_rrmse", panel="paired")
    natural_pm = _participant_method(natural_unit); paired_pm = _participant_method(paired_unit)
    natural_ptm = _participant_task_method(natural_unit); paired_ptm = _participant_task_method(paired_unit)
    effects: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    evaluable: list[str] = []
    for participant in participants:
        tasks_ok = all((participant, task, "MATCH") in natural_ptm and (participant, task, "MATCH") in paired_ptm
                       for task in config["tasks"])
        available = tasks_ok and all((participant, method) in natural_pm for method in ("MATCH", "POP", "WRONG")) \
                    and all((participant, method) in paired_pm for method in ("MATCH", "POP", "WRONG", "QUERY_ORACLE"))
        if available:
            evaluable.append(participant)
        n_p = natural_pm.get((participant, "POP"), 0.0) - natural_pm.get((participant, "MATCH"), 0.0) if available else 0.0
        n_w = natural_pm.get((participant, "WRONG"), 0.0) - natural_pm.get((participant, "MATCH"), 0.0) if available else 0.0
        h_p = paired_pm.get((participant, "POP"), 0.0) - paired_pm.get((participant, "MATCH"), 0.0) if available else 0.0
        h_w = paired_pm.get((participant, "WRONG"), 0.0) - paired_pm.get((participant, "MATCH"), 0.0) if available else 0.0
        row: dict[str, Any] = {"participant": participant, "evaluable": int(available),
                               "N_P": n_p, "N_W": n_w, "H_P": h_p, "H_W": h_w}
        for panel, pm in (("natural", natural_pm), ("paired", paired_pm)):
            prefix = "N" if panel == "natural" else "H"
            if available:
                row[f"{prefix}_time_shift_effect"] = pm[(participant, "POP")] - pm.get((participant, "TIME_SHIFT"), pm[(participant, "POP")])
                row[f"{prefix}_gain_P"] = pm.get((participant, "GAIN_POP"), pm[(participant, "POP")]) - pm.get((participant, "GAIN_MATCH"), pm[(participant, "MATCH")])
                row[f"{prefix}_gain_W"] = pm.get((participant, "GAIN_WRONG"), pm[(participant, "WRONG")]) - pm.get((participant, "GAIN_MATCH"), pm[(participant, "MATCH")])
                nulls.append({"participant": participant, "panel": panel,
                              "pop_minus_time_shift": row[f"{prefix}_time_shift_effect"],
                              "pop_minus_channel_perm": pm[(participant, "POP")] - pm.get((participant, "CHANNEL_PERM"), pm[(participant, "POP")]),
                              "gain_match_effect": row[f"{prefix}_gain_P"]})
            else:
                row[f"{prefix}_time_shift_effect"] = 0.0; row[f"{prefix}_gain_P"] = 0.0; row[f"{prefix}_gain_W"] = 0.0
        effects.append(row)
    # Fold-local null floors are derived from the other 15 participants only.
    for row in effects:
        participant = row["participant"]
        for panel, prefix, pm in (("natural", "N", natural_pm), ("paired", "H", paired_pm)):
            outer = [item for item in nulls if item["participant"] != participant and item["panel"] == panel]
            null_values = [max(0.0, float(item["pop_minus_time_shift"])) for item in outer]
            q95 = float(np.quantile(null_values, 0.95)) if null_values else float("inf")
            pop_risk = pm.get((participant, "POP"), float("nan"))
            relative = float(config["o0"]["minimum_relative_risk"]) * pop_risk if np.isfinite(pop_risk) else 0.0
            row[f"{prefix}_effect_floor"] = max(float(config["o0"]["minimum_absolute_effect"]), relative, q95)
    _write_csv(output / "participant_effects.csv", effects)
    _write_csv(output / "null_controls.csv", nulls)
    _write_csv(output / "wrong_donor_metrics.csv", natural_donors + paired_donors)
    summaries: dict[str, Any] = {}
    criteria: dict[str, bool] = {}
    for index, key in enumerate(("N_P", "N_W", "H_P", "H_W")):
        values = [float(row[key]) for row in effects]
        summary = _effect_summary(values, config, index)
        summaries[key] = summary
        prefix = key[0]
        mean_floor = float(np.mean([float(row[f"{prefix}_effect_floor"]) for row in effects]))
        summary["mean_effect_floor"] = mean_floor
        criteria[f"{key}_mean_above_floor"] = summary["mean"] > mean_floor
        criteria[f"{key}_median_positive"] = summary["median"] > 0
        criteria[f"{key}_positive_count"] = summary["positive_count"] >= int(config["o0"]["minimum_positive"])
        criteria[f"{key}_exact_p"] = summary["exact_one_sided_p"] < float(config["o0"]["exact_one_sided_alpha"])
        criteria[f"{key}_bootstrap_lower"] = summary["bootstrap_ci_low"] > 0
        criteria[f"{key}_loo"] = summary["leave_one_participant_out_min_mean"] > 0
        panel_ptm = natural_ptm if prefix == "N" else paired_ptm
        method_right = "POP" if key.endswith("P") else "WRONG"
        for task in config["tasks"]:
            task_values = [panel_ptm.get((p, task, method_right), 0.0) - panel_ptm.get((p, task, "MATCH"), 0.0)
                           for p in participants]
            task_mean = float(np.mean(task_values))
            summary[f"{task}_mean"] = task_mean
            criteria[f"{key}_{task}_nonnegative"] = task_mean >= 0
    criteria["minimum_evaluable_participants"] = len(evaluable) >= int(config["o0"]["minimum_evaluable_participants"])
    n_shift_advantage = [float(row["N_P"]) - float(row["N_time_shift_effect"]) for row in effects]
    h_shift_advantage = [float(row["H_P"]) - float(row["H_time_shift_effect"]) for row in effects]
    criteria["natural_actual_greater_than_time_shift"] = exact_signflip_p(n_shift_advantage) < 0.025 and np.mean(n_shift_advantage) > 0
    criteria["paired_actual_greater_than_time_shift"] = exact_signflip_p(h_shift_advantage) < 0.025 and np.mean(h_shift_advantage) > 0
    criteria["gain_normalized_direction"] = all(np.mean([float(row[key]) for row in effects]) > 0 for key in ("N_gain_P", "N_gain_W", "H_gain_P", "H_gain_W"))
    oracle_values = [float(row["mask_rrmse"]) for row in paired_rows if row["method"] == "QUERY_ORACLE"]
    criteria["oracle_recovery"] = bool(oracle_values) and max(oracle_values) < float(config["o0"]["oracle_tolerance"])
    outside_values = [float(row["outside_mask_max_change"]) for row in paired_rows]
    criteria["paired_mask_outside_identity"] = bool(outside_values) and max(outside_values) == 0.0
    # Correction must collapse on the lowest-EOG decile relative to high-EOG windows.
    low = [float(row["correction_rms"]) for row in natural_rows if row["method"] == "MATCH" and row["eog_panel"] == "low"]
    high = [float(row["correction_rms"]) for row in natural_rows if row["method"] == "MATCH" and row["eog_panel"] == "high"]
    criteria["zero_eog_no_stable_correction"] = bool(low and high) and np.mean(low) <= 0.25 * np.mean(high)
    criteria["three_way_disjoint"] = bool(paired_rows) and all(int(row["three_way_disjoint"]) == 1 for row in paired_rows)
    hash_groups: dict[tuple[str, str, str, str], set[str]] = {}
    for row in paired_rows:
        key = (row["participant"], row["session"], row["task"], row["pair"])
        hash_groups.setdefault(key, set()).add(row["common_input_hash"])
    criteria["common_inputs_identical_across_contexts"] = bool(hash_groups) and all(len(value) == 1 for value in hash_groups.values())
    criteria = {key: bool(value) for key, value in criteria.items()}
    passed = all(criteria.values())
    decision = {
        "data_protocol": "PASS" if criteria["minimum_evaluable_participants"] else "INSUFFICIENT",
        "O0": "PASS" if passed else "FAIL",
        "O1": "NOT_RUN", "route": "O0_PASS_O1_AUTHORIZED" if passed else "SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO",
        "scientific_unit": "participant", "policy_denominator": 16, "evaluable_participants": len(evaluable),
        "effects": summaries, "criteria": criteria,
        "sealed_reads": 0, "o1_submitted": False,
    }
    _write_json(output / "o0_summary.json", decision)
    _write_json(output / "route_decision.json", decision)
    _write_json(run_dir / "result_summary.json", decision)
    return decision


def _job_rows() -> list[dict[str, str]]:
    path = CODE_ROOT / "reports" / "slurm" / "counterfactual_operator_headroom_v19_job_ids.txt"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows.append({"job_id": parts[0], "stage": parts[1] if len(parts) > 1 else "", "state": parts[2] if len(parts) > 2 else ""})
    return rows


def final_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    output = _root(config)
    decision = json.loads((output / "route_decision.json").read_text(encoding="utf-8"))
    ledger_rows = list(itertools.chain.from_iterable(
        _read_csv(path) for path in sorted((output / "resolved_path_ledger").glob("*.csv"))))
    sealed_reads = sum(row.get("sealed") == "1" for row in ledger_rows)
    if sealed_reads:
        raise AssertionError("sealed path ledger is not empty")
    prepared_root = Path(str(config["derived_root"])) / "prepared"
    eeg_names: set[tuple[str, ...]] = set(); eog_names: set[tuple[str, ...]] = set()
    eeg_rms: list[float] = []; eog_rms: list[float] = []
    for path in sorted(prepared_root.glob("sub-*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            eeg = np.asarray(data["eeg"], dtype=np.float64); eog = np.asarray(data["eog"], dtype=np.float64)
            eeg_names.add(tuple(data["eeg_names"].astype(str))); eog_names.add(tuple(data["eog_names"].astype(str)))
            eeg_rms.append(float(np.sqrt(np.mean(np.square(eeg))))); eog_rms.append(float(np.sqrt(np.mean(np.square(eog)))))
    gauge_audit = {
        "prepared_records": len(eeg_rms), "eeg_channel_orders": len(eeg_names), "eog_channel_orders": len(eog_names),
        "eeg_channels": sorted({len(value) for value in eeg_names}), "eog_channels": sorted({len(value) for value in eog_names}),
        "eeg_rms_uv_median": float(np.median(eeg_rms)), "eog_rms_uv_median": float(np.median(eog_rms)),
        "physical_unit": "microvolt", "source_multiplier": 1e6,
        "gauge_replay_pass": len(eeg_names) == 1 and len(eog_names) == 1 and all(np.isfinite(eeg_rms + eog_rms)),
    }
    if not gauge_audit["gauge_replay_pass"]:
        raise AssertionError("EOG gauge/physical-scale replay failed")
    inference_paths = sorted((Path(str(config["derived_root"])) / "operators" / "contexts").glob("sub-*/*.npz"))
    for path in inference_paths:
        with np.load(path, allow_pickle=False) as data:
            if "C_query" in data.files:
                raise AssertionError(f"C_query leaked into inference package {path}")
            participant = path.parent.name
            if participant in set(data["donors"].astype(str)):
                raise AssertionError(f"recipient leaked into POP/WRONG donors at {path}")
    gauge_audit["inference_packages_checked"] = len(inference_paths)
    gauge_audit["C_query_in_inference"] = False
    _write_json(output / "gauge_scale_replay.json", gauge_audit)
    import subprocess
    forbidden = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "taas_submission"], cwd=CODE_ROOT, check=False)
    if forbidden.returncode != 0:
        raise AssertionError("forbidden taas_submission diff detected")
    decision["sealed_reads"] = 0
    _write_json(output / "route_decision.json", decision)
    effect_rows = _read_csv(output / "participant_effects.csv")
    lines = [
        "# Counterfactual Operator-Swap Headroom v19",
        "",
        "This is development evidence for participant-session calibration and paired operator-swap semi-simulation only. "
        "It is not a natural EEG counterfactual, stable brain physiology result, or validated subject-aware denoiser.",
        "",
        "## Routing outcome",
        "",
        f"- O0: `{decision['O0']}`",
        f"- Route: `{decision['route']}`",
        f"- O1: `{decision['O1']}`",
        f"- Evaluable: {decision['evaluable_participants']}/16; policy denominator: 16.",
        "- All source waveform/marker reads were allowlist-checked before open; sealed reads: 0.",
        "",
        "## Participant-first effects",
        "",
        "| Effect | Mean | Median | Positive | Exact one-sided p | Descriptive 95% bootstrap | Frozen mean floor |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("N_P", "N_W", "H_P", "H_W"):
        value = decision["effects"][key]
        lines.append(f"| {key} | {value['mean']:.6f} | {value['median']:.6f} | {value['positive_count']}/16 | "
                     f"{value['exact_one_sided_p']:.6f} | [{value['bootstrap_ci_low']:.6f}, {value['bootstrap_ci_high']:.6f}] | "
                     f"{value['mean_effect_floor']:.6f} |")
    failed = [key for key, value in decision["criteria"].items() if not value]
    lines += ["", "## Gate audit", "", f"Failed criteria ({len(failed)}): " + (", ".join(failed) if failed else "none"), "",
              "O0-A uses evaluator-only later EOG solely to score natural transfer. O0-B constructs `y=x+C_query e` "
              "with operator recipient, carrier donor, and EOG donor distinct and keeps `x/e/y/mask` identical across arms. "
              "`C_query` is absent from inference packages.", "",
              "The strong population is the participant-equal mean over all 15 nonrecipient development participants. "
              "WRONG donors are scored separately then averaged within recipient.", "",
              "## Boundaries", "",
              "Mobile sealed participants, PhysioMotion sealed participants, SHU Day-4/5, and PhysioTrait Day-200 remained unopened. "
              "No DET, diffusion, identity model, or other neural network was trained.", ""]
    report = CODE_ROOT / "reports" / "counterfactual_operator_headroom_v19.md"
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines), encoding="utf-8")
    summary = {"route": decision["route"], "O0": decision["O0"], "O1": decision["O1"],
               "participant_rows": len(effect_rows), "sealed_reads": sealed_reads,
               "jobs_recorded": len(_job_rows()), "neural_networks_trained": False,
               "gauge_scale_replay": gauge_audit, "forbidden_a_track_diff": False}
    _write_json(output / "result_summary.json", summary)
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def run_stage(config: Mapping[str, Any], stage: str, run_dir: Path, task_index: int | None = None) -> Mapping[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if stage == "j0-preflight":
        return preflight_stage(config, run_dir)
    if stage == "j2-prepare":
        if task_index is None: raise ValueError("j2-prepare requires array index")
        return prepare_stage(config, run_dir, task_index)
    if stage == "j3-fit-raw":
        if task_index is None: raise ValueError("j3-fit-raw requires array index")
        return fit_raw_stage(config, run_dir, task_index)
    if stage == "j3-build-contexts":
        return build_contexts_stage(config, run_dir)
    if stage == "j4-o0-natural":
        if task_index is None: raise ValueError("j4-o0-natural requires array index")
        return natural_stage(config, run_dir, task_index)
    if stage == "j5-o0-paired":
        if task_index is None: raise ValueError("j5-o0-paired requires array index")
        return paired_stage(config, run_dir, task_index)
    if stage == "j6-o0-route":
        return aggregate_o0_stage(config, run_dir)
    if stage == "j9-final":
        return final_stage(config, run_dir)
    raise ValueError(f"unsupported stage {stage}")
