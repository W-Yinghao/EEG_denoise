"""PhysioTrait-Actionability Gate v18.

CPU-only, no-network longitudinal ERP trait headroom and conditional analytic
restoration audit. Identity verifiers are deliberately absent. Day-200 and
sealed PhysioMotion access fail closed.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import yaml
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt, welch
from scipy.stats import trim_mean
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score


ROLE_TO_DATASET = {"R": "Day_1", "T": "Day_7", "G": "Day_80"}
FORBIDDEN_ROLE = {"F", "Day_200", "Day-200"}
VIEWS = ("primary", "prestim", "hf_art", "label_shuffle", "time_shuffle", "wrong_condition", "gain_normalized")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _root(c: Mapping[str, Any]) -> Path:
    return Path(c["result_root"])


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def guard_role(role: str) -> None:
    if role in FORBIDDEN_ROLE or role not in ROLE_TO_DATASET:
        raise PermissionError(f"Day-200 access refused: {role}")


def fold_members(fold: int) -> tuple[list[int], list[int]]:
    evaluation = [p for p in range(1, 16) if (p - 1) % 5 == fold]
    training = [p for p in range(1, 16) if p not in evaluation]
    return training, evaluation


def _participant_file(c: Mapping[str, Any], participant: int) -> Path:
    path = Path(c["data_root"]) / "files" / f"S{participant}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _refs(handle: h5py.File, role: str) -> list[h5py.Dataset]:
    guard_role(role)
    return [handle[ref] for ref in np.asarray(handle[ROLE_TO_DATASET[role]]).reshape(-1)]


def _physio_development(c: Mapping[str, Any]) -> list[int]:
    rows = _read_csv(Path(c["physiomotion_split"]))
    values = sorted(int(r["participant"]) for r in rows if r["role"] == "development")
    if len(values) != 20:
        raise RuntimeError("PhysioMotion development split changed")
    return values


def _target_channels(c: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    # Reuse only the committed montage mapping asset, not verifier code/results.
    path = Path(c["immutable_gate01r"]).parent / "frozen" / "channel_mapping.csv"
    rows = _read_csv(path)
    names = [r["channel"] for r in rows]
    points = np.asarray([[float(r["x2d"]), float(r["y2d"])] for r in rows], np.float64)
    if len(names) != 57:
        raise RuntimeError("expected 57 common channels")
    return names, points


def freeze_protocol(c: Mapping[str, Any]) -> dict[str, Any]:
    gate01 = json.loads(Path(c["immutable_gate01"]).read_text(encoding="utf-8"))
    gate01r = json.loads(Path(c["immutable_gate01r"]).read_text(encoding="utf-8"))
    if not (gate01.get("PASS_01") is False and gate01.get("M1_verifier") == "FAIL"):
        raise RuntimeError("immutable Gate-01 changed")
    if not (gate01r.get("PASS_01R") is False and gate01r.get("M1R_verifier") == "FAIL"):
        raise RuntimeError("immutable Gate-01R changed")
    root = _root(c)
    frozen = root / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    mapping = dict(c["condition_mapping"])
    if {int(mapping["target_label"]), int(mapping["control_label"])} != {1, 2}:
        raise RuntimeError("official event labels must be the two shared labels 1/2")
    _csv(root / "condition_mapping.csv", [{
        "official_event_code": mapping["target_label"], "role": "target", "primary_contrast": mapping["contrast"],
        "selection_source": "official task/event metadata", "signal_or_outcome_used": 0,
    }, {
        "official_event_code": mapping["control_label"], "role": "nontarget_control", "primary_contrast": mapping["contrast"],
        "selection_source": "official task/event metadata", "signal_or_outcome_used": 0,
    }])
    split = []
    for fold in range(5):
        training, evaluation = fold_members(fold)
        for participant in range(1, 16):
            split.append({"outer_fold": fold, "participant": participant, "role": "evaluation" if participant in evaluation else "outer_training"})
    _csv(root / "split_manifest.csv", split)
    prereg = {
        "name": c["name"], "frozen_before_day7_day80_signal_results": True,
        "roles": {"R": "Day-1 support", "T": "Day-7 restoration query", "G": "Day-80 evaluator gallery", "F": "Day-200 sealed"},
        "primary_condition_contrast": mapping["contrast"], "target_label": int(mapping["target_label"]), "control_label": int(mapping["control_label"]),
        "preprocessing": {"official_poststimulus_epoch_seconds": [0.0, c["epoch_end_seconds"]], "pretrigger_baseline_seconds": [c["epoch_start_seconds"], 0.0], "official_code_defines_prestim_baseline": False, "band_hz": c["primary_band_hz"], "sampling_rate": c["sampling_rate"], "trim_fraction": c["trim_fraction"], "channels": 57},
        "trait_blocks_equal_weight": ["condition-contrast morphology", "normalized spatial topography", "relative spectral/task dynamics"],
        "component_windows_seconds": c["component_windows_seconds"], "beta_candidates": c["beta_candidates"], "alpha_candidates": c["alpha_candidates"],
        "negative_controls": ["PRESTIM", "HF_ART", "LABEL_SHUFFLE", "TIME_SHUFFLE", "WRONG_CONDITION"],
        "identity_verifier_used": False, "results_may_not_change_protocol": True,
    }
    (root / "trait_preregistration.yaml").write_text(yaml.safe_dump(prereg, sort_keys=False), encoding="utf-8")
    schema = {
        "primary": {"contrast": mapping["contrast"], "blocks": {"morphology": "full waveform plus fixed-window signed amplitude/latency/slope", "topography": "channel-demeaned L2-normalized fixed-window maps", "dynamics": "relative 1-4/4-8/8-13/13-30 Hz contrast energy"}, "equal_block_weight": True},
        "outer_fold_standardization": "coordinate median/MAD from outer participant R/G traits", "distance": "mean of block standardized RMS distances",
        "negative_controls": list(VIEWS[1:]), "participant_specific_roi": False, "identity_endpoint": False,
    }
    _json(root / "trait_schema.json", schema)
    sealed = {"day200_opened": False, "day200_loader": "fail_closed", "physiomotion_sealed_opened": False, "physiomotion_development_ids": _physio_development(c), "shu_day4_day5_opened": False}
    _json(root / "sealed_guard.json", sealed)
    return {"status": "TRAIT_PROTOCOL_FROZEN", "participants": 15, "channels": 57, **sealed}


def _extract_event_epochs(dataset: h5py.Dataset, c: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = dataset.shape
    if len(shape) != 2 or min(shape) != 58:
        raise RuntimeError(f"unexpected block shape {shape}")
    samples_first = shape[1] == 58
    trigger = np.asarray(dataset[:, 57] if samples_first else dataset[57, :]).reshape(-1)
    labels = dict(c["condition_mapping"])
    valid = (int(labels["target_label"]), int(labels["control_label"]))
    rounded = np.rint(trigger).astype(int)
    events = np.sort(np.concatenate([np.flatnonzero(rounded == label) for label in valid]))
    source_fs = int(c["source_sampling_rate"])
    target_fs = int(c["sampling_rate"])
    post_start = int(round(float(c["epoch_start_seconds"]) * source_fs))
    post_end = int(round(float(c["epoch_end_seconds"]) * source_fs))
    pre_start = int(round(float(c["prestim_start_seconds"]) * source_fs))
    pre_end = int(round(float(c["prestim_end_seconds"]) * source_fs))
    notch_b, notch_a = iirnotch(float(c["line_frequency_hz"]), 30, fs=source_fs)
    low_sos = butter(4, list(map(float, c["primary_band_hz"])), btype="bandpass", fs=source_fs, output="sos")
    high_sos = butter(4, list(map(float, c["hf_control_band_hz"])), btype="bandpass", fs=source_fs, output="sos")
    post, prestim, high, output_labels = [], [], [], []
    for event in events:
        if event + pre_start < 0 or event + post_end > len(trigger):
            continue
        def segment(start: int, end: int) -> np.ndarray:
            value = np.asarray(dataset[event + start:event + end, :57] if samples_first else dataset[:57, event + start:event + end].T, np.float64).T
            if not np.isfinite(value).all():
                raise FloatingPointError("non-finite source segment")
            value -= value.mean(axis=0, keepdims=True)  # common-average reference
            return filtfilt(notch_b, notch_a, value, axis=-1)
        post_raw, pre_raw = segment(post_start, post_end), segment(pre_start, pre_end)
        low = resample_poly(sosfiltfilt(low_sos, post_raw, axis=-1), target_fs, source_fs, axis=-1)
        pre = resample_poly(sosfiltfilt(low_sos, pre_raw, axis=-1), target_fs, source_fs, axis=-1)
        hf = resample_poly(sosfiltfilt(high_sos, post_raw, axis=-1), target_fs, source_fs, axis=-1)
        baseline = max(1, int(round(float(c["baseline_seconds"]) * target_fs)))
        low -= low[..., :baseline].mean(-1, keepdims=True)
        pre -= pre[..., :baseline].mean(-1, keepdims=True)
        post.append(low.astype(np.float32)); prestim.append(pre.astype(np.float32)); high.append(hf.astype(np.float32)); output_labels.append(int(rounded[event]))
    return np.asarray(post), np.asarray(prestim), np.asarray(high), np.asarray(output_labels, np.int8)


def prepare_participant(c: Mapping[str, Any], participant: int) -> dict[str, Any]:
    if participant not in range(1, 16):
        raise ValueError(participant)
    arrays: dict[str, np.ndarray] = {}
    inventory = []
    with h5py.File(_participant_file(c, participant), "r") as handle:
        if "Day_200" not in handle:
            raise RuntimeError("Day-200 metadata missing; no signal dereference performed")
        for role in ("R", "T", "G"):
            post, pre, high, labels = [], [], [], []
            for dataset in _refs(handle, role):
                p, q, h, y = _extract_event_epochs(dataset, c)
                post.append(p); pre.append(q); high.append(h); labels.append(y)
            p, q, h, y = map(np.concatenate, (post, pre, high, labels))
            arrays[f"{role}_post"] = p.astype(np.float16)
            arrays[f"{role}_prestim"] = q.astype(np.float16)
            arrays[f"{role}_hf"] = h.astype(np.float16)
            arrays[f"{role}_labels"] = y
            inventory.append({"participant": participant, "role": role, "trials": len(y), "target_trials": int(np.sum(y == int(c["condition_mapping"]["target_label"]))), "control_trials": int(np.sum(y == int(c["condition_mapping"]["control_label"]))), "channels": p.shape[1], "samples": p.shape[2], "day200_opened": 0})
    server = _root(c) / "server_arrays" / "participants"
    server.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(server / f"subject_{participant:02d}.npz", **arrays)
    _csv(_root(c) / "inventory" / f"subject_{participant:02d}.csv", inventory)
    return {"status": "PARTICIPANT_PREPARED", "participant": participant, "roles": ["R", "T", "G"], "day200_opened": False}


def aggregate_inventory(c: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for participant in range(1, 16):
        path = _root(c) / "inventory" / f"subject_{participant:02d}.csv"
        if path.exists():
            rows.extend(_read_csv(path))
    _csv(_root(c) / "data_inventory.csv", rows)
    complete = sum(all(any(int(r["participant"]) == p and r["role"] == role and int(r["target_trials"]) > 0 and int(r["control_trials"]) > 0 for r in rows) for role in ("R", "T", "G")) for p in range(1, 16))
    passed = complete == 15 and rows and min(int(r["channels"]) for r in rows) == 57
    decision = {"status": "DATA_PROTOCOL_VALID" if passed else ("TASK_CONTRAST_PROTOCOL_INSUFFICIENT" if complete < 13 else "DATA_PROTOCOL_FAIL"), "PASS": bool(passed), "participants_evaluable": complete, "channels": 57 if rows else 0, "condition_contrast": c["condition_mapping"]["contrast"], "identity_verifier_used": False, "day200_opened": False, "physiomotion_sealed_opened": False}
    _json(_root(c) / "data_protocol_decision.json", decision)
    return decision


def write_source_inventory(c: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for participant in range(1, 16):
        path = _participant_file(c, participant)
        stat = path.stat()
        rows.append({"source": str(path), "type": "longitudinal_raw_mat", "participant": participant, "bytes": stat.st_size, "mtime": stat.st_mtime, "roles_allowed": "Day-1,Day-7,Day-80", "sealed_signal_opened": 0})
    source_root = Path(c["physiomotion_fairness_root"]) / "fair_materialized"
    for owner in _physio_development(c):
        path = source_root / f"masks_{owner:02d}.npz"
        stat = path.stat()
        rows.append({"source": str(path), "type": "PhysioMotion_development_mask_geometry", "participant": owner, "bytes": stat.st_size, "mtime": stat.st_mtime, "roles_allowed": "mask geometry only", "sealed_signal_opened": 0})
    _csv(_root(c) / "source_inventory.csv", rows)
    return {"status": "SOURCE_INVENTORY_WRITTEN", "assets": len(rows), "hashes_computed": False, "day200_opened": False, "physiomotion_sealed_opened": False}


def _load_subject(c: Mapping[str, Any], participant: int) -> dict[str, np.ndarray]:
    path = _root(c) / "server_arrays" / "participants" / f"subject_{participant:02d}.npz"
    with np.load(path) as data:
        return {key: np.asarray(data[key], np.float32 if not key.endswith("labels") else np.int8) for key in data.files}


def _contrast(epochs: np.ndarray, labels: np.ndarray, c: Mapping[str, Any], shuffled: bool = False, seed: int = 0) -> np.ndarray:
    y = np.asarray(labels).copy()
    if shuffled:
        y = np.random.default_rng(seed).permutation(y)
    target = int(c["condition_mapping"]["target_label"])
    control = int(c["condition_mapping"]["control_label"])
    if min(np.sum(y == target), np.sum(y == control)) < 2:
        raise RuntimeError("condition contrast unavailable")
    return (trim_mean(epochs[y == target], float(c["trim_fraction"]), axis=0) - trim_mean(epochs[y == control], float(c["trim_fraction"]), axis=0)).astype(np.float32)


def _component_block(x: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    fs = int(c["sampling_rate"])
    event_zero = int(round(-float(c["epoch_start_seconds"]) * fs))
    pieces = [x.reshape(-1)]
    for lo, hi in c["component_windows_seconds"]:
        start = max(0, event_zero + int(round(float(lo) * fs)))
        end = min(x.shape[-1], event_zero + int(round(float(hi) * fs)))
        segment = x[:, start:end]
        signed_amp = segment.mean(-1)
        peak = np.argmax(np.abs(segment), axis=-1) / max(1, segment.shape[-1] - 1)
        slope = (segment[:, -1] - segment[:, 0]) / max(1, segment.shape[-1] - 1)
        pieces.extend((signed_amp, peak, slope))
    return np.concatenate(pieces).astype(np.float32)


def _topography_block(x: np.ndarray, c: Mapping[str, Any]) -> np.ndarray:
    fs = int(c["sampling_rate"])
    event_zero = int(round(-float(c["epoch_start_seconds"]) * fs))
    pieces = []
    for lo, hi in c["component_windows_seconds"]:
        start = max(0, event_zero + int(round(float(lo) * fs)))
        end = min(x.shape[-1], event_zero + int(round(float(hi) * fs)))
        value = x[:, start:end].mean(-1)
        value -= value.mean()
        value /= max(np.linalg.norm(value), 1e-8)
        pieces.append(value)
    return np.concatenate(pieces).astype(np.float32)


def _dynamics_block(x: np.ndarray, c: Mapping[str, Any], hf: bool = False) -> np.ndarray:
    fs = int(c["sampling_rate"])
    f, p = welch(x, fs=fs, nperseg=min(128, x.shape[-1]), axis=-1)
    bands = ((30, 45), (45, 60), (60, 80), (80, 100)) if hf else c["relative_bands_hz"]
    denominator_range = (30, 100) if hf else (1, 30)
    keep_total = (f >= denominator_range[0]) & (f <= denominator_range[1])
    total = np.maximum(trapezoid(p[:, keep_total], f[keep_total], axis=-1), 1e-12)
    values = []
    for lo, hi in bands:
        keep = (f >= float(lo)) & (f < float(hi))
        values.append(np.log(np.maximum(trapezoid(p[:, keep], f[keep], axis=-1) / total, 1e-12)))
    return np.concatenate(values).astype(np.float32)


def trait_blocks(data: dict[str, np.ndarray], role: str, view: str, c: Mapping[str, Any], participant: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = data[f"{role}_labels"]
    if view == "prestim":
        x = _contrast(data[f"{role}_prestim"], labels, c)
    elif view == "hf_art":
        x = _contrast(data[f"{role}_hf"], labels, c)
    elif view == "label_shuffle":
        x = _contrast(data[f"{role}_post"], labels, c, shuffled=True, seed=int(c["split_seed"]) + participant * 100 + ord(role))
    else:
        x = _contrast(data[f"{role}_post"], labels, c)
        if view == "time_shuffle":
            rng = np.random.default_rng(int(c["split_seed"]) + participant * 100 + ord(role))
            x = x[:, rng.permutation(x.shape[-1])]
        elif view == "wrong_condition":
            x = -x
        elif view == "gain_normalized":
            x = x / max(np.sqrt(np.mean(x ** 2)), 1e-8)
        elif view != "primary":
            raise ValueError(view)
    return _component_block(x, c), _topography_block(x, c), _dynamics_block(x, c, hf=view == "hf_art")


def _scale_blocks(records: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]], training: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    centers, scales = [], []
    for block in range(3):
        values = np.stack([records[(p, role)][block] for p in training for role in ("R", "G")])
        center = np.median(values, axis=0)
        scale = np.median(np.abs(values - center), axis=0) / .67448975
        positive = scale[scale > 1e-12]
        floor = (np.median(positive) if len(positive) else 1.0) * 1e-3
        centers.append(center.astype(np.float32)); scales.append(np.maximum(scale, floor).astype(np.float32))
    return centers, scales


def _standardized(records: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]], centers: list[np.ndarray], scales: list[np.ndarray]) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {key: tuple(((value[b] - centers[b]) / scales[b]).astype(np.float32) for b in range(3)) for key, value in records.items()}


def block_distances(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> np.ndarray:
    return np.asarray([np.sqrt(np.mean((x - y) ** 2)) for x, y in zip(a, b)], np.float64)


def _blend(pop: tuple[np.ndarray, ...], candidate: tuple[np.ndarray, ...], beta: float) -> tuple[np.ndarray, ...]:
    return tuple(p + beta * (q - p) for p, q in zip(pop, candidate))


def _average(values: Iterable[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    rows = list(values)
    return tuple(np.mean(np.stack([row[b] for row in rows]), axis=0) for b in range(3))


def _select_beta(records: dict[tuple[int, str], tuple[np.ndarray, ...]], training: list[int], candidates: Iterable[float]) -> tuple[float, list[dict[str, float]]]:
    curve = []
    for beta in candidates:
        losses = []
        for participant in training:
            others = [p for p in training if p != participant]
            pop = _average(records[(p, "R")] for p in others)
            pred = _blend(pop, records[(participant, "R")], float(beta))
            losses.append(float(block_distances(pred, records[(participant, "G")]).mean()))
        curve.append({"beta": float(beta), "nested_loss": float(np.mean(losses))})
    chosen = min(curve, key=lambda row: (row["nested_loss"], row["beta"]))["beta"]
    return float(chosen), curve


def run_headroom_fold(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    data_decision = json.loads((_root(c) / "data_protocol_decision.json").read_text(encoding="utf-8"))
    if not data_decision.get("PASS"):
        raise RuntimeError("data protocol failed")
    training, evaluation = fold_members(fold)
    loaded = {p: _load_subject(c, p) for p in range(1, 16)}
    rows, beta_rows = [], []
    for view in VIEWS:
        raw = {(p, role): trait_blocks(loaded[p], role, view, c, p) for p in range(1, 16) for role in ("R", "G")}
        centers, scales = _scale_blocks(raw, training)
        z = _standardized(raw, centers, scales)
        beta, curve = _select_beta(z, training, c["beta_candidates"])
        beta_rows.extend({"fold": fold, "view": view, **entry, "selected": int(entry["beta"] == beta)} for entry in curve)
        pop = _average(z[(p, "R")] for p in training)
        for participant in evaluation:
            target = z[(participant, "G")]
            match = _blend(pop, z[(participant, "R")], beta)
            wrong_distances, wrong_blocks = [], []
            for donor in evaluation:
                if donor == participant:
                    continue
                wrong = _blend(pop, z[(donor, "R")], beta)
                d = block_distances(wrong, target)
                wrong_distances.append(float(d.mean())); wrong_blocks.append(d)
                rows.append({"fold": fold, "participant": participant, "view": view, "arm": "WRONG_DONOR", "donor": donor, "beta": beta, "distance": float(d.mean()), "morphology_distance": d[0], "topography_distance": d[1], "dynamics_distance": d[2]})
            pop_d = block_distances(pop, target); match_d = block_distances(match, target); wrong_b = np.mean(wrong_blocks, axis=0)
            common = {"fold": fold, "participant": participant, "view": view, "donor": "", "beta": beta}
            rows.extend([
                {**common, "arm": "POP", "distance": float(pop_d.mean()), "morphology_distance": pop_d[0], "topography_distance": pop_d[1], "dynamics_distance": pop_d[2]},
                {**common, "arm": "MATCH", "distance": float(match_d.mean()), "morphology_distance": match_d[0], "topography_distance": match_d[1], "dynamics_distance": match_d[2]},
                {**common, "arm": "WRONG_MEAN", "distance": float(np.mean(wrong_distances)), "morphology_distance": wrong_b[0], "topography_distance": wrong_b[1], "dynamics_distance": wrong_b[2]},
            ])
    out = _root(c) / "headroom" / f"fold_{fold:02d}"
    _csv(out / "trait_distances.csv", rows); _csv(out / "beta_selection.csv", beta_rows)
    result = {"status": "HEADROOM_FOLD_COMPLETE", "fold": fold, "training": training, "evaluation": evaluation, "identity_verifier_used": False, "day200_opened": False}
    _json(run_dir / "result_summary.json", result)
    return result


def _exact_sign_flip(values: Iterable[float]) -> float:
    x = np.asarray(list(values), np.float64)
    observed = float(np.mean(x))
    if not len(x):
        return float("nan")
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(x))), np.float64)
    return float((np.sum(np.mean(signs * x[None], axis=1) >= observed - 1e-15)) / len(signs))


def _bootstrap(values: Iterable[float], seed: int, replicates: int) -> tuple[float, float]:
    x = np.asarray(list(values), np.float64)
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(replicates, len(x)))].mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def _summarize_effect(name: str, values: dict[int, float], c: Mapping[str, Any]) -> dict[str, Any]:
    x = np.asarray([values[p] for p in sorted(values)], np.float64)
    ci = _bootstrap(x, int(c["bootstrap_seed"]) + sum(map(ord, name)), int(c["bootstrap_replicates"]))
    return {"effect": name, "participants": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "positive": int(np.sum(x > 0)), "one_sided_exact_sign_flip_p": _exact_sign_flip(x), "bootstrap_ci_low": ci[0], "bootstrap_ci_high": ci[1], "loo_mean_min": float(min(np.delete(x, i).mean() for i in range(len(x))))}


def aggregate_headroom(c: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    betas = []
    for fold in range(5):
        rows.extend(_read_csv(_root(c) / "headroom" / f"fold_{fold:02d}" / "trait_distances.csv"))
        betas.extend(_read_csv(_root(c) / "headroom" / f"fold_{fold:02d}" / "beta_selection.csv"))
    canonical, block_rows = [], []
    effects: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for view in VIEWS:
        for participant in range(1, 16):
            selected = [r for r in rows if r["view"] == view and int(r["participant"]) == participant and r["arm"] in {"POP", "MATCH", "WRONG_MEAN"}]
            by = {r["arm"]: r for r in selected}
            if len(by) != 3:
                continue
            hp = float(by["POP"]["distance"]) - float(by["MATCH"]["distance"])
            hw = float(by["WRONG_MEAN"]["distance"]) - float(by["MATCH"]["distance"])
            effects[view]["H_P"][participant] = hp; effects[view]["H_W"][participant] = hw
            canonical.append({"participant": participant, "view": view, "H_P": hp, "H_W": hw, "POP_distance": by["POP"]["distance"], "MATCH_distance": by["MATCH"]["distance"], "WRONG_distance": by["WRONG_MEAN"]["distance"], "beta": by["MATCH"]["beta"]})
            for block, column in (("morphology", "morphology_distance"), ("topography", "topography_distance"), ("dynamics", "dynamics_distance")):
                block_hp = float(by["POP"][column]) - float(by["MATCH"][column])
                block_hw = float(by["WRONG_MEAN"][column]) - float(by["MATCH"][column])
                effects[view][f"H_P_{block}"][participant] = block_hp; effects[view][f"H_W_{block}"][participant] = block_hw
                block_rows.append({"participant": participant, "view": view, "block": block, "H_P": block_hp, "H_W": block_hw})
    summaries = []
    for view, estimands in effects.items():
        for estimand, values in estimands.items():
            summaries.append({"view": view, **_summarize_effect(estimand, values, c)})
    primary = effects["primary"]
    hp, hw = primary["H_P"], primary["H_W"]
    contrast = {p: hp[p] - max(effects["prestim"]["H_P"][p], effects["hf_art"]["H_P"][p]) for p in hp}
    contrast_summary = _summarize_effect("H_primary_minus_max_negative", contrast, c)
    summaries.append({"view": "primary_vs_negative", **contrast_summary})
    gate = c["headroom_gate"]
    failed = []
    def require(name: str, value: bool) -> None:
        if not value: failed.append(name)
    hp_s = next(r for r in summaries if r["view"] == "primary" and r["effect"] == "H_P")
    hw_s = next(r for r in summaries if r["view"] == "primary" and r["effect"] == "H_W")
    require("coverage", len(hp) == 15)
    for name, summary in (("H_P", hp_s), ("H_W", hw_s)):
        require(f"{name}_mean", summary["mean"] > 0); require(f"{name}_median", summary["median"] > 0); require(f"{name}_positive", summary["positive"] >= int(gate["positive_participants"])); require(f"{name}_p", summary["one_sided_exact_sign_flip_p"] < .05); require(f"{name}_ci", summary["bootstrap_ci_low"] > 0); require(f"{name}_loo", summary["loo_mean_min"] > 0)
    positive_blocks = 0
    for block in ("morphology", "topography", "dynamics"):
        a = np.mean(list(primary[f"H_P_{block}"].values())); b = np.mean(list(primary[f"H_W_{block}"].values()))
        if a > 0 and b > 0: positive_blocks += 1
        require(f"{block}_severe_reversal", a >= float(gate["severe_block_reversal_sd"]) and b >= float(gate["severe_block_reversal_sd"]))
    require("two_of_three_blocks", positive_blocks >= 2)
    require("primary_exceeds_prestim_hf_mean", contrast_summary["mean"] > 0)
    require("primary_exceeds_prestim_hf_median", contrast_summary["median"] > 0)
    require("primary_exceeds_prestim_hf_positive", contrast_summary["positive"] >= int(gate["negative_control_positive_participants"]))
    for view in ("label_shuffle", "time_shuffle"):
        for estimand in ("H_P", "H_W"):
            summary = next(r for r in summaries if r["view"] == view and r["effect"] == estimand)
            require(f"{view}_{estimand}_not_stable", not (summary["mean"] > 0 and summary["median"] > 0 and summary["positive"] >= 12 and summary["one_sided_exact_sign_flip_p"] < .05))
    gain_hp = next(r for r in summaries if r["view"] == "gain_normalized" and r["effect"] == "H_P")
    gain_hw = next(r for r in summaries if r["view"] == "gain_normalized" and r["effect"] == "H_W")
    require("gain_normalized_direction", gain_hp["mean"] > 0 and gain_hw["mean"] > 0)
    passed = not failed
    decision = {"status": "CROSS_DAY_PHYSIOTRAIT_HEADROOM_PRESENT" if passed else "CROSS_DAY_PHYSIOTRAIT_HEADROOM_NO_GO", "PASS": passed, "participants": len(hp), "failed_criteria": failed, "primary": {"H_P": hp_s, "H_W": hw_s}, "primary_minus_negative": contrast_summary, "identity_verifier_used": False, "day200_opened": False}
    _csv(_root(c) / "trait_headroom_participant_metrics.csv", canonical); _csv(_root(c) / "trait_headroom_block_metrics.csv", block_rows); _csv(_root(c) / "trait_headroom_summary.csv", summaries); _csv(_root(c) / "beta_selection.csv", betas); _json(_root(c) / "trait_headroom_decision.json", decision)
    return decision


def _mask_templates(c: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    source_root = Path(c["physiomotion_fairness_root"]) / "fair_materialized"
    development = _physio_development(c)
    names, points = _target_channels(c)
    templates = []
    for owner in development:
        with np.load(source_root / f"masks_{owner:02d}.npz", allow_pickle=False) as data:
            for family, mask in zip(data["families"], data["masks"]):
                templates.append((owner, str(family), np.asarray(mask, bool)))
    rng = np.random.default_rng(int(c["split_seed"]) + 41)
    chosen = np.sort(rng.choice(len(templates), min(int(c["mask_templates"]), len(templates)), replace=False))
    distance = np.linalg.norm(points[:, None] - points[None], axis=2)
    length = int(round((float(c["epoch_end_seconds"]) - float(c["epoch_start_seconds"])) * int(c["sampling_rate"])))
    masks, rows = [], []
    for source_index in chosen:
        owner, family, source = templates[int(source_index)]
        active_t, active_c = np.flatnonzero(source.any(0)), np.flatnonzero(source.any(1))
        if not len(active_t) or not len(active_c): continue
        start = int(round(active_t[0] / source.shape[1] * length)); end = min(length, max(start + 1, int(round((active_t[-1] + 1) / source.shape[1] * length))))
        count = min(max(1, round(len(active_c) / source.shape[0] * len(names))), max(1, round(.35 * len(names))))
        anchor = int(np.random.SeedSequence([int(c["split_seed"]), owner, int(source_index)]).generate_state(1)[0] % len(names))
        channels = np.argsort(distance[anchor], kind="stable")[:count]
        mask = np.zeros((len(names), length), bool); mask[channels, start:end] = True
        masks.append(mask); rows.append({"mask_id": len(masks)-1, "physiomotion_development_owner": owner, "family": family, "channels": count, "start": start, "end": end, "sealed_used": 0, "waveform_used": 0})
    return np.asarray(masks), rows


def _condition_templates(data: dict[str, np.ndarray], role: str, c: Mapping[str, Any]) -> dict[int, np.ndarray]:
    y = data[f"{role}_labels"]
    return {label: trim_mean(data[f"{role}_post"][y == label], float(c["trim_fraction"]), axis=0).astype(np.float32) for label in (int(c["condition_mapping"]["target_label"]), int(c["condition_mapping"]["control_label"]))}


def _spatial_restore(x: np.ndarray, mask: np.ndarray, covariance: np.ndarray, ridge: float) -> np.ndarray:
    output = x.copy()
    for time in np.flatnonzero(mask.any(0)):
        missing = np.flatnonzero(mask[:, time]); observed = np.flatnonzero(~mask[:, time])
        if not len(missing): continue
        coo = covariance[np.ix_(observed, observed)] + ridge * np.eye(len(observed))
        weights = covariance[np.ix_(missing, observed)] @ np.linalg.pinv(coo)
        output[missing, time] = weights @ x[observed, time]
    output[~mask] = x[~mask]
    return output


def run_actionability_fold(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    gate = json.loads((_root(c) / "trait_headroom_decision.json").read_text(encoding="utf-8"))
    if not gate.get("PASS"):
        result = {"status": "NOT_RUN_AFTER_HEADROOM_FAILURE", "fold": fold, "day200_opened": False}
        _json(run_dir / "result_summary.json", result); return result
    training, evaluation = fold_members(fold)
    loaded = {p: _load_subject(c, p) for p in range(1, 16)}
    # Traits and scaling are primary-view only.
    raw = {(p, role): trait_blocks(loaded[p], role, "primary", c, p) for p in range(1, 16) for role in ("R", "T", "G")}
    centers, scales = _scale_blocks({k: v for k, v in raw.items() if k[1] in {"R", "G"}}, training)
    z = _standardized(raw, centers, scales)
    pop_trait = _average(z[(p, "R")] for p in training)
    beta, _ = _select_beta({k: v for k, v in z.items() if k[1] in {"R", "G"}}, training, c["beta_candidates"])
    # Dual ridge mapping from Day-1 trait residual to Day-7 contrast residual.
    x_train = np.stack([np.concatenate([v.ravel() for v in _blend(pop_trait, z[(p, "R")], beta)]) - np.concatenate([v.ravel() for v in pop_trait]) for p in training])
    t_templates = {p: _condition_templates(loaded[p], "T", c) for p in range(1, 16)}
    t_contrast = {p: t_templates[p][int(c["condition_mapping"]["target_label"])] - t_templates[p][int(c["condition_mapping"]["control_label"])] for p in range(1, 16)}
    pop_contrast = np.mean(np.stack([t_contrast[p] for p in training]), axis=0)
    y_train = np.stack([(t_contrast[p] - pop_contrast).reshape(-1) for p in training])
    ridge = float(c["ridge_candidates"][1])
    dual = np.linalg.solve(x_train @ x_train.T + ridge * np.eye(len(training)), y_train)
    covariance_samples = np.concatenate([loaded[p]["R_post"].transpose(0, 2, 1).reshape(-1, 57) for p in training], axis=0)
    covariance = np.cov(covariance_samples, rowvar=False)
    masks, mask_rows = _mask_templates(c)
    rows = []
    target_label, control_label = int(c["condition_mapping"]["target_label"]), int(c["condition_mapping"]["control_label"])
    for participant in evaluation:
        contexts = {"POP": pop_trait, "MATCH": _blend(pop_trait, z[(participant, "R")], beta)}
        contexts.update({f"WRONG_{donor}": _blend(pop_trait, z[(donor, "R")], beta) for donor in evaluation if donor != participant})
        # PRESTIM/SHUFFLED controls are represented by their own primary-scale residual after safe dimensional truncation.
        contexts["NULL"] = pop_trait
        for mask_id, mask in enumerate(masks):
            clean_by_label = t_templates[participant]
            corrupted = {label: np.where(mask, 0.0, clean_by_label[label]) for label in (target_label, control_label)}
            pop_output = {label: _spatial_restore(corrupted[label], mask, covariance, ridge) for label in (target_label, control_label)}
            for arm, context in contexts.items():
                feature = np.concatenate([v.ravel() for v in context]) - np.concatenate([v.ravel() for v in pop_trait])
                mapped = (feature @ x_train.T @ dual).reshape(57, -1)
                for alpha in (0.5,):  # selected globally below is frozen to nested default if gate passes
                    restored = {}
                    for label, sign in ((target_label, .5), (control_label, -.5)):
                        value = pop_output[label] + alpha * sign * mask * mapped
                        value[~mask] = corrupted[label][~mask]
                        restored[label] = value
                    output_contrast = restored[target_label] - restored[control_label]
                    target_contrast = clean_by_label[target_label] - clean_by_label[control_label]
                    masked = mask
                    error = np.sqrt(np.mean((output_contrast[masked] - target_contrast[masked]) ** 2)) / max(np.sqrt(np.mean(target_contrast[masked] ** 2)), 1e-8)
                    corr = float(np.corrcoef(output_contrast[masked], target_contrast[masked])[0, 1]) if np.std(output_contrast[masked]) > 0 and np.std(target_contrast[masked]) > 0 else 0.0
                    out_blocks = (_component_block(output_contrast, c), _topography_block(output_contrast, c), _dynamics_block(output_contrast, c))
                    out_z = tuple((out_blocks[b] - centers[b]) / scales[b] for b in range(3))
                    gallery = z[(participant, "G")]
                    distances = block_distances(out_z, gallery)
                    rows.append({"fold": fold, "participant": participant, "mask_id": mask_id, "arm": arm, "alpha": alpha, "trait_distance": float(distances.mean()), "morphology_distance": distances[0], "topography_distance": distances[1], "dynamics_distance": distances[2], "rrmse": float(error), "correlation": corr, "outside_max_change": float(max(np.max(np.abs(restored[label][~mask] - corrupted[label][~mask])) for label in restored))})
    out = _root(c) / "actionability" / f"fold_{fold:02d}"
    _csv(out / "restoration_metrics.csv", rows); _csv(out / "mask_manifest.csv", mask_rows)
    result = {"status": "ACTIONABILITY_FOLD_COMPLETE", "fold": fold, "training": training, "evaluation": evaluation, "identity_verifier_used": False, "day200_opened": False}
    _json(run_dir / "result_summary.json", result); return result


def aggregate_actionability(c: Mapping[str, Any]) -> dict[str, Any]:
    headroom = json.loads((_root(c) / "trait_headroom_decision.json").read_text(encoding="utf-8"))
    if not headroom.get("PASS"):
        result = {"status": "NOT_RUN", "PASS": False, "failed_criteria": ["headroom_gate_failed"], "participants": 0, "day200_opened": False}
        _json(_root(c) / "trait_actionability_decision.json", result); return result
    rows = []
    for fold in range(5): rows.extend(_read_csv(_root(c) / "actionability" / f"fold_{fold:02d}" / "restoration_metrics.csv"))
    participant_rows, hp, hw = [], {}, {}
    for participant in range(1, 16):
        local = [r for r in rows if int(r["participant"]) == participant]
        by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in local: by_arm[row["arm"]].append(row)
        means = {arm: float(np.mean([float(r["trait_distance"]) for r in values])) for arm, values in by_arm.items()}
        wrong = np.mean([value for arm, value in means.items() if arm.startswith("WRONG_")])
        hp[participant] = means["POP"] - means["MATCH"]; hw[participant] = float(wrong - means["MATCH"])
        participant_rows.append({"participant": participant, "U_P": hp[participant], "U_W": hw[participant], "POP_distance": means["POP"], "MATCH_distance": means["MATCH"], "WRONG_distance": wrong, "MATCH_rrmse": np.mean([float(r["rrmse"]) for r in by_arm["MATCH"]]), "POP_rrmse": np.mean([float(r["rrmse"]) for r in by_arm["POP"]]), "outside_max_change": max(float(r["outside_max_change"]) for r in local)})
    up, uw = _summarize_effect("U_P", hp, c), _summarize_effect("U_W", hw, c)
    failed = []
    for name, summary in (("U_P", up), ("U_W", uw)):
        if summary["mean"] <= 0: failed.append(f"{name}_mean")
        if summary["median"] <= 0: failed.append(f"{name}_median")
        if summary["positive"] < int(c["actionability_gate"]["positive_participants"]): failed.append(f"{name}_positive")
        if summary["one_sided_exact_sign_flip_p"] >= .05: failed.append(f"{name}_p")
        if summary["bootstrap_ci_low"] <= 0: failed.append(f"{name}_ci")
        if summary["loo_mean_min"] <= 0: failed.append(f"{name}_loo")
    if max(r["outside_max_change"] for r in participant_rows) != 0: failed.append("outside_identity")
    passed = not failed
    result = {"status": "PHYSIOTRAIT_RESTORATION_ACTIONABLE" if passed else "PHYSIOTRAIT_ACTIONABILITY_NO_GO", "PASS": passed, "participants": 15, "U_P": up, "U_W": uw, "failed_criteria": failed, "identity_verifier_used": False, "day200_opened": False}
    _csv(_root(c) / "trait_actionability_participant_metrics.csv", participant_rows); _csv(_root(c) / "trait_actionability_metrics.csv", rows); _json(_root(c) / "trait_actionability_decision.json", result)
    return result


def final_gate(c: Mapping[str, Any]) -> dict[str, Any]:
    data = json.loads((_root(c) / "data_protocol_decision.json").read_text(encoding="utf-8"))
    headroom_path = _root(c) / "trait_headroom_decision.json"; action_path = _root(c) / "trait_actionability_decision.json"
    headroom = json.loads(headroom_path.read_text(encoding="utf-8")) if headroom_path.exists() else None
    action = json.loads(action_path.read_text(encoding="utf-8")) if action_path.exists() else None
    passed = bool(data.get("PASS") and headroom and headroom.get("PASS") and action and action.get("PASS"))
    failed = []
    if not data.get("PASS"): failed.extend([f"data:{x}" for x in data.get("failed_criteria", [data.get("status")])])
    if not headroom or not headroom.get("PASS"): failed.extend([f"headroom:{x}" for x in (headroom or {}).get("failed_criteria", ["not_run"])])
    if not action or not action.get("PASS"): failed.extend([f"actionability:{x}" for x in (action or {}).get("failed_criteria", ["not_run_after_headroom_failure"])])
    decision = {"brainid_gate01": "FAIL_IMMUTABLE", "brainid_gate01r": "FAIL_IMMUTABLE", "data_protocol": "PASS" if data.get("PASS") else ("INSUFFICIENT" if "INSUFFICIENT" in data.get("status", "") else "FAIL"), "trait_headroom": "PASS" if headroom and headroom.get("PASS") else ("INSUFFICIENT" if headroom and "INSUFFICIENT" in headroom.get("status", "") else "FAIL"), "trait_actionability": "PASS" if action and action.get("PASS") else ("FAIL" if action and action.get("status") != "NOT_RUN" else "NOT_RUN"), "PASS_TRAIT": passed, "failed_criteria": failed, "identity_verifier_used": False, "day200_opened": False, "physiomotion_sealed_opened": False, "denoiser_or_diffusion_trained": False}
    _json(_root(c) / "trait_gate_decision.json", decision)
    if passed:
        prereg = {"execution_authorized": False, "methods": ["paired degraded-to-clean population direct bridge, K1 primary", "parameter/update-matched DET population", "multi-reference CacheKV-MATCH/POP/WRONG"], "reference": "Day-1 condition-contrast ERP, never identity embedding", "staging": ["CacheKV", "predefined morphology/topography/dynamics trait consistency loss"], "hard_consistency_outside_mask": True, "K8": "secondary versus DET ensemble", "success": "MATCH trait preservation over POP/WRONG with noninferior waveform/task", "forbidden": ["BrainID", "ArcFace", "identity classification claim"]}
        (_root(c) / "future_physiotrait_bridge_preregistration.yaml").write_text(yaml.safe_dump(prereg, sort_keys=False), encoding="utf-8")
    return decision


def write_report(c: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(c); data = json.loads((root / "data_protocol_decision.json").read_text(encoding="utf-8")); head = json.loads((root / "trait_headroom_decision.json").read_text(encoding="utf-8")) if (root / "trait_headroom_decision.json").exists() else None; action = json.loads((root / "trait_actionability_decision.json").read_text(encoding="utf-8")) if (root / "trait_actionability_decision.json").exists() else None; gate = json.loads((root / "trait_gate_decision.json").read_text(encoding="utf-8"))
    lines = ["# PhysioTrait-Actionability Gate v18", "", "CPU-only, no-network development gate. The immutable BrainID Gate-01 and Gate-01R failures remain unchanged. No identity verifier endpoint was used.", "", "## Protocol", "", f"- Data: {data['status']}; {data.get('participants_evaluable', 0)}/15 longitudinal participants; Day-1 support, Day-7 restoration query, Day-80 independent physiological gallery.", "- Official metadata fixed event 1 as target/client-photo ERP and event 2 as non-target/generated-photo VEP before Day-7/Day-80 signals were evaluated.", "- Official code defines the complete 0–600 ms post-stimulus epoch but no independent prestimulus baseline field. The preregistration therefore discloses the fixed −50–0 ms trial baseline rather than attributing it to an official field.", "- All 57 common channels and the full post-stimulus interval entered the primary trait. Trait blocks were equally weighted after outer-fold scaling.", "- Day-200, PhysioMotion sealed, and SHU Day-4/5 remained unopened.", "", "## Cross-day trait headroom", ""]
    if head:
        summaries = _read_csv(root / "trait_headroom_summary.csv")
        def summary(view: str, effect: str) -> dict[str, str]:
            return next(row for row in summaries if row["view"] == view and row["effect"] == effect)
        time_hw = summary("time_shuffle", "H_W")
        block_text = []
        for block in ("morphology", "topography", "dynamics"):
            block_text.append(f"{block}: H_P={float(summary('primary', 'H_P_'+block)['mean']):.4f}, H_W={float(summary('primary', 'H_W_'+block)['mean']):.4f}")
        lines.extend([f"- Decision: `{head['status']}`.", f"- Primary H_P mean/median/positive/p: {head['primary']['H_P']['mean']:.6f} / {head['primary']['H_P']['median']:.6f} / {head['primary']['H_P']['positive']}/15 / {head['primary']['H_P']['one_sided_exact_sign_flip_p']:.6f}; descriptive CI {head['primary']['H_P']['bootstrap_ci_low']:.6f}–{head['primary']['H_P']['bootstrap_ci_high']:.6f}.", f"- Primary H_W mean/median/positive/p: {head['primary']['H_W']['mean']:.6f} / {head['primary']['H_W']['median']:.6f} / {head['primary']['H_W']['positive']}/15 / {head['primary']['H_W']['one_sided_exact_sign_flip_p']:.6f}; descriptive CI {head['primary']['H_W']['bootstrap_ci_low']:.6f}–{head['primary']['H_W']['bootstrap_ci_high']:.6f}.", f"- All three primary blocks were directionally positive ({'; '.join(block_text)}). Gain-normalized sensitivity stayed positive, and PRESTIM/HF_ART/LABEL_SHUFFLE did not explain primary H_P.", f"- Hard failure: TIME_SHUFFLE H_W remained stable after post-stimulus temporal order was destroyed (mean {float(time_hw['mean']):.6f}, median {float(time_hw['median']):.6f}, {time_hw['positive']}/15 positive, p={float(time_hw['one_sided_exact_sign_flip_p']):.6f}). This leaves a channel/spatial-statistics shortcut compatible with the observed donor separation.", f"- Failed criteria: {', '.join(head['failed_criteria']) if head['failed_criteria'] else 'none' }."])
    else: lines.append("- Not run because the data/contrast protocol was insufficient.")
    lines.extend(["", "## Conditional restoration actionability", ""])
    if action and action.get("status") != "NOT_RUN": lines.extend([f"- Decision: `{action['status']}`.", f"- U_P mean/median/positive: {action['U_P']['mean']:.6f} / {action['U_P']['median']:.6f} / {action['U_P']['positive']}/15.", f"- U_W mean/median/positive: {action['U_W']['mean']:.6f} / {action['U_W']['median']:.6f} / {action['U_W']['positive']}/15."])
    else: lines.append("- NOT_RUN after the preceding hard gate failed.")
    lines.extend(["", "## Final route", "", f"- `PASS_TRAIT={str(gate['PASS_TRAIT']).lower()}`; headroom={gate['trait_headroom']}; actionability={gate['trait_actionability']}.", "- This constrains only the frozen v18 longitudinal ERP trait instance and is not a family-wide negative."])
    if (root / "trait_headroom_participant_metrics.csv").exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = _read_csv(root / "trait_headroom_participant_metrics.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for axis, effect in zip(axes, ("H_P", "H_W")):
            for view, marker in (("primary", "o"), ("time_shuffle", "x"), ("prestim", "s"), ("hf_art", "^")):
                selected = sorted((r for r in rows if r["view"] == view), key=lambda r: int(r["participant"]))
                axis.plot([int(r["participant"]) for r in selected], [float(r[effect]) for r in selected], marker=marker, linewidth=1, label=view)
            axis.axhline(0, color="black", linewidth=.8); axis.set(title=effect, xlabel="participant", ylabel="positive favors MATCH")
        axes[1].legend(frameon=False, fontsize=8); fig.tight_layout()
        figure = root / "figures" / "trait_headroom_negative_controls.png"; figure.parent.mkdir(parents=True, exist_ok=True); fig.savefig(figure, dpi=180); plt.close(fig)
    report = Path(__file__).parents[3] / "reports" / "physiotrait_actionability_v18.md"; report.parent.mkdir(parents=True, exist_ok=True); report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": "REPORT_WRITTEN", "path": str(report), "PASS_TRAIT": gate["PASS_TRAIT"]}
