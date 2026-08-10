"""PhysioMotion metadata freeze and subject clean-patch retrieval headroom."""

from __future__ import annotations

import csv
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from scipy.signal import resample_poly, welch


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result(c: Mapping[str, Any]) -> Path: return Path(c["result_root"])


def _preprocessed(c: Mapping[str, Any], participant: int, run: int) -> Path:
    root = Path(c["data_root"]) / "derivatives" / "preprocessed_BIDS" / f"sub-{participant}" / "eeg"
    return root / f"sub-{participant}_task-artifact_run-{run:02d}_eeg.edf"


def _annotation(c: Mapping[str, Any], participant: int, run: int) -> Path:
    return Path(c["data_root"]) / "derivatives" / "Manual_Annotations" / f"sub{participant}_run{run:02d}.csv"


def _family(label: str) -> str | None:
    value = label.strip().lower().replace("-", "_")
    if any(token in value for token in ("blink", "sacc", "eyem", "eye_movement")): return None
    if "head" in value and any(token in value for token in ("hor", "ver", "horizontal", "vertical", "head")): return "head_motion"
    if "chew" in value: return "chewing"
    if "tongue" in value: return "tongue"
    if "swallow" in value: return "swallowing"
    if "eyebrow" in value or "facial" in value or "face" in value: return "facial_emg"
    return None


def _split(c: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(c["split_seed"])); development, sealed = [], []
    for first in range(1, 31, 3):
        group = list(range(first, first + 3)); chosen = group[int(rng.integers(0, 3))]; sealed.append(chosen); development.extend(p for p in group if p != chosen)
    return sorted(development), sorted(sealed)


def stage_metadata(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    root = Path(c["data_root"]); description = json.loads((root / "dataset_description.json").read_text())
    if description.get("DatasetDOI") != "doi:10.18112/openneuro.ds006386.v1.0.1": raise RuntimeError(description)
    development, sealed = _split(c); split_rows = []
    rng = np.random.default_rng(int(c["split_seed"])); shuffled = list(development); rng.shuffle(shuffled); folds = {p: index // 4 for index, p in enumerate(shuffled)}
    for participant in range(1, 31):
        split_rows.append({"participant": participant, "role": "development" if participant in development else "sealed", "cv_fold": folds.get(participant, ""), "signal_opened": 0, "annotation_opened": 0})
    _csv_write(_result(c) / "metadata" / "frozen_participant_split.csv", split_rows)
    inventory, dates, dev_labels = [], defaultdict(set), defaultdict(int)
    ordered_layouts: set[str] = set()
    dev_family_rows, dev_family_seconds = defaultdict(int), defaultdict(float)
    for participant in range(1, 31):
        scans = root / f"sub-{participant}" / f"sub-{participant}_scans.tsv"
        scan_rows = _csv_read(scans) if scans.exists() else []
        for row in scan_rows:
            match = re.search(r"run-(\d+)", row.get("filename", "")); date = row.get("acq_time", "")[:10]
            if match and date: dates[participant].add(date)
        for run in range(1, 7):
            edf = _preprocessed(c, participant, run); ann = _annotation(c, participant, run)
            side = edf.with_name(edf.name.replace("_eeg.edf", "_eeg.json")); channels = edf.with_name(edf.name.replace("_eeg.edf", "_channels.tsv"))
            side_values = json.loads(side.read_text()) if side.exists() else {}
            channel_rows = _csv_read(channels) if channels.exists() else []
            if channel_rows: ordered_layouts.add(";".join(row["name"] for row in channel_rows))
            inventory.append({"participant": participant, "run": run, "role": "development" if participant in development else "sealed", "edf_exists": int(edf.exists()), "sidecar_exists": int(side.exists()), "channels_exists": int(channels.exists()), "annotation_exists_without_open": int(ann.exists()), "channels": len(channel_rows), "sampling_rate": side_values.get("SamplingFrequency", ""), "recording_duration": side_values.get("RecordingDuration", ""), "signal_opened": 0})
            if participant in development and ann.exists():
                for row in _csv_read(ann):
                    label = row["label"].strip().lower(); dev_labels[label] += 1; family = _family(label)
                    if family:
                        dev_family_rows[family] += 1
                        dev_family_seconds[family] += max(0.0, float(row["stop_time"]) - float(row["start_time"]))
    _csv_write(_result(c) / "metadata" / "file_inventory.csv", inventory)
    _csv_write(_result(c) / "metadata" / "development_annotation_labels.csv", [{"label": key, "rows": value, "primary_family": _family(key) or "excluded_or_secondary"} for key, value in sorted(dev_labels.items())])
    cross_day = {p: len(dates[p]) > 1 for p in development}; protocol = "cross-day" if all(cross_day.values()) else "repeated-run"
    summary = {"status": "PHYSIOMOTION_METADATA_FROZEN", "dataset": "OpenNeuro ds006386 v1.0.1", "participants": 30, "runs_per_participant": [1, 2, 3, 4, 5, 6], "runs_expected": 180, "development": development, "sealed": sealed, "sealed_signal_or_annotations_opened": False, "development_cv_folds": 5, "channel_counts": sorted({int(row["channels"]) for row in inventory if row["channels"] != ""}), "sampling_rates": sorted({float(row["sampling_rate"]) for row in inventory if row["sampling_rate"] != ""}), "ordered_channel_layout_count": len(ordered_layouts), "ordered_channel_layouts": sorted(ordered_layouts), "development_annotation_primary_family_rows": dict(dev_family_rows), "development_annotation_primary_family_channel_seconds": dict(dev_family_seconds), "development_baseline_annotation_rows": int(dev_labels.get("open_base", 0) + dev_labels.get("close_base", 0)), "acquisition_dates_by_participant": {str(p): sorted(dates[p]) for p in range(1, 31)}, "participants_with_multiple_dates_in_development": sum(cross_day.values()), "primary_protocol_name": protocol, "cross_day_claim_allowed": protocol == "cross-day", "files_complete": all(row["edf_exists"] and row["sidecar_exists"] and row["channels_exists"] and row["annotation_exists_without_open"] for row in inventory)}
    _json(_result(c) / "metadata" / "dataset_audit.json", summary); _json(run_dir / "result_summary.json", summary); return summary


def _development(c: Mapping[str, Any]) -> set[int]:
    rows = _csv_read(_result(c) / "metadata" / "frozen_participant_split.csv"); return {int(row["participant"]) for row in rows if row["role"] == "development"}


def _read_edf(c: Mapping[str, Any], participant: int, run: int):
    if participant not in _development(c): raise PermissionError(f"sealed participant {participant} signal access refused")
    import mne
    raw = mne.io.read_raw_edf(_preprocessed(c, participant, run), preload=False, verbose="ERROR")
    return raw


def _uniform_starts(start: float, stop: float, length: float, cap: int) -> list[float]:
    values = np.arange(start, stop - length + 1e-9, length)
    if len(values) <= cap: return [float(v) for v in values]
    return [float(values[index]) for index in np.linspace(0, len(values) - 1, cap, dtype=int)]


def _extract(raw: Any, starts: list[float], seconds: float, target_fs: int) -> np.ndarray:
    patches = []
    for start in starts:
        data = raw.get_data(start=int(round(start * raw.info["sfreq"])), stop=int(round((start + seconds) * raw.info["sfreq"])))
        data = resample_poly(data, target_fs, int(round(raw.info["sfreq"])), axis=-1).astype(np.float32)
        expected = int(seconds * target_fs)
        if data.shape[-1] >= expected: patches.append(data[..., :expected])
    return np.stack(patches) if patches else np.empty((0, len(raw.ch_names), int(seconds * target_fs)), np.float32)


def stage_prepare(c: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    development = sorted(_development(c)); participant = development[task_index]; seconds = float(c["patch_seconds"]); fs = int(c["sampling_rate"]); cap = int(c["support_bank_patches_per_state"]); qcap = int(c["query_patches_per_state_run"])
    support, query, query_state, query_run, natural, natural_mask, natural_family, natural_run = [], [], [], [], [], [], [], []
    channel_names = None; template_masks, template_families = [], []
    for run in range(1, 7):
        ann = _csv_read(_annotation(c, participant, run)); raw = _read_edf(c, participant, run); channel_names = raw.ch_names if channel_names is None else channel_names
        if raw.ch_names != channel_names: raise RuntimeError("channel order differs across runs")
        for state in ("open_base", "close_base"):
            starts = []
            for row in ann:
                if row["label"].strip().lower() == state and row["channel"].strip().upper() == "ALL": starts += _uniform_starts(float(row["start_time"]), float(row["stop_time"]), seconds, cap if run == 1 else qcap)
            patches = _extract(raw, starts, seconds, fs)
            if run == 1:
                support.append((state, patches[:cap]))
            else:
                query.extend(patches); query_state.extend([state] * len(patches)); query_run.extend([run] * len(patches))
        grouped: dict[tuple[str, float, float], list[str]] = defaultdict(list)
        for row in ann:
            family = _family(row["label"])
            if family: grouped[(family, float(row["start_time"]), float(row["stop_time"]))].append(row["channel"])
        for (family, start, stop), channels in grouped.items():
            center = .5 * (start + stop); patch_start = max(0.0, center - seconds / 2); patch = _extract(raw, [patch_start], seconds, fs)
            if not len(patch): continue
            mask = np.zeros((len(channel_names), int(seconds * fs)), bool)
            # A retrieval query must retain observed context. Preserve the true
            # annotated channel footprint, while globally clipping only its
            # temporal span to half the fixed two-second inpainting patch.
            masked_seconds = min(stop - start, seconds * float(c["max_mask_fraction"]))
            masked_start, masked_stop = center - masked_seconds / 2, center + masked_seconds / 2
            left = max(0, int(round((masked_start - patch_start) * fs))); right = min(mask.shape[1], int(round((masked_stop - patch_start) * fs)))
            indices = range(len(channel_names)) if any(ch.upper() == "ALL" for ch in channels) else [channel_names.index(ch) for ch in channels if ch in channel_names]
            if right > left:
                for index in indices: mask[index, left:right] = True
            if mask.any():
                template_masks.append(mask); template_families.append(family)
                if run > 1: natural.append(patch[0]); natural_mask.append(mask); natural_family.append(family); natural_run.append(run)
    support_by_state = {state: patches for state, patches in support}
    # A participant is usable when at least one baseline state has a valid K=8
    # early-run bank. Query states without a matching bank remain unavailable.
    eligible = any(len(support_by_state.get(state, [])) >= 8 for state in ("open_base", "close_base")) and len(query) > 0
    target = _result(c) / "prepared" / f"participant_{participant:02d}.npz"; target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, participant=participant, channels=np.asarray(channel_names), support_open=support_by_state.get("open_base", np.empty((0, len(channel_names), int(seconds * fs)), np.float32)), support_close=support_by_state.get("close_base", np.empty((0, len(channel_names), int(seconds * fs)), np.float32)), query=np.asarray(query, np.float32), query_state=np.asarray(query_state), query_run=np.asarray(query_run), artifact_masks=np.asarray(template_masks, bool), artifact_families=np.asarray(template_families), natural=np.asarray(natural, np.float32), natural_mask=np.asarray(natural_mask, bool), natural_family=np.asarray(natural_family), natural_run=np.asarray(natural_run), eligible=int(eligible))
    summary = {"participant": participant, "eligible": bool(eligible), "support_open": len(support_by_state.get("open_base", [])), "support_close": len(support_by_state.get("close_base", [])), "later_clean_patches": len(query), "primary_artifact_templates": len(template_masks), "natural_artifact_patches": len(natural), "sealed_opened": False}; _json(run_dir / "result_summary.json", summary); return summary


def _retrieve(query: np.ndarray, mask: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    observed = ~mask
    if int(observed.sum()) == 0: raise RuntimeError("retrieval mask leaves no observed context")
    q = query[observed].astype(np.float64); q = (q - q.mean()) / max(q.std(), 1e-8)
    candidates = bank[:, observed].astype(np.float64); candidates = (candidates - candidates.mean(1, keepdims=True)) / np.maximum(candidates.std(1, keepdims=True), 1e-8)
    scores = candidates @ q / len(q); chosen = np.argsort(scores, kind="stable")[-k:]
    return np.mean(bank[chosen], axis=0)


def _metrics(clean: np.ndarray, restored: np.ndarray, mask: np.ndarray, fs: int) -> dict[str, float]:
    truth, pred = clean[mask], restored[mask]; rrmse = np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1e-12); corr = np.corrcoef(pred, truth)[0, 1] if len(truth) > 1 and np.std(pred) > 0 and np.std(truth) > 0 else 0.0
    _, p_clean = welch(clean, fs=fs, nperseg=min(fs, clean.shape[-1]), axis=-1); freq, p_pred = welch(restored, fs=fs, nperseg=min(fs, clean.shape[-1]), axis=-1); band = (freq >= 1) & (freq <= 45); spectral = np.mean(np.abs(np.log(np.maximum(p_pred[:, band], 1e-18)) - np.log(np.maximum(p_clean[:, band], 1e-18))))
    t_clean = np.sqrt(np.mean(clean ** 2, axis=-1)); t_pred = np.sqrt(np.mean(restored ** 2, axis=-1)); topography = np.linalg.norm(t_pred / max(np.linalg.norm(t_pred), 1e-12) - t_clean / max(np.linalg.norm(t_clean), 1e-12))
    return {"rrmse": float(rrmse), "correlation": float(corr), "spectral_error": float(spectral), "topography_error": float(topography)}


def stage_headroom(c: Mapping[str, Any], task_index: int, run_dir: Path) -> dict[str, Any]:
    split = _csv_read(_result(c) / "metadata" / "frozen_participant_split.csv"); recipients = sorted(int(row["participant"]) for row in split if row["role"] == "development" and int(row["cv_fold"]) == task_index); training = sorted(int(row["participant"]) for row in split if row["role"] == "development" and int(row["cv_fold"]) != task_index)
    loaded = {p: np.load(_result(c) / "prepared" / f"participant_{p:02d}.npz") for p in sorted(set(recipients + training))}; rows = []
    try:
        masks_by_family = defaultdict(list)
        for p in training:
            for mask, family in zip(loaded[p]["artifact_masks"], loaded[p]["artifact_families"]): masks_by_family[str(family)].append(mask)
        families = list(c["primary_families"]); k = int(c["retrieval_k"]); fs = int(c["sampling_rate"])
        for recipient in recipients:
            data = loaded[recipient]
            if not int(data["eligible"]): continue
            for index, (clean, state, query_run) in enumerate(zip(data["query"], data["query_state"], data["query_run"])):
                state_key = "support_open" if str(state) == "open_base" else "support_close"; match_bank = data[state_key]
                donor_banks = {p: loaded[p][state_key][:int(c["support_bank_patches_per_state"])] for p in training if len(loaded[p][state_key]) >= k}
                if len(match_bank) < k or not donor_banks: continue
                pop_bank = np.concatenate(list(donor_banks.values()), axis=0)
                for family in families:
                    templates = masks_by_family[family]
                    if not templates: continue
                    mask = np.asarray(templates[(recipient * 10000 + index) % len(templates)], bool)
                    methods = {"MATCH": _retrieve(clean, mask, match_bank, k), "POP": _retrieve(clean, mask, pop_bank, k)}
                    for donor, bank in donor_banks.items(): methods[f"WRONG-{donor}"] = _retrieve(clean, mask, bank, k)
                    for method, prediction in methods.items():
                        restored = clean.copy(); restored[mask] = prediction[mask]; rows.append({"fold": task_index, "participant": recipient, "query_run": int(query_run), "state": str(state), "family": family, "method": method, **_metrics(clean, restored, mask, fs)})
    finally:
        for value in loaded.values(): value.close()
    _csv_write(_result(c) / "headroom" / f"fold_{task_index:02d}.csv", rows); summary = {"fold": task_index, "recipients": recipients, "outer_training": training, "rows": len(rows), "recipient_coverage": len({row["participant"] for row in rows}), "sealed_opened": False}; _json(run_dir / "result_summary.json", summary); return summary


def _signflip(values: np.ndarray) -> float:
    observed = float(np.mean(values)); means = [np.mean(values * np.asarray(signs)) for signs in itertools.product((-1, 1), repeat=len(values))]; return float(np.mean(np.asarray(means) >= observed - 1e-15))


def stage_headroom_aggregate(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    rows = []
    for fold in range(5): rows += _csv_read(_result(c) / "headroom" / f"fold_{fold:02d}.csv")
    participant_family, participant, method_metrics = [], [], []
    development = sorted(_development(c))
    for p in development:
        take = [row for row in rows if int(row["participant"]) == p]; families = sorted({row["family"] for row in take}); family_effects = []
        for family in families:
            group = [row for row in take if row["family"] == family]; match = np.mean([float(row["rrmse"]) for row in group if row["method"] == "MATCH"]); pop = np.mean([float(row["rrmse"]) for row in group if row["method"] == "POP"]); donor = defaultdict(list)
            for row in group:
                if row["method"].startswith("WRONG-"): donor[row["method"]].append(float(row["rrmse"]))
            wrong = np.mean([np.mean(value) for value in donor.values()]); effect = {"participant": p, "family": family, "H_P": float(pop - match), "H_W": float(wrong - match), "match_rrmse": float(match), "pop_rrmse": float(pop), "mean_wrong_rrmse": float(wrong)}; participant_family.append(effect); family_effects.append(effect)
            for method in ("MATCH", "POP"):
                selected = [row for row in group if row["method"] == method]
                method_metrics.append({"participant": p, "family": family, "method": method, **{key: float(np.mean([float(row[key]) for row in selected])) for key in ("rrmse", "correlation", "spectral_error", "topography_error")}})
            donor_methods = sorted({row["method"] for row in group if row["method"].startswith("WRONG-")})
            method_metrics.append({"participant": p, "family": family, "method": "mean-WRONG", **{key: float(np.mean([np.mean([float(row[key]) for row in group if row["method"] == donor_method]) for donor_method in donor_methods])) for key in ("rrmse", "correlation", "spectral_error", "topography_error")}})
        if family_effects:
            participant.append({"participant": p, "H_P": float(np.mean([row["H_P"] for row in family_effects])), "H_W": float(np.mean([row["H_W"] for row in family_effects])), "families": len(family_effects), "retrieval_available": 1})
        else:
            participant.append({"participant": p, "H_P": 0.0, "H_W": 0.0, "families": 0, "retrieval_available": 0})
            for family in c["primary_families"]:
                participant_family.append({"participant": p, "family": family, "H_P": 0.0, "H_W": 0.0, "match_rrmse": "", "pop_rrmse": "", "mean_wrong_rrmse": ""})
    def summary(key: str):
        values = np.asarray([row[key] for row in participant]); return {"mean": float(np.mean(values)), "median": float(np.median(values)), "positive": int(np.sum(values > 0)), "n": len(values), "one_sided_exact_sign_flip": _signflip(values), "participant_values": values.tolist()}
    effects = {"H_P": summary("H_P"), "H_W": summary("H_W")}; family_summary = []
    for family in c["primary_families"]:
        take = [row for row in participant_family if row["family"] == family]; family_summary.append({"family": family, "participants": len(take), "H_P": float(np.mean([row["H_P"] for row in take])) if take else float("nan"), "H_W": float(np.mean([row["H_W"] for row in take])) if take else float("nan")})
    participant_method = []
    for p in development:
        for method in ("MATCH", "POP", "mean-WRONG"):
            take = [row for row in method_metrics if int(row["participant"]) == p and row["method"] == method]
            if take: participant_method.append({"participant": p, "method": method, **{key: float(np.mean([row[key] for row in take])) for key in ("rrmse", "correlation", "spectral_error", "topography_error")}})
    method_summary = []
    for method in ("MATCH", "POP", "mean-WRONG"):
        take = [row for row in participant_method if row["method"] == method]
        method_summary.append({"method": method, "participants": len(take), **{key: float(np.mean([row[key] for row in take])) for key in ("rrmse", "correlation", "spectral_error", "topography_error")}})
    coverage = sum(int(row["retrieval_available"]) for row in participant); directions = sum(row["H_P"] > 0 and row["H_W"] > 0 for row in family_summary); passed = len(participant) == 20 and all(effects[key]["mean"] > 0 and effects[key]["median"] > 0 and effects[key]["positive"] >= 14 and effects[key]["one_sided_exact_sign_flip"] < .05 for key in ("H_P", "H_W")) and directions >= 3
    route = {"status": "PHYSIOMOTION_SUBJECT_RETRIEVAL_HEADROOM_GO" if passed else "PHYSIOMOTION_SUBJECT_RETRIEVAL_HEADROOM_NO_GO", "gpu_training_authorized": bool(passed), "retrieval_available_participants": coverage, "availability_denominator": 20, "blocked_participants_retained_as_zero_ITT": 20 - coverage, "families_jointly_positive": directions, "sealed_opened": False, "development_only": True}
    _csv_write(_result(c) / "headroom" / "participant_effects.csv", participant); _csv_write(_result(c) / "headroom" / "participant_family_effects.csv", participant_family); _csv_write(_result(c) / "headroom" / "participant_family_method_metrics.csv", method_metrics); _csv_write(_result(c) / "headroom" / "participant_method_metrics.csv", participant_method); _csv_write(_result(c) / "headroom" / "method_summary_participant_first.csv", method_summary); _csv_write(_result(c) / "headroom" / "family_summary.csv", family_summary); _json(_result(c) / "headroom" / "result_summary.json", {"effects": effects, "method_summary": method_summary, "routing": route}); _json(_result(c) / "headroom" / "route_decision.json", route); _json(run_dir / "result_summary.json", route); return route


def stage_report(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((_result(c) / "metadata" / "dataset_audit.json").read_text()); headroom_path = _result(c) / "headroom" / "result_summary.json"; headroom = json.loads(headroom_path.read_text()) if headroom_path.exists() else None
    audit = ["# PhysioMotion Artifact data audit", "", f"Dataset: `{metadata['dataset']}`. Coverage is 30 participants × runs 01–06, with {metadata['channel_counts']} channels and {metadata['sampling_rates']} Hz metadata. Ordered layout count is {metadata['ordered_channel_layout_count']}.", "", f"The metadata-only split freezes 20 development and 10 sealed participants. Sealed IDs are {metadata['sealed']}; no sealed signal or annotation was opened.", "", f"Only {metadata['participants_with_multiple_dates_in_development']}/20 development participants have more than one acquisition date. The primary protocol is therefore accurately named `{metadata['primary_protocol_name']}`, not cross-day.", "", "Primary artifacts exclude blink, saccade, horizontal/vertical eye movement, and any combination containing them. The retained strata are head motion, chewing, tongue, swallowing, and eyebrow/facial EMG."]
    Path("reports/physiomotion_data_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    _csv_write(Path("datasets/splits/physiomotion_development_v1.csv"), _csv_read(_result(c) / "metadata" / "frozen_participant_split.csv"))
    if headroom:
        effects, route = headroom["effects"], headroom["routing"]; lines = ["# PhysioMotion subject clean-patch retrieval headroom", "", f"Decision: `{route['status']}`. Scientific denominator: 20 development participants; retrieval available for {route['retrieval_available_participants']}, with {route['blocked_participants_retained_as_zero_ITT']} blocked participants retained as zero-effect ITT units.", "", "Retrieval uses fixed K=8, two-second clean patches, z-normalized correlation on unmasked context, equal outer-participant bank quotas, and only outer-training artifact annotations for masks.", ""]
        for key in ("H_P", "H_W"):
            value = effects[key]; lines.append(f"- {key}: mean {value['mean']:+.5f}, median {value['median']:+.5f}, {value['positive']}/20, one-sided exact p={value['one_sided_exact_sign_flip']:.6f}.")
        lines += ["", "Participant-first restoration metrics among the 17 evaluable participants:"]
        for row in headroom["method_summary"]:
            lines.append(f"- {row['method']}: RRMSE {row['rrmse']:.5f}, correlation {row['correlation']:.5f}, spectral error {row['spectral_error']:.5f}, topography error {row['topography_error']:.5f}.")
        lines += ["", f"Jointly positive primary artifact-family strata: {route['families_jointly_positive']}/5.", "", "If the frozen headroom gate fails, no diffusion or deterministic GPU model is trained. Such a failure closes only this fixed retrieval representation on the development split."]
        Path("reports/physiomotion_subject_headroom.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        restoration = ["# PhysioMotion subject-aware masked restoration", "", f"Routing decision: `{route['status']}`.", "", "The preregistered no-training retrieval headroom gate failed. Therefore the retrieval-conditioned deterministic and masked clean diffusion models were not trained, no one-seed model screen was run, no extra seeds were submitted, and sealed participants remained unopened.", "", "This fail-closed result constrains the fixed K=8, two-second, z-normalized-correlation clean-patch retrieval representation on repeated-run development data. It is not a negative claim about masked diffusion or personalization families."]
        Path("reports/physiomotion_subject_restoration.md").write_text("\n".join(restoration) + "\n", encoding="utf-8")
    summary = {"metadata": metadata, "headroom": headroom, "development_only": True, "sealed_opened": False}; _json(_result(c) / "result_summary.json", summary); _json(run_dir / "result_summary.json", summary); return summary
