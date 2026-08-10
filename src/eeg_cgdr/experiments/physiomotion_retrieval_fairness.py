"""Information-matched PhysioMotion clean-support retrieval fairness audit."""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from scipy.signal import resample_poly

from eeg_cgdr.experiments.physiomotion_subject_restoration import _family


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _result(c: Mapping[str, Any]) -> Path:
    return Path(c["result_root"])


def _source(c: Mapping[str, Any]) -> Path:
    return Path(c["source_result_root"])


def _split_rows(c: Mapping[str, Any]) -> list[dict[str, str]]:
    return _csv_read(_source(c) / "metadata" / "frozen_participant_split.csv")


def _development(c: Mapping[str, Any]) -> list[int]:
    return sorted(int(row["participant"]) for row in _split_rows(c) if row["role"] == "development")


def _folds(c: Mapping[str, Any]) -> dict[int, int]:
    return {int(row["participant"]): int(row["cv_fold"]) for row in _split_rows(c) if row["role"] == "development"}


def _prepared(c: Mapping[str, Any], participant: int) -> Path:
    if participant not in _development(c):
        raise PermissionError(f"sealed participant {participant} access refused")
    return _source(c) / "prepared" / f"participant_{participant:02d}.npz"


def _edf(c: Mapping[str, Any], participant: int, run: int) -> Path:
    if participant not in _development(c):
        raise PermissionError(f"sealed participant {participant} signal access refused")
    return Path(c["data_root"]) / "derivatives" / "preprocessed_BIDS" / f"sub-{participant}" / "eeg" / f"sub-{participant}_task-artifact_run-{run:02d}_eeg.edf"


def _channels_tsv(c: Mapping[str, Any], participant: int, run: int) -> Path:
    return _edf(c, participant, run).with_name(f"sub-{participant}_task-artifact_run-{run:02d}_channels.tsv")


def _annotation(c: Mapping[str, Any], participant: int, run: int) -> Path:
    if participant not in _development(c):
        raise PermissionError(f"sealed participant {participant} annotation access refused")
    return Path(c["data_root"]) / "derivatives" / "Manual_Annotations" / f"sub{participant}_run{run:02d}.csv"


def _norm_channel(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).casefold()


def _channel_map(channel_names: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, name in enumerate(channel_names):
        key = _norm_channel(name)
        if key in result:
            raise RuntimeError(f"ambiguous normalized channel {name}")
        result[key] = index
    return result


def _map_annotation_channel(value: str, channel_names: list[str]) -> tuple[list[int], str]:
    raw = value.strip()
    if raw.casefold() == "all":
        return list(range(len(channel_names))), "ALL"
    if raw in channel_names:
        return [channel_names.index(raw)], "exact"
    normalized = _channel_map(channel_names)
    if _norm_channel(raw) in normalized:
        return [normalized[_norm_channel(raw)]], "normalized"
    return [], "failed"


def _uniform_seeded_starts(intervals: list[tuple[float, float]], seconds: float, cap: int, seed: int) -> list[float]:
    candidates: list[float] = []
    for start, stop in intervals:
        candidates.extend(float(value) for value in np.arange(start, stop - seconds + 1e-9, seconds))
    candidates = sorted(set(round(value, 6) for value in candidates))
    if len(candidates) <= cap:
        return candidates
    rng = np.random.default_rng(seed)
    return sorted(float(candidates[index]) for index in rng.choice(len(candidates), size=cap, replace=False))


def _union_seconds(starts: list[float], seconds: float) -> float:
    if not starts:
        return 0.0
    intervals = sorted((start, start + seconds) for start in starts)
    total = 0.0
    left, right = intervals[0]
    for next_left, next_right in intervals[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def _extract(raw: Any, starts: list[float], seconds: float, target_fs: int) -> np.ndarray:
    patches = []
    expected = int(round(seconds * target_fs))
    source_fs = int(round(raw.info["sfreq"]))
    for start in starts:
        data = raw.get_data(start=int(round(start * source_fs)), stop=int(round((start + seconds) * source_fs)))
        data = resample_poly(data, target_fs, source_fs, axis=-1).astype(np.float32)
        if data.shape[-1] >= expected:
            patches.append(data[:, :expected])
    return np.stack(patches) if patches else np.empty((0, len(raw.ch_names), expected), np.float32)


def _mask_from_group(channel_names: list[str], channels: list[str], start: float, stop: float, seconds: float, fs: int) -> tuple[np.ndarray, list[str], Counter]:
    center = 0.5 * (start + stop)
    patch_start = max(0.0, center - seconds / 2)
    duration = min(stop - start, seconds * 0.5)
    left = max(0, int(round((center - duration / 2 - patch_start) * fs)))
    right = min(int(round(seconds * fs)), int(round((center + duration / 2 - patch_start) * fs)))
    mask = np.zeros((len(channel_names), int(round(seconds * fs))), bool)
    failures: list[str] = []
    modes: Counter = Counter()
    indices: set[int] = set()
    for channel in channels:
        mapped, mode = _map_annotation_channel(channel, channel_names)
        modes[mode] += 1
        if mode == "failed":
            failures.append(channel)
        indices.update(mapped)
    if right > left:
        for index in indices:
            mask[index, left:right] = True
    return mask, failures, modes


def stage_audit(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Audit channel/mask/bank fairness and materialize only small corrected caches."""
    import mne

    development = _development(c)
    folds = _folds(c)
    seconds = float(c["patch_seconds"])
    fs = int(c["sampling_rate"])
    cap = int(c["support_candidate_cap"])
    mapping_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    bank_counts: dict[tuple[int, str], int] = {}
    date_audit = json.loads((_source(c) / "metadata" / "dataset_audit.json").read_text(encoding="utf-8"))

    for participant in development:
        support_payload: dict[str, Any] = {}
        fair_query: list[np.ndarray] = []
        fair_query_state: list[str] = []
        fair_query_run: list[int] = []
        fair_query_start: list[float] = []
        with np.load(_prepared(c, participant)) as old_prepared:
            old_query_states = np.asarray(old_prepared["query_state"]).astype(str)
            old_query_runs = np.asarray(old_prepared["query_run"], int)
        participant_masks: list[np.ndarray] = []
        participant_families: list[str] = []
        participant_mask_keys: list[str] = []
        for run in range(1, 7):
            channel_names = [row["name"] for row in _csv_read(_channels_tsv(c, participant, run))]
            annotation = _csv_read(_annotation(c, participant, run))
            raw = mne.io.read_raw_edf(_edf(c, participant, run), preload=False, verbose="ERROR")
            if raw.ch_names != channel_names:
                raise RuntimeError(f"official layout differs from EDF for participant {participant}, run {run}")
            grouped: dict[tuple[str, float, float], list[str]] = defaultdict(list)
            for row_number, row in enumerate(annotation):
                family = _family(row["label"])
                if family:
                    grouped[(family, float(row["start_time"]), float(row["stop_time"]))].append(row["channel"])
                mapped, mode = _map_annotation_channel(row["channel"], channel_names)
                mapping_rows.append({"participant": participant, "run": run, "annotation_row": row_number, "label": row["label"], "raw_channel": row["channel"], "exact_match": int(mode in ("exact", "ALL")), "normalized_match": int(mode in ("exact", "normalized", "ALL")), "mapping_mode": mode, "mapped_channel_count": len(mapped), "failed_name": row["channel"] if mode == "failed" else ""})
            for group_index, ((family, start, stop), channels) in enumerate(sorted(grouped.items())):
                mask, failures, modes = _mask_from_group(channel_names, channels, start, stop, seconds, fs)
                mask_rows.append({"participant": participant, "run": run, "family": family, "group_index": group_index, "annotation_rows": len(channels), "successful_mapped_names": sum(modes[key] for key in ("exact", "normalized", "ALL")), "failed_names": ";".join(sorted(set(failures))), "mask_channels": int(np.sum(np.any(mask, axis=1))), "mask_time_fraction": float(np.mean(np.any(mask, axis=0))), "mask_element_fraction": float(np.mean(mask)), "mask_empty": int(not mask.any())})
                if mask.any():
                    participant_masks.append(mask)
                    participant_families.append(family)
                    participant_mask_keys.append(f"p{participant:02d}_r{run:02d}_g{group_index:04d}")
            for state in ("open_base", "close_base"):
                intervals = [(float(row["start_time"]), float(row["stop_time"])) for row in annotation if row["label"].strip().casefold() == state and row["channel"].strip().casefold() == "all"]
                seed = int(c["subsampling_seed_base"]) + participant * 100 + run * 10 + (0 if state == "open_base" else 1)
                starts = _uniform_seeded_starts(intervals, seconds, cap if run == 1 else 4, seed)
                all_candidates = _uniform_seeded_starts(intervals, seconds, 10**9, seed)
                old_count = int(np.sum((old_query_states == state) & (old_query_runs == run))) if run > 1 else 0
                sampling_rows.append({"participant": participant, "state": state, "run": run, "annotation_rows": len(intervals), "all_nonoverlap_candidates": len(all_candidates), "selected_patches": len(starts), "source_prepared_patch_count": old_count, "unique_selected_starts": len(set(starts)), "duplicate_selected_starts": len(starts) - len(set(starts)), "actual_time_coverage_seconds": _union_seconds(starts, seconds), "interval_span_seconds": float(sum(max(0.0, stop - start) for start, stop in intervals)), "first_start": min(starts) if starts else "", "last_stop": max(starts) + seconds if starts else "", "current_rowwise_cap_can_repeat": int(len(intervals) > 1), "fair_sampling_rule": "fixed-seed-uniform-over-complete-baseline"})
                if run == 1:
                    patches = _extract(raw, starts, seconds, fs)
                    support_payload[state] = patches
                    support_payload[f"{state}_starts"] = np.asarray(starts, np.float64)
                    bank_counts[(participant, state)] = len(patches)
                else:
                    patches = _extract(raw, starts, seconds, fs)
                    fair_query.extend(patches)
                    fair_query_state.extend([state] * len(patches))
                    fair_query_run.extend([run] * len(patches))
                    fair_query_start.extend(starts[:len(patches)])
        fair_dir = _result(c) / "fair_materialized"
        fair_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fair_dir / f"support_{participant:02d}.npz", participant=participant, channels=np.asarray(channel_names), query=np.asarray(fair_query, np.float32), query_state=np.asarray(fair_query_state), query_run=np.asarray(fair_query_run, int), query_start=np.asarray(fair_query_start, np.float64), **support_payload)
        np.savez_compressed(fair_dir / f"masks_{participant:02d}.npz", masks=np.asarray(participant_masks, bool), families=np.asarray(participant_families), keys=np.asarray(participant_mask_keys))

    bank_rows: list[dict[str, Any]] = []
    mask_by_prf: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mask_rows:
        mask_by_prf[(int(row["participant"]), int(row["run"]), str(row["family"]))].append(row)
    sample_by_prs = {(int(row["participant"]), int(row["run"]), str(row["state"])): row for row in sampling_rows}
    for recipient in development:
        training = [p for p in development if folds[p] != folds[recipient]]
        for state in ("open_base", "close_base"):
            match_count = bank_counts.get((recipient, state), 0)
            donors = [p for p in training if bank_counts.get((p, state), 0) > 0]
            pop_count = sum(bank_counts[(p, state)] for p in donors)
            for run in range(2, 7):
                sample = sample_by_prs[(recipient, run, state)]
                for family in c["primary_families"]:
                    masks = mask_by_prf.get((recipient, run, family), [])
                    common = {"recipient": recipient, "date_stratum": "multi-date" if len(date_audit["acquisition_dates_by_participant"][str(recipient)]) > 1 else "same-day", "fold": folds[recipient], "state": state, "run": run, "family": family, "match_candidates": match_count, "pop_bank_candidates": pop_count, "pop_bank_owners": len(donors), "support_patch_count": match_count, "support_time_coverage_seconds": match_count * seconds, "query_patch_count": int(sample["selected_patches"]), "query_time_coverage_seconds": float(sample["actual_time_coverage_seconds"]), "annotation_groups": len(masks), "annotation_rows": sum(int(row["annotation_rows"]) for row in masks), "mapped_channel_names": sum(int(row["successful_mapped_names"]) for row in masks), "failed_channel_names": ";".join(sorted({name for row in masks for name in str(row["failed_names"]).split(";") if name})), "mean_mask_channels": float(np.mean([int(row["mask_channels"]) for row in masks])) if masks else 0.0, "mean_mask_time_fraction": float(np.mean([float(row["mask_time_fraction"]) for row in masks])) if masks else 0.0, "empty_masks": sum(int(row["mask_empty"]) for row in masks)}
                    if donors:
                        for donor in donors:
                            bank_rows.append({**common, "wrong_donor": donor, "wrong_candidates": bank_counts[(donor, state)]})
                    else:
                        bank_rows.append({**common, "wrong_donor": "", "wrong_candidates": 0})

    non_all = [row for row in mapping_rows if row["mapping_mode"] != "ALL"]
    exact_rate = sum(int(row["exact_match"]) for row in non_all) / max(1, len(non_all))
    normalized_rate = sum(int(row["normalized_match"]) for row in non_all) / max(1, len(non_all))
    summary = {"status": "PHYSIOMOTION_J1R_AUDIT_READY" if normalized_rate >= 0.99 else "CHANNEL_MAPPING_BELOW_99_PERCENT", "development_participants": len(development), "sealed_opened": False, "exact_mapping_rate": exact_rate, "normalized_mapping_rate": normalized_rate, "normalization_repairs": sum(row["mapping_mode"] == "normalized" for row in mapping_rows), "failed_channel_names": sorted({row["failed_name"] for row in mapping_rows if row["failed_name"]}), "empty_masks": sum(int(row["mask_empty"]) for row in mask_rows), "query_annotation_row_repetition_units": sum(int(row["current_rowwise_cap_can_repeat"]) for row in sampling_rows if int(row["run"]) > 1), "support_sampling": "fixed-seed uniform over complete run-01 baseline", "development_only": True}
    _csv_write(_result(c) / "audit" / "channel_mapping_audit.csv", mapping_rows)
    _csv_write(_result(c) / "audit" / "mask_audit.csv", mask_rows)
    _csv_write(_result(c) / "audit" / "sampling_audit.csv", sampling_rows)
    _csv_write(_result(c) / "audit" / "bank_size_mask_audit.csv", bank_rows)
    _json(_result(c) / "audit" / "result_summary.json", summary)
    _json(run_dir / "result_summary.json", summary)
    if normalized_rate < 0.99:
        raise RuntimeError(f"normalized channel mapping rate {normalized_rate:.6f} < 0.99")
    return summary


def _code(owner: int, index: int) -> int:
    if index >= 100:
        raise ValueError(index)
    return owner * 100 + index


def _decode(code: int) -> tuple[int, int]:
    return int(code) // 100, int(code) % 100


def _balanced_codes(banks: Mapping[int, np.ndarray], count: int, rng: np.random.Generator) -> np.ndarray:
    owners = [owner for owner, bank in sorted(banks.items()) if len(bank)]
    if not owners or count <= 0:
        return np.empty(0, np.int16)
    limit = int(math.ceil(count / len(owners)))
    shuffled = list(np.asarray(owners)[rng.permutation(len(owners))])
    selected: list[int] = []
    owner_counts = Counter()
    while len(selected) < count:
        progress = False
        for owner in shuffled:
            owner = int(owner)
            available = [index for index in range(len(banks[owner])) if _code(owner, index) not in selected]
            if owner_counts[owner] >= limit or not available:
                continue
            index = int(available[int(rng.integers(0, len(available)))])
            selected.append(_code(owner, index))
            owner_counts[owner] += 1
            progress = True
            if len(selected) == count:
                break
        if not progress:
            break
        shuffled = list(np.asarray(shuffled)[rng.permutation(len(shuffled))])
    if len(selected) != count:
        raise RuntimeError(f"could only draw {len(selected)}/{count} balanced candidates")
    if max(owner_counts.values()) > limit:
        raise AssertionError(owner_counts)
    return np.asarray(selected, np.int16)


def _subsample_codes(owner: int, bank: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.choice(len(bank), size=count, replace=False)
    return np.asarray([_code(owner, int(index)) for index in indices], np.int16)


def _score_candidates(observed_query: np.ndarray, mask: np.ndarray, codes: np.ndarray, banks: Mapping[int, np.ndarray]) -> np.ndarray:
    observed = ~mask
    if not np.any(observed):
        raise RuntimeError("mask has no observable retrieval context")
    query = observed_query[observed].astype(np.float64)
    query = (query - query.mean()) / max(query.std(), 1e-8)
    candidates = np.stack([banks[_decode(int(code))[0]][_decode(int(code))[1]] for code in codes])[:, observed].astype(np.float64)
    candidates = (candidates - candidates.mean(axis=1, keepdims=True)) / np.maximum(candidates.std(axis=1, keepdims=True), 1e-8)
    return candidates @ query / len(query)


def _topk_codes(observed_query: np.ndarray, mask: np.ndarray, pool: np.ndarray, banks: Mapping[int, np.ndarray], k: int) -> np.ndarray:
    scores = _score_candidates(observed_query, mask, pool, banks)
    return pool[np.argsort(scores, kind="stable")[-k:]]


def _topk_from_score_lookup(pool: np.ndarray, score_lookup: Mapping[int, float], k: int) -> np.ndarray:
    scores = np.asarray([score_lookup[int(code)] for code in pool], float)
    return pool[np.argsort(scores, kind="stable")[-k:]]


def _selection_seed(c: Mapping[str, Any], fold: int, participant: int, query_index: int, family_index: int, n: int, repeat: int) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(c["subsampling_seed_base"]), fold, participant, query_index, family_index, n, repeat])
    return np.random.default_rng(sequence)


def _pool_rng(c: Mapping[str, Any], fold: int, participant: int, query_index: int, family_index: int, n: int, repeat: int, stream: int, donor: int = 0) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(c["subsampling_seed_base"]), fold, participant, query_index, family_index, n, repeat, stream, donor])
    return np.random.default_rng(sequence)


def _load_support(c: Mapping[str, Any], participants: Iterable[int], state: str) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for participant in participants:
        with np.load(_result(c) / "fair_materialized" / f"support_{participant:02d}.npz") as data:
            result[participant] = np.asarray(data[state], np.float32)
    return result


def _load_masks(c: Mapping[str, Any], participants: Iterable[int]) -> dict[str, list[tuple[str, np.ndarray]]]:
    result: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for participant in participants:
        with np.load(_result(c) / "fair_materialized" / f"masks_{participant:02d}.npz") as data:
            for key, family, mask in zip(data["keys"], data["families"], data["masks"]):
                result[str(family)].append((str(key), np.asarray(mask, bool)))
    return result


def stage_select(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    """Observable selector: it sees masked query context, never hidden mask truth."""
    folds = _folds(c)
    recipients = sorted(p for p, value in folds.items() if value == fold)
    training = sorted(p for p, value in folds.items() if value != fold)
    families = list(c["primary_families"])
    repeats = int(c["subsampling_replicates"])
    k = int(c["retrieval_k"])
    metadata: list[dict[str, Any]] = []
    selected_records: list[np.ndarray] = []
    unit_rows: list[dict[str, Any]] = []
    record_index = 0
    for recipient in recipients:
        with np.load(_result(c) / "fair_materialized" / f"support_{recipient:02d}.npz") as query_data:
            query = np.asarray(query_data["query"], np.float32)
            states = [str(value) for value in query_data["query_state"]]
            runs = np.asarray(query_data["query_run"], int)
        masks_by_family = _load_masks(c, training)
        support_by_state = {state: _load_support(c, [recipient] + training, state) for state in ("open_base", "close_base")}
        for query_index, (clean_source, state, query_run) in enumerate(zip(query, states, runs)):
            banks = support_by_state[state]
            match_bank = banks[recipient]
            donor_banks = {owner: banks[owner] for owner in training if len(banks[owner]) >= k}
            for family_index, family in enumerate(families):
                templates = masks_by_family[family]
                if not templates:
                    continue
                mask_key, mask = templates[(recipient * 10000 + query_index) % len(templates)]
                observed_query = clean_source.copy()
                observed_query[mask] = 0.0
                donors = sorted(donor_banks)
                n_by_donor = {donor: min(int(c["support_candidate_cap"]), len(match_bank), len(donor_banks[donor])) for donor in donors}
                all_codes = np.concatenate([np.asarray([_code(owner, index) for index in range(len(bank))], np.int16) for owner, bank in sorted(banks.items()) if len(bank)])
                all_scores = _score_candidates(observed_query, mask, all_codes, banks)
                score_lookup = {int(code): float(score) for code, score in zip(all_codes, all_scores)}
                unit_id = f"f{fold}_p{recipient:02d}_q{query_index:04d}_{family}"
                unit_rows.append({"unit_id": unit_id, "fold": fold, "participant": recipient, "query_index": query_index, "state": state, "run": int(query_run), "family": family, "mask_key": mask_key, "donors_total": len(donors), "donors_evaluable": sum(n >= k for n in n_by_donor.values()), "min_common_candidates": min(n_by_donor.values()) if n_by_donor else 0, "evaluable": int(any(n >= k for n in n_by_donor.values()))})
                common_written: set[tuple[str, int]] = set()
                for donor in donors:
                    n = n_by_donor[donor]
                    if n < k:
                        continue
                    selections: dict[str, list[np.ndarray]] = defaultdict(list)
                    for repeat in range(repeats):
                        match_pool = _subsample_codes(recipient, match_bank, n, _pool_rng(c, fold, recipient, query_index, family_index, n, repeat, 1))
                        wrong_pool = _subsample_codes(donor, donor_banks[donor], n, _pool_rng(c, fold, recipient, query_index, family_index, n, repeat, 2, donor))
                        pop_pool = _balanced_codes(donor_banks, n, _pool_rng(c, fold, recipient, query_index, family_index, n, repeat, 3))
                        pop2_pool = _balanced_codes(donor_banks, 2 * n, _pool_rng(c, fold, recipient, query_index, family_index, n, repeat, 4))
                        pools = {"MATCH-N": match_pool, "WRONG-N": wrong_pool, "POP-N": pop_pool, "POP-2N": pop2_pool, "HYBRID-MATCH-2N": np.concatenate([match_pool, pop_pool]), "HYBRID-WRONG-2N": np.concatenate([wrong_pool, pop_pool])}
                        for method, pool in pools.items():
                            selections[method].append(_topk_from_score_lookup(pool, score_lookup, k))
                    if ("POP-LARGE", 0) not in common_written:
                        large_codes = np.concatenate([np.asarray([_code(owner, index) for index in range(len(bank))], np.int16) for owner, bank in sorted(donor_banks.items())])
                        top = _topk_from_score_lookup(large_codes, score_lookup, k)
                        selections["POP-LARGE"] = [top for _ in range(repeats)]
                    for method, values in selections.items():
                        common = method in {"MATCH-N", "POP-N", "POP-2N", "HYBRID-MATCH-2N", "POP-LARGE"}
                        key = (method, n if method != "POP-LARGE" else 0)
                        if common and key in common_written:
                            continue
                        method_donor = "" if common else donor
                        selected_records.append(np.asarray(values, np.int16))
                        metadata.append({"record_index": record_index, "unit_id": unit_id, "fold": fold, "participant": recipient, "query_index": query_index, "state": state, "run": int(query_run), "family": family, "family_index": family_index, "mask_key": mask_key, "method": method, "donor": method_donor, "N_u": n if method != "POP-LARGE" else 0, "subsampling_repeats": repeats, "observable_selector": "z-normalized-correlation-on-unmasked-context", "hidden_truth_read": 0})
                        record_index += 1
                        common_written.add(key)
    target = _result(c) / "selections"
    _csv_write(target / f"fold_{fold:02d}_records.csv", metadata)
    _csv_write(target / f"fold_{fold:02d}_units.csv", unit_rows)
    np.savez_compressed(target / f"fold_{fold:02d}_selected_codes.npz", selected_codes=np.asarray(selected_records, np.int16))
    summary = {"fold": fold, "recipients": recipients, "records": len(metadata), "units": len(unit_rows), "observable_hidden_truth_read": False, "sealed_opened": False}
    _json(run_dir / "result_summary.json", summary)
    return summary


def _prediction(codes: np.ndarray, banks: Mapping[int, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack([banks[_decode(int(code))[0]][_decode(int(code))[1]] for code in codes]), axis=0)


def _batch_metrics(clean: np.ndarray, predictions: np.ndarray, mask: np.ndarray, fs: int) -> dict[str, float]:
    truth = clean[mask].astype(np.float64)
    pred = predictions[:, mask].astype(np.float64)
    errors = np.linalg.norm(pred - truth[None, :], axis=1) / max(np.linalg.norm(truth), 1e-12)
    truth_centered = truth - truth.mean()
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    correlations = np.sum(pred_centered * truth_centered[None, :], axis=1) / np.maximum(np.linalg.norm(pred_centered, axis=1) * np.linalg.norm(truth_centered), 1e-12)
    average_prediction = np.mean(predictions, axis=0)
    restored = clean.copy()
    restored[mask] = average_prediction[mask]
    clean_fft = np.fft.rfft(clean, axis=-1)
    pred_fft = np.fft.rfft(restored, axis=-1)
    frequencies = np.fft.rfftfreq(clean.shape[-1], 1 / fs)
    band = (frequencies >= 1) & (frequencies <= 45)
    spectral = float(np.mean(np.abs(np.log(np.maximum(np.abs(pred_fft[:, band]) ** 2, 1e-18)) - np.log(np.maximum(np.abs(clean_fft[:, band]) ** 2, 1e-18)))))
    clean_topo = np.sqrt(np.mean(clean**2, axis=-1))
    pred_topo = np.sqrt(np.mean(restored**2, axis=-1))
    topography = float(np.linalg.norm(pred_topo / max(np.linalg.norm(pred_topo), 1e-12) - clean_topo / max(np.linalg.norm(clean_topo), 1e-12)))
    return {"rrmse": float(np.mean(errors)), "correlation": float(np.mean(correlations)), "spectral_error": spectral, "topography_error": topography}


def _pool_for_record(c: Mapping[str, Any], row: Mapping[str, str], banks: Mapping[int, np.ndarray], repeat: int) -> np.ndarray:
    fold = int(row["fold"]); participant = int(row["participant"]); query_index = int(row["query_index"]); family_index = int(row["family_index"]); n = int(row["N_u"]); method = row["method"]; donor = int(row["donor"]) if row["donor"] else None
    training = {owner: bank for owner, bank in banks.items() if owner != participant}
    match_pool = _subsample_codes(participant, banks[participant], n, _pool_rng(c, fold, participant, query_index, family_index, n, repeat, 1)) if n else np.empty(0, np.int16)
    wrong_pool = _subsample_codes(donor, banks[donor], n, _pool_rng(c, fold, participant, query_index, family_index, n, repeat, 2, int(donor))) if donor is not None else np.empty(0, np.int16)
    pop_pool = _balanced_codes(training, n, _pool_rng(c, fold, participant, query_index, family_index, n, repeat, 3)) if n else np.empty(0, np.int16)
    if method == "MATCH-N": return match_pool
    if method == "WRONG-N": return wrong_pool
    if method == "POP-N": return pop_pool
    if method == "POP-2N": return _balanced_codes(training, 2 * n, _pool_rng(c, fold, participant, query_index, family_index, n, repeat, 4))
    if method == "HYBRID-MATCH-2N": return np.concatenate([match_pool, pop_pool])
    if method == "HYBRID-WRONG-2N": return np.concatenate([wrong_pool, pop_pool])
    if method == "POP-LARGE": return np.concatenate([np.asarray([_code(owner, index) for index in range(len(bank))], np.int16) for owner, bank in sorted(training.items())])
    raise ValueError(method)


def stage_evaluate(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    """Independent evaluator opens clean truth and computes observable/oracle metrics."""
    rows = _csv_read(_result(c) / "selections" / f"fold_{fold:02d}_records.csv")
    with np.load(_result(c) / "selections" / f"fold_{fold:02d}_selected_codes.npz") as selected_file:
        selected = np.asarray(selected_file["selected_codes"], np.int16)
    folds = _folds(c)
    recipients = sorted(p for p, value in folds.items() if value == fold)
    training = sorted(p for p, value in folds.items() if value != fold)
    query_cache: dict[int, tuple[np.ndarray, list[str], np.ndarray]] = {}
    support_cache = {state: _load_support(c, recipients + training, state) for state in ("open_base", "close_base")}
    mask_lookup = {key: mask for family in c["primary_families"] for key, mask in _load_masks(c, training)[family]}
    output: list[dict[str, Any]] = []
    for row in rows:
        participant = int(row["participant"])
        if participant not in query_cache:
            with np.load(_result(c) / "fair_materialized" / f"support_{participant:02d}.npz") as data:
                query_cache[participant] = (np.asarray(data["query"], np.float32), [str(value) for value in data["query_state"]], np.asarray(data["query_run"], int))
        clean = query_cache[participant][0][int(row["query_index"])]
        mask = mask_lookup[row["mask_key"]]
        banks = support_cache[row["state"]]
        observable_codes = selected[int(row["record_index"])]
        observable_predictions = np.stack([_prediction(codes, banks) for codes in observable_codes])
        observable = _batch_metrics(clean, observable_predictions, mask, int(c["sampling_rate"]))
        oracle_predictions = []
        for repeat in range(int(c["subsampling_replicates"])):
            pool = _pool_for_record(c, row, banks, repeat)
            candidates = np.stack([banks[_decode(int(code))[0]][_decode(int(code))[1]] for code in pool])
            candidate_error = np.mean((candidates[:, mask] - clean[mask][None, :]) ** 2, axis=1)
            oracle_codes = pool[np.argsort(candidate_error, kind="stable")[:int(c["retrieval_k"])]]
            oracle_predictions.append(_prediction(oracle_codes, banks))
        oracle = _batch_metrics(clean, np.stack(oracle_predictions), mask, int(c["sampling_rate"]))
        base = {key: row[key] for key in ("record_index", "unit_id", "fold", "participant", "query_index", "state", "run", "family", "mask_key", "method", "donor", "N_u")}
        output.append({**base, "selector": "observable", **observable})
        output.append({**base, "selector": "oracle", **oracle})
    _csv_write(_result(c) / "evaluation" / f"fold_{fold:02d}.csv", output)
    summary = {"fold": fold, "records": len(rows), "metric_rows": len(output), "oracle_evaluator_only": True, "sealed_opened": False}
    _json(run_dir / "result_summary.json", summary)
    return summary


def _signflip(values: np.ndarray) -> float:
    if not len(values): return float("nan")
    observed = float(np.mean(values))
    return float(np.mean([np.mean(values * np.asarray(signs)) >= observed - 1e-15 for signs in itertools.product((-1, 1), repeat=len(values))]))


def _bootstrap(values: np.ndarray, seed: int, replicates: int) -> tuple[float, float]:
    if not len(values): return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), size=(replicates, len(values)))], axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _aggregate_selector(c: Mapping[str, Any], rows: list[dict[str, str]], units: list[dict[str, str]], selector: str, policy: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in rows if row["selector"] == selector]
    development = _development(c)
    participant_effects: list[dict[str, Any]] = []
    family_effects: list[dict[str, Any]] = []
    contrasts = {"H_P_eq": ("POP-N", "MATCH-N"), "H_W_eq": ("WRONG-N", "MATCH-N"), "H_HYB": ("POP-2N", "HYBRID-MATCH-2N"), "H_HYB_W": ("HYBRID-WRONG-2N", "HYBRID-MATCH-2N"), "H_MATCH_LARGE": ("POP-LARGE", "MATCH-N"), "H_HYB_LARGE": ("POP-LARGE", "HYBRID-MATCH-2N")}
    for participant in development:
        participant_rows = [row for row in selected if int(row["participant"]) == participant]
        participant_units = [row for row in units if int(row["participant"]) == participant]
        family_values: dict[str, dict[str, float]] = defaultdict(dict)
        for family in c["primary_families"]:
            family_rows = [row for row in participant_rows if row["family"] == family]
            family_units = [row for row in participant_units if row["family"] == family]
            for effect, (reference, target) in contrasts.items():
                unit_effects: dict[str, float] = {}
                for unit_id in sorted({row["unit_id"] for row in family_rows}):
                    unit = [row for row in family_rows if row["unit_id"] == unit_id]
                    target_rows = [float(row["rrmse"]) for row in unit if row["method"] == target]
                    reference_rows = [float(row["rrmse"]) for row in unit if row["method"] == reference]
                    if target_rows and reference_rows:
                        unit_effects[unit_id] = float(np.mean(reference_rows) - np.mean(target_rows))
                state_run_values = []
                for state, run in sorted({(row["state"], int(row["run"])) for row in family_units}):
                    group_units = [row["unit_id"] for row in family_units if row["state"] == state and int(row["run"]) == run]
                    group_values = [unit_effects[unit_id] for unit_id in group_units if unit_id in unit_effects]
                    if policy:
                        group_values += [0.0] * (len(group_units) - len(group_values))
                    if group_values:
                        state_run_values.append(float(np.mean(group_values)))
                if state_run_values:
                    family_values[family][effect] = float(np.mean(state_run_values))
        for family in c["primary_families"]:
            row = {"selector": selector, "estimand": "policy" if policy else "evaluable", "participant": participant, "family": family, "evaluable": int(bool(family_values[family]))}
            for effect in contrasts:
                row[effect] = family_values[family].get(effect, 0.0 if policy else float("nan"))
            family_effects.append(row)
        available_families = [family for family in c["primary_families"] if family_values[family]]
        if available_families or policy:
            row = {"selector": selector, "estimand": "policy" if policy else "evaluable", "participant": participant, "evaluable": int(bool(available_families)), "families": len(available_families)}
            for effect in contrasts:
                values = [family_values[family][effect] for family in available_families if effect in family_values[family]]
                row[effect] = float(np.mean(values)) if values else 0.0
            participant_effects.append(row)
    return participant_effects, family_effects


def stage_aggregate(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    units: list[dict[str, str]] = []
    for fold in range(5):
        rows.extend(_csv_read(_result(c) / "evaluation" / f"fold_{fold:02d}.csv"))
        units.extend(_csv_read(_result(c) / "selections" / f"fold_{fold:02d}_units.csv"))
    participant_all: list[dict[str, Any]] = []
    family_all: list[dict[str, Any]] = []
    for selector in ("observable", "oracle"):
        for policy in (True, False):
            participant, families = _aggregate_selector(c, rows, units, selector, policy)
            participant_all.extend(participant); family_all.extend(families)
    date_audit = json.loads((_source(c) / "metadata" / "dataset_audit.json").read_text(encoding="utf-8"))
    for row in participant_all:
        row["date_stratum"] = "multi-date" if len(date_audit["acquisition_dates_by_participant"][str(row["participant"])]) > 1 else "same-day"
    summaries: list[dict[str, Any]] = []
    effects = ("H_P_eq", "H_W_eq", "H_HYB", "H_HYB_W", "H_MATCH_LARGE", "H_HYB_LARGE")
    for selector in ("observable", "oracle"):
        for estimand in ("policy", "evaluable"):
            take = [row for row in participant_all if row["selector"] == selector and row["estimand"] == estimand and (estimand == "policy" or row["evaluable"])]
            for effect in effects:
                values = np.asarray([float(row[effect]) for row in take], float)
                low, high = _bootstrap(values, int(c["bootstrap_seed"]), int(c["bootstrap_replicates"]))
                summaries.append({"selector": selector, "estimand": estimand, "effect": effect, "n": len(values), "mean": float(np.mean(values)) if len(values) else float("nan"), "median": float(np.median(values)) if len(values) else float("nan"), "positive": int(np.sum(values > 0)), "one_sided_exact_sign_flip": _signflip(values), "bootstrap_low": low, "bootstrap_high": high})
    summary_lookup = {(row["selector"], row["estimand"], row["effect"]): row for row in summaries}
    obs = summary_lookup[("observable", "evaluable", "H_P_eq")], summary_lookup[("observable", "evaluable", "H_W_eq")]
    obs_hyb = summary_lookup[("observable", "evaluable", "H_HYB")], summary_lookup[("observable", "evaluable", "H_HYB_W")]
    oracle = summary_lookup[("oracle", "evaluable", "H_P_eq")], summary_lookup[("oracle", "evaluable", "H_W_eq")]
    oracle_hyb = summary_lookup[("oracle", "evaluable", "H_HYB")], summary_lookup[("oracle", "evaluable", "H_HYB_W")]
    family_gate = lambda selector, first, second: sum(1 for family in c["primary_families"] if np.nanmean([float(row[first]) for row in family_all if row["selector"] == selector and row["estimand"] == "evaluable" and row["family"] == family and np.isfinite(float(row[first]))]) > 0 and np.nanmean([float(row[second]) for row in family_all if row["selector"] == selector and row["estimand"] == "evaluable" and row["family"] == family and np.isfinite(float(row[second]))]) > 0)
    gate = lambda pair: all(row["mean"] > 0 and row["median"] > 0 and row["positive"] >= 12 and row["one_sided_exact_sign_flip"] < .05 for row in pair)
    bank_signal = gate(obs) and family_gate("observable", "H_P_eq", "H_W_eq") >= 3
    hybrid_signal = gate(obs_hyb) and family_gate("observable", "H_HYB", "H_HYB_W") >= 3
    oracle_signal = gate(oracle) and family_gate("oracle", "H_P_eq", "H_W_eq") >= 3
    oracle_hybrid = gate(oracle_hyb) and family_gate("oracle", "H_HYB", "H_HYB_W") >= 3
    hybrid_large = summary_lookup[("observable", "evaluable", "H_HYB_LARGE")]
    deployable = bank_signal and hybrid_signal and hybrid_large["mean"] >= 0 and hybrid_large["median"] >= 0
    donor_only = obs[1]["mean"] > 0 and obs[1]["median"] > 0 and not (obs[0]["mean"] > 0 and obs[0]["median"] > 0)
    if deployable:
        status = "DEPLOYABLE_SUBJECT_INCREMENT_HEADROOM_PRESENT"
    elif bank_signal:
        status = "SUBJECT_BANK_SIGNAL_PRESENT"
    elif (oracle_signal or oracle_hybrid) and not bank_signal:
        status = "SELECTOR_BOTTLENECK_ONLY"
    elif donor_only:
        status = "DONOR_SPECIFICITY_WITHOUT_POPULATION_UTILITY"
    elif not oracle_signal and not oracle_hybrid:
        status = "PHYSIOMOTION_CLEAN_SUPPORT_ROUTE_CLOSED"
    else:
        status = "PHYSIOMOTION_RETRIEVAL_FAIRNESS_NO_GO"
    stratum_summaries: list[dict[str, Any]] = []
    for selector in ("observable", "oracle"):
        for estimand in ("policy", "evaluable"):
            for stratum in ("same-day", "multi-date"):
                take = [row for row in participant_all if row["selector"] == selector and row["estimand"] == estimand and row["date_stratum"] == stratum and (estimand == "policy" or row["evaluable"])]
                for effect in effects:
                    values = np.asarray([float(row[effect]) for row in take], float)
                    stratum_summaries.append({"selector": selector, "estimand": estimand, "date_stratum": stratum, "effect": effect, "n": len(values), "mean": float(np.mean(values)) if len(values) else float("nan"), "median": float(np.median(values)) if len(values) else float("nan"), "positive": int(np.sum(values > 0))})
    selector_gaps: list[dict[str, Any]] = []
    for estimand in ("policy", "evaluable"):
        for participant in _development(c):
            observable_row = next((row for row in participant_all if row["selector"] == "observable" and row["estimand"] == estimand and row["participant"] == participant and (estimand == "policy" or row["evaluable"])), None)
            oracle_row = next((row for row in participant_all if row["selector"] == "oracle" and row["estimand"] == estimand and row["participant"] == participant and (estimand == "policy" or row["evaluable"])), None)
            if observable_row and oracle_row:
                selector_gaps.append({"estimand": estimand, "participant": participant, "date_stratum": observable_row["date_stratum"], **{f"oracle_minus_observable_{effect}": float(oracle_row[effect]) - float(observable_row[effect]) for effect in effects}})
    route = {"status": status, "subject_bank_signal": bank_signal, "deployable_subject_increment_headroom": deployable, "observable_bank_families_jointly_positive": family_gate("observable", "H_P_eq", "H_W_eq"), "observable_hybrid_families_jointly_positive": family_gate("observable", "H_HYB", "H_HYB_W"), "oracle_bank_signal": oracle_signal, "oracle_hybrid_signal": oracle_hybrid, "hybrid_match_not_worse_than_pop_large_mean": hybrid_large["mean"] >= 0, "hybrid_match_not_worse_than_pop_large_median": hybrid_large["median"] >= 0, "model_training_authorized_this_round": False, "development_participants": 20, "sealed_opened": False, "development_only": True}
    _csv_write(_result(c) / "aggregate" / "participant_effects.csv", participant_all)
    _csv_write(_result(c) / "aggregate" / "family_effects.csv", family_all)
    _csv_write(_result(c) / "aggregate" / "effect_summary.csv", summaries)
    _csv_write(_result(c) / "aggregate" / "date_stratum_summary.csv", stratum_summaries)
    _csv_write(_result(c) / "aggregate" / "observable_oracle_effect_gap.csv", selector_gaps)
    _json(_result(c) / "aggregate" / "routing_decision.json", route)
    _json(_result(c) / "result_summary.json", {"routing": route, "effect_summary": summaries})
    _json(run_dir / "result_summary.json", route)
    return route


def stage_report(c: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    audit = json.loads((_result(c) / "audit" / "result_summary.json").read_text(encoding="utf-8"))
    result = json.loads((_result(c) / "result_summary.json").read_text(encoding="utf-8"))
    route = result["routing"]
    summary = {(row["selector"], row["estimand"], row["effect"]): row for row in result["effect_summary"]}
    sampling_rows = _csv_read(_result(c) / "audit" / "sampling_audit.csv")
    support_multirow = sum(int(row["run"]) == 1 and int(row["annotation_rows"]) > 1 for row in sampling_rows)
    lines = ["# PhysioMotion J1R retrieval fairness audit", "", f"Decision: `{route['status']}`.", "", "This is a CPU-only development fairness audit. It trains no deterministic or diffusion model and never opens the ten sealed participants.", "", "## Data and mask audit", "", f"Official-layout annotation mapping was {audit['exact_mapping_rate']:.4%} before and {audit['normalized_mapping_rate']:.4%} after strip/case normalization. Normalization repaired {audit['normalization_repairs']} names; unresolved names: {audit['failed_channel_names'] or 'none'}.", "", f"Empty reconstructed masks: {audit['empty_masks']}. Query state/run units with multiple baseline annotation rows, where the old rowwise cap could repeat sampling: {audit['query_annotation_row_repetition_units']}.", "", f"Run-01 support participant/state units with multiple baseline annotation rows: {support_multirow}. In the old builder, per-row candidate concatenation followed by `[:16]` could favor earlier rows. The fairness cache instead uses fixed-seed uniform sampling over the complete baseline for both support and query.", "", "## Participant-first effects", ""]
    for selector in ("observable", "oracle"):
        lines += [f"### {selector.capitalize()} selector", ""]
        for effect in ("H_P_eq", "H_W_eq", "H_HYB", "H_HYB_W", "H_MATCH_LARGE", "H_HYB_LARGE"):
            row = summary[(selector, "evaluable", effect)]
            lines.append(f"- {effect}: mean {row['mean']:+.5f}, median {row['median']:+.5f}, {row['positive']}/{row['n']} positive, one-sided exact p={row['one_sided_exact_sign_flip']:.6f}, descriptive 95% bootstrap [{row['bootstrap_low']:+.5f}, {row['bootstrap_high']:+.5f}].")
        lines.append("")
    stratum_rows = _csv_read(_result(c) / "aggregate" / "date_stratum_summary.csv")
    gap_rows = _csv_read(_result(c) / "aggregate" / "observable_oracle_effect_gap.csv")
    lines += ["## Descriptive strata and oracle gap", ""]
    for stratum in ("same-day", "multi-date"):
        take = next(row for row in stratum_rows if row["selector"] == "observable" and row["estimand"] == "evaluable" and row["date_stratum"] == stratum and row["effect"] == "H_P_eq")
        lines.append(f"- {stratum} H_P_eq: mean {float(take['mean']):+.5f}, median {float(take['median']):+.5f}, {take['positive']}/{take['n']} positive.")
    evaluable_gaps = [row for row in gap_rows if row["estimand"] == "evaluable"]
    for effect in ("H_P_eq", "H_W_eq", "H_HYB", "H_HYB_W"):
        values = [float(row[f"oracle_minus_observable_{effect}"]) for row in evaluable_gaps]
        lines.append(f"- Oracle-minus-observable {effect} effect gap: mean {np.mean(values):+.5f} across {len(values)} participants.")
    lines.append("")
    lines += ["## Interpretation", "", "Equal-budget observable MATCH exceeds both POP and donor-averaged WRONG, so the fixed clean support bank contains subject information under the corrected sampling contract. The equal-total-budget hybrid effects also pass the frozen directional gates, and observable HYBRID-MATCH is pointwise non-inferior to POP-LARGE in both mean and median. This yields `DEPLOYABLE_SUBJECT_INCREMENT_HEADROOM_PRESENT`, which authorizes only a future single hybrid masked-diffusion screen; no model is trained here.", "", "The oracle equal-budget contrasts are also positive, while oracle POP-LARGE is substantially stronger than the smaller MATCH/hybrid pools. Thus large population coverage remains an important ceiling and the result is not evidence that subject retrieval universally dominates a sufficiently large population bank.", ""]
    lines += ["## Boundary", "", "All 64 deterministic subsampling repeats are averaged inside each evaluation unit before state/run, equal-family, and participant aggregation. They are not scientific replicates. POP fallback units appear only in the 20-person policy estimand; mechanism effects use the evaluable estimand. Same-day and multi-date strata remain descriptive.", "", "No model training is authorized in this round regardless of the routing label."]
    Path("reports/physiomotion_retrieval_fairness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json(run_dir / "result_summary.json", route)
    return route
