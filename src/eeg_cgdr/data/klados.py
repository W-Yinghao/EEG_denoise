"""Klados--Bamidis v4 paired EEG/EOG loading and evidence-backed grouping."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.io import loadmat


@dataclass(frozen=True)
class KladosRecord:
    record_id: int
    clean: np.ndarray
    contaminated: np.ndarray
    veog: np.ndarray
    heog: np.ndarray

    @property
    def samples(self) -> int:
        return int(self.clean.shape[1])


def _numeric_payload(path: Path) -> dict[str, np.ndarray]:
    payload = loadmat(path)
    return {
        str(name): np.asarray(value)
        for name, value in payload.items()
        if not name.startswith("__")
    }


def load_klados_records(config: dict[str, Any]) -> list[KladosRecord]:
    root = Path(config["data_root"])
    files = config["files"]
    contaminated = _numeric_payload(root / files["contaminated"])
    clean = _numeric_payload(root / files["clean"])
    heog = _numeric_payload(root / files["heog"])
    veog = _numeric_payload(root / files["veog"])
    expected = int(config["official_description"]["records"])
    records: list[KladosRecord] = []
    for record_id in range(1, expected + 1):
        keys = {
            "contaminated": f"sim{record_id}_con",
            "clean": f"sim{record_id}_resampled",
            "heog": f"heog_{record_id}",
            "veog": f"veog_{record_id}",
        }
        try:
            y = np.asarray(contaminated[keys["contaminated"]], dtype=np.float64)
            x = np.asarray(clean[keys["clean"]], dtype=np.float64)
            h = np.asarray(heog[keys["heog"]], dtype=np.float64).reshape(-1)
            v = np.asarray(veog[keys["veog"]], dtype=np.float64).reshape(-1)
        except KeyError as exc:
            raise ValueError(f"missing Klados variable for record {record_id}: {exc}") from exc
        if x.ndim != 2 or y.shape != x.shape or x.shape[0] != 19:
            raise ValueError(f"invalid EEG shape for record {record_id}: clean={x.shape}, y={y.shape}")
        if h.size != x.shape[1] or v.size != x.shape[1]:
            raise ValueError(f"unaligned EOG for record {record_id}")
        if not all(np.isfinite(array).all() for array in (x, y, h, v)):
            raise ValueError(f"non-finite value in record {record_id}")
        records.append(KladosRecord(record_id, x, y, v, h))
    return records


def _transfer(record: KladosRecord) -> tuple[np.ndarray, float, int, float]:
    eog = np.stack([record.veog, record.heog], axis=0)
    artifact = record.contaminated - record.clean
    gram = eog @ eog.T
    rank = int(np.linalg.matrix_rank(gram))
    condition = float(np.linalg.cond(gram))
    transfer = artifact @ eog.T @ np.linalg.pinv(gram)
    residual = artifact - transfer @ eog
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(artifact), 1e-12))
    return transfer, relative_residual, rank, condition


def _coefficient_pairs(
    transfers: list[np.ndarray], max_distance: float, minimum_gap: float
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    count = len(transfers)
    distances = np.full((count, count), np.inf, dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            denominator = 0.5 * (
                np.linalg.norm(transfers[left]) + np.linalg.norm(transfers[right])
            ) + 1e-12
            value = float(np.linalg.norm(transfers[left] - transfers[right]) / denominator)
            distances[left, right] = distances[right, left] = value
    nearest = np.argmin(distances, axis=1)
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for left in range(count):
        right = int(nearest[left])
        if left in used or right in used or int(nearest[right]) != left:
            continue
        pairs.append((min(left, right) + 1, max(left, right) + 1))
        used.update((left, right))
    pair_distances = [float(distances[left - 1, right - 1]) for left, right in pairs]
    nonpair_minima: list[float] = []
    for left, right in pairs:
        mask = np.ones(count, dtype=bool)
        mask[[left - 1, right - 1]] = False
        nonpair_minima.extend(
            [float(np.min(distances[left - 1, mask])), float(np.min(distances[right - 1, mask]))]
        )
    max_pair = max(pair_distances, default=float("inf"))
    min_nonpair = min(nonpair_minima, default=0.0)
    gap = float(min_nonpair / max(max_pair, 1e-15))
    passed = (
        len(pairs) * 2 == count
        and max_pair <= max_distance
        and gap >= minimum_gap
    )
    return sorted(pairs), {
        "mutual_nearest_pair_count": len(pairs),
        "max_pair_relative_distance": max_pair,
        "minimum_nonpair_relative_distance": min_nonpair,
        "pair_to_nonpair_gap": gap,
        "passed": passed,
    }


def _write_split(
    output: Path,
    records: list[KladosRecord],
    pairs: list[tuple[int, int]],
    config: dict[str, Any],
) -> dict[str, Any]:
    sampling_rate = int(config["official_description"]["sampling_rate"])
    required = int(config["split"]["required_calibration_seconds"] * sampling_rate)
    eligible = [
        pair
        for pair in pairs
        if records[pair[0] - 1].samples >= required and records[pair[1] - 1].samples >= required
    ]
    if not eligible:
        raise ValueError("no evidence-backed pair supports the required calibration duration")
    test_pair = eligible[0]
    remaining = [pair for pair in pairs if pair != test_pair]
    validation_count = int(config["split"]["validation_participants"])
    validation_pairs = remaining[-validation_count:]
    train_pairs = remaining[:-validation_count]
    pair_to_split = {pair: "train" for pair in train_pairs}
    pair_to_split.update({pair: "validation" for pair in validation_pairs})
    pair_to_split[test_pair] = "test"
    fields = [
        "dataset_version", "outer_fold", "split", "participant", "session", "record",
        "calibration_start", "calibration_end", "query_start", "query_end", "sampling_rate",
        "status",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for pair_number, pair in enumerate(sorted(pairs), start=1):
        participant = f"coefficient_pair_{pair_number:02d}"
        split = pair_to_split[pair]
        for within_pair, record_id in enumerate(pair):
            record = records[record_id - 1]
            duration = record.samples / sampling_rate
            row = {
                "dataset_version": config["dataset_version"],
                "outer_fold": config["split"]["outer_fold"],
                "split": split,
                "participant": participant,
                "session": f"source_record_{within_pair + 1}",
                "record": f"sim{record_id}",
                "calibration_start": "",
                "calibration_end": "",
                "query_start": "",
                "query_end": "",
                "sampling_rate": sampling_rate,
                "status": "population_source",
            }
            if split == "test" and within_pair == 0:
                row["calibration_start"] = 0.0
                row["calibration_end"] = min(30.0, duration)
                row["status"] = "held_out_calibration"
            elif split == "test":
                row["query_start"] = 0.0
                row["query_end"] = duration
                row["status"] = "held_out_query"
            rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "train_participants": len(train_pairs),
        "validation_participants": len(validation_pairs),
        "test_participants": 1,
        "held_out_pair": list(test_pair),
        "rows": len(rows),
    }


def analyze_klados_metadata(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    records = load_klados_records(config)
    transfers: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    sampling_rate = int(config["official_description"]["sampling_rate"])
    for record in records:
        transfer, residual, eog_rank, condition = _transfer(record)
        transfers.append(transfer)
        diagnostics.append(
            {
                "record": f"sim{record.record_id}",
                "samples": record.samples,
                "duration_seconds": record.samples / sampling_rate,
                "eog_rank": eog_rank,
                "eog_condition_number": condition,
                "mixture_relative_residual": residual,
                "clean_scale_median_abs": float(np.median(np.abs(record.clean))),
                "eog_scale_median_abs": float(
                    np.median(np.abs(np.concatenate([record.veog, record.heog])))
                ),
            }
        )
    evidence = config["pair_evidence"]
    pairs, pair_diagnostics = _coefficient_pairs(
        transfers,
        float(evidence["max_relative_coefficient_distance"]),
        float(evidence["minimum_pair_to_nonpair_gap"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split_manifest.csv"
    if pair_diagnostics["passed"]:
        split_summary = _write_split(split_path, records, pairs, config)
        status = "paired_provenance_supported"
    else:
        split_summary = None
        status = "blocked_pair_mapping"
    result = {
        "status": status,
        "dataset_version": config["dataset_version"],
        "official_facts": config["official_description"],
        "record_count": len(records),
        "time_alignment_verified": True,
        "clean_target_available": True,
        "ground_truth_transfer_recoverable": max(
            diagnostic["mixture_relative_residual"] for diagnostic in diagnostics
        ) < 1e-6,
        "pair_evidence": pair_diagnostics,
        "pairs": [list(pair) for pair in pairs],
        "split": split_summary,
        "records": diagnostics,
        "unknowns": [
            "physical signal units are not stated in the official article",
            "derived coefficient-pair pseudonyms are not original participant identifiers",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if split_summary is not None:
        canonical_split = Path("datasets/splits/klados_v4_outer_fold_00.csv")
        canonical_split.parent.mkdir(parents=True, exist_ok=True)
        canonical_split.write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")
    return result
