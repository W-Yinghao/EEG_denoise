"""Post-output P-C oracle ceiling and deployable discrete selector.

Query EOG/clean targets are opened only here, after route outputs are frozen.
They define development labels and diagnostic ceilings, never inference
features.  Deployable predictions are leave-one-unit-out and use only query EEG,
POP--MATCH disagreement, and calibration-support summaries.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from eeg_cgdr.experiments.literature_guided_v3 import (
    CODE_ROOT,
    _annotation_opener,
    _base_config,
    _continuous,
    _klados_eval_records,
    _prepared_v3,
    _sge_matching_support,
    _training_donor_supports,
)


RESULT_ROOT = CODE_ROOT / "results/cgdr/literature_guided_exploration_v3"


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_flat(value: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return value.transpose(1, 0, 2)[:, valid].reshape(-1)


def _relative_error(output: np.ndarray, clean: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = []
    epsilon = np.finfo(np.float64).eps
    for index in range(output.shape[0]):
        keep = valid[index]
        prediction = output[index, :, keep].astype(np.float64)
        target = clean[index, :, keep].astype(np.float64)
        result.append(np.linalg.norm(prediction - target) / max(np.linalg.norm(target), epsilon))
    return np.asarray(result)


def _eeg_features(observed: np.ndarray, pop: np.ndarray, candidate: np.ndarray, support_latent: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    centered = observed.astype(np.float64) - observed.mean(axis=-1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=(1, 2)))
    fourth = np.mean(centered ** 4, axis=(1, 2)) / np.maximum(np.mean(centered ** 2, axis=(1, 2)) ** 2, epsilon)
    line = np.mean(np.abs(np.diff(centered, axis=-1)), axis=(1, 2)) / np.maximum(rms, epsilon)
    disagreement = np.sqrt(np.mean((candidate.astype(np.float64) - pop.astype(np.float64)) ** 2, axis=(1, 2))) / np.maximum(rms, epsilon)
    support_absolute = np.abs(support_latent.astype(np.float64))
    support_summary = np.asarray([
        np.quantile(support_absolute, 0.50), np.quantile(support_absolute, 0.90),
    ])
    return np.column_stack((rms, fourth, line, disagreement,
                            np.full(rms.shape, support_summary[0]),
                            np.full(rms.shape, support_summary[1])))


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, penalty: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (train_x - mean) / scale
    design = np.column_stack((np.ones(normalized.shape[0]), normalized))
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)
    return weights, mean, scale


def _predict_ridge(model: tuple[np.ndarray, np.ndarray, np.ndarray], value: np.ndarray) -> np.ndarray:
    weights, mean, scale = model
    return np.column_stack((np.ones(value.shape[0]), (value - mean) / scale)) @ weights


def _window_eog_proxy(observed: np.ndarray, output: np.ndarray, eog: np.ndarray) -> np.ndarray:
    """Independent evaluator-only absolute EEG--EOG correlation reduction."""
    windows, _, samples = observed.shape
    external = np.asarray(eog, dtype=np.float64)
    if external.ndim == 1:
        external = external[None]
    usable = min(external.shape[-1], windows * samples)
    window_count = usable // samples
    if window_count < windows:
        raise ValueError("query EOG does not span all frozen EEG windows")
    external = external[:, : windows * samples].reshape(external.shape[0], windows, samples).transpose(1, 0, 2)
    result = np.empty(windows, dtype=np.float64)
    for index in range(windows):
        def coherence_proxy(eeg: np.ndarray) -> float:
            values = []
            for eeg_channel in eeg[index]:
                for eog_channel in external[index]:
                    if np.std(eeg_channel) > 1e-8 and np.std(eog_channel) > 1e-8:
                        values.append(abs(float(np.corrcoef(eeg_channel, eog_channel)[0, 1])))
            return float(np.mean(values)) if values else 0.0
        result[index] = coherence_proxy(observed) - coherence_proxy(output)
    return result


def _load_units(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    base = _base_config(config)
    klados_arrays = RESULT_ROOT / "server_arrays/P_A_RAW_SUPPORT_TOKENS/klados/fold_00"
    klados_prepared = _prepared_v3(config, {"dataset": "klados", "fold_index": 0, "seed": 20260811})
    klados_wrong_support = _training_donor_supports(klados_prepared, minimum=3)[0][1][1]
    for unit, mechanism, _, _ in _klados_eval_records(base):
        archive = np.load(klados_arrays / f"{unit}.npz")
        pop = archive["STRONG_POP"]
        match = archive["DIFF_MATCH"]
        wrong = archive["DIFF_WRONG_1"]
        clean = mechanism.clean_windows.astype(np.float32)
        valid = mechanism.valid_time_weight.astype(bool)
        pop_loss = _relative_error(pop, clean, valid)
        match_loss = _relative_error(match, clean, valid)
        wrong_loss = _relative_error(wrong, clean, valid)
        target = (match_loss < pop_loss).astype(np.float64)
        support = np.asarray(mechanism.calibration.eog)[None]
        units.append({
            "dataset": "klados", "unit_id": unit, "exact_cell": klados_prepared.fold.layout_id,
            "observed": mechanism.observed_windows.astype(np.float32), "pop": pop, "match": match,
            "wrong": wrong, "raw": archive["RAW"], "support_latent": support,
            "wrong_support_latent": klados_wrong_support,
            "target": target, "benefit": pop_loss - match_loss,
            "wrong_target": (wrong_loss < pop_loss).astype(np.float64),
            "wrong_benefit": pop_loss - wrong_loss,
            "preservation": np.ones_like(target),
        })
    for fold in range(25):
        row = {"dataset": "sgeyesub", "fold_index": fold, "seed": 20260811}
        prepared = _prepared_v3(config, row)
        wrong_support_latent = _training_donor_supports(prepared, minimum=3)[0][1][1]
        arrays_root = RESULT_ROOT / f"server_arrays/P_A_RAW_SUPPORT_TOKENS/sgeyesub/fold_{fold:02d}"
        for unit, heldout in prepared.heldout.items():
            path = arrays_root / f"{unit.replace('/', '__')}.npz"
            if not path.is_file():
                continue
            archive = np.load(path)
            pop = archive["STRONG_POP"]
            match = archive["DIFF_MATCH"]
            wrong = archive["DIFF_WRONG_1"]
            annotated = _annotation_opener(base, prepared, unit)()
            annotations = annotated.query_annotations
            if annotations is None:
                continue
            benefit = _window_eog_proxy(heldout.query.observed, match, annotations.external_eog)
            wrong_benefit = _window_eog_proxy(heldout.query.observed, wrong, annotations.external_eog)
            pop_benefit = _window_eog_proxy(heldout.query.observed, pop, annotations.external_eog)
            relative_match_change = np.sqrt(np.mean((match - heldout.query.observed) ** 2, axis=(1, 2))) / np.maximum(
                np.sqrt(np.mean(heldout.query.observed ** 2, axis=(1, 2))), 1e-8)
            relative_pop_change = np.sqrt(np.mean((pop - heldout.query.observed) ** 2, axis=(1, 2))) / np.maximum(
                np.sqrt(np.mean(heldout.query.observed ** 2, axis=(1, 2))), 1e-8)
            preservation = (relative_match_change <= relative_pop_change + 0.02).astype(np.float64)
            target = ((benefit > pop_benefit) & (preservation > 0)).astype(np.float64)
            support = _sge_matching_support(base, prepared, fold, unit)[1]
            units.append({
                "dataset": "sgeyesub", "unit_id": unit,
                "exact_cell": f"{prepared.fold.study}|{prepared.fold.layout_id}|{prepared.fold.sampling_rate_hz:g}",
                "observed": heldout.query.observed.astype(np.float32), "pop": pop, "match": match,
                "wrong": wrong, "raw": archive["RAW"], "support_latent": support,
                "wrong_support_latent": wrong_support_latent,
                "target": target, "benefit": benefit - pop_benefit, "preservation": preservation,
                "wrong_target": (wrong_benefit > pop_benefit).astype(np.float64),
                "wrong_benefit": wrong_benefit - pop_benefit,
            })
    return units


def _oracle_ceiling(units: list[dict[str, Any]], coverage_levels: list[float]) -> list[dict[str, Any]]:
    rows = []
    for unit in units:
        benefit = np.asarray(unit["benefit"], dtype=np.float64)
        order = np.argsort(benefit)[::-1]
        for coverage in coverage_levels:
            count = int(round(coverage * benefit.size))
            chosen = order[:count]
            rows.append({
                "dataset": unit["dataset"], "unit_id": unit["unit_id"], "exact_cell": unit["exact_cell"],
                "coverage": coverage, "selected_windows": count, "window_count": benefit.size,
                "oracle_utility_vs_pop": float(benefit[chosen].sum() / max(benefit.size, 1)),
                "selected_preservation_rate": float(np.mean(np.asarray(unit["preservation"])[chosen])) if count else 1.0,
                "role": "hindsight_oracle_query_outcome_diagnostic_only",
            })
    return rows


def _deployable_cross_unit(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for test in units:
        train = [
            unit for unit in units
            if unit["dataset"] == test["dataset"]
            and unit["exact_cell"] == test["exact_cell"]
            and unit["unit_id"] != test["unit_id"]
        ]
        if not train:
            rows.append({
                "dataset": test["dataset"], "unit_id": test["unit_id"], "exact_cell": test["exact_cell"],
                "windows": len(test["target"]), "status": "blocked_singleton_exact_cell",
                "query_external_signal_in_inference": False,
            })
            continue
        policies = (
            ("global_query_eeg_gate", "match", slice(0, 3)),
            ("matching_support_gate", "match", slice(None)),
            ("wrong_support_gate", "wrong", slice(None)),
        )
        for policy, candidate_name, feature_slice in policies:
            support_field = "support_latent" if candidate_name == "match" else "wrong_support_latent"
            target_field = "target" if candidate_name == "match" else "wrong_target"
            benefit_field = "benefit" if candidate_name == "match" else "wrong_benefit"
            train_y = np.concatenate([unit[target_field] for unit in train])
            train_x = np.concatenate([
                _eeg_features(unit["observed"], unit["pop"], unit[candidate_name], unit[support_field])[:, feature_slice]
                for unit in train
            ])
            model = _fit_ridge(train_x, train_y)
            test_x_full = _eeg_features(test["observed"], test["pop"], test[candidate_name], test[support_field])
            test_x = test_x_full[:, feature_slice]
            scores = _predict_ridge(model, test_x)
            activity_floor = float(np.quantile(train_x[:, 0], 0.10))
            actions = np.where(scores >= 0.5, "MATCH", "POP")
            candidate = test[candidate_name]
            change_candidate = np.sqrt(np.mean((candidate - test["observed"]) ** 2, axis=(1, 2)))
            change_pop = np.sqrt(np.mean((test["pop"] - test["observed"]) ** 2, axis=(1, 2)))
            disagreement = np.sqrt(np.mean((candidate - test["pop"]) ** 2, axis=(1, 2)))
            abstain = (test_x_full[:, 0] <= activity_floor) & (np.minimum(change_candidate, change_pop) > disagreement)
            actions[abstain] = "IDENTITY"
            chosen = np.stack([
                candidate[index] if action == "MATCH" else test["raw"][index] if action == "IDENTITY" else test["pop"][index]
                for index, action in enumerate(actions)
            ])
            preservation = np.asarray(test["preservation"], dtype=np.float64)
            selected_preservation = np.where(actions == "MATCH", preservation, 1.0)
            rows.append({
                "dataset": test["dataset"], "unit_id": test["unit_id"], "exact_cell": test["exact_cell"],
                "policy": policy, "candidate_context": candidate_name,
                "status": "success_leave_one_unit_out_exact_cell",
                "windows": len(actions), "match_fraction": float(np.mean(actions == "MATCH")),
                "pop_fraction": float(np.mean(actions == "POP")), "identity_fraction": float(np.mean(actions == "IDENTITY")),
                "target_accuracy_diagnostic": float(np.mean((actions == "MATCH") == (np.asarray(test[target_field]) > 0))),
                "mean_outcome_utility_vs_pop": float(np.sum(np.asarray(test[benefit_field])[actions == "MATCH"]) / len(actions)),
                "selected_preservation_rate": float(np.mean(selected_preservation)),
                "output_input_rms_ratio": float(np.sqrt(np.mean(chosen.astype(np.float64) ** 2)) / max(np.sqrt(np.mean(test["observed"].astype(np.float64) ** 2)), 1e-8)),
                "query_external_signal_in_inference": False,
                "evaluation": "leave_one_source_or_stem_out_development",
            })
    return rows


def run(config_path: Path, run_dir: Path) -> Mapping[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("v3 config must be a mapping")
    units = _load_units(config)
    if len([unit for unit in units if unit["dataset"] == "klados"]) != 16:
        raise ValueError("P-C requires all 16 Klados source records")
    if len([unit for unit in units if unit["dataset"] == "sgeyesub"]) != 58:
        raise ValueError("P-C requires all 58 compatible SGE stems (59 denominator)")
    coverage = [float(value) for value in config["selective_policy"]["coverage_levels"]]
    ceiling = _oracle_ceiling(units, coverage)
    output = RESULT_ROOT / "selective_policy"
    _write_csv(output / "oracle_ceiling.csv", ceiling)
    ceiling_groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in ceiling:
        ceiling_groups[(str(row["dataset"]), float(row["coverage"]))].append(row)
    positive_ceiling = any(
        all(
            float(np.mean([entry["oracle_utility_vs_pop"] for entry in ceiling_groups[(dataset, coverage)]])) > 0
            and float(np.mean([entry["selected_preservation_rate"] for entry in ceiling_groups[(dataset, coverage)]])) >= 0.98
            for dataset in ("klados", "sgeyesub")
        )
        for coverage in (0.5, 0.8)
    )
    deployable = _deployable_cross_unit(units) if positive_ceiling else []
    _write_csv(output / "deployable_selector.csv", deployable)
    summary = {
        "status": "completed_full_development_selective_policy" if positive_ceiling else "completed_low_selective_ceiling",
        "oracle_ceiling_role": "diagnostic_query_outcome_only",
        "deployable_selector_run": bool(positive_ceiling),
        "query_external_signal_used_for_inference": False,
        "klados_source_records": 16,
        "sge_compatible_stems": 58,
        "sge_registered_denominator": 59,
        "deterministic_ensemble_uncertainty": "unavailable_in_one_seed_screen",
    }
    _write_json(output / "result_summary.json", summary); _write_json(run_dir / "result_summary.json", summary)
    return summary
