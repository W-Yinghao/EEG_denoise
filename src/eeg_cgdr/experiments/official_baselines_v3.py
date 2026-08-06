"""Bounded runtime audit for the v3 official population baselines.

This module does not silently redesign incomplete upstream methods.  It reuses
the already completed full EEGDfus benchmark, tests bundled official examples,
and emits explicit blockers when a frozen-split reconstruction is not defined
by the released source/data.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_lit_explore_v3"))
RESULT_ROOT = CODE_ROOT / "results/cgdr/literature_guided_exploration_v3/baseline_audit"
MAIN_ROOT = Path("/home/infres/yinwang/denoiseNet")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _standardize_eegoar_signal(
    signal: np.ndarray,
    channels: list[str],
    official_channels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the released 64-channel placement without its Windows-only path."""

    value = np.asarray(signal, dtype=np.float32)
    names = [str(channel).upper() for channel in channels]
    official = [str(channel).upper() for channel in official_channels]
    if value.ndim != 2 or value.shape[1] != len(names) or len(official) != 64:
        raise ValueError("EEGOAR signal/channel dimensions differ from the released contract")
    selected = [index for index, channel in enumerate(names) if channel in official]
    destinations = [official.index(names[index]) for index in selected]
    if not selected or len(set(destinations)) != len(destinations):
        raise ValueError("EEGOAR channel mapping is empty or non-unique")
    output = np.zeros((1, value.shape[0], 64), dtype=np.float32)
    # Index the two-dimensional view first.  Combining an integer index with
    # an advanced column index on the 3-D array moves the indexed axis to the
    # front and silently transposes this assignment.
    output[0][:, destinations] = value[:, selected]
    mask = np.zeros((1, 64), dtype=bool)
    mask[0, destinations] = True
    return output, mask


def _eegdfus_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = MAIN_ROOT / "results/cgdr/eegdfus_benchmark/full_aggregate/result_summary.json"
    if not source.is_file():
        return ([{"method": "EEGDfus", "status": "blocked_missing_prior_full_run"}], {})
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for cell in payload.get("cell_summary_rows", []):
        if not isinstance(cell, Mapping):
            continue
        rows.append({
            "method": "EEGDfus" if cell.get("arm") == "conditional_diffusion" else "EEGDfus-matched-deterministic",
            "status": "completed_prior_full_population_benchmark",
            "implementation": "official_source_port",
            "protocol": cell.get("protocol"),
            "artifact": cell.get("noise_type"),
            "rrmse": cell.get("mean_rrmse_temporal"),
            "correlation": cell.get("mean_correlation"),
            "delta_snr_db": cell.get("mean_snr_improvement_db"),
            "optimizer_updates": cell.get("optimizer_updates"),
            "identity_unit": "source_epoch_not_participant",
            "subject_awareness_evidence": False,
        })
    return rows, payload


def _eegoar_runtime() -> dict[str, Any]:
    repository = CODE_ROOT / "external_repos/EEGOAR-Net"
    row: dict[str, Any] = {
        "method": "EEGOAR-Net",
        "implementation": "official_pretrained_weights",
        "repository_commit": _git_head(repository),
        "subject_awareness_evidence": False,
    }
    try:
        import tensorflow as tf  # type: ignore

        sys.path.insert(0, str(repository))
        from EEGOARNET_architecture import EEGOARNET  # type: ignore
        model = EEGOARNET()
        model.load_weights(str(repository / "EEGOAR-Net_weights.h5"))
        recordings = np.load(repository / "materials/eeg_examples.npy", allow_pickle=True)[()]
        channels = [str(value) for value in recordings["channel_labels"]]
        official_channels = list(np.load(repository / "materials/channel_set_64ch.npy", allow_pickle=True)[()])
        signal = np.asarray(recordings["Subjects"]["S1"], dtype=np.float32)
        standardized, channel_mask = _standardize_eegoar_signal(signal, channels, official_channels)
        samples = standardized.shape[1] // 128
        epochs = standardized[0, : samples * 128].reshape(samples, 128, 64)
        masks = np.tile(channel_mask[0], (samples, 1))
        output = model.predict([epochs[..., None], masks], verbose=0)
        input_rms = float(np.sqrt(np.mean(np.square(epochs.astype(np.float64)))))
        output_rms = float(np.sqrt(np.mean(np.square(output.astype(np.float64)))))
        row.update({
            "status": "official_bundled_example_runtime_passed",
            "tensorflow_version": tf.__version__,
            "example_epochs": int(samples),
            "finite_output": bool(np.isfinite(output).all()),
            "output_input_rms_ratio": output_rms / max(input_rms, np.finfo(np.float64).eps),
            "frozen_split_status": "blocked_no_official_training_or_split_entry",
        })
    except Exception as error:
        row.update({
            "status": "blocked_official_runtime_dependency_or_weight_incompatibility",
            "failure": f"{type(error).__name__}: {error}",
            "frozen_split_status": "blocked_no_official_training_or_split_entry",
        })
    return row


def _d4pm_runtime() -> dict[str, Any]:
    repository = CODE_ROOT / "external_repos/D4PM"
    data = Path("/projects/EEG-foundation-model/eegdenoisenet/github-8d290661146c7189c98cc04812d37371d4b9426c")
    present = {path.name for path in data.glob("*_all_epochs.npy")}
    expected = {"EEG_all_epochs.npy", "EOG_all_epochs.npy", "EMG_all_epochs.npy", "ECG_all_epochs.npy"}
    row: dict[str, Any] = {
        "method": "D4PM",
        "implementation": "official_source_audit",
        "repository_commit": _git_head(repository),
        "available_native_arrays": ";".join(sorted(present)),
        "missing_native_arrays": ";".join(sorted(expected - present)),
        "query_artifact_label_in_official_test": True,
        "subject_awareness_evidence": False,
    }
    try:
        module_path = repository / "denoising_model_eegdnet_class.py"
        spec = importlib.util.spec_from_file_location("d4pm_clean_model", module_path)
        if spec is None or spec.loader is None:
            raise ImportError("cannot create D4PM model spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        row["model_module_import"] = "passed"
    except Exception as error:
        row["model_module_import"] = f"blocked: {type(error).__name__}: {error}"
    if expected - present:
        row["status"] = "blocked_exact_official_matrix_missing_ECG_array"
    else:
        row["status"] = "source_runnable_but_deployable_label_free_protocol_not_defined"
    row["deployable_reconstruction_status"] = "not_run_requires_label_free_frozen_protocol"
    row["oracle_extra_information_status"] = "not_mixed_with_deployable_results"
    return row


def _source_only_rows() -> list[dict[str, Any]]:
    specifications = (
        ("Essentia", "reconstructed_from_paper", "released training entry contains placeholder loaders and author-local paths"),
        ("SGEYESUB", "source_faithful_python_port", "MATLAB numerical parity not established"),
        ("DeepSeparator", "ported_legacy_population_baseline", "single-channel EEGdenoiseNet target; no participant claim"),
        ("ART", "official_source_available_target_port_not_run", "ICA-labelled source-mixture target differs from the frozen paired target"),
        ("IC-U-Net", "official_source_available_inference_port_not_run", "ICA brain/non-brain target and 30-channel contract differ from the frozen split"),
        ("ICA+ICLabel", "library_baseline_available", "requires frozen ICA fit scope on each real-data protocol"),
        ("ASR", "library_baseline_available", "requires clean calibration semantics; no clean waveform truth on SGE"),
        ("MNE-EOGRegression", "library_baseline_available", "support-only EOG regression comparator"),
    )
    return [
        {"method": method, "status": status, "implementation": status, "limitation": limitation,
         "subject_awareness_evidence": False}
        for method, status, limitation in specifications
    ]


def _prior_native_sge_row() -> dict[str, Any]:
    """Reuse the previously completed all-study source-faithful port output."""

    paths = (
        CODE_ROOT / "results/cgdr/sgeyesub_operator_specificity/development/metrics.csv",
        CODE_ROOT / "results/cgdr/sgeyesub_operator_specificity/evaluation/metrics.csv",
    )
    selected: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            selected.extend(
                row for row in csv.DictReader(stream)
                if row.get("method_id") == "native_sgeyesub_python_release_internal"
                and str(row.get("status", "")).startswith("success")
            )
    row: dict[str, Any] = {
        "method": "SGEYESUB-source-faithful-Python",
        "implementation": "source_faithful_python_port_not_MATLAB_parity",
        "subject_awareness_evidence": False,
        "prior_outcome_exposure": True,
    }
    if len(selected) != 59:
        row.update({"status": "blocked_prior_all_study_matrix_incomplete", "successful_stems": len(selected)})
        return row
    fields = (
        "eog_coherence_reduction", "nonartifact_observation_preservation",
        "reference_free_psd_distortion", "reference_free_covariance_distortion",
        "condition_erp_observation_relative_preservation", "observation_change_ratio",
    )
    row.update({
        "status": "completed_prior_all_study_development_matrix",
        "successful_stems": 59,
        "registered_denominator": 59,
        **{
            field: float(np.nanmean([float(value[field]) for value in selected]))
            for field in fields
        },
    })
    return row


def run(run_dir: Path) -> Mapping[str, Any]:
    rows, eegdfus = _eegdfus_rows()
    rows.extend((_eegoar_runtime(), _d4pm_runtime(), _prior_native_sge_row(), *_source_only_rows()))
    _write_csv(RESULT_ROOT / "population_baseline_runtime.csv", rows)
    summary = {
        "status": "completed_official_population_baseline_runtime_audit",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "eegdfus_full_matrix_reused": eegdfus.get("matrix_cells_completed") == 8,
        "eegdfus_scientific_scope": eegdfus.get("claim_scope"),
        "methods": {str(row["method"]): str(row["status"]) for row in rows},
        "exact_reproduction_claimed": False,
    }
    _write_json(RESULT_ROOT / "population_baseline_runtime_summary.json", summary)
    _write_json(run_dir / "result_summary.json", summary)
    return summary
