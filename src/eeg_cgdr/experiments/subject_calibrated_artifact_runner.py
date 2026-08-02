"""Slurm stages for subject-calibrated ocular-artifact latent modelling.

This is intentionally a small experiment runner, not a general workflow
engine.  The scientific boundary is fixed by
``configs/cgdr/subject_calibrated_artifact_development.yaml``: all SGEYESUB
studies are development evidence, Eye-BCI is inspected through manifests only,
and no confirmation signal or outcome is opened in this round.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW

from eeg_cgdr.experiments.eye_bci_confirmation_exposure_audit import (
    audit_eye_bci_confirmation_exposure,
    write_exposure_audit,
)
from eeg_cgdr.models.artifact_latent_deterministic import (
    ArtifactLatentModelConfig,
    DeterministicArtifactEstimator,
)
from eeg_cgdr.models.artifact_latent_diffusion import (
    ArtifactLatentDiffusion,
    ArtifactLatentDiffusionConfig,
)
from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)
from saddpm.utils.ema import EMA


PROTOCOL_ID = "subject_calibrated_artifact_latent_diffusion_development_v1"
IMPLEMENTATION_VERSION = "subject_calibrated_artifact_runner_v2"
CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
DATA_ROOT = Path("/projects/EEG-foundation-model")
OLD_DECISION = CODE_ROOT / "results/cgdr/diffusion_incremental_decision_v2/result_summary.json"
REPORT_PATH = CODE_ROOT / "reports/subject_calibrated_artifact_diffusion_development.md"
CONFIRMATION_PLAN_PATH = CODE_ROOT / "reports/subject_calibrated_artifact_confirmation_plan.md"


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _output_root(config: Mapping[str, Any]) -> Path:
    root = Path(str(_mapping(config, "outputs")["root"]))
    expected = CODE_ROOT / "results/cgdr/subject_calibrated_artifact_diffusion"
    if root != expected:
        raise ValueError("subject-artifact output root changed")
    return root


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("harness_level") != 1 or config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("subject-artifact harness/protocol identity changed")
    if config.get("scientific_role") != "development_exploratory_not_confirmation":
        raise ValueError("this round cannot be promoted to confirmation")
    boundaries = _mapping(config, "boundaries")
    required_false = {
        "participant_identity_input",
        "personalized_clean_brain_prior",
        "query_eog_or_eye_tracking_input",
        "query_artifact_label_input",
        "query_outcome_input",
        "best_of_k_selection",
        "confirmation_outcomes_this_round",
    }
    if any(boundaries.get(key) != "forbidden" for key in required_false):
        raise ValueError("information boundary was weakened")
    latent = _mapping(config, "artifact_latent")
    if latent.get("posterior_samples") != 8 or latent.get("posterior_point_estimate") != (
        "arithmetic_mean_in_artifact_latent_space"
    ):
        raise ValueError("posterior output rule must remain arithmetic K=8 mean")
    diffusion = _mapping(config, "primary_diffusion")
    required_diffusion = {
        "prediction_target": "v",
        "schedule": "cosine",
        "timesteps": 1000,
        "ema_decay": 0.999,
        "min_snr_gamma": 5.0,
        "observation_anchored_subtraction": True,
    }
    if any(diffusion.get(key) != value for key, value in required_diffusion.items()):
        raise ValueError("primary artifact diffusion contract changed")
    if tuple(_mapping(config, "training").get("seeds", ())) != (
        20260811,
        20260812,
        20260813,
    ):
        raise ValueError("three frozen training seeds changed")
    data_root = Path(str(_mapping(config, "data")["root"]))
    if data_root != DATA_ROOT:
        raise ValueError("data root changed")
    _output_root(config)


def _implementation() -> dict[str, Any]:
    head = os.environ.get("DENOISENET_GIT_HEAD", "").strip()
    if len(head) != 40 or any(char not in "0123456789abcdefABCDEF" for char in head):
        raise RuntimeError("DENOISENET_GIT_HEAD must be the scheduled Git commit")
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "git_commit": head.lower(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local_not_allowed"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


def _save_stage_config(config: Mapping[str, Any], run_dir: Path) -> None:
    _save_yaml(run_dir / "resolved_config.yaml", config)


def _run_j0(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Read only the requested target paths and existing lightweight manifests."""

    data = _mapping(config, "data")
    targets = {
        "sgeyesub": Path(str(_mapping(data, "sgeyesub")["data_root"])),
        "klados_v4": Path(str(_mapping(data, "klados_v4")["data_root"])),
        "eye_bci": Path(str(_mapping(data, "eye_bci")["data_root"])),
    }
    availability = {name: path.is_dir() for name, path in targets.items()}
    if not availability["sgeyesub"] or not availability["klados_v4"]:
        raise RuntimeError("required development data target directory is unavailable")
    old = json.loads(OLD_DECISION.read_text(encoding="utf-8"))
    if old.get("current_M2_status") != "current_M2_no_incremental_value":
        raise ValueError("frozen pre-round decision is not the expected 3ad4856 state")
    exposure = audit_eye_bci_confirmation_exposure()
    exposure_path = CODE_ROOT / "reports/eye_bci_confirmation_exposure_audit.json"
    write_exposure_audit(exposure_path, exposure)
    payload = {
        "status": "passed_j0_manifest_and_target_availability",
        "protocol_id": PROTOCOL_ID,
        **_implementation(),
        "target_directory_availability": availability,
        "full_data_root_scan_performed": False,
        "data_download_performed": False,
        "file_hashes_computed": False,
        "confirmation_signal_or_outcome_opened": False,
        "eye_bci_exposure_audit": str(exposure_path),
        "known_fresh_confirmation_pairs": len(
            exposure["known_fresh_confirmation_participant_sessions"]
        ),
        "cross_session_confirmation_schedulable_now": exposure[
            "cross_session_calibration_to_query"
        ]["schedulable_now"],
        "retained_frozen_pilot": {
            "computational_completion": "passed",
            "output_validity": "failed_output_scale_sanity",
            "scientific_comparison_eligibility": "diagnostic_only",
            "protocol_decision": "inconclusive",
            "family_wide_status": "not_tested",
            "label": (
                "completed numeric run that failed output-scale and neural-"
                "preservation validity; protocol-bounded negative pilot"
            ),
        },
    }
    _atomic_json(run_dir / "j0_audit.json", payload)
    _atomic_json(_output_root(config) / "validity/j0_audit.json", payload)
    return payload


def _run_j1(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    """Validate real support/query separation and latent-transfer construction."""

    from eeg_cgdr.experiments.subject_artifact_data import (
        validate_real_subject_artifact_inputs,
    )

    result = validate_real_subject_artifact_inputs(config)
    payload = {
        "status": "passed_j1_real_record_context_validation",
        "protocol_id": PROTOCOL_ID,
        **_implementation(),
        **dict(result),
        "confirmation_signal_or_outcome_opened": False,
    }
    _atomic_json(run_dir / "j1_cpu_validation.json", payload)
    _atomic_json(_output_root(config) / "validity/j1_cpu_validation.json", payload)
    return payload


def run_stage(
    config: Mapping[str, Any],
    run_dir: str | Path,
    stage: str,
    task_index: int | None,
) -> dict[str, Any]:
    """Execute exactly one registered Slurm stage."""

    _validate_config(config)
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _save_stage_config(config, destination)
    if stage == "j0-audit":
        if task_index is not None:
            raise ValueError("j0-audit rejects array tasks")
        return _run_j0(config, destination)
    if stage == "j1-cpu":
        if task_index is not None:
            raise ValueError("j1-cpu rejects array tasks")
        return _run_j1(config, destination)
    if stage == "validity":
        if task_index is not None:
            raise ValueError("validity rejects array tasks")
        return _run_validity(config, destination)
    if stage == "train":
        return _run_training(config, destination, task_index)
    if stage == "evaluate":
        return _run_evaluation(config, destination, task_index)
    if stage == "aggregate":
        if task_index is not None:
            raise ValueError("aggregate rejects array tasks")
        return _run_aggregate(config, destination)
    if stage == "finalize":
        if task_index is not None:
            raise ValueError("finalize rejects array tasks")
        return _run_finalize(config, destination)
    raise ValueError(f"unknown subject-artifact stage: {stage}")


def _run_validity(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    from eeg_cgdr.experiments.subject_artifact_validity_runner import (
        run_subject_artifact_validity,
    )

    attempts: list[dict[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    for implementation in (
        "primary_attempt_0",
        "primary_attempt_1",
        "primary_attempt_2",
        "residual_sdedit_backup",
    ):
        try:
            result = run_subject_artifact_validity(
                config,
                run_dir,
                implementation,
            )
        except RuntimeError as error:
            if "activation is not supported" in str(error) or "requires completed" in str(error):
                continue
            raise
        attempts.append(
            {
                "implementation": implementation,
                "status": result.get("status"),
                "passed": bool(result.get("passed")),
            }
        )
        selected = result
        if bool(result.get("passed")):
            break
    if selected is None:
        raise RuntimeError("no validity implementation was legally executable")
    passed = bool(selected.get("passed"))
    payload = {
        "status": (
            "passed_V0_to_V3"
            if passed
            else "completed_model_validity_failed"
        ),
        "protocol_id": PROTOCOL_ID,
        **_implementation(),
        "passed": passed,
        "model_validity": "passed" if passed else "failed",
        "scientific_comparison_eligibility": (
            "eligible_for_full_development_factorial" if passed else "blocked"
        ),
        "confirmation_eligibility": False,
        "selected_implementation": selected.get("implementation"),
        "attempts": attempts,
        "selected_result": dict(selected),
        "query_confirmation_outcomes_opened": False,
    }
    _atomic_json(run_dir / "validity_stage.json", payload)
    _atomic_json(_output_root(config) / "validity/result_summary.json", payload)
    return payload


def _run_training(
    config: Mapping[str, Any], run_dir: Path, task_index: int | None
) -> dict[str, Any]:
    if task_index is None:
        raise ValueError("J3 training requires a 0-149 array task")
    gate_path = _output_root(config) / "validity/result_summary.json"
    if not gate_path.is_file():
        raise RuntimeError("J3 training requires completed V0-V3")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed_V0_to_V3" or gate.get("passed") is not True:
        raise RuntimeError("J3 training is blocked because V0-V3 did not pass")
    from eeg_cgdr.experiments.subject_artifact_development_train import (
        run_subject_artifact_training,
    )

    implementation = _implementation()
    return dict(
        run_subject_artifact_training(
            config,
            run_dir,
            int(task_index),
            str(implementation["git_commit"]),
        )
    )


def _run_evaluation(
    config: Mapping[str, Any], run_dir: Path, task_index: int | None
) -> dict[str, Any]:
    if task_index is None:
        raise ValueError("J4 evaluation requires a 0-74 array task")
    gate_path = _output_root(config) / "validity/result_summary.json"
    if not gate_path.is_file():
        raise RuntimeError("J4 evaluation requires completed V0-V3")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed_V0_to_V3" or gate.get("passed") is not True:
        raise RuntimeError("J4 evaluation is blocked because V0-V3 did not pass")
    from eeg_cgdr.experiments.subject_artifact_development_eval import (
        run_subject_artifact_evaluation,
    )

    return dict(
        run_subject_artifact_evaluation(
            config,
            run_dir,
            int(task_index),
            _implementation(),
        )
    )


def _run_aggregate(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    raise NotImplementedError("aggregation is gated on completed development outputs")


def _run_finalize(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    raise NotImplementedError("finalization follows the scientific go/no-go")


__all__ = ["run_stage"]
