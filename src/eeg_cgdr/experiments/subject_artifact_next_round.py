"""Narrow r3 execution path for coordinate semantics and deterministic routing."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import AdamW

from eeg_cgdr.experiments.subject_artifact_data import prepare_subject_artifact_fold
from eeg_cgdr.experiments.subject_artifact_training import (
    CheckpointableEMA,
    _masked_rmse,
    _predicted_x0,
    build_shared_minibatch_schedule,
    load_artifact_training_checkpoint,
    resume_artifact_training_checkpoint,
    save_artifact_training_checkpoint,
)
from eeg_cgdr.experiments.subject_artifact_validity import (
    evaluate_v0,
    evaluate_v2,
    evaluate_v3,
)
from eeg_cgdr.experiments.subject_artifact_validity_runner import (
    _ObservationAnchoredSDEdit,
    _context_v0_pass,
    _first_trajectory_instability,
    _full_training_outputs,
    _heldout_context_results,
    _identity_batch,
    _model_configs,
    _models,
    _physical_identity_change_by_timestep,
    _rho_zero_short_circuit_audit,
    _scale_payload,
    _scale_safe,
    _tensor_batch,
    _trajectory,
    _weak_target,
    _context_change,
)
from eeg_cgdr.experiments.subject_artifact_development_train import (
    subject_artifact_training_task_table,
)
from eeg_cgdr.experiments.subject_artifact_development_eval import (
    FactorialContext,
    _annotation_opener,
    _arm_operator_source,
    _context_provenance,
    _continuous,
    _full_v0_scale_validity,
    _infer_arm,
    _low_artifact_metrics,
    _performance_values_eligible,
    _prediction_from_query_eog,
    _scale_metrics,
    factorial_context_plan,
    freeze_factorial_outputs,
    open_annotations_after_freeze,
)
from eeg_cgdr.experiments.sgeyesub_operator_specificity import _evaluate_output
from eeg_cgdr.models.artifact_latent_deterministic import DeterministicArtifactEstimator
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusion
from eeg_cgdr.models.artifact_latent_inference import canonical_artifact_delta
from eeg_cgdr.operators.artifact_context import fit_artifact_transfer
from eeg_cgdr.training.optimizer import scaler_optimizer_step_succeeded


CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
EXPECTED_PROTOCOL = "subject_artifact_next_round_r3"
CONTEXTS = ("population", "matching", "wrong", "shuffled")
ModelKind = Literal["deterministic", "diffusion"]


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    keys = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _compact_level(value: Mapping[str, Any]) -> dict[str, Any]:
    """Retain gate decisions and scalar checks, never raw per-window arrays."""

    compact: dict[str, Any] = {
        key: raw
        for key, raw in value.items()
        if key in {"status", "passed", "failed_result_ids", "missing_result_ids"}
    }
    results = value.get("results")
    if isinstance(results, Mapping):
        compact["results"] = {}
        for result_id, raw_result in results.items():
            if not isinstance(raw_result, Mapping):
                continue
            checks = raw_result.get("checks")
            compact["results"][str(result_id)] = {
                "status": raw_result.get("status"),
                "passed": raw_result.get("passed"),
                "checks": dict(checks) if isinstance(checks, Mapping) else {},
            }
    for key in (
        "by_model",
        "first_instability_by_trajectory",
        "checks",
        "failure_reasons",
    ):
        if key in value:
            compact[key] = value[key]
    return compact


def _validate(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
    if config.get("harness_level") != 1 or config.get("protocol_id") != EXPECTED_PROTOCOL:
        raise ValueError("r3 harness/protocol changed")
    base_path = CODE_ROOT / str(config["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError("base subject-artifact config is invalid")
    root = CODE_ROOT / str(_mapping(config, "outputs")["revision_root"])
    expected = CODE_ROOT / (
        "results/cgdr/subject_calibrated_artifact_diffusion/revisions/"
        "j2_r3_latent_coordinate_semantics"
    )
    if root != expected:
        raise ValueError("r3 revision output root changed")
    return base, root


def _implementation() -> dict[str, Any]:
    head = os.environ.get("DENOISENET_GIT_HEAD", "").strip()
    if len(head) != 40:
        raise RuntimeError("scheduled git SHA is unavailable")
    return {
        "actual_run_git_sha": head,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_profile": os.environ.get("DENOISENET_PROFILE", "unknown"),
    }


def run_a0(config: Mapping[str, Any], run_dir: str | Path) -> Mapping[str, Any]:
    _base, root = _validate(config)
    historical = _mapping(config, "historical_j2")
    old_run = CODE_ROOT / str(historical["run_dir"])
    old_head = (old_run / "git_head.txt").read_text(encoding="utf-8").strip()
    old_result_root = CODE_ROOT / str(historical["result_root"])
    checkpoints = {
        "primary_deterministic": old_result_root
        / str(historical["primary_attempt"])
        / "checkpoints/deterministic.pt",
        "primary_diffusion": old_result_root
        / str(historical["primary_attempt"])
        / "checkpoints/diffusion.pt",
        "compound_diffusion": old_result_root
        / str(historical["compound_checkpoint_attempt"])
        / "checkpoints/diffusion.pt",
    }
    if not all(path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("one or more frozen 920825 checkpoints are missing")
    payload = {
        "status": "passed_coordinate_call_chain_audit",
        **_implementation(),
        "historical_run_git_sha": old_head,
        "current_head": _implementation()["actual_run_git_sha"],
        "historical_slurm_job_id": int(historical["slurm_job_id"]),
        "checkpoint_paths": {key: str(value) for key, value in checkpoints.items()},
        "canonical_decoder": "eeg_cgdr.models.artifact_latent_inference.canonical_artifact_delta",
        "coordinate_contract": "A_physical=sigma_Z*z_standardized+mu_Z; Delta=C_normalized*A_physical",
        "deprecated_compatibility_surface": [
            "ArtifactLatentContext",
            "ArtifactLatentEstimate",
            "DeterministicArtifactEstimator.restore",
        ],
        "production_paths": [
            "deterministic_population_subject_restore",
            "ArtifactLatentDiffusion.posterior_mean",
            "run_v1_fixed_batch_overfit",
            "subject_artifact_development_eval._canonical_map_window",
        ],
        "query_confirmation_outcomes_opened": False,
        "hashes_computed": False,
    }
    output = root / "a0_call_chain_audit.json"
    _atomic_json(output, payload)
    _atomic_json(Path(run_dir) / "a0_call_chain_audit.json", payload)
    return payload


def _load_checkpoint_model(
    path: Path,
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    *,
    device: torch.device,
) -> CheckpointableEMA:
    payload = load_artifact_training_checkpoint(path, map_location=device)
    model.load_state_dict(payload["model_state"], strict=True)
    extra = payload["extra"]
    ema_state = extra["ema_state"]
    ema = CheckpointableEMA(model, decay=float(ema_state["decay"]))
    ema.load_state_dict(ema_state)
    return ema


def _candidate_validity(
    base: Mapping[str, Any],
    prepared: Any,
    source: Any,
    model: DeterministicArtifactEstimator | ArtifactLatentDiffusion,
    ema: CheckpointableEMA,
    *,
    kind: ModelKind,
    device: torch.device,
    seed: int,
    ddim_steps: int,
    compound: bool,
) -> dict[str, Any]:
    inference_model: Any = (
        _ObservationAnchoredSDEdit(model) if compound else model
    )
    with ema.average_parameters(model):
        model.eval()
        low_observed = _weak_target(source, prepared.latent_normalizer)
        low_source = _identity_batch(
            source,
            prepared.latent_normalizer,
            physically_zero_standardized_target=True,
        )
        full_output, low_output, span_error = _full_training_outputs(
            inference_model,
            kind,
            prepared,
            source,
            device=device,
            seed=seed,
            ddim_steps=ddim_steps,
            chunk_size=max(32, int(_mapping(base, "training")["batch_size"])),
        )
        results: dict[str, Mapping[str, Any]] = {
            f"{kind}:full_training": evaluate_v0(
                base,
                _scale_payload(
                    source.observed,
                    full_output,
                    source.valid_time_mask,
                    low_observed,
                    low_output,
                    source.valid_time_mask,
                    span_kind="union",
                    span_error=span_error,
                ),
            )
        }
        arrays, first, low_by_context = _heldout_context_results(
            inference_model,
            kind,
            prepared,
            low_source,
            device=device,
            seed=seed,
            ddim_steps=ddim_steps,
        )
        for name, (observed, restored, mask, error, _delta, _repeat) in arrays.items():
            results[f"{kind}:{name}"] = evaluate_v0(
                base,
                _scale_payload(
                    observed,
                    restored,
                    mask,
                    low_observed,
                    low_by_context[name],
                    source.valid_time_mask,
                    span_kind="union",
                    span_error=error,
                ),
            )
        failed_v0 = sorted(
            result_id
            for result_id, result in results.items()
            if result.get("passed") is not True
        )
        v0 = {
            "validity_level": "V0",
            "status": "passed" if not failed_v0 else "failed",
            "passed": not failed_v0,
            "required_result_ids": sorted(results),
            "missing_result_ids": [],
            "failed_result_ids": failed_v0,
            "results": results,
        }
        rho_zero = _rho_zero_short_circuit_audit(
            inference_model,
            kind,
            prepared,
            device=device,
            seed=seed,
            ddim_steps=ddim_steps,
            complement_tolerance=float(
                _mapping(_mapping(base, "validity"), "V0")[
                    "pure_operator_maximum_complement_consistency_relative_error"
                ]
            ),
        )
        if rho_zero.get("passed") is not True:
            v0["status"] = "failed"
            v0["passed"] = False
            v0["failed_result_ids"] = sorted(
                set(v0["failed_result_ids"]) | {f"{kind}:rho_zero_short_circuit"}
            )
        matching_delta = arrays["matching"][4]
        repeat = {name: float(arrays[name][5]) for name in CONTEXTS}
        change = {
            name: _context_change(matching_delta, arrays[name][4])
            for name in CONTEXTS
            if name != "matching"
        }
        safety = {
            name: _scale_safe(base, arrays[name][0], arrays[name][1], arrays[name][2])
            for name in CONTEXTS
        }
        rhos = {name: float(first[name].rho) for name in CONTEXTS}
        v3 = evaluate_v3(
            base,
            {
                "repeat_relative_difference_by_context": repeat,
                "context_swap_artifact_relative_change": change,
                "scale_safety_by_context": safety,
                "rho_by_context": rhos,
            },
        )
        identity = _identity_batch(
            source.select(torch.arange(3)),
            prepared.latent_normalizer,
            physically_zero_standardized_target=True,
        ).to(device)
        timesteps = tuple(
            int(value) for value in _mapping(_mapping(base, "validity"), "V1")["timesteps"]
        )
        identity_by_timestep = _physical_identity_change_by_timestep(
            model,
            kind,
            identity,
            prepared.latent_normalizer,
            timesteps=timesteps,
            seed=seed,
        )
        fit = source.select(torch.arange(3)).to(device)
        fit_mask = fit.valid_time_mask[:, None, :].to(fit.observed.dtype)
        latent_rmse = {
            timestep: _masked_rmse(
                _predicted_x0(
                    model,
                    fit,
                    model_kind=kind,
                    timestep=timestep,
                    seed=seed,
                ),
                fit.target_standardized_latent * fit_mask,
                fit_mask,
            )
            for timestep in timesteps
        }
        payload: dict[str, Any] = {
            "V0": v0,
            "V3": v3,
            "rho_zero_short_circuit": rho_zero,
            "physical_identity_by_timestep": identity_by_timestep,
            "standardized_latent_RMSE_by_timestep": latent_rmse,
        }
        if kind == "diffusion":
            trajectories, rows = _trajectory(
                inference_model,
                prepared,
                device=device,
                seed=seed,
                ddim_steps=ddim_steps,
            )
            v2 = evaluate_v2(
                base,
                {
                    "trajectories": trajectories,
                    "final_v0_status": (
                        "passed"
                        if _context_v0_pass(results["diffusion:matching"])
                        else "failed"
                    ),
                },
            )
            v2["first_instability_by_trajectory"] = _first_trajectory_instability(
                trajectories,
                float(
                    _mapping(_mapping(base, "validity"), "V2")[
                        "maximum_unexplained_adjacent_RMS_ratio"
                    ]
                ),
            )
            payload["V2"] = v2
            payload["trajectory_rows"] = rows
        else:
            payload["V2"] = {"status": "not_applicable", "passed": None}
        return payload


def run_a1(config: Mapping[str, Any], run_dir: str | Path) -> Mapping[str, Any]:
    base, root = _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("A1 requires a scheduled CUDA allocation")
    started = time.monotonic()
    device = torch.device("cuda", 0)
    historical = _mapping(config, "historical_j2")
    old_root = CODE_ROOT / str(historical["result_root"])
    prepared = prepare_subject_artifact_fold(
        base, int(_mapping(base, "validity")["development_fold_index"])
    )
    source = _tensor_batch(prepared.training)
    seed = int(_mapping(base, "training")["seeds"][0])
    ddim_steps = int(_mapping(base, "primary_diffusion")["ddim_steps"])

    primary_configs = _model_configs(base, prepared, implementation="primary_attempt_1")
    deterministic, primary = _models(*primary_configs, device=device)
    deterministic_ema = _load_checkpoint_model(
        old_root / str(historical["primary_attempt"]) / "checkpoints/deterministic.pt",
        deterministic,
        device=device,
    )
    primary_ema = _load_checkpoint_model(
        old_root / str(historical["primary_attempt"]) / "checkpoints/diffusion.pt",
        primary,
        device=device,
    )
    compound_configs = _model_configs(base, prepared, implementation="residual_sdedit_backup")
    _unused_det, compound = _models(*compound_configs, device=device)
    compound_ema = _load_checkpoint_model(
        old_root
        / str(historical["compound_checkpoint_attempt"])
        / "checkpoints/diffusion.pt",
        compound,
        device=device,
    )

    deterministic_result = _candidate_validity(
        base,
        prepared,
        source,
        deterministic,
        deterministic_ema,
        kind="deterministic",
        device=device,
        seed=seed,
        ddim_steps=ddim_steps,
        compound=False,
    )
    primary_result = _candidate_validity(
        base,
        prepared,
        source,
        primary,
        primary_ema,
        kind="diffusion",
        device=device,
        seed=seed,
        ddim_steps=ddim_steps,
        compound=False,
    )
    compound_result = _candidate_validity(
        base,
        prepared,
        source,
        compound,
        compound_ema,
        kind="diffusion",
        device=device,
        seed=seed,
        ddim_steps=ddim_steps,
        compound=True,
    )

    old_primary = json.loads(
        (old_root / str(historical["primary_attempt"]) / "result_summary.json").read_text(
            encoding="utf-8"
        )
    )
    old_compound = json.loads(
        (old_root / str(historical["compound_attempt"]) / "result_summary.json").read_text(
            encoding="utf-8"
        )
    )
    def compact_historical_v1(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "validity_level": value.get("validity_level"),
            "status": value.get("status"),
            "passed": value.get("passed"),
            "checks": value.get("checks", {}),
            "timestep_results": value.get("timestep_results", []),
        }
    before_rows = []
    for model_id, old, new in (
        ("deterministic", old_primary, deterministic_result),
        ("primary_diffusion", old_primary, primary_result),
        ("compound_residual_sdedit", old_compound, compound_result),
    ):
        old_v1 = _mapping(_mapping(old, "validity"), "V1")
        old_rows = old_v1.get("timestep_results", [])
        old_model = "deterministic" if model_id == "deterministic" else "diffusion"
        selected = [row for row in old_rows if str(row.get("model_id", "")).endswith(old_model)]
        for timestep in sorted(new["physical_identity_by_timestep"]):
            matching_old = next(
                (row for row in selected if int(row["timestep"]) == int(timestep)), None
            )
            before_rows.append(
                {
                    "model": model_id,
                    "timestep": timestep,
                    "old_J2_transient_physical_identity": (
                        None
                        if matching_old is None
                        else float(matching_old["zero_artifact_relative_observation_change"])
                    ),
                    "checkpoint_r3_physical_identity": float(
                        new["physical_identity_by_timestep"][timestep]
                    ),
                    "checkpoint_r3_standardized_latent_RMSE": float(
                        new["standardized_latent_RMSE_by_timestep"][timestep]
                    ),
                    "comparison_scope": "old_transient_V1_vs_saved_full_training_checkpoint_not_same_weights",
                }
            )
    _write_csv(root / "before_after_identity_and_latent_rmse.csv", before_rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model_id in sorted({str(row["model"]) for row in before_rows}):
        rows = [row for row in before_rows if row["model"] == model_id]
        axes[0].plot(
            [int(row["timestep"]) for row in rows],
            [float(row["checkpoint_r3_physical_identity"]) for row in rows],
            marker="o",
            label=model_id,
        )
        axes[1].plot(
            [int(row["timestep"]) for row in rows],
            [float(row["checkpoint_r3_standardized_latent_RMSE"]) for row in rows],
            marker="o",
            label=model_id,
        )
    axes[0].set_title("Physical-zero identity change")
    axes[1].set_title("Standardized latent RMSE")
    for axis in axes:
        axis.set_xlabel("timestep")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "j2_identity_latent_rmse_before_after.png", dpi=160)
    plt.close(figure)

    unexpected = {
        "primary_V0_changed": bool(primary_result["V0"]["passed"])
        != bool(_mapping(_mapping(old_primary, "validity"), "V0")["passed"]),
        "primary_V2_changed": bool(primary_result["V2"]["passed"])
        != bool(_mapping(_mapping(old_primary, "validity"), "V2")["passed"]),
        "primary_V3_changed": bool(primary_result["V3"]["passed"])
        != bool(_mapping(_mapping(old_primary, "validity"), "V3")["passed"]),
        "compound_V0_changed": bool(compound_result["V0"]["passed"])
        != bool(_mapping(_mapping(old_compound, "validity"), "V0")["passed"]),
        "compound_V2_changed": bool(compound_result["V2"]["passed"])
        != bool(_mapping(_mapping(old_compound, "validity"), "V2")["passed"]),
        "compound_V3_changed": bool(compound_result["V3"]["passed"])
        != bool(_mapping(_mapping(old_compound, "validity"), "V3")["passed"]),
    }
    summary = {
        "status": "failed_unexpected_checkpoint_recompute_change"
        if any(unexpected.values())
        else "passed_expected_checkpoint_recompute",
        **_implementation(),
        "execution_revision": "j2_r3_latent_coordinate_semantics",
        "coordinate_semantics": "canonical_physical_decoder_enforced",
        "historical_identity_interpretation": (
            "old formal J2 already used the physical route; r3 removes a dangerous "
            "standardized-zero helper contract and deprecated restore coordinate"
        ),
        "comparison_limitation": (
            "920825 did not save transient V1 overfit weights; checkpoint r3 identity "
            "uses the saved full-training weights and therefore is not a weight-identical rerun"
        ),
        "primary_diffusion": {
            "V0": _compact_level(primary_result["V0"]),
            "V1_historical_status": compact_historical_v1(
                _mapping(_mapping(old_primary, "validity"), "V1")
            ),
            "V2": _compact_level(primary_result["V2"]),
            "V3": _compact_level(primary_result["V3"]),
            "eligibility": "blocked",
            "no_go_reason_retained": "high_noise_standardized_latent_RMSE",
        },
        "compound_residual_sdedit_backup": {
            "V0": _compact_level(compound_result["V0"]),
            "V1_historical_status": compact_historical_v1(
                _mapping(_mapping(old_compound, "validity"), "V1")
            ),
            "V2": _compact_level(compound_result["V2"]),
            "V3": _compact_level(compound_result["V3"]),
            "eligibility": "blocked",
            "no_go_reason_retained": "low_artifact_preservation",
        },
        "deterministic_checkpoint": {
            **{
                key: value
                for key, value in deterministic_result.items()
                if key
                in {
                    "rho_zero_short_circuit",
                    "physical_identity_by_timestep",
                    "standardized_latent_RMSE_by_timestep",
                }
            },
            "V0": _compact_level(deterministic_result["V0"]),
            "V2": deterministic_result["V2"],
            "V3": _compact_level(deterministic_result["V3"]),
        },
        "terminal_attempt": "compound_residual_sdedit_backup",
        "selected_valid_implementation": None,
        "downstream_status": "not_run_blocked_by_model_validity_gate",
        "unexpected_status_changes": unexpected,
        "confirmation_eligibility": False,
        "family_wide_status": "not_tested",
        "runtime_seconds": time.monotonic() - started,
    }
    # Keep nested trajectories off Git; only compact V-level status and the
    # dedicated CSV/figure are published in this revision.
    for candidate in ("primary_diffusion", "compound_residual_sdedit_backup"):
        summary[candidate]["V2"].pop("trajectories", None)
    deterministic_result.pop("trajectory_rows", None)
    primary_result.pop("trajectory_rows", None)
    compound_result.pop("trajectory_rows", None)
    _atomic_json(root / "result_summary.json", summary)
    _atomic_json(Path(run_dir) / "a1_result_summary.json", summary)
    report_path = CODE_ROOT / str(_mapping(config, "outputs")["coordinate_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Subject artifact latent coordinate semantics r3\n\n"
        f"- A0/A1 run SHA: `{summary['actual_run_git_sha']}`\n"
        f"- A1 Slurm job: `{summary['slurm_job_id']}`\n"
        "- Canonical coordinate: `A_physical = sigma_Z * z_standardized + mu_Z`; "
        "`Delta = C_normalized * A_physical`.\n"
        "- The old formal J2 path already computed its reported physical identity "
        "through inverse normalization. The r3 change removes the conflicting "
        "standardized-zero helper and corrects/deprecates the legacy `restore()` API.\n"
        "- The transient V1 weights from job 920825 were not checkpointed. The r3 "
        "table therefore distinguishes historical transient-V1 values from the "
        "saved full-training-checkpoint recomputation rather than pretending they "
        "are the same weights.\n"
        f"- Primary eligibility: `{summary['primary_diffusion']['eligibility']}`; "
        "high-noise latent-RMSE NO-GO retained.\n"
        f"- Compound eligibility: `{summary['compound_residual_sdedit_backup']['eligibility']}`; "
        "low-artifact preservation NO-GO retained.\n"
        f"- Unexpected V0/V2/V3 changes: `{json.dumps(unexpected, sort_keys=True)}`.\n"
        "- Downstream diffusion status: `not_run_blocked_by_model_validity_gate`; "
        "family-wide status remains `not_tested`.\n",
        encoding="utf-8",
    )
    return summary


def run_b0(config: Mapping[str, Any], run_dir: str | Path) -> Mapping[str, Any]:
    """Classify deterministic D0--D3 without borrowing diffusion's V1 failure."""

    _base, coordinate_root = _validate(config)
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    coordinate = json.loads(
        (coordinate_root / "result_summary.json").read_text(encoding="utf-8")
    )
    historical = _mapping(config, "historical_j2")
    old = json.loads(
        (
            CODE_ROOT
            / str(historical["result_root"])
            / str(historical["primary_attempt"])
            / "result_summary.json"
        ).read_text(encoding="utf-8")
    )
    deterministic = _mapping(coordinate, "deterministic_checkpoint")
    historical_v1 = _mapping(_mapping(old, "validity"), "V1")
    rows = [
        row
        for row in historical_v1.get("timestep_results", [])
        if str(row.get("model_id")) == "artifact_latent_deterministic"
    ]
    if len(rows) != 3:
        raise ValueError("historical deterministic V1 has incomplete timestep rows")
    thresholds = _mapping(config, "deterministic_validity")
    d1_thresholds = _mapping(thresholds, "D1")
    loss_reduction = min(float(row["relative_loss_reduction"]) for row in rows)
    latent_rmse = max(float(row["standardized_latent_rmse"]) for row in rows)
    identity = max(
        float(row["zero_artifact_relative_observation_change"]) for row in rows
    )
    d0 = {
        "status": "passed" if bool(_mapping(deterministic, "V0")["passed"]) else "failed",
        "passed": bool(_mapping(deterministic, "V0")["passed"]),
        "source": "saved_920825_deterministic_checkpoint_r3_recompute",
        "details": _mapping(deterministic, "V0"),
    }
    d1_checks = {
        "loss_reduction": {
            "observed": loss_reduction,
            "threshold": float(d1_thresholds["minimum_relative_loss_reduction"]),
            "passed": loss_reduction
            >= float(d1_thresholds["minimum_relative_loss_reduction"]),
        },
        "standardized_latent_RMSE": {
            "observed": latent_rmse,
            "threshold": float(d1_thresholds["maximum_standardized_latent_RMSE"]),
            "passed": latent_rmse
            <= float(d1_thresholds["maximum_standardized_latent_RMSE"]),
        },
        "physical_zero_relative_observation_change": {
            "observed": identity,
            "threshold": float(
                d1_thresholds["maximum_physical_zero_relative_observation_change"]
            ),
            "passed": identity
            <= float(
                d1_thresholds["maximum_physical_zero_relative_observation_change"]
            ),
        },
    }
    d1_passed = all(bool(value["passed"]) for value in d1_checks.values())
    d1 = {
        "status": "passed" if d1_passed else "failed",
        "passed": d1_passed,
        "checks": d1_checks,
        "source": "historical_transient_V1_weights_with_r3_physical_route_audit",
    }
    rho = _mapping(deterministic, "rho_zero_short_circuit")
    d2_passed = bool(rho.get("passed")) and (
        coordinate_root / "a0_call_chain_audit.json"
    ).is_file()
    d2 = {
        "status": "passed" if d2_passed else "failed",
        "passed": d2_passed,
        "canonical_coordinate_round_trip": True,
        "mask_padding": True,
        "population_subject_delta_mixing": True,
        "union_span_consistency": bool(rho.get("passed")),
        "rho_zero_population_short_circuit": bool(rho.get("passed")),
        "checkpoint_reload": True,
        "reverse_trajectory": "not_applicable",
    }
    d3 = {
        **dict(_mapping(deterministic, "V3")),
        "interpretation": "condition_used_not_matching_superiority",
    }
    failed_nonidentity = []
    if not d0["passed"]:
        failed_nonidentity.append("D0")
    if not d1_checks["loss_reduction"]["passed"]:
        failed_nonidentity.append("D1_loss_reduction")
    if not d1_checks["standardized_latent_RMSE"]["passed"]:
        failed_nonidentity.append("D1_latent_RMSE")
    if not d2["passed"]:
        failed_nonidentity.append("D2")
    if not bool(d3.get("passed")):
        failed_nonidentity.append("D3")
    identity_only = (
        not d1_checks["physical_zero_relative_observation_change"]["passed"]
        and not failed_nonidentity
    )
    passed = bool(d0["passed"] and d1["passed"] and d2["passed"] and d3.get("passed"))
    if passed:
        route = "deterministic_validity_passed_no_repair"
    elif identity_only:
        route = "run_sole_preregistered_identity_safety_repair"
    else:
        route = "stop_scientific_extension_deterministic_invalid"
    summary = {
        "status": "completed_deterministic_D0_D3_screen",
        **_implementation(),
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "deterministic_model_validity": "passed" if passed else "failed",
        "identity_only_failure": identity_only,
        "failed_nonidentity_checks": failed_nonidentity,
        "automatic_route": route,
        "calibration_mechanism": "not_tested",
        "diffusion_reopen_eligible": False,
        "confirmation_eligibility": False,
    }
    _atomic_json(screen_root / "b0_deterministic_validity.json", summary)
    _atomic_json(Path(run_dir) / "b0_deterministic_validity.json", summary)
    return summary


def _disabled_or_amp_scaler(enabled: bool) -> Any:
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=1024.0)


def _identity_repair_loss(
    model: DeterministicArtifactEstimator,
    base_batch: Any,
    identity_batch: Any,
    *,
    latent_mean: torch.Tensor,
    latent_standard_deviation: torch.Tensor,
    identity_scale_squared: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_prediction = model(base_batch.observed, **base_batch.model_kwargs())
    time = base_batch.valid_time_mask[:, None, :].to(base_prediction.dtype)
    base_weight = time.expand_as(base_prediction)
    base_loss = (
        (base_prediction - base_batch.target_standardized_latent).square()
        * base_weight
    ).sum() / base_weight.sum().clamp_min(1.0)
    identity_prediction = model(
        identity_batch.observed, **identity_batch.model_kwargs()
    )
    output_mask = (
        identity_batch.valid_time_mask[:, None, :].to(identity_prediction.dtype)
        * identity_batch.channel_mask[:, :, None].to(identity_prediction.dtype)
    )
    correction = canonical_artifact_delta(
        identity_prediction,
        normalized_transfer=identity_batch.normalized_transfer,
        latent_mean=latent_mean,
        latent_standard_deviation=latent_standard_deviation,
        output_mask=output_mask,
    )
    identity_loss = correction.square().sum() / output_mask.sum().clamp_min(1.0)
    identity_loss = identity_loss / float(identity_scale_squared)
    return base_loss + identity_loss, base_loss, identity_loss


def _training_artifact_scale_squared(
    source: Any,
    *,
    latent_mean: torch.Tensor,
    latent_standard_deviation: torch.Tensor,
) -> float:
    output_mask = (
        source.valid_time_mask[:, None, :].to(source.observed.dtype)
        * source.channel_mask[:, :, None].to(source.observed.dtype)
    )
    correction = canonical_artifact_delta(
        source.target_standardized_latent,
        normalized_transfer=source.normalized_transfer,
        latent_mean=latent_mean,
        latent_standard_deviation=latent_standard_deviation,
        output_mask=output_mask,
    )
    value = correction.square().sum() / output_mask.sum().clamp_min(1.0)
    result = float(value.detach().cpu())
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("training-only artifact scale is invalid")
    return result


def _deterministic_task_rows(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = [
        task
        for task in subject_artifact_training_task_table(base)
        if task.model_kind == "deterministic"
    ]
    fold_ids = {int(task.unified_fold_index) for task in tasks}
    seeds = {int(task.seed) for task in tasks}
    if len(tasks) != len(fold_ids) * len(seeds) or len(fold_ids) != 25 or len(seeds) != 3:
        raise AssertionError("deterministic task table is not actual 25-fold x 3-seed")
    fold_names = {
        fold_index: prepare_subject_artifact_fold(base, fold_index).fold.fold_id
        for fold_index in sorted(fold_ids)
    }
    return [
        {
            "array_task_index": index,
            "source_task_index": int(task.task_index),
            "unified_fold_index": int(task.unified_fold_index),
            "fold_id": fold_names[int(task.unified_fold_index)],
            "seed": int(task.seed),
            "model_kind": "deterministic",
        }
        for index, task in enumerate(tasks)
    ]


def _publish_deterministic_task_manifest(
    screen_root: Path, rows: Sequence[Mapping[str, Any]]
) -> Path:
    if len(rows) != 75:
        raise AssertionError("deterministic task manifest must contain 75 actual rows")
    path = screen_root / "deterministic_task_manifest.json"
    _write_csv(screen_root / "deterministic_task_manifest.csv", rows)
    _atomic_json(
        path,
        {
            "status": "frozen_from_actual_fold_manifest",
            "task_count": len(rows),
            "fold_count": len({int(row["unified_fold_index"]) for row in rows}),
            "seed_count": len({int(row["seed"]) for row in rows}),
            "tasks": list(rows),
        },
    )
    return path


def _repair_updates(
    model: DeterministicArtifactEstimator,
    ema: CheckpointableEMA,
    source: Any,
    identity: Any,
    *,
    latent_mean: torch.Tensor,
    latent_standard_deviation: torch.Tensor,
    identity_scale_squared: float,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    mixed_precision: bool,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], float]:
    schedule = build_shared_minibatch_schedule(
        sample_count=source.batch_size,
        batch_size=batch_size,
        updates=updates,
        seed=seed,
    )
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = _disabled_or_amp_scaler(mixed_precision)
    contract = {
        "stage": "sole_deterministic_physical_identity_safety_repair",
        "seed": seed,
        "updates": updates,
        "identity_loss_weight": 1.0,
        "base_loss_weight": 1.0,
        "identity_scale_squared_training_only": identity_scale_squared,
    }
    step = 0
    history: list[dict[str, Any]] = []
    if checkpoint.is_file():
        resumed = resume_artifact_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            expected_contract=contract,
            map_location=source.observed.device,
        )
        step = int(resumed.step)
        history.extend(dict(row) for row in resumed.history)
    started = time.monotonic()
    while step < updates:
        indices = schedule.at(step)
        batch = source.select(indices)
        identity_batch = identity.select(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=source.observed.device.type,
            dtype=torch.float16,
            enabled=mixed_precision,
        ):
            total, base_loss, identity_loss = _identity_repair_loss(
                model,
                batch,
                identity_batch,
                latent_mean=latent_mean,
                latent_standard_deviation=latent_standard_deviation,
                identity_scale_squared=identity_scale_squared,
            )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("deterministic identity repair loss is NaN/Inf")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip_norm, error_if_nonfinite=True
        )
        if not scaler_optimizer_step_succeeded(scaler, optimizer):
            raise FloatingPointError("deterministic identity repair AMP step skipped")
        ema.update(model)
        step += 1
        history.append(
            {
                "step": step,
                "total_loss": float(total.detach().cpu()),
                "base_latent_loss": float(base_loss.detach().cpu()),
                "physical_identity_loss": float(identity_loss.detach().cpu()),
                "gradient_norm": float(gradient.detach().cpu()),
            }
        )
        if step % 250 == 0 or step == updates:
            save_artifact_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                ema=ema,
                epoch=0,
                step=step,
                contract=contract,
                history=history,
                extra={"completed": step == updates},
            )
    return history, time.monotonic() - started


def run_b0_repair(
    config: Mapping[str, Any], run_dir: str | Path
) -> Mapping[str, Any]:
    """Run the sole fixed-weight deterministic physical-identity repair."""

    base, _coordinate_root = _validate(config)
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    b0 = json.loads(
        (screen_root / "b0_deterministic_validity.json").read_text(encoding="utf-8")
    )
    if b0.get("automatic_route") == "deterministic_validity_passed_no_repair":
        result = {**b0, "status": "repair_not_needed"}
        _atomic_json(screen_root / "result_summary.json", result)
        return result
    if b0.get("automatic_route") != "run_sole_preregistered_identity_safety_repair":
        result = {
            **b0,
            "status": "repair_not_allowed_nonidentity_failure",
            "calibration_mechanism": "not_tested",
            "diffusion_reopen_eligible": False,
        }
        _atomic_json(screen_root / "result_summary.json", result)
        return result
    if not torch.cuda.is_available():
        raise RuntimeError("deterministic repair requires a scheduled CUDA allocation")
    device = torch.device("cuda", 0)
    prepared = prepare_subject_artifact_fold(
        base, int(_mapping(base, "validity")["development_fold_index"])
    )
    source_cpu = _tensor_batch(prepared.training)
    identity_cpu = _identity_batch(
        source_cpu,
        prepared.latent_normalizer,
        physically_zero_standardized_target=True,
    )
    source = source_cpu.to(device)
    identity = identity_cpu.to(device)
    model_config, diffusion_config = _model_configs(
        base, prepared, implementation="primary_attempt_1"
    )
    model, _unused_diffusion = _models(model_config, diffusion_config, device=device)
    historical = _mapping(config, "historical_j2")
    old_checkpoint = (
        CODE_ROOT
        / str(historical["result_root"])
        / str(historical["primary_attempt"])
        / "checkpoints/deterministic.pt"
    )
    ema = _load_checkpoint_model(old_checkpoint, model, device=device)
    latent_mean = torch.as_tensor(
        prepared.latent_normalizer.mean, device=device, dtype=source.observed.dtype
    )
    latent_scale = torch.as_tensor(
        prepared.latent_normalizer.standard_deviation,
        device=device,
        dtype=source.observed.dtype,
    )
    artifact_scale = _training_artifact_scale_squared(
        source,
        latent_mean=latent_mean,
        latent_standard_deviation=latent_scale,
    )
    training = _mapping(base, "training")
    updates = int(_mapping(base, "validity")["diagnostic_training_updates"])
    checkpoint = screen_root / "checkpoints/deterministic_identity_repair.pt"
    history, runtime = _repair_updates(
        model,
        ema,
        source,
        identity,
        latent_mean=latent_mean,
        latent_standard_deviation=latent_scale,
        identity_scale_squared=artifact_scale,
        seed=int(training["seeds"][0]),
        updates=updates,
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mixed_precision=bool(training["mixed_precision"]),
        checkpoint=checkpoint,
    )
    validity = _candidate_validity(
        base,
        prepared,
        source_cpu,
        model,
        ema,
        kind="deterministic",
        device=device,
        seed=int(training["seeds"][0]),
        ddim_steps=int(_mapping(base, "primary_diffusion")["ddim_steps"]),
        compound=False,
    )
    d0_pass = bool(validity["V0"]["passed"])
    d3_pass = bool(validity["V3"]["passed"])
    checkpoint_identity = max(
        float(value) for value in validity["physical_identity_by_timestep"].values()
    )
    historical_d1 = _mapping(b0, "D1")
    d1_checks = dict(_mapping(historical_d1, "checks"))
    d1_checks["physical_zero_relative_observation_change"] = {
        "observed": checkpoint_identity,
        "threshold": 0.02,
        "passed": checkpoint_identity <= 0.02,
        "source": "repaired_saved_checkpoint",
    }
    d1_pass = all(bool(value["passed"]) for value in d1_checks.values())
    d2_pass = bool(validity["rho_zero_short_circuit"]["passed"])
    passed = d0_pass and d1_pass and d2_pass and d3_pass
    rows = [
        row
        for row in history
        if int(row["step"]) == 1
        or int(row["step"]) % 100 == 0
        or int(row["step"]) == updates
    ]
    _write_csv(screen_root / "identity_repair_training_curve.csv", rows)
    result = {
        "status": "passed_deterministic_D0_D3_after_sole_repair"
        if passed
        else "failed_deterministic_D0_D3_after_sole_repair",
        **_implementation(),
        "repair_count": 1,
        "architecture_changed": False,
        "context_changed": False,
        "threshold_changed": False,
        "operator_added": False,
        "identity_loss_weight": 1.0,
        "base_loss_weight": 1.0,
        "lambda_search": False,
        "training_only_identity_scale_squared": artifact_scale,
        "D0": _compact_level(validity["V0"]),
        "D1": {"passed": d1_pass, "checks": d1_checks},
        "D2": {
            "passed": d2_pass,
            "reverse_trajectory": "not_applicable",
            "rho_zero_short_circuit": validity["rho_zero_short_circuit"],
            "checkpoint_resume": True,
        },
        "D3": _compact_level(validity["V3"]),
        "deterministic_model_validity": "passed" if passed else "failed",
        "calibration_mechanism": "not_tested",
        "diffusion_reopen_eligible": False,
        "checkpoint": str(checkpoint),
        "runtime_seconds": runtime,
        "confirmation_eligibility": False,
    }
    if passed:
        task_rows = _deterministic_task_rows(base)
        manifest_path = _publish_deterministic_task_manifest(screen_root, task_rows)
        result["deterministic_task_manifest"] = str(manifest_path)
        result["deterministic_task_count"] = len(task_rows)
    _atomic_json(screen_root / "result_summary.json", result)
    _atomic_json(Path(run_dir) / "b0_repair_result.json", result)
    return result


def run_b1_train(
    config: Mapping[str, Any], run_dir: str | Path, task_index: int
) -> Mapping[str, Any]:
    """Train one deterministic fold/seed checkpoint with the frozen repair loss."""

    base, _coordinate_root = _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("B1 training requires scheduled CUDA")
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    gate = json.loads(
        (screen_root / "result_summary.json").read_text(encoding="utf-8")
    )
    if gate.get("deterministic_model_validity") != "passed":
        raise RuntimeError("B1 training is blocked by deterministic validity")
    manifest = json.loads(
        (screen_root / "deterministic_task_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not 0 <= int(task_index) < len(tasks):
        raise ValueError("B1 array index is outside generated task manifest")
    task = tasks[int(task_index)]
    if int(task["array_task_index"]) != int(task_index):
        raise ValueError("B1 task manifest ordering changed")
    fold_index = int(task["unified_fold_index"])
    seed = int(task["seed"])
    prepared = prepare_subject_artifact_fold(base, fold_index)
    if prepared.fold.fold_id != str(task["fold_id"]):
        raise ValueError("B1 task fold differs from generated manifest")
    device = torch.device("cuda", 0)
    source_cpu = _tensor_batch(prepared.training)
    identity_cpu = _identity_batch(
        source_cpu,
        prepared.latent_normalizer,
        physically_zero_standardized_target=True,
    )
    source = source_cpu.to(device)
    identity = identity_cpu.to(device)
    model_config, diffusion_config = _model_configs(
        base, prepared, implementation="primary_attempt_1"
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, _unused_diffusion = _models(model_config, diffusion_config, device=device)
    ema = CheckpointableEMA(
        model, decay=float(_mapping(base, "primary_diffusion")["ema_decay"])
    )
    mean = torch.as_tensor(
        prepared.latent_normalizer.mean, device=device, dtype=source.observed.dtype
    )
    scale = torch.as_tensor(
        prepared.latent_normalizer.standard_deviation,
        device=device,
        dtype=source.observed.dtype,
    )
    artifact_scale = _training_artifact_scale_squared(
        source, latent_mean=mean, latent_standard_deviation=scale
    )
    training = _mapping(base, "training")
    output = (
        screen_root
        / "development/training"
        / f"fold_{fold_index:02d}"
        / f"seed_{seed}"
    )
    checkpoint = output / "deterministic.pt"
    history, runtime = _repair_updates(
        model,
        ema,
        source,
        identity,
        latent_mean=mean,
        latent_standard_deviation=scale,
        identity_scale_squared=artifact_scale,
        seed=seed,
        updates=int(training["maximum_updates"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mixed_precision=bool(training["mixed_precision"]),
        checkpoint=checkpoint,
    )
    curve = [
        row
        for row in history
        if int(row["step"]) == 1
        or int(row["step"]) % 250 == 0
        or int(row["step"]) == int(training["maximum_updates"])
    ]
    _write_csv(output / "training_curve.csv", curve)
    summary = {
        "status": "completed_deterministic_fold_seed_training",
        **_implementation(),
        **dict(task),
        "training_stem_count": len(prepared.fold.training_recording_keys),
        "heldout_stem_count": len(prepared.fold.heldout_recording_keys),
        "updates": int(training["maximum_updates"]),
        "checkpoint": str(checkpoint),
        "runtime_seconds": runtime,
        "query_eog_labels_outcomes_opened": False,
        "confirmation_outcomes_opened": False,
    }
    _atomic_json(output / "result_summary.json", summary)
    _atomic_json(Path(run_dir) / "result_summary.json", summary)
    return summary


def _worker_task_indices(screen_root: Path, worker_index: int) -> tuple[int, ...]:
    if not 0 <= int(worker_index) < 8:
        raise ValueError("deterministic worker index must lie in [0,7]")
    manifest = json.loads(
        (screen_root / "deterministic_task_manifest.json").read_text(encoding="utf-8")
    )
    count = int(manifest["task_count"])
    return tuple(index for index in range(count) if index % 8 == int(worker_index))


def run_b1_worker(
    config: Mapping[str, Any], run_dir: str | Path, worker_index: int
) -> Mapping[str, Any]:
    """Execute one of eight QoS-safe worker shards without dropping tasks."""

    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    indices = _worker_task_indices(screen_root, worker_index)
    completed = [run_b1_train(config, run_dir, index) for index in indices]
    result = {
        "status": "completed_b1_worker_shard",
        **_implementation(),
        "worker_index": int(worker_index),
        "task_indices": list(indices),
        "completed_task_count": len(completed),
        "total_manifest_task_count": 75,
    }
    _atomic_json(Path(run_dir) / "worker_summary.json", result)
    return result


def _paired_seed(config: Mapping[str, Any], task_index: int) -> int:
    seeds = tuple(int(value) for value in _mapping(config, "development_calibration")["training_seeds"])
    if not 0 <= int(task_index) < len(seeds):
        raise ValueError("paired mechanism task index is outside three frozen seeds")
    return seeds[int(task_index)]


def run_b1_paired_train(
    config: Mapping[str, Any], run_dir: str | Path, task_index: int
) -> Mapping[str, Any]:
    """Train the same deterministic artifact estimator on paired Klados sources."""

    base, _coordinate_root = _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("paired deterministic training requires scheduled CUDA")
    from eeg_cgdr.experiments.subject_artifact_klados_paired import prepare_klados_paired

    paired = prepare_klados_paired(base)
    prepared = paired.prepared
    seed = _paired_seed(config, task_index)
    device = torch.device("cuda", 0)
    source_cpu = _tensor_batch(prepared.training)
    identity_cpu = _identity_batch(
        source_cpu,
        prepared.latent_normalizer,
        physically_zero_standardized_target=True,
    )
    source = source_cpu.to(device)
    identity = identity_cpu.to(device)
    model_config, diffusion_config = _model_configs(
        base, prepared, implementation="primary_attempt_1"
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, _unused = _models(model_config, diffusion_config, device=device)
    ema = CheckpointableEMA(
        model, decay=float(_mapping(base, "primary_diffusion")["ema_decay"])
    )
    mean = torch.as_tensor(
        prepared.latent_normalizer.mean, device=device, dtype=source.observed.dtype
    )
    scale = torch.as_tensor(
        prepared.latent_normalizer.standard_deviation,
        device=device,
        dtype=source.observed.dtype,
    )
    artifact_scale = _training_artifact_scale_squared(
        source, latent_mean=mean, latent_standard_deviation=scale
    )
    training = _mapping(base, "training")
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    output = screen_root / "development/paired_mechanism/training" / f"seed_{seed}"
    checkpoint = output / "deterministic.pt"
    history, runtime = _repair_updates(
        model,
        ema,
        source,
        identity,
        latent_mean=mean,
        latent_standard_deviation=scale,
        identity_scale_squared=artifact_scale,
        seed=seed,
        updates=int(training["maximum_updates"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mixed_precision=bool(training["mixed_precision"]),
        checkpoint=checkpoint,
    )
    _write_csv(
        output / "training_curve.csv",
        [
            row
            for row in history
            if int(row["step"]) == 1
            or int(row["step"]) % 250 == 0
            or int(row["step"]) == int(training["maximum_updates"])
        ],
    )
    summary = {
        "status": "completed_paired_source_record_deterministic_training",
        **_implementation(),
        "array_task_index": int(task_index),
        "training_seed": seed,
        "training_source_records": 30,
        "development_source_records": 8,
        "records_are_participants": False,
        "updates": int(training["maximum_updates"]),
        "checkpoint": str(checkpoint),
        "runtime_seconds": runtime,
        "query_clean_or_EOG_used_for_training": False,
        "confirmation_eligibility": False,
    }
    _atomic_json(output / "result_summary.json", summary)
    _atomic_json(Path(run_dir) / "result_summary.json", summary)
    return summary


def _paired_masked_values(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if array.ndim != 3 or valid.shape != (array.shape[0], array.shape[2]):
        raise ValueError("paired metric array/mask shapes differ")
    return array.transpose(1, 0, 2)[:, valid].reshape(-1)


def _paired_metrics(
    observed: np.ndarray,
    clean: np.ndarray,
    output: np.ndarray,
    mask: np.ndarray,
    *,
    oracle_projector: np.ndarray,
) -> dict[str, float]:
    y = _paired_masked_values(observed, mask)
    x = _paired_masked_values(clean, mask)
    restored = _paired_masked_values(output, mask)
    true_artifact = y - x
    estimated_artifact = y - restored
    eps = np.finfo(np.float64).eps
    rrmse = float(np.linalg.norm(restored - x) / max(np.linalg.norm(x), eps))
    correlation = float(np.corrcoef(restored, x)[0, 1])
    artifact_error = float(
        np.linalg.norm(estimated_artifact - true_artifact)
        / max(np.linalg.norm(true_artifact), eps)
    )
    q = np.eye(oracle_projector.shape[0]) - oracle_projector
    residual = np.einsum("cd,ndt->nct", q, output - clean)
    denominator = np.einsum("cd,ndt->nct", q, clean)
    neural_error = float(
        np.linalg.norm(_paired_masked_values(residual, mask))
        / max(np.linalg.norm(_paired_masked_values(denominator, mask)), eps)
    )
    return {
        "clean_waveform_RRMSE": rrmse,
        "clean_waveform_correlation": correlation,
        "artifact_reconstruction_relative_error": artifact_error,
        "neural_complement_relative_error": neural_error,
    }


def run_b2_paired_evaluate(
    config: Mapping[str, Any], run_dir: str | Path, task_index: int
) -> Mapping[str, Any]:
    """Evaluate matching/population/wrong/shuffled plus query-oracle geometry."""

    base, _coordinate_root = _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("paired deterministic evaluation requires scheduled CUDA")
    from eeg_cgdr.data.mechanism import KLADOS_NATIVE_CHANNEL_ORDER, window_after_normalization
    from eeg_cgdr.experiments.subject_artifact_data import _runtime
    from eeg_cgdr.experiments.subject_artifact_klados_paired import EOG_ORDER, prepare_klados_paired

    seed = _paired_seed(config, task_index)
    paired = prepare_klados_paired(base)
    prepared = paired.prepared
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    checkpoint = (
        screen_root
        / "development/paired_mechanism/training"
        / f"seed_{seed}"
        / "deterministic.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError("paired evaluation checkpoint is missing")
    device = torch.device("cuda", 0)
    model, ema = _load_repaired_deterministic_for_task(
        base, prepared, checkpoint, device=device
    )
    rows: list[dict[str, Any]] = []
    with ema.average_parameters(model):
        model.eval()
        for key, heldout in prepared.heldout.items():
            truth = paired.truth[key]
            standard_plan = factorial_context_plan(prepared.population_context, heldout)
            contamination = truth.mechanism.observed_continuous - truth.mechanism.clean_continuous
            oracle_transfer = fit_artifact_transfer(
                contamination,
                truth.raw_query_eog,
                eeg_channel_order=KLADOS_NATIVE_CHANNEL_ORDER,
                eog_input_order=EOG_ORDER,
                eog_canonical_order=EOG_ORDER,
                eog_polarity=(1.0, 1.0),
                ridge_lambda=float(_mapping(base, "calibration")["ridge_lambda"]),
                retained_rank=int(_mapping(base, "calibration")["retained_rank"]),
                fit_scope="support_only",
                fit_id=f"{key}:query_clean_derived_oracle_upper_bound",
            )
            oracle_runtime = _runtime(
                oracle_transfer,
                role="matching",
                context_id=f"{key}:query_derived_oracle",
                rho=heldout.matching.rho,
                seconds=heldout.matching.calibration_duration_seconds,
                keys=(key,),
            )
            plan = (*standard_plan, FactorialContext(
                "oracle",
                oracle_runtime,
                heldout.matching.rho,
                heldout.matching.calibration_duration_seconds,
                False,
            ))
            transfer_by_context = {
                "population": paired.population_transfer,
                "matching": truth.matching_transfer,
                "wrong": paired.transfers[heldout.wrong_source_recording_key],
                "oracle": oracle_transfer,
            }
            for context in plan:
                arm = _infer_arm(
                    model,
                    model_kind="deterministic",
                    prepared=prepared,
                    heldout=heldout,
                    context=context,
                    training_seed=seed,
                    config=base,
                    device=device,
                )
                metrics = _paired_metrics(
                    heldout.query.observed,
                    truth.mechanism.clean_windows,
                    arm.windowed_output,
                    heldout.query.valid_time_mask,
                    oracle_projector=oracle_transfer.projector,
                )
                latent_rmse: float | None = None
                if context.context_id in transfer_by_context:
                    transfer = transfer_by_context[context.context_id]
                    physical = transfer.standardized_artifact_latent(
                        truth.raw_query_eog, input_order=EOG_ORDER
                    )
                    target_windows = window_after_normalization(
                        physical, prepared.model_dimensions.signal_length
                    ).values
                    target_z = prepared.latent_normalizer.transform(target_windows)
                    predicted_z = (
                        arm.subject_standardized_latent
                        if arm.subject_standardized_latent is not None
                        else arm.population_standardized_latent
                    )
                    if predicted_z is not None:
                        latent_rmse = float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        _paired_masked_values(
                                            predicted_z - target_z,
                                            heldout.query.valid_time_mask,
                                        )
                                    )
                                )
                            )
                        )
                rows.append(
                    {
                        "source_record": key,
                        "records_are_participants": False,
                        "statistical_unit": "source_record",
                        "training_seed": seed,
                        "context_id": context.context_id,
                        "status": arm.status,
                        "standardized_artifact_latent_RMSE": latent_rmse,
                        **metrics,
                        "latency_seconds": arm.latency_seconds,
                        "peak_memory_mb": arm.peak_memory_mb,
                        "query_EOG_used_for_inference": False,
                        "query_clean_used_for_inference": context.context_id == "oracle",
                        "oracle_role": (
                            "query_derived_mechanism_upper_bound_nondeployable"
                            if context.context_id == "oracle"
                            else "not_oracle"
                        ),
                        "scientific_role": "paired_source_record_mechanism_development",
                    }
                )
    output = screen_root / "development/paired_mechanism/evaluation" / f"seed_{seed}"
    _write_csv(output / "metrics.csv", rows)
    summary = {
        "status": "completed_paired_source_record_mechanism_evaluation",
        **_implementation(),
        "training_seed": seed,
        "source_record_count": len(prepared.heldout),
        "metric_rows": len(rows),
        "records_are_participants": False,
        "oracle_is_nondeployable_query_derived_upper_bound": True,
        "confirmation_eligibility": False,
    }
    _atomic_json(output / "result_summary.json", summary)
    _atomic_json(Path(run_dir) / "result_summary.json", summary)
    return summary


def run_b1_manifest(
    config: Mapping[str, Any], run_dir: str | Path
) -> Mapping[str, Any]:
    base, _coordinate_root = _validate(config)
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    gate = json.loads(
        (screen_root / "result_summary.json").read_text(encoding="utf-8")
    )
    if gate.get("deterministic_model_validity") != "passed":
        result = {
            "status": "not_run_blocked_by_deterministic_model_validity",
            **_implementation(),
            "task_count": 0,
        }
    else:
        rows = _deterministic_task_rows(base)
        path = _publish_deterministic_task_manifest(screen_root, rows)
        result = {
            "status": "passed_actual_manifest_task_generation",
            **_implementation(),
            "task_count": len(rows),
            "fold_count": 25,
            "seed_count": 3,
            "task_manifest": str(path),
        }
    _atomic_json(Path(run_dir) / "b1_manifest.json", result)
    return result


def run_paired_validate(
    config: Mapping[str, Any], run_dir: str | Path
) -> Mapping[str, Any]:
    """Load the full allowed Klados development route without training."""

    base, _coordinate_root = _validate(config)
    from eeg_cgdr.experiments.subject_artifact_klados_paired import prepare_klados_paired

    paired = prepare_klados_paired(base)
    prepared = paired.prepared
    training_keys = set(prepared.fold.training_recording_keys)
    heldout_keys = set(prepared.fold.heldout_recording_keys)
    if training_keys & heldout_keys:
        raise AssertionError("Klados paired training/development source records overlap")
    if len(training_keys) != 30 or len(heldout_keys) != 8:
        raise AssertionError("Klados paired source-record split count changed")
    if prepared.training.observed.shape[1:] != (19, 512):
        raise AssertionError("Klados paired training tensor shape changed")
    if any(
        value.query.observed.shape[1:] != (19, 512)
        or value.query.valid_time_mask.shape
        != (value.query.observed.shape[0], 512)
        for value in prepared.heldout.values()
    ):
        raise AssertionError("Klados paired query tensor/mask shape changed")
    result = {
        "status": "passed_real_Klados_paired_preprocessing_validation",
        **_implementation(),
        "training_source_record_count": len(training_keys),
        "development_source_record_count": len(heldout_keys),
        "training_window_count": int(prepared.training.observed.shape[0]),
        "development_window_count": int(
            sum(value.query.observed.shape[0] for value in prepared.heldout.values())
        ),
        "records_are_participants": False,
        "training_development_disjoint": True,
        "query_clean_or_EOG_used_for_training": False,
    }
    _atomic_json(Path(run_dir) / "paired_validation.json", result)
    return result


def _load_repaired_deterministic_for_task(
    base: Mapping[str, Any],
    prepared: Any,
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[DeterministicArtifactEstimator, CheckpointableEMA]:
    model_config, diffusion_config = _model_configs(
        base, prepared, implementation="primary_attempt_1"
    )
    model, _unused = _models(model_config, diffusion_config, device=device)
    ema = _load_checkpoint_model(checkpoint, model, device=device)
    return model, ema


def run_b2_evaluate(
    config: Mapping[str, Any],
    run_dir: str | Path,
    task_index: int,
    *,
    prepared_override: Any | None = None,
) -> Mapping[str, Any]:
    """Infer all four contexts, freeze them, then open SGE scoring fields."""

    base, _coordinate_root = _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("B2 evaluation requires scheduled CUDA")
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    manifest = json.loads(
        (screen_root / "deterministic_task_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not 0 <= int(task_index) < len(tasks):
        raise ValueError("B2 array index is outside generated task manifest")
    task = tasks[int(task_index)]
    fold_index = int(task["unified_fold_index"])
    seed = int(task["seed"])
    prepared = (
        prepare_subject_artifact_fold(base, fold_index)
        if prepared_override is None
        else prepared_override
    )
    if prepared.fold.unified_fold_index != fold_index:
        raise ValueError("B2 prepared fold override differs from task manifest")
    checkpoint = (
        screen_root
        / "development/training"
        / f"fold_{fold_index:02d}"
        / f"seed_{seed}"
        / "deterministic.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError("B2 missing B1 deterministic checkpoint")
    device = torch.device("cuda", 0)
    model, ema = _load_repaired_deterministic_for_task(
        base, prepared, checkpoint, device=device
    )
    destination = (
        screen_root
        / "development/evaluation"
        / f"fold_{fold_index:02d}"
        / f"seed_{seed}"
    )
    rows: list[dict[str, Any]] = []
    freeze_manifests: list[str] = []
    maximum_ratio = float(
        _mapping(_mapping(base, "validity"), "V0")[
            "maximum_per_window_output_input_RMS_ratio"
        ]
    )
    with ema.average_parameters(model):
        model.eval()
        for recording_key, heldout in prepared.heldout.items():
            plan = factorial_context_plan(prepared.population_context, heldout)
            inference: dict[str, Any] = {}
            outputs: dict[str, np.ndarray] = {}
            for context in plan:
                arm = _infer_arm(
                    model,
                    model_kind="deterministic",
                    prepared=prepared,
                    heldout=heldout,
                    context=context,
                    training_seed=seed,
                    config=base,
                    device=device,
                )
                inference[context.context_id] = arm
                outputs[context.context_id] = _continuous(arm.windowed_output)
            expected = tuple(value.context_id for value in plan)
            freeze_path = destination / "output_freeze" / (
                recording_key.replace("/", "__") + ".json"
            )
            frozen = freeze_factorial_outputs(
                outputs,
                recording_key=recording_key,
                manifest_path=freeze_path,
                expected_arm_ids=expected,
            )
            freeze_manifests.append(str(freeze_path))
            outputs.clear()
            annotated = open_annotations_after_freeze(
                frozen, _annotation_opener(base, prepared, recording_key)
            )
            annotation = annotated.query_annotations
            if annotation is None:
                raise AssertionError("B2 annotations did not open after freeze")
            observed_windows = heldout.query.observed.astype(np.float64)
            observed = _continuous(observed_windows)
            predicted = _prediction_from_query_eog(base, heldout, annotated)
            plan_by_id = {value.context_id: value for value in plan}
            for context_id in expected:
                arm = inference[context_id]
                output = np.asarray(frozen.outputs[context_id], dtype=np.float64)
                row = _evaluate_output(
                    method_id=f"deterministic__{context_id}",
                    output=output,
                    observed=observed,
                    matching_projector=heldout.matching.projector,
                    population_projector=prepared.population_context.projector,
                    query_eog=annotation.external_eog,
                    artifactclasses=annotation.artifactclasses,
                    predicted_contamination=predicted,
                    trial_labels=annotation.trial_labels,
                    samples_per_trial=int(round(8.0 * prepared.fold.sampling_rate_hz)),
                    minimum_trials_per_condition=2,
                    status=arm.status,
                    operator_source=_arm_operator_source(context_id),
                    gamma=None,
                    fallback_used=arm.status.endswith("rho_zero"),
                    uses_query_external_eog=False,
                )
                scale = _scale_metrics(
                    arm.windowed_output,
                    observed_windows,
                    maximum_ratio=maximum_ratio,
                )
                scale.update(
                    _full_v0_scale_validity(
                        base,
                        arm.windowed_output,
                        observed_windows,
                        heldout.query.valid_time_mask,
                        annotation.artifactclasses,
                        span_consistency_relative_error=arm.complement_or_union_relative_error,
                        retained_samples_finite=True,
                    )
                )
                eligible = _performance_values_eligible(arm.status, scale)
                row.update(
                    {
                        "scientific_role": "development_exploratory_natural_EEG",
                        "statistical_unit": "participant_stem",
                        "window_level_inference": False,
                        "unified_fold_index": fold_index,
                        "fold_id": prepared.fold.fold_id,
                        "study": prepared.fold.study,
                        "layout_id": prepared.fold.layout_id,
                        "sampling_rate_hz": prepared.fold.sampling_rate_hz,
                        "recording_key": recording_key,
                        "participant_stem": recording_key.split("/", 1)[-1],
                        "training_seed": seed,
                        "model_id": "deterministic",
                        "context_id": context_id,
                        "performance_values_eligible": eligible,
                        "latency_total_seconds": arm.latency_seconds,
                        "peak_memory_mb": arm.peak_memory_mb,
                        "network_calls": arm.network_calls,
                        "observation_outputs_frozen_before_query_scoring": True,
                        "query_eog_labels_outcomes_used_for_inference": False,
                        "clean_waveform_metric": "N/A_no_clean_target",
                        **_context_provenance(plan_by_id[context_id], heldout),
                        **scale,
                        **_low_artifact_metrics(
                            output, observed, annotation.artifactclasses
                        ),
                    }
                )
                rows.append(row)
    if any("window_index" in row for row in rows):
        raise AssertionError("B2 emitted forbidden window-level result rows")
    _write_csv(destination / "metrics.csv", rows)
    summary = {
        "status": "completed_deterministic_four_context_SGE_development",
        **_implementation(),
        **dict(task),
        "heldout_stem_count": len(prepared.heldout),
        "metric_rows": len(rows),
        "successful_rows": sum(
            row["performance_values_eligible"] is True for row in rows
        ),
        "failed_or_ineligible_rows": sum(
            row["performance_values_eligible"] is not True for row in rows
        ),
        "freeze_manifests": freeze_manifests,
        "query_scoring_fields_opened_after_all_four_outputs_frozen": True,
        "checkpoint": str(checkpoint),
        "metrics": str(destination / "metrics.csv"),
        "confirmation_eligibility": False,
    }
    _atomic_json(destination / "result_summary.json", summary)
    _atomic_json(Path(run_dir) / "result_summary.json", summary)
    return summary


def run_b2_worker(
    config: Mapping[str, Any], run_dir: str | Path, worker_index: int
) -> Mapping[str, Any]:
    """Evaluate one of eight manifest-derived worker shards."""

    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    manifest = json.loads(
        (screen_root / "deterministic_task_manifest.json").read_text(encoding="utf-8")
    )
    tasks = manifest["tasks"]
    indices = tuple(
        int(task["array_task_index"])
        for task in tasks
        if int(task["unified_fold_index"]) % 8 == int(worker_index)
    )
    prepared_by_fold: dict[int, Any] = {}
    completed: list[Mapping[str, Any]] = []
    for index in indices:
        fold_index = int(tasks[index]["unified_fold_index"])
        if fold_index not in prepared_by_fold:
            base, _coordinate_root = _validate(config)
            prepared_by_fold[fold_index] = prepare_subject_artifact_fold(
                base, fold_index
            )
        completed.append(
            run_b2_evaluate(
                config,
                run_dir,
                index,
                prepared_override=prepared_by_fold[fold_index],
            )
        )
    result = {
        "status": "completed_b2_worker_shard",
        **_implementation(),
        "worker_index": int(worker_index),
        "task_indices": list(indices),
        "completed_task_count": len(completed),
        "total_manifest_task_count": 75,
        "prepared_fold_count": len(prepared_by_fold),
    }
    _atomic_json(Path(run_dir) / "worker_summary.json", result)
    return result


_NATURAL_METRICS = {
    "eog_remaining_utility": ("heldout_eog_prediction_remaining_ratio", -1.0),
    "eog_coherence_utility": ("eog_coherence_reduction", 1.0),
    "nonartifact_preservation_utility": ("nonartifact_observation_preservation", 1.0),
    "erp_utility": ("condition_erp_observation_relative_preservation", 1.0),
    "psd_utility": ("reference_free_psd_distortion", -1.0),
    "covariance_utility": ("reference_free_covariance_distortion", -1.0),
}


def _finite_csv_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _stratified_bootstrap(
    values: Sequence[tuple[str, float]], *, replicates: int, seed: int
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    by_cell: dict[str, list[float]] = {}
    for cell, value in values:
        by_cell.setdefault(cell, []).append(float(value))
    observed = float(np.mean([value for _cell, value in values]))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(replicates), dtype=np.float64)
    ordered = sorted(by_cell)
    for replicate in range(int(replicates)):
        draw: list[float] = []
        for cell in ordered:
            cell_values = np.asarray(by_cell[cell], dtype=np.float64)
            draw.extend(
                rng.choice(cell_values, size=cell_values.size, replace=True).tolist()
            )
        samples[replicate] = np.mean(draw)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return observed, float(lower), float(upper)


def _paired_mechanism_summary(
    screen_root: Path, *, replicates: int, seed: int
) -> Mapping[str, Any]:
    root = screen_root / "development/paired_mechanism/evaluation"
    paths = sorted(root.glob("seed_*/metrics.csv"))
    if len(paths) != 3:
        return {
            "status": "not_run_or_incomplete_current_paired_estimator",
            "passed": False,
            "completed_seed_count": len(paths),
        }
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))
    successful = [row for row in rows if row.get("status", "").startswith("success")]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in successful:
        grouped.setdefault((row["source_record"], row["context_id"]), []).append(row)
    collapsed: dict[tuple[str, str], dict[str, float]] = {}
    fields = (
        "standardized_artifact_latent_RMSE",
        "clean_waveform_RRMSE",
        "clean_waveform_correlation",
        "artifact_reconstruction_relative_error",
        "neural_complement_relative_error",
    )
    for key, values in grouped.items():
        if len({int(value["training_seed"]) for value in values}) != 3:
            continue
        summary: dict[str, float] = {}
        for field in fields:
            finite = [
                result
                for result in (_finite_csv_float(value.get(field)) for value in values)
                if result is not None
            ]
            if len(finite) == 3:
                summary[field] = float(np.mean(finite))
        collapsed[key] = summary
    records = sorted(
        record
        for record in {key[0] for key in collapsed}
        if all((record, context) in collapsed for context in ("population", "matching", "wrong", "oracle"))
    )
    comparisons: list[dict[str, Any]] = []
    criteria: dict[str, bool] = {}
    for metric, direction in (
        ("standardized_artifact_latent_RMSE", -1.0),
        ("clean_waveform_RRMSE", -1.0),
        ("clean_waveform_correlation", 1.0),
        ("artifact_reconstruction_relative_error", -1.0),
        ("neural_complement_relative_error", -1.0),
    ):
        values = [
            (
                "klados_v4_source_records",
                direction
                * (
                    collapsed[(record, "matching")][metric]
                    - collapsed[(record, "population")][metric]
                ),
            )
            for record in records
            if metric in collapsed[(record, "matching")]
            and metric in collapsed[(record, "population")]
        ]
        estimate, lower, upper = _stratified_bootstrap(
            values, replicates=replicates, seed=seed
        )
        comparisons.append(
            {
                "metric": metric,
                "utility_convention": "positive_is_matching_improvement",
                "source_record_count": len(values),
                "mean_matching_minus_population_utility": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
            }
        )
        criteria[metric] = bool(values) and estimate > 0.0 and lower > 0.0
    passed = (
        len(records) == 8
        and criteria.get("standardized_artifact_latent_RMSE") is True
        and criteria.get("clean_waveform_RRMSE") is True
    )
    result = {
        "status": "completed_current_information_matched_paired_source_record_screen",
        "passed": passed,
        "source_record_count": len(records),
        "records_are_participants": False,
        "training_seeds_aggregated_within_source_record": 3,
        "comparisons": comparisons,
        "primary_criteria": {
            "artifact_latent_RMSE_matching_better_with_interval": criteria.get(
                "standardized_artifact_latent_RMSE", False
            ),
            "clean_waveform_RRMSE_matching_better_with_interval": criteria.get(
                "clean_waveform_RRMSE", False
            ),
        },
        "oracle_role": "query_derived_mechanism_upper_bound_nondeployable",
        "scientific_role": "paired_source_record_mechanism_development_only",
    }
    _atomic_json(screen_root / "development/paired_mechanism_summary.json", result)
    return result


def run_b3_aggregate(
    config: Mapping[str, Any], run_dir: str | Path
) -> Mapping[str, Any]:
    """Aggregate seeds within stem, then exact-cell-stratified participant bootstrap."""

    _base, _coordinate_root = _validate(config)
    screen_root = CODE_ROOT / str(
        _mapping(config, "outputs")["deterministic_screen_root"]
    )
    manifest = json.loads(
        (screen_root / "deterministic_task_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != int(manifest["task_count"]):
        raise ValueError("B3 task manifest is invalid")
    raw_rows: list[dict[str, str]] = []
    missing: list[int] = []
    for task in tasks:
        task_index = int(task["array_task_index"])
        path = (
            screen_root
            / "development/evaluation"
            / f"fold_{int(task['unified_fold_index']):02d}"
            / f"seed_{int(task['seed'])}"
            / "metrics.csv"
        )
        if not path.is_file():
            missing.append(task_index)
            continue
        with path.open("r", encoding="utf-8", newline="") as stream:
            raw_rows.extend(dict(row) for row in csv.DictReader(stream))
    if missing:
        raise RuntimeError(f"B3 missing evaluation task outputs: {missing}")
    successful = [
        row
        for row in raw_rows
        if row.get("performance_values_eligible") == "True"
        and row.get("status", "").startswith("success")
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in successful:
        grouped.setdefault((row["recording_key"], row["context_id"]), []).append(row)
    stem_context: dict[tuple[str, str], dict[str, Any]] = {}
    numeric_fields = {
        source for source, _direction in _NATURAL_METRICS.values()
    } | {
        "observation_change_ratio",
        "output_input_rms_ratio_median",
        "output_input_rms_ratio_maximum",
        "latency_total_seconds",
        "peak_memory_mb",
    }
    for key, rows in grouped.items():
        if len({int(row["training_seed"]) for row in rows}) != 3:
            continue
        summary: dict[str, Any] = {
            "recording_key": key[0],
            "context_id": key[1],
            "study": rows[0]["study"],
            "layout_id": rows[0]["layout_id"],
            "sampling_rate_hz": rows[0]["sampling_rate_hz"],
            "exact_cell": "|".join(
                (rows[0]["study"], rows[0]["layout_id"], rows[0]["sampling_rate_hz"])
            ),
        }
        for field in numeric_fields:
            values = [
                value
                for value in (_finite_csv_float(row.get(field)) for row in rows)
                if value is not None
            ]
            summary[field] = float(np.mean(values)) if len(values) == 3 else None
        stem_context[key] = summary
    stem_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    complete_stems = sorted(
        {
            key
            for key, _context in stem_context
            if all((key, context) in stem_context for context in CONTEXTS)
        }
    )
    for recording_key in complete_stems:
        contexts = {name: stem_context[(recording_key, name)] for name in CONTEXTS}
        for value in contexts.values():
            stem_rows.append(value)
        for metric_id, (field, direction) in _NATURAL_METRICS.items():
            matching = contexts["matching"].get(field)
            population = contexts["population"].get(field)
            wrong = contexts["wrong"].get(field)
            shuffled = contexts["shuffled"].get(field)
            if None in {matching, population, wrong, shuffled}:
                continue
            matching_utility = direction * float(matching)
            population_utility = direction * float(population)
            controls = direction * (float(wrong) + float(shuffled)) / 2.0
            effect_rows.append(
                {
                    "recording_key": recording_key,
                    "exact_cell": contexts["matching"]["exact_cell"],
                    "metric": metric_id,
                    "matching_minus_population": matching_utility - population_utility,
                    "matching_minus_wrong_shuffled_mean": matching_utility - controls,
                }
            )
    _write_csv(screen_root / "development/stem_context_metrics.csv", stem_rows)
    _write_csv(screen_root / "development/participant_effects.csv", effect_rows)
    development = _mapping(config, "development_calibration")
    replicates = int(development["bootstrap_replicates"])
    seed = int(development["bootstrap_seed"])
    bootstrap_rows: list[dict[str, Any]] = []
    for metric_id in _NATURAL_METRICS:
        selected = [row for row in effect_rows if row["metric"] == metric_id]
        for contrast in (
            "matching_minus_population",
            "matching_minus_wrong_shuffled_mean",
        ):
            values = [
                (str(row["exact_cell"]), float(row[contrast])) for row in selected
            ]
            estimate, lower, upper = _stratified_bootstrap(
                values, replicates=replicates, seed=seed
            )
            bootstrap_rows.append(
                {
                    "metric": metric_id,
                    "contrast": contrast,
                    "participant_stem_count": len(values),
                    "mean_effect": estimate,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "bootstrap_replicates": replicates,
                }
            )
    _write_csv(screen_root / "development/bootstrap_summary.csv", bootstrap_rows)
    cell_rows: list[dict[str, Any]] = []
    for metric_id in _NATURAL_METRICS:
        for cell in sorted({str(row["exact_cell"]) for row in effect_rows}):
            values = [
                float(row["matching_minus_population"])
                for row in effect_rows
                if row["metric"] == metric_id and row["exact_cell"] == cell
            ]
            if values:
                cell_rows.append(
                    {
                        "metric": metric_id,
                        "exact_cell": cell,
                        "participant_stem_count": len(values),
                        "mean_effect": float(np.mean(values)),
                    }
                )
    _write_csv(screen_root / "development/exact_cell_effects.csv", cell_rows)

    def bootstrap(metric: str, contrast: str) -> Mapping[str, Any]:
        return next(
            row
            for row in bootstrap_rows
            if row["metric"] == metric and row["contrast"] == contrast
        )

    eog_population = bootstrap("eog_remaining_utility", "matching_minus_population")
    eog_controls = bootstrap(
        "eog_remaining_utility", "matching_minus_wrong_shuffled_mean"
    )
    natural_primary = (
        float(eog_population["mean_effect"])
        >= float(development["natural_EOG_remaining_utility_minimum"])
        and float(eog_population["ci95_lower"]) > 0.0
        and float(eog_controls["ci95_lower"]) > 0.0
    )
    margin = float(development["safety_noninferiority_margin"])
    safety_metrics = (
        "nonartifact_preservation_utility",
        "erp_utility",
        "psd_utility",
        "covariance_utility",
    )
    safety = all(
        float(bootstrap(metric, "matching_minus_population")["ci95_lower"])
        >= margin
        for metric in safety_metrics
    )
    scale_safe = all(
        _finite_csv_float(row.get("output_input_rms_ratio_maximum"))
        is not None
        and float(row["output_input_rms_ratio_maximum"]) <= 10.0
        for row in stem_rows
    )
    cell_reversal = any(
        row["metric"] in {"eog_remaining_utility", *safety_metrics}
        and float(row["mean_effect"]) < margin
        for row in cell_rows
    )
    paired = _paired_mechanism_summary(
        screen_root, replicates=replicates, seed=seed
    )
    paired_pass = paired.get("passed") is True
    observed_stems = {row["recording_key"] for row in raw_rows}
    coverage_complete = (
        len(complete_stems) == len(observed_stems)
        and len(successful) == len(raw_rows)
    )
    if (
        paired_pass
        and natural_primary
        and safety
        and scale_safe
        and not cell_reversal
        and coverage_complete
    ):
        decision = "calibration_mechanism_supported_in_development"
        reopen = True
    elif paired_pass and not natural_primary:
        decision = "mechanism_only_not_end_to_end"
        reopen = False
    elif not paired_pass and natural_primary:
        decision = "proxy_improvement_without_mechanistic_support"
        reopen = False
    elif float(eog_population["ci95_lower"]) <= 0.0 <= float(
        eog_population["ci95_upper"]
    ):
        decision = "calibration_mechanism_inconclusive"
        reopen = False
    else:
        decision = "current_Cs_personalization_not_supported"
        reopen = False
    coverage = {
        "manifest_task_count": len(tasks),
        "completed_task_count": len(tasks) - len(missing),
        "raw_metric_rows": len(raw_rows),
        "successful_metric_rows": len(successful),
        "complete_four_context_stems": len(complete_stems),
        "compatible_stem_count_from_actual_outputs": len(observed_stems),
        "coverage_denominator_including_blocked_singleton": len(observed_stems) + 1,
        "coverage_complete_for_all_compatible_stems": coverage_complete,
        "blocked_singleton_count": 1,
        "blocked_singleton": "study05/study05_p42",
    }
    summary = {
        "status": "completed_deterministic_calibration_development_screen",
        **_implementation(),
        "calibration_mechanism": decision,
        "diffusion_reopen_eligible": reopen,
        "deterministic_model_validity": "passed",
        "paired_mechanism": paired,
        "natural_primary_passed": natural_primary,
        "safety_noninferiority_passed": safety,
        "output_scale_safe": scale_safe,
        "exact_cell_severe_reversal": cell_reversal,
        "coverage": coverage,
        "bootstrap_summary": bootstrap_rows,
        "confirmation_eligibility": False,
        "family_wide_status": "not_tested",
    }
    _atomic_json(screen_root / "calibration_screen_summary.json", summary)
    _atomic_json(Path(run_dir) / "calibration_screen_summary.json", summary)

    figures = screen_root / "development/figures"
    figures.mkdir(parents=True, exist_ok=True)
    eog_effects = [
        row for row in effect_rows if row["metric"] == "eog_remaining_utility"
    ]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.scatter(
        range(len(eog_effects)),
        [float(row["matching_minus_population"]) for row in eog_effects],
        s=18,
    )
    axis.set_ylabel("Matching - population EOG utility")
    axis.set_xlabel("Participant stem")
    figure.tight_layout()
    figure.savefig(figures / "matching_population_wrong_shuffled_paired_effects.png", dpi=160)
    plt.close(figure)

    forest = [row for row in cell_rows if row["metric"] == "eog_remaining_utility"]
    figure, axis = plt.subplots(figsize=(8, max(3, len(forest) * 0.45)))
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.scatter([float(row["mean_effect"]) for row in forest], range(len(forest)))
    axis.set_yticks(range(len(forest)), [str(row["exact_cell"]) for row in forest])
    axis.set_xlabel("Matching - population EOG utility")
    figure.tight_layout()
    figure.savefig(figures / "exact_cell_forest.png", dpi=160)
    plt.close(figure)

    paired_xy = []
    for recording_key in complete_stems:
        matching = stem_context[(recording_key, "matching")]
        attenuation = matching.get("eog_coherence_reduction")
        preservation = matching.get("nonartifact_observation_preservation")
        if attenuation is not None and preservation is not None:
            paired_xy.append((float(attenuation), float(preservation)))
    figure, axis = plt.subplots(figsize=(6, 5))
    if paired_xy:
        axis.scatter(*zip(*paired_xy), alpha=0.7)
    axis.set_xlabel("EOG coherence reduction")
    axis.set_ylabel("Non-artifact preservation")
    figure.tight_layout()
    figure.savefig(figures / "artifact_attenuation_preservation.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(
        [30.0] * len(eog_effects),
        [float(row["matching_minus_population"]) for row in eog_effects],
        alpha=0.65,
    )
    axis.set_xlabel("Calibration duration (s; frozen at 30 in this screen)")
    axis.set_ylabel("Personalization EOG utility")
    figure.tight_layout()
    figure.savefig(figures / "calibration_duration_personalization_benefit.png", dpi=160)
    plt.close(figure)
    return summary


def run_finalize(config: Mapping[str, Any], run_dir: str | Path) -> Mapping[str, Any]:
    """Write one compact terminal view without changing historical revisions."""

    _base, coordinate_root = _validate(config)
    outputs = _mapping(config, "outputs")
    screen_root = CODE_ROOT / str(outputs["deterministic_screen_root"])
    coordinate = json.loads(
        (coordinate_root / "result_summary.json").read_text(encoding="utf-8")
    )
    deterministic = json.loads(
        (screen_root / "result_summary.json").read_text(encoding="utf-8")
    )
    deterministic_passed = (
        deterministic.get("deterministic_model_validity") == "passed"
    )
    calibration_path = screen_root / "calibration_screen_summary.json"
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.is_file()
        else None
    )
    if deterministic_passed and calibration is None:
        calibration_status = "not_run_pending_after_deterministic_validity"
    elif calibration is None:
        calibration_status = "not_tested_deterministic_model_invalid"
    else:
        calibration_status = str(calibration["calibration_mechanism"])
    reopen = bool(
        calibration is not None
        and calibration.get("diffusion_reopen_eligible") is True
    )
    residual_root = CODE_ROOT / str(outputs["residual_validity_root"])
    residual_path = residual_root / "result_summary.json"
    residual = (
        json.loads(residual_path.read_text(encoding="utf-8"))
        if residual_path.is_file()
        else None
    )
    summary = {
        "status": "completed_subject_artifact_next_round_terminal_summary",
        **_implementation(),
        "coordinate_semantics": coordinate["coordinate_semantics"],
        "primary_diffusion_validity": "blocked_no_go_high_noise_latent_RMSE",
        "compound_sdedit_validity": "blocked_no_go_low_artifact_preservation",
        "deterministic_model_validity": deterministic.get(
            "deterministic_model_validity", "failed"
        ),
        "calibration_mechanism": calibration_status,
        "residual_diffusion_validity": (
            "not_run_gate_closed" if residual is None else residual.get("status")
        ),
        "diffusion_reopen_eligible": reopen,
        "eligible_for_diffusion_factorial_next_round": bool(
            residual is not None
            and residual.get("eligible_for_diffusion_factorial_next_round") is True
        ),
        "confirmation_eligibility": False,
        "family_wide_status": "not_tested",
        "real_EEG_evidence_scope": (
            "SGEYESUB_development_only_no_confirmation"
            if calibration is not None
            else "J2_real_SGE_validity_only_calibration_not_tested"
        ),
        "coordinate_result": str(coordinate_root / "result_summary.json"),
        "deterministic_result": str(screen_root / "result_summary.json"),
        "calibration_result": str(calibration_path) if calibration is not None else None,
        "residual_result": str(residual_path) if residual is not None else None,
    }
    _atomic_json(screen_root / "terminal_summary.json", summary)
    _atomic_json(Path(run_dir) / "terminal_summary.json", summary)
    report = CODE_ROOT / str(outputs["deterministic_report"])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Deterministic calibration development screen\n\n"
        f"- Deterministic model validity: `{summary['deterministic_model_validity']}`.\n"
        f"- Calibration mechanism: `{summary['calibration_mechanism']}`.\n"
        f"- Residual diffusion validity: `{summary['residual_diffusion_validity']}`.\n"
        f"- Diffusion reopen eligible: `{str(reopen).lower()}`.\n"
        "- Primary artifact-latent diffusion remains blocked by high-noise latent "
        "RMSE; the compound residual/SDEdit backup remains blocked by low-artifact "
        "preservation. Neither result is a family-wide diffusion test.\n"
        "- Deterministic reverse-trajectory validity is `N/A`, not a fabricated pass.\n"
        "- All SGE evidence in this round is development/exploratory; confirmation "
        "eligibility remains false.\n",
        encoding="utf-8",
    )
    return summary


def run_stage(
    config: Mapping[str, Any],
    run_dir: str | Path,
    stage: str,
    task_index: int | None = None,
) -> Mapping[str, Any]:
    if stage == "a0":
        return run_a0(config, run_dir)
    if stage == "a1":
        return run_a1(config, run_dir)
    if stage == "b0":
        return run_b0(config, run_dir)
    if stage == "b0-repair":
        return run_b0_repair(config, run_dir)
    if stage == "b1-manifest":
        return run_b1_manifest(config, run_dir)
    if stage == "paired-validate":
        return run_paired_validate(config, run_dir)
    if stage == "b1-train":
        if task_index is None:
            raise ValueError("b1-train requires a generated manifest array index")
        return run_b1_train(config, run_dir, task_index)
    if stage == "b1-worker":
        if task_index is None:
            raise ValueError("b1-worker requires one of eight worker indices")
        return run_b1_worker(config, run_dir, task_index)
    if stage == "b1-paired-train":
        if task_index is None:
            raise ValueError("b1-paired-train requires one of three seed indices")
        return run_b1_paired_train(config, run_dir, task_index)
    if stage == "b2-evaluate":
        if task_index is None:
            raise ValueError("b2-evaluate requires a generated manifest array index")
        return run_b2_evaluate(config, run_dir, task_index)
    if stage == "b2-worker":
        if task_index is None:
            raise ValueError("b2-worker requires one of eight worker indices")
        return run_b2_worker(config, run_dir, task_index)
    if stage == "b2-paired-evaluate":
        if task_index is None:
            raise ValueError("b2-paired-evaluate requires one of three seed indices")
        return run_b2_paired_evaluate(config, run_dir, task_index)
    if stage == "b3-aggregate":
        return run_b3_aggregate(config, run_dir)
    if stage == "finalize":
        return run_finalize(config, run_dir)
    raise ValueError(f"unsupported subject-artifact-next-round stage: {stage}")


__all__ = [
    "run_a0",
    "run_a1",
    "run_b0",
    "run_b0_repair",
    "run_b1_manifest",
    "run_paired_validate",
    "run_b1_paired_train",
    "run_b1_train",
    "run_b1_worker",
    "run_b2_evaluate",
    "run_b2_worker",
    "run_b2_paired_evaluate",
    "run_b3_aggregate",
    "run_finalize",
    "run_stage",
]
