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

from eeg_cgdr.experiments.subject_artifact_data import prepare_subject_artifact_fold
from eeg_cgdr.experiments.subject_artifact_training import (
    CheckpointableEMA,
    _masked_rmse,
    _predicted_x0,
    load_artifact_training_checkpoint,
)
from eeg_cgdr.experiments.subject_artifact_validity import (
    evaluate_v0,
    evaluate_v2,
    evaluate_v3,
)
from eeg_cgdr.experiments.subject_artifact_validity_runner import (
    _ObservationAnchoredSDEdit,
    _combine_v0_results,
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
from eeg_cgdr.models.artifact_latent_deterministic import DeterministicArtifactEstimator
from eeg_cgdr.models.artifact_latent_diffusion import ArtifactLatentDiffusion


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
        v0 = _combine_v0_results(results)
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
            "V0": primary_result["V0"],
            "V1_historical_status": _mapping(_mapping(old_primary, "validity"), "V1"),
            "V2": primary_result["V2"],
            "V3": primary_result["V3"],
            "eligibility": "blocked",
            "no_go_reason_retained": "high_noise_standardized_latent_RMSE",
        },
        "compound_residual_sdedit_backup": {
            "V0": compound_result["V0"],
            "V1_historical_status": _mapping(_mapping(old_compound, "validity"), "V1"),
            "V2": compound_result["V2"],
            "V3": compound_result["V3"],
            "eligibility": "blocked",
            "no_go_reason_retained": "low_artifact_preservation",
        },
        "deterministic_checkpoint": deterministic_result,
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
    return summary


def run_stage(
    config: Mapping[str, Any], run_dir: str | Path, stage: str
) -> Mapping[str, Any]:
    if stage == "a0":
        return run_a0(config, run_dir)
    if stage == "a1":
        return run_a1(config, run_dir)
    raise ValueError(f"unsupported subject-artifact-next-round stage: {stage}")


__all__ = ["run_a0", "run_a1", "run_stage"]
