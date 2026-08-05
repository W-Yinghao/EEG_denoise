"""Execution adapter for the frozen SGEYESUB natural-EEG comparison.

Scientific definitions live in :mod:`eeg_cgdr.experiments.sgeyesub_diffusion`.
This module only connects those definitions to the real loader, two existing
models, atomic checkpoints and per-fold Slurm entry points.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import TensorDataset

from eeg_cgdr.data.sgeyesub import (
    SgeyesubLayout,
    SgeyesubLoadedRecord,
    SgeyesubReleaseRecord,
    build_sgeyesub_protocol,
    load_sgeyesub_signal_record,
    load_sgeyesub_structure_audit,
)
from eeg_cgdr.experiments.sgeyesub_diffusion import (
    CONDITIONAL_METHOD_ID,
    DETERMINISTIC_METHOD_ID,
    REPORTED_ARM_IDS,
    FullBlockP0Fit,
    FrozenSgeyesubFold,
    OuterTrainingPopulationP0Fit,
    SgeyesubFoldNormalizer,
    WeakSupervisionBundle,
    build_frozen_sgeyesub_folds,
    build_within_stem_weak_pairs,
    eeg_only_frame_attenuation,
    fit_full_block1_p0,
    fit_outer_training_normalizer,
    fit_outer_training_population_p0,
    fit_outer_training_projected_energy_scale,
    freeze_query_arm_outputs,
    matching_soft_proximal,
    sgeyesub_p0_config,
    trial_local_nonoverlap_windows,
    validate_sgeyesub_diffusion_config,
    write_sgeyesub_diffusion_aggregate,
)
from eeg_cgdr.experiments.sgeyesub_operator_specificity import _evaluate_output
from eeg_cgdr.models.conditional_diffusion import OperatorConditionedEEGDiffusion
from eeg_cgdr.models.deterministic_unet import (
    DeterministicUNetConfig,
    TaskMatchedDeterministicUNet,
)
from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
    scaler_optimizer_step_succeeded,
)
from saddpm.diffusion.schedule import DiffusionConfig


PROTOCOL_ID = "sgeyesub_natural_eeg_diffusion_incremental_v1"
STRUCTURE_AUDIT = Path(
    os.environ.get(
        "DENOISENET_SGE_STRUCTURE_AUDIT",
        "reports/dataset_harness/jobs/919218/attempt-0/result.json",
    )
)
SEED = 20260802
UPDATE_TARGET = 6000
ARMS = tuple(REPORTED_ARM_IDS)
IMPLEMENTATION_VERSION = "sgeyesub_diffusion_runner_v2"
_GIT_HEAD_ENVIRONMENT_VARIABLE = "DENOISENET_GIT_HEAD"
_DOWNSTREAM_TASK_NOT_REGISTERED = (
    "N/A_release_internal_condition_labels_support_ERP_proxy_only_"
    "no_registered_downstream_task"
)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _implementation_identity() -> dict[str, str]:
    git_head = os.environ.get(_GIT_HEAD_ENVIRONMENT_VARIABLE, "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", git_head) is None:
        raise RuntimeError(
            f"{_GIT_HEAD_ENVIRONMENT_VARIABLE} must contain the scheduled 40-hex Git HEAD"
        )
    return {
        "implementation_version": IMPLEMENTATION_VERSION,
        "git_head": git_head.lower(),
    }


def _require_completed_development_aggregate(
    config: Mapping[str, Any], expected_identity: Mapping[str, str]
) -> dict[str, Any]:
    """Fail closed before evaluation unless current development completed."""

    development_root = Path(str(_mapping(config, "outputs")["development_root"]))
    summary_path = development_root / "result_summary.json"
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "evaluation requires a readable completed development aggregate"
        ) from error
    if not isinstance(loaded, dict):
        raise RuntimeError("development aggregate summary must be a JSON object")
    required = {
        "protocol_id": PROTOCOL_ID,
        "status": "completed_development_aggregate",
        "partition": "development",
        **dict(expected_identity),
    }
    mismatched = [key for key, value in required.items() if loaded.get(key) != value]
    if mismatched:
        raise RuntimeError(
            "evaluation requires completed development aggregate from the current "
            f"implementation; mismatched fields: {', '.join(mismatched)}"
        )
    return loaded


def _validate_completed_fold_for_aggregate(
    summary: Mapping[str, Any],
    *,
    fold_id: str,
    expected_identity: Mapping[str, str],
) -> None:
    """Require each canonical fold to be complete and from this implementation."""

    if summary.get("status") != "completed_fold" or summary.get("fold_id") != fold_id:
        raise ValueError(f"fold is not terminal-complete: {fold_id}")
    if summary.get("exact_shared_minibatch_sequence_verified") is not True:
        raise ValueError(f"fold lacks exact shared minibatch audit: {fold_id}")
    mismatched = [
        key for key, value in expected_identity.items() if summary.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            f"fold implementation identity mismatch: {fold_id}: "
            f"{', '.join(mismatched)}"
        )


def _metric_contract_fields(status: str, *, inference_seed: int) -> dict[str, Any]:
    return {
        "downstream_task_preservation_when_label_semantics_allow": (
            _DOWNSTREAM_TASK_NOT_REGISTERED
        ),
        "failure_status": "" if str(status).startswith("success") else str(status),
        "inference_seed": int(inference_seed),
    }


def _inference_failure_status(error: BaseException) -> str:
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if (
        cuda_oom_type is not None
        and isinstance(error, cuda_oom_type)
    ) or "out of memory" in str(error).lower():
        return "failed_inference_cuda_oom"
    if isinstance(error, FloatingPointError):
        return "failed_inference_nonfinite"
    if isinstance(error, ValueError):
        return "failed_inference_value_error"
    return "failed_inference_runtime_error"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(str(field))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path, *, typed: bool = True) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows: list[dict[str, Any]] = [dict(row) for row in csv.DictReader(stream)]
    if not typed:
        return rows

    def convert(value: str) -> Any:
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if value == "":
            return ""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    return [{key: convert(value) for key, value in row.items()} for row in rows]


def _save_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")


def _protocol_contract(
    config: Mapping[str, Any],
) -> tuple[
    dict[str, SgeyesubLayout],
    dict[str, SgeyesubReleaseRecord],
    tuple[FrozenSgeyesubFold, ...],
]:
    validate_sgeyesub_diffusion_config(config)
    layouts, records = load_sgeyesub_structure_audit(STRUCTURE_AUDIT)
    plan = build_sgeyesub_protocol(
        layouts,
        records,
        protocol_id=PROTOCOL_ID,
        reference_cell_id="release_preprocessed_as_delivered",
        gamma_candidates=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    split = build_frozen_sgeyesub_folds(config, plan.rows)
    return (
        {value.layout_id: value for value in layouts},
        {value.recording_key: value for value in records},
        split.folds,
    )


def _select_fold(
    folds: Sequence[FrozenSgeyesubFold], partition: str, task_index: int
) -> FrozenSgeyesubFold:
    selected = tuple(value for value in folds if value.partition == partition)
    if partition not in {"development", "evaluation"}:
        raise ValueError("partition must be development or evaluation")
    if isinstance(task_index, bool) or not 0 <= int(task_index) < len(selected):
        raise ValueError("task index lies outside the frozen fold array")
    return selected[int(task_index)]


def _combine_weak_bundles(bundles: Sequence[WeakSupervisionBundle]) -> WeakSupervisionBundle:
    if not bundles:
        raise RuntimeError("no complete within-stem weak supervision bundle exists")
    return WeakSupervisionBundle(
        observed=np.concatenate([value.observed for value in bundles]),
        weak_target=np.concatenate([value.weak_target for value in bundles]),
        projector=np.concatenate([value.projector for value in bundles]),
        attenuation=np.concatenate([value.attenuation for value in bundles]),
        valid_time_mask=np.concatenate([value.valid_time_mask for value in bundles]),
        recording_keys=tuple(key for value in bundles for key in value.recording_keys),
        target_origins=tuple(origin for value in bundles for origin in value.target_origins),
        artifact_origins=tuple(origin for value in bundles for origin in value.artifact_origins),
    )


@dataclass(frozen=True)
class _PreparedFold:
    fold: FrozenSgeyesubFold
    layouts: Mapping[str, SgeyesubLayout]
    records: Mapping[str, SgeyesubReleaseRecord]
    training: Mapping[str, SgeyesubLoadedRecord]
    heldout: Mapping[str, SgeyesubLoadedRecord]
    normalizer: SgeyesubFoldNormalizer
    population: OuterTrainingPopulationP0Fit
    matching: Mapping[str, FullBlockP0Fit]
    pairs: WeakSupervisionBundle
    projected_energy_scale: float


def _prepare_fold(
    config: Mapping[str, Any], partition: str, task_index: int
) -> _PreparedFold:
    layouts, records, folds = _protocol_contract(config)
    fold = _select_fold(folds, partition, task_index)
    root = Path(str(_mapping(config, "dataset")["data_root"]))
    training = {
        key: load_sgeyesub_signal_record(
            root, records[key], layouts[records[key].layout_id], include_query=False
        )
        for key in fold.training_recording_keys
    }
    heldout = {
        key: load_sgeyesub_signal_record(
            root,
            records[key],
            layouts[records[key].layout_id],
            include_query=True,
            include_query_annotations=False,
        )
        for key in fold.heldout_recording_keys
    }
    if set(training) & set(heldout) or any(
        value.query_annotations is not None for value in heldout.values()
    ):
        raise AssertionError("outer-fold leakage or premature query annotation access")
    if any(
        value.support.eeg.shape[0] != fold.eeg_channels
        or value.sampling_rate_hz != fold.sampling_rate_hz
        for value in (*training.values(), *heldout.values())
    ):
        raise ValueError("loaded fold signal differs from frozen channel/rate cell")
    support_eeg = {key: value.support.eeg for key, value in training.items()}
    normalizer = fit_outer_training_normalizer(
        support_eeg, fold.training_recording_keys
    )
    normalized = {key: normalizer.transform(value) for key, value in support_eeg.items()}
    p0_config, movement_threshold = sgeyesub_p0_config(config)
    training_p0 = {
        key: fit_full_block1_p0(
            normalized[key],
            training[key].support.external_eog,
            recording_key=key,
            sampling_rate_hz=fold.sampling_rate_hz,
            config=p0_config,
            movement_threshold=movement_threshold,
        )
        for key in fold.training_recording_keys
    }
    if any(
        value.outcome.status != "eligible" or value.outcome.transfer is None
        for value in training_p0.values()
    ):
        failed = [
            key for key, value in training_p0.items() if value.outcome.status != "eligible"
        ]
        raise RuntimeError(f"outer-training full-block P0 ineligible: {failed}")
    population = fit_outer_training_population_p0(
        normalized,
        {key: training[key].support.external_eog for key in training},
        fold.training_recording_keys,
        sampling_rate_hz=fold.sampling_rate_hz,
        p0_config=p0_config,
        movement_threshold=movement_threshold,
    )
    if population.outcome.status != "eligible" or population.outcome.transfer is None:
        raise RuntimeError("same-cell outer-training population P0 is ineligible")
    energy_scale = fit_outer_training_projected_energy_scale(
        [
            (
                normalized[key],
                training_p0[key].outcome.transfer.projector,  # type: ignore[union-attr]
            )
            for key in fold.training_recording_keys
        ]
    )
    bundles = [
        build_within_stem_weak_pairs(
            normalized[key],
            training[key].support.external_eog,
            training[key].support.artifactclasses,
            transfer=training_p0[key].outcome.transfer,  # type: ignore[arg-type]
            samples_per_trial=records[key].samples_per_trial,
            sampling_rate_hz=fold.sampling_rate_hz,
            recording_key=key,
            projected_energy_scale=energy_scale,
            seed=SEED,
        )
        for key in fold.training_recording_keys
    ]
    matching = {
        key: fit_full_block1_p0(
            normalizer.transform(heldout[key].support.eeg),
            heldout[key].support.external_eog,
            recording_key=key,
            sampling_rate_hz=fold.sampling_rate_hz,
            config=p0_config,
            movement_threshold=movement_threshold,
        )
        for key in fold.heldout_recording_keys
    }
    return _PreparedFold(
        fold=fold,
        layouts=layouts,
        records=records,
        training=training,
        heldout=heldout,
        normalizer=normalizer,
        population=population,
        matching=matching,
        pairs=_combine_weak_bundles(bundles),
        projected_energy_scale=energy_scale,
    )


def _dataset(bundle: WeakSupervisionBundle) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.asarray(bundle.observed, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.weak_target, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.projector, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.attenuation, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.valid_time_mask, dtype=bool)),
    )


def _pair_keys(bundle: WeakSupervisionBundle) -> tuple[str, ...]:
    return tuple(
        f"{key}:target_t{target.trial_ordinal}_{target.start_sample}:"
        f"source_t{source.trial_ordinal}_{source.start_sample}"
        for key, target, source in zip(
            bundle.recording_keys,
            bundle.target_origins,
            bundle.artifact_origins,
        )
    )


def _minibatch_schedule(pair_count: int, batch_size: int) -> tuple[np.ndarray, ...]:
    if pair_count < 1 or batch_size < 1:
        raise ValueError("minibatch schedule dimensions must be positive")
    batches: list[np.ndarray] = []
    epoch = 0
    while len(batches) < UPDATE_TARGET:
        order = np.random.default_rng(SEED + epoch).permutation(pair_count)
        for start in range(0, pair_count, batch_size):
            batches.append(np.asarray(order[start : start + batch_size], dtype=np.int64))
            if len(batches) == UPDATE_TARGET:
                break
        epoch += 1
    return tuple(batches)


def _schedule_rows(
    schedule: Sequence[np.ndarray], pair_keys: Sequence[str]
) -> list[dict[str, Any]]:
    return [
        {
            "successful_update": index + 1,
            "pair_indices": ";".join(str(int(value)) for value in batch),
            "pair_keys": ";".join(pair_keys[int(value)] for value in batch),
        }
        for index, batch in enumerate(schedule)
    ]


def _persist_schedule(
    path: Path, schedule: Sequence[np.ndarray], pair_keys: Sequence[str]
) -> None:
    rows = _schedule_rows(schedule, pair_keys)
    expected = [{key: str(value) for key, value in row.items()} for row in rows]
    if path.is_file():
        if _read_csv(path, typed=False) != expected:
            raise ValueError("stored exact minibatch sequence changed")
    else:
        _write_csv(path, rows)


def _backbone(config: Mapping[str, Any], prepared: _PreparedFold) -> DeterministicUNetConfig:
    matched = _mapping(config, "matched_comparison")
    samples = int(
        _mapping(config, "windowing")["per_study_window_samples"][prepared.fold.study]
    )
    return DeterministicUNetConfig(
        eeg_channels=prepared.fold.eeg_channels,
        signal_length=samples,
        base_channels=int(matched["model_width"]),
    )


def _diffusion(config: Mapping[str, Any]) -> DiffusionConfig:
    raw = _mapping(_mapping(config, "matched_comparison"), "conditional_diffusion")
    return DiffusionConfig(
        num_timesteps=int(raw["num_diffusion_timesteps"]),
        beta_start=float(raw["beta_start"]),
        beta_end=float(raw["beta_end"]),
        schedule=str(raw["beta_schedule"]),
    )


def _model(
    method_id: str, config: Mapping[str, Any], prepared: _PreparedFold
) -> torch.nn.Module:
    backbone = _backbone(config, prepared)
    if method_id == DETERMINISTIC_METHOD_ID:
        return TaskMatchedDeterministicUNet(backbone)
    if method_id == CONDITIONAL_METHOD_ID:
        return OperatorConditionedEEGDiffusion(backbone, _diffusion(config))
    raise ValueError(f"unknown learned arm: {method_id}")


def _checkpoint_paths(
    config: Mapping[str, Any], partition: str, fold_id: str, method_id: str
) -> tuple[Path, Path]:
    root = Path(str(_mapping(config, "outputs")["root"]))
    parent = root / "checkpoints" / partition / fold_id / method_id
    return parent / "last.pt", parent / "final.pt"


def _checkpoint_contract(
    config: Mapping[str, Any],
    prepared: _PreparedFold,
    method_id: str,
) -> dict[str, Any]:
    matched = _mapping(config, "matched_comparison")
    return {
        **_implementation_identity(),
        "protocol_id": PROTOCOL_ID,
        "partition": prepared.fold.partition,
        "fold_id": prepared.fold.fold_id,
        "method_id": method_id,
        "training_recording_keys": list(prepared.fold.training_recording_keys),
        "heldout_recording_keys": list(prepared.fold.heldout_recording_keys),
        "pair_keys": list(_pair_keys(prepared.pairs)),
        "successful_optimizer_updates_target": UPDATE_TARGET,
        "minibatch_sequence_policy": (
            "one_precomputed_exact_6000_update_index_sequence_shared_by_both_arms"
        ),
        "minibatch_sequence_seed": SEED,
        "model_seed": int(_mapping(matched, "model_seed_by_arm")[method_id]),
        "gradient_clip_norm": float(matched["gradient_clip_norm"]),
        "checkpoint_interval_successful_updates": int(
            matched["checkpoint_interval_successful_updates"]
        ),
        "checkpoint_selection": "fixed_6000_successful_update_endpoint",
        "weak_target_semantics": "low_artifact_observed_EEG_not_clean_truth",
        "query_evaluation_fields_visible": False,
        "backbone": asdict(_backbone(config, prepared)),
        "diffusion": asdict(_diffusion(config)) if method_id == CONDITIONAL_METHOD_ID else None,
    }


def _batch_loss(
    model: torch.nn.Module,
    method_id: str,
    batch: Sequence[Tensor],
    device: torch.device,
) -> Tensor:
    observed, target, projector, attenuation, mask = (
        value.to(device, non_blocking=True) for value in batch
    )
    if method_id == DETERMINISTIC_METHOD_ID:
        output = model(
            observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
        weight = mask[:, None, :].to(dtype=output.dtype)
        return ((output - target).square() * weight).sum() / (
            weight.sum() * output.shape[1]
        ).clamp_min(1.0)
    return model.training_loss(
        target,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
    )


def _amp_optimizer_update(
    model: torch.nn.Module,
    method_id: str,
    batch: Sequence[Tensor],
    *,
    device: torch.device,
    optimizer: AdamW,
    scaler: Any,
    amp: bool,
    gradient_clip_norm: float,
) -> Tensor:
    """Run the exact autocast/scaler/unscale/clip update used by both routes."""

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        loss = _batch_loss(model, method_id, batch, device)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"non-finite {method_id} loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not scaler_optimizer_step_succeeded(scaler, optimizer):
        raise FloatingPointError("AMP overflow skipped the optimizer update")
    return loss.detach()


@dataclass(frozen=True)
class _Endpoint:
    fold_id: str
    method_id: str
    status: str
    successful_optimizer_updates: int
    optimizer_step_attempts: int
    skipped_optimizer_steps: int
    parameter_count: int
    training_walltime_seconds: float
    peak_memory_mb: float
    checkpoint: str
    resumed: bool
    minibatch_sequence_updates: int
    minibatch_sequence_verified: bool


def _train(
    config: Mapping[str, Any],
    prepared: _PreparedFold,
    method_id: str,
    schedule: Sequence[np.ndarray],
    *,
    device: torch.device,
    stop_requested: Callable[[], bool],
) -> tuple[torch.nn.Module | None, _Endpoint]:
    matched = _mapping(config, "matched_comparison")
    last_path, final_path = _checkpoint_paths(
        config, prepared.fold.partition, prepared.fold.fold_id, method_id
    )
    contract = _checkpoint_contract(config, prepared, method_id)
    arm_seed = int(_mapping(matched, "model_seed_by_arm")[method_id])
    torch.manual_seed(arm_seed)
    torch.cuda.manual_seed_all(arm_seed)
    model = _model(method_id, config, prepared).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(matched["learning_rate"]),
        weight_decay=float(matched["weight_decay"]),
    )
    amp = bool(matched["mixed_precision"])
    gradient_clip = float(matched["gradient_clip_norm"])
    checkpoint_interval = int(matched["checkpoint_interval_successful_updates"])
    if gradient_clip != 1.0 or checkpoint_interval != 250:
        raise ValueError("frozen gradient-clip/checkpoint interval changed")
    scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024.0)
    step = attempts = skipped = 0
    prior_walltime = 0.0
    prior_peak_memory_mb = 0.0
    resumed = False
    if final_path.is_file():
        payload = load_training_checkpoint(final_path, map_location=device)
        if payload["config"] != contract or int(payload["step"]) != UPDATE_TARGET:
            raise ValueError("terminal checkpoint contract changed")
        if payload["normalizer_state"] != prepared.normalizer.state_dict():
            raise ValueError("terminal checkpoint normalizer changed")
        extra = payload.get("extra", {})
        if extra.get("minibatch_sequence_verified") is not True or int(
            extra.get("minibatch_sequence_updates", -1)
        ) != UPDATE_TARGET:
            raise ValueError("terminal checkpoint lacks exact minibatch audit")
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        peak_memory_mb = max(
            float(extra.get("peak_memory_mb", 0.0)),
            float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
        )
        return model, _Endpoint(
            prepared.fold.fold_id,
            method_id,
            "success_fixed_6000_update_endpoint",
            UPDATE_TARGET,
            int(extra.get("optimizer_step_attempts", UPDATE_TARGET)),
            int(extra.get("skipped_optimizer_steps", 0)),
            sum(value.numel() for value in model.parameters()),
            float(extra.get("cumulative_walltime_seconds", 0.0)),
            peak_memory_mb,
            str(final_path.resolve()),
            True,
            UPDATE_TARGET,
            True,
        )
    if last_path.is_file():
        state = resume_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_config=contract,
            map_location=device,
        )
        if state.normalizer_state != prepared.normalizer.state_dict():
            raise ValueError("resume normalizer changed")
        if state.extra.get("minibatch_sequence_verified") is not True or int(
            state.extra.get("minibatch_sequence_updates", -1)
        ) != UPDATE_TARGET:
            raise ValueError("resume checkpoint lacks exact minibatch audit")
        step = state.step
        attempts = int(state.extra.get("optimizer_step_attempts", step))
        skipped = int(state.extra.get("skipped_optimizer_steps", 0))
        prior_walltime = float(state.extra.get("cumulative_walltime_seconds", 0.0))
        prior_peak_memory_mb = float(state.extra.get("peak_memory_mb", 0.0))
        resumed = True
    if len(schedule) != UPDATE_TARGET:
        raise ValueError("exact minibatch schedule must contain 6000 updates")
    dataset = _dataset(prepared.pairs)
    started = time.perf_counter()
    model.train()
    while step < UPDATE_TARGET and not stop_requested():
        indices = torch.as_tensor(schedule[step], dtype=torch.long)
        batch = tuple(value.index_select(0, indices) for value in dataset.tensors)
        attempts += 1
        try:
            _amp_optimizer_update(
                model,
                method_id,
                batch,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=amp,
                gradient_clip_norm=gradient_clip,
            )
            step += 1
        except FloatingPointError:
            skipped += 1
            raise
        if (
            step % checkpoint_interval == 0
            or step == UPDATE_TARGET
            or stop_requested()
        ):
            cumulative = prior_walltime + time.perf_counter() - started
            save_training_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=step,
                step=step,
                config=contract,
                normalizer=prepared.normalizer.state_dict(),
                extra={
                    "optimizer_step_attempts": attempts,
                    "skipped_optimizer_steps": skipped,
                    "cumulative_walltime_seconds": cumulative,
                    "minibatch_sequence_updates": UPDATE_TARGET,
                    "minibatch_sequence_verified": True,
                    "peak_memory_mb": max(
                        prior_peak_memory_mb,
                        float(
                            torch.cuda.max_memory_allocated(device) / (1024.0**2)
                        ),
                    ),
                },
            )
    cumulative = prior_walltime + time.perf_counter() - started
    endpoint = _Endpoint(
        prepared.fold.fold_id,
        method_id,
        (
            "success_fixed_6000_update_endpoint"
            if step == UPDATE_TARGET
            else "checkpointed_for_resume"
        ),
        step,
        attempts,
        skipped,
        sum(value.numel() for value in model.parameters()),
        cumulative,
        max(
            prior_peak_memory_mb,
            float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
        ),
        str((final_path if step == UPDATE_TARGET else last_path).resolve()),
        resumed,
        UPDATE_TARGET,
        True,
    )
    if step != UPDATE_TARGET:
        return None, endpoint
    payload = load_training_checkpoint(last_path, map_location=device)
    save_training_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=int(payload["epoch"]),
        step=UPDATE_TARGET,
        config=contract,
        normalizer=prepared.normalizer.state_dict(),
        extra=payload["extra"],
    )
    model.eval()
    return model, endpoint


@torch.no_grad()
def _infer_learned(
    model: torch.nn.Module,
    method_id: str,
    windows: np.ndarray,
    projector: np.ndarray,
    attenuation: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float, float, int, int]:
    batch_size = int(_mapping(config, "matched_comparison")["batch_size"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    outputs: list[np.ndarray] = []
    batched_forward_invocations = 0
    network_calls_per_window = 1 if method_id == DETERMINISTIC_METHOD_ID else int(
        _mapping(_mapping(config, "matched_comparison"), "conditional_diffusion")[
            "ddim_network_calls_per_window"
        ]
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for start in range(0, windows.shape[0], batch_size):
        stop = min(start + batch_size, windows.shape[0])
        y = torch.as_tensor(windows[start:stop], dtype=torch.float32, device=device)
        p = torch.as_tensor(projector, dtype=torch.float32, device=device)
        a = torch.as_tensor(attenuation[start:stop], dtype=torch.float32, device=device)
        mask = torch.as_tensor(valid_mask[start:stop], dtype=torch.bool, device=device)
        if method_id == DETERMINISTIC_METHOD_ID:
            output = model(y, projector=p, attenuation=a, valid_time_mask=mask)
            batched_forward_invocations += 1
        else:
            diffusion = _mapping(
                _mapping(config, "matched_comparison"), "conditional_diffusion"
            )
            sampled = model.sample_ddim(
                observed=y,
                projector=p,
                attenuation=a,
                valid_time_mask=mask,
                ddim_steps=int(diffusion["ddim_network_calls_per_window"]),
                eta=float(diffusion["ddim_eta"]),
                generator=generator,
            )
            output = sampled.restored
            if sampled.network_calls != network_calls_per_window:
                raise AssertionError("conditional sampler network-call contract changed")
            batched_forward_invocations += sampled.network_calls
        outputs.append(output.detach().cpu().numpy().astype(np.float64))
    torch.cuda.synchronize(device)
    return (
        np.concatenate(outputs),
        time.perf_counter() - started,
        float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
        network_calls_per_window,
        batched_forward_invocations,
    )


def _continuous(windows: np.ndarray) -> np.ndarray:
    value = np.asarray(windows)
    return np.ascontiguousarray(value.transpose(1, 0, 2).reshape(value.shape[1], -1))


def _record_seed(
    config: Mapping[str, Any], partition: str, fold_id: str, recording_key: str
) -> int:
    initial_state_seed = int(
        _mapping(_mapping(config, "matched_comparison"), "conditional_diffusion")[
            "initial_state_seed"
        ]
    )
    text = f"{partition}|{fold_id}|{recording_key}"
    return initial_state_seed + sum(
        (index + 1) * ord(value) for index, value in enumerate(text)
    )


def _evaluate_fold(
    config: Mapping[str, Any],
    prepared: _PreparedFold,
    models: Mapping[str, torch.nn.Module],
    endpoints: Mapping[str, _Endpoint],
    device: torch.device,
) -> list[dict[str, Any]]:
    population = prepared.population.outcome.transfer
    assert population is not None
    root = Path(str(_mapping(config, "dataset")["data_root"]))
    rows: list[dict[str, Any]] = []
    for recording_key in prepared.fold.heldout_recording_keys:
        record = prepared.heldout[recording_key]
        meta = prepared.records[recording_key]
        assert record.query is not None and record.query_annotations is None
        query = prepared.normalizer.transform(record.query.eeg)
        windowed = trial_local_nonoverlap_windows(
            query,
            samples_per_trial=meta.samples_per_trial,
            sampling_rate_hz=prepared.fold.sampling_rate_hz,
            recording_key=recording_key,
        )
        fit = prepared.matching[recording_key]
        matching_eligible = fit.outcome.status == "eligible" and fit.outcome.transfer is not None
        effective = fit.outcome.transfer if matching_eligible else population
        assert effective is not None
        frame_attenuation = eeg_only_frame_attenuation(
            query, effective.projector, prepared.projected_energy_scale
        )
        attenuation_windows = trial_local_nonoverlap_windows(
            frame_attenuation,
            samples_per_trial=meta.samples_per_trial,
            sampling_rate_hz=prepared.fold.sampling_rate_hz,
            recording_key=recording_key,
        ).values[:, 0, :]
        learned: dict[str, np.ndarray] = {}
        latency: dict[str, float] = {}
        memory: dict[str, float] = {}
        calls: dict[str, int] = {}
        batched_calls: dict[str, int] = {}
        learned_status: dict[str, str] = {}
        inference_seed = _record_seed(
            config,
            prepared.fold.partition,
            prepared.fold.fold_id,
            recording_key,
        )
        for method_id in (DETERMINISTIC_METHOD_ID, CONDITIONAL_METHOD_ID):
            try:
                (
                    output,
                    elapsed,
                    peak,
                    calls_per_window,
                    batched_invocations,
                ) = _infer_learned(
                    models[method_id],
                    method_id,
                    windowed.values,
                    effective.projector,
                    attenuation_windows,
                    windowed.valid_time_mask,
                    config=config,
                    device=device,
                    seed=inference_seed,
                )
                learned_status[method_id] = "success"
            except (RuntimeError, ValueError, FloatingPointError) as error:
                learned_status[method_id] = _inference_failure_status(error)
                output = np.asarray(windowed.values, dtype=np.float64).copy()
                elapsed = float("nan")
                peak = float("nan")
                calls_per_window = 0
                batched_invocations = 0
                if learned_status[method_id] == "failed_inference_cuda_oom":
                    torch.cuda.empty_cache()
            learned[method_id] = _continuous(output)
            latency[method_id] = elapsed
            memory[method_id] = peak
            calls[method_id] = calls_per_window
            batched_calls[method_id] = batched_invocations
        zero = np.zeros(query.shape[1], dtype=np.float64)
        arm_outputs = {
            "raw_observation": query,
            "population_projector_Qy": matching_soft_proximal(
                query, population.projector, zero
            ),
            "matching_projector_Qy": matching_soft_proximal(
                query, effective.projector, zero
            ),
            "matching_projector_soft_proximal": matching_soft_proximal(
                query, effective.projector, frame_attenuation
            ),
            DETERMINISTIC_METHOD_ID: learned[DETERMINISTIC_METHOD_ID],
            CONDITIONAL_METHOD_ID: learned[CONDITIONAL_METHOD_ID],
        }
        frozen = freeze_query_arm_outputs(arm_outputs, recording_key=recording_key)

        # Evaluation-only reopen happens after the core freeze assertion above.
        annotated = load_sgeyesub_signal_record(
            root,
            meta,
            prepared.layouts[meta.layout_id],
            include_query=True,
            include_query_annotations=True,
        )
        if annotated.query_annotations is None:
            raise AssertionError("query evaluation annotations were not opened")
        annotation = annotated.query_annotations
        predicted = None
        if matching_eligible:
            assert fit.outcome.transfer is not None
            standardized = (
                annotation.external_eog - fit.eog_mean
            ) / fit.eog_standard_deviation
            predicted = fit.outcome.transfer.transfer_matrix @ (
                standardized - fit.outcome.transfer.eog_mean
            )
        for method_id in ARMS:
            matching_dependent = method_id not in {
                "raw_observation",
                "population_projector_Qy",
            }
            fallback = matching_dependent and not matching_eligible
            inference_status = learned_status.get(method_id, "success")
            status = (
                inference_status
                if inference_status.startswith("failed")
                else "fallback_POP_ineligible_matching_P0"
                if fallback
                else "success"
            )
            row = _evaluate_output(
                method_id=method_id,
                output=frozen.outputs[method_id],
                observed=query,
                matching_projector=(
                    fit.outcome.transfer.projector if matching_eligible else None  # type: ignore[union-attr]
                ),
                population_projector=population.projector,
                query_eog=annotation.external_eog,
                artifactclasses=annotation.artifactclasses,
                predicted_contamination=predicted,
                trial_labels=annotation.trial_labels,
                samples_per_trial=meta.samples_per_trial,
                minimum_trials_per_condition=2,
                status=status,
                operator_source=(
                    "none"
                    if method_id == "raw_observation"
                    else "same_cell_outer_training_block1"
                    if method_id == "population_projector_Qy" or fallback
                    else "heldout_stem_block1"
                ),
                gamma=None,
                fallback_used=fallback,
                uses_query_external_eog=False,
            )
            endpoint = endpoints.get(method_id)
            compute_contract = _mapping(
                _mapping(config, "evaluation"), "compute_metric_contract"
            )
            expected_network_calls = (
                int(compute_contract["deterministic_network_evaluations_per_window"])
                if method_id == DETERMINISTIC_METHOD_ID
                else int(compute_contract["conditional_DDIM_network_evaluations_per_window"])
                if method_id == CONDITIONAL_METHOD_ID
                else 0
            )
            if not status.startswith("failed") and calls.get(
                method_id, 0
            ) != expected_network_calls:
                raise AssertionError("per-window network-evaluation report changed")
            row.update(
                {
                    "partition": prepared.fold.partition,
                    "fold_id": prepared.fold.fold_id,
                    "study": prepared.fold.study,
                    "participant_stem": record.participant_stem,
                    "recording_key": recording_key,
                    "metric_coordinate": _mapping(config, "evaluation")[
                        "metric_coordinate"
                    ],
                    "query_evaluation_fields_opened_after_all_arm_outputs_frozen": True,
                    "query_evaluation_fields_used_for_fit_selection_or_inference": False,
                    "query_eog_used_for_inference": False,
                    "query_artifactclasses_used_for_inference": False,
                    "inference_placeholder_used_for_freeze_only": status.startswith(
                        "failed"
                    ),
                    "performance_values_eligible": status.startswith("success"),
                    "latency_seconds": (
                        latency.get(method_id, 0.0) / windowed.values.shape[0]
                    ),
                    "latency_total_seconds": latency.get(method_id, 0.0),
                    "latency_seconds_per_window": (
                        latency.get(method_id, 0.0) / windowed.values.shape[0]
                    ),
                    "peak_memory_mb": memory.get(method_id, 0.0),
                    "network_calls_per_window": calls.get(method_id, 0),
                    "batched_forward_invocations": batched_calls.get(method_id, 0),
                    "parameter_count": endpoint.parameter_count if endpoint else 0,
                    "training_walltime_seconds": (
                        endpoint.training_walltime_seconds if endpoint else 0.0
                    ),
                    "clean_waveform_metric": "N/A_no_clean_target",
                    **_metric_contract_fields(
                        status, inference_seed=inference_seed
                    ),
                }
            )
            rows.append(row)
    return rows


def run_sgeyesub_diffusion_cpu_validation(
    config: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    """Validate all folds and one real sealed block1/block2 record per study."""

    layouts, records, folds = _protocol_contract(config)
    root = Path(str(_mapping(config, "dataset")["data_root"]))
    real_rows: list[dict[str, Any]] = []
    for study in ("study01", "study02", "study03", "study04", "study05"):
        meta = next(
            value
            for value in records.values()
            if value.study == study and value.recording_key != "study05/study05_p42"
        )
        loaded = load_sgeyesub_signal_record(
            root,
            meta,
            layouts[meta.layout_id],
            include_query=True,
            include_query_annotations=False,
        )
        if loaded.query is None or loaded.query_annotations is not None:
            raise AssertionError("CPU validation violated the sealed-query boundary")
        support_windows = trial_local_nonoverlap_windows(
            loaded.support.eeg,
            samples_per_trial=meta.samples_per_trial,
            sampling_rate_hz=meta.sampling_rate_hz,
            recording_key=meta.recording_key,
        )
        query_windows = trial_local_nonoverlap_windows(
            loaded.query.eeg,
            samples_per_trial=meta.samples_per_trial,
            sampling_rate_hz=meta.sampling_rate_hz,
            recording_key=meta.recording_key,
        )
        real_rows.append(
            {
                "recording_key": meta.recording_key,
                "study": study,
                "layout_id": meta.layout_id,
                "sampling_rate_hz": meta.sampling_rate_hz,
                "eeg_channels": loaded.support.eeg.shape[0],
                "support_windows": support_windows.values.shape[0],
                "query_windows": query_windows.values.shape[0],
                "query_annotations_opened": False,
                "status": "success_real_block1_block2_loader_validation",
            }
        )
    p42 = records["study05/study05_p42"]
    real_rows.append(
        {
            "recording_key": p42.recording_key,
            "study": p42.study,
            "layout_id": p42.layout_id,
            "sampling_rate_hz": p42.sampling_rate_hz,
            "eeg_channels": p42.channel_count,
            "support_windows": "not_opened",
            "query_windows": "not_opened",
            "query_annotations_opened": False,
            "status": "blocked_no_population_singleton_layout_metadata_retained",
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(run_dir / "resolved_config.yaml", config)
    _write_csv(run_dir / "real_record_validation.csv", real_rows)
    summary = {
        "status": "completed_cpu_validation",
        "protocol_id": PROTOCOL_ID,
        "development_fold_count": sum(value.partition == "development" for value in folds),
        "evaluation_fold_count": sum(value.partition == "evaluation" for value in folds),
        "real_signal_payloads_opened": 5,
        "real_studies_covered": ["study01", "study02", "study03", "study04", "study05"],
        "preblocked_metadata_rows_retained": 1,
        "preblocked_recording_key": p42.recording_key,
        "query_annotations_used": False,
        "scientific_evidence": False,
    }
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def run_sgeyesub_diffusion_integration(
    config: Mapping[str, Any], run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """Real-record two-stem forward/backward/checkpoint engineering smoke."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("integration requires scheduled CUDA")
    implementation_identity = _implementation_identity()
    prepared = _prepare_fold(config, "development", 0)
    bundle = prepared.pairs
    first: dict[str, int] = {}
    for index, key in enumerate(bundle.recording_keys):
        first.setdefault(key, index)
    if len(first) < 2:
        raise RuntimeError("integration requires weak pairs from two independent stems")
    selected = torch.as_tensor(list(first.values())[:2], dtype=torch.long)
    tensors = _dataset(bundle).tensors
    batch = tuple(value.index_select(0, selected) for value in tensors)
    results: list[dict[str, Any]] = []
    matched = _mapping(config, "matched_comparison")
    amp = bool(matched["mixed_precision"])
    gradient_clip = float(matched["gradient_clip_norm"])
    diffusion = _mapping(matched, "conditional_diffusion")
    ddim_steps = int(diffusion["ddim_network_calls_per_window"])
    initial_state_seed = int(diffusion["initial_state_seed"])
    for method_id in (DETERMINISTIC_METHOD_ID, CONDITIONAL_METHOD_ID):
        arm_seed = int(_mapping(matched, "model_seed_by_arm")[method_id])
        torch.manual_seed(arm_seed)
        torch.cuda.manual_seed_all(arm_seed)
        model = _model(method_id, config, prepared).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=float(matched["learning_rate"]),
            weight_decay=float(matched["weight_decay"]),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024.0)
        first_loss = _amp_optimizer_update(
            model,
            method_id,
            batch,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            gradient_clip_norm=gradient_clip,
        )
        checkpoint = run_dir / "checkpoints" / f"{method_id}.pt"
        smoke_contract = {
            **implementation_identity,
            "protocol_id": PROTOCOL_ID,
            "method_id": method_id,
            "smoke": True,
            "amp": amp,
            "gradient_clip_norm": gradient_clip,
        }
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=0,
            step=1,
            config=smoke_contract,
            normalizer=prepared.normalizer.state_dict(),
            extra={"scientific_evidence": False},
        )
        reloaded = _model(method_id, config, prepared).to(device)
        resumed_optimizer = AdamW(
            reloaded.parameters(),
            lr=float(matched["learning_rate"]),
            weight_decay=float(matched["weight_decay"]),
        )
        resumed_scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024.0)
        resume_state = resume_training_checkpoint(
            checkpoint,
            model=reloaded,
            optimizer=resumed_optimizer,
            scaler=resumed_scaler,
            expected_config=smoke_contract,
            map_location=device,
        )
        if resume_state.step != 1 or (
            resume_state.normalizer_state != prepared.normalizer.state_dict()
        ):
            raise AssertionError("integration checkpoint resume contract changed")
        reloaded.train()
        second_loss = _amp_optimizer_update(
            reloaded,
            method_id,
            batch,
            device=device,
            optimizer=resumed_optimizer,
            scaler=resumed_scaler,
            amp=amp,
            gradient_clip_norm=gradient_clip,
        )
        save_training_checkpoint(
            checkpoint,
            model=reloaded,
            optimizer=resumed_optimizer,
            scaler=resumed_scaler,
            epoch=0,
            step=2,
            config=smoke_contract,
            normalizer=prepared.normalizer.state_dict(),
            extra={
                "scientific_evidence": False,
                "resumed_from_step": resume_state.step,
                "successful_optimizer_updates": 2,
            },
        )
        reloaded.eval()
        with torch.no_grad():
            observed, _, projector, attenuation, mask = (
                value.to(device) for value in batch
            )
            if method_id == DETERMINISTIC_METHOD_ID:
                output = reloaded(
                    observed,
                    projector=projector,
                    attenuation=attenuation,
                    valid_time_mask=mask,
                )
                calls = 1
            else:
                generator = torch.Generator(device=device)
                generator.manual_seed(initial_state_seed)
                sampled = reloaded.sample_ddim(
                    observed=observed,
                    projector=projector,
                    attenuation=attenuation,
                    valid_time_mask=mask,
                    ddim_steps=ddim_steps,
                    eta=float(diffusion["ddim_eta"]),
                    generator=generator,
                )
                output = sampled.restored
                calls = sampled.network_calls
                if calls != ddim_steps:
                    raise AssertionError("integration did not run frozen DDIM100")
        if output.shape != observed.shape or not bool(torch.isfinite(output).all()):
            raise FloatingPointError("integration output is invalid")
        results.append(
            {
                "method_id": method_id,
                "first_update_loss": float(first_loss.cpu()),
                "resumed_update_loss": float(second_loss.cpu()),
                "checkpoint_resumed_with_model_optimizer_scaler": True,
                "checkpoint_resumed_from_step": resume_state.step,
                "successful_optimizer_updates": 2,
                "amp_enabled": amp,
                "autocast_dtype": "float16" if amp else "disabled",
                "gradient_clip_norm": gradient_clip,
                "network_calls": calls,
                "formal_ddim_network_calls": (
                    ddim_steps if method_id == CONDITIONAL_METHOD_ID else "N/A"
                ),
                "inference_seed": initial_state_seed,
                **implementation_identity,
                "real_weak_pair_stems_in_batch": list(first)[:2],
            }
        )
    _write_csv(run_dir / "integration_endpoints.csv", results)
    summary = {
        "status": "completed_gpu_integration_smoke",
        "protocol_id": PROTOCOL_ID,
        "scientific_evidence": False,
        "real_data": True,
        "real_independent_stem_count": 2,
        "methods": results,
    }
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def run_sgeyesub_diffusion_fold(
    config: Mapping[str, Any],
    partition: str,
    task_index: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train both matched arms and evaluate all held-out block-2 windows."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("scientific fold requires scheduled CUDA")
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = signal.signal(signal.SIGUSR1, request_stop)
    try:
        implementation_identity = _implementation_identity()
        if partition == "evaluation":
            _require_completed_development_aggregate(
                config, implementation_identity
            )
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        prepared = _prepare_fold(config, partition, task_index)
        if stop:
            run_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "status": "checkpointed_for_resume",
                "protocol_id": PROTOCOL_ID,
                "partition": partition,
                "fold_id": prepared.fold.fold_id,
                "training_endpoints": [],
                "checkpoint_written": False,
                "reason": "SIGUSR1_received_during_fold_preparation_before_training",
                "next_action": "resume_same_fold_same_configuration",
                **implementation_identity,
            }
            _write_json(run_dir / "result_summary.json", summary)
            return summary

        pair_keys = _pair_keys(prepared.pairs)
        batch_size = int(_mapping(config, "matched_comparison")["batch_size"])
        schedule = _minibatch_schedule(len(pair_keys), batch_size)
        canonical = (
            Path(str(_mapping(config, "outputs")[f"{partition}_root"]))
            / prepared.fold.fold_id
        )
        _persist_schedule(
            canonical / "training_minibatch_indices.csv", schedule, pair_keys
        )
        _persist_schedule(
            run_dir / "training_minibatch_indices.csv", schedule, pair_keys
        )
        models: dict[str, torch.nn.Module] = {}
        endpoints: list[_Endpoint] = []
        for method_id in (DETERMINISTIC_METHOD_ID, CONDITIONAL_METHOD_ID):
            if stop:
                break
            model, endpoint = _train(
                config,
                prepared,
                method_id,
                schedule,
                device=device,
                stop_requested=lambda: stop,
            )
            endpoints.append(endpoint)
            if model is None:
                break
            models[method_id] = model
        endpoint_rows = [asdict(value) for value in endpoints]
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_config(run_dir / "resolved_config.yaml", config)
        if endpoint_rows:
            _write_csv(run_dir / "training_endpoints.csv", endpoint_rows)
        if len(models) != 2:
            summary = {
                "status": "checkpointed_for_resume",
                "protocol_id": PROTOCOL_ID,
                "partition": partition,
                "fold_id": prepared.fold.fold_id,
                "training_endpoints": endpoint_rows,
                "next_action": "resume_same_fold_same_configuration",
                **implementation_identity,
            }
            _write_json(run_dir / "result_summary.json", summary)
            return summary
        endpoint_map = {value.method_id: value for value in endpoints}
        metrics = _evaluate_fold(config, prepared, models, endpoint_map, device)
        _write_csv(run_dir / "metrics.csv", metrics)
        summary = {
            "status": "completed_fold",
            "protocol_id": PROTOCOL_ID,
            "partition": partition,
            "fold_id": prepared.fold.fold_id,
            "study": prepared.fold.study,
            "training_recording_keys": list(prepared.fold.training_recording_keys),
            "heldout_recording_keys": list(prepared.fold.heldout_recording_keys),
            "weak_training_pairs": len(pair_keys),
            "training_endpoints": endpoint_rows,
            "exact_shared_minibatch_sequence_updates": len(schedule),
            "exact_shared_minibatch_sequence_verified": all(
                value.minibatch_sequence_verified for value in endpoints
            ),
            "all_arm_outputs_frozen_before_query_evaluation_fields_opened": True,
            "query_evaluation_fields_used_for_fit_selection_or_inference": False,
            "clean_target_available": False,
            "clean_waveform_recovery_claim": False,
            "weak_target_semantics": "low_artifact_observed_EEG_not_clean_truth",
            "metrics": str(run_dir / "metrics.csv"),
            **implementation_identity,
        }
        _write_json(run_dir / "result_summary.json", summary)
        canonical.mkdir(parents=True, exist_ok=True)
        _write_csv(canonical / "metrics.csv", metrics)
        _write_csv(canonical / "training_endpoints.csv", endpoint_rows)
        _write_json(canonical / "result_summary.json", summary)
        return summary
    finally:
        signal.signal(signal.SIGUSR1, previous)


def aggregate_sgeyesub_diffusion_partition(
    config: Mapping[str, Any], partition: str, run_dir: Path
) -> dict[str, Any]:
    """Load complete fold artifacts and delegate all statistics to the core."""

    implementation_identity = _implementation_identity()
    _, _, folds = _protocol_contract(config)
    selected = tuple(value for value in folds if value.partition == partition)
    root = Path(str(_mapping(config, "outputs")[f"{partition}_root"]))
    rows: list[dict[str, Any]] = []
    endpoints: list[dict[str, Any]] = []
    for fold in selected:
        parent = root / fold.fold_id
        summary = json.loads((parent / "result_summary.json").read_text(encoding="utf-8"))
        _validate_completed_fold_for_aggregate(
            summary,
            fold_id=fold.fold_id,
            expected_identity=implementation_identity,
        )
        rows.extend(_read_csv(parent / "metrics.csv"))
        endpoints.extend(dict(value) for value in summary["training_endpoints"])
    if partition == "evaluation":
        preblocked_seed = _record_seed(
            config,
            "evaluation",
            "preblocked_singleton_layout06",
            "study05/study05_p42",
        )
        for method_id in ARMS:
            rows.append(
                {
                    "partition": "evaluation",
                    "fold_id": "preblocked_singleton_layout06",
                    "study": "study05",
                    "participant_stem": "study05_p42",
                    "recording_key": "study05/study05_p42",
                    "method_id": method_id,
                    "status": "blocked_no_population",
                    "fallback_used": False,
                    "query_evaluation_fields_opened_after_all_arm_outputs_frozen": False,
                    "query_evaluation_fields_used_for_fit_selection_or_inference": False,
                    "clean_waveform_metric": "N/A_no_clean_target",
                    "inference_placeholder_used_for_freeze_only": False,
                    "performance_values_eligible": False,
                    **_metric_contract_fields(
                        "blocked_no_population", inference_seed=preblocked_seed
                    ),
                }
            )
    outputs = write_sgeyesub_diffusion_aggregate(
        rows,
        config=config,
        fold_training_endpoints=endpoints,
        partition=partition,
    )
    summary = json.loads(outputs["result_summary"].read_text(encoding="utf-8"))
    summary.update(implementation_identity)
    _write_json(outputs["result_summary"], summary)
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_config(run_dir / "resolved_config.yaml", config)
    _write_csv(run_dir / "metrics.csv", rows)
    _write_csv(run_dir / "training_endpoints.csv", endpoints)
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def validate_sgeyesub_diffusion_runner_config(config: Mapping[str, Any]) -> None:
    """Compatibility alias used by route tests."""

    validate_sgeyesub_diffusion_config(config)
    sgeyesub_p0_config(config)


__all__ = [
    "aggregate_sgeyesub_diffusion_partition",
    "run_sgeyesub_diffusion_cpu_validation",
    "run_sgeyesub_diffusion_fold",
    "run_sgeyesub_diffusion_integration",
    "validate_sgeyesub_diffusion_runner_config",
]
