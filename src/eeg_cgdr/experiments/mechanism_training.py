"""Checkpointable multichannel prior training for the repaired Klados audit."""

from __future__ import annotations

import csv
import json
import random
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    KLADOS_NATIVE_CHANNEL_ORDER,
    KLADOS_TRAIN_RECORDS,
    ChannelNormalizer,
    fit_channel_normalizer,
    prepare_clean_training_windows,
    prepare_population_calibration,
    select_records,
)
from eeg_cgdr.experiments.klados import population_source_transfer
from eeg_cgdr.inference import DatasetPopulationProjector
from eeg_cgdr.models import CleanEEGDiffusionPrior
from eeg_cgdr.operators import P0Config, fit_p0
from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.models.config import ModelConfig


@dataclass(frozen=True)
class MechanismTrainingResult:
    status: str
    checkpoint: Path
    best_checkpoint: Path
    epochs_completed: int
    steps_completed: int
    best_validation_loss: float
    resumed: bool
    population_projector: Optional[DatasetPopulationProjector]


def _loader_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config["klados"]
    return {
        "data_root": raw["data_root"],
        "files": {
            "contaminated": raw["contaminated"],
            "clean": raw["clean"],
            "heog": raw["heog"],
            "veog": raw["veog"],
        },
        "official_description": {"records": 54},
    }


def _p0_config(config: dict[str, Any]) -> P0Config:
    raw = config["p0"]
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=int(raw["bootstrap_replicates"]),
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=float(raw["minimum_bootstrap_success"]),
        maximum_bootstrap_median_distance=float(raw["maximum_bootstrap_median_distance"]),
        maximum_bootstrap_q90_distance=float(raw["maximum_bootstrap_q90_distance"]),
        seed=int(config["seed"]),
    )


def _build_prior(config: dict[str, Any], device: torch.device) -> CleanEEGDiffusionPrior:
    prior = CleanEEGDiffusionPrior(
        ModelConfig(**config["model"]),
        DiffusionConfig(**{
            key: config["diffusion"][key]
            for key in ("num_timesteps", "beta_start", "beta_end", "schedule")
        }),
        prior_mode="joint_multichannel",
        enforce_scientific_schedule=True,
    )
    if prior.model_config.in_channels != 19:
        raise ValueError("Klados scientific prior must model all 19 channels jointly")
    return prior.to(device)


def _contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "CGDR-repaired-multichannel-prior",
        "dataset": "klados_bamidis_v4",
        "montage": list(config["klados"]["channel_order"]),
        "source_records": list(config["klados"]["training_source_records"]),
        "validation_source_records": list(
            config["klados"]["development_source_records"]
        ),
        "source_sampling_rate": int(config["klados"]["source_sampling_rate"]),
        "data_source": {
            key: config["klados"][key]
            for key in ("data_root", "contaminated", "clean", "heog", "veog")
        },
        "preprocessing": {
            key: config["preprocessing"][key]
            for key in ("target_sampling_rate", "window_samples", "normalization")
        },
        "model": dict(config["model"]),
        "diffusion": {
            key: config["diffusion"][key]
            for key in ("num_timesteps", "beta_start", "beta_end", "schedule")
        },
        "training": dict(config["training"]),
        "seed": int(config["seed"]),
        "prior_mode": "joint_multichannel",
    }


def _normalizer_state(normalizer: ChannelNormalizer) -> dict[str, Any]:
    return {
        "mean": normalizer.mean.tolist(),
        "standard_deviation": normalizer.standard_deviation.tolist(),
        "source_records": list(normalizer.source_records),
        "sample_count": int(normalizer.sample_count),
        "semantics": "per_channel_clean_training_source_records",
    }


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epoch", "step", "train_loss", "validation_loss", "best"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_history(path: Path, before_epoch: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if int(row["epoch"]) < before_epoch]


def _validation_loss(
    prior: CleanEEGDiffusionPrior,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
    seed: int,
) -> float:
    loader = DataLoader(
        TensorDataset(values, mask), batch_size=batch_size, shuffle=False, num_workers=0
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total = 0.0
    valid_frames = 0.0
    prior.eval()
    with torch.no_grad():
        for clean_cpu, mask_cpu in loader:
            clean = clean_cpu.to(device, non_blocking=True)
            valid = mask_cpu.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = prior.training_loss(
                    clean, generator=generator, valid_time_mask=valid
                )
            weight = float(valid.sum()) * clean.shape[1]
            total += float(loss) * weight
            valid_frames += weight
    if valid_frames <= 0:
        raise AssertionError("validation contains no valid EEG frames")
    return total / valid_frames


def _fit_population_projector(
    config: dict[str, Any],
    records,
    normalizer: ChannelNormalizer,
) -> tuple[DatasetPopulationProjector, dict[str, Any]]:
    p0_config = _p0_config(config)
    outcomes = []
    details: list[dict[str, Any]] = []
    source_rate = int(config["klados"]["source_sampling_rate"])
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    if source_rate != 200 or target_rate != 256:
        raise ValueError("registered population projector requires native 200 Hz to 256 Hz")
    for record in select_records(records, KLADOS_TRAIN_RECORDS):
        batch = prepare_population_calibration(
            record,
            normalizer,
            source_rate=source_rate,
            target_rate=target_rate,
        )
        outcome = fit_p0(
            batch,
            p0_config,
            movement_threshold=float(config["p0"]["movement_threshold"]),
        )
        outcomes.append(outcome)
        details.append(
            {
                "source_record": record.record_id,
                "status": outcome.status,
                "reasons": list(outcome.reasons),
                "rank": outcome.transfer.rank if outcome.transfer is not None else None,
                "singular_values": (
                    outcome.transfer.diagnostics.get("singular_values")
                    if outcome.transfer is not None
                    else None
                ),
                "ridge_lambda": float(config["p0"]["ridge_lambda"]),
            }
        )
    eligible = sum(outcome.transfer is not None for outcome in outcomes)
    if eligible < int(np.ceil(0.8 * len(outcomes))):
        raise RuntimeError(
            f"only {eligible}/{len(outcomes)} training records have eligible P0 operators"
        )
    population = population_source_transfer(
        outcomes, target_rank=int(config["p0"]["target_rank"])
    )
    if population.transfer is None:
        raise RuntimeError("population projector construction failed")
    population_projector = DatasetPopulationProjector(
        dataset_id="klados_bamidis_v4",
        montage_id="klados_v4_19ch_native_order_256hz",
        projector=np.asarray(population.transfer.projector, dtype=np.float64),
        source="all_training_source_records_sim01_sim30",
    )
    summary = {
        "dataset_id": population_projector.dataset_id,
        "montage_id": population_projector.montage_id,
        "source": population_projector.source,
        "channel_order": list(config["klados"]["channel_order"]),
        "training_source_records": list(config["klados"]["training_source_records"]),
        "source_sampling_rate": source_rate,
        "target_sampling_rate": target_rate,
        "p0": dict(config["p0"]),
        "eligible_records": eligible,
        "records_total": len(outcomes),
        "rank": population.transfer.rank,
        "projector": population_projector.projector.tolist(),
        "source_diagnostics": details,
    }
    return population_projector, summary


def load_population_projector(config: dict[str, Any]) -> DatasetPopulationProjector:
    """Load the small dataset-specific Pi0 artifact with semantic checks."""

    path = Path(config["outputs"]["population_state"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_id": "klados_bamidis_v4",
        "montage_id": "klados_v4_19ch_native_order_256hz",
        "source": "all_training_source_records_sim01_sim30",
    }
    for field_name, expected_value in expected.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(f"population projector {field_name} differs from protocol")
    artifact_contract = {
        "channel_order": list(config["klados"]["channel_order"]),
        "training_source_records": list(config["klados"]["training_source_records"]),
        "source_sampling_rate": int(config["klados"]["source_sampling_rate"]),
        "target_sampling_rate": int(config["preprocessing"]["target_sampling_rate"]),
        "p0": dict(config["p0"]),
    }
    for field_name, expected_value in artifact_contract.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(f"population projector {field_name} differs from config")
    projector = np.asarray(payload.get("projector"), dtype=np.float64)
    channels = len(config["klados"]["channel_order"])
    if (
        tuple(config["klados"]["channel_order"]) != KLADOS_NATIVE_CHANNEL_ORDER
        or int(config["klados"]["source_sampling_rate"]) != 200
        or int(config["preprocessing"]["target_sampling_rate"]) != 256
    ):
        raise ValueError("population projector montage configuration is incompatible")
    if projector.shape != (channels, channels) or not np.isfinite(projector).all():
        raise ValueError("population projector has invalid shape or values")
    if not np.allclose(projector, projector.T, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("population projector is not symmetric")
    if not np.allclose(projector @ projector, projector, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("population projector is not idempotent")
    return DatasetPopulationProjector(projector=projector, **expected)


def train_mechanism_prior(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> MechanismTrainingResult:
    """Train/resume the repaired 19-channel T=1000 prior on sim01-sim30."""

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = load_klados_records(_loader_config(config))
    if tuple(config["klados"]["training_source_records"]) != KLADOS_TRAIN_RECORDS:
        raise ValueError("training source-record split differs from the frozen protocol")
    if tuple(config["klados"]["development_source_records"]) != KLADOS_DEVELOPMENT_RECORDS:
        raise ValueError("development source-record split differs from the frozen protocol")
    if tuple(config["klados"]["channel_order"]) != KLADOS_NATIVE_CHANNEL_ORDER:
        raise ValueError("Klados channel order differs from the registered montage")
    normalizer = fit_channel_normalizer(records)
    source_rate = int(config["klados"]["source_sampling_rate"])
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    if source_rate != 200 or target_rate != 256:
        raise ValueError("Klados mechanism training requires native 200 Hz to 256 Hz")
    window_samples = int(config["preprocessing"]["window_samples"])
    train = prepare_clean_training_windows(
        records,
        normalizer,
        source_records=KLADOS_TRAIN_RECORDS,
        source_rate=source_rate,
        target_rate=target_rate,
        window_samples=window_samples,
    )
    validation = prepare_clean_training_windows(
        records,
        normalizer,
        source_records=KLADOS_DEVELOPMENT_RECORDS,
        source_rate=source_rate,
        target_rate=target_rate,
        window_samples=window_samples,
    )
    train_values = torch.from_numpy(train.values.astype(np.float32, copy=False))
    train_mask = torch.from_numpy(train.valid_time_weight.astype(np.float32, copy=False))
    validation_values = torch.from_numpy(
        validation.values.astype(np.float32, copy=False)
    )
    validation_mask = torch.from_numpy(
        validation.valid_time_weight.astype(np.float32, copy=False)
    )

    prior = _build_prior(config, device)
    training = config["training"]
    optimizer = AdamW(
        prior.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed + 202)
    generators = {"training_noise": noise_generator}
    checkpoint = Path(config["outputs"]["checkpoint"])
    best_checkpoint = Path(config["outputs"]["best_checkpoint"])
    history_path = Path(config["outputs"]["training_history"])
    contract = _contract(config)
    normalizer_state = _normalizer_state(normalizer)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    resumed = False
    if bool(training.get("resume", True)) and checkpoint.is_file():
        state = resume_training_checkpoint(
            checkpoint,
            model=prior,
            optimizer=optimizer,
            scaler=scaler,
            generators=generators,
            expected_config=contract,
            map_location=device,
        )
        if state.normalizer_state != normalizer_state:
            raise ValueError("checkpoint normalizer differs from frozen training records")
        start_epoch = state.epoch + 1
        global_step = state.step
        best_loss = float(state.extra.get("best_validation_loss", float("inf")))
        epochs_without_improvement = int(state.extra.get("epochs_without_improvement", 0))
        resumed = True

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGUSR1, request_stop)
    history = _read_history(history_path, start_epoch)
    last_epoch = start_epoch - 1
    dataset = TensorDataset(train_values, train_mask)
    resumed_terminal_early_stop = bool(
        resumed
        and start_epoch >= int(training["minimum_epochs"])
        and epochs_without_improvement >= int(training["patience"])
    )
    final_epoch = start_epoch if resumed_terminal_early_stop else int(training["epochs"])
    try:
        for epoch in range(start_epoch, final_epoch):
            order_generator = torch.Generator(device="cpu")
            order_generator.manual_seed(seed + 101 + epoch)
            loader = DataLoader(
                dataset,
                batch_size=int(training["batch_size"]),
                shuffle=True,
                generator=order_generator,
                num_workers=int(training["workers"]),
                pin_memory=device.type == "cuda",
                persistent_workers=False,
                drop_last=False,
            )
            prior.train()
            total_loss = 0.0
            total_weight = 0.0
            for clean_cpu, mask_cpu in loader:
                clean = clean_cpu.to(device, non_blocking=True)
                valid = mask_cpu.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    loss = prior.training_loss(
                        clean,
                        generator=noise_generator,
                        valid_time_mask=valid,
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite multichannel prior loss epoch={epoch} step={global_step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    prior.parameters(), float(training["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                weight = float(valid.sum()) * clean.shape[1]
                total_loss += float(loss.detach()) * weight
                total_weight += weight
                global_step += 1
            validation_loss = _validation_loss(
                prior,
                validation_values,
                validation_mask,
                batch_size=int(training["batch_size"]),
                device=device,
                amp=amp,
                seed=seed + 303,
            )
            if total_weight <= 0 or not np.isfinite(validation_loss):
                raise FloatingPointError("invalid training/validation aggregate")
            improved = validation_loss < best_loss - 1.0e-6
            if improved:
                best_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            extra = {
                "best_validation_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "training_windows": int(train.values.shape[0]),
                "validation_windows": int(validation.values.shape[0]),
                "valid_training_frames": int(train.valid_time_weight.sum()),
                "valid_validation_frames": int(validation.valid_time_weight.sum()),
                "prior_mode": "joint_multichannel",
            }
            save_training_checkpoint(
                checkpoint,
                model=prior,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                config=contract,
                normalizer=normalizer_state,
                generators=generators,
                extra=extra,
            )
            if improved:
                save_training_checkpoint(
                    best_checkpoint,
                    model=prior,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    config=contract,
                    normalizer=normalizer_state,
                    generators=generators,
                    extra=extra,
                )
            history.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": total_loss / total_weight,
                    "validation_loss": validation_loss,
                    "best": improved,
                }
            )
            _write_history(history_path, history)
            last_epoch = epoch
            if stop_requested:
                break
            if (
                epoch + 1 >= int(training["minimum_epochs"])
                and epochs_without_improvement >= int(training["patience"])
            ):
                break
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    status = "checkpointed_for_resume" if stop_requested else "completed"
    population_projector: Optional[DatasetPopulationProjector] = None
    cross_channel_audit: Optional[dict[str, Any]] = None
    if status == "completed":
        population_projector, population_summary = _fit_population_projector(
            config, records, normalizer
        )
        population_path = Path(config["outputs"]["population_state"])
        population_path.parent.mkdir(parents=True, exist_ok=True)
        population_path.write_text(
            json.dumps(population_summary, indent=2) + "\n", encoding="utf-8"
        )
        audited_prior, _ = load_repaired_prior(config, device=device)
        audit_count = min(2, int(validation_values.shape[0]))
        influence = audited_prior.assert_cross_channel_dependency(
            validation_values[:audit_count].to(device),
            torch.full(
                (audit_count,),
                int(config["diffusion"]["num_timesteps"]) // 2,
                device=device,
                dtype=torch.long,
            ),
            valid_time_mask=validation_mask[:audit_count].to(device),
            perturbation=1.0e-2,
            minimum_influence=1.0e-10,
        )
        off_diagonal = influence.detach().clone()
        off_diagonal.fill_diagonal_(0.0)
        cross_channel_audit = {
            "status": "passed",
            "probe_windows": audit_count,
            "minimum_channel_max_off_diagonal_influence": float(
                off_diagonal.max(dim=1).values.min()
            ),
        }
    summary = {
        "status": status,
        "epochs_completed": last_epoch + 1,
        "steps_completed": global_step,
        "best_validation_loss": best_loss,
        "resumed": resumed,
        "resumed_terminal_early_stop": resumed_terminal_early_stop,
        "training_source_records": list(KLADOS_TRAIN_RECORDS),
        "validation_source_records": list(KLADOS_DEVELOPMENT_RECORDS),
        "training_windows": int(train.values.shape[0]),
        "validation_windows": int(validation.values.shape[0]),
        "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "cross_channel_dependency": cross_channel_audit,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_status.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return MechanismTrainingResult(
        status=status,
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
        epochs_completed=last_epoch + 1,
        steps_completed=global_step,
        best_validation_loss=best_loss,
        resumed=resumed,
        population_projector=population_projector,
    )


def load_repaired_prior(
    config: dict[str, Any], *, device: torch.device
) -> tuple[CleanEEGDiffusionPrior, ChannelNormalizer]:
    """Load only the repaired T=1000 19-channel checkpoint contract."""

    checkpoint = Path(config["outputs"]["best_checkpoint"])
    payload = load_training_checkpoint(checkpoint, map_location=device)
    if payload["config"] != _contract(config):
        raise ValueError("checkpoint is not the repaired Klados multichannel prior")
    state = payload["normalizer_state"]
    if state.get("semantics") != "per_channel_clean_training_source_records":
        raise ValueError("checkpoint normalizer semantics are not registered")
    normalizer = ChannelNormalizer(
        mean=np.asarray(state["mean"], dtype=np.float64),
        standard_deviation=np.asarray(state["standard_deviation"], dtype=np.float64),
        source_records=tuple(int(value) for value in state["source_records"]),
        sample_count=int(state["sample_count"]),
    )
    if normalizer.source_records != KLADOS_TRAIN_RECORDS:
        raise ValueError("checkpoint normalizer includes non-training source records")
    prior = _build_prior(config, device)
    prior.load_state_dict(payload["model_state"])
    prior.eval()
    return prior, normalizer
