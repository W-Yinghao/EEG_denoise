"""Shared model construction and full clean-prior training."""

from __future__ import annotations

import csv
import json
import random
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from eeg_cgdr.data.eegdenoise import CleanPriorSplit, load_clean_prior_split
from eeg_cgdr.models import CleanEEGDiffusionPrior
from eeg_cgdr.training import resume_training_checkpoint, save_training_checkpoint
from saddpm.diffusion.schedule import DiffusionConfig
from saddpm.models.config import ModelConfig


@dataclass(frozen=True)
class PriorTrainingResult:
    prior: CleanEEGDiffusionPrior
    split: CleanPriorSplit
    checkpoint: Path
    best_checkpoint: Path
    epochs_completed: int
    steps_completed: int
    best_validation_loss: float
    resumed: bool
    stopped_for_signal: bool


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_prior(config: dict[str, Any], device: torch.device) -> CleanEEGDiffusionPrior:
    model_config = ModelConfig(**config["model"])
    diffusion_config = DiffusionConfig(**config["diffusion"])
    return CleanEEGDiffusionPrior(model_config, diffusion_config).to(device)


def load_prior_data(config: dict[str, Any]) -> CleanPriorSplit:
    source = config["clean_prior_data"]
    return load_clean_prior_split(
        source["path"],
        validation_fraction=float(source.get("validation_fraction", 0.1)),
        seed=int(config["seed"]),
    )


def _checkpoint_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "CGDR-clean-prior",
        "dataset": str(config["clean_prior_data"]["dataset"]),
        "model": dict(config["model"]),
        "diffusion": dict(config["diffusion"]),
        "training": dict(config["training"]),
        "seed": int(config["seed"]),
    }


def _make_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _validation_loss(
    prior: CleanEEGDiffusionPrior,
    validation: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
    seed: int,
) -> float:
    dataset = TensorDataset(torch.from_numpy(validation[:, None, :]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    total = 0.0
    samples = 0
    prior.eval()
    with torch.no_grad():
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                loss = prior.training_loss(clean, generator=generator)
            total += float(loss) * clean.shape[0]
            samples += clean.shape[0]
    if samples != validation.shape[0]:
        raise AssertionError("validation did not consume every clean epoch")
    return total / samples


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "step", "train_loss", "validation_loss", "best"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_history(path: Path, *, before_epoch: int) -> list[dict[str, Any]]:
    """Retain completed epoch rows when resuming an interrupted run."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if int(row["epoch"]) < before_epoch]


def train_clean_prior(
    config: dict[str, Any],
    *,
    device: torch.device,
    checkpoint: Path,
    best_checkpoint: Path,
    history_path: Path,
) -> PriorTrainingResult:
    """Train every configured real clean epoch with epoch-boundary resume."""
    seed = int(config["seed"])
    configure_reproducibility(seed)
    split = load_prior_data(config)
    prior = build_prior(config, device)
    training = config["training"]
    optimizer = AdamW(
        prior.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp = bool(training["mixed_precision"]) and device.type == "cuda"
    scaler = _make_scaler(amp)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 101)
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed + 202)
    generators = {"loader": loader_generator, "training_noise": noise_generator}
    contract = _checkpoint_contract(config)
    normalizer = {
        "mean": float(split.mean),
        "standard_deviation": float(split.standard_deviation),
        "sampling_rate": int(split.sampling_rate),
    }

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    epochs_without_improvement = 0
    resumed = False
    if bool(training.get("resume", True)) and checkpoint.exists():
        state = resume_training_checkpoint(
            checkpoint,
            model=prior,
            optimizer=optimizer,
            scaler=scaler,
            generators=generators,
            expected_config=contract,
            map_location=device,
        )
        start_epoch = state.epoch + 1
        global_step = state.step
        best_loss = float(state.extra.get("best_validation_loss", float("inf")))
        epochs_without_improvement = int(state.extra.get("epochs_without_improvement", 0))
        resumed = True

    train_tensor = torch.from_numpy(split.train[:, None, :])
    dataset = TensorDataset(train_tensor)

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGUSR1, request_stop)
    history = _read_history(history_path, before_epoch=start_epoch)
    last_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, int(training["epochs"])):
            # Epoch-indexed shuffling makes an epoch-boundary resume identical to
            # an uninterrupted run.  A persistent worker iterator would consume
            # the same generator for both worker seeds and shuffling on restart.
            epoch_loader_generator = torch.Generator(device="cpu")
            epoch_loader_generator.manual_seed(seed + 101 + epoch)
            loader = DataLoader(
                dataset,
                batch_size=int(training["batch_size"]),
                shuffle=True,
                generator=epoch_loader_generator,
                num_workers=int(training["workers"]),
                pin_memory=device.type == "cuda",
                persistent_workers=False,
                drop_last=False,
            )
            prior.train()
            total_loss = 0.0
            samples = 0
            for (clean_cpu,) in loader:
                clean = clean_cpu.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    loss = prior.training_loss(clean, generator=noise_generator)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite clean-prior loss at epoch={epoch} step={global_step}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    prior.parameters(), float(training["gradient_clip"])
                )
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * clean.shape[0]
                samples += clean.shape[0]
                global_step += 1
            if samples != split.train.shape[0]:
                raise AssertionError("training epoch omitted real clean epochs")
            validation_loss = _validation_loss(
                prior,
                split.validation,
                batch_size=int(training["batch_size"]),
                device=device,
                amp=amp,
                seed=seed + 303,
            )
            if not np.isfinite(validation_loss):
                raise FloatingPointError(
                    f"non-finite validation loss at epoch={epoch}: {validation_loss}"
                )
            improved = validation_loss < best_loss - 1.0e-6
            if improved:
                best_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            extra = {
                "best_validation_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "train_epochs": int(split.train.shape[0]),
                "validation_epochs": int(split.validation.shape[0]),
            }
            save_training_checkpoint(
                checkpoint,
                model=prior,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                config=contract,
                normalizer=normalizer,
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
                    normalizer=normalizer,
                    generators=generators,
                    extra=extra,
                )
            history.append(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": total_loss / samples,
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
        signal.signal(signal.SIGUSR1, previous_handler)

    if not best_checkpoint.exists():
        raise RuntimeError("training completed without a best checkpoint")
    summary = {
        "status": "checkpointed_for_signal" if stop_requested else "completed",
        "epochs_completed": last_epoch + 1,
        "steps_completed": global_step,
        "best_validation_loss": best_loss,
        "resumed": resumed,
        "train_epochs_per_epoch": int(split.train.shape[0]),
        "validation_epochs": int(split.validation.shape[0]),
    }
    history_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return PriorTrainingResult(
        prior=prior,
        split=split,
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
        epochs_completed=last_epoch + 1,
        steps_completed=global_step,
        best_validation_loss=best_loss,
        resumed=resumed,
        stopped_for_signal=stop_requested,
    )


def load_best_prior(
    config: dict[str, Any], checkpoint: Path, device: torch.device
) -> tuple[CleanEEGDiffusionPrior, dict[str, Any]]:
    from eeg_cgdr.training import load_training_checkpoint

    payload = load_training_checkpoint(checkpoint, map_location=device)
    if payload["config"] != _checkpoint_contract(config):
        raise ValueError("best checkpoint does not match this clean-prior configuration")
    prior = build_prior(config, device)
    prior.load_state_dict(payload["model_state"])
    prior.eval()
    return prior, dict(payload["normalizer_state"])
