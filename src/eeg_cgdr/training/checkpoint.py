"""Small atomic training checkpoints with deterministic resume state."""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

_SCHEMA_VERSION = 1
_REQUIRED_KEYS = {
    "schema_version",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "scaler_state",
    "epoch",
    "step",
    "rng_state",
    "config",
    "normalizer_state",
}


@dataclass(frozen=True)
class ResumeState:
    """Control state returned after restoring a training checkpoint."""

    epoch: int
    step: int
    config: dict[str, Any]
    normalizer_state: Any
    extra: dict[str, Any]


def _component_state(component: Any) -> Any:
    if component is None:
        return None
    state_dict = getattr(component, "state_dict", None)
    return state_dict() if callable(state_dict) else component


def _capture_rng_state(
    generators: Optional[Mapping[str, torch.Generator]] = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generators": {},
    }
    if generators is not None:
        state["generators"] = {
            str(name): generator.get_state() for name, generator in generators.items()
        }
    return state


def _restore_rng_state(
    state: Mapping[str, Any],
    generators: Optional[Mapping[str, torch.Generator]] = None,
) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(tuple(numpy_state))
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])

    saved_generators = state.get("generators", {})
    if saved_generators:
        if generators is None:
            raise ValueError(
                "checkpoint has named generator states but no generators were supplied"
            )
        missing = sorted(set(saved_generators) - set(generators))
        if missing:
            raise ValueError(f"missing named generators for resume: {missing}")
        for name, generator_state in saved_generators.items():
            generators[name].set_state(generator_state.cpu())


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Any,
    epoch: int,
    step: int,
    config: Mapping[str, Any],
    normalizer: Any,
    scheduler: Any = None,
    generators: Optional[Mapping[str, torch.Generator]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save everything needed for an epoch-boundary resume.

    The checkpoint is written to a temporary file in the destination directory,
    flushed, and then published with ``os.replace``.  No content hash or external
    manifest is created.
    """

    destination = Path(path)
    if int(epoch) < 0 or int(step) < 0:
        raise ValueError("epoch and step must be non-negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": _component_state(scheduler),
        "scaler_state": _component_state(scaler),
        "epoch": int(epoch),
        "step": int(step),
        "rng_state": _capture_rng_state(generators),
        "config": dict(config),
        "normalizer_state": _component_state(normalizer),
        "extra": dict(extra or {}),
    }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a trusted local CGDR checkpoint."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    missing = sorted(_REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {missing}")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema {payload['schema_version']!r}")
    if not isinstance(payload["config"], dict):
        raise ValueError("checkpoint config must be a dictionary")
    if int(payload["epoch"]) < 0 or int(payload["step"]) < 0:
        raise ValueError("checkpoint epoch and step must be non-negative")
    payload.setdefault("extra", {})
    return payload


def resume_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scaler: Any = None,
    scheduler: Any = None,
    normalizer: Any = None,
    generators: Optional[Mapping[str, torch.Generator]] = None,
    expected_config: Optional[Mapping[str, Any]] = None,
    map_location: str | torch.device = "cpu",
    strict_model: bool = True,
    restore_rng: bool = True,
) -> ResumeState:
    """Reload model/training state and return the saved epoch/step cursor."""

    payload = load_training_checkpoint(path, map_location=map_location)
    if expected_config is not None and payload["config"] != dict(expected_config):
        raise ValueError("checkpoint config does not match the requested run config")

    model.load_state_dict(payload["model_state"], strict=strict_model)
    if optimizer is not None:
        if payload["optimizer_state"] is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        if payload["scheduler_state"] is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None:
        if payload["scaler_state"] is None:
            raise ValueError("checkpoint has no scaler state")
        scaler.load_state_dict(payload["scaler_state"])
    if normalizer is not None:
        loader = getattr(normalizer, "load_state_dict", None)
        if not callable(loader):
            raise TypeError("normalizer must provide load_state_dict for in-place restore")
        loader(payload["normalizer_state"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"], generators)

    return ResumeState(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        config=dict(payload["config"]),
        normalizer_state=payload["normalizer_state"],
        extra=dict(payload.get("extra", {})),
    )


# Short aliases keep call sites readable without duplicating implementation.
save_checkpoint = save_training_checkpoint
load_checkpoint = load_training_checkpoint
resume_checkpoint = resume_training_checkpoint
