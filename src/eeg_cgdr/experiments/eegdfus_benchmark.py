"""Source-bounded EEGDfus benchmark adapter for EEGdenoiseNet.

The upstream repository is loaded dynamically from a frozen, Git-ignored
checkout.  Its model/DDPM/metric implementations are not vendored here.  Two
protocols are intentionally distinct:

``official_native``
    Keeps the upstream independent epoch permutations, 90/10 source split,
    eleven mixtures, and *post-mixing* row-level train/validation split.  The
    wrapper fixes a seed because upstream defines none, and reports the known
    train/validation source-epoch overlap rather than silently repairing it.

``strict_source_epoch``
    Freezes mutually exclusive clean/artifact source-epoch groups before any
    pairing or mixing.  EEGdenoiseNet has no participant identifiers, so these
    groups are never described as participants.

Both protocols train the dynamically loaded official conditional DDPM and a
deterministic condition-only arm built from the same dynamically loaded
backbone.  They see the same noisy condition, mixed pairs, split, batches, and
number of optimizer updates.  Tiny overrides are accepted only for the
explicit ``smoke`` stage; ``full`` enforces the upstream 4000/512/500 budget.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import signal
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)


BENCHMARK_ID = "eegdfus_eegdenoisenet_official_and_strict_v1"
OFFICIAL_COMMIT = "a19a652b3b6346188ae77067e1daf8b90cad005f"
OFFICIAL_EPOCHS = 4000
OFFICIAL_BATCH_SIZE = 512
OFFICIAL_DIFFUSION_STEPS = 500
OFFICIAL_COMBINATIONS = 11
OFFICIAL_TEST_SNR_DB = tuple(float(value) for value in np.linspace(-5.0, 5.0, 11))
EXPECTED_FULL_EVALUATION_MIXTURES = {
    ("official_native", "EOG"): 3740,
    ("official_native", "EMG"): 6160,
    ("strict_source_epoch", "EOG"): 4961,
    ("strict_source_epoch", "EMG"): 6149,
}
EXPECTED_FULL_OPTIMIZER_UPDATES = {
    ("official_native", "EOG"): 208_000,
    ("official_native", "EMG"): 344_000,
    ("strict_source_epoch", "EOG"): 276_000,
    ("strict_source_epoch", "EMG"): 344_000,
}
STRICT_OVERLAP_FIELDS = (
    "train_validation_clean_overlap",
    "train_validation_artifact_overlap",
    "train_evaluation_clean_overlap",
    "train_evaluation_artifact_overlap",
    "validation_evaluation_clean_overlap",
    "validation_evaluation_artifact_overlap",
)
TASK_MATRIX = (
    ("official_native", "EOG", "conditional_diffusion"),
    ("official_native", "EOG", "matched_deterministic"),
    ("official_native", "EMG", "conditional_diffusion"),
    ("official_native", "EMG", "matched_deterministic"),
    ("strict_source_epoch", "EOG", "conditional_diffusion"),
    ("strict_source_epoch", "EOG", "matched_deterministic"),
    ("strict_source_epoch", "EMG", "conditional_diffusion"),
    ("strict_source_epoch", "EMG", "matched_deterministic"),
)


@dataclass(frozen=True)
class EpochPairs:
    """Paired clean/noisy rows plus their real source-epoch identities."""

    clean: np.ndarray
    noisy: np.ndarray
    clean_source_epoch: np.ndarray
    artifact_source_epoch: np.ndarray
    snr_db: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.clean.shape[0])
        if self.clean.ndim != 2 or self.noisy.shape != self.clean.shape:
            raise ValueError("clean and noisy epochs must share shape (N,L)")
        if self.clean_source_epoch.shape != (count,) or self.artifact_source_epoch.shape != (
            count,
        ):
            raise ValueError("source-epoch identifiers must have one entry per pair")
        if self.snr_db.shape != (count,) or not np.isfinite(self.snr_db).all():
            raise ValueError("SNR values must have one finite entry per pair")
        if not np.isfinite(self.clean).all() or not np.isfinite(self.noisy).all():
            raise ValueError("prepared EEGDfus pairs contain NaN/Inf")


@dataclass(frozen=True)
class EvaluationLevel:
    snr_db: float
    pairs: EpochPairs


@dataclass(frozen=True)
class PreparedProtocol:
    protocol: str
    noise_type: str
    train: EpochPairs
    validation: EpochPairs
    evaluation: tuple[EvaluationLevel, ...]
    source_audit: Mapping[str, Any]


@dataclass(frozen=True)
class OfficialModules:
    ddpm_class: type[nn.Module]
    backbone_class: type[nn.Module]
    metrics: ModuleType


class MatchedConditionOnly(nn.Module):
    """Deterministic arm using the exact upstream denoising backbone.

    The noisy deployment condition is supplied to both upstream streams and a
    fixed conditioning scalar replaces diffusion time.  No latent noise or
    iterative sampling is available to this arm.
    """

    visible_inputs = ("noisy_single_channel_epoch",)

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, noisy: Tensor) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError("EEGDfus input must have shape (B,1,L)")
        fixed_condition = torch.ones(
            (noisy.shape[0], 1), dtype=noisy.dtype, device=noisy.device
        )
        return self.backbone(noisy, noisy, fixed_condition)


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def validate_eegdfus_config(config: Mapping[str, Any]) -> None:
    """Reject semantic drift before data or the external model is loaded."""

    if config.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError(f"benchmark_id must be {BENCHMARK_ID}")
    if int(config.get("harness_level", -1)) != 1:
        raise ValueError("EEGDfus benchmark requires HARNESS_LEVEL=1")
    if config.get("claim_scope") != "single_channel_EOG_EMG_stress_test_only":
        raise ValueError("EEGDfus cannot support participant-specific claims")

    source = _mapping(config, "source")
    if source.get("commit") != OFFICIAL_COMMIT:
        raise ValueError("EEGDfus external commit differs from the frozen source")
    if source.get("load_policy") != "dynamic_external_checkout_no_vendoring":
        raise ValueError("official source must be dynamically loaded, not copied")
    if source.get("upstream_license_file") != "absent_at_frozen_commit":
        raise ValueError("missing upstream license boundary must remain explicit")

    data = _mapping(config, "data")
    if data.get("identity_unit") != "source_epoch_not_participant":
        raise ValueError("EEGdenoiseNet source epochs must not be called participants")
    if int(data.get("epoch_samples", 0)) != 512:
        raise ValueError("the official EEGDfus architecture requires 512-sample epochs")
    expected_shapes = _mapping(data, "expected_shapes")
    if tuple(expected_shapes.get("clean_eeg", ())) != (4514, 512):
        raise ValueError("registered clean EEG shape differs from EEGdenoiseNet")
    if tuple(expected_shapes.get("eog", ())) != (3400, 512):
        raise ValueError("registered EOG shape differs from EEGdenoiseNet")
    if tuple(expected_shapes.get("emg", ())) != (5598, 512):
        raise ValueError("registered EMG shape differs from EEGdenoiseNet")

    protocols = _mapping(config, "protocols")
    native = _mapping(protocols, "official_native")
    strict = _mapping(protocols, "strict_source_epoch")
    for name, protocol in (("official_native", native), ("strict_source_epoch", strict)):
        training = _mapping(protocol, "training")
        diffusion = _mapping(protocol, "diffusion")
        if int(training.get("epochs", 0)) != OFFICIAL_EPOCHS:
            raise ValueError(f"{name} must preserve 4000 epochs")
        if int(training.get("batch_size", 0)) != OFFICIAL_BATCH_SIZE:
            raise ValueError(f"{name} must preserve batch_size=512")
        if int(training.get("combinations", 0)) != OFFICIAL_COMBINATIONS:
            raise ValueError(f"{name} must preserve eleven mixture combinations")
        if int(training.get("features", 0)) != 64:
            raise ValueError(f"{name} must preserve features=64")
        if float(training.get("learning_rate", 0.0)) != 1.0e-3:
            raise ValueError(f"{name} must preserve Adam learning_rate=1e-3")
        if int(training.get("scheduler_step_epochs", 0)) != 1500 or float(
            training.get("scheduler_gamma", 0.0)
        ) != 0.1:
            raise ValueError(f"{name} must preserve the upstream StepLR")
        if float(training.get("gradient_clip", 0.0)) != 1.0:
            raise ValueError(f"{name} must preserve gradient clipping at 1.0")
        if int(training.get("validation_interval_epochs", 0)) != 10:
            raise ValueError(f"{name} must preserve validation every 10 epochs")
        if int(training.get("evaluation_batch_size", 0)) != 64:
            raise ValueError(f"{name} must preserve evaluation batch_size=64")
        if training.get("drop_last") is not True:
            raise ValueError(f"{name} must preserve drop_last training batches")
        if int(diffusion.get("num_steps", 0)) != OFFICIAL_DIFFUSION_STEPS:
            raise ValueError(f"{name} must preserve 500 diffusion steps")
        if (
            diffusion.get("schedule") != "linear"
            or float(diffusion.get("beta_start", 0.0)) != 1.0e-4
            or float(diffusion.get("beta_end", 0.0)) != 0.02
        ):
            raise ValueError(f"{name} must preserve the upstream linear schedule")
        if training.get("early_stopping") is not False:
            raise ValueError("full matched arms must not stop before 4000 epochs")
        if training.get("mixed_precision") is not False:
            raise ValueError("the source-faithful benchmark uses upstream FP32 training")

    native_split = _mapping(native, "split")
    if float(native_split.get("train_fraction", 0.0)) != 0.9:
        raise ValueError("native source train fraction must remain 0.9")
    if float(native_split.get("validation_fraction_after_mixing", 0.0)) != 0.2:
        raise ValueError("native post-mixing validation fraction must remain 0.2")
    if native_split.get("known_train_validation_source_overlap") is not True:
        raise ValueError("native train/validation source overlap must be disclosed")
    if native_split.get("upstream_seed_defined") is not False:
        raise ValueError("upstream EEGDfus defines no random seed")
    adapter_seed = int(_mapping(config, "randomness").get("adapter_seed", -1))
    if int(native_split.get("adapter_seed", -2)) != adapter_seed:
        raise ValueError("native split seed differs from the registered adapter seed")

    strict_split = _mapping(strict, "split")
    fractions = tuple(float(value) for value in strict_split.get("fractions", ()))
    if fractions != (0.72, 0.18, 0.1):
        raise ValueError("strict source split must be frozen at 72/18/10 percent")
    if strict_split.get("split_before_pairing_and_mixing") is not True:
        raise ValueError("strict source epochs must be split before mixing")
    if int(strict_split.get("adapter_seed", -2)) != adapter_seed:
        raise ValueError("strict split seed differs from the registered adapter seed")

    matrix = tuple(tuple(str(item) for item in row) for row in config.get("task_matrix", ()))
    if matrix != TASK_MATRIX:
        raise ValueError("EEGDfus task matrix must contain the frozen eight arms")
    matched = _mapping(config, "matched_deterministic")
    if matched.get("optimizer_updates") != "exactly_equal_to_paired_diffusion_arm":
        raise ValueError("deterministic arm cannot receive fewer optimizer updates")
    if tuple(matched.get("visible_inputs", ())) != ("noisy_single_channel_epoch",):
        raise ValueError("matched arm input contract must equal deployment condition")
    execution = _mapping(config, "execution")
    if execution.get("mode") != "eegdfus-benchmark" or tuple(
        execution.get("stages", ())
    ) != ("cpu-tests", "smoke", "full", "aggregate-full"):
        raise ValueError("EEGDfus execution stages differ from the frozen route")
    accepted_job = execution.get("accepted_full_array_job_id")
    if accepted_job is not None and (
        isinstance(accepted_job, bool)
        or not isinstance(accepted_job, int)
        or accepted_job < 1
    ):
        raise ValueError("accepted EEGDfus full-array job ID must be a positive integer")
    accepted_tasks = execution.get("accepted_full_array_task_job_ids")
    if accepted_tasks is not None and (
        not isinstance(accepted_tasks, Mapping)
        or set(accepted_tasks) != set(range(len(TASK_MATRIX)))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in accepted_tasks.values()
        )
    ):
        raise ValueError("accepted EEGDfus array task-job mapping is invalid")
    accepted_head = execution.get("accepted_full_array_git_head")
    if accepted_head is not None and accepted_head != (
        "fd20ff2d6e69db4c05f888893787994b336cd1c3"
    ):
        raise ValueError("accepted EEGDfus producer Git HEAD changed")


def _validate_raw_arrays(clean: np.ndarray, artifact: np.ndarray) -> None:
    if clean.ndim != 2 or artifact.ndim != 2:
        raise ValueError("EEGdenoiseNet components must have shape (epochs,samples)")
    if clean.shape[1] != 512 or artifact.shape[1] != 512:
        raise ValueError("EEGdenoiseNet epochs must contain exactly 512 samples")
    if min(clean.shape[0], artifact.shape[0]) < 10:
        raise ValueError("too few source epochs for an EEGDfus split")
    if not np.isfinite(clean).all() or not np.isfinite(artifact).all():
        raise ValueError("EEGdenoiseNet source arrays contain NaN/Inf")


def _repeat_random_permutations(
    values: np.ndarray,
    source_ids: np.ndarray,
    *,
    combinations: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    if combinations < 1 or len(values) != len(source_ids):
        raise ValueError("invalid permutation inputs")
    rows: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    for _ in range(combinations):
        order = rng.permutation(len(values))
        rows.append(values[order])
        identities.append(source_ids[order])
    return np.concatenate(rows, axis=0), np.concatenate(identities, axis=0)


def _standardize_and_mix(
    clean: np.ndarray,
    artifact: np.ndarray,
    snr_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if clean.shape != artifact.shape or snr_db.shape != (clean.shape[0],):
        raise ValueError("mixing arrays have incompatible shapes")
    # sklearn.preprocessing.scale is the exact primitive used upstream.
    clean_scaled = np.asarray(scale(clean, axis=1), dtype=np.float64)
    artifact_scaled = np.asarray(scale(artifact, axis=1), dtype=np.float64)
    clean_power = np.mean(np.square(clean_scaled), axis=1)
    artifact_power = np.mean(np.square(artifact_scaled), axis=1)
    if bool((artifact_power <= 0.0).any()) or bool((clean_power <= 0.0).any()):
        raise ValueError("zero-power source epoch cannot be mixed")
    snr_amplitude = np.sqrt(np.power(10.0, 0.1 * snr_db))
    # Preserve upstream get_rms semantics: it returns mean square, not RMS.
    coefficient = clean_power / (artifact_power * snr_amplitude)
    noisy = clean_scaled + artifact_scaled * coefficient[:, None]
    return clean_scaled.astype(np.float32), noisy.astype(np.float32)


def _pairs_from_components(
    clean: np.ndarray,
    artifact: np.ndarray,
    clean_ids: np.ndarray,
    artifact_ids: np.ndarray,
    snr_db: np.ndarray,
) -> EpochPairs:
    clean_scaled, noisy = _standardize_and_mix(clean, artifact, snr_db)
    return EpochPairs(
        clean=clean_scaled,
        noisy=noisy,
        clean_source_epoch=np.asarray(clean_ids, dtype=np.int64),
        artifact_source_epoch=np.asarray(artifact_ids, dtype=np.int64),
        snr_db=np.asarray(snr_db, dtype=np.float64),
    )


def _select_pairs(pairs: EpochPairs, index: np.ndarray) -> EpochPairs:
    return EpochPairs(
        clean=pairs.clean[index],
        noisy=pairs.noisy[index],
        clean_source_epoch=pairs.clean_source_epoch[index],
        artifact_source_epoch=pairs.artifact_source_epoch[index],
        snr_db=pairs.snr_db[index],
    )


def _source_sets(pairs: EpochPairs) -> tuple[set[int], set[int]]:
    return (
        set(int(value) for value in pairs.clean_source_epoch),
        set(int(value) for value in pairs.artifact_source_epoch),
    )


def _protocol_source_audit(
    train: EpochPairs, validation: EpochPairs, evaluation: EpochPairs
) -> dict[str, Any]:
    train_clean, train_artifact = _source_sets(train)
    validation_clean, validation_artifact = _source_sets(validation)
    evaluation_clean, evaluation_artifact = _source_sets(evaluation)
    return {
        "identity_unit": "source_epoch_not_participant",
        "train_validation_clean_overlap": len(train_clean & validation_clean),
        "train_validation_artifact_overlap": len(
            train_artifact & validation_artifact
        ),
        "train_evaluation_clean_overlap": len(train_clean & evaluation_clean),
        "train_evaluation_artifact_overlap": len(train_artifact & evaluation_artifact),
        "validation_evaluation_clean_overlap": len(
            validation_clean & evaluation_clean
        ),
        "validation_evaluation_artifact_overlap": len(
            validation_artifact & evaluation_artifact
        ),
        "train_clean_source_epochs": len(train_clean),
        "train_artifact_source_epochs": len(train_artifact),
        "validation_clean_source_epochs": len(validation_clean),
        "validation_artifact_source_epochs": len(validation_artifact),
        "evaluation_clean_source_epochs": len(evaluation_clean),
        "evaluation_artifact_source_epochs": len(evaluation_artifact),
    }


def prepare_official_native(
    clean: np.ndarray,
    artifact: np.ndarray,
    *,
    noise_type: str,
    seed: int,
    combinations: int = OFFICIAL_COMBINATIONS,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.2,
    test_snr_db: Sequence[float] = OFFICIAL_TEST_SNR_DB,
) -> PreparedProtocol:
    """Reproduce upstream EOG/EMG pairing and disclose its validation leak."""

    _validate_raw_arrays(clean, artifact)
    if noise_type not in {"EOG", "EMG"}:
        raise ValueError("noise_type must be EOG or EMG")
    original_clean_count = len(clean)
    rng = np.random.RandomState(seed)
    clean_ids = rng.permutation(clean.shape[0]).astype(np.int64)
    artifact_ids = rng.permutation(artifact.shape[0]).astype(np.int64)
    clean_random = np.asarray(clean[clean_ids], dtype=np.float64)
    artifact_random = np.asarray(artifact[artifact_ids], dtype=np.float64)

    if noise_type == "EOG":
        if len(clean_random) < len(artifact_random):
            raise ValueError("native EOG path expects at least as many EEG epochs")
        clean_random = clean_random[: len(artifact_random)]
        clean_ids = clean_ids[: len(artifact_random)]
    else:
        reuse = len(artifact_random) - len(clean_random)
        if reuse < 0 or reuse > len(clean_random):
            raise ValueError("native EMG reuse rule is undefined for these counts")
        clean_random = np.concatenate((clean_random[:reuse], clean_random), axis=0)
        clean_ids = np.concatenate((clean_ids[:reuse], clean_ids), axis=0)

    count = len(artifact_random)
    train_count = round(train_fraction * count)
    base_train = slice(0, train_count)
    base_evaluation = slice(train_count, count)

    train_clean, train_clean_ids = _repeat_random_permutations(
        clean_random[base_train],
        clean_ids[base_train],
        combinations=combinations,
        rng=rng,
    )
    train_artifact, train_artifact_ids = _repeat_random_permutations(
        artifact_random[base_train],
        artifact_ids[base_train],
        combinations=combinations,
        rng=rng,
    )
    evaluation_clean, evaluation_clean_ids = _repeat_random_permutations(
        clean_random[base_evaluation],
        clean_ids[base_evaluation],
        combinations=combinations,
        rng=rng,
    )
    evaluation_artifact, evaluation_artifact_ids = _repeat_random_permutations(
        artifact_random[base_evaluation],
        artifact_ids[base_evaluation],
        combinations=combinations,
        rng=rng,
    )

    train_snr = rng.uniform(-5.0, 5.0, len(train_clean))
    mixed_train = _pairs_from_components(
        train_clean,
        train_artifact,
        train_clean_ids,
        train_artifact_ids,
        train_snr,
    )
    training_index, validation_index = train_test_split(
        np.arange(len(mixed_train.clean)),
        test_size=validation_fraction,
        random_state=rng,
    )
    training = _select_pairs(mixed_train, np.asarray(training_index, dtype=np.int64))
    validation = _select_pairs(
        mixed_train, np.asarray(validation_index, dtype=np.int64)
    )

    evaluation_levels = tuple(
        EvaluationLevel(
            snr_db=float(level),
            pairs=_pairs_from_components(
                evaluation_clean,
                evaluation_artifact,
                evaluation_clean_ids,
                evaluation_artifact_ids,
                np.full(len(evaluation_clean), float(level), dtype=np.float64),
            ),
        )
        for level in test_snr_db
    )
    audit = _protocol_source_audit(training, validation, evaluation_levels[0].pairs)
    audit.update(
        {
            "upstream_seed_defined": False,
            "adapter_seed": int(seed),
            "split_semantics": "upstream_post_mixing_row_level_train_validation",
            "train_validation_leakage_preserved_not_repaired": True,
            "unused_clean_source_epochs": original_clean_count
            - len(
                set(int(value) for value in training.clean_source_epoch)
                | set(int(value) for value in validation.clean_source_epoch)
                | set(
                    int(value)
                    for value in evaluation_levels[0].pairs.clean_source_epoch
                )
            ),
            "unused_artifact_source_epochs": 0,
        }
    )
    return PreparedProtocol(
        protocol="official_native",
        noise_type=noise_type,
        train=training,
        validation=validation,
        evaluation=evaluation_levels,
        source_audit=audit,
    )


def _partition_source_ids(
    count: int,
    *,
    fractions: tuple[float, float, float],
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    if count < 10 or not math.isclose(sum(fractions), 1.0, abs_tol=1.0e-12):
        raise ValueError("invalid strict source-epoch split")
    order = rng.permutation(count).astype(np.int64)
    train_count = round(fractions[0] * count)
    validation_count = round(fractions[1] * count)
    if train_count < 1 or validation_count < 1 or train_count + validation_count >= count:
        raise ValueError("strict split leaves an empty partition")
    return {
        "train": order[:train_count],
        "validation": order[train_count : train_count + validation_count],
        "evaluation": order[train_count + validation_count :],
    }


def _match_component_count_within_partition(
    component: np.ndarray,
    component_ids: np.ndarray,
    *,
    target_count: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    if target_count < 1 or len(component) < 1:
        raise ValueError("empty strict source partition")
    rows: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    remaining = target_count
    while remaining > 0:
        order = rng.permutation(len(component))
        take = min(remaining, len(component))
        rows.append(component[order[:take]])
        identities.append(component_ids[order[:take]])
        remaining -= take
    return np.concatenate(rows, axis=0), np.concatenate(identities, axis=0)


def _strict_partition_components(
    clean: np.ndarray,
    artifact: np.ndarray,
    clean_ids: np.ndarray,
    artifact_ids: np.ndarray,
    *,
    combinations: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_count = max(len(clean), len(artifact))
    matched_clean, matched_clean_ids = _match_component_count_within_partition(
        clean,
        clean_ids,
        target_count=target_count,
        rng=rng,
    )
    matched_artifact, matched_artifact_ids = _match_component_count_within_partition(
        artifact,
        artifact_ids,
        target_count=target_count,
        rng=rng,
    )
    repeated_clean, repeated_clean_ids = _repeat_random_permutations(
        matched_clean,
        matched_clean_ids,
        combinations=combinations,
        rng=rng,
    )
    repeated_artifact, repeated_artifact_ids = _repeat_random_permutations(
        matched_artifact,
        matched_artifact_ids,
        combinations=combinations,
        rng=rng,
    )
    return (
        repeated_clean,
        repeated_artifact,
        repeated_clean_ids,
        repeated_artifact_ids,
    )


def _strict_partition_pairs(
    clean: np.ndarray,
    artifact: np.ndarray,
    clean_ids: np.ndarray,
    artifact_ids: np.ndarray,
    *,
    combinations: int,
    rng: np.random.RandomState,
    snr_db: np.ndarray,
) -> EpochPairs:
    components = _strict_partition_components(
        clean,
        artifact,
        clean_ids,
        artifact_ids,
        combinations=combinations,
        rng=rng,
    )
    if len(snr_db) != len(components[0]):
        raise ValueError("strict SNR schedule length differs from mixture count")
    return _pairs_from_components(
        components[0], components[1], components[2], components[3], snr_db
    )


def prepare_strict_source_epoch(
    clean: np.ndarray,
    artifact: np.ndarray,
    *,
    noise_type: str,
    seed: int,
    combinations: int = OFFICIAL_COMBINATIONS,
    fractions: tuple[float, float, float] = (0.72, 0.18, 0.1),
    test_snr_db: Sequence[float] = OFFICIAL_TEST_SNR_DB,
) -> PreparedProtocol:
    """Split each real component library before any pairing or augmentation."""

    _validate_raw_arrays(clean, artifact)
    if noise_type not in {"EOG", "EMG"}:
        raise ValueError("noise_type must be EOG or EMG")
    rng = np.random.RandomState(seed)
    clean_groups = _partition_source_ids(
        len(clean), fractions=fractions, rng=rng
    )
    artifact_groups = _partition_source_ids(
        len(artifact), fractions=fractions, rng=rng
    )

    def build(split: str, snr_values: np.ndarray) -> EpochPairs:
        clean_ids = clean_groups[split]
        artifact_ids = artifact_groups[split]
        return _strict_partition_pairs(
            np.asarray(clean[clean_ids], dtype=np.float64),
            np.asarray(artifact[artifact_ids], dtype=np.float64),
            clean_ids,
            artifact_ids,
            combinations=combinations,
            rng=rng,
            snr_db=snr_values,
        )

    train_pair_count = max(
        len(clean_groups["train"]), len(artifact_groups["train"])
    ) * combinations
    validation_pair_count = max(
        len(clean_groups["validation"]), len(artifact_groups["validation"])
    ) * combinations
    training = build("train", rng.uniform(-5.0, 5.0, train_pair_count))
    validation = build(
        "validation", rng.uniform(-5.0, 5.0, validation_pair_count)
    )
    evaluation_clean_ids = clean_groups["evaluation"]
    evaluation_artifact_ids = artifact_groups["evaluation"]
    evaluation_components = _strict_partition_components(
        np.asarray(clean[evaluation_clean_ids], dtype=np.float64),
        np.asarray(artifact[evaluation_artifact_ids], dtype=np.float64),
        evaluation_clean_ids,
        evaluation_artifact_ids,
        combinations=combinations,
        rng=rng,
    )
    evaluation_pair_count = len(evaluation_components[0])
    evaluation_levels = tuple(
        EvaluationLevel(
            snr_db=float(level),
            pairs=_pairs_from_components(
                evaluation_components[0],
                evaluation_components[1],
                evaluation_components[2],
                evaluation_components[3],
                np.full(evaluation_pair_count, float(level), dtype=np.float64),
            ),
        )
        for level in test_snr_db
    )
    audit = _protocol_source_audit(training, validation, evaluation_levels[0].pairs)
    if any(
        int(audit[key]) != 0
        for key in (
            "train_validation_clean_overlap",
            "train_validation_artifact_overlap",
            "train_evaluation_clean_overlap",
            "train_evaluation_artifact_overlap",
            "validation_evaluation_clean_overlap",
            "validation_evaluation_artifact_overlap",
        )
    ):
        raise AssertionError("strict source-epoch split leaked across partitions")
    audit.update(
        {
            "upstream_seed_defined": False,
            "adapter_seed": int(seed),
            "split_semantics": "source_epoch_groups_frozen_before_pairing_and_mixing",
            "train_validation_leakage_preserved_not_repaired": False,
            "unused_clean_source_epochs": 0,
            "unused_artifact_source_epochs": 0,
        }
    )
    return PreparedProtocol(
        protocol="strict_source_epoch",
        noise_type=noise_type,
        train=training,
        validation=validation,
        evaluation=evaluation_levels,
        source_audit=audit,
    )


def source_split_manifest_rows(prepared: PreparedProtocol) -> list[dict[str, Any]]:
    """Return source-level rows; deliberately contains no participant field."""

    memberships: dict[tuple[str, int], set[str]] = defaultdict(set)
    for split, pairs in (
        ("train", prepared.train),
        ("validation", prepared.validation),
        ("evaluation", prepared.evaluation[0].pairs),
    ):
        for value in np.unique(pairs.clean_source_epoch):
            memberships[("clean_EEG", int(value))].add(split)
        for value in np.unique(pairs.artifact_source_epoch):
            memberships[(prepared.noise_type, int(value))].add(split)
    return [
        {
            "dataset": "EEGdenoiseNet",
            "protocol": prepared.protocol,
            "noise_type": prepared.noise_type,
            "identity_unit": "source_epoch_not_participant",
            "source_kind": kind,
            "source_epoch": source_epoch,
            "split_membership": "+".join(sorted(splits)),
        }
        for (kind, source_epoch), splits in sorted(memberships.items())
    ]


def _load_external_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load frozen EEGDfus module: {path}")
    module = importlib.util.module_from_spec(specification)
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
    return module


def validate_official_checkout(
    source_root: Path, *, expected_commit: str
) -> dict[str, Any]:
    """Read-only validation of the ignored source checkout and its boundary."""

    root = source_root.resolve(strict=True)
    if root != Path("/home/infres/yinwang/denoiseNet/.external/EEGDfus"):
        raise ValueError("unexpected EEGDfus external checkout path")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual_commit = completed.stdout.strip()
    if actual_commit != expected_commit or expected_commit != OFFICIAL_COMMIT:
        raise RuntimeError("EEGDfus external checkout is not at the frozen commit")
    tracked_status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status.strip():
        raise RuntimeError("EEGDfus frozen checkout has tracked modifications")
    required = {
        "DDPM.py": root / "DDPM.py",
        "denoising_model_eegdnet.py": root / "denoising_model_eegdnet.py",
        "metrics.py": root / "metrics.py",
    }
    if any(not path.is_file() for path in required.values()):
        raise FileNotFoundError("frozen EEGDfus checkout lacks a required source file")
    license_present = any(
        (root / candidate).is_file()
        for candidate in ("LICENSE", "LICENSE.txt", "COPYING", "NOTICE")
    )
    if license_present:
        raise RuntimeError(
            "EEGDfus upstream license-file status changed; re-audit before use"
        )
    return {
        "root": str(root),
        "commit": actual_commit,
        "tracked_checkout_clean": True,
        "required_files_present": True,
        "upstream_license_file": "absent_at_frozen_commit",
    }


def load_official_modules(source_root: Path, *, expected_commit: str) -> OfficialModules:
    """Load only the frozen external implementation after a read-only Git check."""

    source_status = validate_official_checkout(
        source_root, expected_commit=expected_commit
    )
    root = Path(str(source_status["root"]))
    required = {
        "DDPM.py": root / "DDPM.py",
        "denoising_model_eegdnet.py": root / "denoising_model_eegdnet.py",
        "metrics.py": root / "metrics.py",
    }
    ddpm_module = _load_external_module("_eegdfus_official_ddpm", required["DDPM.py"])
    backbone_module = _load_external_module(
        "_eegdfus_official_eegdnet_model", required["denoising_model_eegdnet.py"]
    )
    metrics_module = _load_external_module(
        "_eegdfus_official_metrics", required["metrics.py"]
    )
    return OfficialModules(
        ddpm_class=ddpm_module.DDPM,
        backbone_class=backbone_module.DualBranchDenoisingModel,
        metrics=metrics_module,
    )


def audit_ssed_source_text(train_text: str, preparation_text: str) -> tuple[str, ...]:
    """Detect, but never repair, known upstream SSED indexing/input issues."""

    findings: list[str] = []
    if (
        "train_test_split(list(range(len(val_test_idx)))" in train_text
        and "Subset(train_val_set, val_idx)" in train_text
        and "Subset(train_val_set, test_idx)" in train_text
    ):
        findings.append(
            "ssed_holdout_indices_are_rebased_then_applied_to_full_dataset"
        )
    if (
        "dataset = [eeg_train, noise_train]" in preparation_text
        and "TensorDataset(y_train, X_train)" in train_text
    ):
        # The training tuple is consumed inside upstream utils.py, not here.
        findings.append("ssed_tensor_dataset_orders_noise_before_clean")
    if "test_loader = DataLoader" in train_text and "train(model" in train_text:
        findings.append("ssed_test_loader_is_constructed_but_not_evaluated")
    return tuple(findings)


def run_eegdfus_cpu_validation(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Validate source/config/data headers on one scheduled CPU job."""

    validate_eegdfus_config(config)
    source = _mapping(config, "source")
    source_status = validate_official_checkout(
        Path(str(source["root"])), expected_commit=str(source["commit"])
    )
    source_root = Path(str(source_status["root"]))
    findings = audit_ssed_source_text(
        (source_root / "train_ssed.py").read_text(encoding="utf-8"),
        (source_root / "Data_Preparation" / "data_prepare_ssed.py").read_text(
            encoding="utf-8"
        ),
    )
    expected_findings = (
        "ssed_holdout_indices_are_rebased_then_applied_to_full_dataset",
        "ssed_tensor_dataset_orders_noise_before_clean",
        "ssed_test_loader_is_constructed_but_not_evaluated",
    )
    if findings != expected_findings:
        raise RuntimeError("frozen EEGDfus SSED audit findings changed")
    utility_text = (source_root / "utils.py").read_text(encoding="utf-8")
    if "(clean_batch, noisy_batch)" not in utility_text:
        raise RuntimeError("frozen EEGDfus training tuple semantics changed")

    data = _mapping(config, "data")
    expected_shapes = _mapping(data, "expected_shapes")
    header_shapes: dict[str, list[int]] = {}
    for key, path_key in (("clean_eeg", "clean_eeg"), ("eog", "eog"), ("emg", "emg")):
        array = np.load(Path(str(data[path_key])), mmap_mode="r", allow_pickle=False)
        actual_shape = tuple(int(value) for value in array.shape)
        expected_shape = tuple(int(value) for value in expected_shapes[key])
        if actual_shape != expected_shape or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"registered EEGdenoiseNet header mismatch for {key}")
        header_shapes[key] = list(actual_shape)

    result = {
        "status": "cpu_semantics_validated",
        "benchmark_id": BENCHMARK_ID,
        "claim_scope": "engineering_validation_only_no_benchmark_result",
        "source": source_status,
        "data_header_shapes": header_shapes,
        "identity_unit": "source_epoch_not_participant",
        "ssed_findings": list(findings),
        "native_issue_policy": "preserve_and_report_do_not_repair",
        "full_budget": {
            "epochs": OFFICIAL_EPOCHS,
            "batch_size": OFFICIAL_BATCH_SIZE,
            "mixtures": OFFICIAL_COMBINATIONS,
            "diffusion_steps": OFFICIAL_DIFFUSION_STEPS,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _load_data(config: Mapping[str, Any], noise_type: str) -> tuple[np.ndarray, np.ndarray]:
    data = _mapping(config, "data")
    clean = np.load(Path(str(data["clean_eeg"])), mmap_mode="r", allow_pickle=False)
    artifact_key = "eog" if noise_type == "EOG" else "emg"
    artifact = np.load(
        Path(str(data[artifact_key])), mmap_mode="r", allow_pickle=False
    )
    clean_value = np.asarray(clean, dtype=np.float64)
    artifact_value = np.asarray(artifact, dtype=np.float64)
    _validate_raw_arrays(clean_value, artifact_value)
    expected_shapes = _mapping(data, "expected_shapes")
    expected_clean = tuple(int(value) for value in expected_shapes["clean_eeg"])
    expected_artifact = tuple(int(value) for value in expected_shapes[artifact_key])
    if clean_value.shape != expected_clean or artifact_value.shape != expected_artifact:
        raise ValueError("EEGdenoiseNet array shape differs from the registered release")
    return clean_value, artifact_value


def _prepare_from_config(
    config: Mapping[str, Any],
    *,
    protocol: str,
    noise_type: str,
    stage: str,
) -> PreparedProtocol:
    clean, artifact = _load_data(config, noise_type)
    protocol_config = _mapping(_mapping(config, "protocols"), protocol)
    training = _mapping(protocol_config, "training")
    split = _mapping(protocol_config, "split")
    seed = int(split["adapter_seed"])
    combinations = int(training["combinations"])
    test_levels: Sequence[float] = OFFICIAL_TEST_SNR_DB
    if stage == "smoke":
        smoke = _mapping(config, "smoke_only_overrides")
        source_count = int(smoke["source_epochs_per_component"])
        clean = clean[:source_count]
        artifact = artifact[:source_count]
        combinations = int(smoke["combinations"])
        test_levels = tuple(float(value) for value in smoke["test_snr_db"])
    if protocol == "official_native":
        return prepare_official_native(
            clean,
            artifact,
            noise_type=noise_type,
            seed=seed,
            combinations=combinations,
            train_fraction=float(split["train_fraction"]),
            validation_fraction=float(split["validation_fraction_after_mixing"]),
            test_snr_db=test_levels,
        )
    if protocol == "strict_source_epoch":
        return prepare_strict_source_epoch(
            clean,
            artifact,
            noise_type=noise_type,
            seed=seed,
            combinations=combinations,
            fractions=tuple(float(value) for value in split["fractions"]),
            test_snr_db=test_levels,
        )
    raise ValueError(f"unknown EEGDfus protocol: {protocol}")


def _epoch_pair_identity_equal(left: EpochPairs, right: EpochPairs) -> bool:
    return all(
        np.array_equal(left_value, right_value)
        for left_value, right_value in (
            (left.clean_source_epoch, right.clean_source_epoch),
            (left.artifact_source_epoch, right.artifact_source_epoch),
            (left.snr_db, right.snr_db),
        )
    )


def _prepared_pairing_equal(left: PreparedProtocol, right: PreparedProtocol) -> bool:
    if (
        left.protocol != right.protocol
        or left.noise_type != right.noise_type
        or len(left.evaluation) != len(right.evaluation)
        or not _epoch_pair_identity_equal(left.train, right.train)
        or not _epoch_pair_identity_equal(left.validation, right.validation)
    ):
        return False
    return all(
        left_level.snr_db == right_level.snr_db
        and _epoch_pair_identity_equal(left_level.pairs, right_level.pairs)
        for left_level, right_level in zip(left.evaluation, right.evaluation, strict=True)
    )


def _official_model_config(
    protocol_config: Mapping[str, Any], *, stage: str, smoke: Mapping[str, Any]
) -> dict[str, Any]:
    training = dict(_mapping(protocol_config, "training"))
    diffusion = dict(_mapping(protocol_config, "diffusion"))
    if stage == "smoke":
        training["epochs"] = int(smoke["epochs"])
        training["batch_size"] = int(smoke["batch_size"])
        diffusion["num_steps"] = int(smoke["diffusion_steps"])
    return {"train": training, "diffusion": diffusion}


def _build_model(
    modules: OfficialModules,
    model_config: Mapping[str, Any],
    *,
    arm: str,
    device: torch.device,
) -> nn.Module:
    features = int(_mapping(model_config, "train")["features"])
    backbone = modules.backbone_class(features).to(device)
    if arm == "conditional_diffusion":
        return modules.ddpm_class(backbone, dict(model_config), device).to(device)
    if arm == "matched_deterministic":
        return MatchedConditionOnly(backbone).to(device)
    raise ValueError(f"unknown EEGDfus arm: {arm}")


def _data_loader(
    pairs: EpochPairs,
    *,
    batch_size: int,
    workers: int,
    shuffle_seed: int | None,
    drop_last: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(pairs.clean).unsqueeze(1),
        torch.from_numpy(pairs.noisy).unsqueeze(1),
    )
    generator = None
    shuffle = shuffle_seed is not None
    if shuffle_seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(shuffle_seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )


def _batch_loss(model: nn.Module, arm: str, clean: Tensor, noisy: Tensor) -> Tensor:
    if arm == "conditional_diffusion":
        return model(clean, noisy)
    restored = model(noisy)
    return torch.nn.functional.l1_loss(restored, clean, reduction="sum")


def _validation_loss(
    model: nn.Module,
    arm: str,
    pairs: EpochPairs,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> float:
    loader = _data_loader(
        pairs,
        batch_size=batch_size,
        workers=workers,
        shuffle_seed=None,
        drop_last=True,
    )
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for clean, noisy in loader:
            loss = _batch_loss(model, arm, clean.to(device), noisy.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite EEGDfus validation loss")
            total += float(loss)
            batches += 1
    if batches == 0:
        raise RuntimeError("EEGDfus validation has no complete batch")
    return total / batches


def _checkpoint_contract(
    config: Mapping[str, Any],
    *,
    protocol: str,
    noise_type: str,
    arm: str,
    stage: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "source_commit": OFFICIAL_COMMIT,
        "dataset": "EEGdenoiseNet",
        "identity_unit": "source_epoch_not_participant",
        "protocol": protocol,
        "noise_type": noise_type,
        "arm": arm,
        "stage": stage,
        "model_config": dict(model_config),
        "randomness": dict(_mapping(config, "randomness")),
    }


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "protocol",
        "noise_type",
        "identity_unit",
        "source_kind",
        "source_epoch",
        "split_membership",
    )
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty EEGDfus metrics")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def eegdfus_rrmse_s_corrected_denominator_shape(
    denoised: np.ndarray,
    clean: np.ndarray,
    *,
    get_psd: Any,
) -> float:
    """Evaluate spectral RRMSE with a PSD-shaped zero denominator.

    Frozen upstream ``RRMSE_s`` computes a 400-bin PSD numerator but passes
    ``zeros(clean.shape)`` (512 samples) to its denominator.  Recent sklearn
    rejects that mismatch.  We leave the official function untouched and give
    this one-shape compatibility result an explicit non-official name.
    """

    clean_psd = np.asarray(get_psd(np.asarray(clean).squeeze()), dtype=np.float64)
    denoised_psd = np.asarray(
        get_psd(np.asarray(denoised).squeeze()), dtype=np.float64
    )
    if clean_psd.shape != denoised_psd.shape or clean_psd.size < 1:
        raise ValueError("EEGDfus corrected PSD arrays have incompatible shapes")
    numerator = float(np.sqrt(np.mean(np.square(denoised_psd - clean_psd))))
    denominator = float(np.sqrt(np.mean(np.square(clean_psd))))
    if not math.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError("EEGDfus corrected spectral denominator is degenerate")
    value = numerator / denominator
    if not math.isfinite(value):
        raise FloatingPointError("EEGDfus corrected spectral RRMSE is non-finite")
    return value


def _evaluate(
    model: nn.Module,
    modules: OfficialModules,
    prepared: PreparedProtocol,
    *,
    arm: str,
    diffusion_steps: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for level in prepared.evaluation:
        loader = _data_loader(
            level.pairs,
            batch_size=batch_size,
            workers=workers,
            shuffle_seed=None,
            drop_last=False,
        )
        outputs: list[np.ndarray] = []
        start = time.perf_counter()
        with torch.no_grad():
            for _clean, noisy in loader:
                noisy_device = noisy.to(device)
                if arm == "conditional_diffusion":
                    restored = model.denoising(noisy_device)
                else:
                    restored = model(noisy_device)
                if not bool(torch.isfinite(restored).all()):
                    raise FloatingPointError("non-finite EEGDfus evaluation output")
                outputs.append(restored.detach().cpu().numpy()[:, 0, :])
        elapsed = time.perf_counter() - start
        output = np.concatenate(outputs, axis=0)
        clean = level.pairs.clean
        noisy = level.pairs.noisy
        rows.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "protocol": prepared.protocol,
                "noise_type": prepared.noise_type,
                "arm": arm,
                "identity_unit": "source_epoch_not_participant",
                "snr_db": level.snr_db,
                "evaluation_mixtures": len(clean),
                "snr_improvement_db": float(
                    modules.metrics.SNR_improvement(noisy, output, clean)
                ),
                "correlation": float(modules.metrics.CC(output, clean)),
                "rrmse_temporal": float(modules.metrics.RRMSE(output, clean)),
                "rrmse_spectral_official": "",
                "rrmse_spectral_official_status": (
                    "blocked_upstream_zero_denominator_shape_400_vs_512"
                ),
                "rrmse_spectral_corrected_psd_denominator_shape": (
                    eegdfus_rrmse_s_corrected_denominator_shape(
                        output,
                        clean,
                        get_psd=modules.metrics.get_PSD,
                    )
                ),
                "evaluation_seconds": elapsed,
                "network_calls_per_output": (
                    diffusion_steps if arm == "conditional_diffusion" else 1
                ),
            }
        )
    return rows


def _task_output_paths(
    config: Mapping[str, Any],
    *,
    stage: str,
    protocol: str,
    noise_type: str,
    arm: str,
) -> tuple[Path, Path, Path]:
    outputs = _mapping(config, "outputs")
    result_root = Path(str(outputs["result_root"])).resolve()
    checkpoint_root = Path(str(outputs["checkpoint_root"])).resolve()
    expected_code_root = Path("/home/infres/yinwang/denoiseNet")
    if (
        expected_code_root not in result_root.parents
        or expected_code_root not in checkpoint_root.parents
    ):
        raise ValueError("EEGDfus outputs must remain under the code root")
    task_name = f"{protocol}/{noise_type.lower()}/{arm}"
    return (
        result_root / stage / task_name,
        checkpoint_root / stage / task_name / "last.pt",
        checkpoint_root / stage / task_name / "best.pt",
    )


def run_eegdfus_stage(
    config: Mapping[str, Any],
    *,
    stage: str,
    task_index: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train/resume/evaluate one frozen matrix cell on one scheduled GPU."""

    validate_eegdfus_config(config)
    if stage not in {"smoke", "full"}:
        raise ValueError("EEGDfus stage must be smoke or full")
    if not 0 <= int(task_index) < len(TASK_MATRIX):
        raise ValueError("EEGDfus task index must lie in [0,7]")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("EEGDfus benchmark requires a scheduled CUDA allocation")

    protocol, noise_type, arm = TASK_MATRIX[int(task_index)]
    protocol_config = _mapping(_mapping(config, "protocols"), protocol)
    smoke = _mapping(config, "smoke_only_overrides")
    model_config = _official_model_config(protocol_config, stage=stage, smoke=smoke)
    training = _mapping(model_config, "train")
    if stage == "full" and (
        int(training["epochs"]) != OFFICIAL_EPOCHS
        or int(training["batch_size"]) != OFFICIAL_BATCH_SIZE
        or int(_mapping(model_config, "diffusion")["num_steps"])
        != OFFICIAL_DIFFUSION_STEPS
    ):
        raise AssertionError("full EEGDfus budget differs from frozen upstream semantics")

    seed = int(_mapping(config, "randomness")["adapter_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prepared = _prepare_from_config(
        config, protocol=protocol, noise_type=noise_type, stage=stage
    )
    modules = load_official_modules(
        Path(str(_mapping(config, "source")["root"])),
        expected_commit=str(_mapping(config, "source")["commit"]),
    )
    model = _build_model(modules, model_config, arm=arm, device=device)
    optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["scheduler_step_epochs"]),
        gamma=float(training["scheduler_gamma"]),
    )
    output_dir, checkpoint, best_checkpoint = _task_output_paths(
        config,
        stage=stage,
        protocol=protocol,
        noise_type=noise_type,
        arm=arm,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **dict(config),
        "resolved_task": {
            "task_index": int(task_index),
            "stage": stage,
            "protocol": protocol,
            "noise_type": noise_type,
            "arm": arm,
            "model_config": model_config,
        },
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _write_manifest(
        output_dir / "split_manifest.csv", source_split_manifest_rows(prepared)
    )

    contract = _checkpoint_contract(
        config,
        protocol=protocol,
        noise_type=noise_type,
        arm=arm,
        stage=stage,
        model_config=model_config,
    )
    start_epoch = 0
    global_updates = 0
    best_validation_loss = float("inf")
    resumed = False
    if bool(training["resume"]) and checkpoint.is_file():
        state = resume_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=contract,
            map_location=device,
        )
        if state.normalizer_state != {"kind": "upstream_per_epoch_scale"}:
            raise ValueError("EEGDfus checkpoint preprocessing state mismatch")
        start_epoch = state.epoch + 1
        global_updates = state.step
        best_validation_loss = float(
            state.extra.get("best_validation_loss", float("inf"))
        )
        resumed = True

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGUSR1, request_stop)
    epochs = int(training["epochs"])
    batch_size = int(training["batch_size"])
    workers = int(training["workers"])
    validation_interval = int(training["validation_interval_epochs"])
    checkpoint_interval = int(training["checkpoint_interval_epochs"])
    last_epoch = start_epoch - 1
    last_validation_loss = float("nan")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            loader = _data_loader(
                prepared.train,
                batch_size=batch_size,
                workers=workers,
                shuffle_seed=seed + 10_000 + epoch,
                drop_last=True,
            )
            model.train()
            batches = 0
            for clean_batch, noisy_batch in loader:
                optimizer.zero_grad()
                loss = _batch_loss(
                    model,
                    arm,
                    clean_batch.to(device),
                    noisy_batch.to(device),
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite EEGDfus loss at epoch={epoch} update={global_updates}"
                    )
                loss.backward()
                parameters = (
                    model.model.parameters()
                    if arm == "conditional_diffusion"
                    else model.parameters()
                )
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(training["gradient_clip"])
                )
                optimizer.step()
                global_updates += 1
                batches += 1
            if batches == 0:
                raise RuntimeError("EEGDfus training has no complete batch")
            scheduler.step()
            last_epoch = epoch

            validate_now = (
                (epoch + 1) % validation_interval == 0
                or stop_requested
                or epoch + 1 == epochs
            )
            improved = False
            if validate_now:
                # Fixed validation randomness without consuming the subsequent
                # training stream.
                python_state = random.getstate()
                numpy_state = np.random.get_state()
                torch_state = torch.get_rng_state()
                cuda_state = torch.cuda.get_rng_state_all()
                random.seed(seed + 20_000 + epoch)
                np.random.seed(seed + 20_000 + epoch)
                torch.manual_seed(seed + 20_000 + epoch)
                torch.cuda.manual_seed_all(seed + 20_000 + epoch)
                last_validation_loss = _validation_loss(
                    model,
                    arm,
                    prepared.validation,
                    batch_size=batch_size,
                    workers=workers,
                    device=device,
                )
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                torch.cuda.set_rng_state_all(cuda_state)
                improved = last_validation_loss < best_validation_loss
                if improved:
                    best_validation_loss = last_validation_loss

            save_now = (
                (epoch + 1) % checkpoint_interval == 0
                or stop_requested
                or epoch + 1 == epochs
            )
            if save_now:
                extra = {
                    "best_validation_loss": best_validation_loss,
                    "last_validation_loss": last_validation_loss,
                    "planned_epochs": epochs,
                    "upstream_seed_defined": False,
                    "source_audit": dict(prepared.source_audit),
                }
                save_training_checkpoint(
                    checkpoint,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    epoch=epoch,
                    step=global_updates,
                    config=contract,
                    normalizer={"kind": "upstream_per_epoch_scale"},
                    extra=extra,
                )
                if improved or not best_checkpoint.is_file():
                    save_training_checkpoint(
                        best_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=None,
                        epoch=epoch,
                        step=global_updates,
                        config=contract,
                        normalizer={"kind": "upstream_per_epoch_scale"},
                        extra=extra,
                    )
            if stop_requested:
                break
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    training_seconds = time.perf_counter() - started
    status = "checkpointed_for_resume" if stop_requested else "completed"
    if status == "checkpointed_for_resume":
        summary = {
            "status": status,
            "benchmark_id": BENCHMARK_ID,
            "protocol": protocol,
            "noise_type": noise_type,
            "arm": arm,
            "stage": stage,
            "epochs_completed": last_epoch + 1,
            "optimizer_updates": global_updates,
            "checkpoint": str(checkpoint),
            "resume_supported": True,
            "resume_command": str(_mapping(config, "execution")["resume_command"]),
            "scientific_result_eligible": False,
        }
        (run_dir / "result_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    if last_epoch + 1 != epochs or not best_checkpoint.is_file():
        raise RuntimeError("EEGDfus full budget did not complete")
    best_payload = load_training_checkpoint(best_checkpoint, map_location=device)
    if best_payload["config"] != contract:
        raise ValueError("best EEGDfus checkpoint contract mismatch")
    if best_payload["normalizer_state"] != {"kind": "upstream_per_epoch_scale"}:
        raise ValueError("best EEGDfus checkpoint preprocessing state mismatch")
    model.load_state_dict(best_payload["model_state"])
    metrics = _evaluate(
        model,
        modules,
        prepared,
        arm=arm,
        diffusion_steps=int(_mapping(model_config, "diffusion")["num_steps"]),
        batch_size=int(training["evaluation_batch_size"]),
        workers=workers,
        device=device,
    )
    _write_metrics(output_dir / "metrics.csv", metrics)
    planned_updates = epochs * (len(prepared.train.clean) // batch_size)
    if global_updates != planned_updates:
        raise RuntimeError(
            "EEGDfus completed without the exact frozen optimizer-update budget"
        )
    summary = {
        "status": "completed_tiny_smoke_only" if stage == "smoke" else "completed",
        "benchmark_id": BENCHMARK_ID,
        "protocol": protocol,
        "noise_type": noise_type,
        "arm": arm,
        "stage": stage,
        "tiny_smoke_only": stage == "smoke",
        "scientific_result_eligible": stage == "full",
        "claim_scope": "single_channel_EOG_EMG_stress_test_only",
        "identity_unit": "source_epoch_not_participant",
        "source_commit": OFFICIAL_COMMIT,
        "upstream_seed_defined": False,
        "adapter_seed": seed,
        "resumed": resumed,
        "epochs_completed": last_epoch + 1,
        "optimizer_updates": global_updates,
        "planned_optimizer_updates": planned_updates,
        "matched_update_budget": global_updates == planned_updates,
        "diffusion_steps": int(_mapping(model_config, "diffusion")["num_steps"]),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "gpu_name": torch.cuda.get_device_name(device),
        "source_audit": dict(prepared.source_audit),
        "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "resume_supported": True,
        "resume_command": str(_mapping(config, "execution")["resume_command"]),
        "metrics": str(output_dir / "metrics.csv"),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "native_known_issues_preserved": (
            [
                "post_mixing_train_validation_source_epoch_overlap",
                "official_RRMSE_s_zero_denominator_shape_400_vs_512",
            ]
            if protocol == "official_native"
            else ["official_RRMSE_s_zero_denominator_shape_400_vs_512"]
        ),
    }
    (output_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _finite_metric(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"EEGDfus metric {key!r} is missing or non-numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"EEGDfus metric {key!r} is non-finite")
    return value


def aggregate_eegdfus_full_cells(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    pairing_acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and aggregate the complete frozen eight-cell benchmark.

    The official-native and strict-source protocols remain separate.  Arm
    differences are paired only within an exact protocol/noise/SNR cell after
    checking that the two arms used identical source manifests, evaluation
    mixture counts, and optimizer-update budgets.
    """

    expected = set(TASK_MATRIX)
    actual = set(cells)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"EEGDfus full aggregate requires all eight cells; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if pairing_acceptance.get("status") != (
        "passed_reconstructed_ordered_pairing_acceptance"
    ):
        raise ValueError("EEGDfus ordered-pair acceptance is missing or failed")
    if pairing_acceptance.get("scientific_threshold_or_method_changed") is not False:
        raise ValueError("EEGDfus pairing acceptance changed a scientific rule")
    if pairing_acceptance.get("submitted_and_resolved_configs_equal") is not True:
        raise ValueError("EEGDfus accepted full cells used unequal configs")
    if pairing_acceptance.get("cell_summaries_bound_to_array_run_directories") is not True:
        raise ValueError("EEGDfus full cells are not bound to one accepted array")
    if pairing_acceptance.get("metric_and_manifest_paths_bound_by_producer_summary") is not True:
        raise ValueError("EEGDfus metrics/manifests are not producer-path bound")
    pairing_rows = pairing_acceptance.get("pairing_rows")
    if not isinstance(pairing_rows, Sequence) or isinstance(pairing_rows, (str, bytes)):
        raise ValueError("EEGDfus pairing reconstruction rows are missing")
    expected_pairing_keys = set(EXPECTED_FULL_EVALUATION_MIXTURES)
    observed_pairing_keys: set[tuple[str, str]] = set()
    for row in pairing_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid EEGDfus pairing reconstruction row")
        pair_key = (str(row.get("protocol")), str(row.get("noise_type")))
        observed_pairing_keys.add(pair_key)
        if row.get("ordered_clean_artifact_snr_pairing_equal") is not True:
            raise ValueError("EEGDfus ordered clean/artifact/SNR pairing differs")
        if int(row.get("evaluation_mixtures_per_snr", -1)) != (
            EXPECTED_FULL_EVALUATION_MIXTURES.get(pair_key)
        ):
            raise ValueError("EEGDfus reconstructed evaluation count is incomplete")
        if int(row.get("snr_levels", -1)) != len(OFFICIAL_TEST_SNR_DB):
            raise ValueError("EEGDfus reconstructed SNR grid is incomplete")
    if observed_pairing_keys != expected_pairing_keys:
        raise ValueError("EEGDfus pairing reconstruction matrix is incomplete")

    metric_names = (
        "snr_improvement_db",
        "correlation",
        "rrmse_temporal",
        "rrmse_spectral_corrected_psd_denominator_shape",
    )
    expected_snr = tuple(float(value) for value in OFFICIAL_TEST_SNR_DB)
    all_metric_rows: list[dict[str, Any]] = []
    cell_summary_rows: list[dict[str, Any]] = []
    indexed_metrics: dict[tuple[str, str, str], dict[float, Mapping[str, Any]]] = {}

    for task_index, key in enumerate(TASK_MATRIX):
        protocol, noise_type, arm = key
        cell = cells[key]
        summary = cell.get("summary")
        metrics = cell.get("metrics")
        manifest = cell.get("split_manifest")
        if not isinstance(summary, Mapping) or not isinstance(metrics, Sequence):
            raise ValueError(f"invalid EEGDfus aggregate payload for {key}")
        if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
            raise ValueError(f"missing EEGDfus source manifest for {key}")
        if (
            summary.get("status") != "completed"
            or summary.get("stage") != "full"
            or summary.get("scientific_result_eligible") is not True
        ):
            raise ValueError(f"EEGDfus cell is not a completed full result: {key}")
        for field, expected_value in (
            ("benchmark_id", BENCHMARK_ID),
            ("protocol", protocol),
            ("noise_type", noise_type),
            ("arm", arm),
            ("identity_unit", "source_epoch_not_participant"),
        ):
            if summary.get(field) != expected_value:
                raise ValueError(f"EEGDfus summary {field} mismatch for {key}")
        updates = int(summary.get("optimizer_updates", -1))
        planned_updates = int(summary.get("planned_optimizer_updates", -2))
        expected_updates = EXPECTED_FULL_OPTIMIZER_UPDATES[(protocol, noise_type)]
        if updates < 1 or updates != planned_updates or not bool(
            summary.get("matched_update_budget")
        ) or updates != expected_updates:
            raise ValueError(f"EEGDfus optimizer budget is incomplete for {key}")
        source_audit = summary.get("source_audit")
        if not isinstance(source_audit, Mapping):
            raise ValueError(f"EEGDfus source audit is missing for {key}")
        if protocol == "strict_source_epoch" and any(
            int(source_audit.get(field, -1)) != 0 for field in STRICT_OVERLAP_FIELDS
        ):
            raise ValueError(f"EEGDfus strict source split leaked for {key}")

        by_snr: dict[float, Mapping[str, Any]] = {}
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in metrics:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"invalid EEGDfus metric row for {key}")
            row = dict(raw_row)
            for field, expected_value in (
                ("benchmark_id", BENCHMARK_ID),
                ("protocol", protocol),
                ("noise_type", noise_type),
                ("arm", arm),
                ("identity_unit", "source_epoch_not_participant"),
            ):
                if row.get(field) != expected_value:
                    raise ValueError(f"EEGDfus metric {field} mismatch for {key}")
            snr_db = _finite_metric(row, "snr_db")
            if snr_db in by_snr:
                raise ValueError(f"duplicate EEGDfus SNR={snr_db} for {key}")
            if str(row.get("rrmse_spectral_official", "")) != "":
                raise ValueError("blocked official spectral RRMSE must remain empty")
            if row.get("rrmse_spectral_official_status") != (
                "blocked_upstream_zero_denominator_shape_400_vs_512"
            ):
                raise ValueError("official spectral RRMSE blockage was not preserved")
            for metric_name in metric_names:
                _finite_metric(row, metric_name)
            evaluation_mixtures = int(row.get("evaluation_mixtures", 0))
            if evaluation_mixtures != EXPECTED_FULL_EVALUATION_MIXTURES[
                (protocol, noise_type)
            ]:
                raise ValueError(f"truncated EEGDfus evaluation cell: {key}")
            expected_calls = (
                OFFICIAL_DIFFUSION_STEPS if arm == "conditional_diffusion" else 1
            )
            if int(row.get("network_calls_per_output", -1)) != expected_calls:
                raise ValueError(f"EEGDfus network-call budget mismatch for {key}")
            normalized = {
                "task_index": task_index,
                **row,
            }
            normalized_rows.append(normalized)
            by_snr[snr_db] = normalized
        if tuple(sorted(by_snr)) != expected_snr:
            raise ValueError(f"EEGDfus full SNR grid mismatch for {key}")
        indexed_metrics[key] = by_snr
        all_metric_rows.extend(normalized_rows)

        def mean_metric(name: str) -> float:
            return float(np.mean([_finite_metric(row, name) for row in normalized_rows]))

        official_statuses = {
            str(row["rrmse_spectral_official_status"]) for row in normalized_rows
        }
        cell_summary_rows.append(
            {
                "task_index": task_index,
                "benchmark_id": BENCHMARK_ID,
                "protocol": protocol,
                "noise_type": noise_type,
                "arm": arm,
                "status": "completed",
                "scientific_result_eligible": True,
                "identity_unit": "source_epoch_not_participant",
                "snr_levels": len(normalized_rows),
                "optimizer_updates": updates,
                "planned_optimizer_updates": planned_updates,
                "training_seconds": _finite_metric(summary, "training_seconds"),
                "peak_gpu_memory_mb": _finite_metric(summary, "peak_gpu_memory_mb"),
                "gpu_name": str(summary.get("gpu_name", "")),
                "mean_snr_improvement_db": mean_metric("snr_improvement_db"),
                "mean_correlation": mean_metric("correlation"),
                "mean_rrmse_temporal": mean_metric("rrmse_temporal"),
                "rrmse_spectral_official": "",
                "rrmse_spectral_official_status": "+".join(sorted(official_statuses)),
                "mean_rrmse_spectral_corrected_psd_denominator_shape": mean_metric(
                    "rrmse_spectral_corrected_psd_denominator_shape"
                ),
                "mean_evaluation_seconds": float(
                    np.mean(
                        [
                            _finite_metric(row, "evaluation_seconds")
                            for row in normalized_rows
                        ]
                    )
                ),
                "network_calls_per_output": int(
                    normalized_rows[0]["network_calls_per_output"]
                ),
                "train_validation_clean_source_overlap": int(
                    source_audit["train_validation_clean_overlap"]
                ),
                "train_validation_artifact_source_overlap": int(
                    source_audit["train_validation_artifact_overlap"]
                ),
            }
        )

    paired_rows: list[dict[str, Any]] = []
    paired_summaries: list[dict[str, Any]] = []
    directions = {
        "snr_improvement_db": "higher",
        "correlation": "higher",
        "rrmse_temporal": "lower",
        "rrmse_spectral_corrected_psd_denominator_shape": "lower",
    }
    for protocol in ("official_native", "strict_source_epoch"):
        for noise_type in ("EOG", "EMG"):
            diffusion_key = (protocol, noise_type, "conditional_diffusion")
            deterministic_key = (protocol, noise_type, "matched_deterministic")
            diffusion_cell = cells[diffusion_key]
            deterministic_cell = cells[deterministic_key]
            if list(diffusion_cell["split_manifest"]) != list(
                deterministic_cell["split_manifest"]
            ):
                raise ValueError(
                    f"EEGDfus paired arms do not share an exact source manifest: "
                    f"{protocol}/{noise_type}"
                )
            diffusion_summary = diffusion_cell["summary"]
            deterministic_summary = deterministic_cell["summary"]
            if diffusion_summary["source_audit"] != deterministic_summary["source_audit"]:
                raise ValueError(
                    f"EEGDfus paired arms have unequal source audits: "
                    f"{protocol}/{noise_type}"
                )
            if int(diffusion_summary["optimizer_updates"]) != int(
                deterministic_summary["optimizer_updates"]
            ):
                raise ValueError(
                    f"EEGDfus paired arms have unequal optimizer updates: "
                    f"{protocol}/{noise_type}"
                )
            same_gpu = str(diffusion_summary.get("gpu_name", "")) == str(
                deterministic_summary.get("gpu_name", "")
            )
            comparison_rows: list[dict[str, Any]] = []
            for snr_db in expected_snr:
                diffusion = indexed_metrics[diffusion_key][snr_db]
                deterministic = indexed_metrics[deterministic_key][snr_db]
                if int(diffusion["evaluation_mixtures"]) != int(
                    deterministic["evaluation_mixtures"]
                ):
                    raise ValueError(
                        f"EEGDfus paired evaluation count differs at "
                        f"{protocol}/{noise_type}/{snr_db}"
                    )
                paired = {
                    "benchmark_id": BENCHMARK_ID,
                    "protocol": protocol,
                    "noise_type": noise_type,
                    "identity_unit": "source_epoch_not_participant",
                    "snr_db": snr_db,
                    "evaluation_mixtures": int(diffusion["evaluation_mixtures"]),
                    "comparison": (
                        "conditional_diffusion_minus_matched_deterministic"
                    ),
                    "paired_source_manifest_equal": True,
                    "paired_source_manifest_scope": "source_membership",
                    "paired_ordered_input_reconstruction_equal": True,
                    "paired_optimizer_updates_equal": True,
                    "rrmse_spectral_official": "",
                    "rrmse_spectral_official_status": (
                        "blocked_upstream_zero_denominator_shape_400_vs_512"
                    ),
                    "conditional_gpu_name": str(diffusion_summary.get("gpu_name", "")),
                    "deterministic_gpu_name": str(
                        deterministic_summary.get("gpu_name", "")
                    ),
                    "latency_comparison_status": (
                        "comparable_same_gpu_model"
                        if same_gpu
                        else "descriptive_only_different_gpu_models"
                    ),
                    "conditional_evaluation_seconds": _finite_metric(
                        diffusion, "evaluation_seconds"
                    ),
                    "deterministic_evaluation_seconds": _finite_metric(
                        deterministic, "evaluation_seconds"
                    ),
                    "evaluation_seconds_delta_if_same_gpu": (
                        _finite_metric(diffusion, "evaluation_seconds")
                        - _finite_metric(deterministic, "evaluation_seconds")
                        if same_gpu
                        else ""
                    ),
                }
                for metric_name in metric_names:
                    diffusion_value = _finite_metric(diffusion, metric_name)
                    deterministic_value = _finite_metric(deterministic, metric_name)
                    paired[f"conditional_{metric_name}"] = diffusion_value
                    paired[f"deterministic_{metric_name}"] = deterministic_value
                    paired[f"delta_{metric_name}"] = (
                        diffusion_value - deterministic_value
                    )
                comparison_rows.append(paired)
                paired_rows.append(paired)

            comparison_summary: dict[str, Any] = {
                "protocol": protocol,
                "noise_type": noise_type,
                "snr_levels": len(comparison_rows),
                "comparison": "conditional_diffusion_minus_matched_deterministic",
                "paired_source_manifest_equal": True,
                "paired_source_manifest_scope": "source_membership",
                "paired_ordered_input_reconstruction_equal": True,
                "paired_optimizer_updates_equal": True,
                "latency_comparison_status": comparison_rows[0][
                    "latency_comparison_status"
                ],
            }
            for metric_name, direction in directions.items():
                deltas = np.asarray(
                    [float(row[f"delta_{metric_name}"]) for row in comparison_rows],
                    dtype=np.float64,
                )
                wins = deltas > 0.0 if direction == "higher" else deltas < 0.0
                comparison_summary[f"mean_delta_{metric_name}"] = float(
                    np.mean(deltas)
                )
                comparison_summary[f"conditional_win_count_{metric_name}"] = int(
                    np.sum(wins)
                )
                comparison_summary[f"metric_direction_{metric_name}"] = direction
            paired_summaries.append(comparison_summary)

    return {
        "status": "completed_full_aggregate",
        "benchmark_id": BENCHMARK_ID,
        "scientific_result_eligible": True,
        "claim_scope": "single_channel_EOG_EMG_stress_test_only",
        "identity_unit": "source_epoch_not_participant",
        "matrix_cells_expected": len(TASK_MATRIX),
        "matrix_cells_completed": len(cell_summary_rows),
        "metric_rows_expected": len(TASK_MATRIX) * len(OFFICIAL_TEST_SNR_DB),
        "metric_rows_completed": len(all_metric_rows),
        "paired_rows_expected": 4 * len(OFFICIAL_TEST_SNR_DB),
        "paired_rows_completed": len(paired_rows),
        "input_pairing_acceptance": dict(pairing_acceptance),
        "protocols_kept_separate": ["official_native", "strict_source_epoch"],
        "protocol_limitations": {
            "official_native": (
                "upstream post-mixing train/validation source overlap preserved "
                "and disclosed"
            ),
            "strict_source_epoch": (
                "disjoint source epochs; EEGdenoiseNet has no participant identity"
            ),
        },
        "comparison_scope": (
            "paired_descriptive_across_frozen_snr_levels_not_independent_inference"
        ),
        "official_spectral_metric": {
            "value": None,
            "status": "blocked_upstream_zero_denominator_shape_400_vs_512",
            "corrected_field": (
                "rrmse_spectral_corrected_psd_denominator_shape"
            ),
        },
        "cell_summary_rows": cell_summary_rows,
        "all_metric_rows": all_metric_rows,
        "paired_rows": paired_rows,
        "paired_summaries": paired_summaries,
    }


def _eegdfus_aggregate_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# EEGDfus full benchmark aggregate",
        "",
        "All eight frozen cells completed. Official-native and strict source-epoch "
        "results are reported separately. Comparisons are paired descriptions over "
        "the frozen SNR grid, not independent statistical replicates.",
        "",
        "The upstream spectral RRMSE remains blocked by the 400-vs-512 denominator "
        "shape mismatch. The explicitly named corrected PSD-denominator-shape metric "
        "is reported alongside the empty official field.",
        "",
        "## Cell means",
        "",
        "| Protocol | Noise | Arm | SNR improvement | Correlation | Temporal "
        "RRMSE | Corrected spectral RRMSE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["cell_summary_rows"]:
        lines.append(
            "| {protocol} | {noise_type} | {arm} | {snr:.6g} | {corr:.6g} | "
            "{rrmse:.6g} | {spectral:.6g} |".format(
                protocol=row["protocol"],
                noise_type=row["noise_type"],
                arm=row["arm"],
                snr=float(row["mean_snr_improvement_db"]),
                corr=float(row["mean_correlation"]),
                rrmse=float(row["mean_rrmse_temporal"]),
                spectral=float(
                    row["mean_rrmse_spectral_corrected_psd_denominator_shape"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Paired conditional-minus-deterministic descriptions",
            "",
            "| Protocol | Noise | ΔSNR improvement | Δcorrelation | Δtemporal "
            "RRMSE | Δcorrected spectral RRMSE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["paired_summaries"]:
        lines.append(
            "| {protocol} | {noise_type} | {snr:.6g} | {corr:.6g} | "
            "{rrmse:.6g} | {spectral:.6g} |".format(
                protocol=row["protocol"],
                noise_type=row["noise_type"],
                snr=float(row["mean_delta_snr_improvement_db"]),
                corr=float(row["mean_delta_correlation"]),
                rrmse=float(row["mean_delta_rrmse_temporal"]),
                spectral=float(
                    row[
                        "mean_delta_rrmse_spectral_corrected_psd_denominator_shape"
                    ]
                ),
            )
        )
    lines.extend(
        [
            "",
            "EEGdenoiseNet exposes source epochs rather than participant identities; "
            "these results cannot support participant-specific or real-EEG deployment claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _training_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the post-submit structural-acceptance annotation."""

    value = json.loads(json.dumps(dict(config)))
    execution = dict(_mapping(value, "execution"))
    execution.pop("accepted_full_array_job_id", None)
    execution.pop("accepted_full_array_git_head", None)
    execution.pop("accepted_full_array_task_job_ids", None)
    execution.pop("acceptance_amendment", None)
    stages = list(execution.get("stages", ()))
    if stages and stages[-1] == "aggregate-full":
        stages.pop()
    execution["stages"] = stages
    execution.pop("aggregate_command", None)
    value["execution"] = execution
    value.pop("resolved_task", None)
    return value


def _full_array_pairing_acceptance(
    config: Mapping[str, Any], *, result_root: Path
) -> dict[str, Any]:
    """Reconstruct frozen input pairing and bind cells to one Slurm array.

    The full cells predate the additive ordered-pair audit fields.  Their
    sbatch run directories nevertheless retain the exact submitted config,
    task index, Git HEAD, and terminal summary.  Reconstructing each arm from
    that frozen config and seed checks clean/artifact order and SNR values
    without reading a performance metric or changing a decision threshold.
    """

    execution = _mapping(config, "execution")
    full_job_id = execution.get("accepted_full_array_job_id")
    if isinstance(full_job_id, bool) or not isinstance(full_job_id, int):
        raise ValueError("full aggregate requires an explicit accepted array job ID")
    accepted_git_head = execution.get("accepted_full_array_git_head")
    if accepted_git_head != "fd20ff2d6e69db4c05f888893787994b336cd1c3":
        raise ValueError("full aggregate accepted producer Git HEAD changed")
    task_job_ids = execution.get("accepted_full_array_task_job_ids")
    if not isinstance(task_job_ids, Mapping) or set(task_job_ids) != set(
        range(len(TASK_MATRIX))
    ):
        raise ValueError("full aggregate requires explicit accepted task-job IDs")
    expected_config = _training_config_view(config)
    code_root = Path("/home/infres/yinwang/denoiseNet")
    task_configs: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    git_heads: set[str] = set()

    for task_index, key in enumerate(TASK_MATRIX):
        protocol, noise_type, arm = key
        task_job_id = int(task_job_ids[task_index])
        run_task = (
            code_root
            / "runs"
            / f"cgdr_eegdfus-benchmark_full_{task_job_id}_{task_index}"
        )
        required = {
            "config": run_task / "config.yaml",
            "git_head": run_task / "git_head.txt",
            "job_id": run_task / "slurm_job_id.txt",
            "task_id": run_task / "slurm_array_task_id.txt",
            "summary": run_task / "result_summary.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"accepted EEGDfus array task {task_index} is incomplete: {missing}"
            )
        task_id = int(required["task_id"].read_text(encoding="utf-8").strip())
        if task_id != task_index:
            raise ValueError("EEGDfus accepted array task index mismatch")
        recorded_job_id = int(
            required["job_id"].read_text(encoding="utf-8").strip()
        )
        if recorded_job_id != task_job_id:
            raise ValueError("EEGDfus accepted array task job ID mismatch")
        submitted_config = yaml.safe_load(
            required["config"].read_text(encoding="utf-8")
        )
        if not isinstance(submitted_config, Mapping) or _training_config_view(
            submitted_config
        ) != expected_config:
            raise ValueError("EEGDfus accepted array tasks used unequal configs")
        git_head = required["git_head"].read_text(encoding="utf-8").strip()
        if not git_head:
            raise ValueError("EEGDfus accepted array task has an empty Git HEAD")
        git_heads.add(git_head)

        cell_dir = result_root / "full" / protocol / noise_type.lower() / arm
        canonical_summary = json.loads(
            (cell_dir / "result_summary.json").read_text(encoding="utf-8")
        )
        run_summary = json.loads(required["summary"].read_text(encoding="utf-8"))
        if run_summary != canonical_summary:
            raise ValueError("EEGDfus canonical cell does not match accepted array output")
        if canonical_summary.get("metrics") != str(cell_dir / "metrics.csv") or (
            canonical_summary.get("split_manifest")
            != str(cell_dir / "split_manifest.csv")
        ):
            raise ValueError("EEGDfus producer summary does not bind canonical artifacts")
        resolved = yaml.safe_load(
            (cell_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(resolved, Mapping):
            raise ValueError("EEGDfus cell resolved config is not a mapping")
        resolved_task = _mapping(resolved, "resolved_task")
        for field, expected in (
            ("task_index", task_index),
            ("stage", "full"),
            ("protocol", protocol),
            ("noise_type", noise_type),
            ("arm", arm),
        ):
            if resolved_task.get(field) != expected:
                raise ValueError(f"EEGDfus resolved task {field} mismatch")
        if _training_config_view(resolved) != expected_config:
            raise ValueError("EEGDfus resolved cell config differs from submitted config")
        task_configs[key] = submitted_config

    if len(git_heads) != 1:
        raise ValueError("EEGDfus accepted array tasks used mixed Git revisions")
    if next(iter(git_heads)) != accepted_git_head:
        raise ValueError("EEGDfus array producer Git HEAD differs from acceptance")

    pairing_rows: list[dict[str, Any]] = []
    for protocol in ("official_native", "strict_source_epoch"):
        for noise_type in ("EOG", "EMG"):
            conditional = _prepare_from_config(
                task_configs[(protocol, noise_type, "conditional_diffusion")],
                protocol=protocol,
                noise_type=noise_type,
                stage="full",
            )
            deterministic = _prepare_from_config(
                task_configs[(protocol, noise_type, "matched_deterministic")],
                protocol=protocol,
                noise_type=noise_type,
                stage="full",
            )
            if not _prepared_pairing_equal(conditional, deterministic):
                raise ValueError(
                    f"EEGDfus reconstructed ordered pairing differs: "
                    f"{protocol}/{noise_type}"
                )
            evaluation_mixtures = len(conditional.evaluation[0].pairs.clean)
            if evaluation_mixtures != EXPECTED_FULL_EVALUATION_MIXTURES[
                (protocol, noise_type)
            ]:
                raise ValueError("EEGDfus reconstructed evaluation count is truncated")
            pairing_rows.append(
                {
                    "protocol": protocol,
                    "noise_type": noise_type,
                    "train_pairs": len(conditional.train.clean),
                    "validation_pairs": len(conditional.validation.clean),
                    "evaluation_mixtures_per_snr": evaluation_mixtures,
                    "snr_levels": len(conditional.evaluation),
                    "ordered_clean_artifact_snr_pairing_equal": True,
                }
            )

    return {
        "status": "passed_reconstructed_ordered_pairing_acceptance",
        "full_array_job_id": full_job_id,
        "task_job_ids": {
            str(task_index): int(task_job_ids[task_index])
            for task_index in range(len(TASK_MATRIX))
        },
        "git_head": next(iter(git_heads)),
        "task_indices": list(range(len(TASK_MATRIX))),
        "submitted_and_resolved_configs_equal": True,
        "cell_summaries_bound_to_array_run_directories": True,
        "metric_and_manifest_paths_bound_by_producer_summary": True,
        "artifact_binding_scope": (
            "exact canonical paths in accepted task summary; no content hashes under "
            "HARNESS_LEVEL=1"
        ),
        "cell_level_ordered_pair_manifest_was_persisted": False,
        "pairing_reconstruction_timing": "post_submit_before_performance_aggregation",
        "scientific_threshold_or_method_changed": False,
        "pairing_rows": pairing_rows,
    }


def run_eegdfus_full_aggregate(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Load all full matrix artifacts and write one small CPU aggregate."""

    validate_eegdfus_config(config)
    result_root = Path(str(_mapping(config, "outputs")["result_root"])).resolve()
    code_root = Path("/home/infres/yinwang/denoiseNet")
    if code_root not in result_root.parents:
        raise ValueError("EEGDfus result root must remain under the code root")
    pairing_acceptance = _full_array_pairing_acceptance(
        config, result_root=result_root
    )
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for protocol, noise_type, arm in TASK_MATRIX:
        cell_dir = result_root / "full" / protocol / noise_type.lower() / arm
        summary_path = cell_dir / "result_summary.json"
        metrics_path = cell_dir / "metrics.csv"
        manifest_path = cell_dir / "split_manifest.csv"
        if not summary_path.is_file() or not metrics_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"incomplete EEGDfus full cell: {cell_dir}")
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            metric_rows = list(csv.DictReader(stream))
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            manifest_rows = list(csv.DictReader(stream))
        cells[(protocol, noise_type, arm)] = {
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "metrics": metric_rows,
            "split_manifest": manifest_rows,
        }

    result = aggregate_eegdfus_full_cells(
        cells, pairing_acceptance=pairing_acceptance
    )
    output_dir = result_root / "full_aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir / "all_cell_metrics.csv", result["all_metric_rows"])
    _write_metrics(output_dir / "cell_summary.csv", result["cell_summary_rows"])
    _write_metrics(output_dir / "paired_arm_comparison.csv", result["paired_rows"])
    public_result = {
        key: value
        for key, value in result.items()
        if key not in {"all_metric_rows", "paired_rows"}
    }
    public_result["outputs"] = {
        "all_cell_metrics": str(output_dir / "all_cell_metrics.csv"),
        "cell_summary": str(output_dir / "cell_summary.csv"),
        "paired_arm_comparison": str(output_dir / "paired_arm_comparison.csv"),
        "result_summary": str(output_dir / "result_summary.json"),
        "report": str(output_dir / "result_summary.md"),
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    (output_dir / "result_summary.json").write_text(
        json.dumps(public_result, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "result_summary.md").write_text(
        _eegdfus_aggregate_markdown(result), encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(public_result, indent=2) + "\n", encoding="utf-8"
    )
    return public_result


__all__ = [
    "BENCHMARK_ID",
    "EXPECTED_FULL_EVALUATION_MIXTURES",
    "EXPECTED_FULL_OPTIMIZER_UPDATES",
    "OFFICIAL_BATCH_SIZE",
    "OFFICIAL_COMBINATIONS",
    "OFFICIAL_COMMIT",
    "OFFICIAL_DIFFUSION_STEPS",
    "OFFICIAL_EPOCHS",
    "OFFICIAL_TEST_SNR_DB",
    "TASK_MATRIX",
    "EpochPairs",
    "EvaluationLevel",
    "MatchedConditionOnly",
    "PreparedProtocol",
    "audit_ssed_source_text",
    "eegdfus_rrmse_s_corrected_denominator_shape",
    "load_official_modules",
    "prepare_official_native",
    "prepare_strict_source_epoch",
    "aggregate_eegdfus_full_cells",
    "run_eegdfus_cpu_validation",
    "run_eegdfus_full_aggregate",
    "run_eegdfus_stage",
    "source_split_manifest_rows",
    "validate_official_checkout",
    "validate_eegdfus_config",
]
