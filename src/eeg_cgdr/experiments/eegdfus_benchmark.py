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

    def __post_init__(self) -> None:
        count = int(self.clean.shape[0])
        if self.clean.ndim != 2 or self.noisy.shape != self.clean.shape:
            raise ValueError("clean and noisy epochs must share shape (N,L)")
        if self.clean_source_epoch.shape != (count,) or self.artifact_source_epoch.shape != (
            count,
        ):
            raise ValueError("source-epoch identifiers must have one entry per pair")
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
    ) != ("cpu-tests", "smoke", "full"):
        raise ValueError("EEGDfus execution stages differ from the frozen route")


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
    )


def _select_pairs(pairs: EpochPairs, index: np.ndarray) -> EpochPairs:
    return EpochPairs(
        clean=pairs.clean[index],
        noisy=pairs.noisy[index],
        clean_source_epoch=pairs.clean_source_epoch[index],
        artifact_source_epoch=pairs.artifact_source_epoch[index],
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


__all__ = [
    "BENCHMARK_ID",
    "OFFICIAL_BATCH_SIZE",
    "OFFICIAL_COMBINATIONS",
    "OFFICIAL_COMMIT",
    "OFFICIAL_DIFFUSION_STEPS",
    "OFFICIAL_EPOCHS",
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
    "run_eegdfus_cpu_validation",
    "run_eegdfus_stage",
    "source_split_manifest_rows",
    "validate_official_checkout",
    "validate_eegdfus_config",
]
