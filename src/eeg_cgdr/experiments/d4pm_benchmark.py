"""Source-bounded D4PM benchmark adapter, scoped to the EEGdenoiseNet EOG route.

The upstream repository (Shao et al., arXiv:2509.14302) is loaded dynamically
from a frozen, Git-ignored checkout.  Its model/DDPM implementations are never
vendored here.  One protocol is executed:

``eog_scoped_seeded_native``
    The upstream EOG route exactly: one independent permutation of each source
    library, the clean library truncated to the artifact count, a contiguous
    80/10/10 row split, eleven within-split mixture permutations, and the
    upstream mixture recipe including its ``get_rms`` mean-square quirk and
    square-root SNR amplitude factor.  The wrapper fixes a seed because upstream
    defines none, and it records source-epoch statistics instead of repairing
    them.

The EMG and ECG routes are deliberately absent.  ``ECG_all_epochs.npy`` is not
in the registered EEGdenoiseNet release and its derivation is undocumented in
the D4PM release, so the full three-artifact native protocol is not
reproducible; see ``reports/d4pm_feasibility_audit.md``.

Five released defects are repaired, each disclosed in the frozen config's
``known_upstream_issues``/``deviations`` blocks:

1.  ``train_d4pm_artifacts.py`` raises ``NameError``.  The artifact branch is
    trained with the four-head ``DualBranchDenoisingModel_noise`` because
    ``test_joint.py`` constructs that architecture to load the artifact
    checkpoint.  The choice is labelled ``minimally_repaired_source-faithful``.
2.  The prepared-data ``.data/`` vs ``./data`` path mismatch is bypassed: the
    adapter prepares arrays in memory and never uses the upstream writer.
3.  Upstream defines no seed; ``adapter_seed`` is frozen in the config.
4.  Upstream checkpoints are model-only; these carry both branch weights,
    optimizer, scheduler, epoch, step, and RNG state so a walltime kill wastes
    nothing.
5.  The released evaluation covers only the first 50 EOG rows.  All frozen test
    rows are evaluated at all eleven SNR levels; the upstream 550-output subset
    is retained as a separate labelled diagnostic.

Spectral metric: the upstream definition was checked for the EEGDfus-style
400-vs-512 denominator defect and it is ABSENT -- ``test_joint.py`` compares
512-bin FFT magnitudes of clean and denoised signals, so the shapes agree.  The
metric is nevertheless reported under the explicit name
``rrmse_spectral_fft_magnitude`` because it is an FFT-magnitude ratio and not
the Welch-PSD ``RRMSE_s`` of the EEGdenoiseNet literature.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
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
from sklearn.preprocessing import scale
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from eeg_cgdr.training import (
    load_training_checkpoint,
    resume_training_checkpoint,
    save_training_checkpoint,
)


BENCHMARK_ID = "d4pm_eegdenoisenet_eog_scoped_v1"
SOURCE_COMMIT = "5be2b3c72973fea6c879e63cd83067ff66aace13"
SOURCE_ROOT = Path("/home/infres/yinwang/denoiseNet/.external/D4PM")
CODE_ROOT = Path("/home/infres/yinwang/denoiseNet")
OFFICIAL_EPOCHS = 4000
OFFICIAL_BATCH_SIZE = 1024
OFFICIAL_FEATURES = 128
OFFICIAL_DIFFUSION_STEPS = 500
OFFICIAL_COMBINATIONS = 11
OFFICIAL_TRAIN_FRACTION = 0.8
OFFICIAL_LEARNING_RATE = 1.0e-3
OFFICIAL_TEST_SNR_DB = tuple(float(value) for value in np.linspace(-5.0, 5.0, 11))
JOINT_LAMBDA_DC = 0.5
JOINT_GAMMA = 1.0
JOINT_ETA = 0.3
EOG_CLASS_LABEL = 0
UPSTREAM_EXAMPLE_ROWS = 50
PROTOCOL = "eog_scoped_seeded_native"
NOISE_TYPE = "EOG"
EXPECTED_FULL_TRAIN_PAIRS = 29_920
EXPECTED_FULL_VALIDATION_PAIRS = 3_740
EXPECTED_FULL_EVALUATION_MIXTURES = 3_740
EXPECTED_FULL_UPDATES_PER_BRANCH = 116_000
ARM_BRANCH_COUNT = {"joint_dual_diffusion": 2, "matched_deterministic": 1}
SPECTRAL_METRIC_STATUS = (
    "upstream_fft_magnitude_definition_shape_verified_not_welch_psd_rrmse_s"
)
TASK_MATRIX = (
    (PROTOCOL, NOISE_TYPE, "joint_dual_diffusion"),
    (PROTOCOL, NOISE_TYPE, "matched_deterministic"),
)
REQUIRED_SOURCE_FILES = (
    "DDPM_joint.py",
    "denoising_model_eegdnet_class.py",
    "denoising_model_eegdnet_class_noise.py",
    "test_joint.py",
    "train_d4pm.py",
    "train_d4pm_artifacts.py",
    "utils.py",
    "Data_Preparation/data_for_eegdnet.py",
)
LICENSE_CANDIDATES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "NOTICE")


@dataclass(frozen=True)
class EpochPairs:
    """Paired clean/artifact/noisy rows plus their real source-epoch identities."""

    clean: np.ndarray
    artifact: np.ndarray
    noisy: np.ndarray
    clean_source_epoch: np.ndarray
    artifact_source_epoch: np.ndarray
    snr_db: np.ndarray

    def __post_init__(self) -> None:
        count = int(self.clean.shape[0])
        if self.clean.ndim != 2 or self.noisy.shape != self.clean.shape:
            raise ValueError("clean and noisy epochs must share shape (N,L)")
        if self.artifact.shape != self.clean.shape:
            raise ValueError("artifact component must share the clean epoch shape")
        if self.clean_source_epoch.shape != (count,) or self.artifact_source_epoch.shape != (
            count,
        ):
            raise ValueError("source-epoch identifiers must have one entry per pair")
        if self.snr_db.shape != (count,) or not np.isfinite(self.snr_db).all():
            raise ValueError("SNR values must have one finite entry per pair")
        if (
            not np.isfinite(self.clean).all()
            or not np.isfinite(self.noisy).all()
            or not np.isfinite(self.artifact).all()
        ):
            raise ValueError("prepared D4PM pairs contain NaN/Inf")


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
class D4PMModules:
    ddpm_class: type[nn.Module]
    clean_backbone_class: type[nn.Module]
    artifact_backbone_class: type[nn.Module]


class MatchedConditionOnly(nn.Module):
    """Deterministic arm using the exact upstream clean-branch backbone.

    The noisy deployment condition is supplied to both upstream streams, a fixed
    conditioning scalar replaces diffusion time, and the same artifact class
    label is visible.  No latent noise or iterative sampling is available.
    """

    visible_inputs = ("noisy_single_channel_epoch", "artifact_class_label")

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.model = backbone

    def forward(self, noisy: Tensor, label: Tensor) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise ValueError("D4PM input must have shape (B,1,L)")
        fixed_condition = torch.ones(
            (noisy.shape[0], 1), dtype=noisy.dtype, device=noisy.device
        )
        return self.model(noisy, noisy, fixed_condition, label.view(-1, 1))


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def validate_d4pm_config(config: Mapping[str, Any]) -> None:
    """Reject semantic drift before data or the external model is loaded."""

    if config.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError(f"benchmark_id must be {BENCHMARK_ID}")
    if int(config.get("harness_level", -1)) != 1:
        raise ValueError("D4PM benchmark requires HARNESS_LEVEL=1")
    if config.get("claim_scope") != "single_channel_EOG_stress_test_only":
        raise ValueError("D4PM cannot support participant-specific claims")

    source = _mapping(config, "source")
    if source.get("commit") != SOURCE_COMMIT:
        raise ValueError("D4PM external commit differs from the frozen source")
    if source.get("load_policy") != "dynamic_external_checkout_no_vendoring":
        raise ValueError("D4PM source must be dynamically loaded, not copied")
    if source.get("upstream_license_file") != "absent_at_frozen_commit":
        raise ValueError("missing upstream license boundary must remain explicit")

    data = _mapping(config, "data")
    if data.get("identity_unit") != "source_epoch_not_participant":
        raise ValueError("EEGdenoiseNet source epochs must not be called participants")
    if int(data.get("epoch_samples", 0)) != 512:
        raise ValueError("the D4PM architecture requires 512-sample epochs")
    if data.get("ecg") != "absent_from_registered_release":
        raise ValueError("ECG must remain declared absent from the registered release")
    expected_shapes = _mapping(data, "expected_shapes")
    if tuple(expected_shapes.get("clean_eeg", ())) != (4514, 512):
        raise ValueError("registered clean EEG shape differs from EEGdenoiseNet")
    if tuple(expected_shapes.get("eog", ())) != (3400, 512):
        raise ValueError("registered EOG shape differs from EEGdenoiseNet")
    if tuple(expected_shapes.get("emg", ())) != (5598, 512):
        raise ValueError("registered EMG shape differs from EEGdenoiseNet")

    scope = _mapping(config, "scope")
    if tuple(scope.get("artifact_routes_executed", ())) != (NOISE_TYPE,):
        raise ValueError("this benchmark is frozen to the EOG route only")
    if sorted(str(value) for value in scope.get("artifact_routes_not_executed", ())) != [
        "ECG",
        "EMG",
    ]:
        raise ValueError("the unexecuted EMG/ECG routes must stay declared")
    if int(scope.get("class_label_used", -1)) != EOG_CLASS_LABEL:
        raise ValueError("the EOG route must use the upstream class label 0")

    protocols = _mapping(config, "protocols")
    protocol = _mapping(protocols, PROTOCOL)
    training = _mapping(protocol, "training")
    diffusion = _mapping(protocol, "diffusion")
    split = _mapping(protocol, "split")
    mixture = _mapping(protocol, "mixture")
    sampler = _mapping(protocol, "joint_sampler")

    if int(training.get("epochs", 0)) != OFFICIAL_EPOCHS:
        raise ValueError("the scoped protocol must preserve 4000 epochs")
    if int(training.get("batch_size", 0)) != OFFICIAL_BATCH_SIZE:
        raise ValueError("the scoped protocol must preserve batch_size=1024")
    if int(training.get("features", 0)) != OFFICIAL_FEATURES:
        raise ValueError("the scoped protocol must preserve features=128")
    if int(training.get("combinations", 0)) != OFFICIAL_COMBINATIONS:
        raise ValueError("the scoped protocol must preserve eleven mixture combinations")
    if float(training.get("learning_rate", 0.0)) != OFFICIAL_LEARNING_RATE:
        raise ValueError("the scoped protocol must preserve Adam learning_rate=1e-3")
    if int(training.get("scheduler_step_epochs", 0)) != 1500 or float(
        training.get("scheduler_gamma", 0.0)
    ) != 0.1:
        raise ValueError("the scoped protocol must preserve the upstream StepLR")
    if float(training.get("gradient_clip", 0.0)) != 1.0:
        raise ValueError("the scoped protocol must preserve gradient clipping at 1.0")
    if int(training.get("validation_interval_epochs", 0)) != 10:
        raise ValueError("the scoped protocol must preserve validation every 10 epochs")
    if training.get("drop_last") is not True:
        raise ValueError("the scoped protocol must preserve drop_last training batches")
    if training.get("early_stopping") is not False:
        raise ValueError("full matched arms must not stop before 4000 epochs")
    if training.get("mixed_precision") is not False:
        raise ValueError("the source-faithful benchmark uses upstream FP32 training")
    accumulation = int(training.get("gradient_accumulation_steps", 0))
    if accumulation < 1 or OFFICIAL_BATCH_SIZE % accumulation != 0:
        raise ValueError("gradient accumulation must divide the frozen batch exactly")
    if int(diffusion.get("num_steps", 0)) != OFFICIAL_DIFFUSION_STEPS:
        raise ValueError("the scoped protocol must preserve 500 diffusion steps")
    if (
        diffusion.get("schedule") != "linear"
        or float(diffusion.get("beta_start", 0.0)) != 1.0e-4
        or float(diffusion.get("beta_end", 0.0)) != 0.02
    ):
        raise ValueError("the scoped protocol must preserve the upstream linear schedule")

    if float(split.get("train_fraction", 0.0)) != OFFICIAL_TRAIN_FRACTION:
        raise ValueError("the scoped native train fraction must remain 0.8")
    if split.get("clean_library_truncated_to_artifact_count") is not True:
        raise ValueError("the EOG route truncates the clean library, not the artifact one")
    if int(split.get("source_epoch_repetitions_within_split", 0)) != OFFICIAL_COMBINATIONS:
        raise ValueError("within-split source repetition must stay disclosed as 11")
    if split.get("upstream_seed_defined") is not False:
        raise ValueError("upstream D4PM defines no random seed")
    adapter_seed = int(_mapping(config, "randomness").get("adapter_seed", -1))
    if int(split.get("adapter_seed", -2)) != adapter_seed:
        raise ValueError("split seed differs from the registered adapter seed")
    if mixture.get("snr_helper") != "upstream_get_rms_returns_mean_square_not_rms":
        raise ValueError("the upstream mean-square SNR quirk must stay disclosed")
    if mixture.get("recipe_policy") != (
        "preserve_exactly_and_disclose_do_not_physically_correct"
    ):
        raise ValueError("the upstream mixture recipe must not be physically corrected")

    if float(sampler.get("lambda_dc", 0.0)) != JOINT_LAMBDA_DC:
        raise ValueError("the joint sampler must preserve lambda_dc=0.5")
    if float(sampler.get("gamma", 0.0)) != JOINT_GAMMA:
        raise ValueError("the joint sampler must preserve gamma=1")
    if float(sampler.get("eta", 0.0)) != JOINT_ETA:
        raise ValueError("the joint sampler must preserve eta=0.3")

    arms = _mapping(config, "arms")
    joint = _mapping(arms, "joint_dual_diffusion")
    matched = _mapping(arms, "matched_deterministic")
    if tuple(joint.get("branches", ())) != ("clean_eeg", "artifact"):
        raise ValueError("the joint arm must train both the clean and artifact branches")
    if joint.get("artifact_branch_architecture") != (
        "DualBranchDenoisingModel_noise_four_attention_heads"
    ):
        raise ValueError("the artifact branch must use the four-head noise class")
    if joint.get("artifact_branch_repair") != "minimally_repaired_source-faithful":
        raise ValueError("the artifact-architecture repair must stay labelled")
    if matched.get("optimizer_updates") != (
        "exactly_equal_to_one_paired_diffusion_branch"
    ):
        raise ValueError("the deterministic arm cannot receive fewer optimizer updates")
    if tuple(matched.get("visible_inputs", ())) != MatchedConditionOnly.visible_inputs:
        raise ValueError("matched arm input contract must equal the deployment condition")

    budget = _mapping(config, "budget")
    for field, expected in (
        ("train_pairs", EXPECTED_FULL_TRAIN_PAIRS),
        ("validation_pairs", EXPECTED_FULL_VALIDATION_PAIRS),
        ("evaluation_mixtures_per_snr", EXPECTED_FULL_EVALUATION_MIXTURES),
        ("optimizer_updates_per_branch", EXPECTED_FULL_UPDATES_PER_BRANCH),
        ("joint_dual_diffusion_total_updates", 2 * EXPECTED_FULL_UPDATES_PER_BRANCH),
        ("matched_deterministic_total_updates", EXPECTED_FULL_UPDATES_PER_BRANCH),
    ):
        if int(budget.get(field, -1)) != expected:
            raise ValueError(f"frozen D4PM budget field {field} differs from the plan")

    matrix = tuple(tuple(str(item) for item in row) for row in config.get("task_matrix", ()))
    if matrix != TASK_MATRIX:
        raise ValueError("D4PM task matrix must contain the frozen two arms")

    issues = _mapping(config, "known_upstream_issues")
    for field in (
        "artifact_trainer_nameerror",
        "artifact_architecture_mismatch",
        "prepared_data_path_mismatch",
        "no_seed",
        "model_only_checkpoints",
        "released_evaluation_is_a_50_row_example",
        "ecg_not_reproducible",
        "get_rms_returns_mean_square",
        "spectral_rrmse_shape_check",
    ):
        if not str(issues.get(field, "")).strip():
            raise ValueError(f"known upstream issue {field} must stay disclosed")

    execution = _mapping(config, "execution")
    if execution.get("mode") != "d4pm-benchmark" or tuple(
        execution.get("stages", ())
    ) != ("cpu-tests", "smoke", "full", "aggregate-full"):
        raise ValueError("D4PM execution stages differ from the frozen route")
    if execution.get("gpu_walltime") != "23:59:59":
        raise ValueError("D4PM GPU routes must request 23:59:59")
    if tuple(execution.get("gpu_excluded_nodes", ())) != ("node54",):
        raise ValueError("D4PM GPU routes must exclude node54")
    if execution.get("array") != "0-1%2":
        raise ValueError("D4PM full/smoke arrays must be 0-1%2")


def _validate_raw_arrays(clean: np.ndarray, artifact: np.ndarray) -> None:
    if clean.ndim != 2 or artifact.ndim != 2:
        raise ValueError("EEGdenoiseNet components must have shape (epochs,samples)")
    if clean.shape[1] != 512 or artifact.shape[1] != 512:
        raise ValueError("EEGdenoiseNet epochs must contain exactly 512 samples")
    if len(clean) < len(artifact):
        raise ValueError("the EOG route needs at least as many clean EEG epochs")
    if min(clean.shape[0], artifact.shape[0]) < 10:
        raise ValueError("too few source epochs for a D4PM split")
    if not np.isfinite(clean).all() or not np.isfinite(artifact).all():
        raise ValueError("EEGdenoiseNet source arrays contain NaN/Inf")


def _repeat_random_permutations(
    values: np.ndarray,
    source_ids: np.ndarray,
    *,
    combinations: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate upstream ``random_signal``: ``combin_num`` stacked permutations."""

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the upstream mixture recipe verbatim, mean-square quirk included."""

    if clean.shape != artifact.shape or snr_db.shape != (clean.shape[0],):
        raise ValueError("mixing arrays have incompatible shapes")
    # sklearn.preprocessing.scale is the exact primitive used upstream.
    clean_scaled = np.asarray(scale(clean, axis=1), dtype=np.float64)
    artifact_scaled = np.asarray(scale(artifact, axis=1), dtype=np.float64)
    # Preserve upstream get_rms semantics: it returns mean square, not RMS.
    clean_power = np.mean(np.square(clean_scaled), axis=1)
    artifact_power = np.mean(np.square(artifact_scaled), axis=1)
    if bool((artifact_power <= 0.0).any()) or bool((clean_power <= 0.0).any()):
        raise ValueError("zero-power source epoch cannot be mixed")
    snr_amplitude = np.sqrt(np.power(10.0, 0.1 * snr_db))
    coefficient = clean_power / (artifact_power * snr_amplitude)
    artifact_component = artifact_scaled * coefficient[:, None]
    noisy = clean_scaled + artifact_component
    return (
        clean_scaled.astype(np.float32),
        artifact_component.astype(np.float32),
        noisy.astype(np.float32),
    )


def _pairs_from_components(
    clean: np.ndarray,
    artifact: np.ndarray,
    clean_ids: np.ndarray,
    artifact_ids: np.ndarray,
    snr_db: np.ndarray,
) -> EpochPairs:
    clean_scaled, artifact_component, noisy = _standardize_and_mix(
        clean, artifact, snr_db
    )
    return EpochPairs(
        clean=clean_scaled,
        artifact=artifact_component,
        noisy=noisy,
        clean_source_epoch=np.asarray(clean_ids, dtype=np.int64),
        artifact_source_epoch=np.asarray(artifact_ids, dtype=np.int64),
        snr_db=np.asarray(snr_db, dtype=np.float64),
    )


def _select_pairs(pairs: EpochPairs, index: np.ndarray) -> EpochPairs:
    return EpochPairs(
        clean=pairs.clean[index],
        artifact=pairs.artifact[index],
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
        "train_validation_artifact_overlap": len(train_artifact & validation_artifact),
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


def prepare_eog_scoped(
    clean: np.ndarray,
    artifact: np.ndarray,
    *,
    seed: int,
    combinations: int = OFFICIAL_COMBINATIONS,
    train_fraction: float = OFFICIAL_TRAIN_FRACTION,
    test_snr_db: Sequence[float] = OFFICIAL_TEST_SNR_DB,
) -> PreparedProtocol:
    """Reproduce the upstream EOG route exactly, seeded, without EMG/ECG.

    Upstream draws one independent permutation per source library, truncates the
    clean library to the EOG count, splits contiguously into 80 / 10 / 10 rows,
    applies ``combin_num`` further permutations *within* each split, and finally
    shuffles the training rows once.  Source-epoch statistics are recorded, not
    repaired: every source epoch appears ``combinations`` times inside its own
    split, and the clean epochs dropped by the truncation are counted.
    """

    _validate_raw_arrays(clean, artifact)
    original_clean_count = len(clean)
    rng = np.random.RandomState(seed)
    clean_ids = rng.permutation(clean.shape[0]).astype(np.int64)
    artifact_ids = rng.permutation(artifact.shape[0]).astype(np.int64)
    clean_random = np.asarray(clean[clean_ids], dtype=np.float64)
    artifact_random = np.asarray(artifact[artifact_ids], dtype=np.float64)

    # EEGforEOG_all_random = EEG_all_random[0:EOG_all_random.shape[0]]
    clean_random = clean_random[: len(artifact_random)]
    clean_ids = clean_ids[: len(artifact_random)]

    count = len(artifact_random)
    train_count = round(train_fraction * count)
    remaining = count - train_count
    validation_count = remaining // 2
    test_count = remaining - validation_count
    if train_count < 1 or validation_count < 1 or test_count < 1:
        raise ValueError("the scoped EOG split leaves an empty partition")
    spans = {
        "train": slice(0, train_count),
        "validation": slice(train_count, train_count + validation_count),
        "evaluation": slice(
            train_count + validation_count,
            train_count + validation_count + test_count,
        ),
    }

    mixed: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name in ("train", "validation", "evaluation"):
        span = spans[split_name]
        split_clean, split_clean_ids = _repeat_random_permutations(
            clean_random[span], clean_ids[span], combinations=combinations, rng=rng
        )
        split_artifact, split_artifact_ids = _repeat_random_permutations(
            artifact_random[span],
            artifact_ids[span],
            combinations=combinations,
            rng=rng,
        )
        mixed[split_name] = (
            split_clean,
            split_artifact,
            split_clean_ids,
            split_artifact_ids,
        )

    train_clean, train_artifact, train_clean_ids, train_artifact_ids = mixed["train"]
    training = _pairs_from_components(
        train_clean,
        train_artifact,
        train_clean_ids,
        train_artifact_ids,
        rng.uniform(-5.0, 5.0, len(train_clean)),
    )
    # Upstream shuffles the assembled training rows once before training.
    training = _select_pairs(
        training, rng.permutation(len(training.clean)).astype(np.int64)
    )

    (
        validation_clean,
        validation_artifact,
        validation_clean_ids,
        validation_artifact_ids,
    ) = mixed["validation"]
    validation = _pairs_from_components(
        validation_clean,
        validation_artifact,
        validation_clean_ids,
        validation_artifact_ids,
        rng.uniform(-5.0, 5.0, len(validation_clean)),
    )

    (
        evaluation_clean,
        evaluation_artifact,
        evaluation_clean_ids,
        evaluation_artifact_ids,
    ) = mixed["evaluation"]
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
            "split_semantics": (
                "upstream_contiguous_source_row_split_before_within_split_mixing"
            ),
            "source_epoch_repetitions_within_split": int(combinations),
            "source_overlap_recorded_not_repaired": True,
            "clean_source_epochs_dropped_by_truncation": int(
                original_clean_count - count
            ),
            "unused_artifact_source_epochs": 0,
            "cross_artifact_class_overlap": "not_applicable_eog_only_scope",
            "mixture_recipe": (
                "upstream_mean_square_get_rms_with_sqrt_snr_factor_preserved"
            ),
        }
    )
    return PreparedProtocol(
        protocol=PROTOCOL,
        noise_type=NOISE_TYPE,
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
        raise ImportError(f"cannot load frozen D4PM module: {path}")
    module = importlib.util.module_from_spec(specification)
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
    return module


def validate_d4pm_checkout(
    source_root: Path, *, expected_commit: str
) -> dict[str, Any]:
    """Read-only validation of the ignored source checkout and its boundary."""

    root = source_root.resolve(strict=True)
    if root != SOURCE_ROOT:
        raise ValueError("unexpected D4PM external checkout path")
    actual_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != expected_commit or expected_commit != SOURCE_COMMIT:
        raise RuntimeError("D4PM external checkout is not at the frozen commit")
    tracked_status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status.strip():
        raise RuntimeError("D4PM frozen checkout has tracked modifications")
    missing = [name for name in REQUIRED_SOURCE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"frozen D4PM checkout lacks required files: {missing}")
    if any((root / candidate).is_file() for candidate in LICENSE_CANDIDATES):
        raise RuntimeError(
            "D4PM upstream license-file status changed; re-audit before use"
        )
    return {
        "root": str(root),
        "commit": actual_commit,
        "tracked_checkout_clean": True,
        "required_files_present": True,
        "upstream_license_file": "absent_at_frozen_commit",
        "vendored": False,
    }


def load_d4pm_modules(source_root: Path, *, expected_commit: str) -> D4PMModules:
    """Load only the frozen external implementation after a read-only Git check."""

    source_status = validate_d4pm_checkout(source_root, expected_commit=expected_commit)
    root = Path(str(source_status["root"]))
    ddpm_module = _load_external_module("_d4pm_official_ddpm", root / "DDPM_joint.py")
    clean_module = _load_external_module(
        "_d4pm_official_clean_backbone", root / "denoising_model_eegdnet_class.py"
    )
    artifact_module = _load_external_module(
        "_d4pm_official_artifact_backbone",
        root / "denoising_model_eegdnet_class_noise.py",
    )
    if int(getattr(clean_module, "n_heads")) != 2:
        raise RuntimeError("frozen D4PM clean backbone is no longer two-head")
    if int(getattr(artifact_module, "n_heads")) != 4:
        raise RuntimeError("frozen D4PM artifact backbone is no longer four-head")
    return D4PMModules(
        ddpm_class=ddpm_module.DDPM,
        clean_backbone_class=clean_module.DualBranchDenoisingModel,
        artifact_backbone_class=artifact_module.DualBranchDenoisingModel_noise,
    )


def audit_d4pm_source_text(
    artifact_trainer_text: str,
    preparation_text: str,
    evaluation_text: str,
    utility_text: str,
) -> tuple[str, ...]:
    """Detect, but never repair in place, the released D4PM defects."""

    findings: list[str] = []
    if (
        "from denoising_model_eegdnet_class_noise import DualBranchDenoisingModel_noise"
        in artifact_trainer_text
        and "DualBranchDenoisingModel_noise(" not in artifact_trainer_text
        and "DualBranchDenoisingModel(config['train']['feats'])" in artifact_trainer_text
    ):
        findings.append("artifact_trainer_instantiates_an_undefined_model_name")
    if "ECG_all_epochs.npy" in preparation_text:
        findings.append("preparation_unconditionally_requires_an_absent_ecg_library")
    if ".data/data_for_test" in preparation_text and (
        "./data/data_for_test" in evaluation_text
    ):
        findings.append("prepared_output_path_differs_from_the_evaluation_input_path")
    if "np.random.seed" not in preparation_text and "torch.manual_seed" not in (
        artifact_trainer_text
    ):
        findings.append("no_random_seed_is_defined_anywhere_in_the_released_route")
    if "torch.save(model.state_dict()" in utility_text and (
        "optimizer.state_dict()" not in utility_text
    ):
        findings.append("released_checkpoints_contain_model_weights_only")
    if "[i, :50, :, :]" in evaluation_text:
        findings.append("released_evaluation_uses_a_fifty_row_example_subset")
    return tuple(findings)


EXPECTED_SOURCE_FINDINGS = (
    "artifact_trainer_instantiates_an_undefined_model_name",
    "preparation_unconditionally_requires_an_absent_ecg_library",
    "prepared_output_path_differs_from_the_evaluation_input_path",
    "no_random_seed_is_defined_anywhere_in_the_released_route",
    "released_checkpoints_contain_model_weights_only",
    "released_evaluation_uses_a_fifty_row_example_subset",
)


def run_d4pm_cpu_validation(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Validate source/config/data headers on one scheduled CPU job."""

    validate_d4pm_config(config)
    source = _mapping(config, "source")
    source_status = validate_d4pm_checkout(
        Path(str(source["root"])), expected_commit=str(source["commit"])
    )
    root = Path(str(source_status["root"]))
    findings = audit_d4pm_source_text(
        (root / "train_d4pm_artifacts.py").read_text(encoding="utf-8"),
        (root / "Data_Preparation" / "data_for_eegdnet.py").read_text(encoding="utf-8"),
        (root / "test_joint.py").read_text(encoding="utf-8"),
        (root / "utils.py").read_text(encoding="utf-8"),
    )
    if findings != EXPECTED_SOURCE_FINDINGS:
        raise RuntimeError("frozen D4PM source audit findings changed")

    data = _mapping(config, "data")
    expected_shapes = _mapping(data, "expected_shapes")
    header_shapes: dict[str, list[int]] = {}
    for key in ("clean_eeg", "eog", "emg"):
        array = np.load(Path(str(data[key])), mmap_mode="r", allow_pickle=False)
        actual_shape = tuple(int(value) for value in array.shape)
        if actual_shape != tuple(
            int(value) for value in expected_shapes[key]
        ) or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"registered EEGdenoiseNet header mismatch for {key}")
        header_shapes[key] = list(actual_shape)
    ecg_path = Path(str(data["root"])) / "ECG_all_epochs.npy"
    if ecg_path.exists():
        raise RuntimeError(
            "ECG_all_epochs.npy appeared; re-audit before widening the frozen scope"
        )

    result = {
        "status": "cpu_semantics_validated",
        "benchmark_id": BENCHMARK_ID,
        "claim_scope": "engineering_validation_only_no_benchmark_result",
        "source": source_status,
        "data_header_shapes": header_shapes,
        "ecg_library_present": False,
        "identity_unit": "source_epoch_not_participant",
        "source_findings": list(findings),
        "native_issue_policy": "preserve_and_report_do_not_repair",
        "spectral_metric_status": SPECTRAL_METRIC_STATUS,
        "full_budget": {
            "epochs": OFFICIAL_EPOCHS,
            "batch_size": OFFICIAL_BATCH_SIZE,
            "features": OFFICIAL_FEATURES,
            "mixtures": OFFICIAL_COMBINATIONS,
            "diffusion_steps": OFFICIAL_DIFFUSION_STEPS,
            "optimizer_updates_per_branch": EXPECTED_FULL_UPDATES_PER_BRANCH,
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _load_data(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data = _mapping(config, "data")
    clean = np.asarray(
        np.load(Path(str(data["clean_eeg"])), mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
    )
    artifact = np.asarray(
        np.load(Path(str(data["eog"])), mmap_mode="r", allow_pickle=False),
        dtype=np.float64,
    )
    _validate_raw_arrays(clean, artifact)
    expected_shapes = _mapping(data, "expected_shapes")
    if clean.shape != tuple(
        int(value) for value in expected_shapes["clean_eeg"]
    ) or artifact.shape != tuple(int(value) for value in expected_shapes["eog"]):
        raise ValueError("EEGdenoiseNet array shape differs from the registered release")
    return clean, artifact


def _prepare_from_config(
    config: Mapping[str, Any], *, stage: str
) -> PreparedProtocol:
    clean, artifact = _load_data(config)
    protocol_config = _mapping(_mapping(config, "protocols"), PROTOCOL)
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
    return prepare_eog_scoped(
        clean,
        artifact,
        seed=seed,
        combinations=combinations,
        train_fraction=float(split["train_fraction"]),
        test_snr_db=test_levels,
    )


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


def _model_config(
    protocol_config: Mapping[str, Any], *, stage: str, smoke: Mapping[str, Any]
) -> dict[str, Any]:
    training = dict(_mapping(protocol_config, "training"))
    diffusion = dict(_mapping(protocol_config, "diffusion"))
    sampler = dict(_mapping(protocol_config, "joint_sampler"))
    if stage == "smoke":
        training["epochs"] = int(smoke["epochs"])
        training["batch_size"] = int(smoke["batch_size"])
        training["evaluation_batch_size"] = int(smoke["evaluation_batch_size"])
        diffusion["num_steps"] = int(smoke["diffusion_steps"])
    return {"train": training, "diffusion": diffusion, "joint_sampler": sampler}


def _build_model(
    modules: D4PMModules,
    model_config: Mapping[str, Any],
    *,
    arm: str,
    device: torch.device,
) -> nn.Module:
    """Build one arm.

    The joint arm returns a ``ModuleDict`` holding both upstream DDPM branches so
    that a single checkpoint carries both.  The artifact branch uses the
    four-head noise class, which is the ``minimally_repaired_source-faithful``
    reading of the released ``NameError``.
    """

    features = int(_mapping(model_config, "train")["features"])
    payload = {
        "diffusion": dict(_mapping(model_config, "diffusion")),
        "train": dict(_mapping(model_config, "train")),
    }
    if arm == "joint_dual_diffusion":
        clean = modules.ddpm_class(
            modules.clean_backbone_class(features).to(device), payload, device
        ).to(device)
        artifact = modules.ddpm_class(
            modules.artifact_backbone_class(features).to(device), payload, device
        ).to(device)
        return nn.ModuleDict({"clean_eeg": clean, "artifact": artifact}).to(device)
    if arm == "matched_deterministic":
        return MatchedConditionOnly(
            modules.clean_backbone_class(features).to(device)
        ).to(device)
    raise ValueError(f"unknown D4PM arm: {arm}")


def _branch_modules(model: nn.Module, arm: str) -> tuple[tuple[str, nn.Module], ...]:
    if arm == "joint_dual_diffusion":
        return (("clean_eeg", model["clean_eeg"]), ("artifact", model["artifact"]))
    return (("clean_eeg", model),)


def _data_loader(
    pairs: EpochPairs,
    *,
    batch_size: int,
    workers: int,
    shuffle_seed: int | None,
    drop_last: bool,
) -> DataLoader:
    count = len(pairs.clean)
    dataset = TensorDataset(
        torch.from_numpy(pairs.clean).unsqueeze(1),
        torch.from_numpy(pairs.artifact).unsqueeze(1),
        torch.from_numpy(pairs.noisy).unsqueeze(1),
        torch.full((count, 1), float(EOG_CLASS_LABEL), dtype=torch.float32),
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


def _branch_loss(
    branch: nn.Module,
    branch_name: str,
    arm: str,
    clean: Tensor,
    artifact: Tensor,
    noisy: Tensor,
    label: Tensor,
) -> Tensor:
    if arm == "joint_dual_diffusion":
        target = clean if branch_name == "clean_eeg" else artifact
        return branch(target, noisy, label)
    restored = branch(noisy, label)
    return torch.nn.functional.l1_loss(restored, clean, reduction="sum")


def _validation_losses(
    model: nn.Module,
    arm: str,
    pairs: EpochPairs,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> dict[str, float]:
    loader = _data_loader(
        pairs,
        batch_size=batch_size,
        workers=workers,
        shuffle_seed=None,
        drop_last=True,
    )
    model.eval()
    branches = _branch_modules(model, arm)
    totals = {name: 0.0 for name, _ in branches}
    batches = 0
    with torch.no_grad():
        for clean, artifact, noisy, label in loader:
            clean = clean.to(device)
            artifact = artifact.to(device)
            noisy = noisy.to(device)
            label = label.to(device)
            for name, branch in branches:
                loss = _branch_loss(
                    branch, name, arm, clean, artifact, noisy, label
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite D4PM validation loss")
                totals[name] += float(loss)
            batches += 1
    if batches == 0:
        raise RuntimeError("D4PM validation has no complete batch")
    return {name: value / batches for name, value in totals.items()}


def _checkpoint_contract(
    config: Mapping[str, Any],
    *,
    arm: str,
    stage: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "source_commit": SOURCE_COMMIT,
        "dataset": "EEGdenoiseNet",
        "identity_unit": "source_epoch_not_participant",
        "protocol": PROTOCOL,
        "noise_type": NOISE_TYPE,
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
        raise ValueError("cannot write empty D4PM metrics")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def d4pm_rrmse_temporal(denoised: np.ndarray, clean: np.ndarray) -> float:
    """Upstream ``rrmse_per_sample`` mean, vectorised over rows."""

    mse = np.mean(np.square(clean - denoised), axis=-1)
    reference = np.mean(np.square(clean), axis=-1)
    if bool((reference <= 0.0).any()):
        raise ValueError("D4PM temporal RRMSE reference power is degenerate")
    return float(np.mean(np.sqrt(mse) / np.sqrt(reference)))


def d4pm_correlation(denoised: np.ndarray, clean: np.ndarray) -> float:
    """Upstream ``compute_cc_per_sample`` mean (identical to per-row pearsonr)."""

    clean_centered = clean - clean.mean(axis=-1, keepdims=True)
    denoised_centered = denoised - denoised.mean(axis=-1, keepdims=True)
    numerator = np.sum(clean_centered * denoised_centered, axis=-1)
    denominator = np.sqrt(
        np.sum(np.square(clean_centered), axis=-1)
        * np.sum(np.square(denoised_centered), axis=-1)
    )
    if bool((denominator <= 0.0).any()):
        raise ValueError("D4PM correlation denominator is degenerate")
    return float(np.mean(numerator / denominator))


def d4pm_snr_db(denoised: np.ndarray, clean: np.ndarray) -> float:
    """Upstream ``snr_per_sample`` mean, including its 1e-8 guard."""

    signal_power = np.sum(np.square(clean), axis=-1)
    error_power = np.sum(np.square(clean - denoised), axis=-1)
    return float(np.mean(10.0 * np.log10(signal_power / (error_power + 1.0e-8))))


def d4pm_rrmse_spectral_fft_magnitude(
    denoised: np.ndarray, clean: np.ndarray
) -> float:
    """Upstream spectral RRMSE: RRMSE over 512-bin FFT magnitudes.

    The EEGDfus-style 400-vs-512 zero-denominator shape defect was checked for
    and is absent here: ``test_joint.py`` transforms both signals with the same
    full-length FFT, so numerator and denominator shapes agree.  The name stays
    explicit because this is an FFT-magnitude ratio, not the Welch-PSD
    ``RRMSE_s`` of the EEGdenoiseNet literature.
    """

    if denoised.shape != clean.shape:
        raise ValueError("D4PM spectral RRMSE inputs have incompatible shapes")
    clean_magnitude = np.abs(np.fft.fft(clean, axis=-1))
    denoised_magnitude = np.abs(np.fft.fft(denoised, axis=-1))
    value = d4pm_rrmse_temporal(denoised_magnitude, clean_magnitude)
    if not math.isfinite(value):
        raise FloatingPointError("D4PM spectral RRMSE is non-finite")
    return value


def _restore_arm(
    model: nn.Module,
    arm: str,
    noisy: Tensor,
    label: Tensor,
    *,
    sampler: Mapping[str, Any],
) -> Tensor:
    if arm == "joint_dual_diffusion":
        clean_branch = model["clean_eeg"]
        # Upstream sets ``model_eeg.model_h = model_noise.model``.  Writing
        # through ``__dict__`` keeps the identical attribute lookup without
        # re-registering the artifact backbone as a second submodule, which
        # would duplicate it inside any later state_dict.
        clean_branch.__dict__["model_h"] = model["artifact"].model
        return clean_branch.joint_denoising(
            noisy,
            label,
            lambda_dc=float(sampler["lambda_dc"]),
            gamma=float(sampler["gamma"]),
            eta=float(sampler["eta"]),
            continous=False,
        )
    return model(noisy, label)


def _evaluate(
    model: nn.Module,
    prepared: PreparedProtocol,
    *,
    arm: str,
    diffusion_steps: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    sampler: Mapping[str, Any],
    row_limit: int | None = None,
    evaluation_scope: str = "all_frozen_test_rows",
) -> list[dict[str, Any]]:
    model.eval()
    branches = len(_branch_modules(model, arm))
    rows: list[dict[str, Any]] = []
    for level in prepared.evaluation:
        pairs = level.pairs
        if row_limit is not None:
            index = np.arange(min(int(row_limit), len(pairs.clean)), dtype=np.int64)
            pairs = _select_pairs(pairs, index)
        loader = _data_loader(
            pairs,
            batch_size=batch_size,
            workers=workers,
            shuffle_seed=None,
            drop_last=False,
        )
        outputs: list[np.ndarray] = []
        start = time.perf_counter()
        with torch.no_grad():
            for _clean, _artifact, noisy, label in loader:
                restored = _restore_arm(
                    model, arm, noisy.to(device), label.to(device), sampler=sampler
                )
                if not bool(torch.isfinite(restored).all()):
                    raise FloatingPointError("non-finite D4PM evaluation output")
                outputs.append(restored.detach().cpu().numpy()[:, 0, :])
        elapsed = time.perf_counter() - start
        output = np.asarray(np.concatenate(outputs, axis=0), dtype=np.float64)
        clean = np.asarray(pairs.clean, dtype=np.float64)
        noisy_reference = np.asarray(pairs.noisy, dtype=np.float64)
        rows.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "protocol": prepared.protocol,
                "noise_type": prepared.noise_type,
                "arm": arm,
                "identity_unit": "source_epoch_not_participant",
                "evaluation_scope": evaluation_scope,
                "snr_db": level.snr_db,
                "evaluation_mixtures": len(clean),
                "rrmse_temporal": d4pm_rrmse_temporal(output, clean),
                "correlation": d4pm_correlation(output, clean),
                "snr_output_db": d4pm_snr_db(output, clean),
                "snr_input_db": d4pm_snr_db(noisy_reference, clean),
                "snr_improvement_db": (
                    d4pm_snr_db(output, clean) - d4pm_snr_db(noisy_reference, clean)
                ),
                "rrmse_spectral_fft_magnitude": d4pm_rrmse_spectral_fft_magnitude(
                    output, clean
                ),
                "rrmse_spectral_metric_status": SPECTRAL_METRIC_STATUS,
                "evaluation_seconds": elapsed,
                "network_calls_per_output": (
                    diffusion_steps * branches
                    if arm == "joint_dual_diffusion"
                    else 1
                ),
            }
        )
    return rows


def _task_output_paths(
    config: Mapping[str, Any], *, stage: str, arm: str
) -> tuple[Path, Path, Path]:
    outputs = _mapping(config, "outputs")
    result_root = Path(str(outputs["result_root"])).resolve()
    checkpoint_root = Path(str(outputs["checkpoint_root"])).resolve()
    if CODE_ROOT not in result_root.parents or CODE_ROOT not in checkpoint_root.parents:
        raise ValueError("D4PM outputs must remain under the code root")
    task_name = f"{PROTOCOL}/{NOISE_TYPE.lower()}/{arm}"
    return (
        result_root / stage / task_name,
        checkpoint_root / stage / task_name / "last.pt",
        checkpoint_root / stage / task_name / "best.pt",
    )


def run_d4pm_stage(
    config: Mapping[str, Any],
    *,
    stage: str,
    task_index: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train/resume/evaluate one frozen matrix cell on one scheduled GPU.

    The joint arm trains BOTH upstream branches in this single cell.  A single
    Adam and StepLR span both branches, which is numerically identical to two
    independent Adam/StepLR pairs because Adam state is per-parameter and only
    the branch whose loss was just backpropagated holds gradients at its step;
    it buys one checkpointable optimizer state.  Every checkpoint carries both
    branch weights, optimizer, scheduler, epoch, step, and RNG state, so a
    SIGUSR1 walltime kill wastes nothing and exits 75 for requeue.
    """

    validate_d4pm_config(config)
    if stage not in {"smoke", "full"}:
        raise ValueError("D4PM stage must be smoke or full")
    if not 0 <= int(task_index) < len(TASK_MATRIX):
        raise ValueError("D4PM task index must lie in [0,1]")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D4PM benchmark requires a scheduled CUDA allocation")

    _protocol, _noise_type, arm = TASK_MATRIX[int(task_index)]
    protocol_config = _mapping(_mapping(config, "protocols"), PROTOCOL)
    smoke = _mapping(config, "smoke_only_overrides")
    model_config = _model_config(protocol_config, stage=stage, smoke=smoke)
    training = _mapping(model_config, "train")
    sampler = _mapping(model_config, "joint_sampler")
    if stage == "full" and (
        int(training["epochs"]) != OFFICIAL_EPOCHS
        or int(training["batch_size"]) != OFFICIAL_BATCH_SIZE
        or int(training["features"]) != OFFICIAL_FEATURES
        or int(_mapping(model_config, "diffusion")["num_steps"])
        != OFFICIAL_DIFFUSION_STEPS
    ):
        raise AssertionError("full D4PM budget differs from frozen upstream semantics")

    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    if accumulation < 1 or batch_size % accumulation != 0:
        raise ValueError("gradient accumulation must divide the frozen batch exactly")
    micro_batch = batch_size // accumulation

    seed = int(_mapping(config, "randomness")["adapter_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prepared = _prepare_from_config(config, stage=stage)
    modules = load_d4pm_modules(
        Path(str(_mapping(config, "source")["root"])),
        expected_commit=str(_mapping(config, "source")["commit"]),
    )
    model = _build_model(modules, model_config, arm=arm, device=device)
    branches = _branch_modules(model, arm)
    optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["scheduler_step_epochs"]),
        gamma=float(training["scheduler_gamma"]),
    )
    output_dir, checkpoint, best_checkpoint = _task_output_paths(
        config, stage=stage, arm=arm
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **dict(config),
        "resolved_task": {
            "task_index": int(task_index),
            "stage": stage,
            "protocol": PROTOCOL,
            "noise_type": NOISE_TYPE,
            "arm": arm,
            "branches": [name for name, _ in branches],
            "model_config": model_config,
            "execution_adaptation": {
                "gradient_accumulation_steps": accumulation,
                "micro_batch_size": micro_batch,
                "effective_batch_size": batch_size,
                "enabled": accumulation > 1,
            },
        },
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    _write_manifest(
        output_dir / "split_manifest.csv", source_split_manifest_rows(prepared)
    )

    contract = _checkpoint_contract(
        config, arm=arm, stage=stage, model_config=model_config
    )
    start_epoch = 0
    updates_per_branch = 0
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
            raise ValueError("D4PM checkpoint preprocessing state mismatch")
        start_epoch = state.epoch + 1
        updates_per_branch = int(state.extra.get("optimizer_updates_per_branch", 0))
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
    workers = int(training["workers"])
    validation_interval = int(training["validation_interval_epochs"])
    checkpoint_interval = int(training["checkpoint_interval_epochs"])
    gradient_clip = float(training["gradient_clip"])
    last_epoch = start_epoch - 1
    last_validation_losses: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        for epoch in range(start_epoch, epochs):
            loader = _data_loader(
                prepared.train,
                batch_size=micro_batch,
                workers=workers,
                shuffle_seed=seed + 10_000 + epoch,
                drop_last=True,
            )
            model.train()
            batches = 0
            pending = 0
            optimizer.zero_grad(set_to_none=True)
            for clean_batch, artifact_batch, noisy_batch, label_batch in loader:
                clean_batch = clean_batch.to(device)
                artifact_batch = artifact_batch.to(device)
                noisy_batch = noisy_batch.to(device)
                label_batch = label_batch.to(device)
                pending += 1
                for name, branch in branches:
                    loss = _branch_loss(
                        branch,
                        name,
                        arm,
                        clean_batch,
                        artifact_batch,
                        noisy_batch,
                        label_batch,
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(
                            f"non-finite D4PM loss at epoch={epoch} "
                            f"branch={name} update={updates_per_branch}"
                        )
                    loss.backward()
                if pending < accumulation:
                    continue
                for _name, branch in branches:
                    torch.nn.utils.clip_grad_norm_(
                        branch.model.parameters(), gradient_clip
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                updates_per_branch += 1
                batches += 1
            if batches == 0:
                raise RuntimeError("D4PM training has no complete batch")
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
                last_validation_losses = _validation_losses(
                    model,
                    arm,
                    prepared.validation,
                    batch_size=micro_batch,
                    workers=workers,
                    device=device,
                )
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                torch.cuda.set_rng_state_all(cuda_state)
                total_validation = float(sum(last_validation_losses.values()))
                improved = total_validation < best_validation_loss
                if improved:
                    best_validation_loss = total_validation

            save_now = (
                (epoch + 1) % checkpoint_interval == 0
                or stop_requested
                or epoch + 1 == epochs
            )
            if save_now:
                extra = {
                    "best_validation_loss": best_validation_loss,
                    "last_validation_losses": dict(last_validation_losses),
                    "optimizer_updates_per_branch": updates_per_branch,
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
                    step=updates_per_branch,
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
                        step=updates_per_branch,
                        config=contract,
                        normalizer={"kind": "upstream_per_epoch_scale"},
                        extra=extra,
                    )
            if stop_requested:
                break
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    training_seconds = time.perf_counter() - started
    if stop_requested:
        summary = {
            "status": "checkpointed_for_resume",
            "benchmark_id": BENCHMARK_ID,
            "protocol": PROTOCOL,
            "noise_type": NOISE_TYPE,
            "arm": arm,
            "stage": stage,
            "epochs_completed": last_epoch + 1,
            "optimizer_updates_per_branch": updates_per_branch,
            "checkpoint": str(checkpoint),
            "resume_supported": True,
            "resume_command": str(_mapping(config, "execution")["resume_command"]),
            "scientific_result_eligible": False,
        }
        (run_dir / "result_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    planned_updates = epochs * (len(prepared.train.clean) // batch_size)
    if last_epoch + 1 != epochs or not best_checkpoint.is_file():
        raise RuntimeError("D4PM full budget did not complete")
    if updates_per_branch != planned_updates:
        raise AssertionError(
            "D4PM completed without the exact frozen optimizer-update budget: "
            f"{updates_per_branch} != {planned_updates}"
        )
    if stage == "full" and planned_updates != EXPECTED_FULL_UPDATES_PER_BRANCH:
        raise AssertionError("D4PM full per-branch update plan drifted from the config")

    best_payload = load_training_checkpoint(best_checkpoint, map_location=device)
    if best_payload["config"] != contract:
        raise ValueError("best D4PM checkpoint contract mismatch")
    if best_payload["normalizer_state"] != {"kind": "upstream_per_epoch_scale"}:
        raise ValueError("best D4PM checkpoint preprocessing state mismatch")
    model.load_state_dict(best_payload["model_state"])
    diffusion_steps = int(_mapping(model_config, "diffusion")["num_steps"])
    evaluation_batch = int(training["evaluation_batch_size"])
    metrics = _evaluate(
        model,
        prepared,
        arm=arm,
        diffusion_steps=diffusion_steps,
        batch_size=evaluation_batch,
        workers=workers,
        device=device,
        sampler=sampler,
    )
    _write_metrics(output_dir / "metrics.csv", metrics)
    diagnostic = _evaluate(
        model,
        prepared,
        arm=arm,
        diffusion_steps=diffusion_steps,
        batch_size=evaluation_batch,
        workers=workers,
        device=device,
        sampler=sampler,
        row_limit=UPSTREAM_EXAMPLE_ROWS,
        evaluation_scope="upstream_example_first_50_rows_diagnostic_only",
    )
    diagnostic_name = str(
        _mapping(config, "outputs")["upstream_example_diagnostic_filename"]
    )
    _write_metrics(output_dir / diagnostic_name, diagnostic)

    summary = {
        "status": "completed_tiny_smoke_only" if stage == "smoke" else "completed",
        "benchmark_id": BENCHMARK_ID,
        "protocol": PROTOCOL,
        "noise_type": NOISE_TYPE,
        "arm": arm,
        "stage": stage,
        "tiny_smoke_only": stage == "smoke",
        "scientific_result_eligible": stage == "full",
        "claim_scope": "single_channel_EOG_stress_test_only",
        "identity_unit": "source_epoch_not_participant",
        "source_commit": SOURCE_COMMIT,
        "source_load_policy": "dynamic_external_checkout_no_vendoring",
        "upstream_seed_defined": False,
        "adapter_seed": seed,
        "resumed": resumed,
        "branches": [name for name, _ in branches],
        "epochs_completed": last_epoch + 1,
        "optimizer_updates_per_branch": updates_per_branch,
        "planned_optimizer_updates_per_branch": planned_updates,
        "optimizer_updates_total": updates_per_branch * len(branches),
        "matched_update_budget": updates_per_branch == planned_updates,
        "gradient_accumulation_steps": accumulation,
        "micro_batch_size": micro_batch,
        "effective_batch_size": batch_size,
        "diffusion_steps": diffusion_steps,
        "joint_sampler": dict(sampler),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "gpu_name": torch.cuda.get_device_name(device),
        "source_audit": dict(prepared.source_audit),
        "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "resume_supported": True,
        "resume_command": str(_mapping(config, "execution")["resume_command"]),
        "metrics": str(output_dir / "metrics.csv"),
        "upstream_example_diagnostic": str(output_dir / diagnostic_name),
        "split_manifest": str(output_dir / "split_manifest.csv"),
        "run_dir": str(run_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "git_head": os.environ.get("DENOISENET_GIT_HEAD", ""),
        "spectral_metric_status": SPECTRAL_METRIC_STATUS,
        "scoped_known_issues_preserved": [
            "ecg_route_not_reproducible_scope_frozen_to_eog",
            "upstream_get_rms_mean_square_snr_recipe_preserved",
            "source_epoch_repeats_eleven_times_within_each_split",
            "artifact_branch_four_head_repair_minimally_repaired_source-faithful",
        ],
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
        raise ValueError(f"D4PM metric {key!r} is missing or non-numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"D4PM metric {key!r} is non-finite")
    return value


METRIC_NAMES = (
    "rrmse_temporal",
    "correlation",
    "snr_improvement_db",
    "rrmse_spectral_fft_magnitude",
)
METRIC_DIRECTIONS = {
    "rrmse_temporal": "lower",
    "correlation": "higher",
    "snr_improvement_db": "higher",
    "rrmse_spectral_fft_magnitude": "lower",
}


def aggregate_d4pm_full_cells(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    pairing_acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and aggregate the complete frozen two-cell benchmark.

    Arm differences are paired only within an exact SNR cell, after checking
    that both arms used identical source manifests, identical reconstructed
    ordered pairings, identical evaluation counts, and the same per-branch
    optimizer-update budget.
    """

    expected = set(TASK_MATRIX)
    actual = set(cells)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"D4PM full aggregate requires both cells; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if pairing_acceptance.get("status") != (
        "passed_reconstructed_ordered_pairing_acceptance"
    ):
        raise ValueError("D4PM ordered-pair acceptance is missing or failed")
    if pairing_acceptance.get("scientific_threshold_or_method_changed") is not False:
        raise ValueError("D4PM pairing acceptance changed a scientific rule")
    if pairing_acceptance.get("submitted_and_resolved_configs_equal") is not True:
        raise ValueError("D4PM accepted full cells used unequal configs")
    if pairing_acceptance.get("both_arms_saw_identical_inputs") is not True:
        raise ValueError("D4PM arms did not provably share one ordered pairing")
    pairing_rows = pairing_acceptance.get("pairing_rows")
    if not isinstance(pairing_rows, Sequence) or isinstance(pairing_rows, (str, bytes)):
        raise ValueError("D4PM pairing reconstruction rows are missing")
    if not pairing_rows:
        raise ValueError("D4PM pairing reconstruction produced no rows")
    for row in pairing_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid D4PM pairing reconstruction row")
        if row.get("ordered_clean_artifact_snr_pairing_equal") is not True:
            raise ValueError("D4PM ordered clean/artifact/SNR pairing differs")
        if int(row.get("train_pairs", -1)) != EXPECTED_FULL_TRAIN_PAIRS:
            raise ValueError("D4PM reconstructed training count drifted")
        if int(row.get("validation_pairs", -1)) != EXPECTED_FULL_VALIDATION_PAIRS:
            raise ValueError("D4PM reconstructed validation count drifted")
        if int(row.get("evaluation_mixtures_per_snr", -1)) != (
            EXPECTED_FULL_EVALUATION_MIXTURES
        ):
            raise ValueError("D4PM reconstructed evaluation count is incomplete")
        if int(row.get("snr_levels", -1)) != len(OFFICIAL_TEST_SNR_DB):
            raise ValueError("D4PM reconstructed SNR grid is incomplete")

    expected_snr = tuple(float(value) for value in OFFICIAL_TEST_SNR_DB)
    all_metric_rows: list[dict[str, Any]] = []
    cell_summary_rows: list[dict[str, Any]] = []
    indexed_metrics: dict[tuple[str, str, str], dict[float, Mapping[str, Any]]] = {}

    for task_index, key in enumerate(TASK_MATRIX):
        _protocol, _noise_type, arm = key
        cell = cells[key]
        summary = cell.get("summary")
        metrics = cell.get("metrics")
        manifest = cell.get("split_manifest")
        if not isinstance(summary, Mapping) or not isinstance(metrics, Sequence):
            raise ValueError(f"invalid D4PM aggregate payload for {key}")
        if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes)):
            raise ValueError(f"missing D4PM source manifest for {key}")
        if (
            summary.get("status") != "completed"
            or summary.get("stage") != "full"
            or summary.get("scientific_result_eligible") is not True
        ):
            raise ValueError(f"D4PM cell is not a completed full result: {key}")
        for field, expected_value in (
            ("benchmark_id", BENCHMARK_ID),
            ("protocol", PROTOCOL),
            ("noise_type", NOISE_TYPE),
            ("arm", arm),
            ("identity_unit", "source_epoch_not_participant"),
        ):
            if summary.get(field) != expected_value:
                raise ValueError(f"D4PM summary {field} mismatch for {key}")
        updates = int(summary.get("optimizer_updates_per_branch", -1))
        planned = int(summary.get("planned_optimizer_updates_per_branch", -2))
        if (
            updates != planned
            or updates != EXPECTED_FULL_UPDATES_PER_BRANCH
            or not bool(summary.get("matched_update_budget"))
        ):
            raise ValueError(f"D4PM optimizer budget is incomplete for {key}")
        branch_count = len(list(summary.get("branches", ())))
        if branch_count != ARM_BRANCH_COUNT[arm]:
            raise ValueError(f"D4PM branch count mismatch for {key}")
        if int(summary.get("optimizer_updates_total", -1)) != updates * branch_count:
            raise ValueError(f"D4PM total update accounting mismatch for {key}")
        source_audit = summary.get("source_audit")
        if not isinstance(source_audit, Mapping):
            raise ValueError(f"D4PM source audit is missing for {key}")

        by_snr: dict[float, Mapping[str, Any]] = {}
        normalized_rows: list[dict[str, Any]] = []
        for raw_row in metrics:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"invalid D4PM metric row for {key}")
            row = dict(raw_row)
            for field, expected_value in (
                ("benchmark_id", BENCHMARK_ID),
                ("protocol", PROTOCOL),
                ("noise_type", NOISE_TYPE),
                ("arm", arm),
                ("identity_unit", "source_epoch_not_participant"),
                ("evaluation_scope", "all_frozen_test_rows"),
            ):
                if row.get(field) != expected_value:
                    raise ValueError(f"D4PM metric {field} mismatch for {key}")
            if row.get("rrmse_spectral_metric_status") != SPECTRAL_METRIC_STATUS:
                raise ValueError("D4PM spectral metric disclosure was not preserved")
            snr_db = _finite_metric(row, "snr_db")
            if snr_db in by_snr:
                raise ValueError(f"duplicate D4PM SNR={snr_db} for {key}")
            for metric_name in METRIC_NAMES:
                _finite_metric(row, metric_name)
            if int(row.get("evaluation_mixtures", 0)) != (
                EXPECTED_FULL_EVALUATION_MIXTURES
            ):
                raise ValueError(f"truncated D4PM evaluation cell: {key}")
            expected_calls = (
                OFFICIAL_DIFFUSION_STEPS * ARM_BRANCH_COUNT[arm]
                if arm == "joint_dual_diffusion"
                else 1
            )
            if int(row.get("network_calls_per_output", -1)) != expected_calls:
                raise ValueError(f"D4PM network-call budget mismatch for {key}")
            normalized = {"task_index": task_index, **row}
            normalized_rows.append(normalized)
            by_snr[snr_db] = normalized
        if tuple(sorted(by_snr)) != expected_snr:
            raise ValueError(f"D4PM full SNR grid mismatch for {key}")
        indexed_metrics[key] = by_snr
        all_metric_rows.extend(normalized_rows)

        def mean_metric(name: str) -> float:
            return float(np.mean([_finite_metric(row, name) for row in normalized_rows]))

        cell_summary_rows.append(
            {
                "task_index": task_index,
                "benchmark_id": BENCHMARK_ID,
                "protocol": PROTOCOL,
                "noise_type": NOISE_TYPE,
                "arm": arm,
                "status": "completed",
                "scientific_result_eligible": True,
                "identity_unit": "source_epoch_not_participant",
                "snr_levels": len(normalized_rows),
                "branches": branch_count,
                "optimizer_updates_per_branch": updates,
                "optimizer_updates_total": updates * branch_count,
                "training_seconds": _finite_metric(summary, "training_seconds"),
                "peak_gpu_memory_mb": _finite_metric(summary, "peak_gpu_memory_mb"),
                "gpu_name": str(summary.get("gpu_name", "")),
                "mean_rrmse_temporal": mean_metric("rrmse_temporal"),
                "mean_correlation": mean_metric("correlation"),
                "mean_snr_improvement_db": mean_metric("snr_improvement_db"),
                "mean_snr_output_db": mean_metric("snr_output_db"),
                "mean_rrmse_spectral_fft_magnitude": mean_metric(
                    "rrmse_spectral_fft_magnitude"
                ),
                "rrmse_spectral_metric_status": SPECTRAL_METRIC_STATUS,
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
                "source_epoch_repetitions_within_split": int(
                    source_audit.get("source_epoch_repetitions_within_split", 0)
                ),
                "train_evaluation_clean_source_overlap": int(
                    source_audit["train_evaluation_clean_overlap"]
                ),
                "train_evaluation_artifact_source_overlap": int(
                    source_audit["train_evaluation_artifact_overlap"]
                ),
            }
        )

    joint_key = (PROTOCOL, NOISE_TYPE, "joint_dual_diffusion")
    deterministic_key = (PROTOCOL, NOISE_TYPE, "matched_deterministic")
    joint_cell = cells[joint_key]
    deterministic_cell = cells[deterministic_key]
    if list(joint_cell["split_manifest"]) != list(deterministic_cell["split_manifest"]):
        raise ValueError("D4PM paired arms do not share an exact source manifest")
    joint_summary = joint_cell["summary"]
    deterministic_summary = deterministic_cell["summary"]
    if joint_summary["source_audit"] != deterministic_summary["source_audit"]:
        raise ValueError("D4PM paired arms have unequal source audits")
    if int(joint_summary["optimizer_updates_per_branch"]) != int(
        deterministic_summary["optimizer_updates_per_branch"]
    ):
        raise ValueError("D4PM paired arms have unequal per-branch optimizer updates")
    same_gpu = str(joint_summary.get("gpu_name", "")) == str(
        deterministic_summary.get("gpu_name", "")
    )

    paired_rows: list[dict[str, Any]] = []
    for snr_db in expected_snr:
        joint = indexed_metrics[joint_key][snr_db]
        deterministic = indexed_metrics[deterministic_key][snr_db]
        if int(joint["evaluation_mixtures"]) != int(
            deterministic["evaluation_mixtures"]
        ):
            raise ValueError(f"D4PM paired evaluation count differs at {snr_db}")
        paired = {
            "benchmark_id": BENCHMARK_ID,
            "protocol": PROTOCOL,
            "noise_type": NOISE_TYPE,
            "identity_unit": "source_epoch_not_participant",
            "snr_db": snr_db,
            "evaluation_mixtures": int(joint["evaluation_mixtures"]),
            "comparison": "joint_dual_diffusion_minus_matched_deterministic",
            "paired_source_manifest_equal": True,
            "paired_source_manifest_scope": "source_membership",
            "paired_ordered_input_reconstruction_equal": True,
            "paired_per_branch_optimizer_updates_equal": True,
            "joint_branches": ARM_BRANCH_COUNT["joint_dual_diffusion"],
            "deterministic_branches": ARM_BRANCH_COUNT["matched_deterministic"],
            "total_compute_asymmetry_favours": "joint_dual_diffusion",
            "rrmse_spectral_metric_status": SPECTRAL_METRIC_STATUS,
            "joint_gpu_name": str(joint_summary.get("gpu_name", "")),
            "deterministic_gpu_name": str(deterministic_summary.get("gpu_name", "")),
            "latency_comparison_status": (
                "comparable_same_gpu_model"
                if same_gpu
                else "descriptive_only_different_gpu_models"
            ),
            "joint_evaluation_seconds": _finite_metric(joint, "evaluation_seconds"),
            "deterministic_evaluation_seconds": _finite_metric(
                deterministic, "evaluation_seconds"
            ),
            "evaluation_seconds_delta_if_same_gpu": (
                _finite_metric(joint, "evaluation_seconds")
                - _finite_metric(deterministic, "evaluation_seconds")
                if same_gpu
                else ""
            ),
        }
        for metric_name in METRIC_NAMES:
            joint_value = _finite_metric(joint, metric_name)
            deterministic_value = _finite_metric(deterministic, metric_name)
            paired[f"joint_{metric_name}"] = joint_value
            paired[f"deterministic_{metric_name}"] = deterministic_value
            paired[f"delta_{metric_name}"] = joint_value - deterministic_value
        paired_rows.append(paired)

    paired_summary: dict[str, Any] = {
        "protocol": PROTOCOL,
        "noise_type": NOISE_TYPE,
        "snr_levels": len(paired_rows),
        "comparison": "joint_dual_diffusion_minus_matched_deterministic",
        "paired_source_manifest_equal": True,
        "paired_source_manifest_scope": "source_membership",
        "paired_ordered_input_reconstruction_equal": True,
        "paired_per_branch_optimizer_updates_equal": True,
        "latency_comparison_status": paired_rows[0]["latency_comparison_status"],
    }
    for metric_name, direction in METRIC_DIRECTIONS.items():
        deltas = np.asarray(
            [float(row[f"delta_{metric_name}"]) for row in paired_rows],
            dtype=np.float64,
        )
        wins = deltas > 0.0 if direction == "higher" else deltas < 0.0
        paired_summary[f"mean_delta_{metric_name}"] = float(np.mean(deltas))
        paired_summary[f"joint_win_count_{metric_name}"] = int(np.sum(wins))
        paired_summary[f"metric_direction_{metric_name}"] = direction

    return {
        "status": "completed_full_aggregate",
        "benchmark_id": BENCHMARK_ID,
        "scientific_result_eligible": True,
        "claim_scope": "single_channel_EOG_stress_test_only",
        "identity_unit": "source_epoch_not_participant",
        "matrix_cells_expected": len(TASK_MATRIX),
        "matrix_cells_completed": len(cell_summary_rows),
        "metric_rows_expected": len(TASK_MATRIX) * len(OFFICIAL_TEST_SNR_DB),
        "metric_rows_completed": len(all_metric_rows),
        "paired_rows_expected": len(OFFICIAL_TEST_SNR_DB),
        "paired_rows_completed": len(paired_rows),
        "input_pairing_acceptance": dict(pairing_acceptance),
        "scope_limitations": {
            "artifact_routes_executed": [NOISE_TYPE],
            "ecg": (
                "ECG_all_epochs.npy absent from the registered release and "
                "undocumented upstream; the three-artifact native protocol is "
                "not reproducible"
            ),
            "emg": "out of scope for this appendix anchor row",
            "mixture_recipe": (
                "upstream get_rms returns mean square and is combined with a "
                "square-root SNR factor; preserved exactly and disclosed"
            ),
            "source_epochs": (
                "each source epoch appears eleven times inside its own split; "
                "EEGdenoiseNet exposes no participant identity here"
            ),
        },
        "comparison_scope": (
            "paired_descriptive_across_frozen_snr_levels_not_independent_inference"
        ),
        "spectral_metric": {
            "field": "rrmse_spectral_fft_magnitude",
            "status": SPECTRAL_METRIC_STATUS,
            "upstream_denominator_shape_defect_present": False,
        },
        "cell_summary_rows": cell_summary_rows,
        "all_metric_rows": all_metric_rows,
        "paired_rows": paired_rows,
        "paired_summaries": [paired_summary],
    }


def _d4pm_aggregate_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# D4PM EOG-scoped benchmark aggregate",
        "",
        "Both frozen cells completed. This is the upstream EOG route of the D4PM "
        "native recipe, seeded and minimally repaired; it is not an exact "
        "as-released reproduction and it is not a three-artifact result.",
        "",
        "The ECG library required by the released preparation is absent from the "
        "registered EEGdenoiseNet release and undocumented upstream, so the full "
        "native protocol is not reproducible. The upstream mixture recipe, "
        "including its mean-square `get_rms` and square-root SNR factor, is "
        "preserved exactly rather than physically corrected.",
        "",
        "The spectral column is an FFT-magnitude RRMSE matching the upstream "
        "definition. It is not the Welch-PSD RRMSE_s of the EEGdenoiseNet "
        "literature and the two are not interchangeable.",
        "",
        "## Cell means",
        "",
        "| Arm | Branches | Temporal RRMSE | Correlation | SNR improvement | "
        "FFT-magnitude spectral RRMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["cell_summary_rows"]:
        lines.append(
            "| {arm} | {branches} | {rrmse:.6g} | {corr:.6g} | {snr:.6g} | "
            "{spectral:.6g} |".format(
                arm=row["arm"],
                branches=int(row["branches"]),
                rrmse=float(row["mean_rrmse_temporal"]),
                corr=float(row["mean_correlation"]),
                snr=float(row["mean_snr_improvement_db"]),
                spectral=float(row["mean_rrmse_spectral_fft_magnitude"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired joint-minus-deterministic descriptions",
            "",
            "| Δtemporal RRMSE | Δcorrelation | ΔSNR improvement | "
            "Δspectral RRMSE |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in result["paired_summaries"]:
        lines.append(
            "| {rrmse:.6g} | {corr:.6g} | {snr:.6g} | {spectral:.6g} |".format(
                rrmse=float(row["mean_delta_rrmse_temporal"]),
                corr=float(row["mean_delta_correlation"]),
                snr=float(row["mean_delta_snr_improvement_db"]),
                spectral=float(row["mean_delta_rrmse_spectral_fft_magnitude"]),
            )
        )
    lines.extend(
        [
            "",
            "Both arms received the same ordered clean/artifact/SNR pairing, "
            "reconstructed deterministically from the frozen config and seed. "
            "Per-branch optimizer updates are equal; the joint arm trains two "
            "branches, so its total optimizer-step and parameter budget is about "
            "twice the deterministic arm's -- an asymmetry that favours the "
            "diffusion arm and is disclosed rather than corrected.",
            "",
            "EEGdenoiseNet exposes source epochs rather than participant "
            "identities; these results cannot support participant-specific or "
            "real-EEG deployment claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _training_config_view(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the aggregate-side route annotation before config comparison."""

    value = json.loads(json.dumps(dict(config)))
    execution = dict(_mapping(value, "execution"))
    stages = list(execution.get("stages", ()))
    if stages and stages[-1] == "aggregate-full":
        stages.pop()
    execution["stages"] = stages
    execution.pop("aggregate_command", None)
    value["execution"] = execution
    value.pop("resolved_task", None)
    return value


def _d4pm_pairing_acceptance(
    config: Mapping[str, Any], *, result_root: Path
) -> dict[str, Any]:
    """Bind both cells to their producing runs and reconstruct the ordered pairing.

    The reconstruction re-derives each arm's clean/artifact/SNR ordering from the
    frozen config plus the frozen seed and requires the two arms to be identical.
    It reads no performance metric and changes no decision threshold.
    """

    expected_config = _training_config_view(config)
    git_heads: set[str] = set()
    task_configs: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    bindings: list[dict[str, Any]] = []

    for task_index, key in enumerate(TASK_MATRIX):
        _protocol, _noise_type, arm = key
        cell_dir = result_root / "full" / PROTOCOL / NOISE_TYPE.lower() / arm
        summary_path = cell_dir / "result_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"incomplete D4PM full cell: {cell_dir}")
        canonical_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = Path(str(canonical_summary.get("run_dir", "")))
        if not run_dir.is_dir() or CODE_ROOT not in run_dir.parents:
            raise ValueError(f"D4PM cell {arm} is not bound to a run directory")
        required = {
            "config": run_dir / "config.yaml",
            "git_head": run_dir / "git_head.txt",
            "job_id": run_dir / "slurm_job_id.txt",
            "task_id": run_dir / "slurm_array_task_id.txt",
            "summary": run_dir / "result_summary.json",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"D4PM producer run {arm} is incomplete: {missing}")
        if int(required["task_id"].read_text(encoding="utf-8").strip()) != task_index:
            raise ValueError("D4PM producer array task index mismatch")
        if str(canonical_summary.get("slurm_job_id", "")) != (
            required["job_id"].read_text(encoding="utf-8").strip()
        ):
            raise ValueError("D4PM producer job ID mismatch")
        if json.loads(required["summary"].read_text(encoding="utf-8")) != (
            canonical_summary
        ):
            raise ValueError("D4PM canonical cell does not match its producer output")
        if canonical_summary.get("metrics") != str(cell_dir / "metrics.csv") or (
            canonical_summary.get("split_manifest")
            != str(cell_dir / "split_manifest.csv")
        ):
            raise ValueError("D4PM producer summary does not bind canonical artifacts")
        submitted_config = yaml.safe_load(
            required["config"].read_text(encoding="utf-8")
        )
        if not isinstance(submitted_config, Mapping) or _training_config_view(
            submitted_config
        ) != expected_config:
            raise ValueError("D4PM cells used unequal submitted configs")
        resolved = yaml.safe_load(
            (cell_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(resolved, Mapping):
            raise ValueError("D4PM cell resolved config is not a mapping")
        resolved_task = _mapping(resolved, "resolved_task")
        for field, expected_value in (
            ("task_index", task_index),
            ("stage", "full"),
            ("protocol", PROTOCOL),
            ("noise_type", NOISE_TYPE),
            ("arm", arm),
        ):
            if resolved_task.get(field) != expected_value:
                raise ValueError(f"D4PM resolved task {field} mismatch")
        if _training_config_view(resolved) != expected_config:
            raise ValueError("D4PM resolved cell config differs from the submitted one")
        git_head = required["git_head"].read_text(encoding="utf-8").strip()
        if not git_head:
            raise ValueError("D4PM producer run has an empty Git HEAD")
        git_heads.add(git_head)
        task_configs[key] = submitted_config
        bindings.append(
            {
                "task_index": task_index,
                "arm": arm,
                "run_dir": str(run_dir),
                "slurm_job_id": str(canonical_summary.get("slurm_job_id", "")),
            }
        )

    if len(git_heads) != 1:
        raise ValueError("D4PM cells used mixed Git revisions")

    joint = _prepare_from_config(
        task_configs[(PROTOCOL, NOISE_TYPE, "joint_dual_diffusion")], stage="full"
    )
    deterministic = _prepare_from_config(
        task_configs[(PROTOCOL, NOISE_TYPE, "matched_deterministic")], stage="full"
    )
    if not _prepared_pairing_equal(joint, deterministic):
        raise ValueError("D4PM reconstructed ordered pairing differs between arms")

    return {
        "status": "passed_reconstructed_ordered_pairing_acceptance",
        "git_head": next(iter(git_heads)),
        "task_indices": list(range(len(TASK_MATRIX))),
        "cell_bindings": bindings,
        "submitted_and_resolved_configs_equal": True,
        "both_arms_saw_identical_inputs": True,
        "reconstruction_basis": "frozen_config_plus_frozen_adapter_seed",
        "artifact_binding_scope": (
            "exact canonical paths in the producer summary; no content hashes "
            "under HARNESS_LEVEL=1"
        ),
        "scientific_threshold_or_method_changed": False,
        "pairing_rows": [
            {
                "protocol": PROTOCOL,
                "noise_type": NOISE_TYPE,
                "train_pairs": len(joint.train.clean),
                "validation_pairs": len(joint.validation.clean),
                "evaluation_mixtures_per_snr": len(joint.evaluation[0].pairs.clean),
                "snr_levels": len(joint.evaluation),
                "ordered_clean_artifact_snr_pairing_equal": True,
            }
        ],
    }


def run_d4pm_full_aggregate(
    config: Mapping[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Load both full matrix artifacts and write one small CPU aggregate."""

    validate_d4pm_config(config)
    result_root = Path(str(_mapping(config, "outputs")["result_root"])).resolve()
    if CODE_ROOT not in result_root.parents:
        raise ValueError("D4PM result root must remain under the code root")
    pairing_acceptance = _d4pm_pairing_acceptance(config, result_root=result_root)
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in TASK_MATRIX:
        arm = key[2]
        cell_dir = result_root / "full" / PROTOCOL / NOISE_TYPE.lower() / arm
        summary_path = cell_dir / "result_summary.json"
        metrics_path = cell_dir / "metrics.csv"
        manifest_path = cell_dir / "split_manifest.csv"
        if (
            not summary_path.is_file()
            or not metrics_path.is_file()
            or not manifest_path.is_file()
        ):
            raise FileNotFoundError(f"incomplete D4PM full cell: {cell_dir}")
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            metric_rows = list(csv.DictReader(stream))
        with manifest_path.open("r", encoding="utf-8", newline="") as stream:
            manifest_rows = list(csv.DictReader(stream))
        cells[key] = {
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
            "metrics": metric_rows,
            "split_manifest": manifest_rows,
        }

    result = aggregate_d4pm_full_cells(cells, pairing_acceptance=pairing_acceptance)
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
        _d4pm_aggregate_markdown(result), encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(public_result, indent=2) + "\n", encoding="utf-8"
    )
    return public_result


__all__ = [
    "ARM_BRANCH_COUNT",
    "BENCHMARK_ID",
    "EXPECTED_FULL_EVALUATION_MIXTURES",
    "EXPECTED_FULL_TRAIN_PAIRS",
    "EXPECTED_FULL_UPDATES_PER_BRANCH",
    "EXPECTED_FULL_VALIDATION_PAIRS",
    "EXPECTED_SOURCE_FINDINGS",
    "JOINT_ETA",
    "JOINT_GAMMA",
    "JOINT_LAMBDA_DC",
    "OFFICIAL_BATCH_SIZE",
    "OFFICIAL_COMBINATIONS",
    "OFFICIAL_DIFFUSION_STEPS",
    "OFFICIAL_EPOCHS",
    "OFFICIAL_FEATURES",
    "OFFICIAL_TEST_SNR_DB",
    "SOURCE_COMMIT",
    "SPECTRAL_METRIC_STATUS",
    "TASK_MATRIX",
    "D4PMModules",
    "EpochPairs",
    "EvaluationLevel",
    "MatchedConditionOnly",
    "PreparedProtocol",
    "aggregate_d4pm_full_cells",
    "audit_d4pm_source_text",
    "d4pm_correlation",
    "d4pm_rrmse_spectral_fft_magnitude",
    "d4pm_rrmse_temporal",
    "d4pm_snr_db",
    "load_d4pm_modules",
    "prepare_eog_scoped",
    "run_d4pm_cpu_validation",
    "run_d4pm_full_aggregate",
    "run_d4pm_stage",
    "source_split_manifest_rows",
    "validate_d4pm_checkout",
    "validate_d4pm_config",
]
