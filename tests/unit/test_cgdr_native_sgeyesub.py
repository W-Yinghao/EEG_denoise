"""Contracts for the release-internal native SGEYESUB Python port."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eeg_cgdr.baselines.native_sgeyesub import (
    OFFICIAL_ALPHA,
    OFFICIAL_BETA,
    OFFICIAL_SOURCE_COMMIT,
    REFERENCE_EQUIVALENCE_STATUS,
    NativeSGEyeSubConfig,
    fit_native_sgeyesub,
)


CHANNEL_TYPES = ("EEG", "EEG", "EEG", "EEG", "EEG", "EEG", "EOG", "artifactclasses")
CHANNEL_LABELS = tuple(f"channel_{index}" for index in range(len(CHANNEL_TYPES)))
LAYOUT_ID = "layout_test_exact"


def _six_class_calibration() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic full-rank block-1 data with all six official classes."""

    rng = np.random.default_rng(20260802)
    samples_per_class = 160
    labels = np.repeat(np.arange(1, 7, dtype=np.int64), samples_per_class)
    samples = labels.size
    noise = rng.normal(scale=0.35, size=(6, samples))
    horizontal = np.zeros(samples, dtype=np.float64)
    horizontal[labels == 1] = -1.0
    horizontal[labels == 2] = 1.0
    vertical = np.zeros(samples, dtype=np.float64)
    vertical[labels == 3] = -1.0
    vertical[labels == 4] = 1.0
    blink = np.zeros(samples, dtype=np.float64)
    blink[labels == 5] = 2.0
    horizontal_mixing = np.array([1.3, -0.9, 0.5, 0.1, -0.2, 0.3])[:, None]
    vertical_mixing = np.array([0.2, 0.4, -1.2, 1.0, 0.3, -0.1])[:, None]
    blink_mixing = np.array([1.1, 0.8, 0.4, -0.3, -0.6, -0.9])[:, None]
    eeg = (
        noise
        + horizontal_mixing @ horizontal[None, :]
        + vertical_mixing @ vertical[None, :]
        + blink_mixing @ blink[None, :]
    )
    eog = 0.8 * horizontal + 0.6 * vertical + blink
    label_channel = labels.astype(np.float64)
    return np.vstack([eeg, eog, label_channel]), labels


def _fit_eligible():
    data, labels = _six_class_calibration()
    outcome = fit_native_sgeyesub(
        data,
        labels,
        channel_labels=CHANNEL_LABELS,
        channel_types=CHANNEL_TYPES,
        layout_id=LAYOUT_ID,
        block_id=1,
    )
    assert outcome.status == "eligible", outcome.reasons
    assert outcome.model is not None
    return outcome


def test_native_sgeyesub_fit_apply_shape_and_only_eeg_channels_change() -> None:
    outcome = _fit_eligible()
    model = outcome.model
    assert model is not None
    rng = np.random.default_rng(17)
    block2 = rng.normal(size=(len(CHANNEL_TYPES), 80, 3))

    corrected = model.apply(
        block2,
        channel_labels=CHANNEL_LABELS,
        channel_types=CHANNEL_TYPES,
        layout_id=LAYOUT_ID,
        block_id=2,
    )

    assert corrected.shape == block2.shape
    assert corrected.dtype == np.float64
    assert model.correction_matrix.shape == (6, 6)
    assert model.unmixing_matrix.shape == (6, 3)
    assert model.mixing_matrix.shape == (6, 3)
    assert model.channel_labels == CHANNEL_LABELS
    assert model.layout_id == LAYOUT_ID
    assert not np.array_equal(corrected[:6], block2[:6])
    assert np.array_equal(corrected[6:], block2[6:])


def test_fit_is_deterministic_and_reports_nonexact_reference_status() -> None:
    first = _fit_eligible()
    second = _fit_eligible()
    assert first.model is not None and second.model is not None

    assert np.array_equal(first.model.correction_matrix, second.model.correction_matrix)
    assert np.array_equal(first.model.unmixing_matrix, second.model.unmixing_matrix)
    assert first.diagnostics["official_source_commit"] == OFFICIAL_SOURCE_COMMIT
    assert first.diagnostics["reference_equivalence_status"] == REFERENCE_EQUIVALENCE_STATUS
    assert "not_numerically_cross_validated" in REFERENCE_EQUIVALENCE_STATUS
    assert first.diagnostics["two_stage_order"] == "horizontal_vertical_then_blink"


def test_fit_api_cannot_receive_query_or_trial_outcomes() -> None:
    fit_parameters = set(inspect.signature(fit_native_sgeyesub).parameters)
    apply_parameters = set(inspect.signature(_fit_eligible().model.apply).parameters)  # type: ignore[union-attr]

    assert fit_parameters == {
        "block1_data",
        "artifactclasses",
        "channel_labels",
        "channel_types",
        "layout_id",
        "block_id",
        "config",
    }
    assert not {"query", "block2_data", "trial_labels", "outcomes"} & fit_parameters
    assert apply_parameters == {
        "data",
        "channel_labels",
        "channel_types",
        "layout_id",
        "block_id",
    }
    assert not {"artifactclasses", "trial_labels", "outcomes"} & apply_parameters


def test_missing_artifact_class_is_explicitly_ineligible() -> None:
    data, labels = _six_class_calibration()
    labels_without_blinks = np.array(labels, copy=True)
    labels_without_blinks[labels_without_blinks == 5] = 6

    outcome = fit_native_sgeyesub(
        data,
        labels_without_blinks,
        channel_labels=CHANNEL_LABELS,
        channel_types=CHANNEL_TYPES,
        layout_id=LAYOUT_ID,
        block_id=1,
    )

    assert outcome.status == "ineligible"
    assert outcome.model is None
    assert outcome.reasons == ("missing_artifactclasses_5",)


def test_unlabelled_zero_samples_are_ignored_not_rejected() -> None:
    data, labels = _six_class_calibration()
    labels_with_unlabelled = np.array(labels, copy=True)
    labels_with_unlabelled[0] = 0

    outcome = fit_native_sgeyesub(
        data,
        labels_with_unlabelled,
        channel_labels=CHANNEL_LABELS,
        channel_types=CHANNEL_TYPES,
        layout_id=LAYOUT_ID,
        block_id=1,
    )

    assert outcome.status == "eligible", outcome.reasons
    assert outcome.diagnostics["class_counts"][0] == 1


def test_singular_rest_covariance_is_explicitly_ineligible() -> None:
    data, labels = _six_class_calibration()
    data[:6, labels == 6] = 0.0

    outcome = fit_native_sgeyesub(
        data,
        labels,
        channel_labels=CHANNEL_LABELS,
        channel_types=CHANNEL_TYPES,
        layout_id=LAYOUT_ID,
        block_id=1,
    )

    assert outcome.status == "ineligible"
    assert outcome.model is None
    assert outcome.reasons == ("singular_covariance_rest",)


def test_official_alpha_beta_and_optimizer_controls_are_fixed() -> None:
    config = NativeSGEyeSubConfig()
    assert config.alpha == config.plr_lambda_l2 == OFFICIAL_ALPHA == 1.0
    assert config.beta == config.plr_lambda_l1 == OFFICIAL_BETA == 0.01
    assert config.tolerance == 1.0e-3
    assert config.maximum_iterations == 10_000

    with pytest.raises(ValueError, match="alpha"):
        NativeSGEyeSubConfig(alpha=0.5)
    with pytest.raises(ValueError, match="beta"):
        NativeSGEyeSubConfig(beta=0.02)
    with pytest.raises(ValueError, match="tolerance"):
        NativeSGEyeSubConfig(tolerance=1.0e-4)
    with pytest.raises(ValueError, match="maximum_iterations"):
        NativeSGEyeSubConfig(maximum_iterations=100)


def test_release_internal_block_roles_and_layout_are_enforced() -> None:
    data, labels = _six_class_calibration()
    with pytest.raises(ValueError, match="block 1"):
        fit_native_sgeyesub(
            data,
            labels,
            channel_labels=CHANNEL_LABELS,
            channel_types=CHANNEL_TYPES,
            layout_id=LAYOUT_ID,
            block_id=2,  # type: ignore[arg-type]
        )

    model = _fit_eligible().model
    assert model is not None
    with pytest.raises(ValueError, match="block 2"):
        model.apply(
            data,
            channel_labels=CHANNEL_LABELS,
            channel_types=CHANNEL_TYPES,
            layout_id=LAYOUT_ID,
            block_id=1,  # type: ignore[arg-type]
        )
    wrong_layout = list(CHANNEL_TYPES)
    wrong_layout[0] = "EOG"
    with pytest.raises(ValueError, match="layout"):
        model.apply(
            data,
            channel_labels=CHANNEL_LABELS,
            channel_types=wrong_layout,
            layout_id=LAYOUT_ID,
            block_id=2,
        )
    reordered_labels = list(CHANNEL_LABELS)
    reordered_labels[0], reordered_labels[1] = reordered_labels[1], reordered_labels[0]
    with pytest.raises(ValueError, match="layout"):
        model.apply(
            data,
            channel_labels=reordered_labels,
            channel_types=CHANNEL_TYPES,
            layout_id=LAYOUT_ID,
            block_id=2,
        )
    with pytest.raises(ValueError, match="layout"):
        model.apply(
            data,
            channel_labels=CHANNEL_LABELS,
            channel_types=CHANNEL_TYPES,
            layout_id="other_layout",
            block_id=2,
        )
