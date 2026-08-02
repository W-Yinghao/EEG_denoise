"""Algebra and leakage boundaries for subject-calibrated artifact context.

These tests are intended for the aggregate CPU Slurm validation job.  They do
not constitute EEG experiment evidence and must not be run on the login node.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eeg_cgdr.operators.artifact_context import (
    POSTERIOR_SAMPLE_COUNT,
    SupportOnlyRho,
    fit_artifact_transfer,
    fit_eog_standardization,
    freeze_support_only_rho,
    population_subject_mixing_correction,
    posterior_mean_k8,
)


EEG_ORDER = ("Fp1", "Fp2", "F3", "F4", "Cz")
EOG_INPUT_ORDER = ("VEOG", "HEOG")
EOG_CANONICAL_ORDER = ("HEOG", "VEOG")
EOG_POLARITY = (-1.0, 1.0)


def _canonical(raw_eog: np.ndarray) -> np.ndarray:
    return np.stack((-raw_eog[1], raw_eog[0]), axis=0)


def _fit(
    normalized_transfer: np.ndarray,
    *,
    fit_scope: str,
    fit_id: str,
    seed: int,
):
    rng = np.random.default_rng(seed)
    raw_eog = rng.normal(size=(2, 257)) * np.asarray([[3.0], [0.7]])
    canonical = _canonical(raw_eog)
    latent = (canonical - canonical.mean(axis=1, keepdims=True)) / canonical.std(
        axis=1,
        keepdims=True,
    )
    eeg = (
        np.asarray(normalized_transfer, dtype=np.float64) @ latent
        + np.arange(len(EEG_ORDER), dtype=np.float64)[:, None]
    )
    return fit_artifact_transfer(
        eeg,
        raw_eog,
        eeg_channel_order=EEG_ORDER,
        eog_input_order=EOG_INPUT_ORDER,
        eog_canonical_order=EOG_CANONICAL_ORDER,
        eog_polarity=EOG_POLARITY,
        ridge_lambda=0.0,
        retained_rank=2,
        fit_scope=fit_scope,  # type: ignore[arg-type]
        fit_id=fit_id,
    )


def test_fp64_ridge_normalization_scale_rank_and_singular_values() -> None:
    rng = np.random.default_rng(20260802)
    samples = 311
    raw_eog = rng.normal(size=(2, samples)) * np.asarray([[2.5], [0.4]])
    canonical = _canonical(raw_eog)
    mean = canonical.mean(axis=1, keepdims=True)
    standard_deviation = canonical.std(axis=1, keepdims=True)
    standardization_scale = 1.0 / standard_deviation
    latent = standardization_scale * (canonical - mean)
    generating_transfer = np.asarray(
        [
            [1.3, -0.1],
            [0.2, 0.9],
            [-0.5, 0.4],
            [0.7, 1.1],
            [-0.2, 0.3],
        ],
        dtype=np.float64,
    )
    eeg = (
        generating_transfer @ latent
        + 0.02 * rng.normal(size=(len(EEG_ORDER), samples))
        + np.arange(len(EEG_ORDER), dtype=np.float64)[:, None]
    )
    ridge = 3.25

    transfer = fit_artifact_transfer(
        eeg,
        raw_eog,
        eeg_channel_order=EEG_ORDER,
        eog_input_order=EOG_INPUT_ORDER,
        eog_canonical_order=EOG_CANONICAL_ORDER,
        eog_polarity=EOG_POLARITY,
        ridge_lambda=ridge,
        retained_rank=2,
        fit_scope="support_only",
        fit_id="subject-support-A",
    )

    y = eeg.astype(np.float64) - eeg.mean(axis=1, keepdims=True)
    z = latent.astype(np.float64)
    gram = z @ z.T + ridge * np.eye(z.shape[0], dtype=np.float64)
    expected_normalized = np.linalg.solve(gram, (y @ z.T).T).T
    expected_full = expected_normalized
    expected_transfer_scale = np.linalg.norm(expected_full, axis=0)
    expected_normalized_columns = expected_full / expected_transfer_scale[None, :]
    expected_basis, expected_singular_values, _ = np.linalg.svd(
        expected_full,
        full_matrices=False,
    )

    assert transfer.transfer_matrix.dtype == np.float64
    assert transfer.raw_transfer_matrix.dtype == np.float64
    assert transfer.transfer_normalized.dtype == np.float64
    assert transfer.rank == 2
    assert transfer.fit_scope == "support_only"
    assert transfer.fit_id == "subject-support-A"
    assert transfer.eog_metadata.canonical_order == EOG_CANONICAL_ORDER
    assert transfer.eog_metadata.polarity == EOG_POLARITY
    np.testing.assert_allclose(
        transfer.eog_metadata.mean,
        mean[:, 0],
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        transfer.eog_metadata.standard_deviation,
        standard_deviation[:, 0],
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        transfer.eog_metadata.standardization_scale,
        standardization_scale[:, 0],
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        transfer.transfer_normalized,
        expected_normalized_columns,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        transfer.raw_transfer_matrix,
        expected_full,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        transfer.transfer_matrix,
        transfer.transfer_normalized @ np.diag(transfer.transfer_scale),
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        transfer.transfer_scale,
        expected_transfer_scale,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        transfer.singular_values,
        expected_singular_values,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        transfer.projector,
        expected_basis[:, : transfer.rank] @ expected_basis[:, : transfer.rank].T,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    source = inspect.getsource(fit_artifact_transfer)
    assert "np.linalg.solve" in source
    assert "np.linalg.inv" not in source


def test_eog_order_polarity_and_support_statistics_are_frozen_for_query() -> None:
    support = np.asarray(
        [[2.0, 4.0, 8.0, 10.0], [1.0, -1.0, 3.0, 5.0]],
        dtype=np.float64,
    )
    metadata = fit_eog_standardization(
        support,
        input_order=EOG_INPUT_ORDER,
        canonical_order=EOG_CANONICAL_ORDER,
        polarity=EOG_POLARITY,
        source_id="support-only-A",
        fit_scope="support_only",
    )
    expected_support = _canonical(support)
    np.testing.assert_array_equal(
        metadata.canonicalize(support, input_order=EOG_INPUT_ORDER),
        expected_support,
    )
    assert metadata.fit_scope == "support_only"
    assert metadata.standardization == "per_channel_population_std_ddof0"

    query = np.asarray(
        [[100.0, 200.0], [-40.0, -80.0]],
        dtype=np.float64,
    )
    expected_query = (
        _canonical(query) - np.asarray(metadata.mean)[:, None]
    ) * np.asarray(metadata.standardization_scale)[:, None]
    np.testing.assert_allclose(
        metadata.standardize(query, input_order=EOG_INPUT_ORDER),
        expected_query,
        rtol=0.0,
        atol=0.0,
    )
    assert tuple(metadata.mean) != tuple(_canonical(query).mean(axis=1))


def test_normalized_latent_and_full_transfer_predictions_are_identical() -> None:
    normalized = np.asarray(
        [[1.0, 0.0], [0.0, 0.7], [0.3, 0.2], [-0.2, 0.8], [0.4, -0.3]]
    )
    transfer = _fit(
        normalized,
        fit_scope="support_only",
        fit_id="support-A",
        seed=12,
    )
    query_eog = np.asarray(
        [[4.0, -1.0, 3.5], [0.5, -2.0, 1.0]],
        dtype=np.float64,
    )
    standardized_eog = transfer.standardized_eog(
        query_eog,
        input_order=EOG_INPUT_ORDER,
    )
    latent = transfer.standardized_artifact_latent(
        query_eog,
        input_order=EOG_INPUT_ORDER,
    )
    np.testing.assert_allclose(
        latent,
        np.diag(transfer.transfer_scale) @ standardized_eog,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        transfer.predict_contamination(query_eog, input_order=EOG_INPUT_ORDER),
        transfer.predict_contamination_from_full_transfer(
            query_eog,
            input_order=EOG_INPUT_ORDER,
        ),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    observed = np.arange(15, dtype=np.float64).reshape(5, 3)
    np.testing.assert_allclose(
        transfer.reconstruct(observed, query_eog, input_order=EOG_INPUT_ORDER),
        observed
        - transfer.transfer_matrix @ standardized_eog,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_three_eog_columns_are_preserved_when_effective_rank_is_two() -> None:
    rng = np.random.default_rng(73)
    samples = 401
    order = ("VEOG", "HEOG", "REOG")
    polarity = (1.0, -1.0, 1.0)
    raw_eog = rng.normal(size=(3, samples))
    canonical = raw_eog * np.asarray(polarity)[:, None]
    standardized = (
        canonical - canonical.mean(axis=1, keepdims=True)
    ) / canonical.std(axis=1, keepdims=True)
    # The raw transfer has rank three; development retains rank two without
    # deleting the third EOG coordinate from the runtime matrix.
    full_transfer = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, -0.2, 0.9],
            [-0.4, 0.7, -0.1],
            [0.2, 0.1, 0.6],
        ],
        dtype=np.float64,
    )
    eeg = full_transfer @ standardized
    transfer = fit_artifact_transfer(
        eeg,
        raw_eog,
        eeg_channel_order=EEG_ORDER,
        eog_input_order=order,
        eog_canonical_order=order,
        eog_polarity=polarity,
        ridge_lambda=0.0,
        retained_rank=2,
        fit_scope="support_only",
        fit_id="support-three-eog",
    )

    assert transfer.transfer_matrix.shape == (len(EEG_ORDER), 3)
    assert transfer.raw_transfer_matrix.shape == (len(EEG_ORDER), 3)
    assert transfer.transfer_normalized.shape == (len(EEG_ORDER), 3)
    assert transfer.transfer_scale.shape == (3,)
    assert transfer.singular_values.shape == (3,)
    assert transfer.rank == 2
    assert transfer.numerical_rank == 3
    assert transfer.eeg_subspace_basis.shape == (len(EEG_ORDER), 2)
    assert np.linalg.matrix_rank(transfer.transfer_matrix) == 2
    assert not np.allclose(transfer.transfer_matrix[:, 2], 0.0)
    assert not np.allclose(transfer.transfer_matrix, transfer.raw_transfer_matrix)
    np.testing.assert_allclose(
        transfer.raw_transfer_matrix,
        full_transfer,
        rtol=5.0e-13,
        atol=5.0e-13,
    )
    np.testing.assert_allclose(
        transfer.transfer_matrix,
        transfer.transfer_normalized @ np.diag(transfer.transfer_scale),
        rtol=5.0e-13,
        atol=5.0e-13,
    )


def _population_and_subject():
    population = _fit(
        np.asarray(
            [[1.0, 0.0], [0.4, 0.0], [0.0, 0.8], [0.0, 0.2], [0.1, 0.3]]
        ),
        fit_scope="outer_training_only",
        fit_id="population-cell-A",
        seed=30,
    )
    subject = _fit(
        np.asarray(
            [[0.0, 0.1], [0.0, 0.9], [0.7, 0.0], [0.5, 0.2], [0.0, 0.4]]
        ),
        fit_scope="support_only",
        fit_id="support-A",
        seed=31,
    )
    return population, subject


def test_rho_zero_short_circuits_subject_factory_and_obeys_population_q() -> None:
    population, _ = _population_and_subject()
    raw_eog = np.arange(34, dtype=np.float64).reshape(2, 17) / 10.0
    observed = np.ones((len(EEG_ORDER), 17), dtype=np.float64)
    calls = 0

    def forbidden_subject():
        nonlocal calls
        calls += 1
        raise AssertionError("rho=0 constructed a subject context")

    result = population_subject_mixing_correction(
        observed,
        raw_eog,
        eog_input_order=EOG_INPUT_ORDER,
        population_transfer=population,
        rho=freeze_support_only_rho(
            0.0,
            support_id="support-A",
            selection_source="support_only_diagnostics",
        ),
        subject_transfer_factory=forbidden_subject,
    )

    assert calls == 0
    assert not result.subject_context_constructed
    assert result.subject_contamination is None
    assert result.branch == "population"
    q0 = np.eye(len(EEG_ORDER)) - population.projector
    np.testing.assert_allclose(q0 @ result.correction, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(result.restored_eeg - observed, result.correction)


def test_subject_endpoint_and_mixed_correction_have_registered_geometry() -> None:
    population, subject = _population_and_subject()
    raw_eog = np.arange(38, dtype=np.float64).reshape(2, 19) / 7.0
    observed = np.linspace(-1.0, 1.0, len(EEG_ORDER) * 19).reshape(
        len(EEG_ORDER),
        19,
    )
    calls = 0

    def subject_factory():
        nonlocal calls
        calls += 1
        return subject

    subject_result = population_subject_mixing_correction(
        observed,
        raw_eog,
        eog_input_order=EOG_INPUT_ORDER,
        population_transfer=population,
        rho=freeze_support_only_rho(
            1.0,
            support_id="support-A",
            selection_source="pre_frozen_development",
        ),
        subject_transfer_factory=subject_factory,
    )
    assert calls == 1
    assert subject_result.branch == "subject"
    qs = np.eye(len(EEG_ORDER)) - subject.projector
    np.testing.assert_allclose(qs @ subject_result.correction, 0.0, atol=2.0e-12)

    mixed_result = population_subject_mixing_correction(
        observed,
        raw_eog,
        eog_input_order=EOG_INPUT_ORDER,
        population_transfer=population,
        rho=freeze_support_only_rho(
            0.35,
            support_id="support-A",
            selection_source="support_only_diagnostics",
        ),
        subject_transfer_factory=subject_factory,
    )
    assert calls == 2
    assert mixed_result.branch == "mixed"
    assert mixed_result.subject_contamination is not None
    np.testing.assert_allclose(
        mixed_result.mixed_contamination,
        0.65 * mixed_result.population_contamination
        + 0.35 * mixed_result.subject_contamination,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    union_q = np.eye(len(EEG_ORDER)) - mixed_result.union_projector
    np.testing.assert_allclose(union_q @ mixed_result.correction, 0.0, atol=2.0e-12)
    concatenated = np.concatenate(
        [
            population.transfer_matrix,
            subject.transfer_matrix,
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        union_q @ concatenated,
        0.0,
        atol=2.0e-12,
    )


def test_rho_and_transfer_fit_interfaces_reject_leakage_escape_hatches() -> None:
    forbidden = {
        "query",
        "query_eeg",
        "query_eog",
        "query_target",
        "clean",
        "clean_target",
        "outcome",
        "metric",
        "score",
    }
    assert forbidden.isdisjoint(inspect.signature(fit_artifact_transfer).parameters)
    assert forbidden.isdisjoint(inspect.signature(freeze_support_only_rho).parameters)
    with pytest.raises(ValueError, match="support_only"):
        SupportOnlyRho(
            value=0.5,
            support_id="support-A",
            selection_source="support_only_diagnostics",
            fit_scope="support_and_query",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        freeze_support_only_rho(
            0.5,
            support_id="support-A",
            selection_source="support_only_diagnostics",
            query_metric=1.0,  # type: ignore[call-arg]
        )


def test_k8_posterior_waveform_mean_forbids_best_of_k() -> None:
    samples = [
        np.full((3, 5), float(index), dtype=np.float32)
        for index in range(POSTERIOR_SAMPLE_COUNT)
    ]
    mean = posterior_mean_k8(samples)
    assert mean.dtype == np.float64
    np.testing.assert_array_equal(mean, np.full((3, 5), 3.5, dtype=np.float64))

    with pytest.raises(ValueError, match="exactly K=8"):
        posterior_mean_k8(samples[:-1])
    with pytest.raises(ValueError, match="best-of-K"):
        posterior_mean_k8(
            samples,
            output_rule="best_of_k",  # type: ignore[arg-type]
        )
    selection_forbidden = {
        "target",
        "clean_target",
        "outcome",
        "metric",
        "score",
    }
    assert selection_forbidden.isdisjoint(
        inspect.signature(posterior_mean_k8).parameters
    )
