"""Static contracts for the deferred B6 POP-SHRINK operator."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from eeg_cgdr.operators.pop_shrink import (
    ProjectorCompatibilityKey,
    spectral_projector_shrink,
)


DATASET = "klados_bamidis_v4"
MONTAGE = "klados_v4_19ch_native_order_256hz"


def _compatibility(channels: int) -> ProjectorCompatibilityKey:
    return ProjectorCompatibilityKey(
        dataset_id=DATASET,
        montage_id=MONTAGE,
        reference_id="native_reference",
        preprocessing_id="mechanism_audit_v1",
        channel_order=tuple(f"EEG{index:02d}" for index in range(channels)),
    )


COMPATIBILITY = _compatibility(4)


def _projector(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64))
    return q @ q.T


def _call(
    pi0: np.ndarray,
    pic: np.ndarray | None,
    *,
    rank: int,
    gamma: float,
    context_eligible: bool = True,
    minimum_eigengap: float = 1.0e-6,
):
    compatibility = _compatibility(pi0.shape[0])
    return spectral_projector_shrink(
        pi0,
        pic,
        rank=rank,
        gamma=gamma,
        context_eligible=context_eligible,
        population_compatibility=compatibility,
        population_fit_scope="outer_training_only",
        context_compatibility=compatibility,
        context_fit_scope="support_only",
        minimum_eigengap=minimum_eigengap,
    )


def test_pop_shrink_endpoints_return_exact_input_projectors() -> None:
    pi0 = _projector(np.eye(5)[:, :2]).astype(np.float32)
    pic = _projector(np.eye(5)[:, 2:4]).astype(np.float32)

    population_endpoint = _call(pi0, pic, rank=2, gamma=0.0)
    context_endpoint = _call(pi0, pic, rank=2, gamma=1.0)

    assert population_endpoint.status == "eligible"
    assert context_endpoint.status == "eligible"
    assert np.array_equal(population_endpoint.projector, pi0)
    assert np.array_equal(context_endpoint.projector, pic)
    assert population_endpoint.projector.dtype == pi0.dtype
    assert context_endpoint.projector.dtype == pic.dtype
    assert population_endpoint.diagnostics["endpoint"] == "Pi0"
    assert context_endpoint.diagnostics["endpoint"] == "PiC"


def test_pop_shrink_middle_is_symmetric_idempotent_and_fixed_rank() -> None:
    angle = 0.35
    pi0 = _projector(np.eye(6)[:, :2])
    context_basis = np.stack(
        [
            np.cos(angle) * np.eye(6)[:, 0] + np.sin(angle) * np.eye(6)[:, 2],
            np.cos(2 * angle) * np.eye(6)[:, 1]
            + np.sin(2 * angle) * np.eye(6)[:, 3],
        ],
        axis=1,
    )
    pic = _projector(context_basis)

    result = _call(pi0, pic, rank=2, gamma=0.25)

    assert result.status == "eligible"
    assert result.projector is not None
    assert np.allclose(result.projector, result.projector.T, atol=1.0e-12)
    assert np.allclose(
        result.projector @ result.projector,
        result.projector,
        atol=1.0e-12,
    )
    assert np.linalg.matrix_rank(result.projector, tol=1.0e-8) == 2
    assert result.diagnostics["spectral_eigengap"] > 0.0


def test_pop_shrink_is_invariant_to_basis_reparameterization() -> None:
    rng = np.random.default_rng(20260802)
    population_basis, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    context_basis, _ = np.linalg.qr(rng.normal(size=(8, 2)))
    within_subspace_rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))

    first = _call(
        _projector(population_basis),
        _projector(context_basis),
        rank=2,
        gamma=0.25,
    )
    reparameterized = _call(
        _projector(population_basis @ within_subspace_rotation),
        _projector(context_basis @ within_subspace_rotation.T),
        rank=2,
        gamma=0.25,
    )

    assert first.status == reparameterized.status == "eligible"
    assert np.allclose(first.projector, reparameterized.projector, atol=1.0e-10)


def test_pop_shrink_is_equivariant_under_eeg_coordinate_rotation() -> None:
    rng = np.random.default_rng(71)
    coordinate_rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    pi0 = _projector(np.eye(7)[:, :2])
    pic = _projector(
        np.stack(
            [
                0.9 * np.eye(7)[:, 0] + 0.435889894 * np.eye(7)[:, 3],
                0.8 * np.eye(7)[:, 1] + 0.6 * np.eye(7)[:, 4],
            ],
            axis=1,
        )
    )

    original = _call(pi0, pic, rank=2, gamma=0.75)
    rotated = _call(
        coordinate_rotation @ pi0 @ coordinate_rotation.T,
        coordinate_rotation @ pic @ coordinate_rotation.T,
        rank=2,
        gamma=0.75,
    )

    assert original.status == rotated.status == "eligible"
    assert np.allclose(
        rotated.projector,
        coordinate_rotation @ original.projector @ coordinate_rotation.T,
        atol=1.0e-10,
    )


def test_pop_shrink_ambiguous_top_subspace_falls_back_to_pop() -> None:
    pi0 = np.diag([1.0, 0.0, 0.0])
    pic = np.diag([0.0, 1.0, 0.0])

    result = _call(
        pi0,
        pic,
        rank=1,
        gamma=0.5,
        minimum_eigengap=1.0e-5,
    )

    assert result.status == "fallback_POP"
    assert result.projector is None
    assert result.reasons == ("spectral_eigengap",)
    assert result.fallback == "POP"
    assert result.diagnostics["spectral_eigengap"] == pytest.approx(0.0)
    assert not result.context_projector_constructed


class _ContextMustNotBeRead:
    def __array__(self, *args, **kwargs):
        raise AssertionError("an ineligible P0 context was inspected")


def test_ineligible_context_short_circuits_without_reading_or_constructing_it() -> None:
    pi0 = np.diag([1.0, 1.0, 0.0, 0.0])

    result = spectral_projector_shrink(
        pi0,
        _ContextMustNotBeRead(),  # type: ignore[arg-type]
        rank=2,
        gamma=0.25,
        context_eligible=False,
        population_compatibility=COMPATIBILITY,
        population_fit_scope="outer_training_only",
        context_compatibility=None,
        context_fit_scope=None,
    )

    assert result.status == "fallback_POP"
    assert result.projector is None
    assert result.reasons == ("context_ineligible",)
    assert not result.context_projector_constructed


@pytest.mark.parametrize(
    ("context", "reason"),
    [
        (np.eye(3), "context_projector_shape"),
        (
            np.array(
                [
                    [1.0, 0.2, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "context_projector_not_symmetric",
        ),
        (np.diag([0.5, 0.5, 0.0, 0.0]), "context_projector_not_idempotent"),
        (np.diag([1.0, 0.0, 0.0, 0.0]), "context_projector_rank"),
    ],
)
def test_invalid_context_projector_falls_back_to_pop(
    context: np.ndarray,
    reason: str,
) -> None:
    pi0 = np.diag([1.0, 1.0, 0.0, 0.0])

    result = _call(pi0, context, rank=2, gamma=0.25)

    assert result.status == "fallback_POP"
    assert result.projector is None
    assert result.reasons == (reason,)
    assert not result.context_projector_constructed


def test_pop_endpoint_does_not_read_or_construct_context() -> None:
    pi0 = np.diag([1.0, 1.0, 0.0, 0.0])
    result = spectral_projector_shrink(
        pi0,
        _ContextMustNotBeRead(),  # type: ignore[arg-type]
        rank=2,
        gamma=0.0,
        context_eligible=True,
        population_compatibility=COMPATIBILITY,
        population_fit_scope="outer_training_only",
        context_compatibility=None,
        context_fit_scope=None,
    )

    assert result.status == "eligible"
    assert np.array_equal(result.projector, pi0)
    assert not result.context_projector_constructed


@pytest.mark.parametrize(
    "context_compatibility",
    [
        ProjectorCompatibilityKey(
            dataset_id="other_dataset",
            montage_id=MONTAGE,
            reference_id="native_reference",
            preprocessing_id="mechanism_audit_v1",
            channel_order=("Fp1", "Fp2", "F3", "F4"),
        ),
        ProjectorCompatibilityKey(
            dataset_id=DATASET,
            montage_id="other_montage",
            reference_id="native_reference",
            preprocessing_id="mechanism_audit_v1",
            channel_order=("Fp1", "Fp2", "F3", "F4"),
        ),
        ProjectorCompatibilityKey(
            dataset_id=DATASET,
            montage_id=MONTAGE,
            reference_id="average_reference",
            preprocessing_id="mechanism_audit_v1",
            channel_order=("Fp1", "Fp2", "F3", "F4"),
        ),
        ProjectorCompatibilityKey(
            dataset_id=DATASET,
            montage_id=MONTAGE,
            reference_id="native_reference",
            preprocessing_id="different_preprocessing",
            channel_order=("Fp1", "Fp2", "F3", "F4"),
        ),
        ProjectorCompatibilityKey(
            dataset_id=DATASET,
            montage_id=MONTAGE,
            reference_id="native_reference",
            preprocessing_id="mechanism_audit_v1",
            channel_order=("Fp2", "Fp1", "F3", "F4"),
        ),
    ],
)
def test_compatibility_mismatch_falls_back_without_a_context_projector(
    context_compatibility: ProjectorCompatibilityKey,
) -> None:
    pi0 = np.diag([1.0, 1.0, 0.0, 0.0])
    result = spectral_projector_shrink(
        pi0,
        pi0,
        rank=2,
        gamma=0.25,
        context_eligible=True,
        population_compatibility=COMPATIBILITY,
        population_fit_scope="outer_training_only",
        context_compatibility=context_compatibility,
        context_fit_scope="support_only",
    )

    assert result.status == "fallback_POP"
    assert result.projector is None
    assert result.reasons == ("compatibility_mismatch",)


def test_fit_scope_contract_rejects_population_or_context_leakage() -> None:
    pi0 = np.diag([1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="outer_training_only"):
        spectral_projector_shrink(
            pi0,
            pi0,
            rank=2,
            gamma=0.25,
            context_eligible=True,
            population_compatibility=COMPATIBILITY,
            population_fit_scope="all_records",  # type: ignore[arg-type]
            context_compatibility=COMPATIBILITY,
            context_fit_scope="support_only",
        )

    context_failure = spectral_projector_shrink(
        pi0,
        pi0,
        rank=2,
        gamma=0.25,
        context_eligible=True,
        population_compatibility=COMPATIBILITY,
        population_fit_scope="outer_training_only",
        context_compatibility=COMPATIBILITY,
        context_fit_scope="support_and_query",  # type: ignore[arg-type]
    )
    assert context_failure.status == "fallback_POP"
    assert context_failure.reasons == ("context_fit_scope",)


def test_population_projector_failure_is_fatal_because_pop_is_unavailable() -> None:
    invalid_pi0 = np.diag([0.5, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="POP population projector is invalid"):
        _call(invalid_pi0, np.diag([1.0, 1.0, 0.0, 0.0]), rank=2, gamma=0.25)


def test_public_signature_has_no_query_or_target_escape_hatch() -> None:
    signature = inspect.signature(spectral_projector_shrink)
    forbidden = {
        "y",
        "query",
        "query_eeg",
        "query_clean",
        "clean",
        "clean_target",
        "eog",
        "events",
    }

    assert forbidden.isdisjoint(signature.parameters)
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    pi0 = np.diag([1.0, 1.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        spectral_projector_shrink(
            pi0,
            pi0,
            rank=2,
            gamma=0.25,
            context_eligible=True,
            population_compatibility=COMPATIBILITY,
            population_fit_scope="outer_training_only",
            context_compatibility=COMPATIBILITY,
            context_fit_scope="support_only",
            query_clean=np.zeros((4, 10)),  # type: ignore[call-arg]
        )
