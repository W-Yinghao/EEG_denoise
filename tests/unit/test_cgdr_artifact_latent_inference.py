"""Population/subject delta-mixing and stochastic short-circuit contracts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from eeg_cgdr.models.artifact_latent_inference import (
    ArtifactInferenceContext,
    deterministic_population_subject_restore,
    diffusion_population_subject_restore,
)


def _context(role: str, context_id: str, *, swapped: bool = False):
    normalized = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]], dtype=torch.float64
    )
    basis = torch.linalg.qr(normalized).Q[:, :2]
    if swapped:
        normalized = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0], [0.25, 0.5]], dtype=torch.float64
        )
        basis = torch.linalg.qr(normalized).Q[:, :2]
    scale = torch.tensor([2.0, 3.0], dtype=torch.float64)
    return ArtifactInferenceContext(
        context_id=context_id,
        role=role,  # type: ignore[arg-type]
        full_transfer=normalized * scale[None, :],
        normalized_transfer=normalized,
        transfer_scale=scale,
        singular_values=torch.tensor([4.0, 2.0], dtype=torch.float64),
        rank=2,
        calibration_duration_seconds=30.0,
        latent_mean=torch.tensor([0.1, -0.2], dtype=torch.float64),
        latent_standard_deviation=torch.tensor([0.5, 0.25], dtype=torch.float64),
        subspace_basis=basis,
    )


class _DeterministicSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, observed: torch.Tensor, **kwargs):
        self.calls.append({"observed": observed, **kwargs})
        marker = float(torch.as_tensor(kwargs["normalized_transfer"])[0, 0])
        return torch.full(
            (observed.shape[0], 2, observed.shape[-1]),
            1.0 + marker,
            dtype=observed.dtype,
            device=observed.device,
        )


class _DiffusionSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def posterior_mean(self, **kwargs):
        self.calls.append(kwargs)
        observed = kwargs["observed"]
        normalized = torch.as_tensor(
            kwargs["normalized_transfer"],
            dtype=observed.dtype,
            device=observed.device,
        )
        marker = 1.0 + float(normalized[0, 0])
        standardized = torch.full(
            (observed.shape[0], 2, observed.shape[-1]),
            marker,
            dtype=observed.dtype,
            device=observed.device,
        )
        mean = torch.as_tensor(
            kwargs["latent_mean"], dtype=observed.dtype, device=observed.device
        )
        standard_deviation = torch.as_tensor(
            kwargs["latent_standard_deviation"],
            dtype=observed.dtype,
            device=observed.device,
        )
        latent = standardized * standard_deviation[None, :, None] + mean[
            None, :, None
        ]
        delta = torch.einsum("ce,bet->bct", normalized, latent)
        return SimpleNamespace(
            standardized_latent_mean=standardized,
            standardized_latent_standard_deviation=torch.zeros_like(standardized),
            correction=delta,
            sample_count=8,
            network_calls=800,
        )


def _inputs():
    observed = torch.linspace(-1.0, 1.0, 2 * 3 * 8, dtype=torch.float64).reshape(
        2, 3, 8
    )
    channels = torch.ones(3, dtype=torch.bool)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[:, -1] = False
    return observed, channels, valid


def test_deterministic_rho_zero_never_constructs_or_calls_subject() -> None:
    observed, channels, valid = _inputs()
    model = _DeterministicSpy()
    factory_calls = 0

    def forbidden_subject():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("rho=0 constructed a subject context")

    result = deterministic_population_subject_restore(
        model,
        observed,
        population_context=_context("population", "pop"),
        rho=0.0,
        subject_context_factory=forbidden_subject,
        channel_mask=channels,
        valid_time_mask=valid,
    )

    assert factory_calls == 0
    assert len(model.calls) == 1
    assert model.calls[0]["observed"] is observed
    assert result.branch == "population"
    assert result.subject is None
    assert not result.subject_context_constructed
    assert result.complement_relative_error < 1.0e-12
    pure_q = torch.eye(3, dtype=observed.dtype) - result.geometry_projector
    torch.testing.assert_close(
        torch.einsum("cd,bdt->bct", pure_q, result.mixed_delta),
        torch.zeros_like(result.mixed_delta),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_deterministic_nonzero_rho_mixes_mapped_deltas_not_projectors() -> None:
    observed, channels, valid = _inputs()
    model = _DeterministicSpy()
    result = deterministic_population_subject_restore(
        model,
        observed,
        population_context=_context("population", "pop"),
        rho=0.25,
        subject_context_factory=lambda: _context("subject", "subject", swapped=True),
        channel_mask=channels,
        valid_time_mask=valid,
    )

    assert len(model.calls) == 2
    assert model.calls[0]["observed"] is model.calls[1]["observed"] is observed
    assert model.calls[0]["rho"] == model.calls[1]["rho"] == 0.25
    assert result.subject is not None
    expected = 0.75 * result.population.mapped_delta + 0.25 * result.subject.mapped_delta
    torch.testing.assert_close(result.mixed_delta, expected)
    output_mask = valid[:, None, :].to(dtype=observed.dtype)
    torch.testing.assert_close(
        result.restored,
        (observed * output_mask - expected) * output_mask,
    )
    assert result.branch == "mixed"
    assert result.shared_latent_normalization
    assert result.complement_relative_error < 1.0e-12
    union_q = torch.eye(3, dtype=observed.dtype) - result.geometry_projector
    torch.testing.assert_close(
        torch.einsum("cd,bdt->bct", union_q, result.mixed_delta),
        torch.zeros_like(result.mixed_delta),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_diffusion_rho_zero_consumes_only_population_seed_stream() -> None:
    observed, channels, valid = _inputs()
    model = _DiffusionSpy()
    factory_calls = 0

    def forbidden_subject():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("rho=0 constructed a subject context")

    seeds = tuple(range(80, 88))
    result = diffusion_population_subject_restore(
        model,
        observed,
        population_context=_context("population", "pop"),
        rho=0.0,
        subject_context_factory=forbidden_subject,
        channel_mask=channels,
        valid_time_mask=valid,
        sample_seeds=seeds,
        ddim_steps=100,
    )

    assert factory_calls == 0
    assert len(model.calls) == 1
    assert model.calls[0]["sample_seeds"] == seeds
    assert result.population.sampler_calls == 1
    assert result.subject is None
    assert result.shared_diffusion_seeds
    assert result.mixed_delta[..., -1].count_nonzero() == 0


def test_diffusion_nonzero_rho_reuses_exact_k8_stream_for_both_branches() -> None:
    observed, channels, valid = _inputs()
    model = _DiffusionSpy()
    seeds = tuple(range(800, 808))
    result = diffusion_population_subject_restore(
        model,
        observed,
        population_context=_context("population", "pop"),
        rho=0.6,
        subject_context_factory=lambda: _context("subject", "subject", swapped=True),
        channel_mask=channels,
        valid_time_mask=valid,
        sample_seeds=seeds,
        ddim_steps=100,
        record_trajectory=True,
    )

    assert len(model.calls) == 2
    assert model.calls[0]["sample_seeds"] is model.calls[1]["sample_seeds"]
    assert model.calls[0]["sample_seeds"] == seeds
    assert model.calls[0]["rho"] == model.calls[1]["rho"] == 0.6
    assert model.calls[0]["observed"] is model.calls[1]["observed"] is observed
    assert result.subject is not None
    expected = 0.4 * result.population.mapped_delta + 0.6 * result.subject.mapped_delta
    torch.testing.assert_close(result.mixed_delta, expected)
    assert result.shared_model_weights
    assert result.shared_latent_normalization
    assert result.shared_diffusion_seeds
    assert result.complement_relative_error < 1.0e-12


def test_context_normalization_mismatch_is_rejected_before_subject_sampling() -> None:
    observed, channels, valid = _inputs()
    model = _DiffusionSpy()
    subject = _context("subject", "subject")
    object.__setattr__(
        subject,
        "latent_mean",
        torch.tensor([9.0, 9.0], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="latent normalization differs"):
        diffusion_population_subject_restore(
            model,
            observed,
            population_context=_context("population", "pop"),
            rho=0.5,
            subject_context_factory=lambda: subject,
            channel_mask=channels,
            valid_time_mask=valid,
            sample_seeds=tuple(range(8)),
            ddim_steps=10,
        )
    assert len(model.calls) == 1


def test_public_inference_api_has_no_query_eog_label_or_outcome_escape_hatch() -> None:
    forbidden = {
        "eog",
        "query_eog",
        "eye_tracking",
        "query_eye_tracking",
        "label",
        "query_label",
        "outcome",
        "query_outcome",
        "clean_target",
        "query_clean_target",
    }
    for function in (
        deterministic_population_subject_restore,
        diffusion_population_subject_restore,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)
    assert forbidden.isdisjoint(ArtifactInferenceContext.__dataclass_fields__)
