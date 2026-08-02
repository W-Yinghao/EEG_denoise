"""Calibration operators used by CGDR."""

from .artifact_context import (
    ArtifactContextCorrection,
    ArtifactTransfer,
    EOGStandardizationMetadata,
    POSTERIOR_SAMPLE_COUNT,
    SupportOnlyRho,
    fit_artifact_transfer,
    fit_eog_standardization,
    freeze_support_only_rho,
    population_subject_mixing_correction,
    posterior_mean_k8,
)
from .p0 import CalibrationBatch, P0Config, P0FitOutcome, P0Transfer, fit_p0
from .pop_shrink import (
    ProjectorCompatibilityKey,
    PopShrinkOutcome,
    spectral_projector_shrink,
)

__all__ = [
    "ArtifactContextCorrection",
    "ArtifactTransfer",
    "CalibrationBatch",
    "EOGStandardizationMetadata",
    "POSTERIOR_SAMPLE_COUNT",
    "P0Config",
    "P0FitOutcome",
    "P0Transfer",
    "ProjectorCompatibilityKey",
    "PopShrinkOutcome",
    "SupportOnlyRho",
    "fit_artifact_transfer",
    "fit_eog_standardization",
    "fit_p0",
    "freeze_support_only_rho",
    "population_subject_mixing_correction",
    "posterior_mean_k8",
    "spectral_projector_shrink",
]
