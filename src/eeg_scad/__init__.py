"""Authoritative clean-room SCAD V22 package."""

from .models.scad_artifact_diffusion import SCADArtifactDiffusion, SCADConfig
from .models.deterministic_artifact_unet import DeterministicArtifactEstimator

__all__ = ["SCADArtifactDiffusion", "SCADConfig", "DeterministicArtifactEstimator"]

