"""V22 full-field architecture with context-consistent base supervision."""
from eeg_scad.models.scad_artifact_diffusion import SCADArtifactDiffusion,SCADConfig

class V22FixedFullField(SCADArtifactDiffusion):
    wrong_receives_ordinary_base_loss=False

