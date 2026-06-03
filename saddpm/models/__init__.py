"""Neural network modules (U-Net, FiLM, subject embedding, dual decoder, ArcFace, EEGNet)."""

from .arcface import ArcFace
from .config import ModelConfig
from .dual_decoder import DualDecoderSADDPM
from .eegnet import EEGNet, EEGNetConfig
from .film import FiLM
from .subject_embed import SubjectEmbedding
from .unet1d import UNet1D

__all__ = [
    "ModelConfig", "UNet1D", "FiLM", "SubjectEmbedding",
    "DualDecoderSADDPM", "ArcFace", "EEGNet", "EEGNetConfig",
]
