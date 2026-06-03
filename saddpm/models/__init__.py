"""Neural network modules (U-Net, FiLM, subject embedding, dual decoder, ArcFace, EEGNet)."""

from .config import ModelConfig
from .film import FiLM
from .subject_embed import SubjectEmbedding
from .unet1d import UNet1D

__all__ = ["ModelConfig", "UNet1D", "FiLM", "SubjectEmbedding"]
