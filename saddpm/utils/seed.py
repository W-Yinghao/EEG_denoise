"""Global reproducibility seeding (handoff §6).

Seeds Python ``random``, NumPy, and (if installed) PyTorch — including CUDA and cuDNN
determinism — and returns the seed so callers can log it.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all RNGs for reproducibility.

    Args:
        seed: the integer seed to apply everywhere.
        deterministic: if True, request cuDNN deterministic algorithms (slower but reproducible).

    Returns:
        The seed that was applied (for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed
