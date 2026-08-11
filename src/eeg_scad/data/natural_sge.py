"""Natural SGEYESUB development bundles are physically split from evaluator EOG."""
from pathlib import Path
import numpy as np


def load_inference(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:
        assert "eog" not in z.files and "C_query" not in z.files
        return {k:np.asarray(z[k]) for k in z.files}


def load_evaluator(path:Path)->dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

