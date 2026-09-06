from __future__ import annotations

import numpy as np


def canonicalize_operator(operator: np.ndarray, eog_scale: np.ndarray, eeg_scale: np.ndarray | None = None) -> np.ndarray:
    """Map a physical EOG transfer into standardized EOG/optionally EEG coordinates."""
    value=np.asarray(operator,dtype=np.float64)*np.asarray(eog_scale,dtype=np.float64)[None]
    if eeg_scale is not None:value=value/np.asarray(eeg_scale,dtype=np.float64)[:,None]
    return value


def canonical_operator_features(operator: np.ndarray, epsilon: float=1e-8) -> np.ndarray:
    value=np.asarray(operator,dtype=np.float64);norm=max(float(np.linalg.norm(value)),epsilon)
    singular=np.linalg.svd(value,compute_uv=False)
    return np.concatenate(((value/norm).reshape(-1),np.log(singular+epsilon),[np.log(norm+epsilon)])).astype(np.float32)


def robust_scale(values: np.ndarray, epsilon: float=1e-6) -> tuple[np.ndarray,np.ndarray]:
    values=np.asarray(values,dtype=np.float64);center=np.median(values,axis=-1);mad=np.median(np.abs(values-center[:,None]),axis=-1)*1.4826
    return center,np.maximum(mad,epsilon)

