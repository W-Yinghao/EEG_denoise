"""Ground-truth denoising metrics (EEGdenoiseNet, Zhang et al. 2021, eqs 7-9).

All operate on single-channel epoch arrays ``(N, L)`` (denoised vs clean) and return per-epoch
values; :func:`benchmark_by_snr` aggregates them per SNR level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x ** 2, axis=-1))


def rrmse_temporal(denoised: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """RRMSE_temporal = RMS(f(y) − x) / RMS(x), per epoch (eq. 7)."""
    return _rms(denoised - clean) / (_rms(clean) + 1e-12)


def _psd(x: np.ndarray, fs: int, max_hz: float) -> np.ndarray:
    """Power spectral density up to ``max_hz``, per epoch ``(N, F)``."""
    spec = np.fft.rfft(x, axis=-1)
    power = (np.abs(spec) ** 2) / x.shape[-1]
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs)
    return power[:, freqs <= max_hz]


def rrmse_spectral(denoised: np.ndarray, clean: np.ndarray, fs: int, max_hz: float = 120.0) -> np.ndarray:
    """RRMSE_spectral = RMS(PSD(f(y)) − PSD(x)) / RMS(PSD(x)), per epoch (eq. 8)."""
    pd = _psd(denoised, fs, max_hz)
    pc = _psd(clean, fs, max_hz)
    return _rms(pd - pc) / (_rms(pc) + 1e-12)


def correlation_coefficient(denoised: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """CC = Cov(f(y), x) / sqrt(Var(f(y)) Var(x)), per epoch (eq. 9)."""
    d = denoised - denoised.mean(axis=-1, keepdims=True)
    c = clean - clean.mean(axis=-1, keepdims=True)
    cov = np.mean(d * c, axis=-1)
    denom = np.sqrt(d.var(axis=-1) * c.var(axis=-1)) + 1e-12
    return cov / denom


@dataclass
class BenchmarkResult:
    snr_levels_db: List[float]
    rrmse_t: List[float]   # mean per SNR
    rrmse_s: List[float]
    cc: List[float]
    overall: Dict[str, float]


def benchmark_by_snr(
    denoised: np.ndarray,
    clean: np.ndarray,
    test_snr_db: np.ndarray,
    snr_levels_db: np.ndarray,
    fs: int,
    max_hz: float = 120.0,
) -> BenchmarkResult:
    """Mean RRMSE_t, RRMSE_s, CC per SNR level + overall means."""
    rt = rrmse_temporal(denoised, clean)
    rs = rrmse_spectral(denoised, clean, fs, max_hz)
    cc = correlation_coefficient(denoised, clean)
    per_t, per_s, per_c = [], [], []
    for db in snr_levels_db:
        m = test_snr_db == db
        per_t.append(float(rt[m].mean()))
        per_s.append(float(rs[m].mean()))
        per_c.append(float(cc[m].mean()))
    return BenchmarkResult(
        snr_levels_db=[float(x) for x in snr_levels_db],
        rrmse_t=per_t, rrmse_s=per_s, cc=per_c,
        overall={"rrmse_t": float(rt.mean()), "rrmse_s": float(rs.mean()), "cc": float(cc.mean())},
    )
