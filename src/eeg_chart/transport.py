"""Transport building blocks: SH lift, covariance alignment, ocular Procrustes.

All constructions follow docs/prop5_endpoint_correction.md.  T_s = Q_s G_s L_s
maps montage channel space (C_s) into the canonical K-dimensional head space
(K = 121 real spherical-harmonic coefficients, degree <= 10).
"""
from __future__ import annotations

import numpy as np

try:  # scipy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_new
    _HAVE_NEW_SPH = True
except ImportError:  # scipy <= 1.14: sph_harm(m, n, azimuth, polar)
    from scipy.special import sph_harm as _sph_harm_old
    _HAVE_NEW_SPH = False


def _sph_harm(ell: int, m: int, polar: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    if _HAVE_NEW_SPH:
        return _sph_harm_new(ell, m, polar, azimuth)
    return _sph_harm_old(m, ell, azimuth, polar)


SH_DEGREE = 10
K_CANONICAL = (SH_DEGREE + 1) ** 2  # 121


def real_sh_basis(positions: np.ndarray, degree: int = SH_DEGREE) -> np.ndarray:
    """Real spherical-harmonic design matrix Y in R^{C x K} at unit-sphere positions."""
    xyz = np.asarray(positions, np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("positions must be C x 3")
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        raise ValueError("electrode positions must be nonzero")
    xyz = xyz / norms
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))          # polar
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])                    # azimuth
    columns = []
    for ell in range(degree + 1):
        for m in range(-ell, ell + 1):
            harmonic = _sph_harm(ell, abs(m), theta, phi)
            if m < 0:
                columns.append(np.sqrt(2.0) * (-1.0) ** m * harmonic.imag)
            elif m == 0:
                columns.append(harmonic.real)
            else:
                columns.append(np.sqrt(2.0) * (-1.0) ** m * harmonic.real)
    return np.stack(columns, axis=1)


def sh_lift(positions: np.ndarray, ridge: float = 1e-3, degree: int = SH_DEGREE) -> np.ndarray:
    """Perrin-style regularized SH least-squares lift L in R^{K x C}.

    L x are the SH coefficients of channel data x; L has full column rank for
    C <= K, so pinv(L) L = I_C exactly (round-trip contract)."""
    basis = real_sh_basis(positions, degree)
    channels = basis.shape[0]
    if channels > basis.shape[1]:
        raise ValueError("more electrodes than SH coefficients")
    gram = basis.T @ basis + float(ridge) * np.eye(basis.shape[1])
    lift = np.linalg.solve(gram, basis.T)
    if np.linalg.matrix_rank(lift, tol=1e-10) != channels:
        raise ValueError("SH lift lost column rank")
    return lift


def orth(matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis of the column span with a DETERMINISTIC sign convention
    (largest-|entry| coordinate positive per column) so repeated constructions
    are bit-identical."""
    u, s, _ = np.linalg.svd(np.asarray(matrix, np.float64), full_matrices=False)
    rank = int(np.sum(s > tol * max(s[0], 1e-300)))
    basis = u[:, :rank]
    for index in range(rank):
        anchor = np.argmax(np.abs(basis[:, index]))
        if basis[anchor, index] < 0:
            basis[:, index] = -basis[:, index]
    return basis


def ordered_frame(matrix: np.ndarray) -> np.ndarray:
    """Orthonormalize COLUMNS IN ORDER (Gram-Schmidt via QR, diag(R) > 0).

    Ocular frames are physically pinned (VEOG/HEOG[/blink] order); SVD-based
    orth() would reorder columns and destroy the frame correspondence that the
    minimal rotation maps column-to-column."""
    q, r = np.linalg.qr(np.asarray(matrix, np.float64))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def minimal_rotation(frame_from: np.ndarray, frame_to: np.ndarray) -> np.ndarray:
    """Rotation in SO(K) carrying frame_from onto frame_to, identity on the
    orthogonal complement of span(frame_from) + span(frame_to)."""
    a = np.asarray(frame_from, np.float64)
    b = np.asarray(frame_to, np.float64)
    if a.shape != b.shape:
        raise ValueError("frames must have identical shape")
    dim, r = a.shape
    union = orth(np.concatenate((a, b), axis=1))
    m = union.shape[1]
    a_w = union.T @ a
    b_w = union.T @ b
    a_comp = orth(np.eye(m) - a_w @ a_w.T)[:, :m - r] if m > r else np.zeros((m, 0))
    if m > r:
        raw = (np.eye(m) - b_w @ b_w.T) @ a_comp
        b_comp = orth(raw)[:, :m - r]
        if b_comp.shape[1] != m - r:
            raise ValueError("principal angle at pi: minimal rotation undefined")
        # Align the complement pairing so the rotation is the direct (minimal) one.
        u, _, vt = np.linalg.svd(a_comp.T @ b_comp)
        b_comp = b_comp @ (u @ vt).T
    else:
        b_comp = np.zeros((m, 0))
    a_full = np.concatenate((a_w, a_comp), axis=1)
    b_full = np.concatenate((b_w, b_comp), axis=1)
    if np.linalg.det(a_full) * np.linalg.det(b_full) < 0 and m > r:
        b_comp[:, -1] = -b_comp[:, -1]
        b_full = np.concatenate((b_w, b_comp), axis=1)
    rotation_w = b_full @ a_full.T
    rotation = np.eye(dim) + union @ (rotation_w - np.eye(m)) @ union.T
    return rotation


def ledoit_wolf_covariance(samples: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance of K-dim samples (samples: K x T)."""
    from sklearn.covariance import ledoit_wolf

    value = np.asarray(samples, np.float64)
    covariance, _ = ledoit_wolf(value.T, assume_centered=False)
    return covariance


def airm_frechet_mean(covariances: list[np.ndarray], iterations: int = 30,
                      tol: float = 1e-10) -> np.ndarray:
    """Affine-invariant Riemannian Fréchet mean of SPD matrices."""
    mean = np.mean(np.stack(covariances), axis=0)
    for _ in range(iterations):
        root, inv_root = _spd_sqrt(mean)
        logs = np.mean([_spd_log(inv_root @ value @ inv_root) for value in covariances], axis=0)
        step = root @ _spd_exp(logs) @ root
        if np.linalg.norm(step - mean) <= tol * np.linalg.norm(mean):
            mean = step
            break
        mean = step
    return (mean + mean.T) / 2


def _spd_eigh(matrix: np.ndarray):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return np.clip(values, 1e-12, None), vectors


def _spd_sqrt(matrix: np.ndarray):
    values, vectors = _spd_eigh(matrix)
    return (vectors * np.sqrt(values)) @ vectors.T, (vectors / np.sqrt(values)) @ vectors.T


def _spd_log(matrix: np.ndarray) -> np.ndarray:
    values, vectors = _spd_eigh(matrix)
    return (vectors * np.log(values)) @ vectors.T


def _spd_exp(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.exp(values)) @ vectors.T


def spd_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = _spd_eigh(matrix)
    return (vectors * values ** exponent) @ vectors.T


__all__ = ["K_CANONICAL", "SH_DEGREE", "airm_frechet_mean", "ledoit_wolf_covariance",
           "minimal_rotation", "ordered_frame", "orth", "real_sh_basis", "sh_lift",
           "spd_power"]
