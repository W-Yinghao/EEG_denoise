"""Coordinate-stability diagnostics used by the V26 forensic audit."""
from __future__ import annotations

import numpy as np


def rotate_basis(basis: np.ndarray, coefficient: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply an equivalent orthogonal change of latent coordinates."""
    return basis @ rotation, rotation.T @ coefficient


def projector_distance(left: np.ndarray, right: np.ndarray) -> float:
    ql = np.linalg.qr(left)[0]
    qr = np.linalg.qr(right)[0]
    return float(np.linalg.norm(ql @ ql.T - qr @ qr.T, ord="fro"))


def procrustes_align(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align candidate columns to reference and return aligned basis and rotation."""
    u, _, vt = np.linalg.svd(candidate.T @ reference, full_matrices=False)
    rotation = u @ vt
    return candidate @ rotation, rotation


def rotation_fixture(seed: int = 20260828) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(46, 8))
    basis, _ = np.linalg.qr(basis)
    coefficient = rng.normal(size=(8, 256))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    changed_basis, changed_coefficient = rotate_basis(basis, coefficient, rotation)
    artifact = basis @ coefficient
    changed_artifact = changed_basis @ changed_coefficient
    return {
        "sensor_artifact_max_difference": float(np.max(np.abs(artifact - changed_artifact))),
        "latent_target_relative_difference": float(np.linalg.norm(coefficient - changed_coefficient) / np.linalg.norm(coefficient)),
        "projector_distance": projector_distance(basis, changed_basis),
        "rotation_magnitude": float(np.linalg.norm(rotation - np.eye(8), ord="fro")),
    }


__all__ = ["rotate_basis", "projector_distance", "procrustes_align", "rotation_fixture"]
