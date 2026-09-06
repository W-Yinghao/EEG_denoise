"""Mandatory Prop-5' geometry gates as unit tests (U0-a contract)."""
from __future__ import annotations

import numpy as np
import pytest

from eeg_chart.analytic import canonical_clean, gauge_null_rotation
from eeg_chart.geodesic import (ANGLE_CAP, max_principal_angle, rho_eb, rotation_distance,
                                rotation_geodesic, rotation_log, transport_family)
from eeg_chart.transport import (K_CANONICAL, airm_frechet_mean, minimal_rotation,
                                 ordered_frame, orth, real_sh_basis, sh_lift, spd_power)


R_OCULAR = 2


@pytest.fixture(scope="module")
def geometry():
    rng = np.random.default_rng(20260814)
    positions = rng.standard_normal((19, 3))
    positions[:, 2] = np.abs(positions[:, 2])          # upper hemisphere, cap-like
    lift = sh_lift(positions, ridge=1e-3)
    lift_pinv = np.linalg.pinv(lift)
    K = K_CANONICAL
    samples = [rng.standard_normal((K, 400)) for _ in range(3)]
    covariances = [np.cov(s) + 0.5 * np.eye(K) for s in samples]
    sigma_bar = airm_frechet_mean(covariances)
    sigma_hat = covariances[0]
    u_canon = ordered_frame(rng.standard_normal((K, R_OCULAR)))
    # Realistic regime: subject/population ocular frames concentrate near the
    # canonical frame (the 15-degree gate) with PINNED column order.  Noise is
    # scaled by 1/sqrt(K): per-coordinate noise of scale c perturbs a unit
    # column by angle ~atan(c * sqrt(K)).  Wide angles hit the cap (tested
    # separately).
    u_subject = ordered_frame(u_canon + 0.020 * rng.standard_normal((K, R_OCULAR)))
    u_population = ordered_frame(u_canon + 0.012 * rng.standard_normal((K, R_OCULAR)))
    rotation = minimal_rotation(u_subject, u_canon)
    base = minimal_rotation(u_population, u_canon)
    return dict(lift=lift, lift_pinv=lift_pinv, sigma_bar=sigma_bar, sigma_hat=sigma_hat,
                u_canon=u_canon, u_subject=u_subject, u_population=u_population,
                rotation=rotation, base=base, rng=rng)


def test_lift_full_rank_round_trip(geometry):
    lift, lift_pinv = geometry["lift"], geometry["lift_pinv"]
    assert lift.shape == (K_CANONICAL, 19)
    assert np.max(np.abs(lift_pinv @ lift - np.eye(19))) <= 1e-10


def test_minimal_rotation_properties(geometry):
    for frame_key, rot_key in (("u_subject", "rotation"), ("u_population", "base")):
        rotation = geometry[rot_key]
        assert np.max(np.abs(rotation.T @ rotation - np.eye(K_CANONICAL))) <= 1e-10
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-8)
        np.testing.assert_allclose(rotation @ geometry[frame_key], geometry["u_canon"], atol=1e-10)


def test_round_trip_gate(geometry):
    for rho in (0.0, 0.5, 1.0):
        arm = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                               geometry["sigma_hat"], geometry["rotation"], geometry["base"], rho)
        assert np.max(np.abs(arm.pinv @ arm.transport - np.eye(19))) <= 1e-10


def test_rho_zero_bit_identity(geometry):
    lift, lift_pinv = geometry["lift"], geometry["lift_pinv"]
    pop_transport = geometry["base"] @ lift
    pop_pinv = lift_pinv @ geometry["base"].T
    arm = transport_family(lift, lift_pinv, geometry["sigma_bar"], geometry["sigma_hat"],
                           geometry["rotation"], geometry["base"], 0.0)
    assert np.array_equal(arm.transport, pop_transport)
    assert np.array_equal(arm.pinv, pop_pinv)
    y = geometry["rng"].standard_normal((19, 128))
    sigma_bar_inv = spd_power(geometry["sigma_bar"], -1.0)
    pop_arm = transport_family(lift, lift_pinv, geometry["sigma_bar"], None,
                               geometry["base"], geometry["base"], 0.0)
    out_a = canonical_clean(arm, geometry["u_canon"], sigma_bar_inv, y)
    out_b = canonical_clean(pop_arm, geometry["u_canon"], sigma_bar_inv, y)
    assert np.array_equal(out_a, out_b)


def test_geodesic_endpoints(geometry):
    q0 = rotation_geodesic(geometry["rotation"], geometry["base"], 0.0)
    q1 = rotation_geodesic(geometry["rotation"], geometry["base"], 1.0)
    assert np.array_equal(q0, geometry["base"])
    assert np.array_equal(q1, geometry["rotation"])
    arm1 = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                            geometry["sigma_hat"], geometry["rotation"], geometry["base"], 1.0)
    assert not arm1.abstained
    manual = geometry["rotation"] @ arm1.align @ geometry["lift"]
    np.testing.assert_allclose(arm1.transport, manual, atol=1e-12)


def test_locality_gate(geometry):
    union = orth(np.concatenate((geometry["u_subject"], geometry["u_canon"],
                                 geometry["u_population"]), axis=1))
    assert union.shape[1] <= 3 * R_OCULAR
    projector = np.eye(K_CANONICAL) - union @ union.T
    for rho in (0.3, 0.7, 1.0):
        q_rho = rotation_geodesic(geometry["rotation"], geometry["base"], rho)
        deviation = (q_rho @ geometry["base"].T - np.eye(K_CANONICAL)) @ projector
        assert np.max(np.abs(deviation)) <= 1e-10


def test_no_correction_path_returns_y(geometry):
    y = geometry["rng"].standard_normal((19, 64))
    for rho in (0.0, 0.5, 1.0):
        arm = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                               geometry["sigma_hat"], geometry["rotation"], geometry["base"], rho)
        np.testing.assert_allclose(arm.pinv @ (arm.transport @ y), y, atol=1e-9)


def test_angle_cap_abstains(geometry):
    rng = np.random.default_rng(5)
    flipped = orth(-geometry["u_population"] + 0.05 * rng.standard_normal(geometry["u_population"].shape))
    wide = minimal_rotation(flipped, geometry["u_canon"])
    if max_principal_angle(wide, geometry["base"]) > ANGLE_CAP:
        arm = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                               geometry["sigma_hat"], wide, geometry["base"], 0.8)
        assert arm.abstained and arm.rho == 0.0
        assert np.array_equal(arm.transport, geometry["base"] @ geometry["lift"])


def test_rho_eb_closed_form():
    assert rho_eb(1.0, 4.0) == pytest.approx(0.5)
    assert rho_eb(0.0, 1.0) == 0.0
    assert rho_eb(5.0, 0.0) == 1.0
    assert rho_eb(10.0, 0.1, hard_abstain=True) == 0.0


def test_gauge_null_matched_spectrum(geometry):
    gauge = gauge_null_rotation(geometry["rotation"], geometry["base"], seed=11)
    _, _, angles_true = rotation_log(geometry["rotation"], geometry["base"])
    _, _, angles_gauge = rotation_log(gauge, geometry["base"])
    assert np.max(np.abs(gauge.T @ gauge - np.eye(K_CANONICAL))) <= 1e-9
    np.testing.assert_allclose(np.sort(angles_gauge)[-len(angles_true):],
                               np.sort(angles_true), atol=1e-6)
    assert rotation_distance(gauge, geometry["base"]) > 0


def test_truncated_whitening_contracts(geometry):
    from eeg_chart.transport import truncated_inv_root

    for rho in (0.0, 0.5, 1.0):
        arm = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                               geometry["sigma_hat"], geometry["rotation"], geometry["base"],
                               rho, whitening="truncated")
        assert np.max(np.abs(arm.pinv @ arm.transport - np.eye(19))) <= 1e-10
    pop_arm = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                               None, geometry["base"], geometry["base"], 0.0)
    zero_arm = transport_family(geometry["lift"], geometry["lift_pinv"], geometry["sigma_bar"],
                                geometry["sigma_hat"], geometry["rotation"], geometry["base"],
                                0.0, whitening="truncated")
    assert np.array_equal(zero_arm.transport, pop_arm.transport)
    assert np.array_equal(zero_arm.pinv, pop_arm.pinv)
    white, white_inv = truncated_inv_root(geometry["sigma_hat"])
    assert np.max(np.abs(white @ white_inv - np.eye(len(white)))) <= 1e-8
    gains = np.linalg.eigvalsh(white)
    assert gains.max() / gains.min() <= 100.0 + 1e-6


def test_truncated_whitening_pass_through():
    from eeg_chart.transport import truncated_inv_root

    rng = np.random.default_rng(2)
    basis = orth(rng.standard_normal((40, 40)))
    values = np.concatenate((np.full(5, 1e4), np.ones(30), np.full(5, 1e-8)))
    matrix = (basis * values) @ basis.T
    white, _ = truncated_inv_root(matrix, var_keep=0.99, kappa_cap=100.0)
    gains = np.linalg.eigvalsh(white)
    # tiny-eigenvalue directions must NOT be blown up to 1e4 gains
    assert gains.max() / gains.min() <= 100.0 + 1e-6


def test_sh_basis_shape(geometry):
    basis = real_sh_basis(np.asarray([[0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]]))
    assert basis.shape == (3, K_CANONICAL)
    assert np.isfinite(basis).all()
