"""Contracts for the development-only Klados B6 gamma diagnostic."""

from __future__ import annotations

import numpy as np

from eeg_cgdr.experiments.klados_b6_gamma_diagnostic import (
    DEVELOPMENT_SOURCE_RECORDS,
    GAMMA_GRID,
    METHOD_M2,
    deterministic_qy,
    deterministic_quadratic_soft_proximal,
    select_global_gamma,
    validate_diagnostic_config,
)


def _config() -> dict:
    return {
        "harness_level": 1,
        "diagnostic_protocol": "klados_b6_gamma_development_v1",
        "base_config": (
            "configs/cgdr/mechanism_audit_klados_padding_repair_development.yaml"
        ),
        "backup_config": "configs/deferred_backups/b6_pop_shrink.yaml",
        "development_source_records": list(DEVELOPMENT_SOURCE_RECORDS),
        "gamma_grid": list(GAMMA_GRID),
        "claim_boundary": {
            "partition": "development_only",
            "records_are_participants": False,
            "confirmatory": False,
            "evaluation_records_reused": False,
            "evaluation_source_records_previously_used_in_diagnosis": [
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
            ],
        },
        "deferred_b6_activation": {
            "activate_in_memory_for_this_diagnostic": True,
            "persisted_backup_config_must_remain_disabled": True,
        },
        "deterministic_soft_proximal": {
            "tau": 0.5,
            "tau_frozen": True,
            "closed_form": "x=Qy+tau*Pi*y",
            "quadratic_penalty_lambda": 1.0,
        },
        "m2": {
            "candidate": "M2",
            "trust_radius": 0.1,
            "trust_radius_provenance": (
                "historical_development_sampler_choice_reused_only_as_exploratory_constant"
            ),
            "seeds": [20260801, 20260802, 20260803, 20260804, 20260805],
            "output_rule": "posterior_mean_of_seed_samples",
        },
        "selection": {
            "method": METHOD_M2,
            "primary_metric": "e_parallel",
            "primary_direction": "lower",
            "preservation_metric": "e_perp",
            "preservation_reference_gamma": 0.0,
            "maximum_median_preservation_delta": 0.05,
            "minimum_valid_record_fraction": 0.75,
            "minimum_operator_eligibility_fraction": 0.75,
            "metric_tie_tolerance": 1.0e-12,
            "tie_break_order": list(GAMMA_GRID),
        },
        "output_root": "results/cgdr/klados_b6_gamma_development",
    }


def test_protocol_is_development_source_record_only() -> None:
    contract = validate_diagnostic_config(_config())

    assert contract.development_records == (31, 32, 33, 34, 35, 36, 44, 45)
    assert contract.gamma_grid == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert contract.soft_tau == 0.5


def test_hard_qy_and_soft_proximal_have_exact_subspace_actions() -> None:
    projector = np.diag([1.0, 0.0, 1.0])
    observed = np.array(
        [[2.0, -1.0], [3.0, 4.0], [-2.0, 5.0]], dtype=np.float64
    )
    hard = deterministic_qy(observed, projector)
    soft = deterministic_quadratic_soft_proximal(
        observed, projector, tau=0.25
    )
    complement = np.eye(3) - projector

    assert np.array_equal(complement @ hard, complement @ observed)
    assert np.array_equal(projector @ hard, np.zeros_like(observed))
    assert np.array_equal(complement @ soft, complement @ observed)
    assert np.array_equal(projector @ soft, 0.25 * (projector @ observed))


def test_soft_proximal_is_closed_form_quadratic_argmin() -> None:
    projector = np.diag([1.0, 1.0, 0.0])
    observed = np.array(
        [[1.0, 2.0, -1.0], [4.0, -2.0, 3.0], [7.0, 8.0, 9.0]],
        dtype=np.float64,
    )
    tau = 0.25
    penalty_lambda = 1.0 / tau - 1.0
    solution = deterministic_quadratic_soft_proximal(
        observed, projector, tau=tau
    )
    objective_gradient = solution - observed + penalty_lambda * (projector @ solution)

    assert np.allclose(objective_gradient, 0.0, atol=1.0e-12, rtol=0.0)
    assert np.array_equal(
        deterministic_quadratic_soft_proximal(observed, projector, tau=0.0),
        deterministic_qy(observed, projector),
    )
    assert np.array_equal(
        deterministic_quadratic_soft_proximal(observed, projector, tau=1.0),
        observed,
    )


def _selection_rows(
    *,
    primary: dict[float, float],
    preservation_delta: dict[float, float],
) -> list[dict]:
    rows: list[dict] = []
    for record in DEVELOPMENT_SOURCE_RECORDS:
        for gamma in GAMMA_GRID:
            rows.append(
                {
                    "source_record": f"sim{record:02d}",
                    "gamma": gamma,
                    "method": METHOD_M2,
                    "status": "success",
                    "b6_status": "eligible",
                    "e_parallel": primary[gamma],
                    "e_perp": 1.0 + preservation_delta[gamma],
                }
            )
    return rows


def test_global_gamma_rejects_primary_winner_that_breaks_preservation() -> None:
    rows = _selection_rows(
        primary={0.0: 1.0, 0.25: 0.8, 0.5: 0.6, 0.75: 0.4, 1.0: 0.7},
        preservation_delta={
            0.0: 0.0,
            0.25: 0.01,
            0.5: 0.02,
            0.75: 0.08,
            1.0: 0.03,
        },
    )

    result = select_global_gamma(rows, _config())

    assert result["status"] == "development_gamma_frozen"
    assert result["best_global_gamma"] == 0.5
    rejected = next(row for row in result["gamma_summaries"] if row["gamma"] == 0.75)
    assert rejected["preservation_constraint_passed"] is False


def test_global_gamma_tie_uses_preregistered_order_and_keeps_fallbacks_in_denominator() -> None:
    rows = _selection_rows(
        primary={0.0: 1.0, 0.25: 0.5, 0.5: 0.5, 0.75: 0.7, 1.0: 0.8},
        preservation_delta={gamma: 0.0 for gamma in GAMMA_GRID},
    )
    for row in rows:
        if row["gamma"] == 0.25 and row["source_record"] in {"sim31", "sim32", "sim33"}:
            row["b6_status"] = "fallback_POP"

    result = select_global_gamma(rows, _config())

    assert result["best_global_gamma"] == 0.5
    gamma_quarter = next(
        row for row in result["gamma_summaries"] if row["gamma"] == 0.25
    )
    assert gamma_quarter["valid_record_fraction"] == 1.0
    assert gamma_quarter["operator_eligibility_fraction"] == 5 / 8
    assert gamma_quarter["operator_eligibility_fraction_passed"] is False
    assert gamma_quarter["eligible_for_global_selection"] is False

    for row in rows:
        if row["gamma"] == 0.25:
            row["b6_status"] = "eligible"
    tied = select_global_gamma(rows, _config())
    assert tied["best_global_gamma"] == 0.25
