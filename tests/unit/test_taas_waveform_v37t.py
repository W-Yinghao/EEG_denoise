from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from eeg_scad.evaluation.uncertainty_v37t import (
    constant_width_interval, ensemble_crps, error_dispersion, interval_metrics,
    projected_variance,
)


def test_empirical_intervals_and_constant_reference() -> None:
    samples = np.arange(16, dtype=float)[:, None, None, None]
    target = np.full((1, 1, 1), 7.5)
    empirical = interval_metrics(samples, target, .8)
    constant = constant_width_interval(samples.mean(0), samples, target, .8)
    assert empirical["coverage"] == 1.0
    assert empirical["interval_width"] > 0
    assert constant["coverage"] == 1.0


def test_ensemble_crps_zero_for_exact_ensemble() -> None:
    target = np.ones((2, 3, 4))
    samples = np.repeat(target[None], 16, axis=0)
    assert ensemble_crps(samples, target) == 0.0


def test_error_dispersion_detects_ranked_windows() -> None:
    rng = np.random.default_rng(37)
    target = np.zeros((12, 2, 4))
    scales = np.arange(1, 13, dtype=float)[:, None, None]
    samples = rng.normal(size=(16, 12, 2, 4)) * scales[None]
    assert error_dispersion(samples, target) > 0.5


def test_projected_variance_decomposition() -> None:
    samples = np.zeros((4, 2, 3, 5))
    samples[:, :, 0] = np.arange(4)[:, None, None]
    pi = np.zeros((2, 3, 3)); pi[:, 0, 0] = 1
    parallel, complement = projected_variance(samples, pi)
    assert parallel > 0
    assert complement == 0


def test_registered_contract_is_k16_without_target_selection() -> None:
    from eeg_scad.cli.run_v37t import _cfg
    cfg = _cfg()
    assert cfg["uncertainty_k"] == 16
    assert cfg["point_k"] == 1
    assert cfg["lambda_y"] == .5
    assert cfg["energy_mode"] == "final_only"


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/taas_waveform_v37t"
BASE = "a90cabf5ed7167e0bc6cfc01257e74592b6e7d85"


def test_base_and_ledger_v38() -> None:
    if (ROOT / ".git").exists():
        assert subprocess.check_output(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT).decode() == ""
    else:
        assert json.loads((RESULT / "source_registry.json").read_text())["base_commit"] == BASE
    ledger = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert "**版本：** v3.8" in ledger
    assert "V37T最终定位为B" in ledger


def test_scope_boundary_and_manuscript_unchanged() -> None:
    scope = (ROOT / "reports/v37t_paper_scope_boundary.md").read_text()
    assert "TAAS ownership" in scope and "Debias/privacy-paper ownership" in scope
    if (ROOT / ".git").exists():
        changed = subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "taas_submission"], cwd=ROOT, text=True)
        assert changed == ""
    else:
        assert json.loads((RESULT / "source_registry.json").read_text())["manuscript_unchanged"] is True


def test_all_frozen_bindings_verified_without_substitution() -> None:
    rows = list(csv.DictReader((RESULT / "checkpoint_binding.csv").open()))
    assert len(rows) == 115
    assert {row["v37t_binding"] for row in rows} <= {"verified", "closed_form_or_historical"}


def test_all_stochastic_cells_are_k16_and_unselected() -> None:
    records = [json.loads(path.read_text()) for path in sorted((RESULT / "stochastic").glob("cell_*.json"))]
    assert len(records) == 15
    assert all(row["K"] == 16 and row["target_selected_samples"] == 0 for row in records)
    assert all(row["query_auxiliary_reads"] == 0 and row["sealed_reads"] == 0 for row in records)


def test_stochastic_archive_contract() -> None:
    record = json.loads(next((RESULT / "stochastic").glob("cell_*.json")).read_text())
    with np.load(record["path"], allow_pickle=False) as archive:
        assert archive["V27_ENERGY_SDEDIT_L05"].shape[0] == 16
        assert archive["V26_CALIB_SDEDIT_MATCH"].shape[0] == 16
        assert archive["V26_POP_SDEDIT"].shape[0] == 16


def test_uncertainty_is_participant_first() -> None:
    rows = list(csv.DictReader((RESULT / "uncertainty_summary.csv").open()))
    assert len({row["participant"] for row in rows}) == 15
    assert {float(row["level"]) for row in rows} == {.5, .8, .9}
    assert all(0 <= float(row["coverage"]) <= 1 for row in rows)


def test_exact_duration_rows_are_v31_superseding_evidence() -> None:
    rows = list(csv.DictReader((RESULT / "support_evidence.csv").open()))
    duration = [row for row in rows if row["evidence"] == "exact_duration_v31"]
    assert duration
    assert {int(row["duration_seconds"]) for row in duration} == {0, 5, 10, 30, 120}
    five = [row for row in duration if row["duration_seconds"] == "5"]
    assert all(float(row["effective_seconds"]) == 4 for row in five)


def test_natural_pareto_keeps_l05_primary() -> None:
    rows = list(csv.DictReader((RESULT / "natural_pareto.csv").open()))
    assert {row["method"] for row in rows} == {"V27_ENERGY_SDEDIT_L05", "V27_ENERGY_SDEDIT_L2", "V27_ENERGY_SDEDIT_L8"}
    diagnosis = json.loads((RESULT / "development_diagnosis.json").read_text())
    assert diagnosis["primary_method"] == "V27_ENERGY_SDEDIT_L05"
    assert diagnosis["final_positioning"].startswith("B_")


def test_no_erp_or_ssvep_preservation_alias() -> None:
    for path in (RESULT / "common_natural_summary.csv", RESULT / "natural_pareto.csv"):
        header = next(csv.reader(path.open()))
        assert "erp_proxy" not in header and "ssvep_proxy" not in header
