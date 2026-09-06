from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from eeg_scad.energy.partial_observation import partial_observation_prox, partial_observation_solve
from eeg_scad.energy.projector import diagnostics, projector
from eeg_scad.energy.temporal_confidence import calibrate_quantiles, temporal_confidence

ROOT = Path(__file__).resolve().parents[2]
BASE = "7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c"


def fixture(dtype=torch.float64):
    generator = torch.Generator().manual_seed(27)
    basis = torch.randn(2, 6, 3, generator=generator, dtype=dtype)
    pi = projector(basis)
    candidate = torch.randn(2, 6, 11, generator=generator, dtype=dtype)
    anchor = torch.randn(2, 6, 11, generator=generator, dtype=dtype)
    mask = torch.rand(2, 11, generator=generator, dtype=dtype)
    return basis, pi, candidate, anchor, mask


def test_ledger_v16_terminal_update():
    text = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert "**版本：** v1.6" in text and "V27" in text


def test_projector_symmetric_idempotent_and_rank():
    _, pi, *_ = fixture(); values = diagnostics(pi[0])
    assert values["rank"] == 3 and values["symmetry_error"] < 1e-12 and values["idempotence_error"] < 1e-12


def test_projector_rotation_invariant():
    basis, pi, *_ = fixture(); q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=basis.dtype))
    assert torch.max(torch.abs(pi-projector(basis@q))) < 1e-12


def test_closed_form_matches_dense_linear_solve_float64():
    _, pi, candidate, anchor, mask = fixture()
    closed = partial_observation_prox(candidate, anchor, pi, mask, 1.0, 2.0)
    dense = partial_observation_solve(candidate, anchor, pi, mask, 1.0, 2.0)
    assert torch.max(torch.abs(closed-dense)) <= 1e-10


def test_closed_form_float32_finite():
    _, pi, candidate, anchor, mask = fixture(torch.float32)
    assert torch.isfinite(partial_observation_prox(candidate, anchor, pi, mask, 1, 8)).all()


def test_zero_penalties_return_candidate():
    _, pi, candidate, anchor, mask = fixture()
    assert torch.equal(partial_observation_prox(candidate, anchor, pi, mask, 0, 0), candidate)


def test_zero_mask_penalizes_all_directions():
    _, pi, candidate, anchor, mask = fixture(); mask.zero_()
    output = partial_observation_prox(candidate, anchor, pi, mask, 0, 2)
    assert torch.max(torch.abs(output-candidate/3)) < 1e-12


def test_one_mask_releases_span_but_penalizes_complement():
    _, pi, candidate, anchor, mask = fixture(); mask.fill_(1)
    output = partial_observation_prox(candidate, anchor, pi, mask, 0, 2)
    parallel = torch.einsum("bij,bjt->bit", pi, candidate)
    expected = parallel + (candidate-parallel)/3
    assert torch.max(torch.abs(output-expected)) < 1e-12


def test_zero_projector_is_global_shrinkage():
    _, pi, candidate, anchor, mask = fixture(); pi.zero_()
    output = partial_observation_prox(candidate, anchor, pi, mask, 0, 2)
    assert torch.max(torch.abs(output-candidate/3)) < 1e-12


def test_identity_projector_is_temporal_shrinkage():
    _, pi, candidate, anchor, mask = fixture(); pi[:] = torch.eye(6)
    output = partial_observation_prox(candidate, anchor, pi, mask, 0, 2)
    expected = candidate/(1+2*(1-mask[:, None])**2)
    assert torch.max(torch.abs(output-expected)) < 1e-12


def test_temporal_confidence_range_smoothing_and_replay():
    det = torch.arange(400, dtype=torch.float32).reshape(2, 2, 100)/100
    pop = det.flip(-1); q50, q90 = calibrate_quantiles(det, pop)
    first = temporal_confidence(det, pop, q50, q90, 10); second = temporal_confidence(det, pop, q50, q90, 10)
    assert torch.equal(first, second) and first.min() >= 0 and first.max() <= 1


def test_stepwise_uses_same_closed_form():
    source = (ROOT / "src/eeg_scad/models/calib_energy_sdedit.py").read_text()
    assert "partial_observation_prox" in source and "model._predict" in source and "model.alpha_bar" in source


def test_no_forbidden_large_methods_or_query_auxiliary():
    source = "\n".join(path.read_text() for path in (ROOT / "src/eeg_scad/energy").glob("*.py"))
    for forbidden in ("query_EOG", "query_operator", "query_event", "DPS", "DDRM", "DDNM", "K8"):
        assert forbidden not in source


def test_participant_first_contract_declared():
    source = (ROOT / "src/eeg_scad/evaluation/aggregate_v26.py").read_text()
    assert "participant_first" in source and "nanmean" in source


def test_v26_inputs_are_read_only_paths():
    assert not (ROOT / "results/calib_sdedit_v26").is_symlink()
    source = inspect.getsource(__import__("eeg_scad.cli.run_v27", fromlist=["preflight"]).preflight)
    assert "BASE" in source and BASE == "7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c"
