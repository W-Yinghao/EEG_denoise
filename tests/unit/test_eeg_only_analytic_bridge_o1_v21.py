from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from eeg_cgdr.experiments.eeg_only_analytic_bridge_o1_v21 import (
    _generate_unrestricted,
    _plus_one,
    _rand_stats,
    exact_signflip,
    solve_bridge,
)


ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/cgdr/eeg_only_analytic_bridge_o1_v21.yaml"


def test_zero_mask_identity_and_zero_coefficients() -> None:
    rng=np.random.default_rng(1);y=rng.normal(size=(6,40));b=rng.normal(size=(6,4));z,c,k=solve_bridge(y,b,np.ones(6),np.zeros(40,dtype=bool),.1,1.)
    assert np.max(np.abs(z))==0 and np.max(np.abs(c))==0 and k==0
    np.testing.assert_array_equal(y-c,y)


def test_span_consistency() -> None:
    rng=np.random.default_rng(2);y=rng.normal(size=(8,60));b=rng.normal(size=(8,4));mask=np.zeros(60,dtype=bool);mask[10:40]=1
    _,c,_=solve_bridge(y,b,np.linspace(.5,1.5,8),mask,.1,1.)
    off=c-b@np.linalg.pinv(b)@c
    assert np.linalg.norm(off)/(np.linalg.norm(c)+1e-12)<1e-12
    assert np.max(np.abs(c[:,~mask]))==0


def test_dual_solver_and_kkt() -> None:
    rng=np.random.default_rng(3);y=rng.normal(size=(7,80));b=rng.normal(size=(7,4));mask=np.zeros(80,dtype=bool);mask[5:55]=1
    z1,c1,k1=solve_bridge(y,b,np.ones(7),mask,.01,10.,"block");z2,c2,k2=solve_bridge(y,b,np.ones(7),mask,.01,10.,"dense")
    np.testing.assert_allclose(z1,z2,atol=1e-10,rtol=0);np.testing.assert_allclose(c1,c2,atol=1e-10,rtol=0)
    assert max(k1,k2)<1e-8


def test_context_intervention_changes_output() -> None:
    rng=np.random.default_rng(4);y=rng.normal(size=(9,50));mask=np.ones(50,dtype=bool);b=rng.normal(size=(9,4));b2=b.copy();b2[:,0]*=2
    c1=solve_bridge(y,b,np.ones(9),mask,.1,1.)[1];c2=solve_bridge(y,b2,np.ones(9),mask,.1,1.)[1]
    assert np.linalg.norm(c1-c2)>0


def test_unrestricted_assignment_allows_fixed_points_and_replays() -> None:
    recipients=[f"p{i}" for i in range(15)];owners=recipients+["p15"]
    a,u,_,_=_generate_unrestricted(recipients,owners,1000,20260822);b,v,_,_=_generate_unrestricted(recipients,owners,1000,20260822)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(u,v)
    assert a.shape==(1000,15) and np.any(a==np.arange(15)[None])
    assert all(len(set(row.tolist()))==15 for row in a)


def test_plus_one_and_signflip() -> None:
    null=np.asarray([-1.,0.,1.]);assert _plus_one(null,2.)==.25;assert _plus_one(null,1.)==.5
    assert exact_signflip(np.ones(15))==pytest.approx(1/32768)


def test_randomization_fixed_effect_fixture() -> None:
    risk=np.ones((3,4));np.fill_diagonal(risk[:,:3],.1);pop=np.full(3,.8);a=np.asarray([[1,2,3],[2,3,0]],dtype=np.uint8)
    p,w=_rand_stats(risk,pop,a);assert np.all(p<.7) and np.all(w<.9)


def test_config_freezes_blocks_grid_and_scientific_units() -> None:
    c=yaml.safe_load(CONFIG.read_text());assert c["blocks"]["S120"]==[0.,120.] and c["blocks"]["Qgen"]==[150.,270.]
    assert c["blocks"]["Qnatural"]==[300.,"record_end"] and len(c["primary_recipients"])==15 and len(c["development_participants"])==16
    assert c["policy_only_participant"]=="sub-24" and c["grid"]["eta_z"]==[.001,.01,.1,1.,10.] and c["grid"]["eta_d"]==[0.,.1,1.,10.]


def test_submitter_rejects_gpu() -> None:
    submit=ROOT/"scripts/slurm/eeg_only_analytic_bridge_o1_v21/submit.sh"
    a=subprocess.run([str(submit),"A100","p0-preflight"],capture_output=True,text=True);b=subprocess.run([str(submit),"cpu","p0-preflight","--gres","gpu:1"],capture_output=True,text=True)
    assert a.returncode==2 and b.returncode==2 and "forbidden" in b.stderr


def test_source_heads_and_A_track() -> None:
    c=yaml.safe_load(CONFIG.read_text())
    for path,key in (("source_v19_worktree","source_v19_commit"),("source_audit_worktree","source_audit_commit"),("source_v20_worktree","source_v20_commit"),("a_track_worktree","a_track_commit")):
        assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=c[path],text=True).strip()==c[key]
    assert subprocess.run(["git","diff","--quiet","HEAD","--","taas_submission"],cwd=c["a_track_worktree"]).returncode==0


def test_V20_authorization_is_exact() -> None:
    c=yaml.safe_load(CONFIG.read_text());decision=json.loads((Path(c["source_v20_result"])/"route_decision.json").read_text())
    assert decision["scientific_route"]=="V20_NATURAL_TRANSFER_PASS" and decision["O1_status"]=="O1_AUTHORIZED_NOT_RUN"


def test_inference_bundle_manifest_has_no_auxiliary() -> None:
    path=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21/inference_bundle_manifest.csv"
    if not path.is_file():pytest.skip("P1 not run")
    import csv
    with path.open(newline="") as f:rows=list(csv.DictReader(f))
    assert rows and all(r["contains_query_EOG"]==r["contains_query_operator"]==r["contains_event"]=="0" for r in rows)


def test_access_ledgers_and_output_freeze() -> None:
    root=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21";path=root/"output_freeze.json"
    if not path.is_file():pytest.skip("P8 not run")
    value=json.loads(path.read_text());assert value["frozen"] and value["query_EOG_reads"]==value["query_event_reads"]==value["query_operator_reads"]==value["sealed_reads"]==0
    assert value["raw_development_read_count"]==0


def test_frozen_corrections_preserve_float64_span_precision() -> None:
    root=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21";manifest=root/"output_manifest.csv"
    if not manifest.is_file():pytest.skip("P8 not run")
    import csv
    with manifest.open(newline="") as f:rows=list(csv.DictReader(f))
    assert rows
    with np.load(rows[0]["path"],allow_pickle=False) as z:
        assert z["corrections"].dtype==np.float64


def test_operator_equivalence_and_context_completeness() -> None:
    root=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21";path=root/"operator_coordinate_contract.json"
    if not path.is_file():pytest.skip("P2 not run")
    value=json.loads(path.read_text());assert value["max_equivalence_error"]<=1e-12
    import csv
    with (root/"operator_packages.csv").open(newline="") as f:rows=list(csv.DictReader(f))
    assert len(rows)>=89 and all(int(r["donors"])==15 and int(r["channels"])==46 and int(r["regressors"])==4 for r in rows)


def test_nested_selection_excludes_outer_and_sub24() -> None:
    path=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21/selection_trace.json"
    if not path.is_file():pytest.skip("P5 not run")
    value=json.loads(path.read_text());assert value["outer_heldout_outcomes_used"] is False and value["sub24_used"] is False and value["grid_points"]==20


def test_context_intervention_uses_only_active_detector_masks() -> None:
    root=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21";contexts=root/"context_intervention_validity.csv";masks=root/"query_eeg_mask_manifest.csv"
    if not contexts.is_file() or not masks.is_file():pytest.skip("P6 not run")
    import csv
    with contexts.open(newline="") as f:context_rows=list(csv.DictReader(f))
    with masks.open(newline="") as f:mask_rows=list(csv.DictReader(f))
    active={(r["participant"],f'{r["session"]}_{r["task"]}') for r in mask_rows if int(r["masked_samples"])>0}
    assert context_rows and all((r["participant"],r["unit"]) in active for r in context_rows)


def test_primary_randomization_is_unrestricted_and_v20_secondary() -> None:
    root=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21";path=root/"unrestricted_assignment_metadata.json"
    if not path.is_file():pytest.skip("P10 not run")
    value=json.loads(path.read_text());assert "fixed points allowed" in value["algorithm"] and value["seed"]==20260822 and value["accepted_replicates"]==100000
    secondary=json.loads((root/"fixed_point_free_sensitivity.json").read_text());assert "sensitivity only" in secondary["label"]


def test_terminal_never_runs_model_gpu_or_diffusion() -> None:
    path=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21/terminal_manifest.json"
    if not path.is_file():pytest.skip("P13 not run")
    value=json.loads(path.read_text());assert value["DET_executed"] is False and value["diffusion_executed"] is False and value["GPU_jobs"]==value["raw_reads"]==value["sealed_reads"]==0


def test_oracle_diagnostics_cannot_set_primary_pass() -> None:
    path=ROOT/"results/cgdr/eeg_only_analytic_bridge_o1_v21/route_decision.json"
    if not path.is_file():pytest.skip("P13 not run")
    value=json.loads(path.read_text());
    if value["scientific_route"]=="O1_EEG_ONLY_BRIDGE_PASS":assert all(value["criteria"][key] for key in ("technical","detector","P","W","absolute_safety","population_relative_safety"))
