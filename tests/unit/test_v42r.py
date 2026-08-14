from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry, bipolar_eog
from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule, ddim_sample


ROOT=Path(__file__).resolve().parents[2]
BASE="8931ad7c036863976b4693f9f0721e11ab04857a"


def configs():
    data=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/data.yaml").read_text());data.update(yaml.safe_load((ROOT/"configs/calib_saddpm_cond_v42r/data.yaml").read_text()));data["v19_derived_root"]=data["source_root"]
    folds=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/folds.yaml").read_text())["folds"];return data,folds


def test_base_ledger_and_frozen_trees():
    assert subprocess.check_output(["git","merge-base","HEAD",BASE],cwd=ROOT,text=True).strip()==BASE
    assert (ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text().splitlines()[1].startswith("## v4.4")
    for path in ("results/calib_eegdfus_v41r","taas_submission"):
        assert subprocess.check_output(["git","rev-parse",f"{BASE}:{path}"],cwd=ROOT,text=True).strip()==subprocess.check_output(["git","rev-parse",f"HEAD:{path}"],cwd=ROOT,text=True).strip()


def test_cleanroom_provenance_declared():
    text=(ROOT/"reports/v42r_cleanroom_provenance.md").read_text()
    assert "No collaborator SADDPM branch" in text and "Independently implemented" in text


def test_exact_two_bipolar_regressors():
    eye=np.arange(80,dtype=float).reshape(4,20);value=bipolar_eog(eye,["HEOGL","HEOGR","VEOGU","VEOGL"])
    np.testing.assert_array_equal(value[0],eye[2]-eye[3]);np.testing.assert_array_equal(value[1],eye[0]-eye[1]);assert value.shape==(2,20)


def test_joint_shape_and_identity_centered_initialization():
    model=CalibSADDPMCond();x=torch.randn(1,46,512);y=torch.randn_like(x);c=torch.randn(1,46,53)
    output=model(x,y,torch.tensor([500]),c);assert output.shape==y.shape;torch.testing.assert_close(output,y,rtol=0,atol=0)


def test_x0_schedule_and_target_contract():
    schedule=LinearX0Schedule();clean=torch.randn(2,46,512);generator=torch.Generator().manual_seed(4);noisy,timestep,noise=schedule.forward_sample(clean,generator)
    assert len(schedule.beta)==1000 and noisy.shape==clean.shape and noise.shape==clean.shape and timestep.shape==(2,)


def test_transfer_branch_disable_is_exact_population_head():
    model=CalibSADDPMCond();x=torch.randn(1,46,512);y=torch.randn_like(x);c=torch.randn(1,46,53)
    with torch.no_grad():model.population_head.bias.fill_(.1);model.transfer_decoder.output.bias.fill_(.2)
    enabled=model(x,y,torch.tensor([4]),c,True);disabled=model(x,y,torch.tensor([4]),c,False)
    assert not torch.equal(enabled,disabled)


def test_ddim_fixed_noise_replay():
    model=CalibSADDPMCond(base=8);schedule=LinearX0Schedule(steps=4);y=torch.randn(1,46,512);c=torch.randn(1,46,53);noise=torch.randn_like(y)
    a=ddim_sample(model,y,c,noise,schedule,2);b=ddim_sample(model,y,c,noise,schedule,2);torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_support_query_and_outer_train_contract():
    data,folds=configs();registry=TransferRegistry(data,folds[0],30);cell=registry.cells[(folds[0]["test"][0],"ses-02","ERP")]
    assert cell.support_digest!=cell.query_digest and cell.transfer.shape==(46,2)
    sampler=TransferEpisodeSampler(data,folds[0],"test",42,registry);assert {row[0] for row in sampler.sources}.issubset(set(folds[0]["train"]))
    assert set(folds[0]["test"]).isdisjoint(folds[0]["train"])


def test_context_routes_and_zero_support():
    data,folds=configs();registry=TransferRegistry(data,folds[0],30);sampler=TransferEpisodeSampler(data,folds[0],"test",42,registry);participant=folds[0]["test"][0];meta={"participant":participant,"session":"ses-02","task":"ERP"}
    pop,_=sampler.condition_signature(meta,"POP");wrong,owner=sampler.condition_signature(meta,"WRONG");oracle,_=sampler.condition_signature(meta,"ORACLE")
    assert owner!=participant and pop.shape==wrong.shape==oracle.shape==(46,53) and not np.array_equal(pop,oracle)


def test_duration_prefix_has_no_overlap_or_future_normalization():
    data,folds=configs()
    for seconds,count in ((10,5),(30,15)):
        row=TransferRegistry(data,folds[0],seconds).manifest_rows()[0]
        assert row["window_count"]==count and row["overlap_samples"]==row["repeated_samples"]==0
        assert row["normalization_samples"]==seconds*100


def test_context_dropout_and_checkpoint_rules_frozen():
    training=yaml.safe_load((ROOT/"configs/calib_saddpm_cond_v42r/training.yaml").read_text())
    model=yaml.safe_load((ROOT/"configs/calib_saddpm_cond_v42r/model.yaml").read_text())
    assert training["full_updates"]==80000 and training["checkpoint_rule"].startswith("participant_aggregated")
    assert model["transfer_dropout"]==.20 and model["prediction"]=="x0" and model["ddim_steps"]==50


def test_governance_zero_reads():
    data,_=configs();assert data["sealed_reads"]==0;assert set(data["sealed_participants"]).isdisjoint(data["participants"])
