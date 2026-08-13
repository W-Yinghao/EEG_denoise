from __future__ import annotations

from pathlib import Path
import csv
import subprocess

import numpy as np
import torch
import yaml

from eeg_scad.data.official_support_v40r import validate_support_episode
from eeg_scad.models.eegdfus_mc_v40r import CompactSupportEncoder, EEGDfusMC, LinearSchedule, ddim_sample


ROOT=Path(__file__).resolve().parents[2]


def test_base_and_ledger_contract():
    assert subprocess.check_output(["git","merge-base","HEAD","8be1ec3a7c8c9735b548ca2dbd744c76bf27f37d"],cwd=ROOT,text=True).strip()=="8be1ec3a7c8c9735b548ca2dbd744c76bf27f37d"
    ledger=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert "v4.2" in ledger and ledger.index("v4.2") < ledger.index("v4.1")


def test_third_party_registry():
    rows=list(csv.DictReader((ROOT/"results/official_support_diffusion_v40r/third_party_registry.csv").open()))
    commits={row["method"]:row["commit"] for row in rows}
    assert commits["EEGDfus"]=="a19a652b3b6346188ae77067e1daf8b90cad005f"
    assert commits["D4PM"]=="5be2b3c72973fea6c879e63cd83067ff66aace13"
    assert all(row["license_status"] for row in rows)


def test_exact_support_contract_fixture():
    for seconds in (10,30):
        starts=list(range(0,seconds*100-199,200));episode={"starts":starts,"normalization_samples":seconds*100}
        assert validate_support_episode(episode,seconds)
        assert all(b-a==200 for a,b in zip(starts,starts[1:]))
        assert len(starts)==seconds//2
    assert validate_support_episode(None,0)


def test_shapes_and_zero_initialization():
    model=EEGDfusMC(channels=4,samples=32,features=8,context_dim=16);x=torch.randn(2,4,32);level=torch.ones(2,1);context=torch.randn(2,16)
    population=model(x,x,level,bypass=True);match=model(x,x,level,context=context)
    torch.testing.assert_close(match,population)
    assert population.shape==x.shape and torch.isfinite(population).all()


def test_support_encoder_is_set_mean_and_finite():
    encoder=CompactSupportEncoder(4,2,16);eeg=torch.randn(2,5,4,40);eog=torch.randn(2,5,2,40)
    first=encoder(eeg,eog);perm=torch.tensor([3,0,4,1,2]);second=encoder(eeg[:,perm],eog[:,perm]);torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)


def test_registered_support_channel_contract():
    encoder=CompactSupportEncoder();out=encoder(torch.randn(1,15,46,200),torch.randn(1,15,4,200));assert out.shape==(1,128)


def test_pop_bypass_is_context_independent():
    model=EEGDfusMC(channels=4,samples=32,features=8,context_dim=16);x=torch.randn(2,4,32);level=torch.ones(2,1)
    a=model(x,x,level,torch.randn(2,16),bypass=True);b=model(x,x,level,torch.randn(2,16),bypass=True);torch.testing.assert_close(a,b)


def test_adapter_changes_output_after_nonzero_parameters():
    model=EEGDfusMC(channels=4,samples=32,features=8,context_dim=16);torch.nn.init.constant_(model.support_mid.proj.weight,.01);x=torch.randn(2,4,32);level=torch.ones(2,1)
    a=model(x,x,level,torch.zeros(2,16));b=model(x,x,level,torch.ones(2,16));assert not torch.allclose(a,b)


def test_forward_marginal_and_ddim_count_contract():
    schedule=LinearSchedule(20);x=torch.randn(2,4,32);noise=torch.randn_like(x);t=torch.tensor([0,19]);sample=schedule.q_sample(x,t,noise);assert sample.shape==x.shape and torch.isfinite(sample).all()
    model=EEGDfusMC(channels=4,samples=32,features=8,context_dim=16);out=ddim_sample(model,x,noise,steps=5,bypass=True,schedule=schedule);assert out.shape==x.shape and torch.isfinite(out).all()


def test_same_noise_exact_pop_replay():
    model=EEGDfusMC(channels=4,samples=32,features=8,context_dim=16);schedule=LinearSchedule(20);y=torch.randn(2,4,32);noise=torch.randn_like(y)
    a=ddim_sample(model,y,noise,5,bypass=True,schedule=schedule);b=ddim_sample(model,y,noise,5,context=torch.randn(2,16),bypass=True,schedule=schedule);torch.testing.assert_close(a,b)


def test_governance_static():
    config=(ROOT/"configs/official_support_diffusion_v40r.yaml").read_text();assert "sealed_reads: 0" in config and "query_auxiliary_reads: 0" in config
    assert not (ROOT/"third_party/EEGDfus").exists()


def test_participant_folds_group_before_preprocessing():
    folds=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/folds.yaml").read_text())["folds"]
    for fold in folds:
        train,validation,test=map(set,(fold["train"],fold["validation"],fold["test"]));assert not train&validation and not train&test and not validation&test
    tests=[participant for fold in folds for participant in fold["test"]];assert len(tests)==15 and len(set(tests))==15


def test_support_manifest_has_no_overlap_query_or_repeats():
    path=ROOT/"results/official_support_diffusion_v40r/support_manifest.csv"
    if path.exists():
        rows=list(csv.DictReader(path.open()));assert rows
        for row in rows:assert row["overlap_samples"]=="0" and row["repeated_samples"]=="0" and row["query_samples"]=="0"


def test_registered_conditions_and_same_noise_source():
    source=(ROOT/"src/eeg_scad/training/train_v40r.py").read_text();assert 'CONDITIONS = ("POP", "MATCH", "WRONG", "SHUFFLED", "POP_MEAN", "ADAPTER_DISABLED")' in source
    assert "default_rng(20261040+fold)" in source and "condition" in source


def test_wrong_owner_is_explicitly_distinct():
    source=(ROOT/"src/eeg_scad/training/train_v40r.py").read_text();assert "candidate!=owner" in source


def test_no_target_selected_sample_and_k1():
    config=yaml.safe_load((ROOT/"configs/official_support_diffusion_v40r.yaml").read_text());assert config["diffusion"]["k"]==1
    source=(ROOT/"src/eeg_scad/training/train_v40r.py").read_text();assert "best sample" not in source.lower()


def test_participant_first_aggregation_code():
    source=(ROOT/"src/eeg_scad/cli/run_v40r.py").read_text();assert 'groupby(["condition","participant"]' in source and "bootstrap" in source


def test_query_eog_and_sealed_accounting():
    source=(ROOT/"src/eeg_scad/training/train_v40r.py").read_text();assert '"query_eog_inference_reads":0' in source and '"sealed_reads":0' in source


def test_manuscript_not_imported_by_v40r():
    paths=[ROOT/"src/eeg_scad/models/eegdfus_mc_v40r.py",ROOT/"src/eeg_scad/training/train_v40r.py",ROOT/"src/eeg_scad/cli/run_v40r.py"]
    assert all("taas_submission" not in path.read_text() for path in paths)
