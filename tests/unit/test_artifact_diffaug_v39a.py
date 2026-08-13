from __future__ import annotations
import json,subprocess
from pathlib import Path
import numpy as np,torch,yaml
from eeg_scad.models.artifact_generators_v39a import ArtifactCritic,ArtifactGenerator,ConditionalArtifactDiffusion,SpatialArtifactCodec,SupportDenoiserV39
from eeg_scad.training.artifact_diffaug_v39a import _fidelity,evaluate_denoiser

ROOT=Path(__file__).resolve().parents[2];BASE="e55d9df9c20afb28b4697658c3abce2ff4895610"
def test_base_ledger_and_governance():
    if (ROOT/".git").exists():
        assert subprocess.check_output(["git","merge-base","--is-ancestor",BASE,"HEAD"],cwd=ROOT,text=True)=="";assert subprocess.check_output(["git","diff","--name-only",BASE,"--","taas_submission"],cwd=ROOT,text=True)==""
        changed=subprocess.check_output(["git","diff","--name-only",BASE,"--","results"],cwd=ROOT,text=True).splitlines();assert all(path.startswith("results/artifact_diffaug_v39a/") for path in changed)
    ledger=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert "**版本：** v4.1" in ledger and "V39A" in ledger
def test_fixed_spatial_codec_round_trip_projection():
    a=np.random.default_rng(1).normal(size=(10,46,256)).astype(np.float32);c=SpatialArtifactCodec.fit(a,8);u=c.encode(a);assert u.shape==(10,8,256) and c.decode(u).shape==a.shape
def test_matched_generator_shapes():
    c=torch.randn(3,131);g=ArtifactGenerator();d=ArtifactCritic();x=g(torch.randn(3,16,256),c);assert x.shape==(3,8,256) and d(x,c).shape==(3,)
def test_diffusion_forward_and_ten_step_replay():
    m=ConditionalArtifactDiffusion().eval();x=torch.randn(2,8,256);c=torch.randn(2,131);t=torch.tensor([2,900]);assert m.q_sample(x,t,torch.randn_like(x)).shape==x.shape;n=torch.randn_like(x);torch.testing.assert_close(m.sample(c,n,10),m.sample(c,n,10))
def test_same_denoiser_architecture_for_all_arms():
    cfg=yaml.safe_load((ROOT/"configs/artifact_diffaug_v39a.yaml").read_text());assert len(cfg["denoiser_methods"])==5;m=SupportDenoiserV39();assert m(torch.randn(2,46,256),torch.randn(2,128)).shape==(2,46,256)
def test_equal_augmentation_exposure_registered():
    cfg=yaml.safe_load((ROOT/"configs/artifact_diffaug_v39a.yaml").read_text());assert cfg["augmentation_count_per_carrier"]==8;source=(ROOT/"src/eeg_scad/training/artifact_diffaug_v39a.py").read_text();assert '"training_rows":8*n' in source and '"updates":epochs*int(np.ceil(8*n/64))' in source and 'generator_sample("Empirical-Resample",8,c,b' in source
def test_support_contract_is_prefix_only_nonoverlap():
    source=(ROOT/"src/eeg_scad/data/artifact_diffaug_v39a.py").read_text();assert "range(0,prefix-length+1,length)" in source and "len(starts)==15" in source and "eog[:,:prefix]" in source
def test_outer_test_absent_from_generator_training():
    source=(ROOT/"src/eeg_scad/training/artifact_diffaug_v39a.py").read_text();assert 'sample_targets(data,fold_cfg,"train"' in source and "SpatialArtifactCodec.fit(train" in source
def test_query_eog_not_in_denoiser_signature():
    import inspect
    assert set(inspect.signature(SupportDenoiserV39.forward).parameters)=={"self","y","context"}
def test_no_target_selected_generated_sample():
    source=(ROOT/"src/eeg_scad/training/artifact_diffaug_v39a.py").read_text();assert "argmin" not in source and "target_selected" not in source
def test_participant_first_aggregation_and_biological_n():
    source=(ROOT/"src/eeg_scad/cli/run_v39a.py").read_text();assert 'groupby(["method","participant"]' in source and '"participant_coverage":15' in source
def test_sealed_and_manuscript_contract():
    cfg=yaml.safe_load((ROOT/"configs/artifact_diffaug_v39a.yaml").read_text());assert cfg["sealed_reads"]==0 and cfg["query_eog_inference_reads"]==0 and cfg["manuscript_modified"] is False

def test_exposure_uses_training_bank_and_detects_exact_copy():
    rng=np.random.default_rng(9);training=rng.normal(size=(4,3,256)).astype(np.float32);target=rng.normal(size=(2,3,256)).astype(np.float32);generated=np.stack((training[:2],training[:2]));row=_fidelity("fixture",generated,target,training,np.asarray(["a","b"]),9);assert row["exact_copy_rate"]==1.0

def test_natural_evaluator_preserves_channel_time_axis_order():
    class Identity(torch.nn.Module):
        def forward(self,y,context):return y
    rng=np.random.default_rng(4);shape=(2,46,256);bank={"clean":rng.normal(size=shape).astype(np.float32),"y":rng.normal(size=shape).astype(np.float32),"artifact":rng.normal(size=shape).astype(np.float32),"latent":rng.normal(size=(2,4,256)).astype(np.float32),"context":np.zeros((2,128),np.float32),"meta":[{"participant":"sub-02","session":"S","task":"T"},{"participant":"sub-03","session":"S","task":"T"}]};rows=evaluate_denoiser(Identity(),bank,torch.device("cpu"),"fixture",0,1,True);assert len(rows)==2 and all(np.isfinite(row["psd_distortion"]) for row in rows)

def test_registered_support_interventions_present():
    source=(ROOT/"src/eeg_scad/data/artifact_diffaug_v39a.py").read_text();assert all(value in source for value in ("population_context","mean_wrong_support","registered_shuffled_support"))

def test_committed_manifest_support_query_and_generator_roles():
    import pandas as pd
    targets=pd.read_csv(ROOT/"results/artifact_diffaug_v39a/artifact_target_manifest.csv");support=pd.read_csv(ROOT/"results/artifact_diffaug_v39a/support_manifest.csv");assert set(targets.split)=={"train","validation","test_paired","test_natural"};assert (support.support_seconds==30).all() and (support.overlap_samples==0).all() and (support.repeated_samples==0).all() and (support.query_samples==0).all();assert set(targets[targets.split=="train"].teacher_provenance_status)=={"known_generating_process","proxy"}
