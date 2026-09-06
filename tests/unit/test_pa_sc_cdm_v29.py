from __future__ import annotations
import inspect,json
from pathlib import Path
import torch,yaml
from eeg_scad.data.folds import load_folds,validate_folds
from eeg_scad.models.pop_adapter_cdm import PopAdapterCDM
from eeg_scad.models.pop_adapter_det import PopAdapterDET
from eeg_scad.models.pop_clean_cdm import PopCleanCDM
from eeg_scad.models.support_adapter_cdm import SupportAdapterCDM
from eeg_scad.models.support_adapter_common import SupportResidualAdapter
from eeg_scad.models.support_adapter_det import SupportAdapterDET
from eeg_scad.training.train_v29 import _rank

ROOT=Path(__file__).resolve().parents[2]
def fixture(length=32):
    g=torch.Generator().manual_seed(29);y=torch.randn((2,46,length),generator=g);pop=torch.randn((2,46,length),generator=g);context=torch.randn((2,128),generator=g);return y,pop,context

def test_ledger_v19_loaded_and_active_route():
    text=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert ("**版本：** v1.9" in text or "**版本：** v2.0" in text) and "V29" in text
def test_base_sha_and_v28_frozen():assert json.loads((ROOT/"results/sc_cdm_v28/terminal_manifest.json").read_text())["base_commit"]=="40eae116e70e9de7fe0af55d64ee25551932c4a8"
def test_folds_exact_and_disjoint():
    folds=load_folds(ROOT/"configs/pa_sc_cdm_v29/folds.yaml");participants=yaml.safe_load((ROOT/"configs/pa_sc_cdm_v29/data.yaml").read_text())["participants"];validate_folds(folds,participants)
def test_zero_initialization_and_full_amplitude():
    y,pop,c=fixture();m=SupportAdapterDET(8);assert torch.equal(m(y,pop,c),pop);assert not hasattr(m,"output_scale")
def test_cdm_zero_initialization():
    y,pop,c=fixture();m=SupportAdapterCDM(8);t=torch.ones(2,dtype=torch.long);assert torch.equal(m.predict_x0(y,y,pop,c,t),pop)
def test_exact_pop_bypass_not_zero_context():
    y,pop,c=fixture();m=SupportAdapterDET(8);m.adapter.output.bias.data.fill_(1);assert torch.equal(m(y,pop,c,True),pop) and not torch.equal(m(y,pop,torch.zeros_like(c)),pop)
def test_support_and_time_are_separate():
    model=SupportResidualAdapter(184,8,True);assert model.support_projection is not model.time_projection and "context +" not in inspect.getsource(model.forward)
def test_pop_adapter_capacity_matches_support_adapter():
    assert sum(p.numel() for p in PopAdapterDET(8).parameters())-128==sum(p.numel() for p in SupportAdapterDET(8).parameters())
    assert sum(p.numel() for p in PopAdapterCDM(8).parameters())-128==sum(p.numel() for p in SupportAdapterCDM(8).parameters())
def test_same_noise_pop_bypass_trajectory():
    y,_,c=fixture();noise=torch.randn_like(y);pop=PopCleanCDM(width=8).eval();adapter=SupportAdapterCDM(8).eval();a,ta=pop.sample(y,noise,10);b,tb=adapter.sample(pop,y,c,noise,10,True);assert torch.equal(a,b) and len(ta)==len(tb)==10
def test_match_wrong_can_change_after_training():
    y,pop,c=fixture();m=SupportAdapterDET(8);m.adapter.output.bias.data.fill_(.1);m.adapter.blocks[0].support_film.weight.data.fill_(.01);assert m(y,pop,c).shape==m(y,pop,-c).shape
def test_context_rank_same_target_normalized():
    y,pop,c=fixture();target=torch.ones_like(y);assert torch.isfinite(_rank(target,target+1,target+2,target,.01))
def test_training_semantics_match_only():
    source=(ROOT/"src/eeg_scad/training/train_v29.py").read_text();assert "_paired_loss(match,target,cfg)" in source and "_paired_loss(wrong" not in source and "_paired_loss(bypass" not in source
def test_natural_loss_increment_only():
    source=(ROOT/"src/eeg_scad/training/train_v29.py").read_text();assert "increment=pred-pop" in source
def test_resume_state_complete():
    source=(ROOT/"src/eeg_scad/training/train_v29.py").read_text()
    for key in ("optimizer","scheduler","ema","global_step","paired_natural_rng","support_rng","wrong_support_rng","diffusion_rng"):assert f'"{key}"' in source
def test_query_auxiliary_forbidden():assert set(SupportAdapterCDM.forbidden_fields)>={"query_EOG","query_operator","query_event"}
def test_zero_artifact_metric_exclusion_registered():
    source=(ROOT/"src/eeg_scad/cli/run_v29.py").read_text();assert 'metric["snr_improvement"]=np.nan;metric["artifact_rrmse"]=np.nan' in source
def test_k1_ddim10_governance():
    cfg=yaml.safe_load((ROOT/"configs/pa_sc_cdm_v29/evaluation.yaml").read_text());assert cfg["K"]==1 and cfg["ddim_steps"]==10
def test_sealed_registry_and_development_only():
    cfg=yaml.safe_load((ROOT/"configs/pa_sc_cdm_v29/data.yaml").read_text());assert len(cfg["sealed_participants"])==8 and cfg["development_only"]
def test_pop_route_is_direct_bypass():assert "return population if bypass" in inspect.getsource(SupportAdapterDET.forward)
def test_support_encoder_is_frozen_by_training():
    source=(ROOT/"src/eeg_scad/training/train_v29.py").read_text();assert "freeze(pop);freeze(support)" in source
def test_adapter_output_shape():
    y,pop,c=fixture();assert SupportAdapterDET(8)(y,pop,c).shape==y.shape
def test_project_manifest_uses_no_proxy_tasks():
    cfg=yaml.safe_load((ROOT/"configs/pa_sc_cdm_v29/evaluation.yaml").read_text());assert cfg["erp_status"]==cfg["ssvep_status"]=="unavailable"
def test_governance_source_has_zero_query_reads():
    source=(ROOT/"src/eeg_scad/cli/run_v29.py").read_text();assert '"query_EOG_reads":0' in source and '"sealed_reads":0' in source
