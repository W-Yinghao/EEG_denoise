from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from eeg_scad.data.folds import load_folds,validate_folds
from eeg_scad.evaluation.natural_metrics_v28 import attenuation_consistency,natural_metrics_v28
from eeg_scad.evaluation.task_preservation_v28 import inventory
from eeg_scad.models.pop_clean_cdm import PopCleanCDM
from eeg_scad.models.pop_clean_det import PopCleanDET
from eeg_scad.models.support_clean_cdm import SupportCleanCDM
from eeg_scad.models.support_clean_det import SupportCleanDET

ROOT=Path(__file__).resolve().parents[2]


def fixture(length=32):
    generator=torch.Generator().manual_seed(28);y=torch.randn((2,46,length),generator=generator);clean=y-.1*torch.randn(y.shape,generator=generator);context=torch.randn((2,128),generator=generator);return y,clean,context


def test_ledger_v17_loaded():
    text=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert "**版本：** v1.7" in text and "V28 SC-CDM" in text


def test_base_sha_and_v27_frozen():
    assert json.loads((ROOT/"results/calib_energy_v27/terminal_manifest.json").read_text())["base_commit"]=="7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c"


def test_fold_reuse_and_disjointness():
    folds=load_folds(ROOT/"configs/sc_cdm_v28/folds.yaml");participants=json.loads(json.dumps(__import__("yaml").safe_load((ROOT/"configs/sc_cdm_v28/data.yaml").read_text())))["participants"];validate_folds(folds,participants)
    for fold in folds:assert not (set(fold["train"])&set(fold["validation"]) or set(fold["train"])&set(fold["test"]) or set(fold["validation"])&set(fold["test"]))


def test_clean_x0_shapes():
    y,clean,context=fixture();g=torch.Generator().manual_seed(1);prediction,_,state=SupportCleanCDM(width=8).training_prediction(clean,y,context,g);assert prediction.shape==state.shape==clean.shape


def test_contaminated_y_is_condition_input():
    y,clean,context=fixture();model=SupportCleanCDM(width=8).eval();state=torch.zeros_like(y);t=torch.zeros(2,dtype=torch.long);assert not torch.equal(model.predict_x0(state,y,context,t),model.predict_x0(state,y+1,context,t))


def test_support_film_context_changes_output():
    y,_,context=fixture();model=SupportCleanDET(width=8).eval();assert float((model(y,context)-model(y,-context)).abs().max())>0


def test_population_models_have_no_support_argument():
    assert PopCleanDET.uses_subject_support is False and PopCleanCDM.uses_subject_support is False


def test_same_backbone_parameter_count():
    assert sum(p.numel() for p in PopCleanDET(width=8).parameters())==sum(p.numel() for p in SupportCleanDET(width=8).parameters())
    assert sum(p.numel() for p in PopCleanCDM(width=8).parameters())==sum(p.numel() for p in SupportCleanCDM(width=8).parameters())


def test_full_gaussian_initialization_and_fixed_noise_replay():
    y,_,context=fixture();noise=torch.randn_like(y);model=SupportCleanCDM(width=8).eval();a,ta=model.sample(y,context,noise,10);b,tb=model.sample(y,context,noise,10);assert len(ta)==len(tb)==10 and torch.equal(a,b) and abs(ta[0]["state_rms"]-float(noise.square().mean().sqrt()))<1e-7


def test_ddim_25_call_count_and_k1():
    y,_,context=fixture();value,trajectory=SupportCleanCDM(width=8).eval().sample(y,context,torch.randn_like(y),25);assert value.shape==y.shape and len(trajectory)==25


def test_identity_architecture_is_observation_anchored():
    y,_,context=fixture();model=SupportCleanDET(width=8)
    for parameter in model.parameters():parameter.data.zero_()
    assert torch.equal(model(y,context),y)


def test_natural_metric_renames_legacy_preservation():
    rng=np.random.default_rng(1);y=rng.normal(size=(46,256));clean=y-.01*rng.normal(size=y.shape);eog=rng.normal(size=(4,256));teacher=rng.normal(size=y.shape);scale=np.ones(46);metrics=natural_metrics_v28(y,clean,eog,teacher,scale);assert metrics["preservation_legacy"]==metrics["low_eog_observation_retention"] and "preservation" not in metrics


def test_erp_ssvep_aliases_absent():
    rng=np.random.default_rng(2);metrics=natural_metrics_v28(rng.normal(size=(46,64)),rng.normal(size=(46,64)),rng.normal(size=(4,64)),rng.normal(size=(46,64)),np.ones(46));assert "erp_proxy" not in metrics and "ssvep_proxy" not in metrics and metrics["erp_status"]=="unavailable"


def test_natural_artifact_reference_is_corrected_teacher_not_support_operator():
    source=(ROOT/"src/eeg_scad/cli/run_v28.py").read_text();assert 'evaluator["teacher_artifact"][i]' in source and 'evaluator["latent"][i],query["cs"][i]' not in source


def test_current_corrected_observation_is_named_standard_not_raw_like():
    source=(ROOT/"src/eeg_scad/cli/run_v28.py").read_text();assert 'for method in ("STANDARD",*methods)' in source and 'for method in ("STANDARD",*pred)' in source


def test_attenuation_remaining_exact_consistency():
    assert attenuation_consistency(.5,6.020599913279624)<1e-12


def test_event_inventory_na_without_metadata():
    rows=inventory({});assert all(row["status"]=="unavailable" and row["proxy_substitution_forbidden"] for row in rows)


def test_event_inventory_requires_complete_metadata():
    rows={row["outcome"]:row for row in inventory({"event_markers":[1],"trial_boundaries":[0,1],"condition_labels":["a"]})};assert rows["ERP"]["status"]=="supported" and rows["SSVEP"]["status"]=="unavailable"


def test_identity_rows_exclusion_is_registered():
    assert "if identity:metric[\"snr_improvement\"]=np.nan;metric[\"artifact_rrmse\"]=np.nan" in (ROOT/"src/eeg_scad/cli/run_v28.py").read_text()


def test_resume_state_is_complete():
    source=(ROOT/"src/eeg_scad/training/train_v28.py").read_text()
    for key in ("optimizer","scheduler","ema","amp_scaler","data_rng","support_rng","wrong_support_rng","diffusion_rng","stream_rng"):assert f'"{key}"' in source


def test_round_a_main_and_paired_only_checkpoints_are_distinct_routes():
    source=(ROOT/"src/eeg_scad/cli/run_v28.py").read_text();assert 'v=variant(kind)' in source and 'variant("support_cdm_paired" if label=="SUPPORT_CLEAN_CDM"' not in source


def test_query_auxiliary_forbidden_fields():
    assert set(SupportCleanCDM.forbidden_fields)>={"query_EOG","query_operator","query_event"}


def test_bundle_freeze_is_not_treated_as_fold_latency():
    source=(ROOT/"src/eeg_scad/cli/run_v28.py").read_text();assert 'if not {"fold","seed","windows","seconds"}.issubset(row)' in source


def test_governance_config_k1_and_sealed_list():
    import yaml
    data=yaml.safe_load((ROOT/"configs/sc_cdm_v28/data.yaml").read_text());assert data["K"]==1 and len(data["sealed_participants"])==8 and data["development_only"]
