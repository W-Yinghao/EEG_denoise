import inspect
import numpy as np
from eeg_cgdr.experiments import bci2b_clean_posterior_v12 as v

def test_score_only_identity():
    l=np.ones((2,3,5));d=2*l;f=3*l
    assert np.array_equal(l+(f-d),2*l)

def test_primary_metrics_crop_padding():
    value=np.zeros((1,3,512),np.float32);value[...,500:]=100
    assert np.all(v._features(value)==v._features(value[...,:500]))

def test_score_audit_does_not_train():
    source=inspect.getsource(v.stage_score_audit)
    assert ".fit(" not in source or "LinearDiscriminantAnalysis" not in source

def test_clean_model_forbids_evaluator_fields():
    from eeg_cgdr.models.clean_posterior_diffusion import DeterministicCleanEstimator
    assert "clean_target" in DeterministicCleanEstimator.forbidden_fields
    assert "query_transfer" in DeterministicCleanEstimator.forbidden_fields

def test_v12_inference_is_evaluator_blind():
    source=inspect.getsource(v.stage_infer)
    assert "evaluator.npz" not in source and "paired_x" not in source

def test_training_uses_observation_anchored_state():
    from eeg_cgdr.models import clean_posterior_diffusion as model
    source=inspect.getsource(model.CleanPosteriorDiffusion.training_loss)
    assert "state=alpha.sqrt()*x_lin" in source

def test_v111_report_generator_contains_final_routing():
    from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as old
    source=inspect.getsource(old.stage_finalize)
    assert "Diffusion increment passed" in source and "inference_resource_summary.csv" in source
