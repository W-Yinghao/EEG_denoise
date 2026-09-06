import inspect
from eeg_cgdr.experiments import bci2b_subject_diffusion_replication as v

def test_two_seed_task_mapping():
    c={"new_seeds":[20260810,20260811]}
    assert v._task(c,0)==(20260810,0) and v._task(c,17)==(20260811,8)

def test_inference_stages_are_evaluator_blind():
    for function in (v.stage_infer,v.stage_donor_operators,v.stage_multi_donor_infer):
        source=inspect.getsource(function)
        assert "evaluator.npz" not in source and "paired_x" not in source

def test_donor_operators_are_seed_independent_and_gpu_inference_uses_cache():
    builder=inspect.getsource(v.stage_donor_operators)
    inference=inspect.getsource(v.stage_multi_donor_infer)
    assert "_load_session" in builder
    assert "_load_session" not in inference
    assert "donor_operators" in inference

def test_primary_panel_is_k8_only():
    source=inspect.getsource(v.stage_infer)
    assert "_infer_k" in source and ",fold,8" in source.replace(" ","")
    assert "32" not in source

def test_wrong_is_scored_per_donor():
    source=inspect.getsource(v.stage_aggregate)
    assert "donor_means" in source and "mean_donor_rrmse" in source

def test_participant_first_effects():
    source=inspect.getsource(v.stage_aggregate)
    assert "for subject in range(1,10)" in source
