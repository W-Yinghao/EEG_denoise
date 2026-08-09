import inspect
import numpy as np
from eeg_cgdr.experiments import bci2b_operator_shrinkage as s

def test_task_mapping():
    c={"seeds":[20260808,20260810,20260811]}
    assert s._task(c,0)==(20260808,0)
    assert s._task(c,26)==(20260811,8)

def test_strict_owner_checks_are_explicit():
    source=inspect.getsource(s.stage_prepare)
    assert "value is None" in source
    assert "POP_FALLBACK" in source
    assert "eligible_120" in source

def test_shrinkage_does_not_open_evaluator():
    source=inspect.getsource(s.stage_shrink_infer)
    assert "evaluator.npz" not in source
    assert "paired_x" not in source
    assert "natural_labels" not in source

def test_wrong_uses_recipient_lambda():
    source=inspect.getsource(s.stage_shrink_infer)
    assert 'hp+lam*(np.asarray(inf[f"h_wrong_{d}"])-hp)' in source

def test_tie_prefers_smaller_lambda():
    source=inspect.getsource(s._select_lambda)
    assert "(np.mean(loss[lam]),lam)" in source

def test_exact_sign_flip_known_case():
    assert s._sign_flip(np.ones(9))==2/512

def test_oppost_lambda_zero_is_explicit_identity_dispatch():
    source=inspect.getsource(s.stage_oppost_select)
    assert "strength=0 if late==0 else 1" in source
    source=inspect.getsource(s.bounded_oppost_sample)
    assert "strength==0" in source

def test_oppost_reports_frozen_weight_thresholds():
    source=inspect.getsource(s.stage_oppost_oracle)
    for threshold in ("0_10","0_25","0_50","0_90"):
        assert f"fraction_steps_gt_{threshold}" in source

def test_oppost_inference_does_not_open_evaluator():
    source=inspect.getsource(s.stage_oppost_infer)
    assert "evaluator.npz" not in source
    assert "paired_a" not in source
    assert "_half_support_transfer" in source

def test_bci2a_support_extraction_does_not_open_query():
    source=inspect.getsource(s.stage_bci2a_extract)
    assert "query]" not in source
    assert "load_with_events" in source

def test_bci2a_wrong_uses_recipient_lambda():
    source=inspect.getsource(s.stage_bci2a_evaluate)
    assert "hp+lam*(support[donor]" in source
