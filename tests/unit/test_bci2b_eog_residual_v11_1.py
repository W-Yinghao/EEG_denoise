import inspect
import numpy as np
from eeg_cgdr.experiments import bci2b_eog_residual_v11_1 as v
def test_primary_crop_excludes_padding():
    x=np.zeros((1,3,512),np.float32);x[...,500:]=100
    f,p=v._welch(x,250,500);assert np.all(p==0) and f[-1]==125
def test_inference_is_evaluator_blind():
    source=inspect.getsource(v.stage_infer)+inspect.getsource(v._infer_k);assert "evaluator.npz" not in source
def test_contract_constants_are_frozen_in_config():
    import yaml
    c=yaml.safe_load(open("configs/cgdr/bci2b_eog_residual_v11_1.yaml"));assert c["crop_length"]==500 and c["primary_paired_band"]==[1,45] and c["primary_natural_band"]==[8,30]
