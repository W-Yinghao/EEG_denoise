from __future__ import annotations
import numpy as np
import torch
from eeg_cgdr.experiments.sge_basis_score_factorial_v9 import _basis_batch, _personalization_coverage
from eeg_cgdr.models.artifact_subspace_score_lora import LoRAConv1d

def test_basis_batch_uses_subject_or_population_without_target_leakage()->None:
    geometry={"keys":np.asarray(["p1","p2"]),"population_basis":np.zeros((4,2),np.float32),"population_mask":np.ones(2,bool),"bases":np.stack((np.ones((4,2)),2*np.ones((4,2)))).astype(np.float32),"masks":np.ones((2,2),bool)}
    bases,masks=_basis_batch(np.asarray([0,1]),np.asarray(["p1","p2"]),geometry,np.asarray([True,False]))
    np.testing.assert_array_equal(bases[0],geometry["bases"][0]);np.testing.assert_array_equal(bases[1],geometry["population_basis"]);assert masks.shape==(2,2)

def test_lora_inherits_base_device_and_dtype()->None:
    base=torch.nn.Conv1d(4,8,3,padding=1,dtype=torch.float64)
    lora=LoRAConv1d(base,rank=2)
    assert lora.down.weight.device==base.weight.device
    assert lora.up.weight.dtype==base.weight.dtype
    assert lora(torch.randn(2,4,16,dtype=torch.float64)).shape==(2,8,16)

def test_support_ineligible_fallback_stays_in_coverage_denominator()->None:
    rows=[{"personalization_eligible":1},{"personalization_eligible":0},{"personalization_eligible":1}]
    assert _personalization_coverage(rows,3)==2/3
