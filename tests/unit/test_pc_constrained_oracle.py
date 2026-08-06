from __future__ import annotations

import numpy as np

from eeg_cgdr.experiments.pc_constrained_oracle import _candidate_masks, _choose


def test_constrained_oracle_uses_exact_coverage_and_rejects_unsafe_best() -> None:
    benefit=np.asarray([10.0,3.0,2.0,1.0]); risk=np.asarray([10.0,0.0,0.0,0.0])
    def evaluator(mask: np.ndarray):
        unsafe=bool(mask[0])
        return {"artifact_utility":float(benefit[mask].sum()),"preservation_utility":-1.0 if unsafe else 0.0,
                "psd_utility":0.0,"covariance_utility":0.0,"output_input_RMS_ratio":1.0}
    mask,metrics=_choose(benefit,risk,2,evaluator,-0.02)
    assert mask is not None and mask.sum()==2 and not mask[0]
    assert metrics is not None and metrics["artifact_utility"]==5.0


def test_candidate_masks_never_treat_windows_as_statistical_units() -> None:
    masks=_candidate_masks(np.arange(10.0),np.zeros(10),5)
    assert masks and all(mask.shape==(10,) and int(mask.sum())==5 for mask in masks)

