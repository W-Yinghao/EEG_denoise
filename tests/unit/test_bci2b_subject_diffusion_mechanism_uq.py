import numpy as np
from eeg_cgdr.experiments.bci2b_subject_diffusion_mechanism_uq import _crps,_energy,_independent_realizations,_uq_metrics

def test_degenerate_predictive_distribution_has_expected_scores():
    target=np.ones((2,1,4),np.float32);samples=np.repeat(target[None],3,axis=0)
    assert _crps(samples,target)==0
    assert _energy(samples,target)==0
    assert _uq_metrics(samples,target)["mean_rrmse"]==0

def test_clean_carrier_realization_audit():
    clean=np.stack([np.zeros((1,4)),np.zeros((1,4)),np.ones((1,4))])
    result=_independent_realizations(clean)
    assert result["unique_clean_carriers"]==2
    assert result["carriers_with_multiple_contaminations"]==1
    assert result["posterior_calibration_identifiable"]

