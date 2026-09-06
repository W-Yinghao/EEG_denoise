"""Corrected V28 natural metrics with explicit observation-retention semantics."""
from __future__ import annotations

import numpy as np
from scipy import signal


def natural_metrics_v28(y:np.ndarray,clean:np.ndarray,eog:np.ndarray,teacher_artifact:np.ndarray,eeg_scale:np.ndarray)->dict[str,float|str]:
    """Score a frozen inference output with evaluator-only corrected targets.

    ``teacher_artifact`` is the V24-corrected standardized field already
    constructed as ``C_query @ Z_e``.  Reconstructing it here with a support
    operator would violate the query/support role contract and bias the
    artifact metric toward MATCH.
    """
    physical_y=y*eeg_scale[:,None];physical_clean=clean*eeg_scale[:,None];correction=physical_y-physical_clean;e=eog-np.mean(eog,axis=1,keepdims=True);field=teacher_artifact*eeg_scale[:,None];energy=np.sqrt(np.mean(e*e,axis=0));lo=energy<=np.quantile(energy,.3);hi=energy>=np.quantile(energy,.7)
    remaining=float(np.linalg.norm((field-correction)[:,hi])/max(np.linalg.norm(field[:,hi]),1e-12));atten=float(-20*np.log10(max(remaining,1e-12)));change=float(np.linalg.norm(correction[:,lo])/max(np.linalg.norm(physical_y[:,lo]),1e-12));retention=1-change
    f,p0=signal.welch(physical_y[:,lo],fs=100,nperseg=min(128,int(np.sum(lo))),axis=-1);_,p1=signal.welch(physical_clean[:,lo],fs=100,nperseg=min(128,int(np.sum(lo))),axis=-1);keep=(f>=1)&(f<=15);psd=float(np.mean(np.abs(np.log(np.maximum(p0[:,keep],1e-10))-np.log(np.maximum(p1[:,keep],1e-10)))));cov0=np.cov(physical_y[:,lo]);cov1=np.cov(physical_clean[:,lo]);cov=float(np.linalg.norm(cov1-cov0)/max(np.linalg.norm(cov0),1e-12))
    return {"heldout_eog_remaining_ratio":remaining,"artifact_attenuation_db":atten,"low_eog_observation_change":change,"low_eog_observation_retention":retention,"preservation_legacy":retention,"psd_distortion":psd,"covariance_distortion":cov,"output_input_rms_ratio":float(np.sqrt(np.mean(physical_clean**2))/max(np.sqrt(np.mean(physical_y**2)),1e-12)),"observation_change_ratio":float(np.linalg.norm(correction)/max(np.linalg.norm(physical_y),1e-12)),"eeg_eog_coherence_reduction":float(np.linalg.norm(physical_y@e.T)-np.linalg.norm(physical_clean@e.T))/max(np.linalg.norm(physical_y@e.T),1e-12),"frontal_residual_topography":float(np.linalg.norm(np.std((field-correction)[:8,hi],axis=1))),"erp_status":"unavailable","ssvep_status":"unavailable","erd_ers_status":"unavailable"}


def attenuation_consistency(remaining:float,attenuation:float)->float:
    return abs(float(attenuation)-float(-20*np.log10(max(remaining,1e-12))))


__all__=["natural_metrics_v28","attenuation_consistency"]
