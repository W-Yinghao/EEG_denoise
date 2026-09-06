from __future__ import annotations
import numpy as np
from scipy import signal


def natural_metrics(y:np.ndarray,predicted_artifact:np.ndarray,eog:np.ndarray,c_query:np.ndarray,eeg_scale:np.ndarray)->dict[str,float]:
    physical_y=y*eeg_scale[:,None];physical_a=predicted_artifact*eeg_scale[:,None];clean=physical_y-physical_a;e=eog-np.mean(eog,axis=1,keepdims=True);field=c_query@e;energy=np.sqrt(np.mean(e*e,axis=0));lo=energy<=np.quantile(energy,.3);hi=energy>=np.quantile(energy,.7)
    remaining=float(np.linalg.norm((field-physical_a)[:,hi])/max(np.linalg.norm(field[:,hi]),1e-12));pres=1-float(np.linalg.norm(physical_a[:,lo])/max(np.linalg.norm(physical_y[:,lo]),1e-12));f,p0=signal.welch(physical_y[:,lo],fs=100,nperseg=min(128,int(np.sum(lo))),axis=-1);_,p1=signal.welch(clean[:,lo],fs=100,nperseg=min(128,int(np.sum(lo))),axis=-1);keep=(f>=1)&(f<=15);psd=float(np.mean(np.abs(np.log(np.maximum(p0[:,keep],1e-10))-np.log(np.maximum(p1[:,keep],1e-10)))))
    cov0=np.cov(physical_y[:,lo]);cov1=np.cov(clean[:,lo]);cov=float(np.linalg.norm(cov1-cov0)/max(np.linalg.norm(cov0),1e-12));atten=float(20*np.log10(max(np.linalg.norm(field[:,hi]),1e-12)/max(np.linalg.norm((field-physical_a)[:,hi]),1e-12)))
    return {"heldout_eog_remaining_ratio":remaining,"artifact_attenuation_db":atten,"preservation":pres,"psd_distortion":psd,"covariance_distortion":cov,"output_input_rms_ratio":float(np.sqrt(np.mean(clean**2))/max(np.sqrt(np.mean(physical_y**2)),1e-12)),"observation_change_ratio":float(np.linalg.norm(physical_a)/max(np.linalg.norm(physical_y),1e-12)),"eeg_eog_coherence_reduction":float(np.linalg.norm(physical_y@e.T)-np.linalg.norm(clean@e.T))/max(np.linalg.norm(physical_y@e.T),1e-12),"frontal_residual_topography":float(np.linalg.norm(np.std((field-physical_a)[:8,hi],axis=1))),"erp_proxy":pres,"ssvep_proxy":pres}

