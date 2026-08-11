from __future__ import annotations
import numpy as np
from scipy import signal


def _rrmse(target:np.ndarray,prediction:np.ndarray)->float:return float(np.linalg.norm(prediction-target)/max(np.linalg.norm(target),1e-12))
def _corr(target:np.ndarray,prediction:np.ndarray)->float:
    x=target.reshape(-1)-np.mean(target);y=prediction.reshape(-1)-np.mean(prediction);return float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
def _spectral(target:np.ndarray,prediction:np.ndarray,rate:float=100.)->float:
    f,p0=signal.welch(target,fs=rate,nperseg=min(128,target.shape[-1]),axis=-1);_,p1=signal.welch(prediction,fs=rate,nperseg=min(128,prediction.shape[-1]),axis=-1);keep=(f>=1)&(f<=15);return float(np.sqrt(np.mean((np.log(np.maximum(p0[:,keep],1e-10))-np.log(np.maximum(p1[:,keep],1e-10)))**2)))


def paired_metrics(x:np.ndarray,y:np.ndarray,artifact:np.ndarray,predicted_artifact:np.ndarray)->dict[str,float]:
    clean=y-predicted_artifact;noise0=y-x;noise1=clean-x
    return {"rrmse_temporal":_rrmse(x,clean),"rrmse_spectral":_spectral(x,clean),"correlation":_corr(x,clean),"snr_improvement":float(20*np.log10(max(np.linalg.norm(noise0),1e-12)/max(np.linalg.norm(noise1),1e-12))),"artifact_rmse":float(np.sqrt(np.mean((predicted_artifact-artifact)**2))),"artifact_correlation":_corr(artifact,predicted_artifact),"clean_output_rms_ratio":float(np.sqrt(np.mean(clean**2))/max(np.sqrt(np.mean(x**2)),1e-12)),"observation_change_ratio":float(np.linalg.norm(predicted_artifact)/max(np.linalg.norm(y),1e-12))}

