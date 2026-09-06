"""Diagnostics specific to V29 population-anchored residual adapters."""
from __future__ import annotations
import numpy as np
import torch
from torch import Tensor


def increment_diagnostics(population:Tensor,match:Tensor,wrong:Tensor,mask:Tensor|None=None,projector:Tensor|None=None)->dict[str,float]:
    increment=match-population;wrong_increment=wrong-population;result={"adapter_rms":float(increment.square().mean().sqrt()),"wrong_adapter_rms":float(wrong_increment.square().mean().sqrt()),"match_wrong_output_distance":float((match-wrong).square().mean().sqrt()),"distance_to_population":float(increment.norm()/population.norm().clamp_min(1e-8))}
    if mask is not None:
        result["low_mask_increment_rms"]=float((increment*(1-mask)).square().mean().sqrt());result["high_mask_increment_rms"]=float((increment*mask).square().mean().sqrt())
    if projector is not None:
        q=torch.eye(increment.shape[1],device=increment.device)[None]-projector;result["complement_increment_rms"]=float(torch.einsum("bij,bjt->bit",q,increment).square().mean().sqrt())
    return result


def residual_scale(y:np.ndarray,x:np.ndarray,raw:np.ndarray)->dict[str,float]:
    return {"rms_y":float(np.sqrt(np.mean(y*y))),"rms_x_minus_y":float(np.sqrt(np.mean((x-y)**2))),"rms_raw_network_output":float(np.sqrt(np.mean(raw*raw))),"rms_scaled_network_output":float(np.sqrt(np.mean((.1*raw)**2))),"rms_prediction_minus_y":float(np.sqrt(np.mean((.1*raw)**2)))}


__all__=["increment_diagnostics","residual_scale"]
