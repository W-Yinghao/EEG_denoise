from __future__ import annotations
import math
import torch
from torch import Tensor


def cosine_alpha_bar(timesteps:int)->Tensor:
    steps=torch.arange(timesteps+1,dtype=torch.float64);f=torch.cos(((steps/timesteps+.008)/(1.008))*math.pi/2)**2;f=f/f[0]
    return f[1:].clamp(1e-7,1).float()


def extract(values:Tensor,timestep:Tensor,ndim:int)->Tensor:return values.gather(0,timestep).reshape(len(timestep),*((1,)*(ndim-1)))

