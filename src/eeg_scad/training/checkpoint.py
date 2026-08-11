from __future__ import annotations
import copy
import torch
from torch import nn


class EMA:
    def __init__(self,module:nn.Module,decay:float=.999)->None:self.decay=decay;self.shadow={k:v.detach().clone() for k,v in module.state_dict().items()}
    @torch.no_grad()
    def update(self,module:nn.Module)->None:
        for k,v in module.state_dict().items():
            if torch.is_floating_point(v):self.shadow[k].lerp_(v.detach(),1-self.decay)
            else:self.shadow[k].copy_(v)
    def copy_to(self,module:nn.Module)->None:module.load_state_dict(self.shadow)
    def state_dict(self)->dict:return {"decay":self.decay,"shadow":self.shadow}
    def load_state_dict(self,state:dict)->None:self.decay=float(state["decay"]);self.shadow=state["shadow"]


def clone_with_ema(module:nn.Module,ema:EMA)->nn.Module:
    result=copy.deepcopy(module);ema.copy_to(result);return result

