from __future__ import annotations
import torch
from torch import Tensor


def ranking_loss(correct:Tensor,wrong:Tensor,target:Tensor,margin:float)->Tensor:
    good=(correct-target).square().flatten(1).mean(1);bad=(wrong-target).square().flatten(1).mean(1);return torch.relu(margin+good-bad).mean()


def zero_identity_loss(prediction:Tensor,target:Tensor)->Tensor:
    mask=target.square().flatten(1).mean(1)<1e-12
    return prediction[mask].square().mean() if torch.any(mask) else prediction.new_zeros(())

