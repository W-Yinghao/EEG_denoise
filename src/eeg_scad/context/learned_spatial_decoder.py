from __future__ import annotations
import torch
from torch import Tensor

def normalize_basis(value:Tensor)->Tensor:return value/torch.linalg.vector_norm(value,dim=1,keepdim=True).clamp_min(1e-8)
def decode_residual(anchor:Tensor,basis:Tensor,coefficient:Tensor)->Tensor:return anchor+torch.einsum("bcr,brt->bct",basis,coefficient)
def ridge_latent(target:Tensor,basis:Tensor,ratio:float=0.01)->Tensor:
    gram=torch.einsum("bcr,bcs->brs",basis,basis);ridge=ratio*torch.diagonal(gram,dim1=1,dim2=2).mean(1);eye=torch.eye(gram.shape[-1],device=gram.device)[None];rhs=torch.einsum("bcr,bct->brt",basis,target);return torch.linalg.solve(gram+ridge[:,None,None]*eye,rhs)
def basis_decorrelation(basis:Tensor)->Tensor:
    gram=torch.einsum("bcr,bcs->brs",basis,basis);eye=torch.eye(gram.shape[-1],device=gram.device)[None];return (gram-eye).square().mean()

__all__=["normalize_basis","decode_residual","ridge_latent","basis_decorrelation"]
