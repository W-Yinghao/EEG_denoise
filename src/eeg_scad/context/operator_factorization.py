"""Operator-factorized spatial basis used by OF-SCAD.

The basis is always represented by unit-L2 columns. Column norms are absorbed
into coefficient coordinates, making coefficient targets and spatial decoding
use one auditable coordinate system.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class BasisDiagnostics:
    condition_number:float
    minimum_singular_value:float
    effective_rank:int
    deviation_energy_ratio:float
    projector_distance:float


def factorize_operator(population:np.ndarray,support:np.ndarray,epsilon:float=1e-8)->tuple[np.ndarray,np.ndarray,BasisDiagnostics]:
    c0=np.asarray(population,np.float64);cs=np.asarray(support,np.float64)
    if c0.shape!=cs.shape or c0.ndim!=2:raise ValueError("operator shapes must agree")
    raw=np.concatenate((c0,cs-c0),axis=1);scales=np.linalg.norm(raw,axis=0);basis=raw/np.maximum(scales,epsilon)[None]
    singular=np.linalg.svd(basis,compute_uv=False);rank=int(np.sum(singular>max(singular[0]*1e-7,epsilon))) if singular.size else 0
    gram=basis.T@basis;condition=float(np.linalg.cond(gram+epsilon*np.eye(gram.shape[0])))
    q0,_=np.linalg.qr(c0);qs,_=np.linalg.qr(cs);projector=float(np.linalg.norm(q0@q0.T-qs@qs.T,ord="fro"))
    diagnostic=BasisDiagnostics(condition,float(singular[-1]) if singular.size else 0.,rank,float(np.linalg.norm(cs-c0)/max(np.linalg.norm(c0),epsilon)),projector)
    return basis.astype(np.float32),scales.astype(np.float32),diagnostic


def population_basis(population:np.ndarray,epsilon:float=1e-8)->tuple[np.ndarray,np.ndarray,BasisDiagnostics]:
    return factorize_operator(population,population,epsilon)


def decode_numpy(basis:np.ndarray,coefficients:np.ndarray)->np.ndarray:return np.einsum("cd,dt->ct",basis,coefficients,optimize=True)
def decode_torch(basis:Tensor,coefficients:Tensor)->Tensor:return torch.einsum("bcd,bdt->bct",basis,coefficients)


def absorb_column_scales(scales:np.ndarray,raw_coefficients:np.ndarray)->np.ndarray:
    return np.asarray(scales)[:,None]*np.asarray(raw_coefficients)


def operator_summary(basis:np.ndarray,scales:np.ndarray,epsilon:float=1e-8)->np.ndarray:
    gram=np.asarray(basis).T@np.asarray(basis);singular=np.linalg.svd(np.asarray(basis),compute_uv=False)
    values=np.concatenate((np.log(np.maximum(np.asarray(scales),epsilon)),singular,np.diag(gram)))
    return values.astype(np.float32)

