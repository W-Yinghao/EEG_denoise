from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from .operator_factorization import decode_numpy,decode_torch


def relative_ridge(basis:np.ndarray,ratio:float)->float:
    gram=np.asarray(basis,dtype=np.float64).T@np.asarray(basis,dtype=np.float64)
    return float(ratio*np.trace(gram)/max(gram.shape[0],1))


def project_numpy(field:np.ndarray,basis:np.ndarray,ratio:float)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    b=np.asarray(basis,np.float64);y=np.asarray(field,np.float64);lam=relative_ridge(b,ratio);q=np.linalg.solve(b.T@b+lam*np.eye(b.shape[1]),b.T@y);projected=decode_numpy(b,q);return q.astype(np.float32),projected.astype(np.float32),(y-projected).astype(np.float32)


def ridge_target_numpy(artifact:np.ndarray,basis:np.ndarray,ratio:float)->tuple[np.ndarray,np.ndarray,float]:
    q,decoded,_=project_numpy(artifact,basis,ratio);error=float(np.linalg.norm(np.asarray(artifact)-decoded)/max(np.linalg.norm(artifact),1e-8));return q,decoded,error


def project_torch(field:Tensor,basis:Tensor,ratio:float)->tuple[Tensor,Tensor,Tensor]:
    gram=torch.einsum("bcd,bce->bde",basis,basis);lam=ratio*torch.diagonal(gram,dim1=1,dim2=2).sum(1)/basis.shape[-1];eye=torch.eye(basis.shape[-1],device=basis.device,dtype=basis.dtype)[None];rhs=torch.einsum("bcd,bct->bdt",basis,field);q=torch.linalg.solve(gram+lam[:,None,None]*eye,rhs);projected=decode_torch(basis,q);return q,projected,field-projected


class CoefficientStandardizer:
    def __init__(self,mean:np.ndarray,std:np.ndarray)->None:self.mean=np.asarray(mean,np.float32);self.std=np.maximum(np.asarray(std,np.float32),1e-6)
    @classmethod
    def fit(cls,values:np.ndarray)->"CoefficientStandardizer":return cls(np.mean(values,axis=(0,2)),np.std(values,axis=(0,2)))
    def transform(self,value:np.ndarray)->np.ndarray:return (value-self.mean[...,None])/self.std[...,None]
    def inverse(self,value:np.ndarray)->np.ndarray:return value*self.std[...,None]+self.mean[...,None]

