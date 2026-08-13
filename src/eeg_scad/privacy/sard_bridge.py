"""Task-matched cross-subject representation transport for V38P."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from torch import Tensor, nn

from .sandiff import cosine_alpha_bar


def probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values-values.max(1, keepdims=True)); return (values/values.sum(1, keepdims=True)).astype(np.float32)


@dataclass(frozen=True)
class FrozenSemantics:
    tertiles: dict[int, tuple[float, float]]
    means: dict[tuple[int, int], np.ndarray]
    class_means: dict[int, np.ndarray]
    global_mean: np.ndarray

    @classmethod
    def fit(cls, z: np.ndarray, logits: np.ndarray) -> "FrozenSemantics":
        pred, conf = semantics(logits); tertiles = {}; means = {}; class_means = {}
        for task in sorted(np.unique(pred)):
            mask = pred == task; cuts = np.quantile(conf[mask], (1/3, 2/3)); tertiles[int(task)] = tuple(map(float, cuts)); class_means[int(task)] = z[mask].mean(0).astype(np.float32)
            levels = np.digitize(conf[mask], cuts)
            for level in range(3):
                chosen = z[mask][levels == level]; means[(int(task), level)] = (chosen if len(chosen) else z[mask]).mean(0).astype(np.float32)
        return cls(tertiles, means, class_means, np.asarray(z).mean(0).astype(np.float32))

    def assign(self, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred, conf = semantics(logits); levels = np.asarray([np.digitize(value, self.tertiles.get(int(task), (-np.inf, np.inf))) for task, value in zip(pred, conf)], dtype=np.int64); return pred, levels

    def prototype(self, logits: np.ndarray) -> np.ndarray:
        pred, levels = self.assign(logits); return np.stack([self.means.get((int(task), int(level)), self.class_means.get(int(task), self.global_mean)) for task, level in zip(pred, levels)]).astype(np.float32)


def semantics(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prob = probabilities(logits); pred = prob.argmax(1); ordered = np.sort(prob, axis=1); return pred.astype(np.int64), (ordered[:, -1]-ordered[:, -2]).astype(np.float32)


@dataclass(frozen=True)
class ContextNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, contexts: np.ndarray) -> "ContextNormalizer":
        mean = contexts.mean(0); scale = contexts.std(0); scale = np.where(scale > 1e-5, scale, 1.0); return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, context: np.ndarray) -> np.ndarray: return ((context-self.mean)/self.scale).astype(np.float32)


def support_context(z: np.ndarray, logits: np.ndarray, subject: np.ndarray, semantics_model: FrozenSemantics) -> tuple[np.ndarray, np.ndarray]:
    residual = z-semantics_model.prototype(logits); owners = np.unique(subject); contexts = np.stack([residual[subject == owner].mean(0) for owner in owners]); return owners, contexts.astype(np.float32)


@dataclass(frozen=True)
class DonorBank:
    z: np.ndarray
    logits: np.ndarray
    subject: np.ndarray
    predicted: np.ndarray
    level: np.ndarray
    tertiles: dict[int, tuple[float, float]]

    @classmethod
    def fit(cls, z: np.ndarray, logits: np.ndarray, subject: np.ndarray, semantics_model: FrozenSemantics) -> "DonorBank":
        predicted, level = semantics_model.assign(logits); return cls(np.asarray(z, dtype=np.float32), np.asarray(logits, dtype=np.float32), np.asarray(subject, dtype=np.int64), predicted, level, semantics_model.tertiles)

    def sample_indices(self, source_logits: np.ndarray, source_subject: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
        predicted, levels = semantics(source_logits)[0], self._levels(source_logits); result = np.empty(len(source_logits), dtype=np.int64); routes = []
        for index, (task, level, owner) in enumerate(zip(predicted, levels, source_subject)):
            other = self.subject != owner; exact = np.flatnonzero(other & (self.predicted == task) & (self.level == level))
            if len(exact): candidates, route = exact, "exact_stratum"
            else:
                same = np.flatnonzero(other & (self.predicted == task))
                if len(same):
                    distance = np.abs(self.level[same]-level); candidates, route = same[distance == distance.min()], "nearest_stratum"
                else: candidates, route = np.flatnonzero(other), "global_fallback"
            if not len(candidates): raise ValueError("no cross-subject donor")
            result[index] = int(rng.choice(candidates)); routes.append(route)
        return result, routes

    def _levels(self, source_logits: np.ndarray) -> np.ndarray:
        pred, conf = semantics(source_logits); result = np.empty(len(pred), dtype=np.int64)
        for task in np.unique(pred):
            result[pred == task] = np.digitize(conf[pred == task], self.tertiles[int(task)])
        return result

    def sample(self, source_z: np.ndarray, source_logits: np.ndarray, source_subject: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        indices, routes = self.sample_indices(source_logits, source_subject, np.random.default_rng(seed)); return self.z[indices]-source_z, indices, routes

    def sample_many(self, source_z: np.ndarray, source_logits: np.ndarray, source_subject: np.ndarray, count: int, seed: int) -> np.ndarray:
        return np.stack([self.sample(source_z, source_logits, source_subject, seed+index)[0] for index in range(count)])


@dataclass(frozen=True)
class BridgeScaler:
    z_mean: np.ndarray
    z_scale: np.ndarray
    logits_mean: np.ndarray
    logits_scale: np.ndarray
    delta_scale: np.ndarray

    @classmethod
    def fit(cls, z: np.ndarray, logits: np.ndarray, delta: np.ndarray) -> "BridgeScaler":
        def scale(x):
            value=x.std(0); return np.where(value>1e-5,value,1.0).astype(np.float32)
        return cls(z.mean(0).astype(np.float32), scale(z), logits.mean(0).astype(np.float32), scale(logits), scale(delta))

    def condition(self, z: np.ndarray, logits: np.ndarray, context: np.ndarray) -> np.ndarray:
        return np.concatenate(((z-self.z_mean)/self.z_scale, (logits-self.logits_mean)/self.logits_scale, context), axis=1).astype(np.float32)
    def normalize_delta(self, delta: np.ndarray) -> np.ndarray: return (delta/self.delta_scale).astype(np.float32)
    def restore_delta(self, delta: Tensor) -> Tensor: return delta*torch.as_tensor(self.delta_scale, device=delta.device, dtype=delta.dtype)


class OneStepBridge(nn.Module):
    def __init__(self, condition_dim: int = 258, dim: int = 128) -> None:
        super().__init__(); self.net=nn.Sequential(nn.Linear(condition_dim,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,dim))
    def forward(self, condition: Tensor) -> Tensor: return self.net(condition)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int=32) -> None:
        super().__init__(); self.dim=dim; self.net=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Linear(64,dim))
    def forward(self,t:Tensor)->Tensor:
        half=self.dim//2;freq=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/max(half-1,1));a=t.float()[:,None]*freq[None];return self.net(torch.cat((a.sin(),a.cos()),1))


class SARDBridge(nn.Module):
    """Compact x0-prediction diffusion over dynamic donor residuals."""
    def __init__(self, condition_dim: int=258, dim: int=128, timesteps: int=1000) -> None:
        super().__init__(); self.register_buffer("alpha_bar",cosine_alpha_bar(timesteps));self.time=TimeEmbedding(32);self.input=nn.Linear(dim+condition_dim+32,256);self.blocks=nn.ModuleList([nn.Sequential(nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256)) for _ in range(3)]);self.output=nn.Linear(256,dim)
    def forward(self,state:Tensor,condition:Tensor,t:Tensor)->Tensor:
        h=self.input(torch.cat((state,condition,self.time(t)),1))
        for block in self.blocks:h=h+block(h)
        return self.output(h)
    def q_sample(self,x0:Tensor,t:Tensor,noise:Tensor)->Tensor:
        alpha=self.alpha_bar[t].to(x0.dtype)[:,None];return alpha.sqrt()*x0+(1-alpha).sqrt()*noise
    @torch.no_grad()
    def sample(self,condition:Tensor,noise:Tensor,steps:int=10)->Tensor:
        state=noise.clone();schedule=torch.linspace(len(self.alpha_bar)-1,0,steps,device=state.device).round().long()
        for index,tvalue in enumerate(schedule):
            timestep=torch.full((len(state),),int(tvalue),device=state.device,dtype=torch.long);x0=self(state,condition,timestep)
            if index+1==len(schedule):state=x0;continue
            alpha=self.alpha_bar[tvalue].to(state.dtype);next_alpha=self.alpha_bar[schedule[index+1]].to(state.dtype);epsilon=(state-alpha.sqrt()*x0)/(1-alpha).sqrt().clamp_min(1e-8);state=next_alpha.sqrt()*x0+(1-next_alpha).sqrt()*epsilon
        return state


class SourceAdversary(nn.Module):
    def __init__(self, dim:int, subjects:int)->None:
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,subjects))
    def forward(self,z:Tensor)->Tensor:return self.net(z)


@dataclass(frozen=True)
class GaussianBridge:
    coefficient: np.ndarray
    intercept: np.ndarray
    cholesky: dict[tuple[int,int],np.ndarray]
    semantics_model: FrozenSemantics

    @classmethod
    def fit(cls, condition:np.ndarray,target:np.ndarray,source_logits:np.ndarray,semantics_model:FrozenSemantics)->"GaussianBridge":
        ridge=Ridge(alpha=1.).fit(condition,target);residual=target-ridge.predict(condition);pred,level=semantics_model.assign(source_logits);cholesky={}
        for task in sorted(np.unique(pred)):
            for cell in range(3):
                chosen=residual[(pred==task)&(level==cell)];chosen=chosen if len(chosen)>=8 else residual[pred==task];cov=LedoitWolf().fit(chosen).covariance_;scale=max(np.trace(cov)/len(cov),1e-8);cholesky[(int(task),cell)]=np.linalg.cholesky(cov+np.eye(len(cov))*scale*1e-5).astype(np.float32)
        return cls(ridge.coef_.T.astype(np.float32),np.asarray(ridge.intercept_,dtype=np.float32),cholesky,semantics_model)
    def sample_many(self,condition:np.ndarray,source_logits:np.ndarray,count:int,seed:int)->np.ndarray:
        mean=condition@self.coefficient+self.intercept;pred,level=self.semantics_model.assign(source_logits);rng=np.random.default_rng(seed);result=[]
        for _ in range(count):
            values=np.empty_like(mean,dtype=np.float32)
            for i,(task,cell) in enumerate(zip(pred,level)):values[i]=mean[i]+rng.standard_normal(mean.shape[1]).astype(np.float32)@self.cholesky[(int(task),int(cell))].T
            result.append(values)
        return np.stack(result)


__all__=["BridgeScaler","ContextNormalizer","DonorBank","FrozenSemantics","GaussianBridge","OneStepBridge","SARDBridge","SourceAdversary","probabilities","semantics","support_context"]
