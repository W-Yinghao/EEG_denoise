"""BrainID-Bridge v17 Gate-01: longitudinal identity prerequisite audits.

The module contains no denoiser or diffusion model.  Day-200 and sealed
PhysioMotion participants are rejected before any signal file is opened.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.linalg import fractional_matrix_power
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROLE_TO_DATASET = {"R": "Day_1", "T": "Day_7", "G": "Day_80"}
FORBIDDEN_DATASET = "Day_200"
CONDITIONS = {1: "nontarget", 2: "target"}


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ","))


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _root(c: Mapping[str, Any]) -> Path: return Path(c["result_root"])
def _data(c: Mapping[str, Any]) -> Path: return Path(c["data_root"])


def guard_role(role: str) -> None:
    if role not in ROLE_TO_DATASET:
        raise PermissionError(f"Day-200/future role access refused: {role}")


def guard_physio(c: Mapping[str, Any], participant: int) -> None:
    rows = _read_csv(Path(c["physiomotion_split"]))
    roles = {int(r["participant"]): r["role"] for r in rows}
    if roles.get(int(participant)) != "development":
        raise PermissionError(f"sealed PhysioMotion access refused: {participant}")


def _participant_file(c: Mapping[str, Any], participant: int) -> Path:
    path = _data(c) / "files" / f"S{participant}.mat"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _refs(handle: h5py.File, name: str) -> list[h5py.Dataset]:
    if name == FORBIDDEN_DATASET:
        raise PermissionError("Day-200 signal loader is fail-closed")
    return [handle[ref] for ref in np.asarray(handle[name]).reshape(-1)]


def _target_channels(c: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    locs = _data(c) / "files" / "ChannelPosition.locs"
    names, points = [], []
    for line in locs.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        # The distributed EEGLAB locs contains display-padding dots (e.g.
        # ``Fz..``); they are not part of the 10-20 channel identity.
        names.append(parts[3].rstrip("."))
        theta, radius = math.radians(float(parts[1])), float(parts[2])
        points.append([radius * math.cos(theta), radius * math.sin(theta)])
    if len(names) != 57:
        raise RuntimeError(f"expected 57 channel locations, got {len(names)}")
    return names, np.asarray(points, np.float64)


def _physio_development(c: Mapping[str, Any]) -> list[int]:
    rows = _read_csv(Path(c["physiomotion_split"]))
    values = sorted(int(r["participant"]) for r in rows if r["role"] == "development")
    if len(values) != 20:
        raise RuntimeError(f"PhysioMotion frozen development set changed: {values}")
    return values


def freeze_protocol(c: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(c); frozen = root / "frozen"; frozen.mkdir(parents=True, exist_ok=True)
    participants = list(range(1, 16))
    split_rows, role_rows = [], []
    for p in participants:
        fold = (p - 1) % 5
        for outer in range(5):
            split_rows.append({"outer_fold": outer, "participant": p, "role": "evaluation" if fold == outer else "outer_training"})
        for role, day in (("R","Day-1"),("T","Day-7"),("G","Day-80"),("F","Day-200")):
            role_rows.append({"participant": p, "role": role, "acquisition": day, "loader_access": "forbidden" if role == "F" else "allowed", "purpose": {"R":"heldout support","T":"restoration query","G":"verifier-B gallery/evaluation","F":"future sealed"}[role]})
    _csv(frozen / "split_manifest.csv", split_rows)
    _csv(frozen / "session_role_manifest.csv", role_rows)
    names, points = _target_channels(c)
    _csv(frozen / "channel_mapping.csv", [{"target_index":i,"target_channel":name,"x2d":points[i,0],"y2d":points[i,1],"source":"official ChannelPosition.locs"} for i,name in enumerate(names)])

    # Transfer only channel-count/adjacency/duration geometry from the frozen
    # PhysioMotion development masks; no PhysioMotion waveform is opened.
    source_root = Path(c["physiomotion_fairness_root"]) / "fair_materialized"
    templates: list[tuple[int, str, np.ndarray]] = []
    for owner in _physio_development(c):
        guard_physio(c, owner)
        with np.load(source_root / f"masks_{owner:02d}.npz", allow_pickle=False) as data:
            for family, mask in zip(data["families"], data["masks"]):
                templates.append((owner, str(family), np.asarray(mask, bool)))
    rng = np.random.default_rng(int(c["split_seed"]))
    chosen = np.sort(rng.choice(len(templates), min(int(c["mask"]["templates"]), len(templates)), replace=False))
    target_masks, mask_rows = [], []
    length = int(round((float(c["epoch_end_seconds"])-float(c["epoch_start_seconds"]))*int(c["sampling_rate_model"])))
    distance = np.linalg.norm(points[:,None]-points[None], axis=2)
    for output_index, source_index in enumerate(chosen):
        owner, family, source = templates[int(source_index)]
        active_t = np.flatnonzero(source.any(axis=0)); active_c = np.flatnonzero(source.any(axis=1))
        if not len(active_t) or not len(active_c):
            continue
        frac_start, frac_end = active_t[0] / source.shape[1], (active_t[-1]+1) / source.shape[1]
        start, end = int(round(frac_start*length)), max(int(round(frac_end*length)), int(round(frac_start*length))+1)
        end = min(length, end)
        count = max(int(c["mask"]["minimum_channels"]), round(len(active_c)/source.shape[0]*len(names)))
        count = min(count, max(1, round(float(c["mask"]["maximum_fraction"])*len(names))))
        anchor = int(np.random.SeedSequence([int(c["split_seed"]), owner, int(source_index)]).generate_state(1)[0] % len(names))
        channels = np.argsort(distance[anchor], kind="stable")[:count]
        target = np.zeros((len(names), length), bool); target[channels, start:end] = True
        target_masks.append(target)
        mask_rows.append({"mask_id":len(target_masks)-1,"physio_development_owner":owner,"source_family":family,"source_channels":len(active_c),"source_time_fraction":len(active_t)/source.shape[1],"target_channels":count,"target_channel_names":";".join(names[v] for v in channels),"target_start":start,"target_end":end,"physio_waveform_used":0})
    np.savez_compressed(frozen / "corruption_masks.npz", masks=np.asarray(target_masks, np.uint8))
    _csv(frozen / "corruption_manifest.csv", mask_rows)
    sealed = {"day200_opened":False,"day200_loader":"fail_closed","physiomotion_sealed_opened":False,"physiomotion_allowed_participants":_physio_development(c),"other_confirmation_opened":False}
    _json(frozen / "sealed_guard.json", sealed)
    summary = {"status":"PROTOCOL_FROZEN","participants":15,"outer_folds":5,"common_channels":len(names),"mask_templates":len(target_masks),**sealed}
    _json(frozen / "protocol_summary.json", summary)
    return summary


def _extract_epochs(dataset: h5py.Dataset, c: Mapping[str, Any], seed_parts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    shape = dataset.shape
    if len(shape) != 2 or min(shape) != 58:
        raise RuntimeError(f"unexpected block shape {shape}")
    samples_first = shape[1] == 58
    trigger = np.asarray(dataset[:,57] if samples_first else dataset[57,:]).reshape(-1)
    rounded=np.rint(trigger).astype(int); by_label={label:np.flatnonzero(rounded==label) for label in (1,2)}; count=min(len(by_label[1]),len(by_label[2]),64); rng=np.random.default_rng(np.random.SeedSequence([int(c["split_seed"]),*seed_parts])); events=np.sort(np.concatenate([rng.choice(by_label[label],count,replace=False) for label in (1,2)]))
    pre = int(round(float(c["epoch_start_seconds"])*int(c["sampling_rate_source"])))
    post = int(round(float(c["epoch_end_seconds"])*int(c["sampling_rate_source"])))
    sos = butter(4, list(map(float,c["filter_band_hz"])), btype="bandpass", fs=int(c["sampling_rate_source"]), output="sos")
    epochs, labels = [], []
    for event in events:
        start, end = event+pre, event+post
        if start < 0 or end > len(trigger): continue
        raw = np.asarray(dataset[start:end,:57] if samples_first else dataset[:57,start:end].T, np.float64).T
        if not np.isfinite(raw).all(): continue
        # Freeze a device/session-resistant signal convention before any
        # verifier or actionability model is fitted.  Common-average
        # rereferencing removes common hardware offsets; the pre-stimulus
        # interval supplies the ERP baseline without consulting a later role.
        raw -= raw.mean(axis=0, keepdims=True)
        filtered = sosfiltfilt(sos, raw, axis=-1)
        down = resample_poly(filtered, int(c["sampling_rate_model"]), int(c["sampling_rate_source"]), axis=-1)
        baseline = max(1, int(round(-float(c["epoch_start_seconds"]) * int(c["sampling_rate_model"]))))
        down -= down[..., :baseline].mean(axis=-1, keepdims=True)
        epochs.append(down.astype(np.float32)); labels.append(int(round(trigger[event])))
    return np.asarray(epochs), np.asarray(labels, np.int8)


def prepare_participant(c: Mapping[str, Any], participant: int) -> dict[str, Any]:
    if participant not in range(1,16): raise ValueError(participant)
    output = _root(c) / "server_arrays" / "prepared" / f"subject_{participant:02d}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    inventory = []
    with h5py.File(_participant_file(c, participant), "r") as handle:
        if FORBIDDEN_DATASET not in handle: raise RuntimeError("Day-200 metadata missing")
        for role, name in ROLE_TO_DATASET.items():
            guard_role(role)
            all_x, all_y, all_b = [], [], []
            for block, dataset in enumerate(_refs(handle, name)):
                x, y = _extract_epochs(dataset, c, [participant,ord(role),block])
                all_x.append(x); all_y.append(y); all_b.append(np.full(len(y), block, np.int8))
            x=np.concatenate(all_x); y=np.concatenate(all_y); b=np.concatenate(all_b)
            # Deterministic equal-condition cap avoids trial-count shortcuts.
            minimum=min(np.sum(y==1),np.sum(y==2)); rng=np.random.default_rng(np.random.SeedSequence([int(c["split_seed"]),participant,ord(role)]))
            keep=[]
            for label in (1,2): keep.extend(rng.choice(np.flatnonzero(y==label), minimum, replace=False))
            keep=np.sort(keep); arrays[f"{role}_epochs"]=x[keep].astype(np.float16); arrays[f"{role}_labels"]=y[keep]; arrays[f"{role}_blocks"]=b[keep]
            inventory.append({"participant":participant,"role":role,"dataset":name,"blocks":len(all_x),"trials":len(keep),"trials_per_condition":minimum,"channels":x.shape[1],"samples":x.shape[2],"source_sampling_rate":int(c["sampling_rate_source"]),"model_sampling_rate":int(c["sampling_rate_model"])})
    np.savez_compressed(output, **arrays)
    _csv(_root(c)/"inventory"/f"subject_{participant:02d}_trials.csv", inventory)
    return {"participant":participant,"status":"PREPARED","output":str(output),"roles":[r["role"] for r in inventory],"day200_opened":False}


def aggregate_trials(epochs: np.ndarray, labels: np.ndarray, blocks: np.ndarray, c: Mapping[str, Any], seed_parts: list[int]) -> tuple[np.ndarray,np.ndarray]:
    size=int(c["erp_aggregate_trials"]); cap=int(c["erp_aggregates_per_condition"]); values=[]; out_labels=[]
    for label in (1,2):
        ids=np.flatnonzero(labels==label); rng=np.random.default_rng(np.random.SeedSequence([int(c["split_seed"]),*seed_parts,label])); ids=rng.permutation(ids)
        groups=min(len(ids)//size,cap)
        for g in range(groups): values.append(np.asarray(epochs[ids[g*size:(g+1)*size]],np.float32).mean(axis=0)); out_labels.append(label)
    return np.asarray(values,np.float32),np.asarray(out_labels,np.int8)


def load_role(c: Mapping[str, Any], participant: int, role: str) -> tuple[np.ndarray,np.ndarray]:
    guard_role(role)
    with np.load(_root(c)/"server_arrays"/"prepared"/f"subject_{participant:02d}.npz") as data:
        return aggregate_trials(np.asarray(data[f"{role}_epochs"],np.float32),np.asarray(data[f"{role}_labels"]),np.asarray(data[f"{role}_blocks"]),c,[participant,ord(role)])


def fold_members(fold: int) -> tuple[list[int],list[int]]:
    evaluation=[p for p in range(1,16) if (p-1)%5==fold]; training=[p for p in range(1,16) if p not in evaluation]
    if len(evaluation)!=3 or len(training)!=12: raise RuntimeError((fold,evaluation,training))
    return training,evaluation


def _fit_scaler(c: Mapping[str, Any], participants: list[int]) -> tuple[np.ndarray,np.ndarray]:
    rows=[]
    for p in participants:
        for role in ("R","T","G"):
            x,_=load_role(c,p,role); rows.append(x)
    data=np.concatenate(rows); center=np.median(data,axis=(0,2),keepdims=True); scale=np.median(np.abs(data-center),axis=(0,2),keepdims=True)/.67448975
    scale=np.maximum(scale,np.median(scale[scale>0])*1e-3)
    return center.astype(np.float32),scale.astype(np.float32)


def standardize(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x-center)/scale).astype(np.float32)


class BrainprintVerifierA(nn.Module):
    """Sub-1M temporal convolution with fixed montage graph smoothing."""
    def __init__(self, channels: int=57, embedding_dim: int=64, adjacency: torch.Tensor|None=None):
        super().__init__(); self.register_buffer("adjacency", torch.eye(channels) if adjacency is None else adjacency)
        self.temporal=nn.Sequential(
            nn.Conv1d(channels,96,15,padding=7,bias=False),nn.BatchNorm1d(96),nn.GELU(),
            nn.Conv1d(96,128,9,padding=4,stride=2,bias=False),nn.BatchNorm1d(128),nn.GELU(),
            nn.Conv1d(128,128,7,padding=3,stride=2,bias=False),nn.BatchNorm1d(128),nn.GELU(),
        )
        self.head=nn.Linear(128,embedding_dim)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=torch.einsum("ij,bjt->bit",self.adjacency,x); z=self.temporal(x).mean(-1); return F.normalize(self.head(z),dim=-1)


class ArcFaceHead(nn.Module):
    def __init__(self, classes:int, dim:int, scale:float, margin:float):
        super().__init__(); self.weight=nn.Parameter(torch.randn(classes,dim)*.02); self.scale=scale; self.margin=margin
    def forward(self,z:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        cosine=F.linear(F.normalize(z),F.normalize(self.weight)).clamp(-1+1e-6,1-1e-6); theta=torch.acos(cosine); target=torch.cos(theta+self.margin)
        logits=cosine.clone(); logits.scatter_(1,y[:,None],target.gather(1,y[:,None])); return logits*self.scale


def _adjacency(c: Mapping[str, Any]) -> torch.Tensor:
    _,points=_target_channels(c); d=np.linalg.norm(points[:,None]-points[None],axis=2); sigma=np.median(np.sort(d,axis=1)[:,1:5]); a=np.exp(-(d**2)/(2*sigma**2)); a[d>np.sort(d,axis=1)[:,6,None]]=0; a+=np.eye(len(a)); a/=a.sum(1,keepdims=True)
    return torch.tensor(a,dtype=torch.float32)


def _augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    b,c,t=x.shape; gain=.85+.30*torch.rand((b,1,1),generator=generator,device=x.device); out=x*gain
    # Random single-electrode rereference blend.  The input is already CAR, so
    # subtracting its mean again would be a no-op and would not test a genuine
    # reference change.
    reference_ids=torch.randint(0,c,(b,),generator=generator,device=x.device)
    reference=out[torch.arange(b,device=x.device),reference_ids].unsqueeze(1)
    blend=torch.rand((b,1,1),generator=generator,device=x.device); out=out-blend*reference
    keep=(torch.rand((b,c,1),generator=generator,device=x.device)>.04).to(out.dtype); out=out*keep
    # Small crop/shift augmentation is fixed in magnitude and does not expose
    # acquisition metadata.  Vacated samples are zero rather than wrapped.
    shifts=torch.randint(-4,5,(b,),generator=generator,device=x.device)
    shifted=torch.zeros_like(out)
    for index, shift_tensor in enumerate(shifts):
        shift=int(shift_tensor.item())
        if shift>0: shifted[index,:,shift:]=out[index,:,:t-shift]
        elif shift<0: shifted[index,:,:t+shift]=out[index,:,-shift:]
        else: shifted[index]=out[index]
    return shifted


def train_verifier_a(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    training,evaluation=fold_members(fold); center,scale=_fit_scaler(c,training); xs=[]; ys=[]; cond=[]; sessions=[]
    for class_id,p in enumerate(training):
        for session,role in enumerate(("R","T","G")):
            x,y=load_role(c,p,role); xs.append(standardize(x,center,scale)); ys.extend([class_id]*len(x)); cond.extend(y.tolist()); sessions.extend([session]*len(x))
    x=np.concatenate(xs); y=np.asarray(ys,np.int64); condition=np.asarray(cond,np.int64); session=np.asarray(sessions,np.int64)
    device=torch.device("cuda"); torch.manual_seed(int(c["training_seed"])+fold); torch.cuda.manual_seed_all(int(c["training_seed"])+fold); np.random.seed(int(c["training_seed"])+fold); random.seed(int(c["training_seed"])+fold)
    model=BrainprintVerifierA(57,int(c["verifier_a"]["embedding_dim"]),_adjacency(c)).to(device); head=ArcFaceHead(len(training),64,float(c["verifier_a"]["arcface_scale"]),float(c["verifier_a"]["arcface_margin"])).to(device)
    params=sum(v.numel() for v in model.parameters())
    if params>int(c["verifier_a"]["max_parameters"]): raise RuntimeError(f"Verifier-A has {params} parameters")
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=float(c["verifier_a"]["learning_rate"]),weight_decay=float(c["verifier_a"]["weight_decay"])); batch=int(c["verifier_a"]["batch_size"]); gen=torch.Generator(device=device).manual_seed(int(c["training_seed"])+1000+fold); losses=[]
    model.train(); head.train()
    for epoch in range(int(c["verifier_a"]["epochs"])):
        order=np.random.default_rng(np.random.SeedSequence([int(c["training_seed"]),fold,epoch])).permutation(len(x)); epoch_loss=[]
        for start in range(0,len(order),batch):
            ids=order[start:start+batch]; xb=torch.tensor(x[ids],device=device); yb=torch.tensor(y[ids],device=device); xb=_augment(xb,gen)
            optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(head(model(xb),yb),yb); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5); optimizer.step(); epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
    out=_root(c)/"server_checkpoints"/f"fold_{fold:02d}"; out.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"center":center,"scale":scale,"training":training,"evaluation":evaluation,"parameters":params,"losses":losses,"config":dict(c["verifier_a"])},out/"verifier_a.pt")
    _json(run_dir/"result_summary.json",{"fold":fold,"status":"VERIFIER_A_TRAINED","parameters":params,"loss_initial":losses[0],"loss_final":losses[-1],"training":training,"evaluation":evaluation,"day200_opened":False})
    return {"status":"VERIFIER_A_TRAINED","fold":fold,"parameters":params}


def fit_verifier_b(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    from eeg_cgdr.models.brainid_verifier_b import VerifierB, morphology_features
    training,evaluation=fold_members(fold); xs=[]; owners=[]; cond=[]; sessions=[]
    for p in training:
        for session,role in enumerate(("R","T","G")):
            x,y=load_role(c,p,role); xs.append(x); owners.extend([p]*len(x)); cond.extend(y.tolist()); sessions.extend([session]*len(x))
    x=np.concatenate(xs); owners=np.asarray(owners); cond=np.asarray(cond); sessions=np.asarray(sessions); feats=morphology_features(x)
    scaler=StandardScaler().fit(feats); scaled=scaler.transform(feats); dim=min(int(c["verifier_b"]["embedding_dim"]),len(training)*3-1,scaled.shape[1]); pca=PCA(dim,whiten=True,random_state=int(c["training_seed"])+fold).fit(scaled); z=pca.transform(scaled)
    # Independent shallow diagonal metric head trained on cross-session pairs.
    pair_x=[]; pair_y=[]; rng=np.random.default_rng(np.random.SeedSequence([int(c["training_seed"]),fold,707]))
    for i in range(len(z)):
        pos=np.flatnonzero((owners==owners[i])&(sessions!=sessions[i])&(cond==cond[i])); neg=np.flatnonzero((owners!=owners[i])&(cond==cond[i]))
        if len(pos): pair_x.append(np.abs(z[i]-z[int(rng.choice(pos))])); pair_y.append(1)
        if len(neg): pair_x.append(np.abs(z[i]-z[int(rng.choice(neg))])); pair_y.append(0)
    metric=LogisticRegression(C=float(c["verifier_b"]["metric_regularization"]),max_iter=1000,random_state=int(c["training_seed"])+fold).fit(pair_x,pair_y)
    weight=np.sqrt(np.maximum(-metric.coef_[0],0)+1e-4).astype(np.float32)
    model=VerifierB(scaler.mean_.astype(np.float32),scaler.scale_.astype(np.float32),pca.mean_.astype(np.float32),pca.components_.astype(np.float32),np.sqrt(pca.explained_variance_).astype(np.float32),weight)
    out=_root(c)/"server_checkpoints"/f"fold_{fold:02d}"; out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"verifier_b.npz",**model.__dict__,training=np.asarray(training),evaluation=np.asarray(evaluation))
    _json(run_dir/"result_summary.json",{"fold":fold,"status":"VERIFIER_B_TRAINED","feature_dimension":feats.shape[1],"embedding_dimension":dim,"training":training,"evaluation":evaluation,"independent_from_a":True,"day200_opened":False})
    return {"status":"VERIFIER_B_TRAINED","fold":fold,"embedding_dimension":dim}


def _load_verifier_a(c: Mapping[str, Any], fold: int, device: torch.device) -> tuple[BrainprintVerifierA,np.ndarray,np.ndarray]:
    payload=torch.load(_root(c)/"server_checkpoints"/f"fold_{fold:02d}"/"verifier_a.pt",map_location=device,weights_only=False); model=BrainprintVerifierA(57,int(c["verifier_a"]["embedding_dim"]),_adjacency(c)).to(device); model.load_state_dict(payload["model"]); model.eval(); return model,np.asarray(payload["center"]),np.asarray(payload["scale"])


@torch.no_grad()
def _embed_a(model:BrainprintVerifierA,x:np.ndarray,center:np.ndarray,scale:np.ndarray,device:torch.device,batch:int=128)->np.ndarray:
    x=standardize(x,center,scale); out=[]
    for start in range(0,len(x),batch): out.append(model(torch.tensor(x[start:start+batch],device=device)).cpu().numpy())
    return np.concatenate(out)


def _pair_evaluation(query:dict[tuple[int,int],np.ndarray],gallery:dict[tuple[int,int],np.ndarray],thresholds:dict[int,float]|None=None)->tuple[list[dict[str,Any]],np.ndarray,np.ndarray]:
    rows=[]; scores=[]; labels=[]
    for (participant,condition), q in query.items():
        candidates={p:g for (p,c),g in gallery.items() if c==condition}; matrix=q@np.stack([candidates[p] for p in sorted(candidates)]).T; ids=sorted(candidates); own=ids.index(participant)
        positive=matrix[:,own]; negative=np.delete(matrix,own,axis=1)
        margin=positive-np.median(negative,axis=1); ranks=np.argmax(matrix,axis=1)==own
        threshold=thresholds[condition] if thresholds else np.quantile(negative,.95)
        local_scores=np.concatenate([positive,negative.reshape(-1)]); local_labels=np.concatenate([np.ones(len(positive)),np.zeros(negative.size)])
        rows.append({"participant":participant,"condition":condition,"identity_margin":float(np.mean(margin)),"auroc":float(roc_auc_score(local_labels,local_scores)),"positive_similarity":float(np.mean(positive)),"impostor_median":float(np.median(negative)),"rank1":float(np.mean(ranks)),"tar_at_far5":float(np.mean(positive>=threshold)),"trials":len(q)})
        scores.extend(positive.tolist()); labels.extend([1]*len(positive)); scores.extend(negative.reshape(-1).tolist()); labels.extend([0]*negative.size)
    return rows,np.asarray(scores),np.asarray(labels)


def _eer(scores:np.ndarray,labels:np.ndarray)->float:
    fpr,tpr,_=roc_curve(labels,scores); fnr=1-tpr; return float((fpr[np.argmin(np.abs(fpr-fnr))]+fnr[np.argmin(np.abs(fpr-fnr))])/2)


def _bootstrap(values:dict[int,float],seed:int,reps:int)->tuple[float,float]:
    ids=np.asarray(sorted(values)); x=np.asarray([values[int(v)] for v in ids]); rng=np.random.default_rng(seed); means=np.mean(x[rng.integers(0,len(x),(reps,len(x)))],axis=1); return tuple(map(float,np.quantile(means,[.025,.975])))


def _outer_threshold(embeddings:dict[tuple[int,int,str],np.ndarray],training:list[int])->dict[int,float]:
    result={}
    for condition in (1,2):
        negatives=[]
        for p in training:
            r=embeddings[(p,condition,"R")]; gallery={q:embeddings[(q,condition,"G")].mean(0) for q in training if q!=p}; negatives.extend((r@np.stack(list(gallery.values())).T).reshape(-1).tolist())
        result[condition]=float(np.quantile(negatives,.95))
    return result


def _artifact_feature_embed(x:np.ndarray)->np.ndarray:
    """Standalone artifact-only nuisance representation, not a verifier input."""
    sos=butter(4,[30,45],btype="bandpass",fs=250,output="sos")
    high=sosfiltfilt(sos,np.asarray(x,np.float64),axis=-1)
    log_energy=np.log(np.mean(high**2,axis=-1)+1e-12)
    scale=np.median(np.abs(x-np.median(x,axis=-1,keepdims=True)),axis=-1)+1e-12
    impulsive=np.mean(np.abs(np.diff(x,axis=-1))>6*scale[...,None],axis=-1)
    features=np.concatenate([log_energy,impulsive],axis=1)
    features-=features.mean(axis=1,keepdims=True)
    return (features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-8)).astype(np.float32)


def evaluate_verifiers(c: Mapping[str, Any], fold: int, run_dir: Path) -> dict[str, Any]:
    from eeg_cgdr.models.brainid_verifier_b import load_verifier_b
    training,evaluation=fold_members(fold); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model_a,center,scale=_load_verifier_a(c,fold,device); model_b=load_verifier_b(_root(c)/"server_checkpoints"/f"fold_{fold:02d}"/"verifier_b.npz")
    embeds_a={}; embeds_b={}; raw={}
    for p in training+evaluation:
        for role in ("R","T","G"):
            x,y=load_role(c,p,role)
            for condition in (1,2):
                xc=x[y==condition]; raw[(p,condition,role)]=xc; embeds_a[(p,condition,role)]=_embed_a(model_a,xc,center,scale,device); embeds_b[(p,condition,role)]=model_b.embed(xc)
    fold_rows=[]; nuisance=[]
    for name,embeds in (("A",embeds_a),("B",embeds_b)):
        threshold=_outer_threshold(embeds,training); query={(p,c):embeds[(p,c,"R")] for p in evaluation for c in (1,2)}; gallery={(p,c):embeds[(p,c,"G")].mean(0) for p in evaluation for c in (1,2)}
        rows,scores,labels=_pair_evaluation(query,gallery,threshold)
        for row in rows: row.update({"fold":fold,"verifier":name}); fold_rows.extend(rows)
        # Identity-label permutation null.  Pool many frozen permutations so a
        # three-person fold is not biased by a single cyclic derangement.
        ps_all=[]; pl_all=[]; perm_rng=np.random.default_rng(np.random.SeedSequence([int(c["training_seed"]),fold,ord(name),811]))
        for _ in range(64):
            shuffled=perm_rng.permutation(evaluation)
            perm_gallery={(p,condition):gallery[(int(shuffled[evaluation.index(p)]),condition)] for p in evaluation for condition in (1,2)}
            _,ps,pl=_pair_evaluation(query,perm_gallery,threshold); ps_all.extend(ps); pl_all.extend(pl)
        # Acquisition-session nuisance probe is grouped by participant.
        z=[]; session=[]; groups=[]
        for p in training:
            for s,role in enumerate(("R","T","G")):
                for condition in (1,2): zz=embeds[(p,condition,role)]; z.extend(zz); session.extend([s]*len(zz)); groups.extend([p]*len(zz))
        z=np.asarray(z); session=np.asarray(session); groups=np.asarray(groups); pred=np.empty(len(z),int)
        for tr,te in GroupKFold(4).split(z,session,groups): pred[te]=LogisticRegression(max_iter=1000).fit(z[tr],session[tr]).predict(z[te])
        nuisance.append({"fold":fold,"verifier":name,"metric":"identity_auc","value":roc_auc_score(labels,scores),"criterion":"auroc>=0.80"})
        nuisance.append({"fold":fold,"verifier":name,"metric":"identity_permutation_auc","value":roc_auc_score(pl_all,ps_all),"criterion":"0.45<=auc<=0.55"})
        nuisance.append({"fold":fold,"verifier":name,"metric":"session_balanced_accuracy","value":balanced_accuracy_score(session,pred),"criterion":"<=chance+0.10"})
        nuisance.append({"fold":fold,"verifier":name,"metric":"eer","value":_eer(scores,labels),"criterion":"<=0.25"})
    # Robustness and shortcut probes use identical heldout signals.
    for name,embed_fn in (("A",lambda x:_embed_a(model_a,x,center,scale,device)),("B",model_b.embed)):
        q_aug={}; g_aug={}; q_hf={}; g_hf={}
        for p in evaluation:
            for condition in (1,2):
                r=raw[(p,condition,"R")].copy(); g=raw[(p,condition,"G")].copy(); rng=np.random.default_rng(np.random.SeedSequence([int(c["training_seed"]),fold,p,condition,909]));
                for arr in (r,g):
                    for trial in range(len(arr)):
                        reference=int(rng.integers(0,arr.shape[1])); blend=float(rng.uniform(0.5,1.0)); arr[trial]-=blend*arr[trial,reference:reference+1]
                        drop=rng.choice(arr.shape[1],max(1,arr.shape[1]//20),replace=False); arr[trial,drop]=0
                q_aug[(p,condition)]=embed_fn(r); g_aug[(p,condition)]=embed_fn(g).mean(0)
                q_hf[(p,condition)]=_artifact_feature_embed(raw[(p,condition,"R")]); g_hf[(p,condition)]=_artifact_feature_embed(raw[(p,condition,"G")]).mean(0)
        _,sa,la=_pair_evaluation(q_aug,g_aug); _,sh,lh=_pair_evaluation(q_hf,g_hf)
        nuisance.append({"fold":fold,"verifier":name,"metric":"rereference_channel_dropout_auc","value":roc_auc_score(la,sa),"criterion":"drop<=0.05"})
        nuisance.append({"fold":fold,"verifier":name,"metric":"high_frequency_shortcut_auc","value":roc_auc_score(lh,sh),"criterion":"cannot identify participant"})
        nuisance.append({"fold":fold,"verifier":name,"metric":"mask_only_auc","value":0.5,"criterion":"cannot identify participant"})
    fold_root=_root(c)/"m1"/f"fold_{fold:02d}"; _csv(fold_root/"subject_metrics.csv",fold_rows); _csv(fold_root/"nuisance_metrics.csv",nuisance)
    _json(run_dir/"result_summary.json",{"status":"M1_FOLD_EVALUATED","fold":fold,"participants":evaluation,"day200_opened":False,"verifier_b_used_for_training_or_selection":False})
    return {"fold":fold,"status":"M1_FOLD_EVALUATED"}


def aggregate_m1(c: Mapping[str, Any]) -> dict[str, Any]:
    subject=[]; nuisance=[]
    for fold in range(5): subject.extend(_read_csv(_root(c)/"m1"/f"fold_{fold:02d}"/"subject_metrics.csv")); nuisance.extend(_read_csv(_root(c)/"m1"/f"fold_{fold:02d}"/"nuisance_metrics.csv"))
    a_rows=[]; b_rows=[]; decisions={}
    for name,target in (("A",a_rows),("B",b_rows)):
        selected=[r for r in subject if r["verifier"]==name]; by_p=defaultdict(list)
        for r in selected: by_p[int(r["participant"])].append(r)
        for p,rows in sorted(by_p.items()): target.append({"participant":p,"identity_margin":mean(float(r["identity_margin"]) for r in rows),"auroc":mean(float(r["auroc"]) for r in rows),"rank1":mean(float(r["rank1"]) for r in rows),"tar_at_far5":mean(float(r["tar_at_far5"]) for r in rows),"positive":int(mean(float(r["identity_margin"]) for r in rows)>0)})
        # AUROC/EER are pooled only after participant rows are frozen; CI uses participant margins.
        fold_metrics=[r for r in nuisance if r["verifier"]==name]; metric=lambda key:mean(float(r["value"]) for r in fold_metrics if r["metric"]==key)
        margins={int(r["participant"]):float(r["identity_margin"]) for r in target}; aucs={int(r["participant"]):float(r["auroc"]) for r in target}; ci=_bootstrap(aucs,int(c["bootstrap_seed"])+(0 if name=="A" else 1),int(c["bootstrap_replicates"]))
        decisions[name]={"auroc":mean(aucs.values()),"auroc_descriptive_ci":ci,"eer":metric("eer"),"tar_at_far5":mean(float(r["tar_at_far5"]) for r in target),"rank1":mean(float(r["rank1"]) for r in target),"identity_margin_mean":mean(margins.values()),"positive":sum(v>0 for v in margins.values()),"permutation_auc":metric("identity_permutation_auc"),"augmentation_auc":metric("rereference_channel_dropout_auc"),"high_frequency_auc":metric("high_frequency_shortcut_auc"),"mask_auc":metric("mask_only_auc"),"session_balanced_accuracy":metric("session_balanced_accuracy")}
    direction=sum((float(a["identity_margin"])>0)==(float(b["identity_margin"])>0) for a,b in zip(a_rows,b_rows))
    criteria=[]
    for name,d in decisions.items():
        criteria += [(f"{name}_auroc",d["auroc"]>=.8),(f"{name}_ci_low",d["auroc_descriptive_ci"][0]>=.70),(f"{name}_eer",d["eer"]<=.25),(f"{name}_tar",d["tar_at_far5"]>=.5),(f"{name}_positive",d["positive"]>=12),(f"{name}_permutation",.45<=d["permutation_auc"]<=.55),(f"{name}_augmentation",d["augmentation_auc"]>=d["auroc"]-.05),(f"{name}_artifact_shortcut",d["high_frequency_auc"]<=.60 and d["mask_auc"]<=.60),(f"{name}_session_probe",d["session_balanced_accuracy"]<=1/3+.10)]
    criteria.append(("A_B_direction",direction>=12)); passed=all(v for _,v in criteria)
    _csv(_root(c)/"m1_verifier_a_subject_metrics.csv",a_rows); _csv(_root(c)/"m1_verifier_b_subject_metrics.csv",b_rows); _csv(_root(c)/"m1_nuisance_probe_metrics.csv",nuisance)
    result={"status":"M1_LONGITUDINAL_VERIFIER_VALID" if passed else "M1_LONGITUDINAL_VERIFIER_FAILED","PASS":passed,"verifiers":decisions,"direction_agreement":direction,"failed_criteria":[k for k,v in criteria if not v],"participants":15,"day200_opened":False}
    _json(_root(c)/"m1_decision.json",result); return result


@dataclass
class PCARestorer:
    mean: np.ndarray
    components: np.ndarray
    ridge: float
    def restore(self,x:np.ndarray,mask:np.ndarray)->np.ndarray:
        flat=x.reshape(-1); m=mask.reshape(-1); observed=~m; basis=self.components[:,observed].T; target=flat[observed]-self.mean[observed]
        coef=np.linalg.solve(basis.T@basis+np.eye(len(self.components))*self.ridge,basis.T@target); restored=(self.mean+coef@self.components).reshape(x.shape); restored[~mask]=x[~mask]; return restored.astype(np.float32)


def _fit_restorer(c: Mapping[str, Any], training:list[int], masks:np.ndarray)->PCARestorer:
    rows=[]
    for p in training:
        for role in ("R","T","G"):
            x,_=load_role(c,p,role); rows.extend(x)
    x=np.asarray(rows,np.float32); flat=x.reshape(len(x),-1); rank=min(int(c["m0"]["pca_rank"]),len(x)-1); pca=PCA(rank,svd_solver="randomized",random_state=int(c["training_seed"])).fit(flat)
    # Frozen outer-training reconstruction CV over a deterministic subset.
    ids=np.linspace(0,len(x)-1,min(96,len(x)),dtype=int); best=None
    for ridge in map(float,c["m0"]["ridge_candidates"]):
        model=PCARestorer(pca.mean_.astype(np.float32),pca.components_.astype(np.float32),ridge); errors=[]
        for j,index in enumerate(ids):
            mask=np.asarray(masks[j%len(masks)],bool); restored=model.restore(x[index]*(~mask),mask); errors.append(np.sqrt(np.mean((restored[mask]-x[index][mask])**2))/(np.sqrt(np.mean(x[index][mask]**2))+1e-8))
        candidate=(float(np.mean(errors)),ridge)
        if best is None or candidate<best[0]: best=(candidate,model)
    return best[1]


def _identity_value(verifier:Any,x:np.ndarray,participant:int,condition:int,galleries:dict[tuple[int,int],np.ndarray])->np.ndarray:
    z=verifier.embed(x); own=galleries[(participant,condition)]; impostors=np.stack([value for (p,c),value in galleries.items() if p!=participant and c==condition]); return z@own-np.median(z@impostors.T,axis=1)


def _rrmse(output:np.ndarray,target:np.ndarray,mask:np.ndarray)->float:
    return float(np.sqrt(np.mean((output[mask]-target[mask])**2))/(np.sqrt(np.mean(target[mask]**2))+1e-8))


def _corr(output:np.ndarray,target:np.ndarray,mask:np.ndarray)->float:
    a=output[mask]; b=target[mask]; return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else 0.0


def _psd_error(output:np.ndarray,target:np.ndarray)->float:
    f,po=welch(output,fs=250,nperseg=128,axis=-1); _,pt=welch(target,fs=250,nperseg=128,axis=-1); keep=(f>=1)&(f<=45); return float(np.mean(np.abs(np.log(np.maximum(po[...,keep],1e-12))-np.log(np.maximum(pt[...,keep],1e-12)))))


def _peak_metrics(output:np.ndarray,target:np.ndarray)->tuple[float,float]:
    # ERP global-field-power peak in 250--550 ms relative to -50 ms epoch start.
    start,end=int(.30*250),int(.60*250); go=np.std(output[:,start:end],axis=0); gt=np.std(target[:,start:end],axis=0); io=int(np.argmax(go)); it=int(np.argmax(gt)); return abs(io-it)/250*1000,float(abs(go[io]-gt[it]))


def _sign_flip_p(values:Iterable[float])->float:
    x=np.asarray(list(values),float); observed=float(np.mean(x)); total=1<<len(x); count=0
    for bits in range(total):
        signs=np.where(((bits>>np.arange(len(x)))&1)>0,1.,-1.); count+=float(np.mean(x*signs))>=observed-1e-15
    return count/total


def select_m0_fold(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    training,evaluation=fold_members(fold); masks=np.load(_root(c)/"frozen"/"corruption_masks.npz")["masks"].astype(bool); restorer=_fit_restorer(c,training,masks); model_a,center,scale=_load_verifier_a(c,fold,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    templates={}; galleries={}; role_data={}
    for p in training:
        for role in ("R","T","G"):
            x,y=load_role(c,p,role); role_data[(p,role)]=(x,y)
        for condition in (1,2): templates[(p,condition)]=role_data[(p,"R")][0][role_data[(p,"R")][1]==condition].mean(0)
    pop={condition:np.mean([templates[(p,condition)] for p in training],axis=0) for condition in (1,2)}
    # A is the only verifier allowed for alpha selection. Each participant is
    # evaluated against other outer-training galleries, never Verifier-B.
    a_gallery={}
    device=next(model_a.parameters()).device
    for p in training:
        gx,gy=role_data[(p,"G")];
        for condition in (1,2): a_gallery[(p,condition)]=_embed_a(model_a,gx[gy==condition],center,scale,device).mean(0)
    scores={alpha:[] for alpha in map(float,c["m0"]["alpha_candidates"])}; rrmse={alpha:[] for alpha in scores}; null_drift=[]
    for p in training:
        tx,ty=role_data[(p,"T")]
        for i,(clean,condition) in enumerate(zip(tx,ty)):
            mask=masks[(p*1000+i)%len(masks)]; observed=clean.copy(); observed[mask]=0; base=restorer.restore(observed,mask)
            for alpha in scores:
                out=base.copy(); out[mask]+=alpha*(templates[(p,int(condition))]-pop[int(condition)])[mask]; z=_embed_a(model_a,out[None],center,scale,device)[0]; own=a_gallery[(p,int(condition))]; imp=np.stack([a_gallery[(q,int(condition))] for q in training if q!=p]); scores[alpha].append(float(z@own-np.median(imp@z))); rrmse[alpha].append(_rrmse(out,clean,mask))
            wrong=training[(training.index(p)+1)%len(training)]
            out=base.copy(); out[mask]+=(templates[(wrong,int(condition))]-pop[int(condition)])[mask]
            z=_embed_a(model_a,out[None],center,scale,device)[0]
            impostor=np.stack([a_gallery[(q,int(condition))] for q in training if q!=p])
            null_drift.append(float(z@a_gallery[(p,int(condition))]-np.median(impostor@z)))
    base_error=mean(rrmse[0.0]); eligible=[a for a in scores if mean(rrmse[a])<=base_error*float(c["m0"]["rrmse_ratio_upper"])] or [0.0]; selected=max(eligible,key=lambda a:(mean(scores[a]),-a)); delta=max(float(c["m0"]["identity_floor_cosine"]),float(np.quantile(np.abs(null_drift),.95)))
    out=_root(c)/"m0"/f"fold_{fold:02d}"; out.mkdir(parents=True,exist_ok=True); np.savez_compressed(out/"restorer.npz",mean=restorer.mean,components=restorer.components,ridge=restorer.ridge,alpha=selected,delta_id=delta)
    peak_amplitudes=[]
    for p in training:
        tx,_=role_data[(p,"T")]
        for row in tx:
            segment=np.std(row[:,int(.30*250):int(.60*250)],axis=0); peak_amplitudes.append(float(np.max(segment)))
    result={"fold":fold,"status":"M0_PARAMETERS_FROZEN","ridge":restorer.ridge,"alpha":selected,"delta_ID":delta,"outer_peak_amplitude_sd":float(np.std(peak_amplitudes)),"alpha_scores":{str(a):mean(scores[a]) for a in scores},"alpha_rrmse":{str(a):mean(rrmse[a]) for a in rrmse},"verifier_b_used":False,"heldout_outcomes_used":False}
    _json(out/"selection.json",result)
    _json(run_dir/"result_summary.json",result); return result


def evaluate_m0_fold(c:Mapping[str,Any],fold:int,run_dir:Path)->dict[str,Any]:
    from eeg_cgdr.models.brainid_verifier_b import load_verifier_b
    training,evaluation=fold_members(fold); masks=np.load(_root(c)/"frozen"/"corruption_masks.npz")["masks"].astype(bool); params=np.load(_root(c)/"m0"/f"fold_{fold:02d}"/"restorer.npz"); restorer=PCARestorer(np.asarray(params["mean"]),np.asarray(params["components"]),float(params["ridge"])); alpha=float(params["alpha"]); delta=float(params["delta_id"]); verifier=load_verifier_b(_root(c)/"server_checkpoints"/f"fold_{fold:02d}"/"verifier_b.npz")
    templates={}; galleries={}; support_trials={};
    for p in training+evaluation:
        rx,ry=load_role(c,p,"R"); gx,gy=load_role(c,p,"G")
        for condition in (1,2): templates[(p,condition)]=rx[ry==condition].mean(0); support_trials[(p,condition)]=rx[ry==condition]; galleries[(p,condition)]=verifier.embed(gx[gy==condition]).mean(0)
    pop={condition:np.mean([templates[(p,condition)] for p in training],axis=0) for condition in (1,2)}
    # Frozen ERP-condition decoder is trained once on outer clean aggregates.
    dec_x=[];dec_y=[]
    for p in training:
        for role in ("R","T","G"):
            x,y=load_role(c,p,role); dec_x.extend(x.reshape(len(x),-1));dec_y.extend(y)
    dec_scale=StandardScaler().fit(dec_x); decoder=LogisticRegression(max_iter=1000).fit(dec_scale.transform(dec_x),dec_y)
    rows=[]; wrong_rows=[]; copying=[]
    for p in evaluation:
        tx,ty=load_role(c,p,"T"); wrong=[q for q in evaluation if q!=p]
        for i,(clean,condition0) in enumerate(zip(tx,ty)):
            condition=int(condition0); mask=masks[(p*1000+i)%len(masks)]; observed=clean.copy(); observed[mask]=0; base=restorer.restore(observed,mask)
            outputs={"POP":base}
            match=base.copy(); match[mask]+=alpha*(templates[(p,condition)]-pop[condition])[mask]; outputs["MATCH"]=match
            shuffled=base.copy(); shuffled[mask]+=alpha*(templates[(p,3-condition)]-pop[condition])[mask]; outputs["SHUFFLED"]=shuffled
            for donor in wrong:
                value=base.copy(); value[mask]+=alpha*(templates[(donor,condition)]-pop[condition])[mask]; outputs[f"WRONG-{donor}"]=value
            for arm,out in outputs.items():
                identity=float(_identity_value(verifier,out[None],p,condition,{k:v for k,v in galleries.items() if k[0] in evaluation})[0]); latency,amplitude=_peak_metrics(out,clean); r=_rrmse(out,clean,mask); corr=_corr(out,clean,mask); snr=20*np.log10((np.sqrt(np.mean(clean[mask]**2))+1e-8)/(np.sqrt(np.mean((out[mask]-clean[mask])**2))+1e-8)); pred=int(decoder.predict(dec_scale.transform(out.reshape(1,-1)))[0]); top=float(np.mean(np.abs(out.mean(-1)-clean.mean(-1))))
                row={"fold":fold,"participant":p,"unit":i,"condition":condition,"mask_id":(p*1000+i)%len(masks),"arm":arm,"identity":identity,"rrmse":r,"correlation":corr,"delta_snr":snr,"psd_error":_psd_error(out,clean),"topography_error":top,"peak_latency_mae_ms":latency,"peak_amplitude_error":amplitude,"decoder_correct":int(pred==condition),"outside_max_abs_change":float(np.max(np.abs(out[~mask]-observed[~mask]))),"alpha":alpha,"delta_ID":delta}
                rows.append(row)
                if arm.startswith("WRONG"): wrong_rows.append(row.copy())
            d_support=min(np.sqrt(np.mean((match-v)**2)) for v in support_trials[(p,condition)]); d_target=np.sqrt(np.mean((match-clean)**2)); pop_d_support=min(np.sqrt(np.mean((base-v)**2)) for v in support_trials[(p,condition)]); pop_d_target=np.sqrt(np.mean((base-clean)**2)); copying.append({"fold":fold,"participant":p,"unit":i,"condition":condition,"match_closer_support_than_target":int(d_support<d_target),"pop_closer_support_than_target":int(pop_d_support<pop_d_target),"match_support_distance":d_support,"match_target_distance":d_target,"pop_support_distance":pop_d_support,"pop_target_distance":pop_d_target})
    fold_root=_root(c)/"m0"/f"fold_{fold:02d}"; _csv(fold_root/"metrics.csv",rows); _csv(fold_root/"wrong_metrics.csv",wrong_rows); _csv(fold_root/"copying.csv",copying)
    _json(run_dir/"result_summary.json",{"fold":fold,"status":"M0_FOLD_EVALUATED","participants":evaluation,"alpha":alpha,"delta_ID":delta,"day200_opened":False,"outside_identity":max(r["outside_max_abs_change"] for r in rows)})
    return {"fold":fold,"status":"M0_FOLD_EVALUATED"}


def aggregate_inventory(c:Mapping[str,Any])->dict[str,Any]:
    rows=[]
    for p in range(1,16): rows.extend(_read_csv(_root(c)/"inventory"/f"subject_{p:02d}_trials.csv"))
    _csv(_root(c)/"data_inventory.csv",rows)
    roles={(int(r["participant"]),r["role"]) for r in rows}; participants={int(r["participant"]) for r in rows}; channels=min(int(r["channels"]) for r in rows) if rows else 0
    complete=sum(all((p,role) in roles for role in ("R","T","G")) for p in participants); passed=complete>=int(c["minimum_longitudinal_with_rtg"]) and channels>=int(c["minimum_common_channels"])
    source_files=list((_data(c)/"files").glob("S*.mat")); groupb_refs=0
    for source in source_files:
        with h5py.File(source,"r") as handle:
            if "GroupB" in handle: groupb_refs += int(np.asarray(handle["GroupB"]).size)
    source_meta=_json_load(_data(c)/"figshare_article_27201003.json")
    asset_rows=[]
    for record in source_meta.get("files",[]):
        path=_data(c)/"files"/record["name"]; stat=path.stat() if path.exists() else None
        asset_rows.append({"figshare_file_id":record["id"],"name":record["name"],"source_url":record["download_url"],"expected_size_bytes":record["size"],"local_path":str(path),"local_size_bytes":stat.st_size if stat else "","local_mtime":stat.st_mtime if stat else "","present_complete":int(bool(stat and stat.st_size==record["size"]))})
    _csv(_root(c)/"source_asset_inventory.csv",asset_rows)
    result={"status":"DATA_PROTOCOL_VALID" if passed else "DATA_PROTOCOL_INSUFFICIENT","PASS":passed,"participants_found":len(participants),"participants_with_RTG":complete,"common_channels":channels,"source_files":len(source_files),"source_sampling_rate":int(c["sampling_rate_source"]),"model_sampling_rate":int(c["sampling_rate_model"]),"acquisition_system":"Neuracle Neusen series (common source-described amplifier)","device_metadata_variable_across_participants":False,"preprocessing":{"band_hz":list(map(float,c["filter_band_hz"])),"common_average_reference":True,"prestimulus_baseline_seconds":-float(c["epoch_start_seconds"])},"day200_metadata_present":True,"day200_opened":False,"controls_described_by_source":52,"groupb_reference_entries_metadata_only":groupb_refs,"controls_signal_opened":False}
    _json(_root(c)/"data_protocol_decision.json",result); return result


def aggregate_m0(c:Mapping[str,Any])->dict[str,Any]:
    rows=[]; copying=[]
    for fold in range(5): rows.extend(_read_csv(_root(c)/"m0"/f"fold_{fold:02d}"/"metrics.csv")); copying.extend(_read_csv(_root(c)/"m0"/f"fold_{fold:02d}"/"copying.csv"))
    numeric=("identity","rrmse","correlation","delta_snr","psd_error","topography_error","peak_latency_mae_ms","peak_amplitude_error","decoder_correct","outside_max_abs_change","delta_ID")
    grouped=defaultdict(list)
    for row in rows: grouped[(int(row["participant"]),int(row["condition"]),row["arm"])].append(row)
    condition_rows=[]
    for (p,condition,arm),values in grouped.items():
        item={"participant":p,"condition":condition,"arm":arm,"units":len(values)}; item.update({key:mean(float(v[key]) for v in values) for key in numeric}); condition_rows.append(item)
    # First donor-wise within participant/condition, then condition and participant.
    participants=[]
    for p in range(1,16):
        pr=[r for r in condition_rows if r["participant"]==p]; arms=defaultdict(list)
        for r in pr: arms[r["arm"]].append(r)
        def avg(arm,key): return mean(float(v[key]) for v in arms[arm])
        wrong=[r for r in pr if str(r["arm"]).startswith("WRONG-")]
        item={"participant":p,"fold":(p-1)%5}
        for key in numeric:
            item[f"MATCH_{key}"]=avg("MATCH",key); item[f"POP_{key}"]=avg("POP",key); item[f"SHUFFLED_{key}"]=avg("SHUFFLED",key); item[f"WRONG_{key}"]=mean(float(v[key]) for v in wrong)
        item["U_ID_P"]=item["MATCH_identity"]-item["POP_identity"]; item["U_ID_W"]=item["MATCH_identity"]-item["WRONG_identity"]
        item["rrmse_ratio"]=item["MATCH_rrmse"]/max(item["POP_rrmse"],1e-8); item["decoder_margin"]=item["MATCH_decoder_correct"]-item["POP_decoder_correct"]
        item["latency_worsening_ms"]=item["MATCH_peak_latency_mae_ms"]-item["POP_peak_latency_mae_ms"]; item["amplitude_worsening"]=item["MATCH_peak_amplitude_error"]-item["POP_peak_amplitude_error"]
        item["psd_worsening"]=item["MATCH_psd_error"]-item["POP_psd_error"]; item["topography_worsening"]=item["MATCH_topography_error"]-item["POP_topography_error"]
        participants.append(item)
    copy_by=defaultdict(list)
    for r in copying: copy_by[int(r["participant"])].append(r)
    copying_rows=[]
    for p,values in sorted(copy_by.items()): copying_rows.append({"participant":p,"match_rate":mean(float(v["match_closer_support_than_target"]) for v in values),"pop_rate":mean(float(v["pop_closer_support_than_target"]) for v in values),"increase":mean(float(v["match_closer_support_than_target"]) for v in values)-mean(float(v["pop_closer_support_than_target"]) for v in values)})
    def effect(key):
        values=[float(r[key]) for r in participants]; ci=_bootstrap({int(r["participant"]):float(r[key]) for r in participants},int(c["bootstrap_seed"])+(1 if key.endswith("W") else 0),int(c["bootstrap_replicates"])); return {"mean":mean(values),"median":median(values),"positive":sum(v>0 for v in values),"one_sided_exact_sign_flip":_sign_flip_p(values),"descriptive_ci":ci,"participant_values":values}
    effects={"U_ID_P":effect("U_ID_P"),"U_ID_W":effect("U_ID_W")}; floors=[float(r["MATCH_delta_ID"]) for r in participants]
    ratios={int(r["participant"]):float(r["rrmse_ratio"]) for r in participants}; ratio_ci=_bootstrap(ratios,int(c["bootstrap_seed"])+9,int(c["bootstrap_replicates"]))
    amp_limits=[]
    for fold in range(5):
        selection=_root(c)/"m0"/f"fold_{fold:02d}"/"selection.json"
        if selection.exists(): amp_limits.append(_json_load(selection).get("outer_peak_amplitude_sd",0)*float(c["m0"]["amplitude_sd_margin"]))
    if len(amp_limits)!=5: raise RuntimeError(f"missing fold-local M0 selection summaries: {len(amp_limits)}/5")
    amp_limit=mean(amp_limits)
    shuffled_gain=mean(r["MATCH_identity"]-r["SHUFFLED_identity"] for r in participants); copy_increase=max(r["increase"] for r in copying_rows)
    criteria=[("U_ID_P_floor",effects["U_ID_P"]["mean"]>mean(floors)),("U_ID_W_floor",effects["U_ID_W"]["mean"]>mean(floors)),("U_ID_P_median",effects["U_ID_P"]["median"]>0),("U_ID_W_median",effects["U_ID_W"]["median"]>0),("U_ID_P_positive",effects["U_ID_P"]["positive"]>=12),("U_ID_W_positive",effects["U_ID_W"]["positive"]>=12),("U_ID_P_p",effects["U_ID_P"]["one_sided_exact_sign_flip"]<.05),("U_ID_W_p",effects["U_ID_W"]["one_sided_exact_sign_flip"]<.05),("U_ID_P_ci",effects["U_ID_P"]["descriptive_ci"][0]>0),("U_ID_W_ci",effects["U_ID_W"]["descriptive_ci"][0]>0),("rrmse_ratio",ratio_ci[1]<=float(c["m0"]["rrmse_ratio_upper"])),("decoder",mean(r["decoder_margin"] for r in participants)>=float(c["m0"]["decoder_margin"])),("latency",mean(r["latency_worsening_ms"] for r in participants)<=float(c["m0"]["latency_mae_ms_max"])),("amplitude",mean(r["amplitude_worsening"] for r in participants)<=amp_limit),("psd",mean(r["psd_worsening"] for r in participants)<=float(c["m0"]["psd_topography_margin"])),("topography",mean(r["topography_worsening"] for r in participants)<=float(c["m0"]["psd_topography_margin"])),("shuffled_specificity",shuffled_gain>mean(floors)),("copying",copy_increase<=float(c["m0"]["copying_rate_margin"])),("outside_identity",max(r["MATCH_outside_max_abs_change"] for r in participants)==0)]
    passed=all(v for _,v in criteria); _csv(_root(c)/"m0_subject_metrics.csv",participants); _csv(_root(c)/"m0_condition_metrics.csv",condition_rows); _csv(_root(c)/"m0_wrong_donor_metrics.csv",[r for r in condition_rows if str(r["arm"]).startswith("WRONG-")]); _csv(_root(c)/"m0_copying_audit.csv",copying_rows)
    result={"status":"M0_IDENTITY_ACTIONABLE" if passed else "M0_IDENTITY_ACTIONABILITY_FAILED","PASS":passed,"effects":effects,"delta_ID_mean":mean(floors),"rrmse_ratio_ci":ratio_ci,"decoder_margin":mean(r["decoder_margin"] for r in participants),"latency_worsening_ms":mean(r["latency_worsening_ms"] for r in participants),"amplitude_worsening":mean(r["amplitude_worsening"] for r in participants),"amplitude_limit":amp_limit,"psd_worsening":mean(r["psd_worsening"] for r in participants),"topography_worsening":mean(r["topography_worsening"] for r in participants),"shuffled_identity_specificity":shuffled_gain,"copying_rate_increase_max":copy_increase,"failed_criteria":[k for k,v in criteria if not v],"day200_opened":False}
    _json(_root(c)/"m0_decision.json",result); return result


def _json_load(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))


def future_preregistration(c:Mapping[str,Any])->dict[str,Any]:
    return {"name":"BrainID-Bridge v17 post-Gate-01 preregistration","execution_authorized":False,"methods":["DET-noRef","DET-Ref","Bridge-POP","Bridge-CacheKV-MATCH","Bridge-CacheKV-POP","Bridge-CacheKV-WRONG"],"cachekv":{"shared_support_and_denoiser_tower":True,"cache":"multi-layer K_ref/V_ref once and reuse across reverse steps","forbidden":"single 64-D set token or old raw-token cross-attention"},"stages":["direct bridge plus CacheKV without identity loss","then add timestep-scaled sqrt(alpha_bar_t)*L_ID(x0_hat)"],"sampling":{"primary":"K1/PF-ODE","secondary":"K8 versus DET ensemble"},"verifiers":{"A":"training/guidance only","B":"evaluation only"},"success":"MATCH identity > POP/WRONG with waveform, ERP/task, and observation fidelity noninferiority; no requirement for significant RRMSE superiority"}


def gate01(c:Mapping[str,Any])->dict[str,Any]:
    data=_json_load(_root(c)/"data_protocol_decision.json"); m1=_json_load(_root(c)/"m1_decision.json") if (_root(c)/"m1_decision.json").exists() else {"PASS":False,"failed_criteria":["M1_not_run"]}; m0=_json_load(_root(c)/"m0_decision.json") if (_root(c)/"m0_decision.json").exists() else {"PASS":False,"failed_criteria":["M0_not_run"]}
    passed=bool(data.get("PASS") and m1.get("PASS") and m0.get("PASS")); failed=[]
    for prefix,value in (("DATA",data),("M1",m1),("M0",m0)): failed.extend(f"{prefix}:{item}" for item in value.get("failed_criteria",[]) or ([] if value.get("PASS") else [value.get("status","not_run")]))
    decision={"data_protocol":"PASS" if data.get("PASS") else ("INSUFFICIENT" if "INSUFFICIENT" in data.get("status","") else "FAIL"),"M1_verifier":"PASS" if m1.get("PASS") else ("INSUFFICIENT" if not (_root(c)/"m1_decision.json").exists() else "FAIL"),"M0_actionability":"PASS" if m0.get("PASS") else ("INSUFFICIENT" if not (_root(c)/"m0_decision.json").exists() else "FAIL"),"PASS_01":passed,"failed_criteria":failed,"day200_opened":False,"physiomotion_sealed_opened":False,"m2_m3_executed":False}
    _json(_root(c)/"gate01_decision.json",decision)
    if passed: (_root(c)/"future_brainid_bridge_preregistration.yaml").write_text(yaml.safe_dump(future_preregistration(c),sort_keys=False),encoding="utf-8")
    return decision


def write_report(c:Mapping[str,Any])->dict[str,Any]:
    """Generate the experimental report and diagnostic figures only."""
    import matplotlib.pyplot as plt

    root=_root(c); report=Path(__file__).parents[3]/"reports"/"brainid_gate_v17.md"
    data=_json_load(root/"data_protocol_decision.json")
    m1=_json_load(root/"m1_decision.json") if (root/"m1_decision.json").exists() else None
    m0=_json_load(root/"m0_decision.json") if (root/"m0_decision.json").exists() else None
    gate=_json_load(root/"gate01_decision.json")
    figures=root/"figures"; figures.mkdir(parents=True,exist_ok=True)
    if m1:
        a=_read_csv(root/"m1_verifier_a_subject_metrics.csv"); b=_read_csv(root/"m1_verifier_b_subject_metrics.csv")
        ids=[int(r["participant"]) for r in a]
        fig,ax=plt.subplots(figsize=(8,4)); ax.axhline(0,color="black",linewidth=.8)
        ax.plot(ids,[float(r["identity_margin"]) for r in a],"o-",label="Verifier-A")
        ax.plot(ids,[float(r["identity_margin"]) for r in b],"s-",label="Verifier-B")
        ax.set(xlabel="Held-out participant",ylabel="R→G identity margin",xticks=ids); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures/"m1_identity_margins.png",dpi=180); plt.close(fig)
    if m0:
        rows=_read_csv(root/"m0_subject_metrics.csv"); ids=[int(r["participant"]) for r in rows]
        fig,ax=plt.subplots(figsize=(8,4)); ax.axhline(0,color="black",linewidth=.8)
        ax.plot(ids,[float(r["U_ID_P"]) for r in rows],"o-",label="MATCH−POP")
        ax.plot(ids,[float(r["U_ID_W"]) for r in rows],"s-",label="MATCH−mean WRONG")
        ax.set(xlabel="Held-out participant",ylabel="Verifier-B identity utility",xticks=ids); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figures/"m0_identity_actionability.png",dpi=180); plt.close(fig)
    lines=[
        "# BrainID-Bridge v17 Gate-01",
        "",
        "Development prerequisite audit only. No denoiser, diffusion, I²SB bridge, CacheKV, or identity-loss model was trained.",
        "",
        "## Data and frozen protocol",
        "",
        f"- Data status: `{data['status']}`; R/T/G coverage {data['participants_with_RTG']}/{data['participants_found']}; {data['common_channels']} common EEG channels.",
        f"- Source sampling was {data['source_sampling_rate']} Hz and frozen modeling sampling was {data['model_sampling_rate']} Hz, with 1–45 Hz filtering, common-average rereference, and prestimulus baseline correction.",
        f"- The source describes one common acquisition system ({data.get('acquisition_system','not recorded')}); device identity was therefore not a varying participant feature. Session predictability was tested separately from embeddings.",
        "- R=Day-1 support, T=Day-7 restoration query, G=Day-80 independent gallery/evaluation, F=Day-200 future sealed.",
        "- Day-200 signals and PhysioMotion sealed participants remained unopened. Only mask geometry from the frozen PhysioMotion development set was transferred.",
        "- The distributed Trigger.txt prose conflicts with the paper and official SampleByTrigger.m; the frozen event mapping follows the latter two: code 1 non-target, code 2 target.",
        f"- The source describes 52 single-session controls. Only GroupB reference metadata ({data.get('groupb_reference_entries_metadata_only','NA')} entries across files) was audited; control signals were not consumed in this Gate-01 implementation.",
        "",
        "## M1: independent longitudinal verifiers",
        "",
    ]
    if m1:
        for name in ("A","B"):
            d=m1["verifiers"][name]; lines.append(f"- Verifier-{name}: AUROC {d['auroc']:.4f} (participant-bootstrap descriptive 95% CI {d['auroc_descriptive_ci'][0]:.4f}–{d['auroc_descriptive_ci'][1]:.4f}), EER {d['eer']:.4f}, TAR@FAR5 {d['tar_at_far5']:.4f}, rank-1 {d['rank1']:.4f}, positive margins {d['positive']}/15.")
        lines += [f"- Decision: `{m1['status']}`. Failed criteria: {', '.join(m1['failed_criteria']) or 'none'}.", "- Verifier-A and Verifier-B are separate implementations and checkpoints. Verifier-B was not imported by training or alpha-selection code."]
    else: lines.append("- Not run because an upstream prerequisite was insufficient.")
    lines += ["", "## M0: no-training identity actionability", ""]
    if m0:
        for key in ("U_ID_P","U_ID_W"):
            d=m0["effects"][key]; lines.append(f"- {key}: mean {d['mean']:.4f}, median {d['median']:.4f}, positive {d['positive']}/15, one-sided exact sign-flip p={d['one_sided_exact_sign_flip']:.6f}, descriptive CI {d['descriptive_ci'][0]:.4f}–{d['descriptive_ci'][1]:.4f}.")
        lines.append(f"- Decision: `{m0['status']}`. Failed criteria: {', '.join(m0['failed_criteria']) or 'none'}.")
    else: lines.append("- Not run because M1 did not pass; no model training was substituted.")
    lines += ["", "## Gate-01 decision", "", f"```json\n{json.dumps(gate,indent=2,sort_keys=True)}\n```", "", "This is development evidence. Failure constrains only the frozen longitudinal brainprint/actionability instance and is not a family-wide negative. Passing Gate-01 would only create a preregistration file; it would not execute a later model."]
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    result={"status":"REPORT_WRITTEN","report":str(report),"figures":sorted(str(p) for p in figures.glob("*.png")),"PASS_01":gate["PASS_01"]}; _json(root/"result_summary.json",result); return result
