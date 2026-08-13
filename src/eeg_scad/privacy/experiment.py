"""Training and evaluation harness for the V32P SANDiff pilot."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bci2a import BCI2ATrials, concatenate, load_bci2a_session, outer_folds
from .leace import LEACE
from .models import EEGNetRepresentation, LatentDANN, OneStepSanitizer, SubjectAdversary
from .sandiff import SANDiff


STRENGTHS = {"weak": 0.33, "medium": 0.66, "strong": 1.0}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cache(data_root: Path, cache_path: Path) -> dict[str, object]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with np.load(cache_path) as payload:
            return json.loads(str(payload["metadata"]))
    arrays: dict[str, np.ndarray] = {}
    inventory = []
    for subject in range(1, 10):
        for session in ("T", "E"):
            trials = load_bci2a_session(data_root, subject, session)
            prefix = f"A{subject:02d}{session}"
            for name, value in asdict(trials).items():
                arrays[f"{prefix}_{name}"] = value
            inventory.append({
                "subject": subject, "session": session, "trials": len(trials.task),
                "artifact_flagged": int(trials.artifact_flag.sum()),
                "class_counts": np.bincount(trials.task, minlength=4).tolist(),
            })
    metadata = {"sessions": inventory, "waveform_sealed_reads": 0, "trial_contract": "one 2-s 0.5--2.5-s-post-cue window per official trial"}
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(cache_path, **arrays)
    return metadata


def load_cached(cache_path: Path, subjects: list[int], session: str) -> BCI2ATrials:
    parts = []
    with np.load(cache_path) as payload:
        for subject in subjects:
            prefix = f"A{subject:02d}{session}"
            parts.append(BCI2ATrials(**{name: payload[f"{prefix}_{name}"] for name in BCI2ATrials.__dataclass_fields__}))
    return concatenate(parts)


def _loader(x: np.ndarray, *ys: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    tensors = [torch.from_numpy(np.ascontiguousarray(x)).float()]
    tensors += [torch.from_numpy(np.ascontiguousarray(y)).long() for y in ys]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def encode(model: EEGNetRepresentation, trials: BCI2ATrials, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval(); reps=[]; logits=[]
    for (x,) in _loader(trials.eeg, batch_size=256, shuffle=False, seed=0):
        out, z = model(x.to(device), return_representation=True)
        reps.append(z.cpu().numpy()); logits.append(out.cpu().numpy())
    return np.concatenate(reps), np.concatenate(logits)


def _balanced(logits: np.ndarray, target: np.ndarray) -> float:
    return float(balanced_accuracy_score(target, logits.argmax(axis=1)))


def train_eegnet(train: BCI2ATrials, val: BCI2ATrials, device: torch.device, seed: int, output: Path) -> EEGNetRepresentation:
    seed_all(seed); model=EEGNetRepresentation().to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best=-1.0; patience=0
    for epoch in range(80):
        model.train()
        for x,y in _loader(train.eeg, train.task, batch_size=64, shuffle=True, seed=seed+epoch):
            optimizer.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(x.to(device)), y.to(device)); loss.backward(); optimizer.step()
        _, val_logits=encode(model,val,device); score=_balanced(val_logits,val.task)
        if score > best + 1e-5:
            best=score; patience=0; output.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"epoch":epoch,"validation_balanced_accuracy":best,"seed":seed},output)
        else: patience+=1
        if epoch>=25 and patience>=12: break
    model.load_state_dict(torch.load(output,map_location=device,weights_only=True)["model"]); return model


def _task_logits(head: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return head(z)


def train_dann(z_train: np.ndarray, y_train: np.ndarray, s_train: np.ndarray, z_val: np.ndarray, y_val: np.ndarray, head: nn.Module, device: torch.device, seed: int, output: Path) -> LatentDANN:
    seed_all(seed); unique=sorted(np.unique(s_train).tolist()); mapping={s:i for i,s in enumerate(unique)}; mapped=np.asarray([mapping[int(s)] for s in s_train],dtype=np.int64)
    model=LatentDANN(z_train.shape[1],len(unique)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    head.eval(); [p.requires_grad_(False) for p in head.parameters()]
    best=-1.0; patience=0
    for epoch in range(50):
        model.train(); scale=min(1.0,(epoch+1)/15)
        for z,y,s in _loader(z_train,y_train,mapped,batch_size=128,shuffle=True,seed=seed+epoch):
            z,y,s=z.to(device),y.to(device),s.to(device); transformed,subject_logits=model(z,scale)
            loss=nn.functional.cross_entropy(head(transformed),y)+0.25*nn.functional.cross_entropy(subject_logits,s)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad(): pred=head(model.transform(torch.from_numpy(z_val).float().to(device))).cpu().numpy()
        score=_balanced(pred,y_val)
        if score>best+1e-5:
            best=score;patience=0;output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"subjects":unique,"epoch":epoch,"validation_balanced_accuracy":best},output)
        else: patience+=1
        if epoch>=20 and patience>=10:break
    model.load_state_dict(torch.load(output,map_location=device,weights_only=True)["model"]);return model


def _donors(task: np.ndarray, subject: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result=np.empty(len(task),dtype=np.int64)
    for i,(label,owner) in enumerate(zip(task,subject)):
        candidates=np.flatnonzero((task==label)&(subject!=owner))
        if not len(candidates): raise ValueError("no task-matched cross-subject donor")
        result[i]=rng.choice(candidates)
    return result


def train_sanitizer(kind: str, z_train: np.ndarray, logits_train: np.ndarray, task: np.ndarray, subject: np.ndarray, z_val: np.ndarray, logits_val: np.ndarray, task_val: np.ndarray, subject_val: np.ndarray, leace: LEACE, head: nn.Module, device: torch.device, seed: int, output: Path):
    seed_all(seed); z_keep=leace.transform(z_train); z_priv=z_train-z_keep
    unique=sorted(np.unique(subject).tolist()); mapping={s:i for i,s in enumerate(unique)}; mapped=np.asarray([mapping[int(s)] for s in subject],dtype=np.int64)
    model=(OneStepSanitizer(z_train.shape[1]) if kind=="one_step" else SANDiff(z_train.shape[1])).to(device)
    adversary=SubjectAdversary(z_train.shape[1],len(unique)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); adv_opt=torch.optim.AdamW(adversary.parameters(),lr=1e-3,weight_decay=1e-4)
    head.eval();[p.requires_grad_(False) for p in head.parameters()]
    val_keep=leace.transform(z_val);val_private=z_val-val_keep
    best=float("inf");patience=0
    rng=np.random.default_rng(seed)
    for epoch in range(80):
        donor=_donors(task,subject,rng)
        ds=TensorDataset(torch.from_numpy(z_keep).float(),torch.from_numpy(z_priv[donor]).float(),torch.from_numpy(logits_train).float(),torch.from_numpy(task).long(),torch.from_numpy(mapped).long())
        loader=DataLoader(ds,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch))
        model.train();adversary.train();epoch_loss=[]
        for keep,target_private,task_logits,y,s in loader:
            keep,target_private,task_logits,y,s=[v.to(device) for v in (keep,target_private,task_logits,y,s)]
            if kind=="one_step": pred=model(keep,task_logits)
            else:
                t=torch.randint(0,len(model.alpha_bar),(len(keep),),device=device);noise=torch.randn_like(target_private);state=model.q_sample(target_private,t,noise);pred=model(state,keep,task_logits,t)
            sanitized=keep+pred
            adv_opt.zero_grad(set_to_none=True);adv_loss=nn.functional.cross_entropy(adversary(sanitized.detach()),s);adv_loss.backward();adv_opt.step()
            for p in adversary.parameters():p.requires_grad_(False)
            task_loss=nn.functional.cross_entropy(head(sanitized),y)
            distribution=nn.functional.mse_loss(pred,target_private)
            privacy=-nn.functional.cross_entropy(adversary(sanitized),s)
            loss=distribution+0.5*task_loss+0.10*privacy+0.05*((pred.mean(0)-target_private.mean(0))**2).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
            for p in adversary.parameters():p.requires_grad_(True)
            epoch_loss.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_donor=_donors(task_val,subject_val,np.random.default_rng(seed+100000))
            vk=torch.from_numpy(val_keep).float().to(device);vl=torch.from_numpy(logits_val).float().to(device);target=torch.from_numpy(val_private[val_donor]).float().to(device)
            if kind=="one_step":prediction=model(vk,vl)
            else:
                vt=torch.full((len(vk),),500,device=device,dtype=torch.long);vn=torch.randn(target.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed));prediction=model(model.q_sample(target,vt,vn),vk,vl,vt)
            val_objective=float((nn.functional.mse_loss(prediction,target)+0.5*nn.functional.cross_entropy(head(vk+prediction),torch.from_numpy(task_val).long().to(device))).cpu())
        if val_objective<best-1e-5:
            best=val_objective;patience=0;output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict(),"adversary":adversary.state_dict(),"epoch":epoch,"validation_objective":best,"seed":seed,"kind":kind},output)
        else:patience+=1
        if epoch>=30 and patience>=12:break
    model.load_state_dict(torch.load(output,map_location=device,weights_only=True)["model"]);return model


@torch.no_grad()
def replace(model, kind: str, keep: np.ndarray, logits: np.ndarray, device: torch.device, seed: int) -> np.ndarray:
    model.eval();outputs=[];generator=torch.Generator(device=device).manual_seed(seed)
    for start in range(0,len(keep),256):
        k=torch.from_numpy(keep[start:start+256]).float().to(device);l=torch.from_numpy(logits[start:start+256]).float().to(device)
        if kind=="one_step": out=model(k,l)
        else: out=model.sample(k,l,reverse_steps=10,noise=torch.randn(k.shape,device=device,generator=generator))
        outputs.append(out.cpu().numpy())
    return np.concatenate(outputs)


def _ece(probabilities: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    confidence=probabilities.max(1);prediction=probabilities.argmax(1);total=0.0
    for low,high in zip(np.linspace(0,1,bins+1)[:-1],np.linspace(0,1,bins+1)[1:]):
        mask=(confidence>=low)&(confidence<(high if high<1 else high+1e-9))
        if mask.any():total+=mask.mean()*abs((prediction[mask]==target[mask]).mean()-confidence[mask].mean())
    return float(total)


def _verification(z_gallery: np.ndarray, s_gallery: np.ndarray, z_query: np.ndarray, s_query: np.ndarray, seed: int) -> tuple[float,float]:
    scaler=StandardScaler().fit(z_gallery);g=scaler.transform(z_gallery);q=scaler.transform(z_query)
    centroids=np.stack([g[s_gallery==s].mean(0) for s in sorted(np.unique(s_gallery))]);owners=np.asarray(sorted(np.unique(s_gallery)))
    gn=centroids/np.maximum(np.linalg.norm(centroids,axis=1,keepdims=True),1e-8);qn=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8)
    pred=owners[(qn@gn.T).argmax(1)];nearest=float(balanced_accuracy_score(s_query,pred))
    rng=np.random.default_rng(seed);scores=[];labels=[]
    for _ in range(min(20000,len(q)*20)):
        i=int(rng.integers(len(q))); same=bool(rng.integers(2)); candidates=np.flatnonzero(s_gallery==s_query[i] if same else s_gallery!=s_query[i]);j=int(rng.choice(candidates));
        scores.append(float(qn[i]@(g[j]/max(np.linalg.norm(g[j]),1e-8))));labels.append(int(same))
    return nearest,float(roc_auc_score(labels,scores))


def evaluate_representation(name: str, seed: int, z_train: np.ndarray, y_train: np.ndarray, z_gallery: np.ndarray, y_gallery: np.ndarray, s_gallery: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, s_test: np.ndarray, fixed_head: nn.Module, device: torch.device, fold: int, strength: str = "na") -> tuple[dict[str,object],list[dict[str,object]]]:
    head=fixed_head.to(device).eval()
    with torch.no_grad():logits=head(torch.from_numpy(z_test).float().to(device)).cpu().numpy()
    probabilities=np.exp(logits-logits.max(1,keepdims=True));probabilities/=probabilities.sum(1,keepdims=True)
    fixed=float(balanced_accuracy_score(y_test,logits.argmax(1)))
    task_probe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed)).fit(z_train,y_train)
    retrained=float(balanced_accuracy_score(y_test,task_probe.predict(z_test)))
    linear=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed)).fit(z_gallery,s_gallery)
    linear_acc=float(balanced_accuracy_score(s_test,linear.predict(z_test)))
    adaptive=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=300,random_state=seed,batch_size=64)).fit(z_gallery,s_gallery)
    adaptive_acc=float(balanced_accuracy_score(s_test,adaptive.predict(z_test)))
    verify_acc,verify_auc=_verification(z_gallery,s_gallery,z_test,s_test,seed)
    participant_rows=[];per=[]
    for subject in sorted(np.unique(s_test)):
        mask=s_test==subject;value=float(balanced_accuracy_score(y_test[mask],logits[mask].argmax(1)));per.append(value);participant_rows.append({"fold":fold,"method":name,"seed":seed,"strength":strength,"participant":int(subject+1),"fixed_head_balanced_accuracy":value})
    row={"fold":fold,"method":name,"seed":seed,"strength":strength,"fixed_head_balanced_accuracy":fixed,"retrained_head_balanced_accuracy":retrained,"calibration_error":_ece(probabilities,y_test),"worst_participant_accuracy":float(min(per)),"between_participant_variance":float(np.var(per,ddof=1)),"linear_subject_probe_balanced_accuracy":linear_acc,"adaptive_subject_attack_balanced_accuracy":adaptive_acc,"cross_session_verification_balanced_accuracy":verify_acc,"cross_session_same_different_auroc":verify_auc,"n_test_trials":len(y_test)}
    return row,participant_rows


def _transform_latency(transform: Callable[[], np.ndarray], repeats: int = 20) -> tuple[float,float]:
    samples=[]
    for _ in range(3):transform()
    for _ in range(repeats):
        start=time.perf_counter();transform();samples.append((time.perf_counter()-start)*1000)
    return float(np.median(samples)),float(np.percentile(samples,95))


def run_fold(cache_path: Path, result_root: Path, fold: int, device: torch.device) -> dict[str,object]:
    split=outer_folds()[fold];train_subjects=split["train_subjects"];validation_subjects=split["validation_subjects"];test_subjects=split["test_subjects"]
    train=load_cached(cache_path,train_subjects,"T");val=load_cached(cache_path,validation_subjects,"E");gallery=load_cached(cache_path,test_subjects,"T");test=load_cached(cache_path,test_subjects,"E")
    run=result_root/"runtime"/f"fold_{fold}";run.mkdir(parents=True,exist_ok=True)
    base_path=run/"eegnet.pt";base=train_eegnet(train,val,device,20260920+fold,base_path)
    z_train,l_train=encode(base,train,device);z_val,l_val=encode(base,val,device);z_gallery,l_gallery=encode(base,gallery,device);z_test,l_test=encode(base,test,device)
    leace=LEACE.fit(z_train,train.subject);np.savez(run/"leace.npz",mean=leace.mean,eraser=leace.eraser,rank=leace.rank)
    dann_path=run/"dann.pt";dann=train_dann(z_train,train.task,train.subject,z_val,val.task,base.task_head,device,20260920+fold,dann_path)
    rows=[];participants=[];latency=[];bindings=[]
    def add(name,seed,strength,zt,zg,ze):
        row,part=evaluate_representation(name,seed,zt,train.task,zg,gallery.task,gallery.subject,ze,test.task,test.subject,base.task_head,device,fold,strength);rows.append(row);participants.extend(part)
    add("RAW",20260920,"na",z_train,z_gallery,z_test)
    add("LEACE",20260920,"na",leace.transform(z_train),leace.transform(z_gallery),leace.transform(z_test))
    with torch.no_grad():
        dann_train=dann.transform(torch.from_numpy(z_train).float().to(device)).cpu().numpy();dann_gallery=dann.transform(torch.from_numpy(z_gallery).float().to(device)).cpu().numpy();dann_test=dann.transform(torch.from_numpy(z_test).float().to(device)).cpu().numpy()
    add("DANN",20260920,"medium",dann_train,dann_gallery,dann_test)
    bindings.extend([{"fold":fold,"model":"EEGNet","seed":20260920+fold,"path":str(base_path.resolve()),"sha256":sha256(base_path)}, {"fold":fold,"model":"DANN","seed":20260920+fold,"path":str(dann_path.resolve()),"sha256":sha256(dann_path)}])
    keeps={k:leace.transform(v) for k,v in {"train":z_train,"val":z_val,"gallery":z_gallery,"test":z_test}.items()};privates={"train":z_train-keeps["train"],"gallery":z_gallery-keeps["gallery"],"test":z_test-keeps["test"]}
    logits={"train":l_train,"val":l_val,"gallery":l_gallery,"test":l_test}
    for seed in (20260920,20260921):
        for kind,method in (("one_step","one_step"),("sandiff","SANDiff")):
            path=run/f"{kind}_{seed}.pt";model=train_sanitizer(kind,z_train,l_train,train.task,train.subject,z_val,l_val,val.task,val.subject,leace,base.task_head,device,seed+fold*100,path)
            replacements={key:replace(model,kind,keeps[key],logits[key],device,seed+fold*1000+index) for index,key in enumerate(("train","gallery","test"))}
            bindings.append({"fold":fold,"model":method,"seed":seed,"path":str(path.resolve()),"sha256":sha256(path)})
            for strength,alpha in STRENGTHS.items():
                sanitized={key:keeps[key]+(1-alpha)*privates[key]+alpha*replacements[key] for key in ("train","gallery","test")}
                add(method,seed,strength,sanitized["train"],sanitized["gallery"],sanitized["test"])
            batch_keep=torch.from_numpy(keeps["test"][:64]).float().to(device);batch_logits=torch.from_numpy(l_test[:64]).float().to(device)
            if kind=="one_step": fn=lambda:model(batch_keep,batch_logits).detach().cpu().numpy()
            else:
                fixed=torch.randn(batch_keep.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed));fn=lambda:model.sample(batch_keep,batch_logits,reverse_steps=10,noise=fixed).cpu().numpy()
            median,p95=_transform_latency(fn);latency.append({"fold":fold,"method":method,"seed":seed,"batch_size":64,"median_ms":median,"p95_ms":p95,"parameters":sum(p.numel() for p in model.parameters())})
    payload={"fold":fold,"split":split,"metrics":rows,"participant_effects":participants,"latency":latency,"checkpoint_binding":bindings,"leace_rank":leace.rank,"waveform_sealed_reads":0,"phase_c_waveform_interaction":"deferred_not_comparable"}
    (run/"fold_result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


__all__ = ["prepare_cache", "run_fold", "seed_all", "STRENGTHS"]
