"""Two-stage full-pool consolidation for V33P.

Stage A uses participant-disjoint train/validation groups to freeze epochs and
the SANDiff full-sampler checkpoint rule. Stage B refits from scratch on all six
non-test participants for exactly the frozen number of epochs.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bci2a import outer_folds
from .experiment import (
    STRENGTHS,
    _donors,
    _loader,
    encode,
    evaluate_representation,
    load_cached,
    replace,
    seed_all,
    sha256,
)
from .leace import LEACE
from .models import EEGNetRepresentation, LatentDANN, OneStepSanitizer, SubjectAdversary
from .sandiff import SANDiff


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _task_ba(model: EEGNetRepresentation, data, device: torch.device) -> float:
    _, logits = encode(model, data, device)
    return float(balanced_accuracy_score(data.task, logits.argmax(1)))


def train_eegnet_stage_a(train, validation, device: torch.device, seed: int, output: Path) -> tuple[EEGNetRepresentation, int, list[dict[str, float]]]:
    seed_all(seed); model=EEGNetRepresentation().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best_score=-1.0;best_epoch=0;best_state=None;curve=[]
    for epoch in range(80):
        model.train()
        for x,y in _loader(train.eeg,train.task,batch_size=64,shuffle=True,seed=seed+epoch):
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device));loss.backward();optimizer.step()
        score=_task_ba(model,validation,device);curve.append({"epoch":epoch+1,"validation_fixed_head_ba":score})
        if score>best_score:
            best_score=score;best_epoch=epoch+1;best_state=_cpu_state(model)
    model.load_state_dict(best_state);output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":best_state,"selected_epoch":best_epoch,"validation_fixed_head_ba":best_score,"selection_stage":"A"},output)
    return model,best_epoch,curve


def train_eegnet_exact(train, device: torch.device, seed: int, epochs: int, output: Path) -> EEGNetRepresentation:
    seed_all(seed);model=EEGNetRepresentation().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        for x,y in _loader(train.eeg,train.task,batch_size=64,shuffle=True,seed=seed+epoch):
            optimizer.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(model(x.to(device)),y.to(device));loss.backward();optimizer.step()
    state=_cpu_state(model);output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":state,"epochs":epochs,"selection_stage":"B_refit"},output);return model


def _privacy_balance(row: dict[str, object]) -> float:
    return float((row["fixed_head_balanced_accuracy"]+row["retrained_head_balanced_accuracy"])/2-0.25*row["adaptive_subject_attack_balanced_accuracy"]-0.10*abs(row["cross_session_same_different_auroc"]-0.5))


def _sanitized(model, kind: str, z: dict[str,np.ndarray], logits: dict[str,np.ndarray], leace: LEACE, device: torch.device, seed: int, alpha: float = 1.0) -> dict[str,np.ndarray]:
    keep={key:leace.transform(value) for key,value in z.items()};private={key:z[key]-keep[key] for key in z}
    replacement={key:replace(model,kind,keep[key],logits[key],device,seed+index) for index,key in enumerate(z)}
    return {key:keep[key]+(1-alpha)*private[key]+alpha*replacement[key] for key in z}


def _validation_row(model,kind,z,logits,task,subject,leace,head,device,seed,fold,epoch) -> dict[str,object]:
    sanitized=_sanitized(model,kind,z,logits,leace,device,seed,1.0)
    row,_=evaluate_representation(kind,seed,sanitized["train"],task["train"],sanitized["gallery"],task["gallery"],subject["gallery"],sanitized["query"],task["query"],subject["query"],head,device,fold,"strong")
    row.update({"epoch":epoch,"checkpoint_rule":"full_10_step" if kind=="SANDiff" else "full_output","privacy_utility_balance":_privacy_balance(row)})
    return row


def train_sanitizer_stage_a(kind: str,z,logits,task,subject,leace: LEACE,head: nn.Module,device: torch.device,seed: int,output_dir: Path,fold: int):
    train_keep=leace.transform(z["train"]);train_private=z["train"]-train_keep;owners=subject["train"]
    unique=sorted(np.unique(owners).tolist());mapping={value:index for index,value in enumerate(unique)};mapped=np.asarray([mapping[int(value)] for value in owners],dtype=np.int64)
    model=(OneStepSanitizer(z["train"].shape[1]) if kind=="one_step" else SANDiff(z["train"].shape[1])).to(device);adversary=SubjectAdversary(z["train"].shape[1],len(unique)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);adv_optimizer=torch.optim.AdamW(adversary.parameters(),lr=1e-3,weight_decay=1e-4)
    head.eval();[parameter.requires_grad_(False) for parameter in head.parameters()]
    rng=np.random.default_rng(seed);full_best=-1e9;full_epoch=10;full_state=None;single_best=1e9;single_epoch=1;single_state=None;selection=[]
    for epoch in range(80):
        donor=_donors(task["train"],owners,rng)
        dataset=TensorDataset(torch.from_numpy(train_keep).float(),torch.from_numpy(train_private[donor]).float(),torch.from_numpy(logits["train"]).float(),torch.from_numpy(task["train"]).long(),torch.from_numpy(mapped).long())
        loader=DataLoader(dataset,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch))
        model.train();adversary.train()
        for keep,target_private,task_logits,y,s in loader:
            keep,target_private,task_logits,y,s=[value.to(device) for value in (keep,target_private,task_logits,y,s)]
            if kind=="one_step":prediction=model(keep,task_logits)
            else:
                t=torch.randint(0,len(model.alpha_bar),(len(keep),),device=device);noise=torch.randn_like(target_private);prediction=model(model.q_sample(target_private,t,noise),keep,task_logits,t)
            sanitized=keep+prediction
            adv_optimizer.zero_grad(set_to_none=True);adv_loss=nn.functional.cross_entropy(adversary(sanitized.detach()),s);adv_loss.backward();adv_optimizer.step()
            for parameter in adversary.parameters():parameter.requires_grad_(False)
            loss=nn.functional.mse_loss(prediction,target_private)+0.5*nn.functional.cross_entropy(head(sanitized),y)-0.10*nn.functional.cross_entropy(adversary(sanitized),s)+0.05*((prediction.mean(0)-target_private.mean(0))**2).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
            for parameter in adversary.parameters():parameter.requires_grad_(True)
        # The historical V32P rule: a deterministic mid-timestep objective.
        model.eval()
        with torch.no_grad():
            query_keep=torch.from_numpy(leace.transform(z["query"])).float().to(device);query_private=torch.from_numpy(z["query"]-leace.transform(z["query"])).float().to(device);query_logits=torch.from_numpy(logits["query"]).float().to(device);query_task=torch.from_numpy(task["query"]).long().to(device)
            if kind=="one_step":single_prediction=model(query_keep,query_logits)
            else:
                t=torch.full((len(query_keep),),500,device=device,dtype=torch.long);noise=torch.randn(query_private.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed));single_prediction=model(model.q_sample(query_private,t,noise),query_keep,query_logits,t)
            objective=float((nn.functional.mse_loss(single_prediction,query_private)+0.5*nn.functional.cross_entropy(head(query_keep+single_prediction),query_task)).cpu())
        if objective<single_best:
            single_best=objective;single_epoch=epoch+1;single_state=_cpu_state(model)
        if (epoch+1)%10==0:
            row=_validation_row(model,kind,z,logits,task,subject,leace,head,device,seed,fold,epoch+1);selection.append(row)
            if row["privacy_utility_balance"]>full_best:
                full_best=row["privacy_utility_balance"];full_epoch=epoch+1;full_state=_cpu_state(model)
    output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"model":single_state,"selected_epoch":single_epoch,"validation_single_timestep_objective":single_best,"rule":"single_timestep"},output_dir/f"{kind}_single.pt")
    torch.save({"model":full_state,"selected_epoch":full_epoch,"validation_full_sampler_balance":full_best,"rule":"full_10_step" if kind=="SANDiff" else "full_output"},output_dir/f"{kind}_full.pt")
    return {"single_epoch":single_epoch,"single_objective":single_best,"full_epoch":full_epoch,"full_balance":full_best,"curve":selection}


def train_sanitizer_exact(kind: str,z_train,logits_train,task,subject,leace: LEACE,head: nn.Module,device: torch.device,seed: int,epochs_to_capture: set[int],output_dir: Path):
    keep=leace.transform(z_train);private=z_train-keep;unique=sorted(np.unique(subject));mapping={value:index for index,value in enumerate(unique)};mapped=np.asarray([mapping[value] for value in subject],dtype=np.int64)
    model=(OneStepSanitizer(z_train.shape[1]) if kind=="one_step" else SANDiff(z_train.shape[1])).to(device);adversary=SubjectAdversary(z_train.shape[1],len(unique)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);adv_optimizer=torch.optim.AdamW(adversary.parameters(),lr=1e-3,weight_decay=1e-4);head.eval();[p.requires_grad_(False) for p in head.parameters()]
    rng=np.random.default_rng(seed);paths={}
    for epoch in range(max(epochs_to_capture)):
        donor=_donors(task,subject,rng);dataset=TensorDataset(torch.from_numpy(keep).float(),torch.from_numpy(private[donor]).float(),torch.from_numpy(logits_train).float(),torch.from_numpy(task).long(),torch.from_numpy(mapped).long());loader=DataLoader(dataset,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch))
        model.train();adversary.train()
        for k,target,l,y,s in loader:
            k,target,l,y,s=[value.to(device) for value in (k,target,l,y,s)]
            if kind=="one_step":prediction=model(k,l)
            else:
                t=torch.randint(0,len(model.alpha_bar),(len(k),),device=device);prediction=model(model.q_sample(target,t,torch.randn_like(target)),k,l,t)
            sanitized=k+prediction;adv_optimizer.zero_grad(set_to_none=True);nn.functional.cross_entropy(adversary(sanitized.detach()),s).backward();adv_optimizer.step()
            for p in adversary.parameters():p.requires_grad_(False)
            loss=nn.functional.mse_loss(prediction,target)+0.5*nn.functional.cross_entropy(head(sanitized),y)-0.10*nn.functional.cross_entropy(adversary(sanitized),s)+0.05*((prediction.mean(0)-target.mean(0))**2).mean();optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
            for p in adversary.parameters():p.requires_grad_(True)
        if epoch+1 in epochs_to_capture:
            path=output_dir/f"{kind}_epoch_{epoch+1}.pt";path.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":_cpu_state(model),"epochs":epoch+1,"selection_stage":"B_refit"},path);paths[epoch+1]=path
    return paths


def train_dann_exact(z_train,y_train,s_train,head,device,seed,epochs,output):
    seed_all(seed);unique=sorted(np.unique(s_train));mapping={value:index for index,value in enumerate(unique)};mapped=np.asarray([mapping[value] for value in s_train],dtype=np.int64);model=LatentDANN(z_train.shape[1],len(unique)).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);head.eval();[p.requires_grad_(False) for p in head.parameters()]
    for epoch in range(epochs):
        model.train();scale=min(1.0,(epoch+1)/15)
        for z,y,s in _loader(z_train,y_train,mapped,batch_size=128,shuffle=True,seed=seed+epoch):
            transformed,subject_logits=model(z.to(device),scale);loss=nn.functional.cross_entropy(head(transformed),y.to(device))+0.25*nn.functional.cross_entropy(subject_logits,s.to(device));optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":_cpu_state(model),"epochs":epochs},output);return model


def train_dann_stage_a(z,task,subject,head,device,seed,fold,output):
    seed_all(seed);unique=sorted(np.unique(subject["train"]));mapping={value:index for index,value in enumerate(unique)};mapped=np.asarray([mapping[value] for value in subject["train"]],dtype=np.int64);model=LatentDANN(z["train"].shape[1],len(unique)).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);head.eval();[p.requires_grad_(False) for p in head.parameters()]
    best=-1e9;best_epoch=5;best_state=None;curve=[]
    for epoch in range(50):
        model.train();scale=min(1.0,(epoch+1)/15)
        for value,y,s in _loader(z["train"],task["train"],mapped,batch_size=128,shuffle=True,seed=seed+epoch):
            transformed,subject_logits=model(value.to(device),scale);loss=nn.functional.cross_entropy(head(transformed),y.to(device))+0.25*nn.functional.cross_entropy(subject_logits,s.to(device));optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (epoch+1)%5==0:
            model.eval()
            with torch.no_grad():transformed={key:model.transform(torch.from_numpy(value).float().to(device)).cpu().numpy() for key,value in z.items()}
            row,_=evaluate_representation("DANN",seed,transformed["train"],task["train"],transformed["gallery"],task["gallery"],subject["gallery"],transformed["query"],task["query"],subject["query"],head,device,fold,"medium");row.update({"epoch":epoch+1,"privacy_utility_balance":_privacy_balance(row)});curve.append(row)
            if row["privacy_utility_balance"]>best:best=row["privacy_utility_balance"];best_epoch=epoch+1;best_state=_cpu_state(model)
    output.parent.mkdir(parents=True,exist_ok=True);torch.save({"model":best_state,"selected_epoch":best_epoch,"validation_balance":best},output);return {"full_epoch":best_epoch,"full_balance":best,"curve":curve}


@torch.no_grad()
def _latency(model,kind,keep,logits,device,batch_size,seed):
    k=torch.from_numpy(keep[:batch_size]).float().to(device);l=torch.from_numpy(logits[:batch_size]).float().to(device);noise=torch.randn(k.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed))
    def call():
        if kind=="one_step":return model(k,l)
        return model.sample(k,l,reverse_steps=10,noise=noise)
    for _ in range(20):call()
    if device.type=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    values=[]
    for _ in range(100):
        start=time.perf_counter();call()
        if device.type=="cuda":torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
    memory=int(torch.cuda.max_memory_allocated()) if device.type=="cuda" else 0
    return {"batch_size":batch_size,"median_ms":float(np.median(values)),"p95_ms":float(np.percentile(values,95)),"peak_gpu_memory_bytes":memory,"parameters":sum(p.numel() for p in model.parameters())}


def run_consolidation_fold(cache_path: Path,result_root: Path,fold: int,seed: int,device: torch.device) -> dict[str,object]:
    split=outer_folds()[fold];train_a=load_cached(cache_path,split["train_subjects"],"T");val_gallery=load_cached(cache_path,split["validation_subjects"],"T");val_query=load_cached(cache_path,split["validation_subjects"],"E");full_subjects=sorted(split["train_subjects"]+split["validation_subjects"]);full=load_cached(cache_path,full_subjects,"T");test_gallery=load_cached(cache_path,split["test_subjects"],"T");test_query=load_cached(cache_path,split["test_subjects"],"E")
    run=result_root/"runtime"/f"fold_{fold}_seed_{seed}";run.mkdir(parents=True,exist_ok=True)
    stage_a_model,eegnet_epoch,eeg_curve=train_eegnet_stage_a(train_a,val_query,device,seed,run/"stage_a_eegnet.pt")
    za_train,la_train=encode(stage_a_model,train_a,device);za_gallery,la_gallery=encode(stage_a_model,val_gallery,device);za_query,la_query=encode(stage_a_model,val_query,device);leace_a=LEACE.fit(za_train,train_a.subject)
    z_a={"train":za_train,"gallery":za_gallery,"query":za_query};l_a={"train":la_train,"gallery":la_gallery,"query":la_query};task_a={"train":train_a.task,"gallery":val_gallery.task,"query":val_query.task};subject_a={"train":train_a.subject,"gallery":val_gallery.subject,"query":val_query.subject}
    dann_selection=train_dann_stage_a(z_a,task_a,subject_a,stage_a_model.task_head,device,seed+3000,fold,run/"stage_a"/"dann_full.pt")
    one_selection=train_sanitizer_stage_a("one_step",z_a,l_a,task_a,subject_a,leace_a,stage_a_model.task_head,device,seed+1000,run/"stage_a",fold)
    sand_selection=train_sanitizer_stage_a("SANDiff",z_a,l_a,task_a,subject_a,leace_a,stage_a_model.task_head,device,seed+2000,run/"stage_a",fold)
    final_model=train_eegnet_exact(full,device,seed,eegnet_epoch,run/"eegnet_full_pool.pt");z_full,l_full=encode(final_model,full,device);z_gallery,l_gallery=encode(final_model,test_gallery,device);z_query,l_query=encode(final_model,test_query,device);leace=LEACE.fit(z_full,full.subject)
    dann=train_dann_exact(z_full,full.task,full.subject,final_model.task_head,device,seed+3000,dann_selection["full_epoch"],run/"dann_full_pool.pt")
    one_paths=train_sanitizer_exact("one_step",z_full,l_full,full.task,full.subject,leace,final_model.task_head,device,seed+1000,{one_selection["full_epoch"]},run/"stage_b")
    sand_paths=train_sanitizer_exact("SANDiff",z_full,l_full,full.task,full.subject,leace,final_model.task_head,device,seed+2000,{sand_selection["single_epoch"],sand_selection["full_epoch"]},run/"stage_b")
    rows=[];participants=[];latency=[];bindings=[]
    def add(name,strength,zt,zg,zq):
        row,part=evaluate_representation(name,seed,zt,full.task,zg,test_gallery.task,test_gallery.subject,zq,test_query.task,test_query.subject,final_model.task_head,device,fold,strength);rows.append(row);participants.extend(part)
    add("RAW","na",z_full,z_gallery,z_query);add("LEACE","na",leace.transform(z_full),leace.transform(z_gallery),leace.transform(z_query))
    with torch.no_grad():add("DANN","medium",dann.transform(torch.from_numpy(z_full).float().to(device)).cpu().numpy(),dann.transform(torch.from_numpy(z_gallery).float().to(device)).cpu().numpy(),dann.transform(torch.from_numpy(z_query).float().to(device)).cpu().numpy())
    z_sets={"train":z_full,"gallery":z_gallery,"query":z_query};logit_sets={"train":l_full,"gallery":l_gallery,"query":l_query};keep_query=leace.transform(z_query)
    one=OneStepSanitizer(z_full.shape[1]).to(device);one.load_state_dict(torch.load(one_paths[one_selection["full_epoch"]],map_location=device,weights_only=True)["model"])
    sand_full=SANDiff(z_full.shape[1]).to(device);sand_full.load_state_dict(torch.load(sand_paths[sand_selection["full_epoch"]],map_location=device,weights_only=True)["model"])
    sand_single=SANDiff(z_full.shape[1]).to(device);sand_single.load_state_dict(torch.load(sand_paths[sand_selection["single_epoch"]],map_location=device,weights_only=True)["model"])
    for kind,name,model in (("one_step","one_step",one),("SANDiff","SANDiff",sand_full)):
        for strength,alpha in STRENGTHS.items():
            san=_sanitized(model,kind,z_sets,logit_sets,leace,device,seed+5000,alpha);add(name,strength,san["train"],san["gallery"],san["query"])
    san_single=_sanitized(sand_single,"SANDiff",z_sets,logit_sets,leace,device,seed+5000,1.0);add("SANDiff_single_checkpoint","strong",san_single["train"],san_single["gallery"],san_single["query"])
    for kind,name,model in (("one_step","one_step",one),("SANDiff","SANDiff",sand_full)):
        for batch in (1,64):
            item=_latency(model,kind,keep_query,l_query,device,batch,seed);item.update({"fold":fold,"seed":seed,"method":name});latency.append(item)
    path_map={"EEGNet":run/"eegnet_full_pool.pt","DANN":run/"dann_full_pool.pt","one_step":one_paths[one_selection["full_epoch"]],"SANDiff_full":sand_paths[sand_selection["full_epoch"]],"SANDiff_single":sand_paths[sand_selection["single_epoch"]]}
    for name,path in path_map.items():bindings.append({"fold":fold,"seed":seed,"model":name,"path":str(path.resolve()),"sha256":sha256(path),"training_subjects":";".join(map(str,full_subjects))})
    selection_rows=[]
    for model_name,selection in (("one_step",one_selection),("SANDiff",sand_selection)):
        selection_rows.append({"fold":fold,"seed":seed,"model":model_name,"eegnet_epoch":eegnet_epoch,"single_timestep_epoch":selection["single_epoch"],"single_timestep_objective":selection["single_objective"],"full_sampler_epoch":selection["full_epoch"],"full_sampler_balance":selection["full_balance"],"selection_train_subjects":";".join(map(str,split["train_subjects"])),"selection_validation_subjects":";".join(map(str,split["validation_subjects"])),"final_refit_subjects":";".join(map(str,full_subjects))})
    selection_rows.append({"fold":fold,"seed":seed,"model":"DANN","eegnet_epoch":eegnet_epoch,"single_timestep_epoch":"na","single_timestep_objective":"na","full_sampler_epoch":dann_selection["full_epoch"],"full_sampler_balance":dann_selection["full_balance"],"selection_train_subjects":";".join(map(str,split["train_subjects"])),"selection_validation_subjects":";".join(map(str,split["validation_subjects"])),"final_refit_subjects":";".join(map(str,full_subjects))})
    payload={"fold":fold,"seed":seed,"split":split,"full_pool_subjects":full_subjects,"selection_summary":selection_rows,"selection_curves":{"one_step":one_selection["curve"],"SANDiff":sand_selection["curve"],"DANN":dann_selection["curve"],"EEGNet":eeg_curve},"metrics":rows,"participant_effects":participants,"latency":latency,"checkpoint_binding":bindings,"leace_rank_full_pool":leace.rank,"waveform_sealed_reads":0}
    (run/"fold_result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");return payload


__all__=["run_consolidation_fold","train_eegnet_exact","train_eegnet_stage_a","train_sanitizer_stage_a"]
