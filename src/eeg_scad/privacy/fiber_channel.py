"""Frozen exact-fiber channel validation and head-aware privacy accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, pairwise_distances
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .bci2a import outer_folds
from .experiment import _verification, encode, evaluate_representation, load_cached, sha256
from .fiber import FiberOneStep, FiberSANDiff, HeadFiber
from .fiber_experiment import _load_eegnet, distribution_fidelity, exact_preservation, replace_fiber
from .leace import LEACE


@dataclass(frozen=True)
class FiberStratifiedResampler:
    """Training-only class/confidence-stratified empirical fiber channel."""

    fibers: np.ndarray
    predicted_class: np.ndarray
    confidence: np.ndarray
    tertiles: dict[int, tuple[float, float]]

    @classmethod
    def fit(cls, training_fibers: np.ndarray, training_h: np.ndarray) -> "FiberStratifiedResampler":
        fibers=np.asarray(training_fibers,dtype=np.float32).copy();h=np.asarray(training_h,dtype=np.float32)
        predicted=h.argmax(1);confidence=np.linalg.norm(h,axis=1);tertiles={}
        for task in sorted(np.unique(predicted)):
            values=confidence[predicted==task];cuts=np.quantile(values,[1/3,2/3]);tertiles[int(task)]=(float(cuts[0]),float(cuts[1]))
        return cls(fibers,predicted.astype(np.int64),confidence.astype(np.float32),tertiles)

    def sample(self, query_h: np.ndarray, *, seed: int) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Sample using only query H and training state; no source U or subject input."""
        h=np.asarray(query_h,dtype=np.float32);classes=h.argmax(1);confidence=np.linalg.norm(h,axis=1);rng=np.random.default_rng(seed);result=[];coverage=[]
        for index,(task,value) in enumerate(zip(classes,confidence)):
            cuts=self.tertiles.get(int(task));level=int(np.digitize(value,cuts)) if cuts is not None else -1
            exact=np.flatnonzero((self.predicted_class==task)&(np.digitize(self.confidence,cuts)==level)) if cuts is not None else np.empty(0,dtype=int)
            if len(exact):candidates=exact;route="exact_stratum"
            else:
                same=np.flatnonzero(self.predicted_class==task)
                if len(same):candidates=same;route="class_fallback"
                else:candidates=np.arange(len(self.fibers));route="global_fallback"
            donor=int(rng.choice(candidates));result.append(self.fibers[donor]);coverage.append({"query_index":index,"predicted_class":int(task),"confidence_tertile":level,"fallback_route":route,"donor_training_index":donor})
        return np.asarray(result,dtype=np.float32),coverage

    def sample_many(self, query_h: np.ndarray, *, releases: int, seed: int) -> np.ndarray:
        return np.stack([self.sample(query_h,seed=seed+release)[0] for release in range(releases)])


def strong_model_replacement(model, kind: str, centered_logits: np.ndarray, fiber_dim: int, device: torch.device, seed: int) -> np.ndarray:
    """Strong replacement path intentionally has no source-fiber argument."""
    return replace_fiber(model,kind,centered_logits,fiber_dim,device,seed)


def compose_strong_release(geometry: HeadFiber, z_head: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    return geometry.compose(z_head,replacement)


def _attacker(family: str, seed: int):
    if family=="linear":return make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced",random_state=seed))
    return make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=300,random_state=seed,batch_size=64))


def head_aware_attacks(method: str, fold: int, seed: int, gallery: dict[str,np.ndarray], query: dict[str,np.ndarray], gallery_subject: np.ndarray, query_subject: np.ndarray) -> tuple[list[dict[str,object]],list[dict[str,object]]]:
    rows=[];participants=[]
    features={
        "A_H":(gallery["H"],query["H"]),
        "A_Z":(gallery["Z"],query["Z"]),
        "A_HZ":(np.concatenate([gallery["H"],gallery["Z"]],axis=1),np.concatenate([query["H"],query["Z"]],axis=1)),
        "A_HU":(np.concatenate([gallery["H"],gallery["U"]],axis=1),np.concatenate([query["H"],query["U"]],axis=1)),
    }
    labels=np.unique(gallery_subject)
    for family in ("linear","adaptive_mlp"):
        for feature,(x_gallery,x_query) in features.items():
            model=_attacker(family,seed+{"A_H":0,"A_Z":1,"A_HZ":2,"A_HU":3}[feature]);model.fit(x_gallery,gallery_subject);prediction=model.predict(x_query);probability=model.predict_proba(x_query)
            ce=float(log_loss(query_subject,probability,labels=model.classes_));_,auc,_,per_auc=_verification(x_gallery,gallery_subject,x_query,query_subject,seed)
            rows.append({"fold":fold,"seed":seed,"method":method,"attacker":family,"feature":feature,"balanced_accuracy":float(balanced_accuracy_score(query_subject,prediction)),"cross_entropy":ce,"same_different_verification_auroc":auc,"gallery_session":"T","query_session":"E"})
            class_index={int(value):index for index,value in enumerate(model.classes_)}
            for subject in sorted(np.unique(query_subject)):
                mask=query_subject==subject;subject_ce=float(-np.log(np.maximum(probability[mask,class_index[int(subject)]],1e-12)).mean())
                participants.append({"fold":fold,"seed":seed,"method":method,"attacker":family,"feature":feature,"participant":int(subject+1),"subject_recall":float(np.mean(prediction[mask]==subject)),"cross_entropy":subject_ce,"same_different_verification_auroc":per_auc[int(subject)]})
    return rows,participants


def multisample_diagnostics(method: str, releases: np.ndarray, training_fibers: np.ndarray) -> dict[str,float|str]:
    """Aggregate all registered releases without target-based sample selection."""
    count,n_query,_=releases.shape
    within=float(np.var(releases,axis=0,ddof=1).sum(axis=1).mean()) if count>1 else 0.0
    if within < 1e-10: within=0.0
    means=releases.mean(axis=0);between=float(np.var(means,axis=0,ddof=1).sum())
    pair_distances=[];duplicates=[]
    for query_index in range(n_query):
        distances=pairwise_distances(releases[:,query_index,:]);upper=distances[np.triu_indices(count,1)];pair_distances.extend(upper.tolist());duplicates.extend((upper<1e-8).astype(float).tolist())
    nearest=[]
    for release in releases:
        for start in range(0,n_query,64):nearest.extend(pairwise_distances(release[start:start+64],training_fibers).min(axis=1).tolist())
    duplicate_rate=float(np.mean(duplicates));sample_diversity=float(np.mean(pair_distances))
    if within==0.0:duplicate_rate=1.0;sample_diversity=0.0
    return {"method":method,"release_count":count,"within_H_sample_variance":within,"between_H_variance":between,"conditional_covariance_trace":within,"nearest_training_fiber_distance":float(np.mean(nearest)),"duplicate_rate":duplicate_rate,"sample_diversity":sample_diversity,"sample_selection":"all_16_registered_releases"}


def run_validation_fold(v34_root: Path, v33_root: Path, result_root: Path, fold: int, seed: int, device: torch.device) -> dict[str,object]:
    split=outer_folds()[fold];run=result_root/"runtime"/f"fold_{fold}_seed_{seed}";run.mkdir(parents=True,exist_ok=True)
    source=v34_root/"results"/"fiber_sandiff_v34p"/"runtime"/f"fold_{fold}_seed_{seed}";cache=v33_root/"results"/"sandiff_v33p"/"runtime"/"bci2a_trials.npz"
    eegnet_path=v33_root/"results"/"sandiff_v33p"/"runtime"/f"fold_{fold}_seed_{seed}"/"eegnet_full_pool.pt"
    model=_load_eegnet(eegnet_path,device);geometry=HeadFiber.from_linear(model.task_head)
    full_subjects=sorted(split["train_subjects"]+split["validation_subjects"]);full=load_cached(cache,full_subjects,"T");gallery_data=load_cached(cache,split["test_subjects"],"T");query_data=load_cached(cache,split["test_subjects"],"E")
    z={};task={"train":full.task,"gallery":gallery_data.task,"query":query_data.task};subject={"train":full.subject,"gallery":gallery_data.subject,"query":query_data.subject}
    z["train"],_=encode(model,full,device);z["gallery"],_=encode(model,gallery_data,device);z["query"],_=encode(model,query_data,device)
    decomposition={key:geometry.decompose(value) for key,value in z.items()};z_head={key:value[0] for key,value in decomposition.items()};u={key:value[1] for key,value in decomposition.items()};h={key:value[2] for key,value in decomposition.items()}
    one_path=source/"stage_b"/"Fiber-OneStep.pt";sand_path=source/"stage_b"/"Fiber-SANDiff.pt"
    one=FiberOneStep(geometry.fiber_dim).to(device);one.load_state_dict(torch.load(one_path,map_location=device,weights_only=True)["model"]);one.eval()
    sand=FiberSANDiff(geometry.fiber_dim).to(device);sand.load_state_dict(torch.load(sand_path,map_location=device,weights_only=True)["model"]);sand.eval()
    resampler=FiberStratifiedResampler.fit(u["train"],h["train"])
    release={"RAW":z,"HEAD_ONLY":z_head,"LEACE":{}}
    leace=LEACE.fit(z["train"],subject["train"]);release["LEACE"]={key:leace.transform(value) for key,value in z.items()}
    coverage=[]
    for method,channel in (("Fiber-OneStep",one),("Fiber-SANDiff",sand)):
        release[method]={}
        for set_index,key in enumerate(("train","gallery","query")):
            offset=0 if method=="Fiber-OneStep" else 100
            replacement=strong_model_replacement(channel,method,h[key],geometry.fiber_dim,device,seed+5000+offset+set_index)
            release[method][key]=compose_strong_release(geometry,z_head[key],replacement)
    release["Fiber-Stratified-Resample"]={}
    for set_index,key in enumerate(("train","gallery","query")):
        replacement,details=resampler.sample(h[key],seed=seed+7000+set_index);release["Fiber-Stratified-Resample"][key]=compose_strong_release(geometry,z_head[key],replacement)
        counts={route:sum(item["fallback_route"]==route for item in details) for route in ("exact_stratum","class_fallback","global_fallback")}
        coverage.append({"fold":fold,"seed":seed,"set":key,"queries":len(details),**counts,"donor_pool":"outer_training_Session_T","strata_statistics":"outer_training_only"})
    metrics=[];participants=[];attacks=[];attack_participants=[];exact=[];fidelity=[]
    for method,sets in release.items():
        row,part=evaluate_representation(method,seed,sets["train"],task["train"],sets["gallery"],task["gallery"],subject["gallery"],sets["query"],task["query"],subject["query"],model.task_head,device,fold,"strong" if method.startswith("Fiber-") else "na");metrics.append(row);participants.extend(part)
        gallery_u=geometry.decompose(sets["gallery"])[1];query_u=geometry.decompose(sets["query"])[1]
        attack_rows,part_rows=head_aware_attacks(method,fold,seed,{"H":h["gallery"],"Z":sets["gallery"],"U":gallery_u},{"H":h["query"],"Z":sets["query"],"U":query_u},subject["gallery"],subject["query"]);attacks.extend(attack_rows);attack_participants.extend(part_rows)
        if method in ("HEAD_ONLY","Fiber-OneStep","Fiber-Stratified-Resample","Fiber-SANDiff"):
            result=exact_preservation(geometry,z["query"],sets["query"],task["query"]);result["H_recovery_max_abs_error"]=result["max_centered_logit_error"];exact.append({"fold":fold,"seed":seed,"method":method,**result})
        if method in ("Fiber-OneStep","Fiber-Stratified-Resample","Fiber-SANDiff"):
            fidelity.append({"fold":fold,"seed":seed,"method":method,**distribution_fidelity(u["query"],query_u,h["query"],seed)})
    multisample=[]
    one_replacement=strong_model_replacement(one,"Fiber-OneStep",h["query"],geometry.fiber_dim,device,seed+9000);one_many=np.repeat(one_replacement[None],16,axis=0);multisample.append({"fold":fold,"seed":seed,**multisample_diagnostics("Fiber-OneStep",one_many,u["train"])})
    resample_many=resampler.sample_many(h["query"],releases=16,seed=seed+10000);multisample.append({"fold":fold,"seed":seed,**multisample_diagnostics("Fiber-Stratified-Resample",resample_many,u["train"])})
    sand_many=np.stack([strong_model_replacement(sand,"Fiber-SANDiff",h["query"],geometry.fiber_dim,device,seed+11000+index) for index in range(16)]);multisample.append({"fold":fold,"seed":seed,**multisample_diagnostics("Fiber-SANDiff",sand_many,u["train"])})
    conditional=[]
    for method in ("Fiber-OneStep","Fiber-Stratified-Resample","Fiber-SANDiff"):
        for family in ("linear","adaptive_mlp"):
            lookup={(row["attacker"],row["feature"]):row for row in attacks if row["method"]==method}
            conditional.append({"fold":fold,"seed":seed,"method":method,"attacker":family,"CE_A_H":lookup[(family,"A_H")]["cross_entropy"],"CE_A_HU":lookup[(family,"A_HU")]["cross_entropy"],"conditional_fiber_leakage":lookup[(family,"A_H")]["cross_entropy"]-lookup[(family,"A_HU")]["cross_entropy"],"interpretation":"finite cross-session closure diagnostic, not CMI"})
    bindings=[{"fold":fold,"seed":seed,"model":"V34P_EEGNet","path":str(eegnet_path.resolve()),"sha256":sha256(eegnet_path)},{"fold":fold,"seed":seed,"model":"V34P_Fiber-OneStep","path":str(one_path.resolve()),"sha256":sha256(one_path)},{"fold":fold,"seed":seed,"model":"V34P_Fiber-SANDiff","path":str(sand_path.resolve()),"sha256":sha256(sand_path)}]
    payload={"fold":fold,"seed":seed,"split":split,"checkpoint_binding":bindings,"metrics":metrics,"participant_effects":participants,"head_aware_attacks":attacks,"attack_participant_effects":attack_participants,"conditional_fiber_leakage":conditional,"exact_preservation":exact,"distribution_fidelity":fidelity,"multisample_diversity":multisample,"resample_coverage":coverage,"code_path_audit":{"strong_replacement_receives_source_U":False,"resample_donor_pool":"outer_training_Session_T","resample_strata_statistics":"outer_training_only","randomness_uses_query_subject":False,"target_selected_multisample":False},"latency_benchmark_run":False,"waveform_sealed_reads":0}
    (run/"fold_result.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");return payload


__all__=["FiberStratifiedResampler","strong_model_replacement","head_aware_attacks","multisample_diagnostics","run_validation_fold"]
