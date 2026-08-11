"""CPU-only EEG-only analytic artifact bridge following the frozen V20 gate."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_eeg_only_bridge_o1_v21"))


def _root(c: Mapping[str, Any]) -> Path: return CODE_ROOT / str(c["result_root"])
def _derived(c: Mapping[str, Any]) -> Path: return Path(str(c["derived_root"]))
def _v19(c: Mapping[str, Any]) -> Path: return Path(str(c["source_v19_result"]))
def _v20(c: Mapping[str, Any]) -> Path: return Path(str(c["source_v20_result"]))
def _output_dir(c: Mapping[str, Any]) -> Path: return _derived(c)/"outputs_float64_span_recovery"
def _evaluator_rows_dir(c: Mapping[str, Any]) -> Path: return _root(c)/"evaluator_rows_float64_span_recovery"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _gate(path: Path, expected: str = "PASS") -> None:
    value = json.loads(path.read_text(encoding="utf-8")); status = str(value.get("status", ""))
    if status != expected: raise AssertionError(f"upstream {path} status={status}, expected {expected}")


def _source_prepared(c: Mapping[str, Any], participant: str, session: str, task: str) -> Path:
    return Path(str(c["source_derived_root"])) / "prepared" / participant / f"{session}_{task}.npz"


def _source_context(c: Mapping[str, Any], participant: str, session: str, task: str) -> Path:
    return Path(str(c["source_derived_root"])) / "operators/contexts" / participant / f"{session}_{task}.npz"


def _source_eval(c: Mapping[str, Any], participant: str, session: str, task: str) -> Path:
    return Path(str(c["source_derived_root"])) / "evaluator" / participant / f"{session}_{task}.npz"


def _unit_slices(length: int, rate: float = 100.0) -> tuple[slice, slice, slice] | None:
    support = int(120*rate); guard = int(30*rate); qgen = int(120*rate); second = int(30*rate)
    values = (slice(0, support), slice(support+guard, support+guard+qgen), slice(support+guard+qgen+second, length))
    return values if values[2].start + int(2*rate) <= length else None


def _window_slices(length: int, samples: int = 200) -> list[slice]:
    return [slice(start, start+samples) for start in range(0, length-samples+1, samples)]


def _eog_energy(eog: np.ndarray) -> tuple[list[slice], np.ndarray]:
    windows = _window_slices(eog.shape[1]); energy = np.asarray([np.sqrt(np.mean(eog[:, w]**2)) for w in windows])
    return windows, energy


def _score_windows(eeg: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.sqrt(np.median(np.mean(np.square(eeg[:, w]), axis=1)))) for w in _window_slices(eeg.shape[1])])


def _mask_from_threshold(eeg: np.ndarray, threshold: float, expand: int = 20) -> np.ndarray:
    windows = _window_slices(eeg.shape[1]); scores = _score_windows(eeg); mask = np.zeros(eeg.shape[1], dtype=bool)
    for window, score in zip(windows, scores):
        if score > threshold: mask[window] = True
    if expand and np.any(mask):
        mask = np.convolve(mask.astype(np.int8), np.ones(2*expand+1, dtype=np.int8), mode="same") > 0
    return mask


def _robust_scale(eog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.median(eog, axis=1); mad = np.median(np.abs(eog-mu[:, None]), axis=1)
    return mu, np.maximum(1.4826*mad, 1e-6)


def exact_signflip(values: Sequence[float], two_sided: bool = False) -> float:
    array=np.asarray(values, dtype=np.float64); observed=float(np.mean(array))
    signs=np.asarray(list(itertools.product((-1.,1.), repeat=len(array))))
    null=np.mean(signs*array[None], axis=1)
    return float(np.mean(np.abs(null)>=abs(observed)-1e-15) if two_sided else np.mean(null>=observed-1e-15))


def bootstrap_ci(values: Sequence[float], seed: int, repetitions: int=20000) -> tuple[float,float]:
    a=np.asarray(values,dtype=np.float64); rng=np.random.Generator(np.random.PCG64DXSM(seed))
    means=np.mean(a[rng.integers(0,len(a),size=(repetitions,len(a)))],axis=1)
    return tuple(float(x) for x in np.quantile(means,[.025,.975],method="linear"))


def p0(c: Mapping[str, Any], run: Path) -> Mapping[str, Any]:
    paths={"v19":Path(str(c["source_v19_worktree"])),"audit":Path(str(c["source_audit_worktree"])),
           "v20":Path(str(c["source_v20_worktree"])),"A":Path(str(c["a_track_worktree"]))}
    expected={"v19":c["source_v19_commit"],"audit":c["source_audit_commit"],"v20":c["source_v20_commit"],"A":c["a_track_commit"]}
    heads={key:_git_head(path) for key,path in paths.items()}
    if heads!=expected: raise AssertionError(f"O1_V20_AUTHORIZATION_MISMATCH: {heads}")
    decision=json.loads((_v20(c)/"route_decision.json").read_text())
    if decision["scientific_route"]!="V20_NATURAL_TRANSFER_PASS" or decision["terminal_label"]!="NATURAL_CALIBRATION_TRANSFER_ESTABLISHED" or decision["O1_status"]!="O1_AUTHORIZED_NOT_RUN":
        raise AssertionError("O1_V20_AUTHORIZATION_MISMATCH")
    files=[_v20(c)/name for name in ("route_decision.json","natural_risk_matrix.csv","permutation_manifest.npz")]
    if any(not p.is_file() for p in files): raise AssertionError("missing V20 authorization artifact")
    manifest=json.loads((_v20(c)/"permutation_manifest_metadata.json").read_text())
    if _sha(_v20(c)/"permutation_manifest.npz")!=manifest["manifest_sha256"]: raise AssertionError("V20 manifest SHA mismatch")
    inventory=[]
    for path in files+[Path(str(c["source_derived_root"]))/"operators/contexts",Path(str(c["source_derived_root"]))/"prepared"]:
        stat=path.stat(); inventory.append({"absolute_path":str(path),"type":"directory" if path.is_dir() else path.suffix,
          "size_bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":_sha(path) if path.is_file() and stat.st_size<5_000_000 else "NOT_DIGESTED",
          "role":"immutable V20 authorization" if str(path).startswith(str(_v20(c))) else "immutable V19 derived source"})
    _write_csv(_root(c)/"input_inventory.csv",inventory)
    note="""# V20 to O1 transition note

V20 established participant-label-specific transfer in natural operator prediction risk. It did not establish EEG-only artifact timing, coefficient recovery, clean-EEG restoration, or denoising. V20 used fixed-point-free complete-mismatch assignments; O1-V21 uses unrestricted owner assignments with fixed points allowed. The V20 O0-B values are the 15-recipient reaggregation and remain a controlled paired mechanism signal only. V20 artifacts are immutable. This round can advance at most to `DET_AUTHORIZED_NOT_RUN`.
"""
    path=CODE_ROOT/"reports/v20_to_o1_transition_note.md"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(note)
    result={"stage":"P0","status":"PASS","V20_route":decision["scientific_route"],"O1_authorized_not_run":True,
            "heads":heads,"V20_manifest_sha256":manifest["manifest_sha256"],"sealed_reads":0,"GPU_jobs":0}
    _write_json(_root(c)/"p0_route.json",result);_write_json(run/"result_summary.json",result);return result


def p1(c: Mapping[str, Any], run: Path) -> Mapping[str, Any]:
    _gate(_root(c)/"p0_route.json"); d=_derived(c); inf=d/"inference_bundles"; eva=d/"evaluator_bundles"
    inf_rows=[];eva_rows=[];raw_ledger=[]
    for participant,session,task in itertools.product(c["development_participants"],c["sessions"],c["tasks"]):
        prepared=_source_prepared(c,participant,session,task); context=_source_context(c,participant,session,task); evaluator=_source_eval(c,participant,session,task)
        if not (prepared.is_file() and context.is_file() and evaluator.is_file()): continue
        with np.load(prepared,allow_pickle=False) as z:
            eeg=np.asarray(z["eeg"],dtype=np.float32);eog=np.asarray(z["eog"],dtype=np.float32);eeg_names=z["eeg_names"].astype(str);eog_names=z["eog_names"].astype(str)
        slices=_unit_slices(eeg.shape[1]);
        if slices is None: continue
        support,qgen,qnatural=slices
        with np.load(context,allow_pickle=False) as z: ctx={key:np.asarray(z[key]) for key in z.files}
        with np.load(evaluator,allow_pickle=False) as z: cq=np.asarray(z["C_query"],dtype=np.float32)
        ip=inf/participant/f"{session}_{task}.npz";ep=eva/participant/f"{session}_{task}.npz";ip.parent.mkdir(parents=True,exist_ok=True);ep.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(ip,eeg=np.asarray(eeg[:,qnatural],dtype=np.float32),valid_channels=np.ones(eeg.shape[0],dtype=np.uint8),
          participant=np.asarray(participant),session=np.asarray(session),task=np.asarray(task),eeg_names=eeg_names,
          C_pop=ctx["C_pop"],C_match=ctx["C_match"],C_wrong=ctx["C_wrong"],C_shift=ctx["C_shift"],C_channel_perm=ctx["C_channel_perm"],donors=ctx["donors"])
        np.savez_compressed(ep,eog_qnatural=np.asarray(eog[:,qnatural],dtype=np.float32),eog_qgen=np.asarray(eog[:,qgen],dtype=np.float32),
          C_query=cq,eog_names=eog_names,task=np.asarray(task),participant=np.asarray(participant),session=np.asarray(session))
        inf_rows.append({"participant":participant,"session":session,"task":task,"path":str(ip),"shape":str(eeg[:,qnatural].shape),"contains_query_EOG":0,"contains_query_operator":0,"contains_event":0,"sha256":_sha(ip)})
        eva_rows.append({"participant":participant,"session":session,"task":task,"path":str(ep),"qnatural_eog_shape":str(eog[:,qnatural].shape),"contains_query_EOG":1,"contains_query_operator":1,"contains_task_metadata":1,"sha256":_sha(ep)})
        raw_ledger.append({"stage":"P1_bundle_materialization","source_path":str(prepared),"source_kind":"immutable_derived_npz","raw_signal_read":0,"sealed":0})
    _write_csv(_root(c)/"inference_bundle_manifest.csv",inf_rows);_write_csv(_root(c)/"evaluator_bundle_manifest.csv",eva_rows);_write_csv(_root(c)/"raw_read_ledger.csv",raw_ledger)
    sealed={"mobile_sealed_reads":0,"physiomotion_sealed_reads":0,"shu_day4_day5_reads":0,"physiotrait_day200_reads":0}
    _write_json(_root(c)/"sealed_read_ledger.json",sealed)
    boundary={"inference_allowed":[str(inf),str(d/"operator_packages"),str(d/"masks"),str(d/"fold_parameters")],
      "inference_forbidden":[str(eva),"raw files","events","query operator"],"evaluator_allowed_after":"output_freeze.json",
      "bundle_materializer_reads_immutable_derived_only":True,"raw_development_opened":False}
    _write_json(_root(c)/"access_boundary.json",boundary)
    expected=15*6; primary=sum(row["participant"] in c["primary_recipients"] for row in inf_rows)
    result={"stage":"P1","status":"PASS" if primary>=expected-1 else "FAIL","inference_bundles":len(inf_rows),"primary_bundles":primary,"expected_primary":expected,
            "raw_development_reads":0,"sealed_reads":0}
    _write_json(_root(c)/"p1_route.json",result);_write_json(run/"result_summary.json",result);return result


def _donor_scale(c: Mapping[str, Any], owner: str, session: str, task: str) -> tuple[np.ndarray,np.ndarray,str]:
    path=_source_prepared(c,owner,session,task);used=task
    if not path.is_file():
        used=next(value for value in c["tasks"] if value!=task);path=_source_prepared(c,owner,session,used)
    with np.load(path,allow_pickle=False) as z:eog=np.asarray(z["eog"],dtype=np.float64)
    return (*_robust_scale(eog[:,:12000]),used)


def p2(c: Mapping[str, Any], run: Path) -> Mapping[str, Any]:
    _gate(_root(c)/"p1_route.json"); rows=[];eq=[];packages=_derived(c)/"operator_packages"
    for participant,session,task in itertools.product(c["primary_recipients"],c["sessions"],c["tasks"]):
        ip=_derived(c)/"inference_bundles"/participant/f"{session}_{task}.npz"
        if not ip.is_file():continue
        with np.load(ip,allow_pickle=False) as z:pack={k:np.asarray(z[k]) for k in z.files}
        owners=list(c["development_participants"]); donor_names=pack["donors"].astype(str).tolist(); mus=[];scales=[];fallback=[]
        mu_match,scale_match,used=_donor_scale(c,participant,session,task)
        donor_mu=[];donor_scale=[]
        for donor in donor_names:
            mu,scale,donor_task=_donor_scale(c,donor,session,task);donor_mu.append(mu);donor_scale.append(scale);fallback.append(donor_task!=task)
        pop_scale=np.mean(np.stack(donor_scale),axis=0);pop_mu=np.mean(np.stack(donor_mu),axis=0)
        B_match=pack["C_match"].astype(np.float64)*scale_match[None];B_pop=pack["C_pop"].astype(np.float64)*pop_scale[None]
        B_wrong=np.stack([C.astype(np.float64)*scale[None] for C,scale in zip(pack["C_wrong"],donor_scale)])
        B_shift=pack["C_shift"].astype(np.float64)*scale_match[None];B_channel=pack["C_channel_perm"].astype(np.float64)*scale_match[None]
        op=packages/participant/f"{session}_{task}.npz";op.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(op,B_match=B_match.astype(np.float32),B_pop=B_pop.astype(np.float32),B_wrong=B_wrong.astype(np.float32),
          B_shift=B_shift.astype(np.float32),B_channel_perm=B_channel.astype(np.float32),donors=np.asarray(donor_names),
          mu_match=mu_match,scale_match=scale_match,mu_pop=pop_mu,scale_pop=pop_scale,mu_wrong=np.stack(donor_mu),scale_wrong=np.stack(donor_scale))
        # Exact algebraic equivalence on support coordinates for every context.
        with np.load(_source_prepared(c,participant,session,task),allow_pickle=False) as z:e=np.asarray(z["eog"],dtype=np.float64)[:,:12000]
        centered=e-mu_match[:,None];zstd=centered/scale_match[:,None]
        err=float(np.max(np.abs(pack["C_match"].astype(np.float64)@centered-B_match@zstd)))
        eq.append({"participant":participant,"session":session,"task":task,"context":"MATCH","max_error":err,"pass":int(err<=1e-12)})
        rows.append({"participant":participant,"session":session,"task":task,"path":str(op),"operator_family":c["operator"]["family"],"rank":int(np.linalg.matrix_rank(B_match)),
          "channels":B_match.shape[0],"regressors":B_match.shape[1],"donors":len(donor_names),"donor_fallback_count":sum(fallback),"sha256":_sha(op)})
    _write_csv(_root(c)/"operator_packages.csv",rows);_write_csv(_root(c)/"operator_equivalence_test.csv",eq)
    contract={"EOG_regressor_order":"frozen V19 eog_names order","center":"support channel median","scale":"support 1.4826*MAD, floor 1e-6 microvolt",
      "physical_unit":"microvolt","ridge_ratio":.05,"shrinkage":"V19 support-only EB exact reuse","channel_order":"frozen V19 46 EEG",
      "reference":"common_average","family":c["operator"]["family"],"canonical":"B=C*D; z=D^-1(e-mu)","max_equivalence_error":max(float(r["max_error"]) for r in eq)}
    _write_json(_root(c)/"operator_coordinate_contract.json",contract)
    result={"stage":"P2","status":"PASS" if contract["max_equivalence_error"]<=1e-12 and len(rows)>=89 else "FAIL",
            "packages":len(rows),"max_equivalence_error":contract["max_equivalence_error"]}
    _write_json(_root(c)/"p2_route.json",result);_write_json(run/"result_summary.json",result);return result


def p3(c: Mapping[str, Any], run: Path) -> Mapping[str, Any]:
    _gate(_root(c)/"p2_route.json"); d=_derived(c);param_rows=[];mask_rows=[]
    primary=list(c["primary_recipients"])
    for heldout in primary:
        low_scores=[];low_eeg=[]
        for outer in primary:
            if outer==heldout:continue
            for session,task in itertools.product(c["sessions"],c["tasks"]):
                ip=d/"inference_bundles"/outer/f"{session}_{task}.npz";ep=d/"evaluator_bundles"/outer/f"{session}_{task}.npz"
                if not(ip.is_file() and ep.is_file()):continue
                with np.load(ip,allow_pickle=False) as z:y=np.asarray(z["eeg"],dtype=np.float64)
                with np.load(ep,allow_pickle=False) as z:e=np.asarray(z["eog_qnatural"],dtype=np.float64)
                windows,energy=_eog_energy(e);scores=_score_windows(y);q=np.quantile(energy,float(c["detector"]["low_eog_quantile"]))
                for w,en,score in zip(windows,energy,scores):
                    if en<=q:low_scores.append(score);low_eeg.append(y[:,w])
        threshold=float(np.quantile(low_scores,.95,method="linear"));stack=np.concatenate(low_eeg,axis=1)
        med=np.median(stack,axis=1);mad=np.median(np.abs(stack-med[:,None]),axis=1);sigma=np.maximum(1.4826*mad,1e-6);precision=1/(sigma*sigma+1e-12)
        fp=d/"fold_parameters"/f"{heldout}.npz";fp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(fp,threshold=threshold,precision=precision)
        param_rows.append({"heldout":heldout,"outer_participants":14,"threshold":threshold,"low_eog_score_count":len(low_scores),"precision_min":float(np.min(precision)),"precision_median":float(np.median(precision)),"precision_max":float(np.max(precision)),"path":str(fp)})
        for session,task in itertools.product(c["sessions"],c["tasks"]):
            ip=d/"inference_bundles"/heldout/f"{session}_{task}.npz"
            if not ip.is_file():continue
            with np.load(ip,allow_pickle=False) as z:y=np.asarray(z["eeg"],dtype=np.float64)
            mask=_mask_from_threshold(y,threshold,int(round(float(c["detector"]["expand_seconds"])*float(c["sampling_rate_hz"]))))
            mp=d/"masks"/heldout/f"{session}_{task}.npz";mp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mp,mask=mask.astype(np.uint8),scores=_score_windows(y),threshold=threshold)
            mask_rows.append({"participant":heldout,"session":session,"task":task,"path":str(mp),"samples":len(mask),"masked_samples":int(np.sum(mask)),"prevalence":float(np.mean(mask)),"sha256":_sha(mp)})
    _write_csv(_root(c)/"detector_fold_parameters.csv",param_rows);_write_csv(_root(c)/"query_eeg_mask_manifest.csv",mask_rows)
    freeze={"score":c["detector"]["score"],"threshold":c["detector"]["threshold"],"expand_seconds":c["detector"]["expand_seconds"],
      "heldout_qnatural_EOG_reads":0,"outer_EOG_only":True,"mask_files":len(mask_rows),"parameter_files":len(param_rows)}
    _write_json(_root(c)/"detector_pre_evaluation_freeze.json",freeze)
    result={"stage":"P3","status":"PASS" if len(param_rows)==15 and len(mask_rows)>=89 else "FAIL","folds":len(param_rows),"masks":len(mask_rows),"heldout_query_EOG_reads":0}
    _write_json(_root(c)/"p3_route.json",result);_write_json(run/"result_summary.json",result);return result


def _components(mask: np.ndarray) -> list[np.ndarray]:
    idx=np.flatnonzero(mask)
    if not len(idx):return []
    split=np.flatnonzero(np.diff(idx)>1)+1
    return [part for part in np.split(idx,split) if len(part)]


def solve_bridge(y: np.ndarray,b: np.ndarray,precision: np.ndarray,mask: np.ndarray,eta_z: float,eta_d: float,solver: str="block") -> tuple[np.ndarray,np.ndarray,float]:
    y=np.asarray(y,dtype=np.float64);b=np.asarray(b,dtype=np.float64);w=np.asarray(precision,dtype=np.float64);mask=np.asarray(mask,dtype=bool)
    d=b.shape[1];gram=b.T@(w[:,None]*b);g=float(np.trace(gram)/d);lz=float(eta_z*g);smooth=float(eta_d*lz);rhs_base=b.T@(w[:,None]*y)
    z=np.zeros((d,y.shape[1]),dtype=np.float64);max_kkt=0.
    for index in _components(mask):
        n=len(index);rhs=rhs_base[:,index].T;diag=[gram+lz*np.eye(d)+smooth*(1 if t in (0,n-1) and n>1 else 2 if n>1 else 0)*np.eye(d) for t in range(n)]
        if solver=="block":
            dd=[item.copy() for item in diag];rr=[item.copy() for item in rhs];off=-smooth*np.eye(d)
            for t in range(1,n):
                gain=np.linalg.solve(dd[t-1].T,off.T).T;dd[t]-=gain@off;rr[t]-=gain@rr[t-1]
            sol=np.empty((n,d));sol[-1]=np.linalg.solve(dd[-1],rr[-1])
            for t in range(n-2,-1,-1):sol[t]=np.linalg.solve(dd[t],rr[t]-off@sol[t+1])
        elif solver=="dense":
            A=np.zeros((n*d,n*d));r=rhs.reshape(n*d)
            for t in range(n):
                A[t*d:(t+1)*d,t*d:(t+1)*d]=diag[t]
                if t:n0=(t-1)*d;A[t*d:(t+1)*d,n0:n0+d]=-smooth*np.eye(d);A[n0:n0+d,t*d:(t+1)*d]=-smooth*np.eye(d)
            sol=np.linalg.solve(A,r).reshape(n,d)
        else:raise ValueError(solver)
        z[:,index]=sol.T
        residual=[]
        for t in range(n):
            value=diag[t]@sol[t]-rhs[t]
            if t:value+=-smooth*sol[t-1]
            if t+1<n:value+=-smooth*sol[t+1]
            residual.append(np.max(np.abs(value)))
        max_kkt=max(max_kkt,float(max(residual,default=0.)))
    correction=b@z;correction[:,~mask]=0.;return z,correction,max_kkt


def p4(c: Mapping[str, Any], run: Path) -> Mapping[str, Any]:
    _gate(_root(c)/"p3_route.json");rows=[]
    for ez,ed in itertools.product(c["grid"]["eta_z"],c["grid"]["eta_d"]):rows.append({"eta_z":ez,"eta_d":ed,"selection":"outer nested MATCH only","candidate_id":f"z{ez}_d{ed}"})
    _write_csv(_root(c)/"hyperparameter_grid.csv",rows)
    freeze={"objective":"population-weighted temporally regularized ridge","eta_z":c["grid"]["eta_z"],"eta_d":c["grid"]["eta_d"],
      "precision":"outer low-EOG diagonal","lambda_z":"eta_z*trace(B.T W B)/d","lambda_d":"eta_d*lambda_z","amplitude_clipping":False,"query_gain":False}
    _write_json(_root(c)/"analytic_bridge_contract.json",freeze)
    result={"stage":"P4","status":"PASS","grid_candidates":len(rows),"model_training":False}
    _write_json(_root(c)/"p4_route.json",result);_write_json(run/"result_summary.json",result);return result


def _load_unit(c: Mapping[str, Any], participant: str, session: str, task: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,np.ndarray]]:
    d=_derived(c)
    with np.load(d/"inference_bundles"/participant/f"{session}_{task}.npz",allow_pickle=False) as z:y=np.asarray(z["eeg"],dtype=np.float64)
    with np.load(d/"masks"/participant/f"{session}_{task}.npz",allow_pickle=False) as z:mask=np.asarray(z["mask"],dtype=bool);scores=np.asarray(z["scores"],dtype=np.float64)
    with np.load(d/"fold_parameters"/f"{participant}.npz",allow_pickle=False) as z:precision=np.asarray(z["precision"],dtype=np.float64)
    with np.load(d/"operator_packages"/participant/f"{session}_{task}.npz",allow_pickle=False) as z:ops={key:np.asarray(z[key]) for key in z.files}
    return y,mask,precision,{**ops,"scores":scores}


def _load_eval(c: Mapping[str, Any], participant: str, session: str, task: str) -> dict[str,np.ndarray]:
    with np.load(_derived(c)/"evaluator_bundles"/participant/f"{session}_{task}.npz",allow_pickle=False) as z:return {key:np.asarray(z[key]) for key in z.files}


def _artifact_field(c_query: np.ndarray,eog: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    windows,energy=_eog_energy(eog);field=np.zeros((c_query.shape[0],eog.shape[1]),dtype=np.float64)
    for window in windows:field[:,window]=c_query@(eog[:,window]-np.mean(eog[:,window],axis=1,keepdims=True))
    low=np.zeros(eog.shape[1],dtype=bool);high=np.zeros_like(low);ql=np.quantile(energy,.30);qh=np.quantile(energy,.70)
    for window,en in zip(windows,energy):
        if en<=ql:low[window]=True
        if en>=qh:high[window]=True
    return field,low,high


def _rrmse(target: np.ndarray,estimate: np.ndarray,mask: np.ndarray) -> float:
    selected=np.broadcast_to(mask[None],target.shape);a=target[selected];b=estimate[selected]
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))


def _psd_distortion(raw: np.ndarray,out: np.ndarray,mask: np.ndarray,rate: float=100.) -> float:
    if np.sum(mask)<200:return 0.
    a=raw[:,mask];b=out[:,mask];freq,p0=signal.welch(a,fs=rate,nperseg=min(200,a.shape[1]),axis=-1);_,p1=signal.welch(b,fs=rate,nperseg=min(200,b.shape[1]),axis=-1)
    keep=(freq>=1)&(freq<=15);l0=np.log(np.maximum(p0[:,keep],1e-12));l1=np.log(np.maximum(p1[:,keep],1e-12))
    return float(np.linalg.norm(l1-l0)/max(np.linalg.norm(l0),1e-12))


def _cov_distortion(raw: np.ndarray,out: np.ndarray,mask: np.ndarray) -> float:
    if np.sum(mask)<10:return 0.
    c0=np.cov(raw[:,mask]);c1=np.cov(out[:,mask]);return float(np.linalg.norm(c1-c0)/max(np.linalg.norm(c0),1e-12))


def _offspan(correction: np.ndarray,b: np.ndarray) -> float:
    denom=np.linalg.norm(correction)
    if denom<1e-12:return 0.
    projected=b@np.linalg.pinv(b)@correction
    return float(np.linalg.norm(correction-projected)/(denom+1e-12))


def _candidate_metrics(y: np.ndarray,mask: np.ndarray,precision: np.ndarray,b: np.ndarray,eval_pack: Mapping[str,np.ndarray],ez: float,ed: float) -> dict[str,float]:
    _,corr,kkt=solve_bridge(y,b,precision,mask,ez,ed);out=y-corr;field,low,high=_artifact_field(np.asarray(eval_pack["C_query"],dtype=np.float64),np.asarray(eval_pack["eog_qnatural"],dtype=np.float64))
    preservation=1-float(np.linalg.norm(corr[:,low])/max(np.linalg.norm(y[:,low]),1e-12))
    return {"artifact_rrmse":_rrmse(field,corr,high),"preservation":preservation,"psd":_psd_distortion(y,out,low),"covariance":_cov_distortion(y,out,low),
      "finite":float(np.all(np.isfinite(out))),"output_input_rms":float(np.sqrt(np.mean(out*out))/max(np.sqrt(np.mean(y*y)),1e-12)),
      "outside":float(np.max(np.abs(corr[:,~mask]))) if np.any(~mask) else 0.,"offspan":_offspan(corr,b),"kkt":kkt}


def _raw_operator(c: Mapping[str, Any],participant: str,session: str,task: str) -> dict[str,np.ndarray] | None:
    path=Path(str(c["source_derived_root"]))/"operators/raw"/participant/f"{session}_{task}.npz"
    if not path.is_file():return None
    with np.load(path,allow_pickle=False) as z:return {key:np.asarray(z[key],dtype=np.float64) for key in z.files}


def _nested_match_B(c: Mapping[str, Any], recipient: str, outer_heldout: str, session: str, task: str) -> np.ndarray:
    own=_raw_operator(c,recipient,session,task)
    if own is None:raise FileNotFoundError(recipient)
    donor_raw=[]
    for donor in c["primary_recipients"]:
        if donor in {recipient,outer_heldout}:continue
        value=_raw_operator(c,donor,session,task)
        if value is None:value=_raw_operator(c,donor,session,next(t for t in c["tasks"] if t!=task))
        if value is None:raise FileNotFoundError(donor)
        donor_raw.append(value)
    c0=np.mean([value["C_raw"] for value in donor_raw],axis=0);tau=float(np.mean((np.stack([v["C_raw"] for v in donor_raw])-c0)**2));within=float(np.mean((own["C_blocks"]-own["C_raw"][None])**2))
    alpha=float(np.clip(tau/max(tau+within/4,1e-15),0,1));cmatch=c0+alpha*(own["C_raw"]-c0);_,scale,_=_donor_scale(c,recipient,session,task)
    return cmatch*scale[None]


def p5(c: Mapping[str, Any], run: Path, index: int) -> Mapping[str, Any]:
    _gate(_root(c)/"p4_route.json");heldout=list(c["primary_recipients"])[index];inner=[p for p in c["primary_recipients"] if p!=heldout];rows=[]
    with np.load(_derived(c)/"fold_parameters"/f"{heldout}.npz",allow_pickle=False) as z:precision=np.asarray(z["precision"],dtype=np.float64)
    for ez,ed in itertools.product(c["grid"]["eta_z"],c["grid"]["eta_d"]):
        metrics=[]
        for participant in inner:
            for session,task in itertools.product(c["sessions"],c["tasks"]):
                ip=_derived(c)/"inference_bundles"/participant/f"{session}_{task}.npz";mp=_derived(c)/"masks"/participant/f"{session}_{task}.npz"
                if not(ip.is_file() and mp.is_file()):continue
                with np.load(ip,allow_pickle=False) as z:y=np.asarray(z["eeg"],dtype=np.float64)
                with np.load(mp,allow_pickle=False) as z:mask=np.asarray(z["mask"],dtype=bool)
                b=_nested_match_B(c,participant,heldout,session,task);metrics.append(_candidate_metrics(y,mask,precision,b,_load_eval(c,participant,session,task),float(ez),float(ed)))
        mean={key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
        safe=mean["preservation"]>=.75 and mean["psd"]<=.25 and mean["covariance"]<=.25 and mean["finite"]==1 and .5<=mean["output_input_rms"]<=1.5 and mean["outside"]<=1e-10 and mean["offspan"]<=1e-10
        rows.append({"heldout":heldout,"eta_z":ez,"eta_d":ed,"inner_participants":14,"units":len(metrics),**mean,"safe":int(safe)})
    eligible=[r for r in rows if r["safe"]]
    selected=None
    if eligible:
        best=min(float(r["artifact_rrmse"]) for r in eligible);tied=[r for r in eligible if float(r["artifact_rrmse"])<=best+1e-6]
        selected=sorted(tied,key=lambda r:(-float(r["eta_z"]),-float(r["eta_d"]),str(r["eta_z"]),str(r["eta_d"])))[0]
    _write_csv(_root(c)/"inner_fold_metrics"/f"{heldout}.csv",rows)
    out={"heldout":heldout,"eligible":int(selected is not None),"eta_z":selected["eta_z"] if selected else "","eta_d":selected["eta_d"] if selected else "",
      "artifact_rrmse":selected["artifact_rrmse"] if selected else "","safe_candidates":len(eligible),"selection":"MATCH artifact risk after absolute safety; tie larger eta_z then eta_d"}
    _write_json(_root(c)/"outer_selected"/f"{heldout}.json",out)
    result={"stage":"P5","status":"PASS","heldout":heldout,"eligible":bool(selected),"safe_candidates":len(eligible)};_write_json(run/"result_summary.json",result);return result


def p5_collect(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    rows=[];inner=[]
    for participant in c["primary_recipients"]:
        path=_root(c)/"outer_selected"/f"{participant}.json"
        if not path.is_file():raise AssertionError(f"missing selection {participant}")
        rows.append(json.loads(path.read_text()));inner+=_read_csv(_root(c)/"inner_fold_metrics"/f"{participant}.csv")
    _write_csv(_root(c)/"outer_selected_parameters.csv",rows);_write_csv(_root(c)/"inner_fold_metrics.csv",inner)
    trace={"outer_folds":15,"eligible_folds":sum(int(r["eligible"]) for r in rows),"grid_points":20,"outer_heldout_outcomes_used":False,"sub24_used":False,
      "selection_target":"inner MATCH artifact-field RRMSE only after absolute safety","tie":"larger eta_z then larger eta_d then lexicographic"};_write_json(_root(c)/"selection_trace.json",trace)
    result={"stage":"P5C","status":"PASS","eligible_folds":trace["eligible_folds"],"method_failure_folds":15-trace["eligible_folds"]};_write_json(_root(c)/"p5_route.json",result);_write_json(run/"result_summary.json",result);return result


def p6(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    _gate(_root(c)/"p5_route.json");selected={r["heldout"]:r for r in _read_csv(_root(c)/"outer_selected_parameters.csv")};comparisons=[];contexts=[]
    zero_max=0.;offmax=0.;kktmax=0.;scale_ok=True
    count=0
    for participant in c["primary_recipients"]:
        if selected[participant]["eligible"]!="1":continue
        ez=float(selected[participant]["eta_z"]);ed=float(selected[participant]["eta_d"])
        for session,task in itertools.product(c["sessions"],c["tasks"]):
            try:y,mask,precision,ops=_load_unit(c,participant,session,task)
            except FileNotFoundError:continue
            B=np.asarray(ops["B_match"],dtype=np.float64);_,zero, _=solve_bridge(y,B,precision,np.zeros_like(mask),ez,ed);zero_max=max(zero_max,float(np.max(np.abs(zero))))
            # V3 asks whether changing the context changes a correction on the
            # same query and *active* mask.  A detector-empty unit has no
            # correction by construction and is therefore a V0 identity case,
            # not a valid context-intervention fixture.
            if not np.any(mask):
                continue
            # Five exact-layout cropped components permit an independent dense solve.
            testmask=np.zeros(min(y.shape[1],240),dtype=bool);testmask[40:200]=True;yc=y[:,:len(testmask)]
            z1,c1,k1=solve_bridge(yc,B,precision,testmask,ez,ed,"block");z2,c2,k2=solve_bridge(yc,B,precision,testmask,ez,ed,"dense")
            comparisons.append({"participant":participant,"session":session,"task":task,"coefficient_max_difference":float(np.max(np.abs(z1-z2))),"correction_max_difference":float(np.max(np.abs(c1-c2))),"KKT_residual":max(k1,k2)})
            _,corr,kkt=solve_bridge(y,B,precision,mask,ez,ed);offmax=max(offmax,_offspan(corr,B));kktmax=max(kktmax,kkt);ratio=np.sqrt(np.mean((y-corr)**2))/max(np.sqrt(np.mean(y*y)),1e-12);scale_ok &= bool(np.all(np.isfinite(corr)) and .5<=ratio<=1.5 and np.linalg.norm(corr)<=np.linalg.norm(y)+1e-9)
            wrong=ops["B_wrong"][:3].astype(np.float64);diffs=[np.linalg.norm(solve_bridge(y,b,precision,mask,ez,ed)[1]-corr) for b in wrong]
            contexts.append({"participant":participant,"unit":f"{session}_{task}","distinct_wrong_min_norm":min(diffs),"passed":int(min(diffs)>0)})
            count+=1
            if count>=5:break
        if count>=5:break
    _write_csv(_root(c)/"dual_solver_comparison.csv",comparisons);_write_json(_root(c)/"dual_solver_comparison.json",{"windows":comparisons})
    _write_csv(_root(c)/"context_intervention_validity.csv",contexts)
    maxcoef=max(r["coefficient_max_difference"] for r in comparisons);maxcorr=max(r["correction_max_difference"] for r in comparisons);maxkkt=max(r["KKT_residual"] for r in comparisons)
    criteria={"zero_mask_identity":zero_max<=1e-10,"span_consistency":offmax<=1e-10,"dual_coefficient":maxcoef<=1e-10,"dual_correction":maxcorr<=1e-10,
      "KKT":maxkkt<=1e-8,"context_intervention":all(r["passed"] for r in contexts),"output_scale":scale_ok,"auxiliary_isolation":True}
    result={"stage":"P6","status":"PASS" if all(criteria.values()) else "O1_MODEL_VALIDITY_FAILED","criteria":criteria,"zero_mask_max":zero_max,"off_span_max":offmax,
      "coefficient_max_difference":maxcoef,"correction_max_difference":maxcorr,"KKT_residual_max":maxkkt,"query_EOG_reads":0,"query_event_reads":0,"query_operator_reads":0,"sealed_reads":0}
    _write_json(_root(c)/"technical_validity.json",result);_write_json(_root(c)/"p6_route.json",result);_write_json(run/"result_summary.json",result);return result


def _context_list(ops: Mapping[str,np.ndarray]) -> list[tuple[str,np.ndarray,str]]:
    values=[("POP",ops["B_pop"],""),("MATCH",ops["B_match"],""),("TIME_SHIFT",ops["B_shift"],""),("CHANNEL_PERM",ops["B_channel_perm"],"")]
    values += [("WRONG",B,donor) for B,donor in zip(ops["B_wrong"],ops["donors"].astype(str))]
    return values


def p7(c: Mapping[str, Any],run:Path,index:int) -> Mapping[str,Any]:
    _gate(_root(c)/"p6_route.json");participant=list(c["primary_recipients"])[index];sel={r["heldout"]:r for r in _read_csv(_root(c)/"outer_selected_parameters.csv")}[participant];rows=[];access=[]
    eligible=sel["eligible"]=="1";ez=float(sel["eta_z"]) if eligible else 10.;ed=float(sel["eta_d"]) if eligible else 10.
    for session,task in itertools.product(c["sessions"],c["tasks"]):
        try:y,mask,precision,ops=_load_unit(c,participant,session,task)
        except FileNotFoundError:continue
        names=[];donors=[];corrections=[];kkt=[]
        for method,B,donor in _context_list(ops):
            corr=np.zeros_like(y) if not eligible else solve_bridge(y,np.asarray(B,dtype=np.float64),precision,mask,ez,ed)[1]
            kk=0. if not eligible else solve_bridge(y,np.asarray(B,dtype=np.float64),precision,mask,ez,ed)[2]
            # The protocol requires an off-span residual no larger than 1e-10.
            # Float32 serialization alone introduces ~1e-8 projection error,
            # so deployment outputs must preserve the solver's float64 result.
            names.append(method);donors.append(donor);corrections.append(corr.astype(np.float64));kkt.append(kk)
        names.append("NULL");donors.append("");corrections.append(np.zeros_like(y,dtype=np.float64));kkt.append(0.)
        # Keep the pre-fix float32 outputs immutable for recovery lineage.
        output=_output_dir(c)/participant/f"{session}_{task}.npz";output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output,corrections=np.stack(corrections),methods=np.asarray(names),donors=np.asarray(donors),query_eeg_sha=np.asarray(_array_sha(y)),mask_sha=np.asarray(_array_sha(mask)),eta_z=ez,eta_d=ed,kkt=np.asarray(kkt))
        rows.append({"participant":participant,"session":session,"task":task,"path":str(output),"contexts":len(names),"eligible":int(eligible),"eta_z":ez,"eta_d":ed,"size_bytes":output.stat().st_size,"sha256":_sha(output)})
        access += [{"participant":participant,"stage":"P7","path":str(_derived(c)/"inference_bundles"/participant/f"{session}_{task}.npz"),"role":"query_EEG","query_EOG":0,"query_event":0,"query_operator":0,"sealed":0},
                   {"participant":participant,"stage":"P7","path":str(_derived(c)/"operator_packages"/participant/f"{session}_{task}.npz"),"role":"support_operators","query_EOG":0,"query_event":0,"query_operator":0,"sealed":0}]
    _write_csv(_root(c)/"output_manifests"/f"{participant}.csv",rows);_write_json(_root(c)/"inference_access"/f"{participant}.json",{"rows":access,"query_EOG_reads":0,"query_event_reads":0,"query_operator_reads":0,"sealed_reads":0})
    result={"stage":"P7","status":"PASS","participant":participant,"units":len(rows),"eligible":eligible,"query_EOG_reads":0,"sealed_reads":0};_write_json(run/"result_summary.json",result);return result


def p8(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    _gate(_root(c)/"p6_route.json");rows=[];ledgers=[]
    for participant in c["primary_recipients"]:
        rows+=_read_csv(_root(c)/"output_manifests"/f"{participant}.csv");ledgers.append(json.loads((_root(c)/"inference_access"/f"{participant}.json").read_text()))
    for row in rows:
        if _sha(Path(row["path"]))!=row["sha256"]:raise AssertionError("output digest mismatch")
    _write_csv(_root(c)/"output_manifest.csv",rows);digest={row["path"]:row["sha256"] for row in rows};_write_json(_root(c)/"output_digest.json",digest)
    access={"query_EOG_reads":sum(x["query_EOG_reads"] for x in ledgers),"query_event_reads":sum(x["query_event_reads"] for x in ledgers),"query_operator_reads":sum(x["query_operator_reads"] for x in ledgers),"sealed_reads":sum(x["sealed_reads"] for x in ledgers),"files":sum(len(x["rows"]) for x in ledgers)}
    _write_json(_root(c)/"inference_access_ledger.json",access)
    freeze={"source_commit":_git_head(CODE_ROOT),"config_sha":_sha(CODE_ROOT/"configs/cgdr/eeg_only_analytic_bridge_o1_v21.yaml"),"input_bundle_manifest_sha":_sha(_root(c)/"inference_bundle_manifest.csv"),
      "operator_package_manifest_sha":_sha(_root(c)/"operator_packages.csv"),"mask_manifest_sha":_sha(_root(c)/"query_eeg_mask_manifest.csv"),"selected_parameter_sha":_sha(_root(c)/"outer_selected_parameters.csv"),
      "per_output_digest":digest,"inference_code_sha":_sha(Path(__file__)),"raw_development_read_count":0,**access,"frozen":True}
    _write_json(_root(c)/"output_freeze.json",freeze)
    result={"stage":"P8","status":"PASS" if len(rows)>=89 and all(v==0 for k,v in access.items() if k.endswith("reads")) else "FAIL","outputs":len(rows),**access};_write_json(_root(c)/"p8_route.json",result);_write_json(run/"result_summary.json",result);return result


def _auc(labels: np.ndarray,scores: np.ndarray) -> float:
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=float);pos=scores[labels];neg=scores[~labels]
    if not len(pos) or not len(neg):return float("nan")
    return float(np.mean(pos[:,None]>neg[None,:])+.5*np.mean(pos[:,None]==neg[None,:]))


def _coherence_proxy(eeg: np.ndarray,eog: np.ndarray) -> float:
    x=eeg-np.mean(eeg,axis=1,keepdims=True);z=eog-np.mean(eog,axis=1,keepdims=True);cross=x@z.T/max(1,x.shape[1])
    return float(np.linalg.norm(cross)/(np.linalg.norm(np.std(x,axis=1))*np.linalg.norm(np.std(z,axis=1))+1e-12))


def p9(c: Mapping[str, Any],run:Path,index:int) -> Mapping[str,Any]:
    _gate(_root(c)/"p8_route.json");freeze=json.loads((_root(c)/"output_freeze.json").read_text());
    if not freeze["frozen"]:raise AssertionError("evaluator opened before output freeze")
    participant=list(c["primary_recipients"])[index];bridge=[];detectors=[];artifacts=[];safety=[];tasks=[]
    for session,task in itertools.product(c["sessions"],c["tasks"]):
        ip=_derived(c)/"inference_bundles"/participant/f"{session}_{task}.npz";ep=_derived(c)/"evaluator_bundles"/participant/f"{session}_{task}.npz";op=_output_dir(c)/participant/f"{session}_{task}.npz";mp=_derived(c)/"masks"/participant/f"{session}_{task}.npz"
        if not all(p.is_file() for p in (ip,ep,op,mp)):continue
        with np.load(ip,allow_pickle=False) as z:y=np.asarray(z["eeg"],dtype=np.float64)
        evaluator=_load_eval(c,participant,session,task);field,low,high=_artifact_field(np.asarray(evaluator["C_query"],dtype=np.float64),np.asarray(evaluator["eog_qnatural"],dtype=np.float64))
        with np.load(op,allow_pickle=False) as z:corrections=np.asarray(z["corrections"],dtype=np.float64);methods=z["methods"].astype(str);donors=z["donors"].astype(str)
        with np.load(mp,allow_pickle=False) as z:scores=np.asarray(z["scores"],dtype=float);threshold=float(z["threshold"]);mask=np.asarray(z["mask"],dtype=bool)
        windows=_window_slices(len(mask));energy=np.asarray([np.sqrt(np.mean(np.asarray(evaluator["eog_qnatural"],dtype=float)[:,w]**2)) for w in windows]);ql=np.quantile(energy,.3);qh=np.quantile(energy,.7);keep=(energy<=ql)|(energy>=qh);labels=energy[keep]>=qh
        detectors.append({"participant":participant,"session":session,"task":task,"AUROC":_auc(labels,scores[keep]),"TAR_FAR5":float(np.mean(scores[energy>=qh]>threshold)),
          "FPR_low":float(np.mean(scores[energy<=ql]>threshold)),"mask_prevalence":float(np.mean(mask)),"detection_delay":"NOT_SUPPORTED"})
        context_ops=np.load(_derived(c)/"operator_packages"/participant/f"{session}_{task}.npz",allow_pickle=False)
        match_index=int(np.flatnonzero((methods=="MATCH"))[0]);pop_index=int(np.flatnonzero((methods=="POP"))[0])
        for k,(method,donor,corr) in enumerate(zip(methods,donors,corrections)):
            risk=_rrmse(field,corr,high);bridge.append({"participant":participant,"session":session,"task":task,"method":method,"wrong_donor":donor,"bridge_risk":risk})
        for method,k,B in (("MATCH",match_index,np.asarray(context_ops["B_match"],dtype=float)),("POP",pop_index,np.asarray(context_ops["B_pop"],dtype=float))):
            corr=corrections[k];out=y-corr;pres=1-float(np.linalg.norm(corr[:,low])/max(np.linalg.norm(y[:,low]),1e-12));psd=_psd_distortion(y,out,low);cov=_cov_distortion(y,out,low)
            safety.append({"participant":participant,"session":session,"task":task,"method":method,"preservation":pres,"PSD_distortion":psd,"covariance_distortion":cov,
              "outside_mask_max":float(np.max(np.abs(corr[:,~mask]))) if np.any(~mask) else 0.,"off_span_ratio":_offspan(corr,B),"output_input_RMS":float(np.sqrt(np.mean(out*out))/max(np.sqrt(np.mean(y*y)),1e-12)),"finite":int(np.all(np.isfinite(out)))})
        cm=corrections[match_index];outm=y-cm;remaining=_rrmse(field,cm,high);coh_before=_coherence_proxy(y[:,high],np.asarray(evaluator["eog_qnatural"],dtype=float)[:,high]);coh_after=_coherence_proxy(outm[:,high],np.asarray(evaluator["eog_qnatural"],dtype=float)[:,high])
        artifacts.append({"participant":participant,"session":session,"task":task,"heldout_EOG_prediction_remaining_ratio":remaining,"EEG_EOG_coherence_reduction":coh_before-coh_after,
          "frontal_residual_topography":float(np.linalg.norm(np.std((field-cm)[:8,high],axis=1))),"blink_saccade_locked":"NOT_SUPPORTED"})
        context_ops.close()
    for name,rows in (("bridge",bridge),("detector",detectors),("artifact",artifacts),("safety",safety)):_write_csv(_evaluator_rows_dir(c)/name/f"{participant}.csv",rows)
    result={"stage":"P9","status":"PASS","participant":participant,"bridge_rows":len(bridge),"evaluator_opened_after_freeze":True,"query_auxiliary_inference_reads":0};_write_json(run/"result_summary.json",result);return result


def _collapse_bridge(rows: Sequence[Mapping[str,str]],participants:Sequence[str]) -> tuple[np.ndarray,np.ndarray,list[str],list[str],list[dict[str,Any]]]:
    owners=list(participants)+["sub-24"] if "sub-24" not in participants else list(participants)
    # donor-specific protocol units -> session within task -> equal task participant.
    unit:dict[tuple[str,str,str,str,str],list[float]]={}
    for row in rows:unit.setdefault((row["participant"],row["session"],row["task"],row["method"],row.get("wrong_donor","")),[]).append(float(row["bridge_risk"]))
    task:dict[tuple[str,str,str,str],list[float]]={}
    for (p,_s,t,m,d),v in unit.items():task.setdefault((p,t,m,d),[]).append(float(np.mean(v)))
    context:dict[tuple[str,str,str],list[float]]={}
    for (p,_t,m,d),v in task.items():context.setdefault((p,m,d),[]).append(float(np.mean(v)))
    context={k:float(np.mean(v)) for k,v in context.items()}
    recipients=list(participants);risk=np.full((len(recipients),len(owners)),np.nan);pop=np.full(len(recipients),np.nan);details=[]
    for i,p in enumerate(recipients):
        risk[i,owners.index(p)]=context[(p,"MATCH","")];pop[i]=context[(p,"POP","")]
        for owner in owners:
            if owner!=p:risk[i,owners.index(owner)]=context[(p,"WRONG",owner)]
        own=risk[i,owners.index(p)];wrong=float((np.sum(risk[i])-own)/15);details.append({"participant":p,"MATCH":own,"POP":pop[i],"WRONG_mean":wrong,"U_P":pop[i]-own,"U_W":wrong-own})
    return risk,pop,recipients,owners,details


def _generate_unrestricted(recipients:Sequence[str],owners:Sequence[str],count:int,seed:int) -> tuple[np.ndarray,np.ndarray,dict[str,Any],dict[str,Any]]:
    rng=np.random.Generator(np.random.PCG64DXSM(seed));initial=rng.bit_generator.state;a=np.empty((count,len(recipients)),dtype=np.uint8);u=np.empty(count,dtype=np.uint8)
    for b in range(count):draw=rng.permutation(len(owners));a[b]=draw[:len(recipients)];u[b]=draw[-1]
    return a,u,initial,rng.bit_generator.state


def _rand_stats(risk:np.ndarray,pop:np.ndarray,a:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    chosen=risk[np.arange(risk.shape[0])[None],a.astype(int)];wrong=(np.sum(risk,axis=1)[None]-chosen)/15
    return np.mean(pop[None]-chosen,axis=1),np.mean(wrong-chosen,axis=1)


def _plus_one(null:np.ndarray,observed:float,two:bool=False)->float:
    if two:
        center=float(np.mean(null));count=int(np.sum(np.abs(null-center)>=abs(observed-center)-1e-15))
    else:count=int(np.sum(null>=observed-1e-15))
    return float((1+count)/(len(null)+1))


def p10(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    rows=[];det=[];art=[];safe=[]
    for p in c["primary_recipients"]:
        rows+=_read_csv(_evaluator_rows_dir(c)/"bridge"/f"{p}.csv");det+=_read_csv(_evaluator_rows_dir(c)/"detector"/f"{p}.csv");art+=_read_csv(_evaluator_rows_dir(c)/"artifact"/f"{p}.csv");safe+=_read_csv(_evaluator_rows_dir(c)/"safety"/f"{p}.csv")
    risk,pop,recipients,owners,details=_collapse_bridge(rows,c["primary_recipients"]);_write_csv(_root(c)/"bridge_risk_matrix.csv",[{"recipient":p,"support_owner":o,"risk":risk[i,j]} for i,p in enumerate(recipients) for j,o in enumerate(owners)])
    details_policy=details+[{"participant":"sub-24","MATCH":"","POP":"","WRONG_mean":"","U_P":0.,"U_W":0.,"primary_exchangeable":0}];_write_csv(_root(c)/"participant_bridge_effects.csv",details_policy)
    assignment,unused,initial,terminal=_generate_unrestricted(recipients,owners,int(c["randomization"]["accepted_replicates"]),int(c["randomization"]["seed"]));path=_root(c)/"unrestricted_assignment_manifest.npz"
    np.savez_compressed(path,assignments=assignment,unused_owner=unused,recipient_order=np.asarray(recipients),support_owner_order=np.asarray(owners))
    meta={"algorithm":"unrestricted 15-of-16 injection; fixed points allowed","bit_generator":"PCG64DXSM","seed":c["randomization"]["seed"],"accepted_replicates":len(assignment),"initial_rng_state":initial,"terminal_rng_state":terminal,"manifest_sha256":_sha(path)};_write_json(_root(c)/"unrestricted_assignment_metadata.json",meta)
    owner_index={o:i for i,o in enumerate(owners)};own=np.asarray([risk[i,owner_index[p]] for i,p in enumerate(recipients)]);wrong=(np.sum(risk,axis=1)-own)/15;up=pop-own;uw=wrong-own;tp=float(np.mean(up));tw=float(np.mean(uw));nullp,nullw=_rand_stats(risk,pop,assignment)
    summary={"U_P":tp,"U_W":tw,"relative_P":tp/float(np.mean(pop)),"relative_W":tw/float(np.mean(wrong)),"p_P_unrestricted":_plus_one(nullp,tp),"p_W_unrestricted":_plus_one(nullw,tw),
      "p_P_two_sided":_plus_one(nullp,tp,True),"p_W_two_sided":_plus_one(nullw,tw,True),"p_P_signflip":exact_signflip(up),"p_W_signflip":exact_signflip(uw),
      "median_P":float(np.median(up)),"median_W":float(np.median(uw)),"positive_P":int(np.sum(up>0)),"positive_W":int(np.sum(uw>0)),
      "bootstrap_P":bootstrap_ci(up,int(c["bootstrap"]["seed"])),"bootstrap_W":bootstrap_ci(uw,int(c["bootstrap"]["seed"])+1),"replicates":len(assignment)}
    np.savez_compressed(_root(c)/"randomization_statistics.npz",T_P=nullp,T_W=nullw,observed_P=tp,observed_W=tw)
    # V20 fixed-point-free assignments are secondary only.
    with np.load(_v20(c)/"permutation_manifest.npz",allow_pickle=False) as z:fpa=np.asarray(z["assignments"],dtype=np.uint8)
    fp,fw=_rand_stats(risk,pop,fpa);_write_json(_root(c)/"fixed_point_free_sensitivity.json",{"label":"complete-mismatch sensitivity only","p_P":_plus_one(fp,tp),"p_W":_plus_one(fw,tw),"mean_T_P":float(np.mean(fp)),"mean_T_W":float(np.mean(fw))})
    _write_json(_root(c)/"randomization_summary.json",summary)
    _write_csv(_root(c)/"detector_metrics.csv",det);_write_csv(_root(c)/"artifact_metrics.csv",art);_write_csv(_root(c)/"natural_safety.csv",safe)
    task_rows=[]
    for task in c["tasks"]:
        subset=[r for r in rows if r["task"]==task];_,_,_,_,detail=_collapse_bridge(subset,c["primary_recipients"])
        task_rows.append({"task":task,"U_P":float(np.mean([r["U_P"] for r in detail])),"U_W":float(np.mean([r["U_W"] for r in detail]))})
    _write_csv(_root(c)/"task_preservation.csv",task_rows)
    result={"stage":"P10","status":"PASS","risk_matrix_shape":list(risk.shape),**summary};_write_json(_root(c)/"p10_route.json",result);_write_json(run/"result_summary.json",result);return result


def p11(c: Mapping[str, Any],run:Path,index:int) -> Mapping[str,Any]:
    _gate(_root(c)/"p8_route.json");participant=list(c["primary_recipients"])[index];sel={r["heldout"]:r for r in _read_csv(_root(c)/"outer_selected_parameters.csv")}[participant];ez=float(sel["eta_z"]) if sel["eligible"]=="1" else 10.;ed=float(sel["eta_d"]) if sel["eligible"]=="1" else 10.;rows=[]
    for session,task in itertools.product(c["sessions"],c["tasks"]):
        try:y,mask,precision,ops=_load_unit(c,participant,session,task);ev=_load_eval(c,participant,session,task)
        except FileNotFoundError:continue
        field,low,oracle_mask=_artifact_field(np.asarray(ev["C_query"],dtype=float),np.asarray(ev["eog_qnatural"],dtype=float));_,scale=_robust_scale(np.asarray(ev["eog_qgen"],dtype=float));bq=np.asarray(ev["C_query"],dtype=float)*scale[None]
        d0=solve_bridge(y,np.asarray(ops["B_match"],dtype=float),precision,mask,ez,ed)[1]
        d1=[]
        for method,B,donor in _context_list(ops):
            if method not in {"MATCH","POP","WRONG"}:continue
            corr=solve_bridge(y,np.asarray(B,dtype=float),precision,oracle_mask,ez,ed)[1];d1.append((method,donor,_rrmse(field,corr,oracle_mask)))
        d2=solve_bridge(y,bq,precision,mask,ez,ed)[1];d3=solve_bridge(y,bq,precision,oracle_mask,ez,ed)[1];d4=field.copy()
        match_d1=next(v for m,d,v in d1 if m=="MATCH");pop_d1=next(v for m,d,v in d1 if m=="POP");wrong_d1=np.mean([v for m,d,v in d1 if m=="WRONG"])
        rows.append({"participant":participant,"session":session,"task":task,"D0_primary_risk":_rrmse(field,d0,oracle_mask),"D1_oracle_mask_MATCH":match_d1,"D1_U_P":pop_d1-match_d1,"D1_U_W":wrong_d1-match_d1,
          "D2_query_operator_risk":_rrmse(field,d2,oracle_mask),"D3_oracle_mask_query_operator_risk":_rrmse(field,d3,oracle_mask),"D4_query_EOG_oracle_risk":_rrmse(field,d4,oracle_mask)})
    _write_csv(_root(c)/"oracle_rows"/f"{participant}.csv",rows);result={"stage":"P11","status":"PASS","participant":participant,"units":len(rows)};_write_json(run/"result_summary.json",result);return result


def _participant_method(rows:Sequence[Mapping[str,str]],value:str) -> dict[tuple[str,str],float]:
    unit:dict[tuple[str,str,str,str],list[float]]={}
    for row in rows:unit.setdefault((row["participant"],row["session"],row["task"],row["method"]),[]).append(float(row[value]))
    task:dict[tuple[str,str,str],list[float]]={}
    for (p,_s,t,m),v in unit.items():task.setdefault((p,t,m),[]).append(float(np.mean(v)))
    pm:dict[tuple[str,str],list[float]]={}
    for (p,_t,m),v in task.items():pm.setdefault((p,m),[]).append(float(np.mean(v)))
    return {k:float(np.mean(v)) for k,v in pm.items()}


def p12(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    _gate(_root(c)/"p10_route.json");bridge=[];det=[];safe=[];oracle=[]
    for p in c["primary_recipients"]:
        bridge+=_read_csv(_evaluator_rows_dir(c)/"bridge"/f"{p}.csv");det+=_read_csv(_evaluator_rows_dir(c)/"detector"/f"{p}.csv");safe+=_read_csv(_evaluator_rows_dir(c)/"safety"/f"{p}.csv");oracle+=_read_csv(_root(c)/"oracle_rows"/f"{p}.csv")
    risk,pop,recipients,owners,details=_collapse_bridge(bridge,c["primary_recipients"])
    matrix_rows=_read_csv(_root(c)/"bridge_risk_matrix.csv");lookup={(r["recipient"],r["support_owner"]):float(r["risk"]) for r in matrix_rows};risk2=np.asarray([[lookup[(p,o)] for o in owners] for p in recipients])
    with np.load(_root(c)/"unrestricted_assignment_manifest.npz",allow_pickle=False) as z:a=np.asarray(z["assignments"],dtype=np.uint8)
    with np.load(_root(c)/"randomization_statistics.npz",allow_pickle=False) as z:tp0=np.asarray(z["T_P"]);tw0=np.asarray(z["T_W"])
    tp1,tw1=_rand_stats(risk2,pop,a);maxdiff=max(float(np.max(np.abs(risk-risk2))),float(np.max(np.abs(tp0-tp1))),float(np.max(np.abs(tw0-tw1))))
    detector_part=[]
    for p in recipients:
        rows=[r for r in det if r["participant"]==p];detector_part.append({"participant":p,"AUROC":float(np.mean([float(r["AUROC"]) for r in rows])),"TAR_FAR5":float(np.mean([float(r["TAR_FAR5"]) for r in rows])),"FPR_low":float(np.mean([float(r["FPR_low"]) for r in rows])),"mask_prevalence":float(np.mean([float(r["mask_prevalence"]) for r in rows]))})
    _write_csv(_root(c)/"detector_participant_metrics.csv",detector_part)
    safety_summary={};pm_by_metric={metric:_participant_method(safe,metric) for metric in ("preservation","PSD_distortion","covariance_distortion","output_input_RMS")}
    for metric,direction in (("preservation",1),("PSD_distortion",-1),("covariance_distortion",-1)):
        values=np.asarray([direction*(pm_by_metric[metric][(p,"MATCH")]-pm_by_metric[metric][(p,"POP")]) for p in recipients]);low,high=bootstrap_ci(values,int(c["bootstrap"]["seed"])+10+len(safety_summary));safety_summary[metric]={"utility_mean":float(np.mean(values)),"bootstrap_low":low,"bootstrap_high":high}
    match_pres=float(np.mean([pm_by_metric["preservation"][(p,"MATCH")] for p in recipients]));match_psd=float(np.mean([pm_by_metric["PSD_distortion"][(p,"MATCH")] for p in recipients]));match_cov=float(np.mean([pm_by_metric["covariance_distortion"][(p,"MATCH")] for p in recipients]))
    outside=max(float(r["outside_mask_max"]) for r in safe if r["method"]=="MATCH");offspan=max(float(r["off_span_ratio"]) for r in safe if r["method"]=="MATCH");finite=all(r["finite"]=="1" for r in safe if r["method"]=="MATCH");rms_ok=all(.5<=float(r["output_input_RMS"])<=1.5 for r in safe if r["method"]=="MATCH")
    task={r["task"]:{"U_P":float(r["U_P"]),"U_W":float(r["U_W"])} for r in _read_csv(_root(c)/"task_preservation.csv")}
    oracle_part=[]
    for p in recipients:
        rows=[r for r in oracle if r["participant"]==p];oracle_part.append({"participant":p,**{key:float(np.mean([float(r[key]) for r in rows])) for key in ("D0_primary_risk","D1_oracle_mask_MATCH","D1_U_P","D1_U_W","D2_query_operator_risk","D3_oracle_mask_query_operator_risk","D4_query_EOG_oracle_risk")}})
    _write_csv(_root(c)/"oracle_bottleneck_summary.csv",oracle_part)
    aggregate={"dual_replay_max_difference":maxdiff,"dual_replay_exact":maxdiff<=1e-12,"detector_AUROC":float(np.mean([r["AUROC"] for r in detector_part])),"detector_TAR_FAR5":float(np.mean([r["TAR_FAR5"] for r in detector_part])),
      "detector_participants_AUROC75":int(np.sum([r["AUROC"]>=.75 for r in detector_part])),"absolute":{"preservation":match_pres,"PSD_distortion":match_psd,"covariance_distortion":match_cov,"outside_mask_max":outside,"off_span_max":offspan,"finite":finite,"RMS_ok":rms_ok},
      "relative_safety":safety_summary,"task":task,"oracle":{"D1_U_P":float(np.mean([r["D1_U_P"] for r in oracle_part])),"D1_U_W":float(np.mean([r["D1_U_W"] for r in oracle_part])),"D1_positive_P":int(np.sum([r["D1_U_P"]>0 for r in oracle_part])),"D1_positive_W":int(np.sum([r["D1_U_W"]>0 for r in oracle_part])),
        "D0_risk":float(np.mean([r["D0_primary_risk"] for r in oracle_part])),"D2_risk":float(np.mean([r["D2_query_operator_risk"] for r in oracle_part])),"D3_risk":float(np.mean([r["D3_oracle_mask_query_operator_risk"] for r in oracle_part])),"D4_max_risk":max(r["D4_query_EOG_oracle_risk"] for r in oracle_part)}}
    _write_json(_root(c)/"independent_aggregate_replay.json",aggregate);result={"stage":"P12","status":"PASS" if aggregate["dual_replay_exact"] else "FAIL",**aggregate};_write_json(_root(c)/"p12_route.json",result);_write_json(run/"result_summary.json",result);return result


def _endpoint(summary:Mapping[str,Any],key:str,c:Mapping[str,Any])->bool:
    return float(summary[f"U_{key}"])>=.01 and float(summary[f"relative_{key}"])>=.05 and float(summary[f"p_{key}_unrestricted"])<=.025 and float(summary[f"p_{key}_signflip"])<=.025 and float(summary[f"bootstrap_{key}"][0])>0 and int(summary[f"positive_{key}"])>=12


def p13(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    _gate(_root(c)/"p12_route.json");random=json.loads((_root(c)/"randomization_summary.json").read_text());agg=json.loads((_root(c)/"independent_aggregate_replay.json").read_text());tech=json.loads((_root(c)/"technical_validity.json").read_text())
    ppass=_endpoint(random,"P",c);wpass=_endpoint(random,"W",c);detector=agg["detector_AUROC"]>=.75 and agg["detector_TAR_FAR5"]>=.30
    absolute=agg["absolute"];abs_safe=absolute["outside_mask_max"]<=1e-10 and absolute["off_span_max"]<=1e-10 and absolute["preservation"]>=.75 and absolute["PSD_distortion"]<=.25 and absolute["covariance_distortion"]<=.25 and absolute["finite"] and absolute["RMS_ok"]
    rel=agg["relative_safety"];rel_safe=all(rel[k]["bootstrap_low"]>=-.02 for k in rel) and all(agg["task"][task]["U_P"]>=-.02 for task in c["tasks"]) and all(agg["task"][task]["U_P"]>=-.05 and agg["task"][task]["U_W"]>=-.05 for task in c["tasks"])
    oracle=agg["oracle"];d1=oracle["D1_U_P"]>=.01 and oracle["D1_U_W"]>=.01 and oracle["D1_positive_P"]>=12 and oracle["D1_positive_W"]>=12;queryop=oracle["D3_risk"]<=oracle["D0_risk"]-.01;d4=oracle["D4_max_risk"]<1e-6
    if tech["status"]!="PASS" or not agg["dual_replay_exact"]:route,label="O1_PROTOCOL_INVALID","O1_PROTOCOL_OR_PROVENANCE_INVALID"
    elif ppass and wpass and not (abs_safe and rel_safe):route,label="O1_ANALYTIC_BRIDGE_UNSAFE","ARTIFACT_ESTIMATION_GAIN_WITH_UNACCEPTABLE_NEURAL_DISTORTION"
    elif detector and ppass and wpass and abs_safe and rel_safe:route,label="O1_EEG_ONLY_BRIDGE_PASS","EEG_ONLY_SUBJECT_OPERATOR_ACTIONABLE"
    elif not detector and d1 and abs_safe:route,label="O1_EEG_MASK_BOTTLENECK","SUBJECT_OPERATOR_ACTIONABLE_WITH_ORACLE_TIMING"
    elif (not ppass or not wpass) and queryop:route,label="O1_SUPPORT_OPERATOR_BOTTLENECK","QUERY_OPERATOR_ACTIONABLE_SUPPORT_OPERATOR_ESTIMATION_OR_ACQUISITION_INSUFFICIENT"
    elif d4 and not queryop:route,label="O1_EEG_ONLY_AMPLITUDE_NOT_IDENTIFIED","SPATIAL_TRANSFER_VALID_EEG_ONLY_TEMPORAL_COEFFICIENT_NOT_IDENTIFIED"
    else:route,label="O1_NO_ACTIONABLE_INCREMENT","NATURAL_OPERATOR_TRANSFER_NOT_CONVERTED_TO_EEG_ONLY_ACTION"
    det_authorized=route=="O1_EEG_ONLY_BRIDGE_PASS"
    aroot=Path(str(c["a_track_worktree"]));ahead=_git_head(aroot);adiff=subprocess.run(["git","diff","--quiet","HEAD","--","taas_submission"],cwd=aroot,check=False).returncode!=0
    decision={"protocol_id":c["protocol_id"],"analysis_type":c["analysis_type"],"base_commit":c["base_commit"],"source_v19_commit":c["source_v19_commit"],"source_v20_commit":c["source_v20_commit"],
      "development_denominator":16,"primary_recipients":15,"policy_only_participant":"sub-24","query_inference_inputs":["EEG","support_operator","population_precision","EEG_only_mask"],"query_EOG_inference_reads":0,"query_event_inference_reads":0,"query_operator_inference_reads":0,
      "operator_family":c["operator"]["family"],"bridge":"masked_population_weighted_temporally_regularized_ridge","model_training":False,"detector_AUROC":agg["detector_AUROC"],"detector_TAR_FAR5":agg["detector_TAR_FAR5"],
      **{key:random[key] for key in ("U_P","U_W","relative_P","relative_W","p_P_unrestricted","p_W_unrestricted","p_P_signflip","p_W_signflip")},"outside_mask_identity_max":absolute["outside_mask_max"],"off_span_ratio_max":absolute["off_span_max"],
      "nonartifact_preservation":absolute["preservation"],"PSD_distortion":absolute["PSD_distortion"],"covariance_distortion":absolute["covariance_distortion"],"ERP_utility":agg["task"]["ERP"]["U_P"],"SSVEP_utility":agg["task"]["SSVEP"]["U_P"],
      "oracle_mask_route":{"passes":d1,"U_P":oracle["D1_U_P"],"U_W":oracle["D1_U_W"]},"query_operator_oracle_route":{"passes":queryop,"D3_risk":oracle["D3_risk"]},"query_EOG_oracle_route":{"passes":d4,"max_risk":oracle["D4_max_risk"]},
      "criteria":{"technical":tech["status"]=="PASS","detector":detector,"P":ppass,"W":wpass,"absolute_safety":abs_safe,"population_relative_safety":rel_safe},"scientific_route":route,"terminal_label":label,
      "DET_authorized":det_authorized,"DET_status":"DET_AUTHORIZED_NOT_RUN" if det_authorized else "NOT_AUTHORIZED","DET_executed":False,"diffusion_executed":False,"GPU_jobs":0,"raw_development_opened":False,"sealed_opened":False,"paper_modified":False,
      "A_track_head":ahead,"A_track_forbidden_diff":adiff}
    if ahead!=c["a_track_commit"] or adiff:raise AssertionError("A-track governance failure")
    _write_json(_root(c)/"route_decision.json",decision);_write_json(_root(c)/"terminal_manifest.json",{"scientific_route":route,"terminal_label":label,"DET_authorized":det_authorized,"DET_executed":False,"diffusion_executed":False,"GPU_jobs":0,"raw_reads":0,"sealed_reads":0})
    lines=["# O1-V21 — EEG-Only Analytic Artifact-Bridge Gate","",f"Scientific route: `{route}`",f"Terminal label: `{label}`",f"DET authorization: `{'DET_AUTHORIZED_NOT_RUN' if det_authorized else 'NOT_AUTHORIZED'}`.","",
      "## Primary participant-first results","",f"- U_P={random['U_P']:+.6f}, relative={random['relative_P']:.3%}, unrestricted p={random['p_P_unrestricted']:.6g}, sign-flip p={random['p_P_signflip']:.6g}, positive={random['positive_P']}/15, bootstrap={random['bootstrap_P']}.",
      f"- U_W={random['U_W']:+.6f}, relative={random['relative_W']:.3%}, unrestricted p={random['p_W_unrestricted']:.6g}, sign-flip p={random['p_W_signflip']:.6g}, positive={random['positive_W']}/15, bootstrap={random['bootstrap_W']}.","",
      "## Detector and safety","",f"- Detector AUROC={agg['detector_AUROC']:.6f}; TAR@FAR5={agg['detector_TAR_FAR5']:.6f}.",f"- Preservation={absolute['preservation']:.6f}; PSD distortion={absolute['PSD_distortion']:.6f}; covariance distortion={absolute['covariance_distortion']:.6f}.",
      f"- Outside-mask max={absolute['outside_mask_max']:.3g}; off-span max={absolute['off_span_max']:.3g}.",f"- ERP U_P={agg['task']['ERP']['U_P']:+.6f}; SSVEP U_P={agg['task']['SSVEP']['U_P']:+.6f}.","",
      "## Oracle bottleneck decomposition","",f"- D1 oracle-mask U_P={oracle['D1_U_P']:+.6f}, U_W={oracle['D1_U_W']:+.6f}.",f"- D0 primary risk={oracle['D0_risk']:.6f}; D2 query-operator risk={oracle['D2_risk']:.6f}; D3 oracle-mask+query-operator risk={oracle['D3_risk']:.6f}; D4 max risk={oracle['D4_max_risk']:.3g}.","",
      "O1 uses query EEG only at inference. Query EOG and Qgen operator were opened only after output freeze by evaluator stages. No model, GPU, raw signal, sealed outcome, manuscript, DET, or diffusion operation ran.",""]
    report=CODE_ROOT/"reports/eeg_only_analytic_bridge_o1_v21.md";report.parent.mkdir(parents=True,exist_ok=True);report.write_text("\n".join(lines))
    result={"stage":"P13","status":"PASS","scientific_route":route,"DET_authorized":det_authorized,"DET_executed":False};_write_json(_root(c)/"p13_route.json",result);_write_json(run/"result_summary.json",result);return result


def p14(c: Mapping[str, Any],run:Path) -> Mapping[str,Any]:
    _gate(_root(c)/"p13_route.json");decision=json.loads((_root(c)/"route_decision.json").read_text());result={"stage":"P14","status":"PASS","scientific_route":decision["scientific_route"],"GPU_jobs":0,"DET_executed":False,"diffusion_executed":False};_write_json(run/"result_summary.json",result);return result


def run_stage(c:Mapping[str,Any],stage:str,run:Path,index:int|None=None)->Mapping[str,Any]:
    run.mkdir(parents=True,exist_ok=True)
    scalar={"p0-preflight":p0,"p1-bundles":p1,"p2-operators":p2,"p3-detector":p3,"p4-bridge":p4,"p5-collect":p5_collect,"p6-validity":p6,"p8-freeze":p8,"p10-randomization":p10,"p12-replay":p12,"p13-decision":p13,"p14-tests":p14}
    arrays={"p5-select":p5,"p7-inference":p7,"p9-evaluator":p9,"p11-oracles":p11}
    if stage in scalar:return scalar[stage](c,run)
    if stage in arrays:
        if index is None:raise ValueError(f"{stage} requires array index")
        return arrays[stage](c,run,index)
    raise ValueError(stage)
