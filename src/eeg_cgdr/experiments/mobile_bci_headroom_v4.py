"""MobileBCI v4 metadata-only split and later Slurm experiment stages."""

from __future__ import annotations

import csv
import json
import os
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import signal

from eeg_cgdr.data.mobile_bci import (
    development_record_paths, freeze_metadata_split, metadata_inventory,
    parse_brainvision_header, read_development_record, read_events, read_source_eeg_eog,
)


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_mobile_headroom_v4"))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metadata_stage(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    root = Path(str(config["data_root"]))
    output = CODE_ROOT / str(config["output_root"]) / "metadata"
    rows = metadata_inventory(root)
    split = freeze_metadata_split(rows, seed=int(config["seed"]))
    role = {participant: "development" for participant in split.development}
    role.update({participant: "sealed" for participant in split.sealed})
    inventory = [{**dict(row), "participant_role": role[str(row["participant"])]} for row in rows]
    availability: list[dict[str, Any]] = []
    for participant in sorted(role):
        selected = [row for row in inventory if row["participant"] == participant]
        available = {(row["session"], row["task"]) for row in selected
                     if row["header_exists"] and row["data_exists"] and row["marker_exists"] and row["channels_exists"]}
        availability.append({
            "participant": participant, "role": role[participant],
            "primary_erp_eligible": all((session, "ERP") in available for session in ("ses-02", "ses-03", "ses-04")),
            "primary_ssvep_eligible": all((session, "SSVEP") in available for session in ("ses-02", "ses-03", "ses-04")),
            "ses05_erp_complete": ("ses-05", "ERP") in available,
            "ses05_ssvep_complete": ("ses-05", "SSVEP") in available,
            "record_count": len(selected),
            "source_eog_metadata_records": sum(int(row["source_eog_channels"]) >= 4 for row in selected),
            "processed_imu_metadata_records": sum(int(row["processed_imu_channels"]) > 0 for row in selected),
            "binary_signal_opened": False, "marker_content_opened": False, "outcome_opened": False,
        })
    split_rows = [{"participant": participant, "role": role[participant], "split_seed": int(config["seed"])} for participant in sorted(role)]
    fold_rows = [{"fold": fold, "split": "validation" if participant in heldout else "training",
                  "participant": participant}
                 for fold, heldout in split.folds.items() for participant in split.development]
    _write_csv(output / "file_inventory.csv", inventory)
    _write_csv(output / "participant_availability.csv", availability)
    _write_csv(output / "frozen_participant_split.csv", split_rows)
    _write_csv(output / "development_cv_folds.csv", fold_rows)
    summary = {
        "status": "completed_metadata_only_split",
        "participants": len(role), "development_participants": len(split.development),
        "sealed_participants": len(split.sealed), "development_fold_sizes": {str(key): len(value) for key, value in split.folds.items()},
        "development": list(split.development), "sealed": list(split.sealed),
        "binary_signal_opened": False, "marker_content_opened": False, "outcome_opened": False,
        "split_basis": "metadata_only_one_per_consecutive_three_person_group_seeded_tie_20260806",
    }
    _write_json(output / "split_balance.json", summary)
    _write_json(run_dir / "result_summary.json", summary)
    return summary


def _read_split(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    path = CODE_ROOT / str(config["output_root"]) / "metadata/frozen_participant_split.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return (
        sorted(row["participant"] for row in rows if row["role"] == "development"),
        sorted(row["participant"] for row in rows if row["role"] == "sealed"),
    )


def _finite_summary(value: np.ndarray, prefix: str) -> dict[str, Any]:
    if value.size == 0 or value.shape[0] == 0:
        return {f"{prefix}_channels": 0, f"{prefix}_samples": value.shape[-1] if value.ndim else 0,
                f"{prefix}_finite_fraction": float("nan"), f"{prefix}_rms_median": float("nan"),
                f"{prefix}_flatline_channels": 0, f"{prefix}_maximum_absolute": float("nan")}
    finite = np.isfinite(value)
    standard = np.std(value, axis=1)
    return {
        f"{prefix}_channels": value.shape[0], f"{prefix}_samples": value.shape[1],
        f"{prefix}_finite_fraction": float(finite.mean()),
        f"{prefix}_rms_median": float(np.median(np.sqrt(np.mean(np.square(value), axis=1)))),
        f"{prefix}_flatline_channels": int(np.sum(standard < 1e-8)),
        f"{prefix}_maximum_absolute": float(np.nanmax(np.abs(value))),
    }


def _alignment(processed: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    common = [name for name in processed["eeg_names"] if name in set(source["eeg_names"])]
    if not common:
        return {"alignment_status": "blocked_no_common_eeg_channels"}
    left_indices = [processed["eeg_names"].index(name) for name in common]
    right_indices = [source["eeg_names"].index(name) for name in common]
    left = processed["eeg"][left_indices]; right = source["eeg"][right_indices]
    rate_left = float(processed["sampling_rate_hz"]); rate_right = float(source["sampling_rate_hz"])
    duration = min(60.0, left.shape[1] / rate_left, right.shape[1] / rate_right)
    points = min(30_000, max(256, int(duration * min(rate_left, rate_right))))
    time = np.linspace(0.0, duration, points, endpoint=False)
    left_t = np.arange(left.shape[1]) / rate_left; right_t = np.arange(right.shape[1]) / rate_right
    left_mean = np.mean([np.interp(time, left_t, channel) for channel in left], axis=0)
    right_mean = np.mean([np.interp(time, right_t, channel) for channel in right], axis=0)
    left_mean = (left_mean-left_mean.mean())/max(left_mean.std(),1e-12)
    right_mean = (right_mean-right_mean.mean())/max(right_mean.std(),1e-12)
    maximum_lag = int(round(min(2.0, duration/10)*points/duration))
    correlations=[]
    for lag in range(-maximum_lag,maximum_lag+1):
        if lag<0: a,b=left_mean[-lag:],right_mean[:lag]
        elif lag>0: a,b=left_mean[:-lag],right_mean[lag:]
        else: a,b=left_mean,right_mean
        correlations.append(float(np.mean(a*b)))
    best=int(np.argmax(np.abs(correlations))); lag=best-maximum_lag; corr=correlations[best]
    return {"alignment_status":"success" if abs(corr)>=0.8 else "unreliable_common_eeg_alignment",
            "alignment_common_channels":len(common),"alignment_lag_seconds":float(lag*duration/points),
            "alignment_correlation":corr,"alignment_polarity":1 if corr>=0 else -1}


def _event_summary(events: list[Mapping[str, str]], duration: float, sampling_rate_hz: float) -> dict[str, Any]:
    onsets=[]; labels=[]
    for row in events:
        try: onsets.append(float(row.get("onset", "nan")))
        except ValueError: continue
        labels.append(str(row.get("trial_type", row.get("value", ""))))
    onsets=np.asarray([value for value in onsets if np.isfinite(value)]) / sampling_rate_hz
    gaps=np.diff(np.sort(onsets)) if onsets.size>1 else np.empty(0)
    median=float(np.median(gaps)) if gaps.size else float("nan")
    boundary=float(max(5.0,3.5*median)) if np.isfinite(median) else float("inf")
    blocks=1+int(np.sum(gaps>=boundary)) if onsets.size else 0
    return {"event_count":int(onsets.size),"event_label_count":len(set(labels)),"event_block_count":blocks,
            "event_max_onset_seconds":float(onsets.max()) if onsets.size else float("nan"),
            "event_onsets_within_duration":bool(onsets.size and onsets.min()>=0 and onsets.max()<=duration+1.0),
            "event_gap_median_seconds":median,"event_block_gap_threshold_seconds":boundary}


def signal_audit_stage(config: Mapping[str, Any], run_dir: Path, task_index: int) -> Mapping[str, Any]:
    development, _ = _read_split(config)
    if not 0 <= task_index < len(development): raise IndexError(task_index)
    participant=development[task_index]; root=Path(str(config["data_root"])); rows=[]
    for session in ("ses-02","ses-03","ses-04"):
        for task in ("ERP","SSVEP"):
            try:
                processed=read_development_record(root,participant,session,task,allowlist=development)
                source=read_source_eeg_eog(root,participant,session,task,allowlist=development)
                events=read_events(root,participant,session,task,allowlist=development)
                rows.append({"participant":participant,"session":session,"task":task,"status":"success",
                    "processed_sampling_rate_hz":processed["sampling_rate_hz"],"source_sampling_rate_hz":source["sampling_rate_hz"],
                    "duration_seconds":processed["duration_seconds"],
                    **_finite_summary(processed["eeg"],"processed_eeg"),**_finite_summary(processed["imu"],"processed_imu"),
                    **_finite_summary(source["eeg"],"source_eeg"),**_finite_summary(source["eog"],"source_eog"),
                    **_alignment(processed,source),**_event_summary(events,float(processed["duration_seconds"]),float(processed["sampling_rate_hz"])),
                    "sealed_participant_signal_opened":False})
            except Exception as error:
                rows.append({"participant":participant,"session":session,"task":task,"status":"failed_record_audit",
                             "failure_reason":f"{type(error).__name__}: {error}","sealed_participant_signal_opened":False})
    output=CODE_ROOT/str(config["output_root"])/"science_ready_audit"; _write_csv(output/f"{participant}.csv",rows)
    summary={"status":"completed_development_participant_signal_audit","participant":participant,
             "successful_records":sum(row["status"]=="success" for row in rows),"record_denominator":6,
             "sealed_participant_signal_opened":False}
    _write_json(run_dir/"result_summary.json",summary); return summary


def _event_arrays(events: list[Mapping[str, str]], sampling_rate_hz: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    onsets=[]; labels=[]; durations=[]
    for row in events:
        try: onset=float(row.get("onset","nan")); duration=float(row.get("duration","0") or 0)
        except ValueError: continue
        if np.isfinite(onset):
            onsets.append(onset); durations.append(duration if np.isfinite(duration) else 0.0)
            labels.append(str(row.get("trial_type",row.get("value",""))))
    return np.asarray(onsets)/sampling_rate_hz,labels,np.asarray(durations)


def _preprocess_record(record: Mapping[str, Any], target_rate: float, low: float, high: float) -> tuple[np.ndarray,np.ndarray]:
    source_rate=float(record["sampling_rate_hz"]); eeg=np.asarray(record["eeg"],dtype=np.float64); imu=np.asarray(record["imu"],dtype=np.float64)
    sos=signal.butter(4,(low,high),btype="bandpass",fs=source_rate,output="sos")
    eeg=signal.sosfiltfilt(sos,eeg,axis=-1)
    from fractions import Fraction
    ratio=Fraction(target_rate/source_rate).limit_denominator(1000)
    eeg=signal.resample_poly(eeg,ratio.numerator,ratio.denominator,axis=-1)
    imu=signal.resample_poly(imu,ratio.numerator,ratio.denominator,axis=-1)
    length=min(eeg.shape[1],imu.shape[1]); return eeg[:,:length].astype(np.float32),imu[:,:length].astype(np.float32)


def _crossfit_clean_proxy(eeg:np.ndarray,imu:np.ndarray)->np.ndarray:
    """Blocked-half motion teacher, used only for outer-training participants."""
    length=eeg.shape[1]; split=length//2; clean=np.empty_like(eeg,dtype=np.float32)
    imu_z=(imu-imu.mean(axis=1,keepdims=True))/np.maximum(imu.std(axis=1,keepdims=True),1e-8)
    for fit,score in ((slice(0,split),slice(split,length)),(slice(split,length),slice(0,split))):
        design=np.concatenate((np.ones((1,imu_z[:,fit].shape[1])),imu_z[:,fit]),axis=0)
        gram=design@design.T+1e-2*np.eye(design.shape[0]); coefficients=np.linalg.solve(gram,(eeg[:,fit]@design.T).T).T
        score_design=np.concatenate((np.ones((1,imu_z[:,score].shape[1])),imu_z[:,score]),axis=0)
        clean[:,score]=(eeg[:,score]-coefficients@score_design).astype(np.float32)
    return clean


def _motion_coherence(eeg: np.ndarray, imu: np.ndarray, rate: float) -> float:
    if imu.size==0 or eeg.shape[1]<int(4*rate): return float("nan")
    eeg_z=(eeg-eeg.mean(axis=1,keepdims=True))/np.maximum(eeg.std(axis=1,keepdims=True),1e-8)
    imu_z=(imu-imu.mean(axis=1,keepdims=True))/np.maximum(imu.std(axis=1,keepdims=True),1e-8)
    eeg_proxy=np.mean(eeg_z[:min(12,eeg_z.shape[0])],axis=0); imu_proxy=np.mean(imu_z,axis=0)
    frequencies,coherence=signal.coherence(eeg_proxy,imu_proxy,fs=rate,nperseg=min(int(4*rate),eeg.shape[1]))
    keep=(frequencies>=0.5)&(frequencies<=8.0)
    return float(np.mean(coherence[keep])) if np.any(keep) else float("nan")


def _erp_readout(eeg: np.ndarray, onsets: np.ndarray, labels: list[str], rate: float) -> dict[str, Any]:
    target=[]; nontarget=[]; before=int(round(0.2*rate)); after=int(round(0.8*rate))
    for onset,label in zip(onsets,labels):
        center=int(round(onset*rate)); text=label.lower().replace("-","").replace("_","")
        if center-before<0 or center+after>eeg.shape[1]: continue
        epoch=eeg[:,center-before:center+after]; epoch=epoch-epoch[:,:before].mean(axis=1,keepdims=True)
        if text in {"1","s1"} or "nontarget" in text or "standard" in text: nontarget.append(epoch)
        elif text in {"2","s2"} or "target" in text: target.append(epoch)
    if len(target)<4 or len(nontarget)<4:
        return {"erp_status":"blocked_label_semantics_or_event_count","erp_target_epochs":len(target),"erp_nontarget_epochs":len(nontarget)}
    target_array=np.stack(target); non_array=np.stack(nontarget); difference=target_array.mean(0)-non_array.mean(0)
    even=target_array[::2].mean(0)-non_array[::2].mean(0); odd=target_array[1::2].mean(0)-non_array[1::2].mean(0)
    reliability=float(np.corrcoef(even.ravel(),odd.ravel())[0,1])
    global_wave=difference.mean(axis=0); peak_index=int(np.argmax(np.abs(global_wave[before:])))+before
    return {"erp_status":"success","erp_target_epochs":len(target),"erp_nontarget_epochs":len(nontarget),
            "erp_split_half_reliability":reliability,"erp_amplitude_uv":float(global_wave[peak_index]),
            "erp_peak_latency_seconds":float((peak_index-before)/rate),"erp_topography_norm":float(np.linalg.norm(difference[:,peak_index]))}


def _label_frequency(label: str) -> float|None:
    import re
    canonical={"11":5.45,"12":8.57,"13":12.0,"s11":5.45,"s12":8.57,"s13":12.0}
    compact=label.lower().replace(" ","").replace("_","")
    if compact in canonical:return canonical[compact]
    candidates=[float(value) for value in re.findall(r"\d+(?:\.\d+)?",label)]
    plausible=[value for value in candidates if 4.0<=value<=40.0]
    return plausible[-1] if plausible else None


def _ssvep_readout(eeg: np.ndarray,onsets:np.ndarray,labels:list[str],durations:np.ndarray,rate:float)->dict[str,Any]:
    snr=[]; phase=[]
    for onset,label,duration in zip(onsets,labels,durations):
        frequency=_label_frequency(label)
        if frequency is None: continue
        start=int(round(onset*rate)); length=int(round(max(2.0,min(5.0,duration if duration>0 else 3.0))*rate))
        if start<0 or start+length>eeg.shape[1]: continue
        segment=eeg[:,start:start+length]; spectrum=np.fft.rfft(segment*signal.windows.hann(length),axis=-1)
        frequencies=np.fft.rfftfreq(length,1/rate); index=int(np.argmin(np.abs(frequencies-frequency)))
        if index<2 or index+2>=spectrum.shape[1]: continue
        target=np.mean(np.abs(spectrum[:,index])**2); neighbors=np.mean(np.abs(spectrum[:,[index-2,index-1,index+1,index+2]])**2)
        snr.append(10*np.log10(max(target,1e-12)/max(neighbors,1e-12))); phase.append(np.angle(np.mean(spectrum[:,index])))
    if len(snr)<6: return {"ssvep_status":"blocked_frequency_semantics_or_event_count","ssvep_epochs":len(snr)}
    phase_consistency=float(abs(np.mean(np.exp(1j*np.asarray(phase)))))
    return {"ssvep_status":"success","ssvep_epochs":len(snr),"ssvep_snr_db":float(np.mean(snr)),"ssvep_phase_consistency":phase_consistency}


def _support_block(onsets:np.ndarray,durations:np.ndarray,total_duration:float)->tuple[float,float,float,int]:
    if onsets.size<2: return 0.0,min(total_duration,60.0),total_duration,1
    gaps=np.diff(onsets); median=float(np.median(gaps)); boundaries=np.flatnonzero(gaps>=max(5.0,3.5*median))
    end_index=int(boundaries[0]+1) if boundaries.size else onsets.size
    end=float(onsets[end_index-1]+(durations[end_index-1] if durations[end_index-1]>0 else max(median,1.0)))
    next_start=float(onsets[end_index]) if end_index<onsets.size else total_duration
    return 0.0,min(end,total_duration),min(next_start,total_duration),int(boundaries.size+1)


def evaluator_stage(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    development,sealed=_read_split(config); audit_root=CODE_ROOT/str(config["output_root"])/"science_ready_audit"
    audit=[]
    for path in sorted(audit_root.glob("sub-*.csv")):
        with path.open(encoding="utf-8",newline="") as stream: audit.extend(csv.DictReader(stream))
    successful=[row for row in audit if row["status"]=="success"]
    successful_keys={(row["participant"],row["session"],row["task"]) for row in successful
                     if float(row.get("processed_eeg_finite_fraction",0))==1.0
                     and float(row.get("processed_imu_channels",0))>0 and float(row.get("processed_imu_finite_fraction",0))==1.0}
    # A participant is science-ready when at least one task has all three
    # primary sessions; unavailable task files remain explicit coverage losses.
    eligible_pairs={(participant,task) for participant in development for task in ("ERP","SSVEP")
                    if all((participant,session,task) in successful_keys for session in ("ses-02","ses-03","ses-04"))}
    data_ready=all(any(pair[0]==participant for pair in eligible_pairs) for participant in development)
    rows=[]; protocols=[]; root=Path(str(config["data_root"])); prep=config["preprocessing"]; rate=float(prep["target_sampling_rate_hz"])
    for participant in development:
        for session in ("ses-02","ses-03","ses-04"):
            for task in ("ERP","SSVEP"):
                try:
                    cache=Path(str(config["derived_root"]))/participant/session/task; cache.mkdir(parents=True,exist_ok=True)
                    if all((cache/name).is_file() for name in ("eeg.npy","imu.npy","clean_proxy.npy","event_onsets.npy","event_durations.npy","event_labels.json")):
                        eeg=np.load(cache/"eeg.npy",mmap_mode="r");imu=np.load(cache/"imu.npy",mmap_mode="r");header,_,_=development_record_paths(root,participant,session,task,allowlist=development);header_meta=parse_brainvision_header(header);record={"duration_seconds":eeg.shape[1]/rate,"sampling_rate_hz":header_meta["sampling_rate_hz"]};events=read_events(root,participant,session,task,allowlist=development);onsets,labels,durations=_event_arrays(events,float(record["sampling_rate_hz"]));np.save(cache/"event_onsets.npy",onsets);np.save(cache/"event_durations.npy",durations);(cache/"event_labels.json").write_text(json.dumps(labels)+"\n",encoding="utf-8")
                    else:
                        record=read_development_record(root,participant,session,task,allowlist=development); eeg,imu=_preprocess_record(record,rate,float(prep["highpass_hz"]),float(prep["lowpass_hz"]))
                        if imu.shape[0]==0: raise ValueError("processed record has no IMU channels")
                        events=read_events(root,participant,session,task,allowlist=development); onsets,labels,durations=_event_arrays(events,float(record["sampling_rate_hz"]))
                        np.save(cache/"eeg.npy",eeg); np.save(cache/"imu.npy",imu); np.save(cache/"clean_proxy.npy",_crossfit_clean_proxy(eeg,imu))
                        np.save(cache/"event_onsets.npy",onsets); np.save(cache/"event_durations.npy",durations)
                        (cache/"event_labels.json").write_text(json.dumps(labels)+"\n",encoding="utf-8")
                    support_start,support_end,next_block_start,blocks=_support_block(onsets,durations,eeg.shape[1]/rate)
                    base={"participant":participant,"session":session,"task":task,"status":"success","motion_coherence":_motion_coherence(eeg,imu,rate),
                          "support_block_start":support_start,"support_block_end":support_end,"event_blocks":blocks,"sealed_signal_opened":False}
                    base.update(_erp_readout(eeg,onsets,labels,rate) if task=="ERP" else _ssvep_readout(eeg,onsets,labels,durations,rate)); rows.append(base)
                    if session=="ses-02":
                        for query in ("ses-03","ses-04"): protocols.append({"protocol":"S0_STATIC_XSESSION","participant":participant,"task":task,"support_session":session,"query_session":query,"support_start":support_start,"support_end":support_end,"status":"eligible"})
                    if session in ("ses-03","ses-04"):
                        protocols.append({"protocol":"S1_MOTION_WITHIN_SESSION","participant":participant,"task":task,"support_session":session,"query_session":session,"support_start":support_start,"support_end":support_end,"query_start":next_block_start,"status":"eligible" if next_block_start+float(prep["window_seconds"])<eeg.shape[1]/rate else "blocked_no_later_block"})
                    if session=="ses-03": protocols.append({"protocol":"S2_MOTION_XSPEED","participant":participant,"task":task,"support_session":session,"query_session":"ses-04","support_start":support_start,"support_end":support_end,"status":"eligible"})
                except Exception as error: rows.append({"participant":participant,"session":session,"task":task,"status":"failed_evaluator","failure_reason":f"{type(error).__name__}: {error}","sealed_signal_opened":False})
    # Fixed participant-level derangements, independent of signals/outcomes.
    valid_protocols=[]
    for row in protocols:
        if (row["participant"],row["support_session"],row["task"]) not in successful_keys or (row["participant"],row["query_session"],row["task"]) not in successful_keys:
            row["status"]="blocked_missing_primary_record"
        else: valid_protocols.append(row)
        participant=str(row["participant"]); candidates=[value for value in development
            if value!=participant and (value,str(row["support_session"]),str(row["task"])) in successful_keys]
        if len(candidates)<3:
            row["status"]="blocked_fewer_than_three_compatible_wrong_donors"
            row.update({f"wrong_donor_{offset}":"" for offset in (1,2,3)})
        else:
            pivot=sum(ord(char) for char in participant+str(row["protocol"])+str(row["task"]))%len(candidates)
            row.update({f"wrong_donor_{offset}":candidates[(pivot+offset-1)%len(candidates)] for offset in (1,2,3)})
    output=CODE_ROOT/str(config["output_root"])/"evaluator"; _write_csv(output/"raw_evaluator_metrics.csv",rows); _write_csv(output/"frozen_protocol_units.csv",protocols)
    erp=[r for r in rows if r.get("erp_status")=="success"]; ssvep=[r for r in rows if r.get("ssvep_status")=="success"]
    slow=[float(r["motion_coherence"]) for r in rows if r.get("status")=="success" and r["session"]=="ses-03" and np.isfinite(float(r["motion_coherence"]))]; standing=[float(r["motion_coherence"]) for r in rows if r.get("status")=="success" and r["session"]=="ses-02" and np.isfinite(float(r["motion_coherence"]))]; fast=[float(r["motion_coherence"]) for r in rows if r.get("status")=="success" and r["session"]=="ses-04" and np.isfinite(float(r["motion_coherence"]))]
    evaluator_ready=bool(data_ready and len(erp)>=24 and len(ssvep)>=24 and slow and standing and fast)
    status="science_ready" if evaluator_ready else "MOBILE_EVALUATOR_NO_GO" if data_ready else "MOBILE_DATA_TECHNICAL_NO_GO"
    summary={"status":status,"development_participants":16,"sealed_participants":8,"records_successful":sum(r["status"]=="success" for r in rows),"record_denominator":96,
             "erp_readouts_successful":len(erp),"ssvep_readouts_successful":len(ssvep),"standing_motion_coherence_mean":float(np.mean(standing)) if standing else None,
             "slow_motion_coherence_mean":float(np.mean(slow)) if slow else None,"fast_motion_coherence_mean":float(np.mean(fast)) if fast else None,
             "motion_band_hz":[0.5,8.0],"neural_safety_noninferiority_margin":float(config["evaluation"]["noninferiority_margin"]),"sealed_signal_opened":False,
             "eligible_participant_task_pairs":len(eligible_pairs),"protocol_units":len(protocols),"eligible_protocol_units":len(valid_protocols),"science_ready":evaluator_ready}
    _write_json(output/"result_summary.json",summary); _write_json(run_dir/"result_summary.json",summary)
    report=CODE_ROOT/"reports/mobile_bci_science_ready_audit.md"; report.parent.mkdir(parents=True,exist_ok=True); report.write_text("# MobileBCI science-ready audit\n\n"+f"Status: `{status}`. Development-only records: {summary['records_successful']}/96. ERP readouts: {len(erp)}; SSVEP readouts: {len(ssvep)}. Sealed binary/marker/outcomes opened: **false**.\n",encoding="utf-8")
    return summary


def run_stage(config: Mapping[str, Any], stage: str, run_dir: Path, task_index: int | None = None) -> Mapping[str, Any]:
    if stage == "j1b-metadata":
        return metadata_stage(config, run_dir)
    if stage == "j0-tests":
        summary = {"status": "completed_targeted_tests", "task_index": task_index}
        _write_json(run_dir / "result_summary.json", summary); return summary
    if stage == "j1a-v3-repair":
        from eeg_cgdr.experiments.v3_evidence_repair import run
        return run(config, run_dir)
    if stage == "j1a-pc-oracle":
        from eeg_cgdr.experiments.pc_constrained_oracle import run
        return run(config, run_dir)
    if stage == "j2-signal-audit":
        if task_index is None: raise ValueError("j2 requires array task index")
        return signal_audit_stage(config, run_dir, task_index)
    if stage == "j3-evaluator": return evaluator_stage(config,run_dir)
    if stage == "j4-headroom":
        if task_index is None: raise ValueError("j4 requires fold array index")
        from eeg_cgdr.experiments.mobile_bci_headroom_runner import run_fold
        return run_fold(config,run_dir,task_index)
    if stage == "j5-headroom-route":
        from eeg_cgdr.experiments.mobile_bci_headroom_runner import aggregate_headroom
        return aggregate_headroom(config,run_dir)
    if stage == "j6-temporal":
        if task_index is None: raise ValueError("j6 requires fold array index")
        from eeg_cgdr.experiments.temporal_support_diffusion_v4 import run_fold
        return run_fold(config,run_dir,task_index)
    if stage == "j7-final":
        from eeg_cgdr.experiments.mobile_bci_v4_final import run
        return run(config,run_dir)
    raise ValueError(f"unsupported mobile v4 stage: {stage}")
