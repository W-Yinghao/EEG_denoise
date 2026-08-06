"""MobileBCI v5 data repair and fixed 60-second protocol construction."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from eeg_cgdr.data.mobile_bci import (
    metadata_inventory, read_development_record, read_events, read_source_eeg_eog,
)
from eeg_cgdr.experiments.mobile_bci_headroom_v4 import (
    _alignment, _crossfit_clean_proxy, _erp_readout, _finite_summary,
    _motion_coherence, _preprocess_record, _ssvep_readout,
)


CODE_ROOT=Path(os.environ.get("DENOISENET_CODE_ROOT","/home/infres/yinwang/denoiseNet_mobile_diffusion_v5"))


def _write_csv(path:Path,rows:list[Mapping[str,Any]])->None:
    path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({key for row in rows for key in row})
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)


def _write_json(path:Path,value:Mapping[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(value),indent=2,sort_keys=True)+"\n",encoding="utf-8")


def metadata_stage(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    development=list(config["split"]["development"]);sealed=list(config["split"]["sealed"])
    inventory=metadata_inventory(Path(str(config["data_root"])))
    rows=[]
    for row in inventory:
        participant=str(row["participant"]);role="development" if participant in development else "sealed" if participant in sealed else "excluded"
        rows.append({**dict(row),"participant_role":role,"signal_opened":False,"marker_content_opened":False})
    root=CODE_ROOT/str(config["output_root"])/"metadata";_write_csv(root/"file_inventory.csv",rows)
    split_rows=[{"participant":participant,"role":"development","sealed_signal_opened":False} for participant in development]+[{"participant":participant,"role":"sealed","sealed_signal_opened":False} for participant in sealed]
    _write_csv(root/"frozen_participant_split.csv",split_rows)
    folds=[]
    for fold,heldout in config["split"]["outer_folds"].items():
        for participant in development:folds.append({"fold":int(fold),"participant":participant,"split":"validation" if participant in heldout else "training","inner_validation":participant in config["split"]["inner_validation"][fold]})
    _write_csv(root/"development_cv_folds.csv",folds)
    summary={"status":"completed_v5_metadata_only_split","development":development,"sealed":sealed,"sealed_signal_marker_outcome_opened":False,"new_derived_root":str(config["derived_root"])}
    _write_json(root/"split_summary.json",summary);_write_json(run_dir/"result_summary.json",summary);return summary


def _events_100hz(events:list[Mapping[str,str]],rate:float)->tuple[np.ndarray,list[str],np.ndarray]:
    onsets=[];labels=[];durations=[]
    for row in events:
        try:onset=float(row.get("onset","nan"));duration=float(row.get("duration","0") or 0)
        except ValueError:continue
        if np.isfinite(onset):
            # MobileBCI stores onset as a 100-Hz sample index, but duration is
            # already expressed in seconds.  Dividing both fields corrupted
            # the historical SSVEP interval and is repaired only in v5 closure.
            onsets.append(onset/rate);durations.append(max(0.0,duration));labels.append(str(row.get("trial_type",row.get("value",""))))
    return np.asarray(onsets,dtype=np.float64),labels,np.asarray(durations,dtype=np.float64)


def preprocess_participant(config:Mapping[str,Any],run_dir:Path,index:int)->Mapping[str,Any]:
    development=list(config["split"]["development"])
    if not 0<=index<len(development):raise IndexError(index)
    participant=development[index];data_root=Path(str(config["data_root"]));derived=Path(str(config["derived_root"]));prep=config["preprocessing"];rows=[]
    for session in ("ses-02","ses-03","ses-04"):
        for task in ("ERP","SSVEP"):
            try:
                record=read_development_record(data_root,participant,session,task,allowlist=development)
                eeg,imu=_preprocess_record(record,float(prep["target_sampling_rate_hz"]),float(prep["highpass_hz"]),float(prep["lowpass_hz"]))
                if eeg.shape[0]!=46 or imu.shape[0]==0:raise ValueError(f"unexpected processed shape EEG={eeg.shape}, IMU={imu.shape}")
                events=read_events(data_root,participant,session,task,allowlist=development);onsets,labels,durations=_events_100hz(events,float(prep["event_index_rate_hz"]));duration=eeg.shape[1]/float(prep["target_sampling_rate_hz"])
                within=bool(onsets.size and onsets.min()>=0 and onsets.max()<=duration+1e-6)
                if not within:raise ValueError(f"event onset outside duration: max={onsets.max() if onsets.size else None}, duration={duration}")
                destination=derived/participant/session/task;destination.mkdir(parents=True,exist_ok=True)
                np.save(destination/"eeg.npy",eeg);np.save(destination/"imu.npy",imu);np.save(destination/"clean_proxy.npy",_crossfit_clean_proxy(eeg,imu));np.save(destination/"event_onsets.npy",onsets);np.save(destination/"event_durations.npy",durations);(destination/"event_labels.json").write_text(json.dumps(labels)+"\n",encoding="utf-8")
                alignment={"alignment_status":"source_not_checked"}
                try:alignment=_alignment(record,read_source_eeg_eog(data_root,participant,session,task,allowlist=development))
                except Exception as error:alignment={"alignment_status":"source_unavailable_or_unreadable","alignment_failure":f"{type(error).__name__}: {error}"}
                rows.append({"participant":participant,"session":session,"task":task,"status":"success","duration_seconds":duration,"event_count":len(onsets),"event_onsets_within_duration":within,"processed_eeg_channels":eeg.shape[0],"processed_imu_channels":imu.shape[0],"support_eog_enabled":False,"experiment_name":"processed_EEG_plus_IMU",**_finite_summary(eeg,"processed_eeg"),**_finite_summary(imu,"processed_imu"),**alignment,"sealed_signal_opened":False})
            except Exception as error:rows.append({"participant":participant,"session":session,"task":task,"status":"missing" if isinstance(error,FileNotFoundError) else "failed","failure_reason":f"{type(error).__name__}: {error}","event_onsets_within_duration":False,"support_eog_enabled":False,"sealed_signal_opened":False})
    output=CODE_ROOT/str(config["output_root"])/"data_audit";_write_csv(output/f"{participant}.csv",rows)
    summary={"status":"completed_v5_development_preprocess","participant":participant,"successful":sum(row["status"]=="success" for row in rows),"denominator":6,"all_successful_events_within_duration":all(bool(row["event_onsets_within_duration"]) for row in rows if row["status"]=="success"),"sealed_signal_opened":False};_write_json(run_dir/"result_summary.json",summary);return summary


def _cache(config:Mapping[str,Any],participant:str,session:str,task:str,name:str)->np.ndarray:
    return np.load(Path(str(config["derived_root"]))/participant/session/task/f"{name}.npy",mmap_mode="r")


def protocol_stage(config:Mapping[str,Any],run_dir:Path)->Mapping[str,Any]:
    development=list(config["split"]["development"]);audit_root=CODE_ROOT/str(config["output_root"])/"data_audit";audit=[]
    for path in sorted(audit_root.glob("sub-*.csv")):
        with path.open(encoding="utf-8",newline="") as stream:audit.extend(csv.DictReader(stream))
    good={(row["participant"],row["session"],row["task"]) for row in audit if row["status"]=="success" and row["event_onsets_within_duration"].lower()=="true"}
    if any(row["status"]=="success" and row["event_onsets_within_duration"].lower()!="true" for row in audit):raise AssertionError("successful record violates event_onsets_within_duration")
    rate=float(config["preprocessing"]["target_sampling_rate_hz"]);support=60.0;guard=5.0;window=float(config["preprocessing"]["window_seconds"]);protocols=[];readouts=[]
    for participant in development:
        for session in ("ses-02","ses-03","ses-04"):
            for task in ("ERP","SSVEP"):
                key=(participant,session,task)
                if key not in good:continue
                eeg=np.asarray(_cache(config,participant,session,task,"eeg"));imu=np.asarray(_cache(config,participant,session,task,"imu"));onsets=np.asarray(_cache(config,participant,session,task,"event_onsets"));durations=np.asarray(_cache(config,participant,session,task,"event_durations"));labels=json.loads((Path(str(config["derived_root"]))/participant/session/task/"event_labels.json").read_text());metric=_erp_readout(eeg,onsets,labels,rate) if task=="ERP" else _ssvep_readout(eeg,onsets,labels,durations,rate);readouts.append({"participant":participant,"session":session,"task":task,"status":"success","motion_coherence":_motion_coherence(eeg,imu,rate),**metric})
        for task in ("ERP","SSVEP"):
            def duration(session:str)->float:return _cache(config,participant,session,task,"eeg").shape[1]/rate if (participant,session,task) in good else 0.0
            for query in ("ses-03","ses-04"):
                status="eligible" if duration("ses-02")>=support and duration(query)>=window else "missing"
                protocols.append({"protocol":"S0_STATIC_XSESSION","participant":participant,"task":task,"support_session":"ses-02","query_session":query,"support_start":0.0,"support_end":support,"query_start":0.0,"status":status})
            for session in ("ses-03","ses-04"):
                status="eligible" if duration(session)>=support+guard+window else "blocked_no_later_block" if duration(session)>0 else "missing"
                protocols.append({"protocol":"S1_MOTION_WITHIN_SESSION","participant":participant,"task":task,"support_session":session,"query_session":session,"support_start":0.0,"support_end":support,"query_start":support+guard,"status":status})
            status="eligible" if duration("ses-03")>=support and duration("ses-04")>=window else "missing"
            protocols.append({"protocol":"S2_MOTION_XSPEED","participant":participant,"task":task,"support_session":"ses-03","query_session":"ses-04","support_start":0.0,"support_end":support,"query_start":0.0,"status":status})
    # Metadata/protocol-compatible, deterministic wrong donors; no signal/outcome matching.
    for row in protocols:
        candidates=[p for p in development if p!=row["participant"] and (p,row["support_session"],row["task"]) in good and _cache(config,p,row["support_session"],row["task"],"eeg").shape[1]/rate>=support]
        if row["status"]=="eligible" and len(candidates)<3:row["status"]="missing"
        if len(candidates)>=3:
            pivot=sum(ord(c) for c in row["participant"]+row["protocol"]+row["task"])%len(candidates)
            for index in (1,2,3):row[f"wrong_donor_{index}"]=candidates[(pivot+index-1)%len(candidates)]
        else:
            for index in (1,2,3):row[f"wrong_donor_{index}"]=""
    output=CODE_ROOT/str(config["output_root"])/"protocol";_write_csv(output/"frozen_protocol_units.csv",protocols);_write_csv(output/"raw_readout_metrics.csv",readouts)
    counts={status:sum(row["status"]==status for row in protocols) for status in ("eligible","blocked_no_later_block","missing")};task_counts={task:sum(row["status"]=="eligible" and row["task"]==task for row in protocols) for task in ("ERP","SSVEP")}
    summary={"status":"science_ready" if counts["eligible"]>0 else "MOBILE_DATA_TECHNICAL_NO_GO","experiment_name":"processed_EEG_plus_IMU","source_eog_enabled":False,"processed_source_alignment_reliable":False,"eligible_protocol_units":counts["eligible"],"blocked_no_later_block":counts["blocked_no_later_block"],"missing_protocol_units":counts["missing"],"eligible_by_task":task_counts,"protocol_denominator":len(protocols),"all_successful_events_within_duration":True,"support_budget_seconds":support,"guard_seconds":guard,"sealed_signal_opened":False};_write_json(output/"result_summary.json",summary);_write_json(run_dir/"result_summary.json",summary);return summary
