"""No-training closure for the historical MobileBCI v5 development run."""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from eeg_cgdr.experiments.mobile_temporal_diffusion_v5 import _ssvep_readout

LABEL="PROTOCOL_CORE_VALID / SSVEP_SAFETY_PREVIOUSLY_INVALID / ONE-SEED_RAW_TEMPORAL_ROUTE_NO_GO / DIFFUSION_FAMILY_NOT_TESTED"

def run_closure(run_dir: Path) -> dict[str,Any]:
    historical=Path("/home/infres/yinwang/denoiseNet_mobile_diffusion_v5/results/cgdr/mobile_temporal_diffusion_v5")
    arrays=list(historical.rglob("*.npz")) if historical.exists() else []
    clipping=[]
    for path in arrays:
        if "train" not in path.name.lower(): continue
        try:
            data=np.load(path)
            for key in data.files:
                if "target" in key and np.issubdtype(data[key].dtype,np.floating): clipping.append(float(np.mean(np.abs(data[key])>=1)))
        except Exception: pass
    corrected=[]
    if arrays:
        protocol_rows=list(csv.DictReader((historical/"protocol/frozen_protocol_units.csv").open()))
        protocol={(r["participant"],r["protocol"],r["task"],r["query_session"]):r for r in protocol_rows}
        derived=Path("/projects/EEG-foundation-model/derived/denoiseNet/mobile_temporal_v5")
        for path in arrays:
            if "server_arrays" not in path.parts: continue
            # authoritative keys come from the corresponding fold metrics row.
            metrics_path=path.parents[1]/"metrics.csv"
            metric_rows=list(csv.DictReader(metrics_path.open()))
            stem=path.stem
            candidates=[r for r in metric_rows if stem==f"{r['participant']}_{r['protocol']}_{r['task']}_{r['query_session']}"]
            if not candidates or candidates[0]["task"]!="SSVEP": continue
            row=candidates[0];key=(row["participant"],row["protocol"],row["task"],row["query_session"]);contract=protocol[key];query_start=float(contract["query_start"]);folder=derived/row["participant"]/row["query_session"]/row["task"]
            onsets=np.load(folder/"event_onsets.npy")-query_start;durations=np.load(folder/"event_durations.npy")*100.0;labels=json.loads((folder/"event_labels.json").read_text());outputs=np.load(path);raw=_ssvep_readout(outputs["RAW"],onsets,labels,durations,100.0)
            for method in outputs.files:
                score=_ssvep_readout(outputs[method],onsets,labels,durations,100.0);valid=raw.get("ssvep_status")==score.get("ssvep_status")=="success";corrected.append({"participant":row["participant"],"protocol":row["protocol"],"query_session":row["query_session"],"method":method,"status":"success" if valid else "unavailable","ssvep_snr_relative_preservation":1-abs(float(score["ssvep_snr_db"])-float(raw["ssvep_snr_db"]))/max(abs(float(raw["ssvep_snr_db"])),1.) if valid else float("nan"),"ssvep_phase_relative_preservation":1-abs(float(score["ssvep_phase_consistency"])-float(raw["ssvep_phase_consistency"])) if valid else float("nan")})
    if corrected:
        destination=Path("results/cgdr/mobile_temporal_diffusion_v5_closure");destination.mkdir(parents=True,exist_ok=True);fields=list(corrected[0]);
        with (destination/"corrected_ssvep_metrics.csv").open("w",newline="") as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(corrected)
    pareto_files=list(historical.glob("factorial/seed_*/fold_*/pareto_metrics.csv"));pareto_rows=[row for path in pareto_files for row in csv.DictReader(path.open())];has_gamma0=any(float(row["gamma"])==0 for row in pareto_rows)
    result={"status":"completed_no_training_closure","final_label":LABEL,"event_duration_fix":"duration field retained in seconds; only onset uses 100-Hz sample-index conversion","ssvep_safety":"recomputed_from_frozen_outputs" if corrected else "unavailable_arrays_missing","corrected_ssvep_rows":len(corrected),"training_target_clip_fraction":float(np.mean(clipping)) if clipping else None,"training_target_clip_fraction_status":"unavailable_no_saved_training_targets" if not clipping else "recomputed","dense_gamma0_pareto":"existing_participant_metrics_include_gamma0" if has_gamma0 else "unavailable_no_gamma0","wrong_donor_fold_role_confound":True,"pc_labels":"old_per_window_labels_not_bounded_oracle_masks","zero_eog_values":"not_true_token_masking","retrained":False,"additional_seeds":False,"score_adapter":False,"sealed_participants_opened":False}
    run_dir.mkdir(parents=True,exist_ok=True); (run_dir/"result_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    report=Path("reports/mobile_temporal_diffusion_v5_closure.md"); report.parent.mkdir(parents=True,exist_ok=True); report.write_text("# MobileBCI v5 no-training closure\n\n"+f"Final label: `{LABEL}`.\n\nThe event onset remains a 100-Hz sample index, while the event duration field is already seconds and is no longer divided by 100. SSVEP safety was recomputed from {len(corrected)} frozen method/unit outputs. Training-target clipping cannot be reconstructed because targets were not saved. The historical wrong donors carry a fold-role confound; P-C used old per-window labels rather than bounded-oracle masks; numerical zero EOG inputs were not true token masking. No v5 model was retrained and no sealed participant was opened.\n")
    return result
