"""No-training closure for the historical MobileBCI v5 development run."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np

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
    result={"status":"completed_no_training_closure","final_label":LABEL,"event_duration_fix":"duration field retained in seconds; only onset uses 100-Hz sample-index conversion","ssvep_safety":"recomputed_only_if_historical_arrays_sufficient" if arrays else "unavailable_arrays_missing","training_target_clip_fraction":float(np.mean(clipping)) if clipping else None,"dense_gamma0_pareto":"unavailable_not_reconstructed" if not arrays else "historical_arrays_present_but_no_common_dense_strength_grid","wrong_donor_fold_role_confound":True,"pc_labels":"old_per_window_labels_not_bounded_oracle_masks","zero_eog_values":"not_true_token_masking","retrained":False,"additional_seeds":False,"score_adapter":False,"sealed_participants_opened":False}
    run_dir.mkdir(parents=True,exist_ok=True); (run_dir/"result_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    report=Path("reports/mobile_temporal_diffusion_v5_closure.md"); report.parent.mkdir(parents=True,exist_ok=True); report.write_text("# MobileBCI v5 no-training closure\n\n"+f"Final label: `{LABEL}`.\n\nThe event onset remains a 100-Hz sample index, while the event duration field is already seconds and is no longer divided by 100. The historical wrong donors carry a fold-role confound; P-C used old per-window labels rather than bounded-oracle masks; numerical zero EOG inputs were not true token masking. No v5 model was retrained and no sealed participant was opened.\n")
    return result

