"""Task-preservation availability audit for V28."""
from __future__ import annotations

from typing import Mapping


REQUIRED={"ERP":("event_markers","trial_boundaries","condition_labels"),"SSVEP":("stimulation_frequency","stimulation_phase","trial_boundaries"),"ERD_ERS":("motor_task_markers","baseline_interval","task_interval")}


def inventory(metadata:Mapping[str,object])->list[dict[str,object]]:
    rows=[]
    for outcome,required in REQUIRED.items():
        missing=[field for field in required if field not in metadata or metadata[field] in (None,"",[])];rows.append({"outcome":outcome,"status":"unavailable" if missing else "supported","missing_fields":";".join(missing),"proxy_substitution_forbidden":True})
    return rows


__all__=["inventory","REQUIRED"]
