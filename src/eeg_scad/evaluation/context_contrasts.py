from __future__ import annotations
from collections import defaultdict
from typing import Any,Mapping,Sequence
import numpy as np


def participant_first(rows:Sequence[Mapping[str,Any]],metric:str)->list[dict[str,Any]]:
    unit=defaultdict(list)
    for r in rows:unit[(r["participant"],r["session"],r["task"],r["seed"],r["method"])].append(float(r[metric]))
    task=defaultdict(list)
    for (p,_s,t,seed,m),v in unit.items():task[(p,t,seed,m)].append(float(np.mean(v)))
    participant=defaultdict(list)
    for (p,_t,seed,m),v in task.items():participant[(p,seed,m)].append(float(np.mean(v)))
    return [{"participant":p,"seed":seed,"method":m,metric:float(np.mean(v))} for (p,seed,m),v in sorted(participant.items())]

