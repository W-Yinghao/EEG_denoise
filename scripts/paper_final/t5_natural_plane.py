#!/usr/bin/env python3
"""PAPER-FINAL T5 — natural-plane completion (5-condition table).

Deviation note (one sentence, per ground rule 3): the mismatched and shuffled
natural rows ALREADY EXIST in storage — V44-S1 ran the full 7-arm natural panel
(all 15 fold-seed units carry WRONG / WRONG_gated / SHUFFLED natural rows that were
simply never aggregated into the natural table) — so T5 is a CPU aggregation of
stored evaluator outputs, not a new GPU run.

Conditions reported: unguided (NO_A0) / population (POP) / matched (MATCH_gated) /
mismatched (WRONG_gated primary: another participant's matrix through the same gate,
matching the paper's accepted-mismatched paired row; ungated WRONG as reference) /
shuffled (SHUFFLED).  Endpoints: EOG attenuation (dB), low-EOG retention,
EOG-EEG coherence reduction; participant-first with 5000-draw bootstrap.
"""
from __future__ import annotations

import json

import numpy as np

from pf_common import ARRAYS, OUT, participant_means, stat, stored_stage1_natural_rows

CONDITIONS = ("NO_A0", "POP", "MATCH_gated", "WRONG_gated", "WRONG", "SHUFFLED")
METRICS = ("attenuation_db", "low_eog_observation_retention", "coherence_reduction")


def main() -> None:
    natural = stored_stage1_natural_rows()
    table, contrasts = {}, {}
    match = {metric: participant_means(natural, "MATCH_gated", metric)
             for metric in METRICS}
    for condition in CONDITIONS:
        table[condition] = {}
        for metric in METRICS:
            per = participant_means(natural, condition, metric)
            table[condition][metric] = stat(list(per.values()))
            if condition != "MATCH_gated":
                common = [p for p in per if p in match[metric]]
                contrasts.setdefault(f"MATCH_gated_minus_{condition}", {})[metric] = \
                    stat([match[metric][p] - per[p] for p in common])
    decision = {
        "note": "aggregated from stored V44-S1 stage1_result.json natural rows "
                "(5 folds x 3 seeds, 4 windows/cell); rows existed in storage — "
                "no new GPU sampling was needed for this table",
        "five_condition_table": {c: {m: table[c][m]["mean"] for m in METRICS}
                                 for c in CONDITIONS},
        "full_stats": table, "contrasts": contrasts,
    }
    (OUT / "t5_natural_plane.json").write_text(json.dumps(decision, indent=2,
                                                          sort_keys=True) + "\n")
    np.savez_compressed(
        ARRAYS / "t5_natural_plane.npz",
        conditions=np.asarray(CONDITIONS),
        attenuation_db=np.asarray([table[c]["attenuation_db"]["mean"]
                                   for c in CONDITIONS]),
        retention=np.asarray([table[c]["low_eog_observation_retention"]["mean"]
                              for c in CONDITIONS]),
        coherence_reduction=np.asarray([table[c]["coherence_reduction"]["mean"]
                                        for c in CONDITIONS]),
        attenuation_ci=np.asarray([[table[c]["attenuation_db"]["bootstrap_low"],
                                    table[c]["attenuation_db"]["bootstrap_high"]]
                                   for c in CONDITIONS]),
        retention_ci=np.asarray(
            [[table[c]["low_eog_observation_retention"]["bootstrap_low"],
              table[c]["low_eog_observation_retention"]["bootstrap_high"]]
             for c in CONDITIONS]))
    print(json.dumps(decision["five_condition_table"], indent=1))


if __name__ == "__main__":
    main()
