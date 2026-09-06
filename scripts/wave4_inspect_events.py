#!/usr/bin/env python3
"""WAVE4 E1 support: establish the true event structure in both streams.

`Trig` in the EEG export and several Tobii event columns are HELD values, not edge
lists; the alignment must use transitions. This diagnostic reports value
distributions and edge counts so the matcher is built on the real structure.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

EEG = Path("/projects/EEG-foundation-model/eye_bci/syn64005218-neuroscan/S01/Sess01/"
           "Neuroscan/ME011.csv")
TOBII = Path("/projects/EEG-foundation-model/eye_bci/syn64005218-tobii/S01/Sess01/Tobii/"
             "ME011.csv")
OUT = Path(__file__).resolve().parents[1] / "results/wave4_optical/alignment"
TOBII_EVENT_COLUMNS = ("StudioEvent", "StudioEventData", "ExternalEvent",
                       "ExternalEventValue", "EventMarkerValue", "KeyPressEvent",
                       "MouseEvent")


def edges(rows, key, time_key):
    values = Counter()
    transitions = []
    previous = None
    for row in rows:
        value = row.get(key)
        values[value] += 1
        if value != previous:
            if value not in (None, "", "NA"):
                transitions.append((row[time_key], value))
            previous = value
    return values, transitions


def main() -> None:
    payload = {}
    with EEG.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for column in ("Trig", "Cues"):
        values, transitions = edges(rows, column, "Time")
        payload[f"eeg_{column}"] = {
            "nonempty_rows": int(sum(v for k, v in values.items() if k not in (None, "", "NA"))),
            "unique_values": len(values),
            "top_values": [[str(k), int(v)] for k, v in values.most_common(8)],
            "edge_count": len(transitions),
            "first_edges": [[float(t), str(v)] for t, v in transitions[:10]],
        }
    with TOBII.open(newline="") as handle:
        trows = list(csv.DictReader(handle))
    for column in TOBII_EVENT_COLUMNS:
        values, transitions = edges(trows, column, "RecordingTimestamp")
        payload[f"tobii_{column}"] = {
            "nonempty_rows": int(sum(v for k, v in values.items() if k not in (None, ""))),
            "unique_values": len(values),
            "top_values": [[str(k), int(v)] for k, v in values.most_common(6)],
            "edge_count": len(transitions),
            "first_edges": [[str(t), str(v)] for t, v in transitions[:10]],
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e1_event_structure.json").write_text(json.dumps(payload, indent=2,
                                                            sort_keys=True) + "\n")
    for key, value in payload.items():
        print(f"{key:28s} nonempty={value['nonempty_rows']:<8d} uniq={value['unique_values']:<5d} "
              f"edges={value['edge_count']:<6d} ex={value['first_edges'][:3]}")


if __name__ == "__main__":
    main()
