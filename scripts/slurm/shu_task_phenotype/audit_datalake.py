"""Audit the existing SHU-MI datalake copy without opening sealed sessions.

Session 04/05 keys are inventory metadata only.  Payload deserialization is
deliberately restricted to sessions 01--03 until the scientific route gate.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
from pathlib import Path

import lmdb
import numpy as np


KEY_RE = re.compile(
    r"^sub-(?P<subject>\d{3})_ses-(?P<session>\d{2})_task_motorimagery_eeg-(?P<trial>\d+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = lmdb.open(
        str(args.lmdb), subdir=True, readonly=True, lock=False,
        readahead=False, max_dbs=32,
    )
    counts: collections.Counter[tuple[int, int]] = collections.Counter()
    labels: dict[tuple[int, int], collections.Counter[int]] = collections.defaultdict(collections.Counter)
    examples: dict[int, dict[str, object]] = {}
    malformed: list[str] = []
    with env.begin() as txn:
        for key_b, value in txn.cursor():
            key = key_b.decode("utf-8", errors="strict")
            if key == "__keys__":
                continue
            match = KEY_RE.fullmatch(key)
            if match is None:
                malformed.append(key)
                continue
            subject = int(match["subject"])
            session = int(match["session"])
            counts[(subject, session)] += 1
            # Fail closed: no payload from session 04/05 is opened here.
            if session <= 3:
                record = pickle.loads(value)
                labels[(subject, session)][int(record["label"])] += 1
                if session not in examples:
                    sample = np.asarray(record["sample"])
                    examples[session] = {
                        "key": key,
                        "shape": list(sample.shape),
                        "dtype": str(sample.dtype),
                        "finite": bool(np.isfinite(sample).all()),
                        "median_abs": float(np.median(np.abs(sample))),
                        "p01": float(np.quantile(sample, 0.01)),
                        "p99": float(np.quantile(sample, 0.99)),
                    }

    matrix = [
        {
            "subject": subject,
            "session": session,
            "trials": counts[(subject, session)],
            "label_counts": dict(sorted(labels[(subject, session)].items())) if session <= 3 else "not_opened",
            "payload_opened": session <= 3,
        }
        for subject in range(1, 26)
        for session in range(1, 6)
    ]
    output = {
        "source": str(args.lmdb),
        "participants": len({subject for subject, _ in counts}),
        "sessions": len({session for _, session in counts}),
        "entries": sum(counts.values()),
        "all_25x5_present": all(counts[(s, d)] > 0 for s in range(1, 26) for d in range(1, 6)),
        "malformed_keys": malformed,
        "examples_development_only": examples,
        "session_04_05_payloads_opened": False,
        "coverage": matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
