"""Bounded MobileBCI index and development split for v3.

Session 02 is standing (0 m/s); sessions 03--05 are slow walking, fast
walking, and slight running.  This stage reads only BIDS headers/channels and
filenames, not signal outcomes.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_lit_explore_v3"))
ROOT = Path("/projects/EEG-foundation-model/mobile_bci")
OUTPUT = CODE_ROOT / "results/cgdr/literature_guided_exploration_v3/baseline_audit"


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _header_rate(path: Path) -> float:
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("SamplingInterval="):
            microseconds = float(line.split("=", 1)[1])
            return 1_000_000.0 / microseconds
    raise ValueError(f"BrainVision header lacks SamplingInterval: {path}")


def _channel_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            kind = str(row.get("type", "unknown")).upper()
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def _source_channel_metadata(
    *, participant: str, session: str, task: str
) -> tuple[Path | None, dict[str, int]]:
    """Return the official source-record channel table when it is present.

    The released BIDS derivative stores EEG and motion channels together, but
    omits the four EOG channels retained in ``sourcedata``.  Both layouts are
    useful and must not be conflated in the availability audit.
    """
    source = (
        ROOT / "sourcedata" / participant / session / "eeg"
        / f"{participant}_{session}_task-{task}_channels.tsv"
    )
    if not source.is_file():
        return None, {}
    return source, _channel_counts(source)


def run(run_dir: Path) -> Mapping[str, Any]:
    if not ROOT.is_dir():
        raise FileNotFoundError("official MobileBCI download is not published yet")
    pattern = re.compile(r"sub-(\d+)_ses-(\d+)_task-(ERP|SSVEP)_eeg\.vhdr$")
    rows: list[dict[str, Any]] = []
    for participant_dir in sorted(ROOT.glob("sub-[0-9][0-9]")):
        for header in sorted(participant_dir.glob("ses-[0-9][0-9]/eeg/*_eeg.vhdr")):
            match = pattern.match(header.name)
            if match is None:
                continue
            participant, session, task = match.groups()
            channel_file = header.with_name(header.name.replace("_eeg.vhdr", "_channels.tsv"))
            if not channel_file.is_file():
                raise FileNotFoundError(f"MobileBCI channel metadata missing: {channel_file}")
            counts = _channel_counts(channel_file)
            source_channels, source_counts = _source_channel_metadata(
                participant=f"sub-{participant}", session=f"ses-{session}", task=task
            )
            participant_number = int(participant)
            session_number = int(session)
            rows.append({
                "participant": f"sub-{participant}",
                "participant_role": "outer_training_development" if participant_number <= 16 else "heldout_development",
                "session": f"ses-{session}", "task": task,
                "motion_condition": {1: "erp_training_standing", 2: "standing_0.0mps", 3: "slow_walking_0.8mps", 4: "fast_walking_1.6mps", 5: "slight_running_2.0mps"}.get(session_number, "unknown"),
                "support_or_query": "support" if session_number == 2 else "query" if session_number in {3, 4, 5} else "training_auxiliary",
                "sampling_rate_hz": _header_rate(header),
                "processed_eeg_channels": counts.get("EEG", 0),
                "processed_eog_channels": counts.get("EOG", 0),
                "processed_imu_channels": counts.get("IMU", 0),
                "source_eeg_channels": source_counts.get("EEG", 0),
                "source_eog_channels": source_counts.get("EOG", 0),
                "header": str(header), "channels_tsv": str(channel_file),
                "source_channels_tsv": str(source_channels) if source_channels else "",
                "outcomes_opened": False,
            })
    participants = sorted({row["participant"] for row in rows})
    if len(participants) != 24:
        raise ValueError(f"MobileBCI index expected 24 participants, found {len(participants)}")
    support = {(row["participant"], row["task"]) for row in rows if row["support_or_query"] == "support"}
    query = {(row["participant"], row["task"]) for row in rows if row["support_or_query"] == "query"}
    eligible = sorted(support & query)
    _write_csv(OUTPUT / "mobile_bci_development_split.csv", rows)
    summary = {
        "status": "completed_mobile_bci_header_and_split_audit",
        "participants": len(participants), "record_headers": len(rows),
        "eligible_participant_task_pairs": len(eligible),
        "support_session": "ses-02_standing", "query_sessions": ["ses-03", "ses-04", "ses-05"],
        "participant_split": {"outer_training_development": "sub-01--sub-16", "heldout_development": "sub-17--sub-24"},
        "channel_layout": {
            "processed_bids": "46 EEG plus 27 IMU channels; EOG omitted",
            "official_sourcedata": "46 EEG plus 4 EOG channels; IMU stored separately",
        },
        "evaluator_feasibility": "EOG and IMU are available from official sourcedata; signal outcomes remain unopened",
        "test_outcomes_opened": False, "scientific_role": "development_not_confirmation",
    }
    (OUTPUT / "mobile_bci_index_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
