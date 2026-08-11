from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from eeg_scad.data.counterfactual_pairs import _load_signal, _query_operator, fold_eeg_scale
from eeg_scad.data.splits import load_folds, validate_folds
from eeg_scad.data.v24_coordinate_contract import CoordinateCell, comparison_metrics, robust_center_scale


ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_pa_el_scad_v24"))
RESULT = ROOT / "results/pa_el_scad_v24"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24")


def _cfg(name: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / f"configs/pa_el_scad_v24/{name}.yaml").read_text())


def _folds() -> list[dict[str, Any]]:
    return load_folds(ROOT / "configs/pa_el_scad_v24/folds.yaml")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_clean(path: Path) -> bool:
    return subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True).strip() == ""


def preflight(run: Path) -> dict[str, Any]:
    data = _cfg("data")
    folds = _folds()
    validate_folds(folds, data["participants"])
    v23 = Path(data["v23_worktree"])
    v19 = Path(data["v19_worktree"])
    a_track = Path(data["a_track_worktree"])
    checks = {
        "base_ancestry": subprocess.run(["git", "merge-base", "--is-ancestor", data["base_commit"], "HEAD"], cwd=ROOT).returncode == 0,
        "v23_exact": _head(v23) == data["v23_commit"],
        "v19_exact": _head(v19) == data["v19_commit"],
        "a_track_exact": _head(a_track) == data["a_track_commit"],
        "v23_tracked_clean": _tracked_clean(v23),
        "v19_tracked_clean": _tracked_clean(v19),
        "a_track_tracked_clean": _tracked_clean(a_track),
        "folds_exact_v23": (ROOT / "configs/pa_el_scad_v24/folds.yaml").read_text() == (v23 / "configs/of_scad_v23/folds.yaml").read_text(),
        "sealed_absent": not set(data["participants"]) & set(data["sealed_participants"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    inventory = []
    paths = [
        v23 / "src/eeg_scad/data/online_counterfactual.py",
        v23 / "src/eeg_scad/context/operator_normalization.py",
        v23 / "results/of_scad_v23/projection_ceilings.csv",
        v23 / "results/of_scad_v23/method_summary.csv",
        v23 / "results/of_scad_v23/participant_effects.csv",
        v23 / "results/of_scad_v23/training_exposure.csv",
        v23 / "reports/v23_final_development_diagnosis.md",
        v23 / "reports/slurm/v23_job_ids.txt",
    ]
    for path in paths:
        inventory.append({
            "absolute_path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else 0,
            "sha256": _digest(path) if path.is_file() else "",
            "source_commit": data["v23_commit"],
            "read_only": True,
        })
    _csv(RESULT / "v23_source_inventory.csv", inventory)
    registry = {
        "v23_source_commit": data["v23_commit"],
        "v19_source_commit": data["v19_commit"],
        "v23_historical_interpretation": {
            "engineering": "valid",
            "operator_factorized_context": "weak_or_heterogeneous_signal",
            "diffusion": "deterministic_better",
            "natural": "artifact_reduction_insufficient",
        },
        "inventory_digests": {row["absolute_path"]: row["sha256"] for row in inventory},
    }
    _json(RESULT / "v23_digest_registry.json", registry)
    report = [
        "# V24 transition audit from frozen V23",
        "",
        f"V23 is bound read-only at `{data['v23_commit']}`. Its historical reports, result rows, checkpoints and job lineage are not modified.",
        "",
        "V24 first adjudicates the operator/EOG/EEG coordinate contract. V23 scientific outputs remain historical until that audit is complete; no GPU stage is authorized by this preflight alone.",
        "",
        f"Preflight status: `{status}`. Sealed reads: `0`; manuscript changes: `0`.",
    ]
    (ROOT / "reports/v24_v23_transition_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/v24_v23_transition_audit.md").write_text("\n".join(report) + "\n")
    result = {"stage": "R0", "status": status, "checks": checks, "sealed_reads": 0, "gpu_jobs": 0}
    _json(RESULT / "preflight.json", result)
    _json(run / "result_summary.json", result)
    return result


def coordinate_audit(run: Path) -> dict[str, Any]:
    pre = json.loads((RESULT / "preflight.json").read_text())
    if pre["status"] != "PASS":
        raise RuntimeError("R0 preflight did not pass")
    data = _cfg("data")
    contract = _cfg("coordinate_contract")
    root = Path(data["v19_derived_root"])
    folds = _folds()
    fold_for = {participant: fold for fold in folds for participant in fold["test"]}
    rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    rng = np.random.Generator(np.random.PCG64DXSM(20260828))
    max_windows = int(contract["audit_windows_per_cell"])
    for participant, session, task in itertools.product(data["participants"], data["sessions"], data["tasks"]):
        signal_path = root / "prepared" / participant / f"{session}_{task}.npz"
        query_path = _query_operator(root, participant, session, task)
        if not signal_path.is_file() or not query_path.is_file():
            unit_rows.append({"participant": participant, "session": session, "task": task, "status": "missing_source_cell"})
            continue
        fold = fold_for[participant]
        eeg_scale = fold_eeg_scale(data, fold["train"])
        eeg, eog = _load_signal(root, participant, session, task)
        support = eog[:, : int(data["support_samples"])]
        support_center, support_scale = robust_center_scale(support)
        with np.load(query_path, allow_pickle=False) as archive:
            raw_operator = np.asarray(archive["C_query"], dtype=np.float64)
        cell = CoordinateCell(raw_operator, eeg_scale, support_center, support_scale)
        starts_max = eog.shape[1] - int(data["window_samples"])
        if starts_max <= int(data["qnatural_start"]):
            unit_rows.append({"participant": participant, "session": session, "task": task, "status": "insufficient_qnatural"})
            continue
        starts = rng.integers(int(data["qnatural_start"]), starts_max + 1, size=max_windows)
        raw_all, canonical_all, committed_all = [], [], []
        for window, start in enumerate(starts):
            waveform = np.asarray(eog[:, start : start + int(data["window_samples"])], dtype=np.float64)
            # Literal V23 replay centers the sampled waveform, not support median.
            centered = waveform - np.mean(waveform, axis=1, keepdims=True)
            raw = (raw_operator @ centered) / eeg_scale[:, None]
            canonical = cell.canonical @ (centered / support_scale[:, None])
            committed = (cell.canonical @ centered) / eeg_scale[:, None]
            raw_all.append(raw); canonical_all.append(canonical); committed_all.append(committed)
            rc = comparison_metrics(raw, canonical)
            rv = comparison_metrics(raw, committed)
            rows.append({
                "fold": fold["fold"], "participant": participant, "session": session, "task": task,
                "window": window, "start": int(start),
                **{f"raw_vs_canonical_{key}": value for key, value in rc.items()},
                **{f"raw_vs_v23_{key}": value for key, value in rv.items()},
                "eog_median": float(np.median(waveform)), "eog_mad": float(np.median(np.abs(waveform - np.median(waveform)))),
                "eog_rms": float(np.sqrt(np.mean(waveform * waveform))),
                "support_scale_min": float(np.min(support_scale)), "support_scale_max": float(np.max(support_scale)),
                "eeg_scale_min": float(np.min(eeg_scale)), "eeg_scale_max": float(np.max(eeg_scale)),
            })
        raw_stack = np.stack(raw_all); canonical_stack = np.stack(canonical_all); committed_stack = np.stack(committed_all)
        rc = comparison_metrics(raw_stack.reshape(-1, raw_stack.shape[-1]), canonical_stack.reshape(-1, canonical_stack.shape[-1]))
        rv = comparison_metrics(raw_stack.reshape(-1, raw_stack.shape[-1]), committed_stack.reshape(-1, committed_stack.shape[-1]))
        unit_rows.append({
            "fold": fold["fold"], "participant": participant, "session": session, "task": task,
            "status": "audited", "windows": max_windows, "prepared_eog_unit": "microvolt",
            "prepared_producer": "v19_prepare_stage_preprocessed_source_eog_no_amplitude_standardization",
            **{f"raw_vs_canonical_{key}": value for key, value in rc.items()},
            **{f"raw_vs_v23_{key}": value for key, value in rv.items()},
        })
    _csv(RESULT / "coordinate_cell_comparison.csv", rows)
    _csv(RESULT / "coordinate_unit_summary.csv", unit_rows)
    audited = [row for row in unit_rows if row["status"] == "audited"]
    equivalent_correct = bool(audited) and max(float(row["raw_vs_canonical_relative_frobenius_difference"]) for row in audited) <= float(contract["relative_tolerance"])
    v23_equivalent = bool(audited) and max(float(row["raw_vs_v23_relative_frobenius_difference"]) for row in audited) <= float(contract["relative_tolerance"])
    complete = len(audited) == len(data["participants"]) * len(data["sessions"]) * len(data["tasks"])
    if not complete or not equivalent_correct:
        verdict = "V23_COORDINATE_INDETERMINATE"
    elif v23_equivalent:
        verdict = "V23_COORDINATE_EQUIVALENT"
    else:
        verdict = "V23_COORDINATE_MISMATCH_CONFIRMED"
    decision = {
        "coordinate_verdict": verdict,
        "prepared_eog_unit": "microvolt",
        "prepared_eog_centered": False,
        "prepared_eog_scaled": False,
        "prepared_eog_whitened": False,
        "producer_evidence": "V19 prepare_stage saves filtered/resampled source EOG directly after winsorization; config source unit is microvolt",
        "audited_cells": len(audited),
        "audited_windows": len(rows),
        "raw_canonical_max_relative_difference": max((float(row["raw_vs_canonical_relative_frobenius_difference"]) for row in audited), default=None),
        "raw_v23_min_relative_difference": min((float(row["raw_vs_v23_relative_frobenius_difference"]) for row in audited), default=None),
        "raw_v23_median_relative_difference": float(np.median([float(row["raw_vs_v23_relative_frobenius_difference"]) for row in audited])) if audited else None,
        "raw_v23_max_relative_difference": max((float(row["raw_vs_v23_relative_frobenius_difference"]) for row in audited), default=None),
        "v24_training_authorized": verdict != "V23_COORDINATE_INDETERMINATE",
        "v23_artifacts_modified": False,
        "sealed_reads": 0,
    }
    _json(RESULT / "coordinate_audit.json", decision)
    lines = [
        "# V24 operator/EOG/EEG coordinate audit", "",
        f"Verdict: `{verdict}`.", "",
        "The V19 producer stores source EOG in microvolts after common preprocessing (winsorization, 0.5–15 Hz filtering and resampling), without amplitude standardization. V23 then formed `D_y^-1 C_raw D_e`, multiplied it by centered physical EOG without `D_e^-1`, and divided by `D_y` a second time.", "",
        f"The mathematically equivalent raw and canonical routes agreed over {len(rows)} real windows from {len(audited)} participant/session/task cells; maximum relative difference `{decision['raw_canonical_max_relative_difference']:.3e}`.",
        f"The V23 committed route differed from the correct raw route with median relative difference `{decision['raw_v23_median_relative_difference']:.6f}` (range `{decision['raw_v23_min_relative_difference']:.6f}`–`{decision['raw_v23_max_relative_difference']:.6f}`).", "",
        "No V23 file was changed. V24 will use corrected assets and will not use V23 coefficient statistics or checkpoints as scientific initialization." if verdict == "V23_COORDINATE_MISMATCH_CONFIRMED" else "V23 assets may be reused only under the validated coordinate contract.", "",
        "Sealed reads: `0`. GPU jobs before this verdict: `0`.",
    ]
    (ROOT / "reports/v24_coordinate_audit.md").write_text("\n".join(lines) + "\n")
    if verdict == "V23_COORDINATE_MISMATCH_CONFIRMED":
        note = [
            "# V23 coordinate supersession note", "",
            "V23 numerical outputs remain reproducible and immutable, but the online generator used a non-equivalent operator/EOG/EEG coordinate composition. Its absolute paired effects, projection ceilings and natural interpretation are therefore historical development results under an invalid coordinate construction and are excluded from later paper evidence.", "",
            "The V23 engineering harness, splits, role assignments, seeds, source recordings and support/query blocks remain reusable. V24 rematerializes artifacts, observations, EOG latents, projection ceilings and targets under the corrected contract. No V23 artifact is deleted or overwritten.",
        ]
        (ROOT / "reports/v23_coordinate_supersession_note.md").write_text("\n".join(note) + "\n")
    elif verdict == "V23_COORDINATE_EQUIVALENT":
        (ROOT / "reports/v23_coordinate_validation_note.md").write_text("# V23 coordinate validation note\n\nThe committed route was numerically equivalent to the physical/canonical identity over every audited cell.\n")
    _json(run / "result_summary.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "r0-preflight":
        preflight(args.run_dir)
    elif args.stage == "r1-coordinate-audit":
        coordinate_audit(args.run_dir)
    else:
        raise ValueError(f"unknown V24 stage: {args.stage}")


if __name__ == "__main__":
    main()

