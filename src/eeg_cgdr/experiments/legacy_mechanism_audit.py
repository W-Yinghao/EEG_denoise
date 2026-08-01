"""Analysis-only audit for pre-repair exploratory CGDR outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EVIDENCE_STATUS = "exploratory_pre_repair_not_gate_evidence"


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_float(row: dict[str, str], field: str) -> float:
    import math

    raw = row.get(field, "").strip()
    if not raw:
        raise ValueError(f"mechanism-audit metric is missing: {field}")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"mechanism-audit metric is non-finite: {field}")
    return value


def _posterior_mean_row(
    rows: list[dict[str, str]], *, method: str, calibration_seconds: int
) -> dict[str, str]:
    selected = [
        row
        for row in rows
        if row.get("method_id") == method
        and _is_true(row.get("aggregate_across_seeds"))
        and int(float(row.get("calibration_seconds", "-1"))) == calibration_seconds
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one posterior-mean row for {method} at "
            f"{calibration_seconds}s; found {len(selected)}"
        )
    return selected[0]


def run_legacy_mechanism_audit(
    config: dict[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Evaluate one frozen direction check without issuing a formal gate verdict."""

    if config.get("mode") != "exploratory_mechanism_audit":
        raise ValueError("mechanism-audit requires exploratory_mechanism_audit mode")
    if config.get("evidence_status") != EVIDENCE_STATUS:
        raise ValueError("mechanism-audit input must be explicitly marked exploratory")
    formal_gate = config["formal_gate"]
    if formal_gate.get("status") != "NOT_RUN_BLOCKED":
        raise ValueError("mechanism-audit cannot approve or execute a formal gate")

    source = config["source"]
    source_summary_path = Path(source["result_summary"])
    metrics_path = Path(source["metrics"])
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("evidence_status") != EVIDENCE_STATUS:
        raise ValueError("source result summary is not marked pre-repair exploratory")
    with metrics_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    check = config["check"]
    duration = int(check["calibration_seconds"])
    method_id = str(check["method"])
    reference_id = str(check["reference"])
    method = _posterior_mean_row(rows, method=method_id, calibration_seconds=duration)
    reference = _posterior_mean_row(
        rows, method=reference_id, calibration_seconds=duration
    )
    primary_field = str(check["primary_metric"])
    safety_field = str(check["safety_metric"])
    method_primary = _finite_float(method, primary_field)
    reference_primary = _finite_float(reference, primary_field)
    method_safety = _finite_float(method, safety_field)
    reference_safety = _finite_float(reference, safety_field)
    if check["primary_direction"] != "lower":
        raise ValueError("only the frozen lower primary direction is supported")
    if check["safety_direction"] != "lower_or_equal":
        raise ValueError("only the frozen lower-or-equal safety direction is supported")
    primary_improved = method_primary < reference_primary
    safety_preserved = method_safety <= reference_safety
    exploratory_status = (
        "direction_check_met_but_insufficient_units"
        if primary_improved and safety_preserved
        else "single_source_exploratory_check_failed"
    )

    result = {
        "status": "completed",
        "evidence_status": EVIDENCE_STATUS,
        "experiment_id": config["experiment_id"],
        "source": {
            "result_summary": str(source_summary_path),
            "metrics": str(metrics_path),
            "dataset": source["dataset"],
            "outer_unit": source["outer_unit"],
            "independent_units": int(source["independent_units"]),
        },
        "exploratory_check": {
            "status": exploratory_status,
            "calibration_seconds": duration,
            "method": method_id,
            "reference": reference_id,
            "primary_metric": primary_field,
            "method_primary": method_primary,
            "reference_primary": reference_primary,
            "primary_improved": primary_improved,
            "safety_metric": safety_field,
            "method_safety": method_safety,
            "reference_safety": reference_safety,
            "safety_preserved": safety_preserved,
        },
        "formal_gate": {
            "gate_id": formal_gate["gate_id"],
            "status": "NOT_RUN_BLOCKED",
            "reason": formal_gate["reason"],
        },
        "claim_boundary": (
            "A single-source pre-repair direction check is diagnostic only and "
            "cannot pass or fail formal G1."
        ),
    }
    output = config["outputs"]
    root = Path(output["root"])
    root.mkdir(parents=True, exist_ok=True)
    summary_path = Path(output["summary"])
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    note_path = Path(output["note"])
    note_path.write_text(
        "# Exploratory pre-repair mechanism audit\n\n"
        f"The single-source direction check is `{exploratory_status}`. "
        "This output is exploratory debugging evidence only. Formal G1 was "
        "not run and remains blocked. The command did not train, fit, sample, "
        "or modify the source metrics.\n",
        encoding="utf-8",
    )
    (run_dir / "legacy_mechanism_audit_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "evidence_status": EVIDENCE_STATUS,
                "summary": str(summary_path),
                "formal_gate_status": "NOT_RUN_BLOCKED",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result
