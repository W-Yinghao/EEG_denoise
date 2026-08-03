"""Mechanical J6 finalization for the subject-artifact development round.

The finalizer deliberately does not open signal data, arrays, checkpoints, or
metric tables.  It validates a small set of terminal JSON decisions and copies
the J5 verdict into the final summary and human-readable reports.  It never
submits a confirmation job.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_ID = "subject_calibrated_artifact_latent_diffusion_development_v1"
EXECUTION_REVISION = "j2_v1_identity_routing_r2"
CPU_ENVIRONMENT = "/home/infres/yinwang/anaconda3/envs/eeg2025"
CONFIG_RELATIVE_PATH = Path(
    "configs/cgdr/subject_calibrated_artifact_development_j2r2.yaml"
)


@dataclass(frozen=True)
class FinalizerPaths:
    code_root: Path
    config: Path
    validity_gate: Path
    j0: Path
    j1: Path
    j5: Path
    old_frozen_decision: Path
    job_ledger: Path
    output_summary: Path
    development_report: Path
    confirmation_plan: Path
    terminal_manifest: Path


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _paths(config: Mapping[str, Any]) -> FinalizerPaths:
    outputs = _mapping(config, "outputs")
    output_root = Path(str(outputs.get("root", "")))
    if len(output_root.parents) < 3:
        raise ValueError("subject-artifact output root is not inside a code root")
    code_root = output_root.parents[2]
    expected_root = code_root / "results/cgdr/subject_calibrated_artifact_diffusion"
    if output_root != expected_root:
        raise ValueError("subject-artifact output root changed")
    revision_root = output_root / "revisions" / EXECUTION_REVISION
    expected_outputs = {
        "validity_root": revision_root / "validity",
        "development_root": revision_root / "development",
        "checkpoint_root": revision_root / "checkpoints",
        "metrics": revision_root / "metrics.csv",
        "summary": revision_root / "result_summary.json",
        "figures": revision_root / "figures",
    }
    for key, expected in expected_outputs.items():
        if Path(str(outputs.get(key, ""))) != expected:
            raise ValueError(f"subject-artifact revision output changed: {key}")
    validity_root = expected_outputs["validity_root"]
    development_root = expected_outputs["development_root"]
    return FinalizerPaths(
        code_root=code_root,
        config=code_root / CONFIG_RELATIVE_PATH,
        validity_gate=validity_root / "result_summary.json",
        j0=validity_root / "j0_audit.json",
        j1=validity_root / "j1_cpu_validation.json",
        j5=development_root / "aggregate/result_summary.json",
        old_frozen_decision=(
            code_root
            / "results/cgdr/diffusion_incremental_decision_v2/result_summary.json"
        ),
        job_ledger=(
            code_root
            / "reports/slurm/subject_calibrated_artifact_diffusion_job_ids.txt"
        ),
        output_summary=expected_outputs["summary"],
        development_report=(
            code_root / "reports/subject_calibrated_artifact_diffusion_development.md"
        ),
        confirmation_plan=(
            code_root / "reports/subject_calibrated_artifact_confirmation_plan.md"
        ),
        terminal_manifest=revision_root / "terminal_manifest.json",
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("harness_level") != 1
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("scientific_role")
        != "development_exploratory_not_confirmation"
        or _mapping(config, "validity").get("execution_revision")
        != EXECUTION_REVISION
    ):
        raise ValueError("J6 received a different development protocol")
    boundaries = _mapping(config, "boundaries")
    forbidden = (
        "query_eog_or_eye_tracking_input",
        "query_artifact_label_input",
        "query_outcome_input",
        "best_of_k_selection",
        "confirmation_outcomes_this_round",
    )
    if any(boundaries.get(key) != "forbidden" for key in forbidden):
        raise ValueError("J6 information boundary was weakened")


def _validate_j0_j1(
    j0: Mapping[str, Any], j1: Mapping[str, Any]
) -> None:
    if (
        j0.get("protocol_id") != PROTOCOL_ID
        or j0.get("status") != "passed_j0_manifest_and_target_availability"
        or j0.get("confirmation_signal_or_outcome_opened") is not False
        or j0.get("file_hashes_computed") is not False
        or j0.get("data_download_performed") is not False
    ):
        raise ValueError("J0 terminal audit is absent, stale, or unsafe")
    if (
        j1.get("protocol_id") != PROTOCOL_ID
        or j1.get("status") != "passed_j1_real_record_context_validation"
        or j1.get("confirmation_signal_or_outcome_opened") is not False
    ):
        raise ValueError("J1 terminal validation is absent, stale, or unsafe")


def _validate_old_decision(value: Mapping[str, Any]) -> None:
    if (
        value.get("current_M2_status") != "current_M2_no_incremental_value"
        or value.get("diffusion_family_wide_status") != "not_tested"
        or value.get("formal_G1_status") != "NOT_RUN_BLOCKED"
        or value.get("formal_G3_status") != "NOT_RUN_BLOCKED"
    ):
        raise ValueError("the retained frozen pre-round decision changed")


def _validate_validity_gate(
    config: Mapping[str, Any], gate: Mapping[str, Any]
) -> Mapping[str, Any]:
    passed = gate.get("passed") is True
    expected_gate_status = (
        "passed_V0_to_V3" if passed else "completed_model_validity_failed"
    )
    expected_attempt_status = "passed" if passed else "failed"
    expected_model_validity = "passed" if passed else "failed"
    selected = gate.get("selected_result")
    implementation = str(gate.get("selected_implementation", ""))
    if (
        gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("execution_revision") != EXECUTION_REVISION
        or gate.get("status") != expected_gate_status
        or gate.get("model_validity") != expected_model_validity
        or not isinstance(selected, Mapping)
    ):
        raise ValueError("revision validity wrapper is absent or inconsistent")
    validity = selected.get("validity")
    expected_detail = (
        Path(str(_mapping(config, "outputs")["validity_root"]))
        / EXECUTION_REVISION
        / implementation
        / "result_summary.json"
    )
    if (
        selected.get("execution_revision") != EXECUTION_REVISION
        or selected.get("implementation") != implementation
        or selected.get("status") != expected_attempt_status
        or (selected.get("passed") is True) is not passed
        or selected.get("model_validity") != expected_model_validity
        or not isinstance(validity, Mapping)
        or set(validity) != {"V0", "V1", "V2", "V3"}
        or any(
            not isinstance(value, Mapping)
            or value.get("status") not in {"passed", "failed", "blocked"}
            for value in validity.values()
        )
        or Path(str(selected.get("attempt_result_path", ""))) != expected_detail
    ):
        raise ValueError("validity wrapper and selected attempt disagree")
    return selected


def _validate_j5(
    gate: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> None:
    validity_passed = gate.get("passed") is True
    expected_model = "passed" if validity_passed else "failed"
    status = str(aggregate.get("status", ""))
    status_valid = (
        status.startswith("completed_development_aggregate_")
        if validity_passed
        else status == "completed_fail_closed_model_validity_failed"
    )
    confirmation = aggregate.get("confirmation_eligibility")
    decision = aggregate.get("protocol_decision")
    if (
        aggregate.get("protocol_id") != PROTOCOL_ID
        or aggregate.get("execution_revision") != EXECUTION_REVISION
        or not status_valid
        or aggregate.get("model_validity") != expected_model
        or not isinstance(confirmation, bool)
        or not isinstance(decision, str)
        or not decision
        or aggregate.get("query_confirmation_outcomes_opened") is not False
        or (
            not validity_passed
            and (confirmation is not False or decision != "inconclusive")
        )
    ):
        raise ValueError("J5 aggregate is absent, stale, or scientifically unsafe")


def _read_job_ledger(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    entries: list[dict[str, str]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split("|")]
        if len(fields) < 3 or not fields[1].isdigit():
            raise ValueError(f"invalid subject-artifact job ledger line {line_number}")
        entries.append(
            {
                "stage": fields[0],
                "job_id": fields[1],
                "status": fields[2],
                "details": " | ".join(fields[3:]),
            }
        )
    if not entries:
        raise ValueError("subject-artifact job ledger is empty")
    return entries


def _validate_job_ids(
    ledger: Sequence[Mapping[str, str]],
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> None:
    listed = {str(value["job_id"]) for value in ledger}
    for name, source in sources:
        job_id = str(source.get("slurm_job_id", ""))
        if not job_id.isdigit() or job_id not in listed:
            raise ValueError(f"{name} Slurm job is absent from the round ledger")


def _nested_status(value: Mapping[str, Any], key: str) -> str:
    nested = value.get(key)
    return (
        str(nested.get("status", "missing"))
        if isinstance(nested, Mapping)
        else "missing"
    )


def _verdict(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "model_validity",
        "scientific_comparison_eligibility",
        "protocol_decision",
        "topic_status",
        "confirmation_eligibility",
        "family_wide_status",
        "real_EEG_evidence_scope",
    )
    result = {field: aggregate.get(field) for field in fields}
    if result["real_EEG_evidence_scope"] is None:
        result["real_EEG_evidence_scope"] = "not_reported_by_fail_closed_J5"
    result.update(
        {
            "G_calibration_status": _nested_status(aggregate, "G_calibration"),
            "G_diffusion_status": _nested_status(aggregate, "G_diffusion"),
            "uncertainty_status": _nested_status(aggregate, "uncertainty"),
            "calibration_duration_status": _nested_status(
                aggregate, "calibration_duration"
            ),
        }
    )
    return result


def _confirmation_blockers(
    gate: Mapping[str, Any], aggregate: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if gate.get("passed") is not True:
        blockers.append("V0_to_V3_not_passed")
    calibration = aggregate.get("G_calibration")
    paired = (
        calibration.get("paired_mechanism")
        if isinstance(calibration, Mapping)
        else None
    )
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("overall_passed") is not True
        or not isinstance(paired, Mapping)
        or paired.get("status") != "passed"
    ):
        blockers.append("paired_mechanism_evidence_missing_or_not_passed")
    uncertainty = aggregate.get("uncertainty")
    if (
        not isinstance(uncertainty, Mapping)
        or uncertainty.get("overall_passed") is not True
    ):
        blockers.append("matched_uncertainty_comparison_missing_or_not_passed")
    if not blockers and aggregate.get("confirmation_eligibility") is not True:
        blockers.append("J5_confirmation_eligibility_false")
    return blockers


def _job_table(entries: Sequence[Mapping[str, str]]) -> str:
    rows = ["| Stage | Job ID | Recorded status |", "|---|---:|---|"]
    rows.extend(
        f"| {entry['stage']} | {entry['job_id']} | {entry['status']} |"
        for entry in entries
    )
    return "\n".join(rows)


def _development_report(
    *,
    verdict: Mapping[str, Any],
    gate: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    ledger: Sequence[Mapping[str, str]],
    paths: FinalizerPaths,
    blockers: Sequence[str],
) -> str:
    selected = gate["selected_result"]
    validity = selected["validity"]
    validity_rows = "\n".join(
        f"- {level}: {value.get('status', 'missing')}"
        for level, value in validity.items()
    )
    return f"""# Subject-calibrated artifact diffusion development

This is a mechanical rendering of the frozen J5 verdict. J6 did not reopen
EEG, EOG, labels, arrays, checkpoints, or metric tables and did not recompute
any scientific statistic.

## Engineering and validity

- J0: passed manifest/availability audit.
- J1: passed real-record context validation.
- J2 model validity: {verdict['model_validity']}.
- J5 computational status: {aggregate.get('computational_completion')}.

{validity_rows}

## Scientific go/no-go

- Matching calibration: {verdict['G_calibration_status']}.
- Diffusion versus matched deterministic: {verdict['G_diffusion_status']}.
- Uncertainty contribution: {verdict['uncertainty_status']}.
- Calibration-duration evidence: {verdict['calibration_duration_status']}.
- Topic status: {verdict['topic_status']}.
- Protocol decision: {verdict['protocol_decision']}.
- Confirmation eligibility: {str(verdict['confirmation_eligibility']).lower()}.
- Confirmation blockers: {', '.join(blockers) if blockers else 'none'}.

The real-EEG evidence scope remains:
`{verdict['real_EEG_evidence_scope']}`.

The retained pre-round M2 decision is unchanged at
`{paths.old_frozen_decision}`. No confirmation outcome was opened and no
confirmation job was generated.

## Slurm ledger

{_job_table(ledger)}

## Result paths

- J5 aggregate: `{paths.j5}`
- Final summary: `{paths.output_summary}`
- Terminal manifest: `{paths.terminal_manifest}`
"""


def _confirmation_report(
    *,
    eligible: bool,
    blockers: Sequence[str],
    paths: FinalizerPaths,
) -> str:
    state = "eligible_for_a_separate_frozen_round" if eligible else "blocked"
    details = (
        "none" if not blockers else "\n".join(f"- {value}" for value in blockers)
    )
    return f"""# Subject-calibrated artifact confirmation plan

Status: `{state}`

J6 generated no Slurm confirmation job and opened no candidate confirmation
signal, EOG, label, or outcome. The development verdict and configuration are
frozen at:

- configuration: `{paths.config}`
- development summary: `{paths.output_summary}`

Current blockers:

{details}

Only a later, separately authorized round may create an independent
confirmation split and submission after every blocker is resolved without
using confirmation outcomes for method or threshold selection.
"""


def run_subject_artifact_finalize(
    config: Mapping[str, Any],
    run_dir: str | Path,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize J6 from terminal lightweight inputs without recomputation."""

    started = time.monotonic()
    _validate_config(config)
    paths = _paths(config)
    j0 = _read_json(paths.j0)
    j1 = _read_json(paths.j1)
    gate = _read_json(paths.validity_gate)
    aggregate = _read_json(paths.j5)
    old_decision = _read_json(paths.old_frozen_decision)
    ledger = _read_job_ledger(paths.job_ledger)
    _validate_j0_j1(j0, j1)
    _validate_old_decision(old_decision)
    _validate_validity_gate(config, gate)
    _validate_j5(gate, aggregate)

    git_commit = str(implementation.get("git_commit", "")).lower()
    job_id = str(implementation.get("slurm_job_id", ""))
    if (
        len(git_commit) != 40
        or any(value not in "0123456789abcdef" for value in git_commit)
        or not job_id.isdigit()
        or implementation.get("slurm_profile") != "cpu"
    ):
        raise ValueError("J6 implementation or CPU Slurm allocation is invalid")
    _validate_job_ids(
        ledger,
        (
            ("J0", j0),
            ("J1", j1),
            ("J2", gate),
            ("J5", aggregate),
        ),
    )

    copied_verdict = _verdict(aggregate)
    blockers = _confirmation_blockers(gate, aggregate)
    if aggregate.get("confirmation_eligibility") is True and blockers:
        raise ValueError("J5 confirmation verdict contradicts its hard blockers")
    confirmation_eligible = bool(
        aggregate.get("confirmation_eligibility") is True and not blockers
    )
    summary: dict[str, Any] = {
        "status": "completed_j6_mechanical_finalization",
        "stage": "J6_finalize",
        "protocol_id": PROTOCOL_ID,
        "execution_revision": EXECUTION_REVISION,
        "scientific_role": "development_exploratory_not_confirmation",
        "rendering_rule": "mechanical_copy_of_J5_no_metric_recomputation",
        "source_paths": {
            "J0": str(paths.j0),
            "J1": str(paths.j1),
            "J2": str(paths.validity_gate),
            "J5": str(paths.j5),
            "old_frozen_decision": str(paths.old_frozen_decision),
            "job_ledger": str(paths.job_ledger),
        },
        "verdict": copied_verdict,
        **copied_verdict,
        "confirmation_blockers": blockers,
        "confirmation_eligibility": confirmation_eligible,
        "confirmation_job_generated": False,
        "confirmation_job_id": None,
        "query_confirmation_outcomes_opened": False,
        "development_report": str(paths.development_report),
        "confirmation_plan": str(paths.confirmation_plan),
        "terminal_manifest": str(paths.terminal_manifest),
        "git_commit": git_commit,
        "slurm_job_id": job_id,
    }
    _atomic_text(
        paths.development_report,
        _development_report(
            verdict=copied_verdict,
            gate=gate,
            aggregate=aggregate,
            ledger=ledger,
            paths=paths,
            blockers=blockers,
        ),
    )
    _atomic_text(
        paths.confirmation_plan,
        _confirmation_report(
            eligible=confirmation_eligible,
            blockers=blockers,
            paths=paths,
        ),
    )
    _atomic_json(paths.output_summary, summary)

    run_destination = Path(run_dir)
    command = (
        f"{CPU_ENVIRONMENT}/bin/python -m eeg_cgdr.cli.main subject-artifact "
        f"--config {paths.config} --run-dir {run_destination} --stage finalize"
    )
    manifest = {
        "job_id": job_id,
        "implementation_git_sha": git_commit,
        "final_report_git_sha": "pending_post_J6_report_commit",
        "config": str(paths.config),
        "command": command,
        "conda_environment": CPU_ENVIRONMENT,
        "status": "passed",
        "runtime_seconds": max(0.0, time.monotonic() - started),
        "result_path": str(paths.output_summary),
    }
    _atomic_json(paths.terminal_manifest, manifest)
    return summary


__all__ = ["run_subject_artifact_finalize"]
