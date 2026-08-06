"""Compact, evidence-bounded report for the MobileBCI v5 development screen."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_mobile_diffusion_v5"))


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> str:
    try:
        return f"{float(value):+.5f}"
    except (TypeError, ValueError):
        return "N/A"


def run(config: Mapping[str, Any], run_dir: Path) -> Mapping[str, Any]:
    root = CODE_ROOT / str(config["output_root"])
    protocol = json.loads((root / "protocol/result_summary.json").read_text())
    routing = json.loads((root / "aggregate/routing_decision.json").read_text())
    methods = _csv(root / "aggregate/method_summary.csv")
    effects = _csv(root / "aggregate/paired_effects.csv")
    pc_root = CODE_ROOT / str(config["pc_output_root"])
    pc = json.loads((pc_root / "result_summary.json").read_text()) if (pc_root / "result_summary.json").is_file() else {"status": "not_completed"}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    lines = [
        "# MobileBCI protocol repair and direct diffusion interaction screen (v5)",
        "",
        "This is development exploration, not confirmation. The eight sealed participants were not opened.",
        "",
        "## Data and protocol validity",
        "",
        f"The repaired experiment is explicitly **processed-EEG+IMU**; source EOG is disabled. Event onsets use the official 100 Hz sample-index interpretation. Of {protocol['protocol_denominator']} protocol units, {protocol['eligible_protocol_units']} were eligible, {protocol['blocked_no_later_block']} were blocked for no later block, and {protocol['missing_protocol_units']} were missing. ERP/SSVEP eligible counts were {protocol['eligible_by_task']['ERP']}/{protocol['eligible_by_task']['SSVEP']}.",
        "",
        "## Direct one-seed factorial",
        "",
        "| Protocol | H_D DIFF-MATCH−DET-MATCH | H_S NULL | H_S WRONG | Safety | Population Pareto dominance | Extra seeds |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for name, decision in routing["protocol_decisions"].items():
        effect = decision["mean_effects"]
        lines.append(f"| {name} | {_number(effect['diffusion_increment'])} | {_number(effect['support_utility'])} | {_number(effect['specificity'])} | {'pass' if decision['safety_passed'] else 'fail'} | {'yes' if decision['pareto_dominated_by_population'] else 'no'} | {'authorized' if decision['additional_seeds_authorized'] else 'not authorized'} |")
    lines.extend(["", "The direct interaction and temporal-order effects are in `paired_effects.csv`; units are participants, not windows. DIFF-NULL and DIFF-POP are the same learned context-dropout population arm in this formulation and are not counted as independent replications.", "", "## Method operating points", "", "| Protocol | Method | Motion-coherence reduction | Low-motion preservation | PSD distortion | Covariance distortion | Observation change |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in methods:
        if row.get("method") in {"RAW", "POP", "DET-NULL", "DET-MATCH", "DIFF-NULL", "DIFF-MATCH", "DIFF-SHUFFLED"}:
            lines.append(f"| {row['protocol']} | {row['method']} | {_number(row.get('motion_coherence_reduction'))} | {_number(row.get('nonartifact_observation_preservation'))} | {_number(row.get('reference_free_psd_distortion'))} | {_number(row.get('reference_free_covariance_distortion'))} | {_number(row.get('observation_change_ratio'))} |")
    lines.extend(["", "## P-C bounded-candidate selector diagnostic", "", f"Status: `{pc.get('status')}`. Infeasible units abstain to POP and remain in the full denominator. M0 uses the seven frozen features with training-unit achieved-coverage calibration; M1 adds output disagreement. This diagnostic is independent of the Mobile factorial.", "", "## Scientific boundary", ""])
    if routing["additional_seeds_authorized"]:
        conclusion = "At least one protocol met the pre-specified one-seed route for optimization-stability seeds; this remains exposed development evidence."
    else:
        conclusion = "The current v5 formulation did not meet the pre-specified route for extra optimization seeds. This closes only this implementation, not diffusion or personalization families."
    lines.extend([conclusion, "", f"Factorial run code began from commit `d2bfe9a`; aggregation/report code HEAD was `{head}`. Confirmation eligibility is false.", ""])
    report = CODE_ROOT / "reports/mobile_temporal_diffusion_v5.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    summary = {**routing, "protocol_validity": protocol, "pc_selector_diagnostic": pc, "report": str(report), "report_generation_git_head": head}
    (root / "result_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
