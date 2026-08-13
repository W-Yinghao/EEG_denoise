"""V31 claim narrowing and exact frozen support-duration repair.

No stage fits or updates model parameters, opens sealed data, edits the
manuscript, or sends the prepared editor email.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from eeg_scad.cli import run_v30 as v30
from eeg_scad.energy.projector import projector
from eeg_scad.evaluation.claim_package_v31 import (
    claim_rows,
    reviewer_rows,
    scope_comparison,
    write_scope_and_consultation_reports,
)
from eeg_scad.evaluation.common_panel_v30 import load_panel, read_role_rows, sha256
from eeg_scad.evaluation.natural_metrics_v28 import natural_metrics_v28
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.support_duration_v31 import (
    attach_exact_support,
    duration_contract,
    materialize_exact_support_bank,
    validate_exact_manifest,
)


ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", Path(__file__).resolve().parents[3]))
RESULT = ROOT / "results/claim_narrowing_v31"
CONFIG = ROOT / "configs/claim_narrowing_v31.yaml"
BASE = "220dcbaaabdef0cb8d1ac91b87b0d1cc8b7109cf"
V24 = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24")
V30_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/frozen_candidate_v30")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/claim_narrowing_v31")
ROLE = ROOT / "results/pa_el_scad_v24/role_manifest.csv"
V26_SEEDS = (20260828, 20260829, 20260830)
V29_SEEDS = (20260905, 20260906, 20260907)


def _cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def _folds() -> list[dict[str, Any]]:
    return yaml.safe_load((ROOT / "configs/pa_sc_cdm_v29/folds.yaml").read_text())["folds"]


def _owners() -> list[str]:
    return [*_cfg()["participants"], _cfg()["auxiliary_support_owner"]]


def _index() -> int:
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def _cell(index: int | None = None) -> tuple[int, int, int, int]:
    index = _index() if index is None else index
    fold, slot = index // 3, index % 3
    return fold, slot, V26_SEEDS[slot], V29_SEEDS[slot]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _panel_index(fold: int, stream: str) -> list[int]:
    return json.loads((ROOT / f"results/frozen_candidate_v30/panel_index_fold_{fold}.json").read_text())[stream]


def _panel(fold: int, stream: str, evaluator: bool = False) -> dict[str, Any]:
    return load_panel(V24, ROLE, fold, stream, _panel_index(fold, stream), evaluator)


def _bank(fold: int) -> dict[str, np.ndarray]:
    with np.load(DERIVED / f"support_bank_exact/fold_{fold}.npz", allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _git(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()


def _binding_payload() -> dict[str, Any]:
    terminal = json.loads((ROOT / "results/frozen_candidate_v30/terminal_manifest.json").read_text())
    inventory = ROOT / "results/frozen_candidate_v30/checkpoint_inventory.csv"
    files = {
        "checkpoint_inventory": inventory,
        "all_donor_matrix": ROOT / "results/frozen_candidate_v30/all_donor_matrix.csv",
        "falsification_effects": ROOT / "results/frozen_candidate_v30/falsification_effects.csv",
        "privacy_linkage": ROOT / "results/frozen_candidate_v30/privacy_linkage.csv",
        "step_latency_effects": ROOT / "results/frozen_candidate_v30/step_latency_effects.csv",
    }
    return {
        "v30_terminal_commit": BASE,
        "v30_lineage": {
            "base": terminal["base_commit"],
            "implementation": terminal["implementation_commit"],
            "common_panel": terminal["common_panel_commit"],
            "specificity": terminal["specificity_commit"],
            "duration_latency": terminal["duration_latency_commit"],
            "natural_privacy": terminal["natural_commit"],
            "ledger_v2_2": terminal["ledger_v2_2_commit"],
            "report": terminal["report_commit"],
            "terminal": BASE,
        },
        "digests": {name: _file_digest(path) for name, path in files.items()},
        "rows": {name: max(sum(1 for _ in path.open()) - 1, 0) for name, path in files.items()},
        "selected_candidate": "none",
        "sealed_confirmation_authorized": False,
        "V30_unchanged": True,
        "sealed_reads": 0,
    }


def preflight(run: Path) -> dict[str, Any]:
    cfg = _cfg()
    expected = {
        "V25": "a7d9d647b69e152255b62dbca917a4b3ed082915",
        "V26": "7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c",
        "V27": "40eae116e70e9de7fe0af55d64ee25551932c4a8",
        "V28": "f7aec43e8fae1d18c2831ee44b00eae9a0098e7e",
        "V29": "9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37",
        "V30": BASE,
    }
    refs = {
        "V25": _git("codex/setcalibdiff-raw-support-v25"),
        "V26": _git("codex/calib-sdedit-v26"),
        "V27": _git("codex/calib-energy-v27"),
        "V28": _git("codex/support-clean-conditional-diffusion-v28"),
        "V29": _git("codex/pop-anchored-support-adapter-v29"),
        "V30": _git("codex/frozen-candidate-consolidation-v30"),
    }
    ledger = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    manuscript_status = subprocess.check_output(
        ["git", "diff", "--name-only", BASE, "--", "taas_submission"], cwd=ROOT, text=True
    ).strip()
    checks = {
        "base_sha": _git(BASE) == BASE,
        "base_is_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT).returncode == 0,
        "source_refs": refs == expected,
        "ledger_v2_3": "**版本：** v2.3" in ledger and "V31" in ledger,
        "development_only": cfg["development_only"] is True,
        "new_model_training": cfg["new_model_training"] is False,
        "sealed_registry": cfg["sealed_participants"] == ["sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22"],
        "manuscript_unchanged": manuscript_status == "",
        "selected_candidate_none": json.loads((ROOT / "results/frozen_candidate_v30/final_candidate_selection.json").read_text())["selected_candidate"] == "none",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    binding = _binding_payload()
    _json(RESULT / "v30_binding.json", binding)
    (ROOT / "reports/v31_v30_freeze_note.md").write_text(
        "# V31 V30 freeze note\n\n"
        f"V31 is based exactly on V30 terminal `{BASE}`. The checkpoint inventory, all-donor, falsification, privacy, latency, selection, and Git lineage are digest-bound in `v30_binding.json`. V30 remains read-only; `selected_candidate=none`, sealed confirmation remains unauthorized, and this round trains no model.\n\n"
        "The externally named v2.3 ledger file was not present on the server. The explicit V31 user instruction was therefore synchronized as the v2.3 start state on top of the complete committed v2.2 ledger, with that provenance recorded rather than inferred silently.\n"
    )
    (ROOT / "reports/v31_project_plan.md").write_text(
        "# V31 project plan\n\n"
        "V31 supersedes only the flawed V30 support-duration evidence, reruns four frozen methods on the unchanged common development panel, and prepares claim, scope, AE-consultation, reviewer-response, and manuscript-blueprint documents. It does not train a model, open sealed data, modify manuscript source, create a PR, merge master, or send email.\n"
    )
    value = {
        "stage": "R0", "status": "PASS", "checks": checks, "refs": refs,
        "external_ledger_v2_3_file_found": False,
        "ledger_sync_source": "V31 user instruction applied to committed v2.2 ledger",
        "query_EOG_inference_reads": 0, "query_operator_inference_reads": 0,
        "event_inference_reads": 0, "sealed_reads": 0,
    }
    _json(run / "result_summary.json", value)
    return value


def duration_audit(run: Path) -> dict[str, Any]:
    from eeg_scad.evaluation.common_panel_v30 import support_starts

    starts = {str(duration): support_starts(duration) for duration in (5, 10, 30, 120)}
    overlap = any(right < left + 200 for left, right in zip(starts["5"], starts["5"][1:]))
    source = (ROOT / "src/eeg_scad/evaluation/common_panel_v30.py").read_text()
    validator = (ROOT / "src/eeg_scad/evaluation/support_duration.py").read_text()
    audit = {
        "verdict": "V30_DURATION_EVIDENCE_SUPERSEDED",
        "V30_support_starts_5s": starts["5"],
        "V30_5s_overlap_present": overlap,
        "V30_short_duration_uses_full_120s_EOG_center_scale": "eog[:, :12000]" in source,
        "V30_120s_window_count": len(starts["120"]),
        "V30_120s_all_nonoverlap_count": 60,
        "V30_validator_checks_overlap": "right < left + 200" in validator or "np.diff" in validator,
        "V30_validator_checks_future_normalization": "normalization" in validator,
        "scope_of_supersession": "V30 duration evidence only; other V30 findings remain frozen",
        "V30_modified": False,
    }
    required = overlap and audit["V30_short_duration_uses_full_120s_EOG_center_scale"] and audit["V30_120s_window_count"] == 16 and not audit["V30_validator_checks_overlap"] and not audit["V30_validator_checks_future_normalization"]
    if not required:
        raise RuntimeError(audit)
    _json(RESULT / "support_duration_audit.json", audit)
    (ROOT / "reports/v31_support_duration_audit.md").write_text(
        "# V31 support-duration implementation audit\n\n"
        "Verdict: `V30_DURATION_EVIDENCE_SUPERSEDED`. This supersedes only V30 duration rows, not its common-panel, specificity, falsification, natural, latency, privacy, or final-selection findings.\n\n"
        "The committed V30 implementation had four defects: `support_starts(5s)` produced overlapping 2 s windows; short-duration EOG center/scale came from the first 120 s; the 120 s condition used 16 windows rather than all 60 non-overlapping windows; and its validator checked uniqueness/bounds but not overlap or future-normalization. V31 leaves V30 untouched and repairs the evidence in a new namespace.\n"
    )
    _json(run / "result_summary.json", {"stage": "R1", "status": "PASS", **audit, "sealed_reads": 0})
    return audit


def materialize(run: Path) -> dict[str, Any]:
    data = yaml.safe_load((ROOT / "configs/setcalibdiff_v25/data.yaml").read_text())
    rows: list[dict[str, Any]] = []
    for fold in _folds():
        rows.extend(materialize_exact_support_bank(
            data, fold, _owners(), DERIVED / f"support_bank_exact/fold_{fold['fold']}.npz"
        ))
    validation = validate_exact_manifest(rows)
    _csv(RESULT / "exact_support_manifest.csv", rows)
    value = {
        "stage": "R2", "status": "PASS", **validation,
        "support_owners": len(_owners()), "query_auxiliary_reads": 0, "sealed_reads": 0,
    }
    _json(run / "result_summary.json", value)
    return value


def _support_outputs(
    batch: Mapping[str, Any],
    fold: int,
    v26_seed: int,
    v29_seed: int,
    encoded: Mapping[str, torch.Tensor] | None,
    noise_seed: int,
) -> dict[str, np.ndarray]:
    # Reuse the frozen V30 execution path.  The only changed input is the exact
    # V31 support tensor; checkpoints, query, noise, K, sigma, and steps remain fixed.
    return v30._outputs_with_encoded(batch, fold, v26_seed, v29_seed, encoded, noise_seed)


@torch.no_grad()
def duration_inference(run: Path) -> dict[str, Any]:
    fold, slot, v26_seed, v29_seed = _cell()
    device = torch.device("cuda")
    bank = _bank(fold)
    paired = _panel(fold, "paired", False)
    natural = _panel(fold, "natural", False)
    paired_evaluator = _panel(fold, "paired", True)
    _, det, _ = v30.v26._load_bundle(fold, v26_seed, device)
    full_context: torch.Tensor | None = None
    full_pi: torch.Tensor | None = None
    paired_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    natural_outputs: dict[str, np.ndarray] = {}
    owners = _owners()
    for duration in _cfg()["durations_seconds"]:
        contract = duration_contract(int(duration))
        noise_seed = 20260930 + fold * 100 + slot
        if duration == 0:
            pbatch, nbatch, encoded, nencoded = paired, natural, None, None
            elapsed = 0.0
        else:
            pbatch = attach_exact_support(paired, bank, owners, int(duration))
            nbatch = attach_exact_support(natural, bank, owners, int(duration))
            start = time.perf_counter()
            encoded = v30._encode(det, pbatch["support_eeg"], pbatch["support_eog"], device)
            nencoded = v30._encode(det, nbatch["support_eeg"], nbatch["support_eog"], device)
            elapsed = 1000 * (time.perf_counter() - start) / max(len(paired["y"]) + len(natural["y"]), 1)
            if duration == 120:
                full_context = encoded["context"]
                full_pi = projector(encoded["basis"])
        outputs = _support_outputs(pbatch, fold, v26_seed, v29_seed, encoded, noise_seed)
        noutputs = _support_outputs(nbatch, fold, v26_seed, v29_seed, nencoded, noise_seed + 5000)
        for method, clean in outputs.items():
            for index, meta in enumerate(paired["meta"]):
                risk = paired_metrics(
                    paired_evaluator["x"][index], paired["y"][index],
                    paired_evaluator["artifact"][index], paired["y"][index] - clean[index],
                )["rrmse_temporal"]
                paired_rows.append({
                    "panel": "paired", "fold": fold, "slot": slot,
                    "participant": meta["participant"], "session": meta["session"], "task": meta["task"],
                    "method": method, "paired_risk": risk, **contract,
                    "support_encoding_ms_per_query": elapsed,
                    "same_checkpoint": 1, "same_query": 1, "same_noise": 1, "K": 1,
                })
        for method, clean in noutputs.items():
            natural_outputs[f"{method}_D{duration}"] = clean
        if duration:
            current_pi = projector(encoded["basis"])
            diagnostic_rows.append({
                "fold": fold, "slot": slot, **contract,
                "context_stability_to_120": "pending" if duration != 120 else 0.0,
                "projector_stability_to_120": "pending" if duration != 120 else 0.0,
                "support_encoding_ms_per_query": elapsed,
                "context": encoded["context"].cpu().numpy(),
                "projector": current_pi.cpu().numpy(),
            })
    if full_context is None or full_pi is None:
        raise RuntimeError("full support reference missing")
    # Replace only the temporary arrays used for within-cell diagnostics.
    diagnostic_csv: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        context = row.pop("context")
        pi = row.pop("projector")
        row["context_stability_to_120"] = float(np.linalg.norm(context - full_context.cpu().numpy(), axis=1).mean())
        row["projector_stability_to_120"] = float(np.linalg.norm(pi - full_pi.cpu().numpy(), axis=(1, 2)).mean())
        diagnostic_csv.append(row)
    paired_path = DERIVED / f"inference/paired_fold_{fold}_slot_{slot}.csv"
    natural_path = DERIVED / f"inference/natural_fold_{fold}_slot_{slot}.npz"
    diagnostic_path = DERIVED / f"inference/diagnostic_fold_{fold}_slot_{slot}.csv"
    _csv(paired_path, paired_rows)
    natural_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(natural_path, **natural_outputs)
    _csv(diagnostic_path, diagnostic_csv)
    value = {
        "stage": "R3", "status": "PASS", "fold": fold, "slot": slot,
        "paired_rows": len(paired_rows), "natural_outputs": len(natural_outputs),
        "paired_sha256": _file_digest(paired_path), "natural_sha256": _file_digest(natural_path),
        "diagnostic_sha256": _file_digest(diagnostic_path),
        "new_model_trained": False, "same_checkpoint": True, "same_query": True,
        "same_diffusion_noise": True, "K": 1,
        "query_EOG_inference_reads": 0, "query_operator_inference_reads": 0,
        "event_inference_reads": 0, "sealed_reads": 0,
    }
    _json(run / "result_summary.json", value)
    return value


def freeze_outputs(run: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        for slot in range(3):
            path = DERIVED / f"inference/natural_fold_{fold}_slot_{slot}.npz"
            rows.append({"fold": fold, "slot": slot, "path": str(path), "sha256": _file_digest(path)})
    _csv(RESULT / "natural_output_manifest.csv", rows)
    value = {
        "stage": "R3_FREEZE", "status": "PASS", "outputs": len(rows),
        "query_EOG_inference_reads": 0, "query_operator_inference_reads": 0,
        "event_inference_reads": 0, "sealed_reads": 0,
    }
    _json(RESULT / "output_freeze.json", value)
    _json(run / "result_summary.json", value)
    return value


def duration_evaluator(run: Path) -> dict[str, Any]:
    if json.loads((RESULT / "output_freeze.json").read_text())["status"] != "PASS":
        raise RuntimeError("natural output freeze missing")
    fold, slot, _, _ = _cell()
    inference = _panel(fold, "natural", False)
    evaluator = _panel(fold, "natural", True)
    scale = np.load(V24 / f"fold_{fold}/eeg_scale.npy")
    path = DERIVED / f"inference/natural_fold_{fold}_slot_{slot}.npz"
    registered = next(row for row in csv.DictReader((RESULT / "natural_output_manifest.csv").open()) if int(row["fold"]) == fold and int(row["slot"]) == slot)
    if _file_digest(path) != registered["sha256"]:
        raise RuntimeError("frozen natural output digest mismatch")
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            method, duration = key.rsplit("_D", 1)
            clean = np.asarray(archive[key])
            contract = duration_contract(int(duration))
            for index, meta in enumerate(inference["meta"]):
                metric = natural_metrics_v28(
                    inference["y"][index], clean[index], evaluator["latent"][index],
                    evaluator["teacher_artifact"][index], scale,
                )
                rows.append({
                    "panel": "natural", "fold": fold, "slot": slot,
                    "participant": meta["participant"], "session": meta["session"], "task": meta["task"],
                    "method": method,
                    "natural_remaining_ratio": metric["heldout_eog_remaining_ratio"],
                    "low_eog_observation_retention": metric["low_eog_observation_retention"],
                    **contract,
                })
    output = DERIVED / f"evaluation/natural_fold_{fold}_slot_{slot}.csv"
    _csv(output, rows)
    value = {
        "stage": "R4_EVALUATOR", "status": "PASS", "fold": fold, "slot": slot,
        "rows": len(rows), "evaluator_after_freeze": True,
        "query_EOG_reads": "evaluator_only", "query_operator_reads": "evaluator_only",
        "event_reads": 0, "sealed_reads": 0,
    }
    _json(run / "result_summary.json", value)
    return value


def _mean_rows(rows: list[dict[str, str]], metric: str, panel: str) -> list[dict[str, Any]]:
    method_of = lambda row: row.get("method", "SUPPORT_ENCODING")
    keys = sorted({(method_of(row), int(row["duration_seconds"])) for row in rows})
    result: list[dict[str, Any]] = []
    for method, duration in keys:
        chosen = [float(row[metric]) for row in rows if method_of(row) == method and int(row["duration_seconds"]) == duration]
        contract = duration_contract(duration)
        result.append({
            "evidence_status": "V31_exact_duration_contract", "panel": panel,
            "method": method, "metric": metric, "mean": float(np.mean(chosen)),
            "median": float(np.median(chosen)), "rows": len(chosen), **contract,
        })
    return result


def duration_aggregate(run: Path) -> dict[str, Any]:
    paired: list[dict[str, str]] = []
    natural: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    for path in sorted((DERIVED / "inference").glob("paired_*.csv")):
        paired.extend(csv.DictReader(path.open()))
    for path in sorted((DERIVED / "evaluation").glob("natural_*.csv")):
        natural.extend(csv.DictReader(path.open()))
    for path in sorted((DERIVED / "inference").glob("diagnostic_*.csv")):
        diagnostics.extend(csv.DictReader(path.open()))
    rows = _mean_rows(paired, "paired_risk", "paired")
    rows += _mean_rows(natural, "natural_remaining_ratio", "natural")
    rows += _mean_rows(natural, "low_eog_observation_retention", "natural")
    for metric in ("context_stability_to_120", "projector_stability_to_120", "support_encoding_ms_per_query"):
        rows += _mean_rows(diagnostics, metric, "diagnostic")
    historical = []
    for row in csv.DictReader((ROOT / "results/frozen_candidate_v30/support_duration_effects.csv").open()):
        historical.append({
            "evidence_status": "historical_invalid_duration_contract",
            "panel": row["panel"], "method": row["method"], "metric": row["metric"],
            "mean": row["mean"], "median": row["median"], "rows": row["rows"],
            **duration_contract(int(row["duration_seconds"])),
        })
    _csv(RESULT / "support_duration_repair.csv", rows + historical)
    value = {
        "stage": "R4", "status": "PASS", "support_duration_repaired": True,
        "exact_rows": len(rows), "historical_superseded_rows": len(historical),
        "V30_duration_evidence_status": "V30_DURATION_EVIDENCE_SUPERSEDED",
        "V30_other_science_unchanged": True, "new_model_trained": False,
        "sealed_reads": 0,
    }
    _json(RESULT / "duration_repair_summary.json", value)
    exact = [row for row in rows if row["evidence_status"] == "V31_exact_duration_contract"]
    table_rows = [
        f"| {row['panel']} | {row['method']} | {row['duration_seconds']} | {row['effective_seconds']} | {row['window_count']} | {row['metric']} | {float(row['mean']):.6f} |"
        for row in exact
    ]
    (ROOT / "reports/v31_support_duration_repair.md").write_text(
        "# V31 exact support-duration repair\n\n"
        "Frozen checkpoints were replayed on the unchanged V30 common panel with the same queries, K=1, and same per-cell diffusion noise. Windows are chronological, non-overlapping 2 s prefixes; EOG coordinates use only the declared acquisition prefix. The 5 s condition therefore has a 5 s acquisition span but two windows / 4 s effective model exposure. Zero support is the architectural population bypass.\n\n"
        "V30 rows remain in the CSV as `historical_invalid_duration_contract`; only `V31_exact_duration_contract` rows are active evidence.\n\n"
        "| panel | method | acquisition s | effective s | windows | metric | mean |\n|---|---|---:|---:|---:|---|---:|\n"
        + "\n".join(table_rows) + "\n"
    )
    _json(run / "result_summary.json", value)
    return value


def documents(run: Path) -> dict[str, Any]:
    write_scope_and_consultation_reports(ROOT / "reports", RESULT)
    _csv(RESULT / "claim_evidence_matrix.csv", claim_rows())
    _json(RESULT / "scope_comparison.json", scope_comparison())
    _csv(RESULT / "reviewer_response_map.csv", reviewer_rows())
    value = {
        "stage": "R5_R8", "status": "PASS", "claims": len(claim_rows()),
        "reviewer_rows": len(reviewer_rows()), "scope_A_prepared": True,
        "scope_B_prepared": True, "recommended_scope": "A_audit_centric",
        "AE_email_prepared": True, "AE_email_sent": False,
        "manuscript_modified": False, "sealed_reads": 0,
    }
    _json(run / "result_summary.json", value)
    return value


def final_package(run: Path) -> dict[str, Any]:
    repair = json.loads((RESULT / "duration_repair_summary.json").read_text())
    decision = {
        "base_v30_commit": BASE,
        "new_model_trained": False,
        "sealed_opened": False,
        "support_duration_repaired": bool(repair["support_duration_repaired"]),
        "scope_A_status": "prepared",
        "scope_B_status": "prepared",
        "recommended_scope": "A_audit_centric",
        "AE_email_prepared": True,
        "AE_email_sent": False,
        "manuscript_modified": False,
        "next_action": "USER_REVIEW_AND_AE_CONSULTATION",
    }
    _json(RESULT / "decision.json", decision)
    report = """# V31 final package

V31 does not produce a new method claim or select a final candidate. It supersedes only V30's support-duration evidence, preserves all other V30 findings, and prepares two evidence-bounded revision scopes.

Recommended for consultation: **Scope A — audit-centric**. This is a recommendation to ask the AE, not an automatic manuscript decision. Scope B remains prepared as a higher-risk method-centric alternative around V27 EnergySDEdit lambda_y=0.5.

The prepared AE email has not been sent. No manuscript source was modified or compiled, no sealed outcome was opened, and no new model was trained. The next action is `USER_REVIEW_AND_AE_CONSULTATION`.
"""
    (ROOT / "reports/v31_final_package.md").write_text(report)
    value = {"stage": "R9", "status": "PASS", **decision, "sealed_reads": 0}
    _json(run / "result_summary.json", value)
    return value


def ledger_check(run: Path) -> dict[str, Any]:
    path = ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md"
    text = path.read_text()
    value = {
        "stage": "R9_LEDGER", "status": "PASS", "version": "v2.4",
        "sha256": _file_digest(path), "V31_recorded": "V31 — Claim Narrowing" in text,
        "sealed_reads": 0,
    }
    if "**版本：** v2.4" not in text or not value["V31_recorded"]:
        raise RuntimeError(value)
    _json(RESULT / "ledger_sync.json", value)
    _json(run / "result_summary.json", value)
    return value


def _lineage() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in sorted((RESULT / "runs").glob("*/job_*")):
        paths = sorted(job.glob("task_*")) or [job]
        for path in paths:
            complete = (path / "result_summary.json").is_file() or (path / "pytest.txt").is_file()
            job_id = job.name.removeprefix("job_")
            status = "accepted" if complete else "failed"
            recovery_of = ""
            if job_id in {"939597", "939603", "939605"}:
                status = "failed"
            elif job_id == "939598":
                status = "recovery"
                recovery_of = "939597"
            elif job_id == "939607":
                status = "recovery"
                recovery_of = "939605"
            rows.append({
                "stage": job.parent.name,
                "job_id": job_id,
                "array_task": path.name.removeprefix("task_") if path.name.startswith("task_") else "",
                "status": status,
                "recovery_of": recovery_of,
                "scientific_setting_changed": False,
            })
    return rows


def package(run: Path) -> dict[str, Any]:
    lineage = _lineage()
    _csv(RESULT / "job_lineage.csv", lineage)
    (ROOT / "reports/slurm").mkdir(parents=True, exist_ok=True)
    lines = ["# V31 Slurm lineage", "stage\tjob_id\tarray_task\tstatus\trecovery_of\tscientific_setting_changed"]
    lines += ["\t".join(str(row[key]) for key in ("stage", "job_id", "array_task", "status", "recovery_of", "scientific_setting_changed")) for row in lineage]
    (ROOT / "reports/slurm/v31_job_ids.txt").write_text("\n".join(lines) + "\n")

    def tests(stage: str) -> int:
        paths = sorted((RESULT / f"runs/{stage}").glob("job_*/pytest.txt"))
        match = re.search(r"(\d+) passed", paths[-1].read_text()) if paths else None
        return int(match.group(1)) if match else 0

    decision = json.loads((RESULT / "decision.json").read_text())
    queue = subprocess.check_output(["squeue", "--me", "--noheader", "-o", "%i %j %T"], text=True)
    jobs = lambda status: sorted({row["job_id"] for row in lineage if row["status"] == status}, key=int)
    ledger = ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md"
    terminal = {
        "protocol_id": "claim_narrowing_ae_consult_duration_repair_v31",
        "development_only": True,
        "base_commit": BASE,
        "implementation_commit": "3b2911f2dbc1ea1f32618b02ab92db53f5aa6bc8",
        "duration_repair_commit": "2a3d4489cc1e6cc5c46c2236321d1b9d63cfdc2a",
        "claim_package_commit": "5ab0087f8642bf9a7623ffa009b3d3d4631b9534",
        "ledger_v2_4_commit": "f952faedb745eef5d7bd350c446b183dfe3f14d6",
        "report_package_commit": "9b2af58e0e3e34cc850d8dd0fa982f76b6799bdd",
        "terminal_commit": "SELF_REFERENTIAL_REPORTED_EXTERNALLY",
        "remote_sha": "reported_after_push",
        "push_status": "push_verified_after_terminal_commit",
        "accepted_jobs": jobs("accepted"),
        "failed_jobs": jobs("failed"),
        "recovery_jobs": jobs("recovery"),
        "current_v31_jobs": [line for line in queue.splitlines() if "v31_" in line],
        "targeted_tests": tests("r10-tests"),
        "clean_archive_tests": tests("r11-clean"),
        "query_EOG_inference_reads": 0,
        "query_operator_inference_reads": 0,
        "event_inference_reads": 0,
        "sealed_reads": 0,
        "A_track": "0c4f2301c1f873120fe54537cde3c76fff7ea3a2",
        "A_track_unchanged": True,
        "manuscript_unchanged": True,
        "AE_email_prepared": True,
        "AE_email_sent": False,
        "project_ledger_version": "v2.4",
        "project_ledger_sha256": _file_digest(ledger),
        **decision,
    }
    _json(RESULT / "terminal_manifest.json", terminal)
    _json(run / "result_summary.json", terminal)
    return terminal


STAGES = {
    "r0-preflight": preflight,
    "r1-duration-audit": duration_audit,
    "r2-materialize": materialize,
    "r3-duration-infer": duration_inference,
    "r3-freeze": freeze_outputs,
    "r4-duration-evaluator": duration_evaluator,
    "r4-duration-aggregate": duration_aggregate,
    "r5-r8-documents": documents,
    "r9-final-package": final_package,
    "r9-ledger": ledger_check,
    "r12-package": package,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args.run_dir)


if __name__ == "__main__":
    main()
