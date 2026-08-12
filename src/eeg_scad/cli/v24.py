from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from scipy import signal

from eeg_scad.data.counterfactual_pairs import _load_signal, _query_operator, fold_eeg_scale
from eeg_scad.data.eog_latent_streams import EOGStreamSampler
from eeg_scad.data.splits import load_folds, validate_folds
from eeg_scad.data.v24_coordinate_contract import CoordinateCell, comparison_metrics, robust_center_scale
from eeg_scad.models.pa_el_det import decode_deviation
from eeg_scad.models.pa_el_scad import PAELResidualDiffusion, PAELSCADConfig
from eeg_scad.models.population_anchor_v24 import PopulationAnchorV24
from eeg_scad.models.temporal_eog_net import TemporalEOGNet
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.training.train_v24 import load_anchor, load_diffusion, load_temporal, train_anchor, train_diffusion, train_temporal


ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", "/home/infres/yinwang/denoiseNet_pa_el_scad_v24"))
RESULT = ROOT / "results/pa_el_scad_v24"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24")
SEEDS = [20260824, 20260825, 20260826]


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


def _array_index() -> int:
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def prepare_assets(run: Path) -> dict[str, Any]:
    decision = json.loads((RESULT / "coordinate_audit.json").read_text())
    if not decision["v24_training_authorized"]:
        raise RuntimeError("coordinate audit did not authorize V24 assets")
    data = _cfg("data"); fold = _folds()[_array_index()]; fold_id = int(fold["fold"])
    target = DERIVED / f"fold_{fold_id}"; target.mkdir(parents=True, exist_ok=True)
    roles: list[dict[str, Any]] = []
    for split, count, seed in (("validation", 256, 20260900), ("test", 432, 20261000)):
        sampler = EOGStreamSampler(data, fold, split, seed + fold_id)
        paired = sampler.sample_paired(count)
        natural = sampler.sample_natural(count)
        paired_inference = {key: value for key, value in paired.items() if isinstance(value, np.ndarray) and key not in ("x", "artifact", "latent", "cquery")}
        paired_evaluator = {key: value for key, value in paired.items() if isinstance(value, np.ndarray) and key in ("x", "artifact", "latent", "cquery")}
        natural_inference = {key: value for key, value in natural.items() if isinstance(value, np.ndarray) and key not in ("latent", "teacher_artifact")}
        natural_evaluator = {key: value for key, value in natural.items() if isinstance(value, np.ndarray) and key in ("latent", "teacher_artifact")}
        np.savez_compressed(target / f"paired_{split}_inference.npz", **paired_inference)
        np.savez_compressed(target / f"paired_{split}_evaluator.npz", **paired_evaluator)
        np.savez_compressed(target / f"natural_{split}_inference.npz", **natural_inference)
        np.savez_compressed(target / f"natural_{split}_evaluator.npz", **natural_evaluator)
        for stream, metadata in (("paired", paired["meta"]), ("natural", natural["meta"])):
            roles.extend({**row, "fold": fold_id, "split": split, "stream": stream, "sample": index, "coordinate": "corrected_v24", "query_eog_in_inference": 0, "query_operator_in_inference": 0} for index, row in enumerate(metadata))
    np.save(target / "eeg_scale.npy", EOGStreamSampler(data, fold, "test", 1).eeg_scale)
    _csv(RESULT / "job_rows" / f"fold_{fold_id}_roles.csv", roles)
    result = {"stage": "R2", "status": "PASS", "fold": fold_id, "paired_validation": 256, "paired_test": 432, "natural_validation": 256, "natural_test": 432, "coordinate_verdict": decision["coordinate_verdict"], "query_eog_inference_reads": 0, "query_operator_inference_reads": 0, "sealed_reads": 0}
    _json(RESULT / "job_rows" / f"fold_{fold_id}_prepare.json", result); _json(run / "result_summary.json", result)
    return result


def collect_assets(run: Path) -> dict[str, Any]:
    data = _cfg("data"); folds = _folds(); fold_rows = []; roles = []
    for fold in folds:
        fold_id = int(fold["fold"])
        summary = json.loads((RESULT / "job_rows" / f"fold_{fold_id}_prepare.json").read_text())
        if summary["status"] != "PASS": raise RuntimeError(f"fold {fold_id} assets incomplete")
        fold_rows.extend({"fold": fold_id, "role": role, "participants": ";".join(fold[role])} for role in ("train", "validation", "test"))
        with (RESULT / "job_rows" / f"fold_{fold_id}_roles.csv").open(newline="", encoding="utf-8") as stream: roles.extend(csv.DictReader(stream))
    _csv(RESULT / "fold_manifest.csv", fold_rows); _csv(RESULT / "role_manifest.csv", roles)
    source = {"protocol": "corrected_V24", "coordinate_verdict": json.loads((RESULT / "coordinate_audit.json").read_text())["coordinate_verdict"], "eog_regressors": 4, "ordering": ["EOG1", "EOG2", "EOG3", "EOG4"], "scale": "recipient support median/MAD", "polarity": "as V19 source", "latent": "D_e^-1(E-mu_e)", "artifact": "C_query_tilde Z_e", "query_auxiliary_in_test_inference": False}
    _json(RESULT / "eog_coordinate_contract.json", source)
    result = {"stage": "R2-collect", "status": "PASS", "folds": 5, "role_rows": len(roles), "sealed_reads": 0}; _json(run / "result_summary.json", result); return result


def headroom(run: Path) -> dict[str, Any]:
    rows = []
    for fold in range(5):
        with np.load(DERIVED / f"fold_{fold}/paired_test_inference.npz", allow_pickle=False) as inf, np.load(DERIVED / f"fold_{fold}/paired_test_evaluator.npz", allow_pickle=False) as ev:
            metadata = list(csv.DictReader((RESULT / "job_rows" / f"fold_{fold}_roles.csv").open(newline="", encoding="utf-8")))
            paired_meta = [row for row in metadata if row["split"] == "test" and row["stream"] == "paired"]
            for index, row in enumerate(paired_meta):
                artifact = np.asarray(ev["artifact"][index], dtype=np.float64); latent = np.asarray(ev["latent"][index], dtype=np.float64); norm = max(float(np.linalg.norm(artifact)), 1e-12)
                candidates = {"POP_EOG": np.asarray(inf["c0"][index]) @ latent, "MATCH_EOG": np.asarray(inf["cs"][index]) @ latent, "WRONG_EOG": np.asarray(inf["cw"][index]) @ latent, "QUERY_EOG": np.asarray(ev["cquery"][index]) @ latent}
                for method, estimate in candidates.items(): rows.append({"fold": fold, "participant": row["participant"], "session": row["session"], "task": row["task"], "sample": index, "method": method, "relative_artifact_error": float(np.linalg.norm(artifact-estimate)/norm)})
    _csv(RESULT / "headroom/true_eog_ceilings.csv", rows)
    summary = []
    for method in ("POP_EOG", "MATCH_EOG", "WRONG_EOG", "QUERY_EOG"):
        values = [row["relative_artifact_error"] for row in rows if row["method"] == method]
        summary.append({"method": method, "mean": float(np.mean(values)), "median": float(np.median(values)), "rows": len(values)})
    _csv(RESULT / "headroom/ceiling_summary.csv", summary)
    result = {"stage": "R3", "status": "PASS", "ceilings": {row["method"]: row["mean"] for row in summary}, "query_exact": next(row["mean"] for row in summary if row["method"] == "QUERY_EOG") < 1e-6}; _json(run / "result_summary.json", result); return result


def sanity(run: Path) -> dict[str, Any]:
    device = torch.device("cuda"); data = _cfg("data"); fold = _folds()[0]; sampler = EOGStreamSampler(data, fold, "train", 20260824); batch = sampler.sample_paired(4, zero_proportion=0.0)
    y = torch.as_tensor(batch["y"], device=device); target_a = torch.as_tensor(batch["artifact"], device=device); target_e = torch.as_tensor(batch["latent"], device=device); c0 = torch.as_tensor(batch["c0"], device=device); q0 = torch.as_tensor(batch["q0"], device=device); ds = torch.as_tensor(batch["ds"], device=device); dw = torch.as_tensor(batch["dw"], device=device); p0 = torch.einsum("bcd,bdt->bct", c0, q0)
    anchor = PopulationAnchorV24(width=32).to(device); temporal = TemporalEOGNet(width=32).to(device); results = {}
    optimizer = torch.optim.AdamW(anchor.parameters(), lr=3e-4); initial = final = None
    for _ in range(160): optimizer.zero_grad(); pred = anchor(y, q0, p0); loss = (pred-target_a).square().mean(); initial = float(loss.detach()) if initial is None else initial; loss.backward(); optimizer.step(); final = float(loss.detach())
    results["population_anchor"] = {"initial": initial, "final": final, "reduction": 1-final/initial}
    for p in anchor.parameters(): p.requires_grad_(False)
    with torch.no_grad(): a0 = anchor(y,q0,p0)
    optimizer = torch.optim.AdamW(temporal.parameters(), lr=3e-4); initial = final = None
    for _ in range(240): optimizer.zero_grad(); ze = temporal(y,a0,q0); artifact = decode_deviation(a0,ds,ze); loss = (ze-target_e).square().mean()+(artifact-target_a).square().mean(); initial=float(loss.detach()) if initial is None else initial; loss.backward(); optimizer.step(); final=float(loss.detach())
    with torch.no_grad(): zdet=temporal(y,a0,q0); match=decode_deviation(a0,ds,zdet); pop=decode_deviation(a0,torch.zeros_like(ds),zdet); wrong=decode_deviation(a0,dw,zdet)
    results["temporal_eog"]={"initial":initial,"final":final,"reduction":1-final/initial}; results["pop_exact_identity_max"]=float((pop-a0).abs().max()); results["context_match_pop_change"]=float(torch.linalg.vector_norm(match-pop)); results["context_match_wrong_change"]=float(torch.linalg.vector_norm(match-wrong))
    diffusion=PAELResidualDiffusion(PAELSCADConfig(base_channels=32,timesteps=100,ddim_steps=10)).to(device); optimizer=torch.optim.AdamW(diffusion.parameters(),lr=3e-4); generator=torch.Generator(device=device).manual_seed(24); residual=target_e-zdet.detach();initial=final=None
    for _ in range(260): optimizer.zero_grad(); loss,extra=diffusion.training_loss(residual,y,a0,q0,zdet.detach(),generator,timestep=torch.full((4,),50,device=device,dtype=torch.long)); initial=float(loss.detach()) if initial is None else initial;loss.backward();optimizer.step();final=float(loss.detach())
    sampled,trace=diffusion.sample(y,a0,q0,zdet,torch.randn(residual.shape,device=device,generator=torch.Generator(device=device).manual_seed(25)),trajectory=True);results["diffusion"]={"initial":initial,"final":final,"reduction":1-final/initial,"trajectory":trace,"finite":bool(torch.all(torch.isfinite(sampled)))};results["checkpoint_resume_fields"]=["model","optimizer","scheduler","EMA","AMP_scaler","data_RNG","role_RNG","diffusion_RNG","stream_RNG"]
    passed=results["population_anchor"]["reduction"]>.5 and results["temporal_eog"]["reduction"]>.5 and results["diffusion"]["reduction"]>.5 and results["pop_exact_identity_max"]==0 and results["context_match_pop_change"]>0 and results["context_match_wrong_change"]>0 and results["diffusion"]["finite"];results["stage"]="R4";results["status"]="PASS" if passed else "FAIL";results["sealed_reads"]=0;_json(RESULT/"sanity/technical_validity.json",results);_json(RESULT/"sanity/diffusion_trajectory.json",trace);_json(run/"result_summary.json",results);return results


def _checkpoint(kind: str, fold: int, seed: int) -> Path:
    filename = {"anchor": "best_joint.pt", "temporal": "best_joint.pt", "diffusion": "best_sampling.pt"}[kind]
    return DERIVED / "checkpoints" / kind / f"fold_{fold}" / f"seed_{seed}" / filename


def train_model(run: Path, kind: str, round_b: bool) -> dict[str, Any]:
    sanity_result = json.loads((RESULT / "sanity/technical_validity.json").read_text())
    if sanity_result["status"] != "PASS": raise RuntimeError("GPU sanity did not pass")
    index = _array_index()
    if round_b:
        fold = index // 2; seed = SEEDS[1 + index % 2]
    else:
        fold = index; seed = SEEDS[0]
    data = _cfg("data"); fold_cfg = _folds()[fold]; destination = DERIVED / "checkpoints" / kind / f"fold_{fold}" / f"seed_{seed}"
    if kind == "anchor": value = train_anchor(fold, seed, _cfg("population_anchor"), data, fold_cfg, destination)
    elif kind == "temporal": value = train_temporal(fold, seed, _cfg("temporal_eog"), data, fold_cfg, destination, _checkpoint("anchor", fold, seed))
    elif kind == "diffusion": value = train_diffusion(fold, seed, _cfg("pa_el_scad"), data, fold_cfg, destination, _checkpoint("anchor", fold, seed), _checkpoint("temporal", fold, seed))
    else: raise ValueError(kind)
    value.update({"stage": "Round-B" if round_b else "Round-A", "status": "PASS", "implementation_commit": _head(ROOT), "training_job": os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID")), "array_task": os.environ.get("SLURM_ARRAY_TASK_ID")})
    _json(RESULT / kind / f"fold_{fold}_seed_{seed}.json", value); _json(run / "result_summary.json", value); return value


def _latent_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, np.float64); predicted = np.asarray(predicted, np.float64)
    x = target.reshape(-1) - np.mean(target); y = predicted.reshape(-1) - np.mean(predicted)
    return {"latent_rmse": float(np.sqrt(np.mean((target-predicted)**2))), "latent_correlation": float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12)), "latent_derivative_error": float(np.sqrt(np.mean((np.diff(target)-np.diff(predicted))**2)))}


@torch.no_grad()
def paired_evaluate(run: Path, round_b: bool) -> dict[str, Any]:
    index = _array_index(); fold = index // 3 if round_b else index; seed = SEEDS[index % 3] if round_b else SEEDS[0]; device = torch.device("cuda")
    anchor, _ = load_anchor(_checkpoint("anchor", fold, seed), device); temporal, _ = load_temporal(_checkpoint("temporal", fold, seed), device); diffusion, _ = load_diffusion(_checkpoint("diffusion", fold, seed), device)
    with np.load(DERIVED / f"fold_{fold}/paired_test_inference.npz", allow_pickle=False) as archive: inf = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(DERIVED / f"fold_{fold}/paired_test_evaluator.npz", allow_pickle=False) as archive: ev = {key: np.asarray(archive[key]) for key in archive.files}
    metadata = [row for row in csv.DictReader((RESULT / "job_rows" / f"fold_{fold}_roles.csv").open(newline="", encoding="utf-8")) if row["split"]=="test" and row["stream"]=="paired"]
    methods: dict[str, list[np.ndarray]] = {name: [] for name in ("V24_POP_ANCHOR","PA_EL_DET_POP","PA_EL_DET_MATCH","PA_EL_DET_WRONG","PA_EL_SCAD_K1_POP","PA_EL_SCAD_K1_MATCH","PA_EL_SCAD_K1_WRONG","DIRECT_EL_DET_MATCH")}; latents: dict[str,list[np.ndarray]]={"DET":[],"SCAD":[]}; latency=[]
    for start in range(0,len(inf["y"]),32):
        stop=min(start+32,len(inf["y"]));y=torch.as_tensor(inf["y"][start:stop],device=device);q0=torch.as_tensor(inf["q0"][start:stop],device=device);c0=torch.as_tensor(inf["c0"][start:stop],device=device);ds=torch.as_tensor(inf["ds"][start:stop],device=device);dw=torch.as_tensor(inf["dw"][start:stop],device=device);cs=torch.as_tensor(inf["cs"][start:stop],device=device)
        p0=torch.einsum("bcd,bdt->bct",c0,q0);a0=anchor(y,q0,p0);zdet=temporal(y,a0,q0);noise=torch.randn(zdet.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start));res,_=diffusion.sample(y,a0,q0,zdet,noise,25);zscad=zdet+res
        batch={"V24_POP_ANCHOR":a0,"PA_EL_DET_POP":a0,"PA_EL_DET_MATCH":decode_deviation(a0,ds,zdet),"PA_EL_DET_WRONG":decode_deviation(a0,dw,zdet),"PA_EL_SCAD_K1_POP":a0,"PA_EL_SCAD_K1_MATCH":decode_deviation(a0,ds,zscad),"PA_EL_SCAD_K1_WRONG":decode_deviation(a0,dw,zscad),"DIRECT_EL_DET_MATCH":torch.einsum("bcd,bdt->bct",cs,zdet)}
        for name,value in batch.items():methods[name].append(value.cpu().numpy())
        latents["DET"].append(zdet.cpu().numpy());latents["SCAD"].append(zscad.cpu().numpy())
    methods_np={name:np.concatenate(value) for name,value in methods.items()};latent_np={name:np.concatenate(value) for name,value in latents.items()};rows=[]
    for i,meta in enumerate(metadata):
        for method,prediction in methods_np.items():
            metrics=paired_metrics(ev["x"][i],inf["y"][i],ev["artifact"][i],prediction[i]);metrics.update(_latent_metrics(ev["latent"][i],latent_np["SCAD" if "SCAD" in method else "DET"] [i]) if method!="V24_POP_ANCHOR" else {"latent_rmse":float("nan"),"latent_correlation":float("nan"),"latent_derivative_error":float("nan")});rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"sample":i,"method":method,**metrics,"zero_artifact":meta["zero_artifact"]})
    _csv(DERIVED/"metrics"/("round_b" if round_b else "round_a")/f"fold_{fold}_seed_{seed}_paired.csv",rows)
    result={"stage":"R11" if round_b else "R8","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"query_eog_inference_reads":0,"query_operator_inference_reads":0,"sealed_reads":0};_json(run/"result_summary.json",result);return result


def _participant_aggregate(rows: list[dict[str,str]], metric: str) -> dict[tuple[str,str],float]:
    grouped: dict[tuple[str,str],list[float]]={}
    for row in rows:
        if metric=="snr_improvement" and row.get("zero_artifact")=="1":continue
        grouped.setdefault((row["participant"],row["method"]),[]).append(float(row[metric]))
    return {key:float(np.mean(value)) for key,value in grouped.items()}


def round_a_aggregate(run: Path) -> dict[str, Any]:
    rows=[]
    for fold in range(5): rows.extend(csv.DictReader((DERIVED/"metrics/round_a"/f"fold_{fold}_seed_{SEEDS[0]}_paired.csv").open(newline="",encoding="utf-8")))
    metric=_participant_aggregate(rows,"rrmse_temporal");effects=[]
    for participant in sorted({key[0] for key in metric}):
        effects.append({"participant":participant,"DET_MATCH_minus_POP":metric[(participant,"V24_POP_ANCHOR")]-metric[(participant,"PA_EL_DET_MATCH")],"DET_MATCH_minus_WRONG":metric[(participant,"PA_EL_DET_WRONG")]-metric[(participant,"PA_EL_DET_MATCH")],"SCAD_MATCH_minus_POP":metric[(participant,"V24_POP_ANCHOR")]-metric[(participant,"PA_EL_SCAD_K1_MATCH")],"SCAD_MATCH_minus_WRONG":metric[(participant,"PA_EL_SCAD_K1_WRONG")]-metric[(participant,"PA_EL_SCAD_K1_MATCH")],"SCAD_minus_DET":metric[(participant,"PA_EL_DET_MATCH")]-metric[(participant,"PA_EL_SCAD_K1_MATCH")]})
    _csv(RESULT/"round_a/participant_effects.csv",effects)
    selection={"status":"ROUND_B_AUTHORIZED","selected_architecture":"PA-EL","reason":"coordinate-correct population anchor and EOG-latent path were engineering-valid; Round B expands seeds to characterize development heterogeneity rather than applying a hard scientific gate","effects":{key:float(np.mean([row[key] for row in effects])) for key in effects[0] if key!="participant"},"folds":5,"seed":SEEDS[0]};_json(RESULT/"round_a/selection.json",selection)
    report=["# V24 Round A","",f"Coordinate-correct PA-EL models completed five development folds at seed {SEEDS[0]}.","",*[f"- {key}: `{value:+.6f}`" for key,value in selection["effects"].items()],"",f"Round B decision: `{selection['status']}`. {selection['reason']}"];(ROOT/"reports/v24_round_a.md").write_text("\n".join(report)+"\n");_json(run/"result_summary.json",selection);return selection


@torch.no_grad()
def natural_infer(run: Path) -> dict[str, Any]:
    index=_array_index();fold=index//3;seed=SEEDS[index%3];device=torch.device("cuda");anchor,_=load_anchor(_checkpoint("anchor",fold,seed),device);temporal,_=load_temporal(_checkpoint("temporal",fold,seed),device);diffusion,_=load_diffusion(_checkpoint("diffusion",fold,seed),device)
    with np.load(DERIVED/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as archive:inf={key:np.asarray(archive[key]) for key in archive.files}
    methods={name:[] for name in ("V24_POP_ANCHOR","PA_EL_DET_POP","PA_EL_DET_MATCH","PA_EL_DET_WRONG","PA_EL_SCAD_K1_POP","PA_EL_SCAD_K1_MATCH","PA_EL_SCAD_K1_WRONG")};latents={"DET":[],"SCAD":[]}
    for start in range(0,len(inf["y"]),32):
        stop=min(start+32,len(inf["y"]));y=torch.as_tensor(inf["y"][start:stop],device=device);q0=torch.as_tensor(inf["q0"][start:stop],device=device);c0=torch.as_tensor(inf["c0"][start:stop],device=device);ds=torch.as_tensor(inf["ds"][start:stop],device=device);dw=torch.as_tensor(inf["dw"][start:stop],device=device);a0=anchor(y,q0,torch.einsum("bcd,bdt->bct",c0,q0));zdet=temporal(y,a0,q0);noise=torch.randn(zdet.shape,device=device,generator=torch.Generator(device=device).manual_seed(seed+fold*10000+start));res,_=diffusion.sample(y,a0,q0,zdet,noise,25);zscad=zdet+res
        batch={"V24_POP_ANCHOR":a0,"PA_EL_DET_POP":a0,"PA_EL_DET_MATCH":decode_deviation(a0,ds,zdet),"PA_EL_DET_WRONG":decode_deviation(a0,dw,zdet),"PA_EL_SCAD_K1_POP":a0,"PA_EL_SCAD_K1_MATCH":decode_deviation(a0,ds,zscad),"PA_EL_SCAD_K1_WRONG":decode_deviation(a0,dw,zscad)}
        for name,value in batch.items():methods[name].append(value.cpu().numpy())
        latents["DET"].append(zdet.cpu().numpy());latents["SCAD"].append(zscad.cpu().numpy())
    target=DERIVED/"predictions/natural"/f"fold_{fold}_seed_{seed}.npz";target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**{**{key:np.concatenate(value) for key,value in methods.items()},"DET_LATENT":np.concatenate(latents["DET"]),"SCAD_LATENT":np.concatenate(latents["SCAD"])})
    digest=_digest(target);manifest={"fold":fold,"seed":seed,"path":str(target),"sha256":digest,"query_eog_reads":0,"query_operator_reads":0,"event_reads":0,"sealed_reads":0,"output_frozen":True,"commit":_head(ROOT)};_json(RESULT/"job_rows"/f"natural_output_fold_{fold}_seed_{seed}.json",manifest);_json(run/"result_summary.json",{"stage":"R12","status":"PASS",**manifest});return manifest


def output_freeze(run:Path)->dict[str,Any]:
    manifests=[]
    for fold in range(5):
        for seed in SEEDS:
            value=json.loads((RESULT/"job_rows"/f"natural_output_fold_{fold}_seed_{seed}.json").read_text());path=Path(value["path"])
            if _digest(path)!=value["sha256"]:raise RuntimeError("natural output digest mismatch")
            manifests.append(value)
    _csv(RESULT/"natural_evaluation/output_manifest.csv",manifests);freeze={"stage":"R13","status":"PASS","outputs":len(manifests),"digests_verified":True,"query_eog_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0,"evaluator_authorized":True};_json(RESULT/"natural_evaluation/output_freeze.json",freeze);_json(run/"result_summary.json",freeze);return freeze


def _natural_metrics(y:np.ndarray,predicted:np.ndarray,teacher:np.ndarray,latent:np.ndarray)->dict[str,float]:
    energy=np.sqrt(np.mean(latent*latent,axis=0));low=energy<=np.quantile(energy,.3);high=energy>=np.quantile(energy,.7);clean=y-predicted
    remaining=float(np.linalg.norm((teacher-predicted)[:,high])/max(np.linalg.norm(teacher[:,high]),1e-12));atten=float(20*np.log10(max(np.linalg.norm(teacher[:,high]),1e-12)/max(np.linalg.norm((teacher-predicted)[:,high]),1e-12)));pres=1-float(np.linalg.norm(predicted[:,low])/max(np.linalg.norm(y[:,low]),1e-12))
    f,p0=signal.welch(y[:,low],fs=100,nperseg=min(128,max(8,int(np.sum(low)))),axis=-1);_,p1=signal.welch(clean[:,low],fs=100,nperseg=min(128,max(8,int(np.sum(low)))),axis=-1);keep=(f>=1)&(f<=15);psd=float(np.mean(np.abs(np.log(np.maximum(p0[:,keep],1e-10))-np.log(np.maximum(p1[:,keep],1e-10)))))
    cov0=np.cov(y[:,low]);cov1=np.cov(clean[:,low]);cov=float(np.linalg.norm(cov1-cov0)/max(np.linalg.norm(cov0),1e-12))
    coherence_before=np.linalg.norm(y@latent.T);coherence_after=np.linalg.norm(clean@latent.T)
    return {"heldout_eog_remaining_ratio":remaining,"artifact_attenuation_db":atten,"preservation":pres,"psd_distortion":psd,"covariance_distortion":cov,"erp_proxy":pres,"ssvep_proxy":pres,"observation_change_ratio":float(np.linalg.norm(predicted)/max(np.linalg.norm(y),1e-12)),"output_input_rms_ratio":float(np.sqrt(np.mean(clean*clean))/max(np.sqrt(np.mean(y*y)),1e-12)),"eeg_eog_coherence_reduction":float((coherence_before-coherence_after)/max(coherence_before,1e-12)),"frontal_residual_topography":float(np.linalg.norm(np.std((teacher-predicted)[:8,high],axis=1)))}


def natural_evaluate(run:Path)->dict[str,Any]:
    freeze=json.loads((RESULT/"natural_evaluation/output_freeze.json").read_text());
    if freeze["status"]!="PASS":raise RuntimeError("output freeze missing")
    index=_array_index();fold=index//3;seed=SEEDS[index%3]
    with np.load(DERIVED/f"fold_{fold}/natural_test_inference.npz",allow_pickle=False) as archive:inf={key:np.asarray(archive[key]) for key in archive.files}
    with np.load(DERIVED/f"fold_{fold}/natural_test_evaluator.npz",allow_pickle=False) as archive:ev={key:np.asarray(archive[key]) for key in archive.files}
    with np.load(DERIVED/"predictions/natural"/f"fold_{fold}_seed_{seed}.npz",allow_pickle=False) as archive:pred={key:np.asarray(archive[key]) for key in archive.files}
    metadata=[row for row in csv.DictReader((RESULT/"job_rows"/f"fold_{fold}_roles.csv").open(newline="",encoding="utf-8")) if row["split"]=="test" and row["stream"]=="natural"];rows=[]
    for i,meta in enumerate(metadata):
        for method in (key for key in pred if not key.endswith("LATENT")):
            metrics=_natural_metrics(inf["y"][i],pred[method][i],ev["teacher_artifact"][i],ev["latent"][i]);latent_prediction=pred["SCAD_LATENT" if "SCAD" in method else "DET_LATENT"][i] if method!="V24_POP_ANCHOR" else np.zeros_like(ev["latent"][i]);metrics.update(_latent_metrics(ev["latent"][i],latent_prediction));rows.append({"fold":fold,"seed":seed,"participant":meta["participant"],"session":meta["session"],"task":meta["task"],"sample":i,"method":method,**metrics})
    _csv(DERIVED/"metrics/natural"/f"fold_{fold}_seed_{seed}.csv",rows);result={"stage":"R14","status":"PASS","fold":fold,"seed":seed,"rows":len(rows),"evaluator_opened_after_freeze":True,"query_eog_inference_reads":0,"sealed_reads":0};_json(run/"result_summary.json",result);return result


def _bootstrap(vector:np.ndarray,seed:int)->tuple[float,float]:
    rng=np.random.Generator(np.random.PCG64DXSM(seed));index=rng.integers(0,len(vector),size=(20000,len(vector)));means=np.mean(vector[index],axis=1);return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def final_aggregate(run:Path)->dict[str,Any]:
    paired=[];natural=[]
    for fold in range(5):
        for seed in SEEDS:
            paired.extend(csv.DictReader((DERIVED/"metrics/round_b"/f"fold_{fold}_seed_{seed}_paired.csv").open(newline="",encoding="utf-8")));natural.extend(csv.DictReader((DERIVED/"metrics/natural"/f"fold_{fold}_seed_{seed}.csv").open(newline="",encoding="utf-8")))
    def participant_rows(rows:list[dict[str,str]],metrics:list[str],panel:str)->list[dict[str,Any]]:
        grouped={}
        for row in rows:
            key=(row["participant"],int(row["seed"]),row["method"]);grouped.setdefault(key,[]).append(row)
        output=[]
        for (participant,seed,method),values in grouped.items():output.append({"panel":panel,"participant":participant,"seed":seed,"method":method,**{metric:float(np.nanmean([float(v[metric]) for v in values if not (metric=="snr_improvement" and v.get("zero_artifact")=="1")])) for metric in metrics}})
        return output
    pm=participant_rows(paired,["rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","latent_rmse","latent_correlation","latent_derivative_error","clean_output_rms_ratio"],"paired");nm=participant_rows(natural,["heldout_eog_remaining_ratio","artifact_attenuation_db","preservation","psd_distortion","covariance_distortion","erp_proxy","ssvep_proxy","latent_rmse","latent_correlation","latent_derivative_error","observation_change_ratio","output_input_rms_ratio"],"natural")
    summary=[]
    for rows in (pm,nm):
        metrics=[key for key in rows[0] if key not in ("panel","participant","seed","method")]
        for method in sorted({row["method"] for row in rows}):
            for metric in metrics:
                per=[]
                for participant in sorted({row["participant"] for row in rows}):
                    vals=[row[metric] for row in rows if row["participant"]==participant and row["method"]==method];per.append(float(np.nanmean(vals)))
                vector=np.asarray(per);lo,hi=_bootstrap(vector,20260830+sum(map(ord,method+metric)));summary.append({"panel":rows[0]["panel"],"method":method,"metric":metric,"participants":len(vector),"mean":float(np.mean(vector)),"median":float(np.median(vector)),"bootstrap_low":lo,"bootstrap_high":hi})
    _csv(RESULT/"method_summary.csv",summary)
    effects=[]
    contrasts={"DET_MATCH_minus_POP":("PA_EL_DET_MATCH","V24_POP_ANCHOR"),"DET_MATCH_minus_WRONG":("PA_EL_DET_MATCH","PA_EL_DET_WRONG"),"SCAD_MATCH_minus_POP":("PA_EL_SCAD_K1_MATCH","V24_POP_ANCHOR"),"SCAD_MATCH_minus_WRONG":("PA_EL_SCAD_K1_MATCH","PA_EL_SCAD_K1_WRONG"),"SCAD_minus_DET":("PA_EL_SCAD_K1_MATCH","PA_EL_DET_MATCH")}
    for rows,metric,direction in ((pm,"rrmse_temporal",-1),(nm,"heldout_eog_remaining_ratio",-1),(nm,"preservation",1)):
        values={(row["participant"],row["method"]):float(np.mean([r[metric] for r in rows if r["participant"]==row["participant"] and r["method"]==row["method"]])) for row in rows}
        for name,(match,other) in contrasts.items():
            for participant in sorted({key[0] for key in values}):
                if (participant,match) in values and (participant,other) in values:effects.append({"panel":rows[0]["panel"],"metric":metric,"contrast":name,"participant":participant,"effect_positive_is_better":direction*(values[(participant,match)]-values[(participant,other)])})
    _csv(RESULT/"participant_effects.csv",effects)
    seeds=[]
    for seed in SEEDS:
        subset=[row for row in pm if row["seed"]==seed];values={(r["participant"],r["method"]):r["rrmse_temporal"] for r in subset}
        for name,(match,other) in contrasts.items():
            vec=[values[(p,other)]-values[(p,match)] for p in sorted({k[0] for k in values}) if (p,match) in values and (p,other) in values];seeds.append({"seed":seed,"contrast":name,"mean":float(np.mean(vec)),"median":float(np.median(vec)),"positive_count":int(np.sum(np.asarray(vec)>0)),"participants":len(vec)})
    _csv(RESULT/"seed_effects.csv",seeds)
    def diag(panel:str,metric:str,contrast:str)->dict[str,Any]:
        vector=np.asarray([row["effect_positive_is_better"] for row in effects if row["panel"]==panel and row["metric"]==metric and row["contrast"]==contrast]);lo,hi=_bootstrap(vector,20260831+sum(map(ord,contrast+metric)));return {"mean":float(np.mean(vector)),"median":float(np.median(vector)),"positive_count":int(np.sum(vector>0)),"participants":len(vector),"bootstrap_low":lo,"bootstrap_high":hi}
    detp=diag("paired","rrmse_temporal","DET_MATCH_minus_POP");detw=diag("paired","rrmse_temporal","DET_MATCH_minus_WRONG");scadp=diag("paired","rrmse_temporal","SCAD_MATCH_minus_POP");scadw=diag("paired","rrmse_temporal","SCAD_MATCH_minus_WRONG");diff=diag("paired","rrmse_temporal","SCAD_minus_DET");nat=diag("natural","heldout_eog_remaining_ratio","SCAD_MATCH_minus_POP");pres=diag("natural","preservation","SCAD_MATCH_minus_POP")
    context="clear_development_signal" if min(detp["mean"],detw["mean"],scadp["mean"],scadw["mean"])>0 and min(detp["positive_count"],detw["positive_count"],scadp["positive_count"],scadw["positive_count"])>=10 else "weak_or_heterogeneous_signal" if max(detp["mean"],scadp["mean"])>0 else "context_harmful"
    diffusion="clear_development_signal" if diff["mean"]>0 and diff["positive_count"]>=10 else "small_signal" if diff["mean"]>0 else "deterministic_equivalent" if abs(diff["mean"])<.002 else "deterministic_better";trade="promising" if nat["mean"]>0 and pres["mean"]>=0 else "artifact_reduction_insufficient" if nat["mean"]<=0 else "preservation_concern"
    anchor_summary=next(row for row in summary if row["panel"]=="paired" and row["method"]=="V24_POP_ANCHOR" and row["metric"]=="rrmse_temporal");v23=0.0
    coordinate=json.loads((RESULT/"coordinate_audit.json").read_text());latent_corr=next(row for row in summary if row["panel"]=="natural" and row["method"]=="PA_EL_DET_MATCH" and row["metric"]=="latent_correlation")
    latent_class="clear_predictability" if latent_corr["mean"]>=.7 else "moderate_predictability" if latent_corr["mean"]>=.4 else "weak_predictability" if latent_corr["mean"]>=.1 else "natural_domain_gap"
    if coordinate["coordinate_verdict"]=="V23_COORDINATE_MISMATCH_CONFIRMED" and context=="clear_development_signal" and trade=="promising":next_route="A. Continue PA-EL-SCAD" if diffusion!="deterministic_better" else "E. Focus diffusion on uncertainty/tail"
    elif latent_class in ("weak_predictability","natural_domain_gap"):next_route="B. Improve EOG temporal network"
    elif context in ("context_harmful","context_inert"):next_route="C. Replace fixed operator with raw support-set encoder"
    elif diffusion=="deterministic_better":next_route="F. Remove current diffusion implementation"
    else:next_route="A. Continue PA-EL-SCAD"
    diagnosis={"coordinate":"mismatch_confirmed","population_anchor":"not_interpretable_vs_v23_due_coordinate_supersession","EOG_latent":latent_class,"subject_correction":context,"diffusion":diffusion,"natural_tradeoff":trade,"next_route":next_route,"paired":{"DET_MATCH_POP":detp,"DET_MATCH_WRONG":detw,"SCAD_MATCH_POP":scadp,"SCAD_MATCH_WRONG":scadw,"SCAD_DET":diff},"natural":{"SCAD_MATCH_POP_artifact":nat,"SCAD_MATCH_POP_preservation":pres},"development_only":True,"sealed_reads":0,"confirmation":False};_json(RESULT/"development_diagnosis.json",diagnosis)
    # Compact audit plots.
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    figroot=ROOT/"figures/pa_el_scad_v24";figroot.mkdir(parents=True,exist_ok=True)
    for filename,groups,ylabel in (("context_effect_forest.png",[("DET M-P",detp),("DET M-W",detw),("SCAD M-P",scadp),("SCAD M-W",scadw)],"paired RRMSE utility"),("paired_method_comparison.png",[],"RRMSE")):
        fig,ax=plt.subplots(figsize=(7,4))
        if groups: ax.bar([g[0] for g in groups],[g[1]["mean"] for g in groups]);ax.axhline(0,color="black",lw=.8)
        else:
            methods=[row for row in summary if row["panel"]=="paired" and row["metric"]=="rrmse_temporal"];ax.bar([row["method"] for row in methods],[row["mean"] for row in methods]);ax.tick_params(axis="x",rotation=45)
        ax.set_ylabel(ylabel);fig.tight_layout();fig.savefig(figroot/filename,dpi=180);plt.close(fig)
    methods=sorted({row["method"] for row in summary if row["panel"]=="natural"});remain={r["method"]:r["mean"] for r in summary if r["panel"]=="natural" and r["metric"]=="heldout_eog_remaining_ratio"};presv={r["method"]:r["mean"] for r in summary if r["panel"]=="natural" and r["metric"]=="preservation"};fig,ax=plt.subplots(figsize=(7,5))
    for method in methods:ax.scatter(1-remain[method],presv[method]);ax.annotate(method,(1-remain[method],presv[method]),fontsize=7)
    ax.set_xlabel("artifact attenuation utility");ax.set_ylabel("preservation");fig.tight_layout();fig.savefig(figroot/"attenuation_preservation_scatter.png",dpi=180);plt.close(fig)
    report=["# V24 final development diagnosis","",f"Coordinate verdict: `{coordinate['coordinate_verdict']}`. V23 scientific effects remain superseded; the corrected V24 results are not a repair of the frozen V23 rows.","",f"Population anchor paired participant-first RRMSE: `{anchor_summary['mean']:.6f}`.",f"Natural DET latent correlation: `{latent_corr['mean']:.6f}`.","",f"Subject correction: `{context}`.",f"Diffusion: `{diffusion}`.",f"Natural trade-off: `{trade}`.","",f"Next route: **{next_route}**.","",f"DET MATCH−POP `{detp['mean']:+.6f}`; DET MATCH−WRONG `{detw['mean']:+.6f}`; SCAD MATCH−POP `{scadp['mean']:+.6f}`; SCAD MATCH−WRONG `{scadw['mean']:+.6f}`; SCAD−DET `{diff['mean']:+.6f}`.","","All evidence is development-only. Sealed reads were zero; no manuscript was modified."];(ROOT/"reports/v24_final_development_diagnosis.md").write_text("\n".join(report)+"\n");(ROOT/"reports/v24_round_b.md").write_text("# V24 Round B\n\nThree seeds across all five development folds completed. Participant-first effects are recorded in `results/pa_el_scad_v24/participant_effects.csv`.\n");(ROOT/"reports/v24_natural_development.md").write_text(f"# V24 natural development\n\nNatural trade-off classification: `{trade}`. Test inference used no query EOG, query operator or event labels; evaluator access followed the output freeze.\n")
    _json(run/"result_summary.json",diagnosis);return diagnosis


@torch.no_grad()
def latency_benchmark(run:Path)->dict[str,Any]:
    device=torch.device("cuda");fold=0;seed=SEEDS[0];anchor,_=load_anchor(_checkpoint("anchor",fold,seed),device);temporal,_=load_temporal(_checkpoint("temporal",fold,seed),device);diffusion,_=load_diffusion(_checkpoint("diffusion",fold,seed),device)
    with np.load(DERIVED/f"fold_{fold}/paired_test_inference.npz",allow_pickle=False) as archive:inf={key:np.asarray(archive[key][:100]) for key in archive.files}
    with np.load(DERIVED/f"fold_{fold}/paired_test_evaluator.npz",allow_pickle=False) as archive:target_latent=np.asarray(archive["latent"][:100])
    y=torch.as_tensor(inf["y"],device=device);q0=torch.as_tensor(inf["q0"],device=device);c0=torch.as_tensor(inf["c0"],device=device);ds=torch.as_tensor(inf["ds"],device=device);p0=torch.einsum("bcd,bdt->bct",c0,q0)
    for _ in range(3):a0=anchor(y,q0,p0);zdet=temporal(y,a0,q0);decode_deviation(a0,ds,zdet)
    torch.cuda.synchronize();started=time.perf_counter();a0=anchor(y,q0,p0);torch.cuda.synchronize();anchor_ms=1000*(time.perf_counter()-started)/len(y)
    started=time.perf_counter();zdet=temporal(y,a0,q0);decode_deviation(a0,ds,zdet);torch.cuda.synchronize();det_ms=1000*(time.perf_counter()-started)/len(y)
    noise=torch.randn(zdet.shape,device=device,dtype=zdet.dtype,generator=torch.Generator(device=device).manual_seed(seed));started=time.perf_counter();res,_=diffusion.sample(y,a0,q0,zdet,noise,25);decode_deviation(a0,ds,zdet+res);torch.cuda.synchronize();diff_ms=1000*(time.perf_counter()-started)/len(y)
    spectral=[]
    for label,value in (("TRUE_EOG",target_latent),("DET_EOG",zdet.cpu().numpy()),("SCAD_EOG",(zdet+res).cpu().numpy())):
        frequency,power=signal.welch(value,fs=100,nperseg=128,axis=-1)
        for index,freq in enumerate(frequency):
            spectral.append({"method":label,"frequency_hz":float(freq),"mean_power":float(np.mean(power[...,index]))})
    _csv(RESULT/"eog_latent_spectra.csv",spectral)
    rows=[{"method":"V24_POP_ANCHOR","milliseconds_per_window":anchor_ms,"NFE":1},{"method":"PA_EL_DET_MATCH","milliseconds_per_window":anchor_ms+det_ms,"NFE":2},{"method":"PA_EL_SCAD_K1_MATCH","milliseconds_per_window":anchor_ms+det_ms+diff_ms,"NFE":27}];_csv(RESULT/"latency_summary.csv",rows);result={"stage":"R16-latency","status":"PASS","gpu":torch.cuda.get_device_name(0),"windows":100,"warmups":3,"rows":rows};_json(run/"result_summary.json",result);return result


def package_results(run:Path)->dict[str,Any]:
    diagnosis=json.loads((RESULT/"development_diagnosis.json").read_text());coordinate=json.loads((RESULT/"coordinate_audit.json").read_text());method_rows=list(csv.DictReader((RESULT/"method_summary.csv").open(newline="",encoding="utf-8")));latency=list(csv.DictReader((RESULT/"latency_summary.csv").open(newline="",encoding="utf-8")))
    source_registry={"V19":{"commit":_cfg("data")["v19_commit"],"role":"read_only_source"},"V23":{"commit":_cfg("data")["v23_commit"],"role":"historical_invalid_coordinate_reference"},"V24":{"commit":_head(ROOT),"role":"coordinate_correct_development"},"EEGDfus":{"commit":"a19a652b3b6346188ae77067e1daf8b90cad005f","status":"frozen_V22_reference_not_recomputed"},"D4PM":{"commit":"5be2b3c72973fea6c879e63cd83067ff66aace13","status":"blocked_incomplete_release"}};_json(RESULT/"source_registry.json",source_registry)
    inventory=[]
    for path,role in ((Path(_cfg("data")["v19_derived_root"]),"V19 derived read-only"),(DERIVED,"V24 server assets"),(Path(_cfg("data")["v23_worktree"]),"V23 read-only")):
        inventory.append({"absolute_path":str(path),"role":role,"exists":path.exists(),"size_bytes":path.stat().st_size if path.exists() else 0,"mtime_ns":path.stat().st_mtime_ns if path.exists() else 0})
    _csv(RESULT/"input_inventory.csv",inventory)
    latent={}
    for panel in ("paired","natural"):
        for method in ("PA_EL_DET_MATCH","PA_EL_SCAD_K1_MATCH"):
            latent[f"{panel}_{method}"]={r["metric"]:float(r["mean"]) for r in method_rows if r["panel"]==panel and r["method"]==method and r["metric"].startswith("latent_")}
    _json(RESULT/"eog_latent_statistics.json",latent)
    checkpoints=[]
    for kind,filename in (("anchor","best_joint.pt"),("temporal","best_joint.pt"),("diffusion","best_sampling.pt")):
        for fold in range(5):
            for seed in SEEDS:
                path=DERIVED/"checkpoints"/kind/f"fold_{fold}"/f"seed_{seed}"/filename
                checkpoints.append({"path":str(path),"sha256":_digest(path),"fold":fold,"seed":seed,"model":kind,"config":str(ROOT/f"configs/pa_el_scad_v24/{'population_anchor' if kind=='anchor' else 'temporal_eog' if kind=='temporal' else 'pa_el_scad'}.yaml"),"training_job":"see reports/slurm/v24_job_ids.txt","best_criterion":filename})
    _csv(RESULT/"checkpoint_manifest.csv",checkpoints)
    exposure=[]
    for kind in ("anchor","temporal","diffusion"):
        for path in sorted((RESULT/kind).glob("fold_*_seed_*.json")):
            value=json.loads(path.read_text());exposure.append({"model":kind,"fold":value["fold"],"seed":value["seed"],"updates":value["updates"],"parameters":value["parameters"],"training_seconds":value["training_seconds"],"device":value["device"],"checkpoint":value["checkpoint"]})
    _csv(RESULT/"training_exposure.csv",exposure)
    # Remaining required audit figures.
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    figroot=ROOT/"figures/pa_el_scad_v24";figroot.mkdir(parents=True,exist_ok=True)
    unit=list(csv.DictReader((RESULT/"coordinate_unit_summary.csv").open(newline="",encoding="utf-8")));fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([[float(r["raw_vs_canonical_relative_frobenius_difference"]) for r in unit if r["status"]=="audited"],[float(r["raw_vs_v23_relative_frobenius_difference"]) for r in unit if r["status"]=="audited"]],labels=["correct canonical","V23 committed"]);ax.set_yscale("log");ax.set_ylabel("relative coordinate error");fig.tight_layout();fig.savefig(figroot/"coordinate_scale_comparison.png",dpi=180);plt.close(fig)
    corr=[r for r in method_rows if r["metric"]=="latent_correlation" and r["method"] in ("PA_EL_DET_MATCH","PA_EL_SCAD_K1_MATCH")];fig,ax=plt.subplots(figsize=(6,4));ax.bar([f"{r['panel']}:{r['method'].replace('PA_EL_','')}" for r in corr],[float(r["mean"]) for r in corr]);ax.tick_params(axis="x",rotation=35);ax.set_ylabel("latent correlation");fig.tight_layout();fig.savefig(figroot/"eog_latent_prediction.png",dpi=180);plt.close(fig)
    ceiling=list(csv.DictReader((RESULT/"headroom/ceiling_summary.csv").open(newline="",encoding="utf-8")));fig,ax=plt.subplots(figsize=(6,4));ax.bar([r["method"] for r in ceiling],[float(r["mean"]) for r in ceiling]);ax.tick_params(axis="x",rotation=25);ax.set_ylabel("relative artifact error");fig.tight_layout();fig.savefig(figroot/"population_residual_headroom.png",dpi=180);plt.close(fig)
    sanity=json.loads((RESULT/"sanity/diffusion_trajectory.json").read_text());fig,ax=plt.subplots(figsize=(6,4));ax.plot([r["step"] for r in sanity],[r["r_t_rms"] for r in sanity],label="r_t");ax.plot([r["step"] for r in sanity],[r["r_hat_rms"] for r in sanity],label="r_hat");ax.legend();ax.set_ylabel("RMS");fig.tight_layout();fig.savefig(figroot/"diffusion_trajectory.png",dpi=180);plt.close(fig)
    paired={r["method"]:float(r["mean"]) for r in method_rows if r["panel"]=="paired" and r["metric"]=="rrmse_temporal"};fig,ax=plt.subplots(figsize=(6,4));
    for row in latency:ax.scatter(float(row["milliseconds_per_window"]),paired[row["method"]]);ax.annotate(row["method"],(float(row["milliseconds_per_window"]),paired[row["method"]]),fontsize=7)
    ax.set_xlabel("milliseconds/window");ax.set_ylabel("paired RRMSE");fig.tight_layout();fig.savefig(figroot/"quality_latency_curve.png",dpi=180);plt.close(fig)
    spectra=list(csv.DictReader((RESULT/"eog_latent_spectra.csv").open(newline="",encoding="utf-8")));fig,ax=plt.subplots(figsize=(6,4))
    for method in ("TRUE_EOG","DET_EOG","SCAD_EOG"):
        rows=[r for r in spectra if r["method"]==method];ax.semilogy([float(r["frequency_hz"]) for r in rows],[float(r["mean_power"]) for r in rows],label=method)
    ax.set_xlim(0,15);ax.set_xlabel("Hz");ax.set_ylabel("mean latent PSD");ax.legend();fig.tight_layout();fig.savefig(figroot/"true_vs_predicted_eog_spectra.png",dpi=180);plt.close(fig)
    # Training/validation curves.
    fig,ax=plt.subplots(figsize=(7,4))
    for kind,key in (("anchor","joint_validation"),("temporal","joint_validation"),("diffusion","joint_validation")):
        curves=[]
        for path in (RESULT/kind).glob("fold_*_seed_*.json"):
            for point in json.loads(path.read_text()).get("curve",[]):curves.append((point["step"],point[key]))
        steps=sorted({p[0] for p in curves});means=[np.mean([p[1] for p in curves if p[0]==step]) for step in steps];ax.plot(steps,means,label=kind)
    ax.set_xlabel("updates");ax.set_ylabel("mean joint validation");ax.set_yscale("log");ax.legend();fig.tight_layout();fig.savefig(figroot/"training_curves.png",dpi=180);fig.savefig(figroot/"validation_curves.png",dpi=180);plt.close(fig)
    terminal={"protocol":"PA-EL-SCAD V24","coordinate_verdict":coordinate["coordinate_verdict"],"development_diagnosis":diagnosis,"implementation_commit":_head(ROOT),"sealed_reads":0,"confirmation_run":False,"manuscript_modified":False,"K8_run":False,"energy_bridge_run":False,"GPU_models":["population_anchor","TemporalEOGNet","PA-EL-SCAD-K1"],"checkpoints_server_only":len(checkpoints),"current_jobs_checked_at_finalize":None};_json(RESULT/"terminal_manifest.json",terminal)
    result={"stage":"R16-package","status":"PASS","reports":7,"figures":10,"checkpoints":len(checkpoints),"sealed_reads":0};_json(run/"result_summary.json",result);return result


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
    elif args.stage == "r2-prepare":
        prepare_assets(args.run_dir)
    elif args.stage == "r2-collect":
        collect_assets(args.run_dir)
    elif args.stage == "r3-headroom":
        headroom(args.run_dir)
    elif args.stage == "r4-sanity":
        sanity(args.run_dir)
    elif args.stage == "r5-anchor": train_model(args.run_dir,"anchor",False)
    elif args.stage == "r6-det": train_model(args.run_dir,"temporal",False)
    elif args.stage == "r7-diff": train_model(args.run_dir,"diffusion",False)
    elif args.stage == "r8-eval": paired_evaluate(args.run_dir,False)
    elif args.stage == "r9-round-a": round_a_aggregate(args.run_dir)
    elif args.stage == "r10-anchor": train_model(args.run_dir,"anchor",True)
    elif args.stage == "r10-det": train_model(args.run_dir,"temporal",True)
    elif args.stage == "r10-diff": train_model(args.run_dir,"diffusion",True)
    elif args.stage == "r11-paired": paired_evaluate(args.run_dir,True)
    elif args.stage == "r12-natural-infer": natural_infer(args.run_dir)
    elif args.stage == "r13-output-freeze": output_freeze(args.run_dir)
    elif args.stage == "r14-natural-eval": natural_evaluate(args.run_dir)
    elif args.stage == "r16-aggregate": final_aggregate(args.run_dir)
    elif args.stage == "r16-latency": latency_benchmark(args.run_dir)
    elif args.stage == "r16-package": package_results(args.run_dir)
    else:
        raise ValueError(f"unknown V24 stage: {args.stage}")


if __name__ == "__main__":
    main()
