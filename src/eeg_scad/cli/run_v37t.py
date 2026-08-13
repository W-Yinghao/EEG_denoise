"""Frozen V37T waveform consolidation and K=16 uncertainty workflow."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from eeg_scad.cli import run_v30 as v30
from eeg_scad.cli import v26
from eeg_scad.context.learned_spatial_decoder import decode_residual
from eeg_scad.energy.partial_observation import partial_observation_prox
from eeg_scad.energy.projector import projector
from eeg_scad.energy.temporal_confidence import temporal_confidence
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.uncertainty_v37t import (
    constant_width_interval, ensemble_crps, error_dispersion, interval_metrics,
    participant_mean, projected_variance,
)


ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", Path(__file__).resolve().parents[3]))
RESULT = ROOT / "results/taas_waveform_v37t"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/taas_waveform_v37t")
V30_ROOT = Path("/home/infres/yinwang/denoiseNet_frozen_candidate_v30/results/frozen_candidate_v30")
V31_ROOT = Path("/home/infres/yinwang/denoiseNet_claim_narrowing_v31/results/claim_narrowing_v31")
V27_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_energy_v27")
BASE = "a90cabf5ed7167e0bc6cfc01257e74592b6e7d85"


def _cfg() -> dict[str, Any]: return yaml.safe_load((ROOT / "configs/taas_waveform_v37t.yaml").read_text())
def _index() -> int: return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()
def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def preflight() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ledger = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    checks = {
        "base_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=ROOT).returncode == 0,
        "base_ref_exact": subprocess.check_output(["git", "rev-parse", "codex/fiber-openbmi-v36p"], cwd=ROOT, text=True).strip() == BASE,
        "ledger_v3_7": "**版本：** v3.7" in ledger,
        "scope_split": "V37T" in ledger and "Fiber-Gaussian" in ledger,
        "v25_v31_read_only": not bool(subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "results", "reports/v2*", "reports/v3*"], cwd=ROOT, text=True).strip()),
        "manuscript_unchanged": not bool(subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "taas_submission"], cwd=ROOT, text=True).strip()),
        "sealed_reads": 0,
    }
    if not all(value is True for key, value in checks.items() if key != "sealed_reads"): raise RuntimeError(checks)
    result = {"stage": "R0", "status": "PASS", "base_commit": BASE, "head": head, **checks}
    _json(RESULT / "source_registry.json", result); return result


def inventory_common() -> dict[str, Any]:
    required = {"V25", "V26", "V27", "V22"}; inventory = list(csv.DictReader((V30_ROOT / "checkpoint_inventory.csv").open()))
    selected = [row for row in inventory if row["source"] in required and (row["source"] != "V22" or row["model"] == "EEGDFUS")]
    for row in selected:
        path = Path(row["checkpoint_path"])
        if row["sha256"] not in ("closed_form_no_checkpoint", ""):
            if not path.is_file(): row["v37t_binding"] = "missing_no_substitution"
            else: row["v37t_binding"] = "verified" if _digest(path) == row["sha256"] else "digest_mismatch"
        else: row["v37t_binding"] = "closed_form_or_historical"
    if any(row["v37t_binding"] == "digest_mismatch" for row in selected): raise RuntimeError("frozen checkpoint digest mismatch")
    _csv(RESULT / "checkpoint_binding.csv", selected)
    methods = {"RAW", "STANDARD", "V25_SET_CALIB_DET_MATCH", "V25_SET_CALIB_DET_POP", "V26_CALIB_SDEDIT_MATCH", "V26_POP_SDEDIT", "V27_ENERGY_SDEDIT_L05", "V27_ENERGY_SDEDIT_L2", "V27_ENERGY_SDEDIT_L8", "V27_ENERGY_DET_L05", "V27_ENERGY_DET_L2", "V27_ENERGY_DET_L8", "EEGDFUS"}
    paired = [row for row in csv.DictReader((V30_ROOT / "paired_method_summary.csv").open()) if row["method"] in methods]
    natural = [row for row in csv.DictReader((V30_ROOT / "natural_method_summary.csv").open()) if row["method"] in methods]
    for row in paired + natural: row["source"] = "V30_common_panel_220dcbaa"; row["absolute_or_contrast"] = "absolute"
    # No compatible strong U-Net checkpoint was registered in V30; absence is explicit.
    paired.append({"method": "STRONG_UNET", "metric": "all", "mean": "", "status": "not_comparable_missing_registered_checkpoint", "source": "V30 inventory", "absolute_or_contrast": "absolute"})
    natural.append({"method": "STRONG_UNET", "metric": "all", "mean": "", "status": "not_comparable_missing_registered_checkpoint", "source": "V30 inventory", "absolute_or_contrast": "absolute"})
    _csv(RESULT / "common_paired_summary.csv", paired); _csv(RESULT / "common_natural_summary.csv", natural)
    support = []
    support.extend({"evidence": "all_donor", **row} for row in csv.DictReader((V30_ROOT / "all_donor_group_summary.csv").open()) if row["method"] in ("V25_SET_CALIB_DET_MATCH", "V26_CALIB_SDEDIT_MATCH"))
    support.extend({"evidence": "falsification", **row} for row in csv.DictReader((V30_ROOT / "falsification_effects.csv").open()) if row["method"] in ("V25_SET_CALIB_DET", "V26_CALIB_SDEDIT"))
    support.extend({"evidence": "exact_duration_v31", **row} for row in csv.DictReader((V31_ROOT / "support_duration_repair.csv").open()) if row["method"] in ("V25_SET_CALIB_DET", "V26_CALIB_SDEDIT"))
    _csv(RESULT / "support_evidence.csv", support)
    pareto_metrics = {"remaining_ratio", "attenuation_db", "low_eog_observation_retention", "psd_distortion", "covariance_distortion"}
    pareto = [row for row in natural if row.get("method", "").startswith("V27_ENERGY_SDEDIT_L") and row.get("metric") in pareto_metrics]
    _csv(RESULT / "natural_pareto.csv", pareto)
    result = {"stage": "R1", "status": "PASS", "checkpoint_rows": len(selected), "paired_rows": len(paired), "natural_rows": len(natural), "support_rows": len(support), "primary_operating_point": "V27_ENERGY_SDEDIT_L05", "sealed_reads": 0}
    _json(RESULT / "inventory_summary.json", result); return result


@torch.no_grad()
def stochastic_cell(index: int | None = None) -> dict[str, Any]:
    cfg = _cfg(); index = _index() if index is None else index; fold, slot = index // 3, index % 3
    seed = int(cfg["v26_seeds"][slot]); device = torch.device("cuda")
    batch = v30._panel(fold, "paired", False, 120); anchor, det, models = v26._load_bundle(fold, seed, device)
    all_samples = {name: [] for name in ("V26_CALIB_SDEDIT_MATCH", "V26_POP_SDEDIT", "V27_ENERGY_SDEDIT_L05")}
    deterministic, projectors = [], []
    with np.load(V27_DERIVED / f"calibration/fold_{fold}_seed_{seed}.npz", allow_pickle=False) as archive:
        q50, q90 = float(archive["q50"]), float(archive["q90"])
    started = time.time()
    for start in range(0, len(batch["y"]), 16):
        sl = slice(start, min(start+16, len(batch["y"]))); y = torch.as_tensor(batch["y"][sl], device=device); q0 = torch.as_tensor(batch["q0"][sl], device=device); c0 = torch.as_tensor(batch["c0"][sl], device=device)
        pop = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0))
        encoded = det.encode_support(torch.as_tensor(batch["support_eeg"][sl], device=device), torch.as_tensor(batch["support_eog"][sl], device=device))
        coefficient = det.coefficient(y, pop, q0, encoded["context"]); det_artifact = decode_residual(pop, encoded["basis"], coefficient)
        refine = models["calib_refine_det"](y, det_artifact, pop, encoded["context"])
        pi = projector(encoded["basis"]); mask = temporal_confidence(det_artifact, pop, q50, q90, 10)
        deterministic.append((y-refine).cpu().numpy()); projectors.append(pi.cpu().numpy())
        chunk = {name: [] for name in all_samples}
        for draw in range(int(cfg["uncertainty_k"])):
            noise_seed = int(cfg["noise_seed"]) + fold*100000 + slot*10000 + draw*100 + start
            noise = torch.randn(pop.shape, device=device, generator=torch.Generator(device=device).manual_seed(noise_seed))
            match, _ = models["calib_sdedit"].sample(y, det_artifact, pop, encoded["context"], noise, float(cfg["sigma_start"]), int(cfg["ddim_steps"]))
            pop_sample, _ = models["pop_sdedit"].sample(y, pop, noise, float(cfg["sigma_start"]), int(cfg["ddim_steps"]))
            energy = partial_observation_prox(match, refine, pi, mask, float(cfg["lambda_a"]), float(cfg["lambda_y"]))
            chunk["V26_CALIB_SDEDIT_MATCH"].append((y-match).cpu().numpy())
            chunk["V26_POP_SDEDIT"].append((y-pop_sample).cpu().numpy())
            chunk["V27_ENERGY_SDEDIT_L05"].append((y-energy).cpu().numpy())
        for name in all_samples: all_samples[name].append(np.stack(chunk[name], axis=0))
    arrays = {name: np.concatenate(parts, axis=1).astype(np.float32) for name, parts in all_samples.items()}
    arrays["MATCHED_DETERMINISTIC"] = np.concatenate(deterministic).astype(np.float32)
    arrays["support_projector"] = np.concatenate(projectors).astype(np.float32)
    path = DERIVED / f"stochastic/fold_{fold}_slot_{slot}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **arrays)
    record = {"fold": fold, "slot": slot, "seed": seed, "path": str(path), "sha256": _digest(path), "queries": len(batch["y"]), "K": int(cfg["uncertainty_k"]), "target_selected_samples": 0, "noise_seed_rule": "37000000+fold*100000+slot*10000+draw*100+chunk_start", "seconds": time.time()-started, "query_auxiliary_reads": 0, "sealed_reads": 0}
    _json(RESULT / f"stochastic/cell_{fold}_{slot}.json", record); return record


def aggregate_uncertainty() -> dict[str, Any]:
    cfg = _cfg(); rows = []; point_rows = []; manifest = []
    for fold in range(5):
        batch = v30._panel(fold, "paired", False, 120); evaluator = v30._panel(fold, "paired", True); target_all = np.asarray(evaluator["x"])
        participants = np.asarray([row["participant"] for row in batch["meta"]]); identity = np.asarray([row.get("zero_artifact") == "1" for row in batch["meta"]])
        for slot in range(3):
            record = json.loads((RESULT / f"stochastic/cell_{fold}_{slot}.json").read_text()); path = Path(record["path"])
            if _digest(path) != record["sha256"]: raise RuntimeError(f"stochastic digest mismatch: {path}")
            manifest.append(record)
            with np.load(path, allow_pickle=False) as archive:
                deterministic = np.asarray(archive["MATCHED_DETERMINISTIC"]); pis = np.asarray(archive["support_projector"])
                with np.load(v30.DERIVED / f"common/paired/fold_{fold}_slot_{slot}.npz", allow_pickle=False) as common:
                    det_artifact = np.asarray(batch["y"]) - np.asarray(common["V25_SET_CALIB_DET_MATCH"])
                    pop_artifact = np.asarray(batch["y"]) - np.asarray(common["V25_SET_CALIB_DET_POP"])
                with np.load(V27_DERIVED / f"calibration/fold_{fold}_seed_{int(cfg['v26_seeds'][slot])}.npz", allow_pickle=False) as calibration:
                    q50, q90 = float(calibration["q50"]), float(calibration["q90"])
                mask = temporal_confidence(torch.as_tensor(det_artifact), torch.as_tensor(pop_artifact), q50, q90, 10)
                refine_artifact = torch.as_tensor(np.asarray(batch["y"])-deterministic)
                energy_det = np.asarray(batch["y"]) - partial_observation_prox(refine_artifact, refine_artifact, torch.as_tensor(pis), mask, 1.0, .5).numpy()
                for method in ("V26_CALIB_SDEDIT_MATCH", "V26_POP_SDEDIT", "V27_ENERGY_SDEDIT_L05"):
                    samples = np.asarray(archive[method])
                    for participant in sorted(set(participants)):
                        chosen = participants == participant; current, target, pi = samples[:, chosen], target_all[chosen], pis[chosen]
                        mean = np.mean(current, axis=0); median = np.median(current, axis=0)
                        summaries = [("single_draw", current[0]), ("sample_mean", mean), ("sample_median", median)]
                        if method == "V26_CALIB_SDEDIT_MATCH": summaries.append(("matched_deterministic", deterministic[chosen]))
                        if method == "V27_ENERGY_SDEDIT_L05": summaries.append(("matched_deterministic", energy_det[chosen]))
                        for summary, prediction in summaries:
                            metrics = [paired_metrics(target[i], np.asarray(batch["y"])[chosen][i], np.asarray(evaluator["artifact"])[chosen][i], np.asarray(batch["y"])[chosen][i]-prediction[i]) for i in range(len(prediction))]
                            point_rows.append({"fold": fold, "slot": slot, "participant": participant, "method": method, "summary": summary, "rrmse_temporal": float(np.mean([m["rrmse_temporal"] for m in metrics])), "rrmse_spectral": float(np.mean([m["rrmse_spectral"] for m in metrics])), "correlation": float(np.mean([m["correlation"] for m in metrics]))})
                        parallel, complement = projected_variance(current, pi)
                        base = {"fold": fold, "slot": slot, "participant": participant, "method": method, "K": len(current), "sample_variance": float(np.mean(np.var(current, axis=0, ddof=1))), "crps": ensemble_crps(current, target), "error_dispersion_spearman": error_dispersion(current, target), "parallel_variance": parallel, "complement_variance": complement, "identity_variance": float(np.mean(np.var(current[:, identity[chosen]], axis=0, ddof=1))) if np.any(identity[chosen]) else float("nan")}
                        for level in (.5, .8, .9):
                            empirical = interval_metrics(current, target, level); constant = constant_width_interval(mean, current, target, level)
                            rows.append({**base, "level": level, "reference": "ensemble_quantiles", **empirical})
                            rows.append({**base, "level": level, "reference": "matched_mean_constant_variance", **constant, "error_dispersion_spearman": 0.0})
    _csv(RESULT / "stochastic_samples_manifest.csv", manifest); _csv(DERIVED / "uncertainty_cell_rows.csv", rows); _csv(DERIVED / "point_summary_rows.csv", point_rows)
    participant = participant_mean(rows, ("participant", "method", "level", "reference")); _csv(RESULT / "uncertainty_summary.csv", participant)
    points = participant_mean(point_rows, ("participant", "method", "summary")); _csv(RESULT / "participant_effects.csv", points)
    result = {"stage": "R3", "status": "PASS", "K": int(cfg["uncertainty_k"]), "cells": len(manifest), "participants": len(set(row["participant"] for row in participant)), "all_samples_aggregated": True, "target_selected_samples": 0, "uncertainty_rows": len(participant), "point_rows": len(points), "sealed_reads": 0}
    _json(RESULT / "uncertainty_aggregate.json", result); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("preflight", "inventory", "stochastic", "aggregate")); parser.add_argument("--index", type=int); args = parser.parse_args()
    value = preflight() if args.stage == "preflight" else inventory_common() if args.stage == "inventory" else stochastic_cell(args.index) if args.stage == "stochastic" else aggregate_uncertainty()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__": main()
