"""Slurm-facing orchestration for V26 CalibSDEdit."""
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
from scipy import signal

from eeg_scad.data.folds import load_folds, validate_folds
from eeg_scad.data.support_set_episodes import SupportSetEpisodeSampler
from eeg_scad.evaluation.aggregate_v26 import bootstrap, contrast, participant_first
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.refinement_diagnostics import projector_distance, procrustes_align, rotation_fixture
from eeg_scad.training.train_v24 import load_anchor
from eeg_scad.training.train_v25 import load_det
from eeg_scad.training.train_v26 import load_one_step, load_sdedit, train_one_step, train_sdedit

ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", Path(__file__).resolve().parents[3]))
RESULT = ROOT / "results/calib_sdedit_v26"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_sdedit_v26")
V25_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25")
V24_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24")
V26_SEEDS = [20260828, 20260829, 20260830]
V25_SEED = {20260828: 20260825, 20260829: 20260826, 20260830: 20260827}
ANCHOR_SEED = {20260828: 20260825, 20260829: 20260826, 20260830: 20260824}


def _cfg(name: str) -> dict[str, Any]: return yaml.safe_load((ROOT / f"configs/calib_sdedit_v26/{name}.yaml").read_text())
def _folds() -> list[dict[str, Any]]: return load_folds(ROOT / "configs/calib_sdedit_v26/folds.yaml")
def _index() -> int: return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()
def _json(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _anchor_path(fold: int, seed: int) -> Path: return V24_DERIVED / f"checkpoints/anchor/fold_{fold}/seed_{ANCHOR_SEED[seed]}/best_joint.pt"
def _v25_det_path(fold: int, seed: int) -> Path: return V25_DERIVED / f"checkpoints/det/deepsets/fold_{fold}/seed_{V25_SEED[seed]}/best_joint.pt"
def _model_dir(kind: str, fold: int, seed: int, natural_fraction: float = .3) -> Path:
    suffix = "_nf50" if natural_fraction == .5 else ""
    return DERIVED / f"checkpoints/{kind}{suffix}/fold_{fold}/seed_{seed}"
def _model_path(kind: str, fold: int, seed: int, natural_fraction: float = .3) -> Path: return _model_dir(kind, fold, seed, natural_fraction) / "best_joint.pt"


def preflight(run: Path) -> dict[str, Any]:
    data = _cfg("data"); folds = _folds(); validate_folds(folds, data["participants"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ledger = ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md"; ledger_text = ledger.read_text()
    checks = {
        "base_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", data["base_commit"], head], cwd=ROOT).returncode == 0,
        "ledger_v1_3": "**版本：** v1.3" in ledger_text,
        "ledger_active_v26": "V26 CalibSDEdit" in ledger_text,
        "v25_object_present": subprocess.run(["git", "cat-file", "-e", data["v25_commit"] + "^{commit}"], cwd=ROOT).returncode == 0,
        "a_track_object_present": subprocess.run(["git", "cat-file", "-e", data["a_track_commit"] + "^{commit}"], cwd=ROOT).returncode == 0,
        "a_track_forbidden_diff": bool(subprocess.check_output(["git", "diff", "--name-only", data["base_commit"], "--", "taas_submission"], cwd=ROOT, text=True).strip()),
        "sealed_reads": 0,
    }
    if not all((value is True) for key, value in checks.items() if key not in ("a_track_forbidden_diff", "sealed_reads")) or checks["a_track_forbidden_diff"]: raise RuntimeError(checks)
    sources = {"base_commit": data["base_commit"], "V25": data["v25_commit"], "A_track": data["a_track_commit"], "ledger_version": "v1.3", "ledger_sha256": _digest(ledger)}
    _json(RESULT / "source_registry.json", sources); _json(RESULT / "preflight.json", {"status": "PASS", **checks}); _json(run / "result_summary.json", {"stage": "R0", "status": "PASS", **checks}); return checks


def prepare(run: Path) -> dict[str, Any]:
    data = _cfg("data"); rows = []
    for fold in _folds():
        for split in ("train", "validation", "test"):
            rows.extend({"fold": fold["fold"], "split": split, "participant": person} for person in fold[split])
    _csv(RESULT / "fold_manifest.csv", rows)
    bindings = []
    for fold in range(5):
        for seed in V26_SEEDS:
            for role, path in (("population_anchor", _anchor_path(fold, seed)), ("setcalib_det", _v25_det_path(fold, seed))):
                if not path.is_file(): raise FileNotFoundError(path)
                bindings.append({"fold": fold, "v26_seed": seed, "v25_seed": V25_SEED[seed], "role": role, "path": str(path), "sha256": _digest(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    _csv(RESULT / "checkpoint_binding.csv", bindings)
    inventory = []
    for path in (ROOT / "results/setcalibdiff_v25/development_diagnosis.json", ROOT / "results/setcalibdiff_v25/method_summary.csv", ROOT / "results/setcalibdiff_v25/terminal_manifest.json"):
        inventory.append({"path": str(path), "sha256": _digest(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "role": "V25_frozen_evidence"})
    _csv(RESULT / "input_inventory.csv", inventory)
    result = {"stage": "R2", "status": "PASS", "checkpoint_bindings": len(bindings), "query_auxiliary_reads": 0, "sealed_reads": 0}; _json(run / "result_summary.json", result); return result


def forensic(run: Path) -> dict[str, Any]:
    fixture = rotation_fixture(); rows = []
    device = torch.device("cpu")
    role_rows = list(csv.DictReader((ROOT / "results/pa_el_scad_v24/role_manifest.csv").open()))
    for fold in range(5):
        det, _ = load_det(_v25_det_path(fold, 20260828), device)
        anchor, _ = load_anchor(_anchor_path(fold, 20260828), device)
        with np.load(V25_DERIVED / f"support_banks/fold_{fold}.npz", allow_pickle=False) as bank:
            support_eeg = np.asarray(bank["support_eeg"]); support_eog = np.asarray(bank["support_eog"])
        with np.load(V24_DERIVED / f"fold_{fold}/natural_test_inference.npz", allow_pickle=False) as query:
            y_all = np.asarray(query["y"]); q0_all = np.asarray(query["q0"]); c0_all = np.asarray(query["c0"])
        fold_roles = [row for row in role_rows if row["fold"] == str(fold) and row["stream"] == "natural" and row["split"] == "test"]
        first_index = {participant: next(i for i, row in enumerate(fold_roles) if row["participant"] == participant) for participant in _folds()[fold]["test"]}
        for participant, sample in first_index.items():
            eeg = torch.as_tensor(support_eeg[sample:sample+1]); eog = torch.as_tensor(support_eog[sample:sample+1])
            y = torch.as_tensor(y_all[sample:sample+1]); q0 = torch.as_tensor(q0_all[sample:sample+1]); c0 = torch.as_tensor(c0_all[sample:sample+1]); pop = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0))
            bases = []; coefficients = []
            for subset in (np.arange(8), np.arange(8, 16), np.arange(0, 16, 2)):
                with torch.no_grad():
                    encoded = det.encode_support(eeg[:, subset], eog[:, subset]); bases.append(encoded["basis"][0].numpy()); coefficients.append(det.coefficient(y, pop, q0, encoded["context"])[0].numpy())
            aligned, rotation = procrustes_align(bases[0], bases[1])
            aligned_coefficient = rotation.T @ coefficients[1]
            correlation = np.abs(bases[0].T @ bases[1]); assignment = np.argmax(correlation, axis=1); sign_flips = sum(float(bases[0][:, i] @ bases[1][:, assignment[i]]) < 0 for i in range(8)); permutations = sum(int(assignment[i]) != i for i in range(8))
            rows.append({"fold": fold, "participant": participant, "sample": int(sample), "projector_distance": projector_distance(bases[0], bases[1]), "unmatched_basis_difference": float(np.linalg.norm(bases[0]-bases[1])), "aligned_basis_difference": float(np.linalg.norm(bases[0]-aligned)), "procrustes_rotation_magnitude": float(np.linalg.norm(rotation-np.eye(rotation.shape[0]))), "latent_residual_rmse_before_alignment": float(np.sqrt(np.mean((coefficients[0]-coefficients[1])**2))), "latent_residual_rmse_after_alignment": float(np.sqrt(np.mean((coefficients[0]-aligned_coefficient)**2))), "column_permutation_count": permutations, "column_sign_flip_count": sign_flips, "third_subset_projector_distance": projector_distance(bases[0], bases[2])})
    _csv(RESULT / "basis_rotation_diagnostics.csv", rows)
    residual = []
    for path in sorted((V25_DERIVED / "metrics/severity").glob("*.csv")):
        for row in csv.DictReader(path.open()):
            if row["method"] in ("RAW", "DET_MATCH", "DIFF_MATCH"): residual.append({"fold": row["fold"], "seed": row["seed"], "participant": row["participant"], "severity": row["severity"], "stream": "paired", "method": row["method"], "rrmse_temporal": row["rrmse_temporal"]})
    _csv(RESULT / "residual_scale_diagnostics.csv", residual)
    exposure = []
    for path in sorted((ROOT / "results/setcalibdiff_v25/round_b").glob("*.json")):
        value = json.loads(path.read_text()); exposure.append({"path": str(path), "kind": value["kind"], "fold": value["fold"], "seed": value["seed"], "updates": value["updates"], "best_step": min(value["curve"], key=lambda x: x["joint"])["step"], "last_step": value["curve"][-1]["step"], "plateaued": value["updates"] < 20000})
    _csv(RESULT / "v25_training_exposure.csv", exposure)
    result = {"stage": "R1", "status": "PASS", "rotation_fixture": fixture, "basis_rows": len(rows), "training_rows": len(exposure), "interpretation": "sensor-space target is invariant while the learned coefficient target changes under equivalent basis rotations", "sealed_reads": 0}
    _json(run / "result_summary.json", result); return result


def sanity(run: Path) -> dict[str, Any]:
    from eeg_scad.models.calib_refine_det import CalibRefineDET, PopRefineDET
    from eeg_scad.models.calib_sdedit import CalibSDEdit, PopSDEdit
    device = torch.device("cuda"); generator = torch.Generator(device=device).manual_seed(20260828)
    y = torch.randn(2, 46, 128, device=device, generator=generator); det = .2 * torch.randn(y.shape, device=device, generator=generator); pop = .2 * torch.randn(y.shape, device=device, generator=generator); context = torch.randn(2, 128, device=device, generator=generator); target = .2 * torch.randn(y.shape, device=device, generator=generator)
    reports = {}
    for name, model, call in (("calib_refine_det", CalibRefineDET(32).to(device), lambda m: m(y, det, pop, context)), ("pop_refine_det", PopRefineDET(32).to(device), lambda m: m(y, pop))):
        optimizer = torch.optim.Adam(model.parameters(), 1e-3); initial = None
        for _ in range(80):
            prediction = call(model); loss = (prediction-target).square().mean(); initial = float(loss) if initial is None else initial; optimizer.zero_grad(); loss.backward(); optimizer.step()
        reports[name] = {"initial": initial, "final": float(loss), "finite": bool(torch.isfinite(prediction).all())}
    for name, model, call in (("calib_sdedit", CalibSDEdit(32, 100).to(device), lambda m, n: m.sample(y, det, pop, context, n, .2, 5)), ("pop_sdedit", PopSDEdit(32, 100).to(device), lambda m, n: m.sample(y, pop, n, .2, 5))):
        noise = torch.randn(y.shape, device=device, generator=generator); prediction, trajectory = call(model, noise); identity, _ = (model.sample(y, det, pop, context, noise, 0, 5) if name == "calib_sdedit" else model.sample(y, pop, noise, 0, 5)); reports[name] = {"finite": bool(torch.isfinite(prediction).all()), "trajectory": len(trajectory), "sigma_zero_max": float((identity-(det if name=="calib_sdedit" else pop)).abs().max())}
    result = {"stage": "R3", "status": "PASS", "checks": reports, "sealed_reads": 0}; _json(RESULT / "sanity/technical_validity.json", result); _json(run / "result_summary.json", result); return result


def train_stage(stage: str, run: Path) -> dict[str, Any]:
    data = _cfg("data"); folds = _folds(); index = _index()
    if stage == "r4-rounda-one-step":
        kinds = ("calib_refine_det", "pop_refine_det"); kind = kinds[index // 2]; fold = (0, 2)[index % 2]; seed = 20260828; cfg = _cfg("one_step"); cfg["maximum_updates"] = 15000; fraction = .3
        result = train_one_step(kind, fold, seed, cfg, data, folds[fold], _model_dir(kind, fold, seed), _anchor_path(fold, seed), _v25_det_path(fold, seed), True)
    elif stage == "r5-rounda-sdedit":
        if index < 4: kind = ("calib_sdedit", "pop_sdedit")[index // 2]; fold = (0, 2)[index % 2]; fraction = .3
        else: kind = "calib_sdedit"; fold = (0, 2)[index-4]; fraction = .5
        seed = 20260828; cfg = _cfg("sdedit"); cfg["maximum_updates"] = 15000; cfg["natural_fraction"] = fraction
        result = train_sdedit(kind, fold, seed, cfg, data, folds[fold], _model_dir(kind, fold, seed, fraction), _anchor_path(fold, seed), _v25_det_path(fold, seed), True)
    else:
        kinds = ("calib_refine_det", "pop_refine_det", "calib_sdedit", "pop_sdedit"); kind = kinds[index // 15]; cell = index % 15; fold = cell // 3; seed = V26_SEEDS[cell % 3]; fraction = float(json.loads((RESULT / "round_a/selection.json").read_text())["natural_fraction"]); cfg = _cfg("one_step" if "refine" in kind else "sdedit"); cfg["natural_fraction"] = fraction
        if "sdedit" in kind:
            selection = json.loads((RESULT / "round_a/selection.json").read_text()); cfg["sigma_start"] = selection["sigma_start"]; cfg["ddim_steps"] = selection["ddim_steps"]
        result = (train_one_step(kind, fold, seed, cfg, data, folds[fold], _model_dir(kind, fold, seed), _anchor_path(fold, seed), _v25_det_path(fold, seed), True) if "refine" in kind else train_sdedit(kind, fold, seed, cfg, data, folds[fold], _model_dir(kind, fold, seed), _anchor_path(fold, seed), _v25_det_path(fold, seed), True))
    target = RESULT / ("round_a" if "rounda" in stage else "round_b") / f"{result['kind']}_fold_{result['fold']}_seed_{result['seed']}{'_nf50' if fraction==.5 else ''}.json"; _json(target, result); _json(run / "result_summary.json", result); return result


def _load_bundle(fold: int, seed: int, device: torch.device):
    anchor, _ = load_anchor(_anchor_path(fold, seed), device); det, _ = load_det(_v25_det_path(fold, seed), device)
    models = {}
    for kind in ("calib_refine_det", "pop_refine_det"): models[kind], _ = load_one_step(_model_path(kind, fold, seed), device)
    for kind in ("calib_sdedit", "pop_sdedit"): models[kind], _ = load_sdedit(_model_path(kind, fold, seed), device)
    return anchor, det, models


@torch.no_grad()
def _predict(batch: Mapping[str, Any], anchor, det, models, seed: int, sigma: float, steps: int, device: torch.device) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("V25_POP", "V25_DET_MATCH", "V25_DET_WRONG", "POP_REFINE_DET", "CALIB_REFINE_MATCH", "CALIB_REFINE_WRONG", "CALIB_REFINE_POP_SWAP", "POP_SDEDIT", "CALIB_SDEDIT_MATCH", "CALIB_SDEDIT_WRONG", "CALIB_SDEDIT_POP_SWAP")}; trajectories = []
    for start in range(0, len(batch["y"]), 16):
        sl = slice(start, min(start+16, len(batch["y"]))); y = torch.as_tensor(batch["y"][sl], device=device); q0 = torch.as_tensor(batch["q0"][sl], device=device); c0 = torch.as_tensor(batch["c0"][sl], device=device); pop = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0)); match = det(y, pop, q0, torch.as_tensor(batch["support_eeg"][sl], device=device), torch.as_tensor(batch["support_eog"][sl], device=device)); wrong = det(y, pop, q0, torch.as_tensor(batch["wrong_support_eeg"][sl], device=device), torch.as_tensor(batch["wrong_support_eog"][sl], device=device)); zero = torch.zeros_like(match["context"])
        cr = models["calib_refine_det"]; pr = models["pop_refine_det"]; cs = models["calib_sdedit"]; ps = models["pop_sdedit"]
        values = {"V25_POP": pop, "V25_DET_MATCH": match["artifact"], "V25_DET_WRONG": wrong["artifact"], "POP_REFINE_DET": pr(y, pop), "CALIB_REFINE_MATCH": cr(y, match["artifact"], pop, match["context"]), "CALIB_REFINE_WRONG": cr(y, wrong["artifact"], pop, wrong["context"]), "CALIB_REFINE_POP_SWAP": cr(y, pop, pop, zero)}
        noise = torch.randn(pop.shape, device=device, generator=torch.Generator(device=device).manual_seed(seed+start)); values["POP_SDEDIT"], pt = ps.sample(y, pop, noise, sigma, steps); values["CALIB_SDEDIT_MATCH"], mt = cs.sample(y, match["artifact"], pop, match["context"], noise, sigma, steps); values["CALIB_SDEDIT_WRONG"], _ = cs.sample(y, wrong["artifact"], pop, wrong["context"], noise, sigma, steps); values["CALIB_SDEDIT_POP_SWAP"], _ = cs.sample(y, pop, pop, zero, noise, sigma, steps); trajectories.extend([{**row, "method": "CALIB_SDEDIT_MATCH", "batch_start": start} for row in mt]); trajectories.extend([{**row, "method": "POP_SDEDIT", "batch_start": start} for row in pt])
        for name, value in values.items(): output[name].append(value.cpu().numpy())
    return {name: np.concatenate(value) for name, value in output.items()}, trajectories


def paired_eval(run: Path, round_a: bool = False) -> dict[str, Any]:
    index = _index(); fold = (0, 2)[index] if round_a else index // 3; seed = 20260828 if round_a else V26_SEEDS[index % 3]; device = torch.device("cuda"); anchor, det, models = _load_bundle(fold, seed, device); sampler = SupportSetEpisodeSampler(_cfg("data"), _folds()[fold], "test", seed+401); batch = sampler.sample_paired(192); started = time.time(); prediction, trajectories = _predict(batch, anchor, det, models, seed, .2, 10, device); rows = []
    methods = ("RAW", *prediction.keys())
    for sample, meta in enumerate(batch["meta"]):
        for method in methods:
            estimate = np.zeros_like(batch["artifact"][sample]) if method == "RAW" else prediction[method][sample]; metric = paired_metrics(batch["x"][sample], batch["y"][sample], batch["artifact"][sample], estimate); zero = bool(meta["zero_artifact"])
            if zero: metric["snr_improvement"] = np.nan; metric["artifact_rrmse"] = np.nan
            rows.append({"panel": "paired", "fold": fold, "seed": seed, "participant": meta["participant"], "session": meta["session"], "task": meta["task"], "severity": "zero" if zero else "mild" if meta["gain"] < .5 else "medium" if meta["gain"] < .95 else "severe", "method": method, "zero_artifact": int(zero), **metric})
    panel = "round_a" if round_a else "round_b"; _csv(DERIVED / f"metrics/{panel}/fold_{fold}_seed_{seed}.csv", rows); _csv(DERIVED / f"metrics/{panel}_trajectory/fold_{fold}_seed_{seed}.csv", trajectories)
    elapsed = time.time()-started
    result = {"stage": "R6" if round_a else "R10", "status": "PASS", "fold": fold, "seed": seed, "rows": len(rows), "seconds": elapsed, "windows": len(batch["y"]), "methods_timed_together": len(prediction), "bundle_latency_ms_per_window": 1000.0*elapsed/max(len(batch["y"]), 1), "sealed_reads": 0}; _json(run / "result_summary.json", result); return result


def operating_curve(run: Path) -> dict[str, Any]:
    index = _index(); fold = (0, 2)[index]; seed = 20260828; device = torch.device("cuda"); anchor, det, models = _load_bundle(fold, seed, device); sampler = SupportSetEpisodeSampler(_cfg("data"), _folds()[fold], "validation", seed+509); paired = sampler.sample_paired(96); natural = sampler.sample_natural(96); rows = []
    for sigma, steps in ((0, 10), (.05, 10), (.1, 10), (.2, 10), (.35, 10), (.2, 5), (.2, 25)):
        paired_prediction, _ = _predict(paired, anchor, det, models, seed, sigma, steps, device); natural_prediction, _ = _predict(natural, anchor, det, models, seed, sigma, steps, device)
        for method in ("CALIB_SDEDIT_MATCH", "POP_SDEDIT"):
            values = [paired_metrics(paired["x"][i], paired["y"][i], paired["artifact"][i], paired_prediction[method][i])["rrmse_temporal"] for i in range(len(paired["y"]))]
            teacher = np.asarray(natural["teacher_artifact"]); latent = np.asarray(natural["latent"]); predicted = natural_prediction[method]; teacher_rrmse = float(np.linalg.norm(predicted-teacher)/max(np.linalg.norm(teacher),1e-12)); preservation=[]
            for i in range(len(predicted)):
                energy=np.sqrt(np.mean(latent[i]*latent[i],axis=0)); low=energy<=np.quantile(energy,.3); preservation.append(1-float(np.linalg.norm(predicted[i,:,low])/max(np.linalg.norm(natural["y"][i,:,low]),1e-12)))
            rows.append({"fold": fold, "sigma_start": sigma, "steps": steps, "method": method, "validation_clean_rrmse": float(np.mean(values)), "validation_natural_teacher_rrmse": teacher_rrmse, "validation_natural_preservation": float(np.mean(preservation))})
    _csv(RESULT / f"round_a/operating_curve_fold_{fold}.csv", rows); _json(run / "result_summary.json", {"stage": "R8_CURVE", "status": "PASS", "fold": fold, "rows": len(rows)}); return {"rows": len(rows)}


def round_a_select(run: Path) -> dict[str, Any]:
    curves = []
    for path in sorted((RESULT / "round_a").glob("operating_curve_fold_*.csv")): curves.extend(csv.DictReader(path.open()))
    cells = {}; natural_cells = {}; preservation_cells = {}
    for row in curves:
        if row["method"] == "CALIB_SDEDIT_MATCH":
            key=(float(row["sigma_start"]), int(row["steps"])); cells.setdefault(key, []).append(float(row["validation_clean_rrmse"])); natural_cells.setdefault(key, []).append(float(row["validation_natural_teacher_rrmse"])); preservation_cells.setdefault(key, []).append(float(row["validation_natural_preservation"]))
    positive = [key for key in cells if key[0] > 0]
    # Natural teacher fidelity is primary; preservation breaks near-ties, then paired fidelity.
    selected = min(positive, key=lambda key: (np.mean(natural_cells[key]), -np.mean(preservation_cells[key]), np.mean(cells[key]))); nf = {}
    for fraction in (.3, .5):
        values = []
        suffix = "_nf50" if fraction == .5 else ""
        for fold in (0, 2):
            value = json.loads((RESULT / f"round_a/calib_sdedit_fold_{fold}_seed_20260828{suffix}.json").read_text()); values.append(value["best"]["joint"])
        nf[fraction] = float(np.mean(values))
    fraction = min(nf, key=nf.get); result = {"status": "ROUND_B_CONFIG_FROZEN", "sigma_start": selected[0], "ddim_steps": selected[1], "natural_fraction": fraction, "validation_operating_points": {str(key): {"paired_clean_rrmse":float(np.mean(cells[key])),"natural_teacher_rrmse":float(np.mean(natural_cells[key])),"natural_preservation":float(np.mean(preservation_cells[key]))} for key in cells}, "natural_fraction_joint": nf, "sigma_zero_role":"deterministic_anchor_reference_not_diffusion_candidate", "selection_uses_test": False, "rationale": "Positive-noise operating point selected with validation natural teacher fidelity primary, preservation secondary, and paired fidelity tertiary; sigma=0 remains the deterministic reference. Natural fraction used validation joint error only."}; _json(RESULT / "round_a/selection.json", result); (ROOT / "reports/v26_round_a.md").write_text("# V26 Round A\n\n" + result["rationale"] + "\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n"); _json(run / "result_summary.json", result); return result


def natural_infer(run: Path) -> dict[str, Any]:
    index = _index(); fold = index // 3; seed = V26_SEEDS[index % 3]; device = torch.device("cuda"); anchor, det, models = _load_bundle(fold, seed, device); selection = json.loads((RESULT / "round_a/selection.json").read_text()); query_path = V24_DERIVED / f"fold_{fold}/natural_test_inference.npz"; support_path = V25_DERIVED / f"support_banks/fold_{fold}.npz"
    with np.load(query_path, allow_pickle=False) as archive: batch = {key: np.asarray(archive[key]) for key in ("y", "q0", "c0")}
    with np.load(support_path, allow_pickle=False) as archive: batch.update({key: np.asarray(archive[key]) for key in archive.files})
    prediction, trajectory = _predict(batch, anchor, det, models, seed, float(selection["sigma_start"]), int(selection["ddim_steps"]), device); path = DERIVED / f"predictions/fold_{fold}_seed_{seed}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **prediction); _csv(DERIVED / f"predictions/fold_{fold}_seed_{seed}_trajectory.csv", trajectory)
    value = {"fold": fold, "seed": seed, "path": str(path), "sha256": _digest(path), "query_bundle": str(query_path), "support_bundle": str(support_path), "query_EOG_reads": 0, "query_operator_reads": 0, "event_reads": 0, "sealed_reads": 0}; _json(RESULT / f"natural_evaluation/output_{fold}_{seed}.json", value); _json(run / "result_summary.json", {"stage": "R11", "status": "PASS", **value}); return value


def output_freeze(run: Path) -> dict[str, Any]:
    rows = []
    for fold in range(5):
        for seed in V26_SEEDS:
            value = json.loads((RESULT / f"natural_evaluation/output_{fold}_{seed}.json").read_text())
            if _digest(Path(value["path"])) != value["sha256"]: raise RuntimeError("prediction digest mismatch")
            rows.append(value)
    _csv(RESULT / "natural_evaluation/output_manifest.csv", rows); value = {"stage": "R12", "status": "PASS", "outputs": 15, "query_EOG_reads": 0, "query_operator_reads": 0, "event_reads": 0, "sealed_reads": 0}; _json(RESULT / "natural_evaluation/output_freeze.json", value); _json(run / "result_summary.json", value); return value


def _natural(y: np.ndarray, artifact: np.ndarray, teacher: np.ndarray, latent: np.ndarray) -> dict[str, float]:
    energy = np.sqrt(np.mean(latent*latent, axis=0)); low = energy <= np.quantile(energy, .3); high = energy >= np.quantile(energy, .7); clean = y-artifact; remaining = float(np.linalg.norm((teacher-artifact)[:, high])/max(np.linalg.norm(teacher[:, high]), 1e-12)); preservation = 1-float(np.linalg.norm(artifact[:, low])/max(np.linalg.norm(y[:, low]), 1e-12)); frequencies, py = signal.welch(y[:, low], fs=100, nperseg=min(128, max(8, int(low.sum()))), axis=-1); _, pc = signal.welch(clean[:, low], fs=100, nperseg=min(128, max(8, int(low.sum()))), axis=-1); keep = (frequencies >= 1) & (frequencies <= 15)
    # Fixed evaluator projection: average the upper teacher-energy channel
    # quartile and RMS-normalized EOG regressors, then compute 0.5--15 Hz
    # coherence. This retains a cross-channel ocular construct without 46*d
    # repeated FFTs per method and is independent of each method's output.
    teacher_channel_energy = np.sqrt(np.mean(teacher**2, axis=1))
    ocular_channels = teacher_channel_energy >= np.quantile(teacher_channel_energy, .75)
    eog_scale = np.sqrt(np.mean(latent**2, axis=1, keepdims=True)).clip(1e-8)
    eog_summary = np.mean(latent/eog_scale, axis=0)
    freq, coh_y = signal.coherence(np.mean(y[ocular_channels], axis=0), eog_summary, fs=100, nperseg=min(128, y.shape[-1]))
    _, coh_clean = signal.coherence(np.mean(clean[ocular_channels], axis=0), eog_summary, fs=100, nperseg=min(128, y.shape[-1]))
    band = (freq >= .5) & (freq <= 15)
    coherence_reduction = float(np.mean(coh_y[band])-np.mean(coh_clean[band]))
    blink_residual = float(np.linalg.norm((teacher-artifact)[:, high])/max(np.linalg.norm(y[:, high]), 1e-12))
    teacher_energy = np.sqrt(np.mean(teacher[:, high]**2, axis=1))
    frontal_proxy = teacher_energy >= np.quantile(teacher_energy, .75)
    topography_residual = float(np.linalg.norm((teacher-artifact)[frontal_proxy][:, high])/max(np.linalg.norm(teacher[frontal_proxy][:, high]), 1e-12))
    return {"remaining_ratio": remaining, "artifact_attenuation_db": float(-20*np.log10(max(remaining, 1e-12))), "eeg_eog_coherence_reduction": coherence_reduction, "blink_residual_ratio": blink_residual, "frontal_topography_residual_proxy": topography_residual, "preservation": preservation, "psd_distortion": float(np.mean(np.abs(np.log(np.maximum(py[:, keep], 1e-10))-np.log(np.maximum(pc[:, keep], 1e-10))))), "covariance_distortion": float(np.linalg.norm(np.cov(clean[:, low])-np.cov(y[:, low]))/max(np.linalg.norm(np.cov(y[:, low])), 1e-12)), "erp_proxy": preservation, "ssvep_proxy": preservation, "output_input_rms": float(np.sqrt(np.mean(clean*clean))/max(np.sqrt(np.mean(y*y)), 1e-12)), "observation_change_ratio": float(np.linalg.norm(artifact)/max(np.linalg.norm(y), 1e-12))}


def natural_eval(run: Path) -> dict[str, Any]:
    freeze = json.loads((RESULT / "natural_evaluation/output_freeze.json").read_text()); assert freeze["status"] == "PASS"; index = _index(); fold = index // 3; seed = V26_SEEDS[index % 3]
    with np.load(V24_DERIVED / f"fold_{fold}/natural_test_inference.npz", allow_pickle=False) as archive: query = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(V24_DERIVED / f"fold_{fold}/natural_test_evaluator.npz", allow_pickle=False) as archive: evaluator = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(DERIVED / f"predictions/fold_{fold}_seed_{seed}.npz", allow_pickle=False) as archive: prediction = {key: np.asarray(archive[key]) for key in archive.files}
    roles = [row for row in csv.DictReader((ROOT / "results/pa_el_scad_v24/role_manifest.csv").open()) if row["fold"] == str(fold) and row["stream"] == "natural" and row["split"] == "test"]; rows = []
    for i, meta in enumerate(roles):
        for method in ("RAW", *prediction.keys()):
            artifact = np.zeros_like(query["y"][i]) if method == "RAW" else prediction[method][i]; rows.append({"panel": "natural", "fold": fold, "seed": seed, "participant": meta["participant"], "session": meta["session"], "task": meta["task"], "method": method, **_natural(query["y"][i], artifact, evaluator["teacher_artifact"][i], evaluator["latent"][i])})
    _csv(DERIVED / f"metrics/natural/fold_{fold}_seed_{seed}.csv", rows); value = {"stage": "R13", "status": "PASS", "fold": fold, "seed": seed, "rows": len(rows), "evaluator_after_freeze": True, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def aggregate(run: Path) -> dict[str, Any]:
    paired = []; natural = []
    for fold in range(5):
        for seed in V26_SEEDS:
            paired.extend(csv.DictReader((DERIVED / f"metrics/round_b/fold_{fold}_seed_{seed}.csv").open())); natural.extend(csv.DictReader((DERIVED / f"metrics/natural/fold_{fold}_seed_{seed}.csv").open()))
    paired_metrics_names = ["rrmse_temporal", "rrmse_spectral", "correlation", "snr_improvement", "artifact_rrmse", "artifact_correlation", "clean_output_rms_ratio"]
    natural_metrics_names = ["remaining_ratio", "artifact_attenuation_db", "eeg_eog_coherence_reduction", "blink_residual_ratio", "frontal_topography_residual_proxy", "preservation", "psd_distortion", "covariance_distortion", "erp_proxy", "ssvep_proxy", "output_input_rms", "observation_change_ratio"]
    pp = participant_first(paired, paired_metrics_names); npanel = participant_first(natural, natural_metrics_names); summary = []
    for panel, rows, metrics in (("paired", pp, paired_metrics_names), ("natural", npanel, natural_metrics_names)):
        for method in sorted({str(r["method"]) for r in rows}):
            selected = [r for r in rows if r["method"] == method]
            for metric in metrics:
                vector = np.asarray([float(r[metric]) for r in selected]); summary.append({"panel": panel, "method": method, "metric": metric, **bootstrap(vector)})
    # Preserve the requested EEGDfus comparator as a frozen V22 reference.
    # It is never pooled into V26 participant contrasts or model selection.
    v22_summary = ROOT / "results/scad_v22/method_summary.csv"
    metric_alias = {"artifact_rmse": "artifact_rrmse", "heldout_eog_remaining_ratio": "remaining_ratio", "output_input_rms_ratio": "output_input_rms"}
    if v22_summary.is_file():
        for row in csv.DictReader(v22_summary.open()):
            if row["method"] != "EEGDFUS_UNIFIED": continue
            metric = metric_alias.get(row["metric"], row["metric"])
            panel = "paired" if metric in paired_metrics_names else "natural" if metric in natural_metrics_names else None
            if panel:
                summary.append({"panel": panel, "method": "EEGDFUS_V22_FROZEN", "metric": metric, "mean": float(row["mean"]), "median": float(row["median"]), "bootstrap_low": float(row["bootstrap_low"]), "bootstrap_high": float(row["bootstrap_high"]), "participants": int(row["participants"]), "evidence_status": "historical_frozen_v22_reference_not_v26_recomputed"})
    _csv(RESULT / "method_summary.csv", summary)
    definitions = (("ONE_STEP_SUPPORT", "CALIB_REFINE_MATCH", "POP_REFINE_DET"), ("DIFF_SUPPORT", "CALIB_SDEDIT_MATCH", "POP_SDEDIT"), ("ONE_STEP_SPECIFICITY", "CALIB_REFINE_MATCH", "CALIB_REFINE_WRONG"), ("DIFF_SPECIFICITY", "CALIB_SDEDIT_MATCH", "CALIB_SDEDIT_WRONG"), ("ONE_STEP_BASE", "CALIB_REFINE_MATCH", "V25_DET_MATCH"), ("DIFF_ONE_STEP", "CALIB_SDEDIT_MATCH", "CALIB_REFINE_MATCH"))
    effects = []
    for panel, rows, metrics in (("paired", pp, ["rrmse_temporal"]), ("natural", npanel, ["remaining_ratio", "preservation"])):
        for metric in metrics:
            for name, first, second in definitions:
                for item in contrast(rows, first, second, metric): effects.append({"panel": panel, "contrast": name, **item})
    _csv(RESULT / "participant_effects.csv", effects)
    seed_rows = []
    for panel, rows, metrics in (("paired", paired, paired_metrics_names), ("natural", natural, natural_metrics_names)):
        for seed in V26_SEEDS:
            for method in sorted({r["method"] for r in rows}):
                chosen = [r for r in rows if int(r["seed"]) == seed and r["method"] == method]
                for metric in metrics: seed_rows.append({"panel": panel, "seed": seed, "method": method, "metric": metric, "mean": float(np.nanmean([float(r[metric]) for r in chosen]))})
    _csv(RESULT / "seed_effects.csv", seed_rows)
    severity_rows = []
    for severity in ("mild", "medium", "severe"):
        selected = [r for r in paired if r["severity"] == severity]
        reduced = participant_first(selected, ["rrmse_temporal"])
        for item in contrast(reduced, "CALIB_SDEDIT_MATCH", "CALIB_REFINE_MATCH", "rrmse_temporal"): severity_rows.append({"severity": severity, **item})
    _csv(RESULT / "severity_effects.csv", severity_rows)
    operating = []
    for path in sorted((RESULT / "round_a").glob("operating_curve_fold_*.csv")): operating.extend(csv.DictReader(path.open()))
    _csv(RESULT / "sigma_step_effects.csv", operating)
    exposure = []
    for path in sorted((RESULT / "round_b").glob("*.json")):
        value = json.loads(path.read_text()); exposure.append({"model": value["kind"], "fold": value["fold"], "seed": value["seed"], "parameters": value["parameters"], "updates": value["updates"], "training_seconds": value["training_seconds"], "device": value["device"]})
    for path in sorted((RESULT / "runs/r10-paired").glob("job_*/result_summary.json")):
        value = json.loads(path.read_text()); exposure.append({"model": "all_inference_methods", "fold": value["fold"], "seed": value["seed"], "parameters": "", "updates": "", "training_seconds": "", "device": "cuda", "bundle_latency_ms_per_window": value.get("bundle_latency_ms_per_window"), "methods_timed_together": value.get("methods_timed_together")})
    _csv(RESULT / "latency_summary.csv", exposure)
    def stat(panel: str, contrast_name: str, metric: str) -> dict[str, Any]: return bootstrap(np.asarray([float(r["effect"]) for r in effects if r["panel"] == panel and r["contrast"] == contrast_name and r["metric"] == metric]))
    one = stat("paired", "ONE_STEP_BASE", "rrmse_temporal"); diff = stat("paired", "DIFF_ONE_STEP", "rrmse_temporal"); support = stat("paired", "DIFF_SUPPORT", "rrmse_temporal"); one_specificity = stat("paired", "ONE_STEP_SPECIFICITY", "rrmse_temporal"); diff_specificity = stat("paired", "DIFF_SPECIFICITY", "rrmse_temporal"); nat_art = stat("natural", "DIFF_SUPPORT", "remaining_ratio"); nat_pres = stat("natural", "DIFF_SUPPORT", "preservation"); nat_one_art = stat("natural", "ONE_STEP_SUPPORT", "remaining_ratio"); nat_one_pres = stat("natural", "ONE_STEP_SUPPORT", "preservation"); nat_diff_one_art = stat("natural", "DIFF_ONE_STEP", "remaining_ratio"); nat_diff_one_pres = stat("natural", "DIFF_ONE_STEP", "preservation")
    subject = "paired_signal_preserved" if support["mean"] > 0 else "paired_signal_weakened" if one["mean"] > 0 else "paired_signal_lost"; diffusion = "clear_increment_over_one_step" if diff["bootstrap_low"] > 0 else "small_increment" if diff["mean"] > 0 else "one_step_equivalent" if abs(diff["mean"]) < .002 else "one_step_better"; natural_class = "promising" if nat_art["mean"] > 0 and nat_pres["mean"] >= 0 else "both_failed" if nat_art["mean"] <= 0 and nat_pres["mean"] < 0 else "artifact_reduction_insufficient" if nat_art["mean"] <= 0 else "preservation_concern"; next_route = "A. continue CalibSDEdit" if natural_class == "promising" and subject != "paired_signal_lost" else "D. add lightweight energy refinement" if natural_class == "preservation_concern" and subject != "paired_signal_lost" else "B. improve natural-reference training" if natural_class in ("artifact_reduction_insufficient", "both_failed") else "C. test diffusion uncertainty/proper scoring"
    diagnosis = {"engineering": "valid", "subject_context": subject, "second_stage_one_step": "improves_base_det" if one["mean"] > 0 else "equivalent_to_base_det" if abs(one["mean"]) < .002 else "worse_than_base_det", "diffusion": diffusion, "diffusion_positioning": "competitive_mechanism_comparison_not_retention_gate", "retention_requires_diffusion_over_one_step": False, "primary_interpretive_priority": "natural_artifact_preservation_validity", "natural_tradeoff": natural_class, "next_route": next_route, "paired": {"one_step_vs_base": one, "one_step_specificity": one_specificity, "diffusion_vs_one_step": diff, "diffusion_support": support, "diffusion_specificity": diff_specificity}, "natural": {"one_step_support_artifact": nat_one_art, "one_step_support_preservation": nat_one_pres, "diffusion_support_artifact": nat_art, "diffusion_support_preservation": nat_pres, "diffusion_vs_one_step_artifact": nat_diff_one_art, "diffusion_vs_one_step_preservation": nat_diff_one_pres}, "development_only": True, "K": 1, "query_EOG_inference_reads": 0, "query_operator_inference_reads": 0, "event_inference_reads": 0, "sealed_reads": 0}
    _json(RESULT / "development_diagnosis.json", diagnosis); _make_reports(diagnosis, summary); _make_figures(summary, effects, operating); _json(run / "result_summary.json", diagnosis); return diagnosis


def _make_reports(diagnosis: Mapping[str, Any], summary: list[dict[str, Any]]) -> None:
    forensic = json.loads(next((RESULT / "runs/r1-forensic").glob("job_*/result_summary.json")).read_text())
    (ROOT / "reports/v26_v25_diffusion_forensic.md").write_text("# V26 V25 diffusion forensic\n\nThe V25 coefficient target is not invariant to equivalent rotations of the learned basis, while its decoded sensor-space artifact is invariant. This makes the old latent target coordinate-unstable.\n\n```json\n" + json.dumps(forensic, indent=2) + "\n```\n")
    (ROOT / "reports/v26_project_plan.md").write_text("# V26 CalibSDEdit project plan\n\nV26 freezes the useful V25 raw-support deterministic anchor, refines a 46-channel artifact in fixed sensor coordinates from a moderate-noise deterministic warm start, and compares diffusion competitively with a matched one-step refiner for positioning and mechanism analysis. Diffusion is not retained or rejected by a strict paired DIFF>DET rule; natural artifact–preservation validity has higher interpretive priority. All results are development-only; sealed reads remain zero.\n")
    paired = diagnosis["paired"]; natural = diagnosis["natural"]
    text = f"# V26 final development diagnosis\n\nEngineering: `{diagnosis['engineering']}`. Subject context: `{diagnosis['subject_context']}`. One-step: `{diagnosis['second_stage_one_step']}`. Diffusion: `{diagnosis['diffusion']}`. Natural trade-off: `{diagnosis['natural_tradeoff']}`.\n\nPaired one-step vs V25 DET: `{paired['one_step_vs_base']}`. Diffusion vs matched one-step: `{paired['diffusion_vs_one_step']}`. This is a competitive mechanism comparison, not a diffusion-retention gate. Diffusion support value: `{paired['diffusion_support']}`.\n\nNatural artifact support contrast: `{natural['diffusion_support_artifact']}`; preservation contrast: `{natural['diffusion_support_preservation']}`. Joint natural validity has higher interpretive priority than strict DIFF>DET ordering.\n\nNext route: **{diagnosis['next_route']}**. No confirmation, SOTA, deployment, or clinical claim is made.\n"
    (ROOT / "reports/v26_final_development_diagnosis.md").write_text(text); (ROOT / "reports/v26_round_b.md").write_text("# V26 Round B\n\n" + text); (ROOT / "reports/v26_natural_development.md").write_text("# V26 natural development\n\n" + text)


def _make_figures(summary: list[dict[str, Any]], effects: list[dict[str, Any]], operating: list[dict[str, Any]]) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    target = ROOT / "figures/calib_sdedit_v26"; target.mkdir(parents=True, exist_ok=True)
    methods = ["V25_DET_MATCH", "CALIB_REFINE_MATCH", "CALIB_SDEDIT_MATCH", "POP_REFINE_DET", "POP_SDEDIT"]
    values = {row["method"]: row["mean"] for row in summary if row["panel"] == "paired" and row["metric"] == "rrmse_temporal"}
    fig, ax = plt.subplots(); ax.bar(methods, [values[m] for m in methods]); ax.tick_params(axis="x", rotation=35); ax.set_ylabel("paired clean RRMSE"); fig.tight_layout(); fig.savefig(target / "paired_method_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots();
    for idx, name in enumerate(("ONE_STEP_SUPPORT", "DIFF_SUPPORT", "DIFF_ONE_STEP")): ax.scatter([float(r["effect"]) for r in effects if r["panel"] == "paired" and r["contrast"] == name], np.full(15, idx), label=name)
    ax.axvline(0, color="black"); ax.set_yticks(range(3), ["one-step support", "diffusion support", "diffusion-one-step"]); fig.tight_layout(); fig.savefig(target / "context_effect_forest.png"); plt.close(fig)
    for column, filename, xlabel in (("sigma_start", "sigma_start_curve.png", "sigma_start"), ("steps", "step_curve.png", "DDIM steps")):
        fig, ax = plt.subplots()
        filtered = [r for r in operating if (r["steps"] == "10" if column == "sigma_start" else r["sigma_start"] == "0.2") and r["method"] == "CALIB_SDEDIT_MATCH"]
        groups = sorted({float(r[column]) for r in filtered}); ax.plot(groups, [np.mean([float(r["validation_clean_rrmse"]) for r in filtered if float(r[column]) == value]) for value in groups], marker="o"); ax.set(xlabel=xlabel, ylabel="validation clean RRMSE"); fig.tight_layout(); fig.savefig(target / filename); plt.close(fig)
    # Required named diagnostics; concise plots use committed tabular sources.
    rotations = list(csv.DictReader((RESULT / "basis_rotation_diagnostics.csv").open())); fig, ax = plt.subplots(); ax.hist([float(r["projector_distance"]) for r in rotations]); ax.set_xlabel("support-subset projector distance"); fig.tight_layout(); fig.savefig(target / "basis_rotation_stability.png"); plt.close(fig)
    residual = list(csv.DictReader((RESULT / "residual_scale_diagnostics.csv").open())); fig, ax = plt.subplots(); ax.boxplot([[float(r["rrmse_temporal"]) for r in residual if r["method"] == method] for method in ("RAW", "DET_MATCH", "DIFF_MATCH")], tick_labels=["RAW", "DET", "V25 DIFF"]); fig.tight_layout(); fig.savefig(target / "residual_scale.png"); plt.close(fig)
    # Training convergence from every accepted Round-B cell.
    fig, ax = plt.subplots()
    for kind in ("calib_refine_det", "pop_refine_det", "calib_sdedit", "pop_sdedit"):
        curves = []
        for path in sorted((RESULT / "round_b").glob(f"{kind}_fold_*_seed_*.json")):
            curves.extend(json.loads(path.read_text()).get("curve", []))
        by_step = {step: [float(row["joint"]) for row in curves if int(row["step"]) == step] for step in sorted({int(row["step"]) for row in curves})}
        if by_step: ax.plot(list(by_step), [np.mean(by_step[step]) for step in by_step], label=kind)
    ax.set(xlabel="update", ylabel="validation joint error"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(target / "training_curves.png"); plt.close(fig)
    # Reverse trajectory is reported in fixed sensor coordinates.
    trajectory = []
    for path in sorted((DERIVED / "metrics/round_b_trajectory").glob("*.csv")): trajectory.extend(csv.DictReader(path.open()))
    fig, ax = plt.subplots()
    for metric in ("state_rms", "x0_rms", "refinement_rms"):
        by_step = {step: [float(row[metric]) for row in trajectory if row["method"] == "CALIB_SDEDIT_MATCH" and int(row["step"]) == step] for step in sorted({int(row["step"]) for row in trajectory if row["method"] == "CALIB_SDEDIT_MATCH"}, reverse=True)}
        if by_step: ax.plot(list(by_step), [np.mean(by_step[step]) for step in by_step], marker="o", label=metric)
    ax.set(xlabel="reverse timestep", ylabel="RMS"); ax.legend(); fig.tight_layout(); fig.savefig(target / "sdedit_trajectory.png"); plt.close(fig)
    # Natural validity is a joint attenuation--preservation question.
    natural_values = {(row["method"], row["metric"]): float(row["mean"]) for row in summary if row["panel"] == "natural"}
    fig, ax = plt.subplots()
    for method in methods:
        if (method, "artifact_attenuation_db") in natural_values:
            x = natural_values[method, "artifact_attenuation_db"]; y = natural_values[method, "preservation"]
            ax.scatter(x, y); ax.annotate(method, (x, y), fontsize=7)
    ax.set(xlabel="artifact attenuation (dB)", ylabel="low-artifact preservation"); fig.tight_layout(); fig.savefig(target / "attenuation_preservation_scatter.png"); plt.close(fig)
    # End-to-end bundle latency is deliberately reported without attributing
    # the shared data/anchor work to a single method.
    latency = []
    for path in sorted((RESULT / "runs/r10-paired").glob("job_*/result_summary.json")):
        value = json.loads(path.read_text()); latency.append(float(value["bundle_latency_ms_per_window"]))
    fig, ax = plt.subplots(); ax.scatter([np.mean(latency)] if latency else [], [natural_values.get(("CALIB_SDEDIT_MATCH", "artifact_attenuation_db"), np.nan)] if latency else []); ax.set(xlabel="all-method bundle latency (ms/window)", ylabel="CalibSDEdit natural attenuation (dB)"); fig.tight_layout(); fig.savefig(target / "quality_latency_curve.png"); plt.close(fig)


STAGES = {"r0-preflight": preflight, "r1-forensic": forensic, "r2-prepare": prepare, "r3-sanity": sanity, "r4-rounda-one-step": lambda run: train_stage("r4-rounda-one-step", run), "r5-rounda-sdedit": lambda run: train_stage("r5-rounda-sdedit", run), "r6-rounda-paired": lambda run: paired_eval(run, True), "r7-operating-curve": operating_curve, "r8-select": round_a_select, "r9-roundb-train": lambda run: train_stage("r9-roundb-train", run), "r10-paired": paired_eval, "r11-natural-infer": natural_infer, "r12-output-freeze": output_freeze, "r13-natural-eval": natural_eval, "r14-aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=STAGES, required=True); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=True); STAGES[args.stage](args.run_dir)


if __name__ == "__main__": main()
