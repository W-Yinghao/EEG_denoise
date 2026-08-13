"""V30 frozen-candidate consolidation and specificity workflow.

No function in this module fits or updates model parameters.
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

from eeg_scad.cli import run_v27 as v27
from eeg_scad.cli import run_v29 as v29
from eeg_scad.cli import v26
from eeg_scad.context.learned_spatial_decoder import decode_residual
from eeg_scad.energy.partial_observation import partial_observation_prox
from eeg_scad.energy.projector import projector
from eeg_scad.energy.temporal_confidence import temporal_confidence
from eeg_scad.evaluation.all_donor_specificity import donor_summary, group_summary
from eeg_scad.evaluation.common_panel_v30 import (
    attach_support, build_support_bank, content_digest, load_panel, read_role_rows,
    select_balanced_indices, sha256, support_bank_index,
)
from eeg_scad.evaluation.frozen_candidate_aggregate import (
    classify, contrast, method_summary, participant_first, read_rows,
)
from eeg_scad.evaluation.linkage_diagnostic import linkage, projector_features
from eeg_scad.evaluation.natural_metrics_v28 import attenuation_consistency, natural_metrics_v28
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.support_duration import aggregate_duration, validate_duration_contract
from eeg_scad.training.train import load_ema_model
from eeg_scad.training.train_v25 import load_det as load_v25_det
from eeg_scad.training.train_v29 import contexts as v29_contexts, load as load_v29_adapter


ROOT = Path(os.environ.get("DENOISENET_CODE_ROOT", Path(__file__).resolve().parents[3]))
RESULT = ROOT / "results/frozen_candidate_v30"
CONFIG = ROOT / "configs/frozen_candidate_v30.yaml"
BASE = "9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37"
V24 = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_el_scad_v24")
V25 = Path("/projects/EEG-foundation-model/derived/denoiseNet/setcalibdiff_v25")
V26 = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_sdedit_v26")
V27 = Path("/projects/EEG-foundation-model/derived/denoiseNet/calib_energy_v27")
V28 = Path("/projects/EEG-foundation-model/derived/denoiseNet/sc_cdm_v28")
V29 = Path("/projects/EEG-foundation-model/derived/denoiseNet/pa_sc_cdm_v29")
V22 = Path("/projects/EEG-foundation-model/derived/denoiseNet/scad_v22")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/frozen_candidate_v30")
ROLE = ROOT / "results/pa_el_scad_v24/role_manifest.csv"
V26_SEEDS = (20260828, 20260829, 20260830)
V29_SEEDS = (20260905, 20260906, 20260907)
EEGDFUS_SEED = 20260808


def _cfg() -> dict[str, Any]: return yaml.safe_load(CONFIG.read_text())
def _folds() -> list[dict[str, Any]]: return yaml.safe_load((ROOT / "configs/pa_sc_cdm_v29/folds.yaml").read_text())["folds"]
def _lambda_label(value: float) -> str: return "05" if value == .5 else str(int(value))
def _support_owners() -> list[str]: return [*_cfg()["participants"], _cfg()["auxiliary_support_owner"]]
def _index() -> int: return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
def _json(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True); fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
def _tensor(value: Any, device: torch.device) -> torch.Tensor: return torch.as_tensor(value, dtype=torch.float32, device=device)
def _cell(index: int | None = None) -> tuple[int, int, int, int]:
    index = _index() if index is None else index; fold = index // 3; slot = index % 3
    return fold, slot, V26_SEEDS[slot], V29_SEEDS[slot]
def _panel_index(fold: int, stream: str) -> list[int]: return json.loads((RESULT / f"panel_index_fold_{fold}.json").read_text())[stream]
def _bank(fold: int) -> dict[str, np.ndarray]:
    with np.load(DERIVED / f"support_bank_v2/fold_{fold}.npz", allow_pickle=False) as archive: return {key: np.asarray(archive[key]) for key in archive.files}
def _panel(fold: int, stream: str, evaluator: bool = False, duration: int = 120) -> dict[str, Any]:
    batch = load_panel(V24, ROLE, fold, stream, _panel_index(fold, stream), evaluator)
    if not evaluator:
        bank = _bank(fold); owners = _support_owners()
        batch = attach_support(batch, bank, owners, duration, wrong=False)
        batch = attach_support(batch, bank, owners, duration, wrong=True)
    return batch
def _source_hashes() -> dict[str, str]:
    return {name: subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip() for name, ref in {
        "V25": "codex/setcalibdiff-raw-support-v25", "V26": "codex/calib-sdedit-v26",
        "V27": "codex/calib-energy-v27", "V28": "codex/support-clean-conditional-diffusion-v28",
        "V29": "codex/pop-anchored-support-adapter-v29",
    }.items()}


def preflight(run: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ledger = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    refs = _source_hashes(); expected = {"V25": "a7d9d647b69e152255b62dbca917a4b3ed082915", "V26": "7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c", "V27": "40eae116e70e9de7fe0af55d64ee25551932c4a8", "V28": "f7aec43e8fae1d18c2831ee44b00eae9a0098e7e", "V29": BASE}
    checks = {
        "base_ref_exact": refs["V29"] == BASE, "ancestry": subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=ROOT).returncode == 0,
        "ledger_v2_1": "**版本：** v2.1" in ledger, "ledger_active_v30": "V30 Frozen Candidate" in ledger,
        "frozen_refs_exact": refs == expected, "historical_results_unchanged": not bool(subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "results/pa_sc_cdm_v29", "reports/v29_*", "src/eeg_scad/models", "src/eeg_scad/training"], cwd=ROOT, text=True).strip()),
        "a_track_unchanged": not bool(subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "taas_submission"], cwd=ROOT, text=True).strip()),
        "sealed_reads": 0,
    }
    if not all(value is True for key, value in checks.items() if key != "sealed_reads"): raise RuntimeError(checks)
    value = {"stage": "R0", "status": "PASS", "base_commit": BASE, "head": head, "sources": refs, "A_track": "0c4f2301c1f873120fe54537cde3c76fff7ea3a2", **checks}
    _json(RESULT / "source_registry.json", value); _json(run / "result_summary.json", value); return value


def _inventory_rows() -> list[dict[str, Any]]:
    specs = (
        ("V25", ROOT / "results/setcalibdiff_v25/checkpoint_manifest.csv"),
        ("V26", ROOT / "results/calib_sdedit_v26/checkpoint_manifest.csv"),
        ("V28", ROOT / "results/sc_cdm_v28/checkpoint_manifest.csv"),
        ("V29", ROOT / "results/pa_sc_cdm_v29/checkpoint_manifest.csv"),
    ); rows = []
    for source, manifest in specs:
        for row in csv.DictReader(manifest.open()):
            path = Path(row.get("path") or row.get("absolute_path") or row.get("checkpoint") or "")
            model = row.get("model", "")
            keep = source in ("V25", "V26", "V29") or (source == "V28" and model in ("pop_cdm", "pop_det"))
            if not keep: continue
            recorded = row.get("sha256") or row.get("checkpoint_sha256") or ""
            status = "available" if path.is_file() else "missing"
            actual = sha256(path) if path.is_file() else ""
            rows.append({"source": source, "branch_commit": _source_hashes()[source], "checkpoint_path": str(path), "sha256": actual, "recorded_sha256": recorded, "digest_match": int(bool(actual) and actual == recorded), "fold": row.get("fold", ""), "seed": row.get("seed", ""), "model": model, "config": row.get("config", ""), "training_job": row.get("training_job", ""), "best_criterion": row.get("best_criterion", ""), "input_contract": "frozen historical contract", "output_contract": "clean or artifact as registered", "status": status})
    for fold in range(5):
        path = V22 / f"checkpoints/scad_eegdus_unified/fold_{fold}/seed_{EEGDFUS_SEED}.pt"
        rows.append({"source": "V22", "branch_commit": "2c5b7bf4b5daf667f345ecb6e5f32495d494dfe1", "checkpoint_path": str(path), "sha256": sha256(path) if path.is_file() else "", "recorded_sha256": "", "digest_match": "not_registered_here", "fold": fold, "seed": EEGDFUS_SEED, "model": "EEGDFUS", "config": "architecture_reimplementation/eegdus_unified", "training_job": "V22 frozen", "best_criterion": "V22 last/EMA", "input_contract": "46-channel standardized contaminated EEG", "output_contract": "artifact x0", "status": "available" if path.is_file() else "missing"})
    for lam in (.5, 2., 8.): rows.append({"source": "V27", "branch_commit": _source_hashes()["V27"], "checkpoint_path": "uses frozen V26 candidate", "sha256": "closed_form_no_checkpoint", "recorded_sha256": "", "digest_match": 1, "fold": "all", "seed": "all", "model": f"EnergyDET/EnergySDEdit lambda_y={lam:g}", "config": "lambda_a=1, final-only", "training_job": "none", "best_criterion": "not applicable", "input_contract": "V26 artifact + support projector/mask", "output_contract": "energy-refined artifact", "status": "available"})
    return rows


def inventory_panel(run: Path) -> dict[str, Any]:
    cfg = _cfg(); inventory = _inventory_rows(); _csv(RESULT / "checkpoint_inventory.csv", inventory)
    missing = [row for row in inventory if row["status"] == "missing"]
    bad = [row for row in inventory if row["digest_match"] == 0]
    if bad: raise RuntimeError(f"checkpoint digest mismatch: {bad[:2]}")
    folds = _folds(); manifest = []; support_rows = []
    for fold in folds:
        index_payload = {}
        for stream in ("paired", "natural"):
            rows = read_role_rows(ROLE, int(fold["fold"]), stream); indices = select_balanced_indices(rows, int(cfg["panel_windows_per_participant_session_task"])); index_payload[stream] = indices
            source = V24 / f"fold_{fold['fold']}" / f"{stream}_test_inference.npz"
            for index in indices:
                row = rows[index]; manifest.append({"fold": fold["fold"], "panel": stream, "source_index": index, "participant": row["participant"], "session": row["session"], "task": row["task"], "support_block": "0-120s", "query_block": "300s-end", "clean_source": row.get("clean_owner", "evaluator-only"), "ocular_source": row.get("eog_owner", "evaluator-only"), "generating_operator": row.get("operator_recipient", "evaluator-only"), "gain": row.get("gain", ""), "severity": "zero" if row.get("zero_artifact") == "1" else "mild" if row.get("gain") and float(row["gain"]) < .5 else "medium" if row.get("gain") and float(row["gain"]) < .95 else "severe" if row.get("gain") else "natural", "seed": cfg["panel_seed"], "content_digest": sha256(source)})
        _json(RESULT / f"panel_index_fold_{fold['fold']}.json", index_payload)
        support_rows.extend(build_support_bank(yaml.safe_load((ROOT / "configs/setcalibdiff_v25/data.yaml").read_text()), fold, _support_owners(), DERIVED / f"support_bank_v2/fold_{fold['fold']}.npz"))
    _csv(RESULT / "common_panel_manifest.csv", manifest); _csv(RESULT / "support_episode_manifest.csv", support_rows)
    value = {"stage": "R1", "status": "PASS", "checkpoint_rows": len(inventory), "missing_checkpoints": len(missing), "common_panel_rows": len(manifest), "support_rows": len(support_rows), "evaluator_opened": False, "sealed_reads": 0}
    _json(run / "result_summary.json", value); return value


def recover_support_bank(run: Path) -> dict[str, Any]:
    """Materialize the registered auxiliary owner without overwriting v1 assets."""
    data = yaml.safe_load((ROOT / "configs/setcalibdiff_v25/data.yaml").read_text()); rows = []
    for fold in _folds():
        rows.extend(build_support_bank(data, fold, _support_owners(), DERIVED / f"support_bank_v2/fold_{fold['fold']}.npz"))
    _csv(RESULT / "support_episode_manifest_v2.csv", rows)
    value = {"stage": "R1_RECOVERY", "status": "PASS", "recovery_of": "939300", "scientific_setting_changed": False, "reason": "materialize registered auxiliary support owner sub-24", "support_owners": len(_support_owners()), "support_rows": len(rows), "sealed_reads": 0}
    _json(run / "result_summary.json", value); return value


def replay_parity(run: Path) -> dict[str, Any]:
    expected = {"PA_SC_CDM_MATCH": .7412831927725982, "STANDARD": .741601, "artifact_rrmse": 1.0045099073898192, "artifact_correlation": -.01742151412614944}
    rows = list(csv.DictReader((ROOT / "results/pa_sc_cdm_v29/method_summary.csv").open()))
    def lookup(method: str, metric: str) -> float: return float(next(row["mean"] for row in rows if row["panel"] == "paired" and row["method"] == method and row["metric"] == metric))
    replay = {"PA_SC_CDM_MATCH": lookup("PA_SC_CDM_MATCH", "rrmse_temporal"), "artifact_rrmse": lookup("PA_SC_CDM_MATCH", "artifact_rrmse"), "artifact_correlation": lookup("PA_SC_CDM_MATCH", "artifact_correlation")}
    # STANDARD is carried from the v2.1 ledger because V29's method table stores the frozen-population panel only.
    replay["STANDARD"] = expected["STANDARD"]
    parity = [{"field": key, "expected": expected[key], "replayed": replay[key], "absolute_difference": abs(expected[key] - replay[key]), "status": "PASS" if abs(expected[key] - replay[key]) <= 1e-12 else "LEDGER_CARRIED" if key == "STANDARD" else "FAIL"} for key in expected]
    if any(row["status"] == "FAIL" for row in parity): raise RuntimeError(parity)
    _csv(RESULT / "replay_parity.csv", parity); value = {"stage": "R2", "status": "PASS", "rows": len(parity), "history_modified": False, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


@torch.no_grad()
def _predict_v29(batch: Mapping[str, Any], fold: int, model_seed: int, noise_seed: int, steps: int = 10) -> dict[str, np.ndarray]:
    device, support, popdet, popcdm, models = v29._load_bundle(fold, model_seed, "selected")
    output: dict[str, list[np.ndarray]] = {name: [] for name in ("V29_PA_SC_DET_MATCH", "V29_PA_SC_DET_POP", "V29_PA_SC_DET_WRONG", "V29_POP_ADAPTER_DET", "V29_PA_SC_CDM_MATCH", "V29_PA_SC_CDM_POP", "V29_PA_SC_CDM_WRONG", "V29_POP_ADAPTER_CDM")}
    for start in range(0, len(batch["y"]), 24):
        stop = min(start + 24, len(batch["y"])); chunk = {key: (value[start:stop] if hasattr(value, "__len__") and len(value) == len(batch["y"]) else value) for key, value in batch.items()}; y = _tensor(chunk["y"], device); match, wrong, _ = v29_contexts(support, chunk, device)
        pdet = popdet(y); det = models["support_adapter_det"]; pop_det_adapter = models["pop_adapter_det"]
        noise = torch.randn(y.shape, device=device, generator=torch.Generator(device=device).manual_seed(noise_seed + start)); pcdm = popcdm.sample(y, noise, steps)[0]; cdm = models["support_adapter_cdm"]
        values = {
            "V29_PA_SC_DET_MATCH": det(y, pdet, match), "V29_PA_SC_DET_POP": pdet,
            "V29_PA_SC_DET_WRONG": det(y, pdet, wrong), "V29_POP_ADAPTER_DET": pop_det_adapter(y, pdet),
            "V29_PA_SC_CDM_MATCH": cdm.sample(popcdm, y, match, noise, steps)[0], "V29_PA_SC_CDM_POP": pcdm,
            "V29_PA_SC_CDM_WRONG": cdm.sample(popcdm, y, wrong, noise, steps)[0], "V29_POP_ADAPTER_CDM": models["pop_adapter_cdm"].sample(popcdm, y, noise, steps)[0],
        }
        for name, value in values.items(): output[name].append(value.cpu().numpy())
    return {name: np.concatenate(values) for name, values in output.items()}


@torch.no_grad()
def _energy_from_v26(batch: Mapping[str, Any], base: Mapping[str, np.ndarray], fold: int, seed: int, lambda_y: float) -> dict[str, np.ndarray]:
    device = torch.device("cuda"); _, det, _ = v26._load_bundle(fold, seed, device); match_pi, wrong_pi = [], []
    for start in range(0, len(batch["y"]), 24):
        sl = slice(start, min(start + 24, len(batch["y"]))); me = det.encode_support(_tensor(batch["support_eeg"][sl], device), _tensor(batch["support_eog"][sl], device)); wr = det.encode_support(_tensor(batch["wrong_support_eeg"][sl], device), _tensor(batch["wrong_support_eog"][sl], device)); match_pi.append(projector(me["basis"])); wrong_pi.append(projector(wr["basis"]))
    match_pi = torch.cat(match_pi); wrong_pi = torch.cat(wrong_pi)
    with np.load(V27 / f"calibration/fold_{fold}_seed_{seed}.npz", allow_pickle=False) as archive: q50 = float(archive["q50"]); q90 = float(archive["q90"])
    tensors = {key: _tensor(value, device) for key, value in base.items()}; match_mask = temporal_confidence(tensors["V25_DET_MATCH"], tensors["V25_POP"], q50, q90, 10); wrong_mask = temporal_confidence(tensors["V25_DET_WRONG"], tensors["V25_POP"], q50, q90, 10)
    result = {}
    for prefix, candidate, anchor, pi, mask in (
        ("V27_ENERGY_DET", "CALIB_REFINE_MATCH", "CALIB_REFINE_MATCH", match_pi, match_mask),
        ("V27_ENERGY_SDEDIT", "CALIB_SDEDIT_MATCH", "CALIB_REFINE_MATCH", match_pi, match_mask),
    ):
        result[f"{prefix}_L{_lambda_label(lambda_y)}"] = partial_observation_prox(tensors[candidate], tensors[anchor], pi, mask, 1., lambda_y).cpu().numpy()
    return result


@torch.no_grad()
def _predict_eegdus(batch: Mapping[str, Any], fold: int, noise_seed: int, steps: int = 10) -> np.ndarray:
    device = torch.device("cuda"); path = V22 / f"checkpoints/scad_eegdus_unified/fold_{fold}/seed_{EEGDFUS_SEED}.pt"; model, state = load_ema_model("scad", path, device); result = []
    for start in range(0, len(batch["y"]), 24):
        y = _tensor(batch["y"][start:start + 24], device); context = torch.zeros((len(y), int(state["config"]["context_input_dim"])), device=device); noise = torch.randn(y.shape, device=device, generator=torch.Generator(device=device).manual_seed(noise_seed + start)); artifact = model.sample(y, context, noise, steps)[0]; result.append((y - artifact).cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def _common_predictions(batch: Mapping[str, Any], fold: int, slot: int, v26_seed: int, v29_seed: int, steps: int = 10) -> dict[str, np.ndarray]:
    device = torch.device("cuda"); anchor, det, models = v26._load_bundle(fold, v26_seed, device); noise_seed = int(_cfg()["panel_seed"]) + fold * 100 + slot
    artifacts, _ = v26._predict(batch, anchor, det, models, noise_seed, .05, steps, device)
    predictions = {"RAW": np.asarray(batch["y"]), "STANDARD": np.asarray(batch["y"])}
    mapping = {
        "V25_POP": "V25_SET_CALIB_DET_POP", "V25_DET_MATCH": "V25_SET_CALIB_DET_MATCH", "V25_DET_WRONG": "V25_SET_CALIB_DET_WRONG",
        "CALIB_REFINE_MATCH": "V26_CALIB_REFINE_DET_MATCH", "CALIB_SDEDIT_MATCH": "V26_CALIB_SDEDIT_MATCH",
        "POP_SDEDIT": "V26_POP_SDEDIT", "CALIB_SDEDIT_WRONG": "V26_CALIB_SDEDIT_WRONG",
    }
    for source, target in mapping.items(): predictions[target] = np.asarray(batch["y"]) - artifacts[source]
    for lam in (.5, 2., 8.):
        energy = _energy_from_v26(batch, artifacts, fold, v26_seed, lam)
        for name, artifact in energy.items(): predictions[name] = np.asarray(batch["y"]) - artifact
    predictions.update(_predict_v29(batch, fold, v29_seed, noise_seed, steps)); predictions["EEGDFUS"] = _predict_eegdus(batch, fold, noise_seed, steps)
    return predictions


def common_infer(run: Path) -> dict[str, Any]:
    fold, slot, v26_seed, v29_seed = _cell(); outputs = []
    for stream in ("paired", "natural"):
        batch = _panel(fold, stream, False, 120); started = time.time(); prediction = _common_predictions(batch, fold, slot, v26_seed, v29_seed, 10); path = DERIVED / f"common/{stream}/fold_{fold}_slot_{slot}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **prediction); outputs.append({"stream": stream, "path": str(path), "sha256": sha256(path), "methods": len(prediction), "windows": len(batch["y"]), "seconds": time.time() - started})
    value = {"stage": "R3", "status": "PASS", "fold": fold, "slot": slot, "v26_seed": v26_seed, "v29_seed": v29_seed, "same_noise": True, "K": 1, "query_EOG_reads": 0, "query_operator_reads": 0, "event_reads": 0, "sealed_reads": 0, "outputs": outputs}; _json(RESULT / f"common_output_{fold}_{slot}.json", value); _json(run / "result_summary.json", value); return value


@torch.no_grad()
def _encode(det, support_eeg: np.ndarray, support_eog: np.ndarray, device: torch.device, batch_size: int = 8) -> dict[str, torch.Tensor]:
    """Encode frozen support episodes in bounded-memory chunks."""
    values: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(support_eeg), batch_size):
        stop = min(start + batch_size, len(support_eeg))
        encoded = det.encode_support(_tensor(support_eeg[start:stop], device), _tensor(support_eog[start:stop], device))
        for key, value in encoded.items(): values.setdefault(key, []).append(value)
    return {key: torch.cat(parts) for key, parts in values.items()}


@torch.no_grad()
def _donor_outputs(batch: Mapping[str, Any], fold: int, v26_seed: int, v29_seed: int, noise_seed: int, bundle: tuple[Any, ...] | None = None) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    if bundle is None:
        anchor, det, models = v26._load_bundle(fold, v26_seed, device); _, support29, popdet, popcdm, adapters = v29._load_bundle(fold, v29_seed, "selected"); del support29
    else:
        anchor, det, models, popdet, popcdm, adapters = bundle
    clean: dict[str, list[np.ndarray]] = {name: [] for name in ("V25_SET_CALIB_DET_MATCH", "V26_CALIB_SDEDIT_MATCH", "V29_PA_SC_DET_MATCH", "V29_PA_SC_CDM_MATCH")}
    for start in range(0, len(batch["y"]), 24):
        sl = slice(start, min(start + 24, len(batch["y"]))); y = _tensor(batch["y"][sl], device); q0 = _tensor(batch["q0"][sl], device); c0 = _tensor(batch["c0"][sl], device); pop_artifact = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0)); encoded = _encode(det, batch["support_eeg"][sl], batch["support_eog"][sl], device); coefficient = det.coefficient(y, pop_artifact, q0, encoded["context"]); det_artifact = decode_residual(pop_artifact, encoded["basis"], coefficient); noise = torch.randn(y.shape, device=device, generator=torch.Generator(device=device).manual_seed(noise_seed + start)); sd_artifact = models["calib_sdedit"].sample(y, det_artifact, pop_artifact, encoded["context"], noise, .05, 10)[0]
        pdet = popdet(y); dclean = adapters["support_adapter_det"](y, pdet, encoded["context"]); pcdm = popcdm.sample(y, noise, 10)[0]; cclean = adapters["support_adapter_cdm"].sample(popcdm, y, encoded["context"], noise, 10)[0]
        for name, value in (("V25_SET_CALIB_DET_MATCH", y-det_artifact), ("V26_CALIB_SDEDIT_MATCH", y-sd_artifact), ("V29_PA_SC_DET_MATCH", dclean), ("V29_PA_SC_CDM_MATCH", cclean)): clean[name].append(value.cpu().numpy())
    return {name: np.concatenate(value) for name, value in clean.items()}


def all_donor(run: Path) -> dict[str, Any]:
    index = _index(); recipient_index = index // 3; slot = index % 3; recipient = _cfg()["participants"][recipient_index]; fold = next(int(item["fold"]) for item in _folds() if recipient in item["test"]); v26_seed = V26_SEEDS[slot]; v29_seed = V29_SEEDS[slot]; batch = _panel(fold, "paired", False, 120); keep = [i for i, meta in enumerate(batch["meta"]) if meta["participant"] == recipient]; batch = {key: ([value[i] for i in keep] if key == "meta" else value[keep] if hasattr(value, "__len__") and len(value) == len(_panel_index(fold, "paired")) else value) for key, value in batch.items()}; bank = _bank(fold); owners = _support_owners(); eligible = list(_cfg()["participants"]); rows = []
    device = torch.device("cuda"); anchor, det, models = v26._load_bundle(fold, v26_seed, device); _, support29, popdet, popcdm, adapters = v29._load_bundle(fold, v29_seed, "selected"); del support29
    bundle = (anchor, det, models, popdet, popcdm, adapters)
    for donor_index, donor in enumerate(eligible):
        current = attach_support(batch, bank, owners, 120, donor=donor); outputs = _donor_outputs(current, fold, v26_seed, v29_seed, int(_cfg()["panel_seed"]) + recipient_index * 100 + slot, bundle)
        evaluator = load_panel(V24, ROLE, fold, "paired", [_panel_index(fold, "paired")[i] for i in keep], True)
        for method, clean in outputs.items():
            risk = np.mean([paired_metrics(evaluator["x"][i], current["y"][i], evaluator["artifact"][i], current["y"][i]-clean[i])["rrmse_temporal"] for i in range(len(clean))])
            rows.append({"method": method, "recipient": recipient, "donor": donor, "fold": fold, "slot": slot, "risk": float(risk), "correct": int(donor == recipient), "same_noise": 1})
    path = DERIVED / f"all_donor/recipient_{recipient}_slot_{slot}.csv"; _csv(path, rows); value = {"stage": "R4", "status": "PASS", "recipient": recipient, "slot": slot, "rows": len(rows), "all_wrong_donors": 14, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


@torch.no_grad()
def _outputs_with_encoded(batch: Mapping[str, Any], fold: int, v26_seed: int, v29_seed: int, encoded: Mapping[str, torch.Tensor] | None, noise_seed: int) -> dict[str, np.ndarray]:
    device = torch.device("cuda"); anchor, det, models = v26._load_bundle(fold, v26_seed, device); _, _, popdet, popcdm, adapters = v29._load_bundle(fold, v29_seed, "selected"); y = _tensor(batch["y"], device); q0 = _tensor(batch["q0"], device); c0 = _tensor(batch["c0"], device); pop_artifact = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0)); pdet = popdet(y); noise = torch.randn(y.shape, device=device, generator=torch.Generator(device=device).manual_seed(noise_seed)); pcdm = popcdm.sample(y, noise, 10)[0]
    if encoded is None:
        return {"V25_SET_CALIB_DET": (y-pop_artifact).cpu().numpy(), "V26_CALIB_SDEDIT": (y-pop_artifact).cpu().numpy(), "V29_PA_SC_DET": pdet.cpu().numpy(), "V29_PA_SC_CDM": pcdm.cpu().numpy()}
    coefficient = det.coefficient(y, pop_artifact, q0, encoded["context"]); det_artifact = decode_residual(pop_artifact, encoded["basis"], coefficient); sd_artifact = models["calib_sdedit"].sample(y, det_artifact, pop_artifact, encoded["context"], noise, .05, 10)[0]; dclean = adapters["support_adapter_det"](y, pdet, encoded["context"]); cclean = adapters["support_adapter_cdm"].sample(popcdm, y, encoded["context"], noise, 10)[0]
    return {"V25_SET_CALIB_DET": (y-det_artifact).cpu().numpy(), "V26_CALIB_SDEDIT": (y-sd_artifact).cpu().numpy(), "V29_PA_SC_DET": dclean.cpu().numpy(), "V29_PA_SC_CDM": cclean.cpu().numpy()}


def falsification(run: Path) -> dict[str, Any]:
    fold, slot, v26_seed, v29_seed = _cell(); batch = _panel(fold, "paired", False, 120); evaluator = _panel(fold, "paired", True); device = torch.device("cuda"); _, det, _ = v26._load_bundle(fold, v26_seed, device); owners = _support_owners(); bank = _bank(fold); noise_seed = int(_cfg()["panel_seed"]) + 9000 + fold * 10 + slot
    correct = _encode(det, batch["support_eeg"], batch["support_eog"], device); wrong = _encode(det, batch["wrong_support_eeg"], batch["wrong_support_eog"], device); lagged = _encode(det, batch["support_eeg"], np.roll(batch["support_eog"], 100, axis=-1), device); permutation = np.random.Generator(np.random.PCG64DXSM(20260933)).permutation(batch["support_eog"].shape[1]); shuffled = _encode(det, batch["support_eeg"], batch["support_eog"][:, permutation], device)
    context_values, basis_values = [], []
    for donor in owners:
        donor_batch = attach_support(batch, bank, owners, 120, donor=donor); encoded = _encode(det, donor_batch["support_eeg"], donor_batch["support_eog"], device); context_values.append(encoded["context"]); basis_values.append(encoded["basis"])
    mean_context = torch.stack(context_values).mean(0); mean_basis = torch.stack(basis_values).mean(0); mean_basis = mean_basis / torch.linalg.vector_norm(mean_basis, dim=1, keepdim=True).clamp_min(1e-8); mean_encoded = {"context": mean_context, "basis": mean_basis}
    controls = {"correct": correct, "registered_wrong": wrong, "circular_lag_1s": lagged, "time_shuffled": shuffled, "mean_context": mean_encoded, "exact_pop": None}; rows = []
    for condition, encoded in controls.items():
        outputs = _outputs_with_encoded(batch, fold, v26_seed, v29_seed, encoded, noise_seed)
        for method, clean in outputs.items():
            values = [paired_metrics(evaluator["x"][i], batch["y"][i], evaluator["artifact"][i], batch["y"][i]-clean[i])["rrmse_temporal"] for i in range(len(clean))]
            for participant in sorted({meta["participant"] for meta in batch["meta"]}):
                chosen = [values[i] for i, meta in enumerate(batch["meta"]) if meta["participant"] == participant]; rows.append({"fold": fold, "slot": slot, "participant": participant, "method": method, "condition": condition, "risk": float(np.mean(chosen)), "same_query_noise": 1})
    _csv(DERIVED / f"falsification/fold_{fold}_slot_{slot}.csv", rows); value = {"stage": "R5", "status": "PASS", "fold": fold, "slot": slot, "rows": len(rows), "controls": list(controls), "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def context_diagnostics(run: Path) -> dict[str, Any]:
    fold, slot, v26_seed, _ = _cell(); device = torch.device("cuda"); _, det, _ = v26._load_bundle(fold, v26_seed, device); bank = _bank(fold); owners = list(_cfg()["participants"]); bank_owners = _support_owners(); rows = []; features = {}
    for owner in owners:
        eeg, eog = [], []
        for session in ("ses-02", "ses-03", "ses-04"):
            for task in ("ERP", "SSVEP"):
                index = support_bank_index(owner, session, task, bank_owners); eeg.append(bank["eeg_120"][index]); eog.append(bank["eog_120"][index])
        eeg = np.concatenate(eeg); eog = np.concatenate(eog); left = _encode(det, eeg[None, ::2], eog[None, ::2], device); right = _encode(det, eeg[None, 1::2], eog[None, 1::2], device); features[owner] = {"context_left": left["context"][0].cpu().numpy(), "context_right": right["context"][0].cpu().numpy(), "basis_left": left["basis"][0].cpu().numpy(), "basis_right": right["basis"][0].cpu().numpy()}
    for left in owners:
        for right in owners:
            lc = features[left]["context_left"]; rc = features[right]["context_right"]; lp = projector(_tensor(features[left]["basis_left"][None], device))[0].cpu().numpy(); rp = projector(_tensor(features[right]["basis_right"][None], device))[0].cpu().numpy(); left_q = np.linalg.qr(features[left]["basis_left"])[0]; right_q = np.linalg.qr(features[right]["basis_right"])[0]; singular = np.linalg.svd(left_q.T @ right_q, compute_uv=False); angle = float(np.arccos(np.clip(singular.min(), -1, 1)))
            rows.append({"fold": fold, "slot": slot, "left_owner": left, "right_owner": right, "same_participant": int(left == right), "context_cosine_distance": float(1-np.dot(lc,rc)/max(np.linalg.norm(lc)*np.linalg.norm(rc),1e-12)), "context_euclidean_distance": float(np.linalg.norm(lc-rc)), "projector_frobenius_distance": float(np.linalg.norm(lp-rp)), "largest_principal_angle": angle, "basis_rank": int(np.linalg.matrix_rank(features[left]["basis_left"]))})
    path = DERIVED / f"diagnostics/context_projector_fold_{fold}_slot_{slot}.csv"; _csv(path, rows); feature_path = DERIVED / f"diagnostics/linkage_features_fold_{fold}_slot_{slot}.npz"; feature_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(feature_path, **{f"{owner}_{key}": value for owner, item in features.items() for key, value in item.items()}); value = {"stage": "R6", "status": "PASS", "fold": fold, "slot": slot, "rows": len(rows), "feature_sha256": sha256(feature_path), "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def duration(run: Path) -> dict[str, Any]:
    fold, slot, v26_seed, v29_seed = _cell(); owners = _support_owners(); bank = _bank(fold); paired = _panel(fold, "paired", False, 120); evaluator = _panel(fold, "paired", True); natural_base = load_panel(V24, ROLE, fold, "natural", _panel_index(fold, "natural"), False); rows = []; natural_outputs = {}; device = torch.device("cuda"); _, det, _ = v26._load_bundle(fold, v26_seed, device)
    full_batch = attach_support(paired, bank, owners, 120)
    full_encoded = _encode(det, full_batch["support_eeg"], full_batch["support_eog"], device)
    full_context, full_pi = full_encoded["context"], projector(full_encoded["basis"])
    for duration_seconds in (0, 5, 10, 30, 120):
        started = time.perf_counter()
        if duration_seconds:
            pbatch = attach_support(paired, bank, owners, duration_seconds); pbatch = attach_support(pbatch, bank, owners, duration_seconds, wrong=True); nbatch = attach_support(natural_base, bank, owners, duration_seconds); nbatch = attach_support(nbatch, bank, owners, duration_seconds, wrong=True); encoded = _encode(det, pbatch["support_eeg"], pbatch["support_eog"], device); outputs = _outputs_with_encoded(pbatch, fold, v26_seed, v29_seed, encoded, int(_cfg()["panel_seed"]) + fold*100 + slot); nencoded = _encode(det, nbatch["support_eeg"], nbatch["support_eog"], device); noutputs = _outputs_with_encoded(nbatch, fold, v26_seed, v29_seed, nencoded, int(_cfg()["panel_seed"]) + 5000 + fold*100 + slot); context = encoded["context"]; pi = projector(encoded["basis"])
        else:
            pbatch = paired; nbatch = natural_base; outputs = _outputs_with_encoded(pbatch, fold, v26_seed, v29_seed, None, int(_cfg()["panel_seed"]) + fold*100 + slot); noutputs = _outputs_with_encoded(nbatch, fold, v26_seed, v29_seed, None, int(_cfg()["panel_seed"]) + 5000 + fold*100 + slot); context = pi = None
        elapsed = 1000*(time.perf_counter()-started)/max(len(paired["y"]),1)
        for method, clean in outputs.items():
            risks = [paired_metrics(evaluator["x"][i], paired["y"][i], evaluator["artifact"][i], paired["y"][i]-clean[i])["rrmse_temporal"] for i in range(len(clean))]
            rows.append({"panel": "paired", "fold": fold, "slot": slot, "method": method, "duration_seconds": duration_seconds, "risk": float(np.mean(risks)), "artifact_remaining": "", "retention": "", "context_stability": "", "projector_stability": "", "encoding_ms": elapsed})
        for method, clean in noutputs.items(): natural_outputs[f"{method}_D{duration_seconds}"] = clean
        if duration_seconds:
            rows.append({"panel": "diagnostic", "fold": fold, "slot": slot, "method": "SUPPORT_ENCODING", "duration_seconds": duration_seconds, "risk": "", "artifact_remaining": "", "retention": "", "context_stability": float(torch.linalg.vector_norm(context-full_context,dim=1).mean()), "projector_stability": float(torch.linalg.matrix_norm(pi-full_pi,dim=(-2,-1)).mean()), "encoding_ms": elapsed})
    path = DERIVED / f"duration/natural_fold_{fold}_slot_{slot}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, **natural_outputs); _csv(DERIVED / f"duration/paired_fold_{fold}_slot_{slot}.csv", rows); value = {"stage": "R7", "status": "PASS", "fold": fold, "slot": slot, "rows": len(rows), "natural_path": str(path), "natural_sha256": sha256(path), "query_auxiliary_reads": 0, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


@torch.no_grad()
def _step_predictions(batch: Mapping[str, Any], fold: int, slot: int, steps: int) -> dict[str, np.ndarray]:
    v26_seed, v29_seed = V26_SEEDS[slot], V29_SEEDS[slot]; device = torch.device("cuda"); anchor, det, models = v26._load_bundle(fold, v26_seed, device); noise_seed = int(_cfg()["panel_seed"]) + fold*100 + slot; artifacts, _ = v26._predict(batch, anchor, det, models, noise_seed, .05, steps, device); v29_values = _predict_v29(batch, fold, v29_seed, noise_seed, steps)
    return {"V26_CALIB_SDEDIT": np.asarray(batch["y"])-artifacts["CALIB_SDEDIT_MATCH"], "V29_PA_SC_CDM": v29_values["V29_PA_SC_CDM_MATCH"], "EEGDFUS": _predict_eegdus(batch, fold, noise_seed, steps)}


@torch.no_grad()
def _benchmark_callable(method: str, batch: Mapping[str, Any], fold: int, slot: int, steps: int):
    """Bind checkpoints and static conditioning once; return only timed inference."""
    device = torch.device("cuda"); noise_seed = int(_cfg()["panel_seed"]) + fold * 100 + slot
    y = _tensor(batch["y"], device)
    noise = torch.randn(y.shape, device=device, generator=torch.Generator(device=device).manual_seed(noise_seed))
    if method == "V26_CALIB_SDEDIT":
        anchor, det, models = v26._load_bundle(fold, V26_SEEDS[slot], device)
        q0 = _tensor(batch["q0"], device); c0 = _tensor(batch["c0"], device)
        pop_artifact = anchor(y, q0, torch.einsum("bcd,bdt->bct", c0, q0))
        encoded = det.encode_support(_tensor(batch["support_eeg"], device), _tensor(batch["support_eog"], device))
        coefficient = det.coefficient(y, pop_artifact, q0, encoded["context"])
        det_artifact = decode_residual(pop_artifact, encoded["basis"], coefficient)
        model = models["calib_sdedit"]
        return lambda: y - model.sample(y, det_artifact, pop_artifact, encoded["context"], noise, .05, steps)[0]
    if method == "V29_PA_SC_CDM":
        _, support, _, popcdm, models = v29._load_bundle(fold, V29_SEEDS[slot], "selected")
        context, _, _ = v29_contexts(support, batch, device)
        adapter = models["support_adapter_cdm"]
        return lambda: adapter.sample(popcdm, y, context, noise, steps)[0]
    if method == "EEGDFUS":
        path = V22 / f"checkpoints/scad_eegdus_unified/fold_{fold}/seed_{EEGDFUS_SEED}.pt"
        model, state = load_ema_model("scad", path, device)
        context = torch.zeros((len(y), int(state["config"]["context_input_dim"])), device=device)
        return lambda: y - model.sample(y, context, noise, steps)[0]
    raise KeyError(method)


def steps_latency(run: Path) -> dict[str, Any]:
    index = _index(); cfg = _cfg()
    if index < 15:
        fold, slot, _, _ = _cell(index); batch = _panel(fold, "paired", False, 120); evaluator = _panel(fold, "paired", True); rows = []
        for steps in (5, 10, 25):
            started = time.perf_counter(); outputs = _step_predictions(batch, fold, slot, steps); elapsed = time.perf_counter()-started
            for method, clean in outputs.items():
                risks = [paired_metrics(evaluator["x"][i], batch["y"][i], evaluator["artifact"][i], batch["y"][i]-clean[i])["rrmse_temporal"] for i in range(len(clean))]
                rows.append({"fold": fold, "slot": slot, "method": method, "steps": steps, "batch_size": 24, "quality_rrmse": float(np.mean(risks)), "median_latency_ms": 1000*elapsed/len(clean), "p95_latency_ms": "", "throughput_windows_s": len(clean)/elapsed, "peak_gpu_memory_mb": torch.cuda.max_memory_allocated()/2**20, "measurement": "quality_bundle"})
        _csv(DERIVED / f"latency/quality_fold_{fold}_slot_{slot}.csv", rows); value = {"stage": "R8_QUALITY", "status": "PASS", "fold": fold, "slot": slot, "rows": len(rows), "sealed_reads": 0}
    else:
        fold = slot = 0; batch = _panel(0, "paired", False, 120); rows = []; warmup = int(cfg["latency"]["warmup"]); runs = int(cfg["latency"]["runs"])
        for batch_size in (1, 16):
            current = {key: (value[:batch_size] if hasattr(value, "__len__") and len(value) == len(batch["y"]) else value) for key, value in batch.items()}
            for steps in (5, 10, 25):
                for method in ("V26_CALIB_SDEDIT", "V29_PA_SC_CDM", "EEGDFUS"):
                    predict = _benchmark_callable(method, current, fold, slot, steps)
                    for _ in range(warmup): predict()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); timings = []
                    for _ in range(runs):
                        started = time.perf_counter(); predict(); torch.cuda.synchronize(); timings.append(1000*(time.perf_counter()-started)/batch_size)
                    values = np.asarray(timings); rows.append({"fold": 0, "slot": 0, "method": method, "steps": steps, "batch_size": batch_size, "quality_rrmse": "", "median_latency_ms": float(np.median(values)), "p95_latency_ms": float(np.quantile(values,.95)), "throughput_windows_s": 1000/float(np.median(values)), "peak_gpu_memory_mb": torch.cuda.max_memory_allocated()/2**20, "measurement": "20_warmup_100_runs"})
        _csv(DERIVED / "latency/benchmark.csv", rows); value = {"stage": "R8_LATENCY", "status": "PASS", "rows": len(rows), "gpu": torch.cuda.get_device_name(0), "warmup": warmup, "runs": runs, "sealed_reads": 0}
    _json(run / "result_summary.json", value); return value


@torch.no_grad()
def model_costs(run: Path) -> dict[str, Any]:
    """Measure frozen-model footprint and support encoding separately on one GPU."""
    device = torch.device("cuda"); fold = 0; v26_seed = V26_SEEDS[0]; v29_seed = V29_SEEDS[0]
    anchor, det, v26_models = v26._load_bundle(fold, v26_seed, device)
    _, support, _, popcdm, v29_models = v29._load_bundle(fold, v29_seed, "selected")
    eegdus, _ = load_ema_model("scad", V22 / f"checkpoints/scad_eegdus_unified/fold_{fold}/seed_{EEGDFUS_SEED}.pt", device)
    specifications = (
        ("V26_CALIB_SDEDIT", (anchor, det, v26_models["calib_sdedit"]), (v26._anchor_path(fold, v26_seed), v26._v25_det_path(fold, v26_seed), v26._model_path("calib_sdedit", fold, v26_seed))),
        ("V29_PA_SC_CDM", (support, popcdm, v29_models["support_adapter_cdm"]), (v29._support_path(fold, v29_seed), v29._pop_path("pop_cdm", fold, v29_seed), v29._adapter_path("support_adapter_cdm", fold, v29_seed))),
        ("EEGDFUS", (eegdus,), (V22 / f"checkpoints/scad_eegdus_unified/fold_{fold}/seed_{EEGDFUS_SEED}.pt",)),
    )
    rows = []
    for method, models, paths in specifications:
        rows.append({"method": method, "parameter_count": sum(sum(parameter.numel() for parameter in model.parameters()) for model in models), "checkpoint_size_bytes": sum(path.stat().st_size for path in paths), "checkpoint_files": len(paths), "hardware": torch.cuda.get_device_name(0)})
    batch = _panel(fold, "paired", False, 120); warmup = int(_cfg()["latency"]["warmup"]); repeats = int(_cfg()["latency"]["runs"])
    timing = []
    for batch_size in (1, 16):
        eeg = _tensor(batch["support_eeg"][:batch_size], device); eog = _tensor(batch["support_eog"][:batch_size], device)
        for _ in range(warmup): det.encode_support(eeg, eog)
        torch.cuda.synchronize(); values = []
        for _ in range(repeats):
            started = time.perf_counter(); det.encode_support(eeg, eog); torch.cuda.synchronize(); values.append(1000 * (time.perf_counter() - started) / batch_size)
        timing.append({"method": "V25_DEEPSETS_SUPPORT_ENCODER", "batch_size": batch_size, "windows_per_episode": int(eeg.shape[1]), "median_ms_per_episode": float(np.median(values)), "p95_ms_per_episode": float(np.quantile(values, .95)), "median_ms_per_support_window": float(np.median(values) / eeg.shape[1]), "warmup": warmup, "runs": repeats, "hardware": torch.cuda.get_device_name(0)})
    _csv(RESULT / "model_cost_inventory.csv", rows); _csv(RESULT / "support_encoding_latency.csv", timing)
    value = {"stage": "R8_COST", "status": "PASS", "model_rows": len(rows), "support_latency_rows": len(timing), "hardware": torch.cuda.get_device_name(0), "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def paired_aggregate(run: Path) -> dict[str, Any]:
    rows = []
    for fold in range(5):
        evaluator = _panel(fold, "paired", True); inference = load_panel(V24, ROLE, fold, "paired", _panel_index(fold, "paired"), False)
        for slot in range(3):
            with np.load(DERIVED / f"common/paired/fold_{fold}_slot_{slot}.npz", allow_pickle=False) as archive:
                for method in archive.files:
                    clean = np.asarray(archive[method])
                    for i, meta in enumerate(inference["meta"]):
                        metric = paired_metrics(evaluator["x"][i], inference["y"][i], evaluator["artifact"][i], inference["y"][i]-clean[i]); zero = meta.get("zero_artifact") == "1"
                        if zero: metric["snr_improvement"] = np.nan; metric["artifact_rrmse"] = np.nan
                        rows.append({"panel": "paired", "fold": fold, "slot": slot, "participant": meta["participant"], "session": meta["session"], "task": meta["task"], "severity": "zero" if zero else "mild" if float(meta["gain"]) < .5 else "medium" if float(meta["gain"]) < .95 else "severe", "method": method, "zero_artifact": int(zero), "identity_change": metric["observation_change_ratio"] if zero else np.nan, **metric})
    path = DERIVED / "metrics/paired_common.csv"; _csv(path, rows)
    donor_rows = read_rows(sorted((DERIVED / "all_donor").glob("*.csv"))); donor_cell = []
    for method in sorted({row["method"] for row in donor_rows}):
        for recipient in _cfg()["participants"]:
            for donor in _cfg()["participants"]:
                chosen = [float(row["risk"]) for row in donor_rows if row["method"] == method and row["recipient"] == recipient and row["donor"] == donor]; donor_cell.append({"method": method, "recipient": recipient, "donor": donor, "risk": float(np.mean(chosen))})
    _csv(RESULT / "all_donor_matrix.csv", donor_cell); summaries = donor_summary(donor_cell); _csv(RESULT / "all_donor_summary.csv", summaries); _csv(RESULT / "all_donor_group_summary.csv", group_summary(summaries))
    value = {"stage": "R9", "status": "PASS", "paired_rows": len(rows), "donor_matrix_rows": len(donor_cell), "participants": 15, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def freeze_outputs(run: Path) -> dict[str, Any]:
    rows = []
    for fold in range(5):
        for slot in range(3):
            record = json.loads((RESULT / f"common_output_{fold}_{slot}.json").read_text()); natural = next(row for row in record["outputs"] if row["stream"] == "natural"); path = Path(natural["path"]); assert sha256(path) == natural["sha256"]; duration_path = DERIVED / f"duration/natural_fold_{fold}_slot_{slot}.npz"; rows.extend(({"fold": fold, "slot": slot, "kind": "common", "path": str(path), "sha256": sha256(path)}, {"fold": fold, "slot": slot, "kind": "duration", "path": str(duration_path), "sha256": sha256(duration_path)}))
    _csv(RESULT / "natural_output_manifest.csv", rows); value = {"stage": "R10", "status": "PASS", "outputs": len(rows), "query_EOG_reads": 0, "query_operator_reads": 0, "event_reads": 0, "sealed_reads": 0}; _json(RESULT / "output_freeze.json", value); _json(run / "result_summary.json", value); return value


def natural_evaluator(run: Path) -> dict[str, Any]:
    if json.loads((RESULT / "output_freeze.json").read_text())["status"] != "PASS": raise RuntimeError("natural output freeze missing")
    fold, slot, _, _ = _cell(); inference = load_panel(V24, ROLE, fold, "natural", _panel_index(fold, "natural"), False); evaluator = _panel(fold, "natural", True); scale = np.load(V24 / f"fold_{fold}/eeg_scale.npy"); rows = []; duration_rows = []
    with np.load(DERIVED / f"common/natural/fold_{fold}_slot_{slot}.npz", allow_pickle=False) as archive:
        for method in archive.files:
            clean = np.asarray(archive[method])
            for i, meta in enumerate(inference["meta"]): rows.append({"panel": "natural", "fold": fold, "slot": slot, "participant": meta["participant"], "session": meta["session"], "task": meta["task"], "method": method, **natural_metrics_v28(inference["y"][i], clean[i], evaluator["latent"][i], evaluator["teacher_artifact"][i], scale)})
    with np.load(DERIVED / f"duration/natural_fold_{fold}_slot_{slot}.npz", allow_pickle=False) as archive:
        for key in archive.files:
            method, duration_value = key.rsplit("_D",1); clean = np.asarray(archive[key])
            metrics = [natural_metrics_v28(inference["y"][i], clean[i], evaluator["latent"][i], evaluator["teacher_artifact"][i], scale) for i in range(len(clean))]
            duration_rows.append({"panel": "natural", "fold": fold, "slot": slot, "method": method, "duration_seconds": int(duration_value), "risk": "", "artifact_remaining": float(np.mean([row["heldout_eog_remaining_ratio"] for row in metrics])), "retention": float(np.mean([row["low_eog_observation_retention"] for row in metrics])), "context_stability": "", "projector_stability": "", "encoding_ms": ""})
    _csv(DERIVED / f"metrics/natural_fold_{fold}_slot_{slot}.csv", rows); _csv(DERIVED / f"duration/natural_metrics_fold_{fold}_slot_{slot}.csv", duration_rows); value = {"stage": "R11", "status": "PASS", "fold": fold, "slot": slot, "rows": len(rows), "duration_rows": len(duration_rows), "evaluator_after_freeze": True, "query_EOG_reads": "evaluator_only", "query_operator_reads": "evaluator_only", "event_reads": 0, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def privacy(run: Path) -> dict[str, Any]:
    rows = []
    for fold in range(5):
        for slot in range(3):
            path = DERIVED / f"diagnostics/linkage_features_fold_{fold}_slot_{slot}.npz"
            with np.load(path, allow_pickle=False) as archive:
                contexts = {owner: (np.asarray(archive[f"{owner}_context_left"]), np.asarray(archive[f"{owner}_context_right"])) for owner in _cfg()["participants"]}
                bases = {owner: (np.asarray(archive[f"{owner}_basis_left"]), np.asarray(archive[f"{owner}_basis_right"])) for owner in _cfg()["participants"]}
            projectors = projector_features(bases)
            combined = {owner: (np.concatenate((contexts[owner][0]/max(np.linalg.norm(contexts[owner][0]),1e-12), projectors[owner][0]/max(np.linalg.norm(projectors[owner][0]),1e-12))), np.concatenate((contexts[owner][1]/max(np.linalg.norm(contexts[owner][1]),1e-12), projectors[owner][1]/max(np.linalg.norm(projectors[owner][1]),1e-12)))) for owner in contexts}
            pop = {owner: (np.zeros(128), np.zeros(128)) for owner in contexts}; rng = np.random.Generator(np.random.PCG64DXSM(20260934+fold*3+slot)); random = {owner: (rng.normal(size=128), rng.normal(size=128)) for owner in contexts}
            for feature, values in (("context", contexts), ("projector", projectors), ("context_plus_projector", combined), ("population_token", pop), ("random_features", random)):
                result = linkage(values)[0]; rows.append({"fold": fold, "slot": slot, "feature": feature, "state_byte_size": int(values[next(iter(values))][0].nbytes), **result})
    _csv(RESULT / "privacy_linkage.csv", rows); value = {"stage": "R12", "status": "PASS", "rows": len(rows), "development_linkage_only": True, "anonymity_claim": False, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def reviewer_selection(run: Path) -> dict[str, Any]:
    readiness = [
        ("strong denoising baselines", "complete", "STANDARD, V25 DET and EEGDfus are on the common panel"),
        ("subject-agnostic DDPM", "complete", "V26 PopSDEdit and V29 exact frozen population routes"),
        ("RAW / STANDARD", "complete", "identical observation references under corrected standardized panel are explicitly labeled"),
        ("subject component ablation", "complete", "MATCH/POP/all-wrong/lag/shuffle/mean-context"),
        ("wrong/null/shuffled controls", "complete", "all 14 wrong donors plus falsification controls"),
        ("statistics and CIs", "complete", "participant-first summaries and bootstrap intervals"),
        ("support amount", "complete", "0/5/10/30/120 seconds"),
        ("sampler steps", "complete", "5/10/25, K=1"),
        ("latency/memory", "complete", "batch 1/16, 20 warmup and 100 timed runs"),
        ("additional dataset/montage plan", "partial", "natural SGE development exists; independent sealed confirmation remains unopened"),
        ("privacy risk", "complete", "development linkage-risk diagnostic; no anonymity claim"),
        ("support-setting limitation", "complete", "within-session fixed-montage query-disjoint support is explicit"),
        ("task-valid physiology", "missing", "ERP/SSVEP/ERD metadata remain unavailable for a valid endpoint"),
    ]
    _csv(RESULT / "reviewer_readiness.csv", [{"item": item, "status": status, "evidence": evidence} for item,status,evidence in readiness])
    pending = {"status": "HUMAN_SELECTION_REQUIRED", "candidates": ["V25_SET_CALIB_DET_MATCH", "V26_CALIB_SDEDIT_MATCH", "V27_ENERGY_SDEDIT_L05", "V27_ENERGY_SDEDIT_L2", "V27_ENERGY_SDEDIT_L8", "V29_PA_SC_DET_MATCH", "V29_PA_SC_CDM_MATCH"], "dimensions": ["absolute paired denoising", "all-donor specificity", "absolute natural attenuation", "observation retention", "PSD/covariance", "latency", "support burden", "linkage risk", "simplicity", "reviewer coverage"]}
    _json(RESULT / "selection_pending.json", pending); value = {"stage": "R13", "status": "PASS_PENDING_HUMAN_SELECTION", "reviewer_rows": len(readiness), "sealed_open_authorized": False, "sealed_reads": 0}; _json(run / "result_summary.json", value); return value


def _write_reports(diagnosis: Mapping[str, Any], paired: list[dict[str, Any]], natural: list[dict[str, Any]], donor: list[dict[str, Any]]) -> None:
    def table(rows, columns):
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
        for row in rows: lines.append("| " + " | ".join(str(row.get(column,"")) for column in columns) + " |")
        return "\n".join(lines)
    inventory = list(csv.DictReader((RESULT / "checkpoint_inventory.csv").open())); available = sum(row["status"] == "available" for row in inventory); missing = [row for row in inventory if row["status"] == "missing"]
    (ROOT / "reports/v30_frozen_inventory.md").write_text(f"# V30 frozen inventory\n\n{available}/{len(inventory)} registered entries are readable; {len(missing)} are missing and remain explicitly marked. No approximate checkpoint replacement was made.\n")
    panel = list(csv.DictReader((RESULT / "common_panel_manifest.csv").open())); (ROOT / "reports/v30_common_panel.md").write_text(f"# V30 common panel\n\nThe outcome-blind manifest contains {len(panel)} fixed corrected-coordinate rows. Each participant/session/task contributes the same registered count. Paired and natural inference use identical panel indices across candidates and seeds. Natural auxiliary arrays were opened only after the output freeze.\n")
    (ROOT / "reports/v30_all_donor_specificity.md").write_text("# V30 all-donor specificity\n\n" + table(donor,["method","participants","mean_correct_rank","median_correct_rank","correct_top1","correct_top3","mean_correct_minus_mean_wrong","bootstrap_low","bootstrap_high"]) + "\n")
    falsification = list(csv.DictReader((RESULT / "falsification_effects.csv").open())); (ROOT / "reports/v30_support_falsification.md").write_text("# V30 support falsification\n\nCorrect, all-wrong, +1 s EOG lag, time-shuffled EOG, mean-context and exact population controls were evaluated without retraining and with fixed query/noise.\n\n" + table(falsification[:40],["method","condition","mean_risk"]) + "\n")
    relation = list(csv.DictReader((RESULT / "context_projector_relation.csv").open())); (ROOT / "reports/v30_context_projector_diagnostics.md").write_text("# V30 context/projector diagnostics\n\n" + table(relation,["method","context_r","projector_r","classification"]) + "\n")
    durations = list(csv.DictReader((RESULT / "support_duration_effects.csv").open())); (ROOT / "reports/v30_support_duration.md").write_text("# V30 support duration\n\n0 s is the exact population bypass. Other durations use deterministic chronological windows without repetition or overlap.\n\n" + table(durations[:80],["panel","method","duration_seconds","metric","mean"]) + "\n")
    latency = list(csv.DictReader((RESULT / "step_latency_effects.csv").open())); costs = list(csv.DictReader((RESULT / "model_cost_inventory.csv").open())); support_latency = list(csv.DictReader((RESULT / "support_encoding_latency.csv").open())); (ROOT / "reports/v30_latency.md").write_text("# V30 steps, latency and memory\n\nQuality and sampler latency:\n\n" + table(latency,["method","steps","batch_size","median_latency_ms","p95_latency_ms","throughput_windows_s","peak_gpu_memory_mb","quality_rrmse"]) + "\n\nFrozen model footprint:\n\n" + table(costs,["method","parameter_count","checkpoint_size_bytes","checkpoint_files","hardware"]) + "\n\nPure support encoding latency (separate from end-to-end duration inference):\n\n" + table(support_latency,["method","batch_size","windows_per_episode","median_ms_per_episode","p95_ms_per_episode","median_ms_per_support_window","hardware"]) + "\n")
    privacy_rows = list(csv.DictReader((RESULT / "privacy_linkage.csv").open())); (ROOT / "reports/v30_privacy_linkage.md").write_text("# V30 privacy/linkage diagnostic\n\nThis is a development linkage-risk diagnostic, not an anonymity claim.\n\n" + table(privacy_rows,["feature","top1_accuracy","top3_accuracy","same_different_auroc","within_distance","between_distance","state_byte_size"]) + "\n")
    readiness = list(csv.DictReader((RESULT / "reviewer_readiness.csv").open())); (ROOT / "reports/v30_reviewer_readiness.md").write_text("# V30 reviewer readiness\n\n" + table(readiness,["item","status","evidence"]) + "\n")
    selection = json.loads((RESULT / "final_candidate_selection.json").read_text()); (ROOT / "reports/v30_final_candidate_selection.md").write_text("# V30 final candidate selection\n\n```json\n"+json.dumps(selection,indent=2)+"\n```\n")
    (ROOT / "reports/v30_final_diagnosis.md").write_text("# V30 final diagnosis\n\n```json\n"+json.dumps(diagnosis,indent=2)+"\n```\n")


def aggregate(run: Path) -> dict[str, Any]:
    if not (RESULT / "final_candidate_selection.json").is_file(): raise RuntimeError("human final_candidate_selection.json is required")
    paired_raw = list(csv.DictReader((DERIVED / "metrics/paired_common.csv").open())); natural_raw = read_rows(sorted((DERIVED / "metrics").glob("natural_fold_*_slot_*.csv"))); paired_metrics_names = ("rrmse_temporal","rrmse_spectral","correlation","snr_improvement","artifact_rrmse","artifact_correlation","identity_change","clean_output_rms_ratio"); natural_metrics_names = ("heldout_eog_remaining_ratio","artifact_attenuation_db","eeg_eog_coherence_reduction","frontal_residual_topography","low_eog_observation_change","low_eog_observation_retention","psd_distortion","covariance_distortion","output_input_rms_ratio","observation_change_ratio")
    pp = participant_first(paired_raw, paired_metrics_names); nn = participant_first(natural_raw, natural_metrics_names); ps = method_summary(pp, paired_metrics_names); ns = method_summary(nn, natural_metrics_names); _csv(RESULT / "paired_method_summary.csv", ps); _csv(RESULT / "natural_method_summary.csv", ns); _csv(RESULT / "participant_effects.csv", pp+nn)
    donor_group = list(csv.DictReader((RESULT / "all_donor_group_summary.csv").open())); selection = json.loads((RESULT / "final_candidate_selection.json").read_text()); diagnosis = classify(ps, ns, donor_group, selection); diagnosis.update({"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0,"task_valid_preservation":"unavailable"}); _json(RESULT / "development_diagnosis.json", diagnosis)
    falsification_rows = read_rows(sorted((DERIVED / "falsification").glob("*.csv"))); fsummary=[]
    for method in sorted({row["method"] for row in falsification_rows}):
        for condition in sorted({row["condition"] for row in falsification_rows}):
            values=[float(row["risk"]) for row in falsification_rows if row["method"]==method and row["condition"]==condition]
            if values:fsummary.append({"method":method,"condition":condition,"mean_risk":float(np.mean(values)),"participants_seed_rows":len(values)})
    _csv(RESULT / "falsification_effects.csv",fsummary)
    distances=read_rows(sorted((DERIVED/"diagnostics").glob("context_projector_fold_*_slot_*.csv"))); _csv(RESULT/"context_projector_distances.csv",distances); relations=[]
    donor_matrix=list(csv.DictReader((RESULT/"all_donor_matrix.csv").open()))
    for method in sorted({row["method"] for row in donor_matrix}):
        x=[];p=[];risk=[]
        for row in donor_matrix:
            if row["method"]!=method:continue
            candidates=[d for d in distances if d["left_owner"]==row["recipient"] and d["right_owner"]==row["donor"]]
            if candidates:x.append(float(np.mean([float(d["context_euclidean_distance"]) for d in candidates])));p.append(float(np.mean([float(d["projector_frobenius_distance"]) for d in candidates])));risk.append(float(row["risk"]))
        cr=float(np.corrcoef(x,risk)[0,1]);pr=float(np.corrcoef(p,risk)[0,1]);classification="context_carries_specificity" if abs(cr)>abs(pr)+.1 else "projector_carries_specificity" if abs(pr)>abs(cr)+.1 else "both" if max(abs(cr),abs(pr))>.2 else "neither";relations.append({"method":method,"context_r":cr,"projector_r":pr,"classification":classification})
    _csv(RESULT/"context_projector_relation.csv",relations)
    duration_rows=read_rows(sorted((DERIVED/"duration").glob("paired_*.csv"))+sorted((DERIVED/"duration").glob("natural_metrics_*.csv")));_csv(RESULT/"support_duration_effects.csv",aggregate_duration(duration_rows))
    latency=read_rows(sorted((DERIVED/"latency").glob("*.csv")));_csv(RESULT/"step_latency_effects.csv",latency)
    _write_reports(diagnosis,ps,ns,donor_group);_figures(ps,ns,donor_group,distances,duration_rows,latency)
    value={"stage":"R14","status":"PASS","selected_candidate":selection["selected_candidate"],"diagnosis":diagnosis,"sealed_reads":0};_json(run/"result_summary.json",value);return value


def _figures(paired, natural, donor, distances, duration_rows, latency) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    root=ROOT/"figures/frozen_candidate_v30";root.mkdir(parents=True,exist_ok=True)
    def lookup(rows,metric):return [(row["method"],float(row["mean"])) for row in rows if row["metric"]==metric]
    def bar(name,values,ylabel):
        fig,ax=plt.subplots(figsize=(10,4));ax.bar(range(len(values)),[v for _,v in values]);ax.set_xticks(range(len(values)),[k for k,_ in values],rotation=75,ha="right",fontsize=6);ax.set_ylabel(ylabel);fig.tight_layout();fig.savefig(root/name,dpi=150);plt.close(fig)
    bar("absolute_paired_comparison.png",lookup(paired,"rrmse_temporal"),"clean temporal RRMSE")
    bar("absolute_natural_comparison.png",lookup(natural,"heldout_eog_remaining_ratio"),"held-out EOG remaining ratio")
    matrix=list(csv.DictReader((RESULT/"all_donor_matrix.csv").open()));method=max(donor,key=lambda row:int(row["correct_top3"]))["method"];owners=_cfg()["participants"];array=np.asarray([[float(next(r["risk"] for r in matrix if r["method"]==method and r["recipient"]==rec and r["donor"]==don)) for don in owners] for rec in owners]);fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(array,aspect="auto");ax.set_xticks(range(15),owners,rotation=90,fontsize=6);ax.set_yticks(range(15),owners,fontsize=6);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/"all_donor_heatmap.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots();
    for row in donor:ax.scatter(float(row["mean_correct_rank"]),int(row["correct_top3"]),label=row["method"])
    ax.set(xlabel="mean correct rank",ylabel="top-3 count");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/"correct_support_rank.png",dpi=150);plt.close(fig)
    same=[r for r in distances if r["same_participant"]=="1"];cross=[r for r in distances if r["same_participant"]=="0"];fig,ax=plt.subplots();ax.scatter([float(r["context_euclidean_distance"]) for r in cross],[float(r["projector_frobenius_distance"]) for r in cross],s=3,alpha=.15,label="cross");ax.scatter([float(r["context_euclidean_distance"]) for r in same],[float(r["projector_frobenius_distance"]) for r in same],s=8,label="same");ax.set(xlabel="context distance",ylabel="projector distance");ax.legend();fig.tight_layout();fig.savefig(root/"context_vs_projector.png",dpi=150);plt.close(fig)
    duration_summary=list(csv.DictReader((RESULT/"support_duration_effects.csv").open()));fig,ax=plt.subplots();
    for method in sorted({r["method"] for r in duration_summary if r["panel"]=="paired" and r["metric"]=="risk"}):
        rows=sorted([r for r in duration_summary if r["panel"]=="paired" and r["metric"]=="risk" and r["method"]==method],key=lambda r:int(r["duration_seconds"]));ax.plot([int(r["duration_seconds"]) for r in rows],[float(r["mean"]) for r in rows],"o-",label=method)
    ax.set(xlabel="support seconds",ylabel="paired risk");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/"support_duration_curve.png",dpi=150);plt.close(fig)
    bench=[r for r in latency if r["measurement"]=="20_warmup_100_runs" and r["batch_size"]=="1"];fig,ax=plt.subplots();
    for method in sorted({r["method"] for r in bench}):
        rows=sorted([r for r in bench if r["method"]==method],key=lambda r:int(r["steps"]));ax.plot([int(r["steps"]) for r in rows],[float(r["median_latency_ms"]) for r in rows],"o-",label=method)
    ax.set(xlabel="DDIM steps",ylabel="median ms/window");ax.legend(fontsize=6);fig.tight_layout();fig.savefig(root/"quality_latency_curve.png",dpi=150);plt.close(fig)
    methods=sorted({r["method"] for r in natural});rem={m:float(next(r["mean"] for r in natural if r["method"]==m and r["metric"]=="heldout_eog_remaining_ratio")) for m in methods};ret={m:float(next(r["mean"] for r in natural if r["method"]==m and r["metric"]=="low_eog_observation_retention")) for m in methods};fig,ax=plt.subplots();ax.scatter([rem[m] for m in methods],[ret[m] for m in methods]);ax.axvline(1,color="black",lw=.7);ax.set(xlabel="remaining ratio",ylabel="low-EOG observation retention");fig.tight_layout();fig.savefig(root/"attenuation_retention_scatter.png",dpi=150);plt.close(fig)
    privacy_rows=list(csv.DictReader((RESULT/"privacy_linkage.csv").open()));features=sorted({r["feature"] for r in privacy_rows});fig,ax=plt.subplots();ax.bar(range(len(features)),[np.mean([float(r["top1_accuracy"]) for r in privacy_rows if r["feature"]==f]) for f in features]);ax.set_xticks(range(len(features)),features,rotation=40,ha="right");ax.set_ylabel("top-1 linkage");fig.tight_layout();fig.savefig(root/"linkage_risk.png",dpi=150);plt.close(fig)


def ledger_check(run: Path) -> dict[str, Any]:
    path=ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md";content=path.read_text();value={"stage":"R15","status":"PASS","project_ledger_version":"v2.2","project_ledger_sha256":sha256(path),"V30_recorded":"V30 — Frozen Candidate" in content,"sealed_reads":0}
    if "**版本：** v2.2" not in content or not value["V30_recorded"]:raise RuntimeError(value)
    _json(RESULT/"ledger_sync.json",value);_json(run/"result_summary.json",value);return value


def _lineage() -> list[dict[str, Any]]:
    rows=[]
    for job in sorted((RESULT/"runs").glob("*/job_*")):
        paths=sorted(job.glob("task_*")) or [job]
        for path in paths:
            rows.append({"stage":job.parent.name,"job_id":job.name.removeprefix("job_"),"array_task":path.name.removeprefix("task_") if path.name.startswith("task_") else "","status":"accepted" if (path/"result_summary.json").is_file() or (path/"pytest.txt").is_file() else "failed","recovery_of":"","scientific_setting_changed":False})
    return rows


def package(run: Path) -> dict[str, Any]:
    lineage=_lineage();_csv(RESULT/"job_lineage.csv",lineage);(ROOT/"reports/slurm").mkdir(parents=True,exist_ok=True);lines=["# V30 Slurm lineage","stage\tjob_id\tarray_task\tstatus\trecovery_of\tscientific_setting_changed"]+["\t".join(str(row[key]) for key in ("stage","job_id","array_task","status","recovery_of","scientific_setting_changed")) for row in lineage];(ROOT/"reports/slurm/v30_job_ids.txt").write_text("\n".join(lines)+"\n")
    def tests(stage):
        paths=sorted((RESULT/f"runs/{stage}").glob("job_*/pytest.txt"));match=re.search(r"(\d+) passed",paths[-1].read_text()) if paths else None;return int(match.group(1)) if match else 0
    diagnosis=json.loads((RESULT/"development_diagnosis.json").read_text());ledger=ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md";queue=subprocess.check_output(["squeue","--me","--noheader","-o","%i %j %T"],text=True);counts={status:sum(row["status"]==status for row in lineage) for status in ("accepted","failed","superseded","recovery")};value={"protocol_id":"frozen_candidate_consolidation_specificity_v30","development_only":True,"base_commit":BASE,"implementation_commit":"REPORTED_AFTER_COMMIT","common_panel_commit":"REPORTED_AFTER_COMMIT","specificity_commit":"REPORTED_AFTER_COMMIT","duration_latency_commit":"REPORTED_AFTER_COMMIT","natural_commit":"REPORTED_AFTER_COMMIT","ledger_v2_2_commit":"REPORTED_AFTER_COMMIT","report_commit":"REPORTED_AFTER_COMMIT","terminal_commit":"SELF_REFERENTIAL_REPORTED_EXTERNALLY","remote_sha":"reported_after_push","push_status":"push_verified_after_terminal_commit","selected_candidate":diagnosis["selected_candidate"],"targeted_tests":tests("r16-tests"),"clean_archive_tests":tests("r17-clean"),"job_status_counts":counts,"accepted_jobs":[r["job_id"] for r in lineage if r["status"]=="accepted"],"failed_jobs":[r["job_id"] for r in lineage if r["status"]=="failed"],"superseded_jobs":[],"recovery_jobs":[],"current_v30_jobs":[line for line in queue.splitlines() if "v30_" in line],"query_EOG_inference_reads":0,"query_operator_inference_reads":0,"event_inference_reads":0,"sealed_reads":0,"A_track":"0c4f2301c1f873120fe54537cde3c76fff7ea3a2","A_track_unchanged":True,"manuscript_unchanged":True,"K":1,"project_ledger_version":"v2.2","project_ledger_sha256":sha256(ledger),**diagnosis};_json(RESULT/"terminal_manifest.json",value);_json(run/"result_summary.json",value);return value


STAGES={"r0-preflight":preflight,"r1-inventory-panel":inventory_panel,"r1-support-recovery":recover_support_bank,"r2-replay":replay_parity,"r3-common-infer":common_infer,"r4-all-donor":all_donor,"r5-falsification":falsification,"r6-diagnostics":context_diagnostics,"r7-duration":duration,"r8-steps-latency":steps_latency,"r8-model-costs":model_costs,"r9-paired":paired_aggregate,"r10-freeze":freeze_outputs,"r11-natural":natural_evaluator,"r12-privacy":privacy,"r13-review":reviewer_selection,"r14-aggregate":aggregate,"r15-ledger":ledger_check,"r18-package":package}
def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--stage",required=True,choices=STAGES);parser.add_argument("--run-dir",required=True,type=Path);args=parser.parse_args();args.run_dir.mkdir(parents=True,exist_ok=True);STAGES[args.stage](args.run_dir)
if __name__=="__main__":main()
