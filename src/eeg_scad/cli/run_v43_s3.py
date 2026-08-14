"""V43-S3 execution CLI: natural-route repair, cross-panel floor, privacy onset.

Adjudication rules are frozen in the V43-S3 addendum of
reports/v43_preregistration.md.  The severity mixture is the ONE registered
change to the S2a recipe.  Frozen V42R/S1/S2 artifacts are read-only.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_scad.cli.run_v43 import (DERIVED, RESULT, REPORT, _load_s2_state, _meta_key,
                                  _participant_means, _stat, bootstrap_draws, configs, holm,
                                  noise_seed, train_gated)
from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler, TransferRegistry,
                                                  bipolar_eog, ridge_transfer)
from eeg_scad.data.v24_coordinate_contract import robust_center_scale
from eeg_scad.evaluation.paired_metrics import paired_metrics


S3_SEEDS = (20261201, 20261202, 20261203)
S2A_POP_REFERENCE = 0.526
NG2_MARGIN = 0.010
NG3_WRONG_MARGIN = 0.005
NG3_MATCH_MARGIN = 0.002
NG3_MATCH_UPPER = 0.005
FLOOR_DURATION_MARGIN = 0.002
HARD_GATE_MIN_SECONDS = 60
NATURAL_SEED_BASE = 610000
WINDOW = 512
ARMS3 = ("POP", "WRONG", "NO_TRANSFER_BRANCH", "MATCH_EB120", "WRONG_EB120")


class SeverityMixtureEpisodeSampler(TransferEpisodeSampler):
    """S2a episode sampler with the REGISTERED severity mixture:
    gain ~ 40% exactly 0, 60% LogUniform(0.05, 1.3).  Everything else
    (sources, windows, signatures, metadata) is the parent's behavior."""

    def sample(self, count, zero_proportion: float = 0.15, recipient_keys=None):
        arrays = {key: [] for key in ("x", "y", "artifact", "signature")}
        meta = []
        for index in range(count):
            if recipient_keys is None:
                recipient, session, task = self.recipients[int(self.rng.integers(len(self.recipients)))]
            else:
                recipient, session, task = recipient_keys[index % len(recipient_keys)]
            cell = self.registry.cells[(recipient, session, task)]
            clean_owner, clean_eeg, _ = self._source({recipient}, session, task)
            eog_owner, _, source_eog = self._source({recipient, clean_owner}, session, task)
            x = self._window(clean_eeg) / self.registry.eeg_scale[:, None]
            physical_eog = self._window(source_eog)
            latent = (physical_eog - cell.eog_center[:, None]) / cell.eog_scale[:, None]
            gain = float(np.exp(self.rng.uniform(np.log(0.05), np.log(1.3))))
            zero = bool(self.rng.random() < 0.40)
            artifact = cell.query_transfer @ (gain * latent)
            if zero:
                artifact = np.zeros_like(artifact)
            y = x + artifact
            signature = self.registry.signature(recipient, session, task, "MATCH")
            for key, value in (("x", x), ("y", y), ("artifact", artifact), ("signature", signature)):
                arrays[key].append(np.asarray(value, np.float32))
            meta.append({
                "participant": recipient, "session": session, "task": task,
                "clean_owner": clean_owner, "eog_owner": eog_owner,
                "operator_recipient": recipient,
                "wrong_owner": self.condition_signature(
                    {"participant": recipient, "session": session, "task": task}, "WRONG")[1],
                "strict_three_way": int(len({recipient, clean_owner, eog_owner}) == 3),
                "gain": gain, "zero_artifact": int(zero),
                "severity_mixture": "40pct_zero_60pct_loguniform_0.05_1.3",
                "query_transfer_source": "independent_Qgen", "query_transfer_in_model_condition": 0,
            })
        return {**{key: np.stack(value) for key, value in arrays.items()}, "meta": meta}


# ------------------------------------------------------------------ training

def stage3_train(fold_id: int, seed: int, updates: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule

    result_dir = RESULT / "stage3" / f"fold_{fold_id}_seed_{seed}"
    curve_path = result_dir / "train_curve.json"
    if curve_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "training already complete"}))
        return
    data, folds, training = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    torch.manual_seed(seed)
    np.random.seed(seed)
    runtime = DERIVED / "stage3" / f"fold_{fold_id}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=False)
    registry30 = TransferRegistry(data, fold, 30, .05)
    train_sampler = SeverityMixtureEpisodeSampler(data, fold, "train", seed + 1, registry30)
    validation_sampler = TransferEpisodeSampler(data, fold, "validation", seed + 2, registry30)
    validation_bank = validation_sampler.sample_balanced(2)
    state, index = _load_s2_state(fold_id)
    model = CalibSADDPMCond().to(device)
    schedule = LinearX0Schedule().to(device)
    curve = train_gated(model, schedule, train_sampler, validation_bank, validation_sampler,
                        device, seed, updates, training["effective_batch"],
                        training["validation_interval"], runtime, state, index)
    payload = torch.load(runtime / "best.pt", map_location="cpu", weights_only=False)
    result_dir.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "updates": updates,
        "recipe": "S2a + registered severity mixture (40% zero, 60% LogUniform(0.05,1.3))",
        "checkpoint": str(runtime / "best.pt"), "checkpoint_best_step": payload["step"],
        "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"],
        "training_curve": curve, "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "best_step": payload["step"],
                      "best_validation_pop_rrmse": payload["best_validation_pop_rrmse"]}))


# ------------------------------------------------------- paired floor re-check

def stage3_eval(fold_id: int, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule
    from eeg_scad.training.train_v42r import _conditions, sample_bank

    result_dir = RESULT / "stage3" / f"fold_{fold_id}_seed_{seed}"
    result_path = result_dir / "stage3_result.json"
    if result_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "eval already complete"}))
        return
    source = json.loads((result_dir / "train_curve.json").read_text())
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    state, index = _load_s2_state(fold_id)
    model = CalibSADDPMCond().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    ns = noise_seed(fold_id, seed)
    rows = []
    for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"], bank["meta"]):
        rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                     "condition": "RAW", "context_owner": "NONE",
                     "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    wrong_owners = [sampler.condition_signature(meta, "WRONG")[1] for meta in bank["meta"]]
    outputs, context = {}, {}
    for condition in ("POP", "WRONG", "NO_TRANSFER_BRANCH"):
        signature, owners = _conditions(sampler, bank["meta"], condition)
        context[condition] = owners
        outputs[condition] = sample_bank(model, schedule, bank["y"], signature, device, ns,
                                         transfer_enabled=condition != "NO_TRANSFER_BRANCH")
    recipient_keys = [_meta_key(meta) for meta in bank["meta"]]
    wrong_keys = ["|".join((owner, meta["session"], meta["task"]))
                  for owner, meta in zip(wrong_owners, bank["meta"])]
    for condition, keys, owners in (("MATCH_EB120", recipient_keys,
                                     [m["participant"] for m in bank["meta"]]),
                                    ("WRONG_EB120", wrong_keys, wrong_owners)):
        signature = np.stack([state["sig_eb120"][index[key]] for key in keys])
        context[condition] = owners
        outputs[condition] = sample_bank(model, schedule, bank["y"], signature, device, ns)
    for condition, output in outputs.items():
        for clean, observed, artifact, prediction, meta, owner in zip(
                bank["x"], bank["y"], bank["artifact"], output, bank["meta"], context[condition]):
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite V43-S3 output")
            rows.append({"fold": fold_id, "seed": seed, "participant": meta["participant"],
                         "condition": condition, "context_owner": owner,
                         "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                         **paired_metrics(clean, observed, artifact, observed - prediction)})
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "checkpoint": source["checkpoint"],
        "noise_seed": ns, "sealed_reads": 0, "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "rows": len(rows)}))


# --------------------------------------------------------- natural freeze/eval

def stage3_natural_freeze(fold_id: int, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import CalibSADDPMCond, LinearX0Schedule
    from eeg_scad.training.train_v42r import sample_bank

    result_dir = RESULT / "stage3" / f"fold_{fold_id}_seed_{seed}"
    freeze_path = result_dir / "natural_freeze.npz"
    manifest_path = result_dir / "natural_freeze.json"
    if manifest_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "freeze already complete"}))
        return
    source = json.loads((result_dir / "train_curve.json").read_text())
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    state, index = _load_s2_state(fold_id)
    root = Path(data["v19_derived_root"])
    ys, pop_sig, match_sig, meta = [], [], [], []
    for participant, session, task in itertools.product(fold["test"], data["sessions"], data["tasks"]):
        key = (participant, session, task)
        if key not in registry30.cells:
            continue
        path = root / "prepared" / participant / f"{session}_{task}.npz"
        # Governance boundary: the freeze opens only the EEG array.
        with np.load(path, allow_pickle=False) as archive:
            eeg = np.asarray(archive["eeg"], np.float64)
        starts = np.linspace(int(data["qnatural_start"]), eeg.shape[1] - WINDOW, 4, dtype=int)
        for start in starts:
            ys.append((eeg[:, start:start + WINDOW] / registry30.eeg_scale[:, None]).astype(np.float32))
            pop_sig.append(registry30.signature(*key, "POP"))
            match_sig.append(state["sig_eb120"][index["|".join(key)]])
            meta.append({"participant": participant, "session": session, "task": task,
                         "start": int(start)})
    model = CalibSADDPMCond().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    ns = NATURAL_SEED_BASE + fold_id * 100 + seed % 100
    y_stack = np.stack(ys)
    out_pop = sample_bank(model, schedule, y_stack, np.stack(pop_sig), device, ns)
    out_match = sample_bank(model, schedule, y_stack, np.stack(match_sig), device, ns)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(freeze_path, y=y_stack, pop=out_pop, match=out_match,
                        participant=np.asarray([m["participant"] for m in meta]),
                        session=np.asarray([m["session"] for m in meta]),
                        task=np.asarray([m["task"] for m in meta]),
                        start=np.asarray([m["start"] for m in meta]))
    manifest_path.write_text(json.dumps({
        "fold": fold_id, "seed": seed, "path": str(freeze_path), "rows": len(meta),
        "arms": ["POP", "MATCH_EB120"], "noise_seed": ns,
        "query_eog_inference_reads": 0, "evaluator_opened": False}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "frozen_windows": len(meta)}))


def stage3_natural_evaluate(fold_id: int, seed: int) -> None:
    from scipy import signal as scipy_signal

    result_dir = RESULT / "stage3" / f"fold_{fold_id}_seed_{seed}"
    manifest_path = result_dir / "natural_freeze.json"
    out_path = result_dir / "natural_result.json"
    if out_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "natural eval complete"}))
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest["evaluator_opened"]:
        raise RuntimeError("natural freeze is not immutable")
    data, folds, _ = configs()
    fold = folds[fold_id]
    registry = TransferRegistry(data, fold, 30, include_query_transfer=True)
    with np.load(manifest["path"], allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    root = Path(data["v19_derived_root"])
    rows = []
    for i in range(len(payload["y"])):
        participant = str(payload["participant"][i])
        session, task = str(payload["session"][i]), str(payload["task"][i])
        start = int(payload["start"][i])
        path = root / "prepared" / participant / f"{session}_{task}.npz"
        with np.load(path, allow_pickle=False) as archive:
            eye = np.asarray(archive["eog"], np.float64)
            names = [str(value) for value in archive["eog_names"]]
        eog = bipolar_eog(eye[:, start:start + WINDOW], names)
        cell = registry.cells[(participant, session, task)]
        latent = (eog - cell.eog_center[:, None]) / cell.eog_scale[:, None]
        teacher = cell.query_transfer @ latent
        energy = np.sqrt(np.mean(latent * latent, axis=0))
        low = energy <= np.quantile(energy, .3)
        high = energy >= np.quantile(energy, .7)
        y = payload["y"][i]
        for condition, key in (("POP", "pop"), ("MATCH_EB120", "match")):
            output = payload[key][i]
            estimate = y - output
            remaining = float(np.linalg.norm(teacher[:, high] - estimate[:, high])
                              / max(np.linalg.norm(teacher[:, high]), 1e-8))
            retention = 1 - float(np.linalg.norm(estimate[:, low]) / max(np.linalg.norm(y[:, low]), 1e-8))
            frequencies, p0 = scipy_signal.welch(y[:, low], fs=100,
                                                 nperseg=min(128, int(low.sum())), axis=-1)
            _, p1 = scipy_signal.welch(output[:, low], fs=100,
                                       nperseg=min(128, int(low.sum())), axis=-1)
            keep = (frequencies >= 1) & (frequencies <= 15)
            covariance = np.cov(y[:, low])
            rows.append({"fold": fold_id, "seed": seed, "participant": participant,
                         "session": session, "task": task, "condition": condition,
                         "heldout_eog_remaining_ratio": remaining,
                         "artifact_attenuation_db": float(-20 * np.log10(max(remaining, 1e-8))),
                         "low_eog_observation_retention": retention,
                         "psd_distortion": float(np.mean(np.abs(np.log(p0[:, keep] + 1e-8)
                                                                - np.log(p1[:, keep] + 1e-8)))),
                         "covariance_distortion": float(np.linalg.norm(np.cov(output[:, low]) - covariance)
                                                        / max(np.linalg.norm(covariance), 1e-8)),
                         "output_input_rms": float(np.sqrt(np.mean(output ** 2))
                                                   / max(np.sqrt(np.mean(y ** 2)), 1e-8)),
                         "evaluator_query_eog_reads": 1})
    manifest["evaluator_opened"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    out_path.write_text(json.dumps({"natural_metrics": rows,
                                    "evaluator_opened_after_freeze": True},
                                   indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "natural_rows": len(rows)}))


# --------------------------------------------------- S3c cross-panel probes

def _gate_lambda(tau2: float, within: float, support_seconds: float, threshold: float) -> tuple[float, bool]:
    if support_seconds < HARD_GATE_MIN_SECONDS or within > threshold:
        return 0.0, True
    return float(np.clip(tau2 / max(tau2 + within / 4.0, 1e-12), 0.0, 1.0)), False


def _subtraction_rows(cells: list[dict], panel: str, unit_key: str) -> dict[str, Any]:
    """cells: list of dicts with keys: unit, support_seconds, a_full, a_blocks,
    a_budget {d: operator}, a_query, episodes [(x, y, drive)]."""
    pop_by_unit = {}
    for cell in cells:
        others = [c["a_full"] for c in cells if c[unit_key] != cell[unit_key]]
        pop_by_unit[cell["unit"]] = np.mean(others, axis=0)
    withins = {cell["unit"]: float(np.mean(np.square(np.stack(cell["a_blocks"])
                                                     - cell["a_full"][None]))) for cell in cells}
    threshold = float(np.percentile(list(withins.values()), 95))
    rows = []
    for index, cell in enumerate(cells):
        pop = pop_by_unit[cell["unit"]]
        tau2 = float(np.mean([np.mean(np.square(c["a_full"] - pop)) for c in cells
                              if c[unit_key] != cell[unit_key]]))
        lam, gated = _gate_lambda(tau2, withins[cell["unit"]], cell["support_seconds"], threshold)
        c_gated = pop + lam * (cell["a_full"] - pop)
        donor = cells[(index + 1) % len(cells)]
        if donor[unit_key] == cell[unit_key]:
            donor = cells[(index + 2) % len(cells)]
        donor_lam, _ = _gate_lambda(tau2, withins[donor["unit"]], donor["support_seconds"], threshold)
        arms = {"C0": pop, "C_gated": c_gated, "C_wrong": donor["a_full"],
                "C_wrong_gated": pop + donor_lam * (donor["a_full"] - pop)}
        for d, operator in cell["a_budget"].items():
            lam_d, _ = _gate_lambda(tau2, withins[cell["unit"]], float(d), threshold)
            arms[f"GATED_{d}s"] = pop + lam_d * (operator - pop)
        for x, y, drive in cell["episodes"]:
            for arm, operator in arms.items():
                out = y - operator @ drive
                rows.append({"panel": panel, "unit": cell["unit"], "arm": arm,
                             "lambda": lam if arm == "C_gated" else np.nan,
                             "hard_gate": int(gated),
                             "rrmse_temporal": float(np.linalg.norm(out - x)
                                                     / max(np.linalg.norm(x), 1e-12))})
    frame = pd.DataFrame(rows)
    per = {arm: frame[frame.arm == arm].groupby("unit").rrmse_temporal.mean()
           for arm in frame.arm.unique()}
    units = per["C0"].index
    result = {"units": len(units), "hard_gate_fraction": float(frame.hard_gate.mean()),
              "arm_means": {arm: float(series.mean()) for arm, series in per.items()},
              "gain_c0_minus_gated_descriptive": _stat((per["C0"] - per["C_gated"]).loc[units]),
              "wrong_gated_minus_c0": {**_stat((per["C_wrong_gated"] - per["C0"]).loc[units]),
                                       "margin": NG3_WRONG_MARGIN,
                                       "pass": bool((per["C_wrong_gated"] - per["C0"]).mean()
                                                    <= NG3_WRONG_MARGIN)},
              "wrong_ungated_minus_c0_descriptive": _stat((per["C_wrong"] - per["C0"]).loc[units])}
    duration = {}
    for arm in per:
        if arm.startswith("GATED_"):
            delta = (per[arm] - per["C0"]).loc[units]
            duration[arm] = {**_stat(delta), "margin": FLOOR_DURATION_MARGIN,
                             "pass": bool(delta.mean() <= FLOOR_DURATION_MARGIN)}
    result["duration_flatness"] = duration
    return result


def stage3c() -> None:
    import yaml
    from eeg_cgdr.data.klados import load_klados_records

    target = RESULT / "stage3c_crosspanel"
    target.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load((root / "configs/cgdr/klados_v4.yaml").read_text())
    records = load_klados_records(config)
    cells = []
    for record in records:
        clean = np.asarray(record.clean, np.float64)
        cont = np.asarray(record.contaminated, np.float64)
        eog = np.stack((np.asarray(record.veog, np.float64).reshape(-1),
                        np.asarray(record.heog, np.float64).reshape(-1)))
        length = min(clean.shape[1], cont.shape[1], eog.shape[1])
        clean, cont, eog = clean[:, :length], cont[:, :length], eog[:, :length]
        half = length // 2
        fs = 200.0
        _, eeg_scale = robust_center_scale(cont[:, :half])
        clean_s, cont_s = clean / eeg_scale[:, None], cont / eeg_scale[:, None]
        center, scale = robust_center_scale(eog[:, :half])
        latent = (eog - center[:, None]) / scale[:, None]
        a_full = ridge_transfer(cont_s[:, :half], latent[:, :half], .05)[0]
        block = half // 4
        a_blocks = [ridge_transfer(cont_s[:, i * block:(i + 1) * block],
                                   latent[:, i * block:(i + 1) * block], .05)[0] for i in range(4)]
        budgets = {}
        for d in (10, 30, 60, 120):
            samples = int(d * fs)
            if samples <= half:
                budgets[d] = ridge_transfer(cont_s[:, :samples], latent[:, :samples], .05)[0]
        episodes = [(clean_s[:, s:s + WINDOW], cont_s[:, s:s + WINDOW], latent[:, s:s + WINDOW])
                    for s in range(half, length - WINDOW + 1, WINDOW)]
        subject = f"P{(record.record_id - 1) % 27 + 1:02d}"
        cells.append({"unit": f"rec{record.record_id:02d}", "subject": subject,
                      "support_seconds": half / fs, "a_full": a_full, "a_blocks": a_blocks,
                      "a_budget": budgets, "episodes": episodes})
    klados_result = _subtraction_rows(cells, "klados", "subject")

    import mne
    bci_root = Path("/projects/EEG-foundation-model/BCI-IV")
    rng = np.random.default_rng(20260815)
    cells = []
    for subject in range(1, 10):
        sessions = {}
        for run in (1, 2, 3):
            matches = sorted(bci_root.glob(f"B{subject:02d}{run:02d}T.gdf"))
            if not matches:
                continue
            raw = mne.io.read_raw_gdf(matches[0], preload=True, verbose="error")
            picks = np.asarray(raw.get_data(), np.float64) * 1e6
            eeg, eog = picks[:3], picks[3:6]
            eeg[np.isnan(eeg)] = 0.0
            eog[np.isnan(eog)] = 0.0
            try:
                events, _ = mne.events_from_annotations(raw, verbose="error")
                first = min(int(s) for s, _, _ in events)
            except Exception:
                first = eeg.shape[1] // 3
            sfreq = float(raw.info["sfreq"])
            sessions[run] = (eeg, eog, max(int(first - 5 * sfreq), int(30 * sfreq)), sfreq)
        if len(sessions) < 3:
            continue
        fs = sessions[1][3]
        support_eeg = np.concatenate([sessions[r][0][:, :sessions[r][2]] for r in (1, 2)], axis=1)
        support_eog = np.concatenate([sessions[r][1][:, :sessions[r][2]] for r in (1, 2)], axis=1)
        _, eeg_scale = robust_center_scale(support_eeg)
        center, scale = robust_center_scale(support_eog)
        # ridge_transfer requires exactly 2 regressors: reduce the 3 monopolar
        # EOG channels to the support-fit top-2 principal components.
        standardized = (support_eog - center[:, None]) / scale[:, None]
        pca = np.linalg.svd(np.cov(standardized), full_matrices=False)[0][:, :2].T
        standardize = lambda v: pca @ ((v - center[:, None]) / scale[:, None])
        support_scaled = support_eeg / eeg_scale[:, None]
        latent_support = standardize(support_eog)
        a_full = ridge_transfer(support_scaled, latent_support, .05)[0]
        quarter = support_scaled.shape[1] // 4
        a_blocks = [ridge_transfer(support_scaled[:, i * quarter:(i + 1) * quarter],
                                   latent_support[:, i * quarter:(i + 1) * quarter], .05)[0]
                    for i in range(4)]
        budgets = {}
        for d in (10, 30, 60, 120):
            samples = int(d * fs)
            if samples <= support_scaled.shape[1]:
                budgets[d] = ridge_transfer(support_scaled[:, :samples],
                                            latent_support[:, :samples], .05)[0]
        gen_eeg, gen_eog, _, _ = sessions[3]
        a_query = ridge_transfer(gen_eeg / eeg_scale[:, None], standardize(gen_eog), .05)[0]
        episodes = []
        for run in (1, 2):
            eeg, eog, stop, _ = sessions[run]
            query_eeg = eeg[:, stop:] / eeg_scale[:, None]
            query_latent = standardize(eog[:, stop:])
            energy = np.sqrt(np.mean(query_latent * query_latent, axis=0))
            starts = np.arange(0, query_eeg.shape[1] - WINDOW, WINDOW)
            if len(starts) < 4:
                continue
            window_energy = np.asarray([energy[s:s + WINDOW].mean() for s in starts])
            low = starts[window_energy <= np.quantile(window_energy, .3)]
            high = starts[window_energy >= np.quantile(window_energy, .7)]
            if len(low) == 0 or len(high) == 0:
                continue
            for s in low[:12]:
                donor = high[int(rng.integers(len(high)))]
                drive = query_latent[:, donor:donor + WINDOW]
                x = query_eeg[:, s:s + WINDOW]
                episodes.append((x, x + a_query @ drive, drive))
        cells.append({"unit": f"B{subject:02d}", "subject": f"B{subject:02d}",
                      "support_seconds": support_scaled.shape[1] / fs, "a_full": a_full,
                      "a_blocks": a_blocks, "a_budget": budgets, "episodes": episodes})
    bci_result = _subtraction_rows(cells, "bci2b", "subject")
    payload = {"klados": klados_result, "bci2b": bci_result,
               "note": "gain rows descriptive; floor rules at S2 margins; gate frozen (no retuning)"}
    (target / "crosspanel_floor.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({panel: {"gain": payload[panel]["gain_c0_minus_gated_descriptive"]["mean"],
                              "hard_gate_fraction": payload[panel]["hard_gate_fraction"],
                              "wrong_gated_pass": payload[panel]["wrong_gated_minus_c0"]["pass"]}
                      for panel in ("klados", "bci2b")}))


# ------------------------------------------------------- S3d privacy onset

def stage3d() -> None:
    from eeg_scad.cli.run_v43 import stage2c_privacy  # noqa: F401  (pattern reference)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.linkage_diagnostic import linkage

    data, folds, _ = configs()
    lambdas = (("0.05", 0.05), ("0.10", 0.10), ("0.15", 0.15), ("0.20", 0.20))
    rows = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        participants = sorted(set(fold["train"] + fold["validation"] + fold["test"]))
        halves = {}
        for participant in participants:
            key = next((k for k in sorted(registry30.cells) if k[0] == participant), None)
            if key is None:
                continue
            eeg, eye, names = registry30._load(*key)
            eog = bipolar_eog(eye, names)
            pair = []
            for start, stop in ((0, 6000), (6000, 12000)):
                center, scale = robust_center_scale(eog[:, start:stop])
                latent = (eog[:, start:stop] - center[:, None]) / scale[:, None]
                scaled_eeg = eeg[:, start:stop] / registry30.eeg_scale[:, None]
                transfer, diagnostics = ridge_transfer(scaled_eeg, latent, registry30.ridge_ratio)
                rms = np.sqrt(np.mean((eog[:, start:stop] - center[:, None]) ** 2, axis=1)).clip(1e-8)
                quality = np.array([np.log(rms[0]), np.log(rms[1]), diagnostics["fit_r2"],
                                    np.log1p(diagnostics["condition_number"])])
                pair.append((transfer, quality))
            halves[participant] = (pair, key)
        for label, lam in lambdas:
            features = {}
            for participant, (pair, key) in halves.items():
                pop_transfer = registry30.population_transfer[key[1:]]
                pop_quality = registry30.population_quality[key[1:]]
                feats = []
                for transfer, quality in pair:
                    clamped = np.clip(quality, eb120.quality_min, eb120.quality_max)
                    gated_transfer = pop_transfer + lam * (transfer - pop_transfer)
                    gated_quality = pop_quality + lam * (clamped - pop_quality)
                    continuous = ((registry30._continuous(gated_transfer, gated_quality)
                                   - registry30.continuous_center) / registry30.continuous_scale)
                    feats.append(continuous.reshape(-1))
                features[participant] = (feats[0], feats[1])
            metrics = linkage(features)[0]
            rows.append({"fold": fold["fold"], "lambda_label": label, "lambda_value": lam,
                         "state_bytes_float32": 46 * 53 * 4, **metrics})
    target = RESULT / "stage3d_privacy_onset"
    target.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "lambda_privacy_onset.csv", index=False)
    print(frame.groupby("lambda_label")[["top1_accuracy", "same_different_auroc"]]
          .mean().round(4).to_string())


# ---------------------------------------------------------------- aggregate

def aggregate_stage3() -> dict[str, Any]:
    rows, natural_rows = [], []
    for fold_id in range(5):
        for seed in S3_SEEDS:
            base = RESULT / "stage3" / f"fold_{fold_id}_seed_{seed}"
            rows += json.loads((base / "stage3_result.json").read_text())["rows"]
            natural_rows += json.loads((base / "natural_result.json").read_text())["natural_metrics"]
    frame = pd.DataFrame(rows)
    per = {condition: _participant_means(frame, condition)
           for condition in ("RAW",) + ARMS3}
    participants = per["POP"].index

    pop_draws = bootstrap_draws(per["POP"].to_numpy())
    ng2_threshold = S2A_POP_REFERENCE + NG2_MARGIN
    ng2 = {"pop_paired_mean": float(per["POP"].mean()), "reference_s2a": S2A_POP_REFERENCE,
           "threshold": ng2_threshold, "pass": bool(per["POP"].mean() <= ng2_threshold)}
    d_wrong = (per["WRONG_EB120"] - per["POP"]).loc[participants]
    reduction = (per["WRONG"] - per["WRONG_EB120"]).loc[participants]
    d_match = (per["MATCH_EB120"] - per["POP"]).loc[participants]
    draws_wrong = bootstrap_draws(d_wrong.to_numpy())
    draws_match = bootstrap_draws(d_match.to_numpy())
    reduction_stat = _stat(reduction)
    upper95 = float(np.quantile(draws_match, .95))
    ng3 = {"wrong_gated": {**_stat(d_wrong), "margin": NG3_WRONG_MARGIN,
                           "reduction_vs_ungated": reduction_stat},
           "match": {**_stat(d_match), "margin": NG3_MATCH_MARGIN,
                     "one_sided_upper95": upper95, "upper_margin": NG3_MATCH_UPPER},
           "pass": bool(d_wrong.mean() <= NG3_WRONG_MARGIN
                        and reduction_stat["bootstrap_low"] > 0
                        and d_match.mean() <= NG3_MATCH_MARGIN
                        and upper95 <= NG3_MATCH_UPPER)}

    nat = pd.DataFrame(natural_rows)
    nat_per = {condition: nat[nat.condition == condition].groupby("participant").mean(numeric_only=True)
               for condition in ("POP", "MATCH_EB120")}
    pop_nat = nat_per["POP"]
    remaining_mean = float(pop_nat.heldout_eog_remaining_ratio.mean())
    attenuation_mean = float(pop_nat.artifact_attenuation_db.mean())
    rms_q99 = float(pop_nat.output_input_rms.quantile(.99))
    ng1 = {"pop_remaining_mean": remaining_mean, "pop_attenuation_db_mean": attenuation_mean,
           "pop_output_input_rms_q99": rms_q99,
           "frozen_reference": {"remaining": 1.082, "attenuation_db": -0.133},
           "pass": bool(remaining_mean < 1 and attenuation_mean > 0 and rms_q99 < 3)}
    p_raw = {
        "N-G1": float(np.mean(bootstrap_draws(pop_nat.heldout_eog_remaining_ratio.to_numpy()) >= 1.0)),
        "N-G2": float(np.mean(pop_draws > ng2_threshold)),
        "N-G3": float(max(np.mean(draws_wrong >= NG3_WRONG_MARGIN),
                          np.mean(draws_match >= NG3_MATCH_MARGIN))),
    }
    p_adjusted = holm(p_raw)
    natural_utilities = None
    if ng1["pass"]:
        common = nat_per["MATCH_EB120"].index.intersection(pop_nat.index)
        natural_utilities = {
            metric: _stat((nat_per["MATCH_EB120"].loc[common, metric]
                           - pop_nat.loc[common, metric]) * direction)
            for metric, direction in (("heldout_eog_remaining_ratio", -1),
                                      ("artifact_attenuation_db", 1),
                                      ("low_eog_observation_retention", 1),
                                      ("psd_distortion", -1), ("covariance_distortion", -1))}
    crosspanel = json.loads((RESULT / "stage3c_crosspanel" / "crosspanel_floor.json").read_text())
    onset = pd.read_csv(RESULT / "stage3d_privacy_onset" / "lambda_privacy_onset.csv")
    onset_summary = onset.groupby("lambda_label")[["top1_accuracy", "same_different_auroc"]] \
        .mean().round(4).reset_index().to_dict("records")
    decision = {
        "preregistration": "reports/v43_preregistration.md (V43-S3 addendum)",
        "stage": "S3_natural_route_repair",
        "N-G1": ng1, "N-G2": ng2, "N-G3": ng3, "secondary_applied": False,
        "holm": {"p_raw": p_raw, "p_adjusted": p_adjusted, "alpha": 0.05},
        "natural_utilities_match_minus_pop": natural_utilities,
        "natural_condition_means": {
            condition: {metric: float(block[metric].mean())
                        for metric in ("heldout_eog_remaining_ratio", "artifact_attenuation_db",
                                       "low_eog_observation_retention", "output_input_rms")}
            for condition, block in nat_per.items()},
        "paired_condition_means": {condition: float(series.mean())
                                   for condition, series in per.items()},
        "crosspanel_floor": crosspanel, "privacy_onset": onset_summary,
        "s3b_triggered": bool(ng1["pass"]), "sealed_reads": 0,
    }
    (RESULT / "stage3" / "decision.json").write_text(json.dumps(decision, indent=2,
                                                                sort_keys=True) + "\n")
    (REPORT / "v43_stage3.md").write_text(
        "# V43 Stage 3 — natural-route repair\n\n"
        "Preregistration: V43-S3 addendum (frozen before submission). Registered severity-"
        "mixture repair on the S2a recipe; frozen V42R natural gate criteria unchanged.\n\n"
        f"Decision: N-G1 **{ng1['pass']}**, N-G2 **{ng2['pass']}**, N-G3 **{ng3['pass']}**; "
        f"S3b triggered: **{decision['s3b_triggered']}**.\n\n## Gates\n\n```json\n"
        + json.dumps({"N-G1": ng1, "N-G2": ng2, "N-G3": ng3, "holm": decision["holm"]},
                     indent=2, sort_keys=True) + "\n```\n\n"
        "## Natural utilities (MATCH_EB120 - POP, positive = better)\n\n```json\n"
        + json.dumps(natural_utilities, indent=2, sort_keys=True) + "\n```\n\n"
        "## Cross-panel floor (S3c)\n\n```json\n"
        + json.dumps(crosspanel, indent=2, sort_keys=True) + "\n```\n\n"
        "## Privacy onset grid (S3d)\n\n"
        + pd.DataFrame(onset_summary).to_markdown(index=False) + "\n")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("stage3c", "stage3d", "aggregate"):
        sub.add_parser(name)
    for name in ("stage3-train", "stage3-eval", "stage3-natural-freeze", "stage3-natural-evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--fold", type=int, required=True)
        p.add_argument("--seed", type=int, required=True)
        if name == "stage3-train":
            p.add_argument("--updates", type=int, required=True)
    args = parser.parse_args()
    if args.stage == "stage3-train":
        stage3_train(args.fold, args.seed, args.updates)
    elif args.stage == "stage3-eval":
        stage3_eval(args.fold, args.seed)
    elif args.stage == "stage3-natural-freeze":
        stage3_natural_freeze(args.fold, args.seed)
    elif args.stage == "stage3-natural-evaluate":
        stage3_natural_evaluate(args.fold, args.seed)
    elif args.stage == "stage3c":
        stage3c()
    elif args.stage == "stage3d":
        stage3d()
    else:
        print(json.dumps(aggregate_stage3(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
