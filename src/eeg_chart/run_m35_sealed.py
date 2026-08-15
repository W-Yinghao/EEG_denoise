"""M35 C-1: the sealed confirmation chain — MobileBCI sealed-8, opened ONCE.

Single isolated single-pass job.  Every choice is frozen in
reports/m35_preregistration.md (committed before any sealed byte was read):
V44-S1 5-fold output-ensemble (seed 20261201); dev-cohort population/gate
objects; V19 preparation replicated exactly for sealed subjects; carriers and
EOG donors from the development cohort; fixed seeds; outputs digest-frozen
before the evaluator opens sealed query EOG.  Every sealed path read is logged.
Reported regardless of outcome.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np


SEALED = ("sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22")
ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/flagship_m35/c1_sealed"
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/flagship_m35")
V19_PREPARED = Path("/projects/EEG-foundation-model/derived/denoiseNet/"
                    "counterfactual_operator_headroom_v19/prepared")
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
SAMPLER_SEED = 20269001
PAIRED_SEED_BASE = 421000
NATURAL_SEED_BASE = 611000
WINDOW = 512

sealed_reads: list[str] = []


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_sealed(data) -> Path:
    """Replicate the frozen V19 preparation for the 8 sealed subjects."""
    from eeg_cgdr.data.mobile_bci import read_source_eeg_eog
    from eeg_cgdr.experiments.counterfactual_operator_v19 import _preprocess

    import yaml
    v19_config = yaml.safe_load((ROOT / "configs/cgdr/counterfactual_operator_headroom_v19"
                                 ".yaml").read_text())
    bids_root = Path(v19_config["data_root"])
    target = DERIVED / "sealed_prepared"
    for participant in SEALED:
        out_dir = target / participant
        out_dir.mkdir(parents=True, exist_ok=True)
        for session, task in itertools.product(("ses-02", "ses-03", "ses-04"),
                                               ("ERP", "SSVEP")):
            out_path = out_dir / f"{session}_{task}.npz"
            if out_path.is_file():
                continue
            try:
                record = read_source_eeg_eog(bids_root, participant, session, task,
                                             allowlist=SEALED)
            except FileNotFoundError:
                continue
            sealed_reads.append(f"{participant}/{session}_{task} (source)")
            eeg_p, eog_p = _preprocess(record["eeg"], record["eog"],
                                       record["sampling_rate_hz"], v19_config)
            np.savez_compressed(out_path, eeg=eeg_p.astype(np.float32),
                                eog=eog_p.astype(np.float32),
                                eeg_names=np.asarray(record["eeg_names"]),
                                eog_names=np.asarray(record["eog_names"]),
                                sampling_rate=np.float64(100.0))
    return target


def build_merged_root(sealed_prepared: Path) -> Path:
    merged = DERIVED / "sealed_root" / "prepared"
    merged.mkdir(parents=True, exist_ok=True)
    for source_dir in sorted(V19_PREPARED.iterdir()):
        link = merged / source_dir.name
        if not link.exists():
            os.symlink(source_dir, link)
    for source_dir in sorted(sealed_prepared.iterdir()):
        link = merged / source_dir.name
        if not link.exists():
            os.symlink(source_dir, link)
    return merged.parent


def chain() -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry, bipolar_eog)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG
    from eeg_scad.cli.run_v44 import sample_bank_eog

    decision_path = RESULT / "decision.json"
    if decision_path.is_file():
        print(json.dumps({"skipped": "sealed chain already complete (single pass)"}))
        return
    freeze_path = RESULT / "sealed_outputs.npz"
    manifest_path = RESULT / "freeze_manifest.json"
    inference_needed = not manifest_path.is_file()
    if not inference_needed:
        # Single-pass discipline: inference already ran and was digest-frozen;
        # only the (previously crashed, never-executed) evaluator runs now.
        manifest = json.loads(manifest_path.read_text())
        if _sha(freeze_path) != manifest["digest_sha256"]:
            raise RuntimeError("sealed freeze digest mismatch — do not proceed")
    data, _, _ = configs()
    sealed_prepared = prepare_sealed(data)
    merged_root = build_merged_root(sealed_prepared)
    data = dict(data)
    data["v19_derived_root"] = str(merged_root)
    fold = {"fold": 99, "train": list(data["participants"]), "validation": [],
            "test": list(SEALED)}
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    for key in registry30.cells:
        if key[0] in SEALED:
            sealed_reads.append(f"{key[0]}/{key[1]}_{key[2]} (registry support+qgen)")
    sampler = TransferEpisodeSampler(data, fold, "test", SAMPLER_SEED, registry30)
    bank = sampler.sample_balanced(8)

    assets = {}
    for key in registry30.cells:
        if key[1:] not in registry30.population_transfer:
            continue
        pop = registry30.population_transfer[key[1:]]
        cell = eb120.cells[key]
        assets[key] = {
            "pinv_query": np.linalg.pinv(registry30.cells[key].query_transfer),
            "C_gated": pop + cell.lam * (cell.transfer - pop),
            "C0": pop,
            "sig_gated": eb120.signature(*key, "EB"),
            "sig_pop": registry30.signature(*key, "POP"),
        }
    natural_meta = []
    if inference_needed:
        device = torch.device("cuda")
        models = []
        for fold_id in range(5):
            source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_20261201"
                                 / "train_curve.json").read_text())
            model = CalibSADDPMEOG().to(device)
            model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                             weights_only=False)["ema"])
            model.eval()
            models.append(model)
        schedule = LinearX0Schedule().to(device)
        outputs_store = {}
        for subject_index, subject in enumerate(SEALED):
            indices = [i for i, m in enumerate(bank["meta"]) if m["participant"] == subject]
            if not indices:
                continue
            sub_y = np.stack([bank["y"][i] for i in indices])
            drives = [assets[(subject, bank["meta"][i]["session"], bank["meta"][i]["task"])]
                      ["pinv_query"] @ np.asarray(bank["artifact"][i], np.float64)
                      for i in indices]
            for arm in ("MATCH_gated", "NO_A0", "POP"):
                a0, sig = [], []
                for local, i in enumerate(indices):
                    key = (subject, bank["meta"][i]["session"], bank["meta"][i]["task"])
                    if arm == "MATCH_gated":
                        a0.append(assets[key]["C_gated"] @ drives[local])
                        sig.append(assets[key]["sig_gated"])
                    elif arm == "NO_A0":
                        a0.append(np.zeros((46, WINDOW)))
                        sig.append(assets[key]["sig_gated"])
                    else:
                        a0.append(assets[key]["C0"] @ drives[local])
                        sig.append(assets[key]["sig_pop"])
                ensemble = np.mean([sample_bank_eog(model, schedule, sub_y, np.stack(a0),
                                                    np.stack(sig), device,
                                                    PAIRED_SEED_BASE + subject_index)
                                    for model in models], axis=0)
                if not np.isfinite(ensemble).all():
                    raise FloatingPointError("nonfinite sealed output")
                outputs_store[f"paired_{subject}_{arm}"] = ensemble.astype(np.float32)
        for subject_index, subject in enumerate(SEALED):
            for session, task in itertools.product(data["sessions"], data["tasks"]):
                key = (subject, session, task)
                if key not in assets:
                    continue
                path = Path(merged_root) / "prepared" / subject / f"{session}_{task}.npz"
                with np.load(path, allow_pickle=False) as archive:  # EEG array only here
                    eeg = np.asarray(archive["eeg"], np.float64)
                sealed_reads.append(f"{subject}/{session}_{task} (natural eeg)")
                starts = np.linspace(int(data["qnatural_start"]), eeg.shape[1] - WINDOW, 4,
                                     dtype=int)
                ys = np.stack([(eeg[:, s:s + WINDOW] / registry30.eeg_scale[:, None])
                               .astype(np.float32) for s in starts])
                for arm in ("POP", "MATCH_gated"):
                    sig = assets[key]["sig_pop"] if arm == "POP" else assets[key]["sig_gated"]
                    a0 = np.zeros_like(ys)  # natural: no generative drive anchor this pass
                    ensemble = np.mean([sample_bank_eog(model, schedule, ys, a0,
                                                        np.stack([sig] * len(ys)), device,
                                                        NATURAL_SEED_BASE + subject_index)
                                        for model in models], axis=0)
                    outputs_store[f"natural_{subject}_{session}_{task}_{arm}"] = \
                        ensemble.astype(np.float32)
                natural_meta.append({"subject": subject, "session": session, "task": task,
                                     "starts": [int(s) for s in starts]})
        RESULT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(freeze_path, **outputs_store)
        manifest = {"digest_sha256": _sha(freeze_path), "frozen_before_evaluator": True,
                    "sealed_reads_log": sealed_reads,
                    "ensemble": "V44-S1 5-fold seed 20261201",
                    "natural_cells": natural_meta}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        natural_meta = manifest["natural_cells"]

    # Paired rows recomputed from the digest-verified frozen outputs.
    rows = []
    for clean, observed, artifact, meta in zip(bank["x"], bank["y"], bank["artifact"],
                                               bank["meta"]):
        rows.append({"participant": meta["participant"], "condition": "RAW",
                     "out_in_rms": 1.0, "zero_artifact": meta["zero_artifact"],
                     **paired_metrics(clean, observed, artifact, np.zeros_like(artifact))})
    with np.load(freeze_path, allow_pickle=False) as archive:
        for subject in SEALED:
            indices = [i for i, m in enumerate(bank["meta"])
                       if m["participant"] == subject]
            if not indices:
                continue
            for arm in ("MATCH_gated", "NO_A0", "POP"):
                ensemble = np.asarray(archive[f"paired_{subject}_{arm}"])
                for local, i in enumerate(indices):
                    prediction = ensemble[local]
                    rows.append({"participant": subject, "condition": arm,
                                 "out_in_rms": float(np.sqrt(np.mean(prediction ** 2))
                                                     / max(np.sqrt(np.mean(
                                                         np.asarray(bank["y"][i]) ** 2)),
                                                           1e-12)),
                                 "zero_artifact": bank["meta"][i]["zero_artifact"],
                                 **paired_metrics(bank["x"][i], bank["y"][i],
                                                  bank["artifact"][i],
                                                  bank["y"][i] - prediction)})
    manifest = json.loads(manifest_path.read_text())

    # ---------------- evaluation (post-freeze; sealed query EOG opens here) ----
    import pandas as pd
    from eeg_scad.cli.run_v43 import bootstrap_draws

    frame = pd.DataFrame(rows)
    per = {arm: frame[frame.condition == arm].groupby("participant").rrmse_temporal.mean()
           for arm in ("RAW", "MATCH_gated", "NO_A0", "POP")}
    participants = per["NO_A0"].index
    rms = frame[frame.condition == "NO_A0"].groupby("participant").out_in_rms.mean()
    q99 = float(rms.quantile(.99))
    precondition = {"noa0_beats_raw": bool((per["RAW"] - per["NO_A0"]).mean() > 0),
                    "noa0_minus_raw": float((per["NO_A0"] - per["RAW"]).mean()),
                    "q99": q99, "q99_in_band": bool(0.90 <= q99 <= 1.10)}
    precondition["pass"] = bool(precondition["noa0_beats_raw"]
                                and precondition["q99_in_band"])
    primary_delta = (per["NO_A0"] - per["MATCH_gated"]).loc[participants]
    draws = bootstrap_draws(primary_delta.to_numpy())
    primary = {"contrast": "MATCH_gated_minus_NO_A0_utility", "n": int(len(primary_delta)),
               "mean": float(primary_delta.mean()),
               "median": float(primary_delta.median()),
               "positive_count": int((primary_delta > 0).sum()),
               "bootstrap_low": float(np.quantile(draws, .025)),
               "bootstrap_high": float(np.quantile(draws, .975)),
               "adjudicable": precondition["pass"],
               "dev_reference": 0.14280771381963858}

    natural_rows = []
    with np.load(freeze_path, allow_pickle=False) as archive:
        for cell in natural_meta:
            subject, session, task = cell["subject"], cell["session"], cell["task"]
            key = (subject, session, task)
            path = Path(merged_root) / "prepared" / subject / f"{session}_{task}.npz"
            with np.load(path, allow_pickle=False) as raw_archive:
                eeg = np.asarray(raw_archive["eeg"], np.float64)
                eye = np.asarray(raw_archive["eog"], np.float64)
                names = [str(v) for v in raw_archive["eog_names"]]
            sealed_reads.append(f"{subject}/{session}_{task} (evaluator eog)")
            reg_cell = registry30.cells[key]
            for w, start in enumerate(cell["starts"]):
                eog = bipolar_eog(eye[:, start:start + WINDOW], names)
                latent = (eog - reg_cell.eog_center[:, None]) / reg_cell.eog_scale[:, None]
                teacher = reg_cell.query_transfer @ latent
                energy = np.sqrt(np.mean(latent * latent, axis=0))
                low = energy <= np.quantile(energy, .3)
                high = energy >= np.quantile(energy, .7)
                y = (eeg[:, start:start + WINDOW] / registry30.eeg_scale[:, None])
                for arm in ("POP", "MATCH_gated"):
                    output = np.asarray(archive[f"natural_{subject}_{session}_{task}_{arm}"]
                                        [w], np.float64)
                    estimate = y - output
                    remaining = float(np.linalg.norm(teacher[:, high] - estimate[:, high])
                                      / max(np.linalg.norm(teacher[:, high]), 1e-8))
                    natural_rows.append({
                        "participant": subject, "condition": arm,
                        "heldout_eog_remaining_ratio": remaining,
                        "artifact_attenuation_db": float(-20 * np.log10(max(remaining, 1e-8))),
                        "low_eog_observation_retention":
                            1 - float(np.linalg.norm(estimate[:, low])
                                      / max(np.linalg.norm(y[:, low]), 1e-8)),
                        "output_input_rms": float(np.sqrt(np.mean(output ** 2))
                                                  / max(np.sqrt(np.mean(y ** 2)), 1e-8))})
    nat = pd.DataFrame(natural_rows)
    nat_per = {arm: nat[nat.condition == arm].groupby("participant").mean(numeric_only=True)
               for arm in ("POP", "MATCH_gated")}
    second_row = {}
    for arm in ("POP", "MATCH_gated"):
        block = nat_per[arm]
        second_row[arm] = {"attenuation_db_mean": float(block.artifact_attenuation_db.mean()),
                           "retention_mean": float(block.low_eog_observation_retention.mean()),
                           "g4_bar": bool(block.artifact_attenuation_db.mean() > 0
                                          and block.low_eog_observation_retention.mean()
                                          >= 0.75)}
    common = nat_per["MATCH_gated"].index.intersection(nat_per["POP"].index)
    second_row["match_minus_pop_descriptive"] = {
        metric: float((nat_per["MATCH_gated"].loc[common, metric]
                       - nat_per["POP"].loc[common, metric]).mean())
        for metric in ("artifact_attenuation_db", "low_eog_observation_retention")}

    decision = {"preregistration": "reports/m35_preregistration.md",
                "stage": "C1_sealed_confirmation_single_pass",
                "precondition": precondition, "primary": primary,
                "natural_second_row": second_row,
                "condition_means": {arm: float(series.mean()) for arm, series in per.items()},
                "freeze": {"sha256": manifest["digest_sha256"],
                           "evaluator_opened_after_freeze": True},
                "sealed_reads_log": sealed_reads,
                "note": "natural arms carry a0=0 (no EOG anchor deployed on natural windows "
                        "in this pass); the paired construction is the adjudicating panel"}
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"precondition": precondition["pass"],
                      "primary_mean": primary["mean"],
                      "primary_ci": [primary["bootstrap_low"], primary["bootstrap_high"]],
                      "positive": primary["positive_count"]}))


if __name__ == "__main__":
    chain()
