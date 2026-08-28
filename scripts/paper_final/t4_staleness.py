#!/usr/bin/env python3
"""PAPER-FINAL T4 — operator lifetime / staleness curve.

cpu  : closed-form re-estimation of the propagation matrix on successive 120-s
       windows across each dev record (per-window robust EOG scaling, fold EEG
       scaling); relative RMS operator displacement vs window start time.
       Also stratifies the STORED V44-S1 natural rows by window start (the stored
       natural rows carry position metadata; the stored paired episodes do not).
gpu  : gain (matched - unguided) vs time-since-calibration on NEW paired episodes
       drawn from early/mid/late thirds of each record (same construction as the
       standard protocol, window starts restricted to the third; recipients, donors,
       gains and zero flags are identical across thirds because the rng call
       sequence is unchanged).  Seed-20261201 models only.  The injection operator
       remains the 150-270-s Qgen fit in all thirds (protocol-fixed); elapsed time
       enters through the EOG and carrier content.
Resume-safe per fold (gpu).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pf_common import (ARRAYS, OUT, SEED, load_model, participant_means, stat,
                       stored_stage1_natural_rows)

UNIT_DIR = OUT / "t4_units"
WINDOW_SECONDS = 120
THIRDS = (0, 1, 2)


# ------------------------------------------------------------------ CPU part

def cpu() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import (TransferRegistry, bipolar_eog,
                                                      ridge_transfer)
    from eeg_scad.data.v24_coordinate_contract import robust_center_scale

    data, folds, _ = configs()
    rate = int(data.get("sampling_rate", 100))
    span = WINDOW_SECONDS * rate
    displacement_rows = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, .05)
        for key in registry30.cells:
            if key[0] not in fold["test"]:
                continue
            eeg, eye, names = registry30._load(*key)
            eog = bipolar_eog(eye, names)
            length = min(eeg.shape[1], eog.shape[1])
            fits = {}
            for start in range(0, length - span + 1, span):
                seg_eog = eog[:, start:start + span]
                center, scale = robust_center_scale(seg_eog)
                latent = (seg_eog - center[:, None]) / scale[:, None]
                scaled = eeg[:, start:start + span] / registry30.eeg_scale[:, None]
                fits[start], _ = ridge_transfer(scaled, latent, registry30.ridge_ratio)
            base = fits[0]
            base_norm = max(np.linalg.norm(base), 1e-12)
            for start, operator in sorted(fits.items()):
                displacement_rows.append({
                    "participant": key[0], "cell": "|".join(key),
                    "window_start_s": start // rate,
                    "relative_displacement": float(
                        np.linalg.norm(operator - base) / base_norm)})

    # elapsed-time stratification of the STORED natural rows (position metadata exists)
    natural = stored_stage1_natural_rows()
    cell_starts: dict[tuple, list[int]] = {}
    for r in natural:
        cell = (r["participant"], r["session"], r["task"])
        cell_starts.setdefault(cell, set()).add(r["start"])
    cell_starts = {c: sorted(s) for c, s in cell_starts.items()}
    natural_strata = {}
    for w in range(4):
        rows_w = [r for r in natural
                  if cell_starts[(r["participant"], r["session"], r["task"])]
                  .index(r["start"]) == w]
        match = participant_means(rows_w, "MATCH_gated", "attenuation_db")
        noa0 = participant_means(rows_w, "NO_A0", "attenuation_db")
        match_r = participant_means(rows_w, "MATCH_gated", "low_eog_observation_retention")
        noa0_r = participant_means(rows_w, "NO_A0", "low_eog_observation_retention")
        common = [p for p in match if p in noa0]
        natural_strata[f"position_{w}"] = {
            "mean_start_s": float(np.mean([r["start"] for r in rows_w]) / 100.0),
            "attenuation_gain_db": stat([match[p] - noa0[p] for p in common]),
            "retention_delta": stat([match_r[p] - noa0_r[p] for p in common]),
        }

    per_start = {}
    for row in displacement_rows:
        per_start.setdefault(row["window_start_s"], {}).setdefault(
            row["participant"], []).append(row["relative_displacement"])
    displacement_curve = {}
    for start_s, per_part in sorted(per_start.items()):
        values = [float(np.mean(v)) for v in per_part.values()]
        if len(values) >= 10:  # keep starts covered by most records
            displacement_curve[str(start_s)] = stat(values)
    decision = {"displacement_curve_by_window_start_s": displacement_curve,
                "natural_gain_by_elapsed_time": natural_strata,
                "note": "displacement relative to the 0-120-s calibration window; "
                        "natural strata use stored V44-S1 rows (3 seeds)"}
    (OUT / "t4_cpu.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v["mean"] for k, v in displacement_curve.items()}))


# ------------------------------------------------------------------ GPU part

def gpu(only_fold: int | None = None) -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import (_bank_drives, _gated_assets, noise_seed,
                                      sample_bank_eog)
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    class ThirdRestrictedSampler(TransferEpisodeSampler):
        third = 0

        def _window(self, value):
            length = int(self.data["window_samples"])
            low0 = int(self.data["qnatural_start"])
            hi0 = value.shape[1] - length
            span = max((hi0 - low0) / 3.0, 1.0)
            low = int(low0 + self.third * span)
            hi = int(low0 + (self.third + 1) * span)
            start = int(self.rng.integers(low, max(low + 1, hi)))
            return np.asarray(value[:, start:start + length], dtype=np.float64)

    data, folds, _ = configs()
    if only_fold is not None:
        folds = [fold for fold in folds if fold["fold"] == only_fold]
    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_id = fold["fold"]
        out_path = UNIT_DIR / f"fold_{fold_id}.json"
        if out_path.is_file():
            print(json.dumps({"fold": fold_id, "skipped": True}), flush=True)
            continue
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        model = load_model(fold_id, device, SEED)
        rows = []
        for third in THIRDS:
            sampler = ThirdRestrictedSampler(data, fold, "test", SEED + 3, registry30)
            sampler.third = third
            bank = sampler.sample_balanced(8)
            drives = _bank_drives(assets, bank)
            keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]
            for arm in ("MATCH_gated", "NO_A0"):
                a0_stack, sig_stack = [], []
                for key, drive in zip(keys, drives):
                    asset = assets[key]
                    a0 = (asset["C_gated"] @ drive if arm == "MATCH_gated"
                          else np.zeros((46, drive.shape[1])))
                    a0_stack.append(a0)
                    sig_stack.append(asset["sig_gated"])
                output = sample_bank_eog(model, schedule, bank["y"],
                                         np.stack(a0_stack), np.stack(sig_stack),
                                         device, noise_seed(fold_id, SEED) + third)
                for episode, (clean, observed, artifact, meta, key) in enumerate(
                        zip(bank["x"], bank["y"], bank["artifact"], bank["meta"],
                            keys)):
                    prediction = output[episode]
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError("nonfinite T4 output")
                    rows.append({"fold": fold_id, "participant": key[0],
                                 "third": third, "condition": arm,
                                 **paired_metrics(clean, observed, artifact,
                                                  observed - prediction)})
        out_path.write_text(json.dumps({"fold": fold_id, "rows": rows},
                                       indent=1, sort_keys=True) + "\n")
        print(json.dumps({"fold": fold_id, "rows": len(rows)}), flush=True)


def aggregate() -> None:
    files = sorted(UNIT_DIR.glob("fold_*.json"))
    if len(files) != 5:
        raise SystemExit(f"expected 5 fold files, found {len(files)}")
    rows = []
    for path in files:
        rows += json.loads(path.read_text())["rows"]
    gain_by_third = {}
    for third in THIRDS:
        third_rows = [r for r in rows if r["third"] == third]
        match = participant_means(third_rows, "MATCH_gated")
        noa0 = participant_means(third_rows, "NO_A0")
        gain_by_third[f"third_{third}"] = stat([noa0[p] - match[p] for p in match])
    cpu_payload = json.loads((OUT / "t4_cpu.json").read_text())
    decision = {"gain_by_record_third": gain_by_third, **cpu_payload}
    (OUT / "t4_staleness.json").write_text(json.dumps(decision, indent=2,
                                                      sort_keys=True) + "\n")
    displacement = cpu_payload["displacement_curve_by_window_start_s"]
    np.savez_compressed(
        ARRAYS / "t4_staleness.npz",
        displacement_start_s=np.asarray([int(k) for k in sorted(displacement,
                                                                key=int)]),
        displacement_mean=np.asarray([displacement[k]["mean"]
                                      for k in sorted(displacement, key=int)]),
        displacement_ci=np.asarray([[displacement[k]["bootstrap_low"],
                                     displacement[k]["bootstrap_high"]]
                                    for k in sorted(displacement, key=int)]),
        gain_third=np.asarray([gain_by_third[f"third_{t}"]["mean"] for t in THIRDS]),
        gain_third_ci=np.asarray([[gain_by_third[f"third_{t}"]["bootstrap_low"],
                                   gain_by_third[f"third_{t}"]["bootstrap_high"]]
                                  for t in THIRDS]))
    print(json.dumps({k: v["mean"] for k, v in gain_by_third.items()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "gpu", "aggregate"])
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "gpu":
        gpu(args.fold)
    else:
        {"cpu": cpu, "aggregate": aggregate}[args.mode]()


if __name__ == "__main__":
    main()
