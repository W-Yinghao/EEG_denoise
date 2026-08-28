#!/usr/bin/env python3
"""PAPER-FINAL S0 — pipeline sanity check + T6 scalp-map data + T6 waveform exemplar.

Sanity gate (ground rule 4): reproduce dev MATCH ~= 0.4310 and NO-guide ~= 0.5738 on
the standard dev episodes with the frozen fold models (all 15 fold-seed units, arms
MATCH_gated and NO_A0, frozen seeds/noise).  The same pass records per-channel RRMSE
(T6 scalp map needs a 46-vector the stored rows do not carry) and dumps the waveform
exemplar for the lowest-ID dev participant (T6.1), including a K=32 predictive band
under the dev-frozen T1 temperature.

Resume-safe: one JSON per fold-seed unit, skip-if-done.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pf_common import (ARRAYS, OUT, REPO, S1_SEEDS, SEED, STAGE1, Z, load_model,
                       participant_means, per_channel_rrmse, stat,
                       stored_stage1_rows)

UNIT_DIR = OUT / "s0_units"
TOL = 0.01
K_CHAINS = 32
EXEMPLAR_PARTICIPANT = "sub-02"   # lowest-ID dev participant (fold 0 test)


def run() -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import (_bank_drives, _gated_assets, noise_seed,
                                      sample_bank_eog)
    from eeg_scad.cli.run_v44_s2 import _posterior_variance
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    data, folds, _ = configs()
    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    ARRAYS.mkdir(parents=True, exist_ok=True)
    temperature = json.loads((OUT / "t1_temperature.json").read_text())["temperatures"]

    for fold in folds:
        fold_id = fold["fold"]
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        for seed in S1_SEEDS:
            out_path = UNIT_DIR / f"fold_{fold_id}_seed_{seed}.json"
            if out_path.is_file():
                print(json.dumps({"fold": fold_id, "seed": seed, "skipped": True}),
                      flush=True)
                continue
            model = load_model(fold_id, device, seed)
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            drives = _bank_drives(assets, bank)
            keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]
            outputs = {}
            for arm in ("MATCH_gated", "NO_A0"):
                a0_stack, sig_stack = [], []
                for key, drive in zip(keys, drives):
                    asset = assets[key]
                    a0 = (asset["C_gated"] @ drive if arm == "MATCH_gated"
                          else np.zeros((len(asset["C0"]), drive.shape[1])))
                    a0_stack.append(a0)
                    sig_stack.append(asset["sig_gated"])
                outputs[arm] = sample_bank_eog(model, schedule, bank["y"],
                                               np.stack(a0_stack), np.stack(sig_stack),
                                               device, noise_seed(fold_id, seed))
            rows, channel_rows = [], []
            for arm in ("MATCH_gated", "NO_A0"):
                for episode, (clean, observed, artifact, meta, key) in enumerate(
                        zip(bank["x"], bank["y"], bank["artifact"], bank["meta"], keys)):
                    prediction = outputs[arm][episode]
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError("nonfinite S0 output")
                    rows.append({"fold": fold_id, "seed": seed, "participant": key[0],
                                 "condition": arm, "zero_artifact": meta["zero_artifact"],
                                 **paired_metrics(clean, observed, artifact,
                                                  observed - prediction)})
                    channel_rows.append({"participant": key[0], "condition": arm,
                                         "per_channel_rrmse": per_channel_rrmse(
                                             np.asarray(clean, np.float64),
                                             np.asarray(prediction, np.float64)).tolist()})

            payload = {"fold": fold_id, "seed": seed, "rows": rows,
                       "channel_rows": channel_rows}

            # --- exemplar + K=32 band, once (fold 0, seed 20261201) ---------------
            if fold_id == 0 and seed == SEED:
                indices = [i for i, k in enumerate(keys)
                           if k[0] == EXEMPLAR_PARTICIPANT
                           and not bank["meta"][i]["zero_artifact"]]
                peaks = [float(np.max(np.abs(drives[i][0]))) for i in indices]
                clear = [i for i, p in zip(indices, peaks) if p >= 2.0]
                episode = (clear[0] if clear
                           else indices[int(np.argmax(peaks))])
                key = keys[episode]
                asset = assets[key]
                drive = drives[episode]
                post_var = _posterior_variance(registry30, eb120, fold)
                chain_out = []
                base_seed = noise_seed(fold_id, seed)
                for chain in range(K_CHAINS):
                    rng = np.random.default_rng(910000 + fold_id * 1000 + seed % 100
                                                + chain * 17)
                    operator = asset["C_gated"] + rng.standard_normal(
                        asset["C_gated"].shape) * np.sqrt(post_var[key])
                    a0 = (operator @ drive)[None]
                    chain_out.append(sample_bank_eog(
                        model, schedule, bank["y"][episode][None], a0,
                        asset["sig_gated"][None], device,
                        base_seed + 31 * (chain + 1))[0])
                ensemble = np.stack(chain_out)
                sigma = ensemble.std(axis=0, ddof=1).clip(1e-9)
                var_op = post_var[key] @ (np.asarray(drive, np.float64) ** 2)
                width80 = (Z[0.80] * temperature["INFL"]
                           * np.sqrt(sigma ** 2 + var_op))
                eeg_names = [str(v) for v in np.load(
                    Path(data["v19_derived_root"]) / "prepared" / key[0]
                    / f"{key[1]}_{key[2]}.npz")["eeg_names"]]
                wanted = [n for n in ("FP1", "FZ", "CZ") if n in eeg_names]
                channels = wanted if len(wanted) == 3 else eeg_names[:3]
                np.savez_compressed(
                    ARRAYS / "t6_waveform_exemplar_dev.npz",
                    participant=key[0], session=key[1], task=key[2],
                    episode_index=episode,
                    eeg_names=np.asarray(eeg_names),
                    fixed_channels=np.asarray(channels),
                    eog_drive=np.asarray(drive, np.float32),
                    contaminated=np.asarray(bank["y"][episode], np.float32),
                    reference=np.asarray(bank["x"][episode], np.float32),
                    linear_regression=np.asarray(
                        bank["y"][episode] - asset["C_gated"] @ drive, np.float32),
                    unguided=np.asarray(outputs["NO_A0"][episode], np.float32),
                    matched=np.asarray(outputs["MATCH_gated"][episode], np.float32),
                    band_mean=ensemble.mean(axis=0).astype(np.float32),
                    band_halfwidth_80=width80.astype(np.float32),
                    band_policy=np.asarray(
                        f"INFL temp {temperature['INFL']} z80 {Z[0.80]:.6f} K=32"))
                payload["exemplar"] = {"participant": key[0], "session": key[1],
                                       "task": key[2], "episode": int(episode),
                                       "channels": channels,
                                       "veog_drive_peak": float(np.max(np.abs(drive[0])))}

            out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
            print(json.dumps({"fold": fold_id, "seed": seed, "rows": len(rows)}),
                  flush=True)


def aggregate() -> None:
    files = sorted(UNIT_DIR.glob("fold_*_seed_*.json"))
    if len(files) != 15:
        raise SystemExit(f"expected 15 unit files, found {len(files)}")
    rows, channel_rows = [], []
    for path in files:
        payload = json.loads(path.read_text())
        rows += payload["rows"]
        channel_rows += payload["channel_rows"]
    match = participant_means(rows, "MATCH_gated")
    noa0 = participant_means(rows, "NO_A0")
    match_mean = float(np.mean(list(match.values())))
    noa0_mean = float(np.mean(list(noa0.values())))
    stored = stored_stage1_rows()
    stored_match = float(np.mean(list(participant_means(stored, "MATCH_gated").values())))
    stored_noa0 = float(np.mean(list(participant_means(stored, "NO_A0").values())))
    ok = abs(match_mean - 0.4310) <= TOL and abs(noa0_mean - 0.5738) <= TOL
    gain = stat([noa0[p] - match[p] for p in match])

    # T6 scalp map: per-channel improvement (matched - unguided) = per-channel RRMSE
    # of NO_A0 minus MATCH_gated, participant-first.  Per-episode per-channel RRMSE
    # explodes on episodes whose 512-sample clean window is nearly flat on a channel,
    # so the within-participant aggregation over episodes uses the MEDIAN.
    per_part: dict[str, dict[str, list[np.ndarray]]] = {}
    for row in channel_rows:
        per_part.setdefault(row["participant"], {}).setdefault(
            row["condition"], []).append(np.asarray(row["per_channel_rrmse"]))
    improvement = np.mean([np.median(v["NO_A0"], axis=0)
                           - np.median(v["MATCH_gated"], axis=0)
                           for v in per_part.values()], axis=0)
    exemplar = np.load(ARRAYS / "t6_waveform_exemplar_dev.npz", allow_pickle=False)
    np.savez_compressed(ARRAYS / "t6_scalp_improvement.npz",
                        improvement_noa0_minus_match=improvement.astype(np.float64),
                        eeg_names=exemplar["eeg_names"])
    decision = {
        "sanity": {"dev_MATCH_gated": match_mean, "target_MATCH": 0.4310,
                   "dev_NO_A0": noa0_mean, "target_NO_A0": 0.5738,
                   "stored_MATCH_gated": stored_match, "stored_NO_A0": stored_noa0,
                   "tolerance": TOL, "pass": bool(ok)},
        "gain_match_vs_noa0": gain,
        "scalp_map": {"array": "paper_final_arrays/t6_scalp_improvement.npz",
                      "mean_improvement": float(improvement.mean()),
                      "max_channel": int(np.argmax(improvement))},
    }
    (OUT / "s0_sanity.json").write_text(json.dumps(decision, indent=2,
                                                   sort_keys=True) + "\n")
    print(json.dumps(decision["sanity"]))
    if not ok:
        raise SystemExit("SANITY GATE FAILED — fix the pipeline before new cells")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "aggregate"])
    args = parser.parse_args()
    (run if args.mode == "run" else aggregate)()


if __name__ == "__main__":
    main()
