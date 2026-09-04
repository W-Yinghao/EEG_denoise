#!/usr/bin/env python3
"""WAVE-6 E2 — full donor sweep of the calibration operator (GPU inference).

Frozen design: reports/prereg_wave6_propagation_FROZEN.md sections 1, 3, 7.

For each recipient episode/window the SAME input is restored with the guide
built from the recipient's own EB operator (OWN), from every eligible training
donor (DONOR_<id>, same session+task cell), and from the population operator
(POP).  The static calibration signature is held at the recipient's POPULATION
signature in every arm, so the only thing that changes across arms is the
propagation relation inside the guide.  OWN is therefore a NEW arm and is not
numerically the published MATCH_gated (which carries sig_gated).

modes
  probe   one unit, two donors, paired only, runs the five frozen QC gates and
          exits non-zero if any fails (so an sbatch afterok dependency is a real
          gate, not a formality)
  run     one (fold, seed) unit, paired + natural, resume-safe

Nothing is retrained: the frozen V44-S1 EMA checkpoints are loaded as-is.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from pf_common import OUT

V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
WAVE6 = OUT / "wave6"
UNITS = WAVE6 / "e2_units"
SEEDS = (20261201, 20261202, 20261203)
EPISODES = 8


def _v44():
    if str(V44_SRC) not in sys.path:
        sys.path.insert(0, str(V44_SRC))
    from eeg_scad.cli import run_v44 as up
    return up


def _setup(fold_id: int, seed: int):
    """Frozen assets for one unit: registry, operators, bank, model."""
    import torch
    up = _v44()
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    data, folds, _ = configs()
    fold = folds[fold_id]
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = up._gated_assets(registry30, eb120)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(EPISODES)

    source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                         / "train_curve.json").read_text())
    device = torch.device("cuda")
    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    return (up, data, fold, registry30, assets, bank, model, schedule, device)


MAX_OWN_OTHER = 5          # a participant has at most 6 cells (3 sessions x 2 tasks)


def _donor_arms(fold, assets, donors_limit: int | None):
    """Eligible donors = this fold's TRAINING participants (never a test one)."""
    have = {k[0] for k in assets}
    donors = [p for p in sorted(fold["train"]) if p in have]
    if donors_limit:
        donors = donors[:donors_limit]
    return donors


def _own_other_cells(assets, key):
    """The recipient's OWN calibration from a DIFFERENT (session, task) cell.

    Plan table row "本人另一记录或条件的校准": the condition that separates
    "same identity" from "compatible propagation relation" — without it the
    sweep cannot tell whether a close stranger beats the recipient's own less
    compatible recording.
    """
    return [k for k in sorted(assets) if k[0] == key[0] and k != key][:MAX_OWN_OTHER]


def _operator(assets, key, arm, donor):
    """Guide operator for one arm; None when this arm has no operator here."""
    if arm == "OWN":
        return assets[key]["C_gated"]
    if arm == "POP":
        return assets[key]["C0"]
    if arm.startswith("OWN_OTHER_"):
        others = _own_other_cells(assets, key)
        index = int(arm.rsplit("_", 1)[1])
        if index >= len(others):
            return None
        return assets[others[index]]["C_gated"]
    dkey = (donor, key[1], key[2])
    if dkey not in assets:
        return None
    return assets[dkey]["C_gated"]


def _donor_label(assets, key, arm, donor):
    """The real id this arm's operator actually came from (P3 addressing)."""
    if arm == "OWN":
        return key[0]
    if arm == "POP":
        return "POP"
    if arm.startswith("OWN_OTHER_"):
        others = _own_other_cells(assets, key)
        index = int(arm.rsplit("_", 1)[1])
        return "|".join(others[index]) if index < len(others) else "MISSING"
    return donor


def _paired_rows(up, fold_id, seed, assets, bank, model, schedule, device, donors):
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    drives = up._bank_drives(assets, bank)
    keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]
    arms = (["OWN"] + [f"OWN_OTHER_{k}" for k in range(MAX_OWN_OTHER)]
            + [f"DONOR_{d}" for d in donors] + ["POP"])
    rows = []
    for arm in arms:
        donor = arm[len("DONOR_"):] if arm.startswith("DONOR_") else None
        a0, sig, present = [], [], []
        for key, drive in zip(keys, drives):
            operator = _operator(assets, key, arm, donor)
            present.append(operator is not None)
            # a missing donor cell is filled with POP purely to keep the tensor
            # rectangular; its rows are never recorded (episodes are independent
            # inside sample_bank_eog, so the filler cannot touch other episodes)
            if operator is None:
                operator = assets[key]["C0"]
            a0.append(operator @ drive)
            sig.append(assets[key]["sig_pop"])
        output = up.sample_bank_eog(model, schedule, bank["y"], np.stack(a0),
                                    np.stack(sig), device, up.noise_seed(fold_id, seed))
        for i, (clean, observed, artifact, prediction, meta) in enumerate(zip(
                bank["x"], bank["y"], bank["artifact"], output, bank["meta"])):
            if not present[i]:
                continue
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"nonfinite output {arm} episode {i}")
            rows.append({
                "kind": "paired", "fold": fold_id, "seed": seed, "episode": i,
                "participant": meta["participant"], "session": meta["session"],
                "task": meta["task"], "arm": arm,
                "donor": _donor_label(assets, keys[i], arm, donor),
                "zero_artifact": meta["zero_artifact"], "gain": meta["gain"],
                **paired_metrics(clean, observed, artifact, observed - prediction)})
    return rows, arms


def _natural_rows(up, fold_id, seed, data, fold, registry30, assets, model,
                  schedule, device, donors):
    rows = []
    arms = (["OWN"] + [f"OWN_OTHER_{k}" for k in range(MAX_OWN_OTHER)]
            + [f"DONOR_{d}" for d in donors] + ["POP"])
    for participant, session, task in itertools.product(
            sorted(fold["test"]), data["sessions"], data["tasks"]):
        key = (participant, session, task)
        if key not in assets:
            continue
        windows = list(up._natural_windows(registry30, data, key))
        if not windows:
            continue
        y_stack = np.stack([w[1] for w in windows])
        drives = np.stack([w[2] for w in windows])
        starts = [w[0] for w in windows]
        # per-window EOG activity, for the E3 event labels (computed here so the
        # label and the restoration come from exactly the same window)
        activity = [{"veog_rms": float(np.sqrt(np.mean(d[0] ** 2))),
                     "heog_rms": float(np.sqrt(np.mean(d[1] ** 2))),
                     "eog_rms": float(np.sqrt(np.mean(d ** 2)))} for d in drives]
        for arm in arms:
            donor = arm[len("DONOR_"):] if arm.startswith("DONOR_") else None
            operator = _operator(assets, key, arm, donor)
            if operator is None:
                continue
            a0 = np.stack([operator @ d for d in drives])
            sig = np.stack([assets[key]["sig_pop"]] * len(windows))
            output = up.sample_bank_eog(model, schedule, y_stack, a0, sig, device,
                                        up.natural_noise_seed(fold_id, seed))
            for i, (y, drive, prediction) in enumerate(zip(y_stack, drives, output)):
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(f"nonfinite natural {arm} {key} {i}")
                rows.append({
                    "kind": "natural", "fold": fold_id, "seed": seed,
                    "participant": participant, "session": session, "task": task,
                    "start": starts[i], "arm": arm,
                    "donor": _donor_label(assets, key, arm, donor),
                    **activity[i],
                    **up._natural_metrics(y, drive, y - prediction)})
    return rows


def probe(fold_id: int = 0, seed: int = SEEDS[0]) -> None:
    """The five frozen QC gates. Exits non-zero on any failure."""
    up, data, fold, registry30, assets, bank, model, schedule, device = _setup(fold_id, seed)
    donors = _donor_arms(fold, assets, donors_limit=2)
    checks: dict[str, object] = {"fold": fold_id, "seed": seed, "donors": donors}

    rows, arms = _paired_rows(up, fold_id, seed, assets, bank, model, schedule,
                              device, donors)
    rows2, _ = _paired_rows(up, fold_id, seed, assets, bank, model, schedule,
                            device, donors)

    # P1 determinism: identical replay of the whole sweep
    own = [r for r in rows if r["arm"] == "OWN"]
    own2 = [r for r in rows2 if r["arm"] == "OWN"]
    p1 = len(own) == len(own2) and all(
        a["rrmse_temporal"] == b["rrmse_temporal"] for a, b in zip(own, own2))
    checks["P1_determinism"] = bool(p1)

    # P2 the manipulation acts
    own_by = {r["episode"]: r["rrmse_temporal"] for r in own}
    deltas = [abs(r["rrmse_temporal"] - own_by[r["episode"]])
              for r in rows if r["arm"].startswith("DONOR_")]
    checks["P2_max_abs_delta"] = float(max(deltas)) if deltas else 0.0
    p2 = bool(deltas) and max(deltas) > 1e-9

    # P3 real-id addressing
    test_set = set(fold["test"])
    p3 = all(
        (r["donor"] == r["participant"] if r["arm"] == "OWN" else True)
        and (r["donor"] == r["arm"][len("DONOR_"):] if r["arm"].startswith("DONOR_") else True)
        and (r["donor"] not in test_set if r["arm"].startswith("DONOR_") else True)
        # an OWN_OTHER row must carry a real cell of the SAME participant, and
        # must not be the recipient's own cell
        and (r["donor"].split("|")[0] == r["participant"]
             and r["donor"] != "|".join((r["participant"], r["session"], r["task"]))
             if r["arm"].startswith("OWN_OTHER_") else True)
        for r in rows)
    checks["P3_own_other_rows"] = sum(1 for r in rows if r["arm"].startswith("OWN_OTHER_"))
    checks["P3_real_id_addressing"] = bool(p3)

    # P4 operator sanity + a non-degenerate distance matrix
    ops = {k: assets[k]["C_gated"] for k in sorted(assets)}
    shapes_ok = all(o.shape == (46, 2) and np.isfinite(o).all() for o in ops.values())
    cells = [k for k in ops if k[1:] == (bank["meta"][0]["session"], bank["meta"][0]["task"])]
    dist = np.array([[np.linalg.norm(ops[a] - ops[b]) for b in cells] for a in cells])
    p4 = bool(shapes_ok and np.isfinite(dist).all()
              and np.allclose(np.diag(dist), 0)
              and float(dist.max()) > float(dist[dist > 0].min()) > 0)
    checks["P4_operator_sanity"] = p4
    checks["P4_distance_range"] = [float(dist[dist > 0].min()), float(dist.max())]

    # P5 magnitude sanity against the published MATCH_gated of the same unit
    published = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                            / "stage1_result.json").read_text())["rows"]
    per_pub: dict[str, list[float]] = {}
    for r in published:
        if r["condition"] == "MATCH_gated":
            per_pub.setdefault(r["participant"], []).append(r["rrmse_temporal"])
    pub_mean = float(np.mean([np.mean(v) for v in per_pub.values()]))
    per_own: dict[str, list[float]] = {}
    for r in own:
        per_own.setdefault(r["participant"], []).append(r["rrmse_temporal"])
    own_mean = float(np.mean([np.mean(v) for v in per_own.values()]))
    p5 = 0.5 * pub_mean <= own_mean <= 2.0 * pub_mean
    checks["P5_own_mean"] = own_mean
    checks["P5_published_match_gated_mean"] = pub_mean
    checks["P5_magnitude_sanity"] = bool(p5)

    checks["all_gates_pass"] = bool(p1 and p2 and p3 and p4 and p5)
    WAVE6.mkdir(parents=True, exist_ok=True)
    (WAVE6 / "e2_probe.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=1))
    if not checks["all_gates_pass"]:
        raise SystemExit("WAVE6 E2 PROBE GATE FAILED — fleet must not launch")


def run(fold_id: int, seed: int) -> None:
    UNITS.mkdir(parents=True, exist_ok=True)
    out = UNITS / f"fold_{fold_id}_seed_{seed}.json"
    if out.is_file():
        payload = json.loads(out.read_text())
        if payload.get("complete"):
            print(json.dumps({"skipped": str(out)}))
            return
    up, data, fold, registry30, assets, bank, model, schedule, device = _setup(fold_id, seed)
    donors = _donor_arms(fold, assets, None)
    paired, arms = _paired_rows(up, fold_id, seed, assets, bank, model, schedule,
                                device, donors)
    natural = _natural_rows(up, fold_id, seed, data, fold, registry30, assets,
                            model, schedule, device, donors)
    keys = {(r["kind"], r.get("participant"), r.get("session"), r.get("task"),
             r.get("episode", r.get("start")), r["arm"]) for r in paired + natural}
    payload = {"fold": fold_id, "seed": seed, "arms": arms, "donors": donors,
               "n_paired": len(paired), "n_natural": len(natural),
               "n_unique_keys": len(keys), "complete": True,
               "rows": paired + natural,
               "frozen": "reports/prereg_wave6_propagation_FROZEN.md#3"}
    out.write_text(json.dumps(payload, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("fold", "seed", "n_paired", "n_natural", "n_unique_keys")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "run"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    args = parser.parse_args()
    if args.mode == "probe":
        probe(args.fold, args.seed)
    else:
        run(args.fold, args.seed)


if __name__ == "__main__":
    main()
