#!/usr/bin/env python3
"""BCI-IV-2a route 2 — full method: a 22-channel population prior guided by the
subject's calibrated operator.

Training pairs are semi-synthetic, exactly as in the main paper: clean windows are
drawn from low-EOG-energy periods, an ocular drive from high-EOG-energy periods is
pushed through a GENERATIVE operator to contaminate them, and the guide is built from
the CALIBRATION operator, so the learnable residual is the operator mismatch rather
than an identity (the SEALED55-2 lesson). Participant-held-out: 3 folds x 3 test
subjects, so no prior ever sees the subject it is later asked to clean.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V44_ROOT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44")
sys.path.insert(0, str(V44_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bci2a import (ARRAYS, DERIVED, FS_OUT, N_EEG, OUT_DIR, SEED, SUBJECTS, WINDOW,
                   _load, _normalise, _decode, _report, _ridge, _scale_and_operator,
                   _trials)

EPISODES_PER_CELL = 96
CLEAN_Q, DRIVE_Q = 0.30, 0.70
UPDATES = 40_000
BATCH, LR, WD, CLIP, EMA_DECAY = 8, 1e-4, 1e-4, 1.0, 0.999
GUIDE_DROP, FEATURE_DROP = 0.30, 0.20
VALIDATE_EVERY = 2_000
FOLDS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
EPISODE_CONTRACT = "BCI2A-1:gen_operator_session_tail/guide_operator_calibration_block"


def _continuous(operator, quality):
    norm = np.log(np.linalg.norm(operator, axis=1, keepdims=True).clip(1e-8))
    return np.concatenate((operator, norm,
                           np.broadcast_to(quality[None], (len(operator), 4))), axis=1)


def _signature(operator, quality, centre, scale):
    continuous = (_continuous(operator, quality) - centre) / scale
    return np.concatenate((continuous, np.eye(len(operator))), axis=1).astype(np.float32)


def episodes() -> None:
    out_root = DERIVED / "episodes"
    out_root.mkdir(parents=True, exist_ok=True)
    report = []
    for subject in SUBJECTS:
        for session in ("T", "E"):
            cell = f"{subject}{session}"
            out = out_root / f"{cell}.npz"
            if out.is_file():
                report.append({"cell": cell, "state": "cached"})
                continue
            raw = _load(cell)
            state = _scale_and_operator(raw)
            wins, lwins, _ = _trials(raw, state)          # (n, 22, 512), (n, 2, 512)
            n_win = len(wins)
            if n_win < 8:
                report.append({"cell": cell, "state": "too_short"})
                continue
            # generative operator fitted on the task windows -- disjoint from the
            # calibration block that supplies the guide, so the learnable residual is
            # the operator mismatch and not an identity (SEALED55-2)
            gen_operator = _ridge(
                np.concatenate(list(wins), axis=1),
                np.concatenate(list(lwins), axis=1))[0]
            wins = np.transpose(wins, (1, 0, 2))           # 22 x n x 512
            lwins = np.transpose(lwins, (1, 0, 2))         # 2  x n x 512
            energy = (lwins ** 2).mean(axis=(0, 2))
            order = np.argsort(energy)
            clean_pool = order[:max(int(n_win * CLEAN_Q), 1)]
            drive_pool = order[-max(int(n_win * (1 - DRIVE_Q)), 1):]
            rng = np.random.default_rng(SEED + abs(hash(cell)) % 100000)
            xs, ys, es = [], [], []
            for _ in range(EPISODES_PER_CELL):
                x = wins[:, rng.choice(clean_pool)]
                e = lwins[:, rng.choice(drive_pool)]
                e = np.roll(e, int(rng.integers(0, WINDOW)), axis=1)
                xs.append(x.astype(np.float32))
                ys.append((x + gen_operator @ e).astype(np.float32))
                es.append(e.astype(np.float32))
            np.savez_compressed(
                out, x=np.stack(xs), y=np.stack(ys), e=np.stack(es),
                operator=state["operator"], gen_operator=gen_operator,
                sub_blocks=state["sub_blocks"], quality=state["quality"],
                contract=np.asarray(EPISODE_CONTRACT))
            anchored = float(np.mean([np.linalg.norm(
                (ys[i] - state["operator"] @ es[i]) - xs[i])
                / max(np.linalg.norm(xs[i]), 1e-9) for i in range(len(xs))]))
            unguided = float(np.mean([np.linalg.norm(ys[i] - xs[i])
                                      / max(np.linalg.norm(xs[i]), 1e-9)
                                      for i in range(len(xs))]))
            report.append({"cell": cell, "state": "built",
                           "rrmse_anchored": anchored, "rrmse_unguided": unguided,
                           "drift": float(np.linalg.norm(
                               gen_operator - state["operator"])
                               / max(np.linalg.norm(state["operator"]), 1e-12)),
                           "clean_rms": float(np.sqrt(np.mean(np.stack(xs) ** 2)))})
            print(json.dumps(report[-1]), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "route2_episodes.json").write_text(json.dumps(report, indent=1) + "\n")
    built = [r for r in report if r["state"] == "built"]
    if built:
        gate = {"cells": len(built),
                "mean_anchored": float(np.mean([r["rrmse_anchored"] for r in built])),
                "mean_unguided": float(np.mean([r["rrmse_unguided"] for r in built])),
                "mean_drift": float(np.mean([r["drift"] for r in built])),
                "clean_rms_median": float(np.median([r["clean_rms"] for r in built]))}
        gate["non_degenerate"] = bool(gate["mean_anchored"] > 0.02)
        gate["guide_helps"] = bool(gate["mean_anchored"] < gate["mean_unguided"])
        gate["scale_ok"] = bool(0.3 <= gate["clean_rms_median"] <= 3.0)
        gate["gate_pass"] = bool(gate["non_degenerate"] and gate["guide_helps"]
                                 and gate["scale_ok"])
        (OUT_DIR / "route2_gate.json").write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n")
        print(json.dumps(gate))
        if not gate["gate_pass"]:
            # Reported, not enforced (operator instruction): training proceeds and the
            # diagnostic travels with the result so the reader can weigh it.
            print(json.dumps({"warning": "episode gate did not pass; training anyway",
                              "gate": gate}), flush=True)


def _bank(cells):
    return {c: {k: v for k, v in np.load(DERIVED / "episodes" / f"{c}.npz").items()}
            for c in cells}


def train(fold: int) -> None:
    import torch
    import torch.nn.functional as F
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    gate_path = OUT_DIR / "route2_gate.json"
    gate_state = json.loads(gate_path.read_text()) if gate_path.is_file() else None
    if not (gate_state or {}).get("gate_pass"):
        print(json.dumps({"warning": "episode gate not passed; training anyway",
                          "gate": gate_state}), flush=True)
    ckpt_path = DERIVED / f"prior_fold{fold}.pt"
    if ckpt_path.is_file():
        stale = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if stale.get("episode_contract") == EPISODE_CONTRACT:
            print(json.dumps({"skipped": "already trained under this contract"}))
            return
        raise SystemExit("checkpoint contract mismatch — remove it deliberately")
    test = {SUBJECTS[i] for i in FOLDS[fold]}
    train_cells = [f"{s}{ss}" for s in SUBJECTS if s not in test for ss in ("T", "E")]
    train_cells = [c for c in train_cells
                   if (DERIVED / "episodes" / f"{c}.npz").is_file()]
    banks = _bank(train_cells)
    rows = np.concatenate([_continuous(b["operator"], b["quality"])
                           for b in banks.values()], axis=0)
    centre, scale = rows.mean(axis=0), rows.std(axis=0).clip(1e-6)
    pop_operator = np.mean([b["operator"] for b in banks.values()], axis=0)
    pop_quality = np.mean([b["quality"] for b in banks.values()], axis=0)
    pop_sig = _signature(pop_operator, pop_quality, centre, scale)
    sigs = {c: _signature(b["operator"], b["quality"], centre, scale)
            for c, b in banks.items()}

    device = torch.device("cuda")
    torch.manual_seed(SEED + fold)
    model = CalibSADDPMEOG(channels=N_EEG).to(device)
    schedule = LinearX0Schedule().to(device)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    generator = torch.Generator(device=device).manual_seed(SEED + fold + 7001)
    rng = np.random.default_rng(SEED + fold)
    pool = [(c, i) for c in train_cells for i in range(len(banks[c]["x"]))]
    curve, best = [], float("inf")
    hold = train_cells[:2]

    def batch_of(items):
        xs, ys, a0s, sg = [], [], [], []
        for cell, index in items:
            b = banks[cell]
            xs.append(b["x"][index]); ys.append(b["y"][index])
            a0s.append(b["operator"] @ b["e"][index]); sg.append(sigs[cell])
        return (np.stack(xs), np.stack(ys),
                np.stack(a0s).astype(np.float32), np.stack(sg))

    def validate(weights):
        from eeg_scad.models.calib_saddpm_eog_v44 import ddim_sample_eog
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(weights); model.eval()
        errors = []
        with torch.no_grad():
            for cell in hold:
                items = [(cell, i) for i in range(8)]
                x, y, a0, sg = batch_of(items)
                noise = torch.randn((len(x), N_EEG, WINDOW), device=device,
                                    generator=torch.Generator(device=device)
                                    .manual_seed(SEED + 991))
                out = ddim_sample_eog(model, torch.from_numpy(y).to(device),
                                      torch.from_numpy(a0).to(device),
                                      torch.from_numpy(sg).to(device), noise,
                                      schedule, 50, True).cpu().numpy()
                num = np.linalg.norm(out - x, axis=(1, 2))
                den = np.clip(np.linalg.norm(x, axis=(1, 2)), 1e-9, None)
                errors.append(float(np.mean(num / den)))
        model.load_state_dict(state); model.train()
        return float(np.median(errors))

    model.train()
    for step in range(1, UPDATES + 1):
        items = [pool[i] for i in rng.integers(0, len(pool), BATCH)]
        x, y, a0, sg = batch_of(items)
        a0 = a0.copy(); a0[rng.random(BATCH) < GUIDE_DROP] = 0.0
        sg = sg.copy()
        for position in np.flatnonzero(rng.random(BATCH) < FEATURE_DROP):
            sg[position] = pop_sig
        clean = torch.from_numpy(x).to(device)
        observed = torch.from_numpy(y).to(device)
        anchor = torch.from_numpy(a0).to(device)
        condition = torch.from_numpy(sg).to(device)
        noisy, timestep, _ = schedule.forward_sample(clean, generator)
        optimiser.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(noisy, observed, anchor, timestep, condition),
                                clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimiser.step()
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[key].mul_(EMA_DECAY).add_(value.detach(), alpha=1 - EMA_DECAY)
                else:
                    ema[key].copy_(value)
        if step % VALIDATE_EVERY == 0 or step == UPDATES:
            score = validate(ema)
            curve.append({"step": step, "loss": float(loss.detach()),
                          "validation_rrmse": score})
            print(json.dumps(curve[-1]), flush=True)
            if score < best:
                best = score
                torch.save({"ema": ema, "step": step, "curve": curve,
                            "episode_contract": EPISODE_CONTRACT, "fold": fold,
                            "centre": centre, "scale": scale,
                            "pop_operator": pop_operator, "pop_quality": pop_quality,
                            "train_cells": train_cells,
                            "best_validation_rrmse": score}, ckpt_path)
    print(json.dumps({"fold": fold, "best_validation_rrmse": best}))


def route2() -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG, ddim_sample_eog

    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    states = {f"{s}{ss}": _scale_and_operator(_load(f"{s}{ss}"))
              for s in SUBJECTS for ss in ("T", "E")}
    rows = []
    for fold, indices in enumerate(FOLDS):
        ckpt = torch.load(DERIVED / f"prior_fold{fold}.pt", map_location=device,
                          weights_only=False)
        model = CalibSADDPMEOG(channels=N_EEG).to(device)
        model.load_state_dict(ckpt["ema"]); model.eval()
        centre, scale = ckpt["centre"], ckpt["scale"]
        pop_sig = _signature(ckpt["pop_operator"], ckpt["pop_quality"], centre, scale)
        for index in indices:
            subject = SUBJECTS[index]
            arms = {}
            for session in ("T", "E"):
                cell = f"{subject}{session}"
                state = states[cell]
                x, e, y = _trials(_load(cell), state)
                sig = _signature(state["operator"], state["quality"], centre, scale)
                arms.setdefault("RAW", {})[session] = (x, y)
                for arm in ("MATCH", "NO_A0", "POP"):
                    if arm == "MATCH":
                        a0 = np.stack([state["operator"] @ e[i] for i in range(len(x))])
                        condition = np.stack([sig] * len(x))
                    elif arm == "NO_A0":
                        a0 = np.zeros((len(x), N_EEG, WINDOW))
                        condition = np.stack([sig] * len(x))
                    else:
                        a0 = np.stack([ckpt["pop_operator"] @ e[i]
                                       for i in range(len(x))])
                        condition = np.stack([pop_sig] * len(x))
                    outputs = []
                    for start in range(0, len(x), 16):
                        stop = min(len(x), start + 16)
                        noise = torch.randn((stop - start, N_EEG, WINDOW),
                                            device=device,
                                            generator=torch.Generator(device=device)
                                            .manual_seed(940000 + index * 100 + start))
                        outputs.append(ddim_sample_eog(
                            model,
                            torch.from_numpy(x[start:stop].astype(np.float32)).to(device),
                            torch.from_numpy(a0[start:stop].astype(np.float32)).to(device),
                            torch.from_numpy(condition[start:stop]).to(device),
                            noise, schedule, 50, True).cpu().numpy())
                    arms.setdefault(arm, {})[session] = (np.concatenate(outputs), y)
                print(json.dumps({"cell": cell, "denoised": True}), flush=True)
            for arm, per in sorted(arms.items()):
                xtr, ytr = per["T"]; xte, yte = per["E"]
                accuracy, kappa = _decode(_normalise(xtr), ytr, _normalise(xte), yte,
                                          SEED, device)
                rows.append({"subject": subject, "arm": arm, "accuracy": accuracy,
                             "kappa": kappa, "fold": fold})
                print(json.dumps(rows[-1]), flush=True)
    _report(rows, "route2", "22-channel population prior + calibrated guide, EEGNet, "
                            "official T/E split, participant-held-out priors")
