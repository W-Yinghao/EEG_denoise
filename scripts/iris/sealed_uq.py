#!/usr/bin/env python3
"""Option B — UQ-calibration confirmation machinery on EEGEyeNet.

Preregistered in reports/iris_prereg_sealed55.md, amendment SEALED55-1 (frozen before
this file ran). Ports the V44 architecture and the F4 operator-posterior width policy
to the EEGEyeNet antisaccade corpus, whose S356 episodes share the V44 tensor geometry
(46 x 512 at 100 Hz, 2-row [VEOG, HEOG] latent, 46x2 ridge operator).

DEV-CLASS ONLY. Every mode in this file refuses any root but the dev-class ones; the
sealed cohort is scored by scripts/iris/sealed_confirm.py after the block is opened,
using the checkpoint and the temperature frozen here.

Modes
  episodes    build dev-class episodes carrying the drive, both operators and the
              SUPPORT sub-block refits (SEALED55-2 contract: inject with the QUERY-half
              generative operator, guide with the SUPPORT-half calibration operator)
  train       train one CalibSADDPMEOG prior on dev-class training subjects
  freeze-temp K=32 chains on dev-class evaluation subjects -> one scalar per policy,
              committed before the sealed block opens
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V44_ROOT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44")
sys.path.insert(0, str(V44_ROOT / "src"))
sys.path.insert(0, str(REPO / "scripts/iris"))

S356_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_s356")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_sealed_uq")
OUT_DIR = REPO / "results/iris/sealed_confirm"

SEED = 20260830
# Bumped whenever the episode construction changes, so a checkpoint trained under an
# older contract can never be silently reused (SEALED55-2 caught exactly that).
EPISODE_CONTRACT = "SEALED55-2:gen_operator_query_half/guide_operator_support_half"
UPDATES = 80_000
BATCH = 8
LR, WEIGHT_DECAY, CLIP, EMA_DECAY = 1e-4, 1e-4, 1.0, 0.999
GUIDE_DROP, FEATURE_DROP = 0.30, 0.20
VALIDATION_EVERY = 2_000
K_CHAINS = 32
SUB_BLOCKS = 4
RIDGE = 0.05
WINDOW, N_CHANNELS = 512, 46
Z = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.90: 1.6448536269514722}
TEMP_GRID = np.arange(0.5, 6.0 + 1e-9, 0.05)
# S356's own evaluation split, reused here as the dev-class UQ calibration cohort
EVAL_SUBJECTS = ("AA0", "AA1", "AA4", "AA5", "AA7", "AA8", "AA9", "AB0", "AB1",
                 "AB2", "AB3", "AB6", "AB7", "AB9", "AC0")


def _ridge_operator(block: np.ndarray, latent: np.ndarray):
    y_c = block - block.mean(axis=1, keepdims=True)
    e_c = latent - latent.mean(axis=1, keepdims=True)
    gram = e_c @ e_c.T
    ridge = RIDGE * max(float(np.trace(gram) / len(gram)), 1e-12)
    operator = (y_c @ e_c.T) @ np.linalg.inv(gram + ridge * np.eye(2))
    fitted = operator @ e_c
    residual = y_c - fitted
    r2 = 1 - float(np.sum(residual ** 2) / max(np.sum(y_c ** 2), 1e-12))
    cond = float(np.linalg.cond(gram + ridge * np.eye(2)))
    return operator, r2, cond


def _continuous(operator: np.ndarray, quality: np.ndarray) -> np.ndarray:
    norm = np.log(np.linalg.norm(operator, axis=1, keepdims=True).clip(1e-8))
    return np.concatenate((operator, norm,
                           np.broadcast_to(quality[None], (len(operator), 4))), axis=1)


def episodes() -> None:
    """Replicate the S356 episode construction, additionally recording the drive, both
    operators and the SUPPORT sub-block refits. The clean-window draw order is unchanged
    (so x matches the banked S356 episodes); y is injected with the QUERY-half generative
    operator per amendment SEALED55-2, so the guided task is not degenerate."""
    import s356_probe as s356
    s356.DERIVED = S356_DERIVED                      # read prepared subjects read-only
    (DERIVED / "episodes").mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.stem for p in (S356_DERIVED / "subjects").glob("*.npz"))
    report = []
    for subject in subjects:
        out = DERIVED / "episodes" / f"{subject}.npz"
        if out.is_file():
            report.append({"subject": subject, "state": "cached"})
            continue
        d = np.load(S356_DERIVED / "subjects" / f"{subject}.npz")
        data = d["data"].astype(np.float64)
        veog = s356._bandpass(d["veog"].astype(np.float64), 0.5, 8.0, s356.FS_OUT)
        heog = s356._bandpass(d["heog"].astype(np.float64), 0.5, 20.0, s356.FS_OUT)
        n = data.shape[1]
        latent = np.stack([veog, heog])
        center = np.median(latent[:, 0:n // 2], axis=1, keepdims=True)
        scale = 1.4826 * np.median(np.abs(latent[:, 0:n // 2] - center),
                                   axis=1, keepdims=True)
        latent = (latent - center) / np.maximum(scale, 1e-9)
        eeg_scale = float(np.sqrt(np.mean(data[:, 0:n // 2] ** 2)))
        # SEALED55-2: the calibration operator (guide + signature) is fitted on the
        # SUPPORT half; the generative operator that injects the artifact is fitted on
        # the QUERY half and is never a model input. The learnable residual is then
        # (C_gen - C_sup) e, the within-subject operator drift — V44's structure.
        sup = data[:, 0:n // 2] / max(eeg_scale, 1e-9)
        sup_lat = latent[:, 0:n // 2]
        operator, r2, cond = _ridge_operator(sup, sup_lat)
        qry = data[:, n // 2:] / max(eeg_scale, 1e-9)
        qry_lat = latent[:, n // 2:]
        gen_operator = _ridge_operator(qry, qry_lat)[0]
        span = sup.shape[1] // SUB_BLOCKS
        blocks = np.stack([_ridge_operator(sup[:, i * span:(i + 1) * span],
                                           sup_lat[:, i * span:(i + 1) * span])[0]
                           for i in range(SUB_BLOCKS)])
        rms = np.sqrt(np.mean(sup_lat ** 2, axis=1)).clip(1e-8)
        quality = np.array([np.log(rms[0]), np.log(rms[1]), r2, np.log1p(cond)])

        payload = {"operator": operator, "gen_operator": gen_operator,
                   "sub_block_operators": blocks,
                   "quality": quality, "eeg_scale": eeg_scale,
                   "operator_drift": float(np.linalg.norm(gen_operator - operator)
                                           / max(np.linalg.norm(operator), 1e-12))}
        # S356's stream, verbatim, so the (x, y) pairs match the banked episodes
        rng = np.random.default_rng(s356.SEED + s356.hash_stable(subject))
        for half in ("support", "query"):
            lo, hi = (0, n // 2) if half == "support" else (n // 2, n)
            block = data[:, lo:hi] / max(eeg_scale, 1e-9)
            lat = latent[:, lo:hi]
            n_win = block.shape[1] // WINDOW
            if n_win < 2:
                break
            wins = block[:, :n_win * WINDOW].reshape(N_CHANNELS, n_win, WINDOW)
            lwins = lat[:, :n_win * WINDOW].reshape(2, n_win, WINDOW)
            energy = (lwins ** 2).mean(axis=(0, 2))
            order = np.argsort(energy)
            clean_pool = order[:max(int(n_win * s356.CLEAN_QUANTILE), 1)]
            drive_pool = order[-max(int(n_win * (1 - s356.DRIVE_QUANTILE)), 1):]
            xs, ys, es = [], [], []
            for _ in range(s356.EPISODES_PER_SUBJECT):
                x = wins[:, rng.choice(clean_pool)]
                e = lwins[:, rng.choice(drive_pool)]
                e = np.roll(e, int(rng.integers(0, WINDOW)), axis=1)
                xs.append(x.astype(np.float32))
                ys.append((x + gen_operator @ e).astype(np.float32))
                es.append(e.astype(np.float32))
            tag = "sup" if half == "support" else "qry"
            payload |= {f"{tag}_x": np.stack(xs), f"{tag}_y": np.stack(ys),
                        f"{tag}_e": np.stack(es)}
        if "sup_x" not in payload:
            report.append({"subject": subject, "state": "too_short"})
            continue
        np.savez_compressed(out, **payload)
        report.append({"subject": subject, "state": "built",
                       "has_query": "qry_x" in payload})
    (OUT_DIR / "uq_episodes_report.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uq_episodes_report.json").write_text(
        json.dumps(report, indent=1) + "\n")
    print(json.dumps({"subjects": len(report),
                      "built": sum(r["state"] == "built" for r in report)}))


def _bank(subjects):
    return {s: dict(np.load(DERIVED / "episodes" / f"{s}.npz")) for s in subjects}


def _signature_space(banks, train_subjects):
    rows = np.concatenate([_continuous(banks[s]["operator"], banks[s]["quality"])
                           for s in train_subjects], axis=0)
    center, scale = rows.mean(axis=0), rows.std(axis=0).clip(1e-6)
    pop_operator = np.mean([banks[s]["operator"] for s in train_subjects], axis=0)
    pop_quality = np.mean([banks[s]["quality"] for s in train_subjects], axis=0)
    return center, scale, pop_operator, pop_quality


def _signature(operator, quality, center, scale):
    continuous = (_continuous(operator, quality) - center) / scale
    return np.concatenate((continuous, np.eye(len(operator))), axis=1).astype(np.float32)


def _split(all_subjects):
    train = [s for s in all_subjects if s not in EVAL_SUBJECTS]
    evaluation = [s for s in all_subjects if s in EVAL_SUBJECTS
                  and "qry_x" in np.load(DERIVED / "episodes" / f"{s}.npz").files]
    return train, evaluation


def train() -> None:
    import torch
    import torch.nn.functional as F
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    ckpt_path = DERIVED / "uq_prior.pt"
    gate = OUT_DIR / "uq_nondegeneracy_check.json"
    if not gate.is_file() or not json.loads(gate.read_text())["gate_pass"]:
        raise SystemExit("non-degeneracy gate has not passed — refusing to train")
    if ckpt_path.is_file():
        import torch
        stale = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if stale.get("episode_contract") != EPISODE_CONTRACT:
            raise SystemExit(
                f"checkpoint was trained under contract {stale.get('episode_contract')!r}, "
                f"current contract is {EPISODE_CONTRACT!r} — remove it deliberately")
        print(json.dumps({"skipped": "prior already trained under this contract"}))
        return
    subjects = sorted(p.stem for p in (DERIVED / "episodes").glob("*.npz"))
    train_subjects, eval_subjects = _split(subjects)
    banks = _bank(train_subjects + eval_subjects)
    center, scale, pop_operator, pop_quality = _signature_space(banks, train_subjects)
    pop_sig = _signature(pop_operator, pop_quality, center, scale)
    sigs = {s: _signature(banks[s]["operator"], banks[s]["quality"], center, scale)
            for s in banks}

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    model = CalibSADDPMEOG().to(device)
    schedule = LinearX0Schedule().to(device)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device=device).manual_seed(SEED + 7001)
    rng = np.random.default_rng(SEED)
    pool = [(s, i) for s in train_subjects
            for i in range(len(banks[s]["sup_x"]))]
    curve, best = [], float("inf")

    def batch_of(indices, half="sup"):
        xs, ys, a0s, sg = [], [], [], []
        for subject, index in indices:
            d = banks[subject]
            xs.append(d[f"{half}_x"][index])
            ys.append(d[f"{half}_y"][index])
            a0s.append(d["operator"] @ d[f"{half}_e"][index])
            sg.append(sigs[subject])
        return (np.stack(xs), np.stack(ys), np.stack(a0s).astype(np.float32),
                np.stack(sg))

    def validate(weights) -> float:
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(weights)
        model.eval()
        errors = []
        with torch.no_grad():
            for subject in eval_subjects:
                d = banks[subject]
                idx = [(subject, i) for i in range(min(8, len(d["qry_x"])))]
                x, y, a0, sg = batch_of(idx, "qry")
                from eeg_scad.models.calib_saddpm_eog_v44 import ddim_sample_eog
                noise = torch.randn((len(x), N_CHANNELS, WINDOW), device=device,
                                    generator=torch.Generator(device=device)
                                    .manual_seed(SEED + 991))
                out = ddim_sample_eog(
                    model, torch.from_numpy(y).to(device),
                    torch.from_numpy(a0).to(device),
                    torch.from_numpy(sg).to(device), noise, schedule, 50, True)
                out = out.cpu().numpy()
                num = np.linalg.norm(out - x, axis=(1, 2))
                den = np.clip(np.linalg.norm(x, axis=(1, 2)), 1e-9, None)
                errors.append(float(np.mean(num / den)))
        model.load_state_dict(state)
        model.train()
        return float(np.mean(errors))

    model.train()
    for step in range(1, UPDATES + 1):
        indices = [pool[i] for i in rng.integers(0, len(pool), BATCH)]
        x, y, a0, sg = batch_of(indices)
        drop = rng.random(BATCH) < GUIDE_DROP
        a0 = a0.copy()
        a0[drop] = 0.0
        sg = sg.copy()
        for position in np.flatnonzero(rng.random(BATCH) < FEATURE_DROP):
            sg[position] = pop_sig
        clean = torch.from_numpy(x).to(device)
        observed = torch.from_numpy(y).to(device)
        anchor = torch.from_numpy(a0).to(device)
        condition = torch.from_numpy(sg).to(device)
        noisy, timestep, _ = schedule.forward_sample(clean, generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy, observed, anchor, timestep, condition)
        loss = F.smooth_l1_loss(prediction, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at update {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if value.dtype.is_floating_point:
                    ema[key].mul_(EMA_DECAY).add_(value.detach(), alpha=1 - EMA_DECAY)
                else:
                    ema[key].copy_(value)
        if step % VALIDATION_EVERY == 0 or step == UPDATES:
            score = validate(ema)
            curve.append({"step": step, "loss": float(loss.detach()),
                          "validation_rrmse": score})
            print(json.dumps(curve[-1]), flush=True)
            payload = {"ema": ema, "step": step, "curve": curve, "seed": SEED,
                       "episode_contract": EPISODE_CONTRACT,
                       "center": center, "scale": scale,
                       "pop_operator": pop_operator, "pop_quality": pop_quality,
                       "train_subjects": train_subjects,
                       "eval_subjects": eval_subjects,
                       "best_validation_rrmse": min(best, score)}
            torch.save(payload, DERIVED / "uq_prior_last.pt")
            if score < best:
                best = score
                payload["best_validation_rrmse"] = score
                torch.save(payload, ckpt_path)
    print(json.dumps({"best_validation_rrmse": best, "updates": UPDATES}))


def _posterior_variance(banks, train_subjects):
    tau2 = np.var(np.stack([banks[s]["operator"] for s in train_subjects]),
                  axis=0, ddof=1).clip(1e-12)
    out = {}
    for subject, d in banks.items():
        v = np.var(d["sub_block_operators"], axis=0, ddof=1).clip(1e-12)
        out[subject] = 1.0 / (1.0 / tau2 + SUB_BLOCKS / v)
    return out


def chains(subjects, banks, post_var, ckpt, device, half="qry"):
    """K=32 operator-sampled trajectories. Returns per-subject error/sigma/var_op."""
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG, ddim_sample_eog

    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    schedule = LinearX0Schedule().to(device)
    center, scale = ckpt["center"], ckpt["scale"]
    out = {}
    for subject in subjects:
        d = banks[subject]
        x = d[f"{half}_x"]
        y = d[f"{half}_y"]
        e = d[f"{half}_e"]
        sig = np.stack([_signature(d["operator"], d["quality"], center, scale)] * len(x))
        chain_outputs = []
        for chain in range(K_CHAINS):
            rng = np.random.default_rng(910000 + chain * 17
                                        + (abs(hash(subject)) % 100000))
            operator = d["operator"] + rng.standard_normal(d["operator"].shape) \
                * np.sqrt(post_var[subject])
            a0 = np.stack([operator @ e[i] for i in range(len(x))]).astype(np.float32)
            noise = torch.randn((len(x), N_CHANNELS, WINDOW), device=device,
                                generator=torch.Generator(device=device)
                                .manual_seed(820000 + chain * 31))
            pred = ddim_sample_eog(model, torch.from_numpy(y).to(device),
                                   torch.from_numpy(a0).to(device),
                                   torch.from_numpy(sig).to(device), noise,
                                   schedule, 50, True).cpu().numpy()
            chain_outputs.append(pred.astype(np.float32))
        ensemble = np.stack(chain_outputs)
        mean = ensemble.mean(axis=0)
        sigma = ensemble.std(axis=0, ddof=1).clip(1e-9)
        var_op = np.stack([post_var[subject] @ (e[i].astype(np.float64) ** 2)
                           for i in range(len(x))])
        out[subject] = {"errors": np.abs(x - mean).astype(np.float16),
                        "sigma": sigma.astype(np.float16),
                        "var_op": var_op.astype(np.float16)}
        print(json.dumps({"subject": subject, "episodes": int(len(x))}), flush=True)
    return out


def freeze_temp() -> None:
    import torch
    ckpt = torch.load(DERIVED / "uq_prior.pt", map_location="cuda",
                      weights_only=False)
    if ckpt.get("episode_contract") != EPISODE_CONTRACT:
        raise SystemExit("checkpoint/episode contract mismatch — refusing")
    subjects = sorted(p.stem for p in (DERIVED / "episodes").glob("*.npz"))
    train_subjects, eval_subjects = _split(subjects)
    banks = _bank(train_subjects + eval_subjects)
    post_var = _posterior_variance(banks, train_subjects)
    arrays = chains(eval_subjects, banks, post_var, ckpt, torch.device("cuda"))
    np.savez_compressed(DERIVED / "devclass_chains.npz",
                        **{f"{k}_{f}": v[f] for k, v in arrays.items()
                           for f in ("errors", "sigma", "var_op")})
    temps = {}
    for policy in ("INFL", "TEMP"):
        for s in TEMP_GRID:
            covs = []
            for value in arrays.values():
                base = (np.sqrt(value["sigma"].astype(np.float64) ** 2
                                + value["var_op"].astype(np.float64))
                        if policy == "INFL" else value["sigma"].astype(np.float64))
                covs.append(float(np.mean(value["errors"].astype(np.float64)
                                          <= Z[0.80] * s * base)))
            if np.mean(covs) >= 0.80:
                temps[policy] = float(round(s, 2))
                break
        else:
            temps[policy] = float(TEMP_GRID[-1])
    payload = {"preregistration": "reports/iris_prereg_sealed55.md (SEALED55-1)",
               "rule": "smallest s in arange(0.50,6.00,0.05) reaching mean 80% coverage "
                       "over dev-class evaluation subjects",
               "temperatures": temps, "dev_class_eval_subjects": eval_subjects,
               "sealed_contact": 0, "frozen_before_opening": True}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uq_temperature.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(temps))


def check() -> None:
    """Non-degeneracy gate (SEALED55-2). The guided task must have a real residual:
    RRMSE of the naive anchored observation y - a0 against x must be clearly nonzero,
    and must be smaller than the unguided baseline y."""
    subjects = sorted(p.stem for p in (DERIVED / "episodes").glob("*.npz"))
    anchored, unguided, drifts = [], [], []
    for subject in subjects:
        with np.load(DERIVED / "episodes" / f"{subject}.npz") as d:
            if "qry_x" not in d:
                continue
            x, y, e = d["qry_x"], d["qry_y"], d["qry_e"]
            a0 = np.stack([d["operator"] @ e[i] for i in range(len(x))])
            den = np.clip(np.linalg.norm(x, axis=(1, 2)), 1e-9, None)
            anchored.append(float(np.mean(
                np.linalg.norm((y - a0) - x, axis=(1, 2)) / den)))
            unguided.append(float(np.mean(
                np.linalg.norm(y - x, axis=(1, 2)) / den)))
            drifts.append(float(d["operator_drift"]))
    payload = {"subjects": len(anchored),
               "rrmse_anchored_y_minus_a0": float(np.mean(anchored)),
               "rrmse_unguided_y": float(np.mean(unguided)),
               "operator_drift_relative": float(np.mean(drifts)),
               "non_degenerate": bool(np.mean(anchored) > 0.02),
               "guide_helps": bool(np.mean(anchored) < np.mean(unguided))}
    payload["gate_pass"] = bool(payload["non_degenerate"] and payload["guide_helps"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uq_nondegeneracy_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    if not payload["gate_pass"]:
        raise SystemExit("NON-DEGENERACY GATE FAILED — do not train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["episodes", "check", "train", "freeze-temp"])
    args = parser.parse_args()
    {"episodes": episodes, "check": check, "train": train,
     "freeze-temp": freeze_temp}[args.mode]()


if __name__ == "__main__":
    main()
