"""WAVE2-T1 first GPU wave: DT-Gibbs G1, OPERA A1, THRESH T1a.

Gates and operationalizations frozen in reports/wave2_preregistration.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_chart.run_wave2 import DECISIONS, RESULT, V44_RESULT, _stat


# ------------------------------------------------------------------ G1 Gibbs

def g1(fold_id: int) -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _bank_drives, _gated_assets, noise_seed, sample_bank_eog
    from eeg_scad.cli.run_v44_s2 import _posterior_variance
    from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    seed = 20261201
    out_path = RESULT / "g1" / f"fold_{fold_id}.json"
    if out_path.is_file():
        print(json.dumps({"fold": fold_id, "skipped": "complete"}))
        return
    source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                         / "train_curve.json").read_text())
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    post_var = _posterior_variance(registry30, eb120, fold)
    sigma_drift = np.load(RESULT / "sigma_drift.npz")["sigma_drift"]
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    drives = _bank_drives(assets, bank)
    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    ns = noise_seed(fold_id, seed)

    # pass 1: prior-mean anchor
    a0 = np.stack([assets[(m["participant"], m["session"], m["task"])]["C_gated"] @ d
                   for m, d in zip(bank["meta"], drives)])
    sig = np.stack([assets[(m["participant"], m["session"], m["task"])]["sig_gated"]
                    for m in bank["meta"]])
    x_hat = sample_bank_eog(model, schedule, bank["y"], a0, sig, device, ns)

    # C-step per cell (pooled over that cell's episodes; posterior blend)
    cstep = {}
    for key in {(m["participant"], m["session"], m["task"]) for m in bank["meta"]}:
        idx = [i for i, m in enumerate(bank["meta"])
               if (m["participant"], m["session"], m["task"]) == key]
        residual = np.concatenate([np.asarray(bank["y"][i], np.float64) - x_hat[i]
                                   for i in idx], axis=1)
        drive = np.concatenate([drives[i] for i in idx], axis=1)
        gram = drive @ drive.T
        ridge = 0.05 * float(np.trace(gram)) / 2
        c_ml = (residual @ drive.T) @ np.linalg.inv(gram + ridge * np.eye(2))
        fit_res = residual - c_ml @ drive
        v = float(np.mean(fit_res ** 2)) / max(float(np.mean(drive ** 2)) * drive.shape[1], 1e-9)
        p = post_var[key] + sigma_drift
        c_post = (c_ml / max(v, 1e-9) + assets[key]["C_gated"] / p) \
            / (1.0 / max(v, 1e-9) + 1.0 / p)
        cstep[key] = {"c": c_post,
                      "delta_to_prior": float(np.mean(np.abs(c_post - assets[key]["C_gated"]))),
                      "delta_to_true": float(np.mean(
                          c_post - registry30.cells[key].query_transfer)),
                      "signed_norm_shift": float(np.linalg.norm(c_post)
                                                 - np.linalg.norm(
                                                     registry30.cells[key].query_transfer))}

    # pass 2 with the C-step operator
    a0_2 = np.stack([cstep[(m["participant"], m["session"], m["task"])]["c"] @ d
                     for m, d in zip(bank["meta"], drives)])
    x_hat2 = sample_bank_eog(model, schedule, bank["y"], a0_2, sig, device, ns)
    rows = []
    for i, meta in enumerate(bank["meta"]):
        for name, output in (("PASS1", x_hat), ("GIBBS1", x_hat2)):
            rows.append({"participant": meta["participant"], "condition": name,
                         "zero_artifact": meta["zero_artifact"],
                         **paired_metrics(bank["x"][i], bank["y"][i], bank["artifact"][i],
                                          bank["y"][i] - output[i])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"fold": fold_id, "rows": rows,
         "cstep": {"|".join(k): {kk: vv for kk, vv in v.items() if kk != "c"}
                   for k, v in cstep.items()}}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "cells": len(cstep)}))


def g1_decide() -> dict:
    rows, cstep = [], []
    for fold_id in range(5):
        payload = json.loads((RESULT / "g1" / f"fold_{fold_id}.json").read_text())
        rows += payload["rows"]
        cstep += list(payload["cstep"].values())
    frame = pd.DataFrame(rows)
    per = lambda c, z=None: (frame[(frame.condition == c)
                                   & ((frame.zero_artifact == z) if z is not None else True)]
                             .groupby("participant").rrmse_temporal.mean())
    delta_all = (per("GIBBS1") - per("PASS1")).dropna()
    delta_clean = (per("GIBBS1", 1) - per("PASS1", 1)).dropna()
    stat_all, stat_clean = _stat(delta_all), _stat(delta_clean)
    band = 0.015
    clean_eq = bool(-band <= stat_clean["bootstrap_low"]
                    and stat_clean["bootstrap_high"] <= band)
    all_eq = bool(-band <= stat_all["bootstrap_low"] and stat_all["bootstrap_high"] <= band)
    true_bias = float(np.mean([c["delta_to_true"] for c in cstep]))
    norm_shift = float(np.mean([c["signed_norm_shift"] for c in cstep]))
    return {"clean_pass_delta": stat_clean, "clean_pass_equivalent": clean_eq,
            "overall_delta": stat_all, "overall_equivalent": all_eq,
            "cstep_bias_to_true_mean": true_bias,
            "cstep_norm_shift_vs_true": norm_shift,
            "direction_note": ("Chat inflated (over-subtraction risk)" if norm_shift > 0
                               else "Chat deflated"),
            "equivalence_band": band,
            "go": bool(clean_eq and all_eq)}


# ------------------------------------------------------------------ A1 OPERA

def a1() -> None:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_chart.prior_model import CanonicalPrior, ddim_denoise
    from eeg_chart.run_m0 import _canon_path
    from eeg_chart.transport import K_CANONICAL

    out_path = RESULT / "a1.json"
    if out_path.is_file():
        print(json.dumps({"skipped": "complete"}))
        return
    device = torch.device("cuda")
    canon = np.load(_canon_path())["u_canon"]
    rng = np.random.default_rng(20269100)
    K, T = K_CANONICAL, 512

    def make_clean(count):
        base = rng.standard_normal((count, K, T)).astype(np.float32)
        kernel = np.exp(-np.arange(16) / 4.0); kernel /= kernel.sum()
        base = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, base)
        u_coeff = rng.standard_normal((count, 2, T)).astype(np.float32) * 1.5
        u_coeff = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1,
                                      u_coeff)
        return (base + np.einsum("kr,brt->bkt", canon, u_coeff)).astype(np.float32)

    def contaminate(clean):
        y = clean.copy()
        for i in range(len(clean)):
            if rng.random() < 0.4:
                continue
            gain = float(np.exp(rng.uniform(np.log(0.05), np.log(1.3)))) * 4.0
            drive = rng.standard_normal((2, T)).astype(np.float32)
            y[i] += gain * np.einsum("kr,rt->kt", canon, drive).astype(np.float32)
        return y

    def u_energy(stack):
        coeff = np.einsum("kr,bkt->brt", canon, stack)
        return float(np.mean(np.sqrt(np.mean(coeff ** 2, axis=(1, 2)))))

    corpus_clean = make_clean(3000)
    proxy = np.asarray([u_energy(corpus_clean[i:i + 1]) for i in range(len(corpus_clean))])
    censored = corpus_clean[proxy <= np.quantile(proxy, 0.4)]     # low-U selection
    results = {}
    for mode in ("censored", "ambient"):
        torch.manual_seed(11)
        model = CanonicalPrior(base=64).to(device)
        schedule = LinearX0Schedule().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(12)
        pool = censored if mode == "censored" else corpus_clean
        for step in range(12000):
            pick = rng.integers(0, len(pool), 16)
            clean = pool[pick]
            if mode == "censored":
                # supervised on clean-window-SELECTED targets (the censoring under test)
                target = clean
                observed = contaminate(clean)
            else:
                # ambient/EM: noisier-than-observed pairs via the calibrated operator
                # process — NO clean-window selection anywhere
                target = contaminate(clean)          # plays the role of raw observed data
                observed = contaminate(target)       # one extra artifact increment
            clean_t = torch.from_numpy(target).to(device)
            observed_t = torch.from_numpy(observed).to(device)
            noisy, timestep, _ = schedule.forward_sample(clean_t, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(noisy, observed_t, timestep), clean_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        test_clean = make_clean(64)
        test_y = contaminate(test_clean)
        with torch.no_grad():
            y_t = torch.from_numpy(test_y).to(device)
            noise = torch.randn(y_t.shape, device=device,
                                generator=torch.Generator(device=device).manual_seed(13))
            x_hat = ddim_denoise(model, y_t, noise, schedule, 50).cpu().numpy()
        deficit = u_energy(x_hat) / max(u_energy(test_clean), 1e-9)
        rms = np.sqrt(np.mean(x_hat ** 2, axis=(1, 2))) \
            / np.sqrt(np.mean(test_y ** 2, axis=(1, 2))).clip(1e-9)
        results[mode] = {"u_energy_ratio_vs_true": deficit,
                         "rms_q99": float(np.quantile(rms, .99)),
                         "rms_median": float(np.median(rms))}
    dissociation = bool(results["censored"]["u_energy_ratio_vs_true"] < 0.9
                        and results["censored"]["rms_q99"] < 0.9
                        and results["ambient"]["u_energy_ratio_vs_true"] >= 0.9
                        and results["ambient"]["rms_q99"] >= 0.9)
    payload = {"modes": results, "double_dissociation_go": dissociation,
               "prediction": ("censored prior shows U-deficit + over-subtraction; "
                              "ambient shows neither")}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"go": dissociation, **{m: results[m] for m in results}}))


# ------------------------------------------------------------------ T1a ICL

def t1a() -> None:
    import torch
    from torch import nn

    out_path = RESULT / "t1a.json"
    if out_path.is_file():
        print(json.dumps({"skipped": "complete"}))
        return
    t0 = json.loads((RESULT / "thresh_t0.json").read_text())
    tau2, w = t0["task_prior"]["tau2"], t0["task_prior"]["W"]
    mode = "full" if t0["harm_prediction_survives_on_paper"] else "transition_only"
    device = torch.device("cuda")
    rng = np.random.default_rng(20269200)
    dim = 8                                     # operator rows sampled per episode

    def episode_batch(batch, n_context):
        pop = np.zeros(dim)
        c_true = pop + rng.standard_normal((batch, dim)) * np.sqrt(tau2)
        e = rng.standard_normal((batch, n_context + 1))
        y = c_true[:, :, None] * e[:, None, :] \
            + rng.standard_normal((batch, dim, n_context + 1)) * np.sqrt(w)
        context = np.concatenate((e[:, None, :n_context].repeat(dim, 1)[..., None],
                                  y[:, :, :n_context, None]), axis=-1)
        query_e = e[:, -1]
        target = c_true * query_e[:, None]
        return (context.astype(np.float32), query_e.astype(np.float32),
                target.astype(np.float32), c_true)

    class ICLModel(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.embed = nn.Linear(2, width)
            self.attention = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(width, 4, width * 2, batch_first=True,
                                           dropout=0.0), 2)
            self.query = nn.Linear(1, width)
            self.out = nn.Linear(width, 1)

        def forward(self, context, query_e):
            b, d, n, _ = context.shape
            tokens = self.embed(context.reshape(b * d, n, 2))
            q = self.query(query_e.repeat_interleave(d)[:, None])[:, None, :]
            encoded = self.attention(torch.cat((tokens, q), dim=1))
            return self.out(encoded[:, -1]).reshape(b, d)

    sizes = (16, 32, 64, 128)
    grid = (1, 2, 4, 8, 16, 32)
    curves = {}
    for width in sizes:
        torch.manual_seed(width)
        model = ICLModel(width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for step in range(4000):
            n_context = int(rng.choice(grid))
            context, query_e, target, _ = episode_batch(64, n_context)
            pred = model(torch.from_numpy(context).to(device),
                         torch.from_numpy(query_e).to(device))
            loss = ((pred - torch.from_numpy(target).to(device)) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        curve = {}
        with torch.no_grad():
            for n_context in grid:
                errs, prior_errs, ctx_errs = [], [], []
                for _ in range(20):
                    context, query_e, target, c_true = episode_batch(128, n_context)
                    pred = model(torch.from_numpy(context).to(device),
                                 torch.from_numpy(query_e).to(device)).cpu().numpy()
                    errs.append(float(np.mean((pred - target) ** 2)))
                    prior_errs.append(float(np.mean(target ** 2)))   # prior-mean (0) predictor
                    e_ctx = context[..., 0][:, 0, :]
                    y_ctx = context[..., 1]
                    ls = (y_ctx * e_ctx[:, None, :]).sum(-1) / (e_ctx ** 2).sum(-1)[:, None]
                    ctx_errs.append(float(np.mean((ls * query_e[:, None] - target) ** 2)))
                curve[n_context] = {"model": float(np.mean(errs)),
                                    "prior_ref": float(np.mean(prior_errs)),
                                    "context_ls_ref": float(np.mean(ctx_errs))}
        crossover = None
        for n_context in grid:
            if curve[n_context]["model"] < 0.5 * curve[n_context]["prior_ref"]:
                crossover = n_context
                break
        curves[width] = {"curve": {str(k): v for k, v in curve.items()},
                         "crossover_n": crossover}
    crossings = [curves[wd]["crossover_n"] for wd in sizes]
    transition = bool(any(c is not None for c in crossings)
                      and len({c for c in crossings if c is not None}) >= 2)
    payload = {"mode": mode, "tau2": tau2, "W": w,
               "closed_form_crossover": float(w / tau2),
               "sizes": {str(wd): curves[wd] for wd in sizes},
               "size_dependent_transition": transition,
               "falsification": (None if transition else
                                 "no size-dependent transition — threshold law rejected "
                                 "on this family")}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": mode, "crossings": crossings, "transition": transition}))


def dp2() -> None:
    dp1_payload = json.loads((DECISIONS / "wave2_dp1.json").read_text())
    gates = dp1_payload["gpu_wave_gates"]
    out = {"gates_applied": gates}
    if gates["G1"] and (RESULT / "g1" / "fold_0.json").is_file():
        out["G1"] = g1_decide()
    else:
        out["G1"] = {"skipped": "gate failed or not run"}
    for name, path in (("A1", RESULT / "a1.json"), ("T1a", RESULT / "t1a.json")):
        out[name] = json.loads(path.read_text()) if path.is_file() \
            else {"skipped": "gate failed or not run"}
    DECISIONS.mkdir(parents=True, exist_ok=True)
    (DECISIONS / "wave2_dp2.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"G1_go": out["G1"].get("go"),
                      "A1_go": out["A1"].get("double_dissociation_go"),
                      "T1a_transition": out["T1a"].get("size_dependent_transition")}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    g1_parser = sub.add_parser("g1")
    g1_parser.add_argument("--fold", type=int, required=True)
    for name in ("a1", "t1a", "dp2"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.unit == "g1":
        g1(args.fold)
    else:
        {"a1": a1, "t1a": t1a, "dp2": dp2}[args.unit]()


if __name__ == "__main__":
    main()
