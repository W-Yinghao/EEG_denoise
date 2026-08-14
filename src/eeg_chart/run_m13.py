"""FLAGSHIP-M13 execution CLI. Rules frozen in reports/m13_preregistration.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_chart.analytic import canonical_clean
from eeg_chart.geodesic import transport_family
from eeg_chart.run_m0 import (PANELS, STRATA, _arms, _load_panel, _canon_path, _stat,
                              _strata_masks, transport_context)


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/flagship_m13"
REPORT = ROOT / "reports"
V44_RESULT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/results/rgcc_eog_v44")
KAPPA_TARGET = 100.0


def _panel_probe(cells, context) -> dict[str, Any]:
    """U1-a-style analytic arm probe under the current context settings."""
    masks = _strata_masks(cells)
    ordered = sorted(context["per_cell"].items())
    unit_rows: dict[str, dict[str, dict[str, list[float]]]] = {}
    for index, (cell_id, entry) in enumerate(ordered):
        donor = next(other for _, other in ordered[index + 1:] + ordered[:index]
                     if other["cell"].subject != entry["cell"].subject)
        arms = _arms(context, entry, donor, gauge_seed=97000 + index)
        unit = entry["cell"].subject if entry["cell"].panel != "klados" else entry["cell"].cell
        for episode in entry["cell"].episodes:
            flags = masks(episode)
            for arm_name, arm in arms.items():
                cleaned = canonical_clean(arm, context["u_canon"], context["sigma_bar_inv"],
                                          episode["y"])
                rrmse = float(np.linalg.norm(cleaned - episode["x"])
                              / max(np.linalg.norm(episode["x"]), 1e-12))
                for stratum, flag in flags.items():
                    if flag:
                        unit_rows.setdefault(stratum, {}).setdefault(arm_name, {}) \
                            .setdefault(unit, []).append(rrmse)
    out = {}
    for stratum in STRATA:
        arms_mean = {arm: {unit: float(np.mean(values)) for unit, values in units.items()}
                     for arm, units in unit_rows.get(stratum, {}).items()}
        common = sorted(set.intersection(*(set(v) for v in arms_mean.values())))
        out[stratum] = {
            "arm_means": {arm: float(np.mean(list(v.values()))) for arm, v in arms_mean.items()},
            "ceiling_pop_minus_oracle": _stat([arms_mean["T-POP"][u] - arms_mean["T-ORACLE"][u]
                                               for u in common]),
            "deployable_match_minus_pop_gain": _stat([arms_mean["T-POP"][u] - arms_mean["T-MATCH"][u]
                                                      for u in common]),
            "gauge_null_minus_pop": _stat([arms_mean["GAUGE-NULL"][u] - arms_mean["T-POP"][u]
                                           for u in common]),
            "wrong_minus_pop": _stat([arms_mean["T-WRONG"][u] - arms_mean["T-POP"][u]
                                      for u in common]),
            "units": len(common)}
    return out


def w1() -> None:
    canon = np.load(_canon_path())["u_canon"]
    target = RESULT / "w1_repair"
    target.mkdir(parents=True, exist_ok=True)
    geometry, probes = {}, {}
    for panel in PANELS:
        cells, lift = _load_panel(panel)
        context = transport_context(cells, lift, canon, whitening="truncated",
                                    split_half_abstain=True)
        roundtrip, kappas, angles = [], [], []
        identity_ok = True
        for cell_id, entry in sorted(context["per_cell"].items()):
            cell = entry["cell"]
            for rho in (0.0, entry["rho"], 1.0):
                arm = transport_family(context["lift"], context["lift_pinv"],
                                       context["sigma_bar"], cell.sigma_support,
                                       entry["rotation"], entry["base"], rho,
                                       whitening="truncated")
                roundtrip.append(float(np.max(np.abs(arm.pinv @ arm.transport
                                                     - np.eye(arm.transport.shape[1])))))
                kappas.append(float(np.linalg.cond(arm.transport)))
            pop_arm = transport_family(context["lift"], context["lift_pinv"],
                                       context["sigma_bar"], None, entry["base"],
                                       entry["base"], 0.0)
            zero_arm = transport_family(context["lift"], context["lift_pinv"],
                                        context["sigma_bar"], cell.sigma_support,
                                        entry["rotation"], entry["base"], 0.0,
                                        whitening="truncated")
            identity_ok &= bool(np.array_equal(zero_arm.transport, pop_arm.transport)
                                and np.array_equal(zero_arm.pinv, pop_arm.pinv))
            match_arm = transport_family(context["lift"], context["lift_pinv"],
                                         context["sigma_bar"], cell.sigma_support,
                                         entry["rotation"], entry["base"], entry["rho"],
                                         whitening="truncated")
            transported = match_arm.transport @ cell.a_query
            from eeg_chart.transport import ordered_frame
            angle = np.degrees(np.arccos(np.clip(np.linalg.svd(
                ordered_frame(transported).T @ context["u_canon"], compute_uv=False),
                -1, 1))).max()
            angles.append(float(angle))
        within = [entry["split_half_distance"] for entry in context["per_cell"].values()]
        between = [entry["cohort_distance"] for entry in context["per_cell"].values()]
        geometry[panel] = {
            "cells": len(context["per_cell"]),
            "roundtrip_max": float(np.max(roundtrip)),
            "roundtrip_gate": bool(np.max(roundtrip) <= 1e-10),
            "rho0_bit_identity": bool(identity_ok),
            "kappa_max": float(np.max(kappas)), "kappa_median": float(np.median(kappas)),
            "kappa_within_target_fraction": float(np.mean(np.asarray(kappas)
                                                          <= KAPPA_TARGET * 50)),
            "frame_angle_within_15deg_fraction": float(np.mean(np.asarray(angles) <= 15.0)),
            "frame_angle_p50": float(np.median(angles)),
            "abstentions": int(sum(entry["abstained"] for entry in context["per_cell"].values())),
            "rho_mean_nonabstained": float(np.mean([entry["rho"] for entry
                                                    in context["per_cell"].values()
                                                    if not entry["abstained"]] or [0.0])),
            "split_half_median": float(np.median(within)),
            "between_median": float(np.median(between)),
            "diagnosis": ("estimation_noise_dominated" if np.median(within) >= np.median(between)
                          else "heterogeneity_dominated"),
        }
        probes[panel] = _panel_probe(cells, context)
    (target / "geometry.json").write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n")
    (target / "u1a_rerun.json").write_text(json.dumps(probes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({panel: {"kappa_max": geometry[panel]["kappa_max"],
                              "abstentions": geometry[panel]["abstentions"],
                              "deployable_all": probes[panel]["all"]
                              ["deployable_match_minus_pop_gain"]["mean"]}
                      for panel in PANELS}))


# ------------------------------------------------------------------ W2 data

PRIOR_DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/flagship_m13")
PAIRED = ("mobilebci", "klados", "bci2b")
TRUTH_FREE = ("sgeyesub", "shu", "physiomotion", "eegdn")
RUN_CORPORA = {
    "p0": list(PAIRED),
    "pooled": list(PAIRED) + list(TRUTH_FREE),
    "lodo_mobilebci": ["klados", "bci2b"] + list(TRUTH_FREE),
    "lodo_klados": ["mobilebci", "bci2b"] + list(TRUTH_FREE),
    "lodo_bci2b": ["mobilebci", "klados"] + list(TRUTH_FREE),
    "lodo_sgeyesub": list(PAIRED) + ["shu", "physiomotion", "eegdn"],
    "ambient": list(PAIRED) + list(TRUTH_FREE),
    "ref_mobilebci": ["mobilebci"], "ref_klados": ["klados"], "ref_bci2b": ["bci2b"],
}
PV2_BAND = (0.90, 1.10)
PV3_MARGIN = 0.005
MASKS_PER_MONTAGE = 8


def w2_harvest() -> None:
    from eeg_chart.prior_data import (CACHE, harvest_bci2b, harvest_eegdenoisenet,
                                      harvest_klados, harvest_mobilebci,
                                      harvest_physiomotion, harvest_sgeyesub, harvest_shu)

    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, harvest in (("mobilebci", harvest_mobilebci), ("klados", harvest_klados),
                          ("bci2b", harvest_bci2b), ("shu", harvest_shu),
                          ("physiomotion", harvest_physiomotion),
                          ("eegdn", harvest_eegdenoisenet)):
        cache = harvest()
        payload = {"windows": cache["windows"], "subjects": cache["subjects"]}
        if "drives" in cache:
            payload["drives"] = cache["drives"]
        np.savez_compressed(CACHE / f"{name}.npz", **payload)
        manifest[name] = {"windows": int(len(cache["windows"])),
                          "montage": cache["montage"],
                          "labels": cache.get("labels")}
    sg_dir = CACHE / "sgeyesub"
    sg_dir.mkdir(exist_ok=True)
    sg_manifest, sg_errors = [], []
    for cache in harvest_sgeyesub():
        if "error" in cache:
            sg_errors.append(cache["error"])
            continue
        stem = cache["montage"].replace("/", "_")
        np.savez_compressed(sg_dir / f"{stem}.npz", windows=cache["windows"],
                            subjects=cache["subjects"])
        sg_manifest.append({"stem": stem, "windows": int(len(cache["windows"])),
                            "labels": cache["labels"]})
    manifest["sgeyesub"] = {"records": sg_manifest, "errors": sg_errors,
                            "windows": int(sum(r["windows"] for r in sg_manifest))}
    if sg_errors and not sg_manifest:
        raise RuntimeError(f"SGEYESUB harvest produced nothing: {sg_errors[:3]}")
    (CACHE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: (entry.get("windows")) for name, entry in manifest.items()}))


class _CorpusPool:
    """Training pool: channel windows + per-montage lifts + fixed mask lifts."""

    def __init__(self, corpora: list[str], seed: int):
        from eeg_chart.prior_data import CACHE, montage_positions
        from eeg_chart.transport import sh_lift

        rng = np.random.default_rng(seed + 555)
        manifest = json.loads((CACHE / "manifest.json").read_text())
        self.groups = []
        self.drives = []

        def lifts_for(positions):
            full = sh_lift(positions)
            masked = [full]
            count = len(positions)
            if count >= 12:
                for _ in range(MASKS_PER_MONTAGE - 1):
                    keep = np.sort(rng.choice(count, size=max(8, int(count * 0.7)),
                                              replace=False))
                    masked.append((sh_lift(positions[keep]), keep))
            return full, masked

        for name in corpora:
            if name == "sgeyesub":
                for record in manifest["sgeyesub"]["records"]:
                    with np.load(CACHE / "sgeyesub" / f"{record['stem']}.npz",
                                 allow_pickle=False) as archive:
                        windows = np.asarray(archive["windows"], np.float32)
                    positions = montage_positions("sgeyesub", labels=record["labels"])
                    full, masked = lifts_for(positions)
                    self.groups.append({"name": f"sgeyesub:{record['stem']}",
                                        "corpus": "sgeyesub", "windows": windows,
                                        "lifts": masked})
                continue
            with np.load(CACHE / f"{name}.npz", allow_pickle=False) as archive:
                windows = np.asarray(archive["windows"], np.float32)
                if "drives" in archive.files:
                    self.drives.append(np.asarray(archive["drives"], np.float32))
            labels = manifest[name].get("labels")
            positions = montage_positions(manifest[name]["montage"].split("_")[0]
                                          if name != "eegdn" else "eegdn", labels=labels) \
                if name != "eegdn" else None
            if name == "eegdn":
                # single-channel montage mask: one random electrode per draw
                base = montage_positions("mobilebci")
                masked = [(sh_lift(base[i:i + 1]), np.asarray([0]))
                          for i in rng.choice(len(base), size=MASKS_PER_MONTAGE, replace=False)]
                self.groups.append({"name": name, "corpus": name, "windows": windows,
                                    "lifts": [m for m in masked]})
                continue
            full, masked = lifts_for(positions)
            self.groups.append({"name": name, "corpus": name, "windows": windows,
                                "lifts": masked})
        if not self.drives:  # LODO holds may remove all drive corpora; fall back to all
            for name in PAIRED:
                path = CACHE / f"{name}.npz"
                if path.is_file():
                    with np.load(path, allow_pickle=False) as archive:
                        if "drives" in archive.files:
                            self.drives.append(np.asarray(archive["drives"], np.float32))
        self.drive_bank = np.concatenate(self.drives) if self.drives else None
        self.corpus_names = sorted({group["corpus"] for group in self.groups})
        self.rng = rng

    def sample(self, batch: int):
        """Returns canonical clean windows (batch, K, 512), corpus-balanced."""
        from eeg_chart.transport import K_CANONICAL

        out = np.zeros((batch, K_CANONICAL, 512), np.float32)
        for index in range(batch):
            corpus = self.corpus_names[int(self.rng.integers(len(self.corpus_names)))]
            candidates = [g for g in self.groups if g["corpus"] == corpus]
            group = candidates[int(self.rng.integers(len(candidates)))]
            window = group["windows"][int(self.rng.integers(len(group["windows"])))]
            lift_entry = group["lifts"][int(self.rng.integers(len(group["lifts"])))]
            if isinstance(lift_entry, tuple):
                lift, keep = lift_entry
                out[index] = (lift @ window[keep]).astype(np.float32)
            else:
                out[index] = (lift_entry @ window).astype(np.float32)
        return out

    def contaminate(self, clean: np.ndarray, u_canon: np.ndarray):
        y = clean.copy()
        for index in range(len(clean)):
            if self.drive_bank is None:
                continue
            zero = self.rng.random() < 0.40
            if zero:
                continue
            gain = float(np.exp(self.rng.uniform(np.log(0.05), np.log(1.3))))
            drive = self.drive_bank[int(self.rng.integers(len(self.drive_bank)))]
            theta = self.rng.uniform(0, 2 * np.pi)
            mix = np.asarray([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
            y[index] += (u_canon @ (gain * (mix @ drive))).astype(np.float32)
        return y


def w2_train(run: str, seed: int, updates: int) -> None:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.training.train_v42r import EMA
    from eeg_chart.prior_model import CanonicalPrior, ddim_denoise

    result_dir = RESULT / "w2_prior" / f"{run}_seed_{seed}"
    curve_path = result_dir / "train_curve.json"
    if curve_path.is_file():
        print(json.dumps({"run": run, "seed": seed, "skipped": "complete"}))
        return
    device = torch.device("cuda")
    torch.manual_seed(seed)
    runtime = PRIOR_DERIVED / "w2_prior" / f"{run}_seed_{seed}"
    runtime.mkdir(parents=True, exist_ok=True)
    canon = np.load(_canon_path())["u_canon"]
    pool = _CorpusPool(RUN_CORPORA[run], seed)
    ambient = run == "ambient"
    model = CanonicalPrior().to(device)
    schedule = LinearX0Schedule().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    ema = EMA(model, .999)
    generator = torch.Generator(device=device).manual_seed(seed + 7001)
    start_step, curve, best = 0, [], float("inf")
    last_path, best_path = runtime / "last.pt", runtime / "best.pt"
    if last_path.is_file():  # checkpoint-resume across submissions
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        ema.model.load_state_dict(payload["ema"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = payload["step"]
        curve = payload["curve"]
        best = payload["best_validation_rrmse"]
        print(json.dumps({"resumed_at": start_step}))
    validation_clean = pool.sample(24)
    validation_y = pool.contaminate(validation_clean, canon)
    batch = 16
    for step in range(start_step + 1, updates + 1):
        clean = pool.sample(batch)
        observed = clean.copy() if ambient else pool.contaminate(clean, canon)
        target = observed if ambient else clean
        clean_t = torch.from_numpy(target).to(device)
        observed_t = torch.from_numpy(observed).to(device)
        noisy, timestep, _ = schedule.forward_sample(clean_t, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.smooth_l1_loss(model(noisy, observed_t, timestep), clean_t)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite prior loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        if step % 2000 == 0 or step == updates:
            with torch.no_grad():
                vy = torch.from_numpy(validation_y).to(device)
                noise = torch.randn(vy.shape, device=device,
                                    generator=torch.Generator(device=device).manual_seed(seed + 99))
                out = ddim_denoise(ema.model, vy, noise, schedule, 50).cpu().numpy()
            score = float(np.mean([np.linalg.norm(o - c) / max(np.linalg.norm(c), 1e-12)
                                   for o, c in zip(out, validation_clean)]))
            curve.append({"step": step, "loss": float(loss.detach()),
                          "validation_rrmse": score})
            payload = {"model": model.state_dict(), "ema": ema.state_dict(),
                       "optimizer": optimizer.state_dict(), "step": step, "curve": curve,
                       "best_validation_rrmse": min(best, score), "run": run, "seed": seed}
            torch.save(payload, last_path)
            if score < best:
                best = score
                torch.save({"ema": ema.state_dict(), "step": step,
                            "validation_rrmse": score}, best_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    curve_path.write_text(json.dumps({
        "run": run, "seed": seed, "updates": updates, "parameters": parameters,
        "corpora": RUN_CORPORA[run], "ambient": ambient,
        "checkpoint": str(best_path), "curve": curve,
        "best_validation_rrmse": best, "sealed_reads": 0}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run": run, "seed": seed, "best_validation_rrmse": best,
                      "parameters": parameters}))


# ------------------------------------------------------------------- W2 eval

def _eval_pairs(panel: str):
    """100 Hz paired episodes per panel + per-subject POP transports (rho=0)."""
    from eeg_chart.prior_data import _resample_100
    cells, lift = _load_panel(panel)
    canon = np.load(_canon_path())["u_canon"]
    context = transport_context(cells, lift, canon)
    episodes = []
    for cell_id, entry in sorted(context["per_cell"].items()):
        cell = entry["cell"]
        arm = transport_family(context["lift"], context["lift_pinv"], context["sigma_bar"],
                               None, entry["base"], entry["base"], 0.0)
        fs = {"mobilebci": 100.0, "klados": 200.0, "bci2b": 250.0}[panel]
        unit = cell.subject if panel != "klados" else cell.cell
        for episode in cell.episodes:
            x_r = _resample_100(episode["x"], fs)
            y_r = _resample_100(episode["y"], fs)
            n_valid = min(x_r.shape[1], y_r.shape[1], 512)
            if n_valid < 64:
                continue
            pad = 512 - n_valid
            # Native 512-sample windows shrink under the registered 100 Hz
            # unification; reflect-pad to the model length and score only the
            # valid segment.
            x = np.pad(x_r[:, :n_valid], ((0, 0), (0, pad)), mode="reflect") if pad \
                else x_r[:, :512]
            y = np.pad(y_r[:, :n_valid], ((0, 0), (0, pad)), mode="reflect") if pad \
                else y_r[:, :512]
            episodes.append({"unit": unit, "x": x, "y": y, "n_valid": n_valid,
                             "transport": arm.transport, "pinv": arm.pinv})
    return episodes, context


def w2_eval(run: str, seed: int) -> None:
    import torch
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_chart.prior_model import CanonicalPrior, ddim_denoise

    result_dir = RESULT / "w2_prior" / f"{run}_seed_{seed}"
    out_path = result_dir / "eval.json"
    if out_path.is_file():
        print(json.dumps({"run": run, "seed": seed, "skipped": "eval complete"}))
        return
    source = json.loads((result_dir / "train_curve.json").read_text())
    device = torch.device("cuda")
    model = CanonicalPrior().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    model.eval()
    schedule = LinearX0Schedule().to(device)
    canon = np.load(_canon_path())["u_canon"]
    results = {}
    for panel in PAIRED:
        episodes, context = _eval_pairs(panel)
        sigma_inv = context["sigma_bar_inv"]
        gram = canon.T @ sigma_inv @ canon + 1e-8 * np.eye(canon.shape[1])
        raw_by_unit, route_by_unit, rms_by_unit = {}, {}, {}
        for start in range(0, len(episodes), 8):
            chunk = episodes[start:start + 8]
            y_canon = np.stack([e["transport"] @ e["y"] for e in chunk]).astype(np.float32)
            y_t = torch.from_numpy(y_canon).to(device)
            noise = torch.randn(y_t.shape, device=device,
                                generator=torch.Generator(device=device).manual_seed(
                                    424242 + start))
            x0 = ddim_denoise(model, y_t, noise, schedule, 50).cpu().numpy()
            for episode, y_c, x_c in zip(chunk, y_canon, x0):
                residual = y_c - x_c
                coefficients = np.linalg.solve(gram, canon.T @ sigma_inv @ residual)
                cleaned_canon = y_c - canon @ coefficients   # analytic likelihood step
                x_hat = episode["pinv"] @ cleaned_canon
                n = episode["n_valid"]
                x_v, y_v, xh_v = episode["x"][:, :n], episode["y"][:, :n], x_hat[:, :n]
                raw = float(np.linalg.norm(y_v - x_v) / max(np.linalg.norm(x_v), 1e-12))
                route = float(np.linalg.norm(xh_v - x_v) / max(np.linalg.norm(x_v), 1e-12))
                rms = float(np.sqrt(np.mean(xh_v ** 2))
                            / max(np.sqrt(np.mean(y_v ** 2)), 1e-12))
                raw_by_unit.setdefault(episode["unit"], []).append(raw)
                route_by_unit.setdefault(episode["unit"], []).append(route)
                rms_by_unit.setdefault(episode["unit"], []).append(rms)
        units = sorted(raw_by_unit)
        utility = [float(np.mean(raw_by_unit[u]) - np.mean(route_by_unit[u])) for u in units]
        rms_q99 = float(np.quantile([np.mean(rms_by_unit[u]) for u in units], .99))
        stat = _stat(utility)
        results[panel] = {"pv1_utility_raw_minus_route": stat,
                          "pv1_pass": bool(stat["mean"] > 0 and stat["bootstrap_low"] > 0),
                          "route_mean_rrmse": float(np.mean([np.mean(route_by_unit[u])
                                                             for u in units])),
                          "raw_mean_rrmse": float(np.mean([np.mean(raw_by_unit[u])
                                                           for u in units])),
                          "pv2_rms_q99": rms_q99,
                          "pv2_pass": bool(PV2_BAND[0] <= rms_q99 <= PV2_BAND[1]),
                          "units": len(units)}
    out_path.write_text(json.dumps({"run": run, "seed": seed, "panels": results},
                                   indent=2, sort_keys=True) + "\n")
    print(json.dumps({panel: {"pv1": results[panel]["pv1_pass"],
                              "pv2": results[panel]["pv2_pass"]} for panel in PAIRED}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="unit", required=True)
    sub.add_parser("w1")
    sub.add_parser("w2-harvest")
    train = sub.add_parser("w2-train")
    train.add_argument("--run", required=True, choices=sorted(RUN_CORPORA))
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--updates", type=int, default=60000)
    evaluate = sub.add_parser("w2-eval")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    w4 = sub.add_parser("w4")
    w4.add_argument("--fold", type=int, required=True)
    w4.add_argument("--seed", type=int, required=True)
    w4.add_argument("--chains", type=int, default=8)
    sub.add_parser("w4-aggregate")
    args = parser.parse_args()
    if args.unit == "w1":
        w1()
    elif args.unit == "w2-harvest":
        w2_harvest()
    elif args.unit == "w2-train":
        w2_train(args.run, args.seed, args.updates)
    elif args.unit == "w2-eval":
        w2_eval(args.run, args.seed)
    elif args.unit == "w4":
        from eeg_chart.posterior_sampling import run_cell
        run_cell(args.fold, args.seed, args.chains, RESULT / "w4_uq")
    elif args.unit == "w4-aggregate":
        from eeg_chart.posterior_sampling import aggregate as w4_aggregate
        payload = w4_aggregate(RESULT / "w4_uq", (20261201, 20261202, 20261203))
        (RESULT / "w4_uq").mkdir(parents=True, exist_ok=True)
        (RESULT / "w4_uq" / "decision.json").write_text(json.dumps(payload, indent=2,
                                                                   sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
