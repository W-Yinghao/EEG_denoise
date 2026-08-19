#!/usr/bin/env python3
"""S356 — real-data scale conditioning probe (the program's final experiment).

Preregistered in reports/iris_prereg_s356.md (frozen before execution). Modes:

  prep       CPU  per-subject npz: 46ch @100 Hz + latent [VEOG,HEOG] + halves
  episodes   CPU  frozen paired-injection episode banks per subject
  train      GPU  one (n, arm) run: COND (FiLM subject embedding) or BLIND twin
  evaluate   GPU  oracle-embedding protocol on the 15 frozen EVAL subjects
  aggregate  CPU  gates G-S356-seal / G-S356-overturn, decision JSON

All seeds frozen (base 20260818). Sealed tree never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/iris"))
MIN_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_min/antisaccade_min")
EXT_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_ext/antisaccade_min")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_s356")
OUT_DIR = REPO / "results/iris/s356"
EVAL_SUBJECTS = ("AA0", "AA1", "AA4", "AA5", "AA7", "AA8", "AA9", "AB0", "AB1",
                 "AB2", "AB3", "AB6", "AB7", "AB9", "AC0")
FS_OUT = 100
WINDOW = 512
N_CHANNELS = 46
FRONTAL_N = 8
RIDGE = 0.05
EPISODES_PER_SUBJECT = 64
CLEAN_QUANTILE, DRIVE_QUANTILE = 0.30, 0.70
GRID = (30, 60, 120, 200, -1)                # -1 = N_max (full training pool)
SEED = 20260818
TRAIN_STEPS = 20_000
BATCH = 32
LR = 2e-4
EMA = 0.999
EMB_DIM = 32
ORACLE_STEPS = 200
ORACLE_LR = 1e-2
GATE_EPS = 0.02
BOOT_SEED, BOOT_DRAWS = 420, 5000
PERI = ("E8", "E25", "E125", "E126", "E127", "E128")


# ------------------------------------------------------------------ prep

def _subject_dirs() -> list[Path]:
    dirs = [d for d in sorted(MIN_ROOT.iterdir()) if d.is_dir()]
    if EXT_ROOT.exists():
        dirs += [d for d in sorted(EXT_ROOT.iterdir()) if d.is_dir()]
    return dirs


def prep() -> None:
    import h5py
    from scipy.signal import resample_poly
    from k3_instrument import _h5_num, _h5_text

    (DERIVED / "subjects").mkdir(parents=True, exist_ok=True)
    report = []
    for subject_dir in _subject_dirs():
        subject = subject_dir.name
        out = DERIVED / "subjects" / f"{subject}.npz"
        if out.is_file():
            report.append({"subject": subject, "state": "cached"})
            continue
        mats = sorted(subject_dir.glob("*.mat"))
        if not mats:
            report.append({"subject": subject, "state": "empty"})
            continue
        try:
            with h5py.File(mats[0], "r") as handle:
                eeg = handle["EEG"]
                srate = float(eeg["srate"][0, 0])
                labels = [_h5_text(handle, r)
                          for r in np.asarray(eeg["chanlocs"]["labels"]).ravel()]
                x_coord = np.asarray(
                    [_h5_num(handle, r)
                     for r in np.asarray(eeg["chanlocs"]["X"]).ravel()])
                idx = {lab: i for i, lab in enumerate(labels)}
                e_chans = [lab for lab in labels
                           if lab.startswith("E") and lab not in PERI
                           and lab[1:].isdigit()]
                frontal = sorted(e_chans, key=lambda lab: -x_coord[idx[lab]])[:FRONTAL_N]
                rest = sorted((lab for lab in e_chans if lab not in frontal),
                              key=lambda lab: int(lab[1:]))
                pick = np.unique(np.round(np.linspace(
                    0, len(rest) - 1, N_CHANNELS - FRONTAL_N)).astype(int))
                chans = frontal + [rest[i] for i in pick][:N_CHANNELS - FRONTAL_N]
                want = sorted(set(idx[lab] for lab in chans + list(PERI)))
                cols = eeg["data"][:, want].astype(np.float32).T
                col = {c: j for j, c in enumerate(want)}
                bad = set()
                auto = handle.get("automagic")
                if auto is not None:
                    for key, sub in (("finalBadChans", None),
                                     ("interpolation", "channels")):
                        node = auto.get(key)
                        if node is not None and sub is not None:
                            node = node.get(sub) if hasattr(node, "get") else None
                        if node is not None and hasattr(node, "shape"):
                            values = np.asarray(node[()]).ravel()
                            if values.dtype.kind in "fiu":
                                bad |= {int(v) for v in values
                                        if np.isfinite(v) and v > 0}
        except Exception as error:                    # noqa: BLE001 - reason-coded
            report.append({"subject": subject, "state": f"error {error}"})
            continue
        factor = int(round(srate / FS_OUT))
        get = lambda lab: resample_poly(cols[col[idx[lab]]].astype(np.float64),  # noqa: E731
                                        1, factor)
        data = np.stack([get(lab) for lab in chans]).astype(np.float32)
        veog = ((get("E25") - get("E127")) + (get("E8") - get("E126"))) / 2.0
        heog = get("E125") - get("E128")
        peri_interp = sorted(bad & {int(lab[1:]) for lab in PERI})
        np.savez_compressed(out, data=data, veog=veog.astype(np.float32),
                            heog=heog.astype(np.float32),
                            channels=np.asarray(chans),
                            peri_interp=np.asarray(peri_interp, int))
        report.append({"subject": subject, "state": "done",
                       "samples": int(data.shape[1]),
                       "peri_interp": peri_interp})
        print(json.dumps(report[-1]), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "prep_report.json").write_text(json.dumps(report, indent=1) + "\n")
    states = {}
    for r in report:
        states[r["state"].split()[0]] = states.get(r["state"].split()[0], 0) + 1
    print(json.dumps(states))


# ------------------------------------------------------------------ episodes

def _bandpass(x, low, high, fs):
    from scipy import signal
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def _episodes_for(subject: str, half: str, rng: np.random.Generator):
    d = np.load(DERIVED / "subjects" / f"{subject}.npz")
    data = d["data"].astype(np.float64)
    veog = _bandpass(d["veog"].astype(np.float64), 0.5, 8.0, FS_OUT)
    heog = _bandpass(d["heog"].astype(np.float64), 0.5, 20.0, FS_OUT)
    n = data.shape[1]
    lo, hi = (0, n // 2) if half == "support" else (n // 2, n)
    seg = slice(lo, hi)
    latent = np.stack([veog, heog])
    center = np.median(latent[:, 0:n // 2], axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(latent[:, 0:n // 2] - center),
                               axis=1, keepdims=True)
    latent = (latent - center) / np.maximum(scale, 1e-9)
    eeg_scale = float(np.sqrt(np.mean(data[:, 0:n // 2] ** 2)))
    block = data[:, seg] / max(eeg_scale, 1e-9)
    lat = latent[:, seg]
    n_win = block.shape[1] // WINDOW
    wins = block[:, :n_win * WINDOW].reshape(N_CHANNELS, n_win, WINDOW)
    lwins = lat[:, :n_win * WINDOW].reshape(2, n_win, WINDOW)
    energy = (lwins ** 2).mean(axis=(0, 2))
    order = np.argsort(energy)
    clean_pool = order[:max(int(n_win * CLEAN_QUANTILE), 1)]
    drive_pool = order[-max(int(n_win * (1 - DRIVE_QUANTILE)), 1):]
    # subject operator on SUPPORT (always), program static ridge 46x2
    sup = data[:, 0:n // 2] / max(eeg_scale, 1e-9)
    sup_lat = latent[:, 0:n // 2]
    y_c = sup - sup.mean(axis=1, keepdims=True)
    e_c = sup_lat - sup_lat.mean(axis=1, keepdims=True)
    gram = e_c @ e_c.T
    ridge = RIDGE * max(float(np.trace(gram) / len(gram)), 1e-12)
    C_s = (y_c @ e_c.T) @ np.linalg.inv(gram + ridge * np.eye(2))
    xs, ys = [], []
    for _ in range(EPISODES_PER_SUBJECT):
        x = wins[:, rng.choice(clean_pool)]
        e = lwins[:, rng.choice(drive_pool)]
        e = np.roll(e, int(rng.integers(0, WINDOW)), axis=1)
        ys.append((x + C_s @ e).astype(np.float32))
        xs.append(x.astype(np.float32))
    return np.stack(xs), np.stack(ys)


def episodes() -> None:
    (DERIVED / "episodes").mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.stem for p in (DERIVED / "subjects").glob("*.npz"))
    for subject in subjects:
        out = DERIVED / "episodes" / f"{subject}.npz"
        if out.is_file():
            continue
        rng = np.random.default_rng(SEED + hash_stable(subject))
        xs, ys = _episodes_for(subject, "support", rng)
        payload = {"sup_x": xs, "sup_y": ys}
        if subject in EVAL_SUBJECTS:
            xq, yq = _episodes_for(subject, "query", rng)
            payload.update({"qry_x": xq, "qry_y": yq})
        np.savez_compressed(out, **payload)
        print(json.dumps({"subject": subject, "eval": subject in EVAL_SUBJECTS}),
              flush=True)
    print(json.dumps({"episodes_built": len(subjects)}))


def hash_stable(text: str) -> int:
    import zlib
    return zlib.crc32(text.encode()) % 100_000


# ------------------------------------------------------------------ model

def build_model(n_subjects: int):
    import torch
    import torch.nn as nn

    class FiLMBlock(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv = nn.Conv1d(cin, cout, 5, stride=stride, padding=2)
            self.norm = nn.GroupNorm(8, cout)
            self.film = nn.Linear(EMB_DIM, 2 * cout)
            self.act = nn.SiLU()

        def forward(self, x, emb):
            h = self.norm(self.conv(x))
            scale, shift = self.film(emb).chunk(2, dim=-1)
            h = h * (1 + scale[..., None]) + shift[..., None]
            return self.act(h)

    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(max(n_subjects, 1), EMB_DIM)
            nn.init.zeros_(self.embed.weight)
            self.down1 = FiLMBlock(N_CHANNELS, 64, 1)
            self.down2 = FiLMBlock(64, 128, 2)
            self.down3 = FiLMBlock(128, 128, 2)
            self.mid = FiLMBlock(128, 128, 1)
            self.up2 = nn.ConvTranspose1d(128, 128, 4, stride=2, padding=1)
            self.upb2 = FiLMBlock(256, 64, 1)
            self.up1 = nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1)
            self.upb1 = FiLMBlock(128, 64, 1)
            self.out = nn.Conv1d(64, N_CHANNELS, 3, padding=1)

        def forward(self, y, emb):
            h1 = self.down1(y, emb)
            h2 = self.down2(h1, emb)
            h3 = self.down3(h2, emb)
            m = self.mid(h3, emb)
            u2 = self.upb2(torch.cat([self.up2(m), h2], dim=1), emb)
            u1 = self.upb1(torch.cat([self.up1(u2), h1], dim=1), emb)
            return self.out(u1)

    return Denoiser()


def _training_pool() -> list[str]:
    have = {p.stem for p in (DERIVED / "subjects").glob("*.npz")}
    dev = [d.name for d in sorted(MIN_ROOT.iterdir()) if d.is_dir()
           and d.name in have and d.name not in EVAL_SUBJECTS]
    ext = ([d.name for d in sorted(EXT_ROOT.iterdir()) if d.is_dir()
            and d.name in have] if EXT_ROOT.exists() else [])
    return dev + ext


def train(n_arg: int, arm: str) -> None:
    import torch

    pool = _training_pool()
    n = len(pool) if n_arg == -1 else n_arg
    subjects = pool[:n]
    tag = f"n{n}_{arm}"
    ckpt = DERIVED / f"model_{tag}.pt"
    if ckpt.is_file():
        print(json.dumps({"tag": tag, "skipped": True}))
        return
    device = torch.device("cuda")
    xs, ys, sid = [], [], []
    for s_index, subject in enumerate(subjects):
        d = np.load(DERIVED / "episodes" / f"{subject}.npz")
        xs.append(d["sup_x"])
        ys.append(d["sup_y"])
        sid.append(np.full(len(d["sup_x"]), s_index))
    x = torch.from_numpy(np.concatenate(xs)).to(device)
    y = torch.from_numpy(np.concatenate(ys)).to(device)
    sid = torch.from_numpy(np.concatenate(sid)).long().to(device)
    model = build_model(n).to(device)
    ema_model = build_model(n).to(device)
    ema_model.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    gen = torch.Generator(device="cpu").manual_seed(SEED + n * 7 + (arm == "COND"))
    for step in range(TRAIN_STEPS):
        pick = torch.randint(0, len(x), (BATCH,), generator=gen).to(device)
        emb = (model.embed(sid[pick]) if arm == "COND"
               else torch.zeros(BATCH, EMB_DIM, device=device))
        pred = model(y[pick], emb)
        loss = torch.mean((pred - (y[pick] - x[pick])) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(EMA).add_(p, alpha=1 - EMA)
        if step % 2000 == 0:
            print(json.dumps({"tag": tag, "step": step,
                              "loss": float(loss)}), flush=True)
    torch.save({"ema": ema_model.state_dict(), "n": n, "arm": arm,
                "subjects": subjects}, ckpt)
    print(json.dumps({"tag": tag, "done": True, "n_subjects": n}))


def evaluate() -> None:
    import torch
    device = torch.device("cuda")
    pool = _training_pool()
    rows = []
    for n_arg in GRID:
        n = len(pool) if n_arg == -1 else n_arg
        for arm in ("COND", "BLIND"):
            ckpt = torch.load(DERIVED / f"model_n{n}_{arm}.pt",
                              map_location=device, weights_only=False)
            model = build_model(ckpt["n"]).to(device)
            model.load_state_dict(ckpt["ema"])
            model.eval()
            for subject in EVAL_SUBJECTS:
                d = np.load(DERIVED / "episodes" / f"{subject}.npz")
                sx = torch.from_numpy(d["sup_x"]).to(device)
                sy = torch.from_numpy(d["sup_y"]).to(device)
                qx = torch.from_numpy(d["qry_x"]).to(device)
                qy = torch.from_numpy(d["qry_y"]).to(device)

                def rrmse(emb_vec):
                    with torch.no_grad():
                        emb = emb_vec[None].expand(len(qy), -1)
                        xh = qy - model(qy, emb)
                        num = torch.linalg.vector_norm(xh - qx, dim=(1, 2))
                        den = torch.linalg.vector_norm(qx, dim=(1, 2)).clamp(1e-9)
                    return float((num / den).mean())

                zero = torch.zeros(EMB_DIM, device=device)
                row = {"n": n, "arm": arm, "subject": subject,
                       "rrmse_zero": rrmse(zero)}
                if arm == "COND":
                    emb = torch.zeros(EMB_DIM, device=device, requires_grad=True)
                    opt = torch.optim.Adam([emb], lr=ORACLE_LR)
                    for _ in range(ORACLE_STEPS):
                        e = emb[None].expand(len(sy), -1)
                        loss = torch.mean((model(sy, e) - (sy - sx)) ** 2)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    row["rrmse_oracle"] = rrmse(emb.detach())
                    row["gain"] = row["rrmse_zero"] - row["rrmse_oracle"]
                rows.append(row)
                print(json.dumps(row), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_rows.json").write_text(json.dumps(rows, indent=1) + "\n")


def _stat(values):
    series = np.asarray(list(values), float)
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.asarray([rng.choice(series, len(series), replace=True).mean()
                        for _ in range(BOOT_DRAWS)])
    return {"mean": float(series.mean()), "n": int(len(series)),
            "positive_count": int((series > 0).sum()),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def aggregate() -> None:
    rows = json.loads((OUT_DIR / "eval_rows.json").read_text())
    ns = sorted({r["n"] for r in rows})
    n_max, n_min = max(ns), min(ns)
    gain = {n: _stat([r["gain"] for r in rows
                      if r["n"] == n and r["arm"] == "COND"]) for n in ns}
    trend = _stat([
        next(r["gain"] for r in rows if r["n"] == n_max and r["arm"] == "COND"
             and r["subject"] == s)
        - next(r["gain"] for r in rows if r["n"] == n_min and r["arm"] == "COND"
               and r["subject"] == s)
        for s in EVAL_SUBJECTS])
    blind = {n: float(np.mean([r["rrmse_zero"] for r in rows
                               if r["n"] == n and r["arm"] == "BLIND"]))
             for n in ns}
    guard = bool(blind[n_max] <= blind[n_min] + 0.02)
    seal = bool(gain[n_max]["bootstrap_high"] < GATE_EPS
                and trend["bootstrap_high"] < GATE_EPS)
    overturn = bool(gain[n_max]["bootstrap_low"] > GATE_EPS)
    verdict = ("C1_SEALED" if (seal and guard) else
               "C1_OVERTURNED" if (overturn and guard) else
               "INSTRUMENT_INVALID" if not guard else "INCONCLUSIVE")
    decision = {
        "prereg": "reports/iris_prereg_s356.md",
        "grid": ns, "gain_by_n": {str(n): gain[n] for n in ns},
        "trend_max_minus_min": trend,
        "blind_rrmse_by_n": blind, "training_guard_pass": guard,
        "gates": {"seal": seal, "overturn": overturn, "epsilon": GATE_EPS},
        "verdict": verdict,
    }
    (OUT_DIR / "s356_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict,
                      "gain_at_max": round(gain[n_max]["mean"], 5),
                      "ci": [round(gain[n_max]["bootstrap_low"], 5),
                             round(gain[n_max]["bootstrap_high"], 5)]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("prep", "episodes", "evaluate", "aggregate",
                 "evaluate2", "aggregate2"):
        sub.add_parser(name)
    t = sub.add_parser("train")
    t.add_argument("--n", type=int, required=True)
    t.add_argument("--arm", choices=["COND", "BLIND"], required=True)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.n, args.arm)
    else:
        {"prep": prep, "episodes": episodes, "evaluate": evaluate,
         "aggregate": aggregate, "evaluate2": evaluate2,
         "aggregate2": aggregate2}[args.mode]()




# ------------------------------------------------- amendment S356-1 (b1dcc2f)

def _injection_ratios() -> dict[str, dict[str, float]]:
    out = {}
    for subject in EVAL_SUBJECTS:
        d = np.load(DERIVED / "episodes" / f"{subject}.npz")
        out[subject] = {
            "support": float(np.sqrt(np.mean((d["sup_y"] - d["sup_x"]) ** 2))
                            / max(np.sqrt(np.mean(d["sup_x"] ** 2)), 1e-12)),
            "query": float(np.sqrt(np.mean((d["qry_y"] - d["qry_x"]) ** 2))
                           / max(np.sqrt(np.mean(d["qry_x"] ** 2)), 1e-12))}
    return out


def _guarded() -> list[str]:
    ratios = _injection_ratios()
    return [s for s in EVAL_SUBJECTS
            if 0.1 <= ratios[s]["support"] <= 20.0
            and 0.1 <= ratios[s]["query"] <= 20.0]


def evaluate2() -> None:
    import torch
    device = torch.device("cuda")
    pool = _training_pool()
    guarded = _guarded()
    rows = []
    for n_arg in (30, -1):
        n = len(pool) if n_arg == -1 else n_arg
        ckpt = torch.load(DERIVED / f"model_n{n}_COND.pt",
                          map_location=device, weights_only=False)
        model = build_model(ckpt["n"]).to(device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        banks = {s: np.load(DERIVED / "episodes" / f"{s}.npz") for s in guarded}
        embeddings = {}
        for subject in guarded:
            d = banks[subject]
            sx = torch.from_numpy(d["sup_x"]).to(device)
            sy = torch.from_numpy(d["sup_y"]).to(device)
            emb = torch.zeros(EMB_DIM, device=device, requires_grad=True)
            opt = torch.optim.Adam([emb], lr=ORACLE_LR)
            for _ in range(ORACLE_STEPS):
                e = emb[None].expand(len(sy), -1)
                loss = torch.mean((model(sy, e) - (sy - sx)) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            embeddings[subject] = emb.detach()

        def rrmse(subject, emb_vec):
            d = banks[subject]
            qx = torch.from_numpy(d["qry_x"]).to(device)
            qy = torch.from_numpy(d["qry_y"]).to(device)
            with torch.no_grad():
                xh = qy - model(qy, emb_vec[None].expand(len(qy), -1))
                num = torch.linalg.vector_norm(xh - qx, dim=(1, 2))
                den = torch.linalg.vector_norm(qx, dim=(1, 2)).clamp(1e-9)
            return float((num / den).mean())

        zero = torch.zeros(EMB_DIM, device=device)
        for subject in guarded:
            wrongs = [rrmse(subject, embeddings[other])
                      for other in guarded if other != subject]
            rows.append({"n": n, "subject": subject,
                         "rrmse_zero": rrmse(subject, zero),
                         "rrmse_own": rrmse(subject, embeddings[subject]),
                         "rrmse_wrong_mean": float(np.mean(wrongs))})
            print(json.dumps(rows[-1]), flush=True)
    (OUT_DIR / "eval2_rows.json").write_text(json.dumps(rows, indent=1) + "\n")


def aggregate2() -> None:
    banked = json.loads((OUT_DIR / "eval_rows.json").read_text())
    e2 = json.loads((OUT_DIR / "eval2_rows.json").read_text())
    ratios = _injection_ratios()
    guarded = _guarded()
    excluded = [s for s in EVAL_SUBJECTS if s not in guarded]
    ns = sorted({r["n"] for r in banked})
    n_max, n_min = max(ns), min(ns)
    gain = {n: _stat([r["gain"] for r in banked
                      if r["n"] == n and r["arm"] == "COND"
                      and r["subject"] in guarded]) for n in ns}
    median_gain = {n: float(np.median([r["gain"] for r in banked
                                       if r["n"] == n and r["arm"] == "COND"
                                       and r["subject"] in guarded])) for n in ns}
    trend = _stat([
        next(r["gain"] for r in banked if r["n"] == n_max and r["arm"] == "COND"
             and r["subject"] == s)
        - next(r["gain"] for r in banked if r["n"] == n_min and r["arm"] == "COND"
               and r["subject"] == s) for s in guarded])
    spec = {}
    for n in sorted({r["n"] for r in e2}):
        sub = [r for r in e2 if r["n"] == n]
        spec[str(n)] = {
            "own_gain": _stat([r["rrmse_zero"] - r["rrmse_own"] for r in sub]),
            "wrong_gain": _stat([r["rrmse_zero"] - r["rrmse_wrong_mean"]
                                 for r in sub]),
            "own_minus_wrong": _stat([r["rrmse_wrong_mean"] - r["rrmse_own"]
                                      for r in sub])}
    seal = bool(gain[n_max]["bootstrap_high"] < GATE_EPS
                and trend["bootstrap_high"] < GATE_EPS)
    overturn_ci = bool(gain[n_max]["bootstrap_low"] > GATE_EPS)
    flat = bool(trend["bootstrap_low"] <= 0.0 <= trend["bootstrap_high"])
    specific = bool(spec[str(n_max)]["own_minus_wrong"]["bootstrap_low"] > 0)
    if seal:
        verdict = "C1_SEALED"
    elif overturn_ci and not flat:
        verdict = "THRESHOLD_RESURFACES"
    elif overturn_ci and flat and specific:
        verdict = "SCOPED_C1_COUNTEREXAMPLE_FLAT_SUBJECT_SPECIFIC"
    elif overturn_ci and not specific:
        verdict = "PROTOCOL_GENERIC_GAIN_C1_UNTHREATENED"
    else:
        verdict = "INCONCLUSIVE"
    decision = {
        "prereg": "reports/iris_prereg_s356.md (amendment S356-1, b1dcc2f)",
        "banked_itt_verdict": "INCONCLUSIVE (never edited)",
        "guard": {"bounds": [0.1, 20.0], "excluded": excluded,
                  "ratios": ratios, "n_guarded": len(guarded)},
        "gain_by_n_guarded": {str(n): gain[n] for n in ns},
        "median_gain_by_n": median_gain,
        "trend_max_minus_min": trend, "trend_flat": flat,
        "subject_specificity": spec, "specific_at_n_max": specific,
        "gates": {"seal": seal, "overturn_ci": overturn_ci,
                  "epsilon": GATE_EPS},
        "verdict": verdict,
    }
    (OUT_DIR / "s356_decision_amended.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict,
                      "gain_max": round(gain[n_max]["mean"], 4),
                      "ci": [round(gain[n_max]["bootstrap_low"], 4),
                             round(gain[n_max]["bootstrap_high"], 4)],
                      "own_minus_wrong":
                          round(spec[str(n_max)]["own_minus_wrong"]["mean"], 4),
                      "specific": specific}))


if __name__ == "__main__":
    main()
