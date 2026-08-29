#!/usr/bin/env python3
"""EEGEyeNet sealed-55 confirmation — driver.

Preregistered in reports/iris_prereg_sealed55.md (frozen before execution).
Option A protocol: the frozen S356 checkpoints are evaluated, unchanged, on the 55
sealed subjects; per-subject 32-d embeddings are fitted on each subject's own SUPPORT
episodes (declared evaluator-only ceiling, model weights frozen) and scored on the
disjoint QUERY episodes against zero and wrong-owner embeddings.

All scientific machinery is imported from scripts/iris/s356_probe.py rather than
reimplemented; only the roots and the cohort change. The frozen iris_s356 derived tree
is never written to.

Modes
  probe     dev-class dry run on eegeyenet_ext subjects — NO sealed contact
  open      write the opening record and chmod the sealed tree (refuses without
            --signoff and --option; refuses if a record already exists)
  run       prep + episodes + inference on the sealed cohort, raw rows banked and
            sha256-frozen before any aggregation (refuses without an opening record)
  aggregate adjudicate the frozen gates of prereg section 4
  reseal    restore chmod 000 after the run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/iris"))

MODEL_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_s356")
SEALED_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_sealed")
SEALED_SUBJECTS_ROOT = SEALED_ROOT / "antisaccade_min"
EXT_ROOT = Path("/projects/EEG-foundation-model/eegeyenet/eegeyenet_ext/antisaccade_min")
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_sealed_confirm")
OUT_DIR = REPO / "results/iris/sealed_confirm"
FREEZE_RECORD = REPO / "results/iris/sealed/sealed_freeze.json"
OPENING_RECORD = REPO / "results/iris/sealed/sealed_opening_record.json"
PREREG = "reports/iris_prereg_sealed55.md"

GATE_EPS = 0.02
GUARD_LO, GUARD_HI = 0.1, 20.0
MAX_GUARD_EXCLUSIONS = 11          # QC1: 20% of 55
BLIND_SANITY_TOL = 0.05            # QC3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _bind(module, root: Path, derived: Path, out_dir: Path, cohort: tuple[str, ...]):
    """Point the frozen S356 machinery at a different cohort and derived root."""
    module.MIN_ROOT = root
    module.EXT_ROOT = Path("/nonexistent")   # _subject_dirs must see one root only
    module.DERIVED = derived
    module.OUT_DIR = out_dir
    module.EVAL_SUBJECTS = cohort


def _cohort(root: Path) -> tuple[str, ...]:
    return tuple(sorted(d.name for d in root.iterdir() if d.is_dir()))


def _guarded(derived: Path, cohort) -> tuple[list[str], dict]:
    ratios = {}
    for subject in cohort:
        path = derived / "episodes" / f"{subject}.npz"
        if not path.is_file():
            continue
        with np.load(path) as d:
            if "qry_x" not in d:
                continue
            ratios[subject] = {
                "support": float(np.sqrt(np.mean((d["sup_y"] - d["sup_x"]) ** 2))
                                 / max(np.sqrt(np.mean(d["sup_x"] ** 2)), 1e-12)),
                "query": float(np.sqrt(np.mean((d["qry_y"] - d["qry_x"]) ** 2))
                               / max(np.sqrt(np.mean(d["qry_x"] ** 2)), 1e-12))}
    keep = [s for s, r in sorted(ratios.items())
            if GUARD_LO <= r["support"] <= GUARD_HI
            and GUARD_LO <= r["query"] <= GUARD_HI]
    return keep, ratios


def _stat(values) -> dict:
    value = np.asarray(values, float)
    rng = np.random.default_rng(420)
    draws = np.asarray([rng.choice(value, len(value), replace=True).mean()
                        for _ in range(5000)])
    return {"mean": float(value.mean()), "median": float(np.median(value)),
            "positive_count": int((value > 0).sum()), "subjects": int(len(value)),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def _prepare(root: Path, derived: Path, out_dir: Path, cohort) -> None:
    import s356_probe as s356
    _bind(s356, root, derived, out_dir, cohort)
    derived.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    s356.prep()
    s356.episodes()


def _infer(derived: Path, cohort, arms=(-1, 30), blind: bool = True) -> list[dict]:
    """Frozen-checkpoint inference. Models are read-only from MODEL_ROOT."""
    import torch
    import s356_probe as s356

    device = torch.device("cuda")
    guarded, ratios = _guarded(derived, cohort)
    rows = []
    for n_arg in arms:
        candidates = sorted(MODEL_ROOT.glob("model_n*_COND.pt"))
        if n_arg == -1:
            ckpt_path = max(candidates, key=lambda p: int(p.stem.split("_")[1][1:]))
        else:
            ckpt_path = MODEL_ROOT / f"model_n{n_arg}_COND.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = s356.build_model(ckpt["n"]).to(device)
        model.load_state_dict(ckpt["ema"])
        model.eval()
        banks = {s: dict(np.load(derived / "episodes" / f"{s}.npz")) for s in guarded}

        embeddings = {}
        for subject in guarded:
            d = banks[subject]
            sx = torch.from_numpy(d["sup_x"]).to(device)
            sy = torch.from_numpy(d["sup_y"]).to(device)
            emb = torch.zeros(s356.EMB_DIM, device=device, requires_grad=True)
            opt = torch.optim.Adam([emb], lr=s356.ORACLE_LR)
            for _ in range(s356.ORACLE_STEPS):
                e = emb[None].expand(len(sy), -1)
                loss = torch.mean((model(sy, e) - (sy - sx)) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            embeddings[subject] = emb.detach()

        def rrmse(subject, emb_vec, net=model):
            d = banks[subject]
            qx = torch.from_numpy(d["qry_x"]).to(device)
            qy = torch.from_numpy(d["qry_y"]).to(device)
            with torch.no_grad():
                xh = qy - net(qy, emb_vec[None].expand(len(qy), -1))
                num = torch.linalg.vector_norm(xh - qx, dim=(1, 2))
                den = torch.linalg.vector_norm(qx, dim=(1, 2)).clamp(1e-9)
            return float((num / den).mean())

        zero = torch.zeros(s356.EMB_DIM, device=device)
        for subject in guarded:
            wrongs = [rrmse(subject, embeddings[other])
                      for other in guarded if other != subject]
            rows.append({"n": int(ckpt["n"]), "subject": subject,
                         "checkpoint": ckpt_path.name,
                         "rrmse_zero": rrmse(subject, zero),
                         "rrmse_own": rrmse(subject, embeddings[subject]),
                         "rrmse_wrong_mean": float(np.mean(wrongs)),
                         "support_ratio": ratios[subject]["support"],
                         "query_ratio": ratios[subject]["query"]})
            print(json.dumps(rows[-1]), flush=True)

        if blind and n_arg == -1:
            blind_path = MODEL_ROOT / ckpt_path.name.replace("_COND", "_BLIND")
            if blind_path.is_file():
                bckpt = torch.load(blind_path, map_location=device,
                                   weights_only=False)
                bmodel = s356.build_model(bckpt["n"]).to(device)
                bmodel.load_state_dict(bckpt["ema"])
                bmodel.eval()
                for subject in guarded:
                    rows.append({"n": int(bckpt["n"]), "subject": subject,
                                 "checkpoint": blind_path.name, "arm": "BLIND",
                                 "rrmse_blind": rrmse(subject, zero, bmodel)})
    return rows


# ------------------------------------------------------------------ modes

def probe(limit: int) -> None:
    """Dev-class dry run on ext subjects. Never touches the sealed tree."""
    if SEALED_ROOT.stat().st_mode & 0o777:
        raise SystemExit("refusing to probe while the sealed tree is unsealed")
    cohort = _cohort(EXT_ROOT)[:limit]
    derived = DERIVED / "probe"
    out_dir = OUT_DIR / "probe"
    _prepare(EXT_ROOT, derived, out_dir, cohort)
    rows = _infer(derived, cohort, arms=(-1,), blind=True)
    guarded, ratios = _guarded(derived, cohort)
    cond = [r for r in rows if r.get("arm") != "BLIND"]
    payload = {
        "mode": "dev_class_probe", "sealed_contact": 0, "cohort": list(cohort),
        "guarded": guarded, "injection_ratios": ratios,
        "gain": _stat([r["rrmse_zero"] - r["rrmse_own"] for r in cond]) if cond else None,
        "own_minus_wrong": _stat([r["rrmse_wrong_mean"] - r["rrmse_own"]
                                  for r in cond]) if cond else None,
        "rows": rows,
        "gate_pass": bool(cond and len(guarded) >= max(3, len(cohort) - 2)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe.json").write_text(json.dumps(payload, indent=2,
                                                   sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in
                      ("mode", "sealed_contact", "guarded", "gate_pass")}))
    if not payload["gate_pass"]:
        raise SystemExit("PROBE GATE FAILED — do not open the sealed block")


def open_block(signoff: str, option: str) -> None:
    if OPENING_RECORD.is_file():
        raise SystemExit("an opening record already exists — the block opens once")
    if option != "A":
        raise SystemExit(f"option {option} has no protocol in {PREREG}")
    if not signoff.strip():
        raise SystemExit("--signoff is required (operator sign-off reference)")
    probe_path = OUT_DIR / "probe" / "probe.json"
    if not probe_path.is_file() or not json.loads(probe_path.read_text())["gate_pass"]:
        raise SystemExit("dev-class probe must pass before opening")
    record = {
        "preregistration": PREREG,
        "prereg_commit": _git_head(),
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "operator_signoff": signoff,
        "option": option,
        "freeze_record_sha256": _sha256(FREEZE_RECORD),
        "sealed_root": str(SEALED_ROOT),
        "mode_before": oct(SEALED_ROOT.stat().st_mode & 0o777),
    }
    OPENING_RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.chmod(SEALED_ROOT, 0o755)
    print(json.dumps({"opened": True, "record": str(OPENING_RECORD)}))


def run() -> None:
    if not OPENING_RECORD.is_file():
        raise SystemExit("no opening record — run `open` with operator sign-off first")
    banked = OUT_DIR / "sealed_rows.json"
    if banked.is_file():
        raise SystemExit("sealed rows already banked — this is a single pass")
    cohort = _cohort(SEALED_SUBJECTS_ROOT)
    _prepare(SEALED_SUBJECTS_ROOT, DERIVED, OUT_DIR, cohort)
    rows = _infer(DERIVED, cohort, arms=(-1, 30), blind=True)
    guarded, ratios = _guarded(DERIVED, cohort)
    payload = {"preregistration": PREREG, "opening_record": json.loads(
        OPENING_RECORD.read_text()), "cohort": list(cohort), "guarded": guarded,
        "injection_ratios": ratios, "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    banked.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    (OUT_DIR / "sealed_rows.sha256").write_text(_sha256(banked) + "\n")
    print(json.dumps({"banked": len(rows), "guarded": len(guarded),
                      "cohort": len(cohort)}))


def aggregate() -> None:
    banked = OUT_DIR / "sealed_rows.json"
    payload = json.loads(banked.read_text())
    if _sha256(banked) != (OUT_DIR / "sealed_rows.sha256").read_text().strip():
        raise SystemExit("banked-row digest mismatch — do not proceed")
    rows = payload["rows"]
    guarded = payload["guarded"]
    cond = [r for r in rows if r.get("arm") != "BLIND"]
    ns = sorted({r["n"] for r in cond})
    n_max, n_min = max(ns), min(ns)
    by_n = {n: [r for r in cond if r["n"] == n and r["subject"] in guarded]
            for n in ns}
    gain = {n: _stat([r["rrmse_zero"] - r["rrmse_own"] for r in by_n[n]]) for n in ns}
    own_minus_wrong = {n: _stat([r["rrmse_wrong_mean"] - r["rrmse_own"]
                                 for r in by_n[n]]) for n in ns}
    per_subject = {n: {r["subject"]: r["rrmse_zero"] - r["rrmse_own"]
                       for r in by_n[n]} for n in ns}
    trend = _stat([per_subject[n_max][s] - per_subject[n_min][s]
                   for s in guarded if s in per_subject[n_min]]) \
        if len(ns) > 1 else None

    g1 = bool(gain[n_max]["bootstrap_low"] > GATE_EPS)
    g2 = bool(own_minus_wrong[n_max]["bootstrap_low"] > 0)
    g3 = bool(trend and trend["bootstrap_low"] <= 0 <= trend["bootstrap_high"])
    exclusions = len(payload["cohort"]) - len(guarded)
    qc1 = bool(exclusions <= MAX_GUARD_EXCLUSIONS)
    blind_rows = [r for r in rows if r.get("arm") == "BLIND"]
    qc3 = None
    if blind_rows:
        blind_mean = float(np.mean([r["rrmse_blind"] for r in blind_rows
                                    if r["subject"] in guarded]))
        zero_mean = float(np.mean([r["rrmse_zero"] for r in by_n[n_max]]))
        qc3 = bool(abs(blind_mean - zero_mean) <= BLIND_SANITY_TOL)

    if not qc1:
        verdict = "INSTRUMENT_LIMITED"
    elif g1 and g2:
        verdict = "CONFIRMED"
    elif g1 and not g2:
        verdict = "PROTOCOL_GENERIC"
    elif g2 and not g1:
        verdict = "SPECIFIC_BUT_SMALL"
    else:
        verdict = "NOT_CONFIRMED"

    decision = {
        "preregistration": PREREG, "verdict": verdict,
        "gates": {"G1_gain_ci_low_gt_eps": g1, "G2_own_minus_wrong_ci_low_gt_0": g2,
                  "G3_flat_in_n": g3, "QC1_exclusions_ok": qc1,
                  "QC3_blind_sanity": qc3, "epsilon": GATE_EPS},
        "gain_by_n": {str(n): gain[n] for n in ns},
        "own_minus_wrong_by_n": {str(n): own_minus_wrong[n] for n in ns},
        "trend_nmax_minus_nmin": trend,
        "cohort_size": len(payload["cohort"]), "guarded_size": len(guarded),
        "guard_exclusions": exclusions,
        "development_reference_s356": {
            "gain_n259": [0.0608, 0.0406, 0.0835],
            "own_minus_wrong": [0.1623, 0.1314, 0.1946],
            "trend": [-0.0224, -0.0491, 0.0158]},
    }
    (OUT_DIR / "sealed_confirm_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "gain": gain[n_max]["mean"],
                      "gain_ci": [gain[n_max]["bootstrap_low"],
                                  gain[n_max]["bootstrap_high"]],
                      "own_minus_wrong": own_minus_wrong[n_max]["mean"],
                      "guarded": len(guarded)}, indent=1))


def reseal() -> None:
    os.chmod(SEALED_ROOT, 0o000)
    print(json.dumps({"resealed": True,
                      "mode": oct(SEALED_ROOT.stat().st_mode & 0o777)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "open", "run", "aggregate",
                                         "reseal"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--signoff", type=str, default="")
    parser.add_argument("--option", type=str, default="")
    args = parser.parse_args()
    if args.mode == "probe":
        probe(args.limit)
    elif args.mode == "open":
        open_block(args.signoff, args.option)
    elif args.mode == "run":
        run()
    elif args.mode == "aggregate":
        aggregate()
    else:
        reseal()


if __name__ == "__main__":
    main()
