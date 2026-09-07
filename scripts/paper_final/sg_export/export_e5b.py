#!/usr/bin/env python3
"""E5b export — held-out paired exemplar (sub-04) with every method (GPU, small).

SERVER_EXPORT_FIGURE_ARRAYS_v4.md item E5b.  Rebuilds the T1 held-out bank EXACTLY as
scripts/paper_final/t1_heldout_uq.py does (its _fold99_context on the sealed root,
sampler seed 20269001, sample_balanced(8), the five seed-20261201 fold checkpoints,
K = 32 chains, the same paired noise seeds and the same operator-posterior draws) and,
for ALL 8 windows of sub-04 (the doc names windows 1, 4, 6 without an index
convention, so every window is exported with 0-based AND 1-based labels), adds:

  unguided           NO_A0 arm  (a0 = 0, gated signature)            same K, seeds, 5-model mean
  population         POP arm    (a0 = C0 @ drive, population signature) same K, seeds, 5-model mean
  linear_regression  y - C_gated @ drive (closed form; the deployed shrunk operator)
  ica                cpu_rows.ica fitter: mne fastica on the recipient cell's continuous
                     record (1-Hz high-pass copy), components removed by EOG correlation,
                     applied to the episode in physical units (y * eeg_scale) / eeg_scale
  asr                cpu_rows.asr fitter: asrpy calibrated on the cell's 120-s prefix
  eye_subspace       cpu_rows.sgeyesub: rank-2 projector of the raw 120-s ridge operator
  matched_mean       alias of the stored `mean` (subject-calibrated, K = 32, 5-model mean)

The matched arm is ALSO rebuilt (matched_mean_rebuilt) with the replayed operator draws
so the pipeline is checked against the stored `mean`.

Gates (assert; non-zero exit and nothing written on failure):
  G1  rebuilt contaminated / reference for sub-04's 8 windows are BIT-IDENTICAL to
      paper_final_arrays/t6_heldout_intervals_sub-04.npz
  G2  rebuilt eog_drive allclose (atol 1e-5) to the stored eog_drive (pinv is LAPACK-
      dependent, so bit-identity is reported but not required)
  G3  matched_mean_rebuilt reproduces the stored `mean` within REBUILD_TOL relative
      deviation on every window (GPU-arch float tolerance; a protocol mismatch would be
      far larger; the exact deviations are stored as rebuild_rel_dev)
Reference fitters that fail (e.g. the asrpy calibration reshape bug) give NaN and
coverage False for the affected windows, never an abort.

Output: results/paper_final/paper_final_arrays/t6_heldout_methods_sub-04.npz via
sg_common.save_npz (existing t6 keys kept verbatim, new keys added).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402
from pf_common import ARRAYS, SEED, load_model, per_channel_rrmse  # noqa: E402
import t1_heldout_uq as t1  # noqa: E402  (frozen protocol: _fold99_context + seeds)

SUBJECT = "sub-04"
STORED = ARRAYS / f"t6_heldout_intervals_{SUBJECT}.npz"
OUT_NAME = f"t6_heldout_methods_{SUBJECT}.npz"
STORED_KEYS = ("mean", "sigma", "var_op", "contaminated", "reference", "eog_drive", "zero_artifact")
ARM_NAMES = ("matched_mean", "unguided", "population", "linear_regression", "ica", "asr", "eye_subspace")
SAMPLED_ARMS = ("matched_rebuilt", "unguided", "population")
REBUILD_TOL = 0.05
EEG_TMP_NAMES = [f"E{i:02d}" for i in range(46)]        # cpu_rows' placeholder channel names


def load_stored() -> dict:
    with np.load(STORED, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# ----------------------------------------------------------------- bank rebuild (T1)
def rebuild_bank():
    from eeg_scad.cli.run_v44 import _bank_drives

    data, fold, registry30, eb120, assets, sampler = t1._fold99_context()
    bank = sampler.sample_balanced(8)
    drives = _bank_drives(assets, bank)
    keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]
    return data, fold, registry30, eb120, assets, bank, drives, keys


def replay_operator_draws(assets, post_var, keys, drives, indices) -> np.ndarray:
    """T1's per-chain operator-posterior draws, replayed over the WHOLE bank in bank
    order (the rng is consumed episode by episode), keeping the rows of `indices`.
    Returns a0 of shape (K, len(indices), 46, T) in float64 exactly as T1 built it."""
    pos = {i: p for p, i in enumerate(indices)}
    out = np.zeros((t1.K_CHAINS, len(indices)) + (46, drives.shape[-1]), np.float64)
    for chain in range(t1.K_CHAINS):
        rng = np.random.default_rng(910000 + t1.FOLD99 * 1000 + SEED % 100 + chain * 17)
        for i, (key, drive) in enumerate(zip(keys, drives)):
            operator = assets[key]["C_gated"] + rng.standard_normal(
                assets[key]["C_gated"].shape) * np.sqrt(post_var[key])
            if i in pos:
                out[chain, pos[i]] = operator @ drive
    return out


def sample_arms(models, schedule, device, sub_y, a0_matched, a0_pop, sig_gated, sig_pop):
    """K chains x 5-model mean per chain, the T1 seeds (subject_index of SUBJECT in
    SEALED), shared noise draw across the three arms within a chain."""
    from eeg_scad.cli.run_v44 import sample_bank_eog

    subject_index = t1.SEALED.index(SUBJECT)
    n = len(sub_y)
    zero = np.zeros_like(a0_pop)
    chains = {arm: np.zeros((t1.K_CHAINS, n, 46, sub_y.shape[-1]), np.float32) for arm in SAMPLED_ARMS}
    t0 = time.time()
    for chain in range(t1.K_CHAINS):
        seed = t1.PAIRED_SEED_BASE + subject_index + 31 * (chain + 1)
        inputs = {"matched_rebuilt": (a0_matched[chain], sig_gated),
                  "unguided": (zero, sig_gated),
                  "population": (a0_pop, sig_pop)}
        for arm in SAMPLED_ARMS:
            a0, sig = inputs[arm]
            ensemble = np.mean([sample_bank_eog(model, schedule, sub_y, a0, sig, device, seed)
                                for model in models], axis=0)
            if not np.isfinite(ensemble).all():
                raise FloatingPointError(f"nonfinite chain output ({arm}, chain {chain})")
            chains[arm][chain] = ensemble
        print(json.dumps({"chain": chain, "done": True, "t": round(time.time() - t0, 1)}), flush=True)
    return chains


# --------------------------------------------------------- reference fitters (cpu_rows)
def fit_eye_subspace(eb120, key):
    """cpu_rows.sgeyesub: rank-2 projector = column space of the raw 120-s ridge fit."""
    c_full = eb120.operator(*key, "RAW")
    q, _ = np.linalg.qr(c_full)
    projector = q @ q.T
    return lambda y: y - projector @ np.asarray(y, np.float64)


def fit_ica(registry30, key):
    """cpu_rows.ica: mne fastica on the continuous record (1-Hz high-pass copy for the
    fit / EOG selection), applied to the unfiltered data in physical units."""
    import mne
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog

    mne.set_log_level("ERROR")
    eeg, eye, names = registry30._load(*key)
    eog = bipolar_eog(eye, names)
    length = min(eeg.shape[1], eog.shape[1])
    stacked = np.vstack([eeg[:, :length], eog[:, :length]])
    info = mne.create_info(EEG_TMP_NAMES + ["VEOG", "HEOG"], float(sg.RATE),
                           ["eeg"] * 46 + ["eog", "eog"])
    raw = mne.io.RawArray(stacked, info, verbose="ERROR")
    raw_filt = raw.copy().filter(l_freq=1.0, h_freq=None, verbose="ERROR")
    decomposition = mne.preprocessing.ICA(n_components=0.999999, method="fastica",
                                          random_state=SEED, max_iter=1000)
    decomposition.fit(raw_filt, picks="eeg")
    bads, _ = decomposition.find_bads_eog(raw_filt)
    decomposition.exclude = sorted(set(bads))
    scale = registry30.eeg_scale[:, None]

    def apply(y):
        physical = np.vstack([np.asarray(y, np.float64) * scale, np.zeros((2, y.shape[-1]))])
        episode = mne.io.RawArray(physical, info, verbose="ERROR")
        return decomposition.apply(episode.copy()).get_data(picks="eeg") / scale

    return apply, len(decomposition.exclude)


def fit_asr(registry30, key):
    """cpu_rows.asr: asrpy calibrated per cell on the 120-s calibration prefix (scaled
    units), transform of the episode; the asrpy calibration reshape bug is data-dependent."""
    import asrpy
    import mne

    mne.set_log_level("ERROR")
    info = mne.create_info(EEG_TMP_NAMES, float(sg.RATE), ["eeg"] * 46)

    def to_raw(array):
        return mne.io.RawArray(np.asarray(array, np.float64), info, verbose="ERROR")

    scale = registry30.eeg_scale[:, None]
    eeg, _, _ = registry30._load(*key)
    calib = eeg[:, :sg.CALIB] / scale
    cleaner = asrpy.ASR(sfreq=float(sg.RATE))
    cleaner.fit(to_raw(calib))
    return lambda y: cleaner.transform(to_raw(y)).get_data()


def reference_methods(registry30, eb120, cell_keys, sub_y):
    n = len(cell_keys)
    shape = (n, 46, sub_y.shape[-1])
    out = {"ica": np.full(shape, np.nan), "asr": np.full(shape, np.nan), "eye_subspace": np.full(shape, np.nan)}
    cov = {k: np.zeros(n, bool) for k in out}
    n_excluded = np.full(n, -1, np.int64)
    errors = {}
    fitters = {"eye_subspace": fit_eye_subspace, "ica": fit_ica, "asr": fit_asr}
    for key in sorted(set(cell_keys)):
        rows = [i for i, k in enumerate(cell_keys) if k == key]
        for name, fit in fitters.items():
            t0 = time.time()
            try:
                if name == "eye_subspace":
                    apply = fit(eb120, key)
                    excluded = None
                elif name == "ica":
                    apply, excluded = fit(registry30, key)
                else:
                    apply = fit(registry30, key)
                    excluded = None
                for i in rows:
                    est = np.asarray(apply(np.asarray(sub_y[i], np.float64)), np.float64)
                    assert est.shape == (46, sub_y.shape[-1]) and np.isfinite(est).all(), name
                    out[name][i] = est
                    cov[name][i] = True
                    if excluded is not None:
                        n_excluded[i] = excluded
                print(json.dumps({"cell": "|".join(key), "method": name, "windows": rows,
                                  "excluded": excluded, "t": round(time.time() - t0, 1)}), flush=True)
            except Exception as error:  # NaN + coverage False, never an abort
                errors[f"{name}|{'|'.join(key)}"] = f"{type(error).__name__}: {str(error)[:200]}"
                print(json.dumps({"cell": "|".join(key), "method": name, "failed": errors[f"{name}|{'|'.join(key)}"]}),
                      flush=True)
                traceback.print_exc()
    return out, cov, n_excluded, errors


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpu", action="store_true",
                        help="debug only: no sampling (unguided/population/rebuild NaN); never for the export")
    args = parser.parse_args()

    stored = load_stored()
    assert all(k in stored for k in STORED_KEYS), sorted(stored)
    t0 = time.time()
    data, fold, registry30, eb120, assets, bank, drives, keys = rebuild_bank()
    print(json.dumps({"bank": int(len(keys)), "t": round(time.time() - t0, 1)}), flush=True)
    indices = [i for i, m in enumerate(bank["meta"]) if m["participant"] == SUBJECT]
    assert len(indices) == stored["contaminated"].shape[0] == 8, (indices, stored["contaminated"].shape)
    cell_keys = [keys[i] for i in indices]
    metas = [bank["meta"][i] for i in indices]
    sub_y = np.stack([bank["y"][i] for i in indices])                      # float32, as T1
    sub_x = np.stack([bank["x"][i] for i in indices])
    sub_drive = np.stack([drives[i] for i in indices])                     # float64, as T1
    zero_artifact = np.asarray([m["zero_artifact"] for m in metas])

    # G1: bit-identical contaminated / reference
    for name, mine in (("contaminated", sub_y), ("reference", sub_x)):
        mine32 = mine.astype(np.float32)
        same = (mine32.shape == stored[name].shape and mine32.dtype == stored[name].dtype
                and mine32.tobytes() == stored[name].tobytes())
        print(json.dumps({"G1": name, "bit_identical": bool(same),
                          "max_abs_diff": float(np.abs(mine32.astype(np.float64) - stored[name]).max())}), flush=True)
        assert same, f"G1 FAILED: rebuilt {name} is not bit-identical to {STORED}"
    assert np.array_equal(zero_artifact, stored["zero_artifact"]), (zero_artifact, stored["zero_artifact"])
    # G2: eog_drive
    drive32 = sub_drive.astype(np.float32)
    drive_bit = drive32.tobytes() == stored["eog_drive"].tobytes()
    drive_diff = float(np.abs(drive32.astype(np.float64) - stored["eog_drive"]).max())
    print(json.dumps({"G2": "eog_drive", "bit_identical": bool(drive_bit), "max_abs_diff": drive_diff}), flush=True)
    assert np.allclose(drive32, stored["eog_drive"], atol=1e-5, rtol=1e-5), "G2 FAILED: eog_drive differs"

    # closed-form arms
    sub_y64 = sub_y.astype(np.float64)
    linear = np.stack([sub_y64[p] - assets[k]["C_gated"] @ sub_drive[p] for p, k in enumerate(cell_keys)])
    sig_gated = np.stack([assets[k]["sig_gated"] for k in cell_keys])
    sig_pop = np.stack([assets[k]["sig_pop"] for k in cell_keys])
    a0_pop = np.stack([assets[k]["C0"] @ sub_drive[p] for p, k in enumerate(cell_keys)])

    # reference methods (CPU) — per-cell fitters mirrored from cpu_rows
    ref_out, ref_cov, n_excluded, ref_errors = reference_methods(registry30, eb120, cell_keys, sub_y)

    # GPU arms
    n, T = len(indices), sub_y.shape[-1]
    sampled = {arm: np.full((n, 46, T), np.nan, np.float32) for arm in SAMPLED_ARMS}
    sampled_sigma = {arm: np.full((n, 46, T), np.nan, np.float32) for arm in SAMPLED_ARMS}
    rebuild_dev = np.full(n, np.nan)
    if not args.skip_gpu:
        import torch
        from eeg_scad.cli.run_v44_s2 import _posterior_variance
        from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

        post_var = _posterior_variance(registry30, eb120, fold)
        a0_matched = replay_operator_draws(assets, post_var, keys, drives, indices)
        device = torch.device("cuda")
        schedule = LinearX0Schedule().to(device)
        models = [load_model(fold_id, device, SEED) for fold_id in range(5)]
        print(json.dumps({"gpu": torch.cuda.get_device_name(0), "models": 5, "K": t1.K_CHAINS,
                          "t": round(time.time() - t0, 1)}), flush=True)
        chains = sample_arms(models, schedule, device, sub_y, a0_matched, a0_pop, sig_gated, sig_pop)
        for arm in SAMPLED_ARMS:
            sampled[arm] = chains[arm].mean(axis=0)
            sampled_sigma[arm] = chains[arm].std(axis=0, ddof=1).clip(1e-9)
        # G3: the rebuilt matched arm reproduces the stored mean
        for p in range(n):
            num = np.linalg.norm(sampled["matched_rebuilt"][p].astype(np.float64) - stored["mean"][p])
            rebuild_dev[p] = num / max(np.linalg.norm(stored["mean"][p].astype(np.float64)), 1e-12)
        sigma_dev = float(np.abs(sampled_sigma["matched_rebuilt"].astype(np.float64) - stored["sigma"]).max())
        print(json.dumps({"G3_rebuild_rel_dev": [round(float(v), 6) for v in rebuild_dev],
                          "sigma_max_abs_diff": sigma_dev}), flush=True)
        assert np.all(rebuild_dev < REBUILD_TOL), f"G3 FAILED: rebuilt matched mean deviates {rebuild_dev}"

    # assemble arms in ARM_NAMES order
    arms = {"matched_mean": stored["mean"].astype(np.float32),
            "unguided": sampled["unguided"], "population": sampled["population"],
            "linear_regression": linear.astype(np.float32),
            "ica": ref_out["ica"].astype(np.float32), "asr": ref_out["asr"].astype(np.float32),
            "eye_subspace": ref_out["eye_subspace"].astype(np.float32)}
    coverage = np.stack([np.isfinite(arms[a]).all(axis=(1, 2)) for a in ARM_NAMES])
    for name in ("ica", "asr", "eye_subspace"):
        assert np.array_equal(coverage[ARM_NAMES.index(name)], ref_cov[name]), name
    ref64 = stored["reference"].astype(np.float64)
    rrmse_all = np.full((len(ARM_NAMES), n), np.nan)
    rrmse_ch = np.full((len(ARM_NAMES), n, 46), np.nan)
    for a, name in enumerate(ARM_NAMES):
        for p in range(n):
            if coverage[a, p]:
                est = arms[name][p].astype(np.float64)
                rrmse_all[a, p] = np.linalg.norm(est - ref64[p]) / max(np.linalg.norm(ref64[p]), 1e-12)
                rrmse_ch[a, p] = per_channel_rrmse(ref64[p], est)
    contaminated_rrmse = np.asarray([np.linalg.norm(sub_y64[p] - ref64[p]) / max(np.linalg.norm(ref64[p]), 1e-12)
                                     for p in range(n)])
    print(json.dumps({"coverage": {a: int(c.sum()) for a, c in zip(ARM_NAMES, coverage)},
                      "rrmse_all_channels_mean_over_covered": {a: float(np.nanmean(rrmse_all[i]))
                                                               for i, a in enumerate(ARM_NAMES)},
                      "contaminated_rrmse": [round(float(v), 4) for v in contaminated_rrmse]}, indent=1), flush=True)

    window0 = np.arange(n, dtype=np.int64)
    sessions = np.asarray([k[1] for k in cell_keys])
    sg.save_npz(
        OUT_NAME,
        **{k: stored[k] for k in STORED_KEYS},                         # existing keys, verbatim
        **arms,
        unguided_sigma=sampled_sigma["unguided"], population_sigma=sampled_sigma["population"],
        matched_mean_rebuilt=sampled["matched_rebuilt"], matched_sigma_rebuilt=sampled_sigma["matched_rebuilt"],
        rebuild_rel_dev=rebuild_dev, rebuild_tolerance=np.asarray(REBUILD_TOL),
        window_index_0based=window0, window_index_1based=window0 + 1,
        coverage=coverage, arm_names=np.asarray(ARM_NAMES),
        rrmse_all_channels=rrmse_all, rrmse_per_channel=rrmse_ch, contaminated_rrmse=contaminated_rrmse,
        subject=np.asarray(SUBJECT),
        participant=np.asarray([k[0] for k in cell_keys]), session=sessions,
        task=np.asarray([k[2] for k in cell_keys]),
        condition=np.asarray([sg.SESSION_CONDITION.get(s, "unknown") for s in sessions]),
        clean_owner=np.asarray([m["clean_owner"] for m in metas]),
        eog_owner=np.asarray([m["eog_owner"] for m in metas]),
        gain=np.asarray([float(m["gain"]) for m in metas]),
        ica_n_excluded=n_excluded,
        reference_errors=np.asarray(json.dumps(ref_errors)),
        eog_drive_bit_identical=np.asarray(bool(drive_bit)),
        stored_source=np.asarray(str(STORED)),
        protocol=np.asarray(
            "T1 held-out bank rebuilt via t1_heldout_uq._fold99_context (fold 99 on the sealed "
            f"root, sampler seed {t1.SAMPLER_SEED}, sample_balanced(8)); five seed-{SEED} fold "
            f"checkpoints; K={t1.K_CHAINS} chains, noise seed PAIRED_SEED_BASE + subject_index + "
            "31*(chain+1) shared across arms within a chain, each chain = mean of the 5 model "
            "outputs, arm output = mean over chains (sigma = std ddof=1); matched arm replays "
            "T1's entrywise operator-posterior draws; population arm uses the fixed C0 (no "
            "posterior draw exists for it); unguided arm a0 = 0 with the gated signature; "
            "linear = y - C_gated @ drive; ica / asr / eye_subspace = cpu_rows fitters on the "
            "recipient cell's record (ICA on the continuous record as cpu_rows does, ASR on the "
            "120-s prefix, eye subspace from the raw 120-s ridge operator)"),
        units=np.asarray("fold-scaled EEG (channel / fold-train eeg_scale); eog_drive in the "
                         "recipient cell's latent [V, H] coordinates; all arms are clean-EEG "
                         "estimates (not artifact estimates)"),
    )
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
