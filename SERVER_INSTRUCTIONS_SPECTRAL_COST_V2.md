# Server instructions: spectral cost without concatenation, plus two small gap fills

Repo: github.com/W-Yinghao/EEG_denoise, branch `codex/paper-final-runs` (work on top of 7c7f3f7). Same light harness as before: Slurm, no verification ceremonies, one commit per item with the manifest, report the numbers in the commit message. Nothing here changes a claim that is already in the manuscript except the band-power block of the natural-window table, which is being replaced.

Order: A (CPU, needed for the paper) → C (CPU, cheap) → B (GPU, optional).

---

## A. Replace `natural_band_power.npz` and `natural_psd_ratio.npz` (E6b / E3b)

### Why

`export_natural_post.py` takes the low-EOG samples of each 5.12-s window (bottom 30 % EOG energy, `n_low = 154`), which form about 12 non-contiguous runs, concatenates them, and runs one Hann Welch segment (`nperseg 154`). Every join is a step discontinuity. The leakage it produces is broadband and differs between the corrected and the raw signal, so it does not cancel in the ratio, and it is largest at low frequency.

Measured on the three exemplar windows of `natural_exemplars.npz` (which carry full waveforms), concatenated Welch minus a band-pass-then-mask estimator, nine-electrode mean:

| condition | delta | theta | alpha |
|---|---|---|---|
| subject calibrated | +1.31 dB | +0.55 | +0.11 |
| calibrated linear | +1.06 | +0.34 | 0.00 |
| ICA | +1.11 | −0.31 | −0.43 |
| ASR | +2.07 | +1.91 | +1.45 |
| eye-subspace | +1.91 | +0.45 | +0.42 |
| unguided | 0.00 | 0.00 | 0.00 |

The concatenated estimator pulls every cost toward 0 dB, and in single windows it flips the sign (subject calibrated, fast-walking window: +0.96 dB concatenated against −1.11 dB band-passed). The unguided arm, whose output is nearly its input, shows no offset, so the offset is the estimator, not the data. The banked E3/E6 arrays are therefore not usable; do not delete them, but do not extend them either.

### Estimator

Per window, per condition, per channel, on fold-scaled EEG exactly as stored by the natural pass:

```python
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 100.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "broad": (0.5, 15.0)}
SOS = {b: butter(4, [lo, hi], btype="band", fs=FS, output="sos") for b, (lo, hi) in BANDS.items()}
BANK = {f"{k}Hz": butter(2, [k - 0.5, k + 0.5], btype="band", fs=FS, output="sos") for k in range(1, 15)}

def band_power_on_mask(x, mask, sos_dict):
    """x: (46, 512) one window, contiguous; mask: (512,) bool = the L (or H) samples.
    Filter the whole contiguous window, then take the mean square over the masked samples only."""
    return {b: (sosfiltfilt(s, x, axis=1)[:, mask] ** 2).mean(1) for b, s in sos_dict.items()}
```

`mask` is the same L mask as before (`sg_common.eog_masks` on the latent drive, bottom 30 %; keep `n_low = 154` and assert it). For each window compute `P_cond[b]` and `P_raw[b]` with the same mask, then

```
ratio_db_window[cond, w, b, ch] = 10 * log10(P_cond / P_raw)
ratio_db[cond, rec, b, ch]      = mean over the 4 windows of the recording (mean of dB)
```

and the participant-first pooling as in the banked files. No concatenation anywhere; the filter sees the full 512-sample window (zero-phase, default `sosfiltfilt` padding).

Compute the same quantities on the H mask (top 30 % EOG energy). This is a sanity check only: on H every ocular-removal method must be strongly negative in delta and the unguided arm near zero.

### Outputs

`results/paper_final/paper_final_arrays/natural_band_power_v2.npz` (+ `.manifest.json`)

- `ratio_db` (10, 90, 4, 12), `ratio_db_window` (10, 360, 4, 12), `power` (10, 90, 4, 12), on L
- `ratio_db_high`, `ratio_db_window_high`, `power_high`, same shapes, on H
- `bands` = ["delta", "theta", "alpha", "broad"], `band_edges_hz` (4, 2)
- `electrodes`, `electrode_index`, `coverage`, `coverage_window`, `n_windows_covered`, `window_recording`, `window_start`, `condition`, `condition_label`, all `recording_*` and `participants` keys exactly as in the banked `natural_band_power.npz`
- `estimator` = "fourth-order zero-phase Butterworth band-pass on the contiguous 512-sample window (scipy sosfiltfilt), mean square over the masked samples; no concatenation", `mask_note` (L and H definitions, n_low = n_high = 154), `seed`, `k_trajectories`, `source`, `arm_note`, `condition_note`, `eeg_names`

`results/paper_final/paper_final_arrays/natural_psd_ratio_v2.npz` (+ manifest)

- `ratio_db` (10, 90, 14, 12) and `ratio_db_window` (10, 360, 14, 12) on L, `ratio_db_high` on H, from the 1-Hz bank (second-order Butterworth, centres 1..14 Hz, 1 Hz wide); `centres_hz` (14,), `bandwidth_hz` = 1.0; same bookkeeping keys as above.

Same 10 conditions and order as the banked files: RAW, MATCH_gated, POP, NO_A0, LINEAR, WRONG_gated, SHUFFLED, ICA, ASR, SGEYESUB. Read the stored natural-pass outputs under `/projects/EEG-foundation-model/derived/denoiseNet/sg_natural_pass`; do not rerun the diffusion arms.

### Gate (one line each, in the commit message)

- `n_low == 154` and `n_high == 154` in every window.
- Unguided arm, L, nine-electrode participant-first mean: |delta, theta, alpha| all below 0.05 dB.
- H-mask delta for subject calibrated, ICA, and eye-subspace all below −3 dB (blink power removed where the blinks are).
- The 10 × 3 nine-electrode participant-first means on L, printed next to the banked concatenated values, so the two can be compared in the manuscript report.

No figure is needed; the paper reports these as a table.

---

## C. Natural endpoints for every arm from the same pass (CPU, cheap)

The natural-endpoint table of the manuscript has no row for calibrated linear regression because the stored T5 rows cover only the diffusion conditions and the three reference methods. The natural pass that produced E2/E3/E6 already holds the LINEAR outputs on the same 360 windows. Compute attenuation (dB), low-EOG retention, and coupling reduction with `run_v44._natural_metrics` for all 10 arms from those outputs and bank

`results/paper_final/paper_final_arrays/natural_endpoints_all_arms.npz` (+ manifest): `attenuation_db`, `retention`, `coherence_reduction` each (10, 360), plus `coverage` and the bookkeeping keys above.

Gate: the five diffusion arms must reproduce the stored T5/g3 participant-first means to 1e-6 (subject calibrated 2.463 dB / 0.843 / 0.190, population 0.909 / 0.773 / 0.105, unguided 0.277 / 0.921 / 0.021, mismatched rule applied 0.570 / 0.780 / 0.084, shuffled −0.384 / 0.365 / 0.083), and ICA / ASR / eye-subspace must reproduce `cpu_reference_rows.json` (2.983 / 0.704 / 0.219, 2.812 / 0.623 / 0.119, 4.869 / 0.495 / 0.213). Report the LINEAR row.

---

## B. Deterministic twin on the third seed (GPU, optional, do last)

Table 3 of the manuscript compares the one-step deterministic network with subject-calibrated diffusion. The twin exists for seeds 20261201 and 20261202 only (`g4_estimators.json`, arm `det_twin_matched`, 240 rows), while the diffusion model has three seeds, and the third seed's window set is much easier (subject-calibrated 0.222 against 0.633 and 0.438). The manuscript currently compares on the two shared seeds; the clean version compares on all three.

Train the deterministic twin for seed 20261203 on the five folds with exactly the stage-1 recipe of `det_fold_*_seed_*` (same inputs, same schedule), evaluate it on seed 20261203's paired window set (the one the diffusion model of that seed was scored on), and append the 120 rows to `g4_estimators.json` under `det_twin_matched` with `seed = 20261203`. Do the same for `det_pop_retrained` if the population-trained deterministic recipe is equally cheap; otherwise skip it.

Report: participant-first RRMSE of the twin on the third seed, and the three-seed twin against three-seed diffusion (paired within participant). Rough cost: five short deterministic trainings on one A100.

---

## Commit and hand-back

One commit per item on `codex/paper-final-runs`: "E6b/E3b banked: band-pass-then-mask spectral cost (natural_band_power_v2, natural_psd_ratio_v2)", "natural endpoints for all ten arms banked", "deterministic twin seed 20261203 banked". Put the gate lines and the summary numbers in the commit messages. Push; no need to render figures or to touch the manuscript.
