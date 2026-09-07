# Server export request: arrays for the SGEYESUB-style figures (v4.1 → v4.2)

Repo: github.com/W-Yinghao/EEG_denoise, branch of the paper_final runs. Everything below is CPU-only and reads stored outputs or refits the 46×2 ridge matrices; no training, no new GPU sampling. Write each item as one `.npz` under `results/paper_final/paper_final_arrays/` and copy the folder back into the local snapshot (`output/results_review_20260905/repo_snapshot/paper_final_arrays/`). Keep the existing key names where a file already exists.

The purpose is presentation, not new claims: these arrays feed figures that readers of Kobler et al. 2020 (SGEYESUB) expect and that the current v4.1 manuscript lacks. Nothing here changes a number in the text.

## E1. Propagation matrices (`e1_operators.npz`)

Per development recording (15 participants × 6 recordings = 90): the shrunk propagation matrix `A_tilde` (46×2), the raw ridge estimate `A_hat` (46×2), the shrinkage weight `lambda`, the reliability-rule verdict, the population matrix of that condition with the participant excluded (46×2), and the later coupling matrix re-estimated from the four natural windows (46×2, the E3 layer-2 object). Keys: `participant`, `session`, `task`, `condition`, `eeg_names` (46), `A_tilde`, `A_hat`, `A_pop`, `A_later`, `lambda`, `accepted`.

Figure it feeds: topographies of the vertical and horizontal columns, rows = own calibration / population / later coupling, on the 32 scalp electrodes, shared colour scale per column. This is the calibrated eye-artifact pattern that SGEYESUB shows as its uncorrected-residual maps; ours is the object the method actually subtracts.

## E2. Residual EOG correlation per channel on natural windows (`natural_residual_corr.npz`)

For each development recording and each of the 360 natural windows, for the conditions raw / subject calibrated / population calibrated / unguided / mismatched (rule applied) / temporally shuffled and the three reference methods (ICA, ASR, eye-subspace projection): the absolute Pearson correlation between each corrected EEG channel and the vertical EOG, and separately the horizontal EOG, computed over the samples of the window. Shape: `(condition, window, 46, 2)`. Also store the same quantity restricted to the high-EOG samples H of Section 4.2 (`corr_high`).

Figure: rows = conditions (raw first), columns = vertical / horizontal, topographies of the participant-averaged |r|, shared colour bar 0–1, plus a channel-averaged bar with 95% CI. This is SGEYESUB Fig. 2a/e for our conditions.

## E3. Spectral cost on low-EOG samples (`natural_psd_ratio.npz`)

For each development recording and each condition above: Welch PSD (2-s window, 1-s overlap, 0.5–15 Hz at 100 Hz) of the corrected and the uncorrected EEG over the low-EOG samples L of each natural window, and the ratio corrected/uncorrected in dB, at the nine electrodes F3 Fz F4 / C3 Cz C4 / P3 Pz P4 (use AFz and Fp1/Fp2 as a fourth row if convenient). Shape: `(condition, recording, electrode, freq)`, plus `freqs`.

Figure: 3×3 grid, PSD ratio in dB against frequency, mean and 95% CI across recordings, 0 dB reference line. SGEYESUB Fig. 3. It turns our scalar retention into a per-frequency cost curve and shows where the eye-subspace projection and ASR pay their retention deficit.

## E4. ERP grand averages and peak topographies (`erp_topographies.npz`)

From the D-wave ERP pipeline, for both cohorts and for raw / subject calibrated / population calibrated / unguided / calibrated linear regression / ICA / ASR / eye-subspace projection: the grand-average target and non-target ERP over all 46 channels on −0.2 to 0.8 s (shape `(condition, class, 46, time)`), the per-participant target averages (`(condition, participant, 46, time)`), and the latency of the P300 peak at Pz on raw. Reuse the epochs already cut for Table 8, same folds and exclusions.

Figure: (a) target-minus-non-target time course at Pz and Fz for each condition, raw in black; (b) topographies at the raw P300 peak latency for raw and each condition on one colour scale. SGEYESUB Fig. 2c and EEGDfus Fig. 8.

## E5. One natural window per movement condition (`natural_exemplars.npz`)

For one development participant with an accepted calibration (prefer the participant used for the current Fig. 1/2 exemplars if it has clean natural windows; otherwise sub-02 or sub-05): for each of standing / walking / running under the ERP task, one 5.12-s natural window with a visible blink and a horizontal movement, storing the raw EEG (46×512), the two EOG channels, and the corrected EEG under subject calibrated / population calibrated / unguided / calibrated linear regression / ICA / ASR / eye-subspace projection. Keys as in `t6_waveform_exemplar_dev.npz` plus `condition`.

Figure: full channel stack (or the 12 frontal-to-parietal midline and lateral channels) raw in black with subject-calibrated in red, EOG on top, scale bar, one column per movement condition. SGEYESUB Fig. 1d and EEGDfus Fig. 7; it is the only real-data before/after strip the paper would have.

## E5b. Paired exemplar with every method (`t6_heldout_methods_sub-04.npz`)

For the held-out participant sub-04, windows 1, 4 and 6 of `t6_heldout_intervals_sub-04.npz` (the three windows shown in the current Fig. 2A): the outputs of calibrated linear regression, the unguided network, population calibration, and the three reference methods on the same contaminated input, all 46 channels. The stored development exemplar `t6_waveform_exemplar_dev.npz` (sub-02, ses-02, episode 1) is unusable for this: its reference has almost no energy, so every method sits at an all-channel RRMSE of about 15.5.

Figure: stacked traces on Fp1 and AFz, contaminated / linear / unguided / population / subject calibrated with the reference overlaid, the EEGDfus Fig. 7 layout, as a row of Fig. 2 or an appendix figure.

## E6. Band power ratios (`natural_band_power.npz`, optional)

For each recording, condition, and band (delta 0.5–4, theta 4–8, alpha 8–13 Hz): power after correction over power before, on the low-EOG samples, at the nine electrodes of E3. Feeds one small table in the style of EEGdenoiseNet Tables 2–3.

## Priority and cost

E1 and E5 first (matrices are already stored; E5 is inference on stored checkpoints for seven conditions on three windows). E2 and E3 are pure post-processing of the natural-window outputs that T5 already produced; if those outputs were not kept, rerun the natural pass for the six diffusion conditions with the fold models (about 360 windows × 6 conditions, K = 1 trajectory is enough for these figures). E4 reuses the D-wave epochs. E6 last. No new gates, no decision JSON: report the file list and shapes.
