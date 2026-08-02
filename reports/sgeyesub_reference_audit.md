# Native SGEYESUB reference audit

This is a lightweight, read-only review of the authors' repository and primary paper. Dataset presence and readability are recorded separately in `datasets/registry/sgeyesub.json`; this document does not replace the local structure audit.

## Frozen reference

- Repository: <https://github.com/rkobler/eyeartifactcorrection>
- Commit: [`2c95b4f46f37670d25399ac0fdd705ae18248b25`](https://github.com/rkobler/eyeartifactcorrection/commit/2c95b4f46f37670d25399ac0fdd705ae18248b25), current `master` head observed on 2026-08-01. This commit includes the latest blink-sign detection fix.
- Code license: LGPL version 3 or later, as stated by the repository `LICENSE` and source headers.
- Primary paper: Kobler et al., NeuroImage 218 (2020), 117000, <https://doi.org/10.1016/j.neuroimage.2020.117000>.
- Public-data anchor: <https://doi.org/10.17605/OSF.IO/2QGRD>. The paper says that preprocessed EEG is public and that raw EEG requires a formal data-sharing agreement. The separate OSF API/download audit recorded the dataset license as CC BY 4.0; it is not inferred from the LGPL code or paper license.

## Native entry points

The reference path is `demo_main.m` -> `algo = sgeyesub()` -> `algo.fit(X_trn, y_trn, eeg_chan_idxs)` -> `algo.apply(x)`. The implementation is `algorithms/sgeyesub.m`; `paradigm/eyeblock_paradigm.m` is the calibration-paradigm entry point.

`data` is channel by sample, `labels` is a sample-wise artifact class vector, and only rows selected by `eeg_chan_idxs` enter fitting and correction. Calibration and application must use the same channel order. At the frozen commit, `demo_main.m` constructs `eeg_chan_idxs` with `eeg_decodechan(EEGTRN.chanlocs, 'EEG', 'type')` (the older equivalent is `eeg_chantype`), so the release-internal native mapping is the exact ordered set of channels whose type is `EEG`. The official class encoding is right, left, up, down, blink, and rest as labels 1 through 6. The public release README additionally defines label 0 as unlabelled; it is ignored by class-conditioned native fitting and interval metrics rather than counted as artifact. These labels are obtained from support-period EOG-derived detections.

OSF `*_prep.set`/`*.fdt` pairs are already preprocessed. The official demo loads a preprocessed `.set` directly and splits it with `EEG.etc.trial_blocks`; it does not run `demo_preprocessing.m` again.

## Fixed rank and correction semantics

Native SGEYESUB does not search rank. It estimates one horizontal and one vertical direction, corrects those two directions, then estimates one blink direction from the residual. The registered rank is therefore fixed at at most three directions.

The implementation uses the published fixed regularization values `alpha = 1` and `beta = 0.01`. Its correction has the form

```text
C_eye = I - A_eye W_eye^T
C = (I - a_blink w_blink^T) C_eye
```

The mixing and unmixing vectors are not constrained to form an orthonormal basis. Consequently `C` is not registered as a symmetric, idempotent orthogonal projector. Native SGEYESUB is an EOG-labelled, participant-calibrated linear baseline, not an oracle and not a clean target. Oracle-span subtraction must use a separate method ID and implementation.

## Calibration and native block use

The standard eye block has about five minutes of data and 27 trials: 9 REST and 6 each of HORZ, VERT, and BLINK. Each trial has a 1-second preparation period, 10-second task period, and 2-3-second break. The reference preprocessing retains task seconds 1 through 9, omitting the first and last task seconds.

The paper's causal block use is:

| Paper dataset | Native support | Native query | Important qualification |
|---|---|---|---|
| EEGDS1 | block 1 | block 2 | The authors used EEGDS1 to select `alpha` and `beta`; treat it as development, not untouched confirmation. |
| EEGDS2 | block 1 | block 2 | Two-block participant recording. |
| EEGDS3 | block 1 | block 2 | Two-block participant recording. |
| EEGDS4 | blocks 1 and 2 | block 3 | Three-block participant recording; the paper excluded one recording for EOG cross-talk. |

The paper describes EEGDS1 as 15 participants plus five pilot measurements, EEGDS2 as 10 participants, EEGDS3 as 15 participants, and EEGDS4 as 15 participants with one excluded from the published comparison. These paper-level descriptions are not a substitute for checking the delivered OSF files.

## Local structure result and remaining unknowns

Job `919218` matched all 59 participant SET/FDT/block-metadata stems, read
`trial_blocks/ids/labels`, sampling and channel metadata, and never opened FDT
signals. Its concise interpretation is in `reports/sgeyesub_structure_audit.md`.
All delivered SET files expose blocks 1/2 only; `study05` lacks `trial_ids`.

The remaining unknowns are the exact `studyXX -> EEGDSX` mapping, the published
exclusion mapping, and the units/semantics of the structured `_block_dt.mat` payload.
That payload remains unidentified auxiliary metadata, not an EEG sample source,
label or fitting input. The OSF API/download audit separately recorded CC BY
4.0. The official `eeg_chan_idxs` rule is resolved. A source-faithful FP64
Python port is registered in `src/eeg_cgdr/baselines/native_sgeyesub.py`; it is
explicitly not claimed numerically equivalent to MATLAB until a cross-runtime
reference comparison is completed. No paper mapping should be guessed from
numbering or participant counts.

## Minimal split

1. Group the outer split by the verified participant stem; all blocks and companion files for one participant stay in one outer fold.
2. Keep EEGDS1 as development provenance for the published `alpha=1`, `beta=0.01` setting.
3. For a held-out EEGDS2/3 participant, use block 1 only as support and block 2 only as query. For EEGDS4 native replication, use blocks 1 and 2 as support and block 3 as query.
4. Do not randomly split epochs across support and query. Preserve `trial_ids` as context identifiers.
5. Support EOG and EOG-derived labels may be used for participant calibration. Query EOG/labels cannot affect fitting, gamma selection, method selection, or any method output. After every output is frozen, they may be opened only to score held-out metrics and make one pre-registered final automatic decision; that decision cannot adapt, reselect, or change a method. Native correction itself is the frozen matrix application.
6. Register drift variants, such as EEGDS4 block 1 support with blocks 2/3 as queries, under a different experiment ID because that is not the paper's native block rule.
