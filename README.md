# Subject-Calibrated EOG-Guided Diffusion for EEG Ocular Artifact Removal

This branch freezes the minimal TAAS submission built around the V11.1 method:
an early same-session EEG+EOG calibration estimates a participant-specific
EOG-to-EEG contamination operator, which conditions a fixed residual diffusion
denoiser applied to later EEG+EOG query windows. The deployment assumption is
therefore **EOG-guided**, not EEG-only.

On the nine BCI Competition IV-2b development participants, three training
seeds give participant-averaged paired-RRMSE effects of
`U_P = +0.03473` (DIFF-POP minus DIFF-MATCH) and `U_W = +0.04836`
(DIFF-WRONG minus DIFF-MATCH), with all 9/9 participants in the same direction.
This is three-seed stability on one development cohort, not independent
replication or confirmation. Mean natural-signal proxy criteria were met, with
one preservation and two covariance participant-level exceptions.

The comparison boundary is explicit: DIFF-MATCH RRMSE is `0.11350`, versus
`0.11225` for DET-MATCH and `0.08760` for LINEAR-MATCH. The evidence supports
subject conditioning within the fixed diffusion pipeline; it does **not** show
that diffusion beats the deterministic or linear estimators.

## Frozen evidence and manuscript

- Manuscript source: [`taas_submission/main.tex`](taas_submission/main.tex)
- Built manuscript: [`taas_submission/main.pdf`](taas_submission/main.pdf)
- Freeze report: [`reports/taas_subject_diffusion_freeze.md`](reports/taas_subject_diffusion_freeze.md)
- Scientific source tables: [`results/cgdr/bci2b_subject_diffusion_replication/`](results/cgdr/bci2b_subject_diffusion_replication/)
- Evidence manifest: [`results/cgdr/taas_subject_diffusion_freeze/evidence_manifest.csv`](results/cgdr/taas_subject_diffusion_freeze/evidence_manifest.csv)

All scientific execution uses Slurm. CPU aggregation, figures, tests, and LaTeX
compilation use `/home/infres/yinwang/anaconda3/envs/eeg2025`; GPU training and
inference use `/home/infres/yinwang/anaconda3/envs/icml`. Raw BCI2b data remain
under `/projects/EEG-foundation-model` and are never committed.

## Scope and limitations

- BCI2b, nine participants, same-session early-support to later-query protocol.
- EOG is visible in both calibration support and query inference.
- POP uses the frozen two-heldout design: recipient and cyclic wrong donor are
  excluded, leaving seven participants for population training. It is not a
  standard eight-subject LOSO population baseline.
- The frozen cyclic unseen-WRONG is the primary specificity control. Each fold
  contains only one such unseen donor; other seen donors are sensitivity tests.
- Participant is the scientific unit (`n=9`); windows and three training seeds
  are not independent samples.
- Historical SADDPM, CGDR, operator-bridge, and clean-posterior routes are
  archived development history and are not the method described by the paper.

## Repository map

```
src/eeg_cgdr/models/eog_residual_diffusion.py   frozen V11.1 model
src/eeg_cgdr/experiments/                       protocols/evaluation
configs/cgdr/                                   frozen experiment configs
scripts/slurm/                                  Slurm entry points
tests/                                          targeted scientific-contract tests
taas_submission/                                ACM TAAS manuscript
reports/                                        scoped reports and job maps
```

No checkpoints, prepared arrays, per-window outputs, raw data, or Slurm logs
belong in Git.
