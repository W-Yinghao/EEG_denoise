# TAAS subject-diffusion freeze

## Frozen scientific conclusion

`MATCHED_SUBJECT_OPERATOR_EFFECT_IS_THREE_SEED_STABLE_WITHIN_THE_FIXED_EOG_GUIDED_DIFFUSION_PIPELINE`

The submission claim is restricted to: early same-session subject calibration
improves a fixed EOG-guided diffusion EEG denoiser on held-out later EEG
relative to population and mismatched-subject conditioning.

This is **three-seed stability on the same development cohort**, not an
independent-cohort evaluation or confirmation. Query EOG is a deployment input.
The nine participants are the only scientific samples; 27 participant--seed
comparisons are not `n=27`.

## Frozen numbers and statistical wording

- DIFF-MATCH RRMSE: `0.11350`.
- DET-MATCH RRMSE: `0.11225`.
- LINEAR-MATCH RRMSE: `0.08760`.
- `U_P = +0.03473`, positive for 9/9 participants.
- `U_W = +0.04836`, positive for 9/9 participants.
- One-sided exact sign-flip `p = 0.001953`.
- Two-sided sign-flip sensitivity `p = 0.003906`.
- Participant bootstrap intervals are descriptive.
- Diffusion-over-deterministic and diffusion-over-linear are **not supported**.

The interaction is

`I_P = [R(DET-POP)-R(DET-MATCH)] - [R(DIFF-POP)-R(DIFF-MATCH)]`.

A positive value means deterministic subject benefit is larger; it is not
diffusion-specific synergy.

## Natural-signal proxy scope

The permitted wording is: “Mean natural-signal proxy criteria were met, with
disclosed participant-level exceptions.” Mean EOG attenuation is `+0.1043` and
mean preservation is `0.7982`. Exceptions are preservation 1/9, covariance
2/9, nonpositive EOG attenuation 0/9, and MI-kappa threshold 0/9.

## Population and donor controls

The frozen POP checkpoint uses two-heldout folds: recipient and cyclic wrong
donor are excluded, leaving seven population-training participants. It is not a
standard eight-subject LOSO population checkpoint. The frozen cyclic
unseen-WRONG is the primary specificity control, with one genuinely unseen
donor per fold. Compatible training-seen donors are a separate donor
sensitivity; they are not multi-unseen-donor evidence.

## Manuscript reset

The title, abstract, introduction, method, experiments, figures, conclusion,
appendix, and README now describe the final V11.1 EOG-guided full-waveform
residual estimator. Historical SADDPM, CGDR, artifact-subspace, population
posterior, and V12 clean-posterior narratives are excluded from the method and
main results. Klados and SGE results from incompatible estimators are not mixed
into the primary table.

Submission figures use a frozen participant/window rule where applicable:
participant 1, first same-session unit, first paired window, first EEG channel,
seed 20260808. This selection is index-based rather than outcome-based.

## Remaining metadata-only submission blockers

The ACM author names, ORCIDs, affiliations, correspondence email, received
dates, and eRights DOI/article metadata remain explicit placeholders because
they are not present in the repository. They require author/ACM input but do not
alter scientific content. No values were invented.

## Reproducibility

The scientific source CSV/JSON files under
`results/cgdr/bci2b_subject_diffusion_replication/` are unchanged. The freeze
script verifies the frozen numbers before building the evidence manifest and
figures. Checkpoints, prepared arrays, per-window outputs, raw data, and Slurm
logs remain uncommitted.

Targeted tests passed 9/9 in Slurm job `931228`. The ACM TAAS source compiled
with `pdflatex`/BibTeX to a 10-page PDF in Slurm job `931316`; the log contains
no unresolved citations or references in its final pass (the expected first
passes report references before BibTeX has populated them). The local cluster
TeX installation lacked the ACM class and several runtime packages, so the
compile job obtains the official CTAN `acmart` archive and a local user-mode
TeX tree. These local TeX dependencies are ignored by Git. Slurm job `931317`
independently materialized a clean temporary Git tree containing exactly the
intended snapshot, passed the same nine tests and clean imports, and recompiled
the 10-page PDF.

## Slurm recovery

An initial submission attempt was rejected before job creation because the
partition name was written as lowercase `cpu`; the cluster exposes `CPU`.
The submission file was corrected without changing computation. Valid job IDs
are recorded in `reports/slurm/taas_subject_diffusion_freeze_job_ids.txt`.
