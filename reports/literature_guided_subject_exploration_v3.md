# Literature-guided subject-aware diffusion exploration v3

Status: completed full-development one-seed exploration; no confirmation or
manuscript claim is made from this directory.

## Inherited evidence boundary

The prior no-go applies to the tested analytic full-C/projector/FIR bridge and
the tested guidance/SDEdit instances. Matching support sometimes produced
specific output changes, while fair absolute comparisons did not establish an
advantage over the strong population cleaner. That evidence neither tests nor
rejects raw-support conditioning, direct support adaptation, or selective use
of independently trained population and matching outputs.

## Data availability and v3 role

| Dataset | Observed server state | Structure | v3 role |
|---|---|---|---|
| EEGdenoiseNet | Present under `/projects/EEG-foundation-model/eegdenoisenet`; clean EEG, EOG and EMG arrays available | Source epochs; no participant identity | Reuse completed official/strict-source EEGDfus population benchmark |
| SSED | No named server path in bounded audit | Not opened | Official EEGDfus second-dataset reproduction blocked by absence |
| Klados v4 | Four MAT containers; 54 source records already mapped | Source-record identity only | Paired mechanism screen; 16 frozen evaluation source records |
| SGEYESUB | 59 release stems, 58 compatible with population cells | Participant stem, block-1 support to block-2 query | Natural ocular-artifact development and MATCH/WRONG intervention |
| Eye-BCI | Present; 31 participants, 63 sessions, 315 Neuroscan CSV files from the accepted prior audit | Participant/session structure | Already-exposed development sessions only; not used for a new confirmation claim |
| EEGEyeNet | Named directory contains metadata/documentation only, not signal files | Signal data unavailable | Blocked; no Google Drive retry |
| MobileBCI | Official OSF R7S9B acquisition completed at `/projects/EEG-foundation-model/mobile_bci`: 2,973 files, 9,125,232,937 declared bytes | Header-only audit found 24 participants and 198 record headers. All source channel tables retain 46 EEG plus four EOG. Processed BIDS tables omit EOG; 196/198 have 46 EEG plus 27 IMU and two lack processed IMU entries. | Development split frozen without opening outcomes: sub-01--16 training-development, sub-17--24 heldout-development; ses-02 standing support to ses-03/04/05 locomotion query; 47 eligible participant-task pairs |

The audit is restricted to named paths and relevant dataset directories. It
does not perform a whole-root scan of the shared 30 TB filesystem.

## Route definitions

- P-A encodes raw, query-disjoint calibration EEG/EOG windows with a
  permutation-invariant token encoder and injects the context by FiLM in every
  major diffusion block. Matching, population, three wrong donors, shuffled,
  and no-support arms share one checkpoint and random stream.
- P-B is a direct support-only, zero-initialized low-rank adapter upper bound on
  a frozen population backbone. It is not described as an amortized
  hypernetwork.
- P-C first measures a hindsight POP-versus-MATCH ceiling. Only if the ceiling
  exists does it fit a leave-one-unit-out discrete selector whose inference
  inputs are query EEG, POP--MATCH disagreement, and support-only quantiles.
- P-D is a fixed ReVIN-style support-statistic control, not an innovation
  claim.

## Slurm execution evidence

J0/J1 completed the named-data, official-source, real-record, split, and unit
tests. The first A100 technical attempt failed before model execution because
CUDA did not initialize; the identical H100 retry passed on real Klados data.
P-A, P-B, and P-D each completed all 26 frozen tasks with at most eight
concurrent GPUs: 16/16 Klados source records and 58 compatible SGE stems while
retaining the blocked singleton in the 59-stem denominator. Two submitted
array entries used unsupported stage aliases and exited before model/data work;
their corrected retries used the same frozen tasks and scientific settings.
P-C completed its full-unit oracle ceiling, and J7/J8 completed aggregation,
15/15 tests, and a clean-checkout import.

## Claim boundary

The candidate literature gap remains support-conditioned personalized
diffusion for physiological EEG artifact removal on unseen participants under
disjoint early-support to later-query evaluation. The review does not yet
support a “first” claim, and v3 results will remain development evidence.

## J7 full-development results

All values below were aggregated by source record or participant stem; windows and sampler draws are not treated as independent units. The one-seed intervals are descriptive development evidence, not confirmation.

### Absolute method results

| Route | Dataset | Method | Units | Primary artifact/clean metric | Preservation | PSD distortion | Covariance distortion | Output/input RMS |
|---|---|---|---:|---:|---:|---:|---:|---:|
| P_A_RAW_SUPPORT_TOKENS | klados | DET-MATCH | 16 | 0.31476 | N/A | N/A | N/A | 0.63160 |
| P_A_RAW_SUPPORT_TOKENS | klados | DET-NO-SUPPORT | 16 | 0.34027 | N/A | N/A | N/A | 0.63553 |
| P_A_RAW_SUPPORT_TOKENS | klados | DET-POP | 16 | 0.43621 | N/A | N/A | N/A | 0.65074 |
| P_A_RAW_SUPPORT_TOKENS | klados | DET-SHUFFLED | 16 | 0.31729 | N/A | N/A | N/A | 0.63204 |
| P_A_RAW_SUPPORT_TOKENS | klados | DET-WRONG-1 | 16 | 0.36878 | N/A | N/A | N/A | 0.63990 |
| P_A_RAW_SUPPORT_TOKENS | klados | DET-WRONG-2 | 16 | 0.41976 | N/A | N/A | N/A | 0.64864 |
| P_A_RAW_SUPPORT_TOKENS | klados | DET-WRONG-3 | 16 | 0.32142 | N/A | N/A | N/A | 0.63303 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-MATCH | 16 | 0.35884 | N/A | N/A | N/A | 0.64362 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-NO-SUPPORT | 16 | 0.36811 | N/A | N/A | N/A | 0.64528 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-POP | 16 | 0.47291 | N/A | N/A | N/A | 0.66639 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-SHUFFLED | 16 | 0.36491 | N/A | N/A | N/A | 0.64260 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-WRONG-1 | 16 | 0.39514 | N/A | N/A | N/A | 0.65165 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-WRONG-2 | 16 | 0.41738 | N/A | N/A | N/A | 0.65624 |
| P_A_RAW_SUPPORT_TOKENS | klados | DIFF-WRONG-3 | 16 | 0.38044 | N/A | N/A | N/A | 0.64910 |
| P_A_RAW_SUPPORT_TOKENS | klados | RAW | 16 | 1.29481 | N/A | N/A | N/A | 1.00000 |
| P_A_RAW_SUPPORT_TOKENS | klados | STRONG-POP | 16 | 0.44054 | N/A | N/A | N/A | 0.65779 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-MATCH | 58 | 0.29535 | 0.82089 | 0.29563 | 0.07894 | 0.77733 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-NO-SUPPORT | 58 | 0.29422 | 0.81233 | 0.30831 | 0.08659 | 0.77794 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-POP | 58 | 0.29316 | 0.81172 | 0.30612 | 0.08331 | 0.77967 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-SHUFFLED | 58 | 0.29672 | 0.82107 | 0.29427 | 0.07854 | 0.77609 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-WRONG-1 | 58 | 0.29915 | 0.81716 | 0.30397 | 0.08232 | 0.77573 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-WRONG-2 | 58 | 0.29276 | 0.82008 | 0.30403 | 0.08220 | 0.77976 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET-WRONG-3 | 58 | 0.28940 | 0.81461 | 0.32446 | 0.08381 | 0.77740 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-MATCH | 58 | 0.26350 | 0.84389 | 0.27136 | 0.06786 | 0.78922 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-NO-SUPPORT | 58 | 0.26349 | 0.84390 | 0.26927 | 0.07134 | 0.78744 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-POP | 58 | 0.26483 | 0.83738 | 0.27859 | 0.07347 | 0.78741 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-SHUFFLED | 58 | 0.26344 | 0.84412 | 0.27038 | 0.06783 | 0.78901 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-WRONG-1 | 58 | 0.26744 | 0.83992 | 0.27271 | 0.07117 | 0.78840 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-WRONG-2 | 58 | 0.27160 | 0.83524 | 0.28370 | 0.06958 | 0.78962 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DIFF-WRONG-3 | 58 | 0.25188 | 0.83666 | 0.28526 | 0.08247 | 0.78685 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | RAW | 58 | 0.00000 | 1.00000 | 0.00000 | 0.00000 | 1.00000 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | STRONG-POP | 58 | 0.32208 | 0.77275 | 0.57525 | 0.09345 | 0.77366 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-MATCH | 16 | 0.34322 | N/A | N/A | N/A | 0.63585 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-NO-SUPPORT | 16 | 0.34028 | N/A | N/A | N/A | 0.63553 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-POP | 16 | 0.34054 | N/A | N/A | N/A | 0.63526 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-SHUFFLED | 16 | 0.34556 | N/A | N/A | N/A | 0.63741 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-WRONG-1 | 16 | 0.34028 | N/A | N/A | N/A | 0.63553 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-WRONG-2 | 16 | 0.34028 | N/A | N/A | N/A | 0.63553 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET-WRONG-3 | 16 | 0.34028 | N/A | N/A | N/A | 0.63553 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-MATCH | 16 | 0.36113 | N/A | N/A | N/A | 0.64348 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-NO-SUPPORT | 16 | 0.36809 | N/A | N/A | N/A | 0.64527 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-POP | 16 | 0.36334 | N/A | N/A | N/A | 0.64414 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-SHUFFLED | 16 | 0.37456 | N/A | N/A | N/A | 0.64693 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-WRONG-1 | 16 | 0.35748 | N/A | N/A | N/A | 0.64273 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-WRONG-2 | 16 | 0.35958 | N/A | N/A | N/A | 0.64298 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DIFF-WRONG-3 | 16 | 0.36873 | N/A | N/A | N/A | 0.64548 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | RAW | 16 | 1.29481 | N/A | N/A | N/A | 1.00000 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | STRONG-POP | 16 | 0.44057 | N/A | N/A | N/A | 0.65781 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-MATCH | 58 | 0.29698 | 0.81416 | 0.30452 | 0.08593 | 0.77663 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-NO-SUPPORT | 58 | 0.29421 | 0.81233 | 0.30829 | 0.08659 | 0.77793 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-POP | 58 | 0.29223 | 0.81376 | 0.30620 | 0.08646 | 0.77782 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-SHUFFLED | 58 | 0.29189 | 0.81711 | 0.30031 | 0.08505 | 0.77673 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-WRONG-1 | 58 | 0.29441 | 0.81322 | 0.30696 | 0.08654 | 0.77757 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-WRONG-2 | 58 | 0.29498 | 0.81349 | 0.30643 | 0.08625 | 0.77771 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET-WRONG-3 | 58 | 0.29503 | 0.81312 | 0.30769 | 0.08653 | 0.77770 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-MATCH | 58 | 0.26315 | 0.84442 | 0.26835 | 0.07110 | 0.78719 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-NO-SUPPORT | 58 | 0.26349 | 0.84390 | 0.26926 | 0.07134 | 0.78744 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-POP | 58 | 0.26325 | 0.84390 | 0.26927 | 0.07133 | 0.78763 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-SHUFFLED | 58 | 0.25798 | 0.84562 | 0.26358 | 0.07031 | 0.78756 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-WRONG-1 | 58 | 0.26403 | 0.84384 | 0.26937 | 0.07144 | 0.78741 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-WRONG-2 | 58 | 0.26406 | 0.84367 | 0.27018 | 0.07142 | 0.78763 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DIFF-WRONG-3 | 58 | 0.26367 | 0.84383 | 0.26925 | 0.07134 | 0.78744 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | RAW | 58 | 0.00000 | 1.00000 | 0.00000 | 0.00000 | 1.00000 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | STRONG-POP | 58 | 0.32208 | 0.77275 | 0.57524 | 0.09346 | 0.77365 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-MATCH | 16 | 0.89576 | N/A | N/A | N/A | 0.82677 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-NO-SUPPORT | 16 | 0.34028 | N/A | N/A | N/A | 0.63553 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-POP | 16 | 0.89707 | N/A | N/A | N/A | 0.82565 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-SHUFFLED | 16 | 0.89173 | N/A | N/A | N/A | 0.82605 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-WRONG-1 | 16 | 0.81379 | N/A | N/A | N/A | 0.79101 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-WRONG-2 | 16 | 0.49079 | N/A | N/A | N/A | 0.67532 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET-WRONG-3 | 16 | 0.82958 | N/A | N/A | N/A | 0.80175 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-MATCH | 16 | 0.85892 | N/A | N/A | N/A | 0.80684 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-NO-SUPPORT | 16 | 0.36809 | N/A | N/A | N/A | 0.64527 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-POP | 16 | 0.90397 | N/A | N/A | N/A | 0.82775 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-SHUFFLED | 16 | 0.83495 | N/A | N/A | N/A | 0.79611 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-WRONG-1 | 16 | 0.80399 | N/A | N/A | N/A | 0.78499 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-WRONG-2 | 16 | 0.58815 | N/A | N/A | N/A | 0.70929 |
| P_D_SUPPORT_STAT_CONTROL | klados | DIFF-WRONG-3 | 16 | 0.81348 | N/A | N/A | N/A | 0.79385 |
| P_D_SUPPORT_STAT_CONTROL | klados | RAW | 16 | 1.29481 | N/A | N/A | N/A | 1.00000 |
| P_D_SUPPORT_STAT_CONTROL | klados | STRONG-POP | 16 | 0.44057 | N/A | N/A | N/A | 0.65781 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-MATCH | 58 | 0.14380 | 0.71066 | 0.46959 | 0.15042 | 0.90885 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-NO-SUPPORT | 58 | 0.29422 | 0.81233 | 0.30828 | 0.08659 | 0.77794 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-POP | 58 | 0.14934 | 0.73285 | 0.45664 | 0.14559 | 0.85972 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-SHUFFLED | 58 | 0.14355 | 0.71090 | 0.46778 | 0.15054 | 0.90867 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-WRONG-1 | 58 | 0.12624 | 0.66362 | 0.60130 | 0.18709 | 0.90467 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-WRONG-2 | 58 | 0.18145 | 0.78718 | 0.36075 | 0.12566 | 0.84126 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET-WRONG-3 | 58 | 0.13166 | 0.62463 | 0.68718 | 0.19383 | 0.96529 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-MATCH | 58 | 0.19751 | 0.72389 | 0.46775 | 0.16126 | 0.85448 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-NO-SUPPORT | 58 | 0.26349 | 0.84390 | 0.26921 | 0.07134 | 0.78744 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-POP | 58 | 0.21390 | 0.75436 | 0.44307 | 0.13060 | 0.82599 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-SHUFFLED | 58 | 0.19712 | 0.72485 | 0.46591 | 0.16045 | 0.85445 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-WRONG-1 | 58 | 0.22381 | 0.69216 | 0.55690 | 0.18174 | 0.83054 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-WRONG-2 | 58 | 0.18884 | 0.80121 | 0.37605 | 0.13430 | 0.83271 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DIFF-WRONG-3 | 58 | 0.23575 | 0.65572 | 0.65201 | 0.19179 | 0.85938 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | RAW | 58 | 0.00000 | 1.00000 | 0.00000 | 0.00000 | 1.00000 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | STRONG-POP | 58 | 0.32208 | 0.77274 | 0.57526 | 0.09346 | 0.77365 |

### Fair paired effects

Positive values favor the first method named by the estimand.

| Route | Dataset | Estimand | n | Mean | Median | 95% CI | Positive units |
|---|---|---|---:|---:|---:|---:|---:|
| P_A_RAW_SUPPORT_TOKENS | klados | DET_subject_DET_MATCH_minus_DET_POP | 16 | 0.12145 | 0.13951 | [0.08225, 0.16171] | 15 |
| P_A_RAW_SUPPORT_TOKENS | klados | H_D_DIFF_MATCH_minus_DET_MATCH | 16 | -0.04408 | -0.03868 | [-0.06831, -0.02163] | 4 |
| P_A_RAW_SUPPORT_TOKENS | klados | H_S1_DIFF_MATCH_minus_DIFF_POP | 16 | 0.11407 | 0.12971 | [0.08474, 0.14047] | 15 |
| P_A_RAW_SUPPORT_TOKENS | klados | H_S2_DIFF_MATCH_minus_mean_WRONG | 16 | 0.03882 | 0.03669 | [0.02093, 0.05942] | 14 |
| P_A_RAW_SUPPORT_TOKENS | klados | INTERACTION_subject_diffusion_minus_deterministic | 16 | -0.00738 | 0.01185 | [-0.03842, 0.02210] | 9 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | DET_subject_DET_MATCH_minus_DET_POP | 58 | 0.00219 | -0.00009 | [-0.00404, 0.00814] | 29 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | H_D_DIFF_MATCH_minus_DET_MATCH | 58 | -0.03185 | -0.01366 | [-0.05169, -0.01427] | 16 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | H_S1_DIFF_MATCH_minus_DIFF_POP | 58 | -0.00133 | 0.00032 | [-0.00800, 0.00537] | 30 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | H_S2_DIFF_MATCH_minus_mean_WRONG | 58 | -0.00014 | 0.00092 | [-0.00508, 0.00447] | 32 |
| P_A_RAW_SUPPORT_TOKENS | sgeyesub | INTERACTION_subject_diffusion_minus_deterministic | 58 | -0.00352 | -0.00192 | [-0.01257, 0.00530] | 28 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | DET_subject_DET_MATCH_minus_DET_POP | 16 | -0.00268 | -0.00183 | [-0.00867, 0.00370] | 6 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | H_D_DIFF_MATCH_minus_DET_MATCH | 16 | -0.01791 | -0.00401 | [-0.04630, 0.00862] | 8 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | H_S1_DIFF_MATCH_minus_DIFF_POP | 16 | 0.00221 | 0.00022 | [-0.00361, 0.00806] | 8 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | H_S2_DIFF_MATCH_minus_mean_WRONG | 16 | 0.00080 | -0.00144 | [-0.00282, 0.00483] | 6 |
| P_B_DIRECT_SUPPORT_ADAPTER | klados | INTERACTION_subject_diffusion_minus_deterministic | 16 | 0.00489 | 0.00736 | [-0.00319, 0.01295] | 9 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | DET_subject_DET_MATCH_minus_DET_POP | 58 | 0.00475 | 0.00000 | [0.00166, 0.00790] | 28 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | H_D_DIFF_MATCH_minus_DET_MATCH | 58 | -0.03384 | -0.02070 | [-0.05260, -0.01619] | 19 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | H_S1_DIFF_MATCH_minus_DIFF_POP | 58 | -0.00011 | 0.00000 | [-0.00196, 0.00198] | 21 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | H_S2_DIFF_MATCH_minus_mean_WRONG | 58 | -0.00077 | -0.00143 | [-0.00257, 0.00125] | 20 |
| P_B_DIRECT_SUPPORT_ADAPTER | sgeyesub | INTERACTION_subject_diffusion_minus_deterministic | 58 | -0.00486 | -0.00183 | [-0.00775, -0.00192] | 21 |
| P_D_SUPPORT_STAT_CONTROL | klados | DET_subject_DET_MATCH_minus_DET_POP | 16 | 0.00131 | -0.04020 | [-0.07325, 0.08990] | 5 |
| P_D_SUPPORT_STAT_CONTROL | klados | H_D_DIFF_MATCH_minus_DET_MATCH | 16 | 0.03684 | 0.03221 | [0.01081, 0.06224] | 12 |
| P_D_SUPPORT_STAT_CONTROL | klados | H_S1_DIFF_MATCH_minus_DIFF_POP | 16 | 0.04505 | 0.03956 | [-0.01754, 0.11381] | 10 |
| P_D_SUPPORT_STAT_CONTROL | klados | H_S2_DIFF_MATCH_minus_mean_WRONG | 16 | -0.12371 | -0.13595 | [-0.19218, -0.05079] | 3 |
| P_D_SUPPORT_STAT_CONTROL | klados | INTERACTION_subject_diffusion_minus_deterministic | 16 | 0.04374 | 0.04870 | [0.01498, 0.07150] | 13 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | DET_subject_DET_MATCH_minus_DET_POP | 58 | -0.00554 | -0.00733 | [-0.03362, 0.02254] | 29 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | H_D_DIFF_MATCH_minus_DET_MATCH | 58 | 0.05371 | 0.00265 | [0.03018, 0.07621] | 31 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | H_S1_DIFF_MATCH_minus_DIFF_POP | 58 | -0.01640 | -0.01754 | [-0.04439, 0.01255] | 27 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | H_S2_DIFF_MATCH_minus_mean_WRONG | 58 | -0.01863 | -0.02167 | [-0.04133, 0.00355] | 21 |
| P_D_SUPPORT_STAT_CONTROL | sgeyesub | INTERACTION_subject_diffusion_minus_deterministic | 58 | -0.01086 | -0.00389 | [-0.03634, 0.01458] | 26 |

### Coverage and route interpretation

- P_A_RAW_SUPPORT_TOKENS / klados: 16/16 successful statistical units; 0 blocked or missing.
- P_A_RAW_SUPPORT_TOKENS / sgeyesub: 58/59 successful statistical units; 1 blocked or missing.
- P_B_DIRECT_SUPPORT_ADAPTER / klados: 16/16 successful statistical units; 0 blocked or missing.
- P_B_DIRECT_SUPPORT_ADAPTER / sgeyesub: 58/59 successful statistical units; 1 blocked or missing.
- P_D_SUPPORT_STAT_CONTROL / klados: 16/16 successful statistical units; 0 blocked or missing.
- P_D_SUPPORT_STAT_CONTROL / sgeyesub: 58/59 successful statistical units; 1 blocked or missing.

- P_A_RAW_SUPPORT_TOKENS: absolute=True; diffusion-direction-consistent=False; subject-utility-consistent=False; wrong-specificity-consistent=False; SGE-safety=True.
- P_B_DIRECT_SUPPORT_ADAPTER: absolute=True; diffusion-direction-consistent=False; subject-utility-consistent=False; wrong-specificity-consistent=False; SGE-safety=True.
- P_D_SUPPORT_STAT_CONTROL: absolute=True; diffusion-direction-consistent=True; subject-utility-consistent=False; wrong-specificity-consistent=False; SGE-safety=False.

### Selective correction

- P-C status: completed_low_selective_ceiling.
- Oracle diagnostic klados at coverage 0.50: utility vs POP=0.12573, preservation=1.00000, n=16.
- Oracle diagnostic klados at coverage 0.80: utility vs POP=0.12363, preservation=1.00000, n=16.
- Oracle diagnostic klados at coverage 1.00: utility vs POP=0.06959, preservation=1.00000, n=16.
- Oracle diagnostic sgeyesub at coverage 0.50: utility vs POP=0.00815, preservation=0.80714, n=58.
- Oracle diagnostic sgeyesub at coverage 0.80: utility vs POP=-0.00159, preservation=0.82112, n=58.
- Oracle diagnostic sgeyesub at coverage 1.00: utility vs POP=-0.02261, preservation=0.82155, n=58.

### Official implementation/runtime status

- EEGDfus: completed_prior_full_population_benchmark (official_source_port); rrmse=0.29653, correlation=0.95304, delta_snr_db=14.50353.
- EEGDfus-matched-deterministic: completed_prior_full_population_benchmark (official_source_port); rrmse=0.31870, correlation=0.94409, delta_snr_db=14.62758.
- EEGDfus: completed_prior_full_population_benchmark (official_source_port); rrmse=0.42054, correlation=0.90651, delta_snr_db=11.11646.
- EEGDfus-matched-deterministic: completed_prior_full_population_benchmark (official_source_port); rrmse=0.40572, correlation=0.90675, delta_snr_db=11.60650.
- EEGDfus: completed_prior_full_population_benchmark (official_source_port); rrmse=0.27397, correlation=0.95971, delta_snr_db=15.48223.
- EEGDfus-matched-deterministic: completed_prior_full_population_benchmark (official_source_port); rrmse=0.26162, correlation=0.96403, delta_snr_db=15.15804.
- EEGDfus: completed_prior_full_population_benchmark (official_source_port); rrmse=0.39494, correlation=0.91321, delta_snr_db=10.98097.
- EEGDfus-matched-deterministic: completed_prior_full_population_benchmark (official_source_port); rrmse=0.37704, correlation=0.92128, delta_snr_db=11.39288.
- EEGOAR-Net: blocked_official_runtime_dependency_or_weight_incompatibility (official_pretrained_weights).
- D4PM: blocked_exact_official_matrix_missing_ECG_array (official_source_audit).
- SGEYESUB-source-faithful-Python: completed_prior_all_study_development_matrix (source_faithful_python_port_not_MATLAB_parity).
- Essentia: reconstructed_from_paper (reconstructed_from_paper).
- SGEYESUB: source_faithful_python_port (source_faithful_python_port).
- DeepSeparator: ported_legacy_population_baseline (ported_legacy_population_baseline).
- ART: official_source_available_target_port_not_run (official_source_available_target_port_not_run).
- IC-U-Net: official_source_available_inference_port_not_run (official_source_available_inference_port_not_run).
- ICA+ICLabel: library_baseline_available (library_baseline_available).
- ASR: library_baseline_available (library_baseline_available).
- MNE-EOGRegression: library_baseline_available (library_baseline_available).

### Next-route recommendation

No route satisfies every development axis; no additional-seed route is recommended from this screen.

This result does not support a family-wide negative conclusion. It identifies only the behavior of the tested raw-support, direct-adapter, selective, and support-statistic instances on the stated development evidence.
