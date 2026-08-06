# Literature-guided subject-aware diffusion exploration v3

Status: active full-development exploration; no confirmation or manuscript
claim is made from this directory.

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

## Active Slurm evidence

J0/J1 completed the named-data, official-source, real-record, split, and unit
tests. The first A100 technical attempt failed before model execution because
CUDA did not initialize; the identical H100 retry passed on real Klados data.
The P-A full one-seed screen is running with 26 frozen tasks and at most eight
concurrent GPUs. Scientific results, route ordering, and recommendations remain
pending until every planned route has full Klados and SGE coverage.

## Claim boundary

The candidate literature gap remains support-conditioned personalized
diffusion for physiological EEG artifact removal on unseen participants under
disjoint early-support to later-query evaluation. The review does not yet
support a “first” claim, and v3 results will remain development evidence.
