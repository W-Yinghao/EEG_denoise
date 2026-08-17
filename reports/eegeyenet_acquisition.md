# EEGEyeNet — acquisition and verification record

Dataset card for `/projects/EEG-foundation-model/eegeyenet/eegeyenet_min`.
Registry entry: `datasets/registry/eegeyenet_min.json`. Acquired 2026-08-15/17.

## Scope, as specified

| In | Out |
| --- | --- |
| `antisaccade_task_data/synchronised_min` — first 30 Drive-listing folders | `synchronised_max` (ICA/MARA/rPCA remove the ocular artifact under study) |
| `dots_data/synchronised_min` ("Large Grid") — all subjects | `processing_speed_data` (VSS) |
| Drive `README`/`LICENSE`, OSF `ktv7m` description PDFs | `prepared/` (16 benchmark ML tensors — enumerated in `eegeyenet_prepared_manifest.json`) |
| | raw unsynchronized |

The Drive folder linked from OSF `ktv7m` is the official distribution; OSF `osfstorage`
holds only the PDFs, so there is no mirror to fall back on.

## Delivered

| Group | Files | Subjects | Hours | Channels | MAT format |
| --- | ---: | ---: | ---: | ---: | --- |
| `antisaccade_min` | 28 | 28 | 6.51 | 129 | v7.3 HDF5, var `EEG` |
| `dots_min` | 177 | 30 | 15.79 | 133 | v5, var **`sEEG`** |
| `docs` | 6 | — | — | — | — |

211 files, 28.89 GiB, 500 Hz throughout. `antisaccade_min` is 28 not 30 because `AA2`
and `AB8` are **empty folders upstream** — verified twice against the Drive listing. The
full `synchronised_min` set has 370 subjects, so the 28 is a slice, not a ceiling.

## Verification (`reports/eegeyenet_min_qc.json`, Slurm job 945707)

`publish` checks only the first 8 bytes of each `.mat`, which a truncated v5 file passes.
So every one of the 205 `.mat` files was additionally opened in full:

**205/205 PASS** — 0 load errors, 0 truncated, 0 max-pipeline contamination, 205/205
carry the eye-tracking event layer. 15,411 labelled blinks and 97,670 labelled saccades
in total.

**The ocular artifact is intact.** All 28 antisaccade files carry an `automagic`
provenance struct recording `EOGRegression = no`, `iclabel = no`, `mara = no`,
`rpca = no`. Only filtering, bad-channel detection and interpolation were applied.

**Honest limit of that check**: the 177 `dots_min` files carry **no `automagic` struct**,
so for them the min pipeline cannot be proven from file contents — the QC marks them
`no_record`, not `min_confirmed`. Their provenance rests on the source folder and the
manifest file ids. Corroborating (not proving): they retain `L-GAZE-*`/`L-AREA` channels
and an average reference, inconsistent with the max pipeline's output.

## Two structural facts that constrain downstream use

1. **Continuous gaze/pupil channels exist only in `dots_min`.** Its 133 channels are
   129 EEG (`E1`–`E128` + `Cz`) plus `TIME`, `L-GAZE-X`, `L-GAZE-Y`, `L-AREA`.
   `antisaccade_min` has 129 channels and stops at `Cz` — event-level eye parameters
   only, no per-sample gaze time series. Any design needing a continuous optical
   reference must be built on dots.
2. **Two loaders are required.** `antisaccade_min` is MAT v7.3 → `h5py`, top-level
   `EEG` (+ `automagic`). `dots_min` is MAT v5 → `scipy.io.loadmat`, top-level
   **`sEEG`**, no `automagic`. Code assuming a single format or the name `EEG` fails on
   177 of 205 files.

Event-level eye parameters are present in **both** groups: `EEG.event` carries
`sac_amplitude`, `sac_vmax`, `sac_startpos_x/y`, `sac_endpos_x/y`, `fix_avgpos_x/y`,
`fix_avgpupilsize`, with types `L_saccade` / `L_fixation` / `L_blink`, alongside the task
trigger codes. Reference is `Cz`-common for antisaccade and average for dots.

## Acquisition route

Anonymous Google Drive route (`drive.usercontent.google.com/download?id=…&confirm=t`),
HTTP Range resume, per-file atomic rename, sharded 6 ways on Slurm partition `CPU`.

Google's public-file limit is on **cumulative bytes served to the client**, not per-file
size or count: the first window (2026-08-15) delivered 21 of 211 files ≈ 2.8 GiB and then
returned the HTML quota page for all 190 remaining requests. The quota had reset by
2026-08-17 and the rerun completed 190/190 with 0 failures. A resumed run is idempotent —
present files are skipped and `.partial` files continue by Range — so a quota block costs
nothing but a retry.

**Caution for manual top-up**: `synchronised_min` and `synchronised_max` have identical
370-subject folder structures and near-identical filenames (per subject the `gip_`/`oip_`
prefix may differ *between* the two folders — e.g. `BZ9` is `oip_` in min but `gip_` in
max), so navigating the Drive UI is error-prone. Use the `id`s in
`eegeyenet_min_manifest.json`, which resolve only to min files.

## Reproduce

```bash
python scripts/eegeyenet_download.py manifest             # enumerate Drive + OSF
sbatch scripts/slurm/eegeyenet_fetch.sbatch               # sharded, resumable fetch
python scripts/eegeyenet_download.py publish              # header-verify + atomic publish
sbatch scripts/slurm/eegeyenet_verify.sbatch              # full-file load + pipeline check
```
