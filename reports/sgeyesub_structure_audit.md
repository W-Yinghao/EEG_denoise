# SGEYESUB release structure audit

Slurm job `919218` completed in three seconds using `eeg2025`. It read the
small MATLAB/HDF5 metadata needed for participant, channel and trial structure,
checked that each SET has a same-stem FDT and block-metadata file, and did not
open any FDT signal payload. No data content hash was computed.

## Observed release

| OSF folder | recordings | channels | rate | samples/trial | blocks in SET | trial IDs |
|---|---:|---:|---:|---:|---|---|
| `study01` | 5 | 83 | 200 Hz | 1600 | 1, 2 | present |
| `study02` | 15 | 89 | 200 Hz | 1600 | 1, 2 | present |
| `study03` | 10 | 86 | 200 Hz | 1600 | 1, 2 | present |
| `study04` | 15 | 80 | 100 Hz | 800 | 1, 2 | present |
| `study05` | 14 | 80 | 256 Hz | 2048 | 1, 2 | absent |

The 59 SET files, 59 FDT files and 59 `_block_dt.mat` stems match one to one.
Six exact channel layouts were observed. For every recording, channel metadata
length equals `nbchan`, trial block and trial label counts equal `trials`, and
trial IDs are either complete or explicitly absent. The result is stored under
`reports/dataset_harness/jobs/919218/attempt-0/result.json`.

`trial_labels` contain the four calibration-paradigm classes. They are not the
same object as the sample-wise six-class `artifactclasses` channel used by the
native fitter. Query labels remain evaluation-only and cannot enter fitting or
selection.

The block metadata files expose one structured payload with fields
`s0/s1/s2/arr`; its units and semantics remain unidentified, so it is not used
as a sample source, label or drift covariate.

## Split consequence

The folder counts strongly resemble participant counts described in the paper,
but that is not enough to declare a mapping. In particular, every delivered SET
contains only blocks 1 and 2, while the paper describes a three-block EEGDS4
native comparison. `study05` also lacks `trial_ids`.

Therefore no `studyXX -> EEGDSX` mapping or native three-block split is frozen.
Participant stem remains the required outer unit, and block 1/2 are verified
disjoint candidate contexts, but the official mapping must be resolved before
claiming a native replication or Stage-A split closure.

Earlier attempts are engineering diagnostics only: `919192` identified the
MATLAB-v7.3 reader mismatch; `919207` exposed safety assertions to tighten;
`919210/919211` identified a SciPy `whosmat` incompatibility on the small block
metadata. The size-bounded reader passed in `919214`, and `919218` is the final
compact result.
