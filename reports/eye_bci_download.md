# Eye-BCI Neuroscan download

## Outcome

- Synapse project: `syn64005218` (`Eye-BCI_multi_dataset`).
- Local selection: `/projects/EEG-foundation-model/eye_bci/syn64005218-neuroscan`.
- Scope: all Neuroscan EEG CSV files; no Phantom AVI, Tobii, or E-Prime files.
- Subjects/sessions/files: 31 / 63 / 315.
- Bytes: 196,586,310,072 (about 183.1 GiB).
- Local hashes were not computed.

The selection is sufficient for the first EEG and external-HEO work because the
Neuroscan files include the scalp channels, `HEO`, `Trig`, `Cues`, `Blinks`, and
the available synchronization columns. Eye-BCI does not provide paired clean EEG.

## Slurm evidence

| Job | Purpose | Result |
|---|---|---|
| `919267` | PAT login and project access | authenticated |
| `919275` | Neuroscan-only metadata and size inventory | 315 files, 196,586,310,072 bytes |
| `919276` | download-scope probe without file content | authorized |
| `919279_0` | S01 pilot | 2,601,183,543 bytes in 99.64 s |
| `919286_[0-3]` | S02-S31 in four balanced shards | all completed |
| `919290` | validate and publish partial tree | published |
| `919311` | all-file size/header/first-row audit | verified readable |
| `919312` | hardened no-extra-file publication recheck | already published |

The four remaining-subject shards transferred 193,985,126,529 bytes in at most
1,887.92 seconds, about 103 MB/s aggregate. The S01 single-task pilot achieved
about 26 MB/s. Parallel network/filesystem streams therefore helped materially;
a GPU or additional cores within one downloader would not help this workload.

## Schema variants

- 303 files have the full 74-column schema.
- The five files in each of `S03/Sess01` and `S06/Sess01` lack
  `RecordingTimestamp` and `LocalTimeStamp`.
- `S10/Sess01/Neuroscan/P3004L101.csv` and
  `S18/Sess01/Neuroscan/ME181.csv` lack `PhanFrame`, `PhanTime`, and `RelTime`.

These known synchronization gaps do not invalidate EEG-only use. They must be
excluded or explicitly marked for endpoints requiring the missing cross-modal
timestamps. `M1` and `M2` remain semantically unresolved and must not be used for
rereferencing until their source meaning is confirmed.

Official sources: [Synapse project](https://www.synapse.org/Synapse%3Asyn64005218/),
[Scientific Data descriptor](https://www.nature.com/articles/s41597-025-04861-9).
