# WAVE4 — STOP 0: E0 manifest query and download-rule verdict

Preregistration commit: `e20eeb4` (frozen before any Synapse call).
Manifest artifact: `results/wave4_optical/manifest/e0_manifest.json`.
Query mode: `wave4-e0-manifest-only`, `content_downloaded: false` — entity metadata and
file-handle sizes only, via the raw Synapse REST API with the existing PAT
(`synapseclient` is absent from every environment on this cluster).

## Manifest, verbatim

| Quantity | Value |
| --- | --- |
| Authenticated | `true` |
| Project | `syn64005218` |
| Tobii files | 325 (315 `.csv` + 10 `.mp4`), 15.63 GiB |
| E-Prime files | 315 `.txt`, 0.0391 GiB |
| Tobii subjects | 31 (S01–S31) |
| Local EEG subjects | 31 (S01–S31) |
| Covered subjects (Tobii ∩ local EEG) | **31** |
| Total download | **15.666 GiB** |
| Data-root free space | 874.44 GiB |

Sampling rate was recorded as `NOT DETERMINABLE from manifest metadata (no content
read)` and deferred to E1, per the frozen mechanics.

## Frozen decision rule, applied

| Clause | Requirement | Observed | Result |
| --- | --- | --- | --- |
| (a) | Tobii covers ≥ 10 subjects that also have local EEG | 31 | **PASS** |
| (b) | Total download ≤ 300 GiB | 15.666 GiB | **PASS** |
| (c) | Free space ≥ 2× download (≥ 31.33 GiB) | 874.44 GiB | **PASS** |

**Verdict: PROCEED.** All three clauses fired, so E0b executed without waiting, exactly
as the execution order specifies.

## E0b download

Sharded (8 shards), resumable (HTTP `Range`), atomically published by rename from
`/projects/EEG-foundation-model/eye_bci/.syn64005218-tobii.partial` to
`.../syn64005218-tobii`. Publish verified all 630 selected files against their manifest
byte counts before the rename; the registry entry
`datasets/registry/eye_bci_tobii.json` was written after publish.

**Declared scope note.** The 10 Tobii scene-video `.mp4` files (S14/S16, 0.697 GiB) were
enumerated in the manifest but NOT downloaded: no measurement in M1–M4 consumes scene
video. The download selection was therefore 630 files (315 Tobii CSV + 315 E-Prime TXT).
The E-Prime logs were downloaded in full per the frozen rule ("E-Prime logs always
downloaded (tiny) if present") — they carry `Stimulus.OnsetTime`, the trigger train the
clock model names as its secondary path.

All downloads went to `/projects/EEG-foundation-model` as instructed. Zero sealed
contact. No file content was read before the rule was applied.
