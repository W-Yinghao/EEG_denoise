# Data lookup status

Slurm job `918918` was cancelled at `2026-08-01T11:04:22Z` after the user
clarified that `/projects/EEG-foundation-model` is a private 30 TB workspace and
does not need an industrial, exhaustive inventory.

The scanner closed cleanly as `PARTIAL` with exit code `4`. It had visited
2,423,427 entries, hashed 10.28 GB across 216,671 allowlisted metadata files,
and emitted 2.23 GB of JSON. Its start/end Git, environment, mount and input
guards matched, so the cancellation was not `stale_input`. These partial files
are cancellation evidence only; no absence, availability, integrity, license,
or scientific claim may be derived from them.

The replacement workflow is deliberately small:

1. List the data-root top level and run one bounded-depth, name-only Slurm
   locator for the four requested datasets.
2. If a name matches, inspect only that candidate directory.
3. If no name matches, verify the official source and access terms, then use a
   versioned `.partial` download under the data root.
4. Record one small registry entry per dataset with path, version, source,
   access state and a short readability note. Use official checksums when they
   exist; do not hash the whole dataset merely to create local provenance.

The exhaustive scanner must not be resubmitted.

## Lightweight replacement result

- Slurm self-test `919128` passed.
- Slurm locator `919129` stopped normally after 10.9 seconds and 100,000
  basenames. It found no name match for Klados--Bamidis, SGEYESUB, Eye-BCI or
  EEGdenoiseNet and reported no directory errors. This is the intentionally
  bounded lookup requested by the user, not an absence proof.
- Slurm source probe `919130` reached the Mendeley landing page, OSF API,
  Synapse metadata API, GIN and GitHub. It read at most 8 KiB from each
  endpoint and retained no response bodies.
- An official EEGdenoiseNet clone at commit
  `8d290661146c7189c98cc04812d37371d4b9426c` supplied the six 256 Hz
  `.mat`/`.npy` payload files without another download.

Targeted checks resolved the four sources without another data-root scan:

- EEGdenoiseNet copy job `919131` published the six official 256 Hz files under
  the data root and read all three NPY headers. Job `919148` then replaced the
  duplicate code-tree files with links to the published copy.
- The anonymous Mendeley public-files route lists exactly one Klados v1 archive
  of 46,757,186 bytes under CC BY 4.0. Job `919153` downloaded it and minimal
  job `919191` checked only the official size and RAR4 signature; no native MAT
  sample was read.
- OSF `2qgrd` explicitly links CC BY 4.0. Job `919172` published its complete
  178-file, 1,611,314,510-byte tree. Job `919175` then read one epoched EEGLAB
  SET/FDT sample from each of five studies; all sampled values were finite.
- Eye-BCI remains `restricted` because Synapse requires a registered account
  token for downloads. No credential or package installation was attempted.

After removing the custom RAR member parser, lightweight submitter/self-test
job `919190` passed. Final targeted SGEYESUB metadata job `919218` matched all
59 participant SET/FDT/block-metadata stems, six channel layouts and the five
study counts in three seconds. It did not open or hash external FDT signals.
Every delivered SET exposed only blocks 1/2, so the paper-level three-block
mapping remains unresolved rather than guessed.

Final job `919220` also parsed the five gate states and four prior configs and
confirmed that the active records use readable config/version fields instead
of local threshold/prior hash fields.
