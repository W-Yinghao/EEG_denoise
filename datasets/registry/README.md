# Dataset registry

Keep one small JSON record per target dataset. A record contains only the local
path, official source/version, access state, a short sample-read result, the
Slurm job ID and notes useful to this project.

Do not create per-file manifests or local hashes merely for registry
bookkeeping. If an official source publishes a checksum, it may be recorded in
`notes`; otherwise a successful targeted read is enough for this private
project. EEG files remain under `/projects/EEG-foundation-model`, never here.
