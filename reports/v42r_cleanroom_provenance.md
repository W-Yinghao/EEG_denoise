# V42R clean-room provenance

Reused audited internal utilities:

- `artifact_transfer_v41r.py`: two-bipolar support transfer estimation, fold-local normalization,
  query-disjoint manifests, and dynamic clean/EOG/operator episodes;
- `paired_metrics.py`: temporal, spectral, correlation, SNR, and artifact-field metrics;
- the frozen V25 five-fold participant split and V31 support-duration contract.

Independently implemented for V42R:

- four-scale joint multichannel 1D U-Net;
- observation-centered x0 population and transfer residual heads;
- permutation-aware transfer-state encoder;
- 1000-step linear forward process and deterministic 50-step DDIM;
- EMA training, POP-route full-sampler validation, checkpointing, and replay.

No collaborator SADDPM branch, superseded SADDPM implementation, third-party model source, or
unlicensed diffusion code is imported. The method is based on the current manuscript-level paired
conditional role and standard published DDPM/DDIM equations.
