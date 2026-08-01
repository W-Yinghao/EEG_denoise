# CSPD server execution boundary

This file supplements the legacy SADDPM repository documentation; it does not reclassify legacy
checkpoints, results, participant-conditioned code, or manuscript numbers as evidence.

- Code/audit root: `/home/infres/yinwang/denoiseNet`.
- Read-only EEG discovery root: `/projects/EEG-foundation-model`.
- All Python, environment, test, data, training, evaluation, and plotting payloads run through
  `scripts/slurm/submit.sh`; the login node is control plane only.
- Only the registered `eeg2025` and `icml` environments may be used. No install or upgrade is
  authorized.
- Dataset state begins as unknown. Phase-I path evidence cannot establish version, license,
  integrity, sample readability, `verified_available`, or `missing`.
- Scientific execution is currently blocked by `CONFLICT-SCI-001` through `003` in
  `reports/attachment_review.md`. Population-base/P0/gate implementations and all empirical claims
  remain absent. G1–G5 configs are disabled with `TBD-PREREG` thresholds; B1–B6 are disabled.
- No push, release, remote archive, EEG upload, or publication is authorized.

Administrative validation command (login-node submission only):

```bash
scripts/slurm/submit.sh cpu validate_control_plane
```

Do not invoke the Python payloads, tests, data readers, or model code directly on the login node.
