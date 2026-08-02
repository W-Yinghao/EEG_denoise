# SGEYESUB MATLAB parity

Status: `blocked_matlab_executable_unavailable`

The official `eyeartifactcorrection` source was checked out by CPU Slurm job
`919777` at commit `2c95b4f46f37670d25399ac0fdd705ae18248b25`.  CPU
Slurm job `919775` then checked the compute-node runtime: no `matlab`
executable or MATLAB module was available, so no license test or numerical
comparison was run.  Octave was not used as a substitute.

Consequently, waveform relative error, projector principal angles, artifact
attenuation, preservation, PSD, covariance, and ERP parity metrics are
`N/A (blocked before execution)`.  The existing Python implementation remains
labelled `source_faithful_not_numerically_cross_validated`; it is not described
as an exact official MATLAB reproduction.

Evidence:

- Runtime result: `reports/sgeyesub_matlab_probe/919775/availability.json`
- Official checkout result:
  `reports/sgeyesub_reference_checkout/919777/checkout.json`
- Frozen protocol: `configs/cgdr/sgeyesub_matlab_parity.yaml`
- Prepared runner: `scripts/matlab/sgeyesub_parity_runner.m`

This runtime blocker is independent of the corrected post-hoc SGEYESUB
operator audit and does not block the Klados sampler/deterministic comparison
or EEGDfus reproduction.
