# V40R third-party official-code audit

All checkouts are outside this repository under `/home/infres/yinwang/v40r_third_party`, detached at
the registered commits, and have clean tracked trees. No third-party source or checkpoint is vendored.

- EEGDfus is the required anchor. The frozen repository has no license file, so it is used only for
  internal reproducibility. Its official architecture/configuration was dynamically imported without
  modifying the checkout. The completed historical run used 4000 epochs, batch 512, Adam at 1e-3,
  epsilon prediction, a 500-step linear schedule, and full ancestral sampling. The local seeded data
  wrapper and corrected spectral denominator are disclosed; classification is
  `reasonable_nonidentical_reproduction`.
- D4PM has no license file and its sparse release does not register enough data/checkpoint/evaluation
  semantics for an auditable run. The one source/import audit is recorded as
  `official_release_not_runnable`; no unofficial implementation is substituted.
- EEGdenoiseNet is MIT licensed and supplies the official data/mixing resource. Its frozen repository
  links model implementations rather than providing them in-tree.
- DeepSeparator provides author training and prediction code and bundled arrays but has no explicit
  license at the pinned revision. It is restricted to internal evaluation in its legacy environment.
- EEGOAR-Net is MIT licensed and supplies weights, but its declared 64-channel preprocessing contract
  is incompatible with the frozen 46-channel natural panel. No interpolation or padding is used, so
  the status is `protocol_incompatible`.
- The established SGEYESUB source/evaluator remains bound to its existing project provenance.

No compatibility patch changes a scientific method. The V40R multichannel model is a separately
documented local port, not presented as unchanged official EEGDfus.
