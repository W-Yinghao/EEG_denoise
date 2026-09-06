# V37T paper scope boundary

V37T formally returns the TAAS revision to subject-aware diffusion for waveform-level ocular EEG
denoising. The frozen primary operating point is CalibEnergy-SDEdit with `lambda_y=0.5`,
`lambda_a=1`, final-only energy, and one draw for the registered point estimate.

## TAAS ownership

- Query-disjoint raw-support conditioning for waveform denoising.
- V25 deterministic/support encoder, V26 sensor-coordinate CalibSDEdit, and V27 partial-observation energy.
- Paired denoising, natural ocular attenuation, low-EOG observation retention, PSD/covariance trade-offs, and multi-sample uncertainty.
- Heterogeneous support intervention, support burden, and documented linkage risk as limitations.

## Debias/privacy-paper ownership

- General exact-fiber theorem and H-visible privacy boundary.
- CMI/TOS and full LEACE analysis.
- Fiber-Gaussian, Fiber-Stratified-Resample, and all V32P–V36P representation privacy results.

The TAAS revision may cite that line only as background or a limitation. It does not use fiber
methods as primary experiments, rename Gaussian sampling as diffusion, or transfer privacy claims
into waveform denoising. `taas_submission/**` remains read-only and is not compiled in V37T.
