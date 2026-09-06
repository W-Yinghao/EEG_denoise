# V37T common-panel point results

All values below are frozen V30 participant-first common-panel results. Absolute values and
relative mechanism comparisons are kept distinct.

## Paired panel

| Method | temporal RRMSE | spectral RRMSE | correlation | artifact RRMSE | identity change |
|---|---:|---:|---:|---:|---:|
| STANDARD | 0.816683 | 0.467400 | 0.870225 | 1.000000 | 0.000000 |
| V25 DET-MATCH | 0.751577 | 0.527347 | 0.880337 | 1.528384 | 0.192301 |
| V26 CalibSDEdit-MATCH | 0.754500 | 0.531895 | 0.879470 | 1.572589 | 0.201864 |
| V26 PopSDEdit | 0.766198 | 0.537068 | 0.876998 | 1.615093 | 0.201539 |
| V27 EnergySDEdit L0.5 | 0.747098 | 0.508301 | 0.881976 | 1.430805 | 0.168827 |
| matched EnergyDET L0.5 | 0.746192 | 0.506768 | 0.882280 | 1.417573 | 0.166550 |
| EEGDfus | 0.826732 | 0.538537 | 0.858090 | 1.679979 | 0.191640 |

L0.5 improves temporal RRMSE over STANDARD and EEGDfus but remains slightly behind its matched
EnergyDET point estimator. Artifact-field RRMSE above one and negative aggregate SNR improvement
remain explicit counterevidence; point performance is not summarized by the favorable clean RRMSE
alone.

## Natural panel

| Method | EOG remaining | attenuation dB | low-EOG retention | PSD distortion | covariance distortion |
|---|---:|---:|---:|---:|---:|
| V25 DET-MATCH | 0.931723 | 1.684950 | 0.778989 | 0.390432 | 0.273294 |
| V26 CalibSDEdit-MATCH | 0.938635 | 1.650918 | 0.766762 | 0.400812 | 0.276250 |
| V26 PopSDEdit | 0.946950 | 1.408797 | 0.777131 | 0.394534 | 0.265586 |
| V27 EnergySDEdit L0.5 | 0.929094 | 1.618737 | 0.807347 | 0.336428 | 0.240253 |
| V27 EnergySDEdit L2 | 0.949958 | 1.230372 | 0.857324 | 0.251419 | 0.183783 |
| V27 EnergySDEdit L8 | 0.996776 | 0.632913 | 0.909404 | 0.161959 | 0.117757 |
| matched EnergyDET L0.5 | 0.922113 | 1.711820 | 0.808079 | 0.334990 | 0.241105 |
| EEGDfus | 0.904122 | 1.058332 | 0.843568 | 0.215173 | 0.117866 |

All three V27 energy settings achieve absolute attenuation (`remaining<1`, `attenuation>0`). The
registered L0.5 point favors attenuation; increasing energy strength raises observation retention
and reduces PSD/covariance change while reducing artifact attenuation. Low-EOG retention is an
observation-retention statistic, not physiological preservation.

Support evidence remains deliberately narrow: paired support benefit exists, but all-donor ranks
and lagged/shuffled controls show heterogeneous, non-unique matched-donor specificity. V31—not the
superseded V30 curve—is authoritative for exact 0/5/10/30/120-second support duration.

