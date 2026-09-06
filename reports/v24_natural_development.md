# V24 natural development

Natural trade-off classification: `artifact_reduction_insufficient`. Test inference
used no query EOG, query operator, or event labels; evaluator access followed the
15-output digest freeze.

| Method | Remaining ratio | Attenuation (dB) | Preservation | PSD distortion | Covariance distortion |
|---|---:|---:|---:|---:|---:|
| V24 population anchor | 0.989638 | +1.243976 | 0.828301 | 0.359130 | 0.218406 |
| PA-EL-DET MATCH | 1.012106 | +1.246205 | 0.782816 | 0.403282 | 0.285062 |
| PA-EL-SCAD-K1 MATCH | 1.363523 | -0.465793 | 0.596115 | 0.552748 | 0.655928 |

TemporalEOGNet had moderate natural latent predictability (participant-first
correlation 0.610227), but the fixed support deviation did not turn it into a useful
artifact correction. Relative to POP, SCAD MATCH artifact utility was -0.373885 and
preservation utility was -0.232186. These two axes are shown separately; attenuation
alone is not interpreted as denoising success.
