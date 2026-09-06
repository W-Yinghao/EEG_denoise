# v3 evidence correction

Historical files were preserved. P-A is **support moment-summary FiLM**, not a raw waveform/token route. P-B is a **support-fitted output-space residual adapter**, not LoRA. P-D is an inference-only normalization OOD hybrid and is excluded from scientific route ranking.

The former POP arm used four training-window exemplars. It is now named `POP-EXEMPLAR`; fair effects are reported separately against `NO-SUPPORT`, `STRONG-POP`, and the mean of three WRONG donors. Therefore the published v3 P-A Klados +0.1141 effect is not a fair population subject-utility estimate.

The narrow corrected conclusion is that these three concrete instances do not advance. This does not test the raw temporal-support route, internal LoRA, or the diffusion/personalization families.

## Corrected comparator audit

The fair re-read materially changes the interpretation. For the moment-summary FiLM route, mean utility was:

| Dataset | MATCH−POP-EXEMPLAR | MATCH−NO-SUPPORT | MATCH−STRONG-POP | MATCH−mean WRONG |
|---|---:|---:|---:|---:|
| Klados | +0.114074 | +0.009277 | +0.081701 | +0.038817 |
| SGEYESUB | −0.001331 | +0.000015 | −0.058582 | −0.000139 |

Thus the often-quoted Klados `+0.1141` is specifically MATCH versus the four-window POP exemplar. It is not a fair subject-utility estimate against no support or the strong population cleaner, and it does not replicate on SGEYESUB. The output-space residual adapter likewise loses to STRONG-POP on SGE (mean utility −0.058935). The normalization hybrid is excluded from route ranking regardless of its numerical contrasts.
