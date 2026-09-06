# V23 forensic audit of frozen V22

The V22 base stream assigned approximately 40% MATCH, 25% POP, 25% WRONG and 10% NO-CONTEXT while retaining the same artifact target. Therefore the ordinary loss explicitly encouraged context invariance; WRONG did receive base supervision. The ranking term was weighted by 0.1 and competed with the ordinary target loss.

All canonical checkpoints used [1200] updates. Validation used the first 64 arrays; diffusion validation used one fixed t=500 estimate. The file stored last weights plus a best scalar rather than a distinct best-weight checkpoint. The fixed generator used a hard 60%-quantile temporal mask. Zero-artifact examples entered the former SNR aggregation and are excluded in V23.

The frozen V22 results and reports were not modified.
