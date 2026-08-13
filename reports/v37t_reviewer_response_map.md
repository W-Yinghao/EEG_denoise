# V37T reviewer-response map

| Requirement | Evidence and proposed response | Status |
|---|---|---|
| Strong baselines | Common panel reports EEGDfus, matched deterministic EnergyDET, V25 DET, and STANDARD. No compatible registered strong U-Net is approximated. | complete with explicit U-Net limitation |
| Subject-agnostic diffusion | V26 PopSDEdit shares the frozen SDEdit mechanism without support. | complete |
| RAW / STANDARD | Both are retained in absolute common-panel tables. | complete |
| Subject/support ablation | MATCH versus population, mean wrong, all donors, lagged/shuffled support, and exact duration are reported. | complete, specificity heterogeneous |
| Statistics | Participant-first means, medians, positive counts, bootstrap intervals, fold and seed effects are inherited; K=16 uncertainty is also participant-first. | complete |
| Support amount | V31 exact, prefix-only 0/5/10/30/120-second curve supersedes invalid V30 duration rows. | complete |
| SADDPM / SADDPM-Cond reconciliation | Historical SADDPM assets are not imported. CalibEnergy-SDEdit is framed conceptually as support-conditioned artifact SDEdit with deterministic warm start and a closed-form final energy. | complete without historical-result reuse |
| Stochastic value | K=16 coverage, interval score/CRPS, error–dispersion association, support-span/complement variance, identity uncertainty, and mean/median point summaries. | complete in V37T |
| Privacy risk | V30 linkage audit is acknowledged as a limitation; the exact-fiber privacy line is owned by a separate paper. | complete |
| Physiological preservation | Low-EOG retention is not renamed as physiology; ERP/SSVEP claims remain unavailable. | claim withdrawn |
| Latency | Only previously frozen latency evidence may be cited; V37T performs no new benchmark and does not select by latency. | secondary |

The response should emphasize an attenuation-oriented, development-stage operating point and mixed
support specificity. It must not claim unique operator recovery, physiological ground truth, formal
safety, or diffusion superiority on every point metric.

