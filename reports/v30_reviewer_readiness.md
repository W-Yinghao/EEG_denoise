# V30 reviewer readiness

| item | status | evidence |
|---|---|---|
| strong denoising baselines | complete | STANDARD, V25 DET and EEGDfus are on the common panel |
| subject-agnostic DDPM | complete | V26 PopSDEdit and V29 exact frozen population routes |
| RAW / STANDARD | complete | identical observation references under corrected standardized panel are explicitly labeled |
| subject component ablation | complete | MATCH/POP/all-wrong/lag/shuffle/mean-context |
| wrong/null/shuffled controls | complete | all 14 wrong donors plus falsification controls |
| statistics and CIs | complete | participant-first summaries and bootstrap intervals |
| support amount | complete | 0/5/10/30/120 seconds |
| sampler steps | complete | 5/10/25, K=1 |
| latency/memory | complete | batch 1/16, 20 warmup and 100 timed runs |
| additional dataset/montage plan | partial | natural SGE development exists; independent sealed confirmation remains unopened |
| privacy risk | complete | development linkage-risk diagnostic; no anonymity claim |
| support-setting limitation | complete | within-session fixed-montage query-disjoint support is explicit |
| task-valid physiology | missing | ERP/SSVEP/ERD metadata remain unavailable for a valid endpoint |
