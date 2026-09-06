# V9 common-randomness audit

AdaptationReplay freezes zero-output LoRA initialization, minibatches, diffusion timesteps, Gaussian noise, dropout RNG, checkpoint steps, and K=8 inference noise. D01/D11 replay the same schedule; donor adapters are trained once per fold/seed and reused. Rank masks are applied to targets, loss, and correction. Query inference sees deployable EEG/condition IDs only.

The first replay (job 928921) was invalidated before evaluation because repeated GPU adaptation and checkpoint reload were not bit-reproducible. Jobs 928959/928962/928964 localized cuDNN/CuBLAS and train/eval-mode causes. Job 928968 passed all real-record replay checks. The scientific replay 928971 used deterministic CUDA kernels and the revisioned `exact_replay_r2_deterministic_cuda` result root; invalid outputs remain separate.
