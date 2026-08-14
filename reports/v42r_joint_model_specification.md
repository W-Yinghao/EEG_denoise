# V42R joint clean-room model specification

V42R implements one joint 46-channel conditional diffusion model over 512-sample windows. At training step (t), the network receives the noisy clean state (x_t), the contaminated observation (y), the sinusoidal time embedding, and one 46-row transfer state. It predicts clean (x_0) through

\[
\widehat x_0 = y + \Delta_{\mathrm{pop}}(x_t,y,t) + \Delta_{\mathrm{transfer}}(x_t,y,t,c).
\]

Both output heads are zero-initialized, so the untrained model returns (y) exactly. The four-scale temporal U-Net uses widths 32/64/96/128, skip connections, and one four-head bottleneck attention block. The transfer encoder applies a shared per-channel MLP to the two bipolar-transfer coefficients, row norm, four support-quality features, and fixed one-hot sensor identity; it then applies one spatial mixing layer and mean/max global pooling. Transfer FiLM is confined to the small residual decoder.

The process uses 1,000 linear training steps, clean-(x_0) prediction, one registered deterministic 50-step DDIM trajectory, and common initial noise across POP/MATCH/WRONG/SHUFFLED/ORACLE/NO_TRANSFER_BRANCH. During training, transfer context is replaced by the outer-training population state with probability 0.20. Checkpoints are selected solely by participant-aggregated validation POP temporal RRMSE.

The canonical AMP pilot failed with a nonfinite gradient at update 1,905. The one registered engineering repair disables mixed precision. It does not change the architecture, target, split, transfer estimator, budget, sampler, or scientific metrics.

