# TAAS-26-0171 项目总纲更新附录
## v4.3 — V40R 后的 Artifact-Transfer Conditioning 路线

> 服务器应将本节置于仓库现有 v4.2 ledger 顶部，保留 v4.2 及更早 provenance 不变。

## 1. V40R 最终判决

```text
final positioning:
C — no support increment

closed hypothesis:
generic raw-support set encoder
+
two frozen FiLM adapters
on the local EEGDfus-MC port
```

V40R 没有证明：

```text
subject/context information is generally useless
support-conditioned EEG diffusion is generally ineffective
```

它证明的是：

> A compact mean-pooled EEG+EOG support embedding did not improve a weak local multichannel
> EEGDfus port.

## 2. 关键诊断

### 2.1 Population multichannel backbone itself was weak

```text
official-native single-channel EEGDfus:
temporal RRMSE 0.296527
correlation 0.953041

local EEGDfus-MC POP:
temporal RRMSE 1.192548
correlation 0.412462
SNR improvement -12.6853 dB
```

因此 V40R 不是在一个已验证的 multichannel denoiser 上测量 support 的小增量。

### 2.2 The port was a major implementation change

```text
channels:
1 -> 46

samples:
512 -> 256

sampling:
500-step ancestral -> 25-step DDIM

batch:
512 -> 16

learning rate:
1e-3 -> 2e-4
```

该模型保留了 EEGDfus 的骨架，但不能被视为官方性能在多通道上的直接延伸。

### 2.3 The support state was identity-linkable but artifact-misaligned

```text
10 s support:
approximately POP

30 s support:
worse than POP

MATCH better than WRONG on average:
yes, but heterogeneous

MATCH better than SHUFFLED:
no
```

Support state 能够链接 participant/context，但没有稳定编码对当前 query 去噪有用的 artifact
transfer。

### 2.4 Natural panel was not a valid support test

```text
POP remaining ratio:
4.010855

POP attenuation:
-5.3941 dB
```

Population backbone itself amplified the natural artifact proxy. Natural MATCH−POP therefore不能用来
判断 support conditioning 的一般价值。

## 3. 下一科学问题

```text
V41R — Calib-EEGDfus
Artifact-Transfer-Conditioned Official-Semantics EEG Diffusion
```

核心问题：

> Does an explicit query-disjoint estimate of EOG-to-EEG artifact transfer improve an otherwise
> official-semantics conditional EEG diffusion model on held-out participant-specific artifacts?

V41R 不再让神经网络从 raw support 自行发现“什么是有用的 subject information”。

Subject/context information被限制为：

```text
support-estimated EOG-to-EEG transfer row
support-only scale/quality summary
fixed sensor identity
```

## 4. 方法边界

```text
backbone:
official single-channel EEGDfus semantics

training/evaluation:
channel-wise shared model, reassembled to multichannel output

subject-aware condition:
explicit artifact-transfer signature

not used:
subject ID
persistent identity embedding
generic raw-support context vector
46-channel EEGDfus-MC port
posterior energy
routing
representation privacy
```

## 5. One-to-one ownership with related work

```text
EEGDfus:
establishes conditional EEG diffusion denoising

V41R increment:
query-disjoint unseen-participant artifact-transfer conditioning
```

Primary estimand:

```text
MATCH support-estimated transfer
minus
POP training-population transfer
```

Wrong, shuffled, and oracle transfer are mechanism controls.

Diffusion is not required to defeat every deterministic method.

## 6. Project state

```text
V40R:
frozen negative result

V41R:
active route

natural panel:
secondary and only interpretable after a valid population model

D4PM:
related work; official release not reproducible enough for the primary empirical table
```
