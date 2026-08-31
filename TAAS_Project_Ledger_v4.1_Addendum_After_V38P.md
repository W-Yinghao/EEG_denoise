# TAAS-26-0171 项目总纲更新附录
## v4.1 — V38P 后的最终 diffusion 方法重置

> 本文件不是完整历史账本。服务器应将本节置于仓库现有 v4.0 ledger 顶部，并保留 v4.0
> 及更早内容不变。

## 1. V38P 最终判决

```text
final positioning:
C — non-diffusion transport preferable

route status:
closed
```

V38P 在 OpenBMI 54 名 participants、6 folds × 2 seeds 上完成全部 canonical 和 registered
repair cells。工程有效，但 SARD-Bridge 明确失败：

```text
RAW adaptive source BA:
0.333241

SARD canonical:
0.502037

SARD registered repair:
0.559259

participant-first privacy utility:
-0.168796
95% CI [-0.187963, -0.150461]
0/54 participants improved
```

Distribution：

```text
conditional energy distance

SARD:
0.504167

OneStep:
0.299691

Gaussian:
0.213304

Resample:
0.190002
```

K=8 augmentation 中，SARD 相对 OneStep 与 Gaussian 只有 `+0.000833` 和 `+0.003611`，
不足以抵消 privacy 与 distribution 的全面退化。

因此不得：

```text
继续调整 SARD-Bridge
恢复旧 Fiber-SANDiff
恢复 waveform SDEdit
在 OpenBMI 上新增 representation-transport architecture
```

## 2. V38P 失败的结构性解释

### 2.1 Source-conditioned shortcut

SARD 同时读取：

```text
source representation
source logits
source support prototype
```

并使用 residual transport：

\[
\widetilde z=z_s+\widehat\delta.
\]

该接口保留了高带宽 source-linked information。有限 adversarial loss不足以消除这一
shortcut；提高 adversary weight反而加重 outer-test linkage。

### 2.2 Train-subject adversary 与 unseen-subject privacy 不一致

训练 adversary识别 outer-train subject classes，而最终 adaptive attacker在未见 outer-test
participants上重新训练。抑制训练类标签不等于删除可迁移的 subject geometry，甚至可能将
其重编码为更易泛化的结构。

### 2.3 Dynamic donor residual仍被 point objective主导

虽然 donor是一对多的，当前 residual diffusion仍以单次动态 donor residual为训练目标；
Gaussian与empirical resampling更直接地逼近 donor population，结果全面更好。

### 2.4 Task-logit preservation留下巨大 identity自由度

保持 frozen task logits只约束task语义，并不限制其余128-d representation中的source
information。V38P没有找到一个有效的非线性source-removal transport。

## 3. Convergent evidence

当前已有三条独立负证据：

```text
waveform SDEdit:
无point或uncertainty增量

exact-fiber SANDiff:
输给Gaussian/Resample

SARD-Bridge:
privacy与distribution均输给非diffusion transport
```

因此：

> Learned diffusion as the direct inference-time cleaner or representation transporter is no longer
> an active route in this project.

这不是对 diffusion family 的一般否定，但继续围绕clean output或representation release进行
架构搜索已不再科学。

## 4. 唯一保留的 diffusion 角色

活动路线切换为：

```text
V39A — Calibration-Conditioned Artifact Diffusion Augmentation
```

Diffusion不再直接预测clean EEG或sanitized representation，而是学习：

\[
p_\theta(a\mid c,\text{artifact type},\text{severity}),
\]

即 query-disjoint calibration context下的一对多 artifact distribution。

生成artifact用于训练同一个 support-conditioned deterministic denoiser：

\[
\widetilde y=x+\widetilde a,
\qquad
\widehat x=f_\phi(\widetilde y,c).
\]

科学问题：

> Does diffusion-generated, calibration-conditioned corruption diversity improve held-out-user EEG
> denoising beyond empirical artifact resampling, conditional Gaussian synthesis, and WGAN-based
> artifact synthesis?

## 5. Subject-aware 定义

```text
subject-aware:
query-disjoint support summarizes the current corruption context

not identity-conditioned:
no subject ID or persistent biometric embedding

diffusion target:
artifact distribution, not neural identity or clean waveform
```

核心原则恢复为：

```text
personalize the corruption distribution, not the brain
```

## 6. V39A 范围

Primary evidence：

```text
paired semi-simulation with held-out participants
natural SGEYESUB attenuation/retention
artifact-distribution fidelity
same-denoiser augmentation utility
```

Primary generator comparison：

```text
empirical resampling
conditional Gaussian
conditional WGAN-GP
conditional diffusion
```

所有 generator产生相同数量的 pseudo-pairs，训练相同 deterministic denoiser。

只允许一次小型 diffusion repair。若 diffusion在distribution fidelity和denoising utility中均
不优于非diffusion generator，则：

```text
diffusion method search closes permanently for TAAS-26-0171
```

此后只允许：

```text
use an already validated diffusion baseline as a secondary comparator
or
prepare a non-diffusion/new-submission route
```

## 7. 与 companion debiasing 论文的边界

TAAS V39A拥有：

```text
artifact distribution
query-disjoint corruption support
diffusion augmentation
waveform denoising utility
```

Companion paper继续拥有：

```text
CMI/TOS
LEACE broad analysis
exact head-fiber theorem
Gaussian privacy channel
representation debiasing
```

V39A禁止运行CMI、TOS、foundation-model identity或exact-fiber实验。
