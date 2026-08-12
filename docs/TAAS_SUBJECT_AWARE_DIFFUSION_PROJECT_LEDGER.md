# TAAS-26-0171 项目总纲与证据账本
## Subject-Aware Diffusion for EEG Denoising

**文档性质：** 项目级权威记录、科学主线约束、分支与证据账本
**版本：** v1.3
**状态日期：** 2026-08-12（V25 结果审阅与 V26 路线同步）
**建议仓库路径：**

```text
docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md
```

---

# 0. 使用规则

本文件是该 TAAS 大修项目的唯一项目级总纲。后续所有服务器指令、执行报告审阅、方法改版、分支切换和稿件规划，都必须先以本文件为基准。

从本文件建立后，固定执行以下纪律：

1. **每次给服务器下达新指令前**
   - 先重新阅读本文件；
   - 更新“当前状态快照”；
   - 更新“证据阶梯”；
   - 更新“分支与 commit 账本”；
   - 更新“本轮科学问题”；
   - 更新“本轮允许与禁止事项”；
   - 再生成服务器指令。

2. **每次收到服务器执行报告后**
   - 先核验 Git、报告、代码、结果和作业 lineage；
   - 将结果写入“时间线与证据账本”；
   - 明确本轮关闭的只是哪个具体实现或假设；
   - 明确仍然存活的论文级主张；
   - 更新下一步；
   - 再给出新的服务器指令。

3. **每次对本文件的更新都必须**
   - 增加版本号和更新时间；
   - 保留历史，不覆盖既有判决；
   - 区分 historical decision、superseding interpretation 和 current status；
   - 区分 development、exploratory、confirmation；
   - 区分 engineering validity、subject-context value、diffusion value 和 natural-EEG validity。

4. **每个服务器分支都应提交本文件的同步副本**
   - 服务器在新 worktree 启动时读取本文件；
   - 在结果 commit 前更新本文件；
   - terminal report 必须列出本文件所在 commit；
   - 不允许只有聊天中的口头状态而仓库中没有同步记录。

5. **每次 ChatGPT 输出应至少包含**
   - 本轮对项目总纲的更新摘要；
   - 更新后的可下载 Markdown；
   - 对服务器报告的审阅或下一条服务器指令。

---

# 1. 外部约束与投稿身份

## 1.1 投稿身份

原投稿：

```text
Subject-Aware Diffusion Models for Cross-Subject EEG Denoising
TAAS-26-0171
```

当前任务不是创建一篇完全独立的新论文，而是完成该稿件的 major revision。

论文可以大改、方法可以推翻重做、novelty scope 可以显著收窄，但最终稿必须保留以下交集：

```text
subject-aware
+
diffusion
+
EEG denoising
```

允许将“subject-aware”重新定义为：

```text
unseen-subject / unseen-context
query-disjoint support calibration
corruption-specific context
```

不再要求保留原稿中的：

```text
closed-set subject embedding
dual decoder
ArcFace subject classifier
orthogonality loss
transductive target embedding
旧 SADDPM
旧 SADDPM-Cond
```

## 1.2 期刊决定

期刊决定为：

```text
MAJOR REVISION
```

修回截止日期：

```text
21-Nov-2026
```

编辑明确要求不能只做表面修改。主要审稿要求包括：

- 更强的 EEG denoising baselines；
- subject-agnostic DDPM baseline；
- raw 或 conventional preprocessing baseline；
- 对 subject component、FiLM、residual branch 和 loss 的系统消融；
- 统计显著性和置信区间；
- 关键超参数和 adaptation procedure 的敏感性；
- 明确 SADDPM 与 SADDPM-Cond 的输入、目标和推理差异；
- target data amount；
- sampling steps 与 latency；
- transductive setting 的限制；
- subject representation 的隐私风险；
- 至少一个额外数据集、montage 或 acquisition protocol。

## 1.3 与 AE 沟通后的项目许可

用户已经与 AE 沟通过：

```text
可以在保留 subject-aware diffusion for EEG denoising
这一核心主题的前提下进行大幅重构。
```

因此，当前不需要维护旧方法的表面连续性；需要维护的是论文问题与可审计的科学连续性。

---

# 2. 不可偏离的论文主线

## 2.1 当前主线

当前最窄、最可守住的主线是：

> 从 unseen subject 的 query-disjoint calibration support 中提取 ocular corruption context，并用该 context 调节一个 observation-anchored diffusion denoiser，以改善后续 EEG ocular-artifact removal。

当前首选任务范围：

```text
ocular artifact only
multichannel EEG
unseen subject / context
query-disjoint support
within-session primary
EEG-only query inference
support阶段允许同步EOG
```

允许进一步收窄到：

```text
固定 montage
固定 preprocessing
固定 support duration
within-session only
```

## 2.2 当前核心建模原则

```text
Personalize the corruption, not the brain.
```

含义：

- support 估计 ocular corruption mechanism；
- 不把 test participant identity 注入 neural prior；
- 不要求 learned biometric embedding；
- query 不更新整个模型；
- support 与 query 必须严格分离；
- participant-specific context 必须正面超过 strong population context；
- wrong context 只能证明 specificity，不能单独证明增量价值。

## 2.3 当前不应成为主线的内容

以下内容暂时不作为主方法：

```text
reliability routing
rollback policy
drift monitor
operator zoo
hard abstention system
safe deployment claim
large CSPD energy-bridge system
full posterior-sampling theory
identity / BrainID representation
pure deterministic denoising paper
pure negative-result paper
```

这些可以在基础方法成立后作为：

```text
optional refinement
diagnostic
appendix analysis
future work
```

---

# 3. 论文成立所需的证据阶梯

本项目不得把一个局部正结果直接升级为论文结论。证据必须按以下层次累计。

| 层级 | 科学问题 | 当前状态 |
|---|---|---|
| C0 | 工程骨架、数据隔离、baseline reproduction 是否有效 | **已建立** |
| C1 | query-disjoint support 是否包含 strong POP 之外的 participant/session corruption information | **V20 已建立** |
| C2 | learned denoiser 是否能把 support context 转化为超过 strong POP 的去噪收益 | **V25 在 paired development 上首次建立小而明确的增量；natural 与 confirmation 尚未建立** |
| C3 | diffusion 是否超过信息匹配的 deterministic estimator | **尚未建立；V25 当前 rank-8 latent residual diffusion 为 0/15 正向，必须移除该实现** |
| C4 | natural EEG 上是否同时获得 artifact attenuation 与 neural preservation | **尚未建立；V25 DET 与 DIFF 均未优于 strong POP，DIFF 明显恶化** |
| C5 | 冻结后的方法是否在 untouched sealed cohort 上复现 | **未运行** |
| C6 | 是否满足 reviewer 要求的 baseline、消融、统计、延迟、support burden 和隐私分析 | **部分完成，尚未形成最终稿证据包** |

## 3.1 C1：support information 已建立

V20 在 15 名 exchangeable recipients 上得到：

```text
N_P = +0.178803
relative improvement = 20.27%
15/15 positive
randomization p = 1/100001

N_W = +0.174162
relative improvement = 19.85%
15/15 positive
randomization p = 1/100001
```

它支持：

> early support contains participant/session-specific ocular-corruption information that predicts later natural query structure.

它不直接支持：

- EEG-only temporal amplitude 已可恢复；
- learned denoising 已改善；
- diffusion 已有效；
- natural cleaning 已成立。

## 3.2 C2：V25 首次建立 learned deterministic support value

V25 使用 raw query-disjoint support EEG+EOG set，通过 DeepSets support encoder 与 learned low-rank decoder得到：

```text
DET MATCH−POP:
+0.007148
95% bootstrap CI [0.001178, 0.013916]
10/15 participants positive

DET MATCH−WRONG:
+0.016731
95% bootstrap CI [0.007804, 0.026910]
14/15 participants positive
```

因此，当前可以安全地写为：

> A learned raw-support representation provides a small but reproducible paired-development increment over a strong population route.

这一结论仍受四个边界限制：

1. 绝对效应较小；
2. 只有 10/15 participants 在 MATCH−POP 上正向；
3. 证据来自 paired development，而非 sealed confirmation；
4. natural EEG 上 DET-MATCH 没有显示 population-relative artifact/preservation improvement。

所以 C2 当前状态是：

```text
paired development established
natural deployment not established
confirmation not run
```

## 3.3 C3：当前 diffusion implementation 已关闭

V25 的当前 diffusion：

```text
rank-8 learned-basis coefficient residual
full-noise x0 prediction
DDIM25
```

结果：

```text
DIFF−DET:
−0.057034
95% bootstrap CI [−0.064313, −0.050080]
0/15 participants positive

mild:
−0.067931

medium:
−0.052379

severe:
−0.039769
```

这不是 tail-only、seed-only 或个别 participant 问题。当前实现应从活动路线中移除。

但该结果不关闭 diffusion family，因为当前实现包含一个重要表示负担：

- residual target 定义在 learned support basis 的 coefficient coordinates；
- learned basis 仅做列归一化与 decorrelation，没有唯一的 sign/order/orientation contract；
- diffusion 网络只接收 context vector，没有显式接收 basis matrix；
- 因而同一 artifact field 可对应不同 latent coordinates；
- full-noise residual generation又会主动扰动一个已经较强的 deterministic estimate。

下一 diffusion 路线必须使用：

```text
coordinate-stable target
deterministic warm start
moderate-noise refinement
matched one-step refiner
```

## 3.4 C4：natural evidence 仍未建立

V25 committed natural result：

```text
DIFF MATCH−POP artifact utility:
−0.080127
1/15 positive

DIFF MATCH−POP preservation utility:
−0.129300
0/15 positive
```

从 committed method summary 可见，DET-MATCH 相对 POP 的自然变化也很小且方向不利：

```text
heldout EOG remaining ratio:
DET-MATCH ≈ 0.99758
POP ≈ 0.98964

preservation:
DET-MATCH ≈ 0.81585
POP ≈ 0.82829
```

因此，V25 的 paired support增量尚未转化为 natural validity。

---

# 4. 当前科学假设

## 4.1 V25 后仍然存活的主假设

> Raw query-disjoint support is a useful corruption context, but diffusion should refine a stable support-conditioned deterministic estimate in a coordinate-invariant signal space rather than regenerate a non-canonical learned latent from pure noise.

这比 V24/V25 之前的主张更窄：

```text
support value:
已有paired development证据

diffusion value:
尚未建立

natural value:
尚未建立
```

## 4.2 V25 后最可能的 diffusion failure mechanism

按证据优先级排序：

1. **Residual coordinate 不唯一。**
   - V25 先由 learned basis \(U_s\) 计算 ridge latent \(h^*\)；
   - diffusion target 是 \(h^*-\widehat h_{\rm det}\)；
   - 对任意正交旋转 \(R\)，\(U_sR\) 与 \(R^\top h\) 表示同一 artifact；
   - 当前 basis 没有 sign/permutation/orientation canonicalization。

2. **Diffusion 未显式观察 spatial basis。**
   - 网络接收 context token，但不直接接收 basis；
   - 它必须从 context 隐式恢复每个 episode 的 latent coordinate system。

3. **Full-noise generation 不适合小 residual refinement。**
   - inference 从纯 Gaussian residual state 开始；
   - deterministic estimate 已经较强；
   - 25-step reverse process可以引入远大于真实 residual 的变化。

4. **Latent loss主导 decoded physical error。**
   - base coefficient residual MSE是主损失；
   - decoded artifact loss权重仅为 0.1；
   - coordinate误差可被放大为自然 EEG distortion。

5. **Natural domain gap。**
   - paired development中support有小增量；
   - natural中support和diffusion均未改善artifact/preservation。

## 4.3 当前下一方法

V26：

```text
CalibSDEdit
Deterministic-Anchored
Support-Conditioned Artifact Diffusion Refinement
```

保留 V25 已验证的：

```text
raw support-set encoder
SetCalibDET
strong population anchor
MATCH / POP / WRONG interventions
```

删除活动路线中的：

```text
rank-8 latent residual diffusion
```

主 diffusion 改为在固定 EEG sensor coordinates 中进行 moderate-noise artifact refinement。

设：

\[
\widehat A_{\rm det}=f_{\rm det}(Y,S),
\qquad
\widehat X_{\rm det}=Y-\widehat A_{\rm det},
\]

\[
\Delta_s=
\widehat A_{\rm det}
-
\widehat A_{\rm pop}.
\]

训练目标为真实 artifact \(A\)。对有限噪声水平 \(t\)：

\[
A_t
=
\sqrt{\bar\alpha_t}A
+
\sqrt{1-\bar\alpha_t}\epsilon.
\]

Refiner：

\[
\widehat A_0
=
F_\theta
\left(
A_t,
Y,
\widehat A_{\rm det},
\widehat X_{\rm det},
\Delta_s,
c_s,
t
\right).
\]

推理不从纯噪声开始，而从 deterministic artifact estimate 的 noised state开始：

\[
A_{t_0}^{\rm init}
=
\sqrt{\bar\alpha_{t_0}}\widehat A_{\rm det}
+
\sqrt{1-\bar\alpha_{t_0}}\epsilon.
\]

随后运行短程 DDIM：

\[
A_{t_0}^{\rm init}
\rightarrow
\widehat A_{\rm diff},
\qquad
\widehat X_{\rm diff}
=
Y-\widehat A_{\rm diff}.
\]

V26 同时训练一个输入、容量和监督尽量匹配的 one-step refiner，以分离：

```text
第二阶段refinement value
vs
iterative diffusion value
```

---

# 5. 当前方法设计边界

## 5.1 V26 必须保留

```text
V25 raw support-set encoder
V25 deterministic MATCH increment
strong independent population anchor
query-disjoint support
EEG-only query inference
same participant folds
same corrected coordinate/data pipeline
same paired and natural-reference streams
same-checkpoint MATCH / WRONG mechanism swaps
independent strong population refiner
matched one-step second-stage refiner
K1 diffusion primary
participant-first aggregation
```

## 5.2 V26 主动删除

```text
V25 rank-8 latent residual diffusion
pure-noise residual initialization
basis-dependent diffusion target
K8 rescue
complex routing
rollback
operator zoo
```

代码可保留为 historical implementation，但不得进入 active result table。

## 5.3 V26 暂不加入

```text
energy bridge
posterior guidance
DPS / DDRM / DDNM
Drifting Models
LoRA / hypernetwork zoo
subject-ID classifier
sealed confirmation
manuscript result insertion
```

## 5.4 开发模式

V26 继续采用：

```text
GPU-first exploratory science
few hard engineering gates
human review between rounds
```

硬边界只用于：

```text
leakage
coordinate mismatch
support/query overlap
query auxiliary leakage
sealed read
nonfinite/scale collapse
checkpoint/resume
participant aggregation
provenance
```

以下是开发信息，不是自动永久关闭条件：

```text
某一noise strength较差
某一fold反向
某个CI跨零
第一轮natural结果混合
```

## 5.5 Energy Bridge 的开放条件

Energy bridge 仍然后置。只有以下大部分成立后才开放：

1. CalibSDEdit K1 至少接近 matched one-step refiner；
2. subject-context value保持；
3. natural artifact reduction出现正向；
4. 主要剩余问题是artifact–neural overlap或preservation；
5. 基础训练和坐标均无疑问。

# 6. 时间线与证据账本

## 6.1 原投稿阶段

### 原稿

```text
Subject-Aware Diffusion Models for Cross-Subject EEG Denoising
```

原稿包含：

```text
SADDPM
+
SADDPM-Cond
```

问题：

- 两个方法割裂；
- SADDPM 为合作者方法，用户无法独立复现；
- 存在潜在研究诚信风险；
- SADDPM-Cond 主要只能与 CNN 比较；
- subject embedding 与真正 artifact mechanism 混杂；
- 缺少 strong modern baselines、subject-agnostic DDPM、消融、统计和延迟分析。

项目决定：

```text
删除旧 SADDPM 证据链
保留 subject-aware diffusion topic
clean-room 重建方法
```

---

## 6.2 早期 clean-room diffusion 尝试

### Commit 3ad4856

结论：

```text
Conditional DDIM100 output-scale collapse
current_M2_no_incremental_value
diffusion_family_not_tested
```

要点：

- 输出幅值爆炸；
- matched U-Net 正常；
- 当前 pure-noise full EEG generation 无效；
- 不构成 diffusion family negative。

### Commit 08d6bfe...

结论：

```text
artifact-latent primary validity failed
backup SDEdit validity failed
topic not yet testable
```

### Commit bf5bd86...

结论：

```text
deterministic validity passed
proxy improvement without mechanistic support
diffusion reopen not eligible
```

### Commit 8aca035...

结论：

```text
residual diffusion slightly improved Klados semi-simulation
no subject-awareness on SGEYESUB
natural preservation failed
```

---

## 6.3 项目级科学审阅与 gate reset

项目审阅后建立顺序：

```text
construct/headroom
→ analytic bridge
→ deterministic model
→ diffusion
→ confirmation
```

审阅要求：

- participant identity 不等于 denoising value；
- MATCH 必须超过 strong POP；
- MATCH > WRONG 只能算 specificity；
- operator-swap 必须固定同一 x/e/y；
- 先自然 transfer，再模型；
- sealed data 保持关闭。

---

## 6.4 BrainID 与 PhysioTrait

### BrainID Gate-01 / 01R

时间：

```text
项目审阅前
```

结果：

```text
identity-related signal present
nuisance-invariant physiological construct not established
```

### PhysioTrait v18

Branch：

```text
codex/physiotrait-actionability-v18
```

Commit：

```text
55b334e21aa7882164056ccaafd5598178dea0b6
```

结果：

```text
H_P = +0.8198
H_W = +1.0504
```

但 TIME_SHUFFLE 后 donor separation 仍在。

判决：

```text
CROSS_DAY_PHYSIOTRAIT_HEADROOM_NO_GO

TRAIT HEADROOM PRESENT,
BUT PHYSIOLOGICAL TEMPORAL CONSTRUCT NOT IDENTIFIED
```

---

## 6.5 Counterfactual operator 路线

### V19

Branch：

```text
codex/counterfactual-operator-headroom-v19
```

Commit：

```text
5ab1918ceebf3b9622ceb2806a274edd01205e8b
```

O0-B controlled ceiling：

```text
H_P = +0.7206
H_W = +0.6621
15/16 positive
```

O0-A natural：

```text
N_P = +0.1676
N_W = +0.1633
15/16 positive
```

原判决：

```text
SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO
```

但该判决依赖 0.2293 null floor。

### V19 provenance audit

Branch：

```text
codex/counterfactual-operator-v19-provenance-audit
```

Commit：

```text
166706f535c01fca997cc89e0018afef2d8fe414
```

结果：

```text
INVALID_NO_EXACT_RECOVERY

NULL_FLOOR_PROTOCOL_OR_PROVENANCE_INVALID_ROUTE_CLOSED
```

解释：

- 数值可精确复现；
- 0.2293 不是有效定义的 permutation / joint max-stat null；
- 不存在唯一 exact recovery；
- V19 natural gate 不可裁决；
- 关闭的是 V19 null-floor route，不是 operator-transfer hypothesis。

---

## 6.6 V20 — valid natural transfer gate

Branch：

```text
codex/calibration-transfer-randomization-v20
```

Commit：

```text
befb1f17093872e254e1aac254536d29e101f35b
```

结果：

```text
V20_NATURAL_TRANSFER_PASS

NATURAL_CALIBRATION_TRANSFER_ESTABLISHED

O1_AUTHORIZED_NOT_RUN
```

Primary：

```text
N_P = +0.178803
N_W = +0.174162
15/15 positive
p = 1/100001
```

意义：

```text
support operator具有自然support→query transfer信息
```

不代表：

```text
EEG-only temporal coefficient可恢复
denoising有效
diffusion有效
```

---

## 6.7 O1-V21 — analytic EEG-only bridge

Branch：

```text
codex/eeg-only-analytic-bridge-o1-v21
```

Commit：

```text
c9eeecb24c2f0aa69aa60b69b1d9fd361fc179cc
```

结果：

```text
O1_EEG_ONLY_AMPLITUDE_NOT_IDENTIFIED

SPATIAL_TRANSFER_VALID_EEG_ONLY_TEMPORAL_COEFFICIENT_NOT_IDENTIFIED
```

Primary：

```text
U_P = -0.059566
U_W = -0.094335
```

Detector：

```text
AUROC = 0.857252
TAR@FAR5 = 0.063060
```

意义：

```text
空间transfer有效
当前解析EEG-only amplitude solver无效
```

它不否定：

```text
nonlinear temporal network
end-to-end conditional model
diffusion
```

---

## 6.8 V22 — SCAD project reset

Branch：

```text
codex/scad-support-conditioned-artifact-diffusion-v22
```

Commit：

```text
2c5b7bf4b5daf667f345ecb6e5f32495d494dfe1
```

结果：

```text
Engineering:
valid

EEGDfus:
local_results_reasonable_but_nonidentical

D4PM:
blocked_incomplete_release

Subject context:
weak_or_heterogeneous_signal

Diffusion:
deterministic_better

Natural:
preservation_concern
```

关键：

```text
SCAD MATCH−POP = -0.00133
SCAD MATCH−WRONG = -0.00047
SCAD-K1−DET1 = -0.07761
```

后续审计发现：

- ordinary base loss 同时训练 MATCH/POP/WRONG 拟合同一 target；
- ranking loss太弱；
- context objective自相矛盾；
- 1200 updates明显不足；
- fixed pair与hard mask存在限制。

Superseding interpretation：

```text
V22不构成context family negative
它主要关闭了conflicting-objective full-field FiLM SCAD
```

---

## 6.9 V23 — OF-SCAD

Branch：

```text
codex/of-scad-operator-factorized-v23
```

Commit：

```text
ad9614c84e88bb0f89023e284e00d311255c019b
```

方法：

```text
context-consistent online counterfactual training
operator-factorized decoder
deterministic coefficient anchor
low-dimensional coefficient-residual diffusion
```

结果：

```text
Engineering:
valid

V22 repair:
objective_repair_helped

Operator-factorized context:
weak_or_heterogeneous_signal

Diffusion:
deterministic_better

Natural:
artifact_reduction_insufficient
```

Projection ceiling：

```text
POP = 0.490055
MATCH = 0.296117
WRONG = 0.410930
QUERY_ORACLE = 0.222561
```

同 checkpoint context swap：

```text
OF-DET MATCH−POP-SWAP = +0.057644
OF-DET MATCH−WRONG-SWAP = +0.008791

OF-SCAD MATCH−POP-SWAP = +0.167918
OF-SCAD MATCH−WRONG-SWAP = +0.045371
```

相对独立 strong POP：

```text
OF-DET subject utility = -0.092447
OF-SCAD subject utility = -0.092598
```

Diffusion：

```text
SCAD-K1−DET1 = -0.003304
6/15 positive
```

Natural：

```text
attenuation utility = -0.421162
preservation utility = -0.227200
```

当前解释：

> operator geometry is informative, but basis-dependent temporal coefficients remain difficult to estimate, and the learned subject route does not beat a strong population model.

---

## 6.10 V24 — PA-EL-SCAD

Branch：

```text
codex/pa-el-scad-eog-latent-v24
```

Terminal commit：

```text
8dadb508fd2d50a089246c4e11c83b7b7628fa42
```

Coordinate audit：

```text
V23_COORDINATE_MISMATCH_CONFIRMED
```

结果：

```text
correct raw/canonical max relative error:
2.57e-16

V23 committed median relative error:
1.1734
```

V23 结果保持不可变，但其 scientific interpretation 被降级为：

```text
historical development result under invalid coordinate construction
```

V24 corrected results：

```text
Natural DET EOG-latent correlation:
0.6102

DET MATCH−POP:
−0.04191

DET MATCH−WRONG:
+0.00348

SCAD MATCH−POP:
−0.20230

SCAD MATCH−WRONG:
+0.02190

SCAD-K1−DET1:
−0.16040
0/15 positive
```

Natural：

```text
POP preservation:
0.8283

DET-MATCH preservation:
0.7828

SCAD-MATCH preservation:
0.5961

SCAD-MATCH remaining ratio:
1.3635
```

判决：

```text
coordinate:
mismatch_confirmed_and_repaired

EOG latent:
moderate_predictability

subject correction:
context_harmful

diffusion:
deterministic_better

natural:
artifact_reduction_insufficient
```

解释：

> A shared EOG temporal latent is moderately predictable, but forcing a fixed support-operator deviation onto a learned population anchor is harmful. The analytic deviation is not an adequate support representation, and residual diffusion amplifies rather than repairs the deterministic error.

下一路线：

```text
Replace fixed operator with raw support-set encoder.
```

工程：

```text
19/19 targeted tests
19/19 clean archive tests
sealed reads = 0
manuscript unchanged
```

## 6.11 V25 — SetCalibDiff

Branch：

```text
codex/setcalibdiff-raw-support-v25
```

Base：

```text
8dadb508fd2d50a089246c4e11c83b7b7628fa42
```

Terminal commit：

```text
a7d9d647b69e152255b62dbca917a4b3ed082915
```

Key commits：

```text
implementation:
13356f9e4454627c00ed7170fe2293e394410ee9

Round A:
fe8c94e7cd582032c713365f7c82217454377bb3

Round B:
3d92fa8...

result-producing:
5d014c6...

ledger v1.2:
b4948537123535ff46acdfb190fd3e6725fe3040

report packaging:
a77e9a970fd192d0fe74f27669d29636b60f3d87
```

Method：

```text
raw support EEG+EOG
DeepSets support encoder
learned low-rank spatial basis
support-conditioned deterministic coefficients
rank-8 coefficient-residual diffusion
```

Paired development：

```text
DET MATCH−POP:
+0.007148
CI [0.001178, 0.013916]
10/15 positive

DET MATCH−WRONG:
+0.016731
CI [0.007804, 0.026910]
14/15 positive
```

Diffusion：

```text
DIFF−DET:
−0.057034
CI [−0.064313, −0.050080]
0/15 positive

mild / medium / severe:
all negative
```

Natural：

```text
DIFF artifact utility:
−0.080127

DIFF preservation utility:
−0.129300
```

判决：

```text
Engineering:
valid

Raw support:
clear_development_signal

Strong population:
support_better on paired development

Diffusion:
deterministic_better

Natural:
artifact_reduction_insufficient
```

解释：

> V25 is the first clean-room learned model to establish a small paired-development benefit from raw query-disjoint support over a strong population route. The current learned-basis residual diffusion is uniformly harmful and must be retired. Natural EEG validity remains unestablished.

工程：

```text
23/23 targeted tests
23/23 clean archive tests
sealed reads = 0
query auxiliary inference reads = 0
A-track unchanged
manuscript unchanged
```

## 6.12 V26 — 当前计划，尚未执行

计划 branch：

```text
codex/calib-sdedit-v26
```

Base：

```text
a7d9d647b69e152255b62dbca917a4b3ed082915
```

当前任务：

1. 同步项目账本 v1.3；
2. 冻结 V25 DET 与 support encoder作为subject-aware anchor；
3. 审计 learned-basis residual coordinate non-identifiability；
4. 删除当前 latent residual diffusion活动路线；
5. 实现 sensor-coordinate one-step artifact refiner；
6. 实现 deterministic-warm-start artifact SDEdit；
7. 单独训练 strong population refiner；
8. Round A 小规模探索 noise strength 与 steps；
9. Round B 五折三seed；
10. paired + natural development；
11. K1 primary，不运行 K8；
12. sealed保持关闭。

# 7. 分支与 commit 账本

| 阶段 | Branch / Commit | 状态 | 作用 |
|---|---|---|---|
| A-track | `0c4f2301...` | 只读 | 历史 clean-room 理论稿 |
| Early diffusion | `3ad4856` | 冻结 | full-EEG DDIM scale failure |
| Artifact latent | `08d6bfe...` | 冻结 | validity failure |
| Deterministic screen | `bf5bd86...` | 冻结 | proxy improvement without mechanism |
| Residual diffusion | `8aca035...` | 冻结 | weak semi-sim signal, no natural support |
| PhysioTrait v18 | `55b334e...` | 冻结 | trait construct failed |
| V19 | `5ab1918...` | 冻结 | strong descriptive operator signal |
| V19 audit | `166706f...` | 冻结 | null protocol invalid |
| V20 | `befb1f1...` | 权威正证据 | natural support→query transfer established |
| O1-V21 | `c9eeecb...` | 冻结 | analytic temporal amplitude not identified |
| V22 | `2c5b7bf...` | 冻结 | SCAD engineering reset |
| V23 | `ad9614c...` | 历史开发；坐标失效 | superseded by V24 coordinate audit |
| V24 | `8dadb50...` | 冻结 | coordinate repaired; fixed operator deviation harmful |
| V25 | `a7d9d647...` | 当前最新完成 | raw support DET signal; latent diffusion uniformly harmful |
| V26 | planned | 当前活动路线 | deterministic-anchored support-conditioned artifact SDEdit |

---

# 8. 数据与证据角色

## 8.1 Counterfactual paired multichannel data

用途：

- clean waveform fidelity；
- operator intervention；
- MATCH / POP / WRONG；
- deterministic vs diffusion；
- SNR stratification；
- artifact field and latent metrics。

限制：

- 属于 semi-simulation；
- 不能证明 natural physiological validity；
- source roles、operator roles、EOG roles必须分离；
- generating operator不能直接等于support operator。

## 8.2 SGEYESUB natural development

用途：

- natural ocular attenuation；
- EEG–EOG association；
- frontal residual；
- low-artifact preservation；
- PSD；
- covariance；
- ERP / SSVEP proxy。

限制：

- 无 clean counterfactual；
- 不能用 RRMSE 声称真实 clean recovery；
- attenuation必须与preservation同时报告。

## 8.3 EEGdenoiseNet

用途：

- EEGDfus reproduction；
- generic EOG / EMG benchmark；
- low-SNR stress；
- reviewer baseline。

限制：

- 单通道source segments；
- 无 participant identity；
- 不承担subject-aware主张。

## 8.4 Sealed data

当前必须保持：

```text
Mobile sealed-8
PhysioMotion sealed-10
SHU Day-4/5
PhysioTrait Day-200
其他confirmation payload
```

状态：

```text
unopened
```

只有模型、support budget、sampler、metrics和统计方案冻结后才允许打开。

---

# 9. 永久实验纪律

## 9.1 强 population comparator

任何 subject-aware claim 必须比较：

```text
MATCH
vs
independent strong POP
```

不能只比较：

```text
MATCH
vs
WRONG
```

## 9.2 Subject value 与 diffusion value必须分开

Subject value：

\[
M(\mathrm{MATCH})-M(\mathrm{POP}).
\]

Diffusion value：

\[
M(\mathrm{DIFF\text{-}K1})-M(\mathrm{DET1}).
\]

如使用 K=8：

\[
M(\mathrm{DIFF\text{-}K8})-M(\mathrm{DET8}).
\]

禁止：

```text
DIFF-K8 vs DET1
```

作为 diffusion-specific evidence。

## 9.3 Wrong context 的角色

WRONG 是：

```text
specificity / intervention negative control
```

不是：

```text
strong population baseline
```

## 9.4 Development 与 confirmation

Development允许：

- GPU端到端训练；
- 查看曲线；
- 调 learning rate；
- 调 architecture；
- 调 loss；
- 调 steps；
- 选择 checkpoint；
- 迭代方法。

Confirmation前才冻结：

- architecture；
- support budget；
- sampler；
- K；
- metrics；
- statistics；
- sealed participants。

## 9.5 硬门只保留给工程与治理

开发阶段硬门只用于：

```text
data leakage
coordinate inconsistency
nonfinite loss/gradient
output-scale collapse
checkpoint/resume错误
split错误
query auxiliary leakage
sealed read
manifest/provenance failure
```

不再为每一个开发实验设置过多僵硬科学阈值。

## 9.6 Participant-first

```text
window和seed不是独立生物样本
```

必须：

- window先在participant内聚合；
- seed先在participant内聚合；
- biological n为participant或participant/session；
- bootstrap以participant为cluster。

## 9.7 Query 边界

Primary EEG-only inference：

```text
query EEG only
```

Query EOG、query operator、event label：

```text
output freeze后 evaluator-only
```

## 9.8 负结果保存

任何失败：

- 保留原输出；
- 标记 failed / superseded / recovery；
- 不覆盖；
- 不通过换指标或删participant救回；
- 只关闭具体实现。

---

# 10. Reviewer response 对照表

| Reviewer/AE 要求 | 当前项目规划 |
|---|---|
| Strong denoising baselines | EEGDfus、D4PM audit、strong U-Net、population anchor |
| Subject-agnostic DDPM | POP-MARGINAL-SCAD / V24 population diffusion baseline |
| Raw / standard preprocessing | 后续最终实验必须保留 |
| Subject mechanism ablation | MATCH / POP / WRONG，operator deviation，support duration |
| FiLM / residual / loss ablation | 旧原稿方法删除；新方法做context、deterministic、diffusion分解 |
| Statistics | participant-first bootstrap / paired tests |
| t0 / K / sampling steps | 10 / 25 / 50 steps，K1/K8与DET8 |
| Target data amount | support 0/5/10/30/60/120 s，按数据允许 |
| Latency | NFE、window latency、GPU memory、quality–latency |
| Another dataset | paired development + natural SGE + sealed confirmation |
| Privacy | support operator / context state，不再是subject embedding；最终仍需审计 |
| SADDPM vs SADDPM-Cond inconsistency | 删除双方法，统一为一个主方法 |
| EMG underperformance | EMG只作population stress，不扩展personalized claim |
| Downstream classifier | final revision需补固定与retrained decoder |
| Transductive limitation | 新方法改为query-disjoint support calibration，不做query optimization |

---

# 11. 当前风险清单

## 高风险

1. V23 coordinate contract可能错误；
2. EOG latent可能难以从EEG-only query稳定预测；
3. strong population model可能已经吸收绝大多数可利用信息；
4. natural domain gap可能使semi-sim正结果无法转化；
5. diffusion可能始终不超过deterministic；
6.修改稿时间窗口有限。

## 中风险

1. support operator deviation对natural query漂移；
2. EOG latent teacher包含neural–EOG correlation；
3. natural evaluator无clean ground truth；
4. reviewers可能质疑方法改动过大；
5. sealed confirmation样本量有限。

## 可控风险

1. old SADDPM不可复现；
2. D4PM release不完整；
3.复杂routing偏离主线；
4.阈值过度冻结；
5.K8 averaging伪装diffusion gain。

---

# 12. 当前下一步

当前已完成：

```text
V25 SetCalibDiff
```

当前人工判决：

1. 保留 raw support-set encoder 与 SetCalibDET；
2. 将 paired MATCH−POP 的小增量登记为 development milestone；
3. 明确 natural validity 尚未建立；
4. 移除 V25 rank-8 latent residual diffusion 的 active status；
5. 不使用 K8、uncertainty 或 severity subgroup救回该实现；
6. 不打开 sealed confirmation；
7. 进入 V26 CalibSDEdit。

V26 执行顺序：

1. 同步项目账本 v1.3；
2. 复用 V25 folds、support episodes、population anchor和DET checkpoints；
3. 运行 learned-basis rotation/sign/permutation诊断；
4. 建立 matched one-step artifact refiner；
5. 建立 deterministic-warm-start artifact SDEdit；
6. 训练独立 population refiner；
7. Round A：2 folds × 1 seed，小规模选择noise strength与steps；
8. Round B：5 folds × 3 seeds；
9. paired participant-first evaluation；
10. natural SGE development evaluation；
11. 更新项目账本。

开发原则：

```text
不使用大量科学gate。
工程正确性失败才停止。
科学选择由effect size、participant heterogeneity、
paired/natural一致性和计算成本综合决定。
```

如果 V26 仍然不能提供 diffusion value，后续优先级：

```text
A. 检验diffusion的proper-score/uncertainty贡献
B. 在DET保持主体结果的前提下尝试轻量energy refinement
C. 使用更标准的clean-signal conditional diffusion bridge
D. 与AE再次确认最终稿能否将diffusion降为辅助/ablation
```

不应立即：

```text
打开sealed data
恢复复杂routing
同时搜索多个operator family
用K8掩盖K1
回到旧SADDPM
```

# 13. 每轮更新模板

每次服务器指令或服务器报告后，在本文件末尾新增一条记录。

```markdown
## YYYY-MM-DD — VXX / 事件名称

### Trigger

- 新服务器指令 / 执行报告 / 审阅 / 方法重构

### Repository

- Base branch:
- Base commit:
- Active branch:
- Implementation commit:
- Result commit:
- Terminal commit:

### Scientific question

- 本轮只回答什么问题？

### Method change

- 新增：
- 删除：
- 冻结：
- 未运行：

### Data boundary

- Development:
- Evaluator-only:
- Sealed:
- Query auxiliary reads:

### Engineering result

- Validity:
- Tests:
- Failed/recovery jobs:
- Current jobs:

### Scientific result

- Subject-context:
- Strong POP comparison:
- Diffusion vs DET:
- Natural attenuation:
- Natural preservation:
- Uncertainty / tail:
- Latency:

### Interpretation

- 本轮证明了什么？
- 本轮没有证明什么？
- 关闭了哪个具体假设？
- 哪个论文级假设仍然存活？

### Mainline impact

- 主线保持 / 收窄 / 改写：
- Novelty scope变化：
- Reviewer response变化：
- Manuscript impact：

### Next action

- 下一条唯一主路线：
- 暂不开放：
```

---

# 14. 每次服务器指令的必备开头

以后每条 server instruction 应以以下纪律开头：

```text
启动本轮前，先读取并核验：

docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md

本轮必须在终端结果提交前同步更新该文件。

若执行计划与该文件中的论文主线、证据阶梯、
当前活动路线或sealed边界冲突，必须fail-closed并报告，
不得自行扩大方法范围。
```

---

# 15. 每次服务器报告的审阅清单

收到报告后必须检查：

## Git

```text
base
ancestry
implementation
result
packaging
terminal
remote/local parity
clean tree
```

## Slurm

```text
accepted
failed
cancelled
superseded
recovery
current queue
```

## Engineering

```text
tests
clean archive
coordinate
scale
checkpoint
resume
RNG
inference/evaluator boundary
```

## Science

```text
strong POP
MATCH
WRONG
DET
DIFF
paired
natural
participant-first
seed-first
latency
```

## Governance

```text
sealed reads
A-track
manuscript
old SADDPM
third-party license
```

## Project ledger

```text
current status updated
timeline updated
branch ledger updated
next action updated
version incremented
```

---

# 16. 当前状态快照

**当前最新完成阶段：**

```text
V25 SetCalibDiff
```

**最新 terminal commit：**

```text
a7d9d647b69e152255b62dbca917a4b3ed082915
```

**Canonical remote branch：**

```text
origin/codex/setcalibdiff-raw-support-v25
```

**当前主要科学事实：**

```text
1. V20建立了support→query corruption information。
2. V24确认并修复了V23 coordinate mismatch。
3. V25 raw support-set DET首次在paired development中超过strong POP。
4. 该subject增量较小：+0.007148，10/15正向。
5. V25 learned-basis residual diffusion为−0.057034，0/15正向。
6. diffusion在mild/medium/severe三个strata均为负。
7. natural artifact与preservation均未成立。
8. 当前最合理的diffusion问题是：
   能否从一个已经有用的deterministic estimate进行moderate-noise refinement，
   而不是从纯噪声生成非canonical latent residual。
```

**当前活动路线：**

```text
V26 CalibSDEdit
deterministic-anchored
support-conditioned artifact diffusion refinement
```

**当前下一问题：**

> Can a moderate-noise, deterministic-warm-start diffusion refiner improve a raw-support-conditioned artifact estimate beyond a matched one-step refiner while preserving the small subject-context benefit and improving natural attenuation–preservation trade-off?

**当前不可打开：**

```text
sealed confirmation
manuscript result insertion
large energy-bridge system
reliability routing
operator zoo
K8 rescue
```

**当前论文主线：**

```text
query-disjoint support-conditioned
subject-aware diffusion
for unseen-subject ocular EEG denoising
```

**当前开发哲学：**

```text
GPU-first exploratory science
few hard engineering gates
human review between rounds
```

# 17. 版本记录

## v1.3 — 2026-08-12

同步 V25 并规划 V26：

- 记录 raw support DET 的首个 paired-development strong-POP increment；
- 将 C2 更新为“paired development established，natural/confirmation unresolved”；
- 冻结 V25 diffusion 的 0/15 participant failure；
- 记录所有 severity strata 均不支持 diffusion；
- 记录 natural artifact/preservation 双重负向；
- 识别 learned-basis latent residual 的 coordinate non-identifiability；
- 移除当前 residual diffusion implementation；
- 将活动路线切换为 deterministic-warm-start artifact SDEdit；
- 保持 K1 primary、K8关闭；
- 保持 sealed data关闭；
- 继续采用少硬门、GPU-first科研开发模式。

## v1.2 — 2026-08-12

服务器 V25 分支中的中间版本：

- 同步 V25 implementation、Round A/B 和 natural result；
- 记录 raw support clear development signal；
- 记录 current residual diffusion deterministic_better；
- 暂停自动 successor，等待人工审阅。

## v1.1 — 2026-08-12

同步 V24：

- 确认并修复 V23 coordinate mismatch；
- 将 V23 科学结果降级为 invalid-coordinate historical development；
- 记录 V24 EOG latent moderate predictability；
- 记录 fixed operator deviation context harmful；
- 记录 residual diffusion 0/15 participant 正向；
- 将活动路线切换为 V25 raw support-set encoder；
- 明确开发阶段减少硬科学 gate，采用 GPU-first exploratory workflow；
- 硬边界仅保留给泄漏、坐标、数值、checkpoint、sealed 和 provenance。

## v1.0 — 2026-08-12

建立项目级总纲，整合：

- 原 TAAS 投稿身份；
- major revision 要求；
- old SADDPM 删除决定；
- early diffusion pilots；
- BrainID / PhysioTrait；
- V19 provenance；
- V20 transfer pass；
- O1 temporal failure；
- V22 SCAD reset；
- V23 OF-SCAD；
- V24 coordinate audit + EOG-latent plan；
- 后续强制同步更新纪律。
