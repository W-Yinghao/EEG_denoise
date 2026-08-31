# TAAS-26-0171 项目总纲与证据账本
## Subject-Aware Diffusion for EEG Denoising

**文档性质：** 项目级权威记录、科学主线约束、分支与证据账本  
**版本：** v2.1  
**状态日期：** 2026-08-13（V29 结果审阅与 V30 frozen-candidate consolidation 路线同步）  
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

本项目不得把统计上稳定但科学量级极小的结果直接升级为论文结论。Development 阶段继续采用少硬门、GPU-first 和人工审阅；本轮开始停止新增大方法族，转向冻结候选、统一评估和修回证据补齐。

| 层级 | 科学问题 | 当前状态 |
|---|---|---|
| C0 | 工程骨架、数据隔离、baseline reproduction 是否有效 | **已建立** |
| C1 | query-disjoint support 是否包含 participant/session corruption information | **V20 已建立** |
| C2 | learned denoiser 是否能把正确 support 转化为 strong POP 之外的收益 | **V25–V27 有 paired development 证据；V29 只建立 support-path 相对 population-adapter 的极小增量，正确 donor specificity 尚未建立** |
| C3 | subject-aware diffusion 是否稳定并处于可信竞争区间 | **V26/V27 已建立；V29 与 matched DET 接近，但绝对去噪仍弱** |
| C4 | natural EEG 上是否产生绝对 artifact attenuation，并兼顾 observation retention 与任务有效保存 | **尚未建立；V29 仅相对 PopAdapter 略优，绝对 remaining ratio 仍大于 1** |
| C5 | 冻结后的最终候选是否在 untouched sealed cohort 上复现 | **未运行** |
| C6 | 是否满足 reviewer 要求的 baseline、消融、统计、support burden、steps/latency、额外数据和 privacy | **部分完成；V30 开始统一补齐** |

## 3.1 C1：support information 已建立

V20：

```text
N_P = +0.178803
N_W = +0.174162
15/15 positive
randomization p = 1/100001
```

支持：

> Early query-disjoint support contains participant/session-specific ocular-corruption information that transfers to later queries.

## 3.2 V29 的 paired 结果必须分成两个问题

### A. Support path versus capacity-matched population adapter

V29 PA-SC-CDM：

```text
MATCH − PopAdapterCDM:
+0.000186146
95% CI [+0.000161403, +0.000208992]
15/15 positive
```

相对 paired RRMSE 约 0.741，这一效应约为：

```text
0.025% relative
```

它是一个一致但非常小的 adapter increment。

### B. Correct support versus wrong support

```text
MATCH − WRONG:
+0.000000455
95% CI [+0.000000026, +0.000000981]
11/15 positive
```

这一效应比 MATCH−PopAdapter 小约 400 倍。V29 adapter diagnostics 也显示：

```text
adapter RMS:
约 2e-3 至 1e-2

MATCH−WRONG output distance:
约 1e-5 至 7e-5
```

因此 V29 当前最准确的判决是：

```text
support-path increment:
statistically consistent but practically tiny

correct-donor specificity:
not established
```

不能把 `clear_paired_signal` 直接解释为当前 participant/session support 已成为 load-bearing corruption calibration。

## 3.3 Capacity control 的正确解释

V29 的 support adapter 比 PopAdapter 更好，但 WRONG 与 MATCH 几乎相同。这说明当前收益可能来自：

- support-context pathway 的非零共同成分；
- context variability 作为 regularizer；
- ranking loss与普通adapter训练差异；
- frozen support encoder 的共享统计；
- 而不是正确 donor 的专属性信息。

V29 只输入 V25 的 128-d context，不显式输入 V25 learned basis/projector。既往 V25 的 MATCH−WRONG 效应明显更大，提示 donor-specific geometry 可能主要位于 basis，而不是几乎重合的 context vector。

V30 不立即训练新 projector adapter；先用 frozen models 做 all-donor、shuffled、support-duration和context-versus-projector诊断。

## 3.4 V29 absolute paired denoising

在 V29 common evaluation panel 中：

```text
STANDARD temporal RRMSE:
0.741601

PA-SC-CDM MATCH:
0.741283

PA-SC-DET MATCH:
0.740942
```

PA-SC-CDM 相对 STANDARD 只改善约 0.000318，且：

```text
SNR improvement:
−0.020102

artifact RRMSE:
1.004510

artifact correlation:
−0.017422
```

所以 `improved_over_v28` 只应解释为：

```text
small consistent improvement over the frozen V28 population route on the registered V29 panel
```

不能解释为：

```text
substantial absolute denoising improvement
```

PA-SC-DET 的 artifact correlation 与 SNR 更合理，仍是重要的竞争定位基线。

## 3.5 V29 natural 结果必须区分相对与绝对

V29 MATCH 相对 PopAdapterCDM：

```text
artifact utility:
+0.000503600
14/15 positive

low-EOG observation-retention utility:
+0.000081275
11/15 positive
```

但 PA-SC-CDM MATCH 的绝对结果为：

```text
held-out EOG remaining ratio:
1.001519

artifact attenuation:
−0.004938 dB

low-EOG observation retention:
0.992258
```

因此：

```text
relative natural direction:
slightly better than PopAdapter

absolute natural artifact removal:
not established
```

`artifact_promising_retention_acceptable` 必须收窄为：

> The frozen support adapter is marginally better than its population-capacity control, but the candidate does not yet achieve absolute natural ocular attenuation.

## 3.6 C3：diffusion positioning

V29：

```text
PA-SC-CDM − PA-SC-DET:
−0.000341
95% CI [−0.001565, +0.001343]
```

这支持：

```text
diffusion and matched DET are in the same narrow performance region
```

不支持 diffusion superiority，也不要求 superiority。

## 3.7 当前项目判决

V29 engineering 接受；架构搜索应停止，但 **V29 不能直接作为已验证最终方法进入 sealed confirmation**。

“Freeze method”在当前应解释为：

```text
freeze all trained candidates and stop new architecture search
```

而不是：

```text
declare V29 the final method without a common-panel specificity and absolute-performance audit
```

---

# 4. 当前科学假设

## 4.1 V29 后仍然存活的主假设

> Query-disjoint support contains ocular-corruption information, and at least one frozen support-conditioned denoising route can exploit it. The unresolved question is which existing route provides the best combination of correct-context specificity, absolute artifact removal, observation retention, and computational cost.

## 4.2 当前主要不确定性

1. V29 的 support-path gain 是否只是 generic nonzero context / capacity effect；
2. 正确 support 是否在 all-wrong donor matrix 中具有可识别排名；
3. V25 basis/projector 是否比 128-d context 更承载 donor-specific geometry；
4. V26/V27 是否在 absolute natural attenuation 上优于 V29；
5. 哪个 frozen candidate最适合作为最终修回方法；
6. support duration、sampler steps和latency是否满足 reviewer要求；
7. stored context/basis是否具有 participant linkage 风险。

## 4.3 当前下一路线

V30：

```text
Frozen Candidate Consolidation,
Specificity Audit,
and Revision Evidence Completion
```

V30不训练新的主模型。它在一个统一 development panel 上重新评价固定候选：

```text
V25 SetCalibDET
V26 CalibSDEdit
V27 EnergySDEdit lambda_y = 0.5 / 2 / 8
V29 PA-SC-DET
V29 PA-SC-CDM
Pop/one-step controls
EEGDfus
RAW / STANDARD
```

并完成：

- all-wrong donor matrix；
- shuffled / lagged support falsification；
- context-versus-projector diagnostics；
- support-duration curve；
- sampler-step / latency curve；
- absolute paired和natural table；
- privacy/linkage diagnostic；
- reviewer-response readiness matrix；
- final candidate human selection record。

## 4.4 V30 的目标

V30不证明新方法，不进行sealed confirmation。它只回答：

1. 当前正结果是否真正依赖正确 support；
2. 哪个 frozen candidate产生绝对 artifact removal；
3. 哪个 candidate具有最合理的 attenuation–retention trade-off；
4. support需要多少时长；
5. diffusion需要多少steps、多少latency；
6. 最终稿应冻结哪个方法和哪组claims；
7. 是否已经具备打开sealed confirmation的科学条件。

---

# 5. 当前方法设计边界

## 5.1 V30 必须保留

```text
V25–V29 frozen checkpoints
same development participants
query-disjoint support
same query/noise for context swaps
participant-first aggregation
corrected natural metric semantics
K=1
RAW / STANDARD
strong DET
subject-agnostic diffusion
EEGDfus
```

## 5.2 V30 禁止

```text
new backbone training
new support encoder
new adapter family
energy re-design
routing / rollback
operator zoo
K8
sealed outcomes
manuscript result insertion
```

## 5.3 科学开发方式

V30没有自动 scientific gate。所有结果完整输出，由人工根据：

```text
absolute performance
correct-context specificity
participant heterogeneity
natural trade-off
latency
support burden
privacy risk
```

选择最终候选。

## 5.4 V30 后的决策

### 情形 A：存在 coherent frozen candidate

同时具备：

```text
correct support有可解释增量
absolute natural artifact方向合理
observation retention无明显恶化
paired性能有竞争力
latency/support burden可报告
```

则：

```text
冻结最终方法与统计方案
进入V31 sealed confirmation
```

### 情形 B：仅有 support-path、无 donor specificity

则：

```text
不得使用participant-specific calibration主张
收窄为support-conditioned / context-assisted viability
并与AE确认scope
```

### 情形 C：所有候选缺乏 absolute natural attenuation

则：

```text
不打开sealed
停止继续堆叠模型
重新确定natural数据在修回稿中的角色
```

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

## 6.12 V26 — CalibSDEdit

Branch：

```text
codex/calib-sdedit-v26
```

Base：

```text
a7d9d647b69e152255b62dbca917a4b3ed082915
```

Implementation：

```text
8257bf0...
```

Round A：

```text
c3eeb4a...
```

Round B / Natural / Ledger v1.4：

```text
9a1c469...
```

Terminal：

```text
7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c
```

Method：

```text
V25 raw-support deterministic anchor
sensor-coordinate artifact x0 diffusion
deterministic warm start
sigma_start = 0.05
DDIM 10 steps
K = 1
matched one-step and population controls
```

Forensic：

```text
equivalent basis rotation:
sensor artifact change = 1.33e-15

coefficient target relative change = 1.60667
```

这证实 V25 learned-basis residual target不适合继续使用。

Paired：

```text
one-step vs V25 DET:
−0.000047
8/15 positive

SDEdit vs one-step:
−0.003575
2/15 positive

SDEdit support vs PopSDEdit:
+0.009143
10/15 positive

SDEdit MATCH vs WRONG:
+0.009850
12/15 positive
```

Natural：

```text
SDEdit support artifact:
+0.009705
12/15 positive

SDEdit support preservation:
−0.007522
5/15 positive

SDEdit vs one-step artifact:
−0.019899

SDEdit vs one-step preservation:
−0.008492
```

判决：

```text
Engineering:
valid

Subject context:
paired_signal_preserved

Matched one-step:
equivalent_to_base_det

Diffusion positioning:
one_step_better but diffusion competitive

Natural:
preservation_concern
```

项目解释：

> V26 is the first clean-room diffusion route that is both stable and clearly support-sensitive. It remains slightly inferior to a matched one-step point estimator, but this is competitive positioning rather than a retention gate. The decisive unresolved issue is natural preservation.

工程：

```text
60/60 Round-B model cells
19/19 targeted tests
19/19 clean-archive tests
query auxiliary reads = 0
sealed reads = 0
A-track unchanged
manuscript unchanged
```

## 6.13 V27 — CalibEnergy

Branch：

```text
codex/calib-energy-v27
```

Base：

```text
7af5a00714fb72eeb75bff0c3c1c4eeb1accea8c
```

Implementation：

```text
1896a4b94f655c6b7a7282ae9384a6ca97a2d7a6
```

Round A：

```text
2d95846...
```

Round B / Natural / Ledger v1.6：

```text
1b3b2f9...
```

Terminal：

```text
40eae116e70e9de7fe0af55d64ee25551932c4a8
```

Frozen energy：

```text
lambda_y = 8
lambda_a = 1
mode = final-only
K = 1
training-fold q50/q90 mask
100 ms smoothing
```

Paired：

```text
EnergySDEdit MATCH−population:
+0.009206
95% CI [+0.001780, +0.019196]
10/15 positive

EnergySDEdit−EnergyDET:
−0.000028
95% CI [−0.000467, +0.000459]
```

Natural：

```text
energy effect on low-EOG correction proxy:
+0.124853
15/15 positive

energy effect on artifact utility:
−0.031430
4/15 positive

post-energy MATCH−population artifact:
−0.007747

post-energy MATCH−population low-EOG retention:
−0.007515
```

判决：

```text
Engineering:
valid

Energy:
improves_preservation_proxy_only

Paired support:
preserved

Diffusion:
competitive_with_one_step

Natural subject-aware joint increment:
not established
```

重要 superseding interpretation：

> The reported preservation gain is a large reduction in low-EOG correction magnitude. It is not independent ERP/SSVEP or downstream physiological-preservation evidence because the current evaluator aliases both proxies to the same scalar.

Stepwise未优于final-only；未运行energy-aware fine-tune。

工程：

```text
15/15 targeted tests
15/15 clean-archive tests
query auxiliary reads = 0
sealed reads = 0
K8 = not run
A-track unchanged
manuscript unchanged
```

## 6.14 V28 — SC-CDM

Branch：

```text
codex/support-clean-conditional-diffusion-v28
```

Base：

```text
40eae116e70e9de7fe0af55d64ee25551932c4a8
```

Implementation：

```text
44e689ac877ca73ae79f2b62efa5fce796a3f85f
```

Metric audit：

```text
2f6702d2bd007c698afa3759d20994106e5ea72a
```

Round A：

```text
5291f05c1225ceb458bde26e257205c286369037
```

Round B：

```text
7bb20735dbf0422c0871c5ce2fe34754a12115c2
```

Natural result：

```text
16337db485555a523be8351e229461eb7e4a0bfd
```

Ledger v1.8：

```text
ac56b341b627f49c075c3550cf1aa83dd124965e
```

Terminal：

```text
f7aec43e8fae1d18c2831ee44b00eae9a0098e7e
```

Method：

```text
clean x0 prediction
contaminated EEG condition
frozen raw-support context
same-backbone population diffusion
matched one-step clean predictor
30% natural consistency
DDIM10
K=1
```

Paired:

```text
MATCH−PopCleanCDM:
−0.000024
6/15 positive

MATCH−WRONG:
+0.000597
9/15 positive

SupportCleanCDM−SupportCleanDET:
−0.001924
5/15 positive
```

Natural:

```text
artifact utility:
+0.000959
10/15 positive

low-EOG observation-retention utility:
−0.001167
6/15 positive

PSD utility:
−0.000515
```

Absolute diagnostic：

```text
SupportCleanCDM temporal RRMSE:
0.731046

STANDARD:
0.731288

V25 DET:
0.706128

SupportCleanCDM artifact RRMSE:
1.008506
```

判决：

```text
Engineering:
valid

Clean conditional diffusion:
stable; competitive only with matched clean-DET architecture

Support mechanism:
paired_signal_weak

Natural artifact:
promising but tiny and uncertain

Natural observation retention:
small systematic concern

Task-valid preservation:
unavailable
```

重要解释：

> V28 successfully repaired the natural metric semantics and implemented a standard clean-x0 conditional diffusion route. However, the model remained close to the observation and did not reliably convert raw support into an advantage over an independently trained population diffusion model.

工程：

```text
60/60 model cells
60/60 checkpoint bindings
24/24 targeted tests
24/24 clean-archive tests
query auxiliary reads = 0
sealed reads = 0
K=1
A-track unchanged
manuscript unchanged
```

## 6.15 V29 — PA-SC-CDM

Branch：

```text
codex/pop-anchored-support-adapter-v29
```

Base：

```text
f7aec43e8fae1d18c2831ee44b00eae9a0098e7e
```

Implementation：

```text
65b535edf07d134f242a5a215b84526213a7a750
```

Forensic：

```text
3f595c6160be7a50651b5486a15d34b11089688c
```

Round A：

```text
63d9c24
```

Round B：

```text
1fda40d5bbd2b69656b4e0ea2ea910222acc87eb
```

Natural result：

```text
fc7abf06546fdf0f2ea7baefda4c76360f0ead2c
```

Ledger v2.0：

```text
624978018df74a7f22f4353b9571986814a274f6
```

Report package：

```text
a46b3328889b75ace1f3a88f3b1384d09a0e2044
```

Terminal：

```text
9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37
```

Method：

```text
frozen V28 population cleaner
zero-initialized support adapter
same-noise MATCH/WRONG/POP
MATCH-only ordinary target loss
counterfactual ranking
increment-only natural consistency
DDIM10
K=1
```

Reported paired：

```text
MATCH−PopAdapterCDM:
+0.000186146
15/15 positive

MATCH−WRONG:
+0.000000455
11/15 positive

PA-SC-DET MATCH−PopAdapterDET:
+0.000221875
15/15 positive
```

Reported natural relative effects：

```text
artifact:
+0.000503600
14/15 positive

retention:
+0.000081275
11/15 positive
```

Superseding interpretation：

```text
Engineering:
valid

Support pathway versus population-capacity adapter:
consistent but extremely small

Correct-donor specificity:
not established

Absolute paired denoising:
near STANDARD; artifact metrics remain weak

Absolute natural attenuation:
not established; MATCH remaining ratio > 1

Diffusion positioning:
competitive with matched DET, not superior
```

工程：

```text
60/60 model/checkpoint bindings
23/23 targeted tests
23/23 clean-archive tests
query auxiliary reads = 0
sealed reads = 0
K=1
A-track unchanged
manuscript unchanged
```

## 6.16 V30 — 当前计划，尚未执行

计划 branch：

```text
codex/frozen-candidate-consolidation-v30
```

Base：

```text
9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37
```

当前任务：

1. 同步ledger v2.1；
2. 冻结V25–V29所有checkpoint和result；
3. 建立统一paired/natural development panel；
4. 重放所有frozen candidate；
5. 运行all-wrong donor matrix；
6. 运行support lag/shuffle falsification；
7. 对比context distance与projector distance；
8. 运行0/5/10/30 s support-duration curve；
9. 运行5/10/25 DDIM steps与latency；
10. 完成absolute baseline table；
11. 完成context/projector linkage privacy probe；
12. 生成人工final-candidate selection record；
13. 不训练新模型；
14. 不打开sealed；
15. 结果后升级ledger v2.2。

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
| V25 | `a7d9d647...` | 冻结 | raw-support DET signal; latent diffusion harmful |
| V26 | `7af5a007...` | 冻结 | stable support-sensitive SDEdit |
| V27 | `40eae116...` | 冻结 | energy Pareto; retention proxy clarified |
| V28 | `f7aec43e...` | 冻结 | clean-CDM stable but near identity |
| V29 | `9ca9c79b...` | 当前最新完成 | tiny support-path gain; donor specificity unresolved |
| V30 | planned | 当前活动路线 | frozen-candidate consolidation and revision evidence |

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
V29 PA-SC-CDM
```

当前人工判决：

1. V29 engineering与provenance合格；
2. zero-init adapter和exact population bypass实现正确；
3. MATCH相对PopAdapter有15/15同向但量级极小的增量；
4. MATCH与WRONG几乎相同，正确support specificity尚未建立；
5. PA-SC-CDM绝对paired性能接近STANDARD；
6. PA-SC-CDM absolute natural remaining ratio仍大于1；
7. 不应直接把V29分类为最终验证方法；
8. 停止新增架构搜索；
9. 进入V30 frozen-candidate consolidation；
10. sealed继续关闭。

V30执行顺序：

1. 同步ledger v2.1；
2. 冻结并绑定V25–V29 checkpoints；
3. 建立common development panel；
4. common-panel重放frozen candidates；
5. all-wrong / shuffled / lagged support specificity；
6. support duration；
7. sampler steps与latency；
8. absolute natural/paired table；
9. privacy linkage probe；
10. reviewer-readiness matrix；
11. 人工选择final candidate或明确none；
12. ledger v2.2。

开发原则：

```text
不训练新主模型。
不以统计显著替代实际量级。
不以相对改善替代absolute denoising。
不把wrong≈match写成participant-specific calibration。
```

V30后：

```text
A. coherent candidate → V31 sealed confirmation
B. support specificity absent → narrow claim / consult AE
C. no absolute natural attenuation → stop model expansion and redefine natural scope
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
V29 PA-SC-CDM
```

**最新 terminal commit：**

```text
9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37
```

**Canonical remote branch：**

```text
origin/codex/pop-anchored-support-adapter-v29
```

**当前主要科学事实：**

```text
1. V20证明support中存在可转移corruption information。
2. V25–V27证明若干support-conditioned route有paired增量。
3. V29 zero-init population-anchored adapter工程正确。
4. V29 MATCH−PopAdapter为+0.000186，15/15同向。
5. 该增量仅约baseline RRMSE的0.025%。
6. V29 MATCH−WRONG仅+4.55e-7，正确donor specificity未建立。
7. V29 PA-SC-CDM paired performance接近STANDARD，artifact metrics弱。
8. V29 natural相对PopAdapter略优，但absolute remaining ratio > 1。
9. 当前不应继续发明新方法，也不应直接打开sealed。
10. 下一步是统一冻结候选并完成specificity、support burden、latency、privacy和baseline证据。
```

**当前活动路线：**

```text
V30 Frozen Candidate Consolidation and Specificity Audit
```

**当前下一问题：**

> Which already-trained candidate, if any, shows correct-context specificity, absolute ocular attenuation, acceptable observation retention, and a defensible cost profile on one common development panel?

**当前不可打开：**

```text
sealed confirmation
new model family
new support encoder
K8
routing/rollback
manuscript result insertion
```

**当前论文主线：**

```text
query-disjoint support-conditioned
subject-aware diffusion
for unseen-context ocular EEG denoising
```

**当前开发哲学：**

```text
freeze model search
consolidate evidence
report absolute and relative outcomes separately
viability and mechanism, not mandatory superiority
```

# 17. 版本记录

## v2.1 — 2026-08-13

同步 V29 并规划 V30：

- 接受V29 engineering和Git/Slurm治理；
- 记录MATCH−PopAdapter 15/15同向的微小增量；
- 将其量级换算为约0.025% relative RRMSE；
- 将MATCH−WRONG的4.55e-7判为donor specificity未建立；
- 区分support-path gain与correct-context calibration；
- 记录PA-SC-CDM absolute paired denoising接近STANDARD；
- 记录natural relative improvement但absolute remaining ratio大于1；
- 将`clear_paired_signal`降级为“tiny support-path signal; specificity unresolved”；
- 停止新增方法架构；
- 将活动路线切换为common-panel frozen-candidate consolidation；
- 加入all-wrong、support-duration、steps/latency、privacy和reviewer-readiness；
- sealed保持关闭。

## v2.0 — 2026-08-13

服务器 V29 分支版本：

- 完成population-anchored zero-init support adapter；
- 完成capacity-matched PopAdapter；
- 记录paired和natural relative effects；
- 预选freeze method and complete revision experiments。

## v1.9 — 2026-08-12

同步 V28 并规划 V29：

- 记录standard clean-x0 conditional diffusion；
- 记录natural metric语义修正与task-valid outcome unavailable；
- 记录MATCH−population约为零、MATCH−WRONG弱方向；
- 将“competitive”收窄为相对matched weak one-step architecture；
- 记录SupportCleanCDM接近STANDARD且弱于V25 DET；
- 识别固定0.1 residual、独立模型训练和未训练NULL route；
- 将活动路线切换为population-frozen support residual adapter；
- ordinary base loss只用于MATCH；
- WRONG/POP只用于counterfactual ranking/intervention；
- natural consistency只约束support increment；
- 保持K1、sealed、A-track和manuscript边界；
- 保持少hard gate和viability-not-superiority定位。

## v1.8 — 2026-08-12

服务器 V28 分支版本：

- 完成clean-signal conditional diffusion；
- 完成same-backbone population/one-step controls；
- 修正natural metric命名；
- 删除虚假ERP/SSVEP aliases；
- 记录weak support signal与small retention concern；
- 预选one small training refinement。

## v1.7 — 2026-08-12

同步 V27 并规划 V28：

- 记录EnergySDEdit paired support增量；
- 记录EnergySDEdit与EnergyDET近似等价；
- 记录energy显著改善reported preservation但牺牲artifact utility；
- 将reported preservation重新解释为low-EOG correction retention；
- 识别ERP/SSVEP proxy只是同一scalar alias；
- 将`both_failed`收窄为“subject-aware natural joint increment未建立”；
- 停止继续扩展post-hoc energy；
- 将活动路线切换为standard clean-signal conditional x0 diffusion；
- 保留raw support encoder、subject-agnostic diffusion和matched one-step；
- 要求先修正natural metric foundation；
- 保持K1、sealed、A-track和manuscript边界；
- 保持viability-not-superiority定位。

## v1.6 — 2026-08-12

服务器V27分支版本：

- 完成lightweight partial-observation energy；
- 冻结lambda_y=8、lambda_a=1、final-only；
- 记录preservation proxy改善与artifact trade-off；
- 保留paired support mechanism；
- 预选standard clean-signal conditional diffusion bridge。

## v1.5 — 2026-08-12

同步 V26 并规划 V27：

- 记录 sensor-coordinate deterministic-warm-start SDEdit；
- 记录 paired diffusion support effect与wrong-support specificity；
- 将 diffusion 状态从“实现失效”更新为“competitive but weaker than one-step”；
- 明确取消 `DIFF > DET` 的强制留存规则；
- 将 natural artifact–preservation validity设为主要解释依据；
- 记录 natural artifact方向略正、preservation小幅但系统性负向；
- 记录 one-step也存在相近preservation代价；
- 将活动路线切换为单一 lightweight partial-observation energy；
- 要求同一energy同时评价 one-step 与 diffusion；
- 保持K1、sealed、manuscript和A-track边界；
- 保持少hard gate、GPU-first、人工审阅的开发模式。

## v1.4 — 2026-08-12

服务器 V26 分支中的版本：

- 完成 V25 latent-coordinate forensic；
- 实现 matched one-step 与 sensor-coordinate SDEdit；
- 记录 paired context signal与natural preservation concern；
- 将 one-step改为竞争性定位而非diffusion生死gate；
- 预选 lightweight energy refinement。

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
