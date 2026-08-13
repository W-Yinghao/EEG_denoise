# TAAS-26-0171 项目总纲与证据账本
## Subject-Aware Diffusion for EEG Denoising

**文档性质：** 项目级权威记录、科学主线约束、分支与证据账本  
**版本：** v2.8  
**状态日期：** 2026-08-13（V32P SANDiff positive-method pilot完成）  
**建议仓库路径：**

```text
docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md
```

---


# v2.8 当前最高优先级更新：V32P SANDiff development outcome

## 执行与边界

```text
branch:
codex/sandiff-private-representation-v32p

base:
274b371ed2d3c7c105f2351f4dd88d4464fe3a66

dataset:
BCI Competition IV-2a only

encoder/task:
EEGNet / four-class motor imagery

waveform sealed reads:
0

manuscript:
unchanged and not compiled
```

采用三个严格 participant-disjoint 3/3/3 train/validation/test 轮换。每个
official trial只形成一个2秒EEGNet输入。Adaptive subject attacker在outer-test
participant的session T sanitized representation上重训，并在session E测试。

## Baseline frontier

```text
method       fixed BA   retrained BA   adaptive subject BA   verification AUROC
RAW          0.320216   0.331790       0.668596              0.593188
LEACE        0.317901   0.327546       0.665123              0.575292
DANN         0.326003   0.327932       0.679784              0.585083
one strong   0.324074   0.326582       0.661073              0.577016
SAND strong  0.325039   0.329861       0.654128              0.573476
```

结论边界：LEACE主要降低verification leakage，对adaptive classifier作用小；
注册DANN实例没有降低adaptive leakage。SANDiff和one-step均形成连续的三点
privacy–utility curve，但adaptive subject accuracy仍明显高于三类chance，不构成
formal anonymity。

## Candidate selection

三档strength只用participant-disjoint validation group选择。Nested selected
outer-test结果：

```text
method       fixed BA   retrained BA   adaptive subject BA   verification AUROC
one-step     0.322724   0.326968       0.667438              0.580559
SANDiff      0.324074   0.332176       0.664738              0.577549
```

Validation balance：

```text
SANDiff:
0.141905

one-step:
0.141384
```

因此：

```text
selected positive candidate:
SANDiff
```

该选择是竞争性定位，不是diffusion全面优于one-step。Participant-first
SANDiff−one-step effect很小且异质；SANDiff相对LEACE的fixed-head utility为
`+0.006173`，8/9 participants正向。10-step SANDiff batch-64 median latency为
9.345 ms，one-step为0.264 ms。

## Slurm lineage

```text
939632_[0-2]:
failed; P100 sm_60与icml PyTorch不兼容；无科学输出

939635_[0-2]:
completed then superseded; test-aggregate selection protocol archived

939638_[0-2]:
completed then superseded; validation-only selection完成，后补participant privacy rows

939641_[0-2]:
accepted; validation-only selection与participant-first privacy evaluation
```

所有recovery保持模型、数据、seed、预算和test panel不变。Phase-C waveform
interaction为`deferred_not_comparable`，因为V27 SGEYESUB waveform outputs不能与
BCI-IV-2a trials直接对齐；没有重建waveform pipeline。

## 下一路线

冻结V32P SANDiff candidate，先人工审阅其小幅utility/privacy改善是否足以支撑修回
叙事。不得自动增加第二dataset、foundation encoder、membership inference或新
diffusion family。


# v2.7 当前最高优先级更新：SANDiff 正向方法路线

## 总判决

下一轮不再把 EEG 与 subject 的关系拆成过细的生理、session、montage 或 acquisition 因果因素。

统一采用操作性定义：

```text
subject-linked nuisance
=
在训练威胁模型下可预测subject，
但对声明任务并非必要的representation component
```

该定义不声称这些成分具有唯一生理来源。

## 当前主方法

```text
SANDiff
Subject-Aware Nuisance Diffusion
for Privacy-Preserving EEG Representations
```

核心思路：

```text
识别subject-linked nuisance
保留task-relevant complement
用task-conditioned diffusion替换private component
而不是简单将其置零
```

## 研究粒度

V32P只做：

```text
一个数据集
一个encoder
少量强baseline
一个matched one-step control
一个selective diffusion method
一个小型validation sweep
```

不在本轮同时研究：

```text
identity versus session causal decomposition
多个foundation models
多个数据集
membership inference
support-specific private subspace
复杂guidance或operator portfolio
```

## Support 的角色

Primary method在推理时：

```text
不需要subject ID
不需要query-disjoint support
```

Subject awareness来自训练阶段对subject-linked nuisance的识别。

Query-disjoint support保留为后续可选增强，不再是主方法成立的前提。

## 正向方法目标

本轮必须形成一个明确的 privacy–utility candidate，而不是继续产生 audit/no-go 路由。

允许一次小型方法修正，但禁止扩展成新的模型动物园。

# v2.6 当前最高优先级更新：V31 对 Route P 的影响

## R.1 总判决

V31 不改变 Route P 作为主路线的选择。

V31 的结果进一步确认：

```text
waveform Route A / V25–V29 没有唯一可进入sealed confirmation的候选；
更多ocular support不保证更好的natural attenuation–retention trade-off；
强participant linkage与有效denoising specificity并不等价。
```

因此，Route P 的主问题仍然成立：

> EEG representation 中存在可被稳定链接的subject/context information，但这部分信息不一定对任务或去噪有用；应研究如何定向删除这种private shortcut，同时保留任务信息。

## R.2 V31 exact-duration结果的解释

V31 repaired duration protocol使用：

```text
0 / 5 / 10 / 30 / 120 s
2 s non-overlapping windows
duration-prefix-only EOG normalization
0 s exact population bypass
same query/checkpoint/noise
K=1
```

代表结果表明：

```text
V25/V26:
paired fidelity随support增加而改善，
但natural remaining ratio不单调，更多support未解决trade-off。

V29:
约5 s即基本饱和，
但仍保持near-identity behavior。
```

对 Route P 的含义不是“support无用”，而是：

1. support中可链接的context information可能在很短时间内饱和；
2. 更长support并不自动提高有效utility；
3. support duration应被解释为privacy/attack budget和sanitizer calibration burden；
4. Route P必须沿用V31 exact prefix-only、non-overlap protocol；
5. 不得把ocular-denoising duration curve直接当作identity-removal duration curve。

## R.3 Route P 的四项正式修订

### 修订 1：区分稳定身份与session/acquisition context

V30/V31 的linkage可能来自：

```text
stable subject physiology
session state
cap placement
reference
impedance
montage/acquisition practice
```

Route P必须分别评价：

```text
within-session subject classification
cross-session identity verification
cross-task identity retrieval
session/acquisition classification
```

不能把within-session linkage自动称为brain identity。

### 修订 2：global private subspace为primary，support不再承担subspace识别主责

Primary private subspace：

```text
P_priv learned only from training participants
using subject labels and cross-session validation
```

Query-disjoint support首先只产生：

```text
ephemeral identity/context prototype c_s
```

用于negative privacy guidance：

```text
move sanitized representation away from the source-specific prototype
```

Support-specific subspace correction：

```text
P_s = P_0 + Delta P(C_s)
```

降为secondary ablation，只有在cross-session identity evidence支持时才解释为local private geometry。

### 修订 3：lagged/shuffled support在Route P中不是自动失败

在ocular operator路线中，lagged/shuffled不劣化反驳同步operator specificity。

在privacy路线中，身份信息可能主要位于：

```text
spatial topography
spectrum
covariance
amplitude distribution
acquisition signature
```

因此time shuffle后仍可link并不反常。

Route P中的lag/shuffle用于定位private signal的来源，而不是作为统一no-go criterion。

### 修订 4：把privacy–utility mismatch设为核心科学假设

V31 reaffirmed：

```text
support state可高度linkable，
但correct-context denoising utility mixed。
```

Route P的核心假设改为：

> Stable subject/context information can be highly learnable yet unnecessary—or even harmful—for the declared task. Selective representation diffusion should remove the linkable component rather than preserve it as a personalization token.

## R.4 Route A与V27的新角色

```text
Route A:
历史几何模板，不再作为primary empirical route。

V27:
waveform-level backup + Route P upstream preprocessing comparator。
```

Route P必须比较：

```text
RAW waveform
STANDARD / ICA
V27-L0.5 waveform cleaning
```

再进入同一encoder与privacy attack，回答：

> Ocular waveform denoising会降低identity leakage，还是会去除短时噪声并暴露更稳定的brainprint？

## R.5 下一执行轮改名

V31已经完成，因此此前规划的 `V31R` 作废。

新的下一轮为：

```text
V32P — Identity-versus-Context Decomposition,
Privacy Baseline Frontier,
and Selective Diffusion Pilot
```

本轮先建立：

1. stable identity vs session/acquisition leakage；
2. deterministic privacy–utility frontier；
3. exact support-duration privacy curve；
4. V27 waveform preprocessing effect；
5. one selective private-subspace diffusion pilot。

不打开waveform Route A的sealed confirmation。

---

# v2.5 当前最高优先级更新：Route P

## P.1 AE 范围许可

用户已经与 AE 沟通过。当前编辑边界为：

```text
该大修实质上接近 reject-and-resubmit；
只要新稿继续属于 subject-aware diffusion for EEG denoising，
允许进行大幅方法、任务与实验重构。
```

因此，不再需要将 V30 的 mixed-result audit 本身做成论文主题。

## P.2 当前主线切换

新的主路线定义为：

```text
Route P — Privacy-Aware Representation Denoising
```

Provisional title：

```text
Denoise the Identity, Preserve the Task:
Subject-Aware Diffusion for Privacy-Preserving EEG Representations
```

核心问题：

> EEG representation 中同时包含 task information 与稳定的 subject/private information。能否通过一个 subject-aware stochastic denoising channel，定向削弱 task-irrelevant identity leakage，同时保持或改善跨被试任务效用？

这里的 “denoising” 明确指：

```text
representation-level nuisance / shortcut / private-factor removal
```

而不是宣称 subject identity 本身等于生理噪声。

Primary information-theoretic target：

\[
\max I(\widetilde Z;Y)
-\lambda I(\widetilde Z;S\mid Y),
\]

其中：

```text
Z: EEG representation
Y: task label / task semantics
S: subject identity or protected subject attribute
Z_tilde: privacy-sanitized representation
```

## P.3 Route A 的新角色

此前 Route A / CSPD 不再作为主稿路线，但不废弃。

保留内容：

1. “preserve the reliable complement, reconstruct the uncertain subspace”的几何思想；
2. query-disjoint support；
3. population prior + context-specific correction；
4. wrong / shuffled / population interventions；
5. V27 attenuation–retention Pareto；
6. 5-step diffusion latency结果。

在 Route P 中，这些元素被重新解释为：

```text
ocular subspace         → private/identity subspace
reliable complement     → task-preserving complement
clean-waveform prior    → task-conditioned population representation prior
artifact attenuation    → identity leakage reduction
waveform preservation   → downstream utility preservation
```

V27 保留为：

```text
fallback waveform-level manuscript route
+
upstream waveform-denoising comparator in Route P
```

## P.4 为什么 Route P 优先于 Route A

V30 的 strongest positive evidence 不是 correct ocular operator specificity，而是：

```text
context + projector linkage top-1 = 0.836
same/different AUROC = 0.962
```

Route A 将这种强linkage视为privacy cost，却未能稳定转化为correct-donor denoising utility。

Route P 直接把这一事实作为机制headroom：

> support state确实含有强subject signal，因此可以研究如何把它从共享representation中定向去除。

此外，用户已有bias/privacy内部结果显示：

```text
raw feature subject-ID accuracy ≈ 0.998
simple residual x − x_ref ≈ 0.019
CMI/H2-CMI or transformed representations can reintroduce substantial identity leakage
```

这说明：

```text
identity is removable,
but removal is representation- and transformation-dependent.
```

## P.5 当前 novelty scope

单独的 EEG identity removal 已不是新问题；单独的 diffusion anonymization 也不是新问题。

当前最有希望的交集是：

1. query-disjoint support estimates an ephemeral private signature or private subspace；
2. diffusion selectively resamples only the private coordinates；
3. task-relevant complement is anchored；
4. privacy strength is continuously controllable；
5. evaluation uses adaptive closed-set and open-set attackers；
6. the method is tested across conventional EEG encoders and at least one EEG foundation model；
7. privacy, cross-subject bias, task utility, latency and state deletion are jointly reported。


## P.6 当前方法草图（V31修订版）

冻结或预训练 EEG encoder：

\[
Z=E(X).
\]

首先从training participants学习两个可区分的因素：

```text
P_id:
跨session稳定、可支持subject linkage的private subspace

P_ctx:
session / acquisition / montage context subspace
```

Primary privacy operator：

\[
P_{\mathrm{priv}} = \operatorname{span}(P_{\mathrm{id}}, P_{\mathrm{ctx}}),
\qquad
Q=I-P_{\mathrm{priv}}.
\]

Query-disjoint support只生成ephemeral prototype：

\[
c_s=A(C_s),
\]

它用于privacy guidance，不默认决定private subspace。

只在private span中前向加噪：

\[
Z_t
=
QZ
+
\sqrt{\bar\alpha_t}P_{\mathrm{priv}}Z
+
\sqrt{1-\bar\alpha_t}P_{\mathrm{priv}}\epsilon.
\]

反向过程条件于：

```text
QZ
task label / frozen teacher logits / task prototype
optional acquisition metadata
```

并通过source-prototype avoidance降低linkage：

\[
E_{\mathrm{priv}}
=
\operatorname{sim}
\bigl(g(\widehat Z_0),c_s\bigr).
\]

Task consistency：

\[
E_{\mathrm{task}}
=
D_{\mathrm{KL}}
\bigl(h(Z)\|h(\widehat Z_0)\bigr).
\]

输出：

\[
\widetilde Z
=
QZ+P_{\mathrm{priv}}\widehat Z_0.
\]

Primary first implementation：

```text
global P_priv
+
query-disjoint negative identity prototype
+
no persistent subject token
```

Secondary ablation：

```text
support-specific Delta P(C_s)
```

只有cross-session结果证明support包含stable identity geometry时，才允许解释该local correction。

Primary constraints：

```text
privacy:
reduce adaptive cross-session identity/linkage attacks

utility:
preserve fixed-head and retrained-head task performance

bias:
improve or preserve worst-subject and cross-subject generalization

context separation:
report subject, session and acquisition leakage separately

fidelity:
limit task-complement distortion
```

## P.7 必须比较的方法

Primary EEG privacy baselines：

```text
RAW / no sanitization
Gaussian perturbation
User-wise perturbation
DANN / GRL
INLP
LEACE
FHVAE-style subject/content disentanglement
ID-RemovalNet or a source-faithful implementation
matched deterministic privacy adapter
```

Diffusion controls：

```text
latent diffusion without privacy guidance
privacy-guided diffusion without support
support-conditioned private-subspace diffusion
matched one-step sanitizer
```

Waveform preprocessing controls：

```text
RAW
ICA / standard preprocessing
V27 waveform-denoised EEG
```


## P.8 数据、support budget与威胁模型（V31修订版）

Primary datasets：

```text
BCI Competition IV-2a / MI4C
P300 or ERN
```

External setting：

```text
one sleep-staging dataset
or
one EEG foundation-model cohort
```

Encoders：

```text
EEGNet
one frozen EEG foundation model: LaBraM or CBraMod
```

Threat models分层：

```text
A. within-session closed-set subject classification
B. cross-session open-set verification
C. cross-task retrieval/linkage
D. session/acquisition classification
E. adaptive attacker retrained after sanitization
F. membership inference
```

Support duration沿用V31 exact contract：

```text
0 / 5 / 10 / 30 / 120 s
non-overlapping chronological prefixes
prefix-only normalization
no repeated or future samples
0 s exact no-support/population route
```

每个duration分别报告：

```text
identity attack
session/acquisition attack
task utility
privacy–utility frontier
support encoding time
stored-state bytes
```

不能只报告一个被冻结的linear subject probe。


## P.9 当前下一执行轮

Next round：

```text
V32P — Identity-versus-Context Decomposition,
Privacy Baseline Frontier,
and Selective Diffusion Pilot
```

本轮内容：

1. two datasets × two encoders；
2. within-session / cross-session / cross-task attacks；
3. RAW / LEACE / INLP / DANN / perturbation / ID-RemovalNet-style baselines；
4. fixed-head + retrained-head utility；
5. exact 0/5/10/30/120 s privacy support curve；
6. V27-L0.5 waveform preprocessing condition；
7. global-private-subspace selective diffusion K=1 pilot；
8. matched one-step sanitizer；
9. privacy–utility–latency Pareto；
10. GPU-first，少hard gates；
11. waveform sealed confirmation保持关闭。

## P.10 当前项目判决

```text
Primary route:
Route P — representation-level privacy denoising

Backup route:
V27 waveform-level attenuation–retention method

Historical design template:
Route A / CSPD geometry

Inactive route:
audit-centric paper framing
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

> Denoise the subject-linked nuisance, preserve the task.

研究对象从 waveform artifact removal 转为 EEG representation denoising：

```text
输入:
EEG representation

需要去除:
subject-linked private nuisance

需要保留:
task-relevant information
```

“Subject-linked”是操作性概念，不对其来源作过细因果解释。它可以同时包含稳定个体差异和采集相关结构，只要该结构在攻击模型下可用于subject linkage。

## 2.2 当前核心方法原则

```text
Use subject awareness to identify what to remove,
not whom to reconstruct.
```

Primary method：

- 使用训练subject labels识别private representation component；
- 保留其task-relevant complement；
- diffusion只重采样private component；
- 输出不再条件于source subject identity；
- 推理不需要subject ID或target support；
- adaptive attacker必须在sanitized outputs上重新训练；
- fixed-head和retrained-head utility同时报告。

## 2.3 当前不作为主线的内容

```text
ocular operator recovery
session/acquisition causal decomposition
query-specific routing
rollback
support-specific projector
large posterior-energy system
K8 sampling
privacy audit-only paper
纯确定性erasure paper
```

V27 waveform denoising保留为后续输入条件对照，不承担新方法主张。


# 3. 当前证据阶梯

| 层级 | 问题 | 当前状态 |
|---|---|---|
| P0 | 数据、participant split、session split和attack protocol是否正确 | **待V32P建立** |
| P1 | 原始representation是否存在可复现的subject leakage | **已有V30 linkage headroom，但需在新任务/encoder下重建** |
| P2 | LEACE/DANN等deterministic方法的privacy–utility frontier | **未运行** |
| P3 | SANDiff是否形成正向privacy–utility candidate | **未运行** |
| P4 | Diffusion相对matched one-step是否提供额外价值 | **未运行；不是生死gate** |
| P5 | 第二数据集或第二encoder是否复现 | **后续轮次** |
| P6 | 是否形成最终TAAS修回证据 | **未建立** |

本轮不再设置大量科学硬阈值。只有数据泄漏、攻击协议错误、nonfinite/scale collapse和checkpoint错误属于硬停止条件。

成功标准采用人工综合判断：

```text
adaptive privacy reduction
task utility
cross-session robustness
comparison with LEACE/DANN
comparison with matched one-step
latency
```

目标是找到至少一个可清楚解释的正向 operating point。


# 4. 当前正向方法：SANDiff

## 4.1 Representation decomposition

\[
Z=E(X).
\]

先用训练subjects上的线性concept-erasure模型得到一个广义private component：

\[
Z_{\mathrm{keep}}=T_{\mathrm{erase}}(Z),
\qquad
Z_{\mathrm{priv}}=Z-Z_{\mathrm{keep}}.
\]

Primary使用LEACE定义初始private component；INLP作为baseline。

这里的private component只表示：

```text
subject-predictive under the registered training attacker
```

不解释为纯生理身份或纯采集噪声。

## 4.2 Selective diffusion

只对 \(Z_{\mathrm{priv}}\) 加噪和反向生成。

条件包括：

```text
Z_keep
task teacher logits or task prototype
diffusion timestep
```

不包括：

```text
source subject ID
persistent subject embedding
test support
```

生成新的population-plausible private component：

\[
\widehat Z_{\mathrm{priv}},
\]

并输出：

\[
\widetilde Z
=
Z_{\mathrm{keep}}
+
\widehat Z_{\mathrm{priv}}.
\]

## 4.3 Training objectives

Primary objectives：

```text
task consistency
adaptive subject privacy
representation realism
limited complement distortion
```

Privacy adversary在sanitized representation上训练。

Matched one-step sanitizer使用同样的：

```text
Z_keep
task condition
training data
parameter scale
```

但一次前向生成replacement。

## 4.4 Positive claim scope

期望主张：

> Selective diffusion can replace subject-linked nuisance in EEG representations while preserving task utility, yielding a controllable privacy–utility frontier.

不要求：

```text
解释subject signal的唯一来源
证明所有subject information有害
diffusion全面胜过deterministic sanitizer
```


# 5. 当前活动路线

V32P：

```text
SANDiff Positive Method Pilot
```

本轮只使用：

```text
BCI Competition IV-2a
EEGNet representation
participant-grouped development split
session-separated privacy evaluation
```

## 5.1 Phase A — Baseline frontier

运行：

```text
RAW representation
LEACE
DANN/GRL
matched one-step sanitizer
```

建立最小 privacy–utility frontier。

## 5.2 Phase B — SANDiff

训练一个 selective diffusion model：

```text
K=1
one canonical sampler setting
small validation-only privacy-strength sweep
```

允许一次小型修正：

```text
private component rank/strength
或
task-consistency weight
```

不允许同时更换backbone、dataset和method family。

## 5.3 Phase C — Optional waveform interaction

若V27-L0.5 frozen outputs可直接复用，则比较：

```text
RAW waveform
STANDARD waveform
V27-L0.5 waveform
```

对同一EEGNet representation和privacy attacker的影响。

若不能直接复用，本轮标记deferred，不重建waveform pipeline。

## 5.4 下一轮扩展

只有在V32P形成可解释candidate后，下一轮才增加：

```text
第二数据集
或
一个EEG foundation encoder
```

本轮不同时扩展二者。

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

## 6.16 V30 — Frozen Candidate Consolidation and Specificity Audit

Branch：

```text
codex/frozen-candidate-consolidation-v30
```

Base：

```text
9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37
```

Implementation：

```text
98a954848d4d97d967c522148962ec12ed6ef79b
```

Common panel：

```text
e6db0245ce6f035372cb977ae6f778c031e02e3c
```

Specificity：

```text
33bc0dd80533e4a266575f7c6ad8f8fe992bd5ef
```

Duration / latency：

```text
001f7266945c7940824672ad80383727f3d4f767
```

Natural / privacy：

```text
f48c4533ceea031db7444232ba3567cd2577707f
```

Ledger v2.2：

```text
b17e76229d65f4923d95fb30035156f3dd82bc9a
```

Report package：

```text
60f9b64d760bdd9d2eafe96925be0e914dad1fc3
```

Terminal：

```text
220dcbaaabdef0cb8d1ac91b87b0d1cc8b7109cf
```

Final selection：

```text
selected_candidate: none
next_route: narrow claim and consult AE
sealed confirmation: not authorized
```

All-donor：

```text
V25 correct top-1:
1/15
median rank 6
correct−mean wrong +0.01446

V26 correct top-1:
1/15
median rank 4
correct−mean wrong +0.01580

V29 CDM correct top-1:
1/15
median rank 9
```

Falsification：

```text
V25/V26 lagged and shuffled support did not worsen risk
```

Candidate trade-off：

```text
V27-L0.5:
best paired and absolute attenuation
but retention and PSD cost

V29:
high observation retention
but no absolute attenuation
```

Privacy：

```text
context+projector top-1 = 0.836
AUROC = 0.962
```

Engineering：

```text
132 accepted cells
29 failed
23 recovery
1 superseded
31/31 targeted tests
31/31 clean-archive tests
query auxiliary reads = 0
sealed reads = 0
A-track unchanged
manuscript unchanged
```

Superseding audit note：

```text
V30 support-duration curve is not manuscript-valid because shorter-duration
normalization used the full 120-second EOG prefix and the window schedule did not
strictly implement non-overlapping exposure budgets.
```


## 6.17 V31 — Claim Narrowing and Exact Duration Repair

Branch：

```text
codex/claim-narrowing-ae-consult-v31
```

Terminal：

```text
274b371ed2d3c7c105f2351f4dd88d4464fe3a66
```

结果：

```text
waveform scientific values unchanged
selected waveform candidate = none
exact duration repair completed
waveform sealed confirmation remained closed
```

V31结束waveform route的继续方法搜索，并为Route P提供了正确的support-budget协议。

## 6.18 V32P — SANDiff positive-method pilot

Branch：

```text
codex/sandiff-private-representation-v32p
```

Base：

```text
274b371ed2d3c7c105f2351f4dd88d4464fe3a66
```

状态与结果：

1. BCI-IV-2a EEGNet representation与subject/task baseline完成；
2. LEACE、DANN、matched one-step和SANDiff完成；
3. adaptive cross-session privacy与fixed/retrained task evaluation完成；
4. 三档privacy strength和两个development seeds完整保留；
5. validation-only nested selection选择SANDiff；
6. V27 waveform ablation因数据不对齐记为`deferred_not_comparable`；
7. waveform sealed reads为0。


# 7. 分支与 commit 账本

| 阶段 | Branch / Commit | 状态 | 作用 |
|---|---|---|---|
| A-track | `0c4f2301...` | 只读 | waveform clean-room理论与历史 |
| V27 | `40eae116...` | frozen | waveform attenuation operating point |
| V30 | `220dcbaa...` | frozen | waveform specificity/latency/privacy consolidation |
| V31 | `274b371e...` | latest complete | exact duration repair and route transition |
| V32P | terminal commit pending | latest complete | SANDiff selected positive representation candidate |

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

| Reviewer / AE 要求 | V30 后状态 | 下一动作 |
|---|---|---|
| stronger denoising baselines | complete | 统一表格进入consultation package |
| subject-agnostic DDPM | complete | 保留V26/V29 population routes |
| RAW / STANDARD | complete | 保留absolute table |
| subject component ablation | complete but mixed | 如实报告all-donor/lag/shuffle |
| wrong/null/shuffled controls | complete | 不将mixed结果写成成功 |
| statistics / CIs | complete | participant-first |
| target/support amount | **partial** | V31修复duration implementation |
| sampling steps | complete | 5/10/25, K=1 |
| latency / memory | complete | A100 fixed benchmark |
| privacy | complete | 高linkage risk，必须进入主文或limitations |
| transductive/support setting | complete | within-session, fixed montage, query-disjoint |
| extra dataset / montage | partial | natural SGE only；sealed未开 |
| task-valid preservation | missing | ERP/SSVEP/ERD-ERS unavailable |
| method clarity | requires new manuscript blueprint | V31生成scope-specific outline |

---

# 11. 当前风险清单

| 风险 | 级别 | 当前处理 |
|---|---:|---|
| correct-support specificity mixed | 极高 | 不再使用强participant-operator claim |
| no single attenuation–retention winner | 高 | 以Pareto / audit framing替代dominance claim |
| task-valid physiology unavailable | 高 | natural claim严格收窄 |
| support state highly linkable | 高 | 作为核心privacy finding处理 |
| V30 support-duration implementation flawed | 高 | V31 exact repair；旧curve降级 |
| large scope change may exceed revision | 高 | 咨询AE，不擅自重写submission |
| deterministic baseline often as strong as diffusion | 中 | viability-not-superiority定位 |
| extensive clean-room history may make manuscript散乱 | 中 | 只保留一条主叙事，其余进appendix/ledger |
| no sealed confirmation | 中 | 未选定scope前保持关闭 |

---


# 12. 当前下一步

下一步固定为：

```text
V32P SANDiff Positive Method Pilot
```

本轮原则：

```text
方法优先
范围适中
不做过细因果拆解
不以audit为论文中心
不在首轮扩展多个数据集和多个encoder
```

服务器需要：

1. 在BCI-IV-2a上建立可信的RAW leakage与task utility；
2. 完成LEACE、DANN和matched one-step；
3. 训练SANDiff；
4. 使用adaptive attacker评价；
5. 输出privacy–utility Pareto；
6. 选择一个validation-defined candidate；
7. 若首个配置较弱，允许一次小型方法修正；
8. 不打开waveform sealed confirmation；
9. 不修改manuscript。

V32P报告后再决定：

```text
扩展第二数据集
或
扩展一个foundation encoder
```

两者不同时开始。

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
V31 Claim Narrowing + Exact Support-Duration Repair
```

**最新 terminal commit：**

```text
274b371ed2d3c7c105f2351f4dd88d4464fe3a66
```

**当前主线：**

```text
SANDiff:
Subject-Aware Nuisance Diffusion
for Privacy-Preserving EEG Representations
```

**当前研究定义：**

```text
subject-linked nuisance
=
subject-predictive but task-unnecessary representation component
```

不进一步假设其唯一生理或采集来源。

**当前方法边界：**

```text
one dataset
one encoder
support-free primary inference
LEACE-defined initial private component
selective diffusion replacement
matched one-step control
adaptive privacy attacker
fixed/retrained task heads
```

**当前下一问题：**

> Can selective diffusion replace subject-linked nuisance in an EEG representation and produce a better privacy–utility operating point than linear erasure and a matched one-step sanitizer?

**当前不开放：**

```text
waveform sealed confirmation
second dataset
foundation-model expansion
support-specific private subspace
membership inference
complex routing/guidance
manuscript modification
```


# 17. 版本记录

## v2.8 — 2026-08-13

- 完成BCI-IV-2a single-encoder SANDiff positive-method pilot；
- 建立严格3/3/3 participant-disjoint train/validation/test rotation；
- 在outer-test session T重训adaptive attacker并在session E测试；
- 完成RAW、LEACE、DANN、matched one-step和SANDiff frontier；
- 完成weak/medium/strong三档与两个development seeds；
- validation-only nested selection选择SANDiff；
- 记录SANDiff相对one-step仅为小幅、异质的competitive positioning；
- 记录adaptive leakage仍远高于chance，不声称formal anonymity；
- V27 waveform interaction记为`deferred_not_comparable`；
- waveform sealed reads保持0，manuscript保持不变。

## v2.7 — 2026-08-13

- 将Route P从过细的identity/context decomposition收敛为subject-linked nuisance；
- 明确该概念是操作性定义，不作唯一因果解释；
- 将primary method改为support-free inference；
- 冻结SANDiff作为正向方法；
- 采用LEACE-defined private component和selective diffusion replacement；
- matched one-step继续作为竞争定位；
- V32P只运行一个数据集、一个encoder和少量baseline；
- 允许一次小型方法修正，不允许模型动物园；
- V27只作为可选waveform input ablation；
- 第二数据集和foundation encoder后置；
- waveform sealed confirmation保持关闭。

## v2.6 — 2026-08-13

- 同步V31 official-v2.3 reconciliation与exact duration repair；
- 记录V31没有改变waveform scientific values；
- 记录V25/V26 paired随support增长但natural non-monotonic；
- 记录V29约5 s饱和且near-identity；
- 确认Route P保持primary，不返回audit-centric scope；
- 将stable identity与session/acquisition context正式拆分；
- 将global private subspace设为primary；
- 将query-disjoint support改为ephemeral negative identity prototype；
- 将support-specific subspace correction降为secondary ablation；
- 明确lagged/shuffled在privacy任务中不是自动no-go；
- 将support duration改为privacy/attack budget并沿用V31 exact contract；
- 下一轮改名为V32P；
- V27保留为waveform preprocessing comparator和backup；
- waveform sealed confirmation继续关闭。


## v2.3 — 2026-08-13

同步 V30 并规划 V31：

- 记录selected_candidate = none；
- 接受不开放sealed confirmation；
- 将specificity分类为mixed；
- 记录V25/V26 correct > mean wrong但correct rarely top-1；
- 记录lagged/shuffled未造成性能下降；
- 记录V27-L0.5 attenuation与retention/PSD trade-off；
- 记录V29 retention高但无absolute attenuation；
- 记录context+projector linkage top-1 0.836、AUROC 0.962；
- 区分confirmation failure与paper impossibility；
- 发现并登记V30 support-duration implementation缺陷；
- 将support amount readiness降级为partial；
- 停止新方法搜索；
- 将活动路线切换为claim narrowing、AE consultation和duration repair；
- 保持sealed与manuscript关闭。

## v2.2 — 2026-08-13

服务器 V30 分支版本：

- 完成common-panel replay；
- 完成all-donor specificity与falsification；
- 完成duration、latency和privacy analyses；
- 人工选择none；
- 预选narrow claim and consult AE。

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


## v2.5 — 2026-08-13

- 记录AE允许近似reject-and-resubmit式大改；
- 明确用户拒绝audit-centric主稿定位；
- 将主线切换为representation-level privacy denoising；
- 将Route A降为几何模板，将V27保留为waveform-level备选；
- 把V30强linkage结果转化为private-subspace headroom；
- 新增conditional privacy-funnel目标；
- 新增ID-RemovalNet、user-wise perturbation、LEACE、INLP、DANN、FHVAE等comparators；
- 新增closed/open-set adaptive privacy attacks；
- 下一轮定义为V31R baseline reproduction + GPU pilot；
- 不打开V30 sealed confirmation。
