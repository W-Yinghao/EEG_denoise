# TAAS 审稿意见分析与修改计划

> 稿件：*Subject-Aware Denoising Diffusion Models for Cross-Subject EEG Denoising*（TAAS，AE：Ziyu Jia）
> 决定：**Major Revision**，回稿截止 **2026-11-21**（约 4 个月）
> 分析基于：提交版 30 页 PDF（`Subject_Aware_..._(1).pdf`）、代码库 `RESULTS.md` / `MANUSCRIPT_UPDATES.md`、已完成实验 `results/m7…m12/`。

---

## 执行进度（2026-07-28）

**A 类写作 —— 已完成并本地验证（改在 `submitted_src/`）：**
- ✅ R2-W1/Q1 Fig 1↔公式一致性：图注/Description 改为"分支预测噪声分量 $\epsilon^c_\theta,\epsilon^s_\phi$（Eq. combined_epsilon）→ $\widehat\epsilon$ → $x_{t-1}$（reverse mean）"，并显式说明图中信号级标注与噪声级记号是同一对象。
- ✅ R2-Q2 SADDPM vs SADDPM-Cond 对照表：新增 `Table~\ref{tab:variant}` 于 §3.6（input/target/learns/data/objective/inference/用途 7 行）。
- ✅ R2-W3 理论精简：Prop 3.2(containment)、Prop 3.3(noise↔reverse 恒等)、Lemma 3.8(x0/eps) 从正文移入附录（statement+proof 并置）；正文保留 Prop cond/orth/da/obs/joint。
- ✅ R1-Q5 Table 5 低相关生理解释：§4.4.2 增一段（z-score 去幅值、trial 平均衰减非锁时成分、残留为被试特异谱形/空间模式；相对结构而非绝对量级才是重点）。
- ✅ R3-W2 transductive：abstract 补显式一句（结论处提交版已有）。
- ✅ R2-Q10 占位符：main.tex 注释 `\acmDOI` 与 `\received` 三行、`\shortauthors` 改 "He et al."；Vol/No/Article 保留（acmart 默认即 0，注释无实际差异且有风险）。
- ⏳ R2-Q9/R1-Q1 超参+分类器细节附录表：SADDPM/adaptation/SADDPM-Cond 超参可写；**下游分类器架构/训练细节待合作组提供**后补。

**验证方式（本地 TeX Live 缺 acmart 依赖 xstring/totpages/… 无法编 acmart）：** 用 `article` 校验壳 `\input` 全部 section 编译——26 页、0 fatal、0 undefined control sequence、0 未解析引用、0 重复标签；全文件花括号与 begin/end 平衡；移出的 5 个标签各定义一次。**权威编译以 Overleaf 为准。**

---

## 一、总体判断

- **三位审稿人一致 Major，但基调正面、无 reject 倾向。** R2 甚至 *best paper = Yes*，三人都认为方法新颖、结论 sound（R1/R2 明确勾选 "sound, factual, accurate = Yes"）。这是一次**"补实验 + 补统计 + 补澄清"型的大修**，没有人质疑方法有根本性错误。可修改性高。
- **AE 把三份意见收敛为 9 条诉求**，全部落在四个方向：
  1. 更强的基线（不止 ICA）：更强去噪基线、subject-agnostic DDPM、raw/常规预处理 EEG；
  2. 系统消融：subject embedding、FiLM、residual(individual) 分支、各 loss 项；
  3. 统计与稳健性：显著性检验、关键超参与 adaptation 过程的敏感性分析；
  4. 澄清：SADDPM vs SADDPM-Cond 的输入/目标/推理区别、图-公式一致性、目标数据量、推理延迟、transductive 局限、embedding 隐私。
- **好消息：稿件里已经有相当一部分"半成品答案"**（§3.6 SADDPM-Cond、§5 EEGdenoiseNet 配对实验、Table 8 embedding 消融、附录 EMG/谱图）。代码库里 `results/m7…m12/` 还压着大量**已跑但未完全进正文**的数据，很多诉求只需"取数-作图-写作"，不必重跑。

---

## 二、前置决策 / 风险（更新于 2026-07-27，用户已决策）

### ✅ 风险 1（已决策）：下游分类表 Table 2–4 —— **保持不动**
- 用户决定：Table 2–4 是**合作组（另一支团队）跑出的真实结果**，用户确认其真实性但**拿不到其代码**。本仓库的复现（M7：SADDPM 0.276 vs ICA 0.284）是**独立的另一套实现**，协议/训练细节无法与合作组对齐，差异不足以证明其表有误——**因此不改、不碰下游表，也不把我们的复现摆到与之矛盾的位置。**
- **由此确定的下游应对策略（不触碰 Table 2–4）：**
  1. **R1-W5 显著性/CI → 已解决（零新实验）**：直接用 Table 2–3 里已发表的 81 个数字做配对检验（见 §"下游显著性结果"）。全部支持原主张，无矛盾。
  2. **R1-W1/W2/W4/Q2（更强/subject-agnostic/raw 下游基线）**：**不在 Table 2–4 上加行**（那需要合作组的 pipeline）。改为在**我们完全掌控、可复现的配对 GT 协议 §5** 上提供 raw（=Noisy 行，已有）、subject-agnostic（null-embedding / 去 embedding 重训，可复现）、更强学习型基线（CNN，已有）；在下游节里说明 ICA 作为经典参照保留，学习型/subject-agnostic/raw 的系统比较放在直接保真度协议 §5。
  3. **R1-W6 t0/K + R1-Q1 分类器细节**：合作组**同意给信息**（不跑新实验）——取值/架构从他们那里拿来写入正文；**敏感性分析**在我们可复现的 SADDPM restoration 上给（M5 的 t\* sweep + probe 的 K）。
- 见 [[manuscript-honesty]]：不改真实结果、也不用我们的复现去反驳它，两头都诚实。

### 用户决策（2026-07-27）
- **合作组配合度 = 只给信息**：可提供 t0/K 取值 + 下游分类器架构/训练/trial 聚合细节，但不跑新条件。→ raw/subject-agnostic/更强基线全部落到可复现的 §5；下游细节按其提供的信息写入。
- **第二数据集 = 补一个完整 MI 数据集**（回应 R1-Q4）。**但见下方 C-0 去风险前置步骤**——直接跑有"在新数据集上自我证伪"的风险。

### 🔴 新洞察：下游复现差距几乎肯定在"分类器"而非"去噪器"
- 我们的 M7 下游复现在 BCI-IV-2a 上 **within-subject 对角仅 0.34**（4 类 chance 0.25），合作组是 **0.85**。去噪器再差也不该把 within-subject 打到 chance——**差距来源是下游 EEGNet 的训练/协议**（欠拟合/协议不同）。
- 后果：**若不先修分类器就拿现有 harness 跑第二数据集，极可能同样 near-chance，反而在新数据集上推翻跨被试主张。**
- 解药正是"合作组给信息"：拿他们的分类器细节 → 对齐我们的 harness → 先在 BCI-IV-2a 复现其模式（验证闸门）→ 再上第二数据集。见 C-0。

### ✅ 风险 2（已解决）：30 页提交版 `.tex` 源已到位
- 用户提供的 `Subject_Aware_..._.zip` 已解压到 **`taas_submission/submitted_src/`**：真源 `main.tex` + 完整 `sections/*.tex`（method 39KB / experiments 29KB / appendix 11KB）+ 真图 `figures/*.pdf` + README。**所有正文修改在此目录进行。** 仓库顶层旧的 18 页 `main.tex`/`sections/` 作废（勿改）。
- README 自带一份"待办占位符 + 待审计项"，与审稿意见吻合（作者块/DOI 占位符、下游分类器名、subject-correlation 度量定义）。

### 🟡 风险 3（保留）：t0 / K 表述与代码现状一致性
- 提交版 §4.3 明确写了 restoration 强度 $t_0$ 与 reverse 样本数 $K$ "fixed across all source–target pairs"。R1-W6 要取值 + 敏感性。
- 代码记忆（M10）：K-ensembling 证伪（平坦，CC 0.7397 K=1→16）、full-gen 优于 t\* warm-start。**敏感性可给**，但写作须诚实：K 在我们的实现里无增益。取值须向合作组核对。

---

## 二·补：下游显著性结果（回应 R1-W5，已算好可直接入稿，未触碰任何结果）

用**已发表的 Table 2/3 的 81 个格子**做配对检验（`scripts/revision/sig_downstream.py`）：

| 比较（SADDPM − ICA） | n | 均值Δ | 95% CI (bootstrap) | 配对t p | Wilcoxon p | 效应量 |
|---|---|---|---|---|---|---|
| 全部 source–target | 81 | +2.25 | [+1.08, +3.54] | 0.001 | 0.001 | dz +0.40 |
| **off-diagonal 跨被试** | 72 | **+2.63** | [+1.34, +4.03] | <0.001 | <0.001 | dz +0.45 |
| 对角 within-subject | 9 | −0.83 | [−2.64, +1.13] | 0.44 (n.s.) | 0.43 | dz −0.27 |
| per-target 均值 | 9 | +2.25 | [+0.92, +3.66] | 0.016 | 0.020 | dz +1.01 |
| **SD 比（稳定性）** | — | 0.70 | [0.57, 0.97] | — | — | 排除 1.0 |

- 论文三大下游主张（总体 +2.25、**跨被试 +2.63**、**方差更小**）**全部统计显著**；唯一 n.s. 的是"对角略降 −0.83"，这恰好与"我们改善跨被试、不牺牲 within-subject"的叙事一致，诚实且有利。
- 印刷的 Mean 行、Table 4 聚合值均被这些数字精确复现（README 里担心的 s6/(s7,s3) 在此版已一致）。
- 入稿动作：Table 4 加一列显著性（p / CI），正文加一句"跨被试增益显著（paired Wilcoxon p<0.001），稳定性提升显著（SD 比 95%CI 排除 1）"。**注意 SD 用 population std(ddof=0) 得 4.76/3.35，与稿中一致；ddof=1 得 5.05/3.55——保持与稿一致用 ddof=0。**

---

## 三、逐条修改计划

按**工作量类型**分三档，便于排期：
- **A 类 = 纯写作/图表修正**（零新实验，1–2 周内可做完）
- **B 类 = 已有数据，只需分析+作图+写作**（数据已在 `results/`，不必重跑）
- **C 类 = 需要新跑实验**（重训变体 / 新评测，走 SLURM，是工作量主体）

### A 类：纯写作 / 图表（建议第 1 周做完，先易后难建立进度）

| 审稿意见 | 动作 | 位置 |
|---|---|---|
| **R2-W1 / Q1，AE**：Fig 1 把分支输出标成信号分量 $x_{t-1}^c,x_{t-1}^s$，但 Eq.(7)(11) 定义为噪声分量 $\epsilon_\theta,\epsilon_\phi$ | 二选一统一：把 Fig 1 的两个输出框改标为 $\epsilon_\theta,\epsilon_\phi$、$\oplus$ 改成"预测噪声之和"，并在图注说明由 $\epsilon$ 反推 $x_{t-1}$；或全文改用信号-分量参数化。**推荐改图**（改动小）。 | `sections/fig_architecture.tex` |
| **R2-Q2，AE**：要 SADDPM vs SADDPM-Cond 的紧凑对照表 | 新增一张表（下方"附：可直接用的对照表"已起草），列 input / target / training objective / inference / 用于哪个实验 | 新增于 §3.6 |
| **R2-W3**：部分理论是标准 DDPM 事实，占正文太多 | 把 Prop 3.2（子类包含 subject-agnostic）、Prop 3.3（噪声误差控制反演误差）、Lemma 3.8 移入附录，正文只留 Prop 3.1 / 3.4 / 3.6 / 3.7 的陈述 + 一句直觉 | §3.5 / §3.7 → 附录 |
| **R3-W2，AE**：transductive（非 zero-shot）要在 abstract + conclusion 更醒目 | Abstract 补一句 "adaptation to a new subject is **transductive**: it requires a small amount of **unlabeled** target EEG to estimate the subject embedding（no labels/clean targets/artifact annotations needed）"；conclusion 已有一句，强化为显式 limitation | Abstract、Conclusion |
| **R2-Q10**：删占位符 | 删/替换 `\acmDOI{XXX}`、`\acmArticle{0}`、`\acmVolume/Number`、`\author{First A. Author...}`、`\received{XX Month 2026}` 三行 | `main.tex` |
| **R1-Q5**：Table 5 绝对相关值很低，如何做生理解释 | 加一段：绝对值低是因为 (i) per-window z-score 去掉了幅值信息，(ii) trial 平均衰减了非锁时成分，残留的是被试特异的**谱形/空间模式**；关键是**对角 > 同行非对角**（相对结构），而非绝对量级。稿中已softened为 "consistency check, not biometric"，再补生理归因即可 | §4.4.2 |
| **R2-Q9 / R1-Q1**：要完整架构/训练/adaptation/inference 超参 + 下游分类器细节 | 汇总现有配置（`RESULTS.md` §12 Assumptions Ledger 全都有）成一张附录超参表；下游明确写 EEGNet-8,2（F1=8,D=2,F2=16,kern=64）、输入=2s 去噪窗、训练 schedule、trial 级聚合规则 | 附录 + §4.3 |
| **R3-Q5 / W4(部分)，AE**：embedding 隐私/本地存储的说明 | 扩写 Ethics & Privacy：embedding 本地存储、不随模型分发、可加噪/丢弃；实证部分（能否 re-id）见 C 类 | Ethics 段 |
| **R1-Q3(部分)**：EMG 为何更差 | 写作层面先给机制解释（EMG 宽带、与神经频段重叠更严重，条件信息不足以定位），并指向已有的 Fig 6/9/10 谱图证据 | §5.1 |

### B 类：已有数据，补分析+作图（建议第 2–4 周；依赖风险 1 的决策）

| 审稿意见 | 动作 | 数据来源 |
|---|---|---|
| **R1-W5，AE**：显著性检验 + 置信区间 | 下游：跨 9 被试对 SADDPM vs ICA / vs raw 做**配对 Wilcoxon / bootstrap CI**（用 9×9 矩阵的 per-target 或 per-pair）；对"低方差"claim 做 Levene/Bartlett 或 bootstrap 方差比 CI。配对 GT：对 per-window CC 做配对检验 + CI | `results/m7/*.csv`；`results/m8…m12/*.csv`（per-window CC 已存） |
| **R1-W6**：t0/K 取值 + 敏感性 | 用 M5 的 t\* sweep（50/100/200/400/600）+ probe 的 K-ensembling（K=1…16）画敏感性曲线；诚实标注 full-gen 是最终选择、K 无增益 | `RESULTS.md` M5/M9/M10；`scripts/probe_ensemble.py` 产出 |
| **R2-Q6 / R3-Q4(部分)**：content 表征能否被 post-hoc 分类器还原被试身份（验证 $\mathcal{L}_o$ + 隐私） | 在 $z_c$ 上训一个 probe 分类器，报准确率（期望 ≈ chance 若 orthogonality 生效）；对比 $z_s$（ArcFace 已 0.937） | M4 checkpoint + 现有 probe 机制 |
| **R2-W5 / Q8**：embedding 空间可视化 + adaptation 是否移向源簇 | 对 9 个学到的 e(s) 做 PCA/t-SNE；叠加 transductive adaptation 的轨迹，看目标 embedding 是否落入/移向源被试簇 | M4 embedding 表 + adaptation 输出 |
| **R1-Q3**：EMG 失败模式（配证据） | 用已有 EMG 谱图（Fig 6/9/10）+ 分 SNR 的 RRMSE 曲线，展示高频残留 | `results/m8,m10/` EMG CSV |

### C 类：需新跑实验（建议第 3–10 周并行，SLURM 主体工作量）

按"性价比 / 审稿人权重"从高到低排：

0. **【前置闸门】对齐下游分类器并在 BCI-IV-2a 复现合作组模式 — 第二数据集的去风险前提**
   - 拿合作组给的下游分类器架构/训练/trial 聚合 + t0/K，**对齐我们的 EEGNet harness**，目标：在 BCI-IV-2a 上复现（或逼近）其 within-subject≈0.85、跨被试 +2.6 的模式。
   - 这是**闸门**：通过 → 我们的下游 pipeline 可信，第二数据集才有意义；不通过 → 说明我们的 SADDPM 去噪器与合作组差异过大，第二数据集改走"配对 GT 去噪验证"或降级 limitation。
   - 诊断重点：当前 M7 对角仅 0.34≈chance，几乎必是分类器欠拟合/协议问题（去噪器不会把 within-subject 打到 chance）。

1. **系统消融（重训变体）— R2-W4/Q3/Q4/Q5，AE（权重最高，三人都提）**
   一次训练一组变体，one-factor-at-a-time：
   - 去 individual/residual 分支（R2-Q3）
   - 去 orthogonality loss $\mathcal{L}_o$（R2-Q4）
   - 去 subject-identification loss $\mathcal{L}_a$，且在 multi-source 设置下（R2-Q5）
   - 去 FiLM（改为纯 concat/加法）（AE）
   - 去 subject embedding（= subject-agnostic，兼作风险 4 的基线）
   汇总成一张消融表（指标：配对 GT 的 CC + 下游）。**部分已有半成品**：M3（仅 FiLM、单 decoder）vs M4（全）可复用，但需补成干净的单因子表。

2. **更强 / 缺失基线 — R1-W1/W2/W4/Q2，AE（主要落在可复现的 §5，不动 Table 2–4）**
   - **subject-agnostic DDPM**（R1-W2）：null-embedding / 去 embedding 版，在**配对 GT §5** 上给（Table 8 的 null 已是雏形，补一个正式去 embedding 重训版）。
   - **更强去噪基线**（R1-W1）：§5 已有 SimpleCNN/ComplexCNN/NovelCNN；下游侧 ICA 作经典参照保留。
   - **raw / 常规预处理 EEG**（R1-W4/Q2）：§5 的 "Noisy(input)" 行即 raw floor；若前置闸门(C-0)通过，可在对齐后的下游 harness 上加一条 raw 下游行作为补充（**独立呈现，不并入 Table 2–3**）。
   - 复用 M6/M7 harness + M8 CNN。**注意**：这些下游侧新行只在 C-0 闸门通过、且明确标注为"我们的独立复现 pipeline"时才加，避免与合作组 Table 2–4 混淆。

3. **超参敏感性 — R2-Q7，AE**
   扫 embedding 维度、$\lambda_o$、$\lambda_a$、$\lambda_e$（若有）、adaptation 步数。可用小网格 + 固定种子，报曲线。

4. **Adaptation 过程分析 — R3-W1/Q1/Q3，R1-W3**
   - 目标数据量 sweep（多少 unlabeled target windows 才稳定，R3-Q1）。
   - **去掉 target-embedding adaptation** 的消融（用固定源 embedding / 均值 embedding vs adapted，R1-W3）——注意 Table 8 是**推理期换 embedding**，不等于"去掉 adaptation 过程"，这条要单独做。
   - adaptation 前后对**下游分类器**准确率的影响（R3-W3）。

5. **推理延迟 + 少步采样 — R3-Q3，AE**
   计时 per 2s 窗（full-gen 步数）；扫 DDIM 步数（如 1000/200/50/20）看 CC-vs-步数、延迟-vs-步数曲线。轻量但需实测。

6. **隐私：re-identification / membership inference — R3-Q4，AE**
   用学到的 embedding / 去噪输出做被试 re-id 与 membership inference 攻击，量化泄露；与 A 类的 Ethics 写作呼应。ArcFace 0.937（真实信号可辨识）是重要背景。

7. **第二个完整 MI 数据集 — R1-Q4（用户已决定投入；受 C-0 闸门约束）**
   - **候选**（推荐度）：**High-Gamma**（14 subj, 128ch, 同 4 类 MI——任务最匹配、不同 montage，直接回应"另一 montage"）；或 **PhysioNet EEGMMIDB**（109 subj, 64ch——跨被试 N 最大、协议不同，但类别定义不同）；**BCI-IV-2b 不推荐**（仅 3 通道双极，ICA 基线会退化，且 2 类）。
   - **流程（gated）**：C-0 闸门通过后，用对齐好的 harness 在新数据集上跑 SADDPM vs ICA (+raw/subject-agnostic)，报跨被试模式是否泛化（同 Table 4 口径 + 显著性）。
   - **备选（若 C-0 不过）**：在新数据集/新 clean-EEG 源上做**配对 GT 去噪**验证（我们强项，CC 稳），泛化 §5 而非 §4；或降级为 limitation。
   - 用哪个数据集到该阶段再定；我推荐 High-Gamma（任务/类别与 2a 最可比）。

8. **resting-state adaptation — R3-Q2（选做/讨论）**
   若 BCI-IV-2a 有静息/试次间片段可用，做一版仅静息 adaptation；否则作为讨论回答。

---

## 四、已在稿中、回复信里"指出即可"的点（省力）

这些审稿人要的东西**稿件已有**，回复信直接引用页/表号：
- SADDPM-Cond 的 x0-prediction 有对照支撑（R2-Strength 6）→ Table 6 + Lemma 3.8。
- EOG **和** EMG 都评了（R1-Strength 5）→ Fig 6–10（附录 EMG）。
- 多通道联合去噪的价值（R1-Strength 6）→ Table 7 + Prop 3.7。
- embedding 的 correct/wrong/null 消融（部分 R1-W3、R2 意图）→ Table 8 + Fig 5（但见 C-4 的区别说明）。
- decorrelation ≠ 独立性（R2-Strength 5）→ 附录 Prop 3.4 证明后一句。
- viability-not-superiority 的 scope 声明（R2-Strength、诚实性）→ §5.3 Scope 段。

---

## 五、建议时间线（截止 2026-11-21，约 16 周）

| 阶段 | 周次 | 内容 |
|---|---|---|
| Phase 0：A类+并行索取 | 1–2 | 做完全部 A 类写作；**同时向合作组索取分类器细节 + t0/K**（第二数据集的关键依赖，越早越好） |
| Phase 1：B类 | 2–4 | 显著性（已完成雏形）、t0/K 敏感性、content-probe、embedding 可视化——都用现有数据/checkpoint |
| Phase 2a：C-0 闸门 | 3–5 | 拿到合作组信息后对齐下游 harness，在 2a 复现其模式（**决定第二数据集怎么走**） |
| Phase 2b：C类实验 | 4–11 | 消融 fleet → §5 基线补全 → 超参扫描 → adaptation 分析 → 延迟 → 隐私 →（闸门过后）第二数据集（SLURM 并行，走 preregistered-experiments 流程） |
| Phase 3：整合 | 11–14 | 结果入正文/表/图，逐条写 response letter，在 `submitted_src/` 重编译 |
| Phase 4：缓冲 | 14–16 | 复核、润色（R1/R3 说 light、R2 说 moderate）、11-21 前提交 |

关键路径：**合作组信息 → C-0 闸门 → 第二数据集**，尽早启动索取。C-1（消融）与 §5 基线补全是评分关键，与闸门并行推进。

---

## 六、response letter 结构建议

- 开头：感谢 + 一段总述（"我们补充了系统消融、subject-agnostic/raw/更强基线（配对 GT 协议）、**下游配对显著性检验与置信区间**、超参与 adaptation 敏感性、embedding 可视化与隐私分析、延迟/少步采样，并修正了图-公式一致性、精简了标准理论、显著化了 transductive 定位"）。
- 按 **AE → R1 → R2 → R3** 顺序，每条 comment 用 **"Comment / Response / Changes (页/表号)"** 三段式。
- **下游表（Table 2–4）保持不变**：对 R1-W5 的回应是"我们在**已报告的**跨被试结果上做了配对显著性检验（off-diagonal +2.63, Wilcoxon p<0.001；稳定性 SD 比 95%CI 排除 1）"，把统计当作对现有结果的**加固**，而非改数。对 R1-W1/W2/W4/Q2 明确说明：下游以 ICA 为经典参照，学习型/subject-agnostic/raw 的系统比较在直接保真度协议 §5 给出。
- 结尾附一张"修改索引表"（comment ↔ 新增表/图/节）。

---

## 附：可直接用的 SADDPM vs SADDPM-Cond 对照表（回应 R2-Q2）

| 维度 | **SADDPM**（生成式复原） | **SADDPM-Cond**（条件去噪） |
|---|---|---|
| 网络输入 | 扩散态 $x_t$（+ subject embedding，FiLM） | $[x_t\,;\,y]$ 沿通道拼接观测 $y$（+ subject embedding，FiLM） |
| 学习目标 | $p_\theta(x_0)$：干净 EEG 的先验分布（$\epsilon$-prediction，dual-decoder + $\mathcal{L}_r/\mathcal{L}_o/\mathcal{L}_a$） | $p(x_0\mid y,s)$：给定含噪观测的后验（**$x_0$-prediction**） |
| 训练数据 | 干净 EEG + subject 标签（无需配对） | 合成配对 (corrupted, clean)，SNR 受控 |
| 推理 | SDEdit：前向噪化到 $t_0$，再 subject-conditioned 反演到 0 | **Full conditional generation**：从纯噪声起，每步条件于 $y$（Palette/SR3） |
| 用于的实验 | 下游 MI 分类 vs ICA（Table 2–4）、subject-correlation（Table 5） | 配对 GT 去噪（Table 6–8，Fig 2–10） |
| 定位 | 无配对 GT 时的间接评测 + subject-aware 机制载体 | 有配对 GT 时的直接去噪保真度评测 |

> 一句话总结（可放正文）：*SADDPM-Cond 不是另一个模型，而是把同一 backbone/embedding/FiLM 从"先验采样式复原"切换为"条件后验去噪"，以便在存在配对 ground truth 时直接测量去噪保真度。*
</content>
</invoke>
