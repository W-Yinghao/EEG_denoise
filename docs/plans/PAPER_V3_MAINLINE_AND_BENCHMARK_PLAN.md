# 论文主线重组方案（v3）+ Benchmark 与相关工作对比选择

**日期**: 2026-08-29 · **基底**: `positive_submission_v2/`（《The Operator Carries the Subject》，ACM TAAS 格式）
**定位要求**（操作者指令）: 正向方法论文；不做审计叙事；不否定 diffusion 方法族；主线 = **subject-aware EEG diffusion denoising**。
**数据事实来源**: `docs/EEG_denoise_arc_results_digest.md`（权威结果记录）+ T1–T6 补充实验（`RESULTS_PAPER_FINAL.md`，本次运行）。

---

## 0. 三条稿件线的关系（先厘清，避免混淆）

| 稿件 | 状态 | 数据 | 处置 |
|---|---|---|---|
| (a) TAAS-26-0171《Subject-Aware Diffusion Models for Cross-Subject EEG Denoising》(SADDPM) | **Major Revision**，AE Ziyu Jia，截止 2026-11-21，三位审稿人积极（R2 best-paper=Yes） | BCI-IV-2a + EEGdenoiseNet | 独立按 REVISION_PLAN 修；协作方 Tables 2–4 不动；本方案不覆盖它，但 §6 说明两篇如何互引不互撞 |
| (b) 8月4日 CGDR 草稿（Klados/SGEYESUB 子空间 diffusion） | 已被程序演进超越（H2 scoped negative） | Klados | 仅作内部历史，不投稿、不引用 |
| (c) `positive_submission_v2`《The Operator Carries the Subject》 | 完整 v2 草稿，图为占位符 | Eye-BCI(46ch) 主 + EEGEyeNet 支撑 | **本方案的对象**：重组为 v3 正向主线，T1–T6 填图表 |

(a) 与 (c) 是同一研究纲领的两篇论文：(a) 证明 *训练期* 的 subject-aware diffusion（embedding 条件化）在跨被试下游任务可行；(c) 证明 *推理期* 的 subject-aware diffusion（标定算子引导）在未见用户上带来大幅重建增益 + 校准的不确定性。两篇互为引用、不同数据集、不同贡献，无重复发表问题。

---

## 1. v3 主线（一句话）

> **一个人群共享的 diffusion 先验，在推理时被"测量得到的被试"唤醒：120 秒标定给出该用户的 EOG→EEG 传播算子，算子引导采样、算子的后验不确定性流入采样分布——于是同一个网络对每个新用户都是 subject-aware 的：更准的重建、经验覆盖达标的预测区间、以及每一环都被测量过的自适应回路。**

关键词序保持 "subject-aware EEG diffusion denoising"：diffusion 是**主角**（承载分布、吸收引导、输出不确定性），标定算子是**它变得 subject-aware 的机制**。v2 已基本是这条线；v3 的工作是 (i) 用 T1–T6 补齐证据环，(ii) 把两处"削 diffusion"的表述改为分工表述（§3），(iii) 把 S356 从"上界注记"升格为"subject-aware 条件化的普适性"一节（§4）。

## 2. v3 章节骨架（相对 v2 的差异用 ▲ 标注）

1. **Introduction** — 同 v2 前三段；第四段结果预告更新为含 held-out UQ（T1）与标定时长曲线（T2）。贡献四条改为：
   C1 方法：标定算子引导的人群 diffusion 去噪器（ridge + EB shrinkage + 可靠性门 + 引导采样），零个人化训练；
   C2 证据：受控消融（T3 完整 2×2 + unshrunk 臂）+ 单次冻结 held-out 复现；
   C3 自适应回路：标定时长曲线（▲T2）、算子寿命/重标定时钟（▲T4）、可靠性门的实测 fail-safe、在线 RLS 对照——完整的 monitor→analyze→plan→execute 回路，每环有数；
   C4 不确定性：算子后验注入采样 → 名义覆盖 + 更优 CRPS，**held-out 上单次确认**（▲T1，闭合 v2 结论里自留的缺口）。
2. **Related Work** — 四轴组织（§5 对比清单）：参考信号线性/自适应算子族；无参考子空间族（ICA/ASR）；人群深度/diffusion 去噪器族；跨被试迁移与标定缩减。定位句：本文位于"算子族 × diffusion 族"的交点，用迁移学习的收缩统计连接两者——检索确认该交点无先例（§5.4 novelty 措辞）。
3. **Method** — 同 v2 三小节；▲ 增补 3.4 "The adaptation loop"（半页）：把 shrinkage/gate/fallback/recalibration 显式画成回路（TAAS 读者的 MAPE-K 语汇），并给出 T2/T4 所测的两个"回路常数"：标定楼层（60 s，设计使然且实测）与重标定时钟（位移曲线拐点）。
4. **Experiments** — 同 v2；▲ 比较条件表加三行：ICA(+EOG-corr)、ASR(120 s 标定)、SGEYESUB 式标定子空间投影（同一 120 s 标定，闭式）——"literature anchor rows on the same protocol; context, not contests"。
5. **Results** —
   5.1 Paired restoration（v2 保留）+ ▲T3 消融矩阵表（单元格 (matched,matched) 点亮；unshrunk 臂给出 shrinkage 的贡献）。
   5.2 Natural + specificity + online（v2 保留）+ ▲T5 五条件 operating-point 图（fig:natural A 实装）。
   5.3 ▲ NEW "Calibration economics"：T2 时长曲线（30 s 被门拒→系统读数即回退；60 s 楼层；120 s 收益）+ T4 寿命曲线（位移随时间增长 0.86→1.43 + gain-by-third）→ 结论：两分钟买到的算子有明确的有效期，回路以重标定为节拍。
   5.4 Predictive intervals（v2 保留）+ ▲T1 held-out 表：dev 冻结温度（INFL 2.40）原封应用于 8 名未见用户的覆盖/CRPS/风险-覆盖——UQ 章从 development-only 升为 held-out 确认。
   5.5 Where the calibration information lives（S356，升格，§4 框架）。
6. **Conclusion** — 移除"interval analysis is development-only"限制（T1 已闭合）；保留 sealed-55 作为"进一步确认已预留"的一句（不承诺时间）。

图：fig:overview（三段系统图）/ fig:denoise（T6.1 波形 + 每被试线）/ fig:natural（T5 五条件平面 + RLS 曲线）/ fig:uq（覆盖曲线 + CRPS/RC + S356 平坦曲线）/ ▲fig:loop（T2 时长曲线 + T4 寿命曲线合板）。可选 ▲ scalp map（T6.2）并入 fig:denoise。

## 3. 两处"削 diffusion"表述的正向改写（数字不动，框架换位）

**v2 原文 A**（5.1 末）"The point-estimate gain of this paper therefore comes from the calibration, not from the sampler."
**v3 改写**："On point error the guided sampler matches the strongest calibrated point estimator (0.4310 vs 0.4353) while running the same frozen network in guided, unguided, population and fallback modes — and it is the only estimator in the table whose output is a conditional distribution. The calibrated operator sets the operating point; the diffusion prior makes it robust and probabilistic."（分工陈述：算子定点，先验供分布与鲁棒性。C05 competitive 措辞保留，无 superiority 声明，不越诚实线。）

**v2 原文 B**（Conclusion）"as a point estimator the calibrated diffusion model is on par with calibrated linear regression, so the case for the sampler rests on its predictive distribution."
**v3 改写**：直接给分布侧战绩收束："the sampler's distribution beats a deterministic ensemble where distributions matter (CRPS 0.1529 vs 0.1548; risk–coverage 0.0920 vs 0.1129; conformal 80% coverage 0.765 vs 0.652), and the physically-motivated operator-posterior width is both sharper and lower-temperature than a pure temperature fix."

**另一个可以正向吸收的点**（来自 F1，v2 已隐含）：坏引导下学习到的修正项会部分吸收误差（rejected 细胞上撤引导仅 +0.0614、CI 过零，而线性通路 +0.2537）——v3 表述为 "the learned corrections give the diffusion pathway a robustness margin the linear pathway does not have"。这是 diffusion 族的正向性质。

**禁区不变**（cleanslate 措辞规则 + manuscript-honesty 备忘）：不重演 DS-DDPM；退化 oracle 数字不进图表；Dev/Sealed 两列永不合并；不写 "best/superior"。

## 4. S356 的升格：subject-aware 条件化的普适性（连接 (a) 与 (c) 的桥）

v2 把 S356 放为"上界注记"。v3 升为一节正向结论，三句话：
1. **信息而非接口**：同一份 120 秒标定信息，经算子接口（本文主方法）或经 support-fitted 32-d 表示接口（S356，deterministic 网络）都产生被试特异的收益（+0.0608 [0.0406,0.0835]，14/14；own−wrong +0.1623）。
2. **与训练池规模无关**：30→259 名训练被试收益平坦（−0.0224 [−0.0491,+0.0158]）——扩大人群模型既不产生也不消除标定个体的必要性。
3. **对 diffusion 族的正向含义**：subject-aware 条件化（如 DS-DDPM、SADDPM 的 embedding 思路）是有效的设计空间，**条件是条件量被目标用户的标定所拟合**；未标定的身份码不携带信息（这是范围陈述，不是否定）。此句同时把论文 (a) 的 embedding 路线纳入同一理论框架——两篇论文由此互证而非互斥。

## 5. Benchmark 选择与相关工作对比（用户核心问题）

### 5.1 论文内评测组合（推荐配置）

| 层 | 数据/协议 | 角色 | 状态 |
|---|---|---|---|
| 主战场 | **Eye-BCI (MobileBCI)** 46ch+EOG，paired-episode RRMSE + natural 三端点，15 dev / 8 held-out 单次冻结 | 贡献级协议：真实信号、subject-wise、封存单开——严于全部已检索去噪文献（该纪律本身写成 evaluation 贡献） | 已有 + T1–T6 |
| 同协议参照行 | linear EOG regression（0.4353）、DET twin（0.4968）、RLS 在线、**ICA(续)、ASR、SGEYESUB 式子空间**（本次 CPU rows） | "context rows, not contests"——回应 TAAS-26-0171 审稿人"要更强 baseline"的同款诉求 | CPU rows 运行中 |
| 社区锚 | **EEGdenoiseNet** EOG(+EMG) CC/RRMSE 表（SADDPM-Cond 0.901/0.988 vs SimpleCNN 等），附录 | 可比性锚点；引 arXiv 2606.08594（基准饱和 + RRMSE-效用鸿沟）与 2509.14665（task-oriented 转向）把 "≈ICA/CNN" 框成饱和基准上的预期行为 | 已有（M8–M11 数字） |
| 支撑实验 | **EEGEyeNet** S356（30–259 训练池） | §4 普适性；检索确认无人用 EEGEyeNet 做过去噪→协议自证 | 已有 |
| 预留 | EEGEyeNet **sealed-55**（chmod 000 未开） | 论文一句话："a sealed 55-participant block is reserved for confirmatory evaluation"（审稿人问外部队列时的王牌，开封需另行预注册） | 冻结中 |
| 不推荐进主线 | Klados（污染模型本身就是线性传播矩阵，结构性利好我们的 ridge 算子——引用披露此 circularity；transport 腿 +0.016 属旗舰纲领，破坏"one method"纯度）、TUAR（检测语料无干净参照，仅引用划界） | | |

**如审稿人索要第二真实队列**：优先顺位 = SGEYESUB 69-被试公开数据（同为标定范式，Kobler 发布）> Klados（披露 circularity 后作 operator-recovery 验证）> 开封 sealed-55（需操作者批准 + 预注册）。

### 5.2 相关工作必比/必引清单（四轴）

**轴1 参考信号算子族**（我们的血统）：Gratton 1983（propagation factor 起点）· Croft & Barry 1998/2000（传播系数随类型/被试变化→标定动机）· Schlögl 2007（自动回归基线）· He 2004 RLS / Kilicarslan 2016 H∞（自适应族；我们的 RLS 对照直接回应）· **SGEYESUB Kobler 2020**（最近亲缘：标定→线性算子；已实现为参照行；差异=shrinkage/生成先验/门控/UQ）· **ARMBR 2025 JNE**（15 秒最小标定线性算子，代码公开——最新亲缘，**建议至少引用对比，理想加为 baseline**；漏引会被审稿人抓）。
**轴2 无参考子空间族**：ICA+**ICLabel**（点名管线，Pion-Tonachini 2019；ADJUST/MARA 作 lineage）· **ASR** Mullen 2015 + Chang 2020（标定数据依赖→正当化我们 120 s；ASR 静默失败 vs 我们显式 withhold——差异句）。
**轴3 人群深度/diffusion 去噪器族**：EEGdenoiseNet 基线 CNN 群 + EEGDnet/GCTNet/EEGDiR/DTP-Net/DenoiseMamba（引用定位"架构竞赛已饱和"）· IC-U-Net/**ART 2025**（多通道真实数据 proxy-endpoint 方法论先例——为我们的 natural 三端点背书）· **EEGDfus**（已实现的 diffusion 锚；官方单通道复现 RRMSE 0.2965 为合法引用数字）· **D4PM 2025**（最新 diffusion EOG SOTA、代码公开——**必须显式处理**：最少 cite+scope"population-level、segment 级切分、无参考通道、无标定、无 UQ，与我们正交"；有余力在 EEGdenoiseNet 附录跑它）· **DS-DDPM**（唯一 subject-aware diffusion 先例——专段对比：训练期学被试 vs 推理期测被试；按 cleanslate 规则不做内部复现）· **EEGOAR-Net 2025**（calibration-free 对立面——必答"2026 年为何还要标定"：subject-specific 算子 +0.16 own-vs-wrong、可靠性门、coverage-verified UQ 三样 calibration-free 给不了）· DeScoD-ECG/SDEMG/CDiffuSE/SGMSE+（谱系）· UDNet TNSRE 2023（唯一 uncertainty-aware EEG 去噪——内部启发式 vs 我们 coverage-verified，novelty 划界）。
**轴4 迁移/标定缩减 + UQ 谱系**：Jayaram 2016（EB shrink-to-population 在 BCI 解码的先例——诚实署源）· DDRM/DPS（已知算子）与 BlindDPS/GibbsDDRM（未知算子）夹出我们"可测算子"的位置 · CSDI/SSSD（K 样本 CRPS 惯例）· Im2Im-UQ/K-RCPS/Conffusion/PUQ/QUTCC（conformal 修复 UQ 谱系）· CF-RNN（医疗时序 conformal）· **BIPSDA 2025**（diffusion 后验不天然校准——预答"samples≠posterior"质疑；我们答案=经验覆盖验证 + 可选 split-conformal 包装）。

### 5.3 可防御的 novelty 措辞（检索核实至 2026-08）

1. "To our knowledge, the first **EOG-reference-guided diffusion sampling** for EEG artifact removal."（最近先例=pose-conditioned EEG-DDM 预印本、PPG→ECG 的 RDDM；均非 EOG、非算子。）
2. "The first **subject-calibrated artifact operator with empirical-Bayes shrinkage toward a population prior** used to guide a shared generative model."（算子 40 年血统如实署源；shrinkage 先例仅在解码迁移。）
3. "The first **coverage-evaluated predictive intervals** in EEG artifact removal"（UDNet 无 interval 无 coverage；conformal×EEG-denoising 检索为空），held-out 确认（T1）后可写 "validated on unseen users"。
4. 评测纪律（subject-wise + frozen single-pass + sealed reserve）作为 named contribution。

### 5.4 审稿人预答火力表（新增实验即答案）

| 可预期质疑 | 答案（已在/将在论文内） |
|---|---|
| 为何不比 SOTA 深度去噪器 | EEGdenoiseNet 附录表 + 饱和引证 + 贡献正交声明（校准/UQ，无一 baseline 可被打分） |
| 为何要标定（EEGOAR-Net） | own-vs-wrong +0.1623；withhold-gate 实测；calibration-free 无 UQ |
| 标定成本多大/多久失效 | T2 曲线（60 s 楼层）+ T4 时钟（位移 0.86→1.43） |
| diffusion samples ≠ posterior | BIPSDA 引证 + 经验覆盖为验证 + 温度在 dev 冻结、held-out 原封（T1） |
| n=8 太小 | 冻结单次协议 + 15/15 dev + sealed-55 预留声明 |
| ICA/ASR 呢 | 本次 CPU rows，同协议 |

## 6. venue 判断（供决策，不代决）

- (c) Operator 论文草稿现为 TAAS 格式。**事实**：TAAS 常规卷 2024–2026 零生物信号论文；对口的 BCI 特刊（客座编辑 Ziyu Jia——恰为 (a) 的 AE）投稿窗 2026-05-30 已关。**选项 A**：联系客座编辑问特刊二批/延期（最低成本探路）；**选项 B**：JNE（文化最合：ARMBR 先例证明"≈ICA 但更便宜/更可控/带 UQ"可发；需加 ARMBR+ICLabel 行与一个 downstream 端点——EEGEyeNet gaze 解码现成）；**选项 C**：TNSRE/JBHI（会被按 6–9 个深度 baseline 的表要求，对 honest-parity 定位最不利，不推荐）。建议 A 探路、B 为主计划；论文主体两者通用，仅 framing 段落切换（TAAS=自适应回路语汇，JNE=神经工程语汇）。
- (a) SADDPM 修回照 REVISION_PLAN 独立进行（11-21 截止），与本方案不冲突；两文互引框架见 §4。

## 7. 执行清单（本次会话已在跑的部分打 ●）

- ● T1–T6 + CPU rows（SLURM，`results/paper_final/` + `paper_final_arrays/`）
- ● T5 五条件 natural 表（已完成，零 GPU：存储行聚合）
- v3 章节改写（§2–§4）：待 T1/T2/T3/T4 数字落地后写 `positive_submission_v3/`
- 参考文献补录：ICLabel/ASR/SGEYESUB/ARMBR/EEGOAR-Net/D4PM/DS-DDPM/UDNet/DeScoD-ECG/CSDI/SSSD/Im2Im-UQ/K-RCPS/Conffusion/BIPSDA/Jayaram/DDRM/DPS/BlindDPS/GibbsDDRM/Gratton 等（v2 bib 已有约半数）
- 修 bib 脏条目（zhao2023ecg 元数据冲突、lugmayr2022mri 键错内容——见 taas_submission 审计）
