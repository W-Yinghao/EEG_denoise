# Downstream-Utility Wave (D-wave) — 设计稿（待批准，未消耗 GPU）

**动机**（操作者指令 2026-08-29）：原 SADDPM 论文（TAAS-26-0171）的主证据就是间接指标
（BCI-IV-2a 分类 52.8% vs ICA 50.6%）；2025–2026 文献（arXiv 2606.08594 的
metric-utility gap、2509.14665 task-oriented、ART/NeuroImage 2025、JNE 2026 综述）
都把 downstream 验证列为去噪论文的预期。Operator 论文目前只有 paired RRMSE +
natural 代理端点，缺一个任务效用块。本设计补齐它。

---

## 0. 先行发现（与本 wave 无关也必须修）：v2 草稿数据集引用错误

`/projects/EEG-foundation-model/mobile_bci/dataset_description.json` 证实主数据集是
**Ambulatory EEG Study**（Lee, Shin, Lee & Lee；Korea University IRB
KUIRB-2019-0194-01；CC0；ref DOI 10.1109/TNSRE.2020.3040264；24 名被试、
32 头皮 + 14 耳周 EEG + 4 EOG、ERP/SSVEP 两种范式、4 种步行速度对应不同 session）。
**v2 草稿 04_experiments.tex 引用的却是 Guttmann-Flury Eye-BCI（上海交大 IRB
E2021216I）——数据集、引文、伦理声明全部张冠李戴**，v3 必须改为 Lee et al. 数据集
descriptor + KUIRB 伦理声明 + CC0 声明（确切 descriptor 书目在写作时上网核对）。
附带收益：ambulatory（步行中记录）让"真实伪迹环境下的去噪"故事更强，且原数据集
论文自带 ERP/SSVEP 解码基准数值可引。

## 1. 侦察结论（只读，已完成）

- BIDS events 齐全：每个 (sub, ses, task) cell 有 `*_events.tsv`。
- **SSVEP**：60 试次/cell，5 s/试次，3 类刺激（value 11/12/13 = 三个闪烁频率；
  频率值 probe 时从原文献 + RAW 枕区谱峰双重确认）。试次 onset 以样本计
  （@100 Hz，probe 时以谱峰对齐验证单位）。5 s 试次 ≈ 我们 512 样本窗口——天然对齐。
- **ERP**：300 试次/cell，2 类（standard/deviant oddball），0.5 s 事件。
- 排除 onset < 120 s 的试次（标定前缀不相交纪律），SSVEP 每 cell 约剩 ~46 试次，
  ERP 约剩 ~260 试次。

## 2. 设计

### D1 — SSVEP 3 类解码（主任务，分类器免训练）

- **解码器**：标准 CCA（参考正弦 + 2 次谐波，3 个候选频率，argmax 相关）——闭式、
  确定性、零训练、零泄漏。规避原论文"协作方 EEGNet 无法复现（本地 0.34 vs 对方
  0.85）"的教训：主端点绝不用深度分类器。
- **试次窗口**：onset 对齐取 512 样本（5.12 s ⊇ 5 s 试次）。
- **条件臂**（每臂同一窗口、同一 CCA）：RAW / NO_A0（unguided）/ MATCH（系统，
  BINARY_NOA0FB 门控）/ POP / LINEAR(y − C_gated·e) 闭式 / ICA / ASR /
  SGEYESUB-style（后三者复用 CPU-rows 已有实现，CPU）。diffusion 臂 3 个 × ~46
  试次 × 90 cell ≈ 12k 窗口采样。
- **端点**：per-participant 准确率；主对比 (i) MATCH − RAW（no-harm 主张的主检验）、
  (ii) MATCH − NO_A0（引导的下游增量）；次要 MATCH − ICA / − ASR / − LINEAR。
  participant-first，5000 次自助，Holm 校正两个主对比。
- **机制分层（预注册）**：按试次 VEOG drive RMS 三分位分层——预测增益集中在高污染
  三分位；低污染层 MATCH ≈ RAW（不伤害）。
- **速度轴（免费）**：ses-02/03/04 对应不同步行速度，按 session 分列报告。

### D2 — ERP 解码 + ERP 保真（第二任务）

- **去噪方式**：连续记录 50% 重叠 512 平铺、取中段拼接，再切 epoch（−0.2..0.8 s，
  基线校正）。
- **解码器**：shrinkage-LDA（时域降采样幅值特征），被试内 5 折试次 CV（固定种子），
  **AUC** 为主（类不平衡）。同样闭式/浅层。
- **ERP 保真端点**（防过度清理，rASR 式敏感性/特异性对）：各条件 trial-average 与
  RAW 低污染试次 trial-average 的相关 + 峰幅保持率。
- 条件臂与统计同 D1。

### D3（可选，另行决定，不在本批）— EEGEyeNet-dots gaze 解码率下降端点
去噪后 EEG 通道的注视可解码性应**下降**（隐私/去混淆声明；文献 sweep 建议）。
工程量大（montage 移植），二期再议。

### Held-out-8
dev 结果冻结之后，单次通过跑 held-out-8 的 D1+D2（同一冻结协议），沿用
Dev/Sealed 两列纪律。

## 3. 预注册解释网格（冻结于任何 GPU 之前）

| 结果格 | 解读 | 论文措辞 |
|---|---|---|
| MATCH ≥ RAW（CI 下界 > −0.02）且高污染层 MATCH > RAW | 效用主张成立 | "preserves decoding overall and improves it where contamination is high" |
| MATCH ≈ RAW 各层（CI 含 0） | no-harm 成立、增益未证 | "does not trade decoding for cleaning"（引 metric-utility gap 正当化） |
| MATCH < RAW（CI 上界 < 0） | 伤害——如实报 | 报告 + 引 2606.08594 讨论重建-效用错位；不删不藏 |
| ICA/ASR 任一显著优于 MATCH | context row 逆转 | 如实报，"context, not contests" 框架不变 |

## 4. 预算与执行

- **Probe（gate）**：1 个 cell（sub-02/ses-02 两任务）端到端：事件单位核验、SSVEP
  频率谱峰确认、CCA 在 RAW 上的合理精度（应接近原数据集论文水平）、试次数核对。
  Probe 通过才开 fleet（本 skill 纪律）。
- **Fleet**：GPU ≈ 2–3 GPU-h（D1 ~0.7 + D2 ~2；`--time=23:59:59`，按 fold 拆分
  array，resume-safe）；CPU 臂并行。
- **产物**：`results/paper_final/dwave/` + `paper_final_arrays/d1_ssvep.npz`、
  `d2_erp.npz`；RESULTS_PAPER_FINAL.md 追加 D 节。
- **论文落点**：Results 新增 5.x "Downstream utility under ambulation"（表 +
  分层条形图）；Discussion 接 metric-utility-gap 文献。

## 5. 风险与诚实条款

- SSVEP 是枕区信号、眼动伪迹前额为主——总体增益可能小；分层设计正是为此。
- ERP（P300 类）与眨眼时频重叠更大，预期效应更明显；但去噪伤害 ERP 幅值的风险也
  更大——ERP 保真端点两面都看。
- 所有解码器代码随论文发布；不引入任何需要"对方代码"才能复现的端点。
