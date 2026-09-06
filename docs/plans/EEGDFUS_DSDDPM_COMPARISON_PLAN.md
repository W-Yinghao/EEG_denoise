# 对比实验规划：EEGDfus 全表 + DS-DDPM Table I/II（2026-08-31，已开始执行）

授权：操作者指令"能比较的部分全部和他们比较"（EEGDfus）+"复现 Table 1/2，规划后直接提交"（DS-DDPM）。

## 一、EEGDfus（JBHI 2025）逐表对照

| 他们的表 | 内容 | 我们可比什么 | 状态 |
|---|---|---|---|
| Fig.4 + Table VI | EEGdenoiseNet 单通道 EOG：EEGDfus 0.182/0.188/0.983；GCTNet 0.184、EEGIFNet 0.206、EEGDnet 0.497 等 | 我们的官方复现（official-native 0.2965/corr 0.9530；strict 0.2740/0.9597，含 11 级 per-SNR）、M8 基线（SimpleCNN 0.9125 CC 等）、SADDPM-Cond（0.901 CC @ -7..2 网格）、D4PM（在跑） | 大部分已存档；**E1 待做**：SADDPM-Cond 在 -5..5 11 级网格重评以并列；**已知落差需正面呈现**：我们按其公开代码+预算复现得到 0.2965，低于其发表值 0.182——两行并排、脚注差异来源（无种子、谱度量 400vs512 缺陷、泄漏保留） |
| Table II | EEGdenoiseNet EMG：0.171/0.154/0.989 | 复现 EMG cell（0.4205/0.9065）、M8 EMG 基线、SADDPM-Cond EMG 0.680 | 已存档；**E2 待做**：同 E1 网格对齐 |
| Table III | Motion artifacts | 无对应数据（EEGdenoiseNet 无 motion 库） | 仅引用，标注不可比 |
| Table I + VII | **SSED = Klados**：0.121/0.127/0.992；DuoCL 0.217、NovelCNN 0.260 | 本地 Klados v4 重建其 SSED 协议（已验证 54×19ch 配对）；跑其代码 as-released + 我们的方法同数据 | **E3 执行中**（964520：EEGDfus as-released 训练+双划分评测）；**发现其 SSED 划分缺陷**：released test 与 train 重叠（train_ssed.py:43 索引之索引），保留+strict 双报 |
| Table VIII | 泛化 EEGdenoiseNet→SSED：0.556/0.571/0.806；finetune 0.391/0.485/0.925 | 我们已有 EEGdenoiseNet 训练的复现 checkpoint + SADDPM-Cond → 零改动测 Klados 行；+10% 微调行 | **E4 排队**（E3 数据就绪后纯推理+小微调） |
| Table IV/V | SSED 多通道同步性 CC/PLV（同步对 0.745→0.894、PLV 0.710→0.804） | 同指标算我们的 Klados 去噪输出 + 我们 46 通道系统在 ambulatory 上同指标 | **E5 排队**（CPU） |
| Fig.7/8 | MNE sample 真实数据定性 + ERP 拓扑保持 | 我们已有更强定量版（D2 ERP 保真 0.931/0.966 + 拓扑改进图）；可选补 MNE sample 定性 | 低优先级，引用已有结果 |
| Table IX/X | 内部超参/步数消融 | 不可比 | 跳过 |

## 二、DS-DDPM（arXiv 2305.04200）Table I/II 复现

**关键对应（已核算）**：其 Table I 的 ICA 半表列均值平均 = **50.58**，与 SADDPM 稿协作方 Table 4 的 ICA 值**完全一致**；DS-DDPM 半表均值 52.87 ≈ 协作方 SADDPM 值 52.83。协作方下游表 = DS-DDPM Table I 协议。本复现同时解锁 TAAS 修回计划卡死的 C-0 门（分类器 = 仓库内 EEGNet + ArcFace + assets/max_acc.pth 预训练骨干）。

| 步骤 | 内容 | 状态 |
|---|---|---|
| F0 | 重建其 single_sep .mat 契约（[trials,22,1500]@250Hz，T→train/E→test，标签 1-4） | **964519 执行中** |
| F1 | 接线 probe（其 loader/模型前反向） | 同上任务内 |
| F2 | 训练 DS-DDPM（其 unet2d_overlap.py，bci_comp_iv_full_mix，100 epochs，subject embedding + ArcFace） | probe 通过后提交 |
| F3 | 采样/去噪（sample_save.py，每被试） | 依赖 F2 |
| F4 | ICA 去噪臂（Infomax，M7 管线） | 与 F2 并行 |
| F5 | Table I：每被试 EEGNet 分类器 ×2 臂，9×9 矩阵 + M 行，对照其 ICA M=50.58/DS-DDPM M=52.87 | 依赖 F3/F4 |
| F6 | Table II：真实-真实 与 生成-真实 被试相关矩阵（计算口径从其代码推断，不明处披露选择） | 依赖 F3 |

依赖已装（labml/tensorboardx 进 icml，披露）。Apache-2.0 允许最小补丁（去 comet writer、数据路径参数化），全部披露。预算 <15 GPU-h。

## 三、纪律要点（两组通用）

- 上游代码动态导入不 vendor；固定 commit（EEGDfus a19a652 / DS-DDPM 12c339a）
- 上游缺陷保留+报告，绝不静默修复；修复版另行标注（released vs strict）
- 上游无种子处添加并披露我们的种子
- 我们的复现值与其发表值**并排呈现**，不互相替代；落差如实呈现并给出可查的差异来源
