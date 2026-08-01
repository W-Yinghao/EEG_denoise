# denoiseNet 服务器执行阶段性交付

状态：**pending-inventory / scientific-blocked**

记录时间：2026-08-01（Europe/Paris）

这是一份等待全根数据盘点作业 `918918` 的阶段性交付，不是最终科学完成报告。作业
`918918` 尚未开始运行，本文没有数据盘点结果、数据注册表、真实 EEG 实验或门结果。
该作业终止后必须用其不可变状态、实际 allocation 和扫描产物更新本文；调度器当前显示
的预计开始时间不能替代完成证据。

## 1. 附件审查、冲突与未解决要求

当前任务附件的语义和视觉内容审查已经完成。三项附件的文件名、大小、SHA-256、媒体
类型、修改时间、读取状态和归档安全审计记录在
[attachment_manifest.json](/home/infres/yinwang/denoiseNet/reports/attachment_manifest.json)；
完整阅读、54 页 PDF 视觉检查、表格/公式/批注检查和可执行要求映射记录在
[attachment_review.md](/home/infres/yinwang/denoiseNet/reports/attachment_review.md) 与
[requirement_traceability.csv](/home/infres/yinwang/denoiseNet/reports/requirement_traceability.csv)。
先前完整 Slurm 提取支持内容审查，但不被提升为当前严格执行 provenance。

当前 hardened 链的控制验证 `918915` 和父作业 `918916` 完成；L40S PDF canary
`918917` 在第 8 页文本后的 PyMuPDF warning audit 处以 exit `3` fail closed。它只有
第 1–7 页完成严格验证，没有 `COMPLETE` marker；五个 sibling PDF 没有提交。因而只能
陈述“内容审查完成”，不能陈述“hardened PDF supplement 成功”。

三项科学权威冲突仍未解决：

- `CONFLICT-SCI-001`：共享查询期 `a_tau` 的 population energy 叙述与 NULL 零
  attenuation/mask 调用合同冲突。
- `CONFLICT-SCI-002`：附件 RQ1–RQ3 与服务器 G1–G3 对真实 EEG 证据的进入顺序不同。
- `CONFLICT-SCI-003`：附件允许较早打开 backup 的叙述与“五关全部通过后才允许
  B1–B6”冲突。

作者、伦理/同意、AI 披露、编辑路径、CCS、投稿系统、最终静态构建和外部归档等人类或
外部动作仍为 blocked；没有代填，也没有论文提交或上传。

## 2. 数据状态、来源、访问与下载

固定数据根为 `/projects/EEG-foundation-model`。登录节点只完成了根目录级轻量核验；
全根只读盘点由 `cpu-high` 作业 `918918` 承担。本文记录时该作业状态为
`PENDING (Priority)`，请求 8 CPU、64 GiB、5 天，原始依赖为
`afterok:918908`。调度器显示的 `2026-08-05T00:40:00` 是可变的预计开始时间，
`SchedNodeList` 也不是 allocation；当前没有节点、`AllocTRES`、退出码或扫描结果。
提交 ID 记录在
[submitted_job_id.txt](/home/infres/yinwang/denoiseNet/reports/data_inventory/submitted_job_id.txt)，
实时证据边界见
[data inventory summary](/home/infres/yinwang/denoiseNet/reports/data_inventory/summary.md) 和
[machine status](/home/infres/yinwang/denoiseNet/reports/data_inventory/current_status.json)。

网络侧第一方元信息审查见
[data_source_metadata_review.md](/home/infres/yinwang/denoiseNet/reports/data_source_metadata_review.md)。
它不能证明服务器存在性、完整性、实际访问、字段或样本可读性。

| 数据集 | 已核验的来源级元信息 | 当前注册状态 | 本任务下载/读取 |
|---|---|---|---|
| Klados–Bamidis v1 | Mendeley v1、DOI 和来源页 CC BY 4.0 声明 | 未生成；inventory pending，registry 不存在 | 未发起下载；未核验本地版本、manifest、时长、通道或样本 |
| SGEYESUB | 参考代码职责与 LGPL-3.0；OSF 数据页/API 本次返回 403 | 未生成；inventory pending，registry 不存在 | 未发起下载；数据许可、版本、访问和样本读取未知 |
| Eye-BCI | 论文、Synapse/DOI 锚点及发布方 CC0 声明 | 未生成；inventory pending，registry 不存在 | 未发起下载；匿名访问、实体版本、字段和样本读取未验证 |
| EEGdenoiseNet | GIN 来源级 CC0 声明和 annex 完整性风险 | 未生成；inventory pending，registry 不存在 | 未发起下载；annex 实体字节、哈希和样本读取未验证 |

当前没有任何 `datasets/registry/<dataset_id>.json`，没有数据集可标为
`verified_available` 或 `missing`，也没有目标化版本/许可/完整性/样本读取作业。
本任务没有发起数据下载、解包、预处理、派生数据写入或原始数据修改；这不否认待盘点
数据根中可能存在的既有副本。即使 918918 最终为
`COMPLETE`，Phase I 的路径/文件名命中也最多支持 `present_unverified`；无命中仍是
`unknown`，不能自动写成 `missing`。

## 3. Git、代码、配置、测试和 push 状态

本次证据边界修订开始时的已提交快照为：

- 代码根：`/home/infres/yinwang/denoiseNet`
- branch：`master`
- HEAD：`12bca55b21a47ef2096bf1e8f856bcba49db7425`
- 本报告首次加入的本地提交：`12bca55b`；本次 Phase-II 合同修订的提交 SHA 以交付时
  `git rev-parse HEAD` 为准
- origin：`https://github.com/W-Yinghao/EEG_denoise.git`
- upstream：无
- 远端 heads/tags：复核时为空
- push/release/upload：未执行

工作树不是 clean。已有用户/并发代理的 tracked 修改和大量 untracked 源码、结果、图、
附件及手稿文件均被保留；本任务没有 reset、清理、覆盖或把它们纳入科学证据。历史
bootstrap 与非空仓库处理见
[bootstrap_state.md](/home/infres/yinwang/denoiseNet/reports/bootstrap_state.md) 和
[repository_audit.md](/home/infres/yinwang/denoiseNet/reports/repository_audit.md)。

当前完成的是行政脚手架：中央 Slurm 提交器、环境/控制面/附件/盘点作业、数据与 gate
schema、必需配置族、blocked stage/gate 状态和 B1–B6 禁用配置。科学包
`src/eeg_cspd/` 与当前合同测试目录仍主要是职责/边界骨架；没有将其描述为已实现或
已通过测试。旧 `saddpm`、M0–M12、旧 V100 脚本、旧 checkpoint、`RESULTS.md` 和
现有 legacy 结果均未采信。

## 4. Conda 与 Slurm 审计

环境审计详情见
[environment_audit_summary.md](/home/infres/yinwang/denoiseNet/reports/environments/environment_audit_summary.md)，
控制面映射和基础设施警告见
[control_plane_audit.md](/home/infres/yinwang/denoiseNet/reports/slurm/control_plane_audit.md)。

| 环境 | 当前严格作业 | 实际启动时 allocation | 显式 / pip lock SHA-256 | 结论 |
|---|---|---|---|---|
| `eeg2025` | `918908` | CPU / nodecpu09 / 2 CPU / 8 GiB | `cc644eead3ffa906a70573727b82974da57145b408b67a920741a4288fc0296d` / `ad6370f70776d2968aa20e56bc4aafb4e0e67305146f69c48f7df3926ebc0207` | completed、provenance complete；CPU imports 通过 |
| `icml` | `918909`, `afterok:918908` | L40S / node39 / 8 CPU / 64 GiB / 1 L40S | `2c04fc1733a53b55abd071d6b1657eabfda8bbb56ef0bf0ab97e8234171958a1` / `7af84a80b5d762b08dc542f8b9edf7457106a7ff9721458bbdd1f22c7179a939` | completed、provenance complete；CUDA 与注册 renderer startup 通过 |

Verified-state 控制作业 `918915` 通过。没有创建第三个环境，没有安装、升级或修改这
两个共享环境。

完整 94-job 行级账本见
[job_ledger.csv](/home/infres/yinwang/denoiseNet/reports/slurm/job_ledger.csv)，失败/恢复链和
证据规则见
[job_ledger.md](/home/infres/yinwang/denoiseNet/reports/slurm/job_ledger.md)。94 个 manifest
的 array 字段均为空；没有 Slurm array。所有这些作业都是控制、环境、附件或盘点行政
作业，不是科学训练或外层测试。

`sacct` 无法连接 SlurmDBD（Connection refused）。因此最终 scheduler elapsed/end、
历史 accounting、能耗、峰值内存等字段不可得；账本中的 allocation 是作业运行时
`scontrol` 快照，不冒充最终 accounting。失败、取消和 prior-bundle 作业全部保留，
包括错误依赖作业 `918885`、依赖取消 `918886`、历次 renderer 失败和最终 canary
`918917`。

## 5. Population base、NULL、P0 与五道门

机器可读阶段状态为：

| 阶段 | 状态 | 核心证据 |
|---|---|---|
| [Stage A](/home/infres/yinwang/denoiseNet/reports/stages/stage_a_status.json) | `blocked` | 918918 pending；无 targeted sample audit、verified real EEG 或 split |
| [Population base](/home/infres/yinwang/denoiseNet/reports/stages/population_base_status.json) | `blocked` | Stage A 未过、CONFLICT-SCI-001/002、无 prior/checkpoint/NULL 测试 |
| [P0 real-EEG slice](/home/infres/yinwang/denoiseNet/reports/stages/p0_slice_status.json) | `blocked` | Base 未闭合、无完整真实外层折、P0 参数仍 TBD-PREREG |
| [G1](/home/infres/yinwang/denoiseNet/reports/gates/g1/gate_status.json) | `blocked` | 无输入运行；阈值占位；Base/P0 未过 |
| [G2](/home/infres/yinwang/denoiseNet/reports/gates/g2/gate_status.json) | `blocked` | G1 未批准；无 operator-source 比较 |
| [G3](/home/infres/yinwang/denoiseNet/reports/gates/g3/gate_status.json) | `blocked` | G1/G2 未批准；无 matched baseline 或 diffusion 比较 |
| [G4](/home/infres/yinwang/denoiseNet/reports/gates/g4/gate_status.json) | `blocked` | G1–G3 未批准；无 EEG-only 校准/risk-coverage |
| [G5](/home/infres/yinwang/denoiseNet/reports/gates/g5/gate_status.json) | `blocked` | G1–G4 未批准；无真实漂移/rollback 运行 |

这些 `blocked` 记录不是统计意义的 `failed` 或 `inconclusive`。每个 gate 的
`input_runs` 为空，指标和参与者统计为空，`approved_next_stage=null`；其中
`threshold_hash` 只绑定当前禁用且含 `TBD-PREREG` 的占位配置，不表示门槛已经冻结。
没有 population posterior checkpoint，没有证明模型使用查询 `y`，没有 NULL
same-seed 路径等价/零调用测试，也没有 P0 `context_result.json`。

## 6. 真实 EEG 实验、seed、基线、失败和复现范围

当前协议下没有合法的真实 EEG 实验运行：

- 真实数据集数、外层折数、参与者/会话数和 seed 数均为零个可采信运行。
- 没有完整原生记录的 P0 fold，没有 G1–G5 input run。
- 没有 oracle/native SGEYESUB、information-matched one-step、task-matched deterministic、
  raw/standard、NULL 或 diffusion 的确认性比较。
- 没有 proper score、覆盖、任务、污染抑制、保持性、风险覆盖、漂移、运行时间、显存或
  能耗科学指标。
- 没有科学失败、弃权或 rollback 分母；现有失败均属于附件/环境行政诊断。
- 没有可复现的科学提交命令，因为任何科学提交在当前状态下都不合法。

Synthetic fixture 只被配置为未来工程测试边界；本文没有把任何 synthetic 数值、旧
M0–M12 数值、旧图表或 legacy test 当作结果。因而不存在
`results/g1_oracle_mechanism.parquet` 至 `results/g5_drift_rollback.parquet` 的合法
确认性行，也没有论文结果表。

## 7. Deferred B1–B6

`ROBUST-M`、`LAG-FIR`、`FB-COV`、`GRAPH-RIDGE`、`CCA/GED-COV` 和
`POP-SHRINK` 的六个配置均保持 `enabled: false`。五道门没有通过，且
`CONFLICT-SCI-003` 未解决；没有诊断 ID、前瞻性修订、新 untouched split、路由冻结、
拟合、搜索、比较或外层测试。

没有创建 `results/deferred_b1_b6.parquet`。这不是缺失的成功结果，而是 deferred
合同的正确表现；没有任何 backup 被称为最佳家族，也没有 RPCA/ICA 等探索项混入正式
B1–B6。

## 8. 当前可支持结论、不可支持结论与下一步

当前可支持的结论仅限行政与证据边界：

- 双根目录、空远端、非空并发工作树、Slurm profile 和两环境合同已经审计。
- 当前严格环境 authority 是 918908/918909，控制验证 918915 通过，环境未被修改。
- 附件内容已经完整审查，三项科学冲突已留痕；918917 的 hardened canary 失败也已保留。
- 四个候选数据源的第一方网络元信息已审查，但本地存在性和可用性尚未审查完成。
- 全根只读 inventory 已合法提交为 918918，但仍在排队。

当前不能支持任何真实 EEG 去噪、个性化、population posterior、P0、扩散必要性、
EEG-only 部署、漂移鲁棒性、统计显著性或资源效率结论。总体科学结论是：
**no admissible real-EEG evidence；scientific chain blocked**。

下一项最小安全工作是继续监控 918918，直到它以不可变 `COMPLETE` 或 `PARTIAL`
状态终止，然后：

1. 用实际 job/allocation/coverage/candidate 证据更新数据盘点摘要、94-job ledger 和本文。
2. 按
   [Phase-II targeted audit contract](/home/infres/yinwang/denoiseNet/reports/targeted_dataset_audit_plan.md)
   对路径命中的候选数据运行目标化版本、许可/访问、完整性和样本读取 Slurm 审计；
   无命中仍保持 `unknown`。
3. 生成四个证据受限的数据注册表；只有满足全部合同才可写
   `verified_available`，不得仅凭 Phase I 写 `missing`。
4. 在至少一个合法真实 EEG 路径和不可变 support/query split 存在后，仍须先由有权主体
   解决 CONFLICT-SCI-001/002/003 并冻结全部 TBD-PREREG。
5. 只有上述条件满足后，才可依次实现和验证 population base/NULL → P0 → G1–G5；
   B1–B6 继续保持禁用。

在 918918 终止并完成上述更新前，不得把本文改称最终科学完成报告。
