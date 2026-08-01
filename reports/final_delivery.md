# denoiseNet 服务器阶段交付

状态：**真实数据路径已建立；Stage A 仍因拆分与科学语义阻塞**

记录日期：2026-08-01（Europe/Paris）

本报告只记录当前实际完成的工作。旧 M0–M12 数值、图、checkpoint 和手稿结论没有被
重新解释为本协议结果；G1–G5 尚未运行。

## 1. 范围修订与附件

附件内容审查、要求追踪和三项科学冲突仍见：

- `reports/attachment_manifest.json`
- `reports/attachment_review.md`
- `reports/requirement_traceability.csv`

用户已明确把行政 harness 收缩为私人科研项目所需的最小形式。全 30 TB 逐文件盘点、
全量哈希、CAS、bundle authority rollover 和重复环境审计不再是活动要求。泄漏边界、
Slurm 执行、代码/数据分离、population base/NULL 语义、G1–G5 顺序和 B1–B6 禁用仍然
有效。

`CONFLICT-SCI-001`–`003` 仍未静默解决：

- population `E0` 与个体 `a_tau`/mask 必须明确分离，才能同时满足 NULL 零个体调用；
- G1–G3 必须按服务器合同纳入真实 EEG，而不能沿用手稿中更早的半模拟“通过”；
- B1–B6 只有五关全部通过后才可能启用。

## 2. 取消的全盘作业

`918918` 在约 4.1 小时内访问 2,423,427 个条目、读取 10.28 GB 元数据并产生 2.23 GB
JSON，显然不适合 30 TB 共享空间。用户要求后只取消该作业；scanner 以 `PARTIAL`/
exit `4` 收尾，未出现 stale input。巨型逐条输出已删除，小型终态证据和 Slurm 日志保留。
该工作负载是 NFS 元数据/I/O 限制，GPU 不会加速；增加 CPU 或并发作业反而可能放大共享元数据压力。
正确修复是取消这项不必要的工作，而不是为它增加算力。

替代查找 `919129` 只比较 basename：10.9 秒、100,000 个名称、无目录错误、四个目标均
无命中。它是决定定向下载的简单查找，不是全盘不存在证明，也不计算文件哈希。

## 3. 当前数据状态

数据根：`/projects/EEG-foundation-model`。检查时约 30 TB 中尚余 1.5 TB。代码根不再保存
EEGdenoiseNet 实体数据；旧路径是指向数据根的符号链接。

| 数据集 | 状态与实际路径 | 访问/读取证据 | 允许角色与限制 |
|---|---|---|---|
| EEGdenoiseNet | `verified_available`；`/projects/EEG-foundation-model/eegdenoisenet/github-8d290...` | `919131` 复制并读取 NPY 头；EEG `[4514,512]`、EOG `[3400,512]`、EMG `[5598,512]`；`919148` 建立兼容链接 | 官方 256 Hz 单通道 paired stress 数据；没有参与者/会话分组，不能证明 P0/G2 个体化或漂移 |
| SGEYESUB | `verified_available`；`/projects/EEG-foundation-model/sgeyesub/osf-2qgrd` | OSF 明确 CC BY 4.0；`919172` 发布 178 文件、1,611,314,510 字节；`919175` 在五个 study 均读到有限值 epoch；`919218` 不打开 FDT 地核对 59 个 participant stem 和 6 种 layout | 真实多参与者 EEG 的独立评估候选；80–89 通道且采样率为 100/200/256 Hz，不能默认池化；全部 SET 只见 block 1/2，与论文三-block 描述尚未映射；没有 clean target |
| Klados–Bamidis v1 | `present_unverified`；`/projects/EEG-foundation-model/klados_bamidis/v1` | Mendeley v1 / CC BY 4.0；`919153` 下载唯一 46,757,186-byte RAR；简化后的 `919191` 只核对官方大小与 RAR4 签名 | 归档尚未解包或读取 MAT；历史诊断 `919182` 的三个成员名不是样本证据；不能声称参与者、通道、clean component 或支持时长存在 |
| Eye-BCI | `restricted`；无本地路径 | 元数据端点可达，发布方称 CC0；Synapse 下载需要注册账户/token，当前没有批准凭据或 `synapseclient` | 不能下载或声称跨会话字段；不盲目拉取高帧率视频 |

小型记录在 `datasets/registry/*.json`。它们只保存路径、版本、访问、读取摘要和作业 ID，
没有 per-file manifest 或本地内容哈希。

SGEYESUB 下载失败链被完整保留：

- `919154` 因预记录总字节数与默认分页结果不一致，在写数据前停止；
- `919162` 暴露 OSF 默认 10-item 分页的重复路径问题，并留下可恢复 `.partial`；
- 修复为每 study `page[size]=100`、显式拒绝重复路径后，`919171` 得到完整稳定列表；
- `919172` 复用 51 个正确 partial 文件并下载其余 127 个，然后原子发布；
- `919173` 错把 epoched SET 当 raw 读取而失败；`919175` 改用
  `read_epochs_eeglab` 后通过。
- metadata 诊断 `919192/919210/919211` 分别暴露 MATLAB-v7.3 与小 MAT 头兼容问题；
  它们都未打开 FDT。设置读取上限、一致性断言并压缩重复记录后，`919218` 通过。

## 4. Git、环境与调度

- 代码根：`/home/infres/yinwang/denoiseNet`
- branch：`master`
- origin：`https://github.com/W-Yinghao/EEG_denoise.git`
- 轻量 harness、数据审计与 SGEYESUB 结构里程碑：`41c2a6f`
- 未 push、未建 release、未上传数据。
- 工作树仍含用户既有/并行的手稿、结果、图和源码修改；本任务未 reset、覆盖或纳入提交。

仅使用已有环境：

- `eeg2025`：名称查找、下载、NumPy/MNE/SciPy 数据读取；
- `icml`：保留给未来 PyTorch/CUDA population base 和采样。

没有创建第三个环境，也没有安装/升级共享环境。reader 作业 `919155/919159` 确认 MNE、
NumPy、SciPy、h5py 可用，但 7z、7za、7zz、unrar、unar、bsdtar、rarfile、libarchive 均
不可用，因此没有强行解包 Klados。

新 dataset 作业使用 `submit.sh` 的轻量分支：只解析已登记 CPU profile、提交 sbatch 并
返回 job ID，不再生成 bundle/hash request JSON。长作业依赖使用 `afterok`；监控最多每
五分钟一次并只报告状态变化。轻量提交器自测 `919190` 已通过。prior 和 gate 状态
使用可读配置路径、split/checkpoint/Git 版本和 `TBD-PREREG/frozen` 状态，不再维护伪精确的本地哈希。
最终配置自测 `919220` 已验证 5 个 gate 状态和 4 个 prior 配置。

## 5. Population base、NULL、P0 与门

当前没有合法 population checkpoint、P0 full-fold 或 gate 结果。Stage A 的新状态见
`reports/stages/stage_a_status.json`：已有真实可读 SGEYESUB 路径，但仍缺少冻结的
participant/session/support/query split。

官方 SGEYESUB 已冻结为独立基线 `native_sgeyesub_commit_2c95b4f`：固定水平、垂直和
残差 blink 方向，`alpha=1`、`beta=0.01`，不是 oracle 或正交 projector。具体审查见
`reports/sgeyesub_reference_audit.md`；本地结构结果见 `reports/sgeyesub_structure_audit.md`。

已完成的最小设计是：

1. 非 subject-conditioned clean-prior score 独立存在；
2. population observation energy 显式为同一查询 `y` 的
   `E0(x;y)=0.5*(x-y)^T Lambda0 (x-y)`；
3. `Lambda0` 只能由外层训练数据拟合，初始标为 generalized Bayes；
4. NULL 只委托 `PopulationPosteriorBase.sample`，不接受或调用 operator、mask、
   attenuation、`DeltaE` 或个体校正；
5. 旧 `ConditionalDiffusionDenoiser` 因不能分离 clean prior 与 `E0`，不被冒充为该基座。

该设计尚未实现为确认性模型，因为 `CONFLICT-SCI-001` 未被有权主体解决，且当前可读的
SGEYESUB 无 clean target、EEGdenoiseNet 无 participant grouping、Klados 尚未解包。

因此：

- P0：未运行；
- G1–G5：均未运行，不是 passed；
- B1–B6：全部 `enabled: false`；
- synthetic fixture：只允许未来工程测试，不能进入门或论文结果。

## 6. 当前可支持与不可支持的结论

可支持：

- 轻量查找与定向下载流程在 Slurm 上工作，不再扫描共享 30 TB 空间；
- EEGdenoiseNet 与 SGEYESUB 的上述具体变体可读；
- SGEYESUB 五个 study 的通道数和采样率确实异构；
- Klados v1 归档下载正确到 RAR4/官方大小层面；
- Eye-BCI 当前受认证阻塞。

不可支持：

- population posterior 已闭合或 NULL 已验证；
- 个体 P0 operator 有效、oracle mechanism 通过或 operator specificity 成立；
- diffusion 必要、EEG-only gate 可部署、漂移可回退；
- 任意显著提升、participant-level 效应、论文结果数值；
- Klados 原生记录时长、60/120 秒支持或 clean/oracle 字段；
- SGEYESUB 是 oracle 或有 paired clean truth。

## 7. 下一项最小安全工作

1. 从官方来源解决 SGEYESUB `studyXX -> EEGDSX` 映射和三-block 描述差异；在此之前不凭
   人数/文件夹编号冻结 native split。
2. Klados 已停在 `present_unverified`；除非已批准的 RAR reader 自然可用，否则不再自制
   parser，也不修改共享 Conda 环境来“救回”它。
3. 由有权主体解决 `CONFLICT-SCI-001/002`，再实现 population base/NULL 的工程测试。
4. 只有 Stage A 与 Base closed 后，才进入完整真实 fold 的 P0，然后严格 G1→G5。

当前阶段的科学结论仍是：**真实数据路径已建立，但没有可采信的去噪或个性化实验结果。**
