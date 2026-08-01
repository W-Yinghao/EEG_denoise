# 数据源网络元信息审查

审查日期：2026-08-01（Europe/Paris）

## 范围与证据边界

本报告只记录候选数据集发布方、论文发布方或参考实现维护方在公开页面上可核验的元信息。它不是对 `/projects/EEG-foundation-model` 的文件盘点，也没有执行数据下载、归档解包、字节级校验、样本读取或许可证法律解释。网络页面所称“公开”“可下载”或某一许可，不能证明服务器上已经存在相应数据，更不能证明本地副本完整、版本匹配、可匿名取得、样本可读或适合本项目的科学角色。

本报告不得单独把任何数据集登记为 `verified_available` 或 `missing`。数据注册表仍须结合 Slurm 全根只读盘点、定向 manifest/哈希审计、实际访问条件与样本读取结果；证据不足时保持 `unknown` 或 `present_unverified`。

## Klados–Bamidis v1

- 第一方版本页：[Mendeley Data，version 1](https://data.mendeley.com/datasets/wb6yvr725d/1)。页面明确标示 2016-05-18 发布、`Version 1`、DOI `10.17632/wb6yvr725d.1`，许可为 `CC BY 4.0`。
- 页面将其描述为半模拟 EEG/EOG 数据：人工把眼动伪迹混入无伪迹 EEG，并保留污染前信号。这只支持其作为配对半模拟机制和可核验生成污染信息的候选来源；不能据此把恢复结果等同于真实部署性能。
- 本任务冻结的是 **v1**。同一 Mendeley 数据集存在后续版本不构成改用最新版的授权；若本地发现其他版本，必须分别登记并报告冲突，不能静默替代 v1。
- 本次网络审查没有取得 v1 文件 manifest、内容校验和或数据字节，也未核验参与者/记录结构、原生时长、通道、采样率和样本可读性。因此不能启用 60/120 秒支持单元格，也不能证明服务器存在性或 `oracle span` 所需字段已经实际可用。

网络元信息结论：版本锚点、DOI 和发布页许可已核验；服务器存在性、实体完整性和科学字段仍未验证。

## Eye-BCI

- 第一方论文：[Scientific Data 数据描述论文](https://www.nature.com/articles/s41597-025-04861-9)，论文 DOI 为 `10.1038/s41597-025-04861-9`。论文描述 EEG、眼动与高速视频的同步多模态、多会话及多范式采集，并声明原始数据托管于 Synapse。
- 数据实体锚点：[Synapse `syn64005218`](https://www.synapse.org/Synapse%3Asyn64005218/metadata/)，数据 DOI 为 [`10.7303/syn64005218`](https://doi.org/10.7303/syn64005218)。论文的 “Distribution for use” 段明确把所述数据标为 `CC0`，并说明共享数据已匿名化；这是发布方对数据分发与许可的声明。
- “公开”与 `CC0` 不等于本执行环境已获得字节。本次审查没有进行无凭据/匿名的数据字节下载，没有获得稳定文件 manifest、校验和或 Synapse 实体的不可变版本号，也没有核验实体是否要求登录、点击条款、认证或分页 API。因此当前只能记录“来源声明公开且 CC0；匿名字节访问和实体版本未验证”。
- 论文元信息可支持把它视为真实跨会话/跨范式漂移审计的候选来源，但不能推断存在配对干净 EEG，也不能把眼动、EOG、事件、同步或蒙太奇字段视为本地已验证。视频公开声明也不能替代服务器端访问、隐私与体量审计。

网络元信息结论：论文、数据 DOI、Synapse 锚点和发布方 CC0 声明已核验；匿名字节下载、不可变实体版本、服务器存在性与样本可读性仍未验证。

## SGEYESUB

- 第一方参考实现：[rkobler/eyeartifactcorrection](https://github.com/rkobler/eyeartifactcorrection)。审查时默认分支为可变的 `master`；仓库明确包含 SGEYESUB 及其他眼动伪迹校正算法的参考实现、校准/测试示例，并标示代码许可为 [`LGPL-3.0`](https://github.com/rkobler/eyeartifactcorrection/blob/master/LICENSE)。在正式运行前仍须冻结确切 commit，而不能只记录 `master`。
- 仓库 README 将论文所用的预处理 EEG 数据指向 [OSF 项目 `2QGRD`](https://osf.io/2qgrd/) 和 DOI [`10.17605/OSF.IO/2QGRD`](https://doi.org/10.17605/OSF.IO/2QGRD)。参考实现仓库的 `LGPL-3.0` 是**代码许可**；它不能外推为独立 OSF 数据实体的许可，也不能证明 OSF 数据版本或访问条件。
- 本次无凭据读取 OSF 项目页以及第一方 API 端点 [`https://api.osf.io/v2/nodes/2qgrd/`](https://api.osf.io/v2/nodes/2qgrd/) 均返回 HTTP 403，未取得可审计的数据许可、版本、manifest 或字节。403 只说明本次读取路径失败，不能据此断言数据永久受限，也不能把数据标为 `missing`；当前数据许可、实体版本和可访问性均为 `unknown`，等待定向审计。
- Native SGEYESUB 必须保留官方校准、秩选择和校正规则，并使用独立方法 ID/provenance。代码与数据元信息均不能把它改称 `oracle`，也不能授权用真实生成 span 替换其估计算法后仍报告为 native SGEYESUB。

网络元信息结论：参考代码职责与 LGPL-3.0 已核验；OSF 数据许可、版本、可访问性、服务器存在性和样本可读性未知。

## EEGdenoiseNet

- 第一方代码入口：[ncclabsustech/EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet)。其 `master` README 把 512 Hz 的 EEG/EMG epoch 数据指向第一方 [G-Node GIN 数据仓库](https://gin.g-node.org/NCClab/EEGdenoiseNet)，代码仓库标示 `MIT`；该代码许可与 GIN 数据许可必须分开记录。
- GIN 审查时页面显示分支 `master`、17 个提交、无 release；顶层当前提交为 [`2bca0a94d1bd41dfa67358934d5b15d1efc0b73a`](https://gin.g-node.org/NCClab/EEGdenoiseNet/commit/2bca0a94d1bd41dfa67358934d5b15d1efc0b73a)，`data/` 条目最后提交为 [`7d242bb7a1f0914df6dfe95703a1a20e55dcdfe6`](https://gin.g-node.org/NCClab/EEGdenoiseNet/commit/7d242bb7a1f0914df6dfe95703a1a20e55dcdfe6)，许可元数据所在 `datacite.yml` 的可见更新提交为 [`47fecb5c606816350f7818ba48148cca947e79c1`](https://gin.g-node.org/NCClab/EEGdenoiseNet/commit/47fecb5c606816350f7818ba48148cca947e79c1)。这些 commit 是审查锚点，不是发布版标签；`master` 仍是可变分支。
- GIN 的 `datacite.yml` 页面元信息将数据许可标为 `Creative Commons CC0 1.0 Public Domain Dedication`。这支持记录 GIN 数据的来源级许可声明，但不把 GitHub 的 MIT 许可扩展到数据，也不证明任何本地副本对应上述 commit。
- GIN 明确警告：普通 `.zip`/`.tar.gz` 仓库归档只包含小文件和目录结构；默认大于 10 MB 的 git-annex 文件可能只是 pointer/stub，真实大文件不会随普通归档包含，须逐个下载或使用 git-annex/GIN 客户端取得。因而“归档可解包”或“文件名存在”不能证明 EEG/EMG 数组完整，定向审计必须验证 annex 内容可得性、实际字节数与内容哈希。
- 发布方描述的是干净 EEG、EOG 和 EMG epoch，可用于合成带真值的噪声 epoch；官方另行指向 `Single-Channel-EEG-Denoise` 工具。这一结构只允许把 EEGdenoiseNet 作为通用 EOG/EMG、单通道压力测试候选，不足以证明参与者级多通道传递算子、个体支持校准、真实跨会话漂移或 P0/G2 个体化主张。

网络元信息结论：GIN 来源级 CC0 声明、可变 master 的关键提交锚点和 annex 完整性风险已核验；服务器存在性、annex 实体字节、哈希与样本可读性仍未验证。

## 对后续状态机的约束

1. 网络元信息不得覆盖 Slurm 全根盘点结论，也不得把 `unknown` 自动转为 `missing`、`present_unverified` 或 `verified_available`。
2. 下载只可在全根盘点完成、状态有证据地确认为 `missing`、来源/版本/许可/访问路径均获准后，通过事务式 Slurm 作业执行；本报告不构成下载授权。
3. 任何命中都还需核验精确版本、许可文本、文件/annex 完整性、校验和、样本读取、字段、通道/参考/采样率及参与者/会话结构。没有这些证据时不得推进 population base、P0 或 G1–G5。
4. Klados–Bamidis 的半模拟机制角色、native SGEYESUB、Eye-BCI 的真实漂移角色和 EEGdenoiseNet 的单通道压力测试角色必须分别登记；网络元信息不能消除这些科学边界。
