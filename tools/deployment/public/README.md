# MaaGakumasu Gakumas Helper

这是基于 [MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu) 的公开派生预览版。它保留上游 Maa 客户端与日常功能，并新增竞技场完整编成读取、本地计分、保守胜率选敌、结果记录以及配套识别资源。

- GKH 版本格式：独立语义化版本 (Semantic Versioning, SemVer) `vMAJOR.MINOR.PATCH`
- 上游 Maa 基线：在每版 Release manifest 与来源记录中单独固定，不拼入 GKH 版本
- 下载与校验：[@REPOSITORY@ Releases](@REPOSITORY@/releases)
- 上游项目：[SuperWaterGod/MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)
- 许可证：AGPL-3.0；内嵌竞技场模拟引擎另按 BSD-3-Clause 与 Node.js 许可发布

> [!WARNING]
> 当前是 Windows x86_64 竞技场预览版，不代表全部识别验收或最终稳定版已经完成。竞技场任务会实际操作游戏并可能消耗挑战次数；请先阅读[首次使用](guides/GETTING_STARTED.md)与[竞技场指南](guides/ARENA_WIN_RATE.md)。

## 快速开始

1. 首次安装时从 Releases 下载 `MaaGakumasu-win-x86_64-vXXX.zip`；`vXXX` 代表资产名中的完整版本，checksums 可用于人工复核下载字节。
2. 解压到新的独立目录，不要覆盖上游 Maa 或旧版 Gakumas Helper。
3. 运行版本内 `deployment\Start-MaaGakumasu-Admin.cmd`，以管理员身份和唯一空 TEMP/TMP 启动；程序不会读取或保存 DMM 账号密码。
4. 首次使用先添加并单独运行“重算竞技场己方总分”，成功后再运行“竞技场胜率分析并挑战（单次）”或“每日挑战 → 胜率选敌”。

详细步骤、默认参数、分辨率要求和失败处理见[首次使用](guides/GETTING_STARTED.md)。

## 相对上游新增

| 领域 | Gakumas Helper 新增能力 |
| --- | --- |
| 竞技场读取 | 读取双方三舞台编成、表现力、支援加成、P 道具、技能卡、重复卡与自定义强化 |
| 本地计分 | 使用固定版本、自包含的 `gakumas-tools` 引擎生成成员分数分布，不访问在线模拟网站 |
| 胜率决策 | 应用同舞台第一名 `+20%` 规则、三局两胜、95% Wilson 区间及三对手 Bonferroni 校正 |
| 节拍与缓存 | 首次自动校准 Worker；己方分数可缓存，日常默认只重读三个对手 |
| 安全闭锁 | 身份、页面、字段、赛季或结果不能唯一确认时停止，不以综合力或猜测结果替代 |
| 结果记录 | 挑战前创建待决记录，挑战后核对总体胜负与三舞台双方成绩，再允许下一轮 |
| 发布治理 | 单提交公开源码快照、来源/许可证清单、Release manifest、校验和及隐私扫描 |

完整差异、继承范围和未完成边界见[与上游的区别](guides/UPSTREAM_DIFFERENCES.md)。

## 使用与反馈入口

- [指导中心](guides/README.md)
- [竞技场胜率功能](guides/ARENA_WIN_RATE.md)
- [故障排查](guides/TROUBLESHOOTING.md)
- [更新与回滚](guides/UPDATE_AND_ROLLBACK.md)
- [隐私与安全](guides/PRIVACY_AND_SAFETY.md)
- [问题反馈与分流](SUPPORT.md)
- [贡献说明](CONTRIBUTING.md)

提交公开 Issue 前请先脱敏。不要上传账号信息、完整配置、完整日志包、未打码游戏截图、己方/对手缓存或训练数据。属于上游原版且能在官方 MaaGakumasu 复现的问题，请提交到[上游 Issues](https://github.com/SuperWaterGod/MaaGakumasu/issues)。

## 安装与更新边界

GKH 版本独立于上游 Maa 标签。项目处于 `0.x` 时，`MINOR` 表示基础大更新，`PATCH` 表示该基础上的内部迭代；项目正式完工时才把 `MAJOR` 升至 `1`。从旧 `vMAJOR.MINOR.PATCH+gkh.*` 命名空间切入独立 GKH SemVer 时须手动安装一次；迁移预览期间保持 MFA 资源更新通道为默认 Stable，不要切换到 Beta 或 Alpha。独立版本之间按 SemVer 递增，但只有发布预检确认 MFA 的目标更新通道不会再选中数值更高的旧组合 Release 后，才允许启用自动升级；满足该门后，唯一客户端更新入口仍是 MFAAvalonia 内置 GitHub 资源更新。完整包会同时更新程序、Agent、资源、UI、模型、图库及竞技场离线基线，不需要手工逐组件校验或第二套客户端更新器。

内置 pip 入口只管理 pip 与 `requirements.txt` 中的 Python 依赖，不更新上述项目资产。竞技场 RIS engine/data 是严格窄例外：每个 Agent 进程首次需要模拟器时可按最新成功 production deployment 的不可变 SHA 成对更新，失败保留旧组件；它不更新客户端、Agent、UI、模型或图库。版本化目录、`current` 联接和本机安装器只服务开发部署与故障维护，不是普通客户端更新路径。

当前只有 GitHub 更新仓库指向本项目；包内若显示上游 `MaaGakumasu` 的 MirrorChyan 入口，它不是 Gakumas Helper 更新通道，请勿用它更新本派生版。

公开包不包含用户配置、日志、缓存、截图、对局记录、私有训练语料、凭据或本机路径。源码仓库使用独立的单提交公开快照，不包含内部开发仓库历史和任务管理记录。

## 来源与声明

本仓库提供派生 Agent、资源、模型、数据及构建脚本的对应源码。完整 Windows 包复用固定 MaaGakumasu 上游发行版；竞技场模拟器来自固定版本的 [surisuririsu/gakumas-tools](https://github.com/surisuririsu/gakumas-tools)。具体版本、文件散列和许可证记录在 Release manifest、[来源说明](ASSET_PROVENANCE.md)与 `THIRD_PARTY_NOTICES` 中。

本项目与万代南梦宫及游戏官方无隶属或背书关系。使用自动化前请自行确认适用规则并承担账号与资源消耗风险。
