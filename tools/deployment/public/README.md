# Gakumas Helper

面向《学园偶像大师》的日常小助手，基于 [MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)。在同一个 Maa 客户端中使用日常任务、自动培育，以及本项目新增的竞技场读取、胜率选敌和结果记录。

[下载最新版本](@REPOSITORY@/releases/latest) · [首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md) · [使用文档](@REPOSITORY@/blob/main/guides/README.md) · [问题反馈](@REPOSITORY@/blob/main/SUPPORT.md)

## 功能

| 分类 | 可以做什么 |
| --- | --- |
| 日常任务 | 沿用上游的启动游戏、收取活动费、安排工作、社团活动与领取奖励 |
| 商店兑换 | 沿用上游的扭蛋兑换、每日兑换与每周免费初星包任务，按用户配置执行 |
| 培育与工具 | 保留上游自动培育、支援卡识别等已有任务和设置；适配范围见[与上游的区别](@REPOSITORY@/blob/main/guides/UPSTREAM_DIFFERENCES.md) |
| 竞技场胜率选敌 | 读取双方编成并在本机模拟，优先选择达到胜率门槛的对手；无人达标时选择预测胜率最高者，仍记录为未达标 |
| 己方重算与复用 | 单独读取并重算己方数据；日常复用兼容缓存，编成或期数变化后由用户重新计算 |
| 对局记录与恢复 | 记录三个舞台赢家与总结果，成功保存后回到竞技场并继续剩余场次；分数异常不阻断，已支持的临时故障由客户端在固定次数和时限内自动恢复 |

竞技场提供三个入口：**重算竞技场己方总分**、**竞技场胜率分析并挑战（单次）**、**每日挑战 → 胜率选敌**。完整规则与缓存说明见[竞技场指南](@REPOSITORY@/blob/main/guides/ARENA_WIN_RATE.md)。

## 下载与开始

当前发布为非预发布 Latest，公开源码位于 `main`。

1. 从 [Latest Release](@REPOSITORY@/releases/latest) 下载 `MaaGakumasu-win-x86_64-vXXX.zip`；`vXXX` 为实际版本号。Release 同时提供校验和，供需要时核对下载文件。
2. 解压到一个新的目录，保留完整包内容。首次安装、旧组合版本迁移或更新入口不可用时均使用此方式。
3. 运行包内 `deployment\Start-MaaGakumasu-Admin.cmd`，按系统提示以管理员身份启动。DMM 用户在客户端选择 **PC** 控制器和 **DMM** 资源，再连接已登录的游戏。
4. 在任务列表中添加需要的任务。第一次使用竞技场时，先单独运行 **重算竞技场己方总分**，确认期数与编成摘要正确。
5. 再运行 **竞技场胜率分析并挑战（单次）**，或选择 **每日挑战 → 胜率选敌**。这两个入口会实际消耗挑战次数。

竞技场默认门槛为 **70%**、模拟 **2000** 次、模拟超时 **180 秒**。完整界面读取可能需要数分钟；180 秒只限制模拟阶段。更多设置和操作步骤见[首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md)。

## 支持范围与当前限制

| 项目 | 当前范围 |
| --- | --- |
| 发布包 | Windows x86_64 完整包；本项目竞技场实测环境为 Windows 11 + DMM PC 日文界面 |
| 游戏画面 | 日常建议使用物理 `720×1280` 或更高的竖向 `9:16` 窗口。Maa 返回的识别图像与物理窗口尺寸可能不同，具体要求见[首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md) |
| 小窗口 | 物理 `540×960` 已完成读取验证，完整日常流程尚未通过验收 |
| 其他环境 | 上游保留的模拟器、其他系统和汉化资源，不代表本项目竞技场已完成等价验证 |

本次转场结果恢复的初步修复已完成离线回归，尚未新增实机验收；**同一版本连续两个游戏日、每天 5 场**及其余跨窗口、独立样本验收仍未全部完成。恢复额度耗尽后，客户端会保留未完成对局并停止新挑战；请按[故障排查](@REPOSITORY@/blob/main/guides/TROUBLESHOOTING.md)反馈。

## 使用文档

| 需要了解 | 文档 |
| --- | --- |
| 安装、连接和首次运行 | [首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md) |
| 竞技场规则、缓存和参数 | [竞技场胜率功能](@REPOSITORY@/blob/main/guides/ARENA_WIN_RATE.md) |
| 与上游功能的关系 | [与上游的区别](@REPOSITORY@/blob/main/guides/UPSTREAM_DIFFERENCES.md) |
| 报错与恢复 | [故障排查](@REPOSITORY@/blob/main/guides/TROUBLESHOOTING.md) |
| 更新、迁移与回滚 | [更新与回滚](@REPOSITORY@/blob/main/guides/UPDATE_AND_ROLLBACK.md) |
| 本机数据和反馈前脱敏 | [隐私与安全](@REPOSITORY@/blob/main/guides/PRIVACY_AND_SAFETY.md) |

## 更新与反馈

已使用独立 GKH 版本时，保持默认 **Stable**，通过 **MFAAvalonia 内置 GitHub 资源更新**获取完整包。旧 `vMAJOR.MINOR.PATCH+gkh.*` 组合版本仍需一次手工迁移；不要切到 Beta 或 Alpha 寻找新版。完整包会共同更新程序、资源和竞技场基线，不需要手工逐组件校验或第二套客户端更新器。当前尚未完成经 MFA 下载、应用和重启的完整实机升级验收，详细边界见[更新与回滚](@REPOSITORY@/blob/main/guides/UPDATE_AND_ROLLBACK.md)。

版本使用独立语义化版本 (Semantic Versioning, SemVer) `vMAJOR.MINOR.PATCH`；具体更新内容见每版 Release Notes，上游基线另行记录。本项目不会声明上游 `MaaGakumasu` 的 MirrorChyan 资源 ID，完整包更新使用本项目 GitHub Releases。

问题、建议与使用疑问请先阅读[反馈与分流](@REPOSITORY@/blob/main/SUPPORT.md)，再提交到[本项目 Issues](@REPOSITORY@/issues)。能在官方 MaaGakumasu 中复现的上游问题，请提交到[上游 Issues](https://github.com/SuperWaterGod/MaaGakumasu/issues)。反馈时提供版本、任务入口、窗口尺寸、失败位置和脱敏后的错误信息；不要上传账号信息、完整配置、日志包或未打码游戏截图。

## 贡献

欢迎提交可复现的问题、文档改进、测试和代码修复。较大改动请先开 Issue 讨论范围，小型修复可直接提交拉取请求 (Pull Request, PR)。开发与来源要求见[贡献说明](@REPOSITORY@/blob/main/CONTRIBUTING.md)。

## 来源与声明

客户端与上游日常能力来自 [MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)，由 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 和 [MFAAvalonia](https://github.com/MaaXYZ/MFAAvalonia) 提供框架与界面支持。竞技场模拟使用固定版本的 [gakumas-tools](https://github.com/surisuririsu/gakumas-tools)。感谢这些项目的开发者与贡献者。

本项目代码按 [AGPL-3.0](LICENSE) 提供；第三方组件保留各自许可证，游戏文字、图形及其他内容权利归原权利人。固定来源、识别资源及第三方通知见[来源说明](ASSET_PROVENANCE.md)与包内 `THIRD_PARTY_NOTICES`。

本项目与游戏官方无隶属或背书关系。程序通过截图、识别和正常界面输入工作，不读取游戏进程内存或处理 DMM 账号密码。自动化可能带来账号和资源消耗风险，请阅读使用文档并自行判断是否使用。
