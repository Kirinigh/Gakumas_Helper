<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

<img src="https://raw.githubusercontent.com/SuperWaterGod/MaaGakumasu/90cc001778cd265d1bd0158e44f8c51ea5c756fa/logo.png" alt="MaaGakumasu 标志" width="160" height="160">

# Gakumas Helper

**《学园偶像大师》日常与竞技场小助手**

基于 [MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)，在同一个客户端里收日常、跑培育、按胜率选择竞技场对手。

[![Latest](https://img.shields.io/github/v/release/Kirinigh/Gakumas_Helper?display_name=tag&label=Latest&color=ec4899)](@REPOSITORY@/releases/latest) [![Windows x86_64](https://img.shields.io/badge/Windows-x86__64-0078D4)](#compatibility) [![AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue)](LICENSE)

**[下载最新版](@REPOSITORY@/releases/latest)** · **[使用手册](@REPOSITORY@/blob/main/guides/README.md)** · **[更新记录](@REPOSITORY@/releases)** · **[反馈问题](@REPOSITORY@/issues/new/choose)**

</div>

> [!WARNING]
> 本项目是非官方自动化工具。使用时可能发生误操作、资源损失，账号也可能受到游戏运营方处罚。请先阅读[免责声明](#disclaimer)，第一次运行时留意游戏画面。

**目录：** [功能概览](#features) · [竞技场](#arena) · [下载安装](#install) · [适配与注意事项](#compatibility) · [文档](#docs) · [更新](#updates) · [反馈与贡献](#contributing) · [免责声明](#disclaimer) · [鸣谢](#credits)

<a id="features"></a>

## 功能概览

| 功能 | 可以做什么 |
| --- | --- |
| 日常领取 | 活动费、邮箱礼物、任务奖励、每周免费礼包 |
| 安排工作 | 领取工作奖励，按设置选择偶像与工作时长 |
| 社团互动 | 请求物品，领取相关奖励 |
| 商店兑换 | 扭蛋、金币与 AP 商店任务，按所选项目执行 |
| 自动培育 | 保留上游的初、NIA 培育及相关选项，见下方说明 |
| 实用工具 | 支援卡库存识别等上游工具 |
| 竞技场 | 读取双方编成、模拟胜率、自动选敌，记录三个舞台及整场挑战的结果 |

日常、培育与实用工具沿用上游功能；本项目主要增加竞技场的编成读取和胜率选敌。详细区别见[与上游 MaaGakumasu 的区别](@REPOSITORY@/blob/main/guides/UPSTREAM_DIFFERENCES.md)。

### 自动培育

沿用上游的以下培育模式和选项：

- **初**：REGULAR、PRO、MASTER。
- **NIA**：PRO、MASTER。
- 指定或自动选择偶像、设置培育次数、自动选择支援卡。
- 使用体力药与培育道具、选择卡片优先级。
- 跳过准备阶段、考试失败重试；NIA 可选择降低试镜难度或手动接管。

> [!NOTE]
> 自动培育仍可能因弹窗、识别错误或游戏更新中断。本项目尚未逐项复测所有上游培育选项；第一次使用前，请检查体力药、道具和循环次数的设置。

<a id="arena"></a>

## 竞技场

Maa 里有三个相关入口：

| 任务入口 | 用途 |
| --- | --- |
| **重算竞技场己方总分** | 读取己方三个舞台的编成并保存，用于首次使用、换编成或换期数后 |
| **竞技场胜率分析并挑战（单次）** | 读取三名对手，选出对手并挑战一场 |
| **每日挑战 → 胜率选敌** | 每场重新读取对手，挑战后记录结果、返回竞技场，继续剩余次数 |

选敌时优先选择达到配置门槛的对手，默认门槛为 **70%**。**无人达标时，仍会挑战预测胜率最高者**，并在日志中注明未达标。

结算记录三个舞台各自的赢家和最终胜负，分数读不准不会阻止保存。部分临时通信、弹窗和读取错误可以自动重试；仍未恢复时会停止，并保留未完成对局。

**换了己方编成或竞技场期数，记得重新运行己方重算。** 程序不会自动发现你换过卡。每日任务也可以临时开启“自动重算己方数据”，首场读取后，后续场次复用。

> [!IMPORTANT]
> 竞技场还在持续改进，暂不保证能连续完成每日全部挑战。建议使用 **720×1280 或更大**的窗口，540×960 下的完整日常流程尚未测通。模拟胜率仅供参考，已知问题见[竞技场指南](@REPOSITORY@/blob/main/guides/ARENA_WIN_RATE.md)和[更新记录](@REPOSITORY@/releases)。

<a id="install"></a>

## 下载安装

### 1. 下载并解压

打开 **[Latest Release](@REPOSITORY@/releases/latest)**，下载：

| 系统 | 文件 |
| --- | --- |
| Windows x86_64 | `MaaGakumasu-win-x86_64-vXXX.zip` |

`vXXX` 是实际版本号。完整包自带 Python 环境，无需另装 Python。请完整解压到一个新文件夹，别把文件混入上游 Maa 或正在使用的旧目录。

### 2. 启动并连接

在解压目录里运行：

```text
deployment\Start-MaaGakumasu-Admin.cmd
```

按系统提示以管理员身份启动。DMM 用户先登录游戏，在 Maa 中选择 **PC** 控制器和 **DMM** 资源，再连接游戏窗口。

第一次连接或启动失败时，先看[首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md)和[故障排查](@REPOSITORY@/blob/main/guides/TROUBLESHOOTING.md)。

### 3. 添加任务

收日常或跑培育，可以从对应预设开始，运行前检查其中勾选的任务。

第一次用竞技场，建议按这个顺序：

1. 单独运行 **重算竞技场己方总分**。
2. 确认期数与己方编成摘要正确。
3. 运行 **竞技场胜率分析并挑战（单次）**。
4. 熟悉后再使用 **每日挑战 → 胜率选敌**。

单次挑战和每日挑战都会消耗竞技场挑战次数。默认每名成员模拟 **2000** 次，模拟时限 **180 秒**；读取双方编成另需时间，整个任务可能持续数分钟。

<a id="compatibility"></a>

## 适配与注意事项

| 环境 | 当前情况 |
| --- | --- |
| 发布包 | 目前提供 Windows x86_64 完整包 |
| MaaFramework | 随包使用 5.13.0，已包含竞技场卡片定位兼容处理 |
| 竞技场主要测试环境 | Windows 11、DMM PC 版、日文游戏界面 |
| 推荐游戏窗口 | 竖向 9:16，物理尺寸 720×1280 或更大 |
| 540×960 小窗口 | 已完成编成读取测试，完整日常挑战尚未通过 |
| 模拟器、汉化资源与其他系统 | 上游有相应支持，本项目竞技场未完成同等测试 |

Maa 返回的识别图像可能经过缩放，与游戏设置中的窗口尺寸不同；[首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md)有具体说明。

- 运行时请让 Maa 操作游戏，避免同时点击、拖动窗口或遮挡游戏画面。
- 开始前确认体力药、商店购买、培育循环等选项符合你的打算。
- 报错后先记下提示和发生位置，不要靠反复点击开始跳过问题。

<a id="docs"></a>

## 使用文档

| 文档 | 内容 |
| --- | --- |
| [首次使用](@REPOSITORY@/blob/main/guides/GETTING_STARTED.md) | 下载、连接、首次竞技场运行与参数设置 |
| [竞技场指南](@REPOSITORY@/blob/main/guides/ARENA_WIN_RATE.md) | 己方缓存、选敌、结果记录 |
| [故障排查](@REPOSITORY@/blob/main/guides/TROUBLESHOOTING.md) | 常见提示与处理方法 |
| [更新与回滚](@REPOSITORY@/blob/main/guides/UPDATE_AND_ROLLBACK.md) | 自动更新、手动迁移、恢复旧版本 |
| [与上游的区别](@REPOSITORY@/blob/main/guides/UPSTREAM_DIFFERENCES.md) | 本项目新增和保留的功能 |
| [隐私与安全](@REPOSITORY@/blob/main/guides/PRIVACY_AND_SAFETY.md) | 本地记录、截图与反馈前脱敏 |

<a id="updates"></a>

## 更新

使用默认 **Stable** 通道，通过 **MFAAvalonia 内置 GitHub 资源更新**检查新版。下载页中标为 **Latest** 的是当前正式发布版。

从旧 `vMAJOR.MINOR.PATCH+gkh.*` 组合版本首次迁移，需要手动下载完整包。不要切换到 Beta 或 Alpha 找新版，这两个通道可能选到编号更高的旧组合版本。

本项目通过自己的 GitHub Releases 更新，暂不支持 Mirror 酱。自动更新仍在测试中。如果更新失败，可以改用完整包安装，操作方法见[更新与回滚](@REPOSITORY@/blob/main/guides/UPDATE_AND_ROLLBACK.md)。

### 后续计划

- [ ] 完成连续多日的竞技场日常测试。
- [ ] 补齐小窗口和更多卡片、道具的实测。
- [ ] 减少重复识别和等待，缩短读取时间。
- [ ] 补完客户端自动更新测试。

<a id="contributing"></a>

## 反馈与贡献

遇到问题，请先搜索 [Issues](@REPOSITORY@/issues)，再通过[问题反馈表单](@REPOSITORY@/issues/new/choose)提交。写清以下信息会更容易复现：

- 使用版本、Windows 版本、游戏窗口大小。
- 运行的任务、完整报错，以及出错前做了什么。
- 识别错误发生在哪个舞台、哪名成员、哪张卡或哪个道具槽。

可以附上少量脱敏日志。**请勿上传整份配置、日志包、缓存或原始游戏截图。** 具体说明见[反馈指南](@REPOSITORY@/blob/main/SUPPORT.md)。

在上游 MaaGakumasu 原版也能复现的问题，可以反馈到[上游 Issues](https://github.com/SuperWaterGod/MaaGakumasu/issues)。暂时不能确定问题归属时，发到本项目即可。

欢迎修正文档、补充测试和提交代码。较大的改动请先开 Issue 讨论；小修复可以直接提交拉取请求 (Pull Request, PR)。开发要求见[贡献说明](@REPOSITORY@/blob/main/CONTRIBUTING.md)。

<a id="disclaimer"></a>

## 免责声明

**下载和运行前，请确认你能接受以下风险。**

### 非官方项目

Gakumas Helper 是免费开源的第三方工具，与《学园偶像大师》的开发商、发行商及运营方没有隶属或合作关系，也不代表获得了官方认可。

### 账号与操作风险

使用自动化工具可能违反游戏的使用条款，导致警告、限制、封禁或数据回滚。请自行阅读并遵守游戏规则；本项目不承诺账号安全。

识别错误、网络异常、游戏更新或配置不当，都可能让程序点错按钮、停止运行，或消耗体力、货币、道具和挑战次数。竞技场模拟也不能保证实际胜负。请先了解所选任务，再自行判断是否使用。

### 无担保与责任说明

软件按“现状”提供，不保证稳定性、准确性或持续可用性。无担保及责任限制以 [AGPL-3.0 第 15—17 条](https://www.gnu.org/licenses/agpl-3.0.html#section15)及适用法律为准。

### 游戏资源与版权

游戏图片、文字、音频、商标等内容的权利归原权利人所有。本项目的开源许可证不改变这些资源的权利归属，也不授予额外的游戏素材使用许可。

## 开源协议

本项目代码使用 [AGPL-3.0](LICENSE) 许可证。第三方代码、模型与其他资源分别遵守各自的许可证和使用条件，详见[来源说明](ASSET_PROVENANCE.md)与包内 `THIRD_PARTY_NOTICES`。

<a id="credits"></a>

## 鸣谢

感谢以下项目的开发者和贡献者：

- [MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)：上游客户端、日常任务与培育功能；本页标志沿用上游。
- [MaaFramework](https://github.com/MaaXYZ/MaaFramework)：自动化框架。
- [MFAAvalonia](https://github.com/MaaXYZ/MFAAvalonia)：客户端界面。
- [gakumas-tools](https://github.com/surisuririsu/gakumas-tools)：竞技场模拟。
- [琴音小助手](https://github.com/XcantloadX/kotones-auto-assistant)：README 写作参考。

其他依赖与资源来源见[来源说明](ASSET_PROVENANCE.md)。如果这个项目帮到了你，欢迎点一个 Star。
