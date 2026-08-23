# 更新与回滚

## 三类更新分别做什么

| 更新 | 负责内容 | 当前状态 |
| --- | --- | --- |
| Maa/MFA GitHub 资源更新 | `agent`、`resource`、任务/UI、模型和 `interface.json` | 派生包把 GitHub 更新仓库指向本项目；首次切入仍需手动安装 |
| pip 更新 | 内嵌 Python 运行依赖 | 沿用 Maa 的依赖入口；不负责竞技场模型、图库或 UI |
| 独立资产热更新 | 单独更新自训练模型、图库或规则数据 | 尚未启用，当前随完整派生包一起发布 |
| MirrorChyan | 上游 Maa 的第三方分发入口 | 当前没有 Gakumas Helper 专用通道；界面若仍显示上游 `MaaGakumasu` 标识，不要用它更新本派生版 |

上游 Maa 的资源更新会重写 Agent、资源和界面配置。不要把本项目文件手工覆盖到上游安装目录；下一次上游更新会把它们删除或替换，也可能形成版本不兼容的混合目录。

> [!WARNING]
> 当前只有 GitHub 更新仓库被改向本项目。包内仍可能保留上游 `MaaGakumasu` 的 MirrorChyan 元数据；该入口不是 Gakumas Helper 更新源，使用它可能切回上游资源并移除派生功能。

## 哪些版本首次安装必须手动完成

当派生版本只是在已安装基础版本后增加 `+gkh...` 时，`+` 后内容属于 SemVer 构建元数据，不提高版本优先级。Maa 更新器不会把同优先级版本当作升级目标。因此 Release Notes 标明“与上游同优先级”时，首次切入必须：

1. 从[本项目 Releases](https://github.com/Kirinigh/Gakumas_Helper/releases)手动下载完整 ZIP；
2. 核对 checksums；
3. 解压到新目录；
4. 以管理员身份启动新目录中的 `MaaGakumasu.exe`。

后续只有发布更高 SemVer 优先级的派生版本后，GitHub 更新入口才可能自动选择它。每一版仍以其 Release Notes 为准。

## 更新前检查

- 阅读 Release Notes，确认支持的系统、架构和预览边界；
- 下载完整 ZIP、Release manifest 和 checksums；
- 使用 `Get-FileHash <文件> -Algorithm SHA256` 与 checksums 对照；
- 保留当前可运行目录，不要先删除；
- 不把旧版的整个 `.local`、`config` 或 `appsettings.json` 公开上传或无差别复制到新版本。

Release manifest 会记录上游 Maa 版本、公开源码 revision、引擎 revision、模型/图库/适配器兼容字段和通知文件散列。checksums 证明下载字节是否一致，但不代表某项业务功能已经完成全部验收。

## 安全回滚

本预览版采用并列目录，而不是原地覆盖：

1. 关闭新版本 Maa，确认没有运行中的任务；
2. 启动之前保留的旧目录；
3. 如果新版曾开始真实竞技场挑战且结果未完整记录，先停止继续挑战，不要通过切换目录规避待决状态；
4. 在 Issue 中说明新旧版本、是否消耗次数和前台错误码。

回滚不需要删除新版本目录。待问题确认后再决定是否保留其本地缓存；任何删除都应由用户在确认目标目录后手工完成。

## 当前未完成边界

当前已建立版本清单、公开源码快照、完整包、校验和和隐私门，但下载失败、兼容失败、校验失败、候选切换和自动回滚的全套演练尚未完成。Gakumas Helper 专用 MirrorChyan 通道和模型/数据独立热更新也未启用。发生更新异常时，首选独立目录手动安装或回到保留的旧目录。
