# MaaGakumasu Gakumas Helper

这是 MaaGakumasu 的公开派生预览版，整合了竞技场编成读取、离线计分与胜率选敌等功能。

- 当前版本：`v1.4.8+gkh.260823`
- 更新仓库：[https://github.com/Kirinigh/Gakumas_Helper](https://github.com/Kirinigh/Gakumas_Helper)
- 上游项目：[SuperWaterGod/MaaGakumasu](https://github.com/SuperWaterGod/MaaGakumasu)
- 许可证：AGPL-3.0；内嵌竞技场模拟引擎另按其 BSD-3-Clause 与 Node.js 许可发布

## 安装

从当前版本的 GitHub Release 下载 `MaaGakumasu-win-x86_64-v1.4.8+gkh.260823.zip`，解压到新的独立目录后，通过管理员权限启动 `MaaGakumasu.exe`。不要直接覆盖已有安装目录；需要保留旧目录作为回滚版本。

`v1.4.8+gkh.260823` 使用 SemVer 构建元数据标识，与上游 `v1.4.8` 具有相同的版本优先级。因此从既有 `v1.4.8` 或本地 `v1.4.8+...` 首次切入本派生通道时，需要手动安装本版；后续自动更新必须使用具有更高 SemVer 优先级的新版本。

## 预览版边界

这是预发布版本，不代表竞技场全部识别验收或项目最终稳定版已经完成。无法唯一确认身份、页面或安全状态时，功能会停止并报告原因；发布和更新本身不会自动执行游戏操作。

公开包不包含用户配置、日志、缓存、截图、对局记录、私有训练语料、凭据或本机路径。源码仓库使用独立的单提交公开快照，不包含内部开发仓库历史和任务管理记录。

## 来源

本仓库提供派生 Agent、资源、模型、数据及构建脚本的对应源码。完整 Windows 包复用固定 MaaGakumasu 上游发行版；竞技场模拟器来自固定版本的 [surisuririsu/gakumas-tools](https://github.com/surisuririsu/gakumas-tools)，具体版本、文件散列和许可证记录在 Release manifest 与包内清单中。
