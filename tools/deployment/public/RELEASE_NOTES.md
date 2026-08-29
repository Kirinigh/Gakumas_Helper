## `@VERSION@` 竞技场公开预览版

这是 Gakumas Helper 本次 MaaGakumasu Windows x86_64 完整派生预览包，包含竞技场编成读取、离线计分、保守胜率选敌、挑战结果记录和配套识别资源。

### 安装与更新

- 首次安装时下载 `MaaGakumasu-win-x86_64-@VERSION@.zip`，按需核对 checksums，解压到新的独立目录，并运行版本内 `deployment\Start-MaaGakumasu-Admin.cmd`。
- GKH 版本为 `@VERSION@`；对应的上游 Maa 标签在 Release manifest 与来源记录中单独固定。
- 从旧 `vMAJOR.MINOR.PATCH+gkh.*` 命名空间迁移时须手动安装本版本一次，并把 MFA 资源更新通道保持为默认 Stable。不要切换到 Beta 或 Alpha：在旧组合 Release 完成通道隔离前，这两个通道会把数值更高的旧版本误判为更新。Stable 看不到 prerelease，因此后续预览版本仍须手动安装。

### 使用入口

- 首次先添加并运行“重算竞技场己方总分”；成功后再运行“竞技场胜率分析并挑战（单次）”或“每日挑战 → 胜率选敌”。
- 默认门槛为三对手家族校正后的单侧 95% Wilson 下限达到 70%，模拟 2000 次，模拟超时 180 秒。
- 完整步骤、缓存边界和故障处理见仓库[指导中心](https://github.com/Kirinigh/Gakumas_Helper/blob/main/guides/README.md)。

### 预览边界

- 本包可供试用，不代表竞技场全部跨窗口、盲测或更大分辨率验收已经完成。
- 无法唯一确认身份、页面、字段、赛季或结果时会失败关闭，不会用低可信结果继续挑战。
- 当前竞技场已测范围为 Windows 11 + DMM PC，最低竖向截图短边 720；540×960 不支持。
- 当前只有 GitHub 更新仓库指向本项目；包内上游 `MaaGakumasu` MirrorChyan 入口不是 Gakumas Helper 更新通道，请勿用于本派生版。模型、图库和 UI 不独立热更新，而是随完整派生包由 MFA 原生入口统一应用；竞技场 RIS engine/data 仅按最新成功 production SHA 成对更新，不构成第二套客户端更新器。

### 可复核信息

- 公开源码提交：`@SOURCE_REVISION@`
- 完整包 SHA-256：`@PACKAGE_SHA256@`
- Release manifest 与 `GakumasHelper-checksums-@VERSION@.txt` 和完整包一并提供。
