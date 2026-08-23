## `@VERSION@` 竞技场公开预览版

这是 Gakumas Helper 的首个 MaaGakumasu Windows x86_64 完整派生预览包，包含竞技场编成读取、离线计分、保守胜率选敌、挑战结果记录和配套识别资源。

### 安装与更新

- 下载 `MaaGakumasu-win-x86_64-@VERSION@.zip`，核对 checksums，解压到新的独立目录，并以管理员权限启动 `MaaGakumasu.exe`。
- 请勿覆盖现有安装；保留旧目录用于回滚。
- 本版本使用 SemVer 构建元数据，与同一基础版本的上游包优先级相同。首次从对应上游版或同基础版本的旧派生包切入时必须手动安装，不能依赖同版自动更新。

### 使用入口

- 首次先添加并运行“重算竞技场己方总分”；成功后再运行“竞技场胜率分析并挑战（单次）”或“每日挑战 → 胜率选敌”。
- 默认门槛为三对手家族校正后的单侧 95% Wilson 下限达到 70%，模拟 2000 次，模拟超时 180 秒。
- 完整步骤、缓存边界和故障处理见仓库[指导中心](https://github.com/Kirinigh/Gakumas_Helper/blob/main/guides/README.md)。

### 预览边界

- 本包可供试用，不代表竞技场全部跨窗口、盲测或更大分辨率验收已经完成。
- 无法唯一确认身份、页面、字段、赛季或结果时会失败关闭，不会用低可信结果继续挑战。
- 当前竞技场已测范围为 Windows 11 + DMM PC，最低竖向截图短边 720；540×960 不支持。
- 当前只有 GitHub 更新仓库指向本项目；包内上游 `MaaGakumasu` MirrorChyan 入口不是 Gakumas Helper 更新通道，请勿用于本派生版。模型/数据独立热更新及自动更新/失败回滚完整演练仍未完成。

### 可复核信息

- 公开源码提交：`@SOURCE_REVISION@`
- 完整包 SHA-256：`@PACKAGE_SHA256@`
- Release manifest 与 `GakumasHelper-checksums-@VERSION@.txt` 和完整包一并提供。
