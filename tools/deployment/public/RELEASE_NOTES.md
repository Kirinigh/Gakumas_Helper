## `@VERSION@` 竞技场公开预览版

这是 Gakumas Helper 的首个 MaaGakumasu 完整 Windows x86_64 派生预览包，包含竞技场编成读取、离线计分、胜率分析与选敌入口。

### 安装与更新

- 下载 `MaaGakumasu-win-x86_64-@VERSION@.zip`，解压到新的独立目录，并以管理员权限启动 `MaaGakumasu.exe`。
- 请勿直接覆盖现有安装；保留旧目录用于回滚。
- 本版本使用 SemVer 构建元数据，与上游 `v1.4.8` 优先级相同。首次从上游版或旧的 `v1.4.8+...` 切入时必须手动安装；不能依赖同版自动更新。

### 预览边界

- TASK-085、TASK-400、TASK-410 的剩余验收门不会因为本包发布而视为完成。
- 无法唯一确认身份、页面或安全状态时，竞技场流程会失败关闭，不会用低可信结果继续挑战。
- 发布动作本身不执行游戏输入；MirrorChyan 与独立模型/数据热更新尚未启用。

### 可复核信息

- 公开源码提交：`@SOURCE_REVISION@`
- 完整包 SHA-256：`@PACKAGE_SHA256@`
- Release manifest 和 `GakumasHelper-checksums-@VERSION@.txt` 与完整包一并提供。
