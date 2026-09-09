# MFAAvalonia 资源更新补丁

Gakumas Helper 的客户端更新继续复用 MFAAvalonia 原生资源更新，不增加第二套更新器。本目录只保存固定上游源码上的最小差分及可复核的构建契约。

## 固定输入

- 上游仓库：`https://github.com/MaaXYZ/MFAAvalonia`
- 标签：`v2.15.2`
- 提交：`6065fe33798b72906c5079fa6f210646801d9a5c`
- 源码 ZIP：`MFAAvalonia-2.15.2.zip`
- 源码 ZIP SHA-256：`76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A`
- `MFAAvalonia/Helper/VersionChecker.cs` 原始 Git blob：`6e6d1118fa414ba21c7efa4f15a58ad95dd08bd7`
- .NET SDK 版本组件：`10 / 0 / 400`，Windows x64 ZIP；实际版本号由三个组件以 `.` 连接
- SDK ZIP SHA-512：`9B8B88590E4DA131BFD0DA7AA089D0FC04D5418D5F8607EC13D55DC5A17B4399AFD54D496C12657FA05C6C6546DC5EAB930F26AC6C50F2D3A7712C0FB378C366`
- 上游许可证：GPL-3.0-only；固定源码根 `LICENSE` SHA-256：`3972DC9744F6499F0F9B2DBF76696F2AE7AD8AF9B23DDE66D6AF86C9DFB36986`，原文随包保存在 `THIRD_PARTY_NOTICES/MFAAvalonia-LICENSE`

## 差分边界

更新源回落修改 `VersionChecker.cs` 的两个资源更新集中入口：

1. 检查远端资源版本前；
2. 下载并应用网络资源包前。

当调用方选择 MirrorChyan、但当前 `interface.json` 没有资源 RID 时，两处都回落到 `interface.json.github`。显式 GitHub 路径、有合法 RID 的上游路径以及本地 ZIP 更新路径保持原语义。

GKH 的 `interface.json` 同时不得携带 `mirrorchyan_rid` 或 `mirrorchyan_multiplatform`。只删除 RID 会使遗留 `DownloadSourceIndex=1` 的安装无法更新；只修改默认设置又无法覆盖既有配置，因此这两个入口的回落是干净安装与既有安装共同需要的最小闭环。

下载处理保留原有更新路径和 3 次尝试额度，修正以下行为：

- 底层 HTTP、传输流或文件异常返回失败时，上层按原有 2 秒、4 秒间隔重试；仅成功才提前返回，不增加重试额度。
- 空响应、声明长度不符或读取失败不能达到 100%；完整读取并刷新文件后才标记下载完成。失败时仅清理本次已创建的半包，重试从头下载。
- 下载失败耗尽后，资源、MFA 和 MaaFw 更新入口向既有任务管理器传播失败，避免正常返回后记为“更新任务完成”。资源包缺失、摘要不符或结构无效同样传播失败；“已下载更新包”日志仅在传输成功后记录。原有摘要和包结构校验继续在应用文件前执行。
- `MaaProcessor` 的任务初始化不再清理资源更新器拥有的 `temp_res`，避免后台下载与任务启动争用目录；其它任务初始化行为保持原样。

没有长度头的合法响应仍可下载；其传输结束本身不能证明归档完整，后续原有摘要校验、解压及资源包结构验证继续负责拒绝损坏内容。没有新增更新器、断点续传、用户设置或版本自动递增行为。

## 复核与构建

在固定源码 ZIP 解压根执行：

```powershell
git apply --check -- "<GKH-source>\tools\deployment\mfa\MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
git apply -- "<GKH-source>\tools\deployment\mfa\MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
dotnet restore .\MFAAvalonia\MFAAvalonia.csproj
dotnet build .\MFAAvalonia\MFAAvalonia.csproj -c Release -r win-x64 --no-restore --no-incremental --disable-build-servers -m:1 -p:SourceRevisionId=6065fe33798b72906c5079fa6f210646801d9a5c -p:ContinuousIntegrationBuild=true -p:Deterministic=true -p:UseSentryCLI=false -p:PathMap="<MFAAvalonia 项目绝对目录>=/_/MFAAvalonia/"
```

Sentry 6.8.0 会把 `ProjectDir` 作为 `Sentry.ProjectDirectory` 程序集元数据写入 Core。补丁中的 `Directory.Build.targets` 只在 `WriteSentryAttributes` 执行期间把该值固定为 `/_/MFAAvalonia/`，随后立即恢复；构建命令再用 `PathMap` 把真实源码前缀映射到同一虚拟路径。这样不会污染工程解析或项目引用，不会泄露本机构建目录，也不会关闭 Sentry 的源码相对路径能力。省略补丁或 `PathMap` 均不属于有效发布构建。

构建前显式设置 `AVALONIA_TELEMETRY_OPTOUT=1`、`DOTNET_CLI_TELEMETRY_OPTOUT=1`、`DOTNET_GENERATE_ASPNET_CERTIFICATE=false` 和 `DOTNET_ADD_GLOBAL_TOOLS_TO_PATH=false`，并先对同一工程、配置和 runtime 执行一次带 `--disable-build-servers -m:1` 的 `dotnet clean`。`-m:1` 是发布构建硬约束：它避免固定 SDK 为项目引用残留大量并行节点；构建结束必须按该 SDK 的 `dotnet.exe` 精确复查零残留。正式 bundle 还须以同一冻结输入干净复建两次并取得相同 DLL SHA-256。

已有缓存的离线构建须显式使用仅含本地缓存源的 NuGet 配置，并关闭 NuGet 网络审计；不得因缓存不足临时安装依赖。恢复依赖时保留工程自身的框架/运行时选择，仅在构建时指定 `win-x64`，避免给 `net8.0` 项目引用额外引入未使用的运行时包。源码解压根若位于另一 Git 工作树内，应用补丁时须令 `GIT_CEILING_DIRECTORIES` 指向解压根的父目录，并检查实际修改文件，防止 Git 静默跳过上游相对路径。

发布构建不直接信任任意 DLL。它要求一个仅含 `manifest.json` 与 `MFAAvalonia.Core.dll` 的 `READY` 交接目录，逐项核对上述来源、补丁散列、SDK、稳定路径映射、上游输入 DLL、实际输出 DLL 和许可证；最终完整包还会重新核对包内 DLL 字节。

对应源码由“固定公开上游源码 + 本补丁”完整提供。发布包不得包含本机 SDK、NuGet 缓存、构建目录、用户配置或日志。

## 离线下载回归

使用已固定的源码 ZIP 和已安装的 .NET 10 SDK：

```powershell
python tools/deployment/mfa/test_download_transport.py --source-zip "<固定源码 ZIP>" --dotnet "<现有 SDK>\dotnet.exe" --work-dir "<尚不存在的测试目录>"
```

测试先核对源码 ZIP SHA-256 和原始 `VersionChecker.cs` Git 对象，随后应用本补丁，从实际源码提取下载、重试、资源下载/摘要校验和更新队列代码编译执行。只替代界面、代理环境和解压出口，不复制下载算法；不运行 Maa 或更新安装。模拟 HTTP 仅监听本机回环地址，覆盖完整下载、断流后成功、HTTP 错误后成功、耗尽三次、短响应、空响应、无长度响应、分块传输中断、文件名扩展名调整、失败状态传播和任务清理期间保持活动下载。

编译直接使用 SDK 内置编译器与标准库引用，不执行依赖恢复或网络安装。测试目录保留编译日志、实际传输断言日志、生成的源码和 `result.json`。它证明固定源码中的传输及失败分支行为；正式 Core 仍须遵守上面的完整构建和双次复建契约。修改补丁会使既有 bundle 的补丁散列绑定失效，必须重新构建、验证新的 bundle 后才能纳入将来的发行包，不能把旧 DLL 当作本补丁已部署。
