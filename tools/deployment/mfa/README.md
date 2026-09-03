# MFAAvalonia 资源更新源最小补丁

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

补丁只修改 `VersionChecker.cs` 的两个资源更新集中入口：

1. 检查远端资源版本前；
2. 下载并应用网络资源包前。

当调用方选择 MirrorChyan、但当前 `interface.json` 没有资源 RID 时，两处都回落到 `interface.json.github`。显式 GitHub 路径、有合法 RID 的上游路径以及本地 ZIP 更新路径保持原语义。

GKH 的 `interface.json` 同时不得携带 `mirrorchyan_rid` 或 `mirrorchyan_multiplatform`。只删除 RID 会使遗留 `DownloadSourceIndex=1` 的安装无法更新；只修改默认设置又无法覆盖既有配置，因此这两个入口的回落是干净安装与既有安装共同需要的最小闭环。

## 复核与构建

在固定源码 ZIP 解压根执行：

```powershell
git apply --check -- "<GKH-source>\tools\deployment\mfa\MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
git apply -- "<GKH-source>\tools\deployment\mfa\MFAAvalonia-v2.15.2-resource-update-github-fallback.patch"
dotnet restore .\MFAAvalonia\MFAAvalonia.csproj -r win-x64
dotnet build .\MFAAvalonia\MFAAvalonia.csproj -c Release -r win-x64 --no-restore --no-incremental --disable-build-servers -m:1 -p:SourceRevisionId=6065fe33798b72906c5079fa6f210646801d9a5c -p:ContinuousIntegrationBuild=true -p:Deterministic=true -p:UseSentryCLI=false -p:PathMap="<MFAAvalonia 项目绝对目录>=/_/MFAAvalonia/"
```

Sentry 6.8.0 会把 `ProjectDir` 作为 `Sentry.ProjectDirectory` 程序集元数据写入 Core。补丁中的 `Directory.Build.targets` 只在 `WriteSentryAttributes` 执行期间把该值固定为 `/_/MFAAvalonia/`，随后立即恢复；构建命令再用 `PathMap` 把真实源码前缀映射到同一虚拟路径。这样不会污染工程解析或项目引用，不会泄露本机构建目录，也不会关闭 Sentry 的源码相对路径能力。省略补丁或 `PathMap` 均不属于有效发布构建。

构建前显式设置 `AVALONIA_TELEMETRY_OPTOUT=1`，并先对同一工程、配置和 runtime 执行一次带 `--disable-build-servers -m:1` 的 `dotnet clean`。`-m:1` 是发布构建硬约束：它避免固定 SDK 为项目引用残留大量并行节点；构建结束必须按该 SDK 的 `dotnet.exe` 精确复查零残留。正式 bundle 还须以同一冻结输入干净复建两次并取得相同 DLL SHA-256。

发布构建不直接信任任意 DLL。它要求一个仅含 `manifest.json` 与 `MFAAvalonia.Core.dll` 的 `READY` 交接目录，逐项核对上述来源、补丁散列、SDK、稳定路径映射、上游输入 DLL、实际输出 DLL 和许可证；最终完整包还会重新核对包内 DLL 字节。

对应源码由“固定公开上游源码 + 本补丁”完整提供。发布包不得包含本机 SDK、NuGet 缓存、构建目录、用户配置或日志。
