# MFAAvalonia 客户端补丁

Gakumas Helper 复用 MFAAvalonia 的资源更新和日志导出入口。本目录保存固定上游源码上的差分、离线回归和构建契约。

## 固定输入

- 上游仓库：`https://github.com/MaaXYZ/MFAAvalonia`
- 标签：`v2.15.2`
- 提交：`6065fe33798b72906c5079fa6f210646801d9a5c`
- 源码 ZIP：`MFAAvalonia-2.15.2.zip`
- 源码 ZIP SHA-256：`76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A`
- `MFAAvalonia/Helper/VersionChecker.cs` 原始 Git blob：`6e6d1118fa414ba21c7efa4f15a58ad95dd08bd7`
- `MFAAvalonia/Helper/FileLogExporter.cs` 原始 Git blob：`689a4f592e24dd85b95ff3fc198aa38803c7306a`
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

日志导出补丁修改 `FileLogExporter.cs` 和导出窗口的自定义日志标签，补上精简界面后反馈包缺少详细记录的问题：

- 原“自定义日志”选项同时收集 `.local/runtime-data/logs` 中的日期日志、标准库回退日志和轮转 ZIP，以及 `.local/arena-win-rate/reader-failures` 中的 `evidence.json`。已有 `custom.log*` 仍可导出，不收集其它缓存或对局结果。
- 失败事务中的已有图片归入错误图片，使用相同的图片开关及时间筛选。旧无选项入口共用这份目录清单，并保留最近五天和排除识别图片的规则。正常窗口入口的原日志选择规则保持不变。
- 使用共享读取复制原文件，ZIP 不作为文本处理；不裁剪长日志。导出包内的 `export-report.txt` 列出所选范围、已包含文件及无法读取的相对路径。它仅随本次导出生成，不写入运行目录。
- 部分文件或选中的日志根无法读取时，返回 `Partial` 并显示“部分导出”；全部无法读取时返回 `Failed` 并说明读取失败。不存在可选失败目录不算错误，未选择的目录不可读也不影响结果。失败图片存在但所选元数据没有生成时，报告缺失；主动取消自定义日志时说明元数据未被选择。

这一层不采集新截图、不追加识别，也不把详细 JSON 重新显示到任务界面。包内“完整导出”仅表示所选文件已复制，不能据此认定所有失败现场都已留存。Core 清单的补丁范围需包含 `diagnostic_log_export`；旧 Core 不能作为本修复已生效的证据。

动态选项补丁为 `select` 新增可选的 `dynamic_cases: true`。未启用的选项保持原有行为。启用后，生成下拉框、展开下拉框及任务结束后恢复可用时，从当前数据目录的 `interface.json` 读取同名选项；不重新加载全部界面、语言或任务，也不执行游戏动作。

- 磁盘列表必须完整保留现有 cases 的顺序、名称、管道覆盖及子选项；允许更新标签、标签参数、描述、兼容项的替代目标和默认项，并在末尾追加名称唯一的新项。整个候选验证通过后才修改内存；文件不完整、删项、重排或改变旧项管道与子选项时继续保留原列表，并将原因写入诊断日志。
- 带有 `replacement_case` 的兼容项不显示在下拉列表中，其目标必须存在且不能再是兼容项；新默认值也必须指向具体项。其余项按名称末尾数值降序显示，排序仅作用于显示副本。持久化的 `index` 始终对应原 cases 顺序，用户选择后仍按名称找回原索引。
- 旧配置选中了兼容项时，统一解析器把它一次性迁移为目标的真实索引并保存。配置加载、下拉初始化和开始任务前均使用此解析器；不展开选项直接开始也会迁移。开始前只在空闲时刷新定义，迁移后的具体选择不会因兼容项将来换目标而漂移。
- 单纯更新标签或新增期数不保存配置、不重建子选项，也不重复订阅语言变化；只有选择实际发生迁移时才保存。共享同一定义的多个下拉框各自更新显示副本。标签与描述由已验证的本地定义提供，这一层不判断哪个期数已经开放。
- case 可附带 `label_args` 字符串字典。先按现有语言表解析 `label`，再将其中的 `{key}` 替换为对应参数；未知占位符保留，参数值不递归展开。例如静态简繁格式标签与 `{"season": "52"}` 组合，无需改写语言表或重新加载语言系统。

此修复对应 Core 清单中的 `dynamic_option_cases`，必须使用包含它的新构建；不新增选季状态文件或修改用户配置格式。

首启默认与布局兼容补丁继续使用现有配置加载路径：

- 公开包根目录的 `config.template.json` 只提供 `UI.LiveView.EnableLiveView` 布尔默认值。配置缺少该键时才补入内存；默认配置、命名配置和新建配置共用此规则。已有值和实例级、旧 scoped 用户选择优先，模板中的其它键不应用。模板缺失、损坏或类型错误时保留既有缺省行为并按需记录诊断，不阻断客户端启动。
- 加载模板不直接改写用户配置文件；后续保存沿用客户端原路径。`NoAutoStart`、任务参数及自动更新等设置不受模板影响。随包只携带根模板，不携带实际 `config.json` 或 `config` 目录。
- 多实例迁移保留全局实时视图值，避免首次启动清理临时 default 实例时丢失模板默认值或用户显式选择；实例自己的值仍优先，其余旧设置继续按原路径迁移。离线回归包含实际 plain-key 迁移方法和临时实例清理后的新实例取值。
- `resource/mfa_layout.json` 仅在没有已保存布局时提供默认布局。已有布局及尺寸、旧版布局键均优先；资源散列或尺寸变化不再覆盖用户布局。首次默认布局仍使用原有布局键和保存回调，不增加迁移状态文件。

这一层对应 Core 清单的 `startup_template_and_saved_layout`。`test_startup_defaults.py` 从固定源码应用补丁，编译实际配置加载、实例取值及布局选择方法，使用 `startup_defaults_harness.cs` 核对缺键、旧值、命名配置、实例优先级、异常模板、首次布局和重复加载。渲染及保存回调在该离线回归中使用测试替身，完整 Core 构建负责真实类型集成；不能仅据此宣称界面实机验收完成。

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

## 离线日志导出回归

```powershell
python tools/deployment/mfa/test_log_export.py --source-zip "<固定源码 ZIP>" --dotnet "<现有 SDK>\dotnet.exe" --work-dir "<尚不存在的测试目录>"
```

测试校验固定源码 ZIP 和原始 `FileLogExporter.cs` Git 对象，应用同一补丁，然后编译完整的实际导出器源码；只替代界面通知和文件选择器接口。两种入口均实际生成 ZIP，并逐项核对文件清单和原字节。

回归覆盖两种 Python 日志后端的文件命名、轮转 ZIP、错误元数据、图片开关及时间范围、活动日志共享读取、独占文件、真实目录访问控制、部分导出提示、空目录和超过 42,000 行的日志。目录访问控制用例只在新测试目录内临时拒绝读取并恢复原权限，需要能设置该目录权限的执行端；不接触正式安装或游戏。日志和实际 ZIP 留在测试目录，整个测试不联网、不恢复依赖。

## 离线动态选项回归

```powershell
python tools/deployment/mfa/test_dynamic_option_cases.py --source-zip "<固定源码 ZIP>" --dotnet "<现有 SDK>\dotnet.exe" --newtonsoft "<现有离线依赖>\Newtonsoft.Json.dll" --work-dir "<尚不存在的测试目录>" --before "<真实投影前 interface.json>" --after "<同次投影后 interface.json>"
```

`--before` 与 `--after` 是可选的成对参数。测试应用当前补丁后，直接编译完整 `DynamicOptionCases.cs`、实际 `CreateComboBoxControl`、开始前迁移和 case 的 `UpdateDisplayName` 方法；只替代渲染、模型生成部分、语言查询、空闲绑定与配置保存回调，不复制刷新和选择算法。覆盖初始化、下拉展开、空闲恢复、共享定义、固定索引、兼容项隐藏及一次性迁移、不展开直接开始、运行中不刷新、标签与描述更新、子选项保留及无效文件的整体拒绝。它读取仓库实际简繁格式模板验证参数渲染，并可使用 Python 生成的真实前后定义核对跨语言合同。

测试使用现有编译器、标准库引用与 Newtonsoft.Json，不启动 Maa、不读取游戏、不恢复依赖。完整 Core 编译及双次复建另行验证 Avalonia 事件和实际模型集成；该方法测试不等同于界面点击验收。
