# Gakumas Helper 版本更新与公开发布运行手册

本文是维护者执行上游同步、公开源码快照、完整包和 GitHub Release 的稳定操作契约。它只描述长期不变的步骤、停止条件和核对关系，不记录当前任务状态、某次发布进度或移动中的版本号。

本文不授予联网取得、远端写入、Release 发布、本机安装或切换活动版本的权限。每个发布批次仍须取得对应授权。

## 1. 当前适用范围

- 当前自动化只批准 `arena_preview` 公开预览版。
- 日期式预览版本使用 `vMAJOR.MINOR.PATCH+gkh.YYMMDD`。
- 正式稳定版、自动更新端到端验收和安装后自动回滚尚未由当前发布器完整支持；不得仅通过去掉 `--prerelease` 将预览包宣称为稳定版。
- 公开发布与本机安装是两个独立批次。发布流程不得覆盖现有安装，也不得自动切换 `current`。

## 2. 完成定义与硬对齐门

一次发布只有在下列关系全部成立时才算完成：

```text
远端 main SHA
= 新版本 tag 指向的 SHA
= GitHub Release 的 tag 目标 SHA
= 公开源码快照 SHA
= GAKUMAS_HELPER_BUILD.json 的 source.revision
= Release manifest 的 application.source_revision
```

同时必须满足：

| 对象 | 必须成立的关系 |
| --- | --- |
| `README.md`、`assets/interface.json`、`pyproject.toml` | 均声明本次派生版本和公开仓库 |
| 候选包 `interface.json`、构建清单、Release manifest | 版本、仓库、上游标签和源码 SHA 一致 |
| 完整 ZIP、Release manifest、checksums | 本地文件名、大小和 SHA-256 互相一致 |
| GitHub Release 资产 | 数量、文件名、大小、`uploaded` 状态和 digest 与本地一致 |
| Release 状态 | 资产复核前保持 draft；公开预览版发布后为非 draft、prerelease |

任一关系不成立时必须停止，不得用发布说明、手工页面编辑或“稍后补齐”代替硬对齐。

## 3. 发布状态机

发布批次只允许按以下顺序前进：

```text
PREPARE
  -> SOURCE_VERIFIED
  -> PUBLIC_SNAPSHOT_VERIFIED
  -> CANDIDATE_VERIFIED
  -> ASSETS_VERIFIED
  -> REMOTE_PREFLIGHT_PASSED
  -> REFS_ALIGNED
  -> DRAFT_VERIFIED
  -> PUBLISHED
```

- 当前状态的全部门未通过时，不得进入下一状态。
- 任何输入、版本、源码提交或资产字节变化都会使后续状态失效，必须从受影响的最早状态重新执行。
- 不得从已运行或被更新器混写的安装目录构建候选。
- 不得从含未提交改动的共享工作树构建公开包。

## 4. 版本语义

1. 批次必须固定上一通道版本，并传给 `--previous-channel-version`。
2. `v1.4.8+gkh.YYMMDD` 到 `v1.4.9+gkh.YYMMDD` 的基础版本升高，具有更高的 SemVer 优先级。
3. 两个版本若只在 `+gkh.YYMMDD` 不同，其 SemVer 优先级相同，因为 `+` 后是构建元数据。此类版本必须标记为人工安装，不能宣称可自动升级。
4. 低于上一通道版本的候选必须由构建器拒绝。
5. 若需要在同一基础版本上连续发布可自动升级的修复，不得临时发明版本格式；必须先单独批准并实现新的单调版本策略及迁移测试。
6. 构建清单中的 `automatic_update_e2e_verified` 为 `false` 时，即使版本优先级升高，发布说明仍须建议人工下载安装，直到成功更新和失败回滚演练完成。

## 5. PREPARE：冻结批次输入

每个批次在修改前固定并记录：

- 官方上游仓库、标签、完整提交 SHA；
- 官方 Windows x86_64 Release 文件名、大小、SHA-256；
- 上一公开通道版本和本次派生版本；
- 公开仓库 URL；
- 本地开发基线提交和隔离工作树；
- 竞技场引擎、Python 依赖、模型、图库和数据的固定 revision/schema；
- 许可证、再分发状态和所需人工复核；
- 本批次允许的联网、远端写入、发布和本机安装边界。

官方 Release digest、官方 checksums 和本地 SHA-256 必须一致。取得的二进制执行 Defender 扫描。远端标签、资产或来源无法唯一确认时停止。

所有命令在执行前还必须绑定为 PowerShell 变量，不在命令行中直接使用未加引号的占位符：

```powershell
$ReleaseWorktree = (Resolve-Path -LiteralPath '<VERIFIED_RELEASE_WORKTREE>').Path
$Python = (Resolve-Path -LiteralPath '<VERIFIED_PYTHON_EXE>').Path
$DevelopmentWorktree = (Resolve-Path -LiteralPath '<DEVELOPMENT_WORKTREE>').Path
$Repository = 'https://github.com/Kirinigh/Gakumas_Helper'
$Version = '<VERSION>'
$ExpectedRemoteMainSha = '<40_HEX_REMOTE_MAIN_SHA>'
$PublicParentDirectory = [System.IO.Path]::GetFullPath('<NEW_VERIFIED_PUBLIC_PARENT_DIRECTORY>')
```

- `$ReleaseWorktree` 必须是本批次已经验证的隔离发布工作树；三个发布脚本一律从该绝对路径调用，不依赖当前目录。
- `$Python` 必须是已经验证的虚拟环境解释器绝对路径。执行 `& $Python --version` 并记录版本；不得在命令失败时退回 PATH 中的 `python`。
- 所有已有输入路径用 `Resolve-Path -LiteralPath` 固定为绝对路径；尚不存在的输出目录用 `[System.IO.Path]::GetFullPath(...)` 固定，并确认其父目录正确且目标不存在。
- 其余尖括号值也先绑定为变量。PowerShell 变量作为独立参数传入，确保含空格的路径或标题不会被拆分。

## 6. SOURCE_VERIFIED：隔离同步与源码验证

1. 从已提交基线创建独立工作树和发布分支；共享脏工作树只读保护。
2. 用一次性上游 URL 查询和取得固定标签，不给开发仓库配置持久 remote。
3. 先比较上游版本差异和本地修改重叠，再合并固定上游标签。
4. 对发现的问题分别记录为上游原生、本地派生、混合或正常行为；正常行为只补必要说明，不伪装为代码缺陷。
5. 更新所有受影响的源码、资源、任务、数据、依赖、版本元数据、来源和用户说明。
6. 运行相关测试、全量 pytest、相关 Ruff、JSON/YAML 解析和 `git diff --check`。
7. 仅在仓库已存在可离线复用的 Node/npm 与 `maa-tools` 时运行其检查；不得为单次发布临时下载工具链。
8. 将发布输入固定为一个完整、已验证的开发提交 SHA。

## 7. PUBLIC_SNAPSHOT_VERIFIED：生成公开源码快照

先实时读取远端 `main`，并把它完整取得到一个不带 remote 的独立父仓库。禁止使用浅克隆、开发仓库或开发仓库的 linked worktree 充当公开父仓库：

```powershell
$RemoteMain = @(git ls-remote --exit-code $Repository 'refs/heads/main')
if ($LASTEXITCODE -ne 0 -or $RemoteMain.Count -ne 1) {
  throw 'Cannot uniquely verify public main'
}
$LiveRemoteMainSha, $RemoteRef = $RemoteMain[0] -split '\s+', 2
if ($LiveRemoteMainSha -notmatch '^[0-9a-f]{40}$' -or
    $RemoteRef -ne 'refs/heads/main' -or
    $LiveRemoteMainSha -ne $ExpectedRemoteMainSha) {
  throw 'Public main does not match the frozen parent'
}

git init -q -b main $PublicParentDirectory
git -C $PublicParentDirectory fetch --no-tags --no-recurse-submodules `
  --no-write-fetch-head $Repository $ExpectedRemoteMainSha
if ($LASTEXITCODE -ne 0) { throw 'Full public-parent fetch failed' }
git -C $PublicParentDirectory update-ref refs/heads/main `
  $ExpectedRemoteMainSha ('0' * 40)
git -C $PublicParentDirectory checkout -q -f main
```

如果保留了上一批构建器生成的独立公开快照，也可直接复用；但其 `HEAD`、`refs/heads/main` 必须同时等于实时远端 `main`，且仍须通过构建器的完整父历史验证。

以下命令使用第 5 节已经固定的变量：

```powershell
& $Python (Join-Path $ReleaseWorktree 'tools\deployment\build_public_source_snapshot.py') `
  --source-root $DevelopmentWorktree `
  --source-revision $DevelopmentSha `
  --output $PublicSnapshotDirectory `
  --version $Version `
  --repository $Repository `
  --public-parent-root $PublicParentDirectory `
  --public-parent-revision $ExpectedRemoteMainSha
```

生成后必须确认：

- 独立 Git 仓库、仅有 `refs/heads/main`、无 remote，`HEAD` 的唯一父提交精确等于 `$ExpectedRemoteMainSha`；
- `$ExpectedRemoteMainSha..HEAD` 恰好一个提交，完整可达历史只有一个根、无 merge，且开发提交对象不存在；
- 工作树干净，`git fsck --full --strict --no-reflogs` 通过且无不可达对象；
- 每个可达提交的原始 Git tree 路径与 blob 都通过公开允许清单和隐私扫描；空树、Windows 设备名、路径大小写碰撞、路径个人标识、伪装或嵌套 ZIP，以及归档成员路径、内容、comment/extra 元数据均失败关闭；不能以 `.gitattributes` 的 `export-ignore` 或 `export-subst` 隐藏内容；
- 无本机运行态、用户数据、凭据、链接/重解析点、内部任务标识或开发历史；
- 每个可达提交都使用固定通用作者、单行版本消息和与版本日期一致的提交日期，不含签名或额外提交头；
- `README.md`、`assets/interface.json`、`pyproject.toml` 的版本、仓库和许可证元数据正确；
- 使用已冻结解释器在公开快照内执行测试并返回 0：

```powershell
Push-Location -LiteralPath $PublicSnapshotDirectory
try {
  & $Python -m pytest -q -p no:cacheprovider
  if ($LASTEXITCODE -ne 0) { throw "Public snapshot pytest failed: $LASTEXITCODE" }
}
finally {
  Pop-Location
}
```

记录生成的公开提交 SHA。后续候选必须从这个公开提交构建，不能再从开发提交或可变工作树构建。

## 8. CANDIDATE_VERIFIED：从公开提交构建候选

```powershell
& $Python (Join-Path $ReleaseWorktree 'tools\deployment\build_derived_package.py') `
  --source-root $PublicSnapshotDirectory `
  --source-revision $PublicSha `
  --upstream-archive $UpstreamArchive `
  --upstream-sha256 $UpstreamSha256 `
  --upstream-tag $UpstreamTag `
  --engine-bundle $EngineBundle `
  --python-site-packages $PythonSitePackages `
  --output $CandidateDirectory `
  --update-repository $Repository `
  --derived-version $Version `
  --previous-channel-version $PreviousVersion
```

必须复核：

- `GAKUMAS_HELPER_BUILD.json`、`interface.json` 和关键文件散列；
- 上游标签/ZIP、公开源码 SHA、引擎、Python 包和更新契约；
- 构建器的内嵌 Python 冒烟使用隔离工作目录，且候选完整文件树在运行前后无新增、删除或内容变化；
- 内嵌 Python 与生产模块可导入；
- 正式候选不得直接用于图形界面启动冒烟；本阶段如需验证窗口行为，只能启动候选的可丢弃完整副本，完成后再次确认原候选文件树散列未变化；
- 候选目录 Defender 扫描无目标检测；
- 冒烟只验证启动/退出，不执行游戏任务或发送游戏输入；
- 不安装候选，不切换活动版本。

## 9. ASSETS_VERIFIED：生成不可变发布资产

```powershell
& $Python (Join-Path $ReleaseWorktree 'tools\deployment\build_release_asset.py') `
  --candidate $CandidateDirectory `
  --output-dir $ReleaseDirectory `
  --release-version $Version `
  --release-repository $Repository `
  --qualification arena_preview
```

仅允许生成三个最终资产：

1. `MaaGakumasu-win-x86_64-<VERSION>.zip`
2. `GakumasHelper-release-<VERSION>.json`
3. `GakumasHelper-checksums-<VERSION>.txt`

必须验证：

- ZIP 路径安全、无重复项、无链接、CRC 和逐成员 SHA-256 正确；
- 候选目录与 ZIP 成员清单、大小和内容完全一致；
- Release manifest 通过严格 schema，且源码 SHA、版本和组件关系正确；
- 完整序列化 manifest 和最终资产通过隐私/内部标识门；
- checksums 只引用本次 ZIP 和 manifest，散列与本地文件一致；
- 最终 ZIP Defender 扫描无目标检测；
- 将最终 ZIP 解压到新的可丢弃目录，在该副本中执行无游戏输入的启动/退出冒烟并核对窗口标题；不得运行或修改正式候选；
- Release Notes 根据实际 `update_contract` 描述版本优先级和人工安装边界。

在 `ASSETS_VERIFIED` 完成前，还必须冻结本次 Release 标题和说明文件。它们不是下载资产，但属于公开发布输入：

- 标题、说明中的版本、公开源码 SHA、包 SHA-256 和安装/更新语义与最终产物一致；
- 标题和说明通过与公开快照相同的本机路径、个人信息、凭据和内部任务标识扫描；
- 说明文件保存为 UTF-8，记录其 SHA-256；标题文本和说明文件路径绑定为 `$ReleaseTitle`、`$ReleaseNotes`；
- 后续任何标题、说明或说明文件字节变化都会使 `ASSETS_VERIFIED` 失效。

三个资产中的任一字节变化都会使 `ASSETS_VERIFIED` 失效。

## 10. REMOTE_PREFLIGHT_PASSED：远端写入前检查

远端写入前必须实时确认：

- 当前默认分支仍为 `main`；
- 远端 `main` 精确 SHA 与本批次记录的预期值一致；
- 目标 tag 不存在；
- 同版本 Release（包括 draft）不存在；
- GitHub 登录身份和仓库权限正确；
- 本地公开快照工作树干净，`HEAD` 等于 `<PUBLIC_SHA>`；
- 本地公开快照 `HEAD^` 等于远端预期 SHA，且两者之间恰好一个提交；
- 三个本地资产的最终大小和 SHA-256 已记录。
- `$ReleaseTitle` 和 `$ReleaseNotes` 仍等于已冻结值，说明文件 SHA-256 未变化。

“查询失败”不等于“不存在”。认证、网络、限流或 API 错误必须单独处理，不能据此继续发布。

## 11. REFS_ALIGNED：原子同步 `main` 与 tag

公开快照已经以已验证的远端 `main` 为唯一父提交。只允许普通快进，并把 `main` 与新 tag 放进同一个原子 push：

```powershell
git -C $PublicSnapshotDirectory push --dry-run --atomic `
  $Repository `
  HEAD:refs/heads/main `
  "HEAD:refs/tags/$Version"

git -C $PublicSnapshotDirectory push --atomic `
  $Repository `
  HEAD:refs/heads/main `
  "HEAD:refs/tags/$Version"
```

要求：

- dry-run 必须先通过；正式 push 前再次读取远端 `main` 和目标 tag；
- 禁止任何 `--force*`、`+` refspec、`--mirror` 或隐式 tag push；远端漂移时让普通 non-fast-forward 检查直接拒绝；
- 不拆成两个独立 push；
- 若远端不支持 `--atomic`，命令必须失败，不得自动降级为部分写入；
- push 返回非零或连接中断后，先重新查询远端引用，再决定是否重试；不得根据“看起来已上传”猜测结果；
- 成功后立即确认远端 `main` 与新 tag 都等于 `<PUBLIC_SHA>`。

旧版本 tag 和 Release 不移动、不删除。线性规则启用后创建的新 tag 必须指向 `main` 上对应的公开提交；迁移前已存在且不在 `main` 祖先链上的不可变 tag/Release 作为历史例外保留，不得为对齐新规则而重写。

## 12. DRAFT_VERIFIED：草稿上传和远端资产核对

引用对齐后创建草稿 Release：

```powershell
gh release create $Version `
  --repo 'Kirinigh/Gakumas_Helper' `
  --verify-tag `
  --title $ReleaseTitle `
  --notes-file $ReleaseNotes `
  --draft `
  --prerelease

gh release upload $Version `
  $PackageZip `
  $ReleaseManifest `
  $Checksums `
  --repo 'Kirinigh/Gakumas_Helper'
```

上传完成后查询 Release API/CLI 并逐项比较：

- `isDraft=true`、`isPrerelease=true`；
- tag 名称和目标 SHA 正确；
- 恰好三个资产；
- 文件名、大小、`uploaded` 状态和 GitHub digest 与本地一致；
- 远端标题和说明与冻结文本一致，且本地说明文件 SHA-256 未变化；
- Release Notes 中的版本、源码 SHA 和资产 SHA-256 与最终文件一致；
- 远端 `main == tag == PUBLIC_SHA` 仍成立。

任何一项不一致时保持 draft，不得公开。

## 13. PUBLISHED：发布和公开复核

```powershell
gh release edit $Version `
  --repo 'Kirinigh/Gakumas_Helper' `
  --draft=false `
  --prerelease=true
```

发布后同时用认证查询和未认证公开 API 复核：

- Release 为非 draft、prerelease，且 `published_at` 非空；
- 远端 `main`、tag、Release tag 目标和两个 manifest 的源码 SHA 一致；
- GitHub 首页 `README.md` 显示本次版本；
- `main` 和 tag 下的 `README.md`、`pyproject.toml`、`assets/interface.json` 均为本次版本；
- 三个资产仍为 `uploaded`，大小和 digest 不变；
- 旧 tag 和旧 Release 仍可用于公开版本回滚；本机旧安装目录的保留与核对属于独立安装批次，不作为本次公开发布完成门。

只有这些检查全部通过后才能宣布发布完成。

## 14. 失败与回滚边界

| 失败位置 | 必须采取的动作 |
| --- | --- |
| 源码、快照、候选或资产验证失败 | 停留在本地隔离目录；不得写远端，不得切换安装 |
| 远端预检发现 SHA 漂移、tag/Release 已存在 | 停止并重新审计，不覆盖现有状态 |
| 原子 push 失败或连接中断 | 查询远端两个引用；在状态明确前不得重试或进入 Release |
| 引用已对齐但创建/上传草稿失败 | 不公开；保留现场并选择继续草稿或经单独授权回退引用，不自动删除/强推 |
| 草稿资产大小或 digest 不匹配 | 保持 draft；重新构建并重新验证，不用发布说明掩盖差异 |
| 发布后发现问题 | 不移动已发布 tag，不覆盖或替换同名资产，不复用版本号；发布新的修正版 |

任何删除 draft、删除未发布 tag、回退远端 `main` 或切换本机安装都属于新的修改动作，必须重新核对精确目标和授权。

## 15. 当前已知自动化缺口

执行下一批次前必须逐项检查；未关闭时不得依赖默认值：

1. `tools/deployment/public/ASSET_PROVENANCE.md` 当前固定记录一个上游版本、提交和 ZIP 散列；升级上游时必须同步并复核。
2. `tools/deployment/public/RELEASE_NOTES.md` 默认描述同基础版本人工安装；基础版本升高时必须根据构建清单实际 `update_contract` 生成或重写说明。
3. `build_release_asset.py` 当前只允许公开预览资格；稳定版必须先扩展代码、测试和验收门。
4. 当前没有统一的 GitHub 发布编排器；`main`、tag、draft、资产核对和最终发布仍须按本文逐项人工执行。
5. 自动更新下载、兼容失败、校验失败、候选切换和完整回滚尚未全部端到端闭环；不得把较高 SemVer 优先级写成“已经验证自动升级”。

这些缺口必须在开发源码或模板中解决并重新生成产物；不得直接手工修改已经验证的候选、ZIP 或 manifest。

## 16. 公开历史不变量

- 初次建库必须显式使用 `--initial-public-root`；已有公开 `main` 后禁止再次生成根提交。
- 后续每版只以上一版实时核验的公开 `main` 为唯一父提交，并只增加一个脱敏提交。开发仓库、开发 worktree、浅历史、merge、replace、graft、alternate、额外 ref 和任意历史隐私门失败都必须拒绝。
- 完整公开链中的版本必须唯一；`MAJOR.MINOR.PATCH` 不得回退，发布日期不得回退，同一核心版本下的 `gkh.YYMMDD` 必须严格递增。同日提升核心版本可以保留，重复版本或倒序版本失败关闭。
- 构建器逐提交读取原始 Git blob 验证完整公开链；公开历史不得依赖导出属性隐藏内容。
- 推送只使用普通 fast-forward。远端 SHA 漂移时停止并重新生成，不通过 force、临时 merge 或父提交替换补救。
- 生成的新公开快照本身就是下一版可复用的父仓库；应保留到下一版成功发布并完成远端复核。

## 17. 维护规则

- 本文只在发布契约、脚本接口或失败语义改变时更新。
- 某次发布的版本、SHA、资产大小、扫描结果和进度写入该版本的构建清单、Release manifest 和发布记录，不写入本文。
- 修改本文时必须同时核对三个发布脚本的 `--help`、相关测试、SemVer 规则和 GitHub Release 行为。
- 文档与脚本冲突时停止发布；先修正并验证冲突，不能任选一方继续。
