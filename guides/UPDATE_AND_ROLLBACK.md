# 更新与回滚

## 三条更新通道分别做什么

Gakumas Helper 的普通客户端升级只使用 MFAAvalonia 内置 GitHub 资源更新 (GitHub resource update)。项目 Release 提供完整 Windows 派生包，其中包含 Maa/MFA、Agent、资源、任务与界面、模型、图库及可离线运行的竞技场基线。竞技场另有一个严格限于 RIS engine/data 的依赖更新通道；它不会替换客户端、Agent、UI 或识别资产。

| 更新 | 负责内容 | 使用方式 |
| --- | --- | --- |
| MFA 内置 GitHub 资源更新 | 完整 Gakumas Helper 派生包 | 正常版本升级的唯一客户端入口 |
| pip 更新 | pip 本身与 `requirements.txt` 中的 Python 依赖 | 沿用 Maa 现有启动逻辑；不更新模型、图库、引擎、数据或 UI |
| RIS 竞技场组件更新 | 同一成功 production SHA 的 `gakumas-engine` + `gakumas-data` | 每个 Agent 进程首次需要竞技场模拟器时检查一次；失败保留旧组件 |
| 独立识别资产热更新 | 单独更新模型、图库或规则数据 | 当前不提供；这些资产随完整包更新 |
| MirrorChyan | 上游 Maa 的第三方分发入口 | Gakumas Helper 当前不声明派生版 RID，也不通过它分发完整包 |

> [!WARNING]
> 当前 Gakumas Helper 完整包不会声明上游 `MaaGakumasu` 的 MirrorChyan 资源 ID。MFA 资源检查在 RID 为空时回落到 `interface.json.github` 指向的本项目 GitHub Releases；如果旧安装仍显示上游 MirrorChyan 身份，请不要从该安装执行更新，改为手动安装当前完整包。

## 内置完整包更新会做什么

MFA 会按版本、通道、平台和架构选择 GitHub Release 资产，下载失败时按原生策略重试。GitHub 提供 Release digest 时会核对 SHA-256；桌面端遇到未提供 digest 的资产会警告后继续。解压后会检查基本包结构，停止正在运行的 Maa 任务和 Agent，通过文件事务应用完整包，并在应用异常时恢复本次已改文件，最后重启客户端。

因此，普通用户不需要在每次更新前再手工下载 manifest、逐组件计算散列、运行 Defender、创建候选目录或切换 `current`。这些属于发布构建或本机维护证据，不是客户端更新协议。

## 首次安装与旧命名空间迁移

Gakumas Helper 使用独立语义化版本 (Semantic Versioning, SemVer) `vMAJOR.MINOR.PATCH`。上游 Maa 标签不参与 GKH 版本编号，而是在 Release manifest 和来源记录中单独固定。项目处于 `0.x` 时，`MINOR` 表示基础大更新，`PATCH` 表示该基础上的内部迭代；项目正式完工时才把 `MAJOR` 升至 `1`。

以下情况仍需手动安装完整 ZIP：

- 机器上还没有 Gakumas Helper；
- 从旧 `vMAJOR.MINOR.PATCH+gkh.*` 命名空间首次迁移到独立 GKH 版本；这是一次性手动切换，不能依赖 MFA 比较两套命名空间；
- 内置更新入口本身无法启动，需要从已发布版本恢复。

手动安装时，从[本项目 Releases](https://github.com/Kirinigh/Gakumas_Helper/releases)下载 `MaaGakumasu-win-x86_64-vXXX.zip`，解压到独立目录，并使用版本内 `deployment\Start-MaaGakumasu-Admin.cmd` 启动。Release 同时提供 checksums 供希望人工复核下载字节的用户使用，但它不是后续日常升级的重复硬门。

当前版本以非预发布版本设为 GitHub Latest；已使用独立 GKH 版本的用户可保持 MFA 默认 Stable，通过原生入口更新完整包。旧组合版本用户仍需上述一次手工迁移。独立 GKH SemVer 不会清除仓库里的旧组合版本；不要改用 Beta 或 Alpha，否则旧组合版本可能因数值更高而被误判为“更新”。发布通道已切换，客户端实际下载、应用和重启的完整升级验收仍未完成。

## pip 更新的边界

启用 `enable_pip_update` 时，Agent 启动会检查并尝试升级 pip。启用 `enable_pip_install` 时，只有 `interface.json` 版本与 `pip_config.last_version` 不同，或旧记录为 `unknown`，才执行 `pip install -U -r requirements.txt`；成功后写回当前版本，失败则记录警告并继续启动。

不要用 pip 安装或替换竞技场模型、图库、引擎、赛季数据或界面资源。模型、图库和 UI 必须来自完整派生 Release；只有 engine/data 可由 RIS 竞技场组件通道共同更新。

## RIS 竞技场组件更新

竞技场首次需要模拟时，Agent 会查询 `surisuririsu/gakumas-tools` 最新成功的 production deployment，并比较其完整提交 SHA。版本不同才下载该不可变 revision 的 `gakumas-engine` 与 `gakumas-data`，使用当前 GKH 自带的 Node、runner、scoring 和协议构建候选；目录、实体、三舞台、引擎 DSL 与真实 runner 协议验证通过后，才原子更新本地活动指针。

- 断网、GitHub 限流、下载或验证失败不会损坏当前组件，也不会触发游戏操作。
- 显式选择的旧期数若仍由当前组件完整支持，可以警告后继续。
- 更新成功后，新期数会加入下拉列表，已选期数保持不变；请按游戏当前期数选择，并核对任务开始时的期数提示。
- 引擎 revision 更新后旧分数样本作废；同赛季己方完整编成快照可直接重模拟，不自动进游戏重读。
- 该通道不更新识别模型、图库或读卡逻辑；这类内容仍随完整 GKH Release 更新。

## 更新失败或新版有问题

- 网络、下载、解压或文件应用失败：保留当前窗口与日志，MFA 文件事务会撤销本次未完成的文件变更；不要混入手工复制的单个 Agent、模型或数据文件。
- 更新完成后发现业务缺陷：停止会消耗游戏资源的任务，到 Releases 手动安装上一已知可用完整版本；不要只回退其中一个客户端组件。
- RIS engine/data 候选失败：保留原活动组件；所选期数无法使用时停止并反馈，不会自动改用旧期数。
- 如果挑战已经开始但结果尚未完整记录：先停止继续挑战，不要用换版本绕过待决记录。
- 反馈时说明更新前后版本、前台错误和是否已开始真实挑战；不要上传完整 `config`、`.local`、日志包或未脱敏截图。

本机开发使用的版本化目录、`current` 联接和 `Install-MaaGakumasuDerived.ps1` 只服务版本目录部署、开发调试或故障维护，不是普通客户端的第二套更新器。
