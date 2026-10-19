# 公开制品来源与权利边界

本表用于区分可公开构建输入、最终运行制品和禁止进入公开包的本机材料。它记录技术来源和项目发布边界，不构成对第三方知识产权的额外授权或法律保证。

来源术语更正（2026-09-05）：游戏官方清单／原画与第三方整理的完整 UI 图分别记录。`gakumas-tools`／`gakumas-images` 的成功部署证明该第三方版本已部署，不证明其中图标由游戏官方发布；历史 P 道具来源字段 `official_rendered_source` 保留兼容含义，不能据其名称扩大来源或授权。新批次公开 UI 裁图按实际发布者和原图记录，不继承目录代码的 BSD-3-Clause 许可证；游戏图形权利仍由原权利人保留。

| 制品 | 固定来源／生成方式 | 公开内容 | 禁止公开内容 | 权利／通知 |
| --- | --- | --- | --- | --- |
| MaaFramework 与 Python 配套运行时 | `MaaXYZ/MaaFramework` 官方 `v5.13.0`（提交 `2bcfa85c66a2eac6ca3e5937f175495275ee0643`）Windows x86_64 ZIP，SHA-256 `179B51B6211FC74317C8DC06F10FAE4724ACA370DF7D1BF694CD00D06FF38EFE`；PyPI 官方 `maafw-5.13.0-py3-none-win_amd64.whl`，SHA-256 `105C22431183446D350864B946FB9448F47FD9D73A20E4B46E276EA2A22067E5`。构建同步更新宿主已有框架库与 Python 包，核验双方共有库和实际版本；MFA 应用与下述 Core 补丁保持各自固定来源 | 官方框架运行库、已有设备工具的对应更新、Python 绑定及许可证；精确覆盖范围写入构建和发布清单 | SDK 调试符号、示例工程、下载缓存、私有配置或临时通信文件 | MaaFramework 按 LGPL-3.0 提供，保留官方 `LICENSE.md`；Python 包保留其许可证目录，其他组件按各自通知提供 |
| MaaGakumasu 应用、Agent、任务资源和基础模板 | `SuperWaterGod/MaaGakumasu` 的官方 `v1.4.9`（提交 `f520438c168e3d911b5775cc9b24297c19182e2e`）Windows x86_64 Release；ZIP SHA-256 `AA90F2BC6CE47F8B78B6CABD5937F96A19D2024ECC253C8BE13DE83680275D2F`，构建时重新核对后替换本项目已提交树 | 本项目清洁修改及固定上游二进制包 | 上游账号、CI 密钥、用户配置与运行日志 | 根目录 `LICENSE`（AGPL-3.0）；不表示项目获得游戏商标或图形所有权 |
| 竞技场独立卡位定位 v1 | 复用 MaaGakumasu v1.4.9 / f520438c168e3d911b5775cc9b24297c19182e2e 已固定权重，Git blob c443c4e14c96309a296e4b269484b34d6c0fcf51；原始 SHA-256 44F0AF9247DF55B423CEC41999BAA8A655E67E86F590F28F0E852C1FEF62F0F7，10,606,101字节。原字节保存为 resource/base/model/detect/arena_cards_v1.onnx，由 ArenaRecognitionCards 独立加载；上游 ProduceRecognitionCards / cards.onnx 继续随培育同步，不替换此专用版本。升级专用版本须先通过竞技场冻结原图的 CPU / DirectML 定位、双排坐标及既有三帧/内容回归 | 已固定 ONNX 权重、独立识别节点与来源说明 | 真实竞技场截图、用户日志、训练数据、检查点及未验证的移动上游权重 | 沿用上游 AGPL-3.0 / 根 LICENSE；没有重新训练，不声称训练集和训练过程可复现 |
| MFA Core 更新、日志导出与期数选项补丁 | `MaaXYZ/MFAAvalonia` 官方 `v2.15.2` 提交 `6065fe33798b72906c5079fa6f210646801d9a5c`；源码 ZIP SHA-256 `76DD02AFE4B1529B1D4F6416B3442E1BD7E64F13B72D8A3B428F955B5E26F67A`，原始 `VersionChecker.cs` Git blob `6e6d1118fa414ba21c7efa4f15a58ad95dd08bd7`。仓库补丁 `tools/deployment/mfa/MFAAvalonia-v2.15.2-resource-update-github-fallback.patch` 在资源检查与资源更新中令 RID 为空时回落 `interface.json.github`，并对下载失败返回值与异常执行有界重试、只在下载成功后继续更新；任务初始化不再清理共享的 `temp_res`，下载失败时仅删除本次已创建的半包，避免并行任务互相删除。同一补丁还补齐日志导出中的后台记录与失败证据，并让竞技场期数列表读取更新后的目录；兼容旧选择并移除重复的“最新”显示项。构建期另将 Sentry 源码根稳定映射到 `/_/MFAAvalonia/`，不把本机目录写入 DLL。构建方式见 `tools/deployment/mfa/README.md`，固定 .NET SDK 版本组件为 `10 / 0 / 400`，ZIP SHA-512 `9B8B88590E4DA131BFD0DA7AA089D0FC04D5418D5F8607EC13D55DC5A17B4399AFD54D496C12657FA05C6C6546DC5EAB930F26AC6C50F2D3A7712C0FB378C366` | 发布包只覆盖补丁后重建的 MFA Core DLL；对应源码由固定官方源码、仓库内补丁和构建说明共同提供 | 未修改的完整 MFA 应用包、构建缓存、用户配置、日志或本机路径 | MFAAvalonia 与该补丁对应源码按 GPL-3.0 提供；发布必须保留上游版权和许可证通知 |
| 技能卡 embedding 基线 | 875／876批次固定官方PC清单`705100:0054`与`gakumas-tools@0d0a85145258f700dbbab66d31acf87361297a37`，追加两张官方原画对应向量；模型不变，旧4174行原字节继承至4176行，覆盖876个业务ID、450个视觉组。完整UI分别来自[Game8 875](https://game8.jp/gakuen-idolmaster/815155)、[Game8 876](https://game8.jp/gakuen-idolmaster/815156)，仅用于既有诊断与强化标记流程 | ONNX 参数、128维数值向量、小型强化标记参考张量、类表及生成／评估／晋级脚本 | 原始卡图、提取图标、数据集manifest、检查点、增强图、实机截图、私有样本指纹 | 项目代码与自产参数按根许可证发布；目录数据保留BSD-3-Clause通知；第三方公开UI图不继承该许可证，游戏图形权利仍由原权利人保留 |
| 竞技场技能卡 embedding | 使用前述875／876的两张官方原画及公开UI参考，沿用原竞技场模型和阈值；旧4174行原字节继承至4176行，与基础图库同为876个业务ID、450个视觉组。具体离线结果与待验范围保留在manifest及评估中 | ONNX参数、数值向量、小型强化标记参考张量、类表、确定性增量工具 | 真实用户截图、ROI、私有正样本字节及其可链接指纹 | 与技能卡基线相同；新卡真实槽位校准仍为PENDING，离线通过不等于实机通过 |
| 竞技场卡面粗参考 | 继承原3407行及既有529修正，按875／876的Game8完整UI来源重放裁剪与缩放，追加两行至3409行；原五个数组按reference_name逐项保留，绑定本批竞技场manifest，粗参考仍为8×16 BGR及掩膜 | 固定数值／掩膜NPZ、公开安全聚合评估、manifest及生成／评估脚本 | 原始卡图、实机截图、ROI、私有manifest、路径／槽位、私有样本字节及其可链接指纹 | 原目录与代码通知按实际来源保留；公开UI发布者和原图记录不构成额外再许可，游戏图形权利仍由原权利人保留 |
| 竞技场卡面费用参考 | 本项目脚本从冻结 `gakumas-tools@3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6` 目录、历史徽标图库和固定图标源确定性重建；manifest 的 `source.catalog_file_set_sha256` 只绑定构建期 `skill_cards.json + customizations.json` 字节，`source_set_sha256` 绑定完整构建输入。晋级时再运行固定 descriptor 回放；发布时由 engine manifest 分别绑定当前包内两份 RIS 文件字节，并用 `ArenaEntityCatalog` 对完整 generic ID、descriptor、hypotheses 与费用 NPZ 做双向语义精确比对，不要求当前 RIS 组合散列等于历史 source hash | `card_cost_reference.npz`、manifest、公开安全的 `production_handoff_evaluation.json` 三件套，以及确定性 builder／promoter／Release 复验脚本和去标识 descriptor fixture | 原始卡图、实机截图、ROI、本机路径、私有逐样本证据及其可链接指纹 | `gakumas-tools` 部分保留 BSD-3-Clause 通知；数值／掩膜制品和去标识 descriptor 不表示获得原始游戏图形的再许可，相关权利仍由原权利人保留 |
| P 道具参考 | 完整普通／`+`参考分别匹配业务ID。2026-09-20 保留434项原PNG字节，追加488／489到436项。488来自[ミーク公开录像31:13](https://www.youtube.com/watch?v=dcF5m8NRrWk&t=1873s)的初始状态预览，489来自[Game8](https://game8.jp/gakuen-idolmaster/816967)强化态图。目录固定`gakumas-tools@41153f09d3ca9e88dbbefd9c533f51b48dcaaa3c`及成功production 6540432420；第三方公开UI来源分别冻结，不冒充官方直出。独立证据`tools/p-item-embedding/evidence/p_item_production_source_published_ui_20260920.json`规范SHA-256为`A79D20EDA9C57B1527E5200B285BD80B4BF84C7CC72E618B32CDDEE3F466C562`，构建与晋级实际重放裁图及旧图库继承 | 436项去元数据图标NPZ、manifest、公开安全production_handoff_evaluation.json及构建／晋级／Release复验脚本。READY_WITH_REAL_SAMPLES_PENDING，新两项真实JJC为PENDING；488源图标约51×52像素，130×130转换不代表原始高清 | 原始资源包、数据集、实机截图、公开面板原图及436张builder源图；487为培育专用支援道具，仅分类、不纳入竞技场图库 | 目录代码／数据保留BSD-3-Clause通知；第三方图片不继承该许可证，游戏图形权利由原权利人保留 |
| P 道具详情文字 | 固定 `vertesan/gakumasu-diff@5b8969e363ae60cd1d9bf64560783b4b445f7da6` 六份主数据的日文分段与 Audition 替换规则投影，覆盖355项；逐项绑定 `gakumas-tools@6c3d00648c32ea72a05086a18c63ab62e0d0c7e5` 的活动身份与效果。六份来源文件名称、Git blob和SHA-256保留在文字资产的source字段 | 游戏条目文字与规则投影 `assets/data/p_item_detail_text.json`，用于已经打开详情后的文字比较 | 第三方解码器或工具实现、完整源归档、实机截图、玩家数据和运行记录 | 来源仓库未提供单独许可声明；原游戏内容权利归原权利人，不纳入本项目AGPL授权，也不因绑定RIS目录而继承目录代码的许可证 |
| 竞技场模拟引擎 | 固定 `surisuririsu/gakumas-tools` 成功 production revision 与源码归档；engine／data／许可证负载由归档重建，`node.exe`、Node.js 许可证、runner、scoring、calibration-cache 共5项由已验证宿主基线复用或生成。本次目录为68项RIS来源加5项宿主来源。晋级器核对源码 tree、canonical manifest、成功 production deployment、严格首父回滚 deployment、runner 协议和实际期数／舞台目录后才生成正式 sibling 输出 | Release 包内固定 JavaScript、数据、Node.js 运行时及完整许可证；精确版本、来源分区与逐文件散列写入 Release manifest 和包内 `assets/arena-winrate/manifest.json` | 任意移动 `latest`、无清单额外文件、只手写 `READY` 或无法由固定归档复算的引擎目录 | BSD-3-Clause 与 Node.js 许可证随二进制包保留 |

本批RIS目录固定为`0d0a85145258f700dbbab66d31acf87361297a37`，与新增卡牌身份配套。既有P道具和费用参考保留各自历史来源绑定，由发布检查验证与当前运行目录的覆盖和语义一致性；目录包含条目不表示所有新卡效果均已完成实机验证。

发布硬门：公开源码按允许清单生成独立、连续的公开提交；完整包只从已验证的公开提交、固定上游 Release、固定引擎和固定 Python 依赖构建。三套技能卡识别集合必须与包内 RIS 技能目录逐项相等，P 道具图库必须完整覆盖竞技场目录。卡面实机增量必须携带 NPZ、manifest、`hard_negative_evaluation.json` 三件套并重放累积硬负门。费用与 P 道具静态参考必须各自携带 NPZ、manifest、`production_handoff_evaluation.json` 三件套并通过 Release 重算；费用还必须以 engine manifest 绑定当前包内 `skill_cards.json + customizations.json` 字节，并通过 `ArenaEntityCatalog` 语义双向精确门；P 道具还必须从发布源码树实际加载固定 SHA-256 的独立生产来源证据，并将 manifest 的来源投影与其精确比对。引擎还必须核对 canonical source、固定归档、68+4 来源分区、成功 production deployment、首父回滚、runner 协议和实际舞台目录。任何用户配置、日志、缓存、截图、阵容、对局记录、凭据、本机路径、私有训练语料或无许可证参考项目内容均失败关闭。

2026-09-20 技能卡批次11：固定目录`41153f09d3ca9e88dbbefd9c533f51b48dcaaa3c`，官方PC历史清单705100:0062；追加877～880，基础／竞技场各4180行、880个ID，粗参考3413行。旧4176／3409行及两编码器不变。877完整UI来自玩家Exl_2002公开截图，878／879来自Game8；880经用户明确授权暂时共用879图片，manifest的`provisional_shared_from`保留此事实，并非独立强化图。新卡面资产要求配套`task095-provisional-shared-identity-v1`读取器，将879／880作为两个候选交给既有详情兜底；不得仅投放资产给旧读取器。原始图片继续排除于公开包。本轮未公开发布或完成实机验证。

### Arena complete-UI reference augmentation (2026-09-21)

The arena-only gallery includes 4176 complete-UI reference vectors from the five
previously frozen dataset manifests. Their manifest hashes are
recorded in the gallery build metadata; every source image hash was checked by the
exporter. No screenshot from the evaluation transactions is used as a reference.
No raw source image or new encoder is distributed. Existing source rights and
redistribution qualifications remain unchanged. User-authorized enablement precedes
pending real-negative, full-flow latency and opponent acceptance; it must
not be described as successful real-device acceptance. The base gallery is unchanged.

### Cost reference extension for catalog 2c575a46 (2026-09-25)

The 54th cost entry (880, green cost 3 with legal values 3/1) derives its base
mask and symbol descriptor directly from the previously promoted cost-3 template.
It is not an independent screenshot of card 880. All 53 previous base rows and
all 12 templates remain unchanged. The borrowed normal-state identity image,
which displays cost 4, is excluded from this cost build. Provenance binds the
previous three-file handoff and the new fixed catalog; the builder and release
validator verify the exact derivation. Independent live validation remains PENDING.

### Gallery batch 12 (2026-10-01)

The frozen catalog is `surisuririsu/gakumas-tools@9a305390a172cf927f168e63cafa8149b24b9448`
(successful production deployment 6763530892). Skill IDs 881–886 and P-item IDs
491–494 are appended. The decoded Japanese master mirror is still at its previous
revision. New identity records therefore explicitly use a joint source: catalog
labels, public internal IDs from `pinisok/GakumasTranslationDataKor@6f1282f6739dc704047236135363119b9ce8f119`,
official PC manifest 705100:0067 and actual complete UI references. These labels
are not represented as decoded official master fields or imported translations.

Skills 882/884/886 and items 492/494 use Game8 images; skill 885 uses the public
kirin_mets_Xeno post 2105128331121471932. Normal skills 881/883 and items 491/493
use user-authorized local Maa captures with Null input. Full screenshots and their
fingerprints remain local; release provenance binds only the isolated card/item
source crops. The native 64×64 item crops are resized to 130×130 without claiming
additional detail. Public images retain UNKNOWN capture provenance, and local
sources retain USER_AUTHORIZED_LOCAL_MAA. Neither becomes independent live
acceptance evidence. No original screenshots or source crops are distributed.

The base gallery contains 4,186 references; the arena gallery contains 8,362,
including the unchanged 4,176 previously enabled full-UI references. Badge
references total 3,419. All previous numerical references and both encoders are
preserved. P-item references increase from 436 to 440 with all prior PNG bytes
preserved. The P-item independent source evidence is
`tools/p-item-embedding/evidence/p_item_production_source_mixed_ui_20261001.json`,
SHA-256 `5252BF4596C18038F77FEBDE20E1E42416CD150F88D11189DA566DD03A35638D`.
Classification covers 886 skills and 494 items; item 495 is precollected locally
only. Source rights, release exclusions, runtime thresholds, the existing 880
provisional identity treatment and pending live/latency acceptance remain unchanged.

### Gallery batch 13 and inactive catalog placeholders (2026-10-10)

The frozen catalog is `gakumas-tools/gakumas-tools@6f138bd7eb4300935a473b3529bd82a1eb6661c6`
(successful production deployment 6961164205). P-item classification covers
499 IDs. Arena references append 496/497 to reach 442 references; all previous
440 PNG payloads, IDs and offsets remain unchanged. Item 495 and new items
498/499 are non-inheritable produce support items and have no arena reference.
The two new UI sources are the Game8 normal-state image and the actual
upgraded-state game image published at
`https://skypenguin.net/2026/10/09/post-170050/`. Capture provenance remains
UNKNOWN. The independently replayable evidence is
`tools/p-item-embedding/evidence/p_item_production_source_published_ui_20261009.json`,
SHA-256 `1A306AF34A0A3BD5B618419E2C4781063B1C38DC46B52A3B27AD566D88F9A3F2`.
The P gallery SHA-256 is
`9B37C597E8B5CCD4D4A9347461B58678B1595C3FBC38694C40FD946484AA8981`.
Source self-queries and offline replay do not establish independent live
recognition or complete-flow latency acceptance. Original screenshots and
builder crops remain excluded; third-party publication does not confer a new
image license, and game graphic rights remain with their original owners.

Skill classification and all three gallery inventories cover 908 IDs, while
only the previous 886 IDs have enabled image references. IDs 887–908 each add
one explicitly disabled neutral placeholder per component. Base reference
rows total 4,208, arena rows 8,384 and coarse card-face rows 3,441; their enabled
prefixes remain 4,186, 8,362 and 3,419 respectively. Both encoders and every
old numeric row are unchanged. The previous 4,176 arena full-UI references are
unchanged. This is catalog coverage, not recognition of 22 new cards.

The included `temporary-card.png` is locally generated neutral artwork marked
TEMPORARY / NO GAME ART, with no game image or user screenshot. Its SHA-256 is
`5F3F4AA46A01B9E23728B657467C3D03524AFE0E899654F713E7FB1B496A644C`.
Schema 2 consumers require `reference_enabled=false` for all placeholder rows
and exclude them from identity search, visual families, upgrade templates,
coarse matching and return-page reference protection. Assets and consumers
must be delivered together; old consumers reject these schema 2 assets.
887/888 retain the decoded official identity from the frozen master mirror;
889–908 retain UNKNOWN_PENDING official identity fields and preview catalog
labels, without invented asset/internal IDs or certified ordinary-state
effects. No actual card artwork or new image training is claimed for this
extension. Replacing it requires verified real sources and fresh validation.

Release validation binds each enabled prefix to a preserved schema 1 base,
retains the existing source and badge qualification gates, verifies the
classification and catalog bindings, and rejects any enabled placeholder.
The handoff stays
`USER_AUTHORIZED_TEMPORARY_PLACEHOLDERS_REAL_SAMPLES_PENDING`; no READY or
independent live PASS is manufactured. Existing rights, runtime thresholds,
necessary detail fallback and outstanding live/latency acceptance remain.

### Upstream card detector update (2026-10-10)

`resource/base/model/detect/cards.onnx` is the exact 10,605,695-byte model from
`SuperWaterGod/MaaGakumasu@3f3d37995ab612a9c725fc86cba4cacef09ed335`
(upstream v1.5.2; model introduced by commit `7fed4d0`). SHA-256 is
`5E9903521E9940296A088C9108EAE369CE622DD7DD27D05620D614DED6E1221C`.
The upstream repository and model metadata declare AGPL-3.0; the model
metadata identifies Ultralytics 8.4.66 and classes cards/recommend/useless.
Keep the upstream copyright and root AGPL license with this redistributed
model. This provenance entry does not certify packaged model loading or
synthetic inference; those remain release verification requirements. Real
game accuracy and complete produce-flow acceptance remain unverified.


### 二十项正式参考及同编号界面更新（2026-10-15／16）

本批纳入技能卡 889～908 的正式身份和真实来源参考，撤销这二十项禁用占位；三个正式组件恢复 schema 1，并保持 `READY_WITH_REAL_SAMPLES_PENDING`。基础图库 4354 行、竞技场图库 8530 行、卡面粗参考 3587 行，均覆盖并启用 908 个业务 ID；P 道具仍为 442 项参考／499 项分类。原 1～888 的分类对象、已启用数组、既有 4176 行完整界面尾集和两套编码器保持；卡面参考名称存储从 U10 扩到 U12，旧名称值及回转旧类型后的字节保持，不放宽身份或候选保护。

10 月 16 日的二十张普通／强化完整界面参考固定来自 `gakumas-tools/gakumas-tools@4d1b68d64bb825e564ad45442df25c3fe51f273b` 的 `packages/gakumas-images/images/skillCards/icons/889.png` 至 `908.png`；逐项 URL、原始大小、SHA-256、裁框和重采样方式保留在卡面 manifest。它们是第三方公开的原生 130×130 界面图，`capture_provenance=UNKNOWN`，不称为官方成品截图或独立实机证据。官方身份／原画与完整界面分别绑定；透明原画只用于其声明的向量参考，不作为完整界面替代。本次未重训或更换任何 ONNX 权重，后续同编号界面替换复用全部既有嵌入向量。

公开内容仅为现有允许范围内的数值图库、小型参考张量、类表、manifest 和公开安全汇总评估；不分发原始卡图、游戏资源包、私有训练集、玩家截图或逐样本私有回执。第三方目录代码的 BSD-3-Clause 不为游戏图形提供额外再许可，原游戏图形权利仍由原权利人保留。离线来源、继承、A/B 和固定回放门已通过；真实 DMM 独立样本、未知／截断／错误槽位拒识和完整读取节拍仍为 PENDING，不能将来源自查询视为完整竞技场实机验收。
