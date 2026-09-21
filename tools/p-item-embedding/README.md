# P-item embedding toolchain

## 2026-09-20 增量更新

固定目录 `41153f09d3ca9e88dbbefd9c533f51b48dcaaa3c`，保留旧 434 张参考原字节，追加 488／489「止まない嵐」普通／强化态，图库共 436 项。488 来自公开育成录像的初始状态预览（31:13），489 来自 Game8 完整强化态 UI；逐图原始尺寸、裁框和来源见 `evidence/p_item_production_source_published_ui_20260920.json`。488 原截图中的图标约 51×52 像素，缩放到 130×130 不增加原始细节，真实竞技场校准仍待完成。

分类索引更新为 489 项、153 对普通／＋，旧 486 项标签不变；新增 487 为共通／不可继承支援卡，488／489 为非凡／P 偶像。计划计数 50／154／171／114，种类计数 309／50／82／48，53 项没有参考图。复建使用 `.local/task095-batch11-20260920/` 的 `p_items.json`、`ProduceItem.yaml`、`master-receipt.json`、`pitem/evidence-a.json` 和 `pitem/ready-a`，原固定数据集不变，revision 为 `task095-p-item-classification-489-v1`。

以下 2026-09-13 部分保留原批次来源和复建命令。

## 全量 P 道具分类索引（2026-09-13）

`assets/data/p_item_classification.json` 按业务 ID 保存全目录 486 条源数据标签，并连接现有固定参考图库的 434 个 ID。52 条没有参考图的目录记录仍完整保留标签，`reference_available=false`；这不表示它们已有可用于识别的图像。竞技场按舞台计划和四槽来源消费分类索引；索引文件保留生成时的离线状态字段，不代表实机验收。普通／＋完整图匹配方案保持不变。

分类维度：

- `plan`：共通 `free`、感性 `sense`、逻辑 `logic`、非凡 `anomaly`，分别 49、154、171、112 条。
- `kind`：P 偶像固有 `p_idol` 307、可继承支援卡 `support_inheritable` 50、不可继承支援卡 `support_non_inheritable` 81、其他 `other` 48。P 偶像和其他按目录来源区分；支援卡使用官方 `isExamEffect`，并核对目录 `mode`。其他来源即使 `mode=stage` 也保持其他类别。新未知来源、模式和未登记冲突均报错，不静默归入其他。

477 条通过官方名称和普通／＋状态唯一映射；其中 8 条名称差异同时由 既有已核对的官方图像 crosswalk 确认。431～439 是目录中的特殊袋类记录，只按明确目录来源标注，官方映射为空并列在 `official_unmapped_business_ids`，不猜测对应的官方袋子。所有记录保留目录计划、模式、来源、偶像 ID、稀有度、福利标记、官方字段及标签依据。

115「ハンターの戦利品」按官方主表标为共通，142「優しさミルクシュガー」标为逻辑；已有视觉 crosswalk 确认其图像身份，原目录的逻辑／共通原值保留在 `source.catalog_plan`，不改运行目录。406～408 保留已登记的感性／逻辑／非凡业务拆分，官方共通字段另行保留。索引包含 152 对普通／＋关系，并校验双向配对和标签一致性。

构建器 `build_p_item_classification.py` 核对既有目录证据、官方主表接收记录、视觉 crosswalk 及实际 NPZ，构建结果使用明确 revision。`select_ids(index, plan=["free", "logic"], kind="support_inheritable")` 支持同维度并集、跨维度交集；共通需显式选择，空选返回空集，未知标签报错。

在项目根目录用项目虚拟环境 Python 复建：

```powershell
python tools/p-item-embedding/build_p_item_classification.py `
  --catalog <冻结输入目录>/p_items.json `
  --master <冻结输入目录>/ProduceItem.yaml `
  --evidence <冻结输入目录>/pitem-followup/evidence-a.json `
  --receipt <冻结输入目录>/master-receipt.json `
  --dataset <固定参考数据集目录>/dataset_manifest.json `
  --gallery-root assets/resource/base/model/embedding/p_item_reference `
  --revision task095-p-item-classification-486-v1 `
  --output assets/data/p_item_classification.json
```

源文件冻结于本机 `.local`；独立检出需要同一批输入。目录 revision 为 `2b1621ea0ef458de6b363143d6dab90b5335c812`，官方主表镜像 revision 为 `8f3f325edfc18a9db53587f94ae9edea7ad66f86`。本轮不刷新上游、不追加图片、不重训或部署。未来更新应使用新批次冻结来源重新构建，并处理新增来源/官方映射及计划差异，不能只增加计数。

本目录保留 P 道具的历史 64×64 RGB → 128 维 L2 嵌入、Top-5 余弦近邻和普通/+状态解析工具，并提供当前生产固定渲染参考图库的构建、晋级与验证工具。两类制品用途独立，均不复用技能卡图库。

## 2026-09-05 适用范围更正

当前生产按完整普通／`+` 固定参考分别匹配业务 ID，保持 Top-24、粗排前 12 守门、7 尺寸精排与完整图库回退；406／407／408 的已登记同图组仍按可靠计划类型解析。用户本次决定沿用该方案，不迁移为“主体视觉组固定匹配＋独立强化解析”。嵌入及其同组独立 `+` 解析只用于历史训练和离线诊断，不进入生产候选、身份或回退决策。下列历史实验和制品记录保留，本次更正不修改运行时、资产、UI、默认值或点击路径，也不追溯改写过去的用户决定。

## 数据集

基础训练数据集固定到 `surisuririsu/gakumas-tools@3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6`，官方图像固定到 DMM PC 清单 `pc-705100-0036`。该历史训练范围为 `sourceType != produce` 的 420 个业务 ID；后续固定渲染图库扩展不改写这份训练事实。

```powershell
.\.local\task075-card-embedding\.venv\Scripts\python.exe tools\p-item-embedding\build_p_item_dataset.py `
  --catalog-csv .local\task085-upstream\p_items.csv `
  --catalog-tree .local\task085-upstream\gakumas-tools-tree.json `
  --official-manifest .local\task075-upstream\manifest-pc.json `
  --official-manifest-label pc-705100-0036 `
  --object-manager-root .local\licensed\GkmasObjectManager\7da94255b10e367bf83fadb805a0c5f643998de9\GkmasObjectManager-7da94255b10e367bf83fadb805a0c5f643998de9 `
  --output-dir .local\task085-p-item-dataset\pc-705100-0036
```

基础数据集构建器会核对 420 个业务 ID、507 个官方 P 道具资产、可解码性、前景遮罩映射、普通/+配对、跨 ID 重复及 406/407/408 官方同图组。固定渲染图标只在完成官方原图映射后用于渲染域适配，不独立决定业务 ID 或图案身份。

## 训练与导出

PyTorch 仅安装在隔离训练环境，主项目运行环境不增加该依赖。历史选定候选延续分阶段检查点到 600 epoch，并为 16/24/32/48 px 增加结构化图库原型；该离线候选输出嵌入和 Top-5，不使用封闭式分类头，不属于当前生产接口。

下面命令用于从当前代码建立新的实验候选；选定 v2 是分阶段续训结果，其每段学习率与训练域记录在最终 `manifest.json`，不能把单次固定学习率命令误写成该成品的逐字节复现步骤。

```powershell
.\.local\task075-card-embedding\.venv\Scripts\python.exe tools\p-item-embedding\train_p_item_embedding.py `
  --dataset-manifest .local\task085-p-item-dataset\pc-705100-0036\dataset_manifest.json `
  --output-dir .local\task085-p-item-embedding\candidate `
  --epochs 30 --batch-size 128 --learning-rate 3e-4 --device cuda
```

精确训练阶段、历史损失、模型/图库/类表散列记录在最终 `manifest.json`。原始官方图像、数据集、检查点、训练中间产物和实机截图都留在 `.local/`，不得提交或分发。生产固定渲染参考图库是单独登记的发行制品，不等同于训练数据；当前图库为 426 项。识别制品自身的晋级与完整 Release 打包会按业务 ID 集合验证竞技场 `stage` P 道具均被历史兼容图库覆盖；RIS 独立数据更新不因图库暂时滞后而被阻断，只记录缺口并交给运行时详情语义安全消歧，无法唯一确认时停止。旧图库分数尚无已证明的开放集拒识能力，不能仅凭“高置信”跳过疑似新实体的详情证明。其来源、散列、用途及游戏图形权利保留边界见 `ASSET_PROVENANCE.md` 与 `THIRD_PARTY_NOTICES/`。

ID 475/476 的目录数据最早已出现在固定的 `gakumas-tools` revision，但当时 `gakumas-images` 只包含至 ID 474，因此曾由 `generate_rendered_reference_extension.py` 从固定官方 DMM 图案、177 个既有背景样本及 473/474 的升级标记规则确定性合成，并作为临时引用要求具体槽位详情确认；该脚本保留为历史制品复现工具，不再是当前活动图库来源。上游随后在不可变提交中补齐 475/476 的 130×130 固定渲染 PNG，并由成功的 production deployment 交付。2026-08-29 图库以这两张上游 PNG 替换旧合成项，当时业务 ID 总数为 426，`provisional_business_ids` 为空；`mode=produce` 的 ID 477 继续由 EntityBank 的 `mode=stage` 契约排除。精确部署、Git blob、文件／像素散列和旧图库继承关系写入 manifest；真实 JJC 样本仍作为后置实机校准项收集，但不再构成临时引用或强制详情点击的依据。原始图和 426 张源图继续只留在 `.local/`，发行组件只包含去元数据后的 NPZ、manifest 与公开安全的 `production_handoff_evaluation.json` 三件套。

## 固定渲染参考的构建、晋级与发布

`build_rendered_reference_gallery.py` 只生成规范 LF JSON 的候选目录：`p_item_rendered_reference_gallery.npz`、`manifest.json`，并写入固定 `build.tool_contract` 与 `production_handoff=PENDING_VALIDATION`。候选不得直接进入生产路径，也不得人工改写为 READY。

`promote_static_reference_handoff.py --component p_item_reference` 必须同时取得候选目录、不可变旧图库、完整 builder 源图目录、权威 `p_items.json`，以及通过 `--p-item-production-source-evidence` 指定的版本化冻结证据。该历史替换批次的证据为 `evidence/p_item_production_source_d476.json`；其规范 LF 字节 SHA-256 在晋级器中固定，独立绑定 `gakumas-tools` revision、production deployment、目录字节／Git blob、`gk-img` gitlink、475/476 路径／文件／像素和旧图库。候选 manifest 的来源投影必须与该证据精确相等，不能用候选自己的 provenance 自证；随后工具再对实际旧图库、426 张源图和目录文件逐字节重算，并验证历史继承、NPZ 容器、唯一业务 ID、PNG 解码／元数据清理及完整图库排名，最后原子生成 NPZ、manifest、`production_handoff_evaluation.json` 三件套。当前正式状态为 `READY_WITH_REAL_SAMPLES_PENDING`；这只表示离线交接门通过，真实 JJC 普通／`+` 校准仍为 `PENDING`。

首个新增 ID 批次起，先运行 `build_append_handoff_evidence.py`，从不可变旧 NPZ、完整数字命名的固定 PNG 源目录、权威 `p_items.json`、revision／deployment 与明确的 added／replaced 集合，确定性生成 evidence schema-v2 和公开 provenance schema-v3；它会核对有序 ID 增量、未改行原 PNG 字节继承、替换像素、竞技场目录覆盖、逐张来源散列及完整图库排名。两份输出仍须由独立审计确认后，把 evidence 的规范 SHA-256 加入晋级器允许清单；生成器输出本身不能自证可信。随后将 provenance 交给图库 builder，再由 promoter 绑定同一 evidence 原子晋级。2026-09-05 已从 Game8 普通图和人鳥日記强化介绍图补齐478～483完整两态参考；公开 UI 来源走 evidence schema-v3／provenance-v4，保留同一生产匹配算法。

公开 UI 裁图使用 `build_append_handoff_evidence.py --published-ui-manifest <来源清单> --published-ui-original-root <冻结原图目录>`；该分支仍要求真实目录 revision／成功 deployment，但不得传图像的 `gk-img`、first-added 或首次图片部署字段。来源清单逐图绑定 ID、名称、强化状态、发布页／原图 URL、冻结日期、原图字节／尺寸、半开裁框、Pillow BICUBIC 130×130 缩放与无元数据 PNG。构建器与晋级器均实际重放裁图，晋级器另传 `--p-item-original-root`。本批独立证据为 `evidence/p_item_production_source_published_ui_20260905.json`；图库由426项原字节继承加6项形成432项，不改变普通／`+` 匹配方案，真实 JJC 保持 `PENDING`。游戏图形权利不继承目录代码的 BSD 许可证。

用户直接提供的游戏 UI 截图使用同一裁图重放入口，但必须显式声明 `source_type=user_provided_ui_crops`，逐图 `capture_provenance=USER_PROVIDED_GAME_UI`、`publisher=USER_PROVIDED`，两个 URL 字段必须为 null；不得伪造公开发布页。该来源生成 evidence schema-v5／provenance-v6，使用独立 `user_provided_ui_crops` 字段和 revision 后缀，旧公开来源不能混入或改标。仍须冻结原图、绑定目录身份、核对全部原字节继承、注册经核验的证据散列、通过正式晋级；缺原图或裁图不一致继续拒绝。原截图只留本地，真实小槽与双窗资格不能从提供图片或旧图库直接继承。2026-09-10 的 485／486 两态图走此来源，434 项图库的旧 432 项原字节保留，新增真实 JJC 校准仍为 PENDING。

聚合报告将可从发行组件重算的 `validation` 与晋级期 `promotion_evidence` 分开保存，并用 `candidate_manifest_sha256` 绑定晋级前候选。Release 会从发布工具源码树实际重新加载同一份固定散列证据，先做 manifest 来源投影比对，再从包内 NPZ／manifest 重算 validation、核对报告并以包内 RIS 的 `p_items.json` 检查竞技场 `stage` 集合覆盖。冻结证据随公开源码保存但不进入用户运行包；未分发的 426 张 builder 源图也不会被冒充为发布时已重新读取。

## 离线评估

以下命令评估的是独立 embedding 模型，不是固定渲染参考的 `production_handoff_evaluation.json`，两者不得互相代替。

```powershell
.\.venv\Scripts\python.exe tools\p-item-embedding\evaluate_p_item_embedding.py `
  --run-dir .local\task085-p-item-embedding\authoritative-705100-0036-v2 `
  --dataset-manifest .local\task085-p-item-dataset\pc-705100-0036\dataset_manifest.json `
  --output .local\task085-p-item-embedding\authoritative-705100-0036-v2\offline_evaluation.json
```

评估会分开报告图案检索、普通/+状态、406/407/408 计划解析、合成未知拒识和延迟。这些嵌入阈值仅为离线诊断，不写入当前固定参考生产运行清单，也不替代其接受阈值与回退门。
