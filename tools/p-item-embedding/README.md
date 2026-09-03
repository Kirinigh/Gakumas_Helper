# P-item embedding toolchain

本目录为竞技场 P 道具提供独立的 64×64 RGB → 128 维 L2 嵌入、Top-5 余弦近邻图库和普通/+状态解析。它不复用技能卡图库，也不修改技能卡识别、Live 决策、UI、默认值或点击路径。

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

PyTorch 仅安装在隔离训练环境，主项目运行环境不增加该依赖。最终候选延续分阶段检查点到 600 epoch，并为 16/24/32/48 px 增加结构化图库原型；生产接口仍然只输出嵌入和 Top-5，不使用封闭式分类头。

下面命令用于从当前代码建立新的实验候选；选定 v2 是分阶段续训结果，其每段学习率与训练域记录在最终 `manifest.json`，不能把单次固定学习率命令误写成该成品的逐字节复现步骤。

```powershell
.\.local\task075-card-embedding\.venv\Scripts\python.exe tools\p-item-embedding\train_p_item_embedding.py `
  --dataset-manifest .local\task085-p-item-dataset\pc-705100-0036\dataset_manifest.json `
  --output-dir .local\task085-p-item-embedding\candidate `
  --epochs 30 --batch-size 128 --learning-rate 3e-4 --device cuda
```

精确训练阶段、历史损失、模型/图库/类表散列记录在最终 `manifest.json`。原始官方图像、数据集、检查点、训练中间产物和实机截图都留在 `.local/`，不得提交或分发。生产固定渲染参考图库是单独登记的发行制品，不等同于训练数据；当前图库为 426 项。RIS 候选激活和完整 Release 打包都会按业务 ID 集合验证全部竞技场 `stage` P 道具已被图库覆盖，缺项时在切换或发布前停止。运行时的全目录详情标题路线只为自定义 bundle、旧活动状态或其他绕过正常激活门的错配提供纵深防御：它不复用相似旧 ID，而是对每个非空槽强制打开详情，并仅按完整标题行唯一消歧；标题无法唯一确认时停止。其来源、散列、用途及游戏图形权利保留边界见 `ASSET_PROVENANCE.md` 与 `THIRD_PARTY_NOTICES/`。

ID 475/476 的目录数据最早已出现在固定的 `gakumas-tools` revision，但当时 `gakumas-images` 只包含至 ID 474，因此曾由 `generate_rendered_reference_extension.py` 从固定官方 DMM 图案、177 个既有背景样本及 473/474 的升级标记规则确定性合成，并作为临时引用要求具体槽位详情确认；该脚本保留为历史制品复现工具，不再是当前活动图库来源。上游随后在不可变提交中补齐 475/476 的 130×130 固定渲染 PNG，并由成功的 production deployment 交付。当前图库只以这两张上游 PNG 替换旧合成项，业务 ID 总数仍为 426，`provisional_business_ids` 为空；`mode=produce` 的 ID 477 继续由 EntityBank 的 `mode=stage` 契约排除。精确部署、Git blob、文件／像素散列和旧图库继承关系写入 manifest；真实 JJC 样本仍作为后置实机校准项收集，但不再构成临时引用或强制详情点击的依据。原始图和 426 张源图继续只留在 `.local/`，发行组件只包含去元数据后的 NPZ、manifest 与公开安全的 `production_handoff_evaluation.json` 三件套。

## 固定渲染参考的构建、晋级与发布

`build_rendered_reference_gallery.py` 只生成规范 LF JSON 的候选目录：`p_item_rendered_reference_gallery.npz`、`manifest.json`，并写入固定 `build.tool_contract` 与 `production_handoff=PENDING_VALIDATION`。候选不得直接进入生产路径，也不得人工改写为 READY。

`promote_static_reference_handoff.py --component p_item_reference` 必须同时取得候选目录、不可变旧图库、完整 builder 源图目录、权威 `p_items.json`，以及通过 `--p-item-production-source-evidence` 指定的版本化冻结证据。当前证据为 `evidence/p_item_production_source_d476.json`；其规范 LF 字节 SHA-256 在晋级器中固定，独立绑定 `gakumas-tools` revision、production deployment、目录字节／Git blob、`gk-img` gitlink、475/476 路径／文件／像素和旧图库。候选 manifest 的来源投影必须与该证据精确相等，不能用候选自己的 provenance 自证；随后工具再对实际旧图库、426 张源图和目录文件逐字节重算，并验证历史继承、NPZ 容器、唯一业务 ID、PNG 解码／元数据清理及完整图库排名，最后原子生成 NPZ、manifest、`production_handoff_evaluation.json` 三件套。当前正式状态为 `READY_WITH_REAL_SAMPLES_PENDING`；这只表示离线交接门通过，真实 JJC 普通／`+` 校准仍为 `PENDING`。

聚合报告将可从发行组件重算的 `validation` 与晋级期 `promotion_evidence` 分开保存，并用 `candidate_manifest_sha256` 绑定晋级前候选。Release 会从发布工具源码树实际重新加载同一份固定散列证据，先做 manifest 来源投影比对，再从包内 NPZ／manifest 重算 validation、核对报告并以包内 RIS 的 `p_items.json` 检查竞技场 `stage` 集合覆盖。冻结证据随公开源码保存但不进入用户运行包；未分发的 426 张 builder 源图也不会被冒充为发布时已重新读取。

## 离线评估

以下命令评估的是独立 embedding 模型，不是固定渲染参考的 `production_handoff_evaluation.json`，两者不得互相代替。

```powershell
.\.venv\Scripts\python.exe tools\p-item-embedding\evaluate_p_item_embedding.py `
  --run-dir .local\task085-p-item-embedding\authoritative-705100-0036-v2 `
  --dataset-manifest .local\task085-p-item-dataset\pc-705100-0036\dataset_manifest.json `
  --output .local\task085-p-item-embedding\authoritative-705100-0036-v2\offline_evaluation.json
```

评估会分开报告图案检索、普通/+状态、406/407/408 计划解析、合成未知拒识和延迟。报告中的阈值仅为离线诊断；在冻结真实竞技场样本前，运行清单不写入生产接受阈值。
