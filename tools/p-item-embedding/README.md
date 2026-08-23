# P-item embedding toolchain

本目录为竞技场 P 道具提供独立的 64×64 RGB → 128 维 L2 嵌入、Top-5 余弦近邻图库和普通/+状态解析。它不复用技能卡图库，也不修改技能卡识别、Live 决策、UI、默认值或点击路径。

## 数据集

业务目录固定到 `surisuririsu/gakumas-tools@3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6`，官方图像固定到 DMM PC 清单 `pc-705100-0036`。当前离线安全范围为 `sourceType != produce` 的 420 个业务 ID；实际竞技场可见子集仍需由冻结实机画面确认。

```powershell
.\.local\task075-card-embedding\.venv\Scripts\python.exe tools\p-item-embedding\build_p_item_dataset.py `
  --catalog-csv .local\task085-upstream\p_items.csv `
  --catalog-tree .local\task085-upstream\gakumas-tools-tree.json `
  --official-manifest .local\task075-upstream\manifest-pc.json `
  --official-manifest-label pc-705100-0036 `
  --object-manager-root .local\licensed\GkmasObjectManager\7da94255b10e367bf83fadb805a0c5f643998de9\GkmasObjectManager-7da94255b10e367bf83fadb805a0c5f643998de9 `
  --output-dir .local\task085-p-item-dataset\pc-705100-0036
```

构建器会核对 420 个业务 ID、507 个官方 P 道具资产、可解码性、前景遮罩映射、普通/+配对、跨 ID 重复及 406/407/408 官方同图组。固定渲染图标只在完成官方原图映射后用于渲染域适配，不独立决定业务 ID 或图案身份。

## 训练与导出

PyTorch 仅安装在隔离训练环境，主项目运行环境不增加该依赖。最终候选延续分阶段检查点到 600 epoch，并为 16/24/32/48 px 增加结构化图库原型；生产接口仍然只输出嵌入和 Top-5，不使用封闭式分类头。

下面命令用于从当前代码建立新的实验候选；选定 v2 是分阶段续训结果，其每段学习率与训练域记录在最终 `manifest.json`，不能把单次固定学习率命令误写成该成品的逐字节复现步骤。

```powershell
.\.local\task075-card-embedding\.venv\Scripts\python.exe tools\p-item-embedding\train_p_item_embedding.py `
  --dataset-manifest .local\task085-p-item-dataset\pc-705100-0036\dataset_manifest.json `
  --output-dir .local\task085-p-item-embedding\candidate `
  --epochs 30 --batch-size 128 --learning-rate 3e-4 --device cuda
```

精确训练阶段、历史损失、模型/图库/类表散列记录在最终 `manifest.json`。原始官方图像、数据集、检查点、训练中间产物和实机截图都留在 `.local/`，不得提交或分发。生产使用的固定 420 项渲染参考图库是单独登记的发行制品，不等同于训练数据；其来源、散列、用途及游戏图形权利保留边界见 `ASSET_PROVENANCE.md` 与 `THIRD_PARTY_NOTICES/`。

## 离线评估

```powershell
.\.venv\Scripts\python.exe tools\p-item-embedding\evaluate_p_item_embedding.py `
  --run-dir .local\task085-p-item-embedding\authoritative-705100-0036-v2 `
  --dataset-manifest .local\task085-p-item-dataset\pc-705100-0036\dataset_manifest.json `
  --output .local\task085-p-item-embedding\authoritative-705100-0036-v2\offline_evaluation.json
```

评估会分开报告图案检索、普通/+状态、406/407/408 计划解析、合成未知拒识和延迟。报告中的阈值仅为离线诊断；在冻结真实竞技场样本前，运行清单不写入生产接受阈值。
