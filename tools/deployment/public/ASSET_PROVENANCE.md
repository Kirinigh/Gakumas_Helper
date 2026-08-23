# 公开制品来源与权利边界

本表用于区分可公开构建输入、最终运行制品和禁止进入公开包的本机材料。它记录技术来源和项目发布边界，不构成对第三方知识产权的额外授权或法律保证。

| 制品 | 固定来源／生成方式 | 公开内容 | 禁止公开内容 | 权利／通知 |
| --- | --- | --- | --- | --- |
| MaaGakumasu 应用、Agent、任务资源和基础模板 | `SuperWaterGod/MaaGakumasu` 的官方 `v1.4.8` Windows x86_64 Release，构建时核对 SHA-256 后替换本项目已提交树 | 本项目清洁修改及固定上游二进制包 | 上游账号、CI 密钥、用户配置与运行日志 | 根目录 `LICENSE`（AGPL-3.0）；不表示项目获得游戏商标或图形所有权 |
| 技能卡 embedding 基线 | 本项目脚本从本机固定官方 DMM 资源训练；最终模型与图库散列写入各自 `manifest.json` | ONNX 参数、128 维数值 embedding、小型强化标记参考张量、类表、生成脚本 | 原始卡图、提取图标、数据集 manifest、检查点、增强图、实机截图、私有样本指纹 | 项目代码与自产参数按根许可证发布；原始游戏图形权利不随 AGPL 转移 |
| 竞技场技能卡 embedding | 前述基线加本项目合成强化覆盖训练；本机正样本截图不进入模型 manifest 或公开包 | ONNX 参数、数值 embedding、小型强化标记参考张量、类表 | 真实用户截图、ROI、私有正样本字节及其可链接指纹 | 与技能卡基线相同；manifest 不含私有样本散列 |
| 竞技场徽标／费用参考 | 本项目构建脚本消费固定 `gakumas-tools@3e9e8ebdc929dedd32cdd8d8911e5d8a905f76b6` 数据和本机官方素材，输出粗粒度颜色条、二值掩膜和费用符号 | 固定数值／掩膜 NPZ、manifest、生成脚本 | 原始卡图、开发截图和标注集 | `gakumas-tools` 部分保留 BSD-3-Clause 通知；游戏图形权利由原权利人保留 |
| P 道具参考 | 固定 `gakumas-tools` 420 个业务 ID 映射加本机官方 DMM 图标生成 | 生产所需的 420 项固定渲染图标容器、manifest、生成脚本 | 原始资源包、对象管理器、训练数据集、检查点、实机截图 | `gakumas-tools` 映射保留 BSD-3-Clause 通知；容器内游戏图标不纳入项目 AGPL 授权，相关权利由原权利人保留 |
| 竞技场模拟引擎 | `surisuririsu/gakumas-tools` 与 Node.js 按脚本构建；每个发布批次通过显式提交、源归档 SHA-256 和 Node.js 版本固定 | Release 包内固定 JavaScript、数据、Node.js 运行时及完整许可证；精确版本与逐文件散列写入 Release manifest 和包内 `assets/arena-winrate/manifest.json` | 任意移动 `latest`、无清单额外文件 | BSD-3-Clause 与 Node.js 许可证随二进制包保留 |

发布硬门：公开源码只来自独立单提交允许清单；完整包只从该公开提交、固定上游 Release、固定引擎和固定 Python 依赖构建。任何用户配置、日志、缓存、截图、阵容、对局记录、凭据、本机路径、私有训练语料或无许可证参考项目内容均失败关闭。
