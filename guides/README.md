# Gakumas Helper 指导中心

本目录是当前公开版本的使用指南。根 [README](../README.md) 提供功能和使用入口；具体操作、排障、反馈和安全约束以这里的专题指南为准。

## 按使用阶段阅读

| 阶段 | 指南 | 解决的问题 |
| --- | --- | --- |
| 安装前 | [与上游的区别](UPSTREAM_DIFFERENCES.md) | 这个派生版新增什么、哪些仍沿用上游 |
| 第一次运行 | [首次使用](GETTING_STARTED.md) | 下载、校验、管理员启动、首次己方重算 |
| 使用竞技场 | [竞技场胜率功能](ARENA_WIN_RATE.md) | 计分规则、缓存、参数、真实挑战边界 |
| 遇到错误 | [故障排查](TROUBLESHOOTING.md) | 按界面提示和错误码恢复，不盲目重试 |
| 更新前后 | [更新与回滚](UPDATE_AND_ROLLBACK.md) | MFA 内置完整包、pip 与 RIS 边界、首次安装和故障恢复 |
| 反馈之前 | [隐私与安全](PRIVACY_AND_SAFETY.md) | 哪些数据只留本机，Issue 如何脱敏 |
| 提交反馈 | [问题反馈与分流](../SUPPORT.md) | 发到本项目还是上游、需要提供什么 |
| 参与开发 | [贡献说明](../CONTRIBUTING.md) | 代码、测试、资源和来源要求 |

## 最短安全路径

1. 使用独立目录安装，不覆盖任何既有版本。
2. 以管理员身份启动 Maa，并保持游戏已经登录。
3. 首次只运行“重算竞技场己方总分”。
4. 确认缓存摘要、期数与编成正确后，再运行会挑战的任务。
5. 遇到不唯一或不完整提示时停止；不要通过降低识别阈值、删除待决记录或反复点击绕过。
6. 反馈时只提交脱敏后的最小复现信息。

## 文档状态

这些指南描述当前公开包中已经存在的用户入口。尚未完成的培育高分策略、跨模块整合、独立模型热更新或 MirrorChyan 发布不会在此冒充可用功能。

## 从源码运行离线评测

在公开源码根目录使用项目的 Python 环境运行。示例使用合成局面和固定种子，无需客户端、账号或第三方数据。

```powershell
python -m tools.task280_gkbench_smoke --local-seed 280
python -m tools.task285_seed_bench --worlds 4 --planning-worlds 4 --output benchmark.json.gz
python -m tools.task280_local_unclipped --arm draw-first --output local-episode.json.gz
python -m tools.task285_seed_bench --continuation prior --two-round --output two-round.json.gz
```

第一个示例校验公开动作绑定与规则回放。第二个示例在相同裁判种子上比较即时得分策略、独立采样规划和首步诊断组，并在报告中分别记录终局分数、非法动作和配对差值。诊断组只用于分析模拟中的随机状态影响。

单局耗时诊断记录超过 5 秒及 6050 毫秒的决策。局面使用独立的整局时限，`--arm prior` 可运行同一局面的对照策略。两轮评测按首轮体力缺口和剩余的实体瓶位构建第二轮局面，使用独立于首轮的裁判种子，按首轮登记分加第二轮原分计算配对结果。两个命令均需指定新的输出文件，报告中的得分用于核对合成规则和评测过程。

离线整局运行器也可接入 `tools.task280_decision_box.open_decision_box()`。`uinet-1.1` 和 `uinet-2.1` 接收调用方提供的模型、考试上下文及已确认的出牌历史，`strong` 接收调用方提供的策略工厂。二次取牌可显式配置策略工厂。三档测试均使用合成替身，覆盖公开观测投影、合法动作绑定和实际离线规则回执。
