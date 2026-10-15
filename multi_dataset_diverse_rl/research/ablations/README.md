# 单 Prompt 模块消融

此目录用于独立机制研究。统一优化器负责候选生成、完整测评、Validation
门槛与保留；外接模块只改变提供给优化器的证据或上下文。
正式团队实验的组合入口和已冻结单 Prompt 实验均保留。

默认配置在 [default.json](../../../experiments/ablations/pattern/default.json)：
每组3次，三次均使用种子42，重复编号为1–3；每次20个优化机会。
不同重复不共享响应缓存，同一重复的组间、轮间相同 Solver 请求可以配对复用。
Optimizer、Gradient、Cluster 不复用响应；本次不导入旧实验缓存。
种子控制抽样和并列选择，不承诺不同重复必然产生不同 Prompt。

| 文件 | 用途 |
|---|---|
| config.py | 配置校验、重复编号、模块身份和预算草案 |
| engine.py | 与模块无关的优化、严格门槛、检查点与完整边界恢复 |
| generation.py | 所有组共用的单 Prompt 优化器与候选检查 |
| interfaces.py | Optimize-only 模块接口及有调用限额的 Teacher 端口 |
| formal_adapters.py | 直接调用正式 Gradient/聚类/校验，提供正式 Memory 原生接口 |
| pattern.py | 已确认的最大支持 Pattern 选择及证据过滤 |
| modules.py | 显式模块注册；未知模块直接拒绝 |
| analysis.py | Validation/Test 准确率、优化 Token、更新/拒绝/耗尽及组内差异 |
| runtime.py、governance.py | Solver 缓存、阶段授权、冻结、真实运行和终端审计 |
| offline_runtime.py、synthetic.py | 网络隔离的假 Provider 与端到端演练 |

Gradient 和 Pattern 的生成、重试、解析、内容校验、分区规范化及合并来自
正式实现，不维护算法副本。单 Prompt 的最大支持数选择是单独声明的实验
适配规则，不等同于正式团队责任权重选择。

使用仓库运行环境规划和演练：

```powershell
& runs/math_v2_1/runtime/Scripts/python.exe tests/formal_zero_api_runner.py --offline-command scripts/ablations/study.py --config experiments/ablations/pattern/default.json --plan --destination runs/ablations/my_draft
& runs/math_v2_1/runtime/Scripts/python.exe tests/formal_zero_api_runner.py --offline-command scripts/ablations/study.py --config experiments/ablations/pattern/default.json --smoke --destination runs/ablations/my_smoke
```

两个输出目录都必须是新的目录。假运行使用合成数据，不能解释为真实性能。
配置和模块身份进入计划哈希；真实实验还需登记、冻结源码/数据/阶段、
预算与授权。V2 已提供受治理的运行端口，准备入口为
`scripts/ablations/prepare.py`，实际执行统一经 `scripts/run_experiment.py`。
配置草案和准备清单都不能授权真实调用。

后续接入模块时，实现 `binding/initial_state/prepare/observe`，由注册表创建
每个组别/重复独有的模块实例。`prepare` 接收当前 Prompt、选中 Optimize
错题和对应观测；`observe` 接收候选、Optimize 观测及最终决策。
模块没有 Validation 原始观测和 Test 读取端口。返回的 JSON 状态随检查点
保存，只在优化器阶段交付上下文，不进入 Solver。

Memory 应直接包装正式 `PopulationEditMemory`。当前已提供原生类构造及
策略/容量清单，但没有注册可执行 Memory 组。开始该消融前，需要确认：
单 Prompt 初始化范围；缺失 Vote/Shadow 字段的表示；记录哪些候选结果；
怎样向共同优化器交付经验。不能伪造团队指标，也不能把 Accuracy 称为 Vote。

完整方法约定见 [设计说明](../../../docs/design/SINGLE_PROMPT_MODULE_ABLATIONS_V1.md)。
本次重复实验与 Solver 缓存修订见
[V3 三次重复修订](../../../docs/design/SINGLE_PROMPT_PATTERN_REPEAT3_SOLVER_CACHE_V3.md)。

当前模块语义对齐规则见 [V6 审计](../../../docs/design/SINGLE_PROMPT_MODULE_ALIGNMENT_V6.md)。
当前默认注册表仅开放 `none` 与完整 `full_gradient_pattern`；旧 Batch Pattern
仅供历史协议检查，Gradient-only 与 Memory 按用户确认保持未启用。
`study.py --plan` 同时输出完整模块对齐清单；未审计的自定义模块不能据此获得真实执行资格。
共同优化器复用正式参考解投影、私有文本防照抄及推理时不可用信息检查。
