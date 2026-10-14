# V2.12 模块化架构重构与消融接口准备

本次完成的是默认行为保持的架构重构，基线为
`c93838ea922353a5772ca146333a025ea0088681`。默认方法仍为
`unified_team_prompt_search_v2_12_count_responsibility_v1`，架构身份为
`V212_MODULAR_MECHANISMS_ARCHITECTURE_V1`。没有算法有效性实验，没有真实 API
调用，没有读取真实 Validation/Test 内容，也没有恢复历史实验或推送授权。

规范和依赖图见 [模块架构契约](../../docs/design/MODULAR_MECHANISMS_V212.md)。
具体接口、来源、输入输出、依赖、角色、数据范围、状态与适配限制见
[能力清单](capabilities.json)，结论边界见 [分类记录](classifier.json)。

## 职责与原有耦合

| 科学职责 | 重构前的耦合 | 现在的边界与决策权 |
|---|---|---|
| 责任 / WHO | 组合根固定实例化责任与成员选择 | 分别注册工厂；五类分类与失败折扣保持；WHO 只有成员选择权 |
| Gradient / Pattern | 分区归一化、责任评分和选择集中在一个函数；独立 Adapter 调用团队评分的空标签路径 | 共享生成、恢复和分区校验；评分、选择分别注入；独立最大唯一支持选择直接使用合法分区 |
| 支持样本 | 默认抽样固定接入 | 独立支持抽样工厂；只从选定 Pattern 的错误支持集选题 |
| Group | Layer1 同时构造、调用、持久化和传递 | 原有 Root、生成/验证、投影权威保留；Group channel 封装送达并可替换或明确关闭 |
| Reflection | 一个 search 方法同时完成 Parent、证据、模型生成、解析和合法性 | 冻结 Parent selection、Evidence composition、SingleStepReflection 分开；候选生成不部署团队 |
| Population | 状态、概率、淘汰与最终选人集中；恢复重复采样公式 | 状态唯一；sampling、retention、assembly 分离；恢复调用同一抽样策略 |
| Fitness / Evaluation | 实际测量、Fitness 构造、入池与最终判定邻接 | 完整 Solver 测量仍唯一；Fitness、pool update、team admission、final admission 分离 |
| Memory | 记录、状态、检索与字符预算送达都由存储类执行 | 存储/事务/恢复唯一；recorder、retrieval、delivery 注入；完整卡片、范围和实际效果保持 |
| 安全与治理 | 随调用链执行 | 共享校验、私有 Shadow、Commit、停止和 checkpoint 明确登记，仍保有原权限 |

共有 **20 个可选择工厂**，另登记 **15 个已有边界能力**。这不是插件加载平台。
Registry 检查操作签名、身份和重复登记；抽样器同时要求 probabilities 与 draw。
兼容替换不需要改 orchestrator 的机会控制流程。离线测试实际替换 WHO、抽样器，
并关闭 Group、Reflection 和 Memory delivery；不是仅检查类名是否存在。
策略输出仍受 Pool 容量、概率合法性、三槽位和最终五成员来源约束；候选生成器
替换后仍执行共享安全及单块编辑检查。非法 retention 输出不会部分改写 live Pool。

本次没有发现需要改变正式科学行为的独立 Bug。测试开发中一次自定义抽样恢复
失败来自未初始化其余四个成员的夹具；补齐五成员状态后通过，未修改默认恢复规则。

尚存的依赖是必要或明确受限的：Pattern 必须依赖合法 Gradient；Group 必须依赖
真实 Root Team；Reflection 必须获得合法修复方向及真实 Parent 观测；Fitness
必须来自完整测量；Memory 必须来自实际编辑历史；Final Assembly 必须联合评估。
共享合法性和来源校验不能通过普通消融开关绕过。

## 默认行为差分

在修改 tracked source 前，以基线源码实际运行并保存六组完整合成回放。
修改后重放相同夹具和确定性响应，比较完整请求、实际 Reflection 文本、事件、
责任、WHO、Gradient/Pattern、支持题、Group、Parent 概率/顺序、候选、Memory、
Full/Shadow、Commit、终态、停止、checkpoint、按角色调用与 Token 计数。
只发布计数和哈希，合成原文及响应保存在 ignored 私有验证目录。

| 场景 | 假 Provider 成功调用 | 语义/请求/状态差分 |
|---|---:|---:|
| 默认完整机会 | 176 | 0 |
| Reflection 首次格式失败后恢复 | 177 | 0 |
| Group 两次格式失败耗尽 | 177 | 0 |
| Gradient 合法 UNCERTAIN | 111 | 0 |
| 两次优化机会 | 247 | 0 |
| 同成员跨机会 Population / Memory 演化 | 240 | 0 |

详见 [完整规范化见证哈希](equivalence.json)。
独立 Solver 并发启动顺序会变化；比较在连续 Solver 请求段内保留所有完整请求
及重复次数，对其排序，保留优化器请求顺序和全部事件/状态顺序。因此不能声称
墙钟时间、线程调度或原始传输启动列表逐字节相同。未丢弃任何请求字段、Token
计数、缓存身份、scope、评分或终态字段。此规则比较的是部分有序轨迹，不是
以最终分数替代过程验证。六组规范化见证也保存为可重跑的当前 regression tests。

这支持默认合成路径的行为等价，不能证明全部模型输出、输入和运行环境的等价。
另外的现有测试覆盖恢复、重复/非法候选、模型异常、来源篡改、私有 gate 和数据越权。

## 独立消融框架衔接

保留既有 `research/ablations` Registry、kernel、Module Provider、生成、Validation
gate、独立 state/checkpoint/cache；没有建立第二套独立优化器。正式 Gradient 生成、
输出恢复、分区归一化、编辑和基础校验均通过同一权威实现复用。

| 机制 | 复用状态 | 边界 |
|---|---|---|
| Gradient | 已由 Adapter 实际调用 | 单 Prompt 自己的错误与普通可观察输出；不虚构团队责任 |
| Pattern 生成与校验 | 已共享 | 团队加权责任与独立最大支持选择是两个明确策略 |
| Gradient-only | 新增 mock 可用模块 | 真实共享诊断、零 Cluster 调用，显式 `MODULE_DISABLED`；非正式新研究 |
| Prompt edit / budgets / guards | 已共享 | 团队与独立生成输入 schema / wire 各自保留 |
| Memory | 经验表达部分可复用，正式独立 Adapter 仍未绑定 | 需要合法个体测量与范围契约；没有伪造 Vote/Shadow 卡片 |
| WHO / Group / Team Vote / Admission / Assembly | 团队专属 | 不能直接接入单 Prompt 或用假五成员填充 |

Gradient-only 集成测试通过现有 smoke 流程，监测到 Gradient 和 Reflection 调用，
没有 Pattern Cluster 调用；输入的 Optimize/Validation/Test 均为生成的假夹具。
这不授予真实 held-out 数据使用权，也不改变历史独立研究的冻结协议和成绩。

## 干预、恢复与版本

非默认组合必须声明 synthetic intervention 且运行在已加载的零网络、去凭据 guard
下；真实执行路径没有打开算法开关。干预组合有独立 module binding、结果身份与
Pool scope，默认组合不能恢复其 checkpoint。原有状态恢复的原子性、provenance、
初始 floor、完整支持集和停止历史校验保持。Detached objects 保留实际组件而非
静默重置到默认策略。接口不兼容在接入时拒绝；Provider 与 integrity 异常保持 fatal。

明确关闭 Group 时保留安全所需 Root evidence，但不生成模型经验；关闭 Reflection
时没有候选且零生成调用；关闭 Memory delivery 时保留真实记录，仅向优化器交付
空卡片。`MODULE_DISABLED`、合法 `NO_SUPPORTED_INSIGHT`、结构耗尽
`GENERATION_FAILED` 和运行异常分别记录。并未宣称每种机制都有合法的禁用对照。

默认方法、策略、Prompt wire、采样 RNG、binding/composition/cache lane 和 checkpoint
schema 标签保持；源文件与科学契约哈希已经变化。新的准备 manifest 为 DRAFT，
source/dataset/split/freeze 尚未绑定、API authorization=false。旧执行 scope、授权、
真实 cache/checkpoint 不能因语义标签相同而继续使用。后续真实消融需独立规范和冻结。

## 检查范围

本任务跨组合、证据、生成、测量、持久化和独立 Adapter，故按最高相关检查范围运行
production compileall、完整 classified current suite、治理/manifest preflight、
确定性 smoke、网络 guard、脱敏、artifact hashes 和 diff check。
实际结果由 [验证记录](verification.json) 给出；不以聚焦测试冒称历史全回放通过。
完整当前集为 **923 passed、2 skipped、0 failed**；最终聚焦集为 **24 passed**，
不与完整集相加计算独立用例数。compileall、当前治理、DRAFT manifest、依赖闭包、
脱敏、链接、artifact hashes 和 diff check 通过；应用进程网络尝试为 0。
真实执行 preflight 为预期 HOLD：`PREEXECUTION_NOT_FROZEN`、
`EXECUTION_BINDING_NOT_FROZEN`。本次不打开真实执行授权。
明确排除 224 个历史/私有模块及 5 个仅存旧 Git 的 retired 用例。
两个 skip 分别是 HoVer、PUPA 尚无冻结真实数据 manifest 的既有数据身份检查；
没有下载或补读这些数据。
历史/私有模块及 Git-only retired 用例保持排除，不恢复过去六轮、GEPA 或旧控制器。
已有历史报告与冻结 manifest 没有改写；两个原有未跟踪报告目录保留。

## 后续单机制研究建议

| 优先项 | 合法对照 | 预期增量成本 | 主要混杂与需要冻结的内容 |
|---|---|---|---|
| Group experience | 保持 Root、支持题与 Parent 计划，仅关闭 Group 生成/送达 | 每机会节省最多两次 Group 请求 | 后续候选/团队轨迹分叉；不是去掉 Root 证据 |
| Memory delivery | 保持真实记录/Pool，交付空经验卡片 | 不新增 Solver/优化器请求，减少输入文本 | 测到送达效果，不能称全部 Memory off；状态依赖和适用性 |
| Pattern selection | 固定合法分区，比较明确评分/选择策略 | 无新增模型调用 | 新修复方向改变支持题及后续演化；不能利用未来 Full 排名 |
| Parent sampling | 固定完整 Fitness 测量，比较兼容抽样策略 | 无额外请求配额；实际去重命中会变 | 采样与历史 Fitness 上下文不同；保留三独立槽位 |
| 独立 Gradient / Pattern | 既有独立 kernel 内使用诊断、聚类或合法无模块原始错误对照 | 聚类开关每机会至多三次 Teacher 请求差异 | 必须匹配 Teacher/Solver 预算解释，不能伪造团队标签 |
| Memory 单 Prompt | 先定义个体实际编辑效果与投递来源再适配 | 未冻结，未知 | 缺少 Team Vote/Shadow，不应复制团队记录或宣称等价 |

上述只是未来设计入口；本次未开展真实单模块消融。工程重构、默认合成等价和
Mock 接线已验证；多模块真实科研效益、Team Vote、Token 效率与泛化收益仍未知。
