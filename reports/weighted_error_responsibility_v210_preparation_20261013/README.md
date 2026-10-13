# V2.10 逐题加权责任：实现与离线验证

本次采用 `R = 4D + 2N + C + R3 + 0.5R4`，连通了成员错误分类、WHO、证据标签、Pattern 全支持集计分及新版本治理。真实模型 API、历史续跑、Validation/Test 调用均为 0；没有 Push。

## 规则与实现

| 类别 | 逐题条件 | 权重 |
|---|---|---:|
| D | 原反事实分配中的错误成员，替换为正确答案后团队 Vote 从错变对 | 4 |
| N | 原分配中的错误成员，未翻转 Vote，但正 margin gain 使修复后 margin 恰为 0 | 2 |
| C | 原分配中的错误成员，无正确 peer，且不满足优先 D/N 条件 | 1 |
| R3 | 团队 Vote 正确，五人中三人正确、两人错误，目标为错误者 | 1 |
| R4 | 团队 Vote 正确，五人中四人正确，目标为唯一错误者 | 0.5 |

R3/R4 采用用户本次明确补充的定义，未借用历史 Module2 的同名 repair-distance 标签。D/N/C 保留原 `(vote_flip_gain, margin_gain)` 最大反事实合法分配及互斥 lane；没有以正确人数重定义 D/N。每个错误 ID 只记一个标签。正确成员不获修复分。计数为整数，分数保留浮点半分，不做归一化。

`weighted_responsibility.py` 提供唯一当前权重和逐题分类。Analyzer 对全部 Optimize 成员观测生成共享标签和未覆盖原因，EvidenceSource 使用这份标签，而不再把所有团队正确的成员错误默认视为 general。WHO 求和后除以 `1 + failure_count`；Pattern 对聚类后全部支持 ID 求和，不使用失败折扣。最高正分、原同分规则和全零 seeded fallback 保留。`primary_lane` 仅为诊断展示，不控制成员总分。重叠标签、重复 ID、非整数/负计数、缺失或旧政策绑定均拒绝。

Gradient 仍为全部目标成员错误产生基于可观察证据的假设；聚类算法与修复文本契约未改。责任标签用于程序侧 Pattern 计分，不增加 Gradient 模型的评分职责。Pattern 选择后仍由既有代表性支持抽样提供 Reflection 的单步编辑证据。

## 未覆盖错误

五类并不覆盖所有成员错误。本次保留原 D/N/C 分配范围；未分配的团队错误不会自动新增责任。团队正确但仅一人或两人正确的错误者也不属于 R3/R4。未覆盖者仍可参与原全零 fallback，权重保持零。

在 V2.7 的 50 个机会父状态中，共 8,359 次成员错误观测，按新规则分布如下；相同题目、成员、状态可能重复出现，不能当作 8,359 道独立题目。

| 分类 | 观测次数 |
|---|---:|
| D | 614 |
| N | 1,044 |
| C | 2,866 |
| R3 | 1,022 |
| R4 | 225 |
| 未覆盖：团队正确、正确人数不满足 R3/R4 | 934 |
| 未覆盖：团队错误、未入原反事实分配 | 1,346 |
| 未覆盖：已分配但不满足 D/N/C lane | 308 |

新增正责任观测 1,247 次，仍有 2,588 次未覆盖。原被选成员的 1,618 次错误中，D/N/C 为 906，R3/R4 为 214，未覆盖为 498；其中团队正确、成员错误 392 次，仅 214 次满足新增两类，另外 178 次保持零分。

## 封存数据只读重算

重算前后均核对 V2.7 封存 trajectory 的既有 SHA256。D/N/C 在全部五成员、50 个父状态中的计数与历史诊断完全相同，差异为 0。保持历史父状态、候选成员、失败计数和已有 Pattern 分区不变：

| 固定状态比较 | 差异 |
|---|---:|
| 仅 D/N/C 从 max 改为 sum：WHO 改变 | 21/50 |
| 完整五类求和：WHO 改变 | 27/50 |
| 完整五类与 D/N/C-only sum：WHO 选择不同 | 11/50 |
| 原被选成员的已有正分 Pattern：胜者改变 | 16/49 |
| WHO 不变时仍改变 Pattern | 9 次 |

这些是固定历史状态的在线可用输入重计分，没有用 Full 结果调整规则。WHO 一旦改变，就没有该新成员对应的新 Gradient/Pattern/Child 观测；原成员的 Pattern 比较仅是条件性比较，不能拼接成一条新的连续搜索轨迹。另一个机会缺乏可行动 Pattern，未虚构新增分区。不能据此声称 Commit、Vote、质量或科研效率改善。

## Parent Population 与成本边界

Parent Population、单步 Reflection、代表性支持抽样、Fitness、Memory、Pool/完整边界恢复、Full/Shadow、Solver decoding、数学评分、aggregation 的有关组件与父提交逐字节一致。仍是每次三次独立有放回抽样、最多三个 Parent、最低分淘汰、Child 不在同一次机会成为 Parent、最终每池最高分组成团队再联合评估，Reflection 无 NO_SAFE_EDIT。

新方法身份 `unified_team_prompt_search_v2_10_weighted_responsibility_v1`，独立 responsibility/WHO/Pattern scoring、binding/composition/cache 身份；旧 V2.9 常量及历史 schema 分支保留。完整权重政策进入 method mechanism、binding、单次执行 scope、manifest schema 和科学哈希。新 preparation 为未冻结 DRAFT，授权仍关闭；不能载入旧方法 checkpoint 或复用旧 API scope。历史报告、manifest 和成绩不变。

固定预算公式不变：每机会最多 3N 候选 Solver 评估、S 单成员 Shadow、最多 3N 次 Gradient draw、3 次 Cluster draw、6 次 Reflection draw；最终额外 Shadow 最多 5S。所有成员错误原本已在 Gradient universe 内，因此 R3/R4 不直接增加生成槽位或评估面板。WHO 变化可改变所选成员的错误数量、重试和输入体积，实际 Token 和墙钟成本的净变化未知。没有真实实验，团队收益、泛化及效率均未知。

## 验证

最终检查记录见 [verification.json](verification.json)。针对性检查涵盖 1,024 种五人答案组合，D/N/C 对照、同正确人数不同 Vote 反事实、R3/R4 真实生产送达、未覆盖统计、互斥/重复拒绝、WHO 折扣、Pattern 半分求和及新旧 binding 拒绝；既有 Parent Population 验证涵盖抽样、入池、最终组队和恢复。

当前完整套件加相关增量检查，按测试身份取最新结果共 **749 passed、2 skipped**（751 项覆盖）。这不是重新运行一次全部最终测试的结果。完整运行中的旧 schema 与旧 max-score 对照断言已按新科学契约更新并定向重跑；新 WHO 配置身份及 benchmark guard 在最终增量中核验。历史/私有测试与 retired Git-only 用例按现行分类排除，不声称所有历史重放通过。后续真实实验必须另行冻结源、范围和单次授权。

两项需要真实验证的假设仍未识别：混合错误类型求和能否改善实际修复质量；R3/R4 是否帮助后续有用互补性，或将更多预算分配给当前冗余错误。没有新模型实验，不能选择性解释历史排序差异为科研收益。
