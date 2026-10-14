# V2.12 按正确人数覆盖全部成员错误

用户最新明确责任只看五人中几人正确、几人错误，不再依据具体答案分布。
本次将当前责任分类整体替换为人数分类，而非仅取消一个旧筛选条件。
V2.10/V2.11 的冻结规范、历史报告和实验成绩保持原样。

| 正确人数 | 错误人数 | 每位错误成员类别 | 每题责任分 |
|---:|---:|---|---:|
| 0 | 5 | C | 1 |
| 1 | 4 | N | 2 |
| 2 | 3 | D | 4 |
| 3 | 2 | R3 | 1 |
| 4 | 1 | R4 | 0.5 |

正确成员修复分为 0。每个错误成员、每题恰好计分一次。无效输出继续依照
既有 benchmark 判分记为错误，不创造额外责任类别。

规范见 [COUNT_RESPONSIBILITY_V212.md](../../docs/design/COUNT_RESPONSIBILITY_V212.md)。
当前方法为 `unified_team_prompt_search_v2_12_count_responsibility_v1`。
这是依据用户新指示进行的科学语义修订；D/N 已不再具有旧 Vote 反事实含义。

## 数据流

`weighted_responsibility.classify_error` 仅读取五个 Boolean correctness。
没有答案、Vote、margin、opportunity 或 assignment 参数。Analyzer 不再调用旧
`compute_repair_eligibility_sets`，而对所有成员错误建立同一套互斥标签。
保留的 `assigned` 诊断字段现表示全部成员错误，不再做筛选或影响得分。
零分未覆盖计数保留为审计字段，在合法五人观测中恒为 0。

WHO 使用 `R = 4D + 2N + C + R3 + 0.5R4` 并保留 `1+failure_count` 分母。
Pattern 在既有语义聚类后，对所有错误支持样本的共享标签求权重和，不使用
WHO 折扣；重复支持 ID、标签重叠和不合法计数仍被拒绝。

BinaryEvidenceSource 要求每条成员错误具有一个责任标签，附带正确人数。
Root Team Group evidence 冻结和验证采用同一当前分类器，实际输入明确给出
标签与正确人数、权重的映射。历史序列化别名 `direct_flip / near_margin /
coverage` 分别表示 D/N/C；新版本中它们不宣称实际翻转 Vote、抵达平票或无
正确 peer。保留别名避免无关接口重构，版本及政策明确区分旧、新含义。

实际答案、等价组、Vote 与 target counterfactual 仍用于原有团队评估和
Group 诊断，不能参与责任分类。两人正确时 Vote 可以赢、平、输，这三种
情况均为 D；人数责任不是实际 marginal Vote contribution 的估计。

## 封存 Optimize 状态只读重算

读取 V2.7 的 50 个既有机会父状态；前后核对封存 trajectory SHA256。
相同题目、成员及状态可以重复出现，8,359 是成员错误观测次数而非独立题数。
比较器来自 V2.11 冻结源码，保持既有 eligible members、失败计数、原目标
成员的 Gradient/Pattern 分区不变。完整统计见
[historical_selection_recompute.json](historical_selection_recompute.json)。

| 类别 | V2.10/V2.11 | V2.12 |
|---|---:|---:|
| D | 614 | 1,041 |
| N | 1,044 | 1,216 |
| C | 2,866 | 4,855 |
| R3 | 1,022 | 1,022 |
| R4 | 225 | 225 |
| 未覆盖零分 | 2,588 | 0 |
| 总成员错误观测 | 8,359 | 8,359 |

原 30.96% 零分遗漏全部消除。其中 817 次归 D、696 次归 N、1,075 次归 C。
同时有原正分错误发生重分类，不能将此变化描述成单纯增加零分题的权重。
相对于 V2.10/V2.11 固定状态重算，WHO 改变 25/50 次，原目标成员的已有
Pattern 胜者改变 24/49 次。相对于实际 V2.7 WHO，改变 27/50 次。

一旦新 WHO 选择不同成员，封存数据没有新成员对应的新 Gradient、Pattern
或候选结果。上述比较不能拼接成新的搜索轨迹，也不能证明候选质量、Commit、
Team Vote、Token 效率或泛化改善；这些仍为未知。

## 保持的流程与版本边界

Parent Population 三次独立有放回抽样、三个 Parent 上限、Fitness、Pool
淘汰、共享修复方向、单步 Reflection、无 NO_SAFE_EDIT、Memory、最终各池
argmax 组队并联合评估，以及 Solver/scoring、Aggregation、Full/Shadow、
停止规则均未重设。有关 15 个组件已与 V2.11 源码进行字节对照。

新方法、责任分类、WHO/Pattern、binding、composition、cache 和 checkpoint
身份进入关闭的当前契约。MATH benchmark protocol 只修订 responsibility
identity；解析、数学判分、数据划分和聚合不变。旧 manifest schema 分支保留。
新 preparation 为未冻结 DRAFT，API 授权关闭，Validation 未授权、Test 封存。

固定生成及评估槽位不增加。新的成员/Pattern 选择可能改变实际错误数量、
上下文、重试和 Token 消耗；本次没有模型调用，不能估计实际净成本或科研收益。
后续真实实验需要另行冻结新的源、范围、预算及单次授权。

## 验证状态

针对性验证已通过，覆盖 7,776 种五人答案组合、32,400 次合成成员错误，
全部有且只有一个正分标签；正确成员不计分。另验证同正确人数下不同答案
分布和不同 Vote 结果、仅 Boolean 输入的分类、旧 assignment 不被调用、
WHO 折扣、Pattern 求和、此前遗漏样本实际送达 WHO/Pattern/Group 及三个
独立 Reflection Slot。合成覆盖不是现实题目分布或效果估计。

生产 compileall、当前依赖闭包、DRAFT manifest schema 和预执行检查已完成；
真实执行预检查按未冻结、无授权返回 HOLD。当前完整分类套件及定向修复按测试身份取最新结果：**899 passed、2 skipped**，
结果及来源范围见 [verification.json](verification.json)。完整运行的旧反事实 parity
断言和未刷新 Lineage 已修正并定向重验；不声称一次全部最终测试重跑。
治理复用定向检查的 `test_complete_governance_audit` 通过结果。
所有执行使用 credential-free 离线网络 guard，真实 API 和 held-out 数据读取为 0。
