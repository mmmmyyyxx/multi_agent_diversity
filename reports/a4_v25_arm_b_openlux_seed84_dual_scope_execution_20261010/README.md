# OpenLux Arm B Canary / Pilot 实验报告

冻结源码：`01306fbf9cde002291ab21e2f93a2c02399f7e9e`。Solver 为 OpenLux gpt-4o-mini；优化角色为 lwj qwen3.7-flash。Seed 84，五个成员分别请求，从相同完整 Arm B Prompt 初始化。两阶段的 Memory、响应缓存、账本及授权独立。

| 阶段 | 初始 Vote / Oracle | 最终 Vote / Oracle | 完成机会 / Commit | Token | 终态 |
|---|---|---|---|---:|---|
| canary | 10/12 / 11/12 | 10/12 / 11/12 | 1 / 1 | 283,304 | CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE |
| pilot | 47/60 / 50/60 | 49/60 / 53/60 | 5 / 2 | 1,680,226 | OPERATIONAL_OPPORTUNITY_CEILING |

## 初始条件、覆盖与搜索效果

canary 初始成员正确数 [9.0, 10.0, 11.0, 9.0, 10.0]；最终 [10.0, 10.0, 11.0, 9.0, 10.0]。成员重叠、错误相关性、独占正确覆盖和 0–5 成员覆盖直方图见 initial_metrics、funnel_summary 与 integrity_audit。
新增 Oracle 覆盖 0 道，损失 0 道；成员正确性修复 [1, 0, 0, 0, 0]，正确题损失 [0, 0, 0, 0, 0]。这区分了新增团队覆盖与已有覆盖内的分布调整。
数学有效但错误 → 正确：[1, 0, 0, 0, 0]；格式无效 → 正确：[0, 0, 0, 0, 0]。Vote 新增正确 0 道、损失 0 道。
新增 Oracle 覆盖中，Vote 判对 0 道、仍判错 0 道；Vote 在原有 Oracle 覆盖内修复 0 道。
格式无效 Profile：0 → 0。实际数学修复与 invalid→valid 转换分别保留在候选效果计数中。
搜索导出 1 个候选，Probe 1，Full 1，Shadow 1。未到达的分支在逐机会记录中标记 NOT_OBSERVED。每机会目标成员、Pattern 哈希、每代 Parent/Child 哈希、修复/损失、排序与 Commit/Reject 见 funnel_summary；第一完整机会另经独立原生复算。
pilot 初始成员正确数 [46.0, 45.0, 46.0, 47.0, 47.0]；最终 [48.0, 49.0, 46.0, 47.0, 47.0]。成员重叠、错误相关性、独占正确覆盖和 0–5 成员覆盖直方图见 initial_metrics、funnel_summary 与 integrity_audit。
新增 Oracle 覆盖 3 道，损失 0 道；成员正确性修复 [4, 4, 0, 0, 0]，正确题损失 [2, 0, 0, 0, 0]。这区分了新增团队覆盖与已有覆盖内的分布调整。
数学有效但错误 → 正确：[4, 4, 0, 0, 0]；格式无效 → 正确：[0, 0, 0, 0, 0]。Vote 新增正确 2 道、损失 0 道。
新增 Oracle 覆盖中，Vote 判对 0 道、仍判错 3 道；Vote 在原有 Oracle 覆盖内修复 2 道。
格式无效 Profile：0 → 1。实际数学修复与 invalid→valid 转换分别保留在候选效果计数中。
搜索导出 6 个候选，Probe 6，Full 5，Shadow 3。未到达的分支在逐机会记录中标记 NOT_OBSERVED。每机会目标成员、Pattern 哈希、每代 Parent/Child 哈希、修复/损失、排序与 Commit/Reject 见 funnel_summary；第一完整机会另经独立原生复算。

## Pilot 逐机会结果

| 机会 | 目标成员（从 0 编号） | Actionable / Exhausted | Layer1 代数 / 导出 | Full | Shadow | Commit |
|---|---:|---|---|---|---|---|
| 1 | 0 | 5 / 9 | 6 / 3 | 2 | PASS | YES |
| 2 | 1 | 9 / 6 | 6 / 2 | 2 | PASS | YES |
| 3 | 2 | 9 / 5 | 0 / 0 | NOT_OBSERVED | NOT_OBSERVED | NO |
| 4 | 0 | 3 / 9 | 6 / 0 | NOT_OBSERVED | NOT_OBSERVED | NO |
| 5 | 3 | 4 / 9 | 6 / 1 | 1 | REJECT | NO |

Pilot Gradient 共 68 个题级诊断、164 条物理响应，38 个耗尽；额外结构恢复 96 条响应，实报 usage 394,485 Token（阶段费用 23.5%）。Solver 额外语义恢复 42 条响应、48,496 Token（2.9%）。这区分 Solver 答案解析成本与优化角色的结构化输出成本。

第五轮 Full 的目标成员从 47/60 提升至 49/60、Vote 保持 49/60，随后 Shadow 拒绝，编辑未部署。第一轮另一个 Full 候选有成员 +4、Oracle +3，但 Vote -1，因此也被拒绝。开发面板的局部收益不能替代 Shadow 判定。

按角色费用见 [api_ledger_summary.json](api_ledger_summary.json)，逐代 Parent/Child 和效果见 [funnel_summary.json](funnel_summary.json)，恢复原因与成本见 [recovery_costs.json](recovery_costs.json)，所有机会的接纳守卫和 Fixed Peers 检查见 [all_opportunity_guard_audit.json](all_opportunity_guard_audit.json)。

## 恢复、费用与完整性

两阶段共计 1,963,530 Token，终态未决预留为 0，各自 charged + reserved 从未超过 2,000,000。按角色和阶段的费用、实报 usage / fallback、预留峰值与回执数见 api_ledger_summary。Solver 语义恢复、Gradient 结构恢复及额外 draw 的实报 usage 成本见 recovery_costs；这些计数不推断供应商账单。

冻结源码与 Startup、模型路由、请求参数、分离的成员 lane、账本哈希链、Provider 回执、终态目录与关闭凭据均已审计。两项 Scope 永久关闭，不复用余额、缓存或初始成绩；没有重跑或额外可用性探测。Validation / Test 调用和原始访问均为 0。

离线验证：当前分类套件 606 通过、2 跳过，220 个历史/私有模块未执行。新增治理首轮 71 通过、11 失败；修正路由节点的 active 登记后，同一失败身份集合 11/11 复查通过。冻结后检查和运行审计均使用零网络守卫；不声称历史重放全部通过。

工程修复及恢复语义见此前 immutable preparation evidence 和 docs/design/GENERATED_OUTPUT_RECOVERY_V25.md；本轮路由迁移见 docs/design/PROVIDER_ROLE_ROUTING_V1.md。没有在真实运行中修改算法、Prompt 初始条件、格式解析、数学等价判定、预算或停止规则。

Seed83 的旧拒绝已在原源码复现：短参数符号关系触发字面匹配，未触发数值硬泄漏。现行守卫结合来源与通用语境处理这一重合；它仍是有限来源规则，不构成不存在语义泄漏的证明。Typed Gradient 输出错误每题最多三次，耗尽后保留其他合法证据；Pattern 对已知合法支持的内容失败局部丢弃，未知、重复或冲突 ID 仍停止。Layer1 保留此前合法候选。身份、来源、账本、授权、存储和未知错误仍须中止。此前进程消失的根因未确定，本轮不宣称已解决。

## 科学判定与下一步

成功执行与接纳编辑分别记录，工程完整性不代表方法有效。Optimize 为搜索面板，Shadow 为自适应接纳面板；本轮没有独立 Validation / Test 泛化证据。Pilot 在完成冻结的 5 次机会后达到运行上限，未耗尽 Token 预算；应视为运行截断，不称为科学收敛。

Pilot Optimize 补入 20 道历史使用样本；两阶段预冻结面板重叠 12 道 Optimize、1 道 Shadow。选择仅依据元数据，且两阶段模型实现独立，不能把两个面板称为独立样本，也不能把阶段间差异解释为优化的因果效应。

下一步优先诊断 Gradient 的 JSON/schema 失败及结构恢复成本，并检验新增正确覆盖如何形成可被 Vote 利用的成员共识。Solver 初始恢复后格式有效率为 100%，本次已部署收益全部来自数学有效但错误的 Profile；当前证据支持优先分析优化角色的结构约束。任何新的调参或独立验证均应另行冻结与授权。本次结果只支持本次开发面板的描述性结论。

执行后的治理回归：82 项通过、0 失败、0 跳过。编译、manifest、治理与脱敏核验通过；最终生成索引另行核验。
