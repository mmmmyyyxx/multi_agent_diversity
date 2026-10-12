# V2.9 Vote-Guided Parent Population：实现与离线验证

本轮是用户明确指定的方法变更。以本地 `905268ba`、远端 `63796dc8` 为审查起点；
V2.8 实验准备已在冻结前关闭，未调用模型。历史实验与成绩未修改。
当前方法的权威定义见 [V2.9 科学契约](../../docs/design/VOTE_GUIDED_PARENT_POPULATION_V29.md)。

## 方法选择

Layer2 保留 WHO、Responsibility 和 Gradient/Pattern 的基本语义。每次有可行动
Pattern 的机会，先从目标成员的机会起始 Pool 独立、有放回抽取三个 Parent，再各做
一次 Reflection。三个候选共享修复方向和最多三条代表性错误支持；Child 不进入本次
抽样范围。输入区分根 Prompt 的诊断与被选 Parent 的实际已缓存解题证据。

每个 Pool 最多三个不同 Prompt。采用归一化 Fitness 的 softmax（β=4）混合 10%
均匀概率，使低分 Parent 保持非零机会。最低当前 Fitness 出池，平分按 Prompt
SHA256 确定顺序；没有部署 Prompt 保护配额。完整 Optimize Accuracy 是新候选的
代理分数；同一预测与固定 Peers 组合获得实际 Vote 后立即替换代理。评分来源、支持
范围和团队上下文明确记录。历史团队不同，混合 Fitness 不是严格可比的贡献估计。

旧六轮链、42 次局部 Metric、四候选导出、修复计数 Probe 门槛与两候选 Full 限制
退出当前生产路径。所有合法唯一 Child 都获得完整 Optimize 评估。NO_SAFE_EDIT
不再合法；固定三个槽位各最多两次结构恢复，不用失效候选填充成功数量。

Pool 保留与正式部署分开。逐机会部署仍通过原有单成员 Full/Shadow 与原子 Commit。
搜索结束按各 Pool 最高 Fitness 组成精确的最终提案，对该组合联合测量 Optimize
Vote，并将原接纳规则明确推广到所有发生变化的成员和单个联合 Shadow 胜者。
最终组队不额外要求比已优化 incumbent 再提升一次；完整指标安全的中性组合可以
接纳。逐机会 Commit 的严格 progress 不变，最终组队不增加机会 Commit 数、不重置
停止计数。这是与旧逐机会 Admission 的明确语义区别，不是兼容性回退。
拒绝时明确报告 FINAL_ASSEMBLY_REJECTED；安全 incumbent 单独保存，不冒充最终
选择结果。最终提案一经冻结不再根据联合评估结果重选。

Memory 简化为成员内、来源明确的单步编辑经验，含实际编辑、意图、完整测量与真实
编辑 Parent 的修复/丢失计数。保持 1200 字符可读卡片预算；过长指令明确省略，不
要求模型执行数字切片。记录仍是有适用范围的观察，不能解释成普遍因果规律。

参考 [C-EVOLVE 官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3f1351a10a3f904e6f3845a94275a909-Abstract-Conference.html)
的 Population 思路；本实现遵循本任务指定的最后 Vote / Accuracy 代理规则，并非
复现该论文的评分、分岛或最终选择机制。

## 成本与科学影响

令完整 Optimize 数量为 N，Shadow 为 S，以下为最多逻辑 Solver 评估：

| 项目 | V2.8 | V2.9 |
|---|---:|---:|
| 每机会 | 2N+78+S | 3N+S |
| N=60、S=40 | 238 | 220 |
| N=150、S=40 | 418 | 490 |
| 最终联合 Optimize 新调用 | 原终态无独立组队 | 0，重组已测预测 |
| 最终联合 Shadow 额外上界 | 0 | 5S |
| Reflection 输出契约调用上界 | 6 | 3×2=6，正常为3 |

Solver 四次语义尝试、Gradient/Cluster 三次输出恢复及传输上界继续计入预算。
完整范围评分减少局部样本代理的不确定性，却增加在同一 Optimize 上的适应性评估。
不会建立独立泛化证据；没有新授权真实预算。实际 Token 节省或性能提升均未知。

这一设计保留明确的风险：旧 Vote 可能来自不同 Peer 状态，单独最优的五个 Parent
不一定构成更好的组合；高 Fitness 也不保证 Reflection 的修复方向仍对该 Parent
有必要。来源标注、真实 Parent 观测与联合终态门禁处理可观察风险，不能证明模型
生成了合理修复。强制尝试替代弃权的净收益仍需真实实验。完整 Optimize 评分能减少
局部代理误差，却不能消除 Solver 随机性或开发集过拟合。

## 验证与限制

验证详见 [verification.json](verification.json)：完整 current 首轮 721 passed、3 failed、2 skipped；
三处旧 fixture 已迁移并定向复验通过。结合后续新增语义/封存入口检查，逐测试去重覆盖
当前收集的全部 735 个用例：733 passed、2 skipped、0 failed。这是完整套件加定向增量
的覆盖汇总，不是声称重跑过一次新的全套。224 个历史/私有模块及 5 个退休旧机制用例
明确排除；不声称全历史回放通过。

Mock 中的成功修复仅证明
实现和治理连通，不能证明真实生成质量、单步多 Parent 优势或团队收益。

| 判断 | 当前证据范围 |
|---|---|
| 新机制正确实现 | OFFLINE_CONFORMANCE_PASS |
| Mock 与离线 conformance | PASS，网络守卫调用尝试 0 |
| 优于旧六轮搜索 | UNKNOWN |
| 真实 Team Vote、Token 效率、泛化 | UNKNOWN |
| 真实 API / 历史续跑 / Validation / Test | 全部 0 |
| Push | 未授权、未执行 |

恢复只支持经校验的完整机会边界；Pool、预测、部署状态、初始能力、Memory、历史和
停止计数共同恢复。机会内部中断与最终选择后状态拒绝自动搜索重放，恢复不授予 API
权限。正式新实验仍需独立冻结、 exact-SHA handoff 和单次明确授权。

历史代码及旧 conformance 的恢复源和分类见
[退休清单](../../docs/archive/v28_layer1_retirement.md)。省略的历史/私有测试不计作 PASS。

后续优先做成本受控的真实完整流程验证，观察三槽位合法候选率、重复/结构失败、
Pool 实际来源和留存、完整评分与部署转化、最终组合接纳及按角色 Token。
若要判断是否优于六轮方法，应另外冻结可比预算的对照；单个 V2.9 执行不能区分
Population、完整评估、弃权取消与最终组队各自的因果贡献。本轮没有执行此方案。

编译、当前依赖闭包、治理、Manifest、闭合 preflight 与 diff 检查通过。旧结果与
Manifest 未修改；真实研究表现仍未测量。新版本处于已验证、未冻结、无可执行授权状态。
