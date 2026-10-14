# V2.12 Count-only Responsibility

按用户最新指示，以五个成员的判分结果分类全部成员错误：正确人数
0/1/2/3/4 分别对应 C/N/D/R3/R4，每位错误成员权重为 1/2/4/1/0.5。
WHO 保留失败折扣，Pattern 使用同一逐题权重求和；具体答案和 Vote 反事实不再决定类别。

V2.7 封存父状态的 8,359 次成员错误观测中，旧规则的 2,588 次零分遗漏全部消除。
固定状态下 WHO 相比 V2.10/V2.11 改变 25/50 次，原目标成员已有 Pattern 胜者改变
24/49 次。这些变化不证明候选质量或团队成绩改善。

当前完整分类套件与相关修复按测试身份取最新结果：**899 passed、2 skipped**。
完整运行的旧分类对照断言和派生 Lineage 已修正并定向重验；不是一次全通过重跑。
7,776 种合成五人配置验证了正责任全覆盖、互斥和答案分布独立性；真实生产
Mock 连通 WHO→Pattern→Group→三个独立 Reflection Slot。有关 15 个 Parent
Population、Memory、Solver/scoring、Aggregation、Admission 等组件保持字节一致。

- [实现、语义及成本说明](implementation.md)
- [历史固定状态重算](historical_selection_recompute.json)
- [验证范围与结果](verification.json)
- [结论分类](classifier.json)
- [新方法规范](../../docs/design/COUNT_RESPONSIBILITY_V212.md)
- [未授权 DRAFT manifest](../../experiments/manifests/count_responsibility_v212_preparation_v1.yaml)

真实 API、Validation/Test 读取和历史写入均为 0。新实验尚未冻结或授权；
多 Parent 效果、真实候选质量、Team Vote、Token 效率和泛化均未知。本次无 Push 授权。
