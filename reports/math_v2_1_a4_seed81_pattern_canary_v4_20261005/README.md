# Pattern Specific Content Guard V3 — Canary attempt4

A4、Seed81 的单次冻结 Canary 已完成一个 production opportunity，随后按冻结条件停止。执行状态为 `EXECUTION_COMPLETE`；owner 在完成独立审计后将其分类为 `VALID_OPERATIONAL_CANARY`。本次用户授权已消费并关闭。

源码为 `011b41d37f6954d128764d3af439a4ff3ce954f8`，冻结仓库提交为 `bbee178b2882c5e1b922cdb3c761d17bacc76084`。Solver 为 qwen3-8b，Reflection / Pattern 为 qwen3.7-flash；各角色保持冻结解码、重试和缓存策略。

| 项目 | 观察结果 |
|---|---:|
| 完成 production opportunity | 1 |
| Pattern / Reflection 成功调用 | 1 / 6 |
| 成功 provider 调用 / 传输失败 | 362 / 0 |
| 提案 / 导出 / TeamProbe / Full | 6 / 4 / 4 / 2 |
| 本次账本计费 token | 234,925 |
| 累计剩余 token | 38,488,351 |
| terminal invalid / 恢复请求 | 7 / 1 |
| Shadow gate 通过 / commit | False / 0 |
| Validation / Test 调用 | 0 / 0 |

WHO、完整 wrong universe、严格 alias 解码与 partition、共享 F、focus 选择、代表样例与 preservation anchor、局部分数、Memory 状态、候选 funnel、TeamProbe / Full、transition、Shadow gate 和最终团队状态均已从本次冻结私有证据与缓存独立复算。账本哈希链、请求字段、成员 lane 缓存身份、原始 artifact inventory 和启动源码身份通过核验。所有审计新增 provider 调用为零。

V3 在本次真实输出中通过了严格抽象检查。离线通用词回归与这次完整生产路径共同支持该实现修复的可运行性。Canary 只验证操作完整性和已观察到的机制使用；不估计方法收益、泛化或 Pattern / Memory 的因果贡献。Optimize 上的分数变化属于搜索过程观察。

具体内容检测仍有冻结声明的边界：无支持上下文的裸专名、事实改写、未定界表达式不保证检出。无效 Solver 输出沿用冻结恢复规则，达到上限后计为 incorrect 且不投票；未放宽解析器或调整重试。历史三个 attempt 及其证据保持原状。

本次未启动 Pilot、Validation 或 Test，也未 push。报告仅包含 hash、计数、类别、配置与指标；原始题目、gold、模型答案、请求与响应保留在本地私有 evidence 中。
