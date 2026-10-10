# Canary / Pilot 精确授权清单

冻结源码：`26b9ffed416470aabbc18477b65c8874a166e454`。以下两项均尚未执行，等待一次精确授权。

| 项目 | Canary | Pilot |
|---|---|---|
| Attempt | a4_v25_arm_b_recovery_seed84_canary_attempt1 | a4_v25_arm_b_recovery_seed84_pilot_attempt1 |
| Seed | 84 | 84 |
| Optimize / Shadow | 12 / 40 | 60 / 40 |
| 最大机会 | 1 | 5 |
| charged + reserved 上限 | 2,000,000 | 2,000,000 |
| Scope SHA256 | e3f3296fe02f212ccde65fd2054e46927b0f24568739afcdc8d78b6b2af089e0 | 5a66f703677a767664d916c271a273fcd34591051904e1e777e3881f798776d1 |
| Startup SHA256 | eaf697eee21945f200c4cb1fce9ac855e97fe3cf4e81fcd3952166e0b0f050f6 | 94f852ef31357ffa0435ae95897115eca3a64d839d5522c50ec830e1db7ebc7d |

合计最多 4,000,000 tokens；各一项 attempt，无重跑。Solver qwen3-8b；Gradient、Pattern、Reflection qwen3.7-flash。所有 thinking=false；并发 8/1。五个初始成员均为同一 B Prompt，两项独立初始化、cache、ledger 和 Memory，不继承 Canary 团队。

宽松解析器 MATH_FLEXIBLE_ANSWER_EXTRACTION_V3 保留；采样、Solver 3600/仅截断6144、四次语义上限、Optimizer 生成容量及 Full/Shadow 阈值不变。Validation/Test 禁止，不推送。

Canary→Pilot 只检查合法完整终态、source/scope/receipt/ledger/access 和恢复边界，不按准确率或 Commit 推进。工程/完整性失败阻止 Pilot；预算终止不得当作科学收敛或重跑理由。

Pilot 的 60 道 Optimize 中 48 道未用、12 道按元数据最少补用；两个预冻结面板重叠 12 道 Optimize、1 道 Shadow。选择未使用成绩，两项 provider realization 独立。

授权依据：任务附件第 7.3 节要求冻结后对两个精确 Scope 一次授权；AGENTS.md 第 8 节要求授权绑定具体源码、角色、阶段和预算。

机器可核对的 manifest、binding、startup、scope、source-closure、membership 和资源上界见 [dual_scope_authorization.json](dual_scope_authorization.json)。
