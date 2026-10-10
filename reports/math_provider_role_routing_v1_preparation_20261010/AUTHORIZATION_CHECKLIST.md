# Canary 与 Pilot 精确授权清单

冻结源码：`01306fbf9cde002291ab21e2f93a2c02399f7e9e`。两个范围均未执行，等待一次性精确授权。

| 项目 | Canary | Pilot |
|---|---|---|
| Attempt | a4_v25_arm_b_openlux_seed84_canary_attempt1 | a4_v25_arm_b_openlux_seed84_pilot_attempt1 |
| Seed | 84 | 84 |
| Optimize / Shadow | 12 / 40 | 60 / 40 |
| 最多机会 | 1 | 5 |
| charged + reserved Token 上限 | 2,000,000 | 2,000,000 |
| Scope SHA256 | bde46ead1b30b60c63b311781f13e4a513f9fd9e864a60de34da59672af13e22 | 53280eb4ff6f785a2be6163bc53d6a81c254e10b591b93b9d0df7b85856d5122 |
| Startup SHA256 | 4e99f0502d0c59f591e2b088778f5cf1104a7ca02b9f05f8c58be0d54fe8cf39 | 2529d5689f51ce8fc8db99c8693f2e7b47de5e457671904d79ba47db066544d8 |

Solver：OpenLux gpt-4o-mini。Reflection、Gradient、Pattern：原 lwj qwen3.7-flash。并发上限 8 / 1。

每阶段只执行一个新 Attempt，合计上限 4,000,000 Token；无重跑、无额外探测调用。两阶段从五个相同 Arm B Prompt 分别初始化，成员响应独立，Memory、缓存、账本和授权均独立。

Canary 工程与完整性审计通过后自动进入 Pilot，不以准确率或 Commit 作为推进条件。合法停止照实记录；工程或完整性失败阻断 Pilot。Validation / Test 禁止访问，不 Push。

Pilot Optimize 的历史使用补入数量为 20；两个预冻结面板重叠 12 道 Optimize、1 道 Shadow。选择只使用元数据，不使用成绩。

授权依据：原任务第 7.3 节要求冻结后对两个精确 Scope 一次性授权；AGENTS.md 第 8 节要求绑定源码、角色、阶段与预算。

完整 manifest、binding、source closure、startup 与 scope 哈希见 [dual_scope_authorization.json](dual_scope_authorization.json)。
