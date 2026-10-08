# 实现、版本与方法依据

| 文件职责 | 改动与必要性 |
|---|---|
| `search/optimization_evidence.py` | 纯确定性 policy、差分、coverage、轮换和 guard 原语；没有新 agent 或管理器 |
| `benchmarks/math_evidence_binding.py` / `math_domain_binding.py` | 新 flat binding 派生、fresh dependency/scope 校验与 HOLD 分派；旧冻结 binding 不改 |
| `math_gradient_pattern_binding.py` / `binary_runtime.py` | Optimize worked solution / metadata 进入 evaluation-only typed example；Full/Probe 输出 scope-bound覆盖后果 |
| `textual_gradients.py` | 单错题参考过程诊断、合法不确定结果、actionable-only clustering 与独立 role evidence |
| `selected_evidence.py` / `current_opportunity.py` | 种子轮换、三个不相交成员集与精确机会审计；同一现有机会图 |
| `current_layer1.py` / `bounded_layer1.py` | hypothesis / abstain envelope、生成后 validation、validation-first selection、实际编辑 lineage |
| `rolling_risk_memory.py` / `pilot_observation.py` | 复用现有 Memory 私有 storage/top-k/事务与 shared risk；记录 bootstrap、diff、各 scope 与 Full 拒绝；原文只在私有 journal |
| `current_policy.py` / `current_composition.py` / `orchestrator.py` | 一个 builder 下完整 V2.3 policy，初始 bootstrap 与同一部署事务；V2.2 absent-policy 路径保留 |
| `provider_runtime.py` / decoding / prediction / optimizer contracts | 新 binding 在相同 Solver/transport/optimizer生成契约下显式可识别；新 evidence policy 进入 request/cache identity |
| `governance/unified_execution.py` / `autonomous_math.py` / repository / manifest schema | 新 scope 和 manifest 一致性；合法 all-uncertain 不被误记为缺失 cluster；历史 schema 接受语义核对 |
| `governance/source_identity.py` / `.gitattributes` | 新规范进入 scientific source hash；本次报告与协议固定 LF，保留精确字节校验的可复核性 |
| `versions.py` / `current_contract.py` / design / registry / lineage / frontier | 明确科学修订、parent 与 HOLD，保留历史常量、manifest 和报告 |
| `tests/current/test_optimization_evidence_v23.py` | 实际合成请求、disjointness、负迁移、rotation、Full Memory、guard、source/split、严格 manifest与observer证据 |

主要新身份为 `unified_team_prompt_search_v2_3`、`MATH_OPTIMIZATION_EVIDENCE_BINDING_V1`、`OPTIMIZATION_EVIDENCE_POLICY_V1`、`REFERENCE_SOLUTION_GRADIENT_INPUT_V3` / prompt V5、`PATTERN_HYPOTHESIS_EDIT_EFFECT_INPUT_V6`、`INDEPENDENT_OPTIMIZE_VALIDATION_SEARCH_V1`、`DISJOINT_ROTATING_OPTIMIZE_EVIDENCE_V1`、`BOOTSTRAPPED_EDIT_EFFECT_ROLLING_MEMORY_V1`。它们改变 optimizer 信息、生成动作、采样、排名与记忆读写，因此不能沿用 V2.2 科学身份。V6 Solver interface、普通轨迹 projection、已有 scoring / recovery、Gradient语义 cluster 原语和 V3 transition 身份保留。

预算没有靠增加 generation 弥补问题：42 是 parent 与六个 child 的两个三题 role 的明确总数；新 Probe 12、Full 120、Shadow ceiling不变。新的 bind/resource proof 同时计入 Solver semantic multiplier、一次独立 Gradient/wrong、reflection和transport。真实小对照需依据实际五机会 horizon再冻结预算；当前离线 profile 不含可执行 ceiling 或授权。

方法参考了 [OptiMAS 的假设证据记忆](https://arxiv.org/html/2608.21918v1) 和 [SEPO 的 edit-effect lineage](https://arxiv.org/html/2608.28067v1)：采用观察、预期、实际干预与 measured fixed/broken 关联的原则。未复制 OptiMAS 的 ReAct 控制器、规划工具或架构演化，也未实现 SEPO 的 typed prompt schema、第二个 attribution agent、Lexicase archive 或训练过滤策略。

检查了论文链接的 [OptiMAS 官方仓库](https://github.com/Opti-MAS/OptiMAS)，当时只有 README，没有可审查实现；SEPO 原论文的算法和附录是本次可核对的方法来源。因此这里是受方法原则启发的现有架构修订，不是复现它们的实现或成绩。论文的实验收益不作为本仓库的有效性证据。
