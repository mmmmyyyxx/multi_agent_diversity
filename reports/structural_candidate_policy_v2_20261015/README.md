# Prompt 候选结构契约 V2：实现与离线对照

本次覆盖当前候选 Guard，不提供旧语义 Guard 的运行开关。新身份为
`STRUCTURAL_SINGLE_BLOCK_CANDIDATE_CONTRACT_V2`；当前证据策略为 V15，独立消融草案为 schema V5 / kernel V5。
没有启动真实实验，没有调用模型 API，也没有重新执行历史 Solver、Validation 或 Test。

## 新旧规则

| 项目 | 历史候选规则 | 新规则 |
|---|---|---|
| Prompt 结构 | Role / Strategy / Answer | 保留，三块均为非空字符串 |
| 修改范围 | 单块替换 | 保留；其他两块字节不变 |
| 总长度、块预算 | 3000 字符及派生预算 | 移除，无候选字符上限 |
| 未修改 | `parent_no_op` 拒绝并重试 | 记录 `NO_OP`，结束生成，跳过候选测评 |
| 答案查表、固定答案、样例复制 | 文本规则拒绝 | 移除，由后续测评反映效果 |
| 优化器依赖、八词重合 | 关键词 / 重合规则拒绝 | 移除；推理隔离由实际输入渠道保证 |
| 重复候选、同伴 Prompt 复制 | 部分团队路径拒绝 | 移除文本重复拒绝；原请求缓存保持 |
| 输出格式 | 严格 JSON 外壳 | 保留类型、单块和完整性检查；恢复完整围栏或唯一 JSON 对象外壳 |
| 性能选择 | 原实验测评和准入规则 | 不变；单 Prompt 仍 20 机会、Validation100 至少多对 1 题 |

重复键、多对象歧义、非字符串、空块、未知块、多块编辑以及明确截断仍为格式错误。
数学正文和无效 JSON 转义不会被猜测重写。no-op 消耗普通机会/槽位，不强制用完剩余合法性重试；
若本轮先有格式失败再产生 no-op，也立即跳过。结构合法不代表性能合格。

必要的 completion-token 容量、输出截断标记、模型请求上下文可执行性、授权和资源记账仍保留。
既有请求容量检查预留输出并保守估算序列化请求；没有新增字符上限，也没有修改模型、SDK、输出容量或重试次数。

## 493 次历史候选对照

覆盖原只读审计的 **493 个唯一生成响应**，分属 7 个响应记录中的冻结源码版本。
逐条核对原响应回执封印、响应哈希、请求中父 Prompt 和原审计索引，使用新解析器重新判定。
原历史版本的判定复现证据仍见[历史审计](../historical_candidate_guard_audit_20261015/README.md)，没有用新代码冒充历史实现。
具体路径、源码、轮次、draw、原拒绝原因和新候选哈希见 [candidate_comparison.json](candidate_comparison.json)。
不发布候选正文或原始响应。

| 原判定 → 新处置 | 次数 |
|---|---:|
| 原通过 → 可进入测评 | 332 |
| 原拒绝 → 可进入测评 | 40 |
| 原 no-op 拒绝 → 记录并跳过 | 92 |
| 原格式拒绝 → 仍拒绝 | 29 |
| 合计 | 493 |

新处置合计：**372 可测评、92 no-op、29 格式拒绝**。
新增可测评的 40 次包含 **23 次长度拒绝、14 次八词重合拒绝、3 次样例复制拒绝**。
其中原审计认定的 **10 次确定误拒全部可测评**；其余被释放的候选不因此被追认为确定误拒。
`example_answer_lookup` 和 `optimizer_only_dependency` 在这 493 次中没有实际触发，相关放行由定向离线测试覆盖。

全部历史记录按组汇总如下；分母为生成 draw，混合了不同历史协议，不作为 A/B/C 效果比较。

| 组 | 记录数 | 新可测评 | 新 no-op | 新格式拒绝 |
|---|---:|---:|---:|---:|
| A | 151 | 89 | 62 | 0 |
| B | 136 | 100 | 16 | 20 |
| C | 206 | 183 | 14 | 9 |

此前主要三重复对照的子集（A/B 源码 `241f7257`；修正 C 源码 `211a05c5`）为：

| 组 | draw 分母 | 原通过 | 新可测评 | 新增可测评 | no-op | 格式拒绝 |
|---|---:|---:|---:|---:|---:|---:|
| A | 65 | 60 | 60 | 0 | 5 | 0 |
| B | 87 | 58 | 60 | 2 | 12 | 15 |
| C | 77 | 56 | 71 | 15 | 3 | 3 |

这是**逐条已记录响应的离线重分类**。新机制会在较早的 no-op 处结束某些旧重试链，
所以这些数量不是新机制真实运行将产生的候选数量。10 次误拒涉及 C 的 4 个历史耗尽机会，
只能证明它们的候选曾被挡在测评之外，不能证明这些候选能提高 Validation。

## 调用路径与共享边界

| 路径 | 本次处理 |
|---|---|
| `search/system_prompt.py`、`candidate_contract.py` | 唯一结构约束和统一候选解析器 |
| `search/optimization_evidence.py` | 候选可执行性不再作关键词/重合审查 |
| `search/reflection_steps.py`、`current_layer1.py`、`population_evaluation.py` | 团队生成、后置检查、no-op、跨槽位重复均对齐；槽位/来源/父状态完整性检查保留 |
| `search/reflector_view.py`、`optimizer_output_contract.py` | 移除块预算、摘要任意长度限制和语义重试指令 |
| `research/ablations/generation.py`、`engine.py`、`governance.py`、`analysis.py` | 独立模块生成、跳过测评、封存审计和统计均识别 no-op |
| `research/gated_prompt.py`、`single_prompt.py` | 共享结构解析与 no-op 跳过；历史执行仍必须恢复原冻结 Git 源码 |
| `benchmarks/math_structured_interface.py`、`math_optimizer_diagnostics.py` | 不再隐藏字符上限或重做内容审查 |
| `search/group_diagnosis.py`、`diagnostic_source_checks.py` | Group 诊断自身的原语义边界保持，独立于候选路径；候选从不调用此诊断检查 |
| Gradient / Pattern / Memory | 诊断分区、支持集、抽样、存储/交付语义未改 |
| 历史 flat-text GEPA / TCS 适配器和审计工具 | 不属于当前三块 Prompt 图，未作为旧候选 Guard 模式接入；历史重放使用原 Git 源码 |

实际 Solver 输入仍只有渲染后的 Prompt 和原始题目，没有新增私有对象、答案或团队渠道。
Optimizer 可以从训练内容提炼或记忆文本；这种文本现在不会被预先判违规。
这不赋予 Solver 读取优化器私有对象的能力。数据阶段隔离、缓存、严格 Validation 门槛、团队
Full/Shadow/Vote 准入和父代选择不变，没有加入 GEPA Pareto 算法。

## 完整性与可比性

[historical_integrity.json](historical_integrity.json) 记录原审计证据哈希和 **1548 个历史输入文件全部未变**的核对结果。
历史 manifest、授权、日志、候选、评分和已完成实验结论没有重写。仓库治理现在按发布前 Git 锚点
`b404deb3e0a440364d7c1835d21a13046968e370` 核对历史 single-Prompt manifest 的精确内容，
而不以修改后的工厂函数重新生成它们；此只读元数据校验不提供旧 Guard 或执行许可。

正式 manifest schema 追加新契约分支，原有分支逐项保留；当前准备指针转向新的
`structural_candidate_contract_v2_preparation_v1.yaml` DRAFT。原准备 manifest 没有修改。
该草案的源码、数据身份和哈希闭包仍未冻结，API 授权为 false，不具备执行资格。

原 A/B/C 数字仍是旧规则下的观测结果。旧 C 确实受到内容误拒的更明显影响，不能由本次离线对照
校正它的准确率或推断新规则下 Gradient–Pattern 的收益。未来应在同一新契约下重新冻结匹配的
A/B/C，才能比较模块效果；新旧结果须分版本报告。本次没有自动准备授权或启动该实验。

## 验证证据

见 [verification.json](verification.json)：生产包编译、完整 current 分类测试、治理与草案 preflight、
493 响应对照、无网络假 Provider 检查、证据完整性和发布脱敏。历史/私有资产测试明确排除，
不宣称完整历史重放通过。提交 SHA 由包含本报告的 Git 提交确定，避免文件内自引用。

最终分类覆盖 **1110 项：1108 通过、2 跳过**；跳过为用户推迟下载的 Hover/PUPA 数据身份检查。
完整分片初轮的 40 项失败/错误（旧断言、准备身份及并行负载下评分启动超时）已全部串行复测通过，
最后 50 项候选/团队快照/科学身份定向检查也通过。统计合并每项最新结果，不把失败轮次算作通过。
224 个历史/私有模块和 5 个已退休 Git 重放用例按分类清单排除。
新规范文档进入科学源码哈希闭包；19 个原有 manifest schema 分支原样保留。
最终生产绑定哈希下的九条 × 20 轮假 Provider smoke 和 preflight 均通过，真实模型调用为零。

参考 GEPA 官方[提案解析](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/instruction_proposal.py)
及[反思修改](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/reflective_mutation.py)
（查阅 2026-10-15）。GEPA 的文本提案恢复与空更新跳过提供参考；本项目仍保留自身三块单块协议，
字节一致 no-op 的处置来自本次明确要求，不宣称完整复制 GEPA。
