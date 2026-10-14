# V2.11 Group-Diagnosed Reflection：离线实现与验证

已接入当前 production composition；只读使用 Root Team Optimize 证据，三个 Reflection Slot 共用一份团队经验。真实 API、Validation/Test 调用均为 0，未运行 Canary/Pilot/Formal，未推送。真实修复质量、Team Vote、Token 效率与泛化效果均为 **UNKNOWN**。

## 范围与来源

用户参考基线为 `d7de59818da609cff6b03c6ebb334fa79011aedd`。启动时 HEAD 为 `821bd56c`，期间独立单 Prompt 消融工作由其原有任务提交到 `86976661265e5dc63dc44c83f75d734dbc62d43a`；本任务保留其内容，最终基于这个实际状态整合。

新方法为 `unified_team_prompt_search_v2_11_group_diagnosed_reflection_v1`，新 binding 为 `MATH_GROUP_DIAGNOSED_REFLECTION_BINDING_V14`，新 composition/cache 为 V14，完整搜索 checkpoint 为 `V211_GROUP_COMPLETE_SEARCH_BOUNDARY_V2`。准备 manifest 为 DRAFT，source/hash/binding 尚未冻结，真实授权为 false。未消费或恢复历史授权、缓存或实验。

## 实际改动与入口

- `search/group_diagnosis.py`：验证五成员 Root Snapshot、逐题类别、等价答案组、现有反事实 Team Impact；生成四类可空经验，执行来源引用、格式、长度、复制与抽象检查。
- `search/reflector_view.py`：Canonical Evidence 与 LLM View 分离，保留完整响应，删除 Preview，只在完整已验证轨迹与正确性相同时合并 root/Parent；直接发送可读 Unicode JSON。
- `search/current_layer1.py`：冻结三个 Parent 后诊断一次；每个 Slot 独立读取实际 Parent Profile、Memory，接收共享 `group_experience`；保存 canonical/view/wire 长度、调用、Token 与生成结果。
- `search/current_composition.py`、`search/provider_runtime.py`：真实接入独立 `group_diagnosis` Role，复用 optimizer 模型和原供应商、Broker、Ledger、物理重试与并发约束。
- `current_contract.py`、`versions.py`、current policy、binding、单次授权 scope、manifest schema、checkpoint、frontier、规范与注册同步更新；历史 schema branches、实验 manifest 和报告不变。
- 独立 single-Prompt study/ablation 的已登记轨迹元数据固定在它们原有身份，避免因团队 Reflection 身份变化污染旧契约。其 Prompt、模型、评分、数据、运行行为与授权没有改动；这不是旧团队 runtime 兼容模式。

原版 WHO/责任权重、Individual Gradient 生成/Schema、Pattern 聚类评分、Support Sampling、Parent Population/概率/容量、Memory 检索与存储、Aggregation、Full/Shadow、Stopping 和数学评分核心已逐文件与 V2.10 对照保持字节一致。没有 Preservation、额外候选评分或额外 Solver 请求。

## Input / Output 契约

Group input：`ROOT_TEAM_GROUP_DIAGNOSIS_INPUT_V1`。包含目标成员、已选 Pattern/支持 IDs、Root Team 身份和聚合规则、五个完整 Prompt，以及 1–3 道 Optimize 支持题的题目、gold、完整参考解、五份完整实际响应、有效性/正确性、实际答案/等价组、Vote 和现有 target counterfactual。逐题正确、错误、无效集合互不混淆；同一成员可跨题交换类别。

Group output：`ROOT_TEAM_GROUP_EXPERIENCE_V1`。必须有 status 与 `wrong_peer_contrast`、`negative_experience`、`correct_peer_contrast`、`positive_experience`；四项独立可 null。非空项有 observation、hypothesis、modification_implication 与支持样本/member IDs。每个文本字段 ≤300 字符、单条 ≤900、总 compact JSON ≤6,000 字符。引用、peer 正误类别、重复成员/样本、特定内容与精确 Pattern 重复均受检查。自然语言事实忠实度、因果措辞与语义新颖性不能仅靠这些确定性检查证明，仍需真实输出审计。

四项全空为合法 `NO_SUPPORTED_INSIGHT`；两次格式尝试耗尽为独立 `GENERATION_FAILED`，继续 Reflection 时明确标记失败。Provider、完整性、来源、资源或持久化异常 fail closed，保留 canonical 与失败材料，不换模型或静默扩容。资源上限为配置的 100,000 输入 UTF-8 bytes、131,072 保守 context bound；输出 4,096 tokens，加原有 10-token 计量容差。超限不截断、不删除成员。

Reflection input：`GROUP_DIAGNOSED_READABLE_REFLECTION_INPUT_V6`；instruction 为 `GROUP_DIAGNOSED_REFLECTION_INSTRUCTION_V1`。保留 selected Parent 全文、Pattern、问题/gold、原有显式 4,096 字符参考解视图、完整 root/Parent 响应、Memory 和 Edit Capacity，加共享 Root Team 经验。

原始五字段诊断仍在 canonical。LLM view 保留 observed_failure、diagnosis、reusable_correction；observed_failure 的独特价值无法机械判定，故保守保留。代码与实际序列化请求确认这些诊断是 Parent 抽样前建立的共享 Root 诊断，不是 Parent-specific 新观察；suggested_block/expected_effect 从默认 View 移除，Root/Parent 来源依然明确。其增量效用仍需真实实验检验，不能声称语义删除必然无损。

## 合成案例与实际发送文本

测试覆盖 1/4、2/3、3/2、4/1 正误分布，以及同为两人正确时 wrong classes 不同导致 Vote 输、平、赢的情形。正确人数不替代反事实 Vote Flip。合成实例五个相同 Prompt 的错误成员提供一条 wrong-peer 对照，其余三项为空；Fake Provider 实际生成并送达三个 Slot 的经验见 [synthetic_experience.json](synthetic_experience.json)。该实例没有真实模型或科研效果含义。

下表比较**同一 canonical records 的 V2.10 wire template 与 V2.11 实际发送文本**，不是旧模型重跑。新长度已包含经验与新增指令。

| 合成场景 | 每 Slot 旧输入字符 | 每 Slot 新输入字符 | 减少 |
|---|---:|---:|---:|
| 三题短响应，root/Parent 同一次执行 | 13,004 | 5,916 | 54.51% |
| 一题长 Unicode 响应，root/Parent 同一次执行 | 40,746 | 12,044 | 70.44% |

短场景另有一次 Group 输入 9,712 字符；不能只报告 Reflection 下降而忽略额外诊断成本。完整统计在 [input_statistics.json](input_statistics.json)，包括 Preview 删除、合并次数、实际响应保留量及各次 wire 长度。Token 估算用字符/4，仅为示意，**不是 Qwen tokenizer 或实际计费**。真实 provider usage 为 null；Fake Provider 的输入/输出数是测试桩值，不能外推费用。

## 恢复、计费与安全

正常一次 Group 输出，最多两次仅格式恢复，三个 Slot 共用最终经验，不受 Group 结果改变抽样。Root Evidence 在诊断前持久化；各次输入、原始输出/拒绝、接受结果和 fatal failure 私有保存。底层 response receipts 与 Token Ledger 不绕过。

资源证明新增最多 `2 × opportunities` Group response draws，并应用原 transport multiplier；不增加 Solver、Gradient 或 Reflection generation 上限。Broker 分角色计量记录所有已收到的响应，包括格式拒绝和截断，补齐旧 Reflection wrapper 仅计正常返回的遗漏。运输失败的保守费用仍由 durable Ledger/receipts 记载；provider 未提供可靠 usage 时，accounted tokens 不等于真实报告 usage。

Group 不读取 Memory/Shadow/Validation/Test；Reflection 的原有 Memory 科学范围不变。原 guard 保留，并检查 peer 响应复制、完整 peer Prompt 复制及推理时不存在的 peer/group 上下文依赖。没有新增 Preservation 或放宽 admission。

## 验证与研究边界

当前分类测试累计最终覆盖 **898 cases：896 passed、2 skipped、0 unresolved failed**。完整套件首轮 877 passed / 18 failed / 2 skipped；修复后按 case 合并受影响范围的补测证据，未声称又跑了一次单批全绿完整套件。详细分批结果见 [verification.json](verification.json)。224 个历史/私有模块及 5 个 retired Git-only cases 明确排除；没有历史 replay 全通过的声明。

Production compileall、current dependency closure、当前治理、历史 schema branch 不变校验、新 draft manifest、确定性假 provider 端到端、网络 guard、脱敏/算术/链接和 diff 检查通过。真实 preflight 预期 HOLD（未冻结、未授权），不是 ready-to-run。旧治理 CLI 的已知基线不作为当前生产 gate。

已证明新增角色、来源/类别约束、共享送达、直接可读投影、严格去重、预算与异常恢复在离线用例中正确运行。**未证明** Group Diagnosis 提供可靠新洞察、Reflector 采纳了正确经验、诊断字段删减无损、最终 Vote 或真实 Token 效率改善。下一步只能在新的源/协议/模型/预算/访问冻结及单次授权后进行成本受控验证，并分别检查 Group 输出忠实度、经验送达与实际编辑、完整评估和总费用；本任务不启动实验或做消融。

科学 source hash：`5b730c3948d6ced6740a8701bc0f76ca220d562d3efba441b790c029f71b815c`。报告是证据，不是方法规范；规范见 [V2.11 amendment](../../docs/design/GROUP_DIAGNOSED_REFLECTION_V211.md)。
