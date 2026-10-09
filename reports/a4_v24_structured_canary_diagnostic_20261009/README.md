# V2.4 Canary 诊断：运行有效，优化链路未进入

Canary 已完整执行并经负责人复核，判定 **CANARY_VALID**。五成员均为 **3/12（25%）**，Team Vote 与 Oracle 均为 **3/12**。停止原因为 `NO_FEASIBLE_OPPORTUNITY`：每个成员只有 3 道正确题，低于冻结 panel 所需的 6 道。优化机会、Gradient、Pattern、mutation、Full、Shadow、commit 均为 0。

这支持对初始输出接口和机会可行性的诊断；**没有测得 Structured System Prompt 优化效果，也不能判为 optimizer 失败或收敛。** Pilot 已按用户指令在初始化阶段停止，未完成 Optimize60 初始画像。本报告以 Canary 为主要诊断证据，Pilot 仅记录停止和记账情况。

## Canary 的真实观察

| 项目 | 结果 |
|---|---:|
| 固定 Optimize 题数 / 成员数 | 12 / 5 |
| 逻辑画像 / 物理 Solver 调用 | 60 / 195 |
| 五成员正确数 | 3、3、3、3、3 |
| 初始 → 最终 Vote / Oracle | 3 → 3 / 3 → 3 |
| 可解析画像 | 15/60（25%） |
| 四次后仍无效 | 45/60（75%） |
| 额外格式恢复 / 成功恢复 | 135 / 0 |
| 容量截断 / 传输失败 | 0 / 0 |
| 实际计费 | 137,572 tokens |
| 未结预约 / Validation / Test | 0 / 0 / 0 |

负责人检查了 22 种不同的终态普通响应，覆盖全部 60 条画像；均有可见的数学步骤。可见推导比例为 60/60，但这只是普通内容中的步骤存在性，不是数学正确性或隐藏 reasoning 证明。运行中的 `WRITTEN_SOLUTION_PRESENT=15` 与 `FINAL_BOUNDARY_INVALID=45` 是边界处理状态，不能用前者解释为只有 15 条存在数学推导。成员的正确集合完全相同，交集及并集均为 3；响应文本则有变化，不能从相同 prompt 推断相同随机输出。

## 格式覆盖限制与恢复成本

195 条原始响应中，15 条有效，156 条为 `MALFORMED_FINAL_BOUNDARY`，20 条为 `CONFLICTING_FINAL_ANSWERS`，4 条缺失可接受的终态标记。60 条终态画像中，40 条边界异常、5 条冲突、15 条有效。

真实响应出现普通步骤标题、嵌套最终标签及字面答案标签。冻结 parser 把每个三级标题都当作显式声明；嵌套最终标签可直接触发 malformed，多个声明必须在有限归一化后文本一致，并且必须存在终态声明。它也会保守拒绝某些数学上等价而文本不同的重复表达。原始文本特征计数见 [格式取证](format_forensic.json)；这些特征会重叠，不能相加作失败总数。

| 语义 draw | 调用数 | 有效数 | 计费 tokens |
|---|---:|---:|---:|
| 1 | 60 | 15 | 37,309 |
| 2 | 45 | 0 | 34,022 |
| 3 | 45 | 0 | 33,356 |
| 4 | 45 | 0 | 32,885 |

额外恢复共花费 **100,263 tokens，占 Canary 计费的 72.88%**；恢复请求保持冻结的消息及采样，没有注入新的格式修复指令。每条无效画像完成四次 draw 后记 0，固定分母保留。全部响应以 stop 结束，因此 Canary 未触发 6144 容量恢复分支。没有修改 parser，也没有计算替代 parser 的准确率；无法据此声称模型潜在数学正确率是多少。

## 为什么没有机会

五成员各有 3 道正确、9 道错误，满足至少 4 道错误的条件，却不满足至少 6 道正确的条件。该条件为冻结的独立 Mutation / SearchValidation / TeamProbe 与 preservation 证据设计的一部分。调度在生成机会之前终止，所以没有尝试诊断这 45 个错误的成员—题目对，也没有优化 Answer 区块来修复表达。

| Funnel | 实际数量 / 状态 |
|---|---|
| 错误成员—题目对 / 不同错误题 | 45 / 9 |
| 可行成员 / 机会 | 0 / 0 |
| Gradient / Pattern | 0 / NOT_OBSERVED |
| mutation / 合法 edits / 候选评估 | 0 / NOT_OBSERVED |
| SearchValidation / Probe / Full / Shadow | 0 / NOT_OBSERVED |
| commit | 0 |

直接机械阻断点是 **opportunity feasibility**。本次没有观测到 budget、local export、Full preservation、team admission 或 optimizer 模型质量成为流失阶段；也没有“Gradient 全为 UNCERTAIN”或“候选全部失败”的证据。Role、Strategy、Answer 编辑次数均为 0，合法编辑率没有分母。初始和最终状态、完整 prompt hashes 均相同，没有成员能力丢失或团队变化。详见 [可行性诊断](feasibility_diagnosis.json)、[Funnel](funnel_summary.json) 与 [区块演化](prompt_block_evolution.json)。

Memory 从本轮真实画像建立了 5 条初始 competence；没有 edit、success、failure 或 shared 写入，没有后续机会读取，也没有 committed competence 更新。因此本次不能评价 Memory 的优化贡献。详见 [Memory 摘要](memory_summary.json)。

## 有效性、覆盖与限制

冻结执行源码为 `1f74d4df2dc2591f58b31807264a1505370035fe`，准备报告提交为 `1cad9f1a2584621fd031b3d49c513e8fe9e24c43`。两者不同是既有冻结设计；源码闭包、manifest、binding、startup、数据和 scope 均在真实执行前验证。独立审计重建全部 60 个 cache/request identity，核对精确三块 System、原始 User、非 thinking 请求和每次语义 draw，使用冻结原生数学等价评分与 plurality 复算，结果与运行完全一致。

195 份响应回执、账本 hash chain、预约/计费以及 271 个原始 inventory 文件均通过验证。实测最大并发为 8，最大 charged+reserved 暴露为 188,477 tokens，小于独立 2M 上限。负责人完成真实初始证据复核；第一完整机会复核因没有机会而 NOT_OBSERVED，没有伪造或强制生成。

零 API 专项审计及当前分类 suite 为 **368 passed、2 skipped**（370 selected），明确排除 **220 个历史、退役或私有测试模块**；未宣称全部历史 replay 通过。compileall 和执行前治理检查通过，network guard 记录 0 个连接尝试。Solver 六阶段消息接口、混合 Gradient、单区块及多代 edit chain、Memory 和旧方法隔离的离线检查只证明接口/约束，不构成真实优化收益。未修改科学实现、阈值、seed、parser、停止规则或冻结 manifest。

**OBSERVED FACT：** 格式提取无效率高、恢复无成功、可行性条件未满足、优化未进入。

**SUPPORTED HYPOTHESIS：** 当前初始输出习惯与保守边界契约的适配，是本次 measured competence 和优化可达性的显著限制。

**NOT ESTABLISHED：** 排除格式因素后的数学准确率、真实 Gradient/Pattern 质量、mutation/Memory 效果、团队收益、泛化或 optimizer 因果结论。

## Pilot 停止与授权关闭

Pilot 采用独立新 scope，未导入 Canary 或 V2.3 画像、缓存或 Memory。授权文本的 manifest 摘要少两位，原文和差异保留；完整 execution、startup、binding、task、attempt、source 均匹配既有冻结，按原任务明确允许的授权交接修复范围处理。没有收到或声称收到额外人类确认，没有修改任何冻结文件或扩大 scope。详见 [单次授权消费](authorization_consumption_summary.json)。

用户要求停止时，Pilot 已返回 120 次 Solver 响应，完成 36/300 条初始化画像，尚未完成 baseline 或进入优化。对应进程已停止，原始运行 lifecycle 的 RUNNING 字段原样保留，另有真实 owner stop receipt 记录终止，未伪造 runtime 完成状态。已返回调用实际费用 **95,336 tokens**；7 个未返回预约按既有 crash-recovery 规则保守计费 **60,438 tokens**。Pilot 总记账 **155,774**，保守部分不是已知 provider 实际账单，未结预约为 0。

两 scope 合计已知实际费用 **232,908 tokens**；含未知请求保守记账合计 **293,346**。两份单次授权均已消费、关闭，不得恢复或重跑，也不得转移剩余额度。Validation/Test 均为 0。详见 [Pilot 停止记录](pilot_stop_disposition.json) 与 [计费摘要](api_ledger_summary.json)。

## 下一步诊断建议

优先做零 API 的初始化输出契约与边界提取适配审查，并检查低正确覆盖下独立 panel 是否仍允许 Answer repair 进入搜索。现有证据不支持直接更换 optimizer、增加 generations 或追加 Pilot 预算。若选择改 seed、parser 或 feasibility，必须明确新的科学版本、对照设计、冻结身份与单次 API 授权；本报告没有实施这些变更，也没有启动后续实验。

公开文件只包含摘要、统计、类别、哈希及审计结论。完整题目、答案、prompt、响应、私有账本和中间证据留在私有运行目录。旧报告和无关未跟踪文件均保留。本报告是证据，不改变规范方法。

发布前当前治理、manifest/失败登记 preflight、脱敏、事实断言、compileall 与字节检查通过；科学冻结文件和两份原始 manifest 保持原字节。详见 [发布检查](publication_checks.json)。
