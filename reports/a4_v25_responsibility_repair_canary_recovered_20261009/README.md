# V2.5 Canary：恢复完成与诊断报告

**Canary 已恢复完成，判定 `CANARY_VALID_WITH_TERMINAL_RECOVERY`。** 唯一机会按冻结生产图到达 `CANARY_ONE_PRODUCTION_OPPORTUNITY_COMPLETE`，正式团队更新 **0** 次，最终 Optimize12 的 Vote / Oracle 均为 **5/12**，与初始相同。恢复新增 API **0** 次、计费 **0 tokens**；未运行 Pilot、Validation 或 Test。

## 恢复性质与完整性

原 runner 已完成所有科学测量和 Shadow 转移，但进程在负责人审阅及终态落盘阶段消失。原目录仍保留 `EXECUTION_ABORTED`，原[中断报告](../a4_v25_responsibility_repair_canary_interrupted_20261009/README.md)和其结论未改写。用户随后明确授权续跑完成并推送。

本次在无凭据、启动前断网保护下，使用同一个冻结 attempt 的 **336 条密封 Solver cache** 和 **24 条原始优化器回执**，通过未改动的生产 composition 重建原运行。24 个优化器请求逐字匹配；初始状态、机会、评估的科学字段、Shadow 转移和六代局部 lineage 一致。唯一三处差异是局部搜索的 Solver 调用/费用计数：缓存重放产生零新调用，原账本仍是实际费用依据。

恢复在单独私有目录中生成终态、团队、Memory、停止结果和恢复回执。原 **1,428** 个证据文件逐字节一致；**1,067** 个成功/错误回执、**2,135** 条账本事件及哈希链重新通过审计。模型、seed、分组、阈值、预算、缓存和科学方法均未变动，也没有新增机会或新 attempt。原始单次 API scope 保持已消费并关闭。

冻结执行源码与 source commit 一致，scientific、benchmark、dataset 范围一致；三处治理元数据变化来自已发布的中断记录。冻结治理以原 source commit 核验，没有放宽新运行 preflight。六项负控验证缺失 cache、cache 写入、请求不匹配、回执耗尽、账本篡改和 cache 篡改均会拒绝。进程消失的根因仍未查明，本次不宣称监督缺陷已修复。

## 初始画像与格式成本

五成员正确数为 **3、3、4、4、3 / 12**，最终相同；初始/最终 Vote 和 Oracle 均为 **5/12**。60 条逻辑画像产生 **192** 条成功 Solver 响应和 **2** 次超时。终态有效 **17/60**，格式无效 **43/60**，重复格式失败 **35** 条。17 条有效结果均数学正确；不能从格式失败推断潜在数学正确率。

首次有效 15 条，第二/第三次恢复各 1 条，四次仍失败 43 条。初始额外格式生成 **132** 次，provider 报告消耗 **156,768 tokens**。数学错误与格式错误在 Memory 和评分中保持区别。旧 V2.4 的 195 条响应只进行离线解析诊断，没有进入本次新初始评分、Memory 或 cache。

## 唯一机会：Gradient、搜索与 Probe

五成员均符合修复资格，D/N/C 均为 **7/0/0**、责任值 **28**，选中 member 0。真实全零回退没有出现。诊断该成员 9 条错误，Gradient 实际 **17** 次：**5 ACTIONABLE、0 UNCERTAIN、4 最终耗尽**；12 条结构失败 draw 没有被伪称为合法 UNCERTAIN。一轮聚类形成支持数 4、D/N/C 为 3/0/0 的 Pattern。

Mutation3、SearchValidation3、Independent Probe6 互不重叠；3 条分配错误位于 Mutation，已见重测不构成独立泛化证据。六代生成 **3 个已评估 Answer 编辑、2 个 CONTRACT_INVALID、1 个 NO_SAFE_EDIT**。实际局部 metric **24/42**。合法链长度为 1、2、3，各代 SearchValidation 净收益均为 −1，仍允许已评估临时父节点演化和外层导出。本次没有真实跨 Role/Strategy/Answer 复合编辑。

三个候选进入 Probe，一个因没有修复分配错误而淘汰；两个各修复 2/3 已见错误，独立 Vote Delta 均为 0，通过冻结风险条件晋级 Full。

| 候选 | 已见修复 | 独立 Vote / Target Delta | Full Vote / Target / Oracle Delta | Full 原正确丢失 |
|---|---:|---:|---:|---:|
| G2 | 2 / 3 | +0 / -1 | +1 / +0 / +1 | 2 |
| G5 | 2 / 3 | +0 / +0 | +2 / +1 / +2 | 1 |

两者通过 Full，赢家 G5 携带完整三步链进入 Shadow。Full 为 Optimize 训练内复核，不能把其收益写成独立评估收益；两者仍有 collateral loss，没有建立无损 additive repair。

## Shadow、终态与 Memory

单一赢家 Shadow40：团队 Vote **16→20**，目标成员正确数 **13→10**，格式无效 **26→27**。目标成员下降 3/40，冻结规则判 `catastrophic_target_member_regression`，因此 **REJECT**，没有团队更新。正式最终团队与初始 prompt 字节一致，Optimize Vote/Oracle 保持 **5/12**。这是保护规则的实际拒绝，不是 transition 实现错误的证据。

终态 Memory 通过原生产图重建并保存：初始 Competence 5 条、私有编辑 3 条（`FULL_SUPPORTED` 2、`INCONCLUSIVE` 1）、共享风险 1 条、stateful writes 4、Memory LLM 调用 0。`FULL_SUPPORTED` 表示 Full 阶段证据，不能理解为已获 Shadow 通过或正式部署。一次机会未观察到后续检索收益，不能证明 Memory 的因果作用。

## 费用、验证与结论范围

原 provider 报告实际用量 **1,130,364 tokens**，超时未知用量保守占额 **40,987**，账本 charged **1,171,351**，预约 **0**。保守占额不等于已知实际账单；2M 授权额度剩余 **828,649**，scope 关闭后不构成继续调用许可。全部 transport attempts **1,067**，其中 Solver **1,043**（1,039 成功、4 超时），Gradient 17、Cluster 1、Mutation 6。最高并发预约 8，最大 charged+reserved **1,207,827**，未触发预算停止。

恢复后完整运行 Current Suite：首轮 **379 passed、11 failed、2 skipped**。11 个失败均由上次中断发布把 `initial_accuracy` 写成非契约枚举导致；已改回 `MEASURED_FROM_CURRENT_CANARY`，科学代码与冻结参数没有改动。修正后完整治理模块 **53 passed**；逐 case 合并验证覆盖全部 **392** 个 current 用例，最终 **390 passed、2 skipped、0 failed**。未重跑不受元数据修改影响的科学测试；首轮结果及完整复测身份见 `tests_results.json`。220 个历史/私有模块明确排除，未声称历史 replay 全通过。断网恢复、账本/终态复核和六项负控均通过；compileall、当前治理、manifest、脱敏、字节 hash 及 diff 检查的最终结果记录在同一验证文件。原源码的 fake provider 最终入口验证仍保持有效。

本次观察到 V2.5 能从低初始正确率进入真实修复搜索、保留部分 Gradient、继续非正局部父节点，并产生两个 Full positive；**正式团队没有提升**。核心观察是训练内收益与 Shadow 目标成员保留发生冲突。单次 Optimize12 和自适应 Shadow40 不能建立泛化、优化器因果效应、收敛或完整 Pilot 有效性。

source `7217081d43fff2e94ab46ed28378717e3e912520`；attempt `a4_v25_seed81_canary_attempt1`；startup `4cdba0f516bb33c1b8a4264edab191110a8cd2a2b60fdfcaa93f5dd3cb6bd775`。公开文件只包含 hash、计数、类别、指标和审计解释；题目、答案、prompt、Solver 轨迹和终态 Memory 原文保持私有。
