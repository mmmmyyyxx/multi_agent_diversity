# A4 V2.3：优化证据、独立验证与编辑记忆修订

**2026-10-08 按用户指令停止并发布。** 运行中的最终完整回归已终止，没有启动真实模型实验。新修订专项测试 **22 passed**；开发阶段完整 current suite **1,657 passed、2 skipped、1,468 deselected**，但该结果不能替代最终代码的完整回归。最终发布验证状态为 **PARTIALLY_VERIFIED_USER_STOP**。实验、实现、停止和预算汇总见 [stop_summary.md](stop_summary.md)。

已实现一项明确版本化的科学方法修订：Optimize worked solution 进入独立 Gradient，Mutation / SearchValidation / TeamProbe 成员彼此分离，preservation 从当前已提交正确集合轮换，Memory 从初始实测覆盖启动，并记录实际 prompt diff 与局部、Probe、Full 的 fixed / broken / retained 后果。五成员、单成员更新、固定 peers、等权 plurality 和 V3 Full transition 保持不变。

本报告只证明离线接口与合成执行符合新契约，**没有真实模型有效性或泛化结论**。本轮 API 计费、真实 Shadow / Validation / Test 调用均为 0；旧授权没有重新开启。下一轮真实比较为 **HOLD / READY_TO_RUN=false**，仍需新 source、binding、manifest、baseline、预算和单次授权冻结。

实现起点是 `3de088e42c0427a3c4b6a236331348eba0bb2f1a`。七机会旧 Pilot 的执行源是 `6731aa3f345eff7feac12140b4892f4e07291d17`；它的 42 proposals、零 commit 和两次 Full 损失来自原冻结证据。旧 Pilot 没有可见数学轨迹，当前起点已有 V6 轨迹接口，V2.3 则增加了下面的新证据学习行为。三者不可混为同一次执行。

完整规范见 [OPTIMIZATION_EVIDENCE_V23](../../docs/design/OPTIMIZATION_EVIDENCE_V23.md)，逐字段审计见 [information_flow_audit.md](information_flow_audit.md)，版本与文件职责见 [implementation.md](implementation.md)。

合成 provider 的实际序列化请求经过检查：Solver 收到 standalone procedure、public problem 和 V6 interface；独立 Gradient 收到匹配成员/例子的真实普通响应轨迹及单独标记的 worked solution；mutation 只收到三条 Mutation 观察、一个 Pattern 假设和有界私有 Memory。SearchValidation、TeamProbe 的题面/答案没有进入生成前的 mutation 请求。请求原文和完整 diff 留在 ignored 私有证据；公开 [wire trace](synthetic_wire_trace.json) 给出字段、成员、哈希与计数。

```text
旧执行：procedure → short answer → Gradient hypothesis → same-panel ranking
新修订：procedure → visible solution + final payload
                 → separate reference-grounded Gradient
                 → focused hypothesis edit
                 → disjoint SearchValidation → fixed-peer Probe → Full
                 → measured edit Memory → next same-member mutation
```

机会在 Gradient 调用前检查是否至少有 4 条当前错误、6 条当前正确证据，保证 singleton Pattern 也有足够不相交的 role 容量。不足时返回 `NO_FEASIBLE_OPPORTUNITY`，不会先花诊断调用再因 role 构造不足中断。这一技术限制已进入新 policy；不是 generated candidate 的局部准入 veto。

一个故意设定的坏编辑在合成 SearchValidation 上为 **+1**，Full 却是 **新增 4、丢失 7、净 -3**，被 V3 transition 拒绝并记录为 `FULL_REFUTED`。另一个编辑保留全部 12 个初始正确例子并新增 4 个，提交为 `COMMITTED`。坏编辑的生成摘要声称添加 substitution，实际 diff 添加的是 case/sign 检查；Memory 使用实际 diff。见 [完整脱敏合成记录](synthetic_memory_records.json)。这些数字是测试夹具的预设行为，不是 qwen 实测成绩。

后续同成员合成搜索读到了 Full 拒绝的收益和损失。fake mutation provider 依据这一输入返回 `NO_SAFE_EDIT`，验证的是证据接通；不能由此推断真实 optimizer 一定学会规避同类编辑。另一项合成测试显示 Mutation positive、SearchValidation negative 可以被独立看见，validation 排名不会再被 mutation 收益遮蔽。

局部预算是 `父 prompt (3+3) + 六个 child (3+3) = 42` 次逻辑 Solver 评估；六次生成、四个 exports、两个 Full 名额没有增加。Probe 从四个候选的最多六题改为独立三题，即最多 12 次；Full 仍最多 120 次。新资源派生明确计算 role、Solver recovery 与 transport 乘数，旧预算/授权不继承执行资格。

下一步最小对照是 [两个完整方法单元](../../experiments/protocols/math_optimization_evidence_v23/next_comparison.json)：A 为带相同 V6 Solver 接口的 V2.2，B 为完整 V2.3，各五次机会、相同模型/解码/初始团队/Optimize60、种子 81。必须重新测量初始 baseline，不能复用旧 22/60。这是整体修订的比较，不能分别归因 reference、validation 或 Memory。任何保护集访问和真正调用都需另行授权。

测试与字节/治理核验结果见 [verification.md](verification.md)。可见轨迹仍可能省略重要步骤，参考解可能有多种等价路径，小 validation 会有噪声且被后续迭代自适应使用；词法 guard 不是完整语义证明，复合编辑不能归因到单独条款。最关键的真实问题——是否能生成并保留 P2 式 additive repair——仍待新比较验证。
