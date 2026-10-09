# V2.5 Canary 中断证据报告

这份报告记录一次新的 V2.5 Canary 在完成初始画像、唯一机会的搜索、Probe、Full 和 Shadow 计算后，于官方终态持久化之前发生的 runner 进程中断。独立审计已通过，但运行目录没有生成官方 `execution_summary.json`、`final_state_private.json`、`final_team_private.json` 或 `EXECUTION_COMPLETE` 生命周期标记。因此本报告判定为 **NOT_EVALUABLE_AS_COMPLETE_CANARY**，不把中间结果包装成完整 Canary 成果，也没有重跑已消费授权。

## 已观测事实

- 初始 60 个逻辑画像：成员正确数 **[3.0, 3.0, 4.0, 4.0, 3.0] / 12**，Vote/Oracle **5.0/5**。
- 初始画像使用 192 条成功 Solver 响应；可解析 17，终态无效 43，重复格式失败 35。
- 已完成 1 次机会的真实证据链：Gradient 17 次，Actionable 5，UNCERTAIN 0，最终耗尽 4；局部生成 6 代，合法候选 3，Probe 3，Full 2。
- Shadow gate 的独立重算为 **REJECT**：Vote 16→20，目标成员 13→10；因此轨迹中没有 commit。
- 账本 charged **1,171,351 tokens**，Provider 报告实际 **1,130,364**，超时保守计费 **40,987**，未结预留 **0**。Validation/Test 均为 0。

## 完整性判定

独立审计复核了冻结 source/startup/scope、原生 cache、Gradient 输入、互斥样本划分、Probe/Full/Shadow 指标、私有回执和哈希链账本，审计本身调用 API **0** 次。`FIRST_COMPLETE_OPPORTUNITY` owner review 已写入，但 runner 进程已退出，未消费该 review，也未写入官方终态文件。按照 fail-closed 规则，生命周期被记录为进程中断，不能声明 Canary 完成、有效或可用于完整 efficacy 结论。

## 范围限制

这些证据支持“搜索链路实际运行到 Shadow gate 并拒绝候选”的观察，不支持完整 Canary 的终态有效性、泛化、Pilot、Validation/Test 性能或任何因果归因。授权已经消费并关闭；没有继续执行、追加 API 调用或重跑。

完整身份和私有原始证据保存在运行目录；本目录只发布 hash、计数、类别和指标，不发布题目、答案、Prompt 或 Solver 原文。
