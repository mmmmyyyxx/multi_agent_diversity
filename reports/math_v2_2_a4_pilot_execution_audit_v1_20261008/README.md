# V2.2 A4 Pilot 前审计与工程修复

本轮完成零 API 审计和可确定的实现修复，**真实 Pilot 保持 HOLD**。用户附件第 5 节要求 finite bound 同时逻辑正确且资源现实；目前只证明了提交有限性，尚未建立资源现实的完整调用上界。本轮没有初始化真实团队、没有创建/消耗新的冻结 API scope，旧 attempt6 未补跑。

基线 `e9b5f55317ddc88fee21c49c7c87ee8254b9c0f2`；经过测试的修复 source `edb5536d9a5839f099bcce74d401ea2e94080e61`。当前方法仍为 `unified_team_prompt_search_v2_2` / `initial_competence_target_or_team_progress_v3`。预期执行 binding 仍是 `MATH_V2_2_EXECUTION_BINDING_V1`，**没有冻结或执行的新 binding/manifest/attempt/cache namespace**。

## A. 审计与修复

- 修复 V2.2 MATH adapter 落入旧 prediction protocol 分支的问题，恢复既有 V4 recovery protocol。
- TARGET 部署原先已经写 SUCCESS；补上 outcome 中缺失的 team/target gain 和 progress path，验证后续同成员检索可见。
- 增加 current-parent 绑定、完整整数测量、fixed-peer 和 materialized Full 一致性断言。原有 current peers 路径通过连续提交验证；不声称曾发生真实 stale-peer 事故。
- Windows 原子替换仅对 5/32/33 做最多八次有界恢复；永久锁、非 sharing 错误、receipt 篡改和 ledger corruption 继续失败关闭。
- 严格 Vote 增益相关命中全部分类，未发现 current TARGET 路径另有严格 Vote gate。promotion 接受正 target signal，但仍受原 top2 ceiling 约束。

完整 fake-provider 路径验证第一笔 TARGET commit：target 增益、Vote 不变、Shadow 允许零 target 增益、private SUCCESS 可检索。随后 synthetic sequential state 验证使用已更新 peers，第三成员修复转化为 Vote 增益。错误 Full 的部署回滚，Shadow 拒绝不改变团队、不写 SUCCESS；floor 不 rebase。它们是代码可达性证据，不是本轮真实效果。

WHO、Pattern/F、Gradient/Guard/Recovery、Memory 检索、Layer1 数值限制、两个 Full ceiling、plurality、Shadow、安全门槛和 patience 均未改变。五个初始化 prompt 的 byte identity 和独立 member/cache lanes 通过零 API 校验；真实 realized overlap、错误相关性、Oracle/Vote 未测量，不借用旧团队的对称 realization。

## B. 有限界与 HOLD

在完整 Optimize60 上，score 和 Vote 为整数 0..60，单次只替换一个成员。定义：

```text
Phi = 61 * Vote + sum(member_scores)
```

TARGET 路径 Vote 不变、target 至少加一，Phi 至少加一。TEAM 路径 Vote 至少加一、target 最多下降 60，Phi 仍至少加一。Phi 最大为 3960，因此保守最多 3960 次 commit；若真实初始 Phi 已测量，可减去它，但本轮不能假设初始 score 是旧的 22。另以生产 equivalence plurality 和四行合成样本验证：Vote 3→4、target 3→1 的合法 TEAM 更新使简单 `Vote+sum(score)` 从 20 降为 19，而加权 potential 从 32 升为 35；这不是新的实验 cell。

这不等于最多 3960 opportunities。failure count 在未提交成员上跨提交保留；沿用不改变 WHO 的保守 epoch 推导，令 `H_next=241*H+242`，一个 epoch segment 至多 `5*H_next` opportunities，至多 `2*(3960+1)=7922` segments。该精确算术上界的十进制位数至少 **18872**。它是上界的大小，绝不表示实际 Pilot 必须做这么多调用。浮点 discount 在该极端范围的实现也未被此精确算术证明覆盖。

所以提交有限性已经闭合，**资源现实的完成上界仍未闭合**。未把 arbitrary provider cap 当成收敛证明，未把资源耗尽当 scientific saturation，未重用 V2.1 的 N-commit 证明。`TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN` 与生产执行 HOLD 保留。

## C. 验证状态

完整 current suite：**1551 passed、0 failed、2 skipped、1468 deselected**，errors=0。V2.2 专项 38 passed；包含 1005 周期与并发 readers 的 persistence stress、crash recovery、receipt 冲突/篡改、Guard V6、Partition Completion、first-valid bounded Gradient Recovery、governance/schema/source/replay preservation 检查。compileall、依赖闭包、MATH 数据与初始化身份审计通过，网络 guard 在 application import 前生效，network attempts=0。

完整 suite 的失败按原始结果保留。隔离复测结果见 [engineering verification](engineering_verification.json)；隔离通过不改写 full suite 的失败。初版新测试 fixture 漏掉 evidence view 导致一次测试失败，补齐 fixture 后专项为 38/38，未放宽生产 guard。未执行完整 historical/private-artifact replay，deselected 不视为通过。

## D. 本轮真实 Pilot 与账本

真实 provider/transport/token 增量均为 0；opportunities、proposals、Full 与提交均没有真实运行。真实初始/最终 member scores、Vote、Oracle 和 scientific stop reason 均为未测量。本轮是 preexecution HOLD，不能把它记作一个有效的 0-commit Pilot。Validation=0、Test=0。

权威账本只读校验：累计 charged=3,414,772，remaining=36,585,228，reserved=0，累计 ceiling=40M。4497 个历史 tracked 文件与旧 attempt6 的 2560 个 raw 文件哈希不变。

## E. 结论与下一步

**ESTABLISHED**：零 API 合法 TARGET 路径、current-parent 连续更新、整数提交 potential。**SUPPORTED**：fake-provider 下的事务和 Memory conformance、有界 Windows sharing recovery。**NOT SUPPORTED**：新真实 A4 efficacy、组件因果效果或 generalization 的断言。**INCONCLUSIVE**：本轮真实 Q1–Q4，因未执行 Pilot。

下一步是在不改 WHO/failure-discount/epoch 科学语义的前提下，寻找并验证更紧、资源现实的 completion bound；没有建立前不能冻结真实 scope。当前任务级条件授权已记录，未消费；一旦需要改变上述方法，须作为科学设计问题另行处理。此处没有预写下一实验或追加 Validation/Test。

机器证据：[findings](findings.json)、[bound](finite_bound_assessment.json)、[tests](engineering_verification.json)、[accounting](provider_accounting_read_only.json)、[preservation](historical_preservation.json)、[occurrences](strict_vote_occurrence_classification.json)、[source](source_identity_summary.json)、[raw hashes](raw_evidence_hashes.json)。只发布脱敏 hashes、计数、类别和指标；私有 provider/problem/answer 内容未提交。
