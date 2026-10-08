# 用户停止后的实验与实现情况

2026-10-08，用户要求“停止，梳理实验情况并给出报告，推送”。已终止本轮仍在运行的四个 Python 回归进程，保留已有私有证据，不再启动测试、canary、Pilot 或真实模型调用。最终 current suite 没有完整 XML/终止回执，不能推算完整通过数，也不伪造 PASS。

| 项目 | 完成情况 | 可以成立的结论 |
|---|---|---|
| 历史手工 prompt 容量诊断 | 已完成并发布，P2 两次均 25/60；固定团队 Vote 为 21、23，均值 22/60 | Optimize 上观察到 additive repair 的覆盖属性；不是新版 baseline 或受控 optimizer 因果证据 |
| V2.2 真实 A4 Pilot | 用户截停的 7 机会完整前缀；第 8 机会不完整 | 42 proposals，33 合格局部评分，27 Probe，2 Full，0 commit；不能称完整 Pilot 或科学收敛 |
| 42 候选 failure audit | 已完成，零 API，原证据封存 | 证据覆盖失配和不可执行的 optimizer-context 引用风险；未复现 gate 错误或严格净收益 export 错排 |
| V6 Solver 可见轨迹修复 | 起点提交已实现、推送且完成离线验证 | 书面解答进入 permitted optimizer feedback，最终 payload 独立评分；真实 Solver 输出质量未测 |
| 本轮 V2.3 方法修订 | 源码、规范、版本、协议、信息流审计与合成记录已实现 | 参考解诊断、独立 Optimize roles、轮换保留、初始覆盖 Memory 和实际编辑后果接通；仅有离线证据 |
| 本轮最终完整回归 | 按用户指令中断，无完成结果 | 发布状态 PARTIALLY_VERIFIED_USER_STOP；专项通过不能替代最终完整回归 |
| 下一轮真实两臂比较 | 只准备计划，HOLD / READY_TO_RUN=false | 没有新 API 授权、冻结执行或有效性结果 |

## 旧真实实验发生了什么

五个初始成员均为 22/60，部署团队 Vote/Oracle 为 22/60。七个目标成员依次为 0、1、3、2、4、0、1，期间没有部署更新，因此是对同一初始团队的七次局部尝试。

实际 panel 每次只有四题：三条 repair 加同一个原正确 anchor。Mutation、局部 validation 和 Probe 复用同一集合/请求实现。33 个合格候选相对机会 root 是 3 positive、25 neutral、5 negative；三个 positive 都修复同一道题。没有证据表明 export 经常舍弃已测净收益更好的候选。

| 已测 Full | 正确/60 | 新增 | 丢失旧正确 | 保留旧正确 | Vote/60 | Oracle/60 |
|---|---|---|---|---|---|---|
| O1G4 | 18 | 2 | 6 | 16 | 22 | 24 |
| O1G6 | 19 | 2 | 5 | 17 | 22 | 24 |

11 次丢失全部在局部 panel/Probe 外，去重为六道。gate 正确拒绝两者；同分第三个 positive 没有 Full，其能力未知。旧报告原文、私有 prompt、原题、答案和参考解均不改写、不公开。来源：[停止报告](../math_v2_2_gradient_pattern_seed81_pilot_v2_execution_20261008/README.md)、[候选审计](../math_v2_2_optimizer_failure_audit_v1_20261008/README.md)。

## 本轮已经完成什么

新版本为 `unified_team_prompt_search_v2_3`。保留五成员、固定 peers、单成员更新、等权 plurality、数学评分和 V3 Full transition。没有增加 generation、替换模型或加入新 agent/controller。

Optimize worked solution 与模型自身可见轨迹分别进入逐错题 Gradient；允许 UNCERTAIN。Mutation、SearchValidation、TeamProbe 各三题、彼此不相交，当前正确集合按 seed/member 轮换。六代局部评估上限由父与 child 的两组 role 推导为 42；四个 exports、两个 Full 名额不变。

原 Memory 从已有实测结果启动五条私有 competence 记录，记录实际 parent/child diff、修复假设、fixed/broken/retained、局部/Probe/Full 状态。拒绝 Full 也可被后续同成员 mutation 读取。当前覆盖只在真实 commit 事务后更新，候选-only 改善不会冒充部署能力。新规范进入 scientific source hash。

合成坏编辑：SearchValidation +1，Full 新增 4、丢失 7、净 -3，记为 FULL_REFUTED。合成好编辑保留原 12 道并新增 4 道，记为 COMMITTED。后续 fake provider 根据已拒绝记录返回 NO_SAFE_EDIT。这些响应由夹具预设，**不是 qwen 实测提升，也不是记忆有效性的真实因果证据**。完整脱敏 [记录](synthetic_memory_records.json) 和 [请求字段追踪](synthetic_wire_trace.json) 已保存。

## 验证与停止边界

最终源码专项 **22 passed**，应用 network attempts 0；compileall 和 current governance 通过。开发阶段完整 current suite **1,657 passed、2 skipped、1,468 历史/私有用例排除**；另一次 focused trajectory/governance 为 267 passed。这些运行早于最终少量补齐，不能升级为最终完整回归 PASS。两次后续完整回归分别因发布前修订重启和用户停止而中断。

HoVer/PuPa 的两个 skip 是真实数据未冻结。旧治理 CLI 的三项 registry 兼容错误与起点相同，新增零；该 CLI 未通过。最终版本还需要另行获准完成 current suite。详见 [verification.md](verification.md) 与 [verification.json](verification.json)。

本轮 **真实 API 调用 0、新增计费 tokens 0，真实 Shadow / Validation / Test 调用 0**。旧 attempt 账本与关闭授权保持原字节：累计 4,068,192，剩余 35,931,808，未结预约 0。旧两次 raw 共 2,349 个文件、四份历史报告及 parent binding/manifest 核验不变。

两臂真实比较 A=同 V6 接口的 V2.2、B=完整 V2.3，各五机会，只是 [未执行计划](../../experiments/protocols/math_optimization_evidence_v23/next_comparison.json)。必须新冻结 source/manifest/binding/cache/预算/访问与单次授权，并重新测初始 baseline。当前不能确认新版能生成并保留 P2 式 additive repair，也不能区分各修订组件的贡献。
