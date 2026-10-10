# 当前科学语义对齐与 Pilot 离线诊断

本轮仅修正规范作用范围并增加离线可观测性。Unified Team Prompt Search、正式生成、评分、Full/Shadow 接纳、Aggregation 和 `team_epoch_no_commit_v1` 均未修改。真实 API 调用、Validation/Test 原始读取为零；未设计或执行新实验，未授权 Push。

## 审计边界与规范修正

最新远端及父报告提交为 `30a57d82213fb7788581e3fe8fd9ba7903dfb302`。历史响应逐条使用 `01306fbf9cde002291ab21e2f93a2c02399f7e9e` 的原生 Gradient validator 复核，当前代码只做诊断计数。Canary 8 条、Pilot 164 条的接纳、拒绝类别、恢复、耗尽均与原记录一致；所有私有终态 inventory 字节哈希核验通过。历史分数、classifier、manifest、报告和私有原始响应均未改写。

| 作用范围 | 对齐内容 |
|---|---|
| 全局科学原则 | 有用正确覆盖与实际团队表现；差异不等于互补；可观察修复假设不等于隐藏原因；工程有效性不等于方法效果 |
| MATH Scientific Contract | validity 与 binary correctness 分离；Oracle 与 Vote 分离；初始完整 Optimize 正确数下限；有限逐题损失及 validity 退化由冻结门禁决定 |
| 当前 V2.5/V3 | 每个合法实际评估的局部编辑继续搜索；当前 assigned repair + independent Probe；Full、winner-only Shadow、Commit 与停止分层 |
| 单次实验 | seed、Arm、模型、面板、预算、机会 cap 和授权归属 manifest；旧 seed81/V2.2 budget 不再冒充当前设置 |

新增规范入口见 `docs/design/SCIENTIFIC_PRINCIPLES.md` 与 `docs/design/MATH_SCIENTIFIC_CONTRACT.md`。V2.5 主契约、V3、parser/recovery 文档和旧契约 scope 标注已对齐；旧契约的完整原文仍在原冻结 Git source 中。

## 停止诊断

| 阶段 / 机会 | Commit | 部署 Vote 增量 | 连续无 Commit 机会 | 连续无 Vote 改善机会 |
|---|---|---:|---:|---:|
| Canary 1 | YES | 0 | 0 | 1 |
| Pilot 1 | YES | +1 | 0 | 0 |
| Pilot 2 | YES | +1 | 0 | 0 |
| Pilot 3 | NO | 0 | 1 | 1 |
| Pilot 4 | NO | 0 | 2 | 2 |
| Pilot 5 | NO | 0 | 3 | 3 |

计数采用部署后的团队分数；未接纳候选的 Full Vote 不会重置它们。二者都不控制运行。原生 epoch stopper 复核未达到科学停止：Pilot 最后一个 epoch 仅覆盖 5 个可选成员中的 3 个，完整无 Commit epoch 数为 0。运行终止仍是五机会 cap；Canary 的 target-only Commit 说明 Commit 与 Vote 改善不可混用。

## Gradient 结构失败与成本

Pilot 68 个逻辑诊断消耗 164 条响应：首次接纳 15 个，重试后接纳 15 个，三次耗尽 38 个。初次失败的 53 个中恢复 15 个（28.3%）。以下按原生校验顺序记录每条拒绝的首个可定位原因；不同字段的次级超限另列在 JSON，不能相加当作唯一失败数。

| 首个拒绝原因 | Pilot 响应数 | 由前一拒绝触发的后续重试 usage Token |
|---|---:|---:|
| observed_failure 超过 240 字符 | 55 | 156,806 |
| diagnosis 超过 240 字符 | 41 | 125,746 |
| suggested_block 不在冻结枚举 | 12 | 42,498 |
| observed_failure 空或类型不符 | 1 | 0 |
| JSON 反斜杠转义无效 | 20 | 58,587 |
| 其他 JSON 语法问题 | 3 | 6,696 |
| reusable_correction 超过 400 字符 | 1 | 0 |
| 原冻结来源/泄漏守卫拒绝 | 1 | 4,152 |
| 合计 | 134 | 394,485 |

此表费用归于上一个拒绝，排除各逻辑诊断的首次响应。它说明重试由什么失败触发，不证明供应商内部根因，也不等于删除检查后的可实现节约。

Pilot 96 条额外响应共 394,485 Token，占阶段 usage 23.5%；其中输入 365,902（92.8%）、输出 28,583。38 个最终耗尽诊断共用 472,318 Token；其额外 draw 用 314,401 Token，占全部重试费用 79.7%。成功恢复的额外费用为 80,084 Token。大量费用来自对相同输入的重复生成请求；不据此推断供应商缓存或账单。

Canary 3 个逻辑诊断 / 8 条响应，首次接纳 0、恢复 1、耗尽 2；5 条额外响应耗费 19,890 Token。两阶段所有 Gradient finish_reason 都为 stop，未观察到输出容量截断。Pilot 96/134 的首拒是两段诊断说明字段超限，正文可复用修复的 400 字符超限仅为首拒 1 次；但 18 条 JSON 可读响应同时有该次级超限。不能将这些都归因于 correction 长度或 Solver parser。

## 已支持的结论、假设与下一步

已支持：字段长度、JSON 转义、block schema 是本次结构拒绝的主要来源，重试输入成本高；原科学门禁保持有效。结构失败遮蔽的后续内容是否安全未被证明，不能因来源拒绝次数少就放宽守卫。没有证据将 Commit 或收益归因于 Gradient、Pattern 或 Memory 的单一组件。

优先研究方向是提高生成结果对冻结 schema 的遵守，包括更简短的诊断说明、正确 JSON 转义、单一合法 block，以及避免相同完整输入的高成本重发。具体生成约束、请求压缩、重试反馈或预算变动都只是待验证方案，会改变请求/生成语义，必须另行版本冻结；本轮没有实现这些方案，没有提高长度上限、自动截断文本或修复 JSON 后接纳。

Oracle 向 Vote 转化保留为研究问题。父 Pilot 新增三道 Oracle 覆盖仍全部 Vote 错，Vote 的两道修复来自已有覆盖。仅五次机会不足以判断更长轨迹是否转化，暂不据此改变 Aggregation 或 Acceptance Gate。Optimize/自适应 Shadow 的描述性观察不提供独立泛化证据。

## 完整性限制与验证

父报告的 README、verification 与 sha256 manifest 在 Git 发布字节和本地字节之间仅有 CRLF/LF 投影差异；本地原 seal 与原终态 inventory 均通过，Git payload 逐文件核对仅允许该行尾差异。该发布字节限制单独记录于 provenance，未追溯重写父报告。新报告使用 LF bytes 写入并在暂存与提交后核对，避免将本地 seal 错当成 Git blob seal。

当前生产模块未改动；新增诊断模块没有 search/admission/stop read point。18 项针对性回归通过；完整当前套件 624 通过、2 跳过，220 个历史/私有模块按分类排除。compileall、治理、manifest、脱敏、生成索引与 diff 检查通过，详见 verification.json；历史私有回放不宣称全通过。

细项见 [Gradient 诊断](gradient_diagnostic.json)、[机会计数](opportunity_streaks.json)、[验证](verification.json) 和 [来源](provenance.json)。
