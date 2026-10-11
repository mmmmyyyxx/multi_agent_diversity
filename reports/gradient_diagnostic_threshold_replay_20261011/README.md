# Gradient 两个诊断说明字段的离线阈值回放

建议下一正式版本先采用 observed_failure / diagnosis 各 400 字符。本轮只是可行性研究，当前正式 Prompt、Validator、Policy Identity 均保持原 240；reusable_correction 保持 400、expected_effect 保持 240。Solver、Aggregation、Full/Shadow Gate、正式停止和历史实验身份、成绩均未修改。

## 方法与完整性

以诊断父提交 79828718、历史成绩提交 30a57d82、实际执行源码 01306fbf 为精确来源。先重放原 172 条 Canary/Pilot 响应并核对终态 inventory 全部文件字节哈希、原始判定及 ledger。随后在内存 AST 中仅替换两字段共用长度比较；将替换恢复为 240 后必须与整棵原始 AST 完全一致。注册的 5×5 矩阵为 240、320、400、480、640 的全部组合，不选择或删除响应。

每条响应均经原 provider 输入 schema、Solver trajectory 与 reference_solution 来源核验，以及原 JSON 解码、精确字段集、disposition/block、expected_effect、400 字符 correction 和全部抽象/泄漏检查。保留 UNCERTAIN 的 null 规则。单条拒绝时只在内存重复交给原三次恢复循环来取得原生失败分类，不产生新响应或新费用。负控确认 correction 超长、effect 超长、非法 block、复制 reference、UNCERTAIN 非 null、额外 key、非法 JSON 和伪造 reference 来源仍被拒绝。诊断文本不作截断、删字或 JSON 修复。

计数分两层：全部历史物理响应独立通过数，包含反事实提前停止后不会到达的 suffix；每个历史逻辑诊断只取最早的完整通过。可避免费用只加总该响应之后的真实历史 suffix usage，不减掉使其通过的那条响应，不估计未生成响应或下游成本。

## Pilot 对比

| 两字段各上限 | 全部物理响应完整通过 | 逻辑诊断有完整通过 | 首次即通过 | 原耗尽获救 | 可避免后续重试 | 对应 usage Token |
|---:|---:|---:|---:|---:|---:|---:|
| 240 | 30/164 | 30/68 | 15/68 | 0 | 0 | 0 |
| 320 | 77/164 | 56/68 | 34/68 | 26 | 45 | 183,677 |
| 400 | 94/164 | 60/68 | 39/68 | 30 | 55 | 224,175 |
| 480 | 99/164 | 61/68 | 40/68 | 31 | 56 | 228,768 |
| 640 | 101/164 | 61/68 | 40/68 | 31 | 57 | 233,083 |

原 96 条额外重试用 394,485 Token。400 对应的局部可避免 224,175 Token 占原重试 56.8%、原 Pilot 阶段 usage 13.3%；输入为 208,252，输出为 15,923。原 38 个耗尽诊断中 30 个在已有响应中找到完整通过，仍有 8 个耗尽。全部通过都是 ACTIONABLE，没有把 UNCERTAIN 计作可用修复。该数量不证明修复正确或团队提高。

400→480 只增加 1 个逻辑诊断和 1 次可避免重试（4,593 Token）；480→640 不增加逻辑诊断，仅多省 1 次重试（4,315 Token）。400 已达到此网格最大局部节省的 96.2%，故优先推荐 400。25 组合也表明 diagnosis 是更主要瓶颈：仅将其提升至 400、observed_failure 留 240，可省 33 次 / 136,844 Token；反向仅提升 observed_failure 至 400 可省 6 次 / 24,822 Token。400/480 的非对称组合与 480/480 有相同逻辑通过和费用，非对称优势仍属单个历史样本观察。

Canary 仅 3 个逻辑诊断 / 8 条响应。320 及以上均有 4 条物理响应通过、2 个逻辑诊断通过，可避免 3 次重试 / 11,499 Token；更高上限不增加局部收益。Canary 用于补充一致性，不能视为阈值选择后的独立验证。

## 安全检查和字段作用

400 下另有 1 条 Pilot 响应暴露原先被长度挡住的泄漏拒绝，4 条暴露其他 correction 拒绝，均不计入收益。原已有的 1 条泄漏拒绝也保留。不得从原首拒分布直接推断长度放宽后的完整通过数。640 下也仍保留这些检查；JSON、schema、effect 或 correction 失败并未因诊断上限变化而消失。

observed_failure / diagnosis 只进入私有诊断记录。Pattern clustering 的 gradients 列表及 repair evidence 用的是校验后的 reusable_correction；不向 Solver 输送诊断字段。两字段长度更宽不等于 Pattern 修复长度更宽，但接受更早或不同 correction 会改变其集合、聚类、选择、后续优化和成本。

## 无法离线确定的风险及正式变更要求

历史生成 Prompt 仍要求 240。改成 400 的新 Prompt 可能令模型输出更长、生成成本更高或改变 correction；本回放不能预测新分布，也不能证明 224,175 Token 会在新 Pilot 实现。下游聚类/优化因 correction 改变，故本地 suffix 费用不能解释为完整新轨迹节约、供应商账单或性能收益。统计样本来自一个只有五次机会的 Pilot，存在回顾性选择与相关响应偏差。通过原安全守卫也不是语义安全证明，诊断原因仍是可观察证据上的假设。

更多合法 correction 也可能触发额外聚类、搜索或评估；即使 Gradient 局部重试费用下降，整个优化过程总费用仍可能增加。

本轮不切换生产阈值。下一正式版本如采纳推荐，应同时更新 Prompt 明确的两字段上限、Validator 的 policy 读点、Gradient policy / prompt / 完整组合与 binding 身份、规范和边界及安全回归测试；保持 correction 400、effect 240、原 schema/source/leakage/recovery，冻结新 source、manifest、cache 和独立授权。不复用历史身份、checkpoint、响应为新实验产物或已消费授权。

## 验证与交付

25 项针对性离线测试通过；完整当前套件 649 通过、2 跳过，220 个历史/私有模块按分类排除。治理、编译、manifest、脱敏和报告发布字节检查通过，记录于 verification.json；历史私有回放不宣称全通过。扫描历史混合 trace 以选择 Optimize Gradient，终态 inventory 的其他文件只核验哈希；Shadow 响应不作诊断输入或重评估。真实 API、Validation/Test 原始访问和新模型评估为零。父报告原 CRLF/LF 投影限制继续保留，不重写旧 seal；本报告以 LF 字节写入并校验发布哈希。

完整矩阵、剩余原生失败类别和长度分布见 [阈值数据](threshold_comparison.json)；未来版本要求见 [研究建议](../../docs/research/GRADIENT_DIAGNOSTIC_LENGTH_FEASIBILITY_V1.md)。
