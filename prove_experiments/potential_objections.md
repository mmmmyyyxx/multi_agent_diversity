# 潜在质疑与需要回答的问题

## 质疑 1：指标只检测 prompt 措辞，不检测真实推理策略

需要的实验：

- P2 同策略 paraphrase 负对照。
- P3 显式 mixed-strategy 干预。
- 基于完整 trace 的人工盲评。

需要的回答：

同策略 paraphrase 不应产生高 family diversity，而 mixed-strategy prompts 应产生更高 diversity。人工也应认为 mixed-strategy 组的方法更多样。

## 质疑 2：judge 不稳定

需要的实验：

- P1 同一 trace 重复判定。
- 第二 critic model 的 audit 子集。

需要的回答：

major-family 和 primary-family 一致率必须足够高。如果只有 leaf label 不稳定，则训练 reward 应使用 weighted tree，而不是 strict leaf equality。

## 质疑 3：指标测到的是模型身份、输出风格或长度

需要的实验：

- P4 跨 LLM 策略迁移。
- 比较 strategy-tree diversity、raw trace embedding diversity 和 trace length。

需要的回答：

strategy prompt effect 应大于 model identity effect。指标不应主要由 trace length 解释。

## 质疑 4：指标奖励了无意义的“人工多样性”

需要的实验：

- 人工盲评。
- invalid trace penalty 监控。
- 准确率作为安全指标记录，即使不优化准确率。

需要的回答：

高 diversity trace 仍然必须是有效推理轨迹。多样性提升不能来自空输出、格式错误或无意义 trace。

## 质疑 5：指标太严格，训练无法优化

需要的实验：

- P5 reward weight sweep。
- candidate signal rate 分析。
- early stopping 和 update applied rate 分析。

需要的回答：

至少一个中等强度设置应提升验证集 diversity。如果 strict 设置失败但 softened 设置有效，说明指标有效但 reward 强度需要校准。

## 质疑 6：MMLU 本身不支持很多有效策略

需要的实验：

- P8 任务依赖检查。
- MMLU subject-level 分析。

需要的回答：

reachable diversity 应随 subject 和 dataset 改变。受限任务上 diversity 低不一定是失败，只要多方法任务上干预有效。

## 质疑 7：bank baseline 已经解决问题，训练没必要

需要的实验：

- 比较 `bank_baseline`、`shared_div` 和 P5 reward sweep。
- 检查 shared 初始化是否能在不预设角色的情况下接近 bank baseline。

需要的回答：

如果 shared training 无法接近 bank baseline，说明自动角色分化机制、rewriter 或搜索过程还弱。指标仍可能有效，但训练算法尚未充分利用它。

