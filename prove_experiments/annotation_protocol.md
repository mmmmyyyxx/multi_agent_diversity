# 人工标注协议

该协议只用于证明实验，不进入训练流程。

## 标注者看到什么

对每个 question-level group，展示：

- 可用时展示题目文本。
- 五条匿名完整 trace：Agent A 到 Agent E。
- 不展示模型名。
- 不展示 prompt。
- 不展示实验设置。
- 不展示自动 family label。
- 不展示 gold answer 或 vote correctness，除非额外标注正确性；本实验不需要。

## 主标注任务

问题：
这五条 trace 使用的推理方法有多不同？

评分：

1. 方法相同，差异主要是措辞、长度或顺序。
2. 基本相同，有少量变化。
3. 有混合，但方法仍明显重叠。
4. 多个 agent 使用了清楚不同的方法。
5. 方法高度多样，agent 从明显不同的推理路径解决问题。

标注时忽略：

- 最终答案是否正确。
- trace 是否写得更流畅。
- trace 是否更长。
- 细微措辞差异。

标注时关注：

- trace 是从定义、选项、公式、分类、反证、例子、反向检查、模拟还是分解开始。
- agent 是否使用不同中间表示。
- agent 是否用不同方式验证、排除或比较选项。
- 推理顺序是否反映不同方法，而不只是重写。

## 可选 coarse tags

每条 trace 可选标 1 到 2 个 coarse tags：

- definition/concept match
- option elimination
- option comparison
- backward verification
- rule/principle application
- decomposition
- direct computation
- algebraic derivation
- case analysis
- counterexample
- simulation/tracing
- causal/mechanism reasoning
- other clear method
- unclear/no real reasoning

## 抽样计划

目标 80 组：

- 20 组高 strategy-tree diversity。
- 20 组低 strategy-tree diversity。
- 20 组高 embedding diversity 但低 strategy-tree diversity。
- 20 组低 embedding diversity 但高 strategy-tree diversity，如存在。

为了省成本，可先标 40 组；如果结果模糊，再扩展。

## 一致性与有效性

计算：

- 人工 1 到 5 分数与 `team_family_diversity` 的 Spearman correlation。
- metric-high 与 metric-low 组的人工分数差异。
- 可选 coarse tags 与 primary/secondary family label 的一致性。

好的结果不要求人工 tag 与自动 family 完全一致。最重要的问题是：指标认为高多样性的组，人类是否也认为方法更多样。

