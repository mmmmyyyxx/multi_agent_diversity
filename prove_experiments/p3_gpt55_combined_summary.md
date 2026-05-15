# P3 GPT-5.5 综合验证结论

本报告合并两个互补实验，但证据优先级不同：

- 主证据：`p3_normal_judge_gpt55`。让 GPT-5.5 在完整 taxonomy、major-family tree、family definitions 和正常 judge 同等信息下，重新给单条 trace 贴策略标签。这个实验和原策略树指标处在同一评判空间里，因此对“原 judge/taxonomy 是否把真实策略差异贴错”最有解释力。
- 补充证据：`p3_prompt_following_gpt55`。让 GPT-5.5 不看 taxonomy，只看原始策略指令和 trace，直接判断模型是否遵循了目标策略。这个实验更接近外部语义评审，用来辅助解释“低 exact target hit 是模型没遵循，还是 taxonomy leaf 命中低估了真实策略响应”。

两个实验抽样的是同一类高风险样本：target 不包含 `option_contrast`，但原自动 judge 的 primary 标签是 `option_contrast`。样本数为 40，候选池为 722。

## 综合结论

最重要的结论应优先来自实验一：在与正常 judge 基本相同的信息条件下，GPT-5.5 不支持原自动 judge 把这些高风险 trace 的 primary 标签判成 `option_contrast`。原 judge 的 `option_contrast` primary 只有 3/40 被 GPT-5.5 支持，而 GPT-5.5 对这些复判的平均 confidence 为 0.8960。因此，P3 中 mixed exact target hit 偏低，首先应被解释为 **原 judge/taxonomy 存在明显的 `option_contrast` 吸附和 primary/secondary 排序问题**。

第二层结论仍然来自实验一：GPT-5.5 并没有把大多数样本精确贴回 target leaf。整体 `pair exact target hit` 只有 0.1250，但 `pair same-major hit` 达到 0.7000。这说明问题不是“原 judge 错了，所以目标策略就精确命中了”，而是更细：原 judge 的 `option_contrast` 判定多数不成立；GPT-5.5 认为很多 trace 落在目标策略附近的大类；但 leaf 级 exact target hit 仍然过严，受 taxonomy 边界和主次标签排序影响很大。

实验二作为补充证据：GPT-5.5 直接看原始策略指令和 trace 时，认为 29/40 样本遵循目标策略，35/40 至少部分遵循。这支持实验一的解释：低 exact target hit 不能简单等价于“模型完全不听策略 prompt”。不过，实验二不是 taxonomy 内部复判，不能替代实验一；它主要说明真实策略遵循信号确实存在，并帮助区分 judge/taxonomy 问题和模型/prompt 问题。

综合起来，P3 的结果应理解为三个因素叠加：

1. 原自动 judge/taxonomy 对 `option_contrast` 有明显吸附，容易把 MMLU 多选题中常见的“逐项检查选项”表面形式提升成 primary label。
2. 模型很多时候确实响应了目标策略，但响应方式未必精确落在目标 leaf 上，而是落在同一 major family 或邻近 leaf。
3. 仍存在真实的 prompt 遵循失败，尤其是 `rule_or_principle_application` 和 qwen2.5-7b-instruct 的部分样本。

因此，当前最强结论不是“策略干预完全成功”，也不是“mixed 策略干预失败”，而是：**原策略树 judge 的 leaf 级标签，尤其 `option_contrast` primary，对 MMLU trace 的真实策略差异存在系统性低估和错贴；策略干预有真实响应信号，但需要用更稳的 taxonomy/judge 或 major-level 指标来评估。**

## 实验一：GPT-5.5 作为 Normal Taxonomy Judge（主证据）

GPT-5.5 输入包含完整 taxonomy leaf labels、major-family tree、base family definitions、reasoning_summary 要求、confidence/evidence 规则、返回 JSON schema，以及单条 trace 的正常 judge 元信息。它不看目标策略、模型名、run 名称、原 judge 标签、gold answer 或 group/vote 信息。

### 是否仍支持原 judge 的 option_contrast

| n | GPT primary option | GPT pair option | original judge supported | judge/taxonomy questioned | confidence |
|---|---|---|---|---|---|
| 40 | 0.0750 | 0.1750 | 0.0750 | 0.9250 | 0.8960 |

解释：原自动 judge 的 primary 全部是 `option_contrast`，但 GPT-5.5 只有 3/40 也把 primary 判为 `option_contrast`。如果把 secondary 也算上，也只有 7/40 含 `option_contrast`。这强烈说明原 judge 的 `option_contrast` primary 有系统性过度吸附。

### GPT-5.5 标签是否命中目标策略

| n | primary exact target hit | pair exact target hit | primary same-major hit | pair same-major hit |
|---|---|---|---|---|
| 40 | 0.1250 | 0.1250 | 0.6000 | 0.7000 |

解释：GPT-5.5 不支持原 judge 的 `option_contrast`，但它也没有把大多数样本精确贴回目标 leaf。整体 exact target hit 只有 0.1250；不过 same-major hit 达到 0.6000/0.7000，说明许多 trace 的方法确实靠近目标策略大类，而不是完全无关。

### 按目标策略拆分

| target | n | GPT primary option | pair exact target hit | pair same-major hit |
|---|---|---|---|---|
| concept_definition_match | 10 | 0.0000 | 0.5000 | 0.8000 |
| answer_to_stem_backward_check\|option_contradiction_check | 10 | 0.2000 | 0.0000 | 0.9000 |
| rule_or_principle_application | 10 | 0.0000 | 0.0000 | 0.0000 |
| decomposition\|stem_evidence_alignment | 10 | 0.1000 | 0.0000 | 1.0000 |

解释：`concept_definition_match` 是 leaf 级恢复最明显的目标，pair exact 为 0.5000。`answer_to_stem_backward_check|option_contradiction_check` 和 `decomposition|stem_evidence_alignment` exact 为 0，但 same-major 很高，说明 GPT-5.5 认为它们靠近目标大类。`rule_or_principle_application` 最弱，既没有 exact，也没有 same-major，说明该 prompt 最容易被模型转写成普通事实判断或选项排除。

## 实验二：GPT-5.5 直接判断 Prompt Following（补充证据）

这个实验不让 GPT-5.5 看 taxonomy，而是看原始策略指令和 trace，直接判断 trace 是否遵循目标策略。它的作用是补充解释实验一中的 leaf exact target hit 偏低：如果 taxonomy exact hit 低，但 prompt-following 认为遵循，那么更可能是 taxonomy leaf 边界或 judge 贴标问题；如果二者都低，才更像真实模型/prompt 失败。

### Overall

| n | followed_rate | mean_score | partial_or_better | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|
| 40 | 0.7250 | 3.9000 | 0.8750 | 0.7000 | 0.1250 | 0.1750 |

解释：GPT-5.5 认为 29/40 样本遵循了目标策略；35/40 至少部分遵循。只有 5/40 明确更像模型/prompt 没有稳定诱导目标策略。这说明原 mixed exact target hit 低，很大程度是 judge/taxonomy 贴标问题，而不是策略指令完全无效。

### 按目标策略拆分

| target | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|
| concept_definition_match | 10 | 0.7000 | 3.7000 | 0.7000 | 0.2000 | 0.1000 |
| answer_to_stem_backward_check\|option_contradiction_check | 10 | 0.9000 | 4.1000 | 0.8000 | 0.1000 | 0.1000 |
| rule_or_principle_application | 10 | 0.5000 | 3.6000 | 0.5000 | 0.2000 | 0.3000 |
| decomposition\|stem_evidence_alignment | 10 | 0.8000 | 4.2000 | 0.8000 | 0.0000 | 0.2000 |

解释：`answer_to_stem_backward_check|option_contradiction_check` 和 `decomposition|stem_evidence_alignment` 的真实遵循较好。`rule_or_principle_application` 最弱，说明这个 prompt 需要重写得更具体，要求模型先显式给出规则/原则，再用该规则解释 stem，而不是直接用领域事实排除选项。

### 按模型拆分

| model | n | followed_rate | partial_or_better | mean_score |
|---|---|---|---|---|
| deepseek-chat | 12 | 0.8333 | 1.0000 | 4.1667 |
| gemini-2.5-flash-lite | 12 | 0.8333 | 0.9167 | 3.9167 |
| gpt-4o-mini | 8 | 0.7500 | 0.8750 | 4.2500 |
| qwen2.5-7b-instruct | 8 | 0.3750 | 0.6250 | 3.1250 |

解释：qwen2.5-7b-instruct 是主要风险来源，常见问题是短答、无可见推理、或没有体现目标方法。deepseek-chat、gemini-2.5-flash-lite 和 gpt-4o-mini 在这组样本中总体更能响应显式策略要求。

## 两个实验合起来怎么读

读法上应先看实验一，再用实验二解释实验一中无法完全区分的部分。

实验一已经给出主结论：原自动 judge 的 `option_contrast` primary 大多不被 GPT-5.5 支持；同时 GPT-5.5 taxonomy exact target hit 仍低，但 same-major hit 明显更高。实验二再回答：这些 exact 未命中的样本里，有多少其实仍遵循了原始策略指令。

逐样本联合后有几个关键数字：

| 关系 | 数量 | 含义 |
|---|---:|---|
| GPT-5.5 taxonomy pair exact 未命中，但 prompt-following 认为遵循 | 25/40 | leaf 级 taxonomy 命中低估了真实策略遵循 |
| GPT-5.5 taxonomy pair exact 命中，且 prompt-following 认为遵循 | 4/40 | taxonomy 和直接遵循判断一致支持目标策略 |
| GPT-5.5 taxonomy pair exact 未命中，且 prompt-following 也认为未遵循 | 10/40 | 真实模型/prompt 风险或策略表达不足 |
| GPT-5.5 taxonomy pair exact 命中，但 prompt-following 认为未遵循 | 1/40 | taxonomy 标签可能捕捉到局部形式，但策略指令未主导 trace |

如果放宽到 same-major，结论更清晰：

| 关系 | 数量 | 含义 |
|---|---:|---|
| GPT-5.5 taxonomy pair same-major 命中，且 prompt-following 至少部分遵循 | 28/40 | 大多数样本在策略大类和直接遵循两个维度上都有信号 |
| GPT-5.5 taxonomy pair same-major 未命中，但 prompt-following 至少部分遵循 | 7/40 | taxonomy major 映射仍可能漏掉真实策略响应 |
| GPT-5.5 taxonomy pair same-major 未命中，且 prompt-following 也未部分遵循 | 5/40 | 更可信的模型/prompt 失败样本 |

综合来看，实验一是更强证据：在 taxonomy 内部，GPT-5.5 已经证明原 judge 的 `option_contrast` primary 大多站不住。实验二是补充证据：在 taxonomy 外部，多数样本仍被认为遵循或部分遵循原始策略指令。两者合起来说明，exact leaf 指标过严，尤其在 MMLU 多选题上会受到 `option_contrast` 吸附和邻近 leaf 边界影响；major-level 指标和 prompt-following 评审可以作为辅助解释，但不应替代 taxonomy judge 的主分析。

## 对 P3 结论的修正

原始说法“mixed exact target hit 低，说明模型对策略遵循不强”需要改成更严谨的版本。这里优先依据实验一，再用实验二作补充：

> P3 中 mixed exact target hit 低，首先反映的是原 judge/taxonomy 的 `option_contrast` primary 吸附和 leaf 级边界问题，而不能直接解释为模型不遵循策略。GPT-5.5 在同一 taxonomy 和正常 judge 信息下复判时，只支持 3/40 个原 `option_contrast` primary，但给出 0.7000 的 pair same-major target hit，说明许多 trace 位于目标策略附近的大类。GPT-5.5 prompt-following 评审进一步显示，多数高风险 trace 仍遵循或部分遵循原始策略指令。因此，P3 证明的是：当前 leaf exact target hit 低估真实策略响应；需要修正 taxonomy leaf 边界、`option_contrast` 判定规则、primary/secondary 排序，以及部分 prompt 的可执行性。

## 后续修改建议

1. 收紧 `option_contrast` 的判定：只有当“选项之间的相互比较”是主导组织结构时，才把它作为 primary；如果 trace 先建立概念、规则、证据或机制，再检查选项，应把 `option_contrast` 放 secondary 或不贴。
2. 报告 P3 时同时给出三类指标：原 judge exact hit、GPT-5.5 taxonomy same-major hit、GPT-5.5 prompt-following followed/partial rate。
3. 重写 `rule_or_principle_application` prompt，要求模型显式输出“规则/原则是什么”和“该规则如何约束 stem”，再进入选项。
4. 对 qwen2.5-7b-instruct 加强格式与可见推理约束，避免短答导致无法判定策略。
5. 在主实验结论里避免只用 exact leaf hit 支撑策略遵循失败，应把 exact leaf 解释为严格 taxonomy 对齐，而不是唯一的真实策略遵循指标。
