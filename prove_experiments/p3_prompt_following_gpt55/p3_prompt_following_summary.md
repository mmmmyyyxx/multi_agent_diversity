# P3 Prompt-following GPT-5.5 Validation

目标：抽样 `target` 不包含 `option_contrast`、但自动 judge 的 primary 标签为 `option_contrast` 的 trace，让 GPT-5.5 只看原始策略指令和 trace，判断模型是否真的遵循了策略指令。

- candidate_count: 722
- sampled_count: 40
- evaluated_count: 40

判读规则：

- GPT-5.5 认为遵循，而自动 judge 判 `option_contrast`：更像 judge/taxonomy 把选项形式过度吸附为 `option_contrast`。
- GPT-5.5 也认为没有遵循：更像模型/prompt 没有稳定诱导目标策略。
- GPT-5.5 认为部分遵循：说明两者都有可能，需要看具体 trace。

## Overall

| n | followed_rate | mean_score | partial_or_better | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|
| 40 | 0.7250 | 3.9000 | 0.8750 | 0.7000 | 0.1250 | 0.1750 |

## By Target

| agent | target | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|---|
| 0 | concept_definition_match | 10 | 0.7000 | 3.7000 | 0.7000 | 0.2000 | 0.1000 |
| 2 | answer_to_stem_backward_check\|option_contradiction_check | 10 | 0.9000 | 4.1000 | 0.8000 | 0.1000 | 0.1000 |
| 3 | rule_or_principle_application | 10 | 0.5000 | 3.6000 | 0.5000 | 0.2000 | 0.3000 |
| 4 | decomposition\|stem_evidence_alignment | 10 | 0.8000 | 4.2000 | 0.8000 | 0.0000 | 0.2000 |

## By Model And Target

| model | agent | target | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | concept_definition_match | 3 | 1.0000 | 4.6667 | 1.0000 | 0.0000 | 0.0000 |
| deepseek-chat | 2 | answer_to_stem_backward_check\|option_contradiction_check | 3 | 1.0000 | 4.0000 | 0.6667 | 0.0000 | 0.3333 |
| deepseek-chat | 3 | rule_or_principle_application | 3 | 0.3333 | 3.6667 | 0.3333 | 0.0000 | 0.6667 |
| deepseek-chat | 4 | decomposition\|stem_evidence_alignment | 3 | 1.0000 | 4.3333 | 1.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-lite | 0 | concept_definition_match | 3 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-lite | 2 | answer_to_stem_backward_check\|option_contradiction_check | 3 | 1.0000 | 4.3333 | 1.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-lite | 3 | rule_or_principle_application | 3 | 0.3333 | 3.0000 | 0.3333 | 0.3333 | 0.3333 |
| gemini-2.5-flash-lite | 4 | decomposition\|stem_evidence_alignment | 3 | 1.0000 | 4.3333 | 1.0000 | 0.0000 | 0.0000 |
| gpt-4o-mini | 0 | concept_definition_match | 2 | 0.5000 | 4.0000 | 0.5000 | 0.0000 | 0.5000 |
| gpt-4o-mini | 2 | answer_to_stem_backward_check\|option_contradiction_check | 2 | 0.5000 | 3.5000 | 0.5000 | 0.5000 | 0.0000 |
| gpt-4o-mini | 3 | rule_or_principle_application | 2 | 1.0000 | 4.5000 | 1.0000 | 0.0000 | 0.0000 |
| gpt-4o-mini | 4 | decomposition\|stem_evidence_alignment | 2 | 1.0000 | 5.0000 | 1.0000 | 0.0000 | 0.0000 |
| qwen2.5-7b-instruct | 0 | concept_definition_match | 2 | 0.0000 | 1.5000 | 0.0000 | 1.0000 | 0.0000 |
| qwen2.5-7b-instruct | 2 | answer_to_stem_backward_check\|option_contradiction_check | 2 | 1.0000 | 4.5000 | 1.0000 | 0.0000 | 0.0000 |
| qwen2.5-7b-instruct | 3 | rule_or_principle_application | 2 | 0.5000 | 3.5000 | 0.5000 | 0.5000 | 0.0000 |
| qwen2.5-7b-instruct | 4 | decomposition\|stem_evidence_alignment | 2 | 0.0000 | 3.0000 | 0.0000 | 0.0000 | 1.0000 |

## 结果解读

这组评审直接回答“模型是否遵循了原始策略指令”。整体上，GPT-5.5 认为 40 条高风险样本中有 29 条遵循了目标策略，`followed_rate=0.7250`；如果把部分遵循也算作有实质策略响应，则 35/40 达到 `adherence_score >= 3`，`partial_or_better=0.8750`。平均遵循分数为 3.9000。

这说明低 `mixed exact target hit` 不能简单解释为“模型完全不听策略 prompt”。在这些原 judge primary 全部为 `option_contrast` 的样本里，GPT-5.5 反而多数认为 trace 实际上遵循或部分遵循了原策略。因此，与 normal-judge 复判结果结合看，主要问题更像是自动 judge/taxonomy 把 MMLU 多选题的选项比较表面形式过度吸附成 `option_contrast`，尤其是把 secondary 或格式性选项检查提升成 primary。

按目标策略看，`answer_to_stem_backward_check|option_contradiction_check` 和 `decomposition|stem_evidence_alignment` 的遵循最好，followed 分别为 0.9000 和 0.8000，平均分为 4.1000 和 4.2000。`concept_definition_match` 也有 0.7000 的 followed，平均分 3.7000。最弱的是 `rule_or_principle_application`，followed 只有 0.5000，平均分 3.6000，ambiguous 也最高，说明该指令更容易退化成“用领域事实逐项排除选项”，而不是明确先提出规则/原则再应用。

按模型看，deepseek-chat 和 gemini-2.5-flash-lite 的 followed 都是 0.8333，gpt-4o-mini 为 0.7500，qwen2.5-7b-instruct 只有 0.3750。qwen 的问题主要来自短答、无可见推理、或没有体现目标方法；这类样本应归为模型/prompt 遵循不足，而不是 judge 误判。

因此，综合诊断是：多数高风险样本属于 `judge_taxonomy_likely`，即“模型有策略响应，但原 judge 贴成了 option_contrast”；少数属于 `model_prompt_likely`，即模型确实没有按目标策略做；还有一部分 ambiguous，通常是目标方法出现了，但不是主导组织结构，或者 trace 太短/格式不完整。

对后续论文或实验报告来说，这个结果很关键：它说明 P3 里 mixed exact target hit 偏低并不等价于策略干预失败，而是混合了三种因素：一是 judge 对 `option_contrast` 的 primary 吸附；二是 taxonomy 中一些 MMLU option-semantics 标签边界过近；三是部分模型，尤其 qwen，在显式策略指令下仍可能输出短答或普通选项排除。
