# P3 GPT-5.5 Prompt Following 复核

这个实验让 GPT-5.5 只看原始策略指令、题目片段和 trace，直接判断模型是否遵循了策略 prompt。
它不等价于正式 taxonomy judge，但可以补充回答：自动 judge 判到 `option_contrast` 时，模型到底是没有听 prompt，还是 trace 表面仍像选项比较。

- 样本数：40
- 数据来源：`prove_experiments\p3_prompt_following_gpt55\p3_prompt_following_analysis_rows.csv`

## 指标中文含义

| 指标 | 中文含义 |
|---|---|
| `followed_rate` | GPT-5.5 判断 trace 基本遵循原策略指令的比例。 |
| `mean_score` | 1-5 分平均遵循度，1 表示明显不遵循，5 表示强遵循。 |
| `partial_or_better` | 遵循度不低于 3 分的比例，即至少部分遵循。 |
| `judge_taxonomy_likely` | 更像 judge/taxonomy 把 trace 过度吸附到 `option_contrast` 的比例。 |
| `model_prompt_likely` | 更像模型或 prompt 本身没有稳定诱导目标策略的比例。 |
| `ambiguous` | 证据不足或两种解释都可能的比例。 |

## 总体结果

| n | followed_rate | mean_score | partial_or_better | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|
| 40 | 0.6000 | 3.7250 | 0.8000 | 0.6000 | 0.2250 | 0.1750 |

## 按目标策略

| agent | target | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination | 8 | 0.8750 | 4.6250 | 0.8750 | 0.0000 | 0.1250 |
| 1 | rule_or_principle_application | 8 | 0.2500 | 3.0000 | 0.2500 | 0.5000 | 0.2500 |
| 2 | decomposition | 8 | 0.8750 | 4.3750 | 0.8750 | 0.1250 | 0.0000 |
| 3 | case_analysis | 8 | 0.8750 | 4.1250 | 0.8750 | 0.0000 | 0.1250 |
| 4 | edge_case_analysis | 8 | 0.1250 | 2.5000 | 0.1250 | 0.5000 | 0.3750 |

## 按模型和目标策略

| model | agent | target | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | distractor_elimination | 2 | 1.0000 | 5.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek-chat | 1 | rule_or_principle_application | 2 | 0.0000 | 3.0000 | 0.0000 | 0.5000 | 0.5000 |
| deepseek-chat | 2 | decomposition | 2 | 1.0000 | 5.0000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek-chat | 3 | case_analysis | 2 | 1.0000 | 4.5000 | 1.0000 | 0.0000 | 0.0000 |
| deepseek-chat | 4 | edge_case_analysis | 2 | 0.0000 | 2.0000 | 0.0000 | 0.5000 | 0.5000 |
| gemini-2.5-flash-lite | 0 | distractor_elimination | 2 | 1.0000 | 4.5000 | 1.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-lite | 1 | rule_or_principle_application | 2 | 0.5000 | 3.5000 | 0.5000 | 0.0000 | 0.5000 |
| gemini-2.5-flash-lite | 2 | decomposition | 2 | 0.5000 | 3.0000 | 0.5000 | 0.5000 | 0.0000 |
| gemini-2.5-flash-lite | 3 | case_analysis | 2 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-lite | 4 | edge_case_analysis | 2 | 0.0000 | 2.5000 | 0.0000 | 0.5000 | 0.5000 |
| gpt-4o-mini | 0 | distractor_elimination | 2 | 1.0000 | 5.0000 | 1.0000 | 0.0000 | 0.0000 |
| gpt-4o-mini | 1 | rule_or_principle_application | 2 | 0.5000 | 3.5000 | 0.5000 | 0.5000 | 0.0000 |
| gpt-4o-mini | 2 | decomposition | 2 | 1.0000 | 5.0000 | 1.0000 | 0.0000 | 0.0000 |
| gpt-4o-mini | 3 | case_analysis | 2 | 1.0000 | 4.0000 | 1.0000 | 0.0000 | 0.0000 |
| gpt-4o-mini | 4 | edge_case_analysis | 2 | 0.5000 | 3.5000 | 0.5000 | 0.0000 | 0.5000 |
| qwen2.5-7b-instruct | 0 | distractor_elimination | 2 | 0.5000 | 4.0000 | 0.5000 | 0.0000 | 0.5000 |
| qwen2.5-7b-instruct | 1 | rule_or_principle_application | 2 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | 0.0000 |
| qwen2.5-7b-instruct | 2 | decomposition | 2 | 1.0000 | 4.5000 | 1.0000 | 0.0000 | 0.0000 |
| qwen2.5-7b-instruct | 3 | case_analysis | 2 | 0.5000 | 4.0000 | 0.5000 | 0.0000 | 0.5000 |
| qwen2.5-7b-instruct | 4 | edge_case_analysis | 2 | 0.0000 | 2.0000 | 0.0000 | 1.0000 | 0.0000 |

## 结论读法

- 如果 `followed_rate` 高，而 normal taxonomy judge 仍判 `option_contrast`，则更支持 judge/taxonomy 吸附解释。
- 如果 `followed_rate` 低，则更支持模型或 prompt 遵循能力不足解释。
- 该实验只作为补充证据，因为它没有让 GPT-5.5 执行正式 taxonomy judge 的完整任务。
