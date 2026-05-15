# P3/P4 目标策略命中率拆解

本分析把 `target_exact_hit_rate` 按目标策略、agent 和模型拆开。`exact_hit` 表示 primary 或 secondary leaf 标签命中 prompt 中声明的目标 leaf；`same_major_any_hit` 表示 primary 或 secondary 至少落入目标 major family。

## 指标中文含义

| 指标/列名 | 中文含义 | 读数方向 |
|---|---|---|
| `agent` / `agent_id` | 五个 agent 中的编号，0-4 分别对应 prompt 文件里的五条策略指令。 | 用来定位是哪条策略指令。 |
| `target leaf` / `target_label` | prompt 显式要求该 agent 使用的目标细粒度策略标签。多个标签用 `\|` 连接，表示命中任意一个都算 exact hit。 | 不是分数，是目标定义。 |
| `target major` / `target_major_label` | 目标 leaf 标签映射到的粗粒度 major family。多个 major 用 `\|` 连接。 | 用来计算宽松命中。 |
| `n` | 该行统计的 agent-question 样本数。整体表中通常为 4 个模型 × 100 题 = 400。 | 越大越稳定。 |
| `exact` / `exact_hit_rate` | primary 或 secondary leaf 标签是否精确命中目标 leaf 的比例。 | 越高表示越严格符合指定细策略。 |
| `same-major(any)` / `same_major_any_hit_rate` | primary 或 secondary 的 major family 是否落入目标 major family 的比例。 | 越高表示至少落入相近粗策略。 |
| `top primary` | 该组样本中最常见的 primary leaf 标签。 | 用来判断模型实际最常表现出的策略。 |
| `top primary share` | 最常见 primary leaf 标签所占比例。 | 越高表示该 agent 的输出越被单一策略形态支配。 |
| `top secondary` | 该组样本中最常见的 secondary leaf 标签。 | 用来观察辅助策略或次要策略。 |
| `agent acc` / `agent_answer_acc` | 单个 agent 自己答案的准确率，不是五 agent 投票准确率。 | 越高表示该 agent 答题越准。 |
| `primary_exact_hit_rate` | 只看 primary leaf 是否精确命中目标 leaf。 | 比 `exact` 更严格。 |
| `secondary_exact_hit_rate` | 只看 secondary leaf 是否精确命中目标 leaf。 | 用来判断目标策略是否退到次策略位置。 |
| `same_major_primary_hit_rate` | 只看 primary major 是否命中目标 major。 | 用来判断主策略大类是否对齐。 |
| `top_primary_major` | 最常见的 primary major family。 | 用来判断输出主要落在哪个粗策略大类。 |
| `top_primary_major_share` | 最常见 primary major family 所占比例。 | 越高表示粗策略越集中。 |
| `primary_counts_json` | primary leaf 标签的完整计数字典。 | 用于追查被哪些标签吸走。 |
| `secondary_counts_json` | secondary leaf 标签的完整计数字典。 | 用于追查目标策略是否作为次策略出现。 |

注意：这里的 `exact` 是“策略树标签是否命中目标 leaf”，不是“答案是否正确”。答案正确率看 `agent acc`。

## Mixed 策略目标整体情况

| agent | target leaf | target major | n | exact | same-major(any) | top primary | top primary share | top secondary | agent acc |
|---|---|---|---|---|---|---|---|---|---|
| 0 | concept_definition_match | mmlu_option_semantics | 400 | 0.1400 | 0.7000 | option_contrast | 0.4525 | option_contrast | 0.7750 |
| 1 | distractor_elimination\|option_contrast | mmlu_option_semantics | 400 | 0.6600 | 0.7375 | option_contrast | 0.5500 | option_contrast | 0.7500 |
| 2 | answer_to_stem_backward_check\|option_contradiction_check | mmlu_option_semantics | 400 | 0.1750 | 0.7975 | option_contrast | 0.6100 | option_contrast | 0.7325 |
| 3 | rule_or_principle_application | mmlu_domain_reasoning | 400 | 0.0375 | 0.1800 | option_contrast | 0.3675 | option_contrast | 0.7700 |
| 4 | decomposition\|stem_evidence_alignment | mmlu_option_semantics\|representation_formalization | 400 | 0.2525 | 0.8525 | option_contrast | 0.3750 | option_contrast | 0.7700 |

## Same-elimination 对照整体情况

| agent | target leaf | n | exact | same-major(any) | top primary | top primary share |
|---|---|---|---|---|---|---|
| 0 | distractor_elimination\|option_contrast | 400 | 0.6725 | 0.7450 | option_contrast | 0.5625 |
| 1 | distractor_elimination\|option_contrast | 400 | 0.6550 | 0.7325 | option_contrast | 0.5425 |
| 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 400 | 0.6675 | 0.7450 | option_contrast | 0.5500 |
| 3 | distractor_elimination\|option_contrast | 400 | 0.7575 | 0.8050 | option_contrast | 0.6375 |
| 4 | distractor_elimination\|option_contrast | 400 | 0.6800 | 0.7625 | option_contrast | 0.5775 |

## Mixed 策略按模型拆解

| model | agent | target leaf | exact | same-major(any) | top primary | top primary share |
|---|---|---|---|---|---|---|
| deepseek-chat | 0 | concept_definition_match | 0.1400 | 0.4300 | decomposition | 0.3500 |
| deepseek-chat | 1 | distractor_elimination\|option_contrast | 0.5200 | 0.5900 | option_contrast | 0.4100 |
| deepseek-chat | 2 | answer_to_stem_backward_check\|option_contradiction_check | 0.1000 | 0.7600 | option_contrast | 0.6000 |
| deepseek-chat | 3 | rule_or_principle_application | 0.0400 | 0.1900 | option_contrast | 0.2400 |
| deepseek-chat | 4 | decomposition\|stem_evidence_alignment | 0.5500 | 0.9100 | decomposition | 0.5500 |
| gemini-2.5-flash-lite | 0 | concept_definition_match | 0.1000 | 0.7700 | option_contrast | 0.5900 |
| gemini-2.5-flash-lite | 1 | distractor_elimination\|option_contrast | 0.7700 | 0.8100 | option_contrast | 0.6100 |
| gemini-2.5-flash-lite | 2 | answer_to_stem_backward_check\|option_contradiction_check | 0.3200 | 0.8000 | option_contrast | 0.5900 |
| gemini-2.5-flash-lite | 3 | rule_or_principle_application | 0.0300 | 0.1600 | option_contrast | 0.5500 |
| gemini-2.5-flash-lite | 4 | decomposition\|stem_evidence_alignment | 0.0600 | 0.8300 | option_contrast | 0.5500 |
| gpt-4o-mini | 0 | concept_definition_match | 0.1700 | 0.8100 | option_contrast | 0.5600 |
| gpt-4o-mini | 1 | distractor_elimination\|option_contrast | 0.7400 | 0.7900 | option_contrast | 0.6800 |
| gpt-4o-mini | 2 | answer_to_stem_backward_check\|option_contradiction_check | 0.0500 | 0.8800 | option_contrast | 0.7700 |
| gpt-4o-mini | 3 | rule_or_principle_application | 0.0500 | 0.1900 | option_contrast | 0.4000 |
| gpt-4o-mini | 4 | decomposition\|stem_evidence_alignment | 0.1100 | 0.8100 | option_contrast | 0.4600 |
| qwen2.5-7b-instruct | 0 | concept_definition_match | 0.1500 | 0.7900 | option_contrast | 0.4300 |
| qwen2.5-7b-instruct | 1 | distractor_elimination\|option_contrast | 0.6100 | 0.7600 | option_contrast | 0.5000 |
| qwen2.5-7b-instruct | 2 | answer_to_stem_backward_check\|option_contradiction_check | 0.2300 | 0.7500 | option_contrast | 0.4800 |
| qwen2.5-7b-instruct | 3 | rule_or_principle_application | 0.0300 | 0.1800 | concept_definition_match | 0.3400 |
| qwen2.5-7b-instruct | 4 | decomposition\|stem_evidence_alignment | 0.2900 | 0.8600 | decomposition | 0.2900 |

## 主要读法

- mixed exact 低不是单一原因。`option_contrast/distractor_elimination` 的 exact 明显较高，说明模型和 judge 对这种选项排除策略相对一致；`concept_definition_match`、`rule_or_principle_application`、`decomposition/stem_evidence_alignment` 明显较低，说明这些目标更容易被实际题型和 judge 标签边界冲淡。
- `same-major(any)` 往往高于 exact，说明不少 trace 没有命中目标 leaf，但仍落在相近的大类。这更像 taxonomy 粒度/标签边界问题，而不完全是模型不遵循。
- 对 `answer_to_stem_backward_check/option_contradiction_check`，如果 exact 和 same-major 都低，且 top primary 常落在普通 `option_contrast`，则说明 prompt 虽然要求 backward check，但模型实际常退化成普通选项比较，judge 也难以从 trace 中看到显式 backward-check 证据。
