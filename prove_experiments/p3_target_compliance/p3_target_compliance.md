# P3 目标策略命中分析

本分析把自动 taxonomy judge 给出的 `primary` / `secondary` 策略标签，与 prompt 中显式指定的目标策略进行对齐。
这里的命中率衡量的是“trace 被判到目标策略标签或目标主类的比例”，不是答案正确率。

## 指标中文含义

| 指标 | 中文含义 | 解读方向 |
|---|---|---|
| `agent_id` | 五个 agent 的编号，0-4 对应 prompt 文件中的五条策略指令。 | 用来定位是哪条策略指令。 |
| `target_label` / `target leaf` | prompt 显式要求使用的细粒度策略标签。多个标签用 `|` 连接。 | 这是目标定义，不是分数。 |
| `target_major_label` / `target major` | 目标 leaf 所属的主策略类。 | 用于判断粗粒度策略是否对齐。 |
| `n` | 该行统计的 agent-question 样本数。 | 越大越稳定。 |
| `exact_hit_rate` / `exact` | `primary` 或 `secondary` leaf 精确命中目标 leaf 的比例。 | 越高表示越严格遵循指定细策略。 |
| `primary_exact_hit_rate` | 只看 `primary` leaf 是否精确命中目标 leaf。 | 比 `exact` 更严格。 |
| `secondary_exact_hit_rate` | 只看 `secondary` leaf 是否精确命中目标 leaf。 | 用来判断目标策略是否退到次策略位置。 |
| `same_major_primary_hit_rate` | 只看 `primary` 所属主类是否命中目标主类。 | 衡量主策略方向是否对齐。 |
| `same_major_any_hit_rate` / `same-major(any)` | `primary` 或 `secondary` 所属主类是否命中目标主类。 | 衡量较宽松的粗粒度策略遵循。 |
| `top_primary` | 该组样本中最常见的 `primary` leaf。 | 用来观察实际 trace 最常被判成哪类策略。 |
| `top_primary_share` | 最常见 `primary` leaf 的占比。 | 越高说明越被单一策略形态支配。 |
| `top_secondary` | 该组样本中最常见的 `secondary` leaf。 | 用来观察辅助或次要策略。 |
| `top_primary_major` | 最常见 `primary` 所属主类。 | 用来观察粗粒度策略落点。 |
| `agent_answer_acc` / `agent acc` | 单个 agent 自己答案的准确率。 | 与策略命中不同，衡量答题对错。 |
| `primary_counts_json` | `primary` leaf 的完整计数字典。 | 用于追查被哪些标签吸走。 |
| `secondary_counts_json` | `secondary` leaf 的完整计数字典。 | 用于追查目标策略是否作为次策略出现。 |

## P3 目标策略设计确认

| agent | 目标 leaf | 所属主类 | 策略含义 |
|---|---|---|---|
| 0 | `distractor_elimination` | `mmlu_option_semantics` | 逐项排除干扰项，保留最符合题干的选项。 |
| 1 | `rule_or_principle_application` | `mmlu_domain_reasoning` | 先识别领域规则、定理、原则或机制，再把规则应用到题干。 |
| 2 | `decomposition` | `representation_formalization` | 把题干拆成事实、约束和子问题，再合并得到答案。 |
| 3 | `case_analysis` | `logical_proof` | 枚举相关条件、情形或分支，逐一检验。 |
| 4 | `edge_case_analysis` | `optimization_boundary_meta` | 检查边界条件、限定词、例外或极端情形。 |

这五个目标 leaf 分别属于五个不同主类，因此 P3 的 mixed 条件可以干净地检验“显式策略 prompt 是否能提高跨主类策略多样性”。

## Mixed 策略目标整体情况

| agent | target leaf | target major | n | exact | same-major(any) | top primary | top primary share | top secondary | agent acc |
|---|---|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination | mmlu_option_semantics | 500 | 0.2480 | 0.7040 | option_contrast | 0.4560 | option_contrast | 0.8300 |
| 1 | rule_or_principle_application | mmlu_domain_reasoning | 500 | 0.1780 | 0.3660 | option_contrast | 0.2520 | option_contrast | 0.8560 |
| 2 | decomposition | representation_formalization | 500 | 0.3260 | 0.3400 | decomposition | 0.3220 | option_contrast | 0.8360 |
| 3 | case_analysis | logical_proof | 500 | 0.1500 | 0.1700 | option_contrast | 0.4260 | option_contrast | 0.8500 |
| 4 | edge_case_analysis | optimization_boundary_meta | 500 | 0.0440 | 0.0440 | option_contrast | 0.4340 | option_contrast | 0.8480 |

## Same-elimination 对照整体情况

| agent | target leaf | n | exact | same-major(any) | top primary | top primary share | agent acc |
|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination\|option_contrast | 500 | 0.6480 | 0.7460 | option_contrast | 0.5460 | 0.8440 |
| 1 | distractor_elimination\|option_contrast | 500 | 0.6280 | 0.7360 | option_contrast | 0.4720 | 0.8280 |
| 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 500 | 0.6760 | 0.7560 | option_contrast | 0.5340 | 0.8520 |
| 3 | distractor_elimination\|option_contrast | 500 | 0.7280 | 0.7900 | option_contrast | 0.6320 | 0.8140 |
| 4 | distractor_elimination\|option_contrast | 500 | 0.6660 | 0.7700 | option_contrast | 0.5800 | 0.8320 |

## Mixed 策略按模型拆解

| model | agent | target leaf | target major | exact | same-major(any) | top primary | top primary share | agent acc |
|---|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | distractor_elimination | mmlu_option_semantics | 0.2200 | 0.4900 | decomposition | 0.3700 | 0.8900 |
| deepseek-chat | 1 | rule_or_principle_application | mmlu_domain_reasoning | 0.1500 | 0.3400 | concept_definition_match | 0.2200 | 0.9100 |
| deepseek-chat | 2 | decomposition | representation_formalization | 0.7800 | 0.7900 | decomposition | 0.7800 | 0.8800 |
| deepseek-chat | 3 | case_analysis | logical_proof | 0.1500 | 0.1900 | decomposition | 0.4000 | 0.9100 |
| deepseek-chat | 4 | edge_case_analysis | optimization_boundary_meta | 0.0700 | 0.0700 | option_contrast | 0.3500 | 0.9000 |
| gemini-2.5-flash-lite | 0 | distractor_elimination | mmlu_option_semantics | 0.0500 | 0.7700 | option_contrast | 0.5400 | 0.7000 |
| gemini-2.5-flash-lite | 1 | rule_or_principle_application | mmlu_domain_reasoning | 0.1000 | 0.2800 | option_contrast | 0.4800 | 0.8100 |
| gemini-2.5-flash-lite | 2 | decomposition | representation_formalization | 0.0900 | 0.1000 | option_contrast | 0.5300 | 0.7400 |
| gemini-2.5-flash-lite | 3 | case_analysis | logical_proof | 0.0900 | 0.1000 | option_contrast | 0.5900 | 0.7500 |
| gemini-2.5-flash-lite | 4 | edge_case_analysis | optimization_boundary_meta | 0.0100 | 0.0100 | option_contrast | 0.5700 | 0.7700 |
| gpt-4o-mini | 0 | distractor_elimination | mmlu_option_semantics | 0.2900 | 0.7800 | option_contrast | 0.6600 | 0.8600 |
| gpt-4o-mini | 1 | rule_or_principle_application | mmlu_domain_reasoning | 0.1600 | 0.2900 | option_contrast | 0.3600 | 0.8300 |
| gpt-4o-mini | 2 | decomposition | representation_formalization | 0.0800 | 0.0900 | option_contrast | 0.4600 | 0.8200 |
| gpt-4o-mini | 3 | case_analysis | logical_proof | 0.1500 | 0.1700 | option_contrast | 0.6000 | 0.8500 |
| gpt-4o-mini | 4 | edge_case_analysis | optimization_boundary_meta | 0.0000 | 0.0000 | option_contrast | 0.6000 | 0.8600 |
| qwen2.5-7b-instruct | 0 | distractor_elimination | mmlu_option_semantics | 0.3500 | 0.7200 | option_contrast | 0.2800 | 0.7500 |
| qwen2.5-7b-instruct | 1 | rule_or_principle_application | mmlu_domain_reasoning | 0.2600 | 0.4500 | rule_or_principle_application | 0.2600 | 0.7900 |
| qwen2.5-7b-instruct | 2 | decomposition | representation_formalization | 0.5400 | 0.5700 | decomposition | 0.5400 | 0.8000 |
| qwen2.5-7b-instruct | 3 | case_analysis | logical_proof | 0.0800 | 0.0900 | option_contrast | 0.4000 | 0.8000 |
| qwen2.5-7b-instruct | 4 | edge_case_analysis | optimization_boundary_meta | 0.1300 | 0.1300 | option_contrast | 0.3300 | 0.7600 |
| qwen3.5-plus | 0 | distractor_elimination | mmlu_option_semantics | 0.3300 | 0.7600 | option_contrast | 0.4600 | 0.9500 |
| qwen3.5-plus | 1 | rule_or_principle_application | mmlu_domain_reasoning | 0.2200 | 0.4700 | concept_definition_match | 0.2400 | 0.9400 |
| qwen3.5-plus | 2 | decomposition | representation_formalization | 0.1400 | 0.1500 | option_contrast | 0.3400 | 0.9400 |
| qwen3.5-plus | 3 | case_analysis | logical_proof | 0.2800 | 0.3000 | option_contrast | 0.3500 | 0.9400 |
| qwen3.5-plus | 4 | edge_case_analysis | optimization_boundary_meta | 0.0100 | 0.0100 | option_contrast | 0.3200 | 0.9500 |

## Same-elimination 对照按模型拆解

| model | agent | target leaf | exact | same-major(any) | top primary | top primary share | agent acc |
|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | distractor_elimination\|option_contrast | 0.5300 | 0.5900 | option_contrast | 0.4600 | 0.8900 |
| deepseek-chat | 1 | distractor_elimination\|option_contrast | 0.4500 | 0.4800 | decomposition | 0.4100 | 0.9000 |
| deepseek-chat | 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 0.5400 | 0.6000 | option_contrast | 0.4400 | 0.9200 |
| deepseek-chat | 3 | distractor_elimination\|option_contrast | 0.4900 | 0.5500 | option_contrast | 0.4300 | 0.8800 |
| deepseek-chat | 4 | distractor_elimination\|option_contrast | 0.6100 | 0.6400 | option_contrast | 0.5400 | 0.9000 |
| gemini-2.5-flash-lite | 0 | distractor_elimination\|option_contrast | 0.7400 | 0.7600 | option_contrast | 0.6500 | 0.7500 |
| gemini-2.5-flash-lite | 1 | distractor_elimination\|option_contrast | 0.7100 | 0.7700 | option_contrast | 0.5700 | 0.7700 |
| gemini-2.5-flash-lite | 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 0.7500 | 0.7900 | option_contrast | 0.6100 | 0.7800 |
| gemini-2.5-flash-lite | 3 | distractor_elimination\|option_contrast | 0.7600 | 0.8200 | option_contrast | 0.6100 | 0.7600 |
| gemini-2.5-flash-lite | 4 | distractor_elimination\|option_contrast | 0.7100 | 0.7600 | option_contrast | 0.6000 | 0.7300 |
| gpt-4o-mini | 0 | distractor_elimination\|option_contrast | 0.7800 | 0.7900 | option_contrast | 0.7200 | 0.8600 |
| gpt-4o-mini | 1 | distractor_elimination\|option_contrast | 0.7900 | 0.8300 | option_contrast | 0.6900 | 0.8500 |
| gpt-4o-mini | 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 0.8100 | 0.8500 | option_contrast | 0.7100 | 0.8600 |
| gpt-4o-mini | 3 | distractor_elimination\|option_contrast | 0.8300 | 0.8400 | option_contrast | 0.8000 | 0.8500 |
| gpt-4o-mini | 4 | distractor_elimination\|option_contrast | 0.8300 | 0.8500 | option_contrast | 0.7400 | 0.8700 |
| qwen2.5-7b-instruct | 0 | distractor_elimination\|option_contrast | 0.5400 | 0.8300 | option_contrast | 0.4600 | 0.7700 |
| qwen2.5-7b-instruct | 1 | distractor_elimination\|option_contrast | 0.4900 | 0.7900 | option_contrast | 0.3400 | 0.7100 |
| qwen2.5-7b-instruct | 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 0.6200 | 0.8000 | option_contrast | 0.4800 | 0.7600 |
| qwen2.5-7b-instruct | 3 | distractor_elimination\|option_contrast | 0.8900 | 0.9500 | option_contrast | 0.7700 | 0.6500 |
| qwen2.5-7b-instruct | 4 | distractor_elimination\|option_contrast | 0.5400 | 0.8400 | option_contrast | 0.5200 | 0.7400 |
| qwen3.5-plus | 0 | distractor_elimination\|option_contrast | 0.6500 | 0.7600 | option_contrast | 0.4400 | 0.9500 |
| qwen3.5-plus | 1 | distractor_elimination\|option_contrast | 0.7000 | 0.8100 | option_contrast | 0.4200 | 0.9100 |
| qwen3.5-plus | 2 | distractor_elimination\|option_contrast\|option_contradiction_check | 0.6600 | 0.7400 | option_contrast | 0.4300 | 0.9400 |
| qwen3.5-plus | 3 | distractor_elimination\|option_contrast | 0.6700 | 0.7900 | option_contrast | 0.5500 | 0.9300 |
| qwen3.5-plus | 4 | distractor_elimination\|option_contrast | 0.6400 | 0.7600 | option_contrast | 0.5000 | 0.9200 |

## 主要读法

- `exact` 高，说明 trace 的细粒度 leaf 标签更接近目标策略；`same-major(any)` 高，说明至少在主类层面落到了目标策略方向。
- mixed 条件下如果 `exact` 较低但 `same-major(any)` 较高，说明模型可能没有稳定执行到指定 leaf，但已经发生了较粗粒度的策略转向。
- 如果 `top_primary` 大量集中到 `option_contrast`，说明 MMLU 多选题形式和 judge 的 primary 规则仍会把不同策略吸到选项比较形态；这需要结合 GPT-5.5 normal taxonomy judge 与 prompt-following 复核一起解释。
- P3 的关键证据不是单个 leaf 是否完全可控，而是 same 与 mixed 在 team-level diversity、major diversity 和同质性上的系统差异。
