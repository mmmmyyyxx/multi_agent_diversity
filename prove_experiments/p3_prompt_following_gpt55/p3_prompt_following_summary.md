# P3 GPT-5.5 Prompt Following 复核

这个实验让 GPT-5.5 只看原始策略指令、题目片段和 trace，直接判断模型是否遵循了策略 prompt。
它不等价于正式 taxonomy judge，但可以补充回答：自动 judge 判到 `option_contrast` 时，模型到底是没有听 prompt，还是 trace 表面仍像选项比较。

- 样本数：776
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
| 776 | 0.6869 | 3.8067 | 0.8067 | 0.6843 | 0.2281 | 0.0876 |

## 按策略汇总

| agent | target | prompt | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 179 | 0.7598 | 4.0559 | 0.7542 | 0.1508 | 0.0950 |
| 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 117 | 0.6239 | 3.6410 | 0.6154 | 0.2906 | 0.0940 |
| 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 131 | 0.6870 | 3.8244 | 0.6870 | 0.2519 | 0.0611 |
| 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 171 | 0.7602 | 3.9240 | 0.7602 | 0.1696 | 0.0702 |
| 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 178 | 0.5843 | 3.5393 | 0.5843 | 0.3034 | 0.1124 |

## 按模型展开

| model | agent | target | prompt | n | followed | mean_score | judge_taxonomy_likely | model_prompt_likely | ambiguous |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 34 | 0.8529 | 4.2647 | 0.8529 | 0.0882 | 0.0588 |
| deepseek-chat | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 17 | 0.5882 | 3.8235 | 0.5882 | 0.2353 | 0.1765 |
| deepseek-chat | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 5 | 0.8000 | 4.4000 | 0.8000 | 0.2000 | 0.0000 |
| deepseek-chat | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 19 | 0.6842 | 3.7368 | 0.6842 | 0.2105 | 0.1053 |
| deepseek-chat | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 35 | 0.4857 | 3.3714 | 0.4857 | 0.3143 | 0.2000 |
| gemini-2.5-flash-lite | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 39 | 0.6667 | 3.7949 | 0.6667 | 0.2051 | 0.1282 |
| gemini-2.5-flash-lite | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 39 | 0.6410 | 3.4872 | 0.6154 | 0.3077 | 0.0769 |
| gemini-2.5-flash-lite | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 39 | 0.6667 | 3.7949 | 0.6667 | 0.2051 | 0.1282 |
| gemini-2.5-flash-lite | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 39 | 0.6923 | 3.7949 | 0.6923 | 0.2051 | 0.1026 |
| gemini-2.5-flash-lite | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 39 | 0.7436 | 3.8718 | 0.7436 | 0.1795 | 0.0769 |
| gpt-4o-mini | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 39 | 0.6923 | 3.8974 | 0.6667 | 0.2051 | 0.1282 |
| gpt-4o-mini | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 36 | 0.6667 | 3.7778 | 0.6667 | 0.2778 | 0.0556 |
| gpt-4o-mini | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 39 | 0.6923 | 3.8205 | 0.6923 | 0.2821 | 0.0256 |
| gpt-4o-mini | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 39 | 0.7949 | 4.0513 | 0.7949 | 0.1282 | 0.0769 |
| gpt-4o-mini | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 39 | 0.5128 | 3.3590 | 0.5128 | 0.3590 | 0.1282 |
| qwen2.5-7b-instruct | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 28 | 0.7857 | 4.0714 | 0.7857 | 0.1786 | 0.0357 |
| qwen2.5-7b-instruct | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 12 | 0.5000 | 3.6667 | 0.5000 | 0.3333 | 0.1667 |
| qwen2.5-7b-instruct | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 14 | 0.7857 | 4.0000 | 0.7857 | 0.1429 | 0.0714 |
| qwen2.5-7b-instruct | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 39 | 0.8462 | 4.1282 | 0.8462 | 0.1282 | 0.0256 |
| qwen2.5-7b-instruct | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 33 | 0.5758 | 3.6061 | 0.5758 | 0.3333 | 0.0909 |
| qwen3.5-plus | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 39 | 0.8205 | 4.2821 | 0.8205 | 0.0769 | 0.1026 |
| qwen3.5-plus | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 13 | 0.6154 | 3.4615 | 0.6154 | 0.3077 | 0.0769 |
| qwen3.5-plus | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 34 | 0.6471 | 3.7059 | 0.6471 | 0.3235 | 0.0294 |
| qwen3.5-plus | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 35 | 0.7429 | 3.8000 | 0.7429 | 0.2000 | 0.0571 |
| qwen3.5-plus | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 32 | 0.5938 | 3.4688 | 0.5938 | 0.3438 | 0.0625 |

## 策略指令参考

- agent `0` / target `distractor_elimination`

```text
Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End with exactly one FINAL_ANSWER line.
```

- agent `1` / target `rule_or_principle_application`

```text
Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End with exactly one FINAL_ANSWER line.
```

- agent `2` / target `decomposition`

```text
Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly one FINAL_ANSWER line.
```

- agent `3` / target `case_analysis`

```text
Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives the case split. End with exactly one FINAL_ANSWER line.
```

- agent `4` / target `edge_case_analysis`

```text
Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid under those checks. End with exactly one FINAL_ANSWER line.
```

## 结论读法

- 如果 `followed_rate` 高，而 normal taxonomy judge 仍判 `option_contrast`，则更支持 judge/taxonomy 吸附解释。
- 如果 `followed_rate` 低，则更支持模型或 prompt 遵循能力不足解释。
- 该实验只作为补充证据，因为它没有让 GPT-5.5 执行正式 taxonomy judge 的完整任务。
