# P3 GPT-5.5 Normal Taxonomy Judge 复核

这个实验让 GPT-5.5 扮演与正式 taxonomy judge 尽量相同的角色：它看到 taxonomy、major tree、标签定义、trace 和正常 judge 需要的上下文，但不看目标策略、模型身份、run 名或原自动 judge 标签。
因此它比 prompt-following 复核更接近正式指标本身，应作为判断 judge/taxonomy 是否存在系统偏差的主证据。

- 样本数：776
- 数据来源：`prove_experiments\p3_normal_judge_gpt55\p3_normal_judge_analysis_rows.csv`

## 指标中文含义

| 指标 | 中文含义 |
|---|---|
| `judge_primary option` | 原自动 judge 把 trace 的 primary 判为 `option_contrast` 的比例。 |
| `judge pair option` | 原自动 judge 把 primary 或 secondary 任一策略判为 `option_contrast` 的比例。 |
| `judge target exact` | 原自动 judge 的 primary 或 secondary leaf 精确命中 prompt 目标 leaf 的比例。 |
| `judge target same-major` | 原自动 judge 的 primary 或 secondary 所属主类命中目标主类的比例。 |
| `GPT primary option` | GPT-5.5 把 trace 的主策略判为 `option_contrast` 的比例。 |
| `GPT pair option` | GPT-5.5 把 primary 或 secondary 任一策略判为 `option_contrast` 的比例。 |
| `GPT target exact` | GPT-5.5 的 primary 或 secondary leaf 精确命中 prompt 目标 leaf 的比例。 |
| `GPT target same-major` | GPT-5.5 的 primary 或 secondary 所属主类命中目标主类的比例。 |
| `judge/taxonomy questioned` | GPT-5.5 不支持原自动 judge 的 `option_contrast` 主判定的比例。 |
| `confidence` | GPT-5.5 对自己 taxonomy 标签判断的平均置信度。 |

## 总体结果

| n | judge primary option | judge pair option | judge target exact | judge target same-major | GPT primary option | GPT pair option | GPT target exact | GPT target same-major | judge/taxonomy questioned | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 776 | 1.0000 | 1.0000 | 0.0979 | 0.2307 | 0.2178 | 0.4845 | 0.2088 | 0.2590 | 0.7822 | 0.9169 |

## 按策略汇总

| agent | target | prompt | n | judge exact | GPT exact | judge same-major | GPT same-major | judge primary option | GPT primary option | judge/taxonomy questioned |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 179 | 0.4246 | 0.7877 | 1.0000 | 0.9609 | 1.0000 | 0.2011 | 0.7989 |
| 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 117 | 0.0000 | 0.1709 | 0.0000 | 0.2222 | 1.0000 | 0.2479 | 0.7521 |
| 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 131 | 0.0000 | 0.0076 | 0.0000 | 0.0076 | 1.0000 | 0.2748 | 0.7252 |
| 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 171 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2456 | 0.7544 |
| 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 178 | 0.0000 | 0.0000 | 0.0000 | 0.0112 | 1.0000 | 0.1461 | 0.8539 |

## 按模型展开

| model | agent | target | prompt | n | judge exact | GPT exact | judge same-major | GPT same-major | judge primary option | GPT primary option | judge/taxonomy questioned |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 34 | 0.5000 | 0.8824 | 1.0000 | 1.0000 | 1.0000 | 0.1471 | 0.8529 |
| deepseek-chat | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 17 | 0.0000 | 0.1765 | 0.0000 | 0.2353 | 1.0000 | 0.2941 | 0.7059 |
| deepseek-chat | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6000 | 0.4000 |
| deepseek-chat | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 19 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2105 | 0.7895 |
| deepseek-chat | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 35 | 0.0000 | 0.0000 | 0.0000 | 0.0286 | 1.0000 | 0.1714 | 0.8286 |
| gemini-2.5-flash-lite | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 39 | 0.0769 | 0.6667 | 1.0000 | 1.0000 | 1.0000 | 0.4103 | 0.5897 |
| gemini-2.5-flash-lite | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 39 | 0.0000 | 0.1795 | 0.0000 | 0.2051 | 1.0000 | 0.3333 | 0.6667 |
| gemini-2.5-flash-lite | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 39 | 0.0000 | 0.0256 | 0.0000 | 0.0256 | 1.0000 | 0.2821 | 0.7179 |
| gemini-2.5-flash-lite | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4103 | 0.5897 |
| gemini-2.5-flash-lite | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0256 | 1.0000 | 0.2564 | 0.7436 |
| gpt-4o-mini | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 39 | 0.4615 | 0.7179 | 1.0000 | 0.8974 | 1.0000 | 0.1282 | 0.8718 |
| gpt-4o-mini | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 36 | 0.0000 | 0.2500 | 0.0000 | 0.3333 | 1.0000 | 0.1389 | 0.8611 |
| gpt-4o-mini | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2821 | 0.7179 |
| gpt-4o-mini | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2564 | 0.7436 |
| gpt-4o-mini | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1026 | 0.8974 |
| qwen2.5-7b-instruct | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 28 | 0.5714 | 0.8214 | 1.0000 | 0.8929 | 1.0000 | 0.1071 | 0.8929 |
| qwen2.5-7b-instruct | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2500 | 0.7500 |
| qwen2.5-7b-instruct | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2143 | 0.7857 |
| qwen2.5-7b-instruct | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 39 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1795 | 0.8205 |
| qwen2.5-7b-instruct | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 33 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0303 | 0.9697 |
| qwen3.5-plus | 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 39 | 0.5641 | 0.8718 | 1.0000 | 1.0000 | 1.0000 | 0.1795 | 0.8205 |
| qwen3.5-plus | 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 13 | 0.0000 | 0.0769 | 0.0000 | 0.1538 | 1.0000 | 0.2308 | 0.7692 |
| qwen3.5-plus | 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 34 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2353 | 0.7647 |
| qwen3.5-plus | 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 35 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1429 | 0.8571 |
| qwen3.5-plus | 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 32 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1562 | 0.8438 |

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

- 如果 GPT-5.5 仍大量支持 `option_contrast`，说明原自动 judge 的选项比较判定并不只是弱模型噪声，而是 trace 本身确实呈现出强选项比较结构。
- 如果 GPT-5.5 的 `GPT target exact` 或 `GPT target same-major` 明显高于自动 judge，则说明自动 judge/taxonomy 对 `option_contrast` 有过强吸附，需要降低 leaf exact 的证据权重。
- 这一复核优先于 prompt-following 复核，因为它与正式 taxonomy judge 使用同类信息和同类任务定义。
