# P3 GPT-5.5 综合结论

本文合并两类 GPT-5.5 复核。证据优先级是：Normal Taxonomy Judge 为主，Prompt Following 为补充。

## 证据优先级

1. **Normal Taxonomy Judge**：GPT-5.5 得到与正式 judge 尽量相同的信息，直接重新给 trace 打 taxonomy 标签。这是更强证据。
2. **Prompt Following**：GPT-5.5 只看原始策略指令和 trace，判断是否遵循 prompt。这是辅助诊断，用来区分 judge/taxonomy 吸附和模型/prompt 遵循不足。

## 核心统计

| normal n | judge exact | GPT exact | judge same-major | GPT same-major | judge primary option | GPT primary option | judge/taxonomy questioned |
|---|---|---|---|---|---|---|---|
| 40 | 0.0500 | 0.2250 | 0.2000 | 0.3000 | 1.0000 | 0.2750 | 0.7250 |


| prompt n | followed rate | mean score | judge taxonomy likely | model prompt likely | ambiguous |
|---|---|---|---|---|---|
| 40 | 0.6000 | 3.7250 | 0.6000 | 0.2250 | 0.1750 |

## 按策略联合对照

这一表把原自动 judge 的目标策略命中、GPT-5.5 taxonomy rejudge 的目标策略命中、以及 GPT-5.5 prompt-following 诊断放在同一处。`target exact` 是目标 leaf 精确命中，`same-major` 是目标主类命中。

| agent | target | prompt excerpt | normal n | judge exact | GPT exact | judge same-major | GPT same-major | judge primary option | GPT primary option | prompt n | GPT followed | mean score | judge taxonomy likely | model prompt likely | ambiguous |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination | Use a distractor-elimination strategy. Treat each answer choice as a candidate, reject choices that conflict with the stem, and keep the best-supported remaining choice. End wi ... | 8 | 0.2500 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.2500 | 8 | 0.8750 | 4.6250 | 0.8750 | 0.0000 | 0.1250 |
| 1 | rule_or_principle_application | Use a domain-rule strategy. First state the governing rule, theorem, principle, mechanism, or domain law, then apply that rule to the facts in the stem before choosing. End wit ... | 8 | 0.0000 | 0.2500 | 0.0000 | 0.3750 | 1.0000 | 0.2500 | 8 | 0.2500 | 3.0000 | 0.2500 | 0.5000 | 0.2500 |
| 2 | decomposition | Use a decomposition strategy. Break the stem into key facts, constraints, and sub-questions, solve each part in order, and combine the parts into one answer. End with exactly o ... | 8 | 0.0000 | 0.1250 | 0.0000 | 0.1250 | 1.0000 | 0.2500 | 8 | 0.8750 | 4.3750 | 0.8750 | 0.1250 | 0.0000 |
| 3 | case_analysis | Use a case-analysis strategy. Enumerate the relevant cases, conditions, or scenarios implied by the stem, test each case for consistency, and choose the answer that survives th ... | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3750 | 8 | 0.8750 | 4.1250 | 0.8750 | 0.0000 | 0.1250 |
| 4 | edge_case_analysis | Use an edge-case and exception-checking strategy. Look for boundary conditions, qualifiers, extreme cases, or exceptions in the stem, then decide which answer remains valid und ... | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2500 | 8 | 0.1250 | 2.5000 | 0.1250 | 0.5000 | 0.3750 |

## 策略 prompt 原文

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


## 综合判断

- 优先看 normal taxonomy judge：GPT-5.5 并没有大规模继续支持原自动 judge 的 `option_contrast` 主判定，说明自动 judge/taxonomy 存在明显的 `option_contrast` 吸附风险。
- 再看 prompt-following：GPT-5.5 认为 60.00% 的抽样 trace 基本遵循了原始策略指令，80.00% 至少部分遵循。这进一步说明 leaf exact hit 偏低不能直接等同于模型完全不听策略 prompt。
- 同时，prompt-following 也显示不同策略可执行性不均衡：`distractor_elimination`、`decomposition`、`case_analysis` 更容易被执行；`rule_or_principle_application` 和 `edge_case_analysis` 更弱。
- 因此，P3 的主证据应是 team-level diversity、major diversity 和 homogeneity 的系统变化；exact target hit 更适合作为诊断指标，而不是最终有效性的唯一标准。
