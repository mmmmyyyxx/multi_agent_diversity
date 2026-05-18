# P3 GPT-5.5 Normal Taxonomy Judge 复核

这个实验让 GPT-5.5 扮演与正式 taxonomy judge 尽量相同的角色：它看到 taxonomy、major tree、标签定义、trace 和正常 judge 需要的上下文，但不看目标策略、模型身份、run 名或原自动 judge 标签。
因此它比 prompt-following 复核更接近正式指标本身，应作为判断 judge/taxonomy 是否存在系统偏差的主证据。

- 样本数：40
- 数据来源：`prove_experiments\p3_normal_judge_gpt55\p3_normal_judge_analysis_rows.csv`

## 指标中文含义

| 指标 | 中文含义 |
|---|---|
| `GPT primary option` | GPT-5.5 把 trace 的主策略判为 `option_contrast` 的比例。 |
| `GPT pair option` | GPT-5.5 把 `primary` 或 `secondary` 任一策略判为 `option_contrast` 的比例。 |
| `GPT target exact` | GPT-5.5 的 `primary` 或 `secondary` leaf 精确命中 prompt 目标 leaf 的比例。 |
| `GPT target same-major` | GPT-5.5 的 `primary` 或 `secondary` 所属主类命中目标主类的比例。 |
| `original judge supported` | GPT-5.5 也支持原自动 judge 的 `option_contrast` 主判定的比例。 |
| `judge/taxonomy questioned` | GPT-5.5 不支持原自动 judge 的 `option_contrast` 主判定的比例。 |
| `confidence` | GPT-5.5 对自己 taxonomy 标签判断的平均置信度。 |

## 总体结果

| n | GPT primary option | GPT pair option | GPT target exact | GPT target same-major | original judge supported | judge/taxonomy questioned | confidence |
|---|---|---|---|---|---|---|---|
| 40 | 0.2750 | 0.3500 | 0.2250 | 0.3000 | 0.2750 | 0.7250 | 0.9008 |

## 按目标策略

| agent | target | n | GPT primary option | GPT pair option | GPT target exact | GPT target same-major | judge/taxonomy questioned |
|---|---|---|---|---|---|---|---|
| 0 | distractor_elimination | 8 | 0.2500 | 0.3750 | 0.7500 | 1.0000 | 0.7500 |
| 1 | rule_or_principle_application | 8 | 0.2500 | 0.3750 | 0.2500 | 0.3750 | 0.7500 |
| 2 | decomposition | 8 | 0.2500 | 0.3750 | 0.1250 | 0.1250 | 0.7500 |
| 3 | case_analysis | 8 | 0.3750 | 0.3750 | 0.0000 | 0.0000 | 0.6250 |
| 4 | edge_case_analysis | 8 | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 0.7500 |

## 按模型和目标策略

| model | agent | target | n | GPT primary option | GPT target exact | GPT target same-major | judge/taxonomy questioned |
|---|---|---|---|---|---|---|---|
| deepseek-chat | 0 | distractor_elimination | 2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| deepseek-chat | 1 | rule_or_principle_application | 2 | 0.0000 | 0.5000 | 1.0000 | 1.0000 |
| deepseek-chat | 2 | decomposition | 2 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| deepseek-chat | 3 | case_analysis | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| deepseek-chat | 4 | edge_case_analysis | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| gemini-2.5-flash-lite | 0 | distractor_elimination | 2 | 0.5000 | 1.0000 | 1.0000 | 0.5000 |
| gemini-2.5-flash-lite | 1 | rule_or_principle_application | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| gemini-2.5-flash-lite | 2 | decomposition | 2 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| gemini-2.5-flash-lite | 3 | case_analysis | 2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| gemini-2.5-flash-lite | 4 | edge_case_analysis | 2 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| gpt-4o-mini | 0 | distractor_elimination | 2 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| gpt-4o-mini | 1 | rule_or_principle_application | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| gpt-4o-mini | 2 | decomposition | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| gpt-4o-mini | 3 | case_analysis | 2 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| gpt-4o-mini | 4 | edge_case_analysis | 2 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| qwen2.5-7b-instruct | 0 | distractor_elimination | 2 | 0.5000 | 0.0000 | 1.0000 | 0.5000 |
| qwen2.5-7b-instruct | 1 | rule_or_principle_application | 2 | 0.5000 | 0.0000 | 0.0000 | 0.5000 |
| qwen2.5-7b-instruct | 2 | decomposition | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| qwen2.5-7b-instruct | 3 | case_analysis | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| qwen2.5-7b-instruct | 4 | edge_case_analysis | 2 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## 结论读法

- 如果 GPT-5.5 仍大量支持 `option_contrast`，说明原自动 judge 的选项比较判定并不只是弱模型噪声，而是 trace 本身确实呈现出强选项比较结构。
- 如果 GPT-5.5 的 `GPT target exact` 或 `GPT target same-major` 明显高于自动 judge，则说明自动 judge/taxonomy 对 `option_contrast` 有过强吸附，需要降低 leaf exact 的证据权重。
- 这一复核优先于 prompt-following 复核，因为它与正式 taxonomy judge 使用同类信息和同类任务定义。
