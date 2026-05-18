# P3 有效 Trace 质量分析

本文件只使用已存在的 P3 输出进行离线统计，不调用 API。有效 trace 的定义是：同一道题的 5 个 agent trace 全部非空、长度足够、包含 `FINAL_ANSWER`、没有明显重复，并且答案字段非空。

## 按 run 汇总

| run | model | condition | valid questions | valid rate | valid family_div | valid major_div | valid homogeneity | valid vote_acc | valid target exact | valid target same-major |
|---|---|---|---|---|---|---|---|---|---|---|
| P3_mixed_strategy_deepseek_chat_seed42 | deepseek-chat | mixed | 100/100 | 1.0000 | 0.5564 | 0.7584 | 0.4745 | 0.9100 | 0.2740 | 0.3760 |
| P3_mixed_strategy_gemini_flash_lite_seed42 | gemini-2.5-flash-lite | mixed | 99/100 | 0.9900 | 0.4546 | 0.3902 | 0.7575 | 0.8081 | 0.0687 | 0.2525 |
| P3_mixed_strategy_gpt4omini_seed42 | gpt-4o-mini | mixed | 100/100 | 1.0000 | 0.4807 | 0.4243 | 0.7157 | 0.8600 | 0.1360 | 0.2660 |
| P3_mixed_strategy_qwen25_7b_seed42 | qwen2.5-7b-instruct | mixed | 81/100 | 0.8100 | 0.6224 | 0.7362 | 0.4601 | 0.7654 | 0.2988 | 0.4074 |
| P3_same_elimination_deepseek_chat_seed42 | deepseek-chat | same | 100/100 | 1.0000 | 0.4563 | 0.5023 | 0.6625 | 0.9300 | 0.5240 | 0.5720 |
| P3_same_elimination_gemini_flash_lite_seed42 | gemini-2.5-flash-lite | same | 100/100 | 1.0000 | 0.4124 | 0.3162 | 0.8058 | 0.7900 | 0.7340 | 0.7800 |
| P3_same_elimination_gpt4omini_seed42 | gpt-4o-mini | same | 100/100 | 1.0000 | 0.3395 | 0.2156 | 0.8566 | 0.8600 | 0.8080 | 0.8320 |
| P3_same_elimination_qwen25_7b_seed42 | qwen2.5-7b-instruct | same | 31/100 | 0.3100 | 0.5538 | 0.5313 | 0.6095 | 0.6774 | 0.4968 | 0.5806 |

## 有效 trace 口径下的 paired 检验

| metric | paired n | mixed - same | 95% CI | Wilcoxon p |
|---|---|---|---|---|
| team_family_diversity | 329 | 0.0900 | [0.0665, 0.1146] | 0.0000 |
| team_family_homogeneity_rate | 329 | -0.1213 | [-0.1527, -0.0896] | 0.0000 |
| team_major_family_diversity | 329 | 0.1755 | [0.1325, 0.2200] | 0.0000 |

## 读法

- 如果有效 trace 口径下 mixed 仍高于 same，说明 P3 的提升不是由坏 trace 直接抬高。
- 如果某些模型有效题数较低，它们仍可用于趋势观察，但不应单独承担强结论。
- 正式结论应同时报告四模型总体结果、有效 trace 口径、目标命中拆解和 GPT-5.5 复核。
