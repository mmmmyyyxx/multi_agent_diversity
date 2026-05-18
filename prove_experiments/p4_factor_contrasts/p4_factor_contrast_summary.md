# P4 模型身份与 Prompt 因子对比

本分析把 P4 分成四种对比。`same_model_same_prompt` 用同一个 run 内的 team 主类多样性表示；其余三类用同一道题下两个 team 的 `major_family_distribution` 距离表示。

| contrast | unit | n | mean major distribution distance | mean family diversity | mean homogeneity |
|---|---|---|---|---|---|
| same_model_same_prompt | within_team | 1200 | 0.4221 | 0.4735 | 0.7023 |
| same_model_different_prompt | between_team_same_question | 1200 | 0.1939 |  |  |
| different_model_same_prompt | between_team_same_question | 1800 | 0.2920 |  |  |
| different_model_different_prompt | between_team_same_question | 3600 | 0.2998 |  |  |

读法：如果 `different_model_same_prompt` 明显高于 `same_model_different_prompt`，说明模型身份/输出风格是更强的策略分布来源；如果反过来，说明策略 prompt 是更强来源。
