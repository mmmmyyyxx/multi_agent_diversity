# P7 GPT-5.5 Blind Validation

- evaluator_model: gpt-5.5
- candidate_groups: 800
- sampled_groups: 80
- matched_evaluations: 80

## Bucket Counts

- high_strategy: candidates=241, sampled=20
- low_strategy: candidates=241, sampled=20
- high_text_low_strategy: candidates=44, sampled=20
- low_text_high_strategy: candidates=44, sampled=20

## GPT Evaluation

- mean_gpt_method_diversity_score: 1.5125
- strategy_tree_vs_gpt_spearman: rho=0.0075, n=80
- major_tree_vs_gpt_spearman: rho=-0.0592, n=80
- trace_text_vs_gpt_spearman: rho=0.6829, n=80
- high_strategy_minus_low_strategy_gpt_score: mean=-0.0250, 95% CI=[-0.3000, 0.2750]

判读：如果 strategy_tree_vs_gpt_spearman 为正，且 high_strategy 组 GPT 分数高于 low_strategy 组，说明策略树多样性与独立 GPT-5.5 盲评的构念判断一致。
