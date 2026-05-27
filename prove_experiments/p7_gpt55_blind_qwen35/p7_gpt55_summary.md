# P7 GPT-5.5 Blind Validation

- evaluator_model: gpt-5.5
- candidate_groups: 1200
- sampled_groups: 80
- matched_evaluations: 80

## Bucket Counts

- high_strategy: candidates=363, sampled=20
- low_strategy: candidates=361, sampled=20
- high_text_low_strategy: candidates=96, sampled=20
- low_text_high_strategy: candidates=81, sampled=20

## GPT Evaluation

- mean_gpt_method_diversity_score: 1.1500
- strategy_tree_vs_gpt_spearman: rho=0.0875, n=80
- major_tree_vs_gpt_spearman: rho=0.0000, n=80
- trace_text_vs_gpt_spearman: rho=0.1609, n=80
- trace_embedding_vs_gpt_spearman: rho=0.2091, n=80
- high_strategy_minus_low_strategy_gpt_score: mean=0.0000, 95% CI=[-0.1750, 0.1500]

## Bucket Means

| bucket | n | family_div | major_div | trace_embedding_div | trace_token_div | GPT-5.5 score |
|---|---:|---:|---:|---:|---:|---:|
| high_strategy | 20 | 0.7242 | 0.8567 | 0.0476 | 0.1320 | 1.2000 |
| low_strategy | 20 | 0.2564 | 0.1665 | 0.0360 | 0.1376 | 1.0000 |
| high_text_low_strategy | 20 | 0.3233 | 0.1530 | 0.0505 | 0.1998 | 1.3000 |
| low_text_high_strategy | 20 | 0.6885 | 0.8316 | 0.0319 | 0.0605 | 1.1000 |

## Correlations

| metric | Spearman rho vs GPT-5.5 score | n |
|---|---:|---:|
| family_div | 0.0875 | 80 |
| major_div | 0.0000 | 80 |
| trace_embedding_div | 0.2091 | 80 |
| trace_token_div | 0.1609 | 80 |

判读：本实验把 GPT-5.5 盲评当作人类感知参考时，trace token/embedding 多样性与 GPT-5.5 分数高度同向；family/major 策略树多样性与 GPT-5.5 分数几乎不相关。也就是说，GPT-5.5 更像是在读取完整 trace 的可见展开差异，而不是自动 taxonomy 下的 family 分布。
