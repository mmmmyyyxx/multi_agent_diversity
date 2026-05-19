# P7 GPT-5.5 Blind Validation

- evaluator_model: gpt-5.5
- candidate_groups: 1200
- sampled_groups: 80
- matched_evaluations: 80

## Bucket Counts

- high_strategy: candidates=361, sampled=20
- low_strategy: candidates=361, sampled=20
- high_text_low_strategy: candidates=71, sampled=20
- low_text_high_strategy: candidates=78, sampled=20

## GPT Evaluation

- mean_gpt_method_diversity_score: 1.5125
- strategy_tree_vs_gpt_spearman: rho=0.0075, n=80
- major_tree_vs_gpt_spearman: rho=-0.0592, n=80
- trace_text_vs_gpt_spearman: rho=0.6829, n=80
- trace_embedding_vs_gpt_spearman: rho=0.6722, n=80
- high_strategy_minus_low_strategy_gpt_score: mean=-0.0250, 95% CI=[-0.3000, 0.2750]

## Bucket Means

| bucket | n | family_div | major_div | trace_embedding_div | trace_token_div | GPT-5.5 score |
|---|---:|---:|---:|---:|---:|---:|
| high_strategy | 20 | 0.7445 | 0.8697 | 0.1370 | 0.3986 | 1.8000 |
| low_strategy | 20 | 0.2623 | 0.0722 | 0.0870 | 0.2116 | 1.2500 |
| high_text_low_strategy | 20 | 0.2962 | 0.0325 | 0.2058 | 0.5217 | 1.8000 |
| low_text_high_strategy | 20 | 0.6878 | 0.7973 | 0.0278 | 0.0672 | 1.2000 |

## Correlations

| metric | Spearman rho vs GPT-5.5 score | n |
|---|---:|---:|
| family_div | 0.0075 | 80 |
| major_div | -0.0592 | 80 |
| trace_embedding_div | 0.6722 | 80 |
| trace_token_div | 0.6829 | 80 |

判读：本实验把 GPT-5.5 盲评当作人类感知参考时，trace token/embedding 多样性与 GPT-5.5 分数高度同向；family/major 策略树多样性与 GPT-5.5 分数几乎不相关。也就是说，GPT-5.5 更像是在读取完整 trace 的可见展开差异，而不是自动 taxonomy 下的 family 分布。
