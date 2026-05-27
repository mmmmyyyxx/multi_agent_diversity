# P7 GPT-5.5 Blind Validation

- evaluator_model: gpt-5.5
- candidate_groups: 1700
- sampled_groups: 120
- matched_evaluations: 120

## Bucket Counts

- high_text: candidates=511, sampled=20
- low_text: candidates=511, sampled=20
- high_strategy: candidates=511, sampled=20
- low_strategy: candidates=511, sampled=20
- high_text_low_strategy: candidates=131, sampled=20
- low_text_high_strategy: candidates=109, sampled=20

## GPT Evaluation

- mean_gpt_method_diversity_score: 1.4917
- strategy_tree_vs_gpt_spearman: rho=0.0534, n=120
- major_tree_vs_gpt_spearman: rho=-0.1677, n=120
- trace_text_vs_gpt_spearman: rho=0.8015, n=120
- trace_embedding_vs_gpt_spearman: rho=0.7801, n=120
- high_strategy_minus_low_strategy_gpt_score: mean=0.3000, 95% CI=[0.1000, 0.5000]
- high_text_minus_low_text_gpt_score: mean=1.2500, 95% CI=[1.0000, 1.5000]
- high_text_low_strategy_minus_low_text_high_strategy_gpt_score: mean=1.1000, 95% CI=[0.8500, 1.3500]

## Bucket Means

| bucket | n | family_div | major_div | trace_embedding_div | trace_token_div | GPT-5.5 score |
|---|---:|---:|---:|---:|---:|---:|
| high_text | 20 | 0.5062 | 0.3125 | 0.3334 | 0.6279 | 2.2500 |
| low_text | 20 | 0.2646 | 0.1317 | 0.0034 | 0.0198 | 1.0000 |
| high_strategy | 20 | 0.9282 | 0.9590 | 0.0581 | 0.1930 | 1.3000 |
| low_strategy | 20 | 0.0000 | 0.0000 | 0.0090 | 0.0561 | 1.0000 |
| high_text_low_strategy | 20 | 0.2842 | 0.0292 | 0.3019 | 0.5945 | 2.2500 |
| low_text_high_strategy | 20 | 0.7054 | 0.8382 | 0.0134 | 0.0576 | 1.1500 |

## Correlations

| metric | Spearman rho vs GPT-5.5 score | n |
|---|---:|---:|
| family_div | 0.0534 | 120 |
| major_div | -0.1677 | 120 |
| trace_embedding_div | 0.7801 | 120 |
| trace_token_div | 0.8015 | 120 |

判读：本实验把 GPT-5.5 盲评当作人类感知参考时，trace token/embedding 多样性与 GPT-5.5 分数高度同向；family/major 策略树多样性与 GPT-5.5 分数几乎不相关。也就是说，GPT-5.5 更像是在读取完整 trace 的可见展开差异，而不是自动 taxonomy 下的 family 分布。
