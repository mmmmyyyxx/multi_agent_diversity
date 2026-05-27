# P7 Text Metric Alignment

- input_rows: 120
- top_bottom_k: 20
- gpt_positive_threshold: score >= 2

## Metric Summary

| metric | Spearman rho | AUC score>=2 | AUC score>=3 | high mean metric | low mean metric | high GPT | low GPT | delta GPT | 95% CI | high score>=2 | low score>=2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| trace_token_div | 0.8015 | 0.9732 | 0.9012 | 0.6557 | 0.0167 | 2.4000 | 1.0000 | 1.4000 | [1.2000, 1.6000] | 1.0000 | 0.0000 |
| trace_embedding_div | 0.7801 | 0.9601 | 0.8935 | 0.3334 | 0.0034 | 2.2500 | 1.0000 | 1.2500 | [1.0000, 1.5000] | 0.9500 | 0.0000 |
| family_div | 0.0534 | 0.5420 | 0.4618 | 0.9282 | 0.0000 | 1.3000 | 1.0000 | 0.3000 | [0.1000, 0.5000] | 0.3000 | 0.0000 |
| major_div | -0.1677 | 0.4181 | 0.3665 | 0.9806 | 0.0000 | 1.2000 | 1.7500 | -0.5500 | [-0.9000, -0.2000] | 0.2000 | 0.6000 |

## Embedding vs Token Bucket Overlap

| side | overlap | Jaccard |
|---|---:|---:|
| high | 14/20 | 0.5385 |
| low | 14/20 | 0.5385 |

## Reading

On this fixed P7 sample, the strongest single metric by Spearman is `trace_token_div`. Because these 120 rows were originally sampled with embedding-diversity extremes, the result should be read as an in-sample diagnostic, not a fresh sampling experiment.
