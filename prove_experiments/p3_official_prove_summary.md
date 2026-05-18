# P3 证明实验汇总

## 指标中文含义

| 指标 | 中文含义 |
|---|---|
| `mean_family_diversity` | 策略树 leaf 层面的平均团队多样性。越高表示五个 agent 被判到的细策略越分散。 |
| `mean_family_homogeneity_rate` | 平均同质性。越高表示 agent 之间策略越相似。 |
| `mean_major_family_diversity` | 主类层面的平均团队多样性。越高表示五个 agent 更常落到不同主策略类。 |
| `mean_intra_family_diversity` | 同一主类内部的 leaf 多样性。 |
| `low_confidence_share` | judge 对策略标签低置信的比例。越高表示标签更不稳定。 |
| `all_same_pair_rate` | 五个 agent 策略对完全相同的题目比例。越低表示策略差异更明显。 |
| `target_exact_hit_rate` | `primary` 或 `secondary` leaf 精确命中 prompt 目标 leaf 的比例。 |
| `target_same_major_hit_rate` | `primary` 所属主类命中目标主类，或 leaf 精确命中的比例。 |
| `vote_acc` | 五个 agent 多数投票答案准确率。 |

## Run 级汇总

| run_name | probe_name | model | eval_size | mean_family_diversity | mean_family_homogeneity_rate | mean_major_family_diversity | mean_intra_family_diversity | low_confidence_share | all_same_pair_rate | target_exact_hit_rate | target_same_major_hit_rate | vote_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P3_mixed_strategy_deepseek_chat_seed42 | mixed_strategy_mmlu | deepseek-chat | 100 | 0.5564 | 0.4745 | 0.7584 | 0.3545 | 0.0380 | 0.0200 | 0.2740 | 0.3640 | 0.9100 |
| P3_mixed_strategy_gemini_flash_lite_seed42 | mixed_strategy_mmlu | gemini-2.5-flash-lite | 100 | 0.4533 | 0.7588 | 0.3927 | 0.5139 | 0.0020 | 0.1500 | 0.0680 | 0.2300 | 0.8100 |
| P3_mixed_strategy_gpt4omini_seed42 | mixed_strategy_mmlu | gpt-4o-mini | 100 | 0.4807 | 0.7157 | 0.4243 | 0.5371 | 0.0180 | 0.0500 | 0.1360 | 0.2500 | 0.8600 |
| P3_mixed_strategy_qwen25_7b_seed42 | mixed_strategy_mmlu | qwen2.5-7b-instruct | 100 | 0.6281 | 0.4536 | 0.7302 | 0.5259 | 0.0120 | 0.0300 | 0.2720 | 0.3800 | 0.7700 |
| P3_same_elimination_deepseek_chat_seed42 | same_elimination_mmlu | deepseek-chat | 100 | 0.4563 | 0.6625 | 0.5023 | 0.4103 | 0.0240 | 0.1000 | 0.5240 | 0.5400 | 0.9300 |
| P3_same_elimination_gemini_flash_lite_seed42 | same_elimination_mmlu | gemini-2.5-flash-lite | 100 | 0.4124 | 0.8058 | 0.3162 | 0.5085 | 0.0020 | 0.1900 | 0.7340 | 0.7460 | 0.7900 |
| P3_same_elimination_gpt4omini_seed42 | same_elimination_mmlu | gpt-4o-mini | 100 | 0.3395 | 0.8566 | 0.2156 | 0.4634 | 0.0160 | 0.1800 | 0.8080 | 0.8240 | 0.8600 |
| P3_same_elimination_qwen25_7b_seed42 | same_elimination_mmlu | qwen2.5-7b-instruct | 100 | 0.4953 | 0.6913 | 0.2933 | 0.6973 | 0.0160 | 0.0400 | 0.6160 | 0.8140 | 0.7600 |

## Same vs Mixed 总体对照

| 对比项 | same 平均 | mixed 平均 | mixed - same |
|---|---|---|---|
| leaf 策略多样性 | 0.4259 | 0.5296 | 0.1038 |
| 主类策略多样性 | 0.3318 | 0.5764 | 0.2446 |
| 策略同质性 | 0.7540 | 0.6006 | -0.1534 |

## 统计检验

### paired: mixed - same 的 leaf 策略多样性
- 配对样本数=400, mean_delta=0.1038, 95% bootstrap CI=[0.0820, 0.1244], Wilcoxon 近似 p=0.0000

### paired: mixed - same 的策略同质性
- 配对样本数=400, mean_delta=-0.1534, 95% bootstrap CI=[-0.1822, -0.1240], Wilcoxon 近似 p=0.0000

### paired: mixed - same 的主类策略多样性
- 配对样本数=400, mean_delta=0.2446, 95% bootstrap CI=[0.2049, 0.2848], Wilcoxon 近似 p=0.0000

### 模型身份效应检查
- strategy_effect_major_disagreement: 0.1753
- model_identity_effect_major_disagreement: 0.2355
- strategy_gt_model_identity: 0

### leaf 多样性与主类分歧 Spearman 相关
- n=8, Spearman rho=0.8095

