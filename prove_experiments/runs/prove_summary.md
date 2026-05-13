# Prove Experiment Summary

| run_name | probe_name | model | eval_size | mean_family_diversity | mean_family_homogeneity_rate | mean_major_family_diversity | low_confidence_share | all_same_pair_rate | target_exact_hit_rate | target_same_major_hit_rate | vote_acc | lambda_diversity | lambda_homogeneity | same_major_family_weight | update_applied_rate | candidate_family_shift_rate | candidate_invalid_delta | optimization_signal_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P4_mixed_strategy_deepseek_chat_seed42 | mixed_strategy_mmlu | deepseek-chat | 100 | 0.5821 | 0.5077 | 0.7040 | 0.0120 | 0.0400 | 0.2700 | 0.5380 | 0.8800 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_mixed_strategy_gemini_flash_lite_seed42 | mixed_strategy_mmlu | gemini-2.5-flash-lite | 100 | 0.4180 | 0.8227 | 0.3194 | 0.0040 | 0.1700 | 0.2560 | 0.6140 | 0.8000 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_mixed_strategy_gpt4omini_seed42 | mixed_strategy_mmlu | gpt-4o-mini | 100 | 0.4805 | 0.6995 | 0.4096 | 0.0020 | 0.0700 | 0.2240 | 0.6380 | 0.8600 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_mixed_strategy_qwen25_7b_seed42 | mixed_strategy_mmlu | qwen2.5-7b-instruct | 100 | 0.5777 | 0.4991 | 0.5562 | 0.0400 | 0.0100 | 0.2620 | 0.6320 | 0.7100 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_deepseek_chat_seed42 | same_elimination_mmlu | deepseek-chat | 100 | 0.4773 | 0.6936 | 0.4822 | 0.0120 | 0.1600 | 0.5660 | 0.5900 | 0.9000 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_gemini_flash_lite_seed42 | same_elimination_mmlu | gemini-2.5-flash-lite | 100 | 0.3873 | 0.8395 | 0.2829 | 0.0060 | 0.2000 | 0.7580 | 0.7660 | 0.7900 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_gpt4omini_seed42 | same_elimination_mmlu | gpt-4o-mini | 100 | 0.4219 | 0.7635 | 0.3486 | 0.0040 | 0.1300 | 0.7760 | 0.8040 | 0.8400 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_qwen25_7b_seed42 | same_elimination_mmlu | qwen2.5-7b-instruct | 100 | 0.5121 | 0.6402 | 0.3454 | 0.0060 | 0.0400 | 0.6460 | 0.7860 | 0.7200 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |

## 自动对照提示

- mixed/same diversity delta: 0.0650；应为正，才支持显式策略干预有效。

## 统计检验

### paired_mixed_minus_same_family_diversity
- paired n=400, mean_delta=0.0650, 95% bootstrap CI=[0.0445, 0.0856], Wilcoxon p~0.0000

### paired_mixed_minus_same_homogeneity
- paired n=400, mean_delta=-0.1019, 95% bootstrap CI=[-0.1282, -0.0759], Wilcoxon p~0.0000

### paired_mixed_minus_same_major_diversity
- paired n=400, mean_delta=0.1325, 95% bootstrap CI=[0.0905, 0.1719], Wilcoxon p~0.0000

### model_identity_check
- strategy_effect_major_disagreement: 0.1030
- model_identity_effect_major_disagreement: 0.2135
- strategy_gt_model_identity: 0

### family_vs_major_disagreement_spearman
- n=8, Spearman rho=0.9048

