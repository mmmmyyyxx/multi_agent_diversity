# Prove Experiment Summary

| run_name | probe_name | model | eval_size | mean_family_diversity | mean_family_homogeneity_rate | mean_major_family_diversity | low_confidence_share | all_same_pair_rate | target_exact_hit_rate | target_same_major_hit_rate | vote_acc | lambda_diversity | lambda_homogeneity | same_major_family_weight | update_applied_rate | candidate_family_shift_rate | candidate_invalid_delta | optimization_signal_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P4_mixed_strategy_deepseek_chat_seed42 | mixed_strategy_mmlu | deepseek-chat | 100 | 0.5819 | 0.5092 | 0.7033 | 0.0120 | 0.0400 | 0.2680 | 0.5360 | 0.8900 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_mixed_strategy_gemini_flash_lite_seed42 | mixed_strategy_mmlu | gemini-2.5-flash-lite | 100 | 0.4268 | 0.8141 | 0.3276 | 0.0060 | 0.1600 | 0.2560 | 0.6140 | 0.8000 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_mixed_strategy_gpt4omini_seed42 | mixed_strategy_mmlu | gpt-4o-mini | 100 | 0.4805 | 0.6995 | 0.4096 | 0.0020 | 0.0700 | 0.2240 | 0.6380 | 0.8600 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_mixed_strategy_qwen25_7b_seed42 | mixed_strategy_mmlu | qwen2.5-7b-instruct | 100 | 0.5278 | 0.6076 | 0.4650 | 0.0100 | 0.0300 | 0.2680 | 0.6680 | 0.7500 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_definition_deepseek_chat_seed42 | same_definition_mmlu | deepseek-chat | 100 | 0.4807 | 0.5370 | 0.6377 | 0.0120 | 0.1000 | 0.2840 | 0.4220 | 0.9100 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_definition_gemini_flash_lite_seed42 | same_definition_mmlu | gemini-2.5-flash-lite | 100 | 0.4598 | 0.7653 | 0.4038 | 0.0140 | 0.1600 | 0.1200 | 0.6620 | 0.8400 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_definition_gpt4omini_seed42 | same_definition_mmlu | gpt-4o-mini | 100 | 0.4722 | 0.7896 | 0.3205 | 0.0000 | 0.1200 | 0.2640 | 0.7340 | 0.8700 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_definition_qwen25_7b_seed42 | same_definition_mmlu | qwen2.5-7b-instruct | 100 | 0.4934 | 0.6981 | 0.3699 | 0.0180 | 0.0400 | 0.3320 | 0.7600 | 0.7300 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_deepseek_chat_seed42 | same_elimination_mmlu | deepseek-chat | 100 | 0.4786 | 0.6915 | 0.4889 | 0.0140 | 0.1500 | 0.5680 | 0.5920 | 0.9000 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_gemini_flash_lite_seed42 | same_elimination_mmlu | gemini-2.5-flash-lite | 100 | 0.4034 | 0.8256 | 0.3051 | 0.0060 | 0.1500 | 0.7660 | 0.7720 | 0.8000 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_gpt4omini_seed42 | same_elimination_mmlu | gpt-4o-mini | 100 | 0.4210 | 0.7618 | 0.3486 | 0.0040 | 0.1300 | 0.7760 | 0.8040 | 0.8400 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |
| P4_same_elimination_qwen25_7b_seed42 | same_elimination_mmlu | qwen2.5-7b-instruct | 100 | 0.4565 | 0.7277 | 0.2851 | 0.0080 | 0.0100 | 0.7200 | 0.8180 | 0.7700 | 0.0000 | 0.0000 | 0.5000 |  |  |  |  |

## 自动对照提示

- mixed/same diversity delta: 0.0461；应为正，才支持显式策略干预有效。

## 统计检验

### paired_mixed_minus_same_family_diversity
- paired n=400, mean_delta=0.0277, 95% bootstrap CI=[0.0044, 0.0507], Wilcoxon p~0.1087

### paired_mixed_minus_same_homogeneity
- paired n=400, mean_delta=-0.0399, 95% bootstrap CI=[-0.0648, -0.0142], Wilcoxon p~0.0090

### paired_mixed_minus_same_major_diversity
- paired n=400, mean_delta=0.0434, 95% bootstrap CI=[0.0047, 0.0817], Wilcoxon p~0.0273

### model_identity_check
- strategy_effect_major_disagreement: 0.0530
- model_identity_effect_major_disagreement: 0.2680
- strategy_gt_model_identity: 0

### family_vs_major_disagreement_spearman
- n=12, Spearman rho=0.8252

