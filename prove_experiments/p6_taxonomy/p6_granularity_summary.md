# P6 Taxonomy Granularity Sensitivity

| run_name | question_count | major_only_diversity_mean | weighted_tree_diversity_mean | strict_leaf_diversity_mean | weighted_minus_major_diversity_mean | strict_minus_weighted_diversity_mean | weighted_tree_diversity_human_spearman_rho |
|---|---|---|---|---|---|---|---|
| P4_mixed_strategy_deepseek_chat_seed42 | 100 | 0.7181 | 0.5821 | 0.7845 | -0.1360 | 0.2024 |  |
| P4_mixed_strategy_gemini_flash_lite_seed42 | 100 | 0.2748 | 0.4180 | 0.3973 | 0.1432 | -0.0207 |  |
| P4_mixed_strategy_gpt4omini_seed42 | 100 | 0.3987 | 0.4805 | 0.6156 | 0.0818 | 0.1351 |  |
| P4_mixed_strategy_qwen25_7b_seed42 | 100 | 0.5683 | 0.5777 | 0.8673 | 0.0094 | 0.2895 |  |
| P4_same_elimination_deepseek_chat_seed42 | 100 | 0.4583 | 0.4773 | 0.5219 | 0.0189 | 0.0446 |  |
| P4_same_elimination_gemini_flash_lite_seed42 | 100 | 0.2415 | 0.3873 | 0.3408 | 0.1457 | -0.0464 |  |
| P4_same_elimination_gpt4omini_seed42 | 100 | 0.3510 | 0.4219 | 0.4347 | 0.0709 | 0.0128 |  |
| P4_same_elimination_qwen25_7b_seed42 | 100 | 0.3647 | 0.5121 | 0.7469 | 0.1474 | 0.2348 |  |

判读：weighted_tree 应在 major-only 与 strict leaf-only 之间；如果 human_spearman 可用，优先看 weighted_tree 是否最好或接近最好。
