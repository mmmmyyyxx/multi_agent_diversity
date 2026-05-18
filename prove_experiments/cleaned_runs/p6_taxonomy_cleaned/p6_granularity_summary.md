# P6 Taxonomy Granularity Sensitivity

| run_name | question_count | major_only_diversity_mean | weighted_tree_diversity_mean | strict_leaf_diversity_mean | weighted_minus_major_diversity_mean | strict_minus_weighted_diversity_mean | weighted_tree_diversity_human_spearman_rho |
|---|---|---|---|---|---|---|---|
| P4_mixed_strategy_deepseek_chat_seed42 | 100 | 0.7181 | 0.5819 | 0.7782 | -0.1362 | 0.1963 |  |
| P4_mixed_strategy_gemini_flash_lite_seed42 | 100 | 0.2767 | 0.4268 | 0.3962 | 0.1501 | -0.0307 |  |
| P4_mixed_strategy_gpt4omini_seed42 | 100 | 0.3987 | 0.4805 | 0.6156 | 0.0818 | 0.1351 |  |
| P4_mixed_strategy_qwen25_7b_seed42 | 100 | 0.4796 | 0.5278 | 0.7815 | 0.0482 | 0.2537 |  |
| P4_same_definition_deepseek_chat_seed42 | 100 | 0.6307 | 0.4807 | 0.7374 | -0.1500 | 0.2567 |  |
| P4_same_definition_gemini_flash_lite_seed42 | 100 | 0.3531 | 0.4598 | 0.4874 | 0.1067 | 0.0276 |  |
| P4_same_definition_gpt4omini_seed42 | 100 | 0.2795 | 0.4722 | 0.5490 | 0.1927 | 0.0768 |  |
| P4_same_definition_qwen25_7b_seed42 | 100 | 0.3606 | 0.4934 | 0.7555 | 0.1328 | 0.2621 |  |
| P4_same_elimination_deepseek_chat_seed42 | 100 | 0.4631 | 0.4786 | 0.5282 | 0.0155 | 0.0496 |  |
| P4_same_elimination_gemini_flash_lite_seed42 | 100 | 0.2476 | 0.4034 | 0.3559 | 0.1558 | -0.0475 |  |
| P4_same_elimination_gpt4omini_seed42 | 100 | 0.3510 | 0.4210 | 0.4347 | 0.0699 | 0.0137 |  |
| P4_same_elimination_qwen25_7b_seed42 | 100 | 0.3131 | 0.4565 | 0.6308 | 0.1434 | 0.1743 |  |

判读：weighted_tree 应在 major-only 与 strict leaf-only 之间；如果 human_spearman 可用，优先看 weighted_tree 是否最好或接近最好。
