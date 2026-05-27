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
| P4_mixed_strategy_deepseek_chat_seed42 | mixed_strategy_mmlu | deepseek-chat | 100 | 0.5819 | 0.5092 | 0.7033 | 0.4605 | 0.0120 | 0.0400 | 0.2680 | 0.5360 | 0.8900 |
| P4_mixed_strategy_gemini_flash_lite_seed42 | mixed_strategy_mmlu | gemini-2.5-flash-lite | 100 | 0.4268 | 0.8141 | 0.3276 | 0.5260 | 0.0060 | 0.1600 | 0.2560 | 0.6140 | 0.8000 |
| P4_mixed_strategy_gpt4omini_seed42 | mixed_strategy_mmlu | gpt-4o-mini | 100 | 0.4805 | 0.6995 | 0.4096 | 0.5515 | 0.0020 | 0.0700 | 0.2240 | 0.6380 | 0.8600 |
| P4_mixed_strategy_qwen25_7b_seed42 | mixed_strategy_mmlu | qwen2.5-7b-instruct | 100 | 0.5278 | 0.6076 | 0.4650 | 0.5906 | 0.0100 | 0.0300 | 0.2680 | 0.6680 | 0.7500 |
| P4_mixed_strategy_qwen35_plus_seed42 | mixed_strategy_mmlu | qwen3.5-plus | 100 | 0.5810 | 0.5969 | 0.5892 | 0.5729 | 0.0020 | 0.0400 | 0.2060 | 0.3140 | 0.9500 |
| P4_same_definition_deepseek_chat_seed42 | same_definition_mmlu | deepseek-chat | 100 | 0.4807 | 0.5370 | 0.6377 | 0.3238 | 0.0120 | 0.1000 | 0.2840 | 0.4220 | 0.9100 |
| P4_same_definition_gemini_flash_lite_seed42 | same_definition_mmlu | gemini-2.5-flash-lite | 100 | 0.4598 | 0.7653 | 0.4038 | 0.5159 | 0.0140 | 0.1600 | 0.1200 | 0.6620 | 0.8400 |
| P4_same_definition_gpt4omini_seed42 | same_definition_mmlu | gpt-4o-mini | 100 | 0.4722 | 0.7896 | 0.3205 | 0.6239 | 0.0000 | 0.1200 | 0.2640 | 0.7340 | 0.8700 |
| P4_same_definition_qwen25_7b_seed42 | same_definition_mmlu | qwen2.5-7b-instruct | 100 | 0.4934 | 0.6981 | 0.3699 | 0.6169 | 0.0180 | 0.0400 | 0.3320 | 0.7600 | 0.7300 |
| P4_same_definition_qwen35_plus_seed42 | same_definition_mmlu | qwen3.5-plus | 100 | 0.4716 | 0.7785 | 0.3976 | 0.5456 | 0.0000 | 0.2300 | 0.3360 | 0.6200 | 0.9500 |
| P4_same_elimination_deepseek_chat_seed42 | same_elimination_mmlu | deepseek-chat | 100 | 0.4786 | 0.6915 | 0.4889 | 0.4683 | 0.0140 | 0.1500 | 0.5680 | 0.5920 | 0.9000 |
| P4_same_elimination_gemini_flash_lite_seed42 | same_elimination_mmlu | gemini-2.5-flash-lite | 100 | 0.4034 | 0.8256 | 0.3051 | 0.5018 | 0.0060 | 0.1500 | 0.7660 | 0.7720 | 0.8000 |
| P4_same_elimination_gpt4omini_seed42 | same_elimination_mmlu | gpt-4o-mini | 100 | 0.4210 | 0.7618 | 0.3486 | 0.4934 | 0.0040 | 0.1300 | 0.7760 | 0.8040 | 0.8400 |
| P4_same_elimination_qwen25_7b_seed42 | same_elimination_mmlu | qwen2.5-7b-instruct | 100 | 0.4565 | 0.7277 | 0.2851 | 0.6279 | 0.0080 | 0.0100 | 0.7200 | 0.8180 | 0.7700 |
| P4_same_elimination_qwen35_plus_seed42 | same_elimination_mmlu | qwen3.5-plus | 100 | 0.5017 | 0.7490 | 0.3709 | 0.6324 | 0.0020 | 0.0600 | 0.6660 | 0.7180 | 0.9600 |
| P4_same_prompt_gpt4omini_seed42 | same_prompt_mmlu | gpt-4o-mini | 100 | 0.3181 | 0.8820 | 0.1905 | 0.4456 | 0.0000 | 0.3000 | 0.8220 | 0.8280 | 0.8500 |
| P4_same_prompt_qwen35_plus_seed42 | same_prompt_mmlu | qwen3.5-plus | 100 | 0.4362 | 0.8025 | 0.3258 | 0.5467 | 0.0000 | 0.2800 | 0.6300 | 0.6840 | 0.9600 |

## Same vs Mixed 总体对照

| 对比项 | same 平均 | mixed 平均 | mixed - same |
|---|---|---|---|
| leaf 策略多样性 | 0.4494 | 0.5196 | 0.0702 |
| 主类策略多样性 | 0.3704 | 0.4989 | 0.1286 |
| 策略同质性 | 0.7507 | 0.6455 | -0.1053 |

## 统计检验

### paired: mixed - same 的 leaf 策略多样性
- 配对样本数=500, mean_delta=0.0441, 95% bootstrap CI=[0.0241, 0.0632], Wilcoxon 近似 p=0.0002

### paired: mixed - same 的策略同质性
- 配对样本数=500, mean_delta=-0.0683, 95% bootstrap CI=[-0.0929, -0.0460], Wilcoxon 近似 p=0.0000

### paired: mixed - same 的主类策略多样性
- 配对样本数=500, mean_delta=0.0731, 95% bootstrap CI=[0.0385, 0.1083], Wilcoxon 近似 p=0.0000

### 模型身份效应检查
- strategy_effect_major_disagreement: 0.0794
- model_identity_effect_major_disagreement: 0.2995
- strategy_gt_model_identity: 0

### leaf 多样性与主类分歧 Spearman 相关
- n=17, Spearman rho=0.8064

