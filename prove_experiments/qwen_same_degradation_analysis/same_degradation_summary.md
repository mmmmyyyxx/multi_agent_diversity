# Qwen Same Prompt Degradation Analysis

## Summary by run/group

| run_name | group | n | degraded_rate | bad_agent_mean | family_div | major_div | homogeneity | all_same_rate | vote_acc |
|---|---|---|---|---|---|---|---|---|---|
| P4_same_definition_qwen25_7b_seed42 | normal | 45 | 0.0000 | 0.0000 | 0.5393 | 0.5326 | 0.6788 | 0.0667 | 0.6889 |
| P4_same_definition_qwen25_7b_seed42 | degraded | 55 | 1.0000 | 2.4364 | 0.4558 | 0.2368 | 0.7139 | 0.0182 | 0.7636 |
| P4_same_definition_qwen25_7b_seed42 | all | 100 | 0.5500 | 1.3400 | 0.4934 | 0.3699 | 0.6981 | 0.0400 | 0.7300 |
| P4_same_elimination_qwen25_7b_seed42 | normal | 32 | 0.0000 | 0.0000 | 0.5262 | 0.4618 | 0.6372 | 0.0312 | 0.6875 |
| P4_same_elimination_qwen25_7b_seed42 | degraded | 68 | 1.0000 | 2.1324 | 0.4236 | 0.2019 | 0.7703 | 0.0000 | 0.8088 |
| P4_same_elimination_qwen25_7b_seed42 | all | 100 | 0.6800 | 1.4500 | 0.4565 | 0.2851 | 0.7277 | 0.0100 | 0.7700 |
| P4_same_definition_gpt4omini_seed42 | normal | 100 | 0.0000 | 0.0000 | 0.4722 | 0.3205 | 0.7896 | 0.1200 | 0.8700 |
| P4_same_definition_gpt4omini_seed42 | degraded | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P4_same_definition_gpt4omini_seed42 | all | 100 | 0.0000 | 0.0000 | 0.4722 | 0.3205 | 0.7896 | 0.1200 | 0.8700 |
| P4_same_elimination_gpt4omini_seed42 | normal | 100 | 0.0000 | 0.0000 | 0.4210 | 0.3486 | 0.7618 | 0.1300 | 0.8400 |
| P4_same_elimination_gpt4omini_seed42 | degraded | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P4_same_elimination_gpt4omini_seed42 | all | 100 | 0.0000 | 0.0000 | 0.4210 | 0.3486 | 0.7618 | 0.1300 | 0.8400 |
| P4_same_definition_deepseek_chat_seed42 | normal | 99 | 0.0000 | 0.0000 | 0.4812 | 0.6441 | 0.5340 | 0.1010 | 0.9091 |
| P4_same_definition_deepseek_chat_seed42 | degraded | 1 | 1.0000 | 1.0000 | 0.4277 | 0.0000 | 0.8388 | 0.0000 | 1.0000 |
| P4_same_definition_deepseek_chat_seed42 | all | 100 | 0.0100 | 0.0100 | 0.4807 | 0.6377 | 0.5370 | 0.1000 | 0.9100 |
| P4_same_elimination_deepseek_chat_seed42 | normal | 99 | 0.0000 | 0.0000 | 0.4762 | 0.4866 | 0.6924 | 0.1515 | 0.8990 |
| P4_same_elimination_deepseek_chat_seed42 | degraded | 1 | 1.0000 | 1.0000 | 0.7135 | 0.7219 | 0.6000 | 0.0000 | 1.0000 |
| P4_same_elimination_deepseek_chat_seed42 | all | 100 | 0.0100 | 0.0100 | 0.4786 | 0.4889 | 0.6915 | 0.1500 | 0.9000 |
| P4_same_definition_gemini_flash_lite_seed42 | normal | 75 | 0.0000 | 0.0000 | 0.4327 | 0.3607 | 0.7835 | 0.1600 | 0.8800 |
| P4_same_definition_gemini_flash_lite_seed42 | degraded | 25 | 1.0000 | 3.4000 | 0.5411 | 0.5330 | 0.7108 | 0.1600 | 0.7200 |
| P4_same_definition_gemini_flash_lite_seed42 | all | 100 | 0.2500 | 0.8500 | 0.4598 | 0.4038 | 0.7653 | 0.1600 | 0.8400 |
| P4_same_elimination_gemini_flash_lite_seed42 | normal | 74 | 0.0000 | 0.0000 | 0.3489 | 0.2230 | 0.8725 | 0.1622 | 0.8514 |
| P4_same_elimination_gemini_flash_lite_seed42 | degraded | 26 | 1.0000 | 3.7692 | 0.5587 | 0.5388 | 0.6920 | 0.1154 | 0.6538 |
| P4_same_elimination_gemini_flash_lite_seed42 | all | 100 | 0.2600 | 0.9800 | 0.4034 | 0.3051 | 0.8256 | 0.1500 | 0.8000 |
