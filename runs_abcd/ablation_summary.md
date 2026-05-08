# Ablation Summary

| run_dir | setting | baseline_only | init_mode | diversity_reward_enabled | final_prompt_cosine_diversity | final_trace_cosine_diversity | final_reasoning_summary_cosine_diversity | final_test_mean_family_diversity | final_test_mean_family_homogeneity_rate | final_train_mean_llm_direct_diversity_score | final_test_mean_llm_direct_diversity_score | final_train_vote_acc | final_test_vote_acc | disagreement_rate | prompt_drift_cosine_distance | update_applied_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\C_bank_no_div | C_bank_no_div | 0 | bank | 0 | 0.5378 | 0.1796 | 0.3568 | 0.3853 | 0.7350 | 0.0000 | 0.0000 | 0.8650 | 0.8200 | 0.3100 | 0.8721 | 0.1575 |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\D_bank_div | D_bank_div | 0 | bank | 1 | 0.6228 | 0.1227 | 0.2461 | 0.4176 | 0.7243 | 0.0000 | 0.0000 | 0.8350 | 0.7500 | 0.3000 | 0.8947 | 0.1175 |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\B_shared_div | B_shared_div | 0 | shared | 1 | 0.5493 | 0.1545 | 0.4012 | 0.3770 | 0.7284 | 0.0000 | 0.0000 | 0.8550 | 0.8000 | 0.3800 | 0.8071 | 0.1025 |
