# Ablation Summary

| run_dir | setting | baseline_only | init_mode | diversity_reward_enabled | final_prompt_cosine_diversity | final_trace_cosine_diversity | final_reasoning_summary_cosine_diversity | final_test_mean_family_diversity | final_test_mean_family_homogeneity_rate | final_train_mean_llm_direct_diversity_score | final_test_mean_llm_direct_diversity_score | final_train_vote_acc | final_test_vote_acc | disagreement_rate | prompt_drift_cosine_distance | update_applied_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\C_bank_no_div | C_bank_no_div | 0 | bank | 0 | 0.5378 | 0.1796 | 0.3568 | 0.3853 | 0.7350 | 0.0000 | 0.0000 | 0.8650 | 0.8200 | 0.3100 | 0.8721 | 0.1575 |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\D_bank_div | D_bank_div | 0 | bank | 1 | 0.6228 | 0.1227 | 0.2461 | 0.4176 | 0.7243 | 0.0000 | 0.0000 | 0.8350 | 0.7500 | 0.3000 | 0.8947 | 0.1175 |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\F_bank_testonly | F_bank_testonly | 1 | bank | 1 | 0.7330 | 0.0973 | 0.3073 | 0.4421 | 0.6846 | 0.0000 | 0.0000 | 0.0000 | 0.7400 | 0.2800 | 0.0000 | None |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\A_shared_no_div | A_shared_no_div | 0 | shared | 0 | 0.5410 | 0.1529 | 0.3659 | 0.4290 | 0.6877 | 0.0000 | 0.0000 | 0.8350 | 0.8400 | 0.3700 | 0.8296 | 0.1575 |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\B_shared_div | B_shared_div | 0 | shared | 1 | 0.5493 | 0.1545 | 0.4012 | 0.3770 | 0.7284 | 0.0000 | 0.0000 | 0.8550 | 0.8000 | 0.3800 | 0.8071 | 0.1025 |
| D:\games waiting for me\grade_one\experiments\multi_agent_diversity\runs_abcd\E_shared_testonly | E_shared_testonly | 1 | shared | 1 | 0.0000 | 0.0559 | 0.2323 | 0.4102 | 0.7511 | 0.0000 | 0.0000 | 0.0000 | 0.7700 | 0.2300 | 0.0000 | None |
