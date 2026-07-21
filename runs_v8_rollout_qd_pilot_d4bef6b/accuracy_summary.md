# Task-Level Accuracy Summary

| task_id | benchmark | setting | n | vote_acc_mean | mean_individual_acc_mean | best_individual_acc_mean | oracle_acc_mean | aggregation_gap_mean | mean_useful_diversity_mean | mean_vote_margin_mean | mean_boundary_useful_diversity_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| disambiguation_qa | BBH | shared_accuracy_only_tcs_vote_first | 1 | 0.504 | 0.464 | 0.496 | 0.616 | 0.112 | 0.0 | -0.0688 | 0.002667 |
| disambiguation_qa | BBH | shared_accuracy_rollout_embedding_tcs | 1 | 0.712 | 0.672 | 0.728 | 0.848 | 0.136 | 0.051145 | 0.3488 | 0.002667 |
| disambiguation_qa | BBH | shared_vote_ready_rollout_diversity_tcs | 1 | 0.624 | 0.6128 | 0.72 | 0.824 | 0.2 | 0.04919 | 0.2256 | 0.0 |
