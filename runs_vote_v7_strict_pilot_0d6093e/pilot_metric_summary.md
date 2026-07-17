# Pilot Metric Summary

Mean +/- sample standard deviation across seeds 42 and 43.

| Task | Setting | Vote Acc | Mean Agent Acc | Oracle Acc | Gap | Vote Margin | Triple Error | Shared Rescue | Shared Creation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| disambiguation_qa | shared_vote_pareto_tcs_static | 0.4750 +/- 0.0825 | 0.4850 +/- 0.0684 | 0.6833 +/- 0.0943 | 0.2083 | -0.0283 | 0.5250 | 0.2085 | 0.7915 |
| disambiguation_qa | shared_vote_pareto_tcs_boundary_selector | 0.5583 +/- 0.1061 | 0.5383 +/- 0.0825 | 0.7583 +/- 0.1061 | 0.2000 | 0.0767 | 0.4417 | 0.3641 | 0.6359 |
| disambiguation_qa | shared_vote_error_pareto_tcs | 0.5000 +/- 0.0471 | 0.4900 +/- 0.0047 | 0.6583 +/- 0.0118 | 0.1583 | -0.0200 | 0.5000 | 0.2472 | 0.7528 |
| disambiguation_qa | shared_vote_error_pareto_tcs_residual_specialization | 0.5167 +/- 0.0000 | 0.5083 +/- 0.0071 | 0.6667 +/- 0.0236 | 0.1500 | 0.0167 | 0.4833 | 0.1742 | 0.8258 |
| disambiguation_qa | shared_vote_error_pareto_tcs_residual_cycle_guard | 0.5500 +/- 0.0236 | 0.4933 +/- 0.0424 | 0.6917 +/- 0.1296 | 0.1417 | -0.0100 | 0.4583 | 0.3197 | 0.6803 |
| sports_understanding | shared_vote_pareto_tcs_static | 0.8583 +/- 0.0118 | 0.8517 +/- 0.0024 | 0.9583 +/- 0.0118 | 0.1000 | 0.7033 | 0.1417 | 0.3472 | 0.6528 |
| sports_understanding | shared_vote_pareto_tcs_boundary_selector | 0.8417 +/- 0.0118 | 0.8533 +/- 0.0094 | 0.9500 +/- 0.0000 | 0.1083 | 0.7067 | 0.1583 | 0.4141 | 0.5859 |
| sports_understanding | shared_vote_error_pareto_tcs | 0.8667 +/- 0.0471 | 0.8583 +/- 0.0306 | 0.9250 +/- 0.0118 | 0.0583 | 0.7167 | 0.1333 | 0.3333 | 0.6667 |
| sports_understanding | shared_vote_error_pareto_tcs_residual_specialization | 0.8250 +/- 0.0118 | 0.8400 +/- 0.0047 | 0.9500 +/- 0.0000 | 0.1250 | 0.6800 | 0.1750 | 0.3000 | 0.7000 |
| sports_understanding | shared_vote_error_pareto_tcs_residual_cycle_guard | 0.8583 +/- 0.0118 | 0.8617 +/- 0.0165 | 0.9583 +/- 0.0354 | 0.1000 | 0.7233 | 0.1417 | 0.3655 | 0.6345 |
