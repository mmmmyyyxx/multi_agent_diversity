# Agent behavior clustering

`strict_comparison_status = exploratory_due_to_unmatched_repeated_test_observations`

## 1. Data availability and strictness

All 15 target runs were audited; 15/15 contained reconstructable 75-example train and 125-example test profiles. Profiles were reconstructed from each run's read-only exact-observation cache and checked against its saved behavior matrices. Cross-setting test differences remain exploratory because these historical runs predate the cumulative v2 observation chain.

## 2. Train/test k=2 partitions

| Setting | Seed | Train | Test | Train strength | Test strength | Stability |
|---|---:|---|---|---|---|---|
| shared_independent_accuracy | 44 | `{0,1,2,4}|{3}` | `{0,1,2,4}|{3}` | strong | strong | stable |
| shared_independent_accuracy | 45 | `{0,1,2,4}|{3}` | `{0,1,2,4}|{3}` | strong | strong | stable |
| shared_independent_accuracy | 46 | `{1,2,3,4}|{0}` | `{0,1,2}|{3,4}` | moderate | moderate | unstable |
| shared_peer_state_vote_first | 44 | `{0,2,3}|{1,4}` | `{0,2,3}|{1,4}` | moderate | strong | stable |
| shared_peer_state_vote_first | 45 | `{0,2,4}|{1,3}` | `{0,2,4}|{1,3}` | strong | strong | stable |
| shared_peer_state_vote_first | 46 | `{1,2,4}|{0,3}` | `{0,1,2,4}|{3}` | moderate | moderate | partially_stable |
| shared_peer_state_member_pareto | 44 | `{0,2,3}|{1,4}` | `{0,1,2,3}|{4}` | strong | moderate | partially_stable |
| shared_peer_state_member_pareto | 45 | `{1,2,3,4}|{0}` | `{2,3,4}|{0,1}` | moderate | strong | partially_stable |
| shared_peer_state_member_pareto | 46 | `{0,1,3,4}|{2}` | `{0,1,3,4}|{2}` | strong | moderate | stable |
| shared_member_aware_responsibility | 44 | `{1,2,3,4}|{0}` | `{2,3,4}|{0,1}` | moderate | strong | partially_stable |
| shared_member_aware_responsibility | 45 | `{2,3,4}|{0,1}` | `{0,1,2,4}|{3}` | strong | strong | unstable |
| shared_member_aware_responsibility | 46 | `{0,1,2,3}|{4}` | `{0,1,2,3}|{4}` | moderate | strong | stable |
| shared_member_aware_full | 44 | `{0,1,2,4}|{3}` | `{0,1,3,4}|{2}` | moderate | moderate | unstable |
| shared_member_aware_full | 45 | `{1,2,3,4}|{0}` | `{1,2,3}|{0,4}` | moderate | moderate | partially_stable |
| shared_member_aware_full | 46 | `{0,2,3,4}|{1}` | `{1,3,4}|{0,2}` | moderate | moderate | unstable |

## 3. Structural patterns

- `shared_independent_accuracy`: dominant `4+1`; patterns {'4+1': 5, '3+2': 1}; strengths {'strong': 4, 'moderate': 2}; train/test stable seeds 2/3.
- `shared_peer_state_vote_first`: dominant `3+2`; patterns {'3+2': 5, '4+1': 1}; strengths {'moderate': 3, 'strong': 3}; train/test stable seeds 2/3.
- `shared_peer_state_member_pareto`: dominant `4+1`; patterns {'3+2': 2, '4+1': 4}; strengths {'strong': 3, 'moderate': 3}; train/test stable seeds 1/3.
- `shared_member_aware_responsibility`: dominant `4+1`; patterns {'4+1': 4, '3+2': 2}; strengths {'moderate': 2, 'strong': 4}; train/test stable seeds 1/3.
- `shared_member_aware_full`: dominant `4+1`; patterns {'4+1': 4, '3+2': 2}; strengths {'moderate': 6}; train/test stable seeds 0/3.

## 4. Train/test and cross-seed stability

Train/test stability totals are 6 stable, 5 partially stable, and 4 unstable runs. S1 and S2 each preserve the partition in two seeds; S3 and S4 do so in one; S5 does so in none. Exact Agent-ID partitions are generally less stable than cluster-size structure. Across seeds, S1 keeps a 4+1 train structure in all three seeds and S2 keeps a 3+2 train structure in all three, but S2 has zero exact cross-seed ID matches. S3/S4 mix 4+1 and 3+2, while S5 is consistently 4+1 on train but changes which Agent is isolated.

Bootstrap partition support is below 0.70 for: shared_peer_state_vote_first seed 44 train, shared_peer_state_member_pareto seed 44 test, shared_independent_accuracy seed 46 train, shared_peer_state_vote_first seed 46 train, shared_member_aware_full seed 46 train. These are weak-bootstrap partitions, not evidence of no available data.

## 5. S4/S5 high-frequency targets

- `shared_member_aware_responsibility` seed 44: Agent0 cluster=[0] singleton=True minority=True agent attempt/accepted shares=0.281/0.200 efficiency=0.222; Agent1 cluster=[1, 2, 3, 4] singleton=False minority=False agent attempt/accepted shares=0.281/0.100 efficiency=0.111.
- `shared_member_aware_full` seed 44: Agent4 cluster=[0, 1, 2, 4] singleton=False minority=False agent attempt/accepted shares=0.281/0.312 efficiency=0.556.
- `shared_member_aware_responsibility` seed 45: Agent0 cluster=[0, 1] singleton=False minority=True agent attempt/accepted shares=0.406/0.100 efficiency=0.077.
- `shared_member_aware_full` seed 45: Agent3 cluster=[1, 2, 3, 4] singleton=False minority=False agent attempt/accepted shares=0.500/0.250 efficiency=0.125.
- `shared_member_aware_responsibility` seed 46: Agent2 cluster=[0, 1, 2, 3] singleton=False minority=False agent attempt/accepted shares=0.375/0.182 efficiency=0.167.
- `shared_member_aware_full` seed 46: Agent3 cluster=[0, 2, 3, 4] singleton=False minority=False agent attempt/accepted shares=0.500/0.308 efficiency=0.250.

S4's highest-frequency target is a singleton in one side of the seed-44 tie and a two-member minority in seed 45, but belongs to the majority cluster in seed 46. S5's highest-frequency target belongs to the majority cluster in all three seeds. Therefore repeated selection does not generally create a singleton specialist.

## 6. Responsibility and update efficiency

The diagnostic high-responsibility/low-efficiency rule flagged: shared_member_aware_responsibility seed 44 train: [0, 1, 3]; shared_member_aware_responsibility seed 44 test: [0, 1, 3]; shared_member_aware_full seed 44 train: [1, 2]; shared_member_aware_full seed 44 test: [1, 2]; shared_member_aware_responsibility seed 45 train: [0]; shared_member_aware_responsibility seed 45 test: [0]; shared_member_aware_full seed 45 train: [3]; shared_member_aware_full seed 45 test: [3]; shared_member_aware_responsibility seed 46 train: [0, 2, 3]; shared_member_aware_responsibility seed 46 test: [0, 2, 3]; shared_member_aware_full seed 46 train: [3]; shared_member_aware_full seed 46 test: [3].
The clearest high-attempt/low-acceptance cases are S4 seed 45 Agent0 (13/32 attempts, 1 acceptance), S4 seed 46 Agent2 (12/32, 2), S5 seed 45 Agent3 (16/32, 2), and S5 seed 46 Agent3 (16/32, 4). S5 seed 44 is a counterexample: its most-selected Agent4 has 5 acceptances from 9 attempts. The mismatch diagnosis is therefore recurrent but not universal.

## 7. What cluster strength represents

Partitions are defined only from correctness vectors. Disagreement, double-fault, and same-wrong quantities are auxiliary contrasts. Between-cluster answer disagreement is higher for every setting on average, and within-cluster double fault is higher, as expected for correctness-similarity clusters. Same-wrong contrast is mixed and is negative for S2 and S5, so these are not consistently same-error clusters.

Across all 30 splits, the correctness correlation gap has correlation -0.177 with mean M, 0.003 with mean same-wrong excess, and -0.159 with team vote accuracy. This does not support either 'stronger clusters imply higher M' or 'clusters merely track lower same-wrong'; the observed associations are weak and exploratory.

## 8. Responsibility-space mismatch interpretation

High responsibility combined with many attempts and below-average acceptance efficiency is compatible with a responsibility-versus-prompt-search mismatch, but it is diagnostic rather than causal. Candidate-search history was not added to responsibility or scheduling.

## 9. Conclusions awaiting strict v2 reruns

All cross-setting test comparisons, claims that S4 or S5 causes a particular cluster, and claims linking cluster strength to superior final test performance must await complete v2 matched-observation runs. Within-run partitions and train/test stability describe the observations actually saved by each historical run.

## 10. Dynamic clustering

`dynamic_clustering_status = unavailable`: this report does not infer intermediate profiles from prompt-hash trajectories. Only final active train/test profiles are analyzed.
