# Method: Accuracy-First Sequential State Optimization

## 1. What The Project Optimizes

This repository evolves prompts for a fixed team of five solver agents. It does not update model weights and is not policy-gradient reinforcement learning. Candidate rewards rank prompts during search; validation selects the best epoch; final test runs once after restoring `best_prompts.json`.

The current V9 identity is:

```text
method_version = v9_state_conditioned_error
state_update_mode = sequential_single_agent
reward_mode = state_distribution_vote_reward
candidate_selection_mode = sequential_accuracy_first_state_reward
active_team_selector_version = sequential_accuracy_first_v1
best_state_selection_mode = state_conditioned_vote_first
```

V8 settings and their completed artifacts retain their historical behavior. In particular, V8 may still use rollout archives and joint team selection. V9 does not.

## 2. Fixed Voting Rule

All five agents have equal weight. The normalized answer with the largest vote count wins; top ties use the configured deterministic tie-break. V9 does not add reliability weights, confidence weights, a judge, a router, learned aggregation, or test-time best-agent selection.

For every question:

```text
G = number of correct agents
C0 = G=0, C1 = G=1, ..., C5 = G=5
```

New runs report `c0` through `c5`. `c3plus = c3 + c4 + c5` remains a compatibility summary for old analysis scripts.

## 3. End-To-End Search

```text
training rollout window
  -> choose one target agent by deterministic rotating order
  -> choose parents from that agent's prompt memory
  -> Teacher-Critic-Student candidate generation
  -> Stage A candidate-batch prefilter
  -> Stage B full fixed-probe evaluation
  -> accuracy, invalid, and non-collapse constraints
  -> accuracy-first lexicographic selection against incumbent
  -> immediately activate an accepted prompt
  -> immediately refresh the fixed-probe team snapshot
  -> rebuild that agent's five-prompt memory
```

Only optimization-train data supplies search evidence. Validation selects epochs. Test never generates, ranks, retains, or activates prompts.

## 4. Sequential Agent Updates

One update attempts to change at most one agent. The order rotates by epoch:

```text
epoch 0: 0,1,2,3,4
epoch 1: 1,2,3,4,0
epoch 2: 2,3,4,0,1
```

The runner uses one-based epoch labels, but the first epoch still uses the epoch-0 order. `epoch_agent_order` and `current_agent_order_index` are checkpointed, so interruption and resume preserve the next target.

An accepted prompt becomes active immediately. Every later update is evaluated against the true current four peer prompts. V9 never combines historical prompts across agents and never enumerates a Cartesian product of prompt teams.

## 5. Two-Stage Candidate Evaluation

### Stage A: cheap prefilter

Stage A evaluates generated prompts on a smaller batch with three disjoint pools:

```text
representative: natural optimization samples
coverage: current C0/C1 cases
conversion: current C2/C3 cases
```

It estimates target accuracy, invalid output, ordinary repairs, and likely state transitions. Stage A only shortlists candidates; it cannot activate a prompt.

### Stage B: full fixed acceptance probe

The top `state_full_probe_acceptance_candidates` Stage-A prompts, the incumbent, and rollback/memory prompts are evaluated on the complete fixed optimization probe. Per-agent prompt/question results are cached.

Only Stage B decides:

```text
final target accuracy and correct count
final invalid constraint
correct-set and safe-trace constraints
state-vote reward
candidate ordering and acceptance
prompt-memory rebuilding
```

`stage_b_full_probe_solver_calls` is reported separately. Removing joint enumeration is a semantic correction, not a solver-cost claim.

## 6. Accuracy First

Accuracy is both a hard constraint and the first candidate key. The target prompt must satisfy:

```text
candidate_correct_count >= active_correct_count - local allowance
candidate_correct_count >= initial_correct_count - global allowance
```

Defaults are strict: both loss epsilons and the question-count accuracy band are zero. Invalid output must also stay within `invalid_guard_epsilon`.

Feasible candidates are compared by:

```text
1. target correct count
2. target accuracy
3. state-vote reward
4. lower invalid count
5. diversity constraint slack
6. earlier generation
7. stable prompt hash
```

A lower-accuracy candidate cannot win through Vote, state reward, trace distance, prompt distance, or diversity. A higher-accuracy feasible candidate may win even when its secondary reward is lower. At equal accuracy, reward must improve by at least `state_min_secondary_reward_gain`.

## 7. State And Vote Reward

The configurable state potentials are:

```text
Phi(C0)=0.00  Phi(C1)=1.00  Phi(C2)=1.75
Phi(C3)=3.25  Phi(C4)=3.60  Phi(C5)=3.75
```

For a candidate replacing one target agent:

```text
distribution = mean(Phi(candidate_G) - Phi(active_G))
vote         = state_reward_vote_weight * (candidate_vote_acc - active_vote_acc)
balance      = state_reward_bottom2_weight * (candidate_bottom2_acc - active_bottom2_acc)
state_vote_reward = distribution + vote + balance
```

This reward is secondary to target accuracy. Diversity is absent from it.

Wrong-answer changes have zero optimization value. Changing one wrong answer label to another does not earn state, Vote, diversity, memory, or selection credit when `G` is unchanged. Largest-wrong cluster, same-wrong rate, option count, and raw vote changes may remain diagnostics only.

## 8. Diversity As Non-Collapse Constraints

V9 does not maximize generic rollout diversity. It applies two independent feasibility constraints after accuracy.

### Correct-set complementarity

For each agent, define the set of fixed-probe questions it answers correctly. V9 records mean and minimum pairwise Jaccard distance across the five agents. Candidate diversity must stay above both an active-team local floor and an initial-team global floor.

### Safe C4/C5 trace diversity

Trace distance is computed only when:

```text
candidate team state is C4 or C5
target and peer are both correct
both traces are valid
both embeddings are available
```

C4 pairs have weight 1.0 and C5 pairs 1.5. Wrong traces and C0-C3 are excluded. A C5-to-C4 regression receives no diversity benefit. If no comparable pairs exist, the safe-trace constraint is unavailable and is skipped rather than treated as zero.

## 9. Per-Agent Prompt Memory

Each agent keeps at most five prompts:

```text
active
accuracy_best
state_vote_best
safe_diversity_parent
rollback_or_recent_success
```

Memory supplies deterministic generation parents and rollback candidates. It never activates a prompt automatically and never participates in cross-agent combinations. Entries deduplicate by prompt hash and fixed-probe outcome signature; the safe-diversity slot may retain a distinct safe-trace signature. Reaccept cooldown and a bounded recent-accept cycle window reduce prompt cycling.

## 10. Validation And Final Test

Validation ordering is:

```text
1. mean individual accuracy
2. plurality vote accuracy
3. lower C0
4. higher C3+C4+C5
5. higher C4+C5
6. bottom-2 accuracy
7. lower invalid rate
8. earlier epoch
```

This keeps team competence ahead of aggregation gains. Final test restores the validation-selected prompts from `best_prompts.json`.

## 11. Settings

The unchanged B baseline is:

```text
shared_accuracy_rollout_embedding_tcs
```

V9 ablations are:

```text
shared_v9_sequential_accuracy
shared_v9_sequential_accuracy_state
shared_v9_sequential_accuracy_state_vote
shared_v9_sequential_accuracy_state_vote_diversity
```

The last is the complete method. Historical V9 archive, C2-split, and hybrid aliases were removed rather than silently mapped to new behavior.

## 12. Checkpoint And Outputs

V9 requires `state_conditioned_checkpoint_version = 3`. A V9 v2 checkpoint is incompatible and resume fails explicitly. Checkpoints persist prompt memory, cached probe profiles, initial baselines, the current snapshot, rotating-order cursor, accepted history, cycle state, random state, and cost state.

Important outputs:

```text
run_meta.json
history.json
best_prompts.json
prompt_history.json
update_logs.jsonl
sequential_update_history.jsonl
solver_rollout_records.jsonl
training_checkpoint.json while incomplete
cost_summary.json
final_summary.json
```

V9 does not write `joint_team_selection_history.jsonl`. Run metadata states:

```text
v9_update_mode = sequential_single_agent
joint_team_enumeration_enabled = false
joint_team_combination_count = 0
equal_vote_weighting = true
```

The historical default fingerprint remains:

```text
48c2f27cdcda64d2f7b32d008957b4903c683f49012988c4e5cab301ed29d5fa
```

## 13. Code Map

```text
multi_dataset_diverse_rl/sequential_state.py             V9 reward, constraints, keys, memory
multi_dataset_diverse_rl/state_conditioned.py            state snapshots and legacy readers
multi_dataset_diverse_rl/optimization/training_controller.py rotating target order
multi_dataset_diverse_rl/optimization/prompt_update_controller.py Stage A/B and activation
multi_dataset_diverse_rl/qd/joint_controller.py           fixed-probe snapshot; V8 joint path
multi_dataset_diverse_rl/persistence/checkpoint.py        resume state and compatibility
scripts/experiment_config.py                              named settings
scripts/run_task_level_accuracy.py                        task-level runner
tests/test_v9_sequential_accuracy_first.py                current V9 invariants
```

## 14. Verification

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
& $PY -m compileall multi_dataset_diverse_rl scripts tests
git diff --check
```

After static and unit verification, run one deterministic single-task smoke. It verifies mechanism integrity only; it is not evidence of accuracy improvement.
