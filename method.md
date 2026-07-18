# Method: Vote-Oriented Multi-Agent Prompt Search

The opt-in v8 competence-first extension is documented in [V8_COMPETENCE_DEPTH_METHOD.md](V8_COMPETENCE_DEPTH_METHOD.md). Existing v7 settings below remain unchanged.

## Overview

This project evolves prompts for a fixed team of solver agents. It does not update model weights and it is not policy-gradient RL. Reward only ranks candidate prompts inside a per-agent beam search.

The primary metric is `vote_acc`: agents answer the same question and the plurality-vote answer is compared with gold. `aggregation_mode=plurality` is the canonical name. The historical `majority` value remains an alias and produces the same plurality result.

The method prioritizes:

1. Preserving the updated target agent's accuracy.
2. Improving final team vote accuracy directly.
3. Improving the gold-vs-wrong vote margin before a vote flip is possible.
4. Dispersing dominant wrong answers only near a vote boundary.

## Training Loop

For each training item:

```text
question + prompt_i -> trace_i + answer_i
```

Every `update_every` steps, the system:

1. Builds a recent-window vote diagnosis.
2. Selects one or two agents with reward-relevant repair pressure.
3. Generates candidate prompts.
4. Evaluates each candidate against the same baseline team on the same batch.
5. Retains the best per-agent prompt beam.
6. Uses beam top-1 as the active prompt.

Validation chooses the best state; final test restores the prompts recorded in `best_prompts.json`.

## Roles

| Field | Current role |
| --- | --- |
| `agent_model` | Solver rollout model. |
| `optimizer_model` | Prompt-generation model for one-shot and Teacher, rewrite, Student, retry, and repair calls. |
| `evaluator_model` | TCS Critic and optional trace-diversity evaluator. |
| `embedding_model` | Local trace-diversity diagnostic model. |

## Vote Diagnostics

The existing task matcher is used to compute:

```text
G = gold_vote_count
H = largest_wrong_vote_count
normalized_vote_margin = (G - H) / num_agents
```

`normalized_vote_margin` is in `[-1, 1]`; it is `-1.0` when every answer is empty.

`boundary_useful_diversity` is nonzero only when at least one agent is correct, there are multiple valid wrong answers, and the team is close to the vote boundary:

```python
if G > 0 and W > 1 and abs(G - H) <= 1:
    boundary_useful_diversity = 1.0 - H / W
else:
    boundary_useful_diversity = 0.0
```

Trace embedding diversity and `oracle_acc` remain diagnostics. They do not control reward, Pareto objectives, agent selection, or best-state selection.

## Candidate Evaluation

Candidate evaluation is baseline-relative:

```text
baseline_prompts  = current active prompts
candidate_prompts = baseline prompts with one target prompt replaced
```

Its data source is always `optimization_train`. `fixed_pool`, `stratified`,
and `random` candidate evaluation never read `val_data`; validation is reserved
for epoch and `vote_first` state selection, and test is reserved for the final
restored-prompt evaluation. Repeats consume a deterministic permutation before
reusing examples. Each candidate log records its total evaluated examples,
unique question count, repeat count, and `candidate_eval_data_source`.

The same batch yields:

```text
baseline_mean_vote_margin
candidate_mean_vote_margin
vote_margin_delta

baseline_boundary_useful_diversity
candidate_boundary_useful_diversity
boundary_useful_diversity_delta
boundary_diversity_gain = max(0.0, boundary_useful_diversity_delta)

vote_gain_count / vote_gain_rate
vote_loss_count / vote_loss_rate
net_vote_count / net_vote_delta
```

A gain changes a wrong baseline vote into a correct candidate vote. A loss does the reverse. The implementation asserts:

```text
net_vote_delta = vote_gain_rate - vote_loss_rate
               = candidate_team_accuracy - baseline_team_accuracy
```

`coverage_delta`, oracle metrics, trace diversity, useful diversity, invalid rate, and rescue metrics remain logged for analysis.

## Reward Modes

| Mode | Purpose |
| --- | --- |
| `accuracy_only` | Ablation: target-agent accuracy only. |
| `guarded_diversity` | Ablation: target guard plus trace-embedding diversity delta. |
| `vote_useful_diversity` | Recommended: target quality plus vote, margin, and boundary-diversity signals. |

`vote_useful_diversity` first requires:

```text
candidate_target_accuracy >= baseline_target_accuracy - accuracy_guard_epsilon
candidate_invalid_rate <= baseline_invalid_rate + invalid_guard_epsilon
```

An infeasible candidate receives `-1.0`. A feasible candidate uses:

```python
reward = (
    effective_weight_target_accuracy * candidate_target_accuracy
    + effective_weight_vote_delta * vote_delta
    + effective_weight_vote_margin * vote_margin_delta
    + effective_weight_boundary_diversity * boundary_diversity_gain
    - reward_weight_invalid_delta * max(0.0, invalid_delta)
)
```

The raw boundary-diversity delta remains logged. Reward only consumes its
positive part, so a candidate that improves the correct vote enough to leave a
fragile boundary is not penalized for its boundary score becoming zero.

`update_logs.jsonl` records both the raw delta and clipped gain, plus the
weighted target-accuracy, vote-delta, vote-margin, boundary-diversity,
invalid-penalty, and guard-penalty reward components.

Phase-adaptive scheduling adjusts vote-delta, vote-margin, boundary-diversity, target-accuracy, and guard weights. Effective values are written to `update_logs.jsonl`.

## Agent Selection

The update score is:

```text
3.0 * per_agent_error_count
+ 2.0 * per_agent_team_wrong_error_count
+ 2.0 * per_agent_invalid_rate
+ 2.0 * per_agent_pivotal_fix_count
+ 1.0 * per_agent_dominant_wrong_redundancy_count
```

A v8 pivotal fix asks whether changing an incorrect target agent to gold makes the actual plurality aggregator correct under the configured tie-break and the same question hash. It is not inferred from `K>=3` or a vote-margin threshold. Dominant-wrong redundancy remains descriptive. Only positive-score agents are selected.

## Boundary-Aware Residual Specialization (v7)

The current method keeps vote contribution context separate from capability
evidence:

```text
vote_context_profile: where an accepted change contributed to the vote
capability_profile: which task-independent residual error family the agent has reliably repaired
```

`BehaviorContext` names vote contribution contexts. It is not a capability or
agent role.
Capability families include entity binding, relation tracking, qualifier or
negation handling, temporal order, option comparison, contradiction checking,
numeric or symbolic reasoning, commonsense consistency, final verification,
output validity, and unknown. Classification is deterministic from existing
trace and validity diagnostics and makes no extra LLM call.

Candidate rows explicitly measure individual fixes and regressions, pivotal
rescues and losses, shared-error rescue and creation, and dominant wrong-cluster
breaks and creations. The aggregate boundary signal is:

```text
4 * pivotal_rescue_rate - 4 * pivotal_loss_rate
+ shared_error_rescue_score - 1.5 * shared_error_creation_score
+ 0.5 * same_wrong_cluster_break_rate - 0.5 * same_wrong_cluster_create_rate
```

Agent selection uses rates rather than raw counts. Pivotal errors receive the
largest weight, followed by near-boundary errors, dominant wrong clusters,
shared errors, general errors, and invalid outputs. Capability affinity and
team coverage gaps are small bonuses; neither assigns a permanent specialist.

Accepted active prompts accumulate capability evidence. Support is shrunk by
`support / (support + specialization_support_shrinkage)`, so one rare pivotal
sample remains nonzero instead of being discarded. Weighted gains and losses
are cumulative, losses use `capability_loss_weight=1.5`, and only positive
posterior evidence enters the EMA profile. Prompt edits are fast state;
capability profiles update after two accepted edits by default or at epoch end.
Rejected candidates and evaluations that never become active do not update the
profile.

The full v7 guard compares candidates with both accepted states and rejected
failure states on shared question hashes. Similar behavior is rejected only
when paired behavior utility does not improve. Student candidates must also
declare preserved mechanisms, exactly one modified mechanism, a change
summary, target residual family, expected shared-error effect, and risk control.

Each accepted state stores a bounded behavior fingerprint keyed by stable
question hash: target correctness, canonical answer SHA256, team-vote
correctness, margin bucket, and behavior context. It stores no question text or
reasoning trace. The residual cycle guard rejects repeated accepted or rejected
behavior only when paired utility has not improved. The mechanism trust region
requires a declared local edit and stronger evidence for large rewrites after
warmup.

## Teacher-Critic-Student

The default prompt-evolution architecture is:

```bash
--optimizer_architecture teacher_critic_student
```

Teacher turns abstract vote diagnostics into a guiding question. Critic audits it; rejection feeds back to Teacher for rewrite and re-review. After `teacher_critic_max_rounds`, the highest Critic-scored question can proceed with `teacher_question_forced_best_score=true`. Student returns strict JSON candidates.

Teacher sees aggregate target errors, vote failures, pivotal-fix counts, dominant wrong-answer redundancy, and invalid-output signals. In v7, voting failures are used only as abstract evidence of harmful shared-error mechanisms; Teacher proposes one local residual repair while preserving pivotal-correct behavior. It does not receive raw gold answers or sample-specific task content.

New TCS candidates must include Teacher/Critic/Student provenance. Existing beam candidates are exempt. Audit a run with:

```powershell
python scripts/audit_tcs_run.py <run_dir_or_root>
```

Candidate provenance uses two separate fields. `candidate_pool_source` says how
the item entered evaluation (`optimizer`, `existing_beam`, or
`current_active_fallback`), while `candidate_source` identifies the generation
mechanism (`teacher_critic_student`, `optimizer` for one-shot generation, or a
fallback mechanism). Trajectory guards use the pool source. The older `source`
field is retained only as a checkpoint and log compatibility alias.

## Candidate And State Selection

`scalar_reward` keeps candidates by scalar reward.

`vote_pareto` first applies the target-accuracy and invalid-rate guards, then uses exactly three Pareto objectives:

```text
maximize vote_gain_rate
minimize vote_loss_rate
maximize candidate_target_accuracy
```

Within a Pareto rank, the deterministic order is vote delta, lower vote loss,
vote gain, vote-margin delta, target accuracy, boundary-diversity delta, and
lower invalid rate.

`vote_error_pareto` is the separate v7 selector. It preserves the old three
objectives and adds:

```text
maximize boundary_shared_error_net_gain
```

Its optional dependence guard requires zero pivotal loss by default and does
not permit shared-error creation to exceed rescue by more than `0.02`.
The old `vote_pareto` dominance and crowding semantics are unchanged.

`vote_first` chooses validation states by:

```text
vote_acc descending
mean_individual_acc descending
mean_vote_margin descending
mean_invalid_rate ascending
earlier epoch
```

`vote_first` is the default best-state selection mode. `existing` remains
available only for compatibility with earlier scalar validation runs.

`mean_boundary_useful_diversity` remains a diagnostic: zero can mean either no
useful complementarity or that the team has already left a fragile boundary.
`best_prompts.json` stores it alongside the selection-key metrics and diagnostic
`selected_oracle_acc`.

## Final Metrics

Validation and test include:

```text
vote_acc
mean_individual_acc
best_individual_acc
oracle_acc
aggregation_gap
mean_vote_margin
mean_boundary_useful_diversity
mean_embedding_diversity
mean_useful_diversity
mean_invalid_rate
vote_tie_rate
mean_pairwise_double_fault
mean_pairwise_error_covariance
same_wrong_pair_rate
triple_joint_error_rate
majority_failure_tail_rate
coverage_depth_c1 ... coverage_depth_c5
mean_boundary_conditional_error
mean_pivotal_fix_rate
mean_pivotal_hold_rate
shared_error_rescue_rate
shared_error_creation_rate
boundary_shared_error_net_gain
dominant_wrong_cluster_size
gold_vs_largest_wrong_margin
```

`oracle_acc` means at least one agent is correct. `aggregation_gap = oracle_acc - plurality_vote_acc` diagnoses correct paths that did not become the final vote.

The default tie protocol is deterministic `random`: it is seeded by the run
seed and question hash. All matched settings must use the same
`vote_tie_break`; final exports include `vote_tie_rate`.

## Split Integrity And Reproducibility

`run_task_level_accuracy.py` performs a split preflight before launching a
task. A strict manifest must have zero normalized-question overlap for
opt/validation, opt/test, and validation/test. It writes split counts, overlap
counts, and SHA256 file hashes into `run_meta.json`. Reused files are marked
`paper_compatible_reused_file` and carry a leakage warning rather than being
called strict.

Each run also records the full Git commit, dirty-tree state, and protocol
version `vote_oriented_v7_residual_specialization`. Formal runs should start from a committed, clean
tree. Training checkpoints store a fingerprinted behavior configuration;
changing reward, guard, candidate-evaluation, selection, tie-break, model, or
TCS, or trajectory settings rejects resume instead of mixing optimization semantics. The current checkpoint schema is version 4; schema-v3 checkpoints fail clearly rather than silently restarting.

The matched v7 ablations are:

```text
shared_vote_pareto_tcs_static
shared_vote_pareto_tcs_boundary_selector
shared_vote_error_pareto_tcs
shared_vote_error_pareto_tcs_residual_specialization
shared_vote_error_pareto_tcs_residual_cycle_guard
```

All five use `reward_schedule_mode=static`. The historical
`shared_vote_pareto_tcs` setting remains available for earlier phase-adaptive
runs, but it is not part of this matched selector ablation.

They retain the same shared initialization, TCS budget, fixed optimization
pool, candidate batch, solver model, split, tie break, epochs, and beam size as
`shared_vote_pareto_tcs`. New v7 settings use a static reward schedule so prompt
text uniqueness cannot change reward weights.

## Resume

Interrupted runs can resume with the same settings and output root:

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --tasks disambiguation_qa `
  --settings shared_vote_pareto_tcs `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_vote_pareto `
  --resume_completed 1 `
  --resume_from_checkpoint 1
```

An incompatible checkpoint fails instead of silently restarting in the same directory.

## Recommended Commands

Smoke run:

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --tasks disambiguation_qa `
  --settings shared_baseline,shared_scalar_tcs_vote_first,shared_vote_pareto_tcs `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_vote_smoke `
  --epochs 1 `
  --train_size 20 `
  --val_size 20 `
  --test_size 40 `
  --update_every 10 `
  --candidate_eval_strategy fixed_pool `
  --candidate_eval_pool_size 50 `
  --candidate_eval_batch_size 20 `
  --candidate_reuse_recorded_rollouts 1
```

Matched scalar-vs-Pareto experiment:

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --tasks disambiguation_qa,geometric_shapes,ruin_names,sports_understanding `
  --settings shared_scalar_tcs_vote_first,shared_vote_pareto_tcs `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_bbh_vote_pareto_matched `
  --epochs 2 `
  --train_size 80 `
  --val_size 60 `
  --test_size 100 `
  --update_every 10 `
  --beam_size 3 `
  --num_candidates_per_parent 2 `
  --candidate_eval_strategy fixed_pool `
  --candidate_eval_pool_size 100 `
  --candidate_eval_batch_size 24 `
  --candidate_reuse_recorded_rollouts 1
```

Task-level MARS comparison remains external: MAD exports `accuracy_results.jsonl` by `task_id` and never depends on a local MARS checkout.
