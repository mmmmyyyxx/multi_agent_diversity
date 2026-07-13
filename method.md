# Method: Vote-Oriented Multi-Agent Prompt Search

## Overview

This project evolves prompts for a fixed team of solver agents. It does not update model weights and it is not policy-gradient RL. Reward only ranks candidate prompts inside a per-agent beam search.

The primary metric is `vote_acc`: agents answer the same question and the plurality-vote answer is compared with gold. The public configuration name `aggregation_mode=majority` is retained, although multi-class behavior is technically plurality vote.

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

A pivotal fix asks whether changing an incorrect agent to gold would make a wrong or tied vote clearly correct. Dominant-wrong redundancy finds incorrect agents repeating the largest wrong cluster near the boundary. Only positive-score agents are selected.

## Teacher-Critic-Student

The default prompt-evolution architecture is:

```bash
--optimizer_architecture teacher_critic_student
```

Teacher turns abstract vote diagnostics into a guiding question. Critic audits it; rejection feeds back to Teacher for rewrite and re-review. After `teacher_critic_max_rounds`, the highest Critic-scored question can proceed with `teacher_question_forced_best_score=true`. Student returns strict JSON candidates.

Teacher sees aggregate target errors, vote failures, pivotal-fix counts, dominant wrong-answer redundancy, and invalid-output signals. It does not receive raw gold answers or sample-specific task content.

New TCS candidates must include Teacher/Critic/Student provenance. Existing beam candidates are exempt. Audit a run with:

```powershell
python scripts/audit_tcs_run.py <run_dir_or_root>
```

## Candidate And State Selection

`scalar_reward` keeps candidates by scalar reward.

`vote_pareto` first applies the target-accuracy and invalid-rate guards, then uses exactly three Pareto objectives:

```text
maximize vote_gain_rate
minimize vote_loss_rate
maximize candidate_target_accuracy
```

Within a Pareto rank, the deterministic order is vote delta, lower vote loss, vote gain, vote-margin delta, target accuracy, boundary-diversity delta, lower invalid rate, then candidate ID.

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
```

`oracle_acc` means at least one agent is correct. `aggregation_gap = oracle_acc - vote_acc` diagnoses correct paths that did not become the final vote.

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
version `vote_oriented_v1`. Formal runs should start from a committed, clean
tree. Training checkpoints store a fingerprinted behavior configuration;
changing reward, guard, candidate-evaluation, selection, tie-break, model, or
TCS settings rejects resume instead of mixing optimization semantics.

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
