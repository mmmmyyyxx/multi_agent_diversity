# Multi-Agent Diversity

This repository implements vote-oriented evolutionary prompt search for multi-agent reasoning. It evolves prompts, not model weights. A fixed team of solver agents answers each question, and the primary evaluation metric is final team `vote_acc`.

See [method.md](method.md) for the implementation guide.

## Current Method

The recommended method is `vote_useful_diversity`:

- preserve target-agent accuracy with a guard;
- reject candidates that worsen invalid-output rate;
- optimize direct `vote_delta`;
- use `vote_margin_delta` before a vote flip is available;
- reward only the positive part of `boundary_useful_diversity_delta` near a gold-vs-wrong vote boundary, so a stronger correct vote is never penalized for leaving that boundary.

`oracle_acc`, aggregation gap, trace embedding diversity, answer-level useful diversity, coverage, and rescue statistics are diagnostics. They are not candidate-selection objectives.

Supported reward modes are:

```text
accuracy_only
guarded_diversity
vote_useful_diversity
```

Candidate selection supports:

```text
scalar_reward
vote_pareto
vote_error_pareto
```

`vote_pareto` uses only vote gain, vote loss, and target-agent accuracy as Pareto objectives. Validation selection supports `existing` and `vote_first`; `vote_first` is the default.
`vote_error_pareto` is the v7 four-objective variant that also maximizes
`boundary_shared_error_net_gain`; it does not alter legacy `vote_pareto`.

Candidate prompts are evaluated only on optimization/train data. Validation is
used only for epoch and best-state selection; test is evaluated once after the
selected prompts are restored. Candidate logs report raw and clipped boundary
gain, reward components, and total versus unique evaluated questions.

## Teacher-Critic-Student

The default optimizer architecture is Teacher-Critic-Student (TCS). Teacher receives abstract vote-oriented diagnosis, Critic audits and can request rewrites, and Student emits strict JSON prompt candidates. TCS calls, provenance, retries, and JSON repair are logged.

Audit an existing run without changing it:

```powershell
python scripts/audit_tcs_run.py <run_dir_or_root>
```

## Boundary-Aware Residual Specialization (v7)

V7 keeps vote contribution contexts separate from capability residual
families. All agents start from the same zero-evidence capability profile.
Only accepted active prompts add cumulative, reliability-shrunk evidence for
task-independent mechanisms such as relation tracking, qualifier handling,
option comparison, contradiction checking, or final verification. A rare
pivotal improvement is shrunk but never discarded solely because support is
one. Profiles update every two accepted edits by default or at epoch end.

The v7 selector prioritizes pivotal and near-boundary errors over raw error
volume. Candidate evaluation reuses the existing paired rows to log pivotal
rescue/loss, shared-error rescue/creation, and same-wrong-cluster transitions;
no solver calls are added. The full setting also checks accepted and rejected
behavior archives and requires one declared local mechanism edit from Student.

Matched v7 settings are:

```text
shared_vote_pareto_tcs_boundary_selector
shared_vote_error_pareto_tcs
shared_vote_error_pareto_tcs_residual_specialization
shared_vote_error_pareto_tcs_residual_cycle_guard
```

These use a static reward schedule. Prompt hash uniqueness remains diagnostic
and cannot alter v7 reward weights or the accuracy guard.

## Task-Level Accuracy Comparison

MARS and this repository run separately. MAD does not read or depend on a local MARS checkout. Run MAD at the same `task_id` granularity, export `accuracy_results.jsonl`, then join it with a separately generated MARS `summary.csv`.

The main MAD metric is `vote_acc`. Since it is multi-agent vote accuracy, always report `mean_individual_acc` and `best_individual_acc` alongside it.

```powershell
python scripts/run_task_level_accuracy.py `
  --manifest configs/task_level_comparison.yaml `
  --benchmarks BBH,MMLU `
  --settings shared_baseline,shared_guarded_beam,bank_guarded_beam `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_task_level_accuracy
```

```powershell
python scripts/compare_external_accuracy.py `
  --mars_summary path/to/mars/summary.csv `
  --mad_results runs_task_level_accuracy/accuracy_results.jsonl `
  --out_csv comparison/mars_vs_mad_accuracy.csv `
  --out_md comparison/mars_vs_mad_accuracy.md
```

If a task manifest reuses one CSV for train, validation, and test, describe that result as a `paper-compatible setting`, not a strict no-leakage split. Strict manifests are checked before launch for normalized-question overlap across opt/validation/test and record counts plus SHA256 file hashes in `run_meta.json`.

## Vote-Pareto Smoke Run

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

## Matched Scalar Vs Pareto Run

The matched settings are `shared_scalar_tcs_vote_first` and `shared_vote_pareto_tcs`. Both use shared initialization, TCS, fixed-pool candidate evaluation, `vote_useful_diversity`, and `vote_first`. Only candidate selection differs.

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

## Resume

Resume an incomplete run with the same output root and configuration:

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

Incompatible checkpoints fail clearly instead of silently restarting in the same output directory.

`run_meta.json` records the Git commit, dirty-tree state, and protocol version
`vote_oriented_v7_residual_specialization`. Checkpoint schema version is 4.
Formal runs should start from a clean, committed tree; checkpoints also reject
resume when behavior-affecting reward, evaluation, selection, model, tie-break,
TCS, or trajectory configuration differs. Schema-v3 checkpoints are rejected
clearly instead of being resumed under v7 state semantics.

## Cost Reporting

Each run writes `llm_calls.jsonl` and `cost_summary.json` with solver, optimizer, evaluator, token, cost, and latency statistics. Cost is reported only: it never adds a budget limit, changes ranking, or stops training.
