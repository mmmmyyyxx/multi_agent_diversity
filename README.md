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
```

`vote_pareto` uses only vote gain, vote loss, and target-agent accuracy as Pareto objectives. Validation selection supports `existing` and `vote_first`; `vote_first` is the default.

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

## Emergent Behavioral Specialization

Emergent specialization is optional and disabled by default. When enabled, all
agents still begin with the same neutral behavior profile; there are no fixed
roles and no agent-ID-specific rules. An agent's profile is an EMA of positive,
supported behavior transitions from prompts that actually became active.
Rejected candidates never update it.
Contexts below `specialization_min_context_support` are ignored as noisy evidence.

The same optimization/train candidate-evaluation rows provide task-independent
contexts, transition vectors, and compact behavior fingerprints. No extra
solver rollout, semantic-clustering call, validation access, or test access is
used. Profile affinity is a small agent-selection bonus and trajectory
alignment is only a late Pareto tie-break, never a reward or Pareto objective.

The behavioral cycle guard rejects exact prompt repeats and high-similarity
historical behavior only when the candidate has no meaningful vote, target
accuracy, or margin improvement. The prompt trust region permits local edits
and requires stronger vote evidence for large rewrites after warmup.

Matched settings are:

```text
shared_vote_pareto_tcs
shared_vote_pareto_tcs_cycle_guard
shared_vote_pareto_tcs_emergent
```

`trajectory_events.jsonl` records accepted/rejected transitions. Profile
entropy, pairwise JSD, alignment, and rejection rates are diagnostics only and
never participate in `vote_first`. JSD growth alone is not success; interpret
it together with vote and individual accuracy.

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

`run_meta.json` records the Git commit, dirty-tree state, protocol version
`vote_oriented_v6_emergent_specialization`, and checkpoint schema version 3.
Formal runs should start from a clean, committed tree; checkpoints also reject
resume when behavior-affecting reward, evaluation, selection, model, tie-break,
TCS, or emergent-trajectory configuration differs.

## Cost Reporting

Each run writes `llm_calls.jsonl` and `cost_summary.json` with solver, optimizer, evaluator, token, cost, and latency statistics. Cost is reported only: it never adds a budget limit, changes ranking, or stops training.
