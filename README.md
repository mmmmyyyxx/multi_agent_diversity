# Multi-Agent Diversity

This repository evolves prompts for a fixed five-agent reasoning team. Model weights and equal plurality vote weights stay unchanged.

The current V9 method is accuracy-first sequential state optimization. It updates one agent at a time, evaluates final acceptance on a fixed full probe, activates accepted prompts immediately, and uses diversity only to prevent collapse. All A0-A3 settings share an accuracy/invalid-only Stage A; Vote, state, and diversity differences begin in Stage B. It uses the true equal-weight plurality Vote delta as a secondary signal. Wrong-answer dispersion has no training value and is excluded from optimizer inputs. V9 is not rollout-QD and does not use rollout archives or enumerate prompt-team combinations.

Read [method.md](method.md) for the complete implementation guide.

## Current Settings

```text
shared_v9_sequential_accuracy
shared_v9_sequential_accuracy_state
shared_v9_sequential_accuracy_state_vote
shared_v9_sequential_accuracy_state_vote_diversity
```

The complete method is `shared_v9_sequential_accuracy_state_vote_diversity`. The unchanged rollout-diversity comparison is `shared_accuracy_rollout_embedding_tcs`.

Before a matched A0-A3 pilot, run `python scripts/preflight_v9_pilot.py --workspace .` from a clean committed tree.

## Deterministic Smoke

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
$SHA = (git rev-parse --short HEAD)
$OUT = "runs_v9_sequential_smoke_$SHA"

& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_v9_sequential_accuracy_state_vote_diversity `
  --seeds 42 `
  --dataset_format mars `
  --out_root $OUT `
  --run_concurrency 1 `
  --agents 5 `
  --epochs 1 `
  --train_size 10 `
  --val_size 8 `
  --test_size 8 `
  --update_every 10 `
  --num_candidates_per_parent 1 `
  --candidate_eval_strategy fixed_pool `
  --candidate_eval_pool_size 10 `
  --candidate_eval_batch_size 8 `
  --candidate_batch_representative_size 4 `
  --candidate_batch_coverage_size 2 `
  --candidate_batch_conversion_size 2 `
  --candidate_eval_concurrency 1 `
  --optimizer_parent_concurrency 1 `
  --train_rollout_concurrency 1 `
  --eval_solver_call_concurrency 5 `
  --candidate_eval_execution_mode factorized_cached `
  --candidate_reuse_recorded_rollouts 1 `
  --solver_rollout_singleflight 1 `
  --aggregation_mode plurality `
  --vote_tie_break random `
  --state_bottom2_reward_enabled 0 `
  --resume_from_checkpoint 1
```

The smoke checks execution and artifact integrity only. It does not establish a performance result.

## Resume

Repeat the same command with the same output root and behavior-affecting arguments, plus:

```powershell
--resume_completed 1
--resume_from_checkpoint 1
```

V9 requires checkpoint version 3. Incompatible checkpoints fail explicitly instead of silently restarting in the same directory.

## Outputs

Each V9 run writes:

```text
history.json
best_prompts.json
prompt_history.json
update_logs.jsonl
sequential_update_history.jsonl
solver_rollout_records.jsonl
training_checkpoint.json while incomplete
run_meta.json
llm_calls.jsonl
cost_summary.json
final_summary.json
```

The task runner also writes root-level accuracy tables and summaries. Historical V8 runs remain readable with their recorded method versions.
