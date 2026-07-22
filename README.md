# Multi-Agent Diversity

This repository implements **Peer-State Counterfactual Prompt Optimization** for a fixed five-agent reasoning ensemble. Model weights and equal plurality-vote weights remain unchanged. The optimizer estimates each agent-example pair's counterfactual team value under current peer answers, assigns residual team failures to Students, and accepts only competence-feasible prompt updates.

The method does not optimize generic diversity. Complementary behavior is expected to emerge because different Students own different residual responsibilities.

Read [method.md](method.md) for the complete formulation and implementation map.

## Settings

```text
shared_baseline
shared_independent_accuracy_tcs
shared_peer_state_credit_round_robin
shared_peer_state_responsibility
shared_peer_state_full
```

Old setting names are intentionally unsupported.

## Preflight

Run from a clean committed tree before an experiment:

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY scripts/preflight_peer_state.py --workspace .
```

## Task-Level Run

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"

& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_peer_state_full `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_peer_state_disambiguation_seed42 `
  --epochs 3 `
  --train_size 75 `
  --val_size 50 `
  --test_size 125 `
  --update_every 10 `
  --candidate_eval_pool_size 75 `
  --eval_solver_call_concurrency 20 `
  --resume_from_checkpoint 1 `
  --resume_completed 1
```

Resume by repeating the exact command with the same output root. Only checkpoint version 1 with `method_version=peer_state_counterfactual_v1` is accepted.

## Outputs

```text
run_meta.json
history.json
best_prompts.json
final_summary.json
peer_state_history.jsonl
responsibility_assignments.jsonl
candidate_decisions.jsonl
prompt_memory_history.jsonl
llm_calls.jsonl
training_checkpoint.json while incomplete
```

Final test data is used only after validation has selected `best_prompts.json`.
