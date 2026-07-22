# Multi-Agent Diversity

This repository implements one formal method: **Peer-State Counterfactual Prompt Optimization** (`peer_state_counterfactual_v1`). It centrally optimizes five solver prompts for equal-weight plurality vote.

```text
Team rollout
-> peer-conditioned oracle opportunity
-> residual responsibility assignment
-> responsibility-conditioned Teacher-Critic-Student generation
-> paired candidate rollout
-> competence-constrained vote-first update
```

Vote accuracy is the objective, individual competence is constrained, and soft vote utility is a dense search signal. The canonical method uses shared-identical initialization and tie-as-abstain.

## Experiment Settings

| Setting | Purpose |
|---|---|
| `shared_baseline` | No prompt optimization |
| `shared_independent_accuracy_tcs` | Independent accuracy optimization |
| `shared_peer_state_credit_round_robin` | Peer-state vote optimization with round-robin targets |
| `shared_peer_state_responsibility` | Adds dynamic residual responsibility and online refresh |
| `shared_peer_state_full` | Adds responsibility-conditioned TCS to the previous setting |

Detailed semantics and the isolated ablation contract are in [method.md](method.md).

## Preflight

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"

& $PY scripts/run_task_level_accuracy.py --help
& $PY -m pytest -q
& $PY -m compileall multi_dataset_diverse_rl scripts tests
git diff --check
& $PY scripts/preflight_peer_state.py --workspace .
& $PY scripts/deterministic_peer_state_smoke.py
```

The deterministic smoke uses local fake models and makes no external API calls.

## Real API Smoke Template

Set role-specific endpoint variables if solver, optimizer, and evaluator use different services. The optimizer has independent `optimizer_api_key_env` and `optimizer_base_url_env` configuration.

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"

& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_peer_state_full `
  --seeds 42 `
  --dataset_format mars `
  --out_root runs_peer_state_api_smoke `
  --epochs 1 `
  --train_size 12 `
  --val_size 8 `
  --test_size 12 `
  --update_every 6 `
  --candidate_eval_pool_size 12 `
  --num_candidates_per_parent 2 `
  --stage_b_candidate_budget 2 `
  --eval_solver_call_concurrency 2 `
  --max_total_llm_calls 800 `
  --max_total_tokens 600000 `
  --resume_completed 0 `
  --resume_from_checkpoint 0
```

## Outputs

Each run writes metadata and exact identity, training history, selected prompts, final metrics, typed peer-state and responsibility diagnostics, TCS context budgets, candidate funnels and guard rejections, per-attempt LLM logs, and cost totals. Root-level `accuracy_results.jsonl`, `accuracy_results.csv`, and `experiment_runs.jsonl` provide the matched experiment matrix.
