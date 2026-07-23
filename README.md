# Multi-Agent Diversity

This repository implements one formal method: **Peer-State Counterfactual Prompt Optimization** (`peer_state_counterfactual_v2`). It centrally optimizes five solver prompts for equal-weight plurality vote.

```text
Team rollout
-> peer-conditioned oracle opportunity
-> residual responsibility assignment
-> responsibility-conditioned Teacher-Critic-Student generation
-> paired candidate rollout
-> competence-constrained vote-first update
```

Vote accuracy is the objective, individual competence is constrained, and soft vote utility is a dense search signal. The canonical method uses shared-identical initialization, tie-as-abstain, and one persistent prompt-question cache shared by optimization, validation, test, and matched settings with the same seed.

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

The V2 TCS gate treats Critic as a hard-legality auditor. Critic must restate
derived current/gold transition facts exactly; state misreads are retried.
Task-internal specialization and uncertain empirical benefit are not blockers,
and Critic score is diagnostic only. Stage A/B rollout determines whether an
approved hypothesis actually works.

The task runner automatically creates `<out_root>/_shared_solver_cache.sqlite`.
The cache key includes solver request identity, output contract, parser, decoding
replica seed, prompt hash, and question hash, but never the setting name.

Before a real run, pass the intended experiment arguments to preflight. It validates splits, API roles, budgets, output identity, and RunIdentity without making model calls:

```powershell
& $PY scripts/preflight_peer_state.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_baseline,shared_peer_state_full `
  --seeds 42 `
  --out_root runs_peer_state_api_smoke `
  --train_size 8 --val_size 8 --test_size 8 `
  --num_candidates_per_parent 1 --stage_b_candidate_budget 1 `
  --max_total_llm_calls 500 --max_total_tokens 300000
```

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

Transport and resume are tested separately from method quality:

```powershell
& $PY scripts/critic_calibration_replay.py `
  --out_dir runs_critic_calibration_v2 `
  --evaluator_model deepseek-chat `
  --critic_json_max_retries 2 `
  --max_total_llm_calls 30 `
  --max_total_tokens 60000

& $PY scripts/real_api_role_transport_smoke.py `
  --out_dir runs_role_transport_smoke `
  --answer_format option_letter `
  --num_candidates_per_parent 1 `
  --max_total_llm_calls 20 `
  --max_total_tokens 30000

& $PY scripts/real_api_resume_smoke.py --help
```

The calibration replay calls only the evaluator. It must accept at least one
valid task-internal repair, reject every memorizing fixture, and parse all
derived fact restatements before another end-to-end smoke is attempted.

The transport smoke calls solver, Teacher, Critic, and Student independently;
Student transport does not depend on Critic approval. The resume smoke waits for
an exact atomic checkpoint, terminates the child process, resumes two copies of
the same checkpoint through the shared solver cache, and compares the resulting
prompts, decisions, responsibility records, validation history, final metrics,
RunIdentity, and cache contents.

## Outputs

Each run writes metadata and exact identity, training history, selected prompts, final metrics, typed peer-state and responsibility diagnostics, recursive TCS context-field audits, per-round TCS parse/approval logs, exact candidate and Stage A funnels, invalid solver-output excerpts, per-attempt LLM logs, and cost totals. Root-level `accuracy_results.jsonl`, `accuracy_results.csv`, and `experiment_runs.jsonl` provide the matched experiment matrix.
