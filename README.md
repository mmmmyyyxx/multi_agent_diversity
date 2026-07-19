# Multi-Agent Diversity

This repository performs evolutionary prompt search for a fixed team of reasoning agents. It changes prompts, not model weights.

The current V8 method is Stable Quality-Diversity Lineage Optimization. Start with [method.md](method.md) for the complete implementation guide. See [V8_COMPETENCE_DEPTH_METHOD.md](V8_COMPETENCE_DEPTH_METHOD.md), [V8_2_HYBRID_PROGRESSIVE_METHOD.md](V8_2_HYBRID_PROGRESSIVE_METHOD.md), and [V8_EXPERIMENT_PLAN.md](V8_EXPERIMENT_PLAN.md) for focused notes.

## Current V8 Setting

The existing setting name is intentionally unchanged:

```text
shared_vote_tcs_competence_depth2_progressive_residual_hybrid
```

It now resolves to:

```text
method_version               = v8_stable_qd_lineage
beam_policy_version          = quality_diversity_archive_v1
active_team_selector_version = joint_quality_diversity_v1
lineage_policy_version       = stable_lineage_anchor_v1
mechanism_distance_version   = mechanism_sequence_embedding_v1
```

The method uses:

- competence and actual plurality-boundary signals to choose update targets;
- Teacher-Critic-Student to generate task-repair and mechanism-alternative prompts;
- hard competence and validity guards before diversity is considered;
- a three-slot per-agent quality-diversity archive;
- fixed-probe behavioral profiles as the primary differentiation signal;
- normalized mechanism sequence and embedding distance as secondary evidence;
- offline enumeration of all `3^5 = 243` beam teams;
- a quality-feasible epsilon-Pareto frontier before team diversity selection;
- committed lineage anchors, drift control, peer-collapse prevention, and switch hysteresis.

Prompt textual diversity is not an optimization target. Diversity never compensates for competence failure. Early search permits symmetry breaking; late search stabilizes useful per-agent lineages.

The setting's historical V8.2 safe/exploit/explore behavior has been replaced. Existing result directories remain readable by their recorded method version, but old V8 checkpoints are rejected explicitly.

## Voting

The canonical aggregator is plurality: the most frequent answer wins, with the configured deterministic tie-break for top ties. `aggregation_mode=majority` remains a compatibility alias for plurality.

Candidate counterfactuals, pivotal-boundary metrics, joint team enumeration, validation, and test evaluation use the same aggregator.

## Data Protocol

Prompt optimization uses the optimization training split. A fixed probe sampled from that split drives competence scheduling, behavior profiles, and joint active-team selection. Validation selects the best state. Final test runs after restoring validation-selected prompts.

Strict manifests check normalized-question overlap and record split counts and hashes in `run_meta.json`.

## Targeted Smoke

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
$SHA = (git rev-parse --short HEAD)
$OUT = "runs_v8_stable_qd_lineage_smoke_$SHA"

& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_vote_tcs_competence_depth2_progressive_residual_hybrid `
  --seeds 42 `
  --dataset_format mars `
  --out_root $OUT `
  --run_concurrency 1 `
  --agents 5 `
  --epochs 2 `
  --train_size 20 `
  --val_size 20 `
  --test_size 20 `
  --update_every 10 `
  --beam_size 3 `
  --num_candidates_per_parent 2 `
  --candidate_eval_strategy fixed_pool `
  --candidate_eval_pool_size 20 `
  --candidate_eval_batch_size 10 `
  --candidate_eval_repeats 1 `
  --candidate_eval_execution_mode factorized_cached `
  --candidate_reuse_recorded_rollouts 1 `
  --solver_rollout_singleflight 1 `
  --candidate_eval_prompt_dedup 1 `
  --candidate_eval_cache_logging 1 `
  --aggregation_mode plurality `
  --vote_tie_break random `
  --eval_test_each_epoch 0 `
  --resume_from_checkpoint 1
```

This smoke checks execution integrity. It is not an accuracy claim and does not require a committed lineage in two epochs.

## Resume

Resume with exactly the same behavior-affecting arguments, output root, and split manifest:

```powershell
& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_vote_tcs_competence_depth2_progressive_residual_hybrid `
  --seeds 42 `
  --dataset_format mars `
  --out_root $OUT `
  --resume_completed 1 `
  --resume_from_checkpoint 1
```

Include the original run's size, beam, evaluation, and concurrency arguments in a real resume command. Behavior fingerprint mismatches fail rather than silently restarting and contaminating the directory.

## Outputs

Each run writes:

```text
history.json
best_prompts.json
prompt_history.json
update_logs.jsonl
solver_rollout_records.jsonl
quality_diversity_archive.jsonl
joint_team_selection_history.jsonl
behavior_profile_history.jsonl
lineage_history.jsonl
training_checkpoint.json while incomplete
run_meta.json
llm_calls.jsonl
cost_summary.json
```

The task runner also writes `accuracy_results.csv`, `accuracy_results.jsonl`, and summaries at the output root.

## Metrics

Report plurality accuracy with mean individual accuracy, bottom-2, C1, C2, and oracle coverage. Stable-QD diagnostics include pairwise behavior and mechanism distance, lineage drift and status, peer collapse, feasible/frontier counts, selected active sources, niche occupancy, and stable specialization score.

Cost fields are reporting-only and never alter ranking or stop training.

## Tests

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
git diff --check
```

## Historical Methods

Legacy reward modes and scalar/Pareto settings remain available for historical comparison. Match runs by `method_version`, setting, split, seed, models, and evaluation budget. Do not interpret a shared setting name as identical behavior across method versions.
