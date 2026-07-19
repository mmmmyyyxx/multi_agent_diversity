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
- a six-item Safe archive plus a three-item joint representative beam per agent;
- fixed-probe behavioral profiles as the primary differentiation signal;
- normalized mechanism sequence and embedding distance as secondary evidence;
- offline enumeration of all `3^5 = 243` beam teams;
- event-driven joint refresh with dirty-prompt probe shortlists;
- one TCS repair and one direct open mechanism-exploration channel per supported parent;
- hierarchical integer-count quality bands before team diversity selection;
- committed lineage anchors, drift control, peer-collapse prevention, and switch hysteresis.

Prompt textual diversity is not an optimization target. Diversity never compensates for competence failure. Early search permits symmetry breaking; late search stabilizes useful per-agent lineages.

## Search-Space Preservation

Initial candidates pass a cheap schema, completeness, duplicate, and mechanism-step screen. A bounded feedback-aware refill is triggered when the batch lacks two Safe non-incumbents, a Safe repair, or a Safe distinct mechanism. Safe candidates can participate in team selection; mildly regressing but novel Probation branches can only reproduce in later updates; catastrophic candidates are discarded. Team-relative rescue, shared-error, and same-wrong metrics are recomputed for each joint combination. Two deterministic probe folds, hierarchical count bands, active-change limits, and two-snapshot lineage commitment reduce probe overfitting without expanding solver calls for cached prompt-question pairs.

The setting's historical V8.2 safe/exploit/explore behavior has been replaced. V8 never performs the legacy per-epoch beam refresh. Joint refresh is event-driven and only probes new dirty prompts; team enumeration is offline. Existing result directories remain readable by their recorded method version. Checkpoint v6 stores the new refresh/generation policy state and older incompatible Stable-QD checkpoints fail explicitly.

Specific unknown mechanisms can enter stable semantic families when they pass the specificity gate. Refill is checked after raw evaluation, archive compression, and representative selection. Joint quality uses a frontier of real prompt teams, never a synthetic component-wise maximum.

## Architecture

`Config` is composed from 11 sections while preserving old flat CLI names as aliases. Named experiments are sparse presets with validated overrides. `system.py` is a small public facade; generation, evaluation, metrics, QD policy, lifecycle, and persistence live in dedicated packages. Checkpoint v6 stores semantic families and the real-team quality-anchor frontier and explicitly validates resume compatibility.

Start with `method.md`, then use `VERSION_PRESET_MAP.md`, `V8_ACCEPTANCE_AUDIT.md`, and `RUNS_CATALOG.md` for version, implementation, and local evidence maps.

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

Report plurality accuracy with mean individual accuracy, bottom-2, C1, C2, and oracle coverage. Stable-QD diagnostics include pairwise behavior and mechanism distance, lineage drift and status, peer collapse, quality-floor/final-band counts, fold-quality rejections, selected active sources, niche occupancy, starvation, Probation conversion/expiry, and stable specialization score.

Cost fields are reporting-only and never alter ranking or stop training.

## Tests

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
git diff --check
```

## Historical Methods

Legacy reward modes and scalar/Pareto settings remain available for historical comparison. Match runs by `method_version`, setting, split, seed, models, and evaluation budget. Do not interpret a shared setting name as identical behavior across method versions.
