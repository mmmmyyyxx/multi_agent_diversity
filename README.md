# Multi-Agent Diversity

This repository performs evolutionary prompt search for a fixed team of reasoning agents. It changes prompts, not model weights.

The current V8 line is Accuracy-First Rollout-Diversity Prompt Optimization. Start with [method.md](method.md) for the complete implementation guide. Historical Stable-QD notes remain available for reproducing earlier runs.

## Current V8 Settings

The rollout-only settings are:

```text
shared_accuracy_rollout_embedding_tcs
shared_vote_ready_rollout_diversity_tcs
```

They resolve to:

```text
method_version = v8_accuracy_rollout_embedding
method_version = v8_rollout_qd_vote_ready
```

The methods use:

- observed solver errors and actual plurality-boundary signals to choose update targets;
- Teacher-Critic-Student repair plus direct open rollout exploration;
- hard accuracy, validity, Vote-loss, and C3-loss guards before diversity;
- a six-item Safe archive plus a three-item joint representative beam per agent;
- fixed-probe answer, correctness, invalid, and trace-embedding profiles;
- rollout-signature deduplication instead of prompt or mechanism niches;
- offline enumeration of all `3^5 = 243` beam teams;
- one TCS repair and one direct open rollout-exploration channel per supported parent;
- Vote and C3 before rollout diversity in joint team selection;
- validation-only best epoch selection followed by one restored final test.

Prompt text, optimizer-reported mechanisms, and artificial capability labels are not optimization evidence. Diversity never compensates for quality failure.

Acceptance diagnostics are reporting-only. Each prediction records correct
agent count, gold vote count, largest wrong cluster, plurality margin, Oracle
status, top-tie status, invalid-agent count, and normalization anomaly status.
Split summaries report C0/C1/C2/C3+ conversion, Oracle-to-Vote conversion,
C1/C2 vote failures, top-tie wins/losses, and wrong-cluster concentration.
These fields reuse the canonical plurality vote and do not affect reward or
team selection.

Critic direct approval uses strict optional boolean parsing. Boolean `true`
and normalized string `"true"` are accepted declarations; false strings,
numbers, collections, and unknown strings reject direct pass. Missing or null
`passed` remains compatible with score-only historical Critic output.
Rewrite and forced-best behavior is unchanged.

## Rollout Search Space

Candidates pass the minimal rollout schema, completeness, and validity checks, then are evaluated against the active team. Safe candidates satisfy target-accuracy, invalid-rate, Vote-loss, and C3-loss guards. The archive deduplicates by prompt hash and fixed-probe rollout signature. Joint combinations are evaluated offline from cached fixed-probe answer, correctness, invalid, and trace profiles.

The rollout methods never perform legacy per-epoch beam refresh. Existing result directories remain readable by their recorded method version. Checkpoint v6 fingerprints all rollout objective and guard settings; incompatible checkpoints fail explicitly.

Candidate accounting exposes a deduplicated funnel for TCS, Open exploration,
incumbent, and other candidates from generation through active selection.
Funnel identities are checkpointed for resume safety. Joint refresh also
records Safe profile coverage, excluded dirty shortlist counts, oldest
unprofiled Safe age, representative profile coverage, and representative
behavior distances. These diagnostics do not change shortlist or ranking.

Historical mechanism and lineage code remains available only for old settings and run analysis.

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
$OUT = "runs_v8_rollout_qd_smoke_$SHA"

& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_accuracy_rollout_embedding_tcs,shared_vote_ready_rollout_diversity_tcs `
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
  --settings shared_accuracy_rollout_embedding_tcs,shared_vote_ready_rollout_diversity_tcs `
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
final_summary.json
```

The task runner also writes `accuracy_results.csv`, `accuracy_results.jsonl`, and summaries at the output root.

The final preformal acceptance smoke completed in
`runs_v8_preformal_acceptance_799df8c` with 1151 total LLM calls, two joint
refreshes, zero skipped refreshes, zero legacy refresh calls, and zero
team-level solver calls. It produced Vote 0.55, Mean 0.54, Oracle 0.95, and
Oracle-to-Vote conversion 0.5789. This is an execution-integrity check, not a
formal result or method comparison.

The 20/20/20 run above is the complete pre-commit acceptance smoke. The final
frozen commit is additionally checked by one separate 8/8/8, one-epoch clean
smoke under `runs_v8_postcommit_clean_smoke_<frozen-short-sha>`. That smaller
run verifies Git provenance, real API execution, final artifacts, and
checkpoint cleanup; its accuracy is not a method result.

## Metrics

Report plurality accuracy with best/mean/bottom-2 individual accuracy, C1/C2/C3, Oracle-to-Vote conversion, gold margin, wrong concentration, same-wrong rate, rollout diversity, invalid rate, active sources, funnel conversion, and cost.

Cost fields are reporting-only and never alter ranking or stop training.

## Tests

```powershell
$PY = "D:\Anaconda\envs\DL\python.exe"
& $PY -m pytest -q
git diff --check
```

## Historical Methods

Legacy reward modes and scalar/Pareto settings remain available for historical comparison. Match runs by `method_version`, setting, split, seed, models, and evaluation budget. Do not interpret a shared setting name as identical behavior across method versions.
