# Multi-Agent Diversity

This repository performs evolutionary prompt search for a fixed team of reasoning agents. It changes prompts, not model weights.

The current line is V9 State-Conditioned Correlated-Error Optimization. Start with [method.md](method.md) for the implementation guide. V8 settings and artifacts remain available with unchanged semantics.

## Current V9 Settings

The main setting and matched ablations are:

```text
shared_state_conditioned_error_tcs
shared_v9_accuracy_only
shared_v9_accuracy_coverage
shared_v9_accuracy_coverage_c2split
shared_v9_accuracy_coverage_c2split_trace_tiebreak
```

They use:

```text
method_version = v9_state_conditioned_error
reward_mode = rollout_state_conditioned
candidate_selection_mode = state_conditioned_accuracy_first
best_state_selection_mode = state_conditioned_vote_first
```

V9 uses:

- C0/C1 correct-coverage repair, C2 vote conversion, and general accuracy as separate routes;
- representative, coverage, and option-stratified conversion candidate pools;
- representative-pool accuracy and invalid guards before state utility;
- an accuracy epsilon band before coverage or C2 candidates enter the Archive;
- an incumbent, accuracy, coverage, and conversion representative per agent;
- fixed-probe offline enumeration of at most `4^5 = 1024` teams;
- trace distance only as the final optional tie-break;
- no prompt-text, mechanism, capability, persona, or random-error diversity reward;
- training-time validation baseline guard and validation-only state selection;
- validation-only best epoch selection followed by one restored final test.

Wrong-answer dispersion has task value only when a C2 target remains wrong and reduces the dominant wrong cluster. C0, C1, and C3+ receive zero wrong-dispersion task gain.

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

## State-Conditioned Search Space

Candidates pass schema, completeness, and validity checks, then are evaluated against the active team. Safe candidates satisfy representative-pool target-accuracy and invalid guards plus directional C1/C2/C3 and Vote-loss guards. The Archive deduplicates by prompt hash and fixed-probe rollout signature. Joint combinations are evaluated offline from cached answer, correctness, invalid, and trace profiles.

V9 never performs legacy per-epoch beam refresh. Existing result directories remain readable by their recorded method version. Checkpoint v6 plus the V9 state marker fingerprints behavior settings; incompatible checkpoints fail explicitly.

Candidate accounting exposes a deduplicated funnel for TCS, Open exploration,
incumbent, and other candidates from generation through active selection.
Funnel identities are checkpointed for resume safety. Joint refresh also
records Safe profile coverage, excluded dirty shortlist counts, oldest
unprofiled Safe age, representative profile coverage, and representative
behavior distances. These diagnostics do not change shortlist or ranking.

Historical V8 rollout, mechanism, and lineage code remains available for reproduction and old-run analysis.

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
$OUT = "runs_v9_state_conditioned_smoke_$SHA"

& $PY scripts/run_task_level_accuracy.py `
  --workspace . `
  --manifest configs/task_level_comparison_strict_bbh_seed42.yaml `
  --tasks disambiguation_qa `
  --settings shared_state_conditioned_error_tcs `
  --seeds 42 `
  --dataset_format mars `
  --out_root $OUT `
  --run_concurrency 1 `
  --agents 5 `
  --epochs 1 `
  --train_size 20 `
  --val_size 20 `
  --test_size 20 `
  --update_every 10 `
  --beam_size 3 `
  --num_candidates_per_parent 2 `
  --candidate_eval_strategy fixed_pool `
  --candidate_eval_pool_size 20 `
  --candidate_eval_batch_size 12 `
  --candidate_batch_representative_size 6 `
  --candidate_batch_coverage_size 3 `
  --candidate_batch_conversion_size 3 `
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
  --settings shared_state_conditioned_error_tcs `
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
