# v12 Seed46 Pilot Report

```text
Pilot method execution: BASIC PASS
Pilot official certification: PASS after offline revalidation
```

## Scope

- Task: `disambiguation_qa`
- Seed: `46`
- Matrix: official S0-S5 six-setting pilot
- Planned updates: 8 for every optimized setting
- Final test: disabled
- Source commit: `c81f9525e6009189ad1c50ab53204822b6742875`
- Method version: `member_aware_peer_state_v12`
- Checkpoint version: `21`

This directory contains only aggregate, sanitized facts. It excludes prompts,
questions, gold answers, model responses, API endpoints, credentials, caches,
checkpoints, and local paths.

## Online gates

The run-specific preflight passed with no errors. The one authorized real-API
transport smoke also passed:

- Solver returned a valid strict answer without truncation.
- Teacher and Critic returned schema-valid JSON without truncation.
- Student returned two valid candidates without truncation.
- All four role calls completed successfully.

## Pilot lifecycle

- Complete runs: 6/6
- Optimized settings completed: 8/8 updates each
- Final test evaluations: 0
- Validation evaluations: 0
- Infrastructure-failed updates: 0
- Frozen initialization: matched
- Cumulative task-seed cache-chain continuity: passed for all six settings
- Adjacent-setting module isolation: passed for all four comparisons
- Total pilot tokens: 3,845,691
- Accepted updates across the matrix: 12

Accepted updates by setting:

| Setting | Accepted updates |
| --- | ---: |
| `shared_baseline` | 0 |
| `shared_generic_evolution` | 1 |
| `shared_vote_state_diagnosis` | 3 |
| `shared_member_aware_responsibility` | 4 |
| `shared_responsibility_conditioned_evolution` | 3 |
| `shared_full_rcru` | 1 |

The initial all-settings process completed S0 and S1, then terminated after
trying to compare empty per-agent test arrays while
`final_test_enabled=0`. The remaining S2-S5 runs were completed sequentially
in the same pilot root using the same frozen initialization and cumulative
comparison cache. This preserved the frozen source and cache-chain
continuity.

## Original pilot gate

```text
gate = FAIL
complete_run_count = 6
blocker_count = 6
major_count = 0
```

The six blockers are caused by the no-test audit path:

1. S1 completed its run artifacts, but the runner failed before adding four
   zero-valued no-test drift fields to its comparison manifest.
2. The official audit treated absent final-test member counts as mismatches
   for all five optimized settings, although this pilot requires final testing
   to be disabled.

The original audit did not report incomplete runs, cache-chain discontinuity,
frozen-initialization mismatch, module-isolation failure, or infrastructure
failure.

## Offline official revalidation

The no-test runner and audit paths were repaired without changing the method,
RCRU, checkpoint version, or any original pilot artifact. No API call and no
pilot rerun was performed.

The existing artifacts were then revalidated in explicit offline mode:

```text
audit_version = final_method_stage_gate_v5
audit_mode = offline_existing_artifact_revalidation
gate = PASS
complete_run_count = 6
blocker_count = 0
major_count = 0
```

Identity separation:

- Run source commit:
  `c81f9525e6009189ad1c50ab53204822b6742875`
- Auditor commit:
  `16318b15359344563e85e6419a4e6d1b4c42ab44`
- Method runtime semantics changed: false
- Original run artifacts modified: false

The audit normalized all six legacy no-test manifests using
`legacy_no_test_manifest_v1`. Normalization was permitted only because every
run independently established:

- `final_test_enabled=false`
- final-test evaluation count `0`
- no selected or initial test metrics
- no test observation hashes or member observations
- no test use for training or selection

The normalization records test observation, member-count, and drift status as
`not_applicable`; it does not synthesize zero-valued member counts.

Revalidated protocol facts:

- Cache-chain continuity: 6/6
- Adjacent-setting isolation: 4/4
- Final-test evaluations: 0
- Validation evaluations: 0
- Infrastructure failures: 0
- Unresolved findings: 0

## S5 RCRU empirical health

### Candidate flow

- Full RCRU candidate evaluations: 11
- Layer-1 passes: 2
- Layer-2 passes: 1
- Layer-3 passes: 1
- Accepted updates: 1
- Candidate acceptance rate: 1/11

All 11 sanitized candidate records have finite bootstrap LCB values. No
layer-2 or layer-3 candidate is missing its bootstrap LCB.

The accepted candidate had:

- target gain: `+2`
- vote gain: `0`
- positive support: `3`
- negative support: `0`
- bootstrap LCB: `1.0`

### Rejection reasons

| Reason | Count |
| --- | ---: |
| `target_regression` | 9 |
| `no_vote_or_lane_progress` | 4 |
| `insufficient_lane_support` | 3 |
| `team_vote_regression` | 0 |
| `terminal_invalid_regression` | 0 |
| `negative_lane_bootstrap_lcb` | 0 |

The main bottleneck was candidate target regression, not the bootstrap LCB
guard. RCRU thresholds were not changed.

### Operational and cost result

- Infrastructure-failed updates: 0
- Critic semantic-rejection exhaustion events: 2
- Output-contract contamination rejections: 1
- Freeze events: 2
- All-actionable-members-frozen stop: no
- S5 total tokens: 928,493
- S5 Solver tokens: 888,617
- S5 role-model tokens: 39,876
- S5 Solver calls: 855
- S5 Teacher/Critic/Student calls: 15/14/7
- Tokens per Stage-A candidate: 84,408.45
- Tokens per accepted update: 928,493

The two freeze events affected different agents late in the pilot. There was
no mass rapid freeze and no common infrastructure or parser failure.

## Decision

```text
Pilot execution: COMPLETE
Pilot method execution: BASIC PASS
RCRU operational viability: BASIC PASS
Official pilot gate: PASS_AFTER_OFFLINE_REVALIDATION
Formal disambiguation experiment eligibility: GO
Formal experiment execution: NOT_STARTED
```

This pilot contains no final-test observations and provides no efficacy or
generalization conclusion.
