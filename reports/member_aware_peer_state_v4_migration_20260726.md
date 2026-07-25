# Member-Aware Peer-State v4 Migration Report

Date: 2026-07-26

## Scope

This migration changes the formal candidate-preservation contract, Student
invalid recovery, and validation/test execution. It does not report method
efficacy and did not use a real API.

## Version Changes

| Component | v4 value |
|---|---|
| Method | `member_aware_peer_state_v4` |
| Checkpoint | `9` |
| Candidate acceptance | `aggregate_nondegrading_target_improvement_v1` |
| Preservation policy | `diagnostic_only_sample_preservation_v1` |
| Validation selection | `cached_unique_state_validation_selection_v2` |
| Student invalid recovery | `feedback_retry_then_upstream_regenerate_v1` |
| TCS protocol | `aggregated_small_model_tcs_v4` |

Checkpoint v8 and earlier are intentionally incompatible with v4.

## Formal Candidate Acceptance

v3 used per-example preservation limits and active/initial competence floors.
v4 replaces them with four aggregate conditions relative to the incumbent:

1. The target member's correct count strictly increases.
2. Team vote-correct count does not decrease.
3. `(V_count, g_min, g_sum)` strictly Pareto-dominates the incumbent.
4. The target member's terminal-invalid count does not increase.

The method no longer requires zero loss on each fixed-probe example. Vote,
unique-correct, and pivotal-correct gains and losses are still computed and
audited symmetrically. They can inform deterministic late tie-breaking, but no
individual loss category independently rejects a candidate.

The following obsolete configuration and guard paths were deleted:

- local and global accuracy-loss epsilon;
- legacy invalid-guard epsilon;
- vote-loss limit;
- unique-correct loss limit;
- pivotal-correct loss limit;
- local and global terminal-invalid allowances;
- active/initial target competence-floor decisions.

All optimized settings apply the same aggregate feasibility contract before
their existing setting-specific candidate preference.

## Student Invalid Recovery

A Student response is filtered with explicit rejection classes for invalid
JSON, schema failure, a missing candidate list, empty/non-string candidates,
length violations, parent-identical prompts, duplicates, and sample
memorization.

If a response has no valid candidate, the next request includes structured
feedback and retries the same approved repair plan. One cycle contains the
initial call plus three retries. If all four calls fail, the program performs
one fresh Teacher-Critic regeneration from the same bounded diagnosis context
and runs one final four-call Student cycle. Therefore:

- maximum Student cycles per update: 2;
- maximum Student calls per update: 8;
- maximum upstream Teacher-Critic regenerations: 1;
- a partially valid response is consumed immediately;
- invalid candidates never enter Stage A.

Teacher/Critic transport retry and normal Critic-grounded semantic revision
remain distinct from this Student recovery chain. Recovery outcomes and
terminal classes are persisted and written to
`student_recovery_observations.jsonl`.

## Validation and Test Isolation

Validation remains the sole checkpoint-selection signal. v4 hashes the complete
prompt-team state and evaluates each unique state once. Rejected updates reuse
the cached metrics for the unchanged team. Accepted states are evaluated only
when their hash is new.

After validation selection is finalized, an optimized run evaluates only the
selected team on test and does so exactly once. It does not evaluate its initial
team separately on test. In matched runs, `shared_baseline` must run first and
supplies the common initial-test reference used to calculate optimized member
gains. Resume reuses a persisted completed selected-test evaluation.

Checkpoint v9 persists the validation state cache, evaluation/reuse counters,
selected validation checkpoint, selection-complete flag, test isolation
counters, and selected-test metrics.

## Deliberately Unchanged Semantics

- exactly five frozen-model agents and equal-weight plurality voting;
- top-count tie as abstention;
- single-target replacement with four fixed peers;
- integer objective vector `(V_count, g_min, g_sum)`;
- member-aware responsibility assignment and improvement need;
- potential-aware target scheduling, cooldown, and max-wait protection;
- programmatic diagnosis aggregation over the full fixed probe;
- bounded Teacher and Critic roles and normal Critic-grounded revision;
- three-channel member-aware Stage A and setting-specific ablations;
- immutable strict Solver `FINAL_ANSWER` contract;
- request-local first-valid Solver recovery with `solver_max_tokens=1800`;
- optimization/validation/test split isolation and exact run identity;
- atomic accepted updates and one responsibility refresh per team transition.

## Offline Evidence

New deterministic coverage includes:

- aggregate acceptance with a five-gain/one-loss candidate;
- acceptance of vote-neutral target improvement;
- rejection of aggregate vote regression;
- four invalid Student calls followed by upstream regeneration and a valid
  fifth call;
- proof that invalid Student candidates do not enter Stage A;
- two unique validation states, one cached reuse, and exactly one selected-test
  evaluation.

The final exact pytest, compile, preflight, smoke, and whitespace results are
recorded in the implementation commit and completion report.

## Remaining Real-API Risks

Offline tests cannot establish the real-model rate of Student schema recovery,
the quality of the regenerated Teacher-Critic plan, candidate funnel depth, or
the frequency with which sample-level losses still satisfy aggregate
non-regression. They also cannot establish method efficacy. A later explicitly
authorized API task should first verify the bounded recovery and audit records,
then run a matched efficacy pilot. No API call or experiment was made during
this migration.
