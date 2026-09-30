# Formal V3 Attempt4: concurrent local Solver pre-execution freeze

`SEMANTIC_EQUIVALENCE = PASS`
`READY_FOR_FORMAL_V3_ATTEMPT4_REAL_AUTHORIZATION = YES`
`REAL_ATTEMPT4_AUTHORIZED = NO`

This is a zero-API execution-only change. Formal V3 scientific settings,
Layer2 V4, the official pinned GEPA search core, Optimize100, Shadow50,
post-freeze Validation50 policy, and Test50 isolation are unchanged. The old
execution source is `9e498496abff9228bd10697d8d97d36f56fca3f9`; the new
frozen execution source is `6b16c8abe8717895c26a6a05bbdbc2e9e636388e`.
Attempt3 Seed80's raw results and published report remain intact. The six
Attempt4 preparations are new, independent, execution-gated identities; none
contains a consumed authorization.

## Execution architecture and scope

Seed80 was slow partly because GEPA local evaluation submitted each row of a
batch synchronously and waited for it. Although the old Formal runner set
`eval_solver_call_concurrency=8`, the local GEPA path's effective concurrency
was **1**. Initialization, TeamMiniBatch, Full Optimize, and Shadow already
used `FixedProbeEvaluator`'s gather path.

The new path validates a full GEPA batch against frozen Optimize rows, sets
`local_optimizer_solver_eval` once, submits unique indices through the existing
FixedProbe gather path, records per-response physical token usage, preserves
duplicate logical cache hits, and returns observations in the original GEPA
batch order. The Attempt4 manifest sets the Solver semaphore to **16**;
the fake benchmark observed **16** concurrent in-flight calls at that setting.
Reflection iterations, GEPA proposals and acceptance, Layer2 opportunities,
candidate admission, winner-only Shadow, and commits remain sequential.

No Solver request bytes or request identity were changed. The
`PromptQuestionEvaluator` run-local cache and inflight suppression remain in
place; `CommonSolverEvaluator` retains its exact-request cache. The
`evaluation_replica_seed` remains in the prompt-question cache key, and the
Formal runner still sets `shared_solver_cache_path=""`. There is no
cross-seed response reuse. Official GEPA `cache_evaluation=False` remains
unchanged. The only cache-path adjustment wakes duplicate waiters when an
inflight owner fails or is cancelled; it does not alter successful evaluation
or search decisions.

## Zero-API evidence

The 32-row, fixed 50 ms fake-Solver benchmark completed the same 32 logical
evaluations and 32 physical calls at every setting:

| Path | Configured concurrency | Max observed in flight | Seconds | Speedup vs old serial |
| --- | ---: | ---: | ---: | ---: |
| Old serial adapter | 1 effective | 1 | 1.9970 | 1.00× |
| New batch path | 1 | 1 | 1.9899 | 1.00× |
| New batch path | 4 | 4 | 0.4976 | 4.01× |
| New batch path | 8 | 8 | 0.2467 | 8.10× |
| New batch path | 16 | 16 | 0.1236 | 16.16× |
| New batch path | 32 | 32 | 0.0658 | 30.35× |

Attempt4 freezes **16**, based on this fake calibration. The benchmark is an
execution check; it does not predict real provider throughput or change the
scientific budget. Thirty-two was measured but not selected.

Deterministic fake-provider runs compared the frozen old source at concurrency
8 with the new source at concurrency 16 in four full-path scenarios: Native,
Layer2 no commit, Layer2 Shadow rejection, and Layer2 commit. Complete
scientific summaries, semantic ledger multisets, GEPA lineage, GEPA candidates,
run logs, saved GEPA states, fake physical call counts, stops, and final team
states matched. Only `event.telemetry.wall_seconds` was excluded from the
summary comparison. The comparison includes GEPA training/validation delivery,
reflection and candidate sequence, and the Layer2 decision and team-hash
trace; no scientific field was removed. See
[`semantic_equivalence_audit.json`](semantic_equivalence_audit.json).

The guarded relevant test run passed **112 tests**. The two independently
generated six-cell preparations had **42 byte-identical artifacts**. Offline
admission rejected **127/127** per-file source poison cases before provider
construction. All six cells have unchanged scientific payload, spec identity,
GEPA dependency, and split access policy compared with Attempt3.

All work used a fresh process with credentials removed and a network guard
active before application import. **Real API calls: 0. Validation50 calls: 0.
Test50 calls: 0. Authorization consumptions: 0.** No Seed80 rerun or Seed81/82
real search was started. Formal Attempt4 requires a new explicit real-API
authorization before any cell may execute.

## Report inventory

- `freeze_identity.json`: six new attempt identities, source and protocol hashes.
- `source_closure.json`: 127 frozen source paths and poison-audit result.
- `semantic_equivalence_audit.json`: old/new scientific equality checks.
- `concurrency_audit.json`: benchmark and fake ledger integrity checks.
- `cache_reuse_audit.json`: run-local and exact-request cache evidence.
- `test_summary.json`: guarded tests, compilation, and access counts.
- `sha256_manifest.json`: hashes of this published report set.

The private synthetic provider transcripts and split files remain under
ignored `runs/` storage. This report contains hashes, identities, and counts
only.

## Future optimizations

Real-provider concurrency and rate-limit behavior remain unmeasured. No
additional execution optimization is included here.
