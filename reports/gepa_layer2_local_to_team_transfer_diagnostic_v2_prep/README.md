# Online GEPA local-to-team transfer diagnostic v2 — zero-API handoff

`READY_FOR_AUTHORIZATION`, **not authorized and not executed**. This is an
execution-integrity amendment, not a new scientific treatment or an efficacy
result. The unexecuted v1 freeze is immutable and classified
`SUPERSEDED_BEFORE_EXECUTION_BY_SAMPLING_INTEGRITY_FIX`; its preparation bundle
was not edited. Earlier, preliminary v2 zero-API prep bundles are not the final
freeze and cannot pass the final source-identity check.

| Identity | Final frozen value |
| --- | --- |
| Baseline/report HEAD before closure | `4d30116cc07ba203bc1e4ab99b8ffdbe0caedbea` |
| Execution source | `727c50a9c7d24ec826b83dd0bce276255d67cfbc` |
| Attempt | `gepa_layer2_local_to_team_transfer_diagnostic_v2` |
| Protocol SHA256 | `69076bb01e15d715feccd7d0458e456dbb44ecd0d67fa06582508fe314356737` |
| Preregistration SHA256 | `a9785115e65b5dc9dcc60b13e4378704e69cf1ae357d31faaf5c4201fd24ce43` |
| Run identity SHA256 | `6886062d6141bccc6f363a05507c72d1fdb7864e8ebb180b43a7eb47025dfa3a` |
| Provider / Solver / Reflection | `lwj` / `qwen3-8b` thinking false / `qwen3.7-flash` |
| GEPA | v0.1.1, frozen commit `b4dbb55b7601dac448cdb836d5a401ca7d9eb920` |
| API / Validation50 / Test50 calls | `0 / 0 / 0` |

The online diagnostic controller now raises a typed sampling-integrity error
immediately after local optimization for accepted/frontier mismatch, >1 accept,
or absent/non-positive strict local acceptance evidence. The poison fixtures
show zero TeamMiniBatch, ordinary Full, diagnostic Full, Shadow, commit and
next-opportunity calls after mismatch. Final accepted-row reconciliation is
retained independently.

Each completed accepted-candidate row now carries durable-ledger stage costs.
MiniBatch, Full and Shadow are candidate-attributable by update and candidate
ID. Local Solver and Reflection costs are *opportunity-shared*, not added to
candidate totals as if independent. The reflection ledger exposes physical
attempt records, not an unambiguous logical-call ID; logical reflection calls
remain `null` when such rows exist. Cache-hit token echoes are distinguished
from provider-success token sums. Thus candidate-level stage telemetry is
`PARTIAL` in the strict sense of per-candidate attribution, while the
per-opportunity/global durable-ledger partition is exact and audited.

Mandatory Full remains admission-inert when MiniBatch fails. On promotion,
the diagnostic reuses ordinary Full exactly once. Scheduler feedback receives
only the ordinary committed-member ID. Duplicate accepted event groups are
marked without deduplicating or changing the sample. An emergency ceiling
during Full or Shadow produces an aborted incomplete attempt, never a complete
`TARGET_NOT_REACHED` artifact.

The unconstrained successful-call envelope remains 2,880 versus an operational
hard ceiling of 1,200 successes / 4,800 attempts. Three full 300-row Shadow
evaluations plus 500 initialization rows could alone exceed 1,200 even before
local search and candidate evaluations. Five complete accepted mutations are
therefore not guaranteed; the ceiling is not a scientific budget target.

Three independent zero-API freezes produced identical preregistration and
run-identity hashes. Fresh-process preflights returned
`PREREGISTERED_NOT_EXECUTED`; an unauthorized execution probe stopped with
`ABORT_PRE_PROVIDER: authorization required` without creating a run root.
See `verification.json` and `protocol_implementation_matrix.json`. No real
API execution, automatic retry, Validation50 or Test50 access was performed.

If the user later grants this exact attempt, execution must use a clean
checkout of the **execution source** above and the final private preparation
bundle. A later report commit is not an execution source. No push is implied
by this task.
