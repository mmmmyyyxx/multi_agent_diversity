# V2.3 bounded execution, capacity and initial evidence reuse

This explicit opt-in execution revision preserves the V2.3 optimization graph,
model instructions, scoring, sampling, evidence roles, Pattern, Memory and Full
transition. Original freezes retain their original policy constants and bytes.
New attempts bind `BOUNDED_SOLVER_CAPACITY_EXECUTION_V1`,
`MATH_SOLVER_CAPACITY_INVALID_RECOVERY_V2`,
`VERIFIED_INITIAL_REALIZATION_PREFIX_REUSE_V1` and
`MATH_PARALLEL_TOKEN_ACCOUNTING_V1` in the manifest, startup and API scope.

Independent Solver batches use eight workers and an eight-slot physical request
semaphore. Initialization, local validation, Probe, Full and winner-only Shadow
restore input order before evaluation. Generations, opportunities, selection,
commit and Memory remain sequential. A shared per-key Future prevents duplicate
in-flight Solver realizations. Reservations, counters and fsynced evidence writes
are serialized; network calls run outside the accounting lock. A failed batch
stops dispatch and drains issued requests before closing or recovering its ledger.
Charged plus reserved tokens cannot exceed the fresh two-million-token ceiling.
Unknown transport charges retain the original full-reservation treatment.

The first semantic draw uses 3600 output tokens. A subsequent draw uses 6144 if
and only if the immediately preceding response was length-truncated, otherwise
3600. Messages, model and sampling controls remain byte-identical. Four total
successful semantic draws remain the limit; transport retries are separate.
Truncation remains invalid even with a final marker. No footer repair, higher
capacity escalation, additional semantic retry or accuracy-dependent adaptation
is permitted. The recovery identity explicitly distinguishes changed capacity
from the original identical-request policy. Provider acceptance and actual
request/usage metadata require real Canary review; offline tests cannot prove
that the provider supports 6144.

One complete historical attempt is selected before observing new results. Every
initial member/example request is reconstructed and matched to its exact wire
bytes, member lane, immutable V6 interface and original sealed cache, receipts,
startup and closed ledger. A private frozen manifest indexes all 300 rows.
Old caches, contexts, response hashes and authorization scopes remain untouched.
Only initial Optimize observations may import historical realizations; subsequent
optimization requests use the new attempt's own cache and provider evidence.

Import the compatible prefix uniformly for every initial observation, stopping
before the first draw whose required capacity differs from the historical actual
request. Retain semantic ordinals and count imported draws toward the four-draw
limit. Old normal 3600 results are explicitly 3600 realizations. An unchanged
four-draw terminal invalid remains invalid; it does not receive four extra tries.
Changed capacity requires new measurement for the remaining legal draws.
No correctness-based source selection or mixing of historical attempts is allowed.
New logical hashes reference original physical hashes, receipts, source and
ledger through an immutable import journal. Imported usage is zero in the new
ledger and is separately disclosed as historical paid usage. New competence,
correct/incorrect/invalid sets and preservation floors are recomputed from the
complete reconstructed initial state, never forced to historical scores.

Canary first resolves the first eight affected initial observations in fixed
member/example order and requires an owner review before completing initialization.
These results populate the same initial cache keys; this adds no evaluations.
The complete 300-profile review precedes Gradient; the first complete opportunity
review precedes the next opportunity. All three receipt-bound owner reviews have
the original 3600-second deadline. Deterministic audit is automatic; actual
trajectory/evidence judgment requires the active scientific owner. Never create
an unconditional approval process. Naturally unreached branches remain unobserved.

This is a reconstructed-baseline single-arm V2.3 experiment, not a paired model
comparison or a wholly fresh repeated baseline. Model stationarity across old
and new calls is unproven; report that limitation and the exact imported/new
counts. No held-out generalization or optimizer causality follows from Optimize.
Fresh source, policy, reuse manifest, accounting, cache, attempt and one exact
single-use authorization are required before paid execution.
