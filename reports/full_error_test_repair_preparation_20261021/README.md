# A2 Test scoring repair preparation

The three original A2 training repeats completed all sixty rounds. Validation
correct counts are82,84,83 of100; retained updates are3,2,3. Original artifacts
are immutable. Test stopped in its first repeat with a local mathematical worker
timeout, not a provider failure; no complete Test accuracy is available.

An offline replay of the actual saved expression pair reproduced the native
eight-second timeout. The existing exact power-certificate worker completed in
0.777 seconds with an exact negative certificate. The result is a new local
diagnostic observation, not an accepted historical Test score. No model API was
called. Private evidence remains untracked and only hashes/counters are published.

The new [protocol](../../docs/design/SINGLE_PROMPT_FULL_ERROR_TEST_RESUME_V3.md)
and [manifest](../../experiments/manifests/single_prompt_full_error_test_resume_v3.json)
prepare a separate Solver-only Test continuation. It preserves all350 paid draws
and288 recorded terminal observations, rehashes the complete parent attempt and
rejects prior-score drift. The successful semantic/inflight budget is3250 new plus
350 inherited,3600 combined. Network failures are excluded from that semantic
ceiling but fully accounted; Tokens remain uncapped. Production scoring and the
original training method are unchanged. Local thirty-second scoring retries the
same answer, retaining a private trace and stopping for diagnosis after three
same-context timeouts. Test/API execution remains unauthorized until exact freeze.

Verification is in progress; final check evidence will be recorded before freeze.
This document makes no new Test efficacy or independent generalization claim.
