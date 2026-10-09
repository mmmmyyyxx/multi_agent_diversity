# Independent MATH Baseline Stage 1 Execution V1

This contract authorizes no V2.5 search, new Canary, Stage 2, Shadow,
Validation or Test. The sole Unified optimization entrypoint remains
`scripts/run_experiment.py`. The separate baseline entrypoint is
`scripts/run_math_baseline_calibration.py`; it never constructs a search graph.

The scientific parent is the immutable `MATH_BASELINE_PAIRED_CALIBRATION_PLAN_V1`
at `experiments/protocols/math_baseline_calibration_v1/protocol.json`.
Stage 1 retains its 60 hashed Optimize memberships, exact A/B/C prompts,
raw User problem, qwen3-8b, thinking=false, native Parsers, common diagnostic,
seed 81, sequential rotated ABC/BCA/CAB order and invalid-only four draws.
There are 180 logical evaluations, at most 720 successful draws and at most
15,120 physical attempts. Every completed logical evaluation persists all
draws, a sealed checkpoint and immutable response receipts.

Execution identity: `MATH_BASELINE_STAGE1_EXECUTION_V1`.
Accounting identity: `FRESH_MATH_BASELINE_STAGE1_ACCOUNTING_V1`.
The independent charged plus reserved ceiling is 1,450,000 tokens, concurrency
one, with HTTP UTF-8 serialized request bytes plus 4096 input reservation and
the requested output capacity. Missing/untrusted usage, failure or an unresolved
call consumes its full reservation. Provider billing is not inferred from this
operational bound. Journal hashes, fsync and an OS owner lock govern accounting;
JSON snapshots are derived, and failure detaches the observer without retrying
a model request. Output exceeding its frozen bound fails closed.

Capacity is 6144 only after the immediately preceding successful semantic draw
finished with truncation; otherwise 3600. All sampling fields and messages are
unchanged. Up to 21 physical attempts use the same exact HTTP body per draw,
SDK retries zero, timeout 120 seconds. Only APIConnectionError, APITimeoutError,
RateLimitError and InternalServerError are transport-retry categories, with
1.5-second exponential backoff capped at 60 seconds, matching the original
transport policy. Other errors, persistence failure, hash mismatch, contradictory
thinking evidence or model alias mismatch terminate the attempt. Backend weight
revision remains unknown when not exposed by the Provider.

The source implementation and public preregistration are committed first.
The private frozen manifest then binds the exact clean main HEAD, static source
closure, protocol and preregistration bytes, data/reference/membership hashes,
Provider configuration fingerprint, exact command, source, limits and the user's
Stage 1 authorization. This avoids self-referential commit hashes. An immutable
private handoff records the frozen manifest hash and exact command. Both
real_api_authorized and READY_TO_RUN require this completed source freeze.
The old design manifest's false flags remain unchanged; it is a design artifact.

The context-specific user instruction "授权进行api调用" accepts the proposed
Stage 1 scope. It is recorded as a private authorization receipt and digest,
not permission for Stage 2 or another attempt. Startup rechecks HEAD, clean
tracked source, static file hashes, source bytes, protocol, membership,
authorization and Provider configuration. An exclusive consumed-scope marker
precedes the first HTTP call. Fresh run and accounting directories are mandatory;
old ledger, cache, output, credentials authorization or balance is never imported.
No scientific execution resume is supported. A failure preserves evidence;
operational recovery may inspect or conservatively charge unresolved calls
offline, but cannot dispatch new calls under the consumed scope.

Native output validity alone decides whether to generate another semantic draw.
Gold correctness and common diagnostics cannot change sampling, order, stopping,
selection or prompts. Valid wrong answers stop. Math scoring, paired improved /
regressed counts, subject/level summaries and descriptive paired-example bootstrap
are audited after the fixed panel terminates. An incomplete panel yields factual
partial counters and no arm efficacy comparison. No artificial accuracy gate or
automatic rerun follows a valid unfavorable result.

All raw questions, references, responses, request/response bodies, receipts and
per-example outcomes remain in ignored private artifacts. Public evidence contains
hashes, counters, categories and metrics. The existing audit and Canary remain
immutable. Stage 2 needs separate review and a fresh exact authorization.
