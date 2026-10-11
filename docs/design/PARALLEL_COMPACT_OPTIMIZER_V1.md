# Parallel compact optimizer amendment

The current method is `unified_team_prompt_search_v2_6_parallel_compact_v1`.
This version changes optimizer input representation and Gradient formatting
guidance, so model behavior and efficacy require fresh prospective evidence.
The diagnostic limits remain 400/400/400/240; parser and leakage guards remain
strict. No outcome, selection, transition or stopping threshold changes.

Default execution uses `STRUCTURED_PARALLEL_COMPACT_EXECUTION_V4`: sixteen
Solver slots and eight optimizer slots, with eight independent per-example
Gradient workers. Per-example recovery remains serial and bounded to three
draws. Results and diagnostics are merged in wrong-universe input order after
all issued workers drain. Pattern clustering, Reflection, opportunity evolution
and team admission remain serial. Failures stop dispatch and drain issued work
before accounting or client closure. Serial reference execution physics remain
available for deterministic conformance comparisons; formal execution requires
the new default identity. Execution policies are exact closed objects and
enter provider request identity, scope, binding and manifest.

`LOSSLESS_OPTIMIZER_TRAJECTORY_ENCODING_V1` applies only after existing
provenance validation at the optimizer input boundary. A repeated nonempty
`visible_solution` can be replaced by an exact Unicode character interval
into the same trajectory's complete `observed_response`, with a SHA256 of
the view. Encode only when an exact substring exists and its reference is
smaller than the repeated text. Otherwise retain the original view. A compact
view must decode to the original packet byte-equivalent JSON value. Full
response, references, prompts, mathematical evidence and provenance remain
available; original private profiles and trajectories remain untouched.
The wire instruction explains this reference. Gradient and Reflection both
use it; Solver and aggregator inputs never receive encoding or optimizer
context. Explicit JSON-only guidance changes no validation/recovery rule.

`MONOTONIC_PROVIDER_TIMING_V1` records slot waiting/occupancy, physical
transport durations, UTC start/end markers and measured retry backoff. Each
physical attempt has a stable request hash and attempt number. Linked slot
identifiers and monotonic/UTC markers permit occupancy reconstruction. Timings
enter private durable response receipts before accounting reconciliation, and the
observation journal; they never enter scoring, cache keys or provider payloads.
Slot occupancy includes broker accounting, persistence and retries, whereas
transport duration ends immediately on provider return/error. Cache hits do
not fabricate physical attempt timings. These metrics describe local client
waiting, not supplier-internal queue time.

Fresh accounting identities permit sixteen reservations with unchanged token
ceilings, full missing-usage charges, durable journaling and recovery. Older
journals, consumed scopes, caches and frozen experiment evidence cannot be
reopened by the new runtime. Preparation is DRAFT, has no real API grant, and
does not authorize Validation/Test. Paid execution requires a fresh complete
source/protocol/provider/capacity/accounting freeze and a single-use handoff.
