# Level-B GEPA canary v2 transport-failure audit

## Outcome

**EXTERNAL_TRANSPORT_FAILURE / SCIENTIFIC NON-RESULT**

The authorized fresh attempt stopped during fixed-probe initialization with
`APIConnectionError`. Reflection, local GEPA search, TeamMiniBatch, Full,
Shadow, Validation, and Test were not reached. This attempt provides no new
evidence about the already-established Level-B local empirical path or about
local-to-team transfer.

The requested zero-API audit did **not** pass all three engineering checks:

1. **Frozen retry exhaustion: FAIL.** `COMMON_SOLVER_CONTRACT_V1` specifies a
   four-attempt transport cap, but the run-local Solver adapter's retry
   predicate does not classify the OpenAI SDK `APIConnectionError` as
   retryable. A zero-network fake-exception replay reached the transport once,
   recorded one in-memory failed attempt, and raised immediately. Therefore the
   contract-level retry loop was not exhausted.
2. **Request-contract parity: PARTIAL / NOT ARTIFACT-VERIFIABLE.** The source
   uses one deterministic serializer and one transport closure for every
   initialization request, fixing model, message schema, temperature,
   max-tokens, thinking mode, timeout, and provider route. However, the failed
   request identity and sanitized request metadata were not durably persisted,
   so the fatal request cannot be compared directly with the 90 successful
   records. Because initialization is concurrent, it should not be described
   as a strictly ordered “91st request.”
3. **Lifecycle and ledger completeness: FAIL.** Lifecycle correctly records
   `RUNNING -> ABORTED`, provider-boundary reached, and failure category
   `APIConnectionError`. The durable ledger contains 90 unique successful
   initialization Solver records with no duplicates and correct token
   arithmetic, but contains no failed-attempt record. Exact failed-attempt
   accounting is therefore not recoverable from disk.

## Persisted accounting

| Field | Value |
|---|---:|
| Durable ledger rows | 90 |
| Unique record identities | 90 |
| Unique request identities | 90 |
| Successful provider calls | 90 |
| Failed provider-attempt rows | 0 |
| Cache hits | 0 |
| Input tokens | 17,710 |
| Output tokens | 5,792 |
| Total tokens | 23,502 |
| Validation calls | 0 |
| Test calls | 0 |

## Decision

No scientific-method repair is indicated. A governance-only refreeze is **not
yet sufficient** for another real-provider attempt. Before any new freeze, the
transport adapter must minimally recognize the SDK connection exception under
the already-frozen retry policy, and every failed contract-level attempt must
be durably recorded without storing request or response text. This is an
engineering/accounting closure, not an algorithm or scientific-protocol change.

The aborted attempt remains read-only and must not be resumed or used as
scientific evidence.

## Verification

- Focused common-contract and canary tests: **16 passed**.
- Canonical `tests/` suite: **1050 passed, 1 failed**. The sole failure is the
  pre-existing V16 M20 historical-cache coverage test; its immutable SQLite
  evidence does not cover the exact fixed probe expected by that historical
  audit. It is unrelated to this read-only report.
- `compileall`: PASS.
- Report JSON parsing, SHA-256 replay, sanitization scan, and
  `git diff --check`: PASS.
- Running pytest from the repository root is not a valid canonical invocation
  because preserved historical run clones contain duplicate test-module names;
  no historical directory was altered to suppress those collection errors.
