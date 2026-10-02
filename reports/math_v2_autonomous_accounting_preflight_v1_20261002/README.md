# MATH A1 Seed81 autonomous budget preflight

Status: **STOP_TOKEN_ACCOUNTING_UNRESOLVED**, before the first transport.
This is a zero-provider audit. Canary and Pilot were not started. No new
execution source, startup identity, single-use attempt authorization or
SEARCH_COMPLETE_RECEIPT was created.

## Authorization and identity

The user's task authorizes A1 Seed81 Canary, necessary operational repairs,
a fresh Pilot and a single post-search Validation300 comparison, with a
cumulative ceiling of 30,000,000 real provider input plus output tokens.
Task-wide authorization was received; no attempt authorization was consumed.
The task explicitly forbids pushing.

- Baseline: `4435d276280b9b990f7ca2564d29b507ee02e55e`.
- Historical frozen source: `470091faecf6b0de6e8b892778d12a66892ebe49`.
- Prospective Canary: `math_unified_v2_A1_seed81_canary_attempt2`.
- Solver: `qwen3-8b`, thinking false; Reflection: `qwen3.7-flash`; provider: `lwj`.
- Pattern and Memory: OFF.
- The parent manifest passed its existing Canary preflight on the baseline
  before this audit added governance metadata. That historical readiness did
  not include this task's cumulative token accounting requirement.
- The new audit manifest is DRAFT and non-executable. The governance changes
  enter source closure; old preps cannot authorize the changed working source.

## Why execution stopped

Task sections 4 and 5 require a reliable conservative next-call upper bound
before every transport and forbid counting missing usage as zero. The frozen
binding has a request output limit of 1,800 tokens and call-count ceilings,
but no tokenizer identity, provider framing bound, alias-to-upstream accounting
binding, or failed-request usage contract. A usage field available after a
successful response cannot establish the required bound before sending it.

The public [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) and
[Alibaba Qwen3.7 Flash documentation](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-7-flash)
describe upstream implementations. They do not establish the tokenizer,
framing, hidden billable generation or failure accounting of the frozen `lwj`
deployment. Applying those contracts to this provider would be an inference
without a frozen provider binding.

Validation300 membership metadata also lacks an input-length bound. The
recommended 6M reserve is a planning floor, not a demonstrated worst-case cost
of the complete paired evaluation. Its conservative required reserve remains
unresolved. No Validation rows were parsed to obtain lengths. No shortened
outputs, scientific stopping, metric budget or split sizes were introduced.

This is a preexecution accounting evidence blocker, categorized as
`F_EXTERNAL_NONRECOVERABLE_WITH_CURRENT_ACCOUNTING_EVIDENCE`. No runtime
repair attempt or model probe was used to guess a billing contract. Closing it
requires reliable provider-bound accounting evidence covering both model
aliases, request framing, all billable output and uncertain transport outcomes,
plus a conservative Validation reserve that preserves held-out isolation.
There is no basis for declaring the scientific method invalid or ineffective.

## Durable accounting and preservation

The ignored control directory contains an exclusively created, fsynced and
read-back-verified `AUTONOMOUS_REAL_TOKEN_BUDGET_V1.json`. It has not been
overwritten. The report contains a sanitized copy. Consumption by attempt is
empty; all stage, role and model counters are zero. No transport reservation
was admitted. The ledger is closed at this STOP; it is not an implemented or
verified live reservation guard for future requests.

All existing consumption marker hashes were preserved. The previous task's
historical attempt1 spent 560 tokens before this task's baseline; those calls
are identified separately and excluded from the new task's consumption.
Original manifests, preps, failed-run artifacts and reports remain unchanged.

## Results

| Stage | Attempts started | Real calls | Input tokens | Output tokens |
|---|---:|---:|---:|---:|
| Canary | 0 | 0 | 0 | 0 |
| Pilot search | 0 | 0 | 0 | 0 |
| Validation initial | 0 | 0 | 0 | 0 |
| Validation final | 0 | 0 | 0 | 0 |
| Total | 0 | 0 | 0 | 0 |

All trajectory, admission, commit, responsibility and throughput statistics
are **not available because there was no execution**, rather than measured
zeros. Initial/final VoteAcc, member accuracy, OracleAcc, paired changes and
bootstrap intervals are likewise unavailable. Scientific validity is
`NOT_EVALUABLE_NO_EXECUTION`; efficacy is `UNEVALUATED`.

Verification results are in [verification.json](verification.json).
An existing governance assertion was corrected to allow closed Canary
readiness after an operational STOP, while retaining all Formal and held-out
authorization checks. Production and scientific sources are unchanged.
The full current invocation recorded 620 passed, 2 skipped and this one stale
assertion failure. After correction, the entire affected governance module
passed all 43 tests. No clean full-suite rerun is claimed; no failures remain
unresolved. Compileall, deterministic preflight, governance, sanitization and
diff checks passed under the zero-API verification boundary.
The explicitly classified current suite is separate from excluded historical
replay/private-artifact tests; full historical replay passing is not claimed.

```text
TOTAL_REAL_TOKEN_AUTHORIZATION = 30000000
TOTAL_REAL_TOKENS_CONSUMED = 0
TOTAL_REAL_TOKENS_REMAINING = 30000000
CANARY_ATTEMPTS = 0
CANARY_FINAL_STATUS = NOT_STARTED_TOKEN_ACCOUNTING_UNRESOLVED
PILOT_ATTEMPTS = 0
PILOT_SEARCH_COMPLETE = NO
PILOT_VALIDATION_COMPLETE = NO
PILOT_FINAL_STATUS = INCOMPLETE
PILOT_VOTEACC_INITIAL = NOT_AVAILABLE
PILOT_VOTEACC_FINAL = NOT_AVAILABLE
PILOT_VOTEACC_DELTA = NOT_AVAILABLE
PILOT_ORACLE_INITIAL = NOT_AVAILABLE
PILOT_ORACLE_FINAL = NOT_AVAILABLE
PILOT_ORACLE_DELTA = NOT_AVAILABLE
LOCAL_REJECTED_EXPORTED = NOT_AVAILABLE
LOCAL_REJECTED_COMMITTED = NOT_AVAILABLE
VALIDATION_USED_DURING_SEARCH = NO
TEST_RAW_READS = 0
TEST_MODEL_CALLS = 0
SCIENTIFIC_EFFICACY_VERIFIED = NO
SOTA_VERIFIED = NO
A2_A3_A4_AUTHORIZED = NO
FORMAL_3SEED_AUTHORIZED = NO
```

No push was performed. This report does not grant readiness or a consumable
execution authorization.
