# MATH token accounting repair V2

The user explicitly replaces the previous provider-billing-proof requirement
with `RESERVATION_V2`. The previous accounting STOP evidence stays immutable.
Baseline: `039dd93b07e2d686727ab5ee0bd580557729ca41`, descending from
`4435d276280b9b990f7ca2564d29b507ee02e55e`.

## Operational policy

- Total accounting authorization: 30,000,000 input plus output tokens.
- Every physical attempt reserves the exact UTF-8 serialized request bytes
  plus 4,096 input margin plus the transmitted 1,800-token output hard cap.
- This is an **OPERATIONAL_ACCOUNTING_BOUND**, not provider-billing proof.
- Reliable usage releases the reservation and charges actual input/output.
  Missing, invalid or untrusted usage and failures without reliable usage
  charge the full reservation permanently. Retries each reserve and charge.
- The fsynced hash-chained journal is authoritative; its JSON snapshot is a
  derived view. OS locking permits one live owner. Recovery fully charges
  outstanding reservations; charged totals never decrease.
- Truncated outputs abort without being accepted as scientific output.
  Budget/provider/parser failures escape ordinary GEPA exception swallowing.
- Temperature, Solver thinking, models, scientific prompts, parsing,
  aggregation, responsibility, GEPA physics, splits and scientific stopping
  are unchanged. The inherited 1,800-token output cap is retained.

## Held-out accounting preparation

An isolated `ACCOUNTING_DATA_PREP_CONTEXT` read 300 frozen Validation problems
to emit IDs, hashes and blank-prompt serialized-request byte lengths only.
It made zero model calls and zero correctness evaluations. No problem, gold,
solution, score or output is exposed to search. Test raw reads remain zero.

Initial-team reserve: **9,934,900**. The initial final-team envelope reserve is
**9,961,000**. Total protection: **19,895,900** accounting tokens. Every search
transport protects that reserve. Observed prompt lengths can only enlarge
the final-team envelope; no candidate is selected using held-out content.

## Execution scope

Only MATH A1 Seed81 is authorized: a fresh Canary followed, after independent
PASS audit and evidence freeze, by a fresh Pilot. Search uses Optimize150 and
private winner-only Shadow300. Validation's actual model evaluation follows
an immutable SEARCH_COMPLETE_RECEIPT under separate phase admission. Test,
other arms/seeds/benchmarks and Formal execution remain unauthorized.

The original initial five-member team and strict `math_verify_no_fallback_v1`
parser remain frozen. No regeneration is introduced. Accounting safety is
separate from scientific convergence and efficacy. Source/preexecution
identities and complete zero-API verification are recorded in the separate
freeze receipt produced after source commit; this document grants no runnable
authorization by itself. No push is authorized.
