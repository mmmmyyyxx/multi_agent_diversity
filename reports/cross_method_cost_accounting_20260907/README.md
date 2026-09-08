# Cross-method optimization cost accounting

Status: **PASS_WITH_DISCLOSED_LIMITATIONS**. This is a zero-API, read-only
accounting audit. `API calls = 0`, `Test calls = 0`, and historical artifacts
modified = 0.

## OPTIMIZATION ONLY

| Method | Seeds | Native search units | Commits | Solver calls | Optimizer calls | Input tok | Output tok | Total tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MARS | 75-77 | 37 iterations | — | 2,000 | 805 | unavailable | unavailable | 2,951,027 |
| GEPA | 75-77 | 66 proposals | — | 1,996 | 66 | 3,014,901 | 739,502 | 3,754,403 |
| Diversity P1 | 76-77 exact | 33 opportunities | 13 | 15,209 | 429 | 6,525,897 | 4,141,051 | 10,666,948 |
| Diversity P1 | 75 partial | 20 opportunities | 6 | unavailable | unavailable | unavailable | unavailable | unavailable |

MARS iteration, GEPA proposal, and Diversity update opportunity are not treated
as interchangeable updates. Provider calls, provider tokens, and task-model
evaluations are the shared resource axes.

## Diversity P1

| Seed | Opportunities | Actual commits | Solver calls | Optimizer calls | Total tokens | Tokens/commit |
|---:|---:|---:|---:|---:|---:|---:|
| 75 | 20 | 6 | unavailable | unavailable | unavailable | unavailable |
| 76 | 17 | 8 | 9,205 | 219 | 6,484,315 | 810,539.375 |
| 77 | 16 | 5 | 6,004 | 210 | 4,182,633 | 836,526.600 |
| 76-77 | 33 | 13 | 15,209 | 429 | 10,666,948 | 820,534.462 |

The previously quoted `7/9/6 = 22 commits` is an off-by-one state-count error:
each count includes the initial state. Raw write-back ledgers establish
`6/8/5 = 19` actual commits. Consequently, an exact 22-commit average is not a
valid quantity. The exact recoverable average is 820,534.462 tokens per commit
for Seeds76-77 (13 commits). The three-seed total cannot be recovered exactly
because Seed75's authoritative raw ledger was deleted; no Seed76/77 mean is
imputed to Seed75.

Seed75 retains exact evidence for 20 opportunities, 6 commits, 6,757 provider
attempts (6,756 successful, 1 failed), but not role partitions or token usage.
Its recovery status is `PARTIALLY_RECOVERED`.

## Cost concentration

- Diversity Seeds76-77: solver 93.55%, optimizer 6.45% of tokens.
- MARS Seeds75-77: solver 69.96%, optimizer 30.04%.
- GEPA Seeds75-77: solver 88.15%, optimizer 11.85%.

Diversity is most solver-dominated, reflecting five-member/candidate rollout
evaluation. MARS spends the largest relative share on optimizer/meta reasoning.

The old 4.13M-token-per-trajectory approximation understates Seed76 by
2,354,315 tokens (36.31%) and Seed77 by 52,633 (1.26%). Against the exact
Seed76-77 mean of 5,333,474, it is low by 1,203,474 (22.57%). A three-seed
comparison is unavailable because Seed75 tokens are unrecoverable.

## Comparability

Optimization-only provider calls and tokens are exact sums of the retained
usage records where populated. All three repositories retained
provider-reported input/output usage, while their local clients derived
`total_tokens = input_tokens + output_tokens`; none retained an independent
provider total-token field. Input/output partitions are not strictly comparable
for MARS because its historical solver ledger persisted only total tokens.
Provider-attempt de-duplication is fully auditable for GEPA, cache-key auditable
for MARS, and not recoverable for Diversity because its raw JSONL lacks request
IDs. These limitations are recorded in `provenance.json`.
Tokens per 1 percentage-point validation gain are `DESCRIPTIVE_ONLY`: the three
historical optimization/evaluation contracts were not identical. Native search
unit efficiency is only meaningful within each method.

COMMON_SOLVER_CONTRACT_V1 Seed76/77 and all MARS/GEPA ExternalValidation replays
are excluded from training cost and reported separately in
`evaluation_replay_cost.csv`.
