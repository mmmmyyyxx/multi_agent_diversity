# V2.3 real initial prefix and owner-review timeout

**Execution aborted at `CANARY_OWNER_REVIEW_TIMEOUT` before optimization.**
Initial profiling completed, but the required owner review was not written
within 3600 seconds. This is an operational failure and does not establish
V2.3 optimizer success or failure. Canary did not pass; Pilot was not entered.

| Observation | Result |
|---|---:|
| Logical initial profiles | 300 |
| Physical qwen3-8b Solver calls | 357 |
| Initial correct counts / 60 | 52, 49, 52, 51, 50 |
| Valid final boundaries | 286 |
| Exhausted invalid profiles | 14 |
| Opportunities / proposals / commits | 0 / 0 / 0 |
| Input / output tokens | 72,016 / 332,661 |
| Charged / pending reservations | 404,677 / 0 |
| Closed unused allowance | 1,595,323 |
| Shadow / Validation / Test calls | 0 / 0 / 0 |

[Independent integrity audit](terminal_prefix_audit.json) verifies 667 private
artifacts, 357 sealed responses, exact dispatch for all 300 memberships, cache
seals, accounting replay and consumed-approval rejection. The owner remains
responsible for the missing review. The delegated monitor later reported an
account usage limit; its causal timing is not established.

[Failure analysis](failure_analysis.json) separates operational G from unobserved
scientific A–F. Empty [opportunity](pilot_opportunities.csv) and
[candidate](candidate_evaluation.csv) tables have zero observed rows, rather
than failed candidate outcomes. [Memory](memory_trace.json) contains five
initial competence records and no edit effects.

The [cleanup report](../a4_v23_only_cleanup_canary_pilot_20261008/README.md), old
frozen manifests, approvals, ledgers and private evidence remain intact. See
[freeze](freeze.json), [consumed authorization](authorization_receipt.json),
[Canary evidence](canary_execution.json), [accounting](api_ledger_summary.json),
[regression](regression.json) and [scientific limits](scientific_conclusion.md).

Recovery changes the execution handoff and regression coverage only. The
scientific method, review guards and deadlines are preserved. A new independent
attempt is prepared separately with direct owner supervision and requires its
own exact single-use approval. No additional paid call has been made.

Current regression: **390 passed, 2 skipped**, 213 historical/private modules
explicitly omitted. Compileall and current governance pass. Existing historical
CLI findings are disclosed separately and are not converted to PASS.
