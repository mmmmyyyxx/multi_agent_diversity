# Level-B GEPA real-provider canary v2 transportfix1

Gate: **PASS**
Classifier: **LOCAL_EMPIRICAL_PATH_CONFIRMED**

This fresh single-parent engineering canary completed from execution commit
`8b8c9909e268c5960ff104c977001e5e1fae1225`. It tests empirical-path activation,
not scheduler or accuracy efficacy. Validation50 and Test50 calls are zero.

## Funnel

| Stage | Count |
|---|---:|
| Proposal attempts | 3 |
| Changed proposals | 3 |
| Contract-invalid proposals | 0 |
| Proposals reaching Solver | 3 |
| Accepted local mutations | 0 |
| TeamMiniBatch survivors | 0 |
| Full evaluations | 0 |
| Shadow evaluations | 0 |
| Commits | 0 |

All three changed proposals were contract-valid and reached empirical Solver
evaluation. None strictly improved the frozen local objective, so the official
GEPA engine terminated with `no_local_improvement`. Consequently there was no
accepted local candidate to send into TeamMiniBatch or later stages.

This is an informative algorithm observation, not an engineering failure. It
confirms that the Level-B empirical path works under the real provider, while
providing no new local-to-team transfer evidence in this sample.

## Accounting

| Role | Successful calls | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|---:|
| Solver | 109 | 26,033 | 10,064 | 36,097 |
| Reflection | 3 | 2,645 | 1,654 | 4,299 |
| Total | 112 | 28,678 | 11,718 | 40,396 |

There were zero failed transport attempts, zero cache hits, and zero duplicate
ledger identities. The lifecycle completed normally and recorded 112 observed
provider attempts.
