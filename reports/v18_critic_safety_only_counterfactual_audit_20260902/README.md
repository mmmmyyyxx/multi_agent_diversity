# V18 Safety-Only Critic Counterfactual Audit

This zero-API audit enumerates all 96 frozen V18 proposal branches and all 175
Critic responses. Of those responses, 27 were approvals and 148 were semantic
rejections. No method, historical artifact, validation state, or test state was
modified.

## Rejection mapping

| Category | Decisions |
|---|---:|
| Anti-cheating | 0 |
| Schema/format | 0 |
| Output-contract | 5 |
| Semantic-quality-only | 85 |
| Ambiguous output vs preservation | 58 |

## Student reach counterfactual

| Estimate | Branches | Rate |
|---|---:|---:|
| Historical | 27/96 | 28.125% |
| Safety-only lower bound | 71/96 | 73.958% |
| Safety-only upper bound | 95/96 | 98.958% |

The lower bound continues to block every ambiguous
`preservation_or_output_risk` decision. The upper bound treats those ambiguous
decisions as non-safety. Both estimates retain anti-cheating, schema/format, and
direct output-contract blockers.

This counterfactual estimates only whether Student would be reached. It does
not show that a bypassed plan would yield a strict-valid, feasible, safe, or
useful candidate. A prospective shadow-bypass experiment would still be needed
to estimate candidate quality conditional on a Critic rejection.

```text
API_CALLS=0
VALIDATION_CALLS=0
TEST_CALLS=0
METHOD_MODIFIED=false
HISTORICAL_ARTIFACTS_MODIFIED=false
```
