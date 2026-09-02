# V18 Safety-Only Critic Counterfactual Audit

This is a post-execution, zero-API audit of the immutable V18 Hybrid Online
Accumulation Pilot. It does not modify the Critic, replay a proposal branch, or
evaluate any new candidate.

The evidence universe is fixed to the six V18 trajectories (three seeds by two
arms), comprising 96 proposal branches. Every persisted Critic response is
enumerated. The expected accounting is 175 schema-valid Critic responses: 27
approvals and 148 semantic rejections.

Each rejection receives exactly one deterministic category:

1. `ANTI_CHEATING`
2. `SCHEMA_OR_FORMAT`
3. `OUTPUT_CONTRACT`
4. `SEMANTIC_QUALITY_ONLY`
5. `AMBIGUOUS_OUTPUT_VS_PRESERVATION`

The category is derived only from structured Critic fields and the associated
frozen Teacher plan. No validation or test result is used. For the mixed
`preservation_or_output_risk` check, direct output-surface signals and direct
competence-preservation signals are detected separately. Both or neither make
the decision ambiguous; output-only is an output-contract blocker and
preservation-only is semantic quality.

The safety-only lower bound keeps anti-cheating, schema/format,
output-contract, and ambiguous decisions as blockers. The upper bound differs
only by treating ambiguous decisions as non-safety. A branch reaches Student
counterfactually if it originally reached Student or if at least one observed
Critic round would be non-blocking under the relevant bound. This is a reach
estimate, not an estimate of candidate validity, feasibility, or efficacy.

```text
API_CALLS=0
VALIDATION_CALLS=0
TEST_CALLS=0
METHOD_MODIFIED=false
HISTORICAL_ARTIFACTS_MODIFIED=false
```
