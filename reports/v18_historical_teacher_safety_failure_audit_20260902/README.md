# V18 Historical Teacher Safety Failure Audit

This zero-API audit applies the frozen deterministic hard-safety patterns to
historical structured Teacher plans. It publishes only aggregate counts,
rates, hashes, and structural labels; no plan, prompt, question, answer,
response, provider configuration, cache, or runtime-state content is included.

- Structured Teacher plans: **175**
- Hard-safety unsafe plans: **166 (94.9%)**
- Dominant contaminated field: **preservation_rule**
- Dominant-field coverage among unsafe plans: **97.6%**
- Unsafe first plans with retry: **75**
- Retry safety-repair rate: **0.0%**
- Same-marker retry recurrence: **100.0%**
- Critic-blocked branches ending unsafe: **94.2%**
- Critic-passed branches ending unsafe: **96.3%**
- Blocked minus passed unsafe-rate difference: **-2.1%**

Frozen diagnostic decision: **STABLE_PATTERN_NOT_DISCRIMINATIVE_FOR_CANONICAL_BLOCKING**.

The pattern is stable enough to explain why the prospective deterministic
safety gate blocked every sampled branch. It is not discriminative for the
historical canonical Critic outcome: passed branches were at least as likely
to contain a hard-safety trigger as blocked branches. Consequently, Teacher
safety contamination and canonical semantic blocking are distinct historical
mechanisms rather than one sufficient explanation for all pre-Student loss.

This audit evaluates association and recurrence, not a causal Teacher-prompt
intervention. The prospective plans are not reclassified below their persisted
coarse categories because their text was intentionally not retained.

```text
API_CALLS=0
TEST_ACCESSED=false
HISTORICAL_ARTIFACTS_MODIFIED=false
```
