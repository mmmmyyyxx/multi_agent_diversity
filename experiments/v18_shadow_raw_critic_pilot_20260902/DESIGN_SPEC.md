# V18 Canonical Critic Shadow-Raw Fixed-Parent Pilot

This diagnostic estimates useful-candidate probability conditional on a
canonical LLM Critic rejection. It does not replace or modify the canonical
Critic.

Six parent/target cases are frozen before API execution: for each of Seeds 59,
60, and 61, the earliest historically Critic-blocked Hybrid branch and the
earliest historically Critic-passed Hybrid branch.

For each case, the canonical control runs first. The Shadow-Raw arm replays the
same first Teacher response and same canonical Critic response. If that Critic
response is a valid semantic rejection, Shadow-Raw records the original
decision and analytically changes only its effective continuation decision so
the unchanged Student receives the rejected plan. An approved or malformed
Critic response is not altered.

Both arms retain the same parent, target, peers, train pool, Teacher and Student
settings, candidate budget, Common-Safe rule, ranking, and loss-blind generic
revision policy. No prompt is committed and no trajectory state is mutated.
Train-side candidates and hypothetical winners are frozen before winner-only
validation. Test access is prohibited.

Primary estimand:

```text
P(Common-Safe feasible or WOULD_COMMIT candidate | canonical Critic reject)
```

Frozen interpretation:

- `CRITIC_OVER_FILTERING_CAUSALLY_SUPPORTED`: at least two bypassed branches
  produce a Common-Safe candidate, at least one produces a hypothetical commit,
  and aggregate winner validation Vote delta is at least -1.
- `CRITIC_FILTERING_JUSTIFIED`: no bypassed branch produces a Common-Safe
  candidate.
- `MIXED_SHADOW_RAW_SIGNAL`: neither condition above and at least three valid
  rejected-plan witnesses execute.
- `NO_CLEAR_SIGNAL`: fewer than three valid rejected-plan witnesses execute.

No result may change the frozen cases, classifier, budget, or validation rule.
