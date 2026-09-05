# Vote-Aligned Static/P0/P1 Confirmatory Replication v1

This is a new prospective confirmatory replication. It is not an automatic
extension of the completed Seed75 pilot, and Seed75 outcomes are not pooled
into its confirmatory decision.

## Frozen scope

The experiment repeats the exact Seed75 intervention definitions on two new
seeds:

| Seed | Optimize100 | Shadow50 | Arms |
|---:|---|---|---|
| 76 | fold A + fold C | fold B | Static, P0, P1 |
| 77 | fold B + fold C | fold A | Static, P0, P1 |

The remaining Validation50 is evaluated once per frozen final state only after
all four trainable trajectories have terminated. Test50 is prohibited.

- `Static`: exact frozen initial five-prompt team; no target selection,
  candidate generation, update, write-back, or commit.
- `P0_SHADOW_D2_GENERIC`: the Seed75 actionable-member round-robin target
  scheduler on the shared D2 Generic + Common-Safe + winner-only Shadow
  pipeline.
- `P1_SHADOW_VOTE_ALIGNED_GENERIC`: the exact Seed75 scheduler priority
  `direct_flip > near_margin > pure_coverage > fallback_rr`; all other
  components are identical to P0.

Both trainable arms have at most 32 update opportunities, two target slots,
two independent source candidates per target, the same loss-blind generic
revision opportunity, and the same early stop after six consecutive updates
without a Shadow-approved commit. P0 and P1 start from the same frozen initial
team and initial solver-cache evidence within each seed.

Models are frozen to Solver `qwen3-8b` with thinking disabled and
Teacher/Critic/Student/Evaluator `qwen3.7-flash`. The semantic Critic used by
Seed75 remains enabled. No W1 score, M2F, responsibility-conditioned proposal,
new Critic, altered scheduler weight, validation feedback, checkpoint
selection, or test access is permitted.

## Prospective hypotheses

The primary confirmatory hypothesis is evaluated on final Validation50:

```text
P1 - P0: delta(VoteAcc - MeanMemberAcc) > 0
```

The two absolute-improvement checks are:

```text
P1 - Static: delta MeanMemberAcc > 0
P1 - Static: delta VoteAcc > 0
```

The plurality-effective redundancy check is:

```text
P1 G>=3 count > P0 G>=3 count
```

`G>=1` and the full `G0..G5` histogram are reported, but increased `G>=1`
alone does not satisfy the depth hypothesis.

## Frozen interpretation

For each seed, replication is supported only when all four inequalities above
are strict. The final classifier is:

- `CONFIRMATORY_REPLICATION_SUPPORTED`: both Seed76 and Seed77 satisfy all
  four inequalities.
- `PARTIAL_CONFIRMATORY_REPLICATION`: exactly one seed satisfies all four and
  all four mean paired contrasts across the two seeds are strictly positive.
- `CONFIRMATORY_REPLICATION_NOT_SUPPORTED`: otherwise.

These thresholds cannot be changed after any Seed76/77 result is observed.
No extra seed, arm, retry, or update may be added based on efficacy. A
protocol, source, persistence, or infrastructure failure produces `HOLD` and
preserves evidence; it is not an efficacy result.

## Authorization state

This protocol is zero-API preregistration only. It does not authorize Phase B.
A fresh, explicit user authorization and a clean source freeze are required
before either Seed76 or Seed77 can call a model API.

