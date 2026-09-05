# Seed75 Static No-Training Control

This is a post-hoc supplementary diagnostic added after observing the completed
Seed75 P0/P1 paired pilot. It does not change that pilot's preregistered
`NO_CLEAR_SIGNAL` classification.

The control restores the exact frozen initial five-member ensemble used by P0
and P1. It performs zero prompt updates, target selections, candidate
generations, revisions, Common-Safe decisions, Shadow write-back decisions, or
commits. Existing Optimize100 observations are reused from the frozen
initialization cache. Only missing Shadow50 and Validation50 outcomes may be
evaluated, once each, through an evaluation-only path. Test50 is inaccessible.

The supplementary classifiers are frozen before evaluation:

- `P1_ABSOLUTE_MEMBER_GAIN` when P1 minus Static Validation MeanMember is
  greater than `+0.01`;
- `P1_MEMBER_ROUGHLY_PRESERVED` when its absolute value is at most `0.01`;
- `P1_ABSOLUTE_MEMBER_DEGRADATION` when it is less than `-0.01`.

The descriptive ensemble-structure classifier is `POSITIVE`, `NEUTRAL`, or
`NEGATIVE` according to the sign of P1 minus Static Validation
`VoteAcc - MeanMemberAcc`. No significance claim is made from one seed.

