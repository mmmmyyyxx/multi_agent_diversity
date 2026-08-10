# v15 Module 3 Common-Safe Scalar Reward Replay

- Task status: **COMPLETE**
- API calls: **0**
- Replay validation: **PASS**
- Best scalar: **NONE**
- Recommended Module3: **S2_ONLY**

This directory compares S2, frozen M3-B, and three fixed equal-weight scalar
responsibility signals on the same 88-candidate Seed46 S3 Stage-B pool. Actual
parents, target pairs, responsibility state, candidates, and rollouts remain
fixed. Hypothetical winners are not propagated.

SRA, SRB, and SRC reproduce M3-B exactly. Their one change versus S2 has zero
responsibility gain, so responsibility-attributable activation is 0/32. The
evidence does not justify retaining a third module merely as formal machinery.

No API, candidate generation, rollout, validation, test call, W1 selector, or
formal source modification occurred.
