# V18 Teacher-Critic Pipeline Simplification Four-Arm Pilot

This experiment is isolated from the canonical method and makes no trajectory
commit. It compares six frozen V18 fixed-parent cases across four arms:

- A: canonical Teacher and canonical semantic hard-veto Critic.
- B: Teacher-Clean and canonical semantic hard-veto Critic.
- C: Teacher-Clean, conservative deterministic hard gate, no semantic Critic.
- D: Teacher-Clean, conservative deterministic hard gate, non-blocking advisory Critic.

B/C/D share exact Teacher responses when their requests are identical. All arms
use the same parent, target, peers, responsibility evidence, train pool, Student
settings, source-candidate budget, loss-blind revision policy, rollout,
Common-Safe rule, and ranking. Candidate decisions are frozen before winner-only
Val50 evaluation. Test125 is never accessed.

The hard gate rejects only malformed/missing required fields, explicit gold or
sample-ID leakage, explicit fixed-answer hard-coding, detectable direct copying
of a peer procedure, or an explicit instruction to modify the immutable output
contract. Mere mention of a final answer or preserving output behavior is not a
violation.

The arm-selection rule is implemented and tested in
`scripts/v18_teacher_critic_pipeline_support.py`. Validation Vote is primary;
Oracle is mechanism-only and cannot select an arm.
