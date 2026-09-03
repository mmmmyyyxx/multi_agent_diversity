# Canonical vs No-Semantic-Critic Online Trajectory Experiment

This prospective online experiment freezes Seed68 only and compares
only the canonical Teacher/semantic-Critic pipeline (A) against the previously
selected Teacher-Clean/deterministic-hard-gate/no-semantic-Critic pipeline (C).

Each arm receives eight online update opportunities with identical
initialization, target selection, responsibility, candidate/revision budgets,
Student, rollout, Common-Safe acceptance, ranking, max-one commit, and plurality
aggregation. Validation is unavailable during optimization. After a trajectory
is frozen, its initial and accepted prompt-team states may be evaluated on the
validation split solely for post-hoc transition provenance. Test is prohibited.

The primary endpoint is paired final validation Vote correct count. Across the
50 paired validation observations, an absolute difference of at most one
correct answer is the preregistered approximately-neutral band. Classifier
precedence is: throughput-and-Vote supported, throughput-only, throughput with
transfer regression, then no clear online advantage. The prospective scope was
reduced from three seeds to one by explicit user instruction before the first
trajectory completed; the interrupted earlier root is excluded from evidence.
