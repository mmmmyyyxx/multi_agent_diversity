# v4 Full High-Frequency Pilot (seed 42)

This is a sanitized, aggregate-only evidence bundle for the one authorized
`shared_member_aware_full` real-API pilot at commit
`e352bcf6db3dc90a2d42279852563a2c04edd48c`. The protocol used 24 fixed training
updates (`epochs=8`, `train_size=75`, `update_every=25`) and a single final
test of the final active state.

## Protocol result

The lifecycle assertions passed: validation was unused (`0` calls and states),
all 24 planned updates completed, the selected state was the final active team
at update 24, and test was called exactly once after training. No Solver
terminal invalid output occurred; all 3,850 resolved Solver requests were valid
on their first attempt.

## Training dynamics

The train vote changed from `28/75` to
`51/75`. Mean member correctness changed
from `28.0/75` to
`48.0/75`; the minimum member
count changed from `28/75` to
`37/75`. `15` of 24 updates
were accepted. Inspect `training_dynamics.jsonl`,
`team_differentiation_trajectory.jsonl`, and
`update_transition_decomposition.jsonl` for the full aggregate trajectories.

The candidate funnel is partitioned into updates 0-7, 8-15, and 16-23. One
update ended after the Critic exhausted its permitted semantic revisions; this
is recorded as a role-pipeline terminal outcome, not an infrastructure or
Solver failure.

The three phases added train-vote gains of `11`, `6`, and `6`, respectively;
the final eight updates therefore still had positive train-side marginal gain.
The initial identical team had off-diagonal double fault `0.627`, correctness
correlation `1.000`, and effective-member proxy `1.000`; the final train team
had `0.217`, `0.405`, and `1.909`. However, same-wrong excess rose from
`0.081` to `0.328`. The evidence therefore supports large competence gains and
reduced shared-error incidence, but does not establish unqualified useful
differentiation under the stricter same-wrong-excess criterion.

## Final test

The frozen Baseline reference was `52/125`;
the final active team was `81/125` (gain
`29`). Per-agent
correct-count gains were `[27, 25, 12, 28, 26]`. This is a
development dynamics comparison only, not a formal matched efficacy or
generalization claim.

## Contents

- `final_summary.json`: sanitized final-state and test lifecycle summary.
- `training_dynamics.jsonl`, `team_differentiation_trajectory.jsonl`: aggregate
  train-state trajectories.
- `update_transition_decomposition.jsonl`: G/H/M and vote-transition summaries
  for accepted updates.
- `candidate_funnel.json`, `candidate_decisions_sanitized.jsonl`: candidate
  funnel and aggregate decision evidence.
- `target_scheduler_summary.json`, `student_recovery_summary.json`,
  `solver_invalid_provenance.json`: scheduling and reliability accounting.
- `final_test_differentiation.json`: frozen-reference comparison and limits.
- `dynamics_analysis.json`: three-phase aggregate dynamics comparison.
- `cost_summary.json`, `run_meta_sanitized.json`, `sha256_manifest.json`:
  cost, protocol, and integrity metadata.

No prompts, role text, questions, gold labels, per-question answers, raw API
responses, endpoints, credentials, SQLite caches, checkpoints, or local paths
are included.
