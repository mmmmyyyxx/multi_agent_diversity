# v4 High-Frequency Offline Audit

This sanitized report replays decisions only at historically observed team
states. It is not a counterfactual training trajectory and does not establish
that either scheduler would improve final vote accuracy.

## Supported findings

- The historical `max_wait=4` target sequence was nearly uniform: frequencies
  were `4, 5, 5, 5, 5` across the five agents.
- Replaying the relative-gain definition at the same observed states changes
  one decision with wait 4 and 17 decisions with wait 8. With wait 8, 23 of 24
  decisions enter the relative-potential pool and none is overdue on the
  historical wait counters. This supports treating wait 8 as a next-pilot
  configuration to test, not as a proven improvement.
- Across 15 accepted updates and 1,125 reconstructed train examples, the most
  common coverage transitions were `G=0 -> G=1` (45) and `G=1 -> G=2` (40),
  while `G=1 -> G=0` occurred 6 times and `G=2 -> G=1` 8 times. These are
  symmetric audit counts, not new hard rejection rules.
- Solver used 1,538,173 of 1,649,118 recorded tokens (93.27%). Role totals are
  exact; phase, candidate, and incremental Stage-B costs are unavailable in the
  historical call log.
- In the local Stage-B top-rank replay, the Stage-A first-ranked candidate
  matched the historical accepted candidate in 13 of 15 accepted updates. The
  remaining cases demonstrate why reducing the Stage-B budget remains a future
  decision, not a change made here.

## Explicitly unavailable from the historical artifact

- Per-example unique/pivotal retention and new-coverage lifetime.
- Exact role-by-phase token attribution, prompt-length trajectory, and
  per-candidate incremental Solver costs.
- Full responsibility/pattern Jaccard continuity across selected updates.

Future runs now instrument per-example member opportunities and accepted
G/H/M transitions so these values can be emitted without reconstructing them
from raw histories.

## Files

- `observed_state_scheduler_replay.jsonl` and
  `scheduler_replay_summary.json`: scheduler replay.
- `g_transition_audit_sanitized.jsonl` and
  `protection_transition_summary.json`: reconstructed accepted-update geometry.
- `specialization_stability_sanitized.jsonl`: available pattern and correct-set
  continuity fields.
- `stage_b_preference_replay.jsonl` and `stage_b_budget_replay.json`: local
  candidate-ranking/budget simulations.
- `token_cost_breakdown_sanitized.json` and `artifact_availability.json`:
  exact cost totals and unavailable-field inventory.

No questions, gold labels, literal answers, prompts, raw role text, API
responses, credentials, caches, checkpoints, or absolute paths are included.
