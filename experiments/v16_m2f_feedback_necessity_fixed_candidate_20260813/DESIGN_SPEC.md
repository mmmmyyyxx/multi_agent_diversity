# M2F Candidate-Specific Feedback Necessity Isolation

This is a fixed-candidate, train-side causal control. It does not change Module1,
the canonical v15 method, Common-Safe acceptance, or the seven historical M20
source candidates.

Each source candidate receives exactly one `qwen3-14b` revision call in each arm.
Both arms receive the same parent prompt, source candidate, successful assigned
responsibility evidence, instruction, temperature, and token ceiling. F1 receives
no candidate-specific loss examples, loss identities, or loss counts. F2 alone
receives the actual candidate-specific competence-loss evidence and associated
counts. Both resulting prompts use the same frozen rollout evaluator and pools.

The primary analysis compares targeting retention, collateral loss and recovery,
compatibility rescue, CriticalNet, OracleDelta, and VoteNet. CriticalNet,
OracleDelta, and VoteNet are mechanism-analysis metrics only: they do not affect
generation, ranking, Common-Safe acceptance, or any team update.

The frozen descriptive classifier is:

- `FEEDBACK_NECESSARY`: F2 is noninferior on targeting retention and strictly
  better in at least three of collateral loss, compatibility rescues, CriticalNet,
  VoteNet, and collateral recovery.
- `EXTRA_REVISION_SUFFICIENT`: F1 is noninferior on targeting retention and F2
  fails to improve at least three of those five directions.
- `MIXED`: all other outcomes.

The seven paired cases and their identities remain frozen regardless of outcomes.
No validation or test data, candidate commit, optimizer update, or parent mutation
is permitted.
