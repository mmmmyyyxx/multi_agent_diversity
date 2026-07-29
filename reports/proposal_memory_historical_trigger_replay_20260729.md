# Proposal Memory Historical Trigger Replay

This is a read-only replay over sanitized v5 artifacts. It verifies only the
availability of complete-key inputs; it does not infer a counterfactual
trajectory or efficacy under v6 acceptance.

| Source | Sanitized decision rows | Complete-key replay |
|---|---:|---|
| seed44 | 32 | unavailable |
| seed45 | 32 | unavailable |

The checked-in sanitized candidate-decision files currently contain no rows, so
they do not expose the joint state version, target prompt hash, and owned
residual set required to construct a `ProposalMemoryKey`. The requested seed44
update ranges and seed45 cross-target non-hit examples are therefore marked
**unavailable** rather than guessed. This does not use raw runs, prompts,
questions, answers, caches, or API calls.

`complete_key_replay_performed = false`
