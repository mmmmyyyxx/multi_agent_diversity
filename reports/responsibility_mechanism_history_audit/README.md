# Responsibility mechanism history audit

This is a read-only replay over sanitized v11 observed states. It does not estimate counterfactual training, final test, or efficacy.

| Label | Owner/target semantics | Replay availability |
|---|---|---|
| H0 | repair evidence with distinct all-agent target selection | unavailable |
| H1 | owner vector included improvement_need | unavailable |
| A | five-axis owner and global-error target pool | available: 32 archived target decisions |
| B | four repair axes only | available: 14 observed states |
| P | relative-gain potential precedes team repair | source/audit only |
| R | repair-only owner plus assigned-portfolio target | available: 14 observed states |

Policy A remains the archived behavior. B and R are deterministic replays on 14 fixed responsibility states; their target rows are not a counterfactual 32-update trajectory because historical decision-to-state joins are unavailable. H0/H1 unavailable fields are explicitly not zero.
