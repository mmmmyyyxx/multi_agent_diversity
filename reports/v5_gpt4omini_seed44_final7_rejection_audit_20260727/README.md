# v5 Seed-44 Terminal Rejection-Streak Audit

Offline audit of the final seven rejected Full updates (25 through 31). This is diagnostic evidence from one seed, not a method-effect claim.

## Findings

- All 7 updates reused team-state version 10 and all 13 candidates were unacceptable.
- Candidate assigned-residual repairs total `1`; coverage gains total `9`.
- Rejection reasons: `{'member_objective_regression': 12, 'target_not_improved': 13, 'team_vote_regression': 6}`.
- Target frequencies: `{'0': 2, '1': 0, '2': 2, '3': 1, '4': 2}`; Agent 4 share is `0.286`, so the streak is not exclusively Agent 4.
- The second candidate objective strictly dominates the first at updates `[25, 29]` but remains unacceptable under the hard guards.
- Repeated structural-pattern bundles: `[{'pattern_ids': ['c3dbe8489e619472bb48', '15c32ceffa352dcbaea2', 'af61b8fee6f0622d7525'], 'update_indices': [25, 29]}, {'pattern_ids': ['dad78bc7d6ba79b3cb06', '15c32ceffa352dcbaea2', '5292a3c0399f4bfb57bc'], 'update_indices': [26, 27, 28, 30, 31]}]`.

`last7_rejection_audit.json` contains the target-owned direct-fix, coverage, oracle and responsibility-age statistics, candidate repair counts, guard outcomes, and hash-only structural-pattern identifiers. Prompts, questions, answers, raw role outputs, cache references, checkpoints, and local paths are excluded.
